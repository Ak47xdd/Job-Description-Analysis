from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from JobAnalyze.v2.token_override import apply_token_matching_override

try:
    from model.prep.sym_map import canonicalize_skill_text
except ImportError:
    canonicalize_skill_text = lambda text: str(text or "").lower()


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = max(64, int(os.getenv("SBERT_MAX_SEQ_LENGTH", "128")))
BATCH_SIZE = max(1, int(os.getenv("SBERT_BATCH_SIZE", "1")))
SBERT_BACKEND = os.getenv("SBERT_BACKEND", "onnx").strip().lower()
REQUIRE_ONNX = os.getenv("SBERT_REQUIRE_ONNX", "true").strip().lower() not in {"0", "false", "no"}
SBERT_ONNX_FILE = os.getenv("SBERT_ONNX_FILE", "onnx/model.onnx").strip()
SBERT_ONNX_PROVIDER = os.getenv("SBERT_ONNX_PROVIDER", "CPUExecutionProvider").strip()
SBERT_ONNX_DISABLE_CPU_ARENA = os.getenv("SBERT_ONNX_DISABLE_CPU_ARENA", "true").strip().lower() not in {"0", "false", "no"}
SBERT_THRESHOLD_FILE = os.getenv("SBERT_THRESHOLD_FILE", "").strip()
ARTIFACT_MANIFEST_REQUIRED = os.getenv("SBERT_ARTIFACT_MANIFEST_REQUIRED", "true").strip().lower() not in {"0", "false", "no"}
CORE_DOMAIN_THRESHOLD = float(os.getenv("SBERT_CORE_DOMAIN_THRESHOLD", "0.80"))

# High-confidence semantic aliases. These are deliberately narrow: they do not
# rename labels or alter the SBERT embedding. They provide deterministic
# evidence for a legacy skill when a modern phrase is strong evidence for it.
SEMANTIC_ALIAS_MAP = {
    "prompt engineering": (
        "agentic pair programming",
        "multi-agent compositions",
    ),
}
SEMANTIC_ALIAS_ENABLED = os.getenv("SBERT_SEMANTIC_ALIASES", "true").strip().lower() not in {"0", "false", "no"}
SEMANTIC_ALIAS_MIN_SCORE = float(os.getenv("SBERT_SEMANTIC_ALIAS_MIN_SCORE", "0.0"))
SEMANTIC_ALIAS_MARGIN = float(os.getenv("SBERT_SEMANTIC_ALIAS_MARGIN", "0.01"))

CORE_DOMAIN_LABELS = {
    "machine learning": ("machine learning", "machine-learning", "ml"),
    "deep learning": ("deep learning", "deep-learning"),
}

_tokenizer = None
_onnx_session = None
_classifier_weights = None
_label_vocab = None
_label_thresholds = None
_embedding_dim = None
_memory_stages: dict[str, dict] = {}


def _rss_mb() -> float | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024.0, 2)
    except (FileNotFoundError, OSError, ValueError):
        return None
    return None


def _record_memory(stage: str) -> None:
    rss = _rss_mb()
    if rss is not None:
        _memory_stages[stage] = {"rss_mb": rss}


def memory_diagnostics() -> dict:
    _record_memory("diagnostic")
    return {
        "rssMb": _rss_mb(),
        "stages": dict(_memory_stages),
        "tokenizerLoaded": _tokenizer is not None,
        "onnxSessionLoaded": _onnx_session is not None,
        "classifierLoaded": _classifier_weights is not None,
        "labelVocabLoaded": _label_vocab is not None,
        "thresholdsLoaded": _label_thresholds is not None,
        "onnxFile": SBERT_ONNX_FILE,
        "provider": SBERT_ONNX_PROVIDER,
        "maxSeqLength": MAX_SEQ_LENGTH,
        "batchSize": BATCH_SIZE,
        "thresholdFile": SBERT_THRESHOLD_FILE or str(ROOT / "model_out" / "v2_sentence_transformer" / "per_skill_thresholds.json"),
        "semanticAliasesEnabled": SEMANTIC_ALIAS_ENABLED,
        "semanticAliasMinScore": SEMANTIC_ALIAS_MIN_SCORE,
        "semanticAliasMargin": SEMANTIC_ALIAS_MARGIN,
        "artifactManifestRequired": ARTIFACT_MANIFEST_REQUIRED,
    }


def _normalise_model_repo(model_name: str) -> str:
    if "/" not in model_name and not Path(model_name).exists():
        return f"sentence-transformers/{model_name}"
    return model_name


def _resolve_hf_file(model_name: str, filename: str) -> str:
    model_path = Path(model_name)
    if model_path.exists():
        candidate = model_path / filename
        if not candidate.exists():
            raise FileNotFoundError(f"Missing model file {candidate}.")
        return str(candidate)
    try:
        return hf_hub_download(repo_id=_normalise_model_repo(model_name), filename=filename)
    except Exception as exc:
        raise RuntimeError(f"Could not download '{filename}' from '{model_name}'.") from exc


def _resolve_onnx_path(model_name: str) -> str:
    return _resolve_hf_file(model_name, SBERT_ONNX_FILE)


def _load_embedding_runtime(model_name: str):
    global _tokenizer, _onnx_session, _embedding_dim

    if REQUIRE_ONNX and SBERT_BACKEND != "onnx":
        raise RuntimeError("Render-safe SBERT requires SBERT_BACKEND=onnx.")

    if _tokenizer is not None and _onnx_session is not None and _embedding_dim is not None:
        return _tokenizer, _onnx_session, _embedding_dim

    _record_memory("before_tokenizer")
    tokenizer_path = _resolve_hf_file(model_name, "tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)
    _record_memory("after_tokenizer")

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.enable_mem_pattern = False
    session_options.enable_mem_reuse = True
    session_options.enable_cpu_mem_arena = not SBERT_ONNX_DISABLE_CPU_ARENA
    session_options.log_severity_level = 3

    _record_memory("before_onnx_session")
    session = ort.InferenceSession(
        _resolve_onnx_path(model_name),
        sess_options=session_options,
        providers=[SBERT_ONNX_PROVIDER],
    )
    _record_memory("after_onnx_session")

    outputs = session.get_outputs()
    if not outputs or len(outputs[0].shape) != 3:
        raise RuntimeError("Expected ONNX transformer output [batch, sequence, hidden].")
    hidden_dim = outputs[0].shape[-1]
    if not isinstance(hidden_dim, int):
        raise RuntimeError(f"Could not determine embedding dimension from {outputs[0].shape}.")

    _tokenizer, _onnx_session, _embedding_dim = tokenizer, session, hidden_dim
    return _tokenizer, _onnx_session, _embedding_dim


def _load_numpy_classifier(weights_path: Path, input_dim: int, num_labels: int, hidden_dim: int):
    npz_path = weights_path.with_suffix(".npz")
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Missing lightweight classifier artifact: {npz_path}. "
            "Run python model/model_sbert.py locally; it now regenerates .npz, thresholds, "
            "and the artifact manifest together."
        )

    with np.load(npz_path, allow_pickle=False) as data:
        w1 = np.asarray(data["w1"], dtype=np.float32)
        b1 = np.asarray(data["b1"], dtype=np.float32)
        w2 = np.asarray(data["w2"], dtype=np.float32)
        b2 = np.asarray(data["b2"], dtype=np.float32)

    expected = [
        (w1, (hidden_dim, input_dim)),
        (b1, (hidden_dim,)),
        (w2, (num_labels, hidden_dim)),
        (b2, (num_labels,)),
    ]
    for value, shape in expected:
        if value.shape != shape:
            raise ValueError(f"Invalid NumPy classifier shape {value.shape}; expected {shape}.")
    return w1, b1, w2, b2


def _resolve_threshold_file(out_dir: Path) -> Path:
    path = Path(SBERT_THRESHOLD_FILE) if SBERT_THRESHOLD_FILE else out_dir / "per_skill_thresholds.json"
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required SBERT threshold artifact: {path}. "
            "Deployment will not fall back to a global threshold."
        )
    return path


def _load_per_skill_thresholds(threshold_path: Path, label_vocab: list[str]) -> np.ndarray:
    with threshold_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    labels = payload.get("labels") if isinstance(payload, dict) else None
    if not isinstance(labels, dict):
        raise ValueError(f"Invalid SBERT threshold artifact: {threshold_path}")

    vocab_set = set(label_vocab)
    missing = [label for label in label_vocab if label not in labels]
    extra = [label for label in labels if label not in vocab_set]
    if missing or extra:
        raise ValueError(
            "SBERT threshold vocabulary mismatch: "
            f"missing={missing[:10]}, extra={extra[:10]}. Regenerate per_skill_thresholds.json."
        )

    thresholds = np.empty(len(label_vocab), dtype=np.float32)
    for index, label in enumerate(label_vocab):
        entry = labels[label]
        if not isinstance(entry, dict) or "threshold" not in entry:
            raise ValueError(f"Missing threshold for SBERT label {label!r} in {threshold_path}")
        threshold = float(entry["threshold"])
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError(f"Invalid threshold {threshold!r} for SBERT label {label!r}")
        thresholds[index] = threshold
    return thresholds


def _validate_artifact_manifest(out_dir: Path, label_path: Path, threshold_path: Path, weights_path: Path) -> dict:
    """Reject mixed/stale SBERT artifacts before serving predictions."""
    manifest_path = out_dir / "artifact_manifest.json"
    if not manifest_path.exists():
        if ARTIFACT_MANIFEST_REQUIRED:
            raise FileNotFoundError(
                f"Missing SBERT artifact manifest: {manifest_path}. "
                "Run python model/model_sbert.py locally and commit all generated v2 artifacts."
            )
        return {}

    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    required = (
        "classifier_pt_sha256",
        "classifier_npz_sha256",
        "thresholds_sha256",
        "label_vocab_sha256",
    )
    missing = [key for key in required if not manifest.get(key)]
    if missing:
        raise ValueError(f"Incomplete SBERT artifact manifest {manifest_path}: missing={missing}")

    actual = {
        "classifier_pt_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "classifier_npz_sha256": hashlib.sha256(weights_path.with_suffix(".npz").read_bytes()).hexdigest(),
        "thresholds_sha256": hashlib.sha256(threshold_path.read_bytes()).hexdigest(),
        "label_vocab_sha256": hashlib.sha256(
            json.dumps(
                json.loads(label_path.read_text(encoding="utf-8")),
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest(),
    }
    mismatches = [
        key for key in required
        if str(manifest[key]) != actual[key]
    ]
    if mismatches:
        raise RuntimeError(
            "Stale/mixed SBERT deployment artifacts detected: "
            f"{mismatches}. Retrain/export with python model/model_sbert.py and deploy "
            "the generated .pt, .npz, threshold, vocabulary, config, and artifact_manifest.json together."
        )
    return manifest


def _load_artifacts():
    global _classifier_weights, _label_vocab, _label_thresholds

    if (
        _tokenizer is not None
        and _onnx_session is not None
        and _classifier_weights is not None
        and _label_vocab is not None
        and _label_thresholds is not None
        and _embedding_dim is not None
    ):
        return _tokenizer, _onnx_session, _classifier_weights, _label_vocab, _embedding_dim

    _record_memory("before_artifacts")
    label_path = ROOT / "model" / "prep" / "v2" / "label_vocab_v2.json"
    out_dir = ROOT / "model_out" / "v2_sentence_transformer"
    weights_path = out_dir / "skill_classifier_sbert_v2.pt"
    config_path = out_dir / "model_config.json"
    threshold_path = _resolve_threshold_file(out_dir)

    if not label_path.exists():
        raise FileNotFoundError(f"Missing {label_path}.")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing {weights_path}.")

    with label_path.open(encoding="utf-8") as handle:
        label_vocab = json.load(handle)
    if not isinstance(label_vocab, list) or not label_vocab or len(label_vocab) != len(set(label_vocab)):
        raise ValueError(f"Invalid or duplicate SBERT label vocabulary: {label_path}")

    config = {}
    if config_path.exists():
        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)

    vocab_hash = hashlib.sha256(
        json.dumps(label_vocab, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    configured_vocab_hash = config.get("label_vocab_sha256")
    if configured_vocab_hash and configured_vocab_hash != vocab_hash:
        raise ValueError(
            "SBERT label vocabulary hash mismatch: model_config.json was generated "
            "for a different label ordering. Retrain/export the v2 classifier."
        )

    config_embedding_model = config.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
    config_input_dim = int(config.get("input_dim", 384))
    config_hidden_dim = int(config.get("hidden_dim", 64))

    _validate_artifact_manifest(
        out_dir,
        label_path,
        threshold_path,
        weights_path,
    )

    tokenizer, session, embedding_dim = _load_embedding_runtime(config_embedding_model)
    if embedding_dim != config_input_dim:
        config_input_dim = embedding_dim

    weights = _load_numpy_classifier(
        weights_path, embedding_dim, len(label_vocab), config_hidden_dim
    )
    thresholds = _load_per_skill_thresholds(threshold_path, label_vocab)
    _classifier_weights, _label_vocab, _label_thresholds = weights, label_vocab, thresholds
    _record_memory("after_classifier")
    return tokenizer, session, weights, label_vocab, embedding_dim


def _build_input_text(job_desc: str, role: str, job_type: str) -> str:
    normalized_desc = canonicalize_skill_text(job_desc)
    return f"Role: {role}. Job type: {job_type}. Job description: {normalized_desc}"


def _encode_embeddings(texts, tokenizer: Tokenizer, session):
    encodings = tokenizer.encode_batch(texts)
    if not encodings:
        return np.empty((0, 384), dtype=np.float32)

    max_len = min(MAX_SEQ_LENGTH, max(len(encoding.ids) for encoding in encodings))
    if max_len < 1:
        raise RuntimeError("Tokenizer returned an empty sequence.")

    batch_size = len(encodings)
    input_ids = np.zeros((batch_size, max_len), dtype=np.int64)
    attention_mask = np.zeros((batch_size, max_len), dtype=np.int64)
    type_ids = np.zeros((batch_size, max_len), dtype=np.int64)

    for index, encoding in enumerate(encodings):
        length = min(len(encoding.ids), max_len)
        input_ids[index, :length] = encoding.ids[:length]
        attention_mask[index, :length] = encoding.attention_mask[:length]
        if encoding.type_ids:
            type_ids[index, :length] = encoding.type_ids[:length]

    session_inputs = {item.name for item in session.get_inputs()}
    candidates = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": type_ids,
    }
    ort_inputs = {
        name: candidates[name]
        for name in session_inputs
        if name in candidates
    }
    missing = session_inputs.difference(ort_inputs)
    if missing:
        raise RuntimeError(f"Tokenizer did not produce required ONNX inputs: {sorted(missing)}")

    token_embeddings = session.run(None, ort_inputs)[0].astype(np.float32, copy=False)
    mask = attention_mask.astype(np.float32, copy=False)[:, :, None]
    summed = np.sum(token_embeddings * mask, axis=1, dtype=np.float32)
    counts = np.clip(mask.sum(axis=1, dtype=np.float32), 1e-9, None)
    embeddings = summed / counts
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return np.asarray(embeddings / np.clip(norms, 1e-12, None), dtype=np.float32)


def _classifier_predict(embeddings, weights):
    w1, b1, w2, b2 = weights
    hidden = np.maximum(0.0, embeddings @ w1.T + b1)
    logits = np.clip(hidden @ w2.T + b2, -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32, copy=False)


def _apply_semantic_aliases(probabilities, label_vocab, raw_job_descs, thresholds):
    """Use narrow phrase evidence without changing the global/per-skill floors.

    An alias only applies when the model already gives the target skill a
    minimum semantic score. When triggered, its score is raised just above
    that skill's own calibrated threshold, rather than to 1.0. This keeps the
    existing threshold policy intact and avoids globally boosting unrelated
    skills.
    """
    if not SEMANTIC_ALIAS_ENABLED:
        return probabilities

    if not 0.0 <= SEMANTIC_ALIAS_MIN_SCORE <= 1.0:
        raise ValueError("SBERT_SEMANTIC_ALIAS_MIN_SCORE must be between 0 and 1.")
    if SEMANTIC_ALIAS_MARGIN < 0.0:
        raise ValueError("SBERT_SEMANTIC_ALIAS_MARGIN must be non-negative.")

    label_to_index = {label: index for index, label in enumerate(label_vocab)}

    for row_index, raw_text in enumerate(raw_job_descs):
        normalized = canonicalize_skill_text(raw_text)
        for target_label, aliases in SEMANTIC_ALIAS_MAP.items():
            target_index = label_to_index.get(target_label)
            if target_index is None:
                continue
            if any(alias in normalized for alias in aliases):
                score = float(probabilities[row_index, target_index])
                threshold = float(thresholds[target_index])
                if score >= SEMANTIC_ALIAS_MIN_SCORE:
                    probabilities[row_index, target_index] = max(
                        score,
                        min(1.0, threshold + SEMANTIC_ALIAS_MARGIN),
                    )
    return probabilities


def _apply_core_domain_guard(probabilities, label_vocab, raw_job_descs):
    """Require stronger structural confidence for broad ML domain labels."""
    if not 0.0 <= CORE_DOMAIN_THRESHOLD <= 1.0:
        raise ValueError("SBERT_CORE_DOMAIN_THRESHOLD must be between 0 and 1.")

    for row_index, raw_text in enumerate(raw_job_descs):
        text = str(raw_text or "").lower()
        for label, aliases in CORE_DOMAIN_LABELS.items():
            try:
                label_index = label_vocab.index(label)
            except ValueError:
                continue
            explicit = any(alias in text for alias in aliases)
            if not explicit and float(probabilities[row_index, label_index]) < CORE_DOMAIN_THRESHOLD:
                probabilities[row_index, label_index] = 0.0
    return probabilities


def _apply_hybrid_token_override(probabilities, label_vocab, job_desc):
    """Apply deterministic lexical matching after SBERT classification."""
    if os.getenv("SBERT_TOKEN_OVERRIDE", "true").strip().lower() in {"0", "false", "no"}:
        return probabilities

    for index in range(probabilities.shape[0]):
        updated, _matches = apply_token_matching_override(
            probabilities[index],
            label_vocab,
            canonicalize_skill_text(job_desc[index] if isinstance(job_desc, list) else job_desc),
        )
        probabilities[index] = updated
    return probabilities


def JobAnalyze_v2_SBERT(job_desc="", role="", job_type="", top_k=50):
    return JobAnalyze_v2_SBERT_batch([job_desc], role, job_type, top_k)[0]


def JobAnalyze_v2_SBERT_batch(job_descs, role="", job_type="", top_k=50):
    if top_k < 1:
        return [[] for _ in job_descs]
    if not job_descs:
        return []

    tokenizer, session, weights, label_vocab, _ = _load_artifacts()
    thresholds = _label_thresholds
    texts = [_build_input_text(text or "", role, job_type) for text in job_descs]
    results = []

    for start in range(0, len(texts), BATCH_SIZE):
        embeddings = _encode_embeddings(texts[start:start + BATCH_SIZE], tokenizer, session)
        probabilities = _classifier_predict(embeddings, weights)
        batch_descs = job_descs[start:start + BATCH_SIZE]
        probabilities = _apply_hybrid_token_override(
            probabilities,
            label_vocab,
            batch_descs,
        )
        probabilities = _apply_core_domain_guard(
            probabilities,
            label_vocab,
            batch_descs,
        )
        probabilities = _apply_semantic_aliases(
            probabilities,
            label_vocab,
            batch_descs,
            thresholds,
        )

        for row in probabilities:
            eligible = [
                (skill, float(score), float(threshold))
                for skill, score, threshold in zip(label_vocab, row, thresholds)
                if float(score) >= float(threshold)
            ]
            ranked = sorted(eligible, key=lambda item: -item[1])
            results.append([(skill, score) for skill, score, _ in ranked[:top_k]])

        del embeddings, probabilities
        gc.collect()
        _record_memory("after_inference_batch")

    return results
