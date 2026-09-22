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


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = max(64, int(os.getenv("SBERT_MAX_SEQ_LENGTH", "128")))
BATCH_SIZE = max(1, int(os.getenv("SBERT_BATCH_SIZE", "1")))
SBERT_BACKEND = os.getenv("SBERT_BACKEND", "onnx").strip().lower()
REQUIRE_ONNX = os.getenv("SBERT_REQUIRE_ONNX", "true").strip().lower() not in {"0", "false", "no"}
SBERT_ONNX_FILE = os.getenv("SBERT_ONNX_FILE", "onnx/model.onnx").strip()
SBERT_ONNX_PROVIDER = os.getenv("SBERT_ONNX_PROVIDER", "CPUExecutionProvider").strip()
SBERT_ONNX_DISABLE_CPU_ARENA = os.getenv("SBERT_ONNX_DISABLE_CPU_ARENA", "true").strip().lower() not in {"0", "false", "no"}

_tokenizer = None
_onnx_session = None
_classifier_weights = None
_label_vocab = None
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
        "onnxFile": SBERT_ONNX_FILE,
        "provider": SBERT_ONNX_PROVIDER,
        "maxSeqLength": MAX_SEQ_LENGTH,
        "batchSize": BATCH_SIZE,
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
            "Run python model/export_sbert_classifier_numpy.py locally and commit the generated .npz."
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


def _load_artifacts():
    global _classifier_weights, _label_vocab

    if (
        _tokenizer is not None
        and _onnx_session is not None
        and _classifier_weights is not None
        and _label_vocab is not None
        and _embedding_dim is not None
    ):
        return _tokenizer, _onnx_session, _classifier_weights, _label_vocab, _embedding_dim

    _record_memory("before_artifacts")
    label_path = ROOT / "model" / "prep" / "v2" / "label_vocab_v2.json"
    out_dir = ROOT / "model_out" / "v2_sentence_transformer"
    weights_path = out_dir / "skill_classifier_sbert_v2.pt"
    config_path = out_dir / "model_config.json"

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

    # The label vocabulary and classifier artifact are the source of truth for
    # the output dimension. model_config.json is metadata and may be stale after
    # a retraining run. This prevents a stale config from breaking deployment
    # when the vocabulary legitimately grows or shrinks.
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

    tokenizer, session, embedding_dim = _load_embedding_runtime(config_embedding_model)
    if embedding_dim != config_input_dim:
        # A stale input_dim is recoverable; the classifier artifact below is
        # validated against the actual encoder dimension.
        config_input_dim = embedding_dim

    # Load against the vocabulary count. If the retrained classifier has not
    # been exported, _load_numpy_classifier will fail with the exact tensor
    # shape mismatch instead of silently pairing the wrong labels and weights.
    weights = _load_numpy_classifier(
        weights_path, embedding_dim, len(label_vocab), config_hidden_dim
    )
    _classifier_weights, _label_vocab = weights, label_vocab
    _record_memory("after_classifier")
    return tokenizer, session, weights, label_vocab, embedding_dim


def _build_input_text(job_desc: str, role: str, job_type: str) -> str:
    return f"Role: {role}. Job type: {job_type}. Job description: {job_desc}"


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


def _apply_hybrid_token_override(probabilities, label_vocab, job_desc: str):
    """Apply deterministic lexical matching after SBERT classification.

    Exact, isolated skill names mentioned in the raw JD are authoritative for
    detection. This runs before top-k selection so an explicitly mentioned
    skill cannot be hidden by a low embedding score.
    """
    if os.getenv("SBERT_TOKEN_OVERRIDE", "true").strip().lower() in {"0", "false", "no"}:
        return probabilities

    overridden = []
    for index in range(probabilities.shape[0]):
        updated, matches = apply_token_matching_override(
            probabilities[index],
            label_vocab,
            job_desc[index] if isinstance(job_desc, list) else job_desc,
        )
        probabilities[index] = updated
        overridden.append(matches)
    return probabilities


def JobAnalyze_v2_SBERT(job_desc="", role="", job_type="", top_k=50):
    return JobAnalyze_v2_SBERT_batch([job_desc], role, job_type, top_k)[0]


def JobAnalyze_v2_SBERT_batch(job_descs, role="", job_type="", top_k=50):
    if top_k < 1:
        return [[] for _ in job_descs]
    if not job_descs:
        return []

    tokenizer, session, weights, label_vocab, _ = _load_artifacts()
    texts = [_build_input_text(text or "", role, job_type) for text in job_descs]
    results = []

    for start in range(0, len(texts), BATCH_SIZE):
        embeddings = _encode_embeddings(texts[start:start + BATCH_SIZE], tokenizer, session)
        probabilities = _classifier_predict(embeddings, weights)
        probabilities = _apply_hybrid_token_override(
            probabilities,
            label_vocab,
            texts[start:start + BATCH_SIZE],
        )

        for row in probabilities:
            ranked = sorted(zip(label_vocab, row), key=lambda item: -float(item[1]))
            results.append([(skill, float(score)) for skill, score in ranked[:top_k]])

        del embeddings, probabilities
        gc.collect()
        _record_memory("after_inference_batch")

    return results
