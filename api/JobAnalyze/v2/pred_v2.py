"""
pred_v2.py - JobAnalyze v2 SBERT inference.

Render-safe runtime:
- Uses the original all-MiniLM-L6-v2 transformer through ONNX Runtime directly.
- Does not instantiate SentenceTransformer, avoiding the extra model/pipeline objects.
- Uses the same 384-d mean-pooling + L2-normalization pipeline as Sentence Transformers.
- Keeps the trained SBERT classifier unchanged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MAX_SEQ_LENGTH = max(64, int(os.getenv("SBERT_MAX_SEQ_LENGTH", "128")))
BATCH_SIZE = max(1, int(os.getenv("SBERT_BATCH_SIZE", "1")))
SBERT_BACKEND = os.getenv("SBERT_BACKEND", "onnx").strip().lower()
REQUIRE_ONNX = os.getenv("SBERT_REQUIRE_ONNX", "true").strip().lower() not in {"0", "false", "no"}

# FP32 is the default. This preserves the transformer weights used by the
# original model. Quantized files remain opt-in through SBERT_ONNX_FILE.
SBERT_ONNX_FILE = os.getenv("SBERT_ONNX_FILE", "onnx/model.onnx").strip()
SBERT_ONNX_PROVIDER = os.getenv("SBERT_ONNX_PROVIDER", "CPUExecutionProvider").strip()
SBERT_ONNX_DISABLE_CPU_ARENA = (
    os.getenv("SBERT_ONNX_DISABLE_CPU_ARENA", "true").strip().lower()
    not in {"0", "false", "no"}
)

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


class SBERTSkillClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        hidden_dim: int = 64,
        dropout: float = 0.30,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


_tokenizer: AutoTokenizer | None = None
_onnx_session: ort.InferenceSession | None = None
_classifier: SBERTSkillClassifier | None = None
_label_vocab: list[str] | None = None
_embedding_dim: int | None = None


def _normalise_model_repo(model_name: str) -> str:
    """Expand the short model name used by older config files."""
    if "/" not in model_name and not Path(model_name).exists():
        return f"sentence-transformers/{model_name}"
    return model_name


def _resolve_onnx_path(model_name: str) -> str:
    """Resolve an ONNX file locally or download only that file from HF Hub."""
    model_path = Path(model_name)
    if model_path.exists():
        candidate = model_path / SBERT_ONNX_FILE
        if not candidate.exists():
            raise FileNotFoundError(
                f"Missing ONNX model {candidate}. "
                f"Expected SBERT_ONNX_FILE={SBERT_ONNX_FILE}."
            )
        return str(candidate)

    repo_id = _normalise_model_repo(model_name)
    try:
        return hf_hub_download(repo_id=repo_id, filename=SBERT_ONNX_FILE)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download ONNX model '{SBERT_ONNX_FILE}' from '{repo_id}'. "
            "Verify the model repository and SBERT_ONNX_FILE."
        ) from exc


def _load_embedding_runtime(model_name: str) -> tuple[AutoTokenizer, ort.InferenceSession, int]:
    """Load tokenizer + raw ONNX transformer without SentenceTransformer."""
    global _tokenizer, _onnx_session, _embedding_dim

    if REQUIRE_ONNX and SBERT_BACKEND != "onnx":
        raise RuntimeError(
            "Render-safe SBERT requires SBERT_BACKEND=onnx. "
            "PyTorch/SentenceTransformer fallback is disabled."
        )

    if _tokenizer is not None and _onnx_session is not None and _embedding_dim is not None:
        return _tokenizer, _onnx_session, _embedding_dim

    repo_or_path = _normalise_model_repo(model_name)
    tokenizer = AutoTokenizer.from_pretrained(repo_or_path, use_fast=True)
    onnx_path = _resolve_onnx_path(model_name)

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_options.enable_mem_pattern = False
    session_options.enable_cpu_mem_arena = not SBERT_ONNX_DISABLE_CPU_ARENA

    session = ort.InferenceSession(
        onnx_path,
        sess_options=session_options,
        providers=[SBERT_ONNX_PROVIDER],
    )

    outputs = session.get_outputs()
    if not outputs:
        raise RuntimeError(f"ONNX model has no outputs: {onnx_path}")

    output_shape = outputs[0].shape
    if len(output_shape) != 3:
        raise RuntimeError(
            f"Expected transformer token output [batch, sequence, hidden], got {output_shape}."
        )

    hidden_dim = output_shape[-1]
    if not isinstance(hidden_dim, int):
        raise RuntimeError(
            f"Could not determine ONNX embedding dimension from output shape {output_shape}."
        )

    _tokenizer, _onnx_session, _embedding_dim = tokenizer, session, hidden_dim
    return tokenizer, session, hidden_dim


def _load_artifacts() -> tuple[
    AutoTokenizer,
    ort.InferenceSession,
    SBERTSkillClassifier,
    list[str],
    int,
]:
    global _classifier, _label_vocab

    if (
        _tokenizer is not None
        and _onnx_session is not None
        and _classifier is not None
        and _label_vocab is not None
        and _embedding_dim is not None
    ):
        return _tokenizer, _onnx_session, _classifier, _label_vocab, _embedding_dim

    label_path = ROOT / "model" / "prep" / "v2" / "label_vocab_v2.json"
    out_dir = ROOT / "model_out" / "v2_sentence_transformer"
    weights_path = out_dir / "skill_classifier_sbert_v2.pt"
    config_path = out_dir / "model_config.json"

    if not label_path.exists():
        raise FileNotFoundError(f"Missing {label_path}. Run model/prep/data_prep.py.")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing {weights_path}. Run model/model_sbert.py.")

    with label_path.open(encoding="utf-8") as file:
        label_vocab = json.load(file)

    if (
        not isinstance(label_vocab, list)
        or not label_vocab
        or len(label_vocab) != len(set(label_vocab))
    ):
        raise ValueError(f"Invalid or duplicate SBERT label vocabulary: {label_path}")

    config = {}
    if config_path.exists():
        with config_path.open(encoding="utf-8") as file:
            config = json.load(file)

    configured_labels = config.get("num_labels")
    if configured_labels is not None and int(configured_labels) != len(label_vocab):
        raise ValueError(
            f"SBERT artifacts disagree: config has {configured_labels} labels, "
            f"but vocabulary has {len(label_vocab)}. Regenerate and retrain."
        )

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    first_weight = state_dict.get("net.0.weight")
    output_weight = state_dict.get("net.3.weight")
    output_bias = state_dict.get("net.3.bias")

    if first_weight is None or output_weight is None or output_bias is None:
        raise ValueError(
            f"Invalid SBERT checkpoint {weights_path}: missing classifier layers."
        )

    checkpoint_input_dim = int(first_weight.shape[1])
    checkpoint_labels = int(output_weight.shape[0])

    if checkpoint_labels != len(label_vocab):
        raise ValueError(
            f"Stale SBERT checkpoint: checkpoint outputs {checkpoint_labels} labels, "
            f"but vocabulary contains {len(label_vocab)}. Retrain with the current artifacts."
        )

    embedding_model_name = config.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
    configured_input_dim = int(config.get("input_dim", 384))

    if checkpoint_input_dim != configured_input_dim:
        raise ValueError(
            "Stale SBERT checkpoint: checkpoint input dimension is "
            f"{checkpoint_input_dim}, but model_config.json says {configured_input_dim}. "
            "Run data_prep.py, sentence_transformer_v2.py, and model_sbert.py again."
        )

    tokenizer, session, embedding_dim = _load_embedding_runtime(embedding_model_name)

    if embedding_dim != configured_input_dim:
        raise ValueError(
            f"SBERT embedding dimension mismatch: ONNX encoder produces {embedding_dim}, "
            f"but model_config.json specifies {configured_input_dim}."
        )

    classifier = SBERTSkillClassifier(
        input_dim=configured_input_dim,
        num_labels=len(label_vocab),
        hidden_dim=int(config.get("hidden_dim", 64)),
        dropout=float(config.get("dropout", 0.30)),
    )
    classifier.load_state_dict(state_dict)
    classifier.eval()

    _classifier, _label_vocab = classifier, label_vocab
    return tokenizer, session, classifier, label_vocab, embedding_dim


def _build_input_text(job_desc: str, role: str, job_type: str) -> str:
    return f"Role: {role}. Job type: {job_type}. Job description: {job_desc}"


def _encode_embeddings(
    texts: list[str],
    tokenizer: AutoTokenizer,
    session: ort.InferenceSession,
) -> np.ndarray:
    """Reproduce all-MiniLM-L6-v2 mean pooling + L2 normalization."""
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="np",
    )

    session_inputs = {item.name for item in session.get_inputs()}
    ort_inputs = {
        name: np.asarray(value, dtype=np.int64)
        for name, value in encoded.items()
        if name in session_inputs
    }

    missing = session_inputs.difference(ort_inputs)
    if missing:
        raise RuntimeError(
            f"Tokenizer did not produce required ONNX inputs: {sorted(missing)}"
        )

    token_embeddings = session.run(None, ort_inputs)[0].astype(np.float32, copy=False)

    attention_mask = encoded["attention_mask"].astype(np.float32, copy=False)
    mask = attention_mask[:, :, None]
    summed = np.sum(token_embeddings * mask, axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    embeddings = summed / counts

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-12, None)

    return np.asarray(embeddings, dtype=np.float32)


def JobAnalyze_v2_SBERT(
    job_desc: str = "",
    role: str = "",
    job_type: str = "",
    top_k: int = 50,
) -> List[Tuple[str, float]]:
    return JobAnalyze_v2_SBERT_batch(
        [job_desc], role=role, job_type=job_type, top_k=top_k
    )[0]


def JobAnalyze_v2_SBERT_batch(
    job_descs: list[str],
    role: str = "",
    job_type: str = "",
    top_k: int = 50,
) -> list[List[Tuple[str, float]]]:
    """Analyze multiple descriptions with one lightweight ONNX encoder call."""
    if top_k < 1:
        return [[] for _ in job_descs]
    if not job_descs:
        return []

    tokenizer, session, classifier, label_vocab, _ = _load_artifacts()

    texts = [
        _build_input_text(text or "", role, job_type)
        for text in job_descs
    ]

    all_probs: list[np.ndarray] = []

    # Keep batches tiny on Render so intermediate token tensors cannot create
    # a large transient RSS spike.
    for start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[start : start + BATCH_SIZE]
        embeddings = _encode_embeddings(batch_texts, tokenizer, session)

        with torch.inference_mode():
            logits = classifier(torch.from_numpy(embeddings))
            probs = torch.sigmoid(logits).cpu().numpy()

        all_probs.append(np.asarray(probs, dtype=np.float32))

    probabilities = np.concatenate(all_probs, axis=0)

    results: list[List[Tuple[str, float]]] = []
    for row in probabilities:
        ranked = sorted(
            zip(label_vocab, row),
            key=lambda item: -float(item[1]),
        )
        results.append(
            [(skill, float(score)) for skill, score in ranked[:top_k]]
        )

    return results
