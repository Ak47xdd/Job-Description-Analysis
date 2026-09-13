"""
pred_v2.py - JobAnalyze 6k v2 SBERT inference.

The checkpoint, model_config.json, label vocabulary, and Sentence Transformer
embedding dimension must all describe the same training run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class SBERTSkillClassifier(nn.Module):
    def __init__(self, input_dim: int, num_labels: int, hidden_dim: int = 64, dropout: float = 0.30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


_embedding_model: SentenceTransformer | None = None
_classifier: SBERTSkillClassifier | None = None
_label_vocab: list[str] | None = None


def _load_artifacts() -> tuple[SentenceTransformer, SBERTSkillClassifier, list[str]]:
    global _embedding_model, _classifier, _label_vocab

    if _embedding_model is not None and _classifier is not None and _label_vocab is not None:
        return _embedding_model, _classifier, _label_vocab

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
    if not isinstance(label_vocab, list) or not label_vocab or len(label_vocab) != len(set(label_vocab)):
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
        raise ValueError(f"Invalid SBERT checkpoint {weights_path}: missing classifier layers.")

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
            "The checkpoint was trained with a different architecture. Run:\n"
            "  python model/prep/data_prep.py\n"
            "  python model/sentence_transformer_v2.py\n"
            "  python model/model_sbert.py"
        )

    embedding_model = SentenceTransformer(embedding_model_name)
    embedding_dim = int(embedding_model.get_sentence_embedding_dimension())
    if embedding_dim != configured_input_dim:
        raise ValueError(
            f"SBERT embedding dimension mismatch: encoder produces {embedding_dim}, "
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

    _embedding_model, _classifier, _label_vocab = embedding_model, classifier, label_vocab
    return embedding_model, classifier, label_vocab


def _build_input_text(job_desc: str, role: str, job_type: str) -> str:
    return f"Role: {role}. Job type: {job_type}. Job description: {job_desc}"


def JobAnalyze_v2_SBERT(job_desc: str = "", role: str = "", job_type: str = "", top_k: int = 50) -> List[Tuple[str, float]]:
    if top_k < 1:
        return []
    embedding_model, classifier, label_vocab = _load_artifacts()
    embedding = embedding_model.encode(
        [_build_input_text(job_desc, role, job_type)],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    with torch.no_grad():
        probs = torch.sigmoid(classifier(torch.from_numpy(embedding))).cpu().numpy()[0]
    ranked = sorted(zip(label_vocab, probs), key=lambda item: -float(item[1]))
    return [(skill, float(score)) for skill, score in ranked[:top_k]]
