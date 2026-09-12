"""
pred_v2.py - JobAnalyze 6k v2 SBERT inference.

Uses the same Sentence Transformer + PyTorch MLP architecture as the
v2 SBERT training pipeline in model/model_sbert.py.

The input text format must match model/sentence_transformer_v2.py:
    Role: <role>. Job type: <job_type>. Job description: <job_desc>

Artifacts:
    model_out/v2_sentence_transformer/skill_classifier_sbert_v2.pt
    model_out/v2_sentence_transformer/model_config.json
    model/prep/v2/label_vocab_v2.json

Install:
    pip install sentence-transformers
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

# Keep this identical to the training/embedding pipeline.
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


# Load the relatively expensive Sentence Transformer once when the module is imported.
# This prevents downloading/loading it for every prediction.
_embedding_model: SentenceTransformer | None = None
_classifier: SBERTSkillClassifier | None = None
_label_vocab: list[str] | None = None


def _load_artifacts() -> tuple[SentenceTransformer, SBERTSkillClassifier, list[str]]:
    global _embedding_model, _classifier, _label_vocab

    if _embedding_model is not None and _classifier is not None and _label_vocab is not None:
        return _embedding_model, _classifier, _label_vocab

    prep_dir = ROOT / "model" / "prep"
    label_path = prep_dir / "v2" / "label_vocab_v2.json"
    out_dir = ROOT / "model_out" / "v2_sentence_transformer"
    weights_path = out_dir / "skill_classifier_sbert_v2.pt"
    config_path = out_dir / "model_config.json"

    if not label_path.exists():
        raise FileNotFoundError(
            f"Missing {label_path}. Make sure the v2 label preparation pipeline has been run."
        )
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Missing {weights_path}. Run model/model_sbert.py to train the SBERT model."
        )

    with label_path.open(encoding="utf-8") as file:
        label_vocab = json.load(file)

    config = {}
    if config_path.exists():
        with config_path.open(encoding="utf-8") as file:
            config = json.load(file)

    embedding_model_name = config.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
    input_dim = int(config.get("input_dim", 384))
    hidden_dim = int(config.get("hidden_dim", 64))
    dropout = float(config.get("dropout", 0.30))

    embedding_model = SentenceTransformer(embedding_model_name)

    # Fail early if the encoder and classifier dimensions do not agree.
    embedding_dim = int(embedding_model.get_sentence_embedding_dimension())
    if embedding_dim != input_dim:
        raise ValueError(
            f"SBERT embedding dimension mismatch: model produces {embedding_dim}, "
            f"but classifier expects {input_dim}."
        )

    classifier = SBERTSkillClassifier(
        input_dim=input_dim,
        num_labels=len(label_vocab),
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    classifier.load_state_dict(state_dict)
    classifier.eval()

    _embedding_model = embedding_model
    _classifier = classifier
    _label_vocab = label_vocab

    return embedding_model, classifier, label_vocab


def _build_input_text(job_desc: str, role: str, job_type: str) -> str:
    """Match the exact text template used during SBERT feature generation."""
    return (
        f"Role: {role}. Job type: {job_type}. Job description: {job_desc}"
    )


def JobAnalyze_v2_SBERT(
    job_desc: str = "",
    role: str = "",
    job_type: str = "",
    top_k: int = 50,
) -> List[Tuple[str, float]]:
    """Return the top-k skill predictions from the v2 SBERT model."""
    if top_k < 1:
        return []

    embedding_model, classifier, label_vocab = _load_artifacts()
    combined_text = _build_input_text(job_desc, role, job_type)

    embedding = embedding_model.encode(
        [combined_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    with torch.no_grad():
        logits = classifier(torch.from_numpy(embedding))
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    ranked = sorted(zip(label_vocab, probs), key=lambda item: -float(item[1]))
    return [(skill, float(score)) for skill, score in ranked[:top_k]]
