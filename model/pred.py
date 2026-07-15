"""
pred.py - predictor for evaluation
"""

import json
import pickle
from typing import List, Tuple
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from torch import nn

class SkillClassifier(nn.Module):
    def __init__(self, input_dim, num_labels, hidden_dim=32, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x):
        return self.net(x)

def JobAnalyze_6k(job_desc: str = "", role: str = "", job_type: str = "", top_k: int = 50) -> List[Tuple[str, float]]:
    """
    Predict top-k skills.
    
    Current top-k = 48
    """
    prep_dir = ROOT / "model" / "prep"
    if not prep_dir.exists():
        alt = ROOT / "prep"
        if alt.exists():
            prep_dir = alt

    label_path = prep_dir / "label_vocab.json"
    vector_path = prep_dir / "vectorizer.pkl"
    weights_path = ROOT / "model_out" / "skill_classifier.pt"

    if not label_path.exists():
        raise FileNotFoundError(
            f"Missing {label_path}. Make sure you ran data_prep and model training."
        )
    if not vector_path.exists():
        raise FileNotFoundError(
            f"Missing {vector_path}. Make sure you ran data_prep."
        )
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Missing {weights_path}. Make sure you ran model/model.py."
        )

    with open(label_path, encoding="utf-8") as f:
        label_vocab = json.load(f)
    with open(vector_path, "rb") as f:
        vectorizer = pickle.load(f)

    input_dim = int(getattr(vectorizer, "vocabulary_", {}).__len__()) or vectorizer.transform([""]).shape[1]

    model = SkillClassifier(input_dim, len(label_vocab))
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))

    model.eval()
    
    combined_text = f"{job_desc} {role} {job_type}"
    X = vectorizer.transform([combined_text]).toarray().astype(np.float32)
    
    with torch.no_grad():
        logits = model(torch.tensor(X))
        probs = torch.sigmoid(logits).numpy()[0]
        
    ranked = sorted(zip(label_vocab, probs), key=lambda x: -x[1])
    return ranked[:top_k]
    
        