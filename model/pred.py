import json
import pickle
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

def predict(job_desc, role="", job_type="", top_k=50):
    prep_dir = ROOT / "model" / "prep" if (ROOT / "model" / "prep").exists() else (ROOT / "prep")

    with open(prep_dir / "label_vocab.json", encoding="utf-8") as f:
        label_vocab = json.load(f)
    with open(prep_dir / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
        
    model = SkillClassifier(80, len(label_vocab))    
    model.load_state_dict(torch.load(ROOT / 'model_out' / 'skill_classifier.pt', map_location='cpu'))
    
    model.eval()
    
    combined_text = f"{job_desc} {role} {job_type}"
    X = vectorizer.transform([combined_text]).toarray().astype(np.float32)
    
    with torch.no_grad():
        logits = model(torch.tensor(X))
        probs = torch.sigmoid(logits).numpy()[0]
        
    ranked = sorted(zip(label_vocab, probs), key=lambda x: -x[1])
    return ranked[:top_k]
    
        