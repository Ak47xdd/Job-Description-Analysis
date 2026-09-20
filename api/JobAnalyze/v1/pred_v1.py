
import gc,json,pickle,warnings
from pathlib import Path
from typing import List,Tuple
import numpy as np
import torch
from torch import nn
warnings.filterwarnings("ignore",message="InconsistentVersionWarning: Trying to unpickle estimator*")
warnings.filterwarnings("ignore",message="Trying to unpickle estimator*")
ROOT=Path(__file__).resolve().parent.parent.parent.parent
try:
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
except RuntimeError: pass
class SkillClassifier(nn.Module):
    def __init__(self,input_dim,num_labels,hidden_dim=32,dropout=0.3):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden_dim,num_labels))
    def forward(self,x): return self.net(x)
_label_vocab=None
_vectorizer=None
_model=None
def _load_model():
    global _label_vocab,_vectorizer,_model
    if _label_vocab is not None and _vectorizer is not None and _model is not None:
        return _label_vocab,_vectorizer,_model
    prep_dir=ROOT/"model"/"prep"
    if not prep_dir.exists():
        alt=ROOT/"prep"; prep_dir=alt if alt.exists() else prep_dir
    label_path=prep_dir/"v1"/"label_vocab.json"
    vector_path=prep_dir/"v1"/"vectorizer.pkl"
    weights_path=ROOT/"model_out"/"v1"/"skill_classifier.pt"
    if not label_path.exists(): raise FileNotFoundError(f"Missing {label_path}.")
    if not vector_path.exists(): raise FileNotFoundError(f"Missing {vector_path}.")
    if not weights_path.exists(): raise FileNotFoundError(f"Missing {weights_path}.")
    with label_path.open(encoding="utf-8") as f: label_vocab=json.load(f)
    with vector_path.open("rb") as f: vectorizer=pickle.load(f)
    input_dim=len(getattr(vectorizer,"vocabulary_",{})) or vectorizer.transform([""]).shape[1]
    model=SkillClassifier(input_dim,len(label_vocab))
    state_dict=torch.load(weights_path,map_location="cpu",weights_only=True)
    model.load_state_dict(state_dict); model.eval()
    _label_vocab,_vectorizer,_model=label_vocab,vectorizer,model
    del state_dict; gc.collect()
    return _label_vocab,_vectorizer,_model
def JobAnalyze_6k(job_desc="",role="",job_type="",top_k=50):
    label_vocab,vectorizer,model=_load_model()
    X=vectorizer.transform([f"{job_desc} {role} {job_type}"]).toarray().astype(np.float32)
    with torch.inference_mode():
        logits=model(torch.from_numpy(X))
        probs=torch.sigmoid(logits).numpy()[0]
    ranked=sorted(zip(label_vocab,probs),key=lambda x:-float(x[1]))
    del X,logits,probs
    gc.collect()
    return ranked[:top_k]
