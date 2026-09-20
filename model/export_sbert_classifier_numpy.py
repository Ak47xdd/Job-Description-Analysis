
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"model_out"/"v2_sentence_transformer"/"skill_classifier_sbert_v2.pt"
DST=ROOT/"model_out"/"v2_sentence_transformer"/"skill_classifier_sbert_v2.npz"
if not SRC.exists(): raise FileNotFoundError(f"Missing trained checkpoint: {SRC}")
state=torch.load(SRC,map_location="cpu",weights_only=True)
required=("net.0.weight","net.0.bias","net.3.weight","net.3.bias")
missing=[key for key in required if key not in state]
if missing: raise ValueError(f"Checkpoint is missing classifier tensors: {missing}")
arrays={"w1":state["net.0.weight"].detach().cpu().numpy().astype(np.float32),"b1":state["net.0.bias"].detach().cpu().numpy().astype(np.float32),"w2":state["net.3.weight"].detach().cpu().numpy().astype(np.float32),"b2":state["net.3.bias"].detach().cpu().numpy().astype(np.float32)}
np.savez(DST,**arrays)
print(f"Wrote {DST}")
