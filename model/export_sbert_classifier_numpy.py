"""Export the trained SBERT classifier to the lightweight NumPy runtime artifact.

Run after training, or let model/model_sbert.py perform the export automatically.
The label vocabulary is treated as the source of truth for the classifier output
dimension.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "model_out" / "v2_sentence_transformer" / "skill_classifier_sbert_v2.pt"
DST = ROOT / "model_out" / "v2_sentence_transformer" / "skill_classifier_sbert_v2.npz"
LABEL_FILE = ROOT / "model" / "prep" / "v2" / "label_vocab_v2.json"
CONFIG_FILE = ROOT / "model_out" / "v2_sentence_transformer" / "model_config.json"

if not SRC.exists():
    raise FileNotFoundError(f"Missing trained checkpoint: {SRC}")
if not LABEL_FILE.exists():
    raise FileNotFoundError(f"Missing label vocabulary: {LABEL_FILE}")

with LABEL_FILE.open(encoding="utf-8") as file:
    vocab = json.load(file)

if not isinstance(vocab, list) or not vocab or len(vocab) != len(set(vocab)):
    raise ValueError(f"Invalid or duplicate label vocabulary: {LABEL_FILE}")

state = torch.load(SRC, map_location="cpu", weights_only=True)
required = ("net.0.weight", "net.0.bias", "net.3.weight", "net.3.bias")
missing = [key for key in required if key not in state]
if missing:
    raise ValueError(f"Checkpoint is missing classifier tensors: {missing}")

w1 = state["net.0.weight"].detach().cpu().numpy().astype(np.float32)
b1 = state["net.0.bias"].detach().cpu().numpy().astype(np.float32)
w2 = state["net.3.weight"].detach().cpu().numpy().astype(np.float32)
b2 = state["net.3.bias"].detach().cpu().numpy().astype(np.float32)

if w2.shape[0] != len(vocab):
    raise ValueError(
        f"SBERT checkpoint/vocabulary mismatch: checkpoint has {w2.shape[0]} "
        f"outputs but vocabulary has {len(vocab)} labels. "
        "Regenerate v2 data and retrain the classifier from the same label vocabulary."
    )

# Derive the hidden dimension from the checkpoint instead of assuming 64.
# The current SBERTSkillClassifier uses HIDDEN_DIM=384, so its first layer is
# expected to be (384, 384). Deriving it here also keeps export compatible if
# the model architecture is intentionally changed in the future.
if w1.ndim != 2 or w1.shape[1] != 384:
    raise ValueError(
        f"Invalid first layer shape {w1.shape}; expected (hidden_dim, 384)."
    )
hidden_dim = int(w1.shape[0])
if hidden_dim <= 0:
    raise ValueError(f"Invalid hidden dimension: {hidden_dim}.")
if b1.shape != (hidden_dim,):
    raise ValueError(
        f"Invalid first layer bias shape {b1.shape}; expected {(hidden_dim,)}."
    )
if w2.shape != (len(vocab), hidden_dim):
    raise ValueError(
        f"Invalid output layer shape {w2.shape}; "
        f"expected {(len(vocab), hidden_dim)}."
    )
if b2.shape != (len(vocab),):
    raise ValueError(
        f"Invalid classifier bias shape {b2.shape}; expected {(len(vocab),)}."
    )

vocab_hash = hashlib.sha256(
    json.dumps(vocab, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
).hexdigest()

np.savez(DST, w1=w1, b1=b1, w2=w2, b2=b2)

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

if CONFIG_FILE.exists():
    with CONFIG_FILE.open(encoding="utf-8") as file:
        config = json.load(file)
else:
    config = {}

config.update({
    "input_dim": int(w1.shape[1]),
    "num_labels": int(len(vocab)),
    "hidden_dim": hidden_dim,
    "label_vocab_path": "model/prep/v2/label_vocab_v2.json",
    "label_vocab_sha256": vocab_hash,
    "classifier_pt_sha256": sha256_file(SRC),
    "classifier_npz_sha256": sha256_file(DST),
})

with CONFIG_FILE.open("w", encoding="utf-8") as file:
    json.dump(config, file, indent=2)
    file.write("\n")

print(f"Wrote {DST}")
print(f"Validated classifier: input=384, hidden={hidden_dim}, labels={len(vocab)}")
print(f"Synchronized {CONFIG_FILE}: labels={len(vocab)}, input_dim={w1.shape[1]}, hidden_dim={hidden_dim}")
