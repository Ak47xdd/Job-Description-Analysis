"""Validate every v2 SBERT training/runtime artifact against one vocabulary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LABEL_FILE = ROOT / "model" / "prep" / "v2" / "label_vocab_v2.json"
DATA_FILE = ROOT / "model" / "prep" / "v2" / "prepared_data_v2.npz"
MANIFEST_FILE = ROOT / "model" / "prep" / "v2" / "data_manifest.json"
EMBED_DIR = ROOT / "model" / "prep" / "v2_sentence_transformer"
OUT_DIR = ROOT / "model_out" / "v2_sentence_transformer"


def vocab_hash(vocab: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(vocab, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def main() -> None:
    required = [LABEL_FILE, DATA_FILE, OUT_DIR / "skill_classifier_sbert_v2.pt",
                OUT_DIR / "skill_classifier_sbert_v2.npz", OUT_DIR / "model_config.json"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing SBERT artifacts:\n" + "\n".join(missing))

    vocab = json.loads(LABEL_FILE.read_text(encoding="utf-8"))
    if not isinstance(vocab, list) or not vocab or len(vocab) != len(set(vocab)):
        raise ValueError("label_vocab_v2.json is invalid or contains duplicates.")
    n_labels = len(vocab)
    vhash = vocab_hash(vocab)

    split = np.load(DATA_FILE)
    y_train, y_test = split["y_train"], split["y_test"]
    if y_train.shape[1] != n_labels or y_test.shape[1] != n_labels:
        raise ValueError(
            f"Prepared labels mismatch: train={y_train.shape[1]}, "
            f"test={y_test.shape[1]}, vocab={n_labels}."
        )

    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8")) if MANIFEST_FILE.exists() else {}
    if manifest.get("num_labels") not in (None, n_labels):
        raise ValueError("data_manifest.json label count does not match vocabulary.")
    if manifest.get("label_vocab_sha256") not in (None, vhash):
        raise ValueError("data_manifest.json vocabulary hash does not match vocabulary.")

    metadata_path = EMBED_DIR / "metadata.json"
    embeddings = np.load(EMBED_DIR / "embeddings.npy", mmap_mode="r")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if embeddings.ndim != 2 or embeddings.shape[1] != 384:
        raise ValueError(f"Expected 384-D embeddings, got {embeddings.shape}.")
    if int(metadata.get("num_samples", -1)) != len(embeddings):
        raise ValueError("Embedding metadata sample count does not match embeddings.npy.")

    state = torch.load(
        OUT_DIR / "skill_classifier_sbert_v2.pt",
        map_location="cpu",
        weights_only=True,
    )
    for key in ("net.0.weight", "net.0.bias", "net.3.weight", "net.3.bias"):
        if key not in state:
            raise ValueError(f"Checkpoint missing {key}.")

    shapes = {
        "w1": tuple(state["net.0.weight"].shape),
        "b1": tuple(state["net.0.bias"].shape),
        "w2": tuple(state["net.3.weight"].shape),
        "b2": tuple(state["net.3.bias"].shape),
    }
    expected = {"w1": (64, 384), "b1": (64,), "w2": (n_labels, 64), "b2": (n_labels,)}
    if shapes != expected:
        raise ValueError(f"Checkpoint shapes {shapes} do not match {expected}.")

    with np.load(OUT_DIR / "skill_classifier_sbert_v2.npz", allow_pickle=False) as data:
        npz_shapes = {key: tuple(data[key].shape) for key in ("w1", "b1", "w2", "b2")}
    if npz_shapes != expected:
        raise ValueError(f"NumPy artifact shapes {npz_shapes} do not match {expected}.")

    config = json.loads((OUT_DIR / "model_config.json").read_text(encoding="utf-8"))
    if int(config.get("num_labels", -1)) != n_labels:
        raise ValueError("model_config.json label count does not match vocabulary.")
    if config.get("label_vocab_sha256") != vhash:
        raise ValueError("model_config.json vocabulary hash does not match vocabulary.")
    if int(config.get("input_dim", -1)) != 384 or int(config.get("hidden_dim", -1)) != 64:
        raise ValueError("model_config.json dimensions do not match the classifier.")

    print("SBERT artifact validation passed.")
    print(f"labels={n_labels} input_dim=384 hidden_dim=64 embeddings={len(embeddings)}")
    print(f"label_vocab_sha256={vhash}")


if __name__ == "__main__":
    main()
