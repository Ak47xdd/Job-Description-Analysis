"""
JobAnalyze v2-SBERT training pipeline.

Trains the same multi-label PyTorch MLP architecture as model/model.py,
but replaces TF-IDF inputs with precomputed Sentence Transformer embeddings.

Prerequisites:
    1. python model/prep/data_prep.py
    2. python model/sentence_transformer_v2.py

Run:
    python model/model_sbert.py

Artifacts:
    model_out/v2_sentence_transformer/skill_classifier_sbert_v2.pt
    model_out/v2_sentence_transformer/training_history_sbert_v2.json
    model_out/v2_sentence_transformer/model_config.json
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
PREP_DIR = ROOT / "model" / "prep"
EMBED_DIR = PREP_DIR / "v2_sentence_transformer"
LABEL_FILE = PREP_DIR / "v2" / "label_vocab_v2.json"
SPLIT_FILE = PREP_DIR / "v2" / "prepared_data_v2.npz"
OUT_DIR = ROOT / "model_out" / "v2_sentence_transformer"
DATA_FILE = ROOT / "data" / "clean" / "v2" / "cleaned_job_descriptions_v2.csv"
DATA_MANIFEST = PREP_DIR / "v2" / "data_manifest.json"
EMBED_METADATA = EMBED_DIR / "metadata.json"

SEED = 42
BATCH_SIZE = 32
EPOCHS = 130
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 64
DROPOUT = 0.30


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class SkillDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]


class SBERTSkillClassifier(nn.Module):
    def __init__(self, input_dim: int, num_labels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_pipeline_step(script: Path) -> None:
    print(f"Refreshing SBERT artifact prerequisite: {script.relative_to(ROOT)}")
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)


def _ensure_fresh_v2_inputs() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing cleaned v2 dataset: {DATA_FILE}")

    dataset_hash = _sha256_file(DATA_FILE)
    with DATA_FILE.open("r", encoding="utf-8", newline="") as handle:
        row_count = max(0, sum(1 for _ in csv.reader(handle)) - 1)
    manifest_ok = False

    if DATA_MANIFEST.exists():
        try:
            manifest = json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))
            manifest_ok = (
                manifest.get("source_dataset_sha256") == dataset_hash
                and int(manifest.get("num_rows", -1)) == row_count
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            manifest_ok = False

    if not manifest_ok:
        _run_pipeline_step(ROOT / "model" / "prep" / "data_prep.py")

    embedding_ok = False
    if EMBED_METADATA.exists() and (EMBED_DIR / "embeddings.npy").exists():
        try:
            metadata = json.loads(EMBED_METADATA.read_text(encoding="utf-8"))
            embedding_ok = (
                metadata.get("source_csv_sha256") == dataset_hash
                and int(metadata.get("num_samples", -1)) == row_count
                and metadata.get("model_name") == "all-MiniLM-L6-v2"
                and int(metadata.get("embedding_dimension", -1)) == 384
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            embedding_ok = False

    if not embedding_ok:
        _run_pipeline_step(ROOT / "model" / "sentence_transformer_v2.py")


def load_data():
    _ensure_fresh_v2_inputs()
    embeddings_path = EMBED_DIR / "embeddings.npy"
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Missing {embeddings_path}. Run model/sentence_transformer_v2.py first."
        )
    if not SPLIT_FILE.exists() or not LABEL_FILE.exists():
        raise FileNotFoundError(
            "Missing shared v2 label/split artifacts. Run "
            "python model/prep/data_prep.py first."
        )

    embeddings = np.load(embeddings_path).astype(np.float32)
    split_data = np.load(SPLIT_FILE)
    idx_train = split_data["idx_train"].astype(np.int64)
    idx_test = split_data["idx_test"].astype(np.int64)
    y_train = split_data["y_train"].astype(np.float32)
    y_test = split_data["y_test"].astype(np.float32)

    with LABEL_FILE.open(encoding="utf-8") as file:
        vocab = json.load(file)

    if not isinstance(vocab, list) or not vocab or len(vocab) != len(set(vocab)):
        raise ValueError(
            f"Invalid or duplicate SBERT label vocabulary: {LABEL_FILE}"
        )

    num_labels = len(vocab)
    if y_train.ndim != 2 or y_test.ndim != 2:
        raise ValueError(
            f"Expected 2-D multi-label targets, got train={y_train.shape}, "
            f"test={y_test.shape}. Regenerate v2 data with model/prep/data_prep.py."
        )
    if y_train.shape[1] != num_labels or y_test.shape[1] != num_labels:
        raise ValueError(
            "SBERT label artifact mismatch: "
            f"label_vocab={num_labels}, y_train={y_train.shape[1]}, "
            f"y_test={y_test.shape[1]}. "
            "Run python model/prep/data_prep.py to regenerate the shared "
            "v2 prepared data and label vocabulary together."
        )
    if len(idx_train) != len(y_train) or len(idx_test) != len(y_test):
        raise ValueError(
            "SBERT split/index mismatch: "
            f"train indexes={len(idx_train)}, train labels={len(y_train)}, "
            f"test indexes={len(idx_test)}, test labels={len(y_test)}."
        )
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2-D SBERT embeddings, got {embeddings.shape}.")
    if len(embeddings) == 0 or max(idx_train.max(), idx_test.max()) >= len(embeddings):
        raise ValueError(
            "SBERT embeddings do not cover the saved v2 dataset indexes. "
            "Regenerate embeddings with model/sentence_transformer_v2.py."
        )

    return (
        embeddings[idx_train],
        embeddings[idx_test],
        y_train,
        y_test,
        vocab,
    )


def main() -> None:
    set_seed()
    X_train, X_test, y_train, y_test, vocab = load_data()

    train_loader = DataLoader(
        SkillDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        SkillDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False
    )

    # Never allow a stale checkpoint to survive into a new training run.
    best_checkpoint = OUT_DIR / "skill_classifier_sbert_v2.pt"
    if best_checkpoint.exists():
        best_checkpoint.unlink()

    model = SBERTSkillClassifier(X_train.shape[1], len(vocab))

    pos_counts = y_train.sum(axis=0)
    neg_counts = len(y_train) - pos_counts
    pos_weight = torch.tensor(neg_counts / (pos_counts + 1e-6), dtype=torch.float32)
    pos_weight = torch.clamp(pos_weight, max=10.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    history = {"train_loss": [], "test_loss": []}
    best_test_loss = float("inf")

    print(f"SBERT input dimension: {X_train.shape[1]}")
    print(f"Labels: {len(vocab)} | Train: {len(X_train)} | Test: {len(X_test)}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                test_loss += criterion(model(xb), yb).item() * len(xb)
        test_loss /= len(test_loader.dataset)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), OUT_DIR / "skill_classifier_sbert_v2.pt")

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}/{EPOCHS} | "
                f"train_loss={train_loss:.4f} | test_loss={test_loss:.4f}"
            )

    # Restore the best checkpoint before exporting lightweight inference
    # weights. This keeps the .pt, .npz, config, and vocabulary synchronized.
    best_checkpoint = OUT_DIR / "skill_classifier_sbert_v2.pt"
    if not best_checkpoint.exists():
        raise RuntimeError("Training produced no checkpoint; refusing to export stale artifacts.")
    best_state = torch.load(best_checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(best_state)
    model.eval()

    state = model.state_dict()
    expected_output_shape = (len(vocab), HIDDEN_DIM)
    if tuple(state["net.3.weight"].shape) != expected_output_shape:
        raise RuntimeError(
            "Refusing to export SBERT classifier: "
            f'final layer is {tuple(state["net.3.weight"].shape)}, '
            f"expected {expected_output_shape}."
        )
    if tuple(state["net.3.bias"].shape) != (len(vocab),):
        raise RuntimeError(
            "Refusing to export SBERT classifier: "
            f'final bias is {tuple(state["net.3.bias"].shape)}, '
            f"expected {(len(vocab),)}."
        )

    vocab_hash = hashlib.sha256(
        json.dumps(vocab, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    np.savez(
        OUT_DIR / "skill_classifier_sbert_v2.npz",
        w1=state["net.0.weight"].detach().cpu().numpy().astype(np.float32),
        b1=state["net.0.bias"].detach().cpu().numpy().astype(np.float32),
        w2=state["net.3.weight"].detach().cpu().numpy().astype(np.float32),
        b2=state["net.3.bias"].detach().cpu().numpy().astype(np.float32),
    )

    with (OUT_DIR / "training_history_sbert_v2.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    with (OUT_DIR / "model_config.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "embedding_model": "all-MiniLM-L6-v2",
                "input_dim": int(X_train.shape[1]),
                "num_labels": len(vocab),
                "label_vocab_path": "model/prep/v2/label_vocab_v2.json",
                "label_vocab_sha256": vocab_hash,
                "hidden_dim": HIDDEN_DIM,
                "dropout": DROPOUT,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "best_test_loss": best_test_loss,
            },
            file,
            indent=2,
        )

    print(f"Saved best SBERT classifier to {OUT_DIR}")
    print(f"Final artifact dimensions: input={X_train.shape[1]}, labels={len(vocab)}")


if __name__ == "__main__":
    main()
