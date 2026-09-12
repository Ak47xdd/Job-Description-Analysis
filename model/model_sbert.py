"""
JobAnalyze v2-SBERT training pipeline.

Trains the same multi-label PyTorch MLP architecture as model/model.py,
but replaces TF-IDF inputs with precomputed Sentence Transformer embeddings.

Prerequisites:
    python model/sentence_transformer_v2.py

Run:
    python model/model_sbert.py

Artifacts:
    model_out/v2_sentence_transformer/skill_classifier_sbert_v2.pt
    model_out/v2_sentence_transformer/training_history_sbert_v2.json
    model_out/v2_sentence_transformer/model_config.json
"""

from __future__ import annotations

import json
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


def load_data():
    embeddings_path = EMBED_DIR / "embeddings.npy"
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Missing {embeddings_path}. Run model/sentence_transformer_v2.py first."
        )
    if not SPLIT_FILE.exists() or not LABEL_FILE.exists():
        raise FileNotFoundError(
            "Missing v2 prepared labels/splits. Run the v2 data preparation pipeline first."
        )

    embeddings = np.load(embeddings_path).astype(np.float32)
    split_data = np.load(SPLIT_FILE)
    idx_train = split_data["idx_train"].astype(np.int64)
    idx_test = split_data["idx_test"].astype(np.int64)
    y = np.concatenate([split_data["y_train"], split_data["y_test"]], axis=0)

    # prepared_data stores labels in split order, so reconstruct original-row labels
    # using the saved indexes. This keeps SBERT embeddings aligned with the TF-IDF split.
    y_by_index = np.empty_like(y)
    y_by_index[idx_train] = split_data["y_train"]
    y_by_index[idx_test] = split_data["y_test"]

    with LABEL_FILE.open(encoding="utf-8") as file:
        vocab = json.load(file)

    return (
        embeddings[idx_train],
        embeddings[idx_test],
        y_by_index[idx_train],
        y_by_index[idx_test],
        vocab,
    )


def evaluate_loss(model, loader, criterion):
    model.eval()
    losses = []
    with torch.no_grad():
        for xb, yb in loader:
            losses.append(criterion(model(xb), yb).item())
    return float(np.mean(losses)) if losses else 0.0


def main() -> None:
    set_seed()
    X_train, X_test, y_train, y_test, vocab = load_data()

    train_loader = DataLoader(SkillDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(SkillDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    model = SBERTSkillClassifier(X_train.shape[1], len(vocab))

    pos_counts = y_train.sum(axis=0)
    neg_counts = len(y_train) - pos_counts
    pos_weight = torch.tensor(neg_counts / (pos_counts + 1e-6), dtype=torch.float32)
    pos_weight = torch.clamp(pos_weight, max=10.0)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    history = {"train_loss": [], "test_loss": []}
    best_test_loss = float("inf")
    best_state = None

    print(f"SBERT input dimension: {X_train.shape[1]}")
    print(f"Labels: {len(vocab)} | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        test_loss = evaluate_loss(model, test_loader, criterion)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | train_loss {train_loss:.4f} | test_loss {test_loss:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), OUT_DIR / "skill_classifier_sbert_v2.pt")
    (OUT_DIR / "training_history_sbert_v2.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (OUT_DIR / "model_config.json").write_text(json.dumps({
        "embedding_model": "all-MiniLM-L6-v2",
        "input_dim": int(X_train.shape[1]),
        "num_labels": len(vocab),
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "best_test_loss": best_test_loss,
    }, indent=2), encoding="utf-8")

    print(f"\nSaved best model to: {OUT_DIR / 'skill_classifier_sbert_v2.pt'}")


if __name__ == "__main__":
    main()
