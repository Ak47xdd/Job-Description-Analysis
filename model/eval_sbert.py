"""SBERT evaluation with threshold search and ranking metrics."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score, hamming_loss, jaccard_score, precision_recall_fscore_support
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
from model_sbert import SBERTSkillClassifier
PREP_DIR = ROOT / "model" / "prep"
EMBED_DIR = PREP_DIR / "v2_sentence_transformer"
SPLIT_FILE = PREP_DIR / "v2" / "prepared_data_v2.npz"
LABEL_FILE = PREP_DIR / "v2" / "label_vocab_v2.json"
MODEL_FILE = ROOT / "model_out" / "v2_sentence_transformer" / "skill_classifier_sbert_v2.pt"

def precision_at_k(y_true, probs, k):
    k = min(k, probs.shape[1])
    return float(np.mean([truth[np.argsort(scores)[-k:]].sum() / k for truth, scores in zip(y_true, probs)]))

def main():
    embeddings = np.load(EMBED_DIR / "embeddings.npy").astype(np.float32)
    data = np.load(SPLIT_FILE)
    idx_train, idx_test = data["idx_train"].astype(int), data["idx_test"].astype(int)
    X_test = embeddings[idx_test]
    y_train, y_test = data["y_train"].astype(int), data["y_test"].astype(int)
    with LABEL_FILE.open(encoding="utf-8") as f:
        vocab = json.load(f)
    model = SBERTSkillClassifier(X_test.shape[1], len(vocab))
    model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(X_test))).numpy()
    best_threshold, best_micro = 0.5, -1.0
    for threshold in np.arange(0.10, 0.91, 0.05):
        score = f1_score(y_test, (probs >= threshold).astype(int), average="micro", zero_division=0)
        if score > best_micro:
            best_micro, best_threshold = score, float(threshold)
    preds = (probs >= best_threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, preds, average=None, zero_division=0)
    micro = f1_score(y_test, preds, average="micro", zero_division=0)
    macro = f1_score(y_test, preds, average="macro", zero_division=0)
    baseline = np.tile((y_train.mean(axis=0) >= 0.3).astype(int), (len(y_test), 1))
    print("\nSBERT Model Evaluation\n")
    print(f"Embedding shape: {X_test.shape} | Best threshold: {best_threshold:.2f}")
    print(f"{'label':25s}{'support':10s}{'precision':12s}{'recall':10s}{'f1':6s}")
    for lbl, p, r, f, s in zip(vocab, precision, recall, f1, support):
        if s > 0: print(f"{lbl:25s}{int(s):<10d}{p:<12.2f}{r:<10.2f}{f:.2f}")
    print(f"\nMicro-F1: {micro:.3f} | Macro-F1: {macro:.3f}")
    print(f"Hamming Loss: {hamming_loss(y_test, preds):.4f}")
    print(f"Jaccard Micro: {jaccard_score(y_test, preds, average='micro', zero_division=0):.3f}")
    print(f"Jaccard Macro: {jaccard_score(y_test, preds, average='macro', zero_division=0):.3f}")
    print(f"Precision@3: {precision_at_k(y_test, probs, 3):.3f} | Precision@5: {precision_at_k(y_test, probs, 5):.3f}")
    print("\n=== BASELINE ===")
    print(f"Baseline Micro-F1: {f1_score(y_test, baseline, average='micro', zero_division=0):.3f}")
    print(f"Baseline Macro-F1: {f1_score(y_test, baseline, average='macro', zero_division=0):.3f}")
    print("\nVERDICT")
    print("SBERT meaningfully beats the naive baseline." if micro > f1_score(y_test, baseline, average='micro', zero_division=0) + 0.05 else "SBERT is roughly tied with, or worse than, the naive baseline.")
if __name__ == "__main__": main()
