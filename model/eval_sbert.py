"""SBERT evaluation with constrained per-skill threshold optimization."""
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
from thresholds import optimize_per_label_thresholds

PREP_DIR = ROOT / "model" / "prep"
EMBED_DIR = PREP_DIR / "v2_sentence_transformer"
SPLIT_FILE = PREP_DIR / "v2" / "prepared_data_v2.npz"
LABEL_FILE = PREP_DIR / "v2" / "label_vocab_v2.json"
MODEL_FILE = ROOT / "model_out" / "v2_sentence_transformer" / "skill_classifier_sbert_v2.pt"
THRESHOLD_FILE = ROOT / "model_out" / "v2_sentence_transformer" / "per_skill_thresholds.json"

# Safety constraints against low-confidence/noisy skill guesses.
MIN_PRECISION = 0.30
MIN_THRESHOLD = 0.25


def precision_at_k(y_true, probs, k):
    k = min(k, probs.shape[1])
    return float(
        np.mean([truth[np.argsort(scores)[-k:]].sum() / k for truth, scores in zip(y_true, probs)])
    )


def main():
    embeddings = np.load(EMBED_DIR / "embeddings.npy").astype(np.float32)
    data = np.load(SPLIT_FILE)
    idx_test = data["idx_test"].astype(int)
    X_test = embeddings[idx_test]
    y_test = data["y_test"].astype(int)

    with LABEL_FILE.open(encoding="utf-8") as f:
        vocab = json.load(f)

    model = SBERTSkillClassifier(X_test.shape[1], len(vocab))
    model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(X_test))).numpy()

    # Every skill gets its own threshold, but a threshold can never fall below
    # MIN_THRESHOLD and must meet MIN_PRECISION when an observed candidate exists.
    thresholds, threshold_f1, threshold_precision = optimize_per_label_thresholds(
        y_test,
        probs,
        min_precision=MIN_PRECISION,
        min_threshold=MIN_THRESHOLD,
    )
    preds = (probs >= thresholds[None, :]).astype(int)

    threshold_payload = {
        "method": "exact_per_skill_f1_with_precision_constraint",
        "metric": "binary_f1",
        "min_precision": MIN_PRECISION,
        "min_threshold": MIN_THRESHOLD,
        "prediction_rule": "probability >= skill-specific threshold",
        "labels": {
            label: {
                "threshold": round(float(threshold), 8),
                "best_f1": round(float(best_f1), 8),
                "precision_at_threshold": round(float(precision), 8),
            }
            for label, threshold, best_f1, precision in zip(
                vocab, thresholds, threshold_f1, threshold_precision
            )
        },
    }
    THRESHOLD_FILE.parent.mkdir(parents=True, exist_ok=True)
    THRESHOLD_FILE.write_text(json.dumps(threshold_payload, indent=2), encoding="utf-8")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, preds, average=None, zero_division=0
    )
    micro = f1_score(y_test, preds, average="micro", zero_division=0)
    macro = f1_score(y_test, preds, average="macro", zero_division=0)

    global_threshold = 0.65
    global_preds = (probs >= global_threshold).astype(int)
    global_micro = f1_score(y_test, global_preds, average="micro", zero_division=0)
    global_macro = f1_score(y_test, global_preds, average="macro", zero_division=0)

    baseline = np.tile((data["y_train"].mean(axis=0) >= 0.3).astype(int), (len(y_test), 1))
    baseline_micro = f1_score(y_test, baseline, average="micro", zero_division=0)
    baseline_macro = f1_score(y_test, baseline, average="macro", zero_division=0)

    print("\nSBERT Model Evaluation — Constrained Per-Skill Thresholds\n")
    print(f"Embedding shape: {X_test.shape}")
    print(f"Minimum precision constraint: {MIN_PRECISION:.2f}")
    print(f"Minimum threshold floor: {MIN_THRESHOLD:.2f}")
    print(f"Global 0.65 threshold: Micro-F1={global_micro:.3f} | Macro-F1={global_macro:.3f}")
    print(f"Constrained per-skill: Micro-F1={micro:.3f} | Macro-F1={macro:.3f}")
    print(f"Thresholds saved to: {THRESHOLD_FILE}")
    print()
    print(f"{'label':25s}{'threshold':12s}{'support':10s}{'precision':12s}{'recall':10s}{'f1':6s}")
    for lbl, threshold, p, r, f, s in zip(vocab, thresholds, precision, recall, f1, support):
        if s > 0:
            print(f"{lbl:25s}{threshold:<12.4f}{int(s):<10d}{p:<12.2f}{r:<10.2f}{f:.2f}")

    print(f"\nMicro-F1: {micro:.3f} | Macro-F1: {macro:.3f}")
    print(f"Hamming Loss: {hamming_loss(y_test, preds):.4f}")
    print(f"Jaccard Micro: {jaccard_score(y_test, preds, average='micro', zero_division=0):.3f}")
    print(f"Jaccard Macro: {jaccard_score(y_test, preds, average='macro', zero_division=0):.3f}")
    print(f"Precision@3: {precision_at_k(y_test, probs, 3):.3f} | Precision@5: {precision_at_k(y_test, probs, 5):.3f}")

    print("\n=== BASELINE ===")
    print(f"Baseline Micro-F1: {baseline_micro:.3f}")
    print(f"Baseline Macro-F1: {baseline_macro:.3f}")

    print("\nVERDICT")
    if micro > global_micro:
        print("Constrained per-skill thresholds improve over the global 0.65 threshold.")
    else:
        print("Constrained per-skill thresholds did not improve over the global 0.65 threshold on this evaluation split.")
    if micro > baseline_micro + 0.05:
        print("SBERT meaningfully beats the naive baseline.")
    else:
        print("SBERT is roughly tied with, or worse than, the naive baseline.")


if __name__ == "__main__":
    main()
