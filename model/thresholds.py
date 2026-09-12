"""Exact per-label decision-threshold optimization for multi-label classifiers."""
from __future__ import annotations

import numpy as np


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Return the exact threshold that maximizes binary F1 for one label.

    Thresholds are evaluated at the unique model scores. Because predictions
    only change when the threshold crosses a score, this is an exact search,
    not a coarse grid search such as np.arange(..., 0.05).

    Ties use the lowest threshold among equally optimal candidates, which
    favors recall without changing the maximum F1.
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)

    if y_true.shape[0] != scores.shape[0]:
        raise ValueError("y_true and scores must have the same number of rows")
    if y_true.size == 0:
        return 0.5, 0.0

    positives = int(y_true.sum())
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_true = y_true[order]

    # Scan score groups so equal scores are always included together. This
    # exactly matches predictions made with (score >= threshold).
    best_threshold = float(sorted_scores[0])
    best_f1 = 0.0
    tp = 0
    predicted_positive = 0
    i = 0
    n = len(sorted_scores)

    while i < n:
        score = sorted_scores[i]
        j = i
        group_tp = 0
        while j < n and sorted_scores[j] == score:
            group_tp += int(sorted_true[j])
            j += 1

        tp += group_tp
        predicted_positive = j
        fp = predicted_positive - tp
        fn = positives - tp
        denom = 2 * tp + fp + fn
        f1 = (2.0 * tp / denom) if denom else 0.0

        # <= deliberately picks the lower threshold on a tie.
        if f1 >= best_f1:
            best_f1 = f1
            best_threshold = float(score)

        i = j

    # A threshold of zero predicts every class when model scores are positive.
    # It is already represented by the minimum score for sigmoid outputs.
    return best_threshold, best_f1


def optimize_per_label_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the exact F1-optimal threshold independently for every label."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if y_true.ndim != 2 or scores.ndim != 2 or y_true.shape != scores.shape:
        raise ValueError("y_true and scores must be 2D arrays with identical shapes")

    thresholds = np.empty(scores.shape[1], dtype=np.float64)
    best_f1 = np.empty(scores.shape[1], dtype=np.float64)
    for label_idx in range(scores.shape[1]):
        thresholds[label_idx], best_f1[label_idx] = best_f1_threshold(
            y_true[:, label_idx], scores[:, label_idx]
        )
    return thresholds, best_f1
