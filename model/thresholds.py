"""Per-label decision-threshold optimization with safety constraints."""
from __future__ import annotations

import numpy as np


def best_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    min_precision: float = 0.30,
    min_threshold: float = 0.25,
) -> tuple[float, float, float]:
    """Find the lowest threshold satisfying the precision constraint.

    Among thresholds with precision >= ``min_precision`` and threshold >=
    ``min_threshold``, choose the one with the highest F1. Ties choose the
    lower threshold, which favors recall. If no candidate satisfies the
    precision constraint, the threshold floor is used as a safe fallback.

    The search is exact: predictions only change at unique model scores.
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)

    if y_true.shape[0] != scores.shape[0]:
        raise ValueError("y_true and scores must have the same number of rows")
    if y_true.size == 0:
        return max(0.5, min_threshold), 0.0, 0.0
    if not 0.0 <= min_precision <= 1.0:
        raise ValueError("min_precision must be between 0 and 1")
    if not 0.0 <= min_threshold <= 1.0:
        raise ValueError("min_threshold must be between 0 and 1")

    positives = int(y_true.sum())
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_true = y_true[order]

    best_threshold = None
    best_f1 = -1.0
    best_precision = 0.0
    tp = 0
    i = 0
    n = len(sorted_scores)

    while i < n:
        score = float(sorted_scores[i])
        j = i
        group_tp = 0
        while j < n and sorted_scores[j] == sorted_scores[i]:
            group_tp += int(sorted_true[j])
            j += 1

        tp += group_tp
        predicted_positive = j
        fp = predicted_positive - tp
        fn = positives - tp
        precision = tp / predicted_positive if predicted_positive else 0.0
        denom = 2 * tp + fp + fn
        f1 = (2.0 * tp / denom) if denom else 0.0

        if score >= min_threshold and precision >= min_precision:
            if f1 > best_f1 or (np.isclose(f1, best_f1) and (best_threshold is None or score < best_threshold)):
                best_threshold = score
                best_f1 = f1
                best_precision = precision

        i = j

    if best_threshold is None:
        # No observed score satisfies the precision requirement. The floor is
        # retained as a predictable safety boundary; downstream evaluation can
        # reveal that this label cannot meet the requested precision on this set.
        fallback_pred = scores >= min_threshold
        fallback_tp = int(np.logical_and(fallback_pred, y_true).sum())
        fallback_count = int(fallback_pred.sum())
        fallback_precision = fallback_tp / fallback_count if fallback_count else 0.0
        fallback_fn = positives - fallback_tp
        fallback_fp = fallback_count - fallback_tp
        fallback_denom = 2 * fallback_tp + fallback_fp + fallback_fn
        fallback_f1 = (2.0 * fallback_tp / fallback_denom) if fallback_denom else 0.0
        return float(min_threshold), fallback_f1, fallback_precision

    return float(best_threshold), float(best_f1), float(best_precision)


def optimize_per_label_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    min_precision: float = 0.30,
    min_threshold: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimize every label independently under precision/floor constraints."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if y_true.ndim != 2 or scores.ndim != 2 or y_true.shape != scores.shape:
        raise ValueError("y_true and scores must be 2D arrays with identical shapes")

    thresholds = np.empty(scores.shape[1], dtype=np.float64)
    best_f1 = np.empty(scores.shape[1], dtype=np.float64)
    best_precision = np.empty(scores.shape[1], dtype=np.float64)

    for label_idx in range(scores.shape[1]):
        thresholds[label_idx], best_f1[label_idx], best_precision[label_idx] = best_threshold(
            y_true[:, label_idx],
            scores[:, label_idx],
            min_precision=min_precision,
            min_threshold=min_threshold,
        )

    return thresholds, best_f1, best_precision
