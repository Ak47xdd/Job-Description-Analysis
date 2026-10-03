"""Per-label decision-threshold optimization with safety constraints."""
from __future__ import annotations

import numpy as np


def support_aware_threshold_floor(
    support: int,
    *,
    base_floor: float = 0.25,
    low_support_cutoff: int = 5,
    low_support_floor: float = 0.70,
    medium_support_cutoff: int = 10,
    medium_support_floor: float = 0.55,
    established_support_cutoff: int = 20,
    established_support_floor: float = 0.40,
) -> float:
    """Return a conservative threshold floor based on positive-label support."""
    if support < 0:
        raise ValueError("support must be non-negative")
    if support < low_support_cutoff:
        return float(max(base_floor, low_support_floor))
    if support < medium_support_cutoff:
        return float(max(base_floor, medium_support_floor))
    if support < established_support_cutoff:
        return float(max(base_floor, established_support_floor))
    return float(base_floor)


def best_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    min_precision: float = 0.30,
    min_threshold: float = 0.25,
    support_aware: bool = True,
    low_support_cutoff: int = 5,
    low_support_floor: float = 0.70,
    rare_support_cutoff: int = 3,
    rare_support_floor: float = 0.15,
    medium_support_cutoff: int = 10,
    medium_support_floor: float = 0.55,
    established_support_cutoff: int = 20,
    established_support_floor: float = 0.40,
    zero_f1_fallback_thresholds: tuple[float, ...] = (0.50, 0.45, 0.40),
    on_infeasible: str = "raise",
) -> tuple[float, float, float]:
    """Find a threshold whose measured precision satisfies min_precision.

    Among valid thresholds, choose the one with the highest F1. Ties choose the
    lower threshold, which favors recall. Relaxed rare-label thresholds are
    re-checked against the same precision constraint.

    The search is exact: predictions only change at unique model scores. If no
    threshold can satisfy the requested precision, raise ValueError rather than
    returning an unsafe threshold below the precision floor.
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)

    if y_true.shape[0] != scores.shape[0]:
        raise ValueError("y_true and scores must have the same number of rows")
    if y_true.size == 0:
        raise ValueError("Cannot enforce a precision constraint on an empty label set")
    if not 0.0 <= min_precision <= 1.0:
        raise ValueError("min_precision must be between 0 and 1")
    if not 0.0 <= min_threshold <= 1.0:
        raise ValueError("min_threshold must be between 0 and 1")
    if not 0.0 <= rare_support_floor <= 1.0:
        raise ValueError("rare_support_floor must be between 0 and 1")
    if rare_support_cutoff < 0:
        raise ValueError("rare_support_cutoff must be non-negative")
    fallback_thresholds = tuple(float(t) for t in zero_f1_fallback_thresholds)
    if any(t < 0.0 or t > 1.0 for t in fallback_thresholds):
        raise ValueError("zero_f1_fallback_thresholds must be between 0 and 1")
    if any(fallback_thresholds[i] == fallback_thresholds[i + 1] for i in range(len(fallback_thresholds) - 1)):
        raise ValueError("zero_f1_fallback_thresholds must contain unique thresholds")
    if on_infeasible not in {"raise", "disable"}:
        raise ValueError('on_infeasible must be "raise" or "disable"')

    positives = int(y_true.sum())
    effective_min_threshold = (
        max(min_threshold, rare_support_floor)
        if support_aware and positives <= rare_support_cutoff
        else support_aware_threshold_floor(
            positives,
            base_floor=min_threshold,
            low_support_cutoff=low_support_cutoff,
            low_support_floor=low_support_floor,
            medium_support_cutoff=medium_support_cutoff,
            medium_support_floor=medium_support_floor,
            established_support_cutoff=established_support_cutoff,
            established_support_floor=established_support_floor,
        )
        if support_aware
        else float(min_threshold)
    )
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_true = y_true[order]

    best_threshold = None
    best_f1 = 0.0
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

        if score >= effective_min_threshold and precision >= min_precision:
            if f1 > best_f1 or (np.isclose(f1, best_f1) and (best_threshold is None or score < best_threshold)):
                best_threshold = score
                best_f1 = f1
                best_precision = precision

        i = j

    # Rare labels can end up with F1=0 at the support-aware floor because the
    # model never predicts them there. Relax the support floor only through the
    # explicit fallback thresholds, but ALWAYS re-check precision. If a relaxed
    # threshold is too permissive (precision < min_precision), walk back upward
    # until the precision constraint is satisfied.
    if best_f1 == 0.0 and positives > 0:
        fallback_candidates = sorted(
            {t for t in fallback_thresholds if min_threshold <= t < effective_min_threshold},
            reverse=True,
        )
        for candidate_threshold in fallback_candidates:
            candidate_pred = scores >= candidate_threshold
            candidate_count = int(candidate_pred.sum())
            candidate_tp = int(np.logical_and(candidate_pred, y_true).sum())
            candidate_fp = candidate_count - candidate_tp
            candidate_fn = positives - candidate_tp
            candidate_precision = candidate_tp / candidate_count if candidate_count else 0.0
            candidate_denom = 2 * candidate_tp + candidate_fp + candidate_fn
            candidate_f1 = (2.0 * candidate_tp / candidate_denom) if candidate_denom else 0.0
            if candidate_f1 > 0.0 and candidate_precision >= min_precision:
                return float(candidate_threshold), float(candidate_f1), float(candidate_precision)

        # Check every observed score below the support-aware floor, from highest
        # to lowest. This explicitly steps the threshold back up to the first
        # observed score that clears min_precision.
        lower_fallback = max(min_threshold, min(fallback_thresholds, default=min_threshold))
        upward_candidates = sorted(
            {float(score) for score in scores if lower_fallback <= float(score) < effective_min_threshold},
            reverse=True,
        )
        for candidate_threshold in upward_candidates:
            candidate_pred = scores >= candidate_threshold
            candidate_count = int(candidate_pred.sum())
            candidate_tp = int(np.logical_and(candidate_pred, y_true).sum())
            candidate_fp = candidate_count - candidate_tp
            candidate_fn = positives - candidate_tp
            candidate_precision = candidate_tp / candidate_count if candidate_count else 0.0
            candidate_denom = 2 * candidate_tp + candidate_fp + candidate_fn
            candidate_f1 = (2.0 * candidate_tp / candidate_denom) if candidate_denom else 0.0
            if candidate_precision >= min_precision:
                return float(candidate_threshold), float(candidate_f1), float(candidate_precision)

    if best_threshold is None:
        if on_infeasible == "disable":
            # A threshold above 1.0 guarantees that sigmoid probabilities in
            # [0, 1] can never activate this label. The label remains visible
            # to the evaluator as an explicit infeasible/disabled label rather
            # than being assigned an unsafe threshold.
            return float(np.nextafter(1.0, np.inf)), 0.0, 0.0
        raise ValueError(
            "No threshold satisfies the precision constraint: "
            f"min_precision={min_precision:.3f}, min_threshold={effective_min_threshold:.3f}"
        )

    return float(best_threshold), float(best_f1), float(best_precision)


def optimize_per_label_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    min_precision: float = 0.30,
    min_threshold: float = 0.25,
    support_aware: bool = True,
    low_support_cutoff: int = 5,
    low_support_floor: float = 0.70,
    medium_support_cutoff: int = 10,
    medium_support_floor: float = 0.55,
    established_support_cutoff: int = 20,
    established_support_floor: float = 0.40,
    zero_f1_fallback_thresholds: tuple[float, ...] = (0.50, 0.45, 0.40),
    rare_support_cutoff: int = 3,
    rare_support_floor: float = 0.15,
    on_infeasible: str = "raise",
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
            support_aware=support_aware,
            low_support_cutoff=low_support_cutoff,
            low_support_floor=low_support_floor,
            medium_support_cutoff=medium_support_cutoff,
            medium_support_floor=medium_support_floor,
            established_support_cutoff=established_support_cutoff,
            established_support_floor=established_support_floor,
            zero_f1_fallback_thresholds=zero_f1_fallback_thresholds,
            rare_support_cutoff=rare_support_cutoff,
            rare_support_floor=rare_support_floor,
            on_infeasible=on_infeasible,
        )

    return thresholds, best_f1, best_precision
