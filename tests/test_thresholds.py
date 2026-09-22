import numpy as np

from model.thresholds import best_threshold, optimize_per_label_thresholds, support_aware_threshold_floor


def test_support_floor_policy():
    assert support_aware_threshold_floor(3) == 0.70
    assert support_aware_threshold_floor(5) == 0.55
    assert support_aware_threshold_floor(9) == 0.55
    assert support_aware_threshold_floor(10) == 0.40
    assert support_aware_threshold_floor(19) == 0.40
    assert support_aware_threshold_floor(20) == 0.25
    assert support_aware_threshold_floor(100) == 0.25


def test_rare_skill_cannot_optimize_below_high_floor():
    y_true = np.array([1, 0, 0, 0, 0, 0])
    scores = np.array([0.61, 0.60, 0.59, 0.58, 0.10, 0.05])

    threshold, _, _ = best_threshold(
        y_true,
        scores,
        min_precision=0.30,
        min_threshold=0.25,
    )

    assert threshold >= 0.70


def test_support_aware_optimizer_applies_each_label_floor():
    y_true = np.array([
        [1, 1, 1],
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ])
    scores = np.array([
        [0.61, 0.61, 0.61],
        [0.60, 0.60, 0.60],
        [0.59, 0.59, 0.59],
        [0.58, 0.58, 0.58],
        [0.10, 0.10, 0.10],
        [0.05, 0.05, 0.05],
    ])

    thresholds, _, _ = optimize_per_label_thresholds(y_true, scores)

    # Supports are 1, 2 and 3: all must use the strict rare-label floor.
    assert np.all(thresholds >= 0.70)
