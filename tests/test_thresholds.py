import numpy as np
import pytest

from model.thresholds import best_threshold, optimize_per_label_thresholds, support_aware_threshold_floor


def test_support_floor_policy():
    assert support_aware_threshold_floor(3) == 0.70
    assert support_aware_threshold_floor(5) == 0.55
    assert support_aware_threshold_floor(9) == 0.55
    assert support_aware_threshold_floor(10) == 0.40
    assert support_aware_threshold_floor(19) == 0.40
    assert support_aware_threshold_floor(20) == 0.25
    assert support_aware_threshold_floor(100) == 0.25


def test_ultra_rare_skill_can_use_015_floor():
    y_true = np.array([1, 0, 0, 0, 0, 0])
    scores = np.array([0.20, 0.19, 0.10, 0.05, 0.03, 0.01])

    threshold, _, _ = best_threshold(
        y_true,
        scores,
        min_precision=0.30,
        min_threshold=0.25,
    )

    assert threshold == 0.20


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

    # Supports are 1, 2 and 3: all may search below the legacy 0.70 floor,
    # but the new ultra-rare floor of 0.15 still applies.
    assert np.all(thresholds >= 0.15)

def test_zero_f1_fallback_tries_thresholds_sequentially():
    # At 0.50 there is no prediction, so the optimizer must continue to 0.45.
    y_true = np.array([1, 0, 0, 0])
    scores = np.array([0.46, 0.44, 0.20, 0.10])

    threshold, f1, precision = best_threshold(
        y_true,
        scores,
        min_precision=0.30,
        min_threshold=0.25,
    )

    assert threshold == 0.46
    assert f1 > 0.0
    assert precision == 1.0


def test_zero_f1_fallback_steps_back_up_until_precision_clears():
    # 0.45 predicts two rows and has precision 0.50, so it is rejected.
    # The optimizer must step back up to 0.46, where the single prediction is correct.
    y_true = np.array([1, 0, 0, 0])
    scores = np.array([0.46, 0.45, 0.44, 0.43])

    threshold, f1, precision = best_threshold(
        y_true,
        scores,
        min_precision=0.30,
        min_threshold=0.25,
    )

    assert threshold == 0.46
    assert f1 > 0.0
    assert precision == 1.0


def test_precision_constraint_raises_when_no_valid_threshold_exists():
    y_true = np.array([1, 0, 0, 0])
    scores = np.array([0.39, 0.50, 0.49, 0.48])

    with pytest.raises(ValueError, match="No threshold satisfies the precision constraint"):
        best_threshold(
            y_true,
            scores,
            min_precision=0.30,
            min_threshold=0.25,
        )


def test_support_above_three_keeps_existing_floor():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0])
    scores = np.array([0.60, 0.59, 0.58, 0.57, 0.20, 0.10, 0.05])

    threshold, _, precision = best_threshold(
        y_true,
        scores,
        min_precision=0.30,
        min_threshold=0.25,
    )

    assert threshold >= 0.70
    assert precision >= 0.30

def test_infeasible_label_can_be_explicitly_disabled():
    y_true = np.array([1, 0, 0, 0])
    scores = np.array([0.39, 0.50, 0.49, 0.48])

    threshold, f1, precision = best_threshold(
        y_true,
        scores,
        min_precision=0.30,
        min_threshold=0.25,
        on_infeasible="disable",
    )

    assert threshold > 1.0
    assert f1 == 0.0
    assert precision == 0.0


def test_optimizer_can_disable_only_infeasible_labels():
    y_true = np.array([
        [1, 0],
        [0, 0],
        [0, 0],
        [0, 0],
    ])
    scores = np.array([
        [0.90, 0.39],
        [0.20, 0.50],
        [0.10, 0.49],
        [0.05, 0.48],
    ])

    thresholds, f1, precision = optimize_per_label_thresholds(
        y_true,
        scores,
        min_precision=0.30,
        min_threshold=0.25,
        on_infeasible="disable",
    )

    assert thresholds[0] == 0.90
    assert thresholds[1] > 1.0
    assert f1[0] > 0.0
    assert f1[1] == 0.0
    assert precision[0] >= 0.30
