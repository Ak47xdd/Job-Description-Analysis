"""Case-insensitive label comparison helpers for evaluation output."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def normalize_label(label: str) -> str:
    """Normalize a skill label for evaluation-only comparisons."""
    return str(label).strip().lower()


def normalize_prediction_keys(predictions: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize prediction-dictionary keys without changing their values."""
    return {normalize_label(label): value for label, value in predictions.items()}


def normalize_target_labels(targets: Iterable[str]) -> list[str]:
    """Normalize target labels for case-insensitive evaluation."""
    return [normalize_label(label) for label in targets]


def label_intersection(
    predictions: Mapping[str, Any],
    targets: Iterable[str],
) -> set[str]:
    """Return the case-insensitive intersection of prediction and target labels."""
    prediction_keys = set(normalize_prediction_keys(predictions))
    target_keys = set(normalize_target_labels(targets))
    return prediction_keys & target_keys
