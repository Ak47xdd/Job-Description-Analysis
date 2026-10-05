import numpy as np

from api.JobAnalyze.v2.pred_v2 import _apply_core_domain_guard


def test_core_domain_requires_080_without_explicit_term():
    probabilities = np.array([[0.79, 0.84]], dtype=np.float32)
    labels = ["machine learning", "deep learning"]

    updated = _apply_core_domain_guard(
        probabilities.copy(),
        labels,
        ["Python, FastAPI, REST APIs and data analysis."],
    )

    assert updated[0, 0] == 0.0
    assert updated[0, 1] == 0.84


def test_core_domain_allows_explicit_machine_learning():
    probabilities = np.array([[0.71, 0.79]], dtype=np.float32)
    labels = ["machine learning", "deep learning"]

    updated = _apply_core_domain_guard(
        probabilities.copy(),
        labels,
        ["Experience with machine learning pipelines; deep learning is a plus."],
    )

    assert updated[0, 0] == 0.71
    assert updated[0, 1] == 0.79


def test_core_domain_guard_does_not_affect_other_skills():
    probabilities = np.array([[0.40, 0.79, 0.50]], dtype=np.float32)
    labels = ["python", "machine learning", "fastapi"]

    updated = _apply_core_domain_guard(
        probabilities.copy(),
        labels,
        ["Backend engineering with APIs."],
    )

    assert updated[0, 0] == 0.40
    assert updated[0, 1] == 0.0
    assert updated[0, 2] == 0.50
