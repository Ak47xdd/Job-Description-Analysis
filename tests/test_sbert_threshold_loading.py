import json

import numpy as np

from api.JobAnalyze.v2.pred_v2 import _load_per_skill_thresholds


def test_deployment_loads_disabled_n8n_threshold(tmp_path):
    threshold_file = tmp_path / "per_skill_thresholds.json"
    threshold_file.write_text(
        json.dumps(
            {
                "labels": {
                    "python": {"threshold": 0.65, "status": "active"},
                    "n8n": {"threshold": 1.0, "status": "disabled"},
                }
            }
        ),
        encoding="utf-8",
    )

    thresholds = _load_per_skill_thresholds(threshold_file, ["python", "n8n"])

    assert np.allclose(thresholds, [0.65, 1.0])
    assert thresholds[1] == 1.0


def test_deployment_rejects_threshold_vocab_mismatch(tmp_path):
    threshold_file = tmp_path / "per_skill_thresholds.json"
    threshold_file.write_text(
        json.dumps({"labels": {"python": {"threshold": 0.65}}}),
        encoding="utf-8",
    )

    try:
        _load_per_skill_thresholds(threshold_file, ["python", "n8n"])
    except ValueError as exc:
        assert "vocabulary mismatch" in str(exc)
    else:
        raise AssertionError("Expected threshold vocabulary mismatch")


def test_disabled_threshold_filters_false_positive():
    labels = ["python", "n8n"]
    probabilities = np.array([0.80, 0.604], dtype=np.float32)
    thresholds = np.array([0.65, 1.0], dtype=np.float32)

    eligible = [
        skill
        for skill, score, threshold in zip(labels, probabilities, thresholds)
        if float(score) >= float(threshold)
    ]

    assert eligible == ["python"]
    assert "n8n" not in eligible
