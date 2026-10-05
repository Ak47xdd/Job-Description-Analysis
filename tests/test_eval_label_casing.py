from model.eval_label_utils import (
    label_intersection,
    normalize_prediction_keys,
    normalize_target_labels,
)


def test_github_casing_matches_case_insensitively():
    predictions = {"GitHub": 0.885, "Git": 0.885}
    targets = ["Github", "Git"]

    assert label_intersection(predictions, targets) == {"github", "git"}


def test_prediction_dictionary_keys_are_lowercased_for_comparison():
    predictions = {"GitHub": 0.885, "Machine Learning": 0.91}

    normalized = normalize_prediction_keys(predictions)

    assert set(normalized) == {"github", "machine learning"}
    assert normalized["github"] == 0.885


def test_target_labels_are_lowercased_for_comparison():
    targets = ["Github", "Deep Learning", "Git"]

    assert normalize_target_labels(targets) == [
        "github",
        "deep learning",
        "git",
    ]
