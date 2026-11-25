from model.label_display import display_label
from model.prep.data_prep import GENERIC_LABELS


def test_common_abbreviations_keep_readable_casing():
    assert display_label("nlp") == "NLP"
    assert display_label("ai/ml") == "AI/ML"
    assert display_label("ci/cd") == "CI/CD"
    assert display_label("aws/azure") == "AWS/Azure"


def test_product_and_multiword_labels_use_title_case():
    assert display_label("claude code") == "Claude Code"
    assert display_label("machine learning") == "Machine Learning"
    assert display_label("power bi") == "Power BI"
    assert display_label("node.js") == "Node.js"


def test_generic_buckets_are_excluded_but_specific_phrases_are_not():
    assert {"web", "programming", "software"}.issubset(GENERIC_LABELS)
    assert "web development" not in GENERIC_LABELS
    assert "programming languages" not in GENERIC_LABELS
