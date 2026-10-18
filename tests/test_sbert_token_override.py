import numpy as np

from api.JobAnalyze.v2.token_override import (
    apply_token_matching_override,
    detect_exact_keyword_matches,
    detect_regex_fallback_matches,
    select_regex_fallback_labels,
    skill_is_explicitly_present,
)


def test_exact_tokens_are_case_insensitive():
    assert skill_is_explicitly_present("sql", "Experience with SQL and FastAPI.")
    assert skill_is_explicitly_present("fastapi", "Built services with FastAPI.")
    assert skill_is_explicitly_present("jmeter", "Performance testing with JMeter.")
    assert skill_is_explicitly_present("figma", "Designs created in Figma.")


def test_substrings_inside_words_do_not_match():
    assert not skill_is_explicitly_present("git", "GitHub Actions are required.")
    assert not skill_is_explicitly_present("sql", "SQLAlchemy is required.")
    assert not skill_is_explicitly_present("c", "Experience with cloud systems.")


def test_exact_keyword_layer_detects_tools_but_not_abstract_skills():
    labels = ["python", "jmeter", "figma", "git", "fastapi", "system design", "agile development"]
    matches = detect_exact_keyword_matches(
        labels,
        "Use Python, JMeter, Figma, Git and FastAPI. Strong system design and agile development skills.",
    )
    assert matches == ["fastapi", "figma", "git", "jmeter", "python"]


def test_exact_matches_are_promoted_above_their_calibrated_threshold():
    labels = ["python", "jmeter", "figma", "git"]
    probabilities = np.array([0.91, 0.02, 0.03, 0.08], dtype=np.float32)
    thresholds = np.array([0.50, 0.65, 0.70, 0.60], dtype=np.float32)

    updated, matched = apply_token_matching_override(
        probabilities,
        labels,
        "JMeter, Figma and Git are explicitly required.",
        thresholds=thresholds,
    )

    assert matched == ["figma", "git", "jmeter"]
    assert updated[0] == probabilities[0]
    assert updated[1] > thresholds[1]
    assert updated[2] > thresholds[2]
    assert updated[3] > thresholds[3]
    assert np.all(updated <= 1.0)


def test_semantic_only_labels_are_not_forced_by_exact_layer():
    labels = ["system design", "agile development", "infrastructure operations"]
    probabilities = np.array([0.2, 0.3, 0.1], dtype=np.float32)
    thresholds = np.array([0.6, 0.6, 0.6], dtype=np.float32)

    updated, matched = apply_token_matching_override(
        probabilities,
        labels,
        "System design, agile development and infrastructure operations.",
        thresholds=thresholds,
    )

    assert matched == []
    assert np.array_equal(updated, probabilities)


def test_empty_or_unmatched_text_leaves_scores_unchanged():
    labels = ["python", "sql"]
    probabilities = np.array([0.42, 0.37], dtype=np.float32)

    updated, matched = apply_token_matching_override(
        probabilities,
        labels,
        "Experience with JavaScript.",
    )

    assert matched == []
    assert np.array_equal(updated, probabilities)



def test_rare_fallback_selects_disabled_and_zero_f1_rare_labels_only():
    labels = ["disabled_tool", "rare_zero_f1", "rare_working", "common_zero_f1", "system design"]
    payload = {
        "rare_support_cutoff": 5,
        "dataset_support": {
            "disabled_tool": 100,
            "rare_zero_f1": 3,
            "rare_working": 4,
            "common_zero_f1": 100,
            "system design": 1,
        },
        "labels": {
            "disabled_tool": {"status": "disabled", "best_f1": 0.0},
            "rare_zero_f1": {"status": "active", "best_f1": 0.0},
            "rare_working": {"status": "active", "best_f1": 0.25},
            "common_zero_f1": {"status": "active", "best_f1": 0.0},
            "system design": {"status": "disabled", "best_f1": 0.0},
        },
    }

    selected = select_regex_fallback_labels(labels, payload)

    assert selected == ["disabled_tool", "rare_zero_f1"]


def test_rare_fallback_requires_exact_explicit_skill_name():
    fallback_labels = ["jmeter", "fastapi", "sql"]
    assert detect_regex_fallback_matches(
        fallback_labels,
        "JMeter and FastAPI are required; SQLAlchemy is used elsewhere.",
    ) == ["fastapi", "jmeter"]
