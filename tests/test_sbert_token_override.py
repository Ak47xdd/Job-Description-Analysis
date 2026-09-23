import numpy as np

from api.JobAnalyze.v2.token_override import (
    GENERIC_MATCH_BOOST,
    MAX_LEXICAL_PROBABILITY,
    SPECIFIC_MATCH_BOOST,
    apply_token_matching_override,
    skill_is_explicitly_present,
)


def test_exact_tokens_are_case_insensitive():
    assert skill_is_explicitly_present("sql", "Experience with SQL and FastAPI.")
    assert skill_is_explicitly_present("fastapi", "Built services with FastAPI.")


def test_substrings_inside_words_do_not_match():
    assert not skill_is_explicitly_present("git", "GitHub Actions are required.")
    assert not skill_is_explicitly_present("sql", "SQLAlchemy is required.")
    assert not skill_is_explicitly_present("c", "Experience with cloud systems.")


def test_specific_match_is_a_boost_not_a_hard_one():
    labels = ["python", "sql", "fastapi", "git"]
    probabilities = np.array([0.91, 0.12, 0.03, 0.08], dtype=np.float32)

    updated, matched = apply_token_matching_override(
        probabilities,
        labels,
        "Build FastAPI services, write SQL queries, and use Git.",
    )

    assert matched == ["sql", "fastapi", "git"]
    assert updated[0] == probabilities[0]
    assert 0.12 < updated[1] <= MAX_LEXICAL_PROBABILITY
    assert 0.03 < updated[2] <= MAX_LEXICAL_PROBABILITY
    assert 0.08 < updated[3] <= MAX_LEXICAL_PROBABILITY
    assert np.all(updated <= MAX_LEXICAL_PROBABILITY)
    assert np.all(updated < 1.0)


def test_generic_match_gets_smaller_boost():
    labels = ["cloud", "fastapi"]
    probabilities = np.array([0.10, 0.10], dtype=np.float32)

    updated, matched = apply_token_matching_override(
        probabilities,
        labels,
        "Cloud deployment uses FastAPI.",
    )

    assert matched == ["cloud", "fastapi"]
    assert np.isclose(
        updated[0],
        0.10 + (1.0 - 0.10) * GENERIC_MATCH_BOOST,
        atol=1e-6,
    )
    assert np.isclose(
        updated[1],
        0.10 + (1.0 - 0.10) * SPECIFIC_MATCH_BOOST,
        atol=1e-6,
    )
    assert updated[1] > updated[0]


def test_override_can_promote_a_skill_into_top_k():
    labels = ["python", "sql", "fastapi"]
    probabilities = np.array([0.99, 0.01, 0.02], dtype=np.float32)

    updated, matched = apply_token_matching_override(
        probabilities,
        labels,
        "SQL is explicitly required.",
    )

    ranked = sorted(zip(labels, updated), key=lambda item: -float(item[1]))
    assert matched == ["sql"]
    assert ranked[0][0] == "python"
    assert updated[1] > 0.01


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
