import numpy as np

from api.JobAnalyze.v2.token_override import (
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


def test_override_sets_explicit_skills_to_one():
    labels = ["python", "sql", "fastapi", "git"]
    probabilities = np.array([0.91, 0.12, 0.03, 0.08], dtype=np.float32)

    updated, overridden = apply_token_matching_override(
        probabilities,
        labels,
        "Build FastAPI services, write SQL queries, and use Git.",
    )

    assert overridden == ["sql", "fastapi", "git"]
    assert updated.tolist() == [0.91, 1.0, 1.0, 1.0]


def test_override_can_promote_a_skill_into_top_k():
    labels = ["python", "sql", "fastapi"]
    probabilities = np.array([0.99, 0.01, 0.02], dtype=np.float32)

    updated, overridden = apply_token_matching_override(
        probabilities,
        labels,
        "SQL is explicitly required.",
    )

    ranked = sorted(zip(labels, updated), key=lambda item: -float(item[1]))
    assert overridden == ["sql"]
    assert ranked[0] == ("sql", 1.0)
