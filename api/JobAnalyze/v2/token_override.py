"""Deterministic lexical support for the SBERT hybrid classifier.

SBERT handles semantic similarity well, while exact lexical matches can recover
technologies whose names are easy to miss in embeddings. Exact matches are
therefore used as a controlled confidence boost, not as a hard probability of
1.0. Generic taxonomy labels receive a smaller boost than specific technologies.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np


_TOKEN_PREFIX = r"(?<!\w)"
_TOKEN_SUFFIX = r"(?!\w)"

GENERIC_LABELS = frozenset(
    {
        "ai",
        "apis",
        "api",
        "backend",
        "cloud",
        "infrastructure",
        "search",
        "software",
        "technology",
        "data",
        "development",
        "engineering",
    }
)

SPECIFIC_MATCH_BOOST = 0.75
GENERIC_MATCH_BOOST = 0.25
MAX_LEXICAL_PROBABILITY = 0.95


@lru_cache(maxsize=512)
def _compiled_pattern(skill: str) -> re.Pattern[str]:
    escaped = re.escape(skill.strip())
    return re.compile(_TOKEN_PREFIX + escaped + _TOKEN_SUFFIX, re.IGNORECASE)


def skill_is_explicitly_present(skill: str, text: str) -> bool:
    """Return True when *skill* appears as an isolated, case-insensitive token."""
    if not skill or not text:
        return False
    return _compiled_pattern(skill).search(text) is not None


def _lexical_boost(skill: str) -> float:
    return GENERIC_MATCH_BOOST if skill.strip().lower() in GENERIC_LABELS else SPECIFIC_MATCH_BOOST


def apply_token_matching_override(
    probabilities: np.ndarray,
    label_vocab: list[str],
    text: str,
) -> tuple[np.ndarray, list[str]]:
    """Boost explicitly mentioned vocabulary skills without forcing 100%.

    The boost is applied as p' = p + (1 - p) * boost and capped at
    MAX_LEXICAL_PROBABILITY. Generic taxonomy labels use a smaller boost.
    """
    scores = np.asarray(probabilities, dtype=np.float32).copy()
    if scores.ndim != 1:
        raise ValueError(f"Expected a 1-D probability vector, got {scores.shape}.")
    if scores.shape[0] != len(label_vocab):
        raise ValueError(
            f"Probability/vocabulary mismatch: {scores.shape[0]} scores for "
            f"{len(label_vocab)} labels."
        )

    matched: list[str] = []
    for index, skill in enumerate(label_vocab):
        if skill_is_explicitly_present(skill, text):
            boost = _lexical_boost(skill)
            scores[index] = min(
                MAX_LEXICAL_PROBABILITY,
                scores[index] + (1.0 - scores[index]) * boost,
            )
            matched.append(skill)
    return scores, matched
