"""Deterministic token matching for the SBERT hybrid classifier.

Embedding models handle semantic similarity well, but exact technology names
such as SQL, Git, and FastAPI should not depend on an embedding score.
This module provides a conservative lexical override: a vocabulary label is
forced to probability 1.0 only when that exact label occurs as an isolated
token/phrase in the raw job-description text.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np


_TOKEN_PREFIX = r"(?<!\w)"
_TOKEN_SUFFIX = r"(?!\w)"


@lru_cache(maxsize=512)
def _compiled_pattern(skill: str) -> re.Pattern[str]:
    escaped = re.escape(skill.strip())
    return re.compile(_TOKEN_PREFIX + escaped + _TOKEN_SUFFIX, re.IGNORECASE)


def skill_is_explicitly_present(skill: str, text: str) -> bool:
    """Return True when *skill* appears as an isolated, case-insensitive token."""
    if not skill or not text:
        return False
    return _compiled_pattern(skill).search(text) is not None


def apply_token_matching_override(
    probabilities: np.ndarray,
    label_vocab: list[str],
    text: str,
) -> tuple[np.ndarray, list[str]]:
    """Force explicitly mentioned vocabulary skills to probability 1.0.

    The returned list contains the labels overridden by the lexical matcher.
    Matching is performed against the complete vocabulary, so a skill can be
    added to the final top-k results even when the SBERT classifier ranked it
    below the cutoff.
    """
    scores = np.asarray(probabilities, dtype=np.float32).copy()
    if scores.ndim != 1:
        raise ValueError(f"Expected a 1-D probability vector, got {scores.shape}.")
    if scores.shape[0] != len(label_vocab):
        raise ValueError(
            f"Probability/vocabulary mismatch: {scores.shape[0]} scores for "
            f"{len(label_vocab)} labels."
        )

    overridden: list[str] = []
    for index, skill in enumerate(label_vocab):
        if skill_is_explicitly_present(skill, text):
            scores[index] = 1.0
            overridden.append(skill)
    return scores, overridden
