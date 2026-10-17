"""Deterministic exact-keyword layer for the SBERT hybrid classifier.

Named technologies and tools (for example Git, FastAPI, JMeter, and Figma)
are lexical facts when explicitly named in a JD. Regex detects those facts;
SBERT remains responsible for context-heavy concepts such as system design,
agile development, and infrastructure operations.
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

_TOKEN_PREFIX = r"(?<!\w)"
_TOKEN_SUFFIX = r"(?!\w)"

# These concepts depend on contextual evidence and should remain semantic-only.
SEMANTIC_ONLY_LABELS = frozenset({
    "system design",
    "agile development",
    "infrastructure operations",
    "infrastructure",
    "software architecture",
    "problem solving",
    "communication",
    "leadership",
    "teamwork",
    "engineering",
    "development",
    "technology",
    "software",
    "data",
    "cloud",
    "backend",
    "ai/ml",
    "ai",
    "apis",
    "api",
})

# Explicit spellings that should map to an existing canonical output label.
# This map is deliberately conservative; it does not infer a tool from a
# related ecosystem term (e.g. GitHub does not count as the exact token Git).
EXACT_KEYWORD_ALIASES = {
    "fast api": "fastapi",
    "j meter": "jmeter",
    "figma design": "figma",
    "git version control": "git",
    "system design": "system design",
    "agile development": "agile development",
    "infrastructure operations": "infrastructure operations",
}

EXACT_MATCH_MARGIN = 0.02


@lru_cache(maxsize=1024)
def _compiled_pattern(skill: str) -> re.Pattern[str]:
    escaped = re.escape(skill.strip())
    return re.compile(_TOKEN_PREFIX + escaped + _TOKEN_SUFFIX, re.IGNORECASE)


def skill_is_explicitly_present(skill: str, text: str) -> bool:
    """Return True only for an isolated, case-insensitive exact token/phrase."""
    if not skill or not text:
        return False
    return _compiled_pattern(skill).search(text) is not None


def detect_exact_keyword_matches(
    label_vocab: list[str],
    text: str,
) -> list[str]:
    """Find explicitly named vocabulary skills, excluding semantic-only labels."""
    if not text:
        return []
    label_lookup = {str(label).strip().lower(): str(label) for label in label_vocab}
    matches: set[str] = set()

    for label in label_vocab:
        normalized_label = str(label).strip().lower()
        if not normalized_label or normalized_label in SEMANTIC_ONLY_LABELS:
            continue
        if skill_is_explicitly_present(normalized_label, text):
            matches.add(str(label))

    for alias, target in EXACT_KEYWORD_ALIASES.items():
        target_label = label_lookup.get(target)
        if target_label and target.lower() not in SEMANTIC_ONLY_LABELS:
            if skill_is_explicitly_present(alias, text):
                matches.add(target_label)

    return sorted(matches, key=str.lower)


def apply_token_matching_override(
    probabilities: np.ndarray,
    label_vocab: list[str],
    text: str,
    thresholds: np.ndarray | list[float] | None = None,
    margin: float = EXACT_MATCH_MARGIN,
) -> tuple[np.ndarray, list[str]]:
    """Promote exact lexical hits above their own calibrated threshold.

    Unlike a confidence boost, an exact match is deterministic evidence: the
    matching output is made eligible at its per-label threshold. Non-matching
    labels and semantic-only labels are left untouched.
    """
    scores = np.asarray(probabilities, dtype=np.float32).copy()
    if scores.ndim != 1:
        raise ValueError(f"Expected a 1-D probability vector, got {scores.shape}.")
    if scores.shape[0] != len(label_vocab):
        raise ValueError(
            f"Probability/vocabulary mismatch: {scores.shape[0]} scores for "
            f"{len(label_vocab)} labels."
        )
    if margin < 0:
        raise ValueError("margin must be non-negative.")

    threshold_values = None if thresholds is None else np.asarray(thresholds, dtype=np.float32)
    if threshold_values is not None and threshold_values.shape != scores.shape:
        raise ValueError(
            f"Threshold/vocabulary mismatch: {threshold_values.shape} vs {scores.shape}."
        )

    matches = detect_exact_keyword_matches(label_vocab, text)
    index_by_label = {str(label): index for index, label in enumerate(label_vocab)}
    for label in matches:
        index = index_by_label[label]
        if threshold_values is None:
            scores[index] = max(float(scores[index]), 0.90)
        else:
            scores[index] = max(
                float(scores[index]),
                min(1.0, float(threshold_values[index]) + margin),
            )
    return scores, matches
