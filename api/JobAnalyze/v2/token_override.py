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

def select_regex_fallback_labels(
    label_vocab: list[str],
    threshold_payload: dict,
    rare_support_cutoff: int = 5,
) -> list[str]:
    """Select labels recoverable by exact regex when SBERT calibration fails.

    Include labels explicitly disabled by threshold optimization, and labels
    with very low dataset support whose measured best F1 is zero. This changes
    no model threshold: a selected label still requires an exact lexical hit.
    """
    if rare_support_cutoff < 0:
        raise ValueError("rare_support_cutoff must be non-negative.")

    entries = threshold_payload.get("labels", {})
    dataset_support = threshold_payload.get("dataset_support", {})
    if not isinstance(entries, dict) or not isinstance(dataset_support, dict):
        return []

    selected = []
    for label in label_vocab:
        entry = entries.get(label)
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status", "")).strip().lower()
        try:
            support = int(dataset_support.get(label, 0))
            best_f1 = float(entry.get("best_f1", 0.0))
        except (TypeError, ValueError):
            continue

        if status == "disabled" or (support <= rare_support_cutoff and best_f1 <= 0.0):
            if label.strip().lower() not in SEMANTIC_ONLY_LABELS:
                selected.append(label)

    return selected


def detect_regex_fallback_matches(
    fallback_labels: list[str],
    text: str,
) -> list[str]:
    """Return exact regex hits only among the labels selected for fallback."""
    if not text or not fallback_labels:
        return []
    return detect_exact_keyword_matches(fallback_labels, text)



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
    matched_labels: list[str] | None = None,
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

    matches = matched_labels if matched_labels is not None else detect_exact_keyword_matches(label_vocab, text)
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
