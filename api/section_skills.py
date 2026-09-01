"""Section-aware required/preferred skill classification for job descriptions."""

from __future__ import annotations

import re

from helpers import _skill_patterns, SKILL_CATEGORIES


_REQUIRED_HEADING = re.compile(
    r"^(?:required(?:\s+(?:skills|qualifications|requirements))?|requirements|"
    r"required\s+skills|required\s+qualifications|basic\s+qualifications|"
    r"minimum\s+qualifications|must[- ]?have(?:\s+(?:skills|qualifications))?|"
    r"essential(?:\s+(?:skills|qualifications))?|what\s+you(?:'ll|\s+will)\s+need|"
    r"qualifications)$",
    re.I,
)
_PREFERRED_HEADING = re.compile(
    r"^(?:preferred(?:\s+(?:skills|qualifications))?|preferred\s+skills|"
    r"preferred\s+qualifications|bonus(?:\s+(?:skills|qualifications))?|"
    r"nice[- ]?to[- ]?have(?:\s+(?:skills|qualifications))?|desirable(?:\s+(?:skills|qualifications))?|"
    r"plus(?:\s+(?:skills|qualifications))?|good\s+to\s+have(?:\s+(?:skills|qualifications))?|"
    r"additional\s+qualifications|optional(?:\s+(?:skills|qualifications))?)$",
    re.I,
)
_BOUNDARY_HEADING = re.compile(
    r"^(?:responsibilities|key\s+responsibilities|what\s+you(?:'ll|\s+will)\s+do|"
    r"about\s+the\s+role|about\s+us|benefits|compensation|salary|education|experience|"
    r"about\s+you|who\s+you\s+are|job\s+description|overview|description|"
    r"company|about\s+the\s+company)$",
    re.I,
)


def _normalize_heading(line: str) -> str:
    value = line.strip()
    value = re.sub(r"^\s*#{1,6}\s*", "", value)
    value = re.sub(r"^\s*[-*•]\s+", "", value)
    value = re.sub(r"\*\*|__|`", "", value)
    value = re.sub(r"\s*[:：]\s*$", "", value)
    return re.sub(r"\s+", " ", value).strip()


def _heading_kind(line: str) -> str | None:
    raw = line.strip()
    if not raw:
        return None
    normalized = _normalize_heading(raw)
    if _PREFERRED_HEADING.fullmatch(normalized):
        return "preferred"
    if _REQUIRED_HEADING.fullmatch(normalized):
        return "required"
    if _BOUNDARY_HEADING.fullmatch(normalized):
        return "boundary"

    # Support real-world heading variants such as "Required (Must Have) Skills"
    # and "Preferred (Bonus) Skills" even when they are plain text headings.
    # Restrict this fallback to short heading-like lines so body sentences are
    # not accidentally interpreted as section boundaries.
    if len(normalized) <= 90 and len(normalized.split()) <= 12:
        if re.search(r"\b(?:preferred|bonus|nice[- ]?to[- ]?have|desirable|optional|plus)\b", normalized, re.I) and re.search(r"\b(?:skill|skills|qualification|qualifications|requirement|requirements|experience|competenc|have|nice)\b", normalized, re.I):
            return "preferred"
        if re.search(r"\b(?:required|requirements|must[- ]?have|essential|qualifications|mandatory)\b", normalized, re.I) and re.search(r"\b(?:skill|skills|qualification|qualifications|requirement|requirements|experience|competenc|have)\b", normalized, re.I):
            return "required"

    if re.match(r"^#{1,6}\s+", raw):
        if re.search(r"\b(?:preferred|bonus|nice[- ]?to[- ]?have|desirable|optional|plus)\b", normalized, re.I):
            return "preferred"
        if re.search(r"\b(?:required|requirements|must[- ]?have|essential|qualifications)\b", normalized, re.I):
            return "required"
        return "boundary"

    if re.match(r"^\s*\*\*[^*]+\*\*\s*[:：]?\s*$", raw):
        if re.search(r"\b(?:preferred|bonus|nice[- ]?to[- ]?have|desirable|optional|plus)\b", normalized, re.I):
            return "preferred"
        if re.search(r"\b(?:required|requirements|must[- ]?have|essential|qualifications)\b", normalized, re.I):
            return "required"
        return "boundary"
    return None


def _find_sections(jd_text: str) -> tuple[str, str, bool, bool]:
    lines = (jd_text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    required: list[str] = []
    preferred: list[str] = []
    current: str | None = None
    has_required = False
    has_preferred = False
    for line in lines:
        kind = _heading_kind(line)
        if kind == "preferred":
            current = "preferred"; has_preferred = True; continue
        if kind == "required":
            current = "required"; has_required = True; continue
        if kind == "boundary":
            current = None; continue
        if current == "required": required.append(line)
        elif current == "preferred": preferred.append(line)
    return "\n".join(required), "\n".join(preferred), has_required, has_preferred


def _skills_in_text(skills: list[str], text: str) -> set[str]:
    found: set[str] = set()
    for skill in skills:
        if any(re.search(pattern, text, re.I) for pattern in _skill_patterns(skill)):
            found.add(skill)
    return found


def _augment_explicit_section_skills(
    predicted: list[tuple[str, float]],
    required_text: str,
    preferred_text: str,
    has_required: bool,
    has_preferred: bool,
) -> list[tuple[str, float]]:
    """Add skills explicitly written in a classified JD section.

    The ML model remains the source of confidence when it predicted a skill.
    If the model omitted an explicit skill mention, give that deterministic
    section match a conservative 0.60 confidence floor so explicit Required
    skills cannot disappear merely because the model ranked them below top-k.
    """
    confidence = {skill: probability for skill, probability in predicted}
    catalog = [skill for skills in SKILL_CATEGORIES.values() for skill in skills]
    if has_required:
        required_mentions = _skills_in_text(catalog, required_text)
        for skill in required_mentions:
            confidence.setdefault(skill, 0.60)
    if has_preferred:
        preferred_mentions = _skills_in_text(catalog, preferred_text)
        for skill in preferred_mentions:
            confidence.setdefault(skill, 0.60)
    original_order = [skill for skill, _ in predicted]
    added = [skill for skill in confidence if skill not in original_order]
    return predicted + [(skill, confidence[skill]) for skill in sorted(added)]


def classify_required_preferred(predicted: list[tuple[str, float]], jd_text: str) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Classify skills by explicit JD section, with deterministic section matching.

    Explicit Required/Qualifications and Preferred/Bonus sections are authoritative.
    A skill found in both is treated as required. If an explicit section contains a
    catalog skill that the model omitted from its top-k output, it is added with a
    conservative 0.60 section-match confidence so explicit requirements are not lost.
    """
    if not predicted:
        predicted = []
    required_text, preferred_text, has_required, has_preferred = _find_sections(jd_text)
    predicted = _augment_explicit_section_skills(predicted, required_text, preferred_text, has_required, has_preferred)
    if not predicted:
        return [], []

    confidence = {skill: probability for skill, probability in predicted}
    skills = list(confidence)
    required_names = _skills_in_text(skills, required_text) if has_required else set()
    preferred_names = _skills_in_text(skills, preferred_text) if has_preferred else set()

    if has_required:
        required_names = required_names
        preferred_names = preferred_names - required_names
    elif has_preferred:
        required_names = {skill for skill, probability in predicted if probability >= 0.6} - preferred_names
    else:
        required_names = {skill for skill, probability in predicted if probability >= 0.6}
        preferred_names = set()

    preferred_names -= required_names
    required = [(skill, confidence[skill]) for skill, _ in predicted if skill in required_names]
    preferred = [(skill, confidence[skill]) for skill, _ in predicted if skill in preferred_names]
    return required, preferred
