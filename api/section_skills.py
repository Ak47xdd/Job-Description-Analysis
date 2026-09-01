"""Section-aware required/preferred skill classification for job descriptions."""

from __future__ import annotations
import re
from helpers import _skill_patterns, SKILL_CATEGORIES

_REQUIRED_HEADING = re.compile(r"^(?:required(?:\s+(?:skills|qualifications|requirements))?|requirements|required\s+skills|required\s+qualifications|basic\s+qualifications|minimum\s+qualifications|must[- ]?have(?:\s+(?:skills|qualifications))?|essential(?:\s+(?:skills|qualifications))?|what\s+you(?:'ll|\s+will)\s+need|qualifications)(?:\s*\([^)]*\))?$", re.I)
_PREFERRED_HEADING = re.compile(r"^(?:preferred(?:\s+(?:skills|qualifications))?|preferred\s+skills|preferred\s+qualifications|bonus(?:\s+(?:skills|qualifications))?|nice[- ]?to[- ]?have(?:\s+(?:skills|qualifications))?|desirable(?:\s+(?:skills|qualifications))?|plus(?:\s+(?:skills|qualifications))?|good\s+to\s+have(?:\s+(?:skills|qualifications))?|additional\s+qualifications|optional(?:\s+(?:skills|qualifications))?)(?:\s*\([^)]*\))?$", re.I)
_BOUNDARY_HEADING = re.compile(r"^(?:responsibilities|key\s+responsibilities|what\s+you(?:'ll|\s+will)\s+do|about\s+the\s+role|about\s+us|benefits|compensation|salary|education|experience|about\s+you|who\s+you\s+are|job\s+description|overview|description|company|about\s+the\s+company)$", re.I)


def _normalize_heading(line: str) -> str:
    value = line.strip()
    value = re.sub(r"^\s*#{1,6}\s*", "", value)
    value = re.sub(r"^\s*[-*•]\s+", "", value)
    value = re.sub(r"^\s*(?:\*\*|__)+\s*|\s*(?:\*\*|__)+\s*$", "", value)
    value = re.sub(r"\s*[:：]\s*$", "", value)
    return re.sub(r"\s+", " ", value).strip()


def _heading_kind(line: str) -> str | None:
    raw = line.strip()
    if not raw:
        return None
    normalized = _normalize_heading(raw)
    if _PREFERRED_HEADING.fullmatch(normalized): return "preferred"
    if _REQUIRED_HEADING.fullmatch(normalized): return "required"
    if _BOUNDARY_HEADING.fullmatch(normalized): return "boundary"
    if len(normalized) <= 100 and len(normalized.split()) <= 14:
        if re.search(r"\b(?:preferred|bonus|nice[- ]?to[- ]?have|desirable|optional|plus)\b", normalized, re.I) and re.search(r"\b(?:skill|skills|qualification|qualifications|requirement|requirements|experience|competenc|have)\b", normalized, re.I): return "preferred"
        if re.search(r"\b(?:required|requirements|must[- ]?have|essential|qualifications|mandatory)\b", normalized, re.I) and re.search(r"\b(?:skill|skills|qualification|qualifications|requirement|requirements|experience|competenc|have)\b", normalized, re.I): return "required"
    return None


def _find_sections(jd_text: str) -> tuple[str, str, bool, bool]:
    lines = (jd_text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    required, preferred = [], []
    current = None
    has_required = has_preferred = False
    for line in lines:
        kind = _heading_kind(line)
        if kind == "preferred": current = "preferred"; has_preferred = True; continue
        if kind == "required": current = "required"; has_required = True; continue
        if kind == "boundary": current = None; continue
        if current == "required": required.append(line)
        elif current == "preferred": preferred.append(line)
    return "\n".join(required), "\n".join(preferred), has_required, has_preferred


def _skills_in_text(skills: list[str], text: str) -> set[str]:
    return {skill for skill in skills if any(re.search(pattern, text or "", re.I) for pattern in _skill_patterns(skill))}


def _catalog_skills() -> list[str]:
    return [skill for skills in SKILL_CATEGORIES.values() for skill in skills]


def _augment_explicit_section_skills(predicted, required_text, preferred_text, has_required, has_preferred):
    confidence = {skill: probability for skill, probability in (predicted or [])}
    catalog = _catalog_skills()
    for text, active in ((required_text, has_required), (preferred_text, has_preferred)):
        if active:
            for skill in _skills_in_text(catalog, text): confidence.setdefault(skill, 0.60)
    original = {skill for skill, _ in (predicted or [])}
    return (predicted or []) + [(skill, confidence[skill]) for skill in sorted(confidence) if skill not in original]


def classify_required_preferred(predicted, jd_text):
    """Classify skills from explicit Required/Preferred JD sections.

    Explicit section membership is authoritative. A catalog skill explicitly
    written in either section is retained even if the ML model omitted it.
    """
    required_text, preferred_text, has_required, has_preferred = _find_sections(jd_text)
    predicted = _augment_explicit_section_skills(predicted, required_text, preferred_text, has_required, has_preferred)
    if not predicted: return [], []
    confidence = {skill: probability for skill, probability in predicted}
    skills = list(confidence)
    required_names = _skills_in_text(skills, required_text) if has_required else set()
    preferred_names = _skills_in_text(skills, preferred_text) if has_preferred else set()
    if has_required:
        preferred_names -= required_names
    elif has_preferred:
        required_names = {skill for skill, probability in predicted if probability >= 0.6} - preferred_names
    else:
        required_names = {skill for skill, probability in predicted if probability >= 0.6}
        preferred_names = set()
    preferred_names -= required_names
    return (
        [(skill, confidence[skill]) for skill, _ in predicted if skill in required_names],
        [(skill, confidence[skill]) for skill, _ in predicted if skill in preferred_names],
    )
