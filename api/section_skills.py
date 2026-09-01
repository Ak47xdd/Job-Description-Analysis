"""Section-aware required/preferred skill classification for job descriptions."""

from __future__ import annotations

import re

from helpers import _skill_patterns


# Keep these anchored to the normalized heading rather than matching arbitrary
# words in body text. This prevents a sentence such as "preferred experience"
# from accidentally becoming a section boundary.
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

    # Preferred must be checked first because headings such as
    # "Preferred (Bonus) Skills" contain words that may also occur in broader
    # requirement terminology.
    if _PREFERRED_HEADING.fullmatch(normalized):
        return "preferred"
    if _REQUIRED_HEADING.fullmatch(normalized):
        return "required"
    if _BOUNDARY_HEADING.fullmatch(normalized):
        return "boundary"

    # Markdown headings are allowed to contain additional wording, e.g.
    # "Preferred (Bonus) Skills & Qualifications".
    if re.match(r"^#{1,6}\s+", raw):
        if re.search(r"\b(?:preferred|bonus|nice[- ]?to[- ]?have|desirable|optional|plus)\b", normalized, re.I):
            return "preferred"
        if re.search(r"\b(?:required|requirements|must[- ]?have|essential|qualifications)\b", normalized, re.I):
            return "required"
        return "boundary"

    # Bold standalone headings, including **Preferred (Bonus) Skills:**.
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
            current = "preferred"
            has_preferred = True
            continue
        if kind == "required":
            current = "required"
            has_required = True
            continue
        if kind == "boundary":
            current = None
            continue
        if current == "required":
            required.append(line)
        elif current == "preferred":
            preferred.append(line)

    return "\n".join(required), "\n".join(preferred), has_required, has_preferred


def _skills_in_text(skills: list[str], text: str) -> set[str]:
    found: set[str] = set()
    for skill in skills:
        patterns = _skill_patterns(skill)
        if any(re.search(pattern, text, re.I) for pattern in patterns):
            found.add(skill)
    return found


def classify_required_preferred(
    predicted: list[tuple[str, float]], jd_text: str
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Classify model predictions from the JD section where each skill occurs.

    Explicit Required/Qualifications and Preferred/Bonus sections are
    authoritative. A skill found in both is treated as required. When no
    explicit sections exist, retain the model-confidence fallback.
    """
    if not predicted:
        return [], []

    required_text, preferred_text, has_required, has_preferred = _find_sections(jd_text)
    confidence = {skill: probability for skill, probability in predicted}
    skills = list(confidence)

    required_names = _skills_in_text(skills, required_text) if has_required else set()
    preferred_names = _skills_in_text(skills, preferred_text) if has_preferred else set()

    if has_required or has_preferred:
        if not has_required:
            # With only a Preferred section, preserve the existing conservative
            # confidence fallback for skills not explicitly marked as bonus.
            required_names = {skill for skill, probability in predicted if probability >= 0.6}
        if not has_preferred:
            preferred_names = set()
    else:
        required_names = {skill for skill, probability in predicted if probability >= 0.6}
        preferred_names = set()

    # Required wins if a skill appears in both sections.
    preferred_names -= required_names

    required = [(skill, confidence[skill]) for skill, _ in predicted if skill in required_names]
    preferred = [(skill, confidence[skill]) for skill, _ in predicted if skill in preferred_names]
    return required, preferred
