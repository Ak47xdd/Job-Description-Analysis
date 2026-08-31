"""Section-aware required/preferred skill classification for job descriptions."""

from __future__ import annotations

import re

from helpers import _skill_patterns


_REQUIRED_HEADING = re.compile(
    r"\b(?:required|requirements|required\s+skills|required\s+qualifications|"
    r"basic\s+qualifications|minimum\s+qualifications|must[- ]?have|essential|"
    r"what\s+you(?:'ll|\s+will)\s+need|qualifications)\b",
    re.I,
)
_PREFERRED_HEADING = re.compile(
    r"\b(?:preferred|preferred\s+skills|preferred\s+qualifications|bonus|"
    r"nice[- ]?to[- ]?have|nice\s+to\s+have|desirable|plus|good\s+to\s+have|"
    r"additional\s+qualifications|optional)\b",
    re.I,
)

# Common non-requirement headings used as hard section boundaries.
_BOUNDARY_HEADING = re.compile(
    r"\b(?:responsibilities|what\s+you(?:'ll|\s+will)\s+do|about\s+the\s+role|"
    r"about\s+us|benefits|compensation|salary|education|experience|"
    r"about\s+you|who\s+you\s+are|job\s+description|overview|description)\b",
    re.I,
)


def _normalize_heading(line: str) -> str:
    """Normalize markdown/bold/list decoration before heading matching."""
    value = line.strip()
    value = re.sub(r"^\s*#{1,6}\s*", "", value)
    value = re.sub(r"^\s*[-*•]\s+", "", value)
    value = re.sub(r"\*\*|__|`", "", value)
    value = re.sub(r"\s*[:：]\s*$", "", value)
    return re.sub(r"\s+", " ", value).strip()


def _is_heading(line: str) -> bool:
    raw = line.strip()
    if not raw:
        return False
    # Markdown headings are unambiguous.
    if re.match(r"^#{1,6}\s+", raw):
        return True
    normalized = _normalize_heading(raw)
    if _REQUIRED_HEADING.search(normalized) or _PREFERRED_HEADING.search(normalized):
        return True
    if _BOUNDARY_HEADING.search(normalized) and len(normalized) <= 100:
        return True
    # Bold standalone headings such as **Preferred (Bonus) Skills:**
    if re.match(r"^\s*\*\*[^*]+\*\*\s*:?[ \t]*$", raw):
        return True
    return False


def _find_sections(jd_text: str) -> tuple[str, str, bool, bool]:
    lines = (jd_text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    required: list[str] = []
    preferred: list[str] = []
    current: str | None = None
    has_required = False
    has_preferred = False

    for line in lines:
        heading = _normalize_heading(line)
        if _is_heading(line):
            if _PREFERRED_HEADING.search(heading):
                current = "preferred"
                has_preferred = True
                continue
            if _REQUIRED_HEADING.search(heading):
                current = "required"
                has_required = True
                continue
            if _BOUNDARY_HEADING.search(heading):
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
        if any(re.search(pattern, text, re.I) for pattern in _skill_patterns(skill)):
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
            # If only a Preferred section exists, everything explicitly outside
            # it remains eligible as required; model confidence is the fallback.
            required_names = {skill for skill, probability in predicted if probability >= 0.6}
        if not has_preferred:
            preferred_names = set()
    else:
        required_names = {skill for skill, probability in predicted if probability >= 0.6}
        preferred_names = set()

    # Required always wins if a skill is mentioned in both sections.
    preferred_names -= required_names

    required = [(skill, confidence[skill]) for skill, _ in predicted if skill in required_names]
    preferred = [(skill, confidence[skill]) for skill, _ in predicted if skill in preferred_names]
    return required, preferred
