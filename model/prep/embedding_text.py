"""Shared cleaning for SBERT job-description embeddings.

Keep this deterministic and lightweight so offline training and API inference
use exactly the same text policy.
"""
from __future__ import annotations

import re

PREPROCESSING_VERSION = "jd-focus-v1"
MAX_SEQ_LENGTH = 128

_BOILERPLATE_HEADINGS = re.compile(
    r"^\s*(?:"
    r"benefits(?:\s+and\s+perks)?|perks\s+and\s+benefits|"
    r"what\s+we\s+offer|our\s+benefits|employee\s+benefits|"
    r"compensation\s+and\s+benefits|company\s+culture|our\s+culture|"
    r"life\s+at\s+\w+|equal\s+opportunity(?:\s+employer)?|"
    r"diversity\s+and\s+inclusion|eeo\s+statement|"
    r"accommodation\s+statement|privacy\s+notice"
    r")\s*[:.!-]*\s*$",
    re.IGNORECASE,
)
_KEEP_HEADINGS = re.compile(
    r"^\s*(?:"
    r"job\s+description|about\s+the\s+role|role\s+overview|"
    r"responsibilities|what\s+you(?:'|’)ll\s+do|what\s+you\s+will\s+do|"
    r"requirements|minimum\s+qualifications|preferred\s+qualifications|"
    r"qualifications|must\s+have|nice\s+to\s+have|"
    r"required\s+skills|technical\s+skills|skills\s+and\s+experience|"
    r"experience|technology\s+stack|tech\s+stack|"
    r"what\s+you(?:'|’)ll\s+bring|what\s+you\s+will\s+bring"
    r")\s*[:.!-]*\s*$",
    re.IGNORECASE,
)
_BOILERPLATE_PHRASES = re.compile(
    r"(?:"
    r"competitive\s+(?:salary|compensation|pay|benefits)|"
    r"health,?\s+dental(?:,?\s+and\s+vision)?\s+insurance|"
    r"paid\s+time\s+off|\bPTO\b|401\s*\(?k\)?|"
    r"unlimited\s+vacation|free\s+(?:snacks|lunch|coffee)|"
    r"team[- ]building\s+events|work[- ]life\s+balance|"
    r"equal\s+opportunity\s+employer|without\s+regard\s+to\s+"
    r"(?:race|religion|gender|sexual\s+orientation|national\s+origin)|"
    r"reasonable\s+accommodations?\s+(?:will\s+be\s+)?provided"
    r")",
    re.IGNORECASE,
)
_TECHNICAL_SIGNAL = re.compile(
    r"\b(?:python|java|javascript|typescript|sql|postgres(?:ql)?|mysql|"
    r"pytorch|tensorflow|keras|scikit[- ]learn|machine learning|deep learning|"
    r"nlp|natural language processing|computer vision|llm|generative ai|"
    r"api|fastapi|django|flask|react|node\.js|docker|kubernetes|aws|azure|gcp|"
    r"git|github|data pipeline|model training|deployment|algorithm|"
    r"software|engineering|developer|programming|database|cloud|"
    r"analytics|data science|data analysis|requirements|responsibilities)\b",
    re.IGNORECASE,
)


def prepare_job_description(value: object) -> str:
    """Remove obvious non-technical boilerplate while preserving JD content."""
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)

    kept: list[str] = []
    in_boilerplate_section = False
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            if kept and kept[-1] != "":
                kept.append("")
            continue

        if _BOILERPLATE_HEADINGS.match(line):
            in_boilerplate_section = True
            continue
        if in_boilerplate_section:
            if _KEEP_HEADINGS.match(line):
                in_boilerplate_section = False
            else:
                # Do not let a short title-like line keep the parser trapped
                # forever when the next section uses an unrecognized heading.
                is_heading = len(line) <= 64 and not re.search(r"[.!?]$", line)
                if not is_heading:
                    continue
                in_boilerplate_section = False

        # Drop boilerplate-only lines. If a line also contains technical/job
        # content, retain it rather than accidentally deleting useful evidence.
        if _BOILERPLATE_PHRASES.search(line) and not _TECHNICAL_SIGNAL.search(line):
            continue
        kept.append(line)

    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
