from datetime import datetime, timezone
import uuid
import re

SKILL_CATEGORIES: dict[str, list[str]] = {
    "Core Skills": ["python", "sql", "ml", "nlp", "r", "feature engineering", "model training", "model evaluation", "statistics"],
    "Frameworks & Libraries": ["langchain", "langgraph", "tensorflow/pytorch", "scikit-learn", "pandas", "numpy", "hugging face", "llamaindex", "crewai", "autogen", "n8n", "mlflow"],
    "Infrastructure": ["docker", "kubernetes", "aws/azure", "ci/cd", "git", "github", "mlops", "system design"],
    "AI / GenAI": ["llms", "rag", "vectordb", "genai", "agents", "mcp", "prompt engineering", "anthropic /openai sdks", "openai"],
    "Web & Full Stack": ["apis", "react", "javascript", "full stack", "django", ".net", "c#", "c++", "java"],
}

_SKILL_TO_CAT = {skill: cat for cat, skills in SKILL_CATEGORIES.items() for skill in skills}

_EXP_META = {
    "Internship": {"level": "Internship", "years": "0 years", "education": "Bachelor's degree (in progress or completed)", "responsibilities": [{"title": "Support model experimentation and data analysis", "icon": "chart"}, {"title": "Assist in building and testing ML pipelines", "icon": "database"}, {"title": "Document findings and technical learnings", "icon": "text"}, {"title": "Collaborate with senior engineers on live projects", "icon": "users"}]},
    "Junior": {"level": "Junior", "years": "0-2 years", "education": "Bachelor's degree or equivalent industry experience", "responsibilities": [{"title": "Build and maintain ML pipelines end to end", "icon": "database"}, {"title": "Deploy and monitor models in production", "icon": "rocket"}, {"title": "Write clean, tested, reviewable code", "icon": "text"}, {"title": "Contribute to architecture and design discussions", "icon": "users"}]},
    "Senior": {"level": "Senior", "years": "5+ years", "education": "Master's degree or equivalent industry experience", "responsibilities": [{"title": "Own the full model lifecycle in production", "icon": "rocket"}, {"title": "Design distributed training and optimise inference", "icon": "chart"}, {"title": "Maintain large-scale feature pipelines", "icon": "database"}, {"title": "Standardise experiment tracking and registry", "icon": "text"}, {"title": "Mentor engineers and set technical direction", "icon": "users"}]},
}

_TITLE_WORDS = r"engineer|developer|scientist|analyst|architect|manager|specialist|consultant|researcher|intern|designer|administrator|lead|director"
_COMPANY_NOISE = {skill.lower() for skill in _SKILL_TO_CAT} | {
    "the", "our", "this", "a", "an", "we", "you", "your", "team", "role", "position",
    "job", "company", "candidate", "developer", "engineer", "scientist", "analyst",
}


def _clean_extracted_text(value: str) -> str | None:
    value = re.sub(r"\*\*|__|`", "", value)
    value = re.sub(r"\s+", " ", value).strip(" \t\r\n-–—:|,.;")
    return value or None


def _looks_like_company(candidate: str | None) -> bool:
    if not candidate:
        return False
    normalized = candidate.strip().lower()
    if normalized in _COMPANY_NOISE:
        return False
    tokens = [token for token in re.split(r"\s+", normalized) if token]
    if not tokens or len(candidate) > 100:
        return False
    if any(token in _COMPANY_NOISE for token in tokens):
        return False
    if re.search(r"\b(?:git|github|docker|python|sql|aws|azure|kubernetes|tensorflow|pytorch|react|javascript|mlops|nlp|openai|llm|rag|api|mcp)\b", normalized):
        return False
    return bool(re.search(r"[A-Za-z]", candidate))


def _extract_company(jd_text: str) -> str | None:
    """Extract a company only when the JD contains a strong organization-name cue.

    Skill/tool keywords are explicitly rejected. If no high-confidence company
    pattern is present, returning None is preferable to fabricating a company.
    """
    text = jd_text or ""
    patterns = [
        r"\b(?:at|join|joining|from|with)\s+([A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*){0,5})(?=\s*(?:[,.;!?\n]|\b(?:is|are|seeks|seeking|looking|hiring|has|offers|for)\b))",
        r"\b([A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*){0,4})\s+(?:Inc\.?|LLC|Ltd\.?|Limited|Corp\.?|Corporation|Company|Labs?|Technologies|Systems|Solutions|Group)\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            candidate = _clean_extracted_text(match.group(1))
            if _looks_like_company(candidate):
                return candidate
    return None


def _extract_title(jd_text: str) -> str | None:
    text = jd_text or ""
    if not text.strip():
        return None
    for line in text.splitlines()[:8]:
        candidate = _clean_extracted_text(line)
        if candidate and len(candidate) <= 100 and re.search(rf"\b(?:{_TITLE_WORDS})\b", candidate, re.I):
            return candidate
    patterns = [
        rf"\b(?:looking for|seeking|hiring|need|hire)\s+(?:a|an|the)?\s*([A-Za-z][A-Za-z0-9&/+.\- ]{{2,80}}?\b(?:{_TITLE_WORDS}))\b",
        rf"\b(?:position|role|title)\s*[:\-]\s*([A-Za-z][A-Za-z0-9&/+.\- ]{{2,80}}?\b(?:{_TITLE_WORDS}))\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            candidate = _clean_extracted_text(match.group(1))
            if candidate:
                return candidate
    match = re.search(rf"\b([A-Za-z][A-Za-z0-9&/+.\- ]{{2,70}}\b(?:{_TITLE_WORDS}))\b", text, re.I)
    return _clean_extracted_text(match.group(1)) if match else None


_RESPONSIBILITY_HEADING = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:key\s+)?responsibilities\s*:?[ \t]*$|"
    r"^\s*(?:#{1,6}\s*)?(?:what\s+you(?:'|’)ll|what\s+you\s+will|"
    r"what\s+you\s+do|your\s+responsibilities|role\s+and\s+responsibilities)\s*:?[ \t]*$",
    re.I,
)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*)$")
_GENERIC_HEADING = re.compile(r"^\s*#{1,6}\s+.+$|^\s*[A-Z][A-Za-z0-9 /&'’\-]{1,70}:?\s*$")


def _is_section_heading(line: str) -> bool:
    s = line.strip()
    if not s or _BULLET.match(s):
        return False
    return bool(_GENERIC_HEADING.match(s))


def _extract_responsibilities(jd_text: str, limit: int = 5) -> tuple[list[str], bool]:
    lines = (jd_text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    start = None
    for i, line in enumerate(lines):
        if _RESPONSIBILITY_HEADING.match(line.strip()):
            start = i + 1
            break
    if start is None:
        return [], False
    bullets: list[str] = []
    current: str | None = None
    for line in lines[start:]:
        stripped = line.strip()
        if _is_section_heading(stripped) and not _RESPONSIBILITY_HEADING.match(stripped):
            break
        match = _BULLET.match(line)
        if match:
            if current:
                bullets.append(current)
            current = _clean_extracted_text(match.group(1))
        elif current and stripped:
            current = _clean_extracted_text(f"{current} {stripped}")
    if current:
        bullets.append(current)
    return [b for b in bullets if b][:limit], True


def _responsibility_icon(text: str) -> str:
    t = text.lower()
    if re.search(r"test|evaluat|validate|qa|quality", t): return "check"
    if re.search(r"collaborat|team|review|mentor|stakeholder", t): return "users"
    if re.search(r"prototyp|experiment|research", t): return "flask"
    if re.search(r"deploy|production|monitor|infrastructure", t): return "rocket"
    if re.search(r"data|analy|model|train", t): return "chart"
    if re.search(r"build|develop|implement|create", t): return "database"
    return "text"


def _get_complexity(present: list[tuple[str, float]]) -> str:
    cats_hit = {_SKILL_TO_CAT[s] for s, _ in present if s in _SKILL_TO_CAT}
    if len(cats_hit) >= 4: return "High"
    if len(cats_hit) >= 2: return "Medium"
    return "Low"


def _get_compatibility(required: list[tuple[str, float]]) -> tuple[int | None, str | None]:
    if not required:
        return None, None
    avg_conf = sum(p for _, p in required) / len(required)
    score = int(avg_conf * 100)
    if score >= 80: label = "Excellent Match"
    elif score >= 65: label = "Good Match"
    elif score >= 40: label = "Partial Match"
    else: label = "Weak Match"
    return score, label


def _build_categories(present: list[tuple[str, float]]) -> list[dict]:
    buckets = {cat: [] for cat in SKILL_CATEGORIES}
    for skill, prob in present:
        cat = _SKILL_TO_CAT.get(skill)
        if cat:
            buckets[cat].append({"name": skill.title(), "status": "found", "importance": int(prob * 100)})
    return [{"name": cat, "skills": sorted(skills, key=lambda x: -x["importance"])} for cat, skills in buckets.items() if skills]


_SECTION_HEADING = re.compile(r"(?im)^\s*(?:#+\s*)?(?P<heading>[^\n:]{2,100})\s*:?\s*$")
_REQUIRED_HEADING = re.compile(r"\b(?:required|requirements|required\s+skills|required\s+qualifications|basic\s+qualifications|minimum\s+qualifications|must[- ]have|essential|what\s+you(?:'ll|\s+will)\s+need)\b", re.I)
_PREFERRED_HEADING = re.compile(r"\b(?:preferred|preferred\s+skills|preferred\s+qualifications|bonus|nice[- ]to[- ]have|nice\s+to\s+have|desirable|plus|good\s+to\s+have|additional\s+qualifications)\b", re.I)
_SKILL_ALIASES: dict[str, tuple[str, ...]] = {
    "ml": (r"\bml\b", r"\bmachine[ -]learning\b"), "nlp": (r"\bnlp\b", r"\bnatural[ -]language[ -]processing\b"),
    "llms": (r"\bllms?\b", r"\blarge[ -]language[ -]models?\b"), "genai": (r"\bgenai\b", r"\bgenerative[ -]ai\b"),
    "vectordb": (r"\bvectordb\b", r"\bvector[ -]databases?\b"), "scikit-learn": (r"\bscikit[ -]?learn\b",),
    "tensorflow/pytorch": (r"\btensorflow\b", r"\bpytorch\b"), "hugging face": (r"\bhugging[ -]face\b",),
    "ci/cd": (r"\bci[ /-]?cd\b", r"\bcontinuous[ -](?:integration|delivery|deployment)\b"),
    "aws/azure": (r"\baws\b", r"\bazure\b"), "c++": (r"\bc\+\+\b",), "c#": (r"\bc#\b", r"\bcsharp\b"),
    ".net": (r"\b\.net\b", r"\bdotnet\b"), "apis": (r"\bapis?\b", r"\bapi\b"),
}


def _skill_patterns(skill: str) -> tuple[str, ...]:
    if skill in _SKILL_ALIASES:
        return _SKILL_ALIASES[skill]
    escaped = re.escape(skill).replace(r"\ ", r"\s+")
    return (rf"\b{escaped}\b",)


def _split_jd_sections(jd_text: str) -> tuple[str, str, bool, bool]:
    text = jd_text or ""
    if not text.strip(): return "", "", False, False
    matches = list(_SECTION_HEADING.finditer(text))
    required_chunks, preferred_chunks = [], []
    has_required = has_preferred = False
    for index, match in enumerate(matches):
        heading = match.group("heading").strip()
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_text = text[match.end():next_start]
        if _PREFERRED_HEADING.search(heading):
            preferred_chunks.append(section_text); has_preferred = True
        elif _REQUIRED_HEADING.search(heading):
            required_chunks.append(section_text); has_required = True
    return "\n".join(required_chunks), "\n".join(preferred_chunks), has_required, has_preferred


def _skills_in_text(skills: list[str], text: str) -> set[str]:
    found = set()
    for skill in skills:
        if any(re.search(pattern, text, re.I) for pattern in _skill_patterns(skill)):
            found.add(skill)
    return found


def _classify_required_preferred(predicted: list[tuple[str, float]], jd_text: str) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    if not predicted: return [], []
    required_text, preferred_text, has_required, has_preferred = _split_jd_sections(jd_text)
    skills = [s for s, _ in predicted]
    confidence = {s: p for s, p in predicted}
    required_mentions = _skills_in_text(skills, required_text) if has_required else set()
    preferred_mentions = _skills_in_text(skills, preferred_text) if has_preferred else set()
    if has_required:
        required_names = required_mentions
        preferred_names = preferred_mentions - required_mentions
    elif has_preferred:
        preferred_names = preferred_mentions
        required_names = {s for s, p in predicted if p >= 0.6} - preferred_names
    else:
        required_names = {s for s, p in predicted if p >= 0.6}
        preferred_names = set()
    required = [(s, confidence[s]) for s, _ in predicted if s in required_names]
    preferred = [(s, confidence[s]) for s, _ in predicted if s in preferred_names and s not in required_names]
    required_set = {s for s, _ in required}
    return required, [(s, p) for s, p in preferred if s not in required_set]


def _build_summary(predicted: list[tuple[str, float]], role: str, job_type: str, jd_text: str, responsibilities: list[str] | None = None) -> dict:
    required, preferred = _classify_required_preferred(predicted, jd_text)
    top_required = [s.title() for s, _ in required[:5]]
    top_names = ", ".join(s.title() for s, _ in required[:3]) or "the listed technical skills"
    exp_label = _EXP_META.get(job_type, _EXP_META["Junior"])["level"].lower()
    overview = f"A {exp_label} {role} role emphasizing {top_names}, based on the technical requirements identified in the job description."
    if responsibilities:
        overview += " Key responsibilities include " + "; ".join(r.rstrip(".") for r in responsibilities[:2]) + "."
    fallback = []
    cats_hit = {_SKILL_TO_CAT.get(s) for s, _ in required if _SKILL_TO_CAT.get(s)}
    if "Core Skills" in cats_hit: fallback.append("Build, train, and evaluate machine learning models")
    if "AI / GenAI" in cats_hit: fallback.append("Design and deploy AI/GenAI pipelines")
    if "Infrastructure" in cats_hit: fallback.append("Operate and monitor models in production infrastructure")
    if "Frameworks & Libraries" in cats_hit: fallback.append("Implement solutions using modern ML frameworks and libraries")
    if "Web & Full Stack" in cats_hit: fallback.append("Develop APIs and backend services that serve model outputs")
    return {"overview": overview, "responsibilities": responsibilities if responsibilities else fallback, "responsibilitiesSource": "extracted" if responsibilities else "template", "required": top_required, "preferred": [s.title() for s, _ in preferred[:3]]}


def _build_recommendation(compatibility: int | None, job_type: str) -> dict:
    """Derive recommendation from the compatibility score only.

    This keeps the verdict and displayed compatibility percentage consistent.
    """
    if compatibility is None:
        verdict = "Insufficient Data"
        detail = "No required technical skills were identified, so a compatibility verdict cannot be calculated."
        points = [{"type": "warning", "text": "No required technical skills identified"}]
    elif compatibility >= 65:
        verdict = "Good Match"
        detail = "Compatibility is 65% or higher based on the identified required skills."
        points = [{"type": "positive", "text": f"Compatibility score: {compatibility}%"}, {"type": "positive", "text": "Strong alignment with the identified required skills"}]
    elif compatibility >= 40:
        verdict = "Partial Match"
        detail = "Compatibility is between 40% and 64% based on the identified required skills."
        points = [{"type": "warning", "text": f"Compatibility score: {compatibility}%"}, {"type": "positive", "text": "Some required skills are aligned"}]
    else:
        verdict = "Weak Match"
        detail = "Compatibility is below 40% based on the identified required skills."
        points = [{"type": "warning", "text": f"Compatibility score: {compatibility}%"}, {"type": "warning", "text": "Several required skills may need development"}]
    if job_type == "Senior":
        points.append({"type": "warning", "text": "Senior seniority bar applies"})
    return {"verdict": verdict, "detail": detail, "points": points}


def _build_analysis(predicted: list[tuple[str, float]], role: str, job_type: str, jd_text: str) -> dict:
    present = [(s, p) for s, p in predicted if p >= 0.3]
    required, _preferred = _classify_required_preferred(predicted, jd_text)
    responsibilities, responsibilities_found = _extract_responsibilities(jd_text, limit=5)
    compatibility, compatibility_label = _get_compatibility(required)
    detected_title = _extract_title(jd_text)
    company = _extract_company(jd_text)
    summary = _build_summary(predicted, role, job_type, jd_text, responsibilities if responsibilities_found and responsibilities else None)
    exp_meta = dict(_EXP_META.get(job_type, _EXP_META["Junior"]))
    if responsibilities:
        exp_meta["responsibilities"] = [{"title": r, "icon": _responsibility_icon(r)} for r in responsibilities]
        exp_meta["responsibilitiesSource"] = "extracted"
    else:
        exp_meta["responsibilitiesSource"] = "template"
    return {
        "compatibility": compatibility, "compatibilityLabel": compatibility_label,
        "complexity": _get_complexity(present), "technicalSkillCount": len(present), "requiredTechCount": len(required),
        "categories": _build_categories(present),
        "importance": [{"name": s.title(), "value": int(p * 100)} for s, p in predicted[:7]],
        "experienceRequirement": exp_meta, "summary": summary,
        "recommendation": _build_recommendation(compatibility, job_type),
        "id": f"an_{uuid.uuid4().hex[:8]}", "createdAt": datetime.now(timezone.utc).isoformat(),
        "role": role, "experience": job_type, "detectedTitle": detected_title, "company": company,
    }
