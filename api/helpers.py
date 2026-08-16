from datetime import datetime, timezone
import uuid
import re

SKILL_CATEGORIES: dict[str, list[str]] = {
    "Core Skills":            ["python", "sql", "ml", "nlp", "r",
                               "feature engineering", "model training",
                               "model evaluation", "statistics"],
    "Frameworks & Libraries": ["langchain", "langgraph", "tensorflow/pytorch",
                               "scikit-learn", "pandas", "numpy",
                               "hugging face", "llamaindex", "crewai",
                               "autogen", "n8n", "mlflow"],
    "Infrastructure":         ["docker", "kubernetes", "aws/azure", "ci/cd",
                               "git", "github", "mlops", "system design"],
    "AI / GenAI":             ["llms", "rag", "vectordb", "genai", "agents",
                               "mcp", "prompt engineering",
                               "anthropic /openai sdks", "openai"],
    "Web & Full Stack":       ["apis", "react", "javascript", "full stack",
                               "django", ".net", "c#", "c++", "java"],
}

_SKILL_TO_CAT: dict[str, str] = {
    skill: cat
    for cat, skills in SKILL_CATEGORIES.items()
    for skill in skills
}


_EXP_META: dict[str, dict] = {
    "Internship": {
        "level": "Internship",
        "years": "0 years",
        "education": "Bachelor's degree (in progress or completed)",
        "responsibilities": [
            {"title": "Support model experimentation and data analysis",   "icon": "chart"},
            {"title": "Assist in building and testing ML pipelines",       "icon": "database"},
            {"title": "Document findings and technical learnings",         "icon": "text"},
            {"title": "Collaborate with senior engineers on live projects", "icon": "users"},
        ],
    },
    "Junior": {
        "level": "Junior",
        "years": "0-2 years",
        "education": "Bachelor's degree or equivalent industry experience",
        "responsibilities": [
            {"title": "Build and maintain ML pipelines end to end",        "icon": "database"},
            {"title": "Deploy and monitor models in production",           "icon": "rocket"},
            {"title": "Write clean, tested, reviewable code",              "icon": "text"},
            {"title": "Contribute to architecture and design discussions",  "icon": "users"},
        ],
    },
    "Senior": {
        "level": "Senior",
        "years": "5+ years",
        "education": "Master's degree or equivalent industry experience",
        "responsibilities": [
            {"title": "Own the full model lifecycle in production",        "icon": "rocket"},
            {"title": "Design distributed training and optimise inference","icon": "chart"},
            {"title": "Maintain large-scale feature pipelines",           "icon": "database"},
            {"title": "Standardise experiment tracking and registry",      "icon": "text"},
            {"title": "Mentor engineers and set technical direction",      "icon": "users"},
        ],
    },
}


def _extract_company(jd_text: str) -> str:
    patterns = [
        r"\bat\s+([A-Z][A-Za-z0-9&\s]{2,30}?)(?:\s*[,.\n])",
        r"\bjoin\s+([A-Z][A-Za-z0-9&\s]{2,30}?)(?:\s*[,.\n])",
        r"([A-Z][A-Za-z0-9]+\s+(?:Inc|Labs|AI|Systems|Tech|Group|Corp|Ltd))\b",
    ]
    for p in patterns:
        m = re.search(p, jd_text)
        if m:
            return m.group(1).strip()
    return "Unknown"


def _get_complexity(present: list[tuple[str, float]]) -> str:
    cats_hit = {_SKILL_TO_CAT[s] for s, _ in present if s in _SKILL_TO_CAT}
    if len(cats_hit) >= 4: return "High"
    if len(cats_hit) >= 2: return "Medium"
    return "Low"


def _get_compatibility(required: list[tuple[str, float]]) -> tuple[int | None, str | None]:
    """
    Option B — model confidence proxy.
    Average confidence across required skills (prob >= 0.6), scaled 0-100.
    Measures how confidently the model identified skills in the JD, not
    how well a specific user matches. Correlates with role clarity and
    how standard the skill requirements are.
    """
    if not required:
        return None, None
    avg_conf = sum(p for _, p in required) / len(required)
    score    = int(avg_conf * 100)
    if score >= 80:   label = "Excellent Match"
    elif score >= 65: label = "Good Match"
    elif score >= 50: label = "Partial Match"
    else:             label = "Low Match"
    return score, label


def _build_categories(present: list[tuple[str, float]]) -> list[dict]:
    """Group predicted skills by UI category. status is always 'found'."""
    buckets: dict[str, list[dict]] = {cat: [] for cat in SKILL_CATEGORIES}
    for skill, prob in present:
        cat = _SKILL_TO_CAT.get(skill)
        if not cat:
            continue
        buckets[cat].append({
            "name":       skill.title(),
            "status":     "found",           # matches JSON schema exactly
            "importance": int(prob * 100),
        })
    return [
        {"name": cat, "skills": sorted(skills, key=lambda x: -x["importance"])}
        for cat, skills in buckets.items()
        if skills
    ]


def _build_summary(
    required: list[tuple[str, float]],
    role: str,
    job_type: str,
    jd_text: str,
) -> dict:
    """
    Derive a structured summary from predicted skills and inputs.
    Mirrors the schema: overview, responsibilities[], required[], preferred[].
    """
    top_required = [s.title() for s, _ in required[:5]]
    top_names  = ", ".join(s.title() for s, _ in required[:3])
    exp_label  = _EXP_META.get(job_type, _EXP_META["Junior"])["level"].lower()
    overview = (
        f"A {exp_label} {role} role with emphasis on {top_names} "
        f"and production-grade implementation across the full ML lifecycle."
    )

    cats_hit = {_SKILL_TO_CAT.get(s) for s, _ in required if _SKILL_TO_CAT.get(s)}
    responsibilities = []
    if "Core Skills" in cats_hit:
        responsibilities.append("Build, train, and evaluate machine learning models")
    if "AI / GenAI" in cats_hit:
        responsibilities.append("Design and deploy AI/GenAI pipelines")
    if "Infrastructure" in cats_hit:
        responsibilities.append("Operate and monitor models in production infrastructure")
    if "Frameworks & Libraries" in cats_hit:
        responsibilities.append("Implement solutions using modern ML frameworks and libraries")
    if "Web & Full Stack" in cats_hit:
        responsibilities.append("Develop APIs and backend services that serve model outputs")

    preferred = [
        s.title() for s, p in required
        if 0.45 <= p < 0.65
    ][:3]

    return {
        "overview":        overview,
        "responsibilities": responsibilities,
        "required":        top_required,
        "preferred":       preferred,
    }


def _build_recommendation(
    required: list[tuple[str, float]],
    job_type: str,
) -> dict:
    """
    Derive recommendation from skill count and seniority.
    No user skills needed — based purely on JD complexity signal.
    """
    req_count = len(required)
    if req_count >= 8:
        verdict = "Competitive Role"
        detail  = "High number of required skills — strong preparation recommended."
        points  = [
            {"type": "warning",  "text": f"{req_count} required skills identified"},
            {"type": "positive", "text": "Well-defined skill set makes targeted prep easier"},
        ]
    elif req_count >= 5:
        verdict = "Good Match"
        detail  = "Balanced requirements — achievable with solid fundamentals."
        points  = [
            {"type": "positive", "text": f"{req_count} required skills — manageable scope"},
            {"type": "positive", "text": "Role has clear technical expectations"},
        ]
    else:
        verdict = "Accessible Role"
        detail  = "Lower required skill count — strong entry point."
        points  = [
            {"type": "positive", "text": "Accessible technical bar"},
            {"type": "positive", "text": "Good opportunity for skill development"},
        ]

    if job_type == "Senior":
        points.append({"type": "warning", "text": "Senior seniority bar applies"})

    return {"verdict": verdict, "detail": detail, "points": points}


def _build_analysis(
    predicted: list[tuple[str, float]],
    role: str,
    job_type: str,
    jd_text: str,
) -> dict:
    """
    Build the full analysis object matching the /analyzer JSON schema exactly.
    """
    present  = [(s, p) for s, p in predicted if p >= 0.3]
    required = [(s, p) for s, p in predicted if p >= 0.6]

    compatibility, compatibility_label = _get_compatibility(required)

    return {
        "compatibility":      compatibility,
        "compatibilityLabel": compatibility_label,
        "complexity":          _get_complexity(present),
        "technicalSkillCount": len(present),
        "requiredTechCount":   len(required),
        "categories":  _build_categories(present),
        "importance": [
            {"name": s.title(), "value": int(p * 100)}
            for s, p in predicted[:7]
        ],
        "experienceRequirement": _EXP_META.get(job_type, _EXP_META["Junior"]),
        "summary": _build_summary(required, role, job_type, jd_text),
        "recommendation": _build_recommendation(required, job_type),
        "id":          f"an_{uuid.uuid4().hex[:8]}",
        "createdAt":   datetime.now(timezone.utc).isoformat(),
        "role":        role,
        "experience":  job_type,
        "detectedTitle": role,
        "company":     _extract_company(jd_text),
    }
