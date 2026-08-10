"""
JobAnalyze_API.py - Main API Script
v0.12.0 — /JobAnalyze_6k returns both 'answer' (backward compatible)
and 'analysis' matching the /analyzer page JSON schema exactly.
"""

from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi_mcp import FastApiMCP
from pydantic import BaseModel, field_validator, EmailStr
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request
from supabase_auth.errors import AuthApiError
import hashlib
import secrets
import traceback
import uuid
import re
from datetime import datetime, timezone
import uvicorn

from JobAnalyze.v1.pred_v1 import JobAnalyze_6k
from supabase_client import upsert_api_key_db

ALLOWED_ORIGINS = [
    "https://jobselect.vercel.app",
    "https://job-analyzer-view.vercel.app",
    "http://localhost:5173",
    "http://localhost:3000",
]

app = FastAPI(title="Unified JobAuto Model API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

API_KEY_NAME   = "JobAnalyze_6k_Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
API_KEY_DB: dict = {}


def key_func(request: Request) -> str:
    api_key = request.headers.get(API_KEY_NAME)
    return api_key if api_key else get_remote_address(request)

limiter = Limiter(key_func=key_func)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── Skill category map ────────────────────────────────────────────────────────
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

# ── Experience requirement tables ─────────────────────────────────────────────
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
        "years": "0–2 years",
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


# ── Analyzer helpers ──────────────────────────────────────────────────────────
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
    # Top required skills for the 'required' list
    top_required = [s.title() for s, _ in required[:5]]

    # Infer a one-sentence overview from complexity and top skills
    top_names  = ", ".join(s.title() for s, _ in required[:3])
    exp_label  = _EXP_META.get(job_type, _EXP_META["Junior"])["level"].lower()
    overview = (
        f"A {exp_label} {role} role with emphasis on {top_names} "
        f"and production-grade implementation across the full ML lifecycle."
    )

    # Generic responsibilities anchored to actual predicted skill categories
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

        # Complexity and counts
        "complexity":          _get_complexity(present),
        "technicalSkillCount": len(present),
        "requiredTechCount":   len(required),

        # Skill breakdown — status is always "found"
        "categories":  _build_categories(present),
        "importance": [
            {"name": s.title(), "value": int(p * 100)}
            for s, p in predicted[:7]
        ],

        # Experience requirement — derived from Type input
        "experienceRequirement": _EXP_META.get(job_type, _EXP_META["Junior"]),

        # Summary — derived from predicted skills and JD
        "summary": _build_summary(required, role, job_type, jd_text),

        # Recommendation — derived from required skill count and seniority
        "recommendation": _build_recommendation(required, job_type),

        # Metadata
        "id":          f"an_{uuid.uuid4().hex[:8]}",
        "createdAt":   datetime.now(timezone.utc).isoformat(),
        "role":        role,
        "experience":  job_type,
        "detectedTitle": role,
        "company":     _extract_company(jd_text),
    }


# ── Auth models ───────────────────────────────────────────────────────────────
class SignUpRequest(BaseModel):
    name: str
    email: EmailStr
    password: str


class SignInRequest(BaseModel):
    email: EmailStr
    password: str


# ── Key helpers ───────────────────────────────────────────────────────────────
def generate_api(prefix: str = "ja6k") -> str:
    return f"{prefix}_{secrets.token_hex(32)}"


def hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


# ── Verify ────────────────────────────────────────────────────────────────────
async def verify(api_key: str = Security(api_key_header)):
    if not api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="API Key Missing From Header")
    hash_income = hash_key(api_key)
    db_record   = API_KEY_DB.get(hash_income)
    if not db_record:
        try:
            from supabase_client import get_api_key_db
            db_record = get_api_key_db(api_key=api_key)
            if db_record and isinstance(db_record, dict):
                API_KEY_DB[hash_income] = db_record
        except Exception:
            db_record = None
    if not db_record:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Invalid or Expired API Key")
    return db_record


# ── Inference model ───────────────────────────────────────────────────────────
class ModelRequest(BaseModel):
    Job_Desc: str
    Role:     str
    Type:     str

    @field_validator("Job_Desc")
    def jd_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Job_Desc cannot be empty")
        return v.strip()

    @field_validator("Role")
    def role_valid(cls, v):
        allowed = ["AI Engineer", "AI Developer"]
        if v not in allowed:
            raise ValueError(f"Role must be one of {allowed}")
        return v

    @field_validator("Type")
    def type_valid(cls, v):
        allowed = ["Internship", "Junior", "Senior"]
        if v not in allowed:
            raise ValueError(f"Type must be one of {allowed}")
        return v


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def main() -> dict:
    return {"message": "JobAnalyze 6k"}


@app.get("/cron", operation_id="cron_job")
async def cron() -> dict:
    return {"message": "Cron Task Executed"}


@app.post("/auth/create_acc", status_code=status.HTTP_201_CREATED,
          operation_id="sign_up")
async def create_acc(data: SignUpRequest) -> dict:
    from supabase_client import supabase
    email = str(data.email).strip().lower()
    name  = data.name.strip()
    try:
        res = supabase.auth.sign_up({
            "email": email, "password": data.password,
            "options": {"data": {"name": name}},
        })
    except AuthApiError as e:
        msg = str(e).lower()
        if "already registered" in msg or "already exists" in msg:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                detail="An account with this email already exists. Please sign in.")
        if "password" in msg:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Password must be at least 6 characters.")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Unexpected signup error")
    if res.user is None:
        raise HTTPException(status_code=400, detail="Signup failed")
    raw = generate_api()
    try:
        upsert_api_key_db(user_id=hash_key(raw), owner=email, api_key=raw)
    except Exception:
        traceback.print_exc()
    return {"message": "Account Created", "api_key": raw, "name": name, "email": email}


@app.post("/auth/sign_in", operation_id="sign_in")
async def sign_in(data: SignInRequest) -> dict:
    from supabase_client import supabase, get_api_key_db
    email = str(data.email).strip().lower()
    try:
        res = supabase.auth.sign_in_with_password({"email": email, "password": data.password})
    except AuthApiError as e:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Unexpected sign-in error")
    if res.user is None:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    name   = (res.user.user_metadata or {}).get("name", email.split("@")[0])
    record = get_api_key_db(owner=email)
    if not record:
        raise HTTPException(status_code=404, detail="API Key does not exist")
    return {"email": email, "name": name, "api_key": record["api_key"]}


@app.post("/API/Generate", status_code=status.HTTP_201_CREATED,
          operation_id="api_key_creator")
@limiter.limit("5/hour")
async def create_api(request: Request, email: str) -> dict:
    raw = generate_api()
    API_KEY_DB[hash_key(raw)] = {"owner": email}
    upsert_api_key_db(user_id=hash_key(raw), owner=email, api_key=raw)
    return {"owner": email, "api_key": raw,
            "warning": "Copy this key, this is a one time displayed key"}


@app.post("/web_analyze", operation_id="web_analyze")
@limiter.limit("5/minute")
async def web_analyze(request: Request, data: ModelRequest) -> dict:
    resp = JobAnalyze_6k(
        job_desc=data.Job_Desc,
        role=data.Role,
        job_type=data.Type,
    )
    return _build_analysis(data, resp)

@app.post("/JobAnalyze_6k", operation_id="analyze_job_description")
@limiter.limit("10/minute")
async def JobAnalyze_Pred(
    request: Request,
    data: ModelRequest,
    api_client: dict = Depends(verify),
) -> dict:
    raw_predictions = JobAnalyze_6k(
        job_desc=data.Job_Desc,
        role=data.Role,
        job_type=data.Type,
    )
    predicted = [(skill, float(score)) for skill, score in raw_predictions]

    analysis = _build_analysis(
        predicted=predicted,
        role=data.Role,
        job_type=data.Type,
        jd_text=data.Job_Desc,
    )

    return {
        "answer":   predicted,   # unchanged — CLI/MCP backward compatible
        "analysis": analysis,    # full schema match
    }


# ── MCP ───────────────────────────────────────────────────────────────────────
mcp = FastApiMCP(
    app,
    name="JobAnalyze 6k",
    description=(
        "Analyzes a job description and returns ranked technical skills, "
        "skill categories, importance scores, complexity rating, experience "
        "requirements, summary, and recommendation. "
        "Provide Job_Desc (job description text), Role (AI Engineer or AI Developer), "
        "and Type (Internship, Junior, or Senior)."
    ),
    include_operations=["analyze_job_description"],
)
mcp.mount_http()


if __name__ == "__main__":
    uvicorn.run("JobAnalyze_API:app", host="0.0.0.0", port=5000)