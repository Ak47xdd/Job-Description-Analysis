"""
JobAnalyze_API.py - Main API Script
/JobAnalyze_6k returns both 'answer' (backward compatible)
and 'analysis' matching the /analyzer page JSON schema.
"""

from fastapi import FastAPI, Depends, HTTPException, status
from slowapi import Limiter, _rate_limit_exceeded_handler
from fastapi.middleware.cors import CORSMiddleware
from supabase_auth.errors import AuthApiError
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.requests import Request
from fastapi_mcp import FastApiMCP
import traceback
import uvicorn

from JobAnalyze.v1.pred_v1 import JobAnalyze_6k
from supabase_client import upsert_api_key_db
from helpers import _build_analysis
from schemas import (
    JobOpeningCreate,
    NewsItemCreate,
    SignInRequest,
    SignUpRequest,
    ModelRequest
)
from auth import (
    generate_api,
    API_KEY_DB,
    hash_key,
    verify,
)   

ALLOWED_ORIGINS = [
    "https://jobselect.vercel.app",
    "https://job-analyzer-view.vercel.app",
    "http://localhost:5173",
    "http://localhost:3000",
]

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Unified JobAuto Model API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Cron and Root

@app.get("/")
async def main() -> dict:
    return {"message": "JobAnalyze 6k"}


@app.get("/cron", operation_id="cron_job")
async def cron() -> dict:
    return {"message": "Cron Task Executed"}

# User Routes

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

    record = get_api_key_db(owner=email, create_if_missing=True)
    if not record:
        raise HTTPException(status_code=500, detail="Unable to provision API key")

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

# Analyzer Endpoint Routes

@app.post("/web_analyze", operation_id="web_analyze")
@limiter.limit("5/minute")
async def web_analyze(request: Request, data: ModelRequest) -> dict:
    """
    Public Web Analyzer endpoint.

    No sign-in or API key is required. The model output is converted into the
    exact strict JSON schema used by the JobSelect /analyzer page.
    """
    raw_predictions = JobAnalyze_6k(
        job_desc=data.Job_Desc,
        role=data.Role,
        job_type=data.Type,
    )

    predicted = [
        (skill, float(score))
        for skill, score in raw_predictions
    ]

    try:
        return _build_analysis(
            predicted=predicted,
            role=data.Role,
            job_type=data.Type,
            jd_text=data.Job_Desc,
        )
    except Exception:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Web Analyzer failed while building the analysis response.",
        )

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
        "answer":   predicted,
        "analysis": analysis,
    }

# News Route

@app.post("/news", status_code=201, operation_id="create_news_item")
async def create_news_item(
    data: NewsItemCreate,
    api_client: dict = Depends(verify),
) -> dict:
    """
    Publish a new news item. Appears on the homepage newsboard immediately
    if is_published=True (default). Set is_published=False to save as draft.
    """
    from supabase_client import supabase
    res = supabase.table("news_items").insert({
        "title":        data.title,
        "summary":      data.summary,
        "category":     data.category,
        "url":          data.url,
        "body":         data.body,
        "is_published": data.is_published,
    }).execute()
    return res.data[0] if res.data else {}


@app.patch("/news/{item_id}/unpublish", operation_id="unpublish_news_item")
async def unpublish_news_item(
    item_id: str,
    api_client: dict = Depends(verify),
) -> dict:
    """
    Unpublish a news item. Removes it from the public homepage feed
    without deleting the record.
    """
    from supabase_client import supabase
    supabase.table("news_items").update(
        {"is_published": False}
    ).eq("id", item_id).execute()
    return {"unpublished": True, "id": item_id}


@app.get("/news", operation_id="list_news_items")
async def list_news_items(
    include_drafts: bool = False,
    api_client: dict = Depends(verify),
) -> dict:
    """
    List all news items. Pass include_drafts=true to see unpublished drafts.
    The public homepage fetches directly from Supabase (anon key),
    so this endpoint is for admin review only.
    """
    from supabase_client import supabase
    query = supabase.table("news_items").select("*").order(
        "published_at", desc=True
    )
    if not include_drafts:
        query = query.eq("is_published", True)
    res = query.execute()
    return {"items": res.data or []}

# Career Routes

@app.post("/careers/openings", status_code=201, operation_id="create_job_opening")
async def create_opening(
    data: JobOpeningCreate,
    api_client: dict = Depends(verify),   # requires valid API key
) -> dict:
    """Create a new job opening. Immediately visible on the /careers page."""
    from supabase_client import supabase
    res = supabase.table("job_openings").insert({
        "title":        data.title,
        "department":   data.department,
        "type":         data.type,
        "location":     data.location,
        "description":  data.description,
        "requirements": data.requirements,
        "tags":         data.tags,
        "is_open":      True,
    }).execute()
    return res.data[0] if res.data else {}

@app.patch("/careers/openings/{job_id}/close", operation_id="close_job_opening")
async def close_opening(
    job_id: str,
    api_client: dict = Depends(verify),
) -> dict:
    """Mark a role as closed. Removes it from the public /careers page."""
    from supabase_client import supabase
    res = supabase.table("job_openings").update(
        {"is_open": False}
    ).eq("id", job_id).execute()
    return {"closed": True, "id": job_id}


@app.get("/careers/applications", operation_id="list_applications")
async def list_applications(
    job_id: str | None = None,
    status: str | None = None,
    api_client: dict = Depends(verify),
) -> dict:
    """
    List all applications. Filter by job_id or status.
    Service-role only — applications are never exposed to anon.
    """
    from supabase_client import supabase
    query = supabase.table("job_applications").select(
        "id, job_id, job_title, name, email, linkedin_url, status, created_at"
    )
    if job_id: query = query.eq("job_id", job_id)
    if status:  query = query.eq("status", status)
    res = query.order("created_at", desc=True).execute()
    return {"applications": res.data or []}

# MCP

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

# Uvicorn

if __name__ == "__main__":
    uvicorn.run("JobAnalyze_API:app", host="0.0.0.0", port=5000)