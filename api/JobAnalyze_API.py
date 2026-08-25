from fastapi import FastAPI, HTTPException, status, Header
from slowapi import _rate_limit_exceeded_handler
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request
from fastapi_mcp import FastApiMCP
import hmac
import os

from auth import generate_api, API_KEY_DB, hash_key
from routes import users, analyzer, careers, news
from supabase_client import upsert_api_key_db
from rate_limit import limiter

ALLOWED_ORIGINS = [
    "https://jobselect.vercel.app",
]

ALLOWED_HEADERS = [
    "Accept",
    "Content-Type",
    "Authorization",
    "JobAnalyze_6k_Key",
    "X-Admin-Secret"
]

app = FastAPI(
    title="Unified JobAuto Model API",
    docs_url=None,
    redoc_url=None
)

app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=ALLOWED_HEADERS
)

app.include_router(users.router)
app.include_router(analyzer.router)
app.include_router(news.router)
app.include_router(careers.router)

@app.get("/")
async def main() -> dict:
    return {"message": "JobAnalyze 6k"}

@app.get("/cron", operation_id="cron_job")
async def cron() -> dict:
    return {"message": "Cron Task Executed"}

@app.post(
    "/API/Generate",
    status_code=status.HTTP_201_CREATED,
    operation_id="api_key_creator"
)
@limiter.limit("5/hour")
async def create_api(
    request: Request,
    email: str,
    x_admin_secret: str | None = Header(default=None, alias="X-Admin-Secret")
) -> dict:
    configured_secret = os.getenv("ADMIN_SECRET")
    if not configured_secret:
        raise HTTPException(status_code=503, detail="API key generation is not configured.")
    if not x_admin_secret or not hmac.compare_digest(x_admin_secret, configured_secret):
        raise HTTPException(status_code=403, detail="Administrator authorization required.")
    email = email.strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=422, detail="A valid email address is required.")
    raw = generate_api()
    API_KEY_DB[hash_key(raw)] = {"owner": email}
    upsert_api_key_db(user_id=hash_key(raw), owner=email, api_key=raw)
    return {"owner": email, "api_key": raw, "warning": "Copy this key, this is a one time displayed key"}

mcp = FastApiMCP(
    app,
    name="JobAnalyze 6k",
    description="Analyzes a job description and returns ranked technical skills, skill categories, importance scores, complexity rating, experience requirements, summary, and recommendation. Provide Job_Desc (job description text), Role (AI Engineer or AI Developer), and Type (Internship, Junior, or Senior).",
    include_operations=["analyze_job_description"]
)

mcp.mount_http()
