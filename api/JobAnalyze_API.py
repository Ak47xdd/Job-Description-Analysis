from fastapi import FastAPI, Depends, HTTPException, status, Header, Body
from slowapi import Limiter, _rate_limit_exceeded_handler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from supabase_auth.errors import AuthApiError
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.requests import Request
from fastapi_mcp import FastApiMCP
from google_oauth import google_authorize_url, verify_state, exchange_google_code
import traceback
import os
import hmac
import hashlib
import base64
import json
import time

from JobAnalyze.v1.pred_v1 import JobAnalyze_6k
from supabase_client import upsert_api_key_db
from helpers import _build_analysis
from schemas import JobOpeningCreate, NewsItemCreate, SignInRequest, SignUpRequest, ModelRequest
from auth import generate_api, API_KEY_DB, hash_key, verify, require_admin

ALLOWED_ORIGINS = ["https://jobselect.vercel.app", "https://job-analyzer-view.vercel.app", "http://localhost:5173", "http://localhost:3000"]
ALLOWED_HEADERS = ["Accept", "Content-Type", "Authorization", "JobAnalyze_6k_Key", "X-Admin-Secret"]
MAX_JD_LENGTH = 30000
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Unified JobAuto Model API", docs_url=None, redoc_url=None)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=True, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=ALLOWED_HEADERS)

@app.get("/")
async def main() -> dict:
    return {"message": "JobAnalyze 6k"}

@app.get("/cron", operation_id="cron_job")
async def cron() -> dict:
    return {"message": "Cron Task Executed"}

FRONTEND_URL = os.getenv("FRONTEND_URL", "https://jobselect.vercel.app").rstrip("/")


def _oauth_bridge_secret() -> bytes:
    secret = os.getenv("GOOGLE_OAUTH_STATE_SECRET") or os.getenv("SUPA_KEY")
    if not secret:
        raise RuntimeError("GOOGLE_OAUTH_STATE_SECRET or SUPA_KEY must be configured")
    return secret.encode("utf-8")


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64url_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _create_oauth_bridge(*, email: str, name: str) -> str:
    """Create a stateless, signed OAuth hand-off code.

    The previous implementation stored the hand-off in process-local memory.
    That can fail on Render when callback and exchange hit different workers.
    This code contains no API key; the key is looked up server-side.
    """
    now = int(time.time())
    payload = {
        "email": email,
        "name": name,
        "iat": now,
        "exp": now + 120,
        "nonce": _b64url_encode(os.urandom(18)),
    }
    encoded = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(_oauth_bridge_secret(), encoded.encode("ascii"), hashlib.sha256).digest()
    return f"{encoded}.{_b64url_encode(signature)}"


def _consume_oauth_bridge(code: str) -> dict | None:
    try:
        encoded, signature = code.split(".", 1)
        expected = hmac.new(_oauth_bridge_secret(), encoded.encode("ascii"), hashlib.sha256).digest()
        if not hmac.compare_digest(_b64url_decode(signature), expected):
            return None
        payload = json.loads(_b64url_decode(encoded).decode("utf-8"))
        now = int(time.time())
        if int(payload.get("exp", 0)) < now:
            return None
        if now - int(payload.get("iat", now)) > 120:
            return None
        email = str(payload.get("email", "")).strip().lower()
        name = str(payload.get("name", "")).strip()
        if not email or "@" not in email:
            return None
        return {"email": email, "name": name}
    except (ValueError, TypeError, KeyError, json.JSONDecodeError, UnicodeDecodeError):
        return None


@app.get("/auth/google", include_in_schema=False)
async def google_login():
    try:
        return RedirectResponse(url=google_authorize_url(), status_code=302)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

@app.get("/auth/google/callback", include_in_schema=False)
async def google_callback(code: str | None = None, state: str | None = None, error: str | None = None, error_description: str | None = None):
    if error:
        detail = (error_description or error).replace(" ", "_")
        return RedirectResponse(f"{FRONTEND_URL}/login?oauth_error=google_denied&detail={detail}")
    if not code or not state or not verify_state(state):
        return RedirectResponse(f"{FRONTEND_URL}/login?oauth_error=invalid_oauth_state")
    try:
        token_data = exchange_google_code(code)
        google_id_token = token_data.get("id_token")
        if not google_id_token:
            raise RuntimeError("Google did not return an ID token")

        from supabase_client import supabase, get_api_key_db

        # Supabase supports native Google ID-token sign-in. The Google provider
        # must be enabled in Supabase and configured with the same client ID.
        auth_result = supabase.auth.sign_in_with_id_token({
            "provider": "google",
            "token": google_id_token,
        })
        user = auth_result.user
        if user is None or not user.email:
            raise RuntimeError("Supabase did not return a Google user")

        email = str(user.email).strip().lower()
        metadata = user.user_metadata or {}
        name = str(metadata.get("name") or metadata.get("full_name") or email.split("@")[0]).strip()

        # Provision the API key before completing the browser hand-off. This
        # also repairs Google accounts that never received an api_tok row.
        record = get_api_key_db(owner=email, create_if_missing=True)
        if not record or not record.get("api_key"):
            raw = generate_api()
            record = upsert_api_key_db(owner=email, api_key=raw)
            if not record or not record.get("api_key"):
                raise RuntimeError("Unable to provision API key")

        bridge_code = _create_oauth_bridge(email=email, name=name)
        return RedirectResponse(f"{FRONTEND_URL}/auth/google/callback?code={bridge_code}", status_code=302)
    except AuthApiError as exc:
        traceback.print_exc()
        detail = str(exc).replace(" ", "_")
        return RedirectResponse(f"{FRONTEND_URL}/login?oauth_error=supabase_google_failed&detail={detail}")
    except Exception:
        traceback.print_exc()
        return RedirectResponse(f"{FRONTEND_URL}/login?oauth_error=google_failed")

@app.post("/auth/google/exchange", include_in_schema=False)
async def google_exchange(request: Request, code: str | None = None, body: dict | None = Body(default=None)) -> dict:
    """Exchange the browser hand-off code for the user's API credentials.

    Accept both ?code=... and JSON {\"code\": \"...\"} so existing
    frontends using either convention continue to work.
    """
    supplied_code = code
    if not supplied_code and isinstance(body, dict):
        supplied_code = body.get("code")
    if not supplied_code:
        raise HTTPException(status_code=422, detail="OAuth code is required")

    bridge = _consume_oauth_bridge(str(supplied_code))
    if not bridge:
        raise HTTPException(status_code=401, detail="OAuth code is invalid or expired")

    from supabase_client import get_api_key_db
    record = get_api_key_db(owner=bridge["email"], create_if_missing=True)
    if not record or not record.get("api_key"):
        raise HTTPException(status_code=500, detail="Unable to provision API key")

    return {
        "email": bridge["email"],
        "name": bridge["name"] or bridge["email"].split("@")[0],
        "api_key": record["api_key"],
    }

@app.post("/auth/create_acc", status_code=status.HTTP_201_CREATED, operation_id="sign_up")
async def create_acc(data: SignUpRequest) -> dict:
    from supabase_client import supabase
    email = str(data.email).strip().lower()
    name = data.name.strip()
    try:
        res = supabase.auth.sign_up({"email": email, "password": data.password, "options": {"data": {"name": name}}})
    except AuthApiError as e:
        msg = str(e).lower()
        if "already registered" in msg or "already exists" in msg:
            raise HTTPException(status_code=409, detail="An account with this email already exists. Please sign in.")
        if "password" in msg:
            raise HTTPException(status_code=422, detail="Password must be at least 6 characters.")
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
    except AuthApiError:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Unexpected sign-in error")
    if res.user is None:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    name = (res.user.user_metadata or {}).get("name", email.split("@")[0])
    record = get_api_key_db(owner=email, create_if_missing=True)
    if not record:
        raise HTTPException(status_code=500, detail="Unable to provision API key")
    return {"email": email, "name": name, "api_key": record["api_key"]}

@app.post("/API/Generate", status_code=status.HTTP_201_CREATED, operation_id="api_key_creator")
@limiter.limit("5/hour")
async def create_api(request: Request, email: str, x_admin_secret: str | None = Header(default=None, alias="X-Admin-Secret")) -> dict:
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

@app.post("/web_analyze", operation_id="web_analyze")
@limiter.limit("5/minute")
async def web_analyze(request: Request, data: ModelRequest) -> dict:
    if len(data.Job_Desc) > MAX_JD_LENGTH:
        raise HTTPException(status_code=413, detail="Job description is too large.")
    raw_predictions = JobAnalyze_6k(job_desc=data.Job_Desc, role=data.Role, job_type=data.Type)
    predicted = [(skill, float(score)) for skill, score in raw_predictions]
    try:
        return _build_analysis(predicted=predicted, role=data.Role, job_type=data.Type, jd_text=data.Job_Desc)
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Web Analyzer failed while building the analysis response.")

@app.post("/JobAnalyze_6k", operation_id="analyze_job_description")
@limiter.limit("10/minute")
async def JobAnalyze_Pred(request: Request, data: ModelRequest, api_client: dict = Depends(verify)) -> dict:
    if len(data.Job_Desc) > MAX_JD_LENGTH:
        raise HTTPException(status_code=413, detail="Job description is too large.")
    raw_predictions = JobAnalyze_6k(job_desc=data.Job_Desc, role=data.Role, job_type=data.Type)
    predicted = [(skill, float(score)) for skill, score in raw_predictions]
    analysis = _build_analysis(predicted=predicted, role=data.Role, job_type=data.Type, jd_text=data.Job_Desc)
    return {"answer": predicted, "analysis": analysis}

@app.post("/news", status_code=201, operation_id="create_news_item")
async def create_news_item(data: NewsItemCreate, api_client: dict = Depends(require_admin)) -> dict:
    from supabase_client import supabase
    res = supabase.table("news_items").insert({"title": data.title, "summary": data.summary, "category": data.category, "url": data.url, "body": data.body, "is_published": data.is_published}).execute()
    return res.data[0] if res.data else {}

@app.patch("/news/{item_id}/unpublish", operation_id="unpublish_news_item")
async def unpublish_news_item(item_id: str, api_client: dict = Depends(require_admin)) -> dict:
    from supabase_client import supabase
    supabase.table("news_items").update({"is_published": False}).eq("id", item_id).execute()
    return {"unpublished": True, "id": item_id}

@app.get("/news", operation_id="list_news_items")
async def list_news_items(include_drafts: bool = False, api_client: dict = Depends(require_admin)) -> dict:
    from supabase_client import supabase
    query = supabase.table("news_items").select("*").order("published_at", desc=True)
    if not include_drafts:
        query = query.eq("is_published", True)
    res = query.execute()
    return {"items": res.data or []}

@app.post("/careers/openings", status_code=201, operation_id="create_job_opening")
async def create_opening(data: JobOpeningCreate, api_client: dict = Depends(require_admin)) -> dict:
    from supabase_client import supabase
    res = supabase.table("job_openings").insert({"title": data.title, "department": data.department, "type": data.type, "location": data.location, "description": data.description, "requirements": data.requirements, "tags": data.tags, "is_open": True}).execute()
    return res.data[0] if res.data else {}

@app.patch("/careers/openings/{job_id}/close", operation_id="close_job_opening")
async def close_opening(job_id: str, api_client: dict = Depends(require_admin)) -> dict:
    from supabase_client import supabase
    supabase.table("job_openings").update({"is_open": False}).eq("id", job_id).execute()
    return {"closed": True, "id": job_id}

@app.get("/careers/applications", operation_id="list_applications")
async def list_applications(job_id: str | None = None, status: str | None = None, api_client: dict = Depends(require_admin)) -> dict:
    from supabase_client import supabase
    query = supabase.table("job_applications").select("id, job_id, job_title, name, email, linkedin_url, status, created_at")
    if job_id:
        query = query.eq("job_id", job_id)
    if status:
        query = query.eq("status", status)
    res = query.order("created_at", desc=True).execute()
    return {"applications": res.data or []}

mcp = FastApiMCP(app, name="JobAnalyze 6k", description="Analyzes a job description and returns ranked technical skills, skill categories, importance scores, complexity rating, experience requirements, summary, and recommendation. Provide Job_Desc (job description text), Role (AI Engineer or AI Developer), and Type (Internship, Junior, or Senior).", include_operations=["analyze_job_description"])
mcp.mount_http()
