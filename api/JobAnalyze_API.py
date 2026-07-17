"""
JobAnalyze_API.py - Main API Script
Currently Supports:
    - API Creation Endpoint
    - API Authenticator
    - API Encryptor
    - JobAnalyze 6k model Endpoint
    - Auth: Sign Up / Sign In via Supabase
"""
 
from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, field_validator, EmailStr
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request
import hashlib
import secrets
import uvicorn
 
from pred import JobAnalyze_6k
from supabase_client import upsert_api_key_db
 
app = FastAPI(title="Unified JobAuto Model API")
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://job-analyzer-view.vercel.app",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
 
API_KEY_NAME = "JobAnalyze_6k_Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
API_KEY_DB = {}  # in-memory cache
 
 
def key_func(request: Request) -> str:
    api_key = request.headers.get(API_KEY_NAME)
    return api_key if api_key else get_remote_address(request)
 
limiter = Limiter(key_func=key_func)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
 
 
# ── Auth models ──────────────────────────────────────────────────────────────
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
 
 
# ── API key verification ──────────────────────────────────────────────────────
async def verify(api_key: str = Security(api_key_header)):
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key Missing From Header",
        )
 
    hash_income = hash_key(api_key)
    db_record = API_KEY_DB.get(hash_income)
 
    if not db_record:
        try:
            from supabase_client import get_api_key_db
            db_record = get_api_key_db(api_key=api_key)
            if db_record and isinstance(db_record, dict):
                API_KEY_DB[hash_income] = db_record
        except Exception:
            db_record = None
 
    if not db_record:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or Expired API Key, Please get an API Key",
        )
 
    return db_record
 
 
# ── Inference model ───────────────────────────────────────────────────────────
class ModelRequest(BaseModel):
    Job_Desc: str
    Role: str
    Type: str
 
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
 
 
@app.get("/cron")
async def cron() -> dict:
    return {"message": "Cron Task Executed"}
 
 
@app.post("/auth/create_acc", status_code=status.HTTP_201_CREATED)
async def create_acc(data: SignUpRequest) -> dict:
    from supabase_client import supabase
 
    email = str(data.email).strip().lower()
    name  = data.name.strip()
 
    try:
        res = supabase.auth.sign_up({
            "email": email,
            "password": data.password,
            "options": {"data": {"name": name}},
        })
    except Exception as e:
        print(f"Signup error: {repr(e)}")
        raise HTTPException(status_code=400, detail="Account creation failed")
 
    if res.user is None:
        raise HTTPException(status_code=400, detail="Account not created")
 
    raw    = generate_api()
    hashed = hash_key(raw)
 
    upsert_api_key_db(user_id=hashed, owner=email, api_key=raw)
 
    return {"message": "Account Created", "api_key": raw, "name": name, "email": email}
 
 
@app.post("/auth/sign_in")
async def sign_in(data: SignInRequest) -> dict:
    from supabase_client import supabase, get_api_key_db
 
    email = str(data.email).strip().lower()
 
    try:
        res = supabase.auth.sign_in_with_password({
            "email": email,
            "password": data.password,
        })
    except Exception as e:
        print(f"Sign-in error: {repr(e)}")
        raise HTTPException(status_code=401, detail="Invalid email or password")
 
    if res.user is None:
        raise HTTPException(status_code=401, detail="Invalid email or password")
 
    # Pull name from Supabase user_metadata (set during sign_up)
    name = (res.user.user_metadata or {}).get("name", email.split("@")[0])
 
    record = get_api_key_db(owner=email)
    if not record:
        raise HTTPException(status_code=404, detail="API Key does not exist")
 
    return {"email": email, "name": name, "api_key": record["api_key"]}
 
 
@app.post("/API/Generate", status_code=status.HTTP_201_CREATED)
@limiter.limit("5/hour")
async def create_api(request: Request, email: str) -> dict:
    raw    = generate_api()
    hashed = hash_key(raw)
 
    API_KEY_DB[hashed] = {"owner": email}
    upsert_api_key_db(user_id=hashed, owner=email, api_key=raw)
 
    return {
        "owner": email,
        "api_key": raw,
        "warning": "Copy this key, this is a one time displayed key",
    }
 
 
@app.post("/JobAnalyze_6k")
@limiter.limit("10/minute")
async def JobAnalyze_Pred(
    request: Request,
    data: ModelRequest,
    api_client: dict = Depends(verify),
) -> dict:
    resp = JobAnalyze_6k(
        job_desc=data.Job_Desc,
        role=data.Role,
        job_type=data.Type,
    )
    return {"answer": [(skill, float(score)) for skill, score in resp]}
 
 
if __name__ == "__main__":
    uvicorn.run("JobAnalyze_API:app", host="0.0.0.0", port=5000)