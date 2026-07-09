from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
import hashlib
import secrets
import uvicorn

from pred import JobAnalyze_6k

from pydantic import BaseModel, field_validator

from supabase_client import upsert_api_key_db

app = FastAPI(title="Unified JobAuto Model API")


API_KEY_NAME = "JobAnalyze_6k_Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
API_KEY_DB = {}


def generate_api(prefix: str = "ja6k") -> str:
    rand_tok = secrets.token_hex(32)
    return f"{prefix}_{rand_tok}"


def hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


async def verify(api_key: str = Security(api_key_header)):
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key Missing From Header",
        )

    hash_income = hash_key(api_key)

    # 1) Try in-memory cache first
    db_record = API_KEY_DB.get(hash_income)

    # 2) If not cached, query Supabase by api_key
    if not db_record:
        try:
            from supabase_client import get_api_key_db
            db_record = get_api_key_db(api_key=api_key)
            if db_record and isinstance(db_record, dict):
                API_KEY_DB[hash_income] = db_record
        except Exception:
            db_record = None

    # 3) Deny if still not found
    if not db_record:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or Expired API Key, Please get an API Key",
        )

    return db_record

class ModelRequest(BaseModel):
    Job_Desc: str
    Role: str
    Type: str
    
    @field_validator('Job_Desc')
    def jd_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Job_Desc cannot be empty')
        return v.strip()

    @field_validator('Role')
    def role_valid(cls, v):
        allowed = ['AI Engineer', 'AI Developer']
        if v not in allowed:
            raise ValueError(f'Role must be one of {allowed}')
        return v

    @field_validator('Type')
    def type_valid(cls, v):
        allowed = ['Internship', 'Junior', 'Senior']
        if v not in allowed:
            raise ValueError(f'Type must be one of {allowed}')
        return v


@app.get("/")
async def main() -> dict:
    return {"message": "JobAnalyze 6k"}

@app.get("/cron")
async def cron() -> dict:
    return {"message": "Cron Task Executed"}

@app.post("/API/Generate", status_code=status.HTTP_201_CREATED)
async def create_api(owner: str) -> dict:

    raw = generate_api()
    hashed = hash_key(raw)

    API_KEY_DB[hashed] = {
        "owner": owner,
    }

    upsert_api_key_db(user_id=hashed, owner=owner, api_key=raw)


    return {
        "owner": owner,
        "api_key": raw,
        "warning": "Copy this key, this is a one time displayed key",
    }

@app.post("/JobAnalyze_6k")
async def JobAnalyze_Pred(data: ModelRequest, api_client: dict = Depends(verify)) -> dict:
    jd = data.Job_Desc
    role = data.Role
    job_type = data.Type

    resp = JobAnalyze_6k(job_desc=jd.strip(), role=role, job_type=job_type)

    resp_json = [(skill, float(score)) for skill, score in resp]

    return {"answer": resp_json}

if __name__ == "__main__":
    uvicorn.run("JobAnalyze_API:app", port=5000)
