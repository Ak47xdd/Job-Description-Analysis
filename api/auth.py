from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
import secrets
import hashlib

API_KEY_NAME   = "JobAnalyze_6k_Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
API_KEY_DB: dict = {}

def generate_api(prefix: str = "ja6k") -> str:
    return f"{prefix}_{secrets.token_hex(32)}"


def hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


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