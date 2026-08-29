from fastapi import HTTPException, Security, Header, status
from fastapi.security import APIKeyHeader
import secrets
import hashlib
import hmac
import os

API_KEY_NAME   = "JobAnalyze_6k_Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
API_KEY_DB: dict = {}


def generate_api(prefix: str = "ja6k") -> str:
    return f"{prefix}_{secrets.token_hex(32)}"


def hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


async def verify(
    api_key: str = Security(api_key_header),
    authorization: str | None = Header(default=None, alias="Authorization"),
):
    """Validate JobAnalyze API keys from the native or MCP auth header.

    Existing clients can continue sending JobAnalyze_6k_Key. MCP clients can
    use the standard Authorization: Bearer <api-key> form.
    """
    if not api_key and authorization:
        scheme, _, credentials = authorization.partition(" ")
        if scheme.lower() == "bearer":
            api_key = credentials.strip() or None

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key Missing From Header",
        )

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
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or Expired API Key",
        )
    return db_record


async def require_admin(
    api_key: str = Security(api_key_header),
    x_admin_secret: str | None = Header(default=None, alias="X-Admin-Secret"),
    authorization: str | None = Header(default=None, alias="Authorization"),
):
    """Require both a valid JobAnalyze API key and explicit admin authorization.

    ADMIN_API_KEY_HASHES is a comma-separated list of SHA-256 API-key hashes.
    ADMIN_SECRET is an additional deployment-level secret for sensitive
    administrative operations. Neither secret is returned to clients.
    """
    client = await verify(api_key=api_key, authorization=authorization)
    effective_key = api_key
    if not effective_key and authorization:
        scheme, _, credentials = authorization.partition(" ")
        if scheme.lower() == "bearer":
            effective_key = credentials.strip() or None

    configured_hashes = {
        value.strip().lower()
        for value in os.getenv("ADMIN_API_KEY_HASHES", "").split(",")
        if value.strip()
    }

    if not configured_hashes or not effective_key or hash_key(effective_key).lower() not in configured_hashes:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator authorization required.",
        )

    configured_secret = os.getenv("ADMIN_SECRET")
    if not configured_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Administrator authorization is not configured.",
        )

    if not x_admin_secret or not hmac.compare_digest(
        x_admin_secret, configured_secret
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator authorization required.",
        )

    return client
