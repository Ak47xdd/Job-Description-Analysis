from fastapi import HTTPException
from starlette.requests import Request

import hashlib
import base64
import json
import time
import hmac
import os

from supabase_client import upsert_api_key_db
from auth import generate_api

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
    now = int(time.time())
    payload = {"email": email, "name": name, "iat": now, "exp": now + 120, "nonce": _b64url_encode(os.urandom(18))}
    encoded = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(_oauth_bridge_secret(), encoded.encode("ascii"), hashlib.sha256).digest()
    return f"{encoded}.{_b64url_encode(signature)}"

def _consume_oauth_bridge(code: str) -> dict | None:
    try:
        encoded, signature = code.split(".", 1)
        expected = hmac.new(_oauth_bridge_secret(), encoded.encode("ascii"), hashlib.sha256).digest()
        if not hmac.compare_digest(_b64url_decode(signature), expected): return None
        payload = json.loads(_b64url_decode(encoded).decode("utf-8"))
        now = int(time.time())
        if int(payload.get("exp", 0)) < now or now - int(payload.get("iat", now)) > 120: return None
        email = str(payload.get("email", "")).strip().lower(); name = str(payload.get("name", "")).strip()
        if not email or "@" not in email: return None
        return {"email": email, "name": name}
    except (ValueError, TypeError, KeyError, json.JSONDecodeError, UnicodeDecodeError): return None

def _supabase_access_token(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    scheme, _, token = auth.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Supabase access token required")
    return token

def _provision_confirmed_user(*, access_token: str) -> dict:
    from supabase_client import supabase, get_api_key_db
    try:
        user_response = supabase.auth.get_user(access_token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired Supabase session")
    user = user_response.user
    if user is None or not user.email:
        raise HTTPException(status_code=401, detail="Unable to identify authenticated user")
    if not getattr(user, "email_confirmed_at", None):
        raise HTTPException(status_code=403, detail="Please confirm your email address before continuing.")
    email = str(user.email).strip().lower()
    metadata = user.user_metadata or {}
    name = str(metadata.get("name") or metadata.get("full_name") or email.split("@")[0]).strip()
    record = get_api_key_db(owner=email, create_if_missing=True)
    if not record or not record.get("api_key"):
        raw = generate_api()
        record = upsert_api_key_db(owner=email, api_key=raw)
    if not record or not record.get("api_key"):
        raise HTTPException(status_code=500, detail="Unable to provision API key")
    return {"email": email, "name": name, "api_key": record["api_key"]}