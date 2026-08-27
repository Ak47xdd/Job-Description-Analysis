from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import RedirectResponse
from supabase_auth.errors import AuthApiError
from starlette.requests import Request
import traceback

from auth_helpers import _consume_oauth_bridge, _create_oauth_bridge, FRONTEND_URL
from google_oauth import google_authorize_url, verify_state, exchange_google_code
from supabase_client import upsert_api_key_db
from auth import generate_api

router = APIRouter()

@router.get(
    "/auth/google", 
    include_in_schema=False
    )
async def google_login():
    
    try:
        return RedirectResponse(url=google_authorize_url(), status_code=302)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

@router.get(
    "/auth/google/callback", 
    include_in_schema=False
    )
async def google_callback(
    code: str | None = None, 
    state: str | None = None, 
    error: str | None = None,
    error_description: str | None = None):
    
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
        auth_result = supabase.auth.sign_in_with_id_token({"provider": "google", "token": google_id_token})
        user = auth_result.user
        if user is None or not user.email:
            raise RuntimeError("Supabase did not return a Google user")
        email = str(user.email).strip().lower()
        metadata = user.user_metadata or {}
        name = str(metadata.get("name") or metadata.get("full_name") or email.split("@")[0]).strip()
        record = get_api_key_db(owner=email, create_if_missing=True)
        if not record or not record.get("api_key"):
            record = upsert_api_key_db(owner=email, api_key=generate_api())
        if not record or not record.get("api_key"):
            raise RuntimeError("Unable to provision API key")
        bridge = _create_oauth_bridge(email=email, name=name)
        return RedirectResponse(f"{FRONTEND_URL}/auth/google/callback?code={bridge}", status_code=302)
    except AuthApiError as exc:
        traceback.print_exc()
        return RedirectResponse(f"{FRONTEND_URL}/login?oauth_error=supabase_google_failed&detail={str(exc).replace(' ', '_')}")
    except Exception:
        traceback.print_exc()
        return RedirectResponse(f"{FRONTEND_URL}/login?oauth_error=google_failed")

@router.post(
    "/auth/google/exchange", 
    include_in_schema=False
    )
async def google_exchange(
    request: Request, 
    code: str | None = None, 
    body: dict | None = Body(default=None)
    ) -> dict:
    
    supplied_code = code or (body.get("code") if isinstance(body, dict) else None)
    if not supplied_code:
        raise HTTPException(status_code=422, detail="OAuth code is required")
    bridge = _consume_oauth_bridge(str(supplied_code))
    if not bridge:
        raise HTTPException(status_code=401, detail="OAuth code is invalid or expired")
    from supabase_client import get_api_key_db
    record = get_api_key_db(owner=bridge["email"], create_if_missing=True)
    if not record or not record.get("api_key"):
        raise HTTPException(status_code=500, detail="Unable to provision API key")
    return {"email": bridge["email"], "name": bridge["name"] or bridge["email"].split("@")[0], "api_key": record["api_key"]}
