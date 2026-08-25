from google_oauth import google_authorize_url, verify_state, exchange_google_code
from fastapi import HTTPException, status, Body
from fastapi.responses import RedirectResponse
from supabase_auth.errors import AuthApiError
from starlette.requests import Request
from fastapi import APIRouter
import traceback

from schemas import SignInRequest, SignUpRequest
from supabase_client import upsert_api_key_db
from rate_limit import limiter
from auth_helpers import (
    _supabase_access_token,
    _consume_oauth_bridge,
    _create_oauth_bridge,
    _provision_confirmed_user,
    FRONTEND_URL
)
from api.auth import generate_api

router = APIRouter(
    prefix="/users",
    tags=["items"]
)

@router.get("/auth/account", operation_id="account_details")
@limiter.limit("30/minute")
async def account_details(request: Request) -> dict:
    """Return authoritative account data for the authenticated dashboard."""
    from supabase_client import supabase, get_api_key_db
    access_token = _supabase_access_token(request)
    try:
        user_response = supabase.auth.get_user(access_token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired Supabase session")
    user = user_response.user
    if user is None or not user.email:
        raise HTTPException(status_code=401, detail="Unable to identify authenticated user")
    email = str(user.email).strip().lower()
    metadata = user.user_metadata or {}
    name = str(metadata.get("name") or metadata.get("full_name") or email.split("@")[0]).strip()
    app_metadata = user.app_metadata or {}
    identities = getattr(user, "identities", None) or []
    identity_provider = None
    if identities:
        first_identity = identities[0]
        if isinstance(first_identity, dict):
            identity_provider = first_identity.get("provider")
        else:
            identity_provider = getattr(first_identity, "provider", None)
    provider = str(app_metadata.get("provider") or identity_provider or "email")
    provider_label = "Google" if provider.lower() == "google" else "Email & password" if provider.lower() == "email" else provider
    record = get_api_key_db(owner=email, create_if_missing=False)
    api_key = record.get("api_key") if record else None
    return {"user": {"id": str(user.id), "name": name, "email": email, "email_verified": bool(getattr(user, "email_confirmed_at", None)), "provider": provider_label, "created_at": getattr(user, "created_at", None), "last_sign_in_at": getattr(user, "last_sign_in_at", None)}, "api": {"status": "active" if api_key else "not_provisioned", "api_key": api_key, "created_at": record.get("created_at") if record else None}, "usage": {"tracking_enabled": False, "analysis_count": None, "api_request_count": None, "message": "Usage tracking is not enabled yet."}}

@router.get("/auth/google", include_in_schema=False)
async def google_login():
    try: return RedirectResponse(url=google_authorize_url(), status_code=302)
    except RuntimeError as exc: raise HTTPException(status_code=503, detail=str(exc))

@router.get("/auth/google/callback", include_in_schema=False)
async def google_callback(code: str | None = None, state: str | None = None, error: str | None = None, error_description: str | None = None):
    if error:
        detail = (error_description or error).replace(" ", "_")
        return RedirectResponse(f"{FRONTEND_URL}/login?oauth_error=google_denied&detail={detail}")
    if not code or not state or not verify_state(state): return RedirectResponse(f"{FRONTEND_URL}/login?oauth_error=invalid_oauth_state")
    try:
        token_data = exchange_google_code(code); google_id_token = token_data.get("id_token")
        if not google_id_token: raise RuntimeError("Google did not return an ID token")
        from supabase_client import supabase, get_api_key_db
        auth_result = supabase.auth.sign_in_with_id_token({"provider": "google", "token": google_id_token}); user = auth_result.user
        if user is None or not user.email: raise RuntimeError("Supabase did not return a Google user")
        email = str(user.email).strip().lower(); metadata = user.user_metadata or {}
        name = str(metadata.get("name") or metadata.get("full_name") or email.split("@")[0]).strip()
        record = get_api_key_db(owner=email, create_if_missing=True)
        if not record or not record.get("api_key"): record = upsert_api_key_db(owner=email, api_key=generate_api())
        if not record or not record.get("api_key"): raise RuntimeError("Unable to provision API key")
        return RedirectResponse(f"{FRONTEND_URL}/auth/google/callback?code={_create_oauth_bridge(email=email, name=name)}", status_code=302)
    except AuthApiError as exc:
        traceback.print_exc(); return RedirectResponse(f"{FRONTEND_URL}/login?oauth_error=supabase_google_failed&detail={str(exc).replace(' ', '_')}")
    except Exception:
        traceback.print_exc(); return RedirectResponse(f"{FRONTEND_URL}/login?oauth_error=google_failed")

@router.post("/auth/google/exchange", include_in_schema=False)
async def google_exchange(request: Request, code: str | None = None, body: dict | None = Body(default=None)) -> dict:
    supplied_code = code or (body.get("code") if isinstance(body, dict) else None)
    if not supplied_code: raise HTTPException(status_code=422, detail="OAuth code is required")
    bridge = _consume_oauth_bridge(str(supplied_code))
    if not bridge: raise HTTPException(status_code=401, detail="OAuth code is invalid or expired")
    from supabase_client import get_api_key_db
    record = get_api_key_db(owner=bridge["email"], create_if_missing=True)
    if not record or not record.get("api_key"): raise HTTPException(status_code=500, detail="Unable to provision API key")
    return {"email": bridge["email"], "name": bridge["name"] or bridge["email"].split("@")[0], "api_key": record["api_key"]}

@router.post("/auth/create_acc", status_code=status.HTTP_201_CREATED, operation_id="sign_up")
@limiter.limit("5/minute")
async def create_acc(request: Request, data: SignUpRequest) -> dict:
    from supabase_client import supabase
    email = str(data.email).strip().lower(); name = data.name.strip()
    try:
        res = supabase.auth.sign_up({"email": email, "password": data.password, "options": {"data": {"name": name}, "email_redirect_to": f"{FRONTEND_URL}/auth/email-confirmed"}})
    except AuthApiError as e:
        msg = str(e).lower()
        if "already registered" in msg or "already exists" in msg: raise HTTPException(status_code=409, detail="An account with this email already exists. Please sign in instead.")
        if "password" in msg: raise HTTPException(status_code=422, detail="Password must be at least 6 characters.")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        traceback.print_exc(); raise HTTPException(status_code=500, detail="Unexpected signup error")
    if res.user is None: raise HTTPException(status_code=400, detail="Signup failed")
    if not getattr(res.user, "email_confirmed_at", None): return {"message": "Confirmation email sent", "requires_confirmation": True, "name": name, "email": email}
    if not res.session: raise HTTPException(status_code=500, detail="Signup completed but no session was returned")
    return {"message": "Account Created", "requires_confirmation": False, **_provision_confirmed_user(access_token=res.session.access_token)}

@router.post("/auth/resend_confirmation", operation_id="resend_confirmation")
@limiter.limit("3/15 minutes")
async def resend_confirmation(request: Request, data: dict = Body(...)) -> dict:
    from supabase_client import supabase
    email = str(data.get("email", "")).strip().lower()
    if not email or "@" not in email: raise HTTPException(status_code=422, detail="A valid email address is required.")
    try: supabase.auth.resend({"type": "signup", "email": email, "options": {"email_redirect_to": f"{FRONTEND_URL}/auth/email-confirmed"}})
    except AuthApiError: raise HTTPException(status_code=400, detail="Unable to resend confirmation email. Please check the email address and try again.")
    except Exception:
        traceback.print_exc(); raise HTTPException(status_code=500, detail="Unable to resend confirmation email")
    return {"message": "If the account requires confirmation, a new confirmation email has been sent.", "email": email}

@router.post("/auth/provision", operation_id="provision_confirmed_account")
@limiter.limit("10/minute")
async def provision_confirmed_account(request: Request) -> dict:
    return _provision_confirmed_user(access_token=_supabase_access_token(request))

@router.post("/auth/sign_in", operation_id="sign_in")
@limiter.limit("5/minute")
async def sign_in(request: Request, data: SignInRequest) -> dict:
    from supabase_client import supabase
    email = str(data.email).strip().lower()
    try: res = supabase.auth.sign_in_with_password({"email": email, "password": data.password})
    except AuthApiError as e:
        msg = str(e).lower()
        if "email not confirmed" in msg or "email_not_confirmed" in msg: raise HTTPException(status_code=403, detail="Please confirm your email address before signing in. Check your inbox or resend the confirmation email.")
        raise HTTPException(status_code=401, detail="Invalid email or password")
    except Exception:
        traceback.print_exc(); raise HTTPException(status_code=500, detail="Unexpected sign-in error")
    if res.user is None or res.session is None: raise HTTPException(status_code=401, detail="Invalid email or password")
    return _provision_confirmed_user(access_token=res.session.access_token)

@router.delete("/auth/delete_account", operation_id="delete_account")
@limiter.limit("3/hour")
async def delete_account(request: Request) -> dict:
    from supabase_client import supabase
    access_token = _supabase_access_token(request)
    try: user_response = supabase.auth.get_user(access_token)
    except Exception: raise HTTPException(status_code=401, detail="Invalid or expired Supabase session")
    user = user_response.user
    if user is None or not user.email: raise HTTPException(status_code=401, detail="Unable to identify authenticated user")
    user_id = str(user.id); email = str(user.email).strip().lower()
    supabase.table("api_tok").delete().eq("owner", email).execute()
    supabase.auth.admin.delete_user(user_id)
    return {"success": True, "message": "Account deleted successfully"}
