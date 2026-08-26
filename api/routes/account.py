from fastapi import APIRouter, Body, HTTPException
from starlette.requests import Request

from rate_limit import limiter
from auth_helpers import _supabase_access_token

router = APIRouter()
DELETE_CONFIRMATION = "DELETE"


def _get_authenticated_user(request: Request):
    from supabase_client import supabase
    access_token = _supabase_access_token(request)
    try:
        response = supabase.auth.get_user(access_token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired Supabase session")
    user = response.user
    if user is None or not user.email:
        raise HTTPException(status_code=401, detail="Unable to identify authenticated user")
    return supabase, user


def _provider(user) -> str:
    app_metadata = user.app_metadata or {}
    identities = getattr(user, "identities", None) or []
    identity_provider = None
    if identities:
        identity = identities[0]
        identity_provider = identity.get("provider") if isinstance(identity, dict) else getattr(identity, "provider", None)
    return str(app_metadata.get("provider") or identity_provider or "email").strip().lower()


def _delete_user_data(supabase, user) -> dict:
    email = str(user.email).strip().lower()
    user_id = str(user.id)
    try:
        supabase.auth.admin.delete_user(user_id)
    except Exception:
        raise HTTPException(status_code=500, detail="Unable to delete the authenticated account")
    try:
        supabase.table("api_tok").delete().eq("owner", email).execute()
    except Exception:
        # The auth account is already deleted. Do not report a false success for cleanup failures.
        raise HTTPException(status_code=500, detail="Account was deleted, but associated API credentials could not be cleaned up. Please contact support.")
    return {"success": True, "message": "Account deleted successfully"}


@router.get("/auth/account", operation_id="account_details")
@limiter.limit("30/minute")
async def account_details(request: Request) -> dict:
    from supabase_client import get_api_key_db
    supabase, user = _get_authenticated_user(request)
    email = str(user.email).strip().lower()
    metadata = user.user_metadata or {}
    name = str(metadata.get("name") or metadata.get("full_name") or email.split("@")[0]).strip()
    provider = _provider(user)
    provider_label = "Google" if provider == "google" else "Email & password" if provider == "email" else provider
    record = get_api_key_db(owner=email, create_if_missing=False)
    api_key = record.get("api_key") if record else None
    return {"user": {"id": str(user.id), "name": name, "email": email, "email_verified": bool(getattr(user, "email_confirmed_at", None)), "provider": provider_label, "created_at": getattr(user, "created_at", None), "last_sign_in_at": getattr(user, "last_sign_in_at", None)}, "api": {"status": "active" if api_key else "not_provisioned", "api_key": api_key, "created_at": record.get("created_at") if record else None}, "usage": {"tracking_enabled": False, "analysis_count": None, "api_request_count": None, "message": "Usage tracking is not enabled yet."}}


@router.delete("/auth/delete_account/email", operation_id="delete_email_account")
@limiter.limit("3/hour")
async def delete_email_account(request: Request, data: dict = Body(...)) -> dict:
    from supabase_client import supabase
    access_token = _supabase_access_token(request)
    password = str(data.get("password", ""))
    confirmation = str(data.get("confirmation", ""))
    if confirmation != DELETE_CONFIRMATION:
        raise HTTPException(status_code=422, detail="Type DELETE to confirm account deletion")
    if not password:
        raise HTTPException(status_code=422, detail="Your current password is required")
    try:
        current = supabase.auth.get_user(access_token).user
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired Supabase session")
    if current is None or not current.email:
        raise HTTPException(status_code=401, detail="Unable to identify authenticated user")
    if _provider(current) != "email":
        raise HTTPException(status_code=409, detail="This account uses Google sign-in. Use the Google account deletion confirmation instead.")
    try:
        reauth = supabase.auth.sign_in_with_password({"email": str(current.email).strip().lower(), "password": password})
    except Exception:
        raise HTTPException(status_code=401, detail="Incorrect password")
    if not reauth.user or str(reauth.user.id) != str(current.id):
        raise HTTPException(status_code=401, detail="Unable to verify account ownership")
    return _delete_user_data(supabase, current)


@router.delete("/auth/delete_account/oauth", operation_id="delete_oauth_account")
@limiter.limit("3/hour")
async def delete_oauth_account(request: Request, data: dict = Body(...)) -> dict:
    confirmation = str(data.get("confirmation", ""))
    if confirmation != DELETE_CONFIRMATION:
        raise HTTPException(status_code=422, detail="Type DELETE to confirm account deletion")
    supabase, user = _get_authenticated_user(request)
    if _provider(user) != "google":
        raise HTTPException(status_code=409, detail="This account uses email and password sign-in. Use the email account deletion confirmation instead.")
    return _delete_user_data(supabase, user)
