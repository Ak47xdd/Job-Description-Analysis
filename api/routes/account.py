from fastapi import APIRouter, HTTPException
from starlette.requests import Request

from rate_limit import limiter
from auth_helpers import _supabase_access_token

router = APIRouter()

@router.get("/auth/account", operation_id="account_details")
@limiter.limit("30/minute")
async def account_details(request: Request) -> dict:
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
        identity_provider = first_identity.get("provider") if isinstance(first_identity, dict) else getattr(first_identity, "provider", None)
    provider = str(app_metadata.get("provider") or identity_provider or "email")
    provider_label = "Google" if provider.lower() == "google" else "Email & password" if provider.lower() == "email" else provider
    record = get_api_key_db(owner=email, create_if_missing=False)
    api_key = record.get("api_key") if record else None
    return {"user": {"id": str(user.id), "name": name, "email": email, "email_verified": bool(getattr(user, "email_confirmed_at", None)), "provider": provider_label, "created_at": getattr(user, "created_at", None), "last_sign_in_at": getattr(user, "last_sign_in_at", None)}, "api": {"status": "active" if api_key else "not_provisioned", "api_key": api_key, "created_at": record.get("created_at") if record else None}, "usage": {"tracking_enabled": False, "analysis_count": None, "api_request_count": None, "message": "Usage tracking is not enabled yet."}}

@router.delete("/auth/delete_account", operation_id="delete_account")
@limiter.limit("3/hour")
async def delete_account(request: Request) -> dict:
    from supabase_client import supabase
    access_token = _supabase_access_token(request)
    try:
        user_response = supabase.auth.get_user(access_token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired Supabase session")
    user = user_response.user
    if user is None or not user.email:
        raise HTTPException(status_code=401, detail="Unable to identify authenticated user")
    user_id = str(user.id)
    email = str(user.email).strip().lower()
    supabase.table("api_tok").delete().eq("owner", email).execute()
    supabase.auth.admin.delete_user(user_id)
    return {"success": True, "message": "Account deleted successfully"}
