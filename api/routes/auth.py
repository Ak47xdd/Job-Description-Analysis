from fastapi import APIRouter, Body, HTTPException, status
from supabase_auth.errors import AuthApiError
from starlette.requests import Request
import traceback

from schemas import SignInRequest, SignUpRequest
from rate_limit import limiter
from auth_helpers import _supabase_access_token, _provision_confirmed_user, FRONTEND_URL

router = APIRouter()

@router.post("/auth/create_acc", status_code=status.HTTP_201_CREATED, operation_id="sign_up")
@limiter.limit("5/minute")
async def create_acc(request: Request, data: SignUpRequest) -> dict:
    from supabase_client import supabase
    email = str(data.email).strip().lower()
    name = data.name.strip()
    try:
        res = supabase.auth.sign_up({"email": email, "password": data.password, "options": {"data": {"name": name}, "email_redirect_to": f"{FRONTEND_URL}/auth/email-confirmed"}})
    except AuthApiError as e:
        msg = str(e).lower()
        if "already registered" in msg or "already exists" in msg:
            raise HTTPException(status_code=409, detail="An account with this email already exists. Please sign in instead.")
        if "password" in msg:
            raise HTTPException(status_code=422, detail="Password must be at least 6 characters.")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Unexpected signup error")
    if res.user is None:
        raise HTTPException(status_code=400, detail="Signup failed")
    if not getattr(res.user, "email_confirmed_at", None):
        return {"message": "Confirmation email sent", "requires_confirmation": True, "name": name, "email": email}
    if not res.session:
        raise HTTPException(status_code=500, detail="Signup completed but no session was returned")
    return {"message": "Account Created", "requires_confirmation": False, **_provision_confirmed_user(access_token=res.session.access_token)}

@router.post("/auth/resend_confirmation", operation_id="resend_confirmation")
@limiter.limit("3/15 minutes")
async def resend_confirmation(request: Request, data: dict = Body(...)) -> dict:
    from supabase_client import supabase
    email = str(data.get("email", "")).strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=422, detail="A valid email address is required.")
    try:
        supabase.auth.resend({"type": "signup", "email": email, "options": {"email_redirect_to": f"{FRONTEND_URL}/auth/email-confirmed"}})
    except AuthApiError:
        raise HTTPException(status_code=400, detail="Unable to resend confirmation email. Please check the email address and try again.")
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Unable to resend confirmation email")
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
    try:
        res = supabase.auth.sign_in_with_password({"email": email, "password": data.password})
    except AuthApiError as e:
        msg = str(e).lower()
        if "email not confirmed" in msg or "email_not_confirmed" in msg:
            raise HTTPException(status_code=403, detail="Please confirm your email address before signing in. Check your inbox or resend the confirmation email.")
        raise HTTPException(status_code=401, detail="Invalid email or password")
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Unexpected sign-in error")
    if res.user is None or res.session is None:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return _provision_confirmed_user(access_token=res.session.access_token)
