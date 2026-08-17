"""Deployment entrypoint that adds Google OAuth routes to the existing API.

Use this module as the Render/uvicorn application target instead of
api.JobAnalyze_API:app:

    uvicorn api.JobAnalyze_OAuth:app --host 0.0.0.0 --port $PORT
"""

import os
from fastapi import HTTPException
from fastapi.responses import RedirectResponse

from JobAnalyze_API import app
from google_oauth import (
    google_authorize_url,
    verify_state,
    exchange_google_code,
    store_one_time_result,
    consume_one_time_result,
)

FRONTEND_URL = os.getenv("FRONTEND_URL", "https://jobselect.vercel.app").rstrip("/")


@app.get("/auth/google", include_in_schema=False)
async def google_login():
    try:
        return RedirectResponse(url=google_authorize_url(), status_code=302)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/auth/google/callback", include_in_schema=False)
async def google_callback(code: str | None = None, state: str | None = None, error: str | None = None):
    if error:
        return RedirectResponse(f"{FRONTEND_URL}/login?oauth_error=google_denied")
    if not code or not state or not verify_state(state):
        return RedirectResponse(f"{FRONTEND_URL}/login?oauth_error=invalid_oauth_state")

    try:
        token_data = exchange_google_code(code)
        id_token = token_data.get("id_token")
        if not id_token:
            raise RuntimeError("Google did not return an ID token")

        from supabase_client import supabase, get_api_key_db

        auth_result = supabase.auth.sign_in_with_id_token({
            "provider": "google",
            "id_token": id_token,
        })
        user = auth_result.user
        if user is None or not user.email:
            raise RuntimeError("Supabase did not return a Google user")

        email = str(user.email).strip().lower()
        metadata = user.user_metadata or {}
        name = metadata.get("name") or metadata.get("full_name") or email.split("@")[0]

        record = get_api_key_db(owner=email, create_if_missing=True)
        if not record or not record.get("api_key"):
            raise RuntimeError("Unable to provision API key")

        bridge_code = store_one_time_result({
            "email": email,
            "name": name,
            "api_key": record["api_key"],
        })
        return RedirectResponse(f"{FRONTEND_URL}/auth/google/callback?code={bridge_code}")
    except Exception:
        import traceback
        traceback.print_exc()
        return RedirectResponse(f"{FRONTEND_URL}/login?oauth_error=google_failed")


@app.post("/auth/google/exchange", include_in_schema=False)
async def google_exchange(code: str):
    result = consume_one_time_result(code)
    if not result:
        raise HTTPException(status_code=401, detail="OAuth code is invalid or expired")
    return result
