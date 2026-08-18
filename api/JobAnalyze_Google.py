"""Production entrypoint for JobSelect Google OAuth.

This wraps the existing FastAPI application without replacing any of the
existing API routes. The old Google callback is removed and replaced with a
provider-independent flow that verifies the Google identity directly and then
provisions/reuses the JobSelect API key.
"""

from __future__ import annotations

import os
import traceback
import requests
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
from supabase_client import SUPA_URL, SUPA_KEY, get_api_key_db

FRONTEND_URL = os.getenv("FRONTEND_URL", "https://jobselect.vercel.app").rstrip("/")

# Remove the callback routes defined by older versions of JobAnalyze_API.py.
# FastAPI checks routes in order, so they must be removed before registering the
# corrected implementations below.
for route in list(app.router.routes):
    if getattr(route, "path", None) in {"/auth/google", "/auth/google/callback", "/auth/google/exchange"}:
        app.router.routes.remove(route)


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
        access_token = token_data.get("access_token")
        if not access_token:
            raise RuntimeError("Google did not return an access token")

        # Verify the authenticated Google identity directly with Google.
        userinfo = requests.get(
            "https://openidconnect.googleapis.com/v1/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=15,
        )
        userinfo.raise_for_status()
        google_user = userinfo.json()

        email = str(google_user.get("email", "")).strip().lower()
        if not email or not google_user.get("email_verified"):
            raise RuntimeError("Google account email is missing or not verified")

        name = google_user.get("name") or google_user.get("given_name") or email.split("@")[0]

        # Keep the Google account represented in Supabase auth.users. This uses
        # the backend SUPA_KEY, which must be the service-role key on Render.
        admin_response = requests.post(
            f"{SUPA_URL}/auth/v1/admin/users",
            headers={
                "apikey": SUPA_KEY,
                "Authorization": f"Bearer {SUPA_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "email": email,
                "email_confirm": True,
                "user_metadata": {
                    "name": name,
                    "full_name": name,
                    "provider": "google",
                },
            },
            timeout=15,
        )
        # 422 is expected when this Google email is already in auth.users.
        if admin_response.status_code not in (200, 201, 422):
            admin_response.raise_for_status()

        record = get_api_key_db(owner=email, create_if_missing=True)
        if not record or not record.get("api_key"):
            raise RuntimeError("Unable to provision API key")

        bridge_code = store_one_time_result({
            "email": email,
            "name": name,
            "api_key": record["api_key"],
        })
        return RedirectResponse(
            f"{FRONTEND_URL}/auth/google/callback?code={bridge_code}",
            status_code=302,
        )
    except Exception:
        traceback.print_exc()
        return RedirectResponse(f"{FRONTEND_URL}/login?oauth_error=google_failed")


@app.post("/auth/google/exchange", include_in_schema=False)
async def google_exchange(code: str) -> dict:
    result = consume_one_time_result(code)
    if not result:
        raise HTTPException(status_code=401, detail="OAuth code is invalid or expired")
    return result
