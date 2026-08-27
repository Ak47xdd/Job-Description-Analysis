"""Google OAuth helpers for the JobSelect web application."""
from urllib.parse import urlencode
from __future__ import annotations
import requests
import hashlib
import secrets
import base64
import hmac
import json
import time
import os

GOOGLE_AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_SCOPES = "openid email profile"
_PENDING_CODES: dict[str, tuple[float, dict]] = {}


def _secret() -> bytes:
    secret = os.getenv("GOOGLE_OAUTH_STATE_SECRET") or os.getenv("SUPA_KEY")
    if not secret:
        raise RuntimeError("GOOGLE_OAUTH_STATE_SECRET or SUPA_KEY must be configured")
    return secret.encode("utf-8")


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def create_state() -> str:
    payload = {"nonce": secrets.token_urlsafe(24), "iat": int(time.time())}
    encoded = _b64(json.dumps(payload, separators=(",", ":")).encode())
    signature = hmac.new(_secret(), encoded.encode(), hashlib.sha256).digest()
    return f"{encoded}.{_b64(signature)}"


def verify_state(state: str) -> bool:
    try:
        encoded, signature = state.split(".", 1)
        expected = hmac.new(_secret(), encoded.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(_unb64(signature), expected):
            return False
        payload = json.loads(_unb64(encoded))
        return int(time.time()) - int(payload["iat"]) <= 600
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def google_authorize_url() -> str:
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI")
    if not client_id or not redirect_uri:
        raise RuntimeError("GOOGLE_CLIENT_ID and GOOGLE_REDIRECT_URI must be configured")
    state = create_state()
    return f"{GOOGLE_AUTHORIZE_URL}?{urlencode({
        'client_id': client_id,
        'redirect_uri': redirect_uri,
        'response_type': 'code',
        'scope': GOOGLE_SCOPES,
        'state': state,
        'access_type': 'offline',
        'prompt': 'select_account',
    })}"


def exchange_google_code(code: str) -> dict:
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI")
    if not client_id or not client_secret or not redirect_uri:
        raise RuntimeError("GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET and GOOGLE_REDIRECT_URI must be configured")
    response = requests.post(
        GOOGLE_TOKEN_URL,
        data={
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def store_one_time_result(result: dict) -> str:
    _cleanup()
    code = secrets.token_urlsafe(32)
    _PENDING_CODES[code] = (time.time() + 120, result)
    return code


def consume_one_time_result(code: str) -> dict | None:
    _cleanup()
    entry = _PENDING_CODES.pop(code, None)
    if not entry:
        return None
    expires_at, result = entry
    if time.time() > expires_at:
        return None
    return result


def _cleanup() -> None:
    now = time.time()
    for key, (expires_at, _) in list(_PENDING_CODES.items()):
        if expires_at < now:
            _PENDING_CODES.pop(key, None)
