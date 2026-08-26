"""Supabase integration and secure API-key persistence."""

from pathlib import Path
from dotenv import load_dotenv
import base64
import json
import sys
import os
import secrets
import requests

from supabase import create_client, Client

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_env_path = Path(__file__).resolve().parent / ".env"
if not _env_path.exists():
    _env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=_env_path)

try:
    from api.api_key_crypto import encrypt_api_key, decrypt_api_key, hash_api_key, keys_match
except ImportError:
    from api_key_crypto import encrypt_api_key, decrypt_api_key, hash_api_key, keys_match

SUPA_URL = os.getenv("SUPA_URL", "").rstrip("/")
SUPA_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPA_KEY", "")
if not SUPA_URL or not SUPA_KEY:
    raise RuntimeError("SUPA_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPA_KEY) must be configured")


def _jwt_role(token: str) -> str | None:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        decoded = base64.urlsafe_b64decode(payload).decode("utf-8")
        return str(json.loads(decoded).get("role") or "").strip().lower() or None
    except Exception:
        return None


role = _jwt_role(SUPA_KEY)
if role not in {"service_role", "supabase_admin"}:
    raise RuntimeError(
        "The backend Supabase key does not have service-role privileges. "
        "Set SUPABASE_SERVICE_ROLE_KEY to the Supabase service_role secret "
        "(server-side only). Do not use the anon or publishable key here."
    )

os.environ.setdefault("SUPA_KEY", SUPA_KEY)

supabase: Client = create_client(SUPA_URL, SUPA_KEY)
_HEADERS = {
    "apikey": SUPA_KEY,
    "Authorization": f"Bearer {SUPA_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}
_SECURE_COLUMNS = "user_id,owner,api_key_hash,api_key_encrypted"


def _generate_api_key() -> str:
    return f"ja6k_{secrets.token_hex(32)}"


def _request(
    method: str,
    *,
    params: dict | None = None,
    json: dict | None = None,
    prefer: str | None = None,
):
    headers = dict(_HEADERS)
    if prefer:
        headers["Prefer"] = prefer
    response = requests.request(
        method,
        f"{SUPA_URL}/rest/v1/api_tok",
        headers=headers,
        params=params,
        json=json,
        timeout=15,
    )
    response.raise_for_status()
    return response.json() if response.content else []


def _select_secure(*, owner: str | None = None, api_key: str | None = None) -> list[dict]:
    if owner is None and api_key is None:
        return []

    params = {"select": _SECURE_COLUMNS, "limit": 1}
    if owner is not None:
        params["owner"] = f"eq.{owner}"
    else:
        params["api_key_hash"] = f"eq.{hash_api_key(api_key)}"

    return _request("GET", params=params)


def _normalize(row: dict) -> dict:
    encrypted = row.get("api_key_encrypted")
    key_hash = row.get("api_key_hash")

    if not encrypted or not key_hash:
        raise RuntimeError("API key record is incomplete; encrypted credentials are required")

    plaintext = decrypt_api_key(encrypted)
    if not keys_match(plaintext, key_hash):
        raise RuntimeError("Stored API key integrity check failed")

    result = dict(row)
    # Plaintext exists only in memory for the authenticated response.
    result["api_key"] = plaintext
    return result


def _insert_secure_api_key(*, owner: str, api_key: str) -> dict:
    encrypted = encrypt_api_key(api_key)
    key_hash = hash_api_key(api_key)

    data = _request(
        "POST",
        json={
            "owner": owner,
            "api_key_hash": key_hash,
            "api_key_encrypted": encrypted,
        },
        prefer="return=representation",
    )

    row = data[0] if data else {
        "owner": owner,
        "api_key_hash": key_hash,
        "api_key_encrypted": encrypted,
    }
    row["api_key"] = api_key
    return row


def upsert_api_key_db(*, user_id=None, owner: str, api_key: str) -> dict:
    existing = get_api_key_db(owner=owner, create_if_missing=False)
    key_hash = hash_api_key(api_key)
    encrypted = encrypt_api_key(api_key)

    if existing and existing.get("user_id") is not None:
        data = _request(
            "PATCH",
            params={"user_id": f"eq.{existing['user_id']}"},
            json={
                "api_key_hash": key_hash,
                "api_key_encrypted": encrypted,
            },
            prefer="return=representation",
        )
        row = data[0] if data else dict(existing)
        row["api_key"] = api_key
        return row

    return _insert_secure_api_key(owner=owner, api_key=api_key)


def get_api_key_db(
    *,
    owner: str | None = None,
    api_key: str | None = None,
    create_if_missing: bool = False,
) -> dict | None:
    if owner is None and api_key is None:
        return None

    rows = _select_secure(owner=owner, api_key=api_key)
    if rows:
        row = _normalize(rows[0])
        if api_key is not None and not keys_match(api_key, row["api_key_hash"]):
            return None
        return row

    if owner is not None and create_if_missing:
        generated = _generate_api_key()
        return _insert_secure_api_key(owner=owner, api_key=generated)

    return None
