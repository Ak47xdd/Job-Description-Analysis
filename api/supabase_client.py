"""Supabase integration and secure API-key persistence."""

from pathlib import Path
from dotenv import load_dotenv
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
# SUPA_KEY is the canonical backend Supabase credential. New sb_secret_ keys
# are preferred; the legacy service-role variable remains only as a migration
# fallback for older deployments.
SUPA_KEY = os.getenv("SUPA_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
if not SUPA_URL or not SUPA_KEY:
    raise RuntimeError("SUPA_URL and SUPA_KEY must be configured")

IS_SECRET_KEY = SUPA_KEY.startswith("sb_secret_")

# Supabase secret keys are opaque API keys, not JWTs. They must be sent in the
# apikey header and must not be sent as Authorization: Bearer. Legacy
# service_role keys are JWTs, so they retain the Authorization header.
_HEADERS = {
    "apikey": SUPA_KEY,
    "Content-Type": "application/json",
    "Accept": "application/json",
}
if not IS_SECRET_KEY:
    _HEADERS["Authorization"] = f"Bearer {SUPA_KEY}"

# Keep the SDK client available for existing imports. Database operations below
# use the REST client so the correct header behavior is explicit for sb_secret.
supabase: Client = create_client(SUPA_URL, SUPA_KEY)
os.environ.setdefault("SUPA_KEY", SUPA_KEY)
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
