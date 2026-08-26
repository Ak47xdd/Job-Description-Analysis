"""Supabase integration and secure API-key persistence."""

from pathlib import Path
from dotenv import load_dotenv
import sys
import os
import secrets
import requests

from supabase import create_client, Client

from api_key_crypto import encrypt_api_key, decrypt_api_key, hash_api_key, keys_match

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_env_path = Path(__file__).resolve().parent / ".env"
if not _env_path.exists():
    _env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=_env_path)

SUPA_URL = os.getenv("SUPA_URL", "").rstrip("/")
SUPA_KEY = os.getenv("SUPA_KEY", "")

if not SUPA_URL or not SUPA_KEY:
    raise RuntimeError("SUPA_URL and SUPA_KEY must be configured")

supabase: Client = create_client(SUPA_URL, SUPA_KEY)

_HEADERS = {
    "apikey": SUPA_KEY,
    "Authorization": f"Bearer {SUPA_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}

# Preferred schema:
#   api_key_hash        SHA-256 fingerprint used for authentication
#   api_key_encrypted   Fernet ciphertext used only for authenticated display
# Keep api_key temporarily for one-time migration of legacy plaintext rows.
_API_KEY_COLUMNS = "user_id,owner,api_key,api_key_hash,api_key_encrypted"


def _generate_api_key() -> str:
    return f"ja6k_{secrets.token_hex(32)}"


def _secure_record(row: dict) -> dict:
    """Normalize a DB row and migrate a legacy plaintext key when encountered."""
    encrypted = row.get("api_key_encrypted")
    key_hash = row.get("api_key_hash")
    legacy = row.get("api_key")

    if encrypted and key_hash:
        result = dict(row)
        result["api_key"] = decrypt_api_key(encrypted)
        return result

    if legacy:
        # One-time compatibility migration. The plaintext value is immediately
        # replaced by ciphertext + hash and is not returned from Supabase again.
        secure_update = {
            "api_key_hash": hash_api_key(legacy),
            "api_key_encrypted": encrypt_api_key(legacy),
            "api_key": None,
        }
        user_pk = row.get("user_id")
        if user_pk is not None:
            resp = requests.patch(
                f"{SUPA_URL}/rest/v1/api_tok",
                params={"user_id": f"eq.{user_pk}"},
                headers={**_HEADERS, "Prefer": "return=representation"},
                json=secure_update,
                timeout=15,
            )
            resp.raise_for_status()
        result = {**row, **secure_update}
        result["api_key"] = legacy
        return result

    return row


def _insert_api_key(*, owner: str, api_key: str) -> dict:
    """Insert only encrypted/hash representations of a new API key."""
    payload = {
        "owner": owner,
        "api_key": None,
        "api_key_hash": hash_api_key(api_key),
        "api_key_encrypted": encrypt_api_key(api_key),
    }
    resp = requests.post(
        f"{SUPA_URL}/rest/v1/api_tok",
        headers={**_HEADERS, "Prefer": "return=representation"},
        json=payload,
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    row = data[0] if data else {**payload, "api_key": None}
    row["api_key"] = api_key
    return row


def upsert_api_key_db(*, user_id=None, owner: str, api_key: str) -> dict:
    """Create/update an API key without storing plaintext in Supabase."""
    existing = get_api_key_db(owner=owner, create_if_missing=False)
    secure_update = {
        "owner": owner,
        "api_key": None,
        "api_key_hash": hash_api_key(api_key),
        "api_key_encrypted": encrypt_api_key(api_key),
    }

    if existing and existing.get("user_id") is not None:
        user_pk = existing["user_id"]
        resp = requests.patch(
            f"{SUPA_URL}/rest/v1/api_tok",
            params={"user_id": f"eq.{user_pk}"},
            headers={**_HEADERS, "Prefer": "return=representation"},
            json=secure_update,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        row = data[0] if data else {**existing, **secure_update}
        row["api_key"] = api_key
        return row

    return _insert_api_key(owner=owner, api_key=api_key)


def get_api_key_db(*, owner: str | None = None, api_key: str | None = None, create_if_missing: bool = False) -> dict | None:
    """Find an API key by owner or candidate key, migrating legacy rows."""
    if owner is None and api_key is None:
        return None

    if owner is not None:
        params = {"select": _API_KEY_COLUMNS, "owner": f"eq.{owner}", "limit": 1}
    else:
        # Never query Supabase using the encrypted value. Prefer the deterministic
        # hash column; the legacy fallback below is only for old plaintext rows.
        params = {"select": _API_KEY_COLUMNS, "api_key_hash": f"eq.{hash_api_key(api_key)}", "limit": 1}

    resp = requests.get(
        f"{SUPA_URL}/rest/v1/api_tok",
        headers=_HEADERS,
        params=params,
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    if data:
        row = _secure_record(data[0])
        if api_key is not None and not keys_match(api_key, row.get("api_key_hash", "")):
            return None
        return row

    # Legacy compatibility: locate old plaintext records only during migration.
    if api_key is not None:
        legacy_resp = requests.get(
            f"{SUPA_URL}/rest/v1/api_tok",
            headers=_HEADERS,
            params={"select": _API_KEY_COLUMNS, "api_key": f"eq.{api_key}", "limit": 1},
            timeout=15,
        )
        legacy_resp.raise_for_status()
        legacy_data = legacy_resp.json()
        if legacy_data:
            row = _secure_record(legacy_data[0])
            if keys_match(api_key, row.get("api_key_hash", "")):
                return row

    if owner is not None and create_if_missing:
        return _insert_api_key(owner=owner, api_key=_generate_api_key())

    return None
