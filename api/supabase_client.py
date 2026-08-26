"""Supabase integration and secure API-key persistence."""

from pathlib import Path
from dotenv import load_dotenv
import sys
import os
import secrets
import requests

from supabase import create_client, Client

try:
    from api.api_key_crypto import encrypt_api_key, decrypt_api_key, hash_api_key, keys_match
except ImportError:
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
_SECURE_COLUMNS = "user_id,owner,api_key,api_key_hash,api_key_encrypted"
_LEGACY_COLUMNS = "user_id,owner,api_key"


def _generate_api_key() -> str:
    return f"ja6k_{secrets.token_hex(32)}"


def _request(method: str, *, params: dict | None = None, json: dict | None = None, prefer: str | None = None):
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


def _secure_schema_available() -> bool:
    try:
        _request("GET", params={"select": "api_key_hash,api_key_encrypted", "limit": 1})
        return True
    except requests.RequestException:
        return False


def _select_secure_or_legacy(*, owner: str | None = None, api_key: str | None = None) -> list[dict]:
    if owner is None and api_key is None:
        return []

    secure_params = {"select": _SECURE_COLUMNS, "limit": 1}
    if owner is not None:
        secure_params["owner"] = f"eq.{owner}"
    else:
        secure_params["api_key_hash"] = f"eq.{hash_api_key(api_key)}"

    try:
        return _request("GET", params=secure_params)
    except requests.HTTPError:
        # Compatibility only for old rows while the migration is incomplete.
        legacy_params = {"select": _LEGACY_COLUMNS, "limit": 1}
        if owner is not None:
            legacy_params["owner"] = f"eq.{owner}"
        else:
            legacy_params["api_key"] = f"eq.{api_key}"
        return _request("GET", params=legacy_params)


def _migrate_legacy(row: dict) -> dict:
    """Encrypt an existing plaintext key and clear the legacy value."""
    legacy = row.get("api_key")
    if not legacy:
        return row

    key_hash = hash_api_key(legacy)
    encrypted = encrypt_api_key(legacy)
    result = dict(row)
    result["api_key"] = legacy
    result["api_key_hash"] = key_hash
    result["api_key_encrypted"] = encrypted

    user_pk = row.get("user_id")
    if user_pk is not None:
        try:
            data = _request(
                "PATCH",
                params={"user_id": f"eq.{user_pk}"},
                json={
                    "api_key": None,
                    "api_key_hash": key_hash,
                    "api_key_encrypted": encrypted,
                },
                prefer="return=representation",
            )
            if data:
                result.update(data[0])
                result["api_key"] = legacy
        except requests.RequestException:
            # Keep the key usable in memory; retry migration on a later request.
            pass
    return result


def _normalize(row: dict) -> dict:
    encrypted = row.get("api_key_encrypted")
    key_hash = row.get("api_key_hash")
    if encrypted and key_hash:
        try:
            plaintext = decrypt_api_key(encrypted)
        except RuntimeError:
            return row
        result = dict(row)
        result["api_key"] = plaintext
        return result
    if row.get("api_key"):
        return _migrate_legacy(row)
    return row


def _insert_secure_api_key(*, owner: str, api_key: str) -> dict:
    encrypted = encrypt_api_key(api_key)
    key_hash = hash_api_key(api_key)
    payload = {
        "owner": owner,
        "api_key": None,
        "api_key_hash": key_hash,
        "api_key_encrypted": encrypted,
    }
    data = _request("POST", json=payload, prefer="return=representation")
    row = data[0] if data else dict(payload)
    row["api_key"] = api_key
    return row


def _insert_legacy_api_key(*, owner: str, api_key: str) -> dict:
    # Only used before the secure columns have been created. Once the migration
    # is installed, this path is never selected.
    data = _request(
        "POST",
        json={"owner": owner, "api_key": api_key},
        prefer="return=representation",
    )
    row = data[0] if data else {"owner": owner, "api_key": api_key}
    return _migrate_legacy(row)


def _insert_api_key(*, owner: str, api_key: str) -> dict:
    if _secure_schema_available():
        return _insert_secure_api_key(owner=owner, api_key=api_key)
    return _insert_legacy_api_key(owner=owner, api_key=api_key)


def upsert_api_key_db(*, user_id=None, owner: str, api_key: str) -> dict:
    existing = get_api_key_db(owner=owner, create_if_missing=False)
    key_hash = hash_api_key(api_key)
    encrypted = encrypt_api_key(api_key)

    if existing and existing.get("user_id") is not None:
        if _secure_schema_available():
            data = _request(
                "PATCH",
                params={"user_id": f"eq.{existing['user_id']}"},
                json={
                    "api_key": None,
                    "api_key_hash": key_hash,
                    "api_key_encrypted": encrypted,
                },
                prefer="return=representation",
            )
            row = data[0] if data else dict(existing)
            row["api_key"] = api_key
            return row
        return _migrate_legacy({**existing, "api_key": api_key})

    return _insert_api_key(owner=owner, api_key=api_key)


def get_api_key_db(*, owner: str | None = None, api_key: str | None = None, create_if_missing: bool = False) -> dict | None:
    if owner is None and api_key is None:
        return None

    rows = _select_secure_or_legacy(owner=owner, api_key=api_key)
    if rows:
        row = _normalize(rows[0])
        if api_key is not None:
            stored_hash = row.get("api_key_hash")
            if stored_hash:
                if not keys_match(api_key, stored_hash):
                    return None
            elif row.get("api_key") != api_key:
                return None
        return row

    if owner is not None and create_if_missing:
        return _insert_api_key(owner=owner, api_key=_generate_api_key())
    return None
