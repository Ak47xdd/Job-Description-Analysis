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
_API_KEY_COLUMNS = "user_id,owner,api_key,api_key_hash,api_key_encrypted"


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


def _select_secure_or_legacy(*, owner: str | None = None, api_key: str | None = None) -> list[dict]:
    """Read secure records, with a legacy fallback for pre-migration databases."""
    secure_params = {"select": _API_KEY_COLUMNS, "limit": 1}
    if owner is not None:
        secure_params["owner"] = f"eq.{owner}"
    elif api_key is not None:
        secure_params["api_key_hash"] = f"eq.{hash_api_key(api_key)}"
    else:
        return []
    try:
        return _request("GET", params=secure_params)
    except requests.HTTPError:
        legacy_params = {"select": "user_id,owner,api_key", "limit": 1}
        if owner is not None:
            legacy_params["owner"] = f"eq.{owner}"
        else:
            legacy_params["api_key"] = f"eq.{api_key}"
        return _request("GET", params=legacy_params)


def _secure_record(row: dict) -> dict:
    encrypted = row.get("api_key_encrypted")
    key_hash = row.get("api_key_hash")
    legacy = row.get("api_key")
    if encrypted and key_hash:
        result = dict(row)
        result["api_key"] = decrypt_api_key(encrypted)
        return result
    if not legacy:
        return dict(row)

    result = dict(row)
    result["api_key"] = legacy
    user_pk = row.get("user_id")
    if user_pk is not None:
        try:
            response = _request(
                "PATCH",
                params={"user_id": f"eq.{user_pk}"},
                json={
                    "api_key_hash": hash_api_key(legacy),
                    "api_key_encrypted": encrypt_api_key(legacy),
                    "api_key": None,
                },
                prefer="return=minimal",
            )
            result.update({"api_key_hash": hash_api_key(legacy), "api_key_encrypted": encrypt_api_key(legacy), "api_key": None})
            result["api_key"] = legacy
        except (requests.RequestException, RuntimeError):
            pass
    return result


def _insert_api_key(*, owner: str, api_key: str) -> dict:
    secure_payload = {
        "owner": owner,
        "api_key": None,
        "api_key_hash": hash_api_key(api_key),
        "api_key_encrypted": encrypt_api_key(api_key),
    }
    try:
        data = _request("POST", json=secure_payload, prefer="return=representation")
        row = data[0] if data else dict(secure_payload)
        row["api_key"] = api_key
        return row
    except requests.HTTPError as secure_error:
        # Compatibility only for databases where the new columns have not yet
        # been created. Do not hide errors caused by constraints on a migrated DB.
        status = secure_error.response.status_code if secure_error.response is not None else 0
        if status not in (400, 404, 406):
            raise
        data = _request("POST", json={"owner": owner, "api_key": api_key}, prefer="return=representation")
        row = data[0] if data else {"owner": owner, "api_key": api_key}
        return _secure_record(row)


def upsert_api_key_db(*, user_id=None, owner: str, api_key: str) -> dict:
    """Store a key as hash + encrypted ciphertext, never plaintext on a migrated DB."""
    existing = get_api_key_db(owner=owner, create_if_missing=False)
    secure_update = {
        "owner": owner,
        "api_key": None,
        "api_key_hash": hash_api_key(api_key),
        "api_key_encrypted": encrypt_api_key(api_key),
    }
    if existing and existing.get("user_id") is not None:
        try:
            data = _request(
                "PATCH",
                params={"user_id": f"eq.{existing['user_id']}"},
                json=secure_update,
                prefer="return=representation",
            )
            row = data[0] if data else {**existing, **secure_update}
            row["api_key"] = api_key
            return row
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            if status not in (400, 404, 406) or not existing.get("api_key"):
                raise
            data = _request("PATCH", params={"user_id": f"eq.{existing['user_id']}"}, json={"api_key": api_key}, prefer="return=representation")
            row = data[0] if data else {**existing, "api_key": api_key}
            return row
    return _insert_api_key(owner=owner, api_key=api_key)


def get_api_key_db(*, owner: str | None = None, api_key: str | None = None, create_if_missing: bool = False) -> dict | None:
    if owner is None and api_key is None:
        return None
    data = _select_secure_or_legacy(owner=owner, api_key=api_key)
    if data:
        row = _secure_record(data[0])
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
