"""Supabase integration and API-key persistence."""

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


def _request(params: dict) -> list[dict]:
    response = requests.get(
        f"{SUPA_URL}/rest/v1/api_tok",
        headers=_HEADERS,
        params=params,
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def _select_secure_or_legacy(*, owner: str | None = None, api_key: str | None = None) -> list[dict]:
    """Read secure columns, then cleanly fall back to the original schema."""
    secure_params = {"select": _API_KEY_COLUMNS, "limit": 1}
    if owner is not None:
        secure_params["owner"] = f"eq.{owner}"
    elif api_key is not None:
        secure_params["api_key_hash"] = f"eq.{hash_api_key(api_key)}"
    else:
        return []

    try:
        return _request(secure_params)
    except requests.HTTPError:
        # The secure columns may not exist yet. Crucially, do not retain the
        # api_key_hash filter in the legacy query, because that column may also
        # be absent. Search the actual plaintext column only in this temporary
        # compatibility path.
        legacy_params = {"select": "user_id,owner,api_key", "limit": 1}
        if owner is not None:
            legacy_params["owner"] = f"eq.{owner}"
        else:
            legacy_params["api_key"] = f"eq.{api_key}"
        return _request(legacy_params)


def _secure_record(row: dict) -> dict:
    """Normalize a secure or legacy row and lazily migrate legacy keys."""
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

    # Migration is best-effort. A missing migration must never prevent an
    # existing user from signing in or using their old API key.
    user_pk = row.get("user_id")
    if user_pk is not None:
        try:
            secure_update = {
                "api_key_hash": hash_api_key(legacy),
                "api_key_encrypted": encrypt_api_key(legacy),
                "api_key": None,
            }
            response = requests.patch(
                f"{SUPA_URL}/rest/v1/api_tok",
                params={"user_id": f"eq.{user_pk}"},
                headers={**_HEADERS, "Prefer": "return=minimal"},
                json=secure_update,
                timeout=15,
            )
            response.raise_for_status()
            result.update(secure_update)
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
        response = requests.post(
            f"{SUPA_URL}/rest/v1/api_tok",
            headers={**_HEADERS, "Prefer": "return=representation"},
            json=secure_payload,
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        row = data[0] if data else dict(secure_payload)
        row["api_key"] = api_key
        return row
    except requests.HTTPError:
        legacy_payload = {"owner": owner, "api_key": api_key}
        response = requests.post(
            f"{SUPA_URL}/rest/v1/api_tok",
            headers={**_HEADERS, "Prefer": "return=representation"},
            json=legacy_payload,
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        row = data[0] if data else legacy_payload
        return _secure_record(row)


def upsert_api_key_db(*, user_id=None, owner: str, api_key: str) -> dict:
    """Create/update an API key while supporting both DB schemas."""
    existing = get_api_key_db(owner=owner, create_if_missing=False)
    secure_update = {
        "owner": owner,
        "api_key": None,
        "api_key_hash": hash_api_key(api_key),
        "api_key_encrypted": encrypt_api_key(api_key),
    }

    if existing and existing.get("user_id") is not None:
        try:
            response = requests.patch(
                f"{SUPA_URL}/rest/v1/api_tok",
                params={"user_id": f"eq.{existing['user_id']}"},
                headers={**_HEADERS, "Prefer": "return=representation"},
                json=secure_update,
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
            row = data[0] if data else {**existing, **secure_update}
            row["api_key"] = api_key
            return row
        except requests.HTTPError:
            # Secure columns are not available yet. Preserve the old record.
            if existing.get("api_key"):
                return existing
            raise

    return _insert_api_key(owner=owner, api_key=api_key)


def get_api_key_db(*, owner: str | None = None, api_key: str | None = None, create_if_missing: bool = False) -> dict | None:
    """Find an API key by owner or candidate key."""
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
