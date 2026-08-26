"""Supabase integration and API-key persistence."""

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

# Keep api_key during the transition. The migration is deliberately lazy so
# existing accounts continue to work even before the SQL migration is run.
_API_KEY_COLUMNS = "user_id,owner,api_key,api_key_hash,api_key_encrypted"
_SECURE_COLUMNS = "user_id,owner,api_key_hash,api_key_encrypted"


def _generate_api_key() -> str:
    return f"ja6k_{secrets.token_hex(32)}"


def _select(params: dict) -> list[dict]:
    """Read api_tok, falling back to the legacy schema if new columns don't exist."""
    try:
        resp = requests.get(
            f"{SUPA_URL}/rest/v1/api_tok",
            headers=_HEADERS,
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as exc:
        # PostgREST returns 400/404 when a column does not exist. This fallback
        # prevents a missing migration from taking down signup/sign-in.
        if params.get("select") != _API_KEY_COLUMNS:
            raise
        legacy_params = dict(params)
        legacy_params["select"] = "user_id,owner,api_key"
        resp = requests.get(
            f"{SUPA_URL}/rest/v1/api_tok",
            headers=_HEADERS,
            params=legacy_params,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()


def _secure_record(row: dict) -> dict:
    """Return a usable record and migrate a legacy plaintext key when possible."""
    encrypted = row.get("api_key_encrypted")
    key_hash = row.get("api_key_hash")
    legacy = row.get("api_key")

    if encrypted and key_hash:
        result = dict(row)
        result["api_key"] = decrypt_api_key(encrypted)
        return result

    if legacy:
        result = dict(row)
        result["api_key"] = legacy
        secure_update = {
            "api_key_hash": hash_api_key(legacy),
            "api_key_encrypted": encrypt_api_key(legacy),
            "api_key": None,
        }
        user_pk = row.get("user_id")
        if user_pk is not None:
            try:
                resp = requests.patch(
                    f"{SUPA_URL}/rest/v1/api_tok",
                    params={"user_id": f"eq.{user_pk}"},
                    headers={**_HEADERS, "Prefer": "return=minimal"},
                    json=secure_update,
                    timeout=15,
                )
                resp.raise_for_status()
                result.update(secure_update)
            except requests.HTTPError:
                # The SQL migration may not have been applied yet. Do not break
                # existing authentication just because migration is unavailable.
                pass
        return result

    return dict(row)


def _insert_api_key(*, owner: str, api_key: str) -> dict:
    """Insert a new key using the secure schema, with legacy fallback."""
    secure_payload = {
        "owner": owner,
        "api_key": None,
        "api_key_hash": hash_api_key(api_key),
        "api_key_encrypted": encrypt_api_key(api_key),
    }
    try:
        resp = requests.post(
            f"{SUPA_URL}/rest/v1/api_tok",
            headers={**_HEADERS, "Prefer": "return=representation"},
            json=secure_payload,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        row = data[0] if data else dict(secure_payload)
        row["api_key"] = api_key
        return row
    except requests.HTTPError:
        # Compatibility path until the two new columns are present in Supabase.
        legacy_payload = {"owner": owner, "api_key": api_key}
        resp = requests.post(
            f"{SUPA_URL}/rest/v1/api_tok",
            headers={**_HEADERS, "Prefer": "return=representation"},
            json=legacy_payload,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        row = data[0] if data else legacy_payload
        return _secure_record(row)


def upsert_api_key_db(*, user_id=None, owner: str, api_key: str) -> dict:
    """Create/update an API key, preferring secure storage without breaking legacy DBs."""
    existing = get_api_key_db(owner=owner, create_if_missing=False)
    secure_update = {
        "owner": owner,
        "api_key": None,
        "api_key_hash": hash_api_key(api_key),
        "api_key_encrypted": encrypt_api_key(api_key),
    }

    if existing and existing.get("user_id") is not None:
        user_pk = existing["user_id"]
        try:
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
        except requests.HTTPError:
            # Missing secure columns: retain the existing legacy record.
            if existing.get("api_key"):
                return existing
            raise

    return _insert_api_key(owner=owner, api_key=api_key)


def get_api_key_db(
    *,
    owner: str | None = None,
    api_key: str | None = None,
    create_if_missing: bool = False,
) -> dict | None:
    """Find an API key by owner or candidate key."""
    if owner is None and api_key is None:
        return None

    if owner is not None:
        params = {
            "select": _API_KEY_COLUMNS,
            "owner": f"eq.{owner}",
            "limit": 1,
        }
    else:
        params = {
            "select": _API_KEY_COLUMNS,
            "api_key_hash": f"eq.{hash_api_key(api_key)}",
            "limit": 1,
        }

    data = _select(params)

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

    # Old plaintext rows can still authenticate and will be migrated lazily.
    if api_key is not None:
        legacy_params = {
            "select": "user_id,owner,api_key",
            "api_key": f"eq.{api_key}",
            "limit": 1,
        }
        legacy_data = _select(legacy_params)
        if legacy_data:
            return _secure_record(legacy_data[0])

    if owner is not None and create_if_missing:
        return _insert_api_key(owner=owner, api_key=_generate_api_key())

    return None
