"""
supabase_client.py - Main client for the Supabase integration.

Uses the Supabase REST API directly so API-key table operations can be made
with the backend service-role key without depending on the browser session or
Supabase client RLS context.
"""

from pathlib import Path
from dotenv import load_dotenv
import sys
import os
import secrets
import requests

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

_HEADERS = {
    "apikey": SUPA_KEY,
    "Authorization": f"Bearer {SUPA_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}


def _generate_api_key() -> str:
    return f"ja6k_{secrets.token_hex(32)}"


def _insert_api_key(*, owner: str, api_key: str) -> dict:
    """Insert a new API-key row into api_tok."""
    payload = {"owner": owner, "api_key": api_key}

    resp = requests.post(
        f"{SUPA_URL}/rest/v1/api_tok",
        headers={**_HEADERS, "Prefer": "return=representation"},
        json=payload,
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    return data[0] if data else {}


def upsert_api_key_db(*, user_id=None, owner: str, api_key: str) -> dict:
    """
    Create or update the API key for an owner.

    The api_tok schema uses user_id as its primary key, while owner is not
    guaranteed to have a UNIQUE constraint. Therefore using
    ?on_conflict=owner is not a valid/upsert-safe operation. We first look up
    the owner, then PATCH the existing row by its actual primary key, or INSERT
    a new row when none exists.
    """
    existing = get_api_key_db(owner=owner, create_if_missing=False)

    if existing and existing.get("user_id") is not None:
        user_pk = existing["user_id"]
        resp = requests.patch(
            f"{SUPA_URL}/rest/v1/api_tok",
            params={"user_id": f"eq.{user_pk}"},
            headers={**_HEADERS, "Prefer": "return=representation"},
            json={"owner": owner, "api_key": api_key},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return data[0] if data else {**existing, "owner": owner, "api_key": api_key}

    return _insert_api_key(owner=owner, api_key=api_key)


def get_api_key_db(
    *,
    owner: str | None = None,
    api_key: str | None = None,
    create_if_missing: bool = False,
) -> dict | None:
    """
    Retrieve an API-key record.

    When called for an authenticated owner with create_if_missing=True, a key
    is provisioned for legacy Supabase users who existed before api_tok was
    populated. This fixes sign-in for those accounts without changing the
    behavior of API-key verification lookups.
    """
    if owner is None and api_key is None:
        return None

    params = {"select": "user_id,owner,api_key", "limit": 1}

    if owner is not None:
        params["owner"] = f"eq.{owner}"
    else:
        params["api_key"] = f"eq.{api_key}"

    resp = requests.get(
        f"{SUPA_URL}/rest/v1/api_tok",
        headers=_HEADERS,
        params=params,
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    if data:
        return data[0]

    # Only authenticated-owner lookups can provision a missing key.
    # api_key lookups must never create anything.
    if owner is not None and create_if_missing:
        raw = _generate_api_key()
        return _insert_api_key(owner=owner, api_key=raw)

    return None
