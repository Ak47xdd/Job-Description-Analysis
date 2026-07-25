"""
supabase_client.py - Main client for the supabase integration for queries to postgres
Uses direct REST API requests to bypass RLS policies.
"""

from pathlib import Path
from dotenv import load_dotenv
import sys
import os
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

_HEADERS = {
    "apikey": SUPA_KEY,
    "Content-Type": "application/json",
    "Accept": "application/json",
}


def upsert_api_key_db(*, user_id: str, owner: str, api_key: str) -> dict:
    """Upsert an API key into the api_tok table via REST (bypasses RLS)."""
    payload = {"user_id": user_id, "owner": owner, "api_key": api_key}

    resp = requests.post(
        f"{SUPA_URL}/rest/v1/api_tok?on_conflict=owner",
        headers={
            **_HEADERS, "Prefer": "resolution=merge-duplicates,return=representation"
            },
        json=payload,
    )
    resp.raise_for_status()
    data = resp.json()
    return data[0] if data else {}


def get_api_key_db(*, owner: str | None = None, api_key: str | None = None) -> dict | None:
    """Retrieve an API key record from the api_tok table via REST (bypasses RLS)."""
    if owner is None and api_key is None:
        return None

    params = {"select": "user_id,owner,api_key", "limit": 1}

    if owner is not None:
        params["owner"] = f"eq.{owner}"
    elif api_key is not None:
        params["api_key"] = f"eq.{api_key}"

    resp = requests.get(
        f"{SUPA_URL}/rest/v1/api_tok",
        headers=_HEADERS,
        params=params,
    )
    resp.raise_for_status()
    data = resp.json()
    return data[0] if data else None


