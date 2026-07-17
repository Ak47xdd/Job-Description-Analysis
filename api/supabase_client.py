"""
supabase_client.py - Main client for the supabase integration for queries to postgres
"""

from pathlib import Path
from dotenv import load_dotenv
import sys
import os
from supabase import create_client, Client

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_env_path = Path(__file__).resolve().parent / ".env"
if not _env_path.exists():
    _env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=_env_path)

SUPA_URL = os.getenv("SUPA_URL", "")
SUPA_KEY = os.getenv("SUPA_KEY", "")

supabase: Client = create_client(SUPA_URL, SUPA_KEY)

def upsert_api_key_db(*, user_id: str, owner: str, api_key: str) -> dict:
    payload = {"owner": owner, "api_key": api_key}

    resp = (
        supabase.table("api_tok")
        .upsert(payload)
        .execute()
    )
    data = getattr(resp, "data", None)
    return data[0] if data else {}


def get_api_key_db(*, owner: str) -> dict | None:
    if owner is not None:
        resp = (
            supabase.table("api_tok")
            .select("owner", "api_key")
            .eq("owner", owner)
            .limit(1)
            .execute()
        )
        data = getattr(resp, "data", None) or []
        return data[0] if data else None

    if api_key is not None:
        resp = (
            supabase.table("api_tok")
            .select("user_id", "owner", "api_key")
            .eq("api_key", api_key)
            .limit(1)
            .execute()
        )
        data = getattr(resp, "data", None) or []
        return data[0] if data else None

    return None


