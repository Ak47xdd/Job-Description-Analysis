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
    """Upsert an API key record into api_tok.

    Table schema expectation:
      - user_id: BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY
      - owner: VARCHAR(50)
      - api_key: VARCHAR(100)

    Note:
      The project currently uses `user_id` as the hashed API key.
      This function inserts/updates using the provided `user_id`.
    """
    # user_id column is defined as GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    # so we must not include it in INSERT/UPSERT payload.
    # We store the hashed key in `api_key` and use it for lookup.
    payload = {"owner": owner, "api_key": api_key}


    resp = (
        supabase.table("api_tok")
        .upsert(payload)
        .execute()
    )
    data = getattr(resp, "data", None)
    return data[0] if data else {}


def get_api_key_db(*, user_id: str | None = None, api_key: str | None = None) -> dict | None:
    """Fetch API key record from api_tok.

    Primary lookup is by user_id (identity column). If user_id is not provided,
    we first find the matching row by api_key (exact match), then return it.
    """

    if user_id is not None:
        resp = (
            supabase.table("api_tok")
            .select("user_id", "owner", "api_key")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        data = getattr(resp, "data", None) or []
        return data[0] if data else None

    if api_key is not None:
        # Identity columns are not used for filtering here; we filter by api_key.
        # This finds the row and returns its user_id/owner/api_key.
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


