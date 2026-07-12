"""
api_val.py
----------
Handles the API key prompt. Stores the user-entered key in module-level
`key` so model_select can read it lazily at inference time.
"""


from .utils import clear_console, API_title, query
from dotenv import load_dotenv
from pathlib import Path
import os
 

key: str = ""
 
 
def _load_env() -> None:
    candidates = [
        Path.cwd() / ".env",
        Path.home() / ".jobselect" / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    for path in candidates:
        if path.exists():
            load_dotenv(dotenv_path=path, override=False)
            return
 
 
def val_api() -> str:
    """Prompt user for API key exactly once. Stores result in module-level `key`."""
    clear_console()
    API_title()
    query("Enter API Key (Press Enter to run locally)")
 
    global key
    key = input(" >> ").strip()
    return key
 
 
def infer_mode() -> str:
    """
    Return 'API' or 'LOCAL'.
    Call after val_api() has run.
    Checks typed key first, then .env fallback.
    """
    if key:
        return "API"
    _load_env()
    env_key = os.getenv("JOBSELECT_API_KEY", "").strip()
    return "API" if env_key else "LOCAL"