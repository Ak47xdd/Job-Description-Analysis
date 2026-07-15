"""
api_val.py - Handles the API key prompt. Stores the user-entered key in module-level
`key` so model_select can read it lazily at inference time.
"""

from rich import print
from dotenv import load_dotenv
from pathlib import Path
import os

from .utils import clear_console, API_title, query


_API_URL = "https://job-description-analysis.onrender.com"
 
_ENV_DIR  = Path.home() / ".jobselect"
_ENV_FILE = _ENV_DIR / ".env"
 
key: str = ""
 
 
def _load_env() -> None:
    if _ENV_FILE.exists():
        load_dotenv(dotenv_path=_ENV_FILE, override=False)
 
 
def _save_key(api_key: str) -> None:
    try:
        _ENV_DIR.mkdir(parents=True, exist_ok=True)
        _ENV_FILE.write_text(
            f'JOBSELECT_API_URL="{_API_URL}"\n'
            f'JOBSELECT_API_KEY="{api_key}"\n',
            encoding="utf-8",
        )
    except OSError as e:
        print(f"[JobSelect] Warning: could not save key ({e})")
 
 
def _load_saved_key() -> str:
    _load_env()
    return os.getenv("JOBSELECT_API_KEY", "").strip()
 
def query_api() -> str:
    clear_console()
    API_title()
    query("Which Mode?  \n[yellow]- API : A\n[yellow]- LOCAL : L")
    mode = input(" >> ").upper()
    
    return mode
 
def val_api() -> str:
    global key
    
    mode = query_api()
    
    saved = _load_saved_key()
    if saved and mode == "A":
        key = saved
        return key
    elif mode == "L":
        key = ""
        return key 
    
    modes = ('A','L')
    
    if mode not in modes:
        mode = query_api()
              
    clear_console()
    API_title()
    query("Enter API Key (Press Enter to run locally)")
    typed = input(" >> ").strip()
    
    if typed:
        _save_key(typed)
        os.environ["JOBSELECT_API_URL"] = _API_URL
        os.environ["JOBSELECT_API_KEY"] = typed
        key = typed
    else:
        key = ""
 
    return key
 
 
def infer_mode() -> str:
    """Return 'API' or 'LOCAL'. Call after val_api() has run."""
    if key:
        return "API"
    return "LOCAL"