"""
api_val.py
----------
Handles the API key prompt. Stores the user-entered key in module-level
`key` so model_select can read it lazily at inference time.
"""


from .utils import clear_console, API_title, query
from .main_screen import main_screen
from rich import print
from dotenv import load_dotenv
from pathlib import Path
import os
 

_API_URL = "https://job-description-analysis.onrender.com"
 
_ENV_DIR  = Path.home() / ".jobselect"
_ENV_FILE = _ENV_DIR / ".env"
 
key: str = ""
 
 
def _load_env() -> None:
    """Load .env from user config dir if it exists."""
    if _ENV_FILE.exists():
        load_dotenv(dotenv_path=_ENV_FILE, override=False)
 
 
def _save_key(api_key: str) -> None:
    """
    Persist the user-entered key to ~/.jobselect/.env.
    Creates the directory if it doesn't exist.
    Writes both JOBSELECT_API_URL and JOBSELECT_API_KEY so
    model_select can find both on the next run without any prompt.
    """
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
    """Return the key from ~/.jobselect/.env if already saved, else ''."""
    _load_env()
    return os.getenv("JOBSELECT_API_KEY", "").strip()
 
def query_api() -> str:
    clear_console()
    API_title()
    query("Which Mode?  \n[yellow]- API : A\n[yellow]- LOCAL : L")
    mode = input(" >> ").upper()
    
    return mode
 
def val_api() -> str:
    """
    Prompt user for API key.
 
    - If a key is already saved in ~/.jobselect/.env, skip the prompt
      entirely and use the saved key silently.
    - If the user types a new key, save it to ~/.jobselect/.env for
      future runs so they never need to type it again.
    - If the user presses Enter (no key), run in LOCAL mode.
 
    Stores the resolved key in module-level `key`.
    """
    global key
    
    mode = query_api()
    
    saved = _load_saved_key()
    if saved and mode == "A":
        key = saved
        return key
    elif mode == "L":
        key = ""
        return key 
        
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