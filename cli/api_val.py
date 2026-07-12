"""
api_val.py
----------
Handles the API key prompt. Stores the user-entered key in module-level
`key` so model_select can read it lazily at inference time.
 
No imports from model_select or api_logic — circular import is gone.
"""


import os

from .utils import clear_console, API_title, query
 

key: str = ""

 
def val_api() -> str:
    clear_console()
    API_title()
    query("Enter API Key (Press Enter to run locally)")
 
    global key
    key = input(" >> ").strip()
    return key
 
 
def infer_mode() -> str:
    if key:
        return "API"
    env_key = os.getenv("JOBSELECT_API_KEY", "").strip()
    return "API" if env_key else "LOCAL"