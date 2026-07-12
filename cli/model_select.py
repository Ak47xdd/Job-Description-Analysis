"""
model_select.py — inference routing (API first, local fallback).
Never prompts the user. Key is resolved lazily at predict() call time.
"""

from pathlib import Path
from dotenv import load_dotenv
import os
import requests
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_env_path = Path(__file__).resolve().parent / ".env"
if not _env_path.exists():
    _env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=_env_path)

API_URL  = os.getenv("JOBSELECT_API_URL", "").rstrip("/")
_ENV_KEY = os.getenv("JOBSELECT_API_KEY", "").strip()

def _resolve_key() -> str:
    """
    Lazy key resolution at inference time (not import time).
    Priority: user-typed key (from api_val) > .env key.
    """
    try:
        from . import api_val
        if api_val.key:
            return api_val.key
    except ImportError:
        try:
            import api_val  
            if api_val.key:
                return api_val.key
        except ImportError:
            pass
    return _ENV_KEY
 

def _local_predict(jd: str, role: str, job_type: str):
    from model.pred import JobAnalyze_6k as _job_analyze
    return _job_analyze(jd, role=role, job_type=job_type)


def _call_api(jd: str, role: str, job_type: str, api_key: str) -> list[tuple[str, float]]:
    endpoint = f"{API_URL}/JobAnalyze_6k"
    headers  = {"JobAnalyze_6k_Key": api_key}
    payload  = {"Job_Desc": jd, "Role": role, "Type": job_type}
    
    response = requests.post(endpoint, json=payload, headers=headers, timeout=(10, 120))
    response.raise_for_status()
    
    data = response.json()
    return [(skill, float(score)) for skill, score in data["answer"]]


def predict(jd: str, role: str, job_type: str) -> tuple[list[tuple[str, float]], str]:
    api_key = _resolve_key()
    
    if API_URL and api_key:
        try:
            results = _call_api(jd, role, job_type, api_key)
            return results, "API"
        except requests.exceptions.ConnectionError as e:
            print(f"[JobSelect] Connection error: {e} — falling back to LOCAL")
        except requests.exceptions.Timeout:
            print("[JobSelect] Request timed out — falling back to LOCAL")
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else "?"
            print(f"[JobSelect] API error {code} — falling back to LOCAL")
        except Exception as e:
            print(f"[JobSelect] Unexpected error: {e} — falling back to LOCAL")

    results = _local_predict(jd, role=role, job_type=job_type)
    return results, "LOCAL"