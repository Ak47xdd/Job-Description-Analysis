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

<<<<<<< HEAD
API_URL  = os.getenv("JOBSELECT_API_URL", "")
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
=======
API_URL = os.getenv("JOBSELECT_API_URL", "").rstrip("/")
API_KEY = os.getenv("JOBSELECT_API_KEY", "")               

from model.pred import JobAnalyze_6k as _local_predict

def _call_api(jd: str, role: str, job_type: str) -> list[tuple[str, float]]:
>>>>>>> parent of aaf29e0 (CLI)
    endpoint = f"{API_URL}/JobAnalyze_6k"
    headers  = {"JobAnalyze_6k_Key": API_KEY}
    payload  = {"Job_Desc": jd, "Role": role, "Type": job_type}
<<<<<<< HEAD
    
    response = requests.post(endpoint, json=payload, headers=headers, timeout=(10, 120))
    response.raise_for_status()
    
=======

    response = requests.post(endpoint, json=payload, headers=headers, timeout=(10, 120))
    response.raise_for_status()   

>>>>>>> parent of aaf29e0 (CLI)
    data = response.json()
    
    return [(skill, float(score)) for skill, score in data["answer"]]


def predict(jd: str, role: str, job_type: str) -> tuple[list[tuple[str, float]], str]:
<<<<<<< HEAD
    api_key = _resolve_key()
    
    if API_URL and api_key:
=======
    if API_URL and API_KEY:
>>>>>>> parent of aaf29e0 (CLI)
        try:
            results = _call_api(jd, role, job_type)
            return results, "API"
        except requests.exceptions.ConnectionError as e:
            print(f"Connection Error: {e}")
            raise

        except requests.exceptions.Timeout as e:
            print(f"Timeout: {e}")
            raise

        except requests.exceptions.HTTPError as e:
            print(e.response.status_code)
            print(e.response.text)
            raise

        except Exception as e:
<<<<<<< HEAD
            print(f"[JobSelect] Unexpected error: {e} — falling back to LOCAL")
=======
            import traceback
            traceback.print_exc()
            raise
>>>>>>> parent of aaf29e0 (CLI)

    results = _local_predict(jd, role=role, job_type=job_type)
    return results, "LOCAL"
