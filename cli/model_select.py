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

API_URL = os.getenv("JOBAUTO_API_URL", "").rstrip("/")
API_KEY = os.getenv("JOBAUTO_API_KEY", "")               

from model.pred import JobAnalyze_6k as _local_predict

def _call_api(jd: str, role: str, job_type: str) -> list[tuple[str, float]]:
    endpoint = f"{API_URL}/JobAnalyze_6k"
    headers  = {"JobAnalyze_6k_Key": API_KEY}
    payload  = {"Job_Desc": jd, "Role": role, "Type": job_type}

    response = requests.post(endpoint, json=payload, headers=headers, timeout=10)
    response.raise_for_status()   

    data = response.json()
    
    return [(skill, float(score)) for skill, score in data["answer"]]


def predict(jd: str, role: str, job_type: str) -> tuple[list[tuple[str, float]], str]:
    if API_URL and API_KEY:
        try:
            results = _call_api(jd, role, job_type)
            return results, "API"
        except requests.exceptions.ConnectionError:
            pass
        except requests.exceptions.Timeout:
            pass 
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code in (401, 403):
                print(f"[JobSelect] API key issue ({e.response.status_code}) - running in LOCAL mode")
        except Exception:
            pass 

    results = _local_predict(jd, role=role, job_type=job_type)
    return results, "LOCAL"