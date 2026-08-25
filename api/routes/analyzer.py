from fastapi import Depends, HTTPException
from starlette.requests import Request
from fastapi import APIRouter
import traceback

from JobAnalyze.v1.pred_v1 import JobAnalyze_6k
from JobAnalyze_API import limiter
from helpers import _build_analysis
from schemas import ModelRequest
from auth import verify


router = APIRouter(
    prefix="/analyzer",
    tags=["items"]
)

MAX_JD_LENGTH = 30000

@router.post(
    "/web_analyze", 
    operation_id="web_analyze"
    )
@limiter.limit("5/minute")
async def web_analyze(request: Request, data: ModelRequest) -> dict:
    
    if len(data.Job_Desc) > MAX_JD_LENGTH: raise HTTPException(status_code=413, detail="Job description is too large.")
    predicted = [(skill, float(score)) for skill, score in JobAnalyze_6k(job_desc=data.Job_Desc, role=data.Role, job_type=data.Type)]
    try: return _build_analysis(predicted=predicted, role=data.Role, job_type=data.Type, jd_text=data.Job_Desc)
    except Exception:
        traceback.print_exc(); raise HTTPException(status_code=500, detail="Web Analyzer failed while building the analysis response.")

@router.post(
    "/JobAnalyze_6k", 
    operation_id="analyze_job_description"
    )
@limiter.limit("10/minute")
async def JobAnalyze_Pred(request: Request, data: ModelRequest, api_client: dict = Depends(verify)) -> dict:
    
    if len(data.Job_Desc) > MAX_JD_LENGTH: raise HTTPException(status_code=413, detail="Job description is too large.")
    predicted = [(skill, float(score)) for skill, score in JobAnalyze_6k(job_desc=data.Job_Desc, role=data.Role, job_type=data.Type)]
    return {"answer": predicted, "analysis": _build_analysis(predicted=predicted, role=data.Role, job_type=data.Type, jd_text=data.Job_Desc)}