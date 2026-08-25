from fastapi import APIRouter
from fastapi import Depends

from schemas import JobOpeningCreate
from auth import require_admin


router = APIRouter(
    prefix="/careers",
    tags=["items"]
)

@router.post(
    "/careers/openings", 
    status_code=201, 
    operation_id="create_job_opening"
    )
async def create_opening(data: JobOpeningCreate, api_client: dict = Depends(require_admin)) -> dict:
    
    from supabase_client import supabase
    res = supabase.table("job_openings").insert({"title": data.title, "department": data.department, "type": data.type, "location": data.location, "description": data.description, "requirements": data.requirements, "tags": data.tags, "is_open": True}).execute(); return res.data[0] if res.data else {}

@router.patch(
    "/careers/openings/{job_id}/close", 
    operation_id="close_job_opening"
    )
async def close_opening(job_id: str, api_client: dict = Depends(require_admin)) -> dict:
    
    from supabase_client import supabase
    supabase.table("job_openings").update({"is_open": False}).eq("id", job_id).execute(); return {"closed": True, "id": job_id}

@router.get(
    "/careers/applications", 
    operation_id="list_applications"
    )
async def list_applications(
    job_id: str | None = None, 
    status: str | None = None, 
    api_client: dict = Depends(require_admin)
    ) -> dict:
    
    from supabase_client import supabase
    query = supabase.table("job_applications").select("id, job_id, job_title, name, email, linkedin_url, status, created_at")
    if job_id: query = query.eq("job_id", job_id)
    if status: query = query.eq("status", status)
    res = query.order("created_at", desc=True).execute(); return {"applications": res.data or []}