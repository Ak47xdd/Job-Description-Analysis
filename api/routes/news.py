from fastapi import APIRouter
from fastapi import Depends

from schemas import NewsItemCreate
from auth import require_admin

router = APIRouter(
    prefix="/news",
    tags=["items"]
)

@router.post(
    "/news", 
    status_code=201, 
    operation_id="create_news_item"
    )
async def create_news_item(data: NewsItemCreate, api_client: dict = Depends(require_admin)) -> dict:
    
    from supabase_client import supabase
    res = supabase.table("news_items").insert({
        "title": data.title, 
        "summary": data.summary, 
        "category": data.category, 
        "url": data.url, 
        "body": data.body, 
        "is_published": data.is_published
        }).execute()
    return res.data[0] if res.data else {}

@router.patch(
    "/news/{item_id}/unpublish", 
    operation_id="unpublish_news_item"
    )
async def unpublish_news_item(item_id: str, api_client: dict = Depends(require_admin)) -> dict:
    
    from supabase_client import supabase
    supabase.table("news_items").update({"is_published": False}).eq("id", item_id).execute(); 
    return {"unpublished": True, "id": item_id}

@router.get(
    "/news", 
    operation_id="list_news_items"
    )
async def list_news_items(include_drafts: bool = False, api_client: dict = Depends(require_admin)) -> dict:
    
    from supabase_client import supabase
    query = supabase.table("news_items").select("*").order("published_at", desc=True)
    if not include_drafts: query = query.eq("is_published", True)
    res = query.execute() 
    return {"items": res.data or []}