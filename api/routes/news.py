from fastapi import APIRouter

router = APIRouter(
    prefix="/news",
    tags=["items"]
)