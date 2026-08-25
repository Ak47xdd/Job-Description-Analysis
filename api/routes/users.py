from fastapi import APIRouter

from routes.auth import router as auth_router
from routes.oauth import router as oauth_router
from routes.account import router as account_router

router = APIRouter()
router.include_router(auth_router)
router.include_router(oauth_router)
router.include_router(account_router)
