from fastapi import APIRouter
from app.routes.user_routes import router as user_router

router = APIRouter()

router.include_router(user_router)