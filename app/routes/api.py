from fastapi import APIRouter
from app.routes.user_routes import router as user_router
from app.routes.rag_routes import router as rag_router

router = APIRouter()

router.include_router(user_router)
router.include_router(rag_router)