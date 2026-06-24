from fastapi import APIRouter
from app.routes.user_routes import router as user_router
from app.routes.rag_routes import router as rag_router
from app.routes.transcription_routes import router as transcription_router

router = APIRouter()

router.include_router(user_router)
router.include_router(rag_router)
router.include_router(transcription_router)