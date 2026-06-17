from fastapi import APIRouter
from app.controllers.rag_controller import RAGController
from app.schemas.rag import RAGResponse

router = APIRouter(prefix="/rag", tags=["RAG"])

router.post("/ask", response_model=RAGResponse)(RAGController.ask)

