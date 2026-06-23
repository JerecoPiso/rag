from fastapi import APIRouter
from app.controllers.rag_controller import RAGController
from app.schemas.rag import RAGResponse, IngestResponse, SearchResponse, VectorRAGResponse, SyncResponse

router = APIRouter(prefix="/rag", tags=["RAG"])

router.post("/ask",        response_model=RAGResponse)(RAGController.ask)
router.post("/ingest",     response_model=IngestResponse)(RAGController.ingest)
router.post("/search",     response_model=SearchResponse)(RAGController.search)
router.post("/ask-vector", response_model=VectorRAGResponse)(RAGController.ask_vector)
router.post("/sync",       response_model=SyncResponse)(RAGController.sync)

