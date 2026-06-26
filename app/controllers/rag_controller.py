import uuid as _uuid
from fastapi import Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.services.rag_service import RAGService
from app.services.vector_service import VectorService
from app.schemas.rag import RAGRequest, IngestRequest, SearchRequest, VectorRAGRequest, SyncRequest

# In-memory session store: conversation_id → {"active_patient": "..."}
_sessions: dict[str, dict] = {}


class RAGController:
    @staticmethod
    def ask(data: RAGRequest, db: Session = Depends(get_db)):
        return RAGService(db, provider=data.provider).ask(data.question)

    @staticmethod
    def ingest(data: IngestRequest):
        svc = VectorService(collection_name=data.collection)
        count = svc.ingest(data.texts, data.metadatas)
        return {"ingested": count, "collection": data.collection}

    @staticmethod
    def search(data: SearchRequest):
        svc = VectorService(collection_name=data.collection)
        results = svc.search(data.query, top_k=data.top_k)
        return {"query": data.query, "results": results}

    @staticmethod
    def ask_vector(data: VectorRAGRequest, db: Session = Depends(get_db)):
        conv_id = data.conversation_id or str(_uuid.uuid4())
        if conv_id not in _sessions:
            _sessions[conv_id] = {}
        session = _sessions[conv_id]

        svc     = VectorService(collection_name=data.collection)
        history = [{"role": m.role, "content": m.content} for m in data.history]
        result  = svc.ask(data.question, provider=data.provider, history=history, session=session)

        return {**result, "source": "vector", "conversation_id": conv_id}

    @staticmethod
    def sync(data: SyncRequest, db: Session = Depends(get_db)):
        svc = VectorService(collection_name=data.collection)
        return svc.sync_from_db(db, clear=data.clear)
