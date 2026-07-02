import base64
import uuid as _uuid
from fastapi import Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.services.rag_service import RAGService
from app.services.vector_service import VectorService
from app.services.vector_service_v2 import VectorServiceV2
from app.services.speech_service import SpeechService
from app.schemas.rag import RAGRequest, IngestRequest, SearchRequest, VectorRAGRequest, SyncRequest

# In-memory session store: conversation_id → {"active_patient": "..."}
_sessions: dict[str, dict] = {}


class RAGController:
    @staticmethod
    def ask(data: RAGRequest, db: Session = Depends(get_db)):
        result = RAGService(db, provider=data.provider).ask(data.question)

        audio = None
        if result.get("answer"):
            try:
                audio_bytes, media_type = SpeechService().synthesize(result["answer"])
                encoded = base64.b64encode(audio_bytes).decode("ascii")
                audio = f"data:{media_type};base64,{encoded}"
            except Exception:
                pass  # Text answer still stands even if speech synthesis fails.

        return {**result, "audio": audio}

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

        svc     = VectorServiceV2(collection_name=data.collection)
        history = [{"role": m.role, "content": m.content} for m in data.history]
        result  = svc.ask(
            data.question,
            provider=data.provider,
            history=history,
            session=session,
            db=db,
            sql_provider=data.sql_provider or "openai",
        )

        audio = None


        if result.get("answer"):
            try:
                audio_bytes, media_type = SpeechService().synthesize(result["answer"])
                encoded = base64.b64encode(audio_bytes).decode("ascii")
                audio = f"data:{media_type};base64,{encoded}"
            except Exception:
                pass  # Text answer still stands even if speech synthesis fails.

        return {**result, "source": result.get("source", "vector"), "conversation_id": conv_id, "audio": audio}

    @staticmethod
    def sync(data: SyncRequest, db: Session = Depends(get_db)):
        svc = VectorServiceV2(collection_name=data.collection)
        return svc.sync_from_db(db, clear=data.clear)
