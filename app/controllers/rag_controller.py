import base64
import re
import time
import uuid as _uuid
from fastapi import Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.services.rag_service import RAGService
from app.services.vector_service import VectorService
from app.services.speech_service import SpeechService
from app.schemas.rag import RAGRequest, IngestRequest, SearchRequest, VectorRAGRequest, SyncRequest, SyncLatestRequest

# In-memory session store: conversation_id → {"active_patient": "..."}
_sessions: dict[str, dict] = {}


# VectorService.__init__ connects to Qdrant and checks that every collection
# exists (~1.7s per call, measured), so constructing a fresh instance on every
# request was the dominant cost even for a plain "hi". Collections don't
# change between requests, so one instance per collection_name is reused for
# the life of the process instead.
_vector_services: dict[str, VectorService] = {}


def _get_vector_service(collection_name: str) -> VectorService:
    svc = _vector_services.get(collection_name)
    if svc is None:
        svc = VectorService(collection_name=collection_name)
        _vector_services[collection_name] = svc
    return svc

# Questions asking for a count/total of admitted, outpatient, ER, or discharged
# patients are answered more reliably by RAGService's text-to-SQL path (which
# knows about patient_status / discharge_type) than by the vector search path.
_COUNT_PATTERN  = re.compile(r"\b(?:count|how many|number of|total)\b", re.IGNORECASE)
_STATUS_PATTERN = re.compile(
    r"\b(?:admit(?:ted)?|outpatient|opd|inpatient|inp|er|emergency|discharg\w*"
    r"|medicine||medical|pedia\w*|\bob\b|newborn|new born|surger\w*|classification|patient type)\b",
    re.IGNORECASE,
)
_LATEST_PATTERN = re.compile(
    r"\b(?:latest|last|most recent|newest|current|recent)\b",
    re.IGNORECASE,
)


class RAGController:
    @staticmethod
    def _is_status_count_question(question: str) -> bool:
        return bool(_COUNT_PATTERN.search(question) and _STATUS_PATTERN.search(question))

    @staticmethod
    def _is_latest_question(question: str) -> bool:
        return bool(_LATEST_PATTERN.search(question))
    
    @staticmethod
    def ask(data: RAGRequest, db: Session = Depends(get_db)):
        print(f"Provider: {data.provider}, Question: {data.question}")

        result = RAGService(db, provider=data.provider).ask(data.question)
        audio = None
        if result.get("answer"):
            try:
                audio_bytes, media_type = SpeechService().synthesize(
                    result["answer"], provider=data.tts_provider
                )
                encoded = base64.b64encode(audio_bytes).decode("ascii")
                audio = f"data:{media_type};base64,{encoded}"
            except Exception:
                pass  # Text answer still stands even if speech synthesis fails.

        return {**result, "audio": audio}

    @staticmethod
    def warm_up():
        svc = _get_vector_service("patient_info")
        return svc.warm_up()

    @staticmethod
    def ingest(data: IngestRequest):
        svc = _get_vector_service(data.collection)
        count = svc.ingest(data.texts, data.metadatas)
        return {"ingested": count, "collection": data.collection}

    @staticmethod
    def search(data: SearchRequest):
        svc = _get_vector_service(data.collection)
        results = svc.search(data.query, top_k=data.top_k)
        return {"query": data.query, "results": results}

    @staticmethod
    def ask_vector(data: VectorRAGRequest, db: Session = Depends(get_db)):
        _t_req = time.perf_counter()
        conv_id = data.conversation_id or str(_uuid.uuid4())
        if conv_id not in _sessions:
            _sessions[conv_id] = {}
        session = _sessions[conv_id]

        history = [{"role": m.role, "content": m.content} for m in data.history]

        if RAGController._is_status_count_question(data.question) or RAGController._is_latest_question(data.question):
            sql_result = RAGService(db, provider=data.sql_provider or "openai").ask(data.question, history, session=session)
            result = {
                "question": sql_result["question"],
                "context":  [],
                "answer":   sql_result["answer"],
                "source":   "sql",
            }
        else:
            svc    = _get_vector_service(data.collection)
            result = svc.ask(
                data.question,
                provider=data.provider,
                history=history,
                session=session,
                db=db,
                sql_provider=data.sql_provider or "openai",
            )

        audio = None

        print(f"[TIMING] ask_vector answer took {time.perf_counter() - _t_req:.2f}s")
        _t_tts = time.perf_counter()
        if result.get("answer"):
            try:
                audio_bytes, media_type = SpeechService().synthesize(
                    result["answer"], provider=data.tts_provider
                )
                encoded = base64.b64encode(audio_bytes).decode("ascii")
                audio = f"data:{media_type};base64,{encoded}"
            except Exception:
                pass  # Text answer still stands even if speech synthesis fails.
        print(f"[TIMING] TTS synthesis took {time.perf_counter() - _t_tts:.2f}s")
        print(f"[TIMING] ask_vector total request took {time.perf_counter() - _t_req:.2f}s")

        return {**result, "source": result.get("source", "vector"), "conversation_id": conv_id, "audio": audio}

    @staticmethod
    def sync(data: SyncRequest, db: Session = Depends(get_db)):
        svc = _get_vector_service(data.collection)
        return svc.sync_from_db(db, clear=data.clear)

    @staticmethod
    def sync_latest(data: SyncLatestRequest, db: Session = Depends(get_db)):
        svc = _get_vector_service(data.collection)
        return svc.sync_latest(db, since=data.since)
