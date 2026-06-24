from fastapi import Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.services.rag_service import RAGService
from app.services.vector_service import VectorService
from app.schemas.rag import RAGRequest, IngestRequest, SearchRequest, VectorRAGRequest, SyncRequest


class RAGController:
    @staticmethod
    # , _: User = Depends(get_current_user)
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
        svc = VectorService(collection_name=data.collection)
        history = [{"role": m.role, "content": m.content} for m in data.history]
        result = svc.ask(data.question, provider=data.provider, history=history)

        # if not result.get("context_sufficient", True):
        #     sql_provider = data.sql_provider or data.provider
        #     sql_result = RAGService(db, provider=sql_provider).ask(data.question)
        #     return {
        #         **result,
        #         "answer": sql_result["answer"],
        #         "sql": sql_result.get("sql"),
        #         "sql_result": sql_result.get("result"),
        #         "source": "sql",
        #     }

        return {**result, "source": "vector"}

    @staticmethod
    def sync(data: SyncRequest, db: Session = Depends(get_db)):
        svc = VectorService(collection_name=data.collection)
        return svc.sync_from_db(db, clear=data.clear)
