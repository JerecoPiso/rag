from fastapi import Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.services.rag_service import RAGService
from app.schemas.rag import RAGRequest


class RAGController:
    @staticmethod
    def ask(data: RAGRequest, db: Session = Depends(get_db), _: User = Depends(get_current_user)):
        return RAGService(db).ask(data.question)
