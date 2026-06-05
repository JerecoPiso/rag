import uuid

from sqlalchemy import CHAR, Column, Integer, String, Boolean, DateTime
from sqlalchemy.sql import func
from app.core.database import Base

class User(Base):
    __tablename__ = "users"

    id         = Column(Integer, primary_key=True, index=True)
    pid        = Column(CHAR(36), unique=True, index=True, nullable=False, default=lambda: str(uuid.uuid4()))

    name       = Column(String(100), nullable=False)
    email      = Column(String(100), unique=True, index=True, nullable=False)
    password   = Column(String(255), nullable=False)
    is_active  = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())