from pydantic import BaseModel
from typing import Any

class RAGRequest(BaseModel):
    question: str

class RAGResponse(BaseModel):
    question: str
    sql:      str
    result:   list[dict[str, Any]]
    answer:   str
