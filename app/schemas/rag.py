from pydantic import BaseModel
from typing import Any, Literal

class RAGRequest(BaseModel):
    question: str
    provider: Literal["openai", "anthropic", "google"] = "google"

class RAGResponse(BaseModel):
    question: str
    sql:      str
    result:   list[dict[str, Any]]
    answer:   str

# --- Qdrant vector schemas ---

class IngestRequest(BaseModel):
    texts:      list[str]
    metadatas:  list[dict[str, Any]] = []
    collection: str = "rag_documents"

class IngestResponse(BaseModel):
    ingested:   int
    collection: str

class SearchRequest(BaseModel):
    query:      str
    top_k:      int = 5
    collection: str = "rag_documents"

class SearchResult(BaseModel):
    score:    float
    text:     str
    metadata: dict[str, Any]

class SearchResponse(BaseModel):
    query:   str
    results: list[SearchResult]

class VectorRAGRequest(BaseModel):
    question:   str
    collection: str = "rag_documents"
    provider:   Literal["openai", "anthropic", "google", "ollama"] = "ollama"

class VectorRAGResponse(BaseModel):
    question: str
    context:  list[SearchResult]
    answer:   str

class SyncRequest(BaseModel):
    collection: str = "rag_documents"
    clear:      bool = True

class SyncResponse(BaseModel):
    total_ingested: int
    sources:        list[dict[str, Any]]
