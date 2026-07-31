from pydantic import BaseModel
from typing import Any, Literal, List

class RAGRequest(BaseModel):
    question:     str
    provider:     Literal["openai", "anthropic", "google", "ollama"] = "ollama"
    tts_provider: Literal["openai", "piper"] = "piper"

class RAGResponse(BaseModel):
    question: str
    sql:      str
    result:   list[dict[str, Any]]
    answer:   str
    audio:    str | None = None  # data URI (e.g. "data:audio/mpeg;base64,...") of the spoken answer

# --- Qdrant vector schemas ---

class WarmUpResponse(BaseModel):
    llm:   bool
    embed: bool

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

class ChatMessage(BaseModel):
    role:    Literal["user", "assistant"]
    content: str

class VectorRAGRequest(BaseModel):
    question:        str
    collection:      str = "rag_documents"
    provider:        Literal["openai", "anthropic", "google", "ollama"] = "ollama"
    history:         List[ChatMessage] = []
    sql_provider:    Literal["openai", "anthropic", "google", "ollama"] | None = "ollama"
    tts_provider:    Literal["openai", "piper"] = "piper"
    conversation_id: str | None = None

class VectorRAGResponse(BaseModel):
    question:        str
    context:         list[SearchResult]
    answer:          str
    conversation_id: str
    source:          str = "vector"
    audio:           str | None = None  # data URI (e.g. "data:audio/mpeg;base64,...") of the spoken answer

class SyncRequest(BaseModel):
    collection: str = "rag_documents"
    clear:      bool = True

class SyncResponse(BaseModel):
    total_ingested: int
    sources:        list[dict[str, Any]]

class SyncLatestRequest(BaseModel):
    since:      str | None = None  # optional override; defaults to setting_tbl's stored last-sync time
    collection: str = "rag_documents"

class SyncLatestResponse(BaseModel):
    total_ingested: int
    sources:        list[dict[str, Any]]
    since:          str
