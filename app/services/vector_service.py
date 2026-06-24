import uuid
from sqlalchemy.orm import Session
from sqlalchemy import text as sa_text
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchText,
    TextIndexParams, TokenizerType,
)
from fastapi import HTTPException
from app.core.config import settings
from app.utils.emr_formatter import format_record


class VectorService:
    def __init__(self, collection_name: str = "rag_documents"):
        if not settings.QDRANT_URL:
            raise HTTPException(status_code=500, detail="QDRANT_URL is not configured")
        from ollama import Client as OllamaClient
        self.embed_client = OllamaClient(host=settings.OLLAMA_URL)
        self.collection_name = collection_name
        try:
            self.client = QdrantClient(url=settings.QDRANT_URL)
            self.client.get_collections()  # verify Qdrant is reachable
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Cannot connect to Qdrant at {settings.QDRANT_URL}. Make sure Qdrant is running. ({e})"
            )
        try:
            self._ensure_collection()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Cannot connect to Ollama at {settings.OLLAMA_URL}. Make sure Ollama is running and '{settings.OLLAMA_EMBED_MODEL}' is pulled. ({e})"
            )

    def _embed(self, text: str) -> list[float]:
        response = self.embed_client.embed(model=settings.OLLAMA_EMBED_MODEL, input=text)
        return response.embeddings[0]

    def _ensure_collection(self):
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name not in existing:
            dim = len(self._embed("test"))
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="text",
                field_schema=TextIndexParams(
                    type="text",
                    tokenizer=TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=15,
                    lowercase=True,
                ),
            )
        except Exception:
            pass

    def _recreate_collection(self):
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name in existing:
            self.client.delete_collection(self.collection_name)
        self._ensure_collection()

    def ingest(self, texts: list[str], metadatas: list[dict], batch_size: int = 50) -> int:
        if not texts:
            return 0
        padded_meta = list(metadatas) + [{}] * (len(texts) - len(metadatas))
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=self._embed(text),
                payload={"text": text, **meta},
            )
            for text, meta in zip(texts, padded_meta)
        ]
        for i in range(0, len(points), batch_size):
            self.client.upsert(collection_name=self.collection_name, points=points[i:i + batch_size])
        return len(points)

    @staticmethod
    def _is_id_column(col: str) -> bool:
        lower = col.strip().lower()
        return lower == "id" or lower.endswith("_id")

    def sync_from_db(self, db: Session, clear: bool = True) -> dict:
        if clear:
            self._recreate_collection()

        sources = [
            # "_patient_case_vital_vw",
            # "_patient_case_nurses_note_vw",
            "_patient_case_doctors_note_vw",
            # "_patient_case_diet_vw",
            # "_patient_animal_bite_vw",
        ]

        total  = 0
        synced = []

        for source in sources:
            try:
                result  = db.execute(sa_text(f"SELECT * FROM `{source}`"))
                columns = list(result.keys())
                rows    = result.fetchall()
            except Exception:
                continue

            texts: list[str] = []
            metadatas: list[dict] = []
            for row in rows:
                row_dict      = {col: val for col, val in zip(columns, row)}
                row_dict_text = {col: val for col, val in row_dict.items() if not self._is_id_column(col)}
                text_repr     = format_record(row_dict_text, source)
                texts.append(text_repr)
                metadatas.append({"source": source, **{k: str(v) for k, v in row_dict.items()}})

            if texts:
                self.ingest(texts, metadatas)
                total += len(texts)
                synced.append({"source": source, "rows": len(texts)})

        return {"total_ingested": total, "sources": synced}

    _STOPWORDS = {
        "patient", "the", "and", "for", "tell", "about", "what", "who",
        "give", "show", "find", "get", "his", "her", "this", "that",
        "with", "from", "details", "info", "information", "records",
        "case", "cases", "me", "all", "how",
    }

    def search(self, query: str, top_k: int = 10, keyword_query: str = None) -> list[dict]:
        # Semantic (vector) search — uses full history-aware query for context resolution
        vector_hits = self.client.query_points(
            collection_name=self.collection_name,
            query=self._embed(query),
            limit=top_k,
        )

        # Keyword (full-text) search — uses only the current question so history tokens
        # don't flood the results and push out the actual target records
        kw_source = keyword_query if keyword_query is not None else query
        tokens = [w for w in kw_source.lower().split() if len(w) >= 3 and w not in self._STOPWORDS]
        keyword_hits = []
        if tokens:
            conditions = [FieldCondition(key="text", match=MatchText(text=t)) for t in tokens]
            keyword_hits, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(should=conditions),
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )

        seen = set()
        results = []

        for h in vector_hits.points:
            seen.add(h.id)
            results.append({
                "score": h.score,
                "text": h.payload.get("text", ""),
                "metadata": {k: v for k, v in h.payload.items() if k != "text"},
            })

        for h in keyword_hits:
            if h.id not in seen:
                seen.add(h.id)
                results.append({
                    "score": 1.0,
                    "text": h.payload.get("text", ""),
                    "metadata": {k: v for k, v in h.payload.items() if k != "text"},
                })

        return results

    _NO_ANSWER_PHRASES = (
        "don't have enough information",
        "do not have enough information",
        "cannot find",
        "not found in the context",
        "no information",
        "cannot be found",
        "not available in the",
        "not in the context",
        "i'm unable to find",
        "i am unable to find",
        "not mentioned in",
        "not provided in",
        "there is no"
    )

    def _context_has_answer(self, answer: str) -> bool:
        lower = answer.lower()
        return not any(phrase in lower for phrase in self._NO_ANSWER_PHRASES)

    def ask(self, question: str, provider: str = "ollama", history: list[dict] = []) -> dict:
        if history:
            recent_text  = " ".join(m["content"] for m in history[-4:])
            search_query = f"{recent_text} {question}"
        else:
            search_query = question

        hits    = self.search(search_query, top_k=10, keyword_query=question)
        
        context = "\n\n".join(h["text"] for h in hits)

        history_note = (
            "You may also use the conversation history to resolve follow-up references "
            "(e.g. 'he', 'she', 'that patient', 'the same case') to subjects already established.\n"
            if history else ""
        )
        system_prompt = (
            "You are a helpful medical assistant for a hospital system. "
            "Do not answer questions or discuss topics outside this medical domain.\n\n"
            "Use the following retrieved context as your primary source of facts to answer the user's question. "
            f"{history_note}"
            "If the answer cannot be found in the context or conversation history, say you don't have enough information — do not guess or fabricate.\n\n"
            f"Retrieved Context:\n{context}"
        )

        messages = history + [{"role": "user", "content": question}]
        answer   = self._call_llm(provider, system_prompt, messages)
        return {
            "question": question,
            "context": hits,
            "answer": answer,
            "context_sufficient": self._context_has_answer(answer),
        }

    def _call_llm(self, provider: str, system_prompt: str, messages: list[dict]) -> str:
        if provider == "openai":
            from openai import OpenAI
            r = OpenAI(api_key=settings.OPENAI_API_KEY).chat.completions.create(
                model="gpt-4o", max_tokens=1024,
                messages=[{"role": "system", "content": system_prompt}] + messages,
            )
            return r.choices[0].message.content.strip()

        elif provider == "anthropic":
            import anthropic
            r = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY).messages.create(
                model="claude-sonnet-4-6", max_tokens=1024,
                system=system_prompt,
                messages=messages,
            )
            return r.content[0].text.strip()

        elif provider == "google":
            from google import genai
            from google.genai import types
            contents = [
                types.Content(
                    role="user" if m["role"] == "user" else "model",
                    parts=[types.Part(text=m["content"])],
                )
                for m in messages
            ]
            r = genai.Client(api_key=settings.GOOGLE_API_KEY).models.generate_content(
                model="gemini-2.0-flash",
                config=types.GenerateContentConfig(system_instruction=system_prompt),
                contents=contents,
            )
            return r.text.strip()

        elif provider == "ollama":
            res = self.embed_client.chat(
                model=settings.OLLAMA_LLM_MODEL,
                messages=[{"role": "system", "content": system_prompt}] + messages,
                options={"num_ctx": 25000}  # default is 2048, increase as needed
            )
            return res.message.content.strip()

        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")
