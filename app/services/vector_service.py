import re
import json
import uuid
from sqlalchemy.orm import Session
from sqlalchemy import text as sa_text
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchText, MatchValue,
    TextIndexParams, TokenizerType,
)
from fastapi import HTTPException
from app.core.config import settings
from app.utils.emr_formatter import format_record, build_metadata


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
            msg = str(e)
            if "ollama" in msg.lower() or settings.OLLAMA_URL in msg:
                raise HTTPException(
                    status_code=503,
                    detail=f"Cannot connect to Ollama at {settings.OLLAMA_URL}. Make sure Ollama is running and '{settings.OLLAMA_EMBED_MODEL}' is pulled. ({e})"
                )
            raise HTTPException(status_code=503, detail=f"Failed to initialize collection: {e}")

    def _embed(self, text: str) -> list[float]:
        response = self.embed_client.embed(model=settings.OLLAMA_EMBED_MODEL, input=text)
        return response.embeddings[0]

    def _ensure_collection(self):
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name not in existing:
            dim = len(self._embed("test"))
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
            except Exception as e:
                if "already exists" not in str(e).lower():
                    raise
        # Full-text index on the formatted record text
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
        # Text index on patient_name for precise name filtering
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="patient_name",
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
        # Keyword index on record_type for exact filtering
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="record_type",
                field_schema="keyword",
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
                id=str(uuid.uuid5(
                    uuid.NAMESPACE_DNS,
                    f"{meta.get('source', '')}:{meta.get('case_id', '')}",
                )),
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
            "_patient_case_vital_vw",
            "_patient_case_nurses_note_vw",
            "_patient_case_doctors_note_vw",
            "_patient_case_diet_vw",
            "_patient_animal_bite_vw",
            "_patient_case_medicine_vw",
            "_patient_case_medical_consumption_vw",
            "_patient_case_status_vw",
            "_patient_tpr_vw",
            "_patient_opr_vw",
            "_patient_monitor_vw",
            "_patient_fluid_intake_and_output_vw",
            "_patient_diagnostics_vw"
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
                metadatas.append(build_metadata(row_dict, source))

            if texts:
                self.ingest(texts, metadatas)
                total += len(texts)
                synced.append({"source": source, "rows": len(texts)})

        return {"total_ingested": total, "sources": synced}

    # ── Query Decomposition ───────────────────────────────────────────────────

    _VALID_RECORD_TYPES = {
        "DOCTOR_ORDER", "NURSE_NOTE", "DIET_ORDER",
        "VITAL_SIGNS", "ANIMAL_BITE", "MEDICINE",
        "MEDICAL_CONSUMPTION", "CASE_STATUS",
    }

    def _decompose_query(self, question: str, provider: str) -> dict:
        """
        LLM responsibility: extract search_intent and record_type from the current question only.
        Patient name resolution is handled entirely by code — not the LLM.
        """
        system = (
            "You are a query analyzer for a hospital EMR system.\n"
            "Given a question, extract:\n"
            "- search_intent: the core medical information being requested as a clean phrase "
            "(remove filler like 'show me', 'can you tell me', 'what is'; keep all medical terms and any names)\n"
            "- record_type: one of the values below, or null if general:\n"
            "  DOCTOR_ORDER  → doctors orders, doctors notes\n"
            "  NURSE_NOTE    → nurses notes\n"
            "  DIET_ORDER    → diet orders\n"
            "  VITAL_SIGNS   → vitals, blood pressure, temperature, weight\n"
            "  ANIMAL_BITE   → animal bite records\n"
            "  MEDICINE      → medicines, prescriptions, medications\n"
            "  MEDICAL_CONSUMPTION → consumed / given medicines\n"
            "  CASE_STATUS   → case status, admission, discharge\n\n"
            "Respond in JSON only — no explanation:\n"
            '{"search_intent": "...", "record_type": "..."}\n'
            "Use null (not empty string) when a field is not applicable."
        )
        try:
            raw = self._call_llm(provider, system, [{"role": "user", "content": question}], temperature=0).strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
            search_intent = (data.get("search_intent") or question).strip()
            record_type   = (data.get("record_type")   or "").strip().upper()
            if record_type not in self._VALID_RECORD_TYPES:
                record_type = ""
            return {"search_intent": search_intent, "record_type": record_type}
        except Exception:
            return {"search_intent": question, "record_type": ""}

    # ── Search ────────────────────────────────────────────────────────────────

    _STOPWORDS = {
        # "patient", "the", "and", "for", "tell", "about", "what", "who",
        # "give", "show", "find", "get", "his", "her", "this", "that",
        # "with", "from", "details", "info", "information", "records",
        # "case", "cases", "me", "all", "how", "its", "are", "was",
        # "were", "did", "does", "not", "don", "just", "also", "have",
        # "has", "had", "been", "will", "can", "could", "would", "may",
        # "might", "should", "please",
        # "when", "where", "why", "time", "date", "seen", "visit",
        # "last", "ago", "since", "ever", "year", "month", "week", "day",
        # "list", "down", "rephrase", "summarize", "describe", "explain",
        # "latest", "recent", "first", "new", "old", "full",
        # "medical", "record", "chart", "note", "notes", "order", "orders",
        # "doctor", "doctors", "nurse", "nurses", "history", "data",
        # "diagnosis", "complaint", "report", "result", "results",
        "can", "you", "show", "me"
    }

    def _build_filter(self, patient_name: str, record_type: str) -> Filter | None:
        """Build a Qdrant hard filter from patient name and/or record type.

        Uses only the first and last word of the name to avoid over-constraining —
        full names extracted from history (e.g. 'Mark Jhapat Recto Lomeda') would
        require all 4 tokens to match, which fails if any middle/suffix word differs.
        """
        must = []
        if patient_name:
            parts = [p for p in patient_name.lower().split() if len(p) >= 2]
            # first name + last name only; middle names add risk without benefit
            key_parts = [parts[0]] if len(parts) == 1 else [parts[0], parts[-1]]
            for part in key_parts:
                must.append(FieldCondition(key="patient_name", match=MatchText(text=part)))
        if record_type:
            must.append(FieldCondition(key="record_type", match=MatchValue(value=record_type)))
        return Filter(must=must) if must else None

    def search(
        self,
        query: str,
        top_k: int = 10,
        keyword_query: str = None,
        patient_name: str = "",
        record_type: str = "",
    ) -> list[dict]:
        hard_filter = self._build_filter(patient_name, record_type) if patient_name else None

        # Semantic search — embed only the clean search intent; hard filter narrows scope
        vector_hits = self.client.query_points(
            collection_name=self.collection_name,
            query=self._embed(query),
            query_filter=hard_filter,
            limit=top_k,
        )

        # Keyword search — combine hard filter with stopword-filtered text tokens
        kw_source  = keyword_query if keyword_query is not None else query
        # and w not in self._STOPWORDS
        kw_tokens  = [w for w in kw_source.lower().split() if len(w) >= 3 ]
        hard_conds = list(hard_filter.must) if hard_filter else []

        keyword_hits = []
        if hard_conds or kw_tokens:
            # Start strict: hard filter + all keyword tokens AND'd
            all_conds = hard_conds + [FieldCondition(key="text", match=MatchText(text=t)) for t in kw_tokens]
            keyword_hits, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=all_conds),
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            # Fall back to just the hard filter (patient + record type) if no text-token matches
            if not keyword_hits and hard_conds:
                keyword_hits, _ = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(must=hard_conds),
                    limit=top_k,
                    with_payload=True,
                    with_vectors=False,
                )

        seen    = set()
        results = []

        MIN_VECTOR_SCORE = 0.75

        for h in vector_hits.points:
            if h.score < MIN_VECTOR_SCORE:
                continue
            seen.add(h.id)
            results.append({
                "score": h.score,
                "text":  h.payload.get("text", ""),
                "metadata": {k: v for k, v in h.payload.items() if k != "text"},
            })

        for h in keyword_hits:
            if h.id not in seen:
                seen.add(h.id)
                results.append({
                    "score": 1.0,
                    "text":  h.payload.get("text", ""),
                    "metadata": {k: v for k, v in h.payload.items() if k != "text"},
                })

        return results

    # ── Answer Generation ─────────────────────────────────────────────────────

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

    _MAX_HISTORY   = 100
    _MAX_CTX_CHARS = 800000

    # Words that implicitly reference a previously mentioned patient
    _IMPLICIT_REFS = frozenset({
        "he", "she", "they", "his", "her", "him", "their", "them",
        "it", "this", "that", "these", "those",
    })
    _PATIENT_PHRASES = re.compile(
        r'\b(the patient|that patient|same patient|this patient|the same patient)\b',
        re.IGNORECASE,
    )

    # Words that cannot start a valid patient name
    _NON_NAME_STARTS = frozenset({
        "the", "a", "an", "this", "that", "these", "those",
        "our", "their", "your", "my", "its",
    })

    # Ordered extraction patterns — tried in sequence on any text
    _NAME_PATTERNS = [
        # Structured output: "Patient Name: Mark Jhapet Recto Lomeda"
        re.compile(r'Patient\s+Name\s*[:\-]\s*([A-Za-z][A-Za-z\s]{3,50}?)(?:\n|\*|$)', re.IGNORECASE),

        # Any possessive: "Mark Jhapet's ..."
        re.compile(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})'s\b", re.IGNORECASE),

        # Explicit keyword triggers before the name
        re.compile(
            r'(?:records?\s+of|record\s+of|chart\s+of|file\s+of|history\s+of|'
            r'info(?:rmation)?\s+of|details?\s+of|case\s+of|data\s+of|'
            r'patient\s+(?:chart|record|file|history|info(?:rmation)?)\s+of|'
            r'about|show(?:\s+me)?|find|for|get)\s+'
            r'([A-Za-z]{2,}(?:\s+[A-Za-z]{2,}){1,4})',
            re.IGNORECASE,
        ),

        # "of <name>" at the very end of the question (name must be 2+ words)
        re.compile(
            r'\bof\s+([A-Za-z]{3,}(?:\s+[A-Za-z]{2,}){1,4})\s*[?.!]?\s*$',
            re.IGNORECASE,
        ),
    ]

    def _extract_name(self, text: str) -> str:
        """Extract a patient name from any text using the ordered pattern list."""
        for pattern in self._NAME_PATTERNS:
            m = pattern.search(text)
            if not m:
                continue
            name = m.group(1).strip().rstrip("'s").strip()
            parts = name.split()
            # Reject if it starts with an article/determiner or is too short
            if not parts or parts[0].lower() in self._NON_NAME_STARTS:
                continue
            if len(parts) >= 2 or (len(parts) == 1 and len(parts[0]) >= 4):
                return name
        return ""

    def _resolve_active_patient(self, history: list[dict]) -> str:
        """Scan history (most recent first, both roles) for the last active patient name."""
        for msg in reversed(history):
            name = self._extract_name(msg.get("content", ""))
            if name:
                return name
        return ""

    def _references_active_patient(self, question: str) -> bool:
        """Return True if the question uses pronouns or patient-reference phrases."""
        words = set(question.lower().split())
        return bool(words & self._IMPLICIT_REFS) or bool(self._PATIENT_PHRASES.search(question))

    def _substitute_patient(self, text: str, name: str) -> str:
        """Replace implicit references in text with the resolved patient name."""
        result = self._PATIENT_PHRASES.sub(name, text)
        return " ".join(name if w.lower() in self._IMPLICIT_REFS else w for w in result.split())

    def ask(self, question: str, provider: str = "ollama", history: list[dict] = [], session: dict = None) -> dict:
        if session is None:
            session = {}

        # LLM: extract search_intent and record_type only (no patient name — unreliable with local models)
        decomposed    = self._decompose_query(question, provider)
        search_intent = decomposed["search_intent"]
        record_type   = decomposed["record_type"]

        # Layer 1: extract patient name directly from the current question via regex
        patient_name = self._extract_name(question)

        if patient_name:
            # New patient mentioned — update sessionuui
            session["active_patient"] = patient_name
        else:
            # Layer 2: reuse the patient stored in session from a prior turn
            if session.get("active_patient"):
                patient_name = session["active_patient"]
                if self._references_active_patient(question):
                    search_intent = self._substitute_patient(question, patient_name)
            # Layer 3: session is empty — scan history as a last resort
            elif self._references_active_patient(question):
                patient_name = self._resolve_active_patient(history)
                if patient_name:
                    session["active_patient"] = patient_name
                    search_intent = self._substitute_patient(question, patient_name)
        if patient_name and patient_name.strip():
            hits = self.search(
                search_intent,
                top_k=25,
                keyword_query=search_intent,
                patient_name=patient_name,
                record_type=record_type,
            )
        else:
            print("no patient")
            hits = self.search(
                question,
                top_k=25,
                keyword_query=question,
            )

        if not hits:
            if patient_name:
                return {
                    "question": question,
                    "context":  [],
                    "answer":   "I don't have enough information to answer that. No matching patient records were found in the system.",
                    "context_sufficient": False,
                }
            # No patient context — respond conversationally (greetings, small talk, etc.)
            system_prompt = (
                "You are a helpful medical assistant for a hospital system. "
                "Respond naturally to the user's message. For greetings or small talk, reply politely. "
                "Do not discuss topics outside the medical domain."
            )
            messages = history[-self._MAX_HISTORY:] + [{"role": "user", "content": question}]
            answer = self._call_llm(provider, system_prompt, messages)
            return {
                "question": question,
                "context":  [],
                "answer":   answer,
                "context_sufficient": True,
            }

        raw_context = "\n\n".join(h["text"] for h in hits)
        context     = raw_context[:self._MAX_CTX_CHARS]

        history_note = (
            "You may also use the conversation history to resolve follow-up references "
            "(e.g. 'he', 'she', 'that patient', 'the same case') to subjects already established.\n"
            if history else ""
        )
        system_prompt = (
            "You are a helpful medical assistant for a hospital system. "
            "Do not begin responses with phrases such as 'According to the context' or 'Based on the provided context.' "
            "Do not answer questions or discuss topics outside this medical domain.\n\n"
            "To answer the user's question, combine information from BOTH the retrieved context and the conversation history. "
            "Use the retrieved context for facts about patients, diagnoses, and records. "
            "Use the conversation history to understand follow-up references and the flow of the conversation. "
            "If the two sources conflict, prefer the retrieved context. "
            f"{history_note}"
            "If the user provides only a patient name without a specific question, summarize all available medical information about that patient from the context (diagnosis, complaints, treatments, dates, etc.). "
            "If the answer cannot be found in either the context or conversation history, say you don't have enough information — do not guess or fabricate.\n\n"
            f"Retrieved Context:\n{context}"
        )

        messages = history[-self._MAX_HISTORY:] + [{"role": "user", "content": question}]
        answer   = self._call_llm(provider, system_prompt, messages)
        return {
            
            "question": question,
            "context":  hits,
            "answer":   answer,
            "context_sufficient": self._context_has_answer(answer),
        }

    # ── LLM Dispatch ─────────────────────────────────────────────────────────

    def _call_llm(self, provider: str, system_prompt: str, messages: list[dict], temperature: float = 0.7) -> str:
        if provider == "openai":
            from openai import OpenAI
            r = OpenAI(api_key=settings.OPENAI_API_KEY).chat.completions.create(
                model="gpt-4o", max_tokens=1024, temperature=temperature,
                messages=[{"role": "system", "content": system_prompt}] + messages,
            )
            return r.choices[0].message.content.strip()

        elif provider == "anthropic":
            import anthropic
            r = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY).messages.create(
                model="claude-sonnet-4-6", max_tokens=1024,
                temperature=temperature,
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
                config=types.GenerateContentConfig(system_instruction=system_prompt, temperature=temperature),
                contents=contents,
            )
            return r.text.strip()

        elif provider == "ollama":
            res = self.embed_client.chat(
                model=settings.OLLAMA_LLM_MODEL,
                messages=[{"role": "system", "content": system_prompt}] + messages,
                options={"num_ctx": 25000, "temperature": temperature},
            )
            return res.message.content.strip()

        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")
