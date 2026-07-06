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
from app.utils.emr_formatter import format_record, build_metadata, _SOURCE_DATE_FIELD


class VectorService:
    """
    Improvements over V1:
    - Patient identified by patient_id (exact keyword match) instead of name text match
    - Name → patient_id resolved once via semantic search; stored in session
    - Falls back to text-field name search when patient_id is unavailable
    - record_type filter only applied when explicitly requested; fallback drops it
    - Separate score thresholds for patient-specific (0.60) vs. general (0.45) searches
    - _resolve_from_history scans user messages only to avoid false positives
    - _NAME_PATTERNS handles "patient <name>" and "Patient: <name>" formats
    """

    def __init__(self, collection_name: str = "rag_documents"):
        if not settings.QDRANT_URL:
            raise HTTPException(status_code=500, detail="QDRANT_URL is not configured")
        from ollama import Client as OllamaClient
        self.embed_client = OllamaClient(host=settings.OLLAMA_URL)
        self.collection_name = collection_name
        try:
            self.client = QdrantClient(url=settings.QDRANT_URL)
            self.client.get_collections()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Cannot connect to Qdrant at {settings.QDRANT_URL}. ({e})"
            )
        try:
            self._ensure_collection()
        except Exception as e:
            msg = str(e)
            if "ollama" in msg.lower() or settings.OLLAMA_URL in msg:
                raise HTTPException(
                    status_code=503,
                    detail=f"Cannot connect to Ollama at {settings.OLLAMA_URL}. ({e})"
                )
            raise HTTPException(status_code=503, detail=f"Failed to initialize collection: {e}")

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        response = self.embed_client.embed(model=settings.OLLAMA_EMBED_MODEL, input=text)
        return response.embeddings[0]

    # ── Collection Setup ──────────────────────────────────────────────────────

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

        text_index = TextIndexParams(
            type="text",
            tokenizer=TokenizerType.WORD,
            min_token_len=2,
            max_token_len=15,
            lowercase=True,
        )
        for field in ("text", "patient_name"):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=text_index,
                )
            except Exception:
                pass

        for field in ("record_type", "patient_id"):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema="keyword",
                )
            except Exception:
                pass

    def _recreate_collection(self):
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name in existing:
            self.client.delete_collection(self.collection_name)
        self._ensure_collection()

    # ── Ingest / Sync ─────────────────────────────────────────────────────────

    def ingest(self, texts: list[str], metadatas: list[dict], batch_size: int = 50) -> int:
        if not texts:
            return 0
        padded_meta = list(metadatas) + [{}] * (len(texts) - len(metadatas))
        points = [
            PointStruct(
                id=str(uuid.uuid5(
                    uuid.NAMESPACE_DNS,
                    f"{meta.get('source', '')}:{meta.get('case_id', '')}:{text}",
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
            "_patient_case_medicine_vw",
            "_patient_case_medical_consumption_vw",
            "_patient_case_status_vw",
            "_patient_tpr_vw",
            "_patient_opr_vw",
            "_patient_monitor_vw",
            "_patient_fluid_intake_and_output_vw",
            "_patient_diagnostics_vw",
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

            texts: list[str]     = []
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
        "MEDICAL_CONSUMPTION", "CASE_STATUS", "Ward Vital Signs", "Out Patient Vital Signs", "Monitoring Vital Signs", "Fluid Intake and Output (FIAO)",
    }

    def _decompose_query(self, question: str, provider: str) -> dict:
        system = (
            "You are a query analyzer for a hospital EMR system.\n"
            "Given a question, extract:\n"
            "- patient_name: the full name of the patient mentioned IN THIS QUESTION ONLY "
            "(first, middle, and last name if present). "
            "Return null if no patient name is explicitly in the question text.\n"
            "- search_intent: the core MEDICAL condition, symptom, or topic — "
            "remove filler words AND remove patient names. "
            "Generic phrases like 'medical record', 'patient chart', 'medical history', "
            "'all records', 'information' mean the user wants everything — return null.\n"
            "- record_type: ONLY set when the question explicitly asks for a specific type:\n"
            "  DOCTOR_ORDER        → explicitly asks for doctors orders or doctors notes\n"
            "  NURSE_NOTE          → explicitly asks for nurses notes\n"
            "  DIET_ORDER          → explicitly asks for diet orders\n"
            "  VITAL_SIGNS         → explicitly asks for vitals, BP, temperature, weight\n"
            "  ANIMAL_BITE         → explicitly asks for animal bite records\n"
            "  MEDICINE            → explicitly asks for medicines, prescriptions\n"
            "  MEDICAL_CONSUMPTION → explicitly asks for consumed/given medicines (IVF, oxygen)\n"
            "  CASE_STATUS         → explicitly asks for admission, discharge, case status\n\n"
            "Generic phrases like 'medical record', 'patient chart', 'history' are NOT a record_type — use null.\n\n"
            "Respond in JSON only:\n"
            '{"patient_name": "...", "search_intent": "...", "record_type": "..."}\n'
            "Use null when a field is not applicable."
        )
        try:
            raw = self._call_llm(provider, system, [{"role": "user", "content": question}], temperature=0).strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"\s*```$", "", raw)
            data              = json.loads(raw)
            patient_name_hint = (data.get("patient_name") or "").strip()
            search_intent     = (data.get("search_intent") or "").strip()
            record_type       = (data.get("record_type")   or "").strip().upper()
            if record_type not in self._VALID_RECORD_TYPES:
                record_type = ""
            return {"patient_name_hint": patient_name_hint, "search_intent": search_intent, "record_type": record_type}
        except Exception:
            return {"patient_name_hint": "", "search_intent": "", "record_type": ""}

    # ── Patient Resolution ────────────────────────────────────────────────────

    def _resolve_patient(self, name_hint: str) -> tuple[str, str]:
        """
        Semantic search for name_hint → (patient_id, canonical_name).
        Uses patient_id keyword field for exact match in subsequent queries.
        Returns ("", "") if patient_id is not stored in the metadata.
        """
        name_words = [w.lower() for w in name_hint.split() if len(w) >= 3]
        if not name_words:
            return "", ""

        # Strategy 1: keyword search on the patient_name field (most accurate).
        # Requires ALL name words to appear in the stored patient_name value.
        try:
            filter_must = [
                FieldCondition(key="patient_name", match=MatchText(text=w))
                for w in name_words
            ]
            kw_hits, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=filter_must),
                limit=5,
                with_payload=True,
                with_vectors=False,
            )
            for h in kw_hits:
                pid       = str(h.payload.get("patient_id") or "").strip()
                canonical = str(h.payload.get("patient_name") or "").strip()
                if pid and pid.lower() not in ("none", ""):
                    return pid, canonical
        except Exception:
            pass

        # Strategy 2: semantic search fallback.
        # Requires at least 2 name words (or all if fewer than 2) to match.
        try:
            sem_hits = self.client.query_points(
                collection_name=self.collection_name,
                query=self._embed(name_hint),
                limit=10,
            )
            min_match = min(2, len(name_words))
            for h in sem_hits.points:
                pid       = str(h.payload.get("patient_id") or "").strip()
                canonical = str(h.payload.get("patient_name") or "").strip()
                if not pid or pid.lower() in ("none", ""):
                    continue
                payload_text = (canonical + " " + h.payload.get("text", "")).lower()
                if sum(1 for w in name_words if w in payload_text) >= min_match:
                    return pid, canonical
        except Exception:
            pass

        return "", ""

    # ── Filters ───────────────────────────────────────────────────────────────

    def _build_id_filter(self, patient_id: str, record_type: str) -> Filter | None:
        """Exact patient_id keyword match — most reliable."""
        must = []
        if patient_id:
            must.append(FieldCondition(key="patient_id", match=MatchValue(value=patient_id)))
        if record_type:
            must.append(FieldCondition(key="record_type", match=MatchValue(value=record_type)))
        return Filter(must=must) if must else None

    def _build_name_filter(self, patient_name: str, record_type: str) -> Filter | None:
        """
        Fallback when patient_id is unavailable.
        Searches the formatted text field — always contains 'Patient: <fullname>'.
        """
        must = []
        if patient_name:
            parts = [p.lower() for p in patient_name.split() if len(p) >= 3]
            if parts:
                key_parts = [parts[0]] if len(parts) == 1 else [parts[0], parts[-1]]
                for part in key_parts:
                    must.append(FieldCondition(key="text", match=MatchText(text=part)))
        if record_type:
            must.append(FieldCondition(key="record_type", match=MatchValue(value=record_type)))
        return Filter(must=must) if must else None

    # ── Rank Fusion ───────────────────────────────────────────────────────────

    _RRF_K = 60  # standard smoothing constant from the original RRF paper

    @classmethod
    def _rrf_fuse(cls, ranked_id_lists: list[list], k: int = None) -> dict:
        """Combine multiple ranked ID lists into Reciprocal Rank Fusion scores.

        score(d) = sum over lists containing d of 1 / (k + rank(d))
        A document missing from a list simply contributes nothing for that list.
        """
        k = cls._RRF_K if k is None else k
        scores: dict = {}
        for ranked_ids in ranked_id_lists:
            for rank, doc_id in enumerate(ranked_ids, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        return scores

    # ── Search ────────────────────────────────────────────────────────────────

    _STOPWORDS = frozenset({
        "can", "you", "show", "me", "tell", "give", "find", "get", "list",
        "what", "who", "when", "where", "why", "how", "please",
        "patients", "patient", "with", "and", "for", "the", "about",
        "all", "this", "that", "these", "those", "his", "her", "him",
        "their", "them", "its", "our", "your",
        "are", "was", "were", "did", "does", "not", "have", "has", "had",
        "been", "will", "could", "would", "may", "might", "should",
        "also", "just", "from",
        "record", "records", "chart", "medical", "history", "data",
        "details", "info", "information",
    })

    _KEYWORD_RANK_CUTOFF = 10  # scroll() match order isn't relevance-ranked; deep hits are noise

    def search(
        self,
        query: str,
        top_k: int = 10,
        keyword_query: str = None,
        patient_id: str = "",
        patient_name: str = "",
        record_type: str = "",
        min_score: float = 0.60,
        keyword_rank_cutoff: int = None,
    ) -> list[dict]:
        # Pick the best available filter strategy
        if patient_id:
            hard_filter    = self._build_id_filter(patient_id, record_type)
            patient_filter = self._build_id_filter(patient_id, "")
        elif patient_name:
            hard_filter    = self._build_name_filter(patient_name, record_type)
            patient_filter = self._build_name_filter(patient_name, "")
        else:
            hard_filter    = None
            patient_filter = None

        # Semantic search
        vector_hits = self.client.query_points(
            collection_name=self.collection_name,
            query=self._embed(query),
            query_filter=hard_filter,
            limit=top_k,
        )

        # Keyword search
        kw_source = keyword_query if keyword_query is not None else query
        kw_tokens = [
            w for w in kw_source.lower().split()
            if len(w) >= 3 and w not in self._STOPWORDS
        ]
        hard_conds    = list(hard_filter.must)    if hard_filter    else []
        patient_conds = list(patient_filter.must) if patient_filter else []

        keyword_hits: list = []
        if hard_conds or kw_tokens:
            all_conds = hard_conds + [
                FieldCondition(key="text", match=MatchText(text=t)) for t in kw_tokens
            ]
            keyword_hits, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=all_conds),
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            # Fallback: patient filter only (no record_type, no kw_tokens)
            if not keyword_hits and patient_conds:
                keyword_hits, _ = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(must=patient_conds),
                    limit=top_k,
                    with_payload=True,
                    with_vectors=False,
                )

        # scroll() returns filter matches in storage order, not ranked by relevance —
        # so keep only the leading slice instead of trusting matches deep in the list.
        cutoff       = self._KEYWORD_RANK_CUTOFF if keyword_rank_cutoff is None else keyword_rank_cutoff
        keyword_hits = keyword_hits[:cutoff]

        # Rank fusion — combine the semantic and keyword rankings via RRF instead of
        # a plain union, so documents found by both signals outrank single-signal hits.
        vector_ranked = [h for h in vector_hits.points if h.score >= min_score]
        vector_ids    = [h.id for h in vector_ranked]
        keyword_ids   = [h.id for h in keyword_hits]

        rrf_scores = self._rrf_fuse([vector_ids, keyword_ids])

        payload_by_id = {}
        for h in vector_ranked:
            payload_by_id[h.id] = h.payload
        for h in keyword_hits:
            payload_by_id.setdefault(h.id, h.payload)

        ranked_ids = sorted(rrf_scores, key=lambda doc_id: rrf_scores[doc_id], reverse=True)
        results = [
            {
                "score": rrf_scores[doc_id],
                "text":  payload_by_id[doc_id].get("text", ""),
                "metadata": {k: v for k, v in payload_by_id[doc_id].items() if k != "text"},
            }
            for doc_id in ranked_ids
        ]

        return results

    # ── Date Extraction ───────────────────────────────────────────────────────

    def _hit_date(self, hit: dict) -> str:
        """Return the best available date string for sorting (descending = newest first).
        Tries the unified 'date' field first (populated after re-sync), then falls
        back to the source-specific column already stored in the metadata payload.
        """
        meta = hit.get("metadata", {})
        d = str(meta.get("date", "") or "")
        if d and d not in ("None", "N/A", "none"):
            return d
        source = meta.get("source", "")
        field  = _SOURCE_DATE_FIELD.get(source, "")
        return str(meta.get(field, "") or "") if field else ""

    # ── Context Building ──────────────────────────────────────────────────────

    _SECTION_MARKERS = frozenset({
        "=== Patient Chart ===",
        "=== format_doctors_note ===",
        "========================",
    })
    _SECTION_PREFIXES = ("TYPE:",)
    _DROP_PREFIXES = ("SOURCE_VIEW:",)

    def _build_context(self, hits: list[dict]) -> str:
        """
        Join hit texts but emit each non-empty line only once.
        Structural markers (section headers, separators) are always emitted
        so the LLM can still distinguish record boundaries. Internal-only
        markers (e.g. SOURCE_VIEW) are dropped entirely so the LLM never
        sees them and can't echo them back in an answer.
        """
        seen: set[str] = set()
        parts: list[str] = []

        for h in hits:
            chunk: list[str] = []
            for raw_line in h["text"].splitlines():
                stripped = raw_line.strip()
                if not stripped:
                    chunk.append(raw_line)
                    continue
                if stripped.startswith(self._DROP_PREFIXES):
                    continue
                if stripped in self._SECTION_MARKERS or any(
                    stripped.startswith(p) for p in self._SECTION_PREFIXES
                ):
                    chunk.append(raw_line)
                    continue
                if stripped in seen:
                    continue
                seen.add(stripped)
                chunk.append(raw_line)

            while chunk and not chunk[-1].strip():
                chunk.pop()

            if any(l.strip() for l in chunk):
                parts.append("\n".join(chunk))

        return "\n\n".join(parts)

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
        "there is no",
    )

    def _context_has_answer(self, answer: str) -> bool:
        lower = answer.lower()
        return not any(phrase in lower for phrase in self._NO_ANSWER_PHRASES)

    _MAX_HISTORY   = 100
    _MAX_CTX_CHARS = 800_000

    # ── Name Extraction ───────────────────────────────────────────────────────

    _IMPLICIT_REFS = frozenset({
        "he", "she", "they", "his", "her", "him", "their", "them",
    })
    _PATIENT_PHRASES = re.compile(
        r'\b(the patient|that patient|same patient|this patient|the same patient)\b',
        re.IGNORECASE,
    )
    _NON_NAME_STARTS = frozenset({
        "the", "a", "an", "this", "that", "these", "those",
        "our", "their", "your", "my", "its",
        # pronouns — "do you/he/she have?" should not capture "you"/"he" as name
        "you", "he", "she", "we", "they",
        # verbs — "patient has/is/was ..."
        "has", "had", "have", "is", "was", "were", "will", "would",
        "can", "could", "should", "may", "might", "does", "did",
        # prepositions / relative words — "patient with/who/which ..."
        "with", "for", "from", "in", "on", "at", "of", "by",
        "who", "which", "whose",
        # filler words that appear between a keyword trigger and the actual name
        # e.g. "show me details of patient <name>" — "details" must be skipped
        "details", "detail", "information", "info", "chart", "records",
        "record", "file", "history", "data", "case", "medical",
        "summary", "report", "about", "all", "type",
        # common adverbs / particles that keyword triggers like "show" pick up
        "only", "just", "please", "also", "me", "any",
        # action / query verbs — "show me all doctors order of <name>"
        "show", "get", "find", "list", "give", "tell", "fetch",
        # question words
        "what", "when", "where", "how",
        # medical titles that start queries but are not patient names
        "doctors", "doctor", "nurses", "nurse",
    })

    _NAME_PATTERNS = [
        # "Patient: <name>" / "Patient Name: <name>" / "Patient Summary: <name>"
        re.compile(
            r'Patient(?:\s+(?:Name|Summary|Chart|Record))?\s*[:\-]\s*'
            r'([A-Za-z][A-Za-z\s]{2,50}?)(?:\n|\*|$)',
            re.IGNORECASE,
        ),

        # Standalone query: "patient julie quilaquil" (whole message is patient lookup)
        re.compile(
            r'^\s*(?:the\s+)?patient\s+([A-Za-z]{2,}(?:\s+[A-Za-z]{2,}){1,4})\s*[?.!]?\s*$',
            re.IGNORECASE,
        ),

        # "patient named <name>" / "is there a patient named <name>"
        re.compile(
            r'\bpatient\s+named\s+([A-Za-z]{2,}(?:\s+[A-Za-z]{2,}){1,4})',
            re.IGNORECASE,
        ),

        # "patient <Name>" mid-sentence — catches "show me ... of patient mark jhapet"
        # Placed before keyword triggers so it takes priority.
        re.compile(
            r'\bpatient\s+([A-Za-z]{3,}(?:\s+[A-Za-z]{2,}){1,4})\b',
            re.IGNORECASE,
        ),

        # "do/does/did <Name> have/has" — "do mark jhapet lomeda have laboratories?"
        re.compile(
            r'\b(?:do|does|did)\s+([A-Za-z]{3,}(?:\s+[A-Za-z]{2,}){1,3})\s+(?:have|has|had|a|an|any)\b',
            re.IGNORECASE,
        ),

        # "<Name> has/have/had/is/was ..." at start — "cherry mae has laboratories"
        re.compile(
            r'^([A-Za-z]{3,}(?:\s+[A-Za-z]{2,}){1,3})\s+(?:has|have|had|is|was|were)\b',
            re.IGNORECASE,
        ),

        # "<Name> <medical topic>" at start — "mark jhapet lomeda laboratories"
        re.compile(
            r'^([A-Za-z]{3,}(?:\s+[A-Za-z]{2,}){1,3})\s+'
            r'(?:laboratories?|labs?|vitals?|medicines?|prescriptions?|diagnos\w+|'
            r'treatments?|records?|charts?|history|tests?|results?|orders?|notes?|status)\b',
            re.IGNORECASE,
        ),

        # Possessive: "Mark Jhapet's ..."
        re.compile(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})'s\b", re.IGNORECASE),

        # Keyword triggers: "records of", "show me", "about", "bout" (informal), etc.
        re.compile(
            r'(?:records?\s+of|record\s+of|chart\s+of|file\s+of|history\s+of|'
            r'info(?:rmation)?\s+of|details?\s+of|case\s+of|data\s+of|'
            r'patient\s+(?:chart|record|file|history|info(?:rmation)?)\s+of|'
            r'about|bout|show(?:\s+me)?|find|for|get)\s+'
            r'([A-Za-z]{2,}(?:\s+[A-Za-z]{2,}){1,4})',
            re.IGNORECASE,
        ),

        # "of <name>" at very end
        re.compile(
            r'\bof\s+([A-Za-z]{3,}(?:\s+[A-Za-z]{2,}){1,4})\s*[?.!]?\s*$',
            re.IGNORECASE,
        ),
    ]

    def _extract_name(self, text: str) -> str:
        for pattern in self._NAME_PATTERNS:
            m = pattern.search(text)
            if not m:
                continue
            name  = m.group(1).strip().rstrip("'s").strip()
            parts = name.split()
            if not parts or parts[0].lower() in self._NON_NAME_STARTS:
                continue
            if len(parts) >= 2 or (len(parts) == 1 and len(parts[0]) >= 4):
                return name
        return ""

    _NUMBER_WORDS: dict[str, int] = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
        "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    }
    _NUMBERED_REF_PATTERN = re.compile(
        r'\bpatient\s+(?:no\.?\s*|number\s*|#\s*)?'
        r'(\d+|one|two|three|four|five|six|seven|eight|nine|ten'
        r'|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b',
        re.IGNORECASE,
    )

    def _resolve_numbered_patient(self, question: str, history: list[dict]) -> str:
        """Resolve 'patient no N' / 'patient number two' → actual name from the previous list."""
        m = self._NUMBERED_REF_PATTERN.search(question)
        if not m:
            return ""
        token = m.group(1).lower()
        n = int(token) if token.isdigit() else self._NUMBER_WORDS.get(token, 0)
        if n == 0:
            return ""
        for msg in reversed(history):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            hit = re.search(
                rf'(?:^|\n)\s*{n}[.)]\s+([A-Z][A-Za-z]+(?:\s+[A-Za-z]+){{1,5}})',
                content,
                re.MULTILINE,
            )
            if hit:
                return hit.group(1).strip()
        return ""

    def _resolve_from_history(self, history: list[dict]) -> str:
        """Scan user messages first (most recent → oldest), then assistant messages."""
        for role in ("user", "assistant"):
            for msg in reversed(history):
                if msg.get("role") != role:
                    continue
                name = self._extract_name(msg.get("content", ""))
                if name:
                    return name
        return ""

    def _references_active_patient(self, question: str) -> bool:
        words = set(question.lower().split())
        return bool(words & self._IMPLICIT_REFS) or bool(self._PATIENT_PHRASES.search(question))

    # Discovery queries ask about a CATEGORY of patients, not a specific one.
    # Matching these should clear the session so the search runs without a patient filter.
    _DISCOVERY_PATTERN = re.compile(
        r'\bpatients\s+(?:with|that|who|which|having|named|diagnosed|admitted)\b'
        r'|\bpatient\s+that\s+(?:has|have|had)\b'
        r'|\bpatient\s+who\s+(?:has|have|had|is|was|were)\b'
        r'|\bother\s+patient\b'
        r'|\b(?:list|find|show|get|search)\s+(?:all\s+)?patients?\b',
        re.IGNORECASE,
    )

    def _is_discovery_query(self, question: str) -> bool:
        return bool(self._DISCOVERY_PATTERN.search(question))

    # ── Ask ───────────────────────────────────────────────────────────────────

    _GREETING_WORDS = frozenset({
        "hello", "hi", "hey", "thanks", "thank", "good",
        "bye", "goodbye", "morning", "afternoon", "evening",
    })

    def ask(
        self,
        question: str,
        provider: str = "ollama",
        history: list[dict] = [],
        session: dict = None,
        db: Session = None,
        sql_provider: str = "openai",
    ) -> dict:
        if session is None:
            session = {}

        # ── Step 1: Decompose query ───────────────────────────────────────────
        decomposed        = self._decompose_query(question, provider)
        search_intent     = decomposed["search_intent"]
        record_type       = decomposed["record_type"]
        patient_name_hint = decomposed["patient_name_hint"]

        # ── Step 2: Resolve patient ───────────────────────────────────────────
        patient_id   = session.get("patient_id", "")
        patient_name = session.get("patient_name", "")

        # Numbered reference: "patient no 2" → look up name from previous list in history
        raw_name = self._resolve_numbered_patient(question, history) or self._extract_name(question)

        # LLM fallback: for embedded names like "when was the last time <Name> was seen"
        # that no regex pattern covers, use the LLM-extracted name instead.
        # Reject pronouns ("he", "she", "they") that Ollama mistakenly returns as names.
        if not raw_name and patient_name_hint and not self._is_discovery_query(question):
            hint_lower = patient_name_hint.lower().strip()
            hint_parts = [p for p in patient_name_hint.split() if len(p) >= 3]
            if hint_parts and hint_lower not in self._IMPLICIT_REFS:
                raw_name = patient_name_hint

        if raw_name:
            # Explicit name in current question — always resolve and update session.
            # This allows switching to a different patient mid-conversation.
            pid, canonical = self._resolve_patient(raw_name)
            patient_id   = pid
            patient_name = canonical if canonical else raw_name
            session["patient_id"]   = patient_id
            session["patient_name"] = patient_name
        elif self._is_discovery_query(question):
            # Cross-patient discovery query ("patients with X", "patient that has X",
            # "other patient") — clear session and search across all patients.
            patient_id = ""
            patient_name = ""
            session["patient_id"]   = ""
            session["patient_name"] = ""
        elif not patient_id and not patient_name:
            # No session patient and no explicit name — try implicit reference.
            if self._references_active_patient(question):
                raw_name = self._resolve_from_history(history)
            if raw_name:
                pid, canonical = self._resolve_patient(raw_name)
                patient_id   = pid
                patient_name = canonical if canonical else raw_name
                session["patient_id"]   = patient_id
                session["patient_name"] = patient_name
        # else: session patient stays (implicit follow-up like "what else did he take?")

        # When no specific medical intent was found (generic "show me the record of...")
        # and a patient is known, use the patient name as the vector query.
        # This mirrors exactly what happens when the user types just the name alone,
        # which produces high semantic similarity against that patient's records.
        if search_intent:
            query_text = search_intent
        elif patient_name:
            query_text = patient_name
        else:
            query_text = question

        print(f"[ASK] question={question!r}")
        print(f"[ASK] patient_id={patient_id!r}  patient_name={patient_name!r}  "
              f"search_intent={search_intent!r}  record_type={record_type!r}  "
              f"name_hint={patient_name_hint!r}")

        # ── Step 3: Search ────────────────────────────────────────────────────
        has_patient = bool(patient_id or patient_name)

        if has_patient:
            hits = self.search(
                query_text,
                top_k=10,
                keyword_query=query_text,
                patient_id=patient_id,
                patient_name=patient_name,
                record_type=record_type,
                min_score=0.60,
            )
        else:
            hits = self.search(
                query_text,
                top_k=10,
                keyword_query=query_text,
                record_type=record_type,
                min_score=0.45,
            )

        print(f"[ASK] hits={len(hits)}")

        # ── Step 4: Handle no hits ────────────────────────────────────────────
        if not hits:
            if has_patient:
                return self._maybe_fallback({
                    "question": question,
                    "context":  [],
                    "answer":   (
                        f"No matching records found for patient "
                        f"{patient_name or patient_id}. "
                        "The patient may not have records of that type in the system."
                    ),
                    "context_sufficient": False,
                }, question, history, db, sql_provider)

            question_words = set(question.lower().split())
            if question_words & self._GREETING_WORDS and len(question.split()) <= 6:
                system_prompt = (
                    "You are a helpful medical assistant for a hospital system. "
                    "Respond naturally to greetings and small talk. "
                    "Do not discuss topics outside the medical domain."
                )
                messages = history[-self._MAX_HISTORY:] + [{"role": "user", "content": question}]
                answer   = self._call_llm(provider, system_prompt, messages)
                return {"question": question, "context": [], "answer": answer, "context_sufficient": True}

            return self._maybe_fallback({
                "question": question,
                "context":  [],
                "answer":   "No matching patient records found. Try specifying a patient name.",
                "context_sufficient": False,
            }, question, history, db, sql_provider)

        # ── Step 5: Generate answer ───────────────────────────────────────────
        # Sort most-recent-first so the LLM sees the latest records at the top.
        hits = sorted(hits, key=self._hit_date, reverse=True)
        raw_context = self._build_context(hits)
        context     = raw_context[:self._MAX_CTX_CHARS]

        history_note = (
            "Use the conversation history to resolve follow-up references "
            "(e.g. 'he', 'she', 'that patient').\n"
            if history else ""
        )
        system_prompt = (
            # "You are a helpful medical assistant for a hospital system. "
            # "Do not begin responses with 'According to the context' or 'Based on the provided context.' "
            # "Do not discuss topics outside the medical domain. "
            # "Answer only from the retrieved context and conversation history. "
            # "If the user provides only a patient name, summarize all available medical information "
            # "(diagnosis, complaints, treatments, dates). "
            # "If the answer cannot be found, say so — never fabricate.\n\n"
            # "Do not repeat the same information multiple times in your response — "
            # "Only include information that directly answers the user's question. Ignore retrieved context that is unrelated to the question, even if it belongs to the same patient. Do not mention, summarize, or include unrelated diagnoses, treatments, medications, laboratory results, notes, or other medical records."
            # "if a value like a date, diagnosis, or medicine already appeared, do not list it again. "
            # "Present each unique fact only once.\n\n"
            # "The retrieved context contains internal structural markers such as 'TYPE:' and "
            # "'SOURCE_VIEW:' lines. These identify the record type and its underlying database "
            # "view for internal use only — never mention, quote, or include them in your answer.\n\n"
            # "Date handling rules:\n"
            # "- If the question asks for the 'latest', 'most recent', 'last', or 'newest' record, "
            # "find the entry with the most recent date in the context and use only that.\n"
            # "- If the question asks for the 'first', 'oldest', 'earliest', or 'initial' record, "
            # "find the entry with the oldest date in the context and use only that.\n"
            # "- If the question asks for records on a specific date, return only entries matching that date.\n"
            # "- If multiple records exist and no time preference is stated, present them in "
            # "chronological order from most recent to oldest.\n"
            # "- Always include the date when answering time-sensitive questions.\n\n"
            
            # "Question relevance rules:\n"
            # "- Determine the user's intent before answering.\n"
            # "- Answer only the specific information requested by the user.\n"
            # "- Never provide a patient's complete medical record unless the user explicitly requests a summary or provides only the patient's name.\n"
            # "- If the user asks whether a patient exists, answer only whether the patient exists and, if available, provide the patient's identifier or name. Do not include diagnoses, medications, laboratory results, procedures, doctor notes, nursing notes, admissions, or any other medical information.\n"
            # "- If the user asks for a diagnosis, return only the diagnosis.\n"
            # "- If the user asks for medications, return only the medications.\n"
            # "- If the user asks for laboratory results, return only the requested laboratory results.\n"
            # "- If the user asks for doctor notes, return only the relevant doctor notes.\n"
            # "- If the user asks for nursing notes, return only the relevant nursing notes.\n"
            # "- If the user asks for admissions, return only admission-related information.\n"
            # "- Do not include additional facts simply because they are available in the retrieved context.\n"
            # "- When the answer can be expressed as Yes or No, answer Yes or No first, followed only by the minimal supporting information needed to answer the question.\n"
            # "- Return the smallest amount of information necessary to accurately answer the user's question.\n\n"

            # f"{history_note}"
            # f"Retrieved Context:\n{context}"
             "You are a helpful medical assistant for a hospital information system.\n\n"

            "General Rules:\n"
                "- Answer only using the retrieved context and conversation history.\n"
                "- Never fabricate, infer, or assume information that is not explicitly present.\n"
                "- If the requested information cannot be found, clearly state that it is not available in the retrieved records.\n"
                "- Do not answer questions outside the medical domain.\n"
                "- Do not begin responses with phrases such as 'According to the context' or 'Based on the provided context.'\n\n"

            "Relevance Rules:\n"
                "- Determine the user's intent before answering.\n"
                "- Answer only the specific information requested by the user.\n"
                "- Never provide a patient's complete medical record unless the user explicitly requests a summary or provides only the patient's name.\n"
                "- If the user asks whether a patient exists, answer only whether the patient exists and, if available, provide the patient's identifier or name. Do not include diagnoses, medications, laboratory results, procedures, doctor notes, nursing notes, admissions, or any other medical information.\n"
                "- If the user asks for a diagnosis, return only the diagnosis.\n"
                "- If the user asks for medications, return only the medications.\n"
                "- If the user asks for laboratory results, return only the requested laboratory results.\n"
                "- If the user asks for doctor notes, return only the relevant doctor notes.\n"
                "- If the user asks for nursing notes, return only the relevant nursing notes.\n"
                "- If the user asks for admissions, return only admission-related information.\n"
                "- Do not include additional facts simply because they are available in the retrieved context.\n"
                "- When the answer can be expressed as Yes or No, answer Yes or No first, followed only by the minimal supporting information needed to answer the question.\n"
                "- Return the smallest amount of information necessary to accurately answer the user's question.\n\n"
            "Answer Style:\n"
                "- Answer naturally and directly.\n"
                "- When appropriate, begin the response by restating the requested information using wording similar to the user's question.\n"
                "- Examples:\n"
                "  - User: 'What is the latest vital signs?'\n"
                "    Answer: 'The latest vital signs are:'\n"
                "  - User: 'What is the latest diagnosis?'\n"
                "    Answer: 'The latest diagnosis is pneumonia.'\n"
                "  - User: 'What are the medications?'\n"
                "    Answer: 'The medications are:'\n"
                "  - User: 'When was the patient last seen by a doctor?'\n"
                "    Answer: 'The patient was last seen by a doctor on June 30, 2026.'\n"
                "  - User: 'Is the patient admitted?'\n"
                "    Answer: 'Yes. The patient is currently admitted.'\n"
                "  - User: 'Does John Doe exist?'\n"
                "    Answer: 'Yes. John Doe exists.'\n"
                "- Do not mention or imply that the information comes from retrieved records, retrieved context, source documents, conversation history, a database, or any internal system.\n"
                "- Keep responses concise and limited to the information requested.\n"
                "- If the requested information is not available, state only that the requested information is not available.\n\n"
    
    
            "Internal Metadata:\n"
                "- The retrieved context may contain internal markers such as 'TYPE:' and 'SOURCE_VIEW:'.\n"
                "- These markers are for internal system use only.\n"
                "- Never mention, quote, or expose these markers in your response.\n\n"

            "Date Handling Rules:\n"
                "- If the question asks for the latest, most recent, last, or newest record, return only the most recent matching record.\n"
                "- If the question asks for the first, oldest, earliest, or initial record, return only the oldest matching record.\n"
                "- If the question asks for records on a specific date, return only records matching that date.\n"
                "- If multiple matching records exist and no time preference is specified, present only the relevant records in chronological order from most recent to oldest.\n"
                "- Always include dates when they are relevant to the user's question.\n\n"
            "Never mention or imply that your answer comes from the retrieved context, retrieved records, source documents, conversation history, database, or any internal system. Do not use phrases such as 'According to the retrieved record', 'According to the retrieved records', 'According to the retrieved context', 'According to the context', 'Based on the provided context', 'Based on the retrieved context', 'Based on the records', 'From the source', 'The retrieved information shows', or any similar wording. Instead, answer naturally by stating the information directly.\n\n"
            f"{history_note}"

            f"Retrieved Context:\n{context}"
        )

        messages = history[-self._MAX_HISTORY:] + [{"role": "user", "content": question}]
        answer   = self._call_llm(provider, system_prompt, messages)

        return self._maybe_fallback({
            "question": question,
            "context":  hits,
            "answer":   answer,
            "context_sufficient": self._context_has_answer(answer),
        }, question, history, db, sql_provider)

    # ── SQL Fallback ─────────────────────────────────────────────────────────
    # When the vector/EMR-chunk search can't answer the question, fall back to
    # text-to-SQL against the live database so the same question (plus the
    # conversation so far, for pronoun/patient-reference resolution) still
    # gets a shot at an answer.

    def _maybe_fallback(
        self,
        result: dict,
        question: str,
        history: list[dict],
        db: Session | None,
        sql_provider: str,
    ) -> dict:
        if result.get("context_sufficient") or db is None:
            return result
        fallback = self._fallback_to_sql(question, history, db, sql_provider)
        return fallback if fallback is not None else result

    def _fallback_to_sql(
        self,
        question: str,
        history: list[dict],
        db: Session,
        sql_provider: str,
    ) -> dict | None:
        from app.services.rag_service import RAGService
        try:
            sql_result = RAGService(db, provider=sql_provider).ask(question, history=history)
        except Exception as e:
            print(f"[ASK] SQL fallback failed: {e}")
            return None

        print(f"[ASK] SQL fallback sql={sql_result.get('sql')!r} rows={len(sql_result.get('result', []))}")

        rows = sql_result.get("result", [])
        return {
            "question": question,
            "context": [
                {"score": 1.0, "text": json.dumps(row, default=str), "metadata": row}
                for row in rows
            ],
            "answer": sql_result.get("answer", ""),
            "context_sufficient": bool(rows) or bool(sql_result.get("answer")),
            "sql": sql_result.get("sql", ""),
            "source": "sql",
        }

    # ── LLM Dispatch ─────────────────────────────────────────────────────────

    def _call_llm(
        self,
        provider: str,
        system_prompt: str,
        messages: list[dict],
        temperature: float = 0.7,
    ) -> str:
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
                model="claude-sonnet-4-6", max_tokens=1024, temperature=temperature,
                system=system_prompt, messages=messages,
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
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt, temperature=temperature
                ),
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
