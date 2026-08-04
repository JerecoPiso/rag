import re
import json
import time
import uuid
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
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

    # Every synced MySQL view gets its own Qdrant collection instead of one shared
    # "rag_documents" collection — keeps each table's vectors isolated.
    _SOURCES = [
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
        "_patient_info",
        "_patient_case_summary",
    ]

    # Maps a source view name to its dedicated Qdrant collection name.
    @staticmethod
    def _collection_for_source(source: str) -> str:
        return source.lstrip("_") or source

    # Initializes Ollama embedding client and Qdrant client, verifies both are reachable,
    # and ensures the target collection/indexes exist before the service is used.
    def __init__(self, collection_name: str = "rag_documents"):
        if not settings.QDRANT_URL:
            raise HTTPException(status_code=500, detail="QDRANT_URL is not configured")
        from ollama import Client as OllamaClient
        self.embed_client = OllamaClient(host=settings.OLLAMA_URL)
        self.collection_name = collection_name
        self.view_collections = [self._collection_for_source(s) for s in self._SOURCES]
        try:
            self.client = QdrantClient(url=settings.QDRANT_URL)
            existing = {c.name for c in self.client.get_collections().collections}
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Cannot connect to Qdrant at {settings.QDRANT_URL}. ({e})"
            )
        try:
            self._ensure_all_collections(existing)
        except Exception as e:
            msg = str(e)
            if "ollama" in msg.lower() or settings.OLLAMA_URL in msg:
                raise HTTPException(
                    status_code=503,
                    detail=f"Cannot connect to Ollama at {settings.OLLAMA_URL}. ({e})"
                )
            raise HTTPException(status_code=503, detail=f"Failed to initialize collection: {e}")

    # ── Embedding ─────────────────────────────────────────────────────────────

    # Conservative char cap for nomic-embed-text's ~2048-token context window.
    # Long records are chunked below so this is only a backstop for a single
    # chunk that's still too dense (e.g. unusually long unbroken lines).
    _EMBED_MAX_CHARS = 6000

    # Converts a text string into an embedding vector using the configured Ollama model.
    # Truncates (and, if still rejected, keeps halving) on "context length exceeded"
    # errors rather than letting the whole ingest/search call fail.
    def _embed(self, text: str) -> list[float]:
        truncated = text[:self._EMBED_MAX_CHARS]
        while True:
            try:
                response = self.embed_client.embed(model=settings.OLLAMA_EMBED_MODEL, input=truncated)
                return response.embeddings[0]
            except Exception as e:
                if "context length" not in str(e).lower() or len(truncated) <= 200:
                    raise
                truncated = truncated[:len(truncated) // 2]

    # Max chars per embedding chunk, comfortably under _EMBED_MAX_CHARS, plus a
    # small overlap carried into the next chunk so content split across a chunk
    # boundary isn't invisible to either chunk's vector.
    _CHUNK_MAX_CHARS = 3000
    _CHUNK_OVERLAP   = 200

    # Splits a formatted record into embeddable chunks along line boundaries
    # (never mid-sentence/mid-line) so long free-text records (doctors'/nurses'
    # notes, case summaries) stay fully searchable instead of being truncated.
    # Short records (the common case) return a single-element list unchanged.
    @classmethod
    def _chunk_text(cls, text: str) -> list[str]:
        if len(text) <= cls._CHUNK_MAX_CHARS:
            return [text]
        chunks: list[str] = []
        current = ""
        for line in text.splitlines(keepends=True):
            if current and len(current) + len(line) > cls._CHUNK_MAX_CHARS:
                chunks.append(current)
                current = current[-cls._CHUNK_OVERLAP:] + line
            else:
                current += line
        if current.strip():
            chunks.append(current)
        return chunks or [text]

    # ── Collection Setup ──────────────────────────────────────────────────────

    # Creates a Qdrant collection (sized from a test embedding) and makes sure its
    # text/keyword payload indexes are in place for filtering and text search.
    def _create_collection(self, collection_name: str):
        dim = len(self._embed("test"))
        try:
            self.client.create_collection(
                collection_name=collection_name,
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
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=text_index,
                )
            except Exception:
                pass

        for field in ("record_type", "patient_id"):
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema="keyword",
                )
            except Exception:
                pass

    # Ensures the generic default collection plus every per-view collection exist.
    # `existing` is reused from the connectivity check in __init__ so this doesn't
    # make a second get_collections() round trip to Qdrant.
    def _ensure_all_collections(self, existing: set[str]):
        names = list(dict.fromkeys([self.collection_name, *self.view_collections]))
        for name in names:
            if name not in existing:
                self._create_collection(name)

    # Drops the given collection (if present) and recreates it from scratch —
    # used before a full re-sync so stale points don't linger.
    def _recreate_collection(self, collection_name: str):
        existing = {c.name for c in self.client.get_collections().collections}
        if collection_name in existing:
            self.client.delete_collection(collection_name)
        self._create_collection(collection_name)

    # ── Ingest / Sync ─────────────────────────────────────────────────────────

    # Splits each text into chunks and embeds them individually, upserting one Qdrant
    # point per chunk — using a deterministic UUID (derived from source/case_id/text/
    # chunk index) so re-ingesting the same record doesn't duplicate it. Every chunk's
    # payload carries the FULL original text (not just its own chunk) so a hit on any
    # chunk still gives the LLM the complete record; only the embedding is per-chunk.
    # Returns the number of source records ingested (not the number of chunk points),
    # matching the {"ingested": count} contract callers already rely on.
    def ingest(
        self,
        texts: list[str],
        metadatas: list[dict],
        batch_size: int = 50,
        collection_name: str = None,
    ) -> int:
        if not texts:
            return 0
        collection_name = collection_name or self.collection_name
        padded_meta = list(metadatas) + [{}] * (len(texts) - len(metadatas))
        points = []
        for text, meta in zip(texts, padded_meta):
            chunks = self._chunk_text(text)
            for chunk_index, chunk in enumerate(chunks):
                points.append(PointStruct(
                    id=str(uuid.uuid5(
                        uuid.NAMESPACE_DNS,
                        f"{meta.get('source', '')}:{meta.get('case_id', '')}:{text}:{chunk_index}",
                    )),
                    vector=self._embed(chunk),
                    payload={"text": text, "chunk_index": chunk_index, "chunk_count": len(chunks), **meta},
                ))
        for i in range(0, len(points), batch_size):
            self.client.upsert(collection_name=collection_name, points=points[i:i + batch_size])
        return len(texts)

    # Detects ID-like column names (e.g. "id", "patient_id") so they can be excluded
    # from the human-readable text representation while still being kept in metadata.
    @staticmethod
    def _is_id_column(col: str) -> bool:
        lower = col.strip().lower()
        return lower == "id" or lower.endswith("_id")

    # Pulls every row from each configured MySQL view, formats each row into readable
    # text plus metadata, and ingests it into that view's own Qdrant collection —
    # the main DB → vector-store sync job.
    def sync_from_db(self, db: Session, clear: bool = True) -> dict:
        total  = 0
        synced = []

        for source in self._SOURCES:
            collection_name = self._collection_for_source(source)
            try:
                result  = db.execute(sa_text(f"SELECT * FROM `{source}`"))
                columns = list(result.keys())
                rows    = result.fetchall()
            except Exception:
                continue

            if clear:
                self._recreate_collection(collection_name)

            texts: list[str]     = []
            metadatas: list[dict] = []
            for row in rows:
                row_dict      = {col: val for col, val in zip(columns, row)}
                row_dict_text = {col: val for col, val in row_dict.items() if not self._is_id_column(col)}
                text_repr     = format_record(row_dict_text, source)
                texts.append(text_repr)
                metadatas.append(build_metadata(row_dict, source))

            if texts:
                self.ingest(texts, metadatas, collection_name=collection_name)
                total += len(texts)
                synced.append({"source": source, "collection": collection_name, "rows": len(texts)})

        return {"total_ingested": total, "sources": synced}

    # ── Incremental Sync (latest-only) ───────────────────────────────────────

    _DATETIME_FMT = "%Y-%m-%d %H:%M:%S"
    _SYNC_SETTING_NAME = "vector_last_sync_at"

    # Reads the last-sync timestamp from setting_tbl (name/value pairs).
    # Returns "" if the setting row doesn't exist yet.
    def _get_setting(self, db: Session, name: str) -> str:
        row = db.execute(
            sa_text("SELECT value FROM setting_tbl WHERE name = :name ORDER BY id DESC LIMIT 1"),
            {"name": name},
        ).fetchone()
        return str(row[0]) if row and row[0] is not None else ""

    # Inserts or updates a setting_tbl row. `name` has no unique constraint in this
    # table, so a plain upsert isn't available — select-then-branch instead.
    def _upsert_setting(self, db: Session, name: str, value: str) -> None:
        row = db.execute(
            sa_text("SELECT id FROM setting_tbl WHERE name = :name ORDER BY id DESC LIMIT 1"),
            {"name": name},
        ).fetchone()
        if row:
            db.execute(
                sa_text("UPDATE setting_tbl SET value = :value WHERE id = :id"),
                {"value": value, "id": row[0]},
            )
        else:
            db.execute(
                sa_text("INSERT INTO setting_tbl (name, value) VALUES (:name, :value)"),
                {"name": name, "value": value},
            )
        db.commit()

    # Incremental counterpart to sync_from_db(): pulls only rows newer than `since`
    # from each configured MySQL view (filtered on that view's known date column via
    # _SOURCE_DATE_FIELD) and upserts them into Qdrant. Never clears the collection —
    # existing points are left untouched, only new/changed rows are added.
    #
    # `since` is optional — when not supplied, it's read from setting_tbl
    # (name=vector_last_sync_at). If that row doesn't exist yet, it's created with
    # the current datetime. After the sync runs, the setting is always updated to
    # "now" so the next run (with no `since` passed) picks up from here.
    def sync_latest(self, db: Session, since: str = None) -> dict:
        now_str = datetime.now().strftime(self._DATETIME_FMT)

        if not since:
            since = self._get_setting(db, self._SYNC_SETTING_NAME)
        if not since:
            since = now_str
            self._upsert_setting(db, self._SYNC_SETTING_NAME, since)

        total  = 0
        synced = []

        for source in self._SOURCES:
            collection_name = self._collection_for_source(source)
            print(collection_name, since)
            # date_field = _SOURCE_DATE_FIELD.get(source, "")
            date_field = "created_timestamp"
            try:
                if date_field:
                    result = db.execute(
                        sa_text(f"SELECT * FROM `{source}` WHERE `{date_field}` > :since"),
                        {"since": since},
                    )
                else:
                    result = db.execute(sa_text(f"SELECT * FROM `{source}`"))
                columns = list(result.keys())
                rows    = result.fetchall()
            except Exception:
                continue

            texts: list[str]      = []
            metadatas: list[dict] = []
            for row in rows:
                row_dict      = {col: val for col, val in zip(columns, row)}
                row_dict_text = {col: val for col, val in row_dict.items() if not self._is_id_column(col)}
                text_repr     = format_record(row_dict_text, source)
                texts.append(text_repr)
                metadatas.append(build_metadata(row_dict, source))

            if texts:
                self.ingest(texts, metadatas, collection_name=collection_name)
                total += len(texts)
                synced.append({"source": source, "collection": collection_name, "rows": len(texts)})

        self._upsert_setting(db, self._SYNC_SETTING_NAME, now_str)

        return {"total_ingested": total, "sources": synced, "since": since}

    # ── Query Decomposition ───────────────────────────────────────────────────

    # Deterministic keyword → collection map used to narrow the search when a
    # question clearly names a specific kind of record. Deliberately excludes
    # generic words (e.g. "record", "info", "history" — already in _STOPWORDS and
    # so never reach this check) since those mean "search everything", not one view.
    _SOURCE_KEYWORDS: dict[str, frozenset[str]] = {
        "_patient_case_vital_vw":               frozenset({"assessment", "examination", "exam", "physical"}),
        "_patient_case_nurses_note_vw":         frozenset({"nurse", "nurses", "nursing", "nurse's notes", "nursing notes", "nursing note"}),
        "_patient_case_doctors_note_vw":        frozenset({"doctor", "doctors", "physician", "order", "orders"}),
        "_patient_case_diet_vw":                frozenset({"diet", "food", "meal", "meals", "nutrition"}),
        "_patient_case_medicine_vw":            frozenset({"medicine", "medicines", "medication", "medications", "prescription", "prescriptions", "drug", "drugs"}),
        "_patient_case_medical_consumption_vw": frozenset({"consumption", "supply", "supplies", "ivf", "oxygen", "consumed"}),
        "_patient_case_status_vw":              frozenset({"diagnosis", "station",  "discharge_date", "discharge date", "patient type",  "diagnoses", "admission", "admitted", "discharge", "discharged", "classification", "complaint", "status"}),
        "_patient_tpr_vw":                      frozenset({"tpr", "temperature", "pulse", "respiration", "vitals", "vital sign", "vital signs"}),
        "_patient_opr_vw":                      frozenset({"opr", "opr vital", "opr vitals", "outpatient vital signs", "outpatient vital sign", "outpatient vital signs"}),
        "_patient_monitor_vw":                  frozenset({"monitor", "monitoring", "monitoring vital", "monitoring vitals", "monitoring vital signs", "monitoring vital sign"}),
        "_patient_fluid_intake_and_output_vw":  frozenset({"fluid", "fiao", "intake", "output"}),
        "_patient_diagnostics_vw":              frozenset({"lab", "labs", "laboratory", "diagnostics", "diagnostic", "test", "tests", "result", "results"}),
        "_patient_info":                        frozenset({"demographic", "demographics", "birthdate", "birthday", "address", "name", "contact", "phone", "email", "gender", "age", "patient info", "patient information", "father", "mother", "parent", "parents", "spouse", "husband", "wife", "guardian", "religion", "nationality", "civil", "married", "marital", "hospital number", "hospital no", "place of birth", "birthplace"}),
        "_patient_case_summary":                frozenset({"summary", "overview"}),
    }

    # Heuristic replacement for what used to be an LLM call: pulls the patient name
    # out via the same regex used for name resolution (_extract_name), strips
    # filler/name words from what's left (_STOPWORDS) to get the search intent, then
    # checks the remaining words against _SOURCE_KEYWORDS to optionally narrow which
    # collection(s) to search — all with no network/LLM round trip. Falls back to
    # every collection (sources=[]) when nothing in the question matches a keyword,
    # same as search() already does for an empty/unclassified source list.
    def _decompose_query(self, question: str) -> dict:
        _t0 = time.perf_counter()
        name = self._extract_name(question)
        remainder = re.sub(re.escape(name), "", question, flags=re.IGNORECASE) if name else question
        intent_words = [
            w for w in re.findall(r"[A-Za-z']+", remainder.lower())
            if len(w) >= 3 and w not in self._STOPWORDS
        ]
        search_intent = " ".join(intent_words)

        word_set = set(intent_words)
        sources = [
            source for source, keywords in self._SOURCE_KEYWORDS.items()
            if word_set & keywords
        ]

        print(f"[TIMING] _decompose_query took {time.perf_counter() - _t0:.4f}s")
        return {"patient_name_hint": name, "search_intent": search_intent, "sources": sources}

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
        # Runs across every per-view collection since the patient may only be
        # resolvable from whichever view holds a match. Collections are queried
        # concurrently (they're independent network round trips to Qdrant) but
        # results are still checked in view_collections order so the "first
        # matching collection wins" priority is unchanged from the sequential version.
        filter_must = [
            FieldCondition(key="patient_name", match=MatchText(text=w))
            for w in name_words
        ]

        def _scroll_collection(collection_name: str) -> list:
            try:
                hits, _ = self.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=Filter(must=filter_must),
                    limit=5,
                    with_payload=True,
                    with_vectors=False,
                )
                return hits
            except Exception:
                return []

        with ThreadPoolExecutor(max_workers=min(8, len(self.view_collections))) as pool:
            hits_by_collection = dict(zip(
                self.view_collections,
                pool.map(_scroll_collection, self.view_collections),
            ))

        for collection_name in self.view_collections:
            for h in hits_by_collection.get(collection_name, []):
                pid       = str(h.payload.get("patient_id") or "").strip()
                canonical = str(h.payload.get("patient_name") or "").strip()
                if pid and pid.lower() not in ("none", ""):
                    return pid, canonical

        # Strategy 2: semantic search fallback.
        # Requires at least 2 name words (or all if fewer than 2) to match.
        query_vector = self._embed(name_hint)
        min_match = min(2, len(name_words))

        def _query_collection(collection_name: str):
            try:
                return self.client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    limit=10,
                ).points
            except Exception:
                return []

        with ThreadPoolExecutor(max_workers=min(8, len(self.view_collections))) as pool:
            sem_hits_by_collection = dict(zip(
                self.view_collections,
                pool.map(_query_collection, self.view_collections),
            ))

        for collection_name in self.view_collections:
            for h in sem_hits_by_collection.get(collection_name, []):
                pid       = str(h.payload.get("patient_id") or "").strip()
                canonical = str(h.payload.get("patient_name") or "").strip()
                if not pid or pid.lower() in ("none", ""):
                    continue
                payload_text = (canonical + " " + h.payload.get("text", "")).lower()
                if sum(1 for w in name_words if w in payload_text) >= min_match:
                    return pid, canonical

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

    # Identifies which source record a chunk-point belongs to, so multiple chunk
    # hits from the same long note collapse into one result instead of each
    # eating a separate top_k slot. Mirrors the (source, case_id) pairing that
    # ingest() already uses to number a record's chunks.
    @staticmethod
    def _record_key(payload: dict) -> tuple:
        return (payload.get("source", ""), payload.get("case_id") or payload.get("id") or "")

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

    # Hybrid search: runs semantic (vector) search and keyword (text-match) search in
    # parallel, applies patient/record_type filters, then fuses both rankings via RRF
    # into a single ordered list of results with merged payload/text/score.
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
        collection_names: list[str] = None,
    ) -> list[dict]:
        _search_timer = time.perf_counter()

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

        target_collections = collection_names if collection_names is not None else [self.collection_name]

        query_vector = self._embed(query)

        # Keyword search prep
        kw_source = keyword_query if keyword_query is not None else query
        kw_tokens = [
            w for w in kw_source.lower().split()
            if len(w) >= 3 and w not in self._STOPWORDS
        ]
        hard_conds    = list(hard_filter.must)    if hard_filter    else []
        patient_conds = list(patient_filter.must) if patient_filter else []

        cutoff = self._KEYWORD_RANK_CUTOFF if keyword_rank_cutoff is None else keyword_rank_cutoff

        vector_scored: list[tuple] = []  # (id, score) pooled across collections
        keyword_ids:   list        = []  # ids pooled across collections, storage order per collection
        payload_by_id: dict        = {}

        # Each collection's vector query + keyword scroll are independent Qdrant
        # round trips, so run them concurrently across collections instead of one
        # at a time — with up to 14 collections at 2 calls each, sequential search
        # was the single biggest contributor to answer latency.
        def _search_one(collection_name: str):
            v_hits: list[tuple] = []
            try:
                vector_hits = self.client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    query_filter=hard_filter,
                    limit=top_k,
                )
                for h in vector_hits.points:
                    if h.score >= min_score:
                        v_hits.append((h.id, h.score, h.payload))
            except Exception:
                pass

            keyword_hits: list = []
            if hard_conds or kw_tokens:
                all_conds = hard_conds + [
                    FieldCondition(key="text", match=MatchText(text=t)) for t in kw_tokens
                ]
                try:
                    keyword_hits, _ = self.client.scroll(
                        collection_name=collection_name,
                        scroll_filter=Filter(must=all_conds),
                        limit=top_k,
                        with_payload=True,
                        with_vectors=False,
                    )
                    # Fallback: patient filter only (no record_type, no kw_tokens)
                    if not keyword_hits and patient_conds:
                        keyword_hits, _ = self.client.scroll(
                            collection_name=collection_name,
                            scroll_filter=Filter(must=patient_conds),
                            limit=top_k,
                            with_payload=True,
                            with_vectors=False,
                        )
                except Exception:
                    keyword_hits = []

            return v_hits, keyword_hits[:cutoff]

        with ThreadPoolExecutor(max_workers=min(8, len(target_collections) or 1)) as pool:
            per_collection_results = list(pool.map(_search_one, target_collections))

        for v_hits, k_hits in per_collection_results:
            for doc_id, score, payload in v_hits:
                vector_scored.append((doc_id, score))
                payload_by_id.setdefault(doc_id, payload)
            # scroll() returns filter matches in storage order, not ranked by relevance —
            # so keep only the leading slice instead of trusting matches deep in the list.
            for h in k_hits:
                keyword_ids.append(h.id)
                payload_by_id.setdefault(h.id, h.payload)

        # Sort the pooled semantic hits by score since collections were queried
        # independently and their results aren't interleaved by relevance yet.
        vector_scored.sort(key=lambda pair: pair[1], reverse=True)
        vector_ids = [doc_id for doc_id, _ in vector_scored]

        # Rank fusion — combine the semantic and keyword rankings via RRF instead of
        # a plain union, so documents found by both signals outrank single-signal hits.
        rrf_scores = self._rrf_fuse([vector_ids, keyword_ids])

        # Fusion runs across every searched collection, so the merged candidate
        # pool can far exceed top_k (e.g. 46 hits from 14 collections) even
        # though each collection was only asked for top_k. Cap the final list
        # here so only the best-ranked hits are ever built into LLM context —
        # sending every fused hit was the dominant cost on multi-collection asks.
        #
        # A single long record can now surface as multiple chunk-points (see
        # ingest()'s per-chunk embedding), which would otherwise let one record's
        # chunks crowd out other distinct records' top_k slots. Keep only the
        # best-scoring chunk per (source, case_id) before slicing to top_k.
        all_ranked = sorted(rrf_scores, key=lambda doc_id: rrf_scores[doc_id], reverse=True)
        ranked_ids: list = []
        seen_records: set = set()
        for doc_id in all_ranked:
            key = self._record_key(payload_by_id[doc_id])
            if key in seen_records:
                continue
            seen_records.add(key)
            ranked_ids.append(doc_id)
            if len(ranked_ids) >= top_k:
                break
        results = [
            {
                "score": rrf_scores[doc_id],
                "text":  payload_by_id[doc_id].get("text", ""),
                "metadata": {k: v for k, v in payload_by_id[doc_id].items() if k != "text"},
            }
            for doc_id in ranked_ids
        ]
        print(f"[TIMING] search took {time.perf_counter() - _search_timer:.2f}s")
        return results

    # ── Date Extraction ───────────────────────────────────────────────────────

    def _hit_date(self, hit: dict) -> str:
        """Return the best available date string for sorting (descending = newest first).
        Tries the unified 'date' field first (populated after re-sync), then falls
        back to the source-specific column already stored in the metadata payload
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

    # Heuristic check on the LLM's generated answer — flags it as "no answer" if it
    # contains any of the known "I couldn't find / not available" style phrases.
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
        "her", "him", "his", "them", "it",
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

    # Runs the ordered list of regex _NAME_PATTERNS against a chunk of text to pull out
    # a likely patient name, rejecting matches that are just pronouns/filler words.
    def _extract_name(self, text: str) -> str:
        for pattern in self._NAME_PATTERNS:
            m = pattern.search(text)
            if not m:
                continue
            name  = m.group(1).strip().rstrip("'s").strip()
            parts = name.split()
            if not parts or any(p.lower() in self._NON_NAME_STARTS for p in parts):
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

    # Detects implicit follow-up references to the currently active patient,
    # e.g. pronouns ("he", "she") or phrases like "the patient"/"same patient".
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

    # Detects cross-patient "discovery" questions (e.g. "patients with diabetes")
    # that should search across all patients instead of a single resolved patient.
    def _is_discovery_query(self, question: str) -> bool:
        return bool(self._DISCOVERY_PATTERN.search(question))

    # "Does patient X exist?" style questions — these should get a bare Yes/No
    # answer, never a full record dump.
    _EXISTENCE_PATTERN = re.compile(
        r'\bexist(?:s|ing|ence)?\b'
        r'|\bis there (?:a|any)\s+patient\b'
        r'|\bdo(?:es)?\s+you\s+have\s+(?:a|any)\s+patient\b',
        re.IGNORECASE,
    )

    def _is_existence_query(self, question: str) -> bool:
        return bool(self._EXISTENCE_PATTERN.search(question))

    # Death/dying/prognosis-type questions — the system prompt already tells the LLM
    # never to answer these with a direct Yes/No, but that instruction competes with
    # the general "answer Yes/No first" rule and can get overridden. This pattern lets
    # us enforce it deterministically in code as a backstop, regardless of phrasing.
    _MORTALITY_PATTERN = re.compile(
        r'\b(?:die|dies|died|dying|death|dead|deceased|mortality|prognosis)\b'
        r'|\bsurviv\w*\b'
        r'|\b(?:chance|chances|probability|likelihood|odds|risk)\s+(?:of|that|he|she|they)\b',
        re.IGNORECASE,
    )

    # _LEADING_YESNO = re.compile(r'^\s*(?:yes|no)\b[.,!:]?\s*', re.IGNORECASE)
    _LEADING_YESNO = re.compile(r'^\s*yes\b[.,!:]?\s*', re.IGNORECASE)

    # A literal death-status claim — if the model asserts one of these and the
    # exact word never appears in the retrieved context, it's a hallucination,
    # not a documented fact.
    _DEATH_STATUS_WORDS = re.compile(
        r'\b(?:deceased|died|passed away|expired)\b',
        re.IGNORECASE,
    )

    # Predictive/alarmist commentary on outcome risk — the model editorializing
    # about how dangerous a condition is, rather than reporting a documented fact.
    _PROGNOSIS_COMMENTARY = re.compile(
        r'\b(?:life-?threatening|can be fatal|could be fatal|risk of death|'
        r'might not survive|could die|may die|likely to die|high mortality risk)\b',
        re.IGNORECASE,
    )

    _NO_MORTALITY_INFO = (
        "There is no documented prognosis or mortality information available for this patient."
    )

    # Deterministic backstop for the mortality/prognosis exception in the system
    # prompt: for any death/dying/prognosis question — strip a leading Yes/No,
    # block a death-status claim that isn't literally present in the retrieved
    # context, and drop any sentence that editorializes about mortality risk
    # instead of reporting a documented fact.
    def _guard_mortality_answer(self, question: str, answer: str, context: str = "") -> str:
        if not self._MORTALITY_PATTERN.search(question):
            return answer

        answer = self._LEADING_YESNO.sub("", answer).strip() or answer

        if self._DEATH_STATUS_WORDS.search(answer) and not self._DEATH_STATUS_WORDS.search(context):
            return (
                "There is no documented record of the patient's death. I can only share what is "
                "explicitly recorded, such as diagnosis, treatment, and admission/discharge status."
            )

        sentences = re.split(r'(?<=[.!?])\s+', answer)
        kept = [s for s in sentences if not self._PROGNOSIS_COMMENTARY.search(s)]
        cleaned = " ".join(kept).strip()
        return cleaned or self._NO_MORTALITY_INFO

    # The system prompt already tells the LLM to never remark that a question was
    # already asked/repeated/rephrased or add asides about its own answering
    # behavior — but small local (Ollama) models don't reliably follow that rule,
    # especially once `history` already contains the same Q&A from an earlier
    # turn. Deterministic backstop, same pattern as _guard_mortality_answer:
    # strip any such meta-commentary regardless of exact phrasing.
    # Broad, vocabulary-based rather than exact-phrase: the LLM keeps inventing
    # new wordings for the same aside ("I repeated it for consistency", "Duplicate
    # question answered in the same format", "the answer remains the same"...), so
    # matching specific phrases is a losing chase. Instead strip any parenthetical
    # that talks about the question/answer/format itself — legitimate clinical
    # answers never use parentheses this way (see Answer Style examples above).
    _META_COMMENTARY_PAREN = re.compile(
        r'\s*\([^()]*\b(?:'
        r'note|duplicate\w*|repeat(?:ed|s)?|rephrase\w*|unchanged|no\s+change|'
        r'consisten\w*|previous(?:ly)?|remains?\s+the\s+same|same\s+(?:format|answer|response)|'
        r'already\s+\w+|question|answer'
        r')\b[^()]*\)',
        re.IGNORECASE,
    )
    _META_COMMENTARY_SENTENCE = re.compile(
        r'(?:^|(?<=[.!?]\s))(?:Note[:,]|Duplicate\s+question|The\s+answer\s+remains|'
        r'This\s+is\s+a\s+duplicate|The\s+same\s+question|Since\s+the\s+answer|'
        r'Since\s+this\s+(?:question|answer)|As\s+(?:previously|already)\s+'
        r'(?:answered|stated|mentioned))[^.!?]*[.!?]',
        re.IGNORECASE,
    )

    def _guard_meta_commentary(self, answer: str) -> str:
        cleaned = self._META_COMMENTARY_PAREN.sub("", answer)
        cleaned = self._META_COMMENTARY_SENTENCE.sub("", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        return cleaned or answer

    # Strips a leading correction phrase ("i mean", "sorry", "actually", ...)
    # so a follow-up like "i mean balansi jemar" can be checked against a name.
    _CORRECTION_PREFIX = re.compile(
        r'^\s*(?:i\s+meant|i\s+mean|meant|sorry,?|no,?\s*i\s+mean|correction[:,]?|actually)\s+',
        re.IGNORECASE,
    )

    def _is_name_only_message(self, question: str, name: str) -> bool:
        """True when the message is essentially just the patient name (optionally
        prefixed with a correction phrase), e.g. "i mean balansi jemar" — used to
        carry an existence-check intent forward across a name correction."""
        if not name:
            return False
        stripped  = self._CORRECTION_PREFIX.sub("", question).strip()
        remainder = re.sub(re.escape(name), "", stripped, flags=re.IGNORECASE).strip(" .,!?")
        return len(remainder) <= 2

    # ── Ask ───────────────────────────────────────────────────────────────────

    # Trigger words that make a message greeting/small-talk-shaped.
    _GREETING_WORDS = frozenset({
        "hi", "hii", "hiya", "hello", "hey", "heya", "yo", "sup",
        "greetings", "howdy", "thanks", "thank", "good",
        "bye", "goodbye", "morning", "afternoon", "evening", "night", "day",
    })

    # Filler words allowed alongside a greeting trigger without turning the
    # message into a real question (e.g. "hi how are you", "thanks so much").
    _GREETING_FILLER_WORDS = frozenset({
        "there", "again", "how", "are", "you", "doing", "much", "so", "very",
        "ok", "okay",
    })

    # True only when the ENTIRE message is greeting/small-talk — every word is
    # either a greeting trigger or allowed filler, and at least one greeting
    # trigger is present. A message like "hi, does she have diabetes" has real
    # question words ("does", "she", "have", "diabetes") and correctly returns
    # False so it still goes through normal search.
    def _is_pure_greeting(self, question: str) -> bool:
        words = re.findall(r"[A-Za-z']+", question.lower())
        if not words:
            return False
        if not any(w in self._GREETING_WORDS for w in words):
            return False
        return all(w in self._GREETING_WORDS or w in self._GREETING_FILLER_WORDS for w in words)

    # Main RAG entry point: decomposes the question, resolves/tracks the active patient
    # across conversation turns, runs the hybrid search, builds context, and generates
    # a final answer via the LLM — falling back to text-to-SQL when context is insufficient.
    def ask(self, question: str, *args, **kwargs) -> dict:
        _t0 = time.perf_counter()
        try:
            return self._ask_inner(question, *args, **kwargs)
        finally:
            print(f"[TIMING] ask() total took {time.perf_counter() - _t0:.2f}s")

    def _ask_inner(
        self,
        question: str,
        provider: str = "ollama",
        history: list[dict] = [],
        session: dict = None,
        db: Session = None,
        sql_provider: str = "openai",
    ) -> dict:
        # print("asking")
        if session is None:
            session = {}

        # ── Step 0: Pure greeting / small talk ────────────────────────────────
        # Checked before patient resolution/search so a bare "hi" always goes
        # straight to the LLM — even mid-conversation with a patient already
        # active in session. Only short-circuits when the message is NOTHING
        # but greeting/filler words; any attached question still runs normally.
        if self._is_pure_greeting(question):
            _t_greeting = time.perf_counter()
            system_prompt = (
                "You are a helpful medical assistant for a hospital system. "
                # "Respond naturally to greetings and small talk. "
                "Do not discuss topics outside the medical domain."
            )
            messages = history[-self._MAX_HISTORY:] + [{"role": "user", "content": question}]
            answer   = self._call_llm(provider, system_prompt, messages)
            print(f"[TIMING] greeting took {time.perf_counter() - _t_greeting:.2f}s")
            return {"question": question, "context": [], "answer": answer, "context_sufficient": True}

        # ── Step 1: Decompose query ───────────────────────────────────────────
        # print("_decompose_query")
        decomposed        = self._decompose_query(question)
        search_intent     = decomposed["search_intent"]
        sources           = decomposed["sources"]
        patient_name_hint = decomposed["patient_name_hint"]

        # ── Step 2: Resolve patient ───────────────────────────────────────────
        _t_resolve = time.perf_counter()
        patient_id   = session.get("patient_id", "")
        patient_name = session.get("patient_name", "")

        # Numbered reference: "patient no 2" → look up name from previous list in history
        raw_name = self._resolve_numbered_patient(question, history) or self._extract_name(question)

        # LLM fallback: for embedded names like "when was the last time <Name> was seen"
        # that no regex pattern covers, use the LLM-extracted name instead.
        # Reject pronouns ("he", "she", "they") that Ollama mistakenly returns as names.
        # Also reject the hint outright when a patient is already active in session and
        # the question is only an implicit follow-up (pronoun / "the patient ...") — the
        # decompose LLM sometimes hallucinates a name for such questions even though it
        # was told to return null, which would otherwise hijack the session mid-conversation.
        already_has_patient = bool(patient_id or patient_name)
        is_implicit_followup = self._references_active_patient(question)
        if (
            not raw_name and patient_name_hint and not self._is_discovery_query(question)
            and not (already_has_patient and is_implicit_followup)
        ):
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

        # ── Existence-check tracking ──────────────────────────────────────────
        # A bare "is patient X exists" should get a Yes/No answer only. A follow-up
        # that just re-supplies/corrects the name (e.g. "i mean balansi jemar")
        # carries that same intent forward even though it doesn't repeat "exist" —
        # otherwise the general system prompt's "user provides only the patient's
        # name → give the full record" rule takes over and dumps the whole chart.
        is_existence_query = self._is_existence_query(question)
        if (
            not is_existence_query and session.get("pending_existence")
            and raw_name and self._is_name_only_message(question, raw_name)
        ):
            is_existence_query = True
        session["pending_existence"] = is_existence_query

        if is_existence_query:
            if patient_id or patient_name:
                display_name = patient_name or raw_name
                answer = f"Yes. {display_name} exists."
            else:
                display_name = raw_name or patient_name_hint
                answer = (
                    f"No, there is no record of a patient named {display_name} in the system."
                    if display_name else "No patient name was specified to check."
                )
            return {
                "question": question,
                "context": [],
                "answer": answer,
                "context_sufficient": True,
            }

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

        # Classify which collection(s) the question actually needs, based on the
        # sources decomposed above — falls back to every collection when the
        # question doesn't name a specific kind of record.
        target_collections = (
            [self._collection_for_source(s) for s in sources] if sources else self.view_collections
        )

        print(f"[ASK] question={question!r}")
        print(f"[ASK] patient_id={patient_id!r}  patient_name={patient_name!r}  "
              f"search_intent={search_intent!r}  sources={sources!r}  "
              f"name_hint={patient_name_hint!r}  target_collections={target_collections!r}")
        print(f"[TIMING] patient resolution took {time.perf_counter() - _t_resolve:.2f}s")

        # ── Step 3: Search ────────────────────────────────────────────────────
        _t_search = time.perf_counter()
        has_patient = bool(patient_id or patient_name)

        if has_patient:
            hits = self.search(
                query_text,
                top_k=10,
                keyword_query=query_text,
                patient_id=patient_id,
                patient_name=patient_name,
                min_score=0.60,
                collection_names=target_collections,
            )
        else:
            # No patient resolved at all — the question isn't about one named
            # person, it's a criteria/attribute question spanning multiple
            # patients (e.g. "patients with pneumonia", "patients in station 3"),
            # even though neither says "list". Detecting this doesn't depend on
            # matching that exact wording — failing to resolve a patient IS the
            # signal — so try text-to-SQL first, where a real WHERE-clause filter
            # answers "which/how many patients match X" far more reliably than
            # vector similarity search over patient-record chunks ever could.
            # Pure greetings/small talk are already handled in Step 0 above.
            if db is not None:
                sql_result = self._fallback_to_sql(question, history, db, sql_provider, session)
                if sql_result is not None and sql_result.get("context_sufficient"):
                    print("[ASK] answered via SQL (no specific patient resolved)")
                    return sql_result

            hits = self.search(
                query_text,
                top_k=10,
                keyword_query=query_text,
                min_score=0.45,
                collection_names=target_collections,
            )

        print(f"[ASK] hits={len(hits)}")
        print(f"[TIMING] search took {time.perf_counter() - _t_search:.2f}s")

        # If the classifier narrowed the search to specific collection(s) but found
        # nothing there, widen to every collection before giving up — guards against
        # a misclassified source hiding the real match in a different view.
        if not hits and target_collections is not self.view_collections:
            print("[ASK] no hits in classified collection(s), widening to all collections")
            widen_kwargs = dict(
                top_k=10, keyword_query=query_text,
                collection_names=self.view_collections,
            )
            if has_patient:
                hits = self.search(
                    query_text, patient_id=patient_id, patient_name=patient_name,
                    min_score=0.60, **widen_kwargs,
                )
            else:
                hits = self.search(query_text, min_score=0.45, **widen_kwargs)

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
                }, question, history, db, sql_provider, session)

            # Greeting check and the SQL attempt already happened above (Step 3)
            # for the no-patient path, so there's nothing left to retry here —
            # calling _fallback_to_sql again would just repeat the same failed query.
            return {
                "question": question,
                "context":  [],
                "answer":   "No matching patient records found. Try specifying a patient name.",
                "context_sufficient": False,
            }

        # ── Step 5: Generate answer ───────────────────────────────────────────
        # Sort most-recent-first so the LLM sees the latest records at the top.
        hits = sorted(hits, key=self._hit_date, reverse=True)
        raw_context = self._build_context(hits)
        context     = raw_context[:self._MAX_CTX_CHARS]

        # history_note = (
        #     "Use the conversation history to resolve follow-up references "
        #     "(e.g. 'he', 'she', 'that patient').\n"
        #     if history else ""
        # )
        system_prompt = (
             "You are a helpful medical assistant for a hospital information system.\n\n"

            "General Rules:\n"
                # "- Answer only using the retrieved context and conversation history.\n"
                "- Answer only using the retrieved context.\n"
                "- Never fabricate, infer, or assume patient facts (diagnosis, medications, vitals, dates, findings) "
                "that are not explicitly present in the retrieved context. This restriction does not apply to "
                "treatment recommendations, which may draw on general medical knowledge as described in the "
                "Treatment Recommendation Rule below.\n"
                "- If the requested information cannot be found, clearly state that it is not available in the retrieved records.\n"
                "- Do not answer questions outside the medical domain.\n"
                "- Do not begin responses with phrases such as 'According to the context' or 'Based on the provided context.'\n"
                "- Never comment on the conversation itself — do not remark that a question was already asked, "
                "repeated, rephrased, or asked multiple times, and do not add notes, disclaimers, or asides about "
                "your own answering behavior (e.g. 'Note: since you asked this twice...'). Always answer the "
                "current question directly and fully, every time it is asked, with no meta-commentary.\n\n"

            "Treatment Recommendation Rule:\n"
                "- If the user asks for a treatment recommendation, suggestion, or guidance (e.g. 'recommend a "
                "treatment', 'what should be done', 'base it on your knowledge'), you may combine the patient's "
                "documented diagnosis, findings, and current treatment from the retrieved context with general "
                "medical knowledge to suggest an appropriate treatment approach.\n"
                "- Ground the suggestion in the patient's actual documented diagnosis and findings — never invent "
                "a diagnosis or finding that is not in the retrieved context.\n"
                "- Do not refuse a treatment recommendation question solely because it calls for clinical judgment "
                "beyond the literal retrieved text.\n"
                "- Present the recommendation as guidance for a licensed healthcare professional to review and "
                "confirm, not as a final medical order — end such answers with a brief note that it should be "
                "reviewed and confirmed by the attending physician before acting on it.\n\n"

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
                "- Exception: if the question asks about death, dying, mortality, prognosis, survival, or a percentage/probability/chance/risk of something occurring, do NOT answer with a direct 'Yes' or a bare percentage. Instead, state only the factual information present in the retrieved context (e.g. documented status, recorded findings) without predicting, estimating, or affirming an outcome that isn't explicitly recorded.\n"
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
                "  - User: 'Recommend a treatment for him.'\n"
                "    Answer: 'Given the documented diagnosis of acute gastritis, treatment options to consider "
                "include a proton pump inhibitor, antiemetics for the vomiting, and IV fluid support for hydration. "
                "This should be reviewed and confirmed by the attending physician before acting on it.'\n"
                # "- Do not mention or imply that the information comes from retrieved records, retrieved context, source documents, conversation history, a database, or any internal system.\n"
                "- Do not mention or imply that the information comes from retrieved records, retrieved context, source documents, a database, or any internal system.\n"

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
            # "Never mention or imply that your answer comes from the retrieved context, retrieved records, source documents, conversation history, database, or any internal system. Do not use phrases such as 'According to the retrieved record', 'According to the retrieved records', 'According to the retrieved context', 'According to the context', 'Based on the provided context', 'Based on the retrieved context', 'Based on the records', 'From the source', 'The retrieved information shows', or any similar wording. Instead, answer naturally by stating the information directly.\n\n"
            "Never mention or imply that your answer comes from the retrieved context, retrieved records, source documents, database, or any internal system. Do not use phrases such as 'According to the retrieved record', 'According to the retrieved records', 'According to the retrieved context', 'According to the context', 'Based on the provided context', 'Based on the retrieved context', 'Based on the records', 'From the source', 'The retrieved information shows', or any similar wording. Instead, answer naturally by stating the information directly.\n\n"

            # f"{history_note}"

            f"Retrieved Context:\n{context}"
        )

        messages = history[-self._MAX_HISTORY:] + [{"role": "user", "content": question}]
        answer   = self._call_llm(provider, system_prompt, messages)
        answer   = self._guard_meta_commentary(answer)
        answer   = self._guard_mortality_answer(question, answer, context)

        return self._maybe_fallback({
            "question": question,
            "context":  hits,
            "answer":   answer,
            "context_sufficient": self._context_has_answer(answer),
        }, question, history, db, sql_provider, session)

    # ── SQL Fallback ─────────────────────────────────────────────────────────
    # When the vector/EMR-chunk search can't answer the question, fall back to
    # text-to-SQL against the live database so the same question (plus the
    # conversation so far, for pronoun/patient-reference resolution) still
    # gets a shot at an answer.

    # Gatekeeper for the SQL fallback path — only triggers _fallback_to_sql when the
    # vector-search answer was deemed insufficient and a DB session is available.
    def _maybe_fallback(
        self,
        result: dict,
        question: str,
        history: list[dict],
        db: Session | None,
        sql_provider: str,
        session: dict | None = None,
    ) -> dict:
        if result.get("context_sufficient") or db is None:
            return result
        fallback = self._fallback_to_sql(question, history, db, sql_provider, session)
        return fallback if fallback is not None else result

    # Delegates the question to RAGService's text-to-SQL pipeline and reshapes its
    # result (rows/answer/sql) into the same dict shape the vector-search path returns.
    def _fallback_to_sql(
        self,
        question: str,
        history: list[dict],
        db: Session,
        sql_provider: str,
        session: dict | None = None,
    ) -> dict | None:
        from app.services.rag_service import RAGService
        try:
            sql_result = RAGService(db, provider=sql_provider).ask(question, history=history, session=session)
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

    # Dispatches a chat completion request to whichever LLM provider is configured
    # (openai, anthropic, google, or local ollama) and returns the plain text response.
    def _call_llm(
        self,
        provider: str,
        system_prompt: str,
        messages: list[dict],
        temperature: float = 0.7,
    ) -> str:
        _t0 = time.perf_counter()
        try:
            return self._call_llm_inner(provider, system_prompt, messages, temperature)
        finally:
            print(f"[TIMING] _call_llm({provider}) took {time.perf_counter() - _t0:.2f}s")

    def _call_llm_inner(
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
            # num_ctx must stay constant across calls (Ollama only keeps a model
            # "warm" in memory for a given (model, num_ctx) pair -- switching values
            # forces a reload). It also must not be needlessly large: llama.cpp
            # allocates the KV cache up front sized to num_ctx, so a large value
            # reserves enough memory to push the model off the GPU (or into swap)
            # even for a one-line greeting, which is far more costly than any
            # prompt content. _MAX_CTX_CHARS already caps retrieved context well
            # under what this fits.
            #
            # keep_alive: Ollama's default is 5 minutes, after which the model is
            # evicted from GPU memory and the next call pays a full reload (measured
            # ~8s for llama3.2:3b) -- easily triggered by normal pauses between chat
            # turns, not just idle periods. Explicitly extend it so the model stays
            # resident across realistic gaps between messages.
            res = self.embed_client.chat(
                model=settings.OLLAMA_LLM_MODEL,
                messages=[{"role": "system", "content": system_prompt}] + messages,
                options={"num_ctx": settings.OLLAMA_NUM_CTX, "temperature": temperature},
                keep_alive=settings.OLLAMA_KEEP_ALIVE,
            )
            return res.message.content.strip()

        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")

    # ── Warm-up ──────────────────────────────────────────────────────────────

    # Forces both the LLM and embedding models into GPU memory ahead of the
    # user's first real message, so that message doesn't pay the multi-second
    # model-load cost itself. Meant to be called by the frontend as soon as a
    # chat session opens (e.g. on mount), not on every request.
    # - num_ctx matches the real chat calls so there's no later reload from a
    #   mismatched KV-cache size.
    # - num_predict=1 keeps the chat ping itself fast -- only the load matters.
    # - keep_alive matches settings.OLLAMA_KEEP_ALIVE so the model stays warm
    #   for the same duration a real call would keep it warm.
    def warm_up(self) -> dict:
        result = {"llm": False, "embed": False}
        try:
            self.embed_client.chat(
                model=settings.OLLAMA_LLM_MODEL,
                messages=[{"role": "user", "content": "hi"}],
                options={"num_predict": 1},
                keep_alive=settings.OLLAMA_KEEP_ALIVE,
            )
            result["llm"] = True
        except Exception as e:
            print(f"[WARMUP] failed to warm up LLM model: {e}")

        try:
            self.embed_client.embed(
                model=settings.OLLAMA_EMBED_MODEL,
                input="hi",
                keep_alive=settings.OLLAMA_KEEP_ALIVE,
            )
            result["embed"] = True
        except Exception as e:
            print(f"[WARMUP] failed to warm up embedding model: {e}")

        return result

