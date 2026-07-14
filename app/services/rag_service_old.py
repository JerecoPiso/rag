import re
import json
from sqlalchemy import text, inspect
from sqlalchemy.orm import Session
from fastapi import HTTPException
from app.core.config import settings


class RAGService:
    def __init__(self, db: Session, provider: str = "openai"):
        self.db       = db
        self.provider = provider.lower()
        self._init_client()

    def _init_client(self):
        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        elif self.provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        elif self.provider == "google":
            from google import genai
            self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        elif self.provider == "ollama":
            from ollama import Client as OllamaClient
            self.client = OllamaClient(host=settings.OLLAMA_URL)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown provider: {self.provider}")

    def _call_llm(self, prompt: str) -> str:
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model="gpt-4o",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()

        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()

        elif self.provider == "google":
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            return response.text.strip()

        elif self.provider == "ollama":
            response = self.client.chat(
                model=settings.OLLAMA_LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"num_ctx": 25000 },
            )
            return response.message.content.strip()

    def _get_schema(self) -> tuple[str, set[str], dict[str, set[str]]]:
        inspector       = inspect(self.db.get_bind())
        views           = inspector.get_view_names()
        parts           = []
        columns_by_view = {}
        for view in views:
            cols     = inspector.get_columns(view)
            col_defs = ", ".join(f"{c['name']} ({c['type']})" for c in cols)
            parts.append(f"View: {view}\n  Columns: {col_defs}")
            columns_by_view[view] = {c["name"] for c in cols}
        return "\n\n".join(parts), set(views), columns_by_view

    # What each view represents — used to help the LLM pick the right one for a
    # question before SQL is generated, instead of guessing from column names alone.
    _VIEW_DESCRIPTIONS: dict[str, str] = {
        "_patient_case_vital_vw":               "Clinical assessment / vital signs recorded for a patient case.",
        "_patient_case_nurses_note_vw":         "Nurses' notes for a patient case.",
        "_patient_case_doctors_note_vw":        "Doctors' notes/orders for a patient case.",
        "_patient_case_diet_vw":                "Diet orders for a patient case.",
        "_patient_case_medicine_vw":            "Medicines/prescriptions ordered for a patient case.",
        "_patient_case_medical_consumption_vw": "Medical supplies/items consumed for a patient case (e.g. IVF, oxygen).",
        "_patient_case_status_vw":              "Admission/discharge/case status: patient_status (INP/OPD/ER), discharge_date, discharge_type, case_classification, patient_type. And patient's case (diagnosis, chief complaint, etc.).",
        "_patient_tpr_vw":                      "Ward vital signs (temperature/pulse/respiration chart) for admitted patients.",
        "_patient_opr_vw":                      "Out-patient vital signs recorded during an OPD visit.",
        "_patient_monitor_vw":                  "Continuous monitoring vital signs.",
        "_patient_fluid_intake_and_output_vw":  "Fluid intake and output (FIAO) tracking.",
        "_patient_diagnostics_vw":              "Diagnostics/laboratory test results.",
        "_patient_info":                        "General patient demographic/basic info (name, id, etc.).",
        # "_patient_case_summary":                "Summary of a patient's case (diagnosis, chief complaint, etc.).",
    }

    # The actual date/timestamp column for each view — "latest"/"most recent"/"last"/
    # "first"/"earliest" questions must sort/filter on this column, not on an
    # unrelated column (e.g. `doctors_note_order`, which is an ordering/sequence
    # field, not a date).
    _DATE_COLUMNS: dict[str, str] = {
        "_patient_case_doctors_note_vw":        "doctors_note_date",
        "_patient_case_vital_vw":               "vital_capture_timestamp",
        "_patient_case_diet_vw":                "diet_date",
        "_patient_case_nurses_note_vw":         "nurses_note_date",
        "_patient_case_medicine_vw":            "medicine_date",
        "_patient_case_medical_consumption_vw": "medical_comsumption_date",
        "_patient_case_status_vw":              "discharge_date",
        "_patient_tpr_vw":                      "vital_capture_timestamp",
        "_patient_opr_vw":                      "vital_capture_timestamp",
        "_patient_monitor_vw":                  "vital_capture_timestamp",
        "_patient_fluid_intake_and_output_vw":  "vital_capture_timestamp",
        "_patient_diagnostics_vw":              "vital_capture_timestamp",
        "_patient_info":                        "created_timestamp",
    }

    def _select_table(
        self,
        question: str,
        known_views: set[str],
        history: list[dict] | None = None,
    ) -> str:
        """Ask the LLM to pick the single best-matching view for the question,
        from the fixed, described list of views. Returns "" if the answer isn't
        a recognized view name (caller then falls back to letting SQL generation
        pick from the full schema)."""
        catalog = "\n".join(
            f"- {view}: {desc}"
            for view, desc in self._VIEW_DESCRIPTIONS.items()
            if view in known_views
        )
        if not catalog:
            return ""

        history_block = self._format_history(history)
        prompt = (
            f"{history_block}"
            "Given this list of database views and what each one contains:\n\n"
            f"{catalog}\n\n"
            "Pick the ONE view that should be queried to answer the following question. "
            "Respond with ONLY the exact view name, character-for-character as listed above — "
            "no explanation, no markdown, no punctuation.\n\n"
            f"Question: {question}"
        )
        answer = self._call_llm(prompt).strip()
        answer = re.sub(r"^`|`$", "", answer).strip()
        return answer if answer in known_views else ""

    # Table/view names the generated SQL actually references — used to catch
    # the LLM inventing a plausible-looking name (e.g. "_patient_case_diagnosis_vw")
    # that doesn't exist, instead of the real one ("_patient_diagnostics_vw").
    _TABLE_REF_PATTERN = re.compile(r'\b(?:FROM|JOIN)\s+`?([A-Za-z_][A-Za-z0-9_]*)`?', re.IGNORECASE)

    def _referenced_tables(self, sql: str) -> set[str]:
        return {m.group(1) for m in self._TABLE_REF_PATTERN.finditer(sql)}

    def _unknown_tables(self, sql: str, known_views: set[str]) -> set[str]:
        return self._referenced_tables(sql) - known_views

    # Column names the generated SQL actually references — used to catch the LLM
    # inventing a plausible-looking column (e.g. "complaint", "final_diagnosis")
    # that doesn't exist on the selected view.
    _SQL_KEYWORDS = frozenset({
        "select", "from", "where", "and", "or", "not", "in", "is", "null",
        "like", "between", "order", "by", "group", "having", "limit", "offset",
        "asc", "desc", "as", "distinct", "join", "left", "right", "inner", "outer",
        "on", "case", "when", "then", "else", "end", "true", "false",
        "lower", "upper", "max", "min", "count", "sum", "avg", "coalesce",
        "concat", "trim", "date", "now", "cast", "convert", "exists", "all", "any",
        "union", "intersect", "except", "into",
    })
    _STRING_LITERAL_PATTERN = re.compile(r"'(?:[^'\\]|\\.)*'")
    _ALIAS_PATTERN          = re.compile(r'\bAS\s+`?([A-Za-z_][A-Za-z0-9_]*)`?', re.IGNORECASE)
    _IDENTIFIER_PATTERN     = re.compile(r'`([A-Za-z_][A-Za-z0-9_]*)`|\b([A-Za-z_][A-Za-z0-9_]*)\b')

    def _referenced_columns(self, sql: str, known_views: set[str]) -> set[str]:
        cleaned    = self._STRING_LITERAL_PATTERN.sub("''", sql)
        aliases    = {m.group(1).lower() for m in self._ALIAS_PATTERN.finditer(cleaned)}
        views_lower = {v.lower() for v in known_views}
        idents = set()
        for m in self._IDENTIFIER_PATTERN.finditer(cleaned):
            token = (m.group(1) or m.group(2)).lower()
            if token in self._SQL_KEYWORDS or token in views_lower or token in aliases:
                continue
            idents.add(token)
        return idents

    def _unknown_columns(
        self,
        sql: str,
        known_views: set[str],
        columns_by_view: dict[str, set[str]],
    ) -> set[str]:
        referenced_tables = self._referenced_tables(sql) & known_views
        if not referenced_tables:
            return set()
        valid_columns = {
            col.lower()
            for table in referenced_tables
            for col in columns_by_view.get(table, set())
        }
        return {
            col for col in self._referenced_columns(sql, known_views)
            if col not in valid_columns
        }

    @staticmethod
    def _format_history(history: list[dict] | None) -> str:
        if not history:
            return ""
        convo = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history)
        return (
            "Conversation history (use it to resolve references like 'he', 'she', "
            "'that patient', or a patient named earlier if this question doesn't repeat it):\n"
            f"{convo}\n\n"
        )

    def _resolve_patient_name(
        self,
        question: str,
        history: list[dict] | None = None,
        session: dict | None = None,
    ) -> str:
        """
        Determine which patient this question is about, so that follow-up questions
        that don't repeat the name (e.g. "what about his medicines?") still resolve
        to the right patient instead of relying on the SQL-generation LLM to notice
        the reference buried in raw history text.

        - If the question itself explicitly names a patient, that name wins and is
          stored in `session` (so it carries forward to later turns).
        - Otherwise, if a patient is already active in `session`, reuse it.
        - Otherwise, ask the LLM to look at the conversation history for the most
          recently mentioned patient.
        Returns "" if no patient can be determined.
        """
        prompt = (
            f"{self._format_history(history)}"
            "Determine which patient the following question refers to.\n"
            "- If the question itself explicitly names a patient, return that name exactly as written.\n"
            "- If the question uses a pronoun or implicit reference ('he', 'she', 'him', 'her', "
            "'the patient', 'that patient') or names no patient at all, look at the conversation "
            "history above and return the most recently mentioned patient's name.\n"
            "- If no patient name can be determined from the question or history, respond with "
            "exactly: NONE\n\n"
            "Respond with ONLY the patient's name, or NONE — no explanation, no punctuation, no markdown.\n\n"
            f"Question: {question}"
        )
        name = self._call_llm(prompt).strip().strip("`\"'")

        if not name or name.upper() == "NONE":
            return (session or {}).get("patient_name", "")

        if session is not None:
            session["patient_name"] = name
        return name

    def _generate_sql(
        self,
        question: str,
        schema: str,
        history: list[dict] | None = None,
        correction: str = "",
        selected_table: str = "",
        patient_name: str = "",
        selected_columns: set[str] | None = None,
    ) -> str:
        history_block    = self._format_history(history)
        correction_block = f"{correction}\n\n" if correction else ""
        patient_name_block = (
            f"The patient this question refers to is: {patient_name}. Even if the question itself "
            "doesn't repeat the name (e.g. it uses 'he'/'she'/'the patient' or names no one), filter "
            "on this patient using the name-matching rules below.\n\n"
            if patient_name else ""
        )
        date_column = self._DATE_COLUMNS.get(selected_table, "")
        date_column_block = (
            f"When this question asks for the latest, most recent, last, first, "
            f"earliest, or a specific date, base that on the `{date_column}` column "
            f"(e.g. ORDER BY `{date_column}` DESC/ASC, or MAX(`{date_column}`)/MIN(`{date_column}`)) — "
            f"never on an unrelated column such as an ordering/sequence field.\n\n"
            if date_column else ""
        )
        columns_block = (
            f"The ONLY columns that exist on `{selected_table}` are: "
            f"{', '.join(sorted(selected_columns))}. Use ONLY these exact column names — never invent, "
            "guess, or assume a column exists (e.g. `complaint`, `final_diagnosis`) just because it "
            "sounds plausible for the question. If no column here matches what the question is asking "
            "for, pick the closest actual column from this list instead of inventing one.\n\n"
            if selected_columns else ""
        )
        selected_table_block = (
            f"The table/view to use for this question is: `{selected_table}`. "
            "Query it directly as the only table in the FROM clause — do NOT JOIN it with any other "
            "table/view. Every view already contains its own patient name/identifying columns.\n"
            f"{date_column_block}"
            f"{columns_block}"
            if selected_table else ""
        )
        prompt = (
            f"Given this database schema:\n\n{schema}\n\n"
            f"{selected_table_block}"
            "- Query only the selected table/view. Do not JOIN it with any other table/view — "
            "each view already includes the patient's name and identifying info directly.\n"
            f"{patient_name_block}"
            f"{history_block}"
            "Convert the following question to a valid SQL SELECT query.\n"
            ""
            "STRICT RULES:\n"
            "- Output ONLY a SELECT statement. No INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, or any other statement.\n"
            "- Do NOT use markdown, backticks, or any explanation — raw SQL only.\n"
            "- The query must begin with SELECT.\n"
            "- Only reference table/view names that appear EXACTLY (character-for-character) in the "
            "schema above. Never invent, pluralize, or guess a name based on a naming pattern — if you "
            "are not sure a view exists, re-check the schema list rather than assuming it.\n"
            "- If the question contains a person's name (patient or doctor), do NOT use exact match (=). "
            "Use LIKE with wildcards instead, e.g. name LIKE '%mark%jhapet%', and match against each "
            "name part separately combined with AND/OR so partial or misspelled names still match.\n"
            "- Names in the database may be stored in ALL CAPS — use UPPER() or LOWER() on both sides "
            "of the LIKE comparison to make matching case-insensitive, e.g. "
            "LOWER(patient_name) LIKE LOWER('%mark%') AND LOWER(patient_name) LIKE LOWER('%jhapet%').\n"
            "- If the question asks for a count of admitted, outpatient, or ER/emergency consultation "
            "patients, use the `patient_status` column on `_patient_case_status_vw`: "
            "'INP' = admitted, 'OPD' = outpatient, 'ER' = ER consultation.\n"
            "- If the question asks for a count of discharges (e.g. 'discharged today', 'discharged "
            "this week'), filter on the `discharge_date` column to select the relevant day(s), and use "
            "the `discharge_type` column to identify records that represent an actual discharge.\n"
            "- Only filter on `case_classification` if the question itself explicitly names a "
            "classification (e.g. 'medicine', 'pedia', 'ob', 'newborn', 'new born', 'surgery'). Do NOT "
            "add a `case_classification` filter for questions that don't mention one — e.g. a plain "
            "'is there a patient named X' or 'find patient X' question must NOT be filtered by "
            "classification. When a classification IS named, use the `patient_type` column too if the "
            "question also distinguishes patient type. Values in `case_classification` may be prefixed "
            "with the patient type, e.g. 'OPD Surgery', 'OPD Medicine', 'OPD Pedia' — so match with "
            "LIKE '%surgery%' (case-insensitive) rather than an exact match, unless the question "
            "explicitly asks to exclude OPD/other prefixes.\n\n"
            f"{correction_block}"
            f"Question: {question}"
        )
        sql = self._call_llm(prompt)
        sql = re.sub(r"^```(?:sql)?\n?", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"\n?```$", "", sql)
        return sql.strip()

    def _validate_sql(self, sql: str) -> None:
        normalized = sql.strip().upper()
        if not normalized.startswith("SELECT"):
            raise HTTPException(status_code=400, detail="Only SELECT queries are allowed")
        forbidden = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "TRUNCATE", "CREATE", "EXEC", "EXECUTE"]
        for kw in forbidden:
            if re.search(rf"\b{kw}\b", normalized):
                raise HTTPException(status_code=400, detail=f"Query contains forbidden keyword: {kw}")

    def _execute_sql(self, sql: str) -> list[dict]:
        result  = self.db.execute(text(sql))
        columns = list(result.keys())
        rows    = result.fetchall()
        return [dict(zip(columns, row)) for row in rows]

    # Caps on how much of the raw SQL result set gets serialized into the answer-generation
    # prompt. A broad/unfiltered query (e.g. no WHERE clause) can return thousands of rows,
    # which previously blew straight past the model's context window (128k tokens for gpt-4o)
    # since the full JSON dump was inlined verbatim.
    _MAX_RESULT_ROWS  = 200
    _MAX_RESULT_CHARS = 60_000

    def _generate_answer(self, question: str, sql: str, results: list[dict], history: list[dict] | None = None) -> str:
        history_block = self._format_history(history)
        total_rows   = len(results)
        capped_rows  = results[: self._MAX_RESULT_ROWS]
        results_json = json.dumps(capped_rows, indent=2, default=str)
        if len(results_json) > self._MAX_RESULT_CHARS:
            results_json = results_json[: self._MAX_RESULT_CHARS] + "\n... (truncated)"
        truncation_note = (
            f"\n(Showing the first {len(capped_rows)} of {total_rows} total rows — the rest were "
            "truncated to fit. If the question needs an exact total across all rows, say the count is "
            "based on a partial result set rather than guessing.)\n"
            if total_rows > len(capped_rows) else ""
        )
        prompt = (
            f"{history_block}"
            f'A user asked: "{question}"\n\n'
            f"SQL executed:\n{sql}\n\n"
            f"Results:\n{results_json}\n"
            f"{truncation_note}\n"
            "Provide a clear, concise natural language answer based on these results.\n"
            "- Answer ONLY what the question specifically asked for — e.g. if the question asks for "
            "doctor's orders, answer with only the doctor's orders, not other fields from the results "
            "(vitals, medicines, diet, status, etc.) even if they're present in the data above.\n"
            "- Do not volunteer additional information the question didn't ask for.\n"
            "- Return the smallest amount of information necessary to accurately answer the question."
        )
        return self._call_llm(prompt)

    _GREETING_WORDS = frozenset({
        "hello", "hi", "hey", "thanks", "thank", "good",
        "bye", "goodbye", "morning", "afternoon", "evening",
    })

    def _is_greeting(self, question: str) -> bool:
        words = set(re.findall(r"[a-z']+", question.lower()))
        return bool(words & self._GREETING_WORDS) and len(question.split()) <= 6

    def ask(self, question: str, history: list[dict] | None = None, session: dict | None = None) -> dict:
        if self._is_greeting(question):
            answer = self._call_llm(
                "You are a helpful medical assistant for a hospital system. "
                "Respond naturally and briefly to this greeting or small talk. "
                "Do not discuss topics outside the medical domain.\n\n"
                f"Message: {question}"
            )
            return {"question": question, "sql": "", "result": [], "answer": answer}

        schema, known_views, columns_by_view = self._get_schema()
        patient_name     = self._resolve_patient_name(question, history, session)
        selected_table   = self._select_table(question, known_views, history)
        selected_columns = columns_by_view.get(selected_table)
        sql = self._generate_sql(
            question, schema, history, selected_table=selected_table,
            patient_name=patient_name, selected_columns=selected_columns,
        )

        unknown = self._unknown_tables(sql, known_views)
        if unknown:
            # The LLM referenced a table/view that doesn't exist (usually a plausible-looking
            # guess) — give it one shot to self-correct with the mistake spelled out.
            correction = (
                f"Your previous query referenced unknown table/view name(s): {', '.join(sorted(unknown))}. "
                "These do not exist in the schema. Valid view names are exactly: "
                f"{', '.join(sorted(known_views))}. Regenerate the SQL using only these exact names."
            )
            sql = self._generate_sql(
                question, schema, history, correction=correction,
                selected_table=selected_table, patient_name=patient_name, selected_columns=selected_columns,
            )
            unknown = self._unknown_tables(sql, known_views)
            if unknown:
                raise HTTPException(
                    status_code=400,
                    detail=f"Generated SQL references unknown table/view(s): {', '.join(sorted(unknown))}",
                )

        unknown_cols = self._unknown_columns(sql, known_views, columns_by_view)
        if unknown_cols:
            # The LLM invented a column that doesn't exist on the referenced view(s)
            # (e.g. `complaint`, `final_diagnosis`) — give it one shot to self-correct.
            referenced_tables = self._referenced_tables(sql) & known_views
            valid_cols = sorted({
                col for table in referenced_tables for col in columns_by_view.get(table, set())
            })
            correction = (
                f"Your previous query referenced unknown column name(s): {', '.join(sorted(unknown_cols))}. "
                "These do not exist on the referenced view(s). Valid column names are exactly: "
                f"{', '.join(valid_cols)}. Regenerate the SQL using only these exact column names — if none "
                "of them match what the question asked for, pick the closest actual column instead of "
                "inventing one."
            )
            sql = self._generate_sql(
                question, schema, history, correction=correction,
                selected_table=selected_table, patient_name=patient_name, selected_columns=selected_columns,
            )
            unknown_cols = self._unknown_columns(sql, known_views, columns_by_view)
            if unknown_cols:
                raise HTTPException(
                    status_code=400,
                    detail=f"Generated SQL references unknown column(s): {', '.join(sorted(unknown_cols))}",
                )

        self._validate_sql(sql)
        results = self._execute_sql(sql)
        answer  = self._generate_answer(question, sql, results, history)
        return {"question": question, "sql": sql, "result": results, "answer": answer}
