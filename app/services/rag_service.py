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

    def _get_schema(self) -> tuple[str, set[str]]:
        inspector = inspect(self.db.get_bind())
        views     = inspector.get_view_names()
        parts     = []
        for view in views:
            cols     = inspector.get_columns(view)
            col_defs = ", ".join(f"{c['name']} ({c['type']})" for c in cols)
            parts.append(f"View: {view}\n  Columns: {col_defs}")
        return "\n\n".join(parts), set(views)

    # Table/view names the generated SQL actually references — used to catch
    # the LLM inventing a plausible-looking name (e.g. "_patient_case_diagnosis_vw")
    # that doesn't exist, instead of the real one ("_patient_diagnostics_vw").
    _TABLE_REF_PATTERN = re.compile(r'\b(?:FROM|JOIN)\s+`?([A-Za-z_][A-Za-z0-9_]*)`?', re.IGNORECASE)

    def _referenced_tables(self, sql: str) -> set[str]:
        return {m.group(1) for m in self._TABLE_REF_PATTERN.finditer(sql)}

    def _unknown_tables(self, sql: str, known_views: set[str]) -> set[str]:
        return self._referenced_tables(sql) - known_views

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

    def _generate_sql(
        self,
        question: str,
        schema: str,
        history: list[dict] | None = None,
        correction: str = "",
    ) -> str:
        history_block    = self._format_history(history)
        correction_block = f"{correction}\n\n" if correction else ""
        prompt = (
            f"Given this database schema:\n\n{schema}\n\n"
            f"JOIN rule: \\n"
            f"- If the selected table/view is `_patient_case_doctors_note_vw`, query it directly without any JOIN. "
            f"- Otherwise, LEFT JOIN the selected table/view with `_patient_case_doctors_note_vw` ON `patient_case_id`\\n"
            f"- Use the selected table/view as the primary table in the FROM clause.\\n"
            f"- Only perform this JOIN when the selected table/view is not `_patient_case_doctors_note_vw`."
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

    def _generate_answer(self, question: str, sql: str, results: list[dict], history: list[dict] | None = None) -> str:
        history_block = self._format_history(history)
        prompt = (
            f"{history_block}"
            f'A user asked: "{question}"\n\n'
            f"SQL executed:\n{sql}\n\n"
            f"Results:\n{json.dumps(results, indent=2, default=str)}\n\n"
            "Provide a clear, concise natural language answer based on these results."
        )
        return self._call_llm(prompt)

    def ask(self, question: str, history: list[dict] | None = None) -> dict:
        schema, known_views = self._get_schema()
        sql = self._generate_sql(question, schema, history)

        unknown = self._unknown_tables(sql, known_views)
        if unknown:
            # The LLM referenced a table/view that doesn't exist (usually a plausible-looking
            # guess) — give it one shot to self-correct with the mistake spelled out.
            correction = (
                f"Your previous query referenced unknown table/view name(s): {', '.join(sorted(unknown))}. "
                "These do not exist in the schema. Valid view names are exactly: "
                f"{', '.join(sorted(known_views))}. Regenerate the SQL using only these exact names."
            )
            sql = self._generate_sql(question, schema, history, correction=correction)
            unknown = self._unknown_tables(sql, known_views)
            if unknown:
                raise HTTPException(
                    status_code=400,
                    detail=f"Generated SQL references unknown table/view(s): {', '.join(sorted(unknown))}",
                )

        self._validate_sql(sql)
        results = self._execute_sql(sql)
        answer  = self._generate_answer(question, sql, results, history)
        return {"question": question, "sql": sql, "result": results, "answer": answer}
