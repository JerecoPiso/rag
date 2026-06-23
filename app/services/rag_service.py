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

    def _get_schema(self) -> str:
        inspector = inspect(self.db.get_bind())
        parts     = []
        for view in inspector.get_view_names():
            print(view)
            cols     = inspector.get_columns(view)
            col_defs = ", ".join(f"{c['name']} ({c['type']})" for c in cols)
            parts.append(f"View: {view}\n  Columns: {col_defs}")
        return "\n\n".join(parts)

    def _generate_sql(self, question: str, schema: str) -> str:
        prompt = (
            f"Given this database schema:\n\n{schema}\n\n"
            "Convert the following question to a valid SQL SELECT query. "
            "Return ONLY the SQL query — no explanation, no markdown, no backticks.\n\n"
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

    def _generate_answer(self, question: str, sql: str, results: list[dict]) -> str:
        prompt = (
            f'A user asked: "{question}"\n\n'
            f"SQL executed:\n{sql}\n\n"
            f"Results:\n{json.dumps(results, indent=2, default=str)}\n\n"
            "Provide a clear, concise natural language answer based on these results."
        )
        return self._call_llm(prompt)

    def ask(self, question: str) -> dict:
        schema  = self._get_schema()
        sql     = self._generate_sql(question, schema)
        self._validate_sql(sql)
        results = self._execute_sql(sql)
        answer  = self._generate_answer(question, sql, results)
        return {"question": question, "sql": sql, "result": results, "answer": answer}
