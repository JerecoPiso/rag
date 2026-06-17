import re
import json
from openai import OpenAI
from sqlalchemy import text, inspect
from sqlalchemy.orm import Session
from fastapi import HTTPException
from app.core.config import settings


class RAGService:
    def __init__(self, db: Session):
        self.db     = db
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def _get_schema(self) -> str:
        inspector   = inspect(self.db.get_bind())
        parts       = []
        for table in inspector.get_table_names():
            cols = inspector.get_columns(table)
            col_defs = ", ".join(f"{c['name']} ({c['type']})" for c in cols)
            parts.append(f"Table: {table}\n  Columns: {col_defs}")
        return "\n\n".join(parts)

    def _generate_sql(self, question: str, schema: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": (
                    f"Given this database schema:\n\n{schema}\n\n"
                    "Convert the following question to a valid SQL SELECT query. "
                    "Return ONLY the SQL query — no explanation, no markdown, no backticks.\n\n"
                    f"Question: {question}"
                )
            }]
        )
        sql = response.choices[0].message.content.strip()
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
        response = self.client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": (
                    f'A user asked: "{question}"\n\n'
                    f"SQL executed:\n{sql}\n\n"
                    f"Results:\n{json.dumps(results, indent=2, default=str)}\n\n"
                    "Provide a clear, concise natural language answer based on these results."
                )
            }]
        )
        return response.choices[0].message.content.strip()

    def ask(self, question: str) -> dict:
        schema  = self._get_schema()
        sql     = self._generate_sql(question, schema)
        self._validate_sql(sql)
        results = self._execute_sql(sql)
        answer  = self._generate_answer(question, sql, results)
        return {"question": question, "sql": sql, "result": results, "answer": answer}
    
