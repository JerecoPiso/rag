"""One-off: rebuild only the patient_info Qdrant collection from MySQL.

Needed after the _SOURCE_ID_FIELD fix in emr_formatter.py (patient_info's
patient_id metadata was pointing at the wrong column) — this re-ingests just
that collection instead of doing a full 14-collection resync.
"""
from app.core.database import SessionLocal
from app.services.vector_service import VectorService
from app.utils.emr_formatter import format_record, build_metadata

SOURCE = "_patient_info"

db = SessionLocal()
try:
    svc = VectorService(collection_name="rag_documents")
    collection_name = svc._collection_for_source(SOURCE)

    from sqlalchemy import text as sa_text
    result  = db.execute(sa_text(f"SELECT * FROM `{SOURCE}`"))
    columns = list(result.keys())
    rows    = result.fetchall()

    svc._recreate_collection(collection_name)

    texts, metadatas = [], []
    for row in rows:
        row_dict      = {col: val for col, val in zip(columns, row)}
        row_dict_text = {col: val for col, val in row_dict.items() if not svc._is_id_column(col)}
        texts.append(format_record(row_dict_text, SOURCE))
        metadatas.append(build_metadata(row_dict, SOURCE))

    count = svc.ingest(texts, metadatas, collection_name=collection_name)
    print(f"Re-ingested {count} rows into '{collection_name}'")
finally:
    db.close()
