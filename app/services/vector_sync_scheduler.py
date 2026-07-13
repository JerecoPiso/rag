import asyncio
from app.core.database import SessionLocal
from app.services.vector_service import VectorService

# SYNC_INTERVAL_SECONDS = 2 * 60 * 60  # 2 hours
SYNC_INTERVAL_SECONDS = 3 * 60  # 3 minutes


# Background loop: every SYNC_INTERVAL_SECONDS, pulls rows added/changed since the
# last successful run and upserts them into Qdrant. The "since" cursor is persisted
# in setting_tbl (via VectorService.sync_latest) rather than kept in memory, so it
# survives app restarts too.
# Runs for the lifetime of the app process; started once from main.py's startup event.
async def run_periodic_vector_sync():
    while True:
        await asyncio.sleep(SYNC_INTERVAL_SECONDS)

        db = SessionLocal()
        try:
            result = VectorService().sync_latest(db)
            print(f"[VECTOR_SYNC] {result}")
        except Exception as e:
            print(f"[VECTOR_SYNC] failed: {e}")
        finally:
            db.close()
