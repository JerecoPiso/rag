import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.routes.api import router
from app.core.config import settings
from app.services.vector_sync_scheduler import run_periodic_vector_sync
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[LIFESPAN] starting background vector sync task...")
    task = asyncio.create_task(run_periodic_vector_sync())
    yield
    task.cancel()


app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

# app = FastAPI(title=settings.APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3003",
        "http://127.0.0.1:3003",
    ],
    allow_credentials=True,
    allow_methods=["*"],   # IMPORTANT
    allow_headers=["*"],   # IMPORTANT
)
app.include_router(router, prefix="/api")