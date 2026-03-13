from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger

from app.routes import chat_router, documents_router
from app.services import embedding_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting RAG Assistant...")
    embedding_service.load()
    logger.info("✅ Embedding model ready")
    yield
    logger.info("🛑 Shutting down RAG Assistant")


app = FastAPI(
    title="RAG Assistant API",
    description="Retrieval-Augmented Generation system with Llama 3.3 and FAISS",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(chat_router)
app.include_router(documents_router)

# Serve static WebUI
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_ui():
        return FileResponse(str(static_dir / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok"}
