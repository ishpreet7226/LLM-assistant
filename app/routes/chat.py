import time
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from app.models import ChatRequest, ChatResponse, SourceChunk, StoreStats
from app.services import vector_store, ollama_service
from app.config import settings

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Non-streaming RAG query endpoint."""
    start = time.monotonic()

    # 1. Retrieve relevant chunks
    results = vector_store.search(req.query, top_k=req.top_k)
    if not results:
        raise HTTPException(
            status_code=404,
            detail="No documents in the knowledge base. Please upload some documents first."
        )

    # 2. Generate answer
    try:
        answer = await ollama_service.generate(
            query=req.query,
            context_chunks=results,
            temperature=req.temperature,
        )
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise HTTPException(status_code=503, detail=f"LLM error: {str(e)}")

    elapsed_ms = int((time.monotonic() - start) * 1000)

    sources = [
        SourceChunk(
            content=chunk.content[:500],
            source=chunk.source,
            score=round(score, 4),
            chunk_id=chunk.chunk_id,
        )
        for chunk, score in results
    ]

    return ChatResponse(
        answer=answer,
        sources=sources,
        query=req.query,
        model=settings.OLLAMA_MODEL,
        elapsed_ms=elapsed_ms,
    )


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Streaming RAG query - returns Server-Sent Events."""
    results = vector_store.search(req.query, top_k=req.top_k)

    if not results:
        async def err_gen():
            yield f"data: {json.dumps({'error': 'No documents in the knowledge base.'})}\n\n"
        return StreamingResponse(err_gen(), media_type="text/event-stream")

    sources = [
        {
            "content": chunk.content[:400],
            "source": chunk.source,
            "score": round(score, 4),
            "chunk_id": chunk.chunk_id,
        }
        for chunk, score in results
    ]

    async def event_generator():
        # Send sources first
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        # Stream tokens
        try:
            async for token in ollama_service.generate_stream(
                query=req.query,
                context_chunks=results,
                temperature=req.temperature,
            ):
                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/stats", response_model=StoreStats)
async def get_stats():
    """Return knowledge base and model stats."""
    ok, msg = await ollama_service.check_health()
    return StoreStats(
        total_documents=vector_store.total_documents,
        total_chunks=vector_store.total_chunks,
        embedding_model=settings.EMBEDDING_MODEL,
        llm_model=settings.OLLAMA_MODEL,
        ollama_status="online" if ok else f"offline: {msg}",
    )
