from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    top_k: Optional[int] = Field(default=5, ge=1, le=20)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)


class SourceChunk(BaseModel):
    content: str
    source: str
    score: float
    chunk_id: int


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    query: str
    model: str
    elapsed_ms: int


class DocumentInfo(BaseModel):
    filename: str
    size_bytes: int
    chunk_count: int
    uploaded_at: datetime
    doc_type: str


class IngestResponse(BaseModel):
    filename: str
    chunks_created: int
    message: str


class StoreStats(BaseModel):
    total_documents: int
    total_chunks: int
    embedding_model: str
    llm_model: str
    ollama_status: str
