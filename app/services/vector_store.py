import json
import pickle
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import faiss
import numpy as np
from loguru import logger

from app.config import settings
from app.services.embeddings import embedding_service
from app.services.document_processor import DocumentChunk


@dataclass
class StoredChunk:
    content: str
    source: str
    chunk_id: int
    doc_index: int
    added_at: str


class VectorStore:
    def __init__(self):
        self.index_path = Path(settings.VECTOR_STORE_PATH)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        self.faiss_index: Optional[faiss.Index] = None
        self.chunks: list[StoredChunk] = []
        self.doc_registry: dict[str, dict] = {}  # filename -> metadata
        self.dimension = settings.EMBEDDING_DIMENSION

        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        index_file = Path(f"{self.index_path}.index")
        meta_file = Path(f"{self.index_path}.meta")

        if index_file.exists() and meta_file.exists():
            try:
                self.faiss_index = faiss.read_index(str(index_file))
                with open(meta_file, "rb") as f:
                    data = pickle.load(f)
                self.chunks = data["chunks"]
                self.doc_registry = data.get("doc_registry", {})
                logger.info(
                    f"Loaded vector store: {len(self.chunks)} chunks, "
                    f"{self.faiss_index.ntotal} vectors"
                )
            except Exception as e:
                logger.warning(f"Could not load existing index: {e}. Starting fresh.")
                self._init_fresh()
        else:
            logger.info("No existing index found. Creating new vector store.")
            self._init_fresh()

    def _init_fresh(self):
        # Inner-product index on L2-normalised vectors == cosine similarity
        self.faiss_index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []
        self.doc_registry = {}

    def _save(self):
        faiss.write_index(self.faiss_index, f"{self.index_path}.index")
        with open(f"{self.index_path}.meta", "wb") as f:
            pickle.dump({"chunks": self.chunks, "doc_registry": self.doc_registry}, f)

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def add_documents(self, chunks: list[DocumentChunk], filename: str, file_size: int) -> int:
        if not chunks:
            return 0

        texts = [c.content for c in chunks]
        embeddings = embedding_service.embed(texts)

        base_idx = len(self.chunks)
        stored = [
            StoredChunk(
                content=c.content,
                source=c.source,
                chunk_id=c.chunk_id,
                doc_index=base_idx + i,
                added_at=datetime.utcnow().isoformat(),
            )
            for i, c in enumerate(chunks)
        ]

        self.faiss_index.add(embeddings)
        self.chunks.extend(stored)

        self.doc_registry[filename] = {
            "filename": filename,
            "size_bytes": file_size,
            "chunk_count": len(chunks),
            "uploaded_at": datetime.utcnow().isoformat(),
            "doc_type": Path(filename).suffix.lstrip(".").upper(),
        }
        self._save()
        logger.info(f"Added {len(chunks)} chunks for '{filename}'")
        return len(chunks)

    def remove_document(self, filename: str) -> bool:
        """Remove all chunks for a document and rebuild the index."""
        if filename not in self.doc_registry:
            return False

        self.chunks = [c for c in self.chunks if c.source != filename]
        del self.doc_registry[filename]

        # Rebuild index from remaining chunks
        self._init_fresh()
        if self.chunks:
            texts = [c.content for c in self.chunks]
            embeddings = embedding_service.embed(texts)
            self.faiss_index.add(embeddings)

        self._save()
        logger.info(f"Removed document '{filename}'")
        return True

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> list[tuple[StoredChunk, float]]:
        if self.faiss_index.ntotal == 0:
            return []

        query_vec = embedding_service.embed_single(query).reshape(1, -1)
        k = min(top_k, self.faiss_index.ntotal)
        scores, indices = self.faiss_index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @property
    def total_documents(self) -> int:
        return len(self.doc_registry)

    def get_documents(self) -> list[dict]:
        return list(self.doc_registry.values())


vector_store = VectorStore()
