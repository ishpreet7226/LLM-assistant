import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from app.config import settings


class EmbeddingService:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self):
        if self._model is None:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.info("Embedding model loaded successfully")

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts, returns (N, D) float32 array."""
        if self._model is None:
            self.load()
        embeddings = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text, returns (D,) float32 array."""
        return self.embed([text])[0]


embedding_service = EmbeddingService()
