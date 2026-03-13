import json
from typing import AsyncIterator

import httpx
from loguru import logger

from app.config import settings


class OllamaService:
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_MODEL
        self.timeout = settings.OLLAMA_TIMEOUT

    async def check_health(self) -> tuple[bool, str]:
        """Check if Ollama is running and the model is available."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                if resp.status_code != 200:
                    return False, "Ollama unreachable"

                models = [m["name"] for m in resp.json().get("models", [])]
                # Accept partial name match (e.g. "llama3.3" matches "llama3.3:latest")
                matched = any(self.model in m for m in models)
                if not matched:
                    return False, f"Model '{self.model}' not found. Available: {models}"
                return True, "ok"
        except Exception as e:
            return False, str(e)

    def _build_prompt(self, query: str, context_chunks: list[tuple]) -> str:
        context_parts = []
        for i, (chunk, score) in enumerate(context_chunks, 1):
            context_parts.append(
                f"[Source {i}: {chunk.source}]\n{chunk.content}"
            )
        context = "\n\n---\n\n".join(context_parts)

        return (
            f"Context information:\n\n{context}\n\n"
            f"---\n\nUsing only the context above, answer the following question:\n\n"
            f"Question: {query}\n\nAnswer:"
        )

    async def generate(
        self,
        query: str,
        context_chunks: list[tuple],
        temperature: float = 0.7,
    ) -> str:
        """Non-streaming generation."""
        prompt = self._build_prompt(query, context_chunks)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": settings.SYSTEM_PROMPT,
            "stream": False,
            "options": {"temperature": temperature},
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            return resp.json()["response"]

    async def generate_stream(
        self,
        query: str,
        context_chunks: list[tuple],
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Streaming generation - yields token strings."""
        prompt = self._build_prompt(query, context_chunks)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": settings.SYSTEM_PROMPT,
            "stream": True,
            "options": {"temperature": temperature},
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST", f"{self.base_url}/api/generate", json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            yield token
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue


ollama_service = OllamaService()
