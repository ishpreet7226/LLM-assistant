import re
from pathlib import Path
from typing import Optional
from loguru import logger

from app.config import settings


class DocumentChunk:
    def __init__(self, content: str, source: str, chunk_id: int, metadata: dict = None):
        self.content = content
        self.source = source
        self.chunk_id = chunk_id
        self.metadata = metadata or {}


class DocumentProcessor:
    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP

    def process_file(self, file_path: Path) -> list[DocumentChunk]:
        """Process a file and return list of text chunks."""
        suffix = file_path.suffix.lower()
        logger.info(f"Processing file: {file_path.name} ({suffix})")

        try:
            if suffix == ".pdf":
                text = self._extract_pdf(file_path)
            elif suffix in (".docx", ".doc"):
                text = self._extract_docx(file_path)
            elif suffix in (".txt", ".md", ".rst"):
                text = self._extract_text(file_path)
            elif suffix in (".html", ".htm"):
                text = self._extract_html(file_path)
            else:
                text = self._extract_text(file_path)

            chunks = self._chunk_text(text, file_path.name)
            logger.info(f"Created {len(chunks)} chunks from {file_path.name}")
            return chunks

        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            raise

    def process_text(self, text: str, source_name: str) -> list[DocumentChunk]:
        """Process raw text and return chunks."""
        return self._chunk_text(text, source_name)

    def _extract_pdf(self, path: Path) -> str:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        return "\n\n".join(pages)

    def _extract_docx(self, path: Path) -> str:
        import docx
        doc = docx.Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)

    def _extract_text(self, path: Path) -> str:
        import chardet
        raw = path.read_bytes()
        encoding = chardet.detect(raw)["encoding"] or "utf-8"
        return raw.decode(encoding, errors="replace")

    def _extract_html(self, path: Path) -> str:
        from bs4 import BeautifulSoup
        html = self._extract_text(path)
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)

    def _chunk_text(self, text: str, source: str) -> list[DocumentChunk]:
        """Split text into overlapping chunks using sentence-aware splitting."""
        # Clean the text
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = text.strip()

        if not text:
            return []

        # Split into sentences/paragraphs
        paragraphs = re.split(r'\n\n+', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_id = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_size = len(para.split())

            if current_size + para_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    source=source,
                    chunk_id=chunk_id
                ))
                chunk_id += 1

                # Overlap: keep last portion
                overlap_words = self.chunk_overlap
                words = chunk_text.split()
                if len(words) > overlap_words:
                    overlap_text = " ".join(words[-overlap_words:])
                    current_chunk = [overlap_text]
                    current_size = overlap_words
                else:
                    current_chunk = []
                    current_size = 0

            current_chunk.append(para)
            current_size += para_size

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(DocumentChunk(
                content="\n\n".join(current_chunk),
                source=source,
                chunk_id=chunk_id
            ))

        return chunks
