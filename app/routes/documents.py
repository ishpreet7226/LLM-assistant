from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
from loguru import logger

from app.models import IngestResponse, DocumentInfo
from app.services import vector_store, DocumentProcessor
from app.config import settings
from datetime import datetime

router = APIRouter(prefix="/api/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".html", ".rst"}
processor = DocumentProcessor()


@router.post("/upload", response_model=IngestResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and ingest a document into the vector store."""
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {ALLOWED_EXTENSIONS}"
        )

    # Save to disk
    save_dir = Path(settings.DOCUMENTS_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / file.filename

    content = await file.read()
    save_path.write_bytes(content)

    # Process and ingest
    try:
        chunks = processor.process_file(save_path)
        if not chunks:
            raise HTTPException(status_code=422, detail="No text could be extracted from the file.")

        count = vector_store.add_documents(chunks, file.filename, len(content))
        logger.info(f"Ingested '{file.filename}': {count} chunks")

        return IngestResponse(
            filename=file.filename,
            chunks_created=count,
            message=f"Successfully ingested '{file.filename}' into {count} chunks.",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion error for '{file.filename}': {e}")
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.get("/", response_model=list[DocumentInfo])
async def list_documents():
    """List all ingested documents."""
    docs = vector_store.get_documents()
    return [
        DocumentInfo(
            filename=d["filename"],
            size_bytes=d["size_bytes"],
            chunk_count=d["chunk_count"],
            uploaded_at=datetime.fromisoformat(d["uploaded_at"]),
            doc_type=d["doc_type"],
        )
        for d in docs
    ]


@router.delete("/{filename}")
async def delete_document(filename: str):
    """Remove a document from the vector store."""
    removed = vector_store.remove_document(filename)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found.")

    # Also delete from disk
    file_path = Path(settings.DOCUMENTS_DIR) / filename
    file_path.unlink(missing_ok=True)

    return {"message": f"Document '{filename}' removed successfully."}
