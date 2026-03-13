# RAG Assistant

A production-ready Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **Ollama (Llama 3.3)**, **FAISS**, and **sentence-transformers**.

---

## Architecture

```
User Query
    │
    ▼
FastAPI Backend
    ├── /api/chat/stream  ← Streaming RAG endpoint (SSE)
    ├── /api/chat         ← Non-streaming RAG endpoint
    └── /api/documents    ← Document management
         │
         ├── Document Processor   ← PDF, DOCX, TXT, MD, HTML
         │       └── Chunking (512 tokens, 64 overlap)
         │
         ├── Embedding Service    ← sentence-transformers (all-MiniLM-L6-v2)
         │       └── 384-dim float32 vectors
         │
         ├── FAISS Vector Store   ← IndexFlatIP (cosine similarity)
         │       └── Persisted to disk (faiss_index.index + .meta)
         │
         └── Ollama LLM Service   ← Llama 3.3 via HTTP API
                 └── Streaming token generation
```

---

## Prerequisites

### 1. Python 3.10+

### 2. Ollama
Install from [ollama.com](https://ollama.com):
```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Pull Llama 3.3
ollama pull llama3.3

# Start Ollama (if not running as a service)
ollama serve
```

---

## Setup

```bash
# Clone / enter the project
cd rag-assistant

# Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy env config (optional — defaults are sensible)
cp .env.example .env
```

---

## Run

```bash
python run.py
```

Then open **http://localhost:8000** in your browser.

The API docs are available at **http://localhost:8000/docs**.

---

## Usage

### Via WebUI
1. Open `http://localhost:8000`
2. Upload documents using the left sidebar (PDF, TXT, MD, DOCX, HTML)
3. Type your question and hit **Send**
4. Expand **Context Sources** to see which chunks informed the answer

### Via API

**Upload a document:**
```bash
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@your_document.pdf"
```

**Query (streaming):**
```bash
curl -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?", "top_k": 5, "temperature": 0.7}'
```

**Query (non-streaming):**
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize the key findings", "top_k": 5}'
```

**List documents:**
```bash
curl http://localhost:8000/api/documents/
```

**Delete a document:**
```bash
curl -X DELETE http://localhost:8000/api/documents/my_file.pdf
```

---

## Configuration

Edit `.env` to customise:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.3` | Any Ollama model |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `CHUNK_SIZE` | `512` | Words per chunk |
| `CHUNK_OVERLAP` | `64` | Overlapping words between chunks |
| `TOP_K_RESULTS` | `5` | Retrieved chunks per query |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |

### Using a different model
```bash
ollama pull mistral
# Set in .env:
OLLAMA_MODEL=mistral
```

### Using a larger embedding model
```
EMBEDDING_MODEL=all-mpnet-base-v2
EMBEDDING_DIMENSION=768
```
> ⚠️ Changing the embedding model requires re-ingesting all documents.

---

## Project Structure

```
rag-assistant/
├── app/
│   ├── main.py                  # FastAPI app + lifespan
│   ├── config.py                # Pydantic settings
│   ├── models.py                # Request/response schemas
│   ├── routes/
│   │   ├── chat.py              # /api/chat, /api/chat/stream, /api/stats
│   │   └── documents.py         # /api/documents CRUD
│   ├── services/
│   │   ├── embeddings.py        # sentence-transformers singleton
│   │   ├── vector_store.py      # FAISS index + persistence
│   │   ├── document_processor.py# File parsing + chunking
│   │   └── llm.py               # Ollama HTTP client + streaming
│   └── static/
│       └── index.html           # WebUI
├── data/documents/              # Uploaded source files
├── vector_store/                # FAISS index files (auto-created)
├── run.py                       # Entrypoint
├── requirements.txt
└── .env.example
```

---

## How RAG Works Here

1. **Ingestion**: File → extracted text → split into 512-word overlapping chunks → embedded with `sentence-transformers` → stored in FAISS with cosine similarity (IndexFlatIP on L2-normalised vectors).

2. **Retrieval**: User query → embedded → top-K nearest chunks retrieved from FAISS by cosine similarity score.

3. **Generation**: Retrieved chunks injected as context into a structured prompt → streamed through Ollama/Llama 3.3 → tokens sent to the browser via Server-Sent Events.
