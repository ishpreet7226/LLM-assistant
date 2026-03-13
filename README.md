# RAG Assistant

A production-ready Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **Ollama**, **FAISS**, and **sentence-transformers**. Works with any Ollama model — pick one that fits your hardware.

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
         └── Ollama LLM Service   ← Any Ollama model via HTTP API
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

# Start Ollama (if not running as a service)
ollama serve
```

### 3. Choose & Pull an Ollama Model

> **Pick a model based on your available RAM.** Larger models give better answers but need more memory.

| Model | RAM Required | Speed | Quality | Pull Command |
|---|---|---|---|---|
| `tinyllama` | ~1 GB | ⚡ Very Fast | Basic | `ollama pull tinyllama` |
| `llama3.2:1b` | ~2 GB | ⚡ Very Fast | Good | `ollama pull llama3.2:1b` |
| `llama3.2:3b` | ~4 GB | 🚀 Fast | Very Good | `ollama pull llama3.2:3b` |
| `mistral` | ~5 GB | 🚀 Fast | Very Good | `ollama pull mistral` |
| `llama3.1:8b` | ~8 GB | 🟡 Moderate | Great | `ollama pull llama3.1:8b` |
| `llama3.3` | ~16 GB | 🔴 Slow on CPU | Best | `ollama pull llama3.3` |
| `mixtral` | ~26 GB | 🔴 Slow on CPU | Best | `ollama pull mixtral` |

**Recommendations:**
- 💻 **4 GB RAM or less** → `llama3.2:1b` or `tinyllama`
- 💻 **8 GB RAM (MacBook Air / M-series)** → `llama3.2:3b` ✅ *(default)*
- 🖥️ **16 GB RAM** → `llama3.1:8b` or `mistral`
- 🖥️ **32 GB+ RAM / GPU** → `llama3.3` or `mixtral`

After pulling your chosen model, set it in `.env`:
```bash
OLLAMA_MODEL=llama3.2:3b   # replace with your chosen model
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
| `OLLAMA_MODEL` | `llama3.2:3b` | Any Ollama model |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `CHUNK_SIZE` | `512` | Words per chunk |
| `CHUNK_OVERLAP` | `64` | Overlapping words between chunks |
| `TOP_K_RESULTS` | `5` | Retrieved chunks per query |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |

### Switching models at any time
```bash
# 1. Pull the new model
ollama pull mistral

# 2. Update .env
OLLAMA_MODEL=mistral

# 3. Restart the app
python run.py
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

3. **Generation**: Retrieved chunks injected as context into a structured prompt → streamed through Ollama → tokens sent to the browser via Server-Sent Events.
