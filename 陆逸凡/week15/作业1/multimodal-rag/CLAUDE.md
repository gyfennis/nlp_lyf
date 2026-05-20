# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multimodal RAG system for PDF document retrieval and question answering. The system parses PDF documents (using MinerU), stores text and image embeddings in Milvus, and generates answers using Qwen-VL.

## Project Structure

```
multimodal-rag/
├── app/
│   ├── api/           # FastAPI route handlers (upload, chat, knowledge_base)
│   ├── core/          # Configuration (config.py) and dependency injection
│   ├── models/         # SQLAlchemy models (database.py) and Pydantic schemas (schemas.py)
│   ├── services/      # Business logic services
│   │   ├── embedding.py   # BGE text encoding, CLIP image encoding, text chunking
│   │   ├── parser.py     # MinerU API integration for PDF parsing
│   │   ├── qa.py         # Qwen-VL answer generation
│   │   ├── retriever.py  # Milvus vector storage and search
│   │   └── storage.py    # PDF and parsed content file storage
│   ├── workers/
│   │   └── document_worker.py  # Kafka consumer for async document processing
│   └── main.py         # FastAPI app entry point
├── data/              # PDF storage, parsed content, and temp files
├── scripts/           # init_milvus.py, test_rag.py, evaluate.py
├── docker/            # Dockerfiles for API and worker services
└── docker-compose.yml # Full stack deployment (Milvus, Kafka, API, Worker)
```

## Commands

### Development
```bash
cd multimodal-rag
pip install -r requirements.txt

# Run API server
uvicorn app.main:app --reload --port 8000

# Run document worker (separate terminal)
python -m app.workers.document_worker

# Initialize Milvus collections
python scripts/init_milvus.py

# Run tests
python scripts/test_rag.py
```

### Docker Deployment
```bash
cd multimodal-rag
docker-compose up -d  # Start Milvus, Kafka, API, Worker
docker-compose down   # Stop all services
```

## Key Architecture Decisions

### Dual Vector Storage
- Text embeddings → `text_embeddings` Milvus collection (BGE model, 768-dim)
- Image embeddings → `image_embeddings` Milvus collection (CLIP ViT-B/32, 512-dim)
- Both collections use HNSW index for efficient similarity search

### Async Document Processing
1. Upload endpoint saves PDF and publishes `document_id|filepath` to Kafka topic `document_parse_tasks`
2. Worker consumes tasks, calls MinerU API, chunks text, encodes embeddings, inserts to Milvus
3. Processing status tracked in SQLite (`Document.status` field)

### RAG Flow
1. User query → BGE encode → search text collection
2. User query → CLIP encode → search image collection
3. Merge results → build multimodal prompt → Qwen-VL generate answer
4. Response includes citations with `[filename: page]` format

### Knowledge Base Isolation
- Each knowledge base has multiple documents
- Document IDs stored in vector embeddings for filtering
- Deleting a document removes its vectors from both collections

## Configuration

Key settings in `app/core/config.py`:
- `milvus_host/port` - Vector database connection
- `bge_model`, `clip_model` - Embedding model selection
- `qwen_api_base` - Qwen-VL API endpoint
- `kafka_bootstrap_servers`, `kafka_topic` - Message queue
- `mineru_api_url` - PDF parsing service
- `chunk_size=512`, `chunk_overlap=50` - Text splitting parameters

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload/document` | Upload PDF, returns task_id |
| GET | `/upload/status/{task_id}` | Check document processing status |
| DELETE | `/document/{doc_id}` | Delete document and vectors |
| POST | `/knowledge-bases` | Create new knowledge base |
| GET | `/knowledge-bases` | List all knowledge bases |
| POST | `/chat` | RAG question answering |

## Evaluation

The `scripts/evaluate.py` script computes:
- Page match score (0.25 weight) - cited pages in reference pages
- Filename match score (0.25 weight) - cited filenames match reference
- Content similarity (0.50 weight) - Jaccard similarity of answer texts

## Known Constraints

- MinerU API must be running separately (not included in docker-compose)
- Qwen-VL API must be running separately for production use
- Current implementation uses SQLite for development; MySQL available via env var