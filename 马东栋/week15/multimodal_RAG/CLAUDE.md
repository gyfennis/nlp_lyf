# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Multimodal RAG System** - a retrieval-augmented generation system for multimodal documents (PDF, Word, images, tables). The system uses a multi-retrieval strategy with BM25 + BGE vector search, followed by reranking with a local BGE-reranker model, and generates answers via Qwen LLM (Alibaba Cloud).

## Architecture

```
User Query → Router Agent → Multi-Recall (BM25 + BGE) → Reranker → Generator (Qwen) → Answer
                            ↓
                    RRF Fusion (20 candidates)
```

### Key Components

- **Document Processing**: MinerU (local :8000) → Kafka queue → Chunking with overlap
- **Storage**: MySQL (metadata), Milvus Lite (vectors), Local filesystem (original docs)
- **Retrieval**: BM25 (top 10) + BGE vector (top 10) → RRF fusion → 20 candidates
- **Reranking**: Local BGE-reranker-base model → top 10
- **Generation**: Qwen via Alibaba Cloud API (uses `aliyunAPI_KEY` env var)

## Local Services & Ports

| Service | Port | Purpose |
|---------|------|---------|
| MinerU | 8000 | Document parsing (already deployed) |
| Embedding FastAPI | 8001 | BGE embedding service |
| Kafka | 9092 | Message queue |
| MySQL | 3306 | Metadata storage |
| Milvus Lite | file-based | Vector storage (`./milvus_lite.db`) |

## Required Environment Variables

```bash
aliyunAPI_KEY=your_api_key_here  # Alibaba Cloud API key for Qwen
```

## Commands

### Start Embedding Service (FastAPI on port 8001)
```bash
cd embedding_service && python main.py
```

### Start API Server
```bash
uvicorn src.main:app --reload --port 8000
```

### Start Kafka Consumer
```bash
python scripts/run_consumer.py
```

### Initialize Databases
```bash
python scripts/init_mysql.py
python scripts/init_milvus.py
```

## Local Model Paths

- **Embedding**: `E:/multimodal_RAG/models/bge-small-zh-v1.5` (also has nested `bge-small-zh-v1.5/` subdirectory)
- **Reranker**: `E:/multimodal_RAG/models/bge-reranker-base`

## Document Storage

Original documents are stored at: `E:/multimodal_RAG/documents/`

## Key Workflows

### Retrieval Flow
1. BM25召回 top 10
2. BGE向量召回 top 10 (via FastAPI at :8001)
3. RRF融合 → 20条候选
4. BGE-reranker本地打分 → top 10

### Document Processing Flow
1. Upload to local path → MySQL metadata
2. Kafka message queued
3. Consumer calls MinerU at :8000
4. MinerU returns parsed content
5. Chunking with overlap (chunk_size=512, overlap=128)
6. Generate embeddings via :8001
7. Store in Milvus Lite

## Tech Stack

- Agent Framework: openai-agents
- Databases: MySQL 8.0+, Milvus Lite 2.4+
- Message Queue: Kafka (kafka-python)
- Document Parser: MinerU (local :8000)
- Embedding: bge-small-zh-v1.5 via FastAPI
- Reranker: BAAI/bge-reranker-base (local)
- LLM: Qwen series via dashscope/Alibaba Cloud

## Project Status

Core implementation complete. All modules are implemented under `src/`. The `TECHNICAL_DOCUMENT.md` contains detailed specifications.

### Module Map

| Module | Location | Key Class/Function |
|--------|----------|-------------------|
| Config | `src/config.py` | `load_config()`, `get_model_path()` |
| MySQL | `src/storage/mysql_client.py` | `MySQLClient` |
| Milvus | `src/storage/milvus_client.py` | `MilvusVectorStore` |
| Embedding Service | `embedding_service/main.py` | FastAPI on :8001 |
| MinerU Client | `src/document_processing/mineru_client.py` | `MinerUClient` |
| Chunker | `src/document_processing/chunker.py` | `OverlapChunker` |
| Kafka Producer | `src/document_processing/kafka_producer.py` | `DocumentKafkaProducer` |
| Kafka Consumer | `src/document_processing/kafka_consumer.py` | `DocumentKafkaConsumer` |
| Upload Service | `src/document_processing/upload_service.py` | `UploadService` |
| BM25 Recall | `src/retrieval/bm25_recall.py` | `BM25Recall` |
| BGE Recall | `src/retrieval/bge_recall.py` | `BGERecall`, `EmbeddingClient` |
| RRF Fusion | `src/retrieval/fusion.py` | `rrf_fusion()` |
| Reranker | `src/rerank/bge_reranker.py` | `BGEReranker` |
| Qwen LLM | `src/llm/qwen_client.py` | `QwenLLM`, `build_rag_prompt()` |
| Agents | `src/agent/agents.py` | Agent definitions (openai-agents) |
| Workflow | `src/agent/workflow.py` | `RAGWorkflow` |
| API Routes | `src/api/routes/` | document upload + query endpoints |
| API Entry | `src/main.py` | FastAPI on :8000 |
