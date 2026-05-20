# RAG System Tests

## Test Structure

```
tests/
├── test_data.json          # Test queries with expected answers
├── unit/                   # Unit tests
│   ├── test_parser.py      # Document parsing tests
│   ├── test_embedding.py   # BGE/CLIP embedding tests
│   ├── test_retriever.py   # Milvus retrieval tests
│   └── test_qa.py          # Qwen-VL QA tests
├── integration/            # Integration tests
│   └── test_integration.py # Full pipeline tests
├── performance/            # Performance tests
│   └── test_performance.py # Latency and concurrency tests
└── eval/                   # Evaluation modules
    ├── evaluator.py        # RAG evaluator
    └── report_generator.py # HTML report generation
```

## Running Tests

```bash
cd multimodal-rag

# Run all unit tests
pytest tests/unit/ -v

# Run integration tests
pytest tests/integration/ -v

# Run performance tests
pytest tests/performance/ -v

# Run with coverage
pytest tests/unit/ -v --cov=app --cov-report=html

# Run evaluation
python eval/evaluator.py tests/test_data.json --output results.json
python eval/report_generator.py results.json
```

## Test Data Format

Test queries in `test_data.json`:
- `id`: Query identifier (Q001, Q002, ...)
- `question`: Test question
- `expected_answer`: Expected answer text
- `expected_source`: Expected source (filename, page, type)

## Evaluation Metrics

| Metric | Weight | Target |
|--------|--------|--------|
| Filename Match | 0.25 | >= 0.85 |
| Page Match | 0.25 | >= 0.80 |
| Content Similarity (Jaccard) | 0.50 | >= 0.70 |
| **Total** | 1.00 | >= 0.75 |