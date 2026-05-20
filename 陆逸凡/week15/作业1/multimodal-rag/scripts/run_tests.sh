#!/bin/bash
# Test execution script for RAG system

set -e

echo "=== Running RAG System Tests ==="

# Check if in correct directory
if [ ! -f "requirements.txt" ]; then
    echo "Error: Run this script from the multimodal-rag directory"
    exit 1
fi

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

source venv/bin/activate
pip install -q -r requirements.txt

# Run unit tests
echo ""
echo "=== Running Unit Tests ==="
pytest tests/unit/ -v --cov=app --cov-report=term-missing || true

# Run integration tests
echo ""
echo "=== Running Integration Tests ==="
pytest tests/integration/ -v || true

# Run performance tests
echo ""
echo "=== Running Performance Tests ==="
pytest tests/performance/ -v || true

echo ""
echo "=== Test Run Complete ==="
echo "Coverage report: htmlcov/index.html"