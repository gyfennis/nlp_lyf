"""Test script for the RAG pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.embedding import embedding_service
from app.services.retriever import retriever_service


def test_embedding():
    """Test text embedding."""
    print("Testing BGE text embedding...")
    texts = ["这是一个测试句子", "Embedding models are useful"]
    embeddings = embedding_service.encode_texts(texts)
    print(f"  Generated {len(embeddings)} embeddings, shape: {embeddings[0].shape}")
    print("  Text embedding test passed!")


def test_chunking():
    """Test text chunking."""
    print("\nTesting text chunking...")
    text = "这是一个很长的文本，用于测试分块功能。" * 50
    chunks = embedding_service.chunk_text(text, chunk_size=100, overlap=20)
    print(f"  Generated {len(chunks)} chunks")
    print("  Chunking test passed!")


def test_milvus_connection():
    """Test Milvus connection."""
    print("\nTesting Milvus connection...")
    try:
        retriever_service.init_collections()
        print("  Milvus connection successful!")
    except Exception as e:
        print(f"  Milvus connection failed: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("RAG System Test Script")
    print("=" * 50)

    test_chunking()
    test_embedding()
    test_milvus_connection()

    print("\n" + "=" * 50)
    print("Tests completed!")
    print("=" * 50)