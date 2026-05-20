"""Initialize Milvus collections for text and image embeddings."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.retriever import retriever_service
from app.core.config import settings


def init_milvus():
    """Initialize Milvus collections."""
    print("Initializing Milvus collections...")
    print(f"  Text collection: {settings.text_collection}")
    print(f"  Image collection: {settings.image_collection}")

    try:
        retriever_service.init_collections()
        print("Milvus collections initialized successfully!")
    except Exception as e:
        print(f"Error initializing Milvus: {e}")
        print("Make sure Milvus is running at localhost:19530")


if __name__ == "__main__":
    init_milvus()