import sys
sys.path.insert(0, ".")

from src.storage.milvus_client import MilvusVectorStore


def init():
    store = MilvusVectorStore()
    store.create_collection(drop_existing=True)
    print(f"Milvus collection '{store.collection_name}' created successfully.")


if __name__ == "__main__":
    init()
