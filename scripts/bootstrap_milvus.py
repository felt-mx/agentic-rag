import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pymilvus import utility
from infra.milvus.connection import connect_milvus
from infra.milvus.schema import create_chunk_collection
from infra.milvus.indexes import create_dense_index


def bootstrap_milvus():
    connect_milvus()

    # Drop existing collection if it exists
    collection_name = "chunks"
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    collection = create_chunk_collection()
    create_dense_index(collection)
    collection.load()


if __name__ == "__main__":
    bootstrap_milvus()
