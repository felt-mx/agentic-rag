import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pymilvus import utility
from infra.milvus.connection import connect_milvus
from infra.milvus.schema import create_chunk_collection
from infra.milvus.indexes import create_dense_index
from configs.config import MILVUS_DATABASE


def bootstrap_milvus(database: str = None):
    db_name = database or MILVUS_DATABASE

    response = input(
        f"Bootstrap Milvus database '{db_name}'? This will reset the database with new schemas. (y/n): "
    )
    if response.lower() != "y":
        print("Bootstrap cancelled.")
        return

    connect_milvus(database=db_name)

    # Drop existing collection if it exists
    collection_name = "chunks"
    if utility.has_collection(collection_name, using="default"):
        utility.drop_collection(collection_name, using="default")

    collection = create_chunk_collection(database=db_name)
    create_dense_index(collection)
    collection.load()


if __name__ == "__main__":
    bootstrap_milvus()
