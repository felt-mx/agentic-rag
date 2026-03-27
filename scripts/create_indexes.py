import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from infra.milvus.connection import connect_milvus
from pymilvus import Collection

connect_milvus()
collection = Collection("chunks")

# 1. Release to unlock the collection
collection.release()

# 2. Identify and Drop ALL existing indexes
# This clears out the index that's demanding a GPU
print("Dropping all existing indexes...")
for index in collection.indexes:
    print(f"Dropping: {index.index_name} on field: {index.field_name}")
    collection.drop_index(index_name=index.index_name)

# 3. Create CPU-friendly indexes
print("Creating CPU indexes...")

# Dense Vector Index (using HNSW for CPU)
collection.create_index(    
    field_name="dense_embedding",
    index_params={
        "index_type": "HNSW",
        "metric_type": "IP",
        "params": {"M": 16, "efConstruction": 128},
    },
)

# Sparse Vector Index (BM25)
collection.create_index(
    field_name="sparse_embedding",
    index_params={
        "index_type": "SPARSE_INVERTED_INDEX",
        "metric_type": "BM25",
        "params": {"drop_ratio_build": 0.0},
    },
)

# Image Vector Index (if you use it)
collection.create_index(
    field_name="image_embedding",
    index_params={
        "index_type": "HNSW",
        "metric_type": "IP",
        "params": {"M": 16, "efConstruction": 128},
    },
)

# 4. Load back into memory
print("Loading collection...")
collection.load()
print("Loading completed.")
