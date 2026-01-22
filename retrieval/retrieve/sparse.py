from pymilvus import Collection
import numpy as np
from typing import Dict
from infra.milvus.connection import connect_milvus
from infra.milvus.schema import create_chunk_collection


class SparseRetriever:
    def __init__(self, collection_name="chunks"):
        if connect_milvus():
            self.collection: Collection = create_chunk_collection(collection_name)
            self.collection.load()

    def retrieve(self, query_sparse_embedding: Dict[int, float], top_k: int = 5):
        results = self.collection.search()
