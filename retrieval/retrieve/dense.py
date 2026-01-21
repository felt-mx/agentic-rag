import numpy as np
from pymilvus import Collection
from infra.milvus.connection import connect_milvus
from infra.milvus.schema import create_chunk_collection


class DenseRetriever:
    def __init__(self, collection_name="chunks"):
        if connect_milvus():
            self.collection: Collection = create_chunk_collection(
                collection_name)
            self.collection.load()

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5):
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_Field="embedding",
            param={"metric_type": "IP", "params": {}},
            limit=top_k,
            output_fields=[
                "chunk_id",
                "document_id",
                "section_id",
                "text",
                "metadata",
            ],
        )

        return results[0]
