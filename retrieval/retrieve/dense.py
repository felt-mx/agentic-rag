import numpy as np
from pymilvus import Collection
from infra.milvus.connection import connect_milvus
from infra.milvus.schema import create_chunk_collection
from retrieval.models import ScoredChunk


class DenseRetriever:
    def __init__(self, collection_name="chunks"):
        if connect_milvus():
            self.collection: Collection = create_chunk_collection(
                collection_name)
            self.collection.load()

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> list[ScoredChunk]:
        hits = self.collection.search(...)
        results = []

        for hit in hits[0]:
            results.append(
                ScoredChunk(
                    chunk_id=hit.entity.get("chunk_id"),
                    document_id=hit.entity.get("document_id"),
                    section_id=hit.entity.get("section_id"),
                    text=hit.entity.get("text"),
                    metadata=hit.entity.get("metadata"),
                    score=float(hit.score),
                    source="dense",
                )
            )

        return results
