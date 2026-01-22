import numpy as np
from pymilvus import Collection
from infra.milvus.connection import connect_milvus
from infra.milvus.schema import create_chunk_collection
from retrieval.models import ScoredChunk


class DenseRetriever:
    def __init__(self, collection_name="chunks"):
        if connect_milvus():
            self.collection: Collection = create_chunk_collection(collection_name)
            self.collection.load()

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5):
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
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
        hits = results[0]

        return self._to_scored_chunks(hits)

    def _to_scored_chunks(self, hits) -> list[ScoredChunk]:
        scored = []
        for hit in hits:
            scored.append(
                ScoredChunk(
                    chunk_id=hit.entity.get("chunk_id"),
                    document_id=hit.entity.get("document_id"),
                    section_id=hit.entity.get("section_id"),
                    text=hit.entity.get("text"),
                    metadata=hit.entity.get("metadata"),
                    score=hit.score,
                    source="dense",
                )
            )
        return scored
