from pymilvus import Collection, AnnSearchRequest, RRFRanker, WeightedRanker
import numpy as np
from typing import List, Optional
from infra.milvus.connection import connect_milvus
from infra.milvus.schema import create_chunk_collection
from retrieval.models import ScoredChunk


class HybridRetriever:
    def __init__(self, collection_name: str = "chunks", database: str = None):
        if connect_milvus(database=database):
            self.collection: Collection = create_chunk_collection(
                name=collection_name, database=database
            )
            self.collection.load()

    def retrieve(
        self,
        query_text: str,
        dense_embedding: np.ndarray,
        image_embedding: Optional[np.ndarray] = None,
        top_k: int = 5,
        rerank_method: str = "rrf",
        weights: Optional[List[float]] = None,
        use_image: bool = False,
    ) -> List[ScoredChunk]:
        search_requests = []

        # Dense vector search
        dense_request = AnnSearchRequest(
            data=[dense_embedding.tolist()],
            anns_field="dense_embedding",
            param={"metric_type": "IP", "params": {}},
            limit=top_k,
        )

        search_requests.append(dense_request)

        # BM25 sparse search
        sparse_request = AnnSearchRequest(
            # Pass raw text as Milvus will handle BM25 embedding
            data=[query_text],
            anns_field="sparse_embedding",
            param={
                "metric_type": "BM25",
                "params": {},
            },
            limit=top_k,
        )

        search_requests.append(sparse_request)

        # Optional image vector search
        if use_image and image_embedding is not None:
            image_request = AnnSearchRequest(
                data=[image_embedding.tolist()],
                anns_field="image_embedding",
                param={"metric_type": "IP", "params": {}},
                limit=top_k,
            )

            search_requests.append(image_request)

        # Choose ranker
        if rerank_method == "rrf":
            ranker = RRFRanker(k=60)
        elif rerank_method == "weighted":
            if weights is None:
                # Default weights based on number of searches
                num_searches = len(search_requests)
                weights = [1.0 / num_searches] * num_searches
            ranker = WeightedRanker(*weights)
        else:
            raise ValueError(f"Unsupported rerank method: {rerank_method}")

        # Perform hybrid search
        results = self.collection.hybrid_search(
            reqs=search_requests,
            rerank=ranker,
            limit=top_k,
            output_fields=["chunk_id", "document_id",
                           "section_id", "text", "metadata"],
        )

        hits = results[0]
        return self._to_scored_chunks(hits)

    def _to_scored_chunks(self, hits) -> List[ScoredChunk]:
        scored_chunks = []
        for hit in hits:
            chunk = ScoredChunk(
                chunk_id=hit.entity.get("chunk_id"),
                document_id=hit.entity.get("document_id"),
                section_id=hit.entity.get("section_id"),
                text=hit.entity.get("text"),
                metadata=hit.entity.get("metadata"),
                score=hit.score,
                source="hybrid",
            )
            scored_chunks.append(chunk)
        return scored_chunks
