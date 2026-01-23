import numpy as np
from typing import Optional
from retrieval.query.intake import intake_query
from retrieval.scoring.normalize import normalize_scores
from retrieval.retrieve.hybrid import HybridRetriever
from retrieval.rerank.cross_encoder import CrossEncoderReranker
from core.models.embedder import VLLMClient


class RetrievalPipeline:
    def __init__(self, dense_weight: float = 0.6, sparse_weight: float = 0.4):
        self.retriever = HybridRetriever()
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.reranker = CrossEncoderReranker()
        self.embedder = VLLMClient()

    async def retrieve(
        self,
        raw_query: str,
        top_k: int = 5,
        retrieval_k: int = 20,
        rerank_method: str = "weighted",
        image_query: Optional[np.ndarray] = None,
        use_image: bool = False,
        score_threshold: float = 0.7,
    ):
        query = intake_query(raw_query)

        # Generate dense embedding from the query text
        dense_embedding = await self.generate_dense_embedding(query_text=query["text"])

        # Generate image embedding if image query is provided
        image_embedding = None
        if use_image and image_query is not None:
            image_embedding = self.generate_image_embedding(image_query)

        weights = None
        if rerank_method == "weighted":
            if use_image and image_embedding is not None:
                # Normalize weights to sum to 1
                total = self.dense_weight + self.sparse_weight
                image_weight = 0.3
                dense_w = (self.dense_weight / total) * (1 - image_weight)
                sparse_w = (self.sparse_weight / total) * (1 - image_weight)
                weights = [dense_w, sparse_w, image_weight]
            else:
                # Normalize dense and sparse weights to sum to 1
                total = self.dense_weight + self.sparse_weight
                weights = [self.dense_weight / total, self.sparse_weight / total]

        # Perform hybrid retrieval
        results = self.retriever.retrieve(
            query_text=query["text"],
            dense_embedding=dense_embedding,
            image_embedding=image_embedding,
            top_k=retrieval_k,
            rerank_method=rerank_method,
            weights=weights,
            use_image=use_image,
        )

        if self.reranker and results:
            results = await self.reranker.rerank(
                query=query["text"],
                chunks=results,
                top_k=top_k,
            )

        # Normalize scores
        # results = normalize_scores(results)

        results_filtered = [r for r in results if r.score >= score_threshold]

        return results_filtered

    async def generate_dense_embedding(self, query_text: str) -> np.ndarray:
        embedding = await self.embedder.generate(query_text)
        return np.array(embedding)

    def generate_image_embedding(self, image_data: np.ndarray) -> np.ndarray:
        return np.random.rand(
            512
        )  # Placeholder for actual image embedding generation logic
