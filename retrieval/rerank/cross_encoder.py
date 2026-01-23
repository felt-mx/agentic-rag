import httpx
from typing import List
from retrieval.models import ScoredChunk
from .base import BaseReranker
from core.models.reranker import VLLMClient


class CrossEncoderReranker(BaseReranker):
    def __init__(self):
        self.vllm_client = VLLMClient()

    async def rerank(
        self, query: str, chunks: List[ScoredChunk], top_k: int = 5
    ) -> List[ScoredChunk]:
        if not chunks:
            return []

        documents = [chunk.text for chunk in chunks]

        rerank_scores = await self.vllm_client.rerank(query, documents, top_n=top_k)

        new_chunks = []
        for chunk, score in zip(chunks, rerank_scores):
            new_chunk = ScoredChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                section_id=chunk.section_id,
                text=chunk.text,
                metadata=chunk.metadata,
                score=score,
                source="cross_encoder_reranker",
            )
            new_chunks.append(new_chunk)

        new_chunks.sort(key=lambda x: x.score, reverse=True)

        return new_chunks[:top_k]
