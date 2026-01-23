from abc import ABC, abstractmethod
from typing import List
from retrieval.models import ScoredChunk


class BaseReranker(ABC):
    @abstractmethod
    async def rerank(
        self,
        query: str,
        chunks: List[ScoredChunk],
        top_k: int = 5,
    ) -> List[ScoredChunk]:
        pass
