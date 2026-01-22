from abc import ABC, abstractmethod
from typing import List
from core.models.document import Document
from core.models.chunk import Chunk


class BaseChunker(ABC):
    @abstractmethod
    def chunk_document(self, document: Document) -> List[Chunk]:
        pass
