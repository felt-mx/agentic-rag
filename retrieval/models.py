from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ScoredChunk:
    chunk_id: str
    document_id: str
    section_id: str
    text: str
    metadata: Dict
    score: float  # Retriever-native score
    source: str  # dense | sparse | hybrid
