from dataclasses import dataclass
from typing import Any, Dict
import numpy as np


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    section_id: str
    text: str
    metadata: Dict[str, Any]
    embedding: np.ndarray = None  # Optional embedding vector for now
