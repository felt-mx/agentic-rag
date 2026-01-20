from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass(frozen=True)
class Document:
    document_id: str
    metadata: Dict[str, Any]
    sections: List[str]
