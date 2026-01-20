from dataclasses import dataclass
from typing import List, Dict, Any
from .section import Section


@dataclass(frozen=True)
class Document:
    document_id: str
    metadata: Dict[str, Any]
    sections: List[Section]
