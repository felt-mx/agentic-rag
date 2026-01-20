from dataclasses import dataclass
from typing import List
from .block import Block


@dataclass(frozen=True)
class Section:
    section_id: str
    title: str
    level: int  # Hierarchical heading level
    blocks: List[Block]
    metadata: dict
