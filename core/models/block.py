from dataclasses import dataclass
from typing import Any, Dict, Literal

BlockType = Literal["text", "table", "image", "code", "list", "unknown"]


@dataclass(frozen=True)
class Block:
    block_id: str
    type: BlockType
    content: Any
    metadata: Dict[str, Any]
