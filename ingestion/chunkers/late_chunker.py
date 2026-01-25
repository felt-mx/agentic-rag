import uuid
import re
from typing import List, Tuple
import numpy as np
from core.models.document import Document
from core.models.chunk import Chunk
from core.models.embedder import VLLMClient
from .base import BaseChunker


class LateChunker(BaseChunker):
    def __init__(self):
        self.embedder = VLLMClient()

    async def chunk_document(self, document: Document) -> List[Chunk]:
        chunks = []

        for section in document.sections:
            section_text = " ".join(
                [block.content for block in section.blocks if block.type == "text"]
            )

            if not section_text.strip():
                continue

            section_chunks = await self.late_chunk_section(
                section_text,
                document.document_id,
                section.section_id,
                document.metadata,
                section.title,
            )

            chunks.extend(section_chunks)

        return chunks

    async def late_chunk_section(
        self,
        text: str,
        document_id: str,
        section_id: str,
        metadata: dict,
        section_title: str,
    ) -> List[Chunk]:
        if not text.strip():
            return []

        # Pass text directly to API - API handles chunking and embedding
        chunk_results = await self.embedder.late_chunking_embed(
            text=text, task="retrieval.passage", late_chunking=True, batch_size=4096
        )

        print(f"Processing section '{section_title}' - Chunks: {len(chunk_results)}")

        chunks = []
        for result in chunk_results:
            chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                section_id=section_id,
                text=result["text"],
                metadata={
                    **metadata,
                    "section_title": section_title,
                    "chunking_method": "late_chunking",
                    "has_full_content": True,
                },
                dense_embedding=np.array(result["embedding"]),
                image_embedding=None,
            )
            chunks.append(chunk)

        return chunks
