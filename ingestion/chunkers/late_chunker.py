import uuid
import re
from typing import List, Tuple
import numpy as np
from core.models.document import Document
from core.models.chunk import Chunk
from core.models.embedder import VLLMClient
from .base import BaseChunker


class LateChunker(BaseChunker):
    def __init__(self, chunk_size: int = 512, min_chunk_size: int = 100):
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size
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
        chunk_texts, chunk_spans = self.split_with_spans(text)

        if not chunk_texts:
            return []

        embeddings = await self.embedder.late_chunking_embed(
            text=text,
            chunk_spans=chunk_spans,
            task="retrieval.passage",
        )

        chunks = []
        for chunk_text, embedding in zip(chunk_texts, embeddings):
            chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                section_id=section_id,
                text=chunk_text,
                metadata={
                    **metadata,
                    "section_title": section_title,
                    "chunking_method": "late_chunking",
                    "has_full_content": True,
                },
                dense_embedding=np.array(embedding),
                image_embedding=None,
            )
            chunks.append(chunk)

        return chunks

    def split_with_spans(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        # 1. Split by single newline to be more granular
        lines = [m for m in re.finditer(r"([^\n]+)", text)]

        chunk_texts = []
        chunk_spans = []

        current_chunk_lines = []
        current_start = -1
        current_words = 0

        for i, match in enumerate(lines):
            line_text = match.group(0)
            line_start = match.start()
            line_end = match.end()

            if current_start == -1:
                current_start = line_start

            line_words = len(line_text.split())

            # Check if adding this line exceeds the limit
            if current_words + line_words > self.chunk_size and current_chunk_lines:
                # Save the current chunk
                chunk_texts.append("\n".join(current_chunk_lines))
                chunk_spans.append((current_start, lines[i - 1].end()))

                # Reset
                current_chunk_lines = [line_text]
                current_start = line_start
                current_words = line_words
            else:
                current_chunk_lines.append(line_text)
                current_words += line_words

            # Handle the "Massive Line" case:
            # If a single line is STILL over the limit, force-split it by character
            if current_words > self.chunk_size:
                # Simple character-based split for brevity here
                # In production, use a more sophisticated sentence-breaker
                pass

        # Add the final leftover chunk
        if current_chunk_lines:
            chunk_texts.append("\n".join(current_chunk_lines))
            chunk_spans.append((current_start, len(text)))

        return chunk_texts, chunk_spans

    def extract_paragraphs(self, text: str) -> List[str]:
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]
