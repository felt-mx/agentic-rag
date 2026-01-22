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
        paragraphs = self.extract_paragraphs(text)

        chunk_texts = []
        chunk_spans = []
        current_chunk_paras = []
        current_word_count = 0
        current_start_char = 0

        char_position = 0

        for paragraph in paragraphs:
            paragraph_word_count = len(paragraph.split())

            # If adding this paragraph would exceed chunk size
            if current_word_count + paragraph_word_count > self.chunk_size:
                # Save current chunk if it's big enough
                if current_word_count >= self.min_chunk_size:
                    chunk_text = "\n\n".join(current_chunk_paras)
                    chunk_texts.append(chunk_text)
                    chunk_spans.append((current_start_char, char_position))

                    # Start new chunk
                    current_chunk_paras = [paragraph]
                    current_word_count = paragraph_word_count
                    current_start_char = char_position
                else:
                    # Too small to save, just add the paragraph
                    current_chunk_paras.append(paragraph)
                    current_word_count += paragraph_word_count
            else:
                current_chunk_paras.append(paragraph)
                current_word_count += paragraph_word_count

            # Update character position (para + double newline)
            char_position += len(paragraph) + 2

        # Add final chunk
        if current_chunk_paras and current_word_count >= self.min_chunk_size:
            chunk_text = "\n\n".join(current_chunk_paras)
            chunk_texts.append(chunk_text)
            chunk_spans.append((current_start_char, len(text)))

        return chunk_texts, chunk_spans

    def extract_paragraphs(self, text: str) -> List[str]:
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]
