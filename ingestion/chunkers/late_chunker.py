import uuid
import re
from typing import List, Tuple
import numpy as np
from core.models.document import Document
from core.models.chunk import Chunk
from core.models.embedder import VLLMClient
from .base import BaseChunker


class LateChunker(BaseChunker):
    def __init__(
        self, chunk_size: int = 256, overlap: int = 50, min_chunk_size: int = 50
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
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
        if not text.strip():
            return []

        # Pass text directly to API - API handles chunking and embedding
        chunk_results = await self.embedder.late_chunking_embed(
            text=text, task="retrieval.passage", late_chunking=True, batch_size=10000
        )

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

    def split_with_spans(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        # 1. Split by single newline to be more granular
        lines = [m for m in re.finditer(r"([^\n]+)", text)]

        chunk_texts = []
        chunk_spans = []

        # Store line info: (text, start, end, word_count)
        line_info = []
        for match in lines:
            line_text = match.group(0)
            line_info.append(
                (line_text, match.start(), match.end(), len(line_text.split()))
            )

        if not line_info:
            return chunk_texts, chunk_spans

        current_chunk_lines = []
        current_start = -1
        current_words = 0
        chunk_start_line_idx = 0

        for i, (line_text, line_start, line_end, line_words) in enumerate(line_info):
            if current_start == -1:
                current_start = line_start
                chunk_start_line_idx = i

            # Check if adding this line exceeds the limit
            if current_words + line_words > self.chunk_size and current_chunk_lines:
                # Save the current chunk
                chunk_texts.append("\n".join(current_chunk_lines))
                chunk_spans.append(
                    (current_start, line_info[i - 1][2])
                )  # end of previous line

                # Calculate overlap: find lines from the end that fit within overlap word count
                overlap_lines = []
                overlap_words = 0
                overlap_start_idx = i  # Default to current line (no overlap)

                # Walk backwards from the end of the current chunk to collect overlap lines
                for j in range(len(current_chunk_lines) - 1, -1, -1):
                    line_word_count = line_info[chunk_start_line_idx + j][3]
                    if overlap_words + line_word_count <= self.overlap:
                        overlap_lines.insert(0, current_chunk_lines[j])
                        overlap_words += line_word_count
                        overlap_start_idx = chunk_start_line_idx + j
                    else:
                        break

                # Start new chunk with overlap lines plus current line
                if overlap_lines and self.overlap > 0:
                    current_chunk_lines = overlap_lines + [line_text]
                    current_start = line_info[overlap_start_idx][1]
                    current_words = overlap_words + line_words
                    chunk_start_line_idx = overlap_start_idx
                else:
                    current_chunk_lines = [line_text]
                    current_start = line_start
                    current_words = line_words
                    chunk_start_line_idx = i
            else:
                current_chunk_lines.append(line_text)
                current_words += line_words

            # Handle the "Massive Line" case:
            # If a single line is STILL over the limit, force-split it by sentences/words
            if current_words > self.chunk_size and len(current_chunk_lines) == 1:
                massive_line = current_chunk_lines[0]
                sub_texts, sub_spans = self.split_massive_line(
                    massive_line, current_start
                )

                # Add all but the last sub-chunk to results
                for j, (sub_text, sub_span) in enumerate(
                    zip(sub_texts[:-1], sub_spans[:-1])
                ):
                    chunk_texts.append(sub_text)
                    chunk_spans.append(sub_span)

                # Keep the last sub-chunk as the current chunk for potential overlap
                if sub_texts:
                    current_chunk_lines = [sub_texts[-1]]
                    current_start = sub_spans[-1][0]
                    current_words = len(sub_texts[-1].split())
                else:
                    current_chunk_lines = []
                    current_start = -1
                    current_words = 0

        # Add the final leftover chunk
        if current_chunk_lines:
            chunk_texts.append("\n".join(current_chunk_lines))
            chunk_spans.append((current_start, len(text)))

        return chunk_texts, chunk_spans

    def split_massive_line(
        self, line: str, base_offset: int
    ) -> Tuple[List[str], List[Tuple[int, int]]]:
        texts = []
        spans = []

        # First, try to split by sentences
        sentence_pattern = re.compile(r"([^.!?]*[.!?]+(?:\s|$))")
        sentences = sentence_pattern.findall(line)

        if not sentences or len(sentences) == 1:
            # No sentence boundaries found, try splitting by clauses (commas, semicolons)
            clause_pattern = re.compile(r"([^,;]*[,;]?\s*)")
            sentences = [s for s in clause_pattern.findall(line) if s.strip()]

        if not sentences:
            # Last resort: split by words
            sentences = self.split_by_words(line, self.chunk_size)

        current_text = ""
        current_start = base_offset

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            potential_text = (
                (current_text + " " + sentence).strip() if current_text else sentence
            )

            if len(potential_text.split()) > self.chunk_size and current_text:
                # Save current chunk
                texts.append(current_text)
                end_pos = base_offset + line.find(current_text) + len(current_text)
                spans.append((current_start, end_pos))

                # Start new chunk
                current_text = sentence
                current_start = base_offset + line.find(sentence, end_pos - base_offset)
                if current_start < base_offset:
                    current_start = end_pos
            else:
                current_text = potential_text

        # Add remaining text
        if current_text:
            texts.append(current_text)
            spans.append((current_start, base_offset + len(line)))

        return texts, spans

    def split_by_words(self, text: str, max_words: int) -> List[str]:
        words = text.split()
        chunks = []

        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i : i + max_words])
            chunks.append(chunk)

        return chunks

    def extract_paragraphs(self, text: str) -> List[str]:
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]
