import csv
import sys
import uuid
from pathlib import Path
from typing import Set

from core.models.block import Block
from core.models.document import Document
from core.models.section import Section
from .base import BaseParser

# Increase CSV field size limit to handle large context fields
# Windows C long is often 32-bit even on 64-bit systems, causing OverflowError with sys.maxsize
max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int / 10)


class CSVParser(BaseParser):
    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".csv"

    async def parse(self, file_path: Path) -> Document:
        sections = []
        seen_contexts: Set[str] = set()

        doc_meta = {
            "source": str(file_path),
            "file_name": file_path.name,
            "file_type": "csv",
        }

        row_count = 0
        unique_count = 0

        with open(file_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)

            # Check if 'context' column exists
            fieldnames = reader.fieldnames or []
            has_context = "context" in fieldnames
            has_title = "title" in fieldnames

            for row in reader:
                row_count += 1

                if has_context:
                    text_content = row["context"]
                else:
                    # Fallback: join all values if no 'context' column
                    text_content = "\n".join(f"{k}: {v}" for k, v in row.items() if v)

                if not text_content.strip():
                    continue

                # Handle repeats: Deduplicate based on content
                content_hash = text_content.strip()
                if content_hash in seen_contexts:
                    continue

                seen_contexts.add(content_hash)
                unique_count += 1

                # Construct final text with title if available
                final_text = text_content
                if has_title and row.get("title"):
                    final_text = f"Title: {row['title']}\n\n{text_content}"

                # Create Block and Section
                block = Block(
                    block_id=str(uuid.uuid4()),
                    type="text",
                    content=final_text,
                    metadata={"row_number": row_count},
                )

                section_metadata = {
                    "row_number": row_count,
                    "original_id": row.get("id", ""),
                }
                # Add other metadata fields if needed, e.g. question
                if "question" in row:
                    section_metadata["associated_question"] = row["question"]

                sections.append(
                    Section(
                        section_id=str(uuid.uuid4()),
                        title=f"Row {row_count}"
                        + (
                            f": {row['title']}"
                            if has_title and row.get("title")
                            else ""
                        ),
                        level=1,
                        blocks=[block],
                        metadata=section_metadata,
                    )
                )

        return Document(
            document_id=str(uuid.uuid4()),
            metadata={
                **doc_meta,
                "total_rows": row_count,
                "unique_contexts": unique_count,
            },
            sections=sections,
        )
