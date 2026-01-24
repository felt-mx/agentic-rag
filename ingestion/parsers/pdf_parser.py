import uuid
import pypdf
from pathlib import Path
from core.models.document import Document
from core.models.section import Section
from core.models.block import Block
from .base import BaseParser


class PDFParser(BaseParser):
    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".pdf"

    def parse(self, file_path: Path, next_page_lines: int = 30) -> Document:
        sections = []

        with open(file_path, "rb") as f:
            pdf_reader = pypdf.PdfReader(f)
            total_pages = len(pdf_reader.pages)

            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()

                # Add context from next page
                if page_num < total_pages:
                    next_page = pdf_reader.pages[page_num]
                    next_page_text = next_page.extract_text()

                    # Take first N lines from next page
                    lines = next_page_text.split("\n")
                    context_lines = lines[:next_page_lines]
                    context_from_next = "\n".join(context_lines)

                    # Combine with separator
                    combined_text = f"{text}\n{context_from_next}"
                else:
                    combined_text = text

                if combined_text.strip():
                    block = Block(
                        block_id=str(uuid.uuid4()),
                        type="text",
                        content=combined_text,
                        metadata={
                            "page_number": page_num,
                            "has_next_page_context": page_num < total_pages,
                        },
                    )

                    section = Section(
                        section_id=str(uuid.uuid4()),
                        title=f"Page {page_num}",
                        level=1,
                        blocks=[block],
                        metadata={
                            "page_number": page_num,
                            "has_next_page_context": page_num < total_pages,
                        },
                    )

                    sections.append(section)

        document = Document(
            document_id=str(uuid.uuid4()),
            sections=sections,
            metadata={
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": "pdf",
                "total_pages": len(sections),
            },
        )

        return document
