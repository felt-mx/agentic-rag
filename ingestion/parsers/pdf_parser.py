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

    def parse(self, file_path: Path) -> Document:
        sections = []

        with open(file_path, "rb") as f:
            pdf_reader = pypdf.PdfReader(f)

            for (
                page_num,
                page,
            ) in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()

                if text.strip():
                    block = Block(
                        block_id=str(uuid.uuid4()),
                        type="text",
                        content=text,
                        metadata={"page_number": page_num},
                    )

                    section = Section(
                        section_id=str(uuid.uuid4()),
                        title=f"Page {page_num}",
                        level=1,
                        blocks=[block],
                        metadata={"page_number": page_num},
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
