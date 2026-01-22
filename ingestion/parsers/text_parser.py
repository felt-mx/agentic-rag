import uuid
from pathlib import Path
from core.models.document import Document
from core.models.section import Section
from core.models.block import Block
from .base import BaseParser


class TextParser(BaseParser):
    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".txt"

    def parse(self, file_path: Path) -> Document:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        block = Block(
            block_id=str(uuid.uuid4()),
            type="text",
            content=content,
            metadata={},
        )

        section = Section(
            section_id=str(uuid.uuid4()),
            title=file_path.stem,
            level=1,
            blocks=[block],
            metadata={},
        )

        document = Document(
            document_id=str(uuid.uuid4()),
            sections=[section],
            metadata={
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": "txt",
            },
        )

        return document
