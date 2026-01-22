from pathlib import Path
from typing import List, Optional
from .base import BaseParser
from .text_parser import TextParser
from .pdf_parser import PDFParser
from .docx_parser import DocxParser


class ParserRegistry:
    def __init__(self):
        self.parsers: List[BaseParser] = []
        self.register_default_parsers()

    def register_default_parsers(self):
        try:
            self.parsers.append(TextParser())
        except ImportError:
            pass

        try:
            self.parsers.append(PDFParser())
        except ImportError:
            pass

        try:
            self.parsers.append(DocxParser())
        except ImportError:
            pass

    def register(self, parser: BaseParser):
        self.parsers.append(parser)

    def get_parser(self, file_path: Path) -> Optional[BaseParser]:
        for parser in self.parsers:
            if parser.supports(file_path):
                return parser
        return None

    def parse(self, file_path: Path):
        parser = self.get_parser(file_path)
        if parser is None:
            raise ValueError(f"No parser available for file type: {file_path.suffix}")
        return parser.parse(file_path)


__all__ = ["BaseParser", "ParserRegistry", "TextParser", "PDFParser", "DocxParser"]
