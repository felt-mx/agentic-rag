from pathlib import Path
from typing import List, Union, Literal
from core.models.chunk import Chunk
from ingestion.parsers import ParserRegistry
from ingestion.chunkers.late_chunker import LateChunker
from infra.milvus.client import upsert_chunks


class IngestionPipeline:
    def __init__(self, database: str = None):
        self.parser_registry = ParserRegistry()
        self.chunker = LateChunker()
        self.database = database

    async def ingest_file(self, file_path: Union[str, Path]) -> List[Chunk]:
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        document = await self.parser_registry.parse(file_path)
        chunks = await self.chunker.chunk_document(document)
        upsert_chunks(chunks, database=self.database)

        return chunks

    async def ingest_directory(self, dir_path: Union[str, Path]) -> List[Chunk]:
        dir_path = Path(dir_path)

        if not dir_path.exists() or not dir_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        all_chunks = []
        supported_extensions = [".txt", ".pdf", ".docx", ".doc"]

        files = [
            f
            for f in dir_path.rglob("*")
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]

        for i, file_path in enumerate(files, 1):
            try:
                chunks = await self.ingest_file(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error ingesting {file_path.name}: {e}\n")

        return all_chunks
