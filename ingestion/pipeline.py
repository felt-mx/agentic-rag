import logging
import shutil
from pathlib import Path
from typing import List, Union, Literal
from core.models.chunk import Chunk
from ingestion.parsers import ParserRegistry
from ingestion.chunkers.late_chunker import LateChunker
from infra.milvus.client import upsert_chunks

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self, database: str = None):
        self.parser_registry = ParserRegistry()
        self.chunker = LateChunker()
        self.database = database

    async def ingest_file(
        self,
        file_path: Union[str, Path],
        completed_dir: Union[str, Path, None] = None,
    ) -> List[Chunk]:
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info("Parsing: %s", file_path.name)
        document = await self.parser_registry.parse(file_path)
        logger.info("Chunking: %s", file_path.name)
        chunks = await self.chunker.chunk_document(document)
        logger.info("Upserting %d chunk(s) from: %s", len(chunks), file_path.name)
        upsert_chunks(chunks, database=self.database)

        dest_dir = Path(completed_dir) if completed_dir else file_path.parent / "completed"
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(file_path), str(dest_dir / file_path.name))
        logger.info("Moved to completed: %s", file_path.name)

        return chunks

    async def ingest_directory(self, dir_path: Union[str, Path]) -> List[Chunk]:
        dir_path = Path(dir_path)

        if not dir_path.exists() or not dir_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        all_chunks = []
        supported_extensions = [".txt", ".pdf", ".docx", ".doc", ".csv"]

        completed_dir = dir_path / "completed"

        files = [
            f
            for f in dir_path.rglob("*")
            if f.is_file()
            and f.suffix.lower() in supported_extensions
            and completed_dir not in f.parents
        ]

        total = len(files)
        logger.info("Detected %d file(s) to ingest in: %s", total, dir_path)

        for i, file_path in enumerate(files, 1):
            logger.info("[%d/%d] Starting: %s", i, total, file_path.name)
            try:
                chunks = await self.ingest_file(file_path, completed_dir=completed_dir)
                all_chunks.extend(chunks)
                logger.info("[%d/%d] Done (%d chunks): %s", i, total, len(chunks), file_path.name)
            except Exception as e:
                logger.error("[%d/%d] Failed: %s — %s", i, total, file_path.name, e)

        return all_chunks
