"""
Standalone script to ingest documents into Milvus.
Can be run while API is running without affecting it.

Usage:
    python scripts/ingest.py document.pdf
    python scripts/ingest.py /path/to/documents_directory
    python scripts/ingest.py "\\\\server\\share\\documents"   # Windows UNC network path
"""

import sys
import asyncio
import logging
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from infra.milvus.connection import connect_milvus
from ingestion.pipeline import IngestionPipeline
from config.config import MILVUS_DATABASE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG system")
    parser.add_argument("path", help="File or directory to ingest")

    args = parser.parse_args()

    database = MILVUS_DATABASE

    connect_milvus(database=database)

    pipeline = IngestionPipeline(database=database)

    path = Path(args.path)

    if path.is_file():
        logger.info("Ingesting single file: %s", path)
        await pipeline.ingest_file(path)
    elif path.is_dir():
        logger.info("Ingesting directory: %s", path)
        await pipeline.ingest_directory(path)
    else:
        logger.error("Not a valid file or directory: %s", path)
        sys.exit(1)

    logger.info("Ingestion completed.")


if __name__ == "__main__":
    asyncio.run(main())
