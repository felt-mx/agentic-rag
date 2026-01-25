"""
Standalone script to ingest documents into Milvus.
Can be run while API is running without affecting it.

Usage:
    python scripts/ingest.py document.pdf
    python scripts/ingest.py /path/to/documents_directory
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import sys
import asyncio
import argparse
from pathlib import Path
from infra.milvus.connection import connect_milvus
from ingestion.pipeline import IngestionPipeline


async def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG system")
    parser.add_argument("path", help="File or directory to ingest")

    args = parser.parse_args()

    connect_milvus()

    pipeline = IngestionPipeline()

    path = Path(args.path)

    if path.is_file():
        print(f"Ingesting single file: {path}\n")
        await pipeline.ingest_file(path)
    elif path.is_dir():
        print(f"Ingesting directory: {path}\n")
        await pipeline.ingest_directory(path)
    else:
        print(f"Error: {path} is not a valid file or directory")
        sys.exit(1)

    print("Ingestion completed.")


if __name__ == "__main__":
    asyncio.run(main())
