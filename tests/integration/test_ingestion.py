import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ingestion.fake_ingestor import fake_ingest_text, fake_generate_chunks
from infra.milvus.client import upsert_chunks

doc = fake_ingest_text("This is a sample text for testing the ingestion pipeline.")
chunks = fake_generate_chunks(doc)
upsert_chunks(chunks)

print(f"Upserted {len(chunks)} chunks into Milvus.")
