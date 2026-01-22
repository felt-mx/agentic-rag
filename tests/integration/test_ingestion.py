import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ingestion.fake_ingestor import fake_ingest_text, fake_generate_chunks
from infra.milvus.client import upsert_chunks, get_collection_stats

# Create a document
doc = fake_ingest_text(
    "Machine learning is a field of artificial intelligence that focuses on the development of algorithms that can learn from and make predictions on data. It involves training models on large datasets to recognize patterns and make decisions without being explicitly programmed for specific tasks."
)

# Generate chunks
chunks = fake_generate_chunks(doc)

# Upsert into Milvus
upsert_chunks(chunks)

print(f"Upserted {len(chunks)} chunks into Milvus.")

# Verify
get_collection_stats()
