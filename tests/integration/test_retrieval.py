import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from retrieval.pipeline import RetrievalPipeline

retrieval_pipeline = RetrievalPipeline()
raw_query = "What is the capital of France?"
results = retrieval_pipeline.retrieve(raw_query)

print(f"Retrieved {len(results)} results for the query: '{raw_query}'")
