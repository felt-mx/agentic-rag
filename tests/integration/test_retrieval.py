import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from retrieval.pipeline import RetrievalPipeline

retrieval_pipeline = RetrievalPipeline()
results = retrieval_pipeline.retrieve(
    "What is machine learning?", top_k=5, rerank_method="rrf"
)

print(f"Retrieved {len(results)} results for the query: 'What is machine learning?'")
