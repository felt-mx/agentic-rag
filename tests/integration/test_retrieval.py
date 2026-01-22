import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncio
from retrieval.pipeline import RetrievalPipeline


async def main():
    retrieval_pipeline = RetrievalPipeline()
    results = await retrieval_pipeline.retrieve(
        "What is the main goal of the semantic chunking step?",
        top_k=5,
        rerank_method="rrf",
    )

    print(
        f"Retrieved {len(results)} results for the query: 'What is the main goal of the semantic chunking step?'"
    )

    print("Top results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result}")


if __name__ == "__main__":
    asyncio.run(main())
