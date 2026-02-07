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
        "who is beyonce?",
        top_k=10,
        retrieval_k=20,
        rerank_method="weighted",
    )

    print(
        f"Retrieved {len(results)} results for the query: 'what was a popular club in ibiza that started playing dance and house music?'"
    )

    print("Top results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result}")


if __name__ == "__main__":
    asyncio.run(main())
