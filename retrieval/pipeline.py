import numpy as np
from retrieval.query.intake import intake_query
from retrieval.retrieve.dense import DenseRetriever
from retrieval.scoring.normalize import normalize_dense_scores


class RetrievalPipeline:
    def __init__(self):
        self.retriever = DenseRetriever()

    def retrieve(self, raw_query: str):
        query = intake_query(raw_query)
        # Placeholder for actual embedding generation
        query_embedding = np.random.rand(768)
        results = self.retriever.retrieve(query_embedding)
        results = normalize_dense_scores(results)

        return results
