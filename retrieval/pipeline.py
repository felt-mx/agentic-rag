import asyncio
import json
import numpy as np
from retrieval.query.intake import intake_query
from retrieval.retrieve.hybrid import HybridRetriever
from retrieval.rerank.cross_encoder import CrossEncoderReranker
from core.models.embedder import VLLMClient
from core.models.generator import VLLMClient as GeneratorClient


class RetrievalPipeline:
    def __init__(
        self,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        database: str = None,
    ):
        self.retriever = HybridRetriever(database=database)
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.reranker = CrossEncoderReranker()
        self.embedder = VLLMClient()
        self.generator = GeneratorClient()

    async def generate_dense_embedding(self, query_text: str) -> np.ndarray:
        embedding = await self.embedder.generate(query_text)
        return np.array(embedding)

    def generate_image_embedding(self, image_data: np.ndarray) -> np.ndarray:
        return np.random.rand(
            512
        )  # Placeholder for actual image embedding generation logic

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate(chunks: list) -> list:
        """Keep the highest-scoring ScoredChunk per chunk_id."""
        seen: dict = {}
        for chunk in chunks:
            if chunk.chunk_id not in seen or chunk.score > seen[chunk.chunk_id].score:
                seen[chunk.chunk_id] = chunk
        return list(seen.values())

    async def _retrieve_raw(
        self,
        query_text: str,
        retrieval_k: int = 20,
        rerank_method: str = "weighted",
        weights: list = None,
        use_image: bool = False,
        image_embedding=None,
    ) -> list:
        """
        Hybrid retrieval for a single query with no reranking or score filtering.
        Returns a list of ScoredChunk objects.
        """
        query = intake_query(query_text)
        dense_embedding = await self.generate_dense_embedding(query_text=query["text"])

        if weights is None and rerank_method == "weighted":
            if use_image and image_embedding is not None:
                total = self.dense_weight + self.sparse_weight
                image_weight = 0.3
                dense_w = (self.dense_weight / total) * (1 - image_weight)
                sparse_w = (self.sparse_weight / total) * (1 - image_weight)
                weights = [dense_w, sparse_w, image_weight]
            else:
                total = self.dense_weight + self.sparse_weight
                weights = [self.dense_weight / total,
                           self.sparse_weight / total]

        chunks = self.retriever.retrieve(
            query_text=query["text"],
            dense_embedding=dense_embedding,
            image_embedding=image_embedding,
            top_k=retrieval_k,
            rerank_method=rerank_method,
            weights=weights,
            use_image=use_image,
        )
        print(
            f"[_retrieve_raw] query={query['text']!r} → {len(chunks)} chunks from Milvus")
        return chunks

    async def _global_rerank_and_filter(
        self,
        original_query: str,
        chunks: list,
        top_k: int = 5,
        score_threshold: float = 0.25,
    ) -> list:
        """
        Rerank the pooled chunks against the original query and apply the score
        threshold. Returns the standard formatted result dicts.
        """
        if not chunks:
            print("[_global_rerank_and_filter] 0 chunks in — skipping rerank")
            return []

        print(
            f"[_global_rerank_and_filter] {len(chunks)} chunks before rerank")
        if self.reranker:
            chunks = await self.reranker.rerank(
                query=original_query, chunks=chunks, top_k=top_k
            )
        print(f"[_global_rerank_and_filter] {len(chunks)} chunks after rerank")
        if chunks:
            scores = [round(r.score, 4) for r in chunks]
            print(
                f"[_global_rerank_and_filter] scores after rerank (raw): {scores}")

        # Preserve the raw score for each chunk — used in the final output.
        raw_score_map = {c.chunk_id: c.score for c in chunks}

        # Min-max normalise across the batch so the threshold is relative,
        # not absolute.  Cross-encoders often output raw scores clustered near
        # 0; without normalisation a valid top result can be wrongly discarded.
        if len(chunks) > 1:
            raw_scores = np.array([c.score for c in chunks], dtype=float)
            s_min, s_max = raw_scores.min(), raw_scores.max()
            if s_max > s_min:
                from retrieval.models import ScoredChunk
                chunks = [
                    ScoredChunk(
                        chunk_id=c.chunk_id, document_id=c.document_id,
                        section_id=c.section_id, text=c.text,
                        metadata=c.metadata, source=c.source,
                        score=float((c.score - s_min) / (s_max - s_min)),
                    )
                    for c in chunks
                ]
                norm_scores = [round(c.score, 4) for c in chunks]
                print(
                    f"[_global_rerank_and_filter] scores after normalisation: {norm_scores}")

        filtered = [r for r in chunks if r.score >= score_threshold]
        print(
            f"[_global_rerank_and_filter] {len(filtered)} chunks after score_threshold={score_threshold}")
        return [
            {
                "answer": r.text,
                "score": raw_score_map[r.chunk_id],
                "metadata": {
                    "file_name": r.metadata.get("file_name"),
                    "section_title": r.metadata.get("section_title"),
                },
            }
            for r in filtered
        ]

    # ------------------------------------------------------------------
    # Strategy workers
    # ------------------------------------------------------------------

    async def _get_expanded_queries(self, query: str) -> list:
        """
        Ask the LLM to produce 2-3 enriched rewrites of *query*.
        Injects the cached corpus summary so acronyms are expanded with
        domain-correct terminology. Always returns a deduplicated list
        that includes the original query.
        """
        from retrieval.answer.prompt_builder import build_expansion_prompt
        from retrieval.query.corpus_summary import get_corpus_summary

        corpus_summary = get_corpus_summary()
        expansion_prompt = build_expansion_prompt(query, corpus_summary)
        expansion_response = await self.generator.generate(
            expansion_prompt, tools=None, tool_choice=None, temperature=0.3
        )
        raw_content = expansion_response.get("content", "").strip()

        try:
            expanded = json.loads(raw_content)
            if not isinstance(expanded, list):
                raise ValueError("Not a list")
        except Exception:
            expanded = []

        return list(dict.fromkeys([query] + expanded))

    async def retrieve_with_expansion(
        self,
        original_query: str,
        top_k: int = 5,
        retrieval_k: int = 20,
        score_threshold: float = 0.25,
    ) -> list:
        """
        Expansion strategy: generate 2-3 enriched rewrites of the query,
        retrieve in parallel, deduplicate, then globally rerank.
        """
        queries = await self._get_expanded_queries(original_query)

        raw_results = await asyncio.gather(
            *[self._retrieve_raw(q, retrieval_k) for q in queries]
        )

        pooled = self._deduplicate(
            [chunk for batch in raw_results for chunk in batch])
        print(
            f"[retrieve_with_expansion] {len(pooled)} unique chunks after dedup across {len(queries)} queries")
        return await self._global_rerank_and_filter(
            original_query, pooled, top_k, score_threshold
        )

    async def retrieve_hybrid(
        self,
        original_query: str,
        sub_queries: list,
        top_k: int = 5,
        retrieval_k: int = 20,
        score_threshold: float = 0.25,
    ) -> list:
        """
        Hybrid strategy: decompose then expand.
        For each dispatcher-produced sub-query, generate LLM expansions and
        retrieve raw results for all variants in parallel. Pool every chunk
        from every sub-query's expansions, deduplicate, then globally rerank
        against the original query.
        """
        # Expand all sub-queries concurrently
        expanded_per_sub = await asyncio.gather(
            *[self._get_expanded_queries(sq) for sq in sub_queries]
        )

        # Flatten into one unique set of queries to retrieve
        all_queries = list(
            dict.fromkeys(q for variants in expanded_per_sub for q in variants)
        )
        print(
            f"[retrieve_hybrid] {len(sub_queries)} sub-queries expanded to {len(all_queries)} total queries")

        raw_results = await asyncio.gather(
            *[self._retrieve_raw(q, retrieval_k) for q in all_queries]
        )

        pooled = self._deduplicate(
            [chunk for batch in raw_results for chunk in batch])
        print(
            f"[retrieve_hybrid] {len(pooled)} unique chunks after dedup across {len(all_queries)} queries")
        return await self._global_rerank_and_filter(
            original_query, pooled, top_k, score_threshold
        )

    async def retrieve_with_decomposition(
        self,
        original_query: str,
        queries: list,
        top_k: int = 5,
        retrieval_k: int = 20,
        weights_override: list = None,
        score_threshold: float = 0.25,
    ) -> list:
        """
        Decomposition strategy: run _retrieve_raw for each dispatcher-produced
        sub-query in parallel, deduplicate the pooled ScoredChunks, then
        globally rerank against the original query.
        """
        raw_results = await asyncio.gather(
            *[
                self._retrieve_raw(
                    q,
                    retrieval_k,
                    rerank_method="weighted",
                    weights=weights_override,
                )
                for q in queries
            ]
        )

        pooled = self._deduplicate(
            [chunk for batch in raw_results for chunk in batch])
        print(
            f"[retrieve_with_decomposition] {len(pooled)} unique chunks after dedup across {len(queries)} queries")
        return await self._global_rerank_and_filter(
            original_query, pooled, top_k, score_threshold
        )
