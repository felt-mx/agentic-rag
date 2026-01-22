from retrieval.models import ScoredChunk


def normalize_dense_scores(chunks: list[ScoredChunk]) -> list[ScoredChunk]:
    if not chunks:
        return chunks

    scores = [chunk.scores for chunk in chunks]
    min_score, max_scores = min(scores), max(scores)

    if min_score == max_scores:
        return chunks

    return [
        chunk.__class__(
            **{**chunk.__dict__, "score": (chunk.score - min_score) / (max_scores - min_score)})
        for chunk in chunks
    ]
