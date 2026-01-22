from retrieval.models import ScoredChunk


def normalize_dense_scores(chunks: list[ScoredChunk]) -> list[ScoredChunk]:
    if not chunks:
        return chunks

    scores = [chunk.score for chunk in chunks]
    min_score, max_score = min(scores), max(scores)

    if min_score == max_score:
        return chunks

    return [
        chunk.__class__(
            **{
                **chunk.__dict__,
                "score": (chunk.score - min_score) / (max_score - min_score),
            }
        )
        for chunk in chunks
    ]
