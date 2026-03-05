from __future__ import annotations

_corpus_summary: str = ""


async def build_corpus_summary(retriever, generator) -> str:
    """
    Queries Milvus for all distinct (file_name, section_title) metadata pairs,
    then asks the generator LLM to produce a short natural-language description
    of the corpus.  The result is cached in-process for the lifetime of the app.
    """
    global _corpus_summary
    try:
        results = retriever.collection.query(
            expr='chunk_id != ""',
            output_fields=["metadata"],
        )

        distinct = {
            (
                r.get("metadata", {}).get("file_name", ""),
                r.get("metadata", {}).get("section_title", ""),
            )
            for r in results
        }

        metadata_lines = "\n".join(
            f"- File: {fn} | Section: {st}"
            for fn, st in sorted(distinct)
            if fn or st
        )

        if not metadata_lines:
            _corpus_summary = ""
            return _corpus_summary

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Given a list of document file names and section titles from a knowledge base, "
                    "write a concise 3-5 sentence description of the topics, domains, and types of "
                    "content covered. Be factual and specific — do not invent information."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Here are all the distinct file names and section titles in the knowledge base:\n\n"
                    f"{metadata_lines}\n\n"
                    "Describe the knowledge base in 3-5 sentences."
                ),
            },
        ]

        response = await generator.generate(
            prompt, tools=None, tool_choice=None, temperature=0.1
        )
        _corpus_summary = response.get("content", "").strip()

    except Exception as e:
        print(f"[corpus_summary] Warning: failed to build corpus summary: {e}")
        _corpus_summary = ""

    return _corpus_summary


def get_corpus_summary() -> str:
    """Return the cached corpus summary built at startup."""
    return _corpus_summary
