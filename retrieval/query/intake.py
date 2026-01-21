import uuid


def intake_query(raw_query: str) -> dict:
    return {
        "query_id": str(uuid.uuid4()),
        "text": raw_query.strip(),
        "language": "auto",
        "metadata": {},
    }
