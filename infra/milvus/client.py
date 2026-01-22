from pymilvus import Collection
from infra.milvus.connection import connect_milvus
from infra.milvus.schema import create_chunk_collection


def upsert_chunks(chunks: list[any], collection_name="chunks"):
    connect_milvus()
    col: Collection = create_chunk_collection(name=collection_name)
    insert_data = []
    for chunk in chunks:
        insert_data.append(
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "section_id": chunk.section_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "embedding": chunk.embedding.tolist(),
            }
        )
        col.insert(insert_data)
