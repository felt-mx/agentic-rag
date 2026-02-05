from pymilvus import Collection
from typing import List
from core.models.chunk import Chunk
from infra.milvus.connection import connect_milvus
from infra.milvus.schema import create_chunk_collection


def upsert_chunks(
    chunks: List[Chunk],
    collection_name: str = "chunks",
    batch_size: int = 100,
    database: str = None,
):
    if not chunks:
        print("No chunks to upsert")
        return

    connect_milvus(database=database)
    collection: Collection = create_chunk_collection(collection_name, database=database)

    total_chunks = len(chunks)
    total_upserted = 0

    # Process chunks in batches to avoid gRPC message size limits
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]
        data = []

        for chunk in batch:
            entity = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "section_id": chunk.section_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "dense_embedding": chunk.dense_embedding.tolist(),
            }

            # Add image embedding if present
            if chunk.image_embedding is not None:
                entity["image_embedding"] = chunk.image_embedding.tolist()
            else:
                # Provide a zero vector if no image
                entity["image_embedding"] = [0.0] * 1024

            data.append(entity)

        collection.upsert(data)
        total_upserted += len(batch)
        print(
            f"Upserted batch {i // batch_size + 1}/{(total_chunks + batch_size - 1) // batch_size} ({total_upserted}/{total_chunks} chunks)"
        )

    collection.flush()

    print(
        f"Successfully upserted {total_chunks} chunks into collection '{collection_name}'"
    )


def delete_chunks(
    chunk_ids: List[str], collection_name: str = "chunks", database: str = None
):
    connect_milvus(database=database)
    collection: Collection = create_chunk_collection(collection_name, database=database)

    expr = f"chunk_id in {chunk_ids}"
    collection.delete(expr)
    collection.flush()

    print(f"Deleted {len(chunk_ids)} chunks from collection '{collection_name}'")


def get_collection_stats(collection_name: str = "chunks", database: str = None):
    connect_milvus(database=database)
    collection: Collection = create_chunk_collection(collection_name, database=database)

    stats = collection.num_entities
    print(f"Collection '{collection_name}' contains {stats} entities")

    return stats
