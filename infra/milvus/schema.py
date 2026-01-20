from pymilvus import CollectionSchema, FieldSchema, DataType, Collection, connections


def create_chunk_collection(name: str = "chunks", dim=768) -> Collection:
    fields = [
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR,
                    is_primary=True, max_length=64),
        FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="section_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="metadata", dtype=DataType.JSON),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]

    schema = CollectionSchema(fields, description="Chunk for RAG retrieval")
    collection = Collection(name=name, schema=schema)

    return collection
