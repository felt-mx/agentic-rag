from pymilvus import (
    CollectionSchema,
    FieldSchema,
    DataType,
    Collection,
    Function,
    FunctionType,
)


def create_chunk_collection(name: str = "chunks", dim=1024) -> Collection:
    fields = [
        FieldSchema(
            name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, max_length=64
        ),
        FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="section_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(
            name="text", dtype=DataType.VARCHAR, max_length=65535, enable_analyzer=True
        ),
        FieldSchema(name="metadata", dtype=DataType.JSON),
        FieldSchema(name="dense_embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="sparse_embedding", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="image_embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]

    bm25_function = Function(
        name="text_bm25_emb",
        input_field_names=["text"],
        output_field_names=["sparse_embedding"],
        function_type=FunctionType.BM25,
    )

    schema = CollectionSchema(
        fields, description="Chunk for RAG retrieval", functions=[bm25_function]
    )
    collection = Collection(name=name, schema=schema)

    return collection
