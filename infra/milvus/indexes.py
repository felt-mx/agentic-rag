from pymilvus import Collection


def create_dense_index(collection: Collection):
    # Dense embedding index
    collection.create_index(
        field_name="dense_embedding",
        index_params={
            "index_type": "AUTOINDEX",
            "metric_type": "IP",
        },
        index_name="dense_embedding_index",
    )

    # Sparse embedding index
    collection.create_index(
        field_name="sparse_embedding",
        index_params={
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "BM25",
            "params": {"inverted_index_algo": "DAAT_MAXSCORE"},
        },
        index_name="sparse_embedding_index",
    )

    # Image embedding index
    collection.create_index(
        field_name="image_embedding",
        index_params={
            "index_type": "AUTOINDEX",
            "metric_type": "IP",
        },
        index_name="image_embedding_index",
    )
