from pymilvus import Collection


def create_dense_index(collection: Collection):
    index_params = {
        "metric_type": "IP",  # OR Cosine / L2
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200},
    }

    if not collection.has_index():
        collection.create_index(
            field_name="embedding",
            index_params=index_params,
        )
