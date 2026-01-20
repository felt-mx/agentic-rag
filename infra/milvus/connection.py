from pymilvus import connections
from configs.config import MILVUS_HOST, MILVUS_PORT


def connect_milvus():
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

    return True
