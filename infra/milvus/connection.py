from pymilvus import connections, db
from config.config import MILVUS_HOST, MILVUS_PORT, MILVUS_DATABASE


def connect_milvus(database: str = None):
    db_name = database or MILVUS_DATABASE

    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

    # Create database if it doesn't exist
    existing_databases = db.list_database()
    if db_name not in existing_databases:
        db.create_database(db_name)

    # Switch to the specified database
    db.using_database(db_name)

    return True
