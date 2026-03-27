import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from infra.milvus.connection import connect_milvus
from pymilvus import Collection

connect_milvus()
collection = Collection("chunks")

print("Releasing collection...")
collection.release()
print("Collection released.")