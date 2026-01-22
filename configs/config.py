from dotenv import load_dotenv
import os

load_dotenv()

MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))

VLLM_API_URL = os.getenv("VLLM_API_URL", "127.0.0.1")
VLLM_EMBED_API_PORT = int(os.getenv("VLLM_EMBED_API_PORT", "8000"))
VLLM_EMBED_MODEL_NAME = os.getenv(
    "VLLM_EMBED_MODEL_NAME", "intfloat/multilingual-e5-large-instruct"
)
