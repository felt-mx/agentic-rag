from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.v1.router import api_v1_router
from api.socket.server import create_socket_app
from api.socket.mapper import register_socket_handlers
from retrieval.pipeline import RetrievalPipeline
from retrieval.query.corpus_summary import build_corpus_summary
from core.models.generator import VLLMClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build and cache the corpus summary once at startup."""
    try:
        pipeline = RetrievalPipeline()
        generator = VLLMClient()
        await build_corpus_summary(pipeline.retriever, generator)
        print("[startup] Corpus summary built successfully.")
    except Exception as e:
        print(f"[startup] Warning: corpus summary build failed: {e}")
    yield


# Create FastAPI app
agentic_rag_app = FastAPI(
    title="Agentic RAG API", version="1.0.0", lifespan=lifespan
)
agentic_rag_app.include_router(api_v1_router)

# Register socket.io handlers
register_socket_handlers()

# Create the combined app with socket.io
app = create_socket_app(agentic_rag_app)
