from fastapi import FastAPI
from api.v1.router import api_v1_router

agentic_rag_app = FastAPI(title="Agentic RAG API", version="1.0.0")
agentic_rag_app.include_router(api_v1_router)
