from fastapi import FastAPI
from api.v1.router import api_v1_router
from api.socket.server import create_socket_app
from api.socket.mapper import register_socket_handlers

# Create FastAPI app
agentic_rag_app = FastAPI(title="Agentic RAG API", version="1.0.0")
agentic_rag_app.include_router(api_v1_router)

# Register socket.io handlers
register_socket_handlers()

# Create the combined app with socket.io
app = create_socket_app(agentic_rag_app)
