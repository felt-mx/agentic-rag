from fastapi import APIRouter
from api.v1.chat.chat import chat_router

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(chat_router)
