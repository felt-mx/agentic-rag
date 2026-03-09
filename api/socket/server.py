import socketio
from fastapi import FastAPI

# Create socket.io server with CORS enabled
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",
    logger=True,
    engineio_logger=True,
    max_http_buffer_size=5 * 1024 * 1024,  # 5 MB
)


def create_socket_app(app: FastAPI):
    socket_app = socketio.ASGIApp(sio, app)
    return socket_app


def get_sio():
    return sio
