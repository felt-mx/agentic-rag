from .handlers import sio


def register_socket_handlers():
    # Handlers are automatically registered via decorators
    # This function can be used for any additional setup if needed
    pass


def get_socket_events():
    return {
        "client_events": {
            "chat_stream": "Send a chat message for streaming response",
        },
        "server_events": {
            "status": "Connection status updates",
            "stream_start": "Indicates streaming has started",
            "processing": "Shows processing steps and progress",
            "stream_content_start": "Indicates content streaming is about to start",
            "stream_content": "Streams response content chunks",
            "stream_complete": "Complete response data when streaming is done",
            "error": "Error messages",
        },
    }
