# Efficient multi-stage Dockerfile using uv
# Stage 1: Builder - Install dependencies
FROM python:3.11.9-slim AS builder

# Install system dependencies required for Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    default-libmysqlclient-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies using uv (much faster than pip)
RUN uv pip install --system --no-cache -r pyproject.toml

# Stage 2: Runtime - Minimal production image
FROM python:3.11.9-slim

# Install only runtime dependencies (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-libmysqlclient-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser . .

# Create media directory with proper permissions
RUN mkdir -p /app/media && chown -R appuser:appuser /app/media

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Run application
# Use Gunicorn with Uvicorn workers to serve the FastAPI + Socket.IO app
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "api.app:app", "--bind", "0.0.0.0:8000", "--workers", "4", "--timeout", "120"]
