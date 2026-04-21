# Stage 1: Builder
FROM python:3.11.9-slim AS builder

# Install system dependencies required for building Python packages
# (gcc and pkg-config are often needed for compiling C-extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    default-libmysqlclient-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Enable bytecode compilation and copying for faster builds
ENV UV_COMPILE_BYTECODE=1 
ENV UV_LINK_MODE=copy

# Copy dependency files
# We copy both pyproject.toml and uv.lock to ensure a deterministic build
COPY pyproject.toml uv.lock ./

# Install dependencies using uv sync
# --frozen: Forces usage of uv.lock (fails if lockfile is missing/outdated)
# --no-install-project: Installs only dependencies, not the app itself (better caching)
# --no-dev: Excludes development dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --no-cache

RUN python -m compileall -f . 2>&1 && \
    find /app -type f -name "*.py" -not -path "/app/.venv/*" -not -name "__init__.py" -delete

# Stage 2: Runtime
FROM python:3.11.9-slim

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-libmysqlclient-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# --- CRITICAL CHANGE ---
# Instead of copying /usr/local/lib/site-packages, we copy the virtual environment
# created by uv sync from the builder stage.
COPY --from=builder /app/.venv /app/.venv

# Add the virtual environment to the PATH
# This ensures python, gunicorn, and uvicorn are run from the .venv automatically
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code
COPY --chown=appuser:appuser . .

# Create media directory
RUN mkdir -p /app/media && chown -R appuser:appuser /app/media

# Switch to non-root user
USER appuser

EXPOSE 8000

# Run application
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "api.app:app", "--bind", "0.0.0.0:8000", "--workers", "4", "--timeout", "120"]