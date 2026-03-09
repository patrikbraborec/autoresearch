FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Install CPU-only PyTorch first (skip CUDA deps = ~3GB savings)
RUN uv venv && \
    uv pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    uv sync --frozen --no-install-package torch

# Copy the rest of the project
COPY . .

# Run the Actor
CMD ["uv", "run", "python", "main.py"]
