FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Install CPU-only PyTorch first (skip CUDA deps = ~3GB savings)
# Clean caches after each step to avoid running out of disk space
RUN uv venv && \
    uv pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    rm -rf /root/.cache/uv /root/.cache/pip && \
    uv sync --frozen --no-install-package torch --no-install-package triton && \
    rm -rf /root/.cache/uv /root/.cache/pip

# Copy the rest of the project
COPY . .

# Run the Actor
CMD ["uv", "run", "python", "main.py"]
