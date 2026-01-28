FROM python:3.12-slim

# Cache bust: 2026-01-27-v2
WORKDIR /app

# Install uv for fast dependency management
RUN pip install uv

# Copy dependency and build files
COPY pyproject.toml README.md ./

# Install dependencies
RUN uv pip install --system -e .

# Copy application code
COPY app/ app/

# Default port (Railway sets PORT env var)
ENV PORT=8000

# Copy entry point
COPY run.py .

# Use Python entry point that reads PORT from environment
CMD ["python", "run.py"]
