FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency management
RUN pip install uv

# Copy dependency and build files
COPY pyproject.toml README.md ./

# Install dependencies
RUN uv pip install --system -e .

# Copy application code
COPY app/ app/

# Default port (Railway overrides via $PORT)
ENV PORT=8000

# Run with uvicorn - shell form for variable expansion
CMD ["/bin/sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
