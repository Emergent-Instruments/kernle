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
EXPOSE $PORT

# Run with uvicorn - uses $PORT from environment
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
