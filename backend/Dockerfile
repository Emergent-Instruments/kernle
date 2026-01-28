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

# Default port (Railway overrides via PORT env var)
ENV PORT=8000

# Copy and use start script for proper variable expansion
COPY start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]
