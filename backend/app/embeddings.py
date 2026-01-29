"""OpenAI embeddings service for semantic memory search."""

from functools import lru_cache

from openai import AsyncOpenAI

from .config import get_settings
from .database import get_text_fields
from .logging_config import get_logger

logger = get_logger("kernle.embeddings")

# Embedding dimensions for text-embedding-3-small
EMBEDDING_DIMENSIONS = 1536


@lru_cache
def get_openai_client() -> AsyncOpenAI | None:
    """Get cached async OpenAI client, or None if not configured."""
    settings = get_settings()
    if not settings.openai_api_key:
        logger.warning("OPENAI_API_KEY not configured - embeddings disabled")
        return None
    return AsyncOpenAI(api_key=settings.openai_api_key)


async def create_embedding(text: str) -> list[float] | None:
    """
    Create an embedding vector for the given text.

    Uses OpenAI's text-embedding-3-small model.
    Returns None if OpenAI is not configured or on error.
    """
    client = get_openai_client()
    if client is None:
        return None

    settings = get_settings()

    try:
        # Truncate text if too long (model limit is ~8191 tokens)
        # Rough estimate: 4 chars per token
        max_chars = 8000 * 4
        if len(text) > max_chars:
            text = text[:max_chars]
            logger.debug(f"Truncated text to {max_chars} chars for embedding")

        response = await client.embeddings.create(
            model=settings.openai_embedding_model,
            input=text,
        )

        embedding = response.data[0].embedding
        logger.debug(f"Created embedding with {len(embedding)} dimensions")
        return embedding

    except Exception as e:
        logger.error(f"Failed to create embedding: {e}")
        return None


async def create_embeddings_batch(texts: list[str]) -> list[list[float] | None]:
    """
    Create embeddings for multiple texts in a single API call.

    More efficient than calling create_embedding multiple times.
    Returns list of embeddings (or None for failed texts).
    """
    client = get_openai_client()
    if client is None:
        return [None] * len(texts)

    if not texts:
        return []

    settings = get_settings()

    try:
        # Truncate each text
        max_chars = 8000 * 4
        truncated_texts = [t[:max_chars] if len(t) > max_chars else t for t in texts]

        response = await client.embeddings.create(
            model=settings.openai_embedding_model,
            input=truncated_texts,
        )

        # Response data is ordered by index
        embeddings = [None] * len(texts)
        for item in response.data:
            embeddings[item.index] = item.embedding

        logger.debug(f"Created {len(texts)} embeddings in batch")
        return embeddings

    except Exception as e:
        logger.error(f"Failed to create batch embeddings: {e}")
        return [None] * len(texts)


def extract_text_for_embedding(table: str, data: dict) -> str | None:
    """
    Extract text content from a memory record for embedding.

    Different memory types store text in different fields.
    Returns None if no suitable text found.
    """
    # Use centralized config from database.py (single source of truth)
    fields = get_text_fields(table) or ["content", "text", "description"]

    texts = []
    for field in fields:
        if field in data and data[field]:
            value = data[field]
            if isinstance(value, str):
                texts.append(value)
            elif isinstance(value, list):
                # Handle list fields (e.g., steps in playbooks)
                texts.extend(str(item) for item in value if item)

    if not texts:
        return None

    return " ".join(texts)


__all__ = [
    "create_embedding",
    "create_embeddings_batch",
    "extract_text_for_embedding",
    "EMBEDDING_DIMENSIONS",
]
