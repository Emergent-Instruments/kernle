"""Embeddings routes for semantic memory operations."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..auth import CurrentAgent
from ..embeddings import (
    EMBEDDING_DIMENSIONS,
    create_embedding,
    create_embeddings_batch,
)
from ..logging_config import get_logger

logger = get_logger("kernle.routes.embeddings")
router = APIRouter(prefix="/embeddings", tags=["embeddings"])


class EmbeddingRequest(BaseModel):
    """Request to create an embedding for text."""

    text: str = Field(..., min_length=1, max_length=100000)


class EmbeddingResponse(BaseModel):
    """Response with embedding vector."""

    embedding: list[float]
    dimensions: int
    model: str = "text-embedding-3-small"


class BatchEmbeddingRequest(BaseModel):
    """Request to create embeddings for multiple texts."""

    texts: list[str] = Field(..., min_length=1, max_length=100)


class BatchEmbeddingResponse(BaseModel):
    """Response with multiple embedding vectors."""

    embeddings: list[list[float] | None]
    dimensions: int
    model: str = "text-embedding-3-small"
    successful: int
    failed: int


@router.post("", response_model=EmbeddingResponse)
async def create_text_embedding(
    request: EmbeddingRequest,
    auth: CurrentAgent,
):
    """
    Create an embedding vector for the given text.

    Uses OpenAI's text-embedding-3-small model (1536 dimensions).
    Requires authentication.
    """
    log_prefix = f"{auth.user_id}/{auth.agent_id}" if auth.user_id else auth.agent_id
    logger.info(f"EMBED | {log_prefix} | text_len={len(request.text)}")

    embedding = await create_embedding(request.text)

    if embedding is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding service unavailable - check OPENAI_API_KEY configuration",
        )

    return EmbeddingResponse(
        embedding=embedding,
        dimensions=EMBEDDING_DIMENSIONS,
    )


@router.post("/batch", response_model=BatchEmbeddingResponse)
async def create_batch_embeddings(
    request: BatchEmbeddingRequest,
    auth: CurrentAgent,
):
    """
    Create embeddings for multiple texts in a single request.

    More efficient than individual requests for bulk operations.
    Returns partial results if some texts fail.
    """
    log_prefix = f"{auth.user_id}/{auth.agent_id}" if auth.user_id else auth.agent_id
    logger.info(f"EMBED BATCH | {log_prefix} | count={len(request.texts)}")

    embeddings = await create_embeddings_batch(request.texts)

    # Count successes/failures
    successful = sum(1 for e in embeddings if e is not None)
    failed = len(embeddings) - successful

    if successful == 0:
        raise HTTPException(
            status_code=503,
            detail="Embedding service unavailable - check OPENAI_API_KEY configuration",
        )

    return BatchEmbeddingResponse(
        embeddings=embeddings,
        dimensions=EMBEDDING_DIMENSIONS,
        successful=successful,
        failed=failed,
    )
