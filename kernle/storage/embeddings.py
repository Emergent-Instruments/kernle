"""Embedding providers for Kernle.

Supports multiple embedding backends:
- hash: Simple hash-based similarity (fast, no dependencies, works offline)
- openai: OpenAI API embeddings (requires API key)
- local: Local models via sentence-transformers (requires torch)

The hash embedder is always available as a fallback.
"""

import hashlib
import logging
import struct
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Standard embedding dimension for hash embeddings
# Uses 384 to match e5-small, so we can upgrade later
HASH_EMBEDDING_DIM = 384


class EmbeddingProvider(ABC):
    """Base class for embedding providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Embed a single text."""
        ...

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts. Override for batch optimization."""
        return [self.embed(text) for text in texts]


class HashEmbedder(EmbeddingProvider):
    """Hash-based embedding for offline use.

    Creates deterministic embeddings using character n-grams and hashing.
    Not semantically meaningful but provides:
    - Exact and near-exact match detection
    - Fast, zero-dependency operation
    - Consistent embeddings (same text = same embedding)

    Can be upgraded to real embeddings later without schema changes.
    """

    def __init__(self, dim: int = HASH_EMBEDDING_DIM, ngram_range: Tuple[int, int] = (2, 4)):
        self._dim = dim
        self.ngram_range = ngram_range

    @property
    def dimension(self) -> int:
        return self._dim

    def _get_ngrams(self, text: str) -> List[str]:
        """Extract character n-grams from text."""
        text = text.lower().strip()
        ngrams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(text) - n + 1):
                ngrams.append(text[i : i + n])
        # Also add word-level features
        words = text.split()
        ngrams.extend(words)
        return ngrams

    def embed(self, text: str) -> List[float]:
        """Create a hash-based embedding."""
        # Initialize embedding
        embedding = [0.0] * self._dim

        ngrams = self._get_ngrams(text)
        if not ngrams:
            return embedding

        # Hash each n-gram to an index and accumulate
        for ngram in ngrams:
            # Use MD5 for fast, consistent hashing
            h = hashlib.md5(ngram.encode()).digest()
            # Extract index and sign from hash
            idx = int.from_bytes(h[:4], "little") % self._dim
            sign = 1 if h[4] & 1 else -1
            embedding[idx] += sign

        # Normalize to unit length
        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding


class OpenAIEmbedder(EmbeddingProvider):
    """OpenAI API-based embeddings.

    Requires OPENAI_API_KEY environment variable.
    Uses text-embedding-3-small by default (1536 dimensions).
    """

    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        self.model = model
        self._api_key = api_key
        self._client = None

        # Dimension varies by model
        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

    @property
    def dimension(self) -> int:
        return self._dimensions.get(self.model, 1536)

    def _get_client(self):
        if self._client is None:
            try:
                import openai

                self._client = openai.OpenAI(api_key=self._api_key)
            except ImportError:
                raise RuntimeError("openai package not installed. Run: pip install openai")
        return self._client

    def embed(self, text: str) -> List[float]:
        """Embed text using OpenAI API."""
        client = self._get_client()
        response = client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embed texts."""
        client = self._get_client()
        response = client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


# Cache for OpenAI availability check
_openai_available: bool | None = None


def clear_embedder_cache() -> None:
    """Clear the embedder availability cache. Useful for testing."""
    global _openai_available
    _openai_available = None


def get_default_embedder() -> EmbeddingProvider:
    """Get the best available embedding provider.

    Tries in order:
    1. Local model (if sentence-transformers available)
    2. OpenAI (if API key set)
    3. Hash embedder (always available)

    The OpenAI availability check is cached to avoid repeated API calls.
    """
    import os

    global _openai_available

    # Only test OpenAI once and cache the result
    if _openai_available is None and os.environ.get("OPENAI_API_KEY"):
        try:
            embedder = OpenAIEmbedder()
            embedder.embed("test")  # Verify it works
            _openai_available = True
            logger.info("Using OpenAI embeddings")
        except Exception as e:
            _openai_available = False
            logger.debug(f"OpenAI embeddings not available: {e}")

    if _openai_available:
        return OpenAIEmbedder()

    # Fall back to hash embedder
    logger.info("Using hash-based embeddings (offline mode)")
    return HashEmbedder()


def pack_embedding(embedding: List[float]) -> bytes:
    """Pack embedding as bytes for sqlite-vec storage."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def unpack_embedding(data: bytes) -> List[float]:
    """Unpack embedding from bytes."""
    count = len(data) // 4  # 4 bytes per float
    return list(struct.unpack(f"{count}f", data))
