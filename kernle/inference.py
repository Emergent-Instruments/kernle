"""InferenceService — narrow model access for stack components.

Wraps a ModelProtocol into the InferenceService interface that stack
components use. Components never see ModelProtocol directly — they
get this thin wrapper which provides infer(), embed(), and embed_batch().

When no model-level embedding is available, falls back to the local
hash-based embedder from storage/embeddings.py.
"""

from __future__ import annotations

from typing import Optional

from kernle.protocols import InferenceService, ModelMessage, ModelProtocol
from kernle.storage.embeddings import HashEmbedder


class _InferenceServiceImpl:
    """Concrete InferenceService wrapping a ModelProtocol.

    Delegates infer() to model.generate() and provides local
    hash-based embeddings as the default embed() implementation.
    """

    def __init__(self, model: ModelProtocol) -> None:
        self._model = model
        self._embedder = HashEmbedder()

    def infer(self, prompt: str, *, system: Optional[str] = None) -> str:
        """Generate text by routing to the bound model."""
        messages = [ModelMessage(role="user", content=prompt)]
        response = self._model.generate(messages, system=system)
        return response.content

    def embed(self, text: str) -> list[float]:
        """Embed text using the local hash embedder."""
        return self._embedder.embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embed using the local hash embedder."""
        return self._embedder.embed_batch(texts)

    @property
    def embedding_dimension(self) -> int:
        """Dimension of vectors produced by embed()."""
        return self._embedder.dimension

    @property
    def embedding_provider_id(self) -> str:
        """Stable ID for the current embedding source."""
        return "ngram-v1"


def create_inference_service(model: ModelProtocol) -> InferenceService:
    """Create an InferenceService wrapping a model.

    This is the factory function Entity uses when set_model() is called.
    """
    return _InferenceServiceImpl(model)  # type: ignore[return-value]
