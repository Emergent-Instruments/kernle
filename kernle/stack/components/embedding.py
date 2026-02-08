"""EmbeddingComponent - required stack component for semantic search.

Wraps the hash-based embedder from storage/embeddings.py and conforms
to StackComponentProtocol. When an InferenceService is available with
embed() support, the component can optionally delegate to that instead
of the local n-gram embedder.

This is the first stack component and serves as the reference
implementation for the StackComponentProtocol lifecycle.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from kernle.protocols import InferenceService, SearchResult
from kernle.storage.embeddings import HashEmbedder

logger = logging.getLogger(__name__)


class EmbeddingComponent:
    """Required stack component providing embedding for semantic search.

    Works without inference (local n-gram hash embedder). When inference
    is available, can optionally use model-based embeddings for higher
    quality search.

    Conforms to StackComponentProtocol.
    """

    def __init__(self) -> None:
        self._stack_id: Optional[str] = None
        self._inference: Optional[InferenceService] = None
        self._embedder = HashEmbedder()

    # ---- StackComponentProtocol properties ----

    @property
    def name(self) -> str:
        return "embedding-ngram"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def required(self) -> bool:
        return True

    @property
    def needs_inference(self) -> bool:
        return False

    # ---- Lifecycle ----

    def attach(
        self,
        stack_id: str,
        inference: Optional[InferenceService] = None,
    ) -> None:
        """Called when the component is added to a stack."""
        self._stack_id = stack_id
        self._inference = inference

    def detach(self) -> None:
        """Called when the component is removed from a stack."""
        self._stack_id = None
        self._inference = None

    def set_inference(self, inference: Optional[InferenceService]) -> None:
        """Update the inference service when the model changes."""
        self._inference = inference

    def set_storage(self, storage: Any) -> None:
        """Not needed by EmbeddingComponent."""
        pass

    # ---- Lifecycle Hooks ----

    def on_save(self, memory_type: str, memory_id: str, memory: Any) -> Any:
        """Generate embedding for saved memory.

        The actual embedding storage is handled by the SQLiteStorage
        backend via _save_embedding(). This hook is a notification
        point for when the component system takes over embedding
        management from the backend.

        Returns None (no modification to the memory).
        """
        return None

    def on_search(
        self,
        query: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """Post-process search results.

        Currently a pass-through. When the component fully owns
        embedding, this would re-rank results by semantic similarity.
        """
        return results

    def on_load(self, context: dict[str, Any]) -> None:
        """No-op for embeddings during load assembly."""
        return None

    def on_maintenance(self) -> dict[str, Any]:
        """Re-index stale embeddings.

        Returns stats about what was done. Currently reports
        component status; full re-indexing deferred to when the
        component owns the embedding lifecycle.
        """
        return {
            "provider": self.embedding_provider_id,
            "dimension": self.embedding_dimension,
            "has_inference": self._inference is not None,
        }

    # ---- Embedding Interface ----

    def embed(self, text: str) -> list[float]:
        """Embed text using the best available provider.

        If inference is available and provides embeddings, uses that.
        Otherwise falls back to the local hash embedder.
        """
        if self._inference is not None:
            try:
                return self._inference.embed(text)
            except Exception:
                logger.debug("Inference embed failed, falling back to local")
        return self._embedder.embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embed texts."""
        if self._inference is not None:
            try:
                return self._inference.embed_batch(texts)
            except Exception:
                logger.debug("Inference embed_batch failed, falling back to local")
        return self._embedder.embed_batch(texts)

    @property
    def embedding_dimension(self) -> int:
        """Dimension of vectors produced by embed()."""
        if self._inference is not None:
            return self._inference.embedding_dimension
        return self._embedder.dimension

    @property
    def embedding_provider_id(self) -> str:
        """Stable ID for the current embedding source."""
        if self._inference is not None:
            return self._inference.embedding_provider_id
        return "ngram-v1"
