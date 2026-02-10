"""Kernle Core - Stratified memory for synthetic intelligences.

This package provides the main Kernle class, which is the primary interface
for memory operations. It uses the storage abstraction layer to support
both local SQLite storage and cloud Supabase storage.

All public names are re-exported here for backward compatibility:
    from kernle.core import Kernle
    from kernle.core import compute_priority_score
    from kernle.core import MAX_TOKEN_BUDGET, MIN_TOKEN_BUDGET
"""

from kernle.core.kernle_class import Kernle
from kernle.core.utils import (
    DEFAULT_MAX_ITEM_CHARS,
    DEFAULT_TOKEN_BUDGET,
    MAX_TOKEN_BUDGET,
    MEMORY_TYPE_PRIORITIES,
    MIN_TOKEN_BUDGET,
    TOKEN_ESTIMATION_SAFETY_MARGIN,
    _build_memory_echoes,
    _get_memory_hint_text,
    _get_record_attr,
    _get_record_created_at,
    _get_record_tags,
    _truncate_to_words,
    compute_priority_score,
    estimate_tokens,
    truncate_at_word_boundary,
)

__all__ = [
    "Kernle",
    # Constants
    "DEFAULT_TOKEN_BUDGET",
    "MAX_TOKEN_BUDGET",
    "MIN_TOKEN_BUDGET",
    "DEFAULT_MAX_ITEM_CHARS",
    "TOKEN_ESTIMATION_SAFETY_MARGIN",
    "MEMORY_TYPE_PRIORITIES",
    # Functions
    "estimate_tokens",
    "truncate_at_word_boundary",
    "compute_priority_score",
    # Internal helpers (exported for backward compat)
    "_build_memory_echoes",
    "_get_memory_hint_text",
    "_get_record_attr",
    "_get_record_created_at",
    "_get_record_tags",
    "_truncate_to_words",
]
