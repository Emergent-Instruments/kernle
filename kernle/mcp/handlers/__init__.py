"""Handler registry for MCP tools.

Merges HANDLERS and VALIDATORS from all sub-modules into unified dicts.
"""

from typing import Callable, Dict

from kernle.mcp.handlers.identity import HANDLERS as _IDENTITY_H
from kernle.mcp.handlers.identity import VALIDATORS as _IDENTITY_V
from kernle.mcp.handlers.memory import HANDLERS as _MEMORY_H
from kernle.mcp.handlers.memory import VALIDATORS as _MEMORY_V
from kernle.mcp.handlers.processing import HANDLERS as _PROCESSING_H
from kernle.mcp.handlers.processing import VALIDATORS as _PROCESSING_V
from kernle.mcp.handlers.seed import HANDLERS as _SEED_H
from kernle.mcp.handlers.seed import VALIDATORS as _SEED_V
from kernle.mcp.handlers.sync import HANDLERS as _SYNC_H
from kernle.mcp.handlers.sync import VALIDATORS as _SYNC_V
from kernle.mcp.handlers.temporal import HANDLERS as _TEMPORAL_H
from kernle.mcp.handlers.temporal import VALIDATORS as _TEMPORAL_V

HANDLERS: Dict[str, Callable] = {
    **_MEMORY_H,
    **_IDENTITY_H,
    **_TEMPORAL_H,
    **_SYNC_H,
    **_PROCESSING_H,
    **_SEED_H,
}

VALIDATORS: Dict[str, Callable] = {
    **_MEMORY_V,
    **_IDENTITY_V,
    **_TEMPORAL_V,
    **_SYNC_V,
    **_PROCESSING_V,
    **_SEED_V,
}
