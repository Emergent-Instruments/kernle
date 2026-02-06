"""Cycle detection for derived_from chains.

Prevents circular derivation references (e.g., A->B->A) that could
cause infinite loops during lineage traversal.
"""

from typing import TYPE_CHECKING, List, Optional, Set

if TYPE_CHECKING:
    from kernle.storage.base import Storage

MAX_DERIVATION_DEPTH = 10


def check_derived_from_cycle(
    storage: "Storage",
    memory_type: str,
    memory_id: str,
    derived_from: Optional[List[str]],
    _max_depth: int = MAX_DERIVATION_DEPTH,
) -> None:
    """Check if adding derived_from references would create a cycle.

    Walks each ref's own derived_from chain recursively to see if we
    reach back to the target (memory_type:memory_id). Also detects
    direct self-references.

    Args:
        storage: Storage backend (must implement get_memory)
        memory_type: Type of the memory being saved/updated
        memory_id: ID of the memory being saved/updated
        derived_from: Proposed derived_from list (format: "type:id")
        _max_depth: Maximum traversal depth (default 10)

    Raises:
        ValueError: If a circular reference is detected or depth limit exceeded
    """
    if not derived_from:
        return

    target_ref = f"{memory_type}:{memory_id}"

    for ref in derived_from:
        if not ref or ":" not in ref:
            continue

        # Direct self-reference
        if ref == target_ref:
            raise ValueError("Circular derived_from reference detected")

        # Walk the chain from this ref
        ref_type, ref_id = ref.split(":", 1)

        # Skip non-memory refs like "context:..."
        if ref_type in ("context", "kernle"):
            continue

        visited: Set[str] = {target_ref, ref}
        _walk_chain(storage, ref_type, ref_id, target_ref, visited, 1, _max_depth)


def _walk_chain(
    storage: "Storage",
    current_type: str,
    current_id: str,
    target_ref: str,
    visited: Set[str],
    depth: int,
    max_depth: int,
) -> None:
    """Recursively walk derived_from chain looking for target_ref.

    Args:
        storage: Storage backend
        current_type: Type of current node
        current_id: ID of current node
        target_ref: The ref we're checking for cycles against
        visited: Set of already-visited refs to avoid re-walking
        depth: Current depth
        max_depth: Maximum allowed depth

    Raises:
        ValueError: If cycle detected or depth exceeded
    """
    if depth > max_depth:
        raise ValueError("Circular derived_from reference detected")

    memory = storage.get_memory(current_type, current_id)
    if memory is None:
        return

    parent_refs = getattr(memory, "derived_from", None)
    if not parent_refs:
        return

    for parent_ref in parent_refs:
        if not parent_ref or ":" not in parent_ref:
            continue

        if parent_ref == target_ref:
            raise ValueError("Circular derived_from reference detected")

        # Skip non-memory refs
        parent_type, parent_id = parent_ref.split(":", 1)
        if parent_type in ("context", "kernle"):
            continue

        if parent_ref in visited:
            continue

        visited.add(parent_ref)
        _walk_chain(storage, parent_type, parent_id, target_ref, visited, depth + 1, max_depth)
