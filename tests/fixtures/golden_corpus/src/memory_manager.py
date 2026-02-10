"""Memory manager module for the golden corpus test."""


class MemoryManager:
    """Manages memory lifecycle operations."""

    def __init__(self, storage):
        self._storage = storage
        self._cache = {}

    def save(self, key, value):
        """Save a memory entry to storage."""
        self._storage[key] = value
        self._cache[key] = value

    def load(self, key):
        """Load a memory entry from storage or cache."""
        if key in self._cache:
            return self._cache[key]
        return self._storage.get(key)


def process_batch(items):
    """Process a batch of memory items for promotion."""
    results = []
    for item in items:
        if item.get("eligible"):
            results.append({"promoted": True, "id": item["id"]})
    return results


def validate_provenance(chain):
    """Validate a provenance chain for integrity."""
    seen = set()
    for ref in chain:
        if ref in seen:
            return False
        seen.add(ref)
    return True
