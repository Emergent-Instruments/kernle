"""Utility functions for the golden corpus test."""

import hashlib


def compute_hash(content):
    """Compute a SHA-256 hash of content."""
    return hashlib.sha256(content.encode()).hexdigest()


def normalize_text(text):
    """Normalize whitespace in text for comparison."""
    return " ".join(text.split())


def format_memory_ref(mem_type, mem_id):
    """Format a memory reference string."""
    return f"{mem_type}:{mem_id}"
