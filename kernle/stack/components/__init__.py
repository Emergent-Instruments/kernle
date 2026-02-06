"""kernle.stack.components - Stack sub-components.

Components extend what the stack can do. They hook into the stack's
lifecycle: memory saves, loads, searches, and periodic maintenance.

Discovery via entry point: kernle.stack_components
"""

from kernle.stack.components.embedding import EmbeddingComponent

__all__ = ["EmbeddingComponent"]
