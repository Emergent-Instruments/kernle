"""kernle.stack.components - Stack sub-components.

Components extend what the stack can do. They hook into the stack's
lifecycle: memory saves, loads, searches, and periodic maintenance.

Discovery via entry point: kernle.stack_components
"""

from kernle.stack.components.anxiety import AnxietyComponent
from kernle.stack.components.consolidation import ConsolidationComponent
from kernle.stack.components.embedding import EmbeddingComponent
from kernle.stack.components.emotions import EmotionalTaggingComponent
from kernle.stack.components.forgetting import ForgettingComponent
from kernle.stack.components.knowledge import KnowledgeComponent
from kernle.stack.components.metamemory import MetaMemoryComponent
from kernle.stack.components.suggestions import SuggestionComponent

__all__ = [
    "AnxietyComponent",
    "ConsolidationComponent",
    "EmbeddingComponent",
    "EmotionalTaggingComponent",
    "ForgettingComponent",
    "KnowledgeComponent",
    "MetaMemoryComponent",
    "SuggestionComponent",
]
