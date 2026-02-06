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
    "get_default_components",
    "get_minimal_components",
    "load_components_from_discovery",
]

# The default component set — all 8 built-in components.
_DEFAULT_COMPONENT_CLASSES = [
    EmbeddingComponent,
    ForgettingComponent,
    ConsolidationComponent,
    EmotionalTaggingComponent,
    AnxietyComponent,
    SuggestionComponent,
    MetaMemoryComponent,
    KnowledgeComponent,
]


def get_default_components():
    """Return new instances of all 8 built-in stack components.

    This is the default set loaded when a stack is created without
    an explicit component list.
    """
    return [cls() for cls in _DEFAULT_COMPONENT_CLASSES]


def get_minimal_components():
    """Return a minimal component set (embedding only).

    Use this for a functional but bare stack.
    """
    return [EmbeddingComponent()]


def load_components_from_discovery():
    """Discover and instantiate all installed stack components via entry points.

    Uses the ``kernle.stack_components`` entry point group. Any package
    that registers a component class there will be discovered and
    instantiated.

    Returns:
        List of instantiated stack component instances.
    """
    from kernle.discovery import discover_stack_components, load_component

    discovered = discover_stack_components()
    instances = []
    for dc in discovered:
        try:
            cls = load_component(dc)
            instances.append(cls())
        except Exception:
            import logging

            logging.getLogger(__name__).warning(
                "Failed to load component '%s' from %s", dc.name, dc.qualname
            )
    return instances
