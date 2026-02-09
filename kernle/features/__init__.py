"""Feature modules for Kernle.

Each feature is implemented as a mixin class that provides specific
functionality to the main Kernle class.
"""

from kernle.features.anxiety import AnxietyMixin
from kernle.features.belief_revision import BeliefRevisionMixin
from kernle.features.consolidation import ConsolidationMixin
from kernle.features.emotions import EmotionsMixin
from kernle.features.forgetting import ForgettingMixin
from kernle.features.knowledge import KnowledgeMixin
from kernle.features.metamemory import (
    DEFAULT_DECAY_CONFIGS,
    DEFAULT_DECAY_FLOOR,
    DEFAULT_DECAY_PERIOD_DAYS,
    DEFAULT_DECAY_RATE,
    DecayConfig,
    MetaMemoryMixin,
)
from kernle.features.playbooks import PlaybookMixin
from kernle.features.suggestions import SuggestionsMixin
from kernle.features.trust import TrustMixin

__all__ = [
    "AnxietyMixin",
    "BeliefRevisionMixin",
    "ConsolidationMixin",
    "DecayConfig",
    "DEFAULT_DECAY_CONFIGS",
    "DEFAULT_DECAY_FLOOR",
    "DEFAULT_DECAY_PERIOD_DAYS",
    "DEFAULT_DECAY_RATE",
    "EmotionsMixin",
    "ForgettingMixin",
    "KnowledgeMixin",
    "MetaMemoryMixin",
    "PlaybookMixin",
    "SuggestionsMixin",
    "TrustMixin",
]
