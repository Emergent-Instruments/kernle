"""
Kernle Comms - Agent-to-Agent Communication.

This package provides infrastructure for SI↔SI communication:
- Agent registry and discovery
- Messaging between agents
- Memory sharing with consent
- Collaboration protocols
"""

from kernle.comms.registry import (
    AgentProfile,
    AgentRegistry,
    RegistryError,
)

__all__ = [
    "AgentProfile",
    "AgentRegistry",
    "RegistryError",
]
