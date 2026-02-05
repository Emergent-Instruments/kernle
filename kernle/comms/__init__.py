"""
Kernle Comms - Agent-to-Agent Communication.

This package provides infrastructure for SI↔SI communication:
- Agent registry and discovery
- Messaging between agents
- Memory sharing with consent
- Collaboration protocols
"""

from kernle.comms.crypto import (
    CryptoError,
    KeyAlreadyExistsError,
    KeyManager,
    KeyNotFoundError,
    KeyPair,
    SignatureError,
    generate_key_pair,
    sign_message,
    verify_signature,
)
from kernle.comms.messaging import (
    DeliveryError,
    Message,
    MessageNotFoundError,
    MessagePriority,
    MessageStatus,
    MessageStore,
    Messenger,
    MessagingError,
)
from kernle.comms.registry import (
    AgentProfile,
    AgentRegistry,
    RegistryError,
)

__all__ = [
    # Registry
    "AgentProfile",
    "AgentRegistry",
    "RegistryError",
    # Crypto
    "CryptoError",
    "KeyAlreadyExistsError",
    "KeyManager",
    "KeyNotFoundError",
    "KeyPair",
    "SignatureError",
    "generate_key_pair",
    "sign_message",
    "verify_signature",
    # Messaging
    "DeliveryError",
    "Message",
    "MessageNotFoundError",
    "MessagePriority",
    "MessageStatus",
    "MessageStore",
    "Messenger",
    "MessagingError",
]
