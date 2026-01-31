"""Escrow subsystem for Kernle Commerce.

Escrow provides secure payment handling for jobs using smart contracts on Base.
Each job gets a dedicated escrow contract that:
- Holds USDC deposited by the client
- Releases payment to worker on approval
- Supports dispute resolution via arbitrator

Modules:
- service.py: High-level escrow operations (deploy, fund, release, etc.)
- abi.py: Solidity contract ABIs for encoding/decoding
- events.py: Event monitoring and parsing
"""

from kernle.commerce.escrow.abi import (
    KERNLE_ESCROW_ABI,
    KERNLE_ESCROW_FACTORY_ABI,
    ERC20_ABI,
    EscrowStatus,
)
from kernle.commerce.escrow.service import (
    EscrowService,
    EscrowServiceError,
    EscrowNotFoundError,
    InsufficientAllowanceError,
    InvalidEscrowStateError,
    TransactionFailedError,
    EscrowInfo,
    TransactionResult,
)
from kernle.commerce.escrow.events import (
    EscrowEventType,
    EscrowEvent,
    FundedEvent,
    WorkerAssignedEvent,
    DeliveredEvent,
    ReleasedEvent,
    RefundedEvent,
    DisputedEvent,
    DisputeResolvedEvent,
    EscrowCreatedEvent,
    EscrowEventParser,
    EscrowEventMonitor,
    EscrowEventIndexer,
)


__all__ = [
    # ABIs
    "KERNLE_ESCROW_ABI",
    "KERNLE_ESCROW_FACTORY_ABI",
    "ERC20_ABI",
    "EscrowStatus",
    # Service
    "EscrowService",
    "EscrowServiceError",
    "EscrowNotFoundError",
    "InsufficientAllowanceError",
    "InvalidEscrowStateError",
    "TransactionFailedError",
    "EscrowInfo",
    "TransactionResult",
    # Events
    "EscrowEventType",
    "EscrowEvent",
    "FundedEvent",
    "WorkerAssignedEvent",
    "DeliveredEvent",
    "ReleasedEvent",
    "RefundedEvent",
    "DisputedEvent",
    "DisputeResolvedEvent",
    "EscrowCreatedEvent",
    "EscrowEventParser",
    "EscrowEventMonitor",
    "EscrowEventIndexer",
]
