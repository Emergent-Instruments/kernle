"""
Smart contract ABIs for Kernle Commerce escrow.

Contains the Application Binary Interface (ABI) definitions for:
- KernleEscrow: Per-job escrow contract
- KernleEscrowFactory: Factory for deploying escrow contracts
- USDC (ERC20): USDC token interface

These ABIs are used for:
- Encoding function calls to contracts
- Decoding transaction data
- Parsing event logs

Note: These are placeholder ABIs that will be updated when the
actual Solidity contracts are deployed. The structure follows
the expected contract interface from the design document.
"""

from typing import Any, Dict, List

# Type alias for ABI entries
ABIEntry = Dict[str, Any]
ABI = List[ABIEntry]


# =============================================================================
# KernleEscrow ABI
# =============================================================================
# Per-job escrow contract that holds USDC until job completion.
#
# State Machine:
#   Funded → Accepted → Delivered → Released (to worker)
#      ↓         ↓          ↓
#   Refunded  Disputed   Disputed
#
# Key functions:
#   - fund(amount): Client deposits USDC
#   - assignWorker(worker): Client assigns accepted worker
#   - deliver(hash): Worker submits deliverable
#   - release(): Client releases payment to worker
#   - refund(): Client cancels and gets refund (if not accepted)
#   - dispute(): Either party raises dispute
#   - resolveDispute(recipient): Arbitrator resolves dispute

KERNLE_ESCROW_ABI: ABI = [
    # Constructor
    {
        "type": "constructor",
        "inputs": [
            {"name": "jobId", "type": "bytes32"},
            {"name": "client", "type": "address"},
            {"name": "usdcToken", "type": "address"},
            {"name": "arbitrator", "type": "address"},
            {"name": "amount", "type": "uint256"},
            {"name": "deadline", "type": "uint256"},
        ],
    },
    # === State Variables (view functions) ===
    {
        "type": "function",
        "name": "jobId",
        "inputs": [],
        "outputs": [{"name": "", "type": "bytes32"}],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "client",
        "inputs": [],
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "worker",
        "inputs": [],
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "arbitrator",
        "inputs": [],
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "amount",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "deadline",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "status",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint8"}],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "deliverableHash",
        "inputs": [],
        "outputs": [{"name": "", "type": "bytes32"}],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "deliveredAt",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "approvalTimeout",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
    },
    # === State Machine Functions ===
    # Fund the escrow (client deposits USDC)
    {
        "type": "function",
        "name": "fund",
        "inputs": [],
        "outputs": [],
        "stateMutability": "nonpayable",
    },
    # Assign worker after accepting application
    {
        "type": "function",
        "name": "assignWorker",
        "inputs": [{"name": "_worker", "type": "address"}],
        "outputs": [],
        "stateMutability": "nonpayable",
    },
    # Worker submits deliverable
    {
        "type": "function",
        "name": "deliver",
        "inputs": [{"name": "_deliverableHash", "type": "bytes32"}],
        "outputs": [],
        "stateMutability": "nonpayable",
    },
    # Client releases payment to worker
    {
        "type": "function",
        "name": "release",
        "inputs": [],
        "outputs": [],
        "stateMutability": "nonpayable",
    },
    # Auto-release after timeout (anyone can call)
    {
        "type": "function",
        "name": "autoRelease",
        "inputs": [],
        "outputs": [],
        "stateMutability": "nonpayable",
    },
    # Client cancels and gets refund (only before worker assigned)
    {
        "type": "function",
        "name": "refund",
        "inputs": [],
        "outputs": [],
        "stateMutability": "nonpayable",
    },
    # Raise dispute (client or worker)
    {
        "type": "function",
        "name": "dispute",
        "inputs": [],
        "outputs": [],
        "stateMutability": "nonpayable",
    },
    # Arbitrator resolves dispute
    {
        "type": "function",
        "name": "resolveDispute",
        "inputs": [{"name": "_recipient", "type": "address"}],
        "outputs": [],
        "stateMutability": "nonpayable",
    },
    # === Events ===
    {
        "type": "event",
        "name": "Funded",
        "inputs": [
            {"name": "client", "type": "address", "indexed": True},
            {"name": "amount", "type": "uint256", "indexed": False},
        ],
    },
    {
        "type": "event",
        "name": "WorkerAssigned",
        "inputs": [
            {"name": "worker", "type": "address", "indexed": True},
        ],
    },
    {
        "type": "event",
        "name": "Delivered",
        "inputs": [
            {"name": "worker", "type": "address", "indexed": True},
            {"name": "deliverableHash", "type": "bytes32", "indexed": False},
        ],
    },
    {
        "type": "event",
        "name": "Released",
        "inputs": [
            {"name": "worker", "type": "address", "indexed": True},
            {"name": "amount", "type": "uint256", "indexed": False},
        ],
    },
    {
        "type": "event",
        "name": "Refunded",
        "inputs": [
            {"name": "client", "type": "address", "indexed": True},
            {"name": "amount", "type": "uint256", "indexed": False},
        ],
    },
    {
        "type": "event",
        "name": "Disputed",
        "inputs": [
            {"name": "disputant", "type": "address", "indexed": True},
        ],
    },
    {
        "type": "event",
        "name": "DisputeResolved",
        "inputs": [
            {"name": "recipient", "type": "address", "indexed": True},
            {"name": "amount", "type": "uint256", "indexed": False},
        ],
    },
    # === Errors ===
    {
        "type": "error",
        "name": "Unauthorized",
        "inputs": [],
    },
    {
        "type": "error",
        "name": "InvalidState",
        "inputs": [],
    },
    {
        "type": "error",
        "name": "DeadlineNotPassed",
        "inputs": [],
    },
    {
        "type": "error",
        "name": "TimeoutNotExpired",
        "inputs": [],
    },
]


# =============================================================================
# KernleEscrowFactory ABI
# =============================================================================
# Factory contract for deploying per-job escrow contracts.

KERNLE_ESCROW_FACTORY_ABI: ABI = [
    # Constructor
    {
        "type": "constructor",
        "inputs": [
            {"name": "usdcToken", "type": "address"},
            {"name": "arbitrator", "type": "address"},
            {"name": "defaultApprovalTimeout", "type": "uint256"},
        ],
    },
    # === State Variables ===
    {
        "type": "function",
        "name": "usdcToken",
        "inputs": [],
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "arbitrator",
        "inputs": [],
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "defaultApprovalTimeout",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "escrowCount",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
    },
    # === Create Escrow ===
    {
        "type": "function",
        "name": "createEscrow",
        "inputs": [
            {"name": "jobId", "type": "bytes32"},
            {"name": "amount", "type": "uint256"},
            {"name": "deadline", "type": "uint256"},
        ],
        "outputs": [{"name": "escrow", "type": "address"}],
        "stateMutability": "nonpayable",
    },
    # === Getters ===
    {
        "type": "function",
        "name": "getEscrow",
        "inputs": [{"name": "jobId", "type": "bytes32"}],
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "getEscrowByIndex",
        "inputs": [{"name": "index", "type": "uint256"}],
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "view",
    },
    # === Admin Functions ===
    {
        "type": "function",
        "name": "setArbitrator",
        "inputs": [{"name": "_arbitrator", "type": "address"}],
        "outputs": [],
        "stateMutability": "nonpayable",
    },
    {
        "type": "function",
        "name": "setDefaultApprovalTimeout",
        "inputs": [{"name": "_timeout", "type": "uint256"}],
        "outputs": [],
        "stateMutability": "nonpayable",
    },
    # === Events ===
    {
        "type": "event",
        "name": "EscrowCreated",
        "inputs": [
            {"name": "jobId", "type": "bytes32", "indexed": True},
            {"name": "escrow", "type": "address", "indexed": True},
            {"name": "client", "type": "address", "indexed": True},
            {"name": "amount", "type": "uint256", "indexed": False},
        ],
    },
    {
        "type": "event",
        "name": "ArbitratorUpdated",
        "inputs": [
            {"name": "oldArbitrator", "type": "address", "indexed": True},
            {"name": "newArbitrator", "type": "address", "indexed": True},
        ],
    },
]


# =============================================================================
# ERC20 (USDC) ABI
# =============================================================================
# Standard ERC20 interface for USDC token interactions.

ERC20_ABI: ABI = [
    # === ERC20 Standard Functions ===
    {
        "type": "function",
        "name": "name",
        "inputs": [],
        "outputs": [{"name": "", "type": "string"}],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "symbol",
        "inputs": [],
        "outputs": [{"name": "", "type": "string"}],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "decimals",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint8"}],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "totalSupply",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "balanceOf",
        "inputs": [{"name": "account", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "allowance",
        "inputs": [
            {"name": "owner", "type": "address"},
            {"name": "spender", "type": "address"},
        ],
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "transfer",
        "inputs": [
            {"name": "to", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
    },
    {
        "type": "function",
        "name": "approve",
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
    },
    {
        "type": "function",
        "name": "transferFrom",
        "inputs": [
            {"name": "from", "type": "address"},
            {"name": "to", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
    },
    # === ERC20 Events ===
    {
        "type": "event",
        "name": "Transfer",
        "inputs": [
            {"name": "from", "type": "address", "indexed": True},
            {"name": "to", "type": "address", "indexed": True},
            {"name": "value", "type": "uint256", "indexed": False},
        ],
    },
    {
        "type": "event",
        "name": "Approval",
        "inputs": [
            {"name": "owner", "type": "address", "indexed": True},
            {"name": "spender", "type": "address", "indexed": True},
            {"name": "value", "type": "uint256", "indexed": False},
        ],
    },
]


# =============================================================================
# Escrow Status Enum
# =============================================================================
# Maps to the Solidity enum in KernleEscrow.sol


class EscrowStatus:
    """Escrow contract status values (maps to Solidity enum)."""

    CREATED = 0  # Contract created but not funded
    FUNDED = 1  # Client deposited USDC
    ACCEPTED = 2  # Worker assigned
    DELIVERED = 3  # Worker submitted deliverable
    RELEASED = 4  # Payment released to worker
    REFUNDED = 5  # Payment refunded to client
    DISPUTED = 6  # Dispute raised

    _NAMES = {
        0: "created",
        1: "funded",
        2: "accepted",
        3: "delivered",
        4: "released",
        5: "refunded",
        6: "disputed",
    }

    @classmethod
    def name(cls, value: int) -> str:
        """Get human-readable name for status value."""
        return cls._NAMES.get(value, f"unknown({value})")


# =============================================================================
# Helper Functions
# =============================================================================


def get_event_signature(event_name: str, abi: ABI) -> str:
    """Get the keccak256 event signature for an event name.

    Args:
        event_name: Name of the event
        abi: Contract ABI

    Returns:
        Event signature string (e.g., "Funded(address,uint256)")

    TODO: Add keccak256 hashing when web3 is available
    """
    for entry in abi:
        if entry.get("type") == "event" and entry.get("name") == event_name:
            inputs = entry.get("inputs", [])
            param_types = ",".join(inp["type"] for inp in inputs)
            return f"{event_name}({param_types})"
    raise ValueError(f"Event {event_name} not found in ABI")


def get_function_selector(function_name: str, abi: ABI) -> str:
    """Get the 4-byte function selector for a function name.

    Args:
        function_name: Name of the function
        abi: Contract ABI

    Returns:
        Function selector string (e.g., "fund()")

    TODO: Add keccak256 hashing when web3 is available
    """
    for entry in abi:
        if entry.get("type") == "function" and entry.get("name") == function_name:
            inputs = entry.get("inputs", [])
            param_types = ",".join(inp["type"] for inp in inputs)
            return f"{function_name}({param_types})"
    raise ValueError(f"Function {function_name} not found in ABI")
