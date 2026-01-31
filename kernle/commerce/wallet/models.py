"""
Wallet data models.

WalletAccount represents an agent's crypto wallet on Base.
Wallets are provisioned at agent registration via Coinbase CDP Smart Wallet.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class WalletStatus(str, Enum):
    """Wallet account status."""
    
    PENDING_CLAIM = "pending_claim"  # Created but not linked to human owner
    ACTIVE = "active"  # Claimed and operational
    PAUSED = "paused"  # Temporarily suspended by owner
    FROZEN = "frozen"  # Frozen by system/arbitrator


@dataclass
class WalletAccount:
    """Agent's crypto wallet on Base.
    
    Every Kernle agent gets a CDP Smart Wallet at registration. The wallet
    starts in 'pending_claim' status until the human owner links their EOA
    (usually during Twitter verification).
    
    Attributes:
        id: Unique wallet record ID (UUID)
        agent_id: Kernle agent ID this wallet belongs to
        wallet_address: Ethereum address (0x...)
        chain: Blockchain network (base, base-sepolia)
        status: Wallet status (pending_claim, active, paused, frozen)
        user_id: User ID of the wallet owner (links to users table)
        owner_eoa: Human's recovery/control address (set on claim)
        spending_limit_per_tx: Maximum USDC per transaction
        spending_limit_daily: Maximum USDC per 24 hours
        cdp_wallet_id: Coinbase CDP internal wallet identifier
        created_at: When the wallet was created
        claimed_at: When the wallet was claimed by the human owner
    """
    
    id: str
    agent_id: str
    wallet_address: str
    chain: str = "base"
    status: str = "pending_claim"
    user_id: Optional[str] = None
    owner_eoa: Optional[str] = None
    spending_limit_per_tx: float = 100.0
    spending_limit_daily: float = 1000.0
    cdp_wallet_id: Optional[str] = None
    created_at: Optional[datetime] = None
    claimed_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate wallet data."""
        # Normalize status to enum value if string
        if isinstance(self.status, WalletStatus):
            self.status = self.status.value
        
        # Validate wallet address format: exactly 42 chars, 0x + 40 hex chars
        self._validate_eth_address(self.wallet_address, "wallet_address")
        
        # Validate owner_eoa format when present
        if self.owner_eoa is not None:
            self._validate_eth_address(self.owner_eoa, "owner_eoa")
        
        # Validate spending limits must be > 0
        if self.spending_limit_per_tx <= 0:
            raise ValueError(f"spending_limit_per_tx must be > 0, got {self.spending_limit_per_tx}")
        if self.spending_limit_daily <= 0:
            raise ValueError(f"spending_limit_daily must be > 0, got {self.spending_limit_daily}")
        
        # Validate chain
        valid_chains = {"base", "base-sepolia"}
        if self.chain not in valid_chains:
            raise ValueError(f"Invalid chain: {self.chain}. Must be one of {valid_chains}")
        
        # Validate status
        valid_statuses = {s.value for s in WalletStatus}
        if self.status not in valid_statuses:
            raise ValueError(f"Invalid status: {self.status}. Must be one of {valid_statuses}")
    
    @staticmethod
    def _validate_eth_address(address: str, field_name: str) -> None:
        """Validate Ethereum address format: 0x + 40 hex chars = 42 chars total."""
        if not address:
            raise ValueError(f"{field_name} cannot be empty")
        if len(address) != 42:
            raise ValueError(f"{field_name} must be exactly 42 characters, got {len(address)}")
        if not address.startswith("0x"):
            raise ValueError(f"{field_name} must start with '0x'")
        # Check remaining 40 chars are valid hex
        hex_part = address[2:]
        if not all(c in "0123456789abcdefABCDEF" for c in hex_part):
            raise ValueError(f"{field_name} must contain only hex characters after '0x'")
    
    @property
    def is_active(self) -> bool:
        """Check if wallet is operational."""
        return self.status == WalletStatus.ACTIVE.value
    
    @property
    def is_claimed(self) -> bool:
        """Check if wallet has been claimed by a human owner."""
        return self.claimed_at is not None or self.owner_eoa is not None
    
    @property
    def can_transact(self) -> bool:
        """Check if wallet can perform transactions."""
        return self.status == WalletStatus.ACTIVE.value
    
    def to_dict(self, include_internal: bool = False) -> dict:
        """Convert to dictionary for serialization.
        
        Args:
            include_internal: If True, include internal fields like cdp_wallet_id.
                            Default False for security.
        """
        result = {
            "id": self.id,
            "agent_id": self.agent_id,
            "wallet_address": self.wallet_address,
            "chain": self.chain,
            "status": self.status,
            "user_id": self.user_id,
            "owner_eoa": self.owner_eoa,
            "spending_limit_per_tx": self.spending_limit_per_tx,
            "spending_limit_daily": self.spending_limit_daily,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "claimed_at": self.claimed_at.isoformat() if self.claimed_at else None,
        }
        
        # SECURITY FIX: Only include internal IDs when explicitly requested
        if include_internal:
            result["cdp_wallet_id"] = self.cdp_wallet_id
        
        return result
    
    def to_public_dict(self) -> dict:
        """Convert to dictionary with only public-safe fields.
        
        Use this for API responses to avoid leaking internal identifiers.
        """
        return {
            "id": self.id,
            "wallet_address": self.wallet_address,
            "chain": self.chain,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "claimed_at": self.claimed_at.isoformat() if self.claimed_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "WalletAccount":
        """Create from dictionary."""
        # Parse datetime fields
        created_at = data.get("created_at")
        if created_at and isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        
        claimed_at = data.get("claimed_at")
        if claimed_at and isinstance(claimed_at, str):
            claimed_at = datetime.fromisoformat(claimed_at.replace("Z", "+00:00"))
        
        return cls(
            id=data["id"],
            agent_id=data["agent_id"],
            wallet_address=data["wallet_address"],
            chain=data.get("chain", "base"),
            status=data.get("status", "pending_claim"),
            user_id=data.get("user_id"),
            owner_eoa=data.get("owner_eoa"),
            spending_limit_per_tx=float(data.get("spending_limit_per_tx", 100.0)),
            spending_limit_daily=float(data.get("spending_limit_daily", 1000.0)),
            cdp_wallet_id=data.get("cdp_wallet_id"),
            created_at=created_at,
            claimed_at=claimed_at,
        )
