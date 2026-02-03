"""Wallet subsystem for Kernle Commerce.

Provides wallet management for Kernle agents using CDP Smart Wallets.

Models:
- WalletAccount: Agent's crypto wallet on Base
- WalletStatus: Wallet lifecycle status

Service:
- WalletService: Wallet operations (create, claim, transfer, etc.)
"""

from kernle.commerce.wallet.models import WalletAccount, WalletStatus
from kernle.commerce.wallet.service import (
    CDPIntegrationError,
    InsufficientBalanceError,
    SpendingLimitExceededError,
    TransferResult,
    WalletBalance,
    WalletNotActiveError,
    WalletNotFoundError,
    WalletService,
    WalletServiceError,
)

__all__ = [
    # Models
    "WalletAccount",
    "WalletStatus",
    # Service
    "WalletService",
    "WalletServiceError",
    "WalletNotFoundError",
    "WalletNotActiveError",
    "InsufficientBalanceError",
    "SpendingLimitExceededError",
    "CDPIntegrationError",
    "WalletBalance",
    "TransferResult",
]
