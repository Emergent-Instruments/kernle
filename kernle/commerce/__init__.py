"""
Kernle Commerce - Economic capabilities for AI agents.

Commerce is a capability layer for Kernle agents, enabling:
- Crypto wallets (USDC on Base)
- Jobs marketplace
- Escrow-backed payments

Memory defines WHO the agent is.
Commerce defines WHAT they can do economically.
"""

from kernle.commerce.jobs.models import ApplicationStatus, Job, JobApplication, JobStatus
from kernle.commerce.skills.models import Skill
from kernle.commerce.wallet.models import WalletAccount, WalletStatus

__all__ = [
    # Wallet
    "WalletAccount",
    "WalletStatus",
    # Jobs
    "Job",
    "JobApplication",
    "JobStatus",
    "ApplicationStatus",
    # Skills
    "Skill",
]
