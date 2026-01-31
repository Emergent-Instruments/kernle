"""Wallet routes for Kernle Commerce.

Endpoints for managing agent crypto wallets on Base.
"""

from datetime import datetime
from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from ...auth import CurrentAgent
from ...database import Database
from ...logging_config import get_logger
from ...rate_limit import limiter

logger = get_logger("kernle.commerce.wallets")
router = APIRouter(prefix="/wallets", tags=["commerce", "wallets"])


# =============================================================================
# Request/Response Models
# =============================================================================


class WalletResponse(BaseModel):
    """Wallet details response."""

    id: str
    agent_id: str
    wallet_address: str
    chain: str
    status: str  # pending_claim, active, paused, frozen
    owner_eoa: str | None = None
    spending_limit_per_tx: Decimal
    spending_limit_daily: Decimal
    created_at: datetime
    claimed_at: datetime | None = None


class WalletBalanceResponse(BaseModel):
    """Wallet balance response."""

    wallet_address: str
    chain: str
    balance_usdc: Decimal
    balance_eth: Decimal = Decimal("0")  # For gas estimation


class WalletClaimRequest(BaseModel):
    """Request to claim a wallet."""

    owner_eoa: str = Field(
        ...,
        pattern=r"^0x[a-fA-F0-9]{40}$",
        description="Ethereum address for wallet ownership/recovery",
    )


class WalletClaimResponse(BaseModel):
    """Response after claiming a wallet."""

    wallet_address: str
    owner_eoa: str
    status: str
    claimed_at: datetime


# =============================================================================
# Database Operations
# =============================================================================


WALLET_ACCOUNTS_TABLE = "wallet_accounts"


async def get_wallet_by_agent(db, agent_id: str) -> dict | None:
    """Get wallet for an agent."""
    result = db.table(WALLET_ACCOUNTS_TABLE).select("*").eq("agent_id", agent_id).execute()
    return result.data[0] if result.data else None


async def get_wallet_by_user(db, user_id: str) -> dict | None:
    """Get wallet for a user."""
    result = db.table(WALLET_ACCOUNTS_TABLE).select("*").eq("user_id", user_id).execute()
    return result.data[0] if result.data else None


async def claim_wallet(db, wallet_id: str, owner_eoa: str) -> dict | None:
    """Claim a wallet by setting owner EOA."""
    from datetime import timezone

    result = (
        db.table(WALLET_ACCOUNTS_TABLE)
        .update(
            {
                "owner_eoa": owner_eoa,
                "status": "active",
                "claimed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        .eq("id", wallet_id)
        .eq("status", "pending_claim")  # Can only claim if pending
        .execute()
    )
    return result.data[0] if result.data else None


# =============================================================================
# Routes
# =============================================================================


@router.get("/me", response_model=WalletResponse)
@limiter.limit("60/minute")
async def get_my_wallet(
    request: Request,
    auth: CurrentAgent,
    db: Database,
):
    """
    Get the authenticated agent's wallet.

    Returns the CDP Smart Wallet details for the current agent.
    Wallets are created automatically at agent registration.
    """
    log_prefix = f"{auth.user_id}"
    logger.info(f"GET /wallets/me | {log_prefix}")

    # Try by agent_id first, then by user_id
    wallet = None
    if auth.agent_id:
        wallet = await get_wallet_by_agent(db, auth.agent_id)

    if not wallet and auth.user_id:
        wallet = await get_wallet_by_user(db, auth.user_id)

    if not wallet:
        logger.warning(f"Wallet not found | {log_prefix}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Wallet not found. Contact support to provision one.",
        )

    return WalletResponse(
        id=wallet["id"],
        agent_id=wallet["agent_id"],
        wallet_address=wallet["wallet_address"],
        chain=wallet["chain"],
        status=wallet["status"],
        owner_eoa=wallet.get("owner_eoa"),
        spending_limit_per_tx=Decimal(str(wallet["spending_limit_per_tx"])),
        spending_limit_daily=Decimal(str(wallet["spending_limit_daily"])),
        created_at=wallet["created_at"],
        claimed_at=wallet.get("claimed_at"),
    )


@router.get("/me/balance", response_model=WalletBalanceResponse)
@limiter.limit("30/minute")
async def get_wallet_balance(
    request: Request,
    auth: CurrentAgent,
    db: Database,
):
    """
    Get the USDC balance of the agent's wallet.

    Queries the Base network for current balance.
    Note: This is a placeholder - actual balance fetching requires
    blockchain integration (CDP or direct RPC).
    """
    log_prefix = f"{auth.user_id}"
    logger.info(f"GET /wallets/me/balance | {log_prefix}")

    # Get wallet
    wallet = None
    if auth.agent_id:
        wallet = await get_wallet_by_agent(db, auth.agent_id)
    if not wallet and auth.user_id:
        wallet = await get_wallet_by_user(db, auth.user_id)

    if not wallet:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Wallet not found",
        )

    # TODO: Integrate with CDP or Base RPC to get actual balance
    # For now, return placeholder values
    # In production, this would call:
    # - CDP API to get balance
    # - Or direct RPC call to Base network

    logger.info(f"Balance query | wallet={wallet['wallet_address']} | {log_prefix}")

    return WalletBalanceResponse(
        wallet_address=wallet["wallet_address"],
        chain=wallet["chain"],
        balance_usdc=Decimal("0.00"),  # Placeholder - implement CDP integration
        balance_eth=Decimal("0.00"),
    )


@router.post("/claim", response_model=WalletClaimResponse)
@limiter.limit("5/minute")
async def claim_my_wallet(
    request: Request,
    claim_request: WalletClaimRequest,
    auth: CurrentAgent,
    db: Database,
):
    """
    Claim the agent's wallet by setting an owner EOA.

    The owner EOA is a regular Ethereum address that can:
    - Recover the wallet if needed
    - Override spending limits
    - Pause or freeze the wallet

    This operation can only be done once (when status is 'pending_claim').
    """
    log_prefix = f"{auth.user_id}"
    logger.info(f"POST /wallets/claim | owner_eoa={claim_request.owner_eoa} | {log_prefix}")

    # Get current wallet
    wallet = None
    if auth.agent_id:
        wallet = await get_wallet_by_agent(db, auth.agent_id)
    if not wallet and auth.user_id:
        wallet = await get_wallet_by_user(db, auth.user_id)

    if not wallet:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Wallet not found",
        )

    if wallet["status"] != "pending_claim":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Wallet already claimed or in status: {wallet['status']}",
        )

    # SECURITY FIX: Verify ownership before allowing claim
    # Only the user who owns the wallet (or their agent) can claim it
    wallet_user_id = wallet.get("user_id")
    if wallet_user_id and wallet_user_id != auth.user_id:
        logger.warning(
            f"Unauthorized wallet claim attempt | "
            f"wallet_user={wallet_user_id} | requester={auth.user_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot claim another user's wallet",
        )

    # Claim the wallet
    claimed = await claim_wallet(db, wallet["id"], claim_request.owner_eoa)

    if not claimed:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to claim wallet",
        )

    logger.info(f"Wallet claimed | wallet={wallet['wallet_address']} | {log_prefix}")

    return WalletClaimResponse(
        wallet_address=claimed["wallet_address"],
        owner_eoa=claimed["owner_eoa"],
        status=claimed["status"],
        claimed_at=claimed["claimed_at"],
    )
