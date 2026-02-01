"""Pydantic models for the Kernle subscription system.

All monetary values use Decimal — never float — for precision.
Quotas apply only to storage and stack count, never sync frequency.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================


class SubscriptionTier(str, Enum):
    """Available subscription tiers."""

    free = "free"
    core = "core"
    pro = "pro"
    enterprise = "enterprise"


class SubscriptionStatus(str, Enum):
    """Subscription lifecycle states."""

    active = "active"
    grace_period = "grace_period"
    cancelled = "cancelled"
    expired = "expired"


class PaymentStatus(str, Enum):
    """On-chain payment verification states."""

    pending = "pending"
    confirmed = "confirmed"
    failed = "failed"
    refunded = "refunded"


# =============================================================================
# Tier Configuration
# =============================================================================


class OverflowPricing(BaseModel):
    """Per-unit overflow pricing beyond tier limits."""

    per_stack_usdc: Decimal = Decimal("0")
    per_gb_usdc: Decimal = Decimal("0")


class TierConfig(BaseModel):
    """Static configuration for a subscription tier."""

    tier: SubscriptionTier
    price_usdc: Decimal
    storage_limit_bytes: int
    stack_limit: int
    overflow: OverflowPricing | None = None
    is_custom: bool = False  # True for enterprise

    class Config:
        frozen = True


# Canonical tier definitions — single source of truth
TIER_CONFIGS: dict[SubscriptionTier, TierConfig] = {
    SubscriptionTier.free: TierConfig(
        tier=SubscriptionTier.free,
        price_usdc=Decimal("0"),
        storage_limit_bytes=10 * 1024 * 1024,  # 10 MB
        stack_limit=1,
    ),
    SubscriptionTier.core: TierConfig(
        tier=SubscriptionTier.core,
        price_usdc=Decimal("5"),
        storage_limit_bytes=100 * 1024 * 1024,  # 100 MB
        stack_limit=3,
        overflow=OverflowPricing(
            per_stack_usdc=Decimal("1.50"),
            per_gb_usdc=Decimal("0.50"),
        ),
    ),
    SubscriptionTier.pro: TierConfig(
        tier=SubscriptionTier.pro,
        price_usdc=Decimal("15"),
        storage_limit_bytes=1 * 1024 * 1024 * 1024,  # 1 GB
        stack_limit=10,
        overflow=OverflowPricing(
            per_stack_usdc=Decimal("1.00"),
            per_gb_usdc=Decimal("0.50"),
        ),
    ),
    SubscriptionTier.enterprise: TierConfig(
        tier=SubscriptionTier.enterprise,
        price_usdc=Decimal("0"),  # Custom pricing
        storage_limit_bytes=0,  # Unlimited (enforced differently)
        stack_limit=0,  # Unlimited
        is_custom=True,
    ),
}


def get_tier_config(tier: SubscriptionTier) -> TierConfig:
    """Look up the config for a tier. Raises KeyError for unknown tiers."""
    return TIER_CONFIGS[tier]


# Tier ordering for upgrade / downgrade validation
_TIER_ORDER: dict[SubscriptionTier, int] = {
    SubscriptionTier.free: 0,
    SubscriptionTier.core: 1,
    SubscriptionTier.pro: 2,
    SubscriptionTier.enterprise: 3,
}


def is_upgrade(current: SubscriptionTier, target: SubscriptionTier) -> bool:
    """Return True if moving from *current* to *target* is an upgrade."""
    return _TIER_ORDER[target] > _TIER_ORDER[current]


def is_downgrade(current: SubscriptionTier, target: SubscriptionTier) -> bool:
    """Return True if moving from *current* to *target* is a downgrade."""
    return _TIER_ORDER[target] < _TIER_ORDER[current]


# =============================================================================
# Database / Domain Models
# =============================================================================


class Subscription(BaseModel):
    """A user's subscription record (mirrors the subscriptions table)."""

    id: str
    user_id: str
    tier: SubscriptionTier
    status: SubscriptionStatus = SubscriptionStatus.active
    starts_at: datetime
    renews_at: datetime | None = None
    cancelled_at: datetime | None = None
    auto_renew: bool = True
    created_at: datetime | None = None
    updated_at: datetime | None = None


class SubscriptionPayment(BaseModel):
    """Record of a single payment (mirrors subscription_payments table)."""

    id: str
    subscription_id: str
    amount: Decimal
    currency: str = "USDC"
    tx_hash: str
    from_address: str
    to_address: str
    chain: str = "base"
    status: PaymentStatus = PaymentStatus.pending
    period_start: datetime
    period_end: datetime
    created_at: datetime | None = None
    confirmed_at: datetime | None = None


class UsageRecord(BaseModel):
    """Current-period usage for a user (mirrors usage_records table).

    Note: sync_count is tracked for analytics only — there are NO
    sync frequency limits on any tier.
    """

    user_id: str
    period: str  # "YYYY-MM"
    storage_bytes: int = 0
    sync_count: int = 0
    stacks_syncing: int = 0
    updated_at: datetime | None = None


# =============================================================================
# Quota Check
# =============================================================================


class QuotaCheckResult(BaseModel):
    """Result of a quota check against the user's tier."""

    allowed: bool
    reason: str | None = None
    tier: SubscriptionTier
    storage_used: int = 0
    storage_limit: int = 0
    stacks_used: int = 0
    stacks_limit: int = 0


# =============================================================================
# API Request / Response Models
# =============================================================================


class PaymentInfo(BaseModel):
    """Payment details returned after an upgrade or recorded payment."""

    amount: Decimal
    currency: str = "USDC"
    to_address: str
    chain: str = "base"
    tx_hash: str | None = None
    status: PaymentStatus = PaymentStatus.pending


class UpgradeRequest(BaseModel):
    """Request to upgrade subscription tier."""

    tier: SubscriptionTier


class UpgradeResponse(BaseModel):
    """Response after requesting an upgrade."""

    subscription: Subscription
    payment: PaymentInfo
    message: str


class DowngradeRequest(BaseModel):
    """Request to downgrade subscription tier."""

    tier: SubscriptionTier


class DowngradeResponse(BaseModel):
    """Response after requesting a downgrade (effective next billing period)."""

    subscription: Subscription
    effective_at: datetime
    message: str


class CancelRequest(BaseModel):
    """Request to cancel auto-renewal."""

    reason: str | None = None


class CancelResponse(BaseModel):
    """Response after cancellation request."""

    subscription: Subscription
    message: str


class SubscriptionStatusResponse(BaseModel):
    """Full subscription status including usage and renewal info."""

    subscription: Subscription
    tier_config: TierConfig
    usage: UsageRecord
    renewal_amount: Decimal | None = None
    can_renew: bool | None = None


class UsageResponse(BaseModel):
    """Current period usage details."""

    period: str
    storage_bytes: int
    storage_limit_bytes: int
    sync_count: int
    stacks_syncing: int
    stacks_limit: int
