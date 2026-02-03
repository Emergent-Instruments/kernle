"""Subscription management for Kernle cloud sync."""

from .models import (
    TIER_CONFIGS,
    CancelRequest,
    CancelResponse,
    DowngradeRequest,
    DowngradeResponse,
    PaymentInfo,
    PaymentStatus,
    QuotaCheckResult,
    Subscription,
    SubscriptionPayment,
    SubscriptionStatus,
    SubscriptionStatusResponse,
    SubscriptionTier,
    TierConfig,
    UpgradeRequest,
    UpgradeResponse,
    UsageRecord,
    UsageResponse,
)
from .service import (
    SubscriptionService,
    cancel_subscription,
    confirm_payment,
    create_upgrade_payment,
    get_current_usage,
    get_payment,
    # Module-level bridge functions (used by routes)
    get_subscription,
    list_payments,
    reactivate_subscription,
    schedule_downgrade,
)

__all__ = [
    # Enums
    "SubscriptionTier",
    "SubscriptionStatus",
    "PaymentStatus",
    # Config
    "TierConfig",
    "TIER_CONFIGS",
    # Models
    "Subscription",
    "SubscriptionPayment",
    "UsageRecord",
    "QuotaCheckResult",
    # API models
    "UpgradeRequest",
    "UpgradeResponse",
    "DowngradeRequest",
    "DowngradeResponse",
    "CancelRequest",
    "CancelResponse",
    "SubscriptionStatusResponse",
    "UsageResponse",
    "PaymentInfo",
    # Service
    "SubscriptionService",
    # Bridge functions
    "get_subscription",
    "get_current_usage",
    "create_upgrade_payment",
    "get_payment",
    "confirm_payment",
    "schedule_downgrade",
    "cancel_subscription",
    "reactivate_subscription",
    "list_payments",
]
