"""Subscription management for Kernle cloud sync."""

from .models import (
    SubscriptionTier,
    SubscriptionStatus,
    PaymentStatus,
    TierConfig,
    TIER_CONFIGS,
    Subscription,
    SubscriptionPayment,
    UsageRecord,
    QuotaCheckResult,
    UpgradeRequest,
    UpgradeResponse,
    DowngradeRequest,
    DowngradeResponse,
    CancelRequest,
    CancelResponse,
    SubscriptionStatusResponse,
    UsageResponse,
    PaymentInfo,
)
from .service import (
    SubscriptionService,
    # Module-level bridge functions (used by routes)
    get_subscription,
    get_current_usage,
    create_upgrade_payment,
    get_payment,
    confirm_payment,
    schedule_downgrade,
    cancel_subscription,
    reactivate_subscription,
    list_payments,
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
