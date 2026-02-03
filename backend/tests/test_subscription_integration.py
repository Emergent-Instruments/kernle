"""Integration tests for subscription service bridge functions.

Tests the module-level functions that wire together:
- Route layer (dict interface)
- SubscriptionService (model interface)
- On-chain USDC payment verification
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.payments.verification import TransferVerificationResult
from app.subscriptions.models import (
    Subscription,
    SubscriptionStatus,
    SubscriptionTier,
    UsageRecord,
)
from app.subscriptions.service import (
    TREASURY_ADDRESS,
    cancel_subscription,
    confirm_payment,
    create_upgrade_payment,
    get_current_usage,
    get_subscription,
    list_payments,
    reactivate_subscription,
    schedule_downgrade,
)

# =============================================================================
# Fixtures
# =============================================================================


def _make_db_mock():
    """Create a mock Supabase client with chainable query builder."""
    db = MagicMock()
    return db


def _make_subscription(
    user_id: str = "user-1",
    tier: str = "free",
    status: str = "active",
    sub_id: str = "sub-1",
) -> Subscription:
    return Subscription(
        id=sub_id,
        user_id=user_id,
        tier=SubscriptionTier(tier),
        status=SubscriptionStatus(status),
        starts_at=datetime.now(timezone.utc),
        renews_at=datetime.now(timezone.utc) + timedelta(days=30) if tier != "free" else None,
        auto_renew=tier != "free",
    )


def _make_usage(user_id: str = "user-1") -> UsageRecord:
    return UsageRecord(
        user_id=user_id,
        period=datetime.now(timezone.utc).strftime("%Y-%m"),
        storage_bytes=1024,
        sync_count=5,
        stacks_syncing=1,
    )


# =============================================================================
# get_subscription
# =============================================================================


class TestGetSubscription:
    @pytest.mark.asyncio
    async def test_returns_dict_with_tier_info(self):
        sub = _make_subscription(tier="core")
        with patch(
            "app.subscriptions.service.SubscriptionService.get_or_create_subscription",
            new_callable=AsyncMock,
            return_value=sub,
        ):
            result = await get_subscription(MagicMock(), "user-1")

        assert result is not None
        assert result["tier"] == "core"
        assert result["status"] == "active"
        assert result["auto_renew"] is True
        assert result["renewal_amount"] == "5"

    @pytest.mark.asyncio
    async def test_free_tier_no_renewal_amount(self):
        sub = _make_subscription(tier="free")
        with patch(
            "app.subscriptions.service.SubscriptionService.get_or_create_subscription",
            new_callable=AsyncMock,
            return_value=sub,
        ):
            result = await get_subscription(MagicMock(), "user-1")

        assert result["tier"] == "free"
        assert result["renewal_amount"] is None


# =============================================================================
# get_current_usage
# =============================================================================


class TestGetCurrentUsage:
    @pytest.mark.asyncio
    async def test_returns_usage_dict(self):
        sub = _make_subscription(tier="core")
        usage = _make_usage()
        with (
            patch(
                "app.subscriptions.service.SubscriptionService.get_or_create_subscription",
                new_callable=AsyncMock,
                return_value=sub,
            ),
            patch(
                "app.subscriptions.service.SubscriptionService.get_usage",
                new_callable=AsyncMock,
                return_value=usage,
            ),
        ):
            result = await get_current_usage(MagicMock(), "user-1")

        assert result["storage_used"] == 1024
        assert result["storage_limit"] == 100 * 1024 * 1024  # Core = 100MB
        assert result["agents_used"] == 1
        assert result["agents_limit"] == 3  # Core = 3 stacks


# =============================================================================
# create_upgrade_payment
# =============================================================================


class TestCreateUpgradePayment:
    @pytest.mark.asyncio
    async def test_creates_payment_intent(self):
        sub = _make_subscription(tier="free")
        db = MagicMock()

        # Mock the insert to return success
        insert_result = MagicMock()
        insert_result.data = [{"id": "test-payment-id"}]
        db.table.return_value.insert.return_value.execute.return_value = insert_result

        with patch(
            "app.subscriptions.service.SubscriptionService.get_or_create_subscription",
            new_callable=AsyncMock,
            return_value=sub,
        ):
            # Use to_thread mock since the actual call wraps in asyncio.to_thread
            with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
                mock_thread.return_value = insert_result
                result = await create_upgrade_payment(db, "user-1", "core")

        assert result is not None
        assert result["amount"] == "5"
        assert result["currency"] == "USDC"
        assert result["treasury_address"] == TREASURY_ADDRESS

    @pytest.mark.asyncio
    async def test_pro_tier_amount(self):
        sub = _make_subscription(tier="free")
        db = MagicMock()

        insert_result = MagicMock()
        insert_result.data = [{"id": "test-payment-id"}]

        with (
            patch(
                "app.subscriptions.service.SubscriptionService.get_or_create_subscription",
                new_callable=AsyncMock,
                return_value=sub,
            ),
            patch("asyncio.to_thread", new_callable=AsyncMock, return_value=insert_result),
        ):
            result = await create_upgrade_payment(db, "user-1", "pro")

        assert result["amount"] == "15"


# =============================================================================
# confirm_payment — the core integration point
# =============================================================================


VALID_TX_HASH = "0x" + "a1" * 32  # Valid 66-char hex tx hash


def _make_pending_intent(**overrides) -> dict:
    """Create a standard pending payment intent for tests."""
    base = {
        "id": "pay-1",
        "user_id": "user-1",
        "subscription_id": "sub-1",
        "tier": "core",
        "amount": "5",
        "currency": "USDC",
        "treasury_address": TREASURY_ADDRESS,
        "chain": "base",
        "status": "pending",
        "expires_at": (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat(),
    }
    base.update(overrides)
    return base


def _mock_claim_success():
    """Mock asyncio.to_thread to simulate successful atomic claim."""
    async def _side_effect(fn, *args, **kwargs):
        result = MagicMock()
        result.data = [{"id": "pay-1"}]
        return result
    return _side_effect


class TestConfirmPayment:
    """Tests for confirm_payment which wires together verification + upgrade."""

    @pytest.mark.asyncio
    async def test_successful_verification_activates_tier(self):
        """Happy path: on-chain tx verified → tier upgraded."""
        intent = _make_pending_intent()

        verification_result = TransferVerificationResult(
            success=True,
            tx_hash=VALID_TX_HASH,
            chain="base",
            from_address="0xsender",
            to_address=TREASURY_ADDRESS,
            amount=Decimal("5.00"),
            amount_raw=5000000,
            block_number=12345,
            confirmations=3,
        )

        sub = _make_subscription(tier="free")
        upgraded_sub = _make_subscription(tier="core")
        payment_record = MagicMock()

        with (
            patch(
                "app.subscriptions.service.get_payment",
                new_callable=AsyncMock,
                return_value=intent,
            ),
            patch(
                "app.subscriptions.service._check_tx_hash_used",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "app.payments.verify_usdc_transfer",
                new_callable=AsyncMock,
                return_value=verification_result,
            ),
            patch("asyncio.to_thread", new_callable=AsyncMock, side_effect=_mock_claim_success()),
            patch(
                "app.subscriptions.service.SubscriptionService.get_or_create_subscription",
                new_callable=AsyncMock,
                return_value=sub,
            ),
            patch(
                "app.subscriptions.service.SubscriptionService.upgrade_tier",
                new_callable=AsyncMock,
                return_value=(upgraded_sub, MagicMock()),
            ),
            patch(
                "app.subscriptions.service.SubscriptionService.record_payment",
                new_callable=AsyncMock,
                return_value=payment_record,
            ),
            patch(
                "app.subscriptions.service.SubscriptionService.confirm_payment",
                new_callable=AsyncMock,
                return_value=payment_record,
            ),
        ):
            result = await confirm_payment(MagicMock(), "pay-1", VALID_TX_HASH, "user-1")

        assert result is not None
        assert result["status"] == "confirmed"
        assert result["tier"] == "core"
        assert "activated_at" in result

    @pytest.mark.asyncio
    async def test_failed_verification_marks_failed(self):
        """On-chain verification fails → payment marked failed."""
        intent = _make_pending_intent()

        verification_result = TransferVerificationResult(
            success=False,
            tx_hash=VALID_TX_HASH,
            chain="base",
            error="No USDC transfer found in transaction",
            error_code="NO_USDC_TRANSFER",
        )

        with (
            patch(
                "app.subscriptions.service.get_payment",
                new_callable=AsyncMock,
                return_value=intent,
            ),
            patch(
                "app.subscriptions.service._check_tx_hash_used",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch("asyncio.to_thread", new_callable=AsyncMock, side_effect=_mock_claim_success()),
            patch(
                "app.payments.verify_usdc_transfer",
                new_callable=AsyncMock,
                return_value=verification_result,
            ),
        ):
            result = await confirm_payment(MagicMock(), "pay-1", VALID_TX_HASH, "user-1")

        assert result["status"] == "failed"
        assert "No USDC transfer" in result["message"]

    @pytest.mark.asyncio
    async def test_already_confirmed_returns_early(self):
        """Payment already confirmed → return without re-verifying."""
        intent = _make_pending_intent(status="confirmed")

        with patch(
            "app.subscriptions.service.get_payment",
            new_callable=AsyncMock,
            return_value=intent,
        ):
            result = await confirm_payment(MagicMock(), "pay-1", VALID_TX_HASH, "user-1")

        assert result["status"] == "confirmed"
        assert result["tier"] == "core"
        assert "already" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_payment_not_found(self):
        """Unknown payment_id → returns None."""
        with patch(
            "app.subscriptions.service.get_payment",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await confirm_payment(MagicMock(), "no-such-pay", VALID_TX_HASH, "user-1")

        assert result is None

    @pytest.mark.asyncio
    async def test_network_error_marks_pending(self):
        """Verification network error → pending_verification (retry later)."""
        intent = _make_pending_intent()

        with (
            patch(
                "app.subscriptions.service.get_payment",
                new_callable=AsyncMock,
                return_value=intent,
            ),
            patch(
                "app.subscriptions.service._check_tx_hash_used",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch("asyncio.to_thread", new_callable=AsyncMock, side_effect=_mock_claim_success()),
            patch(
                "app.payments.verify_usdc_transfer",
                new_callable=AsyncMock,
                side_effect=Exception("RPC timeout"),
            ),
        ):
            result = await confirm_payment(MagicMock(), "pay-1", VALID_TX_HASH, "user-1")

        assert result["status"] == "pending_verification"
        assert "retry" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_amount_mismatch_fails(self):
        """Tx verified but amount doesn't match → failed."""
        intent = _make_pending_intent()

        verification_result = TransferVerificationResult(
            success=False,
            tx_hash=VALID_TX_HASH,
            chain="base",
            amount=Decimal("3.00"),
            error="Transfer found but doesn't match: amount: expected 5, got 3.000000",
            error_code="TRANSFER_MISMATCH",
        )

        with (
            patch(
                "app.subscriptions.service.get_payment",
                new_callable=AsyncMock,
                return_value=intent,
            ),
            patch(
                "app.subscriptions.service._check_tx_hash_used",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch("asyncio.to_thread", new_callable=AsyncMock, side_effect=_mock_claim_success()),
            patch(
                "app.payments.verify_usdc_transfer",
                new_callable=AsyncMock,
                return_value=verification_result,
            ),
        ):
            result = await confirm_payment(MagicMock(), "pay-1", VALID_TX_HASH, "user-1")

        assert result["status"] == "failed"
        assert "TRANSFER_MISMATCH" in str(result) or "match" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_invalid_tx_hash_rejected(self):
        """Invalid tx_hash format → immediate rejection."""
        intent = _make_pending_intent()

        with patch(
            "app.subscriptions.service.get_payment",
            new_callable=AsyncMock,
            return_value=intent,
        ):
            result = await confirm_payment(MagicMock(), "pay-1", "not-a-valid-hash", "user-1")

        assert result["status"] == "failed"
        assert "Invalid transaction hash" in result["message"]

    @pytest.mark.asyncio
    async def test_tx_hash_replay_rejected(self):
        """Tx hash already used → rejected before verification."""
        intent = _make_pending_intent()

        with (
            patch(
                "app.subscriptions.service.get_payment",
                new_callable=AsyncMock,
                return_value=intent,
            ),
            patch(
                "app.subscriptions.service._check_tx_hash_used",
                new_callable=AsyncMock,
                return_value=True,  # Already used!
            ),
        ):
            result = await confirm_payment(MagicMock(), "pay-1", VALID_TX_HASH, "user-1")

        assert result["status"] == "failed"
        assert "already been used" in result["message"]

    @pytest.mark.asyncio
    async def test_expired_intent_rejected(self):
        """Expired payment intent → rejected."""
        expired_intent = _make_pending_intent(
            expires_at=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        )

        with (
            patch(
                "app.subscriptions.service.get_payment",
                new_callable=AsyncMock,
                return_value=expired_intent,
            ),
            patch("asyncio.to_thread", new_callable=AsyncMock, side_effect=_mock_claim_success()),
        ):
            result = await confirm_payment(MagicMock(), "pay-1", VALID_TX_HASH, "user-1")

        assert result["status"] == "failed"
        assert "expired" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_non_pending_status_rejected(self):
        """Non-pending intent (e.g. failed) → rejected."""
        failed_intent = _make_pending_intent(status="failed")

        with patch(
            "app.subscriptions.service.get_payment",
            new_callable=AsyncMock,
            return_value=failed_intent,
        ):
            result = await confirm_payment(MagicMock(), "pay-1", VALID_TX_HASH, "user-1")

        assert result["status"] == "failed"
        assert "pending" in result["message"].lower()


# =============================================================================
# schedule_downgrade
# =============================================================================


class TestScheduleDowngrade:
    @pytest.mark.asyncio
    async def test_returns_effective_date(self):
        effective = datetime.now(timezone.utc) + timedelta(days=30)
        sub = _make_subscription(tier="core")
        with patch(
            "app.subscriptions.service.SubscriptionService.downgrade_tier",
            new_callable=AsyncMock,
            return_value=(sub, effective),
        ):
            result = await schedule_downgrade(MagicMock(), "user-1", "free")

        assert result is not None
        assert result["effective_at"] == effective

    @pytest.mark.asyncio
    async def test_invalid_downgrade_returns_none(self):
        with patch(
            "app.subscriptions.service.SubscriptionService.downgrade_tier",
            new_callable=AsyncMock,
            side_effect=ValueError("Cannot downgrade"),
        ):
            result = await schedule_downgrade(MagicMock(), "user-1", "pro")

        assert result is None


# =============================================================================
# cancel_subscription
# =============================================================================


class TestCancelSubscription:
    @pytest.mark.asyncio
    async def test_returns_cancelled_info(self):
        now = datetime.now(timezone.utc)
        sub = _make_subscription(tier="core")
        sub.cancelled_at = now
        sub.renews_at = now + timedelta(days=15)
        with patch(
            "app.subscriptions.service.SubscriptionService.cancel_subscription",
            new_callable=AsyncMock,
            return_value=sub,
        ):
            result = await cancel_subscription(MagicMock(), "user-1")

        assert result is not None
        assert "cancelled_at" in result
        assert "active_until" in result


# =============================================================================
# reactivate_subscription
# =============================================================================


class TestReactivateSubscription:
    @pytest.mark.asyncio
    async def test_reactivates_cancelled_sub(self):
        sub = _make_subscription(tier="core", status="active")
        sub.cancelled_at = datetime.now(timezone.utc)

        update_result = MagicMock()
        update_result.data = [{"id": sub.id}]

        with (
            patch(
                "app.subscriptions.service.SubscriptionService.get_or_create_subscription",
                new_callable=AsyncMock,
                return_value=sub,
            ),
            patch("asyncio.to_thread", new_callable=AsyncMock, return_value=update_result),
        ):
            result = await reactivate_subscription(MagicMock(), "user-1")

        assert result is not None
        assert "renews_at" in result


# =============================================================================
# list_payments
# =============================================================================


class TestListPayments:
    @pytest.mark.asyncio
    async def test_returns_paginated_payments(self):
        sub = _make_subscription(tier="core")

        query_result = MagicMock()
        query_result.data = [
            {
                "id": "pay-1",
                "amount": "5.000000",
                "currency": "USDC",
                "tx_hash": "0xabc",
                "from_address": "0xsender",
                "to_address": TREASURY_ADDRESS,
                "chain": "base",
                "status": "confirmed",
                "period_start": "2026-02-01T00:00:00Z",
                "period_end": "2026-03-01T00:00:00Z",
                "created_at": "2026-02-01T12:00:00Z",
                "confirmed_at": "2026-02-01T12:05:00Z",
            }
        ]
        query_result.count = 1

        with (
            patch(
                "app.subscriptions.service.SubscriptionService.get_or_create_subscription",
                new_callable=AsyncMock,
                return_value=sub,
            ),
            patch("asyncio.to_thread", new_callable=AsyncMock, return_value=query_result),
        ):
            result = await list_payments(MagicMock(), "user-1", page=1, per_page=20)

        assert result["total"] == 1
        assert len(result["payments"]) == 1
        assert result["payments"][0]["tx_hash"] == "0xabc"
