"""Tests for wallet service."""

from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional

import pytest

from kernle.commerce.config import CommerceConfig
from kernle.commerce.wallet.models import WalletAccount, WalletStatus
from kernle.commerce.wallet.service import (
    SpendingLimitExceededError,
    TransferResult,
    WalletBalance,
    WalletNotActiveError,
    WalletNotFoundError,
    WalletService,
    WalletServiceError,
)


class InMemoryWalletStorage:
    """In-memory wallet storage for testing."""

    def __init__(self):
        self.wallets: dict[str, WalletAccount] = {}
        self._by_agent: dict[str, str] = {}  # stack_id -> wallet_id
        self._by_address: dict[str, str] = {}  # address -> wallet_id

    def save_wallet(self, wallet: WalletAccount) -> str:
        self.wallets[wallet.id] = wallet
        self._by_agent[wallet.stack_id] = wallet.id
        self._by_address[wallet.wallet_address] = wallet.id
        return wallet.id

    def get_wallet(self, wallet_id: str) -> Optional[WalletAccount]:
        return self.wallets.get(wallet_id)

    def get_wallet_by_agent(self, stack_id: str) -> Optional[WalletAccount]:
        wallet_id = self._by_agent.get(stack_id)
        return self.wallets.get(wallet_id) if wallet_id else None

    def get_wallet_by_address(self, address: str) -> Optional[WalletAccount]:
        wallet_id = self._by_address.get(address)
        return self.wallets.get(wallet_id) if wallet_id else None

    def update_wallet_status(
        self,
        wallet_id: str,
        status: WalletStatus,
        owner_eoa: Optional[str] = None,
    ) -> bool:
        wallet = self.wallets.get(wallet_id)
        if not wallet:
            return False
        wallet.status = status.value
        if owner_eoa:
            wallet.owner_eoa = owner_eoa
            wallet.claimed_at = datetime.now(timezone.utc)
        return True

    def update_spending_limits(
        self,
        wallet_id: str,
        per_tx: Optional[float] = None,
        daily: Optional[float] = None,
    ) -> bool:
        wallet = self.wallets.get(wallet_id)
        if not wallet:
            return False
        if per_tx is not None:
            wallet.spending_limit_per_tx = per_tx
        if daily is not None:
            wallet.spending_limit_daily = daily
        return True

    def list_wallets_by_user(self, user_id: str) -> List[WalletAccount]:
        return [w for w in self.wallets.values() if w.user_id == user_id]

    def atomic_claim_wallet(
        self,
        wallet_id: str,
        owner_eoa: str,
    ) -> bool:
        """Atomically claim a wallet if not already claimed."""
        wallet = self.wallets.get(wallet_id)
        if not wallet:
            return False
        if wallet.owner_eoa is not None:
            return False  # Already claimed
        wallet.owner_eoa = owner_eoa
        wallet.status = WalletStatus.ACTIVE.value
        wallet.claimed_at = datetime.now(timezone.utc)
        return True


@pytest.fixture
def storage():
    """Create in-memory storage for testing."""
    return InMemoryWalletStorage()


@pytest.fixture
def config():
    """Create test configuration."""
    return CommerceConfig(
        chain="base-sepolia",
        spending_limit_per_tx=100.0,
        spending_limit_daily=1000.0,
    )


@pytest.fixture
def service(storage, config):
    """Create wallet service for testing."""
    return WalletService(storage=storage, config=config)


class TestWalletCreation:
    """Tests for wallet creation."""

    def test_create_wallet_basic(self, service, storage):
        """Test creating a wallet with minimal parameters."""
        wallet = service.create_wallet(stack_id="agent-123")

        assert wallet.stack_id == "agent-123"
        assert wallet.status == "pending_claim"
        assert wallet.wallet_address.startswith("0x")
        assert len(wallet.wallet_address) == 42
        assert wallet.cdp_wallet_id is not None
        assert wallet.created_at is not None

        # Verify saved to storage
        saved = storage.get_wallet(wallet.id)
        assert saved is not None
        assert saved.id == wallet.id

    def test_create_wallet_with_user(self, service, storage):
        """Test creating a wallet with user ID."""
        wallet = service.create_wallet(
            stack_id="agent-456",
            user_id="user-789",
        )

        assert wallet.user_id == "user-789"

        # Should be findable by user
        user_wallets = storage.list_wallets_by_user("user-789")
        assert len(user_wallets) == 1
        assert user_wallets[0].id == wallet.id

    def test_create_wallet_uses_config_limits(self, service, config):
        """Test that wallet inherits config spending limits."""
        wallet = service.create_wallet(stack_id="agent-xyz")

        assert wallet.spending_limit_per_tx == config.spending_limit_per_tx
        assert wallet.spending_limit_daily == config.spending_limit_daily


class TestWalletRetrieval:
    """Tests for wallet retrieval."""

    def test_get_wallet_by_id(self, service):
        """Test getting wallet by ID."""
        created = service.create_wallet(stack_id="agent-1")
        fetched = service.get_wallet(created.id)

        assert fetched.id == created.id
        assert fetched.stack_id == "agent-1"

    def test_get_wallet_not_found(self, service):
        """Test getting non-existent wallet raises error."""
        with pytest.raises(WalletNotFoundError, match="not found"):
            service.get_wallet("nonexistent-id")

    def test_get_wallet_for_agent(self, service):
        """Test getting wallet by agent ID."""
        created = service.create_wallet(stack_id="agent-special")
        fetched = service.get_wallet_for_agent("agent-special")

        assert fetched.id == created.id

    def test_get_wallet_for_agent_not_found(self, service):
        """Test getting wallet for agent without wallet."""
        with pytest.raises(WalletNotFoundError, match="No wallet found"):
            service.get_wallet_for_agent("nonexistent-agent")


class TestWalletClaiming:
    """Tests for wallet claiming."""

    def test_claim_wallet_success(self, service):
        """Test successfully claiming a wallet."""
        wallet = service.create_wallet(stack_id="agent-to-claim")

        claimed = service.claim_wallet(
            wallet_id=wallet.id,
            owner_eoa="0x1234567890123456789012345678901234567890",
            actor_id="agent-to-claim",
        )

        assert claimed.status == "active"
        assert claimed.owner_eoa == "0x1234567890123456789012345678901234567890"
        assert claimed.claimed_at is not None

    def test_claim_already_claimed_wallet(self, service):
        """Test claiming already claimed wallet raises error."""
        wallet = service.create_wallet(stack_id="agent-claimed")
        service.claim_wallet(
            wallet_id=wallet.id,
            owner_eoa="0x1234567890123456789012345678901234567890",
            actor_id="agent-claimed",
        )

        with pytest.raises(WalletServiceError, match="already claimed"):
            service.claim_wallet(
                wallet_id=wallet.id,
                owner_eoa="0x0987654321098765432109876543210987654321",
                actor_id="agent-claimed",
            )

    def test_claim_with_invalid_eoa(self, service):
        """Test claiming with invalid EOA format."""
        wallet = service.create_wallet(stack_id="agent-bad-eoa")

        with pytest.raises(WalletServiceError, match="Invalid EOA"):
            service.claim_wallet(
                wallet_id=wallet.id,
                owner_eoa="not-an-address",
                actor_id="agent-bad-eoa",
            )


class TestWalletBalance:
    """Tests for balance checking."""

    def test_get_balance_returns_structure(self, service):
        """Test that get_balance returns proper structure."""
        wallet = service.create_wallet(stack_id="agent-balance")

        balance = service.get_balance(wallet.id)

        assert isinstance(balance, WalletBalance)
        assert balance.wallet_address == wallet.wallet_address
        assert balance.usdc_balance == Decimal("0.00")  # Stub returns 0
        assert balance.chain == "base-sepolia"
        assert balance.as_of is not None

    def test_get_balance_for_agent(self, service):
        """Test getting balance by agent ID."""
        wallet = service.create_wallet(stack_id="agent-bal")
        balance = service.get_balance_for_agent("agent-bal")

        assert balance.wallet_address == wallet.wallet_address


class TestWalletTransfer:
    """Tests for transfer operations."""

    def test_transfer_inactive_wallet_fails(self, service):
        """Test transfer from inactive wallet fails."""
        wallet = service.create_wallet(stack_id="agent-inactive")
        # Wallet is pending_claim, not active

        with pytest.raises(WalletNotActiveError, match="not active"):
            service.transfer(
                wallet_id=wallet.id,
                to_address="0x1234567890123456789012345678901234567890",
                amount=Decimal("10.00"),
                actor_id="agent-inactive",
            )

    def test_transfer_exceeds_limit_fails(self, service):
        """Test transfer exceeding per-tx limit fails."""
        wallet = service.create_wallet(stack_id="agent-limit")
        # Claim wallet to make it active
        service.claim_wallet(
            wallet_id=wallet.id,
            owner_eoa="0x1234567890123456789012345678901234567890",
            actor_id="agent-limit",
        )

        with pytest.raises(SpendingLimitExceededError, match="exceeds per-tx"):
            service.transfer(
                wallet_id=wallet.id,
                to_address="0x0987654321098765432109876543210987654321",
                amount=Decimal("200.00"),  # Over 100 limit
                actor_id="agent-limit",
            )

    def test_transfer_invalid_address_returns_error(self, service):
        """Test transfer to invalid address returns error result."""
        wallet = service.create_wallet(stack_id="agent-bad-dest")
        service.claim_wallet(
            wallet_id=wallet.id,
            owner_eoa="0x1234567890123456789012345678901234567890",
            actor_id="agent-bad-dest",
        )

        result = service.transfer(
            wallet_id=wallet.id,
            to_address="not-an-address",
            amount=Decimal("10.00"),
            actor_id="agent-bad-dest",
        )

        assert result.success is False
        assert "Invalid destination" in result.error

    def test_transfer_success_stub(self, service):
        """Test successful transfer (stub)."""
        wallet = service.create_wallet(stack_id="agent-transfer")
        service.claim_wallet(
            wallet_id=wallet.id,
            owner_eoa="0x1234567890123456789012345678901234567890",
            actor_id="agent-transfer",
        )

        result = service.transfer(
            wallet_id=wallet.id,
            to_address="0x0987654321098765432109876543210987654321",
            amount=Decimal("50.00"),
            actor_id="agent-transfer",
        )

        assert isinstance(result, TransferResult)
        assert result.success is True
        assert result.tx_hash is not None  # Stub returns placeholder


class TestWalletLifecycle:
    """Tests for wallet lifecycle operations."""

    def test_pause_wallet(self, service):
        """Test pausing an active wallet."""
        owner_eoa = "0x1234567890123456789012345678901234567890"
        wallet = service.create_wallet(stack_id="agent-pause")
        service.claim_wallet(
            wallet_id=wallet.id,
            owner_eoa=owner_eoa,
            actor_id="agent-pause",
        )

        # Only owner can pause
        paused = service.pause_wallet(wallet.id, actor_id=owner_eoa)

        assert paused.status == "paused"
        assert not paused.can_transact

    def test_resume_wallet(self, service):
        """Test resuming a paused wallet."""
        owner_eoa = "0x1234567890123456789012345678901234567890"
        wallet = service.create_wallet(stack_id="agent-resume")
        service.claim_wallet(
            wallet_id=wallet.id,
            owner_eoa=owner_eoa,
            actor_id="agent-resume",
        )
        service.pause_wallet(wallet.id, actor_id=owner_eoa)

        # Only owner can resume
        resumed = service.resume_wallet(wallet.id, actor_id=owner_eoa)

        assert resumed.status == "active"
        assert resumed.can_transact

    def test_resume_non_paused_fails(self, service):
        """Test resuming non-paused wallet fails."""
        owner_eoa = "0x1234567890123456789012345678901234567890"
        wallet = service.create_wallet(stack_id="agent-not-paused")
        service.claim_wallet(
            wallet_id=wallet.id,
            owner_eoa=owner_eoa,
            actor_id="agent-not-paused",
        )

        with pytest.raises(WalletServiceError, match="Cannot resume"):
            service.resume_wallet(wallet.id, actor_id=owner_eoa)

    def test_update_spending_limits(self, service, storage):
        """Test updating spending limits."""
        owner_eoa = "0x1234567890123456789012345678901234567890"
        wallet = service.create_wallet(stack_id="agent-limits")
        service.claim_wallet(wallet_id=wallet.id, owner_eoa=owner_eoa, actor_id="agent-limits")

        # Only owner can update spending limits
        updated = service.update_spending_limits(
            wallet_id=wallet.id,
            per_tx=200.0,
            daily=2000.0,
            actor_id=owner_eoa,
        )

        assert updated.spending_limit_per_tx == 200.0
        assert updated.spending_limit_daily == 2000.0

    def test_update_spending_limits_invalid(self, service):
        """Test updating with invalid limits fails."""
        owner_eoa = "0x1234567890123456789012345678901234567890"
        wallet = service.create_wallet(stack_id="agent-bad-limits")
        service.claim_wallet(wallet_id=wallet.id, owner_eoa=owner_eoa, actor_id="agent-bad-limits")

        with pytest.raises(WalletServiceError, match="must be positive"):
            service.update_spending_limits(
                wallet_id=wallet.id,
                per_tx=-10.0,
                actor_id=owner_eoa,
            )


class TestDailySpendingLimits:
    """Tests for daily spending limit enforcement."""

    def test_daily_limit_enforced(self, service, config):
        """Test that daily spending limit is enforced."""
        wallet = service.create_wallet(stack_id="agent-daily")
        service.claim_wallet(
            wallet_id=wallet.id,
            owner_eoa="0x1234567890123456789012345678901234567890",
            actor_id="agent-daily",
        )

        # Set low daily limit for testing
        service.update_spending_limits(
            wallet_id=wallet.id,
            daily=100.0,  # $100 daily limit
            actor_id=wallet.owner_eoa,
        )

        # First transfer should succeed
        result1 = service.transfer(
            wallet_id=wallet.id,
            to_address="0x0987654321098765432109876543210987654321",
            amount=Decimal("50.00"),
            actor_id="agent-daily",
        )
        assert result1.success is True

        # Second transfer within limit should also succeed
        result2 = service.transfer(
            wallet_id=wallet.id,
            to_address="0x0987654321098765432109876543210987654321",
            amount=Decimal("40.00"),
            actor_id="agent-daily",
        )
        assert result2.success is True

        # Third transfer exceeding daily limit should fail
        with pytest.raises(SpendingLimitExceededError, match="daily limit"):
            service.transfer(
                wallet_id=wallet.id,
                to_address="0x0987654321098765432109876543210987654321",
                amount=Decimal("20.00"),  # Would total $110, over $100 limit
                actor_id="agent-daily",
            )

    def test_daily_limit_exact_boundary(self, service):
        """Test transfer at exact daily limit boundary."""
        wallet = service.create_wallet(stack_id="agent-boundary")
        service.claim_wallet(
            wallet_id=wallet.id,
            owner_eoa="0x1234567890123456789012345678901234567890",
            actor_id="agent-boundary",
        )

        # Set daily limit to exactly match transfer
        service.update_spending_limits(
            wallet_id=wallet.id,
            daily=50.0,
            actor_id=wallet.owner_eoa,
        )

        # Transfer exactly at limit should succeed
        result = service.transfer(
            wallet_id=wallet.id,
            to_address="0x0987654321098765432109876543210987654321",
            amount=Decimal("50.00"),
            actor_id="agent-boundary",
        )
        assert result.success is True

        # Any additional transfer should fail
        with pytest.raises(SpendingLimitExceededError, match="daily limit"):
            service.transfer(
                wallet_id=wallet.id,
                to_address="0x0987654321098765432109876543210987654321",
                amount=Decimal("0.01"),
                actor_id="agent-boundary",
            )

    def test_negative_transfer_rejected(self, service):
        """Test that negative transfer amounts are rejected."""
        wallet = service.create_wallet(stack_id="agent-negative")
        service.claim_wallet(
            wallet_id=wallet.id,
            owner_eoa="0x1234567890123456789012345678901234567890",
            actor_id="agent-negative",
        )

        result = service.transfer(
            wallet_id=wallet.id,
            to_address="0x0987654321098765432109876543210987654321",
            amount=Decimal("-10.00"),
            actor_id="agent-negative",
        )

        assert result.success is False
        assert "positive" in result.error.lower()

    def test_zero_transfer_rejected(self, service):
        """Test that zero transfer amounts are rejected."""
        wallet = service.create_wallet(stack_id="agent-zero")
        service.claim_wallet(
            wallet_id=wallet.id,
            owner_eoa="0x1234567890123456789012345678901234567890",
            actor_id="agent-zero",
        )

        result = service.transfer(
            wallet_id=wallet.id,
            to_address="0x0987654321098765432109876543210987654321",
            amount=Decimal("0"),
            actor_id="agent-zero",
        )

        assert result.success is False
        assert "positive" in result.error.lower()

    def test_concurrent_transfers_thread_safety(self, service):
        """Test that concurrent transfers are handled safely."""
        import threading

        wallet = service.create_wallet(stack_id="agent-concurrent")
        service.claim_wallet(
            wallet_id=wallet.id,
            owner_eoa="0x1234567890123456789012345678901234567890",
            actor_id="agent-concurrent",
        )

        # Set a daily limit that should allow some but not all concurrent transfers
        service.update_spending_limits(
            wallet_id=wallet.id,
            daily=100.0,
            actor_id=wallet.owner_eoa,
        )

        results = []
        errors = []

        def make_transfer():
            try:
                result = service.transfer(
                    wallet_id=wallet.id,
                    to_address="0x0987654321098765432109876543210987654321",
                    amount=Decimal("30.00"),
                    actor_id="agent-concurrent",
                )
                results.append(result)
            except SpendingLimitExceededError as e:
                errors.append(e)

        # Launch 5 concurrent transfers of $30 each (total $150, limit is $100)
        threads = [threading.Thread(target=make_transfer) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have some successes and some failures
        successful = [r for r in results if r.success]
        total_successful_amount = len(successful) * 30

        # At most 3 transfers should succeed ($90)
        # Some may fail due to limit
        assert total_successful_amount <= 100
        assert len(successful) + len(errors) == 5
