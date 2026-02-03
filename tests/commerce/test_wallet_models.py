"""Tests for wallet data models."""

from datetime import datetime, timezone

import pytest

from kernle.commerce.wallet.models import WalletAccount, WalletStatus


class TestWalletAccount:
    """Tests for WalletAccount dataclass."""

    def test_create_basic_wallet(self):
        """Test creating a wallet with minimal required fields."""
        wallet = WalletAccount(
            id="wallet-123",
            agent_id="agent-456",
            wallet_address="0x1234567890abcdef1234567890abcdef12345678",
        )

        assert wallet.id == "wallet-123"
        assert wallet.agent_id == "agent-456"
        assert wallet.wallet_address == "0x1234567890abcdef1234567890abcdef12345678"
        assert wallet.chain == "base"
        assert wallet.status == "pending_claim"
        assert wallet.spending_limit_per_tx == 100.0
        assert wallet.spending_limit_daily == 1000.0

    def test_create_wallet_with_all_fields(self):
        """Test creating a wallet with all fields."""
        now = datetime.now(timezone.utc)
        wallet = WalletAccount(
            id="wallet-123",
            agent_id="agent-456",
            wallet_address="0x1234567890abcdef1234567890abcdef12345678",
            chain="base-sepolia",
            status="active",
            user_id="usr_abc123",
            owner_eoa="0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
            spending_limit_per_tx=50.0,
            spending_limit_daily=500.0,
            cdp_wallet_id="cdp-wallet-xyz",
            created_at=now,
            claimed_at=now,
        )

        assert wallet.chain == "base-sepolia"
        assert wallet.status == "active"
        assert wallet.user_id == "usr_abc123"
        assert wallet.owner_eoa == "0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
        assert wallet.spending_limit_per_tx == 50.0
        assert wallet.spending_limit_daily == 500.0
        assert wallet.cdp_wallet_id == "cdp-wallet-xyz"
        assert wallet.created_at == now
        assert wallet.claimed_at == now

    def test_invalid_wallet_address(self):
        """Test that invalid wallet addresses are rejected."""
        with pytest.raises(ValueError, match="wallet_address"):
            WalletAccount(
                id="wallet-123",
                agent_id="agent-456",
                wallet_address="not-a-valid-address",
            )

    def test_invalid_chain(self):
        """Test that invalid chains are rejected."""
        with pytest.raises(ValueError, match="Invalid chain"):
            WalletAccount(
                id="wallet-123",
                agent_id="agent-456",
                wallet_address="0x1234567890abcdef1234567890abcdef12345678",
                chain="ethereum",
            )

    def test_invalid_status(self):
        """Test that invalid statuses are rejected."""
        with pytest.raises(ValueError, match="Invalid status"):
            WalletAccount(
                id="wallet-123",
                agent_id="agent-456",
                wallet_address="0x1234567890abcdef1234567890abcdef12345678",
                status="invalid_status",
            )

    def test_status_enum_value(self):
        """Test that status can be set via enum."""
        wallet = WalletAccount(
            id="wallet-123",
            agent_id="agent-456",
            wallet_address="0x1234567890abcdef1234567890abcdef12345678",
            status=WalletStatus.ACTIVE,
        )

        assert wallet.status == "active"

    def test_is_active(self):
        """Test is_active property."""
        wallet = WalletAccount(
            id="wallet-123",
            agent_id="agent-456",
            wallet_address="0x1234567890abcdef1234567890abcdef12345678",
            status="active",
        )
        assert wallet.is_active is True

        wallet2 = WalletAccount(
            id="wallet-124",
            agent_id="agent-456",
            wallet_address="0x1234567890abcdef1234567890abcdef12345679",
            status="pending_claim",
        )
        assert wallet2.is_active is False

    def test_is_claimed(self):
        """Test is_claimed property."""
        # Not claimed
        wallet = WalletAccount(
            id="wallet-123",
            agent_id="agent-456",
            wallet_address="0x1234567890abcdef1234567890abcdef12345678",
        )
        assert wallet.is_claimed is False

        # Claimed via owner_eoa
        wallet2 = WalletAccount(
            id="wallet-124",
            agent_id="agent-456",
            wallet_address="0x1234567890abcdef1234567890abcdef12345679",
            owner_eoa="0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        )
        assert wallet2.is_claimed is True

        # Claimed via claimed_at
        wallet3 = WalletAccount(
            id="wallet-125",
            agent_id="agent-456",
            wallet_address="0x1234567890abcdef1234567890abcdef12345670",
            claimed_at=datetime.now(timezone.utc),
        )
        assert wallet3.is_claimed is True

    def test_can_transact(self):
        """Test can_transact property."""
        # Active wallet can transact
        wallet = WalletAccount(
            id="wallet-123",
            agent_id="agent-456",
            wallet_address="0x1234567890abcdef1234567890abcdef12345678",
            status="active",
        )
        assert wallet.can_transact is True

        # Paused wallet cannot transact
        wallet2 = WalletAccount(
            id="wallet-124",
            agent_id="agent-456",
            wallet_address="0x1234567890abcdef1234567890abcdef12345679",
            status="paused",
        )
        assert wallet2.can_transact is False

    def test_to_dict(self):
        """Test serialization to dictionary."""
        now = datetime.now(timezone.utc)
        wallet = WalletAccount(
            id="wallet-123",
            agent_id="agent-456",
            wallet_address="0x1234567890abcdef1234567890abcdef12345678",
            chain="base",
            status="active",
            created_at=now,
        )

        d = wallet.to_dict()

        assert d["id"] == "wallet-123"
        assert d["agent_id"] == "agent-456"
        assert d["wallet_address"] == "0x1234567890abcdef1234567890abcdef12345678"
        assert d["chain"] == "base"
        assert d["status"] == "active"
        assert d["created_at"] == now.isoformat()
        assert d["claimed_at"] is None

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "id": "wallet-123",
            "agent_id": "agent-456",
            "wallet_address": "0x1234567890abcdef1234567890abcdef12345678",
            "chain": "base",
            "status": "active",
            "spending_limit_per_tx": 200.0,
            "created_at": "2024-01-15T12:00:00+00:00",
        }

        wallet = WalletAccount.from_dict(data)

        assert wallet.id == "wallet-123"
        assert wallet.agent_id == "agent-456"
        assert wallet.wallet_address == "0x1234567890abcdef1234567890abcdef12345678"
        assert wallet.chain == "base"
        assert wallet.status == "active"
        assert wallet.spending_limit_per_tx == 200.0
        assert wallet.created_at is not None

    def test_from_dict_with_z_suffix(self):
        """Test parsing datetime with Z suffix."""
        data = {
            "id": "wallet-123",
            "agent_id": "agent-456",
            "wallet_address": "0x1234567890abcdef1234567890abcdef12345678",
            "created_at": "2024-01-15T12:00:00Z",
        }

        wallet = WalletAccount.from_dict(data)

        assert wallet.created_at is not None
        assert wallet.created_at.tzinfo is not None


class TestWalletStatus:
    """Tests for WalletStatus enum."""

    def test_all_statuses_defined(self):
        """Test that all expected statuses are defined."""
        statuses = {s.value for s in WalletStatus}
        expected = {"pending_claim", "active", "paused", "frozen"}
        assert statuses == expected

    def test_status_values(self):
        """Test enum string values."""
        assert WalletStatus.PENDING_CLAIM.value == "pending_claim"
        assert WalletStatus.ACTIVE.value == "active"
        assert WalletStatus.PAUSED.value == "paused"
        assert WalletStatus.FROZEN.value == "frozen"
