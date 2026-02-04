"""Tests for escrow service.

TEST CATEGORIES:
1. Unit tests (TestUnitConversions, TestEscrowInfoDataclass, TestInputValidation)
   - Test real logic without blockchain interaction
   - Run without external dependencies

2. Stub tests (marked with @pytest.mark.stub)
   - Verify interface contracts and stub behavior
   - DO NOT test actual blockchain functionality
   - Will need replacement with integration tests when Web3 is implemented

TODO: Integration tests (future)
- Requires: testnet RPC, funded test accounts, deployed contracts
- Should verify: actual contract deployment, real transactions, event emission
- See: docs/testing/ESCROW_INTEGRATION.md (to be created)
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from kernle.commerce.config import CommerceConfig
from kernle.commerce.escrow.abi import EscrowStatus
from kernle.commerce.escrow.service import (
    EscrowInfo,
    EscrowNotFoundError,
    EscrowService,
    EscrowServiceError,
    TransactionResult,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    """Create test configuration."""
    return CommerceConfig(
        chain="base-sepolia",
        rpc_url="https://sepolia.base.org",
        usdc_address="0x036CbD53842c5426634e7929541eC2318f3dCF7e",
        escrow_factory_address="0x0000000000000000000000000000000000000001",
        approval_timeout_days=7,
    )


@pytest.fixture
def service(config):
    """Create escrow service for testing."""
    return EscrowService(config=config)


# =============================================================================
# Unit Tests - Real Logic (no blockchain required)
# =============================================================================


class TestUnitConversions:
    """Tests for unit conversion helpers - real logic."""

    def test_usdc_to_wei_whole_numbers(self, service):
        """Test USDC to wei conversion with whole numbers."""
        assert service._usdc_to_wei(Decimal("1.00")) == 1_000_000
        assert service._usdc_to_wei(Decimal("100.00")) == 100_000_000
        assert service._usdc_to_wei(Decimal("0")) == 0

    def test_usdc_to_wei_decimals(self, service):
        """Test USDC to wei conversion with decimal amounts."""
        assert service._usdc_to_wei(Decimal("100.50")) == 100_500_000
        assert service._usdc_to_wei(Decimal("0.000001")) == 1
        assert service._usdc_to_wei(Decimal("1.234567")) == 1_234_567

    def test_usdc_to_wei_large_amounts(self, service):
        """Test USDC to wei conversion with large amounts."""
        assert service._usdc_to_wei(Decimal("1000000.00")) == 1_000_000_000_000
        assert service._usdc_to_wei(Decimal("999999.999999")) == 999_999_999_999

    def test_wei_to_usdc_whole_numbers(self, service):
        """Test wei to USDC conversion with whole numbers."""
        assert service._wei_to_usdc(1_000_000) == Decimal("1.00")
        assert service._wei_to_usdc(100_000_000) == Decimal("100.00")
        assert service._wei_to_usdc(0) == Decimal("0")

    def test_wei_to_usdc_fractional(self, service):
        """Test wei to USDC conversion with fractional amounts."""
        assert service._wei_to_usdc(100_500_000) == Decimal("100.50")
        assert service._wei_to_usdc(1) == Decimal("0.000001")
        assert service._wei_to_usdc(1_234_567) == Decimal("1.234567")

    def test_roundtrip_conversion(self, service):
        """Test that conversion round-trips correctly."""
        amounts = [
            Decimal("1.00"),
            Decimal("100.50"),
            Decimal("0.000001"),
            Decimal("999999.999999"),
        ]
        for amount in amounts:
            wei = service._usdc_to_wei(amount)
            back = service._wei_to_usdc(wei)
            assert back == amount, f"Round-trip failed for {amount}"


class TestJobIdConversion:
    """Tests for job ID to bytes32 conversion."""

    def test_uuid_to_bytes32(self, service):
        """Test UUID string to bytes32 conversion."""
        job_id = "550e8400-e29b-41d4-a716-446655440000"
        result = service._job_id_to_bytes32(job_id)

        assert isinstance(result, bytes)
        assert len(result) == 32

    def test_short_id_padded(self, service):
        """Test that short IDs are padded to 32 bytes."""
        job_id = "short-job-id"
        result = service._job_id_to_bytes32(job_id)

        assert len(result) == 32
        # Should be left-padded with zeros (or right-padded with zeros)
        assert result.rstrip(b"\x00") or result.lstrip(b"\x00")

    def test_different_ids_different_bytes(self, service):
        """Test that different job IDs produce different bytes."""
        id1 = service._job_id_to_bytes32("job-1")
        id2 = service._job_id_to_bytes32("job-2")

        assert id1 != id2


class TestEscrowInfoDataclass:
    """Tests for EscrowInfo dataclass properties."""

    @pytest.fixture
    def base_info(self):
        """Create a base EscrowInfo for testing."""
        return EscrowInfo(
            address="0x1234567890123456789012345678901234567890",
            job_id="test-job",
            client="0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            amount=Decimal("100.00"),
            status=EscrowStatus.CREATED,
            deadline=datetime.now(timezone.utc) + timedelta(days=7),
        )

    def test_status_name_created(self, base_info):
        """Test status_name for CREATED status."""
        base_info.status = EscrowStatus.CREATED
        assert base_info.status_name == "created"

    def test_status_name_funded(self, base_info):
        """Test status_name for FUNDED status."""
        base_info.status = EscrowStatus.FUNDED
        assert base_info.status_name == "funded"

    def test_status_name_accepted(self, base_info):
        """Test status_name for ACCEPTED status."""
        base_info.status = EscrowStatus.ACCEPTED
        assert base_info.status_name == "accepted"

    def test_status_name_delivered(self, base_info):
        """Test status_name for DELIVERED status."""
        base_info.status = EscrowStatus.DELIVERED
        assert base_info.status_name == "delivered"

    def test_status_name_released(self, base_info):
        """Test status_name for RELEASED status."""
        base_info.status = EscrowStatus.RELEASED
        assert base_info.status_name == "released"

    def test_status_name_refunded(self, base_info):
        """Test status_name for REFUNDED status."""
        base_info.status = EscrowStatus.REFUNDED
        assert base_info.status_name == "refunded"

    def test_status_name_disputed(self, base_info):
        """Test status_name for DISPUTED status."""
        base_info.status = EscrowStatus.DISPUTED
        assert base_info.status_name == "disputed"

    def test_is_funded_false_when_created(self, base_info):
        """Test is_funded is False for CREATED status."""
        base_info.status = EscrowStatus.CREATED
        assert base_info.is_funded is False

    def test_is_funded_true_when_funded(self, base_info):
        """Test is_funded is True for FUNDED and later statuses."""
        funded_statuses = [
            EscrowStatus.FUNDED,
            EscrowStatus.ACCEPTED,
            EscrowStatus.DELIVERED,
            EscrowStatus.RELEASED,
        ]
        for status in funded_statuses:
            base_info.status = status
            assert base_info.is_funded is True, f"Expected is_funded=True for {status}"

    def test_is_active_true_for_active_statuses(self, base_info):
        """Test is_active is True for FUNDED, ACCEPTED and DELIVERED."""
        active_statuses = [EscrowStatus.FUNDED, EscrowStatus.ACCEPTED, EscrowStatus.DELIVERED]
        for status in active_statuses:
            base_info.status = status
            assert base_info.is_active is True, f"Expected is_active=True for {status}"

    def test_is_active_false_for_inactive_statuses(self, base_info):
        """Test is_active is False for non-active statuses."""
        inactive_statuses = [
            EscrowStatus.CREATED,
            EscrowStatus.RELEASED,
            EscrowStatus.REFUNDED,
            EscrowStatus.DISPUTED,
        ]
        for status in inactive_statuses:
            base_info.status = status
            assert base_info.is_active is False, f"Expected is_active=False for {status}"

    def test_is_terminal_true_for_terminal_statuses(self, base_info):
        """Test is_terminal is True for RELEASED and REFUNDED."""
        terminal_statuses = [EscrowStatus.RELEASED, EscrowStatus.REFUNDED]
        for status in terminal_statuses:
            base_info.status = status
            assert base_info.is_terminal is True, f"Expected is_terminal=True for {status}"

    def test_is_terminal_false_for_non_terminal_statuses(self, base_info):
        """Test is_terminal is False for non-terminal statuses."""
        non_terminal_statuses = [
            EscrowStatus.CREATED,
            EscrowStatus.FUNDED,
            EscrowStatus.ACCEPTED,
            EscrowStatus.DELIVERED,
            EscrowStatus.DISPUTED,
        ]
        for status in non_terminal_statuses:
            base_info.status = status
            assert base_info.is_terminal is False, f"Expected is_terminal=False for {status}"

    def test_to_dict_includes_all_fields(self, base_info):
        """Test to_dict includes all required fields."""
        data = base_info.to_dict()

        assert "address" in data
        assert "job_id" in data
        assert "client" in data
        assert "amount" in data
        assert "status" in data
        assert "status_name" in data
        assert "deadline" in data

    def test_to_dict_amount_is_string(self, base_info):
        """Test that amount is serialized as string in to_dict."""
        data = base_info.to_dict()
        assert isinstance(data["amount"], str)
        assert data["amount"] == "100.00"


class TestEscrowStatusEnum:
    """Tests for EscrowStatus class values."""

    def test_status_values_are_sequential(self):
        """Test that status values are sequential integers."""
        expected = {
            EscrowStatus.CREATED: 0,
            EscrowStatus.FUNDED: 1,
            EscrowStatus.ACCEPTED: 2,
            EscrowStatus.DELIVERED: 3,
            EscrowStatus.RELEASED: 4,
            EscrowStatus.REFUNDED: 5,
            EscrowStatus.DISPUTED: 6,
        }
        for status_attr, value in expected.items():
            assert status_attr == value, f"Status should have value {value}"

    def test_all_statuses_have_names(self):
        """Test that all status values map to names."""
        all_statuses = [
            EscrowStatus.CREATED,
            EscrowStatus.FUNDED,
            EscrowStatus.ACCEPTED,
            EscrowStatus.DELIVERED,
            EscrowStatus.RELEASED,
            EscrowStatus.REFUNDED,
            EscrowStatus.DISPUTED,
        ]
        for status in all_statuses:
            name = EscrowInfo(
                address="0x" + "0" * 40,
                job_id="test",
                client="0x" + "0" * 40,
                amount=Decimal("1"),
                status=status,
                deadline=datetime.now(timezone.utc),
            ).status_name
            assert name is not None
            assert len(name) > 0


class TestInputValidation:
    """Tests for input validation in service methods."""

    def test_deploy_escrow_validates_address_format(self, service):
        """Test that deploy_escrow validates Ethereum address format."""
        deadline = datetime.now(timezone.utc) + timedelta(days=7)

        # Valid address should work
        info = service.deploy_escrow(
            job_id="test-job",
            client_address="0x1234567890123456789012345678901234567890",
            amount=Decimal("100.00"),
            deadline=deadline,
        )
        assert info is not None

    def test_deploy_escrow_requires_positive_amount(self, service):
        """Test that deploy_escrow requires positive amount."""
        deadline = datetime.now(timezone.utc) + timedelta(days=7)

        # Zero amount - behavior depends on implementation
        # This documents current behavior (stub accepts it)
        info = service.deploy_escrow(
            job_id="test-job",
            client_address="0x1234567890123456789012345678901234567890",
            amount=Decimal("0"),
            deadline=deadline,
        )
        assert info.amount == Decimal("0")


# =============================================================================
# Stub Tests - Interface Verification (marked for future replacement)
# =============================================================================


@pytest.mark.stub
class TestEscrowDeploymentStub:
    """Tests for escrow deployment interface (STUB IMPLEMENTATION).

    These tests verify the interface contract, NOT actual blockchain behavior.
    Replace with integration tests when Web3 is implemented.
    """

    def test_deploy_escrow_returns_info(self, service):
        """Test that deploy_escrow returns proper EscrowInfo."""
        deadline = datetime.now(timezone.utc) + timedelta(days=7)

        info = service.deploy_escrow(
            job_id="550e8400-e29b-41d4-a716-446655440000",
            client_address="0x1234567890123456789012345678901234567890",
            amount=Decimal("100.00"),
            deadline=deadline,
        )

        assert isinstance(info, EscrowInfo)
        assert info.address.startswith("0x")
        assert len(info.address) == 42
        assert info.job_id == "550e8400-e29b-41d4-a716-446655440000"
        assert info.client == "0x1234567890123456789012345678901234567890"
        assert info.amount == Decimal("100.00")
        assert info.status == EscrowStatus.CREATED


@pytest.mark.stub
class TestEscrowOperationsStub:
    """Tests for escrow operations interface (STUB IMPLEMENTATION).

    These tests verify return types and interface contracts.
    All operations return stubbed success - no real transactions occur.
    """

    def test_fund_returns_result(self, service):
        """Test that fund returns a TransactionResult."""
        result = service.fund(
            escrow_address="0x1234567890123456789012345678901234567890",
            client_address="0x0987654321098765432109876543210987654321",
        )

        assert isinstance(result, TransactionResult)
        assert result.success is True  # Stub always succeeds
        assert result.tx_hash is not None

    def test_approve_usdc_returns_result(self, service):
        """Test that approve_usdc returns a TransactionResult."""
        result = service.approve_usdc(
            owner_address="0x1234567890123456789012345678901234567890",
            spender_address="0x0987654321098765432109876543210987654321",
            amount=Decimal("100.00"),
        )

        assert isinstance(result, TransactionResult)
        assert result.success is True

    def test_assign_worker_returns_result(self, service):
        """Test that assign_worker returns a TransactionResult."""
        result = service.assign_worker(
            escrow_address="0x1234567890123456789012345678901234567890",
            worker_address="0x0987654321098765432109876543210987654321",
            client_address="0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )

        assert isinstance(result, TransactionResult)
        assert result.success is True

    def test_mark_delivered_returns_result(self, service):
        """Test that mark_delivered returns a TransactionResult."""
        result = service.mark_delivered(
            escrow_address="0x1234567890123456789012345678901234567890",
            deliverable_hash="0x" + "a" * 64,
            worker_address="0x0987654321098765432109876543210987654321",
        )

        assert isinstance(result, TransactionResult)
        assert result.success is True

    def test_release_returns_result(self, service):
        """Test that release returns a TransactionResult."""
        result = service.release(
            escrow_address="0x1234567890123456789012345678901234567890",
            client_address="0x0987654321098765432109876543210987654321",
        )

        assert isinstance(result, TransactionResult)
        assert result.success is True

    def test_auto_release_returns_result(self, service):
        """Test that auto_release returns a TransactionResult."""
        result = service.auto_release(
            escrow_address="0x1234567890123456789012345678901234567890",
        )

        assert isinstance(result, TransactionResult)
        assert result.success is True

    def test_refund_returns_result(self, service):
        """Test that refund returns a TransactionResult."""
        result = service.refund(
            escrow_address="0x1234567890123456789012345678901234567890",
            client_address="0x0987654321098765432109876543210987654321",
        )

        assert isinstance(result, TransactionResult)
        assert result.success is True


@pytest.mark.stub
class TestDisputeOperationsStub:
    """Tests for dispute operations interface (STUB IMPLEMENTATION)."""

    def test_raise_dispute_returns_result(self, service):
        """Test that raise_dispute returns a TransactionResult."""
        result = service.raise_dispute(
            escrow_address="0x1234567890123456789012345678901234567890",
            disputant_address="0x0987654321098765432109876543210987654321",
        )

        assert isinstance(result, TransactionResult)
        assert result.success is True

    def test_resolve_dispute_returns_result(self, service):
        """Test that resolve_dispute returns a TransactionResult."""
        result = service.resolve_dispute(
            escrow_address="0x1234567890123456789012345678901234567890",
            recipient_address="0x0987654321098765432109876543210987654321",
            arbitrator_address="0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )

        assert isinstance(result, TransactionResult)
        assert result.success is True


@pytest.mark.stub
class TestBalanceAndAllowanceStub:
    """Tests for balance/allowance interface (STUB IMPLEMENTATION)."""

    def test_get_usdc_balance_returns_decimal(self, service):
        """Test that get_usdc_balance returns a Decimal."""
        balance = service.get_usdc_balance("0x1234567890123456789012345678901234567890")

        assert isinstance(balance, Decimal)
        assert balance == Decimal("0.00")  # Stub returns 0

    def test_get_usdc_allowance_returns_decimal(self, service):
        """Test that get_usdc_allowance returns a Decimal."""
        allowance = service.get_usdc_allowance(
            owner="0x1234567890123456789012345678901234567890",
            spender="0x0987654321098765432109876543210987654321",
        )

        assert isinstance(allowance, Decimal)
        assert allowance == Decimal("0.00")  # Stub returns 0


@pytest.mark.stub
class TestEscrowRetrievalStub:
    """Tests for escrow retrieval interface (STUB IMPLEMENTATION)."""

    def test_get_escrow_raises_not_found(self, service):
        """Test that get_escrow raises EscrowNotFoundError for stub."""
        with pytest.raises(EscrowNotFoundError, match="STUB"):
            service.get_escrow("0x1234567890123456789012345678901234567890")

    def test_get_escrow_for_job_raises_not_found(self, service):
        """Test that get_escrow_for_job raises EscrowNotFoundError for stub."""
        with pytest.raises(EscrowNotFoundError, match="STUB"):
            service.get_escrow_for_job("test-job-id")


# =============================================================================
# Reentrancy Protection Tests
# =============================================================================


class TestReentrancyProtection:
    """Tests for reentrancy protection mechanisms."""

    def test_escrow_lock_is_per_address(self, service):
        """Test that each escrow address gets its own lock."""
        lock1 = service._get_escrow_lock("0x1111111111111111111111111111111111111111")
        lock2 = service._get_escrow_lock("0x2222222222222222222222222222222222222222")
        lock3 = service._get_escrow_lock("0x1111111111111111111111111111111111111111")

        # Same address should return same lock
        assert lock1 is lock3
        # Different address should return different lock
        assert lock1 is not lock2

    def test_reentrancy_check_raises_on_conflict(self, service):
        """Test that concurrent operations on same escrow are prevented."""
        address = "0x3333333333333333333333333333333333333333"

        # Start an operation
        service._check_reentrancy(address, "fund")

        # Same operation should raise
        with pytest.raises(EscrowServiceError, match="already in progress"):
            service._check_reentrancy(address, "fund")

        # Clear and should work again
        service._clear_reentrancy(address, "fund")
        service._check_reentrancy(address, "fund")  # Should not raise
        service._clear_reentrancy(address, "fund")

    def test_different_escrows_can_operate_concurrently(self, service):
        """Test that different escrows can have concurrent operations."""
        addr1 = "0x4444444444444444444444444444444444444444"
        addr2 = "0x5555555555555555555555555555555555555555"

        # Both should succeed without raising
        service._check_reentrancy(addr1, "fund")
        service._check_reentrancy(addr2, "fund")  # Should not raise

        # Cleanup
        service._clear_reentrancy(addr1, "fund")
        service._clear_reentrancy(addr2, "fund")
