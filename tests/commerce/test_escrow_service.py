"""Tests for escrow service.

Note: The escrow service is heavily stubbed as it requires Web3 integration.
These tests verify the interface and stub behavior.
"""

from datetime import datetime, timezone, timedelta
from decimal import Decimal
import pytest

from kernle.commerce.config import CommerceConfig
from kernle.commerce.escrow.service import (
    EscrowService,
    EscrowServiceError,
    EscrowNotFoundError,
    EscrowInfo,
    TransactionResult,
)
from kernle.commerce.escrow.abi import EscrowStatus


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


class TestEscrowDeployment:
    """Tests for escrow deployment."""
    
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
    
    def test_escrow_info_properties(self, service):
        """Test EscrowInfo property methods."""
        deadline = datetime.now(timezone.utc) + timedelta(days=7)
        
        info = service.deploy_escrow(
            job_id="test-job",
            client_address="0x1234567890123456789012345678901234567890",
            amount=Decimal("50.00"),
            deadline=deadline,
        )
        
        assert info.status_name == "created"
        assert info.is_funded is False
        assert info.is_active is False
        assert info.is_terminal is False
    
    def test_escrow_info_to_dict(self, service):
        """Test EscrowInfo serialization."""
        deadline = datetime.now(timezone.utc) + timedelta(days=7)
        
        info = service.deploy_escrow(
            job_id="dict-job",
            client_address="0x1234567890123456789012345678901234567890",
            amount=Decimal("75.00"),
            deadline=deadline,
        )
        
        data = info.to_dict()
        assert data["address"] == info.address
        assert data["job_id"] == "dict-job"
        assert data["status_name"] == "created"
        assert data["amount"] == "75.00"


class TestEscrowOperations:
    """Tests for escrow operations (stubs)."""
    
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


class TestDisputeOperations:
    """Tests for dispute operations (stubs)."""
    
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


class TestBalanceAndAllowance:
    """Tests for balance and allowance checking (stubs)."""
    
    def test_get_usdc_balance_returns_decimal(self, service):
        """Test that get_usdc_balance returns a Decimal."""
        balance = service.get_usdc_balance(
            "0x1234567890123456789012345678901234567890"
        )
        
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


class TestEscrowRetrieval:
    """Tests for escrow retrieval (stubs)."""
    
    def test_get_escrow_raises_not_found(self, service):
        """Test that get_escrow raises EscrowNotFoundError for stub."""
        with pytest.raises(EscrowNotFoundError, match="STUB"):
            service.get_escrow("0x1234567890123456789012345678901234567890")
    
    def test_get_escrow_for_job_raises_not_found(self, service):
        """Test that get_escrow_for_job raises EscrowNotFoundError for stub."""
        with pytest.raises(EscrowNotFoundError, match="STUB"):
            service.get_escrow_for_job("test-job-id")


class TestUnitConversions:
    """Tests for unit conversion helpers."""
    
    def test_usdc_to_wei(self, service):
        """Test USDC to wei conversion."""
        assert service._usdc_to_wei(Decimal("1.00")) == 1_000_000
        assert service._usdc_to_wei(Decimal("100.50")) == 100_500_000
        assert service._usdc_to_wei(Decimal("0.000001")) == 1
    
    def test_wei_to_usdc(self, service):
        """Test wei to USDC conversion."""
        assert service._wei_to_usdc(1_000_000) == Decimal("1.00")
        assert service._wei_to_usdc(100_500_000) == Decimal("100.50")
        assert service._wei_to_usdc(1) == Decimal("0.000001")
