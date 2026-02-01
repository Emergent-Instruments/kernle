"""Tests for USDC payment verification."""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from app.payments.verification import (
    verify_usdc_transfer,
    TransferVerificationResult,
    PaymentVerificationError,
    CHAIN_CONFIG,
    _normalize_address,
    _parse_transfer_log,
    TRANSFER_EVENT_SIGNATURE,
)


class TestNormalizeAddress:
    """Tests for address normalization."""
    
    def test_lowercase_with_prefix(self):
        assert _normalize_address("0xAbC123") == "0xabc123"
    
    def test_adds_prefix(self):
        assert _normalize_address("abc123") == "0xabc123"
    
    def test_handles_padded_address(self):
        # 32-byte padded address (common in event logs)
        padded = "0x000000000000000000000000833589fcd6edb6e08f4c7c32d4f71b54bda02913"
        assert _normalize_address(padded) == "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"
    
    def test_empty_string(self):
        assert _normalize_address("") == ""
    
    def test_none(self):
        assert _normalize_address(None) == ""


class TestParseTransferLog:
    """Tests for ERC20 Transfer event parsing."""
    
    def test_parse_valid_transfer(self):
        log = {
            "topics": [
                TRANSFER_EVENT_SIGNATURE,
                "0x000000000000000000000000sender0000000000000000000000000000000001",
                "0x000000000000000000000000receiver00000000000000000000000000000002",
            ],
            "data": "0x00000000000000000000000000000000000000000000000000000000004c4b40",  # 5,000,000 = $5.00
        }
        
        result = _parse_transfer_log(log, usdc_decimals=6)
        
        assert result is not None
        assert result["amount"] == Decimal("5")
        assert result["amount_raw"] == 5_000_000
        assert "sender" in result["from_address"]
        assert "receiver" in result["to_address"]
    
    def test_parse_wrong_event_signature(self):
        log = {
            "topics": [
                "0xwrongsignature",
                "0x000000000000000000000000sender0000000000000000000000000000000001",
                "0x000000000000000000000000receiver00000000000000000000000000000002",
            ],
            "data": "0x00000000000000000000000000000000000000000000000000000000004c4b40",
        }
        
        result = _parse_transfer_log(log, usdc_decimals=6)
        assert result is None
    
    def test_parse_insufficient_topics(self):
        log = {
            "topics": [TRANSFER_EVENT_SIGNATURE],
            "data": "0x00000000000000000000000000000000000000000000000000000000004c4b40",
        }
        
        result = _parse_transfer_log(log, usdc_decimals=6)
        assert result is None


class TestChainConfig:
    """Tests for chain configuration."""
    
    def test_base_config_exists(self):
        assert "base" in CHAIN_CONFIG
        config = CHAIN_CONFIG["base"]
        assert "rpc_url" in config
        assert "usdc_address" in config
        assert config["usdc_decimals"] == 6
    
    def test_base_sepolia_config_exists(self):
        assert "base_sepolia" in CHAIN_CONFIG
        config = CHAIN_CONFIG["base_sepolia"]
        assert "usdc_address" in config
    
    def test_ethereum_config_exists(self):
        assert "ethereum" in CHAIN_CONFIG
        config = CHAIN_CONFIG["ethereum"]
        assert config["chain_id"] == 1


class TestVerifyUsdcTransfer:
    """Tests for the main verification function."""
    
    @pytest.mark.asyncio
    async def test_unknown_chain(self):
        result = await verify_usdc_transfer(
            tx_hash="0x123",
            chain="unknown_chain",
        )
        
        assert result.success is False
        assert result.error_code == "UNKNOWN_CHAIN"
    
    @pytest.mark.asyncio
    async def test_tx_not_found(self):
        with patch("app.payments.verification._rpc_call", new_callable=AsyncMock) as mock_rpc:
            mock_rpc.return_value = None  # Transaction not found
            
            result = await verify_usdc_transfer(
                tx_hash="0x123",
                chain="base",
            )
            
            assert result.success is False
            assert result.error_code == "TX_NOT_FOUND"
    
    @pytest.mark.asyncio
    async def test_tx_reverted(self):
        with patch("app.payments.verification._rpc_call", new_callable=AsyncMock) as mock_rpc:
            mock_rpc.return_value = {
                "status": "0x0",  # Reverted
                "blockNumber": "0x100",
                "logs": [],
            }
            
            result = await verify_usdc_transfer(
                tx_hash="0x123",
                chain="base",
            )
            
            assert result.success is False
            assert result.error_code == "TX_REVERTED"
    
    @pytest.mark.asyncio
    async def test_insufficient_confirmations(self):
        with patch("app.payments.verification._rpc_call", new_callable=AsyncMock) as mock_rpc:
            # Return receipt, then current block
            mock_rpc.side_effect = [
                {
                    "status": "0x1",
                    "blockNumber": "0x100",  # 256
                    "logs": [],
                },
                "0x100",  # Current block = same as tx block = 0 confirmations
            ]
            
            result = await verify_usdc_transfer(
                tx_hash="0x123",
                chain="base",
                min_confirmations=5,
            )
            
            assert result.success is False
            assert result.error_code == "INSUFFICIENT_CONFIRMATIONS"
    
    @pytest.mark.asyncio
    async def test_no_usdc_transfer(self):
        with patch("app.payments.verification._rpc_call", new_callable=AsyncMock) as mock_rpc:
            mock_rpc.side_effect = [
                {
                    "status": "0x1",
                    "blockNumber": "0x100",
                    "logs": [],  # No logs
                },
                "0x110",  # 16 confirmations
                {"timestamp": "0x65b5e800"},  # Some timestamp
            ]
            
            result = await verify_usdc_transfer(
                tx_hash="0x123",
                chain="base",
            )
            
            assert result.success is False
            assert result.error_code == "NO_USDC_TRANSFER"
    
    @pytest.mark.asyncio
    async def test_successful_verification(self):
        usdc_address = CHAIN_CONFIG["base"]["usdc_address"]
        
        with patch("app.payments.verification._rpc_call", new_callable=AsyncMock) as mock_rpc:
            mock_rpc.side_effect = [
                {
                    "status": "0x1",
                    "blockNumber": "0x100",
                    "logs": [
                        {
                            "address": usdc_address,
                            "topics": [
                                TRANSFER_EVENT_SIGNATURE,
                                "0x000000000000000000000000aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                                "0x000000000000000000000000bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                            ],
                            "data": "0x00000000000000000000000000000000000000000000000000000000004c4b40",  # $5.00
                        }
                    ],
                },
                "0x110",  # Current block
                {"timestamp": "0x65b5e800"},  # Block timestamp
            ]
            
            result = await verify_usdc_transfer(
                tx_hash="0x123",
                chain="base",
            )
            
            assert result.success is True
            assert result.amount == Decimal("5")
            assert result.block_number == 256
            assert result.confirmations == 16
    
    @pytest.mark.asyncio
    async def test_amount_mismatch(self):
        usdc_address = CHAIN_CONFIG["base"]["usdc_address"]
        
        with patch("app.payments.verification._rpc_call", new_callable=AsyncMock) as mock_rpc:
            mock_rpc.side_effect = [
                {
                    "status": "0x1",
                    "blockNumber": "0x100",
                    "logs": [
                        {
                            "address": usdc_address,
                            "topics": [
                                TRANSFER_EVENT_SIGNATURE,
                                "0x000000000000000000000000aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                                "0x000000000000000000000000bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                            ],
                            "data": "0x00000000000000000000000000000000000000000000000000000000004c4b40",  # $5.00
                        }
                    ],
                },
                "0x110",
                {"timestamp": "0x65b5e800"},
            ]
            
            result = await verify_usdc_transfer(
                tx_hash="0x123",
                expected_amount=Decimal("10.00"),  # Expected $10, got $5
                chain="base",
            )
            
            assert result.success is False
            assert result.error_code == "TRANSFER_MISMATCH"
            assert "amount" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_amount_within_tolerance(self):
        usdc_address = CHAIN_CONFIG["base"]["usdc_address"]
        
        with patch("app.payments.verification._rpc_call", new_callable=AsyncMock) as mock_rpc:
            mock_rpc.side_effect = [
                {
                    "status": "0x1",
                    "blockNumber": "0x100",
                    "logs": [
                        {
                            "address": usdc_address,
                            "topics": [
                                TRANSFER_EVENT_SIGNATURE,
                                "0x000000000000000000000000aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                                "0x000000000000000000000000bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                            ],
                            "data": "0x00000000000000000000000000000000000000000000000000000000004c4f48",  # $5.001 (within 1 cent)
                        }
                    ],
                },
                "0x110",
                {"timestamp": "0x65b5e800"},
            ]
            
            result = await verify_usdc_transfer(
                tx_hash="0x123",
                expected_amount=Decimal("5.00"),
                chain="base",
                tolerance=Decimal("0.01"),
            )
            
            assert result.success is True


class TestTransferVerificationResult:
    """Tests for the result dataclass."""
    
    def test_to_dict_success(self):
        result = TransferVerificationResult(
            success=True,
            tx_hash="0x123",
            chain="base",
            from_address="0xaaa",
            to_address="0xbbb",
            amount=Decimal("5.00"),
            amount_raw=5_000_000,
            block_number=100,
            block_timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            confirmations=10,
        )
        
        d = result.to_dict()
        assert d["success"] is True
        assert d["amount"] == "5.00"
        assert d["block_timestamp"] == "2026-01-01T12:00:00+00:00"
    
    def test_to_dict_failure(self):
        result = TransferVerificationResult(
            success=False,
            tx_hash="0x123",
            chain="base",
            error="Something went wrong",
            error_code="SOME_ERROR",
        )
        
        d = result.to_dict()
        assert d["success"] is False
        assert d["error"] == "Something went wrong"
        assert d["amount"] is None
