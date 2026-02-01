"""USDC payment verification on EVM chains (Base, Ethereum, etc).

Verifies that a USDC transfer actually occurred on-chain by:
1. Fetching the transaction receipt via JSON-RPC
2. Parsing ERC20 Transfer event logs
3. Validating amount, sender, and recipient
"""

import httpx
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

logger = logging.getLogger(__name__)

# ERC20 Transfer event signature: Transfer(address indexed from, address indexed to, uint256 value)
TRANSFER_EVENT_SIGNATURE = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"

# Chain configurations
CHAIN_CONFIG = {
    "base": {
        "rpc_url": "https://mainnet.base.org",
        "usdc_address": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
        "usdc_decimals": 6,
        "chain_id": 8453,
        "explorer": "https://basescan.org",
    },
    "base_sepolia": {
        "rpc_url": "https://sepolia.base.org", 
        "usdc_address": "0x036CbD53842c5426634e7929541eC2318f3dCF7e",  # Circle's testnet USDC
        "usdc_decimals": 6,
        "chain_id": 84532,
        "explorer": "https://sepolia.basescan.org",
    },
    "ethereum": {
        "rpc_url": "https://eth.llamarpc.com",
        "usdc_address": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        "usdc_decimals": 6,
        "chain_id": 1,
        "explorer": "https://etherscan.io",
    },
}


class PaymentVerificationError(Exception):
    """Raised when payment verification fails."""
    pass


@dataclass
class TransferVerificationResult:
    """Result of verifying a USDC transfer."""
    
    success: bool
    tx_hash: str
    chain: str
    
    # Transfer details (populated if success=True)
    from_address: Optional[str] = None
    to_address: Optional[str] = None
    amount: Optional[Decimal] = None  # Human-readable (e.g., 5.00 for $5 USDC)
    amount_raw: Optional[int] = None  # Raw units (e.g., 5000000 for $5 USDC)
    
    # Block info
    block_number: Optional[int] = None
    block_timestamp: Optional[datetime] = None
    confirmations: Optional[int] = None
    
    # Error info (populated if success=False)
    error: Optional[str] = None
    error_code: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "tx_hash": self.tx_hash,
            "chain": self.chain,
            "from_address": self.from_address,
            "to_address": self.to_address,
            "amount": str(self.amount) if self.amount else None,
            "amount_raw": self.amount_raw,
            "block_number": self.block_number,
            "block_timestamp": self.block_timestamp.isoformat() if self.block_timestamp else None,
            "confirmations": self.confirmations,
            "error": self.error,
            "error_code": self.error_code,
        }


async def _rpc_call(rpc_url: str, method: str, params: list, timeout: float = 30.0) -> dict:
    """Make a JSON-RPC call to an EVM node."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            rpc_url,
            json={
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
                "id": 1,
            },
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        result = response.json()
        
        if "error" in result:
            raise PaymentVerificationError(f"RPC error: {result['error']}")
        
        return result.get("result")


def _normalize_address(address: str) -> str:
    """Normalize an Ethereum address to lowercase with 0x prefix."""
    if not address:
        return ""
    address = address.lower()
    if not address.startswith("0x"):
        address = "0x" + address
    # Pad to 42 characters if needed (0x + 40 hex chars)
    if len(address) == 66:  # 32-byte padded address from event log
        address = "0x" + address[-40:]
    return address


def _parse_transfer_log(log: dict, usdc_decimals: int) -> Optional[dict]:
    """Parse an ERC20 Transfer event log.
    
    Transfer event: Transfer(address indexed from, address indexed to, uint256 value)
    - topics[0]: event signature
    - topics[1]: from address (indexed, padded to 32 bytes)
    - topics[2]: to address (indexed, padded to 32 bytes)
    - data: value (uint256)
    """
    topics = log.get("topics", [])
    
    if len(topics) < 3:
        return None
    
    # Verify this is a Transfer event
    if topics[0].lower() != TRANSFER_EVENT_SIGNATURE.lower():
        return None
    
    from_address = _normalize_address(topics[1])
    to_address = _normalize_address(topics[2])
    
    # Parse the value from data field
    data = log.get("data", "0x0")
    amount_raw = int(data, 16)
    amount = Decimal(amount_raw) / Decimal(10 ** usdc_decimals)
    
    return {
        "from_address": from_address,
        "to_address": to_address,
        "amount": amount,
        "amount_raw": amount_raw,
    }


async def verify_usdc_transfer(
    tx_hash: str,
    expected_amount: Optional[Decimal] = None,
    expected_from: Optional[str] = None,
    expected_to: Optional[str] = None,
    chain: str = "base",
    min_confirmations: int = 1,
    tolerance: Decimal = Decimal("0.01"),  # Allow 1 cent tolerance for rounding
) -> TransferVerificationResult:
    """Verify a USDC transfer on-chain.
    
    Args:
        tx_hash: Transaction hash to verify
        expected_amount: Expected USDC amount (human-readable, e.g., 5.00 for $5)
        expected_from: Expected sender address
        expected_to: Expected recipient address  
        chain: Chain to verify on ("base", "base_sepolia", "ethereum")
        min_confirmations: Minimum confirmations required
        tolerance: Amount tolerance for matching (default 1 cent)
    
    Returns:
        TransferVerificationResult with success status and transfer details
    
    Raises:
        PaymentVerificationError: If RPC calls fail
    """
    if chain not in CHAIN_CONFIG:
        return TransferVerificationResult(
            success=False,
            tx_hash=tx_hash,
            chain=chain,
            error=f"Unknown chain: {chain}",
            error_code="UNKNOWN_CHAIN",
        )
    
    config = CHAIN_CONFIG[chain]
    rpc_url = config["rpc_url"]
    usdc_address = _normalize_address(config["usdc_address"])
    usdc_decimals = config["usdc_decimals"]
    
    # Normalize inputs
    tx_hash = tx_hash.lower() if tx_hash.startswith("0x") else "0x" + tx_hash.lower()
    if expected_from:
        expected_from = _normalize_address(expected_from)
    if expected_to:
        expected_to = _normalize_address(expected_to)
    
    try:
        # Get transaction receipt
        receipt = await _rpc_call(rpc_url, "eth_getTransactionReceipt", [tx_hash])
        
        if not receipt:
            return TransferVerificationResult(
                success=False,
                tx_hash=tx_hash,
                chain=chain,
                error="Transaction not found or not yet mined",
                error_code="TX_NOT_FOUND",
            )
        
        # Check transaction status (1 = success, 0 = reverted)
        status = int(receipt.get("status", "0x0"), 16)
        if status != 1:
            return TransferVerificationResult(
                success=False,
                tx_hash=tx_hash,
                chain=chain,
                error="Transaction reverted",
                error_code="TX_REVERTED",
            )
        
        # Get block number and current block for confirmations
        block_number = int(receipt.get("blockNumber", "0x0"), 16)
        current_block = await _rpc_call(rpc_url, "eth_blockNumber", [])
        current_block_num = int(current_block, 16)
        confirmations = current_block_num - block_number
        
        if confirmations < min_confirmations:
            return TransferVerificationResult(
                success=False,
                tx_hash=tx_hash,
                chain=chain,
                block_number=block_number,
                confirmations=confirmations,
                error=f"Insufficient confirmations: {confirmations} < {min_confirmations}",
                error_code="INSUFFICIENT_CONFIRMATIONS",
            )
        
        # Get block timestamp
        block = await _rpc_call(rpc_url, "eth_getBlockByNumber", [hex(block_number), False])
        block_timestamp = None
        if block and "timestamp" in block:
            timestamp_int = int(block["timestamp"], 16)
            block_timestamp = datetime.fromtimestamp(timestamp_int, tz=timezone.utc)
        
        # Find USDC Transfer events in logs
        logs = receipt.get("logs", [])
        usdc_transfers = []
        
        for log in logs:
            log_address = _normalize_address(log.get("address", ""))
            if log_address != usdc_address:
                continue
            
            transfer = _parse_transfer_log(log, usdc_decimals)
            if transfer:
                usdc_transfers.append(transfer)
        
        if not usdc_transfers:
            return TransferVerificationResult(
                success=False,
                tx_hash=tx_hash,
                chain=chain,
                block_number=block_number,
                block_timestamp=block_timestamp,
                confirmations=confirmations,
                error="No USDC transfer found in transaction",
                error_code="NO_USDC_TRANSFER",
            )
        
        # Find matching transfer
        for transfer in usdc_transfers:
            # Check from address
            if expected_from and transfer["from_address"] != expected_from:
                continue
            
            # Check to address
            if expected_to and transfer["to_address"] != expected_to:
                continue
            
            # Check amount (with tolerance)
            if expected_amount is not None:
                diff = abs(transfer["amount"] - expected_amount)
                if diff > tolerance:
                    continue
            
            # Found matching transfer!
            return TransferVerificationResult(
                success=True,
                tx_hash=tx_hash,
                chain=chain,
                from_address=transfer["from_address"],
                to_address=transfer["to_address"],
                amount=transfer["amount"],
                amount_raw=transfer["amount_raw"],
                block_number=block_number,
                block_timestamp=block_timestamp,
                confirmations=confirmations,
            )
        
        # No matching transfer found
        # Return details of what we found for debugging
        found_transfer = usdc_transfers[0] if usdc_transfers else None
        error_details = []
        if expected_from and found_transfer and found_transfer["from_address"] != expected_from:
            error_details.append(f"from mismatch: expected {expected_from}, got {found_transfer['from_address']}")
        if expected_to and found_transfer and found_transfer["to_address"] != expected_to:
            error_details.append(f"to mismatch: expected {expected_to}, got {found_transfer['to_address']}")
        if expected_amount and found_transfer:
            error_details.append(f"amount: expected {expected_amount}, got {found_transfer['amount']}")
        
        return TransferVerificationResult(
            success=False,
            tx_hash=tx_hash,
            chain=chain,
            from_address=found_transfer["from_address"] if found_transfer else None,
            to_address=found_transfer["to_address"] if found_transfer else None,
            amount=found_transfer["amount"] if found_transfer else None,
            amount_raw=found_transfer["amount_raw"] if found_transfer else None,
            block_number=block_number,
            block_timestamp=block_timestamp,
            confirmations=confirmations,
            error=f"Transfer found but doesn't match: {'; '.join(error_details)}",
            error_code="TRANSFER_MISMATCH",
        )
        
    except httpx.HTTPError as e:
        logger.error(f"HTTP error verifying transfer {tx_hash}: {e}")
        return TransferVerificationResult(
            success=False,
            tx_hash=tx_hash,
            chain=chain,
            error=f"Network error: {str(e)}",
            error_code="NETWORK_ERROR",
        )
    except PaymentVerificationError as e:
        logger.error(f"Verification error for {tx_hash}: {e}")
        return TransferVerificationResult(
            success=False,
            tx_hash=tx_hash,
            chain=chain,
            error=str(e),
            error_code="VERIFICATION_ERROR",
        )
    except Exception as e:
        logger.exception(f"Unexpected error verifying transfer {tx_hash}")
        return TransferVerificationResult(
            success=False,
            tx_hash=tx_hash,
            chain=chain,
            error=f"Unexpected error: {str(e)}",
            error_code="UNEXPECTED_ERROR",
        )


# Synchronous wrapper for non-async contexts
def verify_usdc_transfer_sync(
    tx_hash: str,
    expected_amount: Optional[Decimal] = None,
    expected_from: Optional[str] = None,
    expected_to: Optional[str] = None,
    chain: str = "base",
    min_confirmations: int = 1,
    tolerance: Decimal = Decimal("0.01"),
) -> TransferVerificationResult:
    """Synchronous wrapper for verify_usdc_transfer."""
    import asyncio
    return asyncio.run(verify_usdc_transfer(
        tx_hash=tx_hash,
        expected_amount=expected_amount,
        expected_from=expected_from,
        expected_to=expected_to,
        chain=chain,
        min_confirmations=min_confirmations,
        tolerance=tolerance,
    ))
