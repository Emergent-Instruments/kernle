"""
Commerce configuration settings.

Environment variables for commerce features:
- CDP_API_KEY: Coinbase Developer Platform API key
- CDP_API_SECRET: CDP API secret
- BASE_RPC_URL: Base mainnet RPC endpoint
- BASE_SEPOLIA_RPC_URL: Base Sepolia testnet RPC endpoint
- USDC_ADDRESS: USDC contract address on Base
- ESCROW_FACTORY_ADDRESS: Deployed escrow factory contract
- ARBITRATOR_ADDRESS: Dispute resolution multisig
"""

import os
from dataclasses import dataclass
from typing import Optional


# Chain configurations
CHAIN_BASE = "base"
CHAIN_BASE_SEPOLIA = "base-sepolia"

# USDC contract addresses
USDC_ADDRESSES = {
    CHAIN_BASE: "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
    CHAIN_BASE_SEPOLIA: "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
}

# Default RPC URLs
DEFAULT_RPC_URLS = {
    CHAIN_BASE: "https://mainnet.base.org",
    CHAIN_BASE_SEPOLIA: "https://sepolia.base.org",
}

# Wallet spending limits
DEFAULT_SPENDING_LIMIT_PER_TX = 100.0  # USDC
DEFAULT_SPENDING_LIMIT_DAILY = 1000.0  # USDC

# Job approval timeout
DEFAULT_APPROVAL_TIMEOUT_DAYS = 7


@dataclass
class CommerceConfig:
    """Commerce subsystem configuration."""
    
    # CDP (Coinbase Developer Platform) settings
    cdp_api_key: Optional[str] = None
    cdp_api_secret: Optional[str] = None
    
    # Blockchain settings
    chain: str = CHAIN_BASE_SEPOLIA  # Default to testnet
    rpc_url: Optional[str] = None
    
    # Contract addresses
    usdc_address: Optional[str] = None
    escrow_factory_address: Optional[str] = None
    arbitrator_address: Optional[str] = None
    
    # Wallet defaults
    spending_limit_per_tx: float = DEFAULT_SPENDING_LIMIT_PER_TX
    spending_limit_daily: float = DEFAULT_SPENDING_LIMIT_DAILY
    
    # Job defaults
    approval_timeout_days: int = DEFAULT_APPROVAL_TIMEOUT_DAYS
    
    @classmethod
    def from_env(cls) -> "CommerceConfig":
        """Load configuration from environment variables."""
        chain = os.getenv("KERNLE_COMMERCE_CHAIN", CHAIN_BASE_SEPOLIA)
        
        return cls(
            cdp_api_key=os.getenv("CDP_API_KEY"),
            cdp_api_secret=os.getenv("CDP_API_SECRET"),
            chain=chain,
            rpc_url=os.getenv(
                "BASE_RPC_URL" if chain == CHAIN_BASE else "BASE_SEPOLIA_RPC_URL",
                DEFAULT_RPC_URLS.get(chain),
            ),
            usdc_address=os.getenv("USDC_ADDRESS", USDC_ADDRESSES.get(chain)),
            escrow_factory_address=os.getenv("ESCROW_FACTORY_ADDRESS"),
            arbitrator_address=os.getenv("ARBITRATOR_ADDRESS"),
            spending_limit_per_tx=float(
                os.getenv("KERNLE_SPENDING_LIMIT_PER_TX", DEFAULT_SPENDING_LIMIT_PER_TX)
            ),
            spending_limit_daily=float(
                os.getenv("KERNLE_SPENDING_LIMIT_DAILY", DEFAULT_SPENDING_LIMIT_DAILY)
            ),
            approval_timeout_days=int(
                os.getenv("KERNLE_APPROVAL_TIMEOUT_DAYS", DEFAULT_APPROVAL_TIMEOUT_DAYS)
            ),
        )
    
    @property
    def is_mainnet(self) -> bool:
        """Check if configured for mainnet."""
        return self.chain == CHAIN_BASE
    
    @property
    def has_cdp_credentials(self) -> bool:
        """Check if CDP credentials are configured."""
        return bool(self.cdp_api_key and self.cdp_api_secret)


# Global config instance (lazy-loaded)
_config: Optional[CommerceConfig] = None


def get_config() -> CommerceConfig:
    """Get the global commerce configuration."""
    global _config
    if _config is None:
        _config = CommerceConfig.from_env()
    return _config


def set_config(config: CommerceConfig) -> None:
    """Set the global commerce configuration (for testing)."""
    global _config
    _config = config
