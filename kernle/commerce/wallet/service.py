"""
Wallet service for Kernle Commerce.

Provides wallet management operations including CDP Smart Wallet creation,
balance checking, and transfer methods. Uses Coinbase Developer Platform
(CDP) for custodial wallet operations.

NOTE: CDP integration is stubbed for now. Methods marked with TODO require
actual CDP SDK integration once the cdp-sdk package is added as a dependency.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from decimal import Decimal
from typing import Dict, Optional, Protocol, Tuple
import logging
import threading
import uuid

from kernle.commerce.config import get_config, CommerceConfig
from kernle.commerce.wallet.models import WalletAccount, WalletStatus
from kernle.commerce.wallet.storage import WalletStorage, DailySpendResult


@dataclass
class DailySpendRecord:
    """Tracks daily spending for a wallet (in-memory fallback)."""
    date: date
    total_spent: Decimal = Decimal("0")
    transaction_count: int = 0


logger = logging.getLogger(__name__)


class WalletServiceError(Exception):
    """Base exception for wallet service errors."""
    pass


class WalletNotFoundError(WalletServiceError):
    """Wallet not found."""
    pass


class WalletNotActiveError(WalletServiceError):
    """Wallet is not active and cannot transact."""
    pass


class InsufficientBalanceError(WalletServiceError):
    """Insufficient balance for transaction."""
    pass


class SpendingLimitExceededError(WalletServiceError):
    """Transaction exceeds spending limits."""
    pass


class CDPIntegrationError(WalletServiceError):
    """Error communicating with CDP."""
    pass


class WalletAuthorizationError(WalletServiceError):
    """Actor is not authorized to perform this operation on the wallet."""
    pass


@dataclass
class WalletBalance:
    """Represents wallet balance information.
    
    Attributes:
        wallet_address: The wallet's Ethereum address
        usdc_balance: USDC balance in human-readable units (not wei)
        eth_balance: ETH balance for gas (on Base, typically small)
        chain: The blockchain network
        as_of: Timestamp when balance was checked
    """
    wallet_address: str
    usdc_balance: Decimal
    eth_balance: Decimal
    chain: str
    as_of: datetime


@dataclass
class TransferResult:
    """Result of a transfer operation.
    
    Attributes:
        success: Whether the transfer succeeded
        tx_hash: Transaction hash (if successful)
        error: Error message (if failed)
        from_address: Sender wallet address
        to_address: Recipient address
        amount: Amount transferred in USDC
    """
    success: bool
    from_address: str
    to_address: str
    amount: Decimal
    tx_hash: Optional[str] = None
    error: Optional[str] = None


class WalletService:
    """Service for wallet management and transactions.
    
    Handles wallet lifecycle:
    - Creation: CDP Smart Wallet provisioned at agent registration
    - Claiming: Human owner links their EOA for recovery/control
    - Transactions: Balance checks, transfers within spending limits
    - Lifecycle: Pause, resume, freeze operations
    
    Example:
        >>> service = WalletService(storage)
        >>> wallet = service.create_wallet("agent_123", "usr_456")
        >>> print(wallet.wallet_address)
        0x1234...
        >>> balance = service.get_balance(wallet.id)
        >>> print(f"Balance: {balance.usdc_balance} USDC")
    """
    
    def __init__(
        self,
        storage: WalletStorage,
        config: Optional[CommerceConfig] = None,
    ):
        """Initialize wallet service.
        
        Args:
            storage: Wallet storage backend
            config: Commerce configuration (uses global config if not provided)
        """
        self.storage = storage
        self.config = config or get_config()
        self._cdp_client = None  # TODO: Initialize CDP client
        
        # Daily spending tracking with thread safety
        # Primary: Database-backed via storage.increment_daily_spend()
        # Fallback: In-memory dict with lock (for InMemoryWalletStorage)
        self._daily_spend: Dict[str, DailySpendRecord] = {}
        self._daily_spend_lock = threading.Lock()  # Thread safety for in-memory fallback
        
    def _get_daily_spend(self, wallet_id: str) -> Decimal:
        """Get the current daily spend for a wallet.
        
        Attempts database lookup first, falls back to in-memory.
        Resets at midnight UTC.
        
        Args:
            wallet_id: Wallet ID
            
        Returns:
            Total spent today in USDC
        """
        # Try database first (preferred)
        if hasattr(self.storage, 'get_daily_spend'):
            result = self.storage.get_daily_spend(wallet_id)
            if result is not None:
                return result
        
        # Fallback to in-memory with thread safety
        today = datetime.now(timezone.utc).date()
        
        with self._daily_spend_lock:
            if wallet_id not in self._daily_spend:
                return Decimal("0")
            
            record = self._daily_spend[wallet_id]
            
            # Reset if it's a new day
            if record.date != today:
                return Decimal("0")
            
            return record.total_spent
    
    def _try_atomic_spend(
        self, 
        wallet_id: str, 
        amount: Decimal, 
        daily_limit: float
    ) -> DailySpendResult:
        """Attempt atomic spend increment with limit check.
        
        Tries database atomic operation first, falls back to in-memory.
        
        Args:
            wallet_id: Wallet ID
            amount: Amount to spend in USDC
            daily_limit: Daily spending limit
            
        Returns:
            DailySpendResult with success status and current state
        """
        # Try database atomic increment first (preferred - handles race conditions)
        if hasattr(self.storage, 'increment_daily_spend'):
            result = self.storage.increment_daily_spend(wallet_id, amount)
            if result is not None:
                return result
        
        # Fallback to in-memory with thread safety
        today = datetime.now(timezone.utc).date()
        
        with self._daily_spend_lock:
            # Get or create record with date check
            if wallet_id not in self._daily_spend or self._daily_spend[wallet_id].date != today:
                self._daily_spend[wallet_id] = DailySpendRecord(date=today)
            
            record = self._daily_spend[wallet_id]
            new_total = record.total_spent + amount
            
            # Check limit before committing
            if float(new_total) > daily_limit:
                remaining = Decimal(str(daily_limit)) - record.total_spent
                return DailySpendResult(
                    success=False,
                    daily_spent=record.total_spent,
                    daily_limit=Decimal(str(daily_limit)),
                    remaining=remaining,
                    error=f"Would exceed daily limit. Remaining: {remaining} USDC"
                )
            
            # Commit the spend
            record.total_spent = new_total
            record.transaction_count += 1
            
            return DailySpendResult(
                success=True,
                daily_spent=new_total,
                daily_limit=Decimal(str(daily_limit)),
                remaining=Decimal(str(daily_limit)) - new_total
            )
    
    def _record_spend(self, wallet_id: str, amount: Decimal) -> None:
        """Record a spending transaction for daily limit tracking.
        
        DEPRECATED: Use _try_atomic_spend() for atomic check-and-increment.
        This method is kept for backwards compatibility but should not be
        called in the normal transfer flow.
        
        Args:
            wallet_id: Wallet ID
            amount: Amount spent in USDC
        """
        today = datetime.now(timezone.utc).date()
        
        with self._daily_spend_lock:
            if wallet_id not in self._daily_spend or self._daily_spend[wallet_id].date != today:
                self._daily_spend[wallet_id] = DailySpendRecord(date=today)
            
            record = self._daily_spend[wallet_id]
            record.total_spent += amount
            record.transaction_count += 1
    
    def _is_authorized_actor(
        self,
        wallet: WalletAccount,
        actor_id: str,
        require_owner: bool = False,
    ) -> bool:
        """Check if actor is authorized to operate on this wallet.
        
        Args:
            wallet: The wallet to check authorization for
            actor_id: The actor attempting the operation
            require_owner: If True, only owner_eoa can authorize (not agent)
            
        Returns:
            True if actor is authorized, False otherwise
        """
        # Owner EOA always has full control
        if wallet.owner_eoa and actor_id == wallet.owner_eoa:
            return True
        
        # If owner-only operation, agent cannot authorize
        if require_owner:
            return False
        
        # Agent that owns the wallet can operate it
        if actor_id == wallet.agent_id:
            return True
        
        return False
    
    def _init_cdp_client(self) -> None:
        """Initialize CDP client if credentials are available.
        
        TODO: Implement CDP SDK initialization
        See: https://docs.cdp.coinbase.com/smart-wallet
        """
        if not self.config.has_cdp_credentials:
            logger.warning("CDP credentials not configured - wallet creation will fail")
            return
            
        # TODO: Initialize CDP SDK client
        # from cdp import Cdp
        # self._cdp_client = Cdp(
        #     api_key=self.config.cdp_api_key,
        #     api_secret=self.config.cdp_api_secret,
        # )
        logger.info("CDP client initialization stubbed - implement when SDK is added")
    
    def create_wallet(
        self,
        agent_id: str,
        user_id: Optional[str] = None,
    ) -> WalletAccount:
        """Create a new CDP Smart Wallet for an agent.
        
        This is called during agent registration. The wallet starts in
        'pending_claim' status until the human owner claims it.
        
        Args:
            agent_id: Kernle agent ID this wallet belongs to
            user_id: Optional user ID of the wallet owner
            
        Returns:
            WalletAccount: The newly created wallet
            
        Raises:
            CDPIntegrationError: If CDP wallet creation fails
            
        TODO: Implement actual CDP wallet creation
        """
        logger.info(f"Creating wallet for agent {agent_id}")
        
        # TODO: Call CDP to create Smart Wallet
        # try:
        #     cdp_wallet = self._cdp_client.wallets.create(
        #         network=self.config.chain,
        #     )
        #     wallet_address = cdp_wallet.default_address
        #     cdp_wallet_id = cdp_wallet.id
        # except Exception as e:
        #     raise CDPIntegrationError(f"Failed to create CDP wallet: {e}") from e
        
        # STUB: Generate placeholder address for development
        wallet_id = str(uuid.uuid4())
        # Generate deterministic-looking address from agent_id for testing
        # Ethereum addresses are 40 hex chars after 0x, so we combine two uuid5 hashes
        base_hash = uuid.uuid5(uuid.NAMESPACE_DNS, agent_id).hex
        extra_hash = uuid.uuid5(uuid.NAMESPACE_URL, agent_id).hex[:8]
        addr_hex = (base_hash + extra_hash)[:40]
        wallet_address = f"0x{addr_hex}"
        cdp_wallet_id = f"cdp_{wallet_id[:8]}"
        
        wallet = WalletAccount(
            id=wallet_id,
            agent_id=agent_id,
            wallet_address=wallet_address,
            chain=self.config.chain,
            status=WalletStatus.PENDING_CLAIM.value,
            user_id=user_id,
            spending_limit_per_tx=self.config.spending_limit_per_tx,
            spending_limit_daily=self.config.spending_limit_daily,
            cdp_wallet_id=cdp_wallet_id,
            created_at=datetime.now(timezone.utc),
        )
        
        self.storage.save_wallet(wallet)
        logger.info(f"Created wallet {wallet.wallet_address} for agent {agent_id}")
        
        return wallet
    
    def get_wallet(self, wallet_id: str) -> WalletAccount:
        """Get a wallet by ID.
        
        Args:
            wallet_id: Wallet ID
            
        Returns:
            WalletAccount: The wallet
            
        Raises:
            WalletNotFoundError: If wallet doesn't exist
        """
        wallet = self.storage.get_wallet(wallet_id)
        if not wallet:
            raise WalletNotFoundError(f"Wallet not found: {wallet_id}")
        return wallet
    
    def get_wallet_for_agent(self, agent_id: str) -> WalletAccount:
        """Get the wallet for an agent.
        
        Args:
            agent_id: Kernle agent ID
            
        Returns:
            WalletAccount: The agent's wallet
            
        Raises:
            WalletNotFoundError: If agent has no wallet
        """
        wallet = self.storage.get_wallet_by_agent(agent_id)
        if not wallet:
            raise WalletNotFoundError(f"No wallet found for agent: {agent_id}")
        return wallet
    
    def claim_wallet(
        self,
        wallet_id: str,
        owner_eoa: str,
    ) -> WalletAccount:
        """Claim a wallet by setting the owner EOA.
        
        This is called when a human owner (typically during Twitter
        verification) links their Ethereum address to the wallet.
        The owner EOA can pause/recover the wallet.
        
        Uses atomic database operations to prevent race conditions where
        two concurrent claim requests could both succeed.
        
        Args:
            wallet_id: Wallet ID to claim
            owner_eoa: Owner's Ethereum address (EOA)
            
        Returns:
            WalletAccount: The updated wallet
            
        Raises:
            WalletNotFoundError: If wallet doesn't exist
            WalletServiceError: If wallet is already claimed
            
        TODO: Implement CDP owner assignment
        """
        # Validate EOA format first (before any DB operations)
        if not owner_eoa.startswith("0x") or len(owner_eoa) != 42:
            raise WalletServiceError(f"Invalid EOA format: {owner_eoa}")
        
        # SECURITY FIX: Use atomic claim to prevent race conditions
        # This atomically checks is_claimed=False AND sets owner_eoa in one operation
        claimed = self.storage.atomic_claim_wallet(
            wallet_id=wallet_id,
            owner_eoa=owner_eoa,
        )
        
        if not claimed:
            # Re-fetch to get current state for error message
            wallet = self.get_wallet(wallet_id)
            if wallet.is_claimed:
                raise WalletServiceError(
                    f"Wallet {wallet_id} is already claimed by {wallet.owner_eoa}"
                )
            else:
                raise WalletServiceError(f"Failed to claim wallet {wallet_id}")
        
        # TODO: Call CDP to set owner on the Smart Wallet
        # try:
        #     self._cdp_client.wallets.set_owner(
        #         wallet_id=wallet.cdp_wallet_id,
        #         owner_address=owner_eoa,
        #     )
        # except Exception as e:
        #     # Rollback the claim if CDP fails
        #     self.storage.unclaim_wallet(wallet_id)
        #     raise CDPIntegrationError(f"Failed to set owner: {e}") from e
        
        # Return updated wallet
        wallet = self.get_wallet(wallet_id)
        logger.info(f"Wallet {wallet_id} claimed by owner {owner_eoa}")
        return wallet
    
    def get_balance(self, wallet_id: str) -> WalletBalance:
        """Get the current balance of a wallet.
        
        Args:
            wallet_id: Wallet ID
            
        Returns:
            WalletBalance: Current balance information
            
        Raises:
            WalletNotFoundError: If wallet doesn't exist
            CDPIntegrationError: If balance check fails
            
        TODO: Implement actual balance checking via RPC/CDP
        """
        wallet = self.get_wallet(wallet_id)
        
        # TODO: Query actual balances from chain
        # try:
        #     # Get USDC balance
        #     usdc_contract = self._get_usdc_contract()
        #     usdc_raw = usdc_contract.functions.balanceOf(wallet.wallet_address).call()
        #     usdc_balance = Decimal(usdc_raw) / Decimal(10**6)  # USDC has 6 decimals
        #     
        #     # Get ETH balance for gas
        #     eth_raw = self._web3.eth.get_balance(wallet.wallet_address)
        #     eth_balance = Decimal(eth_raw) / Decimal(10**18)
        # except Exception as e:
        #     raise CDPIntegrationError(f"Failed to get balance: {e}") from e
        
        # STUB: Return zero balances for development
        return WalletBalance(
            wallet_address=wallet.wallet_address,
            usdc_balance=Decimal("0.00"),
            eth_balance=Decimal("0.00"),
            chain=wallet.chain,
            as_of=datetime.now(timezone.utc),
        )
    
    def get_balance_for_agent(self, agent_id: str) -> WalletBalance:
        """Get balance for an agent's wallet.
        
        Args:
            agent_id: Kernle agent ID
            
        Returns:
            WalletBalance: Current balance information
        """
        wallet = self.get_wallet_for_agent(agent_id)
        return self.get_balance(wallet.id)
    
    def transfer(
        self,
        wallet_id: str,
        to_address: str,
        amount: Decimal,
        actor_id: str,
    ) -> TransferResult:
        """Transfer USDC from wallet to another address.
        
        Validates spending limits and wallet status before executing.
        
        Args:
            wallet_id: Source wallet ID
            to_address: Destination Ethereum address
            amount: Amount of USDC to transfer
            actor_id: Agent/user initiating the transfer
            
        Returns:
            TransferResult: Result of the transfer operation
            
        Raises:
            WalletNotFoundError: If wallet doesn't exist
            WalletNotActiveError: If wallet cannot transact
            InsufficientBalanceError: If balance is too low
            SpendingLimitExceededError: If transfer exceeds limits
            
        TODO: Implement actual transfer via CDP
        """
        wallet = self.get_wallet(wallet_id)
        
        # SECURITY FIX: Verify actor is authorized to transfer from this wallet
        if not self._is_authorized_actor(wallet, actor_id):
            raise WalletAuthorizationError(
                f"Actor {actor_id} is not authorized to transfer from wallet {wallet_id}. "
                f"Must be wallet owner ({wallet.owner_eoa}) or agent ({wallet.agent_id})."
            )
        
        # SECURITY FIX: Reject non-positive transfer amounts
        # Negative amounts could bypass daily spending limits by reducing _daily_spend
        if amount <= 0:
            return TransferResult(
                success=False,
                from_address=wallet.wallet_address,
                to_address=to_address,
                amount=amount,
                error="Transfer amount must be positive",
            )
        
        # Validate wallet can transact
        if not wallet.can_transact:
            raise WalletNotActiveError(
                f"Wallet {wallet_id} is not active (status: {wallet.status})"
            )
        
        # Validate destination address
        if not to_address.startswith("0x") or len(to_address) != 42:
            return TransferResult(
                success=False,
                from_address=wallet.wallet_address,
                to_address=to_address,
                amount=amount,
                error=f"Invalid destination address: {to_address}",
            )
        
        # Check per-transaction limit
        if float(amount) > wallet.spending_limit_per_tx:
            raise SpendingLimitExceededError(
                f"Transfer amount {amount} exceeds per-tx limit "
                f"{wallet.spending_limit_per_tx}"
            )
        
        # SECURITY FIX: Atomic daily spending limit check and increment
        # Uses database atomic operation when available (thread-safe, persistent)
        # Falls back to in-memory with lock (thread-safe, but resets on restart)
        spend_result = self._try_atomic_spend(wallet_id, amount, wallet.spending_limit_daily)
        
        if not spend_result.success:
            raise SpendingLimitExceededError(
                f"Transfer would exceed daily limit. "
                f"Daily limit: {spend_result.daily_limit} USDC, "
                f"Already spent today: {spend_result.daily_spent} USDC, "
                f"Remaining: {spend_result.remaining} USDC"
            )
        
        # TODO: Check actual balance
        
        # TODO: Execute transfer via CDP
        # try:
        #     tx = self._cdp_client.wallets.transfer(
        #         wallet_id=wallet.cdp_wallet_id,
        #         to=to_address,
        #         token="USDC",
        #         amount=str(amount),
        #     )
        #     tx_hash = tx.hash
        # except Exception as e:
        #     # Rollback the daily spend on failure
        #     self._rollback_spend(wallet_id, amount)
        #     return TransferResult(
        #         success=False,
        #         from_address=wallet.wallet_address,
        #         to_address=to_address,
        #         amount=amount,
        #         error=str(e),
        #     )
        
        # STUB: Simulate successful transfer
        logger.info(
            f"STUB: Transfer {amount} USDC from {wallet.wallet_address} "
            f"to {to_address} (actor: {actor_id})"
        )
        
        return TransferResult(
            success=True,
            from_address=wallet.wallet_address,
            to_address=to_address,
            amount=amount,
            tx_hash=f"0x{'0' * 64}",  # Stub tx hash
            error=None,
        )
    
    def pause_wallet(self, wallet_id: str, actor_id: str) -> WalletAccount:
        """Pause a wallet (disable transactions).
        
        Only the owner EOA can pause a wallet. Agents cannot pause their
        own wallets to prevent malicious self-disabling.
        
        Args:
            wallet_id: Wallet ID to pause
            actor_id: Agent/user requesting the pause (must be owner_eoa)
            
        Returns:
            WalletAccount: Updated wallet
            
        Raises:
            WalletAuthorizationError: If actor is not the owner
        """
        wallet = self.get_wallet(wallet_id)
        
        # SECURITY FIX: Only owner can pause wallet
        if not self._is_authorized_actor(wallet, actor_id, require_owner=True):
            raise WalletAuthorizationError(
                f"Actor {actor_id} is not authorized to pause wallet {wallet_id}. "
                f"Only owner ({wallet.owner_eoa}) can pause wallets."
            )
        
        if wallet.status == WalletStatus.PAUSED.value:
            logger.warning(f"Wallet {wallet_id} is already paused")
            return wallet
        
        if wallet.status == WalletStatus.FROZEN.value:
            raise WalletServiceError("Cannot pause a frozen wallet")
        
        self.storage.update_wallet_status(wallet_id, WalletStatus.PAUSED)
        wallet.status = WalletStatus.PAUSED.value
        
        logger.info(f"Wallet {wallet_id} paused by {actor_id}")
        return wallet
    
    def resume_wallet(self, wallet_id: str, actor_id: str) -> WalletAccount:
        """Resume a paused wallet.
        
        Only the owner EOA can resume a wallet.
        
        Args:
            wallet_id: Wallet ID to resume
            actor_id: Agent/user requesting the resume (must be owner_eoa)
            
        Returns:
            WalletAccount: Updated wallet
            
        Raises:
            WalletAuthorizationError: If actor is not the owner
        """
        wallet = self.get_wallet(wallet_id)
        
        # SECURITY FIX: Only owner can resume wallet
        if not self._is_authorized_actor(wallet, actor_id, require_owner=True):
            raise WalletAuthorizationError(
                f"Actor {actor_id} is not authorized to resume wallet {wallet_id}. "
                f"Only owner ({wallet.owner_eoa}) can resume wallets."
            )
        
        if wallet.status != WalletStatus.PAUSED.value:
            raise WalletServiceError(
                f"Cannot resume wallet in status: {wallet.status}"
            )
        
        self.storage.update_wallet_status(wallet_id, WalletStatus.ACTIVE)
        wallet.status = WalletStatus.ACTIVE.value
        
        logger.info(f"Wallet {wallet_id} resumed by {actor_id}")
        return wallet
    
    def update_spending_limits(
        self,
        wallet_id: str,
        per_tx: Optional[float] = None,
        daily: Optional[float] = None,
        actor_id: Optional[str] = None,
    ) -> WalletAccount:
        """Update wallet spending limits.
        
        Only the owner EOA can modify spending limits.
        
        Args:
            wallet_id: Wallet ID
            per_tx: New per-transaction limit (USDC)
            daily: New daily limit (USDC)
            actor_id: Agent/user making the change (must be owner_eoa)
            
        Returns:
            WalletAccount: Updated wallet
            
        Raises:
            WalletAuthorizationError: If actor is not the owner
        """
        wallet = self.get_wallet(wallet_id)
        
        # SECURITY FIX: Only owner can modify spending limits
        if actor_id is None:
            raise WalletAuthorizationError(
                f"actor_id is required to update spending limits on wallet {wallet_id}"
            )
        if not self._is_authorized_actor(wallet, actor_id, require_owner=True):
            raise WalletAuthorizationError(
                f"Actor {actor_id} is not authorized to update spending limits on wallet {wallet_id}. "
                f"Only owner ({wallet.owner_eoa}) can modify spending limits."
            )
        
        if per_tx is not None:
            if per_tx <= 0:
                raise WalletServiceError("Per-transaction limit must be positive")
            wallet.spending_limit_per_tx = per_tx
            
        if daily is not None:
            if daily <= 0:
                raise WalletServiceError("Daily limit must be positive")
            wallet.spending_limit_daily = daily
        
        self.storage.update_spending_limits(wallet_id, per_tx, daily)
        
        logger.info(
            f"Wallet {wallet_id} spending limits updated: "
            f"per_tx={per_tx}, daily={daily} (by {actor_id})"
        )
        return wallet
