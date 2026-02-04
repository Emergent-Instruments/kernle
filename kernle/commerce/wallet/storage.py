"""
Wallet storage layer.

Provides persistence for wallet accounts using Supabase backend.
This is separate from the memory storage to maintain the conceptual
boundary between identity (memory) and capability (commerce).
"""

import logging
import threading
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import List, Optional, Protocol

from kernle.commerce.wallet.models import WalletAccount, WalletStatus

logger = logging.getLogger(__name__)


@dataclass
class DailySpendResult:
    """Result of a daily spend operation."""

    success: bool
    daily_spent: Decimal
    daily_limit: Decimal
    remaining: Decimal
    error: Optional[str] = None


class WalletStorage(Protocol):
    """Protocol for wallet persistence backends."""

    def save_wallet(self, wallet: WalletAccount) -> str:
        """Save a wallet account. Returns the wallet ID."""
        ...

    def get_wallet(self, wallet_id: str) -> Optional[WalletAccount]:
        """Get a wallet by ID."""
        ...

    def get_wallet_by_agent(self, agent_id: str) -> Optional[WalletAccount]:
        """Get a wallet by agent ID."""
        ...

    def get_wallet_by_address(self, address: str) -> Optional[WalletAccount]:
        """Get a wallet by Ethereum address."""
        ...

    def update_wallet_status(
        self,
        wallet_id: str,
        status: WalletStatus,
        owner_eoa: Optional[str] = None,
    ) -> bool:
        """Update wallet status and optionally set owner EOA."""
        ...

    def atomic_claim_wallet(
        self,
        wallet_id: str,
        owner_eoa: str,
    ) -> bool:
        """Atomically claim a wallet if not already claimed.

        This operation checks that the wallet is unclaimed AND sets the
        owner_eoa in a single atomic operation to prevent race conditions.

        Args:
            wallet_id: Wallet ID to claim
            owner_eoa: Owner's Ethereum address

        Returns:
            True if claim succeeded, False if already claimed or wallet not found
        """
        ...

    def update_spending_limits(
        self,
        wallet_id: str,
        per_tx: Optional[float] = None,
        daily: Optional[float] = None,
    ) -> bool:
        """Update wallet spending limits."""
        ...

    def list_wallets_by_user(self, user_id: str) -> List[WalletAccount]:
        """List all wallets owned by a user."""
        ...

    def get_daily_spend(self, wallet_id: str) -> Optional[Decimal]:
        """Get current daily spend for a wallet (resets at midnight UTC).

        Returns None if not implemented or wallet not found.
        """
        ...

    def increment_daily_spend(
        self,
        wallet_id: str,
        amount: Decimal,
    ) -> Optional[DailySpendResult]:
        """Atomically increment daily spend with limit check.

        Returns DailySpendResult on success/failure, or None if not implemented.
        The implementation should:
        1. Reset spending if date changed (midnight UTC)
        2. Check if new total would exceed daily limit
        3. Only increment if within limit
        4. Use atomic database operation to prevent race conditions
        """
        ...


class InMemoryWalletStorage:
    """In-memory wallet storage for testing and development.

    Note: Wallet data is not persisted between process restarts.
    However, daily spend tracking can optionally be persisted to a file
    using the persist_path parameter.

    Args:
        persist_path: Optional path to persist daily spend tracking.
                     If provided, daily spend survives restarts.
    """

    def __init__(self, persist_path: Optional[str] = None):
        self._wallets: dict[str, WalletAccount] = {}
        self._by_agent: dict[str, str] = {}  # agent_id -> wallet_id
        self._by_address: dict[str, str] = {}  # wallet_address -> wallet_id
        self._by_user: dict[str, List[str]] = {}  # user_id -> [wallet_ids]
        # Daily spending tracking with thread safety
        self._daily_spend: dict[str, tuple[date, Decimal, int]] = (
            {}
        )  # wallet_id -> (date, total, count)
        self._daily_spend_lock = threading.Lock()
        # Optional persistence for daily spend (security fix: survive restarts)
        self._persist_path = persist_path
        if persist_path:
            self._load_daily_spend()

    def _load_daily_spend(self) -> None:
        """Load persisted daily spend tracking from file."""
        import json
        from pathlib import Path

        if not self._persist_path:
            return

        path = Path(self._persist_path)
        if not path.exists():
            return

        try:
            with open(path) as f:
                data = json.load(f)

            today = datetime.now(timezone.utc).date()
            for wallet_id, entry in data.items():
                spend_date = date.fromisoformat(entry["date"])
                # Only load if it's still today (daily limits reset at midnight)
                if spend_date == today:
                    self._daily_spend[wallet_id] = (
                        spend_date,
                        Decimal(entry["total"]),
                        entry["count"],
                    )
            logger.debug(f"Loaded daily spend tracking from {path}")
        except Exception as e:
            logger.warning(f"Failed to load daily spend from {path}: {e}")

    def _save_daily_spend(self) -> None:
        """Persist daily spend tracking to file."""
        import json
        from pathlib import Path

        if not self._persist_path:
            return

        path = Path(self._persist_path)
        try:
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            data = {}
            for wallet_id, (spend_date, total, count) in self._daily_spend.items():
                data[wallet_id] = {
                    "date": spend_date.isoformat(),
                    "total": str(total),
                    "count": count,
                }

            with open(path, "w") as f:
                json.dump(data, f)
            logger.debug(f"Saved daily spend tracking to {path}")
        except Exception as e:
            logger.warning(f"Failed to save daily spend to {path}: {e}")

    def _utc_now(self) -> datetime:
        """Get current UTC timestamp."""
        return datetime.now(timezone.utc)

    def save_wallet(self, wallet: WalletAccount) -> str:
        """Save a wallet account."""
        self._wallets[wallet.id] = wallet
        self._by_agent[wallet.agent_id] = wallet.id
        self._by_address[wallet.wallet_address] = wallet.id
        if wallet.user_id:
            if wallet.user_id not in self._by_user:
                self._by_user[wallet.user_id] = []
            if wallet.id not in self._by_user[wallet.user_id]:
                self._by_user[wallet.user_id].append(wallet.id)
        logger.debug(f"Saved wallet {wallet.id} for agent {wallet.agent_id}")
        return wallet.id

    def get_wallet(self, wallet_id: str) -> Optional[WalletAccount]:
        """Get a wallet by ID."""
        return self._wallets.get(wallet_id)

    def get_wallet_by_agent(self, agent_id: str) -> Optional[WalletAccount]:
        """Get a wallet by agent ID."""
        wallet_id = self._by_agent.get(agent_id)
        if wallet_id:
            return self._wallets.get(wallet_id)
        return None

    def get_wallet_by_address(self, address: str) -> Optional[WalletAccount]:
        """Get a wallet by Ethereum address."""
        wallet_id = self._by_address.get(address)
        if wallet_id:
            return self._wallets.get(wallet_id)
        return None

    def update_wallet_status(
        self,
        wallet_id: str,
        status: WalletStatus,
        owner_eoa: Optional[str] = None,
    ) -> bool:
        """Update wallet status and optionally set owner EOA."""
        wallet = self._wallets.get(wallet_id)
        if not wallet:
            return False

        wallet.status = status.value if hasattr(status, "value") else status
        if owner_eoa:
            wallet.owner_eoa = owner_eoa
            wallet.claimed_at = self._utc_now()
        logger.debug(f"Updated wallet {wallet_id} status to {wallet.status}")
        return True

    def update_spending_limits(
        self,
        wallet_id: str,
        per_tx: Optional[float] = None,
        daily: Optional[float] = None,
    ) -> bool:
        """Update wallet spending limits."""
        wallet = self._wallets.get(wallet_id)
        if not wallet:
            return False

        if per_tx is not None:
            wallet.spending_limit_per_tx = per_tx
        if daily is not None:
            wallet.spending_limit_daily = daily
        logger.debug(f"Updated wallet {wallet_id} spending limits")
        return True

    def atomic_claim_wallet(
        self,
        wallet_id: str,
        owner_eoa: str,
    ) -> bool:
        """Atomically claim a wallet if not already claimed.

        In-memory implementation - single-threaded so naturally atomic.
        """
        wallet = self._wallets.get(wallet_id)
        if not wallet:
            return False
        if wallet.owner_eoa is not None:
            return False  # Already claimed
        # In-memory is single-threaded, so this is atomic
        wallet.owner_eoa = owner_eoa
        wallet.status = WalletStatus.ACTIVE.value
        wallet.claimed_at = self._utc_now()
        logger.debug(f"Wallet {wallet_id} atomically claimed by {owner_eoa}")
        return True

    def list_wallets_by_user(self, user_id: str) -> List[WalletAccount]:
        """List all wallets owned by a user."""
        wallet_ids = self._by_user.get(user_id, [])
        return [self._wallets[wid] for wid in wallet_ids if wid in self._wallets]

    def get_daily_spend(self, wallet_id: str) -> Optional[Decimal]:
        """Get current daily spend for a wallet (resets at midnight UTC)."""
        today = datetime.now(timezone.utc).date()

        with self._daily_spend_lock:
            if wallet_id not in self._daily_spend:
                return Decimal("0")

            spend_date, total, _ = self._daily_spend[wallet_id]

            # Reset if it's a new day
            if spend_date != today:
                return Decimal("0")

            return total

    def increment_daily_spend(
        self,
        wallet_id: str,
        amount: Decimal,
    ) -> Optional[DailySpendResult]:
        """Atomically increment daily spend with limit check.

        Thread-safe in-memory implementation.
        """
        if amount <= 0:
            return DailySpendResult(
                success=False,
                daily_spent=Decimal("0"),
                daily_limit=Decimal("0"),
                remaining=Decimal("0"),
                error="Amount must be positive",
            )

        wallet = self._wallets.get(wallet_id)
        if not wallet:
            return None  # Let caller handle missing wallet

        daily_limit = Decimal(str(wallet.spending_limit_daily))
        today = datetime.now(timezone.utc).date()

        with self._daily_spend_lock:
            # Get or reset daily spending
            if wallet_id not in self._daily_spend or self._daily_spend[wallet_id][0] != today:
                current_spent = Decimal("0")
                current_count = 0
            else:
                _, current_spent, current_count = self._daily_spend[wallet_id]

            new_spent = current_spent + amount

            # Check limit
            if new_spent > daily_limit:
                return DailySpendResult(
                    success=False,
                    daily_spent=current_spent,
                    daily_limit=daily_limit,
                    remaining=daily_limit - current_spent,
                    error=f"Would exceed daily limit. Remaining: {daily_limit - current_spent} USDC",
                )

            # Commit the increment
            self._daily_spend[wallet_id] = (today, new_spent, current_count + 1)

        # Persist outside the lock to avoid holding it during I/O
        self._save_daily_spend()

        return DailySpendResult(
            success=True,
            daily_spent=new_spent,
            daily_limit=daily_limit,
            remaining=daily_limit - new_spent,
        )


class SupabaseWalletStorage:
    """Supabase-backed wallet storage.

    Note: This is a placeholder implementation. The actual Supabase
    integration will be added when the backend routes are implemented.
    """

    def __init__(self, supabase_url: str, supabase_key: str):
        """Initialize Supabase connection.

        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase service role key
        """
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        # Will initialize actual client when supabase-py is added as dependency
        self._client = None

    def _utc_now(self) -> datetime:
        """Get current UTC timestamp."""
        return datetime.now(timezone.utc)

    def save_wallet(self, wallet: WalletAccount) -> str:
        """Save a wallet account."""
        logger.info(f"Saving wallet {wallet.id} for agent {wallet.agent_id}")
        # TODO: Implement Supabase insert
        # data = wallet.to_dict()
        # self._client.table("wallet_accounts").upsert(data).execute()
        return wallet.id

    def get_wallet(self, wallet_id: str) -> Optional[WalletAccount]:
        """Get a wallet by ID."""
        logger.debug(f"Getting wallet {wallet_id}")
        # TODO: Implement Supabase query
        # result = self._client.table("wallet_accounts").select("*").eq("id", wallet_id).single().execute()
        # if result.data:
        #     return WalletAccount.from_dict(result.data)
        return None

    def get_wallet_by_agent(self, agent_id: str) -> Optional[WalletAccount]:
        """Get a wallet by agent ID."""
        logger.debug(f"Getting wallet for agent {agent_id}")
        # TODO: Implement Supabase query
        # result = self._client.table("wallet_accounts").select("*").eq("agent_id", agent_id).single().execute()
        # if result.data:
        #     return WalletAccount.from_dict(result.data)
        return None

    def get_wallet_by_address(self, address: str) -> Optional[WalletAccount]:
        """Get a wallet by Ethereum address."""
        logger.debug(f"Getting wallet for address {address}")
        # TODO: Implement Supabase query
        return None

    def update_wallet_status(
        self,
        wallet_id: str,
        status: WalletStatus,
        owner_eoa: Optional[str] = None,
    ) -> bool:
        """Update wallet status."""
        logger.info(f"Updating wallet {wallet_id} status to {status.value}")
        # TODO: Implement Supabase update
        # update_data = {"status": status.value}
        # if owner_eoa:
        #     update_data["owner_eoa"] = owner_eoa
        #     update_data["claimed_at"] = self._utc_now().isoformat()
        # self._client.table("wallet_accounts").update(update_data).eq("id", wallet_id).execute()
        return True

    def update_spending_limits(
        self,
        wallet_id: str,
        per_tx: Optional[float] = None,
        daily: Optional[float] = None,
    ) -> bool:
        """Update wallet spending limits."""
        logger.info(f"Updating spending limits for wallet {wallet_id}")
        # TODO: Implement Supabase update
        return True

    def atomic_claim_wallet(
        self,
        wallet_id: str,
        owner_eoa: str,
    ) -> bool:
        """Atomically claim a wallet if not already claimed.

        Uses Supabase's update with conditional WHERE clause to ensure
        atomicity. The update only succeeds if owner_eoa IS NULL.

        SQL equivalent:
            UPDATE wallet_accounts
            SET owner_eoa = :owner_eoa,
                status = 'active',
                claimed_at = NOW()
            WHERE id = :wallet_id
              AND owner_eoa IS NULL
            RETURNING id;
        """
        logger.info(f"Attempting atomic claim of wallet {wallet_id}")
        # TODO: Implement Supabase atomic update
        # The key is using a conditional update that only succeeds if unclaimed:
        #
        # result = self._client.table("wallet_accounts") \
        #     .update({
        #         "owner_eoa": owner_eoa,
        #         "status": "active",
        #         "claimed_at": self._utc_now().isoformat(),
        #     }) \
        #     .eq("id", wallet_id) \
        #     .is_("owner_eoa", "null") \
        #     .execute()
        #
        # # Check if any rows were updated
        # return len(result.data) > 0
        logger.warning("SupabaseWalletStorage.atomic_claim_wallet is not implemented")
        return False

    def list_wallets_by_user(self, user_id: str) -> List[WalletAccount]:
        """List all wallets owned by a user."""
        logger.debug(f"Listing wallets for user {user_id}")
        # TODO: Implement Supabase query
        return []

    def get_daily_spend(self, wallet_id: str) -> Optional[Decimal]:
        """Get current daily spend for a wallet (resets at midnight UTC).

        Uses the get_wallet_daily_spend database function.
        """
        logger.debug(f"Getting daily spend for wallet {wallet_id}")
        # TODO: Implement when Supabase client is ready
        # result = self._client.rpc(
        #     "get_wallet_daily_spend",
        #     {"p_wallet_id": wallet_id}
        # ).execute()
        # if result.data and len(result.data) > 0:
        #     return Decimal(str(result.data[0]["daily_spent"]))
        return None  # Fall back to service in-memory

    def increment_daily_spend(
        self,
        wallet_id: str,
        amount: Decimal,
    ) -> Optional[DailySpendResult]:
        """Atomically increment daily spend with limit check.

        Uses the increment_wallet_daily_spend database function which:
        1. Resets spending at midnight UTC
        2. Checks if new total exceeds daily limit
        3. Only increments if within limit
        4. Uses row-level locking (FOR UPDATE) for atomicity

        Returns empty result if limit exceeded.
        """
        logger.debug(f"Incrementing daily spend for wallet {wallet_id} by {amount}")
        # TODO: Implement when Supabase client is ready
        # try:
        #     result = self._client.rpc(
        #         "increment_wallet_daily_spend",
        #         {"p_wallet_id": wallet_id, "p_amount": str(amount)}
        #     ).execute()
        #
        #     if not result.data or len(result.data) == 0:
        #         # Empty result means limit exceeded
        #         # Get current state for error message
        #         current = self.get_daily_spend(wallet_id) or Decimal("0")
        #         wallet = self.get_wallet(wallet_id)
        #         daily_limit = Decimal(str(wallet.spending_limit_daily)) if wallet else Decimal("0")
        #         return DailySpendResult(
        #             success=False,
        #             daily_spent=current,
        #             daily_limit=daily_limit,
        #             remaining=daily_limit - current,
        #             error=f"Would exceed daily limit"
        #         )
        #
        #     row = result.data[0]
        #     return DailySpendResult(
        #         success=True,
        #         daily_spent=Decimal(str(row["daily_spent"])),
        #         daily_limit=Decimal(str(row["daily_limit"])),
        #         remaining=Decimal(str(row["remaining"]))
        #     )
        # except Exception as e:
        #     logger.error(f"Failed to increment daily spend: {e}")
        #     return None
        return None  # Fall back to service in-memory
