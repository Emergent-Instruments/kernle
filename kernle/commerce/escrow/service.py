"""
Escrow service for Kernle Commerce.

Provides high-level escrow operations including:
- Deploying escrow contracts via factory
- Funding escrows with USDC
- Managing escrow lifecycle (assign, deliver, release, refund)
- Dispute handling

The escrow system uses smart contracts on Base to hold USDC payments
securely until job completion. Each job gets its own escrow contract.

Flow:
1. Client creates job → JobService.create_job()
2. Client deploys escrow → EscrowService.deploy_escrow()
3. Client funds escrow → EscrowService.fund()
4. Client accepts worker → EscrowService.assign_worker()
5. Worker delivers → EscrowService.mark_delivered()
6. Client approves → EscrowService.release()

NOTE: This service contains stubs for blockchain interaction.
Actual implementation requires Web3 and contract deployment.
"""

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Optional, Set

from kernle.commerce.config import CommerceConfig, get_config
from kernle.commerce.escrow.abi import (
    EscrowStatus,
)

logger = logging.getLogger(__name__)


class EscrowServiceError(Exception):
    """Base exception for escrow service errors."""
    pass


class EscrowNotFoundError(EscrowServiceError):
    """Escrow contract not found."""
    pass


class InsufficientAllowanceError(EscrowServiceError):
    """Insufficient USDC allowance for escrow."""
    pass


class InvalidEscrowStateError(EscrowServiceError):
    """Escrow is in invalid state for requested operation."""
    pass


class TransactionFailedError(EscrowServiceError):
    """Blockchain transaction failed."""
    pass


@dataclass
class EscrowInfo:
    """Information about an escrow contract.

    Attributes:
        address: Escrow contract address
        job_id: Associated job ID (bytes32)
        client: Client address
        worker: Worker address (if assigned)
        amount: Escrow amount in USDC (human-readable)
        status: Current escrow status
        deadline: Job deadline timestamp
        deliverable_hash: Hash of deliverable (if delivered)
        delivered_at: When delivery was submitted
        approval_timeout: Seconds until auto-release after delivery
    """
    address: str
    job_id: str
    client: str
    amount: Decimal
    status: int
    deadline: datetime
    worker: Optional[str] = None
    deliverable_hash: Optional[str] = None
    delivered_at: Optional[datetime] = None
    approval_timeout: int = 604800  # 7 days in seconds

    @property
    def status_name(self) -> str:
        """Get human-readable status name."""
        return EscrowStatus.name(self.status)

    @property
    def is_funded(self) -> bool:
        """Check if escrow is funded."""
        return self.status >= EscrowStatus.FUNDED

    @property
    def is_active(self) -> bool:
        """Check if escrow is in an active state."""
        return self.status in {
            EscrowStatus.FUNDED,
            EscrowStatus.ACCEPTED,
            EscrowStatus.DELIVERED,
        }

    @property
    def is_terminal(self) -> bool:
        """Check if escrow is in a terminal state."""
        return self.status in {
            EscrowStatus.RELEASED,
            EscrowStatus.REFUNDED,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "address": self.address,
            "job_id": self.job_id,
            "client": self.client,
            "worker": self.worker,
            "amount": str(self.amount),
            "status": self.status,
            "status_name": self.status_name,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "deliverable_hash": self.deliverable_hash,
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "approval_timeout": self.approval_timeout,
        }


@dataclass
class TransactionResult:
    """Result of a blockchain transaction.

    Attributes:
        success: Whether transaction succeeded
        tx_hash: Transaction hash
        block_number: Block where tx was included
        gas_used: Gas consumed
        error: Error message if failed
    """
    success: bool
    tx_hash: Optional[str] = None
    block_number: Optional[int] = None
    gas_used: Optional[int] = None
    error: Optional[str] = None


class EscrowService:
    """Service for escrow contract operations.

    Handles all blockchain interactions for the escrow system.
    Uses the KernleEscrowFactory to deploy per-job escrow contracts
    and manages their lifecycle.

    Example:
        >>> service = EscrowService()
        >>> escrow = service.deploy_escrow(
        ...     job_id="550e8400-e29b-41d4-a716-446655440000",
        ...     client_address="0x123...",
        ...     amount=Decimal("100.00"),
        ...     deadline=datetime(2025, 2, 1),
        ... )
        >>> service.fund(escrow.address, client_address="0x123...")
        >>> service.assign_worker(escrow.address, worker_address="0x456...")
        >>> service.release(escrow.address)
    """

    def __init__(
        self,
        config: Optional[CommerceConfig] = None,
    ):
        """Initialize escrow service.

        Args:
            config: Commerce configuration (uses global config if not provided)
        """
        self.config = config or get_config()

        # TODO: Initialize Web3 connection
        # self._web3 = Web3(Web3.HTTPProvider(self.config.rpc_url))
        self._web3 = None

        # Contract instances (lazy-loaded)
        self._factory = None
        self._usdc = None

        # SECURITY FIX: Reentrancy protection
        # Track which escrows are currently in a state-changing operation
        self._operation_locks: Dict[str, threading.Lock] = {}
        self._lock_manager = threading.Lock()
        self._active_operations: Set[str] = set()

    def _get_escrow_lock(self, escrow_address: str) -> threading.Lock:
        """Get or create a lock for an escrow address."""
        if escrow_address not in self._operation_locks:
            with self._lock_manager:
                if escrow_address not in self._operation_locks:
                    self._operation_locks[escrow_address] = threading.Lock()
        return self._operation_locks[escrow_address]

    def _check_reentrancy(self, escrow_address: str, operation: str) -> None:
        """Check for reentrancy and raise if detected.

        Args:
            escrow_address: The escrow being operated on
            operation: Name of the operation

        Raises:
            EscrowServiceError: If reentrancy is detected
        """
        key = f"{escrow_address}:{operation}"
        with self._lock_manager:
            if key in self._active_operations:
                raise EscrowServiceError(
                    f"Reentrancy detected: {operation} already in progress for {escrow_address}"
                )
            self._active_operations.add(key)

    def _clear_reentrancy(self, escrow_address: str, operation: str) -> None:
        """Clear reentrancy flag after operation completes."""
        key = f"{escrow_address}:{operation}"
        with self._lock_manager:
            self._active_operations.discard(key)

    def _get_factory(self):
        """Get factory contract instance.

        TODO: Implement with Web3 contract
        """
        if not self.config.escrow_factory_address:
            raise EscrowServiceError("Escrow factory address not configured")

        # TODO: Return web3 contract instance
        # if self._factory is None:
        #     self._factory = self._web3.eth.contract(
        #         address=self.config.escrow_factory_address,
        #         abi=KERNLE_ESCROW_FACTORY_ABI,
        #     )
        # return self._factory

        logger.warning("STUB: Factory contract not initialized")
        return None

    def _get_usdc(self):
        """Get USDC contract instance.

        TODO: Implement with Web3 contract
        """
        if not self.config.usdc_address:
            raise EscrowServiceError("USDC address not configured")

        # TODO: Return web3 contract instance
        # if self._usdc is None:
        #     self._usdc = self._web3.eth.contract(
        #         address=self.config.usdc_address,
        #         abi=ERC20_ABI,
        #     )
        # return self._usdc

        logger.warning("STUB: USDC contract not initialized")
        return None

    def _get_escrow(self, address: str):
        """Get escrow contract instance.

        Args:
            address: Escrow contract address

        TODO: Implement with Web3 contract
        """
        # TODO: Return web3 contract instance
        # return self._web3.eth.contract(
        #     address=address,
        #     abi=KERNLE_ESCROW_ABI,
        # )

        logger.warning(f"STUB: Escrow contract {address} not initialized")
        return None

    def _job_id_to_bytes32(self, job_id: str) -> bytes:
        """Convert job ID to bytes32.

        Args:
            job_id: Job ID string (UUID format or any string)

        Returns:
            32-byte representation
        """
        import hashlib
        # Use SHA256 hash to get consistent 32 bytes from any input
        return hashlib.sha256(job_id.encode()).digest()

    def _usdc_to_wei(self, amount: Decimal) -> int:
        """Convert USDC amount to wei (6 decimals).

        Args:
            amount: Amount in USDC (e.g., 100.50)

        Returns:
            Amount in smallest units (e.g., 100500000)
        """
        return int(amount * Decimal("1000000"))

    def _wei_to_usdc(self, wei: int) -> Decimal:
        """Convert wei to USDC amount.

        Args:
            wei: Amount in smallest units

        Returns:
            Amount in USDC
        """
        return Decimal(wei) / Decimal("1000000")

    # =========================================================================
    # Escrow Deployment
    # =========================================================================

    def deploy_escrow(
        self,
        job_id: str,
        client_address: str,
        amount: Decimal,
        deadline: datetime,
    ) -> EscrowInfo:
        """Deploy a new escrow contract for a job.

        This creates a new escrow contract via the factory. The contract
        is created but not yet funded.

        Args:
            job_id: Job ID (UUID format)
            client_address: Client's wallet address
            amount: Escrow amount in USDC
            deadline: Job deadline

        Returns:
            EscrowInfo: Information about the deployed escrow

        Raises:
            EscrowServiceError: If deployment fails

        TODO: Implement actual contract deployment
        """
        logger.info(
            f"Deploying escrow for job {job_id}: "
            f"{amount} USDC, deadline {deadline}"
        )

        # Convert parameters
        self._job_id_to_bytes32(job_id)
        self._usdc_to_wei(amount)
        int(deadline.timestamp())

        # TODO: Deploy via factory
        # factory = self._get_factory()
        # tx = factory.functions.createEscrow(
        #     jobId=job_id_bytes,
        #     amount=amount_wei,
        #     deadline=deadline_ts,
        # ).build_transaction({
        #     "from": client_address,
        #     "gas": 500000,
        # })
        # signed_tx = self._web3.eth.account.sign_transaction(tx, private_key)
        # tx_hash = self._web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        # receipt = self._web3.eth.wait_for_transaction_receipt(tx_hash)
        # escrow_address = receipt.logs[0].args.escrow

        # STUB: Generate placeholder escrow address
        import uuid
        # Ethereum addresses are 40 hex chars after 0x
        base_hash = uuid.uuid5(uuid.NAMESPACE_DNS, job_id).hex
        extra_hash = uuid.uuid5(uuid.NAMESPACE_URL, job_id).hex[:8]
        escrow_address = f"0x{(base_hash + extra_hash)[:40]}"

        escrow_info = EscrowInfo(
            address=escrow_address,
            job_id=job_id,
            client=client_address,
            amount=amount,
            status=EscrowStatus.CREATED,
            deadline=deadline,
            approval_timeout=self.config.approval_timeout_days * 86400,
        )

        logger.info(f"STUB: Deployed escrow at {escrow_address}")
        return escrow_info

    def get_escrow(self, address: str) -> EscrowInfo:
        """Get escrow information by address.

        Args:
            address: Escrow contract address

        Returns:
            EscrowInfo: Current escrow state

        Raises:
            EscrowNotFoundError: If escrow doesn't exist

        TODO: Implement actual contract reading
        """
        logger.debug(f"Getting escrow info: {address}")

        # TODO: Read from contract
        # escrow = self._get_escrow(address)
        # return EscrowInfo(
        #     address=address,
        #     job_id=escrow.functions.jobId().call().hex(),
        #     client=escrow.functions.client().call(),
        #     worker=escrow.functions.worker().call() or None,
        #     amount=self._wei_to_usdc(escrow.functions.amount().call()),
        #     status=escrow.functions.status().call(),
        #     deadline=datetime.fromtimestamp(
        #         escrow.functions.deadline().call(),
        #         tz=timezone.utc,
        #     ),
        #     deliverable_hash=escrow.functions.deliverableHash().call().hex() or None,
        #     delivered_at=...,
        #     approval_timeout=escrow.functions.approvalTimeout().call(),
        # )

        # STUB: Return placeholder
        raise EscrowNotFoundError(f"STUB: Escrow {address} not found")

    def get_escrow_for_job(self, job_id: str) -> EscrowInfo:
        """Get escrow by job ID.

        Args:
            job_id: Job ID

        Returns:
            EscrowInfo: Escrow for this job

        Raises:
            EscrowNotFoundError: If no escrow for this job
        """
        logger.debug(f"Getting escrow for job: {job_id}")

        # TODO: Query factory for escrow address
        # factory = self._get_factory()
        # job_id_bytes = self._job_id_to_bytes32(job_id)
        # escrow_address = factory.functions.getEscrow(job_id_bytes).call()
        # if escrow_address == "0x" + "0" * 40:
        #     raise EscrowNotFoundError(f"No escrow for job {job_id}")
        # return self.get_escrow(escrow_address)

        raise EscrowNotFoundError(f"STUB: No escrow for job {job_id}")

    # =========================================================================
    # Escrow Operations
    # =========================================================================

    def fund(
        self,
        escrow_address: str,
        client_address: str,
    ) -> TransactionResult:
        """Fund an escrow with USDC.

        The client must have approved the escrow contract to spend
        the required USDC amount before calling this.

        Args:
            escrow_address: Escrow contract address
            client_address: Client's wallet address

        Returns:
            TransactionResult: Result of the funding transaction

        TODO: Implement actual funding
        """
        logger.info(f"Funding escrow {escrow_address} from {client_address}")

        # TODO: Check allowance and fund
        # escrow = self._get_escrow(escrow_address)
        # amount = escrow.functions.amount().call()
        #
        # # Check USDC allowance
        # usdc = self._get_usdc()
        # allowance = usdc.functions.allowance(client_address, escrow_address).call()
        # if allowance < amount:
        #     raise InsufficientAllowanceError(
        #         f"Allowance {allowance} < required {amount}"
        #     )
        #
        # # Fund escrow
        # tx = escrow.functions.fund().build_transaction({
        #     "from": client_address,
        #     "gas": 150000,
        # })
        # ... sign and send ...

        # STUB
        return TransactionResult(
            success=True,
            tx_hash=f"0x{'0' * 64}",
            block_number=0,
            gas_used=100000,
        )

    def approve_usdc(
        self,
        owner_address: str,
        spender_address: str,
        amount: Decimal,
    ) -> TransactionResult:
        """Approve USDC spending for an escrow contract.

        Must be called before funding an escrow.

        Args:
            owner_address: Token owner (client)
            spender_address: Spender (escrow contract)
            amount: Amount to approve

        Returns:
            TransactionResult: Result of approval transaction

        TODO: Implement actual approval
        """
        logger.info(
            f"Approving {amount} USDC: {owner_address} → {spender_address}"
        )

        # TODO: Implement USDC approve call
        # usdc = self._get_usdc()
        # amount_wei = self._usdc_to_wei(amount)
        # tx = usdc.functions.approve(spender_address, amount_wei).build_transaction({
        #     "from": owner_address,
        #     "gas": 60000,
        # })
        # ... sign and send ...

        # STUB
        return TransactionResult(
            success=True,
            tx_hash=f"0x{'0' * 64}",
        )

    def assign_worker(
        self,
        escrow_address: str,
        worker_address: str,
        client_address: str,
    ) -> TransactionResult:
        """Assign a worker to the escrow.

        Called when client accepts an application. The escrow must be funded.

        Args:
            escrow_address: Escrow contract address
            worker_address: Worker's wallet address
            client_address: Client's wallet address (must be escrow client)

        Returns:
            TransactionResult: Result of assignment transaction

        TODO: Implement actual worker assignment
        """
        logger.info(
            f"Assigning worker {worker_address} to escrow {escrow_address}"
        )

        # TODO: Implement worker assignment
        # escrow = self._get_escrow(escrow_address)
        # tx = escrow.functions.assignWorker(worker_address).build_transaction({
        #     "from": client_address,
        #     "gas": 80000,
        # })
        # ... sign and send ...

        # STUB
        return TransactionResult(
            success=True,
            tx_hash=f"0x{'0' * 64}",
        )

    def mark_delivered(
        self,
        escrow_address: str,
        deliverable_hash: str,
        worker_address: str,
    ) -> TransactionResult:
        """Mark escrow as delivered.

        Called when worker submits deliverable. Starts approval timeout.

        Args:
            escrow_address: Escrow contract address
            deliverable_hash: IPFS/content hash of deliverable
            worker_address: Worker's wallet address

        Returns:
            TransactionResult: Result of delivery transaction

        TODO: Implement actual delivery marking
        """
        logger.info(f"Marking escrow {escrow_address} as delivered")

        # TODO: Implement delivery
        # escrow = self._get_escrow(escrow_address)
        # hash_bytes = bytes.fromhex(deliverable_hash.replace("0x", ""))
        # tx = escrow.functions.deliver(hash_bytes).build_transaction({
        #     "from": worker_address,
        #     "gas": 80000,
        # })
        # ... sign and send ...

        # STUB
        return TransactionResult(
            success=True,
            tx_hash=f"0x{'0' * 64}",
        )

    def release(
        self,
        escrow_address: str,
        client_address: str,
    ) -> TransactionResult:
        """Release escrow payment to worker.

        Called when client approves the deliverable.
        Uses reentrancy guard to prevent double-release attacks.

        Args:
            escrow_address: Escrow contract address
            client_address: Client's wallet address

        Returns:
            TransactionResult: Result of release transaction

        TODO: Implement actual release
        """
        # SECURITY FIX: Reentrancy guard
        escrow_lock = self._get_escrow_lock(escrow_address)

        with escrow_lock:
            self._check_reentrancy(escrow_address, "release")
            try:
                logger.info(f"Releasing escrow {escrow_address}")

                # TODO: Implement release
                # escrow = self._get_escrow(escrow_address)
                # tx = escrow.functions.release().build_transaction({
                #     "from": client_address,
                #     "gas": 100000,
                # })
                # ... sign and send ...

                # STUB
                return TransactionResult(
                    success=True,
                    tx_hash=f"0x{'0' * 64}",
                )
            finally:
                self._clear_reentrancy(escrow_address, "release")

    def auto_release(
        self,
        escrow_address: str,
    ) -> TransactionResult:
        """Trigger auto-release after approval timeout.

        Anyone can call this after the approval timeout has expired.

        Args:
            escrow_address: Escrow contract address

        Returns:
            TransactionResult: Result of auto-release transaction

        TODO: Implement actual auto-release
        """
        logger.info(f"Auto-releasing escrow {escrow_address}")

        # TODO: Implement auto-release
        # escrow = self._get_escrow(escrow_address)
        # tx = escrow.functions.autoRelease().build_transaction({
        #     "gas": 100000,
        # })
        # ... sign and send ...

        # STUB
        return TransactionResult(
            success=True,
            tx_hash=f"0x{'0' * 64}",
        )

    def refund(
        self,
        escrow_address: str,
        client_address: str,
    ) -> TransactionResult:
        """Refund escrow to client (cancel job).

        Can only be called before worker is assigned.
        Uses reentrancy guard to prevent double-refund attacks.

        Args:
            escrow_address: Escrow contract address
            client_address: Client's wallet address

        Returns:
            TransactionResult: Result of refund transaction

        TODO: Implement actual refund
        """
        # SECURITY FIX: Reentrancy guard
        escrow_lock = self._get_escrow_lock(escrow_address)

        with escrow_lock:
            self._check_reentrancy(escrow_address, "refund")
            try:
                logger.info(f"Refunding escrow {escrow_address}")

                # TODO: Implement refund
                # escrow = self._get_escrow(escrow_address)
                # tx = escrow.functions.refund().build_transaction({
                #     "from": client_address,
                #     "gas": 100000,
                # })
                # ... sign and send ...

                # STUB
                return TransactionResult(
                    success=True,
                    tx_hash=f"0x{'0' * 64}",
                )
            finally:
                self._clear_reentrancy(escrow_address, "refund")

    # =========================================================================
    # Disputes
    # =========================================================================

    def raise_dispute(
        self,
        escrow_address: str,
        disputant_address: str,
    ) -> TransactionResult:
        """Raise a dispute on the escrow.

        Can be called by client or worker during accepted/delivered states.

        Args:
            escrow_address: Escrow contract address
            disputant_address: Address raising the dispute

        Returns:
            TransactionResult: Result of dispute transaction

        TODO: Implement actual dispute raising
        """
        logger.info(f"Raising dispute on escrow {escrow_address}")

        # TODO: Implement dispute
        # escrow = self._get_escrow(escrow_address)
        # tx = escrow.functions.dispute().build_transaction({
        #     "from": disputant_address,
        #     "gas": 80000,
        # })
        # ... sign and send ...

        # STUB
        return TransactionResult(
            success=True,
            tx_hash=f"0x{'0' * 64}",
        )

    def resolve_dispute(
        self,
        escrow_address: str,
        recipient_address: str,
        arbitrator_address: str,
    ) -> TransactionResult:
        """Resolve a dispute (arbitrator only).

        Uses reentrancy guard to prevent double-resolution attacks.

        Args:
            escrow_address: Escrow contract address
            recipient_address: Who should receive the funds
            arbitrator_address: Arbitrator's wallet address

        Returns:
            TransactionResult: Result of resolution transaction

        TODO: Implement actual dispute resolution
        """
        # SECURITY FIX: Reentrancy guard
        escrow_lock = self._get_escrow_lock(escrow_address)

        with escrow_lock:
            self._check_reentrancy(escrow_address, "resolve_dispute")
            try:
                logger.info(
                    f"Resolving dispute on {escrow_address}: funds to {recipient_address}"
                )

                # TODO: Implement dispute resolution
                # escrow = self._get_escrow(escrow_address)
                # tx = escrow.functions.resolveDispute(recipient_address).build_transaction({
                #     "from": arbitrator_address,
                #     "gas": 100000,
                # })
                # ... sign and send ...

                # STUB
                return TransactionResult(
                    success=True,
                    tx_hash=f"0x{'0' * 64}",
                )
            finally:
                self._clear_reentrancy(escrow_address, "resolve_dispute")

    # =========================================================================
    # Utilities
    # =========================================================================

    def get_usdc_balance(self, address: str) -> Decimal:
        """Get USDC balance for an address.

        Args:
            address: Wallet address

        Returns:
            USDC balance

        TODO: Implement actual balance check
        """
        # TODO: Implement balance check
        # usdc = self._get_usdc()
        # balance_wei = usdc.functions.balanceOf(address).call()
        # return self._wei_to_usdc(balance_wei)

        return Decimal("0.00")

    def get_usdc_allowance(
        self,
        owner: str,
        spender: str,
    ) -> Decimal:
        """Get USDC allowance.

        Args:
            owner: Token owner address
            spender: Approved spender address

        Returns:
            Approved USDC amount

        TODO: Implement actual allowance check
        """
        # TODO: Implement allowance check
        # usdc = self._get_usdc()
        # allowance_wei = usdc.functions.allowance(owner, spender).call()
        # return self._wei_to_usdc(allowance_wei)

        return Decimal("0.00")
