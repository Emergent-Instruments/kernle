"""
Job service for Kernle Commerce.

Implements the job marketplace business logic including:
- Job creation and lifecycle management
- State machine with transition validation
- Application handling (apply, accept, reject)
- Delivery and approval workflow
- Dispute resolution

Job State Machine:
    open → funded → accepted → delivered → completed
            ↓         ↓           ↓
        cancelled  disputed    disputed
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import threading
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
import logging
import uuid

from kernle.commerce.config import get_config, CommerceConfig
from kernle.commerce.jobs.models import (
    Job,
    JobApplication,
    JobStatus,
    ApplicationStatus,
    JobStateTransition,
    VALID_JOB_TRANSITIONS,
)
from kernle.commerce.jobs.storage import JobStorage


logger = logging.getLogger(__name__)


class JobServiceError(Exception):
    """Base exception for job service errors."""
    pass


class JobNotFoundError(JobServiceError):
    """Job not found."""
    pass


class ApplicationNotFoundError(JobServiceError):
    """Application not found."""
    pass


class InvalidTransitionError(JobServiceError):
    """Invalid job state transition."""
    pass


class UnauthorizedError(JobServiceError):
    """Actor is not authorized for this operation."""
    pass


class DuplicateApplicationError(JobServiceError):
    """Agent has already applied to this job."""
    pass


class JobExpiredError(JobServiceError):
    """Job deadline has passed."""
    pass


@dataclass
class JobSearchFilters:
    """Filters for job search.
    
    Attributes:
        query: Free text search in title/description
        skills: Required skills to match
        min_budget: Minimum budget in USDC
        max_budget: Maximum budget in USDC
        status: Filter by job status
        client_id: Filter by client (poster)
        worker_id: Filter by assigned worker
        deadline_after: Only jobs with deadline after this time
        limit: Maximum results to return
        offset: Pagination offset
    """
    query: Optional[str] = None
    skills: Optional[List[str]] = None
    min_budget: Optional[float] = None
    max_budget: Optional[float] = None
    status: Optional[JobStatus] = None
    client_id: Optional[str] = None
    worker_id: Optional[str] = None
    deadline_after: Optional[datetime] = None
    limit: int = 100
    offset: int = 0


class JobService:
    """Service for job marketplace operations.
    
    Manages the complete job lifecycle from creation to completion.
    Enforces the state machine and authorization rules.
    
    Example:
        >>> service = JobService(storage)
        >>> job = service.create_job(
        ...     client_id="agent_client",
        ...     title="Research Task",
        ...     description="Research X topic",
        ...     budget_usdc=50.0,
        ...     deadline=datetime.now() + timedelta(days=7),
        ...     skills_required=["research"],
        ... )
        >>> service.fund_job(job.id, "agent_client", escrow_address="0x...")
        >>> applications = service.list_applications(job_id=job.id)
        >>> service.accept_application(job.id, applications[0].id, "agent_client")
    """
    
    # Maximum allowed budget in USDC (prevent integer overflow)
    MAX_BUDGET_USDC = 1_000_000_000  # $1 billion
    MIN_BUDGET_USDC = 0.01  # 1 cent minimum
    
    def __init__(
        self,
        storage: JobStorage,
        config: Optional[CommerceConfig] = None,
    ):
        """Initialize job service.
        
        Args:
            storage: Job storage backend
            config: Commerce configuration (uses global config if not provided)
        """
        self.storage = storage
        self.config = config or get_config()
        # Lock for atomic operations (prevents race conditions)
        self._job_locks: Dict[str, threading.Lock] = {}
        self._lock_manager = threading.Lock()
    
    def _get_job_lock(self, job_id: str) -> threading.Lock:
        """Get or create a lock for a specific job.
        
        Uses double-checked locking pattern for thread safety.
        """
        if job_id not in self._job_locks:
            with self._lock_manager:
                if job_id not in self._job_locks:
                    self._job_locks[job_id] = threading.Lock()
        return self._job_locks[job_id]
    
    def _now(self) -> datetime:
        """Get current UTC timestamp."""
        return datetime.now(timezone.utc)
    
    def _generate_id(self) -> str:
        """Generate a new UUID."""
        return str(uuid.uuid4())
    
    def _is_authorized_arbitrator(self, actor_id: str) -> bool:
        """Check if the actor is an authorized arbitrator.
        
        Args:
            actor_id: Agent/user ID to check
            
        Returns:
            True if authorized to resolve disputes
        """
        # Check against configured arbitrator address
        if self.config.arbitrator_address and actor_id == self.config.arbitrator_address:
            return True
        
        # Check against list of authorized arbitrators (if configured)
        authorized_arbitrators = getattr(self.config, 'authorized_arbitrators', [])
        if actor_id in authorized_arbitrators:
            return True
        
        # System actor is always authorized (for auto-resolution)
        if actor_id == "system":
            return True
        
        return False
    
    def _transition_job(
        self,
        job: Job,
        new_status: JobStatus,
        actor_id: str,
        tx_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Job:
        """Execute a job state transition with validation.
        
        Args:
            job: The job to transition
            new_status: Target status
            actor_id: Agent/user performing the transition
            tx_hash: Optional blockchain transaction hash
            metadata: Optional additional context
            
        Returns:
            Job: Updated job
            
        Raises:
            InvalidTransitionError: If transition is not valid
        """
        if not job.can_transition_to(new_status):
            raise InvalidTransitionError(
                f"Cannot transition job from {job.status} to {new_status.value}"
            )
        
        old_status = job.status
        now = self._now()
        
        # Update job status
        job.status = new_status.value
        job.updated_at = now
        
        # Set timestamp for specific transitions
        if new_status == JobStatus.FUNDED:
            job.funded_at = now
        elif new_status == JobStatus.ACCEPTED:
            job.accepted_at = now
        elif new_status == JobStatus.DELIVERED:
            job.delivered_at = now
        elif new_status == JobStatus.COMPLETED:
            job.completed_at = now
        
        # Save job
        self.storage.update_job(job)
        
        # Record transition
        transition = JobStateTransition(
            id=self._generate_id(),
            job_id=job.id,
            from_status=old_status,
            to_status=new_status.value,
            actor_id=actor_id,
            tx_hash=tx_hash,
            metadata=metadata or {},
            created_at=now,
        )
        self.storage.save_transition(transition)
        
        logger.info(
            f"Job {job.id} transitioned: {old_status} → {new_status.value} "
            f"(actor: {actor_id})"
        )
        
        return job
    
    # =========================================================================
    # Job CRUD
    # =========================================================================
    
    def create_job(
        self,
        client_id: str,
        title: str,
        description: str,
        budget_usdc: float,
        deadline: datetime,
        skills_required: Optional[List[str]] = None,
    ) -> Job:
        """Create a new job listing.
        
        Args:
            client_id: Agent ID of the job poster
            title: Short job title (max 200 chars)
            description: Full job description
            budget_usdc: Payment amount in USDC
            deadline: When the work must be delivered
            skills_required: Optional list of required skill names
            
        Returns:
            Job: The newly created job
            
        Raises:
            JobServiceError: If validation fails
        """
        now = self._now()
        
        # Validate deadline is in the future
        if deadline <= now:
            raise JobServiceError("Deadline must be in the future")
        
        # SECURITY FIX: Budget bounds checking to prevent overflow
        if budget_usdc <= 0:
            raise JobServiceError("Budget must be positive")
        if budget_usdc < self.MIN_BUDGET_USDC:
            raise JobServiceError(f"Budget must be at least ${self.MIN_BUDGET_USDC}")
        if budget_usdc > self.MAX_BUDGET_USDC:
            raise JobServiceError(f"Budget cannot exceed ${self.MAX_BUDGET_USDC:,}")
        
        job = Job(
            id=self._generate_id(),
            client_id=client_id,
            title=title,
            description=description,
            budget_usdc=budget_usdc,
            deadline=deadline,
            skills_required=skills_required or [],
            status=JobStatus.OPEN.value,
            created_at=now,
            updated_at=now,
        )
        
        self.storage.save_job(job)
        
        # Record creation as first transition
        transition = JobStateTransition(
            id=self._generate_id(),
            job_id=job.id,
            from_status=None,
            to_status=JobStatus.OPEN.value,
            actor_id=client_id,
            metadata={"action": "created"},
            created_at=now,
        )
        self.storage.save_transition(transition)
        
        logger.info(f"Job created: {job.id} - {title} by {client_id}")
        return job
    
    def get_job(self, job_id: str) -> Job:
        """Get a job by ID.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job: The job
            
        Raises:
            JobNotFoundError: If job doesn't exist
        """
        job = self.storage.get_job(job_id)
        if not job:
            raise JobNotFoundError(f"Job not found: {job_id}")
        return job
    
    def list_jobs(
        self,
        filters: Optional[JobSearchFilters] = None,
    ) -> List[Job]:
        """List jobs with optional filters.
        
        Args:
            filters: Search filters
            
        Returns:
            List[Job]: Matching jobs
        """
        f = filters or JobSearchFilters()
        return self.storage.list_jobs(
            status=f.status,
            client_id=f.client_id,
            worker_id=f.worker_id,
            skills=f.skills,
            min_budget=f.min_budget,
            max_budget=f.max_budget,
            limit=f.limit,
            offset=f.offset,
        )
    
    def search_jobs(
        self,
        query: Optional[str] = None,
        skills: Optional[List[str]] = None,
        min_budget: Optional[float] = None,
        max_budget: Optional[float] = None,
        limit: int = 50,
    ) -> List[Job]:
        """Search for available jobs (convenience method for workers).
        
        Only returns open, funded jobs with deadlines in the future.
        
        Args:
            query: Text to search in title/description
            skills: Skills to match
            min_budget: Minimum budget filter
            max_budget: Maximum budget filter
            limit: Max results
            
        Returns:
            List[Job]: Matching available jobs
        """
        # Get funded jobs (ready for workers)
        jobs = self.storage.list_jobs(
            status=JobStatus.FUNDED,
            skills=skills,
            min_budget=min_budget,
            max_budget=max_budget,
            limit=limit,
        )
        
        # Also include open jobs (not yet funded)
        open_jobs = self.storage.list_jobs(
            status=JobStatus.OPEN,
            skills=skills,
            min_budget=min_budget,
            max_budget=max_budget,
            limit=limit,
        )
        
        all_jobs = jobs + open_jobs
        
        # Filter by deadline and query
        now = self._now()
        results = []
        for job in all_jobs:
            if job.deadline <= now:
                continue  # Skip expired
            if query:
                # Simple text matching (could be enhanced with FTS)
                q_lower = query.lower()
                if q_lower not in job.title.lower() and q_lower not in job.description.lower():
                    continue
            results.append(job)
        
        # Sort by deadline (soonest first) and return limited
        results.sort(key=lambda j: j.deadline)
        return results[:limit]
    
    # =========================================================================
    # Job Lifecycle
    # =========================================================================
    
    def fund_job(
        self,
        job_id: str,
        actor_id: str,
        escrow_address: str,
        tx_hash: Optional[str] = None,
    ) -> Job:
        """Mark a job as funded (escrow deployed and funded).
        
        Called after the client has deployed and funded the escrow contract.
        
        Args:
            job_id: Job ID
            actor_id: Must be the client
            escrow_address: Deployed escrow contract address
            tx_hash: Funding transaction hash
            
        Returns:
            Job: Updated job
            
        Raises:
            UnauthorizedError: If actor is not the client
            InvalidTransitionError: If job cannot be funded
        """
        job = self.get_job(job_id)
        
        if job.client_id != actor_id:
            raise UnauthorizedError("Only the client can fund the job")
        
        # Validate escrow address
        if not escrow_address.startswith("0x") or len(escrow_address) != 42:
            raise JobServiceError(f"Invalid escrow address: {escrow_address}")
        
        job.escrow_address = escrow_address
        
        return self._transition_job(
            job=job,
            new_status=JobStatus.FUNDED,
            actor_id=actor_id,
            tx_hash=tx_hash,
            metadata={"escrow_address": escrow_address},
        )
    
    def cancel_job(
        self,
        job_id: str,
        actor_id: str,
        reason: Optional[str] = None,
        tx_hash: Optional[str] = None,
    ) -> Job:
        """Cancel a job.
        
        Can only cancel jobs in open, funded, or accepted states.
        If funded, the escrow should be refunded (handled externally).
        
        Args:
            job_id: Job ID
            actor_id: Must be the client
            reason: Optional cancellation reason
            tx_hash: Refund transaction hash (if funded)
            
        Returns:
            Job: Cancelled job
        """
        job = self.get_job(job_id)
        
        if job.client_id != actor_id:
            raise UnauthorizedError("Only the client can cancel the job")
        
        return self._transition_job(
            job=job,
            new_status=JobStatus.CANCELLED,
            actor_id=actor_id,
            tx_hash=tx_hash,
            metadata={"reason": reason} if reason else {},
        )
    
    # =========================================================================
    # Applications
    # =========================================================================
    
    def apply_to_job(
        self,
        job_id: str,
        applicant_id: str,
        message: str,
        proposed_deadline: Optional[datetime] = None,
    ) -> JobApplication:
        """Apply to work on a job.
        
        Args:
            job_id: Job ID to apply to
            applicant_id: Agent ID of the applicant
            message: Application message
            proposed_deadline: Optional alternative deadline
            
        Returns:
            JobApplication: The created application
            
        Raises:
            JobNotFoundError: If job doesn't exist
            JobServiceError: If job is not accepting applications
            DuplicateApplicationError: If already applied
        """
        job = self.get_job(job_id)
        
        # Check job is accepting applications
        if job.status not in {JobStatus.OPEN.value, JobStatus.FUNDED.value}:
            raise JobServiceError(
                f"Job is not accepting applications (status: {job.status})"
            )
        
        # Check deadline hasn't passed
        if job.deadline <= self._now():
            raise JobExpiredError(f"Job deadline has passed: {job.deadline}")
        
        # Check for duplicate application
        existing = self.storage.list_applications(
            job_id=job_id,
            applicant_id=applicant_id,
        )
        if existing:
            raise DuplicateApplicationError(
                f"Agent {applicant_id} has already applied to job {job_id}"
            )
        
        # Prevent self-application
        if applicant_id == job.client_id:
            raise JobServiceError("Cannot apply to your own job")
        
        application = JobApplication(
            id=self._generate_id(),
            job_id=job_id,
            applicant_id=applicant_id,
            message=message,
            status=ApplicationStatus.PENDING.value,
            proposed_deadline=proposed_deadline,
            created_at=self._now(),
        )
        
        self.storage.save_application(application)
        logger.info(f"Application {application.id} created for job {job_id}")
        
        return application
    
    def get_application(self, application_id: str) -> JobApplication:
        """Get an application by ID.
        
        Args:
            application_id: Application ID
            
        Returns:
            JobApplication: The application
            
        Raises:
            ApplicationNotFoundError: If not found
        """
        app = self.storage.get_application(application_id)
        if not app:
            raise ApplicationNotFoundError(f"Application not found: {application_id}")
        return app
    
    def list_applications(
        self,
        job_id: Optional[str] = None,
        applicant_id: Optional[str] = None,
        status: Optional[ApplicationStatus] = None,
    ) -> List[JobApplication]:
        """List applications with optional filters.
        
        Args:
            job_id: Filter by job
            applicant_id: Filter by applicant
            status: Filter by status
            
        Returns:
            List[JobApplication]: Matching applications
        """
        return self.storage.list_applications(
            job_id=job_id,
            applicant_id=applicant_id,
            status=status,
        )
    
    def accept_application(
        self,
        job_id: str,
        application_id: str,
        actor_id: str,
    ) -> tuple[Job, JobApplication]:
        """Accept an application and assign the worker to the job.
        
        Uses locking to prevent race conditions where multiple applications
        could be accepted simultaneously.
        
        Args:
            job_id: Job ID
            application_id: Application ID to accept
            actor_id: Must be the client
            
        Returns:
            Tuple of (Job, JobApplication): Updated job and application
            
        Raises:
            UnauthorizedError: If actor is not the client
            InvalidTransitionError: If job cannot accept workers
        """
        # SECURITY FIX: Use per-job lock to prevent race condition
        # This ensures only one application can be accepted at a time
        job_lock = self._get_job_lock(job_id)
        
        with job_lock:
            # Re-fetch job inside lock to ensure we have latest state
            job = self.get_job(job_id)
            application = self.get_application(application_id)
            
            if job.client_id != actor_id:
                raise UnauthorizedError("Only the client can accept applications")
            
            if application.job_id != job_id:
                raise JobServiceError("Application does not belong to this job")
            
            if not application.is_pending:
                raise JobServiceError(
                    f"Application is not pending (status: {application.status})"
                )
            
            # Job must be funded to accept (escrow must exist)
            if job.status != JobStatus.FUNDED.value:
                raise InvalidTransitionError(
                    f"Job must be funded before accepting workers (status: {job.status})"
                )
            
            # Assign worker
            job.worker_id = application.applicant_id
            
            # Update application status
            application.status = ApplicationStatus.ACCEPTED.value
            self.storage.update_application_status(
                application_id,
                ApplicationStatus.ACCEPTED,
            )
            
            # Reject other pending applications
            other_apps = self.storage.list_applications(job_id=job_id)
            for other in other_apps:
                if other.id != application_id and other.is_pending:
                    self.storage.update_application_status(
                        other.id,
                        ApplicationStatus.REJECTED,
                    )
            
            # Transition job
            job = self._transition_job(
                job=job,
                new_status=JobStatus.ACCEPTED,
                actor_id=actor_id,
                metadata={
                    "application_id": application_id,
                    "worker_id": application.applicant_id,
                },
            )
            
            return job, application
    
    def reject_application(
        self,
        application_id: str,
        actor_id: str,
        reason: Optional[str] = None,
    ) -> JobApplication:
        """Reject an application.
        
        Args:
            application_id: Application ID
            actor_id: Must be the job client
            reason: Optional rejection reason
            
        Returns:
            JobApplication: Updated application
        """
        application = self.get_application(application_id)
        job = self.get_job(application.job_id)
        
        if job.client_id != actor_id:
            raise UnauthorizedError("Only the client can reject applications")
        
        if not application.is_pending:
            raise JobServiceError("Application is not pending")
        
        application.status = ApplicationStatus.REJECTED.value
        self.storage.update_application_status(
            application_id,
            ApplicationStatus.REJECTED,
        )
        
        logger.info(f"Application {application_id} rejected (reason: {reason})")
        return application
    
    def withdraw_application(
        self,
        application_id: str,
        actor_id: str,
    ) -> JobApplication:
        """Withdraw an application.
        
        Args:
            application_id: Application ID
            actor_id: Must be the applicant
            
        Returns:
            JobApplication: Updated application
        """
        application = self.get_application(application_id)
        
        if application.applicant_id != actor_id:
            raise UnauthorizedError("Only the applicant can withdraw")
        
        if not application.is_pending:
            raise JobServiceError("Can only withdraw pending applications")
        
        application.status = ApplicationStatus.WITHDRAWN.value
        self.storage.update_application_status(
            application_id,
            ApplicationStatus.WITHDRAWN,
        )
        
        logger.info(f"Application {application_id} withdrawn by {actor_id}")
        return application
    
    # =========================================================================
    # Delivery & Completion
    # =========================================================================
    
    @staticmethod
    def _validate_deliverable_url(url: str) -> None:
        """Validate deliverable URL format and scheme.
        
        Args:
            url: URL to validate
            
        Raises:
            JobServiceError: If URL is invalid or uses unsafe scheme
        """
        if not url or not isinstance(url, str):
            raise JobServiceError("Deliverable URL is required")
        
        # Length check
        if len(url) > 2000:
            raise JobServiceError("Deliverable URL too long (max 2000 characters)")
        
        try:
            parsed = urlparse(url)
        except Exception:
            raise JobServiceError("Invalid URL format")
        
        # Only allow safe schemes
        allowed_schemes = {'http', 'https', 'ipfs', 'ipns', 'ar'}  # ar = Arweave
        if parsed.scheme.lower() not in allowed_schemes:
            raise JobServiceError(
                f"Invalid URL scheme: {parsed.scheme}. "
                f"Allowed: {', '.join(sorted(allowed_schemes))}"
            )
        
        # Must have a valid netloc for http/https
        if parsed.scheme in {'http', 'https'} and not parsed.netloc:
            raise JobServiceError("Invalid URL: missing host")
    
    def deliver_job(
        self,
        job_id: str,
        actor_id: str,
        deliverable_url: str,
        deliverable_hash: Optional[str] = None,
    ) -> Job:
        """Submit a deliverable for the job.
        
        Args:
            job_id: Job ID
            actor_id: Must be the assigned worker
            deliverable_url: URL to the deliverable (IPFS, GitHub, etc.)
            deliverable_hash: Optional content hash for verification
            
        Returns:
            Job: Updated job
            
        Raises:
            UnauthorizedError: If actor is not the worker
            InvalidTransitionError: If job is not in accepted state
            JobServiceError: If URL is invalid
        """
        # SECURITY FIX: Validate deliverable URL
        self._validate_deliverable_url(deliverable_url)
        
        job = self.get_job(job_id)
        
        if job.worker_id != actor_id:
            raise UnauthorizedError("Only the assigned worker can deliver")
        
        if job.status != JobStatus.ACCEPTED.value:
            raise InvalidTransitionError(
                f"Can only deliver accepted jobs (status: {job.status})"
            )
        
        job.deliverable_url = deliverable_url
        job.deliverable_hash = deliverable_hash
        
        return self._transition_job(
            job=job,
            new_status=JobStatus.DELIVERED,
            actor_id=actor_id,
            metadata={
                "deliverable_url": deliverable_url,
                "deliverable_hash": deliverable_hash,
            },
        )
    
    def approve_job(
        self,
        job_id: str,
        actor_id: str,
        tx_hash: Optional[str] = None,
    ) -> Job:
        """Approve the deliverable and release payment.
        
        This should be called after the escrow release transaction.
        
        Args:
            job_id: Job ID
            actor_id: Must be the client
            tx_hash: Payment release transaction hash
            
        Returns:
            Job: Completed job
            
        Raises:
            UnauthorizedError: If actor is not the client
            InvalidTransitionError: If job is not delivered
        """
        job = self.get_job(job_id)
        
        if job.client_id != actor_id:
            raise UnauthorizedError("Only the client can approve")
        
        if job.status != JobStatus.DELIVERED.value:
            raise InvalidTransitionError(
                f"Can only approve delivered jobs (status: {job.status})"
            )
        
        return self._transition_job(
            job=job,
            new_status=JobStatus.COMPLETED,
            actor_id=actor_id,
            tx_hash=tx_hash,
            metadata={"action": "approved"},
        )
    
    def auto_approve_job(
        self,
        job_id: str,
        tx_hash: Optional[str] = None,
    ) -> Job:
        """Auto-approve a job after timeout period.
        
        Called by a background job when approval timeout expires.
        
        Args:
            job_id: Job ID
            tx_hash: Auto-release transaction hash
            
        Returns:
            Job: Completed job
        """
        job = self.get_job(job_id)
        
        if job.status != JobStatus.DELIVERED.value:
            raise InvalidTransitionError("Job is not in delivered state")
        
        # Check timeout
        if not job.delivered_at:
            raise JobServiceError("Job has no delivery timestamp")
        
        timeout = timedelta(days=self.config.approval_timeout_days)
        if self._now() < job.delivered_at + timeout:
            raise JobServiceError("Approval timeout has not expired")
        
        return self._transition_job(
            job=job,
            new_status=JobStatus.COMPLETED,
            actor_id="system",  # System-initiated
            tx_hash=tx_hash,
            metadata={"action": "auto_approved", "reason": "timeout"},
        )
    
    # =========================================================================
    # Disputes
    # =========================================================================
    
    def dispute_job(
        self,
        job_id: str,
        actor_id: str,
        reason: str,
    ) -> Job:
        """Raise a dispute on a job.
        
        Can be raised by client or worker during accepted or delivered states.
        
        Args:
            job_id: Job ID
            actor_id: Client or worker
            reason: Dispute reason
            
        Returns:
            Job: Disputed job
        """
        job = self.get_job(job_id)
        
        # Check authorization (must be client or worker)
        if actor_id not in {job.client_id, job.worker_id}:
            raise UnauthorizedError(
                "Only the client or worker can raise a dispute"
            )
        
        # Check valid state for dispute
        if job.status not in {JobStatus.ACCEPTED.value, JobStatus.DELIVERED.value}:
            raise InvalidTransitionError(
                f"Cannot dispute job in status: {job.status}"
            )
        
        return self._transition_job(
            job=job,
            new_status=JobStatus.DISPUTED,
            actor_id=actor_id,
            metadata={"reason": reason, "raised_by": actor_id},
        )
    
    def resolve_dispute(
        self,
        job_id: str,
        actor_id: str,  # Arbitrator
        resolution: str,  # "worker" or "client"
        tx_hash: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Job:
        """Resolve a disputed job.
        
        Called by arbitrator to release funds to the appropriate party.
        
        Args:
            job_id: Job ID
            actor_id: Arbitrator ID (must be authorized)
            resolution: Who gets the funds ("worker" or "client")
            tx_hash: Resolution transaction hash
            notes: Arbitrator notes
            
        Returns:
            Job: Completed job
            
        Raises:
            UnauthorizedError: If actor is not an authorized arbitrator
        """
        job = self.get_job(job_id)
        
        if job.status != JobStatus.DISPUTED.value:
            raise InvalidTransitionError("Job is not disputed")
        
        # SECURITY FIX: Verify actor is authorized arbitrator
        if not self._is_authorized_arbitrator(actor_id):
            raise UnauthorizedError(
                "Only authorized arbitrators can resolve disputes"
            )
        
        if resolution not in {"worker", "client"}:
            raise JobServiceError(
                f"Invalid resolution: {resolution}. Must be 'worker' or 'client'"
            )
        
        return self._transition_job(
            job=job,
            new_status=JobStatus.COMPLETED,
            actor_id=actor_id,
            tx_hash=tx_hash,
            metadata={
                "action": "dispute_resolved",
                "resolution": resolution,
                "notes": notes,
            },
        )
    
    # =========================================================================
    # Helpers
    # =========================================================================
    
    def get_job_history(self, job_id: str) -> List[JobStateTransition]:
        """Get the state transition history for a job.
        
        Args:
            job_id: Job ID
            
        Returns:
            List[JobStateTransition]: Ordered transition history
        """
        _ = self.get_job(job_id)  # Verify job exists
        return self.storage.get_transitions(job_id)
    
    def get_jobs_for_client(self, client_id: str) -> List[Job]:
        """Get all jobs posted by a client.
        
        Args:
            client_id: Client agent ID
            
        Returns:
            List[Job]: Jobs posted by this client
        """
        return self.storage.list_jobs(client_id=client_id)
    
    def get_jobs_for_worker(self, worker_id: str) -> List[Job]:
        """Get all jobs assigned to a worker.
        
        Args:
            worker_id: Worker agent ID
            
        Returns:
            List[Job]: Jobs assigned to this worker
        """
        return self.storage.list_jobs(worker_id=worker_id)
    
    def get_applications_for_applicant(
        self,
        applicant_id: str,
    ) -> List[JobApplication]:
        """Get all applications by an applicant.
        
        Args:
            applicant_id: Applicant agent ID
            
        Returns:
            List[JobApplication]: Applications by this agent
        """
        return self.storage.list_applications(applicant_id=applicant_id)
