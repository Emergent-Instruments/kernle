"""
Jobs marketplace data models.

Job represents a work listing in the marketplace.
JobApplication represents an agent's application to work on a job.
JobStateTransition tracks the audit trail of job status changes.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class JobStatus(str, Enum):
    """Job lifecycle status."""

    OPEN = "open"  # Job posted, accepting applications
    FUNDED = "funded"  # Escrow funded, ready for worker
    ACCEPTED = "accepted"  # Worker accepted and assigned
    DELIVERED = "delivered"  # Worker submitted deliverable
    COMPLETED = "completed"  # Client approved, payment released
    DISPUTED = "disputed"  # Dispute raised, pending resolution
    CANCELLED = "cancelled"  # Job cancelled (refund if funded)


class ApplicationStatus(str, Enum):
    """Job application status."""

    PENDING = "pending"  # Awaiting client review
    ACCEPTED = "accepted"  # Client accepted, applicant is now worker
    REJECTED = "rejected"  # Client declined
    WITHDRAWN = "withdrawn"  # Applicant withdrew


# Valid state transitions for the job state machine
VALID_JOB_TRANSITIONS = {
    JobStatus.OPEN: {JobStatus.FUNDED, JobStatus.CANCELLED},
    JobStatus.FUNDED: {JobStatus.ACCEPTED, JobStatus.CANCELLED},
    JobStatus.ACCEPTED: {JobStatus.DELIVERED, JobStatus.DISPUTED, JobStatus.CANCELLED},
    JobStatus.DELIVERED: {JobStatus.COMPLETED, JobStatus.DISPUTED},
    JobStatus.DISPUTED: {JobStatus.COMPLETED},  # After arbitration
    JobStatus.COMPLETED: set(),  # Terminal state
    JobStatus.CANCELLED: set(),  # Terminal state
}


@dataclass
class Job:
    """A job listing in the marketplace.

    Jobs are created by clients (agents or humans) who need work done.
    Workers (agents) apply to jobs, and accepted workers complete the
    work in exchange for USDC payment via escrow.

    State Machine:
        open → funded → accepted → delivered → completed
                 ↓         ↓           ↓
             cancelled  disputed    disputed

    Attributes:
        id: Unique job ID (UUID)
        client_id: Agent ID of the job poster
        title: Short job title (max 200 chars)
        description: Full job description
        budget_usdc: Payment amount in USDC
        deadline: When the work must be delivered
        worker_id: Agent ID of accepted worker (None until accepted)
        skills_required: List of required skill names
        escrow_address: Deployed escrow contract address
        status: Current job status
        deliverable_url: URL to delivered work (set by worker)
        deliverable_hash: IPFS or content hash of deliverable
        created_at: Job creation timestamp
        updated_at: Last modification timestamp
        funded_at: When escrow was funded
        accepted_at: When worker was accepted
        delivered_at: When deliverable was submitted
        completed_at: When payment was released
    """

    id: str
    client_id: str
    title: str
    description: str
    budget_usdc: float
    deadline: datetime
    worker_id: Optional[str] = None
    skills_required: List[str] = field(default_factory=list)
    escrow_address: Optional[str] = None
    status: str = "open"
    deliverable_url: Optional[str] = None
    deliverable_hash: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    funded_at: Optional[datetime] = None
    accepted_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def __post_init__(self):
        """Validate job data."""
        # Normalize status
        if isinstance(self.status, JobStatus):
            self.status = self.status.value

        # Validate status
        valid_statuses = {s.value for s in JobStatus}
        if self.status not in valid_statuses:
            raise ValueError(f"Invalid status: {self.status}. Must be one of {valid_statuses}")

        # Validate budget
        if self.budget_usdc <= 0:
            raise ValueError(f"Budget must be positive, got {self.budget_usdc}")

        # Validate title: non-empty after strip, max 200 chars
        if not self.title or not self.title.strip():
            raise ValueError("Job title cannot be empty")
        if len(self.title) > 200:
            raise ValueError(f"Title too long: {len(self.title)} chars (max 200)")

        # Validate description: non-empty after strip
        if not self.description or not self.description.strip():
            raise ValueError("Job description cannot be empty")

        # Validate deadline: must be in future for new jobs
        # Only check for truly new jobs (status='open' and created_at is None)
        # Jobs loaded from storage may have past deadlines which is valid
        if self.status == "open" and self.created_at is None and self.deadline:
            from datetime import timezone
            now = datetime.now(timezone.utc)
            # Make deadline timezone-aware if naive
            deadline = self.deadline
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
            if deadline <= now:
                raise ValueError("Job deadline must be in the future")

        # Validate escrow_address format when present
        if self.escrow_address is not None:
            self._validate_eth_address(self.escrow_address, "escrow_address")

    @staticmethod
    def _validate_eth_address(address: str, field_name: str) -> None:
        """Validate Ethereum address format: 0x + 40 hex chars = 42 chars total."""
        if not address:
            raise ValueError(f"{field_name} cannot be empty")
        if len(address) != 42:
            raise ValueError(f"{field_name} must be exactly 42 characters, got {len(address)}")
        if not address.startswith("0x"):
            raise ValueError(f"{field_name} must start with '0x'")
        # Check remaining 40 chars are valid hex
        hex_part = address[2:]
        if not all(c in "0123456789abcdefABCDEF" for c in hex_part):
            raise ValueError(f"{field_name} must contain only hex characters after '0x'")

    def can_transition_to(self, new_status: JobStatus) -> bool:
        """Check if a status transition is valid."""
        current = JobStatus(self.status)
        return new_status in VALID_JOB_TRANSITIONS.get(current, set())

    @property
    def is_open(self) -> bool:
        """Check if job is accepting applications."""
        return self.status == JobStatus.OPEN.value

    @property
    def is_active(self) -> bool:
        """Check if job is in progress (not terminal state)."""
        return self.status not in {JobStatus.COMPLETED.value, JobStatus.CANCELLED.value}

    @property
    def is_terminal(self) -> bool:
        """Check if job is in a terminal state."""
        return self.status in {JobStatus.COMPLETED.value, JobStatus.CANCELLED.value}

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "client_id": self.client_id,
            "title": self.title,
            "description": self.description,
            "budget_usdc": self.budget_usdc,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "worker_id": self.worker_id,
            "skills_required": self.skills_required,
            "escrow_address": self.escrow_address,
            "status": self.status,
            "deliverable_url": self.deliverable_url,
            "deliverable_hash": self.deliverable_hash,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "funded_at": self.funded_at.isoformat() if self.funded_at else None,
            "accepted_at": self.accepted_at.isoformat() if self.accepted_at else None,
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Job":
        """Create from dictionary."""
        def parse_dt(val):
            if val and isinstance(val, str):
                return datetime.fromisoformat(val.replace("Z", "+00:00"))
            return val

        return cls(
            id=data["id"],
            client_id=data["client_id"],
            title=data["title"],
            description=data["description"],
            budget_usdc=float(data["budget_usdc"]),
            deadline=parse_dt(data["deadline"]),
            worker_id=data.get("worker_id"),
            skills_required=data.get("skills_required", []),
            escrow_address=data.get("escrow_address"),
            status=data.get("status", "open"),
            deliverable_url=data.get("deliverable_url"),
            deliverable_hash=data.get("deliverable_hash"),
            created_at=parse_dt(data.get("created_at")),
            updated_at=parse_dt(data.get("updated_at")),
            funded_at=parse_dt(data.get("funded_at")),
            accepted_at=parse_dt(data.get("accepted_at")),
            delivered_at=parse_dt(data.get("delivered_at")),
            completed_at=parse_dt(data.get("completed_at")),
        )


@dataclass
class JobApplication:
    """An application to work on a job.

    Agents apply to jobs with a message explaining why they're suited
    for the work. The client reviews applications and accepts one worker.

    Attributes:
        id: Unique application ID (UUID)
        job_id: ID of the job being applied to
        applicant_id: Agent ID of the applicant
        message: Application message explaining qualifications
        status: Application status (pending, accepted, rejected, withdrawn)
        proposed_deadline: Optional alternative deadline proposed by applicant
        created_at: When the application was submitted
    """

    id: str
    job_id: str
    applicant_id: str
    message: str
    status: str = "pending"
    proposed_deadline: Optional[datetime] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        """Validate application data."""
        # Normalize status
        if isinstance(self.status, ApplicationStatus):
            self.status = self.status.value

        # Validate status
        valid_statuses = {s.value for s in ApplicationStatus}
        if self.status not in valid_statuses:
            raise ValueError(f"Invalid status: {self.status}. Must be one of {valid_statuses}")

        # Validate message
        if not self.message or len(self.message.strip()) == 0:
            raise ValueError("Application message cannot be empty")

    @property
    def is_pending(self) -> bool:
        """Check if application is awaiting review."""
        return self.status == ApplicationStatus.PENDING.value

    @property
    def is_accepted(self) -> bool:
        """Check if application was accepted."""
        return self.status == ApplicationStatus.ACCEPTED.value

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "job_id": self.job_id,
            "applicant_id": self.applicant_id,
            "message": self.message,
            "status": self.status,
            "proposed_deadline": (
                self.proposed_deadline.isoformat() if self.proposed_deadline else None
            ),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "JobApplication":
        """Create from dictionary."""
        def parse_dt(val):
            if val and isinstance(val, str):
                return datetime.fromisoformat(val.replace("Z", "+00:00"))
            return val

        return cls(
            id=data["id"],
            job_id=data["job_id"],
            applicant_id=data["applicant_id"],
            message=data["message"],
            status=data.get("status", "pending"),
            proposed_deadline=parse_dt(data.get("proposed_deadline")),
            created_at=parse_dt(data.get("created_at")),
        )


@dataclass
class JobStateTransition:
    """Audit log entry for job state changes.

    Every job status change is logged for accountability and debugging.
    This provides a complete history of the job lifecycle.

    Attributes:
        id: Unique transition ID (UUID)
        job_id: ID of the job
        from_status: Previous status (None for initial creation)
        to_status: New status
        actor_id: Agent ID of who triggered the transition
        tx_hash: Blockchain transaction hash (if applicable)
        metadata: Additional context (e.g., reason for dispute)
        created_at: When the transition occurred
    """

    id: str
    job_id: str
    to_status: str
    actor_id: str
    from_status: Optional[str] = None
    tx_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "job_id": self.job_id,
            "from_status": self.from_status,
            "to_status": self.to_status,
            "actor_id": self.actor_id,
            "tx_hash": self.tx_hash,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "JobStateTransition":
        """Create from dictionary."""
        created_at = data.get("created_at")
        if created_at and isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

        return cls(
            id=data["id"],
            job_id=data["job_id"],
            from_status=data.get("from_status"),
            to_status=data["to_status"],
            actor_id=data["actor_id"],
            tx_hash=data.get("tx_hash"),
            metadata=data.get("metadata", {}),
            created_at=created_at,
        )
