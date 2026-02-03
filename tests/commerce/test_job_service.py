"""Tests for job service."""

from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pytest

from kernle.commerce.config import CommerceConfig
from kernle.commerce.jobs.models import (
    ApplicationStatus,
    Job,
    JobApplication,
    JobStateTransition,
    JobStatus,
)
from kernle.commerce.jobs.service import (
    DuplicateApplicationError,
    InvalidTransitionError,
    JobNotFoundError,
    JobService,
    JobServiceError,
    UnauthorizedError,
)


class InMemoryJobStorage:
    """In-memory job storage for testing."""

    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.applications: dict[str, JobApplication] = {}
        self.transitions: dict[str, List[JobStateTransition]] = {}

    def save_job(self, job: Job) -> str:
        self.jobs[job.id] = job
        return job.id

    def get_job(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        client_id: Optional[str] = None,
        worker_id: Optional[str] = None,
        skills: Optional[List[str]] = None,
        min_budget: Optional[float] = None,
        max_budget: Optional[float] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Job]:
        result = list(self.jobs.values())

        if status:
            result = [j for j in result if j.status == status.value]
        if client_id:
            result = [j for j in result if j.client_id == client_id]
        if worker_id:
            result = [j for j in result if j.worker_id == worker_id]
        if skills:
            result = [j for j in result if any(s in j.skills_required for s in skills)]
        if min_budget:
            result = [j for j in result if j.budget_usdc >= min_budget]
        if max_budget:
            result = [j for j in result if j.budget_usdc <= max_budget]

        return result[offset:offset + limit]

    def update_job(self, job: Job) -> bool:
        if job.id not in self.jobs:
            return False
        self.jobs[job.id] = job
        return True

    def save_application(self, application: JobApplication) -> str:
        self.applications[application.id] = application
        return application.id

    def get_application(self, application_id: str) -> Optional[JobApplication]:
        return self.applications.get(application_id)

    def list_applications(
        self,
        job_id: Optional[str] = None,
        applicant_id: Optional[str] = None,
        status: Optional[ApplicationStatus] = None,
        limit: int = 100,
    ) -> List[JobApplication]:
        result = list(self.applications.values())

        if job_id:
            result = [a for a in result if a.job_id == job_id]
        if applicant_id:
            result = [a for a in result if a.applicant_id == applicant_id]
        if status:
            result = [a for a in result if a.status == status.value]

        return result[:limit]

    def update_application_status(
        self,
        application_id: str,
        status: ApplicationStatus,
    ) -> bool:
        app = self.applications.get(application_id)
        if not app:
            return False
        app.status = status.value
        return True

    def save_transition(self, transition: JobStateTransition) -> str:
        if transition.job_id not in self.transitions:
            self.transitions[transition.job_id] = []
        self.transitions[transition.job_id].append(transition)
        return transition.id

    def get_transitions(self, job_id: str) -> List[JobStateTransition]:
        return self.transitions.get(job_id, [])


@pytest.fixture
def storage():
    """Create in-memory storage for testing."""
    return InMemoryJobStorage()


@pytest.fixture
def config():
    """Create test configuration."""
    return CommerceConfig(approval_timeout_days=7)


@pytest.fixture
def service(storage, config):
    """Create job service for testing."""
    return JobService(storage=storage, config=config)


def future_deadline(days: int = 7) -> datetime:
    """Helper to create a future deadline."""
    return datetime.now(timezone.utc) + timedelta(days=days)


class TestJobCreation:
    """Tests for job creation."""

    def test_create_job_basic(self, service, storage):
        """Test creating a job with minimal parameters."""
        job = service.create_job(
            client_id="client-1",
            title="Research Task",
            description="Research AI agents",
            budget_usdc=50.0,
            deadline=future_deadline(7),
        )

        assert job.client_id == "client-1"
        assert job.title == "Research Task"
        assert job.status == "open"
        assert job.budget_usdc == 50.0
        assert job.worker_id is None
        assert job.created_at is not None

        # Verify saved to storage
        saved = storage.get_job(job.id)
        assert saved is not None
        assert saved.id == job.id

        # Verify transition was recorded
        transitions = storage.get_transitions(job.id)
        assert len(transitions) == 1
        assert transitions[0].to_status == "open"

    def test_create_job_with_skills(self, service):
        """Test creating a job with required skills."""
        job = service.create_job(
            client_id="client-2",
            title="Coding Task",
            description="Build an API",
            budget_usdc=200.0,
            deadline=future_deadline(14),
            skills_required=["coding", "automation"],
        )

        assert job.skills_required == ["coding", "automation"]

    def test_create_job_past_deadline_fails(self, service):
        """Test that past deadlines are rejected."""
        past = datetime.now(timezone.utc) - timedelta(days=1)

        with pytest.raises(JobServiceError, match="future"):
            service.create_job(
                client_id="client-3",
                title="Late Job",
                description="This should fail",
                budget_usdc=10.0,
                deadline=past,
            )

    def test_create_job_zero_budget_fails(self, service):
        """Test that zero budget is rejected."""
        with pytest.raises(JobServiceError, match="positive"):
            service.create_job(
                client_id="client-4",
                title="Free Job",
                description="This should fail",
                budget_usdc=0,
                deadline=future_deadline(7),
            )


class TestJobRetrieval:
    """Tests for job retrieval."""

    def test_get_job_by_id(self, service):
        """Test getting job by ID."""
        created = service.create_job(
            client_id="client",
            title="Test Job",
            description="Test",
            budget_usdc=10.0,
            deadline=future_deadline(),
        )

        fetched = service.get_job(created.id)
        assert fetched.id == created.id

    def test_get_job_not_found(self, service):
        """Test getting non-existent job raises error."""
        with pytest.raises(JobNotFoundError, match="not found"):
            service.get_job("nonexistent-id")

    def test_list_jobs_by_client(self, service):
        """Test listing jobs by client."""
        service.create_job(
            client_id="client-A",
            title="Job A",
            description="Test",
            budget_usdc=10.0,
            deadline=future_deadline(),
        )
        service.create_job(
            client_id="client-B",
            title="Job B",
            description="Test",
            budget_usdc=20.0,
            deadline=future_deadline(),
        )

        a_jobs = service.get_jobs_for_client("client-A")
        assert len(a_jobs) == 1
        assert a_jobs[0].client_id == "client-A"


class TestJobFunding:
    """Tests for job funding flow."""

    def test_fund_job_success(self, service, storage):
        """Test successfully funding a job."""
        job = service.create_job(
            client_id="client-fund",
            title="To Fund",
            description="Test",
            budget_usdc=100.0,
            deadline=future_deadline(),
        )

        funded = service.fund_job(
            job_id=job.id,
            actor_id="client-fund",
            escrow_address="0x1234567890123456789012345678901234567890",
        )

        assert funded.status == "funded"
        assert funded.escrow_address == "0x1234567890123456789012345678901234567890"
        assert funded.funded_at is not None

        # Check transition was recorded
        transitions = storage.get_transitions(job.id)
        assert any(t.to_status == "funded" for t in transitions)

    def test_fund_job_wrong_client_fails(self, service):
        """Test that non-client cannot fund job."""
        job = service.create_job(
            client_id="client-real",
            title="Someone else's job",
            description="Test",
            budget_usdc=50.0,
            deadline=future_deadline(),
        )

        with pytest.raises(UnauthorizedError, match="Only the client"):
            service.fund_job(
                job_id=job.id,
                actor_id="not-the-client",
                escrow_address="0x1234567890123456789012345678901234567890",
            )

    def test_fund_job_invalid_address_fails(self, service):
        """Test that invalid escrow address is rejected."""
        job = service.create_job(
            client_id="client",
            title="Bad address",
            description="Test",
            budget_usdc=50.0,
            deadline=future_deadline(),
        )

        with pytest.raises(JobServiceError, match="Invalid escrow"):
            service.fund_job(
                job_id=job.id,
                actor_id="client",
                escrow_address="not-an-address",
            )


class TestJobApplications:
    """Tests for job applications."""

    def test_apply_to_job_success(self, service, storage):
        """Test successfully applying to a job."""
        job = service.create_job(
            client_id="client",
            title="Open Job",
            description="Test",
            budget_usdc=75.0,
            deadline=future_deadline(),
        )

        app = service.apply_to_job(
            job_id=job.id,
            applicant_id="worker-1",
            message="I'd like to work on this!",
        )

        assert app.job_id == job.id
        assert app.applicant_id == "worker-1"
        assert app.status == "pending"
        assert app.message == "I'd like to work on this!"

        # Verify saved
        saved = storage.get_application(app.id)
        assert saved is not None

    def test_apply_duplicate_fails(self, service):
        """Test that duplicate applications are rejected."""
        job = service.create_job(
            client_id="client",
            title="Job",
            description="Test",
            budget_usdc=50.0,
            deadline=future_deadline(),
        )

        service.apply_to_job(
            job_id=job.id,
            applicant_id="worker-dup",
            message="First application",
        )

        with pytest.raises(DuplicateApplicationError):
            service.apply_to_job(
                job_id=job.id,
                applicant_id="worker-dup",
                message="Second application",
            )

    def test_apply_to_own_job_fails(self, service):
        """Test that client cannot apply to their own job."""
        job = service.create_job(
            client_id="self-worker",
            title="My Job",
            description="Test",
            budget_usdc=50.0,
            deadline=future_deadline(),
        )

        with pytest.raises(JobServiceError, match="own job"):
            service.apply_to_job(
                job_id=job.id,
                applicant_id="self-worker",
                message="I'll do it myself",
            )

    def test_apply_to_non_open_job_fails(self, service):
        """Test that closed jobs reject applications."""
        job = service.create_job(
            client_id="client",
            title="Job",
            description="Test",
            budget_usdc=50.0,
            deadline=future_deadline(),
        )
        service.fund_job(
            job_id=job.id,
            actor_id="client",
            escrow_address="0x1234567890123456789012345678901234567890",
        )
        # Have a worker apply and be accepted
        app = service.apply_to_job(
            job_id=job.id,
            applicant_id="worker-1",
            message="First",
        )
        service.accept_application(job.id, app.id, "client")

        # Job is now accepted, shouldn't accept applications
        with pytest.raises(JobServiceError, match="not accepting"):
            service.apply_to_job(
                job_id=job.id,
                applicant_id="worker-2",
                message="Too late",
            )


class TestApplicationAcceptance:
    """Tests for accepting applications."""

    def test_accept_application_success(self, service, storage):
        """Test successfully accepting an application."""
        job = service.create_job(
            client_id="client",
            title="Job",
            description="Test",
            budget_usdc=50.0,
            deadline=future_deadline(),
        )
        service.fund_job(
            job_id=job.id,
            actor_id="client",
            escrow_address="0x1234567890123456789012345678901234567890",
        )
        app = service.apply_to_job(
            job_id=job.id,
            applicant_id="worker",
            message="Please hire me",
        )

        updated_job, updated_app = service.accept_application(
            job_id=job.id,
            application_id=app.id,
            actor_id="client",
        )

        assert updated_job.status == "accepted"
        assert updated_job.worker_id == "worker"
        assert updated_job.accepted_at is not None
        assert updated_app.status == "accepted"

    def test_accept_rejects_other_applications(self, service, storage):
        """Test that accepting one rejects others."""
        job = service.create_job(
            client_id="client",
            title="Popular Job",
            description="Test",
            budget_usdc=500.0,
            deadline=future_deadline(),
        )
        service.fund_job(
            job_id=job.id,
            actor_id="client",
            escrow_address="0x1234567890123456789012345678901234567890",
        )

        app1 = service.apply_to_job(job.id, "worker-1", "I'm first")
        app2 = service.apply_to_job(job.id, "worker-2", "I'm second")

        service.accept_application(job.id, app1.id, "client")

        # Check app2 was rejected
        rejected = storage.get_application(app2.id)
        assert rejected.status == "rejected"

    def test_accept_unfunded_job_fails(self, service):
        """Test that accepting on unfunded job fails."""
        job = service.create_job(
            client_id="client",
            title="Unfunded",
            description="Test",
            budget_usdc=50.0,
            deadline=future_deadline(),
        )
        app = service.apply_to_job(job.id, "worker", "Please")

        with pytest.raises(InvalidTransitionError, match="funded"):
            service.accept_application(job.id, app.id, "client")


class TestDeliveryAndApproval:
    """Tests for delivery and approval flow."""

    def _setup_accepted_job(self, service) -> tuple[Job, JobApplication]:
        """Helper to set up a job in accepted state."""
        job = service.create_job(
            client_id="client",
            title="Work Job",
            description="Do the work",
            budget_usdc=100.0,
            deadline=future_deadline(),
        )
        service.fund_job(
            job_id=job.id,
            actor_id="client",
            escrow_address="0x1234567890123456789012345678901234567890",
        )
        app = service.apply_to_job(job.id, "worker", "I'll do it")
        job, app = service.accept_application(job.id, app.id, "client")
        return job, app

    def test_deliver_job_success(self, service):
        """Test successfully delivering a job."""
        job, _ = self._setup_accepted_job(service)

        delivered = service.deliver_job(
            job_id=job.id,
            actor_id="worker",
            deliverable_url="https://github.com/example/repo",
            deliverable_hash="Qm123abc",
        )

        assert delivered.status == "delivered"
        assert delivered.deliverable_url == "https://github.com/example/repo"
        assert delivered.deliverable_hash == "Qm123abc"
        assert delivered.delivered_at is not None

    def test_deliver_by_non_worker_fails(self, service):
        """Test that non-worker cannot deliver."""
        job, _ = self._setup_accepted_job(service)

        with pytest.raises(UnauthorizedError, match="Only the assigned worker"):
            service.deliver_job(
                job_id=job.id,
                actor_id="not-the-worker",
                deliverable_url="https://example.com",
            )

    def test_approve_job_success(self, service):
        """Test successfully approving a job."""
        job, _ = self._setup_accepted_job(service)
        service.deliver_job(
            job_id=job.id,
            actor_id="worker",
            deliverable_url="https://example.com",
        )

        completed = service.approve_job(
            job_id=job.id,
            actor_id="client",
        )

        assert completed.status == "completed"
        assert completed.completed_at is not None

    def test_approve_undelivered_fails(self, service):
        """Test that approving undelivered job fails."""
        job, _ = self._setup_accepted_job(service)

        with pytest.raises(InvalidTransitionError, match="delivered"):
            service.approve_job(job.id, "client")


class TestDisputes:
    """Tests for dispute handling."""

    def _setup_delivered_job(self, service) -> Job:
        """Helper to set up a delivered job."""
        job = service.create_job(
            client_id="client",
            title="Disputed Job",
            description="Work",
            budget_usdc=100.0,
            deadline=future_deadline(),
        )
        service.fund_job(
            job_id=job.id,
            actor_id="client",
            escrow_address="0x1234567890123456789012345678901234567890",
        )
        app = service.apply_to_job(job.id, "worker", "I'll do it")
        job, _ = service.accept_application(job.id, app.id, "client")
        job = service.deliver_job(
            job_id=job.id,
            actor_id="worker",
            deliverable_url="https://example.com",
        )
        return job

    def test_dispute_by_client(self, service):
        """Test client raising a dispute."""
        job = self._setup_delivered_job(service)

        disputed = service.dispute_job(
            job_id=job.id,
            actor_id="client",
            reason="Work is incomplete",
        )

        assert disputed.status == "disputed"

    def test_dispute_by_worker(self, service):
        """Test worker raising a dispute."""
        job = self._setup_delivered_job(service)

        disputed = service.dispute_job(
            job_id=job.id,
            actor_id="worker",
            reason="Client not responding",
        )

        assert disputed.status == "disputed"

    def test_dispute_by_third_party_fails(self, service):
        """Test that third party cannot dispute."""
        job = self._setup_delivered_job(service)

        with pytest.raises(UnauthorizedError, match="client or worker"):
            service.dispute_job(
                job_id=job.id,
                actor_id="random-agent",
                reason="I don't like this job",
            )

    def test_resolve_dispute_for_worker(self, service):
        """Test resolving dispute in worker's favor."""
        job = self._setup_delivered_job(service)
        service.dispute_job(job.id, "client", "Disputed")

        # Use "system" as actor since it's an authorized arbitrator
        resolved = service.resolve_dispute(
            job_id=job.id,
            actor_id="system",  # system is an authorized arbitrator
            resolution="worker",
        )

        assert resolved.status == "completed"


class TestJobHistory:
    """Tests for job state history."""

    def test_get_job_history(self, service):
        """Test getting complete job history."""
        job = service.create_job(
            client_id="client",
            title="History Job",
            description="Test",
            budget_usdc=50.0,
            deadline=future_deadline(),
        )
        service.fund_job(
            job_id=job.id,
            actor_id="client",
            escrow_address="0x1234567890123456789012345678901234567890",
        )

        history = service.get_job_history(job.id)

        assert len(history) >= 2  # Created + funded
        statuses = [t.to_status for t in history]
        assert "open" in statuses
        assert "funded" in statuses


class TestJobCancellation:
    """Tests for job cancellation."""

    def test_cancel_open_job(self, service):
        """Test cancelling an open job."""
        job = service.create_job(
            client_id="client",
            title="To Cancel",
            description="Test",
            budget_usdc=50.0,
            deadline=future_deadline(),
        )

        cancelled = service.cancel_job(
            job_id=job.id,
            actor_id="client",
            reason="Changed my mind",
        )

        assert cancelled.status == "cancelled"

    def test_cancel_by_non_client_fails(self, service):
        """Test that non-client cannot cancel."""
        job = service.create_job(
            client_id="client",
            title="My Job",
            description="Test",
            budget_usdc=50.0,
            deadline=future_deadline(),
        )

        with pytest.raises(UnauthorizedError, match="Only the client"):
            service.cancel_job(job.id, "not-client")
