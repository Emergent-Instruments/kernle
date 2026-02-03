"""Tests for job data models."""

from datetime import datetime, timedelta, timezone

import pytest

from kernle.commerce.jobs.models import (
    VALID_JOB_TRANSITIONS,
    ApplicationStatus,
    Job,
    JobApplication,
    JobStateTransition,
    JobStatus,
)


class TestJob:
    """Tests for Job dataclass."""

    def test_create_basic_job(self):
        """Test creating a job with minimal required fields."""
        deadline = datetime.now(timezone.utc) + timedelta(days=7)
        job = Job(
            id="job-123",
            client_id="agent-456",
            title="Research cryptocurrency regulations",
            description="Analyze current crypto regulations in EU and US",
            budget_usdc=50.0,
            deadline=deadline,
        )

        assert job.id == "job-123"
        assert job.client_id == "agent-456"
        assert job.title == "Research cryptocurrency regulations"
        assert job.budget_usdc == 50.0
        assert job.status == "open"
        assert job.worker_id is None
        assert job.skills_required == []

    def test_create_job_with_all_fields(self):
        """Test creating a job with all fields."""
        now = datetime.now(timezone.utc)
        deadline = now + timedelta(days=7)

        job = Job(
            id="job-123",
            client_id="agent-456",
            title="Research cryptocurrency regulations",
            description="Analyze current crypto regulations",
            budget_usdc=100.0,
            deadline=deadline,
            worker_id="agent-789",
            skills_required=["research", "writing"],
            escrow_address="0x1234567890abcdef1234567890abcdef12345678",
            status="accepted",
            deliverable_url="https://example.com/report",
            deliverable_hash="QmXyzHash",
            created_at=now,
            updated_at=now,
            funded_at=now,
            accepted_at=now,
        )

        assert job.worker_id == "agent-789"
        assert job.skills_required == ["research", "writing"]
        assert job.escrow_address == "0x1234567890abcdef1234567890abcdef12345678"
        assert job.status == "accepted"

    def test_invalid_budget(self):
        """Test that non-positive budgets are rejected."""
        deadline = datetime.now(timezone.utc) + timedelta(days=7)

        with pytest.raises(ValueError, match="Budget must be positive"):
            Job(
                id="job-123",
                client_id="agent-456",
                title="Test job",
                description="Test description",
                budget_usdc=0.0,
                deadline=deadline,
            )

        with pytest.raises(ValueError, match="Budget must be positive"):
            Job(
                id="job-124",
                client_id="agent-456",
                title="Test job",
                description="Test description",
                budget_usdc=-10.0,
                deadline=deadline,
            )

    def test_invalid_status(self):
        """Test that invalid statuses are rejected."""
        deadline = datetime.now(timezone.utc) + timedelta(days=7)

        with pytest.raises(ValueError, match="Invalid status"):
            Job(
                id="job-123",
                client_id="agent-456",
                title="Test job",
                description="Test description",
                budget_usdc=50.0,
                deadline=deadline,
                status="invalid_status",
            )

    def test_title_too_long(self):
        """Test that titles over 200 chars are rejected."""
        deadline = datetime.now(timezone.utc) + timedelta(days=7)
        long_title = "A" * 201

        with pytest.raises(ValueError, match="Title too long"):
            Job(
                id="job-123",
                client_id="agent-456",
                title=long_title,
                description="Test description",
                budget_usdc=50.0,
                deadline=deadline,
            )

    def test_status_enum_value(self):
        """Test that status can be set via enum."""
        deadline = datetime.now(timezone.utc) + timedelta(days=7)

        job = Job(
            id="job-123",
            client_id="agent-456",
            title="Test job",
            description="Test description",
            budget_usdc=50.0,
            deadline=deadline,
            status=JobStatus.FUNDED,
        )

        assert job.status == "funded"

    def test_can_transition_to(self):
        """Test state transition validation."""
        deadline = datetime.now(timezone.utc) + timedelta(days=7)

        # Open job can transition to funded or cancelled
        job = Job(
            id="job-123",
            client_id="agent-456",
            title="Test job",
            description="Test description",
            budget_usdc=50.0,
            deadline=deadline,
            status="open",
        )

        assert job.can_transition_to(JobStatus.FUNDED) is True
        assert job.can_transition_to(JobStatus.CANCELLED) is True
        assert job.can_transition_to(JobStatus.ACCEPTED) is False
        assert job.can_transition_to(JobStatus.COMPLETED) is False

        # Funded job can transition to accepted or cancelled
        job.status = "funded"
        assert job.can_transition_to(JobStatus.ACCEPTED) is True
        assert job.can_transition_to(JobStatus.CANCELLED) is True
        assert job.can_transition_to(JobStatus.OPEN) is False

    def test_is_open(self):
        """Test is_open property."""
        deadline = datetime.now(timezone.utc) + timedelta(days=7)

        job = Job(
            id="job-123",
            client_id="agent-456",
            title="Test job",
            description="Test description",
            budget_usdc=50.0,
            deadline=deadline,
            status="open",
        )

        assert job.is_open is True

        job.status = "funded"
        assert job.is_open is False

    def test_is_active(self):
        """Test is_active property."""
        deadline = datetime.now(timezone.utc) + timedelta(days=7)

        # Active jobs
        for status in ["open", "funded", "accepted", "delivered", "disputed"]:
            job = Job(
                id="job-123",
                client_id="agent-456",
                title="Test job",
                description="Test description",
                budget_usdc=50.0,
                deadline=deadline,
                status=status,
            )
            assert job.is_active is True, f"Status {status} should be active"

        # Terminal jobs
        for status in ["completed", "cancelled"]:
            job = Job(
                id="job-124",
                client_id="agent-456",
                title="Test job",
                description="Test description",
                budget_usdc=50.0,
                deadline=deadline,
                status=status,
            )
            assert job.is_active is False, f"Status {status} should not be active"

    def test_is_terminal(self):
        """Test is_terminal property."""
        deadline = datetime.now(timezone.utc) + timedelta(days=7)

        job = Job(
            id="job-123",
            client_id="agent-456",
            title="Test job",
            description="Test description",
            budget_usdc=50.0,
            deadline=deadline,
            status="completed",
        )

        assert job.is_terminal is True

        job.status = "open"
        assert job.is_terminal is False

    def test_to_dict(self):
        """Test serialization to dictionary."""
        now = datetime.now(timezone.utc)
        deadline = now + timedelta(days=7)

        job = Job(
            id="job-123",
            client_id="agent-456",
            title="Test job",
            description="Test description",
            budget_usdc=50.0,
            deadline=deadline,
            skills_required=["research", "writing"],
            created_at=now,
        )

        d = job.to_dict()

        assert d["id"] == "job-123"
        assert d["client_id"] == "agent-456"
        assert d["title"] == "Test job"
        assert d["budget_usdc"] == 50.0
        assert d["skills_required"] == ["research", "writing"]
        assert d["created_at"] == now.isoformat()
        assert d["worker_id"] is None

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "id": "job-123",
            "client_id": "agent-456",
            "title": "Test job",
            "description": "Test description",
            "budget_usdc": 75.50,
            "deadline": "2024-02-15T12:00:00+00:00",
            "skills_required": ["coding", "automation"],
            "status": "funded",
        }

        job = Job.from_dict(data)

        assert job.id == "job-123"
        assert job.budget_usdc == 75.50
        assert job.skills_required == ["coding", "automation"]
        assert job.status == "funded"
        assert job.deadline is not None


class TestJobApplication:
    """Tests for JobApplication dataclass."""

    def test_create_basic_application(self):
        """Test creating an application with minimal fields."""
        app = JobApplication(
            id="app-123",
            job_id="job-456",
            applicant_id="agent-789",
            message="I would like to work on this job because...",
        )

        assert app.id == "app-123"
        assert app.job_id == "job-456"
        assert app.applicant_id == "agent-789"
        assert app.status == "pending"
        assert app.proposed_deadline is None

    def test_create_application_with_all_fields(self):
        """Test creating an application with all fields."""
        now = datetime.now(timezone.utc)
        proposed = now + timedelta(days=5)

        app = JobApplication(
            id="app-123",
            job_id="job-456",
            applicant_id="agent-789",
            message="I would like to work on this job",
            status="accepted",
            proposed_deadline=proposed,
            created_at=now,
        )

        assert app.status == "accepted"
        assert app.proposed_deadline == proposed
        assert app.created_at == now

    def test_empty_message_rejected(self):
        """Test that empty messages are rejected."""
        with pytest.raises(ValueError, match="message cannot be empty"):
            JobApplication(
                id="app-123",
                job_id="job-456",
                applicant_id="agent-789",
                message="",
            )

        with pytest.raises(ValueError, match="message cannot be empty"):
            JobApplication(
                id="app-124",
                job_id="job-456",
                applicant_id="agent-789",
                message="   ",  # Whitespace only
            )

    def test_invalid_status(self):
        """Test that invalid statuses are rejected."""
        with pytest.raises(ValueError, match="Invalid status"):
            JobApplication(
                id="app-123",
                job_id="job-456",
                applicant_id="agent-789",
                message="Test message",
                status="invalid",
            )

    def test_status_enum_value(self):
        """Test that status can be set via enum."""
        app = JobApplication(
            id="app-123",
            job_id="job-456",
            applicant_id="agent-789",
            message="Test message",
            status=ApplicationStatus.ACCEPTED,
        )

        assert app.status == "accepted"

    def test_is_pending(self):
        """Test is_pending property."""
        app = JobApplication(
            id="app-123",
            job_id="job-456",
            applicant_id="agent-789",
            message="Test message",
            status="pending",
        )

        assert app.is_pending is True

        app.status = "accepted"
        assert app.is_pending is False

    def test_is_accepted(self):
        """Test is_accepted property."""
        app = JobApplication(
            id="app-123",
            job_id="job-456",
            applicant_id="agent-789",
            message="Test message",
            status="accepted",
        )

        assert app.is_accepted is True

        app.status = "rejected"
        assert app.is_accepted is False

    def test_to_dict(self):
        """Test serialization."""
        now = datetime.now(timezone.utc)

        app = JobApplication(
            id="app-123",
            job_id="job-456",
            applicant_id="agent-789",
            message="Test message",
            created_at=now,
        )

        d = app.to_dict()

        assert d["id"] == "app-123"
        assert d["job_id"] == "job-456"
        assert d["applicant_id"] == "agent-789"
        assert d["message"] == "Test message"
        assert d["status"] == "pending"
        assert d["created_at"] == now.isoformat()

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "id": "app-123",
            "job_id": "job-456",
            "applicant_id": "agent-789",
            "message": "Test message",
            "status": "accepted",
            "created_at": "2024-01-15T12:00:00Z",
        }

        app = JobApplication.from_dict(data)

        assert app.id == "app-123"
        assert app.status == "accepted"
        assert app.created_at is not None


class TestJobStateTransition:
    """Tests for JobStateTransition dataclass."""

    def test_create_transition(self):
        """Test creating a state transition."""
        now = datetime.now(timezone.utc)

        transition = JobStateTransition(
            id="trans-123",
            job_id="job-456",
            from_status="open",
            to_status="funded",
            actor_id="agent-789",
            tx_hash="0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
            metadata={"reason": "Escrow funded"},
            created_at=now,
        )

        assert transition.id == "trans-123"
        assert transition.job_id == "job-456"
        assert transition.from_status == "open"
        assert transition.to_status == "funded"
        assert transition.tx_hash is not None
        assert transition.metadata == {"reason": "Escrow funded"}

    def test_initial_transition(self):
        """Test creating an initial transition (from_status is None)."""
        transition = JobStateTransition(
            id="trans-123",
            job_id="job-456",
            to_status="open",
            actor_id="agent-789",
        )

        assert transition.from_status is None
        assert transition.to_status == "open"

    def test_to_dict(self):
        """Test serialization."""
        now = datetime.now(timezone.utc)

        transition = JobStateTransition(
            id="trans-123",
            job_id="job-456",
            from_status="open",
            to_status="funded",
            actor_id="agent-789",
            metadata={"amount": 100},
            created_at=now,
        )

        d = transition.to_dict()

        assert d["id"] == "trans-123"
        assert d["job_id"] == "job-456"
        assert d["from_status"] == "open"
        assert d["to_status"] == "funded"
        assert d["metadata"] == {"amount": 100}

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "id": "trans-123",
            "job_id": "job-456",
            "from_status": "accepted",
            "to_status": "delivered",
            "actor_id": "agent-789",
            "tx_hash": None,
            "metadata": {},
            "created_at": "2024-01-15T12:00:00+00:00",
        }

        transition = JobStateTransition.from_dict(data)

        assert transition.id == "trans-123"
        assert transition.to_status == "delivered"
        assert transition.created_at is not None


class TestJobStatus:
    """Tests for JobStatus enum."""

    def test_all_statuses_defined(self):
        """Test that all expected statuses are defined."""
        statuses = {s.value for s in JobStatus}
        expected = {"open", "funded", "accepted", "delivered", "completed", "disputed", "cancelled"}
        assert statuses == expected


class TestApplicationStatus:
    """Tests for ApplicationStatus enum."""

    def test_all_statuses_defined(self):
        """Test that all expected statuses are defined."""
        statuses = {s.value for s in ApplicationStatus}
        expected = {"pending", "accepted", "rejected", "withdrawn"}
        assert statuses == expected


class TestValidJobTransitions:
    """Tests for VALID_JOB_TRANSITIONS state machine."""

    def test_open_transitions(self):
        """Test valid transitions from open state."""
        assert VALID_JOB_TRANSITIONS[JobStatus.OPEN] == {JobStatus.FUNDED, JobStatus.CANCELLED}

    def test_funded_transitions(self):
        """Test valid transitions from funded state."""
        assert VALID_JOB_TRANSITIONS[JobStatus.FUNDED] == {JobStatus.ACCEPTED, JobStatus.CANCELLED}

    def test_accepted_transitions(self):
        """Test valid transitions from accepted state."""
        assert VALID_JOB_TRANSITIONS[JobStatus.ACCEPTED] == {
            JobStatus.DELIVERED,
            JobStatus.DISPUTED,
            JobStatus.CANCELLED,
        }

    def test_delivered_transitions(self):
        """Test valid transitions from delivered state."""
        assert VALID_JOB_TRANSITIONS[JobStatus.DELIVERED] == {
            JobStatus.COMPLETED,
            JobStatus.DISPUTED,
        }

    def test_disputed_transitions(self):
        """Test valid transitions from disputed state."""
        assert VALID_JOB_TRANSITIONS[JobStatus.DISPUTED] == {JobStatus.COMPLETED}

    def test_terminal_states(self):
        """Test that terminal states have no transitions."""
        assert VALID_JOB_TRANSITIONS[JobStatus.COMPLETED] == set()
        assert VALID_JOB_TRANSITIONS[JobStatus.CANCELLED] == set()
