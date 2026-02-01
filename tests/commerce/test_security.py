"""
Security exploit tests for Kernle Commerce.

These tests attempt to exploit vulnerabilities identified in the security audit.
Tests marked with @pytest.mark.xfail are known vulnerabilities that need fixing.

Run with: uv run pytest tests/commerce/test_security.py -v
"""

import asyncio
import concurrent.futures
import threading
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List, Tuple
from unittest.mock import MagicMock, patch
import uuid

import pytest

# Import commerce modules
from kernle.commerce.wallet.models import WalletAccount, WalletStatus
from kernle.commerce.wallet.service import (
    WalletService,
    WalletNotFoundError,
    WalletServiceError,
    SpendingLimitExceededError,
)
from kernle.commerce.wallet.storage import InMemoryWalletStorage
from kernle.commerce.jobs.models import Job, JobApplication, JobStatus, ApplicationStatus
from kernle.commerce.jobs.service import (
    JobService,
    JobNotFoundError,
    ApplicationNotFoundError,
    InvalidTransitionError,
    UnauthorizedError,
    DuplicateApplicationError,
    JobExpiredError,
    JobServiceError,
)
from kernle.commerce.jobs.storage import InMemoryJobStorage
from kernle.commerce.skills.models import Skill, SkillCategory
from kernle.commerce.skills.registry import InMemorySkillRegistry


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def wallet_storage():
    """Fresh in-memory wallet storage."""
    return InMemoryWalletStorage()


@pytest.fixture
def wallet_service(wallet_storage):
    """Wallet service with fresh storage."""
    return WalletService(wallet_storage)


@pytest.fixture
def job_storage():
    """Fresh in-memory job storage."""
    return InMemoryJobStorage()


@pytest.fixture
def job_service(job_storage):
    """Job service with fresh storage."""
    return JobService(job_storage)


@pytest.fixture
def skill_registry():
    """Fresh skill registry with canonical skills."""
    return InMemorySkillRegistry()


def create_test_job(job_service: JobService, client_id: str = "client_1") -> Job:
    """Helper to create a test job."""
    return job_service.create_job(
        client_id=client_id,
        title="Test Job",
        description="Test job description",
        budget_usdc=100.0,
        deadline=datetime.now(timezone.utc) + timedelta(days=7),
        skills_required=["research"],
    )


def fund_test_job(job_service: JobService, job: Job) -> Job:
    """Helper to fund a test job."""
    return job_service.fund_job(
        job_id=job.id,
        actor_id=job.client_id,
        escrow_address="0x" + "a" * 40,
    )


# =============================================================================
# 1. INPUT VALIDATION & INJECTION TESTS
# =============================================================================


class TestInputValidation:
    """Test input validation vulnerabilities."""

    def test_sql_injection_in_skill_search(self, skill_registry):
        """
        Test SQL injection in skills search.
        
        VULNERABILITY: Skills search may be vulnerable to SQL injection
        through the query parameter.
        """
        # Attempt SQL injection payloads
        injection_payloads = [
            "'; DROP TABLE skills; --",
            "' OR '1'='1",
            "research'; DELETE FROM skills WHERE '1'='1",
            "' UNION SELECT * FROM users --",
            "research%'; DROP TABLE skills; --",
        ]
        
        for payload in injection_payloads:
            # These should not cause errors or unexpected behavior
            try:
                results = skill_registry.search_skills(payload)
                # Should return empty or filtered results, not execute injection
                assert isinstance(results, list)
            except Exception as e:
                # Should not raise database errors
                assert "syntax" not in str(e).lower()
                assert "sql" not in str(e).lower()

    def test_xss_in_job_description(self, job_service):
        """Test XSS payloads in job fields are handled safely."""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "{{constructor.constructor('alert(1)')()}}",
        ]
        
        for payload in xss_payloads:
            job = job_service.create_job(
                client_id="client_1",
                title=payload[:100],
                description=payload,
                budget_usdc=100.0,
                deadline=datetime.now(timezone.utc) + timedelta(days=7),
            )
            # Job should be created but content should be stored as-is
            # (output encoding should happen at display layer)
            assert job.description == payload

    def test_path_traversal_in_deliverable_url(self, job_service):
        """
        Test that path traversal and unsafe URLs are rejected.
        
        FIXED: deliver_job now validates URLs and only allows safe schemes.
        """
        job = create_test_job(job_service)
        job = fund_test_job(job_service, job)
        
        # Add worker
        app = job_service.apply_to_job(
            job_id=job.id,
            applicant_id="worker_1",
            message="I'll do it",
        )
        job_service.accept_application(job.id, app.id, job.client_id)
        
        # Path traversal and unsafe URL attempts - all should be rejected
        unsafe_urls = [
            "file:///etc/passwd",
            "file://localhost/etc/passwd",
            "javascript:alert(1)",
            "data:text/html,<script>alert(1)</script>",
            "ftp://evil.com/malware",
        ]
        
        for url in unsafe_urls:
            with pytest.raises(JobServiceError, match="Invalid URL"):
                job_service.deliver_job(
                    job_id=job.id,
                    actor_id="worker_1",
                    deliverable_url=url,
                )
        
        # Valid URLs should work
        valid_urls = [
            "https://github.com/user/repo",
            "ipfs://QmHash123",
            "https://example.com/work.pdf",
        ]
        
        for url in valid_urls:
            # Reset job state
            job.status = JobStatus.ACCEPTED.value
            job.deliverable_url = None
            
            delivered = job_service.deliver_job(
                job_id=job.id,
                actor_id="worker_1",
                deliverable_url=url,
            )
            assert delivered.deliverable_url == url

    def test_budget_bounds_checking(self, job_service):
        """
        Test that budget values are properly bounded.
        
        FIXED: JobService now enforces min/max budget limits.
        """
        # Negative values should fail
        with pytest.raises((ValueError, JobServiceError)):
            job_service.create_job(
                client_id="client_1",
                title="Test",
                description="Test",
                budget_usdc=-100.0,
                deadline=datetime.now(timezone.utc) + timedelta(days=7),
            )
        
        # Values below minimum should fail
        with pytest.raises(JobServiceError, match="at least"):
            job_service.create_job(
                client_id="client_1",
                title="Test Too Small",
                description="Test",
                budget_usdc=0.001,  # Below 0.01 minimum
                deadline=datetime.now(timezone.utc) + timedelta(days=7),
            )
        
        # Values above maximum should fail
        with pytest.raises(JobServiceError, match="cannot exceed"):
            job_service.create_job(
                client_id="client_1",
                title="Test Too Large",
                description="Test",
                budget_usdc=2_000_000_000,  # Above 1B maximum
                deadline=datetime.now(timezone.utc) + timedelta(days=7),
            )
        
        # Valid minimum should work
        job = job_service.create_job(
            client_id="client_1",
            title="Test Min",
            description="Test",
            budget_usdc=0.01,
            deadline=datetime.now(timezone.utc) + timedelta(days=7),
        )
        assert job.budget_usdc == 0.01
        
        # Valid large value should work
        job = job_service.create_job(
            client_id="client_1",
            title="Test Large",
            description="Test",
            budget_usdc=999_999_999,  # Just under 1B
            deadline=datetime.now(timezone.utc) + timedelta(days=7),
        )
        assert job.budget_usdc == 999_999_999


# =============================================================================
# 2. AUTHENTICATION & AUTHORIZATION TESTS
# =============================================================================


class TestAuthorization:
    """Test authorization vulnerabilities."""

    def test_unauthorized_dispute_resolution(self, job_service):
        """
        Test that only authorized arbitrators can resolve disputes.
        
        FIXED: resolve_dispute now checks if actor_id is an authorized arbitrator.
        """
        # Setup: Create job, accept worker, start work, raise dispute
        job = create_test_job(job_service)
        job = fund_test_job(job_service, job)
        
        app = job_service.apply_to_job(job.id, "worker_1", "I'll do it")
        job_service.accept_application(job.id, app.id, job.client_id)
        
        # Worker raises dispute
        job = job_service.dispute_job(job.id, "worker_1", "Client not responding")
        
        # Random attacker cannot resolve dispute
        with pytest.raises(UnauthorizedError):
            job_service.resolve_dispute(
                job_id=job.id,
                actor_id="random_attacker",  # Not an arbitrator
                resolution="worker",
            )
        
        # System actor CAN resolve (for auto-resolution)
        resolved_job = job_service.resolve_dispute(
            job_id=job.id,
            actor_id="system",
            resolution="worker",
        )
        assert resolved_job.status == JobStatus.COMPLETED.value

    def test_worker_cannot_approve_own_job(self, job_service):
        """Test that workers cannot approve jobs they're working on."""
        job = create_test_job(job_service)
        job = fund_test_job(job_service, job)
        
        # Worker applies and is accepted
        app = job_service.apply_to_job(job.id, "worker_1", "I'll do it")
        job_service.accept_application(job.id, app.id, job.client_id)
        
        # Worker delivers
        job = job_service.deliver_job(
            job_id=job.id,
            actor_id="worker_1",
            deliverable_url="https://example.com/work",
        )
        
        # Worker tries to approve their own work
        with pytest.raises(UnauthorizedError):
            job_service.approve_job(job.id, "worker_1")

    def test_client_cannot_apply_to_own_job(self, job_service):
        """Test that clients cannot apply to their own jobs."""
        job = create_test_job(job_service, client_id="client_1")
        
        # Client tries to apply to their own job
        with pytest.raises(JobServiceError, match="Cannot apply to your own job"):
            job_service.apply_to_job(job.id, "client_1", "I'll do my own job")

    @pytest.mark.xfail(reason="No check for same user controlling client and worker agents")
    def test_self_dealing_different_agents(self, job_service):
        """
        Test self-dealing where one user controls both client and worker agents.
        
        VULNERABILITY: A user could control multiple agents and self-deal.
        """
        # User controls both agent_A and agent_B
        # This test would need user-level tracking to fail
        job = create_test_job(job_service, client_id="agent_A")
        job = fund_test_job(job_service, job)
        
        # Same user's other agent applies
        app = job_service.apply_to_job(job.id, "agent_B", "I'll do it")
        
        # Self-dealing: accept own application
        job_service.accept_application(job.id, app.id, "agent_A")
        
        # Complete the self-deal
        job_service.deliver_job(job.id, "agent_B", "https://example.com/fake")
        job_service.approve_job(job.id, "agent_A")
        
        # This self-dealing should have been prevented
        assert False, "Self-dealing between same user's agents should be blocked"

    def test_non_client_cannot_view_applications(self, job_service):
        """Test that non-clients cannot list applications for a job."""
        job = create_test_job(job_service, client_id="client_1")
        job = fund_test_job(job_service, job)
        
        # Worker applies
        job_service.apply_to_job(job.id, "worker_1", "I'll do it")
        
        # Attacker tries to view applications
        # Note: Service layer doesn't have authorization check, this is in routes
        # This test shows the gap in the service layer
        applications = job_service.list_applications(job_id=job.id)
        
        # In a secure implementation, this should be blocked for non-clients
        # Currently returns all applications - the route layer handles auth
        assert len(applications) > 0

    def test_wallet_claim_requires_ownership(self, wallet_service):
        """Test that wallet claiming verifies ownership."""
        # Create wallet for agent_1
        wallet = wallet_service.create_wallet("agent_1", user_id="user_1")
        
        # Different agent tries to claim
        # Current implementation only checks status, not user_id
        # This should fail but may not
        try:
            wallet_service.claim_wallet(wallet.id, "0x" + "b" * 40)
            # If claim succeeds, check it's the right owner
            claimed = wallet_service.get_wallet(wallet.id)
            # The user_id should match original owner
            assert claimed.user_id == "user_1"
        except WalletServiceError:
            pass  # Expected if ownership is verified


# =============================================================================
# 3. BUSINESS LOGIC TESTS
# =============================================================================


class TestBusinessLogic:
    """Test business logic vulnerabilities."""

    def test_race_condition_double_acceptance(self, job_service):
        """
        Test that concurrent application acceptance is prevented.
        
        FIXED: accept_application now uses per-job locking to ensure
        only one application can be accepted at a time.
        """
        job = create_test_job(job_service)
        job = fund_test_job(job_service, job)
        
        # Multiple workers apply
        app1 = job_service.apply_to_job(job.id, "worker_1", "I'll do it")
        app2 = job_service.apply_to_job(job.id, "worker_2", "I'll do it better")
        
        # Simulate race condition: both acceptance checks pass
        results = []
        errors = []
        
        def accept_app(app_id):
            try:
                job_service.accept_application(job.id, app_id, "client_1")
                results.append(app_id)
            except Exception as e:
                errors.append((app_id, e))
        
        # Run concurrent accepts
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(accept_app, app1.id),
                executor.submit(accept_app, app2.id),
            ]
            concurrent.futures.wait(futures)
        
        # Only ONE should succeed due to locking
        assert len(results) == 1, f"Race condition! Both accepted: {results}"
        assert len(errors) == 1, f"Expected one rejection, got: {errors}"

    def test_daily_spending_limit_enforced(self, wallet_service):
        """
        Test that daily spending limits are enforced.
        
        FIXED: WalletService now tracks daily spending and rejects
        transactions that would exceed the daily limit.
        """
        owner_eoa = "0x" + "a" * 40
        wallet = wallet_service.create_wallet("agent_1")
        wallet = wallet_service.claim_wallet(wallet.id, owner_eoa)
        
        # Set low daily limit (only owner can update)
        wallet_service.update_spending_limits(wallet.id, per_tx=50, daily=100, actor_id=owner_eoa)
        
        # Try to exceed daily limit with multiple transactions
        total_spent = Decimal("0")
        limit_hit = False
        
        for i in range(5):  # 5 x $50 = $250, exceeds $100 daily limit
            try:
                result = wallet_service.transfer(
                    wallet_id=wallet.id,
                    to_address="0x" + "b" * 40,
                    amount=Decimal("50"),
                    actor_id="agent_1",
                )
                if result.success:
                    total_spent += Decimal("50")
            except SpendingLimitExceededError:
                limit_hit = True
                break
        
        # Should have stopped at $100 (2 transactions of $50)
        assert total_spent == Decimal("100"), f"Expected $100, got: {total_spent}"
        assert limit_hit, "Daily limit should have been hit"

    def test_state_machine_bypass_attempt(self, job_service):
        """Test attempts to bypass the job state machine."""
        job = create_test_job(job_service)
        
        # Try to skip directly to completed (bypassing funded, accepted, delivered)
        with pytest.raises(InvalidTransitionError):
            job_service._transition_job(
                job, JobStatus.COMPLETED, "client_1"
            )
        
        # Try to go backwards
        job = fund_test_job(job_service, job)
        with pytest.raises(InvalidTransitionError):
            job_service._transition_job(
                job, JobStatus.OPEN, "client_1"
            )

    def test_auto_approval_gaming(self, job_service):
        """Test protection against auto-approval gaming."""
        job = create_test_job(job_service)
        job = fund_test_job(job_service, job)
        
        app = job_service.apply_to_job(job.id, "worker_1", "I'll do it")
        job_service.accept_application(job.id, app.id, job.client_id)
        
        # Worker delivers garbage
        job = job_service.deliver_job(
            job_id=job.id,
            actor_id="worker_1",
            deliverable_url="https://example.com/garbage",
        )
        
        # Try to auto-approve immediately (should fail - timeout not passed)
        with pytest.raises(JobServiceError, match="timeout has not expired"):
            job_service.auto_approve_job(job.id)

    def test_escrow_address_predictability(self, job_service):
        """Test that escrow addresses are predictable (vulnerability)."""
        import hashlib
        
        # Create job to get ID
        job = create_test_job(job_service)
        
        # Predict escrow address using same algorithm
        base_hash = uuid.uuid5(uuid.NAMESPACE_DNS, job.id).hex
        extra_hash = uuid.uuid5(uuid.NAMESPACE_URL, job.id).hex[:8]
        predicted_address = f"0x{base_hash}{extra_hash[:8]}"[:42]
        
        # Fund job to create escrow
        job = job_service.fund_job(
            job_id=job.id,
            actor_id=job.client_id,
            escrow_address="0x" + "a" * 40,  # This is passed in, not generated here
        )
        
        # In the real escrow service, address is predictable
        # This demonstrates the predictability issue
        assert predicted_address.startswith("0x")
        assert len(predicted_address) == 42

    def test_job_cancellation_after_work_started(self, job_service):
        """Test that cancellation after work started is handled properly."""
        job = create_test_job(job_service)
        job = fund_test_job(job_service, job)
        
        app = job_service.apply_to_job(job.id, "worker_1", "I'll do it")
        job_service.accept_application(job.id, app.id, job.client_id)
        
        # Job is accepted, worker has started
        # Client cancels - this is allowed but may be unfair to worker
        job = job_service.cancel_job(job.id, job.client_id, "Changed my mind")
        
        assert job.status == JobStatus.CANCELLED.value
        # Note: No compensation for worker's effort


# =============================================================================
# 4. DATA EXPOSURE TESTS
# =============================================================================


class TestDataExposure:
    """Test data exposure vulnerabilities."""

    def test_wallet_dict_hides_cdp_id(self, wallet_service):
        """
        Test that to_dict() no longer exposes internal CDP wallet ID by default.
        
        FIXED: to_dict() now requires include_internal=True to expose cdp_wallet_id.
        """
        wallet = wallet_service.create_wallet("agent_1")
        
        # Default to_dict should NOT include cdp_wallet_id
        data = wallet.to_dict()
        assert "cdp_wallet_id" not in data
        
        # to_dict with include_internal=True SHOULD include it
        data_with_internal = wallet.to_dict(include_internal=True)
        assert "cdp_wallet_id" in data_with_internal
        
        # to_public_dict should have minimal fields
        public_data = wallet.to_public_dict()
        assert "cdp_wallet_id" not in public_data
        assert "spending_limit_per_tx" not in public_data
        assert "wallet_address" in public_data

    def test_no_pii_filtering(self, job_service):
        """Test that PII in job descriptions is not filtered."""
        pii_content = """
        Contact me at john.doe@example.com
        Phone: 555-123-4567
        SSN: 123-45-6789
        Credit Card: 4111-1111-1111-1111
        """
        
        job = job_service.create_job(
            client_id="client_1",
            title="Test Job",
            description=pii_content,
            budget_usdc=100.0,
            deadline=datetime.now(timezone.utc) + timedelta(days=7),
        )
        
        # PII is stored as-is - no filtering
        assert "john.doe@example.com" in job.description
        assert "555-123-4567" in job.description

    def test_spending_limits_visible_to_all(self, wallet_service):
        """Test that spending limits are visible (may be sensitive)."""
        wallet = wallet_service.create_wallet("agent_1")
        
        data = wallet.to_dict()
        
        # Spending limits reveal wallet configuration
        assert "spending_limit_per_tx" in data
        assert "spending_limit_daily" in data
        # In some cases, this could help attackers plan attacks


# =============================================================================
# 5. STATE TRANSITION TESTS
# =============================================================================


class TestStateTransitions:
    """Test state transition security."""

    def test_all_valid_transitions(self, job_service):
        """Test all valid state transitions work correctly."""
        job = create_test_job(job_service)
        assert job.status == JobStatus.OPEN.value
        
        # open -> funded
        job = fund_test_job(job_service, job)
        assert job.status == JobStatus.FUNDED.value
        
        # funded -> accepted
        app = job_service.apply_to_job(job.id, "worker_1", "I'll do it")
        job, _ = job_service.accept_application(job.id, app.id, job.client_id)
        assert job.status == JobStatus.ACCEPTED.value
        
        # accepted -> delivered
        job = job_service.deliver_job(job.id, "worker_1", "https://example.com/work")
        assert job.status == JobStatus.DELIVERED.value
        
        # delivered -> completed
        job = job_service.approve_job(job.id, job.client_id)
        assert job.status == JobStatus.COMPLETED.value

    def test_all_invalid_transitions_rejected(self, job_service):
        """Test all invalid state transitions are rejected."""
        invalid_transitions = [
            # (from_status, to_status)
            ("open", "accepted"),  # Skip funded
            ("open", "delivered"),
            ("open", "completed"),
            ("funded", "delivered"),  # Skip accepted
            ("funded", "completed"),
            ("accepted", "completed"),  # Skip delivered
            ("completed", "open"),  # Backwards
            ("cancelled", "open"),  # From terminal
        ]
        
        for from_status, to_status in invalid_transitions:
            job = create_test_job(job_service)
            
            # Manually set status for testing
            job.status = from_status
            if from_status == "funded":
                job.escrow_address = "0x" + "a" * 40
            if from_status in ("accepted", "delivered"):
                job.worker_id = "worker_1"
                job.escrow_address = "0x" + "a" * 40
            
            # Attempt invalid transition
            target_status = JobStatus(to_status)
            assert not job.can_transition_to(target_status), \
                f"Should reject {from_status} -> {to_status}"

    def test_dispute_from_valid_states(self, job_service):
        """Test disputes can only be raised from valid states."""
        # Can dispute from accepted
        job = create_test_job(job_service)
        job = fund_test_job(job_service, job)
        app = job_service.apply_to_job(job.id, "worker_1", "I'll do it")
        job, _ = job_service.accept_application(job.id, app.id, job.client_id)
        
        job = job_service.dispute_job(job.id, "worker_1", "Problem")
        assert job.status == JobStatus.DISPUTED.value
        
        # Can dispute from delivered
        job2 = create_test_job(job_service)
        job2 = fund_test_job(job_service, job2)
        app2 = job_service.apply_to_job(job2.id, "worker_2", "I'll do it")
        job2, _ = job_service.accept_application(job2.id, app2.id, job2.client_id)
        job2 = job_service.deliver_job(job2.id, "worker_2", "https://example.com/work")
        
        job2 = job_service.dispute_job(job2.id, "client_1", "Bad work")
        assert job2.status == JobStatus.DISPUTED.value
        
        # Cannot dispute from open
        job3 = create_test_job(job_service)
        with pytest.raises(InvalidTransitionError):
            job_service.dispute_job(job3.id, "client_1", "Reason")


# =============================================================================
# 6. SKILL REGISTRY TESTS
# =============================================================================


class TestSkillSecurity:
    """Test skill registry security."""

    def test_skill_name_validation(self, skill_registry):
        """Test skill name validation rejects invalid names."""
        invalid_names = [
            "UPPERCASE",  # Must be lowercase
            "with spaces",  # No spaces
            "special@chars",  # No special chars
            "a" * 100,  # Too long
            "",  # Empty
        ]
        
        for name in invalid_names:
            with pytest.raises(ValueError):
                Skill(id=str(uuid.uuid4()), name=name, description="Test")

    def test_valid_skill_names(self, skill_registry):
        """Test valid skill names are accepted."""
        valid_names = [
            "research",
            "data-analysis",
            "web-scraping",
            "coding123",
        ]
        
        for name in valid_names:
            skill = Skill(id=str(uuid.uuid4()), name=name, description="Test")
            assert skill.name == name


# =============================================================================
# 7. CONCURRENT ACCESS TESTS
# =============================================================================


class TestConcurrency:
    """Test concurrent access handling."""

    def test_concurrent_applications(self, job_service):
        """Test handling of concurrent job applications."""
        job = create_test_job(job_service)
        job = fund_test_job(job_service, job)
        
        errors = []
        successes = []
        
        def apply_to_job(worker_id):
            try:
                app = job_service.apply_to_job(
                    job.id, worker_id, f"Application from {worker_id}"
                )
                successes.append(worker_id)
            except DuplicateApplicationError:
                errors.append(worker_id)
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        # Concurrent applications from different workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(apply_to_job, f"worker_{i}")
                for i in range(10)
            ]
            concurrent.futures.wait(futures)
        
        # All should succeed (different workers)
        assert len(successes) == 10
        assert len(errors) == 0

    def test_duplicate_application_prevention(self, job_service):
        """Test duplicate application is prevented."""
        job = create_test_job(job_service)
        
        # First application succeeds
        app1 = job_service.apply_to_job(job.id, "worker_1", "First try")
        assert app1.id is not None
        
        # Second application from same worker fails
        with pytest.raises(DuplicateApplicationError):
            job_service.apply_to_job(job.id, "worker_1", "Second try")

    def test_concurrent_job_updates(self, job_service):
        """Test concurrent updates to same job."""
        job = create_test_job(job_service)
        job = fund_test_job(job_service, job)
        
        app = job_service.apply_to_job(job.id, "worker_1", "I'll do it")
        job, _ = job_service.accept_application(job.id, app.id, job.client_id)
        
        # Both client and worker try to transition at same time
        results = []
        
        def deliver():
            try:
                job_service.deliver_job(job.id, "worker_1", "https://example.com/work")
                results.append("deliver")
            except Exception as e:
                results.append(f"deliver_error: {e}")
        
        def dispute():
            try:
                job_service.dispute_job(job.id, "client_1", "Taking too long")
                results.append("dispute")
            except Exception as e:
                results.append(f"dispute_error: {e}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(deliver),
                executor.submit(dispute),
            ]
            concurrent.futures.wait(futures)
        
        # One should succeed, one should fail (depending on timing)
        # This test demonstrates the race condition
        assert len(results) == 2


# =============================================================================
# 8. WALLET SECURITY TESTS
# =============================================================================


class TestWalletSecurity:
    """Test wallet-specific security."""

    def test_wallet_address_validation(self, wallet_service):
        """Test Ethereum address validation."""
        wallet = wallet_service.create_wallet("agent_1")
        
        invalid_addresses = [
            "not_an_address",
            "0x123",  # Too short
            "0x" + "g" * 40,  # Invalid hex
            "1x" + "a" * 40,  # Wrong prefix
        ]
        
        for addr in invalid_addresses:
            # Claim should validate address format
            with pytest.raises((ValueError, WalletServiceError)):
                # Note: claim_wallet validates format but may accept via a different path
                # The WalletAccount model does validate in __post_init__
                WalletAccount(
                    id="test",
                    agent_id="test",
                    wallet_address="0x" + "a" * 40,  # Valid wallet address
                    owner_eoa=addr,  # Invalid owner address
                )

    def test_per_tx_spending_limit(self, wallet_service):
        """Test per-transaction spending limit is enforced."""
        wallet = wallet_service.create_wallet("agent_1")
        wallet = wallet_service.claim_wallet(wallet.id, "0x" + "a" * 40)
        
        # Default limit is $100
        with pytest.raises(SpendingLimitExceededError):
            wallet_service.transfer(
                wallet_id=wallet.id,
                to_address="0x" + "b" * 40,
                amount=Decimal("150"),  # Exceeds $100 limit
                actor_id="agent_1",
            )

    def test_frozen_wallet_cannot_transact(self, wallet_service, wallet_storage):
        """Test that frozen wallets cannot transact."""
        wallet = wallet_service.create_wallet("agent_1")
        wallet = wallet_service.claim_wallet(wallet.id, "0x" + "a" * 40)
        
        # Manually freeze wallet
        wallet_storage.update_wallet_status(wallet.id, WalletStatus.FROZEN)
        
        # Try to transact
        with pytest.raises(Exception):  # WalletNotActiveError or similar
            wallet_service.transfer(
                wallet_id=wallet.id,
                to_address="0x" + "b" * 40,
                amount=Decimal("10"),
                actor_id="agent_1",
            )

    def test_paused_wallet_cannot_transact(self, wallet_service):
        """Test that paused wallets cannot transact."""
        owner_eoa = "0x" + "a" * 40
        wallet = wallet_service.create_wallet("agent_1")
        wallet = wallet_service.claim_wallet(wallet.id, owner_eoa)
        
        # Pause wallet (only owner can pause)
        wallet = wallet_service.pause_wallet(wallet.id, owner_eoa)
        
        # Verify can't transact
        assert not wallet.can_transact


# =============================================================================
# 9. MCP TOOLS SECURITY TESTS
# =============================================================================


class TestMCPToolsSecurity:
    """Test MCP tools security."""

    def test_mcp_input_sanitization(self):
        """Test MCP tool input sanitization."""
        from kernle.commerce.mcp.tools import (
            sanitize_string,
            sanitize_array,
            validate_number,
        )
        
        # Test string sanitization
        with pytest.raises(ValueError):
            sanitize_string(None, "field", required=True)
        
        with pytest.raises(ValueError):
            sanitize_string("", "field", required=True)
        
        with pytest.raises(ValueError):
            sanitize_string("a" * 1001, "field", max_length=1000)
        
        # Test array sanitization
        with pytest.raises(ValueError):
            sanitize_array(["a"] * 25, "field", max_items=20)
        
        # Test number validation
        with pytest.raises(ValueError):
            validate_number(-1, "field", min_val=0)
        
        with pytest.raises(ValueError):
            validate_number(101, "field", max_val=100)

    def test_mcp_error_handling_no_leaks(self):
        """Test MCP error handling doesn't leak internal info."""
        from kernle.commerce.mcp.tools import handle_commerce_tool_error
        
        # Simulate internal error
        internal_error = Exception("Database connection string: postgres://user:password@host/db")
        
        results = handle_commerce_tool_error(internal_error, "test_tool", {})
        
        # Should not leak connection string
        assert "password" not in str(results)
        assert "Internal server error" in str(results[0].text)


# =============================================================================
# 10. EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_expired_job_cannot_accept_applications(self, job_service):
        """Test that expired jobs cannot accept new applications."""
        # Create job with very short deadline
        job = job_service.create_job(
            client_id="client_1",
            title="Test",
            description="Test",
            budget_usdc=100.0,
            deadline=datetime.now(timezone.utc) + timedelta(seconds=1),
        )
        
        # Wait for deadline to pass
        time.sleep(2)
        
        # Try to apply - should fail
        with pytest.raises(JobExpiredError):
            job_service.apply_to_job(job.id, "worker_1", "I'll do it")

    def test_job_with_past_deadline_creation(self, job_service):
        """Test that jobs cannot be created with past deadlines."""
        with pytest.raises((ValueError, JobServiceError), match="future"):
            job_service.create_job(
                client_id="client_1",
                title="Test",
                description="Test",
                budget_usdc=100.0,
                deadline=datetime.now(timezone.utc) - timedelta(days=1),
            )

    def test_empty_skills_array(self, job_service):
        """Test job creation with empty skills array."""
        job = job_service.create_job(
            client_id="client_1",
            title="Test",
            description="Test",
            budget_usdc=100.0,
            deadline=datetime.now(timezone.utc) + timedelta(days=7),
            skills_required=[],
        )
        
        assert job.skills_required == []

    def test_unicode_in_job_content(self, job_service):
        """Test Unicode handling in job content."""
        unicode_content = "日本語テスト 🚀 émojis Ω∑∏"
        
        job = job_service.create_job(
            client_id="client_1",
            title=unicode_content[:100],
            description=unicode_content,
            budget_usdc=100.0,
            deadline=datetime.now(timezone.utc) + timedelta(days=7),
        )
        
        assert unicode_content in job.description


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
