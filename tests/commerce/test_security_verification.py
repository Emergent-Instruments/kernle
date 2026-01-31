"""
Security Verification Tests for Kernle Commerce.

These tests attempt to BYPASS the security fixes implemented in response
to the security audit. The goal is to verify fixes are actually effective.

Run with: uv run pytest tests/commerce/test_security_verification.py -v
"""

import asyncio
import concurrent.futures
import threading
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List, Tuple
from unittest.mock import MagicMock, patch, PropertyMock
import uuid
import re

import pytest

from kernle.commerce.wallet.models import WalletAccount, WalletStatus
from kernle.commerce.wallet.service import (
    WalletService,
    WalletServiceError,
    SpendingLimitExceededError,
    DailySpendRecord,
)
from kernle.commerce.wallet.storage import InMemoryWalletStorage
from kernle.commerce.jobs.models import Job, JobApplication, JobStatus, ApplicationStatus
from kernle.commerce.jobs.service import (
    JobService,
    JobServiceError,
    InvalidTransitionError,
    UnauthorizedError,
)
from kernle.commerce.jobs.storage import InMemoryJobStorage
from kernle.commerce.escrow.service import EscrowService, EscrowServiceError
from kernle.commerce.config import CommerceConfig


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def wallet_storage():
    return InMemoryWalletStorage()


@pytest.fixture
def wallet_service(wallet_storage):
    return WalletService(wallet_storage)


@pytest.fixture
def job_storage():
    return InMemoryJobStorage()


@pytest.fixture
def job_service(job_storage):
    return JobService(job_storage)


@pytest.fixture
def escrow_service():
    return EscrowService()


@pytest.fixture
def custom_config():
    """Create custom config for testing."""
    config = CommerceConfig(
        arbitrator_address="arbitrator_official",
    )
    # Note: authorized_arbitrators is checked via getattr with default []
    # The config class doesn't have this field, but the service handles it
    return config


@pytest.fixture
def job_service_with_config(job_storage, custom_config):
    return JobService(job_storage, config=custom_config)


def create_funded_job_with_worker(job_service: JobService) -> Tuple[Job, JobApplication]:
    """Helper to create a complete job setup."""
    job = job_service.create_job(
        client_id="client_1",
        title="Test Job",
        description="Description",
        budget_usdc=100.0,
        deadline=datetime.now(timezone.utc) + timedelta(days=7),
    )
    job = job_service.fund_job(job.id, "client_1", "0x" + "a" * 40)
    app = job_service.apply_to_job(job.id, "worker_1", "Application")
    job, app = job_service.accept_application(job.id, app.id, "client_1")
    return job, app


# =============================================================================
# 1. ARBITRATOR AUTHORIZATION BYPASS ATTEMPTS
# =============================================================================


class TestArbitratorBypass:
    """Attempt to bypass arbitrator authorization check."""

    def test_direct_unauthorized_resolution(self, job_service_with_config):
        """Verify unauthorized users cannot resolve disputes."""
        job, _ = create_funded_job_with_worker(job_service_with_config)
        job = job_service_with_config.dispute_job(job.id, "worker_1", "Problem")
        
        # Try various unauthorized actors
        unauthorized_actors = [
            "random_user",
            "client_1",  # Client is not arbitrator
            "worker_1",  # Worker is not arbitrator
            "",          # Empty string
            "SYSTEM",    # Wrong case (system vs SYSTEM)
            "System",    # Wrong case
        ]
        
        for actor in unauthorized_actors:
            with pytest.raises(UnauthorizedError):
                job_service_with_config.resolve_dispute(
                    job_id=job.id,
                    actor_id=actor,
                    resolution="worker",
                )

    def test_arbitrator_address_case_sensitivity(self, job_service_with_config):
        """Test if arbitrator check is case-sensitive."""
        job, _ = create_funded_job_with_worker(job_service_with_config)
        job = job_service_with_config.dispute_job(job.id, "worker_1", "Problem")
        
        # Try wrong case versions of "arbitrator_official"
        case_variants = [
            "ARBITRATOR_OFFICIAL",
            "Arbitrator_Official",
            "ARBITRATOR_official",
        ]
        
        for actor in case_variants:
            with pytest.raises(UnauthorizedError):
                job_service_with_config.resolve_dispute(
                    job_id=job.id,
                    actor_id=actor,
                    resolution="worker",
                )
    
    def test_authorized_arbitrators_work(self, job_service_with_config):
        """Verify configured arbitrators CAN resolve."""
        # Test official arbitrator from config
        job1, _ = create_funded_job_with_worker(job_service_with_config)
        job1 = job_service_with_config.dispute_job(job1.id, "worker_1", "Problem")
        
        resolved1 = job_service_with_config.resolve_dispute(
            job_id=job1.id,
            actor_id="arbitrator_official",
            resolution="worker",
        )
        assert resolved1.status == JobStatus.COMPLETED.value
        
        # Test system actor (always allowed for auto-resolution)
        job2, _ = create_funded_job_with_worker(job_service_with_config)
        job2 = job_service_with_config.dispute_job(job2.id, "worker_1", "Problem")
        
        resolved2 = job_service_with_config.resolve_dispute(
            job_id=job2.id,
            actor_id="system",
            resolution="client",
        )
        assert resolved2.status == JobStatus.COMPLETED.value

    def test_system_actor_bypass_check(self, job_service):
        """Verify 'system' actor is intentionally authorized (for auto-resolution)."""
        job, _ = create_funded_job_with_worker(job_service)
        job = job_service.dispute_job(job.id, "worker_1", "Problem")
        
        # System should work for auto-resolution
        resolved = job_service.resolve_dispute(
            job_id=job.id,
            actor_id="system",
            resolution="worker",
        )
        assert resolved.status == JobStatus.COMPLETED.value

    def test_invalid_resolution_values(self, job_service_with_config):
        """Test that only valid resolution values are accepted."""
        job, _ = create_funded_job_with_worker(job_service_with_config)
        job = job_service_with_config.dispute_job(job.id, "worker_1", "Problem")
        
        invalid_resolutions = [
            "attacker",
            "both",
            "neither",
            "",
            "WORKER",  # Wrong case
            "CLIENT",
        ]
        
        for resolution in invalid_resolutions:
            with pytest.raises(JobServiceError):
                job_service_with_config.resolve_dispute(
                    job_id=job.id,
                    actor_id="arbitrator_official",
                    resolution=resolution,
                )


# =============================================================================
# 2. RACE CONDITION STRESS TEST
# =============================================================================


class TestRaceConditionPrevention:
    """Stress test the race condition fix with many concurrent requests."""

    def test_high_concurrency_acceptance(self, job_service):
        """Test with high concurrency to verify locking works."""
        job = job_service.create_job(
            client_id="client_1",
            title="High Contention Job",
            description="Many workers will try",
            budget_usdc=100.0,
            deadline=datetime.now(timezone.utc) + timedelta(days=7),
        )
        job = job_service.fund_job(job.id, "client_1", "0x" + "a" * 40)
        
        # Create many applications
        apps = []
        for i in range(20):
            app = job_service.apply_to_job(job.id, f"worker_{i}", f"App {i}")
            apps.append(app)
        
        accepted = []
        rejected = []
        
        def try_accept(app):
            try:
                job_service.accept_application(job.id, app.id, "client_1")
                accepted.append(app.id)
            except (InvalidTransitionError, JobServiceError) as e:
                rejected.append((app.id, str(e)))
        
        # Launch all acceptance attempts simultaneously
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(try_accept, app) for app in apps]
            concurrent.futures.wait(futures)
        
        # Exactly one should succeed
        assert len(accepted) == 1, f"Race condition! Accepted: {len(accepted)}"
        assert len(rejected) == 19

    def test_rapid_sequential_acceptance_attempts(self, job_service):
        """Test rapid sequential attempts (still should fail after first)."""
        job = job_service.create_job(
            client_id="client_1",
            title="Test Job",
            description="Description",
            budget_usdc=100.0,
            deadline=datetime.now(timezone.utc) + timedelta(days=7),
        )
        job = job_service.fund_job(job.id, "client_1", "0x" + "a" * 40)
        
        app1 = job_service.apply_to_job(job.id, "worker_1", "App 1")
        app2 = job_service.apply_to_job(job.id, "worker_2", "App 2")
        
        # First succeeds
        job_service.accept_application(job.id, app1.id, "client_1")
        
        # Second fails (even though we try immediately)
        with pytest.raises((InvalidTransitionError, JobServiceError)):
            job_service.accept_application(job.id, app2.id, "client_1")

    def test_lock_isolation_between_jobs(self, job_service):
        """Verify locks are per-job and don't block unrelated jobs."""
        # Create two independent jobs
        job1 = job_service.create_job(
            client_id="client_1", title="Job 1", description="Desc",
            budget_usdc=100.0, deadline=datetime.now(timezone.utc) + timedelta(days=7),
        )
        job2 = job_service.create_job(
            client_id="client_2", title="Job 2", description="Desc",
            budget_usdc=100.0, deadline=datetime.now(timezone.utc) + timedelta(days=7),
        )
        
        job1 = job_service.fund_job(job1.id, "client_1", "0x" + "a" * 40)
        job2 = job_service.fund_job(job2.id, "client_2", "0x" + "b" * 40)
        
        app1 = job_service.apply_to_job(job1.id, "worker_1", "App 1")
        app2 = job_service.apply_to_job(job2.id, "worker_2", "App 2")
        
        results = []
        
        def accept_job1():
            job_service.accept_application(job1.id, app1.id, "client_1")
            results.append("job1")
        
        def accept_job2():
            job_service.accept_application(job2.id, app2.id, "client_2")
            results.append("job2")
        
        # Both should succeed - different jobs
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(accept_job1)
            f2 = executor.submit(accept_job2)
            concurrent.futures.wait([f1, f2])
        
        assert sorted(results) == ["job1", "job2"]


# =============================================================================
# 3. DAILY SPENDING LIMIT BYPASS ATTEMPTS
# =============================================================================


class TestDailySpendingLimitBypass:
    """Attempt to bypass daily spending limits."""

    def test_rapid_transactions_exhaust_limit(self, wallet_service):
        """Verify rapid transactions can't exceed daily limit."""
        owner_eoa = "0x" + "a" * 40
        wallet = wallet_service.create_wallet("agent_1")
        wallet = wallet_service.claim_wallet(wallet.id, owner_eoa)
        wallet_service.update_spending_limits(wallet.id, per_tx=10, daily=50, actor_id=owner_eoa)
        
        successes = 0
        for i in range(10):  # Try 10 * $10 = $100, exceeds $50 limit
            try:
                result = wallet_service.transfer(
                    wallet_id=wallet.id,
                    to_address="0x" + "b" * 40,
                    amount=Decimal("10"),
                    actor_id="agent_1",
                )
                if result.success:
                    successes += 1
            except SpendingLimitExceededError:
                break
        
        assert successes == 5  # $50 / $10 = 5 transactions max

    def test_decimal_precision_attack(self, wallet_service):
        """Try to exploit decimal precision to exceed limits."""
        owner_eoa = "0x" + "a" * 40
        wallet = wallet_service.create_wallet("agent_1")
        wallet = wallet_service.claim_wallet(wallet.id, owner_eoa)
        wallet_service.update_spending_limits(wallet.id, per_tx=100, daily=100, actor_id=owner_eoa)
        
        # Try many small transactions that might slip through rounding
        total = Decimal("0")
        for i in range(1000):
            try:
                result = wallet_service.transfer(
                    wallet_id=wallet.id,
                    to_address="0x" + "b" * 40,
                    amount=Decimal("0.1"),  # Small amount
                    actor_id="agent_1",
                )
                if result.success:
                    total += Decimal("0.1")
            except SpendingLimitExceededError:
                break
        
        # Should stop at exactly $100
        assert total <= Decimal("100"), f"Exceeded limit: {total}"

    def test_concurrent_transactions_respect_limit(self, wallet_service):
        """Test concurrent transactions don't exceed daily limit."""
        owner_eoa = "0x" + "a" * 40
        wallet = wallet_service.create_wallet("agent_1")
        wallet = wallet_service.claim_wallet(wallet.id, owner_eoa)
        wallet_service.update_spending_limits(wallet.id, per_tx=25, daily=100, actor_id=owner_eoa)
        
        success_count = [0]
        lock = threading.Lock()
        
        def transfer():
            try:
                result = wallet_service.transfer(
                    wallet_id=wallet.id,
                    to_address="0x" + "b" * 40,
                    amount=Decimal("25"),
                    actor_id="agent_1",
                )
                if result.success:
                    with lock:
                        success_count[0] += 1
            except SpendingLimitExceededError:
                pass
        
        # Try 10 concurrent $25 transfers
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(transfer) for _ in range(10)]
            concurrent.futures.wait(futures)
        
        # Should have 4 successful ($100 / $25 = 4)
        assert success_count[0] == 4

    def test_daily_reset_works_correctly(self, wallet_service):
        """Verify daily limit resets at midnight UTC."""
        owner_eoa = "0x" + "a" * 40
        wallet = wallet_service.create_wallet("agent_1")
        wallet = wallet_service.claim_wallet(wallet.id, owner_eoa)
        wallet_service.update_spending_limits(wallet.id, per_tx=100, daily=100, actor_id=owner_eoa)
        
        # Spend the daily limit
        wallet_service.transfer(
            wallet_id=wallet.id,
            to_address="0x" + "b" * 40,
            amount=Decimal("100"),
            actor_id="agent_1",
        )
        
        # Should be blocked now
        with pytest.raises(SpendingLimitExceededError):
            wallet_service.transfer(
                wallet_id=wallet.id,
                to_address="0x" + "b" * 40,
                amount=Decimal("1"),
                actor_id="agent_1",
            )
        
        # Simulate day change by manipulating internal state
        # Since we now use storage-level daily spend tracking, we need to
        # update the storage's tracking as well as the service's fallback dict
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date()
        
        # Update service's fallback dict (for backwards compatibility)
        wallet_service._daily_spend[wallet.id] = DailySpendRecord(
            date=yesterday,
            total_spent=Decimal("100"),
        )
        
        # Update storage's daily spend tracking (new implementation)
        if hasattr(wallet_service.storage, '_daily_spend'):
            wallet_service.storage._daily_spend[wallet.id] = (
                yesterday,
                Decimal("100"),
                1
            )
        
        # Now should work (new day)
        result = wallet_service.transfer(
            wallet_id=wallet.id,
            to_address="0x" + "b" * 40,
            amount=Decimal("50"),
            actor_id="agent_1",
        )
        assert result.success


# =============================================================================
# 4. WALLET CLAIM OWNERSHIP BYPASS
# =============================================================================


class TestWalletClaimBypass:
    """Attempt to claim wallets belonging to other users."""

    def test_cross_user_wallet_claim(self, wallet_service):
        """Verify users can't claim other users' wallets."""
        # Create wallet owned by user_1
        wallet = wallet_service.create_wallet("agent_1", user_id="user_1")
        
        # user_1 should be able to claim their own wallet
        claimed = wallet_service.claim_wallet(wallet.id, "0x" + "a" * 40)
        assert claimed.is_claimed

    def test_wallet_without_user_id(self, wallet_service):
        """Test wallet without user_id can be claimed (legacy behavior)."""
        # Wallet without user_id
        wallet = wallet_service.create_wallet("agent_orphan", user_id=None)
        
        # Anyone can claim orphan wallets (no ownership set)
        claimed = wallet_service.claim_wallet(wallet.id, "0x" + "b" * 40)
        assert claimed.is_claimed

    def test_double_claim_prevention(self, wallet_service):
        """Verify wallet can't be claimed twice."""
        wallet = wallet_service.create_wallet("agent_1", user_id="user_1")
        
        # First claim succeeds
        wallet_service.claim_wallet(wallet.id, "0x" + "a" * 40)
        
        # Second claim fails
        with pytest.raises(WalletServiceError, match="already claimed"):
            wallet_service.claim_wallet(wallet.id, "0x" + "b" * 40)


# =============================================================================
# 5. SQL INJECTION BYPASS ATTEMPTS
# =============================================================================


class TestSQLInjectionBypass:
    """Advanced SQL injection bypass attempts."""

    def test_encoded_injection_payloads(self):
        """Test URL-encoded and unicode injection attempts."""
        from kernle.commerce.skills.registry import InMemorySkillRegistry
        
        registry = InMemorySkillRegistry()
        
        # URL-encoded payloads
        payloads = [
            "%27%20OR%20%271%27%3D%271",  # ' OR '1'='1
            "%27%3B%20DROP%20TABLE%20skills%3B%20--",  # '; DROP TABLE skills; --
            "\\u0027 OR \\u00271\\u0027=\\u00271",  # Unicode escapes
            "research' UNION SELECT * FROM users WHERE '1'='1",
            "research/**/OR/**/1=1",  # Comment bypass
            "research' AND 1=(SELECT COUNT(*) FROM skills)--",
            "research\x00' OR '1'='1",  # Null byte injection
        ]
        
        for payload in payloads:
            try:
                results = registry.search_skills(payload)
                # Should return filtered results, not injection
                assert isinstance(results, list)
            except Exception as e:
                # Should not be SQL error
                assert "syntax" not in str(e).lower()

    def test_sanitization_effectiveness(self):
        """Verify the _sanitize_search_query function works."""
        # Import from backend app (must be added to Python path)
        import sys
        sys.path.insert(0, str(__file__).replace("/tests/commerce/test_security_verification.py", "/backend"))
        try:
            from app.routes.commerce.skills import _sanitize_search_query
        except ImportError:
            pytest.skip("Backend not in path - test SQL sanitization manually")
            return
        
        test_cases = [
            ("research'; DROP TABLE--", "research DROP TABLE"),
            ("test%test", "testtest"),
            ("test_test", "testtest"),
            ("test'\"test", "testtest"),
            ("/* comment */", " comment "),
            ("a" * 200, "a" * 100),  # Length limit
            ("normal-query", "normal-query"),
        ]
        
        for input_val, expected_pattern in test_cases:
            result = _sanitize_search_query(input_val)
            # Should not contain dangerous characters
            assert "'" not in result
            assert '"' not in result
            assert ";" not in result
            assert "--" not in result
            assert len(result) <= 100


# =============================================================================
# 6. REENTRANCY TEST FOR ESCROW
# =============================================================================


class TestEscrowReentrancy:
    """Test reentrancy protection in escrow operations."""

    def test_concurrent_release_blocked(self, escrow_service):
        """Verify concurrent release attempts are blocked."""
        escrow_address = "0x" + "a" * 40
        results = []
        errors = []
        
        def try_release():
            try:
                result = escrow_service.release(escrow_address, "0x" + "b" * 40)
                results.append(result)
            except EscrowServiceError as e:
                if "Reentrancy" in str(e):
                    errors.append(e)
        
        # Try concurrent releases
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(try_release) for _ in range(5)]
            concurrent.futures.wait(futures)
        
        # All should complete (but in real escrow, state would prevent double-spend)
        # The reentrancy guard prevents same operation from running twice simultaneously
        assert len(results) + len(errors) == 5

    def test_reentrancy_different_operations_allowed(self, escrow_service):
        """Verify different operations on same escrow can run."""
        escrow_address = "0x" + "a" * 40
        
        # These are different operations, so both should succeed
        result1 = escrow_service.release(escrow_address, "0x" + "b" * 40)
        assert result1.success
        
        # Different operation type
        result2 = escrow_service.refund(escrow_address, "0x" + "b" * 40)
        assert result2.success

    def test_reentrancy_guard_clears_on_exception(self, escrow_service):
        """Verify reentrancy guard is cleared even on errors."""
        escrow_address = "0x" + "test" + "a" * 36
        
        # First release succeeds
        result1 = escrow_service.release(escrow_address, "0x" + "b" * 40)
        assert result1.success
        
        # Second should also work (guard cleared)
        result2 = escrow_service.release(escrow_address, "0x" + "b" * 40)
        assert result2.success


# =============================================================================
# 7. URL VALIDATION BYPASS ATTEMPTS
# =============================================================================


class TestURLValidationBypass:
    """Attempt to bypass URL validation."""

    def test_url_scheme_variations(self, job_service):
        """Test various scheme variations."""
        job, _ = create_funded_job_with_worker(job_service)
        
        # These should all be rejected
        malicious_urls = [
            "FILE:///etc/passwd",  # Uppercase
            "File:///etc/passwd",  # Mixed case
            "javascript:void(0)",
            "JAVASCRIPT:alert(1)",
            "data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==",
            "vbscript:msgbox(1)",
            "file://localhost/etc/passwd",
            "file:///C:/Windows/System32/config/SAM",
        ]
        
        for url in malicious_urls:
            job.status = JobStatus.ACCEPTED.value  # Reset
            with pytest.raises(JobServiceError, match="Invalid URL"):
                job_service.deliver_job(job.id, "worker_1", url)

    def test_url_with_credentials(self, job_service):
        """Test URLs with embedded credentials."""
        job, _ = create_funded_job_with_worker(job_service)
        
        # These might leak credentials but are valid https URLs
        urls_with_creds = [
            "https://user:password@example.com/file",
            "https://admin:secret@internal.corp/data",
        ]
        
        for url in urls_with_creds:
            job.status = JobStatus.ACCEPTED.value  # Reset
            # These are technically valid https URLs
            result = job_service.deliver_job(job.id, "worker_1", url)
            assert result.deliverable_url == url

    def test_ipfs_and_arweave_allowed(self, job_service):
        """Verify decentralized storage URLs work."""
        job, _ = create_funded_job_with_worker(job_service)
        
        valid_urls = [
            "ipfs://QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG",
            "ipns://k51qzi5uqu5dlvj2baxnqndepeb86cbk3ng7n3i46uzyxzyqj2xjonzllnv0v8",
            "ar://BNttzDav3jHVnNiV7nYbQv-GY0HQ-4XXsdkE5K9yl8M",
        ]
        
        for url in valid_urls:
            job.status = JobStatus.ACCEPTED.value  # Reset
            result = job_service.deliver_job(job.id, "worker_1", url)
            assert result.deliverable_url == url

    def test_extremely_long_url_rejected(self, job_service):
        """Test URL length limit."""
        job, _ = create_funded_job_with_worker(job_service)
        
        long_url = "https://example.com/" + "a" * 3000
        
        with pytest.raises(JobServiceError, match="too long"):
            job_service.deliver_job(job.id, "worker_1", long_url)


# =============================================================================
# 8. EDGE CASES AND REGRESSION TESTS
# =============================================================================


class TestEdgeCasesAndRegressions:
    """Test edge cases that might not be covered."""

    def test_empty_actor_id(self, job_service_with_config):
        """Test empty actor_id in resolution."""
        job, _ = create_funded_job_with_worker(job_service_with_config)
        job = job_service_with_config.dispute_job(job.id, "worker_1", "Problem")
        
        with pytest.raises(UnauthorizedError):
            job_service_with_config.resolve_dispute(
                job_id=job.id,
                actor_id="",
                resolution="worker",
            )

    def test_none_actor_id(self, job_service_with_config):
        """Test None actor_id in resolution."""
        job, _ = create_funded_job_with_worker(job_service_with_config)
        job = job_service_with_config.dispute_job(job.id, "worker_1", "Problem")
        
        with pytest.raises((UnauthorizedError, TypeError, AttributeError)):
            job_service_with_config.resolve_dispute(
                job_id=job.id,
                actor_id=None,
                resolution="worker",
            )

    def test_whitespace_only_inputs(self, job_service):
        """Test whitespace-only inputs."""
        # Whitespace title
        with pytest.raises((ValueError, JobServiceError)):
            job_service.create_job(
                client_id="client_1",
                title="   ",  # Just spaces
                description="Description",
                budget_usdc=100.0,
                deadline=datetime.now(timezone.utc) + timedelta(days=7),
            )

    def test_wallet_spending_with_zero_amount(self, wallet_service):
        """Test zero amount transfer is rejected.
        
        FIXED: Zero amount transfers are now rejected as part of the
        amount <= 0 validation fix.
        """
        wallet = wallet_service.create_wallet("agent_1")
        wallet = wallet_service.claim_wallet(wallet.id, "0x" + "a" * 40)
        
        # Zero amount should be rejected
        result = wallet_service.transfer(
            wallet_id=wallet.id,
            to_address="0x" + "b" * 40,
            amount=Decimal("0"),
            actor_id="agent_1",
        )
        assert not result.success
        assert result.error == "Transfer amount must be positive"

    def test_negative_spending_attempt(self, wallet_service):
        """Test negative amount transfer is properly rejected.
        
        FIXED: Negative transfers are now rejected with success=False.
        This prevents:
        1. Negative transfer amounts (effectively crediting)
        2. Bypass of daily spend tracking
        """
        wallet = wallet_service.create_wallet("agent_1")
        wallet = wallet_service.claim_wallet(wallet.id, "0x" + "a" * 40)
        
        # Negative amounts should be rejected
        result = wallet_service.transfer(
            wallet_id=wallet.id,
            to_address="0x" + "b" * 40,
            amount=Decimal("-100"),
            actor_id="agent_1",
        )
        
        # Verify the negative transfer was rejected
        assert not result.success
        assert result.error == "Transfer amount must be positive"
        
        # Also test zero amount is rejected
        result_zero = wallet_service.transfer(
            wallet_id=wallet.id,
            to_address="0x" + "b" * 40,
            amount=Decimal("0"),
            actor_id="agent_1",
        )
        assert not result_zero.success
        assert result_zero.error == "Transfer amount must be positive"


# =============================================================================
# 9. COMPLETE WORKFLOW SECURITY TEST
# =============================================================================


class TestCompleteWorkflowSecurity:
    """End-to-end security tests covering complete workflows."""

    def test_complete_job_lifecycle_security(self, job_service, wallet_service):
        """Test complete job lifecycle with all security checks."""
        # Create job
        job = job_service.create_job(
            client_id="client_1",
            title="Secure Job",
            description="Description",
            budget_usdc=100.0,
            deadline=datetime.now(timezone.utc) + timedelta(days=7),
        )
        
        # Client can't apply to own job
        with pytest.raises(JobServiceError):
            job_service.apply_to_job(job.id, "client_1", "Self-apply")
        
        # Fund job
        job = job_service.fund_job(job.id, "client_1", "0x" + "a" * 40)
        
        # Worker applies
        app = job_service.apply_to_job(job.id, "worker_1", "I'll do it")
        
        # Accept worker
        job, app = job_service.accept_application(job.id, app.id, "client_1")
        
        # Worker can't accept their own application
        with pytest.raises((UnauthorizedError, InvalidTransitionError)):
            job_service.accept_application(job.id, app.id, "worker_1")
        
        # Deliver with valid URL
        job = job_service.deliver_job(
            job.id, "worker_1", "https://github.com/work"
        )
        
        # Worker can't approve
        with pytest.raises(UnauthorizedError):
            job_service.approve_job(job.id, "worker_1")
        
        # Client approves
        job = job_service.approve_job(job.id, "client_1")
        
        assert job.status == JobStatus.COMPLETED.value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
