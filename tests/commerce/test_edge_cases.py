"""
Adversarial edge case tests for commerce models.

These tests check for missing validation and boundary conditions that could
cause issues in production. Tests marked as xfail are known issues that
should be fixed before production.

Run with: uv run pytest tests/commerce/test_edge_cases.py -v
"""

import pytest
from datetime import datetime, timezone, timedelta

from kernle.commerce.wallet.models import WalletAccount, WalletStatus
from kernle.commerce.jobs.models import Job, JobApplication, JobStatus, ApplicationStatus
from kernle.commerce.skills.models import Skill, SkillCategory


class TestWalletEdgeCases:
    """Adversarial tests for WalletAccount validation gaps."""

    def test_empty_wallet_address_rejected(self):
        """Empty wallet address should raise ValueError."""
        with pytest.raises(ValueError):
            WalletAccount(id="w1", agent_id="a1", wallet_address="")

    def test_short_wallet_address_rejected(self):
        """Wallet address < 42 chars should raise ValueError."""
        with pytest.raises(ValueError):
            WalletAccount(id="w1", agent_id="a1", wallet_address="0x123")

    def test_long_wallet_address_rejected(self):
        """Wallet address > 42 chars should raise ValueError."""
        with pytest.raises(ValueError):
            WalletAccount(id="w1", agent_id="a1", wallet_address="0x" + "a" * 50)

    def test_wallet_address_exact_length(self):
        """Valid wallet address is exactly 42 chars (0x + 40 hex)."""
        # This should always work
        valid_address = "0x" + "a" * 40
        wallet = WalletAccount(id="w1", agent_id="a1", wallet_address=valid_address)
        assert len(wallet.wallet_address) == 42

    def test_negative_spending_limit_per_tx_rejected(self):
        """Negative spending_limit_per_tx should raise ValueError."""
        with pytest.raises(ValueError):
            WalletAccount(
                id="w1",
                agent_id="a1",
                wallet_address="0x" + "a" * 40,
                spending_limit_per_tx=-100.0,
            )

    def test_zero_spending_limit_per_tx_rejected(self):
        """Zero spending_limit_per_tx should raise ValueError."""
        with pytest.raises(ValueError):
            WalletAccount(
                id="w1",
                agent_id="a1",
                wallet_address="0x" + "a" * 40,
                spending_limit_per_tx=0.0,
            )

    def test_negative_spending_limit_daily_rejected(self):
        """Negative spending_limit_daily should raise ValueError."""
        with pytest.raises(ValueError):
            WalletAccount(
                id="w1",
                agent_id="a1",
                wallet_address="0x" + "a" * 40,
                spending_limit_daily=-1000.0,
            )

    def test_zero_spending_limit_daily_rejected(self):
        """Zero spending_limit_daily should raise ValueError."""
        with pytest.raises(ValueError):
            WalletAccount(
                id="w1",
                agent_id="a1",
                wallet_address="0x" + "a" * 40,
                spending_limit_daily=0.0,
            )

    def test_invalid_owner_eoa_rejected(self):
        """Invalid owner_eoa format should raise ValueError."""
        with pytest.raises(ValueError):
            WalletAccount(
                id="w1",
                agent_id="a1",
                wallet_address="0x" + "a" * 40,
                owner_eoa="not-an-address",
            )

    def test_valid_owner_eoa_accepted(self):
        """Valid owner_eoa should be accepted."""
        wallet = WalletAccount(
            id="w1",
            agent_id="a1",
            wallet_address="0x" + "a" * 40,
            owner_eoa="0x" + "b" * 40,
        )
        assert wallet.owner_eoa == "0x" + "b" * 40


class TestJobEdgeCases:
    """Adversarial tests for Job validation gaps."""

    def _future_deadline(self) -> datetime:
        """Get a deadline 7 days in the future."""
        return datetime.now(timezone.utc) + timedelta(days=7)

    def test_empty_title_rejected(self):
        """Empty job title should raise ValueError."""
        with pytest.raises(ValueError):
            Job(
                id="j1",
                client_id="a1",
                title="",
                description="Test description",
                budget_usdc=50.0,
                deadline=self._future_deadline(),
            )

    def test_whitespace_title_rejected(self):
        """Whitespace-only title should raise ValueError."""
        with pytest.raises(ValueError):
            Job(
                id="j1",
                client_id="a1",
                title="   ",
                description="Test description",
                budget_usdc=50.0,
                deadline=self._future_deadline(),
            )

    def test_empty_description_rejected(self):
        """Empty job description should raise ValueError."""
        with pytest.raises(ValueError):
            Job(
                id="j1",
                client_id="a1",
                title="Test Job",
                description="",
                budget_usdc=50.0,
                deadline=self._future_deadline(),
            )

    def test_whitespace_description_rejected(self):
        """Whitespace-only description should raise ValueError."""
        with pytest.raises(ValueError):
            Job(
                id="j1",
                client_id="a1",
                title="Test Job",
                description="   ",
                budget_usdc=50.0,
                deadline=self._future_deadline(),
            )

    def test_past_deadline_rejected(self):
        """Past deadline should raise ValueError."""
        with pytest.raises(ValueError):
            Job(
                id="j1",
                client_id="a1",
                title="Test Job",
                description="Test description",
                budget_usdc=50.0,
                deadline=datetime.now(timezone.utc) - timedelta(days=1),
            )

    def test_invalid_escrow_address_rejected(self):
        """Invalid escrow_address format should raise ValueError."""
        with pytest.raises(ValueError):
            Job(
                id="j1",
                client_id="a1",
                title="Test Job",
                description="Test description",
                budget_usdc=50.0,
                deadline=self._future_deadline(),
                escrow_address="invalid-address",
            )

    def test_valid_escrow_address_accepted(self):
        """Valid escrow_address should be accepted."""
        job = Job(
            id="j1",
            client_id="a1",
            title="Test Job",
            description="Test description",
            budget_usdc=50.0,
            deadline=self._future_deadline(),
            escrow_address="0x" + "a" * 40,
        )
        assert job.escrow_address == "0x" + "a" * 40

    def test_exact_200_char_title_accepted(self):
        """Title of exactly 200 chars should be accepted."""
        title = "A" * 200
        job = Job(
            id="j1",
            client_id="a1",
            title=title,
            description="Test description",
            budget_usdc=50.0,
            deadline=self._future_deadline(),
        )
        assert len(job.title) == 200

    def test_201_char_title_rejected(self):
        """Title over 200 chars should raise ValueError."""
        with pytest.raises(ValueError, match="Title too long"):
            Job(
                id="j1",
                client_id="a1",
                title="A" * 201,
                description="Test description",
                budget_usdc=50.0,
                deadline=self._future_deadline(),
            )

    def test_very_small_budget_accepted(self):
        """Very small positive budget should be accepted (dust amount)."""
        # This is policy - can change if needed
        job = Job(
            id="j1",
            client_id="a1",
            title="Test Job",
            description="Test description",
            budget_usdc=0.000001,
            deadline=self._future_deadline(),
        )
        assert job.budget_usdc == 0.000001


class TestJobApplicationEdgeCases:
    """Adversarial tests for JobApplication validation gaps."""

    def test_whitespace_only_message_rejected(self):
        """Whitespace-only message should raise ValueError."""
        # This IS validated in the model - verify it still works
        with pytest.raises(ValueError, match="message cannot be empty"):
            JobApplication(
                id="a1",
                job_id="j1",
                applicant_id="agent1",
                message="   ",
            )

    def test_very_long_message_accepted(self):
        """Very long messages are accepted (no length limit in model)."""
        # This is INFO level - document behavior
        app = JobApplication(
            id="a1",
            job_id="j1",
            applicant_id="agent1",
            message="x" * 100000,
        )
        assert len(app.message) == 100000


class TestSkillEdgeCases:
    """Adversarial tests for Skill validation gaps."""

    def test_long_skill_name_rejected(self):
        """Skill name > 50 chars should raise ValueError (DB is VARCHAR(50))."""
        with pytest.raises(ValueError):
            Skill(id="s1", name="a" * 51)

    def test_skill_name_exactly_50_chars_accepted(self):
        """Skill name of exactly 50 chars should be accepted."""
        skill = Skill(id="s1", name="a" * 50)
        assert len(skill.name) == 50

    def test_negative_usage_count_rejected(self):
        """Negative usage_count should raise ValueError (DB has non_negative check)."""
        with pytest.raises(ValueError):
            Skill(id="s1", name="coding", usage_count=-5)

    def test_zero_usage_count_accepted(self):
        """Zero usage_count should be accepted."""
        skill = Skill(id="s1", name="coding", usage_count=0)
        assert skill.usage_count == 0

    def test_hyphen_starting_skill_name(self):
        """Skill name starting with hyphen - document behavior."""
        # This is allowed by current validation - may want to reject
        skill = Skill(id="s1", name="-test-skill")
        assert skill.name == "-test-skill"

    def test_skill_name_with_numbers(self):
        """Skill name with numbers should be accepted."""
        skill = Skill(id="s1", name="python3")
        assert skill.name == "python3"

    def test_skill_name_with_consecutive_hyphens(self):
        """Skill name with consecutive hyphens - document behavior."""
        # This is allowed - may want to reject
        skill = Skill(id="s1", name="web--scraping")
        assert skill.name == "web--scraping"


class TestRoundTripSerialization:
    """Tests for to_dict/from_dict round-trip integrity."""

    def test_wallet_round_trip(self):
        """WalletAccount should survive serialization round-trip."""
        now = datetime.now(timezone.utc)
        original = WalletAccount(
            id="w1",
            agent_id="a1",
            wallet_address="0x" + "a" * 40,
            chain="base-sepolia",
            status="active",
            user_id="usr_123",
            owner_eoa="0x" + "b" * 40,
            spending_limit_per_tx=50.0,
            spending_limit_daily=500.0,
            cdp_wallet_id="cdp_xyz",
            created_at=now,
            claimed_at=now,
        )
        
        # Use include_internal=True to include cdp_wallet_id for full round-trip
        data = original.to_dict(include_internal=True)
        restored = WalletAccount.from_dict(data)
        
        assert restored.id == original.id
        assert restored.agent_id == original.agent_id
        assert restored.wallet_address == original.wallet_address
        assert restored.chain == original.chain
        assert restored.status == original.status
        assert restored.user_id == original.user_id
        assert restored.owner_eoa == original.owner_eoa
        assert restored.spending_limit_per_tx == original.spending_limit_per_tx
        assert restored.spending_limit_daily == original.spending_limit_daily
        assert restored.cdp_wallet_id == original.cdp_wallet_id

    def test_job_round_trip(self):
        """Job should survive serialization round-trip."""
        now = datetime.now(timezone.utc)
        deadline = now + timedelta(days=7)
        
        original = Job(
            id="j1",
            client_id="a1",
            title="Test Job",
            description="Test description",
            budget_usdc=100.0,
            deadline=deadline,
            worker_id="a2",
            skills_required=["coding", "research"],
            escrow_address="0x" + "a" * 40,
            status="accepted",
            created_at=now,
            updated_at=now,
            funded_at=now,
            accepted_at=now,
        )
        
        data = original.to_dict()
        restored = Job.from_dict(data)
        
        assert restored.id == original.id
        assert restored.client_id == original.client_id
        assert restored.title == original.title
        assert restored.budget_usdc == original.budget_usdc
        assert restored.skills_required == original.skills_required
        assert restored.status == original.status

    def test_skill_round_trip(self):
        """Skill should survive serialization round-trip."""
        now = datetime.now(timezone.utc)
        
        original = Skill(
            id="s1",
            name="coding",
            description="Software development",
            category="technical",
            usage_count=42,
            created_at=now,
        )
        
        data = original.to_dict()
        restored = Skill.from_dict(data)
        
        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.category == original.category
        assert restored.usage_count == original.usage_count


class TestStateMachineEdgeCases:
    """Tests for job state machine edge cases."""

    def test_cannot_transition_from_completed(self):
        """Completed job cannot transition to any state."""
        job = Job(
            id="j1",
            client_id="a1",
            title="Test",
            description="Test",
            budget_usdc=50.0,
            deadline=datetime.now(timezone.utc) + timedelta(days=7),
            status="completed",
        )
        
        for status in JobStatus:
            assert job.can_transition_to(status) is False

    def test_cannot_transition_from_cancelled(self):
        """Cancelled job cannot transition to any state."""
        job = Job(
            id="j1",
            client_id="a1",
            title="Test",
            description="Test",
            budget_usdc=50.0,
            deadline=datetime.now(timezone.utc) + timedelta(days=7),
            status="cancelled",
        )
        
        for status in JobStatus:
            assert job.can_transition_to(status) is False

    def test_disputed_can_only_complete(self):
        """Disputed job can only transition to completed."""
        job = Job(
            id="j1",
            client_id="a1",
            title="Test",
            description="Test",
            budget_usdc=50.0,
            deadline=datetime.now(timezone.utc) + timedelta(days=7),
            status="disputed",
        )
        
        assert job.can_transition_to(JobStatus.COMPLETED) is True
        assert job.can_transition_to(JobStatus.CANCELLED) is False
        assert job.can_transition_to(JobStatus.OPEN) is False
