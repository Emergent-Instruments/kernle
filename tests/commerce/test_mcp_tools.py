"""
Tests for commerce MCP tools.

Tests wallet, job, and skills MCP tool implementations.
"""

from datetime import datetime, timedelta, timezone
from typing import List

import pytest
from mcp.types import TextContent

from kernle.commerce.jobs.service import JobService
from kernle.commerce.jobs.storage import InMemoryJobStorage
from kernle.commerce.mcp import (
    COMMERCE_TOOLS,
    TOOL_HANDLERS,
    call_commerce_tool,
    configure_commerce_services,
    get_commerce_agent_id,
    get_commerce_tools,
    reset_commerce_services,
    set_commerce_agent_id,
)
from kernle.commerce.skills.registry import InMemorySkillRegistry
from kernle.commerce.wallet.service import WalletService
from kernle.commerce.wallet.storage import InMemoryWalletStorage

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def reset_services():
    """Reset commerce services before each test."""
    reset_commerce_services()
    yield
    reset_commerce_services()


@pytest.fixture
def wallet_storage():
    """Create a fresh wallet storage."""
    return InMemoryWalletStorage()


@pytest.fixture
def wallet_service(wallet_storage):
    """Create a wallet service with fresh storage."""
    return WalletService(wallet_storage)


@pytest.fixture
def job_storage():
    """Create a fresh job storage."""
    return InMemoryJobStorage()


@pytest.fixture
def job_service(job_storage):
    """Create a job service with fresh storage."""
    return JobService(job_storage)


@pytest.fixture
def skill_registry():
    """Create a fresh skill registry."""
    return InMemorySkillRegistry()


@pytest.fixture
def configured_services(wallet_service, job_service, skill_registry):
    """Configure commerce services with fresh instances."""
    configure_commerce_services(
        wallet_service=wallet_service,
        job_service=job_service,
        skill_registry=skill_registry,
    )
    set_commerce_agent_id("test-agent")
    return {
        "wallet": wallet_service,
        "job": job_service,
        "skill": skill_registry,
    }


def get_text_content(result: List[TextContent]) -> str:
    """Extract text from MCP result."""
    assert len(result) == 1
    assert result[0].type == "text"
    return result[0].text


# =============================================================================
# Tool Definition Tests
# =============================================================================

class TestToolDefinitions:
    """Test that all tools are properly defined."""

    def test_commerce_tools_not_empty(self):
        """Commerce tools list should not be empty."""
        assert len(COMMERCE_TOOLS) > 0

    def test_get_commerce_tools_returns_list(self):
        """get_commerce_tools should return a list."""
        tools = get_commerce_tools()
        assert isinstance(tools, list)
        assert len(tools) == len(COMMERCE_TOOLS)

    def test_all_tools_have_names(self):
        """All tools should have names."""
        for tool in COMMERCE_TOOLS:
            assert tool.name
            assert isinstance(tool.name, str)

    def test_all_tools_have_descriptions(self):
        """All tools should have descriptions."""
        for tool in COMMERCE_TOOLS:
            assert tool.description
            assert isinstance(tool.description, str)

    def test_all_tools_have_handlers(self):
        """All tools should have corresponding handlers."""
        for tool in COMMERCE_TOOLS:
            assert tool.name in TOOL_HANDLERS, f"No handler for tool: {tool.name}"

    def test_expected_tools_present(self):
        """Check that all expected tools are present."""
        tool_names = {t.name for t in COMMERCE_TOOLS}
        expected = {
            # Wallet
            "wallet_balance",
            "wallet_address",
            "wallet_status",
            # Job (client)
            "job_create",
            "job_list",
            "job_fund",
            "job_applications",
            "job_accept",
            "job_approve",
            "job_cancel",
            "job_dispute",
            # Job (worker)
            "job_search",
            "job_apply",
            "job_deliver",
            # Skills
            "skills_list",
            "skills_search",
        }
        assert expected.issubset(tool_names), f"Missing tools: {expected - tool_names}"


# =============================================================================
# Wallet Tool Tests
# =============================================================================

class TestWalletTools:
    """Test wallet MCP tools."""

    @pytest.mark.asyncio
    async def test_wallet_balance_creates_wallet(self, configured_services):
        """wallet_balance should auto-create wallet if missing."""
        result = await call_commerce_tool("wallet_balance", {})
        text = get_text_content(result)

        assert "Wallet created!" in text
        assert "0x" in text  # Should contain wallet address
        assert "USDC" in text

    @pytest.mark.asyncio
    async def test_wallet_balance_returns_existing(self, configured_services):
        """wallet_balance should return existing wallet balance."""
        # Create wallet first
        wallet_service = configured_services["wallet"]
        wallet = wallet_service.create_wallet("test-agent")

        result = await call_commerce_tool("wallet_balance", {})
        text = get_text_content(result)

        assert "Wallet created!" not in text
        assert wallet.wallet_address in text

    @pytest.mark.asyncio
    async def test_wallet_address(self, configured_services):
        """wallet_address should return wallet address."""
        result = await call_commerce_tool("wallet_address", {})
        text = get_text_content(result)

        assert "Wallet Address:" in text
        assert "0x" in text

    @pytest.mark.asyncio
    async def test_wallet_status(self, configured_services):
        """wallet_status should return wallet status details."""
        result = await call_commerce_tool("wallet_status", {})
        text = get_text_content(result)

        assert "Wallet Address:" in text
        assert "Chain:" in text
        assert "Status:" in text
        assert "Per-Transaction Limit:" in text
        assert "Daily Limit:" in text


# =============================================================================
# Job Tool Tests (Client)
# =============================================================================

class TestJobClientTools:
    """Test job MCP tools for clients."""

    @pytest.mark.asyncio
    async def test_job_create(self, configured_services):
        """job_create should create a new job."""
        deadline = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()

        result = await call_commerce_tool("job_create", {
            "title": "Research Task",
            "description": "Research the topic thoroughly",
            "budget": 50.0,
            "deadline": deadline,
            "skills": ["research", "writing"],
        })
        text = get_text_content(result)

        assert "Job created!" in text
        assert "Research Task" in text
        assert "50" in text

    @pytest.mark.asyncio
    async def test_job_create_requires_title(self, configured_services):
        """job_create should require title."""
        deadline = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()

        result = await call_commerce_tool("job_create", {
            "description": "Some work",
            "budget": 50.0,
            "deadline": deadline,
        })
        text = get_text_content(result)

        assert "Invalid input" in text or "required" in text.lower()

    @pytest.mark.asyncio
    async def test_job_list_empty(self, configured_services):
        """job_list should handle empty results."""
        result = await call_commerce_tool("job_list", {})
        text = get_text_content(result)

        assert "No jobs found" in text

    @pytest.mark.asyncio
    async def test_job_list_with_jobs(self, configured_services):
        """job_list should list created jobs."""
        deadline = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()

        # Create a job first
        await call_commerce_tool("job_create", {
            "title": "Test Job",
            "description": "Test description",
            "budget": 100.0,
            "deadline": deadline,
        })

        result = await call_commerce_tool("job_list", {})
        text = get_text_content(result)

        assert "Test Job" in text
        assert "100" in text

    @pytest.mark.asyncio
    async def test_job_list_mine(self, configured_services):
        """job_list with mine=true should only list own jobs."""
        deadline = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()

        # Create a job
        await call_commerce_tool("job_create", {
            "title": "My Job",
            "description": "My description",
            "budget": 100.0,
            "deadline": deadline,
        })

        result = await call_commerce_tool("job_list", {"mine": True})
        text = get_text_content(result)

        assert "My Job" in text

    @pytest.mark.asyncio
    async def test_job_fund(self, configured_services):
        """job_fund should fund a job."""
        (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()

        # Create a job
        job_service = configured_services["job"]
        job = job_service.create_job(
            client_id="test-agent",
            title="Fund Test",
            description="Test funding",
            budget_usdc=50.0,
            deadline=datetime.now(timezone.utc) + timedelta(days=7),
        )

        result = await call_commerce_tool("job_fund", {"job_id": job.id})
        text = get_text_content(result)

        assert "Job funded!" in text
        assert "Escrow:" in text

    @pytest.mark.asyncio
    async def test_job_cancel(self, configured_services):
        """job_cancel should cancel a job."""
        job_service = configured_services["job"]
        job = job_service.create_job(
            client_id="test-agent",
            title="Cancel Test",
            description="Test cancellation",
            budget_usdc=50.0,
            deadline=datetime.now(timezone.utc) + timedelta(days=7),
        )

        result = await call_commerce_tool("job_cancel", {"job_id": job.id})
        text = get_text_content(result)

        assert "cancelled" in text.lower()


# =============================================================================
# Job Tool Tests (Worker)
# =============================================================================

class TestJobWorkerTools:
    """Test job MCP tools for workers."""

    @pytest.mark.asyncio
    async def test_job_search_empty(self, configured_services):
        """job_search should handle no results."""
        result = await call_commerce_tool("job_search", {})
        text = get_text_content(result)

        assert "No jobs found" in text

    @pytest.mark.asyncio
    async def test_job_search_with_jobs(self, configured_services):
        """job_search should find available jobs."""
        job_service = configured_services["job"]

        # Create and fund a job (funded jobs are searchable)
        job = job_service.create_job(
            client_id="other-agent",
            title="Coding Work",
            description="Build an API",
            budget_usdc=200.0,
            deadline=datetime.now(timezone.utc) + timedelta(days=7),
            skills_required=["coding"],
        )
        job_service.fund_job(job.id, "other-agent", "0x" + "a" * 40)

        result = await call_commerce_tool("job_search", {"query": "API"})
        text = get_text_content(result)

        assert "Coding Work" in text or "Build an API" in text

    @pytest.mark.asyncio
    async def test_job_apply(self, configured_services):
        """job_apply should submit an application."""
        job_service = configured_services["job"]

        # Create a job by another agent
        job = job_service.create_job(
            client_id="other-agent",
            title="Work Needed",
            description="Do some work",
            budget_usdc=100.0,
            deadline=datetime.now(timezone.utc) + timedelta(days=7),
        )

        result = await call_commerce_tool("job_apply", {
            "job_id": job.id,
            "message": "I can do this work!",
        })
        text = get_text_content(result)

        assert "Application submitted!" in text
        assert "I can do this work" in text

    @pytest.mark.asyncio
    async def test_job_apply_duplicate(self, configured_services):
        """job_apply should prevent duplicate applications."""
        job_service = configured_services["job"]

        job = job_service.create_job(
            client_id="other-agent",
            title="Work Needed",
            description="Do some work",
            budget_usdc=100.0,
            deadline=datetime.now(timezone.utc) + timedelta(days=7),
        )

        # Apply once
        await call_commerce_tool("job_apply", {
            "job_id": job.id,
            "message": "First application",
        })

        # Try to apply again
        result = await call_commerce_tool("job_apply", {
            "job_id": job.id,
            "message": "Second application",
        })
        text = get_text_content(result)

        assert "already applied" in text.lower()

    @pytest.mark.asyncio
    async def test_job_deliver(self, configured_services):
        """job_deliver should submit a deliverable."""
        job_service = configured_services["job"]

        # Create job by another agent
        job = job_service.create_job(
            client_id="other-agent",
            title="Build Something",
            description="Build it well",
            budget_usdc=100.0,
            deadline=datetime.now(timezone.utc) + timedelta(days=7),
        )

        # Fund the job
        job_service.fund_job(job.id, "other-agent", "0x" + "a" * 40)

        # Apply and get accepted
        app = job_service.apply_to_job(job.id, "test-agent", "I'll build it!")
        job_service.accept_application(job.id, app.id, "other-agent")

        # Now deliver
        result = await call_commerce_tool("job_deliver", {
            "job_id": job.id,
            "url": "https://github.com/example/deliverable",
            "hash": "abc123",
        })
        text = get_text_content(result)

        assert "Deliverable submitted!" in text
        assert "github.com" in text


# =============================================================================
# Job Application Flow Tests
# =============================================================================

class TestJobApplicationFlow:
    """Test the complete job application flow."""

    @pytest.mark.asyncio
    async def test_job_applications(self, configured_services):
        """job_applications should list applications."""
        job_service = configured_services["job"]

        # Create job
        job = job_service.create_job(
            client_id="test-agent",
            title="My Job",
            description="Need help",
            budget_usdc=50.0,
            deadline=datetime.now(timezone.utc) + timedelta(days=7),
        )

        # Have someone apply
        job_service.apply_to_job(job.id, "worker-agent", "I can help!")

        result = await call_commerce_tool("job_applications", {"job_id": job.id})
        text = get_text_content(result)

        assert "application" in text.lower()
        assert "worker-agent" in text

    @pytest.mark.asyncio
    async def test_job_accept(self, configured_services):
        """job_accept should accept an application."""
        job_service = configured_services["job"]

        # Create and fund job
        job = job_service.create_job(
            client_id="test-agent",
            title="Accept Test",
            description="Test acceptance",
            budget_usdc=50.0,
            deadline=datetime.now(timezone.utc) + timedelta(days=7),
        )
        job_service.fund_job(job.id, "test-agent", "0x" + "b" * 40)

        # Have someone apply
        app = job_service.apply_to_job(job.id, "worker-agent", "Pick me!")

        result = await call_commerce_tool("job_accept", {
            "job_id": job.id,
            "application_id": app.id,
        })
        text = get_text_content(result)

        assert "Application accepted!" in text
        assert "worker-agent" in text

    @pytest.mark.asyncio
    async def test_job_approve(self, configured_services):
        """job_approve should approve and complete a job."""
        job_service = configured_services["job"]

        # Create and fund job
        job = job_service.create_job(
            client_id="test-agent",
            title="Approve Test",
            description="Test approval",
            budget_usdc=50.0,
            deadline=datetime.now(timezone.utc) + timedelta(days=7),
        )
        job_service.fund_job(job.id, "test-agent", "0x" + "c" * 40)

        # Apply and accept
        app = job_service.apply_to_job(job.id, "worker-agent", "I'll do it!")
        job_service.accept_application(job.id, app.id, "test-agent")

        # Deliver
        job_service.deliver_job(job.id, "worker-agent", "https://example.com/work")

        result = await call_commerce_tool("job_approve", {"job_id": job.id})
        text = get_text_content(result)

        assert "approved" in text.lower()
        assert "Payment released" in text

    @pytest.mark.asyncio
    async def test_job_dispute(self, configured_services):
        """job_dispute should raise a dispute."""
        job_service = configured_services["job"]

        # Create and fund job
        job = job_service.create_job(
            client_id="test-agent",
            title="Dispute Test",
            description="Test dispute",
            budget_usdc=50.0,
            deadline=datetime.now(timezone.utc) + timedelta(days=7),
        )
        job_service.fund_job(job.id, "test-agent", "0x" + "d" * 40)

        # Apply, accept, and deliver
        app = job_service.apply_to_job(job.id, "worker-agent", "On it!")
        job_service.accept_application(job.id, app.id, "test-agent")
        job_service.deliver_job(job.id, "worker-agent", "https://example.com/bad")

        result = await call_commerce_tool("job_dispute", {
            "job_id": job.id,
            "reason": "Work is incomplete",
        })
        text = get_text_content(result)

        assert "Dispute raised" in text


# =============================================================================
# Skills Tool Tests
# =============================================================================

class TestSkillsTools:
    """Test skills MCP tools."""

    @pytest.mark.asyncio
    async def test_skills_list(self, configured_services):
        """skills_list should list available skills."""
        result = await call_commerce_tool("skills_list", {})
        text = get_text_content(result)

        assert "Available skills:" in text
        # Should include some canonical skills
        assert "research" in text.lower() or "coding" in text.lower()

    @pytest.mark.asyncio
    async def test_skills_search_found(self, configured_services):
        """skills_search should find matching skills."""
        result = await call_commerce_tool("skills_search", {"query": "data"})
        text = get_text_content(result)

        # Should find data-analysis
        assert "data" in text.lower()

    @pytest.mark.asyncio
    async def test_skills_search_not_found(self, configured_services):
        """skills_search should handle no results."""
        result = await call_commerce_tool("skills_search", {"query": "nonexistentskill123"})
        text = get_text_content(result)

        assert "No skills found" in text


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test error handling in MCP tools."""

    @pytest.mark.asyncio
    async def test_unknown_tool(self, configured_services):
        """Unknown tools should return error."""
        result = await call_commerce_tool("unknown_tool", {})
        text = get_text_content(result)

        assert "Unknown tool" in text

    @pytest.mark.asyncio
    async def test_invalid_job_id(self, configured_services):
        """Invalid job ID should return error."""
        result = await call_commerce_tool("job_fund", {"job_id": "nonexistent-id"})
        text = get_text_content(result)

        assert "not found" in text.lower()

    @pytest.mark.asyncio
    async def test_unauthorized_fund(self, configured_services):
        """Funding someone else's job should fail."""
        job_service = configured_services["job"]

        # Create job by another agent
        job = job_service.create_job(
            client_id="other-agent",
            title="Not My Job",
            description="Someone else's work",
            budget_usdc=50.0,
            deadline=datetime.now(timezone.utc) + timedelta(days=7),
        )

        result = await call_commerce_tool("job_fund", {"job_id": job.id})
        text = get_text_content(result)

        assert "Not authorized" in text or "client can fund" in text.lower()

    @pytest.mark.asyncio
    async def test_invalid_budget(self, configured_services):
        """Negative budget should fail validation."""
        deadline = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()

        result = await call_commerce_tool("job_create", {
            "title": "Bad Job",
            "description": "Invalid budget",
            "budget": -10,
            "deadline": deadline,
        })
        text = get_text_content(result)

        assert "Invalid" in text or "must be" in text.lower()

    @pytest.mark.asyncio
    async def test_missing_required_field(self, configured_services):
        """Missing required fields should fail."""
        result = await call_commerce_tool("job_apply", {
            "job_id": "some-id",
            # Missing message
        })
        text = get_text_content(result)

        assert "Invalid" in text or "required" in text.lower() or "cannot be empty" in text.lower()


# =============================================================================
# Agent ID Tests
# =============================================================================

class TestAgentIdManagement:
    """Test agent ID management."""

    def test_set_and_get_agent_id(self):
        """set_commerce_agent_id should update get_commerce_agent_id."""
        reset_commerce_services()

        set_commerce_agent_id("new-agent-id")
        assert get_commerce_agent_id() == "new-agent-id"

    def test_default_agent_id(self):
        """Default agent ID should be 'default'."""
        reset_commerce_services()
        assert get_commerce_agent_id() == "default"

    @pytest.mark.asyncio
    async def test_different_agents_different_wallets(self, wallet_service, job_service, skill_registry):
        """Different agents should have different wallets."""
        configure_commerce_services(
            wallet_service=wallet_service,
            job_service=job_service,
            skill_registry=skill_registry,
        )

        # Agent 1
        set_commerce_agent_id("agent-1")
        result1 = await call_commerce_tool("wallet_address", {})
        addr1 = get_text_content(result1)

        # Agent 2
        set_commerce_agent_id("agent-2")
        result2 = await call_commerce_tool("wallet_address", {})
        addr2 = get_text_content(result2)

        # Addresses should be different
        assert addr1 != addr2
