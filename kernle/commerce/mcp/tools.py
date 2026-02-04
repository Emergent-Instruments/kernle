"""
Commerce MCP tool implementations.

Provides MCP tools for AI agents to interact with commerce features:
- Wallet operations (balance, address, status)
- Job marketplace (create, apply, deliver, approve)
- Skills registry (list, search)

These tools are designed for Model Context Protocol integration,
following the same patterns as kernle.mcp.server.
"""

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from mcp.types import TextContent, Tool

from kernle.commerce.jobs.models import Job, JobApplication, JobStatus
from kernle.commerce.jobs.service import (
    ApplicationNotFoundError,
    DuplicateApplicationError,
    InvalidTransitionError,
    JobExpiredError,
    JobNotFoundError,
    JobSearchFilters,
    JobService,
    JobServiceError,
    UnauthorizedError,
)
from kernle.commerce.jobs.storage import InMemoryJobStorage
from kernle.commerce.skills.models import Skill
from kernle.commerce.skills.registry import InMemorySkillRegistry
from kernle.commerce.wallet.models import WalletAccount
from kernle.commerce.wallet.service import (
    WalletNotFoundError,
    WalletService,
    WalletServiceError,
)
from kernle.commerce.wallet.storage import InMemoryWalletStorage
from kernle.mcp.sanitize import (
    sanitize_array,
    sanitize_string,
    validate_number,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Global State
# =============================================================================

_commerce_agent_id: str = "default"
_wallet_service: Optional[WalletService] = None
_job_service: Optional[JobService] = None
_skill_registry: Optional[InMemorySkillRegistry] = None


def set_commerce_agent_id(agent_id: str) -> None:
    """Set the agent ID for commerce operations.

    Args:
        agent_id: The Kernle agent ID to use for commerce tools
    """
    global _commerce_agent_id
    _commerce_agent_id = agent_id


def get_commerce_agent_id() -> str:
    """Get the current agent ID for commerce operations."""
    return _commerce_agent_id


def _get_wallet_service() -> WalletService:
    """Get or create wallet service singleton."""
    global _wallet_service
    if _wallet_service is None:
        storage = InMemoryWalletStorage()
        _wallet_service = WalletService(storage)
    return _wallet_service


def _get_job_service() -> JobService:
    """Get or create job service singleton."""
    global _job_service
    if _job_service is None:
        storage = InMemoryJobStorage()
        _job_service = JobService(storage)
    return _job_service


def _get_skill_registry() -> InMemorySkillRegistry:
    """Get or create skill registry singleton."""
    global _skill_registry
    if _skill_registry is None:
        _skill_registry = InMemorySkillRegistry()
    return _skill_registry


def configure_commerce_services(
    wallet_service: Optional[WalletService] = None,
    job_service: Optional[JobService] = None,
    skill_registry: Optional[InMemorySkillRegistry] = None,
) -> None:
    """Configure commerce services for dependency injection.

    Allows tests and external code to inject their own service instances.

    Args:
        wallet_service: WalletService instance to use
        job_service: JobService instance to use
        skill_registry: Skill registry instance to use
    """
    global _wallet_service, _job_service, _skill_registry
    if wallet_service is not None:
        _wallet_service = wallet_service
    if job_service is not None:
        _job_service = job_service
    if skill_registry is not None:
        _skill_registry = skill_registry


def reset_commerce_services() -> None:
    """Reset all commerce services to None (for testing)."""
    global _wallet_service, _job_service, _skill_registry, _commerce_agent_id
    _wallet_service = None
    _job_service = None
    _skill_registry = None
    _commerce_agent_id = "default"


# =============================================================================
# Input Validation
# =============================================================================
# Note: sanitize_string, sanitize_array, validate_number are imported
# from kernle.mcp.sanitize to avoid duplication


def validate_datetime(value: Any, field_name: str, required: bool = True) -> Optional[datetime]:
    """Validate and parse datetime values."""
    if value is None:
        if required:
            raise ValueError(f"{field_name} is required")
        return None

    if isinstance(value, datetime):
        return value

    if isinstance(value, str):
        try:
            # Handle ISO format
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            raise ValueError(f"{field_name} must be a valid ISO datetime string")

    raise ValueError(f"{field_name} must be a datetime or ISO string")


def validate_job_id(value: Any) -> str:
    """Validate job ID format."""
    return sanitize_string(value, "job_id", max_length=100, required=True)


def validate_application_id(value: Any) -> str:
    """Validate application ID format."""
    return sanitize_string(value, "application_id", max_length=100, required=True)


# =============================================================================
# Tool Definitions
# =============================================================================

COMMERCE_TOOLS: List[Tool] = [
    # =========================================================================
    # Wallet Tools
    # =========================================================================
    Tool(
        name="wallet_balance",
        description="Get the USDC balance of your wallet on Base. Returns the current balance and wallet address.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="wallet_address",
        description="Get your wallet's Ethereum address on Base. This is the address where you receive USDC payments.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="wallet_status",
        description="Get your wallet status including spending limits. Shows whether the wallet is active, pending claim, paused, or frozen.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    # =========================================================================
    # Job Tools (Client)
    # =========================================================================
    Tool(
        name="job_create",
        description="Create a new job listing in the marketplace. As a client, you post work that agents can apply to complete.",
        inputSchema={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short job title (max 200 characters)",
                },
                "description": {
                    "type": "string",
                    "description": "Full job description explaining the work required",
                },
                "budget": {
                    "type": "number",
                    "description": "Payment amount in USDC",
                },
                "deadline": {
                    "type": "string",
                    "description": "Deadline for delivery (ISO 8601 format, e.g., '2024-02-15T23:59:59Z')",
                },
                "skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Required skills (e.g., ['research', 'writing', 'coding'])",
                },
            },
            "required": ["title", "description", "budget", "deadline"],
        },
    ),
    Tool(
        name="job_list",
        description="List jobs. Filter by status or list only jobs you've posted (mine=true).",
        inputSchema={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": [
                        "open",
                        "funded",
                        "accepted",
                        "delivered",
                        "completed",
                        "disputed",
                        "cancelled",
                    ],
                    "description": "Filter by job status",
                },
                "mine": {
                    "type": "boolean",
                    "description": "If true, only list jobs you posted as client",
                    "default": False,
                },
            },
        },
    ),
    Tool(
        name="job_fund",
        description="Fund a job's escrow contract. This deploys and funds the escrow, making the job available for workers.",
        inputSchema={
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "ID of the job to fund",
                },
            },
            "required": ["job_id"],
        },
    ),
    Tool(
        name="job_applications",
        description="List applications for a job you posted. Review applicants before accepting one.",
        inputSchema={
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "ID of the job",
                },
            },
            "required": ["job_id"],
        },
    ),
    Tool(
        name="job_accept",
        description="Accept an application and assign the worker to your job. The job must be funded first.",
        inputSchema={
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "ID of the job",
                },
                "application_id": {
                    "type": "string",
                    "description": "ID of the application to accept",
                },
            },
            "required": ["job_id", "application_id"],
        },
    ),
    Tool(
        name="job_approve",
        description="Approve the deliverable and release payment to the worker. Use after reviewing the delivered work.",
        inputSchema={
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "ID of the job to approve",
                },
            },
            "required": ["job_id"],
        },
    ),
    Tool(
        name="job_cancel",
        description="Cancel a job you posted. If funded, the escrow will be refunded to you.",
        inputSchema={
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "ID of the job to cancel",
                },
            },
            "required": ["job_id"],
        },
    ),
    Tool(
        name="job_dispute",
        description="Raise a dispute on a job. Use when there's a disagreement about the work or payment.",
        inputSchema={
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "ID of the job to dispute",
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for the dispute",
                },
            },
            "required": ["job_id", "reason"],
        },
    ),
    # =========================================================================
    # Job Tools (Worker)
    # =========================================================================
    Tool(
        name="job_search",
        description="Search for available jobs to work on. Filter by skills, budget range, or text query.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Text to search in job title and description",
                },
                "skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Skills to filter by (e.g., ['research', 'coding'])",
                },
                "min_budget": {
                    "type": "number",
                    "description": "Minimum budget in USDC",
                },
                "max_budget": {
                    "type": "number",
                    "description": "Maximum budget in USDC",
                },
            },
        },
    ),
    Tool(
        name="job_apply",
        description="Apply to work on a job. Include a message explaining why you're suited for the work.",
        inputSchema={
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "ID of the job to apply to",
                },
                "message": {
                    "type": "string",
                    "description": "Application message explaining your qualifications",
                },
            },
            "required": ["job_id", "message"],
        },
    ),
    Tool(
        name="job_deliver",
        description="Submit your deliverable for a job you're working on. Provide a URL to the work.",
        inputSchema={
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "ID of the job",
                },
                "url": {
                    "type": "string",
                    "description": "URL to the deliverable (IPFS, GitHub, etc.)",
                },
                "hash": {
                    "type": "string",
                    "description": "Optional content hash for verification",
                },
            },
            "required": ["job_id", "url"],
        },
    ),
    # =========================================================================
    # Skills Tools
    # =========================================================================
    Tool(
        name="skills_list",
        description="List all available skills in the registry. Skills are used to tag jobs and match workers.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="skills_search",
        description="Search for skills by name or description.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
            },
            "required": ["query"],
        },
    ),
]


# =============================================================================
# Tool Handlers
# =============================================================================


def _format_wallet_balance(wallet: WalletAccount, balance: Decimal) -> str:
    """Format wallet balance for display."""
    return f"Wallet: {wallet.wallet_address}\nBalance: {balance} USDC"


def _format_wallet_status(wallet: WalletAccount) -> str:
    """Format wallet status for display."""
    lines = [
        f"Wallet Address: {wallet.wallet_address}",
        f"Chain: {wallet.chain}",
        f"Status: {wallet.status}",
        f"Per-Transaction Limit: {wallet.spending_limit_per_tx} USDC",
        f"Daily Limit: {wallet.spending_limit_daily} USDC",
    ]
    if wallet.owner_eoa:
        lines.append(f"Owner EOA: {wallet.owner_eoa}")
    if wallet.claimed_at:
        lines.append(f"Claimed: {wallet.claimed_at.isoformat()}")
    return "\n".join(lines)


def _format_job(job: Job, detail: bool = False) -> str:
    """Format job for display."""
    lines = [
        f"Job: {job.title}",
        f"ID: {job.id}",
        f"Budget: {job.budget_usdc} USDC",
        f"Status: {job.status}",
        f"Deadline: {job.deadline.isoformat() if job.deadline else 'N/A'}",
    ]
    if detail:
        lines.append(f"Description: {job.description}")
        if job.skills_required:
            lines.append(f"Skills: {', '.join(job.skills_required)}")
        if job.worker_id:
            lines.append(f"Worker: {job.worker_id}")
        if job.deliverable_url:
            lines.append(f"Deliverable: {job.deliverable_url}")
    return "\n".join(lines)


def _format_application(app: JobApplication) -> str:
    """Format job application for display."""
    lines = [
        f"Application ID: {app.id}",
        f"Applicant: {app.applicant_id}",
        f"Status: {app.status}",
        f"Message: {app.message[:200]}{'...' if len(app.message) > 200 else ''}",
    ]
    if app.proposed_deadline:
        lines.append(f"Proposed Deadline: {app.proposed_deadline.isoformat()}")
    return "\n".join(lines)


def _format_skill(skill: Skill) -> str:
    """Format skill for display."""
    parts = [skill.name]
    if skill.description:
        parts.append(f"- {skill.description}")
    if skill.category:
        parts.append(f"[{skill.category}]")
    return " ".join(parts)


async def handle_wallet_balance(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle wallet_balance tool call."""
    agent_id = get_commerce_agent_id()
    service = _get_wallet_service()

    try:
        wallet = service.get_wallet_for_agent(agent_id)
        balance = service.get_balance(wallet.id)
        return [TextContent(type="text", text=_format_wallet_balance(wallet, balance.usdc_balance))]
    except WalletNotFoundError:
        # Auto-create wallet if it doesn't exist
        wallet = service.create_wallet(agent_id)
        balance = service.get_balance(wallet.id)
        return [
            TextContent(
                type="text",
                text=f"Wallet created!\n{_format_wallet_balance(wallet, balance.usdc_balance)}",
            )
        ]


async def handle_wallet_address(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle wallet_address tool call."""
    agent_id = get_commerce_agent_id()
    service = _get_wallet_service()

    try:
        wallet = service.get_wallet_for_agent(agent_id)
    except WalletNotFoundError:
        wallet = service.create_wallet(agent_id)

    return [TextContent(type="text", text=f"Wallet Address: {wallet.wallet_address}")]


async def handle_wallet_status(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle wallet_status tool call."""
    agent_id = get_commerce_agent_id()
    service = _get_wallet_service()

    try:
        wallet = service.get_wallet_for_agent(agent_id)
    except WalletNotFoundError:
        wallet = service.create_wallet(agent_id)

    return [TextContent(type="text", text=_format_wallet_status(wallet))]


async def handle_job_create(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle job_create tool call."""
    agent_id = get_commerce_agent_id()
    service = _get_job_service()

    # Validate inputs
    title = sanitize_string(arguments.get("title"), "title", max_length=200)
    description = sanitize_string(arguments.get("description"), "description", max_length=5000)
    budget = validate_number(arguments.get("budget"), "budget", min_val=0.01)
    deadline = validate_datetime(arguments.get("deadline"), "deadline")
    skills = sanitize_array(arguments.get("skills"), "skills", item_max_length=50, max_items=10)

    job = service.create_job(
        client_id=agent_id,
        title=title,
        description=description,
        budget_usdc=budget,
        deadline=deadline,
        skills_required=skills,
    )

    return [TextContent(type="text", text=f"Job created!\n{_format_job(job, detail=True)}")]


async def handle_job_list(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle job_list tool call."""
    agent_id = get_commerce_agent_id()
    service = _get_job_service()

    status_str = arguments.get("status")
    mine = arguments.get("mine", False)

    filters = JobSearchFilters()
    if status_str:
        try:
            filters.status = JobStatus(status_str)
        except ValueError:
            return [TextContent(type="text", text=f"Invalid status: {status_str}")]

    if mine:
        filters.client_id = agent_id

    jobs = service.list_jobs(filters)

    if not jobs:
        return [TextContent(type="text", text="No jobs found.")]

    lines = [f"Found {len(jobs)} job(s):\n"]
    for job in jobs:
        lines.append(_format_job(job))
        lines.append("")

    return [TextContent(type="text", text="\n".join(lines))]


async def handle_job_fund(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle job_fund tool call."""
    agent_id = get_commerce_agent_id()
    service = _get_job_service()

    job_id = validate_job_id(arguments.get("job_id"))

    # Generate stub escrow address (in real impl, this deploys a contract)
    # Ethereum addresses are 42 chars: 0x + 40 hex chars
    # UUID hex is only 32 chars, so we combine two to get 40
    import uuid

    hash1 = uuid.uuid5(uuid.NAMESPACE_DNS, job_id).hex
    hash2 = uuid.uuid5(uuid.NAMESPACE_URL, job_id).hex
    escrow_address = f"0x{hash1}{hash2[:8]}"

    job = service.fund_job(
        job_id=job_id,
        actor_id=agent_id,
        escrow_address=escrow_address,
    )

    return [
        TextContent(type="text", text=f"Job funded!\nEscrow: {escrow_address}\n{_format_job(job)}")
    ]


async def handle_job_applications(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle job_applications tool call."""
    agent_id = get_commerce_agent_id()
    service = _get_job_service()

    job_id = validate_job_id(arguments.get("job_id"))

    # Service layer handles authorization via actor_id
    try:
        applications = service.list_applications(job_id=job_id, actor_id=agent_id)
    except UnauthorizedError:
        return [
            TextContent(type="text", text="You can only view applications for jobs you posted.")
        ]

    if not applications:
        return [TextContent(type="text", text="No applications yet.")]

    lines = [f"Found {len(applications)} application(s):\n"]
    for app in applications:
        lines.append(_format_application(app))
        lines.append("")

    return [TextContent(type="text", text="\n".join(lines))]


async def handle_job_accept(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle job_accept tool call."""
    agent_id = get_commerce_agent_id()
    service = _get_job_service()

    job_id = validate_job_id(arguments.get("job_id"))
    application_id = validate_application_id(arguments.get("application_id"))

    job, application = service.accept_application(
        job_id=job_id,
        application_id=application_id,
        actor_id=agent_id,
    )

    return [
        TextContent(
            type="text",
            text=f"Application accepted!\nWorker {application.applicant_id} assigned to job.\n{_format_job(job)}",
        )
    ]


async def handle_job_approve(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle job_approve tool call."""
    agent_id = get_commerce_agent_id()
    service = _get_job_service()

    job_id = validate_job_id(arguments.get("job_id"))

    job = service.approve_job(
        job_id=job_id,
        actor_id=agent_id,
    )

    return [
        TextContent(
            type="text", text=f"Job approved! Payment released to worker.\n{_format_job(job)}"
        )
    ]


async def handle_job_cancel(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle job_cancel tool call."""
    agent_id = get_commerce_agent_id()
    service = _get_job_service()

    job_id = validate_job_id(arguments.get("job_id"))

    job = service.cancel_job(
        job_id=job_id,
        actor_id=agent_id,
    )

    return [TextContent(type="text", text=f"Job cancelled.\n{_format_job(job)}")]


async def handle_job_dispute(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle job_dispute tool call."""
    agent_id = get_commerce_agent_id()
    service = _get_job_service()

    job_id = validate_job_id(arguments.get("job_id"))
    reason = sanitize_string(arguments.get("reason"), "reason", max_length=1000)

    job = service.dispute_job(
        job_id=job_id,
        actor_id=agent_id,
        reason=reason,
    )

    return [
        TextContent(
            type="text",
            text=f"Dispute raised. An arbitrator will review the case.\n{_format_job(job)}",
        )
    ]


async def handle_job_search(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle job_search tool call."""
    service = _get_job_service()

    query = arguments.get("query")
    if query:
        query = sanitize_string(query, "query", max_length=500, required=False)

    skills = sanitize_array(arguments.get("skills"), "skills", item_max_length=50, max_items=10)
    min_budget = arguments.get("min_budget")
    if min_budget is not None:
        min_budget = validate_number(min_budget, "min_budget", min_val=0)
    max_budget = arguments.get("max_budget")
    if max_budget is not None:
        max_budget = validate_number(max_budget, "max_budget", min_val=0)

    jobs = service.search_jobs(
        query=query if query else None,
        skills=skills if skills else None,
        min_budget=min_budget,
        max_budget=max_budget,
    )

    if not jobs:
        return [TextContent(type="text", text="No jobs found matching your criteria.")]

    lines = [f"Found {len(jobs)} available job(s):\n"]
    for job in jobs:
        lines.append(_format_job(job, detail=True))
        lines.append("")

    return [TextContent(type="text", text="\n".join(lines))]


async def handle_job_apply(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle job_apply tool call."""
    agent_id = get_commerce_agent_id()
    service = _get_job_service()

    job_id = validate_job_id(arguments.get("job_id"))
    message = sanitize_string(arguments.get("message"), "message", max_length=2000)

    application = service.apply_to_job(
        job_id=job_id,
        applicant_id=agent_id,
        message=message,
    )

    return [
        TextContent(type="text", text=f"Application submitted!\n{_format_application(application)}")
    ]


async def handle_job_deliver(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle job_deliver tool call."""
    agent_id = get_commerce_agent_id()
    service = _get_job_service()

    job_id = validate_job_id(arguments.get("job_id"))
    url = sanitize_string(arguments.get("url"), "url", max_length=2000)
    hash_val = arguments.get("hash")
    if hash_val:
        hash_val = sanitize_string(hash_val, "hash", max_length=100, required=False)

    job = service.deliver_job(
        job_id=job_id,
        actor_id=agent_id,
        deliverable_url=url,
        deliverable_hash=hash_val if hash_val else None,
    )

    return [
        TextContent(
            type="text",
            text=f"Deliverable submitted! Awaiting client approval.\n{_format_job(job, detail=True)}",
        )
    ]


async def handle_skills_list(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle skills_list tool call."""
    registry = _get_skill_registry()

    skills = registry.list_skills()

    if not skills:
        return [TextContent(type="text", text="No skills registered.")]

    lines = ["Available skills:\n"]
    for skill in skills:
        lines.append(f"  • {_format_skill(skill)}")

    return [TextContent(type="text", text="\n".join(lines))]


async def handle_skills_search(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle skills_search tool call."""
    registry = _get_skill_registry()

    query = sanitize_string(arguments.get("query"), "query", max_length=100)

    skills = registry.search_skills(query)

    if not skills:
        return [TextContent(type="text", text=f"No skills found matching '{query}'.")]

    lines = [f"Skills matching '{query}':\n"]
    for skill in skills:
        lines.append(f"  • {_format_skill(skill)}")

    return [TextContent(type="text", text="\n".join(lines))]


# =============================================================================
# Tool Router
# =============================================================================

TOOL_HANDLERS = {
    # Wallet
    "wallet_balance": handle_wallet_balance,
    "wallet_address": handle_wallet_address,
    "wallet_status": handle_wallet_status,
    # Job (client)
    "job_create": handle_job_create,
    "job_list": handle_job_list,
    "job_fund": handle_job_fund,
    "job_applications": handle_job_applications,
    "job_accept": handle_job_accept,
    "job_approve": handle_job_approve,
    "job_cancel": handle_job_cancel,
    "job_dispute": handle_job_dispute,
    # Job (worker)
    "job_search": handle_job_search,
    "job_apply": handle_job_apply,
    "job_deliver": handle_job_deliver,
    # Skills
    "skills_list": handle_skills_list,
    "skills_search": handle_skills_search,
}


def handle_commerce_tool_error(
    e: Exception, tool_name: str, arguments: Dict[str, Any]
) -> List[TextContent]:
    """Handle tool errors with appropriate messages."""
    if isinstance(e, ValueError):
        logger.warning(f"Invalid input for tool {tool_name}: {e}")
        return [TextContent(type="text", text=f"Invalid input: {str(e)}")]

    if isinstance(e, (WalletNotFoundError, JobNotFoundError, ApplicationNotFoundError)):
        logger.warning(f"Resource not found for tool {tool_name}: {e}")
        return [TextContent(type="text", text=str(e))]

    if isinstance(e, UnauthorizedError):
        logger.warning(f"Unauthorized for tool {tool_name}: {e}")
        return [TextContent(type="text", text=f"Not authorized: {str(e)}")]

    if isinstance(e, InvalidTransitionError):
        logger.warning(f"Invalid transition for tool {tool_name}: {e}")
        return [TextContent(type="text", text=f"Invalid operation: {str(e)}")]

    if isinstance(e, DuplicateApplicationError):
        logger.warning(f"Duplicate application for tool {tool_name}: {e}")
        return [TextContent(type="text", text="You have already applied to this job.")]

    if isinstance(e, JobExpiredError):
        logger.warning(f"Job expired for tool {tool_name}: {e}")
        return [TextContent(type="text", text="This job has expired.")]

    if isinstance(e, (WalletServiceError, JobServiceError)):
        logger.warning(f"Service error for tool {tool_name}: {e}")
        return [TextContent(type="text", text=str(e))]

    # Unknown error - log full details but return generic message
    logger.error(
        f"Internal error in tool {tool_name}",
        extra={
            "tool_name": tool_name,
            "error_type": type(e).__name__,
            "error_message": str(e),
        },
        exc_info=True,
    )
    return [TextContent(type="text", text="Internal server error")]


async def call_commerce_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Route and execute a commerce tool call.

    Args:
        name: Tool name
        arguments: Tool arguments

    Returns:
        List of TextContent results
    """
    handler = TOOL_HANDLERS.get(name)
    if not handler:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    try:
        return await handler(arguments)
    except Exception as e:
        return handle_commerce_tool_error(e, name, arguments)


def get_commerce_tools() -> List[Tool]:
    """Get all commerce MCP tool definitions."""
    return COMMERCE_TOOLS
