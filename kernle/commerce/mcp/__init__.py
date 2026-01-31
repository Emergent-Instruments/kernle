"""MCP tools for Kernle Commerce.

Provides Model Context Protocol tools for AI agents to interact
with commerce features:

Wallet tools:
- wallet_balance(): Get USDC balance
- wallet_address(): Get wallet address
- wallet_status(): Get wallet status and limits

Job tools (client):
- job_create(title, description, budget, deadline, skills=[])
- job_list(status=None, mine=False)
- job_fund(job_id)
- job_applications(job_id)
- job_accept(job_id, application_id)
- job_approve(job_id)
- job_cancel(job_id)
- job_dispute(job_id, reason)

Job tools (worker):
- job_search(query=None, skills=[], min_budget=None, max_budget=None)
- job_apply(job_id, message)
- job_deliver(job_id, url, hash=None)

Skills tools:
- skills_list()
- skills_search(query)

Usage:
    from kernle.commerce.mcp import (
        get_commerce_tools,
        call_commerce_tool,
        set_commerce_agent_id,
        COMMERCE_TOOLS,
    )
    
    # Set the agent ID for the session
    set_commerce_agent_id("my-agent-id")
    
    # Get tool definitions for MCP registration
    tools = get_commerce_tools()
    
    # Call a tool
    result = await call_commerce_tool("wallet_balance", {})
"""

from kernle.commerce.mcp.tools import (
    # Tool definitions
    COMMERCE_TOOLS,
    get_commerce_tools,
    # Tool execution
    call_commerce_tool,
    TOOL_HANDLERS,
    # Session configuration
    set_commerce_agent_id,
    get_commerce_agent_id,
    # Service configuration (for testing/DI)
    configure_commerce_services,
    reset_commerce_services,
    # Error handling
    handle_commerce_tool_error,
)

__all__ = [
    # Tool definitions
    "COMMERCE_TOOLS",
    "get_commerce_tools",
    # Tool execution
    "call_commerce_tool",
    "TOOL_HANDLERS",
    # Session configuration
    "set_commerce_agent_id",
    "get_commerce_agent_id",
    # Service configuration
    "configure_commerce_services",
    "reset_commerce_services",
    # Error handling
    "handle_commerce_tool_error",
]
