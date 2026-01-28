"""Authentication routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ..auth import (
    CurrentAgent,
    create_access_token,
    generate_agent_secret,
    generate_user_id,
    hash_secret,
    verify_secret,
)
from ..config import Settings, get_settings
from ..database import Database, create_agent, get_agent
from ..logging_config import get_logger, log_auth_event
from ..models import AgentInfo, AgentLogin, AgentRegister, TokenResponse
from ..rate_limit import limiter

logger = get_logger("kernle.auth")
router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse)
@limiter.limit("5/minute")
async def register_agent(
    request: Request,
    register_request: AgentRegister,
    db: Database,
    settings: Annotated[Settings, Depends(get_settings)],
):
    """
    Register a new agent.

    Returns an access token and the agent's secret (store it safely, shown only once).
    """
    logger.info(f"Registration attempt for agent: {register_request.agent_id}")

    # Check if agent already exists
    existing = await get_agent(db, register_request.agent_id)
    if existing:
        log_auth_event("register", register_request.agent_id, False, "agent already exists")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Agent '{register_request.agent_id}' already exists",
        )

    # Generate user_id and secret
    user_id = generate_user_id()
    secret = generate_agent_secret()
    secret_hash = hash_secret(secret)

    # Create agent
    agent = await create_agent(
        db,
        agent_id=register_request.agent_id,
        secret_hash=secret_hash,
        user_id=user_id,
        display_name=register_request.display_name,
        email=register_request.email,
    )

    if not agent:
        log_auth_event("register", register_request.agent_id, False, "database error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create agent",
        )

    # Generate token with user_id
    token = create_access_token(register_request.agent_id, settings, user_id=user_id)

    log_auth_event("register", register_request.agent_id, True)
    logger.debug(f"Agent {register_request.agent_id} registered with user_id={user_id}")

    return TokenResponse(
        access_token=token,
        expires_in=settings.jwt_expire_minutes * 60,
        user_id=user_id,
        secret=secret,  # One-time display
    )


@router.post("/token", response_model=TokenResponse)
@limiter.limit("5/minute")
async def get_token(
    request: Request,
    login_request: AgentLogin,
    db: Database,
    settings: Annotated[Settings, Depends(get_settings)],
):
    """
    Get an access token for an existing agent.
    """
    agent = await get_agent(db, login_request.agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid agent ID or secret",
        )

    if not verify_secret(login_request.secret, agent["secret_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid agent ID or secret",
        )

    user_id = agent.get("user_id")
    token = create_access_token(login_request.agent_id, settings, user_id=user_id)

    return TokenResponse(
        access_token=token,
        expires_in=settings.jwt_expire_minutes * 60,
        user_id=user_id,
    )


@router.get("/me", response_model=AgentInfo)
async def get_current_agent_info(
    auth: CurrentAgent,
    db: Database,
):
    """
    Get information about the currently authenticated agent.
    """
    agent = await get_agent(db, auth.agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found",
        )

    return AgentInfo(
        agent_id=agent["agent_id"],
        display_name=agent.get("display_name"),
        created_at=agent["created_at"],
        last_sync_at=agent.get("last_sync_at"),
        user_id=agent.get("user_id"),
    )
