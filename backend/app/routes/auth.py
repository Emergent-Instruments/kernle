"""Authentication routes."""

from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import (
    create_access_token,
    generate_agent_secret,
    hash_secret,
    verify_secret,
    CurrentAgent,
)
from ..config import Settings, get_settings
from ..database import Database, create_agent, get_agent
from ..models import AgentInfo, AgentLogin, AgentRegister, TokenResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse)
async def register_agent(
    request: AgentRegister,
    db: Database,
    settings: Annotated[Settings, Depends(get_settings)],
):
    """
    Register a new agent.
    
    Returns an access token and the agent's secret (store it safely, shown only once).
    """
    # Check if agent already exists
    existing = await get_agent(db, request.agent_id)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Agent '{request.agent_id}' already exists",
        )
    
    # Generate and hash secret
    secret = generate_agent_secret()
    secret_hash = hash_secret(secret)
    
    # Create agent
    agent = await create_agent(
        db,
        agent_id=request.agent_id,
        secret_hash=secret_hash,
        display_name=request.display_name,
        email=request.email,
    )
    
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create agent",
        )
    
    # Generate token
    token = create_access_token(request.agent_id, settings)
    
    return TokenResponse(
        access_token=token,
        expires_in=settings.jwt_expire_minutes * 60,
        # Include secret in response header (one-time display)
        # Client should store this securely
    )


@router.post("/token", response_model=TokenResponse)
async def get_token(
    request: AgentLogin,
    db: Database,
    settings: Annotated[Settings, Depends(get_settings)],
):
    """
    Get an access token for an existing agent.
    """
    agent = await get_agent(db, request.agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid agent ID or secret",
        )
    
    if not verify_secret(request.secret, agent["secret_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid agent ID or secret",
        )
    
    token = create_access_token(request.agent_id, settings)
    
    return TokenResponse(
        access_token=token,
        expires_in=settings.jwt_expire_minutes * 60,
    )


@router.get("/me", response_model=AgentInfo)
async def get_current_agent_info(
    agent_id: CurrentAgent,
    db: Database,
):
    """
    Get information about the currently authenticated agent.
    """
    agent = await get_agent(db, agent_id)
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
    )
