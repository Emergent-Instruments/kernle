"""Authentication routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from supabase import create_client

from ..auth import (
    CurrentAgent,
    create_access_token,
    generate_agent_secret,
    generate_api_key,
    generate_user_id,
    get_api_key_prefix,
    hash_api_key,
    hash_secret,
    verify_secret,
)
from ..config import Settings, get_settings
from ..database import (
    Database,
    create_agent,
    create_api_key,
    deactivate_api_key,
    delete_api_key,
    get_agent,
    get_api_key,
    list_api_keys,
)
from ..logging_config import get_logger, log_auth_event
from ..models import (
    AgentInfo,
    AgentLogin,
    AgentRegister,
    APIKeyCreate,
    APIKeyCycleResponse,
    APIKeyInfo,
    APIKeyList,
    APIKeyResponse,
    TokenResponse,
)
from ..rate_limit import limiter

logger = get_logger("kernle.auth")
router = APIRouter(prefix="/auth", tags=["auth"])


class SupabaseTokenExchange(BaseModel):
    """Request to exchange a Supabase access token for a Kernle token."""
    access_token: str


@router.post("/oauth/token", response_model=TokenResponse)
@limiter.limit("10/minute")
async def exchange_supabase_token(
    request: Request,
    token_request: SupabaseTokenExchange,
    db: Database,
    settings: Annotated[Settings, Depends(get_settings)],
):
    """
    Exchange a Supabase OAuth access token for a Kernle access token.
    
    This endpoint verifies the Supabase token, extracts user info,
    and creates/returns a Kernle agent + token.
    """
    try:
        # Create a Supabase client with the user's token to verify it
        # We use the publishable key since we're verifying a user token
        api_key = settings.supabase_publishable_key or settings.supabase_anon_key
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Supabase publishable key not configured",
            )
        
        # Create client and get user from token
        supabase = create_client(settings.supabase_url, api_key)
        user_response = supabase.auth.get_user(token_request.access_token)
        
        if not user_response or not user_response.user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired Supabase token",
            )
        
        supabase_user = user_response.user
        email = supabase_user.email
        supabase_id = supabase_user.id
        
        # Use supabase user ID as agent_id (prefixed to avoid collisions)
        agent_id = f"oauth_{supabase_id[:12]}"
        
        # Check if agent already exists
        existing = await get_agent(db, agent_id)
        
        if existing:
            # Agent exists, just issue a new token
            user_id = existing.get("user_id")
            token = create_access_token(agent_id, settings, user_id=user_id)
            log_auth_event("oauth_login", agent_id, True)
            
            return TokenResponse(
                access_token=token,
                expires_in=settings.jwt_expire_minutes * 60,
                user_id=user_id,
            )
        
        # Create new agent for OAuth user
        user_id = generate_user_id()
        # OAuth users don't have a secret (they auth via Supabase)
        # Store a random hash that can never be matched
        secret_hash = hash_secret(generate_agent_secret())
        
        agent = await create_agent(
            db,
            agent_id=agent_id,
            secret_hash=secret_hash,
            user_id=user_id,
            display_name=email.split("@")[0] if email else None,
            email=email,
        )
        
        if not agent:
            log_auth_event("oauth_register", agent_id, False, "database error")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create agent",
            )
        
        token = create_access_token(agent_id, settings, user_id=user_id)
        log_auth_event("oauth_register", agent_id, True)
        logger.info(f"OAuth agent created: {agent_id} for {email}")
        
        return TokenResponse(
            access_token=token,
            expires_in=settings.jwt_expire_minutes * 60,
            user_id=user_id,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OAuth token exchange error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Failed to verify Supabase token",
        )


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


# =============================================================================
# API Key Endpoints
# =============================================================================


@router.post("/keys", response_model=APIKeyResponse)
@limiter.limit("10/minute")
async def create_new_api_key(
    request: Request,
    auth: CurrentAgent,
    db: Database,
    key_request: APIKeyCreate | None = None,
):
    """
    Create a new API key for the authenticated user.
    
    Returns the raw key ONCE - store it safely as it cannot be retrieved again.
    """
    if not auth.user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User ID not found. Please re-register to get a user_id.",
        )
    
    name = key_request.name if key_request else "Default"
    
    # Generate the key
    raw_key = generate_api_key()
    key_hash = hash_api_key(raw_key)
    key_prefix = get_api_key_prefix(raw_key)
    
    # Store in database
    key_record = await create_api_key(
        db,
        user_id=auth.user_id,
        key_hash=key_hash,
        key_prefix=key_prefix,
        name=name,
    )
    
    if not key_record:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create API key",
        )
    
    logger.info(f"API key created for user {auth.user_id}: {key_prefix}...")
    
    return APIKeyResponse(
        id=str(key_record["id"]),
        name=key_record["name"],
        key=raw_key,  # Shown only once!
        key_prefix=key_prefix,
        created_at=key_record["created_at"],
    )


@router.get("/keys", response_model=APIKeyList)
async def list_user_api_keys(
    auth: CurrentAgent,
    db: Database,
):
    """
    List all API keys for the authenticated user.
    
    Returns metadata only - the raw keys are never stored or retrievable.
    """
    if not auth.user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User ID not found.",
        )
    
    keys = await list_api_keys(db, auth.user_id)
    
    return APIKeyList(
        keys=[
            APIKeyInfo(
                id=str(k["id"]),
                name=k["name"],
                key_prefix=f"{k['key_prefix']}...",
                created_at=k["created_at"],
                last_used_at=k.get("last_used_at"),
                is_active=k["is_active"],
            )
            for k in keys
        ]
    )


@router.delete("/keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(
    key_id: str,
    auth: CurrentAgent,
    db: Database,
):
    """
    Revoke (delete) an API key.
    
    The key will immediately stop working.
    """
    if not auth.user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User ID not found.",
        )
    
    # Check key exists and belongs to user
    key = await get_api_key(db, key_id, auth.user_id)
    if not key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )
    
    deleted = await delete_api_key(db, key_id, auth.user_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete API key",
        )
    
    logger.info(f"API key revoked for user {auth.user_id}: {key['key_prefix']}...")


@router.post("/keys/{key_id}/cycle", response_model=APIKeyCycleResponse)
@limiter.limit("10/minute")
async def cycle_api_key(
    request: Request,
    key_id: str,
    auth: CurrentAgent,
    db: Database,
):
    """
    Cycle an API key: deactivate the old one and create a new one atomically.
    
    Returns the new raw key ONCE - store it safely.
    The old key is deactivated (not deleted) for audit purposes.
    """
    if not auth.user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User ID not found.",
        )
    
    # Check old key exists and belongs to user
    old_key = await get_api_key(db, key_id, auth.user_id)
    if not old_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )
    
    if not old_key["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot cycle an already inactive key",
        )
    
    # Generate new key with same name
    raw_key = generate_api_key()
    key_hash = hash_api_key(raw_key)
    key_prefix = get_api_key_prefix(raw_key)
    
    # Create new key
    new_key_record = await create_api_key(
        db,
        user_id=auth.user_id,
        key_hash=key_hash,
        key_prefix=key_prefix,
        name=old_key["name"],
    )
    
    if not new_key_record:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create new API key",
        )
    
    # Deactivate old key (after new one is created for atomicity)
    await deactivate_api_key(db, key_id, auth.user_id)
    
    logger.info(f"API key cycled for user {auth.user_id}: {old_key['key_prefix']}... -> {key_prefix}...")
    
    return APIKeyCycleResponse(
        old_key_id=key_id,
        new_key=APIKeyResponse(
            id=str(new_key_record["id"]),
            name=new_key_record["name"],
            key=raw_key,
            key_prefix=key_prefix,
            created_at=new_key_record["created_at"],
        ),
    )
