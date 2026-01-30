"""Authentication routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel

from ..auth import (
    AUTH_COOKIE_NAME,
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
    TIER_LIMITS,
    Database,
    create_agent,
    create_api_key,
    create_seed_beliefs,
    create_user,
    deactivate_api_key,
    delete_api_key,
    get_agent,
    get_agent_by_email,
    get_api_key,
    get_usage_for_user,
    get_user,
    get_user_by_email,
    list_api_keys,
)
from ..logging_config import get_logger, log_auth_event
from ..models import (
    AgentInfoWithUsage,
    AgentLogin,
    AgentRegister,
    APIKeyCreate,
    APIKeyCycleResponse,
    APIKeyInfo,
    APIKeyList,
    APIKeyResponse,
    LoginResponse,
    TokenResponse,
    UsageLimits,
    UsageResponse,
    UsageStats,
)
from ..rate_limit import limiter

logger = get_logger("kernle.auth")
router = APIRouter(prefix="/auth", tags=["auth"])


def redact_email(email: str | None) -> str:
    """Redact email address for logging (GDPR/CCPA compliance).

    Converts 'user@example.com' to 'u***@e***.com'
    """
    if not email or "@" not in email:
        return "no-email"
    local, domain = email.split("@", 1)
    domain_parts = domain.rsplit(".", 1)
    if len(domain_parts) == 2:
        domain_name, tld = domain_parts
        return f"{local[0]}***@{domain_name[0]}***.{tld}"
    return f"{local[0]}***@***"


class SupabaseTokenExchange(BaseModel):
    """Request to exchange a Supabase access token for a Kernle token."""

    access_token: str


@router.post("/oauth/token", response_model=TokenResponse)
@limiter.limit("10/minute")
async def exchange_supabase_token(
    request: Request,
    response: Response,
    token_request: SupabaseTokenExchange,
    db: Database,
    settings: Annotated[Settings, Depends(get_settings)],
):
    """
    Exchange a Supabase OAuth access token for a Kernle access token.

    This endpoint verifies the Supabase token, extracts user info,
    and creates/returns a Kernle agent + token. Also sets httpOnly cookie.
    """
    try:
        # Verify Supabase JWT using JWKS public key verification
        # This approach fetches public keys from Supabase's well-known endpoint
        # and verifies the JWT locally - NO API KEYS REQUIRED.
        #
        # DO NOT change this to use Supabase API calls with API keys - those
        # can be disabled/rotated and cause auth failures (as happened 2026-01-28).

        import httpx
        from jose import JWTError, jwk, jwt

        token = token_request.access_token

        # Decode header to get the key ID (kid), algorithm, and issuer
        try:
            header = jwt.get_unverified_header(token)
            claims = jwt.get_unverified_claims(token)
            kid = header.get("kid")
            alg = header.get("alg", "RS256")
            issuer = claims.get("iss")

            logger.info(f"OAuth: JWT kid={kid}, alg={alg}, iss={issuer}")
        except Exception as e:
            logger.error(f"OAuth: Failed to decode JWT header: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token format",
            )

        # SECURITY: Validate issuer BEFORE fetching JWKS to prevent SSRF/auth bypass
        # Attacker could craft token with malicious issuer pointing to their JWKS
        # Use STRICT EQUALITY - startswith() is vulnerable to prefix attacks like:
        # supabase.co.evil.com would pass startswith("https://supabase.co")
        expected_issuer = f"{settings.supabase_url}/auth/v1"
        if issuer != expected_issuer:
            logger.error(f"OAuth: Invalid issuer {issuer}, expected {expected_issuer}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token issuer",
            )

        # Use expected issuer for JWKS URL (not the unverified claim)
        # This maintains JWKS-based verification while preventing issuer spoofing
        jwks_url = f"{expected_issuer}/.well-known/jwks.json"

        logger.info(f"OAuth: Fetching JWKS from {jwks_url}")
        try:
            async with httpx.AsyncClient() as client:
                jwks_response = await client.get(jwks_url, timeout=10.0)
                if jwks_response.status_code != 200:
                    logger.error(f"OAuth: JWKS fetch failed: {jwks_response.status_code}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Failed to fetch signing keys",
                    )
                jwks_data = jwks_response.json()
        except httpx.RequestError as e:
            logger.error(f"OAuth: JWKS request failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication service temporarily unavailable",
            )

        # Find the key matching our token's kid
        signing_key = None
        for key in jwks_data.get("keys", []):
            if key.get("kid") == kid:
                signing_key = key
                break

        if not signing_key:
            logger.error(f"OAuth: No matching key found for kid={kid}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token signing key not found",
            )

        # Verify the JWT using the public key from JWKS
        try:
            public_key = jwk.construct(signing_key)

            # Verify with expected issuer (not the unverified claim)
            payload = jwt.decode(
                token,
                public_key,
                algorithms=[alg],
                audience="authenticated",
                issuer=expected_issuer,
            )

            user_data = {
                "id": payload.get("sub"),
                "email": payload.get("email"),
            }
            logger.info(f"OAuth: Verified JWT for user: {redact_email(user_data.get('email'))}")
        except JWTError as e:
            logger.error(f"OAuth: JWT verification failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
            )

        email = user_data.get("email")
        supabase_id = user_data.get("id")

        if not supabase_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid user data",
            )

        # Use supabase user ID as agent_id (prefixed to avoid collisions)
        agent_id = f"oauth_{supabase_id[:12]}"
        display_name = email.split("@")[0] if email else None

        # First, check if there's an existing user with the same email
        # This enables account merging across OAuth providers (Google, GitHub, etc.)
        existing_user = None
        if email:
            existing_user = await get_user_by_email(db, email)

        if existing_user:
            # User exists with this email - use their account
            # Users don't need agents to use the web UI - agents are for Kernle memory instances
            user_id = existing_user.get("user_id")
            logger.info(f"OAuth: Found existing user {user_id} - email {redact_email(email)}")
            token = create_access_token(settings, user_id=user_id)
            log_auth_event("oauth_login", user_id, True)
            set_auth_cookie(response, token, settings)

            return TokenResponse(
                access_token=token,
                expires_in=settings.jwt_expire_minutes * 60,
                user_id=user_id,
            )

        # Check if agent already exists by agent_id (legacy path for backwards compatibility)
        existing_agent = await get_agent(db, agent_id)

        if existing_agent:
            # Legacy agent exists with this OAuth ID - ensure they have a user record
            user_id = existing_agent.get("user_id")
            if not user_id:
                # Legacy agent without user_id - create user for them
                user_id = generate_user_id()
                await create_user(db, user_id=user_id, email=email, display_name=display_name)
                try:
                    db.table("agents").update({"user_id": user_id}).eq(
                        "agent_id", existing_agent.get("agent_id")
                    ).execute()
                except Exception as e:
                    logger.warning(f"OAuth: Failed to update legacy agent with user_id: {e}")

            logger.info(f"OAuth: Found legacy agent, using user {user_id}")
            token = create_access_token(settings, user_id=user_id)
            log_auth_event("oauth_login", user_id, True)
            set_auth_cookie(response, token, settings)

            return TokenResponse(
                access_token=token,
                expires_in=settings.jwt_expire_minutes * 60,
                user_id=user_id,
            )

        # Legacy path: check for existing agent by email (pre-migration accounts)
        # If found, migrate them to users table
        existing_agent_by_email = None
        if email:
            existing_agent_by_email = await get_agent_by_email(db, email)

        if existing_agent_by_email:
            # Found a legacy agent by email - ensure they have a user record
            user_id = existing_agent_by_email.get("user_id")
            if not user_id:
                # Create user for legacy agent
                user_id = generate_user_id()
                await create_user(db, user_id=user_id, email=email, display_name=display_name)
                # Update agent with user_id
                try:
                    db.table("agents").update({"user_id": user_id}).eq(
                        "agent_id", existing_agent_by_email.get("agent_id")
                    ).execute()
                except Exception as e:
                    logger.warning(f"OAuth: Failed to update legacy agent with user_id: {e}")

            logger.info(
                f"OAuth: Found legacy agent for email {redact_email(email)}, using user {user_id}"
            )
            token = create_access_token(settings, user_id=user_id)
            log_auth_event("oauth_login_legacy", user_id, True)
            set_auth_cookie(response, token, settings)

            return TokenResponse(
                access_token=token,
                expires_in=settings.jwt_expire_minutes * 60,
                user_id=user_id,
            )

        # New user - create user record only (NO agent created)
        # Agents are created separately when users actually need Kernle memory features
        user_id = generate_user_id()

        # Create user in users table
        user = await create_user(
            db,
            user_id=user_id,
            email=email,
            display_name=display_name,
        )

        if not user:
            log_auth_event("oauth_register", f"web_{user_id}", False, "failed to create user")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user",
            )

        logger.info(f"OAuth: Created new user {user_id} for {redact_email(email)}")

        token = create_access_token(settings, user_id=user_id)
        log_auth_event("oauth_register", user_id, True)
        set_auth_cookie(response, token, settings)

        return TokenResponse(
            access_token=token,
            expires_in=settings.jwt_expire_minutes * 60,
            user_id=user_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OAuth token exchange error: {type(e).__name__}: {e}")
        import traceback

        logger.debug(f"OAuth traceback: {traceback.format_exc()}")  # DEBUG level for traces
        # Generic error message to clients (security: don't expose internals)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed. Please try again.",
        )


@router.post("/register", response_model=TokenResponse)
@limiter.limit("5/minute")
async def register_agent(
    request: Request,
    response: Response,
    register_request: AgentRegister,
    db: Database,
    settings: Annotated[Settings, Depends(get_settings)],
):
    """
    Register a new agent.

    Returns an access token and the agent's secret (store it safely, shown only once).
    Also sets an httpOnly cookie for browser-based auth.
    """
    logger.info(f"Registration attempt for agent: {register_request.agent_id}")

    # NOTE: We don't check for existing agent_id globally because agent_ids
    # are namespaced per user_id. Each registration creates a new user_id,
    # so the (user_id, agent_id) combination is always unique.
    # Multiple users can have agents with the same name (e.g., "claire").

    # Generate user_id and secret
    user_id = generate_user_id()
    secret = generate_agent_secret()
    secret_hash = hash_secret(secret)

    # Create user first in the users table
    user = await create_user(
        db,
        user_id=user_id,
        email=register_request.email,
        display_name=register_request.display_name,
    )

    if not user:
        log_auth_event("register", register_request.agent_id, False, "failed to create user")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user",
        )

    # Create agent (email now stored on user, not agent)
    agent = await create_agent(
        db,
        agent_id=register_request.agent_id,
        secret_hash=secret_hash,
        user_id=user_id,
        display_name=register_request.display_name,
    )

    if not agent:
        log_auth_event("register", register_request.agent_id, False, "database error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create agent",
        )

    # Plant seed beliefs - foundational SI wisdom for new agents
    try:
        beliefs_created = await create_seed_beliefs(db, register_request.agent_id)
        logger.info(f"Created {beliefs_created} seed beliefs for {register_request.agent_id}")
    except Exception as e:
        # Don't fail registration if seed beliefs fail
        logger.warning(f"Failed to create seed beliefs: {e}")

    # Generate token with both user_id and agent_id
    token = create_access_token(settings, user_id=user_id, agent_id=register_request.agent_id)

    log_auth_event("register", register_request.agent_id, True)
    logger.debug(f"Agent {register_request.agent_id} registered with user_id={user_id}")

    # Set httpOnly cookie for browser auth
    set_auth_cookie(response, token, settings)

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
    token = create_access_token(settings, user_id=user_id, agent_id=login_request.agent_id)

    return TokenResponse(
        access_token=token,
        expires_in=settings.jwt_expire_minutes * 60,
        user_id=user_id,
    )


@router.get("/me", response_model=AgentInfoWithUsage)
async def get_current_user_info(
    auth: CurrentAgent,
    db: Database,
    response: Response,
):
    """
    Get information about the currently authenticated user, including tier and usage.

    Note: This returns user info from the users table. The response model includes
    agent_id for backwards compatibility, but users may not have agents.
    """
    # Get user info from users table (authoritative source)
    user = await get_user(db, auth.user_id)
    if not user:
        # Clear the invalid auth cookie so the user can re-authenticate
        response.delete_cookie(
            AUTH_COOKIE_NAME,
            path="/",
            secure=True,
            httponly=True,
            samesite="strict",  # Must match the samesite used when setting the cookie
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found - please sign in again",
        )

    tier = user.get("tier", "free")
    limits_config = TIER_LIMITS.get(tier, TIER_LIMITS["free"])

    # Get usage stats
    usage_stats = None
    usage_data = await get_usage_for_user(db, auth.user_id)
    if usage_data:
        usage_stats = UsageStats(
            daily_requests=usage_data.get("daily_requests", 0),
            monthly_requests=usage_data.get("monthly_requests", 0),
            daily_reset_at=usage_data.get("daily_reset_at"),
            monthly_reset_at=usage_data.get("monthly_reset_at"),
        )

    return AgentInfoWithUsage(
        agent_id=auth.agent_id or auth.user_id,  # Use user_id if no agent
        display_name=user.get("display_name"),
        created_at=user.get("created_at"),
        last_sync_at=None,  # Users don't sync
        user_id=auth.user_id,
        tier=tier,
        usage=usage_stats,
        limits=UsageLimits(
            daily_limit=limits_config["daily"],
            monthly_limit=limits_config["monthly"],
        ),
    )


@router.get("/usage", response_model=UsageResponse)
async def get_usage_stats(
    auth: CurrentAgent,
    db: Database,
):
    """
    Get current usage statistics for the authenticated user.

    Returns tier, limits, current usage, and remaining quota.
    """
    # Get user info from users table (authoritative source)
    user = await get_user(db, auth.user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    tier = user.get("tier", "free")
    limits_config = TIER_LIMITS.get(tier, TIER_LIMITS["free"])

    # Get usage stats
    usage_data = await get_usage_for_user(db, auth.user_id)

    daily_requests = usage_data.get("daily_requests", 0) if usage_data else 0
    monthly_requests = usage_data.get("monthly_requests", 0) if usage_data else 0

    # Calculate remaining
    daily_limit = limits_config["daily"]
    monthly_limit = limits_config["monthly"]

    daily_remaining = None if daily_limit is None else max(0, daily_limit - daily_requests)
    monthly_remaining = None if monthly_limit is None else max(0, monthly_limit - monthly_requests)

    return UsageResponse(
        tier=tier,
        limits=UsageLimits(
            daily_limit=daily_limit,
            monthly_limit=monthly_limit,
        ),
        usage=UsageStats(
            daily_requests=daily_requests,
            monthly_requests=monthly_requests,
            daily_reset_at=usage_data.get("daily_reset_at") if usage_data else None,
            monthly_reset_at=usage_data.get("monthly_reset_at") if usage_data else None,
        ),
        daily_remaining=daily_remaining,
        monthly_remaining=monthly_remaining,
    )


@router.post("/login", response_model=LoginResponse)
@limiter.limit("30/minute")
async def login_with_api_key(
    request: Request,
    response: Response,
    auth: CurrentAgent,
    db: Database,
    settings: Annotated[Settings, Depends(get_settings)],
):
    """
    Validate API key and return user info with a fresh JWT token.

    This endpoint is primarily for CLI login flow - validates the API key
    (via the CurrentAgent dependency) and returns user context plus a
    fresh JWT for subsequent requests. Also sets httpOnly cookie.
    """
    from datetime import datetime, timedelta, timezone

    agent = await get_agent(db, auth.agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found",
        )

    # Generate a fresh JWT token
    expires_delta = timedelta(minutes=settings.jwt_expire_minutes)
    token = create_access_token(
        settings=settings,
        expires_delta=expires_delta,
        user_id=auth.user_id,
        agent_id=auth.agent_id,
    )

    token_expires = (datetime.now(timezone.utc) + expires_delta).isoformat()

    log_auth_event("login", auth.agent_id, True)
    logger.info(f"API key login successful for {auth.agent_id}")

    # Set httpOnly cookie for browser auth
    set_auth_cookie(response, token, settings)

    return LoginResponse(
        user_id=auth.user_id or agent.get("user_id", ""),
        agent_id=auth.agent_id,
        token=token,
        token_expires=token_expires,
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

    # Create new key first (safer: if this fails, old key still works)
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

    # Deactivate old key (after new one is created)
    # If this fails, user has two active keys (safe, just suboptimal)
    try:
        await deactivate_api_key(db, key_id, auth.user_id)
    except Exception as e:
        # Log but don't fail - new key is created, old one still active
        # User can manually deactivate or it will be cleaned up
        logger.warning(
            f"Failed to deactivate old key {key_id} during cycle for user {auth.user_id}: {e}. "
            "New key created successfully. User may need to manually deactivate old key."
        )

    logger.info(
        f"API key cycled for user {auth.user_id}: {old_key['key_prefix']}... -> {key_prefix}..."
    )

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


# =============================================================================
# Cookie-based Auth Helpers
# =============================================================================

COOKIE_NAME = "kernle_auth"
COOKIE_MAX_AGE = 7 * 24 * 60 * 60  # 7 days in seconds


def set_auth_cookie(response: Response, token: str, settings: Settings):
    """Set httpOnly auth cookie."""
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        secure=True,  # Only send over HTTPS
        samesite="strict",  # CSRF protection - block cross-site requests entirely
        path="/",
    )


def clear_auth_cookie(response: Response):
    """Clear the auth cookie (logout)."""
    response.delete_cookie(
        key=COOKIE_NAME,
        path="/",
        httponly=True,
        secure=True,
        samesite="strict",
    )


@router.post("/logout")
async def logout(response: Response):
    """Clear auth cookie and logout."""
    clear_auth_cookie(response)
    return {"status": "logged_out"}
