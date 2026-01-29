"""Authentication utilities for Kernle backend."""

import logging
import secrets
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated

import bcrypt
from cachetools import TTLCache
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from .config import Settings, get_settings

# Cookie name for httpOnly auth
AUTH_COOKIE_NAME = "kernle_auth"

# Bearer token scheme
# Make bearer optional to allow cookie fallback
security = HTTPBearer(auto_error=False)

# API Key prefix
API_KEY_PREFIX = "knl_sk_"

# Quota cache: api_key_id -> (allowed: bool, quota_info: dict)
# TTL of 60 seconds provides resilience while keeping quota reasonably fresh
_quota_cache: TTLCache = TTLCache(maxsize=10000, ttl=60)
_quota_cache_lock = threading.Lock()

logger = logging.getLogger("kernle.auth")


async def check_and_increment_quota_cached(
    db,
    api_key_id: str,
    user_id: str,
    tier: str,
) -> tuple[bool, dict]:
    """
    Atomically check quota and increment usage with caching for resilience.

    Behavior:
    - For denials: cache the denial decision briefly to avoid hammering DB
    - For allows: always go to DB to ensure atomic increment
    - DB error: 503 (fail-closed) - don't allow unlimited requests on errors

    This prevents race conditions where concurrent requests could exceed quota.
    """
    from .database import check_and_increment_quota

    cache_key = f"deny:{api_key_id}"

    # Check if we have a cached denial (don't cache allows - need atomic increment)
    with _quota_cache_lock:
        cached_denial = _quota_cache.get(cache_key)

    if cached_denial is not None:
        logger.debug(f"Quota denial cache hit for {api_key_id[:12]}...")
        return False, cached_denial

    # Call atomic check-and-increment
    try:
        allowed, quota_info = await check_and_increment_quota(db, api_key_id, user_id, tier)

        # Cache denials briefly to reduce DB load
        if not allowed:
            with _quota_cache_lock:
                _quota_cache[cache_key] = quota_info

        return allowed, quota_info
    except Exception as e:
        logger.error(f"Quota check failed for {api_key_id[:12]}...: {e}")
        # Fail closed - don't allow unlimited requests when DB is down
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable",
            headers={"Retry-After": "60"},
        )


def generate_user_id() -> str:
    """Generate a stable user_id (usr_ + 12 char hex)."""
    return f"usr_{uuid.uuid4().hex[:12]}"


def generate_api_key() -> str:
    """Generate an API key in format: knl_sk_ + 32 hex chars."""
    return f"{API_KEY_PREFIX}{secrets.token_hex(16)}"


def get_api_key_prefix(key: str) -> str:
    """Extract prefix from API key for storage (first 12 chars after knl_sk_).

    Using 12 chars gives us 5 hex chars after 'knl_sk_' = 16^5 = ~1M possible
    prefixes, making collision-based timing attacks impractical.
    """
    # Format: knl_sk_XXXXXXXX... -> knl_sk_XXXXX (12 chars for lookup)
    if key.startswith(API_KEY_PREFIX):
        return key[:12]  # "knl_sk_XXXXX" - 5 hex chars = 1M possibilities
    return key[:12]


def hash_api_key(key: str) -> str:
    """Hash an API key using bcrypt."""
    return bcrypt.hashpw(key.encode(), bcrypt.gensalt()).decode()


def verify_api_key(plain_key: str, hashed: str) -> bool:
    """Verify an API key against its hash."""
    import logging

    try:
        return bcrypt.checkpw(plain_key.encode(), hashed.encode())
    except Exception:
        # Log but don't expose details - could be malformed input
        logging.getLogger("kernle.auth").debug(
            "API key verification failed due to encoding/hashing error"
        )
        return False


def is_api_key(token: str) -> bool:
    """Check if a token is an API key (vs JWT)."""
    return token.startswith(API_KEY_PREFIX)


def hash_secret(secret: str) -> str:
    """Hash an agent secret using bcrypt."""
    return bcrypt.hashpw(secret.encode(), bcrypt.gensalt()).decode()


def verify_secret(plain: str, hashed: str) -> bool:
    """Verify an agent secret against hash."""
    import logging

    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        # Log but don't expose details - could be malformed input
        logging.getLogger("kernle.auth").debug(
            "Secret verification failed due to encoding/hashing error"
        )
        return False


def generate_agent_secret() -> str:
    """Generate a secure agent secret."""
    return secrets.token_urlsafe(32)


def create_access_token(
    settings: Settings,
    expires_delta: timedelta | None = None,
    user_id: str | None = None,
    agent_id: str | None = None,
) -> str:
    """Create a JWT access token.

    For OAuth users: user_id is required, agent_id is optional
    For agent auth: both may be provided
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)

    # Use user_id as subject for user auth, fall back to agent_id for legacy
    subject = user_id or agent_id
    if not subject:
        raise ValueError("Either user_id or agent_id must be provided")

    to_encode = {
        "sub": subject,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "access",
    }
    # Include both if provided
    if user_id:
        to_encode["user_id"] = user_id
    if agent_id:
        to_encode["agent_id"] = agent_id
    return jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


# SECURITY: Only allow secure algorithms - prevents algorithm confusion attacks
ALLOWED_JWT_ALGORITHMS = ["HS256", "HS384", "HS512"]


def decode_token(token: str, settings: Settings) -> dict:
    """Decode and validate a JWT token."""
    # Validate algorithm is in allowlist
    if settings.jwt_algorithm not in ALLOWED_JWT_ALGORITHMS:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Invalid JWT configuration",
        )
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],  # Only accept configured algorithm
        )
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


class AuthContext:
    """Authentication context - user-centric with optional agent.

    For OAuth users: user_id is set, agent_id may be None
    For API key auth: both user_id and agent_id are set
    """

    def __init__(
        self,
        user_id: str,
        tier: str = "free",
        is_admin: bool = False,
        agent_id: str | None = None,
        api_key_id: str | None = None,
    ):
        self.user_id = user_id
        self.tier = tier
        self.is_admin = is_admin
        self.agent_id = agent_id  # Only set for API key auth or legacy
        self.api_key_id = api_key_id

    def namespaced_agent_id(self, project_name: str | None = None) -> str:
        """Return full namespaced agent_id: {user_id}/{project_name}.

        If project_name contains '/', it's already namespaced - return as-is.
        If no project_name given, returns the agent_id from token (if any).
        """
        name = project_name or self.agent_id or ""
        # Already namespaced?
        if "/" in name:
            return name
        # Namespace with user_id
        if self.user_id:
            return f"{self.user_id}/{name}"
        return name


async def get_current_agent(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    settings: Annotated[Settings, Depends(get_settings)],
    request: Request,
) -> AuthContext:
    """Get the current authenticated agent context from the token, API key, or cookie."""
    # Try Authorization header first, then fall back to cookie
    token = None
    if credentials:
        token = credentials.credentials
    else:
        # Check for httpOnly cookie
        token = request.cookies.get(AUTH_COOKIE_NAME)

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated - provide Authorization header or auth cookie",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if it's an API key
    if is_api_key(token):
        # Import here to avoid circular imports
        from .database import (
            get_supabase_client,
            verify_api_key_auth,
        )

        db = get_supabase_client(settings)
        auth_result = await verify_api_key_auth(db, token)

        if not auth_result:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or inactive API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

        tier = auth_result.get("tier", "free")
        api_key_id = auth_result.get("api_key_id")
        user_id = auth_result["user_id"]
        is_admin = auth_result.get("is_admin", False)

        # Atomically check quota and increment usage (prevents race conditions)
        allowed, quota_info = await check_and_increment_quota_cached(db, api_key_id, user_id, tier)

        if not allowed:
            # Determine reset time for Retry-After header
            exceeded = quota_info.get("exceeded", "daily")
            reset_at = quota_info.get(f"{exceeded}_reset_at")

            headers = {"WWW-Authenticate": "Bearer"}
            if reset_at:
                # Add reset time headers
                headers["X-RateLimit-Reset"] = reset_at
                headers["X-RateLimit-Exceeded"] = exceeded
                # Calculate seconds until reset for Retry-After
                from datetime import datetime, timezone

                from dateutil.parser import parse

                now = datetime.now(timezone.utc)
                reset_dt = parse(reset_at) if isinstance(reset_at, str) else reset_at
                retry_after = max(1, int((reset_dt - now).total_seconds()))
                headers["Retry-After"] = str(retry_after)

            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {exceeded} quota reached. Resets at {reset_at}",
                headers=headers,
            )

        # Usage increment is now atomic with quota check - no separate call needed

        return AuthContext(
            user_id=user_id,
            tier=tier,
            is_admin=is_admin,
            agent_id=auth_result["agent_id"],
            api_key_id=api_key_id,
        )

    # Otherwise, treat as JWT (no quota for JWT auth - used for web UI)
    payload = decode_token(token, settings)

    # JWT subject is user_id for OAuth users, or agent_id for legacy tokens
    subject = payload.get("sub")
    if not subject:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Determine user_id and agent_id from payload
    # New tokens: sub=user_id, user_id=user_id, agent_id may be present
    # Legacy tokens: sub=agent_id, user_id may be present
    user_id = payload.get("user_id")
    agent_id = payload.get("agent_id")

    # If subject starts with "usr_", it's a user_id (new format)
    # If subject starts with "web_", it's a web user placeholder (transitional)
    # Otherwise it's an agent_id (legacy format)
    if subject.startswith("usr_"):
        user_id = subject
    elif subject.startswith("web_"):
        # Transitional format - extract user_id from web_{user_id}
        user_id = subject[4:]  # Remove "web_" prefix
    elif not user_id:
        # Legacy token with agent_id as subject - agent_id is the subject
        agent_id = subject

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token - no user_id found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get tier and is_admin from users table (fail gracefully)
    tier = "free"
    is_admin = False
    try:
        from .database import get_supabase_client, get_user

        db = get_supabase_client(settings)
        user = await get_user(db, user_id)
        if user:
            tier = user.get("tier", "free")
            is_admin = user.get("is_admin", False)
        else:
            logger.warning(f"User not found in database: {user_id}")
    except Exception as e:
        # Log but continue with defaults - don't block auth
        logger.warning(f"User lookup failed for {user_id}, defaulting to free/non-admin: {e}")

    return AuthContext(user_id=user_id, tier=tier, is_admin=is_admin, agent_id=agent_id)


# Type alias for dependency injection
CurrentAgent = Annotated[AuthContext, Depends(get_current_agent)]


async def require_admin(auth: CurrentAgent) -> AuthContext:
    """Require admin privileges for access.

    Use as a dependency on admin-only routes:
        admin: Annotated[AuthContext, Depends(require_admin)]
    """
    if not auth.is_admin:
        logger.warning(f"Admin access denied: user={auth.user_id} is_admin={auth.is_admin}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )

    logger.info(f"Admin access granted: user={auth.user_id}")
    return auth


# Type alias for admin dependency
AdminAgent = Annotated[AuthContext, Depends(require_admin)]
