"""Authentication utilities for Kernle backend."""

import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated

import bcrypt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from .config import Settings, get_settings

# Bearer token scheme
security = HTTPBearer()

# API Key prefix
API_KEY_PREFIX = "knl_sk_"


def generate_user_id() -> str:
    """Generate a stable user_id (usr_ + 12 char hex)."""
    return f"usr_{uuid.uuid4().hex[:12]}"


def generate_api_key() -> str:
    """Generate an API key in format: knl_sk_ + 32 hex chars."""
    return f"{API_KEY_PREFIX}{secrets.token_hex(16)}"


def get_api_key_prefix(key: str) -> str:
    """Extract prefix from API key for storage (first 8 chars after knl_sk_)."""
    # Format: knl_sk_XXXXXXXX... -> knl_sk_X (8 chars for display)
    if key.startswith(API_KEY_PREFIX):
        return key[:8]  # "knl_sk_X"
    return key[:8]


def hash_api_key(key: str) -> str:
    """Hash an API key using bcrypt."""
    return bcrypt.hashpw(key.encode(), bcrypt.gensalt()).decode()


def verify_api_key(plain_key: str, hashed: str) -> bool:
    """Verify an API key against its hash."""
    try:
        return bcrypt.checkpw(plain_key.encode(), hashed.encode())
    except Exception:
        return False


def is_api_key(token: str) -> bool:
    """Check if a token is an API key (vs JWT)."""
    return token.startswith(API_KEY_PREFIX)


def hash_secret(secret: str) -> str:
    """Hash an agent secret using bcrypt."""
    return bcrypt.hashpw(secret.encode(), bcrypt.gensalt()).decode()


def verify_secret(plain: str, hashed: str) -> bool:
    """Verify an agent secret against hash."""
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


def generate_agent_secret() -> str:
    """Generate a secure agent secret."""
    return secrets.token_urlsafe(32)


def create_access_token(
    agent_id: str,
    settings: Settings,
    expires_delta: timedelta | None = None,
    user_id: str | None = None,
) -> str:
    """Create a JWT access token for an agent."""
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)

    to_encode = {
        "sub": agent_id,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "access",
    }
    # Include user_id if provided (for namespacing)
    if user_id:
        to_encode["user_id"] = user_id
    return jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_token(token: str, settings: Settings) -> dict:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


class AuthContext:
    """Context from JWT token containing agent_id and user_id."""
    def __init__(self, agent_id: str, user_id: str | None = None):
        self.agent_id = agent_id
        self.user_id = user_id

    def namespaced_agent_id(self, project_name: str | None = None) -> str:
        """Return full namespaced agent_id: {user_id}/{project_name}.
        
        If project_name contains '/', it's already namespaced - return as-is.
        If no project_name given, returns the agent_id from token.
        """
        name = project_name or self.agent_id
        # Already namespaced?
        if "/" in name:
            return name
        # Namespace with user_id
        if self.user_id:
            return f"{self.user_id}/{name}"
        return name


async def get_current_agent(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    settings: Annotated[Settings, Depends(get_settings)],
    request: Request,
) -> AuthContext:
    """Get the current authenticated agent context from the token or API key."""
    token = credentials.credentials
    
    # Check if it's an API key
    if is_api_key(token):
        # Import here to avoid circular imports
        from .database import get_supabase_client, verify_api_key_auth
        
        db = get_supabase_client(settings)
        auth_result = await verify_api_key_auth(db, token)
        
        if not auth_result:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or inactive API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return AuthContext(
            agent_id=auth_result["agent_id"],
            user_id=auth_result["user_id"],
        )
    
    # Otherwise, treat as JWT
    payload = decode_token(token, settings)
    agent_id = payload.get("sub")
    if not agent_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_id = payload.get("user_id")
    return AuthContext(agent_id=agent_id, user_id=user_id)


# Type alias for dependency injection
CurrentAgent = Annotated[AuthContext, Depends(get_current_agent)]
