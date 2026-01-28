"""Authentication utilities for Kernle backend."""

import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from .config import Settings, get_settings

# Bearer token scheme
security = HTTPBearer()


def generate_user_id() -> str:
    """Generate a stable user_id (usr_ + 12 char hex)."""
    return f"usr_{uuid.uuid4().hex[:12]}"


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
) -> AuthContext:
    """Get the current authenticated agent context from the token."""
    payload = decode_token(credentials.credentials, settings)
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
