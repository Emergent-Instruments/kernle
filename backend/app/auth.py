"""Authentication utilities for Kernle backend."""

import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from .config import Settings, get_settings

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Bearer token scheme
security = HTTPBearer()


def hash_secret(secret: str) -> str:
    """Hash an agent secret."""
    return pwd_context.hash(secret)


def verify_secret(plain: str, hashed: str) -> bool:
    """Verify an agent secret against hash."""
    return pwd_context.verify(plain, hashed)


def generate_agent_secret() -> str:
    """Generate a secure agent secret."""
    return secrets.token_urlsafe(32)


def create_access_token(
    agent_id: str,
    settings: Settings,
    expires_delta: timedelta | None = None,
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
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_agent(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> str:
    """Get the current authenticated agent ID from the token."""
    payload = decode_token(credentials.credentials, settings)
    agent_id = payload.get("sub")
    if not agent_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return agent_id


# Type alias for dependency injection
CurrentAgent = Annotated[str, Depends(get_current_agent)]
