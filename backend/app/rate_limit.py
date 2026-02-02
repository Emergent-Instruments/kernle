"""Rate limiting configuration for Kernle backend."""

from slowapi import Limiter
from slowapi.util import get_remote_address


def get_client_ip(request) -> str:
    """Resolve client IP, honoring X-Forwarded-For when present."""
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        # Use the left-most value (original client)
        client_ip = forwarded_for.split(",")[0].strip()
        if client_ip:
            return client_ip
    return get_remote_address(request)


# Create limiter using client IP address as the key
limiter = Limiter(key_func=get_client_ip)
