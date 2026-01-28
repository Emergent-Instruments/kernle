"""Rate limiting configuration for Kernle backend."""

from slowapi import Limiter
from slowapi.util import get_remote_address

# Create limiter using client IP address as the key
limiter = Limiter(key_func=get_remote_address)
