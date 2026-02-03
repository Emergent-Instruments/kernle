"""Rate limiting configuration for Kernle backend.

Uses trusted proxy configuration to prevent X-Forwarded-For spoofing.
Only trusts forwarded headers from known proxy IPs (e.g., Railway).
"""

import ipaddress
import os
from typing import Optional

from slowapi import Limiter
from slowapi.util import get_remote_address

# Trusted proxy CIDRs — only these sources can set X-Forwarded-For.
# Railway's proxy IPs, localhost, and private ranges for local dev.
# Override with TRUSTED_PROXY_CIDRS env var (comma-separated CIDRs).
_DEFAULT_TRUSTED_CIDRS = [
    "10.0.0.0/8",  # Railway internal network
    "172.16.0.0/12",  # Docker/private
    "192.168.0.0/16",  # Local dev
    "127.0.0.0/8",  # Localhost
    "::1/128",  # IPv6 localhost
]


def _load_trusted_cidrs() -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    """Load trusted proxy CIDRs from env or defaults."""
    raw = os.environ.get("TRUSTED_PROXY_CIDRS", "")
    cidrs = [s.strip() for s in raw.split(",") if s.strip()] if raw else _DEFAULT_TRUSTED_CIDRS
    networks = []
    for cidr in cidrs:
        try:
            networks.append(ipaddress.ip_network(cidr, strict=False))
        except ValueError:
            # Skip invalid CIDRs, log would be nice but keep it simple
            pass
    return networks


_trusted_networks: Optional[list] = None


def _get_trusted_networks():
    global _trusted_networks
    if _trusted_networks is None:
        _trusted_networks = _load_trusted_cidrs()
    return _trusted_networks


def _is_trusted_proxy(ip_str: str) -> bool:
    """Check if an IP is in the trusted proxy list."""
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return any(addr in network for network in _get_trusted_networks())


def get_client_ip(request) -> str:
    """Resolve client IP, only honoring X-Forwarded-For from trusted proxies.

    If the direct connection is from a trusted proxy (e.g., Railway's
    reverse proxy), we trust the X-Forwarded-For header and use the
    leftmost (original client) IP. Otherwise, we use the direct
    connection IP to prevent spoofing.
    """
    direct_ip = get_remote_address(request)

    # Only trust X-Forwarded-For if the direct connection is from a trusted proxy
    if _is_trusted_proxy(direct_ip):
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
            if client_ip:
                return client_ip

    return direct_ip


# Create limiter using client IP address as the key
limiter = Limiter(key_func=get_client_ip)
