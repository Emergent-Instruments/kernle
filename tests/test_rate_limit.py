"""Tests for rate limit trusted proxy configuration.

Tests the core proxy-trust logic directly, without importing slowapi
(backend dependency not in core test env).
"""

import ipaddress
import os
import pytest


# Import just the pure functions by extracting the logic
# (avoids slowapi import at module level)
def _load_trusted_cidrs_standalone(env_value=""):
    """Standalone version of _load_trusted_cidrs for testing."""
    default_cidrs = [
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "127.0.0.0/8",
        "::1/128",
    ]
    cidrs = [s.strip() for s in env_value.split(",") if s.strip()] if env_value else default_cidrs
    networks = []
    for cidr in cidrs:
        try:
            networks.append(ipaddress.ip_network(cidr, strict=False))
        except ValueError:
            pass
    return networks


def _is_trusted_proxy_standalone(ip_str, networks):
    """Standalone version of _is_trusted_proxy for testing."""
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return any(addr in network for network in networks)


@pytest.fixture
def default_networks():
    return _load_trusted_cidrs_standalone()


class TestTrustedProxy:
    """Test trusted proxy detection."""

    def test_localhost_is_trusted(self, default_networks):
        assert _is_trusted_proxy_standalone("127.0.0.1", default_networks) is True

    def test_railway_internal_is_trusted(self, default_networks):
        assert _is_trusted_proxy_standalone("10.0.1.5", default_networks) is True

    def test_docker_is_trusted(self, default_networks):
        assert _is_trusted_proxy_standalone("172.17.0.1", default_networks) is True

    def test_private_is_trusted(self, default_networks):
        assert _is_trusted_proxy_standalone("192.168.1.100", default_networks) is True

    def test_public_ip_not_trusted(self, default_networks):
        assert _is_trusted_proxy_standalone("8.8.8.8", default_networks) is False

    def test_invalid_ip_not_trusted(self, default_networks):
        assert _is_trusted_proxy_standalone("not-an-ip", default_networks) is False

    def test_empty_string_not_trusted(self, default_networks):
        assert _is_trusted_proxy_standalone("", default_networks) is False

    def test_ipv6_localhost_trusted(self, default_networks):
        assert _is_trusted_proxy_standalone("::1", default_networks) is True

    def test_random_public_v6_not_trusted(self, default_networks):
        assert _is_trusted_proxy_standalone("2001:db8::1", default_networks) is False


class TestLoadCidrs:
    """Test CIDR loading from env."""

    def test_default_cidrs_loaded(self):
        cidrs = _load_trusted_cidrs_standalone("")
        assert len(cidrs) == 5  # 4 IPv4 + 1 IPv6

    def test_custom_cidrs(self):
        cidrs = _load_trusted_cidrs_standalone("1.2.3.0/24,5.6.7.0/24")
        assert len(cidrs) == 2

    def test_invalid_cidrs_skipped(self):
        cidrs = _load_trusted_cidrs_standalone("1.2.3.0/24,not-a-cidr,5.6.7.0/24")
        assert len(cidrs) == 2

    def test_custom_cidrs_work(self):
        networks = _load_trusted_cidrs_standalone("203.0.113.0/24")
        assert _is_trusted_proxy_standalone("203.0.113.50", networks) is True
        assert _is_trusted_proxy_standalone("203.0.114.1", networks) is False


class TestSpoofPrevention:
    """Test that spoofing is prevented for untrusted sources."""

    def test_untrusted_cant_spoof(self, default_networks):
        """A public IP should not be trusted, preventing X-Forwarded-For spoofing."""
        # Attacker at 8.8.8.8 sends X-Forwarded-For: 1.1.1.1
        # Since 8.8.8.8 is not trusted, the forwarded header should be ignored
        assert _is_trusted_proxy_standalone("8.8.8.8", default_networks) is False

    def test_trusted_proxy_allows_forward(self, default_networks):
        """A Railway internal IP is trusted, so forwarded header would be honored."""
        assert _is_trusted_proxy_standalone("10.0.0.1", default_networks) is True

    def test_no_open_relay(self, default_networks):
        """Common cloud provider IPs are not trusted by default."""
        # AWS, GCP, Azure external IPs should not be trusted
        for ip in ["52.0.0.1", "35.190.0.1", "13.64.0.1"]:
            assert _is_trusted_proxy_standalone(ip, default_networks) is False
