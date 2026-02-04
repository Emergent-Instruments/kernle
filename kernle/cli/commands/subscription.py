"""Subscription management CLI commands for Kernle Cloud.

Provides command-line interface for subscription/tier management:
- kernle sub tier         — Show current tier, usage, renewal info
- kernle sub upgrade <t>  — Upgrade to a higher tier
- kernle sub downgrade <t> — Downgrade to a lower tier
- kernle sub cancel       — Cancel auto-renewal
- kernle sub usage        — Show detailed usage for current period
- kernle sub payments     — List payment history
"""

import json
import logging
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import argparse

    from kernle import Kernle

logger = logging.getLogger(__name__)

# ── Tier definitions (mirrors CLOUD_PAYMENTS_SPEC.md §4) ────────────────────

TIERS = {
    "free": {
        "name": "Free",
        "price": 0,
        "storage": "10 MB",
        "storage_bytes": 10 * 1024 * 1024,
        "stacks": 1,
        "sync": "Unlimited",
    },
    "core": {
        "name": "Core",
        "price": 5,
        "storage": "100 MB",
        "storage_bytes": 100 * 1024 * 1024,
        "stacks": 3,
        "sync": "Unlimited",
        "overflow_agent": 1.50,
        "overflow_storage": 0.50,
    },
    "pro": {
        "name": "Pro",
        "price": 15,
        "storage": "1 GB",
        "storage_bytes": 1024 * 1024 * 1024,
        "stacks": 10,
        "sync": "Unlimited",
        "overflow_agent": 1.00,
        "overflow_storage": 0.50,
    },
    "enterprise": {
        "name": "Enterprise",
        "price": None,  # custom
        "storage": "Unlimited",
        "storage_bytes": None,
        "stacks": None,  # unlimited
        "sync": "Unlimited",
    },
}

TIER_ORDER = ["free", "core", "pro", "enterprise"]

# ── Helpers ──────────────────────────────────────────────────────────────────


def _load_credentials() -> dict:
    """Load credentials from ~/.kernle/credentials.json."""
    creds_path = Path.home() / ".kernle" / "credentials.json"
    if not creds_path.exists():
        return {}
    try:
        with open(creds_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _get_backend_url(creds: dict) -> Optional[str]:
    """Extract backend URL from credentials."""
    return creds.get("backend_url")


def _get_auth_token(creds: dict) -> Optional[str]:
    """Extract auth token from credentials, trying multiple field names."""
    return creds.get("token") or creds.get("auth_token") or creds.get("api_key")


def _api_request(
    method: str,
    path: str,
    creds: dict,
    body: Optional[dict] = None,
    timeout: int = 30,
) -> dict:
    """Make an authenticated HTTP request to the Kernle backend API.

    Returns the parsed JSON response body.
    Raises SystemExit on auth/connection failures with friendly messages.
    """
    backend_url = _get_backend_url(creds)
    token = _get_auth_token(creds)

    if not backend_url:
        print("✗ Backend not configured")
        print("  Run `kernle auth login` or set KERNLE_BACKEND_URL")
        sys.exit(1)
    if not token:
        print("✗ Not authenticated")
        print("  Run `kernle auth login` or set KERNLE_AUTH_TOKEN")
        sys.exit(1)

    url = f"{backend_url.rstrip('/')}{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    data = json.dumps(body).encode("utf-8") if body else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print("✗ Authentication failed (token expired?)")
            print("  Run `kernle auth login` to re-authenticate")
            sys.exit(1)
        if e.code == 403:
            print("✗ Forbidden — you don't have permission for this action")
            sys.exit(1)
        # Try to parse error body
        try:
            err_body = json.loads(e.read().decode("utf-8"))
            detail = err_body.get("detail") or err_body.get("error") or str(err_body)
        except Exception as parse_err:
            logger.debug(f"Failed to parse error response: {parse_err}")
            detail = e.reason
        print(f"✗ API error ({e.code}): {detail}")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"✗ Could not connect to {backend_url}")
        print(f"  {e.reason}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Request failed: {e}")
        sys.exit(1)


# ── Formatting helpers ───────────────────────────────────────────────────────


def _progress_bar(used: float, limit: float, width: int = 20) -> str:
    """Render a Unicode progress bar: █████░░░░░ 45%"""
    if limit <= 0:
        return "█" * width + " ∞"
    ratio = min(used / limit, 1.0)
    filled = int(ratio * width)
    empty = width - filled
    pct = ratio * 100
    # Colour hint via prefix emoji
    if pct >= 90:
        icon = "🔴"
    elif pct >= 70:
        icon = "🟡"
    else:
        icon = "🟢"
    return f"{icon} {'█' * filled}{'░' * empty} {pct:.0f}%"


def _format_bytes(b: int) -> str:
    """Human-readable byte size."""
    if b < 1024:
        return f"{b} B"
    elif b < 1024**2:
        return f"{b / 1024:.1f} KB"
    elif b < 1024**3:
        return f"{b / (1024 ** 2):.1f} MB"
    else:
        return f"{b / (1024 ** 3):.2f} GB"


def _format_date(iso_str: Optional[str]) -> str:
    """Friendly date string from ISO 8601."""
    if not iso_str:
        return "—"
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception as e:
        logger.debug(f"Failed to parse date '{iso_str}': {e}")
        return iso_str[:19]


def _relative_time(iso_str: Optional[str]) -> str:
    """'3 days from now', '2 hours ago', etc."""
    if not iso_str:
        return "—"
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = dt - now
        secs = delta.total_seconds()
        if abs(secs) < 60:
            return "just now"
        minutes = abs(secs) / 60
        hours = minutes / 60
        days = hours / 24
        suffix = "from now" if secs > 0 else "ago"
        if days >= 1:
            return f"{int(days)}d {suffix}"
        if hours >= 1:
            return f"{int(hours)}h {suffix}"
        return f"{int(minutes)}m {suffix}"
    except Exception as e:
        logger.debug(f"Failed to compute relative time for '{iso_str}': {e}")
        return iso_str[:10]


def _tier_label(tier: str) -> str:
    """Colourful tier label."""
    info = TIERS.get(tier, {})
    name = info.get("name", tier.title())
    icons = {"free": "🆓", "core": "⚡", "pro": "🚀", "enterprise": "🏢"}
    return f"{icons.get(tier, '•')} {name}"


# ── Subcommand handlers ─────────────────────────────────────────────────────


def _cmd_tier(args: "argparse.Namespace", k: "Kernle") -> None:
    """Show current tier, usage summary, and renewal info."""
    creds = _load_credentials()
    data = _api_request("GET", "/api/v1/subscriptions/me", creds)

    output_json = getattr(args, "json", False)
    if output_json:
        print(json.dumps(data, indent=2))
        return

    tier = data.get("tier", "free")
    status = data.get("status", "active")
    auto_renew = data.get("auto_renew", False)
    renews_at = data.get("renews_at")
    renewal_amount = data.get("renewal_amount")
    wallet_balance = data.get("wallet_balance")
    can_renew = data.get("can_renew")

    # Usage summary (may be embedded or require separate call)
    storage_used = data.get("storage_used", 0)
    storage_limit = data.get("storage_limit", TIERS.get(tier, {}).get("storage_bytes", 0) or 0)
    stacks_used = data.get("agents_used", data.get("stacks_used", 0))
    stacks_limit = data.get(
        "agents_limit", data.get("stacks_limit", TIERS.get(tier, {}).get("stacks") or 0)
    )

    print()
    print("  Kernle Cloud Subscription")
    print("  " + "═" * 40)
    print()
    print(f"  Tier:    {_tier_label(tier)}")

    status_icons = {
        "active": "✓ Active",
        "grace_period": "⚠ Grace Period",
        "cancelled": "✗ Cancelled",
        "expired": "✗ Expired",
    }
    print(f"  Status:  {status_icons.get(status, status)}")
    print()

    # Storage bar
    if storage_limit and storage_limit > 0:
        print(f"  Storage: {_format_bytes(storage_used)} / {_format_bytes(storage_limit)}")
        print(f"           {_progress_bar(storage_used, storage_limit)}")
    else:
        print(f"  Storage: {_format_bytes(storage_used)} (unlimited)")
    print()

    # Stacks bar
    if stacks_limit and stacks_limit > 0:
        print(f"  Stacks:  {stacks_used} / {stacks_limit}")
        print(f"           {_progress_bar(stacks_used, stacks_limit)}")
    else:
        print(f"  Stacks:  {stacks_used} (unlimited)")
    print()

    # Renewal info
    if renews_at:
        print(f"  Renews:  {_format_date(renews_at)} ({_relative_time(renews_at)})")
        if renewal_amount is not None:
            print(f"  Amount:  ${float(renewal_amount):.2f} USDC")
        if auto_renew:
            if can_renew is False:
                print("  ⚠ Auto-renew ON but wallet balance too low!")
                if wallet_balance is not None:
                    print(f"    Wallet balance: ${float(wallet_balance):.2f} USDC")
            else:
                print("  ✓ Auto-renew enabled")
        else:
            print("  ✗ Auto-renew disabled")
    elif status == "cancelled":
        cancelled_at = data.get("cancelled_at")
        if cancelled_at:
            print(f"  Cancelled: {_format_date(cancelled_at)}")
    print()

    # Hint
    if tier == "free":
        print("  💡 Upgrade with: kernle sub upgrade core")
    elif tier == "core":
        print("  💡 Upgrade with: kernle sub upgrade pro")
    print()


def _cmd_upgrade(args: "argparse.Namespace", k: "Kernle") -> None:
    """Interactive upgrade flow."""
    target_tier = args.tier.lower()
    output_json = getattr(args, "json", False)

    if target_tier not in TIERS:
        print(f"✗ Unknown tier '{target_tier}'")
        print(f"  Available tiers: {', '.join(TIER_ORDER)}")
        sys.exit(1)

    if target_tier == "enterprise":
        print("🏢 Enterprise tier requires custom arrangement.")
        print("   Contact: hello@kernle.io")
        sys.exit(0)

    creds = _load_credentials()

    # Fetch current subscription
    current = _api_request("GET", "/api/v1/subscriptions/me", creds)
    current_tier = current.get("tier", "free")

    if current_tier == target_tier:
        print(f"You're already on the {_tier_label(target_tier)} tier.")
        sys.exit(0)

    cur_idx = TIER_ORDER.index(current_tier) if current_tier in TIER_ORDER else 0
    tgt_idx = TIER_ORDER.index(target_tier)
    if tgt_idx <= cur_idx:
        print(f"✗ {target_tier.title()} is not an upgrade from {current_tier.title()}.")
        print(f"  Use `kernle sub downgrade {target_tier}` instead.")
        sys.exit(1)

    cur_info = TIERS[current_tier]
    tgt_info = TIERS[target_tier]
    price = tgt_info["price"]

    # ── Show comparison ──────────────────────────────────────────────────
    if not output_json:
        print()
        print("  Upgrade Comparison")
        print("  " + "═" * 46)
        print()
        header = f"  {'':18} {'Current':>12}  →  {'New':>12}"
        print(header)
        print("  " + "─" * 46)
        print(f"  {'Tier':<18} {cur_info['name']:>12}  →  {tgt_info['name']:>12}")
        cur_price = f"${cur_info['price']}" if cur_info["price"] else "Free"
        tgt_price = f"${tgt_info['price']}/mo"
        print(f"  {'Price':<18} {cur_price:>12}  →  {tgt_price:>12}")
        print(f"  {'Storage':<18} {cur_info['storage']:>12}  →  {tgt_info['storage']:>12}")
        cur_stacks = str(cur_info["stacks"]) if cur_info["stacks"] else "∞"
        tgt_stacks = str(tgt_info["stacks"]) if tgt_info["stacks"] else "∞"
        print(f"  {'Stacks':<18} {cur_stacks:>12}  →  {tgt_stacks:>12}")
        print(f"  {'Sync':<18} {'Unlimited':>12}  →  {'Unlimited':>12}")
        print()
        print(f"  💰 Cost: ${price:.2f} USDC / month")

        # Show wallet balance if available
        wallet_balance = current.get("wallet_balance")
        if wallet_balance is not None:
            bal = float(wallet_balance)
            print(f"  💳 Wallet balance: ${bal:.2f} USDC", end="")
            if bal < price:
                print("  ⚠ Insufficient!")
            else:
                print("  ✓")
        print()

    # ── Confirm ──────────────────────────────────────────────────────────
    yes_flag = getattr(args, "yes", False)
    if not yes_flag and not output_json:
        try:
            answer = (
                input(f"  Confirm upgrade to {tgt_info['name']} (${price:.2f} USDC)? [y/N] ")
                .strip()
                .lower()
            )
        except (EOFError, KeyboardInterrupt):
            print("\n  Aborted.")
            sys.exit(0)
        if answer not in ("y", "yes"):
            print("  Aborted.")
            sys.exit(0)

    # ── Call API ─────────────────────────────────────────────────────────
    result = _api_request(
        "POST", "/api/v1/subscriptions/upgrade", creds, body={"tier": target_tier}
    )

    if output_json:
        print(json.dumps(result, indent=2))
        return

    payment = result.get("payment", {})
    sub = result.get("subscription", {})
    tx_hash = payment.get("tx_hash")
    pay_status = payment.get("status", "pending")

    if pay_status == "confirmed":
        print(f"  ✓ Upgraded to {_tier_label(target_tier)}!")
        if tx_hash:
            print(f"  📝 Tx: {tx_hash}")
        if sub.get("renews_at"):
            print(f"  📅 Renews: {_format_date(sub['renews_at'])}")
    else:
        # Payment pending — show instructions for manual transfer
        to_addr = payment.get("to")
        amount = payment.get("amount", price)
        print("  ⏳ Payment pending — complete the transfer to activate:")
        print()
        if to_addr:
            print(f"  Treasury: {to_addr}")
        print(f"  Amount:   ${float(amount):.2f} USDC (Base)")
        if tx_hash:
            print(f"  Ref tx:   {tx_hash}")
        print()
        print("  Once confirmed on-chain your tier will update automatically.")
        print("  Check status with: kernle sub tier")
    print()


def _cmd_downgrade(args: "argparse.Namespace", k: "Kernle") -> None:
    """Downgrade to a lower tier (effective at end of current period)."""
    target_tier = args.tier.lower()
    output_json = getattr(args, "json", False)

    if target_tier not in TIERS:
        print(f"✗ Unknown tier '{target_tier}'")
        print(f"  Available tiers: {', '.join(TIER_ORDER)}")
        sys.exit(1)

    creds = _load_credentials()
    current = _api_request("GET", "/api/v1/subscriptions/me", creds)
    current_tier = current.get("tier", "free")

    cur_idx = TIER_ORDER.index(current_tier) if current_tier in TIER_ORDER else 0
    tgt_idx = TIER_ORDER.index(target_tier)
    if tgt_idx >= cur_idx:
        print(f"✗ {target_tier.title()} is not a downgrade from {current_tier.title()}.")
        if tgt_idx > cur_idx:
            print(f"  Use `kernle sub upgrade {target_tier}` instead.")
        sys.exit(1)

    tgt_info = TIERS[target_tier]
    renews_at = current.get("renews_at")

    if not output_json:
        print()
        print(f"  Downgrade: {_tier_label(current_tier)} → {_tier_label(target_tier)}")
        print()
        if renews_at:
            print(f"  ⏳ Takes effect: {_format_date(renews_at)} ({_relative_time(renews_at)})")
            print("     You keep current tier benefits until then.")
        else:
            print("  ⏳ Takes effect immediately (no active billing period).")

        # Warn about potential data loss
        storage_used = current.get("storage_used", 0)
        tgt_limit = tgt_info.get("storage_bytes") or 0
        if tgt_limit and storage_used > tgt_limit:
            print()
            print(
                f"  ⚠ WARNING: You're using {_format_bytes(storage_used)} but "
                f"{tgt_info['name']} only allows {tgt_info['storage']}."
            )
            print("    You may need to clean up data before the downgrade takes effect.")

        stacks_used = current.get("agents_used", current.get("stacks_used", 0))
        tgt_stacks = tgt_info.get("stacks") or 0
        if tgt_stacks and stacks_used > tgt_stacks:
            print(
                f"  ⚠ WARNING: You have {stacks_used} active stacks but "
                f"{tgt_info['name']} allows {tgt_stacks}."
            )
        print()

    yes_flag = getattr(args, "yes", False)
    if not yes_flag and not output_json:
        try:
            answer = input(f"  Confirm downgrade to {tgt_info['name']}? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Aborted.")
            sys.exit(0)
        if answer not in ("y", "yes"):
            print("  Aborted.")
            sys.exit(0)

    result = _api_request(
        "POST", "/api/v1/subscriptions/downgrade", creds, body={"tier": target_tier}
    )

    if output_json:
        print(json.dumps(result, indent=2))
        return

    effective = result.get("effective_at") or result.get("subscription", {}).get("renews_at")
    print(f"  ✓ Downgrade to {_tier_label(target_tier)} scheduled.")
    if effective:
        print(f"  📅 Effective: {_format_date(effective)}")
    print()


def _cmd_cancel(args: "argparse.Namespace", k: "Kernle") -> None:
    """Cancel auto-renewal with confirmation."""
    output_json = getattr(args, "json", False)
    creds = _load_credentials()

    current = _api_request("GET", "/api/v1/subscriptions/me", creds)
    current_tier = current.get("tier", "free")
    renews_at = current.get("renews_at")
    auto_renew = current.get("auto_renew", False)

    if current_tier == "free":
        print("  You're on the Free tier — nothing to cancel.")
        sys.exit(0)

    if not auto_renew:
        print("  Auto-renewal is already disabled.")
        if renews_at:
            print(f"  Your {_tier_label(current_tier)} tier expires {_format_date(renews_at)}.")
        sys.exit(0)

    if not output_json:
        print()
        print(f"  Cancel auto-renewal for {_tier_label(current_tier)}")
        print()
        if renews_at:
            print(f"  Your current period runs until {_format_date(renews_at)}.")
            print("  You'll keep all benefits until then.")
            print("  After that, you'll be downgraded to the Free tier.")
        print()

    yes_flag = getattr(args, "yes", False)
    if not yes_flag and not output_json:
        try:
            answer = input("  Confirm cancellation? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Aborted.")
            sys.exit(0)
        if answer not in ("y", "yes"):
            print("  Aborted.")
            sys.exit(0)

    result = _api_request("POST", "/api/v1/subscriptions/cancel", creds)

    if output_json:
        print(json.dumps(result, indent=2))
        return

    print("  ✓ Auto-renewal cancelled.")
    expires = result.get("expires_at") or renews_at
    if expires:
        print(f"  📅 Benefits until: {_format_date(expires)}")
    print("  💡 You can re-subscribe anytime with: kernle sub upgrade <tier>")
    print()


def _cmd_usage(args: "argparse.Namespace", k: "Kernle") -> None:
    """Show detailed usage for the current billing period."""
    output_json = getattr(args, "json", False)
    creds = _load_credentials()
    data = _api_request("GET", "/api/v1/usage/me", creds)

    if output_json:
        print(json.dumps(data, indent=2))
        return

    period = data.get("period", "—")
    storage_used = data.get("storage_used", 0)
    storage_limit = data.get("storage_limit", 0)
    sync_count = data.get("sync_count", 0)
    last_sync = data.get("last_sync")
    agents_used = data.get("agents_used", data.get("agents_count", 0))
    agents_limit = data.get("agents_limit", 0)

    print()
    print(f"  Usage — Period: {period}")
    print("  " + "═" * 40)
    print()

    # Storage
    print("  📦 Storage")
    if storage_limit > 0:
        print(f"     {_format_bytes(storage_used)} / {_format_bytes(storage_limit)}")
        print(f"     {_progress_bar(storage_used, storage_limit)}")
    else:
        print(f"     {_format_bytes(storage_used)} (unlimited)")
    print()

    # Stacks / Agents
    print("  🗂  Syncing Stacks")
    if agents_limit > 0:
        print(f"     {agents_used} / {agents_limit}")
        print(f"     {_progress_bar(agents_used, agents_limit)}")
    else:
        print(f"     {agents_used} (unlimited)")
    print()

    # Sync activity
    print("  🔄 Sync Activity")
    print(f"     Total syncs this period: {sync_count:,}")
    if last_sync:
        print(f"     Last sync: {_format_date(last_sync)} ({_relative_time(last_sync)})")
    else:
        print("     Last sync: never")
    print()

    # Overflow estimate (if applicable)
    overflow_agents = data.get("overflow_agents", 0)
    overflow_storage = data.get("overflow_storage_gb", 0)
    if overflow_agents > 0 or overflow_storage > 0:
        print("  💸 Overflow (estimated)")
        if overflow_agents > 0:
            cost_per = data.get("overflow_agent_cost", 1.00)
            print(
                f"     +{overflow_agents} extra stacks × ${cost_per:.2f} = ${overflow_agents * cost_per:.2f}"
            )
        if overflow_storage > 0:
            scost = data.get("overflow_storage_cost", 0.50)
            print(
                f"     +{overflow_storage:.2f} GB extra × ${scost:.2f} = ${overflow_storage * scost:.2f}"
            )
        print()


def _cmd_payments(args: "argparse.Namespace", k: "Kernle") -> None:
    """List payment history."""
    output_json = getattr(args, "json", False)
    limit = getattr(args, "limit", 20)
    creds = _load_credentials()
    data = _api_request("GET", f"/api/v1/subscriptions/payments?limit={limit}", creds)

    payments = data if isinstance(data, list) else data.get("payments", [])

    if output_json:
        print(json.dumps(payments, indent=2))
        return

    if not payments:
        print("  No payment history yet.")
        print("  💡 Upgrade with: kernle sub upgrade core")
        return

    print()
    print("  Payment History")
    print("  " + "═" * 60)
    print()

    for p in payments:
        status = p.get("status", "unknown")
        amount = p.get("amount", "0")
        currency = p.get("currency", "USDC")
        tx_hash = p.get("tx_hash", "")
        period_start = p.get("period_start", "")
        period_end = p.get("period_end", "")
        created = p.get("created_at", "")
        confirmed = p.get("confirmed_at")

        status_icons = {
            "confirmed": "✓",
            "pending": "⏳",
            "failed": "✗",
            "refunded": "↩",
        }
        icon = status_icons.get(status, "?")

        print(f"  {icon} ${float(amount):.2f} {currency}  —  {status}")
        if period_start and period_end:
            ps = period_start[:10]
            pe = period_end[:10]
            print(f"    Period: {ps} → {pe}")
        if tx_hash:
            # Shorten tx hash for display
            short_tx = f"{tx_hash[:10]}…{tx_hash[-6:]}" if len(tx_hash) > 20 else tx_hash
            print(f"    Tx: {short_tx}")
        if confirmed:
            print(f"    Confirmed: {_format_date(confirmed)}")
        elif created:
            print(f"    Created: {_format_date(created)}")
        print()


# ── Main dispatcher ──────────────────────────────────────────────────────────


def cmd_subscription(args: "argparse.Namespace", k: "Kernle") -> None:
    """Main dispatcher for subscription subcommands."""
    action = getattr(args, "sub_action", None)

    if action == "tier":
        _cmd_tier(args, k)
    elif action == "upgrade":
        _cmd_upgrade(args, k)
    elif action == "downgrade":
        _cmd_downgrade(args, k)
    elif action == "cancel":
        _cmd_cancel(args, k)
    elif action == "usage":
        _cmd_usage(args, k)
    elif action == "payments":
        _cmd_payments(args, k)
    else:
        # Default: show tier info
        _cmd_tier(args, k)
