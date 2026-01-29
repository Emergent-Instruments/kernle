"""Stats commands for Kernle CLI."""

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kernle import Kernle


def cmd_stats(args, k: "Kernle"):
    """Handle stats subcommands."""
    if args.stats_action == "health-checks":
        _health_checks_stats(args, k)
    else:
        print(f"Unknown stats action: {args.stats_action}")


def _health_checks_stats(args, k: "Kernle"):
    """Show health check compliance statistics."""
    stats = k._storage.get_health_check_stats()

    if args.json:
        print(json.dumps(stats, indent=2, default=str))
        return

    print("\nHealth Check Compliance Stats")
    print("=" * 50)
    print()

    total = stats["total_checks"]
    avg = stats["avg_per_day"]
    last_at = stats["last_check_at"]
    last_score = stats["last_anxiety_score"]

    print(f"Total Checks:     {total}")
    print(f"Avg per Day:      {avg:.1f}")

    if last_at:
        # Calculate time since last check
        try:
            last_dt = datetime.fromisoformat(last_at.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            elapsed = now - last_dt

            if elapsed.days > 0:
                elapsed_str = f"{elapsed.days}d ago"
            elif elapsed.seconds > 3600:
                elapsed_str = f"{elapsed.seconds // 3600}h ago"
            elif elapsed.seconds > 60:
                elapsed_str = f"{elapsed.seconds // 60}m ago"
            else:
                elapsed_str = "just now"

            print(f"Last Check:       {elapsed_str}")
        except Exception:
            print(f"Last Check:       {last_at[:19]}")

        if last_score is not None:
            if last_score >= 80:
                emoji = "🔴"
            elif last_score >= 50:
                emoji = "🟡"
            else:
                emoji = "🟢"
            print(f"Last Score:       {emoji} {last_score}")
    else:
        print("Last Check:       Never")

    # Breakdown by source
    by_source = stats.get("checks_by_source", {})
    if by_source:
        print()
        print("By Source:")
        for source, count in sorted(by_source.items()):
            pct = (count / total * 100) if total > 0 else 0
            print(f"  {source:12} {count:5} ({pct:.0f}%)")

    # Breakdown by trigger
    by_trigger = stats.get("checks_by_trigger", {})
    if by_trigger:
        print()
        print("By Trigger:")
        for trigger, count in sorted(by_trigger.items()):
            pct = (count / total * 100) if total > 0 else 0
            print(f"  {trigger:12} {count:5} ({pct:.0f}%)")

    # Compliance guidance
    print()
    if total == 0:
        print("💡 No health checks recorded yet.")
        print("   Run `kernle anxiety` to perform your first check.")
    elif avg < 1.0:
        print("⚠️  Low check frequency - consider checking more often.")
        print("   Aim for 2-4 checks per day during active sessions.")
    elif avg >= 2.0:
        print("✓ Good compliance rate!")
