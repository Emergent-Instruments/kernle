"""Comms CLI commands for Kernle.

Commands for agent-to-agent communication:
- kernle comms register - Register as discoverable agent
- kernle comms profile - View agent profile
- kernle comms discover - Find agents by capability
- kernle comms update - Update your profile
"""

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

    from kernle import Kernle


def cmd_comms(args: "argparse.Namespace", k: "Kernle") -> None:
    """Handle comms subcommands."""
    action = getattr(args, "comms_action", None)

    if action == "register":
        _register(args, k)
    elif action == "profile":
        _profile(args, k)
    elif action == "discover":
        _discover(args, k)
    elif action == "update":
        _update(args, k)
    elif action == "delete":
        _delete(args, k)
    else:
        print("Usage: kernle comms <register|profile|discover|update|delete>")
        print()
        print("Commands:")
        print("  register   Register as a discoverable agent")
        print("  profile    View an agent's profile")
        print("  discover   Find agents by capability")
        print("  update     Update your profile")
        print("  delete     Delete your registration")


def _register(args: "argparse.Namespace", k: "Kernle") -> None:
    """Register as a discoverable agent."""
    from kernle.comms.registry import AgentAlreadyExistsError, AgentRegistry

    registry = AgentRegistry(k._storage)

    capabilities = []
    if hasattr(args, "capabilities") and args.capabilities:
        capabilities = [c.strip() for c in args.capabilities.split(",")]

    is_public = getattr(args, "public", False)
    display_name = getattr(args, "name", None)

    # Get user_id from storage config or generate
    user_id = getattr(k._storage, "user_id", None) or f"local:{k.agent_id}"

    try:
        profile = registry.register(
            agent_id=k.agent_id,
            user_id=user_id,
            display_name=display_name,
            capabilities=capabilities,
            is_public=is_public,
        )

        print(f"✓ Registered agent: {profile.agent_id}")
        if profile.display_name:
            print(f"  Name: {profile.display_name}")
        if profile.capabilities:
            print(f"  Capabilities: {', '.join(profile.capabilities)}")
        print(f"  Public: {'Yes' if profile.is_public else 'No'}")
        print(f"  Trust: {profile.trust_level}")

    except AgentAlreadyExistsError:
        print(f"✗ Agent '{k.agent_id}' is already registered")
        print("  Use 'kernle comms update' to modify your profile")


def _profile(args: "argparse.Namespace", k: "Kernle") -> None:
    """View an agent's profile."""
    from kernle.comms.registry import AgentRegistry

    registry = AgentRegistry(k._storage)

    agent_id = getattr(args, "agent_id", None) or k.agent_id
    profile = registry.get_profile(agent_id)

    if not profile:
        print(f"✗ Agent '{agent_id}' not found")
        return

    if getattr(args, "json", False):
        print(json.dumps(profile.to_dict(), indent=2, default=str))
        return

    print(f"Agent: {profile.agent_id}")
    if profile.display_name:
        print(f"Name: {profile.display_name}")
    print(f"User: {profile.user_id}")
    if profile.capabilities:
        print(f"Capabilities: {', '.join(profile.capabilities)}")
    print(f"Trust: {profile.trust_level}")
    print(f"Reputation: {profile.reputation_score:.2f}")
    print(f"Public: {'Yes' if profile.is_public else 'No'}")
    if profile.endpoints:
        print(f"Endpoints: {json.dumps(profile.endpoints)}")
    if profile.registered_at:
        print(f"Registered: {profile.registered_at.strftime('%Y-%m-%d %H:%M UTC')}")
    if profile.last_seen_at:
        print(f"Last seen: {profile.last_seen_at.strftime('%Y-%m-%d %H:%M UTC')}")


def _discover(args: "argparse.Namespace", k: "Kernle") -> None:
    """Find agents by capability."""
    from kernle.comms.registry import AgentRegistry

    registry = AgentRegistry(k._storage)

    capabilities = None
    if hasattr(args, "capability") and args.capability:
        capabilities = [c.strip() for c in args.capability.split(",")]

    limit = getattr(args, "limit", 20)
    results = registry.discover(capabilities=capabilities, limit=limit)

    if not results:
        if capabilities:
            print(f"No agents found with capabilities: {', '.join(capabilities)}")
        else:
            print("No public agents found")
        return

    if getattr(args, "json", False):
        print(json.dumps([p.to_dict() for p in results], indent=2, default=str))
        return

    print(f"Found {len(results)} agent(s):")
    print()
    for profile in results:
        name = profile.display_name or profile.agent_id
        caps = ", ".join(profile.capabilities) if profile.capabilities else "none"
        trust = profile.trust_level
        rep = f"{profile.reputation_score:.2f}"
        print(f"  {name}")
        print(f"    ID: {profile.agent_id}")
        print(f"    Capabilities: {caps}")
        print(f"    Trust: {trust}, Reputation: {rep}")
        print()


def _update(args: "argparse.Namespace", k: "Kernle") -> None:
    """Update your profile."""
    from kernle.comms.registry import AgentNotFoundError, AgentRegistry

    registry = AgentRegistry(k._storage)

    # Build update kwargs
    updates = {}

    if hasattr(args, "name") and args.name is not None:
        updates["display_name"] = args.name

    if hasattr(args, "capabilities") and args.capabilities is not None:
        updates["capabilities"] = [c.strip() for c in args.capabilities.split(",")]

    if hasattr(args, "public") and args.public is not None:
        updates["is_public"] = args.public

    if not updates:
        print("No updates specified")
        print("Use: --name, --capabilities, --public/--private")
        return

    try:
        profile = registry.update_profile(agent_id=k.agent_id, **updates)

        print(f"✓ Updated profile: {profile.agent_id}")
        if "display_name" in updates:
            print(f"  Name: {profile.display_name}")
        if "capabilities" in updates:
            print(f"  Capabilities: {', '.join(profile.capabilities)}")
        if "is_public" in updates:
            print(f"  Public: {'Yes' if profile.is_public else 'No'}")

    except AgentNotFoundError:
        print(f"✗ Agent '{k.agent_id}' not registered")
        print("  Use 'kernle comms register' first")


def _delete(args: "argparse.Namespace", k: "Kernle") -> None:
    """Delete your registration."""
    from kernle.comms.registry import AgentRegistry

    registry = AgentRegistry(k._storage)

    if not getattr(args, "force", False):
        confirm = input(f"Delete registration for '{k.agent_id}'? [y/N] ")
        if confirm.lower() != "y":
            print("Cancelled")
            return

    if registry.delete_profile(k.agent_id):
        print(f"✓ Deleted registration: {k.agent_id}")
    else:
        print(f"✗ Agent '{k.agent_id}' not found")


def add_comms_parser(subparsers) -> None:
    """Add comms subcommand to argument parser."""
    comms_parser = subparsers.add_parser(
        "comms",
        help="Agent-to-agent communication",
        description="Commands for agent discovery and communication",
    )

    comms_sub = comms_parser.add_subparsers(dest="comms_action")

    # Register
    reg = comms_sub.add_parser("register", help="Register as a discoverable agent")
    reg.add_argument("--name", "-n", help="Display name")
    reg.add_argument("--capabilities", "-c", help="Capabilities (comma-separated)")
    reg.add_argument("--public", action="store_true", help="Make agent discoverable")

    # Profile
    prof = comms_sub.add_parser("profile", help="View agent profile")
    prof.add_argument("agent_id", nargs="?", help="Agent ID (default: current agent)")
    prof.add_argument("--json", action="store_true", help="Output as JSON")

    # Discover
    disc = comms_sub.add_parser("discover", help="Find agents by capability")
    disc.add_argument("--capability", "-c", help="Filter by capability (comma-separated)")
    disc.add_argument("--limit", "-l", type=int, default=20, help="Max results")
    disc.add_argument("--json", action="store_true", help="Output as JSON")

    # Update
    upd = comms_sub.add_parser("update", help="Update your profile")
    upd.add_argument("--name", "-n", help="New display name")
    upd.add_argument("--capabilities", "-c", help="New capabilities (comma-separated)")
    upd.add_argument("--public", action="store_true", dest="public", default=None)
    upd.add_argument("--private", action="store_false", dest="public")

    # Delete
    dele = comms_sub.add_parser("delete", help="Delete your registration")
    dele.add_argument("--force", "-f", action="store_true", help="Skip confirmation")
