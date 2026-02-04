"""Key management CLI commands for Kernle Comms.

Commands for managing agent cryptographic keys:
- kernle keys generate - Generate a new key pair
- kernle keys show - Show public key info
- kernle keys rotate - Rotate to a new key pair
- kernle keys export - Export public key
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

    from kernle import Kernle


def cmd_keys(args: "argparse.Namespace", k: "Kernle") -> None:
    """Handle keys subcommands."""
    action = getattr(args, "keys_action", None)

    if action == "generate":
        _generate(args, k)
    elif action == "show":
        _show(args, k)
    elif action == "rotate":
        _rotate(args, k)
    elif action == "export":
        _export(args, k)
    elif action == "delete":
        _delete(args, k)
    else:
        print("Usage: kernle keys <generate|show|rotate|export|delete>")
        print()
        print("Commands:")
        print("  generate  Generate a new Ed25519 key pair")
        print("  show      Show key info (public key, key ID)")
        print("  rotate    Rotate to a new key pair (archives old)")
        print("  export    Export public key for sharing")
        print("  delete    Delete the key pair")


def _generate(args: "argparse.Namespace", k: "Kernle") -> None:
    """Generate a new key pair."""
    from kernle.comms.crypto import KeyAlreadyExistsError, KeyManager

    force = getattr(args, "force", False)
    manager = KeyManager(agent_id=k.agent_id)

    try:
        key_pair = manager.generate(force=force)

        print("✓ Generated Ed25519 key pair")
        print(f"  Agent: {k.agent_id}")
        print(f"  Key ID: {key_pair.key_id}")
        print(f"  Public: {key_pair.public_key[:20]}...")
        print()
        print("Your public key can be shared with other agents.")
        print("Your private key is stored locally and never leaves this machine.")

    except KeyAlreadyExistsError:
        print(f"✗ Key already exists for '{k.agent_id}'")
        print("  Use --force to overwrite, or 'kernle keys rotate' to rotate")


def _show(args: "argparse.Namespace", k: "Kernle") -> None:
    """Show key info."""
    import json

    from kernle.comms.crypto import KeyManager, KeyNotFoundError

    manager = KeyManager(agent_id=k.agent_id)

    try:
        key_pair = manager.get_key_pair()

        if getattr(args, "json", False):
            data = {
                "agent_id": k.agent_id,
                "key_id": key_pair.key_id,
                "public_key": key_pair.public_key,
                "created_at": key_pair.created_at.isoformat() if key_pair.created_at else None,
            }
            print(json.dumps(data, indent=2))
            return

        print(f"Agent: {k.agent_id}")
        print(f"Key ID: {key_pair.key_id}")
        print(f"Public Key: {key_pair.public_key}")
        if key_pair.created_at:
            print(f"Created: {key_pair.created_at.strftime('%Y-%m-%d %H:%M UTC')}")

    except KeyNotFoundError:
        print(f"✗ No key found for '{k.agent_id}'")
        print("  Use 'kernle keys generate' to create one")


def _rotate(args: "argparse.Namespace", k: "Kernle") -> None:
    """Rotate to a new key pair."""
    from kernle.comms.crypto import KeyManager

    manager = KeyManager(agent_id=k.agent_id)

    if not getattr(args, "force", False) and manager.has_key():
        confirm = input("Rotate to a new key pair? This will archive the old key. [y/N] ")
        if confirm.lower() != "y":
            print("Cancelled")
            return

    new_key, old_key = manager.rotate()

    print("✓ Rotated key pair")
    print(f"  Agent: {k.agent_id}")
    print(f"  New Key ID: {new_key.key_id}")
    if old_key:
        print(f"  Old Key ID: {old_key.key_id} (archived)")
    print()
    print("Remember to update your registry profile with the new public key:")
    print("  kernle comms update --public-key")


def _export(args: "argparse.Namespace", k: "Kernle") -> None:
    """Export public key."""
    from kernle.comms.crypto import KeyManager, KeyNotFoundError

    manager = KeyManager(agent_id=k.agent_id)

    try:
        public_key = manager.get_public_key()

        # Check if output file specified
        output_path = getattr(args, "output", None)
        if output_path:
            from pathlib import Path

            Path(output_path).write_text(public_key)
            print(f"✓ Exported public key to: {output_path}")
        else:
            # Print to stdout for piping
            print(public_key)

    except KeyNotFoundError:
        print(f"✗ No key found for '{k.agent_id}'")
        print("  Use 'kernle keys generate' to create one")


def _delete(args: "argparse.Namespace", k: "Kernle") -> None:
    """Delete the key pair."""
    from kernle.comms.crypto import KeyManager

    manager = KeyManager(agent_id=k.agent_id)

    if not manager.has_key():
        print(f"✗ No key found for '{k.agent_id}'")
        return

    if not getattr(args, "force", False):
        confirm = input(f"Delete key pair for '{k.agent_id}'? This cannot be undone. [y/N] ")
        if confirm.lower() != "y":
            print("Cancelled")
            return

    manager.delete()
    print(f"✓ Deleted key pair for '{k.agent_id}'")


def add_keys_parser(subparsers) -> None:
    """Add keys subcommand to argument parser."""
    keys_parser = subparsers.add_parser(
        "keys",
        help="Cryptographic key management",
        description="Manage Ed25519 keys for agent identity and message signing",
    )

    keys_sub = keys_parser.add_subparsers(dest="keys_action")

    # Generate
    gen = keys_sub.add_parser("generate", help="Generate a new Ed25519 key pair")
    gen.add_argument("--force", "-f", action="store_true", help="Overwrite existing key")

    # Show
    show = keys_sub.add_parser("show", help="Show key info")
    show.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # Rotate
    rot = keys_sub.add_parser("rotate", help="Rotate to a new key pair")
    rot.add_argument("--force", "-f", action="store_true", help="Skip confirmation")

    # Export
    exp = keys_sub.add_parser("export", help="Export public key")
    exp.add_argument("--output", "-o", help="Output file path")

    # Delete
    dele = keys_sub.add_parser("delete", help="Delete the key pair")
    dele.add_argument("--force", "-f", action="store_true", help="Skip confirmation")
