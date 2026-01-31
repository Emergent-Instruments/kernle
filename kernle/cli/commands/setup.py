"""Setup command for Kernle CLI - install platform hooks for automatic memory loading."""

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kernle import Kernle


def get_hooks_dir() -> Path:
    """Get the hooks directory from the kernle package."""
    # hooks/ is at the root of the kernle package
    kernle_root = Path(__file__).parent.parent.parent.parent
    return kernle_root / "hooks"


def setup_clawdbot(agent_id: str, force: bool = False, enable: bool = False) -> None:
    """Install Clawdbot/moltbot hook for automatic memory loading.

    Args:
        agent_id: Agent identifier
        force: Overwrite existing hook files
        enable: Automatically enable hook in clawdbot.json
    """
    hooks_dir = get_hooks_dir()
    source = hooks_dir / "clawdbot"

    if not source.exists():
        print("❌ Clawdbot hook files not found in kernle installation")
        print(f"   Expected: {source}")
        return

    # Try user hooks directory first (doesn't require moltbot repo access)
    user_hooks = Path.home() / ".config" / "moltbot" / "hooks" / "kernle-load"
    bundled_hooks = Path.home() / "clawd" / "moltbot" / "src" / "hooks" / "bundled" / "kernle-load"

    # Determine target
    if bundled_hooks.parent.exists():
        target = bundled_hooks
        location = "bundled hooks"
    else:
        target = user_hooks
        location = "user hooks"

    # Check if already exists
    if target.exists() and not force:
        print(f"⚠️  Hook already installed at {target}")
        print("   Use --force to overwrite")
        # Even if files exist, still try to enable if requested
        if enable:
            _enable_clawdbot_hook(agent_id)
        return

    # Create target directory
    target.parent.mkdir(parents=True, exist_ok=True)

    # Copy hook files
    try:
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        print(f"✓ Installed Clawdbot hook to {location}")
        print(f"  Location: {target}")
    except Exception as e:
        print(f"❌ Failed to copy hook files: {e}")
        return

    # Handle enabling in config
    if enable:
        _enable_clawdbot_hook(agent_id)
    else:
        # Check current status and show instructions
        config_path = Path.home() / ".clawdbot" / "clawdbot.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)

                enabled = (
                    config.get("hooks", {})
                    .get("internal", {})
                    .get("entries", {})
                    .get("kernle-load", {})
                    .get("enabled", False)
                )

                if enabled:
                    print("✓ Hook already enabled in config")
                else:
                    print("\n⚠️  Hook not enabled in config")
                    print("   Run with --enable to auto-configure, or add manually:")
                    print(
                        """
{
  "hooks": {
    "internal": {
      "enabled": true,
      "entries": {
        "kernle-load": {
          "enabled": true
        }
      }
    }
  }
}
"""
                    )
            except Exception as e:
                print(f"⚠️  Could not read config: {e}")
        else:
            print(f"\n⚠️  Clawdbot config not found at {config_path}")
            print("   Run with --enable to create config with hook enabled")

        print("\nNext steps:")
        print("  1. Enable hook: kernle setup clawdbot --enable")
        print("  2. Restart Clawdbot gateway: clawdbot gateway restart")
        print(f"  3. Memory will load automatically for agent '{agent_id}'")


def _enable_clawdbot_hook(agent_id: str) -> bool:
    """Enable kernle-load hook in clawdbot.json.

    Returns True if successfully enabled (or already enabled).
    """
    config_path = Path.home() / ".clawdbot" / "clawdbot.json"

    try:
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        else:
            # Create new config
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {}

        # Navigate/create the nested structure
        if "hooks" not in config:
            config["hooks"] = {}
        if "internal" not in config["hooks"]:
            config["hooks"]["internal"] = {}
        if "entries" not in config["hooks"]["internal"]:
            config["hooks"]["internal"]["entries"] = {}

        # Enable internal hooks globally if not set
        if not config["hooks"]["internal"].get("enabled"):
            config["hooks"]["internal"]["enabled"] = True

        # Check if already enabled
        kernle_hook = config["hooks"]["internal"]["entries"].get("kernle-load", {})
        if kernle_hook.get("enabled"):
            print("✓ Hook already enabled in config")
            return True

        # Enable the hook
        config["hooks"]["internal"]["entries"]["kernle-load"] = {"enabled": True}

        # Write back
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print("✓ Enabled kernle-load hook in clawdbot.json")
        print(f"  Config: {config_path}")
        print()
        print("⚠️  Restart Clawdbot gateway for changes to take effect:")
        print("   clawdbot gateway restart")
        return True

    except Exception as e:
        print(f"❌ Failed to enable hook in config: {e}")
        print("   You may need to manually edit ~/.clawdbot/clawdbot.json")
        return False


def setup_claude_code(agent_id: str, force: bool = False, global_install: bool = False) -> None:
    """Install Claude Code/Cowork SessionStart hook."""
    hooks_dir = get_hooks_dir()
    source = hooks_dir / "claude-code" / "settings.json"

    if not source.exists():
        print("❌ Claude Code hook template not found in kernle installation")
        print(f"   Expected: {source}")
        return

    # Determine target
    if global_install:
        target = Path.home() / ".claude" / "settings.json"
        location = "user settings (global)"
    else:
        target = Path.cwd() / ".claude" / "settings.json"
        location = "project settings"

    # Check if already exists
    if target.exists() and not force:
        with open(target) as f:
            content = f.read()
            if "kernle" in content.lower():
                print(f"⚠️  Kernle hook already configured in {target}")
                print("   Use --force to overwrite")
                return

    # Read template
    with open(source) as f:
        template = f.read()

    # Replace placeholder
    config_content = template.replace("YOUR_AGENT_NAME", agent_id)

    # Create target directory
    target.parent.mkdir(parents=True, exist_ok=True)

    # Merge with existing config if present
    if target.exists():
        try:
            with open(target) as f:
                existing = json.load(f)

            new_config = json.loads(config_content)

            # Merge SessionStart hooks
            existing_hooks = existing.get("hooks", {}).get("SessionStart", [])
            new_hooks = new_config["hooks"]["SessionStart"]

            # Check if kernle hook already exists
            has_kernle = any("kernle" in str(h).lower() for h in existing_hooks)

            if not has_kernle:
                existing_hooks.extend(new_hooks)
                if "hooks" not in existing:
                    existing["hooks"] = {}
                existing["hooks"]["SessionStart"] = existing_hooks

                with open(target, "w") as f:
                    json.dump(existing, f, indent=2)
                print(f"✓ Merged Kernle hook into existing {location}")
            else:
                print(f"⚠️  Kernle hook already present in {location}")
        except Exception as e:
            print(f"⚠️  Could not merge with existing config: {e}")
            print("   Writing new config instead")
            with open(target, "w") as f:
                f.write(config_content)
            print(f"✓ Created {location}")
    else:
        # Write new config
        with open(target, "w") as f:
            f.write(config_content)
        print(f"✓ Created {location}")

    print(f"  Location: {target}")
    print(f"\nNext steps:")
    print(f"  1. Start a new Claude Code session in {'~' if global_install else 'this directory'}")
    print(f"  2. Memory will load automatically for agent '{agent_id}'")
    print("\nVerify with: claude")
    print('Then ask: "What are my current values and goals?"')


def cmd_setup(args, k: "Kernle"):
    """Install platform hooks for automatic Kernle memory loading.

    Examples:
        kernle setup clawdbot              # Install for Clawdbot
        kernle setup clawdbot --enable     # Install AND enable in config
        kernle setup claude-code            # Install for Claude Code (project)
        kernle setup claude-code --global   # Install for Claude Code (all projects)
        kernle setup cowork                 # Install for Cowork (same as claude-code)
    """
    platform = getattr(args, "platform", None)
    force = getattr(args, "force", False)
    enable = getattr(args, "enable", False)
    global_install = getattr(args, "global", False)
    agent_id = k.agent_id

    if not platform:
        print("Available platforms:")
        print("  clawdbot      - Clawdbot/moltbot automatic memory loading")
        print("  claude-code   - Claude Code SessionStart hook")
        print("  cowork        - Cowork (same as claude-code)")
        print()
        print("Usage: kernle setup <platform> [--enable] [--force]")
        print()
        print("Options:")
        print("  --enable    Auto-enable hook in config (clawdbot only)")
        print("  --force     Overwrite existing hook files")
        print("  --global    Install globally (claude-code only)")
        return

    if platform == "clawdbot":
        setup_clawdbot(agent_id, force, enable)
    elif platform in ("claude-code", "cowork"):
        setup_claude_code(agent_id, force, global_install)
    else:
        print(f"❌ Unknown platform: {platform}")
        print("Available: clawdbot, claude-code, cowork")
