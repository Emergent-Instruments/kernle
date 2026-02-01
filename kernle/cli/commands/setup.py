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


def _get_memory_flush_prompt(agent_id: str) -> str:
    """Generate the memory flush prompt for pre-compaction checkpoint saving."""
    return f"""Before compaction, save your working state to Kernle:

```bash
kernle -a {agent_id} checkpoint "<describe your current task>" --context "<progress and next steps>"
```

IMPORTANT: Be specific about what you're actually working on.
- ❌ Bad: "Heartbeat complete" or "Saving state"
- ✅ Good: "Building auth API - finished /login endpoint, next: add JWT validation"

The checkpoint should answer: "What exactly am I doing and what's next?"

After saving, continue with compaction."""


def _deep_merge(base: dict, updates: dict) -> dict:
    """Deep merge updates into base dict, preserving existing values."""
    result = base.copy()
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def setup_clawdbot(agent_id: str, force: bool = False) -> None:
    """Install Clawdbot/moltbot hook for automatic memory loading and checkpoint saving."""
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

    # Configure clawdbot.json with both hook enablement and memoryFlush
    config_path = Path.home() / ".clawdbot" / "clawdbot.json"
    config_updated = False

    # Build the config updates we want to apply
    memory_flush_prompt = _get_memory_flush_prompt(agent_id)
    kernle_config = {
        "hooks": {"internal": {"enabled": True, "entries": {"kernle-load": {"enabled": True}}}},
        "agents": {
            "defaults": {
                "compaction": {"memoryFlush": {"enabled": True, "prompt": memory_flush_prompt}}
            }
        },
    }

    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)

            # Check current state
            hook_enabled = (
                config.get("hooks", {})
                .get("internal", {})
                .get("entries", {})
                .get("kernle-load", {})
                .get("enabled", False)
            )

            flush_configured = (
                config.get("agents", {})
                .get("defaults", {})
                .get("compaction", {})
                .get("memoryFlush", {})
                .get("enabled", False)
            )

            if hook_enabled and flush_configured and not force:
                print("✓ Kernle already fully configured")
                print("  - Session start hook: enabled")
                print("  - Pre-compaction flush: enabled")
            else:
                # Merge our config into existing
                merged = _deep_merge(config, kernle_config)

                with open(config_path, "w") as f:
                    json.dump(merged, f, indent=2)

                config_updated = True
                print("✓ Updated clawdbot.json with Kernle configuration")

                if not hook_enabled:
                    print("  - Enabled session start hook")
                if not flush_configured:
                    print("  - Configured pre-compaction memory flush")

        except Exception as e:
            print(f"⚠️  Could not update config: {e}")
            print("\n   Manual configuration required. Add to ~/.clawdbot/clawdbot.json:")
            print(json.dumps(kernle_config, indent=2))
    else:
        # Create new config file
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w") as f:
                json.dump(kernle_config, f, indent=2)
            config_updated = True
            print("✓ Created clawdbot.json with Kernle configuration")
        except Exception as e:
            print(f"⚠️  Could not create config: {e}")
            print("\n   Manual configuration required. Create ~/.clawdbot/clawdbot.json:")
            print(json.dumps(kernle_config, indent=2))

    # Summary
    print("\n" + "=" * 50)
    print("Kernle Setup Complete")
    print("=" * 50)
    print()
    print("Configured for seamless context transitions:")
    print("  1. Session start: Memory auto-loads into KERNLE.md")
    print("  2. Pre-compaction: Agent saves checkpoint before compaction")
    print()

    if config_updated:
        print("⚠️  Restart Clawdbot gateway for changes to take effect:")
        print("   clawdbot doctor --restart")
        print()

    print(f"Memory will persist across sessions for agent '{agent_id}'")


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
    print("\nNext steps:")
    print(f"  1. Start a new Claude Code session in {'~' if global_install else 'this directory'}")
    print(f"  2. Memory will load automatically for agent '{agent_id}'")
    print("\nVerify with: claude")
    print('Then ask: "What are my current values and goals?"')


def cmd_setup(args, k: "Kernle"):
    """Install platform hooks for automatic Kernle memory loading.

    Examples:
        kernle setup clawdbot              # Install for Clawdbot
        kernle setup claude-code            # Install for Claude Code (project)
        kernle setup claude-code --global   # Install for Claude Code (all projects)
        kernle setup cowork                 # Install for Cowork (same as claude-code)
    """
    platform = getattr(args, "platform", None)
    force = getattr(args, "force", False)
    global_install = getattr(args, "global", False)
    agent_id = k.agent_id

    if not platform:
        print("Available platforms:")
        print("  clawdbot      - Clawdbot/moltbot automatic memory loading")
        print("  claude-code   - Claude Code SessionStart hook")
        print("  cowork        - Cowork (same as claude-code)")
        print()
        print("Usage: kernle setup <platform>")
        return

    if platform == "clawdbot":
        setup_clawdbot(agent_id, force)
    elif platform in ("claude-code", "cowork"):
        setup_claude_code(agent_id, force, global_install)
    else:
        print(f"❌ Unknown platform: {platform}")
        print("Available: clawdbot, claude-code, cowork")
