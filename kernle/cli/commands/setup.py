"""Setup command for Kernle CLI - install platform hooks for automatic memory loading."""

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kernle import Kernle


def get_hooks_dir() -> Path:
    """Get the hooks directory from the kernle package."""
    # hooks/ is inside the kernle package at kernle/hooks/
    kernle_pkg = Path(__file__).parent.parent.parent
    return kernle_pkg / "hooks"


def _get_memory_flush_prompt(stack_id: str) -> str:
    """Generate the memory flush prompt for pre-compaction checkpoint saving."""
    return f"""Before compaction, save your working state to Kernle:

```bash
kernle -s {stack_id} checkpoint "<describe your current task>" --context "<progress and next steps>"
```

IMPORTANT: Be specific about what you're actually working on.
- Bad: "Heartbeat complete" or "Saving state"
- Good: "Building auth API - finished /login endpoint, next: add JWT validation"

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


def setup_clawdbot(stack_id: str, force: bool = False, enable: bool = False) -> None:
    """Install Clawdbot/moltbot hook for automatic memory loading and checkpoint saving.

    Args:
        stack_id: Stack identifier
        force: Overwrite existing hook files
        enable: Automatically enable hook and configure memoryFlush in clawdbot.json
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
            _enable_clawdbot_hook(stack_id)
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
        _enable_clawdbot_hook(stack_id)
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
                    print("   Run with --enable to auto-configure, or add manually")
            except Exception as e:
                print(f"⚠️  Could not read config: {e}")
        else:
            print(f"\n⚠️  Clawdbot config not found at {config_path}")
            print("   Run with --enable to create config with hook enabled")

        print("\nNext steps:")
        print("  1. Enable hook: kernle setup clawdbot --enable")
        print("  2. Restart Clawdbot gateway: clawdbot gateway restart")
        print(f"  3. Memory will load automatically for agent '{stack_id}'")


def _enable_clawdbot_hook(stack_id: str) -> bool:
    """Enable kernle-load hook and configure memoryFlush in clawdbot.json.

    This configures both:
    1. Session start hook (loads KERNLE.md)
    2. Pre-compaction memory flush (saves checkpoint)

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

        # Build the config updates we want to apply
        memory_flush_prompt = _get_memory_flush_prompt(stack_id)
        kernle_config = {
            "hooks": {
                "internal": {
                    "enabled": True,
                    "entries": {"kernle-load": {"enabled": True}},
                }
            },
            "agents": {
                "defaults": {
                    "compaction": {"memoryFlush": {"enabled": True, "prompt": memory_flush_prompt}}
                }
            },
        }

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

        if hook_enabled and flush_configured:
            print("✓ Kernle already fully configured")
            print("  - Session start hook: enabled")
            print("  - Pre-compaction flush: enabled")
            return True

        # Merge our config into existing
        merged = _deep_merge(config, kernle_config)

        with open(config_path, "w") as f:
            json.dump(merged, f, indent=2)

        print("✓ Updated clawdbot.json with Kernle configuration")

        if not hook_enabled:
            print("  - Enabled session start hook")
        if not flush_configured:
            print("  - Configured pre-compaction memory flush")

        print()
        print("=" * 50)
        print("Kernle Setup Complete")
        print("=" * 50)
        print()
        print("Configured for seamless context transitions:")
        print("  1. Session start: Memory auto-loads into KERNLE.md")
        print("  2. Pre-compaction: Agent saves checkpoint before compaction")
        print()
        print("⚠️  Restart Clawdbot gateway for changes to take effect:")
        print("   clawdbot gateway restart")
        print()
        print(f"Memory will persist across sessions for agent '{stack_id}'")

        return True

    except Exception as e:
        print(f"❌ Failed to enable hook in config: {e}")
        print("   You may need to manually edit ~/.clawdbot/clawdbot.json")
        return False


def _build_claude_code_hooks(stack_id: str | None = None) -> dict:
    """Build the hooks configuration dict for Claude Code settings.json.

    Args:
        stack_id: If provided, bake -s {stack_id} into hook commands.

    Returns:
        Dict matching Claude Code settings.json hooks schema.
    """
    stack_flag = f" -s {stack_id}" if stack_id else ""

    return {
        "hooks": {
            "SessionStart": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"kernle{stack_flag} hook session-start",
                            "timeout": 10,
                        }
                    ]
                }
            ],
            "PreToolUse": [
                {
                    "matcher": "Write|Edit|NotebookEdit",
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"kernle{stack_flag} hook pre-tool-use",
                            "timeout": 10,
                        }
                    ],
                }
            ],
            "PreCompact": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"kernle{stack_flag} hook pre-compact",
                            "timeout": 10,
                        }
                    ]
                }
            ],
            "SessionEnd": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"kernle{stack_flag} hook session-end",
                            "timeout": 10,
                        }
                    ]
                }
            ],
        }
    }


def setup_claude_code(stack_id: str, force: bool = False, global_install: bool = False) -> None:
    """Install Claude Code hooks for automatic memory lifecycle management.

    Writes hooks to .claude/settings.json that call `kernle hook <event>`.
    Four hooks: SessionStart, PreToolUse, PreCompact, SessionEnd.
    """
    # Determine target
    if global_install:
        target = Path.home() / ".claude" / "settings.json"
        location = "user settings (global)"
    else:
        target = Path.cwd() / ".claude" / "settings.json"
        location = "project settings"

    # Build hooks config
    hooks_config = _build_claude_code_hooks(stack_id)

    # Check if already configured
    if target.exists() and not force:
        try:
            with open(target) as f:
                existing = json.load(f)
            existing_hooks = existing.get("hooks", {})
            has_kernle = any(
                "kernle" in str(existing_hooks.get(event, [])).lower()
                for event in ("SessionStart", "PreToolUse", "PreCompact", "SessionEnd")
            )
            if has_kernle:
                print(f"Kernle hooks already configured in {target}")
                print("   Use --force to overwrite")
                return
        except (json.JSONDecodeError, OSError):
            pass  # Will write fresh

    # Create target directory
    target.parent.mkdir(parents=True, exist_ok=True)

    # Merge with existing or write new
    if target.exists():
        try:
            with open(target) as f:
                existing = json.load(f)
            existing = _deep_merge(existing, hooks_config)
            with open(target, "w") as f:
                json.dump(existing, f, indent=2)
            print(f"Merged Kernle hooks into existing {location}")
        except Exception as e:
            print(f"Could not merge with existing config: {e}")
            print("   Writing new config instead")
            with open(target, "w") as f:
                json.dump(hooks_config, f, indent=2)
            print(f"Created {location}")
    else:
        with open(target, "w") as f:
            json.dump(hooks_config, f, indent=2)
        print(f"Created {location}")

    print(f"  Location: {target}")
    print()
    print("Hooks configured:")
    print("  SessionStart  - Auto-loads memory into session context")
    print("  PreToolUse    - Intercepts writes to memory/ and MEMORY.md")
    print("  PreCompact    - Saves checkpoint before context compaction")
    print("  SessionEnd    - Saves final checkpoint on session termination")
    print()
    loc_desc = "any directory" if global_install else "this directory"
    print(f"Start Claude Code in {loc_desc}: claude")
    print(f"Memory loads automatically for stack '{stack_id}'")
    print()
    print("Verify with: kernle doctor --full")


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
    stack_id = k.stack_id

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
        setup_clawdbot(stack_id, force, enable)
    elif platform in ("claude-code", "cowork"):
        setup_claude_code(stack_id, force, global_install)
    else:
        print(f"❌ Unknown platform: {platform}")
        print("Available: clawdbot, claude-code, cowork")
