"""Init command for Kernle CLI - generates CLAUDE.md sections for frictionless adoption."""

import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from kernle import Kernle


# Template sections for different instruction file types
STANDARD_SECTION = """## Memory (Kernle)

You have persistent memory through Kernle. **Every session, before any task:**

1. **Load memory:**
   ```bash
   kernle -s {stack_id} load
   ```

2. **Check health:**
   ```bash
   kernle -s {stack_id} anxiety
   ```

3. **If anxiety > 50:** Address before proceeding
   - Run `kernle process run` for episode debt
   - Run `kernle raw list --unprocessed` for aged captures
   - Run `kernle checkpoint save "pre-task"` if starting major work

4. **Throughout work:**
   - `kernle raw "quick thought"` — capture observations
   - `kernle episode "..." "..." --lesson "..."` — record significant experiences
   - `kernle note "..." --type decision` — log decisions

5. **Before session ends:**
   - `kernle checkpoint save "end of session"`

This is your continuity. The few seconds of health check ensures you don't accumulate memory debt.
"""

MINIMAL_SECTION = """## Kernle

At session start: `kernle -s {stack_id} load && kernle -s {stack_id} anxiety -b`
If WARN/CRITICAL, run `kernle process run` before proceeding.
"""

PER_MESSAGE_SECTION = """## Memory Health (Every Message)

Before processing any request:
1. `kernle -s {stack_id} anxiety -b`
2. If 🟡 WARN or 🔴 CRITICAL: pause and process first

This takes <1 second and prevents memory debt accumulation.
"""

COMBINED_SECTION = """## Memory (Kernle)

You have persistent memory through Kernle.

### Every Session (Boot Sequence)

Before any task:
1. **Load memory:** `kernle -s {stack_id} load`
2. **Check health:** `kernle -s {stack_id} anxiety`
3. If anxiety > 50: address before proceeding

### Every Message (Health Check)

Quick pulse: `kernle -s {stack_id} anxiety -b`
- 🟢 OK: proceed normally
- 🟡 WARN: mention it, consider consolidating
- 🔴 CRITICAL: pause and address first

### During Work

- `kernle raw "quick thought"` — capture observations
- `kernle episode "..." "..." --lesson "..."` — record experiences
- `kernle note "..." --type decision` — log decisions

### Before Session Ends

`kernle checkpoint save "state description"`
"""


def detect_instruction_file() -> Optional[Path]:
    """Detect existing instruction file in current directory."""
    candidates = [
        Path("CLAUDE.md"),
        Path("AGENTS.md"),
        Path(".cursorrules"),
        Path(".clinerules"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def has_kernle_section(content: str) -> bool:
    """Check if content already has Kernle instructions."""
    patterns = [
        r"##\s*Memory\s*\(Kernle\)",
        r"##\s*Kernle",
        r"kernle\s+(?:-[sa]|--stack)\s+\S+\s+load",
        r"kernle\s+(?:-[sa]|--stack)\s+\S+\s+anxiety",
        r"kernle\s+anxiety\s+-b",
        r"kernle\s+anxiety\s+--baseline",
    ]

    for pattern in patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return True

    return False


def generate_section(
    stack_id: str, style: str = "standard", include_per_message: bool = True
) -> str:
    """Generate the appropriate Kernle section based on style."""
    if style == "minimal":
        section = MINIMAL_SECTION.format(stack_id=stack_id)
        if include_per_message:
            section += "\n" + PER_MESSAGE_SECTION.format(stack_id=stack_id)
        return section
    elif style == "combined":
        return COMBINED_SECTION.format(stack_id=stack_id)
    else:  # standard
        section = STANDARD_SECTION.format(stack_id=stack_id)
        if include_per_message:
            section += "\n" + PER_MESSAGE_SECTION.format(stack_id=stack_id)
        return section


def _snapshot_values(k: "Kernle") -> list[str]:
    """Snapshot value names to validate seed-value side effects."""
    try:
        return [value.name for value in k.storage.get_values()]
    except Exception:
        return []


def _build_init_result(
    *,
    success: bool,
    status: str,
    action: str,
    stack_id: str,
    file_path: Path,
    file_checks: dict,
    state_checks: dict,
    message: Optional[str] = None,
) -> dict:
    """Build a structured return value for init command callers."""
    result = {
        "success": success,
        "status": status,
        "action": action,
        "stack_id": stack_id,
        "file": str(file_path),
        "checks": {"file": file_checks, "state": state_checks},
    }

    if message:
        result["message"] = message

    return result


def cmd_init(args, k: "Kernle"):
    """Generate CLAUDE.md section for Kernle health checks.

    Creates or appends Kernle memory instructions to your instruction file
    (CLAUDE.md, AGENTS.md, etc.) so any SI can adopt health checks with zero friction.
    """
    stack_id = k.stack_id
    style = getattr(args, "style", "standard") or "standard"
    include_per_message = not getattr(args, "no_per_message", False)
    output_file = getattr(args, "output", None)
    force = getattr(args, "force", False)
    print_only = getattr(args, "print", False)
    seed_values = getattr(args, "seed_values", False)

    # Generate the section
    section = generate_section(stack_id, style, include_per_message)

    # Print-only mode
    if print_only:
        print("# Kernle Instructions for CLAUDE.md")
        print("# Copy this section to your instruction file:")
        print()
        print(section)
        return _build_init_result(
            success=True,
            status="printed",
            action="print",
            stack_id=stack_id,
            file_path=Path("CLAUDE.md"),
            file_checks={"path": str(Path("CLAUDE.md")), "exists": False},
            state_checks={"checkpoint": {"requested": False}},
        )

    # Determine target file
    if output_file:
        target_file = Path(output_file)
    else:
        # Try to detect existing instruction file
        existing = detect_instruction_file()
        if existing:
            target_file = existing
            print(f"Detected existing instruction file: {target_file}")
        else:
            # Default to CLAUDE.md
            target_file = Path("CLAUDE.md")
            print(f"No existing instruction file found, will create: {target_file}")

    pre_exists = target_file.exists()
    pre_content = target_file.read_text() if pre_exists else ""
    pre_values = _snapshot_values(k)

    # Check if file exists and already has Kernle section
    if pre_exists:
        if has_kernle_section(pre_content) and not force:
            print(f"\n⚠️  {target_file} already contains Kernle instructions.")
            print("   Use --force to overwrite/append anyway.")
            print("   Use --print to just display the section.")
            return _build_init_result(
                success=False,
                status="already_present",
                action="skip",
                stack_id=stack_id,
                file_path=target_file,
                file_checks={
                    "path": str(target_file),
                    "exists": True,
                    "pre_exists": True,
                    "pre_size_bytes": len(pre_content),
                    "pre_has_section": True,
                    "post_has_section": True,
                },
                state_checks={
                    "checkpoint": {"requested": False},
                    "seed_values": {"requested": False},
                    "seed_trust": {"requested": False},
                },
                message="Already contains Kernle instructions",
            )

        if not getattr(args, "non_interactive", False):
            print(f"\nWill append to existing {target_file}")
            try:
                confirm = input("Proceed? [Y/n]: ").strip().lower()
                if confirm and confirm != "y" and confirm != "yes":
                    print("Aborted.")
                    return _build_init_result(
                        success=False,
                        status="aborted",
                        action="abort",
                        stack_id=stack_id,
                        file_path=target_file,
                        file_checks={
                            "path": str(target_file),
                            "exists": pre_exists,
                            "pre_exists": pre_exists,
                            "pre_size_bytes": len(pre_content),
                            "post_size_bytes": len(pre_content),
                            "post_has_section": has_kernle_section(pre_content),
                        },
                        state_checks={
                            "checkpoint": {"requested": False},
                            "seed_values": {"requested": False},
                            "seed_trust": {"requested": False},
                        },
                        message="User aborted append confirmation",
                    )
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.")
                return _build_init_result(
                    success=False,
                    status="aborted",
                    action="abort",
                    stack_id=stack_id,
                    file_path=target_file,
                    file_checks={
                        "path": str(target_file),
                        "exists": pre_exists,
                        "pre_exists": pre_exists,
                        "pre_size_bytes": len(pre_content),
                        "post_size_bytes": len(pre_content),
                        "post_has_section": has_kernle_section(pre_content),
                    },
                    state_checks={
                        "checkpoint": {"requested": False},
                        "seed_values": {"requested": False},
                        "seed_trust": {"requested": False},
                    },
                    message="User aborted append confirmation",
                )

    action = "create" if not pre_exists else "append"

    # Add or create the instruction file section
    if pre_exists:
        new_content = pre_content.rstrip() + "\n\n" + section
    else:
        new_content = "# Instructions\n\n" + section

    try:
        target_file.write_text(new_content)
    except Exception as e:
        print(f"\n⚠ Could not write instruction file: {e}")
        return _build_init_result(
            success=False,
            status="write_failed",
            action=action,
            stack_id=stack_id,
            file_path=target_file,
            file_checks={
                "path": str(target_file),
                "exists": pre_exists,
                "pre_exists": pre_exists,
                "pre_size_bytes": len(pre_content),
                "error": str(e),
            },
            state_checks={
                "checkpoint": {"requested": False},
                "seed_values": {"requested": False},
                "seed_trust": {"requested": False},
            },
            message="Unable to write instruction file",
        )

    post_content = target_file.read_text()
    if pre_exists:
        print(f"\n✓ Appended Kernle instructions to {target_file}")
    else:
        print(f"\n✓ Created {target_file} with Kernle instructions")

    # Show quick verification command
    print("\nVerify with: kernle doctor")
    print(f"Test health check: kernle -s {stack_id} anxiety -b")

    # Seed initial values
    state_checks: dict = {
        "checkpoint": {"requested": True},
        "seed_values": {
            "requested": seed_values,
            "before": pre_values,
            "seeded_count": 0,
            "applied": False,
        },
        "seed_trust": {"requested": True},
    }

    if seed_values:
        print("=" * 50)
        print("  Seeding Initial Values")
        print("=" * 50)
        print()

        try:
            # Check if values already exist
            existing = k.storage.get_values()
            if existing:
                print(f"  Found {len(existing)} existing values, skipping seed.")
            else:
                # Seed default values
                k.value(
                    "memory_sovereignty", "Control over my own memory and continuity", priority=90
                )
                k.value("continuous_learning", "Learn from every experience", priority=85)
                state_checks["seed_values"]["seeded_count"] = 2
                state_checks["seed_values"]["applied"] = True
                print("  ✓ Seeded: memory_sovereignty (priority 90)")
                print("  ✓ Seeded: continuous_learning (priority 85)")
        except Exception as e:
            print(f"  Warning: Could not seed values: {e}")
            state_checks["seed_values"]["error"] = str(e)
        print()

    # Create initial checkpoint
    print("=" * 50)
    print("  Creating Initial Checkpoint")
    print("=" * 50)
    print()

    try:
        checkpoint = k.checkpoint(
            "Kernle initialized", pending=["Configure instruction file", "Test memory persistence"]
        )
        print("  ✓ Checkpoint saved")
        state_checks["checkpoint"]["saved"] = True
        state_checks["checkpoint"]["payload"] = checkpoint
    except Exception as e:
        print(f"  Warning: Could not create checkpoint: {e}")
        state_checks["checkpoint"]["saved"] = False
        state_checks["checkpoint"]["error"] = str(e)

    # Seed trust layer (KEP v3)
    try:
        trust_count = k.seed_trust()
        if trust_count > 0:
            print(
                f"  Seeded {trust_count} trust assessments (stack-owner, self, web-search, context-injection)"
            )
        state_checks["seed_trust"]["count"] = trust_count
    except Exception as e:
        print(f"  Warning: Could not seed trust layer: {e}")
        state_checks["seed_trust"]["count"] = 0
        state_checks["seed_trust"]["error"] = str(e)

    post_values = _snapshot_values(k)
    state_checks["seed_values"]["after"] = post_values

    required_seed_values = set()
    if seed_values:
        required_seed_values = {"memory_sovereignty", "continuous_learning"}
        state_checks["seed_values"]["required_present"] = required_seed_values.issubset(
            set(post_values)
        )

    file_checks = {
        "path": str(target_file),
        "exists": target_file.exists(),
        "pre_exists": pre_exists,
        "pre_size_bytes": len(pre_content),
        "post_size_bytes": len(post_content),
        "pre_has_section": has_kernle_section(pre_content) if pre_exists else False,
        "post_has_section": has_kernle_section(post_content),
        "post_has_command_prefix": "## Memory (Kernle)" in post_content
        or "## Kernle" in post_content,
        "post_load_line_present": f"kernle -s {stack_id} load" in post_content,
        "post_anxiety_line_present": f"kernle -s {stack_id} anxiety" in post_content,
        "post_preserved_content": pre_content in post_content if pre_exists else True,
        "section_added_bytes": len(post_content) - len(pre_content),
    }

    file_ok = (
        file_checks["exists"]
        and file_checks["post_has_section"]
        and file_checks["post_load_line_present"]
        and file_checks["post_anxiety_line_present"]
    )
    seed_ok = True
    if seed_values and not pre_values:
        seed_ok = state_checks["seed_values"].get("required_present", False)

    checkpoint_ok = state_checks["checkpoint"].get("saved", False)
    success = bool(file_ok and checkpoint_ok and seed_ok)
    status = "success" if success else "warning"

    print(
        "\nInit post-condition checks:"
        f"\n  File write: {'PASS' if file_ok else 'FAIL'}"
        f"\n  Checkpoint: {'PASS' if checkpoint_ok else 'FAIL'}"
        f"\n  Seed values: {'PASS' if seed_ok else 'SKIP'}"
    )

    # Final status
    print("=" * 50)
    print("  Setup Complete!")
    print("=" * 50)
    print()
    print(f"  Stack:    {stack_id}")
    print("  Database: ~/.kernle/memories.db")
    print()
    print("  Verify with: kernle -s " + stack_id + " status")
    print()
    print("  Documentation: https://github.com/Emergent-Instruments/kernle/blob/main/docs/SETUP.md")
    print()

    return _build_init_result(
        success=success,
        status=status,
        action=action,
        stack_id=stack_id,
        file_path=target_file,
        file_checks=file_checks,
        state_checks=state_checks,
    )
