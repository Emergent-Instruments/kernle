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
   kernle -a {agent_id} load
   ```

2. **Check health:**
   ```bash
   kernle -a {agent_id} anxiety
   ```

3. **If anxiety > 50:** Address before proceeding
   - Run `kernle consolidate` for episode debt
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

At session start: `kernle -a {agent_id} load && kernle -a {agent_id} anxiety -b`
If WARN/CRITICAL, run `kernle consolidate` before proceeding.
"""

PER_MESSAGE_SECTION = """## Memory Health (Every Message)

Before processing any request:
1. `kernle -a {agent_id} anxiety -b`
2. If 🟡 WARN or 🔴 CRITICAL: pause and consolidate first

This takes <1 second and prevents memory debt accumulation.
"""

COMBINED_SECTION = """## Memory (Kernle)

You have persistent memory through Kernle.

### Every Session (Boot Sequence)

Before any task:
1. **Load memory:** `kernle -a {agent_id} load`
2. **Check health:** `kernle -a {agent_id} anxiety`
3. If anxiety > 50: address before proceeding

### Every Message (Health Check)

Quick pulse: `kernle -a {agent_id} anxiety -b`
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
        r"kernle\s+(-a\s+\w+\s+)?load",
        r"kernle\s+(-a\s+\w+\s+)?anxiety",
    ]

    for pattern in patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return True

    return False


def generate_section(
    agent_id: str, style: str = "standard", include_per_message: bool = True
) -> str:
    """Generate the appropriate Kernle section based on style."""
    if style == "minimal":
        section = MINIMAL_SECTION.format(agent_id=agent_id)
        if include_per_message:
            section += "\n" + PER_MESSAGE_SECTION.format(agent_id=agent_id)
        return section
    elif style == "combined":
        return COMBINED_SECTION.format(agent_id=agent_id)
    else:  # standard
        section = STANDARD_SECTION.format(agent_id=agent_id)
        if include_per_message:
            section += "\n" + PER_MESSAGE_SECTION.format(agent_id=agent_id)
        return section


def cmd_init(args, k: "Kernle"):
    """Generate CLAUDE.md section for Kernle health checks.

    Creates or appends Kernle memory instructions to your instruction file
    (CLAUDE.md, AGENTS.md, etc.) so any SI can adopt health checks with zero friction.
    """
    agent_id = k.agent_id
    style = getattr(args, "style", "standard") or "standard"
    include_per_message = not getattr(args, "no_per_message", False)
    output_file = getattr(args, "output", None)
    force = getattr(args, "force", False)
    print_only = getattr(args, "print", False)

    # Generate the section
    section = generate_section(agent_id, style, include_per_message)

    # Print-only mode
    if print_only:
        print("# Kernle Instructions for CLAUDE.md")
        print("# Copy this section to your instruction file:")
        print()
        print(section)
        return

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

    # Check if file exists and already has Kernle section
    if target_file.exists():
        content = target_file.read_text()

        if has_kernle_section(content) and not force:
            print(f"\n⚠️  {target_file} already contains Kernle instructions.")
            print("   Use --force to overwrite/append anyway.")
            print("   Use --print to just display the section.")
            return

        # Append mode
        if not getattr(args, "non_interactive", False):
            print(f"\nWill append to existing {target_file}")
            try:
                confirm = input("Proceed? [Y/n]: ").strip().lower()
                if confirm and confirm != "y" and confirm != "yes":
                    print("Aborted.")
                    return
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.")
                return

        # Add separator and append
        new_content = content.rstrip() + "\n\n" + section
        target_file.write_text(new_content)
        print(f"\n✓ Appended Kernle instructions to {target_file}")

    else:
        # Create new file
        header = "# Instructions\n\n"
        target_file.write_text(header + section)
        print(f"\n✓ Created {target_file} with Kernle instructions")

    # Show quick verification command
    print("\nVerify with: kernle doctor")
    print(f"Test health check: kernle -a {agent_id} anxiety -b")
