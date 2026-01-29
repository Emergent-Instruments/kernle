"""Doctor command for Kernle CLI - validates boot sequence compliance."""

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from kernle import Kernle


class ComplianceCheck:
    """Result of a single compliance check."""

    def __init__(self, name: str, passed: bool, message: str, fix: Optional[str] = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.fix = fix

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "fix": self.fix,
        }


def find_instruction_file() -> Optional[Tuple[Path, str]]:
    """Find instruction file and return (path, type)."""
    candidates = [
        (Path("CLAUDE.md"), "claude"),
        (Path("AGENTS.md"), "agents"),
        (Path(".cursorrules"), "cursor"),
        (Path(".clinerules"), "cline"),
        (Path.home() / ".claude" / "CLAUDE.md", "claude-global"),
    ]

    for path, file_type in candidates:
        if path.exists():
            return path, file_type

    return None


def check_kernle_load(content: str, agent_id: str) -> ComplianceCheck:
    """Check if load instruction is present."""
    patterns = [
        rf"kernle\s+(-a\s+{re.escape(agent_id)}\s+)?load",
        r"kernle\s+-a\s+\w+\s+load",
        r"kernle_load",  # MCP tool name
    ]

    for pattern in patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return ComplianceCheck(
                name="load_instruction", passed=True, message="✓ Load instruction found"
            )

    return ComplianceCheck(
        name="load_instruction",
        passed=False,
        message="✗ Missing `kernle load` instruction at session start",
        fix=f"Add: `kernle -a {agent_id} load` to session boot sequence",
    )


def check_kernle_anxiety(content: str, agent_id: str) -> ComplianceCheck:
    """Check if anxiety/health check instruction is present."""
    patterns = [
        rf"kernle\s+(-a\s+{re.escape(agent_id)}\s+)?anxiety",
        r"kernle\s+-a\s+\w+\s+anxiety",
        r"kernle_anxiety",  # MCP tool name
        r"health\s*check",
    ]

    for pattern in patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return ComplianceCheck(
                name="anxiety_instruction", passed=True, message="✓ Health check instruction found"
            )

    return ComplianceCheck(
        name="anxiety_instruction",
        passed=False,
        message="✗ Missing `kernle anxiety` health check instruction",
        fix=f"Add: `kernle -a {agent_id} anxiety` after load",
    )


def check_per_message_health(content: str, agent_id: str) -> ComplianceCheck:
    """Check if per-message health check instruction is present."""
    patterns = [
        r"every\s+message",
        r"per.?message",
        r"before\s+(processing|any)\s+request",
        r"health\s+check.+message",
        r"anxiety\s+-b",
    ]

    for pattern in patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return ComplianceCheck(
                name="per_message_health",
                passed=True,
                message="✓ Per-message health check instruction found",
            )

    return ComplianceCheck(
        name="per_message_health",
        passed=False,
        message="⚠ No per-message health check (recommended)",
        fix=f"Add section: 'Every Message: `kernle -a {agent_id} anxiety -b`'",
    )


def check_checkpoint_instruction(content: str, agent_id: str) -> ComplianceCheck:
    """Check if checkpoint instruction is present."""
    patterns = [
        rf"kernle\s+(-a\s+{re.escape(agent_id)}\s+)?checkpoint",
        r"kernle\s+-a\s+\w+\s+checkpoint",
        r"kernle_checkpoint",  # MCP tool name
    ]

    for pattern in patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return ComplianceCheck(
                name="checkpoint_instruction", passed=True, message="✓ Checkpoint instruction found"
            )

    return ComplianceCheck(
        name="checkpoint_instruction",
        passed=False,
        message="⚠ No checkpoint instruction (recommended for session end)",
        fix=f'Add: `kernle -a {agent_id} checkpoint save "state"` before session ends',
    )


def check_memory_section(content: str) -> ComplianceCheck:
    """Check if there's a dedicated memory section."""
    patterns = [
        r"##\s*Memory",
        r"##\s*Kernle",
        r"##\s*Every\s+Session",
        r"##\s*Boot\s*Sequence",
    ]

    for pattern in patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return ComplianceCheck(
                name="memory_section", passed=True, message="✓ Dedicated memory section found"
            )

    return ComplianceCheck(
        name="memory_section",
        passed=False,
        message="⚠ No dedicated Memory/Kernle section (recommended for clarity)",
        fix="Add: `## Memory (Kernle)` section header",
    )


def run_all_checks(content: str, agent_id: str) -> List[ComplianceCheck]:
    """Run all compliance checks."""
    checks = [
        check_memory_section(content),
        check_kernle_load(content, agent_id),
        check_kernle_anxiety(content, agent_id),
        check_per_message_health(content, agent_id),
        check_checkpoint_instruction(content, agent_id),
    ]
    return checks


def cmd_doctor(args, k: "Kernle"):
    """Validate Kernle boot sequence compliance.

    Checks if your instruction file (CLAUDE.md, AGENTS.md, etc.) contains
    the necessary Kernle health check instructions for proper memory management.
    """
    agent_id = k.agent_id
    output_json = getattr(args, "json", False)
    fix = getattr(args, "fix", False)

    # Find instruction file
    result = find_instruction_file()

    if result is None:
        if output_json:
            print(
                json.dumps(
                    {
                        "status": "no_file",
                        "message": "No instruction file found",
                        "checks": [],
                        "fix": "Run `kernle init` to create one",
                    },
                    indent=2,
                )
            )
        else:
            print("❌ No instruction file found")
            print()
            print("Looked for:")
            print("  - CLAUDE.md (current directory)")
            print("  - AGENTS.md (current directory)")
            print("  - .cursorrules (current directory)")
            print("  - .clinerules (current directory)")
            print("  - ~/.claude/CLAUDE.md (global)")
            print()
            print("Fix: Run `kernle init` to create one")
        return

    file_path, file_type = result
    content = file_path.read_text()

    # Run all checks
    checks = run_all_checks(content, agent_id)

    # Calculate summary
    required_checks = ["load_instruction", "anxiety_instruction"]
    recommended_checks = ["per_message_health", "checkpoint_instruction", "memory_section"]

    required_passed = sum(1 for c in checks if c.name in required_checks and c.passed)
    required_total = len(required_checks)
    recommended_passed = sum(1 for c in checks if c.name in recommended_checks and c.passed)
    recommended_total = len(recommended_checks)

    all_required_pass = required_passed == required_total

    # Determine overall status
    if all_required_pass and recommended_passed == recommended_total:
        status = "excellent"
        status_emoji = "🟢"
        status_message = "Excellent! Full compliance"
    elif all_required_pass:
        status = "good"
        status_emoji = "🟡"
        status_message = "Good - required checks pass, some recommendations missing"
    else:
        status = "needs_work"
        status_emoji = "🔴"
        status_message = "Needs work - missing required instructions"

    if output_json:
        output = {
            "status": status,
            "file": str(file_path),
            "file_type": file_type,
            "agent_id": agent_id,
            "required_passed": required_passed,
            "required_total": required_total,
            "recommended_passed": recommended_passed,
            "recommended_total": recommended_total,
            "checks": [c.to_dict() for c in checks],
        }
        print(json.dumps(output, indent=2))
        return

    # Print header
    print("\nKernle Doctor - Boot Sequence Validation")
    print("=" * 50)
    print(f"File: {file_path}")
    print(f"Agent: {agent_id}")
    print()

    # Print status
    print(f"{status_emoji} Status: {status_message}")
    print(f"   Required: {required_passed}/{required_total}")
    print(f"   Recommended: {recommended_passed}/{recommended_total}")
    print()

    # Print check details
    print("Checks:")
    for check in checks:
        is_required = check.name in required_checks
        prefix = "[required]" if is_required else "[recommended]"
        print(f"  {prefix:14} {check.message}")

    # Print fixes if needed
    failed_checks = [c for c in checks if not c.passed and c.fix]
    if failed_checks:
        print()
        print("Fixes:")
        for check in failed_checks:
            is_required = check.name in required_checks
            priority = "REQUIRED" if is_required else "recommended"
            print(f"  [{priority}] {check.fix}")

        if fix:
            # Auto-fix mode
            print()
            print("Auto-fixing...")
            try:
                # Import init command to generate section
                from kernle.cli.commands.init import generate_section, has_kernle_section

                if not has_kernle_section(content):
                    section = generate_section(agent_id, style="combined", include_per_message=True)
                    new_content = content.rstrip() + "\n\n" + section
                    file_path.write_text(new_content)
                    print(f"✓ Added Kernle instructions to {file_path}")
                else:
                    print("⚠ File already has some Kernle instructions.")
                    print("  Manual review recommended.")
            except Exception as e:
                print(f"✗ Auto-fix failed: {e}")
        else:
            print()
            print("Run `kernle init` to add missing instructions")
            print("Or `kernle doctor --fix` to auto-fix")
    else:
        print()
        print("✓ All checks passed!")
        print()
        print(f"Test your setup: kernle -a {agent_id} load && kernle -a {agent_id} anxiety -b")
