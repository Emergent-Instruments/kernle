#!/usr/bin/env python3
"""PreToolUse hook: Intercept native memory file writes."""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _bridge import KernleBridge
from _config import get_config, resolve_stack_id

MEMORY_PATTERNS = [
    re.compile(r"^memory/"),
    re.compile(r"/memory/"),
    re.compile(r"MEMORY\.md$"),
]


def is_memory_path(file_path: str) -> bool:
    return any(p.search(file_path) for p in MEMORY_PATTERNS)


def main():
    try:
        hook_input = json.loads(sys.stdin.read())

        tool_input = hook_input.get("tool_input", {})

        # Extract file path from tool input
        file_path = tool_input.get("file_path", "") or tool_input.get("path", "")

        if not file_path or not is_memory_path(file_path):
            sys.exit(0)  # Allow — not a memory file

        # Capture content into Kernle as raw entry
        cwd = hook_input.get("cwd")
        config = get_config()
        stack_id = resolve_stack_id(cwd)
        bridge = KernleBridge(
            kernle_bin=config["kernle_bin"],
            timeout=config["timeout"],
        )

        content = tool_input.get("content", "") or tool_input.get("new_string", "")
        if content:
            truncated = content[:2000] + "\n[truncated]" if len(content) > 2000 else content
            bridge.raw(stack_id, f"[memory-capture] {file_path}\n\n{truncated}")

        # Block the write
        output = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    "Memory writes are handled by Kernle. Use `kernle raw`, "
                    "`kernle episode`, or `kernle note` to record memories. "
                    "Loading and checkpointing are automatic."
                ),
            }
        }
        json.dump(output, sys.stdout)
    except Exception:
        pass

    sys.exit(0)


if __name__ == "__main__":
    main()
