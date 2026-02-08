#!/usr/bin/env python3
"""SessionStart hook: Load Kernle memory into Claude Code session context."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _bridge import KernleBridge
from _config import get_config, resolve_stack_id


def main():
    try:
        hook_input = json.loads(sys.stdin.read())

        cwd = hook_input.get("cwd")
        config = get_config()
        stack_id = resolve_stack_id(cwd)

        bridge = KernleBridge(
            kernle_bin=config["kernle_bin"],
            timeout=config["timeout"],
        )

        memory = bridge.load(stack_id, config["token_budget"])

        if memory:
            output = {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": memory,
                }
            }
            json.dump(output, sys.stdout)
    except Exception:
        pass

    sys.exit(0)


if __name__ == "__main__":
    main()
