#!/usr/bin/env python3
"""SessionEnd hook: Auto-checkpoint on session termination."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _bridge import KernleBridge
from _config import get_config, resolve_stack_id
from _transcript import read_last_messages


def main():
    try:
        hook_input = json.loads(sys.stdin.read())

        cwd = hook_input.get("cwd")
        transcript_path = hook_input.get("transcript_path")
        config = get_config()
        stack_id = resolve_stack_id(cwd)

        bridge = KernleBridge(
            kernle_bin=config["kernle_bin"],
            timeout=config["timeout"],
        )

        task, context = read_last_messages(transcript_path)

        # Save both (ignore failures)
        bridge.checkpoint(stack_id, task, context)
        bridge.raw(stack_id, f"Session ended. Task: {task}")
    except Exception:
        pass

    sys.exit(0)


if __name__ == "__main__":
    main()
