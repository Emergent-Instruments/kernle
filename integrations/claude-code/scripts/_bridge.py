"""Kernle CLI wrapper for Claude Code hooks.

Port of integrations/openclaw/src/services/kernle-bridge.ts.
Uses subprocess to call the kernle CLI with timeout and graceful error handling.
"""

import subprocess

DEFAULT_TIMEOUT = 5
MAX_OUTPUT = 1024 * 1024  # 1MB


class KernleBridge:
    """Shell exec wrapper for the kernle CLI."""

    def __init__(self, kernle_bin: str = "kernle", timeout: int = DEFAULT_TIMEOUT):
        self.bin = kernle_bin
        self.timeout = timeout

    def load(self, stack_id: str | None, budget: int | None = None) -> str | None:
        """Load memory for a stack. Returns formatted output or None on failure."""
        args = self._stack_arg(stack_id) + ["load"]
        if budget:
            args += ["--budget", str(budget)]
        return self._exec(args)

    def checkpoint(self, stack_id: str | None, summary: str, context: str | None = None) -> bool:
        """Save a checkpoint. Returns True on success."""
        args = self._stack_arg(stack_id) + ["checkpoint", "save", summary]
        if context:
            args += ["--context", context]
        return self._exec(args) is not None

    def raw(self, stack_id: str | None, content: str) -> bool:
        """Save a raw entry. Returns True on success."""
        args = self._stack_arg(stack_id) + ["raw", content]
        return self._exec(args) is not None

    def _stack_arg(self, stack_id: str | None) -> list[str]:
        if stack_id:
            return ["-s", stack_id]
        return []

    def _exec(self, args: list[str]) -> str | None:
        try:
            cmd = [self.bin] + args
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return None
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return None
