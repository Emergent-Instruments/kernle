"""Transcript JSONL reader for Claude Code hooks.

Reads the conversation transcript file to extract last user/assistant messages
for checkpoint saves.
"""

import json
from pathlib import Path


def read_last_messages(
    transcript_path: str | None,
    max_user: int = 200,
    max_assistant: int = 500,
) -> tuple[str, str | None]:
    """Read last user and assistant messages from a transcript JSONL file.

    Returns (task, context) tuple suitable for checkpoint save.
    Falls back to generic text if transcript can't be read.
    """
    if not transcript_path:
        return ("Session ended", None)

    try:
        lines = Path(transcript_path).read_text().strip().split("\n")
        last_user = None
        last_assistant = None

        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            role = entry.get("role", "")
            content = _extract_text(entry)
            if not content:
                continue

            if role == "user" and last_user is None:
                last_user = _truncate(content, max_user)
            elif role == "assistant" and last_assistant is None:
                last_assistant = _truncate(content, max_assistant)

            if last_user is not None and last_assistant is not None:
                break

        return (last_user or "Session ended", last_assistant)
    except Exception:
        return ("Session ended", None)


def _extract_text(entry: dict) -> str:
    """Extract text content from a transcript entry.

    Content can be a string or a list of content blocks.
    """
    content = entry.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return " ".join(texts)
    return str(content)


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
