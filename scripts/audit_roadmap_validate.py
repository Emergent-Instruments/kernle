#!/usr/bin/env python3
"""Validate logic-audit improvement roadmap structure and hardening conventions."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml


def fail(message: str, errors: List[str]) -> None:
    errors.append(f"FAIL: {message}")


def validate_phase_structure(data: Dict[str, Any], errors: List[str]) -> None:
    phases = data.get("phases")
    if not isinstance(phases, list):
        fail("top-level `phases` must be a list", errors)
        return

    for i, phase in enumerate(phases, 1):
        if not isinstance(phase, dict):
            fail(f"phases[{i}] is not a mapping", errors)
            continue

        for field in ("phase", "priority", "items"):
            if field not in phase:
                fail(f"phases[{i}] missing required field `{field}`", errors)

        if "items" in phase and not isinstance(phase.get("items"), list):
            fail(f"phases[{i}].items must be a list", errors)


def validate_roadmap_items(data: Dict[str, Any], errors: List[str]) -> None:
    phases = data.get("phases") or []
    for i, phase in enumerate(phases, 1):
        if not isinstance(phase, dict):
            continue

        for j, item in enumerate(phase.get("items") or [], 1):
            if not isinstance(item, dict):
                fail(f"phases[{i}].items[{j}] is not a mapping", errors)
                continue
            if "id" not in item:
                fail(f"phases[{i}].items[{j}] missing `id`", errors)
            if "title" not in item:
                fail(f"phases[{i}].items[{j}] missing `title`", errors)


def validate_raw_phase_nesting(raw_text: str, errors: List[str]) -> None:
    phase_line = re.compile(r"^(?P<indent>\s*)phases:\s*$")
    phase_entry = re.compile(r"^(?P<indent>\s*)-\s*phase:\s*")
    lines = raw_text.splitlines()
    phase_key_indent = None
    # locate the indentation level for `phases`
    for idx, line in enumerate(lines, 1):
        m = phase_line.match(line)
        if m:
            phase_key_indent = len(m.group("indent"))
            break

    if phase_key_indent is None:
        fail("`phases` key not found for indentation checks", errors)
        return

    expected = {phase_key_indent, phase_key_indent + 2}
    for idx, line in enumerate(lines, 1):
        m = phase_entry.match(line)
        if not m:
            continue
        indent = len(m.group("indent"))
        if indent not in expected:
            fail(
                f"line {idx}: phase entry indentation {indent} spaces, expected {expected} "
                f"(phase block nesting regression)",
                errors,
            )


def validate_detail_quoting(raw_text: str, errors: List[str]) -> None:
    """Require plain `detail` scalars containing ':' to be quoted on same line."""
    for idx, line in enumerate(raw_text.splitlines(), 1):
        m = re.match(r"^\s*detail:\s*(?P<value>.*)$", line)
        if not m:
            continue
        raw_value = m.group("value").rstrip()
        if not raw_value:
            # Multiline style is not supported in this simple hardening check.
            continue
        if raw_value[0] in {"'", '"'}:
            continue
        if ":" in raw_value:
            fail(
                f"line {idx}: `detail` contains ':' but is not explicitly quoted",
                errors,
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default="audits/logic-audit-improvement-roadmap.yaml",
        help="Path to roadmap YAML file",
    )
    args = parser.parse_args()

    path = Path(args.path)
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise SystemExit(f"ERROR: unable to read {path}: {e}") from e

    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as e:
        raise SystemExit(f"ERROR: YAML parse failed for {path}: {e}") from e
    if data is None:
        raise SystemExit(f"ERROR: empty roadmap YAML in {path}")
    if not isinstance(data, dict):
        raise SystemExit(f"ERROR: expected mapping root in {path}")

    errors: List[str] = []
    validate_phase_structure(data, errors)
    validate_roadmap_items(data, errors)
    validate_raw_phase_nesting(raw_text, errors)
    validate_detail_quoting(raw_text, errors)

    if errors:
        print(f"Validation failed for {path}:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print(f"{path}: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
