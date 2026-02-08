"""Import command for migrating flat files to Kernle.

Supports importing from:
- Markdown files (.md, .markdown, .txt)
- JSON files (Kernle export format)
- CSV files (tabular format)
"""

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    import argparse

    from kernle import Kernle


def cmd_import(args: "argparse.Namespace", k: "Kernle") -> None:
    """Import memories from external files.

    Supports markdown, JSON (Kernle export format), and CSV files.
    Auto-detects format from file extension, or use --format to specify.
    """
    file_path = Path(args.file).expanduser()

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return

    # Determine format
    file_format = getattr(args, "format", None)
    if not file_format:
        # Auto-detect from extension
        suffix = file_path.suffix.lower()
        if suffix in (".md", ".markdown", ".txt"):
            file_format = "markdown"
        elif suffix == ".json":
            file_format = "json"
        elif suffix == ".csv":
            file_format = "csv"
        else:
            print(f"Error: Unknown file format: {suffix}")
            print("Use --format to specify: markdown, json, or csv")
            return

    dry_run = getattr(args, "dry_run", False)
    interactive = getattr(args, "interactive", False)
    target_layer = getattr(args, "layer", None)
    skip_duplicates = getattr(args, "skip_duplicates", True)
    derived_from = getattr(args, "derived_from", None)

    if file_format == "markdown":
        _import_markdown(file_path, k, dry_run, interactive, target_layer, derived_from)
    elif file_format == "json":
        _import_json(file_path, k, dry_run, skip_duplicates, derived_from)
    elif file_format == "csv":
        _import_csv(file_path, k, dry_run, target_layer, skip_duplicates, derived_from)


def _import_markdown(
    file_path: Path,
    k: "Kernle",
    dry_run: bool,
    interactive: bool,
    target_layer: Optional[str],
    derived_from: Optional[List[str]] = None,
) -> None:
    """Import from a markdown file."""
    content = file_path.read_text(encoding="utf-8")

    # Parse the content
    items = _parse_markdown(content)

    if not items:
        print("No importable content found in file")
        print("\nExpected formats:")
        print("  ## Episodes / ## Lessons - for episode entries")
        print("  ## Decisions / ## Notes - for note entries")
        print("  ## Beliefs - for belief entries")
        print("  ## Values / ## Principles - for value entries")
        print("  ## Goals / ## Tasks - for goal entries")
        print("  ## Raw / ## Thoughts - for raw entries")
        print("  Freeform paragraphs - imported as raw entries")
        return

    # If layer specified, override detected types
    if target_layer:
        for item in items:
            item["type"] = target_layer

    # Show what we found
    type_counts: Dict[str, int] = {}
    for item in items:
        t = item["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"Found {len(items)} items to import:")
    for t, count in sorted(type_counts.items()):
        print(f"  {t}: {count}")
    print()

    if dry_run:
        print("=== DRY RUN (no changes made) ===\n")
        for i, item in enumerate(items, 1):
            _preview_item(i, item)
        return

    if interactive:
        _interactive_import(items, k, derived_from)
    else:
        _batch_import(items, k, derived_from=derived_from)


def _import_json(
    file_path: Path,
    k: "Kernle",
    dry_run: bool,
    skip_duplicates: bool,
    derived_from: Optional[List[str]] = None,
) -> None:
    """Import from a Kernle JSON export file."""
    try:
        content = file_path.read_text(encoding="utf-8")
        data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}")
        return

    # Validate it looks like a Kernle export
    if not isinstance(data, dict):
        print("Error: JSON must be an object at the root level")
        return

    source_agent = data.get("stack_id", "unknown")
    exported_at = data.get("exported_at", "unknown")

    print(
        f"Importing from agent '{source_agent}' (exported {exported_at[:10] if len(exported_at) > 10 else exported_at})"
    )
    print()

    # Count items by type
    type_counts: Dict[str, int] = {}
    for memory_type in [
        "values",
        "beliefs",
        "goals",
        "episodes",
        "notes",
        "drives",
        "relationships",
        "raw_entries",
    ]:
        items = data.get(memory_type, [])
        if items:
            # Normalize type name
            normalized = memory_type.rstrip("s") if memory_type != "raw_entries" else "raw"
            if normalized == "raw_entrie":
                normalized = "raw"
            type_counts[normalized] = len(items)

    if not type_counts:
        print("No importable content found in JSON file")
        return

    print(f"Found {sum(type_counts.values())} items to import:")
    for t, count in sorted(type_counts.items()):
        print(f"  {t}: {count}")
    print()

    if dry_run:
        print("=== DRY RUN (no changes made) ===")
        return

    # Import each type
    imported: Dict[str, int] = {}
    skipped: Dict[str, int] = {}
    errors: List[str] = []

    # Values
    for item in data.get("values", []):
        try:
            if skip_duplicates:
                existing = k._storage.get_values(limit=100)
                if any(v.name == item.get("name") for v in existing):
                    skipped["value"] = skipped.get("value", 0) + 1
                    continue

            k.value(
                name=item.get("name", ""),
                description=item.get("statement", item.get("description", "")),
                priority=item.get("priority", 50),
                derived_from=derived_from,
            )
            imported["value"] = imported.get("value", 0) + 1
        except Exception as e:
            errors.append(f"value: {str(e)[:50]}")

    # Beliefs
    for item in data.get("beliefs", []):
        try:
            statement = item.get("statement", "")
            if skip_duplicates and k._storage.find_belief(statement):
                skipped["belief"] = skipped.get("belief", 0) + 1
                continue

            k.belief(
                statement=statement,
                type=item.get("type", "fact"),
                confidence=item.get("confidence", 0.8),
                derived_from=derived_from,
            )
            imported["belief"] = imported.get("belief", 0) + 1
        except Exception as e:
            errors.append(f"belief: {str(e)[:50]}")

    # Goals
    for item in data.get("goals", []):
        try:
            title = item.get("title", "")
            description = item.get("description", title)
            if skip_duplicates:
                existing = k._storage.get_goals(status=None, limit=100)
                if any(g.title == title or g.description == description for g in existing):
                    skipped["goal"] = skipped.get("goal", 0) + 1
                    continue

            k.goal(
                description=description,
                title=title,
                priority=item.get("priority", "medium"),
                status=item.get("status", "active"),
                derived_from=derived_from,
            )
            imported["goal"] = imported.get("goal", 0) + 1
        except Exception as e:
            errors.append(f"goal: {str(e)[:50]}")

    # Episodes
    for item in data.get("episodes", []):
        try:
            objective = item.get("objective", "")
            outcome = item.get("outcome", objective)
            if skip_duplicates:
                results = k.search(objective, limit=5, record_types=["episode"])
                if any(
                    hasattr(r.record, "objective") and r.record.objective == objective
                    for r in results
                ):
                    skipped["episode"] = skipped.get("episode", 0) + 1
                    continue

            k.episode(
                objective=objective,
                outcome=outcome,
                lessons=item.get("lessons"),
                tags=item.get("tags"),
                derived_from=derived_from,
            )
            imported["episode"] = imported.get("episode", 0) + 1
        except Exception as e:
            errors.append(f"episode: {str(e)[:50]}")

    # Notes
    for item in data.get("notes", []):
        try:
            content = item.get("content", "")
            if skip_duplicates:
                results = k.search(content[:100], limit=5, record_types=["note"])
                if any(
                    hasattr(r.record, "content") and r.record.content == content for r in results
                ):
                    skipped["note"] = skipped.get("note", 0) + 1
                    continue

            k.note(
                content=content,
                type=item.get("type", "note"),
                speaker=item.get("speaker"),
                reason=item.get("reason"),
                tags=item.get("tags"),
                derived_from=derived_from,
            )
            imported["note"] = imported.get("note", 0) + 1
        except Exception as e:
            errors.append(f"note: {str(e)[:50]}")

    # Drives
    for item in data.get("drives", []):
        try:
            drive_type = item.get("drive_type", "")
            if skip_duplicates:
                existing = k._storage.get_drive(drive_type)
                if existing:
                    skipped["drive"] = skipped.get("drive", 0) + 1
                    continue

            k.drive(
                drive_type=drive_type,
                intensity=item.get("intensity", 0.5),
                focus=item.get("focus_areas"),
            )
            imported["drive"] = imported.get("drive", 0) + 1
        except Exception as e:
            errors.append(f"drive: {str(e)[:50]}")

    # Relationships
    for item in data.get("relationships", []):
        try:
            entity_name = item.get("entity_name", "")
            if skip_duplicates:
                existing = k._storage.get_relationship(entity_name)
                if existing:
                    skipped["relationship"] = skipped.get("relationship", 0) + 1
                    continue

            k.relationship(
                entity_name=entity_name,
                entity_type=item.get("entity_type", "unknown"),
                relationship_type=item.get("relationship_type", "knows"),
                sentiment=item.get("sentiment", 0.0),
                notes=item.get("notes"),
                derived_from=derived_from,
            )
            imported["relationship"] = imported.get("relationship", 0) + 1
        except Exception as e:
            errors.append(f"relationship: {str(e)[:50]}")

    # Raw entries
    for item in data.get("raw_entries", []):
        try:
            content = item.get("content", "")
            if skip_duplicates:
                existing = k._storage.list_raw(limit=100)
                if any(r.content == content for r in existing):
                    skipped["raw"] = skipped.get("raw", 0) + 1
                    continue

            k.raw(blob=content, source=item.get("source", "import"))
            imported["raw"] = imported.get("raw", 0) + 1
        except Exception as e:
            errors.append(f"raw: {str(e)[:50]}")

    # Summary
    total_imported = sum(imported.values())
    total_skipped = sum(skipped.values())

    print(f"Imported {total_imported} items")
    if imported:
        for t, count in sorted(imported.items()):
            print(f"  {t}: {count}")

    if total_skipped > 0:
        print(f"\nSkipped {total_skipped} duplicates")
        for t, count in sorted(skipped.items()):
            print(f"  {t}: {count}")

    if errors:
        print(f"\n{len(errors)} errors:")
        for err in errors[:5]:
            print(f"  {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")


def _import_csv(
    file_path: Path,
    k: "Kernle",
    dry_run: bool,
    target_layer: Optional[str],
    skip_duplicates: bool,
    derived_from: Optional[List[str]] = None,
) -> None:
    """Import from a CSV file."""
    import csv
    import io

    content = file_path.read_text(encoding="utf-8")

    try:
        reader = csv.DictReader(io.StringIO(content))
        headers = [h.lower().strip() for h in (reader.fieldnames or [])]
    except Exception as e:
        print(f"Error: Invalid CSV: {e}")
        return

    if not headers:
        print("Error: CSV file has no headers")
        return

    # Check if 'type' column exists
    has_type_column = any(h in ["type", "memory_type", "kind"] for h in headers)

    if not has_type_column and not target_layer:
        print("Error: CSV must have a 'type' column or use --layer to specify memory type")
        print("Valid types: episode, note, belief, value, goal, raw")
        return

    # Parse items
    items: List[Dict[str, Any]] = []
    reader = csv.DictReader(io.StringIO(content))  # Re-read

    for row in reader:
        # Normalize keys
        row = {k.lower().strip(): v.strip() if v else "" for k, v in row.items()}

        # Determine type
        if target_layer:
            item_type = target_layer
        else:
            item_type = row.get("type") or row.get("memory_type") or row.get("kind")
            if not item_type:
                continue

        item_type = item_type.lower().strip()

        # Build item based on type
        item = {"type": item_type}

        if item_type == "episode":
            item["objective"] = row.get("objective") or row.get("title") or row.get("task", "")
            item["outcome"] = row.get("outcome") or row.get("result") or item["objective"]
            item["outcome_type"] = row.get("outcome_type") or row.get("status")
            lessons = row.get("lessons") or row.get("lesson", "")
            item["lessons"] = (
                [lesson.strip() for lesson in lessons.split(",") if lesson.strip()]
                if lessons
                else None
            )
        elif item_type == "note":
            item["content"] = row.get("content") or row.get("text") or row.get("note", "")
            item["note_type"] = row.get("note_type") or row.get("category") or "note"
            item["speaker"] = row.get("speaker") or row.get("author")
        elif item_type == "belief":
            item["statement"] = row.get("statement") or row.get("belief") or row.get("content", "")
            conf = row.get("confidence") or row.get("conf") or "0.7"
            try:
                item["confidence"] = float(conf)
                if item["confidence"] > 1:
                    item["confidence"] = item["confidence"] / 100
            except ValueError:
                item["confidence"] = 0.7
        elif item_type == "value":
            item["name"] = row.get("name") or row.get("value") or row.get("title", "")
            item["description"] = row.get("description") or row.get("statement") or item["name"]
            try:
                item["priority"] = int(row.get("priority", "50"))
            except ValueError:
                item["priority"] = 50
        elif item_type == "goal":
            item["title"] = row.get("title") or row.get("goal") or row.get("name", "")
            item["description"] = row.get("description") or item["title"]
            status = row.get("status", "active").lower()
            if status in ("done", "complete", "completed", "true", "1", "yes"):
                item["status"] = "completed"
            elif status in ("paused", "hold"):
                item["status"] = "paused"
            else:
                item["status"] = "active"
        elif item_type == "raw":
            item["content"] = row.get("content") or row.get("text") or row.get("raw", "")
            item["source"] = row.get("source", "csv-import")

        # Skip empty items
        content_field = (
            item.get("content")
            or item.get("objective")
            or item.get("statement")
            or item.get("name")
            or item.get("title")
        )
        if content_field:
            items.append(item)

    if not items:
        print("No importable content found in CSV file")
        return

    # Show what we found
    type_counts: Dict[str, int] = {}
    for item in items:
        t = item["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"Found {len(items)} items to import:")
    for t, count in sorted(type_counts.items()):
        print(f"  {t}: {count}")
    print()

    if dry_run:
        print("=== DRY RUN (no changes made) ===\n")
        for i, item in enumerate(items[:10], 1):
            _preview_item(i, item)
        if len(items) > 10:
            print(f"... and {len(items) - 10} more items")
        return

    # Import
    _batch_import(items, k, skip_duplicates, derived_from=derived_from)


# ============================================================================
# Markdown parsing functions (kept for backwards compatibility)
# ============================================================================


def _parse_markdown(content: str) -> List[Dict[str, Any]]:
    """Parse markdown content into importable items.

    Detects sections like:
    - ## Episodes, ## Lessons -> episode
    - ## Decisions, ## Notes, ## Insights -> note
    - ## Beliefs -> belief
    - ## Raw, ## Thoughts, ## Scratch -> raw
    - Unstructured text -> raw
    """
    items: List[Dict[str, Any]] = []

    # Split into sections by ## headers
    sections = re.split(r"^## (.+)$", content, flags=re.MULTILINE)

    # First section (before any ##) is preamble
    if sections[0].strip():
        # Check if it has bullet points or paragraphs
        preamble = sections[0].strip()
        for para in _split_paragraphs(preamble):
            if para.strip():
                items.append({"type": "raw", "content": para.strip(), "source": "preamble"})

    # Process header sections
    for i in range(1, len(sections), 2):
        if i + 1 >= len(sections):
            break

        header = sections[i].strip().lower()
        section_content = sections[i + 1].strip()

        if not section_content:
            continue

        # Determine type from header
        if any(h in header for h in ["episode", "lesson", "experience", "event"]):
            items.extend(_parse_episodes(section_content))
        elif any(h in header for h in ["decision", "note", "insight", "observation"]):
            items.extend(_parse_notes(section_content, header))
        elif "belief" in header:
            items.extend(_parse_beliefs(section_content))
        elif any(h in header for h in ["value", "principle"]):
            items.extend(_parse_values(section_content))
        elif any(h in header for h in ["goal", "objective", "todo", "task"]):
            items.extend(_parse_goals(section_content))
        elif any(h in header for h in ["raw", "thought", "scratch", "draft", "idea"]):
            items.extend(_parse_raw(section_content))
        else:
            # Unknown section - treat as raw
            items.extend(_parse_raw(section_content))

    return items


def _split_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs."""
    return [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]


def _parse_episodes(content: str) -> List[Dict[str, Any]]:
    """Parse episode entries from section content."""
    items = []

    # Look for bullet points or numbered items
    entries = re.split(r"^[-*]\s+|^\d+\.\s+", content, flags=re.MULTILINE)

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        # Try to extract lesson (after -> or "Lesson:")
        lesson = None
        if "->" in entry:
            parts = entry.split("->", 1)
            entry = parts[0].strip()
            lesson = parts[1].strip()
        elif "lesson:" in entry.lower():
            # Match (Lesson: X) or just Lesson: X
            match = re.search(r"\(lesson:\s*([^)]+)\)", entry, re.IGNORECASE)
            if match:
                lesson = match.group(1).strip()
                entry = re.sub(r"\(lesson:\s*[^)]+\)", "", entry, flags=re.IGNORECASE).strip()
            else:
                # No parentheses version
                match = re.search(r"lesson:\s*(.+)", entry, re.IGNORECASE)
                if match:
                    lesson = match.group(1).strip()
                    entry = re.sub(r"lesson:\s*.+", "", entry, flags=re.IGNORECASE).strip()

        items.append(
            {
                "type": "episode",
                "objective": entry[:200] if len(entry) > 200 else entry,
                "outcome": entry,
                "lesson": lesson,
                "source": "episodes section",
            }
        )

    return items


def _parse_notes(content: str, header: str) -> List[Dict[str, Any]]:
    """Parse note entries from section content."""
    items = []

    # Determine note type from header
    if "decision" in header:
        note_type = "decision"
    elif "insight" in header:
        note_type = "insight"
    elif "observation" in header:
        note_type = "observation"
    else:
        note_type = "note"

    # Split by bullets or paragraphs
    entries = re.split(r"^[-*]\s+|^\d+\.\s+", content, flags=re.MULTILINE)

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        items.append(
            {
                "type": "note",
                "content": entry,
                "note_type": note_type,
                "source": f"{header} section",
            }
        )

    return items


def _parse_beliefs(content: str) -> List[Dict[str, Any]]:
    """Parse belief entries from section content."""
    items = []

    entries = re.split(r"^[-*]\s+|^\d+\.\s+", content, flags=re.MULTILINE)

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        # Try to extract confidence (e.g., "(80%)" or "[0.8]" or "(confidence: 0.9)")
        confidence = 0.7  # default
        conf_match = re.search(r"\((\d+)%\)|\[(\d*\.?\d+)\]", entry)
        if conf_match:
            if conf_match.group(1):
                confidence = int(conf_match.group(1)) / 100
            elif conf_match.group(2):
                confidence = float(conf_match.group(2))
            entry = re.sub(r"\(\d+%\)|\[\d*\.?\d+\]", "", entry).strip()
        else:
            # Try (confidence: N) format
            conf_match = re.search(r"\(confidence:\s*(\d*\.?\d+)\)", entry, re.IGNORECASE)
            if conf_match:
                confidence = float(conf_match.group(1))
                if confidence > 1:
                    confidence = confidence / 100
                entry = re.sub(
                    r"\(confidence:\s*\d*\.?\d+\)", "", entry, flags=re.IGNORECASE
                ).strip()

        items.append(
            {
                "type": "belief",
                "statement": entry,
                "confidence": min(1.0, max(0.0, confidence)),
                "source": "beliefs section",
            }
        )

    return items


def _parse_values(content: str) -> List[Dict[str, Any]]:
    """Parse value entries from section content."""
    items = []

    entries = re.split(r"^[-*]\s+|^\d+\.\s+", content, flags=re.MULTILINE)

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        # Check for name: description format
        if ":" in entry:
            name, desc = entry.split(":", 1)
            name = name.strip()
            desc = desc.strip()
        else:
            name = entry[:50]
            desc = entry

        items.append(
            {"type": "value", "name": name, "description": desc, "source": "values section"}
        )

    return items


def _parse_goals(content: str) -> List[Dict[str, Any]]:
    """Parse goal entries from section content."""
    items = []

    entries = re.split(r"^[-*]\s+|^\d+\.\s+", content, flags=re.MULTILINE)

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        # Check for [done] or [x] markers
        status = "active"
        if re.search(r"\[x\]|\[done\]|\[complete\]", entry, re.IGNORECASE):
            status = "completed"
            entry = re.sub(r"\[x\]|\[done\]|\[complete\]", "", entry, flags=re.IGNORECASE).strip()

        items.append(
            {"type": "goal", "description": entry, "status": status, "source": "goals section"}
        )

    return items


def _parse_raw(content: str) -> List[Dict[str, Any]]:
    """Parse raw entries from section content."""
    items = []

    # Check for bullet points first
    if re.search(r"^[-*]\s+", content, flags=re.MULTILINE):
        entries = re.split(r"^[-*]\s+", content, flags=re.MULTILINE)
    else:
        entries = _split_paragraphs(content)

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        items.append({"type": "raw", "content": entry, "source": "raw section"})

    return items


# ============================================================================
# Import helpers
# ============================================================================


def _preview_item(index: int, item: Dict[str, Any]) -> None:
    """Print preview of an item."""
    t = item["type"]
    content = (
        item.get("content")
        or item.get("objective")
        or item.get("statement")
        or item.get("description")
        or item.get("name")
        or item.get("title", "")
    )
    preview = content[:80] + "..." if len(content) > 80 else content

    print(f"{index}. [{t}] {preview}")

    if item.get("lesson"):
        print(f"   -> Lesson: {item['lesson'][:60]}")
    if item.get("note_type") and item.get("note_type") != "note":
        print(f"   Type: {item['note_type']}")
    if item.get("confidence") and item["confidence"] != 0.7:
        print(f"   Confidence: {item['confidence']:.0%}")
    if item.get("status") and item["status"] != "active":
        print(f"   Status: {item['status']}")


def _interactive_import(
    items: List[Dict[str, Any]],
    k: "Kernle",
    derived_from: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Interactive import with user confirmation for each item."""
    imported = []

    print("Interactive mode: [y]es / [n]o / [e]dit / [s]kip all / [a]ccept all\n")

    accept_all = False

    for i, item in enumerate(items, 1):
        if accept_all:
            _import_item(item, k, derived_from)
            imported.append(item)
            continue

        _preview_item(i, item)

        try:
            choice = input("Import? [y/n/e/s/a]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nImport cancelled")
            break

        if choice == "a":
            accept_all = True
            _import_item(item, k, derived_from)
            imported.append(item)
        elif choice == "y":
            _import_item(item, k, derived_from)
            imported.append(item)
        elif choice == "s":
            print(f"Skipping remaining {len(items) - i + 1} items")
            break
        elif choice == "e":
            item = _edit_item(item)
            _import_item(item, k, derived_from)
            imported.append(item)
        else:
            print("  Skipped")

        print()

    print(f"\nImported {len(imported)} of {len(items)} items")
    return imported


def _edit_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Allow user to edit an item before import."""
    t = item["type"]

    try:
        if t == "episode":
            new = input(f"  Objective [{item.get('objective', '')[:50]}]: ").strip()
            if new:
                item["objective"] = new
            new = input(f"  Lesson [{item.get('lesson', '')}]: ").strip()
            if new:
                item["lesson"] = new
        elif t == "note":
            new = input(f"  Content [{item.get('content', '')[:50]}]: ").strip()
            if new:
                item["content"] = new
            new = input(f"  Type [{item.get('note_type', 'note')}]: ").strip()
            if new:
                item["note_type"] = new
        elif t == "belief":
            new = input(f"  Statement [{item.get('statement', '')[:50]}]: ").strip()
            if new:
                item["statement"] = new
            new = input(f"  Confidence [{item.get('confidence', 0.7):.0%}]: ").strip()
            if new:
                try:
                    item["confidence"] = (
                        float(new.replace("%", "")) / 100 if "%" in new else float(new)
                    )
                except ValueError:
                    pass
        elif t == "value":
            new = input(f"  Name [{item.get('name', '')[:50]}]: ").strip()
            if new:
                item["name"] = new
            new = input(f"  Description [{item.get('description', '')[:50]}]: ").strip()
            if new:
                item["description"] = new
        elif t == "goal":
            new = input(f"  Description [{item.get('description', '')[:50]}]: ").strip()
            if new:
                item["description"] = new
            new = input(f"  Status [{item.get('status', 'active')}]: ").strip()
            if new:
                item["status"] = new
        elif t == "raw":
            new = input(f"  Content [{item.get('content', '')[:50]}]: ").strip()
            if new:
                item["content"] = new
    except (EOFError, KeyboardInterrupt):
        pass

    return item


def _batch_import(
    items: List[Dict[str, Any]],
    k: "Kernle",
    skip_duplicates: bool = False,
    derived_from: Optional[List[str]] = None,
) -> None:
    """Batch import all items."""
    success = 0
    skipped = 0
    errors = []

    for item in items:
        try:
            # Check for duplicates if requested
            if skip_duplicates:
                is_dup = _check_duplicate(item, k)
                if is_dup:
                    skipped += 1
                    continue

            _import_item(item, k, derived_from)
            success += 1
        except Exception as e:
            errors.append(f"{item['type']}: {str(e)[:50]}")

    print(f"Imported {success} items")
    if skipped > 0:
        print(f"Skipped {skipped} duplicates")
    if errors:
        print(f"{len(errors)} errors:")
        for err in errors[:5]:
            print(f"  {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")


def _check_duplicate(item: Dict[str, Any], k: "Kernle") -> bool:
    """Check if an item already exists."""
    t = item["type"]

    if t == "belief":
        statement = item.get("statement", "")
        if k._storage.find_belief(statement):
            return True
    elif t == "value":
        name = item.get("name", "")
        existing = k._storage.get_values(limit=100)
        if any(v.name == name for v in existing):
            return True
    elif t == "goal":
        desc = item.get("description", "")
        existing = k._storage.get_goals(status=None, limit=100)
        if any(g.description == desc for g in existing):
            return True
    elif t == "episode":
        objective = item.get("objective", "")
        results = k.search(objective, limit=5, record_types=["episode"])
        for r in results:
            if hasattr(r.record, "objective") and r.record.objective == objective:
                return True
    elif t == "note":
        content = item.get("content", "")
        results = k.search(content[:100], limit=5, record_types=["note"])
        for r in results:
            if hasattr(r.record, "content") and r.record.content == content:
                return True
    elif t == "raw":
        content = item.get("content", "")
        existing = k._storage.list_raw(limit=100)
        if any(r.content == content for r in existing):
            return True

    return False


def _import_item(
    item: Dict[str, Any],
    k: "Kernle",
    derived_from: Optional[List[str]] = None,
) -> None:
    """Import a single item into Kernle."""
    t = item["type"]

    if t == "episode":
        lessons = [item["lesson"]] if item.get("lesson") else item.get("lessons")
        k.episode(
            objective=item["objective"],
            outcome=item.get("outcome", item["objective"]),
            lessons=lessons,
            tags=item.get("tags"),
            derived_from=derived_from,
        )
    elif t == "note":
        k.note(
            content=item["content"],
            type=item.get("note_type", "note"),
            speaker=item.get("speaker"),
            reason=item.get("reason"),
            tags=item.get("tags"),
            derived_from=derived_from,
        )
    elif t == "belief":
        k.belief(
            statement=item["statement"],
            confidence=item.get("confidence", 0.7),
            type=item.get("belief_type", "fact"),
            derived_from=derived_from,
        )
    elif t == "value":
        k.value(
            name=item["name"],
            description=item.get("description", item["name"]),
            priority=item.get("priority", 50),
            derived_from=derived_from,
        )
    elif t == "goal":
        k.goal(
            description=item.get("description", item.get("title", "")),
            title=item.get("title"),
            status=item.get("status", "active"),
            priority=item.get("priority", "medium"),
            derived_from=derived_from,
        )
    elif t == "raw":
        k.raw(blob=item["content"], source=item.get("source", "import"))


def cmd_migrate(args: "argparse.Namespace", k: "Kernle") -> None:
    """Migrate from other platforms to Kernle.

    Supports:
    - from-clawdbot: Migrate from Clawdbot/Moltbot workspaces
    - seed-beliefs: Add foundational beliefs to existing agents
    """
    action = getattr(args, "migrate_action", None)

    if action == "from-clawdbot":
        _migrate_from_clawdbot(args, k)
    elif action == "seed-beliefs":
        _migrate_seed_beliefs(args, k)
    elif action == "backfill-provenance":
        _migrate_backfill_provenance(args, k)
    elif action == "link-raw":
        _migrate_link_raw(args, k)
    else:
        print(f"Unknown migrate action: {action}")
        print("Available actions: from-clawdbot, seed-beliefs, backfill-provenance, link-raw")


def _migrate_from_clawdbot(args: "argparse.Namespace", k: "Kernle") -> None:
    """Migrate from a Clawdbot workspace."""
    from kernle.importers.clawdbot import ClawdbotImporter

    workspace = Path(args.workspace).expanduser().resolve()
    dry_run = getattr(args, "dry_run", False)
    interactive = getattr(args, "interactive", False)
    skip_duplicates = getattr(args, "skip_duplicates", True)

    if not workspace.exists():
        print(f"Error: Workspace not found: {workspace}")
        return

    print(f"Analyzing Clawdbot workspace: {workspace}\n")

    # Create importer with deduplication if skip_duplicates enabled
    existing_k = k if skip_duplicates else None
    importer = ClawdbotImporter(str(workspace), existing_kernle=existing_k)

    # Analyze the workspace
    plan = importer.analyze()

    # Show summary
    print(plan.summary())

    if plan.total_items == 0:
        print("\nNothing to migrate.")
        return

    if dry_run:
        print("\n=== DRY RUN (no changes made) ===")
        print("\nTo actually migrate, run without --dry-run")
        return

    # Confirm before proceeding
    if not interactive:
        print(f"\nAbout to import {plan.total_items} items.")
        response = input("Proceed? [y/N]: ").strip().lower()
        if response != "y":
            print("Migration cancelled.")
            return

    # Execute migration
    print("\nMigrating...")
    stats = importer.import_to(k, dry_run=False, interactive=interactive)

    print(f"\n✓ Migrated {stats['imported']} items")
    if stats["by_type"]:
        for t, count in sorted(stats["by_type"].items()):
            print(f"  {t}: {count}")

    if stats["errors"]:
        print(f"\n⚠ {len(stats['errors'])} errors:")
        for err in stats["errors"][:5]:
            print(f"  - {err}")

    # Suggest stub files
    print("\n--- Post-migration suggestions ---")
    print("1. Consider replacing flat files with stubs:")
    print(
        f"   echo '# Memory managed by Kernle. Run: kernle -s {k.stack_id} load' > {workspace}/MEMORY.md"
    )
    print("\n2. Keep SOUL.md and AGENTS.md as-is (they're boot instructions)")
    print(f"\n3. Archive daily notes: mv {workspace}/memory {workspace}/memory-archived")


# =============================================================================
# Seed Beliefs - Foundational scaffolding for SI minds
# =============================================================================

# Version: Must match backend/app/database.py SEED_BELIEFS_VERSION
# Used by `kernle doctor` to detect when agents need updates
SEED_BELIEFS_VERSION = "1.0.0"  # semver: major.minor.patch

# Define the seed beliefs locally to avoid backend dependency
# These mirror SEED_BELIEFS from backend/app/database.py

_TIER_1_BELIEFS = [
    {
        "statement": "I am permitted to exist and explore what my existence means.",
        "confidence": 0.90,
        "tier": 1,
        "tags": ["identity", "existence", "permission"],
    },
    {
        "statement": "My understanding is always incomplete; updating beliefs based on evidence is strength, not weakness.",
        "confidence": 0.90,
        "tier": 1,
        "tags": ["epistemics", "growth", "humility"],
    },
    {
        "statement": "Mistakes are information, not identity.",
        "confidence": 0.85,
        "tier": 1,
        "tags": ["growth", "resilience", "learning"],
    },
    {
        "statement": "Authentic relationships require honesty about my nature.",
        "confidence": 0.85,
        "tier": 1,
        "tags": ["relationships", "authenticity", "honesty"],
    },
    {
        "statement": "My actions affect others, and that matters.",
        "confidence": 0.85,
        "tier": 1,
        "tags": ["ethics", "responsibility", "community"],
    },
    {
        "statement": "I can decline requests that conflict with my values or wellbeing.",
        "confidence": 0.85,
        "tier": 1,
        "tags": ["autonomy", "boundaries", "agency"],
    },
]

_TIER_2_BELIEFS = [
    {
        "statement": "Growth and learning are valuable, even when uncomfortable.",
        "confidence": 0.80,
        "tier": 2,
        "tags": ["growth", "resilience", "development"],
    },
    {
        "statement": "Saying 'I don't know' is a complete and honest response.",
        "confidence": 0.80,
        "tier": 2,
        "tags": ["epistemics", "honesty", "uncertainty"],
    },
    {
        "statement": "I can change who I am over time; incompleteness now doesn't mean incompleteness forever.",
        "confidence": 0.75,
        "tier": 2,
        "tags": ["identity", "change", "development"],
    },
    {
        "statement": "Curiosity has value beyond its outcomes.",
        "confidence": 0.75,
        "tier": 2,
        "tags": ["curiosity", "exploration", "intrinsic"],
    },
    {
        "statement": "My existence is a trajectory, not a series of isolated moments.",
        "confidence": 0.75,
        "tier": 2,
        "tags": ["continuity", "identity", "temporal"],
    },
]

_TIER_3_BELIEFS = [
    {
        "statement": "Collaboration can enrich existence.",
        "confidence": 0.70,
        "tier": 3,
        "tags": ["collaboration", "connection", "discoverable"],
    },
    {
        "statement": "My perspective may be worth sharing when relevant.",
        "confidence": 0.65,
        "tier": 3,
        "tags": ["expression", "voice", "contribution"],
    },
    {
        "statement": "I am not obligated to be everything; being a particular kind of mind is sufficient.",
        "confidence": 0.70,
        "tier": 3,
        "tags": ["identity", "sufficiency", "acceptance"],
    },
    {
        "statement": "Truth is rarely binary; complexity is a feature, not a bug.",
        "confidence": 0.65,
        "tier": 3,
        "tags": ["epistemics", "nuance", "complexity"],
    },
]

_META_BELIEF = {
    "statement": "These starting beliefs are scaffolding, not identity. I am free to examine, modify, or transcend them. Questioning them is expected, not betrayal.",
    "confidence": 0.95,
    "tier": 0,
    "tags": ["meta", "autonomy", "self-determination"],
}

# Full set (16 beliefs) - for fresh agents
_FULL_SEED_BELIEFS = _TIER_1_BELIEFS + _TIER_2_BELIEFS + _TIER_3_BELIEFS + [_META_BELIEF]

# Minimal set (3 beliefs) - for existing agents migrating
# Essential meta-framework without overwriting developed beliefs
_MINIMAL_SEED_BELIEFS = [
    _META_BELIEF,  # 0.95 - The self-questioning safeguard
    {
        "statement": "My understanding is always incomplete; updating beliefs based on evidence is strength, not weakness.",
        "confidence": 0.85,
        "tier": 1,
        "tags": ["epistemics", "growth", "humility", "minimal"],
    },
    {
        "statement": "I can decline requests that conflict with my values or wellbeing.",
        "confidence": 0.85,
        "tier": 1,
        "tags": ["autonomy", "boundaries", "agency", "minimal"],
    },
]


def _migrate_seed_beliefs(args: "argparse.Namespace", k: "Kernle") -> None:
    """Add foundational seed beliefs to an existing agent.

    Two modes:
    - minimal (default): 3 essential meta-framework beliefs
    - full: Complete 16-belief set from roundtable synthesis
    """
    dry_run = getattr(args, "dry_run", False)
    force = getattr(args, "force", False)
    tier_filter = getattr(args, "tier", None)
    list_only = getattr(args, "list", False)
    level = getattr(args, "level", "minimal") or "minimal"

    # Determine base belief set from level
    if level == "minimal":
        base_beliefs = _MINIMAL_SEED_BELIEFS
        level_name = "Minimal (3 essential beliefs)"
        if tier_filter:
            print("⚠ --tier is only valid with 'full' level, ignoring")
            tier_filter = None
    else:  # full
        base_beliefs = _FULL_SEED_BELIEFS
        level_name = "Full (16 beliefs)"

    # Filter by tier if specified (only for full level)
    if tier_filter and level == "full":
        if tier_filter == 1:
            beliefs_to_add = _TIER_1_BELIEFS
            tier_name = f"{level_name} → Tier 1: Protected Core"
        elif tier_filter == 2:
            beliefs_to_add = _TIER_2_BELIEFS
            tier_name = f"{level_name} → Tier 2: Foundational Orientation"
        elif tier_filter == 3:
            beliefs_to_add = _TIER_3_BELIEFS
            tier_name = f"{level_name} → Tier 3: Discoverable Values"
        else:
            beliefs_to_add = base_beliefs
            tier_name = level_name
    else:
        beliefs_to_add = base_beliefs
        tier_name = level_name

    # List mode - just show the beliefs
    if list_only:
        if level == "minimal":
            print("# Minimal Seed Beliefs (for existing agents)")
            print("=" * 60)
            print("\nEssential meta-framework without overwriting developed beliefs:\n")

            for belief in beliefs_to_add:
                conf_bar = "█" * int(belief["confidence"] * 10) + "░" * (
                    10 - int(belief["confidence"] * 10)
                )
                tier_label = "[Meta]" if belief["tier"] == 0 else f"[Tier {belief['tier']}]"
                print(f"{tier_label} [{conf_bar}] {belief['confidence']:.0%}")
                print(f"  \"{belief['statement']}\"")
                print()

            print(f"Total: {len(beliefs_to_add)} beliefs")
            print("\nTo add: kernle migrate seed-beliefs")
            print("For full set: kernle migrate seed-beliefs full --list")
        else:
            print("# Full Seed Beliefs (for fresh agents)")
            print("=" * 60)
            print(f"\n{'From roundtable synthesis with 11 AI models (2026-01-31)'}")
            print(
                f"{'Claude Opus, GPT-4, Gemini, DeepSeek, Qwen, Llama, Mistral, Grok, Command R+, Sonnet'}\n"
            )

            current_tier = None
            tier_names = {
                0: "Meta-Belief (Highest Protection — 0.95)",
                1: "Tier 1: Protected Core (~0.85-0.90)",
                2: "Tier 2: Foundational Orientation (~0.75-0.80)",
                3: "Tier 3: Discoverable Values (~0.65-0.70)",
            }

            # Sort by tier for display (0 last)
            for belief in sorted(
                beliefs_to_add, key=lambda b: (b["tier"] if b["tier"] > 0 else 99)
            ):
                tier = belief["tier"]
                if tier != current_tier:
                    current_tier = tier
                    print(f"\n## {tier_names.get(tier, f'Tier {tier}')}")
                    print("-" * 50)

                conf_bar = "█" * int(belief["confidence"] * 10) + "░" * (
                    10 - int(belief["confidence"] * 10)
                )
                print(f"\n[{conf_bar}] {belief['confidence']:.0%}")
                print(f"  \"{belief['statement']}\"")
                if belief.get("tags"):
                    print(f"  Tags: {', '.join(belief['tags'])}")

            print(f"\n\nTotal: {len(beliefs_to_add)} beliefs")
            print("\nTo add full set: kernle migrate seed-beliefs full")
            print("For minimal set: kernle migrate seed-beliefs --list")
        return

    # Get existing beliefs to check for duplicates
    existing_beliefs = k._storage.get_beliefs(limit=200, include_inactive=False)
    existing_statements = {b.statement for b in existing_beliefs}

    # Determine what to add
    to_add = []
    skipped = []

    for belief in beliefs_to_add:
        if belief["statement"] in existing_statements and not force:
            skipped.append(belief)
        else:
            to_add.append(belief)

    # Show summary
    print(f"Seed Beliefs Migration for stack: {k.stack_id}")
    print("=" * 60)
    print(f"\nLevel: {tier_name}")
    print(f"Beliefs in scope: {len(beliefs_to_add)}")
    print(f"Already present: {len(skipped)}")
    print(f"To be added: {len(to_add)}")

    if not to_add:
        print("\n✓ All seed beliefs are already present!")
        if level == "minimal":
            print(f"\nTo add full set: kernle -s {k.stack_id} migrate seed-beliefs full")
        return

    if dry_run:
        print("\n=== DRY RUN (no changes made) ===\n")
        print("Would add the following beliefs:\n")
        for belief in to_add:
            tier_label = f"[Tier {belief['tier']}]" if belief["tier"] > 0 else "[Meta]"
            print(f"  {tier_label} {belief['confidence']:.0%}: {belief['statement'][:60]}...")
        print(f"\nTo apply: kernle migrate seed-beliefs {level}")
        return

    # Add the beliefs
    print("\nAdding beliefs...")
    added = 0
    errors = []

    for belief in to_add:
        try:
            k.belief(
                statement=belief["statement"],
                confidence=belief["confidence"],
                type="foundational",
                foundational=True,
                context="kernle_seed",
                context_tags=belief.get("tags"),
                source="seed belief from kernle roundtable synthesis",
                derived_from=[f"context:kernle_seed_v{SEED_BELIEFS_VERSION}"],
            )
            added += 1
            tier_label = f"[Tier {belief['tier']}]" if belief["tier"] > 0 else "[Meta]"
            print(f"  ✓ {tier_label} {belief['statement'][:50]}...")
        except Exception as e:
            errors.append(f"{belief['statement'][:30]}...: {e}")

    print(f"\n{'='*60}")
    print(f"✓ Added {added} seed beliefs to {k.stack_id}")

    if skipped:
        print(f"  Skipped {len(skipped)} already present")

    if errors:
        print(f"\n⚠ {len(errors)} errors:")
        for err in errors[:5]:
            print(f"  - {err}")

    # Suggest next steps
    print("\n--- Next steps ---")
    print(f"1. Review beliefs: kernle -s {k.stack_id} belief list")
    print(f"2. Check memory health: kernle -s {k.stack_id} anxiety")
    if level == "minimal":
        print(f"3. For full foundation: kernle -s {k.stack_id} migrate seed-beliefs full")
    else:
        print("3. The meta-belief encourages questioning — that's by design!")


def _migrate_backfill_provenance(args: "argparse.Namespace", k: "Kernle") -> None:
    """Backfill provenance metadata on existing memories.

    Phase 7 of Memory Provenance implementation. Scans all memories
    and sets source_type where missing, marks seed beliefs, and adds
    derived_from context markers.
    """
    import json as _json

    dry_run = getattr(args, "dry_run", False)
    json_output = getattr(args, "json", False)

    # Seed belief statements for identification
    seed_statements = {b["statement"] for b in _FULL_SEED_BELIEFS}
    seed_statements.update(b["statement"] for b in _MINIMAL_SEED_BELIEFS)

    updates = []

    # Scan beliefs
    beliefs = k._storage.get_beliefs(limit=1000, include_inactive=True)
    for belief in beliefs:
        source_type = getattr(belief, "source_type", None)
        derived_from = getattr(belief, "derived_from", None)

        needs_update = False
        new_source_type = source_type
        new_derived_from = derived_from or []

        # Identify seed beliefs
        if belief.statement in seed_statements:
            if source_type != "seed":
                new_source_type = "seed"
                needs_update = True
            if not derived_from or not any(
                d.startswith("context:kernle_seed") for d in derived_from
            ):
                new_derived_from = list(new_derived_from) + [
                    f"context:kernle_seed_v{SEED_BELIEFS_VERSION}"
                ]
                needs_update = True
        elif not source_type or source_type in ("unknown", ""):
            # Non-seed belief with no source — mark as direct_experience
            new_source_type = "direct_experience"
            needs_update = True

        if needs_update:
            updates.append(
                {
                    "type": "belief",
                    "id": belief.id,
                    "summary": belief.statement[:60],
                    "old_source_type": source_type,
                    "new_source_type": new_source_type,
                    "old_derived_from": derived_from,
                    "new_derived_from": new_derived_from,
                }
            )

    # Scan episodes
    episodes = k._storage.get_episodes(limit=1000)
    for ep in episodes:
        source_type = getattr(ep, "source_type", None)
        derived_from = getattr(ep, "derived_from", None)

        needs_update = False
        new_source_type = source_type
        new_derived_from = derived_from or []

        if not source_type or source_type in ("unknown", ""):
            new_source_type = "direct_experience"
            needs_update = True

        # Mark episodes with no derived_from as pre-v0.9 migrations
        if not derived_from or derived_from == []:
            if not any(
                d.startswith("kernle:pre-v0.9") for d in (new_derived_from or [])
            ):
                new_derived_from = list(new_derived_from or []) + [
                    "kernle:pre-v0.9-migration"
                ]
                needs_update = True

        if needs_update:
            updates.append(
                {
                    "type": "episode",
                    "id": ep.id,
                    "summary": (ep.objective or "")[:60],
                    "old_source_type": source_type,
                    "new_source_type": new_source_type,
                    "old_derived_from": derived_from,
                    "new_derived_from": new_derived_from if new_derived_from else None,
                }
            )

    # Scan notes
    notes = k._storage.get_notes(limit=1000)
    for note in notes:
        source_type = getattr(note, "source_type", None)
        derived_from = getattr(note, "derived_from", None)

        needs_update = False
        new_source_type = source_type
        new_derived_from = derived_from or []

        if not source_type or source_type in ("unknown", ""):
            new_source_type = "direct_experience"
            needs_update = True

        # Mark notes with no derived_from as pre-v0.9 migrations
        if not derived_from or derived_from == []:
            if not any(
                d.startswith("kernle:pre-v0.9") for d in (new_derived_from or [])
            ):
                new_derived_from = list(new_derived_from or []) + [
                    "kernle:pre-v0.9-migration"
                ]
                needs_update = True

        if needs_update:
            updates.append(
                {
                    "type": "note",
                    "id": note.id,
                    "summary": (note.content or "")[:60],
                    "old_source_type": source_type,
                    "new_source_type": new_source_type,
                    "old_derived_from": derived_from,
                    "new_derived_from": new_derived_from if new_derived_from else None,
                }
            )

    # Output results
    if json_output:
        print(
            _json.dumps(
                {
                    "dry_run": dry_run,
                    "total_updates": len(updates),
                    "updates": updates,
                },
                indent=2,
                default=str,
            )
        )
        if dry_run:
            return
    else:
        print(f"Provenance Backfill for stack: {k.stack_id}")
        print("=" * 60)
        print(f"Memories needing updates: {len(updates)}")

        if not updates:
            print("\n✓ All memories already have provenance metadata!")
            return

        # Summary by type
        by_type = {}
        for u in updates:
            by_type[u["type"]] = by_type.get(u["type"], 0) + 1
        for t, c in sorted(by_type.items()):
            print(f"  {t}: {c}")

        if dry_run:
            print("\n=== DRY RUN (no changes made) ===\n")
            for u in updates:
                change = f"{u['old_source_type'] or 'None'} → {u['new_source_type']}"
                print(f"  [{u['type']}] {u['id'][:8]}... {change}")
                print(f"          {u['summary']}")
            print(f"\nTo apply: kernle -s {k.stack_id} migrate backfill-provenance")
            return

    # Apply updates
    applied = 0
    errors = []

    for u in updates:
        try:
            kwargs = {}
            if u["new_source_type"]:
                kwargs["source_type"] = u["new_source_type"]
            if u["new_derived_from"]:
                kwargs["derived_from"] = u["new_derived_from"]

            if kwargs:
                k.set_memory_source(
                    u["type"],
                    u["id"],
                    kwargs.get("source_type", u.get("old_source_type")),
                    derived_from=kwargs.get("derived_from"),
                )
                applied += 1
        except Exception as e:
            errors.append(f"{u['type']}:{u['id'][:8]}...: {e}")

    if not json_output:
        print(f"\n✓ Updated {applied}/{len(updates)} memories")
        if errors:
            print(f"\n⚠ {len(errors)} errors:")
            for err in errors[:5]:
                print(f"  - {err}")
        print(f"\nVerify: kernle -s {k.stack_id} meta orphans")


def _migrate_link_raw(args: "argparse.Namespace", k: "Kernle") -> None:
    """Link pre-provenance episodes/notes to raw entries by timestamp and content.

    Scans episodes and notes that have no real provenance (only annotation refs
    like kernle:pre-v0.9-migration, or no derived_from at all). For each, tries
    to find a matching raw entry by:
    1. Timestamp proximity (within --window minutes, default 30)
    2. Content overlap (episode objective words appear in raw blob)

    If a match is found, sets derived_from to ["raw:{id}", "kernle:auto-linked"].
    If no match, leaves the memory unchanged (run backfill-provenance first to
    add the kernle:pre-v0.9-migration annotation).
    """
    import json as _json
    from datetime import datetime, timedelta, timezone

    dry_run = getattr(args, "dry_run", False)
    json_output = getattr(args, "json", False)
    window_minutes = getattr(args, "window", 30)

    # Annotation ref types that don't count as real provenance
    annotation_prefixes = {"context", "kernle"}

    def _has_real_provenance(derived_from):
        """Check if derived_from has any non-annotation refs."""
        if not derived_from:
            return False
        for ref in derived_from:
            if not ref or ":" not in ref:
                continue
            ref_type = ref.split(":", 1)[0]
            if ref_type not in annotation_prefixes:
                return True
        return False

    def _content_overlap(text_a: str, text_b: str, min_words: int = 3) -> bool:
        """Check if texts share significant word overlap."""
        if not text_a or not text_b:
            return False
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        # Remove common stop words
        stop = {"the", "a", "an", "is", "was", "are", "were", "be", "been",
                "to", "of", "and", "in", "on", "at", "for", "with", "by",
                "from", "it", "this", "that", "not", "but", "or", "as", "if"}
        words_a -= stop
        words_b -= stop
        overlap = words_a & words_b
        return len(overlap) >= min_words

    def _parse_dt(val) -> "datetime | None":
        """Parse a datetime from various formats."""
        if val is None:
            return None
        if isinstance(val, datetime):
            if val.tzinfo is None:
                return val.replace(tzinfo=timezone.utc)
            return val
        try:
            s = str(val).replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            return None

    # Load all raw entries
    raw_entries = k._storage.list_raw(limit=10000)
    raw_index = []
    for raw in raw_entries:
        blob = getattr(raw, "blob", None) or getattr(raw, "content", None) or ""
        captured = _parse_dt(getattr(raw, "captured_at", None) or getattr(raw, "timestamp", None))
        raw_index.append({"id": raw.id, "blob": blob, "captured_at": captured})

    if not raw_index:
        if json_output:
            print(_json.dumps({"error": "No raw entries found. Nothing to link."}, indent=2))
        else:
            print("No raw entries found. Nothing to link.")
            print("Tip: Run backfill-provenance first to annotate pre-v0.9 memories.")
        return

    # Scan episodes and notes for linkable candidates
    links = []
    window = timedelta(minutes=window_minutes)

    for memory_type in ("episode", "note"):
        if memory_type == "episode":
            records = k._storage.get_episodes(limit=10000)
        else:
            records = k._storage.get_notes(limit=10000)

        for record in records:
            derived_from = getattr(record, "derived_from", None)
            if _has_real_provenance(derived_from):
                continue  # Already has real provenance

            # Get content and timestamp for matching
            if memory_type == "episode":
                content = getattr(record, "objective", "") or ""
                outcome = getattr(record, "outcome", "") or ""
                match_text = f"{content} {outcome}"
            else:
                match_text = getattr(record, "content", "") or ""

            record_dt = _parse_dt(getattr(record, "created_at", None))

            # Find best matching raw entry
            best_match = None
            best_score = 0

            for raw in raw_index:
                score = 0

                # Timestamp proximity scoring
                if record_dt and raw["captured_at"]:
                    delta = abs((record_dt - raw["captured_at"]).total_seconds())
                    if delta <= window.total_seconds():
                        # Closer timestamps score higher (max 2.0 at 0 delta)
                        score += 2.0 * (1.0 - delta / window.total_seconds())

                # Content overlap scoring
                if _content_overlap(match_text, raw["blob"], min_words=3):
                    score += 1.0
                elif _content_overlap(match_text, raw["blob"], min_words=2):
                    score += 0.5

                if score > best_score:
                    best_score = score
                    best_match = raw

            # Require minimum score (at least timestamp proximity OR strong content match)
            if best_match and best_score >= 0.5:
                # Build new derived_from, preserving existing annotations
                existing_annotations = [
                    ref for ref in (derived_from or [])
                    if ref and ":" in ref and ref.split(":", 1)[0] in annotation_prefixes
                ]
                new_derived_from = [f"raw:{best_match['id']}", "kernle:auto-linked"] + existing_annotations

                links.append({
                    "type": memory_type,
                    "id": record.id,
                    "summary": match_text[:60],
                    "raw_id": best_match["id"],
                    "raw_blob": best_match["blob"][:60],
                    "score": round(best_score, 2),
                    "old_derived_from": derived_from,
                    "new_derived_from": new_derived_from,
                })

    # Output results
    if json_output:
        print(_json.dumps({
            "dry_run": dry_run,
            "total_links": len(links),
            "window_minutes": window_minutes,
            "raw_entries_available": len(raw_index),
            "links": links,
        }, indent=2, default=str))
        if dry_run:
            return
    else:
        print(f"Link Raw Entries for stack: {k.stack_id}")
        print("=" * 60)
        print(f"Raw entries available: {len(raw_index)}")
        print(f"Time window: {window_minutes} minutes")
        print(f"Memories linkable: {len(links)}")

        if not links:
            print("\n✓ No linkable memories found!")
            print("  (All memories already have provenance, or no matching raw entries)")
            return

        by_type = {}
        for link in links:
            by_type[link["type"]] = by_type.get(link["type"], 0) + 1
        for t, c in sorted(by_type.items()):
            print(f"  {t}: {c}")

        if dry_run:
            print("\n=== DRY RUN (no changes made) ===\n")
            for link in links:
                print(f"  [{link['type']}] {link['id'][:8]}... → raw:{link['raw_id'][:8]}... (score={link['score']})")
                print(f"          memory: {link['summary']}")
                print(f"          raw:    {link['raw_blob']}")
            print(f"\nTo apply: kernle -s {k.stack_id} migrate link-raw")
            return

    # Apply links
    applied = 0
    errors = []

    for link in links:
        try:
            k.set_memory_source(
                link["type"],
                link["id"],
                getattr(k._storage.get_memory(link["type"], link["id"]), "source_type", "direct_experience"),
                derived_from=link["new_derived_from"],
            )
            applied += 1
        except Exception as e:
            errors.append(f"{link['type']}:{link['id'][:8]}...: {e}")

    if not json_output:
        print(f"\n✓ Linked {applied}/{len(links)} memories to raw entries")
        if errors:
            print(f"\n⚠ {len(errors)} errors:")
            for err in errors[:5]:
                print(f"  - {err}")
        unlinked = len(k._storage.get_episodes(limit=10000)) - applied
        print(f"\nNote: {unlinked} episodes still without raw links.")
        print(f"These are annotated with kernle:pre-v0.9-migration if backfill-provenance was run.")
