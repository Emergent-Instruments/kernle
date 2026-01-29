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

    if file_format == "markdown":
        _import_markdown(file_path, k, dry_run, interactive, target_layer)
    elif file_format == "json":
        _import_json(file_path, k, dry_run, skip_duplicates)
    elif file_format == "csv":
        _import_csv(file_path, k, dry_run, target_layer, skip_duplicates)


def _import_markdown(
    file_path: Path, k: "Kernle", dry_run: bool, interactive: bool, target_layer: Optional[str]
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
        _interactive_import(items, k)
    else:
        _batch_import(items, k)


def _import_json(file_path: Path, k: "Kernle", dry_run: bool, skip_duplicates: bool) -> None:
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

    source_agent = data.get("agent_id", "unknown")
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

            k.raw(content=content, source=item.get("source", "import"), tags=item.get("tags"))
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
    file_path: Path, k: "Kernle", dry_run: bool, target_layer: Optional[str], skip_duplicates: bool
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
    _batch_import(items, k, skip_duplicates)


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


def _interactive_import(items: List[Dict[str, Any]], k: "Kernle") -> List[Dict[str, Any]]:
    """Interactive import with user confirmation for each item."""
    imported = []

    print("Interactive mode: [y]es / [n]o / [e]dit / [s]kip all / [a]ccept all\n")

    accept_all = False

    for i, item in enumerate(items, 1):
        if accept_all:
            _import_item(item, k)
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
            _import_item(item, k)
            imported.append(item)
        elif choice == "y":
            _import_item(item, k)
            imported.append(item)
        elif choice == "s":
            print(f"Skipping remaining {len(items) - i + 1} items")
            break
        elif choice == "e":
            item = _edit_item(item)
            _import_item(item, k)
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


def _batch_import(items: List[Dict[str, Any]], k: "Kernle", skip_duplicates: bool = False) -> None:
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

            _import_item(item, k)
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


def _import_item(item: Dict[str, Any], k: "Kernle") -> None:
    """Import a single item into Kernle."""
    t = item["type"]

    if t == "episode":
        lessons = [item["lesson"]] if item.get("lesson") else item.get("lessons")
        k.episode(
            objective=item["objective"],
            outcome=item.get("outcome", item["objective"]),
            lessons=lessons,
            tags=item.get("tags"),
        )
    elif t == "note":
        k.note(
            content=item["content"],
            type=item.get("note_type", "note"),
            speaker=item.get("speaker"),
            reason=item.get("reason"),
            tags=item.get("tags"),
        )
    elif t == "belief":
        k.belief(
            statement=item["statement"],
            confidence=item.get("confidence", 0.7),
            type=item.get("belief_type", "fact"),
        )
    elif t == "value":
        k.value(
            name=item["name"],
            description=item.get("description", item["name"]),
            priority=item.get("priority", 50),
        )
    elif t == "goal":
        k.goal(
            description=item.get("description", item.get("title", "")),
            title=item.get("title"),
            status=item.get("status", "active"),
            priority=item.get("priority", "medium"),
        )
    elif t == "raw":
        k.raw(content=item["content"], source=item.get("source", "import"), tags=item.get("tags"))
