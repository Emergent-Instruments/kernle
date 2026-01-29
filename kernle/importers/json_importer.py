"""JSON importer for Kernle.

Imports from the JSON format exported by `kernle export --format json`.
This preserves metadata like confidence, timestamps, and relationships.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from kernle import Kernle


@dataclass
class JsonImportItem:
    """A parsed JSON item ready for import."""

    type: str  # episode, note, belief, value, goal, drive, relationship, raw
    data: Dict[str, Any] = field(default_factory=dict)


class JsonImporter:
    """Import memories from Kernle JSON export files.

    Supports the JSON format from `kernle export --format json` and
    `kernle dump --format json`.
    """

    def __init__(self, file_path: str):
        """Initialize with path to JSON file.

        Args:
            file_path: Path to the JSON file to import
        """
        self.file_path = Path(file_path).expanduser()
        self.items: List[JsonImportItem] = []
        self.source_agent_id: Optional[str] = None

    def parse(self) -> List[JsonImportItem]:
        """Parse the JSON file and return importable items.

        Returns:
            List of JsonImportItem objects

        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file is not valid JSON
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        content = self.file_path.read_text(encoding="utf-8")
        self.items, self.source_agent_id = parse_kernle_json(content)
        return self.items

    def import_to(
        self, k: "Kernle", dry_run: bool = False, skip_duplicates: bool = True
    ) -> Dict[str, Any]:
        """Import parsed items into a Kernle instance.

        Args:
            k: Kernle instance to import into
            dry_run: If True, don't actually import, just return counts
            skip_duplicates: If True, skip items that already exist (by content)

        Returns:
            Dict with counts of items imported by type and any errors
        """
        if not self.items:
            self.parse()

        counts: Dict[str, int] = {}
        skipped: Dict[str, int] = {}
        errors: List[str] = []

        for item in self.items:
            try:
                if not dry_run:
                    imported = _import_json_item(item, k, skip_duplicates)
                    if imported:
                        counts[item.type] = counts.get(item.type, 0) + 1
                    else:
                        skipped[item.type] = skipped.get(item.type, 0) + 1
                else:
                    counts[item.type] = counts.get(item.type, 0) + 1
            except Exception as e:
                errors.append(f"{item.type}: {str(e)[:50]}")

        return {
            "imported": counts,
            "skipped": skipped,
            "errors": errors,
            "source_agent_id": self.source_agent_id,
        }


def parse_kernle_json(content: str) -> tuple[List[JsonImportItem], Optional[str]]:
    """Parse Kernle JSON export format.

    Expected format:
    {
        "agent_id": "...",
        "exported_at": "...",
        "values": [...],
        "beliefs": [...],
        "goals": [...],
        "episodes": [...],
        "notes": [...],
        "drives": [...],
        "relationships": [...],
        "raw_entries": [...]  # optional
    }

    Args:
        content: JSON content string

    Returns:
        Tuple of (list of JsonImportItem, source agent_id)

    Raises:
        json.JSONDecodeError: If content is not valid JSON
        ValueError: If the format doesn't match expected structure
    """
    data = json.loads(content)

    # Validate it looks like a Kernle export
    if not isinstance(data, dict):
        raise ValueError("JSON must be an object at the root level")

    agent_id = data.get("agent_id")
    items: List[JsonImportItem] = []

    # Parse each memory type
    for value_data in data.get("values", []):
        items.append(JsonImportItem(type="value", data=value_data))

    for belief_data in data.get("beliefs", []):
        items.append(JsonImportItem(type="belief", data=belief_data))

    for goal_data in data.get("goals", []):
        items.append(JsonImportItem(type="goal", data=goal_data))

    for episode_data in data.get("episodes", []):
        items.append(JsonImportItem(type="episode", data=episode_data))

    for note_data in data.get("notes", []):
        items.append(JsonImportItem(type="note", data=note_data))

    for drive_data in data.get("drives", []):
        items.append(JsonImportItem(type="drive", data=drive_data))

    for rel_data in data.get("relationships", []):
        items.append(JsonImportItem(type="relationship", data=rel_data))

    for raw_data in data.get("raw_entries", []):
        items.append(JsonImportItem(type="raw", data=raw_data))

    return items, agent_id


def _import_json_item(item: JsonImportItem, k: "Kernle", skip_duplicates: bool = True) -> bool:
    """Import a single JSON item into Kernle.

    Args:
        item: The JsonImportItem to import
        k: Kernle instance
        skip_duplicates: If True, skip items that already exist

    Returns:
        True if imported, False if skipped
    """
    t = item.type
    data = item.data

    if t == "episode":
        # Check for duplicate by objective + outcome
        if skip_duplicates:
            existing = k.search(data.get("objective", ""), limit=5, record_types=["episode"])
            for result in existing:
                if hasattr(result.record, "objective") and hasattr(result.record, "outcome"):
                    if result.record.objective == data.get(
                        "objective"
                    ) and result.record.outcome == data.get("outcome"):
                        return False

        k.episode(
            objective=data.get("objective", ""),
            outcome=data.get("outcome", ""),
            outcome_type=data.get("outcome_type"),
            lessons=data.get("lessons"),
            tags=data.get("tags"),
            emotional_valence=data.get("emotional_valence", 0.0),
            emotional_arousal=data.get("emotional_arousal", 0.0),
            emotional_tags=data.get("emotional_tags"),
        )
        return True

    elif t == "note":
        if skip_duplicates:
            existing = k.search(data.get("content", "")[:100], limit=5, record_types=["note"])
            for result in existing:
                if hasattr(result.record, "content"):
                    if result.record.content == data.get("content"):
                        return False

        k.note(
            content=data.get("content", ""),
            type=data.get("type", "note"),
            speaker=data.get("speaker"),
            reason=data.get("reason"),
            tags=data.get("tags"),
        )
        return True

    elif t == "belief":
        statement = data.get("statement", "")
        if skip_duplicates:
            existing = k._storage.find_belief(statement)
            if existing:
                return False

        k.belief(
            statement=statement,
            type=data.get("type", "fact"),
            confidence=data.get("confidence", 0.8),
        )
        return True

    elif t == "value":
        name = data.get("name", "")
        if skip_duplicates:
            existing = k._storage.get_values(limit=100)
            for v in existing:
                if v.name == name:
                    return False

        k.value(
            name=name,
            description=data.get("statement", data.get("description", name)),
            priority=data.get("priority", 50),
        )
        return True

    elif t == "goal":
        description = data.get("description") or data.get("title", "")
        if skip_duplicates:
            existing = k._storage.get_goals(status=None, limit=100)
            for g in existing:
                if g.title == data.get("title") or g.description == description:
                    return False

        k.goal(
            description=description,
            title=data.get("title"),
            priority=data.get("priority", "medium"),
            status=data.get("status", "active"),
        )
        return True

    elif t == "drive":
        drive_type = data.get("drive_type", "")
        if skip_duplicates:
            existing = k._storage.get_drive(drive_type)
            if existing:
                return False

        k.drive(
            drive_type=drive_type,
            intensity=data.get("intensity", 0.5),
            focus=data.get("focus_areas"),
        )
        return True

    elif t == "relationship":
        entity_name = data.get("entity_name", "")
        if skip_duplicates:
            existing = k._storage.get_relationship(entity_name)
            if existing:
                return False

        k.relationship(
            entity_name=entity_name,
            entity_type=data.get("entity_type", "unknown"),
            relationship_type=data.get("relationship_type", "knows"),
            sentiment=data.get("sentiment", 0.0),
            notes=data.get("notes"),
        )
        return True

    elif t == "raw":
        content = data.get("content", "")
        if skip_duplicates:
            existing = k._storage.list_raw(limit=100)
            for r in existing:
                if r.content == content:
                    return False

        k.raw(
            content=content,
            source=data.get("source", "import"),
            tags=data.get("tags"),
        )
        return True

    return False
