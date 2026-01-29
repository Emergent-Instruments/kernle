"""CSV importer for Kernle.

Imports from CSV files with columns for memory fields.
Supports bulk import of memories in tabular format.
"""

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from kernle import Kernle


# Column name mappings for each memory type
COLUMN_MAPPINGS = {
    "episode": {
        "objective": ["objective", "title", "task", "name"],
        "outcome": ["outcome", "result", "description", "desc"],
        "outcome_type": ["outcome_type", "type", "status", "result_type"],
        "lessons": ["lessons", "lesson", "learnings", "learning"],
        "tags": ["tags", "tag", "labels", "label"],
    },
    "note": {
        "content": ["content", "text", "note", "body", "description"],
        "type": ["type", "note_type", "category", "kind"],
        "speaker": ["speaker", "author", "from", "source"],
        "reason": ["reason", "why", "context"],
        "tags": ["tags", "tag", "labels", "label"],
    },
    "belief": {
        "statement": ["statement", "belief", "content", "text", "description"],
        "confidence": ["confidence", "conf", "certainty", "probability"],
        "type": ["type", "belief_type", "category", "kind"],
    },
    "value": {
        "name": ["name", "value", "title"],
        "description": ["description", "desc", "statement", "content", "text"],
        "priority": ["priority", "importance", "weight", "rank"],
    },
    "goal": {
        "title": ["title", "goal", "name", "objective"],
        "description": ["description", "desc", "details", "content"],
        "priority": ["priority", "importance", "urgency"],
        "status": ["status", "state", "complete", "completed"],
    },
    "raw": {
        "content": ["content", "text", "raw", "body", "data"],
        "source": ["source", "from", "origin"],
        "tags": ["tags", "tag", "labels", "label"],
    },
}


@dataclass
class CsvImportItem:
    """A parsed CSV row ready for import."""

    type: str
    data: Dict[str, Any] = field(default_factory=dict)


class CsvImporter:
    """Import memories from CSV files.

    The CSV file must have a 'type' column indicating the memory type
    (episode, note, belief, value, goal, raw) and columns matching
    the expected fields for each type.

    Example CSV:
    ```csv
    type,content,confidence
    belief,"Testing is important",0.9
    belief,"Code should be readable",0.85
    note,"Review the authentication system",
    ```
    """

    def __init__(self, file_path: str, memory_type: Optional[str] = None):
        """Initialize with path to CSV file.

        Args:
            file_path: Path to the CSV file to import
            memory_type: If set, treat all rows as this type (overrides 'type' column)
        """
        self.file_path = Path(file_path).expanduser()
        self.items: List[CsvImportItem] = []
        self.memory_type = memory_type

    def parse(self) -> List[CsvImportItem]:
        """Parse the CSV file and return importable items.

        Returns:
            List of CsvImportItem objects

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If required columns are missing
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        content = self.file_path.read_text(encoding="utf-8")
        self.items = parse_csv(content, self.memory_type)
        return self.items

    def import_to(
        self, k: "Kernle", dry_run: bool = False, skip_duplicates: bool = True
    ) -> Dict[str, Any]:
        """Import parsed items into a Kernle instance.

        Args:
            k: Kernle instance to import into
            dry_run: If True, don't actually import, just return counts
            skip_duplicates: If True, skip items that already exist

        Returns:
            Dict with counts of items imported by type and any errors
        """
        if not self.items:
            self.parse()

        counts: Dict[str, int] = {}
        skipped: Dict[str, int] = {}
        errors: List[str] = []

        for i, item in enumerate(self.items):
            try:
                if not dry_run:
                    imported = _import_csv_item(item, k, skip_duplicates)
                    if imported:
                        counts[item.type] = counts.get(item.type, 0) + 1
                    else:
                        skipped[item.type] = skipped.get(item.type, 0) + 1
                else:
                    counts[item.type] = counts.get(item.type, 0) + 1
            except Exception as e:
                errors.append(f"Row {i + 2}: {item.type}: {str(e)[:50]}")

        return {
            "imported": counts,
            "skipped": skipped,
            "errors": errors,
        }


def parse_csv(content: str, memory_type: Optional[str] = None) -> List[CsvImportItem]:
    """Parse CSV content into importable items.

    Args:
        content: CSV content string
        memory_type: If set, treat all rows as this type

    Returns:
        List of CsvImportItem objects

    Raises:
        ValueError: If the CSV format is invalid
    """
    items: List[CsvImportItem] = []

    reader = csv.DictReader(io.StringIO(content))
    headers = [h.lower().strip() for h in (reader.fieldnames or [])]

    if not headers:
        raise ValueError("CSV file has no headers")

    # Check if 'type' column exists
    has_type_column = any(h in ["type", "memory_type", "kind"] for h in headers)

    if not has_type_column and not memory_type:
        raise ValueError(
            "CSV must have a 'type' column or specify --type when importing. "
            "Valid types: episode, note, belief, value, goal, raw"
        )

    for row in reader:
        # Normalize keys to lowercase
        row = {k.lower().strip(): v.strip() if v else "" for k, v in row.items()}

        # Determine memory type
        if memory_type:
            item_type = memory_type
        else:
            item_type = row.get("type") or row.get("memory_type") or row.get("kind")
            if not item_type:
                continue  # Skip rows without type

        item_type = item_type.lower().strip()

        # Validate type
        if item_type not in COLUMN_MAPPINGS:
            continue  # Skip unknown types

        # Map columns to expected field names
        data = _map_columns(row, item_type)

        # Skip empty rows
        if not any(data.values()):
            continue

        items.append(CsvImportItem(type=item_type, data=data))

    return items


def _map_columns(row: Dict[str, str], memory_type: str) -> Dict[str, Any]:
    """Map CSV columns to memory field names.

    Args:
        row: The CSV row dict
        memory_type: The type of memory

    Returns:
        Dict with normalized field names
    """
    mappings = COLUMN_MAPPINGS.get(memory_type, {})
    result: Dict[str, Any] = {}

    for field_name, aliases in mappings.items():
        for alias in aliases:
            if alias in row and row[alias]:
                value = row[alias]

                # Type conversions
                if field_name == "confidence":
                    try:
                        value = float(value)
                        if value > 1:
                            value = value / 100
                    except ValueError:
                        value = 0.7
                elif field_name == "priority" and memory_type == "value":
                    try:
                        value = int(value)
                    except ValueError:
                        value = 50
                elif field_name in ("tags", "lessons"):
                    # Split comma-separated values
                    value = [v.strip() for v in value.split(",") if v.strip()]

                result[field_name] = value
                break

    return result


def _import_csv_item(item: CsvImportItem, k: "Kernle", skip_duplicates: bool = True) -> bool:
    """Import a single CSV item into Kernle.

    Args:
        item: The CsvImportItem to import
        k: Kernle instance
        skip_duplicates: If True, skip items that already exist

    Returns:
        True if imported, False if skipped
    """
    t = item.type
    data = item.data

    if t == "episode":
        objective = data.get("objective", "")
        if not objective:
            return False

        if skip_duplicates:
            existing = k.search(objective, limit=5, record_types=["episode"])
            for result in existing:
                if hasattr(result.record, "objective"):
                    if result.record.objective == objective:
                        return False

        k.episode(
            objective=objective,
            outcome=data.get("outcome", objective),
            outcome_type=data.get("outcome_type"),
            lessons=data.get("lessons"),
            tags=data.get("tags"),
        )
        return True

    elif t == "note":
        content = data.get("content", "")
        if not content:
            return False

        if skip_duplicates:
            existing = k.search(content[:100], limit=5, record_types=["note"])
            for result in existing:
                if hasattr(result.record, "content"):
                    if result.record.content == content:
                        return False

        k.note(
            content=content,
            type=data.get("type", "note"),
            speaker=data.get("speaker"),
            reason=data.get("reason"),
            tags=data.get("tags"),
        )
        return True

    elif t == "belief":
        statement = data.get("statement", "")
        if not statement:
            return False

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
        if not name:
            return False

        if skip_duplicates:
            existing = k._storage.get_values(limit=100)
            for v in existing:
                if v.name == name:
                    return False

        k.value(
            name=name,
            description=data.get("description", name),
            priority=data.get("priority", 50),
        )
        return True

    elif t == "goal":
        title = data.get("title", "")
        description = data.get("description", title)
        if not title and not description:
            return False

        if skip_duplicates:
            existing = k._storage.get_goals(status=None, limit=100)
            for g in existing:
                if g.title == title or g.description == description:
                    return False

        # Map status values
        status = data.get("status", "active")
        if status.lower() in ("done", "complete", "completed", "true", "1", "yes"):
            status = "completed"
        elif status.lower() in ("paused", "hold", "on hold"):
            status = "paused"
        else:
            status = "active"

        k.goal(
            description=description,
            title=title,
            priority=data.get("priority", "medium"),
            status=status,
        )
        return True

    elif t == "raw":
        content = data.get("content", "")
        if not content:
            return False

        if skip_duplicates:
            existing = k._storage.list_raw(limit=100)
            for r in existing:
                if r.content == content:
                    return False

        k.raw(
            content=content,
            source=data.get("source", "csv-import"),
            tags=data.get("tags"),
        )
        return True

    return False
