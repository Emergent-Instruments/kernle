"""Importers for migrating data into Kernle from various formats.

Supported formats:
- Markdown: File-based memory like MEMORY.md with sections for beliefs, episodes, etc.
- JSON: Kernle export format (from `kernle export --format json`)
- CSV: Simple tabular format for bulk import
- Clawdbot: Migration from Clawdbot/Moltbot workspaces (SOUL.md, MEMORY.md, memory/*.md)
"""

from kernle.importers.clawdbot import ClawdbotImporter, ClawdbotMigration
from kernle.importers.csv_importer import CsvImporter, parse_csv
from kernle.importers.json_importer import JsonImporter, parse_kernle_json
from kernle.importers.markdown import MarkdownImporter, parse_markdown

__all__ = [
    "MarkdownImporter",
    "JsonImporter",
    "CsvImporter",
    "ClawdbotImporter",
    "ClawdbotMigration",
    "parse_markdown",
    "parse_kernle_json",
    "parse_csv",
]
