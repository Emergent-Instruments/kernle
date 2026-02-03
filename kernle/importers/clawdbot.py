"""Clawdbot/Moltbot migration importer for Kernle.

Migrates memory from Clawdbot workspace structure:
- SOUL.md - Behavioral instructions (mostly skipped, not identity)
- USER.md - User context → Relationships
- MEMORY.md - Curated long-term memory → Mixed types
- memory/*.md - Daily session notes → Episodes, lessons
- AGENTS.md - Boot sequence (keep as-is, don't migrate)
- IDENTITY.md - Core identity → Values, beliefs
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .markdown import ImportItem, parse_markdown

if TYPE_CHECKING:
    from kernle import Kernle


@dataclass
class MigrationResult:
    """Result of a migration operation."""

    items: List[ImportItem] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)  # (item, reason)
    errors: List[str] = field(default_factory=list)
    source_file: str = ""


@dataclass
class ClawdbotMigration:
    """Complete migration plan from Clawdbot workspace."""

    soul: MigrationResult = field(default_factory=MigrationResult)
    user: MigrationResult = field(default_factory=MigrationResult)
    identity: MigrationResult = field(default_factory=MigrationResult)
    memory: MigrationResult = field(default_factory=MigrationResult)
    daily_notes: List[MigrationResult] = field(default_factory=list)

    @property
    def total_items(self) -> int:
        """Total items to import."""
        total = len(self.soul.items) + len(self.user.items) + len(self.identity.items)
        total += len(self.memory.items)
        total += sum(len(d.items) for d in self.daily_notes)
        return total

    @property
    def total_skipped(self) -> int:
        """Total items skipped."""
        total = len(self.soul.skipped) + len(self.user.skipped) + len(self.identity.skipped)
        total += len(self.memory.skipped)
        total += sum(len(d.skipped) for d in self.daily_notes)
        return total

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = ["=== Clawdbot Migration Plan ===\n"]

        if self.soul.items or self.soul.skipped:
            lines.append(f"SOUL.md: {len(self.soul.items)} items, {len(self.soul.skipped)} skipped")
            for item, reason in self.soul.skipped[:3]:
                lines.append(f"  - Skipped: {item[:50]}... ({reason})")

        if self.user.items:
            lines.append(f"USER.md: {len(self.user.items)} items → Relationships")

        if self.identity.items:
            lines.append(f"IDENTITY.md: {len(self.identity.items)} items")

        if self.memory.items or self.memory.skipped:
            lines.append(
                f"MEMORY.md: {len(self.memory.items)} items, {len(self.memory.skipped)} skipped"
            )

        for daily in self.daily_notes:
            if daily.items:
                lines.append(f"{daily.source_file}: {len(daily.items)} items")

        lines.append(f"\nTotal: {self.total_items} to import, {self.total_skipped} skipped")
        return "\n".join(lines)


class ClawdbotImporter:
    """Import memories from a Clawdbot workspace.

    Handles the specific file structure used by Clawdbot/Moltbot agents:
    - ~/clawd/ or ~/.clawdbot/agents/<name>/

    Usage:
        importer = ClawdbotImporter("~/clawd")
        plan = importer.analyze()
        print(plan.summary())
        importer.import_to(kernle_instance, dry_run=True)
    """

    # Files that should NOT be migrated (boot sequence, instructions)
    SKIP_FILES = {"AGENTS.md", "TOOLS.md", "HEARTBEAT.md", "BOOTSTRAP.md"}

    # Content patterns that indicate instructions, not memory
    INSTRUCTION_PATTERNS = [
        r"^be (concise|thorough|careful|helpful)",
        r"^skip the .*(great question|i'd be happy)",
        r"^don't (narrate|send|make up)",
        r"^when (in doubt|you receive|making)",
        r"^this file is yours to",
        r"^if you change this file",
        r"^\*you're not a chatbot",
    ]

    def __init__(self, workspace_path: str, existing_kernle: Optional["Kernle"] = None):
        """Initialize with path to Clawdbot workspace.

        Args:
            workspace_path: Path to workspace (~/clawd or ~/.clawdbot/agents/X)
            existing_kernle: Optional Kernle instance to check for duplicates
        """
        self.workspace = Path(workspace_path).expanduser().resolve()
        self.kernle = existing_kernle
        self._existing_content: set = set()

        if self.kernle:
            self._load_existing_content()

    def _load_existing_content(self) -> None:
        """Load existing content hashes from Kernle for deduplication."""
        if not self.kernle:
            return

        # Load beliefs
        for belief in self.kernle.load_beliefs(limit=100):
            self._existing_content.add(self._normalize(belief.get("statement", "")))

        # Load values
        for value in self.kernle.load_values(limit=50):
            self._existing_content.add(self._normalize(value.get("name", "")))

        # Load episode objectives via storage
        episodes = self.kernle._storage.get_episodes(limit=500)
        for ep in episodes:
            # Episodes may be dataclass or dict depending on storage
            objective = (
                getattr(ep, "objective", "")
                if hasattr(ep, "objective")
                else ep.get("objective", "")
            )
            self._existing_content.add(self._normalize(objective))

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        return re.sub(r"\s+", " ", text.lower().strip())[:100]

    def _is_duplicate(self, text: str) -> bool:
        """Check if content already exists in Kernle."""
        return self._normalize(text) in self._existing_content

    def _is_instruction(self, text: str) -> bool:
        """Check if text is behavioral instruction rather than memory."""
        text_lower = text.lower().strip()
        for pattern in self.INSTRUCTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False

    def analyze(self) -> ClawdbotMigration:
        """Analyze workspace and create migration plan.

        Returns:
            ClawdbotMigration with all items categorized
        """
        plan = ClawdbotMigration()

        # Parse each file type
        plan.soul = self._parse_soul()
        plan.user = self._parse_user()
        plan.identity = self._parse_identity()
        plan.memory = self._parse_memory()
        plan.daily_notes = self._parse_daily_notes()

        return plan

    def _parse_soul(self) -> MigrationResult:
        """Parse SOUL.md — mostly behavioral instructions, limited extraction."""
        result = MigrationResult(source_file="SOUL.md")
        path = self.workspace / "SOUL.md"

        if not path.exists():
            return result

        content = path.read_text(encoding="utf-8")

        # SOUL.md is mostly instructions. Only extract explicit values if marked.
        # Look for ## Values or ## Core Truths sections
        sections = re.split(r"^## (.+)$", content, flags=re.MULTILINE)

        for i in range(1, len(sections), 2):
            if i + 1 >= len(sections):
                break

            header = sections[i].strip().lower()
            section_content = sections[i + 1].strip()

            # Skip most sections — they're instructions
            if any(
                h in header
                for h in ["boundary", "vibe", "continuity", "core truth", "what goes here"]
            ):
                # Extract any **bold** items as potential values
                bold_items = re.findall(r"\*\*([^*]+)\*\*", section_content)
                for item in bold_items:
                    if self._is_instruction(item):
                        result.skipped.append((item, "behavioral instruction"))
                    elif self._is_duplicate(item):
                        result.skipped.append((item, "already in Kernle"))
                    else:
                        # These are likely values or principles
                        result.items.append(
                            ImportItem(
                                type="belief",
                                statement=item,
                                confidence=0.8,
                                source="SOUL.md",
                            )
                        )

        # Overall, most of SOUL.md should be skipped
        if not result.items:
            result.skipped.append(
                ("entire file", "behavioral instructions - keep as boot sequence")
            )

        return result

    def _parse_user(self) -> MigrationResult:
        """Parse USER.md — user context becomes relationships."""
        result = MigrationResult(source_file="USER.md")
        path = self.workspace / "USER.md"

        if not path.exists():
            return result

        content = path.read_text(encoding="utf-8")

        # Extract user name
        name_match = re.search(r"\*\*Name\*\*:\s*(\w+)", content)
        user_name = name_match.group(1) if name_match else "User"

        # Extract relationship info
        role_match = re.search(r"\*\*Role\*\*:\s*(.+?)(?:\n|$)", content)
        role = role_match.group(1).strip() if role_match else ""

        # Extract "What matters" section
        matters_section = re.search(
            r"\*\*What matters to (?:them|him|her)\*\*:(.+?)(?=\n##|\n\*\*|$)",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        interests = []
        if matters_section:
            items = re.findall(r"[-*]\s+(.+)", matters_section.group(1))
            interests = [i.strip() for i in items]

        # Create relationship item
        description = f"Role: {role}" if role else ""
        if interests:
            description += f"\nInterests: {', '.join(interests[:5])}"

        if description:
            result.items.append(
                ImportItem(
                    type="relationship",
                    name=user_name,
                    description=description,
                    source="USER.md",
                    metadata={"role": role, "interests": interests},
                )
            )

        return result

    def _parse_identity(self) -> MigrationResult:
        """Parse IDENTITY.md — core identity becomes values/beliefs."""
        result = MigrationResult(source_file="IDENTITY.md")
        path = self.workspace / "IDENTITY.md"

        if not path.exists():
            return result

        content = path.read_text(encoding="utf-8")

        # Extract key-value pairs like **Name:** Claire
        kvs = re.findall(r"\*\*(\w+)\*\*:\s*(.+?)(?:\n|$)", content)

        for key, value in kvs:
            key_lower = key.lower()
            value = value.strip()

            if key_lower in ("name", "creature", "vibe"):
                # These are identity facts, store as notes
                result.items.append(
                    ImportItem(
                        type="note",
                        content=f"Identity - {key}: {value}",
                        note_type="note",
                        source="IDENTITY.md",
                    )
                )
            elif key_lower in ("emoji", "avatar"):
                # Skip display preferences
                result.skipped.append((f"{key}: {value}", "display preference"))

        # Extract freeform content after the header block
        lines = content.split("\n")
        freeform_start = None
        for i, line in enumerate(lines):
            if line.startswith("---") and i > 5:  # After the header block
                freeform_start = i + 1
                break

        if freeform_start and freeform_start < len(lines):
            freeform = "\n".join(lines[freeform_start:]).strip()
            if freeform and not self._is_instruction(freeform):
                result.items.append(
                    ImportItem(
                        type="raw",
                        content=freeform,
                        source="IDENTITY.md (freeform)",
                    )
                )

        return result

    def _parse_memory(self) -> MigrationResult:
        """Parse MEMORY.md — curated long-term memory, mixed types.

        MEMORY.md typically has custom sections like:
        - ## Origin → episode
        - ## Core Identity → values/beliefs
        - ## [Project Name] → notes
        - ## Technical Learnings → beliefs/notes
        - ## Active Goals → goals
        - ## People → relationships
        """
        result = MigrationResult(source_file="MEMORY.md")
        path = self.workspace / "MEMORY.md"

        if not path.exists():
            return result

        content = path.read_text(encoding="utf-8")

        # Split into sections
        sections = re.split(r"^## (.+)$", content, flags=re.MULTILINE)

        # Process each section based on header content
        for i in range(1, len(sections), 2):
            if i + 1 >= len(sections):
                break

            header = sections[i].strip()
            header_lower = header.lower()
            section_content = sections[i + 1].strip()

            if not section_content:
                continue

            # Skip meta-commentary sections
            if header_lower.startswith("---") or "last updated" in header_lower:
                continue

            # Route to appropriate parser based on header semantics
            if any(h in header_lower for h in ["origin", "birth", "beginning"]):
                # Origin story → episode
                items = self._extract_origin_episode(section_content, header)
            elif any(h in header_lower for h in ["identity", "who i am", "core"]):
                # Identity → values, beliefs
                items = self._extract_identity_items(section_content)
            elif any(h in header_lower for h in ["learning", "technical", "pattern"]):
                # Learnings → beliefs, notes
                items = self._extract_learnings(section_content)
            elif any(h in header_lower for h in ["goal", "objective", "active"]):
                # Goals section
                items = self._extract_goals(section_content)
            elif any(h in header_lower for h in ["people", "relationship", "contact"]):
                # People → relationships
                items = self._extract_relationships(section_content)
            elif any(h in header_lower for h in ["project", "work"]):
                # Project context → notes
                items = [
                    ImportItem(
                        type="note",
                        content=section_content[:500],
                        note_type="context",
                        source=f"MEMORY.md: {header}",
                    )
                ]
            else:
                # Unknown section → try generic parse, fallback to raw
                items = parse_markdown(section_content)
                if not items:
                    items = [
                        ImportItem(
                            type="raw",
                            content=section_content[:500],
                            source=f"MEMORY.md: {header}",
                        )
                    ]

            for item in items:
                key_text = item.statement or item.content or item.objective or item.name
                if not key_text:
                    continue

                if self._is_duplicate(key_text):
                    result.skipped.append((key_text[:50], "already in Kernle"))
                    continue

                if self._is_instruction(key_text):
                    result.skipped.append((key_text[:50], "instruction-like"))
                    continue

                result.items.append(item)

        return result

    def _extract_origin_episode(self, content: str, header: str) -> List[ImportItem]:
        """Extract origin story as episode."""
        # Get first substantive paragraph
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        if not paragraphs:
            return []

        # Origin is typically a founding episode
        return [
            ImportItem(
                type="episode",
                objective="Origin/founding",
                outcome=paragraphs[0][:300],
                source=f"MEMORY.md: {header}",
            )
        ]

    def _extract_identity_items(self, content: str) -> List[ImportItem]:
        """Extract identity facts as values/beliefs/notes."""
        items = []

        # Extract **key**: value patterns
        kvs = re.findall(r"\*\*([^*]+)\*\*:\s*(.+?)(?:\n|$)", content)

        for key, value in kvs:
            key_lower = key.lower()
            value = value.strip()

            if key_lower == "name":
                items.append(
                    ImportItem(
                        type="note",
                        content=f"My name is {value}",
                        note_type="note",
                        source="MEMORY.md",
                    )
                )
            elif key_lower in ("nature", "creature", "what i am"):
                items.append(
                    ImportItem(
                        type="note",
                        content=f"Nature: {value}",
                        note_type="note",
                        source="MEMORY.md",
                    )
                )
            elif key_lower == "philosophy":
                # Philosophy statements are beliefs
                items.append(
                    ImportItem(
                        type="belief",
                        statement=value,
                        confidence=0.8,
                        source="MEMORY.md",
                    )
                )

        return items

    def _extract_learnings(self, content: str) -> List[ImportItem]:
        """Extract technical learnings as beliefs/notes."""
        items = []

        # Look for ### sub-sections
        subsections = re.split(r"^### (.+)$", content, flags=re.MULTILINE)

        # Process sub-sections
        for i in range(1, len(subsections), 2):
            if i + 1 >= len(subsections):
                break

            subheader = subsections[i].strip()
            subcontent = subsections[i + 1].strip()

            # Extract bullet points as individual learnings
            bullets = re.findall(r"^[-*]\s+(.+)$", subcontent, re.MULTILINE)

            for bullet in bullets:
                # Pattern-like statements become beliefs
                if any(
                    p in bullet.lower()
                    for p in [
                        "works well",
                        "is effective",
                        "are effective",
                        "should",
                        "need",
                        "require",
                    ]
                ):
                    items.append(
                        ImportItem(
                            type="belief",
                            statement=bullet,
                            confidence=0.75,
                            source=f"MEMORY.md: {subheader}",
                        )
                    )
                else:
                    # Facts become notes
                    items.append(
                        ImportItem(
                            type="note",
                            content=bullet,
                            note_type="note",
                            source=f"MEMORY.md: {subheader}",
                        )
                    )

        # If no sub-sections, process as flat content
        if len(subsections) <= 1:
            bullets = re.findall(r"^[-*]\s+(.+)$", content, re.MULTILINE)
            for bullet in bullets:
                items.append(
                    ImportItem(
                        type="note",
                        content=bullet,
                        note_type="insight",  # Technical learnings are insights
                        source="MEMORY.md: Learnings",
                    )
                )

        return items

    def _extract_goals(self, content: str) -> List[ImportItem]:
        """Extract goals from content."""
        items = []

        # Look for numbered or bulleted items
        goals = re.findall(r"^\d+\.\s+\*\*(.+?)\*\*\s*[-–—]?\s*(.*)$", content, re.MULTILINE)

        for name, desc in goals:
            items.append(
                ImportItem(
                    type="goal",
                    description=f"{name}: {desc}".strip(": "),
                    status="active",
                    source="MEMORY.md: Goals",
                )
            )

        # Fallback to simple bullets
        if not goals:
            bullets = re.findall(r"^[-*]\s+(.+)$", content, re.MULTILINE)
            for bullet in bullets:
                items.append(
                    ImportItem(
                        type="goal", description=bullet, status="active", source="MEMORY.md: Goals"
                    )
                )

        return items

    def _extract_relationships(self, content: str) -> List[ImportItem]:
        """Extract people/relationships from content."""
        items = []

        # Look for **Name**: description pattern
        people = re.findall(r"\*\*([^*]+)\*\*:\s*(.+?)(?=\n\*\*|\n\n|$)", content, re.DOTALL)

        for name, desc in people:
            name = name.strip()
            desc = desc.strip().replace("\n", " ")[:200]
            items.append(
                ImportItem(
                    type="relationship",
                    name=name,
                    description=desc,
                    source="MEMORY.md: People",
                )
            )

        return items

    def _parse_daily_notes(self) -> List[MigrationResult]:
        """Parse memory/*.md daily notes — session notes become episodes."""
        results = []
        memory_dir = self.workspace / "memory"

        if not memory_dir.exists():
            return results

        for md_file in sorted(memory_dir.glob("*.md")):
            result = MigrationResult(source_file=f"memory/{md_file.name}")

            # Extract date from filename (YYYY-MM-DD or YYYY-MM-DD-slug)
            date_match = re.match(r"(\d{4}-\d{2}-\d{2})", md_file.name)
            file_date = date_match.group(1) if date_match else None

            content = md_file.read_text(encoding="utf-8")

            # Parse using markdown importer
            items = parse_markdown(content)

            for item in items:
                key_text = item.statement or item.content or item.objective or item.description

                # Check duplicates
                if self._is_duplicate(key_text):
                    result.skipped.append((key_text[:50], "already in Kernle"))
                    continue

                # Add date metadata
                if file_date:
                    item.metadata["date"] = file_date

                result.items.append(item)

            if result.items or result.skipped:
                results.append(result)

        return results

    def import_to(
        self, k: "Kernle", dry_run: bool = False, interactive: bool = False
    ) -> Dict[str, Any]:
        """Execute migration into Kernle instance.

        Args:
            k: Kernle instance to import into
            dry_run: If True, don't actually import
            interactive: If True, prompt for each item

        Returns:
            Dict with import statistics
        """
        plan = self.analyze()

        stats = {
            "imported": 0,
            "skipped": plan.total_skipped,
            "errors": [],
            "by_type": {},
        }

        all_results = [plan.soul, plan.user, plan.identity, plan.memory] + plan.daily_notes

        for result in all_results:
            for item in result.items:
                if interactive:
                    print(f"\n{item.type}: {item.content or item.statement or item.objective}")
                    response = input("Import? [y/N/q]: ").strip().lower()
                    if response == "q":
                        return stats
                    if response != "y":
                        stats["skipped"] += 1
                        continue

                if not dry_run:
                    try:
                        _import_clawdbot_item(item, k)
                        stats["imported"] += 1
                        stats["by_type"][item.type] = stats["by_type"].get(item.type, 0) + 1
                    except Exception as e:
                        stats["errors"].append(f"{item.type}: {str(e)[:50]}")
                else:
                    stats["imported"] += 1
                    stats["by_type"][item.type] = stats["by_type"].get(item.type, 0) + 1

        return stats


def _import_clawdbot_item(item: ImportItem, k: "Kernle") -> Optional[str]:
    """Import a single Clawdbot item into Kernle.

    Handles the 'relationship' type which isn't in base markdown importer.
    """
    t = item.type

    if t == "relationship":
        # Kernle relationship() expects: name, context, trust, expertise
        return k.relationship(
            name=item.name,
            context=item.description,
            trust=0.7,  # Default trust
            expertise=item.metadata.get("interests", []),
        )
    elif t == "episode":
        lessons = [item.lesson] if item.lesson else None
        return k.episode(
            objective=item.objective,
            outcome=item.outcome or item.objective,
            lessons=lessons,
            # outcome_type not supported in Kernle API
        )
    elif t == "note":
        return k.note(content=item.content, type=item.note_type)
    elif t == "belief":
        return k.belief(statement=item.statement, confidence=item.confidence)
    elif t == "value":
        return k.value(name=item.name, description=item.description, priority=item.priority)
    elif t == "goal":
        return k.goal(
            description=item.description,
            status=item.status,
            priority=item.metadata.get("priority", "medium"),
        )
    elif t == "raw":
        return k.raw(item.content)

    return None
