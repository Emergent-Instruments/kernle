"""Migration commands for Kernle CLI — seed beliefs, backfill provenance, link raw."""

import hashlib
import json as _json
import logging
from typing import TYPE_CHECKING, Dict, List

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import argparse

    from kernle import Kernle


def cmd_migrate(args: "argparse.Namespace", k: "Kernle") -> None:
    """Migrate from other platforms to Kernle.

    Supports:
    - seed-beliefs: Add foundational beliefs to existing agents
    - backfill-provenance: Add provenance metadata to existing memories
    - link-raw: Link pre-provenance episodes/notes to raw entries
    """
    action = getattr(args, "migrate_action", None)

    if action == "seed-beliefs":
        _migrate_seed_beliefs(args, k)
    elif action == "backfill-provenance":
        _migrate_backfill_provenance(args, k)
    elif action == "link-raw":
        _migrate_link_raw(args, k)
    else:
        print(f"Unknown migrate action: {action}")
        print("Available actions: seed-beliefs, backfill-provenance, link-raw")


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


def _snapshot_digest(payload):
    """Build a deterministic hash for migration payload snapshots."""
    return hashlib.sha256(
        _json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


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
            print(f"\nFrom roundtable synthesis for seed beliefs (v{SEED_BELIEFS_VERSION})")
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
                type="assumption",
                foundational=True,
                source_type="seed",
                context="kernle_seed",
                context_tags=belief.get("tags"),
                source="seed belief from kernle roundtable synthesis",
                derived_from=[f"context:kernle_seed_v{SEED_BELIEFS_VERSION}"],
            )
            added += 1
            tier_label = f"[Tier {belief['tier']}]" if belief["tier"] > 0 else "[Meta]"
            print(f"  ✓ {tier_label} {belief['statement'][:50]}...")
        except Exception as e:
            logger.debug("Seed belief failed: %s", e)
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
    dry_run = getattr(args, "dry_run", False)
    json_output = getattr(args, "json", False)

    # Seed belief statements for identification
    seed_statements = {b["statement"] for b in _FULL_SEED_BELIEFS}
    seed_statements.update(b["statement"] for b in _MINIMAL_SEED_BELIEFS)

    updates = []

    def _snapshot_metadata(record):
        """Build a lightweight provenance snapshot for migration diffs."""
        return {
            "source_type": getattr(record, "source_type", None),
            "derived_from": list(getattr(record, "derived_from", None) or []),
        }

    def _collect_update(
        type_name,
        record,
        old_source_type,
        new_source_type,
        old_derived_from,
        new_derived_from,
    ):
        return {
            "type": type_name,
            "id": record.id,
            "summary": getattr(record, "statement", None)
            or getattr(record, "objective", None)
            or getattr(record, "content", None)
            or getattr(record, "name", None)
            or getattr(record, "title", None)
            or getattr(record, "drive_type", None)
            or getattr(record, "entity_name", None)
            or "unknown",
            "old_source_type": old_source_type,
            "new_source_type": new_source_type,
            "old_derived_from": old_derived_from,
            "new_derived_from": new_derived_from,
            "pre_image": _snapshot_metadata(record),
            "post_image": {
                "source_type": new_source_type,
                "derived_from": list(new_derived_from or []),
            },
        }

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
        elif source_type == "processed":
            # Legacy "processed" value — migrate to canonical "processing"
            new_source_type = "processing"
            needs_update = True
        elif not source_type or source_type in ("unknown", ""):
            # Non-seed belief with no source — mark as direct_experience
            new_source_type = "direct_experience"
            needs_update = True

        if needs_update:
            update = _collect_update(
                "belief",
                belief,
                source_type,
                new_source_type,
                derived_from,
                new_derived_from,
            )
            update["summary"] = belief.statement[:60]
            updates.append(update)

    # Scan episodes
    episodes = k._storage.get_episodes(limit=1000)
    for ep in episodes:
        source_type = getattr(ep, "source_type", None)
        derived_from = getattr(ep, "derived_from", None)

        needs_update = False
        new_source_type = source_type
        new_derived_from = derived_from or []

        if source_type == "processed":
            new_source_type = "processing"
            needs_update = True
        elif not source_type or source_type in ("unknown", ""):
            new_source_type = "direct_experience"
            needs_update = True

        # Mark episodes with no derived_from as pre-v0.9 migrations
        if not derived_from or derived_from == []:
            if not any(d.startswith("kernle:pre-v0.9") for d in (new_derived_from or [])):
                new_derived_from = list(new_derived_from or []) + ["kernle:pre-v0.9-migration"]
                needs_update = True

        if needs_update:
            update = _collect_update(
                "episode",
                ep,
                source_type,
                new_source_type,
                derived_from,
                new_derived_from if new_derived_from else None,
            )
            update["summary"] = (ep.objective or "")[:60]
            updates.append(update)

    # Scan notes
    notes = k._storage.get_notes(limit=1000)
    for note in notes:
        source_type = getattr(note, "source_type", None)
        derived_from = getattr(note, "derived_from", None)

        needs_update = False
        new_source_type = source_type
        new_derived_from = derived_from or []

        if source_type == "processed":
            new_source_type = "processing"
            needs_update = True
        elif not source_type or source_type in ("unknown", ""):
            new_source_type = "direct_experience"
            needs_update = True

        # Mark notes with no derived_from as pre-v0.9 migrations
        if not derived_from or derived_from == []:
            if not any(d.startswith("kernle:pre-v0.9") for d in (new_derived_from or [])):
                new_derived_from = list(new_derived_from or []) + ["kernle:pre-v0.9-migration"]
                needs_update = True

        if needs_update:
            update = _collect_update(
                "note",
                note,
                source_type,
                new_source_type,
                derived_from,
                new_derived_from if new_derived_from else None,
            )
            update["summary"] = (note.content or "")[:60]
            updates.append(update)

    # Scan values
    values = k._storage.get_values(limit=1000)
    for value in values:
        source_type = getattr(value, "source_type", None)
        derived_from = getattr(value, "derived_from", None)

        needs_update = False
        new_source_type = source_type
        new_derived_from = derived_from or []

        if source_type == "processed":
            new_source_type = "processing"
            needs_update = True
        elif not source_type or source_type in ("unknown", ""):
            new_source_type = "direct_experience"
            needs_update = True

        if not derived_from or derived_from == []:
            if not any(d.startswith("kernle:pre-v0.9") for d in (new_derived_from or [])):
                new_derived_from = list(new_derived_from or []) + ["kernle:pre-v0.9-migration"]
                needs_update = True

        if needs_update:
            update = _collect_update(
                "value",
                value,
                source_type,
                new_source_type,
                derived_from,
                new_derived_from if new_derived_from else None,
            )
            update["summary"] = (value.name or "")[:60]
            updates.append(update)

    # Scan goals (status=None to include all statuses)
    goals = k._storage.get_goals(status=None, limit=1000)
    for goal in goals:
        source_type = getattr(goal, "source_type", None)
        derived_from = getattr(goal, "derived_from", None)

        needs_update = False
        new_source_type = source_type
        new_derived_from = derived_from or []

        if source_type == "processed":
            new_source_type = "processing"
            needs_update = True
        elif not source_type or source_type in ("unknown", ""):
            new_source_type = "direct_experience"
            needs_update = True

        if not derived_from or derived_from == []:
            if not any(d.startswith("kernle:pre-v0.9") for d in (new_derived_from or [])):
                new_derived_from = list(new_derived_from or []) + ["kernle:pre-v0.9-migration"]
                needs_update = True

        if needs_update:
            update = _collect_update(
                "goal",
                goal,
                source_type,
                new_source_type,
                derived_from,
                new_derived_from if new_derived_from else None,
            )
            update["summary"] = (goal.title or "")[:60]
            updates.append(update)

    # Scan drives
    drives = k._storage.get_drives()
    for drive in drives:
        source_type = getattr(drive, "source_type", None)
        derived_from = getattr(drive, "derived_from", None)

        needs_update = False
        new_source_type = source_type
        new_derived_from = derived_from or []

        if source_type == "processed":
            new_source_type = "processing"
            needs_update = True
        elif not source_type or source_type in ("unknown", ""):
            new_source_type = "direct_experience"
            needs_update = True

        if not derived_from or derived_from == []:
            if not any(d.startswith("kernle:pre-v0.9") for d in (new_derived_from or [])):
                new_derived_from = list(new_derived_from or []) + ["kernle:pre-v0.9-migration"]
                needs_update = True

        if needs_update:
            update = _collect_update(
                "drive",
                drive,
                source_type,
                new_source_type,
                derived_from,
                new_derived_from if new_derived_from else None,
            )
            update["summary"] = (drive.drive_type or "")[:60]
            updates.append(update)

    # Scan relationships
    relationships = k._storage.get_relationships()
    for rel in relationships:
        source_type = getattr(rel, "source_type", None)
        derived_from = getattr(rel, "derived_from", None)

        needs_update = False
        new_source_type = source_type
        new_derived_from = derived_from or []

        if source_type == "processed":
            new_source_type = "processing"
            needs_update = True
        elif not source_type or source_type in ("unknown", ""):
            new_source_type = "direct_experience"
            needs_update = True

        if not derived_from or derived_from == []:
            if not any(d.startswith("kernle:pre-v0.9") for d in (new_derived_from or [])):
                new_derived_from = list(new_derived_from or []) + ["kernle:pre-v0.9-migration"]
                needs_update = True

        if needs_update:
            update = _collect_update(
                "relationship",
                rel,
                source_type,
                new_source_type,
                derived_from,
                new_derived_from if new_derived_from else None,
            )
            update["summary"] = (rel.entity_name or "")[:60]
            updates.append(update)

    # Output results
    updates = sorted(
        updates,
        key=lambda item: (
            item.get("type", ""),
            item.get("id", ""),
            str(item.get("old_source_type") or ""),
            str(item.get("new_source_type") or ""),
        ),
    )

    by_type: Dict[str, int] = {}
    for u in updates:
        by_type[u["type"]] = by_type.get(u["type"], 0) + 1

    if json_output:
        status_snapshot = {
            "total_updates": len(updates),
            "by_type": dict(sorted(by_type.items())),
            "ids": [u["id"] for u in updates],
            "types": sorted(by_type.keys()),
        }
        snapshot = {
            "dry_run": dry_run,
            "total_updates": len(updates),
            "updates": updates,
            "status_snapshot": status_snapshot,
            "status_snapshot_sha256": _snapshot_digest(status_snapshot),
        }
        print(
            _json.dumps(
                snapshot,
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
            logger.debug("Provenance backfill failed for %s:%s: %s", u["type"], u["id"][:8], e)
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
    from datetime import datetime, timedelta, timezone

    dry_run = getattr(args, "dry_run", False)
    json_output = getattr(args, "json", False)
    window_minutes = getattr(args, "window", 30)
    link_all = getattr(args, "link_all", False)

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
        stop = {
            "the",
            "a",
            "an",
            "is",
            "was",
            "are",
            "were",
            "be",
            "been",
            "to",
            "of",
            "and",
            "in",
            "on",
            "at",
            "for",
            "with",
            "by",
            "from",
            "it",
            "this",
            "that",
            "not",
            "but",
            "or",
            "as",
            "if",
        }
        words_a -= stop
        words_b -= stop
        overlap = words_a & words_b
        return len(overlap) >= min_words

    def _snapshot_memory(record):
        source_type = getattr(record, "source_type", None)
        if hasattr(source_type, "value"):
            source_type = source_type.value
        return {
            "source_type": source_type,
            "derived_from": list(getattr(record, "derived_from", None) or []),
        }

    def _memory_post_image(record, derived_from, raw_id=None):
        if record:
            post = _snapshot_memory(record)
        else:
            post = {"source_type": None, "derived_from": []}
        post["derived_from"] = list(derived_from or [])
        post["raw_id"] = raw_id
        return post

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
    links: List[Dict] = []
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
                    ref
                    for ref in (derived_from or [])
                    if ref and ":" in ref and ref.split(":", 1)[0] in annotation_prefixes
                ]
                new_derived_from = [
                    f"raw:{best_match['id']}",
                    "kernle:auto-linked",
                ] + existing_annotations
                pre_image = _snapshot_memory(record)

                links.append(
                    {
                        "type": memory_type,
                        "id": record.id,
                        "summary": match_text[:60],
                        "raw_id": best_match["id"],
                        "raw_blob": best_match["blob"][:60],
                        "score": round(best_score, 2),
                        "old_derived_from": derived_from,
                        "new_derived_from": new_derived_from,
                        "pre_image": pre_image,
                        "post_image": _memory_post_image(
                            record,
                            new_derived_from,
                            f"raw:{best_match['id']}",
                        ),
                        "synthetic": False,
                    }
                )
            elif link_all and match_text.strip():
                # --all: create a synthetic raw entry for unmatched memories
                existing_annotations = [
                    ref
                    for ref in (derived_from or [])
                    if ref and ":" in ref and ref.split(":", 1)[0] in annotation_prefixes
                ]
                pre_image = _snapshot_memory(record)
                links.append(
                    {
                        "type": memory_type,
                        "id": record.id,
                        "summary": match_text[:60],
                        "raw_id": None,
                        "raw_blob": None,
                        "score": 0.0,
                        "old_derived_from": derived_from,
                        "new_derived_from": None,  # filled at apply time
                        "synthetic": True,
                        "synthetic_blob": f"[migrated {memory_type}] {match_text[:500]}",
                        "pre_image": pre_image,
                        "post_image": _memory_post_image(
                            record,
                            existing_annotations + ["kernle:auto-linked", "kernle:synthetic-raw"],
                            None,
                        ),
                        "existing_annotations": existing_annotations,
                    }
                )

    # Separate matched and synthetic links for reporting
    matched_links = [link for link in links if not link.get("synthetic")]
    synthetic_links = [link for link in links if link.get("synthetic")]

    links = sorted(
        links,
        key=lambda item: (
            item.get("type", ""),
            item.get("id", ""),
            str(item.get("raw_id") or ""),
        ),
    )
    matched_links = [link for link in links if not link.get("synthetic")]
    synthetic_links = [link for link in links if link.get("synthetic")]

    # Output results
    if json_output:
        by_type: Dict[str, int] = {}
        for link in links:
            by_type[link["type"]] = by_type.get(link["type"], 0) + 1
        status_snapshot = {
            "total_links": len(links),
            "matched_links": len(matched_links),
            "synthetic_links": len(synthetic_links),
            "types": sorted(by_type.keys()),
            "ids": sorted([link["id"] for link in links]),
        }
        payload = {
            "dry_run": dry_run,
            "total_links": len(links),
            "matched_links": len(matched_links),
            "synthetic_links": len(synthetic_links),
            "window_minutes": window_minutes,
            "raw_entries_available": len(raw_index),
            "link_all": link_all,
            "links": links,
            "status_snapshot": status_snapshot,
            "status_snapshot_sha256": _snapshot_digest(status_snapshot),
        }
        print(
            _json.dumps(
                payload,
                indent=2,
                default=str,
            )
        )
        if dry_run:
            return
    else:
        print(f"Link Raw Entries for stack: {k.stack_id}")
        print("=" * 60)
        print(f"Raw entries available: {len(raw_index)}")
        print(f"Time window: {window_minutes} minutes")
        if link_all:
            print("Mode: --all (synthetic raw entries for unmatched memories)")
        print(
            f"Memories linkable: {len(links)} ({len(matched_links)} matched, {len(synthetic_links)} synthetic)"
        )

        if not links:
            print("\n✓ No linkable memories found!")
            print("  (All memories already have provenance, or no matching raw entries)")
            return

        by_type: Dict[str, int] = {}
        for link in links:
            by_type[link["type"]] = by_type.get(link["type"], 0) + 1
        for t, c in sorted(by_type.items()):
            print(f"  {t}: {c}")

        if dry_run:
            print("\n=== DRY RUN (no changes made) ===\n")
            for link in matched_links:
                print(
                    f"  [{link['type']}] {link['id'][:8]}... → raw:{link['raw_id'][:8]}... (score={link['score']})"
                )
                print(f"          memory: {link['summary']}")
                print(f"          raw:    {link['raw_blob']}")
            if synthetic_links:
                print(f"\n  --- Synthetic raw entries ({len(synthetic_links)}) ---\n")
                for link in synthetic_links:
                    print(f"  [{link['type']}] {link['id'][:8]}... → new synthetic raw")
                    print(f"          memory: {link['summary']}")
            apply_cmd = f"kernle -s {k.stack_id} migrate link-raw"
            if link_all:
                apply_cmd += " --all"
            print(f"\nTo apply: {apply_cmd}")
            return

    # Apply links
    applied = 0
    synthetic_created = 0
    errors = []

    for link in links:
        try:
            if link.get("synthetic"):
                # Create a synthetic raw entry from the memory content
                raw_id = k._storage.save_raw(
                    link["synthetic_blob"],
                    source="migration",
                )
                new_derived_from = [
                    f"raw:{raw_id}",
                    "kernle:auto-linked",
                    "kernle:synthetic-raw",
                ] + link.get("existing_annotations", [])
                link["new_derived_from"] = new_derived_from
                link["raw_id"] = raw_id
                link["post_image"] = _memory_post_image(
                    k._storage.get_memory(link["type"], link["id"]),
                    new_derived_from,
                    f"raw:{raw_id}",
                )
                synthetic_created += 1

            k.set_memory_source(
                link["type"],
                link["id"],
                getattr(
                    k._storage.get_memory(link["type"], link["id"]),
                    "source_type",
                    "direct_experience",
                ),
                derived_from=link["new_derived_from"],
            )
            applied += 1
        except Exception as e:
            logger.debug("Raw link failed for %s:%s: %s", link["type"], link["id"][:8], e)
            errors.append(f"{link['type']}:{link['id'][:8]}...: {e}")

    if not json_output:
        print(f"\n✓ Linked {applied}/{len(links)} memories to raw entries")
        if synthetic_created:
            print(f"  ({synthetic_created} synthetic raw entries created)")
        if errors:
            print(f"\n⚠ {len(errors)} errors:")
            for err in errors[:5]:
                print(f"  - {err}")
