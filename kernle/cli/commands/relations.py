"""Relationship and entity model commands for Kernle CLI."""

import json
import logging
from typing import TYPE_CHECKING

from kernle.cli.commands.helpers import validate_input

if TYPE_CHECKING:
    from kernle import Kernle

logger = logging.getLogger(__name__)


def cmd_relation(args, k: "Kernle"):
    """Manage relationships with other entities (people, agents, orgs)."""
    if args.relation_action == "list":
        relationships = k.load_relationships(limit=50)
        if not relationships:
            print("No relationships recorded yet.")
            print("\nAdd one with: kernle relation add <name> --type person --notes '...'")
            return

        print("Relationships:")
        print("-" * 50)
        for r in relationships:
            trust_pct = int(((r.get("sentiment", 0) + 1) / 2) * 100)
            trust_bar = "█" * (trust_pct // 10) + "░" * (10 - trust_pct // 10)
            interactions = r.get("interaction_count", 0)
            last = r.get("last_interaction", "")[:10] if r.get("last_interaction") else "never"
            print(f"\n  {r['entity_name']} ({r.get('entity_type', 'unknown')})")
            print(f"    Trust: [{trust_bar}] {trust_pct}%")
            print(f"    Interactions: {interactions} (last: {last})")
            if r.get("notes"):
                notes_preview = r["notes"][:60] + "..." if len(r["notes"]) > 60 else r["notes"]
                print(f"    Notes: {notes_preview}")

    elif args.relation_action == "add":
        name = validate_input(args.name, "name", 200)
        entity_type = args.type or "person"
        trust = args.trust if args.trust is not None else 0.5
        notes = validate_input(args.notes, "notes", 1000) if args.notes else None
        derived_from = getattr(args, "derived_from", None)

        _rel_id = k.relationship(
            name,
            trust_level=trust,
            notes=notes,
            entity_type=entity_type,
            derived_from=derived_from,
        )
        print(f"✓ Relationship added: {name}")
        print(f"  Type: {entity_type}, Trust: {int(trust * 100)}%")
        if derived_from:
            print(f"  Derived from: {len(derived_from)} memories")

    elif args.relation_action == "update":
        name = validate_input(args.name, "name", 200)
        trust = args.trust
        notes = validate_input(args.notes, "notes", 1000) if args.notes else None
        entity_type = getattr(args, "type", None)
        derived_from = getattr(args, "derived_from", None)

        if trust is None and notes is None and entity_type is None and derived_from is None:
            print("✗ Provide --trust, --notes, --type, or --derived-from to update")
            return

        _rel_id = k.relationship(
            name,
            trust_level=trust,
            notes=notes,
            entity_type=entity_type,
            derived_from=derived_from,
        )
        print(f"✓ Relationship updated: {name}")
        if derived_from:
            print(f"  Derived from: {len(derived_from)} memories")

    elif args.relation_action == "show":
        name = args.name
        relationships = k.load_relationships(limit=100)
        rel = next((r for r in relationships if r["entity_name"].lower() == name.lower()), None)

        if not rel:
            print(f"No relationship found for '{name}'")
            return

        trust_pct = int(((rel.get("sentiment", 0) + 1) / 2) * 100)
        print(f"## {rel['entity_name']}")
        print(f"Type: {rel.get('entity_type', 'unknown')}")
        print(f"Trust: {trust_pct}%")
        print(f"Interactions: {rel.get('interaction_count', 0)}")
        if rel.get("last_interaction"):
            print(f"Last interaction: {rel['last_interaction']}")
        if rel.get("notes"):
            print(f"\nNotes:\n{rel['notes']}")

    elif args.relation_action == "log":
        name = validate_input(args.name, "name", 200)
        interaction = (
            validate_input(args.interaction, "interaction", 500)
            if args.interaction
            else "interaction"
        )

        # Update relationship to log interaction
        k.relationship(name, interaction_type=interaction)
        print(f"✓ Logged interaction with {name}: {interaction}")

    elif args.relation_action == "history":
        name = validate_input(args.name, "name", 200)
        event_type = getattr(args, "type", None)
        history = k.get_relationship_history(name, event_type=event_type, limit=args.limit)

        if not history:
            print(f"No history found for '{name}'")
            return

        if args.json:
            print(json.dumps(history, indent=2, default=str))
        else:
            print(f"History for {name} ({len(history)} entries):")
            print("-" * 50)

            event_icons = {
                "interaction": ">>",
                "trust_change": "~~",
                "type_change": "->",
                "note": "##",
            }

            for entry in history:
                icon = event_icons.get(entry["event_type"], "*")
                ts = entry["created_at"][:10] if entry["created_at"] else "?"
                print(f"\n  {icon} [{ts}] {entry['event_type']}")
                if entry.get("old_value"):
                    print(f"     From: {entry['old_value']}")
                if entry.get("new_value"):
                    print(f"     To:   {entry['new_value']}")
                if entry.get("notes"):
                    print(f"     Note: {entry['notes']}")


def cmd_entity_model(args, k: "Kernle"):
    """Manage entity models (mental models of other entities)."""
    if args.entity_model_action == "add":
        entity = validate_input(args.entity, "entity", 200)
        observation = validate_input(args.observation, "observation", 2000)
        model_type = args.type
        confidence = args.confidence
        source_episodes = args.episode if args.episode else None

        model_id = k.add_entity_model(
            entity_name=entity,
            model_type=model_type,
            observation=observation,
            confidence=confidence,
            source_episodes=source_episodes,
        )
        print(f"Entity model added: {model_id[:8]}...")
        print(f"  Entity: {entity}")
        print(f"  Type: {model_type}")
        print(f"  Observation: {observation[:60]}{'...' if len(observation) > 60 else ''}")

    elif args.entity_model_action == "list":
        entity = getattr(args, "entity", None)
        model_type = getattr(args, "type", None)
        models = k.get_entity_models(entity_name=entity, model_type=model_type, limit=args.limit)

        if not models:
            print("No entity models found.")
            return

        if args.json:
            print(json.dumps(models, indent=2, default=str))
        else:
            print(f"Entity Models ({len(models)} total)")
            print("=" * 60)

            type_icons = {"behavioral": "[B]", "preference": "[P]", "capability": "[C]"}

            for m in models:
                icon = type_icons.get(m["model_type"], "[?]")
                conf = f"{m['confidence']:.0%}"
                obs_preview = (
                    m["observation"][:50] + "..."
                    if len(m["observation"]) > 50
                    else m["observation"]
                )
                print(f"\n  {icon} {m['entity_name']} ({conf})")
                print(f"      {obs_preview}")
                if m.get("created_at"):
                    print(f"      Added: {m['created_at'][:10]}")

    elif args.entity_model_action == "show":
        model = k.get_entity_model(args.id)

        if not model:
            print(f"Entity model {args.id} not found.")
            return

        if args.json:
            print(json.dumps(model, indent=2, default=str))
        else:
            type_icons = {"behavioral": "[B]", "preference": "[P]", "capability": "[C]"}
            icon = type_icons.get(model["model_type"], "[?]")

            print(f"{icon} Entity Model: {model['entity_name']}")
            print("=" * 60)
            print(f"ID: {model['id']}")
            print(f"Type: {model['model_type']}")
            print(f"Confidence: {model['confidence']:.0%}")
            print(f"\nObservation:\n  {model['observation']}")
            if model.get("source_episodes"):
                print(f"\nSource Episodes: {', '.join(model['source_episodes'])}")
            if model.get("created_at"):
                print(f"Created: {model['created_at'][:10]}")
            if model.get("updated_at"):
                print(f"Updated: {model['updated_at'][:10]}")
