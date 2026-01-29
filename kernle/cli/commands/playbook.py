"""Playbook (procedural memory) commands for Kernle CLI."""

import json
from typing import TYPE_CHECKING

from kernle.cli.commands.helpers import validate_input

if TYPE_CHECKING:
    from kernle import Kernle


def cmd_playbook(args, k: "Kernle"):
    """Handle playbook (procedural memory) commands."""
    if args.playbook_action == "create":
        name = validate_input(args.name, "name", 200)
        description = validate_input(
            args.description or f"Playbook for {name}", "description", 2000
        )

        # Parse steps - support both comma-separated and multiple --step flags
        steps = []
        if args.steps:
            if "," in args.steps:
                steps = [s.strip() for s in args.steps.split(",")]
            else:
                steps = [args.steps]
        if args.step:
            steps.extend(args.step)

        # Parse triggers
        triggers = []
        if args.triggers:
            if "," in args.triggers:
                triggers = [t.strip() for t in args.triggers.split(",")]
            else:
                triggers = [args.triggers]
        if args.trigger:
            triggers.extend(args.trigger)

        # Parse failure modes
        failure_modes = []
        if args.failure_mode:
            failure_modes.extend(args.failure_mode)

        # Parse recovery steps
        recovery_steps = []
        if args.recovery:
            recovery_steps.extend(args.recovery)

        # Parse tags
        tags = []
        if args.tag:
            tags.extend(args.tag)

        playbook_id = k.playbook(
            name=name,
            description=description,
            steps=steps,
            triggers=triggers if triggers else None,
            failure_modes=failure_modes if failure_modes else None,
            recovery_steps=recovery_steps if recovery_steps else None,
            tags=tags if tags else None,
        )

        print(f"✓ Playbook created: {playbook_id[:8]}...")
        print(f"  Name: {name}")
        print(f"  Steps: {len(steps)}")
        if triggers:
            print(f"  Triggers: {len(triggers)}")
        if failure_modes:
            print(f"  Failure modes: {len(failure_modes)}")

    elif args.playbook_action == "list":
        tags = args.tag if args.tag else None
        playbooks = k.load_playbooks(limit=args.limit, tags=tags)

        if not playbooks:
            print("No playbooks found.")
            return

        if args.json:
            print(json.dumps(playbooks, indent=2, default=str))
        else:
            print(f"Playbooks ({len(playbooks)} total)")
            print("=" * 60)

            mastery_icons = {"novice": "🌱", "competent": "🌿", "proficient": "🌳", "expert": "🏆"}

            for p in playbooks:
                icon = mastery_icons.get(p["mastery_level"], "•")
                success_pct = f"{p['success_rate']:.0%}" if p["times_used"] > 0 else "n/a"
                print(f"\n{icon} [{p['id'][:8]}] {p['name']}")
                print(f"   {p['description'][:60]}{'...' if len(p['description']) > 60 else ''}")
                print(
                    f"   Mastery: {p['mastery_level']} | Used: {p['times_used']}x | Success: {success_pct}"
                )
                if p.get("tags"):
                    print(f"   Tags: {', '.join(p['tags'])}")

    elif args.playbook_action == "search":
        query = validate_input(args.query, "query", 500)
        playbooks = k.search_playbooks(query, limit=args.limit)

        if not playbooks:
            print(f"No playbooks found for '{query}'")
            return

        if args.json:
            print(json.dumps(playbooks, indent=2, default=str))
        else:
            print(f"Found {len(playbooks)} playbook(s) for '{query}':\n")

            mastery_icons = {"novice": "🌱", "competent": "🌿", "proficient": "🌳", "expert": "🏆"}

            for i, p in enumerate(playbooks, 1):
                icon = mastery_icons.get(p["mastery_level"], "•")
                success_pct = f"{p['success_rate']:.0%}" if p["times_used"] > 0 else "n/a"
                print(f"{i}. {icon} {p['name']}")
                print(f"   {p['description'][:60]}{'...' if len(p['description']) > 60 else ''}")
                print(
                    f"   Mastery: {p['mastery_level']} | Used: {p['times_used']}x | Success: {success_pct}"
                )
                print()

    elif args.playbook_action == "show":
        playbook = k.get_playbook(args.id)

        if not playbook:
            print(f"Playbook {args.id} not found.")
            return

        if args.json:
            print(json.dumps(playbook, indent=2, default=str))
        else:
            mastery_icons = {"novice": "🌱", "competent": "🌿", "proficient": "🌳", "expert": "🏆"}
            icon = mastery_icons.get(playbook["mastery_level"], "•")

            print(f"{icon} Playbook: {playbook['name']}")
            print("=" * 60)
            print(f"ID: {playbook['id']}")
            print(f"Description: {playbook['description']}")
            print()

            print("## Triggers (when to use)")
            if playbook.get("triggers"):
                for t in playbook["triggers"]:
                    print(f"  • {t}")
            else:
                print("  (none specified)")
            print()

            print("## Steps")
            for i, step in enumerate(playbook["steps"], 1):
                if isinstance(step, dict):
                    print(f"  {i}. {step.get('action', 'Unknown step')}")
                    if step.get("details"):
                        print(f"     Details: {step['details']}")
                    if step.get("adaptations"):
                        print(f"     Adaptations: {step['adaptations']}")
                else:
                    print(f"  {i}. {step}")
            print()

            print("## Failure Modes (what can go wrong)")
            if playbook.get("failure_modes"):
                for f in playbook["failure_modes"]:
                    print(f"  ⚠️  {f}")
            else:
                print("  (none specified)")

            if playbook.get("recovery_steps"):
                print()
                print("## Recovery Steps")
                for i, r in enumerate(playbook["recovery_steps"], 1):
                    print(f"  {i}. {r}")

            print()
            print("## Statistics")
            success_pct = f"{playbook['success_rate']:.0%}" if playbook["times_used"] > 0 else "n/a"
            print(f"  Mastery Level: {playbook['mastery_level']}")
            print(f"  Times Used: {playbook['times_used']}")
            print(f"  Success Rate: {success_pct}")
            print(f"  Confidence: {playbook['confidence']:.0%}")

            if playbook.get("tags"):
                print(f"  Tags: {', '.join(playbook['tags'])}")
            if playbook.get("last_used"):
                print(f"  Last Used: {playbook['last_used'][:10]}")
            if playbook.get("created_at"):
                print(f"  Created: {playbook['created_at'][:10]}")

    elif args.playbook_action == "find":
        situation = validate_input(args.situation, "situation", 1000)
        playbook = k.find_playbook(situation)

        if not playbook:
            print(f"No relevant playbook found for: {situation}")
            return

        if args.json:
            print(json.dumps(playbook, indent=2, default=str))
        else:
            print(f"Recommended Playbook: {playbook['name']}")
            print(f"  {playbook['description'][:80]}")
            print()
            print("Steps:")
            for i, step in enumerate(playbook["steps"], 1):
                if isinstance(step, dict):
                    print(f"  {i}. {step.get('action', 'Unknown step')}")
                else:
                    print(f"  {i}. {step}")
            print()
            print(
                f"(Mastery: {playbook['mastery_level']} | Success: {playbook['success_rate']:.0%})"
            )
            print(f"\nTo record usage: kernle playbook record {playbook['id'][:8]}... --success")

    elif args.playbook_action == "record":
        success = not args.failure

        if k.record_playbook_use(args.id, success):
            result = "success ✓" if success else "failure ✗"
            print(f"✓ Recorded playbook usage: {result}")
        else:
            print(f"Playbook {args.id} not found.")
