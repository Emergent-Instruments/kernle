"""Controlled forgetting commands for Kernle CLI."""

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kernle import Kernle


def cmd_forget(args, k: "Kernle"):
    """Handle controlled forgetting subcommands."""
    if args.forget_action == "candidates":
        threshold = getattr(args, "threshold", 0.3)
        limit = getattr(args, "limit", 20)

        candidates = k.get_forgetting_candidates(threshold=threshold, limit=limit)

        if args.json:
            print(json.dumps(candidates, indent=2, default=str))
        else:
            if not candidates:
                print(f"No forgetting candidates found below threshold {threshold}")
                return

            print(f"Forgetting Candidates (salience < {threshold})")
            print("=" * 60)
            print()

            for i, c in enumerate(candidates, 1):
                salience_bar = "░" * 5  # Low salience = empty bar
                if c["salience"] > 0.1:
                    filled = min(5, int(c["salience"] * 10))
                    salience_bar = "█" * filled + "░" * (5 - filled)

                print(f"{i}. [{c['type']:<10}] {c['id'][:8]}...")
                print(f"   Salience: [{salience_bar}] {c['salience']:.4f}")
                print(f"   Summary: {c['summary'][:50]}...")
                print(
                    f"   Confidence: {c['confidence']:.0%} | Accessed: {c['times_accessed']} times"
                )
                print(f"   Created: {c['created_at']}")
                if c["last_accessed"]:
                    print(f"   Last accessed: {c['last_accessed'][:10]}")
                print()

            print("Run `kernle forget run --dry-run` to preview forgetting")
            print("Run `kernle forget run` to actually forget these memories")

    elif args.forget_action == "run":
        threshold = getattr(args, "threshold", 0.3)
        limit = getattr(args, "limit", 10)
        dry_run = getattr(args, "dry_run", False)

        result = k.run_forgetting_cycle(
            threshold=threshold,
            limit=limit,
            dry_run=dry_run,
        )

        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            mode = "DRY RUN" if dry_run else "LIVE"
            print(f"Forgetting Cycle [{mode}]")
            print("=" * 60)
            print()
            print(f"Threshold: {result['threshold']}")
            print(f"Candidates: {result['candidate_count']}")

            if dry_run:
                print("\n⚠️  DRY RUN - No memories were actually forgotten")
                print("Run without --dry-run to forget these memories")
            else:
                print(f"Forgotten: {result['forgotten']}")
                print(f"Protected (skipped): {result['protected']}")

            print()

            if result["candidates"]:
                print("Affected memories:")
                for c in result["candidates"][:10]:
                    status = (
                        "🔴 forgotten"
                        if not dry_run and result["forgotten"] > 0
                        else "⚪ candidate"
                    )
                    print(f"  {status} [{c['type']:<10}] {c['summary'][:40]}...")

            if not dry_run and result["forgotten"] > 0:
                print(f"\n✓ Forgotten {result['forgotten']} memories")
                print("Run `kernle forget list` to see all forgotten memories")
                print("Run `kernle forget recover <type> <id>` to recover if needed")

    elif args.forget_action == "protect":
        memory_type = args.type
        memory_id = args.id
        unprotect = getattr(args, "unprotect", False)

        success = k.protect(memory_type, memory_id, protected=not unprotect)

        if success:
            if unprotect:
                print(f"✓ Removed protection from {memory_type} {memory_id[:8]}...")
            else:
                print(f"✓ Protected {memory_type} {memory_id[:8]}... from forgetting")
        else:
            print(f"Memory not found: {memory_type} {memory_id}")

    elif args.forget_action == "recover":
        memory_type = args.type
        memory_id = args.id

        success = k.recover(memory_type, memory_id)

        if success:
            print(f"✓ Recovered {memory_type} {memory_id[:8]}...")
        else:
            print(f"Memory not found or not forgotten: {memory_type} {memory_id}")

    elif args.forget_action == "list":
        limit = getattr(args, "limit", 50)

        forgotten = k.get_forgotten_memories(limit=limit)

        if args.json:
            print(json.dumps(forgotten, indent=2, default=str))
        else:
            if not forgotten:
                print("No forgotten memories found.")
                return

            print(f"Forgotten Memories ({len(forgotten)} total)")
            print("=" * 60)
            print()

            for i, f in enumerate(forgotten, 1):
                print(f"{i}. [{f['type']:<10}] {f['id'][:8]}...")
                print(f"   Summary: {f['summary'][:50]}...")
                print(
                    f"   Forgotten at: {f['forgotten_at'][:10] if f['forgotten_at'] else 'unknown'}"
                )
                if f["forgotten_reason"]:
                    print(f"   Reason: {f['forgotten_reason'][:50]}...")
                print(f"   Created: {f['created_at']}")
                print()

            print("To recover a memory: kernle forget recover <type> <id>")

    elif args.forget_action == "salience":
        memory_type = args.type
        memory_id = args.id

        salience = k.calculate_salience(memory_type, memory_id)

        if salience < 0:
            print(f"Memory not found: {memory_type} {memory_id}")
            return

        # Get the memory for more info
        record = k._storage.get_memory(memory_type, memory_id)

        print(f"Salience Analysis: {memory_type} {memory_id[:8]}...")
        print("=" * 50)
        print()

        # Visual salience bar
        filled = min(10, int(salience * 10))
        salience_bar = "█" * filled + "░" * (10 - filled)
        print(f"Salience: [{salience_bar}] {salience:.4f}")
        print()

        # Component breakdown
        confidence = getattr(record, "confidence", 0.8)
        times_accessed = getattr(record, "times_accessed", 0) or 0
        is_protected = getattr(record, "is_protected", False)

        print("Components:")
        print(f"  Confidence: {confidence:.0%}")
        print(f"  Times accessed: {times_accessed}")
        print(f"  Protected: {'Yes ✓' if is_protected else 'No'}")

        last_accessed = getattr(record, "last_accessed", None)
        created_at = getattr(record, "created_at", None)
        if last_accessed:
            print(f"  Last accessed: {last_accessed.isoformat()[:10]}")
        elif created_at:
            print(f"  Created: {created_at.isoformat()[:10]} (never accessed)")

        print()

        # Interpretation
        if is_protected:
            print("Status: 🛡️  PROTECTED - Will never be forgotten")
        elif salience < 0.1:
            print("Status: 🔴 CRITICAL - Very low salience, prime forgetting candidate")
        elif salience < 0.3:
            print("Status: 🟠 LOW - Below default threshold, forgetting candidate")
        elif salience < 0.5:
            print("Status: 🟡 MODERATE - May decay over time")
        else:
            print("Status: 🟢 HIGH - Well-reinforced memory")
