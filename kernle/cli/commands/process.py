"""Memory processing commands for Kernle CLI."""

import json
from typing import TYPE_CHECKING

from kernle.processing import DEFAULT_LAYER_CONFIGS, VALID_TRANSITIONS

if TYPE_CHECKING:
    from kernle import Kernle


def cmd_process(args, k: "Kernle"):
    """Handle process subcommands."""
    if args.process_action == "run":
        transition = getattr(args, "transition", None)
        force = getattr(args, "force", False)
        allow_no_inference_override = getattr(args, "allow_no_inference_override", False)
        auto_promote = getattr(args, "auto_promote", False)

        if transition and transition not in VALID_TRANSITIONS:
            print(f"Invalid transition: {transition}")
            print(f"Valid transitions: {', '.join(sorted(VALID_TRANSITIONS))}")
            return

        try:
            results = k.process(
                transition=transition,
                force=force,
                allow_no_inference_override=allow_no_inference_override,
                auto_promote=auto_promote,
            )
        except RuntimeError as e:
            print(f"Error: {e}")
            print("Memory processing requires a bound model. Use entity.set_model() first.")
            return

        if not results:
            print("No processing triggered.")
            if not force:
                print("Thresholds not met -- use --force to process anyway.")
            else:
                print("No unprocessed memories found.")
            return

        if getattr(args, "json", False):
            output = []
            for r in results:
                output.append(
                    {
                        "transition": r.layer_transition,
                        "source_count": r.source_count,
                        "created": r.created,
                        "suggestions": r.suggestions,
                        "auto_promote": r.auto_promote,
                        "errors": r.errors,
                        "skipped": r.skipped,
                        "skip_reason": r.skip_reason,
                        "inference_blocked": r.inference_blocked,
                        "gate_blocked": r.gate_blocked,
                        "gate_details": r.gate_details,
                    }
                )
            print(json.dumps(output, indent=2, default=str))
        else:
            mode = "auto-promote" if auto_promote else "suggestions"
            print(f"Processing complete ({len(results)} transition(s), mode={mode}):\n")
            for r in results:
                if r.inference_blocked:
                    print(f"  {r.layer_transition}: BLOCKED (no inference) -- {r.skip_reason}")
                elif r.skipped:
                    print(f"  {r.layer_transition}: skipped ({r.skip_reason})")
                elif r.auto_promote:
                    print(
                        f"  {r.layer_transition}: "
                        f"{r.source_count} sources -> {len(r.created)} created"
                    )
                    for c in r.created:
                        print(f"    + {c['type']}:{c['id'][:8]}...")
                    if r.gate_blocked:
                        print(f"    gate blocked: {r.gate_blocked} item(s)")
                        for detail in r.gate_details:
                            print(f"      - {detail}")
                    for err in r.errors:
                        print(f"    ! {err}")
                else:
                    print(
                        f"  {r.layer_transition}: "
                        f"{r.source_count} sources -> {len(r.suggestions)} suggestions"
                    )
                    for s in r.suggestions:
                        print(f"    ? {s['type']}:{s['id'][:8]}... (pending review)")
                    if r.gate_blocked:
                        print(f"    gate blocked: {r.gate_blocked} item(s)")
                        for detail in r.gate_details:
                            print(f"      - {detail}")
                    for err in r.errors:
                        print(f"    ! {err}")

    elif args.process_action == "status":
        try:
            storage = k._storage

            # Raw entries
            unprocessed_raw = storage.list_raw(processed=False, limit=1000)
            raw_count = len(unprocessed_raw)

            # Episodes
            episodes = storage.get_episodes(limit=1000)
            unprocessed_eps = [e for e in episodes if not e.processed]

            # Beliefs
            beliefs = storage.get_beliefs(limit=1000)
            unprocessed_beliefs = [b for b in beliefs if not getattr(b, "processed", False)]

            if getattr(args, "json", False):
                trigger_checks = {}
                source_counts = {
                    "raw_to_episode": raw_count,
                    "raw_to_note": raw_count,
                    "episode_to_belief": len(unprocessed_eps),
                    "episode_to_goal": len(unprocessed_eps),
                    "episode_to_relationship": len(unprocessed_eps),
                    "episode_to_drive": len(unprocessed_eps),
                    "belief_to_value": len(unprocessed_beliefs),
                }
                for transition, count in source_counts.items():
                    cfg = DEFAULT_LAYER_CONFIGS.get(transition)
                    if cfg:
                        trigger_checks[transition] = {
                            "unprocessed": count,
                            "threshold": cfg.quantity_threshold,
                            "would_fire": count >= cfg.quantity_threshold,
                        }
                output = {
                    "unprocessed_raw": raw_count,
                    "unprocessed_episodes": len(unprocessed_eps),
                    "unprocessed_beliefs": len(unprocessed_beliefs),
                    "triggers": trigger_checks,
                }
                print(json.dumps(output, indent=2))
            else:
                print("Memory Processing Status")
                print("=" * 40)
                print(f"Unprocessed raw entries:  {raw_count}")
                print(f"Unprocessed episodes:    {len(unprocessed_eps)}")
                print(f"Unprocessed beliefs:     {len(unprocessed_beliefs)}")
                print()
                print("Trigger Status:")
                trigger_data = {
                    "raw_to_episode": raw_count,
                    "raw_to_note": raw_count,
                    "episode_to_belief": len(unprocessed_eps),
                    "episode_to_goal": len(unprocessed_eps),
                    "episode_to_relationship": len(unprocessed_eps),
                    "episode_to_drive": len(unprocessed_eps),
                    "belief_to_value": len(unprocessed_beliefs),
                }
                for transition, count in trigger_data.items():
                    cfg = DEFAULT_LAYER_CONFIGS.get(transition)
                    if cfg:
                        would_fire = count >= cfg.quantity_threshold
                        status = "READY" if would_fire else "waiting"
                        print(f"  {transition}: {count}/{cfg.quantity_threshold} ({status})")

        except Exception as e:
            print(f"Error gathering status: {e}")

    else:
        print("Usage: kernle process {run|status}")
