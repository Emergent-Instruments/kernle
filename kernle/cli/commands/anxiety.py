"""Anxiety tracking commands for Kernle CLI."""

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kernle import Kernle


def cmd_anxiety(args, k: "Kernle"):
    """Handle anxiety tracking commands."""
    context_tokens = getattr(args, "context", None)
    context_limit = getattr(args, "limit", 200000) or 200000
    source = getattr(args, "source", "cli") or "cli"
    triggered_by = getattr(args, "triggered_by", "manual") or "manual"

    # Emergency mode - run immediately
    if getattr(args, "emergency", False):
        summary = getattr(args, "summary", None)
        result = k.emergency_save(summary=summary)

        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print("🚨 EMERGENCY SAVE COMPLETE")
            print("=" * 50)
            print(f"Checkpoint saved: {'✓' if result['checkpoint_saved'] else '✗'}")
            print(f"Episodes consolidated: {result['episodes_consolidated']}")
            print(f"Identity synthesized: {'✓' if result['identity_synthesized'] else '✗'}")
            print(f"Sync attempted: {'✓' if result['sync_attempted'] else '✗'}")
            if result["sync_attempted"]:
                print(f"Sync success: {'✓' if result['sync_success'] else '✗'}")
            if result["errors"]:
                print("\n⚠️  Errors:")
                for err in result["errors"]:
                    print(f"  - {err}")
            print(
                f"\n{'✓ Emergency save successful' if result['success'] else '⚠️  Partial save (see errors)'}"
            )
        return

    # Get anxiety report
    report = k.get_anxiety_report(
        context_tokens=context_tokens,
        context_limit=context_limit,
        detailed=getattr(args, "detailed", False) or getattr(args, "actions", False),
    )

    # Log health check event for compliance tracking
    try:
        k._storage.log_health_check(
            anxiety_score=report.get("overall_score"), source=source, triggered_by=triggered_by
        )
    except Exception:
        pass  # Don't fail the command if logging fails

    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return

    # Brief mode - single line output for quick health checks
    if getattr(args, "brief", False):
        score = report["overall_score"]
        emoji = report["overall_emoji"]
        _level = report["overall_level"]
        if score >= 80:
            print(f"{emoji} CRITICAL ({score}) - immediate action needed")
        elif score >= 50:
            print(f"{emoji} WARN ({score}) - consider consolidation")
        else:
            print(f"{emoji} OK ({score})")
        return

    # Format the report
    print("\nMemory Anxiety Report")
    print("=" * 50)
    print(
        f"Overall: {report['overall_emoji']} {report['overall_level']} ({report['overall_score']}/100)"
    )
    print()

    # Dimension breakdown
    dim_names = {
        "context_pressure": "Context Pressure",
        "unsaved_work": "Unsaved Work",
        "consolidation_debt": "Consolidation Debt",
        "raw_aging": "Raw Entry Aging",
        "identity_coherence": "Identity Coherence",
        "memory_uncertainty": "Memory Uncertainty",
    }

    for dim_key, dim_label in dim_names.items():
        dim = report["dimensions"][dim_key]
        # Format: "Context Pressure:    🟡 45% (details)"
        score_pct = f"{dim['score']}%"
        if getattr(args, "detailed", False):
            print(f"{dim_label:20} {dim['emoji']} {score_pct:>4} ({dim['detail']})")
        else:
            print(f"{dim_label:20} {dim['emoji']} {score_pct:>4}")

    # Show recommended actions
    if getattr(args, "actions", False) or getattr(args, "detailed", False):
        actions = report.get("recommendations") or k.get_recommended_actions(
            report["overall_score"]
        )
        if actions:
            print("\nRecommended Actions:")
            priority_symbols = {
                "emergency": "🚨",
                "critical": "‼️ ",
                "high": "❗",
                "medium": "⚠️ ",
                "low": "ℹ️ ",
            }
            for i, action in enumerate(actions, 1):
                symbol = priority_symbols.get(action["priority"], "•")
                print(f"  {i}. [{action['priority'].upper():>8}] {symbol} {action['description']}")
                if action.get("command") and getattr(args, "detailed", False):
                    print(f"                      └─ {action['command']}")

    # Auto mode - execute recommended actions
    if getattr(args, "auto", False):
        actions = k.get_recommended_actions(report["overall_score"])
        if not actions:
            print("\n✓ No actions needed - anxiety level is manageable")
            return

        print("\n" + "=" * 50)
        print("Executing recommended actions...")
        print()

        for action in actions:
            method = action.get("method")
            if not method:
                print(f"  ⏭️  Skipping: {action['description']} (manual action)")
                continue

            print(f"  ▶️  {action['description']}...")
            try:
                if method == "checkpoint":
                    k.checkpoint(
                        task="Auto-checkpoint (anxiety management)",
                        context=f"Anxiety level: {report['overall_score']}/100",
                    )
                    print("    ✓ Checkpoint saved")
                elif method == "consolidate":
                    result = k.consolidate(min_episodes=1)
                    print(f"    ✓ Consolidated {result.get('consolidated', 0)} episodes")
                elif method == "synthesize_identity":
                    identity = k.synthesize_identity()
                    print(
                        f"    ✓ Identity synthesized (confidence: {identity.get('confidence', 0):.0%})"
                    )
                elif method == "sync":
                    result = k.sync()
                    if result.get("success"):
                        print(
                            f"    ✓ Synced (pushed: {result.get('pushed', 0)}, pulled: {result.get('pulled', 0)})"
                        )
                    else:
                        print(f"    ⚠️  Sync had issues: {result.get('errors', [])}")
                elif method == "emergency_save":
                    result = k.emergency_save()
                    if result["success"]:
                        print("    ✓ Emergency save completed")
                    else:
                        print(f"    ⚠️  Emergency save had errors: {result['errors']}")
                elif method == "get_uncertain_memories":
                    uncertain = k.get_uncertain_memories(0.5, limit=10)
                    print(f"    ℹ️  Found {len(uncertain)} uncertain memories to review")
                else:
                    print(f"    ⏭️  Skipping: Unknown method {method}")
            except Exception as e:
                print(f"    ✗ Failed: {e}")

        print("\n✓ Auto-execution complete")

        # Show updated anxiety level
        new_report = k.get_anxiety_report(
            context_tokens=context_tokens, context_limit=context_limit
        )
        print(
            f"  New anxiety level: {new_report['overall_emoji']} {new_report['overall_level']} ({new_report['overall_score']}/100)"
        )
    else:
        # Suggest running with --auto
        if report["overall_score"] > 50:
            print("\nRun `kernle anxiety --auto` to execute recommended actions.")
