"""Epoch (temporal era) commands for Kernle CLI."""

import json
from typing import TYPE_CHECKING

from kernle.cli.commands.helpers import validate_input

if TYPE_CHECKING:
    from kernle import Kernle


def cmd_epoch(args, k: "Kernle"):
    """Handle epoch subcommands."""
    if args.epoch_action == "create":
        name = validate_input(args.name, "name", 200)
        trigger = getattr(args, "trigger", "manual") or "manual"

        try:
            epoch_id = k.epoch_create(name=name, trigger_type=trigger)
            if args.json:
                print(json.dumps({"epoch_id": epoch_id, "name": name}))
            else:
                print(f"Epoch created: {name}")
                print(f"  ID: {epoch_id[:8]}...")
                print(f"  Trigger: {trigger}")
        except ValueError as e:
            print(f"Error: {e}")

    elif args.epoch_action == "close":
        epoch_id = getattr(args, "id", None)
        summary = getattr(args, "summary", None)

        if summary:
            summary = validate_input(summary, "summary", 2000)

        result = k.epoch_close(epoch_id=epoch_id, summary=summary)
        if args.json:
            print(json.dumps({"closed": result}))
        else:
            if result:
                print("Epoch closed.")
                if summary:
                    print(f"  Summary: {summary[:60]}...")
            else:
                print("No open epoch to close.")

    elif args.epoch_action == "list":
        epochs = k.get_epochs(limit=getattr(args, "limit", 20))

        if args.json:
            data = [
                {
                    "id": e.id,
                    "epoch_number": e.epoch_number,
                    "name": e.name,
                    "started_at": e.started_at.isoformat() if e.started_at else None,
                    "ended_at": e.ended_at.isoformat() if e.ended_at else None,
                    "trigger_type": e.trigger_type,
                    "summary": e.summary,
                }
                for e in epochs
            ]
            print(json.dumps(data, indent=2, default=str))
        else:
            if not epochs:
                print("No epochs found.")
                return

            print("Epochs")
            print("=" * 60)

            for e in epochs:
                status = "ACTIVE" if e.ended_at is None else "closed"
                started = e.started_at.strftime("%Y-%m-%d") if e.started_at else "unknown"
                ended = e.ended_at.strftime("%Y-%m-%d") if e.ended_at else "now"

                print(f"\n  [{e.epoch_number}] {e.name} ({status})")
                print(f"      ID: {e.id[:8]}...")
                print(f"      Period: {started} - {ended}")
                print(f"      Trigger: {e.trigger_type}")
                if e.summary:
                    print(f"      Summary: {e.summary[:60]}...")

    elif args.epoch_action == "show":
        epoch_id = validate_input(args.id, "epoch_id", 100)
        epoch = k.get_epoch(epoch_id)

        if not epoch:
            print(f"Epoch {epoch_id[:8]}... not found.")
            return

        if args.json:
            data = {
                "id": epoch.id,
                "epoch_number": epoch.epoch_number,
                "name": epoch.name,
                "started_at": epoch.started_at.isoformat() if epoch.started_at else None,
                "ended_at": epoch.ended_at.isoformat() if epoch.ended_at else None,
                "trigger_type": epoch.trigger_type,
                "summary": epoch.summary,
                "key_belief_ids": epoch.key_belief_ids,
                "key_relationship_ids": epoch.key_relationship_ids,
                "key_goal_ids": epoch.key_goal_ids,
                "dominant_drive_ids": epoch.dominant_drive_ids,
            }
            print(json.dumps(data, indent=2, default=str))
        else:
            status = "ACTIVE" if epoch.ended_at is None else "closed"
            started = epoch.started_at.strftime("%Y-%m-%d %H:%M") if epoch.started_at else "unknown"
            ended = epoch.ended_at.strftime("%Y-%m-%d %H:%M") if epoch.ended_at else "now"

            print(f"Epoch #{epoch.epoch_number}: {epoch.name} ({status})")
            print("=" * 60)
            print(f"  ID: {epoch.id}")
            print(f"  Period: {started} - {ended}")
            print(f"  Trigger: {epoch.trigger_type}")
            if epoch.summary:
                print(f"  Summary: {epoch.summary}")
            if epoch.key_belief_ids:
                print(f"  Key beliefs: {len(epoch.key_belief_ids)}")
            if epoch.key_relationship_ids:
                print(f"  Key relationships: {len(epoch.key_relationship_ids)}")
            if epoch.key_goal_ids:
                print(f"  Key goals: {len(epoch.key_goal_ids)}")
            if epoch.dominant_drive_ids:
                print(f"  Dominant drives: {len(epoch.dominant_drive_ids)}")

    elif args.epoch_action == "current":
        epoch = k.get_current_epoch()

        if args.json:
            if epoch:
                data = {
                    "id": epoch.id,
                    "epoch_number": epoch.epoch_number,
                    "name": epoch.name,
                    "started_at": epoch.started_at.isoformat() if epoch.started_at else None,
                    "trigger_type": epoch.trigger_type,
                }
                print(json.dumps(data, indent=2, default=str))
            else:
                print(json.dumps(None))
        else:
            if epoch:
                started = (
                    epoch.started_at.strftime("%Y-%m-%d %H:%M") if epoch.started_at else "unknown"
                )
                print(f"Current epoch: #{epoch.epoch_number} - {epoch.name}")
                print(f"  Started: {started}")
                print(f"  ID: {epoch.id[:8]}...")
            else:
                print("No active epoch.")
