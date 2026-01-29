"""Emotion commands for Kernle CLI."""

import json
from typing import TYPE_CHECKING

from kernle.cli.commands.helpers import validate_input

if TYPE_CHECKING:
    from kernle import Kernle


def cmd_emotion(args, k: "Kernle"):
    """Handle emotion subcommands."""
    if args.emotion_action == "summary":
        summary = k.get_emotional_summary(args.days)

        if args.json:
            print(json.dumps(summary, indent=2, default=str))
        else:
            print(f"Emotional Summary (past {args.days} days)")
            print("=" * 50)

            if summary["episode_count"] == 0:
                print("No emotional data recorded yet.")
                return

            # Valence visualization
            valence = summary["average_valence"]
            valence_pct = (valence + 1) / 2  # Convert -1..1 to 0..1
            valence_bar = "█" * int(valence_pct * 20) + "░" * (20 - int(valence_pct * 20))
            valence_label = (
                "positive" if valence > 0.2 else "negative" if valence < -0.2 else "neutral"
            )
            print(f"Avg Valence:  [{valence_bar}] {valence:+.2f} ({valence_label})")

            # Arousal visualization
            arousal = summary["average_arousal"]
            arousal_bar = "█" * int(arousal * 20) + "░" * (20 - int(arousal * 20))
            arousal_label = "high" if arousal > 0.6 else "low" if arousal < 0.3 else "moderate"
            print(f"Avg Arousal:  [{arousal_bar}] {arousal:.2f} ({arousal_label})")
            print()

            if summary["dominant_emotions"]:
                print("Dominant Emotions:")
                for emotion in summary["dominant_emotions"]:
                    print(f"  • {emotion}")
                print()

            if summary["emotional_trajectory"]:
                print("Trajectory:")
                for point in summary["emotional_trajectory"][-7:]:  # Last 7 days
                    v = point["valence"]
                    trend = "😊" if v > 0.3 else "😢" if v < -0.3 else "😐"
                    print(f"  {point['date']}: {trend} v={v:+.2f} a={point['arousal']:.2f}")

            print(f"\n({summary['episode_count']} emotional episodes)")

    elif args.emotion_action == "search":
        # Parse valence/arousal ranges
        valence_range = None
        arousal_range = None

        if args.positive:
            valence_range = (0.3, 1.0)
        elif args.negative:
            valence_range = (-1.0, -0.3)
        elif args.valence_min is not None or args.valence_max is not None:
            valence_range = (args.valence_min or -1.0, args.valence_max or 1.0)

        if args.calm:
            arousal_range = (0.0, 0.3)
        elif args.intense:
            arousal_range = (0.7, 1.0)
        elif args.arousal_min is not None or args.arousal_max is not None:
            arousal_range = (args.arousal_min or 0.0, args.arousal_max or 1.0)

        results = k.search_by_emotion(
            valence_range=valence_range,
            arousal_range=arousal_range,
            tags=args.tag,
            limit=args.limit,
        )

        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            if not results:
                print("No matching episodes found.")
                return

            print(f"Found {len(results)} matching episode(s):\n")
            for ep in results:
                v = ep.get("emotional_valence", 0) or 0
                a = ep.get("emotional_arousal", 0) or 0
                tags = ep.get("emotional_tags") or []
                mood = "😊" if v > 0.3 else "😢" if v < -0.3 else "😐"

                print(f"{mood} {ep.get('objective', '')[:50]}")
                print(f"   valence: {v:+.2f}  arousal: {a:.2f}")
                if tags:
                    print(f"   emotions: {', '.join(tags)}")
                print(f"   {ep.get('created_at', '')[:10]}")
                print()

    elif args.emotion_action == "tag":
        episode_id = validate_input(args.episode_id, "episode_id", 100)

        if k.add_emotional_association(
            episode_id,
            valence=args.valence,
            arousal=args.arousal,
            tags=args.tag,
        ):
            print(f"✓ Emotional tags added to episode {episode_id[:8]}...")
            print(f"  valence: {args.valence:+.2f}, arousal: {args.arousal:.2f}")
            if args.tag:
                print(f"  emotions: {', '.join(args.tag)}")
        else:
            print(f"✗ Episode {episode_id[:8]}... not found")

    elif args.emotion_action == "detect":
        text = validate_input(args.text, "text", 2000)
        result = k.detect_emotion(text)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if result["confidence"] == 0:
                print("No emotional signals detected.")
            else:
                v = result["valence"]
                a = result["arousal"]
                mood = "😊" if v > 0.3 else "😢" if v < -0.3 else "😐"

                print(f"Detected Emotions: {mood}")
                print(
                    f"  Valence: {v:+.2f} ({'positive' if v > 0 else 'negative' if v < 0 else 'neutral'})"
                )
                print(
                    f"  Arousal: {a:.2f} ({'high' if a > 0.6 else 'low' if a < 0.3 else 'moderate'})"
                )
                print(f"  Tags: {', '.join(result['tags']) if result['tags'] else 'none'}")
                print(f"  Confidence: {result['confidence']:.0%}")

    elif args.emotion_action == "mood":
        # Get mood-relevant memories
        results = k.get_mood_relevant_memories(
            current_valence=args.valence,
            current_arousal=args.arousal,
            limit=args.limit,
        )

        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            mood = "😊" if args.valence > 0.3 else "😢" if args.valence < -0.3 else "😐"
            print(
                f"Memories relevant to mood: {mood} (v={args.valence:+.2f}, a={args.arousal:.2f})"
            )
            print("=" * 50)

            if not results:
                print("No mood-relevant memories found.")
                return

            for ep in results:
                v = ep.get("emotional_valence", 0) or 0
                a = ep.get("emotional_arousal", 0) or 0
                ep_mood = "😊" if v > 0.3 else "😢" if v < -0.3 else "😐"

                print(f"\n{ep_mood} {ep.get('objective', '')[:50]}")
                print(f"   {ep.get('outcome_description', '')[:60]}")
                if ep.get("lessons_learned"):
                    print(f"   → {ep['lessons_learned'][0][:50]}...")
                print(f"   v={v:+.2f} a={a:.2f} | {ep.get('created_at', '')[:10]}")
