"""Identity and consolidation commands for Kernle CLI."""

import json
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kernle import Kernle


def cmd_consolidate(args, k: "Kernle"):
    """Output guided reflection prompt for memory consolidation.

    This command fetches recent episodes and existing beliefs,
    then outputs a structured prompt to guide the agent through
    reflection and pattern identification. The AGENT does the
    reasoning - Kernle just provides the data and structure.
    """
    # Get episode limit from args (default 20)
    limit = getattr(args, "limit", 20) or 20

    # Fetch recent episodes with full details
    episodes = k._storage.get_episodes(limit=limit)
    episodes = [ep for ep in episodes if not ep.is_forgotten]

    # Fetch existing beliefs for context
    beliefs = k._storage.get_beliefs(limit=20)
    beliefs = [b for b in beliefs if b.is_active and not b.is_forgotten]
    beliefs = sorted(beliefs, key=lambda b: b.confidence, reverse=True)

    # Count lessons across episodes
    all_lessons = []
    for ep in episodes:
        if ep.lessons:
            all_lessons.extend(ep.lessons)

    # Find repeated lessons
    lesson_counts = Counter(all_lessons)
    repeated_lessons = [(lesson, count) for lesson, count in lesson_counts.items() if count >= 2]
    repeated_lessons.sort(key=lambda x: -x[1])

    # Output the reflection prompt
    print("## Memory Consolidation - Reflection Prompt")
    print()
    print(
        f"You have {len(episodes)} recent episodes to reflect on. Review them and identify patterns."
    )
    print()

    # Recent Episodes section
    print("### Recent Episodes:")
    if not episodes:
        print("No episodes recorded yet.")
    else:
        for i, ep in enumerate(episodes, 1):
            # Format date
            date_str = ep.created_at.strftime("%Y-%m-%d") if ep.created_at else "unknown"

            # Outcome type indicator
            outcome_icon = (
                "✓"
                if ep.outcome_type == "success"
                else (
                    "○"
                    if ep.outcome_type == "partial"
                    else "✗" if ep.outcome_type == "failure" else "•"
                )
            )

            print(f'{i}. [{date_str}] {outcome_icon} "{ep.objective}"')
            print(f"   Outcome: {ep.outcome}")

            if ep.lessons:
                lessons_str = json.dumps(ep.lessons)
                print(f"   Lessons: {lessons_str}")

            # Emotional context if present
            if ep.emotional_valence != 0 or ep.emotional_arousal != 0:
                valence_label = (
                    "positive"
                    if ep.emotional_valence > 0.2
                    else "negative" if ep.emotional_valence < -0.2 else "neutral"
                )
                arousal_label = (
                    "high"
                    if ep.emotional_arousal > 0.6
                    else "low" if ep.emotional_arousal < 0.3 else "moderate"
                )
                print(f"   Emotion: {valence_label}, {arousal_label} intensity")
                if ep.emotional_tags:
                    print(f"   Feelings: {', '.join(ep.emotional_tags)}")

            print()

    # Current Beliefs section
    print("### Current Beliefs (for context):")
    if not beliefs:
        print("No beliefs recorded yet.")
    else:
        for b in beliefs[:10]:  # Limit to top 10 by confidence
            print(f'- "{b.statement}" (confidence: {b.confidence:.2f})')
    print()

    # Repeated Lessons section (if any)
    if repeated_lessons:
        print("### Patterns Detected:")
        print("These lessons appear in multiple episodes:")
        for lesson, count in repeated_lessons[:5]:
            print(f'- "{lesson}" (appears {count} times)')
        print()

    # Reflection Questions
    print("### Reflection Questions:")
    print("1. Do any patterns emerge across these episodes?")
    print("2. Are there lessons that appear multiple times?")
    print("3. Do any episodes contradict existing beliefs?")
    print("4. What new beliefs (if any) should be added?")
    print("5. Should any existing beliefs be reinforced or revised?")
    print()

    # Instructions for the agent
    print("### Actions:")
    print(f'To add a new belief: kernle -a {k.agent_id} belief add "statement" --confidence 0.8')
    print(f"To reinforce existing: kernle -a {k.agent_id} belief reinforce <belief_id>")
    print(
        f'To revise a belief: kernle -a {k.agent_id} belief revise <belief_id> "new statement" --reason "why"'
    )
    print()
    print("---")
    print("Note: You (the agent) do the reasoning. Kernle just provides the data.")


def cmd_identity(args, k: "Kernle"):
    """Display identity synthesis."""
    if args.identity_action == "show" or args.identity_action is None:
        identity = k.synthesize_identity()

        if getattr(args, "json", False):
            print(json.dumps(identity, indent=2, default=str))
        else:
            print(f"Identity Synthesis for {k.agent_id}")
            print("=" * 50)
            print()
            print("## Narrative")
            print(identity["narrative"])
            print()

            if identity["core_values"]:
                print("## Core Values")
                for v in identity["core_values"]:
                    print(f"  • {v['name']} (priority {v['priority']}): {v['statement']}")
                print()

            if identity["key_beliefs"]:
                print("## Key Beliefs")
                for b in identity["key_beliefs"]:
                    foundational = " [foundational]" if b.get("foundational") else ""
                    print(f"  • {b['statement']} ({b['confidence']:.0%} confidence){foundational}")
                print()

            if identity["active_goals"]:
                print("## Active Goals")
                for g in identity["active_goals"]:
                    print(f"  • {g['title']} [{g['priority']}]")
                print()

            if identity["drives"]:
                print("## Drives")
                for drive_type, intensity in sorted(
                    identity["drives"].items(), key=lambda x: -x[1]
                ):
                    bar = "█" * int(intensity * 10) + "░" * (10 - int(intensity * 10))
                    print(f"  {drive_type:12} [{bar}] {intensity:.0%}")
                print()

            if identity["significant_episodes"]:
                print("## Formative Experiences")
                for ep in identity["significant_episodes"]:
                    outcome_icon = "✓" if ep["outcome"] == "success" else "○"
                    print(f"  {outcome_icon} {ep['objective'][:50]}")
                    if ep.get("lessons"):
                        for lesson in ep["lessons"]:
                            print(f"      → {lesson[:60]}")
                print()

            print(f"Identity Confidence: {identity['confidence']:.0%}")

    elif args.identity_action == "confidence":
        confidence = k.get_identity_confidence()
        if args.json:
            print(json.dumps({"agent_id": k.agent_id, "confidence": confidence}))
        else:
            bar = "█" * int(confidence * 20) + "░" * (20 - int(confidence * 20))
            print(f"Identity Confidence: [{bar}] {confidence:.0%}")

    elif args.identity_action == "drift":
        drift = k.detect_identity_drift(args.days)

        if args.json:
            print(json.dumps(drift, indent=2, default=str))
        else:
            print(f"Identity Drift Analysis (past {drift['period_days']} days)")
            print("=" * 50)

            # Drift score visualization
            drift_score = drift["drift_score"]
            bar = "█" * int(drift_score * 20) + "░" * (20 - int(drift_score * 20))
            interpretation = (
                "stable"
                if drift_score < 0.2
                else (
                    "evolving"
                    if drift_score < 0.5
                    else "significant change" if drift_score < 0.8 else "transformational"
                )
            )
            print(f"Drift Score: [{bar}] {drift_score:.0%} ({interpretation})")
            print()

            if drift["changed_values"]:
                print("## Changed Values")
                for v in drift["changed_values"]:
                    change_icon = "+" if v["change"] == "new" else "~"
                    print(f"  {change_icon} {v['name']}: {v['statement'][:50]}")
                print()

            if drift["evolved_beliefs"]:
                print("## New/Evolved Beliefs")
                for b in drift["evolved_beliefs"]:
                    print(f"  • {b['statement'][:60]} ({b['confidence']:.0%})")
                print()

            if drift["new_experiences"]:
                print("## Recent Significant Experiences")
                for ep in drift["new_experiences"]:
                    outcome_icon = "✓" if ep["outcome"] == "success" else "○"
                    print(f"  {outcome_icon} {ep['objective'][:50]} ({ep['date']})")
                    if ep.get("lessons"):
                        print(f"      → {ep['lessons'][0][:50]}")
                print()
