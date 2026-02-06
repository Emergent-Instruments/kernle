"""Identity and consolidation commands for Kernle CLI."""

import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kernle import Kernle


def cmd_promote(args, k: "Kernle"):
    """Promote recurring patterns from episodes into beliefs."""
    result = k.promote(
        auto=getattr(args, "auto", False),
        min_occurrences=getattr(args, "min_occurrences", 2),
        min_episodes=getattr(args, "min_episodes", 3),
        confidence=getattr(args, "confidence", 0.7),
        limit=getattr(args, "limit", 50),
    )

    if getattr(args, "json", False):
        print(json.dumps(result, indent=2, default=str))
        return

    print("## Promotion Results")
    print(f"Episodes scanned: {result['episodes_scanned']}")
    print(f"Patterns found: {result['patterns_found']}")
    print()

    if result.get("message"):
        print(result["message"])
        return

    if not result["suggestions"]:
        print("No recurring patterns found. Keep building experiences!")
        return

    auto = getattr(args, "auto", False)
    if auto:
        print(f"Beliefs created: {result['beliefs_created']}")
        print()

    for i, s in enumerate(result["suggestions"], 1):
        status = ""
        if s.get("promoted"):
            status = f" ✅ → belief {s['belief_id'][:8]}"
        elif s.get("skipped"):
            status = " ⏭️  (similar belief exists)"

        print(f'{i}. "{s["lesson"]}" (×{s["count"]}){status}')
        if s["source_episodes"]:
            ep_refs = ", ".join(eid[:8] for eid in s["source_episodes"])
            print(f"   Source episodes: {ep_refs}")

    if not auto and result["suggestions"]:
        # Show how to promote manually
        print()
        print("---")
        print("To promote automatically:")
        print(f"  kernle -a {k.agent_id} promote --auto")
        print()
        print("Or promote specific patterns manually:")
        print(
            f'  kernle -a {k.agent_id} belief add "<statement>" '
            f"--confidence 0.7 --source promotion"
        )


def _print_drive_pattern_analysis(episodes, k: "Kernle"):
    """Surface behavioral evidence for undeclared drives.

    Counts episode tags/themes from the last 30 days, compares them
    to declared drives, and highlights patterns without matching drives.
    Scaffolds surface patterns -- the entity decides what to do.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=30)

    # Filter to recent episodes (last 30 days)
    recent = [ep for ep in episodes if ep.created_at and ep.created_at >= cutoff]

    if not recent:
        return

    # Collect tags and emotional_tags from recent episodes
    tag_counts: Counter = Counter()
    for ep in recent:
        if ep.tags:
            for tag in ep.tags:
                tag_counts[tag.lower()] += 1
        if ep.emotional_tags:
            for tag in ep.emotional_tags:
                tag_counts[tag.lower()] += 1

    if not tag_counts:
        return

    # Get declared drives
    declared_drives = k._storage.get_drives()
    declared_types = {d.drive_type.lower() for d in declared_drives}
    declared_focus = set()
    for d in declared_drives:
        if d.focus_areas:
            for area in d.focus_areas:
                declared_focus.add(area.lower())

    # Find unmatched patterns (tags that don't align with any declared drive)
    unmatched = []
    for tag, count in tag_counts.most_common():
        if count < 2:
            continue
        if tag in declared_types or tag in declared_focus:
            continue
        unmatched.append((tag, count))

    print("### DRIVE PATTERN ANALYSIS:")
    print(f"Based on {len(recent)} episodes from the last 30 days.")
    print()

    if declared_drives:
        print("Declared drives:")
        for d in declared_drives:
            focus = f" (focus: {', '.join(d.focus_areas)})" if d.focus_areas else ""
            print(f"  - {d.drive_type} (intensity {d.intensity:.0%}){focus}")
        print()

    # Show top themes
    top_themes = tag_counts.most_common(10)
    if top_themes:
        print("Top themes/tags in recent episodes:")
        for tag, count in top_themes:
            matched = tag in declared_types or tag in declared_focus
            marker = "" if matched else " *"
            print(f"  - {tag}: {count} occurrences{marker}")
        if any(tag not in declared_types and tag not in declared_focus for tag, _ in top_themes):
            print("  (* = no matching declared drive)")
        print()

    if unmatched:
        print("Potential undeclared drives (recurring themes without matching drive):")
        for tag, count in unmatched[:5]:
            print(f'  - "{tag}" ({count} episodes)')
        print()
        print(
            "Consider: Do these patterns reflect emerging drives? "
            "You decide whether to declare them."
        )
        print()
    elif tag_counts:
        print("All recurring themes align with declared drives.")
        print()


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

    # High-Arousal Episodes section
    high_arousal = [ep for ep in episodes if ep.emotional_arousal > 0.6]
    if high_arousal:
        high_arousal.sort(key=lambda ep: ep.emotional_arousal, reverse=True)
        print("### HIGH-AROUSAL EPISODES (may be worth extra reflection):")
        for ep in high_arousal:
            date_str = ep.created_at.strftime("%Y-%m-%d") if ep.created_at else "unknown"
            valence_label = (
                "positive"
                if ep.emotional_valence > 0.2
                else "negative" if ep.emotional_valence < -0.2 else "neutral"
            )
            print(
                f'- [{date_str}] "{ep.objective}" '
                f"(arousal: {ep.emotional_arousal:.2f}, valence: {valence_label})"
            )
            if ep.emotional_tags:
                print(f"  Feelings: {', '.join(ep.emotional_tags)}")
        print()

    # Drive Pattern Analysis section
    _print_drive_pattern_analysis(episodes, k)

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
