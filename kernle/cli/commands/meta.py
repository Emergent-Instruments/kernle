"""Meta-memory commands for Kernle CLI."""

import json
from typing import TYPE_CHECKING

from kernle.cli.commands.helpers import validate_input

if TYPE_CHECKING:
    from kernle import Kernle


def cmd_meta(args, k: "Kernle"):
    """Handle meta-memory subcommands."""
    if args.meta_action == "confidence":
        memory_type = args.type
        memory_id = args.id

        confidence = k.get_memory_confidence(memory_type, memory_id)
        if confidence < 0:
            print(f"✗ Memory {memory_type}:{memory_id[:8]}... not found")
        else:
            bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
            print(f"Confidence: [{bar}] {confidence:.0%}")

    elif args.meta_action == "verify":
        memory_type = args.type
        memory_id = args.id
        evidence = args.evidence

        if k.verify_memory(memory_type, memory_id, evidence):
            print(f"✓ Memory {memory_type}:{memory_id[:8]}... verified")
            new_conf = k.get_memory_confidence(memory_type, memory_id)
            print(f"  New confidence: {new_conf:.0%}")
        else:
            print(f"✗ Could not verify memory {memory_type}:{memory_id[:8]}...")

    elif args.meta_action == "lineage":
        memory_type = args.type
        memory_id = args.id

        lineage = k.get_memory_lineage(memory_type, memory_id)

        if args.json:
            print(json.dumps(lineage, indent=2, default=str))
        else:
            if lineage.get("error"):
                print(f"✗ {lineage['error']}")
                return

            print(f"Lineage for {memory_type}:{memory_id[:8]}...")
            print("=" * 40)
            print(f"Source Type: {lineage['source_type']}")
            print(f"Current Confidence: {lineage.get('current_confidence', 'N/A')}")

            if lineage.get("source_episodes"):
                print("\nSupporting Episodes:")
                for ep_id in lineage["source_episodes"]:
                    print(f"  • {ep_id}")

            if lineage.get("derived_from"):
                print("\nDerived From:")
                for ref in lineage["derived_from"]:
                    print(f"  • {ref}")

            if lineage.get("confidence_history"):
                print("\nConfidence History:")
                for change in lineage["confidence_history"][-5:]:
                    print(
                        f"  {change.get('timestamp', 'N/A')[:10]}: "
                        f"{change.get('old', 'N/A')} → {change.get('new', 'N/A')} "
                        f"({change.get('reason', 'no reason')})"
                    )

    elif args.meta_action == "uncertain":
        threshold = args.threshold
        results = k.get_uncertain_memories(threshold, limit=args.limit)

        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            if not results:
                print(f"No memories below {threshold:.0%} confidence threshold.")
                return

            print(f"Uncertain Memories (confidence < {threshold:.0%})")
            print("=" * 50)
            for mem in results:
                bar = "█" * int(mem["confidence"] * 10) + "░" * (10 - int(mem["confidence"] * 10))
                print(f"[{bar}] {mem['confidence']:.0%} [{mem['type']}] {mem['summary'][:40]}")
                print(f"         ID: {mem['id'][:12]}...  ({mem['created_at']})")

    elif args.meta_action == "propagate":
        memory_type = args.type
        memory_id = args.id

        result = k.propagate_confidence(memory_type, memory_id)

        if result.get("error"):
            print(f"✗ {result['error']}")
        else:
            print(f"✓ Propagated confidence from {memory_type}:{memory_id[:8]}...")
            print(f"  Source confidence: {result['source_confidence']:.0%}")
            print(f"  Derived memories updated: {result['updated']}")

    elif args.meta_action == "source":
        memory_type = args.type
        memory_id = args.id
        source_type = args.source

        if k.set_memory_source(
            memory_type,
            memory_id,
            source_type,
            source_episodes=args.episodes,
            derived_from=args.derived,
        ):
            print(f"✓ Source set for {memory_type}:{memory_id[:8]}...")
            print(f"  Source type: {source_type}")
            if args.episodes:
                print(f"  Source episodes: {', '.join(args.episodes)}")
            if args.derived:
                print(f"  Derived from: {', '.join(args.derived)}")
        else:
            print(f"✗ Could not set source for {memory_type}:{memory_id[:8]}...")

    # Meta-cognition commands
    elif args.meta_action == "knowledge":
        knowledge_map = k.get_knowledge_map()

        if args.json:
            print(json.dumps(knowledge_map, indent=2, default=str))
        else:
            print("Knowledge Map")
            print("=" * 60)
            print()

            domains = knowledge_map.get("domains", [])
            if not domains:
                print("No knowledge domains found yet.")
                print("Add beliefs, episodes, and notes to build your knowledge base.")
                return

            # Coverage icons
            coverage_icons = {"high": "🟢", "medium": "🟡", "low": "🟠", "none": "⚫"}

            print("## Domains")
            print()
            for domain in domains[:15]:
                icon = coverage_icons.get(domain["coverage"], "⚫")
                conf_bar = "█" * int(domain["avg_confidence"] * 5) + "░" * (
                    5 - int(domain["avg_confidence"] * 5)
                )
                print(f"{icon} {domain['name']:<20} [{conf_bar}] {domain['avg_confidence']:.0%}")
                print(
                    f"   Beliefs: {domain['belief_count']:>3}  Episodes: {domain['episode_count']:>3}  Notes: {domain['note_count']:>3}"
                )
                if domain.get("last_updated"):
                    print(f"   Last updated: {domain['last_updated'][:10]}")
                print()

            # Blind spots
            blind_spots = knowledge_map.get("blind_spots", [])
            if blind_spots:
                print("## Blind Spots (little/no knowledge)")
                for spot in blind_spots[:5]:
                    print(f"  ⚫ {spot}")
                print()

            # Uncertain areas
            uncertain = knowledge_map.get("uncertain_areas", [])
            if uncertain:
                print("## Uncertain Areas (low confidence)")
                for area in uncertain[:5]:
                    print(f"  🟠 {area}")
                print()

            print(f"Total domains: {knowledge_map.get('total_domains', 0)}")

    elif args.meta_action == "gaps":
        query = validate_input(args.query, "query", 500)
        result = k.detect_knowledge_gaps(query)

        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print(f'Knowledge Gap Analysis: "{query}"')
            print("=" * 60)
            print()

            # Recommendation with icon
            rec = result["recommendation"]
            if rec == "I can help":
                rec_icon = "🟢"
            elif rec == "I have limited knowledge - proceed with caution":
                rec_icon = "🟡"
            elif rec == "I should learn more":
                rec_icon = "🟠"
            else:  # Ask someone else
                rec_icon = "🔴"

            print(f"Recommendation: {rec_icon} {rec}")
            print(f"Confidence: {result['confidence']:.0%}")
            print(f"Relevant results: {result['search_results_count']}")
            print()

            # Relevant beliefs
            if result.get("relevant_beliefs"):
                print("## Relevant Beliefs")
                for belief in result["relevant_beliefs"]:
                    conf = belief.get("confidence", 0)
                    bar = "█" * int(conf * 5) + "░" * (5 - int(conf * 5))
                    print(f"  [{bar}] {belief['statement'][:60]}...")
                print()

            # Relevant episodes
            if result.get("relevant_episodes"):
                print("## Relevant Episodes")
                for ep in result["relevant_episodes"]:
                    outcome = "✓" if ep.get("outcome_type") == "success" else "○"
                    print(f"  {outcome} {ep['objective'][:55]}...")
                    if ep.get("lessons"):
                        print(f"      → {ep['lessons'][0][:50]}..." if ep["lessons"] else "")
                print()

            # Knowledge gaps
            if result.get("gaps"):
                print("## Potential Gaps")
                for gap in result["gaps"]:
                    print(f"  ❓ {gap}")
                print()

    elif args.meta_action == "boundaries":
        boundaries = k.get_competence_boundaries()

        if args.json:
            print(json.dumps(boundaries, indent=2, default=str))
        else:
            print("Competence Boundaries")
            print("=" * 60)
            print()

            # Overall stats
            conf = boundaries["overall_confidence"]
            success = boundaries["success_rate"]
            conf_bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
            success_bar = "█" * int(success * 10) + "░" * (10 - int(success * 10))

            print(f"Overall Confidence:  [{conf_bar}] {conf:.0%}")
            print(f"Overall Success:     [{success_bar}] {success:.0%}")
            print(f"Experience Depth:    {boundaries['experience_depth']} episodes")
            print(f"Knowledge Breadth:   {boundaries['knowledge_breadth']} domains")
            print()

            # Strengths
            strengths = boundaries.get("strengths", [])
            if strengths:
                print("## Strengths 💪")
                for s in strengths[:5]:
                    conf_bar = "█" * int(s["confidence"] * 5) + "░" * (5 - int(s["confidence"] * 5))
                    print(
                        f"  🟢 {s['domain']:<20} [{conf_bar}] {s['confidence']:.0%} conf, {s['success_rate']:.0%} success"
                    )
                print()

            # Weaknesses
            weaknesses = boundaries.get("weaknesses", [])
            if weaknesses:
                print("## Weaknesses 📚 (learning opportunities)")
                for w in weaknesses[:5]:
                    conf_bar = "█" * int(w["confidence"] * 5) + "░" * (5 - int(w["confidence"] * 5))
                    print(
                        f"  🟠 {w['domain']:<20} [{conf_bar}] {w['confidence']:.0%} conf, {w['success_rate']:.0%} success"
                    )
                print()

            if not strengths and not weaknesses:
                print("Not enough data to determine strengths and weaknesses yet.")
                print("Record more episodes and beliefs to build your competence profile.")

    elif args.meta_action == "learn":
        opportunities = k.identify_learning_opportunities(limit=args.limit)

        if args.json:
            print(json.dumps(opportunities, indent=2, default=str))
        else:
            print("Learning Opportunities")
            print("=" * 60)
            print()

            if not opportunities:
                print("✨ No urgent learning needs identified!")
                print("Your knowledge base appears well-maintained.")
                return

            priority_icons = {"high": "🔴", "medium": "🟡", "low": "🟢"}
            type_icons = {
                "low_coverage_domain": "📚",
                "uncertain_belief": "❓",
                "repeated_failures": "⚠️",
                "stale_knowledge": "📅",
            }

            for i, opp in enumerate(opportunities, 1):
                priority_icon = priority_icons.get(opp["priority"], "⚪")
                type_icon = type_icons.get(opp["type"], "•")

                print(
                    f"{i}. {priority_icon} [{opp['priority'].upper():>6}] {type_icon} {opp['domain']}"
                )
                print(f"   Reason: {opp['reason']}")
                print(f"   Action: {opp['suggested_action']}")
                print()
