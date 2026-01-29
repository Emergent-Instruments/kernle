"""Knowledge mapping mixin for Kernle.

This module provides meta-cognition capabilities - self-awareness
of knowledge domains, competence boundaries, and learning opportunities.
"""

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from kernle.core import Kernle


class KnowledgeMixin:
    """Mixin providing knowledge mapping and meta-cognition capabilities.

    Enables understanding of:
    - What domains I have knowledge about
    - How confident I am in each domain
    - Where my knowledge gaps are
    - What I should learn next
    """

    def _extract_domains_from_tags(self: "Kernle") -> Dict[str, Dict[str, Any]]:
        """Extract knowledge domains from tags across all memory types.

        Returns a dict mapping domain names to their statistics.
        """
        domain_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "belief_count": 0,
                "belief_confidences": [],
                "episode_count": 0,
                "episode_outcomes": [],
                "note_count": 0,
                "goal_count": 0,
                "last_updated": None,
                "tags": set(),
            }
        )

        # Process beliefs
        beliefs = self._storage.get_beliefs(limit=1000)
        for belief in beliefs:
            # Use belief_type as a domain indicator
            domain = belief.belief_type or "general"
            domain_stats[domain]["belief_count"] += 1
            domain_stats[domain]["belief_confidences"].append(belief.confidence)
            if belief.created_at:
                if (
                    domain_stats[domain]["last_updated"] is None
                    or belief.created_at > domain_stats[domain]["last_updated"]
                ):
                    domain_stats[domain]["last_updated"] = belief.created_at

        # Process episodes - extract domains from tags
        episodes = self._storage.get_episodes(limit=1000)
        for episode in episodes:
            tags = episode.tags or []
            # Skip checkpoint tags
            tags = [
                t
                for t in tags
                if t not in ("checkpoint", "working_state", "auto-captured", "manual")
            ]

            if tags:
                for tag in tags:
                    domain_stats[tag]["episode_count"] += 1
                    domain_stats[tag]["episode_outcomes"].append(episode.outcome_type or "partial")
                    domain_stats[tag]["tags"].add(tag)
                    if episode.created_at:
                        if (
                            domain_stats[tag]["last_updated"] is None
                            or episode.created_at > domain_stats[tag]["last_updated"]
                        ):
                            domain_stats[tag]["last_updated"] = episode.created_at
            else:
                # No tags - count in general
                domain_stats["general"]["episode_count"] += 1
                domain_stats["general"]["episode_outcomes"].append(
                    episode.outcome_type or "partial"
                )

        # Process notes
        notes = self._storage.get_notes(limit=1000)
        for note in notes:
            tags = note.tags or []
            if tags:
                for tag in tags:
                    domain_stats[tag]["note_count"] += 1
                    domain_stats[tag]["tags"].add(tag)
                    if note.created_at:
                        if (
                            domain_stats[tag]["last_updated"] is None
                            or note.created_at > domain_stats[tag]["last_updated"]
                        ):
                            domain_stats[tag]["last_updated"] = note.created_at
            else:
                domain_stats["general"]["note_count"] += 1

        # Process goals
        goals = self._storage.get_goals(status=None, limit=1000)
        for goal in goals:
            # Extract domain from goal title (simplified)
            words = goal.title.lower().split()[:2]  # First two words as domain hint
            if words:
                domain = words[0]
                domain_stats[domain]["goal_count"] += 1

        return dict(domain_stats)

    def _calculate_coverage(self: "Kernle", stats: Dict[str, Any]) -> str:
        """Calculate coverage level based on domain statistics."""
        total_items = stats["belief_count"] + stats["episode_count"] + stats["note_count"]

        if total_items == 0:
            return "none"
        elif total_items < 3:
            return "low"
        elif total_items < 10:
            return "medium"
        else:
            return "high"

    def get_knowledge_map(self: "Kernle") -> Dict[str, Any]:
        """Map of knowledge domains with coverage assessment.

        Analyzes beliefs, episodes, and notes to understand what
        domains I have knowledge about and how confident I am.

        Returns:
            {
                "domains": [
                    {
                        "name": "Python programming",
                        "belief_count": 15,
                        "avg_confidence": 0.82,
                        "episode_count": 23,
                        "note_count": 5,
                        "goal_count": 2,
                        "coverage": "high",  # high/medium/low/none
                        "last_updated": datetime
                    },
                    ...
                ],
                "blind_spots": ["GraphQL", "Kubernetes"],  # domains with nothing
                "uncertain_areas": ["Docker networking"]  # low confidence
            }
        """
        domain_stats = self._extract_domains_from_tags()

        domains = []
        uncertain_areas = []

        for name, stats in domain_stats.items():
            confidences = stats["belief_confidences"]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            coverage = self._calculate_coverage(stats)

            domain_info = {
                "name": name,
                "belief_count": stats["belief_count"],
                "avg_confidence": round(avg_confidence, 2),
                "episode_count": stats["episode_count"],
                "note_count": stats["note_count"],
                "goal_count": stats["goal_count"],
                "coverage": coverage,
                "last_updated": (
                    stats["last_updated"].isoformat() if stats["last_updated"] else None
                ),
            }
            domains.append(domain_info)

            # Track uncertain areas (has beliefs but low confidence)
            if stats["belief_count"] > 0 and avg_confidence < 0.5:
                uncertain_areas.append(name)

        # Sort domains by coverage (high first) then by item count
        coverage_order = {"high": 0, "medium": 1, "low": 2, "none": 3}
        domains.sort(
            key=lambda d: (
                coverage_order.get(d["coverage"], 3),
                -(d["belief_count"] + d["episode_count"] + d["note_count"]),
            )
        )

        # Blind spots are harder to detect without a reference list
        # For now, we identify domains with very little data that were mentioned
        blind_spots = [
            d["name"]
            for d in domains
            if d["coverage"] == "none" or (d["coverage"] == "low" and d["avg_confidence"] == 0)
        ]

        return {
            "domains": domains,
            "blind_spots": blind_spots,
            "uncertain_areas": uncertain_areas,
            "total_domains": len(domains),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def detect_knowledge_gaps(self: "Kernle", query: str) -> Dict[str, Any]:
        """Analyze if I have knowledge relevant to a query.

        Searches memory to determine what I know about a topic
        and identifies gaps in my knowledge.

        Args:
            query: The query to check knowledge for

        Returns:
            {
                "has_relevant_knowledge": bool,
                "relevant_beliefs": [...],
                "relevant_episodes": [...],
                "relevant_notes": [...],
                "confidence": float,
                "gaps": ["specific thing I don't know about"],
                "recommendation": "I can help" | "I should learn more" | "Ask someone else"
            }
        """
        # Search for relevant memories
        results = self._storage.search(query, limit=20)

        relevant_beliefs = []
        relevant_episodes = []
        relevant_notes = []
        confidences = []

        for result in results:
            record = result.record
            record_type = result.record_type

            if record_type == "belief":
                relevant_beliefs.append(
                    {
                        "statement": record.statement,
                        "confidence": record.confidence,
                        "type": record.belief_type,
                    }
                )
                confidences.append(record.confidence)
            elif record_type == "episode":
                relevant_episodes.append(
                    {
                        "objective": record.objective,
                        "outcome": record.outcome,
                        "outcome_type": record.outcome_type,
                        "lessons": record.lessons,
                    }
                )
                confidences.append(getattr(record, "confidence", 0.8))
            elif record_type == "note":
                relevant_notes.append(
                    {
                        "content": record.content[:200],
                        "type": record.note_type,
                        "tags": record.tags,
                    }
                )
                confidences.append(getattr(record, "confidence", 0.8))

        # Calculate overall confidence
        has_relevant = len(results) > 0
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Identify gaps based on query analysis vs what we found
        gaps = []
        query_words = set(query.lower().split())
        found_topics = set()

        for b in relevant_beliefs:
            found_topics.update(b["statement"].lower().split())
        for e in relevant_episodes:
            found_topics.update(e["objective"].lower().split())

        # Words in query not found in results might indicate gaps
        potential_gaps = (
            query_words
            - found_topics
            - {"how", "do", "i", "what", "is", "the", "a", "to", "for", "and", "or"}
        )
        if potential_gaps and len(results) < 3:
            gaps = list(potential_gaps)[:3]

        # Determine recommendation
        if not has_relevant:
            recommendation = "Ask someone else"
        elif avg_confidence < 0.5:
            recommendation = "I should learn more"
        elif len(results) < 3:
            recommendation = "I have limited knowledge - proceed with caution"
        else:
            recommendation = "I can help"

        return {
            "has_relevant_knowledge": has_relevant,
            "relevant_beliefs": relevant_beliefs[:5],
            "relevant_episodes": relevant_episodes[:5],
            "relevant_notes": relevant_notes[:5],
            "confidence": round(avg_confidence, 2),
            "gaps": gaps,
            "recommendation": recommendation,
            "search_results_count": len(results),
        }

    def get_competence_boundaries(self: "Kernle") -> Dict[str, Any]:
        """What am I good at vs not good at?

        Analyzes belief confidence distribution, episode outcomes,
        and domain coverage to identify areas of strength and weakness.

        Returns:
            {
                "strengths": [
                    {"domain": "Python", "confidence": 0.9, "success_rate": 0.85},
                    ...
                ],
                "weaknesses": [
                    {"domain": "Docker", "confidence": 0.3, "success_rate": 0.4},
                    ...
                ],
                "overall_confidence": float,
                "success_rate": float,
                "experience_depth": int,  # total episodes
                "knowledge_breadth": int,  # number of domains
            }
        """
        domain_stats = self._extract_domains_from_tags()

        strengths = []
        weaknesses = []
        all_confidences = []
        all_outcomes = []

        for domain_name, stats in domain_stats.items():
            # Skip meta domains
            if domain_name in ("general", "manual", "auto-captured"):
                continue

            confidences = stats["belief_confidences"]
            outcomes = stats["episode_outcomes"]

            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
            success_count = outcomes.count("success")
            success_rate = success_count / len(outcomes) if outcomes else 0.5

            all_confidences.extend(confidences)
            all_outcomes.extend(outcomes)

            total_items = stats["belief_count"] + stats["episode_count"] + stats["note_count"]

            # Need at least some data to make a judgment
            if total_items < 2:
                continue

            domain_info = {
                "domain": domain_name,
                "confidence": round(avg_confidence, 2),
                "success_rate": round(success_rate, 2),
                "episode_count": stats["episode_count"],
                "belief_count": stats["belief_count"],
            }

            # Classify as strength or weakness
            if avg_confidence >= 0.7 and success_rate >= 0.6:
                strengths.append(domain_info)
            elif avg_confidence < 0.5 or success_rate < 0.4:
                weaknesses.append(domain_info)

        # Sort by confidence/success
        strengths.sort(key=lambda x: (x["confidence"], x["success_rate"]), reverse=True)
        weaknesses.sort(key=lambda x: (x["confidence"], x["success_rate"]))

        # Calculate overall metrics
        overall_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.5
        overall_success = all_outcomes.count("success") / len(all_outcomes) if all_outcomes else 0.5

        return {
            "strengths": strengths[:10],
            "weaknesses": weaknesses[:10],
            "overall_confidence": round(overall_confidence, 2),
            "success_rate": round(overall_success, 2),
            "experience_depth": len(all_outcomes),
            "knowledge_breadth": len(
                [d for d in domain_stats if d not in ("general", "manual", "auto-captured")]
            ),
        }

    def identify_learning_opportunities(self: "Kernle", limit: int = 5) -> List[Dict[str, Any]]:
        """What should I learn next?

        Identifies learning opportunities based on:
        - Low-coverage domains that are referenced often
        - Uncertain beliefs that affect decisions
        - Failed episodes that could benefit from more knowledge

        Args:
            limit: Maximum opportunities to return

        Returns:
            List of learning opportunities with priority and reasoning
        """
        opportunities = []

        domain_stats = self._extract_domains_from_tags()

        # 1. Low-coverage but frequently referenced domains
        for domain_name, stats in domain_stats.items():
            if domain_name in ("general", "manual", "auto-captured"):
                continue

            coverage = self._calculate_coverage(stats)
            reference_count = stats["episode_count"] + stats["note_count"]

            if coverage in ("low", "none") and reference_count > 0:
                opportunities.append(
                    {
                        "type": "low_coverage_domain",
                        "domain": domain_name,
                        "reason": f"Referenced {reference_count} times but only {stats['belief_count']} beliefs",
                        "priority": "high" if reference_count > 3 else "medium",
                        "suggested_action": f"Research and form beliefs about {domain_name}",
                    }
                )

        # 2. Uncertain beliefs that might affect decisions
        beliefs = self._storage.get_beliefs(limit=1000)
        low_confidence_beliefs = [
            b for b in beliefs if b.confidence < 0.5 and not getattr(b, "deleted", False)
        ]

        for belief in low_confidence_beliefs[:3]:
            opportunities.append(
                {
                    "type": "uncertain_belief",
                    "domain": belief.belief_type or "general",
                    "reason": f"Belief with only {belief.confidence:.0%} confidence: '{belief.statement[:50]}...'",
                    "priority": "medium",
                    "suggested_action": "Verify or update this belief with evidence",
                }
            )

        # 3. Failed episodes indicating knowledge gaps
        episodes = self._storage.get_episodes(limit=100)
        failed_episodes = [
            e
            for e in episodes
            if e.outcome_type == "failure" and e.tags and "checkpoint" not in e.tags
        ]

        # Group failures by domain
        failure_domains = {}
        for ep in failed_episodes:
            for tag in ep.tags or []:
                if tag not in ("manual", "auto-captured"):
                    failure_domains[tag] = failure_domains.get(tag, 0) + 1

        for domain, count in sorted(failure_domains.items(), key=lambda x: -x[1])[:3]:
            opportunities.append(
                {
                    "type": "repeated_failures",
                    "domain": domain,
                    "reason": f"{count} failed episodes in {domain}",
                    "priority": "high" if count > 2 else "medium",
                    "suggested_action": f"Study {domain} to improve success rate",
                }
            )

        # 4. Areas with no recent activity (might be getting stale)
        now = datetime.now(timezone.utc)
        stale_threshold = timedelta(days=30)

        for domain_name, stats in domain_stats.items():
            if domain_name in ("general", "manual", "auto-captured"):
                continue

            coverage = self._calculate_coverage(stats)
            if coverage in ("medium", "high") and stats["last_updated"]:
                age = now - stats["last_updated"]
                if age > stale_threshold:
                    opportunities.append(
                        {
                            "type": "stale_knowledge",
                            "domain": domain_name,
                            "reason": f"No updates in {age.days} days - knowledge may be outdated",
                            "priority": "low",
                            "suggested_action": f"Review and refresh knowledge about {domain_name}",
                        }
                    )

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        opportunities.sort(key=lambda x: priority_order.get(x["priority"], 2))

        return opportunities[:limit]
