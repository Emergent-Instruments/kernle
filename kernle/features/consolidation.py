"""Advanced consolidation scaffolds for Kernle.

This module provides three interconnected scaffold enhancements for deeper
pattern recognition during memory consolidation:

1. Cross-domain pattern scaffolding: Groups episodes by domain tags and
   detects structural similarities in outcomes across domains.

2. Belief-to-value promotion: Identifies beliefs that have reached
   value-level stability (long-lived, reinforced, multi-domain) and
   suggests promotion.

3. Entity model-to-belief promotion: Detects when multiple entity models
   point toward the same generalization and suggests belief formation.

All scaffolds follow the SI autonomy principle: they surface patterns and
suggestions, but the agent decides what to do with them.
"""

from collections import defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from kernle.core import Kernle


# Thresholds for belief-to-value promotion
BELIEF_VALUE_MIN_AGE_DAYS = 180  # 6 months
BELIEF_VALUE_MIN_REINFORCEMENTS = 3
BELIEF_VALUE_MIN_DOMAINS = 3
BELIEF_VALUE_MIN_CONFIDENCE = 0.8

# Thresholds for cross-domain pattern detection
CROSS_DOMAIN_MIN_EPISODES = 2
CROSS_DOMAIN_MIN_DOMAINS = 2

# Thresholds for entity model-to-belief promotion
ENTITY_MODEL_MIN_ENTITIES = 2
ENTITY_MODEL_MIN_EPISODES = 2


class ConsolidationMixin:
    """Mixin providing advanced consolidation scaffold capabilities."""

    def scaffold_cross_domain_patterns(
        self: "Kernle",
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Detect structural similarities in outcomes across different domains.

        Groups episodes by their tags (used as domain proxies), then identifies
        outcome patterns that repeat across multiple domains.

        Args:
            limit: Maximum episodes to analyze

        Returns:
            Dict with:
            - episodes_scanned: number of episodes analyzed
            - domains_found: number of distinct tag domains
            - patterns: list of cross-domain pattern dicts
            - scaffold: formatted text scaffold for reflection
        """
        episodes = self._storage.get_episodes(limit=limit)
        episodes = [ep for ep in episodes if not ep.is_forgotten]

        if not episodes:
            return {
                "episodes_scanned": 0,
                "domains_found": 0,
                "patterns": [],
                "scaffold": "No episodes available for cross-domain analysis.",
            }

        # Group episodes by tag (each tag is treated as a domain)
        domain_episodes: Dict[str, List] = defaultdict(list)
        for ep in episodes:
            tags = ep.tags or []
            # Also include context_tags as domain indicators
            ctx_tags = getattr(ep, "context_tags", None) or []
            all_tags = set(tags + ctx_tags)
            if not all_tags:
                all_tags = {"untagged"}
            for tag in all_tags:
                domain_episodes[tag].append(ep)

        # For each domain, build outcome-to-lesson mappings
        # Structure: {domain: {outcome_type: [lessons]}}
        domain_outcome_lessons: Dict[str, Dict[str, List[str]]] = {}
        for domain, eps in domain_episodes.items():
            outcome_map: Dict[str, List[str]] = defaultdict(list)
            for ep in eps:
                otype = ep.outcome_type or "unknown"
                if ep.lessons:
                    for lesson in ep.lessons:
                        outcome_map[otype].append(lesson.strip().lower())
                else:
                    # Even without explicit lessons, track the pattern
                    summary = ep.objective[:80].strip().lower()
                    outcome_map[otype].append(f"[{summary}]")
            domain_outcome_lessons[domain] = dict(outcome_map)

        # Detect cross-domain patterns:
        # A pattern is "something that leads to the same outcome in 2+ domains"
        # We look for similar lessons/objectives across domains with same outcome
        patterns = []

        # Build a normalized lesson -> (domain, outcome, episode_ids) index
        lesson_domain_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for domain, eps in domain_episodes.items():
            for ep in eps:
                otype = ep.outcome_type or "unknown"
                if ep.lessons:
                    for lesson in ep.lessons:
                        normalized = lesson.strip().lower()
                        lesson_domain_map[normalized].append(
                            {
                                "domain": domain,
                                "outcome": otype,
                                "episode_id": ep.id,
                                "objective": ep.objective,
                            }
                        )

        # Find lessons that appear in 2+ domains with the same outcome
        for lesson, occurrences in lesson_domain_map.items():
            domains_by_outcome: Dict[str, set] = defaultdict(set)
            episodes_by_outcome: Dict[str, List[str]] = defaultdict(list)
            for occ in occurrences:
                domains_by_outcome[occ["outcome"]].add(occ["domain"])
                episodes_by_outcome[occ["outcome"]].append(occ["episode_id"])

            for outcome, domains in domains_by_outcome.items():
                if len(domains) >= CROSS_DOMAIN_MIN_DOMAINS:
                    patterns.append(
                        {
                            "lesson": lesson,
                            "outcome": outcome,
                            "domains": sorted(domains),
                            "episode_count": len(episodes_by_outcome[outcome]),
                            "episode_ids": episodes_by_outcome[outcome][:10],
                        }
                    )

        # Also detect outcome-type patterns per domain (shortcutting -> failure)
        # Look for domains where a particular outcome type dominates
        outcome_patterns = []
        for domain, eps in domain_episodes.items():
            if len(eps) < CROSS_DOMAIN_MIN_EPISODES:
                continue
            outcome_counts: Dict[str, int] = defaultdict(int)
            for ep in eps:
                outcome_counts[ep.outcome_type or "unknown"] += 1

            total = len(eps)
            for otype, count in outcome_counts.items():
                ratio = count / total
                if ratio >= 0.6 and count >= CROSS_DOMAIN_MIN_EPISODES:
                    outcome_patterns.append(
                        {
                            "domain": domain,
                            "outcome": otype,
                            "count": count,
                            "total": total,
                            "ratio": round(ratio, 2),
                        }
                    )

        # Sort patterns by episode count (most evidence first)
        patterns.sort(key=lambda p: -p["episode_count"])
        patterns = patterns[:10]  # Cap output

        # Build scaffold text
        scaffold = self._format_cross_domain_scaffold(
            episodes, domain_episodes, patterns, outcome_patterns
        )

        return {
            "episodes_scanned": len(episodes),
            "domains_found": len(domain_episodes),
            "patterns": patterns,
            "outcome_patterns": outcome_patterns[:10],
            "scaffold": scaffold,
        }

    def _format_cross_domain_scaffold(
        self: "Kernle",
        episodes: list,
        domain_episodes: Dict[str, list],
        patterns: List[Dict],
        outcome_patterns: List[Dict],
    ) -> str:
        """Format cross-domain analysis into a reflection scaffold."""
        lines = []
        lines.append("## Cross-Domain Pattern Analysis")
        lines.append("")
        lines.append(
            f"Analyzed {len(episodes)} episodes across " f"{len(domain_episodes)} domains."
        )
        lines.append("")

        # Show domain summaries
        if domain_episodes:
            lines.append("### Domains:")
            for domain, eps in sorted(domain_episodes.items(), key=lambda x: -len(x[1]))[:10]:
                outcomes = defaultdict(int)
                for ep in eps:
                    outcomes[ep.outcome_type or "unknown"] += 1
                outcome_str = ", ".join(f"{k}: {v}" for k, v in sorted(outcomes.items()))
                lines.append(f"- **{domain}** ({len(eps)} episodes): {outcome_str}")
            lines.append("")

        # Show cross-domain patterns
        if patterns:
            lines.append("### Cross-Domain Patterns:")
            lines.append("")
            for p in patterns:
                domains_str = ", ".join(p["domains"])
                lines.append(
                    f'- "{p["lesson"]}" -> {p["outcome"]} '
                    f"(in {domains_str}, {p['episode_count']} episodes)"
                )
            lines.append("")
            lines.append(
                "**Reflection prompt:** Are these general patterns? " "Could they become beliefs?"
            )
            lines.append("")

        # Show outcome type patterns
        if outcome_patterns:
            lines.append("### Domain Outcome Tendencies:")
            lines.append("")
            for op in outcome_patterns:
                lines.append(
                    f"- **{op['domain']}**: {op['outcome']} in "
                    f"{op['count']}/{op['total']} episodes "
                    f"({op['ratio']:.0%})"
                )
            lines.append("")

        if not patterns and not outcome_patterns:
            lines.append(
                "No cross-domain patterns detected yet. "
                "More tagged episodes may reveal structural similarities."
            )
            lines.append("")

        return "\n".join(lines)

    def scaffold_belief_to_value(
        self: "Kernle",
        min_age_days: int = BELIEF_VALUE_MIN_AGE_DAYS,
        min_reinforcements: int = BELIEF_VALUE_MIN_REINFORCEMENTS,
        min_domains: int = BELIEF_VALUE_MIN_DOMAINS,
        min_confidence: float = BELIEF_VALUE_MIN_CONFIDENCE,
    ) -> Dict[str, Any]:
        """Identify beliefs that may have reached value-level stability.

        A belief is a candidate for value promotion when:
        - Active for >min_age_days (default 180 / 6 months)
        - Reinforced at least min_reinforcements times
        - Never contradicted (not superseded)
        - Appears across min_domains+ domains
        - Confidence >= min_confidence

        Args:
            min_age_days: Minimum age in days
            min_reinforcements: Minimum reinforcement count
            min_domains: Minimum domain count
            min_confidence: Minimum confidence threshold

        Returns:
            Dict with:
            - beliefs_scanned: number of beliefs analyzed
            - candidates: list of promotion candidate dicts
            - scaffold: formatted text scaffold for reflection
        """
        beliefs = self._storage.get_beliefs(limit=500, include_inactive=False)
        beliefs = [b for b in beliefs if b.is_active and not b.is_forgotten]

        # Also get existing values for dedup checking
        values = self._storage.get_values(limit=200)
        value_statements = {v.statement.strip().lower() for v in values}

        now = datetime.now(timezone.utc)
        candidates = []

        for belief in beliefs:
            # Check age
            created = belief.created_at
            if not created:
                continue
            # Ensure timezone-aware comparison
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            age_days = (now - created).days
            if age_days < min_age_days:
                continue

            # Check reinforcements
            if belief.times_reinforced < min_reinforcements:
                continue

            # Check not contradicted (superseded_by means it was replaced)
            if belief.superseded_by:
                continue

            # Check confidence
            if belief.confidence < min_confidence:
                continue

            # Check domain spread
            domains = set()
            if belief.source_domain:
                domains.add(belief.source_domain)
            if belief.cross_domain_applications:
                domains.update(belief.cross_domain_applications)
            if len(domains) < min_domains:
                continue

            # Check not already a value
            if belief.statement.strip().lower() in value_statements:
                continue

            candidates.append(
                {
                    "belief_id": belief.id,
                    "statement": belief.statement,
                    "confidence": belief.confidence,
                    "age_days": age_days,
                    "times_reinforced": belief.times_reinforced,
                    "domains": sorted(domains),
                    "belief_type": belief.belief_type,
                    "source_domain": belief.source_domain,
                    "abstraction_level": getattr(belief, "abstraction_level", "specific"),
                }
            )

        # Sort by reinforcement count (most stable first)
        candidates.sort(key=lambda c: -c["times_reinforced"])
        candidates = candidates[:10]  # Cap output

        scaffold = self._format_belief_to_value_scaffold(len(beliefs), candidates)

        return {
            "beliefs_scanned": len(beliefs),
            "candidates": candidates,
            "scaffold": scaffold,
        }

    def _format_belief_to_value_scaffold(
        self: "Kernle",
        beliefs_scanned: int,
        candidates: List[Dict],
    ) -> str:
        """Format belief-to-value analysis into a reflection scaffold."""
        lines = []
        lines.append("## Belief-to-Value Promotion Analysis")
        lines.append("")
        lines.append(f"Scanned {beliefs_scanned} active beliefs for value-level stability.")
        lines.append("")

        if candidates:
            lines.append(f"### {len(candidates)} Promotion Candidate(s):")
            lines.append("")
            for i, c in enumerate(candidates, 1):
                lines.append(f'**{i}. "{c["statement"]}"**')
                lines.append(
                    f"   - Active for {c['age_days']} days, "
                    f"reinforced {c['times_reinforced']} times, "
                    f"never contradicted"
                )
                lines.append(f"   - Appears across: {', '.join(c['domains'])}")
                lines.append(
                    f"   - Confidence: {c['confidence']:.0%}, "
                    f"type: {c['belief_type']}, "
                    f"abstraction: {c['abstraction_level']}"
                )
                lines.append("")
                lines.append("   This belief may have reached value-level stability.")
                lines.append(
                    '   Consider: `kernle value "<name>" ' f'"{c["statement"]}" --priority 80`'
                )
                lines.append("")
        else:
            lines.append("No beliefs currently meet value-promotion criteria.")
            lines.append("")
            lines.append(
                "Criteria: active >6 months, reinforced 3+ times, "
                "never contradicted, 3+ domains, confidence >= 80%."
            )
            lines.append("")

        return "\n".join(lines)

    def scaffold_entity_model_to_belief(
        self: "Kernle",
        min_entities: int = ENTITY_MODEL_MIN_ENTITIES,
        min_supporting_episodes: int = ENTITY_MODEL_MIN_EPISODES,
    ) -> Dict[str, Any]:
        """Detect when multiple entity models suggest a common generalization.

        Groups entity models by observation theme and identifies cases where
        similar observations about different entities could form a general belief.

        Args:
            min_entities: Minimum distinct entities with similar observations
            min_supporting_episodes: Minimum total supporting episodes

        Returns:
            Dict with:
            - models_scanned: number of entity models analyzed
            - generalizations: list of potential generalization dicts
            - scaffold: formatted text scaffold for reflection
        """
        models = self._storage.get_entity_models(limit=200)
        if not models:
            return {
                "models_scanned": 0,
                "generalizations": [],
                "scaffold": "No entity models available for analysis.",
            }

        # Group models by model_type and look for similar observations
        # We use a simple word-overlap approach for similarity
        type_groups: Dict[str, list] = defaultdict(list)
        for model in models:
            type_groups[model.model_type].append(model)

        generalizations = []

        for model_type, group_models in type_groups.items():
            if len(group_models) < min_entities:
                continue

            # Find clusters of similar observations using keyword overlap
            clusters = self._cluster_observations(group_models)

            for cluster in clusters:
                entities = {m.entity_name for m in cluster}
                if len(entities) < min_entities:
                    continue

                # Count total supporting episodes
                total_episodes = 0
                all_episode_ids = []
                for m in cluster:
                    eps = m.source_episodes or []
                    total_episodes += len(eps)
                    all_episode_ids.extend(eps)

                if total_episodes < min_supporting_episodes:
                    continue

                # Compute average confidence
                avg_conf = sum(m.confidence for m in cluster) / len(cluster)

                # Generate a generalization suggestion
                observations = [
                    {
                        "entity": m.entity_name,
                        "observation": m.observation,
                        "confidence": m.confidence,
                        "model_id": m.id,
                    }
                    for m in cluster
                ]

                generalizations.append(
                    {
                        "model_type": model_type,
                        "entities": sorted(entities),
                        "entity_count": len(entities),
                        "observations": observations[:10],
                        "total_episodes": total_episodes,
                        "episode_ids": all_episode_ids[:10],
                        "average_confidence": round(avg_conf, 2),
                        "model_ids": [m.id for m in cluster],
                    }
                )

        # Sort by entity count (broadest patterns first)
        generalizations.sort(key=lambda g: -g["entity_count"])
        generalizations = generalizations[:10]

        scaffold = self._format_entity_model_scaffold(len(models), generalizations)

        return {
            "models_scanned": len(models),
            "generalizations": generalizations,
            "scaffold": scaffold,
        }

    def _cluster_observations(
        self: "Kernle",
        models: list,
    ) -> List[list]:
        """Cluster entity models by observation similarity.

        Uses keyword overlap to find groups of models with similar
        observations. This is a simple heuristic approach that avoids
        requiring LLM calls.

        Args:
            models: List of EntityModel objects

        Returns:
            List of clusters (each cluster is a list of EntityModel)
        """
        if not models:
            return []

        # Extract keywords from each observation
        stop_words = {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "shall",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "about",
            "between",
            "under",
            "above",
            "up",
            "down",
            "out",
            "off",
            "over",
            "and",
            "but",
            "or",
            "nor",
            "not",
            "no",
            "so",
            "if",
            "then",
            "than",
            "too",
            "very",
            "just",
            "that",
            "this",
            "it",
            "its",
            "they",
            "them",
            "their",
            "he",
            "she",
            "his",
            "her",
            "we",
            "i",
        }

        def extract_keywords(text: str) -> set:
            words = set(text.lower().split())
            # Remove punctuation from words
            cleaned = set()
            for w in words:
                clean = "".join(c for c in w if c.isalnum())
                if clean and len(clean) > 2 and clean not in stop_words:
                    cleaned.add(clean)
            return cleaned

        model_keywords = [(m, extract_keywords(m.observation)) for m in models]

        # Simple greedy clustering: two models cluster if they share
        # at least 2 meaningful keywords
        min_shared = 2
        clusters: List[list] = []
        assigned = set()

        for i, (model_i, kw_i) in enumerate(model_keywords):
            if i in assigned:
                continue

            cluster = [model_i]
            assigned.add(i)

            for j, (model_j, kw_j) in enumerate(model_keywords):
                if j in assigned or j <= i:
                    continue
                shared = kw_i & kw_j
                if len(shared) >= min_shared:
                    cluster.append(model_j)
                    assigned.add(j)

            if len(cluster) >= 2:
                clusters.append(cluster)

        return clusters

    def _format_entity_model_scaffold(
        self: "Kernle",
        models_scanned: int,
        generalizations: List[Dict],
    ) -> str:
        """Format entity model analysis into a reflection scaffold."""
        lines = []
        lines.append("## Entity Model-to-Belief Analysis")
        lines.append("")
        lines.append(f"Scanned {models_scanned} entity models for generalizable patterns.")
        lines.append("")

        if generalizations:
            lines.append(f"### {len(generalizations)} Possible Generalization(s):")
            lines.append("")
            for i, g in enumerate(generalizations, 1):
                lines.append(
                    f"**{i}. Across {g['entity_count']} entities " f"({g['model_type']}):**"
                )
                lines.append("   Observations:")
                for obs in g["observations"][:5]:
                    lines.append(
                        f'   - {obs["entity"]}: "{obs["observation"]}" '
                        f'(confidence: {obs["confidence"]:.0%})'
                    )
                lines.append("")
                lines.append(
                    f"   Supported by {g['total_episodes']} episode(s), "
                    f"average confidence: {g['average_confidence']:.0%}"
                )
                lines.append("")

                # Build derived-from suggestion
                model_refs = " ".join(f"entity_model:{mid}" for mid in g["model_ids"][:5])
                lines.append(
                    f'   Consider: `kernle belief "<generalization>" '
                    f"--derived-from {model_refs}`"
                )
                lines.append("")
        else:
            lines.append("No generalizable patterns detected across entity models.")
            lines.append("")
            lines.append(
                "More entity model observations about different entities "
                "may reveal shared patterns."
            )
            lines.append("")

        return "\n".join(lines)

    def scaffold_advanced_consolidation(
        self: "Kernle",
        episode_limit: int = 100,
    ) -> Dict[str, Any]:
        """Run all three advanced consolidation scaffolds and combine results.

        This is the main entry point that produces a comprehensive
        consolidation scaffold covering cross-domain patterns,
        belief-to-value promotion, and entity model-to-belief promotion.

        Args:
            episode_limit: Maximum episodes to analyze for cross-domain

        Returns:
            Dict with combined results from all three scaffolds
        """
        cross_domain = self.scaffold_cross_domain_patterns(limit=episode_limit)
        belief_to_value = self.scaffold_belief_to_value()
        entity_to_belief = self.scaffold_entity_model_to_belief()

        # Combine scaffolds
        combined_scaffold = []
        combined_scaffold.append("# Advanced Consolidation Scaffold")
        combined_scaffold.append("")
        combined_scaffold.append(
            "Kernle has analyzed your memories for deeper patterns. "
            "Review the scaffolds below and decide what to act on."
        )
        combined_scaffold.append("")
        combined_scaffold.append("---")
        combined_scaffold.append("")
        combined_scaffold.append(cross_domain["scaffold"])
        combined_scaffold.append("---")
        combined_scaffold.append("")
        combined_scaffold.append(belief_to_value["scaffold"])
        combined_scaffold.append("---")
        combined_scaffold.append("")
        combined_scaffold.append(entity_to_belief["scaffold"])

        return {
            "cross_domain": cross_domain,
            "belief_to_value": belief_to_value,
            "entity_to_belief": entity_to_belief,
            "scaffold": "\n".join(combined_scaffold),
        }
