"""Trust layer CLI commands for Kernle (KEP v3 section 8)."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kernle import Kernle


def cmd_trust(args, k: "Kernle"):
    """Manage trust assessments."""
    action = getattr(args, "trust_action", None)

    if action == "list":
        assessments = k.trust_list()
        if not assessments:
            print("No trust assessments. Run `kernle trust seed` to initialize.")
            return

        print(f"Trust assessments ({len(assessments)}):")
        print()
        for a in assessments:
            dims = a["dimensions"]
            general = dims.get("general", {})
            general_score = general.get("score", 0.0) if isinstance(general, dict) else 0.0
            pct = int(general_score * 100)
            bar = "\u2588" * (pct // 10) + "\u2591" * (10 - pct // 10)
            authority_scopes = [auth.get("scope", "?") for auth in a.get("authority", [])]
            auth_str = ", ".join(authority_scopes) if authority_scopes else "none"

            print(f"  {a['entity']}")
            print(f"    General: [{bar}] {pct}%")

            # Show domain-specific scores
            for domain, dim_data in dims.items():
                if domain != "general" and isinstance(dim_data, dict):
                    d_score = dim_data.get("score", 0.0)
                    d_pct = int(d_score * 100)
                    print(f"    {domain}: {d_pct}%")

            print(f"    Authority: {auth_str}")
            print()

    elif action == "show":
        entity = args.entity
        result = k.trust_show(entity)
        if result is None:
            print(f"No trust assessment for: {entity}")
            return

        print(f"Trust: {result['entity']}")
        print()
        print("Dimensions:")
        for domain, dim_data in result["dimensions"].items():
            score = dim_data.get("score", 0.0) if isinstance(dim_data, dict) else 0.0
            pct = int(score * 100)
            bar = "\u2588" * (pct // 10) + "\u2591" * (10 - pct // 10)
            print(f"  {domain}: [{bar}] {pct}%")

        print()
        print("Authority:")
        for auth in result.get("authority", []):
            scope = auth.get("scope", "unknown")
            requires_evidence = auth.get("requires_evidence", False)
            evidence_str = " (requires evidence)" if requires_evidence else ""
            print(f"  - {scope}{evidence_str}")

        if result.get("evidence_episode_ids"):
            print(f"\nEvidence: {len(result['evidence_episode_ids'])} episodes")

        if result.get("created_at"):
            print(f"Created: {result['created_at']}")
        if result.get("last_updated"):
            print(f"Updated: {result['last_updated']}")

    elif action == "set":
        entity = args.entity
        domain = getattr(args, "domain", "general") or "general"
        score = args.score

        if score < 0.0 or score > 1.0:
            print("Score must be between 0.0 and 1.0")
            return

        assessment_id = k.trust_set(entity, domain=domain, score=score)
        pct = int(score * 100)
        print(f"Trust set: {entity} ({domain}) = {pct}%")
        print(f"  ID: {assessment_id}")

    elif action == "seed":
        count = k.seed_trust()
        if count > 0:
            print(f"Seeded {count} trust assessments.")
        else:
            print("All seed trust assessments already exist.")

        # Show current state
        assessments = k.trust_list()
        print(f"\nCurrent trust ({len(assessments)}):")
        for a in assessments:
            dims = a["dimensions"]
            general = dims.get("general", {})
            score = general.get("score", 0.0) if isinstance(general, dict) else 0.0
            print(f"  {a['entity']}: {int(score * 100)}%")

    elif action == "gate":
        source = args.source
        gate_action = args.gate_action
        domain = getattr(args, "domain", None)

        result = k.gate_memory_input(source, gate_action, target=domain)
        status = "ALLOWED" if result["allowed"] else "DENIED"
        print(f"{status}: {result['reason']}")
        print(f"  Trust level: {result['trust_level']:.2f}")
        print(f"  Domain: {result['domain']}")

    elif action == "compute":
        entity = args.entity
        domain = getattr(args, "domain", "general") or "general"
        apply_result = getattr(args, "apply", False)

        result = k.trust_compute(entity, domain=domain)

        if result.get("source") == "default":
            print(f"No episode history for: {entity}")
            print(f"  Default trust: {int(result['score'] * 100)}%")
        else:
            pct = int(result["score"] * 100)
            bar = "\u2588" * (pct // 10) + "\u2591" * (10 - pct // 10)
            print(f"Computed trust for: {entity}")
            print(f"  Score: [{bar}] {pct}%")
            print(
                f"  Episodes: {result['total']} ({result['positive']:.1f} positive, {result['negative']:.1f} negative)"
            )

            if "self_trust_floor" in result:
                floor_pct = int(result["self_trust_floor"] * 100)
                print(f"  Self-trust floor: {floor_pct}%")

        if apply_result:
            k.trust_set(entity, domain=domain, score=result["score"])
            print("  Applied to stored assessment.")

    elif action == "chain":
        target = args.target
        chain = args.chain
        domain = getattr(args, "domain", "general") or "general"

        result = k.trust_chain(target, chain, domain=domain)

        pct = int(result["score"] * 100)
        bar = "\u2588" * (pct // 10) + "\u2591" * (10 - pct // 10)
        print(f"Transitive trust to: {target}")
        print(f"  Chain: {' -> '.join(chain)} -> {target}")
        print(f"  Domain: {domain}")
        print(f"  Score: [{bar}] {pct}%")
        print()
        print("  Hops:")
        for hop in result.get("hops", []):
            h_pct = int(hop["direct_trust"] * 100)
            c_pct = int(hop["cumulative"] * 100)
            print(
                f"    {hop['entity']}: {h_pct}% direct, decay {hop['depth_decay']:.2f}, cumulative {c_pct}%"
            )

    elif action == "decay":
        entity = args.entity
        days = args.days

        result = k.apply_trust_decay(entity, days)

        if result.get("error"):
            print(f"Error: {result['error']}")
            return

        print(f"Applied trust decay for: {entity}")
        print(f"  Days without interaction: {days}")
        print(f"  Decay factor: {result['decay_factor']:.4f}")
        print()
        for domain, dim_data in result.get("dimensions", {}).items():
            if isinstance(dim_data, dict):
                pct = int(dim_data.get("score", 0) * 100)
                print(f"  {domain}: {pct}%")

    else:
        print("Usage: kernle trust {list|show|set|seed|gate|compute|chain|decay}")
        print("  list                     List all trust assessments")
        print("  show <entity>            Show trust details for an entity")
        print("  set <entity> <score>     Set trust score for an entity")
        print("  seed                     Initialize seed trust templates")
        print("  gate <source> <action>   Check if action is allowed by trust")
        print("  compute <entity>         Compute trust from episode history")
        print("  chain <target> <e1 e2>   Compute transitive trust through a chain")
        print("  decay <entity> <days>    Apply trust decay for N days")
