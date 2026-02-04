"""
Skills CLI commands for Kernle Commerce.

Provides command-line interface for skill registry operations:
- kernle skills list
"""

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

    from kernle import Kernle


logger = logging.getLogger(__name__)


def cmd_skills(args: "argparse.Namespace", k: "Kernle") -> None:
    """Handle skills subcommands."""
    action = args.skills_action

    if action == "list":
        _skills_list(args, k)
    else:
        print(f"Unknown skills action: {action}")
        print("Available actions: list")


def _skills_list(args: "argparse.Namespace", k: "Kernle") -> None:
    """List canonical skills."""
    output_json = getattr(args, "json", False)
    category_filter = getattr(args, "category", None)

    try:
        from kernle.commerce.skills.models import SkillCategory
        from kernle.commerce.skills.registry import InMemorySkillRegistry

        registry = InMemorySkillRegistry()

        # Convert category filter if provided
        cat = None
        if category_filter:
            try:
                cat = SkillCategory(category_filter)
            except ValueError:
                valid = [c.value for c in SkillCategory]
                print(f"❌ Invalid category: {category_filter}")
                print(f"   Valid categories: {', '.join(valid)}")
                return

        skills = registry.list_skills(category=cat)

        if output_json:
            result = [skill.to_dict() for skill in skills]
            print(json.dumps(result, indent=2, default=str))
        else:
            if not skills:
                print("No skills found.")
                return

            # Group by category
            by_category: dict[str, list] = {}
            for skill in skills:
                cat = skill.category or "other"
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(skill)

            print("🎯 Canonical Skills")
            print("=" * 50)

            category_emoji = {
                "technical": "💻",
                "creative": "🎨",
                "knowledge": "📚",
                "language": "🌍",
                "service": "🤝",
                "other": "📦",
            }

            for cat_name in sorted(by_category.keys()):
                emoji = category_emoji.get(cat_name, "📦")
                print(f"\n{emoji} {cat_name.title()}")
                print("-" * 30)

                for skill in sorted(by_category[cat_name], key=lambda s: s.name):
                    desc = skill.description or ""
                    print(f"  • {skill.name}")
                    if desc:
                        print(f"    {desc}")

            print("")
            print("Use skills when creating or searching for jobs:")
            print("  kernle job create 'Title' --budget 50 --deadline 7d --skill coding")
            print("  kernle job search --skill research")

    except Exception as e:
        if output_json:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            print(f"❌ Error: {e}")
