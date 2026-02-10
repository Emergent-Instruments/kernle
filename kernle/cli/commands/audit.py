"""Cognitive audit commands for Kernle CLI."""

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kernle import Kernle


def cmd_audit(args, k: "Kernle"):
    """Handle audit subcommands."""
    from kernle.testing.assertions import CognitiveAssertions

    assertions = CognitiveAssertions(k)

    category = getattr(args, "category", None)

    if category == "structural":
        report = assertions.run_structural()
    elif category == "coherence":
        report = assertions.run_coherence()
    elif category == "quality":
        report = assertions.run_quality()
    elif category == "pipeline":
        report = assertions.run_pipeline()
    else:
        report = assertions.run_all()

    if getattr(args, "json", False):
        output = {
            "passed": report.passed,
            "failed": report.failed,
            "total": report.total,
            "all_passed": report.all_passed,
            "assertions": [
                {
                    "category": a.category,
                    "name": a.name,
                    "passed": a.passed,
                    "message": a.message,
                    "details": a.details,
                }
                for a in report.assertions
            ],
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print(report.summary())
