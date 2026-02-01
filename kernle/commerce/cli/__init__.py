"""CLI commands for Kernle Commerce.

Provides command-line interface for commerce features:
- kernle wallet balance
- kernle wallet address
- kernle wallet status
- kernle job create
- kernle job list
- kernle job apply
- kernle skills list
- etc.

This module provides the command handlers that are registered
in the main CLI (__main__.py).
"""

from kernle.commerce.cli.wallet import cmd_wallet
from kernle.commerce.cli.job import cmd_job
from kernle.commerce.cli.skills import cmd_skills

__all__ = [
    "cmd_wallet",
    "cmd_job",
    "cmd_skills",
]
