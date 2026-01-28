"""CLI command modules for Kernle.

Each module contains related command handlers extracted from __main__.py.
"""

from kernle.cli.commands.anxiety import cmd_anxiety
from kernle.cli.commands.belief import cmd_belief
from kernle.cli.commands.emotion import cmd_emotion
from kernle.cli.commands.forget import cmd_forget
from kernle.cli.commands.identity import cmd_consolidate, cmd_identity
from kernle.cli.commands.meta import cmd_meta
from kernle.cli.commands.playbook import cmd_playbook
from kernle.cli.commands.raw import cmd_raw, resolve_raw_id

__all__ = [
    "cmd_anxiety",
    "cmd_belief",
    "cmd_consolidate",
    "cmd_emotion",
    "cmd_forget",
    "cmd_identity",
    "cmd_meta",
    "cmd_playbook",
    "cmd_raw",
    "resolve_raw_id",
]
