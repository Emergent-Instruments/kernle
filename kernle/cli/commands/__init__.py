"""CLI command modules for Kernle.

Each module contains related command handlers extracted from __main__.py.
"""

from kernle.cli.commands.anxiety import cmd_anxiety
from kernle.cli.commands.audit import cmd_audit
from kernle.cli.commands.auth import cmd_auth, cmd_auth_keys
from kernle.cli.commands.belief import cmd_belief
from kernle.cli.commands.credentials import (
    clear_credentials,
    get_credentials_path,
    load_credentials,
    prompt_backend_url,
    save_credentials,
    warn_non_https_url,
)
from kernle.cli.commands.diagnostic import (
    cmd_boot,
    cmd_drive,
    cmd_dump,
    cmd_export,
    cmd_export_cache,
    cmd_export_full,
    cmd_resume,
    cmd_status,
    cmd_temporal,
)
from kernle.cli.commands.doctor import (
    cmd_doctor,
    cmd_doctor_structural,
)
from kernle.cli.commands.emotion import cmd_emotion
from kernle.cli.commands.epoch import cmd_epoch
from kernle.cli.commands.forget import cmd_forget
from kernle.cli.commands.hook import cmd_hook
from kernle.cli.commands.identity import cmd_identity, cmd_promote
from kernle.cli.commands.init import cmd_init as cmd_init_md
from kernle.cli.commands.memory import (
    cmd_checkpoint,
    cmd_episode,
    cmd_extract,
    cmd_load,
    cmd_note,
    cmd_search,
)
from kernle.cli.commands.meta import cmd_meta
from kernle.cli.commands.migrate import cmd_migrate
from kernle.cli.commands.model import cmd_model
from kernle.cli.commands.narrative import cmd_narrative
from kernle.cli.commands.playbook import cmd_playbook
from kernle.cli.commands.process import cmd_process
from kernle.cli.commands.raw import cmd_raw, resolve_raw_id
from kernle.cli.commands.relations import cmd_entity_model, cmd_relation
from kernle.cli.commands.seed import cmd_seed
from kernle.cli.commands.stats import cmd_stats
from kernle.cli.commands.suggestions import cmd_suggestions, resolve_suggestion_id
from kernle.cli.commands.summary import cmd_summary
from kernle.cli.commands.sync import cmd_sync

__all__ = [
    "cmd_anxiety",
    "cmd_audit",
    "cmd_auth",
    "cmd_auth_keys",
    "cmd_belief",
    "cmd_boot",
    "cmd_checkpoint",
    "cmd_drive",
    "cmd_dump",
    "cmd_entity_model",
    "cmd_episode",
    "cmd_export",
    "cmd_export_cache",
    "cmd_export_full",
    "cmd_extract",
    "cmd_load",
    "cmd_note",
    "cmd_promote",
    "cmd_doctor",
    "cmd_doctor_structural",
    "cmd_emotion",
    "cmd_epoch",
    "cmd_forget",
    "cmd_hook",
    "cmd_identity",
    "cmd_init_md",
    "cmd_meta",
    "cmd_model",
    "cmd_migrate",
    "cmd_narrative",
    "cmd_playbook",
    "cmd_process",
    "cmd_raw",
    "cmd_seed",
    "cmd_relation",
    "cmd_resume",
    "cmd_search",
    "cmd_stats",
    "cmd_status",
    "cmd_suggestions",
    "cmd_summary",
    "cmd_sync",
    "cmd_temporal",
    "clear_credentials",
    "get_credentials_path",
    "load_credentials",
    "prompt_backend_url",
    "resolve_raw_id",
    "resolve_suggestion_id",
    "save_credentials",
    "warn_non_https_url",
]
