"""Logging configuration for Kernle.

Provides file-based logging for all memory operations, making it easy
to diagnose continuity issues and understand memory flow.
"""

import logging
from datetime import datetime
from pathlib import Path


def setup_kernle_logging(agent_id: str = "default", level: str = "INFO") -> logging.Logger:
    """
    Set up file logging for Kernle operations.

    Logs to ~/.kernle/logs/local-{date}.log

    Args:
        agent_id: Agent identifier for log context
        level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger
    """
    # Create logs directory
    log_dir = Path.home() / ".kernle" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Daily log file
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = log_dir / f"local-{today}.log"

    # Get or create logger
    logger = logging.getLogger("kernle")

    # Only add handler if not already configured
    if not logger.handlers:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))

        # File handler with detailed format
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Also log to stderr for DEBUG
        if level.upper() == "DEBUG":
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

    return logger


def log_memory_event(event_type: str, details: str, agent_id: str = "default"):
    """
    Log a memory event to the dedicated memory events log.

    This is for high-level events that help understand memory flow:
    - load: Memory loaded at session start
    - save: Episode/note/belief saved
    - checkpoint: Checkpoint saved
    - sync: Sync operation completed
    - flush: Context compaction triggered

    Args:
        event_type: Type of event (load, save, checkpoint, sync, flush)
        details: Human-readable details
        agent_id: Agent identifier
    """
    log_dir = Path.home() / ".kernle" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    log_file = log_dir / f"memory-events-{today}.log"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] {event_type} | agent={agent_id} | {details}\n")


# Convenience functions for common events
def log_load(agent_id: str, values: int, beliefs: int, episodes: int, checkpoint: bool):
    """Log a memory load event."""
    log_memory_event(
        "load",
        f"values={values}, beliefs={beliefs}, episodes={episodes}, checkpoint={checkpoint}",
        agent_id,
    )


def log_save(agent_id: str, memory_type: str, memory_id: str, summary: str = ""):
    """Log a memory save event."""
    log_memory_event(
        "save", f"type={memory_type}, id={memory_id[:8]}..., summary={summary[:50]}", agent_id
    )


def log_checkpoint(agent_id: str, task: str, context_len: int):
    """Log a checkpoint save event."""
    log_memory_event("checkpoint", f"task={task[:50]}, context_chars={context_len}", agent_id)


def log_sync(agent_id: str, direction: str, count: int, errors: int = 0):
    """Log a sync operation."""
    log_memory_event("sync", f"direction={direction}, count={count}, errors={errors}", agent_id)
