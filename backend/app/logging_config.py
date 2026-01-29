"""Logging configuration for Kernle backend."""

import logging
import sys
from datetime import datetime

# Create logger
logger = logging.getLogger("kernle")
logger.setLevel(logging.DEBUG)

# Console handler with detailed format
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)

# File handler for persistent logs
try:
    from pathlib import Path

    log_dir = Path.home() / ".kernle" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(
        log_dir / f"backend-{datetime.now().strftime('%Y-%m-%d')}.log"
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
except Exception as e:
    logger.warning(f"Could not set up file logging: {e}")


def get_logger(name: str = "kernle") -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


# Convenience functions
def log_sync_operation(
    agent_id: str,
    operation: str,
    table: str,
    record_id: str,
    success: bool,
    error: str | None = None,
):
    """Log a sync operation."""
    status = "SUCCESS" if success else "FAILED"
    msg = f"SYNC | {agent_id} | {operation} | {table}/{record_id} | {status}"
    if error:
        msg += f" | {error}"
    if success:
        logger.info(msg)
    else:
        logger.error(msg)


def log_auth_event(
    event: str, agent_id: str | None = None, success: bool = True, details: str | None = None
):
    """Log an auth event."""
    status = "SUCCESS" if success else "FAILED"
    msg = f"AUTH | {event} | {agent_id or 'unknown'} | {status}"
    if details:
        msg += f" | {details}"
    if success:
        logger.info(msg)
    else:
        logger.warning(msg)


def log_memory_flush(
    agent_id: str, triggered_by: str, context_pct: float | None = None, saved: bool = False
):
    """Log a memory flush event (from memoryFlush hook)."""
    msg = f"MEMORY_FLUSH | {agent_id} | triggered_by={triggered_by}"
    if context_pct is not None:
        msg += f" | context={context_pct:.1f}%"
    msg += f" | saved={saved}"
    logger.info(msg)
