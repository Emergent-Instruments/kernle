"""Input validation methods for Kernle.

Provides both the ``ValidationMixin`` (instance methods used by
:class:`~kernle.core.Kernle`) and standalone ``sanitize_string`` used
by CLI and MCP layers for consistent input sanitization.
"""

import logging
import re
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def sanitize_string(
    value: Any, field_name: str, max_length: int = 1000, required: bool = True
) -> str:
    """Sanitize and validate string inputs.

    Canonical implementation shared by CLI (as ``validate_input``) and
    MCP (as ``sanitize_string``) layers.

    Args:
        value: The value to sanitize.
        field_name: Name of the field for error messages.
        max_length: Maximum allowed string length.
        required: If True, empty strings are rejected.

    Returns:
        Sanitized string.

    Raises:
        ValueError: If validation fails.
    """
    if value is None and not required:
        return ""

    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string, got {type(value).__name__}")

    if required and not value.strip():
        raise ValueError(f"{field_name} cannot be empty")

    if len(value) > max_length:
        raise ValueError(f"{field_name} too long (max {max_length} characters, got {len(value)})")

    # Remove null bytes and control characters except newlines and tabs
    sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", value)

    return sanitized


class ValidationMixin:
    """Input validation operations for Kernle."""

    def _validate_stack_id(self, stack_id: str) -> str:
        """Validate and sanitize agent ID.

        Rejects path traversal attempts before sanitizing.
        """
        if not stack_id or not stack_id.strip():
            raise ValueError("Stack ID cannot be empty")

        stripped = stack_id.strip()

        # Reject path traversal characters and patterns
        if "/" in stripped or "\\" in stripped:
            raise ValueError("Stack ID must not contain path separators")
        if stripped == "." or stripped == "..":
            raise ValueError("Stack ID must not be a relative path component")
        if ".." in stripped.split("."):
            raise ValueError("Stack ID must not contain path traversal sequences")

        # Remove potentially dangerous characters
        sanitized = "".join(c for c in stripped if c.isalnum() or c in "-_.")

        if not sanitized:
            raise ValueError("Stack ID must contain alphanumeric characters")

        if len(sanitized) > 100:
            raise ValueError("Stack ID too long (max 100 characters)")

        return sanitized

    def _validate_checkpoint_dir(self, checkpoint_dir: Path) -> Path:
        """Validate checkpoint directory path."""
        import tempfile

        try:
            # Resolve to absolute path to prevent directory traversal
            resolved_path = checkpoint_dir.resolve()

            # Ensure it's within a safe directory (user's home, system temp, or /tmp)
            home_path = Path.home().resolve()
            tmp_path = Path("/tmp").resolve()
            system_temp = Path(tempfile.gettempdir()).resolve()

            # Use is_relative_to() for secure path validation (Python 3.9+)
            # This properly handles edge cases like /home/user/../etc that startswith() misses
            is_safe = (
                resolved_path.is_relative_to(home_path)
                or resolved_path.is_relative_to(tmp_path)
                or resolved_path.is_relative_to(system_temp)
            )

            # Also allow /var/folders on macOS (where tempfile creates dirs)
            if not is_safe:
                try:
                    var_folders = Path("/var/folders").resolve()
                    private_var_folders = Path("/private/var/folders").resolve()
                    is_safe = resolved_path.is_relative_to(
                        var_folders
                    ) or resolved_path.is_relative_to(private_var_folders)
                except (OSError, ValueError):
                    pass

            if not is_safe:
                raise ValueError("Checkpoint directory must be within user home or temp directory")

            return resolved_path

        except (OSError, ValueError) as e:
            logger.error(f"Invalid checkpoint directory: {e}")
            raise ValueError(f"Invalid checkpoint directory: {e}")

    def _validate_string_input(
        self, value: str, field_name: str, max_length: Optional[int] = 1000
    ) -> str:
        """Validate and sanitize string inputs.

        Args:
            value: String to validate
            field_name: Name of the field (for error messages)
            max_length: Maximum length, or None to skip length check

        Returns:
            Sanitized string
        """
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string")

        if max_length is not None and len(value) > max_length:
            raise ValueError(f"{field_name} too long (max {max_length} characters)")

        # Basic sanitization - remove null bytes and control characters
        sanitized = value.replace("\x00", "").replace("\r\n", "\n")

        return sanitized

    @staticmethod
    def _validate_derived_from(refs: Optional[List[str]]) -> Optional[List[str]]:
        """Validate derived_from references format.

        Accepts refs in format 'type:id' or 'context:description'.
        Filters out empty strings and validates basic structure.

        Args:
            refs: List of memory references

        Returns:
            Validated list, or None if empty
        """
        if not refs:
            return None

        valid_types = {
            "raw",
            "episode",
            "belief",
            "note",
            "value",
            "goal",
            "drive",
            "relationship",
            "context",
            "kernle",
        }
        validated = []
        for ref in refs:
            if not ref or not isinstance(ref, str):
                continue
            ref = ref.strip()
            if ":" not in ref:
                continue  # Skip malformed refs
            ref_type = ref.split(":", 1)[0]
            if ref_type not in valid_types:
                continue  # Skip unknown types
            validated.append(ref)

        return validated if validated else None
