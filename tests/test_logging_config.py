"""Tests for kernle.logging_config module."""

import logging

import pytest

from kernle.logging_config import (
    log_checkpoint,
    log_load,
    log_memory_event,
    log_save,
    log_sync,
    setup_kernle_logging,
)


@pytest.fixture(autouse=True)
def clean_kernle_logger():
    """Remove all handlers from the kernle logger before/after each test."""
    logger = logging.getLogger("kernle")
    logger.handlers.clear()
    logger.setLevel(logging.WARNING)
    yield
    logger.handlers.clear()
    logger.setLevel(logging.WARNING)


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    """Set KERNLE_DATA_DIR so logs go to a temp directory."""
    monkeypatch.setenv("KERNLE_DATA_DIR", str(tmp_path))
    return tmp_path / "logs"


class TestSetupKernleLogging:
    """Tests for setup_kernle_logging."""

    def test_returns_logger(self, log_dir):
        """Should return a logging.Logger instance."""
        logger = setup_kernle_logging(stack_id="test-agent")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "kernle"

    def test_creates_log_directory(self, log_dir):
        """Should create the logs directory if it doesn't exist."""
        assert not log_dir.exists()
        setup_kernle_logging(stack_id="test-agent")
        assert log_dir.exists()

    def test_creates_file_handler(self, log_dir):
        """Should add a file handler to the logger."""
        logger = setup_kernle_logging(stack_id="test-agent")
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1

    def test_log_file_named_with_date(self, log_dir):
        """Should create a log file named local-{date}.log."""
        setup_kernle_logging(stack_id="test-agent")
        log_files = list(log_dir.glob("local-*.log"))
        assert len(log_files) == 1
        assert log_files[0].name.startswith("local-")
        assert log_files[0].name.endswith(".log")

    def test_default_level_info(self, log_dir):
        """Default level should be INFO."""
        logger = setup_kernle_logging(stack_id="test-agent")
        assert logger.level == logging.INFO

    def test_custom_level_debug(self, log_dir):
        """Should respect a custom DEBUG level."""
        logger = setup_kernle_logging(stack_id="test-agent", level="DEBUG")
        assert logger.level == logging.DEBUG

    def test_custom_level_warning(self, log_dir):
        """Should respect a custom WARNING level."""
        logger = setup_kernle_logging(stack_id="test-agent", level="WARNING")
        assert logger.level == logging.WARNING

    def test_custom_level_case_insensitive(self, log_dir):
        """Level string should be case-insensitive."""
        logger = setup_kernle_logging(stack_id="test-agent", level="debug")
        assert logger.level == logging.DEBUG

    def test_invalid_level_falls_back_to_info(self, log_dir):
        """Invalid level string should fall back to INFO."""
        logger = setup_kernle_logging(stack_id="test-agent", level="INVALID")
        assert logger.level == logging.INFO

    def test_debug_adds_console_handler(self, log_dir):
        """DEBUG level should add a StreamHandler in addition to FileHandler."""
        logger = setup_kernle_logging(stack_id="test-agent", level="DEBUG")
        stream_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) == 1

    def test_info_no_console_handler(self, log_dir):
        """INFO level should NOT add a StreamHandler."""
        logger = setup_kernle_logging(stack_id="test-agent", level="INFO")
        stream_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) == 0

    def test_no_duplicate_handlers(self, log_dir):
        """Calling setup twice should not add duplicate handlers."""
        logger1 = setup_kernle_logging(stack_id="test-agent")
        logger2 = setup_kernle_logging(stack_id="test-agent")
        assert logger1 is logger2
        file_handlers = [h for h in logger1.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1

    def test_writes_to_log_file(self, log_dir):
        """Should actually write log messages to the file."""
        logger = setup_kernle_logging(stack_id="test-agent", level="INFO")
        logger.info("test message from unit test")
        # Flush handlers
        for h in logger.handlers:
            h.flush()
        log_files = list(log_dir.glob("local-*.log"))
        assert len(log_files) == 1
        content = log_files[0].read_text()
        assert "test message from unit test" in content

    def test_log_format(self, log_dir):
        """Log messages should use the expected format."""
        logger = setup_kernle_logging(stack_id="test-agent", level="INFO")
        logger.info("format check")
        for h in logger.handlers:
            h.flush()
        log_files = list(log_dir.glob("local-*.log"))
        content = log_files[0].read_text()
        # Format: "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        assert " | INFO | kernle | format check" in content


class TestLogMemoryEvent:
    """Tests for log_memory_event."""

    def test_creates_event_log_file(self, log_dir):
        """Should create a memory-events log file."""
        log_memory_event("load", "test details", stack_id="test-agent")
        event_files = list(log_dir.glob("memory-events-*.log"))
        assert len(event_files) == 1

    def test_event_log_format(self, log_dir):
        """Event log should contain event_type, stack_id, and details."""
        log_memory_event("save", "type=episode, id=abc12345", stack_id="my-agent")
        event_files = list(log_dir.glob("memory-events-*.log"))
        content = event_files[0].read_text()
        assert "save | agent=my-agent | type=episode, id=abc12345" in content

    def test_event_log_appends(self, log_dir):
        """Multiple events should append to the same file."""
        log_memory_event("load", "first event", stack_id="test")
        log_memory_event("save", "second event", stack_id="test")
        event_files = list(log_dir.glob("memory-events-*.log"))
        assert len(event_files) == 1
        content = event_files[0].read_text()
        lines = [line for line in content.strip().split("\n") if line]
        assert len(lines) == 2
        assert "first event" in lines[0]
        assert "second event" in lines[1]

    def test_default_stack_id(self, log_dir):
        """Default stack_id should be 'default'."""
        log_memory_event("sync", "test details")
        event_files = list(log_dir.glob("memory-events-*.log"))
        content = event_files[0].read_text()
        assert "agent=default" in content


class TestConvenienceFunctions:
    """Tests for log_load, log_save, log_checkpoint, log_sync."""

    def test_log_load(self, log_dir):
        """log_load should write a load event with counts."""
        log_load("test-agent", values=5, beliefs=10, episodes=25, checkpoint=True)
        event_files = list(log_dir.glob("memory-events-*.log"))
        content = event_files[0].read_text()
        assert "load | agent=test-agent" in content
        assert "values=5" in content
        assert "beliefs=10" in content
        assert "episodes=25" in content
        assert "checkpoint=True" in content

    def test_log_save(self, log_dir):
        """log_save should write a save event with type and truncated id."""
        log_save(
            "test-agent",
            memory_type="episode",
            memory_id="abcdef1234567890",
            summary="Test episode summary",
        )
        event_files = list(log_dir.glob("memory-events-*.log"))
        content = event_files[0].read_text()
        assert "save | agent=test-agent" in content
        assert "type=episode" in content
        assert "id=abcdef12..." in content
        assert "summary=Test episode summary" in content

    def test_log_checkpoint(self, log_dir):
        """log_checkpoint should write a checkpoint event with task and context length."""
        log_checkpoint("test-agent", task="Working on tests", context_len=150)
        event_files = list(log_dir.glob("memory-events-*.log"))
        content = event_files[0].read_text()
        assert "checkpoint | agent=test-agent" in content
        assert "task=Working on tests" in content
        assert "context_chars=150" in content

    def test_log_sync(self, log_dir):
        """log_sync should write a sync event with direction and counts."""
        log_sync("test-agent", direction="push", count=10, errors=2)
        event_files = list(log_dir.glob("memory-events-*.log"))
        content = event_files[0].read_text()
        assert "sync | agent=test-agent" in content
        assert "direction=push" in content
        assert "count=10" in content
        assert "errors=2" in content

    def test_log_sync_default_errors(self, log_dir):
        """log_sync errors should default to 0."""
        log_sync("test-agent", direction="pull", count=5)
        event_files = list(log_dir.glob("memory-events-*.log"))
        content = event_files[0].read_text()
        assert "errors=0" in content
