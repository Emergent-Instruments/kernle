"""Tests for _bridge.py — KernleBridge CLI wrapper."""

import subprocess
from unittest.mock import patch

import pytest
from _bridge import KernleBridge


@pytest.fixture
def mock_run():
    with patch("_bridge.subprocess.run") as mock:
        yield mock


@pytest.fixture
def bridge():
    return KernleBridge()


class TestLoad:
    def test_calls_kernle_load_with_stack_id(self, mock_run, bridge):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="# Memory\nvalues...\n", stderr=""
        )

        result = bridge.load("my-project")

        mock_run.assert_called_once_with(
            ["kernle", "-s", "my-project", "load"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result == "# Memory\nvalues..."

    def test_includes_budget_when_provided(self, mock_run, bridge):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="memory\n", stderr=""
        )

        bridge.load("test", 4000)

        mock_run.assert_called_once_with(
            ["kernle", "-s", "test", "load", "--budget", "4000"],
            capture_output=True,
            text=True,
            timeout=5,
        )

    def test_omits_stack_flag_when_none(self, mock_run, bridge):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="memory\n", stderr=""
        )

        bridge.load(None)

        mock_run.assert_called_once_with(
            ["kernle", "load"],
            capture_output=True,
            text=True,
            timeout=5,
        )

    def test_returns_none_on_command_failure(self, mock_run, bridge):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="error"
        )

        assert bridge.load("test") is None

    def test_returns_none_on_timeout(self, mock_run, bridge):
        mock_run.side_effect = subprocess.TimeoutExpired("kernle", 5)

        assert bridge.load("test") is None

    def test_returns_none_on_not_found(self, mock_run, bridge):
        mock_run.side_effect = FileNotFoundError()

        assert bridge.load("test") is None


class TestCheckpoint:
    def test_calls_kernle_checkpoint_save(self, mock_run, bridge):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="saved\n", stderr=""
        )

        result = bridge.checkpoint("proj", "task done")

        mock_run.assert_called_once_with(
            ["kernle", "-s", "proj", "checkpoint", "save", "task done"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result is True

    def test_includes_context_when_provided(self, mock_run, bridge):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="saved\n", stderr=""
        )

        bridge.checkpoint("proj", "task", "extra context")

        mock_run.assert_called_once_with(
            [
                "kernle",
                "-s",
                "proj",
                "checkpoint",
                "save",
                "task",
                "--context",
                "extra context",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

    def test_returns_false_on_failure(self, mock_run, bridge):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="error"
        )

        assert bridge.checkpoint("proj", "task") is False


class TestRaw:
    def test_calls_kernle_raw_with_content(self, mock_run, bridge):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="saved\n", stderr=""
        )

        result = bridge.raw("proj", "quick thought")

        mock_run.assert_called_once_with(
            ["kernle", "-s", "proj", "raw", "quick thought"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result is True


class TestCustomOptions:
    def test_uses_custom_binary_path(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok\n", stderr=""
        )
        custom = KernleBridge(kernle_bin="/usr/local/bin/kernle")

        custom.load("proj")

        assert mock_run.call_args[0][0][0] == "/usr/local/bin/kernle"

    def test_uses_custom_timeout(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok\n", stderr=""
        )
        custom = KernleBridge(timeout=10)

        custom.load("proj")

        assert mock_run.call_args[1]["timeout"] == 10
