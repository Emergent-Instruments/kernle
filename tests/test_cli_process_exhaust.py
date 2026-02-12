"""Additional tests for kernle.cli.commands.process."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from kernle.cli.commands.process import cmd_process
from kernle.processing import ProcessingResult


def _make_kernle():
    k = MagicMock()
    k.entity.model = object()  # Skip auto model binding path in _ensure_model.
    return k


class TestCmdProcessRunFormatting:
    def test_no_results_with_force_prints_no_sources_message(self, capsys):
        k = _make_kernle()
        k.process.return_value = []
        args = SimpleNamespace(
            process_action="run",
            transition=None,
            force=True,
            allow_no_inference_override=False,
            auto_promote=False,
            json=False,
        )

        cmd_process(args, k)

        out = capsys.readouterr().out
        assert "No processing triggered." in out
        assert "No unprocessed memories found." in out

    def test_text_output_handles_blocked_skipped_and_gate_details(self, capsys):
        k = _make_kernle()
        k.process.return_value = [
            ProcessingResult(
                layer_transition="episode_to_belief",
                source_count=3,
                inference_blocked=True,
                skip_reason="No inference model bound",
            ),
            ProcessingResult(
                layer_transition="raw_to_note",
                source_count=1,
                skipped=True,
                skip_reason="No unprocessed sources",
            ),
            ProcessingResult(
                layer_transition="raw_to_episode",
                source_count=2,
                auto_promote=True,
                created=[{"type": "episode", "id": "ep-12345678"}],
                gate_blocked=1,
                gate_details=["belief gate failed"],
                errors=["minor warning"],
            ),
            ProcessingResult(
                layer_transition="episode_to_goal",
                source_count=2,
                suggestions=[{"type": "goal", "id": "goal-abcdef12"}],
                gate_blocked=1,
                gate_details=["goal gate failed"],
                errors=["another warning"],
            ),
        ]
        args = SimpleNamespace(
            process_action="run",
            transition=None,
            force=True,
            allow_no_inference_override=False,
            auto_promote=False,
            json=False,
        )

        cmd_process(args, k)

        out = capsys.readouterr().out
        assert "BLOCKED (no inference)" in out
        assert "skipped (No unprocessed sources)" in out
        assert "gate blocked: 1 item(s)" in out
        assert "minor warning" in out
        assert "(pending review)" in out


class TestCmdProcessStatusAndExhaust:
    def test_status_handles_storage_error(self, capsys):
        k = _make_kernle()
        k._storage = MagicMock()
        k._storage.list_raw.side_effect = RuntimeError("storage unavailable")
        args = SimpleNamespace(process_action="status", json=False)

        cmd_process(args, k)
        assert "Error gathering status: storage unavailable" in capsys.readouterr().out

    def test_exhaust_json_output(self, capsys):
        k = _make_kernle()
        cycle = SimpleNamespace(
            cycle_number=1,
            intensity="normal",
            transitions_run=["raw_to_episode"],
            promotions=2,
            errors=[],
        )
        result = SimpleNamespace(
            cycles_completed=1,
            total_promotions=2,
            converged=True,
            convergence_reason="steady_state",
            snapshot={"raw": 0},
            cycle_results=[cycle],
        )
        args = SimpleNamespace(
            process_action="exhaust",
            max_cycles=5,
            no_auto_promote=False,
            dry_run=False,
            batch_size=3,
            verbose=False,
            json=True,
        )

        with patch("kernle.exhaust.ExhaustionRunner") as mock_runner_cls:
            mock_runner = mock_runner_cls.return_value
            mock_runner.run.return_value = result
            cmd_process(args, k)

        mock_runner_cls.assert_called_once_with(k, max_cycles=5, auto_promote=True, batch_size=3)
        mock_runner.run.assert_called_once_with(dry_run=False)
        payload = json.loads(capsys.readouterr().out)
        assert payload["cycles_completed"] == 1
        assert payload["cycles"][0]["cycle"] == 1

    def test_exhaust_text_output_includes_blocked_and_verbose_errors(self, capsys):
        k = _make_kernle()
        blocked_cycle = SimpleNamespace(
            cycle_number=1,
            intensity="high",
            transitions_run=["episode_to_belief"],
            promotions=0,
            errors=[],
            results=[SimpleNamespace(inference_blocked=True)],
        )
        mixed_cycle = SimpleNamespace(
            cycle_number=2,
            intensity="normal",
            transitions_run=["raw_to_note"],
            promotions=1,
            errors=["temporary failure"],
            results=[SimpleNamespace(inference_blocked=False)],
        )
        result = SimpleNamespace(
            cycles_completed=2,
            total_promotions=1,
            converged=False,
            convergence_reason="max_cycles_reached",
            snapshot={"raw": 1},
            cycle_results=[blocked_cycle, mixed_cycle],
        )
        args = SimpleNamespace(
            process_action="exhaust",
            max_cycles=3,
            no_auto_promote=True,
            dry_run=True,
            batch_size=None,
            verbose=True,
            json=False,
        )

        with patch("kernle.exhaust.ExhaustionRunner") as mock_runner_cls:
            mock_runner = mock_runner_cls.return_value
            mock_runner.run.return_value = result
            cmd_process(args, k)

        out = capsys.readouterr().out
        assert "Exhaustion run complete (dry-run):" in out
        assert "blocked — no inference model bound" in out
        assert "1 promotions, 1 errors" in out
        assert "temporary failure" in out

    def test_unknown_process_action_prints_usage(self, capsys):
        cmd_process(SimpleNamespace(process_action="unknown"), _make_kernle())
        assert "Usage: kernle process {run|status|exhaust}" in capsys.readouterr().out
