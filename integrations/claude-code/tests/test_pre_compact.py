"""Tests for pre_compact.py — PreCompact hook."""

import subprocess

from pre_compact import main


class TestPreCompact:
    def test_saves_checkpoint_with_transcript_content(
        self, mock_subprocess, make_hook_input, make_transcript, capture_hook_output
    ):
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="saved\n", stderr=""
        )

        transcript_path = make_transcript(
            [
                {"role": "user", "content": "Fix the login bug"},
                {"role": "assistant", "content": "I fixed it by..."},
            ]
        )

        hook_input = make_hook_input(transcript_path=transcript_path)
        capture_hook_output(main, hook_input)

        # Find checkpoint call
        checkpoint_calls = [c for c in mock_subprocess.call_args_list if "checkpoint" in c[0][0]]
        assert len(checkpoint_calls) == 1
        args = checkpoint_calls[0][0][0]
        # Summary should contain [pre-compact] prefix and task
        summary = args[args.index("save") + 1]
        assert "[pre-compact]" in summary
        assert "Fix the login bug" in summary

    def test_handles_missing_transcript(
        self, mock_subprocess, make_hook_input, capture_hook_output
    ):
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="saved\n", stderr=""
        )

        hook_input = make_hook_input(transcript_path=None)
        _, exit_code = capture_hook_output(main, hook_input)

        assert exit_code == 0
        # Should still try to checkpoint with fallback text
        assert mock_subprocess.called

    def test_exits_zero_on_error(self, mock_subprocess, make_hook_input, capture_hook_output):
        mock_subprocess.side_effect = Exception("fail")

        _, exit_code = capture_hook_output(main, make_hook_input())

        assert exit_code == 0
