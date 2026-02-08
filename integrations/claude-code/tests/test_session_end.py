"""Tests for session_end.py — SessionEnd hook."""

import subprocess

from session_end import main


class TestSessionEnd:
    def test_saves_checkpoint_with_last_user_message(
        self, mock_subprocess, make_hook_input, make_transcript, capture_hook_output
    ):
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="saved\n", stderr=""
        )

        transcript_path = make_transcript(
            [
                {"role": "user", "content": "Deploy the app"},
                {"role": "assistant", "content": "Deployed successfully"},
            ]
        )

        hook_input = make_hook_input(transcript_path=transcript_path)
        capture_hook_output(main, hook_input)

        checkpoint_calls = [c for c in mock_subprocess.call_args_list if "checkpoint" in c[0][0]]
        assert len(checkpoint_calls) == 1
        args = checkpoint_calls[0][0][0]
        summary = args[args.index("save") + 1]
        assert "Deploy the app" in summary

    def test_saves_raw_entry_marking_session_end(
        self, mock_subprocess, make_hook_input, make_transcript, capture_hook_output
    ):
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="saved\n", stderr=""
        )

        transcript_path = make_transcript([{"role": "user", "content": "Do something"}])

        hook_input = make_hook_input(transcript_path=transcript_path)
        capture_hook_output(main, hook_input)

        raw_calls = [c for c in mock_subprocess.call_args_list if "raw" in c[0][0]]
        assert len(raw_calls) == 1
        raw_content = raw_calls[0][0][0][-1]
        assert "Session ended" in raw_content

    def test_uses_fallback_when_no_transcript(
        self, mock_subprocess, make_hook_input, capture_hook_output
    ):
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="saved\n", stderr=""
        )

        hook_input = make_hook_input(transcript_path=None)
        capture_hook_output(main, hook_input)

        checkpoint_calls = [c for c in mock_subprocess.call_args_list if "checkpoint" in c[0][0]]
        args = checkpoint_calls[0][0][0]
        summary = args[args.index("save") + 1]
        assert "Session ended" in summary

    def test_continues_if_checkpoint_fails(
        self, mock_subprocess, make_hook_input, make_transcript, capture_hook_output
    ):
        # First call (checkpoint) fails, second call (raw) succeeds
        mock_subprocess.side_effect = [
            subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="fail"),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="saved\n", stderr=""),
        ]

        transcript_path = make_transcript([{"role": "user", "content": "task"}])

        hook_input = make_hook_input(transcript_path=transcript_path)
        _, exit_code = capture_hook_output(main, hook_input)

        assert exit_code == 0
        # raw should still have been called
        assert mock_subprocess.call_count == 2

    def test_exits_zero_on_error(self, mock_subprocess, make_hook_input, capture_hook_output):
        mock_subprocess.side_effect = Exception("fail")

        _, exit_code = capture_hook_output(main, make_hook_input())

        assert exit_code == 0
