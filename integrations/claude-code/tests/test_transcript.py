"""Tests for _transcript.py — transcript JSONL reader."""

from _transcript import read_last_messages


class TestReadLastMessages:
    def test_reads_last_user_message(self, make_transcript):
        path = make_transcript(
            [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Follow-up question"},
                {"role": "assistant", "content": "Follow-up answer"},
            ]
        )

        task, context = read_last_messages(path)

        assert task == "Follow-up question"
        assert context == "Follow-up answer"

    def test_handles_content_blocks(self, make_transcript):
        path = make_transcript(
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Block question"}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Block answer"}],
                },
            ]
        )

        task, context = read_last_messages(path)

        assert task == "Block question"
        assert context == "Block answer"

    def test_truncates_long_user_message(self, make_transcript):
        path = make_transcript([{"role": "user", "content": "x" * 300}])

        task, _ = read_last_messages(path)

        assert len(task) == 200
        assert task.endswith("...")

    def test_truncates_long_assistant_message(self, make_transcript):
        path = make_transcript(
            [
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "y" * 600},
            ]
        )

        _, context = read_last_messages(path)

        assert len(context) == 500
        assert context.endswith("...")

    def test_returns_fallback_when_no_file(self):
        task, context = read_last_messages("/nonexistent/path.jsonl")

        assert task == "Session ended"
        assert context is None

    def test_returns_fallback_when_none(self):
        task, context = read_last_messages(None)

        assert task == "Session ended"
        assert context is None

    def test_handles_empty_transcript(self, make_transcript):
        path = make_transcript([])

        task, context = read_last_messages(path)

        assert task == "Session ended"
        assert context is None

    def test_skips_invalid_json_lines(self, tmp_path):
        path = tmp_path / "transcript.jsonl"
        path.write_text('not json\n{"role": "user", "content": "valid"}\nalso not json\n')

        task, context = read_last_messages(str(path))

        assert task == "valid"
        assert context is None

    def test_skips_entries_without_content(self, make_transcript):
        path = make_transcript(
            [
                {"role": "user", "content": "real question"},
                {"role": "system", "content": ""},
                {"role": "assistant", "content": "real answer"},
            ]
        )

        task, context = read_last_messages(path)

        assert task == "real question"
        assert context == "real answer"
