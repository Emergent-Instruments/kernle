"""Tests for CLI suggestions command module."""

import json
from argparse import Namespace
from unittest.mock import MagicMock

import pytest

from kernle.cli.commands.suggestions import cmd_suggestions, resolve_suggestion_id


class TestResolveSuggestionId:
    """Test the resolve_suggestion_id helper function."""

    def test_exact_match(self):
        """Test exact match returns the ID."""
        k = MagicMock()
        k.get_suggestion.return_value = {"id": "abc123", "status": "pending"}

        result = resolve_suggestion_id(k, "abc123")

        assert result == "abc123"
        k.get_suggestion.assert_called_once_with("abc123")

    def test_prefix_match_single(self):
        """Test single prefix match returns full ID."""
        k = MagicMock()
        k.get_suggestion.return_value = None  # No exact match
        k.get_suggestions.return_value = [
            {"id": "abc123456789", "status": "pending"},
        ]

        result = resolve_suggestion_id(k, "abc")

        assert result == "abc123456789"
        k.get_suggestions.assert_called_once_with(limit=1000)

    def test_prefix_match_not_found(self):
        """Test no matches raises ValueError."""
        k = MagicMock()
        k.get_suggestion.return_value = None
        k.get_suggestions.return_value = []

        with pytest.raises(ValueError, match="not found"):
            resolve_suggestion_id(k, "nonexistent")

    def test_prefix_match_ambiguous(self):
        """Test multiple prefix matches raises ValueError."""
        k = MagicMock()
        k.get_suggestion.return_value = None
        k.get_suggestions.return_value = [
            {"id": "abc123456789", "status": "pending"},
            {"id": "abc987654321", "status": "pending"},
            {"id": "abcdef123456", "status": "pending"},
        ]

        with pytest.raises(ValueError, match="Ambiguous ID"):
            resolve_suggestion_id(k, "abc")

    def test_ambiguous_shows_match_ids(self):
        """Test ambiguous error shows partial match IDs."""
        k = MagicMock()
        k.get_suggestion.return_value = None
        k.get_suggestions.return_value = [
            {"id": "abc123456789", "status": "pending"},
            {"id": "abc987654321", "status": "pending"},
        ]

        with pytest.raises(ValueError) as excinfo:
            resolve_suggestion_id(k, "abc")

        assert "abc1234567" in str(excinfo.value) or "abc987654" in str(excinfo.value)
        assert "2 suggestions" in str(excinfo.value)


class TestCmdSuggestionsList:
    """Test cmd_suggestions list action."""

    def test_list_empty(self, capsys):
        """Test list with no suggestions."""
        k = MagicMock()
        k.get_suggestions.return_value = []

        args = Namespace(
            suggestions_action="list",
            pending=False,
            approved=False,
            rejected=False,
            type=None,
            limit=50,
            json=False,
        )

        cmd_suggestions(args, k)

        captured = capsys.readouterr()
        assert "No suggestions found" in captured.out

    def test_list_with_status_filter_pending(self, capsys):
        """Test list with pending status filter."""
        k = MagicMock()
        k.get_suggestions.return_value = []

        args = Namespace(
            suggestions_action="list",
            pending=True,
            approved=False,
            rejected=False,
            type=None,
            limit=50,
            json=False,
        )

        cmd_suggestions(args, k)

        k.get_suggestions.assert_called_once_with(
            status="pending",
            memory_type=None,
            limit=50,
        )
        captured = capsys.readouterr()
        assert "No pending suggestions found" in captured.out

    def test_list_with_status_filter_approved(self, capsys):
        """Test list with approved status filter."""
        k = MagicMock()
        k.get_suggestions.return_value = []

        args = Namespace(
            suggestions_action="list",
            pending=False,
            approved=True,
            rejected=False,
            type=None,
            limit=50,
            json=False,
        )

        cmd_suggestions(args, k)

        k.get_suggestions.assert_called_once_with(
            status="promoted",
            memory_type=None,
            limit=50,
        )

    def test_list_with_status_filter_rejected(self, capsys):
        """Test list with rejected status filter."""
        k = MagicMock()
        k.get_suggestions.return_value = []

        args = Namespace(
            suggestions_action="list",
            pending=False,
            approved=False,
            rejected=True,
            type=None,
            limit=50,
            json=False,
        )

        cmd_suggestions(args, k)

        k.get_suggestions.assert_called_once_with(
            status="rejected",
            memory_type=None,
            limit=50,
        )

    def test_list_with_type_filter(self, capsys):
        """Test list with type filter."""
        k = MagicMock()
        k.get_suggestions.return_value = []

        args = Namespace(
            suggestions_action="list",
            pending=False,
            approved=False,
            rejected=False,
            type="episode",
            limit=50,
            json=False,
        )

        cmd_suggestions(args, k)

        k.get_suggestions.assert_called_once_with(
            status=None,
            memory_type="episode",
            limit=50,
        )

    def test_list_json_output(self, capsys):
        """Test list with JSON output."""
        k = MagicMock()
        suggestions = [
            {
                "id": "sugg123",
                "status": "pending",
                "memory_type": "episode",
                "confidence": 0.85,
                "content": {"objective": "Test objective"},
            }
        ]
        k.get_suggestions.return_value = suggestions

        args = Namespace(
            suggestions_action="list",
            pending=False,
            approved=False,
            rejected=False,
            type=None,
            limit=50,
            json=True,
        )

        cmd_suggestions(args, k)

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert len(output) == 1
        assert output[0]["id"] == "sugg123"

    def test_list_formatted_output(self, capsys):
        """Test list with formatted output."""
        k = MagicMock()
        k.get_suggestions.return_value = [
            {
                "id": "sugg123abc",
                "status": "pending",
                "memory_type": "episode",
                "confidence": 0.85,
                "content": {"objective": "Implement user authentication feature"},
            },
            {
                "id": "sugg456def",
                "status": "promoted",
                "memory_type": "belief",
                "confidence": 0.75,
                "content": {"statement": "Testing is important for quality"},
                "promoted_to": "belief789",
            },
            {
                "id": "sugg789ghi",
                "status": "rejected",
                "memory_type": "note",
                "confidence": 0.60,
                "content": {"content": "Some random note content here"},
            },
        ]

        args = Namespace(
            suggestions_action="list",
            pending=False,
            approved=False,
            rejected=False,
            type=None,
            limit=50,
            json=False,
        )

        cmd_suggestions(args, k)

        captured = capsys.readouterr()
        assert "Suggestions (3 total" in captured.out
        assert "1 pending" in captured.out
        assert "1 approved" in captured.out
        assert "1 rejected" in captured.out
        assert "[?]" in captured.out  # pending icon
        assert "[+]" in captured.out  # promoted icon
        assert "[x]" in captured.out  # rejected icon
        assert "EPI" in captured.out  # episode type
        assert "BEL" in captured.out  # belief type
        assert "NOT" in captured.out  # note type
        assert "85%" in captured.out
        assert "75%" in captured.out
        assert "60%" in captured.out
        assert "Implement user authentication" in captured.out
        assert "Testing is important" in captured.out

    def test_list_with_modified_status(self, capsys):
        """Test list shows modified status correctly."""
        k = MagicMock()
        k.get_suggestions.return_value = [
            {
                "id": "sugg123abc",
                "status": "modified",
                "memory_type": "episode",
                "confidence": 0.85,
                "content": {"objective": "Modified suggestion"},
                "promoted_to": "episode456",
            },
        ]

        args = Namespace(
            suggestions_action="list",
            pending=False,
            approved=False,
            rejected=False,
            type=None,
            limit=50,
            json=False,
        )

        cmd_suggestions(args, k)

        captured = capsys.readouterr()
        assert "[*]" in captured.out  # modified icon
        assert "-> episode456" in captured.out

    def test_list_truncates_long_content(self, capsys):
        """Test list truncates content over 60 chars."""
        k = MagicMock()
        long_content = "A" * 100  # More than 60 chars
        k.get_suggestions.return_value = [
            {
                "id": "sugg123abc",
                "status": "pending",
                "memory_type": "note",
                "confidence": 0.85,
                "content": {"content": long_content},
            },
        ]

        args = Namespace(
            suggestions_action="list",
            pending=False,
            approved=False,
            rejected=False,
            type=None,
            limit=50,
            json=False,
        )

        cmd_suggestions(args, k)

        captured = capsys.readouterr()
        assert "..." in captured.out
        assert long_content not in captured.out  # Full content should be truncated

    def test_list_shows_review_hints_for_pending(self, capsys):
        """Test list shows review hints when pending suggestions exist."""
        k = MagicMock()
        k.get_suggestions.return_value = [
            {
                "id": "sugg123abc",
                "status": "pending",
                "memory_type": "note",
                "confidence": 0.85,
                "content": {"content": "Pending suggestion"},
            },
        ]

        args = Namespace(
            suggestions_action="list",
            pending=False,
            approved=False,
            rejected=False,
            type=None,
            limit=50,
            json=False,
        )

        cmd_suggestions(args, k)

        captured = capsys.readouterr()
        assert "kernle suggestions show" in captured.out
        assert "kernle suggestions approve" in captured.out
        assert "kernle suggestions reject" in captured.out

    def test_list_with_status_shows_status_header(self, capsys):
        """Test list with status filter shows status-specific header."""
        k = MagicMock()
        k.get_suggestions.return_value = [
            {
                "id": "sugg123abc",
                "status": "pending",
                "memory_type": "note",
                "confidence": 0.85,
                "content": {"content": "Pending suggestion"},
            },
        ]

        args = Namespace(
            suggestions_action="list",
            pending=True,
            approved=False,
            rejected=False,
            type=None,
            limit=50,
            json=False,
        )

        cmd_suggestions(args, k)

        captured = capsys.readouterr()
        # When status filter is applied, shows "Suggestions (1 pending)" instead of total breakdown
        assert "Suggestions (1 pending)" in captured.out


class TestCmdSuggestionsShow:
    """Test cmd_suggestions show action."""

    def test_show_not_found(self, capsys):
        """Test show with non-existent suggestion."""
        k = MagicMock()
        k.get_suggestion.return_value = None
        k.get_suggestions.return_value = []

        args = Namespace(
            suggestions_action="show",
            id="nonexistent",
            json=False,
        )

        cmd_suggestions(args, k)

        captured = capsys.readouterr()
        assert "Error:" in captured.out
        assert "not found" in captured.out

    def test_show_resolution_error(self, capsys):
        """Test show with ambiguous ID."""
        k = MagicMock()
        k.get_suggestion.return_value = None
        k.get_suggestions.return_value = [
            {"id": "abc123", "status": "pending"},
            {"id": "abc456", "status": "pending"},
        ]

        args = Namespace(
            suggestions_action="show",
            id="abc",
            json=False,
        )

        cmd_suggestions(args, k)

        captured = capsys.readouterr()
        assert "Error:" in captured.out
        assert "Ambiguous" in captured.out

    def test_show_json_output(self, capsys):
        """Test show with JSON output."""
        k = MagicMock()
        suggestion = {
            "id": "sugg123",
            "status": "pending",
            "memory_type": "episode",
            "confidence": 0.85,
            "created_at": "2026-01-15T10:00:00Z",
            "content": {"objective": "Test objective", "outcome": "Test outcome"},
        }
        k.get_suggestion.return_value = suggestion

        args = Namespace(
            suggestions_action="show",
            id="sugg123",
            json=True,
        )

        cmd_suggestions(args, k)

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["id"] == "sugg123"

    def test_show_episode_formatted(self, capsys):
        """Test show episode suggestion with formatted output."""
        k = MagicMock()
        k.get_suggestion.return_value = {
            "id": "sugg123",
            "status": "pending",
            "memory_type": "episode",
            "confidence": 0.85,
            "created_at": "2026-01-15T10:00:00Z",
            "source_raw_ids": ["raw1", "raw2"],
            "content": {
                "objective": "Implement feature X",
                "outcome": "Successfully completed",
                "outcome_type": "success",
                "lessons": ["Lesson 1", "Lesson 2"],
            },
        }

        args = Namespace(
            suggestions_action="show",
            id="sugg123",
            json=False,
        )

        cmd_suggestions(args, k)

        captured = capsys.readouterr()
        assert "Suggestion: sugg123" in captured.out
        assert "Status: Pending Review" in captured.out
        assert "Type: episode" in captured.out
        assert "Confidence: 85%" in captured.out
        assert "Objective: Implement feature X" in captured.out
        assert "Outcome: Successfully completed" in captured.out
        assert "Outcome Type: success" in captured.out
        assert "Lessons:" in captured.out
        assert "- Lesson 1" in captured.out
        assert "- Lesson 2" in captured.out
        assert "Source raw entries:" in captured.out
        assert "Actions:" in captured.out
        assert "Approve:" in captured.out
        assert "Reject:" in captured.out

    def test_show_belief_formatted(self, capsys):
        """Test show belief suggestion with formatted output."""
        k = MagicMock()
        k.get_suggestion.return_value = {
            "id": "sugg456",
            "status": "pending",
            "memory_type": "belief",
            "confidence": 0.75,
            "created_at": "2026-01-15T10:00:00Z",
            "content": {
                "statement": "Testing leads to quality software",
                "belief_type": "principle",
                "confidence": 0.80,
            },
        }

        args = Namespace(
            suggestions_action="show",
            id="sugg456",
            json=False,
        )

        cmd_suggestions(args, k)

        captured = capsys.readouterr()
        assert "Statement: Testing leads to quality software" in captured.out
        assert "Type: principle" in captured.out
        assert "Confidence: 80%" in captured.out

    def test_show_note_formatted(self, capsys):
        """Test show note suggestion with formatted output."""
        k = MagicMock()
        k.get_suggestion.return_value = {
            "id": "sugg789",
            "status": "pending",
            "memory_type": "note",
            "confidence": 0.60,
            "created_at": "2026-01-15T10:00:00Z",
            "content": {
                "content": "Important decision made here",
                "note_type": "decision",
                "speaker": "Team Lead",
                "reason": "Performance considerations",
            },
        }

        args = Namespace(
            suggestions_action="show",
            id="sugg789",
            json=False,
        )

        cmd_suggestions(args, k)

        captured = capsys.readouterr()
        assert "Important decision made here" in captured.out
        assert "Note Type: decision" in captured.out
        assert "Speaker: Team Lead" in captured.out
        assert "Reason: Performance considerations" in captured.out

    def test_show_promoted_suggestion(self, capsys):
        """Test show promoted suggestion displays promoted_to."""
        k = MagicMock()
        k.get_suggestion.return_value = {
            "id": "sugg123",
            "status": "promoted",
            "memory_type": "episode",
            "confidence": 0.85,
            "created_at": "2026-01-15T10:00:00Z",
            "content": {"objective": "Test"},
            "promoted_to": "episode456",
        }

        args = Namespace(
            suggestions_action="show",
            id="sugg123",
            json=False,
        )

        cmd_suggestions(args, k)

        captured = capsys.readouterr()
        assert "Status: Approved" in captured.out
        assert "Promoted to: episode456" in captured.out
        assert "Actions:" not in captured.out  # No actions for non-pending

    def test_show_rejected_suggestion(self, capsys):
        """Test show rejected suggestion displays reason."""
        k = MagicMock()
        k.get_suggestion.return_value = {
            "id": "sugg123",
            "status": "rejected",
            "memory_type": "episode",
            "confidence": 0.85,
            "created_at": "2026-01-15T10:00:00Z",
            "content": {"objective": "Test"},
            "resolution_reason": "Duplicate of existing memory",
        }

        args = Namespace(
            suggestions_action="show",
            id="sugg123",
            json=False,
        )

        cmd_suggestions(args, k)

        captured = capsys.readouterr()
        assert "Status: Rejected" in captured.out
        assert "Resolution reason: Duplicate of existing memory" in captured.out

    def test_show_found_via_prefix_but_then_not_found(self, capsys):
        """Test show when ID resolves via prefix but then suggestion not found."""
        k = MagicMock()
        # resolve_suggestion_id: get_suggestion(partial_id) returns None
        # resolve_suggestion_id: get_suggestions finds match, returns full_id
        # cmd_suggestions: get_suggestion(full_id) returns None
        k.get_suggestion.side_effect = [None, None]
        k.get_suggestions.return_value = [{"id": "abc123456789", "status": "pending"}]

        args = Namespace(
            suggestions_action="show",
            id="abc",
            json=False,
        )

        cmd_suggestions(args, k)

        captured = capsys.readouterr()
        assert "Suggestion abc not found" in captured.out


class TestCmdSuggestionsApprove:
    """Test cmd_suggestions approve action."""

    def test_approve_not_found(self, capsys):
        """Test approve non-existent suggestion."""
        k = MagicMock()
        k.get_suggestion.return_value = None
        k.get_suggestions.return_value = []

        args = Namespace(
            suggestions_action="approve",
            id="nonexistent",
        )

        cmd_suggestions(args, k)

        captured = capsys.readouterr()
        assert "Error:" in captured.out
        assert "not found" in captured.out

    def test_approve_already_resolved(self, capsys):
        """Test approve already resolved suggestion."""
        k = MagicMock()
        k.get_suggestion.return_value = {
            "id": "sugg123",
            "status": "promoted",
            "memory_type": "episode",
        }

        args = Namespace(
            suggestions_action="approve",
            id="sugg123",
        )

        cmd_suggestions(args, k)

        captured = capsys.readouterr()
        assert "already promoted" in captured.out

    def test_approve_success(self, capsys):
        """Test approve succeeds."""
        k = MagicMock()
        k.get_suggestion.return_value = {
            "id": "sugg123",
            "status": "pending",
            "memory_type": "episode",
        }
        k.promote_suggestion.return_value = "episode789"

        args = Namespace(
            suggestions_action="approve",
            id="sugg123",
            objective=None,
            outcome=None,
            statement=None,
            content=None,
        )

        cmd_suggestions(args, k)

        k.promote_suggestion.assert_called_once_with("sugg123", None)
        captured = capsys.readouterr()
        assert "approved (promoted)" in captured.out
        # Output truncates the ID with "..." suffix
        assert "episode7" in captured.out

    def test_approve_with_modifications(self, capsys):
        """Test approve with modifications."""
        k = MagicMock()
        k.get_suggestion.return_value = {
            "id": "sugg123",
            "status": "pending",
            "memory_type": "episode",
        }
        k.promote_suggestion.return_value = "episode789"

        args = Namespace(
            suggestions_action="approve",
            id="sugg123",
            objective="Modified objective",
            outcome="Modified outcome",
            statement=None,
            content=None,
        )

        cmd_suggestions(args, k)

        k.promote_suggestion.assert_called_once_with(
            "sugg123", {"objective": "Modified objective", "outcome": "Modified outcome"}
        )
        captured = capsys.readouterr()
        assert "approved (modified)" in captured.out

    def test_approve_with_statement_modification(self, capsys):
        """Test approve with statement modification (for beliefs)."""
        k = MagicMock()
        k.get_suggestion.return_value = {
            "id": "sugg123",
            "status": "pending",
            "memory_type": "belief",
        }
        k.promote_suggestion.return_value = "belief789"

        args = Namespace(
            suggestions_action="approve",
            id="sugg123",
            objective=None,
            outcome=None,
            statement="Modified belief statement",
            content=None,
        )

        cmd_suggestions(args, k)

        k.promote_suggestion.assert_called_once_with(
            "sugg123", {"statement": "Modified belief statement"}
        )
        captured = capsys.readouterr()
        assert "approved (modified)" in captured.out

    def test_approve_with_content_modification(self, capsys):
        """Test approve with content modification (for notes)."""
        k = MagicMock()
        k.get_suggestion.return_value = {
            "id": "sugg123",
            "status": "pending",
            "memory_type": "note",
        }
        k.promote_suggestion.return_value = "note789"

        args = Namespace(
            suggestions_action="approve",
            id="sugg123",
            objective=None,
            outcome=None,
            statement=None,
            content="Modified note content",
        )

        cmd_suggestions(args, k)

        k.promote_suggestion.assert_called_once_with(
            "sugg123", {"content": "Modified note content"}
        )
        captured = capsys.readouterr()
        assert "approved (modified)" in captured.out

    def test_approve_found_via_prefix_but_then_not_found(self, capsys):
        """Test approve when ID resolves but suggestion then not found."""
        k = MagicMock()
        # resolve_suggestion_id: get_suggestion(partial_id) returns None
        # resolve_suggestion_id: get_suggestions finds match, returns full_id
        # cmd_suggestions: get_suggestion(full_id) returns None
        k.get_suggestion.side_effect = [None, None]
        k.get_suggestions.return_value = [{"id": "abc123456789", "status": "pending"}]

        args = Namespace(
            suggestions_action="approve",
            id="abc",
            objective=None,
            outcome=None,
            statement=None,
            content=None,
        )

        cmd_suggestions(args, k)

        captured = capsys.readouterr()
        assert "Suggestion abc not found" in captured.out

    def test_approve_fails(self, capsys):
        """Test approve when promotion fails."""
        k = MagicMock()
        k.get_suggestion.return_value = {
            "id": "sugg123",
            "status": "pending",
            "memory_type": "episode",
        }
        k.promote_suggestion.return_value = None

        args = Namespace(
            suggestions_action="approve",
            id="sugg123",
            objective=None,
            outcome=None,
            statement=None,
            content=None,
        )

        cmd_suggestions(args, k)

        captured = capsys.readouterr()
        assert "Failed to promote suggestion" in captured.out


class TestCmdSuggestionsReject:
    """Test cmd_suggestions reject action."""

    def test_reject_not_found(self, capsys):
        """Test reject non-existent suggestion."""
        k = MagicMock()
        k.get_suggestion.return_value = None
        k.get_suggestions.return_value = []

        args = Namespace(
            suggestions_action="reject",
            id="nonexistent",
            reason=None,
        )

        cmd_suggestions(args, k)

        captured = capsys.readouterr()
        assert "Error:" in captured.out
        assert "not found" in captured.out

    def test_reject_already_resolved(self, capsys):
        """Test reject already resolved suggestion."""
        k = MagicMock()
        k.get_suggestion.return_value = {
            "id": "sugg123",
            "status": "rejected",
            "memory_type": "episode",
        }

        args = Namespace(
            suggestions_action="reject",
            id="sugg123",
            reason=None,
        )

        cmd_suggestions(args, k)

        captured = capsys.readouterr()
        assert "already rejected" in captured.out

    def test_reject_success(self, capsys):
        """Test reject succeeds."""
        k = MagicMock()
        k.get_suggestion.return_value = {
            "id": "sugg123",
            "status": "pending",
            "memory_type": "episode",
        }
        k.reject_suggestion.return_value = True

        args = Namespace(
            suggestions_action="reject",
            id="sugg123",
            reason=None,
        )

        cmd_suggestions(args, k)

        k.reject_suggestion.assert_called_once_with("sugg123", None)
        captured = capsys.readouterr()
        assert "Suggestion rejected" in captured.out

    def test_reject_with_reason(self, capsys):
        """Test reject with reason."""
        k = MagicMock()
        k.get_suggestion.return_value = {
            "id": "sugg123",
            "status": "pending",
            "memory_type": "episode",
        }
        k.reject_suggestion.return_value = True

        args = Namespace(
            suggestions_action="reject",
            id="sugg123",
            reason="Duplicate of existing memory",
        )

        cmd_suggestions(args, k)

        k.reject_suggestion.assert_called_once_with("sugg123", "Duplicate of existing memory")
        captured = capsys.readouterr()
        assert "Suggestion rejected" in captured.out
        assert "Reason: Duplicate of existing memory" in captured.out

    def test_reject_fails(self, capsys):
        """Test reject when rejection fails."""
        k = MagicMock()
        k.get_suggestion.return_value = {
            "id": "sugg123",
            "status": "pending",
            "memory_type": "episode",
        }
        k.reject_suggestion.return_value = False

        args = Namespace(
            suggestions_action="reject",
            id="sugg123",
            reason=None,
        )

        cmd_suggestions(args, k)

        captured = capsys.readouterr()
        assert "Failed to reject suggestion" in captured.out

    def test_reject_found_via_prefix_but_then_not_found(self, capsys):
        """Test reject when ID resolves but suggestion then not found."""
        k = MagicMock()
        # resolve_suggestion_id: get_suggestion(partial_id) returns None
        # resolve_suggestion_id: get_suggestions finds match, returns full_id
        # cmd_suggestions: get_suggestion(full_id) returns None
        k.get_suggestion.side_effect = [None, None]
        k.get_suggestions.return_value = [{"id": "abc123456789", "status": "pending"}]

        args = Namespace(
            suggestions_action="reject",
            id="abc",
            reason=None,
        )

        cmd_suggestions(args, k)

        captured = capsys.readouterr()
        assert "Suggestion abc not found" in captured.out


class TestCmdSuggestionsExtract:
    """Test cmd_suggestions extract action."""

    def test_extract_no_results(self, capsys):
        """Test extract with no suggestions found."""
        k = MagicMock()
        k.extract_suggestions_from_unprocessed.return_value = []

        args = Namespace(
            suggestions_action="extract",
            limit=50,
        )

        cmd_suggestions(args, k)

        k.extract_suggestions_from_unprocessed.assert_called_once_with(limit=50)
        captured = capsys.readouterr()
        assert "Extracting suggestions" in captured.out
        assert "No suggestions extracted" in captured.out

    def test_extract_with_results(self, capsys):
        """Test extract with suggestions found."""
        k = MagicMock()
        k.extract_suggestions_from_unprocessed.return_value = [
            {"id": "sugg1", "memory_type": "episode"},
            {"id": "sugg2", "memory_type": "episode"},
            {"id": "sugg3", "memory_type": "belief"},
        ]

        args = Namespace(
            suggestions_action="extract",
            limit=50,
        )

        cmd_suggestions(args, k)

        captured = capsys.readouterr()
        assert "Extracted 3 suggestion(s)" in captured.out
        assert "episode: 2" in captured.out
        assert "belief: 1" in captured.out
        assert "kernle suggestions list --pending" in captured.out

    def test_extract_custom_limit(self, capsys):
        """Test extract with custom limit."""
        k = MagicMock()
        k.extract_suggestions_from_unprocessed.return_value = []

        args = Namespace(
            suggestions_action="extract",
            limit=100,
        )

        cmd_suggestions(args, k)

        k.extract_suggestions_from_unprocessed.assert_called_once_with(limit=100)
        captured = capsys.readouterr()
        assert "up to 100 unprocessed" in captured.out
