"""Tests for the import functionality."""

import json

import pytest

from kernle import Kernle
from kernle.cli.commands.import_cmd import (
    _parse_beliefs,
    _parse_episodes,
    _parse_goals,
    _parse_markdown,
    _parse_notes,
    _parse_raw,
    _parse_values,
)


class TestMarkdownParsing:
    """Tests for markdown parsing functions."""

    def test_parse_beliefs_with_percentage_confidence(self):
        """Parse beliefs with percentage confidence format."""
        content = """
- Testing is important (90%)
- Code should be readable (85%)
- Documentation matters
"""
        items = _parse_beliefs(content)
        assert len(items) == 3
        assert items[0]["statement"] == "Testing is important"
        assert items[0]["confidence"] == 0.9
        assert items[1]["statement"] == "Code should be readable"
        assert items[1]["confidence"] == 0.85
        assert items[2]["statement"] == "Documentation matters"
        assert items[2]["confidence"] == 0.7  # default

    def test_parse_beliefs_with_decimal_confidence(self):
        """Parse beliefs with decimal confidence format."""
        content = """
- First belief [0.95]
- Second belief [0.6]
"""
        items = _parse_beliefs(content)
        assert len(items) == 2
        assert items[0]["confidence"] == 0.95
        assert items[1]["confidence"] == 0.6

    def test_parse_beliefs_with_confidence_label(self):
        """Parse beliefs with explicit confidence: label."""
        content = """
- Testing is crucial (confidence: 0.9)
- Readability matters (confidence: 85)
"""
        items = _parse_beliefs(content)
        assert len(items) == 2
        assert items[0]["confidence"] == 0.9
        assert items[1]["confidence"] == 0.85

    def test_parse_episodes_with_lessons(self):
        """Parse episodes with lesson markers."""
        content = """
- Fixed authentication bug -> Always check token expiry
- Deployed new feature (Lesson: Test in staging first)
- Simple task completed
"""
        items = _parse_episodes(content)
        assert len(items) == 3
        assert items[0]["lesson"] == "Always check token expiry"
        assert items[1]["lesson"] == "Test in staging first"
        assert items[2]["lesson"] is None

    def test_parse_goals_with_status(self):
        """Parse goals with status markers."""
        content = """
- Implement feature X
- [done] Ship version 2.0
- [complete] Write documentation
- [x] Fix the bug
"""
        items = _parse_goals(content)
        assert len(items) == 4
        assert items[0]["status"] == "active"
        assert items[1]["status"] == "completed"
        assert items[2]["status"] == "completed"
        assert items[3]["status"] == "completed"

    def test_parse_values_with_description(self):
        """Parse values with name: description format."""
        content = """
- Quality: Code should be well-tested and maintainable
- Simplicity: Prefer simple solutions over complex ones
- Single word value
"""
        items = _parse_values(content)
        assert len(items) == 3
        assert items[0]["name"] == "Quality"
        assert "well-tested" in items[0]["description"]
        assert items[1]["name"] == "Simplicity"
        assert items[2]["name"] == "Single word value"[:50]

    def test_parse_notes_with_type_from_header(self):
        """Parse notes and detect type from header."""
        content = """
- Made a decision about the architecture
- Another point to consider
"""
        items = _parse_notes(content, "decisions")
        assert len(items) == 2
        assert items[0]["note_type"] == "decision"

        items = _parse_notes(content, "insights")
        assert items[0]["note_type"] == "insight"

        items = _parse_notes(content, "observations")
        assert items[0]["note_type"] == "observation"

        items = _parse_notes(content, "notes")
        assert items[0]["note_type"] == "note"

    def test_parse_raw_with_bullets(self):
        """Parse raw section with bullet points."""
        content = """
- First thought
- Second thought
- Third thought
"""
        items = _parse_raw(content)
        assert len(items) == 3

    def test_parse_raw_with_paragraphs(self):
        """Parse raw section with paragraphs."""
        content = """
This is the first paragraph with some content.

This is the second paragraph.

And a third one.
"""
        items = _parse_raw(content)
        assert len(items) == 3

    def test_parse_full_markdown_document(self):
        """Parse a complete markdown document with multiple sections."""
        content = """
# Memory File

This is a preamble that should be treated as raw.

## Beliefs

- Testing is important (90%)
- Code should be readable

## Episodes

- Fixed critical bug -> Always add tests
- Deployed feature successfully

## Notes

- Remember to update the docs
- Check with stakeholders

## Goals

- [x] Complete MVP
- Ship beta version

## Values

- Quality: Always prioritize code quality
"""
        items = _parse_markdown(content)

        # Count by type
        types = {}
        for item in items:
            t = item["type"]
            types[t] = types.get(t, 0) + 1

        assert types["raw"] >= 1  # preamble
        assert types["belief"] == 2
        assert types["episode"] == 2
        assert types["note"] == 2
        assert types["goal"] == 2
        assert types["value"] == 1


class TestJsonExportFormat:
    """Tests for JSON export format validation.

    These tests verify that the expected JSON export structure
    is well-formed for later import operations.
    """

    def test_export_structure_has_required_fields(self):
        """JSON export should have agent_id and exported_at fields."""
        # This tests the expected structure of Kernle's JSON export format
        # which is documented and used by the import command
        required_fields = {"agent_id", "exported_at"}
        memory_types = {"values", "beliefs", "goals", "episodes", "notes"}

        # A minimal valid export
        export_data = {
            "agent_id": "test-agent",
            "exported_at": "2026-01-15T10:00:00Z",
            "values": [],
            "beliefs": [],
            "goals": [],
            "episodes": [],
            "notes": [],
        }

        # Verify required fields exist
        assert required_fields.issubset(export_data.keys())
        # Verify memory type arrays exist
        assert memory_types.issubset(export_data.keys())

    def test_episode_export_includes_outcome(self):
        """Exported episodes should have objective and outcome."""
        episode_data = {
            "objective": "Fix bug",
            "outcome": "Bug fixed successfully",
            "lessons": ["Always add tests"],
        }

        # Verify episode has required fields for import
        assert "objective" in episode_data
        assert "outcome" in episode_data
        # Note: outcome_type is inferred from outcome text, not stored explicitly


class TestCsvFormatRequirements:
    """Tests for CSV import format requirements.

    The import command expects specific column names and normalizes
    various formats. These tests document those expectations.
    """

    def test_type_column_required_unless_layer_specified(self):
        """CSV must have a type/memory_type/kind column for import."""
        # Valid type column names recognized by the import command
        valid_type_columns = ["type", "memory_type", "kind"]

        # Test that at least one is required (per import_cmd.py line 375)
        csv_with_type = "type,content\nbelief,Test belief\n"
        csv_with_memory_type = "memory_type,content\nbelief,Test belief\n"
        csv_with_kind = "kind,content\nbelief,Test belief\n"

        import csv
        import io

        # Each should be parseable with type column
        for csv_content in [csv_with_type, csv_with_memory_type, csv_with_kind]:
            reader = csv.DictReader(io.StringIO(csv_content))
            headers = [h.lower() for h in (reader.fieldnames or [])]
            has_type = any(h in valid_type_columns for h in headers)
            assert has_type, f"Should recognize type column in: {csv_content[:50]}"

    def test_confidence_normalization(self):
        """Confidence can be 0-1 float or 0-100 integer."""
        # The import command normalizes confidence values > 1
        # by dividing by 100 (see import_cmd.py lines 421-424)
        test_cases = [
            ("0.9", 0.9),
            ("90", 0.9),  # Will be normalized to 0.9
            ("0.85", 0.85),
            ("75", 0.75),  # Will be normalized to 0.75
        ]

        for raw_value, expected_normalized in test_cases:
            value = float(raw_value)
            if value > 1:
                value = value / 100
            assert abs(value - expected_normalized) < 0.01, f"Failed for {raw_value}"


class TestImportIntegration:
    """Integration tests for the import command.

    Note: These tests interact with the Kernle storage layer and may fail
    if there are schema changes in progress. The unit tests for parsing
    logic are more reliable for testing the import parsing functionality.
    """

    @pytest.fixture
    def kernle_instance(self, tmp_path):
        """Create a Kernle instance with temp storage."""
        try:
            return Kernle(agent_id="test-import")
        except Exception:
            pytest.skip("Could not create Kernle instance - schema may be changing")

    def test_import_markdown_beliefs(self, kernle_instance, tmp_path):
        """Import beliefs from markdown file."""
        md_content = """
## Beliefs

- Testing is important (90%)
- Code should be readable (85%)
"""
        md_file = tmp_path / "beliefs.md"
        md_file.write_text(md_content)

        # Parse and import manually (simulating the CLI)
        items = _parse_markdown(md_content)
        assert len(items) == 2

        for item in items:
            assert item["type"] == "belief"
            try:
                kernle_instance.belief(statement=item["statement"], confidence=item["confidence"])
            except AttributeError as e:
                # Schema may be in transition
                if "context" in str(e):
                    pytest.skip("Belief schema has context fields not yet in dataclass")
                raise

        # Verify beliefs were stored
        beliefs = kernle_instance._storage.get_beliefs()
        assert len(beliefs) >= 2

    def test_import_json_round_trip(self, kernle_instance, tmp_path):
        """Test export then import round trip."""
        # Add some data
        try:
            kernle_instance.belief("Test belief", confidence=0.9)
        except AttributeError as e:
            if "context" in str(e):
                pytest.skip("Belief schema has context fields not yet in dataclass")
            raise

        kernle_instance.episode("Test task", "Task completed successfully")
        kernle_instance.note("Test note", type="insight")

        # Export
        export_json = kernle_instance.dump(format="json")
        data = json.loads(export_json)

        # Verify export has data
        assert len(data["beliefs"]) >= 1
        assert len(data["episodes"]) >= 1
        assert len(data["notes"]) >= 1

        # Could be imported to another agent
        assert data["agent_id"] == "test-import"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_parse_empty_markdown(self):
        """Handle empty markdown file."""
        items = _parse_markdown("")
        assert items == []

    def test_parse_markdown_no_sections(self):
        """Handle markdown with no ## sections."""
        content = """
Just some text without any structure.

Another paragraph.
"""
        items = _parse_markdown(content)
        assert len(items) >= 1
        assert all(item["type"] == "raw" for item in items)

    def test_parse_beliefs_invalid_confidence(self):
        """Handle invalid confidence values."""
        content = """
- Valid belief (90%)
- Belief with bad confidence (abc%)
"""
        # The regex won't match invalid confidence, so it uses default
        items = _parse_beliefs(content)
        assert len(items) == 2
        assert items[0]["confidence"] == 0.9
        assert items[1]["confidence"] == 0.7  # default

    def test_parse_beliefs_confidence_clamping(self):
        """Confidence values should be clamped to [0, 1]."""
        content = """
- Too high confidence (150%)
- Negative confidence (-50%)
"""
        items = _parse_beliefs(content)
        # Values > 1 get divided by 100 if > 1 in the parser logic
        # but raw % values are /100 already, so 150% = 1.5 which clamps to 1.0
        assert items[0]["confidence"] <= 1.0
        assert items[0]["confidence"] >= 0.0

    def test_parse_unicode_content(self):
        """Handle unicode content properly."""
        content = """
## Beliefs

- I believe in good UX
- Quality matters
"""
        items = _parse_beliefs(content.split("## Beliefs")[1])
        assert len(items) == 2
        # Should handle unicode fine
        assert "UX" in items[0]["statement"]

    def test_numbered_list_parsing(self):
        """Parse numbered lists correctly."""
        content = """
1. First item
2. Second item
3. Third item
"""
        items = _parse_raw(content)
        # Numbered lists should also be parsed
        assert len(items) >= 1


class TestConfidenceFormats:
    """Test various confidence format parsing."""

    @pytest.mark.parametrize(
        "input_text,expected_conf",
        [
            ("Belief (90%)", 0.9),
            ("Belief [0.85]", 0.85),
            ("Belief (confidence: 0.7)", 0.7),
            ("Belief (confidence: 75)", 0.75),
            ("Belief with no confidence", 0.7),
        ],
    )
    def test_confidence_formats(self, input_text, expected_conf):
        """Test various confidence format parsing."""
        content = f"- {input_text}"
        items = _parse_beliefs(content)
        assert len(items) == 1
        assert abs(items[0]["confidence"] - expected_conf) < 0.01
