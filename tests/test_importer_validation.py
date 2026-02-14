"""Tests for importer schema coercion rules (issue #716).

Tests cover:
- Bounds validation for confidence, priority, intensity, sentiment
- Coercion warnings tracking
- Strict mode rejections vs permissive mode defaults
- Rejection reason tracking for silent skips
"""

import json

import pytest

from kernle.importers.csv_importer import (
    CsvImporter,
    _map_columns,
    parse_csv,
)
from kernle.importers.json_importer import (
    JsonImporter,
    JsonImportItem,
    _import_json_item,
)

# ============================================================================
# CSV Importer: Confidence validation
# ============================================================================


class TestCsvConfidenceValidation:
    """Test confidence bounds validation in CSV importer."""

    def test_csv_confidence_out_of_bounds_clamped(self):
        """confidence=150 (>100) auto-scales to 1.5, then clamped to 1.0 in permissive mode."""
        row = {"statement": "Test belief", "confidence": "150"}
        result, warnings, _ = _map_columns(row, "belief")
        # 150 > 1 triggers divide-by-100 = 1.5, which is > 1.0, so clamp to 1.0
        assert result["confidence"] == 1.0
        assert len(warnings) == 1
        assert warnings[0]["field"] == "confidence"

    def test_csv_confidence_negative_clamped(self):
        """confidence=-0.5 should be clamped to 0.0 in permissive mode."""
        row = {"statement": "Test belief", "confidence": "-0.5"}
        result, _, _ = _map_columns(row, "belief")
        assert result["confidence"] == 0.0

    def test_csv_confidence_non_numeric_strict_rejects(self):
        """confidence='high' in strict mode should reject the row."""
        items = parse_csv(
            """type,statement,confidence
belief,Test belief,high
""",
            strict=True,
        )
        assert len(items) == 0

    def test_csv_confidence_non_numeric_permissive_defaults(self):
        """confidence='high' in permissive mode defaults to 0.7 with warning."""
        items, warnings = parse_csv(
            """type,statement,confidence
belief,Test belief,high
""",
            strict=False,
            return_warnings=True,
        )
        assert len(items) == 1
        assert items[0].data["confidence"] == 0.7
        assert len(warnings) >= 1
        assert warnings[0]["field"] == "confidence"
        assert warnings[0]["original"] == "high"
        assert warnings[0]["coerced_to"] == 0.7

    def test_csv_confidence_value_over_100_rejected_strict(self):
        """confidence=150 should be rejected in strict mode."""
        items = parse_csv(
            """type,statement,confidence
belief,Test belief,150
""",
            strict=True,
        )
        assert len(items) == 0

    def test_csv_confidence_value_over_100_clamped_permissive(self):
        """confidence=150 in permissive mode: 150/100=1.5, clamped to 1.0."""
        items, warnings = parse_csv(
            """type,statement,confidence
belief,Test belief,150
""",
            strict=False,
            return_warnings=True,
        )
        assert len(items) == 1
        assert items[0].data["confidence"] == 1.0
        assert any(w["field"] == "confidence" for w in warnings)

    def test_csv_confidence_valid_percentage_auto_scaled(self):
        """confidence=85 should auto-scale to 0.85 (existing behavior preserved)."""
        row = {"statement": "Test belief", "confidence": "85"}
        result, _, _ = _map_columns(row, "belief")
        assert result["confidence"] == pytest.approx(0.85)

    def test_csv_confidence_valid_decimal_preserved(self):
        """confidence=0.92 should be preserved as-is."""
        row = {"statement": "Test belief", "confidence": "0.92"}
        result, _, _ = _map_columns(row, "belief")
        assert result["confidence"] == pytest.approx(0.92)


# ============================================================================
# CSV Importer: Priority validation
# ============================================================================


class TestCsvPriorityValidation:
    """Test priority bounds validation in CSV importer."""

    def test_csv_priority_out_of_bounds_clamped(self):
        """priority=200 should be clamped to 100 in permissive mode."""
        row = {"name": "Quality", "priority": "200"}
        result, _, _ = _map_columns(row, "value")
        assert result["priority"] == 100

    def test_csv_priority_negative_clamped(self):
        """priority=-10 should be clamped to 0 in permissive mode."""
        row = {"name": "Quality", "priority": "-10"}
        result, _, _ = _map_columns(row, "value")
        assert result["priority"] == 0

    def test_csv_priority_non_numeric_strict_rejects(self):
        """priority='high' in strict mode should reject the row."""
        items = parse_csv(
            """type,name,description,priority
value,Quality,Code quality,high
""",
            strict=True,
        )
        assert len(items) == 0

    def test_csv_priority_non_numeric_permissive_defaults(self):
        """priority='high' in permissive mode defaults to 50 with warning."""
        items, warnings = parse_csv(
            """type,name,description,priority
value,Quality,Code quality,high
""",
            strict=False,
            return_warnings=True,
        )
        assert len(items) == 1
        assert items[0].data["priority"] == 50
        assert any(w["field"] == "priority" for w in warnings)


# ============================================================================
# CSV Importer: Intensity validation
# ============================================================================


class TestCsvIntensityValidation:
    """Test intensity bounds validation in CSV importer (via _map_columns)."""

    def test_csv_intensity_out_of_bounds_clamped(self):
        """intensity=2.0 should be clamped to 1.0."""
        # Intensity is relevant for drives in JSON, but _map_columns should
        # handle it if it ever appears in CSV. We test via JSON importer below.
        pass


# ============================================================================
# JSON Importer: Intensity validation
# ============================================================================


class TestJsonIntensityValidation:
    """Test intensity bounds validation in JSON importer."""

    def test_json_intensity_out_of_bounds_clamped(self, kernle_instance):
        """intensity=2.0 should be clamped to 1.0 in permissive mode."""
        k, storage = kernle_instance
        item = JsonImportItem(
            type="drive",
            data={"drive_type": "curiosity", "intensity": 2.0},
        )
        result = _import_json_item(item, k, skip_duplicates=False)
        assert result is True

        drives = storage.get_drives()
        assert len(drives) == 1
        assert drives[0].intensity <= 1.0

    def test_json_intensity_negative_clamped(self, kernle_instance):
        """intensity=-0.5 should be clamped to 0.0."""
        k, storage = kernle_instance
        item = JsonImportItem(
            type="drive",
            data={"drive_type": "growth", "intensity": -0.5},
        )
        result = _import_json_item(item, k, skip_duplicates=False)
        assert result is True

        drives = storage.get_drives()
        assert len(drives) == 1
        assert drives[0].intensity >= 0.0

    def test_json_intensity_strict_rejects_out_of_bounds(self, kernle_instance):
        """intensity=2.0 in strict mode should reject the item."""
        k, storage = kernle_instance
        item = JsonImportItem(
            type="drive",
            data={"drive_type": "curiosity", "intensity": 2.0},
        )
        result = _import_json_item(item, k, skip_duplicates=False, strict=True)
        assert result is False


# ============================================================================
# JSON Importer: Sentiment validation
# ============================================================================


class TestJsonSentimentValidation:
    """Test sentiment bounds validation in JSON importer."""

    def test_json_sentiment_out_of_bounds_clamped(self, kernle_instance):
        """sentiment=5.0 should be clamped to 1.0 in permissive mode."""
        k, storage = kernle_instance
        item = JsonImportItem(
            type="relationship",
            data={
                "entity_name": "TestEntity",
                "entity_type": "person",
                "relationship_type": "friend",
                "sentiment": 5.0,
            },
        )
        result = _import_json_item(item, k, skip_duplicates=False)
        assert result is True

        rel = storage.get_relationship("TestEntity")
        assert rel is not None
        # sentiment 5.0 clamped to 1.0, stored as sentiment on the object
        assert rel.sentiment <= 1.0

    def test_json_sentiment_negative_out_of_bounds_clamped(self, kernle_instance):
        """sentiment=-5.0 should be clamped to -1.0."""
        k, storage = kernle_instance
        item = JsonImportItem(
            type="relationship",
            data={
                "entity_name": "NegEntity",
                "entity_type": "person",
                "relationship_type": "rival",
                "sentiment": -5.0,
            },
        )
        result = _import_json_item(item, k, skip_duplicates=False)
        assert result is True

        rel = storage.get_relationship("NegEntity")
        assert rel is not None
        # sentiment -5.0 clamped to -1.0
        assert rel.sentiment >= -1.0

    def test_json_sentiment_strict_rejects_out_of_bounds(self, kernle_instance):
        """sentiment=5.0 in strict mode should reject the item."""
        k, storage = kernle_instance
        item = JsonImportItem(
            type="relationship",
            data={
                "entity_name": "TestEntity",
                "entity_type": "person",
                "relationship_type": "friend",
                "sentiment": 5.0,
            },
        )
        result = _import_json_item(item, k, skip_duplicates=False, strict=True)
        assert result is False


# ============================================================================
# JSON Importer: Confidence validation
# ============================================================================


class TestJsonConfidenceValidation:
    """Test confidence bounds validation in JSON importer."""

    def test_json_confidence_out_of_bounds_clamped(self, kernle_instance):
        """confidence=1.5 should be clamped to 1.0."""
        k, storage = kernle_instance
        item = JsonImportItem(
            type="belief",
            data={"statement": "Test belief clamped", "confidence": 1.5},
        )
        result = _import_json_item(item, k, skip_duplicates=False)
        assert result is True

        beliefs = storage.get_beliefs()
        assert len(beliefs) == 1
        assert beliefs[0].confidence <= 1.0

    def test_json_confidence_negative_clamped(self, kernle_instance):
        """confidence=-0.5 should be clamped to 0.0."""
        k, storage = kernle_instance
        item = JsonImportItem(
            type="belief",
            data={"statement": "Negative confidence", "confidence": -0.5},
        )
        result = _import_json_item(item, k, skip_duplicates=False)
        assert result is True

        beliefs = storage.get_beliefs()
        assert len(beliefs) == 1
        assert beliefs[0].confidence >= 0.0


# ============================================================================
# JSON Importer: Priority validation
# ============================================================================


class TestJsonPriorityValidation:
    """Test priority bounds validation in JSON importer."""

    def test_json_value_priority_out_of_bounds_clamped(self, kernle_instance):
        """priority=200 should be clamped to 100."""
        k, storage = kernle_instance
        item = JsonImportItem(
            type="value",
            data={"name": "Excessive Priority", "statement": "Test", "priority": 200},
        )
        result = _import_json_item(item, k, skip_duplicates=False)
        assert result is True

        values = storage.get_values()
        assert len(values) == 1
        assert values[0].priority <= 100


# ============================================================================
# Coercion Warnings Tracking
# ============================================================================


class TestCoercionWarningsTracked:
    """Test that coercion warnings are properly tracked and returned."""

    def test_coercion_warnings_tracked(self):
        """Verify warnings list is populated when values are coerced."""
        csv_content = """type,statement,confidence
belief,Belief with bad conf,invalid_value
belief,Belief clamped negative,-0.5
"""
        items, warnings = parse_csv(csv_content, strict=False, return_warnings=True)
        assert len(items) == 2
        assert len(warnings) >= 2

        # Check warning structure
        for w in warnings:
            assert "row" in w
            assert "field" in w
            assert "original" in w
            assert "coerced_to" in w
            assert "reason" in w

    def test_no_warnings_for_valid_data(self):
        """No warnings should be generated for valid data."""
        csv_content = """type,statement,confidence
belief,Valid belief,0.9
belief,Another valid,0.7
"""
        items, warnings = parse_csv(csv_content, strict=False, return_warnings=True)
        assert len(items) == 2
        assert len(warnings) == 0

    def test_json_coercion_warnings_in_import_result(self, tmp_path, kernle_instance):
        """JSON importer should include coercion_warnings in result."""
        k, storage = kernle_instance
        json_file = tmp_path / "test.json"
        json_file.write_text(
            json.dumps(
                {
                    "drives": [
                        {"drive_type": "curiosity", "intensity": 2.0},
                    ],
                }
            )
        )
        importer = JsonImporter(str(json_file))
        result = importer.import_to(k, dry_run=False, skip_duplicates=False)
        assert result["imported"]["drive"] == 1
        assert "coercion_warnings" in result
        assert len(result["coercion_warnings"]) >= 1


# ============================================================================
# Strict Mode
# ============================================================================


class TestStrictModeRejectsMalformed:
    """Test that strict mode rejects malformed values instead of defaulting."""

    def test_strict_mode_rejects_malformed(self):
        """strict=True should reject rows with non-numeric confidence/priority."""
        csv_content = """type,statement,confidence
belief,Good belief,0.9
belief,Bad confidence,not_a_number
"""
        items = parse_csv(csv_content, strict=True)
        # Only the valid row should survive
        assert len(items) == 1
        assert items[0].data["statement"] == "Good belief"

    def test_strict_mode_rejects_out_of_range_confidence(self):
        """strict=True should reject rows with confidence > 100."""
        items = parse_csv(
            """type,statement,confidence
belief,Over 100,150
""",
            strict=True,
        )
        assert len(items) == 0

    def test_strict_mode_rejects_negative_confidence(self):
        """strict=True should reject rows with confidence < 0."""
        items = parse_csv(
            """type,statement,confidence
belief,Negative,-0.5
""",
            strict=True,
        )
        assert len(items) == 0

    def test_strict_mode_preserves_valid_rows(self):
        """strict=True should not reject valid data."""
        csv_content = """type,statement,confidence
belief,Valid belief,0.9
belief,Another valid,85
"""
        items = parse_csv(csv_content, strict=True)
        assert len(items) == 2


# ============================================================================
# Rejection Reasons Tracking
# ============================================================================


class TestRejectionReasonsTracked:
    """Test that rejections (silent skips) are reported with reason."""

    def test_rejection_reasons_tracked(self):
        """Rows rejected in strict mode should report reason."""
        csv_content = """type,statement,confidence
belief,Valid belief,0.9
belief,Bad confidence,xyz
belief,Too high,999
"""
        items, warnings, rejections = parse_csv(
            csv_content, strict=True, return_warnings=True, return_rejections=True
        )
        assert len(items) == 1
        assert len(rejections) >= 2

        # Check rejection structure
        for r in rejections:
            assert "row" in r
            assert "field" in r
            assert "value" in r
            assert "reason" in r

    def test_csv_importer_import_result_includes_rejections(self, tmp_path, kernle_instance):
        """CsvImporter.import_to result should include rejections list."""
        k, storage = kernle_instance
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("""type,statement,confidence
belief,Valid belief,0.9
belief,Missing statement,
""")
        importer = CsvImporter(str(csv_file))
        result = importer.import_to(k, dry_run=False, skip_duplicates=False)
        assert "rejections" in result


# ============================================================================
# Backwards Compatibility
# ============================================================================


class TestBackwardsCompatibility:
    """Verify that default behavior remains permissive (no breaking changes)."""

    def test_default_parse_csv_is_permissive(self):
        """parse_csv() without strict parameter uses permissive mode."""
        csv_content = """type,statement,confidence
belief,Test,invalid
"""
        items = parse_csv(csv_content)
        # Should still produce an item with default confidence
        assert len(items) == 1
        assert items[0].data["confidence"] == 0.7

    def test_default_parse_csv_returns_items_only(self):
        """parse_csv() without return_warnings returns items only."""
        csv_content = """type,statement,confidence
belief,Test,invalid
"""
        result = parse_csv(csv_content)
        # Should return a list, not a tuple
        assert isinstance(result, list)

    def test_csv_importer_default_not_strict(self, tmp_path, kernle_instance):
        """CsvImporter default import is not strict."""
        k, storage = kernle_instance
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("""type,statement,confidence
belief,Permissive test,bad_value
""")
        importer = CsvImporter(str(csv_file))
        result = importer.import_to(k, dry_run=False, skip_duplicates=False)
        assert result["imported"].get("belief", 0) == 1

    def test_json_importer_default_not_strict(self, tmp_path, kernle_instance):
        """JsonImporter default import is not strict."""
        k, storage = kernle_instance
        json_file = tmp_path / "test.json"
        json_file.write_text(
            json.dumps(
                {
                    "drives": [
                        {"drive_type": "curiosity", "intensity": 2.0},
                    ],
                }
            )
        )
        importer = JsonImporter(str(json_file))
        result = importer.import_to(k, dry_run=False, skip_duplicates=False)
        assert result["imported"]["drive"] == 1
