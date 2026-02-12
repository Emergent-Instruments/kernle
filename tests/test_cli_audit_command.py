"""Tests for kernle.cli.commands.audit."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from kernle.cli.commands.audit import cmd_audit


def _report():
    assertion = SimpleNamespace(
        category="quality",
        name="test_assertion",
        passed=True,
        message="ok",
        details={"sample": 1},
    )
    report = MagicMock()
    report.passed = 5
    report.failed = 1
    report.total = 6
    report.all_passed = False
    report.assertions = [assertion]
    report.summary.return_value = "audit summary"
    return report


class TestCmdAudit:
    @pytest.mark.parametrize(
        ("category", "expected_method"),
        [
            ("structural", "run_structural"),
            ("coherence", "run_coherence"),
            ("quality", "run_quality"),
            ("pipeline", "run_pipeline"),
            (None, "run_all"),
        ],
    )
    def test_routes_to_expected_assertion_set(self, category, expected_method, capsys):
        args = SimpleNamespace(category=category, json=False)
        k = MagicMock()
        report = _report()
        assertions = MagicMock()
        getattr(assertions, expected_method).return_value = report

        with patch("kernle.testing.assertions.CognitiveAssertions", return_value=assertions):
            cmd_audit(args, k)

        getattr(assertions, expected_method).assert_called_once()
        assert "audit summary" in capsys.readouterr().out

    def test_json_output_shape(self, capsys):
        args = SimpleNamespace(category="quality", json=True)
        k = MagicMock()
        report = _report()
        assertions = MagicMock()
        assertions.run_quality.return_value = report

        with patch("kernle.testing.assertions.CognitiveAssertions", return_value=assertions):
            cmd_audit(args, k)

        output = json.loads(capsys.readouterr().out)
        assert output["passed"] == 5
        assert output["failed"] == 1
        assert output["all_passed"] is False
        assert output["assertions"][0]["name"] == "test_assertion"
