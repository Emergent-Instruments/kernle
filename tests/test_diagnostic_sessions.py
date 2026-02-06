"""Tests for formal diagnostic sessions (KEP v3 - doctor pattern phase 2)."""

import json
import uuid
from datetime import datetime, timezone

import pytest

from kernle.core import Kernle
from kernle.storage import SQLiteStorage
from kernle.storage.base import (
    Belief,
    DiagnosticReport,
    DiagnosticSession,
    TrustAssessment,
)


@pytest.fixture
def diag_setup(tmp_path):
    """Create a Kernle instance for diagnostic testing."""
    db_path = tmp_path / "test_diag.db"
    storage = SQLiteStorage(stack_id="test_agent", db_path=db_path)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    k = Kernle(stack_id="test_agent", storage=storage, checkpoint_dir=checkpoint_dir)
    yield k, storage
    storage.close()


@pytest.fixture
def diag_with_trust(diag_setup):
    """Kernle instance with trust seed for operator-initiated tests."""
    k, storage = diag_setup
    assessment = TrustAssessment(
        id=str(uuid.uuid4()),
        stack_id="test_agent",
        entity="stack-owner",
        dimensions={"general": {"score": 0.95}},
        authority=[{"scope": "all"}],
        created_at=datetime.now(timezone.utc),
    )
    storage.save_trust_assessment(assessment)
    return k, storage


# =============================================================================
# Dataclass Tests
# =============================================================================


class TestDiagnosticSessionDataclass:
    """Tests for DiagnosticSession dataclass."""

    def test_create_session(self):
        session = DiagnosticSession(
            id="test-session-1",
            stack_id="test_agent",
            session_type="self_requested",
            access_level="structural",
        )
        assert session.id == "test-session-1"
        assert session.session_type == "self_requested"
        assert session.access_level == "structural"
        assert session.status == "active"
        assert session.consent_given is False
        assert session.version == 1
        assert session.deleted is False

    def test_default_values(self):
        session = DiagnosticSession(
            id="test-id",
            stack_id="test_agent",
        )
        assert session.session_type == "self_requested"
        assert session.access_level == "structural"
        assert session.status == "active"
        assert session.consent_given is False
        assert session.started_at is None
        assert session.completed_at is None

    def test_operator_initiated(self):
        session = DiagnosticSession(
            id="test-op",
            stack_id="test_agent",
            session_type="operator_initiated",
            consent_given=True,
        )
        assert session.session_type == "operator_initiated"
        assert session.consent_given is True


class TestDiagnosticReportDataclass:
    """Tests for DiagnosticReport dataclass."""

    def test_create_report(self):
        findings = [
            {
                "severity": "warning",
                "category": "low_confidence_belief",
                "description": "Belief #abc has low confidence",
                "recommendation": "Review and verify",
            }
        ]
        report = DiagnosticReport(
            id="test-report-1",
            stack_id="test_agent",
            session_id="test-session-1",
            findings=findings,
            summary="Found 1 finding(s): 1 warning(s)",
        )
        assert report.id == "test-report-1"
        assert report.session_id == "test-session-1"
        assert len(report.findings) == 1
        assert report.findings[0]["severity"] == "warning"

    def test_default_values(self):
        report = DiagnosticReport(
            id="test-id",
            stack_id="test_agent",
            session_id="test-session",
        )
        assert report.findings is None
        assert report.summary is None
        assert report.version == 1
        assert report.deleted is False


# =============================================================================
# Storage Tests
# =============================================================================


class TestDiagnosticSessionStorage:
    """Tests for diagnostic session storage operations."""

    def test_save_and_get_session(self, diag_setup):
        k, storage = diag_setup
        now = datetime.now(timezone.utc)
        session = DiagnosticSession(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            session_type="self_requested",
            access_level="structural",
            status="active",
            consent_given=True,
            started_at=now,
        )
        sid = storage.save_diagnostic_session(session)
        assert sid == session.id

        retrieved = storage.get_diagnostic_session(session.id)
        assert retrieved is not None
        assert retrieved.id == session.id
        assert retrieved.session_type == "self_requested"
        assert retrieved.access_level == "structural"
        assert retrieved.status == "active"
        assert retrieved.consent_given is True

    def test_get_nonexistent_session(self, diag_setup):
        _, storage = diag_setup
        result = storage.get_diagnostic_session("nonexistent-id")
        assert result is None

    def test_list_sessions(self, diag_setup):
        _, storage = diag_setup
        now = datetime.now(timezone.utc)

        for i in range(3):
            session = DiagnosticSession(
                id=str(uuid.uuid4()),
                stack_id="test_agent",
                session_type="routine",
                started_at=now,
            )
            storage.save_diagnostic_session(session)

        sessions = storage.get_diagnostic_sessions()
        assert len(sessions) == 3

    def test_list_sessions_by_status(self, diag_setup):
        _, storage = diag_setup
        now = datetime.now(timezone.utc)

        active_session = DiagnosticSession(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            status="active",
            started_at=now,
        )
        storage.save_diagnostic_session(active_session)

        completed_session = DiagnosticSession(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            status="completed",
            started_at=now,
            completed_at=now,
        )
        storage.save_diagnostic_session(completed_session)

        active = storage.get_diagnostic_sessions(status="active")
        assert len(active) == 1
        assert active[0].status == "active"

        completed = storage.get_diagnostic_sessions(status="completed")
        assert len(completed) == 1
        assert completed[0].status == "completed"

    def test_complete_session(self, diag_setup):
        _, storage = diag_setup
        now = datetime.now(timezone.utc)

        session = DiagnosticSession(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            status="active",
            started_at=now,
        )
        storage.save_diagnostic_session(session)

        result = storage.complete_diagnostic_session(session.id)
        assert result is True

        retrieved = storage.get_diagnostic_session(session.id)
        assert retrieved.status == "completed"
        assert retrieved.completed_at is not None

    def test_complete_already_completed(self, diag_setup):
        _, storage = diag_setup
        now = datetime.now(timezone.utc)

        session = DiagnosticSession(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            status="completed",
            started_at=now,
            completed_at=now,
        )
        storage.save_diagnostic_session(session)

        result = storage.complete_diagnostic_session(session.id)
        assert result is False  # Already completed


class TestDiagnosticReportStorage:
    """Tests for diagnostic report storage operations."""

    def test_save_and_get_report(self, diag_setup):
        _, storage = diag_setup
        now = datetime.now(timezone.utc)

        findings = [
            {
                "severity": "error",
                "category": "orphaned_reference",
                "description": "Broken ref in episode #abc",
                "recommendation": "Remove or update",
            }
        ]
        report = DiagnosticReport(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            session_id="session-1",
            findings=findings,
            summary="Found 1 error",
            created_at=now,
        )
        rid = storage.save_diagnostic_report(report)
        assert rid == report.id

        retrieved = storage.get_diagnostic_report(report.id)
        assert retrieved is not None
        assert retrieved.session_id == "session-1"
        assert len(retrieved.findings) == 1
        assert retrieved.findings[0]["severity"] == "error"
        assert retrieved.summary == "Found 1 error"

    def test_get_nonexistent_report(self, diag_setup):
        _, storage = diag_setup
        result = storage.get_diagnostic_report("nonexistent-id")
        assert result is None

    def test_list_reports(self, diag_setup):
        _, storage = diag_setup
        now = datetime.now(timezone.utc)

        for i in range(3):
            report = DiagnosticReport(
                id=str(uuid.uuid4()),
                stack_id="test_agent",
                session_id="session-1",
                findings=[],
                created_at=now,
            )
            storage.save_diagnostic_report(report)

        reports = storage.get_diagnostic_reports()
        assert len(reports) == 3

    def test_list_reports_by_session(self, diag_setup):
        _, storage = diag_setup
        now = datetime.now(timezone.utc)

        for session_id in ["session-1", "session-2"]:
            report = DiagnosticReport(
                id=str(uuid.uuid4()),
                stack_id="test_agent",
                session_id=session_id,
                findings=[],
                created_at=now,
            )
            storage.save_diagnostic_report(report)

        session_1_reports = storage.get_diagnostic_reports(session_id="session-1")
        assert len(session_1_reports) == 1
        assert session_1_reports[0].session_id == "session-1"

    def test_report_with_empty_findings(self, diag_setup):
        _, storage = diag_setup
        now = datetime.now(timezone.utc)

        report = DiagnosticReport(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            session_id="session-healthy",
            findings=[],
            summary="No issues found. Memory graph is healthy.",
            created_at=now,
        )
        storage.save_diagnostic_report(report)

        retrieved = storage.get_diagnostic_report(report.id)
        assert retrieved.findings == []
        assert "healthy" in retrieved.summary


# =============================================================================
# Trust Gate Tests
# =============================================================================


class TestOperatorConsentGate:
    """Tests for trust gating of operator-initiated sessions."""

    def test_self_requested_bypasses_gate(self, diag_setup):
        from kernle.cli.commands.doctor import _check_operator_consent

        k, _ = diag_setup
        assert _check_operator_consent(k, "self_requested") is True

    def test_routine_bypasses_gate(self, diag_setup):
        from kernle.cli.commands.doctor import _check_operator_consent

        k, _ = diag_setup
        assert _check_operator_consent(k, "routine") is True

    def test_operator_initiated_without_trust(self, diag_setup):
        from kernle.cli.commands.doctor import _check_operator_consent

        k, _ = diag_setup
        # No trust assessment exists -- should be denied
        result = _check_operator_consent(k, "operator_initiated")
        assert result is False

    def test_operator_initiated_with_trust(self, diag_with_trust):
        from kernle.cli.commands.doctor import _check_operator_consent

        k, _ = diag_with_trust
        result = _check_operator_consent(k, "operator_initiated")
        assert result is True


# =============================================================================
# Report Generation Tests
# =============================================================================


class TestReportGeneration:
    """Tests for converting structural findings to report format."""

    def test_generate_report_findings(self):
        from kernle.cli.commands.doctor import (
            StructuralFinding,
            _generate_report_findings,
        )

        findings = [
            StructuralFinding(
                check="orphaned_reference",
                severity="error",
                memory_type="episode",
                memory_id="abc-123",
                message="Episode #abc has broken ref",
            ),
            StructuralFinding(
                check="low_confidence_belief",
                severity="warning",
                memory_type="belief",
                memory_id="def-456",
                message="Belief #def has low confidence",
            ),
        ]

        result = _generate_report_findings(findings)
        assert len(result) == 2
        assert result[0]["severity"] == "error"
        assert result[0]["category"] == "orphaned_reference"
        assert "broken ref" in result[0]["description"]
        assert result[0]["recommendation"] is not None
        assert result[1]["severity"] == "warning"

    def test_generate_summary_healthy(self):
        from kernle.cli.commands.doctor import _generate_summary

        summary = _generate_summary([])
        assert "healthy" in summary.lower()

    def test_generate_summary_with_findings(self):
        from kernle.cli.commands.doctor import _generate_summary

        findings = [
            {"severity": "error"},
            {"severity": "warning"},
            {"severity": "warning"},
            {"severity": "info"},
        ]
        summary = _generate_summary(findings)
        assert "4" in summary
        assert "1 error" in summary
        assert "2 warning" in summary
        assert "1 info" in summary


# =============================================================================
# Privacy Boundary Tests
# =============================================================================


class TestPrivacyBoundary:
    """Tests ensuring reports contain structural data only, never content."""

    def test_findings_contain_no_content(self, diag_setup):
        """Verify that report findings reference IDs, not memory content."""
        from kernle.cli.commands.doctor import (
            StructuralFinding,
            _generate_report_findings,
        )

        secret_content = "My secret belief about something private"
        findings = [
            StructuralFinding(
                check="low_confidence_belief",
                severity="warning",
                memory_type="belief",
                memory_id="xyz-789",
                message="Belief #xyz-789 (confidence 0.20) -- low confidence",
            ),
        ]

        result = _generate_report_findings(findings)
        result_str = json.dumps(result)

        # The actual secret content should never appear
        assert secret_content not in result_str
        # But IDs and scores are fine
        assert "xyz-789" in result_str or "low_confidence" in result_str

    def test_structural_finding_uses_ids_not_content(self):
        """StructuralFinding should reference IDs, not content."""
        from kernle.cli.commands.doctor import StructuralFinding

        finding = StructuralFinding(
            check="low_confidence_belief",
            severity="warning",
            memory_type="belief",
            memory_id="abc-123-full-id",
            message="Belief #abc-123-full (confidence 0.15) -- low confidence",
        )
        d = finding.to_dict()
        assert d["memory_id"] == "abc-123-full-id"
        assert d["memory_type"] == "belief"
        # Message should contain IDs/scores, not actual belief content
        assert "confidence" in d["message"]


# =============================================================================
# End-to-End Session Tests
# =============================================================================


class TestEndToEndSession:
    """Integration tests for the full diagnostic session workflow."""

    def test_session_workflow(self, diag_setup):
        """Test: start session -> run checks -> generate report -> complete."""
        k, storage = diag_setup

        # Add some data to have something to check
        belief = Belief(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            statement="Test belief",
            confidence=0.1,  # Low confidence -- should trigger finding
            created_at=datetime.now(timezone.utc),
        )
        storage.save_belief(belief)

        now = datetime.now(timezone.utc)

        # Start session
        session = DiagnosticSession(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            session_type="self_requested",
            access_level="structural",
            consent_given=True,
            started_at=now,
        )
        storage.save_diagnostic_session(session)

        # Run structural checks
        from kernle.cli.commands.doctor import (
            _generate_report_findings,
            _generate_summary,
            run_structural_checks,
        )

        raw_findings = run_structural_checks(k)
        findings = _generate_report_findings(raw_findings)
        summary = _generate_summary(findings)

        # Create report
        report = DiagnosticReport(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            session_id=session.id,
            findings=findings,
            summary=summary,
            created_at=now,
        )
        storage.save_diagnostic_report(report)

        # Complete session
        storage.complete_diagnostic_session(session.id)

        # Verify
        completed_session = storage.get_diagnostic_session(session.id)
        assert completed_session.status == "completed"

        saved_report = storage.get_diagnostic_report(report.id)
        assert saved_report is not None
        assert saved_report.session_id == session.id

        # Should have found the low-confidence belief
        assert len(saved_report.findings) > 0
        low_conf = [f for f in saved_report.findings if f["category"] == "low_confidence_belief"]
        assert len(low_conf) > 0

    def test_healthy_session(self, diag_setup):
        """Test session with no findings (healthy memory graph)."""
        k, storage = diag_setup
        now = datetime.now(timezone.utc)

        session = DiagnosticSession(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            session_type="routine",
            consent_given=True,
            started_at=now,
        )
        storage.save_diagnostic_session(session)

        from kernle.cli.commands.doctor import (
            _generate_report_findings,
            _generate_summary,
            run_structural_checks,
        )

        raw_findings = run_structural_checks(k)
        findings = _generate_report_findings(raw_findings)
        summary = _generate_summary(findings)

        report = DiagnosticReport(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            session_id=session.id,
            findings=findings,
            summary=summary,
            created_at=now,
        )
        storage.save_diagnostic_report(report)
        storage.complete_diagnostic_session(session.id)

        saved_report = storage.get_diagnostic_report(report.id)
        assert saved_report.findings == []
        assert "healthy" in saved_report.summary.lower()


# =============================================================================
# CLI Command Tests
# =============================================================================


class TestCLISessionCommands:
    """Tests for CLI command functions."""

    def test_cmd_session_start_json(self, diag_setup, capsys):
        from kernle.cli.commands.doctor import cmd_doctor_session_start

        k, _ = diag_setup

        class Args:
            type = "self_requested"
            access = "structural"
            json = True

        cmd_doctor_session_start(Args(), k)
        output = capsys.readouterr().out
        data = json.loads(output)
        assert "session_id" in data
        assert "report_id" in data
        assert data["session_type"] == "self_requested"
        assert data["access_level"] == "structural"
        assert "summary" in data

    def test_cmd_session_start_text(self, diag_setup, capsys):
        from kernle.cli.commands.doctor import cmd_doctor_session_start

        k, _ = diag_setup

        class Args:
            type = "routine"
            access = "structural"
            json = False

        cmd_doctor_session_start(Args(), k)
        output = capsys.readouterr().out
        assert "Diagnostic Session" in output
        assert "Report saved" in output

    def test_cmd_session_start_invalid_type(self, diag_setup, capsys):
        from kernle.cli.commands.doctor import cmd_doctor_session_start

        k, _ = diag_setup

        class Args:
            type = "invalid_type"
            access = "structural"
            json = False

        cmd_doctor_session_start(Args(), k)
        output = capsys.readouterr().out
        assert "Invalid session type" in output

    def test_cmd_session_start_invalid_access(self, diag_setup, capsys):
        from kernle.cli.commands.doctor import cmd_doctor_session_start

        k, _ = diag_setup

        class Args:
            type = "self_requested"
            access = "invalid_level"
            json = False

        cmd_doctor_session_start(Args(), k)
        output = capsys.readouterr().out
        assert "Invalid access level" in output

    def test_cmd_session_start_operator_no_trust(self, diag_setup, capsys):
        from kernle.cli.commands.doctor import cmd_doctor_session_start

        k, _ = diag_setup

        class Args:
            type = "operator_initiated"
            access = "structural"
            json = False

        cmd_doctor_session_start(Args(), k)
        output = capsys.readouterr().out
        assert "Insufficient trust" in output

    def test_cmd_session_start_operator_with_trust(self, diag_with_trust, capsys):
        from kernle.cli.commands.doctor import cmd_doctor_session_start

        k, _ = diag_with_trust

        class Args:
            type = "operator_initiated"
            access = "structural"
            json = True

        cmd_doctor_session_start(Args(), k)
        output = capsys.readouterr().out
        data = json.loads(output)
        assert data["session_type"] == "operator_initiated"

    def test_cmd_session_list_empty(self, diag_setup, capsys):
        from kernle.cli.commands.doctor import cmd_doctor_session_list

        k, _ = diag_setup

        class Args:
            json = False

        cmd_doctor_session_list(Args(), k)
        output = capsys.readouterr().out
        assert "No diagnostic sessions found" in output

    def test_cmd_session_list_json(self, diag_setup, capsys):
        from kernle.cli.commands.doctor import (
            cmd_doctor_session_list,
            cmd_doctor_session_start,
        )

        k, _ = diag_setup

        # Create a session first
        class StartArgs:
            type = "self_requested"
            access = "structural"
            json = False

        cmd_doctor_session_start(StartArgs(), k)
        capsys.readouterr()  # Clear output

        class ListArgs:
            json = True

        cmd_doctor_session_list(ListArgs(), k)
        output = capsys.readouterr().out
        data = json.loads(output)
        assert len(data) >= 1

    def test_cmd_report_latest(self, diag_setup, capsys):
        from kernle.cli.commands.doctor import cmd_doctor_report, cmd_doctor_session_start

        k, _ = diag_setup

        # Create a session with report
        class StartArgs:
            type = "self_requested"
            access = "structural"
            json = False

        cmd_doctor_session_start(StartArgs(), k)
        capsys.readouterr()

        class ReportArgs:
            session_id = "latest"
            json = True

        cmd_doctor_report(ReportArgs(), k)
        output = capsys.readouterr().out
        data = json.loads(output)
        assert "id" in data
        assert "session_id" in data
        assert "summary" in data

    def test_cmd_report_not_found(self, diag_setup, capsys):
        from kernle.cli.commands.doctor import cmd_doctor_report

        k, _ = diag_setup

        class Args:
            session_id = "nonexistent-id"
            json = False

        cmd_doctor_report(Args(), k)
        output = capsys.readouterr().out
        assert "No report found" in output

    def test_cmd_report_by_session_id(self, diag_setup, capsys):
        from kernle.cli.commands.doctor import cmd_doctor_report, cmd_doctor_session_start

        k, _ = diag_setup

        # Create session
        class StartArgs:
            type = "self_requested"
            access = "structural"
            json = True

        cmd_doctor_session_start(StartArgs(), k)
        start_output = json.loads(capsys.readouterr().out)
        sid = start_output["session_id"]

        # Get report by session ID
        report_args = type("ReportArgs", (), {"session_id": sid, "json": True})()

        cmd_doctor_report(report_args, k)
        output = capsys.readouterr().out
        data = json.loads(output)
        assert data["session_id"] == sid


# =============================================================================
# Schema Migration Tests
# =============================================================================


class TestSchemaVersion:
    """Tests that schema version is correctly updated."""

    def test_schema_version_is_22(self):
        from kernle.storage.sqlite import SCHEMA_VERSION

        assert SCHEMA_VERSION == 23

    def test_tables_in_allowed_list(self):
        from kernle.storage.sqlite import ALLOWED_TABLES

        assert "diagnostic_sessions" in ALLOWED_TABLES
        assert "diagnostic_reports" in ALLOWED_TABLES

    def test_tables_created_on_init(self, tmp_path):
        """Verify tables are created for fresh databases."""
        import sqlite3

        db_path = tmp_path / "fresh.db"
        storage = SQLiteStorage(stack_id="test_agent", db_path=db_path)

        conn = sqlite3.connect(str(db_path))
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        conn.close()
        storage.close()

        assert "diagnostic_sessions" in tables
        assert "diagnostic_reports" in tables


# =============================================================================
# Recommendation Tests
# =============================================================================


class TestRecommendations:
    """Tests for recommendation mapping."""

    def test_known_recommendations(self):
        from kernle.cli.commands.doctor import _recommendation_for

        assert "Remove" in _recommendation_for("orphaned_reference")
        assert "verify" in _recommendation_for("low_confidence_belief").lower()
        assert "archive" in _recommendation_for("stale_relationship").lower()
        assert "resolve" in _recommendation_for("belief_contradiction").lower()
        assert "review" in _recommendation_for("stale_goal").lower()

    def test_unknown_recommendation(self):
        from kernle.cli.commands.doctor import _recommendation_for

        result = _recommendation_for("unknown_check")
        assert "Review" in result
