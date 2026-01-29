"""
Tests for Kernle Playbooks (Procedural Memory).

Playbooks are "how I do things" memory - executable procedures
learned from experience.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest

from kernle.core import Kernle
from kernle.storage import Playbook, SQLiteStorage


class TestPlaybookDataModel:
    """Test the Playbook dataclass."""

    def test_playbook_creation_minimal(self):
        """Test creating a playbook with minimal required fields."""
        playbook = Playbook(
            id=str(uuid.uuid4()),
            agent_id="test_agent",
            name="Deploy to production",
            description="Standard deployment process",
            trigger_conditions=["releasing code", "pushing to main"],
            steps=[{"action": "Run tests", "details": None, "adaptations": None}],
            failure_modes=["Tests fail"],
        )

        assert playbook.name == "Deploy to production"
        assert playbook.mastery_level == "novice"
        assert playbook.times_used == 0
        assert playbook.success_rate == 0.0
        assert playbook.confidence == 0.8

    def test_playbook_creation_full(self):
        """Test creating a playbook with all fields."""
        now = datetime.now(timezone.utc)
        playbook = Playbook(
            id="pb-123",
            agent_id="test_agent",
            name="Debug Memory Leaks",
            description="Process for finding and fixing memory leaks",
            trigger_conditions=["high memory usage", "out of memory errors"],
            steps=[
                {"action": "Run profiler", "details": "Use memory_profiler", "adaptations": None},
                {
                    "action": "Identify leaks",
                    "details": None,
                    "adaptations": {"large datasets": "use sampling"},
                },
                {"action": "Fix and verify", "details": None, "adaptations": None},
            ],
            failure_modes=["Profiler crashes", "Leak not reproducible"],
            recovery_steps=["Reduce dataset size", "Try different profiler"],
            mastery_level="proficient",
            times_used=15,
            success_rate=0.87,
            source_episodes=["ep-1", "ep-2", "ep-3"],
            tags=["debugging", "performance"],
            confidence=0.92,
            last_used=now,
            created_at=now,
        )

        assert playbook.mastery_level == "proficient"
        assert playbook.times_used == 15
        assert playbook.success_rate == 0.87
        assert len(playbook.steps) == 3
        assert playbook.tags == ["debugging", "performance"]


class TestPlaybookStorage:
    """Test playbook storage operations."""

    @pytest.fixture
    def storage(self, tmp_path):
        """Create a temporary SQLite storage."""
        db_path = tmp_path / "test_playbooks.db"
        return SQLiteStorage(agent_id="test_agent", db_path=db_path)

    def test_save_and_get_playbook(self, storage):
        """Test saving and retrieving a playbook."""
        playbook = Playbook(
            id=str(uuid.uuid4()),
            agent_id="test_agent",
            name="Code Review",
            description="Standard code review process",
            trigger_conditions=["PR opened", "code changes"],
            steps=[
                {"action": "Read the PR description", "details": None, "adaptations": None},
                {"action": "Review changes", "details": None, "adaptations": None},
                {"action": "Leave comments", "details": None, "adaptations": None},
            ],
            failure_modes=["PR too large", "Missing context"],
            recovery_steps=["Request PR split", "Ask for context"],
            tags=["code-review", "workflow"],
            created_at=datetime.now(timezone.utc),
        )

        saved_id = storage.save_playbook(playbook)
        assert saved_id == playbook.id

        retrieved = storage.get_playbook(playbook.id)
        assert retrieved is not None
        assert retrieved.name == "Code Review"
        assert retrieved.description == "Standard code review process"
        assert len(retrieved.steps) == 3
        assert retrieved.tags == ["code-review", "workflow"]

    def test_list_playbooks(self, storage):
        """Test listing playbooks."""
        # Create multiple playbooks
        for i in range(5):
            playbook = Playbook(
                id=str(uuid.uuid4()),
                agent_id="test_agent",
                name=f"Playbook {i}",
                description=f"Description for playbook {i}",
                trigger_conditions=[f"trigger {i}"],
                steps=[{"action": f"Step for {i}", "details": None, "adaptations": None}],
                failure_modes=[f"Failure {i}"],
                times_used=i,  # Vary usage count
                tags=["test"] if i % 2 == 0 else ["other"],
                created_at=datetime.now(timezone.utc),
            )
            storage.save_playbook(playbook)

        # List all
        all_playbooks = storage.list_playbooks(limit=10)
        assert len(all_playbooks) == 5

        # Should be ordered by times_used descending
        times_used = [p.times_used for p in all_playbooks]
        assert times_used == sorted(times_used, reverse=True)

        # List with tag filter
        test_tagged = storage.list_playbooks(tags=["test"], limit=10)
        assert len(test_tagged) == 3  # 0, 2, 4 are tagged "test"

    def test_search_playbooks(self, storage):
        """Test searching playbooks."""
        # Create playbooks with searchable content
        deploy_playbook = Playbook(
            id=str(uuid.uuid4()),
            agent_id="test_agent",
            name="Deploy Application",
            description="Deploy the application to production servers",
            trigger_conditions=["release ready", "deploy command"],
            steps=[{"action": "Build", "details": None, "adaptations": None}],
            failure_modes=["Build fails"],
            created_at=datetime.now(timezone.utc),
        )
        storage.save_playbook(deploy_playbook)

        test_playbook = Playbook(
            id=str(uuid.uuid4()),
            agent_id="test_agent",
            name="Run Tests",
            description="Execute the test suite",
            trigger_conditions=["code changed", "pre-deploy"],
            steps=[{"action": "pytest", "details": None, "adaptations": None}],
            failure_modes=["Tests fail"],
            created_at=datetime.now(timezone.utc),
        )
        storage.save_playbook(test_playbook)

        # Search for deploy
        deploy_results = storage.search_playbooks("deploy", limit=10)
        assert len(deploy_results) >= 1
        assert any(p.name == "Deploy Application" for p in deploy_results)

        # Search for test
        test_results = storage.search_playbooks("test", limit=10)
        assert len(test_results) >= 1
        assert any(p.name == "Run Tests" for p in test_results)

    def test_update_playbook_usage(self, storage):
        """Test updating playbook usage statistics."""
        playbook = Playbook(
            id=str(uuid.uuid4()),
            agent_id="test_agent",
            name="Test Playbook",
            description="For testing usage updates",
            trigger_conditions=["test"],
            steps=[{"action": "Test", "details": None, "adaptations": None}],
            failure_modes=["Fail"],
            times_used=0,
            success_rate=0.0,
            mastery_level="novice",
            created_at=datetime.now(timezone.utc),
        )
        storage.save_playbook(playbook)

        # Record some successes
        for _ in range(5):
            storage.update_playbook_usage(playbook.id, success=True)

        updated = storage.get_playbook(playbook.id)
        assert updated.times_used == 5
        assert updated.success_rate == 1.0
        assert updated.mastery_level == "competent"  # 5 uses with 100% success
        assert updated.last_used is not None

        # Record some failures
        for _ in range(5):
            storage.update_playbook_usage(playbook.id, success=False)

        updated = storage.get_playbook(playbook.id)
        assert updated.times_used == 10
        assert updated.success_rate == 0.5  # 5 success, 5 failure
        assert (
            updated.mastery_level == "competent"
        )  # Still competent (10 uses but only 50% success)

    def test_mastery_progression(self, storage):
        """Test that mastery level increases with usage and success."""
        playbook = Playbook(
            id=str(uuid.uuid4()),
            agent_id="test_agent",
            name="Mastery Test",
            description="Testing mastery progression",
            trigger_conditions=["test"],
            steps=[{"action": "Test", "details": None, "adaptations": None}],
            failure_modes=["Fail"],
            times_used=0,
            success_rate=0.0,
            mastery_level="novice",
            created_at=datetime.now(timezone.utc),
        )
        storage.save_playbook(playbook)

        # Novice -> Competent (5 uses, 70% success)
        for i in range(5):
            storage.update_playbook_usage(playbook.id, success=(i < 4))  # 4/5 = 80%

        updated = storage.get_playbook(playbook.id)
        assert updated.mastery_level == "competent"
        assert updated.times_used == 5

        # Competent -> Proficient (10 uses, 80% success)
        for i in range(5):
            storage.update_playbook_usage(playbook.id, success=(i < 4))  # More successes

        updated = storage.get_playbook(playbook.id)
        assert updated.mastery_level == "proficient"
        assert updated.times_used == 10

        # Proficient -> Expert (20 uses, 90%+ success overall)
        # Need to add enough successes to get cumulative rate above 90%
        # Current: 8/10 = 80%. Need 18/20 = 90% for expert.
        # So need 10 more successes out of 10.
        for _ in range(10):
            storage.update_playbook_usage(playbook.id, success=True)  # All successes

        updated = storage.get_playbook(playbook.id)
        # 8 + 10 = 18 successes out of 20 = 90%
        assert updated.mastery_level == "expert"
        assert updated.times_used == 20


class TestPlaybookCore:
    """Test playbook methods in Kernle core."""

    @pytest.fixture
    def kernle(self, tmp_path):
        """Create a Kernle instance with temporary storage."""
        db_path = tmp_path / "test_kernle.db"
        storage = SQLiteStorage(agent_id="test_agent", db_path=db_path)
        return Kernle(agent_id="test_agent", storage=storage)

    def test_create_playbook_simple(self, kernle):
        """Test creating a playbook with simple string steps."""
        playbook_id = kernle.playbook(
            name="Simple Deploy",
            description="A simple deployment process",
            steps=["Run tests", "Build", "Deploy", "Verify"],
            triggers=["releasing code"],
        )

        assert playbook_id is not None

        # Retrieve and verify
        playbook = kernle.get_playbook(playbook_id)
        assert playbook is not None
        assert playbook["name"] == "Simple Deploy"
        assert len(playbook["steps"]) == 4
        assert playbook["steps"][0]["action"] == "Run tests"

    def test_create_playbook_with_dict_steps(self, kernle):
        """Test creating a playbook with detailed dict steps."""
        playbook_id = kernle.playbook(
            name="Advanced Deploy",
            description="Deployment with adaptations",
            steps=[
                {"action": "Run tests", "details": "Use pytest -v", "adaptations": None},
                {
                    "action": "Build",
                    "details": "docker build",
                    "adaptations": {"arm64": "use buildx"},
                },
                {"action": "Deploy", "details": None, "adaptations": None},
            ],
            triggers=["releasing code", "hotfix needed"],
            failure_modes=["Tests fail", "Build fails", "Deploy timeout"],
            recovery_steps=["Fix tests", "Check Docker", "Rollback"],
            tags=["deploy", "production"],
        )

        playbook = kernle.get_playbook(playbook_id)
        assert playbook is not None
        assert playbook["steps"][1]["adaptations"] == {"arm64": "use buildx"}
        assert playbook["failure_modes"] == ["Tests fail", "Build fails", "Deploy timeout"]
        assert playbook["recovery_steps"] == ["Fix tests", "Check Docker", "Rollback"]

    def test_load_playbooks(self, kernle):
        """Test loading playbooks."""
        # Create some playbooks
        kernle.playbook(
            name="Playbook 1",
            description="First playbook",
            steps=["Step 1"],
            tags=["tag1"],
        )
        kernle.playbook(
            name="Playbook 2",
            description="Second playbook",
            steps=["Step 2"],
            tags=["tag2"],
        )

        # Load all
        playbooks = kernle.load_playbooks(limit=10)
        assert len(playbooks) == 2

        # Load by tag
        tag1_playbooks = kernle.load_playbooks(limit=10, tags=["tag1"])
        assert len(tag1_playbooks) == 1
        assert tag1_playbooks[0]["name"] == "Playbook 1"

    def test_find_playbook(self, kernle):
        """Test finding a relevant playbook for a situation."""
        # Create playbooks with different purposes
        kernle.playbook(
            name="Deploy to Production",
            description="Standard production deployment",
            steps=["Test", "Build", "Deploy"],
            triggers=["release ready", "deploy to prod"],
        )
        kernle.playbook(
            name="Debug Performance",
            description="Diagnose performance issues",
            steps=["Profile", "Analyze", "Optimize"],
            triggers=["slow response", "performance degradation"],
        )

        # Find playbook for deployment situation
        deploy_match = kernle.find_playbook("I need to deploy the new release to production")
        assert deploy_match is not None
        assert "Deploy" in deploy_match["name"]

        # Find playbook for performance situation
        perf_match = kernle.find_playbook("The API is responding slowly")
        assert perf_match is not None
        assert "Performance" in perf_match["name"]

    def test_record_playbook_use(self, kernle):
        """Test recording playbook usage."""
        playbook_id = kernle.playbook(
            name="Test Recording",
            description="For testing usage recording",
            steps=["Step 1"],
        )

        # Initial state
        playbook = kernle.get_playbook(playbook_id)
        assert playbook["times_used"] == 0

        # Record success
        result = kernle.record_playbook_use(playbook_id, success=True)
        assert result is True

        playbook = kernle.get_playbook(playbook_id)
        assert playbook["times_used"] == 1
        assert playbook["success_rate"] == 1.0

        # Record failure
        kernle.record_playbook_use(playbook_id, success=False)

        playbook = kernle.get_playbook(playbook_id)
        assert playbook["times_used"] == 2
        assert playbook["success_rate"] == 0.5

    def test_record_nonexistent_playbook(self, kernle):
        """Test recording usage for a non-existent playbook."""
        result = kernle.record_playbook_use("nonexistent-id", success=True)
        assert result is False

    def test_search_playbooks(self, kernle):
        """Test searching playbooks by query."""
        kernle.playbook(
            name="Database Migration",
            description="Migrate database schema",
            steps=["Backup", "Migrate", "Verify"],
            triggers=["schema change"],
        )
        kernle.playbook(
            name="Cache Invalidation",
            description="Clear and rebuild cache",
            steps=["Clear", "Rebuild"],
            triggers=["stale data"],
        )

        # Search
        results = kernle.search_playbooks("database")
        assert len(results) >= 1
        assert any("Database" in r["name"] for r in results)


class TestPlaybookCLI:
    """Test playbook CLI commands."""

    @pytest.fixture
    def mock_kernle(self):
        """Mock Kernle instance for CLI testing."""
        kernle = Mock(spec=Kernle)

        kernle.playbook.return_value = "pb-123"
        kernle.load_playbooks.return_value = [
            {
                "id": "pb-1",
                "name": "Deploy",
                "description": "Deploy process",
                "mastery_level": "competent",
                "times_used": 10,
                "success_rate": 0.9,
                "tags": ["deploy"],
            }
        ]
        kernle.search_playbooks.return_value = [
            {
                "id": "pb-1",
                "name": "Deploy",
                "description": "Deploy process",
                "mastery_level": "competent",
                "times_used": 10,
                "success_rate": 0.9,
            }
        ]
        kernle.get_playbook.return_value = {
            "id": "pb-1",
            "name": "Deploy",
            "description": "Standard deployment",
            "triggers": ["release ready"],
            "steps": [
                {"action": "Test", "details": None, "adaptations": None},
                {"action": "Build", "details": None, "adaptations": None},
            ],
            "failure_modes": ["Tests fail"],
            "recovery_steps": ["Fix tests"],
            "mastery_level": "competent",
            "times_used": 10,
            "success_rate": 0.9,
            "confidence": 0.85,
            "tags": ["deploy"],
            "last_used": "2024-01-15T10:00:00",
            "created_at": "2024-01-01T10:00:00",
        }
        kernle.find_playbook.return_value = {
            "id": "pb-1",
            "name": "Deploy",
            "description": "Standard deployment",
            "steps": [{"action": "Test", "details": None, "adaptations": None}],
            "mastery_level": "competent",
            "success_rate": 0.9,
        }
        kernle.record_playbook_use.return_value = True

        return kernle

    def test_cmd_playbook_create(self, mock_kernle):
        """Test playbook create command."""
        import argparse

        from kernle.cli.__main__ import cmd_playbook

        args = argparse.Namespace(
            playbook_action="create",
            name="Deploy to prod",
            description="Production deployment",
            steps="Test, Build, Deploy",
            step=None,
            triggers="releasing code",
            trigger=None,
            failure_mode=["Tests fail"],
            recovery=["Fix tests"],
            tag=["deploy"],
        )

        from io import StringIO

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_playbook(args, mock_kernle)

        mock_kernle.playbook.assert_called_once()
        call_kwargs = mock_kernle.playbook.call_args[1]
        assert call_kwargs["name"] == "Deploy to prod"
        assert "Deploy" in call_kwargs["steps"]
        assert "✓ Playbook created" in fake_out.getvalue()

    def test_cmd_playbook_list(self, mock_kernle):
        """Test playbook list command."""
        import argparse

        from kernle.cli.__main__ import cmd_playbook

        args = argparse.Namespace(
            playbook_action="list",
            tag=None,
            limit=20,
            json=False,
        )

        from io import StringIO

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_playbook(args, mock_kernle)

        mock_kernle.load_playbooks.assert_called_once()
        output = fake_out.getvalue()
        assert "Deploy" in output
        assert "competent" in output

    def test_cmd_playbook_search(self, mock_kernle):
        """Test playbook search command."""
        import argparse

        from kernle.cli.__main__ import cmd_playbook

        args = argparse.Namespace(
            playbook_action="search",
            query="deploy",
            limit=10,
            json=False,
        )

        from io import StringIO

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_playbook(args, mock_kernle)

        mock_kernle.search_playbooks.assert_called_once_with("deploy", limit=10)
        assert "Deploy" in fake_out.getvalue()

    def test_cmd_playbook_show(self, mock_kernle):
        """Test playbook show command."""
        import argparse

        from kernle.cli.__main__ import cmd_playbook

        args = argparse.Namespace(
            playbook_action="show",
            id="pb-1",
            json=False,
        )

        from io import StringIO

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_playbook(args, mock_kernle)

        mock_kernle.get_playbook.assert_called_once_with("pb-1")
        output = fake_out.getvalue()
        assert "Deploy" in output
        assert "Steps" in output
        assert "Failure Modes" in output

    def test_cmd_playbook_find(self, mock_kernle):
        """Test playbook find command."""
        import argparse

        from kernle.cli.__main__ import cmd_playbook

        args = argparse.Namespace(
            playbook_action="find",
            situation="I need to deploy the app",
            json=False,
        )

        from io import StringIO

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_playbook(args, mock_kernle)

        mock_kernle.find_playbook.assert_called_once_with("I need to deploy the app")
        output = fake_out.getvalue()
        assert "Recommended Playbook" in output
        assert "Deploy" in output

    def test_cmd_playbook_record_success(self, mock_kernle):
        """Test playbook record command with success."""
        import argparse

        from kernle.cli.__main__ import cmd_playbook

        args = argparse.Namespace(
            playbook_action="record",
            id="pb-1",
            success=True,
            failure=False,
        )

        from io import StringIO

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_playbook(args, mock_kernle)

        mock_kernle.record_playbook_use.assert_called_once_with("pb-1", True)
        assert "success ✓" in fake_out.getvalue()

    def test_cmd_playbook_record_failure(self, mock_kernle):
        """Test playbook record command with failure."""
        import argparse

        from kernle.cli.__main__ import cmd_playbook

        args = argparse.Namespace(
            playbook_action="record",
            id="pb-1",
            success=True,
            failure=True,  # --failure flag overrides default success
        )

        from io import StringIO

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_playbook(args, mock_kernle)

        mock_kernle.record_playbook_use.assert_called_once_with("pb-1", False)
        assert "failure ✗" in fake_out.getvalue()
