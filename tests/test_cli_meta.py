"""Tests for CLI meta command module."""

from argparse import Namespace
from unittest.mock import MagicMock

from kernle.cli.commands.meta import cmd_meta


class TestCmdMetaConfidence:
    """Test meta confidence command."""

    def test_confidence_found(self, capsys):
        """Confidence for existing memory."""
        k = MagicMock()
        k.get_memory_confidence.return_value = 0.75

        args = Namespace(
            meta_action="confidence",
            type="belief",
            id="abc123",
        )

        cmd_meta(args, k)

        k.get_memory_confidence.assert_called_with("belief", "abc123")
        captured = capsys.readouterr()
        assert "Confidence:" in captured.out
        assert "75%" in captured.out

    def test_confidence_not_found(self, capsys):
        """Confidence for non-existent memory."""
        k = MagicMock()
        k.get_memory_confidence.return_value = -1.0

        args = Namespace(
            meta_action="confidence",
            type="belief",
            id="nonexistent",
        )

        cmd_meta(args, k)

        captured = capsys.readouterr()
        assert "✗" in captured.out
        assert "not found" in captured.out


class TestCmdMetaVerify:
    """Test meta verify command."""

    def test_verify_success(self, capsys):
        """Successful verification."""
        k = MagicMock()
        k.verify_memory.return_value = True
        k.get_memory_confidence.return_value = 0.9

        args = Namespace(
            meta_action="verify",
            type="note",
            id="abc123",
            evidence="Test evidence",
        )

        cmd_meta(args, k)

        k.verify_memory.assert_called_with("note", "abc123", "Test evidence")
        captured = capsys.readouterr()
        assert "✓" in captured.out
        assert "verified" in captured.out
        assert "90%" in captured.out

    def test_verify_failure(self, capsys):
        """Verification failure."""
        k = MagicMock()
        k.verify_memory.return_value = False

        args = Namespace(
            meta_action="verify",
            type="note",
            id="abc123",
            evidence="Some evidence",
        )

        cmd_meta(args, k)

        captured = capsys.readouterr()
        assert "✗" in captured.out
        assert "Could not verify" in captured.out


class TestCmdMetaLineage:
    """Test meta lineage command."""

    def test_lineage_error(self, capsys):
        """Lineage with error."""
        k = MagicMock()
        k.get_memory_lineage.return_value = {"error": "Memory not found"}

        args = Namespace(
            meta_action="lineage",
            type="belief",
            id="nonexistent",
            json=False,
        )

        cmd_meta(args, k)

        captured = capsys.readouterr()
        assert "✗" in captured.out
        assert "Memory not found" in captured.out

    def test_lineage_success(self, capsys):
        """Successful lineage retrieval."""
        k = MagicMock()
        k.get_memory_lineage.return_value = {
            "source_type": "inferred",
            "current_confidence": 0.8,
            "source_episodes": ["ep123", "ep456"],
            "derived_from": ["belief:b789"],
            "confidence_history": [
                {
                    "timestamp": "2026-01-01T00:00:00Z",
                    "old": 0.7,
                    "new": 0.8,
                    "reason": "reinforced",
                },
            ],
        }

        args = Namespace(
            meta_action="lineage",
            type="belief",
            id="abc123",
            json=False,
        )

        cmd_meta(args, k)

        captured = capsys.readouterr()
        assert "Lineage for belief:abc123" in captured.out
        assert "inferred" in captured.out
        assert "Supporting Episodes" in captured.out
        assert "ep123" in captured.out
        assert "Derived From" in captured.out
        assert "belief:b789" in captured.out
        assert "Confidence History" in captured.out

    def test_lineage_json(self, capsys):
        """Lineage JSON output."""
        k = MagicMock()
        k.get_memory_lineage.return_value = {
            "source_type": "direct",
            "current_confidence": 0.9,
        }

        args = Namespace(
            meta_action="lineage",
            type="belief",
            id="abc123",
            json=True,
        )

        cmd_meta(args, k)

        captured = capsys.readouterr()
        assert '"source_type"' in captured.out
        assert '"direct"' in captured.out


class TestCmdMetaUncertain:
    """Test meta uncertain command."""

    def test_uncertain_no_results(self, capsys):
        """No uncertain memories."""
        k = MagicMock()
        k.get_uncertain_memories.return_value = []

        args = Namespace(
            meta_action="uncertain",
            threshold=0.5,
            limit=20,
            json=False,
        )

        cmd_meta(args, k)

        k.get_uncertain_memories.assert_called_with(0.5, limit=20)
        captured = capsys.readouterr()
        assert "No memories below" in captured.out
        assert "50%" in captured.out

    def test_uncertain_with_results(self, capsys):
        """Uncertain memories found."""
        k = MagicMock()
        k.get_uncertain_memories.return_value = [
            {
                "type": "belief",
                "id": "abc12345678901234567890",
                "summary": "Uncertain belief statement",
                "confidence": 0.3,
                "created_at": "2026-01-01",
            },
        ]

        args = Namespace(
            meta_action="uncertain",
            threshold=0.5,
            limit=20,
            json=False,
        )

        cmd_meta(args, k)

        captured = capsys.readouterr()
        assert "Uncertain Memories" in captured.out
        assert "30%" in captured.out
        assert "belief" in captured.out

    def test_uncertain_json(self, capsys):
        """Uncertain memories JSON output."""
        k = MagicMock()
        k.get_uncertain_memories.return_value = [
            {"type": "belief", "id": "abc123", "summary": "test", "confidence": 0.3}
        ]

        args = Namespace(
            meta_action="uncertain",
            threshold=0.5,
            limit=20,
            json=True,
        )

        cmd_meta(args, k)

        captured = capsys.readouterr()
        assert '"type"' in captured.out
        assert '"belief"' in captured.out


class TestCmdMetaPropagate:
    """Test meta propagate command."""

    def test_propagate_success(self, capsys):
        """Successful propagation."""
        k = MagicMock()
        k.propagate_confidence.return_value = {
            "source_confidence": 0.9,
            "updated": 3,
        }

        args = Namespace(
            meta_action="propagate",
            type="episode",
            id="abc123",
        )

        cmd_meta(args, k)

        k.propagate_confidence.assert_called_with("episode", "abc123")
        captured = capsys.readouterr()
        assert "✓" in captured.out
        assert "90%" in captured.out
        assert "3" in captured.out

    def test_propagate_error(self, capsys):
        """Propagation error."""
        k = MagicMock()
        k.propagate_confidence.return_value = {"error": "Memory not found"}

        args = Namespace(
            meta_action="propagate",
            type="episode",
            id="nonexistent",
        )

        cmd_meta(args, k)

        captured = capsys.readouterr()
        assert "✗" in captured.out


class TestCmdMetaSource:
    """Test meta source command."""

    def test_source_success(self, capsys):
        """Successful source setting."""
        k = MagicMock()
        k.set_memory_source.return_value = True

        args = Namespace(
            meta_action="source",
            type="belief",
            id="abc123",
            source="inferred",
            episodes=["ep1", "ep2"],
            derived=["belief:b1"],
        )

        cmd_meta(args, k)

        k.set_memory_source.assert_called_with(
            "belief",
            "abc123",
            "inferred",
            source_episodes=["ep1", "ep2"],
            derived_from=["belief:b1"],
        )
        captured = capsys.readouterr()
        assert "✓" in captured.out
        assert "Source set" in captured.out
        assert "inferred" in captured.out

    def test_source_failure(self, capsys):
        """Source setting failure."""
        k = MagicMock()
        k.set_memory_source.return_value = False

        args = Namespace(
            meta_action="source",
            type="belief",
            id="nonexistent",
            source="direct",
            episodes=None,
            derived=None,
        )

        cmd_meta(args, k)

        captured = capsys.readouterr()
        assert "✗" in captured.out
        assert "Could not set source" in captured.out


class TestCmdMetaKnowledge:
    """Test meta knowledge command."""

    def test_knowledge_empty(self, capsys):
        """Empty knowledge map."""
        k = MagicMock()
        k.get_knowledge_map.return_value = {
            "domains": [],
            "blind_spots": [],
            "uncertain_areas": [],
            "total_domains": 0,
        }

        args = Namespace(
            meta_action="knowledge",
            json=False,
        )

        cmd_meta(args, k)

        captured = capsys.readouterr()
        assert "Knowledge Map" in captured.out
        assert "No knowledge domains found" in captured.out

    def test_knowledge_with_domains(self, capsys):
        """Knowledge map with domains."""
        k = MagicMock()
        k.get_knowledge_map.return_value = {
            "domains": [
                {
                    "name": "programming",
                    "coverage": "high",
                    "avg_confidence": 0.85,
                    "belief_count": 10,
                    "episode_count": 20,
                    "note_count": 5,
                    "last_updated": "2026-01-28T00:00:00Z",
                }
            ],
            "blind_spots": ["quantum physics", "art history"],
            "uncertain_areas": ["machine learning"],
            "total_domains": 1,
        }

        args = Namespace(
            meta_action="knowledge",
            json=False,
        )

        cmd_meta(args, k)

        captured = capsys.readouterr()
        assert "Knowledge Map" in captured.out
        assert "programming" in captured.out
        assert "85%" in captured.out
        assert "Blind Spots" in captured.out
        assert "quantum physics" in captured.out
        assert "Uncertain Areas" in captured.out
        assert "machine learning" in captured.out

    def test_knowledge_json(self, capsys):
        """Knowledge map JSON output."""
        k = MagicMock()
        k.get_knowledge_map.return_value = {"domains": [], "total_domains": 0}

        args = Namespace(
            meta_action="knowledge",
            json=True,
        )

        cmd_meta(args, k)

        captured = capsys.readouterr()
        assert '"domains"' in captured.out


class TestCmdMetaGaps:
    """Test meta gaps command."""

    def test_gaps_analysis(self, capsys):
        """Gap analysis with results."""
        k = MagicMock()
        k.detect_knowledge_gaps.return_value = {
            "recommendation": "I can help",
            "confidence": 0.85,
            "search_results_count": 5,
            "relevant_beliefs": [{"statement": "Python is a great language", "confidence": 0.9}],
            "relevant_episodes": [
                {
                    "objective": "Built a web app",
                    "outcome_type": "success",
                    "lessons": ["Use virtual environments"],
                }
            ],
            "gaps": ["Advanced async patterns"],
        }

        args = Namespace(
            meta_action="gaps",
            query="Python programming",
            json=False,
        )

        cmd_meta(args, k)

        captured = capsys.readouterr()
        assert "Knowledge Gap Analysis" in captured.out
        assert "I can help" in captured.out
        assert "85%" in captured.out
        assert "Python is a great" in captured.out
        assert "Built a web app" in captured.out
        assert "Advanced async patterns" in captured.out

    def test_gaps_limited_knowledge(self, capsys):
        """Gap analysis with limited knowledge."""
        k = MagicMock()
        k.detect_knowledge_gaps.return_value = {
            "recommendation": "I have limited knowledge - proceed with caution",
            "confidence": 0.3,
            "search_results_count": 1,
            "relevant_beliefs": [],
            "relevant_episodes": [],
            "gaps": ["Everything about this topic"],
        }

        args = Namespace(
            meta_action="gaps",
            query="Quantum computing",
            json=False,
        )

        cmd_meta(args, k)

        captured = capsys.readouterr()
        assert "🟡" in captured.out  # Yellow for limited knowledge
        assert "30%" in captured.out

    def test_gaps_should_learn_more(self, capsys):
        """Gap analysis when should learn more."""
        k = MagicMock()
        k.detect_knowledge_gaps.return_value = {
            "recommendation": "I should learn more",
            "confidence": 0.2,
            "search_results_count": 0,
            "relevant_beliefs": [],
            "relevant_episodes": [],
            "gaps": [],
        }

        args = Namespace(
            meta_action="gaps",
            query="Advanced topic",
            json=False,
        )

        cmd_meta(args, k)

        captured = capsys.readouterr()
        assert "🟠" in captured.out  # Orange for should learn more

    def test_gaps_ask_someone_else(self, capsys):
        """Gap analysis when should ask someone else."""
        k = MagicMock()
        k.detect_knowledge_gaps.return_value = {
            "recommendation": "Ask someone else",
            "confidence": 0.1,
            "search_results_count": 0,
            "relevant_beliefs": [],
            "relevant_episodes": [],
            "gaps": [],
        }

        args = Namespace(
            meta_action="gaps",
            query="Unknown topic",
            json=False,
        )

        cmd_meta(args, k)

        captured = capsys.readouterr()
        assert "🔴" in captured.out  # Red for ask someone else

    def test_gaps_json(self, capsys):
        """Gap analysis JSON output."""
        k = MagicMock()
        k.detect_knowledge_gaps.return_value = {
            "recommendation": "I can help",
            "confidence": 0.9,
            "search_results_count": 10,
        }

        args = Namespace(
            meta_action="gaps",
            query="test",
            json=True,
        )

        cmd_meta(args, k)

        captured = capsys.readouterr()
        assert '"recommendation"' in captured.out


class TestCmdMetaBoundaries:
    """Test meta boundaries command."""

    def test_boundaries_with_data(self, capsys):
        """Boundaries with strengths and weaknesses."""
        k = MagicMock()
        k.get_competence_boundaries.return_value = {
            "overall_confidence": 0.75,
            "success_rate": 0.8,
            "experience_depth": 50,
            "knowledge_breadth": 10,
            "strengths": [
                {"domain": "python", "confidence": 0.9, "success_rate": 0.95},
                {"domain": "testing", "confidence": 0.85, "success_rate": 0.9},
            ],
            "weaknesses": [
                {"domain": "databases", "confidence": 0.3, "success_rate": 0.4},
            ],
        }

        args = Namespace(
            meta_action="boundaries",
            json=False,
        )

        cmd_meta(args, k)

        captured = capsys.readouterr()
        assert "Competence Boundaries" in captured.out
        assert "75%" in captured.out
        assert "80%" in captured.out
        assert "50 episodes" in captured.out
        assert "Strengths" in captured.out
        assert "python" in captured.out
        assert "Weaknesses" in captured.out
        assert "databases" in captured.out

    def test_boundaries_no_data(self, capsys):
        """Boundaries with no strengths/weaknesses."""
        k = MagicMock()
        k.get_competence_boundaries.return_value = {
            "overall_confidence": 0.5,
            "success_rate": 0.5,
            "experience_depth": 0,
            "knowledge_breadth": 0,
            "strengths": [],
            "weaknesses": [],
        }

        args = Namespace(
            meta_action="boundaries",
            json=False,
        )

        cmd_meta(args, k)

        captured = capsys.readouterr()
        assert "Not enough data" in captured.out

    def test_boundaries_json(self, capsys):
        """Boundaries JSON output."""
        k = MagicMock()
        k.get_competence_boundaries.return_value = {
            "overall_confidence": 0.5,
            "success_rate": 0.5,
        }

        args = Namespace(
            meta_action="boundaries",
            json=True,
        )

        cmd_meta(args, k)

        captured = capsys.readouterr()
        assert '"overall_confidence"' in captured.out


class TestCmdMetaLearn:
    """Test meta learn command."""

    def test_learn_no_opportunities(self, capsys):
        """No learning opportunities."""
        k = MagicMock()
        k.identify_learning_opportunities.return_value = []

        args = Namespace(
            meta_action="learn",
            limit=10,
            json=False,
        )

        cmd_meta(args, k)

        k.identify_learning_opportunities.assert_called_with(limit=10)
        captured = capsys.readouterr()
        assert "No urgent learning needs" in captured.out

    def test_learn_with_opportunities(self, capsys):
        """Learning opportunities found."""
        k = MagicMock()
        k.identify_learning_opportunities.return_value = [
            {
                "domain": "databases",
                "priority": "high",
                "type": "low_coverage_domain",
                "reason": "Only 2 episodes in this domain",
                "suggested_action": "Practice more SQL queries",
            },
            {
                "domain": "testing",
                "priority": "medium",
                "type": "uncertain_belief",
                "reason": "Low confidence in testing patterns",
                "suggested_action": "Review testing best practices",
            },
        ]

        args = Namespace(
            meta_action="learn",
            limit=10,
            json=False,
        )

        cmd_meta(args, k)

        captured = capsys.readouterr()
        assert "Learning Opportunities" in captured.out
        assert "databases" in captured.out
        assert "HIGH" in captured.out
        assert "Practice more SQL" in captured.out
        assert "testing" in captured.out
        assert "MEDIUM" in captured.out

    def test_learn_json(self, capsys):
        """Learning opportunities JSON output."""
        k = MagicMock()
        k.identify_learning_opportunities.return_value = [{"domain": "test", "priority": "low"}]

        args = Namespace(
            meta_action="learn",
            limit=10,
            json=True,
        )

        cmd_meta(args, k)

        captured = capsys.readouterr()
        assert '"domain"' in captured.out
        assert '"priority"' in captured.out
