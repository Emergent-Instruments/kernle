"""Tests for skill data models."""

from datetime import datetime, timezone

import pytest

from kernle.commerce.skills.models import CANONICAL_SKILLS, Skill, SkillCategory
from kernle.commerce.skills.registry import InMemorySkillRegistry


class TestSkill:
    """Tests for Skill dataclass."""

    def test_create_basic_skill(self):
        """Test creating a skill with minimal required fields."""
        skill = Skill(
            id="skill-123",
            name="coding",
        )

        assert skill.id == "skill-123"
        assert skill.name == "coding"
        assert skill.description is None
        assert skill.category is None
        assert skill.usage_count == 0

    def test_create_skill_with_all_fields(self):
        """Test creating a skill with all fields."""
        now = datetime.now(timezone.utc)

        skill = Skill(
            id="skill-123",
            name="data-analysis",
            description="Data processing and insights",
            category="technical",
            usage_count=42,
            created_at=now,
        )

        assert skill.name == "data-analysis"
        assert skill.description == "Data processing and insights"
        assert skill.category == "technical"
        assert skill.usage_count == 42
        assert skill.created_at == now

    def test_invalid_skill_name(self):
        """Test that invalid skill names are rejected."""
        # Uppercase not allowed
        with pytest.raises(ValueError, match="Invalid skill name"):
            Skill(id="skill-123", name="Coding")

        # Spaces not allowed
        with pytest.raises(ValueError, match="Invalid skill name"):
            Skill(id="skill-124", name="data analysis")

        # Underscores not allowed (use hyphens)
        with pytest.raises(ValueError, match="Invalid skill name"):
            Skill(id="skill-125", name="data_analysis")

    def test_valid_skill_names(self):
        """Test that valid skill names are accepted."""
        # Lowercase with hyphens
        skill1 = Skill(id="skill-1", name="web-scraping")
        assert skill1.name == "web-scraping"

        # Numbers allowed
        skill2 = Skill(id="skill-2", name="python3")
        assert skill2.name == "python3"

        # Simple lowercase
        skill3 = Skill(id="skill-3", name="research")
        assert skill3.name == "research"

    def test_invalid_category(self):
        """Test that invalid categories are rejected."""
        with pytest.raises(ValueError, match="Invalid category"):
            Skill(
                id="skill-123",
                name="coding",
                category="invalid-category",
            )

    def test_category_enum_value(self):
        """Test that category can be set via enum."""
        skill = Skill(
            id="skill-123",
            name="coding",
            category=SkillCategory.TECHNICAL,
        )

        assert skill.category == "technical"

    def test_is_canonical(self):
        """Test is_canonical property."""
        # Canonical skill
        skill1 = Skill(id="skill-1", name="coding")
        assert skill1.is_canonical is True

        skill2 = Skill(id="skill-2", name="research")
        assert skill2.is_canonical is True

        # Non-canonical skill
        skill3 = Skill(id="skill-3", name="custom-skill")
        assert skill3.is_canonical is False

    def test_to_dict(self):
        """Test serialization to dictionary."""
        now = datetime.now(timezone.utc)

        skill = Skill(
            id="skill-123",
            name="coding",
            description="Software development",
            category="technical",
            usage_count=10,
            created_at=now,
        )

        d = skill.to_dict()

        assert d["id"] == "skill-123"
        assert d["name"] == "coding"
        assert d["description"] == "Software development"
        assert d["category"] == "technical"
        assert d["usage_count"] == 10
        assert d["created_at"] == now.isoformat()

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "id": "skill-123",
            "name": "writing",
            "description": "Content creation",
            "category": "creative",
            "usage_count": 25,
            "created_at": "2024-01-15T12:00:00+00:00",
        }

        skill = Skill.from_dict(data)

        assert skill.id == "skill-123"
        assert skill.name == "writing"
        assert skill.description == "Content creation"
        assert skill.category == "creative"
        assert skill.usage_count == 25
        assert skill.created_at is not None

    def test_from_canonical(self):
        """Test creating skill from canonical registry."""
        skill = Skill.from_canonical("research", "skill-uuid-123")

        assert skill.id == "skill-uuid-123"
        assert skill.name == "research"
        assert skill.description == "Information gathering and analysis"
        assert skill.category == "knowledge"

    def test_from_canonical_unknown(self):
        """Test that unknown canonical skill raises error."""
        with pytest.raises(ValueError, match="Unknown canonical skill"):
            Skill.from_canonical("unknown-skill", "skill-uuid-123")


class TestSkillCategory:
    """Tests for SkillCategory enum."""

    def test_all_categories_defined(self):
        """Test that all expected categories are defined."""
        categories = {c.value for c in SkillCategory}
        expected = {"technical", "creative", "knowledge", "language", "service"}
        assert categories == expected


class TestCanonicalSkills:
    """Tests for CANONICAL_SKILLS constant."""

    def test_canonical_skills_defined(self):
        """Test that canonical skills are defined."""
        expected_skills = {
            "research", "writing", "coding", "data-analysis", "automation",
            "design", "translation", "summarization", "customer-support",
            "market-scanning", "web-scraping"
        }

        assert set(CANONICAL_SKILLS.keys()) == expected_skills

    def test_canonical_skills_have_description(self):
        """Test that all canonical skills have descriptions."""
        for name, info in CANONICAL_SKILLS.items():
            assert "description" in info, f"Skill {name} missing description"
            assert len(info["description"]) > 0, f"Skill {name} has empty description"

    def test_canonical_skills_have_category(self):
        """Test that all canonical skills have categories."""
        for name, info in CANONICAL_SKILLS.items():
            assert "category" in info, f"Skill {name} missing category"
            assert isinstance(info["category"], SkillCategory), f"Skill {name} category not enum"


class TestInMemorySkillRegistry:
    """Tests for InMemorySkillRegistry."""

    def test_init_with_canonical_skills(self):
        """Test that registry is initialized with canonical skills."""
        registry = InMemorySkillRegistry()

        skills = registry.list_skills()
        skill_names = {s.name for s in skills}

        assert "research" in skill_names
        assert "coding" in skill_names
        assert "writing" in skill_names

    def test_get_skill(self):
        """Test getting a skill by name."""
        registry = InMemorySkillRegistry()

        skill = registry.get_skill("coding")

        assert skill is not None
        assert skill.name == "coding"
        assert skill.category == "technical"

    def test_get_unknown_skill(self):
        """Test getting an unknown skill returns None."""
        registry = InMemorySkillRegistry()

        skill = registry.get_skill("nonexistent")

        assert skill is None

    def test_list_skills_by_category(self):
        """Test listing skills filtered by category."""
        registry = InMemorySkillRegistry()

        technical_skills = registry.list_skills(category=SkillCategory.TECHNICAL)

        assert len(technical_skills) > 0
        for skill in technical_skills:
            assert skill.category == "technical"

    def test_search_skills(self):
        """Test searching skills by query."""
        registry = InMemorySkillRegistry()

        results = registry.search_skills("analysis")

        assert len(results) > 0
        # Should find data-analysis
        names = [s.name for s in results]
        assert "data-analysis" in names

    def test_increment_usage(self):
        """Test incrementing skill usage count."""
        registry = InMemorySkillRegistry()

        skill_before = registry.get_skill("coding")
        count_before = skill_before.usage_count

        result = registry.increment_usage("coding")

        assert result is True

        skill_after = registry.get_skill("coding")
        assert skill_after.usage_count == count_before + 1

    def test_increment_usage_unknown_skill(self):
        """Test incrementing usage for unknown skill returns False."""
        registry = InMemorySkillRegistry()

        result = registry.increment_usage("nonexistent")

        assert result is False

    def test_add_custom_skill(self):
        """Test adding a custom skill."""
        registry = InMemorySkillRegistry()

        skill = registry.add_custom_skill(
            name="custom-skill",
            description="A custom skill",
            category=SkillCategory.TECHNICAL,
        )

        assert skill.name == "custom-skill"
        assert skill.description == "A custom skill"
        assert skill.category == "technical"

        # Should be retrievable
        retrieved = registry.get_skill("custom-skill")
        assert retrieved is not None
        assert retrieved.name == "custom-skill"

    def test_list_skills_sorted_by_usage(self):
        """Test that skills are sorted by usage count."""
        registry = InMemorySkillRegistry()

        # Increment usage for some skills
        for _ in range(5):
            registry.increment_usage("coding")
        for _ in range(3):
            registry.increment_usage("research")

        skills = registry.list_skills()

        # First skill should have highest usage
        assert skills[0].name == "coding"
        assert skills[0].usage_count >= skills[1].usage_count
