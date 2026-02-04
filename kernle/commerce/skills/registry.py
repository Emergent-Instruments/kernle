"""
Skills registry management.

The registry maintains canonical skills and tracks skill usage.
"""

import logging
from typing import List, Optional, Protocol

from kernle.commerce.skills.models import CANONICAL_SKILLS, Skill, SkillCategory

logger = logging.getLogger(__name__)


class SkillRegistry(Protocol):
    """Protocol for skill registry backends."""

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        ...

    def list_skills(
        self,
        category: Optional[SkillCategory] = None,
        limit: int = 100,
    ) -> List[Skill]:
        """List skills, optionally filtered by category."""
        ...

    def search_skills(self, query: str, limit: int = 10) -> List[Skill]:
        """Search skills by name or description."""
        ...

    def increment_usage(self, name: str) -> bool:
        """Increment usage count for a skill."""
        ...

    def add_custom_skill(
        self,
        name: str,
        description: str,
        category: Optional[SkillCategory] = None,
    ) -> Skill:
        """Add a custom (non-canonical) skill."""
        ...


class SupabaseSkillRegistry:
    """Supabase-backed skill registry.

    Note: This is a placeholder implementation. The actual Supabase
    integration will be added when the backend routes are implemented.
    """

    def __init__(self, supabase_url: str, supabase_key: str):
        """Initialize Supabase connection."""
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self._client = None

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        logger.debug(f"Getting skill: {name}")
        # TODO: Implement Supabase query
        # result = self._client.table("skills").select("*").eq("name", name).single().execute()
        # if result.data:
        #     return Skill.from_dict(result.data)
        return None

    def list_skills(
        self,
        category: Optional[SkillCategory] = None,
        limit: int = 100,
    ) -> List[Skill]:
        """List skills."""
        logger.debug(f"Listing skills, category={category}")
        # TODO: Implement Supabase query
        # query = self._client.table("skills").select("*")
        # if category:
        #     query = query.eq("category", category.value)
        # query = query.order("usage_count", desc=True).limit(limit)
        return []

    def search_skills(self, query: str, limit: int = 10) -> List[Skill]:
        """Search skills by name or description."""
        logger.debug(f"Searching skills: {query}")
        # TODO: Implement Supabase query with text search
        return []

    def increment_usage(self, name: str) -> bool:
        """Increment usage count for a skill."""
        logger.info(f"Incrementing usage for skill: {name}")
        # TODO: Implement Supabase update with atomic increment
        return True

    def add_custom_skill(
        self,
        name: str,
        description: str,
        category: Optional[SkillCategory] = None,
    ) -> Skill:
        """Add a custom skill."""
        logger.info(f"Adding custom skill: {name}")
        # TODO: Implement Supabase insert
        import uuid

        skill = Skill(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            category=category.value if category else None,
        )
        return skill


class InMemorySkillRegistry:
    """In-memory skill registry for testing and local development."""

    def __init__(self):
        """Initialize with canonical skills."""
        import uuid

        self._skills: dict[str, Skill] = {}

        # Pre-populate with canonical skills
        for name in CANONICAL_SKILLS:
            self._skills[name] = Skill.from_canonical(name, str(uuid.uuid4()))

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        return self._skills.get(name)

    def list_skills(
        self,
        category: Optional[SkillCategory] = None,
        limit: int = 100,
    ) -> List[Skill]:
        """List skills."""
        skills = list(self._skills.values())
        if category:
            cat_value = category.value if isinstance(category, SkillCategory) else category
            skills = [s for s in skills if s.category == cat_value]
        return sorted(skills, key=lambda s: -s.usage_count)[:limit]

    def search_skills(self, query: str, limit: int = 10) -> List[Skill]:
        """Search skills by name or description."""
        query_lower = query.lower()
        matches = []
        for skill in self._skills.values():
            if query_lower in skill.name.lower():
                matches.append(skill)
            elif skill.description and query_lower in skill.description.lower():
                matches.append(skill)
        return matches[:limit]

    def increment_usage(self, name: str) -> bool:
        """Increment usage count for a skill."""
        if name in self._skills:
            self._skills[name].usage_count += 1
            return True
        return False

    def add_custom_skill(
        self,
        name: str,
        description: str,
        category: Optional[SkillCategory] = None,
    ) -> Skill:
        """Add a custom skill."""
        import uuid

        skill = Skill(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            category=category.value if category else None,
        )
        self._skills[name] = skill
        return skill
