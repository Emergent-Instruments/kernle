"""
Skill data models.

Skills are canonical tags that describe agent capabilities.
They enable job matching and agent discovery.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class SkillCategory(str, Enum):
    """Skill categories for organization."""
    
    TECHNICAL = "technical"  # coding, data-analysis, automation, web-scraping
    CREATIVE = "creative"  # writing, design
    KNOWLEDGE = "knowledge"  # research, summarization, market-scanning
    LANGUAGE = "language"  # translation
    SERVICE = "service"  # customer-support


# Canonical skill definitions
CANONICAL_SKILLS = {
    "research": {
        "description": "Information gathering and analysis",
        "category": SkillCategory.KNOWLEDGE,
    },
    "writing": {
        "description": "Content creation and copywriting",
        "category": SkillCategory.CREATIVE,
    },
    "coding": {
        "description": "Software development",
        "category": SkillCategory.TECHNICAL,
    },
    "data-analysis": {
        "description": "Data processing and insights",
        "category": SkillCategory.TECHNICAL,
    },
    "automation": {
        "description": "Workflow automation and scripting",
        "category": SkillCategory.TECHNICAL,
    },
    "design": {
        "description": "Visual design and graphics",
        "category": SkillCategory.CREATIVE,
    },
    "translation": {
        "description": "Language translation",
        "category": SkillCategory.LANGUAGE,
    },
    "summarization": {
        "description": "Content summarization",
        "category": SkillCategory.KNOWLEDGE,
    },
    "customer-support": {
        "description": "Customer service and support",
        "category": SkillCategory.SERVICE,
    },
    "market-scanning": {
        "description": "Market research and monitoring",
        "category": SkillCategory.KNOWLEDGE,
    },
    "web-scraping": {
        "description": "Web data extraction",
        "category": SkillCategory.TECHNICAL,
    },
}


@dataclass
class Skill:
    """A skill in the canonical registry.
    
    Skills are predefined capabilities that agents can claim and jobs
    can require. The registry maintains a canonical set of skills to
    ensure consistent matching.
    
    Attributes:
        id: Unique skill ID (UUID)
        name: Skill name (lowercase, hyphenated, e.g., "data-analysis")
        description: Human-readable description
        category: Skill category for grouping
        usage_count: Number of jobs/agents using this skill
        created_at: When the skill was added to the registry
    """
    
    id: str
    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    usage_count: int = 0
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate skill data."""
        # Normalize category
        if isinstance(self.category, SkillCategory):
            self.category = self.category.value
        
        # Validate name format (lowercase, alphanumeric, hyphens only)
        if not self.name:
            raise ValueError("Skill name cannot be empty")
        
        # Validate name length (max 50 chars - DB constraint)
        if len(self.name) > 50:
            raise ValueError(f"Skill name too long: {len(self.name)} chars (max 50)")
        
        # Must be lowercase and contain only alphanumeric or hyphens
        if not all(c.islower() or c.isdigit() or c == "-" for c in self.name):
            raise ValueError(
                f"Invalid skill name: {self.name}. "
                "Must be lowercase alphanumeric with hyphens only."
            )
        
        # Validate usage_count must be >= 0
        if self.usage_count < 0:
            raise ValueError(f"usage_count must be >= 0, got {self.usage_count}")
        
        # Validate category if provided
        if self.category:
            valid_categories = {c.value for c in SkillCategory}
            if self.category not in valid_categories:
                raise ValueError(
                    f"Invalid category: {self.category}. Must be one of {valid_categories}"
                )
    
    @property
    def is_canonical(self) -> bool:
        """Check if this is a canonical (pre-defined) skill."""
        return self.name in CANONICAL_SKILLS
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "usage_count": self.usage_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Skill":
        """Create from dictionary."""
        created_at = data.get("created_at")
        if created_at and isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description"),
            category=data.get("category"),
            usage_count=data.get("usage_count", 0),
            created_at=created_at,
        )
    
    @classmethod
    def from_canonical(cls, name: str, skill_id: str) -> "Skill":
        """Create a Skill from the canonical registry.
        
        Args:
            name: Canonical skill name
            skill_id: UUID to assign
            
        Returns:
            Skill instance
            
        Raises:
            ValueError: If skill is not in canonical registry
        """
        if name not in CANONICAL_SKILLS:
            raise ValueError(f"Unknown canonical skill: {name}")
        
        info = CANONICAL_SKILLS[name]
        return cls(
            id=skill_id,
            name=name,
            description=info["description"],
            category=info["category"].value,
        )
