"""
Entry point discovery for kernle components.

Uses importlib.metadata.entry_points() to discover installed plugins,
stacks, models, and stack components registered via pyproject.toml
entry point groups.
"""

import importlib.metadata
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from kernle.protocols import (
    ENTRY_POINT_GROUP_MODELS,
    ENTRY_POINT_GROUP_PLUGINS,
    ENTRY_POINT_GROUP_STACK_COMPONENTS,
    ENTRY_POINT_GROUP_STACKS,
)

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredComponent:
    """Metadata about a discovered entry point component."""

    name: str
    group: str
    module: str
    attr: str
    dist_name: Optional[str] = None
    dist_version: Optional[str] = None
    error: Optional[str] = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def qualname(self) -> str:
        """Full qualified reference: 'module:attr'."""
        return f"{self.module}:{self.attr}"


def _get_entry_points(group: str) -> list[importlib.metadata.EntryPoint]:
    """Get entry points for a group, compatible with Python 3.10+."""
    return list(importlib.metadata.entry_points(group=group))


def _entry_point_to_component(ep: importlib.metadata.EntryPoint, group: str) -> DiscoveredComponent:
    """Convert an EntryPoint to a DiscoveredComponent."""
    # ep.value is "module:attr"
    parts = ep.value.split(":")
    module = parts[0]
    attr = parts[1] if len(parts) > 1 else ""

    dist_name = None
    dist_version = None
    if ep.dist is not None:
        dist_name = ep.dist.name
        dist_version = ep.dist.version

    return DiscoveredComponent(
        name=ep.name,
        group=group,
        module=module,
        attr=attr,
        dist_name=dist_name,
        dist_version=dist_version,
    )


def discover_plugins() -> list[DiscoveredComponent]:
    """Discover installed plugins (kernle.plugins entry point group).

    Returns:
        List of discovered plugin components.
    """
    eps = _get_entry_points(ENTRY_POINT_GROUP_PLUGINS)
    return [_entry_point_to_component(ep, ENTRY_POINT_GROUP_PLUGINS) for ep in eps]


def discover_stacks() -> list[DiscoveredComponent]:
    """Discover installed stack implementations (kernle.stacks entry point group).

    Returns:
        List of discovered stack components.
    """
    eps = _get_entry_points(ENTRY_POINT_GROUP_STACKS)
    return [_entry_point_to_component(ep, ENTRY_POINT_GROUP_STACKS) for ep in eps]


def discover_models() -> list[DiscoveredComponent]:
    """Discover installed model implementations (kernle.models entry point group).

    Returns:
        List of discovered model components.
    """
    eps = _get_entry_points(ENTRY_POINT_GROUP_MODELS)
    return [_entry_point_to_component(ep, ENTRY_POINT_GROUP_MODELS) for ep in eps]


def discover_stack_components() -> list[DiscoveredComponent]:
    """Discover installed stack components (kernle.stack_components entry point group).

    Returns:
        List of discovered stack sub-components.
    """
    eps = _get_entry_points(ENTRY_POINT_GROUP_STACK_COMPONENTS)
    return [_entry_point_to_component(ep, ENTRY_POINT_GROUP_STACK_COMPONENTS) for ep in eps]


def discover_all() -> dict[str, list[DiscoveredComponent]]:
    """Discover all registered kernle components across all entry point groups.

    Returns:
        Dict keyed by group name, each value a list of DiscoveredComponent.
    """
    return {
        ENTRY_POINT_GROUP_PLUGINS: discover_plugins(),
        ENTRY_POINT_GROUP_STACKS: discover_stacks(),
        ENTRY_POINT_GROUP_MODELS: discover_models(),
        ENTRY_POINT_GROUP_STACK_COMPONENTS: discover_stack_components(),
    }


def load_component(component: DiscoveredComponent) -> Any:
    """Load (import) a discovered component's class/factory.

    Args:
        component: The component to load.

    Returns:
        The loaded class or callable.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the attribute doesn't exist in the module.
    """
    eps = _get_entry_points(component.group)
    for ep in eps:
        if ep.name == component.name:
            return ep.load()
    raise ImportError(f"Entry point '{component.name}' not found in group '{component.group}'")
