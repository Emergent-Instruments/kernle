"""Tests for Issue #353: InferenceService scope declarations on components.

Verifies each built-in component declares a valid inference_scope.
"""

import pytest

from kernle.stack.components import (
    AnxietyComponent,
    ConsolidationComponent,
    EmbeddingComponent,
    EmotionalTaggingComponent,
    ForgettingComponent,
    KnowledgeComponent,
    MetaMemoryComponent,
    SuggestionComponent,
)

VALID_SCOPES = {"fast", "capable", "embedding", "none"}

EXPECTED_SCOPES = {
    "embedding-ngram": "embedding",
    "emotions": "fast",
    "consolidation": "capable",
    "suggestions": "capable",
    "forgetting": "none",
    "anxiety": "none",
    "metamemory": "fast",
    "knowledge": "none",
}


class TestInferenceScope:
    """Verify inference_scope is declared and valid on all built-in components."""

    @pytest.mark.parametrize(
        "component_cls",
        [
            EmbeddingComponent,
            EmotionalTaggingComponent,
            ConsolidationComponent,
            SuggestionComponent,
            ForgettingComponent,
            AnxietyComponent,
            MetaMemoryComponent,
            KnowledgeComponent,
        ],
    )
    def test_scope_is_valid(self, component_cls):
        component = component_cls()
        scope = component.inference_scope
        assert scope in VALID_SCOPES, (
            f"{component.name} has invalid inference_scope '{scope}'. "
            f"Must be one of {VALID_SCOPES}"
        )

    @pytest.mark.parametrize(
        "component_cls",
        [
            EmbeddingComponent,
            EmotionalTaggingComponent,
            ConsolidationComponent,
            SuggestionComponent,
            ForgettingComponent,
            AnxietyComponent,
            MetaMemoryComponent,
            KnowledgeComponent,
        ],
    )
    def test_scope_matches_expected(self, component_cls):
        component = component_cls()
        expected = EXPECTED_SCOPES.get(component.name)
        assert expected is not None, f"No expected scope for {component.name}"
        assert (
            component.inference_scope == expected
        ), f"{component.name}: expected scope '{expected}', got '{component.inference_scope}'"

    def test_needs_inference_consistency(self):
        """Components with scope 'none' should have needs_inference=False."""
        for cls in [ForgettingComponent, AnxietyComponent, KnowledgeComponent]:
            c = cls()
            if c.inference_scope == "none":
                # 'none' scope doesn't strictly require needs_inference=False
                # (knowledge has needs_inference=True for optional use) but
                # forgetting and anxiety should be False
                if c.name in ("forgetting", "anxiety"):
                    assert (
                        not c.needs_inference
                    ), f"{c.name} has scope 'none' but needs_inference=True"

    def test_all_default_components_have_scope(self):
        """All 11 default components should have inference_scope."""
        from kernle.stack.components import get_default_components

        components = get_default_components()
        assert len(components) == 11
        for c in components:
            assert hasattr(c, "inference_scope"), f"Component {c.name} missing inference_scope"
            assert (
                c.inference_scope in VALID_SCOPES
            ), f"Component {c.name} has invalid scope '{c.inference_scope}'"
