"""Tests for InferenceService implementation."""

from unittest.mock import MagicMock

from kernle.inference import _InferenceServiceImpl, create_inference_service
from kernle.protocols import (
    InferenceService,
    ModelCapabilities,
    ModelProtocol,
    ModelResponse,
)
from kernle.storage.embeddings import HASH_EMBEDDING_DIM

# ---- Fixtures ----


def _make_mock_model(model_id="test-model", response_text="Hello world"):
    """Create a mock ModelProtocol."""
    model = MagicMock(spec=ModelProtocol)
    model.model_id = model_id
    model.capabilities = ModelCapabilities(
        model_id=model_id,
        provider="test",
        context_window=4096,
    )
    model.generate.return_value = ModelResponse(content=response_text)
    return model


# ---- InferenceService Protocol Conformance ----


class TestInferenceServiceProtocol:
    """Test that _InferenceServiceImpl conforms to InferenceService."""

    def test_is_inference_service(self):
        model = _make_mock_model()
        svc = _InferenceServiceImpl(model)
        assert isinstance(svc, InferenceService)

    def test_has_infer_method(self):
        model = _make_mock_model()
        svc = _InferenceServiceImpl(model)
        assert callable(svc.infer)

    def test_has_embed_method(self):
        model = _make_mock_model()
        svc = _InferenceServiceImpl(model)
        assert callable(svc.embed)

    def test_has_embed_batch_method(self):
        model = _make_mock_model()
        svc = _InferenceServiceImpl(model)
        assert callable(svc.embed_batch)

    def test_has_embedding_dimension_property(self):
        model = _make_mock_model()
        svc = _InferenceServiceImpl(model)
        assert isinstance(svc.embedding_dimension, int)

    def test_has_embedding_provider_id_property(self):
        model = _make_mock_model()
        svc = _InferenceServiceImpl(model)
        assert isinstance(svc.embedding_provider_id, str)


# ---- Infer ----


class TestInfer:
    """Test infer() delegates to model.generate()."""

    def test_infer_returns_string(self):
        model = _make_mock_model(response_text="The answer is 42")
        svc = _InferenceServiceImpl(model)
        result = svc.infer("What is the answer?")
        assert result == "The answer is 42"

    def test_infer_calls_generate_with_user_message(self):
        model = _make_mock_model()
        svc = _InferenceServiceImpl(model)
        svc.infer("Hello")
        model.generate.assert_called_once()
        args, kwargs = model.generate.call_args
        messages = args[0]
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"

    def test_infer_passes_system_prompt(self):
        model = _make_mock_model()
        svc = _InferenceServiceImpl(model)
        svc.infer("Hello", system="You are helpful")
        _, kwargs = model.generate.call_args
        assert kwargs["system"] == "You are helpful"

    def test_infer_without_system_prompt(self):
        model = _make_mock_model()
        svc = _InferenceServiceImpl(model)
        svc.infer("Hello")
        _, kwargs = model.generate.call_args
        assert kwargs["system"] is None

    def test_infer_returns_empty_for_empty_response(self):
        model = _make_mock_model(response_text="")
        svc = _InferenceServiceImpl(model)
        result = svc.infer("Hello")
        assert result == ""


# ---- Embed ----


class TestEmbed:
    """Test embed() uses local hash embedder."""

    def test_embed_returns_float_list(self):
        model = _make_mock_model()
        svc = _InferenceServiceImpl(model)
        result = svc.embed("hello world")
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

    def test_embed_returns_correct_dimension(self):
        model = _make_mock_model()
        svc = _InferenceServiceImpl(model)
        result = svc.embed("hello world")
        assert len(result) == HASH_EMBEDDING_DIM

    def test_embed_is_deterministic(self):
        model = _make_mock_model()
        svc = _InferenceServiceImpl(model)
        r1 = svc.embed("hello world")
        r2 = svc.embed("hello world")
        assert r1 == r2

    def test_embed_different_text_different_vectors(self):
        model = _make_mock_model()
        svc = _InferenceServiceImpl(model)
        r1 = svc.embed("hello world")
        r2 = svc.embed("completely different text")
        assert r1 != r2

    def test_embed_does_not_call_model(self):
        model = _make_mock_model()
        svc = _InferenceServiceImpl(model)
        svc.embed("hello world")
        model.generate.assert_not_called()


# ---- Embed Batch ----


class TestEmbedBatch:
    """Test embed_batch()."""

    def test_embed_batch_returns_list_of_lists(self):
        model = _make_mock_model()
        svc = _InferenceServiceImpl(model)
        results = svc.embed_batch(["hello", "world"])
        assert len(results) == 2
        assert all(isinstance(r, list) for r in results)

    def test_embed_batch_consistent_with_embed(self):
        model = _make_mock_model()
        svc = _InferenceServiceImpl(model)
        single = svc.embed("hello")
        batch = svc.embed_batch(["hello"])
        assert single == batch[0]

    def test_embed_batch_empty_list(self):
        model = _make_mock_model()
        svc = _InferenceServiceImpl(model)
        results = svc.embed_batch([])
        assert results == []


# ---- Properties ----


class TestProperties:
    """Test embedding_dimension and embedding_provider_id."""

    def test_embedding_dimension_is_hash_dim(self):
        model = _make_mock_model()
        svc = _InferenceServiceImpl(model)
        assert svc.embedding_dimension == HASH_EMBEDDING_DIM

    def test_embedding_provider_id_is_ngram(self):
        model = _make_mock_model()
        svc = _InferenceServiceImpl(model)
        assert svc.embedding_provider_id == "ngram-v1"


# ---- Factory Function ----


class TestCreateInferenceService:
    """Test the create_inference_service factory."""

    def test_returns_inference_service(self):
        model = _make_mock_model()
        svc = create_inference_service(model)
        assert isinstance(svc, InferenceService)

    def test_factory_wraps_model(self):
        model = _make_mock_model(response_text="factory test")
        svc = create_inference_service(model)
        result = svc.infer("test")
        assert result == "factory test"


# ---- Entity Integration ----


class TestEntityIntegration:
    """Test that Entity._get_inference_service() returns a real service."""

    def test_entity_returns_none_without_model(self):
        from kernle.entity import Entity

        entity = Entity(core_id="test-core")
        assert entity._get_inference_service() is None

    def test_entity_returns_service_with_model(self):
        from kernle.entity import Entity

        entity = Entity(core_id="test-core")
        model = _make_mock_model()
        entity._model = model
        svc = entity._get_inference_service()
        assert svc is not None
        assert isinstance(svc, InferenceService)

    def test_set_model_propagates_inference_to_stacks(self):
        from kernle.entity import Entity

        entity = Entity(core_id="test-core")
        mock_stack = MagicMock()
        mock_stack.stack_id = "stack-1"
        mock_stack.schema_version = 1
        mock_stack.get_stats.return_value = {}
        entity.attach_stack(mock_stack, alias="main")

        model = _make_mock_model()
        entity.set_model(model)

        # on_attach is called with None inference (no model at attach time)
        # then on_model_changed is called with real inference
        mock_stack.on_model_changed.assert_called_once()
        inference_arg = mock_stack.on_model_changed.call_args[0][0]
        assert isinstance(inference_arg, InferenceService)

    def test_attach_stack_passes_inference_when_model_exists(self):
        from kernle.entity import Entity

        entity = Entity(core_id="test-core")
        model = _make_mock_model()
        entity._model = model

        mock_stack = MagicMock()
        mock_stack.stack_id = "stack-1"
        mock_stack.schema_version = 1
        mock_stack.get_stats.return_value = {}
        entity.attach_stack(mock_stack, alias="main")

        # on_attach should receive the inference service
        mock_stack.on_attach.assert_called_once()
        _, inference_arg = mock_stack.on_attach.call_args[0]
        assert isinstance(inference_arg, InferenceService)
