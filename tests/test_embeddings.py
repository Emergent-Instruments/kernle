"""Tests for kernle/storage/embeddings.py."""

import math
import struct
from unittest.mock import MagicMock, patch

import pytest

from kernle.storage.embeddings import (
    HASH_EMBEDDING_DIM,
    HashEmbedder,
    OpenAIEmbedder,
    clear_embedder_cache,
    get_default_embedder,
    pack_embedding,
    unpack_embedding,
)

# ---------------------------------------------------------------------------
# HashEmbedder
# ---------------------------------------------------------------------------


class TestHashEmbedder:
    def test_dimension_default(self):
        e = HashEmbedder()
        assert e.dimension == HASH_EMBEDDING_DIM

    def test_dimension_custom(self):
        e = HashEmbedder(dim=128)
        assert e.dimension == 128

    def test_embed_returns_correct_dimension(self):
        e = HashEmbedder()
        vec = e.embed("hello world")
        assert len(vec) == HASH_EMBEDDING_DIM

    def test_embed_custom_dim(self):
        e = HashEmbedder(dim=64)
        vec = e.embed("hello world")
        assert len(vec) == 64

    def test_embed_empty_text_returns_zero_vector(self):
        e = HashEmbedder()
        vec = e.embed("")
        assert all(v == 0.0 for v in vec)
        assert len(vec) == HASH_EMBEDDING_DIM

    def test_embed_whitespace_only_returns_zero_vector(self):
        e = HashEmbedder()
        vec = e.embed("   ")
        assert all(v == 0.0 for v in vec)

    def test_embed_deterministic(self):
        e = HashEmbedder()
        v1 = e.embed("the quick brown fox")
        v2 = e.embed("the quick brown fox")
        assert v1 == v2

    def test_embed_different_texts_differ(self):
        e = HashEmbedder()
        v1 = e.embed("hello")
        v2 = e.embed("goodbye")
        assert v1 != v2

    def test_embed_unit_length(self):
        e = HashEmbedder()
        vec = e.embed("some text for embedding")
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-6

    def test_embed_single_char(self):
        """Single char is shorter than min ngram_range (2), so only word features."""
        e = HashEmbedder()
        vec = e.embed("a")
        # Should still produce a valid vector (word-level feature "a")
        assert len(vec) == HASH_EMBEDDING_DIM
        assert any(v != 0.0 for v in vec)

    def test_get_ngrams_normal_text(self):
        e = HashEmbedder(ngram_range=(2, 3))
        ngrams = e._get_ngrams("hi there")
        # Character n-grams of size 2 and 3, plus word features
        assert "hi" in ngrams  # 2-gram
        assert "i " in ngrams  # 2-gram
        assert "hi " in ngrams  # 3-gram
        # Word features
        assert "hi" in ngrams
        assert "there" in ngrams

    def test_get_ngrams_short_text(self):
        """Text shorter than min ngram size still returns word features."""
        e = HashEmbedder(ngram_range=(3, 4))
        ngrams = e._get_ngrams("ab")
        # "ab" is 2 chars, no 3-grams or 4-grams possible
        # But word feature "ab" is included
        assert "ab" in ngrams

    def test_get_ngrams_empty(self):
        e = HashEmbedder()
        ngrams = e._get_ngrams("")
        assert ngrams == []

    def test_custom_ngram_range(self):
        e = HashEmbedder(ngram_range=(1, 2))
        ngrams = e._get_ngrams("abc")
        # 1-grams: a, b, c  and 2-grams: ab, bc
        assert "a" in ngrams
        assert "b" in ngrams
        assert "c" in ngrams
        assert "ab" in ngrams
        assert "bc" in ngrams

    def test_embed_batch_default(self):
        """Base class embed_batch calls embed() for each text."""
        e = HashEmbedder()
        texts = ["hello", "world", "foo"]
        results = e.embed_batch(texts)
        assert len(results) == 3
        for i, text in enumerate(texts):
            assert results[i] == e.embed(text)

    def test_embed_batch_empty_list(self):
        e = HashEmbedder()
        assert e.embed_batch([]) == []


# ---------------------------------------------------------------------------
# OpenAIEmbedder
# ---------------------------------------------------------------------------


class TestOpenAIEmbedder:
    def test_dimension_small(self):
        e = OpenAIEmbedder(model="text-embedding-3-small")
        assert e.dimension == 1536

    def test_dimension_large(self):
        e = OpenAIEmbedder(model="text-embedding-3-large")
        assert e.dimension == 3072

    def test_dimension_ada(self):
        e = OpenAIEmbedder(model="text-embedding-ada-002")
        assert e.dimension == 1536

    def test_dimension_unknown_model(self):
        e = OpenAIEmbedder(model="some-future-model")
        assert e.dimension == 1536  # default fallback

    def test_get_client_import_error(self):
        e = OpenAIEmbedder()
        with patch.dict("sys.modules", {"openai": None}):
            # Reset cached client
            e._client = None
            with pytest.raises(RuntimeError, match="openai package not installed"):
                e._get_client()

    def test_get_client_lazy_init(self):
        """Client is created once and reused."""
        mock_openai = MagicMock()
        e = OpenAIEmbedder(api_key="test-key")
        with patch.dict("sys.modules", {"openai": mock_openai}):
            client1 = e._get_client()
            client2 = e._get_client()
            assert client1 is client2
            # OpenAI() should be called only once
            mock_openai.OpenAI.assert_called_once_with(api_key="test-key")

    def test_embed_calls_api_with_correct_params(self):
        """Verify embed() passes the configured model name and input text to the API."""
        e = OpenAIEmbedder(model="text-embedding-3-large")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_item = MagicMock()
        mock_item.embedding = [0.1]
        mock_response.data = [mock_item]
        mock_client.embeddings.create.return_value = mock_response
        e._client = mock_client

        e.embed("hello world")
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-large", input="hello world"
        )

    def test_embed_extracts_first_data_item(self):
        """Verify embed() returns the embedding from response.data[0], not data itself."""
        e = OpenAIEmbedder()
        mock_client = MagicMock()
        mock_response = MagicMock()
        # Put multiple items in data — embed() should take index 0
        item0 = MagicMock()
        item0.embedding = [1.0, 2.0]
        item1 = MagicMock()
        item1.embedding = [9.0, 9.0]
        mock_response.data = [item0, item1]
        mock_client.embeddings.create.return_value = mock_response
        e._client = mock_client

        result = e.embed("test")
        assert result == [1.0, 2.0]

    def test_embed_batch_sends_all_texts(self):
        """Verify embed_batch() sends all texts in a single API call."""
        e = OpenAIEmbedder()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[i]) for i in range(4)]
        mock_client.embeddings.create.return_value = mock_response
        e._client = mock_client

        texts = ["alpha", "beta", "gamma", "delta"]
        e.embed_batch(texts)
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small", input=texts
        )

    def test_embed_batch_preserves_order_from_response(self):
        """Verify embed_batch() returns embeddings in response.data order."""
        e = OpenAIEmbedder()
        mock_client = MagicMock()
        mock_response = MagicMock()
        # Simulate API returning items — batch should map them in order
        items = []
        for val in [[10.0], [20.0], [30.0]]:
            item = MagicMock()
            item.embedding = val
            items.append(item)
        mock_response.data = items
        mock_client.embeddings.create.return_value = mock_response
        e._client = mock_client

        result = e.embed_batch(["x", "y", "z"])
        assert result == [[10.0], [20.0], [30.0]]
        assert len(result) == 3

    def test_embed_batch_fewer_results_than_inputs(self):
        """If API returns fewer items than requested, result has fewer elements."""
        e = OpenAIEmbedder()
        mock_client = MagicMock()
        mock_response = MagicMock()
        # Only 2 items for 3 inputs — the code returns whatever the API gives
        mock_response.data = [MagicMock(embedding=[1.0]), MagicMock(embedding=[2.0])]
        mock_client.embeddings.create.return_value = mock_response
        e._client = mock_client

        result = e.embed_batch(["a", "b", "c"])
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------


class TestGetDefaultEmbedder:
    def setup_method(self):
        clear_embedder_cache()

    def teardown_method(self):
        clear_embedder_cache()

    def test_no_api_key_returns_hash(self):
        with patch.dict("os.environ", {}, clear=True):
            result = get_default_embedder()
            assert isinstance(result, HashEmbedder)

    def test_api_key_but_openai_fails_returns_hash(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            with patch.object(OpenAIEmbedder, "embed", side_effect=RuntimeError("no openai")):
                result = get_default_embedder()
                assert isinstance(result, HashEmbedder)

    def test_api_key_and_openai_works_returns_openai(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            with patch.object(OpenAIEmbedder, "embed", return_value=[0.1, 0.2]):
                result = get_default_embedder()
                assert isinstance(result, OpenAIEmbedder)

    def test_cache_prevents_recheck(self):
        """Once OpenAI availability is cached, subsequent calls use cache."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            with patch.object(OpenAIEmbedder, "embed", return_value=[0.1]) as mock_embed:
                get_default_embedder()
                get_default_embedder()
                # embed("test") should only be called once (cache kicks in)
                mock_embed.assert_called_once_with("test")


class TestClearEmbedderCache:
    def test_resets_openai_available(self):
        import kernle.storage.embeddings as mod

        mod._openai_available = True
        clear_embedder_cache()
        assert mod._openai_available is None


# ---------------------------------------------------------------------------
# pack / unpack
# ---------------------------------------------------------------------------


class TestPackUnpack:
    def test_round_trip(self):
        embedding = [0.1, 0.2, 0.3, -0.5, 1.0]
        packed = pack_embedding(embedding)
        unpacked = unpack_embedding(packed)
        assert len(unpacked) == len(embedding)
        for a, b in zip(embedding, unpacked):
            assert abs(a - b) < 1e-6

    def test_pack_produces_bytes(self):
        embedding = [1.0, 2.0, 3.0]
        packed = pack_embedding(embedding)
        assert isinstance(packed, bytes)
        # 3 floats * 4 bytes each = 12 bytes
        assert len(packed) == 12

    def test_round_trip_high_dim(self):
        embedding = [float(i) / 384 for i in range(384)]
        packed = pack_embedding(embedding)
        unpacked = unpack_embedding(packed)
        assert len(unpacked) == 384
        for a, b in zip(embedding, unpacked):
            assert abs(a - b) < 1e-6

    def test_empty_embedding(self):
        packed = pack_embedding([])
        assert packed == b""
        unpacked = unpack_embedding(b"")
        assert unpacked == []

    def test_unpack_size(self):
        """Unpack uses 4 bytes per float to determine count."""
        data = struct.pack("3f", 1.0, 2.0, 3.0)
        result = unpack_embedding(data)
        assert len(result) == 3

    def test_round_trip_negative_values(self):
        embedding = [-1.0, -0.5, 0.0, 0.5, 1.0]
        unpacked = unpack_embedding(pack_embedding(embedding))
        for a, b in zip(embedding, unpacked):
            assert abs(a - b) < 1e-6
