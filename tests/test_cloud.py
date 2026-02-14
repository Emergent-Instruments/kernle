"""Tests for kernle/storage/cloud.py — CloudClient mixin.

Covers:
- _validate_backend_url: scheme/host validation
- _load_cloud_credentials: credentials.json, config.json fallback, env vars
- cloud_health_check: success, timeout, error, missing creds
- _cloud_search: success, empty, HTTP error, network error, malformed JSON
- _parse_cloud_search_results: episode, note, belief, value, goal parsing
- _create_record_from_cloud: all 5 types + unknown type
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from kernle.storage import Belief, Episode, Goal, Note, SearchResult, Value
from kernle.storage.cloud import CloudClient


@pytest.fixture
def client():
    """A bare CloudClient for unit testing."""
    return CloudClient(stack_id="test-stack")


# ---------------------------------------------------------------------------
# 1. _validate_backend_url
# ---------------------------------------------------------------------------


class TestValidateBackendUrl:
    def test_valid_https(self, client):
        assert client._validate_backend_url("https://api.kernle.ai") == "https://api.kernle.ai"

    def test_valid_http_localhost(self, client):
        assert client._validate_backend_url("http://localhost:8000") == "http://localhost:8000"

    def test_valid_http_127(self, client):
        url = "http://127.0.0.1:8000/v1"
        assert client._validate_backend_url(url) == url

    def test_http_non_local_rejected(self, client):
        assert client._validate_backend_url("http://evil.com/api") is None

    def test_ftp_rejected(self, client):
        assert client._validate_backend_url("ftp://files.example.com") is None

    def test_empty_scheme_rejected(self, client):
        assert client._validate_backend_url("://no-scheme") is None

    def test_missing_netloc_rejected(self, client):
        assert client._validate_backend_url("https://") is None

    def test_empty_string_rejected(self, client):
        assert client._validate_backend_url("") is None

    def test_bare_path_rejected(self, client):
        # urlparse("just-a-path") gives scheme='', netloc=''
        assert client._validate_backend_url("just-a-path") is None


# ---------------------------------------------------------------------------
# 2. _load_cloud_credentials
# ---------------------------------------------------------------------------


class TestLoadCloudCredentials:
    def test_valid_credentials_json(self, client, tmp_path):
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(
            json.dumps({"backend_url": "https://api.kernle.ai", "auth_token": "tok123"})
        )
        with patch("kernle.storage.cloud.get_kernle_home", return_value=tmp_path):
            result = client._load_cloud_credentials()
        assert result == {"backend_url": "https://api.kernle.ai", "auth_token": "tok123"}

    def test_legacy_token_key(self, client, tmp_path):
        """credentials.json with 'token' instead of 'auth_token'."""
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(
            json.dumps({"backend_url": "https://api.kernle.ai", "token": "legacy-tok"})
        )
        with patch("kernle.storage.cloud.get_kernle_home", return_value=tmp_path):
            result = client._load_cloud_credentials()
        assert result is not None
        assert result["auth_token"] == "legacy-tok"

    def test_missing_file_returns_none(self, client, tmp_path):
        with patch("kernle.storage.cloud.get_kernle_home", return_value=tmp_path):
            result = client._load_cloud_credentials()
        assert result is None

    def test_malformed_json_returns_none(self, client, tmp_path):
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text("{not valid json")
        with patch("kernle.storage.cloud.get_kernle_home", return_value=tmp_path):
            result = client._load_cloud_credentials()
        assert result is None

    def test_falls_back_to_config_json(self, client, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(
            json.dumps({"backend_url": "https://config.kernle.ai", "auth_token": "cfg-tok"})
        )
        with patch("kernle.storage.cloud.get_kernle_home", return_value=tmp_path):
            result = client._load_cloud_credentials()
        assert result == {"backend_url": "https://config.kernle.ai", "auth_token": "cfg-tok"}

    def test_env_vars_override_files(self, client, tmp_path, monkeypatch):
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(
            json.dumps({"backend_url": "https://file.kernle.ai", "auth_token": "file-tok"})
        )
        monkeypatch.setenv("KERNLE_BACKEND_URL", "https://env.kernle.ai")
        monkeypatch.setenv("KERNLE_AUTH_TOKEN", "env-tok")
        with patch("kernle.storage.cloud.get_kernle_home", return_value=tmp_path):
            result = client._load_cloud_credentials()
        assert result == {"backend_url": "https://env.kernle.ai", "auth_token": "env-tok"}

    def test_invalid_backend_url_returns_none(self, client, tmp_path):
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps({"backend_url": "ftp://bad.host", "auth_token": "tok"}))
        with patch("kernle.storage.cloud.get_kernle_home", return_value=tmp_path):
            result = client._load_cloud_credentials()
        assert result is None

    def test_caching(self, client, tmp_path):
        """Second call returns cached value without re-reading files."""
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(
            json.dumps({"backend_url": "https://api.kernle.ai", "auth_token": "tok"})
        )
        with patch("kernle.storage.cloud.get_kernle_home", return_value=tmp_path):
            first = client._load_cloud_credentials()
            # Mutate file — should not affect cached result
            creds_file.write_text(
                json.dumps({"backend_url": "https://other.ai", "auth_token": "x"})
            )
            second = client._load_cloud_credentials()
        assert first is second  # same object (cached)

    def test_partial_credentials_from_file_and_config(self, client, tmp_path):
        """credentials.json has backend_url only, config.json fills in auth_token."""
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps({"backend_url": "https://api.kernle.ai"}))
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"auth_token": "cfg-tok"}))
        with patch("kernle.storage.cloud.get_kernle_home", return_value=tmp_path):
            result = client._load_cloud_credentials()
        assert result == {"backend_url": "https://api.kernle.ai", "auth_token": "cfg-tok"}

    def test_malformed_config_json_fallback(self, client, tmp_path):
        """Malformed config.json is gracefully ignored."""
        config_file = tmp_path / "config.json"
        config_file.write_text("{bad json!!!")
        with patch("kernle.storage.cloud.get_kernle_home", return_value=tmp_path):
            result = client._load_cloud_credentials()
        assert result is None


# ---------------------------------------------------------------------------
# 3. cloud_health_check
# ---------------------------------------------------------------------------


def _make_creds(url="https://api.kernle.ai", token="tok"):
    return {"backend_url": url, "auth_token": token}


class TestCloudHealthCheck:
    def test_missing_credentials(self, client):
        with patch.object(client, "_load_cloud_credentials", return_value=None):
            result = client.cloud_health_check()
        assert result["healthy"] is False
        assert "credentials" in result["error"].lower()

    def test_success(self, client):
        resp = MagicMock()
        resp.status = 200
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)

        with (
            patch.object(client, "_load_cloud_credentials", return_value=_make_creds()),
            patch("urllib.request.urlopen", return_value=resp),
        ):
            result = client.cloud_health_check()
        assert result["healthy"] is True
        assert "latency_ms" in result

    def test_url_error(self, client):
        import urllib.error

        with (
            patch.object(client, "_load_cloud_credentials", return_value=_make_creds()),
            patch(
                "urllib.request.urlopen",
                side_effect=urllib.error.URLError("timeout"),
            ),
        ):
            result = client.cloud_health_check()
        assert result["healthy"] is False
        assert "Connection failed" in result["error"]

    def test_general_exception(self, client):
        with (
            patch.object(client, "_load_cloud_credentials", return_value=_make_creds()),
            patch("urllib.request.urlopen", side_effect=RuntimeError("boom")),
        ):
            result = client.cloud_health_check()
        assert result["healthy"] is False
        assert "boom" in result["error"]

    def test_non_200_status(self, client):
        resp = MagicMock()
        resp.status = 503
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)

        with (
            patch.object(client, "_load_cloud_credentials", return_value=_make_creds()),
            patch("urllib.request.urlopen", return_value=resp),
        ):
            result = client.cloud_health_check()
        assert result["healthy"] is False
        assert "503" in result["error"]


# ---------------------------------------------------------------------------
# 4. _cloud_search
# ---------------------------------------------------------------------------


def _mock_urlopen_response(data, status=200):
    """Create a mock response for urllib.request.urlopen."""
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = json.dumps(data).encode("utf-8")
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestCloudSearch:
    def test_missing_credentials(self, client):
        with patch.object(client, "_load_cloud_credentials", return_value=None):
            assert client._cloud_search("hello") is None

    def test_success_with_results(self, client):
        response_data = {
            "results": [
                {
                    "id": "ep-1",
                    "memory_type": "episodes",
                    "content": "test objective",
                    "score": 0.9,
                    "created_at": "2024-01-01T00:00:00Z",
                    "metadata": {"objective": "test objective", "outcome": "ok"},
                }
            ]
        }
        resp = _mock_urlopen_response(response_data)
        with (
            patch.object(client, "_load_cloud_credentials", return_value=_make_creds()),
            patch("urllib.request.urlopen", return_value=resp),
        ):
            results = client._cloud_search("test")
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].record_type == "episode"
        assert results[0].score == 0.9

    def test_empty_results(self, client):
        resp = _mock_urlopen_response({"results": []})
        with (
            patch.object(client, "_load_cloud_credentials", return_value=_make_creds()),
            patch("urllib.request.urlopen", return_value=resp),
        ):
            results = client._cloud_search("nothing")
        assert results == []

    def test_http_error_status(self, client):
        resp = _mock_urlopen_response({}, status=500)
        with (
            patch.object(client, "_load_cloud_credentials", return_value=_make_creds()),
            patch("urllib.request.urlopen", return_value=resp),
        ):
            assert client._cloud_search("test") is None

    def test_network_error(self, client):
        import urllib.error

        with (
            patch.object(client, "_load_cloud_credentials", return_value=_make_creds()),
            patch(
                "urllib.request.urlopen",
                side_effect=urllib.error.URLError("network down"),
            ),
        ):
            assert client._cloud_search("test") is None

    def test_malformed_json_response(self, client):
        resp = MagicMock()
        resp.status = 200
        resp.read.return_value = b"not json at all"
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)

        with (
            patch.object(client, "_load_cloud_credentials", return_value=_make_creds()),
            patch("urllib.request.urlopen", return_value=resp),
        ):
            assert client._cloud_search("test") is None

    def test_record_types_mapping(self, client):
        """record_types are mapped to backend table names in the payload."""
        resp = _mock_urlopen_response({"results": []})
        with (
            patch.object(client, "_load_cloud_credentials", return_value=_make_creds()),
            patch("urllib.request.urlopen", return_value=resp) as mock_open,
        ):
            client._cloud_search("test", record_types=["episode", "belief"])
            # Verify the request payload included memory_types
            call_args = mock_open.call_args
            req = call_args[0][0]
            body = json.loads(req.data.decode("utf-8"))
            assert body["memory_types"] == ["episodes", "beliefs"]

    def test_general_exception(self, client):
        with (
            patch.object(client, "_load_cloud_credentials", return_value=_make_creds()),
            patch("urllib.request.urlopen", side_effect=RuntimeError("unexpected")),
        ):
            assert client._cloud_search("test") is None


# ---------------------------------------------------------------------------
# 5. _parse_cloud_search_results
# ---------------------------------------------------------------------------


class TestParseCloudSearchResults:
    def test_parse_episode(self, client):
        data = {
            "results": [
                {
                    "id": "ep-1",
                    "memory_type": "episodes",
                    "content": "did a thing",
                    "score": 0.95,
                    "created_at": "2024-06-01T12:00:00Z",
                    "metadata": {
                        "objective": "do a thing",
                        "outcome": "success",
                        "outcome_type": "success",
                    },
                }
            ]
        }
        results = client._parse_cloud_search_results(data)
        assert len(results) == 1
        r = results[0]
        assert r.record_type == "episode"
        assert isinstance(r.record, Episode)
        assert r.record.objective == "do a thing"
        assert r.record.outcome == "success"

    def test_parse_note(self, client):
        data = {
            "results": [
                {
                    "id": "n-1",
                    "memory_type": "notes",
                    "content": "important insight",
                    "score": 0.8,
                    "created_at": None,
                    "metadata": {"note_type": "insight", "tags": ["test"]},
                }
            ]
        }
        results = client._parse_cloud_search_results(data)
        assert len(results) == 1
        r = results[0]
        assert r.record_type == "note"
        assert isinstance(r.record, Note)
        assert r.record.content == "important insight"
        assert r.record.note_type == "insight"

    def test_parse_belief(self, client):
        data = {
            "results": [
                {
                    "id": "b-1",
                    "memory_type": "beliefs",
                    "content": "the sky is blue",
                    "score": 0.7,
                    "created_at": "2024-01-01T00:00:00Z",
                    "metadata": {"belief_type": "observation", "confidence": 0.95},
                }
            ]
        }
        results = client._parse_cloud_search_results(data)
        assert len(results) == 1
        r = results[0]
        assert r.record_type == "belief"
        assert isinstance(r.record, Belief)
        assert r.record.statement == "the sky is blue"
        assert r.record.confidence == 0.95

    def test_parse_value(self, client):
        data = {
            "results": [
                {
                    "id": "v-1",
                    "memory_type": "values",
                    "content": "honesty matters",
                    "score": 0.6,
                    "created_at": None,
                    "metadata": {"name": "honesty", "priority": 90},
                }
            ]
        }
        results = client._parse_cloud_search_results(data)
        assert len(results) == 1
        r = results[0]
        assert r.record_type == "value"
        assert isinstance(r.record, Value)
        assert r.record.name == "honesty"
        assert r.record.priority == 90

    def test_parse_goal(self, client):
        data = {
            "results": [
                {
                    "id": "g-1",
                    "memory_type": "goals",
                    "content": "learn rust",
                    "score": 0.5,
                    "created_at": None,
                    "metadata": {
                        "title": "Learn Rust",
                        "description": "Systems programming",
                        "priority": "high",
                        "status": "active",
                    },
                }
            ]
        }
        results = client._parse_cloud_search_results(data)
        assert len(results) == 1
        r = results[0]
        assert r.record_type == "goal"
        assert isinstance(r.record, Goal)
        assert r.record.title == "Learn Rust"
        assert r.record.priority == "high"

    def test_empty_results(self, client):
        results = client._parse_cloud_search_results({"results": []})
        assert results == []

    def test_missing_results_key(self, client):
        results = client._parse_cloud_search_results({})
        assert results == []

    def test_unknown_type_skipped(self, client):
        data = {
            "results": [
                {
                    "id": "x-1",
                    "memory_type": "unknown_type",
                    "content": "mystery",
                    "score": 0.5,
                    "metadata": {},
                }
            ]
        }
        results = client._parse_cloud_search_results(data)
        assert results == []

    def test_default_score(self, client):
        """Items without score default to 1.0."""
        data = {
            "results": [
                {
                    "id": "n-1",
                    "memory_type": "notes",
                    "content": "no score",
                    "metadata": {},
                }
            ]
        }
        results = client._parse_cloud_search_results(data)
        assert results[0].score == 1.0


# ---------------------------------------------------------------------------
# 6. _create_record_from_cloud
# ---------------------------------------------------------------------------


class TestCreateRecordFromCloud:
    def _item(self, **overrides):
        base = {"id": "rec-1", "content": "content", "created_at": "2024-01-01T00:00:00Z"}
        base.update(overrides)
        return base

    def test_episode(self, client):
        item = self._item()
        meta = {
            "objective": "obj",
            "outcome": "ok",
            "outcome_type": "success",
            "lessons_learned": ["a"],
        }
        record = client._create_record_from_cloud("episode", item, meta)
        assert isinstance(record, Episode)
        assert record.objective == "obj"
        assert record.outcome == "ok"
        assert record.outcome_type == "success"
        assert record.lessons == ["a"]
        assert record.stack_id == "test-stack"

    def test_note(self, client):
        item = self._item()
        meta = {"note_type": "decision", "tags": ["t1"]}
        record = client._create_record_from_cloud("note", item, meta)
        assert isinstance(record, Note)
        assert record.note_type == "decision"
        assert record.tags == ["t1"]

    def test_belief(self, client):
        item = self._item()
        meta = {"belief_type": "hypothesis", "confidence": 0.6}
        record = client._create_record_from_cloud("belief", item, meta)
        assert isinstance(record, Belief)
        assert record.belief_type == "hypothesis"
        assert record.confidence == 0.6

    def test_value(self, client):
        item = self._item()
        meta = {"name": "curiosity", "priority": 75}
        record = client._create_record_from_cloud("value", item, meta)
        assert isinstance(record, Value)
        assert record.name == "curiosity"
        assert record.priority == 75

    def test_goal(self, client):
        item = self._item()
        meta = {
            "title": "Ship v2",
            "description": "Release it",
            "priority": "high",
            "status": "active",
        }
        record = client._create_record_from_cloud("goal", item, meta)
        assert isinstance(record, Goal)
        assert record.title == "Ship v2"
        assert record.description == "Release it"

    def test_unknown_type_returns_none(self, client):
        record = client._create_record_from_cloud("playlist", self._item(), {})
        assert record is None

    def test_defaults_when_metadata_empty(self, client):
        """Verify sensible defaults when metadata is empty."""
        item = self._item()
        meta = {}

        ep = client._create_record_from_cloud("episode", item, meta)
        assert ep.objective == "content"  # falls back to item content
        assert ep.outcome == ""

        note = client._create_record_from_cloud("note", item, meta)
        assert note.note_type == "note"

        belief = client._create_record_from_cloud("belief", item, meta)
        assert belief.belief_type == "fact"
        assert belief.confidence == 0.8

        value = client._create_record_from_cloud("value", item, meta)
        assert value.name == ""
        assert value.priority == 50

        goal = client._create_record_from_cloud("goal", item, meta)
        assert goal.title == "content"
        assert goal.priority == "medium"
        assert goal.status == "active"


# ---------------------------------------------------------------------------
# 7. has_cloud_credentials
# ---------------------------------------------------------------------------


class TestHasCloudCredentials:
    def test_true_when_configured(self, client):
        with patch.object(client, "_load_cloud_credentials", return_value=_make_creds()):
            assert client.has_cloud_credentials() is True

    def test_false_when_missing(self, client):
        with patch.object(client, "_load_cloud_credentials", return_value=None):
            assert client.has_cloud_credentials() is False
