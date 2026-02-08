"""Tests for _config.py — stack ID resolution and config."""

import os
from unittest.mock import patch

from _config import get_config, resolve_stack_id


class TestResolveStackId:
    def test_uses_env_var_first(self):
        with patch.dict(os.environ, {"KERNLE_STACK_ID": "env-stack"}):
            assert resolve_stack_id("/some/path") == "env-stack"

    def test_falls_back_to_cwd_dir_name(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("KERNLE_STACK_ID", None)
            assert resolve_stack_id("/Users/test/my-project") == "my-project"

    def test_skips_generic_dir_names(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("KERNLE_STACK_ID", None)
            assert resolve_stack_id("/home/user/workspace") is None

    def test_returns_none_when_no_context(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("KERNLE_STACK_ID", None)
            assert resolve_stack_id(None) is None

    def test_returns_none_for_empty_cwd(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("KERNLE_STACK_ID", None)
            assert resolve_stack_id("") is None

    def test_prefers_env_var_over_cwd(self):
        with patch.dict(os.environ, {"KERNLE_STACK_ID": "from-env"}):
            assert resolve_stack_id("/Users/test/other-project") == "from-env"

    def test_handles_nested_path(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("KERNLE_STACK_ID", None)
            assert resolve_stack_id("/Users/test/deep/nested/project") == "project"

    def test_skips_home_dir_name(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("KERNLE_STACK_ID", None)
            assert resolve_stack_id("/Users/test/home") is None


class TestGetConfig:
    def test_uses_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            for key in ("KERNLE_BIN", "KERNLE_TIMEOUT", "KERNLE_TOKEN_BUDGET"):
                os.environ.pop(key, None)
            config = get_config()
            assert config["kernle_bin"] == "kernle"
            assert config["timeout"] == 5
            assert config["token_budget"] == 8000

    def test_reads_env_vars(self):
        with patch.dict(
            os.environ,
            {
                "KERNLE_BIN": "/custom/kernle",
                "KERNLE_TIMEOUT": "10",
                "KERNLE_TOKEN_BUDGET": "4000",
            },
        ):
            config = get_config()
            assert config["kernle_bin"] == "/custom/kernle"
            assert config["timeout"] == 10
            assert config["token_budget"] == 4000
