"""Tests for the narrowed devtools import gate in CLI __main__.py.

The _import_devtools() helper must:
- Show "pip install kernle-devtools" only when kernle_devtools is genuinely missing
- Propagate ImportError from inside devtools (real bugs, not missing package)
- Propagate ModuleNotFoundError for non-devtools packages
- Show upgrade message for DevtoolsVersionError
- Behave identically across all three gate sites (session start, session list, report)
"""

import importlib
from types import ModuleType
from unittest.mock import patch

import pytest

from kernle.cli.__main__ import _import_devtools

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_devtools_module():
    """Create a fake kernle_devtools module with the expected symbols."""
    mod = ModuleType("kernle_devtools.admin_health.diagnostics")
    mod.cmd_doctor_session_start = lambda args, k: None
    mod.cmd_doctor_session_list = lambda args, k: None
    mod.cmd_doctor_report = lambda args, k: None
    return mod


# ---------------------------------------------------------------------------
# _import_devtools unit tests
# ---------------------------------------------------------------------------


class TestImportDevtoolsMissing:
    """ModuleNotFoundError for kernle_devtools → exit(2) + install message."""

    def test_devtools_missing_shows_install_message(self, capsys):
        with patch.object(
            importlib,
            "import_module",
            side_effect=ModuleNotFoundError(name="kernle_devtools"),
        ):
            with pytest.raises(SystemExit) as exc:
                _import_devtools(
                    "kernle_devtools.admin_health.diagnostics",
                    "cmd_doctor_session_start",
                )
            assert exc.value.code == 2
        captured = capsys.readouterr().out
        assert "pip install kernle-devtools" in captured

    def test_devtools_submodule_missing_shows_install_message(self, capsys):
        """ModuleNotFoundError with name='kernle_devtools.foo' still shows install msg."""
        with patch.object(
            importlib,
            "import_module",
            side_effect=ModuleNotFoundError(name="kernle_devtools.admin_health.diagnostics"),
        ):
            with pytest.raises(SystemExit) as exc:
                _import_devtools(
                    "kernle_devtools.admin_health.diagnostics",
                    "cmd_doctor_session_start",
                )
            assert exc.value.code == 2
        captured = capsys.readouterr().out
        assert "pip install kernle-devtools" in captured


class TestImportDevtoolsPropagates:
    """Errors that are NOT 'devtools missing' must propagate, not be swallowed."""

    def test_devtools_internal_import_error_propagates(self):
        """ImportError from inside devtools (e.g. bad internal import) raises."""
        with patch.object(
            importlib,
            "import_module",
            side_effect=ImportError("cannot import name 'foo' from 'kernle_devtools.bar'"),
        ):
            with pytest.raises(ImportError, match="cannot import name 'foo'"):
                _import_devtools(
                    "kernle_devtools.admin_health.diagnostics",
                    "cmd_doctor_session_start",
                )

    def test_non_devtools_module_not_found_propagates(self):
        """ModuleNotFoundError for a non-devtools package raises."""
        err = ModuleNotFoundError(name="some_other_pkg")
        with patch.object(
            importlib,
            "import_module",
            side_effect=err,
        ):
            with pytest.raises(ModuleNotFoundError) as exc:
                _import_devtools(
                    "kernle_devtools.admin_health.diagnostics",
                    "cmd_doctor_session_start",
                )
            assert exc.value.name == "some_other_pkg"


class TestDevtoolsVersionError:
    """DevtoolsVersionError (subclass of ImportError) → exit(2) + upgrade message."""

    def test_devtools_version_error_shows_upgrade_message(self, capsys):
        # Create a DevtoolsVersionError class that looks like it comes from kernle_devtools
        devtools_version_error_cls = type(
            "DevtoolsVersionError",
            (ImportError,),
            {"__module__": "kernle_devtools"},
        )
        err = devtools_version_error_cls("kernle-devtools requires kernle>=0.13.0")
        with patch.object(importlib, "import_module", side_effect=err):
            with pytest.raises(SystemExit) as exc:
                _import_devtools(
                    "kernle_devtools.admin_health.diagnostics",
                    "cmd_doctor_session_start",
                )
            assert exc.value.code == 2
        captured = capsys.readouterr().out
        assert "upgrade" in captured.lower() or "Upgrade" in captured


class TestImportDevtoolsSuccess:
    """When devtools is available, _import_devtools returns the requested symbol."""

    def test_returns_symbol(self, fake_devtools_module):
        with patch.object(importlib, "import_module", return_value=fake_devtools_module):
            fn = _import_devtools(
                "kernle_devtools.admin_health.diagnostics",
                "cmd_doctor_session_start",
            )
        assert fn is fake_devtools_module.cmd_doctor_session_start

    def test_missing_symbol_raises_attribute_error(self, fake_devtools_module):
        with patch.object(importlib, "import_module", return_value=fake_devtools_module):
            with pytest.raises(AttributeError):
                _import_devtools(
                    "kernle_devtools.admin_health.diagnostics",
                    "nonexistent_function",
                )


# ---------------------------------------------------------------------------
# Integration: all three gate sites behave identically
# ---------------------------------------------------------------------------


class TestAllThreeGatesConsistent:
    """Session start, session list, and report all use _import_devtools consistently."""

    def _run_main(self, argv, k):
        from kernle.cli.__main__ import main

        with patch("sys.argv", ["kernle"] + argv):
            with patch("kernle.cli.__main__.Kernle", return_value=k):
                with patch("kernle.cli.__main__.resolve_stack_id", return_value="test-main"):
                    main()

    @pytest.fixture
    def k(self, tmp_path):
        from kernle.core import Kernle
        from kernle.storage import SQLiteStorage

        s = SQLiteStorage(stack_id="test-main", db_path=tmp_path / "main.db")
        inst = Kernle(stack_id="test-main", storage=s, strict=False)
        yield inst
        s.close()

    @pytest.mark.parametrize(
        "argv",
        [
            ["doctor", "session", "start"],
            ["doctor", "session", "list"],
            ["doctor", "report", "latest"],
        ],
        ids=["session-start", "session-list", "report"],
    )
    def test_all_three_gates_missing_devtools(self, k, capsys, argv):
        """All three gates show install message when devtools is missing."""
        with patch(
            "kernle.cli.__main__._import_devtools",
            side_effect=SystemExit(2),
        ):
            with pytest.raises(SystemExit) as exc:
                self._run_main(argv, k)
            assert exc.value.code == 2

    @pytest.mark.parametrize(
        "argv",
        [
            ["doctor", "session", "start"],
            ["doctor", "session", "list"],
            ["doctor", "report", "latest"],
        ],
        ids=["session-start", "session-list", "report"],
    )
    def test_all_three_gates_propagate_internal_error(self, k, argv):
        """All three gates propagate ImportError (not swallowed as 'install devtools').

        Internal ImportError bubbles up through _import_devtools to main()'s
        general exception handler, which exits with code 1 (not 2).
        """
        with patch(
            "kernle.cli.__main__._import_devtools",
            side_effect=ImportError("bad internal import"),
        ):
            with pytest.raises(SystemExit) as exc:
                self._run_main(argv, k)
            # Exit code 1 (general error), NOT 2 (missing devtools)
            assert exc.value.code == 1
