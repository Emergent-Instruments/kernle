"""Tests for doctor.py backward compatibility wrappers."""

import warnings

import pytest


class TestStructuralDeprecationWarning:
    """Verify deprecated imports from doctor.py emit warnings."""

    def test_structural_finding_deprecation(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from kernle.cli.commands import doctor

            # Clear the module's cached attributes to force __getattr__
            sf = doctor.__getattr__("StructuralFinding")
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "kernle.structural" in str(w[0].message)
            assert sf is not None

    def test_run_structural_checks_deprecation(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from kernle.cli.commands import doctor

            fn = doctor.__getattr__("run_structural_checks")
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert callable(fn)


class TestSessionSymbolErrors:
    """Verify session/report symbols raise ImportError."""

    def test_session_start_raises_import_error(self):
        from kernle.cli.commands import doctor

        with pytest.raises(ImportError, match="kernle-devtools"):
            doctor.__getattr__("cmd_doctor_session_start")

    def test_report_raises_import_error(self):
        from kernle.cli.commands import doctor

        with pytest.raises(ImportError, match="kernle-devtools"):
            doctor.__getattr__("cmd_doctor_report")

    def test_generate_summary_raises_import_error(self):
        from kernle.cli.commands import doctor

        with pytest.raises(ImportError, match="kernle-devtools"):
            doctor.__getattr__("_generate_summary")


class TestUnknownAttribute:
    """Verify unknown attributes raise AttributeError."""

    def test_unknown_attr(self):
        from kernle.cli.commands import doctor

        with pytest.raises(AttributeError):
            doctor.__getattr__("nonexistent_function")
