"""Tests for corpus ingestion pipeline."""

import textwrap

import pytest

from kernle import Kernle
from kernle.corpus import (
    CorpusIngestor,
    chunk_generic,
    chunk_markdown,
    chunk_python,
)


@pytest.fixture
def k(tmp_path):
    """Create a Kernle instance with temp storage in an isolated directory."""
    from kernle.storage.sqlite import SQLiteStorage

    db_path = tmp_path / "test_corpus.db"
    storage = SQLiteStorage(stack_id="test-corpus", db_path=db_path)
    return Kernle(stack_id="test-corpus", storage=storage)


@pytest.fixture
def sample_repo(tmp_path):
    """Create a sample repository structure for testing."""
    # Python file with functions and class
    py_file = tmp_path / "main.py"
    py_file.write_text(textwrap.dedent("""\
        import os

        def hello(name):
            return f"Hello, {name}!"

        def goodbye(name):
            return f"Goodbye, {name}!"

        class Greeter:
            def __init__(self, name):
                self.name = name

            def greet(self):
                return hello(self.name)
    """))

    # Markdown file
    md_file = tmp_path / "README.md"
    md_file.write_text(textwrap.dedent("""\
        # My Project

        This is a project description.

        ## Installation

        Run pip install.

        ## Usage

        Use the greeter class.
    """))

    # Generic text file
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text(textwrap.dedent("""\
        First paragraph of notes.

        Second paragraph of notes.

        Third paragraph of notes.
    """))

    # JavaScript file
    js_dir = tmp_path / "src"
    js_dir.mkdir()
    js_file = js_dir / "app.js"
    js_file.write_text("function main() { console.log('hello'); }\n")

    # File to exclude
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    test_file = test_dir / "test_main.py"
    test_file.write_text("def test_hello(): pass\n")

    # Hidden dir (should be skipped)
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "secret.py").write_text("SECRET = 'hidden'\n")

    # node_modules (should be skipped)
    nm = tmp_path / "node_modules"
    nm.mkdir()
    (nm / "pkg.js").write_text("module.exports = {};\n")

    return tmp_path


# ---------------------------------------------------------------------------
# Chunking unit tests
# ---------------------------------------------------------------------------


class TestChunkPython:
    def test_functions_split_at_def_boundaries(self):
        code = textwrap.dedent("""\
            def foo():
                return 1

            def bar():
                return 2
        """)
        chunks = chunk_python(code, "test.py")
        func_chunks = [c for c in chunks if c["chunk_type"] == "function"]
        assert len(func_chunks) == 2
        names = {c["chunk_name"] for c in func_chunks}
        assert "foo" in names
        assert "bar" in names

    def test_classes_extracted(self):
        code = textwrap.dedent("""\
            class MyClass:
                def method(self):
                    pass
        """)
        chunks = chunk_python(code, "test.py")
        class_chunks = [c for c in chunks if c["chunk_type"] == "class"]
        assert len(class_chunks) == 1
        assert class_chunks[0]["chunk_name"] == "MyClass"

    def test_parse_error_fallback_to_generic(self):
        bad_code = "def incomplete(:\n    pass\n"
        chunks = chunk_python(bad_code, "test.py")
        # Should not raise — falls back to generic
        assert len(chunks) >= 1
        assert chunks[0]["chunk_type"] == "paragraph"


class TestChunkMarkdown:
    def test_sections_split_at_headings(self):
        md = textwrap.dedent("""\
            # Title

            Intro text.

            ## Section One

            Content one.

            ## Section Two

            Content two.
        """)
        chunks = chunk_markdown(md, "test.md")
        section_chunks = [c for c in chunks if c["chunk_type"] == "section"]
        assert len(section_chunks) >= 3
        names = {c["chunk_name"] for c in section_chunks}
        assert "Title" in names
        assert "Section One" in names
        assert "Section Two" in names


class TestChunkGeneric:
    def test_paragraphs_split_at_blank_lines(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.\n"
        chunks = chunk_generic(text, "test.txt")
        assert len(chunks) >= 1
        # All content should be captured
        all_content = " ".join(c["content"] for c in chunks)
        assert "First paragraph" in all_content
        assert "Third paragraph" in all_content


class TestMaxChunkSize:
    def test_chunks_respect_size_limit(self):
        # Create content larger than the chunk size
        content = "\n\n".join(f"Paragraph {i} with some text." for i in range(50))
        chunks = chunk_generic(content, "test.txt", max_chunk_size=200)
        for c in chunks:
            # Single paragraphs that exceed the limit are allowed
            # but grouped paragraphs should respect the limit
            assert len(c["content"]) <= 200 or "\n\n" not in c["content"]


# ---------------------------------------------------------------------------
# CorpusIngestor integration tests
# ---------------------------------------------------------------------------


class TestIngestRepo:
    def test_creates_raw_entries(self, k, sample_repo):
        ingestor = CorpusIngestor(k)
        result = ingestor.ingest_repo(str(sample_repo))

        assert result.files_scanned > 0
        assert result.chunks_created > 0
        assert result.raw_entries_created > 0
        assert not result.errors

    def test_extension_filter(self, k, sample_repo):
        ingestor = CorpusIngestor(k)
        result = ingestor.ingest_repo(str(sample_repo), extensions=["py"])

        # Should only scan .py files
        assert result.files_scanned >= 1
        # Check that raw entries contain Python content
        raw_entries = k._storage.list_raw(limit=1000)
        for entry in raw_entries:
            assert "[corpus:repo]" in entry.blob

    def test_exclude_patterns(self, k, sample_repo):
        ingestor = CorpusIngestor(k)
        ingestor.ingest_repo(
            str(sample_repo),
            extensions=["py"],
            exclude=["tests/*"],
        )

        # Check that test files were excluded
        raw_entries = k._storage.list_raw(limit=1000)
        for entry in raw_entries:
            assert "test_main" not in (entry.blob or "")


class TestIngestDocs:
    def test_creates_raw_entries(self, k, sample_repo):
        ingestor = CorpusIngestor(k)
        result = ingestor.ingest_docs(str(sample_repo))

        assert result.files_scanned > 0
        assert result.chunks_created > 0
        assert result.raw_entries_created > 0


class TestDedup:
    def test_dedup_across_runs(self, k, sample_repo):
        ingestor = CorpusIngestor(k)

        # First run
        result1 = ingestor.ingest_repo(str(sample_repo), extensions=["py"])
        assert result1.raw_entries_created > 0

        # Second run with a fresh ingestor (simulates a new session)
        ingestor2 = CorpusIngestor(k)
        result2 = ingestor2.ingest_repo(str(sample_repo), extensions=["py"])

        # Second run should skip everything
        assert result2.chunks_skipped > 0
        assert result2.raw_entries_created == 0


class TestSourceMetadata:
    def test_blob_has_corpus_header(self, k, sample_repo):
        ingestor = CorpusIngestor(k)
        ingestor.ingest_repo(str(sample_repo), extensions=["py"])

        raw_entries = k._storage.list_raw(limit=1000)
        assert len(raw_entries) > 0

        for entry in raw_entries:
            blob = entry.blob or ""
            assert blob.startswith("[corpus:repo]")
            assert "[file:" in blob
            assert "[chunk:" in blob


class TestStatus:
    def test_status_counts(self, k, sample_repo):
        ingestor = CorpusIngestor(k)
        ingestor.ingest_repo(str(sample_repo), extensions=["py"])

        status = ingestor.get_status()
        assert status["total_corpus_entries"] > 0
        assert status["repo_entries"] > 0
        assert status["docs_entries"] == 0


class TestEdgeCases:
    def test_empty_directory(self, k, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        ingestor = CorpusIngestor(k)
        result = ingestor.ingest_repo(str(empty_dir))

        assert result.files_scanned == 0
        assert result.chunks_created == 0
        assert result.raw_entries_created == 0
        assert not result.errors

    def test_nonexistent_directory(self, k):
        ingestor = CorpusIngestor(k)
        result = ingestor.ingest_repo("/nonexistent/path/to/repo")

        assert result.files_scanned == 0
        assert len(result.errors) == 1
        assert "not a directory" in result.errors[0]

    def test_dry_run_no_writes(self, k, sample_repo):
        ingestor = CorpusIngestor(k)
        result = ingestor.ingest_repo(str(sample_repo), extensions=["py"], dry_run=True)

        assert result.files_scanned > 0
        assert result.chunks_created > 0
        assert result.raw_entries_created == 0

        # Verify nothing was actually written
        raw_entries = k._storage.list_raw(limit=1000)
        corpus_entries = [e for e in raw_entries if (e.blob or "").startswith("[corpus:")]
        assert len(corpus_entries) == 0

    def test_hidden_dirs_skipped(self, k, sample_repo):
        ingestor = CorpusIngestor(k)
        ingestor.ingest_repo(str(sample_repo), extensions=["py"])

        raw_entries = k._storage.list_raw(limit=1000)
        for entry in raw_entries:
            assert ".hidden" not in (entry.blob or "")
            assert "SECRET" not in (entry.blob or "")

    def test_node_modules_skipped(self, k, sample_repo):
        ingestor = CorpusIngestor(k)
        ingestor.ingest_repo(str(sample_repo))

        raw_entries = k._storage.list_raw(limit=1000)
        for entry in raw_entries:
            assert "node_modules" not in (entry.blob or "")
