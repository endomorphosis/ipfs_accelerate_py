"""Targeted native-symbol reads retain canonical generation and edit fences."""
import ast
import copy

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_session as sessions
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_view as views
from ipfs_accelerate_py.agent_supervisor.runtime.code_source_view import SourceViewError
from test.api.test_repository_source_session import RecordingIndex, digest, source


class SymbolIndex(RecordingIndex):
    """Substitute only native indexing; all source capture and writes are real."""
    def __init__(self, state_dir):
        super().__init__(state_dir)
        self.lookup_hook = None
        self.transform = None
        self.lookups = []

    def refresh(self, materialized_root, manifest):
        receipt = super().refresh(materialized_root, manifest)
        self.rows = {}
        for path, entry in manifest["entries"].items():
            if entry["kind"] != "file" or not path.endswith(".py"):
                continue
            tree = ast.parse((materialized_root / path).read_bytes())
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                identity = "native-" + node.name
                kind = ("class" if isinstance(node, ast.ClassDef)
                        else "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function")
                self.rows[identity] = dict(symbol_id=identity, path=path, name="display.name.is.not.a.selector",
                    kind=kind, source_sha256=entry["original_sha256"], span=dict(
                        schema="ipfs-datasets.software-contracts.semantic-source-span@1", path=path,
                        start_line=node.lineno, start_column=node.col_offset,
                        end_line=node.end_lineno, end_column=node.end_col_offset))
        return receipt

    def symbol(self, *, symbol_id):
        self.lookups.append(symbol_id)
        row = copy.deepcopy(self.rows[symbol_id])
        if self.transform is not None:
            row = self.transform(row)
        if self.lookup_hook is not None:
            self.lookup_hook()
        return row


def make(source, tmp_path, **options):
    return sessions.RepositorySourceSession(source, tmp_path / "state",
        index_factory=SymbolIndex, **options)


def refused(code, operation):
    with pytest.raises((sessions.RepositorySourceSessionError, views.RepositorySourceViewError,
                        SourceViewError)) as caught:
        operation()
    assert caught.value.code == code


def test_targeted_read_uses_retained_source_without_importing_or_executing(source, tmp_path):
    target = source / "calc.py"
    target.write_text("raise RuntimeError('module execution forbidden')\n\n" + target.read_text())
    before = target.read_bytes()
    with make(source, tmp_path) as session:
        selected = session.read(generation=1, path="calc.py", symbol_id="native-add")
        assert selected["generation"] == 1 and selected["symbol_id"] == "native-add"
        assert selected["symbol"] == "add" and selected["scope"] == "python_definition"
        assert selected["edit_supported"] is True and selected["representation"] == "compact"
        assert selected["content"].startswith("def add(value):")
        assert "RuntimeError" not in selected["content"] and "def other" not in selected["content"]
        assert selected["original_sha256"] == digest(before)
        assert all(selected[key] is False for key in ("proof_authority", "execution_authority", "completion_authority"))
        assert session._index.lookups == ["native-add"]
        original = session.read(generation=1, path="calc.py", representation="original")
        assert original["content"].encode() == before
        assert target.read_bytes() == before


def test_wrong_path_is_a_bounded_refusal_and_whole_file_fallback_stays_available(source, tmp_path):
    with make(source, tmp_path) as session:
        refused("symbol_path_differs", lambda: session.read(generation=1,
            path="opaque.bin", symbol_id="native-add"))
        assert session.status()["source_current"] is True
        assert session.status()["refresh_required"] is False
        original = session.read(generation=1, path="calc.py", representation="original")
        assert original["content"].encode() == (source / "calc.py").read_bytes()


@pytest.mark.parametrize("source_text,target", [
    ("def outer():\n    def inner():\n        return 1\n    return inner\n", "inner"),
    ("if active:\n    def conditional():\n        return 1\n", "conditional"),
])
def test_unsupported_definition_keeps_generation_available_for_original_fallback(
        source, tmp_path, source_text, target):
    (source / "calc.py").write_text(source_text)
    with make(source, tmp_path) as session:
        refused("unsupported_or_missing_symbol", lambda: session.read(generation=1,
            path="calc.py", symbol_id="native-" + target))
        assert session.generation == 1 and session.status()["source_current"] is True
        assert session.read(generation=1, path="calc.py", representation="original")["content"] == source_text


@pytest.mark.parametrize("field,replacement", [
    ("source_sha256", "0" * 64), ("symbol_id", "native-other"),
])
def test_index_source_or_requested_identity_mismatch_fences_until_refresh(source, tmp_path, field, replacement):
    before = (source / "calc.py").read_bytes()
    with make(source, tmp_path) as session:
        session._index.transform = lambda row: {**row, field: replacement}
        refused("indexed_symbol_source_differs", lambda: session.read(
            generation=1, path="calc.py", symbol_id="native-add"))
        assert session.status()["refresh_required"] is True
        refused("refresh_required", lambda: session.read(generation=1, path="calc.py"))
        assert (source / "calc.py").read_bytes() == before
        session._index.transform = None
        refreshed = session.refresh(generation=1)
        assert refreshed["generation"] == 2 and refreshed["source_current"] is True
        current = session.read(generation=2, path="calc.py", symbol_id="native-add")
        assert current["original_sha256"] == digest(before)


def test_external_file_change_during_native_lookup_prevents_targeted_delivery(source, tmp_path):
    before = (source / "calc.py").read_bytes()
    with make(source, tmp_path) as session:
        session._index.lookup_hook = lambda: (source / "opaque.bin").write_bytes(b"outside pressure changed another file")
        refused("stale_source", lambda: session.read(generation=1,
            path="calc.py", symbol_id="native-add"))
        assert session.status()["refresh_required"] is True
        assert (source / "calc.py").read_bytes() == before
        session._index.lookup_hook = None
        session.refresh(generation=1)
        assert session.read(generation=2, path="calc.py", symbol_id="native-add")["generation"] == 2


def test_explicit_original_definition_noop_preserves_format_under_compact_default(source, tmp_path):
    before = (source / "calc.py").read_bytes()
    with make(source, tmp_path, representation="compact") as session:
        selected = session.read(generation=1, path="calc.py", symbol_id="native-add", representation="original")
        assert selected["representation"] == "original" and "implementation commentary" in selected["content"]
        edited = session.replace_symbol(generation=selected["generation"], path=selected["path"],
            symbol=selected["symbol"], expected_sha256=selected["original_sha256"],
            replacement=selected["content"], representation=selected["representation"])
        assert edited["canonical_changed"] is False and edited["edit"]["semantic_no_op"] is True
        assert edited["generation"] == 2 and edited["refresh_required"] is False
        assert (source / "calc.py").read_bytes() == before
        refused("stale_generation", lambda: session.read(generation=1, path="calc.py", symbol_id="native-add"))


def test_targeted_compact_read_edits_canonical_source_and_refreshes_hash(source, tmp_path):
    before = (source / "calc.py").read_bytes()
    with make(source, tmp_path) as session:
        selected = session.read(generation=1, path="calc.py", symbol_id="native-add")
        edited = session.replace_symbol(generation=1, path=selected["path"], symbol=selected["symbol"],
            expected_sha256=selected["original_sha256"], replacement="def add(value):\n    return value + 17\n",
            representation=selected["representation"])
        assert edited["canonical_changed"] is True and edited["generation"] == 2
        after = (source / "calc.py").read_bytes()
        assert b"return value + 17" in after
        assert before.split(b"# This suffix")[1] == after.split(b"# This suffix")[1]
        current = session.read(generation=2, path="calc.py", symbol_id="native-add")
        assert current["original_sha256"] == digest(after) != selected["original_sha256"]
        assert current["symbol"] == selected["symbol"]


def test_generation_capacity_refuses_targeted_edit_before_mutation(source, tmp_path):
    before = (source / "calc.py").read_bytes()
    with make(source, tmp_path, max_generations=1) as session:
        selected = session.read(generation=1, path="calc.py", symbol_id="native-add")
        refused("generation_limit", lambda: session.replace_symbol(generation=1, path=selected["path"],
            symbol=selected["symbol"], expected_sha256=selected["original_sha256"],
            replacement="def add(value): return 17", representation=selected["representation"]))
        assert (source / "calc.py").read_bytes() == before
        assert session.generation == 1 and session.status()["source_current"] is True
        assert len(session._index.calls) == 1 and not list(session.state_dir.glob("proposal-*"))
