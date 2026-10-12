"""Canonical editing, generation fences and bounded session lifecycle."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_session as sessions
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_view as views


def digest(data):
    return hashlib.sha256(data).hexdigest()


class RecordingIndex:
    """Inject only the native adapter: capture, restore, reads and edits are real."""
    def __init__(self, state_dir):
        state_dir.mkdir(mode=0o700)
        self.closed = False
        self.fail = False
        self.calls = []
        self.current = None

    def refresh(self, materialized_root, manifest):
        self.current = None
        self.calls.append((materialized_root, copy.deepcopy(manifest)))
        if self.fail:
            raise RuntimeError("authored index failure")
        for path, entry in manifest["entries"].items():
            if entry["kind"] == "file":
                assert digest((materialized_root / path).read_bytes()) == entry["original_sha256"]
        self.current = manifest
        encoded = json.dumps(manifest, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode()
        return dict(head={"generation": len(self.calls)}, semantic_state_cid="fixture-state",
                    package_manifest_sha256=digest(encoded), source_tree_sha256=manifest["source_tree_sha256"],
                    coverage={"fixture": True}, source_scope_verified=True)

    def symbols(self, path=None, query="", limit=100):
        assert self.current is not None
        return [{"path": name, "source_sha256": entry["original_sha256"]}
                for name, entry in self.current["entries"].items()
                if entry["kind"] == "file" and (path is None or path == name)
                and query in name][:limit]

    def close(self):
        self.closed = True


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("not indexed")
    (root / "calc.py").write_text(
        "# This source comment is kept in the original but omitted in the compact view.\n"
        "def add(value):\n"
        "    # Deliberately long implementation commentary for a useful compact view.\n"
        "    return value + 1\n\n"
        "# This suffix stays untouched by a replacement.\n"
        "def other():\n"
        "    return 9\n")
    (root / "opaque.bin").write_bytes(b"\xff\x00\x10")
    (root / "empty").mkdir()
    (root / "link").symlink_to("calc.py")
    return root


def make(source, tmp_path, **kwargs):
    return sessions.RepositorySourceSession(source, tmp_path / "state",
                                            index_factory=RecordingIndex, **kwargs)


def assert_error(code, action):
    with pytest.raises((sessions.RepositorySourceSessionError, views.RepositorySourceViewError)) as caught:
        action()
    assert caught.value.code == code


def test_original_materialization_compact_read_and_generation_identity(source, tmp_path):
    with make(source, tmp_path) as session:
        status = session.status()
        assert status["generation"] == 1
        assert status["source_current"] is True
        assert status["proof_authority"] is False
        assert status["execution_authority"] is False
        assert status["completion_authority"] is False
        assert status["capture_counts"]["compacted_files"] == 1
        compact = session.read(generation=1, path="calc.py")
        original = session.read(generation=1, path="calc.py", representation="original")
        assert original["content"].encode() == (source / "calc.py").read_bytes()
        assert original["representation"] == "original"
        assert "implementation commentary" not in compact["content"]
        assert compact["view_bytes"] < original["view_bytes"]
        assert compact["source_tree_sha256"] == status["source_tree_sha256"]
        assert compact["package_manifest_sha256"] == status["package_manifest_sha256"]
        materialized = session._index.calls[0][0]
        assert not (materialized / ".git").exists()
        assert (materialized / "calc.py").read_bytes() == (source / "calc.py").read_bytes()
        assert session.read(generation=1, path="opaque.bin")["encoding"] == "base64"
    assert session._index.closed
    assert (tmp_path / "state").exists()  # Caller owns evidence cleanup.
    assert_error("session_closed", session.status)


def test_listing_pagination_and_native_symbol_routing(source, tmp_path):
    with make(source, tmp_path) as session:
        first = session.list_files(generation=1, limit=2)
        second = session.list_files(generation=1, offset=first["next_offset"], limit=2)
        assert [row["path"] for row in first["entries"] + second["entries"]] == [
            "calc.py", "empty", "link", "opaque.bin"]
        assert second["next_offset"] is None
        rows = session.symbols(generation=1, query="calc", limit=1)["symbols"]
        assert rows == [{"path": "calc.py", "source_sha256": digest((source / "calc.py").read_bytes())}]
        assert session.symbols(generation=1, path="opaque.bin")["symbols"][0]["path"] == "opaque.bin"
        assert_error("not_regular_file", lambda: session.read(generation=1, path="link"))


@pytest.mark.parametrize("change", ["edit", "new", "delete", "mode", "directory", "symlink"])
def test_full_source_changes_require_refresh(source, tmp_path, change):
    with make(source, tmp_path) as session:
        previous = session.status()["package_manifest_sha256"]
        if change == "edit":
            (source / "calc.py").write_text("def changed():\n    return 5\n")
        elif change == "new":
            (source / "new.py").write_text("new = 7\n")
        elif change == "delete":
            (source / "opaque.bin").unlink()
        elif change == "mode":
            (source / "calc.py").chmod(0o600)
        elif change == "directory":
            (source / "new_dir").mkdir()
        else:
            (source / "link").unlink()
            (source / "link").symlink_to("opaque.bin")
        assert_error("stale_source", lambda: session.read(generation=1, path="calc.py"))
        assert session.status()["refresh_required"] is True
        assert_error("refresh_required", lambda: session.symbols(generation=1))
        current = session.refresh(generation=1)
        assert current["generation"] == 2
        assert current["previous_package_manifest_sha256"] == previous
        assert current["source_current"] is True
        assert_error("stale_generation", lambda: session.read(generation=1, path="calc.py"))
        assert session.read(generation=2, path="calc.py")["generation"] == 2
        if change == "new":
            assert session.read(generation=2, path="new.py")["content"] == "new = 7\n"
        elif change == "delete":
            assert_error("not_regular_file", lambda: session.read(generation=2, path="opaque.bin"))


def test_canonical_edit_refreshes_index_and_preserves_unselected_bytes(source, tmp_path):
    with make(source, tmp_path) as session:
        before = (source / "calc.py").read_bytes()
        result = session.replace_symbol(generation=1, path="calc.py", symbol="add",
                                        expected_sha256=digest(before), replacement="def add(value):\n    return value + 5")
        assert result["canonical_changed"] is True
        assert result["refresh_required"] is False
        assert result["generation"] == 2
        after = (source / "calc.py").read_bytes()
        assert b"return value + 5" in after
        assert before.split(b"# This suffix")[1] == after.split(b"# This suffix")[1]
        assert len(session._index.calls) == 2
        assert session.read(generation=2, path="calc.py")["original_sha256"] == digest(after)
        assert_error("stale_generation", lambda: session.replace_symbol(
            generation=1, path="calc.py", symbol="add", expected_sha256=digest(before), replacement="def add(value): return 2"))


def test_compact_digest_cannot_authorize_canonical_write(source, tmp_path):
    with make(source, tmp_path) as session:
        compact = session.read(generation=1, path="calc.py")
        assert compact["compact_sha256"] != compact["original_sha256"]
        before = (source / "calc.py").read_bytes()
        assert_error("stale_source_hash", lambda: session.replace_symbol(
            generation=1, path="calc.py", symbol="add", expected_sha256=compact["compact_sha256"],
            replacement="def add(value): return 7"))
        assert (source / "calc.py").read_bytes() == before


def test_comment_only_proposal_is_not_silently_discarded(source, tmp_path):
    with make(source, tmp_path) as session:
        before = (source / "calc.py").read_bytes()
        assert_error("ambiguous_symbol_noop_use_original_edit", lambda: session.replace_symbol(
            generation=1, path="calc.py", symbol="add", expected_sha256=digest(before),
            replacement="def add(value):\n    # Newly requested comment.\n    return value + 1\n"))
        assert (source / "calc.py").read_bytes() == before
        assert session.status()["source_current"] is True
        assert session.generation == 1


@pytest.mark.parametrize("representation,format", [("compact", "compact"), ("original", "raw")])
def test_exact_delivered_noop_is_explicit_and_preserves_original(source, tmp_path, representation, format):
    with make(source, tmp_path, representation=representation) as session:
        before = (source / "calc.py").read_bytes()
        delivered = sessions.CodeSourceView(source).read("calc.py", "add", format)
        result = session.replace_symbol(generation=1, path="calc.py", symbol="add",
                                        expected_sha256=digest(before), replacement=delivered["content"])
        assert result["canonical_changed"] is False
        assert result["edit"]["semantic_no_op"] is True
        assert result["refresh_required"] is False
        assert (source / "calc.py").read_bytes() == before


def test_post_edit_index_failure_reports_committed_change_and_fences_old_handles(source, tmp_path):
    with make(source, tmp_path) as session:
        session._index.fail = True
        result = session.replace_symbol(generation=1, path="calc.py", symbol="add",
            expected_sha256=digest((source / "calc.py").read_bytes()), replacement="def add(value): return value + 8")
        assert result["canonical_changed"] is True
        assert result["error"] == "index_refresh_failed"
        assert result["generation"] == 1
        assert result["refresh_required"] is True
        assert b"return value + 8" in (source / "calc.py").read_bytes()
        assert_error("refresh_required", lambda: session.read(generation=1, path="calc.py"))
        assert len(list(session.state_dir.glob("generation-*"))) == 1
        session._index.fail = False
        assert session.refresh(generation=1)["generation"] == 2
        assert "return value + 8" in session.read(generation=2, path="calc.py")["content"]


def test_postcommit_editor_error_reports_uncertainty_and_requires_refresh(source, tmp_path, monkeypatch):
    with make(source, tmp_path) as session:
        original_replace = sessions.CodeSourceView.replace
        def failing_after_replace(editor, *args, **kwargs):
            result = original_replace(editor, *args, **kwargs)
            if editor.root == source:
                raise OSError("authored postcommit fsync failure")
            return result
        monkeypatch.setattr(sessions.CodeSourceView, "replace", failing_after_replace)
        result = session.replace_symbol(generation=1, path="calc.py", symbol="add",
            expected_sha256=digest((source / "calc.py").read_bytes()), replacement="def add(value): return 17")
        assert result["canonical_changed"] is None
        assert result["canonical_change"] == "uncertain"
        assert result["refresh_required"] is True
        assert b"return 17" in (source / "calc.py").read_bytes()
        assert_error("refresh_required", lambda: session.list_files(generation=1))
        assert session.refresh(generation=1)["generation"] == 2


def test_failed_editor_without_source_change_does_not_fence_session(source, tmp_path):
    with make(source, tmp_path) as session:
        with pytest.raises(sessions.SourceViewError, match="replacement_symbol_mismatch"):
            session.replace_symbol(generation=1, path="calc.py", symbol="add",
                expected_sha256=digest((source / "calc.py").read_bytes()), replacement="def wrong(): return 1")
        assert session.status()["source_current"] is True
        assert session.generation == 1


def test_source_mutation_during_read_is_rejected(source, tmp_path, monkeypatch):
    with make(source, tmp_path) as session:
        original_read = views.read_repository_file
        def change_during_read(*args, **kwargs):
            result = original_read(*args, **kwargs)
            (source / "opaque.bin").write_bytes(b"external change")
            return result
        monkeypatch.setattr(views, "read_repository_file", change_during_read)
        assert_error("stale_source", lambda: session.read(generation=1, path="calc.py"))


def test_native_query_failure_fences_the_session(source, tmp_path, monkeypatch):
    with make(source, tmp_path) as session:
        def failed_query(**_):
            raise RuntimeError("authored query failure")
        monkeypatch.setattr(session._index, "symbols", failed_query)
        with pytest.raises(RuntimeError, match="authored query failure"):
            session.symbols(generation=1)
        assert session.status()["refresh_required"] is True
        assert_error("refresh_required", lambda: session.read(generation=1, path="calc.py"))
        assert session.refresh(generation=1)["source_current"] is True


def test_initial_index_failure_closes_owner_and_removes_candidate(source, tmp_path):
    instances = []
    def failing_factory(path):
        instance = RecordingIndex(path)
        instance.fail = True
        instances.append(instance)
        return instance
    with pytest.raises(RuntimeError, match="authored index failure"):
        sessions.RepositorySourceSession(source, tmp_path / "state", index_factory=failing_factory)
    assert instances[0].closed
    assert not list((tmp_path / "state").glob("generation-*"))


def test_index_receipt_must_join_exact_manifest(source, tmp_path):
    class WrongIndex(RecordingIndex):
        def refresh(self, *args):
            receipt = super().refresh(*args)
            receipt["source_tree_sha256"] = "0" * 64
            return receipt
    assert_error("invalid_index_receipt", lambda: sessions.RepositorySourceSession(
        source, tmp_path / "state", index_factory=WrongIndex))


@pytest.mark.parametrize("limits,code", [({"max_files": 1}, "entry_limit"),
                                         ({"max_file_bytes": 1}, "file_limit"),
                                         ({"max_total_bytes": 1}, "total_bytes_limit")])
def test_capture_limits_fail_closed(source, tmp_path, limits, code):
    assert_error(code, lambda: make(source, tmp_path, **limits))
    assert not list((tmp_path / "state").glob("generation-*"))


def test_generation_limit_refuses_edit_before_canonical_mutation(source, tmp_path):
    with make(source, tmp_path, max_generations=1) as session:
        before = (source / "calc.py").read_bytes()
        assert_error("generation_limit", lambda: session.replace_symbol(
            generation=1, path="calc.py", symbol="add", expected_sha256=digest(before), replacement="def add(value): return 20"))
        assert (source / "calc.py").read_bytes() == before
        assert session.status()["source_current"] is True


@pytest.mark.parametrize("bound,code", [("file", "file_limit"), ("total", "total_bytes_limit")])
def test_edit_size_preflight_preserves_canonical_source(source, tmp_path, bound, code):
    before = (source / "calc.py").read_bytes()
    maximum = len(before) + (3 if bound == "total" else 0)
    limits = {"max_file_bytes" if bound == "file" else "max_total_bytes": maximum}
    with make(source, tmp_path, **limits) as session:
        # The replacement text fits the input limit; retained, unselected bytes
        # make the complete resulting file exceed the source reservation.
        replacement = "def add(value):\n    return '" + "x" * 230 + "'\n"
        assert len(replacement.encode()) <= session.limits["max_file_bytes"]
        assert_error(code, lambda: session.replace_symbol(generation=1, path="calc.py", symbol="add",
            expected_sha256=digest(before), replacement=replacement))
        assert (source / "calc.py").read_bytes() == before
        assert session.status()["source_current"] is True
        assert not list(session.state_dir.glob("proposal-*"))


@pytest.mark.parametrize("case,code", [("git", "state_inside_git"), ("nonempty", "state_not_empty"),
                                       ("public", "state_not_private"), ("overlap", "overlapping_locations")])
def test_unsafe_state_is_refused(source, tmp_path, case, code):
    state = tmp_path / "state"
    if case == "git":
        (tmp_path / ".git").write_text("gitdir: elsewhere")
    elif case == "nonempty":
        state.mkdir(mode=0o700)
        (state / "keep").write_text("caller data")
    elif case == "public":
        state.mkdir(mode=0o755)
    else:
        state = source / "nested-state"
    assert_error(code, lambda: sessions.RepositorySourceSession(source, state, index_factory=RecordingIndex))
    if case == "nonempty":
        assert (state / "keep").read_text() == "caller data"


def test_state_directory_replacement_is_rejected(source, tmp_path):
    with make(source, tmp_path) as session:
        session.state_dir.rename(tmp_path / "retained-state")
        session.state_dir.mkdir(mode=0o700)
        assert_error("state_replaced", session.status)


def test_status_receipts_cannot_mutate_internal_generation_state(source, tmp_path):
    with make(source, tmp_path) as session:
        status = session.status()
        status["native_index"]["head"]["generation"] = -1
        status["datasets_head"]["generation"] = -2
        assert session.status()["datasets_head"]["generation"] == 1


@pytest.mark.parametrize("generation", [True, 0, -1, "1", None])
def test_invalid_generation_handles_are_rejected(source, tmp_path, generation):
    with make(source, tmp_path) as session:
        assert_error("invalid_generation", lambda: session.read(generation=generation, path="calc.py"))
