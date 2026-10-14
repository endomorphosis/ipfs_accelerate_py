"""Authored repository revision cycles: immutable parents and canonical edits."""
import hashlib
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_revision as revision
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_view as engine
from ipfs_accelerate_py.agent_supervisor.runtime.code_source_view import CodeSourceView


def sha(data):
    return hashlib.sha256(data).hexdigest()


def tree(root):
    result = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        info = path.lstat()
        if path.is_symlink():
            result[rel] = ("link", os.readlink(path))
        elif path.is_dir():
            result[rel] = ("dir", stat.S_IMODE(info.st_mode))
        else:
            result[rel] = ("file", stat.S_IMODE(info.st_mode), path.read_bytes())
    return result


@pytest.fixture
def snapshot(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "calc.py").write_bytes(
        b"# Copyright and comments are retained in original source.\n"
        b"OFFSET = 2\n\n"
        b"def add(value):\n"
        b"    # This implementation comment is absent in the compact view.\n"
        b"    return value + OFFSET\n\n"
        b"# This suffix remains byte identical.\n"
        b"def untouched(): return 'kept'\n")
    (source / "consumer.py").write_text("from calc import add\nprint(add(3))\n")
    (source / "notes.txt").write_text("original notes\n")
    (source / "empty").mkdir()
    (source / "notes.txt").chmod(0o640)
    package = tmp_path / "parent"
    receipt = engine.create_repository_view(source, package)
    return source, package, receipt["manifest_sha256"]


def op(snapshot, kind="replace_symbol", **changes):
    source, _package, _digest = snapshot
    result = {
        "operation": kind,
        "path": "calc.py",
        "expected_sha256": sha((source / "calc.py").read_bytes()),
    }
    if kind == "replace_symbol":
        result.update(symbol="add", replacement="def add(value):\n    return value + OFFSET + 10\n")
    result.update(changes)
    return result


def run_revision(snapshot, tmp_path, operations):
    _source, package, digest = snapshot
    output = tmp_path / "successor"
    receipt = revision.revise_repository_view(package, output,
        expected_manifest_sha256=digest, operations=operations)
    restored = tmp_path / "restored"
    engine.restore_repository_view(output, restored,
        expected_manifest_sha256=receipt["successor_manifest_sha256"])
    return receipt, output, restored


def failure(snapshot, tmp_path, operations, code=None):
    source, package, digest = snapshot
    before_source, before_package = tree(source), tree(package)
    with pytest.raises(ValueError) as caught:
        revision.revise_repository_view(package, tmp_path / "successor",
            expected_manifest_sha256=digest, operations=operations)
    if code:
        assert caught.value.code == code
    assert not (tmp_path / "successor").exists()
    assert tree(source) == before_source
    assert tree(package) == before_package
    assert not list(tmp_path.glob("repository-source-revision-*"))


def test_symbol_edit_restores_real_cross_file_behavior(snapshot, tmp_path):
    source, package, digest = snapshot
    original_source, original_package = tree(source), tree(package)
    original = (source / "calc.py").read_bytes()
    receipt, output, restored = run_revision(snapshot, tmp_path, [op(snapshot)])
    changed = (restored / "calc.py").read_bytes()
    assert changed.startswith(original[:original.index(b"def add")])
    suffix = original[original.index(b"\n\n# This suffix"):]
    assert changed.endswith(suffix)
    assert (restored / "consumer.py").read_bytes() == (source / "consumer.py").read_bytes()
    completed = subprocess.run([sys.executable, "consumer.py"], cwd=restored,
        capture_output=True, check=True, timeout=10)
    assert completed.stdout == b"15\n"
    assert tree(source) == original_source and tree(package) == original_package
    assert receipt["parent_manifest_sha256"] == digest
    assert receipt["successor_manifest_sha256"] != digest
    assert receipt["unchanged_files"] == 2 and receipt["unchanged_files_preserved"]
    assert receipt["operations"][0]["input_kind"] == "compact_symbol"
    assert receipt["operations"][0]["untouched_bytes_preserved"]
    for key in ("behavioral_equivalence_verified", "proof_authority", "task_completion_authority",
                "token_savings_verified", "source_checkout_mutated", "parent_package_mutated",
                "syntax_validation_executes_source"):
        assert receipt[key] is False
    assert not list(tmp_path.glob("repository-source-revision-*"))


def test_exact_compact_noop_retains_original_comments(snapshot, tmp_path):
    source, _package, _digest = snapshot
    compact = CodeSourceView(source).read("calc.py", "add", "compact")["content"]
    receipt, _output, restored = run_revision(snapshot, tmp_path, [op(snapshot, replacement=compact)])
    assert tree(restored) == tree(source)
    assert receipt["operations"][0]["semantic_no_op"] is True
    assert receipt["operations"][0]["changed"] is False
    assert receipt["parent_source_tree_sha256"] == receipt["successor_source_tree_sha256"]


@pytest.mark.parametrize("replacement", [
    "def add(value):\n    # This deliberately added comment must not disappear.\n    return value + OFFSET\n",
    "def add(value): return value + OFFSET",
])
def test_same_ast_nonexact_symbol_proposal_requires_full_source(snapshot, tmp_path, replacement):
    failure(snapshot, tmp_path, [op(snapshot, replacement=replacement)], "ambiguous_symbol_noop_use_write_file")


def test_full_original_write_supports_comment_only_edit_and_mode(snapshot, tmp_path):
    source, _package, _digest = snapshot
    content = (source / "calc.py").read_text().replace("Copyright", "Revised copyright")
    receipt, _output, restored = run_revision(snapshot, tmp_path, [
        op(snapshot, "write_file", content=content, mode=0o751)])
    assert (restored / "calc.py").read_text() == content
    assert stat.S_IMODE((restored / "calc.py").stat().st_mode) == 0o751
    assert receipt["operations"][0]["input_kind"] == "full_original_source"
    assert receipt["operations"][0]["changed"]


def test_batch_add_delete_rename_and_edit_preserves_behavior(snapshot, tmp_path):
    source, _package, _digest = snapshot
    operations = [
        op(snapshot),
        {"operation": "write_file", "path": "new/helper.py", "expected_sha256": None,
         "content": "VALUE = 7\n"},
        {"operation": "delete_file", "path": "notes.txt", "expected_sha256": sha((source / "notes.txt").read_bytes())},
        {"operation": "rename_file", "path": "consumer.py", "destination": "run.py",
         "expected_sha256": sha((source / "consumer.py").read_bytes())},
    ]
    receipt, _output, restored = run_revision(snapshot, tmp_path, operations)
    assert not (restored / "notes.txt").exists() and not (restored / "consumer.py").exists()
    assert (restored / "new/helper.py").read_text() == "VALUE = 7\n"
    assert stat.S_IMODE((restored / "new/helper.py").stat().st_mode) == 0o644
    completed = subprocess.run([sys.executable, "run.py"], cwd=restored,
        capture_output=True, check=True, timeout=10)
    assert completed.stdout == b"15\n"
    assert receipt["operation_count"] == 4
    assert receipt["touched_paths"] == ["calc.py", "consumer.py", "new/helper.py", "notes.txt", "run.py"]


def test_later_invalid_symbol_rolls_back_entire_batch(snapshot, tmp_path):
    source, _package, _digest = snapshot
    failure(snapshot, tmp_path, [op(snapshot), {
        "operation": "replace_symbol", "path": "consumer.py", "symbol": "missing",
        "replacement": "def missing(): return 1\n",
        "expected_sha256": sha((source / "consumer.py").read_bytes()),
    }], "unsupported_or_missing_symbol")


def test_all_expected_hashes_anchored_parent_before_successor(snapshot, tmp_path):
    failure(snapshot, tmp_path, [op(snapshot), {
        "operation": "delete_file", "path": "notes.txt", "expected_sha256": "0" * 64,
    }], "stale_source")


@pytest.mark.parametrize("path", ["../outside.py", "/tmp/outside.py", "./calc.py", "a//b.py", "a/../b.py",
                                  ".git/config", "a/.git/config", "a/.GIT/config", "a\\b.py", "x\n.py", "", "\ud800.py", "é" * 513 + ".py", "a/" * 64 + "b.py"])
def test_unsafe_paths_rejected(snapshot, tmp_path, path):
    failure(snapshot, tmp_path, [{"operation": "write_file", "path": path,
        "expected_sha256": None, "content": "x = 1\n"}], "invalid_operation_path")


@pytest.mark.parametrize("extra", [{"unknown": True}, {"format": "compact"}, {"execute": True}])
def test_unknown_fields_rejected(snapshot, tmp_path, extra):
    failure(snapshot, tmp_path, [op(snapshot, **extra)], "invalid_operation_fields")


@pytest.mark.parametrize("operations", [[], {}, [None], [{"operation": "shell", "path": "calc.py"}],
                                         [{"operation": "delete_file", "path": "calc.py"}]])
def test_invalid_operation_shapes(snapshot, tmp_path, operations):
    failure(snapshot, tmp_path, operations)


@pytest.mark.parametrize("mode", [-1, True, "644", 0o4755, 0o1000])
def test_unsupported_mode_rejected(snapshot, tmp_path, mode):
    failure(snapshot, tmp_path, [op(snapshot, "write_file", content="x = 1\n", mode=mode)], "invalid_operation_mode")


@pytest.mark.parametrize("content", ["def broken(:", "def duplicate(a, a): pass", "return 1", "\ud800"])
def test_invalid_python_or_utf8_write_does_not_publish(snapshot, tmp_path, content):
    failure(snapshot, tmp_path, [op(snapshot, "write_file", content=content)])


def test_python_writes_are_never_executed(snapshot, tmp_path):
    marker = tmp_path / "executed"
    content = "from pathlib import Path\nPath(" + repr(str(marker)) + ").write_text('bad')\n"
    _receipt, _output, restored = run_revision(snapshot, tmp_path, [op(snapshot, "write_file", content=content)])
    assert (restored / "calc.py").read_text() == content
    assert not marker.exists()


def test_duplicate_file_or_ancestor_touches_rejected(snapshot, tmp_path):
    failure(snapshot, tmp_path, [op(snapshot), op(snapshot)], "conflicting_operations")
    failure(snapshot, tmp_path, [
        {"operation": "write_file", "path": "new", "expected_sha256": None, "content": ""},
        {"operation": "write_file", "path": "new/child", "expected_sha256": None, "content": ""},
    ], "conflicting_operations")


def test_existing_file_cannot_be_parent_of_new_path(snapshot, tmp_path):
    failure(snapshot, tmp_path, [{"operation": "write_file", "path": "notes.txt/child",
        "expected_sha256": None, "content": ""}], "unsupported_operation_topology")


def test_rename_cannot_overwrite_snapshot_entry(snapshot, tmp_path):
    failure(snapshot, tmp_path, [op(snapshot, "rename_file", destination="consumer.py")], "rename_destination_exists")


def test_rename_chain_is_not_implicitly_sequential(snapshot, tmp_path):
    failure(snapshot, tmp_path, [op(snapshot, "rename_file", destination="new.py"), {
        "operation": "rename_file", "path": "new.py", "destination": "newer.py",
        "expected_sha256": op(snapshot)["expected_sha256"],
    }], "conflicting_operations")


def test_hidden_non_git_file_supported_as_explicit_original(snapshot, tmp_path):
    _receipt, _output, restored = run_revision(snapshot, tmp_path, [{
        "operation": "write_file", "path": ".config/settings", "expected_sha256": None, "content": "value=1\n",
    }])
    assert (restored / ".config/settings").read_text() == "value=1\n"


def test_readonly_parent_directory_mode_restored(tmp_path):
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "nested/data.txt").write_text("before\n")
    (source / "nested").chmod(0o500)
    package = tmp_path / "parent"
    original = engine.create_repository_view(source, package)
    receipt, _output, restored = run_revision((source, package, original["manifest_sha256"]), tmp_path, [{
        "operation": "write_file", "path": "nested/data.txt",
        "expected_sha256": sha(b"before\n"), "content": "after\n",
    }])
    assert (restored / "nested/data.txt").read_text() == "after\n"
    assert stat.S_IMODE((restored / "nested").stat().st_mode) == 0o500
    assert receipt["unchanged_topology_preserved"]


def test_safe_untouched_symlink_preserved_and_link_targets_refused(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "file.txt").write_text("before")
    (source / "link.txt").symlink_to("file.txt")
    (source / "folder").mkdir()
    (source / "folder_link").symlink_to("folder", target_is_directory=True)
    package = tmp_path / "parent"
    original = engine.create_repository_view(source, package)
    snapshot = source, package, original["manifest_sha256"]
    for path in ("link.txt", "folder_link/new.txt"):
        failure(snapshot, tmp_path, [{"operation": "write_file", "path": path,
            "expected_sha256": None, "content": "after"}], "unsupported_operation_topology")
    receipt, _output, restored = run_revision(snapshot, tmp_path, [{
        "operation": "write_file", "path": "file.txt", "expected_sha256": sha(b"before"), "content": "after",
    }])
    assert os.readlink(restored / "link.txt") == "file.txt"
    assert os.readlink(restored / "folder_link") == "folder"
    assert receipt["unchanged_symlinks"] == 2


def test_existing_successor_is_never_overwritten(snapshot, tmp_path):
    target = tmp_path / "successor"
    target.mkdir()
    (target / "mine").write_text("keep")
    _source, package, digest = snapshot
    before = tree(target)
    with pytest.raises(revision.RepositoryRevisionError, match="invalid_successor_destination"):
        revision.revise_repository_view(package, target, expected_manifest_sha256=digest, operations=[op(snapshot)])
    assert tree(target) == before


def test_successor_cannot_live_in_source_or_parent(snapshot, tmp_path):
    source, package, digest = snapshot
    for target in (source / "output", package / "output"):
        with pytest.raises(revision.RepositoryRevisionError, match="invalid_successor_destination"):
            revision.revise_repository_view(package, target, expected_manifest_sha256=digest, operations=[op(snapshot)])
        assert not target.exists()


def test_parent_blob_tamper_refused(snapshot, tmp_path):
    _source, package, digest = snapshot
    manifest = engine.load_repository_view(package, expected_manifest_sha256=digest)
    blob = package / "blobs" / manifest["entries"]["notes.txt"]["original_sha256"]
    blob.chmod(0o600)
    blob.write_bytes(b"tampered")
    failure(snapshot, tmp_path, [op(snapshot)])


def test_parent_blob_hardlink_refused(snapshot, tmp_path):
    _source, package, digest = snapshot
    manifest = engine.load_repository_view(package, expected_manifest_sha256=digest)
    blob = package / "blobs" / manifest["entries"]["notes.txt"]["original_sha256"]
    os.link(blob, tmp_path / "linked-blob")
    failure(snapshot, tmp_path, [op(snapshot)])


def test_successor_limit_failure_leaves_no_partial_package(snapshot, tmp_path, monkeypatch):
    _source, package, digest = snapshot
    real_create = revision.create_repository_view
    def reject_total(root, output, **kwargs):
        return real_create(root, output, **{**kwargs, "max_total_bytes": 1})
    monkeypatch.setattr(revision, "create_repository_view", reject_total)
    failure(snapshot, tmp_path, [op(snapshot)])


def test_revision_chain_retains_original_checkout_exclusion(snapshot, tmp_path):
    source, _package, _digest = snapshot
    receipt, output, _restored = run_revision(snapshot, tmp_path, [op(snapshot)])
    manifest = engine.load_repository_view(output, expected_manifest_sha256=receipt["successor_manifest_sha256"])
    assert manifest["origin_source_root"] == str(source)
    with pytest.raises(revision.RepositoryRevisionError, match="invalid_successor_destination"):
        revision.revise_repository_view(output, source / "next-successor",
            expected_manifest_sha256=receipt["successor_manifest_sha256"], operations=[{
                "operation": "delete_file", "path": "notes.txt", "expected_sha256": sha(b"original notes\n"),
            }])
    assert not (source / "next-successor").exists()


def test_operation_count_and_content_bounds(snapshot, tmp_path, monkeypatch):
    failure(snapshot, tmp_path, [op(snapshot)] * (revision.MAX_OPERATIONS + 1), "revision_operation_limit")
    failure(snapshot, tmp_path, [op(snapshot, "write_file", content="x" * (1024 * 1024 + 1))],
            "revision_content_limit")
    monkeypatch.setattr(revision, "MAX_BATCH_TEXT_BYTES", 2)
    failure(snapshot, tmp_path, [op(snapshot, "write_file", content="x=1\n")],
            "revision_batch_content_limit")


def test_rename_non_python_into_python_validates_without_publication(snapshot, tmp_path):
    source, _package, _digest = snapshot
    failure(snapshot, tmp_path, [op(snapshot), {
        "operation": "rename_file", "path": "notes.txt", "destination": "notes.py",
        "expected_sha256": sha((source / "notes.txt").read_bytes()),
    }], "invalid_python")


def test_revision_receipt_retains_post_publication_durability_status(snapshot, tmp_path, monkeypatch):
    real_create = revision.create_repository_view
    def observed_not_fsynced(*args, **kwargs):
        result = real_create(*args, **kwargs)
        result["directory_fsync_complete"] = False
        return result
    monkeypatch.setattr(revision, "create_repository_view", observed_not_fsynced)
    receipt, output, _restored = run_revision(snapshot, tmp_path, [op(snapshot)])
    assert output.exists()
    assert receipt["successor_directory_fsync_complete"] is False
