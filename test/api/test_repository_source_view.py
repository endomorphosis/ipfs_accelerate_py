"""Repository-wide byte fidelity and bounded fail-closed package operations."""
import ast
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_view as views


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


@pytest.fixture
def repository(tmp_path):
    source = tmp_path / "source"
    source.mkdir(mode=0o750)
    (source / "lib").mkdir(mode=0o750)
    (source / "empty").mkdir(mode=0o700)
    (source / ".settings").mkdir()
    (source / "lib" / "calc.py").write_bytes(
        b"# Authored copyright and an intentionally retained comment.\r\n"
        b"VALUE = 2\r\n\r\n"
        b"def add(value):\r\n"
        b"    '''A public docstring survives AST rendering.'''\r\n"
        b"    # Implementation comments are retained in the original blob.\r\n"
        b"    return value + VALUE\r\n")
    (source / "lib" / "calc.py").chmod(0o751)
    (source / "README.txt").write_bytes(b"Keep  exact\ttext\r\n")
    (source / "data.bin").write_bytes(b"\x00\xff\x80\n")
    (source / ".settings" / "config.py").write_bytes(b"# hidden files pass through\nx = 1\n")
    (source / "empty-file").write_bytes(b"")
    (source / ".git").mkdir()
    (source / ".git" / "HEAD").write_bytes(b"excluded")
    (source / "lib" / ".git").write_bytes(b"gitdir: not followed")
    (source / "calc-link").symlink_to("lib/calc.py")
    (source / "dangling").symlink_to("future.txt")
    (source / "lib" / "parent-link").symlink_to("../README.txt")
    return source


def capture(repository, tmp_path, **kwargs):
    package = tmp_path / "package"
    receipt = views.create_repository_view(repository, package, **kwargs)
    return package, receipt


def load(package, receipt):
    return views.load_repository_view(package, expected_manifest_sha256=receipt["manifest_sha256"])


def tree(path):
    result = {"": ("directory", stat.S_IMODE(path.stat().st_mode))}
    for root, directories, files in os.walk(path, followlinks=False):
        directories[:] = [name for name in directories if name != ".git"]
        for name in directories + [name for name in files if name != ".git"]:
            item = Path(root) / name
            relative = item.relative_to(path).as_posix()
            if item.is_symlink():
                result[relative] = ("symlink", os.readlink(item))
            elif item.is_dir():
                result[relative] = ("directory", stat.S_IMODE(item.stat().st_mode))
            else:
                result[relative] = ("file", stat.S_IMODE(item.stat().st_mode), item.read_bytes())
    return result


def replace_manifest(package, manifest):
    encoded = canonical(manifest)
    target = package / "manifest.json"
    target.chmod(0o644)
    target.write_bytes(encoded)
    return digest(encoded)


def assert_error(code, action):
    with pytest.raises(views.RepositorySourceViewError) as caught:
        action()
    assert caught.value.code == code


def test_original_round_trip_without_source_preserves_complete_tree(repository, tmp_path):
    expected = tree(repository)
    package, receipt = capture(repository, tmp_path)
    manifest = load(package, receipt)
    assert manifest["excluded_paths"] == [".git", "lib/.git"]
    assert manifest["scope"]["proof_authority"] is False
    assert manifest["scope"]["behavioral_equivalence_verified"] is False
    assert receipt["compacted_files"] == 1
    assert receipt["compact_bytes"] < receipt["original_bytes"]
    assert receipt["directory_fsync_complete"] is True
    shutil.rmtree(repository)
    restored = tmp_path / "restored"
    result = views.restore_repository_view(package, restored, expected_manifest_sha256=receipt["manifest_sha256"])
    assert tree(restored) == expected
    assert not (restored / ".git").exists()
    assert not (restored / "lib" / ".git").exists()
    assert result["source_tree_sha256"] == receipt["source_tree_sha256"]
    again = views.create_repository_view(restored, tmp_path / "second-package")
    assert again["source_tree_sha256"] == receipt["source_tree_sha256"]


def test_compact_restore_and_individual_reads_are_exact_retained_views(repository, tmp_path):
    package, receipt = capture(repository, tmp_path)
    manifest = load(package, receipt)
    compact = views.read_repository_file(package, expected_manifest_sha256=receipt["manifest_sha256"], path="lib/calc.py")
    original = views.read_repository_file(package, expected_manifest_sha256=receipt["manifest_sha256"], path="lib/calc.py", representation="original")
    assert compact["encoding"] == "utf-8"
    assert "Implementation comments" not in compact["content"]
    assert "A public docstring" in compact["content"]
    assert ast.dump(ast.parse(compact["content"])) == ast.dump(ast.parse(original["content"]))
    assert compact["view_sha256"] == manifest["entries"]["lib/calc.py"]["compact_sha256"]
    restored = tmp_path / "compact"
    views.restore_repository_view(package, restored, expected_manifest_sha256=receipt["manifest_sha256"], representation="compact")
    assert (restored / "lib/calc.py").read_bytes() == compact["content"].encode()
    assert (restored / ".settings/config.py").read_bytes() == (repository / ".settings/config.py").read_bytes()
    assert (restored / "data.bin").read_bytes() == (repository / "data.bin").read_bytes()
    assert stat.S_IMODE((restored / "lib/calc.py").stat().st_mode) == 0o751


def test_binary_read_has_explicit_lossless_encoding(repository, tmp_path):
    package, receipt = capture(repository, tmp_path)
    value = views.read_repository_file(package, expected_manifest_sha256=receipt["manifest_sha256"], path="data.bin")
    assert value["encoding"] == "base64"
    assert base64.b64decode(value["content"]) == (repository / "data.bin").read_bytes()
    assert value["rendering"] == "passthrough"


@pytest.mark.parametrize("data", [b"x=1\n", b"not Python !", b"# coding: latin-1\n# \xe9\nx=1\n", b"\xff"])
def test_unprofitable_or_unsupported_python_passes_through(tmp_path, data):
    source = tmp_path / "source"
    source.mkdir()
    (source / "value.py").write_bytes(data)
    package, receipt = capture(source, tmp_path)
    entry = load(package, receipt)["entries"]["value.py"]
    assert entry["rendering"] == "passthrough"
    assert entry["original_sha256"] == entry["compact_sha256"] == digest(data)


def test_no_renderer_needed_to_restore_retained_original(repository, tmp_path, monkeypatch):
    package, receipt = capture(repository, tmp_path)
    monkeypatch.setattr(views.code_source_view, "render_source_bytes", lambda *a, **k: pytest.fail("renderer called during restore"))
    shutil.rmtree(repository)
    views.restore_repository_view(package, tmp_path / "restored", expected_manifest_sha256=receipt["manifest_sha256"])


def test_empty_tree_round_trip(tmp_path):
    source = tmp_path / "empty"
    source.mkdir()
    package, receipt = capture(source, tmp_path)
    assert receipt["entry_count"] == receipt["original_bytes"] == 0
    views.restore_repository_view(package, tmp_path / "restored", expected_manifest_sha256=receipt["manifest_sha256"])
    assert list((tmp_path / "restored").iterdir()) == []


@pytest.mark.parametrize("kwargs,code", [
    ({"max_files": 2}, "entry_limit"),
    ({"max_file_bytes": 2}, "file_limit"),
    ({"max_total_bytes": 3}, "total_bytes_limit"),
    ({"max_files": 0}, "invalid_limits"),
    ({"max_files": True}, "invalid_limits"),
    ({"max_total_bytes": views.MAX_TOTAL_BYTES + 1}, "invalid_limits"),
])
def test_limits_refuse_instead_of_partial_capture(repository, tmp_path, kwargs, code):
    assert_error(code, lambda: capture(repository, tmp_path, **kwargs))
    assert not (tmp_path / "package").exists()
    assert list(tmp_path.glob(".repository-view-*")) == []


@pytest.mark.parametrize("kind,code", [("hardlink", "unsupported_file"), ("fifo", "unsupported_file"), ("specialmode", "unsupported_permission_bits")])
def test_unsupported_files_are_refused(repository, tmp_path, kind, code):
    if kind == "hardlink":
        os.link(repository / "README.txt", repository / "another")
    elif kind == "fifo":
        os.mkfifo(repository / "pipe")
    else:
        (repository / "README.txt").chmod(0o4755)
    assert_error(code, lambda: capture(repository, tmp_path))
    assert not (tmp_path / "package").exists()


def test_hardlinked_symlinks_are_refused(repository, tmp_path):
    os.link(repository / "calc-link", repository / "same-link", follow_symlinks=False)
    assert (repository / "calc-link").lstat().st_nlink == 2
    assert_error("unsupported_file", lambda: capture(repository, tmp_path))
    assert not (tmp_path / "package").exists()
    assert list(tmp_path.glob(".repository-view-*")) == []


@pytest.mark.parametrize("target", ["/tmp/outside", "../outside", "lib/../../outside"])
def test_symlink_escape_refused(repository, tmp_path, target):
    (repository / "bad-link").symlink_to(target)
    assert_error("unsafe_symlink", lambda: capture(repository, tmp_path))


def test_chained_symlink_escape_refused(repository, tmp_path):
    (repository / "lib" / "up").symlink_to("..")
    (repository / "lib" / "bad").symlink_to("up/../outside")
    assert_error("unsafe_symlink", lambda: capture(repository, tmp_path))


def test_symlink_cycle_refused(repository, tmp_path):
    (repository / "a").symlink_to("b")
    (repository / "b").symlink_to("a")
    assert_error("symlink_cycle_or_depth", lambda: capture(repository, tmp_path))


def test_symlink_root_and_output_parent_are_never_followed(repository, tmp_path):
    alias = tmp_path / "alias"
    alias.symlink_to(repository, target_is_directory=True)
    assert_error("unsafe_or_missing_directory", lambda: views.create_repository_view(alias, tmp_path / "package"))
    parent = tmp_path / "parent"
    parent.mkdir()
    output_alias = tmp_path / "output-alias"
    output_alias.symlink_to(parent, target_is_directory=True)
    assert_error("unsafe_or_missing_directory", lambda: views.create_repository_view(repository, output_alias / "package"))
    assert list(parent.iterdir()) == []


def test_disjoint_and_absent_destinations(repository, tmp_path):
    assert_error("overlapping_locations", lambda: views.create_repository_view(repository, repository / "package"))
    package, receipt = capture(repository, tmp_path)
    assert_error("destination_exists", lambda: views.create_repository_view(repository, package))
    assert_error("overlapping_locations", lambda: views.restore_repository_view(package, package / "out", expected_manifest_sha256=receipt["manifest_sha256"]))
    assert_error("overlapping_locations", lambda: views.restore_repository_view(package, repository / "out", expected_manifest_sha256=receipt["manifest_sha256"]))
    existing = tmp_path / "existing"
    existing.mkdir()
    assert_error("destination_exists", lambda: views.restore_repository_view(package, existing, expected_manifest_sha256=receipt["manifest_sha256"]))
    assert list(existing.iterdir()) == []


def test_origin_anchor_is_retained_and_disjoint(repository, tmp_path):
    origin = tmp_path / "first-source"
    origin.mkdir()
    receipt = views.create_repository_view(repository, tmp_path / "package", origin_source_root=origin)
    manifest = load(tmp_path / "package", receipt)
    assert manifest["source_root"] == str(repository)
    assert manifest["origin_source_root"] == str(origin)
    assert_error("overlapping_locations", lambda: views.restore_repository_view(tmp_path / "package", origin / "out", expected_manifest_sha256=receipt["manifest_sha256"]))
    assert_error("overlapping_locations", lambda: views.create_repository_view(repository, origin / "out", origin_source_root=origin))


def test_full_tree_freshness_detects_edit_after_first_capture(repository, tmp_path, monkeypatch):
    render = views._compact
    # Mutate an already captured file without modifying its parent directory.
    def mutate_captured(path, data):
        if path == ".settings/config.py":
            (repository / path).write_bytes(data + b"# changed\n")
        return render(path, data)
    monkeypatch.setattr(views, "_compact", mutate_captured)
    assert_error("snapshot_changed", lambda: capture(repository, tmp_path))
    assert not (tmp_path / "package").exists()
    assert list(tmp_path.glob(".repository-view-*")) == []


def test_freshness_detects_added_unrelated_file(repository, tmp_path, monkeypatch):
    render = views._compact
    def mutate(path, data):
        (repository / "new.txt").write_bytes(b"new")
        return render(path, data)
    monkeypatch.setattr(views, "_compact", mutate)
    with pytest.raises(views.RepositorySourceViewError) as caught:
        capture(repository, tmp_path)
    assert caught.value.code in {"source_changed_during_read", "snapshot_changed"}
    assert not (tmp_path / "package").exists()


def test_wrong_manifest_pin_and_blob_corruption_refused(repository, tmp_path):
    package, receipt = capture(repository, tmp_path)
    assert_error("manifest_digest_mismatch", lambda: views.load_repository_view(package, expected_manifest_sha256="0" * 64))
    manifest = load(package, receipt)
    entry = manifest["entries"]["README.txt"]
    blob = package / "blobs" / entry["original_sha256"]
    blob.chmod(0o644)
    blob.write_bytes(b"x" * entry["original_bytes"])
    assert_error("blob_digest_mismatch", lambda: load(package, receipt))
    assert_error("blob_digest_mismatch", lambda: views.restore_repository_view(package, tmp_path / "restored", expected_manifest_sha256=receipt["manifest_sha256"]))
    assert not (tmp_path / "restored").exists()


def test_missing_unexpected_and_symlink_blobs_refused(repository, tmp_path):
    package, receipt = capture(repository, tmp_path)
    manifest = load(package, receipt)
    blobdir = package / "blobs"
    blobdir.chmod(0o755)
    blob = blobdir / manifest["entries"]["README.txt"]["original_sha256"]
    data = blob.read_bytes()
    blob.unlink()
    assert_error("missing_blob", lambda: load(package, receipt))
    blob.symlink_to(repository / "README.txt")
    assert_error("invalid_package", lambda: load(package, receipt))
    blob.unlink()
    blob.write_bytes(data)
    (blobdir / "unlisted").write_bytes(b"surprise")
    assert_error("unexpected_blob", lambda: load(package, receipt))


@pytest.mark.parametrize("path", ["../escape", "/absolute", "lib/../escape", "lib//x", "lib\\x", ".git/HEAD", "lib/\x01x"])
def test_read_rejects_invalid_paths(repository, tmp_path, path):
    package, receipt = capture(repository, tmp_path)
    assert_error("invalid_path", lambda: views.read_repository_file(package, expected_manifest_sha256=receipt["manifest_sha256"], path=path))


@pytest.mark.parametrize("mutation,code", [
    (lambda m: m["entries"].update({"../escape": m["entries"].pop("README.txt")}), "invalid_path"),
    (lambda m: m["entries"].update({"README.txt/child": m["entries"]["empty-file"]}), "invalid_path_topology"),
    (lambda m: m["entries"]["README.txt"].update(mode=True), "invalid_manifest"),
    (lambda m: m["entries"]["README.txt"].update(unexpected="authority"), "invalid_manifest"),
    (lambda m: m["scope"].update(proof_authority=True), "invalid_manifest"),
    (lambda m: m["scope"].update(proof_authority=0), "invalid_manifest"),
    (lambda m: m.update(source_tree_sha256="0" * 64), "tree_digest_mismatch"),
])
def test_tampered_manifest_schema_and_topology_refused_even_with_new_pin(repository, tmp_path, mutation, code):
    package, receipt = capture(repository, tmp_path)
    manifest = load(package, receipt)
    mutation(manifest)
    pin = replace_manifest(package, manifest)
    assert_error(code, lambda: views.load_repository_view(package, expected_manifest_sha256=pin))


def test_child_before_malformed_parent_returns_bounded_error(repository, tmp_path):
    package, receipt = capture(repository, tmp_path)
    manifest = load(package, receipt)
    # Deliberately noncanonical order reaches topology validation before the
    # malformed parent row's own validation; it must never leak AttributeError.
    manifest["entries"] = {"lib/calc.py": manifest["entries"]["lib/calc.py"], "lib": []}
    data = json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode()
    file = package / "manifest.json"
    file.chmod(0o644)
    file.write_bytes(data)
    assert_error("invalid_path_topology", lambda: views.load_repository_view(package, expected_manifest_sha256=digest(data)))


@pytest.mark.parametrize("style,code", [("duplicate", "duplicate_manifest_key"), ("whitespace", "noncanonical_manifest"), ("deep", "manifest_structure_limit")])
def test_strict_manifest_decoding(repository, tmp_path, style, code):
    package, receipt = capture(repository, tmp_path)
    file = package / "manifest.json"
    data = file.read_bytes()
    if style == "duplicate":
        data = b'{"schema":"repository-source-view@1",' + data[1:]
    elif style == "whitespace":
        data += b"\n"
    else:
        data = b"[" * 40 + b"0" + b"]" * 40
    file.chmod(0o644)
    file.write_bytes(data)
    assert_error(code, lambda: views.load_repository_view(package, expected_manifest_sha256=digest(data)))


def test_failed_create_publication_leaves_no_output_or_staging(repository, tmp_path, monkeypatch):
    monkeypatch.setattr(views, "_publish", lambda *a: (_ for _ in ()).throw(views.RepositorySourceViewError("publication_failed")))
    assert_error("publication_failed", lambda: capture(repository, tmp_path))
    assert not (tmp_path / "package").exists()
    assert list(tmp_path.glob(".repository-view-*")) == []


def test_restore_publication_race_never_overwrites_destination(repository, tmp_path, monkeypatch):
    package, receipt = capture(repository, tmp_path)
    target = tmp_path / "restored"
    publish = views._publish
    def competing_destination(parent_fd, staged, name):
        target.mkdir()
        return publish(parent_fd, staged, name)
    monkeypatch.setattr(views, "_publish", competing_destination)
    assert_error("destination_exists", lambda: views.restore_repository_view(package, target, expected_manifest_sha256=receipt["manifest_sha256"]))
    assert list(target.iterdir()) == []
    assert list(tmp_path.glob(".repository-view-*")) == []


@pytest.mark.parametrize("operation", ["create", "restore"])
def test_postrename_fsync_failure_reports_committed_complete_output(repository, tmp_path, monkeypatch, operation):
    if operation == "restore":
        package, receipt = capture(repository, tmp_path)
    output = tmp_path / ("package" if operation == "create" else "restored")
    original_fsync = views.os.fsync
    def fail_after_rename(fd):
        if output.exists():
            raise OSError("injected parent fsync failure")
        return original_fsync(fd)
    monkeypatch.setattr(views.os, "fsync", fail_after_rename)
    if operation == "create":
        result = views.create_repository_view(repository, output)
        assert load(output, result)["source_tree_sha256"] == result["source_tree_sha256"]
    else:
        result = views.restore_repository_view(package, output, expected_manifest_sha256=receipt["manifest_sha256"])
        assert tree(output) == tree(repository)
    assert result["directory_fsync_complete"] is False
    assert list(tmp_path.glob(".repository-view-*")) == []


def test_failed_restore_blob_read_cleans_partial_staging(repository, tmp_path, monkeypatch):
    package, receipt = capture(repository, tmp_path)
    read = views._read_blob
    count = 0
    def fail_staged_read(fd, digest, size):
        nonlocal count
        # load verifies all blobs before the stage exists; fail restoration only.
        if list(tmp_path.glob(".repository-view-*")):
            count += 1
            if count == 2:
                raise views.RepositorySourceViewError("blob_digest_mismatch")
        return read(fd, digest, size)
    monkeypatch.setattr(views, "_read_blob", fail_staged_read)
    assert_error("blob_digest_mismatch", lambda: views.restore_repository_view(package, tmp_path / "restored", expected_manifest_sha256=receipt["manifest_sha256"]))
    assert not (tmp_path / "restored").exists()
    assert list(tmp_path.glob(".repository-view-*")) == []
