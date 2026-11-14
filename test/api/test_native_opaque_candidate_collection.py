"""Native opaque evidence binds exact bounded Git/worktree bytes, not labels."""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import subprocess
import zipfile

import pytest

from ipfs_accelerate_py.agent_supervisor.proof import code_proof_obligations as native


def git(repo: Path, *argv: str) -> str:
    return subprocess.run(["git", *argv], cwd=repo, check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "config", "user.name", "Opaque Collection Test")
    git(tmp_path, "config", "user.email", "opaque@example.invalid")
    (tmp_path / "baseline.txt").write_text("baseline\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "baseline")
    return tmp_path


def archive(value=b"[]\n"):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as target:
        target.writestr("sections.json", value)
    return stream.getvalue()


def assert_side(entry, side, payload):
    assert getattr(entry, side + "_source") is None
    assert entry.metadata[side + "_sha256"] == hashlib.sha256(payload).hexdigest()
    assert entry.metadata[side + "_size_bytes"] == len(payload)
    assert entry.metadata[side + "_magic_hex"] == payload[:8].hex()


def test_untracked_zip_has_exact_native_evidence_without_payload_encoding(repo):
    payload = archive()
    (repo / "sections.zip").write_bytes(payload)
    entry, = native.collect_git_candidate_diff(repo)
    assert entry.binary is True
    assert_side(entry, "after", payload)
    assert not any(key.startswith("before_") for key in entry.metadata)
    assert entry.after_blob_id == git(repo, "hash-object", "sections.zip")
    assert len(str(entry.to_dict())) < 2000


def test_modified_zip_binds_both_immutable_and_worktree_sides(repo):
    old, new = archive(b"old\n"), archive(b"new\n")
    (repo / "sections.zip").write_bytes(old)
    git(repo, "add", "sections.zip")
    git(repo, "commit", "-qm", "old archive")
    (repo / "sections.zip").write_bytes(new)
    entry, = native.collect_git_candidate_diff(repo)
    assert_side(entry, "before", old)
    assert_side(entry, "after", new)
    assert entry.before_blob_id == git(repo, "rev-parse", "HEAD:sections.zip")
    assert entry.after_blob_id == git(repo, "hash-object", "sections.zip")


def test_revision_candidate_ignores_unrelated_worktree_bytes(repo):
    old, new = archive(b"old\n"), archive(b"new\n")
    (repo / "sections.zip").write_bytes(old)
    git(repo, "add", "sections.zip")
    git(repo, "commit", "-qm", "old archive")
    baseline = git(repo, "rev-parse", "HEAD")
    (repo / "sections.zip").write_bytes(new)
    git(repo, "commit", "-qam", "new archive")
    candidate = git(repo, "rev-parse", "HEAD")
    (repo / "sections.zip").write_bytes(b"unreviewed working tree")
    entry, = native.collect_git_candidate_diff(repo, base_revision=baseline,
                                             candidate_revision=candidate)
    assert_side(entry, "before", old)
    assert_side(entry, "after", new)
    assert entry.after_blob_id == git(repo, "rev-parse", candidate + ":sections.zip")


def test_ascii_pdf_is_opaque_while_ordinary_text_stays_text(repo):
    payload = b"%PDF-1.7\nASCII PDF fixture\n%%EOF\n"
    (repo / "receipt.pdf").write_bytes(payload)
    (repo / "plain.txt").write_text("ordinary text\n")
    entries = {entry.path: entry for entry in native.collect_git_candidate_diff(repo)}
    assert entries["receipt.pdf"].binary is True
    assert_side(entries["receipt.pdf"], "after", payload)
    assert entries["plain.txt"].binary is False
    assert entries["plain.txt"].after_source == "ordinary text\n"
    assert entries["plain.txt"].metadata == {}


@pytest.mark.parametrize("staged", [False, True])
def test_opaque_rename_preserves_both_side_bindings(repo, staged):
    payload = archive()
    (repo / "old.zip").write_bytes(payload)
    git(repo, "add", "old.zip")
    git(repo, "commit", "-qm", "old archive")
    if staged:
        git(repo, "mv", "old.zip", "new.zip")
    else:
        (repo / "old.zip").rename(repo / "new.zip")
    entry, = native.collect_git_candidate_diff(repo)
    assert entry.change_kind is native.DiffChangeKind.RENAME
    assert_side(entry, "before", payload)
    assert_side(entry, "after", payload)


@pytest.mark.parametrize("revision", [False, True])
def test_oversized_source_is_refused_before_payload_read(repo, monkeypatch, revision):
    (repo / "large.zip").write_bytes(b"PK\x03\x04" + b"x" * 1024)
    baseline = "HEAD"
    candidate = None
    if revision:
        baseline = git(repo, "rev-parse", "HEAD")
        git(repo, "add", "large.zip")
        git(repo, "commit", "-qm", "oversized immutable blob")
        candidate = "HEAD"
    monkeypatch.setattr(native, "MAX_NATIVE_CANDIDATE_BLOB_BYTES", 1024)
    with pytest.raises(ValueError, match="bounded native blob limit"):
        native.collect_git_candidate_diff(repo, base_revision=baseline,
                                          candidate_revision=candidate)


def test_hardlinked_opaque_worktree_source_is_refused(repo):
    (repo / "one.zip").write_bytes(archive())
    os.link(repo / "one.zip", repo / "two.zip")
    with pytest.raises(ValueError, match="hardlinked"):
        native.collect_git_candidate_diff(repo)


def test_symlink_does_not_receive_external_payload_evidence(repo, tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-outside.zip")
    outside.write_bytes(archive())
    (repo / "link.zip").symlink_to(outside)
    entry, = native.collect_git_candidate_diff(repo)
    assert entry.after_source is None
    assert entry.metadata == {}


def test_revision_hash_mismatch_cannot_mint_opaque_evidence(repo, monkeypatch):
    (repo / "sections.zip").write_bytes(archive())
    git(repo, "add", "sections.zip")
    git(repo, "commit", "-qm", "archive")
    (repo / "sections.zip").write_bytes(archive(b"new\n"))
    monkeypatch.setattr(native, "_git_blob_identity", lambda *_args: "0" * 40)
    with pytest.raises(ValueError, match="content binding differs"):
        native.collect_git_candidate_diff(repo)
