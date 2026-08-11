"""API contract tests for the real deterministic-Doctor worktree adapter."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.doctor_worktree_adapter import (
    DOCTOR_WORKTREE_ADAPTER_INTERFACE,
    DoctorExactEdit,
    DoctorWorktreeAdapter,
    DoctorWorktreeCasError,
    DoctorWorktreeSecurityError,
    DoctorWorktreeTamperError,
)


def _git(root: Path, *args: str, input_bytes: bytes | None = None) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout.decode("utf-8").strip()


def _repository(tmp_path: Path, files: dict[str, bytes] | None = None) -> Path:
    root = tmp_path / "repo"
    _git(tmp_path, "init", "-q", "-b", "main", str(root))
    _git(root, "config", "user.email", "doctor-test@example.invalid")
    _git(root, "config", "user.name", "Doctor Test")
    for relative, body in (files or {"pkg/a.py": b"value = 1\n"}).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "base")
    return root


def _digest(body: bytes) -> str:
    return "sha256:" + hashlib.sha256(body).hexdigest()


def test_interface_and_scrubbed_no_network_no_secret_git_environment(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    adapter = DoctorWorktreeAdapter(
        root,
        tmp_path / "state",
        ("pkg/a.py",),
        permitted_refs=("refs/heads/main",),
    )
    assert adapter.INTERFACE == DOCTOR_WORKTREE_ADAPTER_INTERFACE
    assert DOCTOR_WORKTREE_ADAPTER_INTERFACE == "DoctorWorktreeAdapter@1"
    environment = adapter._git_env()  # fixed child-process authority surface
    assert environment["GIT_TERMINAL_PROMPT"] == "0"
    assert environment["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert environment["GIT_ALLOW_PROTOCOL"] == ""
    assert "AWS_SECRET_ACCESS_KEY" not in environment
    assert "OPENAI_API_KEY" not in environment
    assert set(environment).issubset(
        {
            "PATH",
            "HOME",
            "XDG_CONFIG_HOME",
            "LC_ALL",
            "LANG",
            "GIT_CONFIG_NOSYSTEM",
            "GIT_CONFIG_GLOBAL",
            "GIT_TERMINAL_PROMPT",
            "GIT_ASKPASS",
            "SSH_ASKPASS",
            "GCM_INTERACTIVE",
            "GIT_ALLOW_PROTOCOL",
        }
    )


def test_create_materializes_disposable_no_checkout_worktree_and_cids(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    adapter = DoctorWorktreeAdapter(root, tmp_path / "state", ("pkg/a.py",))
    session = adapter.prepare(session_id="snapshot")
    try:
        assert session.worktree_root != root
        assert (session.worktree_root / ".git").is_file()
        assert (session.worktree_root / "pkg/a.py").read_bytes() == b"value = 1\n"
        assert session.baseline.tree_cid.startswith("b")
        assert session.baseline.forest_cid.startswith("b")
        assert session.baseline.blob_map()["pkg/a.py"].startswith("b")
        assert session.baseline.tree_cid != session.baseline.forest_cid
        assert (session.checkpoint_dir / "manifest.json").is_file()
        assert session.journal_path.is_file()
    finally:
        session.restore(reason="test")
        session.close()


def test_exact_edit_rereads_bytes_computes_changed_roots_and_ref_cas(
    tmp_path: Path,
) -> None:
    before = b"value = 1\n"
    after = b"value = 2\n"
    root = _repository(tmp_path, {"pkg/a.py": before})
    adapter = DoctorWorktreeAdapter(
        root,
        tmp_path / "state",
        ("pkg/a.py",),
        permitted_refs=("refs/heads/main",),
    )
    base = _git(root, "rev-parse", "refs/heads/main")
    session = adapter.prepare(base_ref="refs/heads/main", session_id="commit")
    receipt = session.apply_group(
        (
            DoctorExactEdit(
                path="pkg/a.py",
                before_hash=_digest(before),
                after_bytes=after,
                step_id="step:a",
            ),
        ),
        group_id="scc:a",
    )
    assert receipt.changed_paths == ("pkg/a.py",)
    assert receipt.before_tree_cid != receipt.after_tree_cid
    assert receipt.before_forest_cid != receipt.after_forest_cid
    assert receipt.effects[0].before_blob_cid != receipt.effects[0].after_blob_cid
    assert receipt.effects[0].after_hash == _digest(after)
    assert receipt.bytes_reread is True
    assert receipt.durable_effect_ref
    cas = session.commit_ref(target_ref="refs/heads/main")
    assert cas.expected_commit_oid == base
    assert cas.desired_commit_oid == _git(root, "rev-parse", "refs/heads/main")
    assert _git(root, "show", "refs/heads/main:pkg/a.py") == "value = 2"
    session.close()
    assert not session.worktree_root.exists()


def test_noop_and_stale_before_hash_fail_closed_and_restore_exact_bytes(
    tmp_path: Path,
) -> None:
    body = b"value = 1\n"
    root = _repository(tmp_path, {"pkg/a.py": body})
    adapter = DoctorWorktreeAdapter(root, tmp_path / "state", ("pkg/a.py",))
    session = adapter.prepare(session_id="stale")
    with pytest.raises(DoctorWorktreeTamperError, match="before_hash"):
        session.apply_group(
            (DoctorExactEdit("pkg/a.py", _digest(b"wrong\n"), b"value = 2\n"),),
            group_id="scc:stale",
        )
    assert (session.worktree_root / "pkg/a.py").read_bytes() == body
    assert session.state.value == "rolled_back"
    session.close()

    second = adapter.prepare(session_id="noop")
    with pytest.raises(DoctorWorktreeTamperError, match="no-op"):
        second.apply_group(
            (DoctorExactEdit("pkg/a.py", _digest(body), body),),
            group_id="scc:noop",
        )
    assert (second.worktree_root / "pkg/a.py").read_bytes() == body
    second.close()


def test_path_symlink_hardlink_and_unexpected_file_escape_are_rejected(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    adapter = DoctorWorktreeAdapter(root, tmp_path / "state", ("pkg/a.py",))
    with pytest.raises(DoctorWorktreeSecurityError):
        DoctorExactEdit("../outside", _digest(b""), b"x")

    session = adapter.prepare(session_id="hostile")
    original = session.worktree_root / "pkg/a.py"
    linked = session.worktree_root / "pkg/hard.py"
    os.link(original, linked)
    with pytest.raises(
        DoctorWorktreeSecurityError, match="hardlinked|unexpected|hostile"
    ):
        adapter.snapshot(session)
    linked.unlink()
    original.unlink()
    original.symlink_to("/etc/passwd")
    with pytest.raises(DoctorWorktreeSecurityError, match="symlink|hostile"):
        adapter.snapshot(session)
    original.unlink()
    original.write_bytes(b"value = 1\n")
    proof = session.restore(reason="hostile-test")
    assert proof.restored
    session.close()


def test_tracked_symlink_is_never_materialized(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    link = root / "pkg/link"
    link.symlink_to("../../outside")
    _git(root, "add", "pkg/link")
    _git(root, "commit", "-q", "-m", "symlink")
    adapter = DoctorWorktreeAdapter(root, tmp_path / "state", ("pkg/a.py",))
    with pytest.raises(DoctorWorktreeSecurityError, match="tracked symlink"):
        adapter.prepare(session_id="symlink")


def test_ref_cas_conflict_never_overwrites_live_ref(tmp_path: Path) -> None:
    before = b"value = 1\n"
    root = _repository(tmp_path, {"pkg/a.py": before})
    adapter = DoctorWorktreeAdapter(
        root,
        tmp_path / "state",
        ("pkg/a.py",),
        permitted_refs=("refs/heads/main",),
    )
    session = adapter.prepare(base_ref="refs/heads/main", session_id="cas-race")
    session.apply_group(
        (DoctorExactEdit("pkg/a.py", _digest(before), b"value = 2\n"),),
        group_id="scc:a",
    )
    base = _git(root, "rev-parse", "refs/heads/main")
    other_tree = _git(root, "rev-parse", f"{base}^{{tree}}")
    other = _git(
        root,
        "commit-tree",
        other_tree,
        "-p",
        base,
        "-m",
        "concurrent",
        input_bytes=None,
    )
    _git(root, "update-ref", "refs/heads/main", other, base)
    with pytest.raises(DoctorWorktreeCasError, match="conflict"):
        session.commit_ref(target_ref="refs/heads/main")
    assert _git(root, "rev-parse", "refs/heads/main") == other
    proof = session.restore(reason="cas-race")
    # CAS never landed, so candidate bytes can be restored while the
    # independently advanced concurrent ref is preserved.
    assert proof.restored
    assert not proof.quarantined
    assert _git(root, "rev-parse", "refs/heads/main") == other
    session.close(remove_worktree=False)


def test_checkpoint_tamper_quarantines_with_root_comparison_proof(
    tmp_path: Path,
) -> None:
    before = b"value = 1\n"
    root = _repository(tmp_path, {"pkg/a.py": before})
    adapter = DoctorWorktreeAdapter(root, tmp_path / "state", ("pkg/a.py",))
    session = adapter.prepare(session_id="tamper")
    session.apply_group(
        (DoctorExactEdit("pkg/a.py", _digest(before), b"value = 2\n"),),
        group_id="scc:a",
    )
    blob = next(session.checkpoint_dir.glob("*.blob"))
    blob.write_bytes(b"tampered")
    proof = session.restore(reason="tamper")
    assert not proof.restored
    assert proof.quarantined
    quarantine = tmp_path / "state/quarantine/tamper.json"
    assert quarantine.is_file()
    assert "quarantined" in quarantine.read_text(encoding="utf-8")
    session.close(remove_worktree=False)
