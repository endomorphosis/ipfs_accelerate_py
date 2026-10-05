"""Real repository replay and restart boundaries for implementation evidence."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.code_proof_obligations import (
    collect_git_candidate_diff,
)
from ipfs_accelerate_py.agent_supervisor.runtime.artifact_store import BoundedArtifactStore
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
)
from ipfs_accelerate_py.agent_supervisor.validation.project_dependency_preflight import (
    canonical_project_dependency_preflight_receipt_bytes,
    project_dependency_preflight_error_receipt,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _repository(tmp_path: Path, files: dict[str, bytes]) -> tuple[Path, Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Replay Fixture")
    _git(repo, "config", "user.email", "replay@example.invalid")
    _git(repo, "config", "core.autocrlf", "false")
    for path, content in files.items():
        (repo / path).write_bytes(content)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "baseline")
    baseline = _git(repo, "rev-parse", "HEAD")
    replica = tmp_path / "replica"
    _git(repo, "worktree", "add", "--detach", str(replica), baseline)
    return repo, replica, baseline


def _apply(replica: Path, patch: str) -> None:
    applied = subprocess.run(
        ["git", "apply", "-"], cwd=replica, input=patch.encode("utf-8"),
        capture_output=True, check=False,
    )
    assert applied.returncode == 0, applied.stderr.decode("utf-8", errors="replace")


def test_selected_candidate_paths_are_literal_not_git_globs(tmp_path: Path) -> None:
    repo, replica, baseline = _repository(
        tmp_path, {"[slot].py": b"selected = 1\n", "s.py": b"unrelated = 1\n"},
    )
    (repo / "[slot].py").write_bytes(b"selected = 2\n")
    (repo / "s.py").write_bytes(b"unrelated = 99\n")
    _git(repo, "add", ".")

    patch = PortalImplementationDaemon._proposal_repo_patch_text(
        repo, baseline_ref=baseline, entries=[
            entry for entry in collect_git_candidate_diff(repo, base_revision=baseline)
            if entry.path == "[slot].py"
        ],
    )
    _apply(replica, patch)
    assert (replica / "[slot].py").read_bytes() == b"selected = 2\n"
    assert (replica / "s.py").read_bytes() == b"unrelated = 1\n"


@pytest.mark.parametrize("kind", ["staged_modify", "staged_add", "untracked_add", "staged_delete"])
def test_candidate_patch_preserves_crlf_bytes(tmp_path: Path, kind: str) -> None:
    repo, replica, baseline = _repository(
        tmp_path, {"existing.py": b"value = 1\r\n# end\r\n"},
    )
    path = "added.py" if kind.endswith("add") else "existing.py"
    operation = "add" if kind.endswith("add") else kind.removeprefix("staged_")
    expected = b"value = 2\r\n# end\r\n"
    if operation == "delete":
        (repo / path).unlink()
    else:
        (repo / path).write_bytes(expected)
    if kind.startswith("staged_"):
        _git(repo, "add", "--", path)

    patch = PortalImplementationDaemon._proposal_repo_patch_text(
        repo, baseline_ref=baseline,
        entries=collect_git_candidate_diff(repo, base_revision=baseline),
    )
    _apply(replica, patch)
    if operation == "delete":
        assert not (replica / path).exists()
    else:
        assert (replica / path).read_bytes() == expected


def test_shared_preflight_store_restart_repairs_torn_current_manifest(tmp_path: Path) -> None:
    root = tmp_path / "preflight-artifacts"
    receipt = project_dependency_preflight_error_receipt(
        tmp_path, ["python -m pytest"], RuntimeError("fixture dependency failure"),
    )
    canonical = canonical_project_dependency_preflight_receipt_bytes(receipt)
    with BoundedArtifactStore(root, refresh_on_lock=True) as initial:
        reference = initial.put_blob(
            canonical, kind="validation_project_dependency_preflight_receipt",
            retention_class="checkpoint", media_type="application/json",
        )
    retained_previous = (root / "manifest.previous.json").read_bytes()
    (root / "manifest.json").write_bytes(b"{torn")

    with BoundedArtifactStore(root, refresh_on_lock=True) as reopened:
        assert reopened.metrics().manifest_recoveries == 1
        assert reopened.read_blob(reference.artifact_id) == canonical
        assert reopened.put_blob(
            canonical, kind="validation_project_dependency_preflight_receipt",
            retention_class="checkpoint", media_type="application/json",
        ) == reference
        assert reopened.usage()["blob_count"] == 1
        restored = json.loads((root / "manifest.json").read_bytes())
        assert restored["blobs"][reference.artifact_id]["kind"] == reference.kind
        assert (root / "manifest.previous.json").read_bytes() == retained_previous


@pytest.mark.parametrize("holder_kind", ["thread", "process"])
def test_preflight_checkpoint_close_honors_lock_deadline(tmp_path: Path, holder_kind: str) -> None:
    store = BoundedArtifactStore(tmp_path / "store", refresh_on_lock=True)
    reference = store.put_blob(b"retained checkpoint", retention_class="checkpoint")
    original = store.manifest_path.read_bytes()
    acquired, release = threading.Event(), threading.Event()
    process = None
    holder = None

    def hold_thread_lock() -> None:
        with store._locked():
            acquired.set()
            assert release.wait(10), "fixture lock was not released"

    if holder_kind == "thread":
        holder = threading.Thread(target=hold_thread_lock, daemon=True)
        holder.start()
        assert acquired.wait(5)
    else:
        process = subprocess.Popen(
            [sys.executable, "-c", (
                "import fcntl,sys; "
                "handle=open(sys.argv[1], 'ab'); "
                "fcntl.flock(handle.fileno(), fcntl.LOCK_EX); "
                "print('locked', flush=True); sys.stdin.read(1)"
            ), str(store.lock_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        assert process.stdout.readline() == b"locked\n"

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(store.close, timeout_seconds=0.05)
            try:
                # This is a generous outer guard: the lock stays held until
                # close has returned, so blocking flock/acquire cannot pass.
                assert pending.result(timeout=1) is False
                assert store.manifest_path.read_bytes() == original
            finally:
                release.set()
                if process is not None:
                    process.communicate(input=b"x", timeout=5)
                if holder is not None:
                    holder.join(timeout=5)
    finally:
        release.set()
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
        if holder is not None:
            holder.join(timeout=5)
        store.close()

    with BoundedArtifactStore(store.path, refresh_on_lock=True) as reopened:
        assert reopened.read_blob(reference) == b"retained checkpoint"


def test_portal_retains_preflight_store_for_retry_after_contended_close(tmp_path: Path) -> None:
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.md", state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json", events_path=tmp_path / "events.jsonl",
        repo_root=tmp_path,
    )
    receipt = project_dependency_preflight_error_receipt(
        tmp_path, ["python -m pytest"], RuntimeError("fixture dependency failure"),
    )
    projection = daemon._persist_dependency_preflight_receipt(receipt)
    store = daemon._dependency_preflight_artifact_store
    acquired, release = threading.Event(), threading.Event()

    def hold_lock() -> None:
        with store._locked():
            acquired.set()
            assert release.wait(10), "fixture lock was not released"

    holder = threading.Thread(target=hold_lock, daemon=True)
    holder.start()
    assert acquired.wait(5)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(daemon.close_event_runtime)
            try:
                with pytest.raises(TimeoutError, match="dependency preflight checkpoint"):
                    pending.result(timeout=2)
                assert daemon._dependency_preflight_artifact_store is store
            finally:
                release.set()
                holder.join(timeout=5)
    finally:
        release.set()
        holder.join(timeout=5)
        daemon.close_event_runtime()
    assert daemon._dependency_preflight_artifact_store is None
    with BoundedArtifactStore(store.path, refresh_on_lock=True) as reopened:
        assert reopened.read_blob(projection["full_receipt_artifact"]) == (
            canonical_project_dependency_preflight_receipt_bytes(receipt)
        )
