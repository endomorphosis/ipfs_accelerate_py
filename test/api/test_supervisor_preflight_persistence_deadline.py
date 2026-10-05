"""A contended preflight checkpoint must defer dispatch and remain retryable."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.artifact_store import BoundedArtifactStore
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as daemon_module
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTask,
    ValidationProjectDependencyPreflightDeferred,
)
from ipfs_accelerate_py.agent_supervisor.validation import project_dependency_preflight as preflight


@contextmanager
def _held_store_lock(store: BoundedArtifactStore, kind: str):
    acquired, release = threading.Event(), threading.Event()
    worker = None
    child = None
    if kind == "thread":
        def hold():
            with store._locked():
                acquired.set()
                release.wait(10)

        worker = threading.Thread(target=hold, daemon=True)
        worker.start()
        assert acquired.wait(5)
    else:
        child = subprocess.Popen(
            [sys.executable, "-c", (
                "import fcntl,sys; handle=open(sys.argv[1], 'ab'); "
                "fcntl.flock(handle.fileno(), fcntl.LOCK_EX); "
                "print('locked', flush=True); sys.stdin.read(1)"
            ), str(store.lock_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        assert child.stdout.readline() == b"locked\n"
    try:
        yield
    finally:
        release.set()
        if worker is not None:
            worker.join(timeout=5)
            assert not worker.is_alive()
        if child is not None:
            try:
                child.communicate(input=b"x", timeout=5)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.communicate(timeout=5)


@pytest.mark.parametrize("holder_kind", ["thread", "process"])
@pytest.mark.parametrize("already_open", [False, True])
def test_contended_preflight_defers_without_publishing_a_success(
    tmp_path: Path, monkeypatch, holder_kind: str, already_open: bool,
) -> None:
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.md", state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json", events_path=tmp_path / "events.jsonl",
        repo_root=tmp_path,
        dependency_preflight_artifact_store_path=tmp_path / "preflight-artifacts",
    )
    candidate = preflight.project_dependency_preflight_error_receipt(
        tmp_path, ["python -m pytest -q"], RuntimeError("fixture"),
    )
    candidate["passed"] = True
    candidate["reason"] = "fixture_dependencies_satisfied"
    candidate.pop("receipt_id")
    candidate.pop("retry_fingerprint")
    candidate["receipt_id"] = preflight._content_sha256(candidate)
    candidate["retry_fingerprint"] = preflight._retry_fingerprint(candidate)
    monkeypatch.setattr(
        daemon_module, "preflight_validation_project_dependencies",
        lambda *_args, **_kwargs: dict(candidate),
    )
    # A short test deadline exercises the production deadline, rather than a
    # provider mock or a timed release of the competing writer's lock.
    monkeypatch.setattr(
        daemon_module, "DEPENDENCY_PREFLIGHT_ARTIFACT_LOCK_TIMEOUT_SECONDS", 0.05,
        raising=False,
    )
    task = PortalTask(
        task_id="LOCK-001", title="Persist preflight before dispatch", status="todo",
        completion="manual", priority="P0", track="runtime",
        validation=["python -m pytest -q"],
    )
    if already_open:
        daemon._persist_dependency_preflight_receipt(candidate)
        holder = daemon._dependency_preflight_artifact_store
    else:
        holder = BoundedArtifactStore(
            daemon.dependency_preflight_artifact_store_path, refresh_on_lock=True,
        )
    original_manifest = holder.manifest_path.read_bytes()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            with _held_store_lock(holder, holder_kind):
                pending = executor.submit(
                    daemon._require_validation_project_dependency_preflight,
                    workspace_path=tmp_path, task=task, attempt=1,
                )
                with pytest.raises(ValidationProjectDependencyPreflightDeferred) as raised:
                    pending.result(timeout=1)
                assert raised.value.receipt["passed"] is False
                assert raised.value.receipt["reason"] == (
                    "project_dependency_preflight_infrastructure_error"
                )
                assert holder.manifest_path.read_bytes() == original_manifest
                event = json.loads(daemon.events_path.read_text())
                projection = event["dependency_preflight"]
                assert projection["inline_receipt"] == raised.value.receipt
                assert projection["completion_authority"] is False
                assert "full_receipt_artifact" not in projection
        # The same daemon can retry after contention; it retains custody of an
        # already-open store and never marks it closed on a failed acquisition.
        assert daemon._require_validation_project_dependency_preflight(
            workspace_path=tmp_path, task=task, attempt=2,
        ) == candidate
        store = daemon._dependency_preflight_artifact_store
        assert store.usage()["blob_count"] == 1
        daemon.close_event_runtime()
        with BoundedArtifactStore(store.path, refresh_on_lock=True) as reopened:
            ref = next(iter(reopened._manifest["blobs"]))
            assert reopened.read_blob(ref) == (
                preflight.canonical_project_dependency_preflight_receipt_bytes(candidate)
            )
    finally:
        daemon.close_event_runtime()
        if not already_open:
            holder.close()


@pytest.mark.parametrize("timeout", [True, False, 0, -1, float("nan"), float("inf"), "1"])
def test_store_rejects_invalid_lock_timeout_before_creating_files(tmp_path, timeout):
    root = tmp_path / "store"
    with pytest.raises(ValueError, match="lock_timeout_seconds"):
        BoundedArtifactStore(root, lock_timeout_seconds=timeout)
    assert not root.exists()
