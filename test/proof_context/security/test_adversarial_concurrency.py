"""PCCE-075: hostile concurrent writers and interrupted execution.

These tests drive the production recovery coordinator, lifecycle ports, fenced
checkpoint store, and disposable Git worktree port. They preserve a minimized
current-tree double-publication failure as explicit no-go evidence: bootstrap
constructs an in-process checkpoint store per runtime even when Kit is required.
The remaining hermetic schedules exercise that stand-in without upgrading it to
cross-process durability or production CAS authority.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import multiprocessing
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

# isort: split
from ipfs_accelerate_py.proof_context import bootstrap, lifecycle, recovery, sandbox

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = Path(__file__).with_name("fixtures") / "concurrency" / "schedules.v1.json"
PATCH = {
    "declared_files": ["src/demo/__init__.py"],
    "files": {"src/demo/__init__.py": "VALUE = 75\n"},
    "adapter_id": "external-patch",
    "approver_id": "coordinator",
}
WORKER_ARGUMENT = "--pcce075-crash-worker"
WORKER_ROOT_ENV = "PCCE075_WORKER_ROOT"
WORKER_LOG_ENV = "PCCE075_WORKER_LOG"
DOUBLE_PUBLICATION_PROBE = "double-publication"
DUPLICATE_TERMINAL_PROBE = "duplicate-terminal"
STUCK_BOUNDARY_PROBE = "stuck-boundary"
PROBE_KINDS = frozenset(
    {
        DOUBLE_PUBLICATION_PROBE,
        DUPLICATE_TERMINAL_PROBE,
        STUCK_BOUNDARY_PROBE,
    }
)
BOOTSTRAP_GIT_TIMEOUT_SECONDS = 8.0
PROCESS_REAP_TIMEOUT_SECONDS = 2.0
THREAD_POLL_SECONDS = 0.01
pytestmark = pytest.mark.timeout(45)


class AuditedCompletedProcess(subprocess.CompletedProcess[str]):
    process_id: int
    process_group: int
    process_absent: bool
    process_group_absent: bool
    process_group_lingered: bool


def _schedule_fixture() -> Mapping[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


SCHEDULES = _schedule_fixture()
CRASH_CASES = tuple(
    (
        str(item["schedule_id"]),
        int(item["seed"]),
        str(item["stage"]),
        str(item["position"]),
        str(item["expected_status"]),
    )
    for item in SCHEDULES["crash_matrix"]
)


def _crash_interrupt_type() -> type[BaseException]:
    """Resolve the live class if another test reloads the recovery module."""

    return recovery.CrashInterrupt


def _process_group_absent(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _wait_for_process_group_exit(process_group: int) -> bool:
    deadline = time.monotonic() + PROCESS_REAP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if _process_group_absent(process_group):
            return True
        time.sleep(THREAD_POLL_SECONDS)
    return _process_group_absent(process_group)


def _process_absent(process_id: int) -> bool:
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _wait_for_process_exit(process_id: int) -> bool:
    deadline = time.monotonic() + PROCESS_REAP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if _process_absent(process_id):
            return True
        time.sleep(THREAD_POLL_SECONDS)
    return _process_absent(process_id)


def _bounded_process(
    command: Sequence[str],
    *,
    cwd: Path | None,
    env: Mapping[str, str],
    timeout: float,
    pass_fds: Sequence[int] = (),
    inherit_process_group: bool = False,
) -> AuditedCompletedProcess:
    """Run behind a parent-owned deadline and audit the entire process group.

    A nested command in a bounded probe inherits the probe's process group. If
    that nested command times out, killing the group deliberately terminates
    the probe controller too; its outer parent remains responsible for the
    final reap audit.
    """

    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=not inherit_process_group,
        pass_fds=tuple(pass_fds),
    )
    process_group = os.getpgrp() if inherit_process_group else process.pid
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            pass
        try:
            stdout, stderr = process.communicate(timeout=PROCESS_REAP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        group_absent = (
            False
            if inherit_process_group
            else _wait_for_process_group_exit(process_group)
        )
        timeout_error = subprocess.TimeoutExpired(command, timeout, stdout, stderr)
        timeout_error.process_group = process_group  # type: ignore[attr-defined]
        timeout_error.process_group_absent = group_absent  # type: ignore[attr-defined]
        raise timeout_error from exc
    completed = AuditedCompletedProcess(command, process.returncode, stdout, stderr)
    completed.process_id = process.pid
    completed.process_group = process_group
    completed.process_absent = process.poll() is not None
    if inherit_process_group:
        completed.process_group_absent = False
        completed.process_group_lingered = False
        return completed

    group_lingered = not _process_group_absent(process_group)
    if group_lingered:
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            pass
    group_absent = _wait_for_process_group_exit(process_group)
    completed.process_group_absent = group_absent
    completed.process_group_lingered = group_lingered
    return completed


def _ensure_process_absent(process: subprocess.Popen[str]) -> Mapping[str, Any]:
    """Kill and reap a process group to a fixed deadline without retaining streams."""

    errors: list[str] = []
    leader_was_running = process.poll() is None
    if leader_was_running:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as exc:
            errors.append(f"process-kill:{type(exc).__name__}")
    try:
        process.communicate(timeout=PROCESS_REAP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as exc:
            errors.append(f"process-rekill:{type(exc).__name__}")
        try:
            process.communicate(timeout=PROCESS_REAP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            errors.append("process-reap-timeout")
    group_lingered = not _process_group_absent(process.pid)
    if group_lingered:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as exc:
            errors.append(f"process-group-kill:{type(exc).__name__}")
    group_absent = _wait_for_process_group_exit(process.pid)
    if not group_absent:
        errors.append("process-group-present")
    return {
        "errors": errors,
        "process_absent": process.poll() is not None,
        "process_group_absent": group_absent,
        "process_group_lingered": bool(group_lingered and not leader_was_running),
    }


def _git(
    repository: Path,
    *arguments: str,
    inherit_process_group: bool = False,
) -> subprocess.CompletedProcess[str]:
    completed = _bounded_process(
        ["git", "-C", str(repository), *arguments],
        cwd=None,
        env={
            "PATH": os.environ.get("PATH", "/usr/bin"),
            "LANG": "C",
            "LC_ALL": "C",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
        },
        timeout=BOOTSTRAP_GIT_TIMEOUT_SECONDS,
        inherit_process_group=inherit_process_group,
    )
    if completed.returncode != 0 or completed.process_group_lingered:
        raise subprocess.CalledProcessError(
            completed.returncode or 70,
            completed.args,
            completed.stdout,
            completed.stderr,
        )
    return completed


def _bounded_bootstrap_git(
    repository: Path,
    *arguments: str,
    cwd: Path | None = None,
    inherit_process_group: bool = False,
) -> str:
    """Test-owned bounded replacement for bootstrap's unbounded Git helper."""

    command = [
        bootstrap._git_executable(),
        "-C",
        str(cwd or repository),
        *arguments,
    ]
    try:
        completed = _bounded_process(
            command,
            cwd=None,
            env=bootstrap._git_env(),
            timeout=BOOTSTRAP_GIT_TIMEOUT_SECONDS,
            inherit_process_group=inherit_process_group,
        )
    except subprocess.TimeoutExpired as exc:
        group_absent = bool(getattr(exc, "process_group_absent", False))
        raise bootstrap.UnavailableCapabilityError(
            "git command exceeded bounded execution",
            details={
                "capability": "git",
                "reason": (
                    "bounded-timeout-process-group-absent"
                    if group_absent
                    else "bounded-timeout-process-group-not-reaped"
                ),
            },
        ) from exc
    if completed.returncode != 0 or completed.process_group_lingered:
        detail = (
            "git left a live descendant process"
            if completed.process_group_lingered
            else (completed.stderr or completed.stdout or "git failed").strip()
        )
        raise bootstrap.UnavailableCapabilityError(
            "git command failed",
            details={"capability": "git", "reason": detail[:120]},
        )
    return (completed.stdout or "").strip()


def _head(repository: Path, *, inherit_process_group: bool = False) -> str:
    return _git(
        repository,
        "rev-parse",
        "HEAD",
        inherit_process_group=inherit_process_group,
    ).stdout.strip()


def _canonical_snapshot(
    repository: Path,
    *,
    inherit_process_group: bool = False,
) -> Mapping[str, str]:
    """Capture the protected ref, commit, committed/index trees, and porcelain."""

    return {
        "ref": _git(
            repository,
            "symbolic-ref",
            "--quiet",
            "HEAD",
            inherit_process_group=inherit_process_group,
        ).stdout.strip(),
        "head": _head(repository, inherit_process_group=inherit_process_group),
        "head_tree": _git(
            repository,
            "rev-parse",
            "HEAD^{tree}",
            inherit_process_group=inherit_process_group,
        ).stdout.strip(),
        "index_tree": _git(
            repository,
            "write-tree",
            inherit_process_group=inherit_process_group,
        ).stdout.strip(),
        "porcelain": _git(
            repository,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            inherit_process_group=inherit_process_group,
        ).stdout,
    }


def _registered_worktrees(
    repository: Path,
    *,
    inherit_process_group: bool = False,
) -> tuple[Path, ...]:
    output = _git(
        repository,
        "worktree",
        "list",
        "--porcelain",
        inherit_process_group=inherit_process_group,
    ).stdout
    return tuple(
        Path(line.removeprefix("worktree ")).resolve()
        for line in output.splitlines()
        if line.startswith("worktree ")
    )


def _synchronize_runtime_modules(*, inherit_process_group: bool = False) -> None:
    """Refresh live classes, then replace every production Git call with a bound."""

    global bootstrap
    bootstrap = importlib.import_module(
        "ipfs_accelerate_py.proof_context.bootstrap"
    )
    importlib.reload(bootstrap)
    if inherit_process_group:

        def bounded_inherited_git(
            repository: Path,
            *arguments: str,
            cwd: Path | None = None,
        ) -> str:
            return _bounded_bootstrap_git(
                repository,
                *arguments,
                cwd=cwd,
                inherit_process_group=True,
            )

        bootstrap._git = bounded_inherited_git  # type: ignore[attr-defined]
    else:
        bootstrap._git = _bounded_bootstrap_git  # type: ignore[attr-defined]


def _run_child_probe_threads(
    calls: Sequence[tuple[str, Callable[[], Any]]],
    *,
    release: Callable[[], None],
    thread_prefix: str,
) -> tuple[list[tuple[str, Any]], Mapping[str, Any]]:
    """Run synchronized calls only inside a parent-bounded child process.

    Waiting here is intentionally delegated to the surrounding killable
    process-group boundary. Daemon workers cannot hold interpreter shutdown,
    and the outer test kills the complete group if this controller stalls.
    """

    completed: list[tuple[str, Any]] = []
    failures: list[BaseException] = []
    completion = threading.Condition()

    def invoke(label: str, call: Callable[[], Any]) -> None:
        try:
            result = call()
        except BaseException as exc:  # noqa: BLE001 - report to the controller
            with completion:
                failures.append(exc)
                completed.append((label, None))
                completion.notify_all()
        else:
            with completion:
                completed.append((label, result))
                completion.notify_all()

    threads = [
        threading.Thread(
            target=invoke,
            args=(label, call),
            name=f"{thread_prefix}-{index}",
            daemon=True,
        )
        for index, (label, call) in enumerate(calls, start=1)
    ]
    for thread in threads:
        thread.start()
    release()
    with completion:
        while len(completed) != len(threads):
            completion.wait()
    for thread in threads:
        thread.join()
    if failures:
        raise failures[0]
    return completed, {
        "errors": [],
        "calls_done": len(completed) == len(calls),
        "threads_absent": not any(thread.is_alive() for thread in threads),
        "termination_authority": "parent-owned-killable-process-group",
    }


def _open_production_runtime(
    root: Path,
    *,
    inherit_process_group: bool = False,
) -> tuple[Path, Path, bootstrap.RuntimeBundle]:
    _synchronize_runtime_modules(inherit_process_group=inherit_process_group)
    repository = root / "repository"
    storage = root / "runtime-state"
    try:
        repository = bootstrap.create_ordinary_python_repository(repository)
        bundle = bootstrap.open_runtime(
            repository,
            options=bootstrap.RuntimeOptions(
                kit_root=storage / "kit",
                worktree_parent=storage / "worktrees",
                require_kit=True,
            ),
        )
        _assert_actual_kit(bundle)
    except BaseException as exc:
        if (repository / ".git").exists():
            failed_cleanup = _cleanup_runtimes(
                repository,
                storage,
                [],
                inherit_process_group=inherit_process_group,
            )
            if failed_cleanup["errors"]:
                exc.add_note(f"runtime-open cleanup: {failed_cleanup['errors']!r}")
        elif storage.exists():
            try:
                shutil.rmtree(storage)
            except OSError as cleanup_error:
                exc.add_note(f"runtime-open storage cleanup: {type(cleanup_error).__name__}")
        raise
    return repository, storage, bundle


def _attempt(
    bundle: bootstrap.RuntimeBundle,
    *,
    writer_id: str,
    generation: int | None = None,
    fence_token: str | None = None,
    lease_expires_at: int | None = None,
) -> recovery.AttemptIdentity:
    base = bundle.session.attempt
    return recovery.AttemptIdentity(
        attempt_id=base.attempt_id,
        writer_id=writer_id,
        writer_generation=generation or base.writer_generation,
        fence_token=fence_token or base.fence_token,
        lease_id=base.lease_id,
        fence_id=base.fence_id,
        identities=base.identities,
        lease_expires_at=lease_expires_at
        if lease_expires_at is not None
        else base.lease_expires_at,
    )


def _coordinator(
    repository: Path,
    bundle: bootstrap.RuntimeBundle,
    store: recovery.FencedCheckpointStore,
    attempt: recovery.AttemptIdentity,
    *,
    clock: Callable[[], int] | None = None,
) -> recovery.RecoveryCoordinator:
    return recovery.RecoveryCoordinator.open(
        repository,
        ports=bundle.session.lifecycle_ports,
        identities=attempt.identities,
        attempt=attempt,
        store=store,
        mode="production",
        clock=clock,
    )


def _seal_cid(record: recovery.RecoveryRecord) -> str | None:
    lifecycle_record = record.lifecycle
    trace = lifecycle_record.get("trace") if isinstance(lifecycle_record, Mapping) else None
    if not isinstance(trace, (list, tuple)):
        return None
    for artifact in trace:
        if not isinstance(artifact, Mapping) or artifact.get("stage") != lifecycle.SEAL_STAGE:
            continue
        payload = artifact.get("payload")
        if isinstance(payload, Mapping) and isinstance(payload.get("seal_cid"), str):
            return str(payload["seal_cid"])
        if isinstance(artifact.get("artifact_cid"), str):
            return str(artifact["artifact_cid"])
    return None


def _seal_cid_from_history(
    store: recovery.FencedCheckpointStore,
    attempt_id: str,
) -> str | None:
    for checkpoint in store.history(attempt_id):
        if checkpoint.get("stage") != lifecycle.SEAL_STAGE:
            continue
        artifact = checkpoint.get("artifact")
        if not isinstance(artifact, Mapping):
            continue
        payload = artifact.get("payload")
        if isinstance(payload, Mapping) and isinstance(payload.get("seal_cid"), str):
            return str(payload["seal_cid"])
    return None


def _stage_from_record(
    record: recovery.RecoveryRecord,
    stage: str,
) -> Mapping[str, Any]:
    lifecycle_record = record.lifecycle
    trace = lifecycle_record.get("trace") if isinstance(lifecycle_record, Mapping) else None
    if isinstance(trace, (list, tuple)):
        for artifact in trace:
            if isinstance(artifact, Mapping) and artifact.get("stage") == stage:
                return artifact
    raise AssertionError(f"stage {stage!r} absent from recovery lifecycle")


def _published_settlements(
    store: recovery.FencedCheckpointStore, attempt_id: str
) -> list[Mapping[str, Any]]:
    return [
        item
        for item in store.history(attempt_id)
        if item.get("stage") == recovery.PUBLISH_BOUNDARY
        and item.get("position") == "after"
        and item.get("settled") is True
        and item.get("published") is True
    ]


def _class_identity(value: object) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def _assert_actual_kit(
    bundle: bootstrap.RuntimeBundle,
    checkpoint_store: object | None = None,
) -> Mapping[str, Any]:
    """Prove Kit is present and name the store actually injected into the coordinator."""

    capability = bundle.kit
    kit_store = bundle.session.kit_store
    actual_checkpoint_store = bundle.session.store if checkpoint_store is None else checkpoint_store
    assert capability.available is True
    assert capability.module == "ipfs_kit_py.proof_context.state_store"
    assert capability.reason is None
    assert kit_store is not None
    assert getattr(kit_store, "interface", None) == "KitProofContextStateStore@0.1"
    assert Path(kit_store.root).resolve() == Path(bundle.session.options.kit_root).resolve()
    assert actual_checkpoint_store is not kit_store
    assert _class_identity(actual_checkpoint_store) == (
        "ipfs_accelerate_py.proof_context.recovery.FencedCheckpointStore"
    )
    assert all(
        callable(getattr(actual_checkpoint_store, name, None))
        for name in ("put", "latest", "history", "reclaim")
    )
    kit_probe_cid = str(kit_store.cid_for(b"pcce075-kit-store-capability-probe"))
    kit_store_binding_cid = recovery.mint_recovery_cid(
        {
            "kind": "actual-kit-store-binding",
            "store_class": _class_identity(kit_store),
            "interface": getattr(kit_store, "interface", None),
            "root_digest": hashlib.sha256(
                str(Path(kit_store.root).resolve()).encode("utf-8")
            ).hexdigest(),
        }
    )
    return {
        "kit_available": capability.available,
        "kit_module": capability.module,
        "kit_version": capability.version,
        "kit_store_class": _class_identity(kit_store),
        "kit_store_interface": getattr(kit_store, "interface", None),
        "kit_store_binding_cid": kit_store_binding_cid,
        "kit_store_probe_cid": kit_probe_cid,
        "kit_store_root_matches_requested": True,
        "declared_checkpoint_authority": recovery.PERSISTENCE_AUTHORITY,
        "actual_checkpoint_store_class": _class_identity(actual_checkpoint_store),
        "actual_checkpoint_scope": "runtime-instance-in-process",
        "checkpoint_store_is_kit_store": actual_checkpoint_store is kit_store,
        "coordinator_store_is_bundle_session_store": (
            actual_checkpoint_store is bundle.session.store
        ),
        "bundle_session_store_class": _class_identity(bundle.session.store),
    }


def _evidence_section(
    applicable: bool,
    identities: Sequence[Mapping[str, Any]],
    observations: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "applicable": applicable,
        "identities": [dict(item) for item in identities],
        "observations": dict(observations),
    }


def _record_worktree_ids(record: object | None) -> set[str]:
    identifiers: set[str] = set()
    if record is None:
        return identifiers
    identities = getattr(record, "identities", None)
    worktree_id = getattr(identities, "worktree_id", None)
    if isinstance(worktree_id, str) and worktree_id:
        identifiers.add(worktree_id)
    lifecycle_record = getattr(record, "lifecycle", None)
    trace = lifecycle_record.get("trace") if isinstance(lifecycle_record, Mapping) else None
    if not isinstance(trace, (tuple, list)):
        return identifiers
    for artifact in trace:
        if not isinstance(artifact, Mapping):
            continue
        payload = artifact.get("payload")
        if not isinstance(payload, Mapping):
            continue
        candidate = payload.get("worktree_id")
        if isinstance(candidate, str) and candidate:
            identifiers.add(candidate)
    return identifiers


def _coordination_sections(
    entries: Sequence[
        tuple[
            bootstrap.RuntimeBundle,
            recovery.AttemptIdentity,
            object,
            object | None,
        ]
    ],
    *,
    process: Mapping[str, Any] | None = None,
    extra_worktree_ids: Sequence[str] = (),
) -> Mapping[str, Mapping[str, Any]]:
    """Describe actual injected stores and all schedule coordination identities."""

    lease_identities: list[Mapping[str, Any]] = []
    fence_identities: list[Mapping[str, Any]] = []
    cas_identities: list[Mapping[str, Any]] = []
    checkpoint_identities: list[Mapping[str, Any]] = []
    worktree_ids = {str(item) for item in extra_worktree_ids if item}
    worktree_identities: list[Mapping[str, Any]] = []
    worktree_parent_bindings: set[str] = set()
    store_observations: list[Mapping[str, Any]] = []
    checkpoint_observations: list[Mapping[str, Any]] = []
    for ordinal, (bundle, attempt, store, record) in enumerate(entries, start=1):
        lease_identities.append(
            {
                "attempt_id": attempt.attempt_id,
                "lease_id": attempt.lease_id,
            }
        )
        fence_identities.append(
            {
                "attempt_id": attempt.attempt_id,
                "fence_id": attempt.fence_id,
                "fence_cid": attempt.fence_token,
                "generation": attempt.writer_generation,
                "writer_id": attempt.writer_id,
            }
        )
        store_identity = recovery.mint_recovery_cid(
            {
                "kind": "injected-coordinator-store-binding",
                "ordinal": ordinal,
                "attempt_id": attempt.attempt_id,
                "writer_id": attempt.writer_id,
                "store_class": _class_identity(store),
            }
        )
        cas_identities.append(
            {
                "store_binding_cid": store_identity,
                "attempt_id": attempt.attempt_id,
            }
        )
        history = tuple(store.history(attempt.attempt_id))  # type: ignore[attr-defined]
        checkpoint_cids = [
            str(item["checkpoint_cid"])
            for item in history
            if isinstance(item.get("checkpoint_cid"), str)
        ]
        checkpoint_identities.extend(
            {
                "store_binding_cid": store_identity,
                "checkpoint_cid": checkpoint_cid,
            }
            for checkpoint_cid in checkpoint_cids
        )
        store_observations.append(
            {
                "store_binding_cid": store_identity,
                "actual_store_class": _class_identity(store),
                "is_bundle_session_store": store is bundle.session.store,
                "is_kit_store": store is bundle.session.kit_store,
                "scope": "in-process-object",
            }
        )
        checkpoint_observations.append(
            {
                "store_binding_cid": store_identity,
                "count": len(checkpoint_cids),
                "latest_checkpoint_cid": checkpoint_cids[-1] if checkpoint_cids else None,
            }
        )
        path = bundle.session.worktree.path
        if path is not None:
            worktree_ids.add(Path(path).name)
        worktree_parent = Path(bundle.session.options.worktree_parent).resolve()
        worktree_parent_binding_cid = recovery.mint_recovery_cid(
            {
                "kind": "disposable-worktree-parent-binding",
                "root_digest": hashlib.sha256(str(worktree_parent).encode("utf-8")).hexdigest(),
            }
        )
        worktree_parent_bindings.add(worktree_parent_binding_cid)
        worktree_identities.append(
            {
                "attempt_id": attempt.attempt_id,
                "worktree_id": Path(path).name if path is not None else None,
                "worktree_parent_binding_cid": worktree_parent_binding_cid,
            }
        )
        worktree_ids.update(_record_worktree_ids(record))
        for item in history:
            artifact = item.get("artifact")
            if not isinstance(artifact, Mapping):
                continue
            payload = artifact.get("payload")
            candidate = payload.get("worktree_id") if isinstance(payload, Mapping) else None
            if isinstance(candidate, str) and candidate:
                worktree_ids.add(candidate)
    return {
        "process": process
        if process is not None
        else _evidence_section(
            False,
            (),
            {"reason": "schedule-executed-inside-the-bounded-spawned-process"},
        ),
        "worktree": _evidence_section(
            bool(worktree_identities),
            [
                *worktree_identities,
                *[
                    {"worktree_id": item, "source": "external-process-observation"}
                    for item in sorted(set(extra_worktree_ids))
                ],
            ],
            {
                "raw_paths_retained": False,
                "identity_source": "runtime-port-and-checkpoint-trace",
                "distinct_worktree_ids": len(worktree_ids),
                "distinct_worktree_parent_bindings": len(worktree_parent_bindings),
            },
        ),
        "lease": _evidence_section(
            bool(entries),
            lease_identities,
            {"expires_at": [attempt.lease_expires_at for _, attempt, _, _ in entries]},
        ),
        "fence": _evidence_section(
            bool(entries),
            fence_identities,
            {
                "current": [
                    {
                        "generation": store.current_generation(attempt.attempt_id),  # type: ignore[attr-defined]
                        "writer_id": store.current_writer(attempt.attempt_id),  # type: ignore[attr-defined]
                        "fence_cid": store.current_token(attempt.attempt_id),  # type: ignore[attr-defined]
                    }
                    for _, attempt, store, _ in entries
                ]
            },
        ),
        "cas": _evidence_section(
            bool(entries),
            cas_identities,
            {"actual_injected_stores": store_observations},
        ),
        "checkpoints": _evidence_section(
            bool(checkpoint_identities),
            checkpoint_identities,
            {"stores": checkpoint_observations},
        ),
    }


def _assert_same_authority_binding(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> None:
    """Compare authority identity while retaining process-local version discovery."""

    stable_fields = set(first) - {"kit_version"}
    assert stable_fields == set(second) - {"kit_version"}
    assert {field: first[field] for field in stable_fields} == {
        field: second[field] for field in stable_fields
    }


def _cleanup_runtimes(
    repository: Path,
    storage: Path,
    bundles: list[bootstrap.RuntimeBundle],
    *,
    abandoned_worktrees: tuple[Path, ...] = (),
    inherit_process_group: bool = False,
) -> Mapping[str, Any]:
    """Attempt every cleanup action even when an earlier one reports failure."""

    errors: list[str] = []
    observed_paths = {
        path.resolve()
        for path in (
            *(bundle.session.worktree.path for bundle in bundles),
            *abandoned_worktrees,
        )
        if path is not None
    }
    for bundle in bundles:
        try:
            result = bundle.session.worktree.discard(
                bundle.session.lifecycle_identities,
                repository,
            )
            if result.get("discarded") is not True:
                errors.append("runtime-discard-not-proven")
        except Exception as exc:  # noqa: BLE001 - preserve and continue cleanup
            errors.append(f"runtime-discard:{type(exc).__name__}")
    for path in abandoned_worktrees:
        try:
            if path.exists() or path.resolve() in _registered_worktrees(
                repository,
                inherit_process_group=inherit_process_group,
            ):
                _git(
                    repository,
                    "worktree",
                    "remove",
                    "--force",
                    str(path),
                    inherit_process_group=inherit_process_group,
                )
        except Exception as exc:  # noqa: BLE001 - preserve and continue cleanup
            errors.append(f"abandoned-worktree:{type(exc).__name__}")
    try:
        registered_before_prune = _registered_worktrees(
            repository,
            inherit_process_group=inherit_process_group,
        )
        storage_root = storage.resolve()
        for path in registered_before_prune:
            if path == repository.resolve():
                continue
            if not path.is_relative_to(storage_root):
                errors.append("out-of-scope-registered-worktree")
                continue
            observed_paths.add(path)
            _git(
                repository,
                "worktree",
                "remove",
                "--force",
                str(path),
                inherit_process_group=inherit_process_group,
            )
    except Exception as exc:  # noqa: BLE001 - preserve and continue cleanup
        errors.append(f"registered-worktree-cleanup:{type(exc).__name__}")
    try:
        _git(
            repository,
            "worktree",
            "prune",
            inherit_process_group=inherit_process_group,
        )
    except Exception as exc:  # noqa: BLE001 - preserve and continue cleanup
        errors.append(f"worktree-prune:{type(exc).__name__}")
    try:
        registered = _registered_worktrees(
            repository,
            inherit_process_group=inherit_process_group,
        )
    except Exception as exc:  # noqa: BLE001 - a missing audit is a cleanup failure
        errors.append(f"worktree-audit:{type(exc).__name__}")
        registered = ()
    try:
        if storage.exists():
            shutil.rmtree(storage)
    except Exception as exc:  # noqa: BLE001 - preserve and continue cleanup
        errors.append(f"storage-remove:{type(exc).__name__}")
    cleanup_identities: list[Mapping[str, Any]] = [
        {
            "storage_binding_cid": recovery.mint_recovery_cid(
                {
                    "kind": "runtime-storage-cleanup-binding",
                    "root_digest": hashlib.sha256(
                        str(storage.resolve()).encode("utf-8")
                    ).hexdigest(),
                }
            )
        }
    ]
    cleanup_identities.extend({"worktree_id": path.name} for path in sorted(observed_paths))
    return {
        "applicable": True,
        "identities": cleanup_identities,
        "observations": {
            "runtime_discard_attempts": len(bundles),
            "abandoned_worktree_cleanup_attempts": len(abandoned_worktrees),
            "registry_audited": True,
        },
        "errors": errors,
        "processes_absent": True,
        "worktrees_absent": all(not path.exists() for path in observed_paths),
        "registry": [
            "canonical-repository" if item == repository.resolve() else "unexpected-worktree"
            for item in registered
        ],
        "storage_absent": not storage.exists(),
    }


def _assert_clean(cleanup: Mapping[str, Any], repository: Path) -> None:
    assert cleanup["applicable"] is True
    assert isinstance(cleanup["identities"], list) and cleanup["identities"]
    assert isinstance(cleanup["observations"], Mapping)
    assert cleanup["observations"]["registry_audited"] is True
    assert {
        name: cleanup[name]
        for name in (
            "errors",
            "processes_absent",
            "worktrees_absent",
            "registry",
            "storage_absent",
        )
    } == {
        "errors": [],
        "processes_absent": True,
        "worktrees_absent": True,
        "registry": ["canonical-repository"],
        "storage_absent": True,
    }


def _assert_sanitized_log(value: Any, *, key: str = "") -> None:
    forbidden_keys = {
        "authorization",
        "credentials",
        "environment",
        "password",
        "stderr",
        "stdout",
        "worktree_path",
    }
    if isinstance(value, Mapping):
        for name, item in value.items():
            normalized = str(name).lower()
            assert normalized not in forbidden_keys
            assert not any(
                fragment in normalized
                for fragment in (
                    "api_key",
                    "apikey",
                    "authorization",
                    "bearer",
                    "credential",
                    "password",
                    "secret",
                    "token",
                )
            )
            _assert_sanitized_log(item, key=normalized)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _assert_sanitized_log(item, key=key)
        return
    assert not isinstance(value, Path)
    if isinstance(value, str):
        assert not Path(value).is_absolute(), (key, value)
        assert not (len(value) > 2 and value[1] == ":" and value[2] in {"/", "\\"}), (
            key,
            value,
        )


def _write_and_read_log(path: Path, payload: dict[str, Any]) -> Mapping[str, Any]:
    payload["schema"] = "ipfs-accelerate.proof-context.v0.1/evidence-log"
    payload["task_id"] = "PCCE-075"
    payload["log_identity_rule"] = SCHEDULES["log_schema"]["log_identity_rule"]
    identity_payload = {**payload, "log_artifact_cid": None}
    payload["log_artifact_cid"] = recovery.mint_recovery_cid(
        {"kind": "full-sanitized-evidence-log", "log": identity_payload}
    )
    schema = SCHEDULES["log_schema"]
    assert set(schema["required"]) <= set(payload)
    assert isinstance(payload["events"], list)
    for sequence, event in enumerate(payload["events"], start=1):
        assert isinstance(event, Mapping)
        assert set(schema["event_required"]) <= set(event)
        assert event["sequence"] == sequence
    for section_name in ("process", "worktree", "lease", "fence", "cas", "checkpoints"):
        section = payload[section_name]
        assert isinstance(section, Mapping)
        assert set(schema["section_required"]) <= set(section)
        assert isinstance(section["applicable"], bool)
        assert isinstance(section["identities"], list)
        assert isinstance(section["observations"], Mapping)
        assert bool(section["identities"]) is section["applicable"]
    cleanup = payload["cleanup"]
    assert isinstance(cleanup, Mapping)
    assert set(schema["cleanup_required"]) <= set(cleanup)
    assert isinstance(cleanup["applicable"], bool)
    assert isinstance(cleanup["identities"], list)
    assert bool(cleanup["identities"]) is cleanup["applicable"]
    assert isinstance(cleanup["observations"], Mapping)
    assert isinstance(cleanup["errors"], list)
    assert isinstance(cleanup["registry"], list)
    assert all(
        isinstance(cleanup[name], bool)
        for name in ("processes_absent", "worktrees_absent", "storage_absent")
    )
    canonical_state = payload["canonical_state"]
    if canonical_state is not None:
        assert isinstance(canonical_state, Mapping)
        assert {
            "ref",
            "head",
            "head_tree",
            "index_tree",
            "porcelain",
        } <= set(canonical_state)
        assert canonical_state["porcelain"] == ""
    assert isinstance(payload["authority_observation"], Mapping)
    assert payload["authority_observation"]
    assert isinstance(payload["disposition"], str) and payload["disposition"]
    _assert_sanitized_log(payload)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    path.write_text(encoded, encoding="utf-8")
    observed = json.loads(path.read_text(encoding="utf-8"))
    assert (
        hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    observed_identity = {**observed, "log_artifact_cid": None}
    assert observed["log_artifact_cid"] == recovery.mint_recovery_cid(
        {"kind": "full-sanitized-evidence-log", "log": observed_identity}
    )
    return observed


def _write_worker_payload(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _spawned_probe_worker(
    tmp_path: Path,
    kind: str,
    parent_id: int,
    ready_path: Path,
    failure_path: Path,
    output_path: Path | None,
    identity_path: Path | None,
    returned_path: Path | None,
) -> None:
    """Run one closed worker operation without entering an ordinary test."""

    exit_code = 70
    try:
        parent_process = multiprocessing.parent_process()
        assert parent_process is not None
        assert parent_process.pid == parent_id == os.getppid()
        os.setsid()
        assert kind in PROBE_KINDS
        ready_path.write_text(
            json.dumps(
                {"pid": os.getpid(), "process_group": os.getpgrp()},
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        if kind == DOUBLE_PUBLICATION_PROBE:
            assert output_path is not None
            assert identity_path is None and returned_path is None
            _double_publication_worker_operation(tmp_path, output_path)
        elif kind == DUPLICATE_TERMINAL_PROBE:
            assert output_path is not None
            assert identity_path is None and returned_path is None
            _duplicate_terminal_worker_operation(tmp_path, output_path)
        else:
            assert kind == STUCK_BOUNDARY_PROBE
            assert output_path is None
            assert identity_path is not None and returned_path is not None
            _stuck_boundary_worker_operation(identity_path)
            returned_path.write_text("inner-probe-returned\n", encoding="utf-8")
            threading._shutdown()  # noqa: SLF001 - exercise the real exit join
        if output_path is not None:
            assert output_path.is_file()
        exit_code = 0
    except BaseException as exc:  # noqa: BLE001 - sanitize across boundary
        failure_path.write_text(
            json.dumps(
                {"error_type": type(exc).__name__},
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
    finally:
        os._exit(exit_code)


def _bounded_spawned_operation(
    tmp_path: Path,
    *,
    kind: str,
    output_path: Path | None,
    timeout: float,
    identity_path: Path | None = None,
    returned_path: Path | None = None,
) -> AuditedCompletedProcess:
    """Spawn a closed worker operation and reap its complete process group."""

    assert kind in PROBE_KINDS
    parent_id = os.getpid()
    ready_path = tmp_path / f".{kind}-spawn-ready.json"
    failure_path = tmp_path / f".{kind}-spawn-failure.json"
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        name=f"pcce075-{kind}-probe",
        target=_spawned_probe_worker,
        args=(
            tmp_path,
            kind,
            parent_id,
            ready_path,
            failure_path,
            output_path,
            identity_path,
            returned_path,
        ),
    )
    process.start()
    assert process.pid is not None
    process_id = process.pid
    deadline = time.monotonic() + timeout
    session_ready = False
    timed_out = False
    exit_code: int | None = None
    failure_type = ""
    process_absent = False
    group_lingered = False
    group_absent = False
    try:
        while time.monotonic() < deadline and process.is_alive():
            if ready_path.is_file():
                try:
                    ready = json.loads(ready_path.read_text(encoding="utf-8"))
                    session_ready = (
                        int(ready["pid"]) == process_id
                        and int(ready["process_group"]) == process_id
                    )
                except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                    session_ready = False
            process.join(
                timeout=min(
                    THREAD_POLL_SECONDS,
                    max(0.0, deadline - time.monotonic()),
                )
            )
        timed_out = process.is_alive()
        if not timed_out:
            process.join(timeout=0)
    finally:
        if process.is_alive():
            try:
                os.killpg(process_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                pass
            try:
                process.kill()
            except ProcessLookupError:
                pass
            except OSError:
                pass
            process.join(timeout=PROCESS_REAP_TIMEOUT_SECONDS)
        process_absent = not process.is_alive()
        if process_absent:
            exit_code = process.exitcode
        if not session_ready and ready_path.is_file():
            try:
                ready = json.loads(ready_path.read_text(encoding="utf-8"))
                session_ready = (
                    int(ready["pid"]) == process_id and int(ready["process_group"]) == process_id
                )
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                session_ready = False
        if failure_path.is_file():
            try:
                failure_type = str(
                    json.loads(failure_path.read_text(encoding="utf-8"))["error_type"]
                )
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                failure_type = "unreadable-sanitized-child-failure"
        group_lingered = not _process_group_absent(process_id)
        if group_lingered:
            try:
                os.killpg(process_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                pass
        group_absent = _wait_for_process_group_exit(process_id)
        for handoff_path in (ready_path, failure_path):
            try:
                handoff_path.unlink(missing_ok=True)
            except OSError:
                pass
        if process_absent:
            process.close()

    if timed_out:
        timeout_error = subprocess.TimeoutExpired(
            ["spawned-worker-operation", kind],
            timeout,
        )
        timeout_error.process_id = process_id  # type: ignore[attr-defined]
        timeout_error.process_group = process_id  # type: ignore[attr-defined]
        timeout_error.process_absent = process_absent  # type: ignore[attr-defined]
        timeout_error.process_group_absent = group_absent  # type: ignore[attr-defined]
        timeout_error.session_ready = session_ready  # type: ignore[attr-defined]
        raise timeout_error

    assert exit_code is not None
    completed = AuditedCompletedProcess(
        ["spawned-worker-operation", kind],
        exit_code,
        "",
        f"child_failure_type={failure_type}" if failure_type else "",
    )
    completed.process_id = process_id
    completed.process_group = process_id
    completed.process_absent = True
    completed.process_group_absent = group_absent
    completed.process_group_lingered = group_lingered
    completed.session_ready = session_ready  # type: ignore[attr-defined]
    return completed


def _bounded_spawn_probe(
    tmp_path: Path,
    *,
    kind: str,
    output_name: str,
) -> Mapping[str, Any]:
    """Execute a race probe in a killable child and finalize its parent audit."""

    worker_payload = tmp_path / f".{kind}-worker-payload.json"
    worker_root = tmp_path / f".{kind}-worker-root"
    worker_root.mkdir()
    schedule = SCHEDULES["bounded_probe_boundary"]
    artifact_cleanup_errors: list[str] = []
    try:
        completed = _bounded_spawned_operation(
            worker_root,
            kind=kind,
            output_path=worker_payload,
            timeout=float(schedule["successful_probe_timeout_seconds"]),
        )
        assert completed.returncode == 0, (completed.stdout, completed.stderr)
        assert completed.process_absent is True
        assert completed.process_group_absent is True
        assert completed.process_group_lingered is False
        assert completed.session_ready is True  # type: ignore[attr-defined]
        payload = json.loads(worker_payload.read_text(encoding="utf-8"))
    finally:
        try:
            worker_payload.unlink(missing_ok=True)
        except OSError as exc:
            artifact_cleanup_errors.append(f"worker-payload:{type(exc).__name__}")
        try:
            if worker_root.exists():
                shutil.rmtree(worker_root)
        except OSError as exc:
            artifact_cleanup_errors.append(f"worker-root:{type(exc).__name__}")
    assert artifact_cleanup_errors == []
    process_id = int(completed.process_id)
    process_group = int(completed.process_group)
    process_binding_cid = recovery.mint_recovery_cid(
        {
            "kind": "bounded-concurrency-probe-process",
            "probe_kind": kind,
            "pid": process_id,
            "process_group": process_group,
            "exit_code": completed.returncode,
        }
    )
    payload["process"] = _evidence_section(
        True,
        [
            {
                "process_binding_cid": process_binding_cid,
                "pid": process_id,
                "process_group": process_group,
            }
        ],
        {
            "exit_code": completed.returncode,
            "process_absent": True,
            "process_group_absent": True,
            "raw_streams_retained": False,
            "boundary": "parent-owned-killable-process-group",
            "child_mode_admission": (
                "dedicated-worker-operation-with-no-ordinary-test-mode"
            ),
        },
    )
    cleanup = dict(payload["cleanup"])
    cleanup["processes_absent"] = True
    cleanup["observations"] = {
        **dict(cleanup["observations"]),
        "probe_process_absent": True,
        "probe_process_group_absent": True,
        "raw_worker_payload_absent": not worker_payload.exists(),
        "raw_worker_root_absent": not worker_root.exists(),
    }
    payload["cleanup"] = cleanup
    payload["probe_boundary"] = {
        "kind": kind,
        "timeout_seconds": schedule["successful_probe_timeout_seconds"],
        "process_binding_cid": process_binding_cid,
        "child_mode_admission": (
            "dedicated-worker-operation-with-no-ordinary-test-mode"
        ),
        "disposition": "bounded-child-completed",
    }
    payload.pop("log_artifact_cid", None)
    return _write_and_read_log(tmp_path / output_name, payload)


class ManualClock:
    def __init__(self, now: int = 0) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now


def test_schedule_fixture_is_closed_exact_and_bound_to_the_production_base() -> None:
    fixture = _schedule_fixture()
    assert fixture["schema"].endswith("adversarial-concurrency-schedules")
    assert fixture["task_id"] == "PCCE-075"
    assert len(str(fixture["production_source_commit"])) == 40
    ancestor = _git(
        PACKAGE_ROOT,
        "merge-base",
        "--is-ancestor",
        str(fixture["production_source_commit"]),
        "HEAD",
    )
    assert ancestor.returncode == 0
    cases = tuple((stage, position) for _, _, stage, position, _ in CRASH_CASES)
    assert cases == recovery.CRASH_MATRIX
    seeds = (
        [
            int(fixture[name]["seed"])
            for name in (
                "concurrent_publication",
                "aba",
                "duplicate_terminal",
                "authoritative_integration",
                "bounded_probe_boundary",
                "process_crash",
            )
        ]
        + [int(seed) for seed in fixture["lease_and_fence_loss"]["seeds"].values()]
        + [seed for _, seed, _, _, _ in CRASH_CASES]
    )
    assert len(seeds) == len(set(seeds))
    assert fixture["process_crash"]["timeout_seconds"] <= 30
    assert fixture["bounded_probe_boundary"]["timeout_seconds"] <= 3
    assert fixture["bounded_probe_boundary"]["successful_probe_timeout_seconds"] <= 30
    assert fixture["bounded_probe_boundary"]["ambient_forgery_admitted"] is False
    # The schedule and executable evidence agree on the closed v8 boundary:
    # spawned children can dispatch only dedicated worker operations.
    assert fixture["bounded_probe_boundary"]["child_mode_admission"] == (
        "dedicated-worker-operation-with-no-ordinary-test-mode"
    )
    assert recovery.INFER_SUCCESS_FROM_PROCESS_EXIT is False
    assert recovery.PERSISTENCE_AUTHORITY == "injected-kit-checkpoint-store"
    concurrent = fixture["concurrent_publication"]
    assert concurrent["security_invariant"]["maximum_accepted_records"] == 1
    assert concurrent["observed_current_tree"]["accepted_records"] == 2
    assert concurrent["observed_current_tree"]["overlap_proven"] is True
    assert concurrent["observed_current_tree"]["completion_order_recorded"] is True
    assert concurrent["observed_current_tree"]["disposition"].startswith("no-go-")
    assert "In-process" in (recovery.FencedCheckpointStore.__doc__ or "")
    process = fixture["process_crash"]
    assert process["no_cleanup_restart_observation"]["repair_required"] is False
    assert process["post_cleanup_fresh_restart_observation"]["accepted"] is True
    integration = fixture["authoritative_integration"]["observed_current_tree"]
    assert integration["proof_execution"] == "unavailable-no-go"
    assert fixture["aba"]["qualification_credit"] is False
    assert fixture["aba"]["disposition"].startswith("no-go-insufficient-")
    required = set(fixture["log_schema"]["required"])
    assert {
        "process",
        "worktree",
        "lease",
        "fence",
        "cas",
        "checkpoints",
        "log_artifact_cid",
    } <= required


def test_separate_process_cannot_manufacture_spawn_probe_admission() -> None:
    """Ambient inputs and forked monkeypatch state expose no test-mode API."""

    attacker_script = """
import importlib.util
import json
import multiprocessing
import os
import sys

kind = "double-publication"
value = "attacker-chosen"
read_descriptor, write_descriptor = os.pipe()
try:
    os.write(write_descriptor, f"{kind}:{value}".encode("ascii"))
finally:
    os.close(write_descriptor)
os.environ["PCCE075_BOUNDED_PROCESS_GROUP_CHILD"] = "1"
os.environ["PCCE075_CONCURRENCY_PROBE"] = kind
os.environ["PCCE075_CONCURRENCY_PROBE_CAPABILITY_FD"] = str(read_descriptor)
os.environ["PCCE075_CONCURRENCY_PROBE_CAPABILITY_VALUE"] = value
try:
    spec = importlib.util.spec_from_file_location("external_pcce075_probe", sys.argv[1])
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    forbidden = {
        "_admitted_probe_kind",
        "_inside_concurrency_probe",
        "_PROBE_ADMISSION_SENTINEL",
        "_ACTIVE_PROBE_ADMISSION",
        "_ACTIVE_PROBE_OUTPUT_PATH",
        "_ACTIVE_PROBE_IDENTITY_PATH",
    }
    ordinary_tests = (
        module.test_minimized_real_runtime_double_publication_is_preserved_as_no_go,
        module.test_concurrent_duplicate_terminal_resume_is_idempotent,
        module.test_bounded_probe_process_group_kills_interpreter_exit_hang_and_descendant,
    )

    def fork_attack(connection):
        for name in forbidden:
            setattr(module, name, (object(), kind, os.getppid(), os.getpid()))
        connection.send(
            {
                "original_admission_surface": original_surface,
                "ordinary_test_switch_references": sorted(
                    forbidden
                    & {
                        referenced
                        for test in ordinary_tests
                        for referenced in test.__code__.co_names
                    }
                ),
                "worker_test_dispatch_references": sorted(
                    set(module._spawned_probe_worker.__code__.co_names)
                    & {test.__name__ for test in ordinary_tests}
                ),
                "worker_uses_dynamic_globals": (
                    "globals" in module._spawned_probe_worker.__code__.co_names
                ),
            }
        )
        connection.close()

    original_surface = sorted(forbidden & set(module.__dict__))
    context = multiprocessing.get_context("fork")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(target=fork_attack, args=(child_connection,))
    process.start()
    child_connection.close()
    result = parent_connection.recv()
    process.join(2)
    result["fork_exitcode"] = process.exitcode
    result["original_admission_surface"] = original_surface
    process.close()
    parent_connection.close()
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
finally:
    os.close(read_descriptor)
"""
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": os.environ.get("PYTHONPATH", str(PACKAGE_ROOT)),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }
    completed = _bounded_process(
        [sys.executable, "-c", attacker_script, str(Path(__file__).resolve())],
        cwd=PACKAGE_ROOT,
        env=environment,
        timeout=5.0,
    )
    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    assert json.loads(completed.stdout) == {
        "fork_exitcode": 0,
        "original_admission_surface": [],
        "ordinary_test_switch_references": [],
        "worker_test_dispatch_references": [],
        "worker_uses_dynamic_globals": False,
    }
    assert completed.process_absent is True
    assert completed.process_group_absent is True
    assert completed.process_group_lingered is False


def test_direct_worker_target_has_no_ordinary_test_dispatch_surface() -> None:
    """The former direct-target exploit can select only closed operations."""

    target_code = _spawned_probe_worker.__code__
    arguments = target_code.co_varnames[: target_code.co_argcount]
    ordinary_test_names = {
        test_minimized_real_runtime_double_publication_is_preserved_as_no_go.__name__,
        test_concurrent_duplicate_terminal_resume_is_idempotent.__name__,
        test_bounded_probe_process_group_kills_interpreter_exit_hang_and_descendant.__name__,
    }
    assert "node_id" not in arguments
    assert "globals" not in target_code.co_names
    assert ordinary_test_names.isdisjoint(target_code.co_names)
    assert PROBE_KINDS == {
        "double-publication",
        "duplicate-terminal",
        "stuck-boundary",
    }


def test_production_bootstrap_git_timeout_kills_the_real_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove the installed test boundary kills a hung Git command and descendants."""

    _synchronize_runtime_modules()
    executable = tmp_path / "hanging-git"
    process_identity = tmp_path / "process-group"
    executable.write_text(
        '#!/bin/sh\nprintf "%s" "$$" > "$PCCE075_HANG_PROCESS_GROUP"\n/bin/sleep 60 &\nwait\n',
        encoding="utf-8",
    )
    executable.chmod(0o700)
    base_environment = bootstrap._git_env()
    monkeypatch.setattr(bootstrap, "_git_executable", lambda: str(executable))
    monkeypatch.setattr(
        bootstrap,
        "_git_env",
        lambda: {
            **base_environment,
            "PCCE075_HANG_PROCESS_GROUP": str(process_identity),
        },
    )
    monkeypatch.setattr(
        sys.modules[__name__],
        "BOOTSTRAP_GIT_TIMEOUT_SECONDS",
        0.2,
    )
    started = time.monotonic()
    with pytest.raises(bootstrap.UnavailableCapabilityError) as unavailable:
        bootstrap._git(tmp_path, "status")
    elapsed = time.monotonic() - started
    assert unavailable.value.code == "unavailable_capability"
    assert unavailable.value.details["reason"] == ("bounded-timeout-process-group-absent")
    assert elapsed < 5
    process_group = int(process_identity.read_text(encoding="utf-8"))
    assert _process_group_absent(process_group)


def _stuck_boundary_worker_operation(identity_path: Path) -> None:
    """Create only the intended stuck shutdown and inherited descendant."""

    blocker = threading.Event()
    stuck_call = threading.Thread(
        target=blocker.wait,
        name="pcce075-intentionally-stuck-call",
        daemon=False,
    )
    stuck_call.start()
    descendant = subprocess.Popen(
        ["/bin/sleep", "60"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    identity_path.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "process_group": os.getpgrp(),
                "descendant_pid": descendant.pid,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def test_bounded_probe_process_group_kills_interpreter_exit_hang_and_descendant(
    tmp_path: Path,
) -> None:
    """A stuck call and non-daemon shutdown join cannot outlive the parent bound."""

    schedule = SCHEDULES["bounded_probe_boundary"]

    identity_path = tmp_path / ".stuck-probe-identity.json"
    returned_path = tmp_path / ".stuck-probe-returned"
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired) as timed_out:
        _bounded_spawned_operation(
            tmp_path,
            kind=STUCK_BOUNDARY_PROBE,
            output_path=None,
            timeout=float(schedule["timeout_seconds"]),
            identity_path=identity_path,
            returned_path=returned_path,
        )
    elapsed = time.monotonic() - started
    assert elapsed < schedule["timeout_seconds"] + PROCESS_REAP_TIMEOUT_SECONDS + 2
    assert bool(getattr(timed_out.value, "process_group_absent", False)) is True
    assert bool(getattr(timed_out.value, "process_absent", False)) is True
    assert bool(getattr(timed_out.value, "session_ready", False)) is True
    process_id = -1
    process_group = -1
    descendant_id = -1
    handoff_cleanup_errors: list[str] = []
    try:
        assert returned_path.read_text(encoding="utf-8") == "inner-probe-returned\n"
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        process_id = int(identity["pid"])
        process_group = int(identity["process_group"])
        descendant_id = int(identity["descendant_pid"])
        assert (
            process_id
            == process_group
            == int(
                timed_out.value.process_group  # type: ignore[attr-defined]
            )
        )
        assert _wait_for_process_exit(process_id)
        assert _wait_for_process_exit(descendant_id)
        assert _process_group_absent(process_group)
    finally:
        for handoff_path in (identity_path, returned_path):
            try:
                handoff_path.unlink(missing_ok=True)
            except OSError as exc:
                handoff_cleanup_errors.append(type(exc).__name__)
    assert handoff_cleanup_errors == []

    process_binding_cid = recovery.mint_recovery_cid(
        {
            "kind": "stuck-probe-process-binding",
            "pid": process_id,
            "process_group": process_group,
        }
    )
    descendant_binding_cid = recovery.mint_recovery_cid(
        {
            "kind": "stuck-probe-descendant-binding",
            "pid": descendant_id,
            "process_group": process_group,
        }
    )
    log = {
        "schedule_id": schedule["schedule_id"],
        "seed": schedule["seed"],
        "events": [
            {
                "sequence": 1,
                "actor": "probe-child",
                "operation": "return-with-stuck-non-daemon-call-and-descendant",
                "generation": 1,
                "outcome": "interpreter-exit-blocked",
            },
            {
                "sequence": 2,
                "actor": "parent-boundary",
                "operation": "deadline-kill-process-group",
                "generation": 1,
                "outcome": schedule["expected_outcome"],
            },
        ],
        "parent_root_cid": None,
        "canonical_state": None,
        "seal_cid": None,
        "published_count": 0,
        "process": _evidence_section(
            True,
            [
                {
                    "process_binding_cid": process_binding_cid,
                    "pid": process_id,
                    "process_group": process_group,
                },
                {
                    "process_binding_cid": descendant_binding_cid,
                    "pid": descendant_id,
                    "process_group": process_group,
                },
            ],
            {
                "timeout_seconds": schedule["timeout_seconds"],
                "controller_absent": True,
                "descendant_absent": True,
                "process_group_absent": True,
                "inner_probe_returned_before_shutdown_timeout": True,
                "raw_streams_retained": False,
            },
        ),
        "worktree": _evidence_section(
            False,
            (),
            {"reason": "boundary-regression-created-no-repository-or-worktree"},
        ),
        "lease": _evidence_section(
            False,
            (),
            {"reason": "boundary-regression-created-no-runtime-lease"},
        ),
        "fence": _evidence_section(
            False,
            (),
            {"reason": "boundary-regression-created-no-runtime-fence"},
        ),
        "cas": _evidence_section(
            False,
            (),
            {"reason": "boundary-regression-created-no-checkpoint-store"},
        ),
        "checkpoints": _evidence_section(
            False,
            (),
            {"reason": "boundary-regression-created-no-checkpoints"},
        ),
        "authority_observation": {
            "boundary": "parent-owned-killable-process-group",
            "child_mode_admission": (
                "dedicated-worker-operation-with-no-ordinary-test-mode"
            ),
            "stuck_effect": schedule["stuck_effect"],
            "timeout_enforced": True,
        },
        "cleanup": {
            "applicable": True,
            "identities": [
                {"process_binding_cid": process_binding_cid},
                {"process_binding_cid": descendant_binding_cid},
            ],
            "observations": {
                "raw_identity_handoff_absent": not identity_path.exists(),
                "raw_return_handoff_absent": not returned_path.exists(),
                "process_group_audited": True,
            },
            "errors": [],
            "processes_absent": True,
            "worktrees_absent": True,
            "registry": [],
            "storage_absent": True,
        },
        "disposition": "evidence-only-killable-probe-boundary",
    }
    assert _write_and_read_log(tmp_path / "stuck-boundary-log.json", log) == log


def _double_publication_worker_operation(
    tmp_path: Path,
    output_path: Path,
) -> None:
    schedule = SCHEDULES["concurrent_publication"]
    _synchronize_runtime_modules(inherit_process_group=True)
    root = tmp_path / "two-writers"
    repository = root / "repository"
    storage = root / "runtime-state"
    kit_root = storage / "kit"
    canonical_before: Mapping[str, str] = {}
    writers = tuple(str(item) for item in schedule["writers"])
    bundles: list[bootstrap.RuntimeBundle] = []
    outcomes: list[dict[str, Any]] = []
    records_by_writer: dict[str, recovery.RecoveryRecord] = {}
    completion_order: list[str] = []
    attempts: list[recovery.AttemptIdentity] = []
    authorities: list[Mapping[str, Any]] = []
    barrier = threading.Barrier(len(writers) + 1)
    overlap_lock = threading.Lock()
    active_callers = 0
    max_active_callers = 0
    concurrency_cleanup: Mapping[str, Any] = {}
    cleanup: Mapping[str, Any] = {}
    try:
        repository = bootstrap.create_ordinary_python_repository(repository)
        canonical_before = _canonical_snapshot(
            repository,
            inherit_process_group=True,
        )
        assert canonical_before["porcelain"] == ""
        for writer in writers:
            bundle = bootstrap.open_runtime(
                repository,
                options=bootstrap.RuntimeOptions(
                    kit_root=kit_root,
                    worktree_parent=storage / f"worktrees-{writer}",
                    require_kit=True,
                ),
            )
            bundles.append(bundle)
            authorities.append(_assert_actual_kit(bundle))
            attempt = _attempt(bundle, writer_id=writer)
            attempts.append(attempt)

        def publish(index: int) -> dict[str, Any]:
            nonlocal active_callers, max_active_callers
            bundle = bundles[index]
            attempt = attempts[index]
            writer = writers[index]
            with overlap_lock:
                active_callers += 1
                max_active_callers = max(max_active_callers, active_callers)
            try:
                barrier.wait(timeout=10)
                record = _coordinator(
                    repository,
                    bundle,
                    bundle.session.store,
                    attempt,
                ).run(PATCH)
                published = _published_settlements(bundle.session.store, attempt.attempt_id)
                seal_cid = _seal_cid(record)
                assert seal_cid is not None
                outcome = {
                    "writer": writer,
                    "status": record.status,
                    "accepted": record.accepted,
                    "published": record.published,
                    "evidence_cid": record.evidence_cid,
                    "seal_cid": seal_cid,
                    "published_settlements": len(published),
                }
                with overlap_lock:
                    records_by_writer[writer] = record
                return outcome
            finally:
                with overlap_lock:
                    active_callers -= 1

        completed_calls, concurrency_cleanup = _run_child_probe_threads(
            [(writers[index], lambda index=index: publish(index)) for index in range(len(writers))],
            release=lambda: barrier.wait(timeout=10),
            thread_prefix="pcce075-two-runtime",
        )
        completion_order = [writer for writer, _ in completed_calls]
        outcomes = [outcome for _, outcome in completed_calls]
        assert (
            _canonical_snapshot(repository, inherit_process_group=True)
            == canonical_before
        )

        assert len({attempt.attempt_id for attempt in attempts}) == 1
        assert len({attempt.writer_generation for attempt in attempts}) == 1
        assert len({attempt.fence_token for attempt in attempts}) == 1
        assert bundles[0].session.store is not bundles[1].session.store
        assert bundles[0].session.kit_store is not bundles[1].session.kit_store
        assert authorities[0] == authorities[1]
        assert max_active_callers == len(writers)
        assert active_callers == 0
        assert sorted(completion_order) == sorted(writers)
        observed = {
            "accepted_records": sum(item["accepted"] is True for item in outcomes),
            "published_settlements": sum(int(item["published_settlements"]) for item in outcomes),
            "distinct_evidence_cids": len({item["evidence_cid"] for item in outcomes})
            == len(outcomes),
            "distinct_runtime_checkpoint_stores": True,
            "shared_kit_root": True,
            "simultaneous_barrier_release": True,
            "overlap_proven": max_active_callers == len(writers),
            "completion_order_recorded": len(completion_order) == len(writers),
            "disposition": "no-go-runtime-storage-owner-reopen-required",
        }
        assert observed == schedule["observed_current_tree"]
        assert (
            observed["accepted_records"]
            > schedule["security_invariant"]["maximum_accepted_records"]
        )
        assert (
            observed["published_settlements"]
            > schedule["security_invariant"]["maximum_published_settlements"]
        )
        assert (
            _canonical_snapshot(repository, inherit_process_group=True)
            == canonical_before
        )
    finally:
        if (repository / ".git").exists():
            cleanup = _cleanup_runtimes(
                repository,
                storage,
                bundles,
                inherit_process_group=True,
            )
            _assert_clean(cleanup, repository)
        elif storage.exists():
            shutil.rmtree(storage)
        assert concurrency_cleanup == {
            "errors": [],
            "calls_done": True,
            "threads_absent": True,
            "termination_authority": "parent-owned-killable-process-group",
        }
        if canonical_before:
            assert (
                _canonical_snapshot(repository, inherit_process_group=True)
                == canonical_before
            )
    attempt_by_writer = {attempt.writer_id: attempt for attempt in attempts}
    events = [
        {
            "sequence": 1,
            "actor": "scheduler",
            "operation": "release-two-runtime-barrier",
            "generation": attempts[0].writer_generation,
            "outcome": "simultaneous",
        },
        *[
            {
                "sequence": index + 1,
                "actor": item["writer"],
                "operation": "concurrent-run-publish",
                "generation": attempt_by_writer[item["writer"]].writer_generation,
                "outcome": f"accepted:{item['evidence_cid']}",
            }
            for index, item in enumerate(outcomes, start=1)
        ],
    ]
    sections = _coordination_sections(
        [
            (
                bundle,
                attempt,
                bundle.session.store,
                records_by_writer[attempt.writer_id],
            )
            for bundle, attempt in zip(bundles, attempts, strict=True)
        ]
    )
    log = {
        "schedule_id": schedule["schedule_id"],
        "seed": schedule["seed"],
        "events": events,
        "parent_root_cid": attempts[0].identities.repository_state_cid,
        "canonical_state": canonical_before,
        "seal_cid": [item["seal_cid"] for item in outcomes],
        "published_count": sum(int(item["published_settlements"]) for item in outcomes),
        "security_invariant": schedule["security_invariant"],
        "observed_current_tree": schedule["observed_current_tree"],
        "completion_order": completion_order,
        "synchronization_observation": {
            "start_barrier_parties": barrier.parties,
            "max_active_callers": max_active_callers,
            "active_callers_after": active_callers,
            "completion_order": completion_order,
            "completion_order_source": "condition-record-after-real-run-return",
        },
        "authority_observation": {
            "per_runtime": authorities,
            "all_bindings_equal": authorities[0] == authorities[1],
        },
        "checkpoint_store_objects_distinct": bundles[0].session.store
        is not bundles[1].session.store,
        "cleanup": {**cleanup, "concurrency_workers": concurrency_cleanup},
        "disposition": schedule["observed_current_tree"]["disposition"],
        **sections,
    }
    observed_log = _write_and_read_log(tmp_path / "two-writers-log.json", log)
    assert observed_log == log
    _write_worker_payload(output_path, observed_log)


def test_minimized_real_runtime_double_publication_is_preserved_as_no_go(
    tmp_path: Path,
) -> None:
    schedule = SCHEDULES["concurrent_publication"]
    log = _bounded_spawn_probe(
        tmp_path,
        kind=DOUBLE_PUBLICATION_PROBE,
        output_name="two-writers-log.json",
    )
    assert log["observed_current_tree"] == schedule["observed_current_tree"]
    assert log["published_count"] == 2
    assert len(log["completion_order"]) == len(schedule["writers"])
    assert log["process"]["observations"]["process_group_absent"] is True
    assert log["cleanup"]["processes_absent"] is True


def _cas_record(
    *,
    attempt_id: str,
    run_id: str,
    trace_id: str,
    generation: int,
    label: str,
) -> Mapping[str, Any]:
    return {
        "attempt_id": attempt_id,
        "stage": recovery.LEASE_BOUNDARY,
        "position": "after",
        "label": label,
        "idempotency_key": recovery.mint_idempotency_key(
            attempt_id=attempt_id,
            run_id=run_id,
            trace_id=trace_id,
            stage=recovery.LEASE_BOUNDARY,
            position="after",
            inbound_cid=None,
            generation=generation,
        ),
    }


def test_synthetic_aba_store_probe_is_insufficient_qualification_no_go(
    tmp_path: Path,
) -> None:
    schedule = SCHEDULES["aba"]
    store = recovery.FencedCheckpointStore()
    attempt_id = "attempt-pcce-075-aba"
    run_id = "run-pcce-075-aba"
    trace_id = "trace-pcce-075-aba"
    token_a = recovery.mint_recovery_cid({"token": "a"})
    token_b = recovery.mint_recovery_cid({"token": "b"})
    events: list[dict[str, Any]] = []

    store.put(
        _cas_record(
            attempt_id=attempt_id,
            run_id=run_id,
            trace_id=trace_id,
            generation=1,
            label="a-generation-1",
        ),
        writer_id="writer-a",
        generation=1,
        fence_token=token_a,
    )
    events.append(
        {
            "sequence": 1,
            "actor": "writer-a",
            "operation": "claim",
            "generation": 1,
            "outcome": "stored",
        }
    )

    generation_two = store.reclaim(
        attempt_id,
        writer_id="writer-b",
        fence_token=token_b,
    )
    store.put(
        _cas_record(
            attempt_id=attempt_id,
            run_id=run_id,
            trace_id=trace_id,
            generation=generation_two,
            label="b-generation-2",
        ),
        writer_id="writer-b",
        generation=generation_two,
        fence_token=token_b,
    )
    events.append(
        {
            "sequence": 2,
            "actor": "writer-b",
            "operation": "reclaim-and-put",
            "generation": generation_two,
            "outcome": "stored",
        }
    )
    generation_three = store.reclaim(
        attempt_id,
        writer_id="writer-a",
        fence_token=token_a,
    )
    store.put(
        _cas_record(
            attempt_id=attempt_id,
            run_id=run_id,
            trace_id=trace_id,
            generation=generation_three,
            label="a-generation-3",
        ),
        writer_id="writer-a",
        generation=generation_three,
        fence_token=token_a,
    )
    events.append(
        {
            "sequence": 3,
            "actor": "writer-a",
            "operation": "aba-reclaim-and-put",
            "generation": generation_three,
            "outcome": "stored",
        }
    )
    try:
        store.put(
            _cas_record(
                attempt_id=attempt_id,
                run_id=run_id,
                trace_id=trace_id,
                generation=1,
                label="delayed-a-generation-1",
            ),
            writer_id="writer-a",
            generation=1,
            fence_token=token_a,
        )
    except recovery.StaleWriterError as exc:
        delayed_outcome = f"{exc.code}:{exc.details['reason']}"
    else:
        delayed_outcome = "unexpectedly-stored"

    events.append(
        {
            "sequence": 4,
            "actor": "writer-a-delayed",
            "operation": "put",
            "generation": 1,
            "outcome": delayed_outcome,
        }
    )
    assert delayed_outcome == schedule["expected_delayed_outcome"]
    assert (generation_two, generation_three) == (2, 3)
    assert store.current_writer(attempt_id) == "writer-a"
    assert store.current_token(attempt_id) == token_a
    assert store.current_generation(attempt_id) == 3
    assert schedule["integration_scope"] == "synthetic-in-process-store-only"
    assert schedule["qualification_credit"] is False
    checkpoint_cids = [str(item["checkpoint_cid"]) for item in store.history(attempt_id)]
    store_binding_cid = recovery.mint_recovery_cid(
        {
            "kind": "synthetic-store-binding",
            "attempt_id": attempt_id,
            "store_class": _class_identity(store),
        }
    )
    log = {
        "schedule_id": schedule["schedule_id"],
        "seed": schedule["seed"],
        "events": events,
        "parent_root_cid": None,
        "canonical_state": None,
        "seal_cid": None,
        "published_count": 0,
        "process": _evidence_section(
            False,
            (),
            {"reason": "synthetic-store-method-schedule-only"},
        ),
        "worktree": _evidence_section(
            False,
            (),
            {"reason": "no-runtime-repository-or-worktree-was-opened"},
        ),
        "lease": _evidence_section(
            False,
            (),
            {"reason": "bare-store-probe-had-no-coordinator-lease"},
        ),
        "fence": _evidence_section(
            True,
            [
                {
                    "attempt_id": attempt_id,
                    "fence_cid": token_a,
                    "generation": 1,
                    "writer_id": "writer-a",
                },
                {
                    "attempt_id": attempt_id,
                    "fence_cid": token_b,
                    "generation": 2,
                    "writer_id": "writer-b",
                },
                {
                    "attempt_id": attempt_id,
                    "fence_cid": token_a,
                    "generation": 3,
                    "writer_id": "writer-a",
                },
            ],
            {"delayed_aba_outcome": delayed_outcome},
        ),
        "cas": _evidence_section(
            True,
            [
                {
                    "store_binding_cid": store_binding_cid,
                    "attempt_id": attempt_id,
                }
            ],
            {
                "actual_store_class": _class_identity(store),
                "scope": schedule["integration_scope"],
                "generation": store.current_generation(attempt_id),
                "writer_id": store.current_writer(attempt_id),
                "fence_cid": store.current_token(attempt_id),
            },
        ),
        "checkpoints": _evidence_section(
            True,
            [
                {
                    "store_binding_cid": store_binding_cid,
                    "checkpoint_cid": item,
                }
                for item in checkpoint_cids
            ],
            {"count": len(checkpoint_cids)},
        ),
        "authority_observation": {
            "declared_checkpoint_authority": recovery.PERSISTENCE_AUTHORITY,
            "actual_checkpoint_store_class": _class_identity(store),
            "actual_checkpoint_scope": schedule["integration_scope"],
            "kit_capability_exercised": False,
            "repository_exercised": False,
        },
        "cleanup": {
            "applicable": False,
            "identities": [],
            "observations": {"reason": "bare-store-probe-created-no-process-worktree-or-storage"},
            "errors": [],
            "processes_absent": True,
            "worktrees_absent": True,
            "registry": [],
            "storage_absent": True,
            "filesystem_created": False,
        },
        "disposition": schedule["disposition"],
    }
    assert _write_and_read_log(tmp_path / "aba-log.json", log) == log


@pytest.mark.parametrize("loss", ("lease", "fence"))
def test_lease_or_fence_loss_after_seal_cannot_publish(tmp_path: Path, loss: str) -> None:
    schedule = SCHEDULES["lease_and_fence_loss"]
    repository, storage, bundle = _open_production_runtime(tmp_path / f"loss-{loss}")
    canonical_before: Mapping[str, str] = {}
    store = recovery.FencedCheckpointStore()
    clock = ManualClock()
    attempt = _attempt(bundle, writer_id="writer-current", lease_expires_at=10)
    seal_cid: str | None = None
    events: list[dict[str, Any]] = []
    try:
        canonical_before = _canonical_snapshot(repository)
        coordinator = _coordinator(repository, bundle, store, attempt, clock=clock)
        coordinator.inject_crash(
            str(schedule["crash_stage"]),
            str(schedule["crash_position"]),
        )
        with pytest.raises(_crash_interrupt_type()):
            coordinator.run(PATCH)
        assert store.invocation_count(attempt.attempt_id, lifecycle.SEAL_STAGE) == 1
        seal_cid = _seal_cid_from_history(store, attempt.attempt_id)
        assert seal_cid is not None
        events.append(
            {
                "sequence": 1,
                "actor": attempt.writer_id,
                "operation": "seal:after-crash",
                "generation": attempt.writer_generation,
                "outcome": f"sealed:{seal_cid}",
            }
        )

        if loss == "lease":
            clock.now = 10
        else:
            successor = recovery.mint_recovery_cid({"token": "successor"})
            store.reclaim(
                attempt.attempt_id,
                writer_id="writer-successor",
                fence_token=successor,
            )
        events.append(
            {
                "sequence": 2,
                "actor": "clock" if loss == "lease" else "writer-successor",
                "operation": f"{loss}-loss",
                "generation": int(store.current_generation(attempt.attempt_id) or 0),
                "outcome": "fenced",
            }
        )
        restarted = _coordinator(repository, bundle, store, attempt, clock=clock)
        record = restarted.resume()
        assert record.status == schedule["expected_status"] == "stale"
        assert record.error == "stale_root"
        assert record.published is schedule["expected_published"] is False
        assert record.accepted is False
        assert _published_settlements(store, attempt.attempt_id) == []
        assert store.invocation_count(attempt.attempt_id, lifecycle.SEAL_STAGE) == 1
        assert _canonical_snapshot(repository) == canonical_before
        events.append(
            {
                "sequence": 3,
                "actor": attempt.writer_id,
                "operation": "resume",
                "generation": attempt.writer_generation,
                "outcome": f"{record.status}:{record.error}",
            }
        )
    finally:
        cleanup = _cleanup_runtimes(repository, storage, [bundle])
        _assert_clean(cleanup, repository)
        if canonical_before:
            assert _canonical_snapshot(repository) == canonical_before

    sections = _coordination_sections([(bundle, attempt, store, record)])
    log = {
        "schedule_id": f"{schedule['schedule_id']}-{loss}",
        "seed": int(schedule["seeds"][loss]),
        "events": events,
        "parent_root_cid": attempt.identities.repository_state_cid,
        "canonical_state": canonical_before,
        "seal_cid": seal_cid,
        "published_count": 0,
        "loss_observation": {
            "kind": loss,
            "expires_at": attempt.lease_expires_at,
            "clock_at_resume": clock.now,
            "valid_at_resume": loss != "lease",
        },
        "authority_observation": _assert_actual_kit(bundle, store),
        "cleanup": cleanup,
        "disposition": "evidence-only-in-process-not-task-approval",
        **sections,
    }
    assert _write_and_read_log(tmp_path / f"{loss}-loss-log.json", log) == log


def _duplicate_terminal_worker_operation(
    tmp_path: Path,
    output_path: Path,
) -> None:
    schedule = SCHEDULES["duplicate_terminal"]
    repository, storage, bundle = _open_production_runtime(
        tmp_path / "terminal",
        inherit_process_group=True,
    )
    canonical_before: Mapping[str, str] = {}
    store = recovery.FencedCheckpointStore()
    attempt = _attempt(bundle, writer_id="writer-terminal")
    caller_count = int(schedule["resume_callers"])
    ready_barrier = threading.Barrier(caller_count + 1)
    call_barrier = threading.Barrier(caller_count)
    state_lock = threading.Lock()
    active_callers = 0
    max_active_callers = 0
    completion_order: list[tuple[str, recovery.RecoveryRecord]] = []
    concurrency_cleanup: Mapping[str, Any] = {}
    first: recovery.RecoveryRecord | None = None
    resumed: list[recovery.RecoveryRecord] = []
    adapter_result_identity: str | None = None
    try:
        canonical_before = _canonical_snapshot(
            repository,
            inherit_process_group=True,
        )
        first = _coordinator(repository, bundle, store, attempt).run(PATCH)
        assert first.accepted is True
        assert _seal_cid(first) is not None
        history_before = tuple(store.history(attempt.attempt_id))
        coordinators = [
            _coordinator(repository, bundle, store, attempt) for _ in range(caller_count)
        ]

        def concurrent_resume(
            caller: str,
            coordinator: recovery.RecoveryCoordinator,
        ) -> recovery.RecoveryRecord:
            nonlocal active_callers, max_active_callers
            with state_lock:
                active_callers += 1
                max_active_callers = max(max_active_callers, active_callers)
            try:
                ready_barrier.wait(timeout=10)
                call_barrier.wait(timeout=10)
                return coordinator.resume()
            finally:
                with state_lock:
                    active_callers -= 1

        completion_order, concurrency_cleanup = _run_child_probe_threads(
            [
                (
                    f"resume-caller-{index}",
                    lambda caller=f"resume-caller-{index}", coordinator=coordinator: (
                        concurrent_resume(caller, coordinator)
                    ),
                )
                for index, coordinator in enumerate(coordinators, start=1)
            ],
            release=lambda: ready_barrier.wait(timeout=10),
            thread_prefix="pcce075-terminal",
        )
        resumed = [record for _, record in completion_order]

        assert all(record.accepted is True for record in resumed)
        assert max_active_callers == schedule["synchronization"]["required_max_active_callers"]
        assert active_callers == 0
        assert len(completion_order) == caller_count
        exact_results = [record.to_mapping() for record in (first, *resumed)]
        assert all(result == exact_results[0] for result in exact_results[1:])
        adapter_result_identity = recovery.mint_recovery_cid(
            {"kind": "identical-adapter-result", "record": exact_results[0]}
        )
        assert {record.evidence_cid for record in resumed} == {first.evidence_cid}
        assert tuple(store.history(attempt.attempt_id)) == history_before
        invariants = schedule["invariants"]
        assert (
            store.invocation_count(attempt.attempt_id, lifecycle.SEAL_STAGE)
            == invariants["seal_invocations"]
        )
        assert (
            store.invocation_count(attempt.attempt_id, lifecycle.DISPOSITION_STAGE)
            == invariants["disposition_invocations"]
        )
        assert (
            len(_published_settlements(store, attempt.attempt_id))
            == invariants["published_settlements"]
        )
        assert invariants["identical_adapter_result_identity"] is True
        assert (
            _canonical_snapshot(repository, inherit_process_group=True)
            == canonical_before
        )
    finally:
        cleanup = _cleanup_runtimes(
            repository,
            storage,
            [bundle],
            inherit_process_group=True,
        )
        _assert_clean(cleanup, repository)
        assert concurrency_cleanup == {
            "errors": [],
            "calls_done": True,
            "threads_absent": True,
            "termination_authority": "parent-owned-killable-process-group",
        }
        if canonical_before:
            assert (
                _canonical_snapshot(repository, inherit_process_group=True)
                == canonical_before
            )

    assert first is not None
    sections = _coordination_sections([(bundle, attempt, store, first)])
    log = {
        "schedule_id": schedule["schedule_id"],
        "seed": schedule["seed"],
        "events": [
            {
                "sequence": 1,
                "actor": "writer-terminal",
                "operation": "run",
                "generation": attempt.writer_generation,
                "outcome": f"accepted:{first.evidence_cid}",
            },
            *[
                {
                    "sequence": index,
                    "actor": caller,
                    "operation": "barrier-synchronized-concurrent-resume",
                    "generation": attempt.writer_generation,
                    "outcome": f"accepted:{record.evidence_cid}",
                }
                for index, (caller, record) in enumerate(
                    completion_order,
                    start=2,
                )
            ],
        ],
        "parent_root_cid": attempt.identities.repository_state_cid,
        "canonical_state": canonical_before,
        "seal_cid": _seal_cid(first),
        "published_count": len(_published_settlements(store, attempt.attempt_id)),
        "adapter_result_identity": adapter_result_identity,
        "synchronization_observation": {
            "ready_barrier_parties": ready_barrier.parties,
            "call_barrier_parties": call_barrier.parties,
            "max_active_callers": max_active_callers,
            "completion_order": [caller for caller, _ in completion_order],
            "completion_order_source": "condition-record-after-real-resume-return",
        },
        "authority_observation": _assert_actual_kit(bundle, store),
        "cleanup": {**cleanup, "concurrency_workers": concurrency_cleanup},
        "disposition": "evidence-only-in-process-not-task-approval",
        **sections,
    }
    observed_log = _write_and_read_log(tmp_path / "duplicate-terminal-log.json", log)
    assert observed_log == log
    _write_worker_payload(output_path, observed_log)


def test_concurrent_duplicate_terminal_resume_is_idempotent(tmp_path: Path) -> None:
    schedule = SCHEDULES["duplicate_terminal"]
    log = _bounded_spawn_probe(
        tmp_path,
        kind=DUPLICATE_TERMINAL_PROBE,
        output_name="duplicate-terminal-log.json",
    )
    invariants = schedule["invariants"]
    assert log["published_count"] == invariants["published_settlements"] == 1
    assert len(log["synchronization_observation"]["completion_order"]) == int(
        schedule["resume_callers"]
    )
    assert log["process"]["observations"]["process_group_absent"] is True
    assert log["cleanup"]["processes_absent"] is True


def test_authoritative_runtime_integration_gaps_are_preserved_as_no_go(
    tmp_path: Path,
) -> None:
    schedule = SCHEDULES["authoritative_integration"]
    repository, storage, bundle = _open_production_runtime(tmp_path / "integration")
    canonical_before: Mapping[str, str] = {}
    record: recovery.RecoveryRecord | None = None
    try:
        canonical_before = _canonical_snapshot(repository)
        record = bundle.session.coordinator.run(PATCH)
        assert record.accepted is True
        assert record.published is True
        descriptor = sandbox.sandbox_descriptor()
        unsupported = set(descriptor["unsupported_features"])
        assert descriptor["runtime_integration_status"] == "not_integrated"
        assert descriptor["production_eligible"] is False
        assert "bootstrap/lifecycle/verification/apply integration" in unsupported
        assert _class_identity(bundle.session.lifecycle_ports.worktree).startswith(
            f"{bootstrap.__name__}."
        )
        assert _class_identity(bundle.session.lifecycle_ports.verification).startswith(
            f"{bootstrap.__name__}."
        )
        assert _class_identity(bundle.session.lifecycle_ports.assurance).startswith(
            f"{bootstrap.__name__}."
        )

        verify = _stage_from_record(record, lifecycle.VERIFY_STAGE)
        verify_payload = verify["payload"]
        assert isinstance(verify_payload, Mapping)
        assert verify_payload == {
            "planner_authority": "canonical",
            "selected_independently": False,
            "kit_port": bootstrap.KIT_PORT,
        }
        assert not {
            "command",
            "executed_tests",
            "proof_cid",
            "proof_executed",
            "verification_receipt_cid",
        } & set(verify_payload)
        assurance = _stage_from_record(record, "assurance")
        assert assurance["payload"] == {
            "accepted": True,
            "critical_survivor": False,
        }
        assert "proof" not in lifecycle.STAGES
        integration_observation = {
            "authoritative_sandbox": "not_integrated",
            "authoritative_hidden_evaluator": "not_integrated",
            "verification_execution": "marker-only-no-command-or-test-receipt",
            "proof_execution": "unavailable-no-go",
            "accepted_despite_gaps": True,
            "disposition": "no-go-authoritative-integration-owner-reopen-required",
        }
        assert integration_observation == schedule["observed_current_tree"]
        assert _seal_cid(record) is not None
        assert _canonical_snapshot(repository) == canonical_before
    finally:
        cleanup = _cleanup_runtimes(repository, storage, [bundle])
        _assert_clean(cleanup, repository)
        if canonical_before:
            assert _canonical_snapshot(repository) == canonical_before

    assert record is not None
    sections = _coordination_sections(
        [(bundle, bundle.session.attempt, bundle.session.store, record)]
    )
    log = {
        "schedule_id": schedule["schedule_id"],
        "seed": schedule["seed"],
        "events": [
            {
                "sequence": 1,
                "actor": "authoritative-runtime",
                "operation": "run",
                "generation": bundle.session.attempt.writer_generation,
                "outcome": f"accepted-with-integration-gaps:{record.evidence_cid}",
            }
        ],
        "parent_root_cid": record.identities.repository_state_cid,
        "canonical_state": canonical_before,
        "seal_cid": _seal_cid(record),
        "published_count": len(
            _published_settlements(
                bundle.session.store,
                bundle.session.attempt.attempt_id,
            )
        ),
        "observed_current_tree": schedule["observed_current_tree"],
        "authority_observation": _assert_actual_kit(bundle),
        "cleanup": cleanup,
        "disposition": schedule["observed_current_tree"]["disposition"],
        **sections,
    }
    assert _write_and_read_log(tmp_path / "integration-no-go-log.json", log) == log


@pytest.mark.parametrize(
    ("schedule_id", "seed", "stage", "position", "expected_status"),
    CRASH_CASES,
    ids=[item[0] for item in CRASH_CASES],
)
def test_exact_crash_matrix_converges_or_requires_repair(
    tmp_path: Path,
    schedule_id: str,
    seed: int,
    stage: str,
    position: str,
    expected_status: str,
) -> None:
    repository, storage, bundle = _open_production_runtime(tmp_path / schedule_id)
    canonical_before: Mapping[str, str] = {}
    store = recovery.FencedCheckpointStore()
    attempt = _attempt(bundle, writer_id=f"writer-{seed}")
    record: recovery.RecoveryRecord | None = None
    try:
        canonical_before = _canonical_snapshot(repository)
        coordinator = _coordinator(repository, bundle, store, attempt)
        coordinator.inject_crash(stage, position)
        with pytest.raises(_crash_interrupt_type()) as interrupted:
            coordinator.run(PATCH)
        assert interrupted.value.stage == stage
        assert interrupted.value.position == position
        latest = store.latest(attempt.attempt_id)
        assert latest is not None
        assert latest.get("published") is not True
        assert _published_settlements(store, attempt.attempt_id) == []
        invocations_at_crash = store.invocation_count(attempt.attempt_id, stage)
        assert invocations_at_crash == (0 if position == "before" else 1)

        record = _coordinator(repository, bundle, store, attempt).resume()
        assert record.status == expected_status
        assert _canonical_snapshot(repository) == canonical_before
        assert record.identities.repository_state_cid == attempt.identities.repository_state_cid
        if position == "during":
            assert record.status == "repair_required"
            assert record.repair_receipt is not None
            assert record.repair_receipt["ambiguous"] is True
            assert record.repair_receipt["stage"] == stage
            assert record.repair_receipt["infer_success_from_process_exit"] is False
            assert record.published is False
            assert record.accepted is False
            assert _published_settlements(store, attempt.attempt_id) == []
        else:
            assert record.status == "succeeded"
            assert record.published is True
            assert record.accepted is True
            assert len(_published_settlements(store, attempt.attempt_id)) == 1
            assert _seal_cid(record) is not None
        assert store.invocation_count(attempt.attempt_id, stage) == 1
    finally:
        cleanup = _cleanup_runtimes(repository, storage, [bundle])
        _assert_clean(cleanup, repository)
        if canonical_before:
            assert _canonical_snapshot(repository) == canonical_before

    assert record is not None
    events = [
        {
            "sequence": index,
            "actor": attempt.writer_id,
            "operation": f"{item.get('stage')}:{item.get('position')}",
            "generation": attempt.writer_generation,
            "outcome": str(item.get("status")),
        }
        for index, item in enumerate(store.history(attempt.attempt_id), start=1)
    ]
    sections = _coordination_sections([(bundle, attempt, store, record)])
    log = {
        "schedule_id": schedule_id,
        "seed": seed,
        "events": events,
        "parent_root_cid": record.identities.repository_state_cid,
        "canonical_state": canonical_before,
        "seal_cid": _seal_cid(record),
        "published_count": len(_published_settlements(store, attempt.attempt_id)),
        "checkpoint_cids": [
            str(item["checkpoint_cid"]) for item in store.history(attempt.attempt_id)
        ],
        "authority_observation": _assert_actual_kit(bundle, store),
        "cleanup": cleanup,
        "disposition": "evidence-only-in-process-not-task-approval",
        **sections,
    }
    assert _write_and_read_log(tmp_path / f"{schedule_id}.json", log) == log


def _run_process_crash_worker() -> int:
    root_text = os.environ.get(WORKER_ROOT_ENV)
    log_text = os.environ.get(WORKER_LOG_ENV)
    if not root_text or not log_text:
        return 64
    root = Path(root_text)
    log_path = Path(log_text)
    schedule = _schedule_fixture()["process_crash"]
    repository, _storage, bundle = _open_production_runtime(root)
    canonical_before = _canonical_snapshot(repository)
    authority = _assert_actual_kit(bundle)
    coordinator = bundle.session.coordinator
    coordinator.inject_crash(str(schedule["stage"]), str(schedule["position"]))
    try:
        coordinator.run(PATCH)
    except _crash_interrupt_type() as interrupted:
        latest = bundle.session.store.latest(bundle.session.attempt.attempt_id)
        payload = {
            "schema": "ipfs-accelerate.proof-context.v0.1/process-crash-log",
            "schedule_id": schedule["schedule_id"],
            "seed": schedule["seed"],
            "pid": os.getpid(),
            "process_group": os.getpgrp(),
            "stage": interrupted.stage,
            "position": interrupted.position,
            "latest": {
                "stage": latest.get("stage") if latest else None,
                "position": latest.get("position") if latest else None,
                "in_flight": latest.get("in_flight") if latest else None,
                "published": latest.get("published") if latest else None,
                "settled": latest.get("settled") if latest else None,
                "checkpoint_cid": latest.get("checkpoint_cid") if latest else None,
            },
            "checkpoint_observation_scope": "child-in-process-pre-exit-only",
            "attempt_id": bundle.session.attempt.attempt_id,
            "writer_id": bundle.session.attempt.writer_id,
            "writer_generation": bundle.session.attempt.writer_generation,
            "fence_token": bundle.session.attempt.fence_token,
            "fence_id": bundle.session.attempt.fence_id,
            "lease_id": bundle.session.attempt.lease_id,
            "lease_expires_at": bundle.session.attempt.lease_expires_at,
            "parent_root_cid": bundle.session.attempt.identities.repository_state_cid,
            "seal_cid": _seal_cid_from_history(
                bundle.session.store,
                bundle.session.attempt.attempt_id,
            ),
            "canonical_before": canonical_before,
            "canonical_after": _canonical_snapshot(repository),
            "authority_observation": authority,
            "worktree_path": str(bundle.session.worktree.path),
            "worktree_id": Path(str(bundle.session.worktree.path)).name,
            "process_exit_inferred_success": recovery.INFER_SUCCESS_FROM_PROCESS_EXIT,
        }
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os._exit(int(schedule["expected_exit_code"]))
    return 65


def test_real_post_crash_restart_exposes_lost_checkpoint_false_success_no_go(
    tmp_path: Path,
) -> None:
    _synchronize_runtime_modules()
    schedule = SCHEDULES["process_crash"]
    worker_root = tmp_path / "worker"
    worker_log = tmp_path / "worker-crash.json"
    repository = worker_root / "repository"
    storage = worker_root / "runtime-state"
    kit_root = storage / "kit"
    python_path = os.environ.get("PYTHONPATH", "")
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin"),
        "PYTHONPATH": os.pathsep.join(part for part in (str(PACKAGE_ROOT), python_path) if part),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        WORKER_ROOT_ENV: str(worker_root),
        WORKER_LOG_ENV: str(worker_log),
        "PYTHONDONTWRITEBYTECODE": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }
    process: subprocess.Popen[str] | None = None
    abandoned_worktree: Path | None = None
    bundles: list[bootstrap.RuntimeBundle] = []
    collision_record: recovery.RecoveryRecord | None = None
    restart_record: recovery.RecoveryRecord | None = None
    child_log: Mapping[str, Any] = {}
    canonical_before: Mapping[str, str] | None = None
    canonical_after_cleanup: Mapping[str, str] | None = None
    intermediate_cleanup: Mapping[str, Any] = {}
    cleanup: Mapping[str, Any] = {}
    process_cleanup: Mapping[str, Any] = {
        "errors": ["process-not-started"],
        "process_absent": False,
        "process_group_absent": False,
    }
    stdout = ""
    stderr = ""
    try:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), WORKER_ARGUMENT],
            cwd=PACKAGE_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=int(schedule["timeout_seconds"]))
        except subprocess.TimeoutExpired:
            process_cleanup = _ensure_process_absent(process)
            pytest.fail(f"crash worker exceeded its bounded deadline: cleanup={process_cleanup!r}")
        assert process.returncode == schedule["expected_exit_code"], (stdout, stderr)
        assert _process_group_absent(process.pid)

        child_log = json.loads(worker_log.read_text(encoding="utf-8"))
        assert child_log["pid"] == process.pid == child_log["process_group"]
        assert child_log["stage"] == schedule["stage"] == lifecycle.APPLY_STAGE
        assert child_log["position"] == schedule["position"] == "during"
        assert child_log["process_exit_inferred_success"] is False
        assert child_log["checkpoint_observation_scope"] == ("child-in-process-pre-exit-only")
        assert child_log["seal_cid"] is None
        canonical_before = child_log["canonical_before"]
        assert child_log["canonical_after"] == canonical_before
        assert canonical_before["porcelain"] == ""
        observed = schedule["child_pre_exit_observation"]
        assert child_log["latest"]["in_flight"] is observed["in_flight"]
        assert child_log["latest"]["published"] is observed["published"]
        assert child_log["latest"]["settled"] is observed["settled"]
        abandoned_worktree = Path(str(child_log["worktree_path"]))
        assert repository.is_dir()
        assert abandoned_worktree.is_dir()
        assert len(_registered_worktrees(repository)) == 2

        collision_bundle = bootstrap.open_runtime(
            repository,
            options=bootstrap.RuntimeOptions(
                kit_root=kit_root,
                worktree_parent=storage / "worktrees",
                require_kit=True,
            ),
        )
        bundles.append(collision_bundle)
        collision_authority = _assert_actual_kit(collision_bundle)
        _assert_same_authority_binding(
            collision_authority,
            child_log["authority_observation"],
        )
        assert collision_bundle.session.attempt.attempt_id == child_log["attempt_id"]
        collision_record = collision_bundle.session.coordinator.resume()
        expected_collision = schedule["no_cleanup_restart_observation"]
        assert collision_record.status == expected_collision["status"] == "unavailable"
        assert collision_record.error == expected_collision["error"] == "unavailable_capability"
        assert collision_record.accepted is expected_collision["accepted"] is False
        assert collision_record.published is expected_collision["published"] is False
        assert collision_record.status != "repair_required"
        assert _canonical_snapshot(repository) == canonical_before

        _git(
            repository,
            "worktree",
            "remove",
            "--force",
            str(abandoned_worktree),
        )
        _git(repository, "worktree", "prune")
        intermediate_cleanup = {
            "abandoned_worktree_absent": not abandoned_worktree.exists(),
            "registry": [
                "canonical-repository" if item == repository.resolve() else "unexpected-worktree"
                for item in _registered_worktrees(repository)
            ],
            "kit_root_retained": kit_root.is_dir(),
        }
        assert intermediate_cleanup == {
            "abandoned_worktree_absent": True,
            "registry": ["canonical-repository"],
            "kit_root_retained": True,
        }

        restart_bundle = bootstrap.open_runtime(
            repository,
            options=bootstrap.RuntimeOptions(
                kit_root=kit_root,
                worktree_parent=storage / "worktrees",
                require_kit=True,
            ),
        )
        bundles.append(restart_bundle)
        restart_authority = _assert_actual_kit(restart_bundle)
        _assert_same_authority_binding(
            restart_authority,
            child_log["authority_observation"],
        )
        assert restart_bundle.session.attempt.attempt_id == child_log["attempt_id"]
        assert restart_bundle.session.store is not collision_bundle.session.store
        restart_record = restart_bundle.session.coordinator.resume()
        expected_restart = schedule["post_cleanup_fresh_restart_observation"]
        assert restart_record.status == expected_restart["status"] == "succeeded"
        assert restart_record.error is None
        assert restart_record.accepted is expected_restart["accepted"] is True
        assert restart_record.published is expected_restart["published"] is True
        assert _seal_cid(restart_record) is not None
        assert _canonical_snapshot(repository) == canonical_before
    finally:
        try:
            if process is not None:
                process_cleanup = _ensure_process_absent(process)
        finally:
            try:
                if repository.is_dir():
                    abandoned = (abandoned_worktree,) if abandoned_worktree is not None else ()
                    cleanup = _cleanup_runtimes(
                        repository,
                        storage,
                        bundles,
                        abandoned_worktrees=abandoned,
                    )
                    cleanup = {
                        **cleanup,
                        "errors": [
                            *cleanup["errors"],
                            *process_cleanup["errors"],
                        ],
                        "processes_absent": bool(
                            process_cleanup["process_absent"]
                            and process_cleanup["process_group_absent"]
                        ),
                    }
                    canonical_after_cleanup = _canonical_snapshot(repository)
            finally:
                try:
                    if worker_root.exists():
                        shutil.rmtree(worker_root)
                finally:
                    worker_log.unlink(missing_ok=True)

    if cleanup:
        cleanup = {
            **cleanup,
            "observations": {
                **cleanup["observations"],
                "raw_worker_log_absent": not worker_log.exists(),
            },
        }

    assert canonical_before is not None
    assert canonical_after_cleanup == canonical_before
    _assert_clean(cleanup, repository)
    assert not worker_root.exists()
    assert not worker_log.exists()
    assert not any(thread.name.startswith("pcce075") for thread in threading.enumerate())
    assert process is not None
    assert collision_record is not None
    assert restart_record is not None
    events = [
        {
            "sequence": 1,
            "actor": "crash-worker",
            "operation": f"{child_log['stage']}:{child_log['position']}",
            "generation": child_log["writer_generation"],
            "outcome": f"exit:{schedule['expected_exit_code']}",
        },
        {
            "sequence": 2,
            "actor": "fresh-runtime-with-collision",
            "operation": "resume-without-cleanup",
            "generation": collision_record.writer_generation,
            "outcome": f"{collision_record.status}:{collision_record.error}",
        },
        {
            "sequence": 3,
            "actor": "cleanup",
            "operation": "discard-abandoned-worktree-retain-kit-root",
            "generation": collision_record.writer_generation,
            "outcome": "clean",
        },
        {
            "sequence": 4,
            "actor": "fresh-runtime-after-cleanup",
            "operation": "resume-from-same-kit-root",
            "generation": restart_record.writer_generation,
            "outcome": f"false-accepted:{restart_record.evidence_cid}",
        },
    ]
    process_binding_cid = recovery.mint_recovery_cid(
        {
            "kind": "bounded-crash-worker-process",
            "schedule_id": schedule["schedule_id"],
            "pid": process.pid,
            "exit_code": process.returncode,
        }
    )
    process_section = _evidence_section(
        True,
        [
            {
                "process_binding_cid": process_binding_cid,
                "pid": process.pid,
                "process_group": int(child_log["process_group"]),
            }
        ],
        {
            "exit_code": process.returncode,
            "timeout_seconds": schedule["timeout_seconds"],
            "process_absent": process_cleanup["process_absent"],
            "process_group_absent": process_cleanup["process_group_absent"],
            "raw_streams_retained": False,
        },
    )
    sections = dict(
        _coordination_sections(
            [
                (
                    bundles[0],
                    bundles[0].session.attempt,
                    bundles[0].session.store,
                    collision_record,
                ),
                (
                    bundles[1],
                    bundles[1].session.attempt,
                    bundles[1].session.store,
                    restart_record,
                ),
            ],
            process=process_section,
            extra_worktree_ids=(str(child_log["worktree_id"]),),
        )
    )
    child_store_binding_cid = recovery.mint_recovery_cid(
        {
            "kind": "child-injected-coordinator-store-binding",
            "attempt_id": child_log["attempt_id"],
            "store_class": child_log["authority_observation"]["actual_checkpoint_store_class"],
        }
    )
    cas_section = dict(sections["cas"])
    cas_section["identities"] = [
        *cas_section["identities"],
        {
            "store_binding_cid": child_store_binding_cid,
            "attempt_id": child_log["attempt_id"],
        },
    ]
    cas_observations = dict(cas_section["observations"])
    cas_observations["actual_injected_stores"] = [
        *cas_observations["actual_injected_stores"],
        {
            "store_binding_cid": child_store_binding_cid,
            "actual_store_class": child_log["authority_observation"][
                "actual_checkpoint_store_class"
            ],
            "is_bundle_session_store": True,
            "is_kit_store": False,
            "scope": "child-in-process-object-lost-on-exit",
        },
    ]
    cas_section["observations"] = cas_observations
    sections["cas"] = cas_section
    checkpoint_section = dict(sections["checkpoints"])
    checkpoint_section["applicable"] = True
    checkpoint_section["identities"] = [
        *checkpoint_section["identities"],
        {
            "store_binding_cid": child_store_binding_cid,
            "checkpoint_cid": child_log["latest"]["checkpoint_cid"],
        },
    ]
    checkpoint_observations = dict(checkpoint_section["observations"])
    checkpoint_observations["child_pre_exit"] = {
        "count": 1,
        "latest_checkpoint_cid": child_log["latest"]["checkpoint_cid"],
        "durable_after_exit": False,
    }
    checkpoint_section["observations"] = checkpoint_observations
    sections["checkpoints"] = checkpoint_section
    lease_section = dict(sections["lease"])
    lease_section["identities"] = [
        *lease_section["identities"],
        {
            "attempt_id": child_log["attempt_id"],
            "lease_id": child_log["lease_id"],
            "source": "crash-worker",
        },
    ]
    sections["lease"] = lease_section
    fence_section = dict(sections["fence"])
    fence_section["identities"] = [
        *fence_section["identities"],
        {
            "attempt_id": child_log["attempt_id"],
            "fence_id": child_log["fence_id"],
            "fence_cid": child_log["fence_token"],
            "generation": child_log["writer_generation"],
            "writer_id": child_log["writer_id"],
            "source": "crash-worker",
        },
    ]
    sections["fence"] = fence_section
    evidence_log = {
        "schedule_id": schedule["schedule_id"],
        "seed": schedule["seed"],
        "events": events,
        "parent_root_cid": child_log["parent_root_cid"],
        "canonical_state": canonical_before,
        "seal_cid": _seal_cid(restart_record),
        "published_count": len(
            _published_settlements(
                bundles[-1].session.store,
                bundles[-1].session.attempt.attempt_id,
            )
        ),
        "child_checkpoint_cid": child_log["latest"]["checkpoint_cid"],
        "collision_record": {
            "status": collision_record.status,
            "error": collision_record.error,
            "accepted": collision_record.accepted,
            "published": collision_record.published,
        },
        "fresh_restart_record": {
            "status": restart_record.status,
            "error": restart_record.error,
            "accepted": restart_record.accepted,
            "published": restart_record.published,
            "evidence_cid": restart_record.evidence_cid,
        },
        "authority_observation": {
            "crash_worker": child_log["authority_observation"],
            "collision_runtime": collision_authority,
            "post_cleanup_runtime": restart_authority,
        },
        "intermediate_cleanup": intermediate_cleanup,
        "cleanup": cleanup,
        "disposition": "no-go-lost-durable-checkpoint-false-success",
        **sections,
    }
    assert _write_and_read_log(tmp_path / "post-crash-restart-log.json", evidence_log) == (
        evidence_log
    )


if __name__ == "__main__" and WORKER_ARGUMENT in sys.argv[1:]:
    raise SystemExit(_run_process_crash_worker())
