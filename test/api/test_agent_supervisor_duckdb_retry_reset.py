from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

duckdb = pytest.importorskip("duckdb")

from ipfs_accelerate_py.agent_supervisor import duckdb_retry_reset as retry_reset

from ipfs_accelerate_py.agent_supervisor.authorization_logic import (
    ControlMutationPolicy,
)
from ipfs_accelerate_py.agent_supervisor.checkout_lock import (
    checkout_mutation_lock_path,
    checkout_repository_id,
)
from ipfs_accelerate_py.agent_supervisor.control_contracts import (
    AuthorizationDecision,
    AuthorizationVerdict,
    IdempotencyKey,
    Operation,
    OperationAuthority,
    OperationRequest,
)
from ipfs_accelerate_py.agent_supervisor.duckdb_retry_reset import (
    RETRY_RESET_GRANT,
    RETRY_RESET_OWNER_FILE,
    RETRY_RESET_POLICY_SCHEMA,
    DuckDBRetryResetAuthorizationError,
    DuckDBRetryResetConflict,
    DuckDBRetryResetCorruptionError,
    DuckDBRetryResetQuiescenceError,
    LaneBinding,
    RetryResetOwnerConfig,
    execute_duckdb_retry_reset,
    inspect_incomplete_retry_resets,
    main,
    prepare_duckdb_retry_reset_execution_intent,
    recover_duckdb_retry_reset_execution_intent,
    retry_reset_execution_intent_binding,
    retry_reset_expected_effect,
)
from ipfs_accelerate_py.agent_supervisor.duckdb_task_source import (
    DuckDBTaskSource,
    TaskSourceConflictError,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    content_identity,
)

NOW = time.time_ns() // 1_000_000
_PARENT_PROCESSES: list[subprocess.Popen[bytes]] = []
_EXECUTION_AUTHORITIES: dict[tuple[str, str], dict[str, Any]] = {}


@pytest.fixture(autouse=True)
def _cleanup_parent_processes() -> Any:
    yield
    _EXECUTION_AUTHORITIES.clear()
    while _PARENT_PROCESSES:
        process = _PARENT_PROCESSES.pop()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def _source(repository_tree_id: str) -> dict[str, object]:
    return {
        "schema": "fixture/formal-plan-input@1",
        "repository_tree_id": repository_tree_id,
        "objectives": [
            {
                "goal_id": "G1",
                "goal_cid": "goal:datasets",
                "owner_actor_id": "owner:supervisor",
                "title": "Datasets retry reset",
                "acceptance_criteria": ["The governed reset is replayable."],
            }
        ],
        "taskboard": [
            {
                "task_id": "DQK-007",
                "task_cid": "task:cid:dqk-007",
                "goal_id": "G1",
                "actor_id": "agent:alpha",
                "acceptance_criteria": ["Focused tests pass."],
                "validation_commands": ["pytest -q test_reset.py"],
            },
            {
                "task_id": "DQK-008",
                "task_cid": "task:cid:dqk-008",
                "goal_id": "G1",
                "depends_on": ["DQK-007"],
                "actor_id": "agent:beta",
                "acceptance_criteria": ["Descendant stays valid."],
                "validation_commands": ["pytest -q test_descendant.py"],
            },
        ],
        "proof_policy": {
            "policy_cid": "policy:cid:datasets-reset",
            "minimum_code_assurance": "candidate",
            "freshness_seconds": 3600,
            "fallback_check_ids": ["fallback:pytest"],
        },
    }


def _queue_payload(
    task_cid: str,
    task_alias: str,
    *,
    attempt_count: int,
    schema: str = "persistent_task_queue_v2",
) -> dict[str, Any]:
    entry = {
        "task_id": task_alias,
        "priority": "P1",
        "track": "datasets",
        "canonical_task_cid": task_cid,
        "canonical_task_key": "task:key:dqk-007",
        "aliases": [task_alias],
        "provenance": [],
        "selection_penalty": 900,
        "attempt_count": attempt_count,
        "last_selected_at": 3.0,
        "last_completed_at": 2.0,
        "consecutive_failures": 4,
        "consecutive_no_change": 3,
        "merge_failure_count": 2,
        "cooldown_until": 999_999.0,
        "notes": "validation failed",
    }
    if schema == "persistent_task_queue_v3":
        entry.update(
            {
                "authority_renewal_key": "renewal:bound",
                "authority_renewal_failure_count": 2,
                "authority_renewal_last_failure_at": 40.0,
                "authority_renewal_cooldown_until": 400.0,
                "authority_renewal_quarantined": True,
                "authority_renewal_reason": "authority unavailable",
            }
        )
    return {
        "schema": schema,
        "updated_at": 1.0,
        "entry_count": 1,
        "entries": {task_cid: entry},
        "aliases": {task_alias: task_cid},
    }


def _write_lane(
    state_root: Path,
    lane: str,
    *,
    task_cid: str = "task:cid:dqk-007",
    task_alias: str = "DQK-007",
    attempt_count: int = 7,
    queue_schema: str = "persistent_task_queue_v2",
) -> dict[str, str]:
    lane_root = state_root / lane
    lane_root.mkdir(parents=True)
    state_path = lane_root / f"{lane}_task_state.json"
    queue_path = lane_root / "task_queue.json"
    state_path.write_text(
        json.dumps(
            {
                "implementation_in_progress": False,
                "active_task_id": "",
                "active_task_cid": "",
                "task_identities": {task_alias: {"canonical_task_cid": task_cid}},
                "implementation_attempts": {task_alias: attempt_count},
                "implementation_attempts_by_cid": {task_cid: attempt_count + 1},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    queue_path.write_text(
        json.dumps(
            _queue_payload(
                task_cid,
                task_alias,
                attempt_count=attempt_count,
                schema=queue_schema,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "state_prefix": lane,
        "state_path": state_path.relative_to(state_root).as_posix(),
        "queue_path": queue_path.relative_to(state_root).as_posix(),
    }


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repository,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def _policy_payload(policy: ControlMutationPolicy) -> dict[str, Any]:
    return {
        "schema": RETRY_RESET_POLICY_SCHEMA,
        "policy_id": policy.policy_id,
        "policy_revision": policy.policy_revision,
        "permits": [item.to_record() for item in policy.permits],
        "current_tree_ids": dict(policy.current_tree_ids),
        "current_objective_revisions": dict(policy.current_objective_revisions),
        "active_lease_fences": dict(policy.active_lease_fences),
    }


def _fixture(
    tmp_path: Path,
    *,
    lanes: int = 2,
    expected_status: str = "completed",
    queue_schema: str = "persistent_task_queue_v2",
) -> tuple[
    DuckDBTaskSource,
    OperationRequest,
    ControlMutationPolicy,
    RetryResetOwnerConfig,
    Path,
]:
    repository = (tmp_path / "repository").resolve()
    repository.mkdir(parents=True)
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Retry Reset Test")
    _git(repository, "config", "user.email", "retry-reset@example.invalid")
    accelerator_source = (tmp_path / "accelerator-source").resolve()
    accelerator_source.mkdir()
    _git(accelerator_source, "init", "-q")
    _git(accelerator_source, "config", "user.name", "Retry Reset Test")
    _git(accelerator_source, "config", "user.email", "retry-reset@example.invalid")
    (accelerator_source / "accelerator.txt").write_text(
        "pinned accelerator\n", encoding="utf-8"
    )
    _git(accelerator_source, "add", "accelerator.txt")
    _git(accelerator_source, "commit", "-q", "-m", "pin accelerator")
    _git(
        repository,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(accelerator_source),
        "ipfs_accelerate_py",
    )
    (repository / "plan-root.txt").write_text("admitted plan\n", encoding="utf-8")
    _git(repository, "add", ".gitmodules", "ipfs_accelerate_py", "plan-root.txt")
    _git(repository, "commit", "-q", "-m", "admit plan")
    task_source_tree = _git(repository, "rev-parse", "HEAD^{tree}")

    state_root = (tmp_path / "state").resolve()
    state_root.mkdir()
    source = DuckDBTaskSource(state_root / "workflow.duckdb")
    source.materialize(_source(task_source_tree))
    task = source.get_task("DQK-007")
    assert task is not None
    selected = task
    if task.status != expected_status:
        selected = source.compare_and_set_status(
            "DQK-007", task.revision, expected_status
        ).task
    lane_bindings = [
        _write_lane(
            state_root,
            f"lane{index}",
            queue_schema=queue_schema,
        )
        for index in range(lanes)
    ]

    # Model an admitted implementation merge after immutable task-source
    # materialization.  Reset must bind this current HEAD independently.
    (repository / "implementation.txt").write_text("merged work\n", encoding="utf-8")
    _git(repository, "add", "implementation.txt")
    _git(repository, "commit", "-q", "-m", "merge implementation")
    head_commit = _git(repository, "rev-parse", "HEAD")
    head_tree = _git(repository, "rev-parse", "HEAD^{tree}")

    parameters = {
        "task_source_kind": "duckdb",
        "database_path": "workflow.duckdb",
        "plan_root_cid": source.snapshot().plan_root_cid,
        "task_source_repository_tree_id": task_source_tree,
        "repository_head_commit": head_commit,
        "task_cid": selected.task_cid,
        "task_alias": selected.task_alias,
        "task_revision": selected.revision,
        "expected_status": expected_status,
        "reopen_status": "retrying",
        "writer_id": "local",
        "writer_fencing_token": 1,
        "lanes": lane_bindings,
        "lifecycle_owner_paths": ["multi_supervisor_runner.pid"],
    }
    repository_root = str(repository)
    repository_id = checkout_repository_id(repository)
    effect = retry_reset_expected_effect(
        repository_root=repository_root,
        state_root=str(state_root),
        repository_id=repository_id,
        tree_id=head_tree,
        parameters=parameters,
    )
    common = {
        "operation": Operation.RETRY,
        "repository_root": repository_root,
        "state_root": str(state_root),
        "repository_id": repository_id,
        "tree_id": head_tree,
        "objective_id": "goal:datasets",
        "objective_revision": "goal-revision:1",
        "policy_id": "policy:datasets-reset",
        "policy_revision": "policy-revision:1",
        "caller": "operator:datasets",
    }
    decision = AuthorizationDecision(
        verdict=AuthorizationVerdict.PERMIT,
        granted_authority=OperationAuthority.MUTATION,
        authorized_effect_ids=(effect.effect_id,),
        grant_ids=(RETRY_RESET_GRANT, "grant:duckdb-writer:local"),
        lease_id="lease:datasets-reset",
        fencing_epoch=1,
        evaluated_at_ms=NOW - 100,
        expires_at_ms=NOW + 600_000,
        **common,
    )
    request = OperationRequest(
        expected_effects=(effect,),
        parameters=parameters,
        idempotency=IdempotencyKey(
            key=f"retry-reset:dqk-007:revision-{selected.revision}",
            operation=Operation.RETRY,
            caller=common["caller"],
            repository_id=repository_id,
            objective_id=common["objective_id"],
        ),
        authorization=decision,
        lease_id="lease:datasets-reset",
        fencing_epoch=1,
        **common,
    )
    policy = ControlMutationPolicy(
        policy_id=common["policy_id"],
        policy_revision=common["policy_revision"],
        permits=(decision,),
        current_tree_ids={repository_id: head_tree},
        current_objective_revisions={"goal:datasets": "goal-revision:1"},
        active_lease_fences={"lease:datasets-reset": 1},
    )
    policy_path = state_root / "control" / "retry-reset-policy.json"
    policy_path.parent.mkdir()
    policy_bytes = (
        json.dumps(_policy_payload(policy), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    policy_path.write_bytes(policy_bytes)
    owner = RetryResetOwnerConfig(
        repository_root=repository_root,
        repository_id=repository_id,
        database_path="workflow.duckdb",
        task_source_repository_tree_id=task_source_tree,
        policy_path=policy_path.relative_to(state_root).as_posix(),
        policy_digest="sha256:" + hashlib.sha256(policy_bytes).hexdigest(),
        lanes=tuple(
            LaneBinding(item["state_prefix"], item["state_path"], item["queue_path"])
            for item in lane_bindings
        ),
        lifecycle_owner_paths=("multi_supervisor_runner.pid",),
    )
    owner_path = state_root / RETRY_RESET_OWNER_FILE
    owner_path.write_text(
        json.dumps(owner.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    owner_path.chmod(0o600)
    return source, request, policy, owner, state_root


def _execute(
    request: OperationRequest,
    policy: ControlMutationPolicy,
    owner: RetryResetOwnerConfig,
    *,
    clock_ms: Any | None = None,
    fault_injector: Any | None = None,
    lock_timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    authority = _execution_authority(request, policy, owner)
    return execute_duckdb_retry_reset(
        request,
        trusted_policy=policy,
        trusted_owner=owner,
        **authority,
        clock_ms=clock_ms or (lambda: NOW),
        lock_timeout_seconds=lock_timeout_seconds,
        fault_injector=fault_injector,
    )


def _parent_prepared(
    request: OperationRequest,
    owner: RetryResetOwnerConfig,
    state_root: Path,
) -> tuple[dict[str, Any], Path]:
    binding = retry_reset._binding_from_parameters(request.parameters)
    owner_path = state_root / retry_reset.RETRY_RESET_OWNER_FILE
    master = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(120)",
            "--duration-seconds",
            "3600",
            "--implementation-supervisor-lanes-per-track",
            str(len(binding.lanes)),
            "--common-arg=--execution-slice-task-id",
            "--common-arg=DQK-007",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    _PARENT_PROCESSES.append(master)
    deadline = time.monotonic() + 2.0
    actual_master = None
    while time.monotonic() < deadline:
        candidate = retry_reset._live_process_identity(master.pid)
        if candidate is not None and candidate.get("argv"):
            actual_master = candidate
            break
        time.sleep(0.01)
    assert actual_master is not None
    owner_identity = retry_reset._live_process_identity(os.getpid())
    assert owner_identity is not None
    repository = Path(request.repository_root)
    head_commit = _git(repository, "rev-parse", "--verify", "HEAD")
    head_tree = _git(repository, "rev-parse", "--verify", "HEAD^{tree}")
    branch = _git(repository, "branch", "--show-current")
    lock = Path(checkout_mutation_lock_path(repository))
    accelerator = (repository / "ipfs_accelerate_py").resolve()
    accelerator_head = _git(accelerator, "rev-parse", "--verify", "HEAD")
    accelerator_tree = _git(
        accelerator, "rev-parse", "--verify", "HEAD^{tree}"
    )
    accelerator_branch = _git(accelerator, "branch", "--show-current")
    accelerator_lock = Path(checkout_mutation_lock_path(accelerator))
    checkout_binding = [
        {
            "role": "parent",
            "repository_root": str(repository),
            "repository_id": request.repository_id,
            "lock_path": str(lock.parent.resolve() / lock.name),
            "branch": branch,
            "head_commit": head_commit,
            "head_tree": head_tree,
            "parent_accelerator_gitlink": accelerator_head,
        },
        {
            "role": "accelerator",
            "repository_root": str(accelerator),
            "repository_id": checkout_repository_id(accelerator),
            "lock_path": str(
                accelerator_lock.parent.resolve() / accelerator_lock.name
            ),
            "branch": accelerator_branch,
            "head_commit": accelerator_head,
            "head_tree": accelerator_tree,
            "parent_accelerator_gitlink": accelerator_head,
        },
    ]
    checkout_binding.sort(key=lambda item: item["lock_path"])
    sha = "sha256:" + "a" * 64
    environment_root = str((state_root / "environment").resolve())
    sealed_launcher = str((state_root / "environment/bin/python").resolve())
    environment_receipt = state_root / "environment-receipt.json"
    environment_receipt.write_text(
        json.dumps(
            {
                "receipt_id": "environment:test",
                "probe": {
                    "environment_root": environment_root,
                    "sealed_python_launcher_path": sealed_launcher,
                    "sealed_python_launcher_sha256": sha,
                    "base_python_sha256": sha,
                    "site_packages_manifest_sha256": sha,
                    "duckdb_version": "1.4.3",
                    "duckdb_record_evidence_sha256": sha,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    environment_digest = "sha256:" + hashlib.sha256(
        environment_receipt.read_bytes()
    ).hexdigest()
    timestamp = datetime.fromtimestamp(NOW / 1_000, timezone.utc).isoformat()
    stored_master = {
        "schema": "ipfs_datasets_py/duckdb-quack-master-identity@3",
        "program_id": "ipfs-datasets-duckdb-quack-v1",
        "repository_root": request.repository_root,
        "master_root": str((state_root / "master").resolve()),
        "master_pid_path": str((state_root / "master/supervisor.pid").resolve()),
        "plan_root_cid": binding.plan_root_cid,
        "repository_tree_id": binding.task_source_repository_tree_id,
        "execution_slice_sha256": sha,
        "execution_slice_task_count": 1,
        "authorization_held_set_sha256": sha,
        "authorization_held_task_count": 0,
        "bootstrap_completion_evidence_id": "",
        "lane_count": len(binding.lanes),
        "created_at": timestamp,
        "python_environment_sha256": sha,
        **{
            key: actual_master[key]
            for key in ("pid", "boot_id", "start_ticks", "cmdline_sha256")
        },
    }
    parent_path = (
        state_root
        / "duckdb-retry-reset/journals/lifecycle"
        / (
            retry_reset._request_canonical_evidence(request)["digest"].removeprefix(
                "sha256:"
            )
            + ".json"
        )
    )
    parent: dict[str, Any] = {
        "schema": "ipfs_datasets_py/duckdb-quack-retry-lifecycle-journal@1",
        "program_id": "ipfs-datasets-duckdb-quack-v1",
        "phase": "prepared",
        "request_id": request.request_id,
        "request_digest": retry_reset._request_canonical_evidence(request)["digest"],
        "request_file_digest": retry_reset._request_canonical_evidence(request)[
            "digest"
        ],
        "repository_root": request.repository_root,
        "runtime_root": request.state_root,
        "database_path": str((state_root / binding.database_path).resolve()),
        "plan_root_cid": binding.plan_root_cid,
        "task_source_repository_tree_id": binding.task_source_repository_tree_id,
        "repository_head_commit": binding.repository_head_commit,
        "repository_head_tree": request.tree_id,
        "checkout_binding": checkout_binding,
        "task": {
            "task_cid": binding.task_cid,
            "task_alias": binding.task_alias,
            "status": binding.expected_status,
            "revision": binding.task_revision,
        },
        "writer": {
            "writer_id": binding.writer_id,
            "fencing_token": binding.writer_fencing_token,
        },
        "owner_configuration": {
            "path": str(owner_path),
            "digest": retry_reset._digest_path(owner_path),
            "payload": owner.to_dict(),
        },
        "policy": {
            "path": str(state_root / owner.policy_path),
            "digest": owner.policy_digest,
            "policy_id": request.policy_id,
            "policy_revision": request.policy_revision,
            "authorization_decision_id": request.authorization.decision_id,
        },
        "authorization": {
            "decision_id": request.authorization.decision_id,
            "evaluated_at_ms": request.authorization.evaluated_at_ms,
            "expires_at_ms": request.authorization.expires_at_ms,
            "lease_id": request.lease_id,
            "fencing_epoch": request.fencing_epoch,
        },
        "environment": {
            "receipt_path": str(environment_receipt.resolve()),
            "receipt_sha256": environment_digest,
            "receipt_id": "environment:test",
            "environment_root": environment_root,
            "sealed_python_launcher_path": sealed_launcher,
            "sealed_python_launcher_sha256": sha,
            "base_python_sha256": sha,
            "site_packages_manifest_sha256": sha,
            "duckdb_version": "1.4.3",
            "duckdb_record_evidence_sha256": sha,
        },
        "old_master": {
            "stored": stored_master,
            "actual": actual_master,
            "lane_count": len(binding.lanes),
            "duration_seconds": "3600",
            "execution_slice": ["DQK-007"],
            "dedicated_session_id": master.pid,
        },
        "old_process_tree": [actual_master],
        "lifecycle_owners": [{"adopted_at": timestamp, **owner_identity}],
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    parent["intent_cid"] = retry_reset._parent_intent_cid(parent)
    return parent, parent_path


def _drained_parent(
    parent: dict[str, Any],
    binding: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    master_pid = int(parent["old_master"]["actual"]["pid"])
    master_process = next(
        (item for item in _PARENT_PROCESSES if item.pid == master_pid), None
    )
    assert master_process is not None
    if master_process.poll() is None:
        master_process.terminate()
        master_process.wait(timeout=5)
    drained = {
        **parent,
        "phase": "leased",
        "execution_intent": binding,
        "drain_process_tree": [parent["old_master"]["actual"]],
        "drain_started_at": "2026-08-09T00:01:00+00:00",
        "drained_at": "2026-08-09T00:02:00+00:00",
    }
    drained["drain_cid"] = retry_reset._parent_drain_cid(drained)
    drained["drained_cid"] = retry_reset._parent_drained_cid(drained)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(drained, indent=2, sort_keys=True) + "\n")
    path.chmod(0o600)
    return drained


def _execution_authority(
    request: OperationRequest,
    policy: ControlMutationPolicy,
    owner: RetryResetOwnerConfig,
) -> dict[str, Any]:
    key = (request.state_root, request.request_id)
    existing = _EXECUTION_AUTHORITIES.get(key)
    if existing is not None:
        return dict(existing)
    state_root = Path(request.state_root)
    parent, parent_path = _parent_prepared(request, owner, state_root)
    projection = prepare_duckdb_retry_reset_execution_intent(
        request,
        trusted_policy=policy,
        trusted_owner=owner,
        parent_prepared=parent,
        parent_journal_path=parent_path,
        request_file_bytes=request.canonical_bytes(),
        clock_ms=lambda: NOW,
    )
    binding = retry_reset_execution_intent_binding(projection)
    drained = _drained_parent(parent, binding, parent_path)
    assertion = {"lease_cid": "lease:test-held"}
    authority = {
        "execution_intent": binding,
        "parent_journal": drained,
        "checkout_lease_assertion": assertion,
        "checkout_lease_verifier": lambda record: dict(record) == assertion,
    }
    _EXECUTION_AUTHORITIES[key] = authority
    return dict(authority)


def _gate_next_task_source_write(
    monkeypatch: pytest.MonkeyPatch,
    *,
    method_name: str,
    lock_path: Path,
) -> tuple[threading.Event, threading.Event, threading.Thread, Any]:
    """Make one real task-source lock acquisition wait behind another owner."""

    called = threading.Event()
    acquired = threading.Event()
    release = threading.Event()
    original = getattr(DuckDBTaskSource, method_name)

    def hold_lock() -> None:
        if not called.wait(timeout=5):
            return
        with retry_reset.exclusive_file_lock(lock_path, timeout_seconds=5):
            acquired.set()
            release.wait(timeout=5)

    holder = threading.Thread(target=hold_lock, daemon=True)

    def gated(self: DuckDBTaskSource, *args: Any, **kwargs: Any) -> Any:
        called.set()
        if not acquired.wait(timeout=5):
            raise RuntimeError("test task-source lock holder did not acquire")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DuckDBTaskSource, method_name, gated)
    holder.start()
    return acquired, release, holder, original


def test_cross_lane_reset_is_content_bound_and_idempotent(tmp_path: Path) -> None:
    source, request, policy, owner, state_root = _fixture(tmp_path)

    receipt = _execute(request, policy, owner)
    replay = _execute(request, policy, owner)

    assert replay == receipt
    task = source.get_task("task:cid:dqk-007")
    assert task is not None and task.status == "retrying"
    assert task.revision == request.parameters["task_revision"] + 1
    assert receipt["plan_root_cid"] == source.snapshot().plan_root_cid
    assert (
        receipt["task_source_repository_tree_id"]
        == source.snapshot().repository_tree_id
        != receipt["repository_tree_id"]
    )
    assert (
        receipt["repository_head_commit"]
        == request.parameters["repository_head_commit"]
    )
    assert receipt["writer_id"] == "local"
    assert receipt["writer_fencing_token"] == 1
    assert len(receipt["lanes"]) == 2
    assert not inspect_incomplete_retry_resets(state_root)
    for lane in receipt["lanes"]:
        assert lane["matched"]
        assert lane["display_attempt_count_before"] == 7
        assert lane["display_attempt_count_after"] == 0
        assert lane["canonical_attempt_count_before"] == 8
        assert lane["canonical_attempt_count_after"] == 0
        assert lane["state_digest_before"] != lane["state_digest_after"]
        assert lane["queue_digest_before"] != lane["queue_digest_after"]
        assert lane["queue_entries_before"][0]["attempt_count"] == 7
        assert lane["queue_entries_after"][0]["attempt_count"] == 7
        assert all(
            value in {0, ""}
            for value in lane["queue_entries_after"][0]["retry"].values()
        )
    events = source.events(cursor=0, limit=20).events
    assert sum(event["event_type"] == "retry_reset_completed" for event in events) == 1


def test_v3_queue_preserves_authority_history_while_clearing_retry_penalties(
    tmp_path: Path,
) -> None:
    _source_db, request, policy, owner, state_root = _fixture(
        tmp_path, lanes=1, queue_schema="persistent_task_queue_v3"
    )

    _execute(request, policy, owner)

    queue = json.loads((state_root / "lane0/task_queue.json").read_text())
    entry = queue["entries"][request.parameters["task_cid"]]
    assert queue["schema"] == "persistent_task_queue_v3"
    assert entry["attempt_count"] == 7
    assert entry["selection_penalty"] == 0
    assert entry["cooldown_until"] == 0
    assert entry["authority_renewal_key"] == "renewal:bound"
    assert entry["authority_renewal_failure_count"] == 2
    assert entry["authority_renewal_quarantined"] is True


@pytest.mark.parametrize(
    "expected_status,status_changed", [("pending", True), ("retrying", False)]
)
def test_exhausted_nonterminal_task_is_reset_without_spurious_same_status_cas(
    tmp_path: Path, expected_status: str, status_changed: bool
) -> None:
    source, request, policy, owner, _state_root = _fixture(
        tmp_path, lanes=1, expected_status=expected_status
    )
    before_revision = source.get_task("DQK-007").revision  # type: ignore[union-attr]

    receipt = _execute(request, policy, owner)

    after = source.get_task("DQK-007")
    assert after is not None and after.status == "retrying"
    assert receipt["status_changed"] is status_changed
    assert after.revision == before_revision + int(status_changed)
    intent_events = [
        event
        for event in source.events(cursor=0, limit=50).events
        if event["event_type"] == "retry_reset_intent"
    ]
    assert len(intent_events) == int(not status_changed)


def test_owner_topology_must_declare_every_lane(tmp_path: Path) -> None:
    source, request, policy, owner, state_root = _fixture(tmp_path, lanes=1)
    extra = _write_lane(state_root, "lane-extra")
    incomplete_owner = replace(
        owner,
        lanes=(
            *owner.lanes,
            LaneBinding(
                extra["state_prefix"], extra["state_path"], extra["queue_path"]
            ),
        ),
    )

    with pytest.raises(DuckDBRetryResetAuthorizationError, match="complete owner"):
        _execute(request, policy, incomplete_owner)

    assert source.get_task("DQK-007").status == "completed"  # type: ignore[union-attr]


def test_undeclared_physical_lane_containing_task_fails_closed(tmp_path: Path) -> None:
    source, request, policy, owner, state_root = _fixture(tmp_path, lanes=1)
    _write_lane(state_root, "lane-extra")

    with pytest.raises(
        DuckDBRetryResetAuthorizationError, match="absent from owner topology"
    ):
        _execute(request, policy, owner)

    assert source.get_task("DQK-007").status == "completed"  # type: ignore[union-attr]


def test_symlink_and_nonfinite_sidecars_fail_closed(tmp_path: Path) -> None:
    source, request, policy, owner, state_root = _fixture(tmp_path, lanes=1)
    queue_path = state_root / "lane0/task_queue.json"
    real_queue = state_root / "queue-outside.json"
    queue_path.rename(real_queue)
    queue_path.symlink_to(real_queue)

    with pytest.raises(DuckDBRetryResetCorruptionError, match="symlink"):
        _execute(request, policy, owner)
    assert source.get_task("DQK-007").status == "completed"  # type: ignore[union-attr]

    queue_path.unlink()
    real_queue.replace(queue_path)
    payload = json.loads(queue_path.read_text())
    payload["entries"][request.parameters["task_cid"]]["cooldown_until"] = float("nan")
    queue_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(DuckDBRetryResetCorruptionError, match="malformed"):
        _execute(request, policy, owner)


def test_active_alias_refuses_even_when_cid_identity_is_missing(tmp_path: Path) -> None:
    source, request, policy, owner, state_root = _fixture(tmp_path, lanes=1)
    state_path = state_root / "lane0/lane0_task_state.json"
    state = json.loads(state_path.read_text())
    state["active_task_id"] = "DQK-007"
    state["active_task_cid"] = ""
    state["task_identities"] = {}
    state["implementation_attempts_by_cid"] = {}
    state_path.write_text(json.dumps(state) + "\n", encoding="utf-8")

    with pytest.raises(DuckDBRetryResetQuiescenceError, match="active"):
        _execute(request, policy, owner)
    assert source.get_task("DQK-007").status == "completed"  # type: ignore[union-attr]


def test_crash_after_database_cas_has_visible_idempotent_recovery(
    tmp_path: Path,
) -> None:
    source, request, policy, owner, state_root = _fixture(tmp_path, lanes=1)

    def crash(phase: str) -> None:
        if phase == "database_mutated":
            raise RuntimeError("injected crash")

    with pytest.raises(RuntimeError, match="injected crash"):
        _execute(request, policy, owner, fault_injector=crash)

    incomplete = inspect_incomplete_retry_resets(state_root)
    assert len(incomplete) == 1
    assert incomplete[0]["phase"] == "prepared"
    assert source.get_task("DQK-007").status == "retrying"  # type: ignore[union-attr]
    lane_state = json.loads((state_root / "lane0/lane0_task_state.json").read_text())
    assert lane_state["implementation_attempts"]["DQK-007"] == 7

    receipt = _execute(request, policy, owner)

    assert receipt["status_after"] == "retrying"
    assert not inspect_incomplete_retry_resets(state_root)
    events = source.events(cursor=0, limit=20).events
    assert sum(event["event_type"] == "status_changed" for event in events) == 2
    assert sum(event["event_type"] == "retry_reset_completed" for event in events) == 1


@pytest.mark.parametrize(
    ("crash_window", "journal_phase"),
    [
        ("lane_sidecar_mutated:lane0", "database_committed"),
        ("receipt_written", "sidecars_committed"),
        ("completion_event_appended", "receipt_committed"),
    ],
)
def test_nontransactional_crash_windows_replay_idempotently(
    tmp_path: Path,
    crash_window: str,
    journal_phase: str,
) -> None:
    source, request, policy, owner, state_root = _fixture(tmp_path, lanes=2)

    def crash(phase: str) -> None:
        if phase == crash_window:
            raise RuntimeError("crash window")

    with pytest.raises(RuntimeError, match="crash window"):
        _execute(request, policy, owner, fault_injector=crash)

    incomplete = inspect_incomplete_retry_resets(state_root)
    assert len(incomplete) == 1
    assert incomplete[0]["phase"] == journal_phase

    receipt = _execute(request, policy, owner)

    assert receipt["status_after"] == "retrying"
    assert not inspect_incomplete_retry_resets(state_root)
    events = source.events(cursor=0, limit=50).events
    assert sum(event["event_type"] == "retry_reset_completed" for event in events) == 1


def test_structurally_valid_self_issued_permit_is_not_trusted(tmp_path: Path) -> None:
    source, request, policy, owner, _state_root = _fixture(tmp_path, lanes=1)
    assert request.authorization is not None
    forged = replace(
        request.authorization,
        grant_ids=(*request.authorization.grant_ids, "grant:self-issued"),
    )
    counterfeit = replace(request, authorization=forged)

    with pytest.raises(
        DuckDBRetryResetAuthorizationError,
        match="not issued by the current policy",
    ):
        _execute(counterfeit, policy, owner)

    assert source.get_task("DQK-007").status == "completed"  # type: ignore[union-attr]


def test_live_master_refuses_reset_even_when_lane_pids_are_absent(
    tmp_path: Path,
) -> None:
    source, request, policy, owner, state_root = _fixture(tmp_path, lanes=1)
    (state_root / "multi_supervisor_runner.pid").write_text(f"{os.getpid()}\n")

    with pytest.raises(
        DuckDBRetryResetQuiescenceError, match="lifecycle owner is live"
    ):
        _execute(request, policy, owner)

    assert source.get_task("DQK-007").status == "completed"  # type: ignore[union-attr]
    incomplete = inspect_incomplete_retry_resets(state_root)
    assert len(incomplete) == 1
    assert incomplete[0]["phase"] == "parent_leased"


def test_malformed_queue_fails_before_database_cas(tmp_path: Path) -> None:
    source, request, policy, owner, state_root = _fixture(tmp_path, lanes=1)
    (state_root / "lane0/task_queue.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(DuckDBRetryResetCorruptionError, match="queue"):
        _execute(request, policy, owner)

    assert source.get_task("DQK-007").status == "completed"  # type: ignore[union-attr]


def test_completed_descendant_blocks_non_cascading_reopen(tmp_path: Path) -> None:
    source, request, policy, owner, _state_root = _fixture(tmp_path, lanes=1)
    descendant = source.get_task("DQK-008")
    assert descendant is not None
    source.compare_and_set_status("DQK-008", descendant.revision, "completed")

    with pytest.raises(DuckDBRetryResetConflict, match="completed descendants"):
        _execute(request, policy, owner)

    assert source.get_task("DQK-007").status == "completed"  # type: ignore[union-attr]


def test_inspect_cli_is_machine_readable_and_nonzero_for_prepared_intent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _source_db, request, policy, owner, state_root = _fixture(tmp_path, lanes=1)

    def crash(phase: str) -> None:
        if phase == "prepared":
            raise RuntimeError("prepared")

    with pytest.raises(RuntimeError, match="prepared"):
        _execute(request, policy, owner, fault_injector=crash)

    assert main(["--inspect-state-root", str(state_root)]) == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["incomplete"][0]["phase"] == "prepared"


def test_inspect_verifies_completed_journal_receipt_and_filename(
    tmp_path: Path,
) -> None:
    _source_db, request, policy, owner, state_root = _fixture(tmp_path, lanes=1)
    receipt = _execute(request, policy, owner)
    assert not inspect_incomplete_retry_resets(state_root)

    receipt_path = (
        Path(receipt["journal_path"]).parents[1]
        / "receipts"
        / (receipt["receipt_cid"] + ".json")
    )
    receipt_payload = json.loads(receipt_path.read_text())
    receipt_payload["writer_id"] = "counterfeit"
    receipt_path.write_text(json.dumps(receipt_payload) + "\n", encoding="utf-8")
    with pytest.raises(DuckDBRetryResetCorruptionError, match="identity"):
        inspect_incomplete_retry_resets(state_root)

    # Restore exact receipt and prove that a renamed journal is rejected too.
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    journal_path = Path(receipt["journal_path"])
    renamed = journal_path.with_name("counterfeit.json")
    journal_path.rename(renamed)
    with pytest.raises(DuckDBRetryResetCorruptionError, match="filename"):
        inspect_incomplete_retry_resets(state_root)


def test_inspector_preserves_completed_authority_after_repository_advances(
    tmp_path: Path,
) -> None:
    _source_db, request, policy, owner, state_root = _fixture(tmp_path, lanes=1)
    _execute(request, policy, owner)
    repository = Path(request.repository_root)
    (repository / "later-generation.txt").write_text(
        "later admitted work\n", encoding="utf-8"
    )
    _git(repository, "add", "later-generation.txt")
    _git(repository, "commit", "-q", "-m", "advance after completed retry")

    assert not inspect_incomplete_retry_resets(state_root)


def test_execute_api_rejects_omitted_durable_intent_before_mutation(
    tmp_path: Path,
) -> None:
    source, request, policy, owner, state_root = _fixture(tmp_path, lanes=1)

    with pytest.raises(
        DuckDBRetryResetAuthorizationError, match="durable execution intent"
    ):
        execute_duckdb_retry_reset(
            request,
            trusted_policy=policy,
            trusted_owner=owner,
            clock_ms=lambda: NOW,
        )

    assert source.get_task("DQK-007").status == "completed"  # type: ignore[union-attr]
    assert not (state_root / "duckdb-retry-reset/journals").exists()
    assert not any(
        event["event_type"] == "retry_reset_completed"
        for event in source.events(cursor=0, limit=100).events
    )


def test_execute_rechecks_freshness_after_lock_delay_before_mutation(
    tmp_path: Path,
) -> None:
    source, request, policy, owner, state_root = _fixture(tmp_path, lanes=1)
    authority = _execution_authority(request, policy, owner)
    assert request.authorization is not None
    current_ms = {"value": NOW}
    first_clock_read = threading.Event()
    clock_guard = threading.Lock()
    clock_reads = 0

    def clock_ms() -> int:
        nonlocal clock_reads
        with clock_guard:
            clock_reads += 1
            if clock_reads == 1:
                first_clock_read.set()
            return current_ms["value"]

    lifecycle_lock = state_root / ".duckdb-retry-reset.lifecycle.lock"
    with ThreadPoolExecutor(max_workers=1) as pool:
        with retry_reset.exclusive_file_lock(lifecycle_lock, timeout_seconds=5):
            future = pool.submit(
                execute_duckdb_retry_reset,
                request,
                trusted_policy=policy,
                trusted_owner=owner,
                **authority,
                clock_ms=clock_ms,
                lock_timeout_seconds=5,
            )
            assert first_clock_read.wait(timeout=2)
            current_ms["value"] = int(request.authorization.expires_at_ms)
        with pytest.raises(
            DuckDBRetryResetAuthorizationError,
            match="while awaiting mutation locks",
        ):
            future.result(timeout=5)

    assert clock_reads >= 2
    assert source.get_task("DQK-007").status == "completed"  # type: ignore[union-attr]
    reset_journal, _receipt_root = retry_reset._journal_paths(state_root, request)
    assert not reset_journal.exists()
    assert not any(
        event["event_type"] == "retry_reset_completed"
        for event in source.events(cursor=0, limit=100).events
    )


def test_prepare_rechecks_freshness_inside_contended_task_source_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, request, policy, owner, state_root = _fixture(tmp_path, lanes=1)
    parent, parent_path = _parent_prepared(request, owner, state_root)
    assert request.authorization is not None
    current_ms = {"value": NOW}
    acquired, release, holder, original = _gate_next_task_source_write(
        monkeypatch,
        method_name="append_event",
        lock_path=source._lock_path,
    )

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                prepare_duckdb_retry_reset_execution_intent,
                request,
                trusted_policy=policy,
                trusted_owner=owner,
                parent_prepared=parent,
                parent_journal_path=parent_path,
                request_file_bytes=request.canonical_bytes(),
                clock_ms=lambda: current_ms["value"],
                lock_timeout_seconds=5,
            )
            assert acquired.wait(timeout=5)
            current_ms["value"] = int(request.authorization.expires_at_ms)
            release.set()
            with pytest.raises(DuckDBRetryResetAuthorizationError, match="expired"):
                future.result(timeout=5)
    finally:
        release.set()
        holder.join(timeout=5)
        monkeypatch.setattr(DuckDBTaskSource, "append_event", original)

    assert not holder.is_alive()
    assert source.get_task("DQK-007").status == "completed"  # type: ignore[union-attr]
    assert not any(
        event["event_type"] == "retry_reset_execution_intent_prepared"
        for event in source.events(cursor=0, limit=100).events
    )
    assert not (state_root / "duckdb-retry-reset/execution-intents").exists()


def test_prepare_rechecks_task_revision_inside_locked_event_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, request, policy, owner, state_root = _fixture(tmp_path, lanes=1)
    parent, parent_path = _parent_prepared(request, owner, state_root)
    entered = threading.Event()
    release = threading.Event()
    original = DuckDBTaskSource.append_event

    def gated(
        selected: DuckDBTaskSource,
        event: Mapping[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if event.get("event_type") == "retry_reset_execution_intent_prepared":
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test event append gate timed out")
        return original(selected, event, *args, **kwargs)

    monkeypatch.setattr(DuckDBTaskSource, "append_event", gated)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                prepare_duckdb_retry_reset_execution_intent,
                request,
                trusted_policy=policy,
                trusted_owner=owner,
                parent_prepared=parent,
                parent_journal_path=parent_path,
                request_file_bytes=request.canonical_bytes(),
                clock_ms=lambda: NOW,
                lock_timeout_seconds=5,
            )
            assert entered.wait(timeout=5)
            task = source.get_task("DQK-007")
            assert task is not None
            source.compare_and_set_status(task.task_cid, task.revision, "failed")
            release.set()
            with pytest.raises(
                TaskSourceConflictError, match="status/revision precondition"
            ):
                future.result(timeout=5)
    finally:
        release.set()

    assert not any(
        event["event_type"] == "retry_reset_execution_intent_prepared"
        for event in source.events(cursor=0, limit=100).events
    )
    assert not (state_root / "duckdb-retry-reset/execution-intents").exists()


def test_pending_cas_expires_inside_task_source_lock_then_replays_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, request, policy, owner, state_root = _fixture(
        tmp_path, lanes=1, expected_status="pending"
    )
    parent, parent_path = _parent_prepared(request, owner, state_root)
    projection = prepare_duckdb_retry_reset_execution_intent(
        request,
        trusted_policy=policy,
        trusted_owner=owner,
        parent_prepared=parent,
        parent_journal_path=parent_path,
        request_file_bytes=request.canonical_bytes(),
        clock_ms=lambda: NOW,
    )
    binding = retry_reset_execution_intent_binding(projection)
    drained = _drained_parent(parent, binding, parent_path)
    assert request.authorization is not None
    current_ms = {"value": NOW}
    execution = {
        "execution_intent": binding,
        "parent_journal": drained,
        "checkout_lease_assertion": {"lease_cid": "lease:held"},
        "checkout_lease_verifier": lambda record: record["lease_cid"]
        == "lease:held",
    }
    acquired, release, holder, original = _gate_next_task_source_write(
        monkeypatch,
        method_name="compare_and_set_status",
        lock_path=source._lock_path,
    )

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                execute_duckdb_retry_reset,
                request,
                trusted_policy=policy,
                trusted_owner=owner,
                **execution,
                clock_ms=lambda: current_ms["value"],
                lock_timeout_seconds=5,
            )
            assert acquired.wait(timeout=5)
            current_ms["value"] = int(request.authorization.expires_at_ms)
            release.set()
            with pytest.raises(
                DuckDBRetryResetAuthorizationError,
                match="DuckDB task-source write lock",
            ):
                future.result(timeout=5)
    finally:
        release.set()
        holder.join(timeout=5)
        monkeypatch.setattr(DuckDBTaskSource, "compare_and_set_status", original)

    assert not holder.is_alive()
    assert source.get_task("DQK-007").status == "pending"  # type: ignore[union-attr]
    reset_path, _receipt_root = retry_reset._journal_paths(state_root, request)
    assert retry_reset._read_bounded_json(reset_path, "retry-reset journal")[
        "phase"
    ] == "prepared"

    expired = int(request.authorization.expires_at_ms) + 1
    receipt = execute_duckdb_retry_reset(
        request,
        trusted_policy=policy,
        trusted_owner=owner,
        **execution,
        clock_ms=lambda: expired,
    )
    replay = execute_duckdb_retry_reset(
        request,
        trusted_policy=policy,
        trusted_owner=owner,
        **execution,
        clock_ms=lambda: expired,
    )
    assert replay == receipt
    events = source.events(cursor=0, limit=100).events
    assert sum(event["event_type"] == "status_changed" for event in events) == 1
    assert sum(event["event_type"] == "retry_reset_completed" for event in events) == 1


def test_retrying_intent_append_expires_inside_task_source_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, request, policy, owner, _state_root = _fixture(
        tmp_path, lanes=1, expected_status="retrying"
    )
    parent, parent_path = _parent_prepared(request, owner, Path(request.state_root))
    projection = prepare_duckdb_retry_reset_execution_intent(
        request,
        trusted_policy=policy,
        trusted_owner=owner,
        parent_prepared=parent,
        parent_journal_path=parent_path,
        request_file_bytes=request.canonical_bytes(),
        clock_ms=lambda: NOW,
    )
    binding = retry_reset_execution_intent_binding(projection)
    drained = _drained_parent(parent, binding, parent_path)
    assert request.authorization is not None
    current_ms = {"value": NOW}
    execution = {
        "execution_intent": binding,
        "parent_journal": drained,
        "checkout_lease_assertion": {"lease_cid": "lease:held"},
        "checkout_lease_verifier": lambda record: record["lease_cid"]
        == "lease:held",
    }
    acquired, release, holder, original = _gate_next_task_source_write(
        monkeypatch,
        method_name="append_event",
        lock_path=source._lock_path,
    )

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                execute_duckdb_retry_reset,
                request,
                trusted_policy=policy,
                trusted_owner=owner,
                **execution,
                clock_ms=lambda: current_ms["value"],
                lock_timeout_seconds=5,
            )
            assert acquired.wait(timeout=5)
            current_ms["value"] = int(request.authorization.expires_at_ms)
            release.set()
            with pytest.raises(
                DuckDBRetryResetAuthorizationError,
                match="DuckDB task-source write lock",
            ):
                future.result(timeout=5)
    finally:
        release.set()
        holder.join(timeout=5)
        monkeypatch.setattr(DuckDBTaskSource, "append_event", original)

    assert not holder.is_alive()
    assert not any(
        event["event_type"] == "retry_reset_intent"
        for event in source.events(cursor=0, limit=100).events
    )
    expired = int(request.authorization.expires_at_ms) + 1
    receipt = execute_duckdb_retry_reset(
        request,
        trusted_policy=policy,
        trusted_owner=owner,
        **execution,
        clock_ms=lambda: expired,
    )
    assert receipt["status_changed"] is False
    events = source.events(cursor=0, limit=100).events
    assert sum(event["event_type"] == "retry_reset_intent" for event in events) == 1
    assert sum(event["event_type"] == "retry_reset_completed" for event in events) == 1


def test_retrying_intent_append_rechecks_task_revision_inside_locked_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, request, policy, owner, state_root = _fixture(
        tmp_path, lanes=1, expected_status="retrying"
    )
    parent, parent_path = _parent_prepared(request, owner, state_root)
    projection = prepare_duckdb_retry_reset_execution_intent(
        request,
        trusted_policy=policy,
        trusted_owner=owner,
        parent_prepared=parent,
        parent_journal_path=parent_path,
        request_file_bytes=request.canonical_bytes(),
        clock_ms=lambda: NOW,
    )
    binding = retry_reset_execution_intent_binding(projection)
    drained = _drained_parent(parent, binding, parent_path)
    entered = threading.Event()
    release = threading.Event()
    original = DuckDBTaskSource.append_event

    def gated(
        selected: DuckDBTaskSource,
        event: Mapping[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if event.get("event_type") == "retry_reset_intent":
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test retry-intent append gate timed out")
        return original(selected, event, *args, **kwargs)

    monkeypatch.setattr(DuckDBTaskSource, "append_event", gated)
    authority = {
        "execution_intent": binding,
        "parent_journal": drained,
        "checkout_lease_assertion": {"lease_cid": "lease:held"},
        "checkout_lease_verifier": lambda record: record["lease_cid"]
        == "lease:held",
    }
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                execute_duckdb_retry_reset,
                request,
                trusted_policy=policy,
                trusted_owner=owner,
                **authority,
                clock_ms=lambda: NOW,
                lock_timeout_seconds=5,
            )
            assert entered.wait(timeout=5)
            task = source.get_task("DQK-007")
            assert task is not None
            source.compare_and_set_status(task.task_cid, task.revision, "failed")
            release.set()
            with pytest.raises(
                TaskSourceConflictError, match="status/revision precondition"
            ):
                future.result(timeout=5)
    finally:
        release.set()

    current = source.get_task("DQK-007")
    assert current is not None and current.status == "failed"
    reset_path, _receipt_root = retry_reset._journal_paths(state_root, request)
    assert retry_reset._read_bounded_json(reset_path, "retry-reset journal")[
        "phase"
    ] == "prepared"
    events = source.events(cursor=0, limit=100).events
    assert not any(event["event_type"] == "retry_reset_intent" for event in events)
    assert not any(
        event["event_type"] == "retry_reset_completed" for event in events
    )


def test_execute_cli_refuses_unorchestrated_intent_omission_and_counterfeit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, request, policy, owner, state_root = _fixture(tmp_path, lanes=1)
    request_path = tmp_path / "retry-request.json"
    request_path.write_text(request.to_json() + "\n", encoding="utf-8")

    assert main(["--request-file", str(request_path)]) == 2
    omission = json.loads(capsys.readouterr().out)
    assert "lifecycle-owner-only" in omission["message"]
    assert "durable execution intent" in omission["message"]
    assert "checkout-lease verifier" in omission["message"]
    assert source.get_task("DQK-007").status == "completed"  # type: ignore[union-attr]
    assert not (state_root / "duckdb-retry-reset/journals").exists()

    counterfeit = _policy_payload(policy)
    counterfeit["policy_revision"] = "policy-revision:counterfeit"
    (state_root / owner.policy_path).write_text(
        json.dumps(counterfeit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert main(["--request-file", str(request_path)]) == 2
    failure = json.loads(capsys.readouterr().out)
    assert "digest does not match" in failure["message"]

    with pytest.raises(SystemExit) as help_exit:
        retry_reset._parse_args(["--help"])
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    assert "lifecycle-owner-only" in help_text
    assert "request to validate and refuse for direct mutation" in help_text
    assert "mutation remains disabled" in help_text


def test_cli_rejects_symlinked_owner_policy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _source_db, request, _policy, owner, state_root = _fixture(tmp_path, lanes=1)
    request_path = tmp_path / "retry-request.json"
    request_path.write_text(request.to_json() + "\n", encoding="utf-8")
    configured = state_root / owner.policy_path
    outside = tmp_path / "policy-outside.json"
    configured.rename(outside)
    configured.symlink_to(outside)

    assert main(["--request-file", str(request_path)]) == 2
    failure = json.loads(capsys.readouterr().out)
    assert "symlink" in failure["message"]


def test_finite_permit_prepares_and_executes_once_after_expiry(
    tmp_path: Path,
) -> None:
    source, request, policy, owner, state_root = _fixture(tmp_path, lanes=1)
    parent, parent_path = _parent_prepared(request, owner, state_root)

    projection = prepare_duckdb_retry_reset_execution_intent(
        request,
        trusted_policy=policy,
        trusted_owner=owner,
        parent_prepared=parent,
        parent_journal_path=parent_path,
        request_file_bytes=request.canonical_bytes(),
        clock_ms=lambda: NOW,
    )
    binding = retry_reset_execution_intent_binding(projection)
    drained = _drained_parent(parent, binding, parent_path)
    assert request.authorization is not None
    expired_clock = lambda: int(request.authorization.expires_at_ms or 0) + 1

    receipt = execute_duckdb_retry_reset(
        request,
        trusted_policy=policy,
        trusted_owner=owner,
        execution_intent=binding,
        parent_journal=drained,
        checkout_lease_assertion={"lease_cid": "lease:held"},
        checkout_lease_verifier=lambda record: record["lease_cid"] == "lease:held",
        clock_ms=expired_clock,
    )
    replay = execute_duckdb_retry_reset(
        request,
        trusted_policy=policy,
        trusted_owner=owner,
        execution_intent=binding,
        parent_journal=drained,
        checkout_lease_assertion={"lease_cid": "lease:held"},
        checkout_lease_verifier=lambda record: record["lease_cid"] == "lease:held",
        clock_ms=expired_clock,
    )

    assert replay == receipt
    assert receipt["execution_intent_cid"] == projection["execution_intent_cid"]
    assert source.get_task("DQK-007").status == "retrying"  # type: ignore[union-attr]
    events = source.events(cursor=0, limit=100).events
    assert sum(
        event["event_type"] == "retry_reset_execution_intent_prepared"
        for event in events
    ) == 1
    assert sum(event["event_type"] == "retry_reset_completed" for event in events) == 1


def test_event_before_projection_and_parent_is_recoverable_after_expiry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _source_db, request, policy, owner, state_root = _fixture(tmp_path, lanes=1)
    parent, parent_path = _parent_prepared(request, owner, state_root)

    def crash(phase: str) -> None:
        if phase == "execution_intent_event_appended":
            raise RuntimeError("event before projection")

    with pytest.raises(RuntimeError, match="event before projection"):
        prepare_duckdb_retry_reset_execution_intent(
            request,
            trusted_policy=policy,
            trusted_owner=owner,
            parent_prepared=parent,
            parent_journal_path=parent_path,
            request_file_bytes=request.canonical_bytes(),
            clock_ms=lambda: NOW,
            fault_injector=crash,
        )

    assert not parent_path.exists()
    incomplete = inspect_incomplete_retry_resets(state_root)
    assert len(incomplete) == 1
    assert incomplete[0]["phase"] == "execution_intent_event_appended"
    assert incomplete[0]["request_id"] == request.request_id
    owner_path = state_root / RETRY_RESET_OWNER_FILE
    owner_bytes = owner_path.read_bytes()
    owner_path.unlink()
    with pytest.raises(
        DuckDBRetryResetAuthorizationError,
        match="no canonical owner configuration",
    ):
        inspect_incomplete_retry_resets(state_root)
    assert main(["--inspect-state-root", str(state_root)]) == 2
    missing_owner = json.loads(capsys.readouterr().out)
    assert missing_owner["ok"] is False
    assert missing_owner["error"] == "DuckDBRetryResetAuthorizationError"

    owner_path.write_bytes(b'{"schema":')
    owner_path.chmod(0o600)
    with pytest.raises(DuckDBRetryResetCorruptionError, match="malformed"):
        inspect_incomplete_retry_resets(state_root)
    assert main(["--inspect-state-root", str(state_root)]) == 2
    corrupt_owner = json.loads(capsys.readouterr().out)
    assert corrupt_owner["ok"] is False
    assert corrupt_owner["error"] == "DuckDBRetryResetCorruptionError"

    owner_path.write_bytes(owner_bytes)
    owner_path.chmod(0o600)
    projection = recover_duckdb_retry_reset_execution_intent(
        request,
        trusted_policy=policy,
        trusted_owner=owner,
        expected_parent_journal_path=parent_path,
    )
    assert projection is not None
    assert Path(projection["projection_path"]).is_file()
    assert projection["parent_prepared"] == parent


def test_inspector_rejects_projection_and_parent_conflicts(tmp_path: Path) -> None:
    _source_db, request, policy, owner, state_root = _fixture(tmp_path, lanes=1)
    parent, parent_path = _parent_prepared(request, owner, state_root)
    projection = prepare_duckdb_retry_reset_execution_intent(
        request,
        trusted_policy=policy,
        trusted_owner=owner,
        parent_prepared=parent,
        parent_journal_path=parent_path,
        request_file_bytes=request.canonical_bytes(),
        clock_ms=lambda: NOW,
    )
    binding = retry_reset_execution_intent_binding(projection)
    drained = _drained_parent(parent, binding, parent_path)
    projection_path = Path(projection["projection_path"])

    forged_projection = json.loads(projection_path.read_text())
    forged_projection["prepared_at_ms"] += 1
    projection_path.write_text(json.dumps(forged_projection) + "\n")
    with pytest.raises(DuckDBRetryResetCorruptionError, match="projection conflicts"):
        inspect_incomplete_retry_resets(state_root)

    projection_path.write_text(json.dumps(projection, indent=2, sort_keys=True) + "\n")
    forged_parent = json.loads(json.dumps(drained))
    forged_parent["execution_intent"]["execution_intent_cid"] = "counterfeit"
    parent_path.write_text(json.dumps(forged_parent) + "\n")
    parent_path.chmod(0o600)
    with pytest.raises(DuckDBRetryResetCorruptionError, match="not canonical"):
        inspect_incomplete_retry_resets(state_root)


def test_event_recovery_rejects_cross_bound_envelope_task_with_rederived_cid(
    tmp_path: Path,
) -> None:
    source, request, policy, owner, state_root = _fixture(tmp_path, lanes=1)
    parent, parent_path = _parent_prepared(request, owner, state_root)

    def crash(phase: str) -> None:
        if phase == "execution_intent_event_appended":
            raise RuntimeError("event before projection")

    with pytest.raises(RuntimeError, match="event before projection"):
        prepare_duckdb_retry_reset_execution_intent(
            request,
            trusted_policy=policy,
            trusted_owner=owner,
            parent_prepared=parent,
            parent_journal_path=parent_path,
            request_file_bytes=request.canonical_bytes(),
            clock_ms=lambda: NOW,
            fault_injector=crash,
        )
    event = next(
        item
        for item in source.events(cursor=0, limit=100).events
        if item["event_type"] == "retry_reset_execution_intent_prepared"
    )
    forged_body = json.loads(json.dumps(event["body"]))
    forged_body["task_cid"] = "task:cid:dqk-008"
    forged_event_cid = content_identity(forged_body)
    connection = duckdb.connect(str(state_root / "workflow.duckdb"))
    try:
        connection.execute(
            "UPDATE task_events SET event_cid = ?, task_cid = ?, body_json = ? "
            "WHERE event_cid = ?",
            [
                forged_event_cid,
                "task:cid:dqk-008",
                json.dumps(
                    forged_body,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                event["event_cid"],
            ],
        )
    finally:
        connection.close()

    with pytest.raises(DuckDBRetryResetCorruptionError, match="content-bound"):
        recover_duckdb_retry_reset_execution_intent(
            request,
            trusted_policy=policy,
            trusted_owner=owner,
            expected_parent_journal_path=parent_path,
        )
    assert not any(
        Path(state_root / "duckdb-retry-reset/execution-intents").glob("*.json")
    )
    assert source.get_task("DQK-007").status == "completed"  # type: ignore[union-attr]


def test_expired_execution_rejects_missing_checkout_assertion_and_orphan_parent(
    tmp_path: Path,
) -> None:
    source, request, policy, owner, state_root = _fixture(tmp_path, lanes=1)
    parent, parent_path = _parent_prepared(request, owner, state_root)
    projection = prepare_duckdb_retry_reset_execution_intent(
        request,
        trusted_policy=policy,
        trusted_owner=owner,
        parent_prepared=parent,
        parent_journal_path=parent_path,
        request_file_bytes=request.canonical_bytes(),
        clock_ms=lambda: NOW,
    )
    binding = retry_reset_execution_intent_binding(projection)
    drained = _drained_parent(parent, binding, parent_path)
    assert request.authorization is not None
    expired_clock = lambda: int(request.authorization.expires_at_ms or 0) + 1

    with pytest.raises(
        DuckDBRetryResetAuthorizationError, match="checkout-lease assertion"
    ):
        execute_duckdb_retry_reset(
            request,
            trusted_policy=policy,
            trusted_owner=owner,
            execution_intent=binding,
            parent_journal=drained,
            clock_ms=lambda: NOW,
        )
    with pytest.raises(
        DuckDBRetryResetAuthorizationError, match="checkout-lease assertion"
    ):
        execute_duckdb_retry_reset(
            request,
            trusted_policy=policy,
            trusted_owner=owner,
            execution_intent=binding,
            parent_journal=drained,
            clock_ms=expired_clock,
        )
    with pytest.raises(
        DuckDBRetryResetAuthorizationError, match="was not accepted"
    ):
        execute_duckdb_retry_reset(
            request,
            trusted_policy=policy,
            trusted_owner=owner,
            execution_intent=binding,
            parent_journal=drained,
            checkout_lease_assertion={"lease_cid": "lease:original"},
            checkout_lease_verifier=lambda record: record["lease_cid"]
            == "lease:changed",
            clock_ms=expired_clock,
        )
    assert source.get_task("DQK-007").status == "completed"  # type: ignore[union-attr]

    forged = {**drained, "drained_cid": "sha256:" + "f" * 64}
    parent_path.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n")
    with pytest.raises(DuckDBRetryResetCorruptionError, match="exact quiescent"):
        execute_duckdb_retry_reset(
            request,
            trusted_policy=policy,
            trusted_owner=owner,
            execution_intent=binding,
            parent_journal=forged,
            checkout_lease_assertion={"lease_cid": "lease:held"},
            checkout_lease_verifier=lambda _record: True,
            clock_ms=expired_clock,
        )


def test_execution_intent_rejects_a_falsely_drained_live_master_session(
    tmp_path: Path,
) -> None:
    source, request, policy, owner, state_root = _fixture(tmp_path, lanes=1)
    parent, parent_path = _parent_prepared(request, owner, state_root)
    projection = prepare_duckdb_retry_reset_execution_intent(
        request,
        trusted_policy=policy,
        trusted_owner=owner,
        parent_prepared=parent,
        parent_journal_path=parent_path,
        request_file_bytes=request.canonical_bytes(),
        clock_ms=lambda: NOW,
    )
    binding = retry_reset_execution_intent_binding(projection)
    falsely_drained = {
        **parent,
        "phase": "leased",
        "execution_intent": binding,
        "drain_process_tree": [parent["old_master"]["actual"]],
        "drain_started_at": "2026-08-09T00:01:00+00:00",
        "drained_at": "2026-08-09T00:02:00+00:00",
    }
    falsely_drained["drain_cid"] = retry_reset._parent_drain_cid(
        falsely_drained
    )
    falsely_drained["drained_cid"] = retry_reset._parent_drained_cid(
        falsely_drained
    )
    parent_path.parent.mkdir(parents=True, exist_ok=True)
    parent_path.write_text(
        json.dumps(falsely_drained, indent=2, sort_keys=True) + "\n"
    )

    with pytest.raises(
        DuckDBRetryResetQuiescenceError, match="process identities remain live"
    ):
        execute_duckdb_retry_reset(
            request,
            trusted_policy=policy,
            trusted_owner=owner,
            execution_intent=binding,
            parent_journal=falsely_drained,
            checkout_lease_assertion={"lease_cid": "lease:claimed"},
            checkout_lease_verifier=lambda _record: True,
            clock_ms=lambda: NOW,
        )

    assert source.get_task("DQK-007").status == "completed"  # type: ignore[union-attr]


def test_concurrent_prepare_and_execute_are_exactly_once(
    tmp_path: Path,
) -> None:
    source, request, policy, owner, state_root = _fixture(tmp_path, lanes=1)
    parent, parent_path = _parent_prepared(request, owner, state_root)

    def prepare() -> dict[str, Any]:
        return prepare_duckdb_retry_reset_execution_intent(
            request,
            trusted_policy=policy,
            trusted_owner=owner,
            parent_prepared=parent,
            parent_journal_path=parent_path,
            request_file_bytes=request.canonical_bytes(),
            clock_ms=lambda: NOW,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        projections = list(pool.map(lambda _index: prepare(), range(2)))
    assert projections[0] == projections[1]
    binding = retry_reset_execution_intent_binding(projections[0])
    drained = _drained_parent(parent, binding, parent_path)
    assert request.authorization is not None
    expired = int(request.authorization.expires_at_ms or 0) + 1

    def execute() -> dict[str, Any]:
        return execute_duckdb_retry_reset(
            request,
            trusted_policy=policy,
            trusted_owner=owner,
            execution_intent=binding,
            parent_journal=drained,
            checkout_lease_assertion={"lease_cid": "lease:held"},
            checkout_lease_verifier=lambda _record: True,
            clock_ms=lambda: expired,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(lambda _index: execute(), range(2)))
    assert receipts[0] == receipts[1]
    events = source.events(cursor=0, limit=100).events
    assert sum(
        event["event_type"] == "retry_reset_execution_intent_prepared"
        for event in events
    ) == 1
    assert sum(event["event_type"] == "retry_reset_completed" for event in events) == 1


def test_prepared_execution_intent_rejects_changed_head_policy_and_writer(
    tmp_path: Path,
) -> None:
    _source_db, request, policy, owner, state_root = _fixture(
        tmp_path / "head", lanes=1
    )
    parent, parent_path = _parent_prepared(request, owner, state_root)
    projection = prepare_duckdb_retry_reset_execution_intent(
        request,
        trusted_policy=policy,
        trusted_owner=owner,
        parent_prepared=parent,
        parent_journal_path=parent_path,
        request_file_bytes=request.canonical_bytes(),
        clock_ms=lambda: NOW,
    )
    assert projection["execution_intent_cid"]

    repository = Path(request.repository_root)
    (repository / "changed.txt").write_text("changed\n", encoding="utf-8")
    _git(repository, "add", "changed.txt")
    _git(repository, "commit", "-q", "-m", "change generation")
    with pytest.raises(DuckDBRetryResetConflict, match="current clean generation"):
        recover_duckdb_retry_reset_execution_intent(
            request,
            trusted_policy=policy,
            trusted_owner=owner,
            expected_parent_journal_path=parent_path,
        )

    _source_db, request, policy, owner, state_root = _fixture(
        tmp_path / "policy", lanes=1
    )
    parent, parent_path = _parent_prepared(request, owner, state_root)
    prepare_duckdb_retry_reset_execution_intent(
        request,
        trusted_policy=policy,
        trusted_owner=owner,
        parent_prepared=parent,
        parent_journal_path=parent_path,
        request_file_bytes=request.canonical_bytes(),
        clock_ms=lambda: NOW,
    )
    policy_path = state_root / owner.policy_path
    changed_policy = json.loads(policy_path.read_bytes())
    changed_policy["policy_revision"] = "policy-revision:changed"
    policy_path.write_text(json.dumps(changed_policy) + "\n", encoding="utf-8")
    with pytest.raises(
        DuckDBRetryResetAuthorizationError, match="owner-pinned digest"
    ):
        recover_duckdb_retry_reset_execution_intent(
            request,
            trusted_policy=policy,
            trusted_owner=owner,
            expected_parent_journal_path=parent_path,
        )

    source, request, policy, owner, state_root = _fixture(
        tmp_path / "writer", lanes=1
    )
    parent, parent_path = _parent_prepared(request, owner, state_root)
    prepare_duckdb_retry_reset_execution_intent(
        request,
        trusted_policy=policy,
        trusted_owner=owner,
        parent_prepared=parent,
        parent_journal_path=parent_path,
        request_file_bytes=request.canonical_bytes(),
        clock_ms=lambda: NOW,
    )
    source.acquire_writer("foreign", expected_fencing_token=1)
    with pytest.raises(DuckDBRetryResetConflict, match="writer owner/fence"):
        recover_duckdb_retry_reset_execution_intent(
            request,
            trusted_policy=policy,
            trusted_owner=owner,
            expected_parent_journal_path=parent_path,
        )
    assert source.get_task("DQK-007").status == "completed"  # type: ignore[union-attr]


def test_prepared_execution_intent_rejects_changed_task_and_conflicting_event(
    tmp_path: Path,
) -> None:
    source, request, policy, owner, state_root = _fixture(
        tmp_path / "task", lanes=1
    )
    parent, parent_path = _parent_prepared(request, owner, state_root)
    prepare_duckdb_retry_reset_execution_intent(
        request,
        trusted_policy=policy,
        trusted_owner=owner,
        parent_prepared=parent,
        parent_journal_path=parent_path,
        request_file_bytes=request.canonical_bytes(),
        clock_ms=lambda: NOW,
    )
    task = source.get_task("DQK-007")
    assert task is not None
    source.compare_and_set_status(task.task_cid, task.revision, "failed")
    with pytest.raises(DuckDBRetryResetConflict, match="task status/revision"):
        recover_duckdb_retry_reset_execution_intent(
            request,
            trusted_policy=policy,
            trusted_owner=owner,
            expected_parent_journal_path=parent_path,
        )


def test_execution_intent_rejects_forged_parent_and_event_envelope(
    tmp_path: Path,
) -> None:
    for index, field in enumerate(
        ("checkout", "environment", "master", "bootstrap", "owner")
    ):
        _source_db, request, policy, owner, state_root = _fixture(
            tmp_path / f"parent-{index}", lanes=1
        )
        parent, parent_path = _parent_prepared(request, owner, state_root)
        forged = json.loads(json.dumps(parent))
        if field == "checkout":
            forged["checkout_binding"][0]["repository_id"] = "repository:forged"
        elif field == "environment":
            forged["environment"]["receipt_sha256"] = "sha256:" + "f" * 64
        elif field == "master":
            forged["old_master"]["actual"]["cmdline_sha256"] = "sha256:" + "f" * 64
        elif field == "bootstrap":
            forged["old_master"]["stored"][
                "bootstrap_completion_evidence_id"
            ] = "baguqeera" + "a" * 52
        else:
            forged["lifecycle_owners"][-1]["pid"] += 1
        forged["intent_cid"] = retry_reset._parent_intent_cid(forged)
        with pytest.raises(DuckDBRetryResetAuthorizationError):
            prepare_duckdb_retry_reset_execution_intent(
                request,
                trusted_policy=policy,
                trusted_owner=owner,
                parent_prepared=forged,
                parent_journal_path=parent_path,
                request_file_bytes=request.canonical_bytes(),
                clock_ms=lambda: NOW,
            )

    source, request, policy, owner, state_root = _fixture(
        tmp_path / "envelope", lanes=1
    )
    parent, parent_path = _parent_prepared(request, owner, state_root)
    prepare_duckdb_retry_reset_execution_intent(
        request,
        trusted_policy=policy,
        trusted_owner=owner,
        parent_prepared=parent,
        parent_journal_path=parent_path,
        request_file_bytes=request.canonical_bytes(),
        clock_ms=lambda: NOW,
    )
    event = next(
        item
        for item in source.events(cursor=0, limit=100).events
        if item["event_type"] == "retry_reset_execution_intent_prepared"
    )
    forged_body = json.loads(json.dumps(event["body"]))
    forged_body["lease"]["lease_id"] = "lease:forged"
    connection = duckdb.connect(str(state_root / "workflow.duckdb"))
    try:
        connection.execute(
            "UPDATE task_events SET body_json = ? WHERE event_cid = ?",
            [
                json.dumps(
                    forged_body,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                event["event_cid"],
            ],
        )
    finally:
        connection.close()
    with pytest.raises(DuckDBRetryResetCorruptionError, match="content-bound"):
        recover_duckdb_retry_reset_execution_intent(
            request,
            trusted_policy=policy,
            trusted_owner=owner,
            expected_parent_journal_path=parent_path,
        )

    source, request, policy, owner, state_root = _fixture(
        tmp_path / "event", lanes=1
    )
    parent, parent_path = _parent_prepared(request, owner, state_root)
    prepare_duckdb_retry_reset_execution_intent(
        request,
        trusted_policy=policy,
        trusted_owner=owner,
        parent_prepared=parent,
        parent_journal_path=parent_path,
        request_file_bytes=request.canonical_bytes(),
        clock_ms=lambda: NOW,
    )
    source.append_event(
        {
            "schema": retry_reset.RETRY_RESET_EXECUTION_INTENT_EVENT_SCHEMA,
            "event_cid": "conflicting:execution-intent",
            "event_type": "retry_reset_execution_intent_prepared",
            "task_cid": "DQK-007",
            "request_id": request.request_id,
            "execution_intent_cid": "conflicting:execution-intent",
            "intent": {"schema": retry_reset.RETRY_RESET_EXECUTION_INTENT_SCHEMA},
        },
        lease={"lease_id": request.lease_id, "fencing_token": 1},
        fence=1,
        writer_id="local",
    )
    with pytest.raises(DuckDBRetryResetConflict, match="duplicate"):
        recover_duckdb_retry_reset_execution_intent(
            request,
            trusted_policy=policy,
            trusted_owner=owner,
            expected_parent_journal_path=parent_path,
        )
