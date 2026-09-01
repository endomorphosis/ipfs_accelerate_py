"""Fail-closed tests for the legacy terminal no-model-effect bridge proof."""

from __future__ import annotations

import fcntl
import json
import os
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.agent_supervisor.runtime import provider_failure_policy
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    _NO_PROVIDER_EVENT_FIELDS,
    _SETUP_EVENT_FIELD_VARIANTS,
    DATABASE_PORTAL_TERMINAL_NO_EFFECT_ROUTE_REARM_EVIDENCE_FIELDS,
    DATABASE_PORTAL_TERMINAL_NO_EFFECT_ROUTE_REARM_EVIDENCE_SCHEMA,
    DatabasePortalAttemptPaths,
    DatabasePortalBridgeError,
    DatabasePortalExecutionBridge,
)


def _bridge(tmp_path: Path) -> DatabasePortalExecutionBridge:
    return DatabasePortalExecutionBridge(
        task_source=object(),
        attempt_root=tmp_path,
        portal_factory=lambda *_args: None,
    )


def _legacy_route() -> llm_router.AgentImplementationRoutePlan:
    return llm_router.resolve_agent_implementation_route(
        primary_provider_id="grok_cli",
        primary_model_id="grok-4.6",
        fallback_provider_id="codex",
        fallback_model_id="gpt-5.6-terra",
        fallback_trigger="primary_quota_exhausted",
        fallback_reasoning_effort="medium",
    )


def _exact_log_fixture(tmp_path: Path) -> tuple[
    DatabasePortalExecutionBridge,
    bytes,
    list[str],
    dict[str, object],
]:
    bridge = _bridge(tmp_path)
    route = _legacy_route()
    workspace = str(tmp_path / "workspace")
    branch = "implementation/pctdd-005-test-attempt-1-1"
    baseline = "a" * 40
    nonce = "b" * 64
    fallback = [
        str(tmp_path / "bin" / "codex"),
        "exec",
        "--ignore-user-config",
        "--ignore-rules",
        "--ephemeral",
        "-s",
        "workspace-write",
        "-C",
        workspace,
        "-m",
        "gpt-5.6-terra",
        "-c",
        'model_reasoning_effort="medium"',
        "-",
    ]
    command = [
        sys.executable,
        "-m",
        "ipfs_accelerate_py.agent_supervisor.grok_cli_runner",
        "--workspace",
        workspace,
        "--model",
        "grok-4.6",
        "--max-turns",
        "100000",
        "--mode",
        "agent",
        "--codex-fallback-reasoning-effort",
        "medium",
        "--codex-fallback-command-json",
        json.dumps(fallback, separators=(",", ":")),
        "--grok-bin",
        str(tmp_path / "bin" / "grok"),
        "--grok-failure-receipt-nonce",
        nonce,
        "--agent-implementation-route-json",
        json.dumps(route.as_binding_dict(), sort_keys=True, separators=(",", ":")),
    ]
    receipt = provider_failure_policy.build_grok_failure_receipt(
        probe_stderr_text="Grok Build usage balance exhausted",
        nonce=nonce,
        model="grok-4.6",
        probe_returncode=1,
        primary_dispatched=False,
    )
    outcome = provider_failure_policy.build_grok_route_outcome(
        receipt=receipt,
        route_plan=route.as_binding_dict(),
        decision="denied",
        verifier_status="not_run",
        fallback_dispatched=False,
        fallback_returncode=None,
    )
    lines = [
        "Task: PCTDD-005 Define prepared canonical block and batch contracts",
        "Started: 2026-09-01T11:29:47.325613+00:00",
        f"Workspace: {workspace}",
        f"Branch: {branch}",
        f"Baseline: {baseline}",
        f"Command: {shlex.join(command)}",
        "",
        provider_failure_policy.render_grok_failure_receipt(receipt),
        "Typed Grok preflight did not authorize fallback; Codex fallback is forbidden",
        provider_failure_policy.render_grok_route_outcome(outcome),
    ]
    context = {
        "task_alias": "PCTDD-005",
        "task_title": "Define prepared canonical block and batch contracts",
        "expected_worktree": workspace,
        "expected_branch": branch,
        "expected_baseline": baseline,
        "runner_returncode": 1,
    }
    return bridge, ("\n".join(lines) + "\n").encode(), command, context


def test_exact_full_log_is_admitted_and_extra_or_duplicate_lines_reject(
    tmp_path: Path,
) -> None:
    bridge, raw, command, context = _exact_log_fixture(tmp_path)

    admitted = bridge._terminal_no_effect_route_log_admission(
        raw_log=raw,
        command=command,
        **context,
    )

    assert admitted is not None
    assert admitted["route_id"] == (
        "agent-supervisor-grok45-terra56-medium-hard-quota-v1"
    )
    assert admitted["failure_class"] == "hard_quota_exhausted"
    assert admitted["verifier_status"] == "not_run"
    assert bridge._terminal_no_effect_route_log_admission(
        raw_log=raw + b"unexpected secondary dispatch\n",
        command=command,
        **context,
    ) is None
    receipt_line = next(
        line
        for line in raw.splitlines()
        if line.startswith(
            provider_failure_policy.GROK_FAILURE_RECEIPT_PREFIX.encode()
        )
    )
    assert bridge._terminal_no_effect_route_log_admission(
        raw_log=raw + receipt_line + b"\n",
        command=command,
        **context,
    ) is None


def _open_attempt(path: Path) -> int:
    return os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )


def _populate_legacy_log_directory(
    logs: Path,
    *,
    alias: str = "pctdd-005",
) -> Path:
    log = logs / f"{alias}-attempt-1.log"
    log.write_bytes(b"sealed\n")
    log.chmod(0o600)
    for name in (
        f"{alias}-attempt-1-context-receipt.json",
        f"{alias}-base-context-capsule.json",
        f"{alias}-base-context-receipt.json",
        f"{alias}-diagnostic-receipt.json",
        f"{alias}-diagnostic-state.json",
    ):
        artifact = logs / name
        artifact.write_bytes(b"{}\n")
        artifact.chmod(0o664)
    return log


def test_terminal_log_reader_rejects_symlink_and_hardlink(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt"
    attempt.mkdir(mode=0o700)
    logs = attempt / "implementation-logs"
    logs.mkdir(mode=0o775)
    logs.chmod(0o775)
    log = _populate_legacy_log_directory(logs)
    attempt_fd = _open_attempt(attempt)
    try:
        raw, _fingerprint, identity = (
            DatabasePortalExecutionBridge._read_exact_terminal_no_effect_log(
                attempt_fd,
                log.name,
            )
        )
    finally:
        os.close(attempt_fd)
    assert raw == b"sealed\n"
    assert identity.startswith("sha256:")

    hardlink = logs / "duplicate.log"
    os.link(log, hardlink)
    attempt_fd = _open_attempt(attempt)
    try:
        with pytest.raises(DatabasePortalBridgeError):
            DatabasePortalExecutionBridge._read_exact_terminal_no_effect_log(
                attempt_fd,
                log.name,
            )
    finally:
        os.close(attempt_fd)
    hardlink.unlink()

    extra = logs / "unexpected-regular.json"
    extra.write_bytes(b"{}\n")
    extra.chmod(0o664)
    attempt_fd = _open_attempt(attempt)
    try:
        with pytest.raises(DatabasePortalBridgeError):
            DatabasePortalExecutionBridge._read_exact_terminal_no_effect_log(
                attempt_fd,
                log.name,
            )
    finally:
        os.close(attempt_fd)
    extra.unlink()

    fifo = logs / "unexpected-fifo"
    os.mkfifo(fifo, 0o664)
    attempt_fd = _open_attempt(attempt)
    try:
        with pytest.raises(DatabasePortalBridgeError):
            DatabasePortalExecutionBridge._read_exact_terminal_no_effect_log(
                attempt_fd,
                log.name,
            )
    finally:
        os.close(attempt_fd)
    fifo.unlink()

    auxiliary = logs / "pctdd-005-diagnostic-state.json"
    auxiliary.unlink()
    auxiliary.symlink_to(log.name)
    attempt_fd = _open_attempt(attempt)
    try:
        with pytest.raises(DatabasePortalBridgeError):
            DatabasePortalExecutionBridge._read_exact_terminal_no_effect_log(
                attempt_fd,
                log.name,
            )
    finally:
        os.close(attempt_fd)
    auxiliary.unlink()
    auxiliary.write_bytes(b"{}\n")
    auxiliary.chmod(0o664)

    log.unlink()
    os.mkfifo(log, 0o600)
    attempt_fd = _open_attempt(attempt)
    try:
        with pytest.raises(DatabasePortalBridgeError):
            DatabasePortalExecutionBridge._read_exact_terminal_no_effect_log(
                attempt_fd,
                log.name,
            )
    finally:
        os.close(attempt_fd)
    log.unlink()
    log.write_bytes(b"sealed\n")
    log.chmod(0o600)

    for child in logs.iterdir():
        child.unlink()
    logs.rmdir()
    outside = tmp_path / "outside-logs"
    outside.mkdir(mode=0o775)
    outside.chmod(0o775)
    (outside / log.name).write_bytes(b"copied\n")
    (outside / log.name).chmod(0o600)
    logs.symlink_to(outside, target_is_directory=True)
    attempt_fd = _open_attempt(attempt)
    try:
        with pytest.raises(DatabasePortalBridgeError):
            DatabasePortalExecutionBridge._read_exact_terminal_no_effect_log(
                attempt_fd,
                log.name,
            )
    finally:
        os.close(attempt_fd)


def test_terminal_log_end_snapshot_revalidation_rejects_mutation_and_replacement(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "attempt"
    attempt.mkdir(mode=0o700)
    logs = attempt / "implementation-logs"
    logs.mkdir(mode=0o775)
    logs.chmod(0o775)
    log = _populate_legacy_log_directory(logs)
    attempt_fd = _open_attempt(attempt)
    try:
        raw, fingerprint, identity = (
            DatabasePortalExecutionBridge._read_exact_terminal_no_effect_log(
                attempt_fd,
                log.name,
            )
        )
        log.write_bytes(b"mutated\n")
        log.chmod(0o600)
        with pytest.raises(DatabasePortalBridgeError):
            DatabasePortalExecutionBridge._revalidate_exact_terminal_no_effect_log(
                attempt_fd,
                log.name,
                expected_raw=raw,
                expected_fingerprint=fingerprint,
                expected_identity_digest=identity,
            )
        log.unlink()
        log.write_bytes(raw)
        log.chmod(0o600)
        with pytest.raises(DatabasePortalBridgeError):
            DatabasePortalExecutionBridge._revalidate_exact_terminal_no_effect_log(
                attempt_fd,
                log.name,
                expected_raw=raw,
                expected_fingerprint=fingerprint,
                expected_identity_digest=identity,
            )
    finally:
        os.close(attempt_fd)


def test_current_effective_kernel_event_cannot_enter_legacy_migration(
    tmp_path: Path,
) -> None:
    bridge = _bridge(tmp_path)
    alias = "PCTDD-005"
    task_cid = "baguqeera" + ("a" * 52)
    task_key = "task/v1/" + ("b" * 64)
    board = "parallel-content-sealing-proof-carrying-tdd-v1"
    identity = {
        "task_id": alias,
        "title": "current event",
        "canonical_task_key": task_key,
        "canonical_task_cid": task_cid,
        "board_namespace": board,
    }
    selected = {"type": "task_selected", **identity, "track": "pctdd-g021"}
    diagnostics = [
        {"type": "implementation_resource_claim_lock_cleared"},
        *({"type": "nested_submodule_initialization_guarded"} for _ in range(5)),
        *({"type": "local_submodule_source_discovered"} for _ in range(7)),
        {"type": "nested_submodule_initialization_guarded"},
    ]
    attempt_identity = {
        "task_id": alias,
        "canonical_task_key": task_key,
        "canonical_task_cid": task_cid,
        "board_namespace": board,
        "attempt": 1,
    }
    legacy_pre_kernel_fields = _NO_PROVIDER_EVENT_FIELDS[
        "pre_implementation_kernel_evaluated"
    ]
    current_pre_kernel_fields = legacy_pre_kernel_fields | {
        "effective_provider_authorized",
        "effective_skip_provider",
        "effective_reason_code",
    }
    assert current_pre_kernel_fields in _SETUP_EVENT_FIELD_VARIANTS[
        "pre_implementation_kernel_evaluated"
    ]
    assert current_pre_kernel_fields != legacy_pre_kernel_fields
    current_pre_kernel = {
        "type": "pre_implementation_kernel_evaluated",
        "timestamp": "2026-09-01T11:29:47+00:00",
        "stream_id": "event-log:sha256:" + ("1" * 64),
        "snapshot_id": "event-log-snapshot:sha256:" + ("2" * 64),
        "sequence": 17,
        "previous_event_id": "sha256:" + ("3" * 64),
        "event_id": "sha256:" + ("4" * 64),
        "event": "pre_implementation_kernel_evaluated",
        **attempt_identity,
        "disposition": "abstain_review",
        "provider_authorized": False,
        "provider_hook_count": 0,
        "skip_provider": True,
        "reason_code": "no_analytical_close",
        "receipt_cid": "baguqeera" + ("c" * 52),
        "residual_packet_cid": "",
        "analytical_candidate_count": 0,
        "kernel_receipt": {},
        "interface": "ImplementationDaemon@pre_implementation_kernel",
        "effective_provider_authorized": True,
        "effective_skip_provider": False,
        "effective_reason_code": "no_analytical_close_provider_dispatched",
    }
    assert frozenset(current_pre_kernel) == current_pre_kernel_fields
    bridge._validate_no_provider_event_shape(current_pre_kernel)
    events = [
        selected,
        *diagnostics,
        {"type": "implementation_protected_path_snapshot_recorded", **attempt_identity},
        {"type": "implementation_started", **attempt_identity},
        current_pre_kernel,
        {"type": "implementation_protected_path_snapshot_cleared", **attempt_identity},
        {"type": "worktree_pool_lease_released"},
        {"type": "implementation_finished", "task_cid": task_cid, **attempt_identity},
        {"type": "daemon_pass"},
    ]
    root = tmp_path / ("c" * 24)
    paths = DatabasePortalAttemptPaths(
        root=root,
        task_projection=root / "task-projection.md",
        binding=root / "database-attempt-binding.json",
        state=root / "portal-task-state.json",
        strategy=root / "portal-strategy.json",
        events=root / "portal-events.jsonl",
        implementation_logs=root / "implementation-logs",
        reconciliation=root / "database-attempt-reconciliations",
    )
    attempt = SimpleNamespace(
        attempt_id="attempt:test",
        claim_id="claim:test",
        task_cid=task_cid,
        attempt_number=2,
        owner_session_id="owner",
        lease_id="lease:test",
        fencing_token=1,
        fence_epoch=1,
    )

    assert bridge._terminal_no_effect_route_rearm_evidence(
        attempt,
        receipt={},
        paths=paths,
        binding={"task_alias": alias, "task_revision": 1},
        durable_binding={},
        identity=identity,
        projection_track="pctdd-g021",
        projection_status="ready",
        directory_names=[
            ".implementation.lock.update.lock",
            ".portal-events.jsonl.lock",
            "database-attempt-binding.json",
            "implementation-logs",
            "implementation_checkpoints",
            "portal-events.jsonl",
            "portal-events.jsonl.manifest.json",
            "portal-strategy.json",
            "portal-task-state.json",
            "task-projection.md",
            "task_queue.json",
        ],
        state={},
        state_digest="sha256:" + ("d" * 64),
        events=events,
        manifest={},
        log_raw=b"not reached",
        log_relative_path="implementation-logs/pctdd-005-attempt-1.log",
        log_identity_digest="sha256:" + ("e" * 64),
    ) is None


def test_terminal_no_effect_schema_exports_one_closed_self_hash_field() -> None:
    assert DATABASE_PORTAL_TERMINAL_NO_EFFECT_ROUTE_REARM_EVIDENCE_SCHEMA.endswith(
        "terminal-no-effect-route-rearm-evidence@1"
    )
    assert "evidence_id" in (
        DATABASE_PORTAL_TERMINAL_NO_EFFECT_ROUTE_REARM_EVIDENCE_FIELDS
    )


def test_shared_event_lock_rejects_fifo_and_live_writer_without_waiting(
    tmp_path: Path,
) -> None:
    bridge = _bridge(tmp_path)
    lock_name = ".portal-events.jsonl.lock"
    lock_path = tmp_path / lock_name
    directory_fd = os.open(
        tmp_path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.mkfifo(lock_path, mode=0o600)
        with pytest.raises(
            DatabasePortalBridgeError,
            match="event lock is not an exact private file",
        ):
            with bridge._shared_private_event_lock(
                directory_fd,
                {lock_name},
            ):
                pytest.fail("FIFO lock must not be admitted")

        lock_path.unlink()
        lock_path.write_bytes(b"")
        lock_path.chmod(0o600)
        writer_fd = os.open(lock_path, os.O_RDWR)
        try:
            fcntl.flock(writer_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with pytest.raises(
                DatabasePortalBridgeError,
                match="event lock is currently unavailable",
            ):
                with bridge._shared_private_event_lock(
                    directory_fd,
                    {lock_name},
                ):
                    pytest.fail("live writer lock must not be shared")
        finally:
            fcntl.flock(writer_fd, fcntl.LOCK_UN)
            os.close(writer_fd)
    finally:
        os.close(directory_fd)
