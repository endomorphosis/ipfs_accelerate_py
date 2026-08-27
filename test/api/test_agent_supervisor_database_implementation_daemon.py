"""Tests for DatabaseImplementationDaemon@1 cutover (DQP-018).

Evidence subset: ready selection, strict shards, lost response, provider
capacity, hard quota, timeout, cancellation, crash, restart, stale worker,
status parity.

Acceptance: Four daemon processes claim distinct work; no task status is
updated in Markdown under database authority; JSON queue/status/events/PID
projections can be absent; crash/restart resumes from committed phase and does
not duplicate provider/effect work.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ipfs_accelerate_py.agent_implementation_route import (
    resolve_agent_implementation_route,
)
from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
    checkout_lock_metadata,
    checkout_mutation_lock_path,
    checkout_repository_id,
)
from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
    DatabaseCoordinationConflictError,
    DatabaseCoordinationError,
)
from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import MergeQueue
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    DATABASE_PROGRAM_JSON_ENV,
    DatabaseProgramConfig,
    SupervisorTrack,
    terminal_task_state_fields,
)
from ipfs_accelerate_py.agent_supervisor.runtime.provider_failure_policy import (
    build_grok_failure_receipt,
    build_grok_route_outcome,
)
from ipfs_accelerate_py.agent_supervisor.task_sources import (
    database_task_source as database_task_source_module,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    canonical_json_bytes as task_body_canonical_json_bytes,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    install_control_plane_schema,
    install_datasets_authoritative_operational_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    TaskSourceConflictError as DatabaseTaskSourceConflictError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    QUACK_OWNER_COMMAND_MAX_BYTES,
    DuckDBConnectionPolicyError,
    connect_duckdb_with_policy,
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    MAX_BODY_BYTES as MAX_TASK_BODY_BYTES,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    TASK_REVISION_HISTORY_PROJECTION_SCHEMA,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.task_source import (
    COMPLETED_STATUSES as TASK_SOURCE_COMPLETED_STATUSES,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    TYPED_DATABASE_CLAIM_PROCESS_SCHEMA,
    _process_birth_content_id,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as implementation_daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon_runner as daemon_runner,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DATABASE_PORTAL_CAPACITY_RETRY_SCHEMA,
    DATABASE_PORTAL_COMPLETION_BINDING_SCHEMA,
    DATABASE_PORTAL_CONSUMED_ATTEMPT_RETRY_SCHEMA,
    DATABASE_PORTAL_CONSUMED_NO_PROGRESS_SCHEMA,
    DATABASE_PORTAL_PROTECTED_PATH_PRESERVATION_SCHEMA,
    DATABASE_PORTAL_PROTECTED_RECONCILIATION_SELF_LOCK_SCHEMA,
    DATABASE_PORTAL_VALIDATION_RETRY_SCHEMA,
    DatabasePortalBridgeConsumedNoProgressError,
    DatabasePortalBridgeDeferred,
    DatabasePortalBridgeError,
    DatabasePortalCandidateRetry,
    DatabasePortalCapacityRetry,
    DatabasePortalConsumedAttemptTerminal,
    DatabasePortalExecutionBridge,
    DatabasePortalProtectedPathPreserved,
    DatabasePortalValidationRetry,
    database_portal_consumed_no_progress_fingerprint,
    database_portal_task_contract_digest,
    is_protected_checkout_setup_block,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    ATTEMPT_PHASE_BLOCKED,
    ATTEMPT_PHASE_COMPLETE,
    ATTEMPT_PHASE_CONTEXT,
    ATTEMPT_PHASE_EFFECT,
    ATTEMPT_PHASE_FAILED,
    ATTEMPT_PHASE_PROVIDER,
    ATTEMPT_PHASE_VALIDATION,
    DATABASE_DECLARED_OUTPUT_REARM_SCHEMA,
    DATABASE_IMPLEMENTATION_DAEMON_INTERFACE,
    DATABASE_PORTAL_COMPLETION_IMPLEMENTATION_COMMIT_MISSING_REASON,
    DATABASE_PORTAL_CROSS_BOARD_COMPLETION_REASONS,
    DATABASE_POST_MERGE_COMPLETION_RECOVERY_SEED_SCHEMA,
    DATABASE_POST_MERGE_COMPLETION_RECOVERY_SEED_SCHEMA_V2,
    DATABASE_POST_MERGE_COMPLETION_TARGET_GENERATION_CHANGED_REASON,
    DATABASE_POST_MERGE_RECOVERY_PREAUTHORIZATION_SCHEMA,
    DATABASE_POST_MERGE_RECOVERY_SCHEMA,
    DATABASE_POST_MERGE_REQUALIFICATION_RECOVERY_SCHEMA,
    DATABASE_PROTECTED_PRESERVATION_TARGET_ANCESTRY_MISSING_REASON,
    DATABASE_PROVIDER_CALLBACK_UNKNOWN_SCHEMA,
    DATABASE_TASK_ATTEMPT_INTERFACE,
    POST_MERGE_DECLARED_OUTPUT_REPAIR_SCHEMA,
    POST_MERGE_DECLARED_OUTPUT_REQUALIFICATION_SCHEMA,
    DatabaseImplementationAuthorityError,
    DatabaseImplementationConflictError,
    DatabaseImplementationCoordinationDriftError,
    DatabaseImplementationDaemon,
    DatabaseTaskAttempt,
    is_database_authority_mode,
    open_database_implementation_daemon,
    parse_args,
    parse_task_text,
    portal_task_identity,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
    build_database_implementation_daemon_from_args,
    build_portal_implementation_daemon_from_args,
    resolve_database_implementation_paths,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for database implementation daemon tests",
)


def _population(task_count: int = 4) -> dict[str, object]:
    tasks = []
    for index in range(1, task_count + 1):
        tasks.append(
            {
                "task_cid": f"task:cid:{index:03d}",
                "task_id": f"DQP-T{index:03d}",
                "goal_cid": "goal:cid:root",
                "status": "ready",
                "priority": "P0",
                "ordinal": index,
                "title": f"Task {index}",
            }
        )
    return {
        "repository_tree_id": "tree:dqp-018",
        "objectives": [
            {
                "objective_id": "objective:dqp-018",
                "objective_alias": "DQP-O018",
                "title": "Daemon cutover",
                "goal_cid": "goal:cid:root",
                "goal_alias": "DQP-G030",
                "status": "open",
            }
        ],
        "tasks": tasks,
    }


def _apmc_bootstrap_frontier_population() -> dict[str, object]:
    completed = tuple(f"APMC-{index:03d}" for index in range(6)) + ("APMC-018",)
    dependencies = {
        "APMC-001": ("APMC-000",),
        "APMC-002": ("APMC-001",),
        "APMC-003": ("APMC-001",),
        "APMC-004": ("APMC-001", "APMC-003"),
        "APMC-005": ("APMC-002", "APMC-004"),
        "APMC-018": ("APMC-000",),
        "APMC-006": ("APMC-001",),
        "APMC-012": ("APMC-002", "APMC-005"),
        "APMC-014": ("APMC-002", "APMC-004"),
    }

    def task(task_id: str, *, ordinal: int, status: str) -> dict[str, object]:
        return {
            "task_cid": f"task:cid:{task_id}",
            "task_id": task_id,
            "goal_cid": "goal:cid:apmc",
            "status": status,
            "priority": "P0",
            "ordinal": ordinal,
            "title": task_id,
            "dependencies": [
                f"task:cid:{dependency}"
                for dependency in dependencies.get(task_id, ())
            ],
        }

    frontier = ("APMC-006", "APMC-012", "APMC-014")
    return {
        "repository_tree_id": "tree:apmc-qualified-bootstrap",
        "objectives": [
            {
                "objective_id": "APMC-G000",
                "objective_alias": "APMC-G000",
                "title": "Autonomous meta-controller",
                "goal_cid": "goal:cid:apmc",
                "goal_alias": "APMC-G000",
                "status": "open",
            }
        ],
        "tasks": [
            *(
                task(task_id, ordinal=index, status="completed")
                for index, task_id in enumerate(completed, start=1)
            ),
            *(
                task(task_id, ordinal=index, status="ready")
                for index, task_id in enumerate(frontier, start=len(completed) + 1)
            ),
        ],
    }


def _open_daemon(
    tmp_path: Path,
    *,
    session: str = "",
    provider_calls: list[str] | None = None,
    effect_calls: list[str] | None = None,
    markdown_path: Path | None = None,
    provider_fn: Callable[[DatabaseTaskAttempt], dict[str, object]] | None = None,
    effect_fn: Callable[
        [DatabaseTaskAttempt, dict[str, object]], dict[str, object]
    ]
    | None = None,
    validation_fn: Callable[
        [DatabaseTaskAttempt, dict[str, object]], dict[str, object]
    ]
    | None = None,
    lease_ms: int = 60_000,
    max_task_attempts: int = 0,
    clock_ms: Callable[[], int] | None = None,
    task_shard_count: int = 1,
    task_shard_index: int = 0,
    strict_task_sharding: bool = False,
    control_path: Path | None = None,
    coordination_path: Path | None = None,
    execution_path: Path | None = None,
    repo_root: Path | None = None,
    merge_target_ref: str = "HEAD",
    task_prefix: str = "",
    process_instance_id: str | None = None,
    validation_retry_successor_recovery_fn: Callable[
        [DatabaseTaskAttempt, object, Mapping[str, object]],
        Mapping[str, object],
    ]
    | None = None,
    lane: str = "",
) -> DatabaseImplementationDaemon:
    database_path = control_path or (tmp_path / "control.duckdb")
    suffix = f"-{lane}" if lane else ""
    resolved_coordination_path = coordination_path or (
        tmp_path / f"coordination{suffix}.duckdb"
    )
    resolved_execution_path = execution_path or (
        tmp_path / f"execution{suffix}.duckdb"
    )

    def default_provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        if provider_calls is not None:
            provider_calls.append(attempt.task_cid)
        return {
            "status": "ok",
            "accepted": True,
            "task_cid": attempt.task_cid,
        }

    def effect(
        attempt: DatabaseTaskAttempt, provider_result: dict[str, object]
    ) -> dict[str, object]:
        if effect_calls is not None:
            effect_calls.append(attempt.task_cid)
        return {
            "status": "applied",
            "task_cid": attempt.task_cid,
            "provider_result": dict(provider_result),
        }

    def validation(
        attempt: DatabaseTaskAttempt, effect_result: dict[str, object]
    ) -> dict[str, object]:
        return {
            "outcome": "passed",
            "evidence_digest": "sha256:" + "a" * 64,
            "argv": ["focused-database-validation", attempt.task_cid],
            "effect_result": dict(effect_result),
        }

    return DatabaseImplementationDaemon(
        database_path=database_path,
        coordination_path=resolved_coordination_path,
        execution_path=resolved_execution_path,
        owner_session_id=session,
        process_instance_id=process_instance_id,
        authority_mode="embedded",
        task_source_kind="duckdb",
        markdown_path=markdown_path,
        # Projections intentionally absent.
        state_path=None,
        strategy_path=None,
        events_path=None,
        pid_path=None,
        queue_path=None,
        lease_ms=lease_ms,
        max_task_attempts=max_task_attempts,
        task_shard_count=task_shard_count,
        task_shard_index=task_shard_index,
        strict_task_sharding=strict_task_sharding,
        provider_fn=provider_fn or default_provider,
        effect_fn=effect_fn or effect,
        validation_fn=validation_fn or validation,
        validation_retry_successor_recovery_fn=(
            validation_retry_successor_recovery_fn
        ),
        require_real_execution=True,
        clock_ms=clock_ms,
        repo_root=repo_root,
        merge_target_ref=merge_target_ref,
        task_prefix=task_prefix,
    )


def _alias_home(task_alias: str, shard_count: int) -> int:
    digest = hashlib.sha256(task_alias.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % shard_count


def _rewrite_as_legacy_typed_deferrals(
    daemon: DatabaseImplementationDaemon,
    attempts: list[DatabaseTaskAttempt],
    reasons: list[str],
) -> None:
    """Model exact typed rows emitted by the pre-main process-wait daemon."""

    assert len(attempts) == len(reasons)
    connection = daemon._require_connection()
    for attempt, reason in zip(attempts, reasons, strict=True):
        typed = daemon._typed_deferral_receipt(attempt, reason=reason)
        phase_body = {
            "reason": reason,
            "portal_retryable_failure": True,
            "deferred": True,
            "attempt_consumed": False,
            "provider_dispatched": False,
            "typed_deferral_slot_consumed": True,
            "typed_deferral": typed,
        }
        connection.execute(
            """
            UPDATE attempt_phases
               SET body_json = ?
             WHERE attempt_id = ? AND phase = 'failed'
            """,
            [
                json.dumps(phase_body, separators=(",", ":"), sort_keys=True),
                attempt.attempt_id,
            ],
        )


def _legacy_leftover_wait_budget(
    daemon: DatabaseImplementationDaemon,
    attempts: list[DatabaseTaskAttempt],
) -> dict[str, object]:
    ordered = sorted(
        attempts,
        key=lambda item: (
            int(item.attempt_number),
            int(item.started_at_ms),
            item.attempt_id,
        ),
        reverse=True,
    )
    matching: list[dict[str, object]] = []
    digest = hashlib.sha256()
    typed_by_attempt: dict[str, dict[str, object]] = {}
    for attempt in ordered:
        failed = [
            item
            for item in daemon.phase_history(attempt.attempt_id)
            if item["phase"] == "failed"
        ]
        typed = daemon._verified_typed_deferral_receipt(
            attempt,
            failed[-1]["body"],
        )
        assert typed is not None
        typed_by_attempt[attempt.attempt_id] = typed
        identity: dict[str, object] = {
            "attempt_id": attempt.attempt_id,
            "attempt_number": int(attempt.attempt_number),
            "reason": typed["reason"],
            "deferral_fingerprint": typed["deferral_fingerprint"],
        }
        matching.append(identity)
        encoded = json.dumps(
            identity,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    current = ordered[0]
    current_typed = typed_by_attempt[current.attempt_id]
    budget: dict[str, object] = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "database-portal-typed-deferral-budget@1"
        ),
        "task_cid": current.task_cid,
        "task_generation": current.task_cid,
        "generation_fingerprint": current_typed["generation_fingerprint"],
        "current_deferral_fingerprint": current_typed[
            "deferral_fingerprint"
        ],
        "typed_deferral_candidate_count": len(matching),
        "typed_deferral_count": len(matching),
        "typed_deferral_count_is_lower_bound": False,
        "verified_typed_deferral_count": len(matching),
        "verified_count_complete": True,
        "max_task_attempts": daemon.max_task_attempts,
        "exhausted": True,
        "attempt_consumed": False,
        "typed_deferral_slot_consumed": True,
        "matching_attempts": matching,
        "matching_attempts_digest": "sha256:" + digest.hexdigest(),
        "matching_attempts_truncated": False,
        "omitted_matching_attempt_count": 0,
    }
    budget["observation_id"] = daemon._database_portal_evidence_digest(
        budget
    )
    return budget


def _block_with_legacy_leftover_wait_budget(
    daemon: DatabaseImplementationDaemon,
    attempts: list[DatabaseTaskAttempt],
    *,
    budget_override: dict[str, object] | None = None,
) -> tuple[DatabaseTaskAttempt, dict[str, object]]:
    latest = max(attempts, key=lambda item: int(item.attempt_number))
    budget = budget_override or _legacy_leftover_wait_budget(daemon, attempts)
    task = daemon.task_source.get(latest.task_cid)
    assert task is not None and task.status == "retrying"
    daemon.task_source.compare_and_set_status(
        latest.task_cid,
        expected_revision=int(task.revision),
        status="blocked",
        receipt={
            "operation": "database_portal_typed_deferral_budget_exhausted",
            "attempt_id": latest.attempt_id,
            "attempt_number": int(latest.attempt_number),
            "claim_id": latest.claim_id,
            "lease_id": latest.lease_id,
            "owner_session_id": latest.owner_session_id,
            "fencing_token": int(latest.fencing_token),
            "fence_epoch": int(latest.fence_epoch),
            "execution_phase": "failed",
            "execution_revision": int(latest.revision),
            "execution_finished_at_ms": latest.finished_at_ms,
            "reason": "typed_portal_deferral_budget_exhausted",
            "retryable": False,
            "attempt_consumed": False,
            "typed_deferral_slot_consumed": True,
            "retry_budget": budget,
            "prior_queue_entry_preserved_inactive": (
                daemon.task_source.get_queue_entry(latest.task_cid) is not None
            ),
            "coordination": {
                "attempt_id": latest.attempt_id,
                "claim_id": latest.claim_id,
                "attempt_number": int(latest.attempt_number),
            },
            "control_expected_status": "retrying",
            "control_expected_revision": int(task.revision),
        },
    )
    return latest, budget


def _sha256_json_identity(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _legacy_portal_effect(
    attempt: DatabaseTaskAttempt,
    *,
    baseline_commit: str,
    baseline_tree: str,
    implementation_commit: str,
) -> dict[str, object]:
    """Return a structurally exact effect without granting semantic authority."""

    binding_id = "sha256:" + "1" * 64
    portal_receipt_id = "sha256:" + "2" * 64
    evidence_digest = "sha256:" + "3" * 64
    completion_event_id = "sha256:" + "4" * 64
    binding: dict[str, object] = {
        "schema": DATABASE_PORTAL_COMPLETION_BINDING_SCHEMA,
        "task_cid": attempt.task_cid,
        "attempt_id": attempt.attempt_id,
        "binding_id": binding_id,
        "portal_receipt_id": portal_receipt_id,
        "evidence_digest": evidence_digest,
        "baseline_commit": baseline_commit,
        "baseline_tree": baseline_tree,
        "implementation_commit": implementation_commit,
        "completion_event_id": completion_event_id,
    }
    binding["receipt_id"] = _sha256_json_identity(binding)
    return {
        "status": "applied",
        "effect": "portal-supervised-accepted-effect",
        "effect_key": f"portal:{attempt.task_cid}:{attempt.attempt_id}",
        "task_cid": attempt.task_cid,
        "attempt_id": attempt.attempt_id,
        "binding_id": binding_id,
        "portal_receipt_id": portal_receipt_id,
        "evidence_digest": evidence_digest,
        "baseline_commit": baseline_commit,
        "baseline_tree": baseline_tree,
        "implementation_commit": implementation_commit,
        "completion_event_id": completion_event_id,
        "portal_completion_binding": binding,
    }


def _single_task_population(task_alias: str) -> dict[str, object]:
    population = _population(1)
    tasks = population["tasks"]
    assert isinstance(tasks, list)
    tasks[0]["task_id"] = task_alias
    return population


TYPED_DEFERRAL_RECOVERY_TEST_PATH = (
    "ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py"
)


def _git_recovery_repo(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "repair-repo"
    repo.mkdir(parents=True)
    for argv in (
        ("init", "-q"),
        ("config", "user.name", "Typed Deferral Test"),
        ("config", "user.email", "typed-deferral@example.invalid"),
    ):
        subprocess.run(
            ["git", "-C", str(repo), *argv],
            check=True,
            capture_output=True,
            text=True,
        )
    (repo / "generation.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repo), "add", "generation.txt"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", "base"],
        check=True,
        capture_output=True,
        text=True,
    )
    return repo, _git_output(repo, "rev-parse", "HEAD"), _git_output(
        repo, "rev-parse", "HEAD^{tree}"
    )


def _git_output(repo: Path, *argv: str, input_text: str | None = None) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *argv],
        check=True,
        capture_output=True,
        text=True,
        input=input_text,
    ).stdout.strip()


def _git_commit(repo: Path, *, name: str, content: str) -> tuple[str, str]:
    target = repo / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    _git_output(repo, "add", name)
    _git_output(repo, "commit", "-m", f"add {name}")
    return _git_output(repo, "rev-parse", "HEAD"), _git_output(
        repo, "rev-parse", "HEAD^{tree}"
    )


def _successful_quota_high_pair() -> tuple[dict[str, object], dict[str, object]]:
    failure = build_grok_failure_receipt(
        probe_stderr_text="Grok Build usage balance exhausted",
        nonce="f" * 64,
        model="grok-4.6",
        probe_returncode=41,
        primary_dispatched=False,
    )
    legacy = resolve_agent_implementation_route(default_route="legacy")
    quota_high = resolve_agent_implementation_route(
        **{**legacy.as_dict(), "fallback_reasoning_effort": "high"}
    )
    outcome = build_grok_route_outcome(
        receipt=failure,
        route_plan=quota_high.as_outcome_dict(),
        quota_evidence_id="sha256:" + ("e" * 64),
        decision="fallback_succeeded",
        verifier_status="confirmed_quota",
        fallback_dispatched=True,
        fallback_returncode=0,
    )
    return failure, outcome


def _consumed_no_progress_evidence(
    daemon: DatabaseImplementationDaemon,
    attempt: DatabaseTaskAttempt,
    *,
    tag: str,
) -> dict[str, object]:
    task = daemon.task_source.get(attempt.task_cid)
    assert task is not None
    snapshot = daemon.task_source.snapshot()
    evidence: dict[str, object] = {
        "schema": DATABASE_PORTAL_CONSUMED_NO_PROGRESS_SCHEMA,
        "failure_kind": "consumed_no_progress",
        "diagnostic_failure_id": f"baguq-failure-{tag}",
        "diagnostic_receipt_id": f"baguq-diagnostic-{tag}",
        "diagnostic_receipt_digest": "sha256:" + "c" * 64,
        "diagnostic_receipt_size": 512,
        "context_receipt_id": f"baguq-context-{tag}",
        "context_receipt_digest": "sha256:" + "d" * 64,
        "context_receipt_size": 1024,
        "log_digest": "sha256:" + hashlib.sha256(tag.encode()).hexdigest(),
        "log_size": len(tag.encode()),
        "repository_id": "repository:database-daemon-test",
        "tree_id": "tree:portal-baseline",
        "control_repository_tree_id": snapshot.repository_tree_id,
        "task_cid": attempt.task_cid,
        "task_contract_digest": database_portal_task_contract_digest(
            task,
        ),
        "database_binding_id": "sha256:" + "b" * 64,
        "database_attempt_id": attempt.attempt_id,
        "database_claim_id": attempt.claim_id,
        "database_lease_id": attempt.lease_id,
        "database_fencing_token": int(attempt.fencing_token),
        "database_fence_epoch": int(attempt.fence_epoch),
        "portal_task_id": attempt.task_alias,
        "portal_attempt_number": 1,
        "returncode": 1,
        "attempt_consumed": True,
        "portal_provider_dispatched": True,
        "provider_effect_state": "unknown_may_have_started",
        "implementation_commit_present": False,
        "implementation_candidate_present": False,
        "validation_state": "not_run",
    }
    evidence["failure_fingerprint"] = (
        database_portal_consumed_no_progress_fingerprint(evidence)
    )
    return evidence


def _capacity_retry_receipt(
    daemon: DatabaseImplementationDaemon,
    attempt: DatabaseTaskAttempt,
    *,
    retry_not_before_ms: int = 2_000_000,
    portal_task_cid: str = "",
) -> dict[str, object]:
    digest = daemon._database_portal_evidence_digest
    primary: dict[str, object] = {
        "schema": "fixture/grok-failure@1",
        "nonce": "4" * 64,
    }
    primary["receipt_id"] = digest(primary)
    route_id = "route:capacity-test"
    invocation_id = "sha256:" + "2" * 64
    logical_id = "sha256:" + "1" * 64
    decision_id = "sha256:" + "3" * 64
    observed_at_ms = 1_000_000
    returncode = 17
    capacity: dict[str, object] = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "codex-terminal-capacity-receipt@1"
        ),
        "source": "grok_cli_runner",
        "failure_class": "usage_limit",
        "reason_code": "codex_usage_limit_reached",
        "primary_receipt_id": primary["receipt_id"],
        "nonce": primary["nonce"],
        "route_id": route_id,
        "invocation_binding_id": invocation_id,
        "logical_attempt_id": logical_id,
        "fallback_provider_id": "codex",
        "fallback_model_id": "gpt-5.6-terra",
        "fallback_reasoning_effort": "high",
        "fallback_returncode": returncode,
        "outcome_decision": "fallback_failed",
        "decision_id": decision_id,
        "provider_dispatched": True,
        "candidate_activity_observed": False,
        "attempt_consumed": True,
        "completion_authority": False,
        "observed_at_ms": observed_at_ms,
        "retry_not_before_ms": retry_not_before_ms,
        "evidence_kind": "codex_jsonl_terminal_error",
        "evidence_sha256": "sha256:" + "5" * 64,
        "evidence_bytes": 100,
        "evidence_overflow": False,
    }
    capacity["receipt_id"] = digest(capacity)
    outcome: dict[str, object] = {
        "route_plan": {
            "route_id": route_id,
            "fallback_provider_id": "codex",
            "fallback_model_id": "gpt-5.6-terra",
            "fallback_reasoning_effort": "high",
        },
        "preflight_receipt_id": primary["receipt_id"],
        "invocation_binding_id": invocation_id,
        "decision": "fallback_failed",
        "decision_id": decision_id,
        "fallback_dispatched": True,
        "fallback_returncode": returncode,
        "fallback_capacity_receipt": capacity,
    }
    outcome["outcome_id"] = digest(outcome)
    proof: dict[str, object] = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "post-dispatch-capacity-retry-proof@1"
        ),
        "task_id": attempt.task_alias,
        "attempt": 1,
        "task_revision_cid": portal_task_cid or attempt.task_cid,
        "logical_attempt_id": logical_id,
        "invocation_binding_id": invocation_id,
        "route_id": route_id,
        "decision_id": decision_id,
        "primary_receipt_id": primary["receipt_id"],
        "route_outcome_id": outcome["outcome_id"],
        "capacity_receipt_id": capacity["receipt_id"],
        "fallback_provider_id": "codex",
        "fallback_model_id": "gpt-5.6-terra",
        "fallback_reasoning_effort": "high",
        "fallback_returncode": returncode,
        "provider_dispatched": True,
        "attempt_consumed": True,
        "observed_at_ms": observed_at_ms,
        "retry_not_before_ms": retry_not_before_ms,
    }
    proof["proof_id"] = digest(proof)
    receipt: dict[str, object] = {
        "schema": DATABASE_PORTAL_CAPACITY_RETRY_SCHEMA,
        "disposition": "retry",
        "reason": "dual_provider_capacity_exhausted",
        "task_cid": attempt.task_cid,
        "task_alias": attempt.task_alias,
        "attempt_id": attempt.attempt_id,
        "claim_id": attempt.claim_id,
        "lease_id": attempt.lease_id,
        "attempt_number": attempt.attempt_number,
        "fencing_token": attempt.fencing_token,
        "fence_epoch": attempt.fence_epoch,
        "portal_attempt": 1,
        "ordinary_retry_generation": 1,
        "max_task_attempts": daemon.max_task_attempts,
        "remaining_task_attempts": daemon.max_task_attempts - 1,
        "attempt_consumed": True,
        "provider_dispatched": True,
        "backoff_seconds": 1_000,
        "retry_not_before_ms": retry_not_before_ms,
        "binding_id": "sha256:" + "6" * 64,
        "events_digest": "sha256:" + "7" * 64,
        "event_stream_id": "event-log:capacity-retry",
        "implementation_event_id": "sha256:" + "8" * 64,
        "post_dispatch_capacity_proof": proof,
        "primary_receipt": primary,
        "route_outcome": outcome,
        "codex_capacity_receipt": capacity,
    }
    receipt["receipt_id"] = digest(receipt)
    return receipt


def _rehash_capacity_retry_receipt(
    daemon: DatabaseImplementationDaemon,
    receipt: dict[str, object],
) -> dict[str, object]:
    value = json.loads(json.dumps(receipt))
    digest = daemon._database_portal_evidence_digest
    capacity = value["codex_capacity_receipt"]
    capacity.pop("receipt_id", None)
    capacity["receipt_id"] = digest(capacity)
    outcome = value["route_outcome"]
    outcome["fallback_capacity_receipt"] = capacity
    outcome.pop("outcome_id", None)
    outcome["outcome_id"] = digest(outcome)
    proof = value["post_dispatch_capacity_proof"]
    proof["capacity_receipt_id"] = capacity["receipt_id"]
    proof["route_outcome_id"] = outcome["outcome_id"]
    proof.pop("proof_id", None)
    proof["proof_id"] = digest(proof)
    value.pop("receipt_id", None)
    value["receipt_id"] = digest(value)
    return value


def _consumed_attempt_retry_receipt(
    daemon: DatabaseImplementationDaemon,
    attempt: DatabaseTaskAttempt,
    *,
    source_task_revision: int,
) -> dict[str, object]:
    receipt: dict[str, object] = {
        "schema": DATABASE_PORTAL_CONSUMED_ATTEMPT_RETRY_SCHEMA,
        "disposition": "retry",
        "reason": "unclassified_post_dispatch_failure",
        "failure_class": "unclassified_post_dispatch_failure",
        "provider_capacity_classification": "unproven",
        "capacity_retry_proven": False,
        "task_cid": attempt.task_cid,
        "task_alias": attempt.task_alias,
        "attempt_id": attempt.attempt_id,
        "claim_id": attempt.claim_id,
        "lease_id": attempt.lease_id,
        "attempt_number": attempt.attempt_number,
        "fencing_token": attempt.fencing_token,
        "fence_epoch": attempt.fence_epoch,
        "source_task_revision": source_task_revision,
        "portal_attempt": 1,
        "ordinary_retry_generation": 1,
        "retry_budget_basis": "portal_attempt",
        "legacy_database_attempts_excluded": True,
        "max_task_attempts": daemon.max_task_attempts,
        "remaining_task_attempts": daemon.max_task_attempts - 1,
        "attempt_consumed": True,
        "provider_dispatched": True,
        "backoff_seconds": 0,
        "retry_not_before_ms": 0,
        "binding_id": "sha256:" + "1" * 64,
        "events_digest": "sha256:" + "2" * 64,
        "event_stream_id": "event-log:consumed-attempt",
        "implementation_started_event_id": "sha256:" + "3" * 64,
        "implementation_finished_event_id": "sha256:" + "4" * 64,
        "baseline_commit": "5" * 40,
        "implementation_returncode": 1,
    }
    receipt["receipt_id"] = daemon._database_portal_evidence_digest(receipt)
    return receipt


def _protected_preservation_receipt(
    daemon: DatabaseImplementationDaemon,
    attempt: DatabaseTaskAttempt,
    *,
    source_task_revision: int,
) -> dict[str, object]:
    receipt: dict[str, object] = {
        "schema": DATABASE_PORTAL_PROTECTED_PATH_PRESERVATION_SCHEMA,
        "disposition": "protected_candidate_preserved",
        "reason": "implementation_protected_path_mutated",
        "task_cid": attempt.task_cid,
        "task_alias": attempt.task_alias,
        "attempt_id": attempt.attempt_id,
        "claim_id": attempt.claim_id,
        "lease_id": attempt.lease_id,
        "attempt_number": attempt.attempt_number,
        "fencing_token": attempt.fencing_token,
        "fence_epoch": attempt.fence_epoch,
        "source_task_revision": source_task_revision,
        "portal_attempt": 2,
        "attempt_consumed": False,
        "provider_dispatched": True,
        "completion_authoritative": False,
        "local_recovery_required": True,
        "mutation_scopes": ["shared_checkout"],
        "protected_paths": [
            (
                "ipfs_accelerate_py/agent_supervisor/todo_daemon/"
                "implementation_daemon.py"
            )
        ],
        "baseline_commit": "b" * 40,
        "implementation_commit": "c" * 40,
        "preserved_commit": "c" * 40,
        "rescue_branch": (
            "rescue/dqp-t001-attempt-2-protected-path-interrupted"
        ),
        "original_branch": "implementation/dqp-t001-attempt-2",
        "original_worktree_path": "/tmp/dqp-t001-attempt-2",
        "binding_id": "sha256:" + "1" * 64,
        "events_digest": "sha256:" + "2" * 64,
        "event_stream_id": "event-log:protected-preservation",
        "implementation_started_event_id": "sha256:" + "3" * 64,
        "protected_mutation_event_id": "sha256:" + "4" * 64,
        "preservation_event_id": "sha256:" + "5" * 64,
        "implementation_finished_event_id": "sha256:" + "6" * 64,
        "protected_path_violation_digest": "sha256:" + "7" * 64,
        "preservation_digest": "sha256:" + "8" * 64,
    }
    receipt["receipt_id"] = daemon._database_portal_evidence_digest(receipt)
    return receipt


def _protected_reconciliation_self_lock_receipt(
    daemon: DatabaseImplementationDaemon,
    target: DatabaseTaskAttempt,
    preservation_seed: dict[str, object],
) -> dict[str, object]:
    binding_id = "sha256:" + "9" * 64
    recovery_identity = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "database-portal-protected-preservation-recovery@1"
        ),
        "source_receipt_id": preservation_seed["receipt_id"],
        "source_attempt_id": preservation_seed["attempt_id"],
        "source_claim_id": preservation_seed["claim_id"],
        "task_cid": target.task_cid,
        "task_alias": target.task_alias,
        "baseline_commit": preservation_seed["baseline_commit"],
        "preserved_commit": preservation_seed["preserved_commit"],
        "rescue_branch": preservation_seed["rescue_branch"],
        "target_attempt_id": target.attempt_id,
        "target_claim_id": target.claim_id,
        "target_attempt_number": target.attempt_number,
        "target_fencing_token": target.fencing_token,
        "target_fence_epoch": target.fence_epoch,
        "target_lease_id": target.lease_id,
        "target_binding_id": binding_id,
    }
    recovery_key = daemon._database_portal_evidence_digest(recovery_identity)
    safe_alias = target.task_alias.lower()
    receipt: dict[str, object] = {
        "schema": DATABASE_PORTAL_PROTECTED_RECONCILIATION_SELF_LOCK_SCHEMA,
        "disposition": "retry_exact_preserved_candidate",
        "reason": "protected_preservation_reconciliation_self_lock",
        "task_cid": target.task_cid,
        "task_alias": target.task_alias,
        "target_attempt_id": target.attempt_id,
        "target_claim_id": target.claim_id,
        "target_lease_id": target.lease_id,
        "target_attempt_number": target.attempt_number,
        "target_fencing_token": target.fencing_token,
        "target_fence_epoch": target.fence_epoch,
        "source_preservation_receipt_id": preservation_seed["receipt_id"],
        "source_attempt_id": preservation_seed["attempt_id"],
        "baseline_commit": preservation_seed["baseline_commit"],
        "preserved_commit": preservation_seed["preserved_commit"],
        "rescue_branch": preservation_seed["rescue_branch"],
        "target_binding_id": binding_id,
        "events_digest": "sha256:" + "8" * 64,
        "event_stream_id": "event-log:protected-reconciliation-self-lock",
        "recovery_key": recovery_key,
        "recovery_branch": (
            f"implementation/{safe_alias}-protected-"
            f"{recovery_key.removeprefix('sha256:')[:20]}"
        ),
        "validation_started_event_id": "sha256:" + "1" * 64,
        "verification_lock_timeout_event_id": "sha256:" + "2" * 64,
        "validation_finished_event_id": "sha256:" + "3" * 64,
        "cleanup_finished_event_id": "sha256:" + "4" * 64,
        "lock_path": "/tmp/vrif-checkout-mutation.lock",
        "lock_owner_pid": 12345,
        "lock_waited_seconds": 30,
        "provider_dispatched": False,
        "attempt_consumed": False,
        "validation_commands_passed": True,
        "verification_deferred": True,
        "merge_attempted": False,
    }
    receipt["receipt_id"] = daemon._database_portal_evidence_digest(receipt)
    return receipt


def test_interface_identities() -> None:
    assert DATABASE_IMPLEMENTATION_DAEMON_INTERFACE == (
        "DatabaseImplementationDaemon@1"
    )
    assert DATABASE_TASK_ATTEMPT_INTERFACE == "DatabaseTaskAttempt@1"
    assert DatabaseImplementationDaemon.INTERFACE == (
        DATABASE_IMPLEMENTATION_DAEMON_INTERFACE
    )
    assert DatabaseTaskAttempt.INTERFACE == DATABASE_TASK_ATTEMPT_INTERFACE
    assert is_database_authority_mode(authority_mode="embedded")
    assert is_database_authority_mode(task_source_kind="duckdb")
    assert not is_database_authority_mode(
        authority_mode="legacy_markdown", task_source_kind="legacy-markdown"
    )


def test_database_completion_seed_vocabulary_matches_task_source() -> None:
    daemon_module = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon"
    )
    assert daemon_module._DATABASE_SUCCESSFUL_CONTROL_STATUSES is (
        TASK_SOURCE_COMPLETED_STATUSES
    )
    assert TASK_SOURCE_COMPLETED_STATUSES == frozenset(
        {"completed", "complete", "done", "skipped"}
    )


def test_strict_database_lane_claims_only_alias_hash_home_tasks(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:strict-lane-0",
        task_shard_count=2,
        task_shard_index=0,
        strict_task_sharding=True,
    )
    try:
        daemon.materialize_population(_population(8))
        claimed_aliases: list[str] = []
        while True:
            attempt = daemon.claim_next()
            if attempt is None:
                break
            claimed_aliases.append(attempt.task_alias)

        expected = {
            f"DQP-T{index:03d}"
            for index in range(1, 9)
            if _alias_home(f"DQP-T{index:03d}", 2) == 0
        }
        assert set(claimed_aliases) == expected
        assert claimed_aliases
        assert all(_alias_home(alias, 2) == 0 for alias in claimed_aliases)
    finally:
        daemon.close()


def test_non_strict_database_lane_preserves_cross_shard_claiming(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:non-strict-lane-0",
        task_shard_count=2,
        task_shard_index=0,
        strict_task_sharding=False,
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        assert attempt.task_alias == "DQP-T001"
        assert _alias_home(attempt.task_alias, 2) == 1
    finally:
        daemon.close()


def test_strict_restart_resumes_exact_in_home_claim(
    tmp_path: Path,
) -> None:
    first = _open_daemon(
        tmp_path,
        session="session:strict-in-home",
        task_shard_count=2,
        task_shard_index=0,
        strict_task_sharding=True,
    )
    try:
        first.materialize_population(_population(3))
        attempt = first.claim_next()
        assert attempt is not None
        assert attempt.task_alias == "DQP-T003"
        assert _alias_home(attempt.task_alias, 2) == 0
    finally:
        first.close()

    provider_calls: list[str] = []
    restarted = _open_daemon(
        tmp_path,
        session="session:strict-in-home",
        provider_calls=provider_calls,
        task_shard_count=2,
        task_shard_index=0,
        strict_task_sharding=True,
    )
    try:
        result = restarted.run_once()["implementation_result"]
        assert result["resumed"] is True
        assert result["status"] == "succeeded"
        assert provider_calls == [attempt.task_cid]
    finally:
        restarted.close()


def test_strict_restart_requeues_pre_provider_out_of_home_attempt(
    tmp_path: Path,
) -> None:
    control_path = tmp_path / "control.duckdb"
    lane_path = tmp_path / "lane-0"
    first = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:lane-0",
        task_shard_count=2,
        task_shard_index=0,
        strict_task_sharding=False,
    )
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        assert attempt.task_alias == "DQP-T001"
        assert _alias_home(attempt.task_alias, 2) == 1
    finally:
        first.close()

    restart_provider_calls: list[str] = []
    restarted = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:lane-0",
        provider_calls=restart_provider_calls,
        task_shard_count=2,
        task_shard_index=0,
        strict_task_sharding=True,
    )
    try:
        result = restarted.run_once()
        implementation = result["implementation_result"]
        assert implementation["reason"] == "strict_resume_not_admitted"
        assert implementation["task_requeued"] is True
        assert implementation["task_quarantined"] is False
        assert restart_provider_calls == []
        failed = restarted.get_attempt(attempt.attempt_id)
        assert failed is not None
        assert failed.status == "failed"
        task = restarted.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "ready"
        assert task.body["completion_receipt"]["operation"] == (
            "database_strict_resume_requeue"
        )
        claim = restarted.coordinator.get_task_claim(attempt.claim_id)
        assert claim is not None
        assert str(getattr(claim.state, "value", claim.state)) == "released"
    finally:
        restarted.close()

    home = _open_daemon(
        tmp_path / "lane-1",
        control_path=control_path,
        session="session:lane-1",
        task_shard_count=2,
        task_shard_index=1,
        strict_task_sharding=True,
    )
    try:
        admitted = home.claim_next()
        assert admitted is not None
        assert admitted.task_cid == attempt.task_cid
        assert admitted.task_alias == "DQP-T001"
    finally:
        home.close()


def test_strict_restart_quarantines_pre_provider_attempt_at_configured_cap(
    tmp_path: Path,
) -> None:
    control_path = tmp_path / "control.duckdb"
    lane_path = tmp_path / "lane-0"
    first = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:strict-attempt-cap",
        max_task_attempts=4,
        task_shard_count=2,
        task_shard_index=0,
        strict_task_sharding=False,
    )
    try:
        first.materialize_population(_population(1))
        first.sync_ready_tasks_into_coordination()
        for expected_attempt_number in range(1, 4):
            prior = first.coordinator.claim_ready_task(
                owner_session_id="session:strict-attempt-cap",
            )
            assert prior is not None
            assert prior.attempt_number == expected_attempt_number
            first.coordinator.release(
                prior.as_fenced_lease(),
                reason="seed configured strict attempt cap",
            )
        attempt = first.claim_next()
        assert attempt is not None
        assert attempt.attempt_number == 4
        assert _alias_home(attempt.task_alias, 2) == 1
    finally:
        first.close()

    provider_calls: list[str] = []
    restarted = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:strict-attempt-cap",
        provider_calls=provider_calls,
        max_task_attempts=4,
        task_shard_count=2,
        task_shard_index=0,
        strict_task_sharding=True,
    )
    try:
        implementation = restarted.run_once()["implementation_result"]
        assert implementation["reason"] == "strict_resume_not_admitted"
        assert implementation["task_requeued"] is False
        assert implementation["task_quarantined"] is True
        assert provider_calls == []
        task = restarted.task_source.get(attempt.task_cid)
        assert task is not None and task.status == "quarantined"
        receipt = task.body["completion_receipt"]
        assert receipt["operation"] == "database_strict_resume_quarantine"
        assert receipt["max_task_attempts"] == 4
        assert receipt["attempt_budget_exhausted"] is True
        assert restarted.claim_next() is None
        attempts = restarted.coordinator.coordination_registry_projection()[
            "task_attempts"
        ]
        assert len(attempts) == 4
        assert max(int(row["attempt_number"]) for row in attempts) == 4
    finally:
        restarted.close()


def test_strict_resume_accepts_exact_legacy_fenced_retry(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}
    provider_calls: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        session="session:strict-rotated-retry",
        provider_calls=provider_calls,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
        strict_task_sharding=True,
    )
    try:
        daemon.materialize_population(_population(1))
        first = daemon.claim_next()
        assert first is not None and first.attempt_number == 1
        assert provider_calls == []

        now["ms"] = 7_000
        second = daemon.claim_next()
        assert second is not None and second.attempt_number == 2
        assert provider_calls == []

        task = daemon.task_source.get(second.task_cid)
        assert task is not None and task.status == "in_progress"
        receipt = task.body["completion_receipt"]
        assert receipt["claim_id"] == first.claim_id
        assert receipt.get("claim_phase_schema", "") == ""
        assert receipt["claimed_from_revision"] + 1 == task.revision
        projection = daemon.coordinator.coordination_registry_projection()
        local_task = next(
            row for row in projection["tasks"] if row["task_cid"] == task.task_cid
        )
        assert local_task["body"]["authoritative_revision"] == receipt[
            "claimed_from_revision"
        ] + 1
        assert local_task["body"]["authoritative_status"] == "in_progress"
        assert daemon._shared_retry_binding_matches_attempt(
            task,
            second,
            local_task_body=local_task["body"],
            local_projection=projection,
        ) is True
        assert daemon._strict_resume_admission_result(second) is None

        result = daemon.resume_attempt(second)
        assert result["resumed"] is True
        assert result["status"] == "succeeded"
        assert provider_calls == [second.task_cid]
    finally:
        daemon.close()


def test_strict_restart_quarantines_effect_committed_out_of_home_attempt(
    tmp_path: Path,
) -> None:
    control_path = tmp_path / "control.duckdb"
    lane_path = tmp_path / "lane-0"
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    first = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:lane-0-effect",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        task_shard_count=2,
        task_shard_index=0,
        strict_task_sharding=False,
    )
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        current = first.commit_phase(
            attempt,
            ATTEMPT_PHASE_CONTEXT,
            body={"test": "strict-restart"},
        )
        current, provider_result, _duplicated = first.run_provider(current)
        current, _effect_result, _duplicated = first.run_effect(
            current,
            provider_result,
        )
        assert current.committed_phase == "effect"
        assert provider_calls == [attempt.task_cid]
        assert effect_calls == [attempt.task_cid]
    finally:
        first.close()

    restart_provider_calls: list[str] = []
    restart_effect_calls: list[str] = []
    restarted = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:lane-0-effect",
        provider_calls=restart_provider_calls,
        effect_calls=restart_effect_calls,
        task_shard_count=2,
        task_shard_index=0,
        strict_task_sharding=True,
    )
    try:
        result = restarted.run_once()
        implementation = result["implementation_result"]
        assert implementation["reason"] == "strict_resume_not_admitted"
        assert implementation["task_requeued"] is False
        assert implementation["task_quarantined"] is True
        assert restart_provider_calls == []
        assert restart_effect_calls == []
        task = restarted.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "quarantined"
        assert task.body["completion_receipt"]["operation"] == (
            "database_strict_resume_quarantine"
        )
        assert restarted.claim_next() is None
    finally:
        restarted.close()


@pytest.mark.parametrize(
    "provider_idempotency_key",
    ["", "provider:custom-crash-key"],
    ids=["canonical-key", "custom-key"],
)
def test_strict_restart_quarantines_provider_receipt_before_phase_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider_idempotency_key: str,
) -> None:
    control_path = tmp_path / "control.duckdb"
    lane_path = tmp_path / "lane-0"
    provider_calls: list[str] = []
    first = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:lane-0-provider-crash",
        provider_calls=provider_calls,
        task_shard_count=2,
        task_shard_index=0,
        strict_task_sharding=False,
    )
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        current = first.commit_phase(
            attempt,
            ATTEMPT_PHASE_CONTEXT,
            body={"test": "provider-receipt-crash"},
        )
        original_commit_phase = first.commit_phase

        def crash_before_provider_phase(
            current_attempt: DatabaseTaskAttempt,
            phase: str,
            *,
            body: dict[str, object] | None = None,
        ) -> DatabaseTaskAttempt:
            if phase == ATTEMPT_PHASE_PROVIDER:
                raise RuntimeError("injected crash after provider receipt")
            return original_commit_phase(current_attempt, phase, body=body)

        monkeypatch.setattr(first, "commit_phase", crash_before_provider_phase)
        with pytest.raises(
            RuntimeError,
            match="injected crash after provider receipt",
        ):
            first.run_provider(
                current,
                idempotency_key=provider_idempotency_key,
            )
        persisted = first.get_attempt(attempt.attempt_id)
        assert persisted is not None
        assert persisted.committed_phase == ATTEMPT_PHASE_CONTEXT
        recorded_key = (
            provider_idempotency_key
            or f"provider:{attempt.attempt_id}"
        )
        assert first.provider_invocation_recorded(
            attempt.attempt_id,
            idempotency_key=recorded_key,
        ) is not None
        assert first.provider_invocation_exists(attempt.attempt_id) is True
        assert provider_calls == [attempt.task_cid]
    finally:
        first.close()

    restart_provider_calls: list[str] = []
    restarted = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:lane-0-provider-crash",
        provider_calls=restart_provider_calls,
        task_shard_count=2,
        task_shard_index=0,
        strict_task_sharding=True,
    )
    try:
        result = restarted.run_once()["implementation_result"]
        assert result["reason"] == "strict_resume_not_admitted"
        assert result["task_requeued"] is False
        assert result["task_quarantined"] is True
        assert restart_provider_calls == []
        task = restarted.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "quarantined"
        receipt = task.body["completion_receipt"]
        assert receipt["operation"] == "database_strict_resume_quarantine"
        assert receipt["provider_phase_committed"] is False
        assert receipt["provider_invocation_receipt_present"] is True
    finally:
        restarted.close()


@pytest.mark.parametrize("raced_alias", ["", "DQP-T001", "DQP-T004"])
def test_strict_database_lane_rechecks_authoritative_alias_after_local_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raced_alias: str,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:strict-race",
        task_shard_count=2,
        task_shard_index=0,
        strict_task_sharding=True,
    )
    try:
        daemon.materialize_population(_population(3))
        original_projection = daemon._stable_authoritative_task_projection
        reads = 0

        def raced_projection() -> tuple[tuple[object, ...], frozenset[str]]:
            nonlocal reads
            tasks, ready_cids = original_projection()
            reads += 1
            if reads < 3:
                return tasks, ready_cids
            return (
                tuple(
                    replace(task, task_alias=raced_alias)
                    if task.task_alias == "DQP-T003"
                    else task
                    for task in tasks
                ),
                ready_cids,
            )

        monkeypatch.setattr(
            daemon,
            "_stable_authoritative_task_projection",
            raced_projection,
        )

        assert daemon.claim_next() is None
        assert reads >= 3
        task = daemon.task_source.get_task("task:cid:003")
        assert task is not None
        assert task.status == "ready"
        projection = daemon.coordinator.coordination_registry_projection()
        claim = next(
            row
            for row in projection["task_claims"]
            if row["task_cid"] == "task:cid:003"
        )
        assert claim["state"] == "released"
    finally:
        daemon.close()


def test_four_daemon_processes_claim_distinct_work(tmp_path: Path) -> None:
    markdown = tmp_path / "board.md"
    markdown.write_text(
        "# Board\n\n## DQP-T001 Sample\n\n- Status: todo\n",
        encoding="utf-8",
    )
    original_markdown = markdown.read_text(encoding="utf-8")

    seed = _open_daemon(tmp_path, session="session:seed", markdown_path=markdown)
    try:
        seed.materialize_population(_population(4))
    finally:
        seed.close()

    claimed: list[str] = []
    for index in range(1, 5):
        daemon = _open_daemon(
            tmp_path,
            session=f"session:{index}",
            markdown_path=markdown,
        )
        try:
            attempt = daemon.claim_next()
            assert attempt is not None, f"session {index} failed to claim"
            claimed.append(attempt.task_cid)
            assert attempt.owner_session_id == f"session:{index}"
            assert attempt.committed_phase == "claimed"
        finally:
            daemon.close()

    assert len(claimed) == 4
    assert len(set(claimed)) == 4

    idle = _open_daemon(tmp_path, session="session:extra", markdown_path=markdown)
    try:
        assert idle.claim_next() is None
        assert idle.markdown_status_write_count == 0
        assert markdown.read_text(encoding="utf-8") == original_markdown
    finally:
        idle.close()


def _sparse_post_merge_history_fixture(
    task: Any,
    *,
    current_operation: str,
) -> tuple[Any, dict[str, Any]]:
    semantic_body = {
        key: value
        for key, value in dict(task.body).items()
        if key != "completion_receipt"
    }
    revision_numbers = (1, 5, 6, 7, 8)
    statuses = ("ready", "in_progress", "retrying", "in_progress", "blocked")
    operations = (
        "database_materialize_task",
        "database_claim",
        "database_portal_retry",
        "database_claim",
        current_operation,
    )
    entries = [
        {
            "revision": revision,
            "status": statuses[index],
            "body": {
                **semantic_body,
                "completion_receipt": {
                    "operation": operations[index],
                },
            },
        }
        for index, revision in enumerate(revision_numbers)
    ]
    projection_body = {
        "schema": TASK_REVISION_HISTORY_PROJECTION_SCHEMA,
        "task_cid": task.task_cid,
        "revisions": entries,
    }
    projection = {
        **projection_body,
        "projection_cid": content_identity(projection_body),
    }
    current = replace(
        task,
        revision=revision_numbers[-1],
        status=statuses[-1],
        body=entries[-1]["body"],
    )
    return current, projection


def test_post_merge_completion_crash_fence_allows_sparse_ordinary_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:sparse-ordinary-history",
    )
    try:
        daemon.materialize_population(_population(1))
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        sparse_task, projection = _sparse_post_merge_history_fixture(
            task,
            current_operation="database_portal_terminal_failure",
        )
        assert [
            entry["revision"] for entry in projection["revisions"]
        ] == [1, 5, 6, 7, 8]
        history_reads = 0

        def sparse_history(_task_cid: str) -> dict[str, Any]:
            nonlocal history_reads
            history_reads += 1
            return projection

        monkeypatch.setattr(
            daemon.task_source,
            "task_revision_history_projection",
            sparse_history,
        )

        assert (
            daemon._post_merge_completion_crash_recovery_context(
                sparse_task,
                require_current_blocked=True,
            )
            is None
        )
        assert history_reads == 0
    finally:
        daemon.close()


def test_post_merge_completion_crash_fence_rejects_sparse_dedicated_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:sparse-dedicated-history",
    )
    try:
        daemon.materialize_population(_population(1))
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        sparse_task, projection = _sparse_post_merge_history_fixture(
            task,
            current_operation=(
                "database_portal_typed_deferral_budget_exhausted"
            ),
        )
        monkeypatch.setattr(
            daemon.task_source,
            "task_revision_history_projection",
            lambda _task_cid: projection,
        )

        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="malformed or stale canonical history",
        ):
            daemon._post_merge_completion_crash_recovery_context(
                sparse_task,
                require_current_blocked=True,
            )
    finally:
        daemon.close()


def test_post_merge_completion_crash_fence_excludes_proven_stale_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:completion-recovery-stale-snapshot",
    )
    try:
        daemon.materialize_population(_population(1))
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        semantic_body = {
            key: value
            for key, value in dict(task.body).items()
            if key != "completion_receipt"
        }
        stale_body = {
            **semantic_body,
            "completion_receipt": {
                "operation": "database_portal_retry",
            },
        }
        current_body = {
            **semantic_body,
            "completion_receipt": {
                "operation": "database_claim",
            },
        }
        projection_body = {
            "schema": TASK_REVISION_HISTORY_PROJECTION_SCHEMA,
            "task_cid": task.task_cid,
            "revisions": [
                {
                    "revision": revision,
                    "status": (
                        "retrying"
                        if revision == 6
                        else "in_progress"
                        if revision == 7
                        else "ready"
                    ),
                    "body": (
                        stale_body
                        if revision == 6
                        else current_body
                        if revision == 7
                        else semantic_body
                    ),
                }
                for revision in range(1, 8)
            ],
        }
        projection = {
            **projection_body,
            "projection_cid": content_identity(projection_body),
        }
        stale_task = replace(
            task,
            status="retrying",
            revision=6,
            body=stale_body,
        )
        monkeypatch.setattr(
            daemon.task_source,
            "task_revision_history_projection",
            lambda _task_cid: projection,
        )
        monkeypatch.setattr(
            daemon.task_source,
            "ready_tasks",
            lambda *, limit: SimpleNamespace(tasks=(stale_task,)),
        )

        context = daemon._post_merge_completion_crash_recovery_context(
            stale_task,
            require_current_blocked=False,
        )

        assert context is not None
        assert context["stale_task_snapshot"] is True
        assert context["task_revision"] == 6
        assert context["canonical_task_revision"] == 7
        assert daemon._automatic_claim_exclusions() == {task.task_cid}

        blocked_candidate = replace(
            stale_task,
            status="blocked",
            body={
                **semantic_body,
                "completion_receipt": {
                    "operation": (
                        "database_portal_typed_deferral_budget_exhausted"
                    ),
                },
            },
        )
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="malformed or stale canonical history",
        ):
            daemon._post_merge_completion_crash_recovery_context(
                blocked_candidate,
                require_current_blocked=True,
            )
        forged_task = replace(
            stale_task,
            body={**stale_body, "fixture_tamper": True},
        )
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="malformed or stale canonical history",
        ):
            daemon._post_merge_completion_crash_recovery_context(
                forged_task,
                require_current_blocked=False,
            )
    finally:
        daemon.close()


def test_post_merge_completion_recovery_claim_fences_preclaim_and_toctou(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:completion-recovery-claim-fence",
    )
    try:
        daemon.materialize_population(_population(1))
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        plausible_history_candidate = replace(task, revision=6)
        original_history_projection = (
            daemon.task_source.task_revision_history_projection
        )
        monkeypatch.setattr(
            daemon.task_source,
            "task_revision_history_projection",
            None,
        )
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="no canonical history projection reader",
        ):
            daemon._post_merge_completion_crash_recovery_context(
                plausible_history_candidate,
                require_current_blocked=False,
            )
        monkeypatch.setattr(
            daemon.task_source,
            "task_revision_history_projection",
            lambda _task_cid: {"schema": "malformed"},
        )
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="malformed or stale canonical history",
        ):
            daemon._post_merge_completion_crash_recovery_context(
                plausible_history_candidate,
                require_current_blocked=False,
            )
        semantic_body = {
            key: value
            for key, value in dict(task.body).items()
            if key != "completion_receipt"
        }
        malformed_window_receipts = [
            {"operation": "database_portal_terminal_failure"},
            {
                "operation": (
                    "database_post_merge_declared_outputs_repair_recovery"
                ),
                "post_merge_completion_recovery_seed": {
                    "schema": "malformed"
                },
            },
            {"operation": "database_claim"},
            {"operation": "database_portal_retry"},
            {"operation": "database_claim"},
            {
                "operation": (
                    "database_portal_typed_deferral_budget_exhausted"
                )
            },
        ]
        malformed_window_body = {
            "schema": TASK_REVISION_HISTORY_PROJECTION_SCHEMA,
            "task_cid": task.task_cid,
            "revisions": [
                {
                    "revision": index + 1,
                    "status": status,
                    "body": {
                        **semantic_body,
                        "completion_receipt": malformed_window_receipts[
                            index
                        ],
                    },
                }
                for index, status in enumerate(
                    (
                        "blocked",
                        "retrying",
                        "in_progress",
                        "retrying",
                        "in_progress",
                        "blocked",
                    )
                )
            ],
        }
        malformed_window = {
            **malformed_window_body,
            "projection_cid": content_identity(malformed_window_body),
        }
        malformed_candidate = replace(
            task,
            status="blocked",
            revision=6,
            body=malformed_window_body["revisions"][-1]["body"],
        )
        monkeypatch.setattr(
            daemon.task_source,
            "task_revision_history_projection",
            lambda _task_cid: malformed_window,
        )
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="malformed dedicated recovery seed",
        ):
            daemon._post_merge_completion_crash_recovery_context(
                malformed_candidate,
                require_current_blocked=False,
            )
        present_none = json.loads(json.dumps(malformed_window))
        present_none["revisions"][1]["body"]["completion_receipt"][
            "post_merge_completion_recovery_seed"
        ] = None
        present_none_body = dict(present_none)
        present_none_body.pop("projection_cid")
        present_none["projection_cid"] = content_identity(
            present_none_body
        )
        monkeypatch.setattr(
            daemon.task_source,
            "task_revision_history_projection",
            lambda _task_cid: present_none,
        )
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="malformed dedicated recovery seed",
        ):
            daemon._post_merge_completion_crash_recovery_context(
                malformed_candidate,
                require_current_blocked=False,
            )
        seeded_claim_marker = json.loads(json.dumps(malformed_window))
        seeded_claim_revisions = seeded_claim_marker["revisions"]
        del seeded_claim_revisions[1]["body"]["completion_receipt"][
            "post_merge_completion_recovery_seed"
        ]
        seeded_claim_revisions[2]["body"]["completion_receipt"][
            "post_merge_completion_recovery_seed"
        ] = None
        seeded_claim_body = dict(seeded_claim_marker)
        seeded_claim_body.pop("projection_cid")
        seeded_claim_marker["projection_cid"] = content_identity(
            seeded_claim_body
        )
        monkeypatch.setattr(
            daemon.task_source,
            "task_revision_history_projection",
            lambda _task_cid: seeded_claim_marker,
        )
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="malformed dedicated recovery seed",
        ):
            daemon._post_merge_completion_crash_recovery_context(
                malformed_candidate,
                require_current_blocked=False,
            )
        one_sided_marker = json.loads(json.dumps(malformed_window))
        one_sided_revisions = one_sided_marker["revisions"]
        del one_sided_revisions[1]["body"]["completion_receipt"][
            "post_merge_completion_recovery_seed"
        ]
        one_sided_revisions[2]["body"]["completion_receipt"][
            "post_merge_completion_recovery_source_attempt_id"
        ] = "attempt:declared-seeded-claim"
        one_sided_body = dict(one_sided_marker)
        one_sided_body.pop("projection_cid")
        one_sided_marker["projection_cid"] = content_identity(
            one_sided_body
        )
        monkeypatch.setattr(
            daemon.task_source,
            "task_revision_history_projection",
            lambda _task_cid: one_sided_marker,
        )
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="malformed dedicated recovery seed",
        ):
            daemon._post_merge_completion_crash_recovery_context(
                malformed_candidate,
                require_current_blocked=False,
            )
        monkeypatch.setattr(
            daemon.task_source,
            "task_revision_history_projection",
            original_history_projection,
        )
        crash_context = {"context_id": "sha256:" + "1" * 64}
        monkeypatch.setattr(
            daemon,
            "_post_merge_completion_crash_recovery_context",
            lambda _task, *, require_current_blocked: crash_context,
        )

        assert daemon._automatic_claim_exclusions() == {task.task_cid}
        assert daemon.claim_next() is None

        observations = 0

        def crash_after_local_claim(
            _task: object,
            *,
            require_current_blocked: bool,
        ) -> dict[str, object] | None:
            nonlocal observations
            assert require_current_blocked is False
            observations += 1
            return None if observations == 1 else crash_context

        released: list[tuple[str, str, str]] = []
        original_release = daemon._release_unadmitted_claim

        def record_release(claim: object, *, reason: str) -> None:
            released.append(
                (
                    str(claim.claim_id),
                    str(claim.lease_id),
                    reason,
                )
            )
            original_release(claim, reason=reason)

        monkeypatch.setattr(
            daemon,
            "_post_merge_completion_crash_recovery_context",
            crash_after_local_claim,
        )
        monkeypatch.setattr(
            daemon,
            "_release_unadmitted_claim",
            record_release,
        )

        assert daemon.claim_next() is None
        assert observations == 2
        assert len(released) == 1
        claim_id, lease_id, reason = released[0]
        assert reason == "shared_board_post_merge_completion_recovery_pending"
        released_claim = daemon.coordinator.get_task_claim(claim_id)
        released_lease = daemon.coordinator.get_lease(lease_id)
        assert released_claim is not None
        assert released_claim.to_dict()["state"] == "released"
        assert released_lease is not None
        assert released_lease.to_dict()["state"] == "released"
        unchanged = daemon.task_source.get(task.task_cid)
        assert unchanged is not None
        assert unchanged.status == "ready"
        assert unchanged.revision == task.revision

        authority_observations = 0

        def history_unavailable_after_local_claim(
            _task: object,
            *,
            require_current_blocked: bool,
        ) -> None:
            nonlocal authority_observations
            assert require_current_blocked is False
            authority_observations += 1
            if authority_observations == 1:
                return None
            raise DatabaseImplementationAuthorityError(
                "fixture canonical history became unavailable"
            )

        monkeypatch.setattr(
            daemon,
            "_post_merge_completion_crash_recovery_context",
            history_unavailable_after_local_claim,
        )
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="canonical history became unavailable",
        ):
            daemon.claim_next()
        assert authority_observations == 2
        assert len(released) == 2
        claim_id, lease_id, reason = released[-1]
        assert reason == (
            "shared_board_post_merge_completion_history_unavailable"
        )
        released_claim = daemon.coordinator.get_task_claim(claim_id)
        released_lease = daemon.coordinator.get_lease(lease_id)
        assert released_claim is not None
        assert released_claim.to_dict()["state"] == "released"
        assert released_lease is not None
        assert released_lease.to_dict()["state"] == "released"
    finally:
        daemon.close()


def test_apmc_bootstrap_completions_unlock_exact_frontier_across_lane_sidecars(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "apmc-control.duckdb"
    seed = DatabaseImplementationDaemon(
        database_path=database_path,
        coordination_path=tmp_path / "seed-coordination.duckdb",
        execution_path=tmp_path / "seed-execution.duckdb",
        owner_session_id="apmc-seed",
        authority_mode="embedded",
        task_source_kind="duckdb",
        require_real_execution=True,
    )
    try:
        seed.materialize_population(_apmc_bootstrap_frontier_population())
    finally:
        seed.close()

    expected_ready = {
        "task:cid:APMC-006",
        "task:cid:APMC-012",
        "task:cid:APMC-014",
    }
    claimed: set[str] = set()
    for lane in range(3):
        coordination_path = tmp_path / f"lane-{lane}-coordination.duckdb"
        execution_path = tmp_path / f"lane-{lane}-execution.duckdb"
        daemon = DatabaseImplementationDaemon(
            database_path=database_path,
            coordination_path=coordination_path,
            execution_path=execution_path,
            owner_session_id=f"apmc-lane-{lane}",
            authority_mode="embedded",
            task_source_kind="duckdb",
            require_real_execution=True,
        )
        try:
            ready = set(daemon.sync_ready_tasks_into_coordination())
            assert ready == expected_ready - claimed
            for task_cid in ready:
                assert daemon.coordinator.claimability(task_cid)["claimable"] is True
            if lane == 0:
                first_projection = daemon.coordinator.coordination_registry_projection()
                assert set(daemon.sync_ready_tasks_into_coordination()) == ready
                assert (
                    daemon.coordinator.coordination_registry_projection()
                    == first_projection
                )
        finally:
            daemon.close()

        # Reopening the exact lane sidecars is an idempotent projection replay.
        daemon = DatabaseImplementationDaemon(
            database_path=database_path,
            coordination_path=coordination_path,
            execution_path=execution_path,
            owner_session_id=f"apmc-lane-{lane}",
            authority_mode="embedded",
            task_source_kind="duckdb",
            require_real_execution=True,
        )
        try:
            assert set(daemon.sync_ready_tasks_into_coordination()) == ready
            attempt = daemon.claim_next()
            assert attempt is not None
            assert attempt.task_cid in ready
            claimed.add(attempt.task_cid)
        finally:
            daemon.close()

    assert claimed == expected_ready


def test_removed_authoritative_task_is_excluded_without_idle_growth(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(tmp_path, session="session:removed-task")
    authoritative_cid = "task:cid:001"
    removed_cid = "task:cid:removed"
    try:
        daemon.materialize_population(_population(1))
        daemon.coordinator.register_task(
            task_cid=removed_cid,
            task_id="REMOVED",
            body={"status": "ready"},
        )
        assert daemon.coordinator.claimability(removed_cid)["claimable"] is True
        assert daemon.sync_ready_tasks_into_coordination() == [authoritative_cid]
        before = daemon.coordinator.coordination_registry_projection()

        for _pass in range(2):
            assert daemon.claim_next(exclude_task_cids=(authoritative_cid,)) is None
            assert daemon.coordinator.coordination_registry_projection() == before
    finally:
        daemon.close()


def test_fresh_strict_lane_seeds_completed_dependency_before_claim(
    tmp_path: Path,
) -> None:
    """A lane-private coordinator reconstructs dependency evidence from control."""

    population = _population(4)
    tasks = population["tasks"]
    assert isinstance(tasks, list)
    tasks[0].update(
        {
            "task_id": "DQP-COMPLETE",
            "status": "skipped",
        }
    )
    tasks[1].update(
        {
            "task_id": "DQP-DEPENDENT",
            "dependencies": ["task:cid:001"],
        }
    )
    tasks[2]["task_id"] = "DQP-OTHER-1"
    tasks[3].update(
        {
            "task_id": "DQP-BLOCKED",
            "status": "blocked",
        }
    )

    source = DatabaseTaskSource(tmp_path / "control.duckdb")
    source.materialize(population, repository_tree_id="tree:dqp-fresh-lane")
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "lane-0" / "coordination.duckdb",
        execution_path=tmp_path / "lane-0" / "execution.duckdb",
        owner_session_id="session:fresh-strict-lane",
        authority_mode="embedded",
        task_source_kind="duckdb",
        task_source=source,
        task_shard_count=2,
        task_shard_index=0,
        strict_task_sharding=True,
        require_real_execution=True,
    )
    try:
        # The prerequisite and off-shard task hash to lane 1; the dependent
        # and blocked task hash to lane 0.  Only the ready dependent is exposed
        # through the lane's Quack-authoritative eligibility sequence.
        assert daemon._task_home_shard_index("DQP-COMPLETE") == 1
        assert daemon._task_home_shard_index("DQP-DEPENDENT") == 0
        assert daemon._task_home_shard_index("DQP-OTHER-1") == 1
        assert daemon._task_home_shard_index("DQP-BLOCKED") == 0
        assert daemon.sync_ready_tasks_into_coordination() == ["task:cid:002"]

        projection = daemon.coordinator.coordination_registry_projection()
        assert {item["task_cid"] for item in projection["tasks"]} == {
            "task:cid:001",
            "task:cid:002",
        }
        assert projection["logical_completions"] == [
            {
                "task_cid": "task:cid:001",
                "status": "succeeded",
                "body": {
                    "authority": "task_source",
                    "authoritative_status": "skipped",
                    "authoritative_revision": 1,
                    "restart_recovery_ready": False,
                    "restart_recovery_owner_session_id": "",
                    "restart_recovery_binding": {},
                    "authoritative_attempt_floor": 0,
                    "authoritative_attempt_floor_source": "",
                },
            }
        ]
        assert daemon.sync_ready_tasks_into_coordination() == ["task:cid:002"]
        assert daemon.coordinator.coordination_registry_projection() == projection

        attempt = daemon.claim_next()
        assert attempt is not None
        assert attempt.task_cid == "task:cid:002"
        # Neither the same-shard blocked task nor the ready task owned by the
        # other strict shard may be claimed from this private coordinator.
        assert daemon.claim_next() is None
        blocked = source.get_task("task:cid:004")
        off_shard = source.get_task("task:cid:003")
        assert blocked is not None and blocked.status == "blocked"
        assert off_shard is not None and off_shard.status == "ready"
    finally:
        daemon.close()
        source.close()


def test_sync_demotes_only_registered_stale_ready_terminal_rows(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:ready-frontier-reconciliation",
    )
    try:
        daemon.materialize_population(_population(1))
        before = daemon.coordinator.coordination_registry_projection()
        assert before["tasks"][0]["ready"] is True
        assert before["logical_completions"] == []

        canonical = daemon.task_source.get("task:cid:001")
        assert canonical is not None
        daemon.task_source._intent.cas_task_status(
            task_cid=canonical.task_cid,
            expected_revision=canonical.revision,
            new_status="completed",
            receipt={"operation": "authoritative_other_lane_completion"},
            allow_completion_without_evidence=True,
        )

        assert daemon.sync_ready_tasks_into_coordination() == []
        after = daemon.coordinator.coordination_registry_projection()
        authoritative_body = {
            "authority": "task_source",
            "authoritative_status": "completed",
            "authoritative_revision": 2,
            "restart_recovery_ready": False,
            "restart_recovery_owner_session_id": "",
            "restart_recovery_binding": {},
            "authoritative_attempt_floor": 0,
            "authoritative_attempt_floor_source": "",
        }
        assert after["tasks"] == [
            {
                **before["tasks"][0],
                "ready": False,
                "body": authoritative_body,
            }
        ]
        assert after["logical_completions"] == [
            {
                "task_cid": "task:cid:001",
                "status": "succeeded",
                "body": authoritative_body,
            }
        ]

        # A later synchronization sees the exact authoritative projection and
        # does not create another row or alter its durable completion fact.
        assert daemon.sync_ready_tasks_into_coordination() == []
        assert daemon.coordinator.coordination_registry_projection() == after
        assert daemon.claim_next() is None
    finally:
        daemon.close()

    fresh = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination-terminal-fresh.duckdb",
        execution_path=tmp_path / "execution-terminal-fresh.duckdb",
        owner_session_id="session:ready-frontier-fresh",
        authority_mode="embedded",
        task_source_kind="duckdb",
        require_real_execution=True,
    )
    try:
        assert fresh.sync_ready_tasks_into_coordination() == []
        fresh_projection = fresh.coordinator.coordination_registry_projection()
        assert fresh_projection["tasks"] == after["tasks"]
        assert fresh_projection["logical_completions"] == after[
            "logical_completions"
        ]
    finally:
        fresh.close()


def test_sync_retries_mixed_revision_reopen_without_demoting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:ready-frontier-mixed-revision",
    )
    try:
        daemon.materialize_population(_population(1))
        canonical = daemon.task_source.get("task:cid:001")
        assert canonical is not None
        daemon.task_source._intent.cas_task_status(
            task_cid=canonical.task_cid,
            expected_revision=canonical.revision,
            new_status="completed",
            receipt={"operation": "other_lane_completion_before_mixed_read"},
            allow_completion_without_evidence=True,
        )

        task_source = daemon.task_source
        original_list_tasks = task_source.list_tasks
        inventory_reads = {"count": 0}

        def list_tasks_with_one_reopen(*args: object, **kwargs: object) -> object:
            page = original_list_tasks(*args, **kwargs)
            inventory_reads["count"] += 1
            if inventory_reads["count"] == 1:
                completed = task_source.get("task:cid:001")
                assert completed is not None
                assert completed.status == "completed"
                task_source._intent.cas_task_status(
                    task_cid=completed.task_cid,
                    expected_revision=completed.revision,
                    new_status="retrying",
                    receipt={"operation": "owner_reopen_between_snapshot_reads"},
                )
            return page

        monkeypatch.setattr(task_source, "list_tasks", list_tasks_with_one_reopen)
        assert daemon.sync_ready_tasks_into_coordination() == ["task:cid:001"]
        assert inventory_reads["count"] >= 2
        projection = daemon.coordinator.coordination_registry_projection()
        assert projection["tasks"][0]["ready"] is True
        assert all(
            event["event_type"] != "control_ready_frontier_reconciled"
            for event in daemon.coordinator.lease_events(limit=100)
        )
    finally:
        daemon.close()


def test_next_sync_repairs_reopen_after_equal_terminal_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:ready-frontier-post-snapshot-reopen",
    )
    try:
        daemon.materialize_population(_population(1))
        canonical = daemon.task_source.get("task:cid:001")
        assert canonical is not None
        daemon.task_source._intent.cas_task_status(
            task_cid=canonical.task_cid,
            expected_revision=canonical.revision,
            new_status="completed",
            receipt={"operation": "other_lane_completion_before_snapshot"},
            allow_completion_without_evidence=True,
        )

        coordinator = daemon.coordinator
        original_synchronize = coordinator.synchronize_authoritative_task
        reopened = {"done": False}

        def synchronize_then_reopen(*args: object, **kwargs: object) -> object:
            result = original_synchronize(*args, **kwargs)
            if not reopened["done"]:
                completed = daemon.task_source.get("task:cid:001")
                assert completed is not None
                assert completed.status == "completed"
                daemon.task_source._intent.cas_task_status(
                    task_cid=completed.task_cid,
                    expected_revision=completed.revision,
                    new_status="retrying",
                    receipt={"operation": "owner_reopen_after_equal_snapshot"},
                )
                reopened["done"] = True
            return result

        monkeypatch.setattr(
            coordinator,
            "synchronize_authoritative_task",
            synchronize_then_reopen,
        )
        assert daemon.sync_ready_tasks_into_coordination() == []
        assert reopened["done"] is True
        assert coordinator.coordination_registry_projection()["tasks"][0][
            "ready"
        ] is False

        assert daemon.sync_ready_tasks_into_coordination() == ["task:cid:001"]
        assert coordinator.coordination_registry_projection()["tasks"][0][
            "ready"
        ] is True
        final_task = coordinator.coordination_registry_projection()["tasks"][0]
        assert final_task["body"]["authority"] == "task_source"
        assert final_task["body"]["authoritative_status"] == "retrying"
    finally:
        daemon.close()


def test_conflicting_local_completion_refuses_coordination_drift(
    tmp_path: Path,
) -> None:
    population = _population(2)
    tasks = population["tasks"]
    assert isinstance(tasks, list)
    tasks[0]["status"] = "skipped"
    tasks[1]["dependencies"] = ["task:cid:001"]

    source = DatabaseTaskSource(tmp_path / "control.duckdb")
    source.materialize(population, repository_tree_id="tree:dqp-local-drift")
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "lane" / "coordination.duckdb",
        execution_path=tmp_path / "lane" / "execution.duckdb",
        owner_session_id="session:local-drift",
        authority_mode="embedded",
        task_source_kind="duckdb",
        task_source=source,
        require_real_execution=True,
    )
    try:
        daemon.coordinator.register_task(
            task_cid="task:cid:001",
            task_id="DQP-T001",
        )
        daemon.coordinator.mark_task_complete(
            "task:cid:001",
            status="failed",
            body={"source": "stale-lane-local-evidence"},
        )

        with pytest.raises(
            DatabaseImplementationCoordinationDriftError,
            match="lane-local completion contradicts",
        ):
            daemon.claim_next()
        assert daemon.list_running_attempts() == []
        dependent = source.get_task("task:cid:002")
        assert dependent is not None and dependent.status == "ready"
    finally:
        daemon.close()
        source.close()


def test_portal_deferral_refreshes_failed_revision_and_releases_exact_lease(
    tmp_path: Path,
) -> None:
    provider_calls: list[str] = []

    def defer_provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_calls.append(attempt.attempt_id)
        if len(provider_calls) == 1:
            raise DatabasePortalBridgeDeferred(
                "validation_project_dependency_preflight_failed",
                backoff_seconds=0,
            )
        return {"status": "ok", "accepted": True, "task_cid": attempt.task_cid}

    daemon = _open_daemon(
        tmp_path,
        session="session:portal-deferral",
        provider_fn=defer_provider,
        strict_task_sharding=True,
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None

        result = daemon._resume_attempt_without_process_crash(attempt)

        assert provider_calls == [attempt.attempt_id]
        assert result["status"] == "failed"
        assert "fail_error" not in result
        failed = daemon.get_attempt(attempt.attempt_id)
        assert failed is not None
        assert failed.committed_phase == "failed"
        assert failed.status == "failed"
        assert failed.revision > attempt.revision
        projection = daemon.coordinator.coordination_registry_projection()
        assert next(
            row["state"]
            for row in projection["task_claims"]
            if row["claim_id"] == attempt.claim_id
        ) == "released"
        assert next(
            row["status"]
            for row in projection["task_attempts"]
            if row["attempt_id"] == attempt.attempt_id
        ) == "released"
        assert next(
            row["state"]
            for row in projection["fenced_leases"]
            if row["lease_id"] == attempt.lease_id
        ) == "released"

        retry = daemon.claim_next()
        assert retry is not None
        assert retry.task_cid == attempt.task_cid
        assert retry.attempt_number == 2
        resumed = daemon.resume_attempt(retry)
        assert resumed["resumed"] is True
        assert resumed["status"] == "succeeded"
        assert provider_calls == [attempt.attempt_id, retry.attempt_id]
    finally:
        daemon.close()


def test_consumed_no_progress_quarantines_and_abstains_after_restart(
    tmp_path: Path,
) -> None:
    control_path = tmp_path / "control.duckdb"
    lane_path = tmp_path / "lane"
    provider_calls: list[str] = []
    failure_evidence: dict[str, object] = {}

    def consumed_failure(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_calls.append(attempt.attempt_id)
        raise DatabasePortalBridgeConsumedNoProgressError(
            "portal_consumed_no_progress",
            failure_evidence=failure_evidence,
        )

    first = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:deterministic-preflight",
        provider_fn=consumed_failure,
        strict_task_sharding=True,
    )
    try:
        first.materialize_population(_population(1))
        attempted = first.claim_next()
        assert attempted is not None
        failure_evidence.update(
            _consumed_no_progress_evidence(
                first,
                attempted,
                tag="preflight-symbol-drift",
            )
        )
        fingerprint = str(failure_evidence["failure_fingerprint"])
        first_result = first._resume_attempt_without_process_crash(attempted)

        assert first_result["status"] == "blocked"
        assert first_result["portal_retryable_failure"] is False
        assert first_result["portal_replay_suppressed"] is True
        assert first_result["task_quarantined"] is True
        assert first_result["root_cause_required"] is True
        assert first_result["failure_fingerprint"] == fingerprint
        assert len(provider_calls) == 1

        blocked = first.get_attempt(attempted.attempt_id)
        assert blocked is not None
        assert blocked.status == "blocked"
        assert blocked.committed_phase == ATTEMPT_PHASE_BLOCKED
        assert first.provider_invocation_exists(blocked.attempt_id) is True
        phases = first.phase_history(blocked.attempt_id)
        blocked_phase = next(
            item for item in phases if item["phase"] == ATTEMPT_PHASE_BLOCKED
        )
        assert blocked_phase["body"]["portal_replay_suppressed"] is True
        assert (
            blocked_phase["body"]["failure_evidence"]["failure_fingerprint"]
            == fingerprint
        )

        task = first.task_source.get(blocked.task_cid)
        assert task is not None
        assert task.status == "quarantined"
        receipt = task.body["completion_receipt"]
        assert receipt["operation"] == (
            "database_portal_neutral_failure_quarantine"
        )
        assert receipt["failure_fingerprint"] == fingerprint
        assert receipt["retry_suppressed"] is True
        assert receipt["circuit_breaker_key"] == fingerprint
        assert receipt["provider_effect_state"] == "unknown_may_have_started"
        intent = first.provider_invocation_recorded(
            blocked.attempt_id,
            idempotency_key=f"provider:{blocked.attempt_id}",
        )
        assert intent is not None
        assert intent["database_binding_id"] == failure_evidence[
            "database_binding_id"
        ]
        assert intent["portal_failure_fingerprint"] == fingerprint
        assert receipt["provider_callback_intent_fingerprint"] == intent[
            "failure_fingerprint"
        ]
        projection = first.coordinator.coordination_registry_projection()
        claims = [
            row
            for row in projection["task_claims"]
            if row["task_cid"] == blocked.task_cid
        ]
        assert len(claims) == 1
        assert claims[0]["state"] == "released"
    finally:
        first.close()

    restarted = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:deterministic-preflight",
        provider_fn=consumed_failure,
        strict_task_sharding=True,
    )
    try:
        replay = restarted.run_once()
        assert replay["selection_idle_reason"] == "no_ready_tasks"
        assert provider_calls == [blocked.attempt_id]
        assert restarted.claim_next() is None
        assert restarted.list_running_attempts() == []
        assert restarted.get_attempt(blocked.attempt_id) == blocked
        projection = restarted.coordinator.coordination_registry_projection()
        assert len(
            [
                row
                for row in projection["task_claims"]
                if row["task_cid"] == blocked.task_cid
            ]
        ) == 1
    finally:
        restarted.close()


def _git_repo_with_output(tmp_path: Path, relative: str = "landed.py") -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Daemon Test"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "daemon-test@example.invalid"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    output = repo / relative
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("landed\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", relative],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "landed output"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return repo


def test_landed_quarantined_task_with_outputs_is_completed(
    tmp_path: Path,
) -> None:
    repo = _git_repo_with_output(tmp_path)
    daemon = _open_daemon(tmp_path / "lane", repo_root=repo)
    try:
        population = _population(1)
        tasks = population["tasks"]
        assert isinstance(tasks, list)
        tasks[0]["outputs"] = [{"path": "landed.py"}]
        daemon.materialize_population(population)
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        daemon.task_source.compare_and_set_status(
            "task:cid:001",
            int(task.revision),
            "quarantined",
            receipt={
                "operation": "database_portal_neutral_failure_quarantine",
                "retry_suppressed": True,
            },
        )
        result = daemon.run_once()
        repaired = result["landed_merge_reconciliations"]
        assert repaired
        assert repaired[0]["completed"] is True
        assert repaired[0]["task_cid"] == "task:cid:001"
        completed = daemon.task_source.get("task:cid:001")
        assert completed is not None
        assert completed.status == "completed"
        assert result["selection_idle_reason"] == "no_ready_tasks"
    finally:
        daemon.close()


def test_declared_output_paths_split_database_body_csv_fields() -> None:
    task = SimpleNamespace(
        outputs=(),
        body={
            "predicted_files": "first.py, nested/second.py",
            "outputs": "third.py",
        },
    )

    assert DatabaseImplementationDaemon._task_declared_output_paths(task) == (
        "first.py",
        "nested/second.py",
        "third.py",
    )


def test_landed_quarantine_rejects_csv_with_unsafe_segment(
    tmp_path: Path,
) -> None:
    repo = _git_repo_with_output(tmp_path)
    daemon = _open_daemon(tmp_path / "lane", repo_root=repo)
    try:
        population = _population(1)
        tasks = population["tasks"]
        assert isinstance(tasks, list)
        tasks[0]["predicted_files"] = "landed.py, ../missing.py"
        daemon.materialize_population(population)
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        daemon.task_source.compare_and_set_status(
            "task:cid:001",
            int(task.revision),
            "quarantined",
            receipt=_unknown_callback_quarantine_receipt(),
        )
        quarantined = daemon.task_source.get("task:cid:001")
        assert quarantined is not None
        assert daemon._task_declared_output_paths(quarantined) == ()
        assert daemon._task_outputs_landed_on_target(quarantined) is False

        result = daemon.run_once()

        assert result["landed_merge_reconciliations"] == []
        assert result["unknown_callback_reopens"] == []
        unchanged = daemon.task_source.get("task:cid:001")
        assert unchanged is not None
        assert unchanged.status == "quarantined"
    finally:
        daemon.close()


def test_landed_quarantine_repairs_comma_separated_predicted_files(
    tmp_path: Path,
) -> None:
    repo = _git_repo_with_output(tmp_path, "pkg/a.py")
    for relative in ("pkg/b.py", "test/test_a.py"):
        output = repo / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("landed\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "pkg/b.py", "test/test_a.py"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "land remaining outputs"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    daemon = _open_daemon(tmp_path / "lane", repo_root=repo)
    try:
        population = _population(1)
        tasks = population["tasks"]
        assert isinstance(tasks, list)
        tasks[0]["predicted_files"] = "pkg/a.py, pkg/b.py, test/test_a.py"
        daemon.materialize_population(population)
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        daemon.task_source.compare_and_set_status(
            "task:cid:001",
            int(task.revision),
            "quarantined",
            receipt=_unknown_callback_quarantine_receipt(),
        )
        quarantined = daemon.task_source.get("task:cid:001")
        assert quarantined is not None
        assert quarantined.status == "quarantined"
        assert daemon._task_declared_output_paths(quarantined) == (
            "pkg/a.py",
            "pkg/b.py",
            "test/test_a.py",
        )
        assert daemon._task_outputs_landed_on_target(quarantined) is True

        result = daemon.run_once()

        repaired = result["landed_merge_reconciliations"]
        assert len(repaired) == 1
        assert repaired[0]["completed"] is True
        assert repaired[0]["landed_outputs"] == [
            "pkg/a.py",
            "pkg/b.py",
            "test/test_a.py",
        ]
        assert result["unknown_callback_reopens"] == []
        completed = daemon.task_source.get("task:cid:001")
        assert completed is not None
        assert completed.status == "completed"
    finally:
        daemon.close()


def test_consumed_no_progress_completes_when_declared_outputs_already_landed(
    tmp_path: Path,
) -> None:
    repo = _git_repo_with_output(tmp_path)
    control_path = tmp_path / "control.duckdb"
    lane_path = tmp_path / "lane"
    failure_evidence: dict[str, object] = {}

    def consumed_failure(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeConsumedNoProgressError(
            "portal_consumed_no_progress",
            failure_evidence=failure_evidence,
        )

    daemon = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:landed-outputs",
        provider_fn=consumed_failure,
        repo_root=repo,
    )
    try:
        population = _population(1)
        tasks = population["tasks"]
        assert isinstance(tasks, list)
        tasks[0]["outputs"] = [{"path": "landed.py"}]
        daemon.materialize_population(population)
        attempted = daemon.claim_next()
        assert attempted is not None
        failure_evidence.update(
            _consumed_no_progress_evidence(
                daemon,
                attempted,
                tag="already-landed-outputs",
            )
        )
        result = daemon._resume_attempt_without_process_crash(attempted)
        assert result["task_quarantined"] is False
        assert result["landed_outputs_completed"] is True
        assert result["status"] == "succeeded"
        task = daemon.task_source.get(attempted.task_cid)
        assert task is not None
        assert task.status == "completed"
    finally:
        daemon.close()


def test_reopened_quarantine_retires_stale_blocked_attempt(
    tmp_path: Path,
) -> None:
    control_path = tmp_path / "control.duckdb"
    lane_path = tmp_path / "lane"
    failure_evidence: dict[str, object] = {}

    def consumed_failure(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeConsumedNoProgressError(
            "portal_consumed_no_progress",
            failure_evidence=failure_evidence,
        )

    first = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:reopen-stale-block",
        provider_fn=consumed_failure,
    )
    try:
        first.materialize_population(_population(1))
        attempted = first.claim_next()
        assert attempted is not None
        failure_evidence.update(
            _consumed_no_progress_evidence(
                first,
                attempted,
                tag="reopen-stale-block",
            )
        )
        first._resume_attempt_without_process_crash(attempted)
        blocked = first.get_attempt(attempted.attempt_id)
        assert blocked is not None
        assert blocked.status == "blocked"
        task = first.task_source.get(blocked.task_cid)
        assert task is not None
        assert task.status == "quarantined"
        first.task_source.compare_and_set_status(
            blocked.task_cid,
            int(task.revision),
            "todo",
            receipt={
                "operation": "reopen_unimplemented_unknown_callback_quarantine",
                "reason": "declared_outputs_missing_on_merge_target",
            },
        )
        reopened = first.task_source.get(blocked.task_cid)
        assert reopened is not None
        assert reopened.status == "todo"

        outcomes = first.reconcile_expired_running_attempts()
        retired = [
            item
            for item in outcomes
            if item.get("reason") == "control_task_left_quarantine"
        ]
        assert retired
        assert retired[0]["attempt_id"] == blocked.attempt_id
        assert retired[0]["status"] == "failed"
        stale = first.get_attempt(blocked.attempt_id)
        assert stale is not None
        assert stale.status == "failed"
        current = first.task_source.get(blocked.task_cid)
        assert current is not None
        assert current.status == "todo"
        assert first.list_running_attempts() == []
    finally:
        first.close()


def _git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Daemon Test"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "daemon-test@example.invalid"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "base"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return repo


def _unknown_callback_quarantine_receipt() -> dict[str, object]:
    return {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "database-portal-neutral-quarantine@1"
        ),
        "operation": "database_portal_neutral_failure_quarantine",
        "failure_kind": "provider_callback_outcome_unknown",
        "retry_suppressed": True,
        "root_cause_required": True,
        "provider_effect_state": "unknown_may_have_started",
        "unknown_callback_reopen_count": 0,
    }


def test_unknown_callback_without_landed_outputs_reopens(
    tmp_path: Path,
) -> None:
    repo = _git_repo(tmp_path)
    daemon = _open_daemon(tmp_path / "lane", repo_root=repo)
    try:
        population = _population(1)
        tasks = population["tasks"]
        assert isinstance(tasks, list)
        tasks[0]["outputs"] = [{"path": "missing.py"}]
        daemon.materialize_population(population)
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        daemon.task_source.compare_and_set_status(
            "task:cid:001",
            int(task.revision),
            "quarantined",
            receipt=_unknown_callback_quarantine_receipt(),
        )
        result = daemon.run_once()
        reopened = result["unknown_callback_reopens"]
        assert reopened
        assert reopened[0]["reopened"] is True
        assert reopened[0]["task_cid"] == "task:cid:001"
        current = daemon.task_source.get("task:cid:001")
        assert current is not None
        assert current.status != "quarantined"
    finally:
        daemon.close()


def test_unknown_callback_without_declared_outputs_stays_quarantined(
    tmp_path: Path,
) -> None:
    repo = _git_repo(tmp_path)
    daemon = _open_daemon(tmp_path / "lane", repo_root=repo)
    try:
        daemon.materialize_population(_population(1))
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        daemon.task_source.compare_and_set_status(
            "task:cid:001",
            int(task.revision),
            "quarantined",
            receipt=_unknown_callback_quarantine_receipt(),
        )
        result = daemon.run_once()
        assert result["unknown_callback_reopens"] == []
        current = daemon.task_source.get("task:cid:001")
        assert current is not None
        assert current.status == "quarantined"
        assert result["selection_idle_reason"] == "no_ready_tasks"
    finally:
        daemon.close()


def test_unknown_callback_reopen_count_survives_later_claim_receipt(
    tmp_path: Path,
) -> None:
    repo = _git_repo(tmp_path)
    daemon = _open_daemon(tmp_path / "lane", repo_root=repo)
    try:
        population = _population(1)
        tasks = population["tasks"]
        assert isinstance(tasks, list)
        tasks[0]["outputs"] = [{"path": "missing.py"}]
        daemon.materialize_population(population)
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        daemon.task_source.compare_and_set_status(
            "task:cid:001",
            int(task.revision),
            "quarantined",
            receipt=_unknown_callback_quarantine_receipt(),
        )
        first = daemon.run_once()
        assert first["unknown_callback_reopens"]
        assert first["unknown_callback_reopens"][0]["unknown_callback_reopen_count"] == 1
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        daemon.task_source.compare_and_set_status(
            "task:cid:001",
            int(task.revision),
            "in_progress",
            receipt={
                "operation": "database_claim",
                "claim_id": "claim:fresh",
                "attempt_id": "attempt:fresh",
            },
        )
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        claim_receipt = task.body.get("completion_receipt")
        assert isinstance(claim_receipt, dict)
        assert claim_receipt.get("unknown_callback_reopen_count") == 1
        receipt = _unknown_callback_quarantine_receipt()
        receipt.pop("unknown_callback_reopen_count", None)
        daemon.task_source.compare_and_set_status(
            "task:cid:001",
            int(task.revision),
            "quarantined",
            receipt=receipt,
        )
        second = daemon.run_once()
        assert second["unknown_callback_reopens"]
        assert second["unknown_callback_reopens"][0]["unknown_callback_reopen_count"] == 2
        current = daemon.task_source.get("task:cid:001")
        assert current is not None
        current_receipt = current.body.get("completion_receipt")
        assert isinstance(current_receipt, dict)
        assert current_receipt.get("unknown_callback_reopen_count") == 2
    finally:
        daemon.close()


def test_unknown_callback_quarantine_receipt_count_does_not_block_reopen(
    tmp_path: Path,
) -> None:
    repo = _git_repo(tmp_path)
    daemon = _open_daemon(tmp_path / "lane", repo_root=repo)
    try:
        population = _population(1)
        tasks = population["tasks"]
        assert isinstance(tasks, list)
        tasks[0]["outputs"] = [{"path": "missing.py"}]
        daemon.materialize_population(population)
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        receipt = _unknown_callback_quarantine_receipt()
        receipt["unknown_callback_reopen_count"] = 4
        daemon.task_source.compare_and_set_status(
            "task:cid:001",
            int(task.revision),
            "quarantined",
            receipt=receipt,
        )
        result = daemon.run_once()
        reopened = result["unknown_callback_reopens"]
        assert reopened
        assert reopened[0]["reopened"] is True
        current = daemon.task_source.get("task:cid:001")
        assert current is not None
        assert current.status != "quarantined"
    finally:
        daemon.close()


def test_unaccepted_unknown_callback_is_retired_not_quarantined(
    tmp_path: Path,
) -> None:
    class SimulatedProcessCrash(BaseException):
        pass

    now = {"ms": 1_000}
    control_path = tmp_path / "control.duckdb"
    lane_path = tmp_path / "lane"
    repo = _git_repo(tmp_path)

    def crash_after_callback_started(
        attempt: DatabaseTaskAttempt,
    ) -> dict[str, object]:
        raise SimulatedProcessCrash("injected unaccepted-claim crash")

    first = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:unaccepted-unknown",
        provider_fn=crash_after_callback_started,
        strict_task_sharding=True,
        repo_root=repo,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        population = _population(1)
        tasks = population["tasks"]
        assert isinstance(tasks, list)
        tasks[0]["outputs"] = [{"path": "missing.py"}]
        first.materialize_population(population)
        attempt = first.claim_next()
        assert attempt is not None
        with pytest.raises(SimulatedProcessCrash):
            first._resume_attempt_without_process_crash(attempt)
        now["ms"] = 10_000
        claim = first.coordinator.get_task_claim(attempt.claim_id)
        assert claim is not None
        first.coordinator.expire_task_claim(claim, now_ms=now["ms"])
        task = first.task_source.get(attempt.task_cid)
        assert task is not None
        first.task_source.compare_and_set_status(
            attempt.task_cid,
            int(task.revision),
            "in_progress",
            receipt={
                "operation": "database_claim",
                "claim_id": "claim:other-owner",
                "attempt_id": "attempt:other-owner",
            },
        )
    finally:
        first.close()

    restarted = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:unaccepted-unknown",
        provider_fn=lambda attempt: {
            "status": "ok",
            "accepted": True,
            "task_cid": attempt.task_cid,
        },
        strict_task_sharding=True,
        repo_root=repo,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        replay = restarted.run_once()
        current = restarted.task_source.get(attempt.task_cid)
        assert current is not None
        assert current.status != "quarantined"
        retired = [
            item
            for item in replay["expired_attempt_reconciliations"]
            if item.get("attempt_id") == attempt.attempt_id
        ]
        assert retired
        assert retired[0]["reason"] != "portal_neutral_failure"
    finally:
        restarted.close()


def test_portal_setup_error_requeues_instead_of_unknown_callback_quarantine(
    tmp_path: Path,
) -> None:
    repo = _git_repo(tmp_path)
    daemon = _open_daemon(
        tmp_path / "lane",
        repo_root=repo,
        provider_fn=lambda attempt: (_ for _ in ()).throw(
            DatabasePortalBridgeError(
                "external_protected_checkout_recovery_required"
            )
        ),
        strict_task_sharding=True,
    )
    try:
        population = _population(1)
        tasks = population["tasks"]
        assert isinstance(tasks, list)
        tasks[0]["outputs"] = [{"path": "missing.py"}]
        daemon.materialize_population(population)
        attempt = daemon.claim_next()
        assert attempt is not None
        result = daemon._resume_attempt_without_process_crash(attempt)
        assert result["retryable"] is True
        assert "external_protected_checkout_recovery_required" in str(
            result.get("reason") or ""
        )
        current = daemon.task_source.get(attempt.task_cid)
        assert current is not None
        assert current.status != "quarantined"
        stale = daemon.get_attempt(attempt.attempt_id)
        assert stale is not None
        assert stale.status != "running"
    finally:
        daemon.close()


def _write_supervisor_protected_recovery_journal(repo: Path) -> Path:
    lock_path = checkout_mutation_lock_path(repo)
    metadata = checkout_lock_metadata(
        kind="merge",
        repo_root=repo,
        task_id="APMC-013",
        attempt=1,
        extra={"operation": "generated_board_update"},
    )
    metadata.update(
        {
            "pid": 999999999,
            "protected_recovery_required": True,
            "protected_recovery_owner": "implementation_supervisor",
            "protected_paths": ["docs/generated.todo.md"],
        }
    )
    lock_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    return lock_path


def test_protected_checkout_setup_block_classifier() -> None:
    assert is_protected_checkout_setup_block(
        "external_protected_checkout_recovery_required"
    )
    assert is_protected_checkout_setup_block(
        "DatabasePortalBridgeError: external protected checkout recovery required"
    )
    assert not is_protected_checkout_setup_block("portal_consumed_no_progress")


def test_supervisor_recovery_journal_defers_before_callback_intent(
    tmp_path: Path,
) -> None:
    repo = _git_repo(tmp_path)
    lock_path = _write_supervisor_protected_recovery_journal(repo)
    provider_calls: list[str] = []

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_calls.append(attempt.task_cid)
        return {"status": "ok", "accepted": True, "task_cid": attempt.task_cid}

    daemon = _open_daemon(
        tmp_path / "lane",
        repo_root=repo,
        provider_fn=provider,
        strict_task_sharding=True,
    )
    try:
        population = _population(1)
        tasks = population["tasks"]
        assert isinstance(tasks, list)
        tasks[0]["outputs"] = [{"path": "missing.py"}]
        daemon.materialize_population(population)
        attempt = daemon.claim_next()
        assert attempt is not None
        result = daemon._resume_attempt_without_process_crash(attempt)
        assert result["retryable"] is True
        assert "external_protected_checkout_recovery_required" in str(
            result.get("reason") or ""
        )
        current = daemon.task_source.get(attempt.task_cid)
        assert current is not None
        assert current.status != "quarantined"
        assert current.status == "todo"
        assert provider_calls == []
        recorded = daemon.provider_invocation_recorded(
            attempt.attempt_id,
            idempotency_key=f"provider:{attempt.attempt_id}",
        )
        assert recorded is None
        assert lock_path.is_file()
        stale = daemon.get_attempt(attempt.attempt_id)
        assert stale is not None
        assert stale.status != "running"
    finally:
        daemon.close()


def test_unknown_callback_reopen_continues_while_outputs_are_missing(
    tmp_path: Path,
) -> None:
    repo = _git_repo(tmp_path)
    daemon = _open_daemon(tmp_path / "lane", repo_root=repo)
    try:
        population = _population(1)
        tasks = population["tasks"]
        assert isinstance(tasks, list)
        tasks[0]["outputs"] = [{"path": "missing.py"}]
        daemon.materialize_population(population)
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        daemon.task_source.compare_and_set_status(
            "task:cid:001",
            int(task.revision),
            "todo",
            receipt={
                "operation": "reopen_unimplemented_unknown_callback_quarantine",
                "unknown_callback_reopen_count": 4,
            },
        )
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        receipt = _unknown_callback_quarantine_receipt()
        daemon.task_source.compare_and_set_status(
            "task:cid:001",
            int(task.revision),
            "quarantined",
            receipt=receipt,
        )
        result = daemon.run_once()
        reopened = result["unknown_callback_reopens"]
        assert reopened
        assert reopened[0]["reopened"] is True
        current = daemon.task_source.get("task:cid:001")
        assert current is not None
        assert current.status != "quarantined"
    finally:
        daemon.close()


def test_reopened_task_retires_stale_running_unknown_callback(
    tmp_path: Path,
) -> None:
    class SimulatedProcessCrash(BaseException):
        pass

    control_path = tmp_path / "control.duckdb"
    lane_path = tmp_path / "lane"
    provider_calls: list[str] = []

    def crash_after_callback_started(
        attempt: DatabaseTaskAttempt,
    ) -> dict[str, object]:
        provider_calls.append(attempt.attempt_id)
        raise SimulatedProcessCrash("injected leftover running crash")

    first = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:stale-running-reopen",
        provider_fn=crash_after_callback_started,
        strict_task_sharding=True,
    )
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        with pytest.raises(
            SimulatedProcessCrash,
            match="injected leftover running crash",
        ):
            first._resume_attempt_without_process_crash(attempt)
        running = first.get_attempt(attempt.attempt_id)
        assert running is not None and running.status == "running"
        task = first.task_source.get(attempt.task_cid)
        assert task is not None
        first.task_source.compare_and_set_status(
            attempt.task_cid,
            int(task.revision),
            "todo",
            receipt={
                "operation": "reopen_unimplemented_unknown_callback_quarantine",
                "unknown_callback_reopen_count": 1,
            },
        )
    finally:
        first.close()

    def success_provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_calls.append(attempt.attempt_id)
        return {"status": "ok", "accepted": True, "task_cid": attempt.task_cid}

    restarted = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:stale-running-reopen",
        provider_fn=success_provider,
        strict_task_sharding=True,
    )
    try:
        replay = restarted.run_once()
        retired = [
            item
            for item in replay["expired_attempt_reconciliations"]
            if item.get("reason") == "control_task_left_in_progress"
        ]
        assert retired
        assert retired[0]["attempt_id"] == attempt.attempt_id
        stale = restarted.get_attempt(attempt.attempt_id)
        assert stale is not None
        assert stale.status == "failed"
        current = restarted.task_source.get(attempt.task_cid)
        assert current is not None
        assert current.status != "quarantined"
        assert attempt.attempt_id not in provider_calls[1:]
    finally:
        restarted.close()


def test_expired_unknown_callback_with_rebound_in_progress_is_retired(
    tmp_path: Path,
) -> None:
    class SimulatedProcessCrash(BaseException):
        pass

    now = {"ms": 1_000}
    control_path = tmp_path / "control.duckdb"
    lane_path = tmp_path / "lane"

    def crash_after_callback_started(
        attempt: DatabaseTaskAttempt,
    ) -> dict[str, object]:
        raise SimulatedProcessCrash("injected rebound in-progress crash")

    first = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:rebound-in-progress",
        provider_fn=crash_after_callback_started,
        strict_task_sharding=True,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        with pytest.raises(SimulatedProcessCrash):
            first._resume_attempt_without_process_crash(attempt)
        task = first.task_source.get(attempt.task_cid)
        assert task is not None
        first.task_source.compare_and_set_status(
            attempt.task_cid,
            int(task.revision),
            "todo",
            receipt={"operation": "reopen_unimplemented_unknown_callback_quarantine"},
        )
        task = first.task_source.get(attempt.task_cid)
        assert task is not None
        first.task_source.compare_and_set_status(
            attempt.task_cid,
            int(task.revision),
            "in_progress",
            receipt={
                "operation": "database_claim",
                "claim_id": "claim:rebound-owner",
                "attempt_id": "attempt:rebound-owner",
            },
        )
    finally:
        first.close()

    now["ms"] = 7_000
    restarted = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:rebound-in-progress",
        provider_fn=lambda attempt: {
            "status": "ok",
            "accepted": True,
            "task_cid": attempt.task_cid,
        },
        strict_task_sharding=True,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        replay = restarted.run_once()
        retired = [
            item
            for item in replay["expired_attempt_reconciliations"]
            if item.get("reason") == "control_task_left_in_progress"
        ]
        assert retired
        assert retired[0]["attempt_id"] == attempt.attempt_id
        stale = restarted.get_attempt(attempt.attempt_id)
        assert stale is not None
        assert stale.status == "failed"
    finally:
        restarted.close()


def test_consumed_no_progress_quarantine_replays_after_commit_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_path = tmp_path / "control.duckdb"
    lane_path = tmp_path / "lane"
    provider_calls: list[str] = []
    failure_evidence: dict[str, object] = {}

    def consumed_failure(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_calls.append(attempt.attempt_id)
        raise DatabasePortalBridgeConsumedNoProgressError(
            "portal_consumed_no_progress",
            failure_evidence=failure_evidence,
        )

    first = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:preflight-commit-crash",
        provider_fn=consumed_failure,
        strict_task_sharding=True,
    )
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        failure_evidence.update(
            _consumed_no_progress_evidence(
                first,
                attempt,
                tag="commit-crash",
            )
        )
        fingerprint = str(failure_evidence["failure_fingerprint"])
        original_commit_phase = first.commit_phase

        def crash_before_blocked_phase(
            current: DatabaseTaskAttempt,
            phase: str,
            *,
            body: dict[str, object] | None = None,
        ) -> DatabaseTaskAttempt:
            if phase == ATTEMPT_PHASE_BLOCKED:
                raise RuntimeError("injected crash before blocked phase")
            return original_commit_phase(current, phase, body=body)

        monkeypatch.setattr(first, "commit_phase", crash_before_blocked_phase)
        with pytest.raises(
            RuntimeError,
            match="injected crash before blocked phase",
        ):
            first._resume_attempt_without_process_crash(attempt)

        running = first.get_attempt(attempt.attempt_id)
        assert running is not None and running.status == "running"
        task = first.task_source.get(attempt.task_cid)
        assert task is not None and task.status == "quarantined"
        assert task.body["completion_receipt"]["failure_fingerprint"] == (
            fingerprint
        )
        assert first.provider_invocation_exists(attempt.attempt_id) is True
        assert provider_calls == [attempt.attempt_id]
    finally:
        first.close()

    restarted = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:preflight-commit-crash",
        provider_fn=consumed_failure,
        strict_task_sharding=True,
    )
    try:
        replay = restarted.run_once()
        reconciled = replay["expired_attempt_reconciliations"]
        assert len(reconciled) == 1
        assert reconciled[0]["reason"] == (
            "portal_neutral_failure"
        )
        assert reconciled[0]["disposition"] == "quarantined"
        assert provider_calls == [attempt.attempt_id]
        terminal = restarted.get_attempt(attempt.attempt_id)
        assert terminal is not None and terminal.status == "failed"
        assert restarted.claim_next() is None
        claims = [
            row
            for row in restarted.coordinator.coordination_registry_projection()[
                "task_claims"
            ]
            if row["task_cid"] == attempt.task_cid
        ]
        assert len(claims) == 1 and claims[0]["state"] == "released"
    finally:
        restarted.close()


def test_cold_restart_rejects_rebound_neutral_receipt_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_path = tmp_path / "control.duckdb"
    lane_path = tmp_path / "lane"
    provider_calls: list[str] = []
    failure_evidence: dict[str, object] = {}

    def consumed_failure(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_calls.append(attempt.attempt_id)
        raise DatabasePortalBridgeConsumedNoProgressError(
            "portal_consumed_no_progress",
            failure_evidence=failure_evidence,
        )

    first = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:rebound-neutral-receipt",
        provider_fn=consumed_failure,
        strict_task_sharding=True,
    )
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        failure_evidence.update(
            _consumed_no_progress_evidence(
                first,
                attempt,
                tag="rebound-neutral-receipt",
            )
        )
        original_commit_phase = first.commit_phase

        def crash_before_blocked_phase(
            current: DatabaseTaskAttempt,
            phase: str,
            *,
            body: dict[str, object] | None = None,
        ) -> DatabaseTaskAttempt:
            if phase == ATTEMPT_PHASE_BLOCKED:
                raise RuntimeError("injected crash before rebound replay")
            return original_commit_phase(current, phase, body=body)

        monkeypatch.setattr(first, "commit_phase", crash_before_blocked_phase)
        with pytest.raises(
            RuntimeError,
            match="injected crash before rebound replay",
        ):
            first._resume_attempt_without_process_crash(attempt)

        task = first.task_source.get(attempt.task_cid)
        assert task is not None and task.status == "quarantined"
        forged_body = dict(task.body)
        forged_receipt = dict(forged_body["completion_receipt"])
        forged_evidence = dict(forged_receipt["failure_evidence"])
        forged_evidence["task_contract_digest"] = "sha256:" + "9" * 64
        forged_evidence["failure_fingerprint"] = (
            database_portal_consumed_no_progress_fingerprint(
                forged_evidence
            )
        )
        forged_receipt["failure_evidence"] = forged_evidence
        forged_receipt["failure_fingerprint"] = forged_evidence[
            "failure_fingerprint"
        ]
        forged_receipt["circuit_breaker_key"] = forged_evidence[
            "failure_fingerprint"
        ]
        evidence_bytes = json.dumps(
            forged_evidence,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        forged_receipt["failure_evidence_digest"] = (
            "sha256:" + hashlib.sha256(evidence_bytes).hexdigest()
        )
        forged_body["completion_receipt"] = forged_receipt
        with first.task_source.intent._connection(write=True) as connection:
            connection.execute(
                "UPDATE tasks SET body_json = ? WHERE task_cid = ?",
                [
                    json.dumps(
                        forged_body,
                        separators=(",", ":"),
                        sort_keys=True,
                        default=str,
                    ),
                    attempt.task_cid,
                ],
            )
        assert provider_calls == [attempt.attempt_id]
    finally:
        first.close()

    restarted = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:rebound-neutral-receipt",
        provider_fn=consumed_failure,
        strict_task_sharding=True,
    )
    try:
        running = restarted.get_attempt(attempt.attempt_id)
        task = restarted.task_source.get(attempt.task_cid)
        assert running is not None and running.status == "running"
        assert task is not None and task.status == "quarantined"
        assert (
            restarted._strict_resume_rejection_receipt_matches(
                task,
                running,
            )
            is False
        )
        assert restarted.reconcile_expired_running_attempts() == []
        claim = restarted.coordinator.get_task_claim(attempt.claim_id)
        assert claim is not None and claim.state.value == "accepted"
        assert provider_calls == [attempt.attempt_id]
    finally:
        restarted.close()


def test_neutral_blocked_claim_release_replays_after_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_path = tmp_path / "control.duckdb"
    lane_path = tmp_path / "lane"
    provider_calls: list[str] = []
    failure_evidence: dict[str, object] = {}

    def consumed_failure(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_calls.append(attempt.attempt_id)
        raise DatabasePortalBridgeConsumedNoProgressError(
            "portal_consumed_no_progress",
            failure_evidence=failure_evidence,
        )

    first = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:preflight-blocked-crash",
        provider_fn=consumed_failure,
        strict_task_sharding=True,
    )
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        failure_evidence.update(
            _consumed_no_progress_evidence(
                first,
                attempt,
                tag="blocked-release-crash",
            )
        )

        def crash_before_exact_release(*args: object, **kwargs: object) -> object:
            raise RuntimeError("injected crash after blocked phase")

        monkeypatch.setattr(
            first.coordinator,
            "release",
            crash_before_exact_release,
        )
        with pytest.raises(
            RuntimeError,
            match="injected crash after blocked phase",
        ):
            first._resume_attempt_without_process_crash(attempt)

        blocked = first.get_attempt(attempt.attempt_id)
        assert blocked is not None
        assert blocked.status == "blocked"
        assert blocked.committed_phase == ATTEMPT_PHASE_BLOCKED
        assert first.provider_invocation_exists(attempt.attempt_id) is True
        task = first.task_source.get(attempt.task_cid)
        assert task is not None and task.status == "quarantined"
        claims = [
            row
            for row in first.coordinator.coordination_registry_projection()[
                "task_claims"
            ]
            if row["claim_id"] == attempt.claim_id
        ]
        assert len(claims) == 1 and claims[0]["state"] == "accepted"
        assert provider_calls == [attempt.attempt_id]
    finally:
        first.close()

    restarted = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:preflight-blocked-crash",
        provider_fn=consumed_failure,
        strict_task_sharding=True,
    )
    try:
        replay = restarted.run_once()
        reconciled = replay["expired_attempt_reconciliations"]
        assert len(reconciled) == 1
        assert reconciled[0]["status"] == "blocked"
        assert reconciled[0]["reason"] == (
            "portal_neutral_failure"
        )
        assert reconciled[0]["disposition"] == "quarantined"
        assert provider_calls == [attempt.attempt_id]
        terminal = restarted.get_attempt(attempt.attempt_id)
        assert terminal is not None and terminal.status == "blocked"
        claims = [
            row
            for row in restarted.coordinator.coordination_registry_projection()[
                "task_claims"
            ]
            if row["claim_id"] == attempt.claim_id
        ]
        assert len(claims) == 1 and claims[0]["state"] == "released"
        second_replay = restarted.run_once()
        assert second_replay["expired_attempt_reconciliations"] == []
        assert second_replay["selection_idle_reason"] == "no_ready_tasks"
        assert provider_calls == [attempt.attempt_id]
    finally:
        restarted.close()


def test_provider_callback_hard_crash_abstains_after_cold_restart(
    tmp_path: Path,
) -> None:
    class SimulatedProcessCrash(BaseException):
        pass

    control_path = tmp_path / "control.duckdb"
    lane_path = tmp_path / "lane"
    provider_calls: list[str] = []

    def crash_after_callback_started(
        attempt: DatabaseTaskAttempt,
    ) -> dict[str, object]:
        provider_calls.append(attempt.attempt_id)
        raise SimulatedProcessCrash("injected hard callback crash")

    first = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:callback-hard-crash",
        provider_fn=crash_after_callback_started,
        strict_task_sharding=True,
    )
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None

        with pytest.raises(
            SimulatedProcessCrash,
            match="injected hard callback crash",
        ):
            first._resume_attempt_without_process_crash(attempt)

        current = first.get_attempt(attempt.attempt_id)
        assert current is not None
        assert current.status == "running"
        assert current.committed_phase == ATTEMPT_PHASE_CONTEXT
        intent = first.provider_invocation_recorded(
            attempt.attempt_id,
            idempotency_key=f"provider:{attempt.attempt_id}",
        )
        assert intent is not None
        assert intent["schema"] == DATABASE_PROVIDER_CALLBACK_UNKNOWN_SCHEMA
        assert intent["callback_state"] == "started_outcome_unknown"
        assert intent["provider_effect_state"] == "unknown_may_have_started"
        assert intent["database_binding_id"] == ""
        assert intent["portal_failure_fingerprint"] == ""
        assert provider_calls == [attempt.attempt_id]
    finally:
        first.close()

    restarted = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:callback-hard-crash",
        provider_fn=crash_after_callback_started,
        strict_task_sharding=True,
    )
    try:
        replay = restarted.run_once()

        assert replay["expired_attempt_reconciliations"] == []
        result = replay["implementation_result"]
        assert result["status"] == "blocked"
        assert result["reason"] == "portal_neutral_failure"
        assert result["failure_kind"] == "provider_callback_outcome_unknown"
        assert result["portal_replay_suppressed"] is True
        assert provider_calls == [attempt.attempt_id]
        blocked = restarted.get_attempt(attempt.attempt_id)
        assert blocked is not None
        assert blocked.status == "blocked"
        assert blocked.committed_phase == ATTEMPT_PHASE_BLOCKED
        task = restarted.task_source.get(attempt.task_cid)
        assert task is not None and task.status == "quarantined"
        receipt = task.body["completion_receipt"]
        assert receipt["provider_effect_state"] == "unknown_may_have_started"
        assert receipt["failure_kind"] == "provider_callback_outcome_unknown"
        claims = [
            row
            for row in restarted.coordinator.coordination_registry_projection()[
                "task_claims"
            ]
            if row["claim_id"] == attempt.claim_id
        ]
        assert len(claims) == 1 and claims[0]["state"] == "released"
    finally:
        restarted.close()


def test_provider_callback_hard_crash_after_expiry_never_redispatches(
    tmp_path: Path,
) -> None:
    class SimulatedProcessCrash(BaseException):
        pass

    now = {"ms": 1_000}
    control_path = tmp_path / "control.duckdb"
    lane_path = tmp_path / "lane"
    provider_calls: list[str] = []

    def crash_after_callback_started(
        attempt: DatabaseTaskAttempt,
    ) -> dict[str, object]:
        provider_calls.append(attempt.attempt_id)
        raise SimulatedProcessCrash("injected expired callback crash")

    first = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:expired-callback-hard-crash",
        provider_fn=crash_after_callback_started,
        strict_task_sharding=True,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        with pytest.raises(
            SimulatedProcessCrash,
            match="injected expired callback crash",
        ):
            first._resume_attempt_without_process_crash(attempt)
        intent = first.provider_invocation_recorded(
            attempt.attempt_id,
            idempotency_key=f"provider:{attempt.attempt_id}",
        )
        assert intent is not None
        assert intent["schema"] == DATABASE_PROVIDER_CALLBACK_UNKNOWN_SCHEMA
        assert provider_calls == [attempt.attempt_id]
    finally:
        first.close()

    now["ms"] = 7_000
    restarted = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:expired-callback-hard-crash",
        provider_fn=crash_after_callback_started,
        strict_task_sharding=True,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        replay = restarted.run_once()
        reconciled = replay["expired_attempt_reconciliations"]
        assert len(reconciled) == 1
        assert reconciled[0]["reason"] == "portal_neutral_failure"
        assert reconciled[0]["disposition"] == "quarantined"
        assert reconciled[0]["retry_required"] is False
        assert replay["implementation_result"] is None
        assert replay["selection_idle_reason"] == "no_ready_tasks"
        assert provider_calls == [attempt.attempt_id]

        terminal = restarted.get_attempt(attempt.attempt_id)
        assert terminal is not None
        assert terminal.status == "failed"
        assert terminal.committed_phase == ATTEMPT_PHASE_FAILED
        task = restarted.task_source.get(attempt.task_cid)
        assert task is not None and task.status == "quarantined"
        receipt = task.body["completion_receipt"]
        assert receipt["failure_kind"] == "provider_callback_outcome_unknown"
        claim = restarted.coordinator.get_task_claim(attempt.claim_id)
        assert claim is not None and claim.state.value == "expired"

        second = restarted.run_once()
        assert second["expired_attempt_reconciliations"] == []
        assert second["selection_idle_reason"] == "no_ready_tasks"
        assert provider_calls == [attempt.attempt_id]
    finally:
        restarted.close()


def test_blocked_response_replay_rejects_different_failure_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure_evidence: dict[str, object] = {}

    def consumed_failure(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeConsumedNoProgressError(
            "portal_consumed_no_progress",
            failure_evidence=failure_evidence,
        )

    daemon = _open_daemon(
        tmp_path,
        session="session:blocked-response-mismatch",
        provider_fn=consumed_failure,
        strict_task_sharding=True,
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        failure_evidence.update(
            _consumed_no_progress_evidence(
                daemon,
                attempt,
                tag="blocked-response-mismatch",
            )
        )
        original_commit_phase = daemon.commit_phase

        def lose_mismatched_blocked_response(
            current: DatabaseTaskAttempt,
            phase: str,
            *,
            body: dict[str, object] | None = None,
        ) -> DatabaseTaskAttempt:
            if phase != ATTEMPT_PHASE_BLOCKED:
                return original_commit_phase(current, phase, body=body)
            forged_body = dict(body or {})
            forged_body["failure_fingerprint"] = "sha256:" + "0" * 64
            original_commit_phase(current, phase, body=forged_body)
            raise RuntimeError("injected lost mismatched blocked response")

        monkeypatch.setattr(
            daemon,
            "commit_phase",
            lose_mismatched_blocked_response,
        )
        with pytest.raises(
            DatabaseImplementationConflictError,
            match="different immutable failure evidence",
        ):
            daemon._resume_attempt_without_process_crash(attempt)

        blocked = daemon.get_attempt(attempt.attempt_id)
        assert blocked is not None and blocked.status == "blocked"
        phase = next(
            item
            for item in daemon.phase_history(attempt.attempt_id)
            if item["phase"] == ATTEMPT_PHASE_BLOCKED
        )
        assert phase["body"]["failure_fingerprint"] == "sha256:" + "0" * 64
        claims = [
            row
            for row in daemon.coordinator.coordination_registry_projection()[
                "task_claims"
            ]
            if row["claim_id"] == attempt.claim_id
        ]
        assert len(claims) == 1 and claims[0]["state"] == "accepted"
    finally:
        daemon.close()


@pytest.mark.parametrize(
    "terminal_phase",
    [ATTEMPT_PHASE_FAILED, ATTEMPT_PHASE_BLOCKED, ATTEMPT_PHASE_COMPLETE],
)
def test_terminal_phase_evidence_is_immutable(
    tmp_path: Path,
    terminal_phase: str,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session=f"session:terminal-immutable:{terminal_phase}",
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        attempt = daemon.commit_phase(
            attempt,
            ATTEMPT_PHASE_CONTEXT,
            body={"step": ATTEMPT_PHASE_CONTEXT},
        )
        if terminal_phase == ATTEMPT_PHASE_COMPLETE:
            for phase in (
                ATTEMPT_PHASE_PROVIDER,
                ATTEMPT_PHASE_EFFECT,
                ATTEMPT_PHASE_VALIDATION,
            ):
                attempt = daemon.commit_phase(
                    attempt,
                    phase,
                    body={"step": phase},
                )

        original_body = {
            "failure_fingerprint": "sha256:" + "1" * 64,
            "failure_evidence_digest": "sha256:" + "2" * 64,
        }
        terminal = daemon.commit_phase(
            attempt,
            terminal_phase,
            body=original_body,
        )
        replay = daemon.commit_phase(
            terminal,
            terminal_phase,
            body=dict(original_body),
        )
        assert replay == terminal

        with pytest.raises(
            DatabaseImplementationConflictError,
            match="different immutable evidence",
        ):
            daemon.commit_phase(
                terminal,
                terminal_phase,
                body={
                    **original_body,
                    "failure_evidence_digest": "sha256:" + "3" * 64,
                },
            )

        terminal_rows = [
            item
            for item in daemon.phase_history(attempt.attempt_id)
            if item["phase"] == terminal_phase
        ]
        assert len(terminal_rows) == 1
        assert terminal_rows[0]["body"] == original_body
    finally:
        daemon.close()


@pytest.mark.parametrize("succeeded", [False, True], ids=["failed", "complete"])
def test_reconciled_terminal_evidence_is_immutable(
    tmp_path: Path,
    succeeded: bool,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session=f"session:reconciled-terminal:{succeeded}",
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        claim = daemon.coordinator.get_task_claim(attempt.claim_id)
        assert claim is not None
        prepared = claim.to_dict()
        prepared["preparation_digest"] = "sha256:" + "4" * 64
        reconciliation = {
            "reason": "first-authoritative-reconciliation",
            "evidence_digest": "sha256:" + "5" * 64,
        }

        terminal = daemon._commit_reconciled_attempt_terminal(
            prepared,
            succeeded=succeeded,
            reconciliation=reconciliation,
        )
        assert terminal is not None
        replay = daemon._commit_reconciled_attempt_terminal(
            prepared,
            succeeded=succeeded,
            reconciliation=dict(reconciliation),
        )
        assert replay == terminal

        with pytest.raises(
            DatabaseImplementationConflictError,
            match="different immutable terminal evidence",
        ):
            daemon._commit_reconciled_attempt_terminal(
                prepared,
                succeeded=succeeded,
                reconciliation={
                    **reconciliation,
                    "evidence_digest": "sha256:" + "6" * 64,
                },
            )

        expected_phase = (
            ATTEMPT_PHASE_COMPLETE if succeeded else ATTEMPT_PHASE_FAILED
        )
        rows = [
            item
            for item in daemon.phase_history(attempt.attempt_id)
            if item["phase"] == expected_phase
        ]
        assert len(rows) == 1
        assert rows[0]["body"]["reconciliation"] == reconciliation
    finally:
        daemon.close()


def test_consumed_failure_stale_task_contract_does_not_quarantine(
    tmp_path: Path,
) -> None:
    failure_evidence: dict[str, object] = {}

    def consumed_failure(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeConsumedNoProgressError(
            "portal_consumed_no_progress",
            failure_evidence=failure_evidence,
        )

    daemon = _open_daemon(
        tmp_path,
        session="session:preflight-stale-task-binding",
        provider_fn=consumed_failure,
        strict_task_sharding=True,
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        failure_evidence.update(
            _consumed_no_progress_evidence(
                daemon,
                attempt,
                tag="stale-task-binding",
            )
        )
        failure_evidence["task_contract_digest"] = "sha256:" + "9" * 64
        failure_evidence["failure_fingerprint"] = (
            database_portal_consumed_no_progress_fingerprint(
                failure_evidence
            )
        )

        with pytest.raises(
            DatabaseImplementationConflictError,
            match="fresh task-bound evaluation is required",
        ):
            daemon._resume_attempt_without_process_crash(attempt)

        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "in_progress"
        current = daemon.get_attempt(attempt.attempt_id)
        assert current is not None and current.status == "running"
        assert current.committed_phase != ATTEMPT_PHASE_BLOCKED
        assert daemon.provider_invocation_exists(attempt.attempt_id) is True
    finally:
        daemon.close()


def test_consumed_failure_structured_validation_race_does_not_quarantine(
    tmp_path: Path,
) -> None:
    failure_evidence: dict[str, object] = {}
    holder: dict[str, DatabaseImplementationDaemon] = {}
    provider_calls: list[str] = []

    def consumed_failure(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_calls.append(attempt.attempt_id)
        daemon = holder["daemon"]
        with daemon.task_source.intent._connection(write=True) as connection:
            connection.execute(
                """
                UPDATE task_validations
                SET argv_json = ?
                WHERE task_cid = ? AND ordinal = 0
                """,
                [json.dumps(["pytest", "changed-contract"]), attempt.task_cid],
            )
        raise DatabasePortalBridgeConsumedNoProgressError(
            "portal_consumed_no_progress",
            failure_evidence=failure_evidence,
        )

    daemon = _open_daemon(
        tmp_path,
        session="session:structured-task-contract-race",
        provider_fn=consumed_failure,
        strict_task_sharding=True,
    )
    holder["daemon"] = daemon
    try:
        daemon.materialize_population(_population(1))
        with daemon.task_source.intent._connection(write=True) as connection:
            connection.execute(
                """
                INSERT INTO task_validations(
                    task_cid, ordinal, argv_json, policy_json
                ) VALUES (?, 0, ?, ?)
                """,
                [
                    "task:cid:001",
                    json.dumps(["pytest", "original-contract"]),
                    "{}",
                ],
            )
        attempt = daemon.claim_next()
        assert attempt is not None
        task_before = daemon.task_source.get(attempt.task_cid)
        assert task_before is not None
        old_revision = task_before.revision
        old_body = dict(task_before.body)
        assert task_before.validations[0]["argv"] == [
            "pytest",
            "original-contract",
        ]
        failure_evidence.update(
            _consumed_no_progress_evidence(
                daemon,
                attempt,
                tag="structured-task-contract-race",
            )
        )

        with pytest.raises(
            DatabaseImplementationConflictError,
            match="provider callback intent is stale or rebound",
        ):
            daemon._resume_attempt_without_process_crash(attempt)

        task_after = daemon.task_source.get(attempt.task_cid)
        assert task_after is not None
        assert task_after.status == "in_progress"
        assert task_after.revision == old_revision
        assert dict(task_after.body) == old_body
        assert task_after.validations[0]["argv"] == [
            "pytest",
            "changed-contract",
        ]
        current = daemon.get_attempt(attempt.attempt_id)
        assert current is not None and current.status == "running"
        assert current.committed_phase == ATTEMPT_PHASE_CONTEXT
        assert provider_calls == [attempt.attempt_id]
    finally:
        daemon.close()


def test_consumed_failure_stale_repository_tree_does_not_quarantine(
    tmp_path: Path,
) -> None:
    failure_evidence: dict[str, object] = {}

    def consumed_failure(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeConsumedNoProgressError(
            "portal_consumed_no_progress",
            failure_evidence=failure_evidence,
        )

    daemon = _open_daemon(
        tmp_path,
        session="session:preflight-stale-tree-binding",
        provider_fn=consumed_failure,
        strict_task_sharding=True,
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        failure_evidence.update(
            _consumed_no_progress_evidence(
                daemon,
                attempt,
                tag="stale-tree-binding",
            )
        )
        failure_evidence["control_repository_tree_id"] = "tree:stale"
        failure_evidence["failure_fingerprint"] = (
            database_portal_consumed_no_progress_fingerprint(
                failure_evidence
            )
        )

        with pytest.raises(
            DatabaseImplementationConflictError,
            match="fresh task-bound evaluation is required",
        ):
            daemon._resume_attempt_without_process_crash(attempt)

        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None and task.status == "in_progress"
        current = daemon.get_attempt(attempt.attempt_id)
        assert current is not None and current.status == "running"
        assert current.committed_phase != ATTEMPT_PHASE_BLOCKED
        assert daemon.provider_invocation_exists(attempt.attempt_id) is True
    finally:
        daemon.close()


def test_consumed_failure_mutated_exception_evidence_fails_closed(
    tmp_path: Path,
) -> None:
    failure_holder: dict[str, Exception] = {}

    def mutated_failure(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise failure_holder["failure"]

    daemon = _open_daemon(
        tmp_path,
        session="session:preflight-mutated-evidence",
        provider_fn=mutated_failure,
        strict_task_sharding=True,
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        evidence = _consumed_no_progress_evidence(
            daemon,
            attempt,
            tag="mutated-after-construction",
        )
        failure = DatabasePortalBridgeConsumedNoProgressError(
            "portal_consumed_no_progress",
            failure_evidence=evidence,
        )
        failure.failure_evidence["tree_id"] = "tree:mutated-after-construction"
        failure_holder["failure"] = failure

        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="failure evidence is invalid",
        ):
            daemon._resume_attempt_without_process_crash(attempt)

        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None and task.status == "in_progress"
        current = daemon.get_attempt(attempt.attempt_id)
        assert current is not None and current.status == "running"
        assert daemon.provider_invocation_exists(attempt.attempt_id) is True
    finally:
        daemon.close()


def test_neutral_failure_cas_replay_rejects_different_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure_evidence: dict[str, object] = {}

    def consumed_failure(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeConsumedNoProgressError(
            "portal_consumed_no_progress",
            failure_evidence=failure_evidence,
        )

    daemon = _open_daemon(
        tmp_path,
        session="session:preflight-cas-evidence-conflict",
        provider_fn=consumed_failure,
        strict_task_sharding=True,
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        failure_evidence.update(
            _consumed_no_progress_evidence(
                daemon,
                attempt,
                tag="expected-cas-evidence",
            )
        )
        real_cas = daemon._cas_task_status_database

        def conflicting_cas(
            task_cid: str,
            *,
            expected_revision: int,
            new_status: str,
            receipt: object = None,
            evidence_digests: object = None,
        ) -> object:
            assert isinstance(receipt, dict)
            forged_evidence = dict(receipt["failure_evidence"])
            forged_evidence["diagnostic_failure_id"] = "failure:different"
            forged_evidence["diagnostic_receipt_id"] = "diagnostic:different"
            forged_evidence["failure_fingerprint"] = (
                database_portal_consumed_no_progress_fingerprint(
                    forged_evidence
                )
            )
            forged_receipt = dict(receipt)
            forged_receipt["failure_evidence"] = forged_evidence
            forged_receipt["failure_fingerprint"] = forged_evidence[
                "failure_fingerprint"
            ]
            forged_receipt["circuit_breaker_key"] = forged_evidence[
                "failure_fingerprint"
            ]
            evidence_bytes = json.dumps(
                forged_evidence,
                separators=(",", ":"),
                sort_keys=True,
                default=str,
            ).encode("utf-8")
            forged_receipt["failure_evidence_digest"] = (
                "sha256:" + hashlib.sha256(evidence_bytes).hexdigest()
            )
            real_cas(
                task_cid,
                expected_revision=expected_revision,
                new_status=new_status,
                receipt=forged_receipt,
                evidence_digests=(
                    str(forged_evidence["failure_fingerprint"]),
                    str(forged_evidence["diagnostic_receipt_digest"]),
                    str(forged_receipt["failure_evidence_digest"]),
                ),
            )
            raise DatabaseTaskSourceConflictError(
                "injected CAS response conflict with different evidence"
            )

        monkeypatch.setattr(
            daemon,
            "_cas_task_status_database",
            conflicting_cas,
        )
        with pytest.raises(
            DatabaseTaskSourceConflictError,
            match="different evidence",
        ):
            daemon._resume_attempt_without_process_crash(attempt)

        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None and task.status == "quarantined"
        assert task.body["completion_receipt"]["failure_fingerprint"] != (
            failure_evidence["failure_fingerprint"]
        )
        current = daemon.get_attempt(attempt.attempt_id)
        assert current is not None and current.status == "running"
        assert current.committed_phase == ATTEMPT_PHASE_CONTEXT
        assert daemon.provider_invocation_exists(attempt.attempt_id) is True
    finally:
        daemon.close()


def test_authoritative_dependency_reopen_invalidates_stale_lane_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(tmp_path, session="session:dependency-reopen")
    dependency_cid = "task:cid:dependency"
    dependent_cid = "task:cid:dependent"
    try:
        daemon.materialize_population(
            {
                "repository_tree_id": "tree:dependency-reopen",
                "objectives": [
                    {
                        "objective_id": "objective:dependency-reopen",
                        "goal_cid": "goal:cid:root",
                        "status": "open",
                    }
                ],
                "tasks": [
                    {
                        "task_cid": dependency_cid,
                        "task_id": "DEP",
                        "goal_cid": "goal:cid:root",
                        "status": "completed",
                        "ordinal": 1,
                    },
                    {
                        "task_cid": dependent_cid,
                        "task_id": "WORK",
                        "goal_cid": "goal:cid:root",
                        "status": "ready",
                        "ordinal": 2,
                        "dependencies": [dependency_cid],
                    },
                ],
            }
        )
        assert daemon.sync_ready_tasks_into_coordination() == [dependent_cid]
        assert daemon.coordinator.claimability(dependent_cid)["claimable"] is True

        real_claim_ready_task = daemon.coordinator.claim_ready_task
        reopened = False

        def claim_then_reopen_dependency(**kwargs: object) -> object:
            nonlocal reopened
            claim = real_claim_ready_task(**kwargs)
            if claim is not None and not reopened:
                dependency = daemon.task_source.get(dependency_cid)
                assert dependency is not None
                daemon.task_source.compare_and_set_status(
                    dependency_cid,
                    expected_revision=int(dependency.revision),
                    status="ready",
                )
                reopened = True
            return claim

        monkeypatch.setattr(
            daemon.coordinator,
            "claim_ready_task",
            claim_then_reopen_dependency,
        )
        assert daemon.claim_next(exclude_task_cids=(dependency_cid,)) is None
        assert reopened is True
        assert daemon.list_running_attempts() == []
        dependent = daemon.task_source.get(dependent_cid)
        assert dependent is not None
        assert dependent.status == "ready"
        assert dependent.revision == 1

        projection = daemon.coordinator.coordination_registry_projection()
        assert {
            (edge["task_cid"], edge["dependency_task_cid"])
            for edge in projection["dependency_edges"]
        } >= {(dependent_cid, dependency_cid)}
        rejected_claims = [
            claim
            for claim in projection["task_claims"]
            if claim["task_cid"] == dependent_cid
        ]
        assert len(rejected_claims) == 1
        assert rejected_claims[0]["state"] == "released"

        assert daemon.sync_ready_tasks_into_coordination() == [dependency_cid]
        blocked = daemon.coordinator.claimability(dependent_cid)
        assert blocked["claimable"] is False
        assert blocked["blocked_dependency_task_cids"] == [dependency_cid]
        assert daemon.claim_next(exclude_task_cids=(dependency_cid,)) is None
    finally:
        daemon.close()


def test_fenced_retry_cannot_bypass_dependency_reopen_after_local_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = {"ms": 1_000}
    daemon = _open_daemon(
        tmp_path,
        session="session:fenced-retry-dependency",
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
        strict_task_sharding=True,
    )
    dependency_cid = "task:cid:retry-dependency"
    dependent_cid = "task:cid:retry-dependent"
    try:
        assert daemon._automatic_claim_forbidden(object()) is False
        assert daemon._shared_claim_binding_for_this_owner(object()) is None
        daemon.materialize_population(
            {
                "repository_tree_id": "tree:fenced-retry-dependency",
                "objectives": [
                    {
                        "objective_id": "objective:fenced-retry-dependency",
                        "goal_cid": "goal:cid:root",
                        "status": "open",
                    }
                ],
                "tasks": [
                    {
                        "task_cid": dependency_cid,
                        "task_id": "RETRY-DEP",
                        "goal_cid": "goal:cid:root",
                        "status": "completed",
                        "ordinal": 1,
                    },
                    {
                        "task_cid": dependent_cid,
                        "task_id": "RETRY-WORK",
                        "goal_cid": "goal:cid:root",
                        "status": "ready",
                        "ordinal": 2,
                        "dependencies": [dependency_cid],
                    },
                ],
            }
        )
        first_attempt = daemon.claim_next(exclude_task_cids=(dependency_cid,))
        assert first_attempt is not None
        assert first_attempt.task_cid == dependent_cid
        current = daemon.task_source.get(dependent_cid)
        assert current is not None
        assert current.status == "in_progress"
        assert current.revision == 2

        now["ms"] = 7_000
        real_claim_ready_task = daemon.coordinator.claim_ready_task
        reopened = False

        def retry_then_reopen_dependency(**kwargs: object) -> object:
            nonlocal reopened
            claim = real_claim_ready_task(**kwargs)
            if claim is not None and not reopened:
                assert claim.task_cid == dependent_cid
                assert claim.attempt_number == 2
                dependency = daemon.task_source.get(dependency_cid)
                assert dependency is not None
                daemon.task_source.compare_and_set_status(
                    dependency_cid,
                    expected_revision=int(dependency.revision),
                    status="ready",
                )
                reopened = True
            return claim

        monkeypatch.setattr(
            daemon.coordinator,
            "claim_ready_task",
            retry_then_reopen_dependency,
        )
        assert daemon.claim_next(exclude_task_cids=(dependency_cid,)) is None
        # After lease expiry the dependent stays in_progress.  A replacement
        # claim is fail-closed if the coordinator no longer projects the task
        # as ready; either way the original attempt remains the only cursor.
        assert [attempt.attempt_id for attempt in daemon.list_running_attempts()] == [
            first_attempt.attempt_id
        ]
        unchanged = daemon.task_source.get(dependent_cid)
        assert unchanged is not None
        assert unchanged.status == "in_progress"
        assert unchanged.revision == 2

        projection = daemon.coordinator.coordination_registry_projection()
        assert {
            (edge["task_cid"], edge["dependency_task_cid"])
            for edge in projection["dependency_edges"]
        } >= {(dependent_cid, dependency_cid)}
        retry_claims = [
            claim
            for claim in projection["task_claims"]
            if claim["task_cid"] == dependent_cid
            and int(claim["attempt_number"]) == 2
        ]
        assert len(retry_claims) <= 1
        if retry_claims:
            assert retry_claims[0]["state"] == "released"

        evidence_digest = "sha256:" + "d" * 64
        daemon.task_source.record_validation_result(
            task_cid=dependency_cid,
            outcome="passed",
            evidence_digest=evidence_digest,
            argv=("dependency-recompleted",),
        )
        reopened_dependency = daemon.task_source.get(dependency_cid)
        assert reopened_dependency is not None
        daemon.task_source.compare_and_set_status(
            dependency_cid,
            expected_revision=int(reopened_dependency.revision),
            status="completed",
            evidence_digests=(evidence_digest,),
        )
        fenced_receipts: list[dict[str, object]] = []
        shared_cas = daemon._cas_task_status_database

        def capture_fenced_receipt(
            task_cid: str,
            *,
            expected_revision: int,
            new_status: str,
            receipt: object,
            evidence_digests: object = None,
        ) -> object:
            assert isinstance(receipt, dict)
            fenced_receipts.append(dict(receipt))
            return shared_cas(
                task_cid,
                expected_revision=expected_revision,
                new_status=new_status,
                receipt=receipt,
                evidence_digests=evidence_digests,
            )

        monkeypatch.setattr(
            daemon,
            "_cas_task_status_database",
            capture_fenced_receipt,
        )
        converged_retry = daemon.claim_next(exclude_task_cids=(dependency_cid,))
        if converged_retry is not None:
            assert converged_retry.task_cid == dependent_cid
            assert converged_retry.attempt_number == (
                3 if retry_claims else 2
            )
            assert len(fenced_receipts) == 1
            refreshed_receipt = fenced_receipts[0]
            assert {
                name: refreshed_receipt[name]
                for name in (
                    "attempt_id",
                    "claim_id",
                    "attempt_number",
                    "lease_id",
                    "owner_session_id",
                    "fencing_token",
                    "fence_epoch",
                )
            } == {
                "attempt_id": converged_retry.attempt_id,
                "claim_id": converged_retry.claim_id,
                "attempt_number": converged_retry.attempt_number,
                "lease_id": converged_retry.lease_id,
                "owner_session_id": converged_retry.owner_session_id,
                "fencing_token": converged_retry.fencing_token,
                "fence_epoch": converged_retry.fence_epoch,
            }
            assert refreshed_receipt["claimed_from_revision"] == 2
            current = daemon.task_source.get(converged_retry.task_cid)
            assert current is not None
            projection = daemon.coordinator.coordination_registry_projection()
            local_task = next(
                row
                for row in projection["tasks"]
                if row["task_cid"] == converged_retry.task_cid
            )
            assert daemon._shared_retry_binding_matches_attempt(
                current,
                converged_retry,
                local_task_body=local_task["body"],
                local_projection=projection,
            ) is True
            assert daemon._strict_resume_admission_result(converged_retry) is None
        else:
            # Lease expiry without a replacement claim still fails closed:
            # the original in-progress cursor is the only live attempt.
            assert unchanged.status == "in_progress"
            assert fenced_receipts == []
    finally:
        daemon.close()


def test_strict_shards_claim_only_home_lane_tasks(tmp_path: Path) -> None:
    seed = _open_daemon(tmp_path, session="session:seed")
    try:
        seed.materialize_population(_population(8))
    finally:
        seed.close()

    claimed: dict[int, str] = {}
    for index in range(4):
        daemon = _open_daemon(
            tmp_path,
            session=f"session:shard-{index}",
            task_shard_count=4,
            task_shard_index=index,
            strict_task_sharding=True,
            task_prefix="DQP-T",
        )
        try:
            attempt = daemon.claim_next()
            assert attempt is not None, f"shard {index} found no home-lane work"
            alias = str(attempt.task_alias or "")
            home = daemon._task_home_shard_index(alias)
            assert home == index, f"{alias} home={home} claimed by shard {index}"
            claimed[index] = alias
        finally:
            daemon.close()

    assert len(set(claimed.values())) == 4


def test_no_markdown_status_update_under_database_authority(tmp_path: Path) -> None:
    markdown = tmp_path / "tasks.md"
    markdown.write_text(
        "# Tasks\n\n## DQP-T001 Work\n\n- Status: todo\n",
        encoding="utf-8",
    )
    before = markdown.read_text(encoding="utf-8")
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        session="session:md",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        markdown_path=markdown,
    )
    try:
        daemon.materialize_population(_population(1))
        result = daemon.run_once()
        assert result["unchanged"] is False
        assert result["markdown_status_writes"] == 0
        assert provider_calls == ["task:cid:001"]
        assert effect_calls == ["task:cid:001"]
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        assert task.status == "completed"
        with pytest.raises(DatabaseImplementationAuthorityError, match="Markdown"):
            daemon.write_markdown_task_status("DQP-T001", "completed")
        assert markdown.read_text(encoding="utf-8") == before
        assert "- Status: completed" not in markdown.read_text(encoding="utf-8")
    finally:
        daemon.close()


def test_json_projections_can_be_absent(tmp_path: Path) -> None:
    daemon = open_database_implementation_daemon(
        tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        owner_session_id="session:proj",
        authority_mode="embedded",
        task_source_kind="duckdb",
    )
    try:
        assert daemon.projections_required() is False
        assert daemon.state_path is None
        assert daemon.strategy_path is None
        assert daemon.events_path is None
        assert daemon.pid_path is None
        assert daemon.queue_path is None
        # No projection files created by open/materialize/run.
        daemon.materialize_population(_population(1))
        daemon.run_once()
        assert not (tmp_path / "task_state.json").exists()
        assert not (tmp_path / "events.jsonl").exists()
        assert not (tmp_path / "task_queue.json").exists()
        assert not list(tmp_path.glob("*.pid"))
    finally:
        daemon.close()


def test_database_task_state_compatibility_projection_marks_exact_idle_completion(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(tmp_path, session="session:terminal-projection")
    state_path = tmp_path / "state" / "lane_task_state.json"
    try:
        daemon.materialize_population(_population(1))

        active_pass = daemon.run_once()
        active_projection = daemon.materialize_task_state_compatibility_projection(
            state_path=state_path,
            pass_result=active_pass,
        )
        assert active_projection["projection_complete"] is True
        assert active_projection["task_count"] == 1
        assert active_projection["completed_count"] == 1
        assert active_projection["implementation_in_progress"] is True

        idle_pass = daemon.run_once()
        assert idle_pass["selection_idle_reason"] == "no_ready_tasks"
        assert idle_pass["unchanged"] is True
        projection = daemon.materialize_task_state_compatibility_projection(
            state_path=state_path,
            pass_result=idle_pass,
        )

        assert projection["projection_complete"] is True
        assert projection["authority"] == "non_authoritative_compatibility_projection"
        assert projection["projection_authority"] is False
        assert projection["authoritative_task_store"] == "duckdb"
        assert projection["task_count"] == 1
        assert projection["completed_count"] == 1
        assert projection["eligible_ready_count"] == 0
        assert projection["blocked_count"] == 0
        assert projection["external_reserved_count"] == 0
        assert projection["active_task_id"] == ""
        assert projection["implementation_in_progress"] is False
        assert projection["task_statuses"] == {"DQP-T001": "completed"}
        persisted = json.loads(state_path.read_text(encoding="utf-8"))
        assert persisted == {
            key: value for key, value in projection.items() if key != "written"
        }
        terminal = terminal_task_state_fields(
            SupervisorTrack(
                name="lane",
                script_path=tmp_path / "unused.py",
                log_path=tmp_path / "lane.log",
                supervisor_pid_path=state_path.parent / "lane_supervisor.pid",
                daemon_pid_path=state_path.parent / "lane_daemon.pid",
            ),
            repo_root=tmp_path,
            fresh_after_epoch_seconds=state_path.stat().st_mtime - 1.0,
        )
        assert terminal["task_state_status"] == "terminal"
        assert terminal["terminal_quiescent"] is True
        assert terminal["task_state_projection_valid"] is True

        persisted["projection_authority"] = True
        state_path.write_text(json.dumps(persisted), encoding="utf-8")
        rejected = terminal_task_state_fields(
            SupervisorTrack(
                name="lane",
                script_path=tmp_path / "unused.py",
                log_path=tmp_path / "lane.log",
                supervisor_pid_path=state_path.parent / "lane_supervisor.pid",
                daemon_pid_path=state_path.parent / "lane_daemon.pid",
            ),
            repo_root=tmp_path,
            fresh_after_epoch_seconds=state_path.stat().st_mtime - 1.0,
        )
        assert rejected["task_state_projection_valid"] is False
        assert rejected["terminal_quiescent"] is False
    finally:
        daemon.close()


def test_database_task_state_compatibility_projection_overwrites_stale_terminal_on_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(tmp_path, session="session:terminal-projection-error")
    state_path = tmp_path / "state" / "lane_task_state.json"
    try:
        daemon.materialize_population(_population(1))
        daemon.run_once()
        idle_pass = daemon.run_once()
        terminal = daemon.materialize_task_state_compatibility_projection(
            state_path=state_path,
            pass_result=idle_pass,
        )
        assert terminal["implementation_in_progress"] is False

        def fail_snapshot() -> object:
            raise RuntimeError("source unavailable")

        monkeypatch.setattr(daemon.task_source, "snapshot", fail_snapshot)
        failed = daemon.materialize_task_state_compatibility_projection(
            state_path=state_path,
            pass_result=idle_pass,
        )

        assert failed["projection_complete"] is False
        assert failed["projection_error"] == "projection_refresh_failed"
        assert failed["implementation_in_progress"] is True
        persisted = json.loads(state_path.read_text(encoding="utf-8"))
        assert persisted["task_count"] == 0
        assert persisted["completed_count"] == 0
        assert persisted["implementation_in_progress"] is True
    finally:
        daemon.close()


def test_database_task_state_compatibility_projection_keeps_skipped_board_nonterminal(
    tmp_path: Path,
) -> None:
    """VRIF completion requires success; skipped is terminal but not complete."""

    daemon = _open_daemon(tmp_path, session="session:terminal-projection-skipped")
    state_path = tmp_path / "state" / "lane_task_state.json"
    try:
        population = _population(1)
        tasks = population["tasks"]
        assert isinstance(tasks, list)
        tasks[0]["status"] = "skipped"
        daemon.materialize_population(population)

        projection = daemon.materialize_task_state_compatibility_projection(
            state_path=state_path,
            pass_result={
                "unchanged": True,
                "write_count": 0,
                "active_task_id": "",
                "selection_idle_reason": "no_ready_tasks",
            },
        )

        assert projection["task_count"] == 1
        assert projection["completed_count"] == 0
        assert projection["task_statuses"] == {"DQP-T001": "skipped"}
        terminal = terminal_task_state_fields(
            SupervisorTrack(
                name="lane",
                script_path=tmp_path / "unused.py",
                log_path=tmp_path / "lane.log",
                supervisor_pid_path=state_path.parent / "lane_supervisor.pid",
                daemon_pid_path=state_path.parent / "lane_daemon.pid",
            ),
            repo_root=tmp_path,
            fresh_after_epoch_seconds=state_path.stat().st_mtime - 1.0,
        )
        assert terminal["task_state_projection_valid"] is True
        assert terminal["terminal_quiescent"] is False
    finally:
        daemon.close()
def test_datasets_authoritative_open_requires_preinstalled_operational_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION",
        "datasets-authoritative-operational-v1",
    )
    control_path = tmp_path / "missing-control.duckdb"
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="preinstalled by the trusted materializer",
    ):
        DatabaseImplementationDaemon(
            database_path=control_path,
            coordination_path=tmp_path / "coordination.duckdb",
            execution_path=tmp_path / "execution.duckdb",
            authority_mode="embedded",
            task_source_kind="duckdb",
        )
    assert not control_path.exists()
    assert not (tmp_path / "execution.duckdb").exists()


def test_datasets_authoritative_open_rejects_full_control_plane_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_path = tmp_path / "full-control.duckdb"
    install_control_plane_schema(control_path)
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION",
        "datasets-authoritative-operational-v1",
    )
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="not the verified datasets-authoritative operational profile",
    ):
        DatabaseImplementationDaemon(
            database_path=control_path,
            coordination_path=tmp_path / "coordination.duckdb",
            execution_path=tmp_path / "execution.duckdb",
            authority_mode="embedded",
            task_source_kind="duckdb",
        )
    with open_duckdb_connection(control_path) as connection:
        names = {str(row[0]) for row in connection.execute("SHOW TABLES").fetchall()}
    assert "proof_obligations" in names
    assert not (tmp_path / "execution.duckdb").exists()


def test_datasets_authoritative_open_rejects_tampered_operational_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_path = tmp_path / "tampered-control.duckdb"
    install_datasets_authoritative_operational_schema(control_path)
    with open_duckdb_connection(control_path) as connection:
        connection.execute(
            "UPDATE schema_migrations SET checksum = 'sha256:tampered'"
        )
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION",
        "datasets-authoritative-operational-v1",
    )
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="not the verified datasets-authoritative operational profile",
    ):
        DatabaseImplementationDaemon(
            database_path=control_path,
            coordination_path=tmp_path / "coordination.duckdb",
            execution_path=tmp_path / "execution.duckdb",
            authority_mode="embedded",
            task_source_kind="duckdb",
        )
    assert not (tmp_path / "execution.duckdb").exists()


def test_datasets_authoritative_open_verifies_existing_operational_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_path = tmp_path / "operational-control.duckdb"
    install_datasets_authoritative_operational_schema(control_path)
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION",
        "datasets-authoritative-operational-v1",
    )
    daemon = DatabaseImplementationDaemon(
        database_path=control_path,
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        authority_mode="embedded",
        task_source_kind="duckdb",
    )
    try:
        evidence = dict(daemon.control_schema_evidence)
        assert evidence["state_schema_revision"] == (
            "datasets-authoritative-operational-v1"
        )
        assert evidence["verified"] is True
        assert evidence["profile_id"]
        assert evidence["schema_fingerprint"]
        daemon.materialize_population(_population(1))
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        assert task.status == "ready"
    finally:
        daemon.close()


def test_crash_restart_resumes_without_duplicating_provider_or_effect(
    tmp_path: Path,
) -> None:
    provider_calls: list[str] = []
    effect_calls: list[str] = []

    first = _open_daemon(
        tmp_path,
        session="session:resume",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        attempt = first.commit_phase(attempt, "context")
        attempt, provider_result, duplicated = first.run_provider(attempt)
        assert duplicated is False
        assert provider_calls == ["task:cid:001"]
        assert attempt.committed_phase == ATTEMPT_PHASE_PROVIDER
        # Crash boundary: process dies after provider commits, before effect.
        assert effect_calls == []
        attempt_id = attempt.attempt_id
    finally:
        first.close()

    second = _open_daemon(
        tmp_path,
        session="session:resume",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        running = second.list_running_attempts()
        assert len(running) == 1
        assert running[0].attempt_id == attempt_id
        assert running[0].committed_phase == ATTEMPT_PHASE_PROVIDER
        result = second.resume_attempt(running[0])
        assert result["resumed"] is True
        assert result["provider_duplicated"] is True
        assert result["effect_duplicated"] is False
        assert provider_calls == ["task:cid:001"]
        assert effect_calls == ["task:cid:001"]
        assert result["committed_phase"] == ATTEMPT_PHASE_COMPLETE
        assert result["status"] == "succeeded"
        task = second.task_source.get("task:cid:001")
        assert task is not None
        assert task.status == "completed"

        # Second resume of a finished attempt is a no-op for provider/effect.
        finished = second.get_attempt(attempt_id)
        assert finished is not None
        again = second.resume_attempt(finished)
        assert again["resumed"] is False
        assert provider_calls == ["task:cid:001"]
        assert effect_calls == ["task:cid:001"]
    finally:
        second.close()


def test_implicit_embedded_owner_is_store_scoped_and_restart_stable(
    tmp_path: Path,
) -> None:
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    first = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        first_owner = first.owner_session_id
        assert first_owner.startswith("embedded-store:")
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        attempt = first.commit_phase(attempt, "context")
        attempt, _, duplicated = first.run_provider(attempt)
        assert duplicated is False
        assert provider_calls == [attempt.task_cid]
    finally:
        first.close()

    second = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        assert second.owner_session_id == first_owner
        result = second.run_once()
        assert result["implementation_result"]["provider_duplicated"] is True
        assert result["implementation_result"]["status"] == "succeeded"
        assert provider_calls == [attempt.task_cid]
        assert effect_calls == [attempt.task_cid]
    finally:
        second.close()


def test_implicit_embedded_owner_is_distinct_for_different_stores(
    tmp_path: Path,
) -> None:
    first = _open_daemon(tmp_path / "first")
    second = _open_daemon(tmp_path / "second")
    try:
        assert first.owner_session_id.startswith("embedded-store:")
        assert second.owner_session_id.startswith("embedded-store:")
        assert first.owner_session_id != second.owner_session_id
    finally:
        second.close()
        first.close()


def test_embedded_writer_lock_rejects_a_concurrent_same_store_opener(
    tmp_path: Path,
) -> None:
    first = _open_daemon(tmp_path)
    try:
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="active database writer",
        ):
            _open_daemon(tmp_path)
        first.materialize_population(_population(1))
        assert first.claim_next() is not None
    finally:
        first.close()

    replacement = _open_daemon(tmp_path)
    try:
        assert replacement.owner_session_id == first.owner_session_id
    finally:
        replacement.close()


def test_effect_phase_resume_skips_both_provider_and_effect(tmp_path: Path) -> None:
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    first = _open_daemon(
        tmp_path,
        session="session:effect",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        attempt = first.commit_phase(attempt, "context")
        attempt, provider_result, _ = first.run_provider(attempt)
        attempt, effect_result, _ = first.run_effect(attempt, provider_result)
        assert attempt.committed_phase == ATTEMPT_PHASE_EFFECT
        assert provider_calls == ["task:cid:001"]
        assert effect_calls == ["task:cid:001"]
        attempt_id = attempt.attempt_id
    finally:
        first.close()

    second = _open_daemon(
        tmp_path,
        session="session:effect",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        attempt = second.get_attempt(attempt_id)
        assert attempt is not None
        result = second.resume_attempt(attempt)
        assert result["provider_duplicated"] is True
        assert result["effect_duplicated"] is True
        assert provider_calls == ["task:cid:001"]
        assert effect_calls == ["task:cid:001"]
        assert result["status"] == "succeeded"
    finally:
        second.close()


def test_vrif_cached_effect_resume_rechecks_current_semantic_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, baseline_commit, baseline_tree = _git_recovery_repo(tmp_path)
    implementation_commit, _ = _git_commit(
        repo,
        name="semantic-candidate.txt",
        content="candidate\n",
    )
    bridge = object.__new__(DatabasePortalExecutionBridge)
    bridge.repository_root = repo
    reject = {"enabled": False}
    semantic_calls: list[dict[str, object]] = []

    def semantic_acceptance(**kwargs: object) -> None:
        semantic_calls.append(dict(kwargs))
        if reject["enabled"]:
            raise DatabasePortalBridgeError(
                "VRIF-030 artifacts differ from the owner-exact benchmark contract"
            )

    monkeypatch.setattr(
        bridge,
        "_verify_vrif_semantic_acceptance",
        semantic_acceptance,
    )

    def legacy_effect(
        attempt: DatabaseTaskAttempt,
        _provider_result: dict[str, object],
    ) -> dict[str, object]:
        return _legacy_portal_effect(
            attempt,
            baseline_commit=baseline_commit,
            baseline_tree=baseline_tree,
            implementation_commit=implementation_commit,
        )

    first = _open_daemon(
        tmp_path,
        session="session:vrif-effect-replay",
        effect_fn=legacy_effect,
        validation_fn=bridge.validate_effect,
    )
    try:
        first.materialize_population(_single_task_population("VRIF-030"))
        attempt = first.claim_next()
        assert attempt is not None
        attempt = first.commit_phase(attempt, ATTEMPT_PHASE_CONTEXT)
        attempt, provider_result, _ = first.run_provider(attempt)
        attempt, cached_effect, duplicated = first.run_effect(
            attempt,
            provider_result,
        )
        assert duplicated is False
        assert cached_effect["status"] == "applied"
        assert attempt.committed_phase == ATTEMPT_PHASE_EFFECT
        attempt_id = attempt.attempt_id
    finally:
        first.close()

    reject["enabled"] = True
    semantic_calls.clear()
    resumed = _open_daemon(
        tmp_path,
        session="session:vrif-effect-replay",
        effect_fn=legacy_effect,
        validation_fn=bridge.validate_effect,
    )
    try:
        cached_attempt = resumed.get_attempt(attempt_id)
        assert cached_attempt is not None
        with pytest.raises(
            DatabasePortalBridgeError,
            match="owner-exact benchmark contract",
        ):
            resumed.resume_attempt(cached_attempt)

        assert len(semantic_calls) == 1
        assert semantic_calls[0]["baseline_commit"] == baseline_commit
        assert semantic_calls[0]["baseline_tree"] == baseline_tree
        assert semantic_calls[0]["implementation_commit"] == implementation_commit
        after = resumed.get_attempt(attempt_id)
        assert after is not None
        assert after.committed_phase == ATTEMPT_PHASE_EFFECT
        assert after.status == "running"
        task = resumed.task_source.get(after.task_cid)
        assert task is not None
        assert task.status == "in_progress"
    finally:
        resumed.close()


def test_vrif_committed_validation_replay_rechecks_semantic_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, baseline_commit, baseline_tree = _git_recovery_repo(tmp_path)
    implementation_commit, _ = _git_commit(
        repo,
        name="legacy-validation-candidate.txt",
        content="candidate\n",
    )
    bridge = object.__new__(DatabasePortalExecutionBridge)
    bridge.repository_root = repo
    reject = {"enabled": False}
    semantic_calls: list[dict[str, object]] = []

    def semantic_acceptance(**kwargs: object) -> None:
        semantic_calls.append(dict(kwargs))
        if reject["enabled"]:
            raise DatabasePortalBridgeError(
                "VRIF-030 artifacts differ from the owner-exact benchmark contract"
            )

    monkeypatch.setattr(
        bridge,
        "_verify_vrif_semantic_acceptance",
        semantic_acceptance,
    )

    def legacy_effect(
        attempt: DatabaseTaskAttempt,
        _provider_result: dict[str, object],
    ) -> dict[str, object]:
        return _legacy_portal_effect(
            attempt,
            baseline_commit=baseline_commit,
            baseline_tree=baseline_tree,
            implementation_commit=implementation_commit,
        )

    first = _open_daemon(
        tmp_path,
        session="session:vrif-validation-replay",
        effect_fn=legacy_effect,
        validation_fn=bridge.validate_effect,
    )
    try:
        first.materialize_population(_single_task_population("VRIF-030"))
        attempt = first.claim_next()
        assert attempt is not None
        attempt = first.commit_phase(attempt, ATTEMPT_PHASE_CONTEXT)
        attempt, provider_result, _ = first.run_provider(attempt)
        attempt, cached_effect, _ = first.run_effect(attempt, provider_result)
        stored_validation = dict(bridge.validate_effect(attempt, cached_effect))
        attempt = first.commit_phase(
            attempt,
            ATTEMPT_PHASE_VALIDATION,
            body=stored_validation,
        )
        assert attempt.committed_phase == ATTEMPT_PHASE_VALIDATION
        attempt_id = attempt.attempt_id
    finally:
        first.close()

    assert len(semantic_calls) == 1
    reject["enabled"] = True
    semantic_calls.clear()
    resumed = _open_daemon(
        tmp_path,
        session="session:vrif-validation-replay",
        effect_fn=legacy_effect,
        validation_fn=bridge.validate_effect,
    )
    try:
        cached_attempt = resumed.get_attempt(attempt_id)
        assert cached_attempt is not None
        with pytest.raises(
            DatabasePortalBridgeError,
            match="owner-exact benchmark contract",
        ):
            resumed.resume_attempt(cached_attempt)

        assert len(semantic_calls) == 1
        after = resumed.get_attempt(attempt_id)
        assert after is not None
        assert after.committed_phase == ATTEMPT_PHASE_VALIDATION
        assert after.status == "running"
        task = resumed.task_source.get(after.task_cid)
        assert task is not None
        assert task.status == "in_progress"
    finally:
        resumed.close()


def test_committed_validation_replay_rejects_fresh_passing_drift(
    tmp_path: Path,
) -> None:
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    stored_validation = {
        "outcome": "passed",
        "evidence_digest": "sha256:" + "a" * 64,
        "argv": ["stored-validation"],
    }
    first = _open_daemon(
        tmp_path,
        session="session:validation-drift",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        attempt = first.commit_phase(attempt, ATTEMPT_PHASE_CONTEXT)
        attempt, provider_result, _ = first.run_provider(attempt)
        attempt, _effect_result, _ = first.run_effect(attempt, provider_result)
        attempt = first.commit_phase(
            attempt,
            ATTEMPT_PHASE_VALIDATION,
            body=stored_validation,
        )
        attempt_id = attempt.attempt_id
    finally:
        first.close()

    def drifted_validation(
        _attempt: DatabaseTaskAttempt,
        _effect_result: dict[str, object],
    ) -> dict[str, object]:
        return {
            **stored_validation,
            "argv": ["different-current-validation"],
        }

    resumed = _open_daemon(
        tmp_path,
        session="session:validation-drift",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        validation_fn=drifted_validation,
    )
    try:
        cached_attempt = resumed.get_attempt(attempt_id)
        assert cached_attempt is not None
        with pytest.raises(DatabaseImplementationAuthorityError):
            resumed.resume_attempt(cached_attempt)

        assert provider_calls == ["task:cid:001"]
        assert effect_calls == ["task:cid:001"]
        after = resumed.get_attempt(attempt_id)
        assert after is not None
        assert after.committed_phase == ATTEMPT_PHASE_VALIDATION
        assert after.status == "running"
        task = resumed.task_source.get(after.task_cid)
        assert task is not None
        assert task.status == "in_progress"
    finally:
        resumed.close()


def test_non_vrif_committed_validation_replay_remains_idempotent(
    tmp_path: Path,
) -> None:
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    validation_calls: list[str] = []
    stored_validation = {
        "outcome": "passed",
        "evidence_digest": "sha256:" + "b" * 64,
        "argv": ["stable-current-validation"],
    }

    def stable_validation(
        attempt: DatabaseTaskAttempt,
        _effect_result: dict[str, object],
    ) -> dict[str, object]:
        validation_calls.append(attempt.task_cid)
        return dict(stored_validation)

    first = _open_daemon(
        tmp_path,
        session="session:stable-validation-replay",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        validation_fn=stable_validation,
    )
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        attempt = first.commit_phase(attempt, ATTEMPT_PHASE_CONTEXT)
        attempt, provider_result, _ = first.run_provider(attempt)
        attempt, _effect_result, _ = first.run_effect(attempt, provider_result)
        attempt = first.commit_phase(
            attempt,
            ATTEMPT_PHASE_VALIDATION,
            body=stored_validation,
        )
        attempt_id = attempt.attempt_id
    finally:
        first.close()

    resumed = _open_daemon(
        tmp_path,
        session="session:stable-validation-replay",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        validation_fn=stable_validation,
    )
    try:
        cached_attempt = resumed.get_attempt(attempt_id)
        assert cached_attempt is not None
        result = resumed.resume_attempt(cached_attempt)

        assert result["resumed"] is True
        assert result["provider_duplicated"] is True
        assert result["effect_duplicated"] is True
        assert result["status"] == "succeeded"
        assert result["committed_phase"] == ATTEMPT_PHASE_COMPLETE
        assert validation_calls == ["task:cid:001"]
        assert provider_calls == ["task:cid:001"]
        assert effect_calls == ["task:cid:001"]
        task = resumed.task_source.get(cached_attempt.task_cid)
        assert task is not None
        assert task.status == "completed"
    finally:
        resumed.close()


def test_provider_heartbeat_renews_exact_task_claim(tmp_path: Path) -> None:
    holder: dict[str, DatabaseImplementationDaemon] = {}
    observed_revisions: list[int] = []

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        daemon = holder["daemon"]
        initial = daemon.coordinator.get_task_claim(attempt.claim_id)
        assert initial is not None
        observed_revisions.append(int(initial.revision))
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            current = daemon.coordinator.get_task_claim(attempt.claim_id)
            assert current is not None
            if int(current.revision) > int(initial.revision):
                observed_revisions.append(int(current.revision))
                break
            time.sleep(0.005)
        assert len(observed_revisions) == 2, "background lease renewal did not run"
        return {"status": "ok", "accepted": True, "task_cid": attempt.task_cid}

    daemon = _open_daemon(
        tmp_path,
        session="session:heartbeat",
        provider_fn=provider,
        lease_ms=5_000,
    )
    holder["daemon"] = daemon
    daemon._lease_heartbeat_interval_seconds = 0.01
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        attempt = daemon.commit_phase(attempt, "context")
        updated, _, duplicated = daemon.run_provider(attempt)
        assert duplicated is False
        assert updated.committed_phase == ATTEMPT_PHASE_PROVIDER
        assert observed_revisions[1] > observed_revisions[0]
    finally:
        daemon.close()


def test_consumed_attempt_terminal_replay_preserves_exact_legacy_failed_phase(
    tmp_path: Path,
) -> None:
    holder: dict[str, DatabaseImplementationDaemon] = {}

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        daemon = holder["daemon"]
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        receipt = _consumed_attempt_retry_receipt(
            daemon,
            attempt,
            source_task_revision=task.revision,
        )
        raise DatabasePortalConsumedAttemptTerminal(receipt)

    daemon = _open_daemon(
        tmp_path,
        session="session:consumed-terminal-replay",
        provider_fn=provider,
        max_task_attempts=3,
    )
    holder["daemon"] = daemon
    try:
        daemon.materialize_population(_population(1))
        result = daemon.run_once()
        attempt = daemon.get_attempt(result["attempt_id"])
        assert attempt is not None
        assert daemon.phase_history(attempt.attempt_id)[-1]["body"] == {
            "attempt_consumed": "unknown",
            "backoff_seconds": 0,
            "deferred": False,
            "portal_retryable_failure": False,
            "portal_terminal_failure": True,
            "provider_dispatched": "unknown",
            "reason": "portal_provider_failed",
            "typed_deferral_slot_consumed": "unknown",
        }
        assert result["implementation_result"]["portal_terminal_failure"] is True
        assert result["implementation_result"]["reason"] == "portal_provider_failed"
    finally:
        daemon.close()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("implementation_returncode", 17),
        ("baseline_commit", "5" * 64),
    ),
)
def test_consumed_attempt_evidence_mirrors_bridge_exact_predicates(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session=f"session:consumed-evidence-{field}",
        max_task_attempts=3,
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        receipt = _consumed_attempt_retry_receipt(
            daemon,
            attempt,
            source_task_revision=task.revision,
        )
        assert daemon._verified_consumed_attempt_retry_receipt(
            attempt,
            receipt,
        ) == receipt
        receipt[field] = value
        receipt.pop("receipt_id")
        receipt["receipt_id"] = daemon._database_portal_evidence_digest(
            receipt
        )
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="consumed-attempt retry evidence failed verification",
        ):
            daemon._verified_consumed_attempt_retry_receipt(
                attempt,
                receipt,
            )
    finally:
        daemon.close()


def test_protected_preservation_is_distinct_and_crosses_lanes(
    tmp_path: Path,
) -> None:
    holder: dict[str, DatabaseImplementationDaemon] = {}
    provider_calls: list[str] = []

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_calls.append(attempt.attempt_id)
        task = holder["daemon"].task_source.get(attempt.task_cid)
        assert task is not None
        receipt = _protected_preservation_receipt(
            holder["daemon"],
            attempt,
            source_task_revision=task.revision,
        )
        raise DatabasePortalProtectedPathPreserved(receipt)

    first = _open_daemon(
        tmp_path,
        session="session:protected-preservation-lane-a",
        provider_fn=provider,
        max_task_attempts=3,
        lease_ms=5_000,
        clock_ms=lambda: 1_000,
        lane="protected-a",
    )
    holder["daemon"] = first
    try:
        first.materialize_population(_population(1))
        result = first.run_once()
        implementation = result["implementation_result"]
        assert implementation["portal_retryable_failure"] is True
        assert implementation["portal_terminal_failure"] is False
        assert implementation["deferred"] is False
        assert implementation["protected_candidate_preserved"] is True
        assert implementation["attempt_consumed"] is False
        assert implementation["provider_dispatched"] is True
        assert implementation["typed_deferral_slot_consumed"] is False
        assert implementation["backoff_seconds"] == 0
        assert implementation["retry_budget_exhausted"] is False
        assert len(provider_calls) == 1

        source = first.get_attempt(result["attempt_id"])
        assert source is not None
        failed = first.phase_history(source.attempt_id)[-1]["body"]
        assert failed["protected_candidate_preserved"] is True
        assert failed["typed_protected_preservation"][
            "preserved_commit"
        ] == "c" * 40
        evidence = first._terminal_retry_evidence(source)
        assert evidence is not None
        assert evidence["typed_deferral_budget"] is None
        seed = evidence["typed_protected_preservation"]
        assert seed["attempt_consumed"] is False
        assert seed["provider_dispatched"] is True
        retrying = first.task_source.get(source.task_cid)
        assert retrying is not None
        assert retrying.status == "retrying"
        control = retrying.body["completion_receipt"]
        assert control["operation"] == (
            "database_portal_protected_preservation_retry"
        )
        assert control["protected_preservation_seed"] == seed
        assert "typed_deferral" not in failed
        assert "consumed_attempt_retry_seed" not in control
    finally:
        first.close()

    second = _open_daemon(
        tmp_path,
        session="session:protected-preservation-lane-b",
        max_task_attempts=3,
        lease_ms=5_000,
        clock_ms=lambda: 7_001,
        lane="protected-b",
    )
    try:
        assert second.get_attempt(source.attempt_id) is None
        successor = second.claim_next()
        assert successor is not None
        claimed = second.task_source.get(source.task_cid)
        assert claimed is not None
        claim_receipt = claimed.body["completion_receipt"]
        assert claim_receipt["operation"] == "database_claim"
        assert claim_receipt["protected_preservation_seed"] == seed
        assert claim_receipt[
            "protected_preservation_source_attempt_id"
        ] == source.attempt_id
        # Claiming only transfers the immutable candidate.  The outer daemon
        # never reclassifies it as an ordinary provider attempt.
        assert len(provider_calls) == 1
    finally:
        second.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_cid", "task:cid:foreign"),
        ("preserved_commit", "d" * 40),
        ("rescue_branch", "rescue/foreign-protected-path-interrupted"),
        ("protected_paths", ["../implementation_daemon.py"]),
    ],
)
def test_protected_preservation_rejects_rehashed_foreign_evidence(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session=f"session:protected-preservation-forgery-{field}",
        max_task_attempts=3,
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        receipt = _protected_preservation_receipt(
            daemon,
            attempt,
            source_task_revision=task.revision,
        )
        receipt[field] = value
        receipt.pop("receipt_id")
        receipt["receipt_id"] = daemon._database_portal_evidence_digest(
            receipt
        )
        with pytest.raises(DatabaseImplementationAuthorityError):
            daemon._verified_protected_preservation_receipt(
                attempt,
                receipt,
            )
    finally:
        daemon.close()


def test_exact_legacy_protected_preservation_block_recovers_once(
    tmp_path: Path,
) -> None:
    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeError(
            "Portal consumed-attempt retry seed state conflicts with its receipt"
        )

    daemon = _open_daemon(
        tmp_path,
        session="session:protected-preservation-legacy-recovery",
        provider_fn=provider,
        max_task_attempts=3,
    )
    try:
        daemon.materialize_population(_population(1))
        failed_result = daemon.run_once()
        source = daemon.get_attempt(failed_result["attempt_id"])
        assert source is not None
        blocked = daemon.task_source.get(source.task_cid)
        assert blocked is not None
        assert blocked.status == "blocked"
        seed = _protected_preservation_receipt(
            daemon,
            source,
            source_task_revision=blocked.revision - 1,
        )
        daemon.bind_protected_preservation_recovery(
            lambda _attempt: seed
        )

        recovered = daemon.reconcile_terminal_portal_failures()
        assert len(recovered) == 1
        assert recovered[0]["changed"] is True
        assert recovered[0]["status"] == "retrying"
        assert recovered[0]["protected_preservation_evidence"] == seed
        retrying = daemon.task_source.get(source.task_cid)
        assert retrying is not None
        assert retrying.status == "retrying"
        receipt = retrying.body["completion_receipt"]
        assert receipt["operation"] == (
            "database_portal_protected_preservation_retry_recovery"
        )
        assert receipt["protected_preservation_seed"] == seed
        assert daemon.reconcile_terminal_portal_failures() == []
    finally:
        daemon.close()


def test_protected_reconciliation_self_lock_rearms_original_seed_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = {"ms": 1_000}
    provider_calls: list[str] = []
    failure_reasons = iter(
        (
            "portal_provider_failed",
            "Portal consumed-attempt retry seed state conflicts with its receipt",
            "not_attempted",
        )
    )

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_calls.append(attempt.attempt_id)
        raise DatabasePortalBridgeError(next(failure_reasons))

    daemon = _open_daemon(
        tmp_path,
        session="session:protected-reconciliation-self-lock",
        provider_fn=provider,
        max_task_attempts=3,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        persist_terminal = daemon._persist_terminal_portal_failure
        monkeypatch.setattr(
            daemon,
            "_persist_terminal_portal_failure",
            lambda attempt, **_kwargs: {
                "task_cid": attempt.task_cid,
                "attempt_id": attempt.attempt_id,
                "status": "blocked",
                "changed": False,
            },
        )
        predecessor_result = daemon.run_once()
        predecessor = daemon.get_attempt(predecessor_result["attempt_id"])
        assert predecessor is not None
        predecessor_claim = daemon.task_source.get(predecessor.task_cid)
        assert predecessor_claim is not None
        assert predecessor_claim.status == "in_progress"
        predecessor_claim_revision = predecessor_claim.revision
        foreign_terminal = {
            "operation": "database_portal_terminal_failure",
            "attempt_id": "attempt:foreign-stale",
            "attempt_number": 1,
            "claim_id": "claim:foreign-stale",
            "lease_id": "lease:foreign-stale",
            "owner_session_id": "session:foreign-stale",
            "fencing_token": 1,
            "fence_epoch": 1,
            "execution_phase": "failed",
            "execution_revision": 1,
            "execution_finished_at_ms": predecessor.started_at_ms - 1,
            "reason": "portal_provider_failed",
            "retryable": False,
            "coordination": {},
            "control_expected_status": "in_progress",
            "control_expected_revision": predecessor_claim_revision,
        }
        daemon.task_source.compare_and_set_status(
            predecessor.task_cid,
            expected_revision=predecessor_claim_revision,
            status="blocked",
            receipt=foreign_terminal,
        )
        consumed_seed = _consumed_attempt_retry_receipt(
            daemon,
            predecessor,
            source_task_revision=predecessor_claim_revision,
        )
        daemon.bind_superseded_consumed_attempt_recovery(
            lambda _attempt: consumed_seed
        )
        now["ms"] = 7_001
        consumed_recovery = daemon.recover_superseded_consumed_portal_attempt(
            predecessor,
            retry_evidence=consumed_seed,
        )
        assert consumed_recovery["changed"] is True
        monkeypatch.setattr(
            daemon,
            "_persist_terminal_portal_failure",
            persist_terminal,
        )

        source = daemon.claim_next()
        assert source is not None
        assert source.attempt_number == 2
        source_claim = daemon.task_source.get(source.task_cid)
        assert source_claim is not None
        source_claim_receipt = source_claim.body["completion_receipt"]
        assert set(source_claim_receipt) == {
            "operation",
            "claim_id",
            "attempt_id",
            "owner_session_id",
            "lease_id",
            "fencing_token",
            "fence_epoch",
            "attempt_number",
            "claimed_from_revision",
            "consumed_attempt_retry_source_attempt_id",
            "consumed_attempt_retry_seed",
        }
        assert source_claim_receipt[
            "consumed_attempt_retry_source_attempt_id"
        ] == predecessor.attempt_id
        assert source_claim_receipt["consumed_attempt_retry_seed"] == (
            consumed_seed
        )
        source_result = daemon.run_once()
        assert source_result["implementation_result"]["attempt_id"] == (
            source.attempt_id
        )
        source = daemon.get_attempt(source.attempt_id)
        assert source is not None
        source_blocked = daemon.task_source.get(source.task_cid)
        assert source_blocked is not None
        assert source_blocked.status == "blocked"
        preservation_seed = _protected_preservation_receipt(
            daemon,
            source,
            source_task_revision=source_blocked.revision - 1,
        )
        daemon.bind_protected_preservation_recovery(
            lambda _attempt: preservation_seed
        )

        now["ms"] = 13_002
        source_recovery = daemon.reconcile_terminal_portal_failures()
        assert len(source_recovery) == 1
        assert source_recovery[0]["changed"] is True
        source_retry = daemon.task_source.get(source.task_cid)
        assert source_retry is not None
        assert source_retry.status == "retrying"
        assert source_retry.body["completion_receipt"]["operation"] == (
            "database_portal_protected_preservation_retry_recovery"
        )
        source_retry_coordination = source_retry.body["completion_receipt"][
            "coordination"
        ]
        assert set(source_retry_coordination) == {
            "claim_id",
            "attempt_id",
            "attempt_number",
            "lease_state",
            "claim_state",
            "claim_revision",
            "coordination_attempt_status",
            "coordination_attempt_revision",
            "expires_at_ms",
            "observed_at_ms",
            "expired_now",
            "claim_absent",
            "historical_expired",
            "superseded_by_newer_fence",
            "successor",
        }
        assert source_retry_coordination["claim_state"] == "expired"
        assert source_retry_coordination["lease_state"] == "expired"
        assert source_retry_coordination["coordination_attempt_status"] == (
            "expired"
        )
        assert source_retry_coordination["expired_now"] is False
        assert source_retry_coordination["claim_absent"] is False
        assert source_retry_coordination["historical_expired"] is True
        assert source_retry_coordination["superseded_by_newer_fence"] is False
        assert source_retry_coordination["successor"] == {}
        for field, value in (
            ("historical_expired", False),
            ("superseded_by_newer_fence", True),
        ):
            tampered_coordination = dict(source_retry_coordination)
            tampered_coordination[field] = value
            with pytest.raises(DatabaseImplementationConflictError):
                daemon._verified_protected_reconciliation_source_retry_coordination(
                    source,
                    tampered_coordination,
                )

        now["ms"] = 19_003
        daemon.close()
        daemon = _open_daemon(
            tmp_path,
            session="session:protected-reconciliation-target-lane",
            provider_fn=provider,
            max_task_attempts=3,
            lease_ms=5_000,
            clock_ms=lambda: now["ms"],
            lane="protected-target",
        )
        target = daemon.claim_next()
        assert target is not None
        assert target.attempt_number == 1
        assert target.attempt_number < source.attempt_number
        assert target.attempt_id != source.attempt_id
        target_claim = daemon.task_source.get(target.task_cid)
        assert target_claim is not None
        assert target_claim.status == "in_progress"
        assert target_claim.body["completion_receipt"][
            "protected_preservation_seed"
        ] == preservation_seed

        def fail_before_callback_intent(
            attempt: DatabaseTaskAttempt,
            **_kwargs: object,
        ) -> tuple[DatabaseTaskAttempt, Mapping[str, Any], bool]:
            """Model the historical pre-intent provider setup failure."""

            provider(attempt)
            raise AssertionError("provider fixture unexpectedly returned")

        monkeypatch.setattr(daemon, "run_provider", fail_before_callback_intent)
        target_result = daemon.run_once()
        assert target_result["implementation_result"]["attempt_id"] == (
            target.attempt_id
        )
        target = daemon.get_attempt(target.attempt_id)
        assert target is not None
        assert target.status == "failed"
        assert daemon.phase_history(target.attempt_id)[-1]["body"] == {
            "attempt_consumed": "unknown",
            "backoff_seconds": 0,
            "deferred": False,
            "portal_retryable_failure": False,
            "portal_terminal_failure": True,
            "provider_dispatched": "unknown",
            "reason": "not_attempted",
            "typed_deferral_slot_consumed": "unknown",
        }
        provider_count = daemon._require_connection().execute(
            "SELECT COUNT(*) FROM provider_invocations WHERE attempt_id = ?",
            [target.attempt_id],
        ).fetchone()
        effect_count = daemon._require_connection().execute(
            "SELECT COUNT(*) FROM effect_claims WHERE attempt_id = ?",
            [target.attempt_id],
        ).fetchone()
        assert provider_count is not None and int(provider_count[0]) == 0
        assert effect_count is not None and int(effect_count[0]) == 0

        target_blocked = daemon.task_source.get(target.task_cid)
        assert target_blocked is not None
        assert target_blocked.status == "blocked"
        history = daemon.task_source.task_revision_history_projection(
            target.task_cid
        )
        exact_chain = history["revisions"][-5:]
        assert [entry["status"] for entry in exact_chain] == [
            "in_progress",
            "blocked",
            "retrying",
            "in_progress",
            "blocked",
        ]
        assert [
            entry["body"]["completion_receipt"]["operation"]
            for entry in exact_chain
        ] == [
            "database_claim",
            "database_portal_terminal_failure",
            "database_portal_protected_preservation_retry_recovery",
            "database_claim",
            "database_portal_terminal_failure",
        ]
        assert exact_chain[1]["body"]["completion_receipt"]["coordination"] == {}
        assert exact_chain[2]["body"]["completion_receipt"][
            "coordination"
        ] == source_retry_coordination
        assert exact_chain[4]["body"]["completion_receipt"]["coordination"] == {}

        observed_recovery_calls: list[tuple[str, str]] = []

        def recover_self_lock(
            failed_target: DatabaseTaskAttempt,
            seed: dict[str, object],
        ) -> dict[str, object]:
            observed_recovery_calls.append(
                (failed_target.attempt_id, str(seed["receipt_id"]))
            )
            return _protected_reconciliation_self_lock_receipt(
                daemon,
                failed_target,
                seed,
            )

        daemon.bind_protected_reconciliation_self_lock_recovery(
            recover_self_lock
        )
        now["ms"] = 25_004
        recovered = daemon.reconcile_terminal_portal_failures()
        assert len(recovered) == 1
        assert recovered[0]["changed"] is True
        assert recovered[0]["status"] == "retrying"
        assert recovered[0]["source_attempt_id"] == source.attempt_id
        assert recovered[0]["protected_preservation_evidence"] == (
            preservation_seed
        )
        assert observed_recovery_calls == [
            (target.attempt_id, str(preservation_seed["receipt_id"]))
        ]

        retrying = daemon.task_source.get(target.task_cid)
        assert retrying is not None
        assert retrying.status == "retrying"
        recovery_control = retrying.body["completion_receipt"]
        assert recovery_control["operation"] == (
            "database_portal_protected_preservation_"
            "reconciliation_retry_recovery"
        )
        context = recovery_control["reconciliation_self_lock_context"]
        assert context["source_claim_task_revision"] == exact_chain[0][
            "revision"
        ]
        assert context["target_blocked_task_revision"] == exact_chain[-1][
            "revision"
        ]
        replay = daemon.recover_protected_reconciliation_self_lock(
            target,
            history_context=context,
        )
        assert replay["changed"] is False
        assert replay["attempt_id"] == target.attempt_id
        assert replay["source_attempt_id"] == source.attempt_id
        # Operator and runner surfaces serialize recovery outcomes directly;
        # an idempotent replay must not leak internal attempt objects.
        json.dumps(replay, sort_keys=True)
        assert daemon.reconcile_terminal_portal_failures() == []
        assert len(observed_recovery_calls) == 1

        # The source lane retains only its own older failed attempt.  It must
        # verify the exact shared recovery wrapper, reject a forged wrapper,
        # and then recognize the wrapper's real target as foreign authority
        # instead of crashing while scanning its historical local row.
        historical_lane = _open_daemon(
            tmp_path,
            session="session:protected-reconciliation-self-lock",
            provider_fn=provider,
            max_task_attempts=3,
            lease_ms=5_000,
            clock_ms=lambda: now["ms"],
        )
        try:
            assert historical_lane.get_attempt(source.attempt_id) is not None
            historical_get = historical_lane.task_source.get
            forged_control = dict(recovery_control)
            forged_control["attempt_id"] = "attempt:forged-target"

            def get_with_forged_recovery(task_cid: str) -> object:
                task = historical_get(task_cid)
                if task_cid != source.task_cid or task is None:
                    return task
                return SimpleNamespace(
                    status=task.status,
                    revision=task.revision,
                    body={
                        **dict(task.body),
                        "completion_receipt": forged_control,
                    },
                )

            monkeypatch.setattr(
                historical_lane.task_source,
                "get",
                get_with_forged_recovery,
            )
            with pytest.raises(
                DatabaseImplementationConflictError,
                match="recovery wrapper does not reproduce",
            ):
                historical_lane.reconcile_terminal_portal_failures()
            monkeypatch.setattr(
                historical_lane.task_source,
                "get",
                historical_get,
            )

            raced_control = dict(recovery_control)
            raced_control.update(
                {
                    "attempt_id": "attempt:raced-target",
                    "claim_id": "claim:raced-target",
                    "lease_id": "lease:raced-target",
                    "owner_session_id": "session:raced-target",
                    "fencing_token": int(
                        recovery_control["fencing_token"]
                    )
                    + 1,
                    "fence_epoch": int(recovery_control["fence_epoch"]) + 1,
                }
            )
            raced_reads = {"count": 0}

            def get_with_raced_recovery(task_cid: str) -> object:
                task = historical_get(task_cid)
                if task_cid != source.task_cid or task is None:
                    return task
                raced_reads["count"] += 1
                if raced_reads["count"] == 1:
                    return task
                return SimpleNamespace(
                    status=task.status,
                    revision=task.revision,
                    body={
                        **dict(task.body),
                        "completion_receipt": raced_control,
                    },
                )

            monkeypatch.setattr(
                historical_lane.task_source,
                "get",
                get_with_raced_recovery,
            )
            with pytest.raises(
                DatabaseImplementationConflictError,
                match="retry belongs to another target",
            ):
                historical_lane.reconcile_terminal_portal_failures()
            monkeypatch.setattr(
                historical_lane.task_source,
                "get",
                historical_get,
            )

            historical_replay = (
                historical_lane.reconcile_terminal_portal_failures()
            )
            assert len(historical_replay) == 1
            assert historical_replay[0]["changed"] is False
            assert historical_replay[0]["reason"] == (
                "failed_attempt_control_superseded"
            )
            assert historical_replay[0]["successor_attempt_id"] == (
                target.attempt_id
            )
            assert historical_replay[0]["control_operation"] == (
                "database_portal_protected_preservation_"
                "reconciliation_retry_recovery"
            )
        finally:
            historical_lane.close()

        successor = daemon.claim_next()
        assert successor is not None
        assert successor.attempt_id not in {
            source.attempt_id,
            target.attempt_id,
        }
        fresh_claim = daemon.task_source.get(successor.task_cid)
        assert fresh_claim is not None
        assert fresh_claim.status == "in_progress"
        fresh_receipt = fresh_claim.body["completion_receipt"]
        assert fresh_receipt["operation"] == "database_claim"
        assert fresh_receipt["protected_preservation_seed"] == (
            preservation_seed
        )
        assert fresh_receipt[
            "protected_preservation_source_attempt_id"
        ] == source.attempt_id
        assert len(provider_calls) == 3
    finally:
        daemon.close()


def test_blocked_self_lock_replay_observes_complete_foreign_control_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A newer shared blocked attempt supersedes stale lane-local history."""

    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeError("not_attempted")

    daemon = _open_daemon(
        tmp_path,
        session="session:blocked-self-lock-history-replay",
        provider_fn=provider,
        max_task_attempts=3,
    )
    try:
        daemon.materialize_population(_population(1))
        failed_result = daemon.run_once()
        attempt = daemon.get_attempt(failed_result["attempt_id"])
        assert attempt is not None
        blocked = daemon.task_source.get(attempt.task_cid)
        assert blocked is not None
        assert blocked.status == "blocked"
        original_receipt = dict(blocked.body["completion_receipt"])
        projected_receipt = {"value": dict(original_receipt)}
        original_get = daemon.task_source.get

        def projected_get(task_cid: str) -> object:
            task = original_get(task_cid)
            if task_cid != attempt.task_cid or task is None:
                return task
            return SimpleNamespace(
                status=task.status,
                revision=task.revision,
                body={
                    **dict(task.body),
                    "completion_receipt": projected_receipt["value"],
                },
            )

        monkeypatch.setattr(daemon.task_source, "get", projected_get)

        def stale_history_replay(
            _attempt: DatabaseTaskAttempt,
            _task: object,
        ) -> dict[str, object]:
            raise DatabaseImplementationConflictError(
                "protected reconciliation self-lock recovery is one-shot"
            )

        monkeypatch.setattr(
            daemon,
            "_protected_reconciliation_self_lock_context",
            stale_history_replay,
        )

        # A partial identity mismatch is not a successor proof and must still
        # reach the fail-closed historical verifier.
        projected_receipt["value"] = {
            **original_receipt,
            "attempt_id": "attempt:partial-foreign",
        }
        with pytest.raises(
            DatabaseImplementationConflictError,
            match="self-lock recovery is one-shot",
        ):
            daemon.reconcile_terminal_portal_failures()

        # A complete foreign attempt/claim/lease identity is the shared
        # owner's newer blocked target, so the stale lane observes a no-op and
        # never tries to replay the one-shot history chain.
        projected_receipt["value"] = {
            **original_receipt,
            "attempt_id": "attempt:newer-blocked",
            "claim_id": "claim:newer-blocked",
            "lease_id": "lease:newer-blocked",
            "owner_session_id": "session:newer-blocked",
            "fencing_token": int(original_receipt["fencing_token"]) + 1,
            "fence_epoch": int(original_receipt["fence_epoch"]) + 1,
        }
        reconciled = daemon.reconcile_terminal_portal_failures()
        assert len(reconciled) == 1
        assert reconciled[0]["changed"] is False
        assert reconciled[0]["reason"] == (
            "failed_attempt_control_superseded"
        )
        assert reconciled[0]["successor_attempt_id"] == (
            "attempt:newer-blocked"
        )
        assert reconciled[0]["control_status"] == "blocked"
        assert reconciled[0]["control_operation"] == (
            "database_portal_terminal_failure"
        )
    finally:
        daemon.close()


@pytest.mark.parametrize("provider_reset_ms", [2_000_000, 0])
def test_typed_capacity_retry_crosses_lanes_without_refunding_attempt(
    tmp_path: Path,
    provider_reset_ms: int,
) -> None:
    holder: dict[str, DatabaseImplementationDaemon] = {}
    source_portal_task_cid = "baguqeera" + "a" * 52

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        receipt = _capacity_retry_receipt(
            holder["daemon"],
            attempt,
            retry_not_before_ms=provider_reset_ms,
            portal_task_cid=source_portal_task_cid,
        )
        raise DatabasePortalCapacityRetry(receipt)

    first = _open_daemon(
        tmp_path,
        session="session:capacity-lane-a",
        provider_fn=provider,
        max_task_attempts=3,
        clock_ms=lambda: 1_000_000,
        lane="a",
    )
    holder["daemon"] = first
    try:
        first.task_source._intent._clock_ms = (  # type: ignore[attr-defined]
            lambda: 1_000_123
        )
        first.materialize_population(_population(1))
        result = first.run_once()
        implementation = result["implementation_result"]
        assert implementation["portal_retryable_failure"] is True
        assert implementation["portal_terminal_failure"] is False
        assert implementation["deferred"] is False
        assert implementation["attempt_consumed"] is True
        assert implementation["provider_dispatched"] is True
        assert implementation["typed_deferral_slot_consumed"] is False
        assert implementation["retry_state"]["status"] == "retrying"

        source = first.get_attempt(result["attempt_id"])
        assert source is not None
        evidence = first._terminal_retry_evidence(source)
        assert evidence is not None
        assert evidence["typed_deferral_budget"] is None
        seed = evidence["typed_capacity_retry"]
        assert seed["task_cid"] == source.task_cid
        assert seed["post_dispatch_capacity_proof"][
            "task_revision_cid"
        ] == source_portal_task_cid
        assert source_portal_task_cid != source.task_cid
        assert seed["portal_attempt"] == 1
        assert seed["retry_not_before_ms"] == provider_reset_ms
        assert seed["remaining_task_attempts"] == 2
        task = first.task_source.get(source.task_cid)
        assert task is not None
        assert task.status == "retrying"
        assert task.body["completion_receipt"]["capacity_retry_seed"] == seed
        assert task.body["completion_receipt"]["retry_not_before_ms"] == 2_000_000
        queue_entry = first.task_source.get_queue_entry(source.task_cid)
        assert queue_entry is not None
        assert queue_entry.retry_not_before_ms == 2_000_000
        encoded_control_receipt = json.dumps(
            task.body["completion_receipt"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        assert len(encoded_control_receipt) < QUACK_OWNER_COMMAND_MAX_BYTES

        # Recomputed nested and top-level tampering must fail semantic
        # verification even when every content identity is internally valid.
        forged = json.loads(json.dumps(seed))
        forged["codex_capacity_receipt"]["source"] = "untrusted_runner"
        forged = _rehash_capacity_retry_receipt(first, forged)
        with pytest.raises(DatabaseImplementationAuthorityError):
            first._verified_capacity_retry_receipt(source, forged)
        shortened = json.loads(json.dumps(seed))
        shortened["retry_not_before_ms"] = 0 if provider_reset_ms else 1
        shortened.pop("receipt_id")
        shortened["receipt_id"] = first._database_portal_evidence_digest(
            shortened
        )
        with pytest.raises(DatabaseImplementationAuthorityError):
            first._verified_capacity_retry_receipt(source, shortened)
        replayed = first.reconcile_terminal_retry_states()
        assert len(replayed) == 1
        assert replayed[0]["status"] == "retrying"
        assert replayed[0]["changed"] is False
        assert first.task_source.get_queue_entry(
            source.task_cid
        ).retry_not_before_ms == 2_000_000
    finally:
        first.close()

    # Lane B has a fresh coordination/execution journal, so source attempt
    # 1 is intentionally absent.  The exact shared control receipt authorizes
    # the handoff, and its private attempt counter can also restart at 1.
    second = _open_daemon(
        tmp_path,
        session="session:capacity-lane-b",
        max_task_attempts=3,
        clock_ms=lambda: 2_000_001,
        lane="b",
    )
    try:
        assert second.get_attempt(source.attempt_id) is None
        successor = second.claim_next()
        assert successor is not None
        assert successor.attempt_number == 1
        claimed = second.task_source.get(source.task_cid)
        assert claimed is not None
        claim_receipt = claimed.body["completion_receipt"]
        assert claim_receipt["operation"] == "database_claim"
        assert claim_receipt["capacity_retry_source_attempt_id"] == (
            source.attempt_id
        )
        assert claim_receipt["capacity_retry_seed"] == seed
    finally:
        second.close()


def test_superseded_consumed_attempt_recovers_and_crosses_lanes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [1_000_000]

    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeError("portal_provider_failed")

    first = _open_daemon(
        tmp_path,
        session="session:consumed-lane-a",
        provider_fn=provider,
        max_task_attempts=3,
        clock_ms=lambda: clock[0],
        lane="consumed-a",
    )
    try:
        first.task_source._intent._clock_ms = (  # type: ignore[attr-defined]
            lambda: clock[0]
        )
        first.materialize_population(_population(1))
        monkeypatch.setattr(
            first,
            "_persist_terminal_portal_failure",
            lambda attempt, **_kwargs: {
                "task_cid": attempt.task_cid,
                "attempt_id": attempt.attempt_id,
                "status": "blocked",
                "changed": False,
            },
        )
        failed_result = first.run_once()
        source = first.get_attempt(failed_result["attempt_id"])
        assert source is not None
        assert source.status == "failed"
        shared_source = first.task_source.get(source.task_cid)
        assert shared_source is not None
        assert shared_source.status == "in_progress"
        source_task_revision = shared_source.revision

        stale_receipt = {
            "operation": "database_portal_terminal_failure",
            "attempt_id": "attempt:foreign-stale",
            "attempt_number": 1,
            "claim_id": "claim:foreign-stale",
            "lease_id": "lease:foreign-stale",
            "owner_session_id": "session:foreign-stale",
            "fencing_token": 1,
            "fence_epoch": 1,
            "execution_phase": "failed",
            "execution_revision": 1,
            "execution_finished_at_ms": source.started_at_ms - 1,
            "reason": "portal_provider_failed",
            "retryable": False,
            "coordination": {},
            "control_expected_status": "in_progress",
            "control_expected_revision": source_task_revision,
        }
        assert first._is_exact_foreign_terminal_failure_receipt(
            source,
            stale_receipt,
            source_revision=source_task_revision,
        )
        malformed_receipts = []
        missing_claim = dict(stale_receipt)
        missing_claim.pop("claim_id")
        malformed_receipts.append(missing_claim)
        malformed_receipts.append(
            {**stale_receipt, "execution_phase": "running"}
        )
        malformed_receipts.append({**stale_receipt, "lease_id": ""})
        malformed_receipts.append(
            {
                **stale_receipt,
                "owner_session_id": source.owner_session_id,
            }
        )
        assert all(
            not first._is_exact_foreign_terminal_failure_receipt(
                source,
                malformed,
                source_revision=source_task_revision,
            )
            for malformed in malformed_receipts
        )
        first.task_source.compare_and_set_status(
            source.task_cid,
            expected_revision=source_task_revision,
            status="blocked",
            receipt=stale_receipt,
        )
        blocked = first.task_source.get(source.task_cid)
        assert blocked is not None
        assert blocked.revision == source_task_revision + 1
        seed = _consumed_attempt_retry_receipt(
            first,
            source,
            source_task_revision=source_task_revision,
        )
        first.bind_superseded_consumed_attempt_recovery(
            lambda _attempt: seed
        )
        clock[0] += 60_001
        recovered = first.reconcile_terminal_portal_failures()
        assert len(recovered) == 1
        assert recovered[0]["changed"] is True
        assert recovered[0]["consumed_attempt_retry_evidence"] == seed
        retrying = first.task_source.get(source.task_cid)
        assert retrying is not None
        assert retrying.status == "retrying"
        recovery_receipt = retrying.body["completion_receipt"]
        assert recovery_receipt["operation"] == (
            "database_portal_superseded_consumed_attempt_recovery"
        )
        assert recovery_receipt["consumed_attempt_retry_seed"] == seed
        assert recovery_receipt["supersession_context"][
            "superseded_control_receipt"
        ] == stale_receipt
        assert len(
            json.dumps(
                recovery_receipt,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ) < QUACK_OWNER_COMMAND_MAX_BYTES
        assert first.reconcile_terminal_portal_failures() == []
    finally:
        first.close()

    clock[0] += 1
    second = _open_daemon(
        tmp_path,
        session="session:consumed-lane-b",
        max_task_attempts=3,
        clock_ms=lambda: clock[0],
        lane="consumed-b",
    )
    racer = _open_daemon(
        tmp_path,
        session="session:consumed-lane-racer",
        max_task_attempts=3,
        clock_ms=lambda: clock[0],
        lane="consumed-racer",
    )
    try:
        second.task_source._intent._clock_ms = (  # type: ignore[attr-defined]
            lambda: clock[0]
        )
        racer.task_source._intent._clock_ms = (  # type: ignore[attr-defined]
            lambda: clock[0]
        )
        assert second.get_attempt(source.attempt_id) is None
        winner: dict[str, DatabaseTaskAttempt] = {}
        losing_receipt: dict[str, object] = {}
        original_loser_cas = racer._cas_task_status_database

        def win_shared_claim_before_loser_cas(
            *args: object,
            **kwargs: object,
        ) -> object:
            receipt = kwargs.get("receipt")
            assert isinstance(receipt, dict)
            losing_receipt.update(receipt)
            if "attempt" not in winner:
                claimed = second.claim_next()
                assert claimed is not None
                winner["attempt"] = claimed
            return original_loser_cas(*args, **kwargs)

        monkeypatch.setattr(
            racer,
            "_cas_task_status_database",
            win_shared_claim_before_loser_cas,
        )
        # Both lanes admitted the retrying revision.  Lane B wins its shared
        # claim immediately before the racer's owner CAS.  The racer must
        # observe an ordinary stale-CAS conflict, release its lane-local
        # lease, and never reinterpret the winner's carried seed.
        assert racer.claim_next() is None
        successor = winner["attempt"]
        assert successor is not None
        assert successor.attempt_number == 1
        claimed = second.task_source.get(source.task_cid)
        assert claimed is not None
        claim_receipt = claimed.body["completion_receipt"]
        assert claim_receipt["operation"] == "database_claim"
        assert claim_receipt[
            "consumed_attempt_retry_source_attempt_id"
        ] == source.attempt_id
        assert claim_receipt["consumed_attempt_retry_seed"] == seed
        losing_claim = racer.coordinator.get_task_claim(
            str(losing_receipt["claim_id"])
        )
        losing_lease = racer.coordinator.get_lease(
            str(losing_receipt["lease_id"])
        )
        assert losing_claim is not None
        assert losing_claim.to_dict()["state"] == "released"
        assert losing_lease is not None
        assert losing_lease.to_dict()["state"] == "released"
        assert racer.get_attempt(str(losing_receipt["attempt_id"])) is None
    finally:
        racer.close()
        second.close()


def test_consumed_attempt_recovery_rejects_nonlegacy_failed_phase(
    tmp_path: Path,
) -> None:
    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeError("provider_authentication_denied")

    daemon = _open_daemon(
        tmp_path,
        session="session:consumed-phase-rejection",
        provider_fn=provider,
        max_task_attempts=3,
    )
    try:
        daemon.materialize_population(_population(1))
        result = daemon.run_once()
        attempt = daemon.get_attempt(result["attempt_id"])
        assert attempt is not None
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="exact legacy generic portal_provider_failed phase",
        ):
            daemon.recover_superseded_consumed_portal_attempt(
                attempt,
                retry_evidence={},
            )
    finally:
        daemon.close()


@pytest.mark.parametrize(
    ("mutation", "error_type"),
    [
        ("foreign_operation", None),
        ("missing_seed", DatabaseImplementationAuthorityError),
        ("wrong_seed_receipt", DatabaseImplementationAuthorityError),
    ],
)
def test_terminal_portal_reconciliation_handles_retrying_projection_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    error_type: type[Exception] | None,
) -> None:
    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeError("portal_provider_failed")

    daemon = _open_daemon(
        tmp_path,
        session="session:validation-retry-recovery-forgery",
        provider_fn=provider,
        max_task_attempts=3,
    )
    try:
        daemon.materialize_population(_population(1))
        failed_result = daemon.run_once()
        attempt = daemon.get_attempt(failed_result["attempt_id"])
        assert attempt is not None
        retry_evidence = _validation_retry_receipt(daemon, attempt)
        daemon.recover_blocked_portal_validation_retry(
            attempt,
            retry_evidence=retry_evidence,
        )
        persisted = daemon.task_source.get(attempt.task_cid)
        assert persisted is not None
        receipt = dict(persisted.body["completion_receipt"])
        if mutation == "foreign_operation":
            receipt["operation"] = "database_portal_retry"
        elif mutation == "missing_seed":
            receipt.pop("validation_retry_seed")
        else:
            seed = dict(receipt["validation_retry_seed"])
            seed["receipt_id"] = "sha256:" + "0" * 64
            receipt["validation_retry_seed"] = seed

        original_get = daemon.task_source.get

        def projected_get(task_cid: str) -> object:
            task = original_get(task_cid)
            if task_cid != attempt.task_cid or task is None:
                return task
            return SimpleNamespace(
                status=task.status,
                revision=task.revision,
                body={**dict(task.body), "completion_receipt": receipt},
            )

        monkeypatch.setattr(daemon.task_source, "get", projected_get)
        if error_type is None:
            # A generic owner/board retry may legitimately replace the
            # validation-recovery receipt.  The historical terminal attempt
            # is observation-only in that case: do not crash the lane and do
            # not perform another control transition.
            assert daemon.reconcile_terminal_portal_failures() == []
            durable = original_get(attempt.task_cid)
            assert durable is not None
            assert durable.status == "retrying"
            assert durable.revision == persisted.revision
        else:
            with pytest.raises(error_type):
                daemon.reconcile_terminal_portal_failures()
    finally:
        daemon.close()


def test_terminal_portal_recovery_projection_observes_newer_fence(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}

    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeError("portal_provider_failed")

    daemon = _open_daemon(
        tmp_path,
        session="session:validation-retry-recovery-newer-fence",
        provider_fn=provider,
        max_task_attempts=3,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        failed_result = daemon.run_once()
        attempt = daemon.get_attempt(failed_result["attempt_id"])
        assert attempt is not None
        daemon.recover_blocked_portal_validation_retry(
            attempt,
            retry_evidence=_validation_retry_receipt(daemon, attempt),
        )
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "retrying"

        source_claim = daemon.coordinator.get_task_claim(attempt.claim_id)
        assert source_claim is not None
        now["ms"] = 7_000
        daemon.coordinator.expire_task_claim(source_claim, now_ms=now["ms"])
        newer = daemon.coordinator.claim_ready_task(
            owner_session_id="session:newer-validation-retry-fence",
            lease_ms=5_000,
            now_ms=now["ms"],
        )
        assert newer is not None
        assert newer.fencing_token > attempt.fencing_token

        reconciled = daemon.reconcile_terminal_portal_failures()
        assert len(reconciled) == 1
        assert reconciled[0]["changed"] is False
        assert reconciled[0]["reason"] == (
            "failed_attempt_coordination_superseded"
        )
        assert reconciled[0]["successor_claim_id"] == newer.claim_id
        assert reconciled[0]["successor_attempt_id"] == newer.attempt_id
        unchanged = daemon.task_source.get(attempt.task_cid)
        assert unchanged is not None
        assert unchanged.status == "retrying"
    finally:
        daemon.close()


def test_exhausted_supersession_with_admitted_repair_survives_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = {"ms": 1_000}
    repo, base_head, base_tree = _git_recovery_repo(tmp_path)

    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeDeferred(
            "validation_project_dependency_preflight_failed",
            backoff_seconds=300,
        )

    daemon = _open_daemon(
        tmp_path,
        session="session:typed-budget-supersession",
        provider_fn=provider,
        lease_ms=5_000,
        max_task_attempts=1,
        clock_ms=lambda: now["ms"],
    )
    try:
        population = _population(1)
        tasks = population["tasks"]
        assert isinstance(tasks, list)
        tasks[0]["base_revision"] = base_head
        tasks[0]["base_repository_tree_id"] = base_tree
        daemon.materialize_population(population)
        exhausted = daemon.run_once()
        task_cid = str(exhausted["claimed_task_cid"])
        assert exhausted["implementation_result"]["retry_budget_exhausted"] is True
        blocked = daemon.task_source.get(task_cid)
        assert blocked is not None and blocked.status == "blocked"

        repair_head, repair_tree = _git_commit(
            repo,
            name=TYPED_DEFERRAL_RECOVERY_TEST_PATH,
            content="quota/high route repair\n",
        )
        failure, route_outcome = _successful_quota_high_pair()
        request = database_task_source_module._build_owner_typed_deferral_budget_supersession_request(
            task_cid=blocked.task_cid,
            task_revision=blocked.revision,
            task_body=blocked.body,
            repair_head=repair_head,
            repair_tree=repair_tree,
            quota_probe_receipt=failure,
            route_outcome=route_outcome,
            owner_command_request_id="d" * 32,
            owner_store_id="store:test-reconciliation",
            owner_store_generation="generation:test-reconciliation",
            admitted_at_ms=int(failure["observed_at_ms"]),
            _owner_admission_sentinel=(
                database_task_source_module._TYPED_DEFERRAL_PROVIDER_EVIDENCE_OWNER_SENTINEL
            ),
        )
        rearmed = daemon.task_source.rearm_blocked_task(
            task_cid,
            receipt=request,
        )
        assert rearmed.task.status == "retrying"

        # Model a lane-local shared-CAS loser: it advanced the same-task fence
        # after the exhausted claim expired, but never inserted a newer daemon
        # attempt or replaced the canonical supersession receipt.  The
        # coordinator must keep rejecting mutation through the old fence while
        # restart reconciliation may observe its exact terminal history.
        failed_attempt = daemon._latest_failed_attempts()[0]
        assert failed_attempt.task_cid == task_cid
        old_claim = daemon.coordinator.get_task_claim(failed_attempt.claim_id)
        assert old_claim is not None
        now["ms"] = 7_000
        daemon.coordinator.expire_task_claim(old_claim, now_ms=now["ms"])
        newer_claim = daemon.coordinator.claim_ready_task(
            owner_session_id="session:typed-budget-cas-loser",
            lease_ms=5_000,
            now_ms=now["ms"],
        )
        assert newer_claim is not None
        assert newer_claim.task_cid == task_cid
        assert newer_claim.fencing_token > old_claim.fencing_token
        newer_lease = daemon.coordinator.get_lease(newer_claim.lease_id)
        assert newer_lease is not None
        daemon.coordinator.release(
            newer_lease,
            reason="shared_board_claim_conflict",
            now_ms=now["ms"],
        )
        with pytest.raises(DatabaseCoordinationError, match="latest"):
            daemon.coordinator.expire_task_claim(old_claim, now_ms=now["ms"])

        # A later unrelated accepted commit is permitted: the admitted repair
        # remains an ancestor and still reproduces its exact tree.
        current_head, _current_tree = _git_commit(
            repo,
            name="unrelated.txt",
            content="later accepted merge\n",
        )
        assert current_head != repair_head
        daemon._merge_repo_root = repo
        assert task_cid in daemon.sync_ready_tasks_into_coordination()
        now["ms"] = 7_000
        reconciled = daemon.reconcile_terminal_retry_states()
        assert len(reconciled) == 1
        assert reconciled[0]["changed"] is False
        assert reconciled[0]["status"] == "retrying"
        assert reconciled[0]["reason"] == (
            "typed_portal_deferral_budget_superseded"
        )
        assert reconciled[0]["coordination"] == blocked.body[
            "completion_receipt"
        ]["coordination"]
        assert daemon.task_source.get(task_cid).status == "retrying"

        successor_projection = (
            daemon.coordinator.get_task_claim_successor_projection(
                task_cid=failed_attempt.task_cid,
                after_fencing_token=failed_attempt.fencing_token,
                after_fence_epoch=failed_attempt.fence_epoch,
            )
        )
        assert successor_projection is not None
        malformed_successor = json.loads(json.dumps(successor_projection))
        malformed_successor["lease"]["attempt_id"] = "attempt:foreign"
        with monkeypatch.context() as patch:
            patch.setattr(
                daemon.coordinator,
                "get_task_claim_successor_projection",
                lambda **_kwargs: malformed_successor,
            )
            with pytest.raises(
                DatabaseImplementationConflictError,
                match="coordination successor does not reproduce",
            ):
                daemon._reconcile_failed_attempt_coordination(failed_attempt)

        expired_lease = daemon.coordinator.get_lease(failed_attempt.lease_id)
        assert expired_lease is not None
        malformed_lease = expired_lease.to_dict()
        malformed_lease["attempt_id"] = "attempt:foreign"
        with monkeypatch.context() as patch:
            patch.setattr(
                daemon.coordinator,
                "get_lease",
                lambda _lease_id: SimpleNamespace(
                    to_dict=lambda: malformed_lease
                ),
            )
            with pytest.raises(
                DatabaseImplementationConflictError,
                match="expired failed execution authority does not reproduce",
            ):
                daemon._reconcile_failed_attempt_coordination(failed_attempt)

        # An authoritative later completion remains immutable even though the
        # historical failed attempt and exhausted observation are retained.
        current = daemon.task_source.get(task_cid)
        assert current is not None
        daemon.task_source._intent.cas_task_status(
            task_cid=task_cid,
            expected_revision=current.revision,
            new_status="completed",
            receipt={"operation": "authoritative_external_completion"},
            allow_completion_without_evidence=True,
        )
        assert daemon.reconcile_terminal_retry_states() == []
        assert daemon.task_source.get(task_cid).status == "completed"
    finally:
        daemon.close()


def test_admitted_exhausted_supersession_is_observed_from_partial_lane(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}
    repo, base_head, base_tree = _git_recovery_repo(tmp_path)

    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeDeferred(
            "validation_project_dependency_preflight_failed",
            backoff_seconds=300,
        )

    producer = _open_daemon(
        tmp_path,
        session="session:typed-budget-partial-producer",
        provider_fn=provider,
        lease_ms=5_000,
        max_task_attempts=2,
        clock_ms=lambda: now["ms"],
    )
    try:
        population = _population(1)
        tasks = population["tasks"]
        assert isinstance(tasks, list)
        tasks[0]["base_revision"] = base_head
        tasks[0]["base_repository_tree_id"] = base_tree
        producer.materialize_population(population)
        first = producer.run_once()
        first_attempt = producer.get_attempt(str(first["attempt_id"]))
        assert first_attempt is not None
        first_evidence = producer._terminal_retry_evidence(first_attempt)
        assert first_evidence is not None
        first_budget = first_evidence["typed_deferral_budget"]
        assert isinstance(first_budget, Mapping)
        assert first_budget["exhausted"] is False
    finally:
        producer.close()

    # Preserve the retired lane's one-attempt execution prefix.  Its new
    # coordination sidecar deliberately has no claim, lease, or attempt rows.
    shutil.copy2(
        tmp_path / "execution.duckdb",
        tmp_path / "execution-partial.duckdb",
    )

    now["ms"] = 301_001
    producer = _open_daemon(
        tmp_path,
        session="session:typed-budget-partial-producer",
        provider_fn=provider,
        lease_ms=5_000,
        max_task_attempts=2,
        clock_ms=lambda: now["ms"],
    )
    try:
        exhausted = producer.run_once()
        assert exhausted["implementation_result"][
            "retry_budget_exhausted"
        ] is True
        task_cid = str(exhausted["claimed_task_cid"])
        blocked = producer.task_source.get(task_cid)
        assert blocked is not None and blocked.status == "blocked"

        repair_head, repair_tree = _git_commit(
            repo,
            name=TYPED_DEFERRAL_RECOVERY_TEST_PATH,
            content="quota/high route repair\n",
        )
        failure, route_outcome = _successful_quota_high_pair()
        request = database_task_source_module._build_owner_typed_deferral_budget_supersession_request(
            task_cid=blocked.task_cid,
            task_revision=blocked.revision,
            task_body=blocked.body,
            repair_head=repair_head,
            repair_tree=repair_tree,
            quota_probe_receipt=failure,
            route_outcome=route_outcome,
            owner_command_request_id="f" * 32,
            owner_store_id="store:test-partial-reconciliation",
            owner_store_generation="generation:test-partial-reconciliation",
            admitted_at_ms=int(failure["observed_at_ms"]),
            _owner_admission_sentinel=(
                database_task_source_module._TYPED_DEFERRAL_PROVIDER_EVIDENCE_OWNER_SENTINEL
            ),
        )
        retrying = producer.task_source.rearm_blocked_task(
            task_cid,
            receipt=request,
        ).task
        assert retrying.status == "retrying"
        task_before = retrying.to_dict()
        queue = producer.task_source.get_queue_entry(task_cid)
        assert queue is not None
        queue_before = queue.to_dict()
    finally:
        producer.close()

    observer = _open_daemon(
        tmp_path,
        session="session:typed-budget-partial-observer",
        provider_fn=provider,
        lease_ms=5_000,
        max_task_attempts=2,
        clock_ms=lambda: now["ms"],
        lane="partial",
    )
    try:
        observer._merge_repo_root = repo
        local_attempts = observer._latest_failed_attempts()
        assert len(local_attempts) == 1
        local_attempt = local_attempts[0]
        assert local_attempt.attempt_id == first_attempt.attempt_id
        local_evidence = observer._terminal_retry_evidence(local_attempt)
        assert local_evidence is not None
        local_budget = local_evidence["typed_deferral_budget"]
        assert isinstance(local_budget, Mapping)
        assert local_budget["exhausted"] is False
        assert observer.coordinator.get_task_claim(
            local_attempt.claim_id
        ) is None

        # A distinct attempt with only a same-millisecond ordering witness is
        # not a proved prefix of the exhausted attempt.
        canonical = observer.task_source.get(task_cid)
        assert canonical is not None
        exhausted_receipt = canonical.body["completion_receipt"][
            "exhausted_receipt"
        ]
        equal_timestamp_attempt = replace(
            local_attempt,
            finished_at_ms=int(
                exhausted_receipt["execution_finished_at_ms"]
            ),
        )
        assert observer._typed_deferral_supersession_reconciliation_observation(
            equal_timestamp_attempt,
            canonical,
            local_evidence,
        ) is None

        reconciled = observer.reconcile_terminal_retry_states()
        assert len(reconciled) == 1
        assert reconciled[0]["changed"] is False
        assert reconciled[0]["status"] == "retrying"
        assert reconciled[0]["reason"] == (
            "typed_portal_deferral_budget_superseded"
        )
        assert reconciled[0]["attempt_id"] == first_attempt.attempt_id
        assert reconciled[0]["supersession_attempt_id"] == str(
            exhausted["attempt_id"]
        )
        assert observer.task_source.get(task_cid).to_dict() == task_before
        observed_queue = observer.task_source.get_queue_entry(task_cid)
        assert observed_queue is not None
        assert observed_queue.to_dict() == queue_before

        # Recomputing the content identity cannot make a caller-controlled
        # top-level attempt identity disagree with its embedded receipt.
        tampered_body = json.loads(json.dumps(task_before["body"]))
        tampered_supersession = tampered_body["completion_receipt"]
        tampered_supersession["claim_id"] = "claim:tampered-top-level"
        tampered_supersession.pop("supersession_id")
        tampered_supersession["supersession_id"] = content_identity(
            tampered_supersession
        )
        tampered_task = SimpleNamespace(
            task_cid=task_before["task_cid"],
            task_alias=task_before["task_alias"],
            revision=task_before["revision"],
            status=task_before["status"],
            body=tampered_body,
        )
        assert observer._typed_deferral_claim_is_admitted(
            tampered_task
        ) is False
    finally:
        observer.close()


@pytest.mark.parametrize(
    "rearm_kind",
    (
        "generic",
        "nonancestor_repair",
        "source_as_repair",
        "source_head_missing",
        "source_tree_mismatch",
        "repair_tree_mismatch",
        "unrelated_descendant",
        "dirty_worktree",
    ),
)
def test_exhausted_generic_or_unadmitted_repair_is_reblocked(
    tmp_path: Path,
    rearm_kind: str,
) -> None:
    now = {"ms": 1_000}
    repo, base_head, base_tree = _git_recovery_repo(tmp_path)

    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeDeferred(
            "validation_project_dependency_preflight_failed",
            backoff_seconds=300,
        )

    daemon = _open_daemon(
        tmp_path,
        session=f"session:typed-budget-{rearm_kind}",
        provider_fn=provider,
        lease_ms=5_000,
        max_task_attempts=1,
        clock_ms=lambda: now["ms"],
    )
    try:
        population = _population(1)
        tasks = population["tasks"]
        assert isinstance(tasks, list)
        tasks[0]["base_revision"] = (
            "8" * 40 if rearm_kind == "source_head_missing" else base_head
        )
        tasks[0]["base_repository_tree_id"] = (
            "9" * 40 if rearm_kind == "source_tree_mismatch" else base_tree
        )
        daemon.materialize_population(population)
        exhausted = daemon.run_once()
        task_cid = str(exhausted["claimed_task_cid"])
        blocked = daemon.task_source.get(task_cid)
        assert blocked is not None and blocked.status == "blocked"

        if rearm_kind == "generic":
            # Simulate a lower-level repository writer bypassing the public
            # DatabaseTaskSource gate.  Restart reconciliation still fails
            # closed and restores the exhausted block.
            daemon.task_source._intent.cas_task_status(
                task_cid=task_cid,
                expected_revision=blocked.revision,
                new_status="retrying",
                receipt={"operation": "generic_owner_retry"},
            )
        else:
            if rearm_kind == "nonancestor_repair":
                # An object-store commit outside current history is not an
                # admitted runtime repair.
                repair_tree = _git_output(repo, "rev-parse", "HEAD^{tree}")
                repair_head = _git_output(
                    repo,
                    "commit-tree",
                    repair_tree,
                    input_text="unrelated orphan repair\n",
                )
            elif rearm_kind == "source_as_repair":
                repair_head, repair_tree = base_head, base_tree
            elif rearm_kind == "unrelated_descendant":
                # Strict descent is necessary but not sufficient: a commit
                # outside the closed production recovery paths cannot reset a
                # provider/validation deferral budget.
                repair_head, repair_tree = _git_commit(
                    repo,
                    name="unrelated.txt",
                    content="unrelated accepted change\n",
                )
            else:
                repair_head, repair_tree = _git_commit(
                    repo,
                    name=TYPED_DEFERRAL_RECOVERY_TEST_PATH,
                    content="quota/high route repair\n",
                )
                if rearm_kind == "repair_tree_mismatch":
                    repair_tree = "7" * 40
            failure, route_outcome = _successful_quota_high_pair()
            request = database_task_source_module._build_owner_typed_deferral_budget_supersession_request(
                task_cid=blocked.task_cid,
                task_revision=blocked.revision,
                task_body=blocked.body,
                repair_head=repair_head,
                repair_tree=repair_tree,
                quota_probe_receipt=failure,
                route_outcome=route_outcome,
                owner_command_request_id="e" * 32,
                owner_store_id="store:test-reblock",
                owner_store_generation="generation:test-reblock",
                admitted_at_ms=int(failure["observed_at_ms"]),
                _owner_admission_sentinel=(
                    database_task_source_module._TYPED_DEFERRAL_PROVIDER_EVIDENCE_OWNER_SENTINEL
                ),
            )
            daemon.task_source.rearm_blocked_task(task_cid, receipt=request)
            if rearm_kind == "dirty_worktree":
                (repo / "unadmitted.txt").write_text(
                    "dirty runtime generation\n",
                    encoding="utf-8",
                )

        daemon._merge_repo_root = repo
        now["ms"] = 7_000
        if rearm_kind != "generic":
            # Model one of four lanes whose reconciliation pass completed just
            # before the owner accepted this rearm.  Claim admission must
            # independently reject it; waiting for the lane's next restart
            # reconciliation would leave a provider-dispatch race.
            assert task_cid not in daemon.sync_ready_tasks_into_coordination()
            assert daemon.claim_next() is None
            assert daemon.task_source.get(task_cid).status == "retrying"
        reconciled = daemon.reconcile_terminal_retry_states()
        assert len(reconciled) == 1
        assert reconciled[0]["changed"] is True
        assert reconciled[0]["status"] == "blocked"
        assert reconciled[0]["reason"] == (
            "typed_portal_deferral_budget_exhausted"
        )
        assert daemon.task_source.get(task_cid).status == "blocked"
    finally:
        daemon.close()


@pytest.mark.parametrize(
    "terminal_status",
    ("completed", "complete", "done", "rejected"),
)
def test_exhausted_typed_deferral_yields_to_later_terminal_control_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_status: str,
) -> None:
    now = {"ms": 1_000}

    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeDeferred(
            "validation_project_dependency_preflight_failed",
            backoff_seconds=300,
        )

    daemon = _open_daemon(
        tmp_path,
        session="session:typed-budget-terminal-precedence",
        provider_fn=provider,
        lease_ms=5_000,
        max_task_attempts=2,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        initial = daemon.run_once()
        task_cid = str(initial["claimed_task_cid"])
        now["ms"] = 301_001

        def crash_before_block(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError("simulated crash before exhausted control CAS")

        monkeypatch.setattr(
            daemon,
            "_persist_typed_deferral_budget_exhausted",
            crash_before_block,
        )
        interrupted = daemon.run_once()
        assert "simulated crash" in interrupted["implementation_result"][
            "fail_error"
        ]

        control = daemon.task_source.get(task_cid)
        assert control is not None
        validation_digest = "sha256:" + ("a" * 64)
        daemon.task_source.record_validation_result(
            task_cid=task_cid,
            outcome="passed",
            evidence_digest=validation_digest,
            argv=("python3", "-m", "pytest", "-q"),
            attempt_id=str(interrupted["attempt_id"]),
        )
        completed = daemon._cas_task_status_database(
            task_cid,
            expected_revision=int(control.revision),
            new_status=terminal_status,
            receipt={"operation": "simulated_external_completion"},
            evidence_digests=(
                (validation_digest,)
                if terminal_status in {"completed", "complete", "done"}
                else None
            ),
        )
        assert completed.task.status == terminal_status

        monkeypatch.undo()
        assert daemon.reconcile_terminal_retry_states() == []
        assert daemon.task_source.get(task_cid).status == terminal_status
    finally:
        daemon.close()


def test_retry_reconciliation_skips_superseded_coordination_fence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = {"ms": 1_000}
    daemon = _open_daemon(
        tmp_path,
        session="session:superseded-retry-fence",
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        failed_attempt = daemon.claim_next()
        assert failed_attempt is not None
        failed_attempt = daemon.commit_phase(failed_attempt, "context")
        failed_attempt = daemon.commit_phase(
            failed_attempt,
            "failed",
            body={
                "reason": "typed_deferral",
                "portal_retryable_failure": True,
                "backoff_seconds": 300,
            },
        )
        old_claim = daemon.coordinator.get_task_claim(failed_attempt.claim_id)
        assert old_claim is not None
        now["ms"] = 7_000
        daemon.coordinator.expire_task_claim(old_claim, now_ms=now["ms"])
        replacement = daemon.coordinator.claim_ready_task(
            owner_session_id="session:newer-fence",
            lease_ms=5_000,
            now_ms=now["ms"],
        )
        assert replacement is not None
        assert replacement.fencing_token > failed_attempt.fencing_token
        replacement_lease = daemon.coordinator.get_lease(replacement.lease_id)
        assert replacement_lease is not None
        daemon.coordinator.release(
            replacement_lease,
            reason="shared_board_claim_conflict",
            now_ms=now["ms"],
        )

        with pytest.raises(DatabaseCoordinationError, match="latest"):
            daemon.coordinator.expire_task_claim(old_claim, now_ms=now["ms"])

        reconciliations = daemon.reconcile_terminal_retry_states()
        assert len(reconciliations) == 1
        reconciliation = reconciliations[0]
        assert reconciliation["changed"] is False
        assert reconciliation["status"] == "in_progress"
        assert reconciliation["reason"] == (
            "failed_attempt_coordination_superseded"
        )
        assert reconciliation["successor_claim_id"] == replacement.claim_id
        assert reconciliation["successor_attempt_id"] == replacement.attempt_id
        assert reconciliation["coordination"][
            "superseded_by_newer_fence"
        ] is True

        control = daemon.task_source.get(failed_attempt.task_cid)
        assert control is not None
        assert control.status == "in_progress"
        control_revision = control.revision
        assert daemon.task_source.get_queue_entry(failed_attempt.task_cid) is None

        # Observation-only reconciliation is stable across owner passes and
        # must not advertise a write on every restart loop.
        monkeypatch.setattr(daemon, "claim_next", lambda: None)
        for _ in range(2):
            observed = daemon.run_once()
            assert observed["write_count"] == 0
            assert observed["unchanged"] is True
            repeated = observed["terminal_retry_reconciliations"]
            assert len(repeated) == 1
            assert repeated[0]["changed"] is False
            assert repeated[0]["successor_claim_id"] == replacement.claim_id
            unchanged = daemon.task_source.get(failed_attempt.task_cid)
            assert unchanged is not None
            assert unchanged.status == "in_progress"
            assert unchanged.revision == control_revision
            assert daemon.task_source.get_queue_entry(
                failed_attempt.task_cid
            ) is None

        # Prepared-completion recovery uses completed/succeeded rather than
        # released/succeeded.  That coherent successor is equally conclusive.
        with daemon.coordinator._lock:  # noqa: SLF001 - focused projection fixture
            connection = daemon.coordinator._require()  # noqa: SLF001
            daemon.coordinator._begin(connection)  # noqa: SLF001
            connection.execute(
                "UPDATE task_claims SET state = ?, revision = revision + 1 "
                "WHERE claim_id = ?",
                ["completed", replacement.claim_id],
            )
            connection.execute(
                "UPDATE fenced_leases SET state = ?, revision = revision + 1 "
                "WHERE lease_id = ?",
                ["completed", replacement.lease_id],
            )
            connection.execute(
                "UPDATE task_attempts SET status = ?, revision = revision + 1 "
                "WHERE attempt_id = ?",
                ["succeeded", replacement.attempt_id],
            )
            daemon.coordinator._commit_if_idle(connection)  # noqa: SLF001
        completed_projection = (
            daemon.coordinator.get_task_claim_successor_projection(
                task_cid=failed_attempt.task_cid,
                after_fencing_token=failed_attempt.fencing_token,
                after_fence_epoch=failed_attempt.fence_epoch,
            )
        )
        assert completed_projection is not None
        assert completed_projection["claim"]["state"] == "completed"
        assert completed_projection["lease"]["state"] == "completed"
        assert completed_projection["attempt"]["status"] == "succeeded"
        completed_successor = daemon._failed_attempt_coordination_successor(
            failed_attempt
        )
        assert completed_successor is not None
        assert completed_successor["claim_state"] == "completed"
        assert completed_successor["coordination_attempt_status"] == "succeeded"
    finally:
        daemon.close()


@pytest.mark.parametrize(
    "reconciliation_kind",
    ("retrying_without_queue", "typed_budget", "terminal_portal"),
)
def test_failed_attempt_persistence_race_proves_successor_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reconciliation_kind: str,
) -> None:
    now = {"ms": 1_000}
    daemon = _open_daemon(
        tmp_path,
        session=f"session:persistence-race:{reconciliation_kind}",
        lease_ms=5_000,
        max_task_attempts=1,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        failed_attempt = daemon.claim_next()
        assert failed_attempt is not None
        failed_attempt = daemon.commit_phase(failed_attempt, "context")
        if reconciliation_kind == "typed_budget":
            reason = "typed_deferral"
            typed_deferral = daemon._typed_deferral_receipt(
                failed_attempt,
                reason=reason,
            )
            body = {
                "reason": reason,
                "portal_retryable_failure": True,
                "portal_terminal_failure": False,
                "deferred": True,
                "attempt_consumed": False,
                "provider_dispatched": False,
                "typed_deferral_slot_consumed": True,
                "backoff_seconds": 300,
                "typed_deferral": typed_deferral,
            }
        elif reconciliation_kind == "terminal_portal":
            body = {
                "reason": "portal_provider_failed",
                "portal_retryable_failure": False,
                "portal_terminal_failure": True,
            }
        else:
            body = {
                "reason": "typed_deferral",
                "portal_retryable_failure": True,
                "backoff_seconds": 300,
            }
        failed_attempt = daemon.commit_phase(
            failed_attempt,
            "failed",
            body=body,
        )
        if reconciliation_kind == "retrying_without_queue":
            control = daemon.task_source.get(failed_attempt.task_cid)
            assert control is not None
            queue_reason = (
                "database_portal_retry:"
                f"{failed_attempt.attempt_id}:typed_deferral"
            )
            daemon._cas_task_status_database(
                failed_attempt.task_cid,
                expected_revision=int(control.revision),
                new_status="retrying",
                receipt={
                    "operation": "database_portal_retry",
                    "attempt_id": failed_attempt.attempt_id,
                    "claim_id": failed_attempt.claim_id,
                    "lease_id": failed_attempt.lease_id,
                    "owner_session_id": failed_attempt.owner_session_id,
                    "fencing_token": int(failed_attempt.fencing_token),
                    "fence_epoch": int(failed_attempt.fence_epoch),
                    "attempt_number": int(failed_attempt.attempt_number),
                    "execution_phase": failed_attempt.committed_phase,
                    "execution_revision": int(failed_attempt.revision),
                    "execution_finished_at_ms": failed_attempt.finished_at_ms,
                    "reason": "typed_deferral",
                    "backoff_seconds": 300,
                    "backoff_ms": 300_000,
                    "retry_not_before_ms": 0,
                    "evidence_source": "portal_retryable_failure",
                    "queue_reason": queue_reason,
                    "queue_reused": False,
                    "queue_receipt": {},
                    "coordination": {},
                    "control_expected_status": "in_progress",
                    "control_expected_revision": int(control.revision),
                },
            )

        before = daemon.task_source.get(failed_attempt.task_cid)
        assert before is not None
        assert daemon.task_source.get_queue_entry(failed_attempt.task_cid) is None
        original_execute = daemon.coordinator.execute_with_task_fence
        successor: dict[str, Any] = {}

        def advance_fence_before_transition(
            claim: Any,
            callback: Any,
            **kwargs: Any,
        ) -> Any:
            if not successor:
                now["ms"] = 7_000
                daemon.coordinator.expire_task_claim(
                    claim,
                    now_ms=now["ms"],
                )
                replacement = daemon.coordinator.claim_ready_task(
                    owner_session_id="session:persistence-race:successor",
                    lease_ms=5_000,
                    now_ms=now["ms"],
                )
                assert replacement is not None
                replacement_lease = daemon.coordinator.get_lease(
                    replacement.lease_id
                )
                assert replacement_lease is not None
                daemon.coordinator.release(
                    replacement_lease,
                    reason="shared_board_claim_conflict",
                    now_ms=now["ms"],
                )
                successor.update(replacement.to_dict())
            return original_execute(claim, callback, **kwargs)

        monkeypatch.setattr(
            daemon.coordinator,
            "execute_with_task_fence",
            advance_fence_before_transition,
        )
        outcomes = (
            daemon.reconcile_terminal_portal_failures()
            if reconciliation_kind == "terminal_portal"
            else daemon.reconcile_terminal_retry_states()
        )

        assert len(outcomes) == 1
        assert outcomes[0]["changed"] is False
        assert outcomes[0]["reason"] == "failed_attempt_coordination_superseded"
        assert outcomes[0]["successor_claim_id"] == successor["claim_id"]
        assert outcomes[0]["successor_attempt_id"] == successor["attempt_id"]
        after = daemon.task_source.get(failed_attempt.task_cid)
        assert after is not None
        assert after.status == before.status
        assert after.revision == before.revision
        assert daemon.task_source.get_queue_entry(failed_attempt.task_cid) is None
    finally:
        daemon.close()


def test_two_lane_shared_control_receipt_supersedes_token_two_with_token_six(
    tmp_path: Path,
) -> None:
    """A lane-local token 2 cannot overwrite another lane's shared token 6."""

    now = {"ms": 1_000}

    def open_lane(lane: str) -> DatabaseImplementationDaemon:
        return DatabaseImplementationDaemon(
            database_path=tmp_path / "control.duckdb",
            coordination_path=tmp_path / f"coordination-{lane}.duckdb",
            execution_path=tmp_path / f"execution-{lane}.duckdb",
            owner_session_id=f"session:{lane}",
            authority_mode="embedded",
            task_source_kind="duckdb",
            lease_ms=5_000,
            require_real_execution=True,
            clock_ms=lambda: now["ms"],
        )

    def consume_local_fences(
        daemon: DatabaseImplementationDaemon,
        count: int,
    ) -> None:
        for _ in range(count):
            claim = daemon.coordinator.claim_ready_task(
                owner_session_id=daemon.owner_session_id,
                lease_ms=5_000,
                now_ms=now["ms"],
            )
            assert claim is not None
            lease = daemon.coordinator.get_lease(claim.lease_id)
            assert lease is not None
            daemon.coordinator.release(
                lease,
                reason="test_fence_warmup",
                now_ms=now["ms"],
            )

    lane_two = open_lane("lane-2")
    lane_three = open_lane("lane-3")
    try:
        lane_two.materialize_population(_population(1))
        assert lane_two.sync_ready_tasks_into_coordination()
        assert lane_three.sync_ready_tasks_into_coordination()
        consume_local_fences(lane_two, 1)
        consume_local_fences(lane_three, 5)

        old_attempt = lane_two.claim_next()
        assert old_attempt is not None
        assert old_attempt.fencing_token == 2
        assert old_attempt.fence_epoch == 2
        old_attempt = lane_two.commit_phase(old_attempt, "context")
        old_attempt = lane_two.commit_phase(
            old_attempt,
            "failed",
            body={
                "reason": "portal_provider_failed",
                "portal_retryable_failure": False,
                "portal_terminal_failure": True,
            },
        )

        now["ms"] = 7_000
        shared = lane_two.task_source.get(old_attempt.task_cid)
        assert shared is not None
        lane_two._cas_task_status_database(
            old_attempt.task_cid,
            expected_revision=int(shared.revision),
            new_status="retrying",
            receipt={"operation": "test_operator_rearm"},
        )
        successor = lane_three.claim_next()
        assert successor is not None
        assert successor.fencing_token == 6
        assert successor.fence_epoch == 6

        before = lane_two.task_source.get(old_attempt.task_cid)
        assert before is not None
        assert before.status == "in_progress"
        before_receipt = dict(before.body["completion_receipt"])
        assert before_receipt["attempt_id"] == successor.attempt_id

        with pytest.raises(DatabaseImplementationConflictError):
            lane_two._persist_terminal_portal_failure(
                old_attempt,
                reason="portal_provider_failed",
            )
        direct_after = lane_two.task_source.get(old_attempt.task_cid)
        assert direct_after is not None
        assert direct_after.revision == before.revision
        assert dict(direct_after.body["completion_receipt"]) == before_receipt

        outcomes = lane_two.reconcile_terminal_portal_failures()
        assert len(outcomes) == 1
        assert outcomes[0]["changed"] is False
        assert outcomes[0]["reason"] == "failed_attempt_control_superseded"
        assert outcomes[0]["successor_attempt_id"] == successor.attempt_id
        assert outcomes[0]["successor_claim_id"] == successor.claim_id
        assert outcomes[0]["successor_fencing_token"] == 6
        assert outcomes[0]["successor_fence_epoch"] == 6

        after = lane_two.task_source.get(old_attempt.task_cid)
        assert after is not None
        assert after.revision == before.revision
        assert dict(after.body["completion_receipt"]) == before_receipt
        assert lane_two.task_source.get_queue_entry(old_attempt.task_cid) is None
    finally:
        lane_three.close()
        lane_two.close()


def test_foreign_control_injected_before_atomic_retry_leaves_no_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The owner guard rejects a foreign receipt before its queue mutation."""

    daemon = _open_daemon(
        tmp_path,
        session="session:atomic-control-race",
    )
    try:
        daemon.materialize_population(_population(1))
        failed_attempt = daemon.claim_next()
        assert failed_attempt is not None
        failed_attempt = daemon.commit_phase(failed_attempt, "context")
        failed_attempt = daemon.commit_phase(
            failed_attempt,
            "failed",
            body={
                "reason": "typed_deferral",
                "portal_retryable_failure": True,
                "backoff_seconds": 300,
            },
        )
        original = daemon.task_source.record_queue_backoff_and_cas_status
        injected: dict[str, Any] = {}

        def inject_foreign_control(**kwargs: Any) -> Any:
            if not injected:
                current = daemon.task_source.get(failed_attempt.task_cid)
                assert current is not None
                foreign_reason = "database_portal_retry:attempt:foreign:capacity"
                daemon._cas_task_status_database(
                    failed_attempt.task_cid,
                    expected_revision=int(current.revision),
                    new_status="retrying",
                    receipt={
                        "operation": "database_portal_retry",
                        "attempt_id": "attempt:foreign",
                        "claim_id": "claim:foreign",
                        "lease_id": "lease:foreign",
                        "owner_session_id": "session:foreign-lane",
                        "fencing_token": 6,
                        "fence_epoch": 6,
                        "queue_reason": foreign_reason,
                    },
                )
                injected["revision"] = int(current.revision) + 1
            return original(**kwargs)

        monkeypatch.setattr(
            daemon.task_source,
            "record_queue_backoff_and_cas_status",
            inject_foreign_control,
        )
        outcomes = daemon.reconcile_terminal_retry_states()

        assert len(outcomes) == 1
        assert outcomes[0]["changed"] is False
        assert outcomes[0]["reason"] == "failed_attempt_control_superseded"
        assert outcomes[0]["successor_attempt_id"] == "attempt:foreign"
        assert outcomes[0]["successor_fencing_token"] == 6
        assert daemon.task_source.get_queue_entry(failed_attempt.task_cid) is None
        current = daemon.task_source.get(failed_attempt.task_cid)
        assert current is not None
        assert current.revision == injected["revision"]
        assert current.body["completion_receipt"]["attempt_id"] == (
            "attempt:foreign"
        )
    finally:
        daemon.close()


def test_retired_lane_observes_complete_foreign_generic_retry_without_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A foreign generic retry supersedes a retired lane's local failure."""

    source = _open_daemon(
        tmp_path,
        session="session:retired-retry-source",
        lane="retired-retry-source",
    )
    try:
        source.materialize_population(_population(1))
        local = source.claim_next()
        assert local is not None
        local = source.commit_phase(local, ATTEMPT_PHASE_CONTEXT)
        local = source.commit_phase(
            local,
            ATTEMPT_PHASE_FAILED,
            body={
                "reason": "transient_provider_failure",
                "portal_retryable_failure": True,
                "backoff_seconds": 0,
            },
        )
        source_retry = source.reconcile_terminal_retry_states()
        assert len(source_retry) == 1
        assert source_retry[0]["status"] == "retrying"
    finally:
        source.close()

    shutil.copy2(
        tmp_path / "execution-retired-retry-source.duckdb",
        tmp_path / "execution-retired-retry-observer.duckdb",
    )

    foreign = _open_daemon(
        tmp_path,
        session="session:foreign-retry-owner",
        lane="foreign-retry-owner",
    )
    try:
        successor = foreign.claim_next()
        assert successor is not None
        assert successor.attempt_id != local.attempt_id
        successor = foreign.commit_phase(successor, ATTEMPT_PHASE_CONTEXT)
        successor = foreign.commit_phase(
            successor,
            ATTEMPT_PHASE_FAILED,
            body={
                "reason": "second_transient_provider_failure",
                "portal_retryable_failure": True,
                "backoff_seconds": 0,
            },
        )
        foreign_retry = foreign.reconcile_terminal_retry_states()
        assert len(foreign_retry) == 1
        assert foreign_retry[0]["status"] == "retrying"
    finally:
        foreign.close()

    observer = _open_daemon(
        tmp_path,
        session="session:retired-retry-observer",
        lane="retired-retry-observer",
    )
    try:
        observed_local = observer.get_attempt(local.attempt_id)
        assert observed_local is not None
        assert observer.coordinator.get_task_claim(local.claim_id) is None
        before = observer.task_source.get(local.task_cid)
        assert before is not None
        assert before.status == "retrying"
        receipt = dict(before.body["completion_receipt"])
        assert receipt["operation"] == "database_portal_retry"
        assert receipt["attempt_id"] == successor.attempt_id
        queue = observer.task_source.get_queue_entry(local.task_cid)
        assert queue is not None
        queue_before = queue.to_dict()

        original_get = observer.task_source.get
        projected_receipt = {"value": dict(receipt)}

        def get_with_projected_receipt(task_cid: str) -> object:
            task = original_get(task_cid)
            if task_cid != local.task_cid or task is None:
                return task
            return SimpleNamespace(
                task_alias=task.task_alias,
                status=task.status,
                revision=task.revision,
                body={
                    **dict(task.body),
                    "completion_receipt": projected_receipt["value"],
                },
            )

        monkeypatch.setattr(
            observer.task_source,
            "get",
            get_with_projected_receipt,
        )
        minimal_receipt = {
            key: receipt[key]
            for key in (
                "operation",
                "attempt_id",
                "claim_id",
                "lease_id",
                "owner_session_id",
                "fencing_token",
                "fence_epoch",
            )
        }
        missing_execution = dict(receipt)
        missing_execution.pop("execution_revision")
        malformed_receipts = [
            minimal_receipt,
            missing_execution,
            {**receipt, "claim_id": local.claim_id},
            {**receipt, "attempt_number": True},
            {**receipt, "execution_finished_at_ms": 0},
            {
                **receipt,
                "coordination": {
                    **dict(receipt["coordination"]),
                    "claim_id": "claim:forged-coordination",
                },
            },
            {
                **receipt,
                "control_expected_revision": int(before.revision),
            },
            {**receipt, "control_expected_status": "completed"},
            {**receipt, "queue_reason": "database_portal_retry:forged"},
            {
                **receipt,
                "retry_not_before_ms": int(receipt["retry_not_before_ms"])
                + 1,
            },
        ]
        for malformed in malformed_receipts:
            projected_receipt["value"] = malformed
            with pytest.raises(
                DatabaseImplementationAuthorityError,
                match="has no coordination claim",
            ):
                observer.reconcile_terminal_retry_states()
        monkeypatch.setattr(observer.task_source, "get", original_get)

        reconciled = observer.reconcile_terminal_retry_states()
        assert len(reconciled) == 1
        assert reconciled[0]["changed"] is False
        assert reconciled[0]["reason"] == "failed_attempt_control_superseded"
        assert reconciled[0]["attempt_id"] == local.attempt_id
        assert reconciled[0]["successor_attempt_id"] == successor.attempt_id
        assert reconciled[0]["successor_claim_id"] == successor.claim_id
        assert reconciled[0]["control_status"] == "retrying"
        assert reconciled[0]["control_operation"] == "database_portal_retry"

        after = observer.task_source.get(local.task_cid)
        assert after is not None
        assert after.to_dict() == before.to_dict()
        queue_after = observer.task_source.get_queue_entry(local.task_cid)
        assert queue_after is not None
        assert queue_after.to_dict() == queue_before
    finally:
        observer.close()


def test_completion_owner_cas_rejects_foreign_shared_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:completion-control-race",
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        attempt = daemon.commit_phase(attempt, "context")
        attempt = daemon.commit_phase(attempt, "provider")
        attempt = daemon.commit_phase(attempt, "effect")
        validation = {
            "outcome": "passed",
            "evidence_digest": "sha256:" + "d" * 64,
            "argv": ["completion-control-race"],
        }
        attempt = daemon.commit_phase(
            attempt,
            "validation",
            body=validation,
        )
        original_cas = daemon._cas_task_status_database
        injected = {"done": False}

        def inject_before_completion_cas(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("new_status") == "completed" and not injected["done"]:
                injected["done"] = True
                current = daemon.task_source.get(attempt.task_cid)
                assert current is not None
                original_cas(
                    attempt.task_cid,
                    expected_revision=int(current.revision),
                    new_status="retrying",
                    receipt={
                        "operation": "database_portal_retry",
                        "attempt_id": "attempt:foreign-completion",
                        "claim_id": "claim:foreign-completion",
                        "lease_id": "lease:foreign-completion",
                        "owner_session_id": "session:foreign-completion",
                        "fencing_token": 6,
                        "fence_epoch": 6,
                        "queue_reason": "test:foreign-completion",
                    },
                )
                retrying = daemon.task_source.get(attempt.task_cid)
                assert retrying is not None
                original_cas(
                    attempt.task_cid,
                    expected_revision=int(retrying.revision),
                    new_status="in_progress",
                    receipt={
                        "operation": "database_claim",
                        "attempt_id": "attempt:foreign-completion",
                        "claim_id": "claim:foreign-completion",
                        "lease_id": "lease:foreign-completion",
                        "owner_session_id": "session:foreign-completion",
                        "fencing_token": 6,
                        "fence_epoch": 6,
                    },
                )
            return original_cas(*args, **kwargs)

        monkeypatch.setattr(
            daemon,
            "_cas_task_status_database",
            inject_before_completion_cas,
        )
        with pytest.raises(database_task_source_module.TaskSourceConflictError):
            daemon.complete_attempt(attempt, validation_result=validation)

        shared = daemon.task_source.get(attempt.task_cid)
        assert shared is not None
        assert shared.status == "in_progress"
        assert shared.body["completion_receipt"]["attempt_id"] == (
            "attempt:foreign-completion"
        )
        execution = daemon.get_attempt(attempt.attempt_id)
        assert execution is not None
        assert execution.status == "running"
        assert not execution.phase_committed(ATTEMPT_PHASE_COMPLETE)
    finally:
        daemon.close()
def test_provider_result_is_rejected_after_fenced_takeover(tmp_path: Path) -> None:
    now = {"ms": 1_000}
    holder: dict[str, DatabaseImplementationDaemon] = {}
    replacement_claim_ids: list[str] = []

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        daemon = holder["daemon"]
        # Cross the renewed deadline and let another session claim the same
        # ready coordination task before this provider result is returned.
        now["ms"] = 7_000
        replacement = daemon.coordinator.claim_ready_task(
            owner_session_id="session:replacement",
            lease_ms=5_000,
            now_ms=now["ms"],
        )
        assert replacement is not None
        assert replacement.task_cid == attempt.task_cid
        replacement_claim_ids.append(replacement.claim_id)
        return {"status": "ok", "accepted": True, "task_cid": attempt.task_cid}

    daemon = _open_daemon(
        tmp_path,
        session="session:stale-provider",
        provider_fn=provider,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    holder["daemon"] = daemon
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        attempt = daemon.commit_phase(attempt, "context")
        with pytest.raises(DatabaseCoordinationError):
            daemon.run_provider(attempt)
        assert replacement_claim_ids
        intent = daemon.provider_invocation_recorded(
            attempt.attempt_id,
            idempotency_key=f"provider:{attempt.attempt_id}",
        )
        assert intent is not None
        assert intent["schema"] == DATABASE_PROVIDER_CALLBACK_UNKNOWN_SCHEMA
        assert intent["callback_state"] == "started_outcome_unknown"
        assert intent["provider_effect_state"] == "unknown_may_have_started"
        stored = daemon.get_attempt(attempt.attempt_id)
        assert stored is not None
        assert stored.committed_phase == "context"
        assert stored.status == "running"
    finally:
        daemon.close()


def test_expired_attempt_cannot_commit_logical_completion(tmp_path: Path) -> None:
    now = {"ms": 1_000}
    daemon = _open_daemon(
        tmp_path,
        session="session:expired-completion",
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        for phase in ("context", "provider", "effect", "validation"):
            attempt = daemon.commit_phase(attempt, phase)
        now["ms"] = 6_000
        with pytest.raises(DatabaseCoordinationError):
            daemon.complete_attempt(
                attempt,
                validation_result={
                    "outcome": "passed",
                    "evidence_digest": "sha256:" + "a" * 64,
                    "argv": ["focused-validation"],
                },
            )
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "in_progress"
        assert daemon.coordinator.claimability(attempt.task_cid)["claimable"] is True
        stored = daemon.get_attempt(attempt.attempt_id)
        assert stored is not None
        assert stored.committed_phase == "validation"
        assert stored.status == "running"
    finally:
        daemon.close()


def test_restart_retires_prepared_absent_expired_attempt_then_refences_retry(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    first = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        first.materialize_population(_population(1))
        old_attempt = first.claim_next()
        assert old_attempt is not None
        old_attempt = first.commit_phase(old_attempt, "context")
        old_attempt, _, duplicated = first.run_provider(old_attempt)
        assert duplicated is False
        old_owner = first.owner_session_id
    finally:
        first.close()

    # No intervening coordinator mutation performs an expiry sweep.
    now["ms"] = 7_000
    replacement = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        assert replacement.owner_session_id == old_owner
        result = replacement.run_once()
        reconciliations = result["expired_attempt_reconciliations"]
        assert len(reconciliations) == 1
        assert reconciliations[0]["status"] == "expired"
        assert reconciliations[0]["provider_evidence_reused"] is False
        assert reconciliations[0]["effect_evidence_reused"] is False
        assert result["attempt_id"] != old_attempt.attempt_id
        assert result["implementation_result"]["status"] == "succeeded"
        assert provider_calls == [old_attempt.task_cid, old_attempt.task_cid]
        assert effect_calls == [old_attempt.task_cid]
        retired = replacement.get_attempt(old_attempt.attempt_id)
        assert retired is not None
        assert retired.status == "failed"
        assert retired.committed_phase == "failed"
        replacement_claim = replacement.coordinator.get_task_claim(
            result["claim_id"]
        )
        assert replacement_claim is not None
        assert replacement_claim.fencing_token > old_attempt.fencing_token
    finally:
        replacement.close()


def test_completed_control_cas_is_recovered_from_prepared_barrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expiry after control CAS cannot expose an uncoordinated completion."""

    now = {"ms": 1_000}
    daemon = _open_daemon(
        tmp_path,
        session="session:prepared-recovery",
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        for phase in ("context", "provider", "effect", "validation"):
            attempt = daemon.commit_phase(attempt, phase)

        original_complete = daemon.coordinator.complete_task_claim

        def expire_at_promotion(*args: object, **kwargs: object) -> object:
            now["ms"] = 7_000
            kwargs["now_ms"] = now["ms"]
            return original_complete(*args, **kwargs)

        monkeypatch.setattr(
            daemon.coordinator,
            "complete_task_claim",
            expire_at_promotion,
        )
        with pytest.raises(DatabaseCoordinationError):
            daemon.complete_attempt(
                attempt,
                validation_result={
                    "outcome": "passed",
                    "evidence_digest": "sha256:" + "b" * 64,
                    "argv": ["focused-validation"],
                },
            )

        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "completed"
        assert task.revision == 3
        readiness = daemon.coordinator.claimability(attempt.task_cid)
        assert readiness["claimable"] is False
        assert readiness["completion_status"] == "prepared"
        prepared = daemon.coordinator.get_prepared_task_completion(
            attempt.task_cid
        )
        assert prepared is not None
        assert prepared["attempt_id"] == attempt.attempt_id
        stored = daemon.get_attempt(attempt.attempt_id)
        assert stored is not None
        assert stored.status == "running"
        assert stored.committed_phase == "validation"

        # Restore the ordinary method.  The next pass proves the exact control
        # receipt, promotes and settles the expired preparation, and repairs
        # the execution projection without rerunning provider/effect work.
        monkeypatch.setattr(
            daemon.coordinator,
            "complete_task_claim",
            original_complete,
        )
        result = daemon.run_once()
        assert result["unchanged"] is False
        assert result["write_count"] == 1
        assert len(result["completion_reconciliations"]) == 1
        assert result["completion_reconciliations"][0]["recovered"] is True
        recovered = daemon.get_attempt(attempt.attempt_id)
        assert recovered is not None
        assert recovered.status == "succeeded"
        assert recovered.committed_phase == ATTEMPT_PHASE_COMPLETE
        claim = daemon.coordinator.get_task_claim(attempt.claim_id)
        assert claim is not None
        assert claim.state.value == "completed"
    finally:
        daemon.close()


def test_restart_recovers_prepared_control_completion_without_prior_expiry_sweep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = {"ms": 1_000}
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    first = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        for phase in ("context", "provider", "effect", "validation"):
            attempt = first.commit_phase(attempt, phase)

        def crash_before_promotion(*args: object, **kwargs: object) -> object:
            now["ms"] = 7_000
            raise RuntimeError("simulated crash before coordination promotion")

        monkeypatch.setattr(
            first.coordinator,
            "complete_task_claim",
            crash_before_promotion,
        )
        with pytest.raises(RuntimeError, match="before coordination promotion"):
            first.complete_attempt(
                attempt,
                validation_result={
                    "outcome": "passed",
                    "evidence_digest": "sha256:" + "e" * 64,
                    "argv": ["focused-validation"],
                },
            )
        task = first.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "completed"
        unswept = first.coordinator.get_task_claim(attempt.claim_id)
        assert unswept is not None
        assert unswept.state.value == "accepted"
    finally:
        first.close()

    replacement = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        result = replacement.run_once()
        assert result["unchanged"] is False
        assert result["write_count"] == 1
        assert len(result["completion_reconciliations"]) == 1
        assert result["completion_reconciliations"][0]["recovered"] is True
        recovered = replacement.get_attempt(attempt.attempt_id)
        assert recovered is not None
        assert recovered.status == "succeeded"
        claim = replacement.coordinator.get_task_claim(attempt.claim_id)
        assert claim is not None
        assert claim.state.value == "completed"
        assert provider_calls == []
        assert effect_calls == []
    finally:
        replacement.close()


def test_promoted_completion_replays_after_local_phase_response_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        session="session:promotion-replay",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        for phase in ("context", "provider", "effect", "validation"):
            attempt = daemon.commit_phase(attempt, phase)

        original_commit_phase = daemon.commit_phase

        def lose_local_complete(
            current: DatabaseTaskAttempt | str,
            phase: str,
            **kwargs: object,
        ) -> DatabaseTaskAttempt:
            if phase == ATTEMPT_PHASE_COMPLETE:
                raise RuntimeError("simulated local COMPLETE outage")
            return original_commit_phase(current, phase, **kwargs)

        monkeypatch.setattr(daemon, "commit_phase", lose_local_complete)
        with pytest.raises(RuntimeError, match="simulated local COMPLETE outage"):
            daemon.complete_attempt(
                attempt,
                validation_result={
                    "outcome": "passed",
                    "evidence_digest": "sha256:" + "d" * 64,
                    "argv": ["focused-validation"],
                },
            )
        promoted = daemon.coordinator.get_prepared_task_completion(
            attempt.task_cid
        )
        assert promoted is not None
        assert promoted["status"] == "succeeded"
        claim = daemon.coordinator.get_task_claim(attempt.claim_id)
        assert claim is not None
        assert claim.state.value == "accepted"
        stored = daemon.get_attempt(attempt.attempt_id)
        assert stored is not None
        assert stored.status == "running"
        assert stored.committed_phase == "validation"

        monkeypatch.setattr(daemon, "commit_phase", original_commit_phase)
        result = daemon.run_once()
        assert result["unchanged"] is False
        assert result["write_count"] == 1
        assert result["implementation_result"] is None
        assert len(result["completion_reconciliations"]) == 1
        repaired = daemon.get_attempt(attempt.attempt_id)
        assert repaired is not None
        assert repaired.status == "succeeded"
        assert repaired.committed_phase == ATTEMPT_PHASE_COMPLETE
        assert provider_calls == []
        assert effect_calls == []
        settled = daemon.coordinator.get_task_claim(attempt.claim_id)
        assert settled is not None
        assert settled.state.value == "released"
    finally:
        daemon.close()


def test_expired_preparation_without_control_cas_is_aborted_and_requeued(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = {"ms": 1_000}
    daemon = _open_daemon(
        tmp_path,
        session="session:prepared-abort",
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        for phase in ("context", "provider", "effect", "validation"):
            attempt = daemon.commit_phase(attempt, phase)

        original_cas = daemon._cas_task_status_database

        def reject_control_completion(*args: object, **kwargs: object) -> object:
            raise RuntimeError("simulated control CAS outage")

        monkeypatch.setattr(
            daemon,
            "_cas_task_status_database",
            reject_control_completion,
        )
        with pytest.raises(RuntimeError, match="simulated control CAS outage"):
            daemon.complete_attempt(
                attempt,
                validation_result={
                    "outcome": "passed",
                    "evidence_digest": "sha256:" + "c" * 64,
                    "argv": ["focused-validation"],
                },
            )
        assert daemon.coordinator.get_prepared_task_completion(
            attempt.task_cid
        ) is not None
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "in_progress"

        monkeypatch.setattr(daemon, "_cas_task_status_database", original_cas)
        now["ms"] = 7_000
        result = daemon.run_once()
        assert len(result["completion_reconciliations"]) == 1
        assert result["completion_reconciliations"][0]["status"] == "aborted"
        assert result["implementation_result"]["status"] == "succeeded"
        assert result["attempt_id"] != attempt.attempt_id
        old_attempt = daemon.get_attempt(attempt.attempt_id)
        assert old_attempt is not None
        assert old_attempt.status == "failed"
        final_completion = daemon.coordinator.get_prepared_task_completion(
            attempt.task_cid
        )
        assert final_completion is not None
        assert final_completion["status"] == "succeeded"
        assert final_completion["attempt_id"] == result["attempt_id"]
        completed = daemon.task_source.get(attempt.task_cid)
        assert completed is not None
        assert completed.status == "completed"
    finally:
        daemon.close()


def test_task_claim_settlement_authority_loss_is_not_suppressed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(tmp_path, session="session:settlement-loss")
    try:
        daemon.materialize_population(_population(1))

        def reject_settlement(*args: object, **kwargs: object) -> object:
            raise DatabaseCoordinationError("simulated settlement authority loss")

        monkeypatch.setattr(
            daemon.coordinator,
            "settle_task_claim",
            reject_settlement,
        )
        with pytest.raises(
            DatabaseCoordinationError,
            match="simulated settlement authority loss",
        ):
            daemon.run_once()
    finally:
        daemon.close()


@pytest.mark.parametrize(
    ("restart_ms", "expected_claim_state"),
    ((2_000, "released"), (7_000, "completed")),
)
def test_restart_settles_promoted_completion_after_local_complete_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    restart_ms: int,
    expected_claim_state: str,
) -> None:
    now = {"ms": 1_000}
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    first = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        first.materialize_population(_population(1))

        def crash_before_settlement(*args: object, **kwargs: object) -> object:
            raise RuntimeError("simulated crash before claim settlement")

        monkeypatch.setattr(
            first.coordinator,
            "settle_task_claim",
            crash_before_settlement,
        )
        with pytest.raises(RuntimeError, match="before claim settlement"):
            first.run_once()
        row = first._require_connection().execute(
            """
            SELECT attempt_id, claim_id FROM database_task_attempts
            WHERE status = 'succeeded'
            """
        ).fetchone()
        assert row is not None
        attempt_id, claim_id = str(row[0]), str(row[1])
        unsettled = first.coordinator.get_task_claim(claim_id)
        assert unsettled is not None
        assert unsettled.state.value == "accepted"
    finally:
        first.close()

    now["ms"] = restart_ms
    replacement = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        result = replacement.run_once()
        assert result["unchanged"] is False
        assert result["write_count"] == 1
        assert len(result["completion_reconciliations"]) == 1
        assert result["completion_reconciliations"][0]["status"] == "succeeded"
        settled = replacement.coordinator.get_task_claim(claim_id)
        assert settled is not None
        assert settled.state.value == expected_claim_state
        local = replacement.get_attempt(attempt_id)
        assert local is not None
        assert local.status == "succeeded"
        assert replacement.coordinator.list_unsettled_task_completions() == []
        assert provider_calls == ["task:cid:001"]
        assert effect_calls == ["task:cid:001"]
    finally:
        replacement.close()


def test_automatic_run_once_never_claims_manual_or_review_only_task(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(tmp_path)
    try:
        population = _population(2)
        tasks = population["tasks"]
        assert isinstance(tasks, list)
        tasks[0]["completion"] = "manual"
        tasks[1]["review_only"] = True
        daemon.materialize_population(population)
        result = daemon.run_once()
        assert result["unchanged"] is True
        assert result["selection_idle_reason"] == "no_ready_tasks"
        assert daemon.list_running_attempts() == []
        assert daemon.coordinator.get_task_claim("claim:missing") is None
        for task_cid in ("task:cid:001", "task:cid:002"):
            task = daemon.task_source.get(task_cid)
            assert task is not None
            assert task.status == "ready"

        # The coordinator still exposes the task to a separately authorized
        # trusted manual-seal path; only automatic daemon dispatch is excluded.
        direct = daemon.coordinator.claim_task(
            task_cid="task:cid:001",
            owner_session_id="session:trusted-manual-seal",
            now_ms=daemon._now_ms(),
        )
        assert direct.task_cid == "task:cid:001"
    finally:
        daemon.close()


def test_parse_args_accepts_database_authority_flags() -> None:
    args = parse_args(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "embedded",
            "--database-path",
            "/tmp/control.duckdb",
            "--owner-session-id",
            "session:cli",
            "--once",
        ]
    )
    assert args.task_source_kind == "duckdb"
    assert args.authority_mode == "embedded"
    assert Path(args.database_path) == Path("/tmp/control.duckdb")
    assert args.owner_session_id == "session:cli"
    paths = resolve_database_implementation_paths(args)
    assert paths["database_path"] == Path("/tmp/control.duckdb")


def test_runner_builds_database_daemon_without_json_projections(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "control.duckdb"
    args = parse_args(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "embedded",
            "--database-path",
            str(database_path),
            "--todo-path",
            str(tmp_path / "unused.md"),
            "--state-dir",
            str(tmp_path / "state"),
            "--state-prefix",
            "dqp",
            "--task-shard-count",
            "4",
            "--task-shard-index",
            "3",
            "--strict-task-sharding",
            "--max-task-attempts",
            "3",
            "--once",
        ]
    )
    daemon = build_database_implementation_daemon_from_args(
        args,
        owner_session_id="session:runner",
    )
    try:
        assert isinstance(daemon, DatabaseImplementationDaemon)
        assert daemon.state_path is None
        assert daemon.events_path is None
        assert daemon.max_task_attempts == 3
        assert daemon.projections_required() is False
        assert daemon.task_shard_count == 4
        assert daemon.task_shard_index == 3
        assert daemon.strict_task_sharding is True
        daemon.materialize_population(_population(1))
        result = daemon.run_once()
        assert result["authority_mode"] == "embedded"
        assert result["markdown_status_writes"] == 0
    finally:
        daemon.close()


@pytest.mark.parametrize(
    ("field_name", "malformed", "message"),
    [
        ("task_shard_count", True, "positive integer"),
        ("task_shard_index", False, "range"),
        ("strict_task_sharding", 1, "boolean"),
    ],
)
def test_database_runner_preserves_exact_shard_types_for_constructor_guard(
    tmp_path: Path,
    field_name: str,
    malformed: object,
    message: str,
) -> None:
    args = parse_args(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "embedded",
            "--database-path",
            str(tmp_path / "control.duckdb"),
            "--todo-path",
            str(tmp_path / "unused.md"),
            "--once",
        ]
    )
    setattr(args, field_name, malformed)
    with pytest.raises(ValueError, match=message):
        build_database_implementation_daemon_from_args(args)


def test_runner_portal_builder_selects_database_daemon(tmp_path: Path) -> None:
    database_path = tmp_path / "control.duckdb"
    args = parse_args(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "embedded",
            "--database-path",
            str(database_path),
            "--todo-path",
            str(tmp_path / "board.md"),
            "--state-dir",
            str(tmp_path / "state"),
            "--state-prefix",
            "dqp",
            "--task-shard-count",
            "2",
            "--task-shard-index",
            "1",
            "--strict-task-sharding",
            "--max-task-attempts",
            "4",
            "--implement",
            "--once",
        ]
    )
    daemon, context = build_portal_implementation_daemon_from_args(
        args,
        repo_root=tmp_path,
    )
    try:
        assert isinstance(daemon, DatabaseImplementationDaemon)
        assert daemon.max_task_attempts == 4
        assert context.state_path.name.startswith("dqp_")
        assert daemon.task_shard_count == 2
        assert daemon.task_shard_index == 1
        assert daemon.strict_task_sharding is True
        assert daemon.require_real_execution is True
    finally:
        daemon.close()

def test_provider_cold_execution_schema_installer_matches_daemon_contract(
    tmp_path: Path,
) -> None:
    """The bootstrap DDL stays provider-cold and is the daemon's exact DDL."""

    database_path = tmp_path / "execution.duckdb"
    program = """
import json
import sys
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_execution_schema import (
    install_database_execution_schema,
)

receipt = install_database_execution_schema(
    Path(sys.argv[1]),
    metadata={
        "authority_mode": "embedded",
        "logical_owner_session_id": "session:test:logical-owner",
        "process_instance_id": "process:test:bootstrap",
        "state_schema_revision": "datasets-authoritative-operational-v1",
        "control_schema_profile_id": "profile:test",
        "control_schema_fingerprint": "sha256:" + "a" * 64,
    },
)
forbidden = sorted(
    name
    for name in sys.modules
    if name == "urllib.request"
    or "llm_router" in name
    or ".providers." in name
    or name.split(".", 1)[0] in {"anthropic", "openai"}
)
print(json.dumps({"forbidden": forbidden, "receipt": receipt}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program, str(database_path)],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    observed = json.loads(completed.stdout)
    assert observed["forbidden"] == []
    assert observed["receipt"]["tables"] == [
        "daemon_execution_metadata",
        "database_task_attempts",
        "attempt_phases",
        "provider_invocations",
        "effect_claims",
        "daemon_execution_events",
    ]

    schema_module = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.database_execution_schema"
    )
    daemon_module = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon"
    )
    assert daemon_module._DAEMON_EXECUTION_SQL == schema_module.DAEMON_EXECUTION_SQL

    duckdb = pytest.importorskip("duckdb")
    connection = connect_duckdb_with_policy(
        duckdb,
        database_path,
        read_only=True,
    )
    try:
        metadata = dict(
            connection.execute(
                "SELECT key, value FROM daemon_execution_metadata ORDER BY key"
            ).fetchall()
        )
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            ).fetchall()
        }
    finally:
        connection.close()
    assert metadata == observed["receipt"]["metadata"]
    assert tables == set(observed["receipt"]["tables"])


def _validation_retry_receipt(
    daemon: DatabaseImplementationDaemon,
    attempt: DatabaseTaskAttempt,
) -> dict[str, object]:
    receipt: dict[str, object] = {
        "schema": DATABASE_PORTAL_VALIDATION_RETRY_SCHEMA,
        "disposition": "retry",
        "reason": "declared_validation_failed",
        "task_cid": attempt.task_cid,
        "task_alias": attempt.task_alias,
        "attempt_id": attempt.attempt_id,
        "claim_id": attempt.claim_id,
        "lease_id": attempt.lease_id,
        "attempt_number": attempt.attempt_number,
        "fencing_token": attempt.fencing_token,
        "fence_epoch": attempt.fence_epoch,
        "portal_attempt": 1,
        "typed_retry_generation": 1,
        "retry_budget_basis": "portal_attempt",
        "legacy_database_attempts_excluded": True,
        "max_task_attempts": daemon.max_task_attempts,
        "remaining_task_attempts": (
            daemon.max_task_attempts - 1
        ),
        "attempt_consumed": True,
        "provider_dispatched": True,
        "backoff_seconds": 0,
        "implementation_commit": "a" * 40,
        "rescue_branch": "rescue/dqp-t001-attempt-1-failed-validation",
        "binding_id": "sha256:" + "2" * 64,
        "events_digest": "sha256:" + "3" * 64,
        "event_stream_id": "event-log:validation-retry",
        "expected_output_event_id": "sha256:" + "1" * 64,
        "proposal_event_id": "sha256:" + "4" * 64,
        "preservation_event_id": "sha256:" + "5" * 64,
        "implementation_event_id": "sha256:" + "6" * 64,
        "proposal_id": "proposal:validation-retry",
        "proposal_receipt_id": "proposal-receipt:validation-retry",
        "proposal_policy_id": "proposal-policy:validation-retry",
        "validation_receipt_id": "validation-dag:validation-retry",
        "failure_review_receipt_id": "failure-review:validation-retry",
        "changed_paths": ["implementation.py", "test_implementation.py"],
        "authoritative_validation_executed": True,
        "proposal_policy_accepted": True,
        "output_policy_passed": True,
        "denial_findings": [],
    }
    receipt["receipt_id"] = daemon._database_portal_evidence_digest(receipt)
    return receipt


def test_direct_selector_never_bypasses_cooldown_when_all_ready_are_cooled() -> None:
    daemon = object.__new__(DatabaseImplementationDaemon)
    daemon.merge_queue = SimpleNamespace(
        has_pending_for_task=lambda _task_cid: False
    )
    daemon.degradation_state = SimpleNamespace(
        degraded_submodules=lambda: []
    )
    daemon.task_queue = SimpleNamespace(
        is_cooled_down=lambda _task_cid: True,
        record_selection=lambda _task_cid: pytest.fail(
            "cooled task was selected"
        ),
    )
    daemon._canonical_ref = lambda task: f"task:cid:{task.task_id}"
    daemon._inflight_submodule_paths = lambda: set()
    task = SimpleNamespace(
        task_id="COOLED-001",
        priority="P0",
        track="implementation",
        depends_on=[],
        metadata={},
    )

    selected = daemon._select_next_task(
        [task],
        {task.task_id: "ready"},
        {},
        {},
        {},
    )

    assert selected is None


def test_materialization_projects_completed_prerequisites_into_coordination(
    tmp_path: Path,
) -> None:
    population = _population(2)
    tasks = population["tasks"]
    assert isinstance(tasks, list)
    tasks[0]["status"] = "completed"
    tasks[1]["dependencies"] = ["task:cid:001"]
    daemon = _open_daemon(tmp_path, session="session:bootstrap")
    try:
        receipt = daemon.materialize_population(population)
        assert receipt["bootstrap_completed_task_cids"] == ["task:cid:001"]
        projection = daemon.coordinator.coordination_registry_projection()
        assert projection["counts"]["logical_completions"] == 1

        attempt = daemon.claim_next()
        assert attempt is not None
        assert attempt.task_cid == "task:cid:002"
    finally:
        daemon.close()


def test_fresh_coordination_sidecar_projects_canonical_completed_dependencies(
    tmp_path: Path,
) -> None:
    population = _population(2)
    tasks = population["tasks"]
    assert isinstance(tasks, list)
    tasks[0]["status"] = "completed"
    tasks[1]["dependencies"] = ["task:cid:001"]

    seed = _open_daemon(tmp_path, session="session:materializer")
    try:
        seed.materialize_population(population)
    finally:
        seed.close()

    # Runtime lanes intentionally use fresh, disposable coordination stores.
    # Their claimability must be reconstructed from the canonical ready
    # frontier rather than depending on materializer-local completion rows.
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "fresh-coordination.duckdb",
        execution_path=tmp_path / "fresh-execution.duckdb",
        owner_session_id="session:fresh-sidecar",
        authority_mode="embedded",
        task_source_kind="duckdb",
        require_real_execution=True,
    )
    try:
        assert daemon.sync_ready_tasks_into_coordination() == ["task:cid:002"]
        assert daemon.sync_ready_tasks_into_coordination() == ["task:cid:002"]
        before_claim = daemon.coordinator.coordination_registry_projection()
        assert before_claim["counts"]["logical_completions"] == 1

        attempt = daemon.claim_next()
        assert attempt is not None
        assert attempt.task_cid == "task:cid:002"
        projection = daemon.coordinator.coordination_registry_projection()
        completions = {
            item["task_cid"]: item["status"]
            for item in projection["logical_completions"]
        }
        assert completions == {"task:cid:001": "succeeded"}
    finally:
        daemon.close()


def test_claim_next_preserves_canonical_ready_order_for_late_task(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(tmp_path, session="session:ordered")
    try:
        population = _population(1)
        first = population["tasks"]
        assert isinstance(first, list)
        first[0]["ordinal"] = 20
        daemon.materialize_population(population)

        # The successor task enters the coordination registry later, but its
        # canonical plan ordinal places it first.  Registration time must not
        # override the intent repository's ready-order authority.
        daemon.task_source._intent.upsert_task(
            task_cid="task:cid:late-preferred",
            task_alias="DQP-LATE-PREFERRED",
            goal_cid="goal:cid:root",
            ordinal=1,
            status="ready",
            priority="P0",
            body={"title": "Late but plan-preferred"},
            identity={"task_cid": "task:cid:late-preferred"},
            dependencies=(),
            outputs=(),
            acceptance=(),
            validations=(),
            expected_revision=0,
        )

        attempt = daemon.claim_next()
        assert attempt is not None
        assert attempt.task_cid == "task:cid:late-preferred"
    finally:
        daemon.close()


def test_database_observer_without_real_execution_never_resumes_or_claims(
    tmp_path: Path,
) -> None:
    """Inherited store authority cannot replace the explicit execution permit."""

    seed = _open_daemon(tmp_path)
    try:
        seed.materialize_population(_population(2))
        running = seed.claim_next()
        assert running is not None
        running = seed.commit_phase(
            running,
            "context",
            body={"source": "pre-reload-real-execution"},
        )
        running_attempt_id = running.attempt_id
    finally:
        seed.close()

    observer = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        authority_mode="embedded",
        task_source_kind="duckdb",
        # Exact failed-reload shape: the database environment survives but
        # programmatic argv lost --implement, so no callbacks or execution
        # permit are present.
        require_real_execution=False,
    )
    try:
        before_tasks = {
            task_cid: observer.task_source.get(task_cid).to_dict()
            for task_cid in ("task:cid:001", "task:cid:002")
        }
        before_attempt = observer.get_attempt(running_attempt_id)
        assert before_attempt is not None
        before_count_row = observer._require_connection().execute(
            """
            SELECT
                (SELECT COUNT(*) FROM database_task_attempts),
                (SELECT COUNT(*) FROM provider_invocations),
                (SELECT COUNT(*) FROM effect_claims)
            """
        ).fetchone()
        before_counts = tuple(
            int(before_count_row[index]) for index in range(3)
        )

        result = observer.run_once()

        assert result["unchanged"] is True
        assert result["write_count"] == 0
        assert result["execution_authorized"] is False
        assert result["selection_idle_reason"] == (
            "database_execution_not_authorized"
        )
        assert result["implementation_result"] is None
        assert result["completion_reconciliations"] == []
        assert result["expired_attempt_reconciliations"] == []
        assert result["terminal_retry_reconciliations"] == []
        assert result["terminal_portal_reconciliations"] == []

        after_tasks = {
            task_cid: observer.task_source.get(task_cid).to_dict()
            for task_cid in ("task:cid:001", "task:cid:002")
        }
        after_attempt = observer.get_attempt(running_attempt_id)
        after_count_row = observer._require_connection().execute(
            """
            SELECT
                (SELECT COUNT(*) FROM database_task_attempts),
                (SELECT COUNT(*) FROM provider_invocations),
                (SELECT COUNT(*) FROM effect_claims)
            """
        ).fetchone()
        after_counts = tuple(
            int(after_count_row[index]) for index in range(3)
        )
        assert after_tasks == before_tasks
        assert after_attempt == before_attempt
        assert after_counts == before_counts == (1, 0, 0)
        assert after_tasks["task:cid:001"]["status"] == "in_progress"
        assert after_tasks["task:cid:002"]["status"] == "ready"

        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="explicit real-execution authority",
        ):
            observer.claim_next()
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="explicit real-execution authority",
        ):
            observer.resume_attempt(running_attempt_id)
        guarded_mutations = (
            (
                "attempt phase commit",
                lambda: observer.commit_phase(running_attempt_id, "provider"),
            ),
            (
                "provider phase",
                lambda: observer.run_provider(before_attempt),
            ),
            (
                "effect phase",
                lambda: observer.run_effect(before_attempt, {}),
            ),
            (
                "task completion",
                lambda: observer.complete_attempt(before_attempt),
            ),
            (
                "prepared completion reconciliation",
                observer.reconcile_prepared_task_completions,
            ),
            (
                "expired attempt reconciliation",
                observer.reconcile_expired_running_attempts,
            ),
            (
                "terminal retry reconciliation",
                observer.reconcile_terminal_retry_states,
            ),
            (
                "terminal failure reconciliation",
                observer.reconcile_terminal_portal_failures,
            ),
        )
        for operation, mutation in guarded_mutations:
            with pytest.raises(
                DatabaseImplementationAuthorityError,
                match=operation,
            ):
                mutation()
        final_count = observer._require_connection().execute(
            "SELECT COUNT(*) FROM database_task_attempts"
        ).fetchone()
        assert int(final_count[0]) == 1
    finally:
        observer.close()


def test_failed_attempt_observes_fresh_typed_admission_as_superseding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:typed-admission-supersession",
    )
    stale = DatabaseTaskAttempt(
        attempt_id="attempt:stale",
        claim_id="claim:stale",
        task_cid="task:typed-admission-supersession",
        task_alias="CASF-TYPED-ADMISSION-SUPERSESSION",
        attempt_number=1,
        owner_session_id="session:stale",
        fencing_token=1,
        fence_epoch=1,
        lease_id="lease:stale",
        committed_phase=ATTEMPT_PHASE_FAILED,
        status="failed",
        started_at_ms=1_000,
        finished_at_ms=2_000,
    )
    admitted_pid = 12_345
    admitted_start_ticks = 19
    admitted_boot_id = "boot:typed-admission-supersession"
    admitted_parent_pid = 1
    admitted_receipt = {
        "operation": "database_attempt_admitted",
        "claim_phase_schema": (
            implementation_daemon_module.TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA
        ),
        "claim_process_attestation": {
            "schema": TYPED_DATABASE_CLAIM_PROCESS_SCHEMA,
            "grant_id": "owner-grant:typed-admission-supersession",
            "client_id": "client:typed-admission-supersession",
            "process_birth_id": _process_birth_content_id(
                admitted_pid,
                admitted_start_ticks,
                admitted_boot_id,
                admitted_parent_pid,
            ),
            "pid": admitted_pid,
            "uid": 0,
            "start_time_ticks": admitted_start_ticks,
            "boot_id": admitted_boot_id,
            "parent_pid": admitted_parent_pid,
        },
        "attempt_id": "attempt:fresh",
        "claim_id": "claim:fresh",
        "attempt_number": 2,
        "lease_id": "lease:fresh",
        "owner_session_id": "session:fresh",
        "fencing_token": 2,
        "fence_epoch": 2,
        "claimed_from_revision": 6,
        "admitted_from_revision": 7,
        "attempt_execution_phase": "claimed",
        "attempt_execution_revision": 1,
    }
    task = SimpleNamespace(
        status="in_progress",
        revision=8,
        body={"completion_receipt": admitted_receipt},
    )
    try:
        monkeypatch.setattr(
            daemon.task_source,
            "claim_process_attestation",
            lambda: admitted_receipt["claim_process_attestation"],
            raising=False,
        )
        monkeypatch.setattr(daemon.task_source, "get", lambda _task_cid: task)
        supersession = daemon._fresh_failed_attempt_control_supersession(stale)
        assert supersession is not None
        assert supersession["reason"] == "failed_attempt_control_superseded"
        assert supersession["control_operation"] == "database_attempt_admitted"
        assert supersession["successor_attempt_id"] == "attempt:fresh"
        assert supersession["control_revision"] == 8
        monkeypatch.setattr(daemon, "_latest_failed_attempts", lambda: [stale])
        monkeypatch.setattr(
            daemon,
            "_terminal_portal_failure_reason",
            lambda _attempt: "portal_provider_failed",
        )
        assert daemon.reconcile_terminal_portal_failures() == [supersession]

        task.body = {
            "completion_receipt": {
                **admitted_receipt,
                "lease_id": stale.lease_id,
            }
        }
        assert daemon._fresh_failed_attempt_control_supersession(stale) is None

        task.body = {"completion_receipt": admitted_receipt}
        monkeypatch.setattr(
            daemon.task_source,
            "claim_process_attestation",
            None,
            raising=False,
        )
        assert daemon._fresh_failed_attempt_control_supersession(stale) is None

        monkeypatch.setattr(
            daemon.task_source,
            "claim_process_attestation",
            lambda: admitted_receipt["claim_process_attestation"],
            raising=False,
        )
        task.body = {
            "completion_receipt": {
                **admitted_receipt,
                "claim_phase_schema": "database-attempt-admission@forged",
            }
        }
        assert daemon._fresh_failed_attempt_control_supersession(stale) is None

        task.body = {
            "completion_receipt": {
                **admitted_receipt,
                "claim_process_attestation": {
                    **admitted_receipt["claim_process_attestation"],
                    "unexpected": "forged",
                },
            }
        }
        assert daemon._fresh_failed_attempt_control_supersession(stale) is None

        task.body = {
            "completion_receipt": {
                **admitted_receipt,
                "claim_process_attestation": {
                    **admitted_receipt["claim_process_attestation"],
                    "process_birth_id": "birth:forged",
                },
            }
        }
        assert daemon._fresh_failed_attempt_control_supersession(stale) is None

        task.body = {"completion_receipt": admitted_receipt}
        task.revision = 9
        assert daemon._fresh_failed_attempt_control_supersession(stale) is None
    finally:
        daemon.close()


def test_portal_failure_terminal_cas_refetches_advanced_attempt(
    tmp_path: Path,
) -> None:
    holder: dict[str, DatabaseImplementationDaemon] = {}
    provider_revisions: list[int] = []

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        daemon = holder["daemon"]
        provider_revisions.append(attempt.revision)
        daemon._record_event(
            "portal_progress_before_failure",
            attempt_id=attempt.attempt_id,
            task_cid=attempt.task_cid,
            body={"provider_revision": attempt.revision},
        )
        raise DatabasePortalBridgeError("portal validation failed")

    daemon = _open_daemon(
        tmp_path,
        session="session:portal-failure-cas",
        provider_fn=provider,
    )
    holder["daemon"] = daemon
    try:
        daemon.materialize_population(_population(1))
        result = daemon.run_once()

        implementation = result["implementation_result"]
        assert implementation["portal_retryable_failure"] is False
        assert implementation["portal_terminal_failure"] is True
        assert implementation["status"] == "failed"
        assert implementation["deferred"] is False
        assert implementation["attempt_consumed"] == "unknown"
        assert implementation["provider_dispatched"] == "unknown"
        assert implementation["backoff_seconds"] == 0
        assert "fail_error" not in implementation
        assert provider_revisions == [2]

        stored = daemon.get_attempt(result["attempt_id"])
        assert stored is not None
        assert stored.status == "failed"
        assert stored.committed_phase == "failed"
        assert stored.revision == 3
        assert [
            (phase["phase"], phase["revision"])
            for phase in daemon.phase_history(stored.attempt_id)
        ] == [("claimed", 1), ("context", 2), ("failed", 3)]

        event_count = daemon._require_connection().execute(
            """
            SELECT COUNT(*) FROM daemon_execution_events
            WHERE attempt_id = ? AND event_type = ?
            """,
            [stored.attempt_id, "portal_progress_before_failure"],
        ).fetchone()
        assert event_count is not None
        assert int(event_count[0]) == 1
        task = daemon.task_source.get(stored.task_cid)
        assert task is not None
        assert task.status == "blocked"
        queue_entry = daemon.task_source.get_queue_entry(stored.task_cid)
        assert queue_entry is None
        assert implementation["terminal_state"]["status"] == "blocked"
    finally:
        daemon.close()


def test_typed_post_dispatch_validation_failure_retries_with_attempt_budget(
    tmp_path: Path,
) -> None:
    holder: dict[str, DatabaseImplementationDaemon] = {}
    now = {"ms": 1_000}

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        receipt = _validation_retry_receipt(holder["daemon"], attempt)
        raise DatabasePortalValidationRetry(receipt)

    daemon = _open_daemon(
        tmp_path,
        session="session:typed-validation-retry",
        provider_fn=provider,
        max_task_attempts=3,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    holder["daemon"] = daemon
    try:
        daemon.materialize_population(_population(1))
        result = daemon.run_once()

        implementation = result["implementation_result"]
        assert implementation["portal_retryable_failure"] is True
        assert implementation["portal_terminal_failure"] is False
        assert implementation["deferred"] is False
        assert implementation["attempt_consumed"] is True
        assert implementation["provider_dispatched"] is True
        assert implementation["typed_deferral_slot_consumed"] is False
        assert implementation["retry_budget_exhausted"] is False
        assert implementation["retry_state"]["status"] == "retrying"

        attempt = daemon.get_attempt(result["attempt_id"])
        assert attempt is not None
        failed = daemon.phase_history(attempt.attempt_id)[-1]["body"]
        assert failed["typed_validation_retry"]["remaining_task_attempts"] == 2
        evidence = daemon._terminal_retry_evidence(attempt)
        assert evidence is not None
        assert evidence["typed_deferral_budget"] is None
        assert evidence["typed_validation_retry"]["attempt_consumed"] is True
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "retrying"
        retry_seed = task.body["completion_receipt"]["validation_retry_seed"]
        assert retry_seed["receipt_id"] == failed["typed_validation_retry"][
            "receipt_id"
        ]

        # The retrying->in_progress claim CAS carries the verified seed into
        # the exact successor record consumed by the fresh Portal bridge.
        now["ms"] = 7_000
        successor = daemon.claim_next()
        assert successor is not None
        assert successor.attempt_number == 2
        claimed = daemon.task_source.get(attempt.task_cid)
        assert claimed is not None
        claim_receipt = claimed.body["completion_receipt"]
        assert claim_receipt["operation"] == "database_claim"
        assert claim_receipt["validation_retry_seed"] == retry_seed
        assert claim_receipt["validation_retry_source_attempt_id"] == (
            attempt.attempt_id
        )
        assert claim_receipt["attempt_number"] == successor.attempt_number
        assert claim_receipt["fencing_token"] == successor.fencing_token
        assert claim_receipt["fence_epoch"] == successor.fence_epoch
        assert claim_receipt["lease_id"] == successor.lease_id
    finally:
        daemon.close()


@pytest.mark.parametrize(
    "path_count",
    (2, 192),
    ids=("ordinary", "bounded-large-receipt"),
)
def test_seed_order_failure_rearms_only_after_exact_bridge_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path_count: int,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Tests"],
        cwd=repo,
        check=True,
    )
    (repo / "inventory").mkdir()
    if path_count == 2:
        changed_paths = [
            "inventory/result.json",
            "inventory/summary.json",
        ]
    else:
        long_directory = "a" * 240
        changed_paths = [
            (
                f"inventory/{long_directory}/{index:03d}-"
                + "b" * 238
                + ".json"
            )
            for index in range(path_count)
        ]
        assert all(
            490 <= len(path.encode("utf-8")) <= 500
            for path in changed_paths
        )
    changed_paths = sorted(changed_paths)
    declared_paths = list(reversed(changed_paths))
    for relative in changed_paths:
        output_path = repo / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('{"result":true}\n', encoding="utf-8")
    subprocess.run(["git", "add", "--", "inventory"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "candidate"], cwd=repo, check=True)
    candidate_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    rescue_branch = "rescue/dqp-t001-attempt-1-failed-validation"
    subprocess.run(
        ["git", "branch", rescue_branch, candidate_commit],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "--allow-empty", "-qm", "target advances"],
        cwd=repo,
        check=True,
    )
    target_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    holder: dict[str, object] = {}
    now = {"ms": 1_000}

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        if attempt.attempt_number == 1:
            daemon = holder["daemon"]
            assert isinstance(daemon, DatabaseImplementationDaemon)
            receipt = _validation_retry_receipt(daemon, attempt)
            receipt.update(
                {
                    "implementation_commit": candidate_commit,
                    "rescue_branch": rescue_branch,
                    "changed_paths": changed_paths,
                }
            )
            receipt.pop("receipt_id")
            receipt["receipt_id"] = daemon._database_portal_evidence_digest(
                receipt
            )
            raise DatabasePortalValidationRetry(receipt)
        if attempt.attempt_number == 2:
            raise DatabasePortalBridgeError(
                "database claim validation retry seed failed verification"
            )
        return {
            "status": "succeeded",
            "accepted": True,
            "task_cid": attempt.task_cid,
        }

    def replay(
        attempt: DatabaseTaskAttempt,
        record: object,
        historical_claim_body: Mapping[str, object],
    ) -> Mapping[str, object]:
        bridge = holder["bridge"]
        assert isinstance(bridge, DatabasePortalExecutionBridge)
        return bridge.verify_validation_retry_successor_recovery(
            attempt,
            record,
            historical_claim_body,
        )

    daemon = _open_daemon(
        tmp_path,
        session="session:seed-order-successor-recovery",
        provider_fn=provider,
        max_task_attempts=4,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
        validation_retry_successor_recovery_fn=replay,
    )
    holder["daemon"] = daemon
    try:
        population = _population(1)
        population["tasks"][0]["outputs"] = [
            {"path": path}
            for path in declared_paths
        ]
        daemon.materialize_population(population)
        initial_task = daemon.task_source.get("task:cid:001")
        assert initial_task is not None
        route_binding = {
            "policy_id": "policy:successor-route-test",
            "task_revision": int(initial_task.revision),
            "task_cid": initial_task.task_cid,
            "task_alias": initial_task.task_alias,
        }

        def bind_test_route(_source: object, task: object) -> Mapping[str, object]:
            assert getattr(task, "task_cid", "") == initial_task.task_cid
            return dict(route_binding)

        def validate_test_route(
            _source: object,
            value: Mapping[str, object],
            *,
            task: object,
            allow_claim_revision: bool = False,
        ) -> Mapping[str, object]:
            if dict(value) != route_binding:
                raise ValueError("rotated test execution route")
            task_revision = int(getattr(task, "revision", 0) or 0)
            if task_revision != route_binding["task_revision"]:
                if not allow_claim_revision:
                    raise ValueError("advanced test route was not admitted")
                task_body = getattr(task, "body", None)
                receipt = (
                    task_body.get("completion_receipt")
                    if isinstance(task_body, Mapping)
                    else None
                )
                if (
                    not isinstance(receipt, Mapping)
                    or dict(
                        receipt.get("execution_route_binding") or {}
                    )
                    != route_binding
                ):
                    raise ValueError("advanced test task lost its route")
            return dict(route_binding)

        monkeypatch.setattr(
            type(daemon.task_source),
            "execution_route_binding_for_task",
            bind_test_route,
            raising=False,
        )
        monkeypatch.setattr(
            type(daemon.task_source),
            "validate_execution_route_binding",
            validate_test_route,
            raising=False,
        )
        holder["bridge"] = DatabasePortalExecutionBridge(
            task_source=daemon.task_source,
            attempt_root=tmp_path / "portal-attempts",
            portal_factory=lambda _paths, _alias: SimpleNamespace(),
            repository_root=repo,
            max_task_attempts=4,
        )

        source_result = daemon.run_once()
        source = daemon.get_attempt(source_result["attempt_id"])
        assert source is not None and source.attempt_number == 1
        assert daemon.task_source.get(source.task_cid).status == "retrying"

        now["ms"] = 7_000
        target_result = daemon.run_once()
        assert "attempt_id" in target_result, target_result
        target = daemon.get_attempt(target_result["attempt_id"])
        assert target is not None and target.attempt_number == 2
        blocked = daemon.task_source.get(target.task_cid)
        assert blocked is not None and blocked.status == "blocked"

        # A moved rescue ref makes the repository replay fail closed without
        # changing the task or queue.  Restoring the exact preserved ref then
        # admits the one typed successor recovery.
        subprocess.run(
            ["git", "branch", "-f", rescue_branch, target_commit],
            cwd=repo,
            check=True,
        )
        assert daemon.reconcile_terminal_portal_failures() == []
        assert daemon.task_source.get(target.task_cid).status == "blocked"
        subprocess.run(
            ["git", "branch", "-f", rescue_branch, candidate_commit],
            cwd=repo,
            check=True,
        )

        if path_count == 2:
            history_projection = (
                daemon.task_source.task_revision_history_projection(
                    target.task_cid
                )
            )
            history_receipts = [
                revision["body"]["completion_receipt"]
                for revision in history_projection["revisions"]
                if isinstance(revision.get("body"), Mapping)
                and isinstance(
                    revision["body"].get("completion_receipt"),
                    Mapping,
                )
            ]
            route_fields = {
                "execution_route_binding",
                "execution_route_policy_id",
                "execution_route_origin_revision",
            }
            routed_receipts = [
                receipt
                for receipt in history_receipts
                if receipt.get("attempt_id")
                in {source.attempt_id, target.attempt_id}
                and set(receipt) & route_fields
            ]
            assert len(routed_receipts) >= 3
            assert all(
                set(receipt) & route_fields == route_fields
                for receipt in routed_receipts
            )
            assert all(
                {
                    field: receipt[field]
                    for field in route_fields
                }
                == {
                    field: routed_receipts[0][field]
                    for field in route_fields
                }
                for receipt in routed_receipts[1:]
            )

            revisions = json.loads(
                json.dumps(history_projection["revisions"])
            )
            source_retry_index = next(
                index
                for index, revision in enumerate(revisions)
                if isinstance(revision.get("body"), Mapping)
                and isinstance(
                    revision["body"].get("completion_receipt"),
                    Mapping,
                )
                and revision["body"]["completion_receipt"].get("operation")
                in {
                    "database_portal_validation_retry",
                    "database_portal_validation_retry_recovery",
                }
            )
            claim_index = next(
                index
                for index, revision in enumerate(revisions)
                if isinstance(revision.get("body"), Mapping)
                and isinstance(
                    revision["body"].get("completion_receipt"),
                    Mapping,
                )
                and revision["body"]["completion_receipt"].get("operation")
                == "database_claim"
                and revision["body"]["completion_receipt"].get("attempt_id")
                == target.attempt_id
            )
            terminal_index = next(
                index
                for index, revision in enumerate(revisions)
                if isinstance(revision.get("body"), Mapping)
                and isinstance(
                    revision["body"].get("completion_receipt"),
                    Mapping,
                )
                and revision["body"]["completion_receipt"].get("operation")
                == "database_portal_terminal_failure"
                and revision["body"]["completion_receipt"].get("attempt_id")
                == target.attempt_id
            )
            source_retry_revision = revisions[source_retry_index][
                "revision"
            ]
            claim_revision = revisions[claim_index]["revision"]
            terminal_revision = revisions[terminal_index]["revision"]
            assert claim_revision == source_retry_revision + 1
            assert terminal_revision == claim_revision + 1

            reservation = json.loads(json.dumps(revisions[claim_index]))
            reservation_receipt = reservation["body"][
                "completion_receipt"
            ]
            reservation_receipt.update(
                {
                    "operation": "database_claim",
                    "claimed_from_revision": source_retry_revision,
                    "claim_phase_schema": (
                        implementation_daemon_module
                        .TYPED_DATABASE_CLAIM_RESERVATION_SCHEMA
                    ),
                    "claim_process_attestation": {
                        "schema": "typed-claim-process@test",
                        "grant_id": "grant:test",
                        "client_id": "client:test",
                        "process_birth_id": "birth:test",
                        "pid": 1,
                        "uid": 0,
                        "start_time_ticks": 1,
                        "boot_id": "boot:test",
                        "parent_pid": 0,
                    },
                }
            )
            reservation["revision"] = source_retry_revision + 1

            admission = json.loads(json.dumps(reservation))
            admission_receipt = admission["body"]["completion_receipt"]
            admission_receipt.update(
                {
                    "operation": "database_attempt_admitted",
                    "claim_phase_schema": (
                        implementation_daemon_module
                        .TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA
                    ),
                    "admitted_from_revision": source_retry_revision + 1,
                    "attempt_execution_phase": "claimed",
                    "attempt_execution_revision": 1,
                }
            )
            admission["revision"] = source_retry_revision + 2

            typed_terminal = json.loads(
                json.dumps(revisions[terminal_index])
            )
            typed_terminal["revision"] = source_retry_revision + 3
            typed_terminal["body"]["completion_receipt"][
                "control_expected_revision"
            ] = source_retry_revision + 2
            typed_revisions = (
                revisions[:claim_index]
                + [reservation, admission, typed_terminal]
                + revisions[terminal_index + 1 :]
            )

            def history_projection_for(
                projected_revisions: list[dict[str, object]],
            ) -> dict[str, object]:
                projection = {
                    "schema": history_projection["schema"],
                    "task_cid": history_projection["task_cid"],
                    "revisions": projected_revisions,
                }
                projection["projection_cid"] = content_identity(projection)
                return projection

            typed_projection = history_projection_for(typed_revisions)
            typed_blocked = replace(
                blocked,
                revision=source_retry_revision + 3,
                body=dict(typed_terminal["body"]),
            )
            with monkeypatch.context() as typed_patch:
                typed_patch.setattr(
                    type(daemon.task_source),
                    "task_revision_history_projection",
                    lambda _source, _task_cid: typed_projection,
                )
                verified_typed = (
                    daemon._verified_validation_retry_successor_authority(
                        target,
                        typed_blocked,
                    )
                )
            assert verified_typed["recovery_receipt"][
                "target_claim_control_revision"
            ] == source_retry_revision + 2

            for corrupt in ("partial", "rotated"):
                corrupted_revisions = json.loads(
                    json.dumps(typed_revisions)
                )
                corrupted_receipt = corrupted_revisions[
                    source_retry_index
                ]["body"]["completion_receipt"]
                if corrupt == "partial":
                    corrupted_receipt.pop("execution_route_policy_id")
                else:
                    corrupted_receipt[
                        "execution_route_origin_revision"
                    ] += 1
                corrupted_projection = history_projection_for(
                    corrupted_revisions
                )
                with monkeypatch.context() as route_patch:
                    route_patch.setattr(
                        type(daemon.task_source),
                        "task_revision_history_projection",
                        lambda _source, _task_cid, value=corrupted_projection: value,
                    )
                    with pytest.raises(
                        (
                            DatabaseImplementationAuthorityError,
                            DatabaseImplementationConflictError,
                        )
                    ):
                        daemon._verified_validation_retry_successor_authority(
                            target,
                            typed_blocked,
                        )

        if path_count == 192:
            # A rejected size admission must happen before the predecessor
            # queue is rewritten.  Use a deliberately tiny bound to exercise
            # that ordering, then restore the production bound for recovery.
            task_before_preflight = daemon.task_source.get(target.task_cid)
            queue_before_preflight = daemon.task_source.get_queue_entry(
                target.task_cid
            )
            assert task_before_preflight is not None
            assert queue_before_preflight is not None
            monkeypatch.setattr(
                implementation_daemon_module,
                "_MAX_TASK_BODY_BYTES",
                1,
            )
            assert daemon.reconcile_terminal_portal_failures() == []
            task_after_preflight = daemon.task_source.get(target.task_cid)
            queue_after_preflight = daemon.task_source.get_queue_entry(
                target.task_cid
            )
            assert task_after_preflight is not None
            assert queue_after_preflight is not None
            assert task_after_preflight.revision == task_before_preflight.revision
            assert task_body_canonical_json_bytes(
                dict(task_after_preflight.body)
            ) == task_body_canonical_json_bytes(
                dict(task_before_preflight.body)
            )
            assert queue_after_preflight.to_dict() == (
                queue_before_preflight.to_dict()
            )
            monkeypatch.setattr(
                implementation_daemon_module,
                "_MAX_TASK_BODY_BYTES",
                MAX_TASK_BODY_BYTES,
            )

        outcomes = daemon.reconcile_terminal_portal_failures()
        assert len(outcomes) == 1
        recovered = daemon.task_source.get(target.task_cid)
        assert recovered is not None and recovered.status == "retrying"
        recovery_receipt = recovered.body["completion_receipt"]
        assert (
            len(task_body_canonical_json_bytes(dict(recovered.body)))
            < MAX_TASK_BODY_BYTES
        )
        assert recovery_receipt["operation"] == (
            "database_portal_validation_retry_successor_recovery"
        )
        assert recovery_receipt["queue_receipt"] == {}
        recovery = recovery_receipt[
            "validation_retry_successor_recovery"
        ]
        assert recovery["source_attempt_id"] == source.attempt_id
        assert recovery["target_attempt_id"] == target.attempt_id
        compact_proof = recovery["bridge_order_repair_proof"]
        assert compact_proof["preserved_commit_verified"] is True
        assert "scoped_outputs" not in compact_proof
        assert "changed_paths" not in compact_proof
        assert compact_proof["path_count"] == path_count

        def ordered_path_digest(paths: list[str]) -> str:
            return "sha256:" + hashlib.sha256(
                task_body_canonical_json_bytes(paths)
            ).hexdigest()

        assert compact_proof["scoped_outputs_ordered_digest"] == (
            ordered_path_digest(declared_paths)
        )
        assert compact_proof["changed_paths_ordered_digest"] == (
            ordered_path_digest(changed_paths)
        )
        assert compact_proof["exact_output_set_digest"] == (
            ordered_path_digest(sorted(changed_paths))
        )

        # Reconstruct the former full proof exactly: its retained proof_id
        # still commits to both ordered arrays even though persistence now
        # carries only bounded digest/count evidence.
        legacy_proof = {
            key: value
            for key, value in compact_proof.items()
            if key
            not in {
                "path_count",
                "scoped_outputs_ordered_digest",
                "changed_paths_ordered_digest",
                "exact_output_set_digest",
            }
        }
        legacy_proof["scoped_outputs"] = declared_paths
        legacy_proof["changed_paths"] = changed_paths
        legacy_proof_body = dict(legacy_proof)
        legacy_proof_id = legacy_proof_body.pop("proof_id")
        assert legacy_proof_id == "sha256:" + hashlib.sha256(
            task_body_canonical_json_bytes(legacy_proof_body)
        ).hexdigest()
        if path_count == 192:
            legacy_receipt = json.loads(json.dumps(recovery_receipt))
            legacy_receipt[
                "validation_retry_successor_recovery"
            ]["bridge_order_repair_proof"] = legacy_proof
            legacy_task_body = dict(recovered.body)
            legacy_task_body["completion_receipt"] = legacy_receipt
            assert (
                len(task_body_canonical_json_bytes(legacy_task_body))
                > MAX_TASK_BODY_BYTES
            )
        assert daemon.reconcile_terminal_portal_failures() == []

        now["ms"] = 13_000
        successor = daemon.claim_next()
        assert successor is not None and successor.attempt_number == 3
        claimed = daemon.task_source.get(successor.task_cid)
        assert claimed is not None and claimed.status == "in_progress"
        claim_receipt = claimed.body["completion_receipt"]
        assert claim_receipt["validation_retry_source_attempt_id"] == (
            source.attempt_id
        )
        assert claim_receipt["validation_retry_seed"] == (
            recovery_receipt["validation_retry_seed"]
        )
        assert successor.fencing_token > target.fencing_token
    finally:
        daemon.close()


def test_transition_invalid_replay_rejects_a_foreign_database_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A losing claim cannot replace a concurrent same-status receipt."""

    daemon = _open_daemon(
        tmp_path,
        session="session:claim-replay",
    )
    try:
        daemon.materialize_population(_population(1))
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        original_cas = daemon.task_source.compare_and_set_status
        foreign_receipt = {
            "operation": "database_claim",
            "attempt_id": "attempt:foreign",
            "claim_id": "claim:foreign",
            "owner_session_id": "session:foreign",
            "attempt_number": 2,
            "fencing_token": 11,
            "fence_epoch": 3,
            "lease_id": "lease:foreign",
        }
        admitted = original_cas(
            task.task_cid,
            expected_revision=int(task.revision),
            status="in_progress",
            receipt=foreign_receipt,
        )
        assert admitted.changed is True
        admitted_revision = int(admitted.task.revision)

        losing_receipt = {
            **foreign_receipt,
            "attempt_id": "attempt:losing",
            "claim_id": "claim:losing",
            "owner_session_id": "session:losing",
            "attempt_number": 1,
            "fencing_token": 10,
            "fence_epoch": 2,
            "lease_id": "lease:losing",
        }
        calls = {"count": 0}

        def transition_invalid(*args: object, **kwargs: object) -> object:
            calls["count"] += 1
            raise RuntimeError("transition_invalid: status already in_progress")

        monkeypatch.setattr(
            daemon.task_source,
            "compare_and_set_status",
            transition_invalid,
        )
        with pytest.raises(
            DatabaseImplementationConflictError,
            match="foreign durable receipt",
        ):
            daemon._cas_task_status_database(
                task.task_cid,
                expected_revision=admitted_revision,
                new_status="in_progress",
                receipt=losing_receipt,
            )
        assert calls["count"] == 1
        current = daemon.task_source.get(task.task_cid)
        assert current is not None
        assert current.revision == admitted_revision
        assert current.body["completion_receipt"] == foreign_receipt

        exact_calls = {"count": 0}

        def transition_invalid_once(*args: object, **kwargs: object) -> object:
            exact_calls["count"] += 1
            if exact_calls["count"] == 1:
                raise RuntimeError(
                    "transition_invalid: status already in_progress"
                )
            return original_cas(*args, **kwargs)

        monkeypatch.setattr(
            daemon.task_source,
            "compare_and_set_status",
            transition_invalid_once,
        )
        replayed = daemon._cas_task_status_database(
            task.task_cid,
            expected_revision=admitted_revision,
            new_status="in_progress",
            receipt=foreign_receipt,
        )
        assert replayed.changed is False
        assert exact_calls["count"] == 2
        current = daemon.task_source.get(task.task_cid)
        assert current is not None
        assert current.revision == admitted_revision
        assert current.body["completion_receipt"] == foreign_receipt
    finally:
        daemon.close()


def test_false_completion_reopen_is_not_consumed_by_generic_output_rearm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    snapshot = SimpleNamespace(
        task_id="VRIF-029",
        canonical_task_id="task:cid:vrif-029",
        metadata={
            "false_positive_completion_reopen": {
                "schema": "queue-owned-marker"
            },
            "completion": {
                "reason": "post_merge_declared_outputs_repaired",
                "repair_receipt": {
                    "entries": [{"path": "already-present.py"}]
                },
            },
        },
    )
    daemon = _open_daemon(
        tmp_path,
        session="session:false-reopen-generic-rearm",
    )
    try:
        daemon.open()
        daemon._merge_repo_root = repo
        daemon._merge_queue = SimpleNamespace(
            completed_requests=lambda **_kwargs: (snapshot,),
        )
        monkeypatch.setattr(
            daemon.task_source,
            "rearm_blocked_task",
            lambda *_args, **_kwargs: pytest.fail(
                "generic rearm consumed Bridge-owned false reopen"
            ),
        )

        result = daemon._rearm_blocked_tasks_with_outputs_on_head()

        assert result["attempted"] is True
        assert result["rearmed"] == 0
        assert result["write_count"] == 0
    finally:
        daemon.close()


def test_unusable_candidate_reaches_retry_handler_without_generic_requeue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalCandidateRetry("no_change_completion_not_allowed")

    daemon = _open_daemon(
        tmp_path,
        session="session:candidate-retry",
        provider_fn=provider,
        max_task_attempts=4,
    )
    try:
        def reject_generic_requeue(*_args: object, **_kwargs: object) -> None:
            pytest.fail("candidate retry entered generic stale-attempt requeue")

        monkeypatch.setattr(
            daemon,
            "_retire_stale_running_attempt",
            reject_generic_requeue,
        )
        monkeypatch.setattr(
            daemon,
            "_requeue_unimplemented_control_task",
            reject_generic_requeue,
        )
        daemon.materialize_population(_population(1))
        result = daemon.run_once()
        implementation = result["implementation_result"]
        assert implementation["portal_retryable_failure"] is True
        assert implementation["portal_terminal_failure"] is False
        assert implementation["attempt_consumed"] is True
        assert implementation["provider_dispatched"] is True
        assert implementation["retry_state"]["status"] == "retrying"
        attempt = daemon.get_attempt(result["attempt_id"])
        assert attempt is not None
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "retrying"
        failed = daemon.phase_history(attempt.attempt_id)[-1]["body"]
        assert failed["reason"] == "no_change_completion_not_allowed"
        assert failed["portal_retryable_failure"] is True
        assert failed["attempt_consumed"] is True
        assert failed["provider_dispatched"] is True
        receipt = task.body["completion_receipt"]
        assert receipt["operation"] != "requeue_unimplemented_stale_attempt"
    finally:
        daemon.close()


def test_reconcile_rearms_blocked_portal_provider_failed(tmp_path: Path) -> None:
    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeError("portal_provider_failed")

    daemon = _open_daemon(
        tmp_path,
        session="session:rearm-portal-provider-failed",
        provider_fn=provider,
        max_task_attempts=4,
    )
    try:
        daemon.materialize_population(_population(1))
        failed_result = daemon.run_once()
        attempt = daemon.get_attempt(failed_result["attempt_id"])
        assert attempt is not None
        assert daemon.task_source.get(attempt.task_cid).status == "blocked"

        outcomes = daemon.reconcile_terminal_portal_failures()
        assert len(outcomes) == 1
        assert outcomes[0]["status"] == "retrying"
        assert outcomes[0]["changed"] is True
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "retrying"
        assert daemon.reconcile_terminal_portal_failures() == []
    finally:
        daemon.close()


def test_reconcile_skips_stale_terminal_after_control_rearm(tmp_path: Path) -> None:
    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeError("portal_provider_failed")

    daemon = _open_daemon(
        tmp_path,
        session="session:rearm-ready-after-terminal-portal",
        provider_fn=provider,
        max_task_attempts=4,
    )
    try:
        daemon.materialize_population(_population(1))
        failed_result = daemon.run_once()
        attempt = daemon.get_attempt(failed_result["attempt_id"])
        assert attempt is not None
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "blocked"
        daemon.task_source.compare_and_set_status(
            attempt.task_cid,
            task.revision,
            "ready",
        )
        assert daemon.task_source.get(attempt.task_cid).status == "ready"
        assert daemon.reconcile_terminal_portal_failures() == []
        assert daemon.task_source.get(attempt.task_cid).status == "ready"
    finally:
        daemon.close()


def test_reconcile_rearms_blocked_checkout_contention(tmp_path: Path) -> None:
    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeError(
            "external_protected_checkout_recovery_required"
        )

    daemon = _open_daemon(
        tmp_path,
        session="session:rearm-checkout-contention",
        provider_fn=provider,
        max_task_attempts=4,
    )
    try:
        daemon.materialize_population(_population(1))
        failed_result = daemon.run_once()
        attempt = daemon.get_attempt(failed_result["attempt_id"])
        assert attempt is not None
        assert daemon.task_source.get(attempt.task_cid).status == "blocked"

        outcomes = daemon.reconcile_terminal_portal_failures()
        assert len(outcomes) == 1
        assert outcomes[0]["status"] == "retrying"
        assert outcomes[0]["changed"] is True
        assert outcomes[0]["evidence_source"] == (
            "portal_checkout_contention_reclassified"
        )
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "retrying"
        assert daemon.reconcile_terminal_portal_failures() == []
    finally:
        daemon.close()


def test_reconcile_rearms_blocked_missing_implementation_commit_at_attempt_cap(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:rearm-missing-commit-cap",
        max_task_attempts=1,
    )
    try:
        daemon.materialize_population(_population(1))
        failed = daemon.claim_next()
        assert failed is not None
        failed = daemon.commit_phase(failed, "context")
        failed = daemon.commit_phase(
            failed,
            "failed",
            body={
                "reason": (
                    DATABASE_PORTAL_COMPLETION_IMPLEMENTATION_COMMIT_MISSING_REASON
                ),
                "portal_retryable_failure": False,
                "portal_terminal_failure": True,
            },
        )
        terminal = daemon._persist_terminal_portal_failure(
            failed,
            reason=DATABASE_PORTAL_COMPLETION_IMPLEMENTATION_COMMIT_MISSING_REASON,
            coordination_evidence=(
                daemon._reconcile_failed_attempt_coordination(failed)
            ),
        )
        assert terminal["status"] == "blocked"
        assert daemon.task_source.get(failed.task_cid).status == "blocked"

        outcomes = daemon.reconcile_terminal_portal_failures()
        assert len(outcomes) == 1
        assert outcomes[0]["status"] == "retrying"
        assert outcomes[0]["changed"] is True
        assert outcomes[0]["evidence_source"] == (
            "portal_completion_handshake_reclassified"
        )
        task = daemon.task_source.get(failed.task_cid)
        assert task is not None
        assert task.status == "retrying"
        assert daemon.reconcile_terminal_portal_failures() == []
    finally:
        daemon.close()


def test_typed_blocked_reopen_stamps_retry_deadline_before_cas() -> None:
    """Quack cooldown binds the CAS receipt deadline, not a later clock."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
        IntentReceipt,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_database_task_source import (
        TypedDatabaseTaskSource,
    )

    captured: dict[str, Any] = {}
    source = TypedDatabaseTaskSource.__new__(TypedDatabaseTaskSource)
    source._clock_ms = lambda: 1_700_000  # type: ignore[method-assign]

    def fake_cas(
        task_cid: str,
        expected_revision: int,
        status: str,
        receipt: Mapping[str, Any] | None = None,
        *,
        expected_control_receipt: Mapping[str, Any] | None = None,
        evidence_digests: object = None,
    ) -> SimpleNamespace:
        captured["cas_receipt"] = dict(receipt or {})
        captured["cas_status"] = status
        captured["expected_revision"] = expected_revision
        captured["expected_control_receipt"] = expected_control_receipt
        return SimpleNamespace(
            previous_status="blocked",
            changed=True,
            revision=int(expected_revision) + 1,
        )

    def fake_cooldown(**kwargs: Any) -> IntentReceipt:
        captured["cooldown"] = dict(kwargs)
        deadline = int(kwargs["now_ms"]) + int(kwargs["delay_ms"])
        return IntentReceipt(
            event_id="event:cooldown",
            event_type="TASK_RETRY_COOLDOWN_RECORDED",
            global_sequence=1,
            recorded_at="typed-state-owner",
            subject_id=str(kwargs["task_cid"]),
            revision=1,
            changed=True,
            details={"retry_not_before_ms": deadline, "queue_revision": 1},
        )

    source.compare_and_set_status = fake_cas  # type: ignore[method-assign]
    source.record_task_retry_cooldown = fake_cooldown  # type: ignore[method-assign]
    receipt = {
        "operation": "database_portal_validation_retry_recovery",
        "attempt_id": "attempt:c2c70e3776d84c3b80aab21dece3b915",
        "claim_id": "claim:30ef15e81c1a412fb329da1b6f1131da",
        "lease_id": "lease:c74f619dda6a4bf897f1a689a4d59cc1",
        "owner_session_id": "pcsm-v1-executor:shard:0-of-4:track:ef26cb9db64a",
        "attempt_number": 2,
        "fencing_token": 2,
        "fence_epoch": 2,
        "queue_reason": (
            "database_portal_retry:attempt:c2c70e3776d84c3b80aab21dece3b915:"
            "portal_completion_handshake_retry"
        ),
        "backoff_ms": 0,
        "retry_not_before_ms": 0,
        "control_expected_revision": 8,
        "evidence_source": "portal_completion_handshake_reclassified",
    }
    result = source.record_queue_backoff_and_cas_status(
        task_cid="baguqeera6mvj3326qcksmlmwafo3s7ppd4s22vnbsn4tjnk4ylbjyqiesypa",
        expected_revision=8,
        expected_control_receipt={
            "operation": "database_portal_terminal_failure"
        },
        status="retrying",
        receipt=receipt,
        delay_ms=0,
        reason=str(receipt["queue_reason"]),
    )
    assert captured["cas_status"] == "retrying"
    assert captured["expected_revision"] == 8
    assert captured["cas_receipt"]["retry_not_before_ms"] == 1_700_000
    assert captured["cas_receipt"]["backoff_ms"] == 0
    assert captured["cooldown"]["now_ms"] == 1_700_000
    assert captured["cooldown"]["delay_ms"] == 0
    assert captured["cooldown"]["expected_task_status"] == "retrying"
    assert captured["cooldown"]["expected_task_revision"] == 8
    assert captured["cooldown"]["attempt_number"] == 2
    assert result["retry_not_before_ms"] == 1_700_000
    assert result["transition_receipt"]["retry_not_before_ms"] == 1_700_000
    assert result["previous_status"] == "blocked"


def test_blocked_retry_prefers_atomic_queue_status_over_typed_cooldown(
    tmp_path: Path,
) -> None:
    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeError("portal_provider_failed")

    daemon = _open_daemon(
        tmp_path,
        session="session:blocked-retry-atomic-queue",
        provider_fn=provider,
        max_task_attempts=4,
    )
    try:
        daemon.materialize_population(_population(1))
        failed_result = daemon.run_once()
        attempt = daemon.get_attempt(failed_result["attempt_id"])
        assert attempt is not None
        assert daemon.task_source.get(attempt.task_cid).status == "blocked"

        def typed_cooldown(**_kwargs: object) -> object:
            raise AssertionError(
                "typed cooldown must not run for blocked recovery"
            )

        daemon.task_source.record_task_retry_cooldown = typed_cooldown  # type: ignore[method-assign]
        outcomes = daemon.reconcile_terminal_portal_failures()
        assert len(outcomes) == 1
        assert outcomes[0]["status"] == "retrying"
        assert outcomes[0]["changed"] is True
        assert daemon.task_source.get(attempt.task_cid).status == "retrying"
    finally:
        daemon.close()


def test_blocked_generic_validation_failure_has_idempotent_typed_recovery(
    tmp_path: Path,
) -> None:
    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeError("portal_provider_failed")

    daemon = _open_daemon(
        tmp_path,
        session="session:validation-retry-recovery",
        provider_fn=provider,
        max_task_attempts=3,
    )
    try:
        daemon.materialize_population(_population(1))
        failed_result = daemon.run_once()
        attempt = daemon.get_attempt(failed_result["attempt_id"])
        assert attempt is not None
        assert daemon.task_source.get(attempt.task_cid).status == "blocked"

        receipt = _validation_retry_receipt(daemon, attempt)
        tampered = dict(receipt)
        tampered["denial_findings"] = ["denied_effect"]
        tampered.pop("receipt_id")
        tampered["receipt_id"] = daemon._database_portal_evidence_digest(
            tampered
        )
        with pytest.raises(DatabaseImplementationAuthorityError):
            daemon.recover_blocked_portal_validation_retry(
                attempt,
                retry_evidence=tampered,
            )
        assert daemon.task_source.get(attempt.task_cid).status == "blocked"

        recovered = daemon.recover_blocked_portal_validation_retry(
            attempt,
            retry_evidence=receipt,
        )
        assert recovered["changed"] is True
        assert recovered["status"] == "retrying"
        assert recovered["validation_retry_evidence"] == receipt
        assert recovered["coordination"]["attempt_id"] == attempt.attempt_id
        assert daemon.task_source.get(attempt.task_cid).status == "retrying"

        repeated = daemon.recover_blocked_portal_validation_retry(
            attempt,
            retry_evidence=receipt,
        )
        assert repeated["changed"] is False
        assert repeated["status"] == "retrying"
        assert repeated["validation_retry_evidence"] == receipt
        assert daemon.reconcile_terminal_portal_failures() == []
        assert daemon.reconcile_terminal_portal_failures() == []
    finally:
        daemon.close()


@pytest.mark.parametrize("control_status", ("completed", "todo"))
def test_terminal_portal_reconciliation_skips_settled_control_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control_status: str,
) -> None:
    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeError("portal_provider_failed")

    daemon = _open_daemon(
        tmp_path,
        session="session:terminal-reconcile-skip",
        provider_fn=provider,
        max_task_attempts=3,
    )
    try:
        daemon.materialize_population(_population(1))
        failed_result = daemon.run_once()
        attempt = daemon.get_attempt(failed_result["attempt_id"])
        assert attempt is not None
        original_get = daemon.task_source.get

        def projected_get(task_cid: str) -> object:
            task = original_get(task_cid)
            if task_cid != attempt.task_cid or task is None:
                return task
            return SimpleNamespace(
                status=control_status,
                revision=task.revision,
                body=dict(task.body),
            )

        monkeypatch.setattr(daemon.task_source, "get", projected_get)
        assert daemon.reconcile_terminal_portal_failures() == []
    finally:
        daemon.close()


def test_terminal_portal_reconciliation_accepts_board_unstall_retrying(
    tmp_path: Path,
) -> None:
    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeError("portal_provider_failed")

    daemon = _open_daemon(
        tmp_path,
        session="session:board-unstall-retrying",
        provider_fn=provider,
        max_task_attempts=3,
    )
    try:
        daemon.materialize_population(_population(1))
        failed_result = daemon.run_once()
        attempt = daemon.get_attempt(failed_result["attempt_id"])
        assert attempt is not None
        prior = daemon.task_source.get(attempt.task_cid)
        assert prior is not None
        assert prior.status in {"blocked", "in_progress"}
        with daemon.task_source._intent._connection(write=True) as connection:
            connection.execute(
                "UPDATE tasks SET status = 'retrying', revision = revision + 1, "
                "updated_at = '2026-08-22T22:00:00Z' WHERE task_cid = ?",
                [attempt.task_cid],
            )
        retried = daemon.task_source.get(attempt.task_cid)
        assert retried is not None
        assert retried.status == "retrying"
        receipt = (retried.body or {}).get("completion_receipt")
        assert not isinstance(receipt, dict) or receipt.get("operation") != (
            "database_portal_validation_retry_recovery"
        )
        assert daemon.reconcile_terminal_portal_failures() == []
        still = daemon.task_source.get(attempt.task_cid)
        assert still is not None
        assert still.status == "retrying"
    finally:
        daemon.close()


def test_proposal_gate_failure_retries_instead_of_blocking(
    tmp_path: Path,
) -> None:
    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeError("proposal_gate_failed")

    daemon = _open_daemon(
        tmp_path,
        session="session:proposal-gate-retry",
        provider_fn=provider,
        max_task_attempts=3,
    )
    try:
        daemon.materialize_population(_population(1))
        failed_result = daemon.run_once()
        attempt = daemon.get_attempt(failed_result["attempt_id"])
        assert attempt is not None
        implementation = failed_result.get("implementation_result") or {}
        assert implementation.get("portal_retryable_failure") is True
        assert implementation.get("portal_terminal_failure") is False
        assert implementation.get("reason") == "proposal_gate_failed"
        assert daemon.task_source.get(attempt.task_cid).status == "retrying"
    finally:
        daemon.close()


def test_quack_attach_contention_defers_instead_of_crashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        QuackTransportContentionError,
    )

    daemon = _open_daemon(
        tmp_path,
        session="session:quack-attach-defer",
        max_task_attempts=3,
    )
    try:
        def boom(*_args: object, **_kwargs: object) -> dict[str, object]:
            raise QuackTransportContentionError(
                "quack control-plane attach contended: Authentication failed"
            )

        monkeypatch.setattr(daemon, "_run_once_impl", boom)
        result = daemon.run_once()
        assert result.get("deferred") is True
        assert result.get("skipped") is True
        assert result.get("reason") == "quack_attach_contended"
        assert result.get("portal_retryable_failure") is True
        assert result.get("portal_terminal_failure") is False
        assert result.get("attempt_consumed") is False
        assert result.get("provider_dispatched") is False
    finally:
        daemon.close()


def test_quack_attach_contention_requests_owner_board_unstall(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        QuackTransportContentionError,
    )

    inbox = tmp_path / "mutations"
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR", str(inbox))
    daemon = _open_daemon(
        tmp_path,
        session="session:quack-attach-unstall",
        max_task_attempts=3,
    )
    try:
        def boom(*_args: object, **_kwargs: object) -> dict[str, object]:
            raise QuackTransportContentionError(
                "quack control-plane attach contended: Authentication failed"
            )

        monkeypatch.setattr(daemon, "_run_once_impl", boom)
        result = daemon.run_once()
        assert result.get("board_unstall_request", {}).get("requested") is True
        requests = list(inbox.glob("*.request.json"))
        assert len(requests) == 1
        payload = json.loads(requests[0].read_text(encoding="utf-8"))
        assert payload["op"] == "board_unstall"
    finally:
        daemon.close()


def test_run_once_unstalls_stale_in_progress_gate_and_claims(
    tmp_path: Path,
) -> None:
    from datetime import datetime, timedelta, timezone

    daemon = _open_daemon(
        tmp_path,
        session="session:unstall-gate",
        max_task_attempts=3,
    )
    try:
        daemon.materialize_population(_population(1))
        stale = (datetime.now(timezone.utc) - timedelta(hours=12)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        with daemon.task_source._intent._connection(write=True) as connection:
            connection.execute(
                "UPDATE tasks SET status = 'in_progress', updated_at = ? "
                "WHERE task_cid = ?",
                [stale, "task:cid:001"],
            )
        stuck = daemon.task_source.get("task:cid:001")
        assert stuck is not None
        assert stuck.status == "in_progress"
        idle = daemon.claim_next()
        assert idle is None
        unstalled = daemon.reconcile_stale_in_progress_gates()
        assert [item["task_cid"] for item in unstalled] == ["task:cid:001"]
        retried = daemon.task_source.get("task:cid:001")
        assert retried is not None
        assert retried.status == "retrying"
        attempt = daemon.claim_next()
        assert attempt is not None
        assert attempt.task_cid == "task:cid:001"
    finally:
        daemon.close()


def test_orphan_in_progress_unstall_retries_gate_without_live_claim(
    tmp_path: Path,
) -> None:
    from datetime import datetime, timedelta, timezone

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    daemon = _open_daemon(
        tmp_path,
        session="session:orphan-unstall",
        max_task_attempts=3,
        repo_root=repo,
    )
    try:
        daemon.materialize_population(_population(1))
        stale = (datetime.now(timezone.utc) - timedelta(minutes=10)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        with daemon.task_source._intent._connection(write=True) as connection:
            connection.execute(
                "UPDATE tasks SET status = 'in_progress', updated_at = ? "
                "WHERE task_cid = ?",
                [stale, "task:cid:001"],
            )
        unstalled = daemon.reconcile_stale_in_progress_gates()
        assert any(
            item.get("task_cid") == "task:cid:001"
            and item.get("reason")
            == "in_progress_without_live_worktree_lifecycle_owner"
            for item in unstalled
        )
        retried = daemon.task_source.get("task:cid:001")
        assert retried is not None
        assert retried.status == "retrying"
    finally:
        daemon.close()


def test_orphan_in_progress_unstall_leaves_live_lifecycle_owner_alone(
    tmp_path: Path,
) -> None:
    from datetime import datetime, timedelta, timezone

    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        WorktreeLifecycleStore,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    daemon = _open_daemon(
        tmp_path,
        session="session:orphan-live",
        max_task_attempts=3,
        repo_root=repo,
    )
    try:
        daemon.materialize_population(_population(1))
        stale = (datetime.now(timezone.utc) - timedelta(minutes=10)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        with daemon.task_source._intent._connection(write=True) as connection:
            connection.execute(
                "UPDATE tasks SET status = 'in_progress', updated_at = ? "
                "WHERE task_cid = ?",
                [stale, "task:cid:001"],
            )
        store = WorktreeLifecycleStore(repo_root=repo)
        store.begin_preparing(
            task_id="DQP-T001",
            canonical_task_cid="task:cid:001",
            attempt=1,
            lane_id="lane-0",
            workspace_path=tmp_path / "live-ws",
            branch="implementation/dqp-t001",
            merge_target="main",
            state_dir=str(tmp_path / "state" / "lane-0"),
        )
        unstalled = daemon.reconcile_stale_in_progress_gates()
        assert unstalled == []
        live = daemon.task_source.get("task:cid:001")
        assert live is not None
        assert live.status == "in_progress"
    finally:
        daemon.close()


def test_stale_in_progress_unstall_leaves_live_attempts_alone(
    tmp_path: Path,
) -> None:
    from datetime import datetime, timedelta, timezone

    daemon = _open_daemon(
        tmp_path,
        session="session:unstall-live",
        max_task_attempts=3,
    )
    try:
        daemon.materialize_population(_population(1))
        recent = (datetime.now(timezone.utc) - timedelta(minutes=20)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        with daemon.task_source._intent._connection(write=True) as connection:
            connection.execute(
                "UPDATE tasks SET status = 'in_progress', updated_at = ? "
                "WHERE task_cid = ?",
                [recent, "task:cid:001"],
            )
        unstalled = daemon.reconcile_stale_in_progress_gates()
        assert unstalled == []
        live = daemon.task_source.get("task:cid:001")
        assert live is not None
        assert live.status == "in_progress"
    finally:
        daemon.close()


def test_quack_attach_contention_still_expires_running_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        QuackTransportContentionError,
    )

    daemon = _open_daemon(
        tmp_path,
        session="session:quack-attach-expire",
        max_task_attempts=3,
    )
    try:
        def boom(*_args: object, **_kwargs: object) -> list[object]:
            raise QuackTransportContentionError(
                "quack control-plane attach contended: Authentication failed"
            )

        expired = [
            {
                "status": "expired",
                "reason": "coordination_lease_expired_before_completion",
            }
        ]
        seen = {"expired": False}

        def expire() -> list[dict[str, object]]:
            seen["expired"] = True
            return expired

        monkeypatch.setattr(daemon, "reconcile_prepared_task_completions", boom)
        monkeypatch.setattr(daemon, "reconcile_expired_running_attempts", expire)
        monkeypatch.setattr(daemon, "reconcile_terminal_portal_failures", boom)
        monkeypatch.setattr(daemon, "reconcile_terminal_retry_states", lambda: [])
        monkeypatch.setattr(daemon, "list_running_attempts", lambda: [])
        monkeypatch.setattr(daemon, "claim_next", lambda: None)
        result = daemon.run_once()
        assert seen["expired"] is True
        assert result.get("expired_attempt_reconciliations") == expired
        assert result.get("selection_idle_reason") == "quack_attach_failed"
        assert result.get("reason") == "quack_attach_contended"
    finally:
        daemon.close()


def test_inflight_process_failure_retries_instead_of_blocking(
    tmp_path: Path,
) -> None:
    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeError("inflight_process")

    daemon = _open_daemon(
        tmp_path,
        session="session:inflight-process-retry",
        provider_fn=provider,
        max_task_attempts=3,
    )
    try:
        daemon.materialize_population(_population(1))
        failed_result = daemon.run_once()
        attempt = daemon.get_attempt(failed_result["attempt_id"])
        assert attempt is not None
        implementation = failed_result.get("implementation_result") or {}
        assert implementation.get("portal_retryable_failure") is True
        assert implementation.get("portal_terminal_failure") is False
        assert implementation.get("reason") == "inflight_process"
        assert daemon.task_source.get(attempt.task_cid).status == "retrying"
    finally:
        daemon.close()


def test_inflight_process_deferral_does_not_exhaust_typed_budget(
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
        DatabasePortalBridgeDeferred,
    )

    now = {"ms": 1_000}
    calls: list[str] = []

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        calls.append(attempt.attempt_id)
        raise DatabasePortalBridgeDeferred("inflight_process", backoff_seconds=30)

    daemon = _open_daemon(
        tmp_path,
        session="session:inflight-deferral-budget",
        provider_fn=provider,
        max_task_attempts=3,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        for _ in range(12):
            daemon.run_once()
            now["ms"] += 31_000
            task = daemon.task_source.get("task:cid:001")
            assert task is not None
            assert task.status != "blocked"
        assert len(calls) >= 4
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        assert task.status == "retrying"
    finally:
        daemon.close()


def test_reconcile_keeps_inflight_deferral_block_without_typed_supersession(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:inflight-deferral-unstall",
        max_task_attempts=3,
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        attempt = daemon.commit_phase(attempt, "context")
        attempt = daemon.commit_phase(
            attempt,
            "failed",
            body={
                "reason": "inflight_process",
                "portal_retryable_failure": True,
                "portal_terminal_failure": False,
            },
        )
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        daemon.task_source.compare_and_set_status(
            task.task_cid,
            expected_revision=int(task.revision),
            status="blocked",
            receipt={
                "operation": "database_portal_typed_deferral_budget_exhausted",
                "attempt_id": attempt.attempt_id,
                "attempt_number": int(attempt.attempt_number),
                "claim_id": attempt.claim_id,
                "lease_id": attempt.lease_id,
                "owner_session_id": attempt.owner_session_id,
                "fencing_token": int(attempt.fencing_token),
                "fence_epoch": int(attempt.fence_epoch),
                "retry_budget": {
                    "matching_attempts": [
                        {"reason": "inflight_process"},
                        {"reason": "inflight_process"},
                        {"reason": "inflight_process"},
                    ]
                },
            },
        )
        blocked = daemon.task_source.get("task:cid:001")
        assert blocked is not None
        assert blocked.status == "blocked"
        blocked_revision = blocked.revision
        outcomes = daemon.reconcile_inflight_deferral_blocks()
        assert outcomes == []
        unchanged = daemon.task_source.get("task:cid:001")
        assert unchanged is not None
        assert unchanged.status == "blocked"
        assert unchanged.revision == blocked_revision
    finally:
        daemon.close()


def test_reconcile_refuses_unbound_inflight_deferral_budget_block(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:inflight-deferral-unbound",
        max_task_attempts=3,
    )
    try:
        daemon.materialize_population(_population(1))
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        daemon.task_source.compare_and_set_status(
            task.task_cid,
            expected_revision=int(task.revision),
            status="blocked",
            receipt={
                "operation": "database_portal_typed_deferral_budget_exhausted",
                "retry_budget": {
                    "matching_attempts": [{"reason": "inflight_process"}],
                },
            },
        )

        result = daemon.run_once()

        assert result["unchanged"] is True
        assert result["write_count"] == 0
        assert result["inflight_deferral_unstalls"] == [
            {
                "task_cid": "task:cid:001",
                "task_alias": "DQP-T001",
                "previous_status": "blocked",
                "status": "blocked",
                "changed": False,
                "reason": "inflight_deferral_exact_attempt_unavailable",
            }
        ]
        blocked = daemon.task_source.get("task:cid:001")
        assert blocked is not None
        assert blocked.status == "blocked"
    finally:
        daemon.close()


@pytest.mark.parametrize(
    ("mutation", "error_type"),
    [
        ("missing_seed", DatabaseImplementationAuthorityError),
        ("wrong_seed_receipt", DatabaseImplementationAuthorityError),
    ],
)
def test_terminal_portal_reconciliation_rejects_foreign_retrying_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    error_type: type[Exception],
) -> None:
    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeError("portal_provider_failed")

    daemon = _open_daemon(
        tmp_path,
        session="session:validation-retry-recovery-forgery",
        provider_fn=provider,
        max_task_attempts=3,
    )
    try:
        daemon.materialize_population(_population(1))
        failed_result = daemon.run_once()
        attempt = daemon.get_attempt(failed_result["attempt_id"])
        assert attempt is not None
        retry_evidence = _validation_retry_receipt(daemon, attempt)
        daemon.recover_blocked_portal_validation_retry(
            attempt,
            retry_evidence=retry_evidence,
        )
        persisted = daemon.task_source.get(attempt.task_cid)
        assert persisted is not None
        receipt = dict(persisted.body["completion_receipt"])
        if mutation == "missing_seed":
            receipt.pop("validation_retry_seed")
        else:
            seed = dict(receipt["validation_retry_seed"])
            seed["receipt_id"] = "sha256:" + "0" * 64
            receipt["validation_retry_seed"] = seed

        original_get = daemon.task_source.get

        def projected_get(task_cid: str) -> object:
            task = original_get(task_cid)
            if task_cid != attempt.task_cid or task is None:
                return task
            return SimpleNamespace(
                status=task.status,
                revision=task.revision,
                body={**dict(task.body), "completion_receipt": receipt},
            )

        monkeypatch.setattr(daemon.task_source, "get", projected_get)
        with pytest.raises(error_type):
            daemon.reconcile_terminal_portal_failures()
    finally:
        daemon.close()


def test_terminal_portal_recovery_projection_rejects_newer_fence(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}

    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeError("portal_provider_failed")

    daemon = _open_daemon(
        tmp_path,
        session="session:validation-retry-recovery-newer-fence",
        provider_fn=provider,
        max_task_attempts=3,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        failed_result = daemon.run_once()
        attempt = daemon.get_attempt(failed_result["attempt_id"])
        assert attempt is not None
        daemon.recover_blocked_portal_validation_retry(
            attempt,
            retry_evidence=_validation_retry_receipt(daemon, attempt),
        )
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "retrying"

        source_claim = daemon.coordinator.get_task_claim(attempt.claim_id)
        assert source_claim is not None
        now["ms"] = 7_000
        daemon.coordinator.expire_task_claim(source_claim, now_ms=now["ms"])
        newer = daemon.coordinator.claim_ready_task(
            owner_session_id="session:newer-validation-retry-fence",
            lease_ms=5_000,
            now_ms=now["ms"],
        )
        assert newer is not None
        assert newer.fencing_token > attempt.fencing_token

        outcomes = daemon.reconcile_terminal_portal_failures()
        assert len(outcomes) == 1
        assert outcomes[0]["changed"] is False
        assert outcomes[0]["status"] == "retrying"
        assert outcomes[0]["reason"] == (
            "failed_attempt_coordination_superseded"
        )
        assert outcomes[0]["successor_claim_id"] == newer.claim_id
        assert outcomes[0]["successor_attempt_id"] == newer.attempt_id
        assert outcomes[0]["coordination"][
            "superseded_by_newer_fence"
        ] is True
        unchanged = daemon.task_source.get(attempt.task_cid)
        assert unchanged is not None
        assert unchanged.status == "retrying"
    finally:
        daemon.close()


def test_restart_accepts_exact_validation_retry_recovery_projection(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}

    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeError("portal_provider_failed")

    first = _open_daemon(
        tmp_path,
        session="session:validation-retry-recovery-before-restart",
        provider_fn=provider,
        max_task_attempts=3,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        first.materialize_population(_population(1))
        failed_result = first.run_once()
        source = first.get_attempt(failed_result["attempt_id"])
        assert source is not None
        first.recover_blocked_portal_validation_retry(
            source,
            retry_evidence=_validation_retry_receipt(first, source),
        )
    finally:
        first.close()

    now["ms"] = 7_000
    restarted = _open_daemon(
        tmp_path,
        session="session:validation-retry-recovery-after-restart",
        max_task_attempts=3,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        result = restarted.run_once()
        assert result["terminal_portal_reconciliations"] == []
        assert result["implementation_result"] is not None
        assert result["implementation_result"]["status"] == "succeeded"
        successor = restarted.get_attempt(result["attempt_id"])
        assert successor is not None
        assert successor.attempt_number > source.attempt_number
    finally:
        restarted.close()


def test_restart_finishes_terminal_portal_failure_control_cas(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:terminal-portal-recovery",
    )
    try:
        daemon.materialize_population(_population(1))
        failed_attempt = daemon.claim_next()
        assert failed_attempt is not None
        failed_attempt = daemon.commit_phase(failed_attempt, "context")
        failed_attempt = daemon.commit_phase(
            failed_attempt,
            "failed",
            body={
                "reason": "untyped_portal_integrity_failure",
                "portal_retryable_failure": False,
                "portal_terminal_failure": True,
            },
        )

        result = daemon.run_once()

        assert result["implementation_result"] is None
        assert len(result["terminal_portal_reconciliations"]) == 1
        blocked = daemon.task_source.get(failed_attempt.task_cid)
        assert blocked is not None
        assert blocked.status == "blocked"
        assert daemon.task_source.get_queue_entry(failed_attempt.task_cid) is None
    finally:
        daemon.close()


def test_typed_portal_deferral_honors_canonical_cooldown_after_lease_expiry(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}
    provider_attempts: list[str] = []

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_attempts.append(attempt.attempt_id)
        raise DatabasePortalBridgeDeferred(
            "validation_project_dependency_preflight_failed",
            backoff_seconds=300,
        )

    daemon = _open_daemon(
        tmp_path,
        session="session:typed-portal-deferral",
        provider_fn=provider,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        first = daemon.run_once()

        assert len(provider_attempts) == 1
        assert first["implementation_result"]["deferred"] is True
        assert first["implementation_result"]["attempt_consumed"] is False
        assert first["implementation_result"]["provider_dispatched"] is False
        assert first["implementation_result"]["backoff_seconds"] == 300
        task_cid = str(first["claimed_task_cid"])
        queue_entry = daemon.task_source.get_queue_entry(task_cid)
        assert queue_entry is not None
        assert queue_entry.retry_not_before_ms == 301_000
        task = daemon.task_source.get(task_cid)
        assert task is not None
        assert task.status == "retrying"
        assert task.revision == 3

        # The coordination lease is expired, but the canonical queue deadline
        # remains authoritative and no replacement attempt is constructed.
        now["ms"] = 7_000
        cooled = daemon.run_once()
        assert cooled["selection_idle_reason"] == "no_ready_tasks"
        assert cooled["implementation_result"] is None
        assert len(provider_attempts) == 1
        assert daemon.task_source.get_queue_entry(
            task_cid
        ).retry_not_before_ms == 301_000

        now["ms"] = 301_001
        retried = daemon.run_once()
        assert len(provider_attempts) == 2
        assert retried["attempt_id"] != first["attempt_id"]
        assert retried["implementation_result"]["deferred"] is True
        retried_task = daemon.task_source.get(task_cid)
        assert retried_task is not None
        assert retried_task.status == "retrying"
        assert retried_task.revision == 5
    finally:
        daemon.close()


def test_typed_portal_deferral_budget_blocks_before_fourth_dispatch(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}
    provider_attempts: list[str] = []

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_attempts.append(attempt.attempt_id)
        raise DatabasePortalBridgeDeferred(
            "validation_project_dependency_preflight_failed",
            backoff_seconds=300,
        )

    daemon = _open_daemon(
        tmp_path,
        session="session:typed-deferral-budget",
        provider_fn=provider,
        lease_ms=5_000,
        max_task_attempts=3,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        first = daemon.run_once()
        task_cid = str(first["claimed_task_cid"])
        assert first["implementation_result"]["retry_budget_exhausted"] is False

        now["ms"] = 301_001
        second = daemon.run_once()
        assert second["implementation_result"]["retry_budget_exhausted"] is False

        now["ms"] = 601_002
        third = daemon.run_once()
        implementation = third["implementation_result"]
        assert implementation["retry_budget_exhausted"] is True
        assert implementation["attempt_consumed"] is False
        assert implementation["typed_deferral_slot_consumed"] is True
        assert implementation["retry_state"] is None
        terminal = implementation["terminal_state"]
        assert terminal["status"] == "blocked"
        assert terminal["reason"] == "typed_portal_deferral_budget_exhausted"
        budget = terminal["retry_budget"]
        assert budget["typed_deferral_count"] == 3
        assert budget["max_task_attempts"] == 3
        assert budget["exhausted"] is True
        assert len(budget["matching_attempts"]) == 3

        task = daemon.task_source.get(task_cid)
        assert task is not None
        assert task.status == "blocked"
        assert len(provider_attempts) == 3

        # Even after every prior cooldown and lease deadline, the blocked
        # control task cannot construct or dispatch attempt four.
        now["ms"] = 1_000_000
        idle = daemon.run_once()
        assert idle["implementation_result"] is None
        assert idle["selection_idle_reason"] == "no_ready_tasks"
        assert len(provider_attempts) == 3
    finally:
        daemon.close()


def test_mixed_leftover_wait_deferrals_do_not_consume_typed_budget(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}
    observed_reasons: list[str] = []
    reasons = [
        "inflight_process",
        "external_protected_checkout_recovery_required",
        "inflight_process",
    ]

    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        reason = reasons[len(observed_reasons)]
        observed_reasons.append(reason)
        raise DatabasePortalBridgeDeferred(
            reason,
            backoff_seconds=30,
        )

    daemon = _open_daemon(
        tmp_path,
        session="session:mixed-leftover-wait-budget",
        provider_fn=provider,
        lease_ms=5_000,
        max_task_attempts=2,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        first = daemon.run_once()
        task_cid = str(first["claimed_task_cid"])
        assert first["implementation_result"]["retry_budget_exhausted"] is False

        for timestamp in (31_001, 61_002):
            now["ms"] = timestamp
            retried = daemon.run_once()
            assert (
                retried["implementation_result"]["retry_budget_exhausted"]
                is False
            )

        task = daemon.task_source.get(task_cid)
        assert task is not None
        assert task.status == "retrying"
        assert observed_reasons == reasons
    finally:
        daemon.close()


def test_run_once_rearms_identity_bound_mixed_leftover_wait_exhaustion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = {"ms": 1_000}
    reasons = [
        "inflight_process",
        "external_protected_checkout_recovery_required",
    ]
    observed_attempts: list[DatabaseTaskAttempt] = []

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        observed_attempts.append(attempt)
        raise DatabasePortalBridgeDeferred(
            reasons[len(observed_attempts) - 1],
            backoff_seconds=30,
        )

    daemon = _open_daemon(
        tmp_path,
        session="session:mixed-leftover-wait-rearm",
        provider_fn=provider,
        lease_ms=5_000,
        max_task_attempts=2,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        daemon.run_once()
        now["ms"] = 31_001
        daemon.run_once()
        attempts = [
            daemon.get_attempt(item.attempt_id) for item in observed_attempts
        ]
        assert all(item is not None for item in attempts)
        exact_attempts = [item for item in attempts if item is not None]
        _rewrite_as_legacy_typed_deferrals(
            daemon,
            exact_attempts,
            reasons,
        )
        attempt, budget = _block_with_legacy_leftover_wait_budget(
            daemon,
            exact_attempts,
        )
        assert daemon.task_source.get(attempt.task_cid).status == "blocked"

        # Keep the test at the reconciliation boundary so the just-rearmed
        # task is not immediately claimed and its recovery receipt remains
        # independently inspectable.
        monkeypatch.setattr(daemon, "claim_next", lambda: None)
        now["ms"] = 100_000
        repaired = daemon.run_once()
        recovery = repaired[
            "leftover_wait_deferral_budget_recovery_reconciliations"
        ]
        assert len(recovery) == 1
        assert recovery[0]["changed"] is True
        assert repaired["terminal_retry_reconciliations"] == []

        rearmed = daemon.task_source.get(attempt.task_cid)
        assert rearmed is not None and rearmed.status == "retrying"
        receipt = rearmed.body["completion_receipt"]
        seed = receipt["leftover_wait_deferral_budget_recovery_seed"]
        assert receipt["attempt_id"] == attempt.attempt_id
        assert receipt["claim_id"] == attempt.claim_id
        assert receipt["fencing_token"] == attempt.fencing_token
        assert seed["attempt_id"] == attempt.attempt_id
        assert seed["blocked_retry_budget"] == budget
        assert seed["blocked_retry_budget_observation_id"] == budget[
            "observation_id"
        ]
        assert seed["blocked_retry_budget_digest"] == (
            daemon._database_portal_evidence_digest(budget)
        )
        assert seed["exhausting_reasons"] == sorted(
            {
                "inflight_process",
                "external_protected_checkout_recovery_required",
            }
        )
        daemon._verified_leftover_wait_deferral_budget_recovery_state(
            attempt,
            rearmed,
        )
    finally:
        daemon.close()


def test_leftover_wait_budget_recovery_rejects_mixed_non_wait_reason(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}
    reasons = ["inflight_process", "provider_capacity_backoff"]
    observed_attempts: list[DatabaseTaskAttempt] = []

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        observed_attempts.append(attempt)
        raise DatabasePortalBridgeDeferred(
            reasons[len(observed_attempts) - 1],
            backoff_seconds=30,
        )

    daemon = _open_daemon(
        tmp_path,
        session="session:leftover-wait-rearm-fail-closed",
        provider_fn=provider,
        lease_ms=5_000,
        max_task_attempts=2,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        daemon.run_once()
        now["ms"] = 31_001
        daemon.run_once()
        attempts = [
            daemon.get_attempt(item.attempt_id) for item in observed_attempts
        ]
        assert all(item is not None for item in attempts)
        exact_attempts = [item for item in attempts if item is not None]
        _rewrite_as_legacy_typed_deferrals(
            daemon,
            exact_attempts,
            reasons,
        )
        attempt, _budget = _block_with_legacy_leftover_wait_budget(
            daemon,
            exact_attempts,
        )

        outcomes = (
            daemon.reconcile_blocked_leftover_wait_deferral_budget_recoveries()
        )
        assert outcomes == []
        assert daemon.task_source.get(attempt.task_cid).status == "blocked"
    finally:
        daemon.close()


@pytest.mark.parametrize("drift_surface", ("direct_seed", "reconciler_budget"))
def test_leftover_wait_recovery_rejects_self_hashed_reason_drift(
    tmp_path: Path,
    drift_surface: str,
) -> None:
    now = {"ms": 1_000}
    reasons = [
        "inflight_process",
        "external_protected_checkout_recovery_required",
    ]
    observed_attempts: list[DatabaseTaskAttempt] = []

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        observed_attempts.append(attempt)
        raise DatabasePortalBridgeDeferred(
            reasons[len(observed_attempts) - 1],
            backoff_seconds=30,
        )

    daemon = _open_daemon(
        tmp_path,
        session="session:leftover-wait-rearm-reason-drift",
        provider_fn=provider,
        lease_ms=5_000,
        max_task_attempts=2,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        daemon.run_once()
        now["ms"] = 31_001
        daemon.run_once()
        attempts = [
            daemon.get_attempt(item.attempt_id) for item in observed_attempts
        ]
        assert all(item is not None for item in attempts)
        exact_attempts = [item for item in attempts if item is not None]
        _rewrite_as_legacy_typed_deferrals(
            daemon,
            exact_attempts,
            reasons,
        )
        budget = _legacy_leftover_wait_budget(daemon, exact_attempts)
        if drift_surface == "reconciler_budget":
            forged_budget = json.loads(json.dumps(budget))
            forged_budget["matching_attempts"][0]["reason"] = (
                "inflight_process"
            )
            matching_digest = hashlib.sha256()
            for identity in forged_budget["matching_attempts"]:
                encoded = json.dumps(
                    identity,
                    separators=(",", ":"),
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
                matching_digest.update(len(encoded).to_bytes(8, "big"))
                matching_digest.update(encoded)
            forged_budget["matching_attempts_digest"] = (
                "sha256:" + matching_digest.hexdigest()
            )
            forged_budget.pop("observation_id")
            forged_budget["observation_id"] = (
                daemon._database_portal_evidence_digest(forged_budget)
            )
            attempt, _ = _block_with_legacy_leftover_wait_budget(
                daemon,
                exact_attempts,
                budget_override=forged_budget,
            )
            outcomes = (
                daemon.reconcile_blocked_leftover_wait_deferral_budget_recoveries()
            )
            assert len(outcomes) == 1
            assert outcomes[0]["changed"] is False
            assert outcomes[0]["reason"] == (
                "leftover_wait_deferral_budget_recovery_not_admitted"
            )
        else:
            attempt, budget = _block_with_legacy_leftover_wait_budget(
                daemon,
                exact_attempts,
            )
            legitimate = (
                daemon._finalize_leftover_wait_deferral_budget_recovery_receipt(
                    attempt=attempt,
                    blocked_retry_budget=budget,
                )
            )
            forged = json.loads(json.dumps(legitimate))
            forged["exhausting_reasons"] = ["inflight_process"]
            forged_body = dict(forged)
            forged_body.pop("receipt_id")
            forged["receipt_id"] = daemon._database_portal_evidence_digest(
                forged_body
            )

            with pytest.raises(
                DatabaseImplementationAuthorityError,
                match="failed independent verification",
            ):
                daemon.recover_blocked_leftover_wait_deferral_budget_retry(
                    attempt,
                    recovery_evidence=forged,
                )
        assert daemon.task_source.get(attempt.task_cid).status == "blocked"
    finally:
        daemon.close()


@pytest.mark.parametrize("leftover_wait_count", (17, 65))
def test_typed_deferral_budget_paginates_past_leftover_wait_history(
    tmp_path: Path,
    leftover_wait_count: int,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session=f"session:typed-history-pagination:{leftover_wait_count}",
        max_task_attempts=3,
    )
    try:
        task_cid = "task:cid:001"
        reasons = [
            "validation_project_dependency_preflight_failed",
            "validation_project_dependency_preflight_failed",
            *(["inflight_process"] * leftover_wait_count),
            "validation_project_dependency_preflight_failed",
        ]
        attempts: list[DatabaseTaskAttempt] = []
        connection = daemon._require_connection()
        for index, reason in enumerate(reasons, start=1):
            attempt = DatabaseTaskAttempt(
                attempt_id=f"attempt:pagination:{index:03d}",
                claim_id=f"claim:pagination:{index:03d}",
                task_cid=task_cid,
                task_alias="DQP-T001",
                attempt_number=index,
                owner_session_id=daemon.owner_session_id,
                fencing_token=index,
                fence_epoch=1,
                lease_id=f"lease:pagination:{index:03d}",
                committed_phase="failed",
                status="failed",
                started_at_ms=index,
                finished_at_ms=index + 1,
                revision=2,
                body={},
            )
            typed = daemon._typed_deferral_receipt(attempt, reason=reason)
            phase_body = {
                "reason": reason,
                "portal_retryable_failure": True,
                "deferred": True,
                "attempt_consumed": False,
                "provider_dispatched": False,
                "typed_deferral_slot_consumed": True,
                "typed_deferral": typed,
            }
            connection.execute(
                """
                INSERT INTO database_task_attempts(
                    attempt_id, claim_id, task_cid, task_alias,
                    attempt_number, owner_session_id, fencing_token,
                    fence_epoch, lease_id, committed_phase, status,
                    started_at_ms, finished_at_ms, revision, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    attempt.attempt_id,
                    attempt.claim_id,
                    attempt.task_cid,
                    attempt.task_alias,
                    attempt.attempt_number,
                    attempt.owner_session_id,
                    attempt.fencing_token,
                    attempt.fence_epoch,
                    attempt.lease_id,
                    attempt.committed_phase,
                    attempt.status,
                    attempt.started_at_ms,
                    attempt.finished_at_ms,
                    attempt.revision,
                    "{}",
                ],
            )
            connection.execute(
                """
                INSERT INTO attempt_phases(
                    attempt_id, phase, committed_at_ms, fencing_token,
                    fence_epoch, revision, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    attempt.attempt_id,
                    "failed",
                    attempt.finished_at_ms,
                    attempt.fencing_token,
                    attempt.fence_epoch,
                    attempt.revision,
                    json.dumps(phase_body, separators=(",", ":"), sort_keys=True),
                ],
            )
            attempts.append(attempt)

        budget = daemon._typed_deferral_budget_observation(attempts[-1])

        assert budget is not None
        assert budget["exhausted"] is True
        assert budget["typed_deferral_count"] == 3
        assert budget["verified_typed_deferral_count"] == 3
        assert budget["verified_count_complete"] is True
        assert budget["typed_deferral_count_is_lower_bound"] is False
        assert len(budget["matching_attempts"]) == 3
    finally:
        daemon.close()


def test_legacy_failed_claim_does_not_consume_typed_deferral_budget(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}
    provider_attempts: list[str] = []

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_attempts.append(attempt.attempt_id)
        raise DatabasePortalBridgeDeferred(
            "validation_project_dependency_preflight_failed",
            backoff_seconds=300,
        )

    daemon = _open_daemon(
        tmp_path,
        session="session:legacy-deferral-migration",
        provider_fn=provider,
        lease_ms=5_000,
        max_task_attempts=1,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        legacy = daemon.claim_next()
        assert legacy is not None
        legacy = daemon.commit_phase(legacy, "context")
        legacy = daemon.commit_phase(
            legacy,
            "failed",
            body={
                "reason": "validation_project_dependency_preflight_failed",
                "portal_retryable_failure": True,
                # Deliberately pre-fix: no explicit deferred disposition or
                # identity-bound typed-deferral receipt.
            },
        )

        recovered = daemon.reconcile_terminal_retry_states()
        assert len(recovered) == 1
        assert recovered[0]["status"] == "retrying"
        assert daemon.task_source.get(legacy.task_cid).status == "retrying"

        # The first patch-era closed typed deferral consumes slot one and
        # blocks.  The legacy claim did not pre-exhaust the migration budget.
        now["ms"] = 301_001
        typed = daemon.run_once()
        assert len(provider_attempts) == 1
        assert typed["implementation_result"]["retry_budget_exhausted"] is True
        assert typed["implementation_result"]["terminal_state"][
            "retry_budget"
        ]["typed_deferral_count"] == 1
    finally:
        daemon.close()


def test_restart_reconciles_exhausted_typed_deferral_without_new_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = {"ms": 1_000}
    provider_attempts: list[str] = []

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_attempts.append(attempt.attempt_id)
        raise DatabasePortalBridgeDeferred(
            "validation_project_dependency_preflight_failed",
            backoff_seconds=300,
        )

    first = _open_daemon(
        tmp_path,
        session="session:typed-budget-restart",
        provider_fn=provider,
        lease_ms=5_000,
        max_task_attempts=2,
        clock_ms=lambda: now["ms"],
    )
    try:
        first.materialize_population(_population(1))
        initial = first.run_once()
        task_cid = str(initial["claimed_task_cid"])
        now["ms"] = 301_001

        def crash_before_block(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError("simulated crash before exhausted control CAS")

        monkeypatch.setattr(
            first,
            "_persist_typed_deferral_budget_exhausted",
            crash_before_block,
        )
        interrupted = first.run_once()
        assert "simulated crash" in interrupted["implementation_result"][
            "fail_error"
        ]
        control = first.task_source.get(task_cid)
        assert control is not None
        assert control.status == "in_progress"
        assert len(provider_attempts) == 2
    finally:
        first.close()

    monkeypatch.undo()
    now["ms"] = 307_000
    replacement = _open_daemon(
        tmp_path,
        session="session:typed-budget-restart",
        provider_fn=provider,
        lease_ms=5_000,
        max_task_attempts=2,
        clock_ms=lambda: now["ms"],
    )
    try:
        reconciled = replacement.run_once()
        assert reconciled["implementation_result"] is None
        assert len(reconciled["terminal_retry_reconciliations"]) == 1
        terminal = reconciled["terminal_retry_reconciliations"][0]
        assert terminal["status"] == "blocked"
        assert terminal["retry_budget"]["typed_deferral_count"] == 2
        assert replacement.task_source.get(task_cid).status == "blocked"
        assert len(provider_attempts) == 2
    finally:
        replacement.close()


def test_exhaustion_blocks_already_retrying_task_and_bounds_evidence_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = {"ms": 1_000}

    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeDeferred(
            "validation_project_dependency_preflight_failed",
            backoff_seconds=300,
        )

    daemon = _open_daemon(
        tmp_path,
        session="session:retrying-budget-reconciliation",
        provider_fn=provider,
        lease_ms=5_000,
        max_task_attempts=2,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        first = daemon.run_once()
        task_cid = str(first["claimed_task_cid"])
        now["ms"] = 301_001

        # Model a pre-budget writer that durably persisted the second typed
        # deferral and queue/CAS but crashed before checking exhaustion.
        original_observation = daemon._typed_deferral_budget_observation
        monkeypatch.setattr(
            daemon,
            "_typed_deferral_budget_observation",
            lambda _attempt: None,
        )
        second = daemon.run_once()
        assert second["implementation_result"]["retry_state"][
            "status"
        ] == "retrying"
        assert daemon.task_source.get(task_cid).status == "retrying"
        assert daemon.task_source.get_queue_entry(task_cid) is not None

        monkeypatch.setattr(
            daemon,
            "_typed_deferral_budget_observation",
            original_observation,
        )
        implementation_daemon_module = importlib.import_module(
            "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon"
        )
        monkeypatch.setattr(
            implementation_daemon_module,
            "_MAX_TYPED_DEFERRAL_ATTEMPT_PREVIEW",
            1,
        )
        reconciled = daemon.reconcile_terminal_retry_states()

        assert len(reconciled) == 1
        terminal = reconciled[0]
        assert terminal["status"] == "blocked"
        assert terminal["control_previous_status"] == "retrying"
        assert terminal["prior_queue_entry_preserved_inactive"] is True
        budget = terminal["retry_budget"]
        assert budget["typed_deferral_count"] == 2
        assert budget["verified_typed_deferral_count"] == 2
        assert budget["verified_count_complete"] is True
        assert len(budget["matching_attempts"]) == 1
        assert budget["matching_attempts_truncated"] is True
        assert budget["omitted_matching_attempt_count"] == 1
        assert budget["matching_attempts_digest"].startswith("sha256:")
        assert daemon.task_source.get(task_cid).status == "blocked"
    finally:
        daemon.close()


def test_typed_deferral_from_old_state_schema_does_not_consume_current_budget(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:typed-budget-schema-generation",
        max_task_attempts=1,
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        attempt = daemon.commit_phase(attempt, "context")
        daemon.state_schema_revision = "state-schema-old"
        typed = daemon._typed_deferral_receipt(
            attempt,
            reason="typed_schema_migration_deferral",
        )
        attempt = daemon.commit_phase(
            attempt,
            "failed",
            body={
                "reason": "typed_schema_migration_deferral",
                "portal_retryable_failure": True,
                "portal_terminal_failure": False,
                "deferred": True,
                "attempt_consumed": False,
                "provider_dispatched": False,
                "typed_deferral_slot_consumed": True,
                "backoff_seconds": 300,
                "typed_deferral": typed,
            },
        )

        daemon.state_schema_revision = "state-schema-new"
        evidence = daemon._terminal_retry_evidence(attempt)
        assert evidence is not None
        assert evidence["typed_deferral_budget"] is None
    finally:
        daemon.close()


def test_restart_reconciles_failed_execution_and_expired_coordination_claim(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}
    first = _open_daemon(
        tmp_path,
        session="session:terminal-retry-recovery",
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        first.materialize_population(_population(1))
        failed_attempt = first.claim_next()
        assert failed_attempt is not None
        failed_attempt = first.commit_phase(failed_attempt, "context")
        failed_attempt = first.commit_phase(
            failed_attempt,
            "failed",
            body={
                "reason": "validation_project_dependency_preflight_failed",
                "portal_retryable_failure": True,
                # Pre-fix receipt: no typed backoff_seconds field.
            },
        )
        task_before = first.task_source.get(failed_attempt.task_cid)
        assert task_before is not None
        assert task_before.status == "in_progress"
        assert first.task_source.get_queue_entry(failed_attempt.task_cid) is None
    finally:
        first.close()

    # The legacy 300-second window elapsed while the supervisor was down.
    now["ms"] = 401_000
    replacement = _open_daemon(
        tmp_path,
        session="session:terminal-retry-recovery",
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        reconciliations = replacement.reconcile_terminal_retry_states()
        assert len(reconciliations) == 1
        reconciliation = reconciliations[0]
        assert reconciliation["backoff_ms"] == 0
        assert reconciliation["retry_not_before_ms"] == 401_000
        assert reconciliation["control_previous_status"] == "in_progress"
        assert reconciliation["control_previous_revision"] == 2
        assert reconciliation["control_new_status"] == "retrying"
        assert reconciliation["control_new_revision"] == 3
        assert reconciliation["coordination"]["expired_now"] is True
        assert reconciliation["coordination"]["claim_state"] == "expired"
        assert reconciliation["coordination"][
            "coordination_attempt_status"
        ] == "expired"

        recovered = replacement.task_source.get(failed_attempt.task_cid)
        assert recovered is not None
        assert recovered.status == "retrying"
        queue_entry = replacement.task_source.get_queue_entry(
            failed_attempt.task_cid
        )
        assert queue_entry is not None
        queue_attempt = queue_entry.attempt

        # Reconciliation is idempotent after both durable writes landed.
        assert replacement.reconcile_terminal_retry_states() == []
        repeated_entry = replacement.task_source.get_queue_entry(
            failed_attempt.task_cid
        )
        assert repeated_entry is not None
        assert repeated_entry.attempt == queue_attempt
        assert repeated_entry.retry_not_before_ms == 401_000

        replacement_attempt = replacement.claim_next()
        assert replacement_attempt is not None
        assert replacement_attempt.attempt_id != failed_attempt.attempt_id
        reclaimed = replacement.task_source.get(failed_attempt.task_cid)
        assert reclaimed is not None
        assert reclaimed.status == "in_progress"
        assert reclaimed.revision == 4
    finally:
        replacement.close()


def test_retry_reconciliation_reuses_attempt_bound_queue_after_cas_crash(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}
    daemon = _open_daemon(
        tmp_path,
        session="session:queue-cas-crash",
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        failed_attempt = daemon.claim_next()
        assert failed_attempt is not None
        failed_attempt = daemon.commit_phase(failed_attempt, "context")
        failed_attempt = daemon.commit_phase(
            failed_attempt,
            "failed",
            body={
                "reason": "typed_deferral",
                "portal_retryable_failure": True,
                "backoff_seconds": 300,
            },
        )
        queue_reason = (
            f"database_portal_retry:{failed_attempt.attempt_id}:typed_deferral"
        )
        daemon.task_source.record_queue_backoff(
            task_cid=failed_attempt.task_cid,
            delay_ms=300_000,
            reason=queue_reason,
        )
        before = daemon.task_source.get_queue_entry(failed_attempt.task_cid)
        assert before is not None

        # Simulate restart after queue commit but before control CAS.
        now["ms"] = 2_000
        reconciliations = daemon.reconcile_terminal_retry_states()
        assert len(reconciliations) == 1
        assert reconciliations[0]["queue_reused"] is True
        after = daemon.task_source.get_queue_entry(failed_attempt.task_cid)
        assert after is not None
        assert after.attempt == before.attempt
        assert after.retry_not_before_ms == before.retry_not_before_ms
        task = daemon.task_source.get(failed_attempt.task_cid)
        assert task is not None
        assert task.status == "retrying"
        assert task.revision == 3
    finally:
        daemon.close()


def test_typed_blocked_recovery_fails_before_cooldown_write() -> None:
    writes: list[str] = []
    captured: dict[str, object] = {}

    class _TypedRecoverySource:
        def __init__(self) -> None:
            self.entry: SimpleNamespace | None = None
            self.task = SimpleNamespace(
                task_cid="task:typed-blocked-recovery",
                status="blocked",
                revision=3,
                body={
                    "completion_receipt": {
                        "operation": "database_portal_terminal_failure"
                    }
                },
            )

        def get(self, _task_cid: str) -> SimpleNamespace:
            return self.task

        def get_queue_entry(self, _task_cid: str) -> SimpleNamespace | None:
            return self.entry

        def record_task_retry_cooldown(self, **kwargs: object) -> SimpleNamespace:
            writes.append("cooldown")
            captured.update(kwargs)
            self.entry = SimpleNamespace(
                retry_not_before_ms=1_000,
                reason=kwargs["reason"],
            )
            return SimpleNamespace(
                changed=True,
                to_dict=lambda: {"changed": True},
            )

    source = _TypedRecoverySource()
    daemon = SimpleNamespace(
        task_source=source,
        _database_portal_backoff_ms=lambda value: int(value),
        _now_ms=lambda: 1_000,
    )
    attempt = DatabaseTaskAttempt(
        attempt_id="attempt:typed-blocked-recovery",
        claim_id="claim:typed-blocked-recovery",
        task_cid=source.task.task_cid,
        task_alias="CASF-TYPED-BLOCKED-RECOVERY",
        attempt_number=2,
        owner_session_id="session:typed-blocked-recovery",
        fencing_token=31,
        fence_epoch=17,
        lease_id="lease:typed-blocked-recovery",
        committed_phase=ATTEMPT_PHASE_FAILED,
        status="failed",
        started_at_ms=100,
        finished_at_ms=900,
        revision=5,
    )

    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="coordination-coupled owner authority",
    ):
        DatabaseImplementationDaemon._persist_task_retry_state(
            daemon,
            attempt,
            reason="portal\r\ncandidate\x00retry",
            backoff_ms=0,
            evidence_source="portal_provider_failed_reclassified",
            allow_blocked_recovery=True,
        )

    assert writes == []
    assert captured == {}
    assert source.entry is None
    assert source.task.status == "blocked"
    assert source.task.revision == 3


def test_typed_protected_blocked_recovery_fails_before_queue_write() -> None:
    writes: list[str] = []

    class _TypedProtectedSource:
        def __init__(self) -> None:
            self.entry: SimpleNamespace | None = None
            self.task = SimpleNamespace(
                task_cid="task:typed-protected-recovery",
                status="blocked",
                revision=3,
                body={
                    "completion_receipt": {
                        "operation": (
                            "database_portal_typed_deferral_budget_exhausted"
                        )
                    }
                },
            )

        def get(self, _task_cid: str) -> SimpleNamespace:
            return self.task

        def get_queue_entry(self, _task_cid: str) -> SimpleNamespace | None:
            return self.entry

        def record_task_retry_cooldown(self, **_kwargs: object) -> None:
            writes.append("cooldown")
            self.entry = SimpleNamespace(reason="stale")
            raise AssertionError("protected recovery must fail before queue mutation")

    source = _TypedProtectedSource()
    daemon = SimpleNamespace(
        task_source=source,
        _database_portal_backoff_ms=lambda value: int(value),
        _now_ms=lambda: 1_000,
    )
    attempt = DatabaseTaskAttempt(
        attempt_id="attempt:typed-protected-recovery",
        claim_id="claim:typed-protected-recovery",
        task_cid=source.task.task_cid,
        task_alias="CASF-TYPED-PROTECTED-RECOVERY",
        attempt_number=1,
        owner_session_id="session:typed-protected-recovery",
        fencing_token=31,
        fence_epoch=17,
        lease_id="lease:typed-protected-recovery",
        committed_phase=ATTEMPT_PHASE_FAILED,
        status="failed",
        started_at_ms=100,
        finished_at_ms=900,
        revision=5,
    )

    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="coordination-coupled owner authority",
    ):
        DatabaseImplementationDaemon._persist_task_retry_state(
            daemon,
            attempt,
            reason="leftover_wait_deferral_budget_cleared",
            backoff_ms=0,
            evidence_source="typed_leftover_wait_recovery",
            leftover_wait_deferral_budget_recovery_evidence={
                "receipt_id": "sha256:" + ("a" * 64)
            },
        )

    assert writes == []
    assert source.entry is None
    assert source.task.status == "blocked"
    assert source.task.revision == 3


def test_embedded_expanded_blocked_recovery_uses_atomic_queue_status() -> None:
    calls: list[str] = []

    class _EmbeddedRecoverySource:
        def __init__(self) -> None:
            self.entry: SimpleNamespace | None = None
            self.task = SimpleNamespace(
                task_cid="task:embedded-expanded-recovery",
                task_alias="CASF-EMBEDDED-EXPANDED-RECOVERY",
                goal_cid="goal:embedded-expanded-recovery",
                status="blocked",
                revision=3,
                body={
                    "completion_receipt": {
                        "operation": "database_portal_terminal_failure"
                    }
                },
            )

        def get(self, _task_cid: str) -> SimpleNamespace:
            return self.task

        def get_queue_entry(self, _task_cid: str) -> SimpleNamespace | None:
            return self.entry

        def record_queue_backoff(self, **_kwargs: object) -> None:
            calls.append("legacy-queue")
            self.entry = SimpleNamespace(reason="stale")
            raise AssertionError("expanded recovery must not write queue first")

        def record_queue_backoff_and_cas_status(
            self,
            **kwargs: object,
        ) -> None:
            calls.append("atomic")
            assert kwargs["expected_revision"] == 3
            assert kwargs["status"] == "retrying"
            assert kwargs["expected_control_receipt"] == {
                "operation": "database_portal_terminal_failure"
            }
            raise DatabaseImplementationConflictError("atomic CAS conflict")

    source = _EmbeddedRecoverySource()
    daemon = SimpleNamespace(
        task_source=source,
        _database_portal_backoff_ms=lambda value: int(value),
        _now_ms=lambda: 1_000,
        _require_control_attempt_receipt=(
            lambda _task, _attempt, **_kwargs: dict(
                source.task.body["completion_receipt"]
            )
        ),
        _execute_with_retry_transition_authority=(
            lambda _attempt, _coordination, transition: transition()
        ),
        _cas_task_status_database=(
            lambda *_args, **_kwargs: calls.append("legacy-cas")
        ),
    )
    attempt = DatabaseTaskAttempt(
        attempt_id="attempt:embedded-expanded-recovery",
        claim_id="claim:embedded-expanded-recovery",
        task_cid=source.task.task_cid,
        task_alias=source.task.task_alias,
        attempt_number=1,
        owner_session_id="session:embedded-expanded-recovery",
        fencing_token=31,
        fence_epoch=17,
        lease_id="lease:embedded-expanded-recovery",
        committed_phase=ATTEMPT_PHASE_FAILED,
        status="failed",
        started_at_ms=100,
        finished_at_ms=900,
        revision=5,
    )

    with pytest.raises(
        DatabaseImplementationConflictError,
        match="atomic CAS conflict",
    ):
        DatabaseImplementationDaemon._persist_task_retry_state(
            daemon,
            attempt,
            reason="portal_provider_failed",
            backoff_ms=0,
            evidence_source="pooled_worktree_create_recovery",
            coordination_evidence={"claim_state": "accepted"},
            pooled_worktree_create_recovery_evidence={
                "receipt_id": "receipt:expanded-recovery"
            },
        )

    assert calls == ["atomic"]
    assert source.entry is None
    assert source.task.status == "blocked"
    assert source.task.revision == 3


def test_legacy_blocked_recovery_fails_before_queue_without_atomic_surface() -> None:
    calls: list[str] = []

    class _LegacyRecoverySource:
        entry: SimpleNamespace | None = None
        task = SimpleNamespace(
            task_cid="task:legacy-blocked-recovery",
            task_alias="CASF-LEGACY-BLOCKED-RECOVERY",
            goal_cid="goal:legacy-blocked-recovery",
            status="blocked",
            revision=3,
            body={
                "completion_receipt": {
                    "operation": "database_portal_terminal_failure"
                }
            },
        )

        def get(self, _task_cid: str) -> SimpleNamespace:
            return self.task

        def get_queue_entry(self, _task_cid: str) -> SimpleNamespace | None:
            return self.entry

        def record_queue_backoff(self, **_kwargs: object) -> None:
            calls.append("legacy-queue")
            self.entry = SimpleNamespace(reason="stale")

    source = _LegacyRecoverySource()
    daemon = SimpleNamespace(
        task_source=source,
        _database_portal_backoff_ms=lambda value: int(value),
        _now_ms=lambda: 1_000,
        _cas_task_status_database=(
            lambda *_args, **_kwargs: calls.append("legacy-cas")
        ),
    )
    attempt = DatabaseTaskAttempt(
        attempt_id="attempt:legacy-blocked-recovery",
        claim_id="claim:legacy-blocked-recovery",
        task_cid=source.task.task_cid,
        task_alias=source.task.task_alias,
        attempt_number=1,
        owner_session_id="session:legacy-blocked-recovery",
        fencing_token=31,
        fence_epoch=17,
        lease_id="lease:legacy-blocked-recovery",
        committed_phase=ATTEMPT_PHASE_FAILED,
        status="failed",
        started_at_ms=100,
        finished_at_ms=900,
        revision=5,
    )

    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="atomic queue/status authority",
    ):
        DatabaseImplementationDaemon._persist_task_retry_state(
            daemon,
            attempt,
            reason="portal_provider_failed",
            backoff_ms=0,
            evidence_source="legacy_blocked_recovery",
            coordination_evidence={"claim_state": "accepted"},
            allow_blocked_recovery=True,
        )

    assert calls == []
    assert source.entry is None
    assert source.task.status == "blocked"
    assert source.task.revision == 3


@pytest.mark.parametrize(
    ("error_type", "error_message"),
    (
        (
            DatabaseImplementationAuthorityError,
            "prepared completion barrier is unavailable",
        ),
        (
            DatabaseImplementationConflictError,
            "coordination identity or fence differs",
        ),
    ),
)
@pytest.mark.parametrize(
    "recovery_method",
    (
        "recover_blocked_portal_inflight_process_retry",
        "recover_blocked_portal_validation_retry_seed_conflict_retry",
        "recover_blocked_leftover_wait_deferral_budget_retry",
        "recover_blocked_portal_pooled_worktree_create_retry",
    ),
)
def test_blocked_recovery_propagates_coordination_errors_without_mutation(
    recovery_method: str,
    error_type: type[Exception],
    error_message: str,
) -> None:
    """Coordination failures cannot be laundered into claim-absent authority."""

    attempt = DatabaseTaskAttempt(
        attempt_id="attempt:coordination-error",
        claim_id="claim:coordination-error",
        task_cid="task:coordination-error",
        task_alias="CASF-COORDINATION-ERROR",
        attempt_number=1,
        owner_session_id="session:coordination-error",
        fencing_token=31,
        fence_epoch=17,
        lease_id="lease:coordination-error",
        committed_phase=ATTEMPT_PHASE_FAILED,
        status="failed",
        started_at_ms=100,
        finished_at_ms=900,
        revision=5,
    )
    task = SimpleNamespace(
        task_cid=attempt.task_cid,
        status="blocked",
        revision=7,
        body={"completion_receipt": {"operation": "protected-block"}},
    )

    class _Source:
        queue_entry: object | None = None

        @staticmethod
        def get(_task_cid: str) -> SimpleNamespace:
            return task

        def get_queue_entry(self, _task_cid: str) -> object | None:
            return self.queue_entry

    source = _Source()
    mutation_calls: list[dict[str, object]] = []

    def reject_coordination(_attempt: DatabaseTaskAttempt) -> None:
        raise error_type(error_message)

    def unexpected_persist(
        _attempt: DatabaseTaskAttempt, **kwargs: object
    ) -> dict[str, object]:
        mutation_calls.append(dict(kwargs))
        return {}

    phase_reason = {
        "recover_blocked_portal_inflight_process_retry": "inflight_process",
        "recover_blocked_portal_validation_retry_seed_conflict_retry": (
            "Portal retry seed state conflicts with its source receipt"
        ),
        "recover_blocked_portal_pooled_worktree_create_retry": (
            "portal_provider_failed"
        ),
    }.get(recovery_method, "leftover_wait")
    verified = {"receipt_id": "sha256:" + ("a" * 64)}
    daemon = SimpleNamespace(
        task_source=source,
        _require_execution_authority=lambda _purpose: None,
        get_attempt=lambda _attempt_id: attempt,
        _latest_failed_attempts=lambda: [attempt],
        phase_history=lambda _attempt_id: [
            {
                "phase": ATTEMPT_PHASE_FAILED,
                "body": {
                    "portal_terminal_failure": True,
                    "portal_retryable_failure": False,
                    "reason": phase_reason,
                },
            }
        ],
        _automatic_claim_forbidden=lambda _task: False,
        _verified_inflight_process_recovery_receipt=(
            lambda *_args, **_kwargs: dict(verified)
        ),
        _verified_validation_retry_seed_conflict_recovery_receipt=(
            lambda *_args, **_kwargs: dict(verified)
        ),
        _verified_blocked_leftover_wait_retry_budget=(
            lambda *_args, **_kwargs: {"exhausted": True}
        ),
        _verified_leftover_wait_deferral_budget_recovery_receipt=(
            lambda *_args, **_kwargs: dict(verified)
        ),
        _verified_pooled_worktree_create_recovery_receipt=(
            lambda *_args, **_kwargs: dict(verified)
        ),
        _reconcile_failed_attempt_coordination=reject_coordination,
        _reconcile_leftover_wait_coordination=reject_coordination,
        _persist_task_retry_state=unexpected_persist,
    )
    original_body = json.loads(json.dumps(task.body))
    method = getattr(DatabaseImplementationDaemon, recovery_method)

    with pytest.raises(error_type, match=error_message):
        method(
            daemon,
            attempt,
            recovery_evidence={"receipt_id": verified["receipt_id"]},
        )

    assert mutation_calls == []
    assert task.status == "blocked"
    assert task.revision == 7
    assert task.body == original_body
    assert source.get_queue_entry(task.task_cid) is None


def test_typed_retrying_reuse_requires_exact_task_queue_validation() -> None:
    calls: list[str] = []

    class _TypedRetryingSource:
        task = SimpleNamespace(
            task_cid="task:typed-retrying-exact-reuse",
            status="retrying",
            revision=4,
            body={"completion_receipt": {"operation": "database_portal_retry"}},
        )
        entry = SimpleNamespace(
            retry_not_before_ms=5_000,
            reason=(
                "database_portal_retry:attempt:typed-retrying-exact-reuse:"
                "typed_deferral"
            ),
        )

        def get(self, _task_cid: str) -> SimpleNamespace:
            return self.task

        def get_queue_entry(self, _task_cid: str) -> SimpleNamespace:
            return self.entry

        @staticmethod
        def record_task_retry_cooldown(**_kwargs: object) -> None:
            raise AssertionError("already-retrying path must not write a new digest")

        @staticmethod
        def validate_retrying_task_cooldown(
            _task_cid: str,
            **kwargs: object,
        ) -> None:
            calls.append("validate")
            assert kwargs["expected_attempt_identity"] == {
                "attempt_id": "attempt:typed-retrying-exact-reuse",
                "claim_id": "claim:typed-retrying-exact-reuse",
                "lease_id": "lease:typed-retrying-exact-reuse",
                "owner_session_id": "session:typed-retrying-exact-reuse",
                "attempt_number": 1,
                "fencing_token": 7,
                "fence_epoch": 3,
            }
            raise DatabaseImplementationAuthorityError(
                "retrying task receipt differs from its typed cooldown"
            )

    source = _TypedRetryingSource()
    daemon = SimpleNamespace(
        task_source=source,
        _database_portal_backoff_ms=lambda value: int(value),
        _now_ms=lambda: 5_000,
        _protect_retry_transition_authority=(
            lambda _attempt, _coordination: None
        ),
    )
    attempt = DatabaseTaskAttempt(
        attempt_id="attempt:typed-retrying-exact-reuse",
        claim_id="claim:typed-retrying-exact-reuse",
        task_cid=source.task.task_cid,
        task_alias="CASF-TYPED-RETRYING-EXACT-REUSE",
        attempt_number=1,
        owner_session_id="session:typed-retrying-exact-reuse",
        fencing_token=7,
        fence_epoch=3,
        lease_id="lease:typed-retrying-exact-reuse",
        committed_phase=ATTEMPT_PHASE_FAILED,
        status="failed",
        started_at_ms=100,
        finished_at_ms=900,
        revision=5,
    )

    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="differs from its typed cooldown",
    ):
        DatabaseImplementationDaemon._persist_task_retry_state(
            daemon,
            attempt,
            reason="typed_deferral",
            backoff_ms=0,
            evidence_source="portal_provider_failed",
        )

    assert calls == ["validate"]


def test_retrying_cooldown_repair_payload_follows_control_receipt() -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_database_task_source import (
        TypedDatabaseTaskSource,
    )

    receipt = {
        "operation": "database_portal_validation_retry_recovery",
        "attempt_id": "attempt:stale-cooldown",
        "claim_id": "claim:stale-cooldown",
        "lease_id": "lease:stale-cooldown",
        "owner_session_id": "session:stale-cooldown",
        "attempt_number": 629,
        "fencing_token": 629,
        "fence_epoch": 629,
        "queue_reason": (
            "database_portal_retry:attempt:stale-cooldown:"
            "portal_pending_merge_claim_retry"
        ),
        "backoff_ms": 0,
        "retry_not_before_ms": 1_000,
        "control_expected_revision": 12,
    }
    task = SimpleNamespace(
        task_cid="task:stale-cooldown",
        task_alias="PCSM-010",
        status="retrying",
        revision=13,
        body={"completion_receipt": receipt},
    )
    payload = TypedDatabaseTaskSource._retrying_cooldown_repair_payload(task)
    assert payload == {
        "task_cid": "task:stale-cooldown",
        "expected_task_revision": 12,
        "expected_task_status": "retrying",
        "attempt_id": "attempt:stale-cooldown",
        "claim_id": "claim:stale-cooldown",
        "lease_id": "lease:stale-cooldown",
        "owner_session_id": "session:stale-cooldown",
        "attempt_number": 629,
        "fencing_token": 629,
        "fence_epoch": 629,
        "delay_ms": 0,
        "reason": receipt["queue_reason"],
        "now_ms": 1_000,
    }
    stale = dict(receipt)
    stale["control_expected_revision"] = 8
    task.body = {"completion_receipt": stale}
    assert TypedDatabaseTaskSource._retrying_cooldown_repair_payload(task) is None


def test_reconcile_retrying_cooldown_bindings_uses_typed_repair() -> None:
    calls: list[str] = []

    class _TypedSource:
        @staticmethod
        def repair_retrying_cooldown_bindings() -> list[dict[str, object]]:
            calls.append("repair")
            return [
                {
                    "task_cid": "task:stale-cooldown",
                    "task_alias": "PCSM-010",
                    "changed": True,
                    "reason": "retrying_cooldown_rebound_to_control_receipt",
                }
            ]

    daemon = SimpleNamespace(
        task_source=_TypedSource(),
        _require_execution_authority=lambda _name: None,
    )
    outcomes = DatabaseImplementationDaemon.reconcile_retrying_cooldown_bindings(
        daemon
    )
    assert calls == ["repair"]
    assert outcomes == [
        {
            "task_cid": "task:stale-cooldown",
            "task_alias": "PCSM-010",
            "changed": True,
            "reason": "retrying_cooldown_rebound_to_control_receipt",
        }
    ]

    empty = SimpleNamespace(
        task_source=SimpleNamespace(),
        _require_execution_authority=lambda _name: None,
    )
    assert (
        DatabaseImplementationDaemon.reconcile_retrying_cooldown_bindings(empty)
        == []
    )


def test_terminal_portal_reason_skips_failed_attempt_without_phase_receipt() -> None:
    daemon = SimpleNamespace(phase_history=lambda _attempt_id: [])
    attempt = SimpleNamespace(attempt_id="attempt:missing-failed-phase")
    assert (
        DatabaseImplementationDaemon._terminal_portal_failure_reason(
            daemon,
            attempt,
        )
        is None
    )
    assert (
        DatabaseImplementationDaemon._terminal_retry_evidence(
            daemon,
            attempt,
        )
        is None
    )


def test_reconcile_landed_merged_tasks_completes_retrying_when_outputs_landed() -> None:
    cas: list[dict[str, object]] = []

    class _Source:
        task = SimpleNamespace(
            task_cid="task:pcsm-010",
            task_alias="PCSM-010",
            status="retrying",
            revision=45,
            body={},
        )

        def list_tasks(self, status=None, limit=50):
            selected = (
                {str(status).strip().lower()}
                if isinstance(status, str)
                else {str(item).strip().lower() for item in (status or ())}
            )
            tasks = (self.task,) if self.task.status in selected else ()
            return SimpleNamespace(tasks=tasks)

        def get(self, _cid: str):
            return self.task

        def record_validation_result(self, **_kwargs: object) -> None:
            return None

    source = _Source()
    daemon = SimpleNamespace(
        repo_root=Path("/tmp"),
        merge_target_ref="HEAD",
        task_source=source,
        _task_outputs_landed_on_target=lambda _task: True,
        _task_declared_output_paths=lambda _task: (
            "artifacts/proof_carrying_semantic_minification/receipts/PCSM-010.json",
        ),
        _record_event=lambda *_args, **_kwargs: None,
    )

    def cas_status(task_cid, *, expected_revision, new_status, receipt, evidence_digests):
        cas.append(
            {
                "task_cid": task_cid,
                "expected_revision": expected_revision,
                "new_status": new_status,
                "evidence_digests": list(evidence_digests),
            }
        )
        source.task.status = new_status
        source.task.revision = int(expected_revision) + 1
        return None

    daemon._cas_task_status_database = cas_status
    daemon._landed_merge_repair_proof = (
        lambda task, attempt_id="": DatabaseImplementationDaemon._landed_merge_repair_proof(
            daemon,
            task,
            attempt_id=attempt_id,
        )
    )
    daemon._complete_landed_quarantined_task = (
        lambda task: DatabaseImplementationDaemon._complete_landed_quarantined_task(
            daemon,
            task,
        )
    )

    outcomes = DatabaseImplementationDaemon.reconcile_landed_merged_tasks(daemon)
    assert len(outcomes) == 1
    assert outcomes[0]["completed"] is True
    assert outcomes[0]["task_alias"] == "PCSM-010"
    assert outcomes[0]["reason"] == "database_landed_merge_repair"
    assert cas[0]["new_status"] == "completed"
    assert cas[0]["expected_revision"] == 45


def test_persist_retry_settles_when_cooldown_matches_receipt_not_attempt() -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        TaskSourceIntegrityError,
    )

    calls: list[str] = []
    entry = SimpleNamespace(
        retry_not_before_ms=1_000,
        reason=(
            "database_portal_retry:attempt:bound-receipt:"
            "portal_pending_merge_claim_retry"
        ),
    )

    class _TypedRetryingSource:
        task = SimpleNamespace(
            task_cid="task:bound-receipt-cooldown",
            status="retrying",
            revision=13,
            body={
                "completion_receipt": {
                    "operation": "database_portal_validation_retry_recovery"
                }
            },
        )

        def get(self, _task_cid: str) -> SimpleNamespace:
            return self.task

        def get_queue_entry(self, _task_cid: str) -> SimpleNamespace:
            return entry

        @staticmethod
        def record_task_retry_cooldown(**_kwargs: object) -> None:
            raise AssertionError("bound receipt cooldown must not be rewritten")

        @staticmethod
        def validate_retrying_task_cooldown(
            _task_cid: str,
            **kwargs: object,
        ) -> SimpleNamespace:
            if kwargs:
                calls.append("expected")
                raise TaskSourceIntegrityError(
                    "retrying task cooldown differs from the expected delay"
                )
            calls.append("bound")
            return entry

    source = _TypedRetryingSource()
    daemon = SimpleNamespace(
        task_source=source,
        _database_portal_backoff_ms=lambda value: int(value),
        _now_ms=lambda: 5_000,
        _protect_retry_transition_authority=(
            lambda _attempt, _coordination: None
        ),
    )
    attempt = DatabaseTaskAttempt(
        attempt_id="attempt:local-failed-delay",
        claim_id="claim:local-failed-delay",
        task_cid=source.task.task_cid,
        task_alias="PCSM-010",
        attempt_number=1,
        owner_session_id="session:local-failed-delay",
        fencing_token=1,
        fence_epoch=1,
        lease_id="lease:local-failed-delay",
        committed_phase=ATTEMPT_PHASE_FAILED,
        status="failed",
        started_at_ms=100,
        finished_at_ms=900,
        revision=5,
    )
    outcome = DatabaseImplementationDaemon._persist_task_retry_state(
        daemon,
        attempt,
        reason="typed_deferral",
        backoff_ms=300_000,
        evidence_source="portal_provider_failed",
    )
    assert calls == ["expected", "bound"]
    assert outcome["changed"] is False
    assert outcome["queue_reused"] is True
    assert outcome["reason"] == "retrying_cooldown_bound_to_control_receipt"
    assert outcome["retry_not_before_ms"] == 1_000


def test_retry_reconciliation_repairs_retrying_without_queue(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}
    daemon = _open_daemon(
        tmp_path,
        session="session:retrying-without-queue",
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        failed_attempt = daemon.claim_next()
        assert failed_attempt is not None
        failed_attempt = daemon.commit_phase(failed_attempt, "context")
        failed_attempt = daemon.commit_phase(
            failed_attempt,
            "failed",
            body={
                "reason": "typed_deferral",
                "portal_retryable_failure": True,
                "backoff_seconds": 300,
            },
        )
        control = daemon.task_source.get(failed_attempt.task_cid)
        assert control is not None
        daemon._cas_task_status_database(
            failed_attempt.task_cid,
            expected_revision=int(control.revision),
            new_status="retrying",
            receipt={"operation": "simulated_cas_before_queue_crash"},
        )
        assert daemon.task_source.get_queue_entry(failed_attempt.task_cid) is None

        repaired = daemon.reconcile_terminal_retry_states()

        assert len(repaired) == 1
        assert repaired[0]["status"] == "retrying"
        assert repaired[0]["changed"] is False
        entry = daemon.task_source.get_queue_entry(failed_attempt.task_cid)
        assert entry is not None
        assert entry.retry_not_before_ms == 301_000
    finally:
        daemon.close()


def test_retry_reconciliation_rejects_superseded_coordination_fence(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}
    daemon = _open_daemon(
        tmp_path,
        session="session:superseded-retry-fence",
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        failed_attempt = daemon.claim_next()
        assert failed_attempt is not None
        failed_attempt = daemon.commit_phase(failed_attempt, "context")
        failed_attempt = daemon.commit_phase(
            failed_attempt,
            "failed",
            body={
                "reason": "typed_deferral",
                "portal_retryable_failure": True,
                "backoff_seconds": 300,
            },
        )
        old_claim = daemon.coordinator.get_task_claim(failed_attempt.claim_id)
        assert old_claim is not None
        now["ms"] = 7_000
        daemon.coordinator.expire_task_claim(old_claim, now_ms=now["ms"])
        replacement = daemon.coordinator.claim_ready_task(
            owner_session_id="session:newer-fence",
            lease_ms=5_000,
            now_ms=now["ms"],
        )
        assert replacement is not None
        assert replacement.fencing_token > failed_attempt.fencing_token

        outcomes = daemon.reconcile_terminal_retry_states()
        assert len(outcomes) == 1
        assert outcomes[0]["changed"] is False
        assert outcomes[0]["status"] == "in_progress"
        assert outcomes[0]["reason"] == (
            "failed_attempt_coordination_superseded"
        )
        assert outcomes[0]["successor_claim_id"] == replacement.claim_id
        assert outcomes[0]["successor_attempt_id"] == replacement.attempt_id
        assert outcomes[0]["coordination"][
            "superseded_by_newer_fence"
        ] is True

        control = daemon.task_source.get(failed_attempt.task_cid)
        assert control is not None
        assert control.status == "in_progress"
        assert daemon.task_source.get_queue_entry(failed_attempt.task_cid) is None
    finally:
        daemon.close()


def test_retry_reconciliation_rejects_manual_task(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:manual-retry-rejection",
    )
    try:
        population = _population(1)
        tasks = population["tasks"]
        assert isinstance(tasks, list)
        tasks[0]["completion"] = "manual"
        daemon.materialize_population(population)
        control = daemon.task_source.get("task:cid:001")
        assert control is not None
        claim = daemon.coordinator.claim_ready_task(
            owner_session_id=daemon.owner_session_id,
            lease_ms=daemon.lease_ms,
            now_ms=daemon._now_ms(),
        )
        assert claim is not None
        daemon._protect_new_claim(claim)
        daemon._cas_task_status_database(
            control.task_cid,
            expected_revision=int(control.revision),
            new_status="in_progress",
            receipt={"operation": "simulated_legacy_manual_claim"},
        )
        failed_attempt = daemon._insert_attempt_from_claim(
            claim,
            task_alias=control.task_alias,
        )
        failed_attempt = daemon.commit_phase(failed_attempt, "context")
        failed_attempt = daemon.commit_phase(
            failed_attempt,
            "failed",
            body={
                "reason": "typed_deferral",
                "portal_retryable_failure": True,
                "backoff_seconds": 300,
            },
        )

        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="manual/review-only",
        ):
            daemon.reconcile_terminal_retry_states()

        unchanged = daemon.task_source.get(failed_attempt.task_cid)
        assert unchanged is not None
        assert unchanged.status == "in_progress"
        assert daemon.task_source.get_queue_entry(failed_attempt.task_cid) is None
    finally:
        daemon.close()


def test_current_control_recheck_rejects_task_that_became_manual(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(tmp_path, session="session:manual-race")
    try:
        daemon.materialize_population(_population(1))
        source = daemon.task_source
        original_get = source.get
        mutated = False

        def become_manual(task_cid: str) -> object:
            nonlocal mutated
            current = original_get(task_cid)
            if current is not None and not mutated:
                source._intent.upsert_task(
                    task_cid=current.task_cid,
                    task_alias=current.task_alias,
                    goal_cid=current.goal_cid,
                    ordinal=current.ordinal,
                    status=current.status,
                    priority=current.priority,
                    plan_cid=current.plan_cid,
                    objective_id=current.objective_id,
                    body={**dict(current.body), "completion": "manual"},
                    identity={"task_cid": current.task_cid},
                    expected_revision=current.revision,
                    dependencies=current.dependencies,
                    outputs=current.outputs,
                    acceptance=current.acceptance,
                    validations=current.validations,
                )
                mutated = True
            return original_get(task_cid)

        monkeypatch.setattr(source, "get", become_manual)
        assert daemon.claim_next() is None
        assert mutated is True
        assert daemon.list_running_attempts() == []
        projection = daemon.coordinator.coordination_registry_projection()
        assert projection["counts"]["active_task_claims"] == 0
        current = original_get("task:cid:001")
        assert current is not None
        assert current.status == "ready"
        assert current.body["completion"] == "manual"
    finally:
        daemon.close()


def test_portal_builder_with_inherited_database_program_without_implement_is_observer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replacement child must not infer execution authority from its DB env."""

    monkeypatch.chdir(tmp_path)
    program = DatabaseProgramConfig(
        authority_mode="embedded",
        task_source_kind="duckdb",
        store_id="control.duckdb",
        store_generation="generation-reload-regression",
        schema_revision="reload-regression-v1",
    )
    monkeypatch.setenv(
        DATABASE_PROGRAM_JSON_ENV,
        json.dumps(program.to_dict(), separators=(",", ":"), sort_keys=True),
    )
    args = parse_args(
        [
            *program.daemon_cli_args(),
            "--todo-path",
            str(tmp_path / "wrong-default-board.md"),
            "--state-dir",
            str(tmp_path / "wrong-default-state"),
            "--state-prefix",
            "wrong-default",
            "--once",
            # Deliberately no --implement: this is the failed reload shape.
        ]
    )
    bind_results: list[object | None] = []
    real_bind = daemon_runner.bind_database_portal_execution_from_args

    def record_bind(*bind_args: object, **bind_kwargs: object) -> object | None:
        result = real_bind(*bind_args, **bind_kwargs)
        bind_results.append(result)
        return result

    monkeypatch.setattr(
        daemon_runner,
        "bind_database_portal_execution_from_args",
        record_bind,
    )
    daemon, _context = daemon_runner.build_portal_implementation_daemon_from_args(
        args,
        repo_root=tmp_path,
    )
    try:
        assert isinstance(daemon, DatabaseImplementationDaemon)
        assert daemon.require_real_execution is False
        assert daemon.execution_callbacks_bound is False
        assert bind_results == [None]
        daemon.materialize_population(_population(1))
        before = daemon.task_source.get("task:cid:001")
        assert before is not None

        result = daemon.run_once()

        after = daemon.task_source.get("task:cid:001")
        assert after is not None
        assert result["execution_authorized"] is False
        assert result["write_count"] == 0
        assert result["selection_idle_reason"] == (
            "database_execution_not_authorized"
        )
        assert after.to_dict() == before.to_dict()
        assert after.status == "ready"
        assert daemon.list_running_attempts() == []
        counts = daemon._require_connection().execute(
            """
            SELECT
                (SELECT COUNT(*) FROM database_task_attempts),
                (SELECT COUNT(*) FROM provider_invocations),
                (SELECT COUNT(*) FROM effect_claims)
            """
        ).fetchone()
        assert tuple(int(counts[index]) for index in range(3)) == (0, 0, 0)
    finally:
        daemon.close()


def test_quack_runner_builders_require_bound_typed_owner_for_lane_sidecars(
    tmp_path: Path,
) -> None:
    control = tmp_path / "control.duckdb"

    def lane_args(index: int):
        return parse_args(
            [
                "--task-source-kind",
                "duckdb",
                "--authority-mode",
                "quack",
                "--database-path",
                str(control),
                "--endpoint-secret-handle",
                "handle:test-quack",
                "--quack-endpoint",
                "quack:127.0.0.1:45671",
                "--state-store-id",
                "state/control.duckdb",
                "--state-store-generation",
                "test-generation",
                "--state-schema-revision",
                "generic-typed-owner-v1",
                "--todo-path",
                str(tmp_path / "unused.md"),
                "--state-dir",
                str(tmp_path / f"lane-{index}"),
                "--state-prefix",
                f"pcpc_lane_{index}",
                "--task-shard-count",
                "4",
                "--task-shard-index",
                str(index),
                "--strict-task-sharding",
                "--once",
            ]
        )

    resolved = [
        resolve_database_implementation_paths(lane_args(index), authority_mode="quack")
        for index in range(3)
    ]
    assert [item["database_path"] for item in resolved] == [
        tmp_path / f"lane-{index}" / "quack-lane-control.duckdb"
        for index in range(3)
    ]
    assert [item["coordination_path"].parent for item in resolved] == [
        tmp_path / f"lane-{index}" for index in range(3)
    ]
    assert [item["execution_path"].parent for item in resolved] == [
        tmp_path / f"lane-{index}" for index in range(3)
    ]
    assert len({item["coordination_path"] for item in resolved}) == 3
    assert len({item["execution_path"] for item in resolved}) == 3

    # Path derivation alone is not Quack authority.  These convenience
    # builders cannot mint or infer a remote-owner credential from argv, so
    # they must fail closed until the launcher injects the exact attached
    # TypedDatabaseTaskSource and its process-bound bootstrap credentials.
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="exact attached TypedDatabaseTaskSource",
    ):
        build_portal_implementation_daemon_from_args(
            lane_args(0),
            repo_root=tmp_path,
        )
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="exact attached TypedDatabaseTaskSource",
    ):
        build_portal_implementation_daemon_from_args(
            lane_args(1),
            repo_root=tmp_path,
        )
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="exact attached TypedDatabaseTaskSource",
    ):
        build_database_implementation_daemon_from_args(
            lane_args(2),
            database_path=control,
        )


def test_database_lanes_register_disjoint_hash_shards(tmp_path: Path) -> None:
    source = DatabaseTaskSource(tmp_path / "control.duckdb")
    source.materialize(_population(8), repository_tree_id="tree:dqp-018")
    daemons = [
        DatabaseImplementationDaemon(
            database_path=tmp_path / "control.duckdb",
            coordination_path=tmp_path / f"lane-{index}" / "coordination.duckdb",
            execution_path=tmp_path / f"lane-{index}" / "execution.duckdb",
            owner_session_id=f"session:lane:{index}",
            authority_mode="embedded",
            task_source_kind="duckdb",
            task_source=source,
            task_shard_count=2,
            task_shard_index=index,
            strict_task_sharding=True,
        )
        for index in range(2)
    ]
    try:
        observed = [set(daemon.sync_ready_tasks_into_coordination()) for daemon in daemons]
        expected = [set(), set()]
        for index in range(1, 9):
            alias = f"DQP-T{index:03d}"
            task_cid = f"task:cid:{index:03d}"
            digest = hashlib.sha256(alias.encode("utf-8")).hexdigest()
            expected[int(digest[:8], 16) % 2].add(task_cid)
        assert observed == expected
        assert observed[0].isdisjoint(observed[1])
        assert observed[0] | observed[1] == {
            f"task:cid:{index:03d}" for index in range(1, 9)
        }
    finally:
        for daemon in daemons:
            daemon.close()
        source.close()


def test_already_in_progress_control_task_creates_no_execution_attempt(
    tmp_path: Path,
) -> None:
    source = DatabaseTaskSource(tmp_path / "control.duckdb")
    source.materialize(_population(1), repository_tree_id="tree:dqp-018")
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "lane" / "coordination.duckdb",
        execution_path=tmp_path / "lane" / "execution.duckdb",
        owner_session_id="session:stale-local-ready",
        authority_mode="embedded",
        task_source_kind="duckdb",
        task_source=source,
        require_real_execution=True,
    )
    try:
        daemon.open()
        task = source.get_task("task:cid:001")
        assert task is not None
        daemon.coordinator.register_task(
            task_cid=task.task_cid,
            task_id=task.task_alias,
        )
        source.compare_and_set_status(
            task.task_cid,
            task.revision,
            "in_progress",
            {"operation": "competing_lane_claim"},
        )
        assert daemon.claim_next() is None
        assert daemon.list_running_attempts() == []
        current = source.get_task(task.task_cid)
        assert current is not None
        assert current.status == "in_progress"
    finally:
        daemon.close()
        source.close()


def test_authoritative_cas_race_creates_no_execution_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = DatabaseTaskSource(tmp_path / "control.duckdb")
    source.materialize(_population(1), repository_tree_id="tree:dqp-018")
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "lane" / "coordination.duckdb",
        execution_path=tmp_path / "lane" / "execution.duckdb",
        owner_session_id="session:cas-race",
        authority_mode="embedded",
        task_source_kind="duckdb",
        task_source=source,
        require_real_execution=True,
    )
    try:
        daemon.open()

        def reject_cas(*_args, **_kwargs):
            raise DatabaseTaskSourceConflictError("competing control CAS won")

        monkeypatch.setattr(source, "compare_and_set_status", reject_cas)
        assert daemon.claim_next() is None
        assert daemon.list_running_attempts() == []
        task = source.get_task("task:cid:001")
        assert task is not None
        assert task.status == "ready"
    finally:
        daemon.close()
        source.close()


def test_portal_bridge_failure_requeues_exact_claim_for_later_reclaim(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}
    provider_attempts: list[DatabaseTaskAttempt] = []

    def fail_first_portal_pass(
        attempt: DatabaseTaskAttempt,
    ) -> dict[str, object]:
        provider_attempts.append(attempt)
        if len(provider_attempts) == 1:
            raise DatabasePortalBridgeDeferred(
                "launch_redteam_forced_retryable_portal_bridge_failure",
                backoff_seconds=1,
            )
        return {
            "status": "ok",
            "accepted": True,
            "task_cid": attempt.task_cid,
        }

    daemon = _open_daemon(
        tmp_path,
        session="session:portal-requeue",
        provider_fn=fail_first_portal_pass,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))

        first_pass = daemon.run_once()
        first_result = first_pass["implementation_result"]
        assert first_result["portal_retryable_failure"] is True
        assert first_result["status"] == "failed"
        assert first_result["deferred"] is True
        assert first_result["retry_state"]["changed"] is True
        assert first_result["retry_state"]["status"] == "retrying"
        assert first_result["retry_state"]["backoff_seconds"] == 1

        first_attempt_id = str(first_result["attempt_id"])
        first_attempt = daemon.get_attempt(first_attempt_id)
        assert first_attempt is not None
        assert first_attempt.status == "failed"
        assert first_attempt.committed_phase == "failed"
        failure_phases = [
            phase
            for phase in daemon.phase_history(first_attempt_id)
            if phase["phase"] == "failed"
        ]
        assert len(failure_phases) == 1
        failure_receipt = failure_phases[0]["body"]
        assert failure_receipt["portal_retryable_failure"] is True
        assert failure_receipt["portal_terminal_failure"] is False
        assert failure_receipt["deferred"] is True
        assert failure_receipt["attempt_consumed"] is False
        assert failure_receipt["provider_dispatched"] is False
        assert failure_receipt["typed_deferral_slot_consumed"] is True
        assert failure_receipt["backoff_seconds"] == 1
        assert failure_receipt["typed_deferral"]["attempt_id"] == first_attempt_id

        initial_claim = daemon.coordinator.get_task_claim(first_attempt.claim_id)
        initial_coordination_attempt = daemon.coordinator.get_task_attempt(
            first_attempt_id
        )
        assert initial_claim is not None
        assert initial_coordination_attempt is not None
        assert initial_claim.state.value == "accepted"
        assert initial_coordination_attempt.status.value == "running"
        control_after_failure = daemon.task_source.get(first_attempt.task_cid)
        assert control_after_failure is not None
        assert control_after_failure.status == "retrying"
        assert daemon.list_running_attempts() == []

        now["ms"] = 6_001
        second_pass = daemon.run_once()
        second_result = second_pass["implementation_result"]
        assert second_result["status"] == "succeeded"
        second_attempt = second_result["attempt"]
        assert second_attempt["attempt_id"] != first_attempt_id
        assert second_attempt["attempt_number"] == 2
        assert second_attempt["fencing_token"] > first_attempt.fencing_token
        assert len(provider_attempts) == 2
        expired_claim = daemon.coordinator.get_task_claim(first_attempt.claim_id)
        assert expired_claim is not None
        assert expired_claim.state.value == "expired"
        final_control = daemon.task_source.get(first_attempt.task_cid)
        assert final_control is not None
        assert final_control.status == "completed"
    finally:
        daemon.close()


def test_database_daemon_rejects_idle_lane_work_stealing(tmp_path: Path) -> None:
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="does not support idle-lane work stealing",
    ):
        DatabaseImplementationDaemon(
            database_path=tmp_path / "control.duckdb",
            coordination_path=tmp_path / "coordination.duckdb",
            execution_path=tmp_path / "execution.duckdb",
            authority_mode="embedded",
            task_source_kind="duckdb",
            task_shard_count=2,
            task_shard_index=0,
            strict_task_sharding=True,
            idle_lane_work_stealing="virgin-transfer",
        )


def test_idle_run_once_idles_on_quack_attach_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:quack-attach-idle",
    )

    def boom() -> None:
        raise DuckDBConnectionPolicyError(
            "quack attach authentication failed uri='quack:127.0.0.1:41327' "
            "token_present=True token_sha16=deadbeefdeadbeef"
        )

    try:
        monkeypatch.setattr(daemon, "reconcile_prepared_task_completions", boom)
        monkeypatch.setattr(
            daemon,
            "claim_next",
            lambda: pytest.fail("attach failure claimed work"),
        )
        result = daemon.run_once()
        assert result["selection_idle_reason"] == "quack_attach_failed"
        assert result["implementation_result"] is None
        assert result["active_task_id"] == ""
        assert result["control_plane_error"]["error_type"] == (
            "DuckDBConnectionPolicyError"
        )
    finally:
        daemon.close()


def test_idle_run_once_idles_on_quack_attach_lock_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:quack-attach-lock-timeout-idle",
    )

    def boom() -> None:
        raise TimeoutError(
            "timed out acquiring DuckDB process lock: "
            f"{tmp_path / 'quack-owner' / 'attach.lock'}"
        )

    try:
        monkeypatch.setattr(daemon, "reconcile_prepared_task_completions", boom)
        monkeypatch.setattr(
            daemon,
            "claim_next",
            lambda: pytest.fail("attach lock timeout claimed work"),
        )
        result = daemon.run_once()
        assert result["selection_idle_reason"] == "quack_attach_failed"
        assert result["implementation_result"] is None
        assert result["active_task_id"] == ""
        assert result["control_plane_error"]["error_type"] == "TimeoutError"
    finally:
        daemon.close()


def test_idle_run_once_idles_on_quack_authorization_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:quack-authorization-idle",
    )

    def boom() -> None:
        raise RuntimeError("Invalid Input Error: Authorization failed")

    try:
        monkeypatch.setattr(daemon, "reconcile_prepared_task_completions", boom)
        monkeypatch.setattr(
            daemon,
            "claim_next",
            lambda: pytest.fail("authorization failure claimed work"),
        )
        result = daemon.run_once()
        assert result["selection_idle_reason"] == "quack_attach_failed"
        assert result["implementation_result"] is None
        assert result["control_plane_error"]["error_type"] == "RuntimeError"
    finally:
        daemon.close()


def test_idle_run_once_invokes_bound_post_merge_recovery_before_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:post-merge-recovery-idle",
    )
    calls: list[str] = []
    reconciliation = {
        "schema": DATABASE_POST_MERGE_RECOVERY_SCHEMA,
        "attempted": True,
        "recovered": True,
        "changed": True,
        "reason": "post_merge_declared_outputs_repaired",
        "write_count": 2,
        "results": [
            {
                "task_cid": "task:cid:blocked",
                "changed": True,
                "previous_status": "blocked",
                "status": "retrying",
            }
        ],
    }

    def recover() -> dict[str, object]:
        calls.append("recover")
        return reconciliation

    def no_ready_task() -> None:
        calls.append("claim")
        return None

    try:
        daemon.bind_post_merge_recovery(recover)
        monkeypatch.setattr(daemon, "claim_next", no_ready_task)

        result = daemon.run_once()

        assert calls == ["recover", "claim"]
        assert result["selection_idle_reason"] == "no_ready_tasks"
        assert result["implementation_result"] is None
        assert result["unchanged"] is False
        assert result["write_count"] == 2
        reported = result["post_merge_recovery"]
        assert reported == reconciliation
        assert reported["results"][0]["status"] == "retrying"
    finally:
        daemon.close()


def test_idle_run_once_rearms_blocked_task_when_outputs_are_on_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    output = repo / "ipfs_accelerate_py" / "agent_supervisor" / "residual_intelligence"
    output.mkdir(parents=True)
    (output / "expert_specs.py").write_text("spec = True\n", encoding="utf-8")

    def git(*args: str) -> None:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    git("init", "-b", "main")
    git("config", "user.email", "vrif@example.test")
    git("config", "user.name", "VRIF")
    git("add", ".")
    git("commit", "-m", "declared outputs")

    daemon = _open_daemon(
        tmp_path,
        session="session:declared-output-rearm",
    )
    repair_snapshot = SimpleNamespace(
        task_id="VRIF-010",
        canonical_task_id="task:cid:blocked-010",
        metadata={
            "completion": {
                "reason": "post_merge_declared_outputs_repaired",
                "candidate_commit": "c9791a30e",
                "repair_receipt": {
                    "entries": [
                        {
                            "path": (
                                "ipfs_accelerate_py/agent_supervisor/"
                                "residual_intelligence/expert_specs.py"
                            )
                        }
                    ]
                },
            }
        },
    )
    cas_calls: list[tuple[str, str]] = []
    list_calls: list[dict[str, object]] = []

    def rearm(
        task_cid: str,
        *,
        receipt: dict[str, object] | None = None,
    ) -> SimpleNamespace:
        cas_calls.append((task_cid, "retrying"))
        assert receipt is not None
        assert receipt["schema"] == DATABASE_DECLARED_OUTPUT_REARM_SCHEMA
        assert receipt["operation"] == "database_declared_outputs_on_head_rearm"
        return SimpleNamespace(
            changed=True,
            task=SimpleNamespace(task_cid=task_cid),
        )

    def list_tasks(**kwargs: object) -> SimpleNamespace:
        list_calls.append(dict(kwargs))
        raise DuckDBConnectionPolicyError(
            "quack attach authentication failed uri='quack:127.0.0.1:41327' "
            "token_present=True token_sha16=deadbeefdeadbeef"
        )

    try:
        daemon.open()
        daemon._merge_repo_root = repo
        daemon._merge_queue = SimpleNamespace(
            completed_requests=lambda **_kwargs: (repair_snapshot,),
        )
        monkeypatch.setattr(
            daemon,
            "reconcile_prepared_task_completions",
            lambda: [],
        )
        monkeypatch.setattr(
            daemon,
            "reconcile_expired_running_attempts",
            lambda: [],
        )
        monkeypatch.setattr(
            daemon,
            "reconcile_terminal_portal_failures",
            lambda: [],
        )
        monkeypatch.setattr(
            daemon,
            "reconcile_terminal_retry_states",
            lambda: [],
        )
        monkeypatch.setattr(daemon, "list_running_attempts", lambda: [])
        monkeypatch.setattr(daemon, "claim_next", lambda: None)
        monkeypatch.setattr(daemon, "reconcile_landed_merged_tasks", lambda: [])
        monkeypatch.setattr(
            daemon,
            "reconcile_unimplemented_unknown_callback_quarantines",
            lambda: [],
        )
        monkeypatch.setattr(daemon, "reconcile_stale_in_progress_gates", lambda: [])
        monkeypatch.setattr(
            daemon,
            "reconcile_blocked_protected_path_recoveries",
            lambda: [],
        )
        monkeypatch.setattr(
            daemon,
            "reconcile_blocked_external_protected_checkout_recoveries",
            lambda: [],
        )
        monkeypatch.setattr(
            daemon,
            "reconcile_blocked_inflight_process_recoveries",
            lambda: [],
        )
        monkeypatch.setattr(
            daemon,
            "reconcile_blocked_validation_retry_seed_conflict_recoveries",
            lambda: [],
        )
        monkeypatch.setattr(
            daemon,
            "reconcile_blocked_leftover_wait_deferral_budget_recoveries",
            lambda: [],
        )
        monkeypatch.setattr(
            daemon,
            "reconcile_blocked_pooled_worktree_create_recoveries",
            lambda: [],
        )
        monkeypatch.setattr(daemon, "reconcile_inflight_deferral_blocks", lambda: [])
        monkeypatch.setattr(daemon.task_source, "list_tasks", list_tasks)
        monkeypatch.setattr(
            daemon.task_source,
            "rearm_blocked_task",
            rearm,
        )
        result = daemon.run_once()
        assert list_calls == []
        assert cas_calls == [("task:cid:blocked-010", "retrying")]
        assert result["selection_idle_reason"] == "no_ready_tasks"
        assert result["write_count"] == 1
        rearm_result = result["declared_output_rearm"]
        assert rearm_result["schema"] == DATABASE_DECLARED_OUTPUT_REARM_SCHEMA
        assert rearm_result["rearmed"] == 1
        assert rearm_result["results"][0]["task_cid"] == "task:cid:blocked-010"
        assert rearm_result["results"][0]["status"] == "retrying"
    finally:
        daemon.close()


def test_idle_run_once_rearms_from_older_repair_receipt_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    output = repo / "ipfs_accelerate_py" / "agent_supervisor" / "residual_intelligence"
    output.mkdir(parents=True)
    (output / "expert_specs.py").write_text("spec = True\n", encoding="utf-8")
    subprocess.run(
        ["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.email", "vrif@example.test"], cwd=repo, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "VRIF"], cwd=repo, check=True
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "declared outputs"],
        cwd=repo,
        check=True,
        capture_output=True,
    )

    completed = tmp_path / "completed"
    completed.mkdir()
    later = {
        "task_id": "VRIF-010",
        "canonical_task_id": "task:cid:blocked-010",
        "metadata": {"completion": {"reason": "merged"}},
    }
    repair = {
        "task_id": "VRIF-010",
        "canonical_task_id": "task:cid:blocked-010",
        "metadata": {
            "completion": {
                "reason": "post_merge_declared_outputs_repaired",
                "candidate_commit": "c9791a30e",
                "repair_receipt": {
                    "entries": [
                        {
                            "path": (
                                "ipfs_accelerate_py/agent_supervisor/"
                                "residual_intelligence/expert_specs.py"
                            )
                        }
                    ]
                },
            }
        },
    }
    (completed / "later.json").write_text(json.dumps(later), encoding="utf-8")
    (completed / "repair.json").write_text(json.dumps(repair), encoding="utf-8")
    later_path = completed / "later.json"
    repair_path = completed / "repair.json"
    os.utime(repair_path, (1_000_000, 1_000_000))
    os.utime(later_path, (2_000_000, 2_000_000))

    daemon = _open_daemon(tmp_path, session="session:older-repair-json")
    cas_calls: list[str] = []

    def rearm(
        task_cid: str, *, receipt: dict[str, object] | None = None
    ) -> SimpleNamespace:
        cas_calls.append(task_cid)
        return SimpleNamespace(changed=True, task=SimpleNamespace(task_cid=task_cid))

    try:
        daemon.open()
        daemon._merge_repo_root = repo
        daemon._merge_queue = SimpleNamespace(
            completed_dir=completed,
            completed_requests=lambda **_kwargs: (),
        )
        monkeypatch.setattr(daemon, "reconcile_prepared_task_completions", lambda: [])
        monkeypatch.setattr(daemon, "reconcile_expired_running_attempts", lambda: [])
        monkeypatch.setattr(daemon, "reconcile_terminal_portal_failures", lambda: [])
        monkeypatch.setattr(daemon, "reconcile_terminal_retry_states", lambda: [])
        monkeypatch.setattr(daemon, "list_running_attempts", lambda: [])
        monkeypatch.setattr(daemon, "claim_next", lambda: None)
        monkeypatch.setattr(daemon.task_source, "rearm_blocked_task", rearm)
        result = daemon.run_once()
        assert cas_calls == ["task:cid:blocked-010"]
        assert result["declared_output_rearm"]["rearmed"] == 1
    finally:
        daemon.close()


def test_idle_run_once_rearms_before_attach_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:rearm-before-attach",
    )
    rearm = {
        "schema": DATABASE_DECLARED_OUTPUT_REARM_SCHEMA,
        "attempted": True,
        "rearmed": 1,
        "results": [
            {
                "task_cid": "task:cid:blocked-010",
                "changed": True,
                "previous_status": "blocked",
                "status": "retrying",
            }
        ],
        "write_count": 1,
    }

    def boom() -> None:
        raise DuckDBConnectionPolicyError(
            "quack attach authentication failed uri='quack:127.0.0.1:41327' "
            "token_present=True token_sha16=deadbeefdeadbeef"
        )

    try:
        monkeypatch.setattr(
            daemon,
            "_rearm_blocked_tasks_with_outputs_on_head",
            lambda: rearm,
        )
        monkeypatch.setattr(daemon, "reconcile_prepared_task_completions", boom)
        monkeypatch.setattr(
            daemon,
            "claim_next",
            lambda: pytest.fail("attach failure claimed work"),
        )
        result = daemon.run_once()
        assert result["selection_idle_reason"] == "quack_attach_failed"
        assert result["write_count"] == 1
        assert result["declared_output_rearm"]["rearmed"] == 1
        assert result["unchanged"] is False
    finally:
        daemon.close()


def test_idle_run_once_settles_invalid_metadata_portal_quarantine_before_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
        checkout_repository_id,
    )
    from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import MergeQueue

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    git("init", "-b", "main")
    git("config", "user.name", "Merge Train Test")
    git("config", "user.email", "merge-train@example.invalid")
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    git("add", "base.txt")
    git("commit", "-m", "base")
    git("switch", "-c", "implementation/side")
    (repo / "side.txt").write_text("side\n", encoding="utf-8")
    git("add", "side.txt")
    git("commit", "-m", "side")
    candidate = git("rev-parse", "HEAD")
    git("switch", "main")
    head_before = git("rev-parse", "HEAD")
    queue = MergeQueue(
        tmp_path / "queue",
        target_repository_id=checkout_repository_id(repo),
        target_branch="main",
        require_target_binding=True,
    )
    request = queue.enqueue(
        branch_name="implementation/side",
        task_id="REF-040",
        canonical_task_id="task:cid:ref-040",
        commit_sha=candidate,
        metadata={
            "schema": "ipfs_accelerate_py/agent-supervisor/merge-candidate@3",
            "todo_path": str(tmp_path / "attempts" / "x" / "task-projection.md"),
            "completion_task_cids": {"REF-040": "task:cid:ref-040"},
            "manual_completion_authority_task_ids": [],
            "manual_completion_authority_required_task_ids": [],
            "manual_completion_authority_epoch_id": "",
            "manual_completion_authority_revocation_generation": 0,
            "manual_completion_authority_context_id": "baguqeera-invalid",
            "task": {"task_id": "REF-040", "outputs": ["base.txt"]},
            "changed_submodule_paths": [],
        },
    )
    claimed = queue.dequeue(consumer_id="merge-train:test")
    assert claimed is not None
    queue.quarantine(
        claimed,
        reason="cross_board_manual_completion_authority_metadata_invalid",
    )

    daemon = _open_daemon(
        tmp_path,
        session="session:invalid-metadata-merge-settle",
    )
    calls: list[str] = []

    def no_ready_task() -> None:
        calls.append("claim")
        return None

    try:
        daemon.bind_merge_train_recovery(
            merge_queue=queue,
            repo_root=repo,
            merge_target_branch="main",
        )
        monkeypatch.setattr(daemon, "claim_next", no_ready_task)
        result = daemon.run_once()
        assert calls == ["claim"]
        settlement = result["merge_quarantine_settlement"]
        assert settlement["attempted"] is True
        assert settlement["settled"] == 1
        assert settlement["results"][0]["status"] == "already_merged"
        assert git("rev-parse", "HEAD") == head_before
        completed = queue.get(request.request_id)
        assert completed is not None
        assert completed.status == "completed"
        side_probe = subprocess.run(
            ["git", "cat-file", "-e", "HEAD:side.txt"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )
        assert side_probe.returncode != 0
    finally:
        daemon.close()


def test_idle_run_once_reports_failed_recovery_as_potentially_changed(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:post-merge-partial-write",
    )
    try:
        daemon.materialize_population(_population(1))

        def recover_then_lose_response() -> None:
            daemon.task_source.record_queue_backoff(
                task_cid="task:cid:001",
                delay_ms=60_000,
                reason="post_merge_recovery_partial_write_fixture",
            )
            raise RuntimeError("response lost after durable queue write")

        daemon.bind_post_merge_recovery(recover_then_lose_response)
        result = daemon.run_once()

        assert result["selection_idle_reason"] == "no_ready_tasks"
        assert result["unchanged"] is False
        assert result["write_count"] == 1
        reconciliation = result["post_merge_recovery"]
        assert reconciliation["reason"] == (
            "post_merge_recovery_callback_failed"
        )
        assert reconciliation["durable_state_uncertain"] is True
        assert reconciliation["write_count"] == 1
        assert daemon.task_source.get_queue_entry("task:cid:001") is not None
    finally:
        daemon.close()


@pytest.mark.parametrize(
    "invalid_result",
    [
        {
            "schema": "foreign/recovery@1",
            "attempted": True,
            "recovered": False,
            "reason": "no_match",
            "write_count": 0,
        },
        {
            "schema": DATABASE_POST_MERGE_RECOVERY_SCHEMA,
            "attempted": False,
            "recovered": "yes",
            "write_count": 0,
        },
    ],
)
def test_idle_run_once_rejects_untyped_post_merge_recovery_result(
    tmp_path: Path,
    invalid_result: dict[str, object],
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:post-merge-invalid-result",
    )
    try:
        daemon.bind_post_merge_recovery(lambda: invalid_result)
        result = daemon.run_once()

        assert result["unchanged"] is False
        assert result["write_count"] == 1
        reconciliation = result["post_merge_recovery"]
        assert reconciliation["schema"] == DATABASE_POST_MERGE_RECOVERY_SCHEMA
        assert reconciliation["attempted"] is True
        assert reconciliation["recovered"] is False
        assert reconciliation["reason"] == "post_merge_recovery_result_invalid"
        assert reconciliation["durable_state_uncertain"] is True
    finally:
        daemon.close()


def test_observer_run_once_never_invokes_bound_post_merge_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:post-merge-recovery-observer",
    )
    recovery_calls: list[str] = []

    def forbidden_recovery() -> dict[str, object]:
        recovery_calls.append("called")
        pytest.fail("observer invoked mutating post-merge recovery")

    try:
        daemon.bind_post_merge_recovery(forbidden_recovery)
        daemon.require_real_execution = False
        monkeypatch.setattr(
            daemon,
            "claim_next",
            lambda: pytest.fail("observer attempted to claim work"),
        )

        result = daemon.run_once()

        assert recovery_calls == []
        assert result["execution_authorized"] is False
        assert result["unchanged"] is True
        assert result["write_count"] == 0
        assert result["selection_idle_reason"] == (
            "database_execution_not_authorized"
        )
        reconciliation = result.get("post_merge_recovery")
        assert reconciliation is None or (
            reconciliation.get("attempted") is False
            and reconciliation.get("write_count") == 0
        )
    finally:
        daemon.close()


def test_terminal_portal_failure_reason_ignores_later_non_portal_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:post-merge-shutdown-phase",
    )
    try:
        daemon.materialize_population(_population(1))
        failed = daemon.claim_next()
        assert failed is not None
        failed = daemon.commit_phase(failed, "context")
        failed = daemon.commit_phase(
            failed,
            "failed",
            body={
                "reason": "post_merge_declared_outputs_missing",
                "portal_retryable_failure": False,
                "portal_terminal_failure": True,
            },
        )
        original_history = daemon.phase_history(failed.attempt_id)

        def history_with_shutdown(_attempt_id: str) -> list[dict[str, object]]:
            return [
                *original_history,
                {
                    "phase": ATTEMPT_PHASE_FAILED,
                    "body": {"reason": "supervisor_signal_shutdown"},
                },
            ]

        monkeypatch.setattr(daemon, "phase_history", history_with_shutdown)
        assert (
            daemon._terminal_portal_failure_reason(failed)
            == "post_merge_declared_outputs_missing"
        )
    finally:
        daemon.close()


def _post_merge_preauthorization(
    daemon: DatabaseImplementationDaemon,
    failed: DatabaseTaskAttempt,
    *,
    request_id: str = "merge-request:vrif-010",
    candidate_commit: str | None = None,
) -> dict[str, object]:
    return {
        "schema": DATABASE_POST_MERGE_RECOVERY_PREAUTHORIZATION_SCHEMA,
        "request_id": request_id,
        "task_cid": failed.task_cid,
        "task_alias": failed.task_alias,
        "candidate_commit": candidate_commit or "a" * 40,
        "source_attempt_id": failed.attempt_id,
        "source_claim_id": failed.claim_id,
        "source_lease_id": failed.lease_id,
        "source_fencing_token": failed.fencing_token,
        "source_fence_epoch": failed.fence_epoch,
        "source_binding_id": "sha256:" + "c" * 64,
        "source_projection_immutable_digest": "sha256:" + "d" * 64,
    }


def _post_merge_repair_recovery_evidence(
    daemon: DatabaseImplementationDaemon,
    failed: DatabaseTaskAttempt,
    *,
    request_id: str = "merge-request:portable-ordinary",
    candidate_commit: str = "a" * 40,
    repair_commit: str = "b" * 40,
) -> dict[str, object]:
    repair_receipt: dict[str, object] = {
        "schema": POST_MERGE_DECLARED_OUTPUT_REPAIR_SCHEMA,
        "task_ids": [failed.task_alias],
        "candidate_commit": candidate_commit,
        "candidate_tree": "1" * 40,
        "baseline_commit": "2" * 40,
        "failed_integration_commit": "3" * 40,
        "repair_parent_commit": "4" * 40,
        "repair_commit": repair_commit,
        "repair_tree": "5" * 40,
        "entries": [
            {
                "path": "output.py",
                "mode": "100644",
                "object_type": "blob",
                "object_id": "6" * 40,
            }
        ],
        "validation": [
            {
                "task_id": failed.task_alias,
                "passed": True,
                "returncode": 0,
                "validation_result_digests": ["sha256:" + "7" * 64],
                "command_count": 1,
                "log_sha256": "8" * 64,
            }
        ],
        "rollback_target": "4" * 40,
    }
    repair_receipt["receipt_id"] = content_identity(repair_receipt)
    evidence: dict[str, object] = {
        "schema": DATABASE_POST_MERGE_RECOVERY_SCHEMA,
        "request_id": request_id,
        "task_cid": failed.task_cid,
        "task_alias": failed.task_alias,
        "candidate_commit": candidate_commit,
        "repair_commit": repair_commit,
        "repair_receipt_id": repair_receipt["receipt_id"],
        "repair_receipt": repair_receipt,
        "source_attempt_id": failed.attempt_id,
        "source_claim_id": failed.claim_id,
        "source_lease_id": failed.lease_id,
        "source_fencing_token": failed.fencing_token,
        "source_fence_epoch": failed.fence_epoch,
        "source_binding_id": "sha256:" + "c" * 64,
        "source_projection_immutable_digest": "sha256:" + "d" * 64,
    }
    evidence["evidence_id"] = daemon._database_portal_evidence_digest(
        evidence
    )
    return evidence


def test_preauthorize_accepts_wrapped_post_merge_terminal_reason(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:post-merge-wrapped-reason",
    )
    try:
        daemon.materialize_population(_population(1))
        failed = daemon.claim_next()
        assert failed is not None
        failed = daemon.commit_phase(failed, "context")
        failed = daemon.commit_phase(
            failed,
            "failed",
            body={
                "reason": (
                    "DatabasePortalBridgeError: "
                    "post_merge_declared_outputs_missing"
                ),
                "portal_retryable_failure": False,
                "portal_terminal_failure": True,
            },
        )
        assert (
            daemon._terminal_portal_failure_reason(failed)
            == (
                "DatabasePortalBridgeError: "
                "post_merge_declared_outputs_missing"
            )
            or daemon._canonical_portal_failure_reason(
                daemon._terminal_portal_failure_reason(failed)
            )
            == "post_merge_declared_outputs_missing"
        )
        terminal = daemon._persist_terminal_portal_failure(
            failed,
            reason="post_merge_declared_outputs_missing",
            coordination_evidence=(
                daemon._reconcile_failed_attempt_coordination(failed)
            ),
        )
        assert terminal["status"] == "blocked"
        authorized = daemon.preauthorize_post_merge_declared_output_recovery(
            _post_merge_preauthorization(daemon, failed)
        )
        assert authorized["authorized"] is True
        assert authorized["task_status"] == "blocked"
    finally:
        daemon.close()


def test_post_merge_recovery_accepts_complete_execution_route_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:post-merge-execution-route-lineage",
    )
    try:
        daemon.materialize_population(_population(1))
        failed = daemon.claim_next()
        assert failed is not None
        failed = daemon.commit_phase(failed, "context")
        failed = daemon.commit_phase(
            failed,
            "failed",
            body={
                "reason": (
                    DATABASE_PORTAL_COMPLETION_IMPLEMENTATION_COMMIT_MISSING_REASON
                ),
                "portal_retryable_failure": False,
                "portal_terminal_failure": True,
            },
        )
        terminal = daemon._persist_terminal_portal_failure(
            failed,
            reason=(
                DATABASE_PORTAL_COMPLETION_IMPLEMENTATION_COMMIT_MISSING_REASON
            ),
            coordination_evidence=(
                daemon._reconcile_failed_attempt_coordination(failed)
            ),
        )
        assert terminal["status"] == "blocked"
        blocked = daemon.task_source.get(failed.task_cid)
        assert blocked is not None
        route = {
            "policy_id": "policy:sealed-route",
            "task_revision": 1,
        }
        receipt = {
            **blocked.body["completion_receipt"],
            "execution_route_binding": route,
            "execution_route_policy_id": route["policy_id"],
            "execution_route_origin_revision": route["task_revision"],
        }
        routed = replace(
            blocked,
            body={**blocked.body, "completion_receipt": receipt},
        )
        original_get = daemon.task_source.get

        def get_routed(task_cid: str) -> object:
            return routed if task_cid == failed.task_cid else original_get(task_cid)

        def validate_route(
            value: object,
            *,
            task: object,
            allow_claim_revision: bool,
        ) -> Mapping[str, object]:
            assert task is routed
            assert allow_claim_revision is True
            if not isinstance(value, Mapping) or dict(value) != route:
                raise ValueError("foreign execution route")
            return route

        monkeypatch.setattr(daemon.task_source, "get", get_routed)
        monkeypatch.setattr(
            daemon.task_source,
            "execution_route_binding_for_task",
            lambda task: route if task is routed else {},
            raising=False,
        )
        monkeypatch.setattr(
            daemon.task_source,
            "validate_execution_route_binding",
            validate_route,
            raising=False,
        )

        assert daemon.post_merge_completion_recovery_task_cids() == (
            failed.task_cid,
        )
        authorized = daemon.preauthorize_post_merge_declared_output_recovery(
            _post_merge_preauthorization(daemon, failed)
        )
        assert authorized["authorized"] is True

        partial_receipt = dict(receipt)
        partial_receipt.pop("execution_route_policy_id")
        partial = replace(
            routed,
            body={**routed.body, "completion_receipt": partial_receipt},
        )
        monkeypatch.setattr(
            daemon.task_source,
            "get",
            lambda task_cid: (
                partial if task_cid == failed.task_cid else original_get(task_cid)
            ),
        )
        assert daemon.post_merge_completion_recovery_task_cids() == ()
        with pytest.raises(
            DatabaseImplementationConflictError,
            match="no exact terminal failure control projection",
        ):
            daemon.preauthorize_post_merge_declared_output_recovery(
                _post_merge_preauthorization(daemon, failed)
            )

        stripped_receipt = {
            key: value
            for key, value in receipt.items()
            if key
            not in {
                "execution_route_binding",
                "execution_route_policy_id",
                "execution_route_origin_revision",
            }
        }
        stripped = replace(
            routed,
            body={**routed.body, "completion_receipt": stripped_receipt},
        )
        monkeypatch.setattr(
            daemon.task_source,
            "get",
            lambda task_cid: (
                stripped if task_cid == failed.task_cid else original_get(task_cid)
            ),
        )
        assert daemon.post_merge_completion_recovery_task_cids() == ()
        with pytest.raises(
            DatabaseImplementationConflictError,
            match="no exact terminal failure control projection",
        ):
            daemon.preauthorize_post_merge_declared_output_recovery(
                _post_merge_preauthorization(daemon, failed)
            )
    finally:
        daemon.close()


def test_base_only_execution_route_lineage_requires_a_legacy_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:base-only-execution-route-lineage",
    )
    try:
        receipt = {"operation": "legacy-terminal"}
        task = SimpleNamespace()
        assert daemon._receipt_has_exact_optional_execution_route_lineage(
            receipt,
            base_fields={"operation"},
            task=task,
        )

        monkeypatch.setattr(
            daemon.task_source,
            "execution_route_binding_for_task",
            lambda _task: {
                "policy_id": "policy:typed-route",
                "task_revision": 1,
            },
            raising=False,
        )
        assert not daemon._receipt_has_exact_optional_execution_route_lineage(
            receipt,
            base_fields={"operation"},
            task=task,
        )
    finally:
        daemon.close()


def test_base_only_execution_route_lineage_accepts_unsealed_typed_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:unsealed-typed-execution-route-lineage",
    )
    try:
        task = SimpleNamespace()
        monkeypatch.setattr(
            daemon.task_source,
            "execution_route_policy",
            None,
            raising=False,
        )
        monkeypatch.setattr(
            daemon.task_source,
            "execution_route_binding_for_task",
            lambda _task: pytest.fail(
                "unsealed typed source binding must not become route authority"
            ),
            raising=False,
        )
        monkeypatch.setattr(
            daemon.task_source,
            "validate_execution_route_binding",
            lambda *_args, **_kwargs: pytest.fail(
                "unsealed typed source validator must not become route authority"
            ),
            raising=False,
        )

        assert daemon._receipt_has_exact_optional_execution_route_lineage(
            {"operation": "legacy-terminal"},
            base_fields={"operation"},
            task=task,
        )
    finally:
        daemon.close()


def test_execution_route_lineage_sealed_policy_failure_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:sealed-execution-route-lineage-failure",
    )
    try:
        task = SimpleNamespace()
        route = {
            "policy_id": "policy:sealed-route",
            "task_revision": 1,
        }
        monkeypatch.setattr(
            daemon.task_source,
            "execution_route_policy",
            SimpleNamespace(policy_id=route["policy_id"]),
            raising=False,
        )
        monkeypatch.setattr(
            daemon.task_source,
            "execution_route_binding_for_task",
            lambda _task: (_ for _ in ()).throw(
                RuntimeError("sealed route binding is unavailable")
            ),
            raising=False,
        )
        monkeypatch.setattr(
            daemon.task_source,
            "validate_execution_route_binding",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("sealed route validation is unavailable")
            ),
            raising=False,
        )

        assert not daemon._receipt_has_exact_optional_execution_route_lineage(
            {"operation": "typed-terminal"},
            base_fields={"operation"},
            task=task,
        )
        assert not daemon._receipt_has_exact_optional_execution_route_lineage(
            {
                "operation": "typed-terminal",
                "execution_route_binding": route,
                "execution_route_policy_id": route["policy_id"],
                "execution_route_origin_revision": route["task_revision"],
            },
            base_fields={"operation"},
            task=task,
        )
    finally:
        daemon.close()


@pytest.mark.parametrize(
    "normalized",
    [
        pytest.param({}, id="empty-binding"),
        pytest.param({"mode": "typed"}, id="missing-policy-and-revision"),
        pytest.param(
            {"policy_id": "", "task_revision": 1},
            id="empty-policy-id",
        ),
        pytest.param(
            {"policy_id": 7, "task_revision": 1},
            id="non-string-policy-id",
        ),
        pytest.param(
            {"policy_id": "policy:typed-route", "task_revision": True},
            id="boolean-task-revision",
        ),
        pytest.param(
            {"policy_id": "policy:typed-route", "task_revision": "1"},
            id="non-integer-task-revision",
        ),
    ],
)
def test_execution_route_lineage_rejects_empty_or_untyped_normalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    normalized: Mapping[str, object],
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:malformed-execution-route-lineage",
    )
    try:
        binding = dict(normalized)
        task = SimpleNamespace()
        monkeypatch.setattr(
            daemon.task_source,
            "execution_route_binding_for_task",
            lambda _task: binding,
            raising=False,
        )
        monkeypatch.setattr(
            daemon.task_source,
            "validate_execution_route_binding",
            lambda _value, *, task, allow_claim_revision: normalized,
            raising=False,
        )
        receipt = {
            "operation": "typed-terminal",
            "execution_route_binding": binding,
            "execution_route_policy_id": normalized.get("policy_id"),
            "execution_route_origin_revision": normalized.get("task_revision"),
        }

        assert not daemon._receipt_has_exact_optional_execution_route_lineage(
            receipt,
            base_fields={"operation"},
            task=task,
        )
    finally:
        daemon.close()


def test_callback_recovery_projection_admits_exact_sibling_lane(
    tmp_path: Path,
) -> None:
    shared_state = tmp_path / "state"
    configured_root = (
        shared_state
        / "lane-1"
        / "pcsm_lane_1_database_portal_attempts"
    )
    source_root = (
        shared_state
        / "lane-2"
        / "pcsm_lane_2_database_portal_attempts"
    )
    configured_root.mkdir(parents=True)
    attempt_root = source_root / ("a" * 24)
    attempt_root.mkdir(parents=True)
    projection = attempt_root / "task-projection.md"
    projection.write_text("# projected task\n", encoding="utf-8")

    verify_root = (
        DatabaseImplementationDaemon
        ._callback_recovery_projection_verification_root
    )
    verified = verify_root(
        configured_attempt_root=configured_root,
        projection_path=projection,
    )

    assert verified == source_root.resolve(strict=True)


@pytest.mark.parametrize(
    "source_parts",
    [
        ("foreign-state", "lane-2", "pcsm_lane_2_database_portal_attempts"),
        ("state", "lane-2", "other_lane_2_database_portal_attempts"),
        ("state", "lane-2", "pcsm_lane_3_database_portal_attempts"),
    ],
)
def test_callback_recovery_projection_rejects_foreign_sibling_lane(
    tmp_path: Path,
    source_parts: tuple[str, ...],
) -> None:
    configured_root = (
        tmp_path
        / "state"
        / "lane-1"
        / "pcsm_lane_1_database_portal_attempts"
    )
    configured_root.mkdir(parents=True)
    source_root = tmp_path.joinpath(*source_parts)
    attempt_root = source_root / ("b" * 24)
    attempt_root.mkdir(parents=True)
    projection = attempt_root / "task-projection.md"
    projection.write_text("# projected task\n", encoding="utf-8")

    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="callback recovery source projection",
    ):
        DatabaseImplementationDaemon._callback_recovery_projection_verification_root(
            configured_attempt_root=configured_root,
            projection_path=projection,
        )


def test_callback_integration_authority_reloads_exact_sibling_lane_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test.api import test_agent_supervisor_database_portal_bridge as bridge_tests

    source_root = (
        tmp_path
        / "state"
        / "lane-2"
        / "vrif_lane_2_database_portal_attempts"
    )
    original_bridge = bridge_tests.DatabasePortalExecutionBridge

    class SiblingProjectionBridge(original_bridge):
        def __init__(self, *args: object, **kwargs: object) -> None:
            kwargs["attempt_root"] = source_root
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(
        bridge_tests,
        "DatabasePortalExecutionBridge",
        SiblingProjectionBridge,
    )
    daemon, qualification, evidence, _train_path, _repo = (
        bridge_tests._callback_integration_authority_fixture(tmp_path)
    )
    configured_root = (
        tmp_path
        / "state"
        / "lane-1"
        / "vrif_lane_1_database_portal_attempts"
    )
    configured_root.mkdir(parents=True)
    daemon._merge_portal_attempt_root = configured_root

    verified = daemon._verified_post_merge_callback_integration_receipt(
        qualification,
        recovery_evidence=evidence,
    )
    assert verified["receipt_id"] == qualification["receipt_id"]

    foreign_root = (
        tmp_path
        / "state"
        / "lane-3"
        / "other_lane_3_database_portal_attempts"
    )
    foreign_root.mkdir(parents=True)
    daemon._merge_portal_attempt_root = foreign_root
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="source authority is invalid",
    ):
        daemon._verified_post_merge_callback_integration_receipt(
            qualification,
            recovery_evidence=evidence,
        )


def test_preauthorize_uses_blocked_receipt_when_phase_omits_portal_flags(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:post-merge-receipt-fallback",
    )
    try:
        daemon.materialize_population(_population(1))
        failed = daemon.claim_next()
        assert failed is not None
        failed = daemon.commit_phase(failed, "context")
        failed = daemon.commit_phase(
            failed,
            "failed",
            body={"reason": "post_merge_declared_outputs_missing"},
        )
        assert daemon._terminal_portal_failure_reason(failed) is None
        terminal = daemon._persist_terminal_portal_failure(
            failed,
            reason="post_merge_declared_outputs_missing",
            coordination_evidence=(
                daemon._reconcile_failed_attempt_coordination(failed)
            ),
        )
        assert terminal["status"] == "blocked"
        authorized = daemon.preauthorize_post_merge_declared_output_recovery(
            _post_merge_preauthorization(daemon, failed)
        )
        assert authorized["authorized"] is True
    finally:
        daemon.close()


def test_preauthorize_accepts_receipt_despite_later_unrelated_portal_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:post-merge-later-terminal",
    )
    try:
        daemon.materialize_population(_population(1))
        failed = daemon.claim_next()
        assert failed is not None
        failed = daemon.commit_phase(failed, "context")
        failed = daemon.commit_phase(
            failed,
            "failed",
            body={
                "reason": "post_merge_declared_outputs_missing",
                "portal_retryable_failure": False,
                "portal_terminal_failure": True,
            },
        )
        daemon._persist_terminal_portal_failure(
            failed,
            reason="post_merge_declared_outputs_missing",
            coordination_evidence=(
                daemon._reconcile_failed_attempt_coordination(failed)
            ),
        )
        original_history = daemon.phase_history(failed.attempt_id)

        def history_with_later_terminal(
            _attempt_id: str,
        ) -> list[dict[str, object]]:
            return [
                *original_history,
                {
                    "phase": ATTEMPT_PHASE_FAILED,
                    "body": {
                        "reason": "implementation_protected_path_incident_latched",
                        "portal_retryable_failure": False,
                        "portal_terminal_failure": True,
                    },
                },
            ]

        monkeypatch.setattr(daemon, "phase_history", history_with_later_terminal)
        authorized = daemon.preauthorize_post_merge_declared_output_recovery(
            _post_merge_preauthorization(daemon, failed)
        )
        assert authorized["authorized"] is True
    finally:
        daemon.close()


def test_preauthorize_rejects_when_receipt_is_not_post_merge_terminal(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:post-merge-foreign-receipt",
    )
    try:
        daemon.materialize_population(_population(1))
        failed = daemon.claim_next()
        assert failed is not None
        failed = daemon.commit_phase(failed, "context")
        failed = daemon.commit_phase(
            failed,
            "failed",
            body={
                "reason": "implementation_protected_path_incident_latched",
                "portal_retryable_failure": False,
                "portal_terminal_failure": True,
            },
        )
        daemon._persist_terminal_portal_failure(
            failed,
            reason="implementation_protected_path_incident_latched",
            coordination_evidence=(
                daemon._reconcile_failed_attempt_coordination(failed)
            ),
        )
        with pytest.raises(
            DatabaseImplementationConflictError,
            match="no longer matches the latest terminal failure",
        ):
            daemon.preauthorize_post_merge_declared_output_recovery(
                _post_merge_preauthorization(daemon, failed)
            )
    finally:
        daemon.close()


def test_preauthorize_accepts_cross_board_completion_terminal(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:post-merge-cross-board-terminal",
    )
    try:
        daemon.materialize_population(_population(1))
        failed = daemon.claim_next()
        assert failed is not None
        failed = daemon.commit_phase(failed, "context")
        failed = daemon.commit_phase(
            failed,
            "failed",
            body={
                "reason": (
                    "cross_board_manual_completion_authority_metadata_invalid"
                ),
                "portal_retryable_failure": False,
                "portal_terminal_failure": True,
            },
        )
        terminal = daemon._persist_terminal_portal_failure(
            failed,
            reason=(
                "cross_board_manual_completion_authority_metadata_invalid"
            ),
            coordination_evidence=(
                daemon._reconcile_failed_attempt_coordination(failed)
            ),
        )
        assert terminal["status"] == "blocked"
        authorized = daemon.preauthorize_post_merge_declared_output_recovery(
            _post_merge_preauthorization(daemon, failed)
        )
        assert authorized["authorized"] is True
        assert authorized["task_status"] == "blocked"
        stale = _post_merge_preauthorization(daemon, failed)
        stale["source_attempt_id"] = "attempt:prior-repair"
        authorized_prior = daemon.preauthorize_post_merge_declared_output_recovery(
            stale
        )
        assert authorized_prior["authorized"] is True
    finally:
        daemon.close()


def test_preauthorize_accepts_binding_changed_resume_receipt_from_later_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:post-merge-binding-changed",
    )
    try:
        daemon.materialize_population(_population(1))
        failed = daemon.claim_next()
        assert failed is not None
        failed = daemon.commit_phase(failed, "context")
        failed = daemon.commit_phase(
            failed,
            "failed",
            body={"reason": "post_merge_declared_outputs_missing"},
        )
        daemon._persist_terminal_portal_failure(
            failed,
            reason="post_merge_declared_outputs_missing",
            coordination_evidence=(
                daemon._reconcile_failed_attempt_coordination(failed)
            ),
        )
        blocked = daemon.task_source.get(failed.task_cid)
        assert blocked is not None
        body = dict(blocked.body)
        receipt = dict(body["completion_receipt"])
        receipt["attempt_id"] = "attempt:later-resume"
        receipt["reason"] = (
            "database Portal attempt binding changed across resume"
        )
        body["completion_receipt"] = receipt
        wrapped = SimpleNamespace(
            task_cid=blocked.task_cid,
            task_alias=blocked.task_alias,
            status=blocked.status,
            revision=blocked.revision,
            body=body,
        )
        original_get = daemon.task_source.get

        def get_task(cid: str) -> object:
            if cid == failed.task_cid:
                return wrapped
            return original_get(cid)

        monkeypatch.setattr(daemon.task_source, "get", get_task)
        authorized = daemon.preauthorize_post_merge_declared_output_recovery(
            _post_merge_preauthorization(daemon, failed)
        )
        assert authorized["authorized"] is True
        assert authorized["task_status"] == "blocked"
    finally:
        daemon.close()


def test_exact_repair_evidence_rearms_only_matching_blocked_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:post-merge-repair-rearm",
    )
    try:
        daemon.materialize_population(_population(1))
        failed = daemon.claim_next()
        assert failed is not None
        failed = daemon.commit_phase(failed, "context")
        failed = daemon.commit_phase(
            failed,
            "failed",
            body={
                "reason": "post_merge_declared_outputs_missing",
                "portal_retryable_failure": False,
                "portal_terminal_failure": True,
            },
        )
        coordination = daemon._reconcile_failed_attempt_coordination(failed)
        terminal = daemon._persist_terminal_portal_failure(
            failed,
            reason="post_merge_declared_outputs_missing",
            coordination_evidence=coordination,
        )
        assert terminal["status"] == "blocked"
        blocked = daemon.task_source.get(failed.task_cid)
        assert blocked is not None
        assert blocked.status == "blocked"

        candidate_commit = "a" * 40
        repair_commit = "b" * 40
        repair_receipt: dict[str, object] = {
            "schema": POST_MERGE_DECLARED_OUTPUT_REPAIR_SCHEMA,
            "task_ids": [failed.task_alias],
            "candidate_commit": candidate_commit,
            "candidate_tree": "1" * 40,
            "baseline_commit": "2" * 40,
            "failed_integration_commit": "3" * 40,
            "repair_parent_commit": "4" * 40,
            "repair_commit": repair_commit,
            "repair_tree": "5" * 40,
            "entries": [
                {
                    "path": "output.py",
                    "mode": "100644",
                    "object_type": "blob",
                    "object_id": "6" * 40,
                }
            ],
            "validation": [
                {
                    "task_id": failed.task_alias,
                    "passed": True,
                    "returncode": 0,
                    "validation_result_digests": [
                        "sha256:" + "7" * 64
                    ],
                    "command_count": 1,
                    "log_sha256": "8" * 64,
                }
            ],
            "rollback_target": "4" * 40,
        }
        repair_receipt["receipt_id"] = content_identity(repair_receipt)
        evidence: dict[str, object] = {
            "schema": DATABASE_POST_MERGE_RECOVERY_SCHEMA,
            "request_id": "merge-request:vrif-010",
            "task_cid": failed.task_cid,
            "task_alias": failed.task_alias,
            "candidate_commit": candidate_commit,
            "repair_commit": repair_commit,
            "repair_receipt_id": repair_receipt["receipt_id"],
            "repair_receipt": repair_receipt,
            "source_attempt_id": failed.attempt_id,
            "source_claim_id": failed.claim_id,
            "source_lease_id": failed.lease_id,
            "source_fencing_token": failed.fencing_token,
            "source_fence_epoch": failed.fence_epoch,
            "source_binding_id": "sha256:" + "c" * 64,
            "source_projection_immutable_digest": "sha256:" + "d" * 64,
        }
        evidence["evidence_id"] = daemon._database_portal_evidence_digest(
            evidence
        )
        preauthorization = {
            "schema": DATABASE_POST_MERGE_RECOVERY_PREAUTHORIZATION_SCHEMA,
            **{
                field: evidence[field]
                for field in (
                    "request_id",
                    "task_cid",
                    "task_alias",
                    "candidate_commit",
                    "source_attempt_id",
                    "source_claim_id",
                    "source_lease_id",
                    "source_fencing_token",
                    "source_fence_epoch",
                    "source_binding_id",
                    "source_projection_immutable_digest",
                )
            },
        }
        authorized = daemon.preauthorize_post_merge_declared_output_recovery(
            preauthorization
        )
        assert authorized["authorized"] is True
        assert authorized["task_status"] == "blocked"
        authorization_id = authorized.pop("authorization_id")
        assert authorization_id == daemon._database_portal_evidence_digest(
            authorized
        )

        foreign = {**evidence, "repair_commit": "c" * 40}
        foreign.pop("evidence_id")
        foreign["evidence_id"] = daemon._database_portal_evidence_digest(
            foreign
        )
        with pytest.raises(DatabaseImplementationAuthorityError):
            daemon.recover_blocked_post_merge_declared_outputs(foreign)
        assert daemon.task_source.get(failed.task_cid).status == "blocked"

        foreign_task = {**evidence, "task_cid": "task:cid:foreign"}
        foreign_task.pop("evidence_id")
        foreign_task["evidence_id"] = (
            daemon._database_portal_evidence_digest(foreign_task)
        )
        with pytest.raises(DatabaseImplementationAuthorityError):
            daemon.recover_blocked_post_merge_declared_outputs(foreign_task)
        assert daemon.task_source.get(failed.task_cid).status == "blocked"

        stale_attempt = {**evidence, "source_attempt_id": "attempt:stale"}
        stale_attempt.pop("evidence_id")
        stale_attempt["evidence_id"] = (
            daemon._database_portal_evidence_digest(stale_attempt)
        )
        with pytest.raises(DatabaseImplementationConflictError):
            daemon.recover_blocked_post_merge_declared_outputs(stale_attempt)
        stale_preauthorization = {
            **preauthorization,
            "source_attempt_id": "attempt:stale",
        }
        with pytest.raises(DatabaseImplementationConflictError):
            daemon.preauthorize_post_merge_declared_output_recovery(
                stale_preauthorization
            )
        assert daemon.task_source.get(failed.task_cid).status == "blocked"

        # Simulate loss of the response after the atomic queue/control
        # transaction commits.  Replay must observe the committed transition
        # without incrementing either authority again.
        recovery_started_ms = daemon._now_ms()
        original_guarded_recovery = (
            daemon.task_source.record_queue_backoff_and_cas_status
        )

        def lose_cas_response(*args: object, **kwargs: object) -> None:
            original_guarded_recovery(*args, **kwargs)
            with pytest.raises(
                DatabaseCoordinationConflictError,
                match="must not re-enter",
            ):
                daemon.coordinator.claim_ready_task(
                    owner_session_id="session:illicit-recovery-successor",
                )
            raise RuntimeError("fixture crash between queue and control CAS")

        monkeypatch.setattr(
            daemon.task_source,
            "record_queue_backoff_and_cas_status",
            lose_cas_response,
        )
        with pytest.raises(
            RuntimeError,
            match="fixture crash between queue and control CAS",
        ):
            daemon.recover_blocked_post_merge_declared_outputs(evidence)
        queue_after_partial_write = daemon.task_source.get_queue_entry(
            failed.task_cid
        )
        assert queue_after_partial_write is not None
        queue_attempt_after_partial_write = queue_after_partial_write.attempt
        assert daemon.task_source.get(failed.task_cid).status == "retrying"
        assert (
            daemon.coordinator.get_task_claim_successor_projection(
                task_cid=failed.task_cid,
                after_fencing_token=failed.fencing_token,
                after_fence_epoch=failed.fence_epoch,
            )
            is None
        )
        daemon.close()
        daemon = _open_daemon(
            tmp_path,
            session="session:post-merge-repair-rearm",
        )

        recovered = daemon.recover_blocked_post_merge_declared_outputs(
            evidence
        )
        assert recovered["schema"] == DATABASE_POST_MERGE_RECOVERY_SCHEMA
        assert recovered["recovered"] is True
        assert recovered["changed"] is False
        assert recovered["status"] == "retrying"
        assert recovered["write_count"] == 0
        assert recovered["task_cid"] == failed.task_cid
        assert recovered["request_id"] == evidence["request_id"]
        assert recovered["repair_commit"] == repair_commit
        assert recovered["evidence_id"] == evidence["evidence_id"]

        retrying = daemon.task_source.get(failed.task_cid)
        assert retrying is not None
        assert retrying.status == "retrying"
        with pytest.raises(
            DatabaseImplementationConflictError,
            match="already rearmed",
        ):
            daemon.preauthorize_post_merge_declared_output_recovery(
                preauthorization
            )
        control_receipt = retrying.body["completion_receipt"]
        assert control_receipt["operation"] == (
            "database_post_merge_declared_outputs_repair_recovery"
        )
        assert control_receipt["repair_evidence_id"] == evidence[
            "evidence_id"
        ]
        assert control_receipt["repair_receipt_id"] == repair_receipt[
            "receipt_id"
        ]
        queue_entry = daemon.task_source.get_queue_entry(failed.task_cid)
        assert queue_entry is not None
        assert queue_entry.attempt == queue_attempt_after_partial_write
        assert (
            recovery_started_ms
            <= queue_entry.retry_not_before_ms
            <= daemon._now_ms()
        )
        assert queue_entry.reason.startswith(
            "database_post_merge_declared_outputs_repair:"
        )

        assert daemon.reconcile_terminal_portal_failures() == []

        daemon.close()
        daemon = _open_daemon(
            tmp_path,
            session="session:post-merge-repair-rearm",
        )
        repeated = daemon.recover_blocked_post_merge_declared_outputs(
            evidence
        )
        assert repeated["recovered"] is True
        assert repeated["changed"] is False
        assert repeated["status"] == "retrying"
        assert repeated["write_count"] == 0
    finally:
        daemon.close()


def test_post_merge_completion_seed_is_one_shot_per_target_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "completion-seed-repo"
    repo.mkdir()

    def git(*argv: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo), *argv],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    git("init", "-q", "-b", "main")
    git("config", "user.name", "Completion Seed Test")
    git("config", "user.email", "completion-seed@example.invalid")
    (repo / "generation.txt").write_text("one\n", encoding="utf-8")
    git("add", "generation.txt")
    git("commit", "-q", "-m", "generation one")
    first_target = git("rev-parse", "HEAD")

    daemon = _open_daemon(
        tmp_path,
        session="session:completion-seed-generation",
        lane="completion-seed",
    )
    try:
        daemon.materialize_population(_population(1))
        source = daemon.claim_next()
        assert source is not None
        task = daemon.task_source.get(source.task_cid)
        assert task is not None
        daemon._merge_repo_root = repo
        daemon._merge_target_branch = "main"
        evidence = {
            "request_id": "merge-request:completion-seed",
            "candidate_commit": "a" * 40,
            "source_attempt_id": "attempt:old-queue-source",
            "source_claim_id": "claim:old-queue-source",
            "source_lease_id": "lease:old-queue-source",
            "source_fencing_token": 7,
            "source_fence_epoch": 3,
            "source_binding_id": "sha256:" + "b" * 64,
            "source_projection_immutable_digest": "sha256:" + "c" * 64,
        }
        seed = daemon._build_post_merge_completion_recovery_seed(
            attempt=source,
            task_revision=int(task.revision),
            evidence=evidence,
            qualified_target_commit=first_target,
            qualification_kind="repair",
            qualification_receipt_id="repair-receipt:completion-seed",
            recovery_evidence_id="sha256:" + "d" * 64,
            terminal_reason=(
                DATABASE_PORTAL_COMPLETION_IMPLEMENTATION_COMMIT_MISSING_REASON
            ),
        )
        assert seed["schema"] == (
            DATABASE_POST_MERGE_COMPLETION_RECOVERY_SEED_SCHEMA
        )
        seed_body = dict(seed)
        seed_id = seed_body.pop("seed_id")
        assert seed_id == daemon._database_portal_evidence_digest(seed_body)
        noncanonical_v2 = {
            **seed_body,
            "schema": DATABASE_POST_MERGE_COMPLETION_RECOVERY_SEED_SCHEMA_V2,
            "recovery_control_revision": seed["source_task_revision"],
        }
        noncanonical_v2["seed_id"] = (
            daemon._database_portal_evidence_digest(noncanonical_v2)
        )
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="recovery seed is invalid",
        ):
            daemon._verified_post_merge_completion_recovery_seed(
                noncanonical_v2
            )

        consumer = replace(
            source,
            attempt_id="attempt:completion-seed-consumer",
            claim_id="claim:completion-seed-consumer",
            lease_id="lease:completion-seed-consumer",
            body={
                "post_merge_completion_recovery_source_attempt_id": (
                    source.attempt_id
                ),
                "post_merge_completion_recovery_seed": seed,
            },
        )
        assert daemon._post_merge_completion_recovery_was_consumed(consumer)

        malformed = replace(
            consumer,
            body={
                "post_merge_completion_recovery_source_attempt_id": (
                    source.attempt_id
                ),
                "post_merge_completion_recovery_seed": {
                    **seed,
                    "seed_id": "sha256:" + "0" * 64,
                },
            },
        )
        assert daemon._post_merge_completion_recovery_was_consumed(malformed)

        (repo / "generation.txt").write_text("two\n", encoding="utf-8")
        git("add", "generation.txt")
        git("commit", "-q", "-m", "generation two")
        assert daemon._post_merge_completion_target_advanced(seed)
        assert not daemon._post_merge_completion_recovery_was_consumed(consumer)

        target_changed_task = SimpleNamespace(
            body={
                "completion_receipt": {
                    "operation": "database_portal_terminal_failure",
                    "attempt_id": consumer.attempt_id,
                    "reason": (
                        DATABASE_POST_MERGE_COMPLETION_TARGET_GENERATION_CHANGED_REASON
                    ),
                }
            }
        )
        assert daemon._is_post_merge_completion_target_generation_changed_terminal(
            consumer,
            target_changed_task,
        )

        historical_source = replace(
            source,
            committed_phase=ATTEMPT_PHASE_FAILED,
            status="failed",
            finished_at_ms=source.started_at_ms + 1,
        )
        terminal_receipt = {
            "operation": "database_portal_terminal_failure",
            "attempt_id": source.attempt_id,
            "attempt_number": source.attempt_number,
            "claim_id": source.claim_id,
            "lease_id": source.lease_id,
            "owner_session_id": source.owner_session_id,
            "fencing_token": source.fencing_token,
            "fence_epoch": source.fence_epoch,
            "execution_phase": ATTEMPT_PHASE_FAILED,
            "execution_revision": source.revision,
            "execution_finished_at_ms": historical_source.finished_at_ms,
            "reason": (
                DATABASE_PORTAL_COMPLETION_IMPLEMENTATION_COMMIT_MISSING_REASON
            ),
            "retryable": False,
            "coordination": {},
            "control_expected_status": "in_progress",
            "control_expected_revision": seed["source_task_revision"] - 1,
        }
        history_body = {
            "schema": TASK_REVISION_HISTORY_PROJECTION_SCHEMA,
            "task_cid": source.task_cid,
            "revisions": [
                {
                    "revision": seed["source_task_revision"],
                    "status": "blocked",
                    "body": {"completion_receipt": terminal_receipt},
                }
            ],
        }
        history = {**history_body, "projection_cid": content_identity(history_body)}
        monkeypatch.setattr(
            daemon.task_source,
            "task_revision_history_projection",
            lambda _task_cid: history,
        )
        assert daemon._post_merge_completion_terminal_receipt_from_history(
            attempt=historical_source,
            seed=seed,
        ) == terminal_receipt

        swapped = {
            **seed,
            "terminal_reason": (
                DATABASE_POST_MERGE_COMPLETION_TARGET_GENERATION_CHANGED_REASON
            ),
        }
        swapped_body = dict(swapped)
        swapped_body.pop("seed_id")
        swapped["seed_id"] = daemon._database_portal_evidence_digest(
            swapped_body
        )
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="terminal history does not reproduce",
        ):
            daemon._post_merge_completion_terminal_receipt_from_history(
                attempt=historical_source,
                seed=daemon._verified_post_merge_completion_recovery_seed(
                    swapped
                ),
            )
    finally:
        daemon.close()


def test_cross_lane_post_merge_completion_recovery_uses_ordinary_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "completion-recovery-repo"
    repo.mkdir()

    def git(*argv: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo), *argv],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    git("init", "-q", "-b", "main")
    git("config", "user.name", "Cross Lane Completion Test")
    git("config", "user.email", "cross-lane-completion@example.invalid")
    artifact = repo / "recovered.txt"
    artifact.write_text("baseline\n", encoding="utf-8")
    git("add", "recovered.txt")
    git("commit", "-q", "-m", "completion baseline")
    baseline_commit = git("rev-parse", "HEAD")
    git("switch", "-q", "-c", "implementation/completion-recovery")
    artifact.write_text("candidate\n", encoding="utf-8")
    git("add", "recovered.txt")
    git("commit", "-q", "-m", "completion candidate")
    candidate_commit = git("rev-parse", "HEAD")
    candidate_tree = git("rev-parse", "HEAD^{tree}")
    candidate_blob = git("rev-parse", "HEAD:recovered.txt")
    git("switch", "-q", "main")
    assert git("rev-list", "--parents", "-n", "1", candidate_commit).split() == [
        candidate_commit,
        baseline_commit,
    ]

    portal_state = tmp_path / "portal-state"
    source_attempt_root = (
        portal_state / "lane-0" / "vrif_lane_0_database_portal_attempts"
    )
    consumer_attempt_root = (
        portal_state / "lane-3" / "vrif_lane_3_database_portal_attempts"
    )
    queue = MergeQueue(
        tmp_path / "merge-queue",
        target_repository_id=checkout_repository_id(repo),
        target_branch="main",
        require_target_binding=True,
    )
    source_portal_calls: list[str] = []
    source_daemon = _open_daemon(
        tmp_path,
        session="session:completion-recovery-source",
        lane="0",
    )
    consumer_daemon: DatabaseImplementationDaemon | None = None
    try:
        population = _population(1)
        [task_population] = population["tasks"]
        assert isinstance(task_population, dict)
        task_population.update(
            {
                "task_id": "VRIF-029",
                "title": "Recover exact post-merge completion",
                "outputs": [{"path": "recovered.txt"}],
                "allowed_paths": ["recovered.txt"],
                "validations": [{"argv": ["focused-recovery-validation"]}],
                "acceptance": [{"criterion": "Recovery is complete"}],
            }
        )
        source_daemon.materialize_population(population)
        source_attempt = source_daemon.claim_next()
        assert source_attempt is not None
        source_record = source_daemon.task_source.get_task(
            source_attempt.task_cid
        )
        assert source_record is not None

        source_bridge = DatabasePortalExecutionBridge(
            task_source=source_daemon.task_source,
            attempt_root=source_attempt_root,
            portal_factory=lambda _paths, alias: (
                source_portal_calls.append(alias)
                or pytest.fail("exact repair unexpectedly dispatched Portal")
            ),
            repository_root=repo,
            merge_queue=queue,
            merge_target_branch="main",
            task_header_prefix="## VRIF-",
        )
        source_paths, source_binding = (
            source_bridge._ensure_attempt_projection(
                source_attempt,
                source_record,
            )
        )
        [projected_task] = parse_task_text(
            source_paths.task_projection.read_text(encoding="utf-8"),
            path=source_paths.task_projection,
            task_header_prefix="## VRIF-",
        )
        local_identity = portal_task_identity(
            projected_task,
            todo_path=source_paths.task_projection,
        )
        assert len(
            {
                str(source_binding["task_cid"]),
                projected_task.canonical_task_cid,
                local_identity.canonical_task_cid,
            }
        ) == 3
        request = queue.enqueue(
            branch_name="implementation/completion-recovery",
            task_id=source_attempt.task_alias,
            canonical_task_id=local_identity.canonical_task_cid,
            canonical_task_key=local_identity.canonical_task_key,
            commit_sha=candidate_commit,
            metadata={
                "schema": (
                    "ipfs_accelerate_py/agent-supervisor/merge-candidate@3"
                ),
                "target_binding_schema": (
                    "ipfs_accelerate_py/agent-supervisor/"
                    "merge-target-binding@1"
                ),
                "target_repository_id": checkout_repository_id(repo),
                "target_branch": "main",
                "implementation_commit": candidate_commit,
                "todo_path": str(source_paths.task_projection),
                "state_path": str(source_paths.state),
                "strategy_path": str(source_paths.strategy),
                "events_path": str(source_paths.events),
                "repo_root": str(repo.absolute()),
                "task_header_prefix": "## VRIF-",
                "task": asdict(projected_task),
                "completion_task_cids": {
                    source_attempt.task_alias: local_identity.canonical_task_cid
                },
                "changed_submodule_paths": [],
            },
        )
        false_claim = queue.claim_pending_request(
            request.request_id,
            consumer_id="merge-train:false-completion-fixture",
        )
        assert false_claim is not None
        queue.complete(false_claim)
        false_completion = queue.get(request.request_id)
        assert false_completion is not None
        assert false_completion.status == "completed"
        reopened = queue.reopen_false_positive_completion(
            false_completion,
            completion_receipt={
                "already_merged": True,
                "canonical_task_id": false_completion.canonical_identity,
                "commit_sha": candidate_commit,
                "distributed_publication_admission": {
                    "admitted": True,
                    "distributed": False,
                    "request_id": request.request_id,
                    "schema": (
                        "ipfs_accelerate_py/agent-supervisor/"
                        "distributed-lane-admission@1"
                    ),
                    "status": "local",
                },
                "finished_at": 2.0,
                "integrated": True,
                "merge_commit": baseline_commit,
                "merged": False,
                "mutation_short_circuited": True,
                "reason": "declared_outputs_already_on_target",
                "request_id": request.request_id,
                "started_at": 1.0,
                "status": "already_merged",
                "target_branch": "main",
                "target_commit": baseline_commit,
                "task_id": source_attempt.task_alias,
            },
        )
        assert reopened is not None
        assert reopened.status == "pending"
        git("merge", "-q", "--ff-only", candidate_commit)
        queue_claim = queue.claim_pending_request(
            request.request_id,
            consumer_id="merge-train:completion-recovery-fixture",
        )
        assert queue_claim is not None
        repair_receipt: dict[str, object] = {
            "schema": POST_MERGE_DECLARED_OUTPUT_REPAIR_SCHEMA,
            "task_ids": [source_attempt.task_alias],
            "candidate_commit": candidate_commit,
            "candidate_tree": candidate_tree,
            "baseline_commit": baseline_commit,
            "failed_integration_commit": candidate_commit,
            "repair_parent_commit": baseline_commit,
            "repair_commit": candidate_commit,
            "repair_tree": candidate_tree,
            "entries": [
                {
                    "path": "recovered.txt",
                    "mode": "100644",
                    "object_type": "blob",
                    "object_id": candidate_blob,
                }
            ],
            "validation": [
                {
                    "task_id": source_attempt.task_alias,
                    "passed": True,
                    "returncode": 0,
                    "validation_result_digests": ["sha256:" + "2" * 64],
                    "command_count": 1,
                    "log_sha256": "3" * 64,
                }
            ],
            "rollback_target": baseline_commit,
        }
        repair_receipt["receipt_id"] = content_identity(repair_receipt)
        queue.complete(
            queue_claim,
            metadata={
                "schema": (
                    "ipfs_accelerate_py/agent-supervisor/"
                    "post-merge-declared-output-completion@1"
                ),
                "status": "already_merged",
                "reason": "post_merge_declared_outputs_repaired",
                "candidate_commit": candidate_commit,
                "target_commit": candidate_commit,
                "repair_receipt": repair_receipt,
            },
        )
        completed_request = queue.get(request.request_id)
        assert completed_request is not None
        assert completed_request.status == "completed"

        source_attempt = source_daemon.commit_phase(source_attempt, "context")
        source_attempt = source_daemon.commit_phase(
            source_attempt,
            ATTEMPT_PHASE_FAILED,
            body={
                "reason": (
                    DATABASE_PORTAL_COMPLETION_IMPLEMENTATION_COMMIT_MISSING_REASON
                ),
                "portal_retryable_failure": False,
                "portal_terminal_failure": True,
            },
        )
        coordination = source_daemon._reconcile_failed_attempt_coordination(
            source_attempt
        )
        terminal = source_daemon._persist_terminal_portal_failure(
            source_attempt,
            reason=(
                DATABASE_PORTAL_COMPLETION_IMPLEMENTATION_COMMIT_MISSING_REASON
            ),
            coordination_evidence=coordination,
        )
        assert terminal["status"] == "blocked"
        blocked_record = source_daemon.task_source.get(source_attempt.task_cid)
        assert blocked_record is not None
        assert blocked_record.status == "blocked"
        blocked_revision = blocked_record.revision

        recovery = source_bridge.recover_post_merge_declared_outputs(
            source_daemon
        )
        assert recovery is not None
        assert recovery["recovered"] is True
        assert recovery["changed"] is True
        assert recovery["status"] == "retrying"
        assert source_portal_calls == []
        retrying_record = source_daemon.task_source.get(
            source_attempt.task_cid
        )
        assert retrying_record is not None
        recovery_control = retrying_record.body["completion_receipt"]
        recovery_seed = recovery_control[
            "post_merge_completion_recovery_seed"
        ]
        assert recovery_seed["schema"] == (
            DATABASE_POST_MERGE_COMPLETION_RECOVERY_SEED_SCHEMA
        )
        assert recovery_seed["source_task_revision"] == blocked_revision
        assert recovery_seed["attempt_id"] == source_attempt.attempt_id
        assert recovery_seed["qualified_target_commit"] == candidate_commit

        source_daemon.close()
        consumer_bridges: list[DatabasePortalExecutionBridge] = []

        def bridge_provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
            return dict(consumer_bridges[0].run_provider(attempt))

        def bridge_effect(
            attempt: DatabaseTaskAttempt,
            provider_result: dict[str, object],
        ) -> dict[str, object]:
            return dict(
                consumer_bridges[0].apply_effect(attempt, provider_result)
            )

        def bridge_validation(
            attempt: DatabaseTaskAttempt,
            effect_result: dict[str, object],
        ) -> dict[str, object]:
            return dict(
                consumer_bridges[0].validate_effect(attempt, effect_result)
            )

        consumer_daemon = _open_daemon(
            tmp_path,
            session="session:completion-recovery-consumer",
            lane="3",
            provider_fn=bridge_provider,
            effect_fn=bridge_effect,
            validation_fn=bridge_validation,
            max_task_attempts=1,
        )
        consumer_daemon._merge_repo_root = repo
        consumer_daemon._merge_target_branch = "main"
        consumer_portal_calls: list[str] = []
        requalification_heads: list[str] = []

        class RecoveryPortal:
            def __init__(self) -> None:
                self.merge_queue = queue
                self.repo_root = repo.absolute()
                self.resolved_merge_target_branch = "main"

            @staticmethod
            def _load_tasks() -> list[object]:
                return [projected_task]

            @staticmethod
            def _run_checkout_mutation_transaction(
                *, callback: object, **_kwargs: object
            ) -> dict[str, object]:
                return callback()

            @staticmethod
            def _run_validation_commands(
                workspace: Path,
                task: object,
                log_path: Path,
                *,
                force_uncached: bool,
            ) -> dict[str, object]:
                assert task.task_id == "VRIF-029"
                assert force_uncached is True
                head = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=workspace,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                requalification_heads.append(head)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(
                    "fresh current-target validation passed\n",
                    encoding="utf-8",
                )
                return {
                    "attempted": True,
                    "passed": True,
                    "returncode": 0,
                    "results": [
                        {
                            "validation_result_digest": (
                                "sha256:" + "4" * 64
                            )
                        }
                    ],
                }

            @staticmethod
            def _cleanup_main_merge_workspace(
                workspace: Path,
                *,
                ephemeral: bool,
            ) -> dict[str, object]:
                assert ephemeral is True
                removed = subprocess.run(
                    [
                        "git",
                        "worktree",
                        "remove",
                        "--force",
                        str(workspace),
                    ],
                    cwd=repo,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                return {"cleaned": removed.returncode == 0}

            @staticmethod
            def close_event_runtime() -> None:
                return None

        def fresh_consumer_bridge() -> DatabasePortalExecutionBridge:
            return DatabasePortalExecutionBridge(
                task_source=consumer_daemon.task_source,
                attempt_root=consumer_attempt_root,
                portal_factory=lambda _paths, alias: (
                    consumer_portal_calls.append(alias) or RecoveryPortal()
                ),
                repository_root=repo,
                merge_queue=queue,
                merge_target_branch="main",
                task_header_prefix="## VRIF-",
            )

        consumer_bridge = fresh_consumer_bridge()
        consumer_bridges.append(consumer_bridge)

        # Put the retained execution/coordination generation well ahead of a
        # future replacement sidecar.  Attempt numbers are lane-local and the
        # replacement will restart them at one; canonical receipt identity,
        # not the incomparable counters, must select a later recurrence.
        assert consumer_daemon.sync_ready_tasks_into_coordination() == [
            source_attempt.task_cid
        ]
        for preclaim_index in range(10):
            preclaim = consumer_daemon.coordinator.claim_ready_task(
                owner_session_id=(
                    "session:completion-recovery-preclaim:"
                    f"{preclaim_index}"
                ),
                lease_ms=60_000,
                now_ms=consumer_daemon._now_ms(),
            )
            assert preclaim is not None
            preclaim_lease = consumer_daemon.coordinator.get_lease(
                preclaim.lease_id
            )
            assert preclaim_lease is not None
            consumer_daemon.coordinator.release(
                preclaim_lease,
                reason="test_attempt_frontier_seed",
                now_ms=consumer_daemon._now_ms(),
            )

        successor = consumer_daemon.claim_next()
        assert successor is not None
        assert successor.attempt_number == 11
        assert successor.owner_session_id == (
            "session:completion-recovery-consumer"
        )
        assert successor.owner_session_id != source_attempt.owner_session_id
        assert successor.attempt_id != source_attempt.attempt_id
        assert successor.claim_id != source_attempt.claim_id
        assert successor.lease_id != source_attempt.lease_id
        assert successor.body[
            "post_merge_completion_recovery_source_attempt_id"
        ] == source_attempt.attempt_id
        assert successor.body["post_merge_completion_recovery_seed"] == (
            recovery_seed
        )
        claimed_record = consumer_daemon.task_source.get(successor.task_cid)
        assert claimed_record is not None
        assert claimed_record.status == "in_progress"
        assert claimed_record.revision == blocked_revision + 2
        claimed_control = claimed_record.body["completion_receipt"]
        assert claimed_control["operation"] == "database_claim"
        assert claimed_control["post_merge_completion_recovery_seed"] == (
            recovery_seed
        )

        # Reproduce the live crash boundary: Portal sealed the exact
        # zero-provider completion lineage, but the daemon lost the accepted
        # response before writing its provider/effect ledgers or database_complete.
        lost_acceptance = consumer_bridge.run_provider(successor)
        assert lost_acceptance["accepted"] is True
        assert consumer_daemon.provider_invocation_recorded(
            successor.attempt_id,
            idempotency_key=f"provider:{successor.attempt_id}",
        ) is None
        assert consumer_daemon.effect_claim_recorded(
            successor.attempt_id,
            idempotency_key=f"effect:{successor.attempt_id}",
        ) is None
        crashed_events = consumer_bridge._verified_event_chain(
            consumer_bridge._paths(successor)
        )
        assert [event["type"] for event in crashed_events] == [
            "worktree_reconciliation_candidate_queued",
            "merge_reconciled",
            "task_completed",
        ]
        assert crashed_events[0]["attempt_consumed"] is False
        assert crashed_events[0]["provider_dispatched"] is False

        failed_seeded = consumer_daemon.commit_phase(
            successor,
            ATTEMPT_PHASE_FAILED,
            body={
                "reason": "coordination_lease_expired_before_completion",
                "portal_retryable_failure": True,
                "backoff_seconds": 0,
            },
        )
        seeded_claim_record = consumer_daemon.coordinator.get_task_claim(
            failed_seeded.claim_id
        )
        assert seeded_claim_record is not None
        expired_now_ms = int(seeded_claim_record.expires_at_ms) + 1
        consumer_daemon._clock_ms = lambda: expired_now_ms
        consumer_daemon.coordinator.expire_task_claim(
            seeded_claim_record,
            now_ms=expired_now_ms,
        )
        seeded_coordination = (
            consumer_daemon._reconcile_failed_attempt_coordination(
                failed_seeded
            )
        )
        generic_retry = consumer_daemon._persist_task_retry_state(
            failed_seeded,
            reason="coordination_lease_expired_before_completion",
            backoff_ms=0,
            evidence_source="expired_seeded_completion_claim",
            coordination_evidence=seeded_coordination,
        )
        assert generic_retry["status"] == "retrying"
        ordinary = consumer_daemon.claim_next()
        assert ordinary is not None
        assert "post_merge_completion_recovery_seed" not in ordinary.body
        typed_reason = "portal_execution_deferred"
        typed_receipt = consumer_daemon._typed_deferral_receipt(
            ordinary,
            reason=typed_reason,
        )
        exhausted_attempt = consumer_daemon.commit_phase(
            ordinary,
            ATTEMPT_PHASE_FAILED,
            body={
                "reason": typed_reason,
                "portal_retryable_failure": True,
                "portal_terminal_failure": False,
                "deferred": True,
                "attempt_consumed": False,
                "provider_dispatched": False,
                "typed_deferral_slot_consumed": True,
                "backoff_seconds": 0,
                "typed_deferral": typed_receipt,
            },
        )
        budget = consumer_daemon._typed_deferral_budget_observation(
            exhausted_attempt
        )
        assert budget is not None
        assert budget["exhausted"] is True
        exhausted_coordination = (
            consumer_daemon._reconcile_failed_attempt_coordination(
                exhausted_attempt
            )
        )
        blocked_again = (
            consumer_daemon._persist_typed_deferral_budget_exhausted(
                exhausted_attempt,
                budget=budget,
                coordination_evidence=exhausted_coordination,
            )
        )
        assert blocked_again["status"] == "blocked"
        history = consumer_daemon.task_source.task_revision_history_projection(
            successor.task_cid
        )
        exact_chain = history["revisions"][-6:]
        assert [entry["status"] for entry in exact_chain] == [
            "blocked",
            "retrying",
            "in_progress",
            "retrying",
            "in_progress",
            "blocked",
        ]
        assert [
            entry["body"]["completion_receipt"]["operation"]
            for entry in exact_chain
        ] == [
            "database_portal_terminal_failure",
            "database_post_merge_declared_outputs_repair_recovery",
            "database_claim",
            "database_portal_retry",
            "database_claim",
            "database_portal_typed_deferral_budget_exhausted",
        ]
        assert consumer_daemon.post_merge_completion_recovery_task_cids() == (
            successor.task_cid,
        )
        blocked_record = consumer_daemon.task_source.get(successor.task_cid)
        assert blocked_record is not None
        assert (
            consumer_daemon._post_merge_completion_crash_recovery_context(
                blocked_record,
                require_current_blocked=True,
            )
            is not None
        )
        def mutable_history_value(value: object) -> object:
            if isinstance(value, Mapping):
                return {
                    str(key): mutable_history_value(item)
                    for key, item in value.items()
                }
            if isinstance(value, (list, tuple)):
                return [mutable_history_value(item) for item in value]
            return value

        seeded_chain = mutable_history_value(exact_chain)
        assert isinstance(seeded_chain, list)
        for offset, entry in enumerate(seeded_chain):
            entry["revision"] = 6 + offset
        seeded_chain[0]["body"]["completion_receipt"][
            "control_expected_revision"
        ] = 5
        seeded_recovery_receipt = seeded_chain[1]["body"][
            "completion_receipt"
        ]
        synthetic_seed = dict(
            seeded_recovery_receipt[
                "post_merge_completion_recovery_seed"
            ]
        )
        synthetic_seed.pop("seed_id")
        synthetic_seed["source_task_revision"] = 6
        synthetic_seed["seed_id"] = (
            consumer_daemon._database_portal_evidence_digest(
                synthetic_seed
            )
        )
        seeded_recovery_receipt[
            "post_merge_completion_recovery_seed"
        ] = synthetic_seed
        seeded_recovery_receipt["control_expected_revision"] = 6
        seeded_chain[2]["body"]["completion_receipt"][
            "post_merge_completion_recovery_seed"
        ] = synthetic_seed
        seeded_chain[3]["body"]["completion_receipt"][
            "control_expected_revision"
        ] = 8
        seeded_chain[5]["body"]["completion_receipt"][
            "control_expected_revision"
        ] = 10

        legacy_seedless_recovery = mutable_history_value(
            seeded_recovery_receipt
        )
        assert isinstance(legacy_seedless_recovery, dict)
        legacy_seedless_recovery.pop(
            "post_merge_completion_recovery_seed"
        )
        legacy_seedless_recovery["control_expected_revision"] = 1
        legacy_retry = mutable_history_value(
            seeded_chain[3]["body"]["completion_receipt"]
        )
        assert isinstance(legacy_retry, dict)
        legacy_retry["control_expected_revision"] = 3
        legacy_chain = [
            mutable_history_value(seeded_chain[0]),
            {
                "revision": 2,
                "status": "retrying",
                "body": {
                    **{
                        key: mutable_history_value(value)
                        for key, value in seeded_chain[0]["body"].items()
                        if key != "completion_receipt"
                    },
                    "completion_receipt": legacy_seedless_recovery,
                },
            },
            mutable_history_value(seeded_chain[4]),
            {
                "revision": 4,
                "status": "retrying",
                "body": {
                    **{
                        key: mutable_history_value(value)
                        for key, value in seeded_chain[0]["body"].items()
                        if key != "completion_receipt"
                    },
                    "completion_receipt": legacy_retry,
                },
            },
            mutable_history_value(seeded_chain[4]),
            mutable_history_value(seeded_chain[0]),
        ]
        for revision, entry in enumerate(legacy_chain, start=1):
            assert isinstance(entry, dict)
            entry["revision"] = revision
        synthetic_revisions = [*legacy_chain, *seeded_chain[1:]]
        synthetic_history_body = {
            "schema": TASK_REVISION_HISTORY_PROJECTION_SCHEMA,
            "task_cid": successor.task_cid,
            "revisions": synthetic_revisions,
        }
        legacy_then_seeded_history = {
            **synthetic_history_body,
            "projection_cid": content_identity(synthetic_history_body),
        }
        synthetic_blocked_record = replace(
            blocked_record,
            revision=11,
            body=synthetic_revisions[-1]["body"],
        )
        canonical_history_projection = (
            consumer_daemon.task_source.task_revision_history_projection
        )
        monkeypatch.setattr(
            consumer_daemon.task_source,
            "task_revision_history_projection",
            lambda _task_cid: legacy_then_seeded_history,
        )
        legacy_compatible_context = (
            consumer_daemon._post_merge_completion_crash_recovery_context(
                synthetic_blocked_record,
                require_current_blocked=True,
            )
        )
        assert legacy_compatible_context is not None
        assert legacy_compatible_context["control_task_revision"] == 6
        assert legacy_compatible_context["exhausted_task_revision"] == 11
        assert legacy_compatible_context["source_seed"] == synthetic_seed
        monkeypatch.setattr(
            consumer_daemon.task_source,
            "task_revision_history_projection",
            canonical_history_projection,
        )

        target_changed_history = mutable_history_value(history)
        assert isinstance(target_changed_history, dict)
        target_changed_chain = target_changed_history["revisions"][-6:]
        target_changed_chain[0]["body"]["completion_receipt"]["reason"] = (
            DATABASE_POST_MERGE_COMPLETION_TARGET_GENERATION_CHANGED_REASON
        )
        target_changed_seed = dict(
            target_changed_chain[1]["body"]["completion_receipt"][
                "post_merge_completion_recovery_seed"
            ]
        )
        target_changed_seed.pop("seed_id")
        target_changed_seed["terminal_reason"] = (
            DATABASE_POST_MERGE_COMPLETION_TARGET_GENERATION_CHANGED_REASON
        )
        target_changed_seed["seed_id"] = (
            consumer_daemon._database_portal_evidence_digest(
                target_changed_seed
            )
        )
        target_changed_chain[1]["body"]["completion_receipt"][
            "post_merge_completion_recovery_seed"
        ] = target_changed_seed
        target_changed_chain[2]["body"]["completion_receipt"][
            "post_merge_completion_recovery_seed"
        ] = target_changed_seed
        target_changed_body = dict(target_changed_history)
        target_changed_body.pop("projection_cid")
        target_changed_history["projection_cid"] = content_identity(
            target_changed_body
        )
        original_history_projection = (
            consumer_daemon.task_source.task_revision_history_projection
        )
        monkeypatch.setattr(
            consumer_daemon.task_source,
            "task_revision_history_projection",
            lambda _task_cid: target_changed_history,
        )
        target_changed_context = (
            consumer_daemon._post_merge_completion_crash_recovery_context(
                blocked_record,
                require_current_blocked=True,
            )
        )
        assert target_changed_context is not None
        assert target_changed_context["source_seed"]["terminal_reason"] == (
            DATABASE_POST_MERGE_COMPLETION_TARGET_GENERATION_CHANGED_REASON
        )
        monkeypatch.setattr(
            consumer_daemon.task_source,
            "task_revision_history_projection",
            original_history_projection,
        )

        # Replace the disposable lane-local coordination store while keeping
        # the canonical control history and execution attempts.  This is the
        # production crash/restart shape: the blocked receipt still seals the
        # exact old fence, but the fresh sidecar cannot reproduce its rows.
        blocked_receipt = blocked_record.body["completion_receipt"]
        portable_coordination = blocked_receipt["coordination"]
        portable_expiry_ms = int(portable_coordination["expires_at_ms"])
        consumer_daemon.close()
        consumer_daemon = DatabaseImplementationDaemon(
            database_path=tmp_path / "control.duckdb",
            coordination_path=(
                tmp_path / "coordination-lane-3-replacement.duckdb"
            ),
            execution_path=tmp_path / "execution-3.duckdb",
            owner_session_id="session:completion-recovery-consumer",
            authority_mode="embedded",
            task_source_kind="duckdb",
            provider_fn=bridge_provider,
            effect_fn=bridge_effect,
            validation_fn=bridge_validation,
            max_task_attempts=1,
            require_real_execution=True,
            clock_ms=lambda: portable_expiry_ms - 1,
        )
        consumer_daemon._merge_repo_root = repo
        consumer_daemon._merge_target_branch = "main"
        consumer_bridge = fresh_consumer_bridge()
        consumer_bridges[0] = consumer_bridge

        replacement_blocked = consumer_daemon.task_source.get(
            successor.task_cid
        )
        assert replacement_blocked is not None
        assert consumer_daemon.coordinator.get_task_claim(
            exhausted_attempt.claim_id
        ) is None
        assert consumer_daemon.coordinator.get_task_attempt(
            exhausted_attempt.attempt_id
        ) is None
        assert consumer_daemon.coordinator.get_lease(
            exhausted_attempt.lease_id
        ) is None
        replacement_projection = (
            consumer_daemon.coordinator.coordination_registry_projection()
        )
        for collection in (
            "tasks",
            "logical_completions",
            "task_claims",
            "task_attempts",
            "fenced_leases",
            "resource_claims",
        ):
            assert all(
                item.get("task_cid") != successor.task_cid
                for item in replacement_projection[collection]
            )

        # A missing local triple never grants ordinary retry authority, and
        # the portable crash classifier remains closed until the sealed lease
        # has certainly expired.  Both checks are read-only.
        before_expiry = replacement_blocked.to_dict()
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="has no coordination claim",
        ):
            consumer_daemon._reconcile_failed_attempt_coordination(
                exhausted_attempt
            )
        assert (
            consumer_daemon._post_merge_completion_crash_recovery_context(
                replacement_blocked,
                require_current_blocked=True,
            )
            is None
        )
        assert successor.task_cid not in (
            consumer_daemon.post_merge_completion_recovery_task_cids()
        )
        assert consumer_daemon.task_source.get(
            successor.task_cid
        ).to_dict() == before_expiry
        assert not (
            consumer_daemon._post_merge_completion_portable_coordination_authority(
                exhausted_attempt,
                persisted={},
            )
        )

        consumer_daemon._clock_ms = lambda: portable_expiry_ms + 1
        portable_context = (
            consumer_daemon._post_merge_completion_crash_recovery_context(
                replacement_blocked,
                require_current_blocked=True,
            )
        )
        assert portable_context is not None
        assert portable_context["portable_coordination_authority"] is True
        assert consumer_daemon.post_merge_completion_recovery_task_cids() == (
            successor.task_cid,
        )

        # Admission is rechecked immediately before the shared CAS.  A new
        # same-task authority appearing in the replacement sidecar wins this
        # race without invoking the callback.
        callback_calls: list[bool] = []
        with monkeypatch.context() as race_patch:
            raced_projection = dict(replacement_projection)
            raced_projection["tasks"] = [
                {
                    "task_cid": successor.task_cid,
                    "ready": True,
                }
            ]
            race_patch.setattr(
                consumer_daemon.coordinator,
                "coordination_registry_projection",
                lambda: raced_projection,
            )
            with pytest.raises(
                DatabaseImplementationConflictError,
                match="portable post-merge completion authority was superseded",
            ):
                consumer_daemon._execute_with_portable_post_merge_completion_authority(
                    exhausted_attempt,
                    portable_coordination,
                    lambda: callback_calls.append(True),
                )
        assert callback_calls == []

        (repo / "target-generation.txt").write_text(
            "completion recovery generation two\n",
            encoding="utf-8",
        )
        git("add", "target-generation.txt")
        git("commit", "-q", "-m", "advance completion target")
        advanced_target = git("rev-parse", "HEAD")
        assert advanced_target != candidate_commit

        # Claims refresh the daemon's connection-owned task source.  A real
        # supervisor tick reconstructs its bridge around that current source.
        consumer_bridge = fresh_consumer_bridge()
        consumer_bridges[0] = consumer_bridge
        replay_recovery = consumer_bridge.recover_post_merge_declared_outputs(
            consumer_daemon
        )
        if replay_recovery is None:
            # The durable priority keyset cursor needs one bounded tick to
            # wrap after this same task CID enters a new target generation.
            replay_recovery = (
                consumer_bridge.recover_post_merge_declared_outputs(
                    consumer_daemon
                )
            )
        assert replay_recovery is not None
        assert replay_recovery["recovered"] is True
        assert replay_recovery["changed"] is True
        retrying_again = consumer_daemon.task_source.get(successor.task_cid)
        assert retrying_again is not None
        assert (
            consumer_daemon._post_merge_completion_crash_recovery_context(
                retrying_again,
                require_current_blocked=False,
            )
            is None
        )
        assert successor.task_cid not in (
            consumer_daemon._automatic_claim_exclusions()
        )
        replay_seed = retrying_again.body["completion_receipt"][
            "post_merge_completion_recovery_seed"
        ]
        assert replay_seed["schema"] == (
            DATABASE_POST_MERGE_COMPLETION_RECOVERY_SEED_SCHEMA_V2
        )
        assert replay_seed["source_task_revision"] == blocked_revision
        assert replay_seed["recovery_control_revision"] == (
            blocked_revision + 5
        )
        assert replay_seed["qualified_target_commit"] == advanced_target
        assert requalification_heads == [advanced_target]

        # The retired ordinary claim remains absent from the replacement
        # sidecar after the shared recovery CAS.
        assert consumer_daemon.coordinator.get_task_claim(
            exhausted_attempt.claim_id
        ) is None

        successor = consumer_daemon.claim_next()
        assert successor is not None
        assert successor.attempt_number == 1
        assert successor.body["post_merge_completion_recovery_seed"] == (
            replay_seed
        )
        assert {
            attempt.task_cid: attempt.attempt_id
            for attempt in consumer_daemon._latest_failed_attempts()
        }[successor.task_cid] == exhausted_attempt.attempt_id
        running_control = consumer_daemon.task_source.get(successor.task_cid)
        assert running_control is not None
        running_control_before = running_control.to_dict()
        assert consumer_daemon.reconcile_terminal_portal_failures() == []
        running_retry_observations = (
            consumer_daemon.reconcile_terminal_retry_states()
        )
        assert len(running_retry_observations) == 1
        assert running_retry_observations[0]["changed"] is False
        assert running_retry_observations[0]["reason"] == (
            "failed_attempt_control_superseded"
        )
        assert consumer_daemon.task_source.get(
            successor.task_cid
        ).to_dict() == running_control_before

        # Lose the v2 zero-provider acceptance at the same boundary.  The
        # next exact suffix must link back through recovery_control_revision
        # while preserving the immutable historical terminal revision.
        second_lost_acceptance = consumer_bridge.run_provider(successor)
        assert second_lost_acceptance["accepted"] is True
        assert consumer_daemon.provider_invocation_recorded(
            successor.attempt_id,
            idempotency_key=f"provider:{successor.attempt_id}",
        ) is None
        second_failed_seeded = consumer_daemon.commit_phase(
            successor,
            ATTEMPT_PHASE_FAILED,
            body={
                "reason": "coordination_lease_expired_before_completion",
                "portal_retryable_failure": True,
                "backoff_seconds": 0,
            },
        )
        second_seeded_claim = consumer_daemon.coordinator.get_task_claim(
            second_failed_seeded.claim_id
        )
        assert second_seeded_claim is not None
        second_seeded_now_ms = int(second_seeded_claim.expires_at_ms) + 1
        consumer_daemon._clock_ms = lambda: second_seeded_now_ms
        consumer_daemon.coordinator.expire_task_claim(
            second_seeded_claim,
            now_ms=second_seeded_now_ms,
        )
        second_seeded_coordination = (
            consumer_daemon._reconcile_failed_attempt_coordination(
                second_failed_seeded
            )
        )
        second_generic_retry = consumer_daemon._persist_task_retry_state(
            second_failed_seeded,
            reason="coordination_lease_expired_before_completion",
            backoff_ms=0,
            evidence_source="expired_v2_seeded_completion_claim",
            coordination_evidence=second_seeded_coordination,
        )
        assert second_generic_retry["status"] == "retrying"
        second_ordinary = consumer_daemon.claim_next()
        assert second_ordinary is not None
        assert "post_merge_completion_recovery_seed" not in (
            second_ordinary.body
        )
        second_typed_receipt = consumer_daemon._typed_deferral_receipt(
            second_ordinary,
            reason=typed_reason,
        )
        second_exhausted_attempt = consumer_daemon.commit_phase(
            second_ordinary,
            ATTEMPT_PHASE_FAILED,
            body={
                "reason": typed_reason,
                "portal_retryable_failure": True,
                "portal_terminal_failure": False,
                "deferred": True,
                "attempt_consumed": False,
                "provider_dispatched": False,
                "typed_deferral_slot_consumed": True,
                "backoff_seconds": 0,
                "typed_deferral": second_typed_receipt,
            },
        )
        assert second_exhausted_attempt.attempt_number == 2
        assert {
            attempt.task_cid: attempt.attempt_id
            for attempt in consumer_daemon._latest_failed_attempts()
        }[successor.task_cid] == exhausted_attempt.attempt_id
        second_budget = consumer_daemon._typed_deferral_budget_observation(
            second_exhausted_attempt
        )
        assert second_budget is not None
        assert second_budget["exhausted"] is True
        assert second_budget["matching_attempts"][0]["attempt_id"] == (
            second_exhausted_attempt.attempt_id
        )
        second_exhausted_coordination = (
            consumer_daemon._reconcile_failed_attempt_coordination(
                second_exhausted_attempt
            )
        )
        second_blocked = (
            consumer_daemon._persist_typed_deferral_budget_exhausted(
                second_exhausted_attempt,
                budget=second_budget,
                coordination_evidence=second_exhausted_coordination,
            )
        )
        assert second_blocked["status"] == "blocked"
        second_blocked_record = consumer_daemon.task_source.get(
            successor.task_cid
        )
        assert second_blocked_record is not None
        second_blocked_before = second_blocked_record.to_dict()
        assert consumer_daemon.reconcile_terminal_portal_failures() == []
        blocked_retry_observations = (
            consumer_daemon.reconcile_terminal_retry_states()
        )
        assert blocked_retry_observations == []
        assert consumer_daemon.task_source.get(
            successor.task_cid
        ).to_dict() == second_blocked_before
        recurrence_context = (
            consumer_daemon._post_merge_completion_crash_recovery_context(
                second_blocked_record,
                require_current_blocked=True,
            )
        )
        assert recurrence_context is not None
        assert recurrence_context["current_attempt"] == (
            second_exhausted_attempt
        )
        assert recurrence_context["source_task_revision"] == blocked_revision
        assert recurrence_context["control_task_revision"] == (
            blocked_revision + 5
        )
        assert recurrence_context["source_seed"] == replay_seed
        assert consumer_daemon.post_merge_completion_recovery_task_cids() == (
            successor.task_cid,
        )

        (repo / "target-generation.txt").write_text(
            "completion recovery generation three\n",
            encoding="utf-8",
        )
        git("add", "target-generation.txt")
        git("commit", "-q", "-m", "advance completion target again")
        final_target = git("rev-parse", "HEAD")
        assert final_target != advanced_target

        consumer_bridge = fresh_consumer_bridge()
        consumer_bridges[0] = consumer_bridge
        final_recovery = consumer_bridge.recover_post_merge_declared_outputs(
            consumer_daemon
        )
        if final_recovery is None:
            final_recovery = (
                consumer_bridge.recover_post_merge_declared_outputs(
                    consumer_daemon
                )
            )
        assert final_recovery is not None
        assert final_recovery["recovered"] is True
        final_retrying = consumer_daemon.task_source.get(successor.task_cid)
        assert final_retrying is not None
        final_seed = final_retrying.body["completion_receipt"][
            "post_merge_completion_recovery_seed"
        ]
        assert final_seed["schema"] == (
            DATABASE_POST_MERGE_COMPLETION_RECOVERY_SEED_SCHEMA_V2
        )
        assert final_seed["source_task_revision"] == blocked_revision
        assert final_seed["recovery_control_revision"] == (
            blocked_revision + 10
        )
        assert final_seed["qualified_target_commit"] == final_target

        second_exhausted_claim = consumer_daemon.coordinator.get_task_claim(
            second_exhausted_attempt.claim_id
        )
        assert second_exhausted_claim is not None
        second_exhausted_now_ms = (
            int(second_exhausted_claim.expires_at_ms) + 1
        )
        consumer_daemon._clock_ms = lambda: second_exhausted_now_ms
        consumer_daemon.coordinator.expire_task_claim(
            second_exhausted_claim,
            now_ms=second_exhausted_now_ms,
        )
        second_exhausted_reconciliation = (
            consumer_daemon._reconcile_failed_attempt_coordination(
                second_exhausted_attempt
            )
        )
        assert second_exhausted_reconciliation["claim_state"] == "expired"
        successor = consumer_daemon.claim_next()
        assert successor is not None
        assert successor.body["post_merge_completion_recovery_seed"] == (
            final_seed
        )
        resumed = consumer_daemon.resume_attempt(successor)

        assert resumed["resumed"] is True
        assert resumed["status"] == "succeeded"
        assert resumed["committed_phase"] == ATTEMPT_PHASE_COMPLETE
        assert resumed["provider_result"]["accepted"] is True
        assert resumed["provider_result"]["baseline_commit"] == (
            baseline_commit
        )
        assert resumed["provider_result"]["implementation_commit"] == (
            candidate_commit
        )
        # Portal was instantiated once for fresh current-target validation;
        # the resumed zero-provider event projection did not instantiate it.
        assert consumer_portal_calls == ["VRIF-029", "VRIF-029"]
        assert requalification_heads == [advanced_target, final_target]
        events = consumer_bridge._verified_event_chain(
            consumer_bridge._paths(successor)
        )
        assert [event["type"] for event in events] == [
            "worktree_reconciliation_candidate_queued",
            "merge_reconciled",
            "task_completed",
        ]
        assert events[0]["attempt_consumed"] is False
        assert events[0]["provider_dispatched"] is False
        assert events[0]["implementation_commit"] == candidate_commit
        assert events[1]["merge_result"]["merged"] is True

        completed_task = consumer_daemon.task_source.get(successor.task_cid)
        assert completed_task is not None
        assert completed_task.status == "completed"
        completion_control = completed_task.body["completion_receipt"]
        assert completion_control["operation"] == "database_complete"
        assert completion_control["attempt_id"] == successor.attempt_id
        assert completion_control["claim_id"] == successor.claim_id
        assert completion_control["lease_id"] == successor.lease_id
        assert completed_task.revision == blocked_revision + 13
        queue_after_completion = queue.get(request.request_id)
        assert queue_after_completion is not None
        assert queue_after_completion.status == "completed"
        assert queue_after_completion.metadata["completion"] == (
            completed_request.metadata["completion"]
        )
    finally:
        if consumer_daemon is not None:
            consumer_daemon.close()
        source_daemon.close()


def _persist_legacy_empty_coordination_terminal(
    daemon: DatabaseImplementationDaemon,
    failed: DatabaseTaskAttempt,
    *,
    reason: str,
) -> None:
    task = daemon.task_source.get(failed.task_cid)
    assert task is not None
    assert task.status == "in_progress"
    claim_receipt = task.body["completion_receipt"]
    daemon._cas_task_status_database(
        failed.task_cid,
        expected_revision=int(task.revision),
        new_status="blocked",
        expected_control_receipt=claim_receipt,
        receipt={
            "operation": "database_portal_terminal_failure",
            "attempt_id": failed.attempt_id,
            "attempt_number": int(failed.attempt_number),
            "claim_id": failed.claim_id,
            "lease_id": failed.lease_id,
            "owner_session_id": failed.owner_session_id,
            "fencing_token": int(failed.fencing_token),
            "fence_epoch": int(failed.fence_epoch),
            "execution_phase": ATTEMPT_PHASE_FAILED,
            "execution_revision": int(failed.revision),
            "execution_finished_at_ms": failed.finished_at_ms,
            "reason": reason,
            "retryable": False,
            "coordination": {},
            "control_expected_status": "in_progress",
            "control_expected_revision": int(task.revision),
        },
    )


def test_terminal_coordination_projection_accepts_exact_producer_history(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}
    daemon = _open_daemon(
        tmp_path,
        session="session:terminal-coordination-producer-history",
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        failed = daemon.claim_next()
        assert failed is not None
        failed = daemon.commit_phase(failed, "context")
        failed = daemon.commit_phase(
            failed,
            "failed",
            body={
                "reason": "post_merge_declared_outputs_missing",
                "portal_retryable_failure": False,
                "portal_terminal_failure": True,
            },
        )

        accepted = daemon._reconcile_failed_attempt_coordination(failed)
        assert (
            daemon._terminal_coordination_projection_state(failed, accepted)
            == "accepted"
        )

        now["ms"] = 7_000
        expired_now = daemon._reconcile_failed_attempt_coordination(failed)
        assert expired_now["expired_now"] is True
        assert (
            daemon._terminal_coordination_projection_state(
                failed,
                expired_now,
            )
            == "expired"
        )
        historical = daemon._reconcile_failed_attempt_coordination(failed)
        assert historical["claim_absent"] is False
        assert historical["historical_expired"] is True
        assert historical["successor"] == {}
        assert (
            daemon._terminal_coordination_projection_state(
                failed,
                historical,
            )
            == "expired"
        )

        legacy_historical = dict(historical)
        legacy_historical.pop("claim_absent")
        assert (
            daemon._terminal_coordination_projection_state(
                failed,
                legacy_historical,
            )
            == "expired"
        )
        for tampered in (
            {**historical, "claim_absent": True},
            {**historical, "claim_absent": 0},
            {**historical, "unexpected": False},
            {**historical, "claim_id": "claim:foreign"},
        ):
            assert not daemon._terminal_coordination_projection_state(
                failed,
                tampered,
            )
    finally:
        daemon.close()


def test_ordinary_post_merge_recovery_uses_expired_portable_coordination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = {"ms": 1_000}
    daemon = _open_daemon(
        tmp_path,
        session="session:ordinary-portable-source",
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        failed = daemon.claim_next()
        assert failed is not None
        failed = daemon.commit_phase(failed, "context")
        failed = daemon.commit_phase(
            failed,
            "failed",
            body={
                "reason": (
                    DATABASE_PORTAL_COMPLETION_IMPLEMENTATION_COMMIT_MISSING_REASON
                ),
                "portal_retryable_failure": False,
                "portal_terminal_failure": True,
            },
        )

        now["ms"] = 7_000
        source_claim = daemon.coordinator.get_task_claim(failed.claim_id)
        assert source_claim is not None
        daemon.coordinator.expire_task_claim(source_claim, now_ms=now["ms"])
        historical = daemon._reconcile_failed_attempt_coordination(failed)
        assert historical["claim_absent"] is False
        assert (
            daemon._terminal_coordination_projection_state(
                failed,
                historical,
            )
            == "expired"
        )
        terminal = daemon._persist_terminal_portal_failure(
            failed,
            reason=(
                DATABASE_PORTAL_COMPLETION_IMPLEMENTATION_COMMIT_MISSING_REASON
            ),
            coordination_evidence=historical,
        )
        assert terminal["status"] == "blocked"
        evidence = _post_merge_repair_recovery_evidence(daemon, failed)
        preauthorization = _post_merge_preauthorization(
            daemon,
            failed,
            request_id=str(evidence["request_id"]),
            candidate_commit=str(evidence["candidate_commit"]),
        )
        expires_at_ms = int(historical["expires_at_ms"])

        daemon.close()
        now["ms"] = expires_at_ms - 1
        daemon = _open_daemon(
            tmp_path,
            session="session:ordinary-portable-replacement",
            lease_ms=5_000,
            clock_ms=lambda: now["ms"],
            coordination_path=(
                tmp_path / "coordination-ordinary-portable-replacement.duckdb"
            ),
            execution_path=tmp_path / "execution.duckdb",
        )
        assert daemon.coordinator.get_task_claim(failed.claim_id) is None
        assert daemon.coordinator.get_task_attempt(failed.attempt_id) is None
        assert daemon.coordinator.get_lease(failed.lease_id) is None

        with pytest.raises(
            DatabaseImplementationConflictError,
            match="current terminal coordination fence",
        ):
            daemon.preauthorize_post_merge_declared_output_recovery(
                preauthorization
            )

        now["ms"] = max(7_001, expires_at_ms + 1)
        authorized = daemon.preauthorize_post_merge_declared_output_recovery(
            preauthorization
        )
        assert authorized["authorized"] is True
        assert authorized["task_status"] == "blocked"

        blocked = daemon.task_source.get(failed.task_cid)
        assert blocked is not None
        control_receipt = dict(blocked.body["completion_receipt"])
        original_get = daemon.task_source.get

        def reject_forged_portable_receipt(
            forged_receipt: Mapping[str, object],
        ) -> None:
            forged_task = replace(
                blocked,
                body={
                    **blocked.body,
                    "completion_receipt": dict(forged_receipt),
                },
            )
            with monkeypatch.context() as receipt_patch:
                receipt_patch.setattr(
                    daemon.task_source,
                    "get",
                    lambda task_cid: (
                        forged_task
                        if task_cid == failed.task_cid
                        else original_get(task_cid)
                    ),
                )
                with pytest.raises(
                    DatabaseImplementationConflictError,
                    match="current terminal coordination fence",
                ):
                    daemon.preauthorize_post_merge_declared_output_recovery(
                        preauthorization
                    )

        legacy_historical = dict(historical)
        legacy_historical.pop("claim_absent")
        assert (
            daemon._terminal_coordination_projection_state(
                failed,
                legacy_historical,
            )
            == "expired"
        )
        reject_forged_portable_receipt(
            {**control_receipt, "coordination": legacy_historical}
        )
        reject_forged_portable_receipt(
            {
                **control_receipt,
                "reason": (
                    DATABASE_PROTECTED_PRESERVATION_TARGET_ANCESTRY_MISSING_REASON
                ),
            }
        )

        # The ordinary retry authority remains closed: only the exact
        # post-merge transition may use the portable sealed fence.
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="has no coordination claim",
        ):
            daemon._reconcile_failed_attempt_coordination(failed)

        original_portable_authority = (
            daemon._post_merge_completion_portable_coordination_authority
        )
        race_checks: list[bool] = []

        def supersede_before_cas(
            attempt: DatabaseTaskAttempt,
            *,
            persisted: Mapping[str, object],
        ) -> bool:
            race_checks.append(True)
            if len(race_checks) == 2:
                return False
            return original_portable_authority(
                attempt,
                persisted=persisted,
            )

        with monkeypatch.context() as race_patch:
            race_patch.setattr(
                daemon,
                "_post_merge_completion_portable_coordination_authority",
                supersede_before_cas,
            )
            with pytest.raises(
                DatabaseImplementationConflictError,
                match=(
                    "portable post-merge completion authority was superseded"
                ),
            ):
                daemon.recover_blocked_post_merge_declared_outputs(evidence)
        assert race_checks == [True, True]
        still_blocked = daemon.task_source.get(failed.task_cid)
        assert still_blocked is not None
        assert still_blocked.status == "blocked"

        portable_checks: list[bool] = []

        def count_portable_rechecks(
            attempt: DatabaseTaskAttempt,
            *,
            persisted: Mapping[str, object],
        ) -> bool:
            portable_checks.append(True)
            return original_portable_authority(
                attempt,
                persisted=persisted,
            )

        monkeypatch.setattr(
            daemon,
            "_post_merge_completion_portable_coordination_authority",
            count_portable_rechecks,
        )
        recovered = daemon.recover_blocked_post_merge_declared_outputs(
            evidence
        )
        assert portable_checks == [True, True]
        assert recovered["recovered"] is True
        assert recovered["changed"] is True
        assert recovered["status"] == "retrying"
        assert recovered["coordination"] == historical
        retrying = daemon.task_source.get(failed.task_cid)
        assert retrying is not None
        assert retrying.status == "retrying"
    finally:
        daemon.close()


def test_preauthorize_accepts_legacy_empty_coordination_only_after_exact_expiry(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}
    daemon = _open_daemon(
        tmp_path,
        session="session:post-merge-legacy-empty-coordination",
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        failed = daemon.claim_next()
        assert failed is not None
        failed = daemon.commit_phase(failed, "context")
        failed = daemon.commit_phase(
            failed,
            "failed",
            body={
                "reason": (
                    DATABASE_PROTECTED_PRESERVATION_TARGET_ANCESTRY_MISSING_REASON
                ),
                "portal_retryable_failure": False,
                "portal_terminal_failure": True,
            },
        )
        _persist_legacy_empty_coordination_terminal(
            daemon,
            failed,
            reason=(
                DATABASE_PROTECTED_PRESERVATION_TARGET_ANCESTRY_MISSING_REASON
            ),
        )
        blocked = daemon.task_source.get(failed.task_cid)
        assert blocked is not None
        assert blocked.body["completion_receipt"]["coordination"] == {}

        # Empty coordination is a legacy persistence shape, not evidence.
        # Preauthorization may recover it only after the coordinator itself
        # independently reproduces the exact terminal claim/attempt/lease.
        with pytest.raises(
            DatabaseImplementationConflictError,
            match="coordination",
        ):
            daemon.preauthorize_post_merge_declared_output_recovery(
                _post_merge_preauthorization(daemon, failed)
            )

        now["ms"] = 7_000
        source_claim = daemon.coordinator.get_task_claim(failed.claim_id)
        assert source_claim is not None
        daemon.coordinator.expire_task_claim(
            source_claim,
            now_ms=now["ms"],
        )
        reproduced = daemon._reconcile_failed_attempt_coordination(failed)
        assert reproduced["claim_id"] == failed.claim_id
        assert reproduced["attempt_id"] == failed.attempt_id
        assert reproduced["attempt_number"] == failed.attempt_number
        assert reproduced["claim_state"] == "expired"
        assert reproduced["lease_state"] == "expired"
        assert reproduced["historical_expired"] is True
        assert reproduced["expired_now"] is False
        assert reproduced["superseded_by_newer_fence"] is False

        authorized = daemon.preauthorize_post_merge_declared_output_recovery(
            _post_merge_preauthorization(daemon, failed)
        )
        assert authorized["authorized"] is True
        assert authorized["task_status"] == "blocked"
    finally:
        daemon.close()


def test_preauthorize_rejects_legacy_empty_coordination_with_newer_fence(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}
    daemon = _open_daemon(
        tmp_path,
        session="session:post-merge-forged-empty-coordination",
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        failed = daemon.claim_next()
        assert failed is not None
        failed = daemon.commit_phase(failed, "context")
        failed = daemon.commit_phase(
            failed,
            "failed",
            body={
                "reason": "post_merge_declared_outputs_missing",
                "portal_retryable_failure": False,
                "portal_terminal_failure": True,
            },
        )
        _persist_legacy_empty_coordination_terminal(
            daemon,
            failed,
            reason="post_merge_declared_outputs_missing",
        )
        blocked = daemon.task_source.get(failed.task_cid)
        assert blocked is not None
        assert blocked.body["completion_receipt"]["coordination"] == {}

        now["ms"] = 7_000
        source_claim = daemon.coordinator.get_task_claim(failed.claim_id)
        assert source_claim is not None
        daemon.coordinator.expire_task_claim(
            source_claim,
            now_ms=now["ms"],
        )
        successor = daemon.coordinator.claim_ready_task(
            owner_session_id="session:post-merge-newer-fence",
            lease_ms=5_000,
            now_ms=now["ms"],
        )
        assert successor is not None
        assert successor.fencing_token > failed.fencing_token

        with pytest.raises(
            DatabaseImplementationConflictError,
            match="coordination|newer|superseded",
        ):
            daemon.preauthorize_post_merge_declared_output_recovery(
                _post_merge_preauthorization(daemon, failed)
            )
    finally:
        daemon.close()


def test_preauthorize_accepts_populated_coordination_after_exact_expiry(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}
    daemon = _open_daemon(
        tmp_path,
        session="session:post-merge-populated-expired",
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        failed = daemon.claim_next()
        assert failed is not None
        failed = daemon.commit_phase(failed, "context")
        failed = daemon.commit_phase(
            failed,
            "failed",
            body={
                "reason": "post_merge_declared_outputs_missing",
                "portal_retryable_failure": False,
                "portal_terminal_failure": True,
            },
        )
        daemon._persist_terminal_portal_failure(
            failed,
            reason="post_merge_declared_outputs_missing",
            coordination_evidence=(
                daemon._reconcile_failed_attempt_coordination(failed)
            ),
        )
        blocked = daemon.task_source.get(failed.task_cid)
        assert blocked is not None
        persisted = blocked.body["completion_receipt"]["coordination"]
        assert persisted["claim_state"] == "accepted"
        assert persisted["lease_state"] == "accepted"
        assert persisted["coordination_attempt_status"] == "running"

        now["ms"] = 7_000
        source_claim = daemon.coordinator.get_task_claim(failed.claim_id)
        assert source_claim is not None
        daemon.coordinator.expire_task_claim(source_claim, now_ms=now["ms"])
        expired_claim = daemon.coordinator.get_task_claim(failed.claim_id)
        expired_attempt = daemon.coordinator.get_task_attempt(failed.attempt_id)
        expired_lease = daemon.coordinator.get_lease(failed.lease_id)
        assert expired_claim is not None
        assert expired_attempt is not None
        assert expired_lease is not None
        before = (
            expired_claim.to_dict(),
            expired_attempt.to_dict(),
            expired_lease.to_dict(),
        )

        authorized = daemon.preauthorize_post_merge_declared_output_recovery(
            _post_merge_preauthorization(daemon, failed)
        )
        assert authorized["authorized"] is True
        assert authorized["task_status"] == "blocked"
        assert (
            daemon.coordinator.get_task_claim(failed.claim_id).to_dict()
            == before[0]
        )
        assert (
            daemon.coordinator.get_task_attempt(failed.attempt_id).to_dict()
            == before[1]
        )
        assert (
            daemon.coordinator.get_lease(failed.lease_id).to_dict()
            == before[2]
        )
    finally:
        daemon.close()


def test_preauthorize_rejects_populated_coordination_with_newer_fence(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}
    daemon = _open_daemon(
        tmp_path,
        session="session:post-merge-populated-newer-fence",
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        failed = daemon.claim_next()
        assert failed is not None
        failed = daemon.commit_phase(failed, "context")
        failed = daemon.commit_phase(
            failed,
            "failed",
            body={
                "reason": "post_merge_declared_outputs_missing",
                "portal_retryable_failure": False,
                "portal_terminal_failure": True,
            },
        )
        daemon._persist_terminal_portal_failure(
            failed,
            reason="post_merge_declared_outputs_missing",
            coordination_evidence=(
                daemon._reconcile_failed_attempt_coordination(failed)
            ),
        )
        blocked = daemon.task_source.get(failed.task_cid)
        assert blocked is not None
        assert blocked.body["completion_receipt"]["coordination"][
            "claim_state"
        ] == "accepted"

        now["ms"] = 7_000
        source_claim = daemon.coordinator.get_task_claim(failed.claim_id)
        assert source_claim is not None
        daemon.coordinator.expire_task_claim(source_claim, now_ms=now["ms"])
        successor = daemon.coordinator.claim_ready_task(
            owner_session_id="session:post-merge-populated-successor",
            lease_ms=5_000,
            now_ms=now["ms"],
        )
        assert successor is not None
        assert successor.fencing_token > failed.fencing_token

        with pytest.raises(
            DatabaseImplementationConflictError,
            match="coordination|fence|newer|superseded",
        ):
            daemon.preauthorize_post_merge_declared_output_recovery(
                _post_merge_preauthorization(daemon, failed)
            )
    finally:
        daemon.close()


def test_preauthorize_does_not_expire_overdue_legacy_empty_coordination(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}
    daemon = _open_daemon(
        tmp_path,
        session="session:post-merge-legacy-empty-overdue",
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        failed = daemon.claim_next()
        assert failed is not None
        failed = daemon.commit_phase(failed, "context")
        failed = daemon.commit_phase(
            failed,
            "failed",
            body={
                "reason": "post_merge_declared_outputs_missing",
                "portal_retryable_failure": False,
                "portal_terminal_failure": True,
            },
        )
        _persist_legacy_empty_coordination_terminal(
            daemon,
            failed,
            reason="post_merge_declared_outputs_missing",
        )
        blocked = daemon.task_source.get(failed.task_cid)
        assert blocked is not None
        assert blocked.body["completion_receipt"]["coordination"] == {}
        source_claim = daemon.coordinator.get_task_claim(failed.claim_id)
        source_attempt = daemon.coordinator.get_task_attempt(failed.attempt_id)
        source_lease = daemon.coordinator.get_lease(failed.lease_id)
        assert source_claim is not None
        assert source_attempt is not None
        assert source_lease is not None
        before = (
            source_claim.to_dict(),
            source_attempt.to_dict(),
            source_lease.to_dict(),
        )
        event_count = len(
            daemon.coordinator.lease_events(
                lease_id=failed.lease_id,
                limit=10_000,
            )
        )
        assert before[0]["state"] == "accepted"
        assert before[1]["status"] == "running"
        assert before[2]["state"] == "accepted"

        now["ms"] = 7_000
        with pytest.raises(
            DatabaseImplementationConflictError,
            match="coordination|fence",
        ):
            daemon.preauthorize_post_merge_declared_output_recovery(
                _post_merge_preauthorization(daemon, failed)
            )

        current_claim = daemon.coordinator.get_task_claim(failed.claim_id)
        current_attempt = daemon.coordinator.get_task_attempt(failed.attempt_id)
        current_lease = daemon.coordinator.get_lease(failed.lease_id)
        assert current_claim is not None
        assert current_attempt is not None
        assert current_lease is not None
        assert current_claim.to_dict() == before[0]
        assert current_attempt.to_dict() == before[1]
        assert current_lease.to_dict() == before[2]
        assert len(
            daemon.coordinator.lease_events(
                lease_id=failed.lease_id,
                limit=10_000,
            )
        ) == event_count
        assert (
            daemon.coordinator.get_task_claim_successor_projection(
                task_cid=failed.task_cid,
                after_fencing_token=failed.fencing_token,
                after_fence_epoch=failed.fence_epoch,
            )
            is None
        )
    finally:
        daemon.close()


def test_preauthorize_requires_successor_projection_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:post-merge-successor-reader-unavailable",
    )
    try:
        daemon.materialize_population(_population(1))
        failed = daemon.claim_next()
        assert failed is not None
        failed = daemon.commit_phase(failed, "context")
        failed = daemon.commit_phase(
            failed,
            "failed",
            body={
                "reason": "post_merge_declared_outputs_missing",
                "portal_retryable_failure": False,
                "portal_terminal_failure": True,
            },
        )
        daemon._persist_terminal_portal_failure(
            failed,
            reason="post_merge_declared_outputs_missing",
            coordination_evidence=(
                daemon._reconcile_failed_attempt_coordination(failed)
            ),
        )
        monkeypatch.setattr(
            daemon.coordinator,
            "get_task_claim_successor_projection",
            None,
        )

        with pytest.raises(
            DatabaseImplementationConflictError,
            match="coordination|fence",
        ):
            daemon.preauthorize_post_merge_declared_output_recovery(
                _post_merge_preauthorization(daemon, failed)
            )
    finally:
        daemon.close()


@pytest.mark.parametrize(
    "terminal_reason",
    (
        "cross_board_manual_completion_authority_metadata_invalid",
        DATABASE_PROTECTED_PRESERVATION_TARGET_ANCESTRY_MISSING_REASON,
    ),
)
def test_preauthorize_accepts_recoverable_completion_terminal(
    tmp_path: Path,
    terminal_reason: str,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:post-merge-cross-board-terminal",
    )
    try:
        daemon.materialize_population(_population(1))
        failed = daemon.claim_next()
        assert failed is not None
        failed = daemon.commit_phase(failed, "context")
        failed = daemon.commit_phase(
            failed,
            "failed",
            body={
                "reason": terminal_reason,
                "portal_retryable_failure": False,
                "portal_terminal_failure": True,
            },
        )
        terminal = daemon._persist_terminal_portal_failure(
            failed,
            reason=terminal_reason,
            coordination_evidence=(
                daemon._reconcile_failed_attempt_coordination(failed)
            ),
        )
        assert terminal["status"] == "blocked"
        authorized = daemon.preauthorize_post_merge_declared_output_recovery(
            _post_merge_preauthorization(daemon, failed)
        )
        assert authorized["authorized"] is True
        assert authorized["task_status"] == "blocked"
        stale = _post_merge_preauthorization(daemon, failed)
        stale["source_attempt_id"] = "attempt:prior-repair"
        if terminal_reason in DATABASE_PORTAL_CROSS_BOARD_COMPLETION_REASONS:
            authorized_prior = (
                daemon.preauthorize_post_merge_declared_output_recovery(stale)
            )
            assert authorized_prior["authorized"] is True
        else:
            with pytest.raises(
                DatabaseImplementationConflictError,
                match="superseded source attempt",
            ):
                daemon.preauthorize_post_merge_declared_output_recovery(stale)
    finally:
        daemon.close()


def test_protected_preservation_ancestry_reason_requires_exact_token() -> None:
    forged = (
        "prefix:"
        f"{DATABASE_PROTECTED_PRESERVATION_TARGET_ANCESTRY_MISSING_REASON}"
        ":suffix"
    )

    assert (
        DatabaseImplementationDaemon._canonical_portal_failure_reason(forged)
        == forged
    )
    assert (
        DatabaseImplementationDaemon._recoverable_post_merge_terminal_reason(
            forged
        )
        == ""
    )
def test_descendant_requalification_recovery_replays_one_queue_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:post-merge-requalification-rearm",
    )
    try:
        daemon.materialize_population(_population(1))
        failed = daemon.claim_next()
        assert failed is not None
        failed = daemon.commit_phase(failed, "context")
        failed = daemon.commit_phase(
            failed,
            "failed",
            body={
                "reason": "post_merge_declared_outputs_missing",
                "portal_retryable_failure": False,
                "portal_terminal_failure": True,
            },
        )
        terminal = daemon._persist_terminal_portal_failure(
            failed,
            reason="post_merge_declared_outputs_missing",
            coordination_evidence=(
                daemon._reconcile_failed_attempt_coordination(failed)
            ),
        )
        assert terminal["status"] == "blocked"

        entry = {
            "path": "output.py",
            "mode": "100644",
            "object_type": "blob",
            "object_id": "6" * 40,
        }
        source_validation = {
            "task_id": failed.task_alias,
            "passed": True,
            "returncode": 0,
            "validation_result_digests": ["sha256:" + "7" * 64],
            "command_count": 1,
            "log_sha256": "8" * 64,
        }
        source_repair: dict[str, object] = {
            "schema": POST_MERGE_DECLARED_OUTPUT_REPAIR_SCHEMA,
            "task_ids": [failed.task_alias],
            "candidate_commit": "a" * 40,
            "candidate_tree": "1" * 40,
            "baseline_commit": "2" * 40,
            "failed_integration_commit": "3" * 40,
            "repair_parent_commit": "4" * 40,
            "repair_commit": "b" * 40,
            "repair_tree": "5" * 40,
            "entries": [entry],
            "validation": [source_validation],
            "rollback_target": "4" * 40,
        }
        source_repair["receipt_id"] = content_identity(source_repair)
        requalification_validation = {
            **source_validation,
            "validation_result_digests": ["sha256:" + "9" * 64],
            "log_sha256": "a" * 64,
        }
        requalification: dict[str, object] = {
            "schema": POST_MERGE_DECLARED_OUTPUT_REQUALIFICATION_SCHEMA,
            "task_ids": [failed.task_alias],
            "candidate_commit": "a" * 40,
            "source_repair_receipt_id": source_repair["receipt_id"],
            "source_repair_commit": "b" * 40,
            "source_repair_receipt": source_repair,
            "current_target_commit": "c" * 40,
            "current_target_tree": "d" * 40,
            "entries": [entry],
            "validation": [requalification_validation],
        }
        requalification["receipt_id"] = content_identity(requalification)
        evidence: dict[str, object] = {
            "schema": DATABASE_POST_MERGE_REQUALIFICATION_RECOVERY_SCHEMA,
            "request_id": "merge-request:requalified",
            "task_cid": failed.task_cid,
            "task_alias": failed.task_alias,
            "candidate_commit": "a" * 40,
            "qualified_target_commit": "c" * 40,
            "requalification_receipt_id": requalification["receipt_id"],
            "requalification_receipt": requalification,
            "source_attempt_id": failed.attempt_id,
            "source_claim_id": failed.claim_id,
            "source_lease_id": failed.lease_id,
            "source_fencing_token": failed.fencing_token,
            "source_fence_epoch": failed.fence_epoch,
            "source_binding_id": "sha256:" + "e" * 64,
            "source_projection_immutable_digest": "sha256:" + "f" * 64,
        }
        evidence["evidence_id"] = daemon._database_portal_evidence_digest(
            evidence
        )

        original_guarded_recovery = (
            daemon.task_source.record_queue_backoff_and_cas_status
        )

        def lose_cas_response(*args: object, **kwargs: object) -> None:
            original_guarded_recovery(*args, **kwargs)
            raise RuntimeError("requalification CAS response lost")

        monkeypatch.setattr(
            daemon.task_source,
            "record_queue_backoff_and_cas_status",
            lose_cas_response,
        )
        with pytest.raises(RuntimeError, match="CAS response lost"):
            daemon.recover_blocked_post_merge_declared_outputs(evidence)
        first_queue = daemon.task_source.get_queue_entry(failed.task_cid)
        assert first_queue is not None

        daemon.close()
        daemon = _open_daemon(
            tmp_path,
            session="session:post-merge-requalification-rearm",
        )
        recovered = daemon.recover_blocked_post_merge_declared_outputs(
            evidence
        )
        assert recovered["recovered"] is True
        assert recovered["changed"] is False
        assert recovered["write_count"] == 0
        assert recovered["qualification_kind"] == "requalification"
        second_queue = daemon.task_source.get_queue_entry(failed.task_cid)
        assert second_queue is not None
        assert second_queue.attempt == first_queue.attempt
        retrying = daemon.task_source.get(failed.task_cid)
        assert retrying is not None and retrying.status == "retrying"
        control = retrying.body["completion_receipt"]
        assert control["operation"] == (
            "database_post_merge_declared_outputs_requalification_recovery"
        )
        assert control["source_repair_receipt_id"] == source_repair[
            "receipt_id"
        ]
        assert control["requalification_receipt_id"] == requalification[
            "receipt_id"
        ]

        daemon.close()
        daemon = _open_daemon(
            tmp_path,
            session="session:post-merge-requalification-rearm",
        )
        repeated = daemon.recover_blocked_post_merge_declared_outputs(
            evidence
        )
        assert repeated["changed"] is False
        assert repeated["write_count"] == 0
        assert daemon.reconcile_terminal_portal_failures() == []
    finally:
        daemon.close()
