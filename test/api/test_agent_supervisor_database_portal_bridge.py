"""Focused safety tests for database-authoritative Portal execution."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
    checkout_repository_id,
)
from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import MergeQueue
from ipfs_accelerate_py.agent_supervisor.merge.merge_train import MergeTrain
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.runtime.event_log import append_jsonl_event
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DATABASE_PORTAL_CAPACITY_RETRY_SCHEMA,
    DATABASE_PORTAL_CONSUMED_ATTEMPT_RETRY_SCHEMA,
    DATABASE_PORTAL_CONSUMED_ATTEMPT_RETRY_SEED_SCHEMA,
    DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA,
    DATABASE_PORTAL_PROTECTED_PATH_PRESERVATION_SCHEMA,
    DATABASE_PORTAL_VALIDATION_RETRY_SCHEMA,
    DatabasePortalBridgeDeferred,
    DatabasePortalBridgeError,
    DatabasePortalCapacityRetry,
    DatabasePortalConsumedAttemptTerminal,
    DatabasePortalExecutionBridge,
    DatabasePortalProtectedPathPreserved,
    DatabasePortalValidationRetry,
    _is_implementation_conflict,
    verify_database_portal_attempt_projection,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DATASETS_AUTHORITATIVE_STATE_SCHEMA_REVISION,
    EXTERNAL_PROTECTED_RECOVERY_BACKOFF_SECONDS,
    SEMANTIC_TRUTH_AUTHORITY_ENV,
    SEMANTIC_WRITER_POLICY_ENV,
    DatabaseImplementationAuthorityError,
    DatabaseImplementationConflictError,
    DatabaseImplementationDaemon,
    DatabaseTaskAttempt,
    PortalImplementationDaemon,
    PortalTaskState,
    parse_args,
    parse_task_text,
    task_declared_output_paths,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
    build_portal_implementation_daemon_from_args,
)
from ipfs_accelerate_py.agent_supervisor.validation.project_dependency_preflight import (
    preflight_validation_project_dependencies,
)
from ipfs_accelerate_py.agent_supervisor.validation.validation_commands import (
    validation_command_repository_root,
)


def _attempt(*, attempt_number: int = 1) -> DatabaseTaskAttempt:
    return DatabaseTaskAttempt(
        attempt_id="attempt:001",
        claim_id="claim:001",
        task_cid="task:cid:004",
        task_alias="LGSWF-004",
        attempt_number=attempt_number,
        owner_session_id="session:bridge",
        fencing_token=7,
        fence_epoch=3,
        lease_id="lease:001",
        committed_phase="claimed",
        status="running",
        started_at_ms=1,
    )


def test_implementation_conflict_matches_main_module_alias() -> None:
    class DatabaseImplementationConflictError(RuntimeError):
        pass

    assert _is_implementation_conflict(
        DatabaseImplementationConflictError("stale row")
    )
    assert not _is_implementation_conflict(RuntimeError("stale row"))
    assert _is_implementation_conflict(
        DatabaseImplementationConflictError("no longer matches")
    )


def _record() -> SimpleNamespace:
    return SimpleNamespace(
        task_cid="task:cid:004",
        task_alias="LGSWF-004",
        goal_cid="goal:inventory",
        plan_cid="plan:lgswf:1",
        revision=11,
        priority="P0",
        dependencies=("task:cid:003",),
        outputs=({"path": "inventory/result.json"},),
        validations=({"argv": ["python3", "-m", "pytest", "focused.py"]},),
        acceptance=({"criterion": "Focused validation passes"},),
        body={
            "objective": "Produce the current authority inventory",
            "completion": "auto",
            "track": "analysis",
            "read_scope": ["ipfs_accelerate_py/agent_supervisor"],
            "write_scope": ["inventory/result.json"],
            "completion_contract": "Focused validation passes",
        },
    )


def _owned_record(owner: str) -> SimpleNamespace:
    record = _record()
    record.outputs = (
        {"path": "ipfs_datasets_py/logic/verification_api.py"},
        {
            "path": (
                "tests/unit/logic/"
                "test_compositional_verification_public_api.py"
            )
        },
    )
    record.validations = (
        {
            "argv": [
                "python -m pytest -q "
                "tests/unit/logic/test_compositional_verification_public_api.py"
            ]
        },
    )
    record.body = {
        **record.body,
        "owning_repository": owner,
        "markdown_metadata": {"owning_repository": owner},
    }
    return record


def test_datasets_authority_marker_reaches_provider_without_state_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION",
        DATASETS_AUTHORITATIVE_STATE_SCHEMA_REVISION,
    )
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_DATABASE_PROGRAM_JSON",
        '{"credential":"must-not-propagate"}',
    )
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "secret-token")
    portal = SimpleNamespace(_canonical_ref=lambda task: "task:cid:004")
    task = SimpleNamespace(task_id="LGSWF-004")

    environment = PortalImplementationDaemon._implementation_process_environment(
        portal,
        task,
        attempt=2,
        checkpoint_dir=tmp_path / "checkpoint",
    )

    assert environment[SEMANTIC_TRUTH_AUTHORITY_ENV] == "ipfs_datasets_py"
    assert environment[SEMANTIC_WRITER_POLICY_ENV] == "reference_only"
    assert "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION" not in environment
    assert "IPFS_ACCELERATE_AGENT_DATABASE_PROGRAM_JSON" not in environment
    assert "IPFS_ACCELERATE_AGENT_QUACK_TOKEN" not in environment

    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION", "schema-v1")
    ordinary_environment = (
        PortalImplementationDaemon._implementation_process_environment(
            portal,
            task,
            attempt=3,
            checkpoint_dir=tmp_path / "ordinary-checkpoint",
        )
    )
    assert SEMANTIC_TRUTH_AUTHORITY_ENV not in ordinary_environment
    assert SEMANTIC_WRITER_POLICY_ENV not in ordinary_environment


class _TaskSource:
    def __init__(self, record: object) -> None:
        self.record = record

    def get_task(self, task_cid: str) -> object | None:
        return self.record if task_cid == "task:cid:004" else None


class _CompletingPortal:
    def __init__(self, paths: object, task_alias: str) -> None:
        self.paths = paths
        self.task_alias = task_alias
        self.closed = False

    def run_once(self) -> dict[str, object]:
        text = self.paths.task_projection.read_text(encoding="utf-8")
        self.paths.task_projection.write_text(
            text.replace("- Status: ready", "- Status: completed"),
            encoding="utf-8",
        )
        self.paths.state.write_text(
            json.dumps(
                {
                    "last_implementation_commit": "a" * 40,
                    "last_merge_returncode": 0,
                }
            ),
            encoding="utf-8",
        )
        self.paths.events.write_text(
            json.dumps(
                {
                    "type": "task_completed",
                    "task_id": self.task_alias,
                    "event_id": "event:complete",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "task_count": 1,
            "completed_count": 1,
            "active_task_id": self.task_alias,
            "implementation_result": {
                "task_id": self.task_alias,
                "returncode": 0,
                "implementation_commit": "a" * 40,
                # Raw model output must not enter the database receipt.
                "model_response": "private provider payload",
            },
            "merge_reconciliation": [
                {
                    "task_id": self.task_alias,
                    "returncode": 0,
                    "merge_commit": "b" * 40,
                    "provider_payload": "private",
                }
            ],
        }

    def close_event_runtime(self) -> None:
        self.closed = True


def _git_candidate_with_rescue_branch(repo: Path) -> tuple[str, str]:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "portal-test@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Portal Test"],
        cwd=repo,
        check=True,
    )
    output = repo / "inventory" / "result.json"
    output.parent.mkdir(parents=True)
    output.write_text('{"candidate":true}\n', encoding="utf-8")
    subprocess.run(["git", "add", "--", str(output.relative_to(repo))], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "candidate"], cwd=repo, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    rescue_branch = "rescue/lgswf-004-attempt-1-failed-validation"
    subprocess.run(
        ["git", "branch", rescue_branch, commit],
        cwd=repo,
        check=True,
    )
    return commit, rescue_branch


class _ValidationFailurePortal:
    def __init__(
        self,
        paths: object,
        task_alias: str,
        *,
        commit: str,
        rescue_branch: str,
        denied_paths: tuple[str, ...] = (),
    ) -> None:
        self.paths = paths
        self.task_alias = task_alias
        self.commit = commit
        self.rescue_branch = rescue_branch
        self.denied_paths = denied_paths

    def run_once(self) -> dict[str, object]:
        changed_paths = ["inventory/result.json"]
        proposal_id = "proposal:validation-retry"
        proposal_receipt_id = "proposal-receipt:validation-retry"
        proposal_policy_id = "proposal-policy:validation-retry"
        proposal_gate = {
            "attempted": True,
            "accepted": True,
            "reason_codes": [],
            "proposal_id": proposal_id,
            "receipt_id": proposal_receipt_id,
            "policy_id": proposal_policy_id,
            "changed_paths": changed_paths,
        }
        review = {
            "decision": "guide_rescue",
            "reason_codes": ["validation_command_failed"],
            "denied_paths": list(self.denied_paths),
            "out_of_scope_paths": [],
            "contract_gap_paths": [],
            "missing_expected_outputs": [],
            "justified_paths": [],
            "receipt_id": "failure-review:validation-retry",
        }
        dag = {
            "receipt_id": "validation-dag:validation-retry",
            "proposal_receipt_id": proposal_receipt_id,
            "objective_id": "task:cid:004",
            "changed_paths": changed_paths,
            "passed": False,
            "coverage_complete": True,
            "uncovered_impact": False,
            "nodes": [
                {
                    "mandatory": True,
                    "selected": True,
                    "disposition": "failed",
                    "returncode": 1,
                    "result_digest": "validation-result:failed",
                }
            ],
        }
        validation = {
            "attempted": True,
            "passed": False,
            "returncode": 1,
            "reason": "declared_validation_failed",
            "auto_rescue_terminal": True,
            "completion_authoritative": False,
            "merge_eligible": False,
            "coverage_errors": [],
            "proposal_gate": proposal_gate,
            "failure_review": review,
            "validation_dag_receipt": dag,
        }
        preservation = {
            "task_id": self.task_alias,
            "attempt": 1,
            "implementation_commit": self.commit,
            "preserved_commit": self.commit,
            "preserved": True,
            "rescue_branch": self.rescue_branch,
            "commit_result": {
                "committed": True,
                "commit": self.commit,
            },
        }
        common = {
            "task_id": self.task_alias,
            "canonical_task_cid": "task:cid:004",
        }
        append_jsonl_event(
            self.paths.events,
            "implementation_expected_outputs_checked",
            {
                **common,
                "proposal_id": proposal_id,
                "passed": True,
                "issues": [],
                "expected_paths": changed_paths,
                "staged_paths": changed_paths,
                "force_staged_paths": [],
            },
        )
        append_jsonl_event(
            self.paths.events,
            "implementation_proposal_validated",
            {
                **common,
                **proposal_gate,
            },
        )
        append_jsonl_event(
            self.paths.events,
            "failed_validation_worktree_preserved",
            {
                **common,
                **preservation,
                "validation_result": validation,
            },
        )
        implementation = {
            **common,
            "attempt": 1,
            "returncode": 1,
            "attempt_consumed": True,
            "provider_dispatched": True,
            "implementation_commit": self.commit,
            "branch": "implementation/lgswf-004-attempt-1",
            "merge_result": {"merged": False, "reason": "not_attempted"},
            "board_completion": {
                "complete": False,
                "pending_merge": False,
                "reason": "implementation_or_validation_failed",
            },
            "validation_result": validation,
            "failed_preservation_result": preservation,
        }
        append_jsonl_event(
            self.paths.events,
            "implementation_finished",
            implementation,
        )
        return {"implementation_result": implementation}


def _capacity_record_id(value: dict[str, object], field: str) -> str:
    body = {key: item for key, item in value.items() if key != field}
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _capacity_event_payload(
    task_alias: str,
    *,
    portal_attempt: int = 1,
) -> dict[str, object]:
    task_cid = "task:cid:004"
    logical_attempt_id = "sha256:" + "1" * 64
    invocation_binding_id = "sha256:" + "2" * 64
    decision_id = "sha256:" + "3" * 64
    route_id = "route:test"
    returncode = 17
    observed_at_ms = 1_000_000
    retry_not_before_ms = 2_000_000
    primary: dict[str, object] = {
        "schema": "fixture/grok-failure@1",
        "nonce": "4" * 64,
    }
    primary["receipt_id"] = _capacity_record_id(primary, "receipt_id")
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
        "invocation_binding_id": invocation_binding_id,
        "logical_attempt_id": logical_attempt_id,
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
    capacity["receipt_id"] = _capacity_record_id(capacity, "receipt_id")
    outcome: dict[str, object] = {
        "route_plan": {
            "route_id": route_id,
            "fallback_provider_id": "codex",
            "fallback_model_id": "gpt-5.6-terra",
            "fallback_reasoning_effort": "high",
        },
        "preflight_receipt_id": primary["receipt_id"],
        "invocation_binding_id": invocation_binding_id,
        "decision": "fallback_failed",
        "decision_id": decision_id,
        "fallback_dispatched": True,
        "fallback_returncode": returncode,
        "fallback_capacity_receipt": capacity,
    }
    outcome["outcome_id"] = _capacity_record_id(outcome, "outcome_id")
    proof: dict[str, object] = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "post-dispatch-capacity-retry-proof@1"
        ),
        "task_id": task_alias,
        "attempt": portal_attempt,
        "task_revision_cid": task_cid,
        "logical_attempt_id": logical_attempt_id,
        "invocation_binding_id": invocation_binding_id,
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
    proof["proof_id"] = _capacity_record_id(proof, "proof_id")
    return {
        "task_id": task_alias,
        "canonical_task_cid": task_cid,
        "attempt": portal_attempt,
        "returncode": returncode,
        "retryable": True,
        "deferred": False,
        "attempt_consumed": True,
        "provider_dispatched": True,
        "typed_deferral_slot_consumed": False,
        "reason": "provider_capacity_exhausted",
        "failure_class": "dual_provider_capacity_exhausted",
        "providers": ["grok", "codex"],
        "post_dispatch_capacity_retry": proof,
        "quota_probe_receipt": primary,
        "route_outcome": outcome,
        "codex_capacity_receipt": capacity,
    }


class _CapacityFailurePortal:
    def __init__(
        self,
        paths: object,
        task_alias: str,
        *,
        calls: list[int],
        portal_attempt: int = 1,
    ) -> None:
        self.paths = paths
        self.task_alias = task_alias
        self.calls = calls
        self.portal_attempt = portal_attempt

    def run_once(self) -> dict[str, object]:
        self.calls.append(self.portal_attempt)
        implementation = _capacity_event_payload(
            self.task_alias,
            portal_attempt=self.portal_attempt,
        )
        append_jsonl_event(
            self.paths.events,
            "implementation_post_dispatch_capacity_retry",
            implementation,
        )
        append_jsonl_event(
            self.paths.events,
            "daemon_pass",
            {"active_task_id": self.task_alias},
        )
        return {"implementation_result": implementation}


def _write_consumed_attempt_failure(
    paths: object,
    task_alias: str,
    *,
    portal_attempt: int = 1,
    max_task_attempts: int = 4,
    finish_updates: dict[str, object] | None = None,
    before_finish_event: str = "",
) -> tuple[dict[str, object], dict[str, object]]:
    baseline_commit = "b" * 40
    branch = f"implementation/lgswf-004-attempt-{portal_attempt}"
    canonical_task_key = "task/v1/closed-consumed-attempt"
    board_namespace = "task-projection.md"
    workspace_path = "/tmp/closed-consumed-attempt-worktree"
    log_path = "/tmp/closed-consumed-attempt.log"
    workspace_setup = {
        "base_commit": baseline_commit,
        "branch": branch,
        "worktree_path": workspace_path,
    }
    append_jsonl_event(
        paths.events,
        "task_selected",
        {
            "board_namespace": board_namespace,
            "canonical_task_cid": "task:cid:004",
            "canonical_task_key": canonical_task_key,
            "task_id": task_alias,
            "title": "Closed consumed-attempt replay fixture",
            "track": "implementation",
        },
    )
    append_jsonl_event(
        paths.events,
        "implementation_protected_path_snapshot_recorded",
        {
            "attempt": portal_attempt,
            "board_namespace": board_namespace,
            "canonical_task_cid": "task:cid:004",
            "canonical_task_key": canonical_task_key,
            "protected_paths": [],
            "task_id": task_alias,
            "workspace_path": workspace_path,
        },
    )
    started = append_jsonl_event(
        paths.events,
        "implementation_started",
        {
            "task_id": task_alias,
            "canonical_task_cid": "task:cid:004",
            "canonical_task_key": canonical_task_key,
            "board_namespace": board_namespace,
            "attempt": portal_attempt,
            "branch": branch,
            "baseline_ref": baseline_commit,
            "provider_dispatched": False,
            "cache_hit": False,
            "checkpoint_directory": "/tmp/closed-consumed-checkpoint",
            "command": ["provider"],
            "execution_mode": "model-assisted",
            "log_path": log_path,
            "outputs": ["inventory/result.json"],
            "saved_duration_seconds": 0.0,
            "setup_duration_seconds": 1.0,
            "timeout_policy": {"source": "test"},
            "workspace_setup": workspace_setup,
            "worktree_lifecycle": {"state": "active"},
            "worktree_path": workspace_path,
        },
    )
    append_jsonl_event(
        paths.events,
        "pre_implementation_kernel_evaluated",
        {
            "analytical_candidate_count": 0,
            "attempt": portal_attempt,
            "board_namespace": board_namespace,
            "canonical_task_cid": "task:cid:004",
            "canonical_task_key": canonical_task_key,
            "disposition": "abstain_review",
            "event": "pre_implementation_kernel_evaluated",
            "interface": "ImplementationDaemon@pre_implementation_kernel",
            "kernel_receipt": {"schema": "closed-test-kernel@1"},
            "provider_authorized": False,
            "provider_hook_count": 0,
            "reason_code": "no_analytical_close",
            "receipt_cid": "bagu-test-kernel-receipt",
            "residual_packet_cid": "",
            "skip_provider": True,
            "task_id": task_alias,
        },
    )
    append_jsonl_event(
        paths.events,
        "implementation_protected_path_snapshot_cleared",
        {
            "attempt": portal_attempt,
            "board_namespace": board_namespace,
            "canonical_task_cid": "task:cid:004",
            "canonical_task_key": canonical_task_key,
            "reason": "failed_agent_terminal_check_unchanged",
            "task_id": task_alias,
        },
    )
    pool_release = {
        "attempted": True,
        "base_commit": baseline_commit,
        "base_ref": "test-branch",
        "branch": branch,
        "cache_hit": False,
        "cache_key": "closed-cache-key",
        "dependency_paths": [],
        "entry_id": "closed-entry",
        "estimated_seconds_saved": 0.0,
        "handoff_reason": "implementation_command_failed",
        "invalidation_reason": "",
        "invalidation_reasons": [],
        "lifecycle_finalize": {
            "fence": 1,
            "finalized": True,
            "reason": "pool_release_implementation_command_failed",
            "state": "terminal",
        },
        "pooled": True,
        "reason": "clean_prepared_workspace",
        "released": True,
        "reused": False,
        "setup_seconds": 1.0,
        "setup_time_saved_seconds": 0.0,
        "worktree_path": workspace_path,
    }
    append_jsonl_event(
        paths.events,
        "worktree_pool_lease_released",
        pool_release,
    )
    if before_finish_event:
        append_jsonl_event(
            paths.events,
            before_finish_event,
            {"task_id": task_alias},
        )
    finished_payload: dict[str, object] = {
        "task_id": task_alias,
        "task_cid": "task:cid:004",
        "canonical_task_cid": "task:cid:004",
        "canonical_task_key": canonical_task_key,
        "board_namespace": board_namespace,
        "attempt": portal_attempt,
        "branch": branch,
        "baseline_ref": baseline_commit,
        "returncode": 1,
        "attempt_consumed": True,
        "provider_dispatched": True,
        "validation_result": {
            "attempted": False,
            "passed": True,
            "reason": "not_run",
            "results": [],
            "returncode": 0,
        },
        "implementation_commit": "",
        "commit_result": {"committed": False},
        "merge_result": {"merged": False, "reason": "not_attempted"},
        "board_completion": {
            "complete": False,
            "pending_merge": False,
            "reason": "implementation_or_validation_failed",
        },
        "failed_preservation_result": {},
        "cache_hit": False,
        "cleanup_result": {
            "cleaned": True,
            "lifecycle_finalize": {
                "finalized": False,
                "reason": "no_lifecycle_record",
            },
            "pool_release": pool_release,
            "pooled": True,
            "reason": "failed_implementation_pool_lease_released",
        },
        "diagnostic_receipt_id": "bagu-test-diagnostic",
        "lifecycle_finalize": {
            "finalized": False,
            "reason": "no_lifecycle_record",
        },
        "log_path": log_path,
        "saved_duration_seconds": 0.0,
        "setup_duration_seconds": 1.0,
        "workspace_setup": workspace_setup,
        "worktree_path": workspace_path,
    }
    finished_payload.update(finish_updates or {})
    finished = append_jsonl_event(
        paths.events,
        "implementation_finished",
        finished_payload,
    )
    append_jsonl_event(
        paths.events,
        "daemon_pass",
        {
            "active_task_id": "",
            "attempt_limited_task_ids": [],
            "blocked_count": 0,
            "completed_count": 0,
            "completion_receipt_task_ids": [],
            "eligible_ready_count": 1,
            "execution_slice_task_cids_by_id": {
                task_alias: "task:cid:004"
            },
            "execution_slice_task_statuses": {task_alias: "ready"},
            "manual_completion_authority_affected_goal_ids": [],
            "manual_completion_authority_dependency_task_ids": [],
            "manual_completion_authority_required_task_ids": [],
            "manual_completion_authority_revalidation_only": False,
            "manual_completion_authority_task_ids": [],
            "manual_completion_renewal_quarantined_task_ids": [],
            "manual_completion_revalidation_only_task_ids": [],
            "manual_completion_revalidation_task_ids": [],
            "max_task_attempts": max_task_attempts,
            "ordinary_provider_dispatch_allowed": True,
            "projection_delta_keys": [],
            "protected_path_conflicts": {},
            "quarantined_manual_completion_status_task_ids": [],
            "ready_count": 1,
            "released_retry_budget_strategy_block_task_ids": [],
            "retry_budget_rearmed_task_ids": [],
            "retry_budget_reset_deferred_task_ids": [],
            "retry_budget_reset_task_ids": [],
            "selectable_ready_count": 1,
            "selection_idle_reason": "",
            "shared_active_merge_task_ids": [],
            "shared_completed_task_ids": [],
            "strict_deprioritized_ready_count": 0,
            "virgin_task_transfer": {
                "granted_away_task_ids": [],
                "granted_to_lane_task_ids": [],
                "mode": "",
                "request_task_id": "",
            },
            "waiting_count": 0,
        },
    )
    return started, finished


def _git_protected_path_candidate(repo: Path) -> tuple[str, str, str]:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "portal-test@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Portal Test"],
        cwd=repo,
        check=True,
    )
    baseline_path = repo / "README.md"
    baseline_path.write_text("baseline\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "baseline"],
        cwd=repo,
        check=True,
    )
    baseline = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    candidate_path = repo / "inventory" / "result.json"
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_text('{"candidate":true}\n', encoding="utf-8")
    subprocess.run(
        ["git", "add", "inventory/result.json"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-qm", "preserved candidate"],
        cwd=repo,
        check=True,
    )
    candidate = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    rescue_branch = (
        "rescue/lgswf-004-attempt-2-protected-path-interrupted"
    )
    subprocess.run(
        ["git", "branch", rescue_branch, candidate],
        cwd=repo,
        check=True,
    )
    return baseline, candidate, rescue_branch


def _write_protected_path_preservation_terminal(
    paths: object,
    task_alias: str,
    *,
    baseline_commit: str,
    preserved_commit: str,
    rescue_branch: str,
    mutation_scope: str = "shared_checkout",
    provider_dispatched: bool = True,
    attempt_consumed: bool = False,
    interposed_event_type: str = "",
    preservation_event_type: str = (
        "protected_path_interrupted_worktree_preserved"
    ),
    later_event_type: str = "",
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    task_cid = "task:cid:004"
    canonical_task_key = "task/v1/protected-preservation"
    board_namespace = "task-projection.md"
    portal_attempt = 2
    branch = "implementation/lgswf-004-attempt-2"
    workspace_path = "/tmp/protected-preservation-worktree"
    protected_path = "docs/architecture/protected.md"
    workspace_setup = {
        "base_commit": baseline_commit,
        "branch": branch,
        "worktree_path": workspace_path,
    }
    common_identity = {
        "task_id": task_alias,
        "canonical_task_cid": task_cid,
        "canonical_task_key": canonical_task_key,
        "board_namespace": board_namespace,
    }
    append_jsonl_event(
        paths.events,
        "task_selected",
        {
            **common_identity,
            "title": "Protected preservation replay fixture",
            "track": "implementation",
        },
    )
    append_jsonl_event(
        paths.events,
        "implementation_protected_path_snapshot_recorded",
        {
            **common_identity,
            "attempt": portal_attempt,
            "protected_paths": [protected_path],
            "workspace_path": workspace_path,
        },
    )
    started = append_jsonl_event(
        paths.events,
        "implementation_started",
        {
            **common_identity,
            "attempt": portal_attempt,
            "branch": branch,
            "baseline_ref": baseline_commit,
            "provider_dispatched": False,
            "cache_hit": False,
            "checkpoint_directory": "/tmp/protected-preservation-checkpoint",
            "command": ["provider"],
            "execution_mode": "model-assisted",
            "log_path": "/tmp/protected-preservation.log",
            "outputs": ["inventory/result.json"],
            "saved_duration_seconds": 0.0,
            "setup_duration_seconds": 1.0,
            "timeout_policy": {"source": "test"},
            "workspace_setup": workspace_setup,
            "worktree_lifecycle": {"state": "active"},
            "worktree_path": workspace_path,
        },
    )
    append_jsonl_event(
        paths.events,
        "pre_implementation_kernel_evaluated",
        {
            **common_identity,
            "analytical_candidate_count": 0,
            "attempt": portal_attempt,
            "disposition": "abstain_review",
            "event": "pre_implementation_kernel_evaluated",
            "interface": "ImplementationDaemon@pre_implementation_kernel",
            "kernel_receipt": {"schema": "protected-test-kernel@1"},
            "provider_authorized": False,
            "provider_hook_count": 0,
            "reason_code": "no_analytical_close",
            "receipt_cid": "bagu-protected-test-kernel-receipt",
            "residual_packet_cid": "",
            "skip_provider": True,
        },
    )
    violation: dict[str, object] = {
        "reason": "implementation_protected_path_mutated",
        "task_id": task_alias,
        "attempt": portal_attempt,
        "workspace_path": workspace_path,
        "protected_paths": [protected_path],
        "mutations": [
            {
                "scope": mutation_scope,
                "path": protected_path,
                "change": "content_changed",
                "before": {"sha256": "1" * 64},
                "after": {"sha256": "2" * 64},
            }
        ],
        "shared_checkout_restored": False,
    }
    mutation = append_jsonl_event(
        paths.events,
        "implementation_protected_path_mutated",
        {
            **common_identity,
            **violation,
        },
    )
    if interposed_event_type:
        append_jsonl_event(
            paths.events,
            interposed_event_type,
            {**common_identity, "attempt": portal_attempt},
        )
    commit_result = {
        "committed": True,
        "commit": preserved_commit,
    }
    cleanup_result = {
        "cleaned": True,
        "removed_worktree": True,
        "deleted_branch": True,
        "reason": "cleaned",
    }
    append_jsonl_event(
        paths.events,
        "cleanup_finished",
        cleanup_result,
    )
    preservation: dict[str, object] = {
        "task_id": task_alias,
        "attempt": portal_attempt,
        "branch": branch,
        "worktree_path": workspace_path,
        "started_at": "2026-08-24T15:00:00+00:00",
        "finished_at": "2026-08-24T15:00:01+00:00",
        "preserved": True,
        "rescue_branch": rescue_branch,
        "implementation_commit": preserved_commit,
        "preserved_commit": preserved_commit,
        "commit_result": commit_result,
        "cleanup_result": cleanup_result,
        "pruned_seeded_context": [],
        "protected_path_violation": violation,
    }
    preserved = append_jsonl_event(
        paths.events,
        preservation_event_type,
        {
            **common_identity,
            **preservation,
        },
    )
    validation = {
        "attempted": False,
        "passed": False,
        "returncode": 1,
        "results": [],
        "reason": "implementation_protected_path_mutated",
        "protected_path_violation": violation,
    }
    finished = append_jsonl_event(
        paths.events,
        "implementation_finished",
        {
            **common_identity,
            "task_cid": task_cid,
            "attempt": portal_attempt,
            "branch": branch,
            "baseline_ref": baseline_commit,
            "returncode": 1,
            "reason": "implementation_protected_path_mutated",
            "deferred": True,
            "attempt_consumed": attempt_consumed,
            "provider_dispatched": provider_dispatched,
            "validation_result": validation,
            "implementation_commit": preserved_commit,
            "commit_result": commit_result,
            "merge_result": {"merged": False, "reason": "not_attempted"},
            "board_completion": {
                "complete": False,
                "pending_merge": False,
                "reason": "implementation_or_validation_failed",
            },
            "failed_preservation_result": preservation,
            "cleanup_result": cleanup_result,
            "log_path": "/tmp/protected-preservation.log",
            "workspace_setup": workspace_setup,
            "worktree_path": workspace_path,
            "protected_path_violation": violation,
        },
    )
    append_jsonl_event(
        paths.events,
        "daemon_pass",
        {"active_task_id": ""},
    )
    if later_event_type:
        append_jsonl_event(
            paths.events,
            later_event_type,
            {**common_identity, "attempt": portal_attempt},
        )
    state = json.loads(paths.state.read_text(encoding="utf-8"))
    state.update(
        {
            "implementation_in_progress": False,
            "active_task_id": "",
            "active_task_cid": "",
            "active_attempt": 0,
            "last_implementation_task_id": task_alias,
            "last_implementation_task_cid": task_cid,
            "last_implementation_returncode": 1,
            "last_implementation_commit": preserved_commit,
            "implementation_attempts": {task_alias: portal_attempt},
            "implementation_attempts_by_cid": {
                task_cid: portal_attempt,
            },
            "last_implementation_finished_at": (
                "2026-08-24T15:00:01+00:00"
            ),
        }
    )
    paths.state.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return started, mutation, preserved, finished


def _prepare_seeded_protected_preservation_replay(
    tmp_path: Path,
    *,
    mutation_scope: str = "shared_checkout",
    provider_dispatched: bool = True,
    attempt_consumed: bool = False,
    interposed_event_type: str = "",
    preservation_event_type: str = (
        "protected_path_interrupted_worktree_preserved"
    ),
    later_event_type: str = "",
) -> tuple[
    SimpleNamespace,
    DatabaseTaskAttempt,
    Path,
    Path,
    str,
    str,
    str,
    tuple[
        dict[str, object],
        dict[str, object],
        dict[str, object],
        dict[str, object],
    ],
]:
    repo = tmp_path / "repo"
    repo.mkdir()
    baseline, preserved_commit, rescue_branch = (
        _git_protected_path_candidate(repo)
    )
    record = _record()
    source = _attempt(attempt_number=189)
    source_bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "source-attempts",
        repository_root=repo,
        portal_factory=lambda _paths, _alias: pytest.fail(
            "source receipt recovery dispatched a provider"
        ),
        max_passes=1,
        max_task_attempts=4,
    )
    source_paths, _source_binding = source_bridge._ensure_attempt_projection(
        source,
        record,
    )
    _write_consumed_attempt_failure(source_paths, source.task_alias)
    record.revision += 1
    consumed_retry = source_bridge.recover_consumed_attempt_retry(source)

    successor = DatabaseTaskAttempt(
        attempt_id="attempt:protected-successor",
        claim_id="claim:protected-successor",
        task_cid=source.task_cid,
        task_alias=source.task_alias,
        attempt_number=1,
        owner_session_id="session:protected-successor-lane",
        fencing_token=1,
        fence_epoch=1,
        lease_id="lease:protected-successor",
        committed_phase="claimed",
        status="running",
        started_at_ms=2,
    )
    record.body = {
        **record.body,
        "completion_receipt": {
            "operation": "database_claim",
            "attempt_id": successor.attempt_id,
            "claim_id": successor.claim_id,
            "attempt_number": successor.attempt_number,
            "fencing_token": successor.fencing_token,
            "fence_epoch": successor.fence_epoch,
            "lease_id": successor.lease_id,
            "consumed_attempt_retry_source_attempt_id": source.attempt_id,
            "consumed_attempt_retry_seed": consumed_retry,
        },
    }
    record.revision += 1
    attempt_root = tmp_path / "protected-successor-attempts"
    staging_bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=attempt_root,
        repository_root=repo,
        portal_factory=lambda _paths, _alias: pytest.fail(
            "fixture staging dispatched a provider"
        ),
        max_passes=1,
        max_task_attempts=4,
    )
    paths, binding = staging_bridge._ensure_attempt_projection(
        successor,
        record,
    )
    staging_bridge._initialize_consumed_attempt_retry_seed(
        attempt=successor,
        record=record,
        paths=paths,
        binding=binding,
    )
    terminal = _write_protected_path_preservation_terminal(
        paths,
        successor.task_alias,
        baseline_commit=baseline,
        preserved_commit=preserved_commit,
        rescue_branch=rescue_branch,
        mutation_scope=mutation_scope,
        provider_dispatched=provider_dispatched,
        attempt_consumed=attempt_consumed,
        interposed_event_type=interposed_event_type,
        preservation_event_type=preservation_event_type,
        later_event_type=later_event_type,
    )
    return (
        record,
        successor,
        repo,
        attempt_root,
        baseline,
        preserved_commit,
        rescue_branch,
        terminal,
    )


def test_bridge_propagates_typed_pre_dispatch_cooldown(tmp_path: Path) -> None:
    class DeferredPortal:
        def __init__(self) -> None:
            self.closed = False

        def run_once(self) -> dict[str, object]:
            return {
                "implementation_result": {
                    "task_id": "LGSWF-004",
                    "returncode": 1,
                    "reason": "validation_project_dependency_preflight_failed",
                    "deferred": True,
                    "attempt_consumed": False,
                    "provider_dispatched": False,
                    "backoff_seconds": 300,
                }
            }

        def close_event_runtime(self) -> None:
            self.closed = True

    portal = DeferredPortal()
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: portal,
        max_passes=1,
    )

    with pytest.raises(DatabasePortalBridgeDeferred) as caught:
        bridge.run_provider(_attempt())

    assert str(caught.value) == "validation_project_dependency_preflight_failed"
    assert caught.value.backoff_seconds == 300
    assert caught.value.attempt_consumed is False
    assert caught.value.provider_dispatched is False
    assert portal.closed is True


def _external_recovery_result() -> dict[str, object]:
    return {
        "blocked": True,
        "reason": "external_protected_checkout_recovery_required",
        "unchanged": True,
        "write_count": 0,
        "implementation_result": {
            "returncode": 1,
            "reason": "external_protected_recovery_owner_active",
            "deferred": True,
            "attempt_consumed": False,
            "provider_dispatched": False,
            "backoff_seconds": (
                EXTERNAL_PROTECTED_RECOVERY_BACKOFF_SECONDS
            ),
        },
        "protected_checkout_recovery": {
            "required": True,
            "recovered": False,
            "adopted": False,
            "blocked": True,
            "reason": "external_protected_checkout_recovery_required",
            "protected_recovery_owner": "implementation_supervisor",
            "foreign_owner_liveness": "verified_live",
            "deferred": True,
            "attempt_consumed": False,
            "provider_dispatched": False,
            "backoff_seconds": (
                EXTERNAL_PROTECTED_RECOVERY_BACKOFF_SECONDS
            ),
            "lock_owner_pid": 1234,
            "lock_path": "/repository/.git/agent-checkout-mutation.lock",
        },
    }


def test_bridge_defers_verified_live_supervisor_recovery_owner(
    tmp_path: Path,
) -> None:
    class LiveForeignRecoveryPortal:
        def run_once(self) -> dict[str, object]:
            return _external_recovery_result()

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: LiveForeignRecoveryPortal(),
        max_passes=1,
    )

    with pytest.raises(DatabasePortalBridgeDeferred) as caught:
        bridge.run_provider(_attempt())

    assert str(caught.value) == (
        "external_protected_recovery_owner_active"
    )
    assert caught.value.backoff_seconds == (
        EXTERNAL_PROTECTED_RECOVERY_BACKOFF_SECONDS
    )
    assert caught.value.attempt_consumed is False
    assert caught.value.provider_dispatched is False


def test_bridge_keeps_untyped_foreign_recovery_terminal(
    tmp_path: Path,
) -> None:
    result = _external_recovery_result()
    result["implementation_result"] = None

    class UntypedForeignRecoveryPortal:
        def run_once(self) -> dict[str, object]:
            return result

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: (
            UntypedForeignRecoveryPortal()
        ),
        max_passes=1,
    )

    with pytest.raises(DatabasePortalBridgeError) as caught:
        bridge.run_provider(_attempt())

    assert not isinstance(caught.value, DatabasePortalBridgeDeferred)
    assert str(caught.value) == (
        "external_protected_checkout_recovery_required"
    )


def test_bridge_uses_safe_default_for_legacy_typed_deferral(
    tmp_path: Path,
) -> None:
    class LegacyDeferredPortal:
        def run_once(self) -> dict[str, object]:
            return {
                "implementation_result": {
                    "returncode": 1,
                    "reason": "legacy_typed_deferral",
                    "deferred": True,
                    "attempt_consumed": False,
                    "provider_dispatched": False,
                }
            }

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: LegacyDeferredPortal(),
        max_passes=1,
    )

    with pytest.raises(DatabasePortalBridgeDeferred) as caught:
        bridge.run_provider(_attempt())

    assert caught.value.backoff_seconds == 300


def test_bridge_does_not_infer_retryability_from_generic_failure_text(
    tmp_path: Path,
) -> None:
    class GenericFailurePortal:
        def run_once(self) -> dict[str, object]:
            return {
                "implementation_result": {
                    "returncode": 1,
                    "reason": "resource_capacity_backoff_requested",
                }
            }

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: GenericFailurePortal(),
        max_passes=1,
    )

    with pytest.raises(DatabasePortalBridgeError) as caught:
        bridge.run_provider(_attempt())

    assert not isinstance(caught.value, DatabasePortalBridgeDeferred)


def test_bridge_classifies_only_preserved_authoritative_validation_failure(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    commit, rescue_branch = _git_candidate_with_rescue_branch(repo)
    record = _record()
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: _ValidationFailurePortal(
            paths,
            alias,
            commit=commit,
            rescue_branch=rescue_branch,
        ),
        repository_root=repo,
        max_passes=1,
        max_task_attempts=3,
    )

    # Production retained 188 legacy outer attempts before this current-schema
    # Portal attempt.  Those coordination identities are not retry-budget
    # consumption; the independently replayed Portal attempt is generation 1.
    production_attempt = _attempt(attempt_number=189)
    with pytest.raises(DatabasePortalValidationRetry) as caught:
        bridge.run_provider(production_attempt)

    retry = caught.value
    assert retry.attempt_consumed is True
    assert retry.provider_dispatched is True
    assert retry.backoff_seconds == 0
    assert retry.retry_receipt["schema"] == DATABASE_PORTAL_VALIDATION_RETRY_SCHEMA
    assert retry.retry_receipt["implementation_commit"] == commit
    assert retry.retry_receipt["rescue_branch"] == rescue_branch
    assert retry.retry_receipt["attempt_number"] == 189
    assert retry.retry_receipt["portal_attempt"] == 1
    assert retry.retry_receipt["typed_retry_generation"] == 1
    assert retry.retry_receipt["retry_budget_basis"] == "portal_attempt"
    assert retry.retry_receipt["legacy_database_attempts_excluded"] is True
    assert retry.retry_receipt["remaining_task_attempts"] == 2
    assert retry.retry_receipt["denial_findings"] == []
    # A later blocked-status CAS advances the control revision but does not
    # invalidate the attempt's immutable task body/claim binding.
    record.revision += 1
    assert (
        bridge.recover_validation_retry(production_attempt)
        == retry.retry_receipt
    )

    successor = DatabaseTaskAttempt(
        attempt_id="attempt:002",
        claim_id="claim:002",
        task_cid="task:cid:004",
        task_alias="LGSWF-004",
        attempt_number=190,
        owner_session_id="session:bridge",
        fencing_token=8,
        fence_epoch=3,
        lease_id="lease:002",
        committed_phase="claimed",
        status="running",
        started_at_ms=2,
    )
    record.body = {
        **record.body,
        "completion_receipt": {
            "operation": "database_claim",
            "attempt_id": successor.attempt_id,
            "claim_id": successor.claim_id,
            "attempt_number": successor.attempt_number,
            "fencing_token": successor.fencing_token,
            "fence_epoch": successor.fence_epoch,
            "lease_id": successor.lease_id,
            "validation_retry_source_attempt_id": (
                production_attempt.attempt_id
            ),
            "validation_retry_seed": retry.retry_receipt,
        },
    }
    record.revision += 1
    observed: dict[str, object] = {}

    class InspectSeedPortal:
        def __init__(self, paths: object) -> None:
            self.paths = paths

        def run_once(self) -> dict[str, object]:
            observed["paths"] = self.paths
            observed["state"] = json.loads(
                self.paths.state.read_text(encoding="utf-8")
            )
            observed["events"] = [
                json.loads(line)
                for line in self.paths.events.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            return {
                "implementation_result": {
                    "returncode": 1,
                    "reason": "stop_after_seed_inspection",
                }
            }

    successor_bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "successor-attempts",
        portal_factory=lambda paths, _alias: InspectSeedPortal(paths),
        repository_root=repo,
        max_passes=1,
        max_task_attempts=3,
    )
    with pytest.raises(DatabasePortalBridgeError, match="stop_after_seed_inspection"):
        successor_bridge.run_provider(successor)
    state = observed["state"]
    assert isinstance(state, dict)
    assert state["implementation_attempts"]["LGSWF-004"] == 1
    assert state["implementation_attempts_by_cid"]["task:cid:004"] == 1
    assert state["last_implementation_commit"] == commit
    assert state["last_implementation_branch"] == rescue_branch
    events = observed["events"]
    assert isinstance(events, list)
    assert events[0]["type"] == "database_portal_validation_retry_seeded"
    assert events[0]["source_retry_receipt_id"] == retry.retry_receipt[
        "receipt_id"
    ]
    successor_paths = observed["paths"]
    portal = PortalImplementationDaemon(
        todo_path=successor_paths.task_projection,
        state_path=successor_paths.state,
        strategy_path=successor_paths.strategy,
        events_path=successor_paths.events,
        repo_root=repo,
        task_header_prefix="LGSWF-",
        max_task_attempts=3,
    )
    projected_task = portal._load_tasks()[0]
    projected_state = PortalTaskState.load(successor_paths.state)
    assert portal._task_attempt(projected_state, projected_task) == 2
    # A successor process can die after Portal charges attempt 2 and records
    # its start.  The sealed seed plus exact in-flight state must remain
    # adoptable instead of being mistaken for seed corruption.
    progressed = dict(state)
    progressed.update(
        {
            "implementation_attempts": {source.task_alias: 2},
            "implementation_attempts_by_cid": {source.task_cid: 2},
            "active_task_id": source.task_alias,
            "active_task_cid": source.task_cid,
            "active_attempt": 2,
            "implementation_in_progress": True,
            "last_implementation_task_id": source.task_alias,
            "last_implementation_task_cid": source.task_cid,
            "last_implementation_returncode": None,
            "last_implementation_finished_at": "",
        }
    )
    successor_paths.state.write_text(
        json.dumps(progressed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    append_jsonl_event(
        successor_paths.events,
        "implementation_started",
        {
            "task_id": source.task_alias,
            "canonical_task_cid": source.task_cid,
            "attempt": 2,
            "provider_dispatched": False,
        },
    )
    progressed_calls: list[str] = []

    class InspectProgressedPortal:
        def run_once(self) -> dict[str, object]:
            progressed_calls.append("adopt")
            return {
                "implementation_result": {
                    "returncode": 1,
                    "reason": "stop_after_progressed_seed_inspection",
                }
            }

    progressed_bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "successor-attempts",
        portal_factory=lambda _paths, _alias: InspectProgressedPortal(),
        max_passes=1,
        max_task_attempts=3,
    )
    with pytest.raises(
        DatabasePortalBridgeError,
        match="stop_after_progressed_seed_inspection",
    ):
        progressed_bridge.run_provider(successor)
    assert progressed_calls == ["adopt"]
    authority = portal._prior_seed_proposal_authority(projected_task)
    assert authority["ok"] is True
    assert authority["database_validation_retry_seed"] is True
    assert authority["authorized_paths"] == ["inventory/result.json"]


def test_bridge_capacity_retry_replays_without_dispatch_and_seeds_successor(
    tmp_path: Path,
) -> None:
    record = _record()
    calls: list[int] = []
    attempt_root = tmp_path / "attempts"
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=attempt_root,
        portal_factory=lambda paths, alias: _CapacityFailurePortal(
            paths,
            alias,
            calls=calls,
        ),
        max_passes=1,
        max_task_attempts=3,
    )
    source = _attempt(attempt_number=189)

    with pytest.raises(DatabasePortalCapacityRetry) as caught:
        bridge.run_provider(source)
    retry = caught.value.retry_receipt
    assert calls == [1]
    assert retry["schema"] == DATABASE_PORTAL_CAPACITY_RETRY_SCHEMA
    assert retry["portal_attempt"] == 1
    assert retry["remaining_task_attempts"] == 2
    assert retry["attempt_consumed"] is True
    assert retry["provider_dispatched"] is True

    # Response loss after the durable Portal event must replay the exact
    # receipt before constructing a provider or advancing attempt state.
    replay = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=attempt_root,
        portal_factory=lambda _paths, _alias: pytest.fail(
            "capacity replay dispatched the provider"
        ),
        max_passes=1,
        max_task_attempts=3,
    )
    with pytest.raises(DatabasePortalCapacityRetry) as replayed:
        replay.run_provider(source)
    assert replayed.value.retry_receipt == retry
    assert calls == [1]

    successor = DatabaseTaskAttempt(
        attempt_id="attempt:002",
        claim_id="claim:002",
        task_cid=source.task_cid,
        task_alias=source.task_alias,
        # Coordination attempt numbers are lane-local; the shared CAS receipt,
        # not a numeric comparison with lane A's 189, orders this handoff.
        attempt_number=1,
        owner_session_id="session:successor-lane",
        fencing_token=1,
        fence_epoch=1,
        lease_id="lease:002",
        committed_phase="claimed",
        status="running",
        started_at_ms=2,
    )
    record.body = {
        **record.body,
        "completion_receipt": {
            "operation": "database_claim",
            "attempt_id": successor.attempt_id,
            "claim_id": successor.claim_id,
            "attempt_number": successor.attempt_number,
            "fencing_token": successor.fencing_token,
            "fence_epoch": successor.fence_epoch,
            "lease_id": successor.lease_id,
            "capacity_retry_source_attempt_id": source.attempt_id,
            "capacity_retry_seed": retry,
        },
    }
    record.revision += 1
    observed: dict[str, object] = {}

    class InspectCapacitySeedPortal:
        def __init__(self, paths: object) -> None:
            self.paths = paths

        def run_once(self) -> dict[str, object]:
            observed["paths"] = self.paths
            observed["state"] = json.loads(
                self.paths.state.read_text(encoding="utf-8")
            )
            observed["events"] = [
                json.loads(line)
                for line in self.paths.events.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            return {
                "implementation_result": {
                    "returncode": 1,
                    "reason": "stop_after_capacity_seed_inspection",
                }
            }

    successor_bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        # A new root models a successor lane with no source attempt files.
        attempt_root=tmp_path / "successor-attempts",
        portal_factory=lambda paths, _alias: InspectCapacitySeedPortal(paths),
        max_passes=1,
        max_task_attempts=3,
    )
    with pytest.raises(
        DatabasePortalBridgeError,
        match="stop_after_capacity_seed_inspection",
    ):
        successor_bridge.run_provider(successor)
    state = observed["state"]
    assert isinstance(state, dict)
    assert state["implementation_attempts"][source.task_alias] == 1
    assert state["implementation_attempts_by_cid"][source.task_cid] == 1
    events = observed["events"]
    assert isinstance(events, list)
    assert events[0]["type"] == "database_portal_capacity_retry_seeded"
    assert events[0]["source_retry_receipt_id"] == retry["receipt_id"]
    successor_paths = observed["paths"]
    portal = PortalImplementationDaemon(
        todo_path=successor_paths.task_projection,
        state_path=successor_paths.state,
        strategy_path=successor_paths.strategy,
        events_path=successor_paths.events,
        repo_root=tmp_path,
        task_header_prefix="LGSWF-",
        max_task_attempts=3,
    )
    projected_task = portal._load_tasks()[0]
    projected_state = PortalTaskState.load(successor_paths.state)
    assert portal._task_attempt(projected_state, projected_task) == 2


def test_bridge_recovers_consumed_attempt_and_seeds_lane_local_successor(
    tmp_path: Path,
) -> None:
    record = _record()
    source = _attempt(attempt_number=189)
    source_bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "source-attempts",
        portal_factory=lambda _paths, _alias: pytest.fail(
            "consumed-attempt recovery dispatched a provider"
        ),
        max_passes=1,
        max_task_attempts=4,
    )
    source_paths, _binding = source_bridge._ensure_attempt_projection(
        source,
        record,
    )
    started, finished = _write_consumed_attempt_failure(
        source_paths,
        source.task_alias,
    )
    # A later control-status CAS may advance the record revision without
    # changing the semantic task projection sealed by the source attempt.
    record.revision += 1

    retry = source_bridge.recover_consumed_attempt_retry(source)
    expected_fields = {
        "schema",
        "disposition",
        "reason",
        "failure_class",
        "provider_capacity_classification",
        "capacity_retry_proven",
        "task_cid",
        "task_alias",
        "attempt_id",
        "claim_id",
        "lease_id",
        "attempt_number",
        "fencing_token",
        "fence_epoch",
        "source_task_revision",
        "portal_attempt",
        "ordinary_retry_generation",
        "retry_budget_basis",
        "legacy_database_attempts_excluded",
        "max_task_attempts",
        "remaining_task_attempts",
        "attempt_consumed",
        "provider_dispatched",
        "backoff_seconds",
        "retry_not_before_ms",
        "binding_id",
        "events_digest",
        "event_stream_id",
        "implementation_started_event_id",
        "implementation_finished_event_id",
        "baseline_commit",
        "implementation_returncode",
        "receipt_id",
    }
    assert set(retry) == expected_fields
    assert retry["schema"] == DATABASE_PORTAL_CONSUMED_ATTEMPT_RETRY_SCHEMA
    assert retry["reason"] == "unclassified_post_dispatch_failure"
    assert retry["provider_capacity_classification"] == "unproven"
    assert retry["capacity_retry_proven"] is False
    assert retry["attempt_number"] == 189
    assert retry["source_task_revision"] == 11
    assert retry["portal_attempt"] == 1
    assert retry["ordinary_retry_generation"] == 1
    assert retry["remaining_task_attempts"] == 3
    assert retry["backoff_seconds"] == 0
    assert retry["retry_not_before_ms"] == 0
    assert retry["implementation_started_event_id"] == started["event_id"]
    assert retry["implementation_finished_event_id"] == finished["event_id"]
    assert retry["receipt_id"] == _capacity_record_id(
        dict(retry),
        "receipt_id",
    )

    successor = DatabaseTaskAttempt(
        attempt_id="attempt:successor",
        claim_id="claim:successor",
        task_cid=source.task_cid,
        task_alias=source.task_alias,
        # Attempt numbers are lane-local.  The exact claim CAS receipt orders
        # this successor even though its local number restarts at one.
        attempt_number=1,
        owner_session_id="session:successor-lane",
        fencing_token=1,
        fence_epoch=1,
        lease_id="lease:successor",
        committed_phase="claimed",
        status="running",
        started_at_ms=2,
    )
    record.body = {
        **record.body,
        "completion_receipt": {
            "operation": "database_claim",
            "attempt_id": successor.attempt_id,
            "claim_id": successor.claim_id,
            "attempt_number": successor.attempt_number,
            "fencing_token": successor.fencing_token,
            "fence_epoch": successor.fence_epoch,
            "lease_id": successor.lease_id,
            "consumed_attempt_retry_source_attempt_id": source.attempt_id,
            "consumed_attempt_retry_seed": retry,
        },
    }
    record.revision += 1
    valid_claim_receipt = dict(record.body["completion_receipt"])
    tampered_retry = dict(retry)
    tampered_retry["capacity_retry_proven"] = True
    record.body = {
        **record.body,
        "completion_receipt": {
            **valid_claim_receipt,
            "consumed_attempt_retry_seed": tampered_retry,
        },
    }
    tampered_bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "tampered-successor-attempts",
        portal_factory=lambda _paths, _alias: pytest.fail(
            "tampered consumed-attempt seed dispatched a provider"
        ),
        max_passes=1,
        max_task_attempts=4,
    )
    with pytest.raises(
        DatabasePortalBridgeError,
        match="consumed-attempt retry seed failed verification",
    ):
        tampered_bridge.run_provider(successor)
    record.body = {
        **record.body,
        "completion_receipt": valid_claim_receipt,
    }
    observed: dict[str, object] = {}

    class InspectConsumedSeedPortal:
        def __init__(self, paths: object) -> None:
            self.paths = paths

        def run_once(self) -> dict[str, object]:
            observed["paths"] = self.paths
            observed["state"] = json.loads(
                self.paths.state.read_text(encoding="utf-8")
            )
            observed["events"] = [
                json.loads(line)
                for line in self.paths.events.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            return {
                "implementation_result": {
                    "returncode": 1,
                    "reason": "stop_after_consumed_seed_inspection",
                }
            }

    successor_bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "successor-attempts",
        portal_factory=lambda paths, _alias: InspectConsumedSeedPortal(paths),
        max_passes=1,
        max_task_attempts=4,
    )
    with pytest.raises(
        DatabasePortalBridgeError,
        match="stop_after_consumed_seed_inspection",
    ):
        successor_bridge.run_provider(successor)
    state = observed["state"]
    assert isinstance(state, dict)
    assert state["implementation_attempts"][source.task_alias] == 1
    assert state["implementation_attempts_by_cid"][source.task_cid] == 1
    events = observed["events"]
    assert isinstance(events, list)
    assert events[0]["type"] == "database_portal_consumed_attempt_retry_seeded"
    assert events[0]["schema"] == DATABASE_PORTAL_CONSUMED_ATTEMPT_RETRY_SEED_SCHEMA
    assert events[0]["source_retry_receipt_id"] == retry["receipt_id"]
    successor_paths = observed["paths"]
    portal = PortalImplementationDaemon(
        todo_path=successor_paths.task_projection,
        state_path=successor_paths.state,
        strategy_path=successor_paths.strategy,
        events_path=successor_paths.events,
        repo_root=tmp_path,
        task_header_prefix="LGSWF-",
        max_task_attempts=4,
    )
    projected_task = portal._load_tasks()[0]
    projected_state = PortalTaskState.load(successor_paths.state)
    assert portal._task_attempt(projected_state, projected_task) == 2
    _write_consumed_attempt_failure(
        successor_paths,
        successor.task_alias,
        portal_attempt=2,
        max_task_attempts=4,
    )
    seeded_replay = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "successor-attempts",
        portal_factory=lambda _paths, _alias: pytest.fail(
            "seeded consumed-attempt terminal replay dispatched N+1"
        ),
        max_passes=1,
        max_task_attempts=4,
    )
    with pytest.raises(DatabasePortalConsumedAttemptTerminal) as replayed:
        seeded_replay.run_provider(successor)
    assert replayed.value.retry_receipt["portal_attempt"] == 2
    assert replayed.value.retry_receipt["remaining_task_attempts"] == 2


def test_bridge_replays_exact_protected_preservation_before_seed_reinit(
    tmp_path: Path,
) -> None:
    (
        record,
        successor,
        repo,
        attempt_root,
        baseline,
        preserved_commit,
        rescue_branch,
        terminal,
    ) = _prepare_seeded_protected_preservation_replay(tmp_path)
    started, mutation, preserved, finished = terminal
    factory_calls: list[str] = []

    def unexpected_factory(_paths: object, _alias: str) -> object:
        factory_calls.append("called")
        return SimpleNamespace(run_once=lambda: {})

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=attempt_root,
        repository_root=repo,
        portal_factory=unexpected_factory,
        max_passes=1,
        max_task_attempts=4,
    )
    recovered = bridge.recover_protected_path_preservation(successor)
    with pytest.raises(DatabasePortalProtectedPathPreserved) as caught:
        bridge.run_provider(successor)

    receipt = caught.value.retry_receipt
    expected_fields = {
        "schema",
        "disposition",
        "reason",
        "task_cid",
        "task_alias",
        "attempt_id",
        "claim_id",
        "lease_id",
        "attempt_number",
        "fencing_token",
        "fence_epoch",
        "source_task_revision",
        "portal_attempt",
        "attempt_consumed",
        "provider_dispatched",
        "completion_authoritative",
        "local_recovery_required",
        "mutation_scopes",
        "protected_paths",
        "baseline_commit",
        "implementation_commit",
        "preserved_commit",
        "rescue_branch",
        "original_branch",
        "original_worktree_path",
        "binding_id",
        "events_digest",
        "event_stream_id",
        "implementation_started_event_id",
        "protected_mutation_event_id",
        "preservation_event_id",
        "implementation_finished_event_id",
        "protected_path_violation_digest",
        "preservation_digest",
        "receipt_id",
    }
    assert set(receipt) == expected_fields
    assert recovered == receipt
    assert caught.value.preservation_receipt == receipt
    assert receipt["schema"] == (
        DATABASE_PORTAL_PROTECTED_PATH_PRESERVATION_SCHEMA
    )
    assert receipt["attempt_consumed"] is False
    assert receipt["provider_dispatched"] is True
    assert receipt["completion_authoritative"] is False
    assert receipt["local_recovery_required"] is True
    assert receipt["portal_attempt"] == 2
    assert receipt["baseline_commit"] == baseline
    assert receipt["implementation_commit"] == preserved_commit
    assert receipt["preserved_commit"] == preserved_commit
    assert receipt["rescue_branch"] == rescue_branch
    assert receipt["mutation_scopes"] == ["shared_checkout"]
    assert receipt["protected_paths"] == [
        "docs/architecture/protected.md"
    ]
    assert receipt["implementation_started_event_id"] == started["event_id"]
    assert receipt["protected_mutation_event_id"] == mutation["event_id"]
    assert receipt["preservation_event_id"] == preserved["event_id"]
    assert receipt["implementation_finished_event_id"] == finished["event_id"]
    assert receipt["receipt_id"] == _capacity_record_id(
        dict(receipt),
        "receipt_id",
    )
    assert factory_calls == []


@pytest.mark.parametrize(
    ("terminal_options", "error_match"),
    [
        pytest.param(
            {"mutation_scope": "workspace"},
            "terminal failed verification",
            id="workspace-mutation",
        ),
        pytest.param(
            {"provider_dispatched": False},
            "terminal failed verification",
            id="provider-not-dispatched",
        ),
        pytest.param(
            {"attempt_consumed": True},
            "terminal failed verification",
            id="attempt-consumed",
        ),
        pytest.param(
            {"interposed_event_type": "unexpected_diagnostic"},
            "event chain is not exact",
            id="arbitrary-interposed-event",
        ),
        pytest.param(
            {
                "interposed_event_type": (
                    "implementation_post_dispatch_capacity_retry"
                )
            },
            "event chain is not exact",
            id="capacity-terminal-interposed",
        ),
        pytest.param(
            {
                "preservation_event_type": (
                    "failed_validation_worktree_preserved"
                )
            },
            "event chain is not exact",
            id="validation-preservation-variant",
        ),
        pytest.param(
            {"later_event_type": "implementation_started"},
            "event chain is not exact",
            id="later-execution-event",
        ),
    ],
)
def test_bridge_rejects_near_protected_preservation_without_dispatch(
    tmp_path: Path,
    terminal_options: dict[str, object],
    error_match: str,
) -> None:
    (
        record,
        successor,
        repo,
        attempt_root,
        _baseline,
        _preserved_commit,
        _rescue_branch,
        _terminal,
    ) = _prepare_seeded_protected_preservation_replay(
        tmp_path,
        **terminal_options,
    )
    factory_calls: list[str] = []

    def unexpected_factory(_paths: object, _alias: str) -> object:
        factory_calls.append("called")
        return SimpleNamespace(run_once=lambda: {})

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=attempt_root,
        repository_root=repo,
        portal_factory=unexpected_factory,
        max_passes=1,
        max_task_attempts=4,
    )
    with pytest.raises(DatabasePortalBridgeError, match=error_match) as caught:
        bridge.run_provider(successor)
    assert type(caught.value) is DatabasePortalBridgeError
    assert factory_calls == []


def test_bridge_fails_closed_on_unconsumed_protected_preservation_seed(
    tmp_path: Path,
) -> None:
    (
        record,
        source_attempt,
        repo,
        source_attempt_root,
        _baseline,
        _preserved_commit,
        _rescue_branch,
        _terminal,
    ) = _prepare_seeded_protected_preservation_replay(tmp_path)
    source_bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=source_attempt_root,
        repository_root=repo,
        portal_factory=lambda _paths, _alias: pytest.fail(
            "source recovery dispatched a provider"
        ),
        max_passes=1,
        max_task_attempts=4,
    )
    preservation_seed = (
        source_bridge.recover_protected_path_preservation(source_attempt)
    )
    target = DatabaseTaskAttempt(
        attempt_id="attempt:protected-seed-target",
        claim_id="claim:protected-seed-target",
        task_cid=source_attempt.task_cid,
        task_alias=source_attempt.task_alias,
        attempt_number=2,
        owner_session_id="session:protected-seed-target",
        fencing_token=2,
        fence_epoch=2,
        lease_id="lease:protected-seed-target",
        committed_phase="claimed",
        status="running",
        started_at_ms=3,
    )
    record.body = {
        **record.body,
        "completion_receipt": {
            "operation": "database_claim",
            "attempt_id": target.attempt_id,
            "claim_id": target.claim_id,
            "attempt_number": target.attempt_number,
            "fencing_token": target.fencing_token,
            "fence_epoch": target.fence_epoch,
            "lease_id": target.lease_id,
            "protected_preservation_source_attempt_id": (
                source_attempt.attempt_id
            ),
            "protected_preservation_seed": dict(preservation_seed),
        },
    }
    record.revision += 1
    target_root = tmp_path / "protected-seed-target-attempts"
    factory_calls: list[str] = []

    def unexpected_factory(_paths: object, _alias: str) -> object:
        factory_calls.append("called")
        return SimpleNamespace(run_once=lambda: {})

    target_bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=target_root,
        repository_root=repo,
        portal_factory=unexpected_factory,
        max_passes=1,
        max_task_attempts=4,
    )
    target_paths = target_bridge._paths(target)
    with pytest.raises(
        DatabasePortalBridgeError,
        match="protected preservation seed consumption is not implemented",
    ):
        target_bridge.run_provider(target)
    assert factory_calls == []
    assert not target_paths.binding.exists()


def test_bridge_replays_consumed_attempt_terminal_without_dispatch(
    tmp_path: Path,
) -> None:
    record = _record()
    source = _attempt(attempt_number=189)
    attempt_root = tmp_path / "attempts"
    first = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=attempt_root,
        portal_factory=lambda _paths, _alias: pytest.fail(
            "fixture should be written before provider construction"
        ),
        max_passes=1,
        max_task_attempts=4,
    )
    paths, binding = first._ensure_attempt_projection(source, record)
    _write_consumed_attempt_failure(
        paths,
        source.task_alias,
        max_task_attempts=4,
    )
    expected = first._consumed_attempt_retry_receipt(
        attempt=source,
        paths=paths,
        binding=binding,
    )
    assert expected is not None

    replay = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=attempt_root,
        portal_factory=lambda _paths, _alias: pytest.fail(
            "consumed-attempt terminal replay dispatched N+1"
        ),
        max_passes=1,
        max_task_attempts=4,
    )
    with pytest.raises(DatabasePortalConsumedAttemptTerminal) as caught:
        replay.run_provider(source)

    assert str(caught.value) == "portal_provider_failed"
    assert caught.value.retry_receipt == expected


@pytest.mark.parametrize(
    ("portal_attempt", "max_task_attempts", "finish_updates", "later_event"),
    (
        (1, 4, {"attempt_consumed": False}, ""),
        (1, 4, {"provider_dispatched": False}, ""),
        (
            1,
            4,
            {
                "validation_result": {
                    "attempted": True,
                    "passed": False,
                    "reason": "declared_validation_failed",
                    "results": [],
                    "returncode": 1,
                }
            },
            "",
        ),
        (1, 4, {"implementation_commit": "c" * 40}, ""),
        (
            1,
            4,
            {
                "reason": "provider_authentication_denied",
                "retryable": False,
                "failure_class": "terminal_provider_failure",
            },
            "",
        ),
        (1, 4, {"exception_result": {"type": "RuntimeError"}}, ""),
        (1, 4, {"timeout_result": {"timed_out": True}}, ""),
        (1, 4, {"termination_result": {"signal": 9}}, ""),
        (1, 4, {"returncode": 2}, ""),
        (1, 4, {"error": "unknown_new_failure_shape"}, ""),
        (4, 4, {}, ""),
        (1, 4, {}, "task_completed"),
    ),
)
def test_bridge_consumed_attempt_recovery_requires_exact_terminal_chain(
    tmp_path: Path,
    portal_attempt: int,
    max_task_attempts: int,
    finish_updates: dict[str, object],
    later_event: str,
) -> None:
    record = _record()
    source = _attempt(attempt_number=189)
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: pytest.fail(
            "ineligible recovery dispatched a provider"
        ),
        max_passes=1,
        max_task_attempts=max_task_attempts,
    )
    paths, _binding = bridge._ensure_attempt_projection(source, record)
    _write_consumed_attempt_failure(
        paths,
        source.task_alias,
        portal_attempt=portal_attempt,
        max_task_attempts=max_task_attempts,
        finish_updates=finish_updates,
    )
    if later_event:
        append_jsonl_event(
            paths.events,
            later_event,
            {
                "task_id": source.task_alias,
                "canonical_task_cid": source.task_cid,
            },
        )
    with pytest.raises(
        DatabasePortalBridgeError,
        match="not eligible for consumed-attempt retry recovery",
    ):
        bridge.recover_consumed_attempt_retry(source)


def test_bridge_consumed_attempt_recovery_rejects_arbitrary_prefinish_event(
    tmp_path: Path,
) -> None:
    record = _record()
    source = _attempt(attempt_number=189)
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: pytest.fail(
            "ineligible recovery dispatched a provider"
        ),
        max_passes=1,
        max_task_attempts=4,
    )
    paths, _binding = bridge._ensure_attempt_projection(source, record)
    _write_consumed_attempt_failure(
        paths,
        source.task_alias,
        max_task_attempts=4,
        before_finish_event="implementation_unknown_failure_detail",
    )

    with pytest.raises(
        DatabasePortalBridgeError,
        match="not eligible for consumed-attempt retry recovery",
    ):
        bridge.recover_consumed_attempt_retry(source)


def test_bridge_rejects_mutually_exclusive_retry_seeds_before_projection(
    tmp_path: Path,
) -> None:
    record = _record()
    attempt = _attempt()
    record.body = {
        **record.body,
        "completion_receipt": {
            "operation": "database_claim",
            "attempt_id": attempt.attempt_id,
            "claim_id": attempt.claim_id,
            "attempt_number": attempt.attempt_number,
            "fencing_token": attempt.fencing_token,
            "fence_epoch": attempt.fence_epoch,
            "lease_id": attempt.lease_id,
            "validation_retry_seed": {},
            "capacity_retry_seed": {},
            "consumed_attempt_retry_seed": {},
        },
    }
    called: list[str] = []
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: called.append("provider"),
        max_passes=1,
        max_task_attempts=4,
    )

    with pytest.raises(DatabasePortalBridgeError, match="conflicting retry seeds"):
        bridge.run_provider(attempt)
    assert called == []
    assert not (tmp_path / "attempts").exists()


def test_bridge_capacity_at_attempt_cap_is_terminal_and_replay_safe(
    tmp_path: Path,
) -> None:
    calls: list[int] = []
    attempt_root = tmp_path / "attempts"
    record = _record()
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=attempt_root,
        portal_factory=lambda paths, alias: _CapacityFailurePortal(
            paths,
            alias,
            calls=calls,
            portal_attempt=3,
        ),
        max_passes=1,
        max_task_attempts=3,
    )

    with pytest.raises(DatabasePortalBridgeError) as caught:
        bridge.run_provider(_attempt())
    assert not isinstance(caught.value, DatabasePortalCapacityRetry)
    assert str(caught.value) == "portal_retry_budget_exhausted"
    assert calls == [3]

    replay = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=attempt_root,
        portal_factory=lambda _paths, _alias: pytest.fail(
            "exhausted capacity replay dispatched the provider"
        ),
        max_passes=1,
        max_task_attempts=3,
    )
    with pytest.raises(DatabasePortalBridgeError) as replayed:
        replay.run_provider(_attempt())
    assert not isinstance(replayed.value, DatabasePortalCapacityRetry)
    assert str(replayed.value) == "portal_retry_budget_exhausted"
    assert calls == [3]


def test_bridge_stale_capacity_event_cannot_override_later_disposition(
    tmp_path: Path,
) -> None:
    record = _record()
    source = _attempt()
    calls: list[int] = []
    attempt_root = tmp_path / "attempts"
    first = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=attempt_root,
        portal_factory=lambda paths, alias: _CapacityFailurePortal(
            paths,
            alias,
            calls=calls,
        ),
        max_passes=1,
        max_task_attempts=3,
    )
    with pytest.raises(DatabasePortalCapacityRetry):
        first.run_provider(source)
    append_jsonl_event(
        first._paths(source).events,
        "implementation_finished",
        {
            "task_id": source.task_alias,
            "canonical_task_cid": source.task_cid,
            "attempt": 2,
            "returncode": 0,
            "attempt_consumed": True,
            "provider_dispatched": True,
        },
    )

    class LaterDispositionPortal:
        def run_once(self) -> dict[str, object]:
            return {
                "implementation_result": {
                    "returncode": 1,
                    "reason": "later_disposition_observed",
                }
            }

    replay = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=attempt_root,
        portal_factory=lambda _paths, _alias: LaterDispositionPortal(),
        max_passes=1,
        max_task_attempts=3,
    )
    with pytest.raises(
        DatabasePortalBridgeError,
        match="later_disposition_observed",
    ) as caught:
        replay.run_provider(source)
    assert not isinstance(caught.value, DatabasePortalCapacityRetry)
    assert calls == [1]


@pytest.mark.parametrize(
    ("max_task_attempts", "denied_paths"),
    ((1, ()), (3, ("outside.py",))),
)
def test_bridge_keeps_exhausted_or_policy_denied_validation_failure_terminal(
    tmp_path: Path,
    max_task_attempts: int,
    denied_paths: tuple[str, ...],
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    commit, rescue_branch = _git_candidate_with_rescue_branch(repo)
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: _ValidationFailurePortal(
            paths,
            alias,
            commit=commit,
            rescue_branch=rescue_branch,
            denied_paths=denied_paths,
        ),
        repository_root=repo,
        max_passes=1,
        max_task_attempts=max_task_attempts,
    )

    with pytest.raises(DatabasePortalBridgeError) as caught:
        bridge.run_provider(_attempt())

    assert not isinstance(caught.value, DatabasePortalValidationRetry)
    assert str(caught.value) == "portal_provider_failed"


def test_bridge_does_not_defer_successful_zero_provider_closure(
    tmp_path: Path,
) -> None:
    class DeterministicPortal(_CompletingPortal):
        def run_once(self) -> dict[str, object]:
            result = super().run_once()
            implementation = result["implementation_result"]
            assert isinstance(implementation, dict)
            implementation["attempt_consumed"] = False
            implementation["provider_dispatched"] = False
            implementation["backoff_seconds"] = 0
            return result

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: DeterministicPortal(paths, alias),
    )

    provider = bridge.run_provider(_attempt())

    assert provider["accepted"] is True


def test_bridge_uses_only_attempt_local_projection_and_seals_receipt(
    tmp_path: Path,
) -> None:
    canonical_board = tmp_path / "canonical-board.md"
    canonical_board.write_text(
        "# Canonical\n\n## LGSWF-004 Authority\n\n- Status: ready\n",
        encoding="utf-8",
    )
    original = canonical_board.read_bytes()
    portals: list[_CompletingPortal] = []

    def factory(paths: object, alias: str) -> _CompletingPortal:
        portal = _CompletingPortal(paths, alias)
        portals.append(portal)
        return portal

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=factory,
    )
    provider = bridge.run_provider(_attempt())
    effect = bridge.apply_effect(_attempt(), provider)
    validation = bridge.validate_effect(_attempt(), effect)

    assert provider["schema"] == DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA
    assert provider["accepted"] is True
    assert provider["provider"] == "PortalImplementationDaemon"
    assert provider["completion_authority"] == "DatabaseImplementationDaemon"
    assert provider["evidence_digest"].startswith("sha256:")
    assert "private provider payload" not in json.dumps(provider)
    assert "provider_payload" not in json.dumps(provider)
    assert effect["status"] == "applied"
    assert validation["outcome"] == "passed"
    assert validation["evidence_digest"] == provider["evidence_digest"]
    assert canonical_board.read_bytes() == original
    assert portals and portals[0].closed is True
    attempt_boards = list((tmp_path / "attempts").glob("*/task-projection.md"))
    assert len(attempt_boards) == 1
    assert "Projection authority: false" in attempt_boards[0].read_text(encoding="utf-8")


def test_database_portal_attempt_projection_verifier_accepts_only_status_mutation(
    tmp_path: Path,
) -> None:
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: _CompletingPortal(paths, alias),
    )
    record = bridge._record_for_attempt(bridge.task_source, _attempt())
    paths, expected_binding = bridge._ensure_attempt_projection(
        _attempt(), record
    )

    verified = verify_database_portal_attempt_projection(
        paths.task_projection,
        expected_task_alias="LGSWF-004",
        expected_task_cid="task:cid:004",
    )
    paths.task_projection.write_text(
        paths.task_projection.read_text(encoding="utf-8").replace(
            "- Status: ready", "- Status: completed"
        ),
        encoding="utf-8",
    )
    status_only = verify_database_portal_attempt_projection(
        paths.task_projection,
        expected_task_alias="LGSWF-004",
        expected_task_cid="task:cid:004",
    )

    assert verified["verified"] is True
    assert verified["binding_id"] == expected_binding["binding_id"]
    assert verified["attempt_id"] == expected_binding["attempt_id"]
    assert verified["claim_id"] == expected_binding["claim_id"]
    assert verified["task_alias"] == expected_binding["task_alias"]
    assert verified["task_cid"] == expected_binding["task_cid"]
    assert verified["projection_authority"] is False
    assert status_only == verified


@pytest.mark.parametrize(
    "tamper",
    (
        "immutable_projection",
        "authority_flag",
        "binding_identity",
        "attempt_directory",
    ),
)
def test_database_portal_attempt_projection_verifier_rejects_tampering(
    tmp_path: Path,
    tamper: str,
) -> None:
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: _CompletingPortal(paths, alias),
    )
    record = bridge._record_for_attempt(bridge.task_source, _attempt())
    paths, _binding = bridge._ensure_attempt_projection(_attempt(), record)

    if tamper == "immutable_projection":
        paths.task_projection.write_text(
            paths.task_projection.read_text(encoding="utf-8").replace(
                "- Acceptance: Focused validation passes",
                "- Acceptance: validation waived",
            ),
            encoding="utf-8",
        )
    elif tamper in {"authority_flag", "binding_identity"}:
        binding = json.loads(paths.binding.read_text(encoding="utf-8"))
        binding.pop("binding_id")
        if tamper == "authority_flag":
            binding["projection_authority"] = True
        else:
            binding["task_cid"] = "task:cid:forged"
        binding["binding_id"] = "sha256:" + hashlib.sha256(
            json.dumps(
                binding,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        paths.binding.write_text(
            json.dumps(binding, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        renamed = paths.root.with_name("copied-attempt-projection")
        paths.root.rename(renamed)
        paths = type(paths)(
            root=renamed,
            task_projection=renamed / paths.task_projection.name,
            binding=renamed / paths.binding.name,
            state=renamed / paths.state.name,
            strategy=renamed / paths.strategy.name,
            events=renamed / paths.events.name,
            implementation_logs=renamed / paths.implementation_logs.name,
        )

    with pytest.raises(DatabasePortalBridgeError):
        verify_database_portal_attempt_projection(
            paths.task_projection,
            expected_task_alias="LGSWF-004",
            expected_task_cid="task:cid:004",
        )


def test_bridge_rejects_projection_contract_tampering(tmp_path: Path) -> None:
    class TamperingPortal(_CompletingPortal):
        def run_once(self) -> dict[str, object]:
            text = self.paths.task_projection.read_text(encoding="utf-8")
            self.paths.task_projection.write_text(
                text.replace(
                    "- Acceptance: Focused validation passes",
                    "- Acceptance: no validation required",
                ),
                encoding="utf-8",
            )
            return {"implementation_result": {"returncode": 0}}

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: TamperingPortal(paths, alias),
    )
    with pytest.raises(DatabasePortalBridgeError, match="outside its mutable status"):
        bridge.run_provider(_attempt())


def test_bridge_scopes_validation_to_checked_nested_repository(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "checkout"
    nested_repository = repository_root / "ipfs_datasets_py"
    nested_repository.mkdir(parents=True)
    (nested_repository / ".git").write_text(
        "gitdir: ../.git/modules/ipfs_datasets_py\n",
        encoding="utf-8",
    )

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_owned_record("ipfs_datasets_py")),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: _CompletingPortal(paths, alias),
        repository_root=repository_root,
        worktree_submodule_paths=("ipfs_datasets_py",),
    )

    provider = bridge.run_provider(_attempt())

    projection_path = next((tmp_path / "attempts").glob("*/task-projection.md"))
    projection = projection_path.read_text(encoding="utf-8")
    assert provider["accepted"] is True
    assert (
        "- Validation: cd ipfs_datasets_py && python -m pytest -q "
        "tests/unit/logic/test_compositional_verification_public_api.py"
    ) in projection
    scoped_command = next(
        line.removeprefix("- Validation: ")
        for line in projection.splitlines()
        if line.startswith("- Validation: ")
    )
    assert validation_command_repository_root(scoped_command) == "ipfs_datasets_py"
    assert (
        "- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py, "
        "ipfs_datasets_py/tests/unit/logic/"
        "test_compositional_verification_public_api.py"
    ) in projection
    assert "- Outputs: ipfs_datasets_py/logic/verification_api.py" not in projection
    assert "- Validation: 'python -m pytest" not in projection


def test_bridge_projection_preserves_database_identity_through_scoped_preflight(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "checkout"
    nested_repository = repository_root / "ipfs_datasets_py"
    target = "tests/unit/logic/test_compositional_verification_public_api.py"
    target_path = nested_repository / target
    target_path.parent.mkdir(parents=True)
    target_payload = b"def test_public_api():\n    assert True\n"
    target_path.write_bytes(target_payload)
    (nested_repository / ".git").write_text(
        "gitdir: ../.git/modules/ipfs_datasets_py\n",
        encoding="utf-8",
    )
    setup_payload = (
        b"from setuptools import setup\n"
        b"setup(extras_require={'lgcvf-validation': ['pytest']})\n"
    )
    (nested_repository / "setup.py").write_bytes(setup_payload)
    scoped_requirements = ["pytest"]
    scoped_requirements_sha256 = hashlib.sha256(
        json.dumps(
            scoped_requirements,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    projected_validation = f"cd ipfs_datasets_py && python -m pytest -q {target}"
    (nested_repository / "pyproject.toml").write_text(
        "\n".join(
            (
                "[project]",
                'name = "bridge-identity-fixture"',
                'version = "0.0.0"',
                'requires-python = ">=3.12"',
                'dynamic = ["dependencies"]',
                "",
                "[tool.ipfs-accelerate-agent-supervisor.project-dependency-preflight]",
                'schema = "ipfs_accelerate_py/agent-supervisor/'
                'scoped-project-dependency-preflight@3"',
                'requires-python = ">=3.12"',
                "authority = { file = \"setup.py\", sha256 = \""
                + hashlib.sha256(setup_payload).hexdigest()
                + "\", extra = \"lgcvf-validation\", "
                "extra-requirements-sha256 = \""
                + scoped_requirements_sha256
                + "\" }",
                "",
                "[[tool.ipfs-accelerate-agent-supervisor."
                "project-dependency-preflight.targets]]",
                f'target = "{target}"',
                'validation-command-sha256 = "'
                + hashlib.sha256(projected_validation.encode("utf-8")).hexdigest()
                + '"',
                'requirements = ["pytest"]',
                'task = { board-namespace = "bridge-authority-v1", '
                'canonical-task-cid = "task:cid:004", declared-output = "'
                f'ipfs_datasets_py/{target}" }}',
                'baseline = { state = "present", sha256 = "'
                + hashlib.sha256(target_payload).hexdigest()
                + '" }',
                "",
            )
        ),
        encoding="utf-8",
    )

    record = _owned_record("ipfs_datasets_py")
    record.body = {**record.body, "board_namespace": "bridge-authority-v1"}
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: _CompletingPortal(paths, alias),
        repository_root=repository_root,
        worktree_submodule_paths=("ipfs_datasets_py",),
        task_header_prefix="## LGSWF-",
    )

    bridge.run_provider(_attempt())
    projection_path = next((tmp_path / "attempts").glob("*/task-projection.md"))
    parsed = parse_task_text(
        projection_path.read_text(encoding="utf-8"),
        path=projection_path,
        task_header_prefix="## LGSWF-",
    )

    assert len(parsed) == 1
    task = parsed[0]
    assert task.canonical_task_cid == _attempt().task_cid
    assert task.canonical_task_key.startswith("task/v1/")
    receipt = preflight_validation_project_dependencies(
        repository_root,
        task.validation,
        task_authority={
            "board_namespace": task.board_namespace,
            "canonical_task_cid": task.canonical_task_cid,
            "declared_outputs": list(task_declared_output_paths(task)),
        },
    )
    assert receipt["passed"] is True
    assert (
        receipt["reason"]
        == "approved_validation_environment_satisfies_project_dependencies"
    )


def test_bridge_rejects_task_body_cid_conflicting_with_database_authority(
    tmp_path: Path,
) -> None:
    record = _record()
    record.body = {**record.body, "canonical_task_cid": "task:cid:forged"}
    factory_calls: list[str] = []

    def factory(paths: object, alias: str) -> _CompletingPortal:
        factory_calls.append(alias)
        return _CompletingPortal(paths, alias)

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=factory,
    )

    with pytest.raises(
        DatabasePortalBridgeError,
        match="task body conflicts with its canonical CID",
    ):
        bridge.run_provider(_attempt())
    assert factory_calls == []


def test_bridge_preserves_root_repository_output_paths(tmp_path: Path) -> None:
    record = _record()
    record.outputs = (
        {"path": "ipfs_accelerate_py/agent_supervisor/runtime.py"},
        {"path": "test/api/test_runtime.py"},
    )
    record.body = {
        **record.body,
        "owning_repository": "ipfs_accelerate_py",
        "markdown_metadata": {"owning_repository": "ipfs_accelerate_py"},
    }
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: _CompletingPortal(paths, alias),
    )

    bridge.run_provider(_attempt())

    projection_path = next((tmp_path / "attempts").glob("*/task-projection.md"))
    projection = projection_path.read_text(encoding="utf-8")
    assert (
        "- Outputs: ipfs_accelerate_py/agent_supervisor/runtime.py, "
        "test/api/test_runtime.py"
    ) in projection
    assert "ipfs_accelerate_py/ipfs_accelerate_py" not in projection


@pytest.mark.parametrize(
    "output",
    (
        "/tmp/escape.py",
        "../escape.py",
        "pkg/../../escape.py",
        "./pkg/module.py",
        "pkg//module.py",
        "pkg/one.py,pkg/two.py",
        "pkg\\module.py",
        "C:/escape.py",
    ),
)
def test_bridge_rejects_output_paths_that_cannot_be_scoped_losslessly(
    tmp_path: Path,
    output: str,
) -> None:
    repository_root = tmp_path / "checkout"
    (repository_root / "ipfs_datasets_py" / ".git").mkdir(parents=True)
    record = _owned_record("ipfs_datasets_py")
    record.outputs = ({"path": output},)
    factory_calls: list[str] = []

    def factory(paths: object, alias: str) -> _CompletingPortal:
        factory_calls.append(alias)
        return _CompletingPortal(paths, alias)

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=factory,
        repository_root=repository_root,
        worktree_submodule_paths=("ipfs_datasets_py",),
    )

    with pytest.raises(
        DatabasePortalBridgeError,
        match="task output path identity is unsafe or ambiguous",
    ):
        bridge.run_provider(_attempt())
    assert factory_calls == []


def test_bridge_rejects_ambiguous_output_mapping(tmp_path: Path) -> None:
    record = _record()
    record.outputs = ({"path": "src/one.py", "output": "src/two.py"},)
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: _CompletingPortal(paths, alias),
    )

    with pytest.raises(
        DatabasePortalBridgeError,
        match="task output mapping has ambiguous path identities",
    ):
        bridge.run_provider(_attempt())


def test_bridge_nested_output_projection_binding_is_stable(tmp_path: Path) -> None:
    repository_root = tmp_path / "checkout"
    (repository_root / "ipfs_datasets_py" / ".git").mkdir(parents=True)
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_owned_record("ipfs_datasets_py")),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: _CompletingPortal(paths, alias),
        repository_root=repository_root,
        worktree_submodule_paths=("ipfs_datasets_py",),
    )

    first = bridge.run_provider(_attempt())
    binding_path = next(
        (tmp_path / "attempts").glob("*/database-attempt-binding.json")
    )
    first_binding = binding_path.read_bytes()
    second = bridge.run_provider(_attempt())

    assert second["binding_id"] == first["binding_id"]
    assert binding_path.read_bytes() == first_binding


def test_bridge_projects_multiple_validations_under_one_repository_transition(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "checkout"
    (repository_root / "ipfs_datasets_py" / ".git").mkdir(parents=True)
    record = _owned_record("ipfs_datasets_py")
    record.validations = (
        {"argv": ["python -m pytest -q tests/unit/logic/test_public_api.py"]},
        {"argv": ["python -m pytest -q tests/unit/logic/test_differential.py"]},
    )
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: _CompletingPortal(paths, alias),
        repository_root=repository_root,
        worktree_submodule_paths=("ipfs_datasets_py",),
    )

    bridge.run_provider(_attempt())

    projection_path = next((tmp_path / "attempts").glob("*/task-projection.md"))
    projection = projection_path.read_text(encoding="utf-8")
    command = next(
        line.removeprefix("- Validation: ")
        for line in projection.splitlines()
        if line.startswith("- Validation: ")
    )
    assert command.count("cd ipfs_datasets_py") == 1
    assert "test_public_api.py && python -m pytest" in command
    assert validation_command_repository_root(command) == "ipfs_datasets_py"


@pytest.mark.parametrize(
    "argv",
    (
        ["python -m pytest -q tests/unit/test_safe.py\n&& rm -rf target"],
        ["python -m pytest -q tests/unit/test_safe.py\x00"],
        [" python -m pytest -q tests/unit/test_safe.py"],
        ["python", 7, "-m", "pytest"],
        [],
    ),
)
def test_bridge_rejects_noncanonical_validation_argv_before_projection(
    tmp_path: Path,
    argv: list[object],
) -> None:
    record = _record()
    record.validations = ({"argv": argv},)
    factory_calls: list[str] = []

    def factory(paths: object, alias: str) -> _CompletingPortal:
        factory_calls.append(alias)
        return _CompletingPortal(paths, alias)

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=factory,
    )

    with pytest.raises(DatabasePortalBridgeError, match="validation argv"):
        bridge.run_provider(_attempt())
    assert factory_calls == []
    assert not list((tmp_path / "attempts").glob("*/task-projection.md"))


@pytest.mark.parametrize(
    ("owner", "message"),
    (
        ("../outside", "owning repository metadata is unsafe"),
        ("other_repository", "not a configured worktree submodule"),
    ),
)
def test_bridge_rejects_unsafe_or_unconfigured_owning_repository(
    tmp_path: Path,
    owner: str,
    message: str,
) -> None:
    repository_root = tmp_path / "checkout"
    (repository_root / "ipfs_datasets_py" / ".git").mkdir(parents=True)
    # Even an initialized nested Git repository is not authority unless it is
    # in the supervisor's configured worktree-submodule allowlist.
    (repository_root / "other_repository" / ".git").mkdir(parents=True)
    factory_calls: list[str] = []

    def factory(paths: object, alias: str) -> _CompletingPortal:
        factory_calls.append(alias)
        return _CompletingPortal(paths, alias)

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_owned_record(owner)),
        attempt_root=tmp_path / "attempts",
        portal_factory=factory,
        repository_root=repository_root,
        worktree_submodule_paths=("ipfs_datasets_py",),
    )

    with pytest.raises(DatabasePortalBridgeError, match=message):
        bridge.run_provider(_attempt())
    assert factory_calls == []


def test_bridge_rejects_validation_root_conflicting_with_owner(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "checkout"
    (repository_root / "ipfs_datasets_py" / ".git").mkdir(parents=True)
    record = _owned_record("ipfs_datasets_py")
    record.validations = ({"argv": ["cd other_repository && python -m pytest -q"]},)
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: _CompletingPortal(paths, alias),
        repository_root=repository_root,
        worktree_submodule_paths=("ipfs_datasets_py",),
    )

    with pytest.raises(
        DatabasePortalBridgeError,
        match="repository root conflicts with owning repository",
    ):
        bridge.run_provider(_attempt())


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_production_database_daemon_cannot_complete_with_default_noops(
    tmp_path: Path,
) -> None:
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        owner_session_id="session:fail-closed",
        authority_mode="embedded_exclusive",
        task_source_kind="duckdb",
        require_real_execution=True,
    )
    try:
        daemon.materialize_population(
            {
                "repository_tree_id": "tree:bridge",
                "tasks": [
                    {
                        "task_cid": "task:cid:004",
                        "task_id": "LGSWF-004",
                        "goal_cid": "goal:inventory",
                        "status": "ready",
                        "priority": "P0",
                        "ordinal": 4,
                        "title": "Inventory",
                    }
                ],
            }
        )
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="no provider executor",
        ):
            daemon.run_once()
        task = daemon.task_source.get_task("task:cid:004")
        assert task is not None
        assert task.status != "completed"
        assert (
            daemon.provider_invocation_recorded(
                daemon.list_running_attempts()[0].attempt_id,
                idempotency_key=f"provider:{daemon.list_running_attempts()[0].attempt_id}",
            )
            is None
        )
    finally:
        daemon.close()


def test_quack_mode_refuses_direct_duckdb_execution(tmp_path: Path) -> None:
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="loopback quack:",
    ):
        DatabaseImplementationDaemon(
            database_path=tmp_path / "control.duckdb",
            authority_mode="quack",
            task_source_kind="duckdb",
        )


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_configured_production_runner_binds_real_portal_bridge(
    tmp_path: Path,
) -> None:
    args = parse_args(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "embedded_exclusive",
            "--database-path",
            str(tmp_path / "control.duckdb"),
            "--todo-path",
            str(tmp_path / "canonical-board.md"),
            "--state-dir",
            str(tmp_path / "state"),
            "--state-prefix",
            "lgswf",
            "--worktree-root",
            ".worktrees",
            "--implement",
            "--once",
        ]
    )
    daemon, _context = build_portal_implementation_daemon_from_args(
        args,
        repo_root=tmp_path,
    )
    try:
        assert isinstance(daemon, DatabaseImplementationDaemon)
        assert daemon.require_real_execution is True
        assert daemon.execution_callbacks_bound is True
        assert daemon.markdown_path is None
        assert daemon.markdown_status_write_count == 0
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_bridge_routes_only_owned_missing_output_quarantine_and_replays_completion(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "recovery@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Recovery Test"],
        cwd=repo,
        check=True,
    )
    output = repo / "inventory" / "result.json"
    output.parent.mkdir(parents=True)
    output.write_text('{"sealed":true}\n', encoding="utf-8")
    subprocess.run(["git", "add", "inventory/result.json"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "sealed candidate"], cwd=repo, check=True)
    candidate = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    candidate_tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    candidate_blob = subprocess.run(
        ["git", "rev-parse", "HEAD:inventory/result.json"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    record = _record()
    record.status = "blocked"
    task_source = _TaskSource(record)
    attempt_root = tmp_path / "lane-0-attempts"
    seed_bridge = DatabasePortalExecutionBridge(
        task_source=task_source,
        attempt_root=attempt_root,
        portal_factory=lambda _paths, _alias: None,
        repository_root=repo,
        task_header_prefix="## LGSWF-",
    )
    paths, binding = seed_bridge._ensure_attempt_projection(_attempt(), record)
    [projected_task] = parse_task_text(
        paths.task_projection.read_text(encoding="utf-8"),
        path=paths.task_projection,
        task_header_prefix="## LGSWF-",
    )

    repository_id = checkout_repository_id(repo)
    queue = MergeQueue(
        tmp_path / "merge-queue",
        target_repository_id=repository_id,
        target_branch="main",
        require_target_binding=True,
    )

    def request_metadata(
        commit: str,
        *,
        owned_paths: object = paths,
    ) -> dict[str, object]:
        return {
            "schema": "ipfs_accelerate_py/agent-supervisor/merge-candidate@3",
            "target_binding_schema": (
                "ipfs_accelerate_py/agent-supervisor/merge-target-binding@1"
            ),
            "target_repository_id": repository_id,
            "target_branch": "main",
            "implementation_commit": commit,
            "todo_path": str(owned_paths.task_projection),
            "state_path": str(owned_paths.state),
            "strategy_path": str(owned_paths.strategy),
            "events_path": str(owned_paths.events),
            "repo_root": str(repo.absolute()),
            "task_header_prefix": "## LGSWF-",
            "task": asdict(projected_task),
            "completion_task_cids": {"LGSWF-004": "task:cid:004"},
            "changed_submodule_paths": [],
        }

    def quarantine(
        commit: str,
        metadata: dict[str, object],
        *,
        reason: str,
    ) -> object:
        request = queue.enqueue(
            branch_name=f"implementation/{commit[:8]}",
            task_id="LGSWF-004",
            canonical_task_id="task:cid:004",
            canonical_task_key=str(projected_task.canonical_task_key),
            commit_sha=commit,
            metadata=metadata,
        )
        claimed = queue.claim_pending_request(
            request.request_id,
            consumer_id=f"fixture:{commit[:8]}",
        )
        assert claimed is not None
        queue.quarantine(claimed, reason=reason)
        stored = queue.get(request.request_id)
        assert stored is not None
        return stored

    ordinary = quarantine(
        "a" * 40,
        request_metadata("a" * 40),
        reason="merge_conflict",
    )

    foreign_root = tmp_path / "lane-1-attempts" / paths.root.name
    foreign_root.mkdir(parents=True)
    foreign_projection = foreign_root / paths.task_projection.name
    foreign_binding = foreign_root / paths.binding.name
    foreign_projection.write_bytes(paths.task_projection.read_bytes())
    foreign_binding.write_bytes(paths.binding.read_bytes())
    foreign_paths = SimpleNamespace(
        task_projection=foreign_projection,
        state=foreign_root / paths.state.name,
        strategy=foreign_root / paths.strategy.name,
        events=foreign_root / paths.events.name,
    )
    foreign = quarantine(
        "b" * 40,
        request_metadata("b" * 40, owned_paths=foreign_paths),
        reason="post_merge_declared_outputs_missing",
    )

    unsealed_root = attempt_root / ("0" * 24)
    unsealed_root.mkdir(parents=True)
    unsealed_projection = unsealed_root / paths.task_projection.name
    unsealed_projection.write_bytes(paths.task_projection.read_bytes())
    unsealed_paths = SimpleNamespace(
        task_projection=unsealed_projection,
        state=unsealed_root / paths.state.name,
        strategy=unsealed_root / paths.strategy.name,
        events=unsealed_root / paths.events.name,
    )
    unsealed = quarantine(
        "c" * 40,
        request_metadata("c" * 40, owned_paths=unsealed_paths),
        reason="post_merge_declared_outputs_missing",
    )
    selected = quarantine(
        candidate,
        request_metadata(candidate),
        reason="post_merge_declared_outputs_missing",
    )
    revived = queue.revive_quarantined(
        selected.request_id,
        reason="fixture selected exact database recovery",
        reset_failures=True,
    )
    assert revived is not None and revived.status == "pending"
    abandoned = queue.claim_pending_request(
        selected.request_id,
        consumer_id="merge-train:999999:dead-fixture",
    )
    assert abandoned is not None and abandoned.status == "processing"
    assert [
        request.request_id for request in queue.processing_requests()
    ] == [selected.request_id]

    repair_receipt: dict[str, object] = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "post-merge-declared-output-repair@1"
        ),
        "task_ids": ["LGSWF-004"],
        "candidate_commit": candidate,
        "candidate_tree": candidate_tree,
        "baseline_commit": candidate,
        "failed_integration_commit": candidate,
        "repair_parent_commit": candidate,
        "repair_commit": candidate,
        "repair_tree": candidate_tree,
        "entries": [
            {
                "path": "inventory/result.json",
                "mode": "100644",
                "object_type": "blob",
                "object_id": candidate_blob,
            }
        ],
        "validation": [
            {
                "task_id": "LGSWF-004",
                "passed": True,
                "returncode": 0,
                "validation_result_digests": [],
                "command_count": 0,
                "log_sha256": "e" * 64,
            }
        ],
        "rollback_target": candidate,
    }
    repair_receipt["receipt_id"] = content_identity(repair_receipt)
    portal_calls: list[str] = []
    requalification_heads: list[str] = []

    class RecoveryPortal:
        def __init__(self) -> None:
            self.merge_queue = queue
            self.repo_root = repo.absolute()
            self.resolved_merge_target_branch = "main"
            self.formal_verification_policy = None
            self.proof_gate = None
            self.proof_cache_dir = tmp_path / "proof-cache"
            self.decision_runtime = None
            self.implementation_cancelled = None

        @staticmethod
        def _merge_train_callback(request: object) -> dict[str, object]:
            assert request.request_id == selected.request_id
            return {
                "merged": True,
                "reason": "post_merge_declared_outputs_repaired",
                "post_merge_declared_output_repair": {
                    "passed": True,
                    "reason": "post_merge_declared_outputs_repaired",
                    "receipt": repair_receipt,
                },
            }

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
            assert task.task_id == "LGSWF-004"
            assert force_uncached is True
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=workspace,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            requalification_heads.append(head)
            assert (workspace / "inventory/result.json").read_text(
                encoding="utf-8"
            ) == '{"sealed":true}\n'
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("fresh current-tree validation passed\n")
            return {
                "attempted": True,
                "passed": True,
                "returncode": 0,
                "results": [
                    {
                        "validation_result_digest": (
                            "sha256:"
                            + hashlib.sha256(head.encode("ascii")).hexdigest()
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
                ["git", "worktree", "remove", "--force", str(workspace)],
                cwd=repo,
                check=False,
                capture_output=True,
                text=True,
            )
            return {"cleaned": removed.returncode == 0}

        @staticmethod
        def close_event_runtime() -> None:
            return None

    def fresh_bridge() -> DatabasePortalExecutionBridge:
        return DatabasePortalExecutionBridge(
            task_source=task_source,
            attempt_root=attempt_root,
            portal_factory=lambda _paths, alias: (
                portal_calls.append(alias) or RecoveryPortal()
            ),
            repository_root=repo,
            merge_queue=queue,
            merge_target_branch="main",
            task_header_prefix="## LGSWF-",
        )

    bridge = fresh_bridge()
    recovered_evidence: list[dict[str, object]] = []
    competing_train = MergeTrain(repo, queue, target_branch="main")
    recovery_lease_observations: list[bool] = []

    class DatabaseAuthority:
        crash_after_queue_completion = True
        latest_source_attempt_id = str(binding["attempt_id"])
        preauthorization_sources: list[dict[str, object]] = []

        @staticmethod
        def _database_portal_evidence_digest(value: object) -> str:
            encoded = json.dumps(
                value,
                separators=(",", ":"),
                sort_keys=True,
                default=str,
            ).encode("utf-8")
            return "sha256:" + hashlib.sha256(encoded).hexdigest()

        def preauthorize_post_merge_declared_output_recovery(
            self,
            source: object,
        ) -> dict[str, object]:
            assert isinstance(source, dict)
            source_dict = dict(source)
            self.preauthorization_sources.append(source_dict)
            if (
                source_dict["source_attempt_id"]
                != self.latest_source_attempt_id
            ):
                raise DatabaseImplementationConflictError(
                    "fixture superseded source attempt"
                )
            result: dict[str, object] = {
                **source_dict,
                "authorized": True,
                "task_status": "blocked",
            }
            result["authorization_id"] = (
                self._database_portal_evidence_digest(result)
            )
            return result

        def recover_blocked_post_merge_declared_outputs(
            self,
            evidence: object,
        ) -> dict[str, object]:
            competing_acquired, _ = competing_train.run_under_consumer_lease(
                lambda: None
            )
            recovery_lease_observations.append(competing_acquired)
            assert competing_acquired is False
            recovered_evidence.append(dict(evidence))
            if self.crash_after_queue_completion:
                self.crash_after_queue_completion = False
                raise RuntimeError("fixture crash after queue completion")
            return {
                "attempted": True,
                "recovered": True,
                "changed": True,
                "status": "retrying",
                "write_count": 2,
            }

    authority = DatabaseAuthority()
    blocker = MergeTrain(repo, queue, target_branch="main")
    with blocker._consumer_lease() as acquired:
        assert acquired is True
        assert bridge.recover_post_merge_declared_outputs(authority) is None
        assert portal_calls == []
    with pytest.raises(RuntimeError, match="fixture crash"):
        bridge.recover_post_merge_declared_outputs(authority)
    completed = queue.get(selected.request_id)
    assert completed is not None and completed.status == "completed"

    output.write_text('{"sealed":false}\n', encoding="utf-8")
    subprocess.run(["git", "add", "inventory/result.json"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "advance past repair"], cwd=repo, check=True)
    assert bridge.recover_post_merge_declared_outputs(authority) is None
    assert len(recovered_evidence) == 1
    assert record.status == "blocked"

    output.unlink()
    subprocess.run(["git", "add", "inventory/result.json"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "remove repaired output"], cwd=repo, check=True)
    missing_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    missing_bridge = fresh_bridge()
    assert missing_bridge.recover_post_merge_declared_outputs(authority) is None
    assert fresh_bridge().recover_post_merge_declared_outputs(authority) is None
    assert requalification_heads == []

    subprocess.run(
        ["git", "restore", "--source", candidate, "--", "inventory/result.json"],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "add", "inventory/result.json"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "restore exact repaired output"], cwd=repo, check=True)
    descendant_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert descendant_head not in {candidate, missing_head}
    # Wrap the completed cursor after the missing-output page before adding a
    # new full page of newer history.
    assert fresh_bridge().recover_post_merge_declared_outputs(authority) is None

    # Put the unresolved completion behind a full 256-row page of newer,
    # schema-matching history.  Rows are inserted in one hermetic fixture
    # transaction so the test measures bridge pagination, not queue writes.
    request_clock = int(selected.request_id.split("-", 1)[0])
    decoy_metadata = json.dumps(
        completed.metadata,
        sort_keys=True,
        separators=(",", ":"),
    )
    stale_decoy_index = 256
    stale_completed_request_id = (
        f"{request_clock + stale_decoy_index + 1}-"
        f"{100000 + stale_decoy_index}-decoy"
    )
    decoy_rows = [
        (
            f"{request_clock + index + 1}-{100000 + index}-decoy",
            f"implementation/decoy-{index}",
            (
                str(selected.task_id)
                if index == stale_decoy_index
                else f"DECOY-{index}"
            ),
            "P2",
            "",
            float(index + 2),
            1,
            decoy_metadata,
            candidate,
            (
                str(selected.canonical_task_id)
                if index == stale_decoy_index
                else f"task:decoy:{index}"
            ),
            (
                str(selected.canonical_task_key)
                if index == stale_decoy_index
                else f"task/v1/decoy-{index}"
            ),
            f"decoy:{index}",
            "completed",
            0.0,
            "",
            0,
            "",
            "",
            2,
            0.0,
            float(index + 2),
            float(index + 2),
        )
        for index in range(257)
    ]
    with queue._connect() as connection:
        connection.executemany(
            "INSERT INTO merge_requests VALUES ("
            + ",".join("?" for _ in range(22))
            + ")",
            decoy_rows,
        )
        connection.commit()
    queue_queries = {
        name: 0
        for name in (
            "completed_requests",
            "pending_requests",
            "quarantined_requests",
            "processing_requests",
        )
    }
    for operation in tuple(queue_queries):
        original = getattr(queue, operation)

        def counted_snapshot(
            *,
            _operation: str = operation,
            _original: object = original,
            **kwargs: object,
        ) -> object:
            queue_queries[_operation] += 1
            return _original(**kwargs)

        setattr(queue, operation, counted_snapshot)

    def assert_one_page_per_stage(before: dict[str, int]) -> None:
        assert all(
            queue_queries[operation] - before[operation] <= 1
            for operation in queue_queries
        )

    before = dict(queue_queries)
    portal_count_before_stale_page = len(portal_calls)
    authority.latest_source_attempt_id = "attempt:superseding"
    assert fresh_bridge().recover_post_merge_declared_outputs(authority) is None
    assert_one_page_per_stage(before)
    assert queue_queries["completed_requests"] - before["completed_requests"] == 1
    assert requalification_heads == []
    assert len(portal_calls) == portal_count_before_stale_page
    assert any(
        source["request_id"] == stale_completed_request_id
        for source in authority.preauthorization_sources
    )
    authority.latest_source_attempt_id = str(binding["attempt_id"])

    # The second fresh bridge resumes page two, validates once, publishes the
    # immutable requalification receipt, then crashes before the database CAS.
    authority.crash_after_queue_completion = True
    before = dict(queue_queries)
    with pytest.raises(RuntimeError, match="fixture crash"):
        fresh_bridge().recover_post_merge_declared_outputs(authority)
    assert_one_page_per_stage(before)
    assert queue_queries["completed_requests"] - before["completed_requests"] == 1
    first_requalification_evidence = dict(recovered_evidence[-1])
    assert requalification_heads == [descendant_head]

    # A reconstructed bridge replays byte-identical cached evidence.  It must
    # not instantiate Portal or append another validation/log receipt.
    before = dict(queue_queries)
    replay_bridge = fresh_bridge()
    result = replay_bridge.recover_post_merge_declared_outputs(authority)
    assert_one_page_per_stage(before)
    assert queue_queries["completed_requests"] - before["completed_requests"] == 1

    assert result is not None
    assert result["recovered"] is True
    assert result["write_count"] == 2
    assert portal_calls == ["LGSWF-004", "LGSWF-004"]
    assert recovery_lease_observations == [False, False, False]
    assert requalification_heads == [descendant_head]
    assert subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip() == descendant_head
    evidence = recovered_evidence[-1]
    assert evidence["source_attempt_id"] == binding["attempt_id"]
    assert evidence["source_claim_id"] == binding["claim_id"]
    assert evidence["source_lease_id"] == binding["lease_id"]
    assert evidence["source_fencing_token"] == binding["fencing_token"]
    assert evidence["source_fence_epoch"] == binding["fence_epoch"]
    assert evidence["source_binding_id"] == binding["binding_id"]
    assert evidence["source_projection_immutable_digest"] == binding[
        "projection_immutable_digest"
    ]
    assert evidence == first_requalification_evidence
    assert evidence["schema"] == (
        "ipfs_accelerate_py/agent-supervisor/"
        "database-post-merge-declared-output-requalification-recovery@1"
    )
    assert evidence["qualified_target_commit"] == descendant_head
    requalification = evidence["requalification_receipt"]
    assert requalification["schema"] == (
        "ipfs_accelerate_py.agent_supervisor."
        "post-merge-declared-output-requalification@1"
    )
    assert set(requalification) == {
        "schema",
        "task_ids",
        "candidate_commit",
        "source_repair_receipt_id",
        "source_repair_commit",
        "source_repair_receipt",
        "current_target_commit",
        "current_target_tree",
        "entries",
        "validation",
        "receipt_id",
    }
    assert requalification["source_repair_receipt_id"] == repair_receipt[
        "receipt_id"
    ]
    assert requalification["source_repair_commit"] == candidate
    assert requalification["source_repair_receipt"] == repair_receipt
    assert requalification["current_target_commit"] == descendant_head
    assert requalification["current_target_tree"] == subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert requalification["entries"] == repair_receipt["entries"]
    assert evidence["requalification_receipt_id"] == requalification[
        "receipt_id"
    ]
    assert queue.get(ordinary.request_id).status == "quarantined"
    assert queue.get(foreign.request_id).status == "quarantined"
    assert queue.get(unsealed.request_id).status == "quarantined"

    record.status = "in_progress"
    assert replay_bridge.recover_post_merge_declared_outputs(authority) is None
    assert len(recovered_evidence) == 3


def test_post_merge_recovery_cursor_writes_only_on_progress_or_wrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = object.__new__(DatabasePortalExecutionBridge)
    writes: list[dict[str, str]] = []
    monkeypatch.setattr(
        bridge,
        "_save_post_merge_recovery_cursors",
        lambda cursors: writes.append(dict(cursors)),
    )
    cursors = {"completed_requests": ""}

    bridge._advance_post_merge_recovery_cursor(
        cursors,
        "completed_requests",
        (),
    )
    assert writes == []

    page = (SimpleNamespace(request_id="request:1"),)
    bridge._advance_post_merge_recovery_cursor(
        cursors,
        "completed_requests",
        page,
    )
    bridge._advance_post_merge_recovery_cursor(
        cursors,
        "completed_requests",
        page,
    )
    assert writes == [{"completed_requests": "request:1"}]

    bridge._advance_post_merge_recovery_cursor(
        cursors,
        "completed_requests",
        (),
    )
    assert writes[-1] == {"completed_requests": ""}
