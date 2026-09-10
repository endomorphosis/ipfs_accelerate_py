"""Focused safety tests for database-authoritative Portal execution."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    database_portal_bridge as database_portal_bridge_module,
)
from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ContextCompilationReceipt,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import (
    ContextBudgetResolution,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    ProcessBirthIdentity,
    WorktreeLifecycleStore,
    current_process_birth,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DATABASE_PORTAL_CHECKOUT_CONTENTION_BACKOFF_SECONDS,
    DATABASE_PORTAL_CONSUMED_NO_PROGRESS_SCHEMA,
    DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA,
    DATABASE_PORTAL_RETRY_DEFERRAL_SCHEMA,
    DatabasePortalBridgeConsumedNoProgressError,
    DATABASE_PORTAL_VALIDATION_RETRY_SCHEMA,
    DATABASE_PORTAL_POOLED_WORKTREE_CREATE_FAILED_REASON,
    DATABASE_PORTAL_POOLED_WORKTREE_CREATE_RECOVERY_SCHEMA,
    DATABASE_PORTAL_QUACK_PREPROJECTION_RECOVERY_SCHEMA,
    DATABASE_PORTAL_VALIDATION_RETRY_SEED_CONFLICT_REASON,
    DATABASE_PORTAL_VALIDATION_RETRY_SEED_CONFLICT_RECOVERY_SCHEMA,
    DatabasePortalBridgeDeferred,
    DatabasePortalBridgeError,
    DatabasePortalCandidateRetry,
    DatabasePortalDeterministicReconciliationDeferred,
    DatabasePortalExecutionBridge,
    database_portal_consumed_no_progress_fingerprint,
    database_portal_task_contract_digest,
    _is_implementation_conflict,
    verify_database_portal_attempt_projection,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DATASETS_AUTHORITATIVE_STATE_SCHEMA_REVISION,
    DATABASE_FALSE_COMPLETION_REINTEGRATION_RECOVERY_SCHEMA,
    SEMANTIC_TRUTH_AUTHORITY_ENV,
    SEMANTIC_WRITER_POLICY_ENV,
    DatabaseImplementationAuthorityError,
    DatabaseImplementationDaemon,
    DatabaseTaskAttempt,
    ImplementationDiagnosticReceipt,
    PortalImplementationDaemon,
    parse_args,
    parse_task_file,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
    build_portal_implementation_daemon_from_args,
)
import hashlib
import subprocess
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import DATABASE_PORTAL_VALIDATION_RETRY_SCHEMA, DatabasePortalValidationRetry
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalTaskState, parse_task_text, task_declared_output_paths
from ipfs_accelerate_py.agent_supervisor.runtime.event_log import append_jsonl_event
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.validation.project_dependency_preflight import preflight_validation_project_dependencies
from ipfs_accelerate_py.agent_supervisor.validation.validation_commands import validation_command_repository_root


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
            "task_key": "task/v1/current-authority-inventory",
            "completion": "auto",
            "track": "analysis",
            "read_scope": ["ipfs_accelerate_py/agent_supervisor"],
            "write_scope": ["inventory/result.json"],
            "completion_contract": "Focused validation passes",
        },
    )


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
        json.dumps(
            {
                "authority_mode": "quack",
                "task_source_kind": "duckdb",
                "endpoint_secret_handle": "handle:test-portal-bridge",
                "quack_endpoint": "quack:127.0.0.1:45123",
                "store_id": "state/test-portal-bridge/control.duckdb",
                "store_generation": "1",
                "schema_revision": "1",
                "failover_policy": "fail_closed",
            }
        ),
    )
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "secret-token")
    portal = SimpleNamespace(
        _canonical_ref=lambda task: "task:cid:004",
        _implementation_untrusted_process_environment=(
            PortalImplementationDaemon._implementation_untrusted_process_environment
        ),
    )
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
    ordinary_environment = PortalImplementationDaemon._implementation_process_environment(
        portal,
        task,
        attempt=3,
        checkpoint_dir=tmp_path / "ordinary-checkpoint",
    )
    assert SEMANTIC_TRUTH_AUTHORITY_ENV not in ordinary_environment
    assert SEMANTIC_WRITER_POLICY_ENV not in ordinary_environment


class _TaskSource:
    def __init__(self, record: object) -> None:
        self.record = record

    def get_task(self, task_cid: str) -> object | None:
        return self.record if task_cid == "task:cid:004" else None

    def snapshot(self) -> object:
        return SimpleNamespace(repository_tree_id="tree:control-plane-current")


class _CompletingPortal:
    def __init__(self, paths: object, task_alias: str) -> None:
        self.paths = paths
        self.task_alias = task_alias
        self.closed = False

    def run_once(self) -> dict[str, object]:
        implementation = {
            "task_id": self.task_alias,
            "task_cid": "task:cid:004",
            "canonical_task_cid": "task:cid:004",
            "returncode": 0,
            "implementation_commit": "a" * 40,
        }
        text = self.paths.task_projection.read_text(encoding="utf-8")
        self.paths.task_projection.write_text(
            text.replace("- Status: ready", "- Status: completed"),
            encoding="utf-8",
        )
        self.paths.state.write_text(
            json.dumps(
                {
                    "last_implementation_task_id": self.task_alias,
                    "last_implementation_task_cid": "task:cid:004",
                    "last_implementation_returncode": 0,
                    "last_implementation_commit": "a" * 40,
                    "last_merge_returncode": 0,
                }
            ),
            encoding="utf-8",
        )
        append_jsonl_event(
            self.paths.events,
            "implementation_finished",
            implementation,
        )
        append_jsonl_event(
            self.paths.events,
            "task_completed",
            {
                "task_id": self.task_alias,
                "canonical_task_key": "task/v1/current-authority-inventory",
                "canonical_task_cid": "task:cid:004",
            },
        )
        return {
            "task_count": 1,
            "completed_count": 1,
            "active_task_id": self.task_alias,
            "implementation_result": {
                **implementation,
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


def _deterministic_record_and_authority() -> tuple[SimpleNamespace, dict[str, object]]:
    attempt = _attempt()
    record = _record()
    authority: dict[str, object] = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "false-completion-deterministic-claim-authority@1"
        ),
        "task_cid": attempt.task_cid,
        "attempt_id": attempt.attempt_id,
        "claim_id": attempt.claim_id,
        "lease_id": attempt.lease_id,
        "fencing_token": attempt.fencing_token,
        "fence_epoch": attempt.fence_epoch,
        "source_attempt_id": "attempt:source",
        "reintegration_evidence_id": "sha256:" + "b" * 64,
        "request_id": "request:integrated",
        "candidate_commit": "1" * 40,
        "target_commit": "2" * 40,
        "target_tree": "3" * 40,
        "preserved_unknown_receipt_id": "",
        "provider_dispatch_policy": "forbidden",
        "effect_execution_policy": "forbidden",
    }
    authority["authority_id"] = (
        database_portal_bridge_module._database_evidence_digest(authority)
    )
    record.body = {
        **record.body,
        "completion_receipt": {
            "operation": "database_claim",
            "attempt_id": attempt.attempt_id,
            "claim_id": attempt.claim_id,
            "lease_id": attempt.lease_id,
            "owner_session_id": attempt.owner_session_id,
            "fencing_token": attempt.fencing_token,
            "fence_epoch": attempt.fence_epoch,
            "attempt_number": attempt.attempt_number,
            "false_completion_reintegration_source_attempt_id": (
                authority["source_attempt_id"]
            ),
            "false_completion_reintegration_seed": {
                "schema": DATABASE_FALSE_COMPLETION_REINTEGRATION_RECOVERY_SCHEMA,
                "source_attempt_id": authority["source_attempt_id"],
                "evidence_id": authority["reintegration_evidence_id"],
                "request_id": authority["request_id"],
                "candidate_commit": authority["candidate_commit"],
                "reintegration_receipt": {
                    "schema": (
                        "ipfs_accelerate_py/agent-supervisor/"
                        "false-completion-reintegration-receipt@1"
                    ),
                    "receipt_id": "baguqeera" + "a" * 52,
                    "reintegration_target_commit": authority["target_commit"],
                    "reintegration_target_tree": authority["target_tree"],
                },
            },
        },
    }
    return record, authority


class _DeterministicCompletingPortal:
    def __init__(
        self,
        paths: object,
        task_alias: str,
        projection: dict[str, object],
        *,
        mutate: object = None,
        run_calls: list[str] | None = None,
    ) -> None:
        self.paths = paths
        self.task_alias = task_alias
        self.projection = projection
        self.mutate = mutate
        self.run_calls = run_calls if run_calls is not None else []
        self.closed = False

    def _implementation(self) -> dict[str, object]:
        pre_dispatch: dict[str, object] = {
            "kind": "false_completion_reintegration",
            "source_task_id": self.task_alias,
            "repair_task_id": "",
            "failure_kind": "false_completion_reintegration",
            "repair_task_cid": "task:cid:004",
            "authority_receipt_id": self.projection["receipt_id"],
            "authority_evidence_id": self.projection["evidence_id"],
            "target_commit": self.projection["target_commit"],
            "target_tree": self.projection["target_tree"],
            "preserved_unknown_receipt_id": self.projection[
                "preserved_unknown_receipt_id"
            ],
            "provider_dispatch_policy": "forbidden",
            "eligible": True,
            "provider_dispatched": False,
            "reason": "declared_validation_proved_existing_contract",
        }
        pre_dispatch["receipt_id"] = content_identity(pre_dispatch)
        implementation: dict[str, object] = {
            "task_id": self.task_alias,
            "task_cid": "task:cid:004",
            "canonical_task_cid": "task:cid:004",
            "attempt": 1,
            "returncode": 0,
            "attempt_consumed": True,
            "provider_dispatched": False,
            "implementation_commit": "",
            "execution_mode": "local-retry-proof",
            "commit_result": {"committed": False, "reason": "no_changes"},
            "merge_result": {"merged": False, "reason": "not_attempted"},
            "validation_result": {
                "attempted": True,
                "passed": True,
                "returncode": 0,
                "reason": "declared_validation_passed",
                "pre_dispatch_no_change": pre_dispatch,
            },
            "cleanup_result": {"cleaned": True, "reason": "removed"},
            "board_completion": {
                "complete": True,
                "pending_merge": False,
                "reason": "validated_no_change",
            },
        }
        if callable(self.mutate):
            self.mutate(implementation)
        return implementation

    def run_once(self) -> dict[str, object]:
        self.run_calls.append(self.task_alias)
        implementation = self._implementation()
        text = self.paths.task_projection.read_text(encoding="utf-8")
        self.paths.task_projection.write_text(
            text.replace("- Status: ready", "- Status: completed"),
            encoding="utf-8",
        )
        self.paths.state.write_text(
            json.dumps(
                {
                    "last_implementation_task_id": self.task_alias,
                    "last_implementation_task_cid": "task:cid:004",
                    "last_implementation_returncode": implementation[
                        "returncode"
                    ],
                    "last_implementation_commit": implementation[
                        "implementation_commit"
                    ],
                }
            ),
            encoding="utf-8",
        )
        append_jsonl_event(
            self.paths.events,
            "implementation_finished",
            implementation,
        )
        append_jsonl_event(
            self.paths.events,
            "task_completed",
            {
                "task_id": self.task_alias,
                "canonical_task_key": "task/v1/current-authority-inventory",
                "canonical_task_cid": "task:cid:004",
            },
        )
        return {"implementation_result": implementation}

    def close_event_runtime(self) -> None:
        self.closed = True


class _DeterministicReceiptOnlyCompletingPortal(
    _DeterministicCompletingPortal
):
    """Match Portal's real validated-no-change terminal event shape."""

    def __init__(
        self,
        *args: object,
        forge_receipt: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.forge_receipt = forge_receipt

    def run_once(self) -> dict[str, object]:
        self.run_calls.append(self.task_alias)
        implementation = self._implementation()
        canonical_task_cid = (
            "task:cid:forged" if self.forge_receipt else "task:cid:004"
        )
        implementation["board_namespace"] = "bridge-test"
        implementation["canonical_task_key"] = (
            "task/v1/current-authority-inventory"
        )
        implementation["todo_update_result"] = {
            "updated": True,
            "task_id": self.task_alias,
            "updated_task_ids": [self.task_alias],
            "missing_task_ids": [],
            "missing_status_task_ids": [],
            "completion_receipts": [
                {
                    "schema": (
                        "ipfs_accelerate_py.agent_supervisor."
                        "member_completion_receipt@1"
                    ),
                    "board_namespace": "bridge-test",
                    "canonical_task_cid": canonical_task_cid,
                    "canonical_task_key": (
                        "task/v1/current-authority-inventory"
                    ),
                    "status": "succeeded",
                    "task_id": self.task_alias,
                }
            ],
        }
        text = self.paths.task_projection.read_text(encoding="utf-8")
        self.paths.task_projection.write_text(
            text.replace("- Status: ready", "- Status: completed"),
            encoding="utf-8",
        )
        self.paths.state.write_text(
            json.dumps(
                {
                    "last_implementation_task_id": self.task_alias,
                    "last_implementation_task_cid": "task:cid:004",
                    "last_implementation_returncode": 0,
                    "last_implementation_commit": "",
                }
            ),
            encoding="utf-8",
        )
        append_jsonl_event(
            self.paths.events,
            "implementation_finished",
            implementation,
        )
        return {"implementation_result": implementation}


def _consumed_no_progress_result(
    paths: object,
    task_alias: str,
    *,
    forged_identity: str = "",
    returncode: int = 1,
    log_text: str = "provider output is untrusted\n",
    committed: bool = False,
) -> dict[str, object]:
    paths.implementation_logs.mkdir(parents=True, exist_ok=True)
    log_path = paths.implementation_logs / "lgswf-004-attempt-1.log"
    log_path.write_text(log_text, encoding="utf-8")
    budget = ContextBudgetResolution(
        supervisor_max_input_tokens=4096,
        provider_context_window=8192,
        provider_max_input_tokens=None,
        reserved_output_tokens=1024,
        reserved_tool_tokens=512,
    )
    context = ContextCompilationReceipt(
        repository_id="repository:bridge-test",
        tree_id="a" * 40,
        objective_id=task_alias,
        policy_id="policy:bridge-test",
        policy_revision="sha256:" + "b" * 64,
        stage="implementation",
        capsule_id="capsule:bridge-test",
        budget_resolution=budget,
        effective_input_limit=int(budget.effective_input_limit or 0),
        input_tokens=128,
        estimator_name="bridge-test",
        estimator_error_bps=0,
    )
    context_path = paths.implementation_logs / "lgswf-004-attempt-1-context-receipt.json"
    context_path.write_text(
        json.dumps(context.to_dict(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    diagnostic = ImplementationDiagnosticReceipt(
        prior_decision_id=context.receipt_id,
        repository_id="repository:bridge-test",
        tree_id="a" * 40,
        failure={
            "kind": "implementation_failure",
            "returncode": returncode,
            "validation": {
                "passed": True,
                "reason": "not_run",
                "returncode": 0,
            },
        },
        # Task metadata may name symbols even though no file candidate exists.
        changed_symbols=("requested_symbol_from_task_metadata",),
    )
    diagnostic_record = diagnostic.to_record()
    if forged_identity:
        diagnostic_record[forged_identity] = f"forged:{forged_identity}"
    (paths.implementation_logs / "lgswf-004-diagnostic-receipt.json").write_text(
        json.dumps(diagnostic_record, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    binding = json.loads(paths.binding.read_text(encoding="utf-8"))
    canonical_task_cid = str(binding["task_cid"])
    return {
        "implementation_result": {
            "task_id": task_alias,
            "task_cid": canonical_task_cid,
            "canonical_task_cid": canonical_task_cid,
            "attempt": 1,
            "returncode": returncode,
            "log_path": str(log_path),
            "context_receipt_path": str(context_path),
            "baseline_ref": "a" * 40,
            "implementation_commit": "c" * 40 if committed else "",
            "commit_result": {"committed": committed},
            "merge_result": {"merged": False, "reason": "not_attempted"},
            "board_completion": {
                "complete": False,
                "pending_merge": False,
                "reason": "implementation_or_validation_failed",
            },
            "validation_result": {
                "attempted": False,
                "passed": True,
                "reason": "not_run",
                "returncode": 0,
                "results": [],
            },
            "attempt_consumed": True,
            "provider_dispatched": True,
            "diagnostic_receipt_id": diagnostic.receipt_id,
        }
    }


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


def test_deterministic_reconciliation_admits_exact_local_validation_proof(
    tmp_path: Path,
) -> None:
    record, authority = _deterministic_record_and_authority()
    run_calls: list[str] = []
    bridge: DatabasePortalExecutionBridge

    def factory(paths: object, alias: str) -> _DeterministicCompletingPortal:
        projection = bridge._deterministic_reconciliation_projection_metadata(
            _attempt(),
            record,
        )
        assert projection is not None
        return _DeterministicCompletingPortal(
            paths,
            alias,
            projection,
            run_calls=run_calls,
        )

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=factory,
        max_passes=1,
    )

    settlement = bridge.run_deterministic_reconciliation(_attempt(), authority)

    assert settlement["accepted"] is True
    assert settlement["provider_dispatched"] is False
    assert settlement["effect_executed"] is False
    assert settlement["validation_result"]["provider_dispatched"] is False
    assert settlement["validation_result"]["effect_executed"] is False
    assert run_calls == ["LGSWF-004"]


def test_deterministic_reconciliation_admits_receipt_only_no_change_completion(
    tmp_path: Path,
) -> None:
    record, authority = _deterministic_record_and_authority()
    bridge: DatabasePortalExecutionBridge

    def factory(
        paths: object,
        alias: str,
    ) -> _DeterministicReceiptOnlyCompletingPortal:
        projection = bridge._deterministic_reconciliation_projection_metadata(
            _attempt(),
            record,
        )
        assert projection is not None
        return _DeterministicReceiptOnlyCompletingPortal(
            paths,
            alias,
            projection,
        )

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=factory,
        max_passes=1,
    )

    settlement = bridge.run_deterministic_reconciliation(_attempt(), authority)

    assert settlement["accepted"] is True
    assert settlement["provider_dispatched"] is False
    assert settlement["effect_executed"] is False


def test_deterministic_reconciliation_rejects_forged_receipt_only_completion(
    tmp_path: Path,
) -> None:
    record, authority = _deterministic_record_and_authority()
    bridge: DatabasePortalExecutionBridge

    def factory(
        paths: object,
        alias: str,
    ) -> _DeterministicReceiptOnlyCompletingPortal:
        projection = bridge._deterministic_reconciliation_projection_metadata(
            _attempt(),
            record,
        )
        assert projection is not None
        return _DeterministicReceiptOnlyCompletingPortal(
            paths,
            alias,
            projection,
            forge_receipt=True,
        )

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=factory,
        max_passes=1,
    )

    with pytest.raises(
        DatabasePortalDeterministicReconciliationDeferred,
    ) as caught:
        bridge.run_deterministic_reconciliation(_attempt(), authority)

    assert caught.value.reason == (
        "deterministic_reconciliation_replay_proof_failed"
    )


def test_deterministic_reconciliation_replays_durable_implementation_after_write_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record, authority = _deterministic_record_and_authority()
    run_calls: list[str] = []
    factory_calls: list[str] = []
    bridge: DatabasePortalExecutionBridge

    def factory(paths: object, alias: str) -> _DeterministicCompletingPortal:
        factory_calls.append(alias)
        projection = bridge._deterministic_reconciliation_projection_metadata(
            _attempt(),
            record,
        )
        assert projection is not None
        return _DeterministicCompletingPortal(
            paths,
            alias,
            projection,
            run_calls=run_calls,
        )

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=factory,
        max_passes=1,
    )
    real_atomic_write_once = database_portal_bridge_module._atomic_write_once
    crashed = False

    def crash_before_settlement(path: Path, payload: bytes) -> object:
        nonlocal crashed
        if (
            path.name == "false-completion-deterministic-settlement.json"
            and not crashed
        ):
            crashed = True
            raise OSError("simulated crash before settlement receipt durability")
        return real_atomic_write_once(path, payload)

    monkeypatch.setattr(
        database_portal_bridge_module,
        "_atomic_write_once",
        crash_before_settlement,
    )
    with pytest.raises(OSError, match="simulated crash"):
        bridge.run_deterministic_reconciliation(_attempt(), authority)

    monkeypatch.setattr(
        database_portal_bridge_module,
        "_atomic_write_once",
        real_atomic_write_once,
    )
    settlement = bridge.run_deterministic_reconciliation(_attempt(), authority)

    assert settlement["accepted"] is True
    assert run_calls == ["LGSWF-004"]
    # Terminal replay occurs before Portal construction and carries the exact
    # local implementation proof instead of an empty portal_passes list.
    assert factory_calls == ["LGSWF-004"]
    passes = settlement["portal_acceptance_receipt"]["portal_evidence"][
        "portal_passes"
    ]
    assert len(passes) == 1
    assert passes[0]["implementation"]["validation_result"][
        "pre_dispatch_no_change"
    ]["reason"] == "declared_validation_proved_existing_contract"


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    (
        (
            "validation_failed",
            "deterministic_reconciliation_validation_failed",
        ),
        (
            "target_mismatch",
            "deterministic_reconciliation_target_mismatch",
        ),
        (
            "proof_missing",
            "deterministic_reconciliation_replay_proof_failed",
        ),
    ),
)
def test_deterministic_reconciliation_types_closed_zero_provider_failures(
    tmp_path: Path,
    mutation: str,
    expected_reason: str,
) -> None:
    record, authority = _deterministic_record_and_authority()
    bridge: DatabasePortalExecutionBridge

    def mutate(implementation: dict[str, object]) -> None:
        validation = implementation["validation_result"]
        assert isinstance(validation, dict)
        pre_dispatch = validation.get("pre_dispatch_no_change")
        assert isinstance(pre_dispatch, dict)
        if mutation == "validation_failed":
            implementation["returncode"] = 1
            validation.update(
                {
                    "passed": False,
                    "returncode": 1,
                    "reason": "declared_validation_failed",
                }
            )
            pre_dispatch["reason"] = (
                "reconciliation_declared_validation_failed"
            )
            pre_dispatch.pop("receipt_id", None)
        elif mutation == "target_mismatch":
            pre_dispatch["target_commit"] = "4" * 40
            pre_dispatch_body = dict(pre_dispatch)
            pre_dispatch_body.pop("receipt_id", None)
            pre_dispatch["receipt_id"] = content_identity(pre_dispatch_body)
        elif mutation == "proof_missing":
            validation.pop("pre_dispatch_no_change")

    def factory(paths: object, alias: str) -> _DeterministicCompletingPortal:
        projection = bridge._deterministic_reconciliation_projection_metadata(
            _attempt(),
            record,
        )
        assert projection is not None
        return _DeterministicCompletingPortal(
            paths,
            alias,
            projection,
            mutate=mutate,
        )

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=factory,
        max_passes=1,
    )

    with pytest.raises(
        DatabasePortalDeterministicReconciliationDeferred
    ) as caught:
        bridge.run_deterministic_reconciliation(_attempt(), authority)

    assert caught.value.reason == expected_reason
    assert caught.value.provider_dispatched is False
    assert caught.value.attempt_consumed is False
    assert caught.value.effect_executed is False
    assert caught.value.source_reason


def test_deterministic_reconciliation_does_not_type_observed_provider_dispatch(
    tmp_path: Path,
) -> None:
    record, authority = _deterministic_record_and_authority()
    bridge: DatabasePortalExecutionBridge

    def mutate(implementation: dict[str, object]) -> None:
        implementation["provider_dispatched"] = True

    def factory(paths: object, alias: str) -> _DeterministicCompletingPortal:
        projection = bridge._deterministic_reconciliation_projection_metadata(
            _attempt(),
            record,
        )
        assert projection is not None
        return _DeterministicCompletingPortal(
            paths,
            alias,
            projection,
            mutate=mutate,
        )

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=factory,
        max_passes=1,
    )

    with pytest.raises(DatabasePortalBridgeError) as caught:
        bridge.run_deterministic_reconciliation(_attempt(), authority)

    assert not isinstance(
        caught.value,
        DatabasePortalDeterministicReconciliationDeferred,
    )
    assert "forbidden provider" in str(caught.value)


def test_deterministic_reconciliation_wraps_predispatch_factory_deferral(
    tmp_path: Path,
) -> None:
    record, authority = _deterministic_record_and_authority()

    def factory(_paths: object, _alias: str) -> object:
        raise FileNotFoundError("Portal factory source missing")

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=factory,
        max_passes=1,
    )

    with pytest.raises(
        DatabasePortalDeterministicReconciliationDeferred
    ) as caught:
        bridge.run_deterministic_reconciliation(_attempt(), authority)

    assert caught.value.reason == (
        "deterministic_reconciliation_pre_provider_deferred"
    )
    assert caught.value.source_reason == "portal_factory_source_missing"
    assert caught.value.provider_dispatched is False
    assert caught.value.attempt_consumed is False
    assert caught.value.effect_executed is False


def test_bridge_projection_preserves_authoritative_database_task_identity(
    tmp_path: Path,
) -> None:
    record = _record()
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: object(),
    )

    paths, binding = bridge._ensure_attempt_projection(_attempt(), record)
    projected = parse_task_file(paths.task_projection, "LGSWF-")

    assert len(projected) == 1
    assert projected[0].canonical_task_key == record.body["task_key"]
    assert projected[0].canonical_task_cid == record.task_cid
    assert binding["canonical_task_key"] == record.body["task_key"]
    assert binding["task_cid"] == record.task_cid


@pytest.mark.parametrize(
    ("field", "label"),
    (("revision", "Revision"), ("receipt", "Receipt")),
)
def test_task_contract_digest_includes_projected_body_field(
    field: str,
    label: str,
) -> None:
    original = _record()
    original.body[field] = "contract-v1"
    changed = _record()
    changed.body[field] = "contract-v2"
    lifecycle_only = _record()
    lifecycle_only.body[field] = "contract-v1"
    lifecycle_only.revision = original.revision + 1

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(original),
        attempt_root=Path("unused"),
        portal_factory=lambda _paths, _alias: object(),
    )
    assert f"- {label}: contract-v1" in bridge._render_projection(
        _attempt(),
        original,
    )
    assert database_portal_task_contract_digest(original) != (
        database_portal_task_contract_digest(changed)
    )
    assert database_portal_task_contract_digest(original) == (
        database_portal_task_contract_digest(lifecycle_only)
    )


@pytest.mark.parametrize(
    ("identity_fields", "message"),
    [
        (
            {"canonical_task_cid": "task:cid:forged"},
            "contradicts its authoritative task CID",
        ),
        (
            {
                "task_key": "task/v1/one",
                "canonical_task_key": "task/v1/two",
            },
            "contradictory canonical task keys",
        ),
        (
            {
                "task_key": "task/v1/one",
                "task key": "task/v1/two",
            },
            "contradictory canonical task keys",
        ),
    ],
)
def test_bridge_projection_rejects_contradictory_database_identity(
    tmp_path: Path,
    identity_fields: dict[str, str],
    message: str,
) -> None:
    record = _record()
    record.body.update(identity_fields)
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: object(),
    )

    with pytest.raises(DatabasePortalBridgeError, match=message):
        bridge._ensure_attempt_projection(_attempt(), record)


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


def test_bridge_preserves_explicit_non_consuming_portal_deferral(
    tmp_path: Path,
) -> None:
    class DeferredPortal(_CompletingPortal):
        def run_once(self) -> dict[str, object]:
            return {
                "implementation_result": {
                    "returncode": 1,
                    "reason": (
                        "validation_project_dependency_preflight_failed"
                    ),
                    "deferred": True,
                    "retryable": True,
                    "deferral_schema": DATABASE_PORTAL_RETRY_DEFERRAL_SCHEMA,
                    "failure_kind": "lifecycle_setup",
                    "attempt_consumed": False,
                    "provider_dispatched": False,
                }
            }

    portals: list[DeferredPortal] = []

    def factory(paths: object, alias: str) -> DeferredPortal:
        portal = DeferredPortal(paths, alias)
        portals.append(portal)
        return portal

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=factory,
    )

    with pytest.raises(
        DatabasePortalBridgeDeferred,
        match="validation_project_dependency_preflight_failed",
    ):
        bridge.run_provider(_attempt())
    assert portals and portals[0].closed is True


def test_bridge_types_current_quack_refusal_before_portal_dispatch(
    tmp_path: Path,
) -> None:
    import duckdb

    class RefusingTaskSource:
        database_path = "quack:127.0.0.1:45123"

        def get_task(self, _task_cid: str) -> object:
            raise duckdb.IOException(
                "IO Error: Failed to send message: IO Error: Could not connect "
                "to server error for HTTP POST to "
                "'http://127.0.0.1:45123/quack'"
            )

    factory_called = False

    def factory(_paths: object, _alias: str) -> object:
        nonlocal factory_called
        factory_called = True
        raise AssertionError("Portal factory must remain outside this boundary")

    bridge = DatabasePortalExecutionBridge(
        task_source=RefusingTaskSource(),
        attempt_root=tmp_path / "attempts",
        portal_factory=factory,
    )

    with pytest.raises(DatabasePortalBridgeDeferred) as caught:
        bridge.run_provider(_attempt())
    assert caught.value.reason == "quack_transport_unavailable"
    assert caught.value.backoff_seconds == 30
    assert caught.value.provider_dispatched is False
    assert caught.value.attempt_consumed is False
    assert factory_called is False
    assert not (tmp_path / "attempts").exists()


def test_bridge_does_not_type_foreign_quack_refusal_as_pre_dispatch(
    tmp_path: Path,
) -> None:
    import duckdb

    class RefusingTaskSource:
        database_path = "quack:127.0.0.1:45123"

        def get_task(self, _task_cid: str) -> object:
            raise duckdb.IOException(
                "IO Error: Failed to send message: IO Error: Could not connect "
                "to server error for HTTP POST to "
                "'http://127.0.0.1:45124/quack'"
            )

    bridge = DatabasePortalExecutionBridge(
        task_source=RefusingTaskSource(),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: _CompletingPortal(paths, alias),
    )

    with pytest.raises(duckdb.IOException) as caught:
        bridge.run_provider(_attempt())
    assert not isinstance(caught.value, DatabasePortalBridgeDeferred)


def test_bridge_seals_historical_quack_preprojection_absence(
    tmp_path: Path,
) -> None:
    task_source = SimpleNamespace(database_path="quack:127.0.0.1:45123")
    bridge = DatabasePortalExecutionBridge(
        task_source=task_source,
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: _CompletingPortal(paths, alias),
    )
    reason = (
        "IO Error: Failed to send message: IO Error: Could not connect to "
        "server error for HTTP POST to 'http://127.0.0.1:45123/quack'"
    )

    receipt = bridge.recover_quack_preprojection_transport_failure(
        _attempt(),
        reason,
    )
    replay = bridge.recover_quack_preprojection_transport_failure(
        _attempt(),
        reason,
    )

    assert receipt == replay
    assert receipt["schema"] == DATABASE_PORTAL_QUACK_PREPROJECTION_RECOVERY_SCHEMA
    assert receipt["absence_sealed"] is True
    assert receipt["projection_present"] is False
    assert receipt["provider_dispatched"] is False
    assert receipt["effect_executed"] is False
    attempt_dir = bridge._paths(_attempt()).root
    assert [item.name for item in attempt_dir.iterdir()] == [
        "database-portal-quack-preprojection-recovery.json"
    ]


def test_bridge_quack_preprojection_recovery_rejects_any_projection_artifact(
    tmp_path: Path,
) -> None:
    bridge = DatabasePortalExecutionBridge(
        task_source=SimpleNamespace(
            database_path="quack:127.0.0.1:45123"
        ),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: _CompletingPortal(paths, alias),
    )
    attempt_dir = bridge._paths(_attempt()).root
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "task-projection.md").write_text("foreign", encoding="utf-8")

    with pytest.raises(
        DatabasePortalBridgeError,
        match="projection or foreign artifact",
    ):
        bridge.recover_quack_preprojection_transport_failure(
            _attempt(),
            "IO Error: Failed to send message: IO Error: Could not connect to "
            "server error for HTTP POST to "
            "'http://127.0.0.1:45123/quack'",
        )


def test_bridge_quack_preprojection_recovery_rejects_foreign_endpoint(
    tmp_path: Path,
) -> None:
    bridge = DatabasePortalExecutionBridge(
        task_source=SimpleNamespace(
            database_path="quack:127.0.0.1:45123"
        ),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: _CompletingPortal(paths, alias),
    )

    with pytest.raises(
        DatabasePortalBridgeError,
        match="not an exact Quack transport refusal",
    ):
        bridge.recover_quack_preprojection_transport_failure(
            _attempt(),
            "IO Error: Failed to send message: IO Error: Could not connect to "
            "server error for HTTP POST to "
            "'http://127.0.0.1:45124/quack'",
        )
    assert not (tmp_path / "attempts").exists()


def test_bridge_seals_consumed_no_progress_without_cause_inference(
    tmp_path: Path,
) -> None:
    class ImportFailurePortal(_CompletingPortal):
        def run_once(self) -> dict[str, object]:
            return _consumed_no_progress_result(
                self.paths,
                self.task_alias,
            )

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: ImportFailurePortal(paths, alias),
    )

    with pytest.raises(
        DatabasePortalBridgeConsumedNoProgressError,
        match="portal_consumed_no_progress",
    ) as raised:
        bridge.run_provider(_attempt())

    evidence = raised.value.failure_evidence
    assert evidence["schema"] == (
        DATABASE_PORTAL_CONSUMED_NO_PROGRESS_SCHEMA
    )
    assert evidence["failure_kind"] == "consumed_no_progress"
    assert evidence["provider_effect_state"] == "unknown_may_have_started"
    assert evidence["portal_provider_dispatched"] is True
    assert evidence["portal_attempt_number"] == 1
    assert evidence["tree_id"] == "a" * 40
    assert evidence["control_repository_tree_id"] == (
        "tree:control-plane-current"
    )
    assert evidence["task_cid"] == "task:cid:004"
    assert evidence["task_contract_digest"].startswith("sha256:")
    assert evidence["log_digest"].startswith("sha256:")
    assert evidence["context_receipt_id"].startswith("baguq")
    assert str(evidence["diagnostic_failure_id"]).startswith("baguq")
    assert str(evidence["diagnostic_receipt_id"]).startswith("baguq")
    assert evidence["failure_fingerprint"].startswith("sha256:")
    assert evidence["failure_fingerprint"] == (
        database_portal_consumed_no_progress_fingerprint(evidence)
    )
    assert "ImportError" not in json.dumps(evidence)
    assert "provider output" not in json.dumps(evidence)


@pytest.mark.parametrize("forged_identity", ["receipt_id", "failure_id"])
def test_bridge_rejects_forged_diagnostic_identity(
    tmp_path: Path,
    forged_identity: str,
) -> None:
    class ForgedDiagnosticPortal(_CompletingPortal):
        def run_once(self) -> dict[str, object]:
            return _consumed_no_progress_result(
                self.paths,
                self.task_alias,
                forged_identity=forged_identity,
            )

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: ForgedDiagnosticPortal(
            paths,
            alias,
        ),
    )

    with pytest.raises(DatabasePortalBridgeError) as raised:
        bridge.run_provider(_attempt())
    assert not isinstance(
        raised.value,
        DatabasePortalBridgeConsumedNoProgressError,
    )


@pytest.mark.parametrize(
    "log_text",
    [
        "arbitrary provider/model output\n",
        (
            "Traceback (most recent call last):\n"
            "ImportError: cannot import name 'spoofed' from 'provider.text'\n"
        ),
    ],
)
def test_bridge_neutralizes_spoofed_failure_text_without_classifying_cause(
    tmp_path: Path,
    log_text: str,
) -> None:
    class SpoofedFailurePortal(_CompletingPortal):
        def run_once(self) -> dict[str, object]:
            return _consumed_no_progress_result(
                self.paths,
                self.task_alias,
                returncode=2,
                log_text=log_text,
            )

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: SpoofedFailurePortal(paths, alias),
    )

    with pytest.raises(DatabasePortalBridgeConsumedNoProgressError) as raised:
        bridge.run_provider(_attempt())
    assert raised.value.failure_evidence["failure_kind"] == (
        "consumed_no_progress"
    )
    assert raised.value.failure_evidence["returncode"] == 2
    assert raised.value.failure_evidence["provider_effect_state"] == (
        "unknown_may_have_started"
    )
    assert "spoofed" not in json.dumps(raised.value.failure_evidence)


def test_bridge_does_not_neutralize_an_implementation_candidate(
    tmp_path: Path,
) -> None:
    class CandidatePortal(_CompletingPortal):
        def run_once(self) -> dict[str, object]:
            return _consumed_no_progress_result(
                self.paths,
                self.task_alias,
                committed=True,
            )

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: CandidatePortal(paths, alias),
    )

    with pytest.raises(DatabasePortalBridgeError) as raised:
        bridge.run_provider(_attempt())
    assert not isinstance(
        raised.value,
        DatabasePortalBridgeConsumedNoProgressError,
    )


def test_bridge_rejects_free_text_capacity_as_retry_authority(
    tmp_path: Path,
) -> None:
    class TextOnlyCapacityPortal(_CompletingPortal):
        def run_once(self) -> dict[str, object]:
            return {
                "implementation_result": {
                    "returncode": 1,
                    "reason": "arbitrary_capacity_backoff_provider_text",
                    "deferred": True,
                    "attempt_consumed": False,
                    "provider_dispatched": False,
                }
            }

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: TextOnlyCapacityPortal(paths, alias),
    )

    with pytest.raises(DatabasePortalBridgeError) as raised:
        bridge.run_provider(_attempt())
    assert not isinstance(raised.value, DatabasePortalBridgeDeferred)


def test_bridge_rejects_completion_event_for_another_canonical_task(
    tmp_path: Path,
) -> None:
    class ForgedCompletionPortal(_CompletingPortal):
        def run_once(self) -> dict[str, object]:
            result = super().run_once()
            events = [
                json.loads(line)
                for line in self.paths.events.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            completion = events[-1]
            completion["canonical_task_cid"] = "task:cid:other"
            completion_body = dict(completion)
            completion_body.pop("event_id")
            completion["event_id"] = "sha256:" + hashlib.sha256(
                json.dumps(
                    completion_body,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            self.paths.events.write_text(
                "".join(
                    json.dumps(event, sort_keys=True, separators=(",", ":"))
                    + "\n"
                    for event in events
                ),
                encoding="utf-8",
            )
            return result

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: ForgedCompletionPortal(paths, alias),
        max_passes=1,
    )

    with pytest.raises(DatabasePortalBridgeError, match="verified task_completed"):
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
        result = daemon.run_once()
        implementation = result["implementation_result"]
        assert implementation["portal_terminal_failure"] is True
        assert implementation["provider_dispatched"] == "unknown"
        assert "no provider executor" in implementation["reason"]
        task = daemon.task_source.get_task("task:cid:004")
        assert task is not None
        assert task.status == "blocked"
        assert daemon.list_running_attempts() == []
        assert (
            daemon.provider_invocation_recorded(
                result["attempt_id"],
                idempotency_key=f"provider:{result['attempt_id']}",
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


def _progressed_implementation_commit(repo: Path) -> tuple[str, str]:
    output = repo / "inventory" / "result.json"
    output.write_text('{"progressed":true}\n', encoding="utf-8")
    subprocess.run(
        ["git", "add", "--", str(output.relative_to(repo))],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-qm", "progressed implementation"],
        cwd=repo,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    branch = "implementation/lgswf-004-attempt-2-progressed"
    subprocess.run(["git", "branch", branch, commit], cwd=repo, check=True)
    return commit, branch


def _mutate_portal_retry_state(
    paths: object,
    *,
    alias: str,
    task_cid: str,
    commit: str,
    branch: str,
    attempts: int = 2,
    returncode: int = 0,
    task_id: str | None = None,
) -> dict[str, object]:
    state = json.loads(paths.state.read_text(encoding="utf-8"))
    assert isinstance(state, dict)
    state["implementation_attempts"] = {alias: attempts}
    state["implementation_attempts_by_cid"] = {task_cid: attempts}
    state["last_implementation_task_id"] = task_id if task_id is not None else alias
    state["last_implementation_returncode"] = returncode
    state["last_implementation_branch"] = branch
    state["last_implementation_commit"] = commit
    paths.state.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return state


class _InspectSeedPortal:
    def __init__(self, paths: object) -> None:
        self.paths = paths

    def run_once(self) -> dict[str, object]:
        return {
            "implementation_result": {
                "returncode": 1,
                "reason": "stop_after_seed_inspection",
            }
        }


def _seeded_validation_retry_successor(tmp_path: Path) -> dict[str, object]:
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
    production_attempt = _attempt(attempt_number=189)
    with pytest.raises(DatabasePortalValidationRetry) as caught:
        bridge.run_provider(production_attempt)
    retry = caught.value
    record.body = {
        **record.body,
        "completion_receipt": {
            "operation": "database_claim",
            "attempt_id": "attempt:002",
            "claim_id": "claim:002",
            "attempt_number": 190,
            "fencing_token": 8,
            "fence_epoch": 3,
            "lease_id": "lease:002",
            "validation_retry_source_attempt_id": (
                production_attempt.attempt_id
            ),
            "validation_retry_seed": retry.retry_receipt,
        },
    }
    record.revision += 1
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
    successor_bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "successor-attempts",
        portal_factory=lambda paths, _alias: _InspectSeedPortal(paths),
        repository_root=repo,
        max_passes=1,
        max_task_attempts=3,
    )
    with pytest.raises(
        DatabasePortalBridgeError, match="stop_after_seed_inspection"
    ):
        successor_bridge.run_provider(successor)
    paths = successor_bridge._paths(successor)
    return {
        "repo": repo,
        "record": record,
        "commit": commit,
        "rescue_branch": rescue_branch,
        "successor": successor,
        "bridge": successor_bridge,
        "paths": paths,
        "retry": retry,
    }


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
                    "retryable": True,
                    "attempt_consumed": False,
                    "provider_dispatched": False,
                    "provider_call_allowed": False,
                    "deferral_schema": DATABASE_PORTAL_RETRY_DEFERRAL_SCHEMA,
                    "failure_kind": "lifecycle_setup",
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
    assert not isinstance(caught.value, DatabasePortalCandidateRetry)


def test_bridge_defers_external_protected_checkout_contention(
    tmp_path: Path,
) -> None:
    class CheckoutContentionPortal:
        def run_once(self) -> dict[str, object]:
            return {
                "blocked": True,
                "unchanged": True,
                "write_count": 0,
                "implementation_result": None,
                "reason": "external_protected_checkout_recovery_required",
            }

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: CheckoutContentionPortal(),
        max_passes=1,
    )

    with pytest.raises(DatabasePortalBridgeDeferred) as caught:
        bridge.run_provider(_attempt())

    assert caught.value.reason == "external_protected_checkout_recovery_required"
    assert caught.value.backoff_seconds == (
        DATABASE_PORTAL_CHECKOUT_CONTENTION_BACKOFF_SECONDS
    )
    assert caught.value.attempt_consumed is False
    assert caught.value.provider_dispatched is False


def test_bridge_keeps_invalid_protected_recovery_journal_terminal(
    tmp_path: Path,
) -> None:
    class InvalidJournalPortal:
        def run_once(self) -> dict[str, object]:
            return {
                "blocked": True,
                "unchanged": True,
                "write_count": 0,
                "implementation_result": None,
                "reason": "protected_recovery_journal_invalid",
            }

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: InvalidJournalPortal(),
        max_passes=1,
    )

    with pytest.raises(DatabasePortalBridgeError) as caught:
        bridge.run_provider(_attempt())

    assert not isinstance(caught.value, DatabasePortalBridgeDeferred)
    assert str(caught.value) == "protected_recovery_journal_invalid"


@pytest.mark.parametrize(
    "payload",
    (
        {
            "returncode": 78,
            "attempt": 1,
            "attempt_consumed": True,
            "provider_dispatched": True,
            "validation_result": {
                "attempted": False,
                "passed": False,
                "reason": "no_change_completion_not_allowed",
            },
        },
        {
            "returncode": 78,
            "attempt": 1,
            "attempt_consumed": True,
            "provider_dispatched": True,
            "validation_result": {
                "attempted": True,
                "passed": False,
                "reason": "proposal_gate_failed",
                "error": "proposal_validation_failed",
            },
            "commit_result": {"reason": "expected_output_ignored_or_unstaged"},
        },
        {
            "returncode": 1,
            "attempt": 2,
            "attempt_consumed": True,
            "provider_dispatched": True,
            "reason": "incomplete_expected_outputs",
        },
    ),
)
def test_bridge_retries_unusable_dispatched_candidates(
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    class CandidatePortal:
        def run_once(self) -> dict[str, object]:
            return {"implementation_result": dict(payload)}

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: CandidatePortal(),
        max_passes=1,
        max_task_attempts=4,
    )

    with pytest.raises(DatabasePortalCandidateRetry) as caught:
        bridge.run_provider(_attempt())

    assert caught.value.attempt_consumed is True
    assert caught.value.provider_dispatched is True
    assert caught.value.reason in {
        "no_change_completion_not_allowed",
        "proposal_gate_failed",
        "incomplete_expected_outputs",
    }


def test_bridge_keeps_exhausted_unusable_candidate_terminal(
    tmp_path: Path,
) -> None:
    class ExhaustedPortal:
        def run_once(self) -> dict[str, object]:
            return {
                "implementation_result": {
                    "returncode": 78,
                    "attempt": 4,
                    "attempt_consumed": True,
                    "provider_dispatched": True,
                    "validation_result": {
                        "reason": "no_change_completion_not_allowed"
                    },
                }
            }

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: ExhaustedPortal(),
        max_passes=1,
        max_task_attempts=4,
    )

    with pytest.raises(DatabasePortalBridgeError) as caught:
        bridge.run_provider(_attempt())

    assert not isinstance(caught.value, DatabasePortalCandidateRetry)
    assert str(caught.value) == "portal_provider_failed"


def test_bridge_stops_candidate_retry_when_durable_attempt_reaches_cap(
    tmp_path: Path,
) -> None:
    class ResetPortalAttempt:
        def run_once(self) -> dict[str, object]:
            return {
                "implementation_result": {
                    "returncode": 78,
                    "attempt": 1,
                    "attempt_consumed": True,
                    "provider_dispatched": True,
                    "validation_result": {
                        "attempted": False,
                        "passed": False,
                        "reason": "no_change_completion_not_allowed",
                    },
                }
            }

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: ResetPortalAttempt(),
        max_passes=1,
        max_task_attempts=4,
    )

    with pytest.raises(DatabasePortalBridgeError) as caught:
        bridge.run_provider(_attempt(attempt_number=4))

    assert not isinstance(caught.value, DatabasePortalCandidateRetry)
    assert str(caught.value) == "portal_provider_failed"


def test_bridge_defers_protected_recovery_fence_contention(
    tmp_path: Path,
) -> None:
    class FenceBlockedPortal:
        def run_once(self) -> dict[str, object]:
            return {
                "blocked": True,
                "unchanged": True,
                "reason": "external_protected_checkout_recovery_required",
            }

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: FenceBlockedPortal(),
        max_passes=1,
    )

    with pytest.raises(DatabasePortalBridgeDeferred) as caught:
        bridge.run_provider(_attempt())

    assert (
        str(caught.value) == "external_protected_checkout_recovery_required"
    )
    assert caught.value.backoff_seconds == 30
    assert caught.value.attempt_consumed is False
    assert caught.value.provider_dispatched is False


def test_bridge_defers_live_inflight_process_skip(
    tmp_path: Path,
) -> None:
    class InflightPortal:
        def run_once(self) -> dict[str, object]:
            return {
                "implementation_result": {
                    "skipped": True,
                    "reason": "inflight_process",
                    "task_id": "PCCE-021",
                    "attempt": 1,
                    "attempt_consumed": False,
                    "provider_dispatched": False,
                }
            }

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: InflightPortal(),
        max_passes=1,
    )

    with pytest.raises(DatabasePortalBridgeDeferred) as caught:
        bridge.run_provider(_attempt())

    assert str(caught.value) == "inflight_process"
    assert caught.value.backoff_seconds == 30
    assert caught.value.attempt_consumed is False
    assert caught.value.provider_dispatched is False


def test_bridge_still_terminals_non_fence_blocked_portal(
    tmp_path: Path,
) -> None:
    class OtherBlockedPortal:
        def run_once(self) -> dict[str, object]:
            return {
                "blocked": True,
                "reason": "crash_reconciliation_inputs_drifted",
            }

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: OtherBlockedPortal(),
        max_passes=1,
    )

    with pytest.raises(DatabasePortalBridgeError) as caught:
        bridge.run_provider(_attempt())

    assert not isinstance(caught.value, DatabasePortalBridgeDeferred)
    assert str(caught.value) == "crash_reconciliation_inputs_drifted"


def test_bridge_defers_paired_supervisor_external_checkout_recovery(
    tmp_path: Path,
) -> None:
    class SupervisorOwnedPortal:
        closed = False

        def run_once(self) -> dict[str, object]:
            return {
                "blocked": True,
                "reason": "external_protected_checkout_recovery_required",
                "protected_checkout_recovery": {
                    "required": True,
                    "adopted": False,
                    "blocked": True,
                    "reason": "external_protected_checkout_recovery_required",
                    "protected_recovery_owner": "implementation_supervisor",
                    "lock_path": str(tmp_path / "implementation-main-merge.lock"),
                },
            }

        def close_event_runtime(self) -> None:
            self.closed = True

    portal = SupervisorOwnedPortal()
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: portal,
        repository_root=tmp_path,
        max_passes=1,
    )

    with pytest.raises(DatabasePortalBridgeDeferred) as caught:
        bridge.run_provider(_attempt())

    assert str(caught.value) == "external_protected_checkout_recovery_required"
    assert caught.value.backoff_seconds == 30
    assert caught.value.attempt_consumed is False
    assert caught.value.provider_dispatched is False
    assert portal.closed is True


def test_bridge_defers_foreign_external_checkout_recovery_fence(
    tmp_path: Path,
) -> None:
    class ForeignOwnedPortal:
        def run_once(self) -> dict[str, object]:
            return {
                "blocked": True,
                "reason": "external_protected_checkout_recovery_required",
                "protected_checkout_recovery": {
                    "required": True,
                    "adopted": False,
                    "blocked": True,
                    "reason": "external_protected_checkout_recovery_required",
                    "protected_recovery_owner": "foreign_owner",
                    "lock_path": str(tmp_path / "implementation-main-merge.lock"),
                },
            }

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: ForeignOwnedPortal(),
        repository_root=tmp_path,
        max_passes=1,
    )

    with pytest.raises(DatabasePortalBridgeDeferred) as caught:
        bridge.run_provider(_attempt())

    assert str(caught.value) == "external_protected_checkout_recovery_required"
    assert caught.value.backoff_seconds == 30
    assert caught.value.attempt_consumed is False
    assert caught.value.provider_dispatched is False


def test_bridge_recovers_external_checkout_only_when_lock_absent(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
        checkout_mutation_lock_path,
    )

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: None,
        repository_root=repo,
    )
    attempt = _attempt()
    lock_path = checkout_mutation_lock_path(repo)
    lock_path.write_text("{}", encoding="utf-8")

    with pytest.raises(DatabasePortalBridgeError) as blocked:
        bridge.recover_external_protected_checkout(attempt)
    assert "lock to be absent" in str(blocked.value)

    lock_path.unlink()
    receipt = bridge.recover_external_protected_checkout(attempt)
    assert receipt["reason"] == "external_protected_checkout_lock_absent"
    assert receipt["source_reason"] == (
        "external_protected_checkout_recovery_required"
    )
    assert receipt["lock_present"] is False
    assert receipt["lock_path"] == str(lock_path)
    assert bridge.recover_external_protected_checkout(attempt) == receipt


def test_bridge_defers_inflight_process_skip(tmp_path: Path) -> None:
    class InflightPortal:
        closed = False

        def run_once(self) -> dict[str, object]:
            return {
                "implementation_result": {
                    "skipped": True,
                    "reason": "inflight_process",
                    "task_id": "LGSWF-004",
                    "attempt": 1,
                    "worktree_path": str(tmp_path / "worktrees" / "live"),
                }
            }

        def close_event_runtime(self) -> None:
            self.closed = True

    portal = InflightPortal()
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: portal,
        max_passes=1,
    )

    with pytest.raises(DatabasePortalBridgeDeferred) as caught:
        bridge.run_provider(_attempt())

    assert str(caught.value) == "inflight_process"
    assert caught.value.backoff_seconds == 30
    assert caught.value.attempt_consumed is False
    assert portal.closed is True


def test_bridge_keeps_other_skipped_reasons_terminal(tmp_path: Path) -> None:
    class OtherSkipPortal:
        def run_once(self) -> dict[str, object]:
            return {
                "implementation_result": {
                    "skipped": True,
                    "reason": "provider_capacity_backoff",
                }
            }

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: OtherSkipPortal(),
        max_passes=1,
    )

    with pytest.raises(DatabasePortalBridgeError) as caught:
        bridge.run_provider(_attempt())

    assert not isinstance(caught.value, DatabasePortalBridgeDeferred)
    assert str(caught.value) == "provider_capacity_backoff"


def test_bridge_defers_worktree_lifecycle_claim_skip(tmp_path: Path) -> None:
    class LifecycleSkipPortal:
        closed = False

        def run_once(self) -> dict[str, object]:
            return {
                "implementation_result": {
                    "skipped": True,
                    "lifecycle_race": True,
                    "reason": "worktree_lifecycle_claim_exists",
                    "attempt_consumed": False,
                    "backoff_seconds": 30,
                }
            }

        def close_event_runtime(self) -> None:
            self.closed = True

    portal = LifecycleSkipPortal()
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: portal,
        max_passes=1,
    )

    with pytest.raises(DatabasePortalBridgeDeferred) as caught:
        bridge.run_provider(_attempt())

    assert str(caught.value) == "worktree_lifecycle_claim_exists"
    assert caught.value.backoff_seconds == 30
    assert caught.value.attempt_consumed is False
    assert portal.closed is True


def test_bridge_reclaims_only_dead_exact_lane_portal_lifecycle_claims(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-q"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    attempt_root = tmp_path / "lane" / "lane_database_portal_attempts"
    attempt_root.mkdir(parents=True)
    current_name = hashlib.sha256(b"attempt:001").hexdigest()[:24]
    stale_name = "f" * 24 if current_name != "f" * 24 else "e" * 24
    stale_state_dir = attempt_root / stale_name
    live_state_dir = attempt_root / ("d" * 24)
    foreign_state_dir = tmp_path / "peer" / ("c" * 24)
    for directory in (stale_state_dir, live_state_dir, foreign_state_dir):
        directory.mkdir(parents=True)

    store = WorktreeLifecycleStore(repo)
    stale = store.begin_preparing(
        task_id="STALE",
        canonical_task_cid="task:stale",
        attempt=1,
        lane_id="lane:stale",
        workspace_path=tmp_path / "worktrees" / "stale",
        branch="implementation/stale",
        merge_target="main",
        state_dir=str(stale_state_dir),
        owner=ProcessBirthIdentity(
            pid=99_999_999,
            start_time_ticks=1,
            boot_id="dead-owner",
        ),
    )
    live = store.begin_preparing(
        task_id="LIVE",
        canonical_task_cid="task:live",
        attempt=1,
        lane_id="lane:live",
        workspace_path=tmp_path / "worktrees" / "live",
        branch="implementation/live",
        merge_target="main",
        state_dir=str(live_state_dir),
        owner=current_process_birth(),
    )
    foreign = store.begin_preparing(
        task_id="FOREIGN",
        canonical_task_cid="task:foreign",
        attempt=1,
        lane_id="peer:foreign",
        workspace_path=tmp_path / "worktrees" / "foreign",
        branch="implementation/foreign",
        merge_target="main",
        state_dir=str(foreign_state_dir),
        owner=ProcessBirthIdentity(
            pid=99_999_998,
            start_time_ticks=1,
            boot_id="dead-peer",
        ),
    )

    observed: dict[str, object] = {}

    def factory(paths: object, alias: str) -> object:
        recovered = store.load_workspace(stale.workspace_path)
        retained_live = store.load_workspace(live.workspace_path)
        retained_foreign = store.load_workspace(foreign.workspace_path)
        observed.update(
            recovered=recovered,
            retained_live=retained_live,
            retained_foreign=retained_foreign,
        )
        return _CompletingPortal(paths, alias)

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=attempt_root,
        portal_factory=factory,
        repository_root=repo,
        max_passes=1,
    )

    receipt = bridge.run_provider(_attempt())

    recovered = observed["recovered"]
    assert recovered.is_terminal
    assert recovered.fence == stale.fence + 1
    assert recovered.terminal_reason == (
        "database_portal_dead_attempt_owner_reclaim"
    )
    assert observed["retained_live"].is_nonterminal
    assert observed["retained_foreign"].is_nonterminal
    assert receipt["accepted"] is True
    event_path = attempt_root / current_name / "portal-events.jsonl"
    events = [json.loads(line) for line in event_path.read_text().splitlines()]
    recovery_events = [
        event
        for event in events
        if event.get("type")
        == "database_portal_worktree_lifecycle_recovered"
    ]
    assert len(recovery_events) == 1
    assert recovery_events[0]["recovered_count"] == 1
    assert recovery_events[0]["recovered"][0]["prior_record_id"] == (
        stale.record_id
    )
    assert recovery_events[0]["recovered"][0]["terminal_record_id"] == (
        recovered.record_id
    )


def test_bridge_defers_portal_factory_source_missing_before_dispatch(
    tmp_path: Path,
) -> None:
    factory_calls: list[str] = []

    def missing_factory(_paths: object, alias: str) -> object:
        factory_calls.append(alias)
        raise FileNotFoundError("Portal source disappeared before construction")

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=missing_factory,
        max_passes=1,
    )

    with pytest.raises(DatabasePortalBridgeDeferred) as caught:
        bridge.run_provider(_attempt())

    assert caught.value.reason == "portal_factory_source_missing"
    assert caught.value.backoff_seconds == 30
    assert caught.value.provider_dispatched is False
    assert caught.value.attempt_consumed is False
    assert factory_calls == ["LGSWF-004"]


def test_bridge_defers_exact_quack_refusal_during_portal_construction(
    tmp_path: Path,
) -> None:
    import duckdb

    quack_uri = "quack:127.0.0.1:45123"
    task_source = _TaskSource(_record())
    task_source.database_path = quack_uri
    factory_calls: list[str] = []

    def refusing_factory(_paths: object, alias: str) -> object:
        factory_calls.append(alias)
        raise duckdb.IOException(
            "IO Error: Failed to send message: IO Error: Could not connect "
            "to server error for HTTP POST to "
            "'http://127.0.0.1:45123/quack'"
        )

    bridge = DatabasePortalExecutionBridge(
        task_source=task_source,
        attempt_root=tmp_path / "attempts",
        portal_factory=refusing_factory,
        max_passes=1,
    )

    with pytest.raises(DatabasePortalBridgeDeferred) as caught:
        bridge.run_provider(_attempt())

    assert caught.value.reason == "quack_transport_unavailable"
    assert caught.value.backoff_seconds == 30
    assert caught.value.provider_dispatched is False
    assert caught.value.attempt_consumed is False
    assert factory_calls == ["LGSWF-004"]


def test_bridge_rejects_foreign_quack_refusal_during_portal_construction(
    tmp_path: Path,
) -> None:
    import duckdb

    task_source = _TaskSource(_record())
    task_source.database_path = "quack:127.0.0.1:45123"

    def refusing_factory(_paths: object, _alias: str) -> object:
        raise duckdb.IOException(
            "IO Error: Failed to send message: IO Error: Could not connect "
            "to server error for HTTP POST to "
            "'http://127.0.0.1:45124/quack'"
        )

    bridge = DatabasePortalExecutionBridge(
        task_source=task_source,
        attempt_root=tmp_path / "attempts",
        portal_factory=refusing_factory,
        max_passes=1,
    )

    with pytest.raises(duckdb.IOException) as caught:
        bridge.run_provider(_attempt())
    assert not isinstance(caught.value, DatabasePortalBridgeDeferred)


@pytest.mark.parametrize(
    ("provider_dispatched", "attempt_consumed", "deferred"),
    [
        (False, False, True),
        (False, True, False),
        (True, False, False),
    ],
)
def test_bridge_types_file_not_found_setup_only_before_dispatch_and_consumption(
    tmp_path: Path,
    provider_dispatched: bool,
    attempt_consumed: bool,
    deferred: bool,
) -> None:
    class MissingSetupPortal:
        closed = False

        def run_once(self) -> dict[str, object]:
            return {
                "implementation_result": {
                    "returncode": 1,
                    "provider_dispatched": provider_dispatched,
                    "attempt_consumed": attempt_consumed,
                    "exception_result": {
                        "exception_type": "FileNotFoundError",
                        "phase": "worktree_setup",
                        "message": "worktree source path disappeared",
                    },
                }
            }

        def close_event_runtime(self) -> None:
            self.closed = True

    portal = MissingSetupPortal()
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: portal,
        max_passes=1,
    )

    if deferred:
        with pytest.raises(DatabasePortalBridgeDeferred) as caught:
            bridge.run_provider(_attempt())
        assert caught.value.reason == (
            DATABASE_PORTAL_POOLED_WORKTREE_CREATE_FAILED_REASON
        )
        assert caught.value.provider_dispatched is False
        assert caught.value.attempt_consumed is False
    else:
        with pytest.raises(DatabasePortalBridgeError) as caught:
            bridge.run_provider(_attempt())
        assert not isinstance(caught.value, DatabasePortalBridgeDeferred)
        assert str(caught.value) == "portal_provider_failed"
    assert portal.closed is True


def test_bridge_defers_pooled_worktree_create_interrupt(tmp_path: Path) -> None:
    class SetupFailPortal:
        closed = False

        def run_once(self) -> dict[str, object]:
            return {
                "implementation_result": {
                    "returncode": 1,
                    "provider_dispatched": False,
                    "attempt_consumed": True,
                    "exception_result": {
                        "exception_type": "RuntimeError",
                        "phase": "worktree_setup",
                        "message": (
                            "failed to create pooled worktree: "
                            "Preparing worktree (new branch 'implementation/x')"
                        ),
                    },
                }
            }

        def close_event_runtime(self) -> None:
            self.closed = True

    portal = SetupFailPortal()
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: portal,
        max_passes=1,
    )
    with pytest.raises(DatabasePortalBridgeDeferred) as caught:
        bridge.run_provider(_attempt())
    assert str(caught.value) == DATABASE_PORTAL_POOLED_WORKTREE_CREATE_FAILED_REASON
    assert caught.value.backoff_seconds == 30
    assert caught.value.attempt_consumed is False
    assert portal.closed is True


def test_bridge_keeps_dispatched_provider_failure_terminal(tmp_path: Path) -> None:
    class DispatchedFailPortal:
        def run_once(self) -> dict[str, object]:
            return {
                "implementation_result": {
                    "returncode": 1,
                    "provider_dispatched": True,
                    "attempt_consumed": True,
                    "reason": "model_failed",
                    "exception_result": {
                        "exception_type": "RuntimeError",
                        "phase": "provider",
                        "message": "failed to create pooled worktree: should-not-match",
                    },
                }
            }

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: DispatchedFailPortal(),
        max_passes=1,
    )
    with pytest.raises(DatabasePortalBridgeError) as caught:
        bridge.run_provider(_attempt())
    assert not isinstance(caught.value, DatabasePortalBridgeDeferred)
    assert str(caught.value) == "model_failed"


def test_bridge_recovers_pooled_worktree_create_when_path_absent(
    tmp_path: Path,
) -> None:
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: None,
        max_passes=1,
    )
    attempt = _attempt()
    paths = bridge._paths(attempt)
    paths.root.mkdir(parents=True)
    append_jsonl_event(
        paths.events,
        "implementation_finished",
        {
            "task_id": attempt.task_alias,
            "canonical_task_cid": attempt.task_cid,
            "task_cid": attempt.task_cid,
            "provider_dispatched": False,
            "attempt_consumed": True,
            "returncode": 1,
            "worktree_path": str(tmp_path / "missing-pooled-worktree"),
            "exception_result": {
                "exception_type": "RuntimeError",
                "phase": "worktree_setup",
                "message": "failed to create pooled worktree: Preparing worktree",
            },
        },
    )
    receipt = bridge.recover_pooled_worktree_create(attempt)
    assert receipt["schema"] == DATABASE_PORTAL_POOLED_WORKTREE_CREATE_RECOVERY_SCHEMA
    assert receipt["reason"] == DATABASE_PORTAL_POOLED_WORKTREE_CREATE_FAILED_REASON
    assert receipt["source_reason"] == "portal_provider_failed"
    assert receipt["worktree_present"] is False
    assert receipt["identity_bound"] is True
    assert bridge.recover_pooled_worktree_create(attempt) == receipt


def test_bridge_does_not_recover_pooled_worktree_create_while_path_present(
    tmp_path: Path,
) -> None:
    leftover = tmp_path / "leftover-pooled-worktree"
    leftover.mkdir()
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: None,
        max_passes=1,
    )
    attempt = _attempt()
    paths = bridge._paths(attempt)
    paths.root.mkdir(parents=True)
    append_jsonl_event(
        paths.events,
        "implementation_finished",
        {
            "task_id": attempt.task_alias,
            "canonical_task_cid": attempt.task_cid,
            "task_cid": attempt.task_cid,
            "provider_dispatched": False,
            "attempt_consumed": True,
            "returncode": 1,
            "worktree_path": str(leftover),
            "exception_result": {
                "exception_type": "RuntimeError",
                "phase": "worktree_setup",
                "message": "failed to create pooled worktree: Preparing worktree",
            },
        },
    )
    with pytest.raises(DatabasePortalBridgeError, match="worktree path to be absent"):
        bridge.recover_pooled_worktree_create(attempt)


def test_bridge_recovers_inflight_process_only_when_runner_absent(
    tmp_path: Path,
) -> None:
    class Detector:
        def __init__(self, live: object) -> None:
            self.live = live
            self.closed = False

        def _find_live_inflight_implementation(self) -> object:
            return self.live

        def close_event_runtime(self) -> None:
            self.closed = True

    live = Detector({"task_id": "LGSWF-004", "worktree_path": "/tmp/live"})
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: live,
    )
    attempt = _attempt()
    with pytest.raises(DatabasePortalBridgeError) as blocked:
        bridge.recover_inflight_process(attempt)
    assert "runner to be absent" in str(blocked.value)
    assert live.closed is True

    absent = Detector(None)
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: absent,
    )
    receipt = bridge.recover_inflight_process(attempt)
    assert receipt["reason"] == "inflight_process_absent"
    assert receipt["source_reason"] == "inflight_process"
    assert receipt["live_runner_present"] is False
    assert bridge.recover_inflight_process(attempt) == receipt


def test_bridge_accepts_identity_bound_progressed_validation_retry_seed(
    tmp_path: Path,
) -> None:
    seeded = _seeded_validation_retry_successor(tmp_path)
    successor = seeded["successor"]
    bridge = seeded["bridge"]
    paths = seeded["paths"]
    progressed_commit, progressed_branch = _progressed_implementation_commit(
        seeded["repo"]
    )
    _mutate_portal_retry_state(
        paths,
        alias=successor.task_alias,
        task_cid=successor.task_cid,
        commit=progressed_commit,
        branch=progressed_branch,
    )

    with pytest.raises(
        DatabasePortalBridgeError, match="stop_after_seed_inspection"
    ):
        bridge.run_provider(successor)

    state = json.loads(paths.state.read_text(encoding="utf-8"))
    assert state["implementation_attempts"][successor.task_alias] == 2
    assert state["last_implementation_returncode"] == 0
    assert state["last_implementation_commit"] == progressed_commit
    assert state["last_implementation_branch"] == progressed_branch


def test_bridge_keeps_foreign_progressed_validation_retry_seed_terminal(
    tmp_path: Path,
) -> None:
    seeded = _seeded_validation_retry_successor(tmp_path)
    successor = seeded["successor"]
    bridge = seeded["bridge"]
    paths = seeded["paths"]
    progressed_commit, progressed_branch = _progressed_implementation_commit(
        seeded["repo"]
    )
    _mutate_portal_retry_state(
        paths,
        alias=successor.task_alias,
        task_cid=successor.task_cid,
        commit=progressed_commit,
        branch=progressed_branch,
        task_id="FOREIGN-001",
    )

    with pytest.raises(DatabasePortalBridgeError) as caught:
        bridge.run_provider(successor)

    assert str(caught.value) == DATABASE_PORTAL_VALIDATION_RETRY_SEED_CONFLICT_REASON
    with pytest.raises(DatabasePortalBridgeError, match="identity-bound"):
        bridge.recover_validation_retry_seed_conflict(successor)


def test_bridge_recovers_identity_bound_validation_retry_seed_conflict(
    tmp_path: Path,
) -> None:
    seeded = _seeded_validation_retry_successor(tmp_path)
    successor = seeded["successor"]
    bridge = seeded["bridge"]
    paths = seeded["paths"]
    progressed_commit, progressed_branch = _progressed_implementation_commit(
        seeded["repo"]
    )
    _mutate_portal_retry_state(
        paths,
        alias=successor.task_alias,
        task_cid=successor.task_cid,
        commit=progressed_commit,
        branch=progressed_branch,
    )

    receipt = bridge.recover_validation_retry_seed_conflict(successor)
    assert (
        receipt["schema"]
        == DATABASE_PORTAL_VALIDATION_RETRY_SEED_CONFLICT_RECOVERY_SCHEMA
    )
    assert receipt["reason"] == "validation_retry_seed_state_progressed"
    assert (
        receipt["source_reason"]
        == DATABASE_PORTAL_VALIDATION_RETRY_SEED_CONFLICT_REASON
    )
    assert receipt["identity_bound"] is True
    assert receipt["observed_commit"] == progressed_commit
    assert receipt["observed_branch"] == progressed_branch
    assert receipt["seed_commit"] == seeded["commit"]
    assert receipt["seed_rescue_branch"] == seeded["rescue_branch"]
    assert bridge.recover_validation_retry_seed_conflict(successor) == receipt


def test_bridge_does_not_recover_invented_validation_retry_seed_commit(
    tmp_path: Path,
) -> None:
    seeded = _seeded_validation_retry_successor(tmp_path)
    successor = seeded["successor"]
    bridge = seeded["bridge"]
    paths = seeded["paths"]
    _mutate_portal_retry_state(
        paths,
        alias=successor.task_alias,
        task_cid=successor.task_cid,
        commit="a" * 40,
        branch="implementation/lgswf-004-attempt-2-progressed",
    )

    with pytest.raises(DatabasePortalBridgeError) as caught:
        bridge.run_provider(successor)
    assert str(caught.value) == DATABASE_PORTAL_VALIDATION_RETRY_SEED_CONFLICT_REASON
    with pytest.raises(DatabasePortalBridgeError, match="identity-bound"):
        bridge.recover_validation_retry_seed_conflict(successor)


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
    authority = portal._prior_seed_proposal_authority(projected_task)
    assert authority["ok"] is True
    assert authority["database_validation_retry_seed"] is True
    assert authority["authorized_paths"] == ["inventory/result.json"]


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
        match="canonical CID|authoritative task CID",
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


def test_implementation_conflict_matches_main_module_alias() -> None:
    class DatabaseImplementationConflictError(RuntimeError):
        pass

    assert _is_implementation_conflict(
        DatabaseImplementationConflictError("stale row")
    )
    assert not _is_implementation_conflict(RuntimeError("stale row"))


def test_post_merge_recovery_is_inert_without_bound_queue(tmp_path: Path) -> None:
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: _CompletingPortal(paths, alias),
    )
    assert bridge.merge_queue is None
    assert bridge.recover_post_merge_declared_outputs(object()) is None


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


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_merge_train_recovery_is_inert_until_bound(tmp_path: Path) -> None:
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        owner_session_id="session:merge-train-recovery",
        authority_mode="embedded_exclusive",
        task_source_kind="duckdb",
        require_real_execution=True,
    )
    try:
        settlement = daemon._settle_invalid_metadata_portal_quarantines()
        recovery = daemon._run_post_merge_recovery()
        assert settlement["attempted"] is False
        assert settlement["reason"] == "merge_train_recovery_not_configured"
        assert settlement["write_count"] == 0
        assert recovery["attempted"] is False
        assert recovery["reason"] == "post_merge_recovery_not_configured"
        assert recovery["write_count"] == 0
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="bound queue and target branch",
        ):
            daemon.bind_merge_train_recovery(
                merge_queue=None,
                repo_root=tmp_path,
                merge_target_branch="",
            )
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_merge_train_recovery_bind_is_one_shot(tmp_path: Path) -> None:
    from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
        checkout_repository_id,
    )
    from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import MergeQueue

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
    (repo / "README").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "add", "README"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "seed"], cwd=repo, check=True)
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        owner_session_id="session:merge-train-bind",
        authority_mode="embedded_exclusive",
        task_source_kind="duckdb",
        require_real_execution=True,
    )
    queue = MergeQueue(
        tmp_path / "queue",
        target_repository_id=checkout_repository_id(repo),
        target_branch="main",
        require_target_binding=True,
    )
    try:
        daemon.bind_merge_train_recovery(
            merge_queue=queue,
            repo_root=repo,
            merge_target_branch="main",
        )
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="already bound",
        ):
            daemon.bind_merge_train_recovery(
                merge_queue=queue,
                repo_root=repo,
                merge_target_branch="main",
            )
        settlement = daemon._settle_invalid_metadata_portal_quarantines()
        assert settlement["attempted"] is True
        assert settlement["settled"] == 0
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_post_merge_rearm_endpoints_fail_closed_on_invalid_payloads(
    tmp_path: Path,
) -> None:
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        owner_session_id="session:post-merge-rearm",
        authority_mode="embedded_exclusive",
        task_source_kind="duckdb",
        require_real_execution=True,
    )
    try:
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="preauthorization source is invalid",
        ):
            daemon.preauthorize_post_merge_declared_output_recovery({})
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="preauthorization source is invalid",
        ):
            daemon.preauthorize_false_completed_merge_recovery({})
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="recovery schema is invalid",
        ):
            daemon.recover_blocked_post_merge_declared_outputs(
                {"schema": "not-a-recovery-schema"}
            )
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="reintegration receipt is malformed",
        ):
            daemon.recover_blocked_false_completed_merge(
                {"schema": "not-a-recovery-schema"}
            )
        captured: list[object] = []
        daemon.bind_post_merge_recovery(lambda: captured.append("bound") or None)
        result = daemon._run_post_merge_recovery()
        assert captured == ["bound"]
        assert result["attempted"] is True
        assert result["recovered"] is False
        assert result["reason"] == "no_recoverable_post_merge_request"
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_post_merge_wrapper_accepts_false_completion_reintegration_schema(
    tmp_path: Path,
) -> None:
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        owner_session_id="session:false-completion-wrapper",
        authority_mode="embedded_exclusive",
        task_source_kind="duckdb",
        require_real_execution=True,
    )
    expected = {
        "schema": DATABASE_FALSE_COMPLETION_REINTEGRATION_RECOVERY_SCHEMA,
        "attempted": True,
        "recovered": True,
        "changed": True,
        "write_count": 1,
        "request_id": "request:false-completion",
    }
    try:
        daemon.bind_post_merge_recovery(lambda: dict(expected))

        assert daemon._run_post_merge_recovery() == expected
    finally:
        daemon.close()


def test_false_completion_recheck_reuses_observation_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lease recheck must reproduce the discovery receipt byte-for-byte."""

    bridge = object.__new__(DatabasePortalExecutionBridge)
    request = SimpleNamespace(
        status="completed",
        metadata={},
        request_id="request:false-completion",
        canonical_task_id="task:false-completion",
        canonical_task_key="task:false-completion",
        dedupe_key="dedupe:false-completion",
        commit_sha="a" * 40,
        target_repository_id="repository:false-completion",
        target_branch="main",
    )
    observation = {
        "false_completion_observation": {
            "observed_target_commit": "b" * 40,
            "completed_claim_generation": 2,
            "completed_finished_at_hex": (50.0).hex(),
            "completed_row_digest": "sha256:" + "c" * 64,
        },
        "false_completion_observation_id": "observation:false-completion",
    }
    observed_times = iter((100.0, 200.0))
    monkeypatch.setattr(
        database_portal_bridge_module.time,
        "time",
        lambda: next(observed_times),
    )

    selected = bridge._false_completion_queue_recovery_receipt(
        request,
        observation,
    )
    assert selected is not None
    rechecked = bridge._false_completion_queue_recovery_receipt(
        request,
        observation,
        observed_at=float(selected["observed_at"]),
    )
    assert rechecked == selected
    assert rechecked["receipt_cid"] == selected["receipt_cid"]

    independently_observed = bridge._false_completion_queue_recovery_receipt(
        request,
        observation,
    )
    assert independently_observed is not None
    assert independently_observed["observed_at"] == 200.0
    assert independently_observed["receipt_cid"] != selected["receipt_cid"]


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_configured_runner_binds_post_merge_recovery_when_queue_is_target_bound(
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
        checkout_repository_id,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
        bind_database_portal_execution_from_args,
    )

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
    (repo / "README").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "add", "README"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "seed"], cwd=repo, check=True)
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        owner_session_id="session:post-merge-bind",
        authority_mode="embedded_exclusive",
        task_source_kind="duckdb",
        require_real_execution=True,
    )
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
            "--merge-queue-dir",
            str(tmp_path / "queue"),
            "--merge-target-branch",
            "main",
            "--implement",
            "--once",
        ]
    )
    try:
        bind_database_portal_execution_from_args(
            daemon,
            args,
            repo_root=repo,
            portal_daemon_class=PortalImplementationDaemon,
        )
        assert daemon._merge_queue is not None
        assert daemon._merge_target_branch == "main"
        assert daemon._post_merge_recovery_fn is not None
        assert daemon._quack_preprojection_transport_recovery_fn is not None
        assert (
            daemon._quack_preprojection_transport_recovery_fn.__name__
            == "recover_quack_preprojection_transport_failure"
        )
        assert checkout_repository_id(repo) == daemon._merge_queue.target_repository_id
    finally:
        daemon.close()


def test_closed_unaccepted_implementation_reason_consumes_protected_path_mutation() -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
        closed_unaccepted_implementation_reason,
    )

    events = [
        {"type": "implementation_started", "task_id": "ASEH-061"},
        {
            "type": "implementation_finished",
            "task_id": "ASEH-061",
            "canonical_task_cid": "cid-061",
            "error": "implementation_protected_path_mutated",
        },
    ]
    assert (
        closed_unaccepted_implementation_reason(
            events, alias="ASEH-061", task_cid="cid-061"
        )
        == "implementation_protected_path_mutated"
    )
    assert (
        closed_unaccepted_implementation_reason(
            events, alias="ASEH-062", task_cid="cid-061"
        )
        == ""
    )
    assert closed_unaccepted_implementation_reason((), alias="ASEH-061") == ""
