"""Focused safety tests for database-authoritative Portal execution."""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ContextCompilationReceipt,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import (
    ContextBudgetResolution,
)
from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
    board_scoped_checkout_mutation_lock_path,
    checkout_mutation_lock_path,
    checkout_repository_id,
)
from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import (
    MergeQueue,
    MergeRequest,
)
from ipfs_accelerate_py.agent_supervisor.merge.merge_train import MergeTrain
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.benchmark import (
    MANIFEST_SCHEMA as VRIF_BENCHMARK_MANIFEST_SCHEMA,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.benchmark import (
    build_frozen_benchmark_contract,
    load_frozen_benchmark,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.benchmark import (
    sha256_identity as vrif_sha256_identity,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    PROGRAM_ID as VRIF_PROGRAM_ID,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    ResidualTaskFamily,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.release import (
    ResidualIntelligenceReleaseReport,
    render_vrif_release_report_markdown,
    validate_release_claims,
)
from ipfs_accelerate_py.agent_supervisor.runtime.event_log import append_jsonl_event
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DATABASE_PORTAL_CAPACITY_RETRY_SCHEMA,
    DATABASE_PORTAL_CHECKOUT_CONTENTION_BACKOFF_SECONDS,
    DATABASE_PORTAL_COMPLETION_BINDING_SCHEMA,
    DATABASE_PORTAL_CONSUMED_ATTEMPT_RETRY_SCHEMA,
    DATABASE_PORTAL_CONSUMED_ATTEMPT_RETRY_SEED_SCHEMA,
    DATABASE_PORTAL_CONSUMED_NO_PROGRESS_SCHEMA,
    DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA,
    DATABASE_PORTAL_POOLED_WORKTREE_CREATE_FAILED_REASON,
    DATABASE_PORTAL_POOLED_WORKTREE_CREATE_RECOVERY_SCHEMA,
    DATABASE_PORTAL_PROTECTED_PATH_PRESERVATION_SCHEMA,
    DATABASE_PORTAL_PROTECTED_RECONCILIATION_SELF_LOCK_SCHEMA,
    DATABASE_PORTAL_RETRY_DEFERRAL_SCHEMA,
    DATABASE_PORTAL_SKIP_CONTENTION_REASONS,
    DATABASE_PORTAL_VALIDATION_RETRY_ORDER_REPAIR_SCHEMA,
    DATABASE_PORTAL_VALIDATION_RETRY_SCHEMA,
    DATABASE_PORTAL_VALIDATION_RETRY_SEED_CONFLICT_REASON,
    DATABASE_PORTAL_VALIDATION_RETRY_SEED_CONFLICT_RECOVERY_SCHEMA,
    INFLIGHT_PROCESS_BACKOFF_SECONDS,
    DatabasePortalBridgeConsumedNoProgressError,
    DatabasePortalBridgeDeferred,
    DatabasePortalBridgeError,
    DatabasePortalCandidateRetry,
    DatabasePortalCapacityRetry,
    DatabasePortalConsumedAttemptTerminal,
    DatabasePortalExecutionBridge,
    DatabasePortalProtectedPathPreserved,
    DatabasePortalValidationRetry,
    _is_implementation_conflict,
    database_portal_consumed_no_progress_fingerprint,
    database_portal_task_contract_digest,
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
    ImplementationDiagnosticReceipt,
    PortalImplementationDaemon,
    PortalTaskState,
    parse_args,
    parse_task_file,
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


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_database_portal_attempt_isolates_foreign_merge_history_and_dequeue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Attempt Scope Test",
            "-c",
            "user.email=attempt-scope@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "base",
        ],
        cwd=repo,
        check=True,
    )

    record = _record()
    record.dependencies = ()
    record.status = "in_progress"
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: None,
        repository_root=repo,
        task_header_prefix="## LGSWF-",
    )
    paths, _binding = bridge._ensure_attempt_projection(_attempt(), record)
    [projected_task] = parse_task_text(
        paths.task_projection.read_text(encoding="utf-8"),
        path=paths.task_projection,
        task_header_prefix="## LGSWF-",
    )

    repository_id = checkout_repository_id(repo)
    attempt_target_branch = "implementation/task-projection"
    queue = MergeQueue(
        tmp_path / "merge-queue",
        target_repository_id=repository_id,
        target_branch=attempt_target_branch,
        require_target_binding=True,
    )

    def metadata(*, owned: bool) -> dict[str, object]:
        root = paths.root if owned else tmp_path / "older-attempt"
        return {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/merge-candidate@3"
            ),
            "todo_path": str(
                paths.task_projection if owned else root / "task-projection.md"
            ),
            "state_path": str(paths.state if owned else root / "state.json"),
            "strategy_path": str(
                paths.strategy if owned else root / "strategy.json"
            ),
            "events_path": str(
                paths.events if owned else root / "events.jsonl"
            ),
            "completion_task_cids": {
                projected_task.task_id: projected_task.canonical_task_cid
            },
        }

    def enqueue(commit: str, *, owned: bool) -> object:
        return queue.enqueue(
            branch_name=f"implementation/{commit[0]}",
            task_id=projected_task.task_id,
            canonical_task_id=projected_task.canonical_task_cid,
            canonical_task_key=projected_task.canonical_task_key,
            commit_sha=commit,
            metadata=metadata(owned=owned),
        )

    stale_completed = enqueue("a" * 40, owned=False)
    claimed_stale = queue.claim_pending_request(
        stale_completed.request_id,
        consumer_id="merge-train:stale-attempt",
    )
    assert claimed_stale is not None
    queue.complete(claimed_stale)
    foreign_pending = enqueue("b" * 40, owned=False)

    daemon = PortalImplementationDaemon(
        todo_path=paths.task_projection,
        state_path=paths.state,
        strategy_path=paths.strategy,
        events_path=paths.events,
        repo_root=repo,
        task_header_prefix="## LGSWF-",
        merge_queue=queue,
        merge_target_branch="main",
        isolate_merge_queue_to_task_projection=True,
        worktree_pool_enabled=False,
        maintenance_interval_seconds=0,
    )
    [task] = daemon._load_tasks()

    assert daemon.merge_queue is queue
    assert daemon._task_projection_merge_queue is not queue
    assert daemon._shared_merge_queue_task_cids(
        "completed_canonical_task_ids"
    ) == set()
    assert daemon._shared_completed_task_cid_bindings() == {}
    assert daemon._shared_merge_queue_task_cids(
        "active_canonical_task_ids"
    ) == set()
    assert daemon._task_has_blocking_pending_merge(task) is False

    result = daemon.run_once()
    assert result["active_task_id"] == projected_task.task_id
    assert result["shared_completed_task_ids"] == []
    assert result["shared_active_merge_task_ids"] == []
    assert queue.get(foreign_pending.request_id).status == "pending"

    current_pending = enqueue("c" * 40, owned=True)
    assert daemon._shared_merge_queue_task_cids(
        "active_canonical_task_ids"
    ) == {projected_task.canonical_task_cid}
    assert daemon._task_has_blocking_pending_merge(task) is True

    observed: dict[str, object] = {}

    def claim_scoped_request(train: MergeTrain) -> dict[str, object]:
        observed["queue"] = train.queue
        request = train._dequeue()
        observed["request"] = request
        return {"request_id": getattr(request, "request_id", "")}

    monkeypatch.setattr(MergeTrain, "run_once", claim_scoped_request)
    progress = daemon._consume_one_merge_candidate()

    assert progress == {"request_id": current_pending.request_id}
    assert observed["queue"] is daemon._task_projection_merge_queue
    assert getattr(observed["request"], "request_id", "") == (
        current_pending.request_id
    )
    assert queue.get(current_pending.request_id).status == "processing"
    assert queue.get(foreign_pending.request_id).status == "pending"


class _TaskSource:
    def __init__(self, record: object) -> None:
        self.record = record

    def get_task(self, task_cid: str) -> object | None:
        return self.record if task_cid == "task:cid:004" else None

    def snapshot(self) -> object:
        return SimpleNamespace(repository_tree_id="tree:control-plane-current")


class _CompletingPortal:
    def __init__(
        self,
        paths: object,
        task_alias: str,
        *,
        baseline_commit: str = "b" * 40,
        implementation_commit: str = "a" * 40,
    ) -> None:
        self.paths = paths
        self.task_alias = task_alias
        self.baseline_commit = baseline_commit
        self.implementation_commit = implementation_commit
        self.closed = False

    def run_once(self) -> dict[str, object]:
        binding = json.loads(self.paths.binding.read_text(encoding="utf-8"))
        canonical_task_cid = str(binding["task_cid"])
        canonical_task_key = str(binding["canonical_task_key"])
        text = self.paths.task_projection.read_text(encoding="utf-8")
        self.paths.task_projection.write_text(
            text.replace("- Status: ready", "- Status: completed"),
            encoding="utf-8",
        )
        self.paths.state.write_text(
            json.dumps(
                {
                    "last_implementation_commit": self.implementation_commit,
                    "last_merge_returncode": 0,
                }
            ),
            encoding="utf-8",
        )
        append_jsonl_event(
            self.paths.events,
            "implementation_finished",
            {
                "task_id": self.task_alias,
                "canonical_task_cid": canonical_task_cid,
                "attempt": 1,
                "returncode": 0,
                "baseline_ref": self.baseline_commit,
                "implementation_commit": self.implementation_commit,
                "validation_result": {
                    "attempted": True,
                    "passed": True,
                    "returncode": 0,
                },
                "merge_result": {
                    "merged": True,
                    "queued": False,
                    "reason": "merged",
                },
            },
        )
        append_jsonl_event(
            self.paths.events,
            "task_completed",
            {
                "task_id": self.task_alias,
                "canonical_task_key": canonical_task_key,
                "canonical_task_cid": canonical_task_cid,
            },
        )
        return {
            "task_count": 1,
            "completed_count": 1,
            "active_task_id": self.task_alias,
            "implementation_result": {
                "task_id": self.task_alias,
                "returncode": 0,
                "implementation_commit": self.implementation_commit,
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
            lines = self.paths.events.read_text(encoding="utf-8").splitlines()
            event = json.loads(lines[-1])
            assert event["type"] == "task_completed"
            event["canonical_task_cid"] = "task:cid:other"
            self.paths.events.write_text(
                "\n".join([*lines[:-1], json.dumps(event)]) + "\n",
                encoding="utf-8",
            )
            return result

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: ForgedCompletionPortal(paths, alias),
        max_passes=1,
    )

    with pytest.raises(DatabasePortalBridgeError, match="matching durable"):
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


def _git_completion_lineage(repo: Path) -> tuple[str, str, str]:
    repo.mkdir(parents=True, exist_ok=True)
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
    artifact = repo / "inventory" / "result.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"baseline":true}\n', encoding="utf-8")
    subprocess.run(["git", "add", "--", "inventory/result.json"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=repo, check=True)
    baseline_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    baseline_tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    artifact.write_text('{"candidate":true}\n', encoding="utf-8")
    subprocess.run(["git", "add", "--", "inventory/result.json"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "implementation"], cwd=repo, check=True)
    implementation_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return baseline_commit, baseline_tree, implementation_commit


_VRIF_BENCHMARK_VALIDATION = (
    "python -m pytest -q test/api/residual_intelligence/test_benchmark.py"
)
_VRIF_BENCHMARK_OUTPUTS = (
    "benchmarks/agent_supervisor/residual_intelligence/manifest.json",
    "benchmarks/agent_supervisor/residual_intelligence/cases.jsonl",
    "test/api/residual_intelligence/test_benchmark.py",
)
_VRIF_TERMINAL_OUTPUTS = (
    (
        "docs/architecture/residual_intelligence_inventory/"
        "final_release_report.json"
    ),
    (
        "docs/architecture/residual_intelligence_inventory/"
        "final_release_report.md"
    ),
    "test/api/residual_intelligence/test_release_report.py",
)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _write_git_fixture(repo: Path, path: str, payload: bytes) -> None:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)


def _commit_git_fixture(repo: Path, message: str) -> str:
    subprocess.run(["git", "add", "--all"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", message], cwd=repo, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_tree(repo: Path, commit: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", f"{commit}^{{tree}}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _init_git_fixture(repo: Path) -> None:
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "vrif-acceptance@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "VRIF Acceptance Test"],
        cwd=repo,
        check=True,
    )


def _vrif_benchmark_record(*, task_cid: str = "task:vrif:030") -> SimpleNamespace:
    return SimpleNamespace(
        task_cid=task_cid,
        task_alias="VRIF-030",
        goal_cid="goal:vrif:release",
        plan_cid="plan:vrif:1",
        revision=31,
        priority="P0",
        dependencies=(),
        outputs=tuple({"path": path} for path in _VRIF_BENCHMARK_OUTPUTS),
        validations=({"argv": [_VRIF_BENCHMARK_VALIDATION]},),
        acceptance=({"criterion": "The frozen benchmark is owner-exact"},),
        body={
            "objective": "Publish the owner-exact frozen benchmark",
            "completion": "auto",
            "track": "benchmark",
        },
    )


def _git_vrif_benchmark_lineage(
    repo: Path,
    *,
    self_consistent_wrong_binding: bool = False,
) -> tuple[str, str, str]:
    """Build a synthetic benchmark using the trusted pure VRIF builder."""

    _init_git_fixture(repo)
    objective_paths = (
        "docs/architecture/agent_supervisor_residual_intelligence.objectives.md",
        "docs/architecture/agent_supervisor_residual_intelligence.todo.md",
    )
    operation_path = "ipfs_accelerate_py/agent_supervisor/control/control_plane.py"
    provider_path = "config/agent_supervisor_residual_intelligence_scheduler.json"
    admission_path = (
        "benchmarks/agent_supervisor/residual_intelligence/"
        "synthetic_training_admission.json"
    )
    split_path = (
        "benchmarks/agent_supervisor/residual_intelligence/"
        "synthetic_split_manifest.json"
    )
    inventory_path = (
        "docs/architecture/residual_intelligence_inventory/"
        "residual_model_call_inventory.json"
    )
    implementation_inputs = {
        objective_paths[0]: b"# Synthetic objectives\n",
        objective_paths[1]: b"# Synthetic tasks\n",
        operation_path: b"# Synthetic operation catalogue\n",
        provider_path: b'{"provider":"synthetic"}\n',
        inventory_path: b'{"model_calls":[]}\n',
    }
    split_root = vrif_sha256_identity({"fixture": "synthetic split"})
    split = {"split_root": split_root}
    admission_body = {
        "schema": "vrif-acceptance-test-admission@1",
        "disposition": "training_unavailable",
        "corpus_root": vrif_sha256_identity({"fixture": "corpus"}),
        "source_rights_root": vrif_sha256_identity({"fixture": "rights"}),
        "split_root": split_root,
    }
    admission = {
        **admission_body,
        "admission_id": content_identity(admission_body),
    }
    implementation_inputs[admission_path] = _json_bytes(admission)
    implementation_inputs[split_path] = _json_bytes(split)
    for path, payload in implementation_inputs.items():
        _write_git_fixture(repo, path, payload)
    for path in _VRIF_BENCHMARK_OUTPUTS:
        _write_git_fixture(repo, path, b"# baseline placeholder\n")
    baseline_commit = _commit_git_fixture(repo, "synthetic VRIF baseline")
    baseline_tree = _git_tree(repo, baseline_commit)

    benchmark_test = (
        b"def test_synthetic_owner_exact_benchmark():\n"
        b"    assert True\n"
    )
    _write_git_fixture(repo, _VRIF_BENCHMARK_OUTPUTS[2], benchmark_test)
    objective_artifacts = {
        path: vrif_sha256_identity(implementation_inputs[path])
        for path in objective_paths
    }
    inventory_identity = vrif_sha256_identity(implementation_inputs[inventory_path])
    base_bindings = {
        "repository_states": vrif_sha256_identity(
            {"commit": baseline_commit, "tree": baseline_tree}
        ),
        "objective_revisions": vrif_sha256_identity(
            {
                "schema": (
                    "ipfs_accelerate_py/agent-supervisor/"
                    "residual-benchmark-objective-revisions@1"
                ),
                "artifacts": objective_artifacts,
            }
        ),
        "operation_catalog": vrif_sha256_identity(
            implementation_inputs[operation_path]
        ),
        "provider_policy": vrif_sha256_identity(
            implementation_inputs[provider_path]
        ),
        "tokenizer": vrif_sha256_identity(
            {
                "admission_id": admission["admission_id"],
                "disposition": "no_learned_tokenizer_admitted",
            }
        ),
        "model_versions": vrif_sha256_identity(
            {
                "inventory_blob_identity": inventory_identity,
                "disposition": "training_unavailable",
            }
        ),
        "validation_policy": vrif_sha256_identity(
            {
                "argv": [[_VRIF_BENCHMARK_VALIDATION]],
                "test_blob_identity": vrif_sha256_identity(benchmark_test),
            }
        ),
    }
    if self_consistent_wrong_binding:
        base_bindings["model_versions"] = vrif_sha256_identity(
            {
                "inventory_blob_identity": inventory_identity,
                "disposition": "self_consistent_but_not_owner_computed",
            }
        )
    contract = build_frozen_benchmark_contract(
        task_families=[family.value for family in ResidualTaskFamily],
        source_commit=baseline_commit,
        source_tree=baseline_tree,
        split_root=split_root,
        base_bindings=base_bindings,
    )
    manifest = {
        "schema": VRIF_BENCHMARK_MANIFEST_SCHEMA,
        "program_identifier": VRIF_PROGRAM_ID,
        "status": "staged_not_qualified",
        "owner_task": "VRIF-030",
        "source_revision": baseline_commit,
        "partitions": contract["partitions"],
        "required_case_kinds": contract["case_kinds"],
        "task_families": [family.value for family in ResidualTaskFamily],
        "training_admission": "training_unavailable",
        "weights_committed": False,
        "large_corpus_committed": False,
        "promotion_evidence": False,
        "benchmark_freeze": contract["benchmark_freeze"],
    }
    cases = b"".join(_json_bytes(case) for case in contract["cases"])
    _write_git_fixture(repo, _VRIF_BENCHMARK_OUTPUTS[0], _json_bytes(manifest))
    _write_git_fixture(repo, _VRIF_BENCHMARK_OUTPUTS[1], cases)
    implementation_commit = _commit_git_fixture(
        repo,
        "synthetic VRIF benchmark implementation",
    )
    return baseline_commit, baseline_tree, implementation_commit


def _vrif_terminal_record() -> SimpleNamespace:
    return SimpleNamespace(
        task_cid="task:vrif:032",
        task_alias="VRIF-032",
        goal_cid="goal:vrif:root",
        plan_cid="plan:vrif:1",
        revision=15,
        priority="P0",
        dependencies=(),
        outputs=tuple({"path": path} for path in _VRIF_TERMINAL_OUTPUTS),
        validations=(),
        acceptance=({"criterion": "The terminal report is owner-exact"},),
        body={
            "objective": "Publish the final root-gated release report",
            "completion": "auto",
            "track": "release",
        },
    )


def _git_vrif_terminal_lineage(
    repo: Path,
    *,
    noncanonical_markdown: bool = False,
    wrong_baseline_tree: bool = False,
) -> tuple[str, str, str]:
    """Build a synthetic release pair using the trusted typed renderer."""

    _init_git_fixture(repo)
    _write_git_fixture(repo, _VRIF_TERMINAL_OUTPUTS[0], b'{"baseline":true}\n')
    _write_git_fixture(repo, _VRIF_TERMINAL_OUTPUTS[1], b"# Baseline report\n")
    _write_git_fixture(repo, _VRIF_TERMINAL_OUTPUTS[2], b"# baseline test\n")
    baseline_commit = _commit_git_fixture(repo, "synthetic VRIF terminal baseline")
    baseline_tree = _git_tree(repo, baseline_commit)

    source_report = (
        Path(__file__).resolve().parents[2]
        / "docs/architecture/residual_intelligence_inventory/"
        "final_release_report.json"
    )
    report = json.loads(source_report.read_text(encoding="utf-8"))
    evaluated_tree = "f" * 40 if wrong_baseline_tree else baseline_tree
    report["end_tree"] = evaluated_tree
    report["drift"]["evaluated_tree"] = evaluated_tree
    typed = validate_release_claims(
        ResidualIntelligenceReleaseReport.from_dict(report)
    )
    report = typed.to_dict()
    markdown = render_vrif_release_report_markdown(report).encode("utf-8")
    if noncanonical_markdown:
        markdown += b"\n<!-- self-consistent substring checks could miss this -->\n"
    _write_git_fixture(repo, _VRIF_TERMINAL_OUTPUTS[0], _json_bytes(report))
    _write_git_fixture(repo, _VRIF_TERMINAL_OUTPUTS[1], markdown)
    _write_git_fixture(
        repo,
        _VRIF_TERMINAL_OUTPUTS[2],
        b"def test_synthetic_terminal_report():\n    assert True\n",
    )
    implementation_commit = _commit_git_fixture(
        repo,
        "synthetic VRIF terminal implementation",
    )
    return baseline_commit, baseline_tree, implementation_commit


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
            "owner_session_id": "session:bridge",
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
        changed_paths: tuple[str, ...] = ("inventory/result.json",),
    ) -> None:
        self.paths = paths
        self.task_alias = task_alias
        self.commit = commit
        self.rescue_branch = rescue_branch
        self.denied_paths = denied_paths
        self.changed_paths = changed_paths

    def run_once(self) -> dict[str, object]:
        changed_paths = list(self.changed_paths)
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
    record.status = "retrying"
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


def _verified_quota_fallback_result(
    *,
    task_alias: str = "LGSWF-004",
    task_cid: str = "task:cid:004",
) -> dict[str, object]:
    receipt = {
        "receipt_id": "sha256:" + "a" * 64,
        "failure_class": "hard_quota_exhausted",
        "primary_dispatched": False,
        "evidence_sha256": "sha256:" + "b" * 64,
    }
    return {
        "implementation_result": {
            "task_id": task_alias,
            "canonical_task_cid": task_cid,
            "attempt": 1,
            "returncode": 1,
            "deferred": True,
            "reason": "provider_capacity_exhausted",
            "attempt_consumed": False,
            "task_prompt_dispatched": False,
            "providers": ["grok"],
            "failure_class": "hard_quota_exhausted",
            "hard_quota_exhausted_providers": ["grok"],
            "backoff_seconds": 300,
            "quota_fallback_authority": {
                "schema": (
                    "ipfs_accelerate_py.agent_supervisor."
                    "grok-quota-fallback-authority@2"
                ),
                "primary_provider": "grok",
                "primary_model": "grok-4.6",
                "failure_class": "hard_quota_exhausted",
                "evidence_sha256": "sha256:" + "b" * 64,
                "task_id": task_alias,
                "canonical_task_cid": task_cid,
                "attempt": 1,
                "primary_returncode": 1,
                "start_event_id": "sha256:" + "c" * 64,
                "start_sequence": 1,
                "command_sha256": "sha256:" + "d" * 64,
                "runner_receipt_id": receipt["receipt_id"],
                "runner_receipt": receipt,
            },
        }
    }


def test_bridge_keeps_verified_quota_fallback_in_same_portal_claim(
    tmp_path: Path,
) -> None:
    class QuotaThenCodexPortal(_CompletingPortal):
        def __init__(self, paths: object, task_alias: str) -> None:
            super().__init__(paths, task_alias)
            self.calls = 0

        def run_once(self) -> dict[str, object]:
            self.calls += 1
            if self.calls > 1:
                return super().run_once()
            return _verified_quota_fallback_result(task_alias=self.task_alias)

    portals: list[QuotaThenCodexPortal] = []

    def factory(paths: object, alias: str) -> QuotaThenCodexPortal:
        portal = QuotaThenCodexPortal(paths, alias)
        portals.append(portal)
        return portal

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=factory,
        max_passes=2,
    )

    provider = bridge.run_provider(_attempt())

    assert provider["accepted"] is True
    assert len(portals) == 1
    assert portals[0].calls == 2


def test_bridge_rejects_quota_fallback_for_a_foreign_task(
    tmp_path: Path,
) -> None:
    class ForeignQuotaPortal:
        def __init__(self) -> None:
            self.calls = 0

        def run_once(self) -> dict[str, object]:
            self.calls += 1
            return _verified_quota_fallback_result(
                task_cid="task:cid:foreign"
            )

    portal = ForeignQuotaPortal()
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: portal,
        max_passes=3,
    )

    with pytest.raises(DatabasePortalBridgeDeferred) as caught:
        bridge.run_provider(_attempt())

    assert caught.value.reason == "provider_capacity_exhausted"
    assert portal.calls == 1


def test_bridge_continues_the_same_quota_fallback_at_most_once(
    tmp_path: Path,
) -> None:
    class RepeatingQuotaPortal:
        def __init__(self) -> None:
            self.calls = 0

        def run_once(self) -> dict[str, object]:
            self.calls += 1
            return _verified_quota_fallback_result()

    portal = RepeatingQuotaPortal()
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: portal,
        max_passes=4,
    )

    with pytest.raises(DatabasePortalBridgeDeferred) as caught:
        bridge.run_provider(_attempt())

    assert caught.value.reason == "provider_capacity_exhausted"
    assert portal.calls == 2


def test_bridge_does_not_continue_unverified_quota_fallback(
    tmp_path: Path,
) -> None:
    class UnverifiedQuotaPortal:
        def run_once(self) -> dict[str, object]:
            return {
                "implementation_result": {
                    "returncode": 1,
                    "deferred": True,
                    "reason": "provider_capacity_exhausted",
                    "attempt_consumed": False,
                    "task_prompt_dispatched": False,
                    "providers": ["grok"],
                    "failure_class": "hard_quota_exhausted",
                    "hard_quota_exhausted_providers": ["grok"],
                    "backoff_seconds": 300,
                    "quota_fallback_authority": {
                        "schema": "unverified",
                    },
                }
            }

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: UnverifiedQuotaPortal(),
        max_passes=2,
    )

    with pytest.raises(DatabasePortalBridgeDeferred) as caught:
        bridge.run_provider(_attempt())

    assert caught.value.reason == "provider_capacity_exhausted"
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


@pytest.mark.parametrize("reason", sorted(DATABASE_PORTAL_SKIP_CONTENTION_REASONS))
def test_bridge_defers_skipped_inflight_and_lock_contention(
    tmp_path: Path,
    reason: str,
) -> None:
    class SkipContentionPortal:
        def run_once(self) -> dict[str, object]:
            return {
                "implementation_result": {
                    "skipped": True,
                    "reason": reason,
                    "task_id": "PCPC-024",
                    "attempt": 1,
                }
            }

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: SkipContentionPortal(),
        max_passes=1,
    )

    with pytest.raises(DatabasePortalBridgeDeferred) as caught:
        bridge.run_provider(_attempt())

    assert caught.value.reason == reason
    assert caught.value.backoff_seconds == (
        DATABASE_PORTAL_CHECKOUT_CONTENTION_BACKOFF_SECONDS
    )
    assert caught.value.attempt_consumed is False
    assert caught.value.provider_dispatched is False


def test_bridge_polls_exact_inflight_process_on_same_claim_until_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClock:
        def __init__(self) -> None:
            self.now = 0.0
            self.sleeps: list[float] = []

        def monotonic(self) -> float:
            return self.now

        def sleep(self, seconds: float) -> None:
            assert 0.0 < seconds <= 15.0
            self.sleeps.append(seconds)
            self.now += seconds

    class InflightThenCompletingPortal(_CompletingPortal):
        def __init__(self, paths: object, task_alias: str) -> None:
            super().__init__(paths, task_alias)
            self.calls = 0
            self.worktree_path = str(tmp_path / "exact-inflight-worktree")

        def run_once(self) -> dict[str, object]:
            self.calls += 1
            if self.calls <= 2:
                return {
                    "implementation_result": {
                        "skipped": True,
                        "reason": "inflight_process",
                        "task_id": self.task_alias,
                        "attempt": 1,
                        "worktree_path": self.worktree_path,
                    }
                }
            return super().run_once()

    clock = FakeClock()
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon."
        "database_portal_bridge._monotonic_seconds",
        clock.monotonic,
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon."
        "database_portal_bridge._sleep_seconds",
        clock.sleep,
    )
    portals: list[InflightThenCompletingPortal] = []

    def factory(paths: object, alias: str) -> InflightThenCompletingPortal:
        portal = InflightThenCompletingPortal(paths, alias)
        portals.append(portal)
        return portal

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=factory,
        # Inflight polls do not consume the ordinary Portal pass budget.
        max_passes=1,
        implementation_timeout=31.0,
    )

    provider = bridge.run_provider(_attempt())

    assert provider["accepted"] is True
    assert len(portals) == 1
    assert portals[0].calls == 3
    assert portals[0].closed is True
    assert clock.sleeps == [15.0, 15.0]


def test_bridge_bounds_inflight_pass_preview_with_streaming_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClock:
        def __init__(self) -> None:
            self.now = 0.0

        def monotonic(self) -> float:
            return self.now

        def sleep(self, seconds: float) -> None:
            assert 0.0 < seconds <= 15.0
            self.now += seconds

    class LongInflightThenCompletingPortal(_CompletingPortal):
        def __init__(self, paths: object, task_alias: str) -> None:
            super().__init__(paths, task_alias)
            self.calls = 0

        def run_once(self) -> dict[str, object]:
            self.calls += 1
            if self.calls <= 20:
                return {
                    "implementation_result": {
                        "skipped": True,
                        "reason": "inflight_process",
                        "task_id": self.task_alias,
                        "canonical_task_cid": "task:cid:004",
                        "attempt": 1,
                        "worktree_path": str(tmp_path / "long-inflight-worktree"),
                    }
                }
            return super().run_once()

    clock = FakeClock()
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon."
        "database_portal_bridge._monotonic_seconds",
        clock.monotonic,
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon."
        "database_portal_bridge._sleep_seconds",
        clock.sleep,
    )
    portals: list[LongInflightThenCompletingPortal] = []

    def factory(paths: object, alias: str) -> LongInflightThenCompletingPortal:
        portal = LongInflightThenCompletingPortal(paths, alias)
        portals.append(portal)
        return portal

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=factory,
        max_passes=1,
        implementation_timeout=301.0,
    )

    provider = bridge.run_provider(_attempt())
    evidence = provider["portal_evidence"]

    assert len(portals) == 1
    assert portals[0].calls == 21
    assert evidence["portal_pass_count"] == 21
    assert evidence["portal_passes_truncated"] is True
    assert len(evidence["portal_passes"]) == 16
    assert evidence["portal_passes_digest"].startswith("sha256:")
    assert len(evidence["portal_passes_digest"]) == len("sha256:") + 64


def test_bridge_defers_exact_inflight_process_at_configured_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClock:
        def __init__(self) -> None:
            self.now = 0.0
            self.sleeps: list[float] = []

        def monotonic(self) -> float:
            return self.now

        def sleep(self, seconds: float) -> None:
            assert 0.0 < seconds <= 15.0
            self.sleeps.append(seconds)
            self.now += seconds

    class InflightPortal:
        def __init__(self, task_alias: str) -> None:
            self.task_alias = task_alias
            self.calls = 0
            self.closed = False

        def run_once(self) -> dict[str, object]:
            self.calls += 1
            return {
                "implementation_result": {
                    "skipped": True,
                    "reason": "inflight_process",
                    "task_id": self.task_alias,
                    "attempt": 1,
                    "worktree_path": str(tmp_path / "exact-inflight-worktree"),
                }
            }

        def close_event_runtime(self) -> None:
            self.closed = True

    clock = FakeClock()
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon."
        "database_portal_bridge._monotonic_seconds",
        clock.monotonic,
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon."
        "database_portal_bridge._sleep_seconds",
        clock.sleep,
    )
    portals: list[InflightPortal] = []

    def factory(_paths: object, alias: str) -> InflightPortal:
        portal = InflightPortal(alias)
        portals.append(portal)
        return portal

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=factory,
        max_passes=1,
        implementation_timeout=20.0,
    )

    with pytest.raises(DatabasePortalBridgeDeferred) as caught:
        bridge.run_provider(_attempt())

    assert caught.value.reason == "inflight_process"
    assert caught.value.backoff_seconds == INFLIGHT_PROCESS_BACKOFF_SECONDS
    assert len(portals) == 1
    # The bridge must not start one more Portal pass after sleeping exactly
    # to the configured deadline.
    assert portals[0].calls == 2
    assert portals[0].closed is True
    assert clock.sleeps == [15.0, 5.0]


def test_bridge_keeps_generic_skip_terminal(
    tmp_path: Path,
) -> None:
    class GenericSkipPortal:
        def run_once(self) -> dict[str, object]:
            return {
                "implementation_result": {
                    "skipped": True,
                    "reason": "completion_gap_missing_precise_edit_targets",
                }
            }

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: GenericSkipPortal(),
        max_passes=1,
    )

    with pytest.raises(DatabasePortalBridgeError) as caught:
        bridge.run_provider(_attempt())

    assert not isinstance(caught.value, DatabasePortalBridgeDeferred)
    assert str(caught.value) == "completion_gap_missing_precise_edit_targets"


def test_recent_log_without_lifecycle_is_not_treated_as_live_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    (repo / "state").mkdir(parents=True)
    missing_worktree = repo / "worktrees" / "pcpc-024-attempt-1"
    log_path = repo / "pcpc-024-attempt-1.log"
    log_path.write_text("still writing\n", encoding="utf-8")
    daemon = PortalImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
    )
    event = {
        "worktree_path": str(missing_worktree),
        "log_path": str(log_path),
        "command": [
            "python",
            "-m",
            "ipfs_accelerate_py.agent_supervisor.grok_cli_runner",
        ],
    }
    monkeypatch.setattr(daemon, "_list_process_commands", lambda: [])
    monkeypatch.setattr(
        daemon, "_docker_isolation_active_for_worktree", lambda _path: False
    )

    assert daemon._implementation_inflight_disposition(event) == {
        "disposition": "unverifiable",
        "reason": "recent_log_without_lifecycle_authority",
    }

    missing_worktree.mkdir(parents=True)
    assert daemon._implementation_inflight_disposition(event) == {
        "disposition": "unverifiable",
        "reason": "recent_log_without_lifecycle_authority",
    }


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


def test_bridge_defers_verified_live_supervisor_recovery_owner(
    tmp_path: Path,
) -> None:
    class LiveForeignRecoveryPortal:
        closed = False

        def run_once(self) -> dict[str, object]:
            return {
                "blocked": True,
                "reason": "external_protected_checkout_recovery_required",
                "unchanged": True,
                "write_count": 0,
                "implementation_result": {
                    "deferral_schema": DATABASE_PORTAL_RETRY_DEFERRAL_SCHEMA,
                    "retryable": True,
                    "failure_kind": "lifecycle_setup",
                    "provider_call_allowed": False,
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
                    "lock_path": str(
                        tmp_path / ".git" / "agent-checkout-mutation.lock"
                    ),
                },
            }

        def close_event_runtime(self) -> None:
            self.closed = True

    portal = LiveForeignRecoveryPortal()
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: portal,
        max_passes=1,
    )

    with pytest.raises(DatabasePortalBridgeDeferred) as caught:
        bridge.run_provider(_attempt())

    assert str(caught.value) == "external_protected_recovery_owner_active"
    assert caught.value.backoff_seconds == (
        EXTERNAL_PROTECTED_RECOVERY_BACKOFF_SECONDS
    )
    assert caught.value.attempt_consumed is False
    assert caught.value.provider_dispatched is False
    assert portal.closed is True


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


def test_bridge_defers_unbound_inflight_process_skip(tmp_path: Path) -> None:
    class InflightPortal:
        closed = False

        def run_once(self) -> dict[str, object]:
            return {
                "implementation_result": {
                    "skipped": True,
                    "reason": "inflight_process",
                    "task_id": "LGSWF-004",
                    "attempt": 1,
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
                    "reason": "generic_skip",
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
    assert str(caught.value) == "generic_skip"


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
            "owner_session_id": successor.owner_session_id,
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
            "implementation_attempts": {successor.task_alias: 2},
            "implementation_attempts_by_cid": {successor.task_cid: 2},
            "active_task_id": successor.task_alias,
            "active_task_cid": successor.task_cid,
            "active_attempt": 2,
            "implementation_in_progress": True,
            "last_implementation_task_id": successor.task_alias,
            "last_implementation_task_cid": successor.task_cid,
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
            "task_id": successor.task_alias,
            "canonical_task_cid": successor.task_cid,
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
        repository_root=repo,
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

    # Near-shapes stay fail-closed: the bridge adopts only the exact charged,
    # provider-undispatched in-flight attempt immediately following the sealed
    # retry seed.
    tampered_progress = dict(progressed)
    tampered_progress["active_attempt"] = 3
    successor_paths.state.write_text(
        json.dumps(tampered_progress, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        DatabasePortalBridgeError,
        match="retry seed state conflicts",
    ):
        progressed_bridge.run_provider(successor)
    assert progressed_calls == ["adopt"]

    successor_paths.state.write_text(
        json.dumps(progressed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    append_jsonl_event(
        successor_paths.events,
        "daemon_pass",
        {"active_task_id": successor.task_alias},
    )
    with pytest.raises(
        DatabasePortalBridgeError,
        match="retry seed state conflicts",
    ):
        progressed_bridge.run_provider(successor)
    assert progressed_calls == ["adopt"]


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
        baseline_commit,
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


def _prepare_protected_preservation_successor_seed(
    tmp_path: Path,
) -> tuple[
    SimpleNamespace,
    DatabaseTaskAttempt,
    Path,
    Path,
    str,
    str,
    str,
    dict[str, object],
]:
    (
        record,
        source_attempt,
        repo,
        source_attempt_root,
        baseline,
        preserved_commit,
        rescue_branch,
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
    subprocess.run(
        ["git", "reset", "--hard", baseline],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return (
        record,
        target,
        repo,
        target_root,
        baseline,
        preserved_commit,
        rescue_branch,
        dict(preservation_seed),
    )


class _ProtectedRecoveryQueue:
    def __init__(self, repo: Path, target_branch: str) -> None:
        self.target_repository_id = checkout_repository_id(repo)
        self.target_branch = target_branch
        self.require_target_binding = True
        self.requests: dict[str, SimpleNamespace] = {}

    def completed_requests(self) -> list[object]:
        return []

    def pending_requests(self, *, limit: int = 100) -> list[object]:
        return [
            request
            for request in self.requests.values()
            if request.status == "pending"
        ][:limit]

    def processing_requests(self, *, limit: int = 100) -> list[object]:
        return [
            request
            for request in self.requests.values()
            if request.status == "processing"
        ][:limit]

    def quarantined_requests(self) -> list[object]:
        return []

    def get(self, request_id: str) -> object | None:
        return self.requests.get(request_id)


class _ProtectedRecoveryPortal:
    def __init__(
        self,
        *,
        paths: object,
        repo: Path,
        queue: _ProtectedRecoveryQueue,
        target_branch: str,
        mode: str,
        provider_hooks: list[str],
    ) -> None:
        self.paths = paths
        self.repo_root = repo.absolute()
        self.merge_queue = queue
        self.resolved_merge_target_branch = target_branch
        self.worktree_root = paths.root / "managed-worktrees"
        self.mode = mode
        self.provider_hooks = provider_hooks
        self.reconcile_calls: list[dict[str, object]] = []
        self.cleanup_calls: list[tuple[str, str]] = []
        self.consume_calls = 0
        self.closed = False
        self.claimed = False
        self.implementation_timeout = (
            0.01 if mode == "queued_deadline" else 30.0
        )
        if mode == "wrong_repo":
            self.repo_root = repo.parent.absolute()
        if not paths.state.exists():
            paths.state.write_text(
                json.dumps(
                    {
                        "implementation_attempts": {"LGSWF-004": 2},
                        "implementation_attempts_by_cid": {
                            "task:cid:004": 2
                        },
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        if mode == "api_absent":
            self.reconcile_validated_worktree_candidate = None  # type: ignore[assignment]
        if mode == "success_todo":
            projection = paths.task_projection.read_text(encoding="utf-8")
            paths.task_projection.write_text(
                projection.replace("- Status: ready", "- Status: todo"),
                encoding="utf-8",
            )

    def run_once(self) -> dict[str, object]:
        self.provider_hooks.append("run_once")
        raise AssertionError("protected recovery dispatched a provider")

    def _load_tasks(self) -> list[object]:
        tasks = parse_task_text(
            self.paths.task_projection.read_text(encoding="utf-8"),
            path=self.paths.task_projection,
            task_header_prefix="## LGSWF-",
        )
        if self.mode == "wrong_task":
            return [
                SimpleNamespace(
                    **{
                        **asdict(tasks[0]),
                        "task_id": "LGSWF-999",
                    }
                )
            ]
        return list(tasks)

    def _implementation_task_claim_path(
        self,
        task_id: str,
        *,
        canonical_task_cid: str,
    ) -> Path:
        assert task_id == "LGSWF-004"
        assert canonical_task_cid == "task:cid:004"
        return self.paths.root / "protected-recovery-task-claim.json"

    def _build_implementation_task_claim_metadata(
        self,
        task: object,
        attempt: int,
        started_at: str,
    ) -> dict[str, object]:
        return {
            "lease_id": "protected-recovery-claim",
            "task_id": task.task_id,
            "canonical_task_cid": task.canonical_task_cid,
            "attempt": attempt,
            "started_at": started_at,
        }

    def _try_acquire_implementation_task_claim(
        self,
        path: Path,
        metadata: dict[str, object],
    ) -> tuple[bool, str, object | None]:
        assert not self.claimed
        self.claimed = True
        path.write_text(json.dumps(metadata), encoding="utf-8")
        return True, "acquired", None

    def _release_implementation_task_claim(
        self,
        path: Path,
        metadata: object,
    ) -> bool:
        self.claimed = False
        path.unlink(missing_ok=True)
        return True

    def _run_checkout_mutation_transaction(
        self,
        *,
        callback: object,
        **_kwargs: object,
    ) -> dict[str, object]:
        return callback()

    def _cleanup_merged_worktree(
        self,
        worktree_path: Path,
        branch_name: str,
        *,
        reusable: bool,
    ) -> dict[str, object]:
        assert reusable is False
        self.cleanup_calls.append((str(worktree_path), branch_name))
        removed = subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree_path)],
            cwd=self.repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        deleted = subprocess.run(
            ["git", "branch", "-D", branch_name],
            cwd=self.repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        return {
            "cleaned": removed.returncode == 0 and deleted.returncode == 0,
            "removed_worktree": removed.returncode == 0,
            "deleted_branch": deleted.returncode == 0,
        }

    def reconcile_validated_worktree_candidate(
        self,
        **kwargs: object,
    ) -> dict[str, object]:
        self.reconcile_calls.append(dict(kwargs))
        worktree_path = Path(str(kwargs["worktree_path"]))
        branch_name = str(kwargs["branch_name"])
        task = kwargs["task"]
        baseline = str(kwargs["baseline_ref"])
        candidate = str(kwargs["candidate_commit"])
        recovery_key = str(kwargs["recovery_key"])
        assert kwargs["changed_submodule_paths"] is None
        assert kwargs["preacquired_task_claim"]
        assert not self.provider_hooks
        result: dict[str, object] = {
            "task_id": task.task_id,
            "task_cid": task.canonical_task_cid,
            "attempt": 1,
            "returncode": 1,
            "attempt_consumed": False,
            "provider_dispatched": False,
            "worktree_path": str(worktree_path),
            "branch": branch_name,
            "baseline_ref": baseline,
            "implementation_commit": candidate,
            "recovery_key": recovery_key,
            "validation_result": {
                "attempted": True,
                "passed": False,
                "returncode": 1,
                "reason": "declared_validation_failed",
            },
            "merge_result": {
                "merged": False,
                "queued": False,
                "reason": "not_attempted",
            },
        }
        if self.mode == "validation_failure":
            return result
        if self.mode in {
            "queued",
            "queued_then_success",
            "queued_busy_then_success",
            "queued_cancelled",
            "queued_crash_after_completion",
            "queued_deadline",
            "queued_foreign_then_success",
            "queued_without_completion_event",
            "queued_terminal",
            "queued_tampered",
        }:
            request_id = "request:protected-recovery"
            result["merge_result"] = {
                "merged": False,
                "queued": True,
                "request_id": request_id,
                "reason": "merge_queued",
            }
            self.merge_queue.requests[request_id] = SimpleNamespace(
                request_id=request_id,
                status="pending",
                branch_name=branch_name,
                task_id=task.task_id,
                canonical_task_id=task.canonical_task_cid,
                commit_sha=candidate,
                metadata={
                    "worktree_path": str(worktree_path),
                    "repo_root": str(self.repo_root),
                    "todo_path": str(self.paths.task_projection),
                    "state_path": str(self.paths.state),
                    "events_path": str(self.paths.events),
                    "target_repository_id": (
                        self.merge_queue.target_repository_id
                    ),
                    "target_branch": self.resolved_merge_target_branch,
                },
            )
            if self.mode == "queued_tampered":
                self.merge_queue.requests[request_id].metadata[
                    "target_branch"
                ] = "foreign-target"
            if self.mode == "queued_terminal":
                self.merge_queue.requests[request_id].status = "quarantined"
            if self.mode == "queued_foreign_then_success":
                self.merge_queue.requests["request:foreign"] = SimpleNamespace(
                    request_id="request:foreign",
                    status="pending",
                    branch_name="implementation/foreign",
                    task_id="FOREIGN-001",
                    canonical_task_id="task:cid:foreign",
                    commit_sha="f" * 40,
                    metadata={
                        "target_repository_id": (
                            self.merge_queue.target_repository_id
                        ),
                        "target_branch": self.resolved_merge_target_branch,
                    },
                )
            append_jsonl_event(
                self.paths.events,
                "worktree_reconciliation_candidate_queued",
                {
                    "task_id": task.task_id,
                    "canonical_task_cid": task.canonical_task_cid,
                    "attempt": 1,
                    "returncode": 1,
                    "attempt_consumed": False,
                    "provider_dispatched": False,
                    "baseline_ref": baseline,
                    "implementation_commit": candidate,
                    "validation_result": {
                        "attempted": True,
                        "passed": True,
                        "returncode": 0,
                    },
                    "merge_result": dict(result["merge_result"]),
                },
            )
            return result
        if self.mode not in {"success", "success_todo"}:
            raise AssertionError(f"unexpected reconciliation mode {self.mode}")
        subprocess.run(
            ["git", "merge", "--ff-only", candidate],
            cwd=self.repo_root,
            check=True,
            capture_output=True,
        )
        projection = self.paths.task_projection.read_text(encoding="utf-8")
        self.paths.task_projection.write_text(
            projection.replace("- Status: ready", "- Status: completed").replace(
                "- Status: todo", "- Status: completed"
            ),
            encoding="utf-8",
        )
        append_jsonl_event(
            self.paths.events,
            "implementation_finished",
            {
                "task_id": task.task_id,
                "canonical_task_cid": task.canonical_task_cid,
                "attempt": 1,
                "returncode": 0,
                "attempt_consumed": False,
                "provider_dispatched": False,
                "baseline_ref": baseline,
                "implementation_commit": candidate,
                "validation_result": {
                    "attempted": True,
                    "passed": True,
                    "returncode": 0,
                },
                "merge_result": {
                    "merged": True,
                    "queued": False,
                    "reason": "merged",
                },
            },
        )
        append_jsonl_event(
            self.paths.events,
            "task_completed",
            {
                "task_id": task.task_id,
                "canonical_task_cid": task.canonical_task_cid,
                "canonical_task_key": task.canonical_task_key,
                "implementation_commit": candidate,
            },
        )
        result.update(
            returncode=0,
            validation_result={
                "attempted": True,
                "passed": True,
                "returncode": 0,
            },
            merge_result={
                "merged": True,
                "queued": False,
                "reason": "merged",
            },
        )
        return result

    def _consume_exact_merge_candidate(
        self,
        request_id: str,
    ) -> dict[str, object] | None:
        if self.mode not in {
            "queued_then_success",
            "queued_busy_then_success",
            "queued_crash_after_completion",
            "queued_foreign_then_success",
            "queued_without_completion_event",
        }:
            return None
        assert request_id == "request:protected-recovery"
        self.consume_calls += 1
        if self.mode == "queued_busy_then_success" and self.consume_calls == 1:
            raise RuntimeError("merge train consumer lease is busy")
        request = self.merge_queue.get(request_id)
        assert request is not None and request.status == "pending"
        assert Path(request.metadata["worktree_path"]).exists()
        subprocess.run(
            ["git", "merge", "--ff-only", request.commit_sha],
            cwd=self.repo_root,
            check=True,
            capture_output=True,
        )
        projection = self.paths.task_projection.read_text(encoding="utf-8")
        self.paths.task_projection.write_text(
            projection.replace("- Status: ready", "- Status: completed").replace(
                "- Status: todo", "- Status: completed"
            ),
            encoding="utf-8",
        )
        request.status = "completed"
        if self.mode == "queued_crash_after_completion":
            raise KeyboardInterrupt(
                "fixture crash after exact queue completion"
            )
        return {
            "status": "merged",
            "request_id": request.request_id,
            "merged": True,
        }

    def _implementation_cancel_requested(self) -> bool:
        return self.mode == "queued_cancelled"

    def close_event_runtime(self) -> None:
        self.closed = True


def _protected_recovery_bridge(
    *,
    record: object,
    target: DatabaseTaskAttempt,
    repo: Path,
    target_root: Path,
    mode: str,
) -> tuple[
    DatabasePortalExecutionBridge,
    dict[str, _ProtectedRecoveryPortal],
    _ProtectedRecoveryQueue,
    list[str],
    list[str],
]:
    target_branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    queue = _ProtectedRecoveryQueue(repo, target_branch)
    provider_hooks: list[str] = []
    factory_calls: list[str] = []
    observed: dict[str, _ProtectedRecoveryPortal] = {}

    def factory(paths: object, alias: str) -> object:
        factory_calls.append(alias)
        portal = _ProtectedRecoveryPortal(
            paths=paths,
            repo=repo,
            queue=queue,
            target_branch=target_branch,
            mode=mode,
            provider_hooks=provider_hooks,
        )
        observed["portal"] = portal
        return portal

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=target_root,
        repository_root=repo,
        portal_factory=factory,
        merge_queue=queue,
        merge_target_branch=target_branch,
        max_passes=1,
        max_task_attempts=4,
    )
    return bridge, observed, queue, provider_hooks, factory_calls


def _write_protected_reconciliation_self_lock_events(
    *,
    bridge: DatabasePortalExecutionBridge,
    target: DatabaseTaskAttempt,
    record: object,
    preservation_seed: dict[str, object],
    tamper: str = "",
) -> tuple[object, str, str]:
    paths, binding = bridge._ensure_attempt_projection(target, record)
    _identity, recovery_key, recovery_branch = (
        bridge._protected_preservation_recovery_identity(
            attempt=target,
            binding=binding,
            seed=preservation_seed,
        )
    )
    task_id = target.task_alias
    task_cid = target.task_cid
    worktree_path = str(paths.root / "historical-reconciliation-worktree")
    log_path = str(paths.root / "historical-validation.log")
    proposal_id = "a" * 64
    policy_id = "b" * 64
    proposal_receipt_id = "c" * 64
    protected_paths = ["docs/architecture/protected.md"]
    lock = {
        "acquired": False,
        "lock_owner_branch": recovery_branch,
        "lock_owner_operation": (
            "provider_dispatch"
            if tamper == "lock_owner_operation"
            else "reconcile_protected_preservation_candidate"
        ),
        "lock_owner_pid": 4242,
        "lock_owner_task_id": task_id,
        "lock_path": str(checkout_mutation_lock_path(bridge.repository_root)),
        "reason": "lock_exists",
        "waited_seconds": 30.0,
    }
    mutations = [
        {
            "path": protected_paths[0],
            "change": "verification_inconclusive",
            "scope": "shared_checkout",
            "before": {"state": "present", "sha256": "1" * 64},
            "after": {
                "error": (
                    "implementation_protected_path_verification_lock_timeout"
                ),
                "state": "error",
            },
        }
    ]
    violation = {
        "task_id": task_id,
        "reason": "implementation_protected_path_verification_lock_timeout",
        "attempt": 1,
        "workspace_path": worktree_path,
        "protected_paths": protected_paths,
        "mutations": (
            [{**mutations[0], "path": "docs/architecture/spliced.md"}]
            if tamper == "spliced_mutations"
            else mutations
        ),
        "verification_deferred": True,
        "shared_checkout_restored": False,
        "lock": lock,
    }
    append_jsonl_event(
        paths.events,
        "implementation_task_claim_lock_cleared",
        {
            "task_id": task_id,
            "branch": "",
            "lock_owner_pid": 4242,
            "lock_path": str(paths.root / "implementation-task-claim.json"),
        },
    )
    append_jsonl_event(
        paths.events,
        "implementation_protected_path_snapshot_recorded",
        {
            "task_id": task_id,
            "attempt": 1,
            "protected_paths": protected_paths,
            "workspace_path": worktree_path,
        },
    )
    if tamper == "interposed_provider_dispatch":
        append_jsonl_event(
            paths.events,
            "implementation_started",
            {
                "task_id": task_id,
                "canonical_task_cid": task_cid,
                "attempt": 1,
                "provider_dispatched": True,
            },
        )
    append_jsonl_event(
        paths.events,
        "worktree_reconciliation_validation_started",
        {
            "task_id": task_id,
            "task_cid": task_cid,
            "attempt": 1,
            "attempt_consumed": False,
            "provider_dispatched": False,
            "baseline_ref": preservation_seed["baseline_commit"],
            "implementation_commit": preservation_seed["preserved_commit"],
            "branch": recovery_branch,
            "recovery_key": recovery_key,
            "worktree_path": worktree_path,
            "log_path": log_path,
        },
    )
    append_jsonl_event(
        paths.events,
        "implementation_expected_outputs_checked",
        {
            "task_id": task_id,
            "expected_paths": ["inventory/result.json"],
            "staged_paths": ["inventory/result.json"],
            "force_staged_paths": [],
            "issues": [],
            "passed": tamper != "expected_outputs_failed",
            "proposal_id": proposal_id,
            "completion_authoritative": False,
            "proof_authoritative": False,
        },
    )
    append_jsonl_event(
        paths.events,
        "implementation_proposal_validated",
        {
            "task_id": task_id,
            "attempted": True,
            "accepted": True,
            "changed_paths": ["inventory/result.json"],
            "reason_codes": [],
            "proposal_id": proposal_id,
            "policy_id": policy_id,
            "receipt_id": proposal_receipt_id,
            "repository_tree_id": preservation_seed["baseline_commit"],
            "completion_authoritative": False,
            "proof_authoritative": False,
        },
    )
    append_jsonl_event(
        paths.events,
        "implementation_protected_path_verification_lock_timeout",
        {
            "task_id": task_id,
            "attempt": 1,
            "workspace_path": worktree_path,
            "reason": "implementation_protected_path_verification_lock_timeout",
            "verification_deferred": True,
            "shared_checkout_restored": False,
            "protected_paths": protected_paths,
            "mutations": mutations,
            "lock": lock,
        },
    )
    validation = {
        "attempted": True,
        "passed": False,
        "returncode": 1,
        "reason": "implementation_protected_path_mutated",
        "results": [
            {
                "command": "python3 -m pytest -q focused.py",
                "returncode": 0,
                "timed_out": False,
            }
        ],
        "stages": [{"stage": "targeted", "passed": True}],
        "validation_dag_receipt": {"passed": True},
        "proposal_gate": {
            "accepted": True,
            "attempted": True,
            "changed_paths": ["inventory/result.json"],
            "completion_authoritative": False,
            "policy_id": policy_id,
            "proof_authoritative": False,
            "proposal_id": proposal_id,
            "reason_codes": [],
            "receipt_id": proposal_receipt_id,
            "repository_tree_id": preservation_seed["baseline_commit"],
        },
        "protected_path_violation": violation,
    }
    append_jsonl_event(
        paths.events,
        "worktree_reconciliation_validation_finished",
        {
            "task_id": task_id,
            "task_cid": task_cid,
            "attempt": 1,
            "returncode": 1,
            "attempt_consumed": False,
            "provider_dispatched": False,
            "baseline_ref": preservation_seed["baseline_commit"],
            "implementation_commit": preservation_seed["preserved_commit"],
            "branch": recovery_branch,
            "recovery_key": recovery_key,
            "worktree_path": worktree_path,
            "log_path": log_path,
            "validation_result": validation,
            "protected_path_violation": violation,
            "commit_result": {"committed": False},
            "merge_result": {"merged": False, "reason": "not_attempted"},
        },
    )
    append_jsonl_event(
        paths.events,
        "cleanup_finished",
        {
            "branch": recovery_branch,
            "worktree_path": worktree_path,
            "cleaned": True,
            "removed_worktree": True,
            "deleted_branch": True,
        },
    )
    return paths, recovery_key, recovery_branch


def test_bridge_recovers_exact_historical_protected_reconciliation_self_lock(
    tmp_path: Path,
) -> None:
    record, target, repo, target_root, *_commits, seed = (
        _prepare_protected_preservation_successor_seed(tmp_path)
    )
    bridge, _observed, _queue, provider_hooks, factory_calls = (
        _protected_recovery_bridge(
            record=record,
            target=target,
            repo=repo,
            target_root=target_root,
            mode="validation_failure",
        )
    )
    paths, recovery_key, recovery_branch = (
        _write_protected_reconciliation_self_lock_events(
            bridge=bridge,
            target=target,
            record=record,
            preservation_seed=seed,
        )
    )

    receipt = bridge.recover_protected_reconciliation_self_lock(target, seed)

    assert [
        event["type"] for event in bridge._verified_event_chain(paths)
    ] == [
        "implementation_task_claim_lock_cleared",
        "implementation_protected_path_snapshot_recorded",
        "worktree_reconciliation_validation_started",
        "implementation_expected_outputs_checked",
        "implementation_proposal_validated",
        "implementation_protected_path_verification_lock_timeout",
        "worktree_reconciliation_validation_finished",
        "cleanup_finished",
    ]
    assert receipt["schema"] == (
        DATABASE_PORTAL_PROTECTED_RECONCILIATION_SELF_LOCK_SCHEMA
    )
    assert receipt["recovery_key"] == recovery_key
    assert receipt["recovery_branch"] == recovery_branch
    assert receipt["provider_dispatched"] is False
    assert receipt["attempt_consumed"] is False
    assert receipt["validation_commands_passed"] is True
    assert receipt["verification_deferred"] is True
    assert receipt["merge_attempted"] is False
    assert receipt["receipt_id"].startswith("sha256:")
    assert factory_calls == []
    assert provider_hooks == []


@pytest.mark.parametrize(
    "tamper",
    [
        "lock_owner_operation",
        "interposed_provider_dispatch",
        "expected_outputs_failed",
        "spliced_mutations",
    ],
)
def test_bridge_protected_reconciliation_self_lock_tamper_fails_closed(
    tmp_path: Path,
    tamper: str,
) -> None:
    record, target, repo, target_root, *_commits, seed = (
        _prepare_protected_preservation_successor_seed(tmp_path)
    )
    bridge, _observed, _queue, provider_hooks, factory_calls = (
        _protected_recovery_bridge(
            record=record,
            target=target,
            repo=repo,
            target_root=target_root,
            mode="validation_failure",
        )
    )
    _write_protected_reconciliation_self_lock_events(
        bridge=bridge,
        target=target,
        record=record,
        preservation_seed=seed,
        tamper=tamper,
    )

    with pytest.raises(DatabasePortalBridgeError):
        bridge.recover_protected_reconciliation_self_lock(target, seed)

    assert factory_calls == []
    assert provider_hooks == []


def test_bridge_zero_provider_reconciles_protected_preservation_seed(
    tmp_path: Path,
) -> None:
    (
        record,
        target,
        repo,
        target_root,
        baseline_commit,
        preserved_commit,
        _rescue_branch,
        _seed,
    ) = _prepare_protected_preservation_successor_seed(tmp_path)
    bridge, observed, _queue, provider_hooks, factory_calls = (
        _protected_recovery_bridge(
            record=record,
            target=target,
            repo=repo,
            target_root=target_root,
            mode="success_todo",
        )
    )

    receipt = bridge.run_provider(target)
    effect = bridge.apply_effect(target, receipt)
    validation = bridge.validate_effect(target, effect)
    portal = observed["portal"]
    state_after = json.loads(
        bridge._paths(target).state.read_text(encoding="utf-8")
    )

    assert receipt["accepted"] is True
    assert receipt["task_cid"] == target.task_cid
    completion_binding = validation["portal_completion_binding"]
    assert completion_binding["baseline_commit"] == baseline_commit
    assert completion_binding["implementation_commit"] == preserved_commit
    assert completion_binding["baseline_tree"] == subprocess.run(
        ["git", "rev-parse", f"{baseline_commit}^{{tree}}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert provider_hooks == []
    assert factory_calls == [target.task_alias]
    assert len(portal.reconcile_calls) == 1
    assert state_after["implementation_attempts"] == {"LGSWF-004": 2}
    assert state_after["implementation_attempts_by_cid"] == {
        "task:cid:004": 2
    }
    assert portal.cleanup_calls
    assert subprocess.run(
        ["git", "merge-base", "--is-ancestor", preserved_commit, "HEAD"],
        cwd=repo,
        check=False,
    ).returncode == 0

    replay = bridge.run_provider(target)
    assert replay["accepted"] is True
    assert replay["receipt_id"] != ""
    assert factory_calls == [target.task_alias, target.task_alias]
    assert len(portal.reconcile_calls) == 1
    assert observed["portal"].reconcile_calls == []
    assert provider_hooks == []


@pytest.mark.parametrize(
    "mode",
    [
        "wrong_task",
        "api_absent",
        "wrong_repo",
        "validation_failure",
        "queued_tampered",
    ],
)
def test_bridge_protected_recovery_failures_never_dispatch_or_consume(
    tmp_path: Path,
    mode: str,
) -> None:
    (
        record,
        target,
        repo,
        target_root,
        _baseline,
        _preserved_commit,
        rescue_branch,
        _seed,
    ) = _prepare_protected_preservation_successor_seed(tmp_path)
    bridge, observed, _queue, provider_hooks, _factory_calls = (
        _protected_recovery_bridge(
            record=record,
            target=target,
            repo=repo,
            target_root=target_root,
            mode=mode,
        )
    )

    with pytest.raises(
        DatabasePortalBridgeError,
    ):
        bridge.run_provider(target)
    portal = observed["portal"]
    assert provider_hooks == []
    assert subprocess.run(
        ["git", "rev-parse", f"refs/heads/{rescue_branch}"],
        cwd=repo,
        check=False,
        capture_output=True,
    ).returncode == 0
    if mode in {"validation_failure", "queued_tampered"}:
        assert len(portal.reconcile_calls) == 1
        assert portal.cleanup_calls
        state_after = json.loads(
            bridge._paths(target).state.read_text(encoding="utf-8")
        )
        assert state_after["implementation_attempts"] == {"LGSWF-004": 2}
    else:
        assert portal.reconcile_calls == []


def test_bridge_rejects_conflicting_deterministic_recovery_branch_on_replay(
    tmp_path: Path,
) -> None:
    (
        record,
        target,
        repo,
        target_root,
        baseline,
        _preserved_commit,
        _rescue_branch,
        _seed,
    ) = _prepare_protected_preservation_successor_seed(tmp_path)
    bridge, observed, _queue, provider_hooks, _factory_calls = (
        _protected_recovery_bridge(
            record=record,
            target=target,
            repo=repo,
            target_root=target_root,
            mode="validation_failure",
        )
    )
    with pytest.raises(DatabasePortalBridgeError):
        bridge.run_provider(target)
    first_portal = observed["portal"]
    [first_call] = first_portal.reconcile_calls
    recovery_branch = str(first_call["branch_name"])
    subprocess.run(
        ["git", "branch", recovery_branch, baseline],
        cwd=repo,
        check=True,
    )

    with pytest.raises(
        DatabasePortalBridgeError,
        match="conflicts with an existing checkout",
    ):
        bridge.run_provider(target)
    second_portal = observed["portal"]
    assert second_portal is not first_portal
    assert second_portal.reconcile_calls == []
    assert provider_hooks == []
    assert subprocess.run(
        ["git", "rev-parse", f"refs/heads/{recovery_branch}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip() == baseline


def test_bridge_bounded_queue_continuation_completes_without_provider(
    tmp_path: Path,
) -> None:
    (
        record,
        target,
        repo,
        target_root,
        _baseline,
        _preserved_commit,
        rescue_branch,
        _seed,
    ) = _prepare_protected_preservation_successor_seed(tmp_path)
    bridge, observed, queue, provider_hooks, factory_calls = (
        _protected_recovery_bridge(
            record=record,
            target=target,
            repo=repo,
            target_root=target_root,
            mode="queued_busy_then_success",
        )
    )

    receipt = bridge.run_provider(target)
    portal = observed["portal"]
    [call] = portal.reconcile_calls
    recovery_worktree = Path(str(call["worktree_path"]))
    recovery_branch = str(call["branch_name"])
    assert provider_hooks == []
    assert receipt["accepted"] is True
    assert portal.consume_calls == 2
    assert portal.cleanup_calls
    assert not recovery_worktree.exists()
    assert queue.get("request:protected-recovery").status == "completed"
    assert queue.get("request:protected-recovery").metadata[
        "worktree_path"
    ] == str(recovery_worktree)
    assert subprocess.run(
        ["git", "rev-parse", f"refs/heads/{recovery_branch}"],
        cwd=repo,
        check=False,
        capture_output=True,
    ).returncode != 0
    assert subprocess.run(
        ["git", "rev-parse", f"refs/heads/{rescue_branch}"],
        cwd=repo,
        check=False,
        capture_output=True,
    ).returncode == 0
    replay = bridge.run_provider(target)
    assert replay["accepted"] is True
    assert factory_calls == [target.task_alias, target.task_alias]
    assert len(portal.reconcile_calls) == 1
    assert observed["portal"].reconcile_calls == []
    assert provider_hooks == []


def _complete_protected_recovery_queue_request(
    *,
    bridge: DatabasePortalExecutionBridge,
    target: DatabaseTaskAttempt,
    queue: _ProtectedRecoveryQueue,
) -> None:
    request = queue.get("request:protected-recovery")
    assert request is not None
    subprocess.run(
        ["git", "merge", "--ff-only", request.commit_sha],
        cwd=bridge.repository_root,
        check=True,
        capture_output=True,
    )
    paths = bridge._paths(target)
    projection = paths.task_projection.read_text(encoding="utf-8")
    paths.task_projection.write_text(
        projection.replace("- Status: ready", "- Status: completed").replace(
            "- Status: todo", "- Status: completed"
        ),
        encoding="utf-8",
    )
    request.status = "completed"


def test_bridge_exact_queue_continuation_never_consumes_foreign_request(
    tmp_path: Path,
) -> None:
    record, target, repo, target_root, *_rest = (
        _prepare_protected_preservation_successor_seed(tmp_path)
    )
    bridge, observed, queue, provider_hooks, _factory_calls = (
        _protected_recovery_bridge(
            record=record,
            target=target,
            repo=repo,
            target_root=target_root,
            mode="queued_foreign_then_success",
        )
    )

    receipt = bridge.run_provider(target)

    assert receipt["accepted"] is True
    assert observed["portal"].consume_calls == 1
    assert queue.get("request:foreign").status == "pending"
    assert provider_hooks == []


def test_bridge_repairs_exact_postmerge_completion_event_without_run_once(
    tmp_path: Path,
) -> None:
    (
        record,
        target,
        repo,
        target_root,
        baseline_commit,
        preserved_commit,
        *_rest,
    ) = _prepare_protected_preservation_successor_seed(tmp_path)
    bridge, observed, _queue, provider_hooks, _factory_calls = (
        _protected_recovery_bridge(
            record=record,
            target=target,
            repo=repo,
            target_root=target_root,
            mode="queued_without_completion_event",
        )
    )

    receipt = bridge.run_provider(target)
    events = bridge._verified_event_chain(bridge._paths(target))
    [completion] = [
        event for event in events if event.get("type") == "task_completed"
    ]

    assert receipt["accepted"] is True
    assert completion["completion_receipt_repair"] is True
    assert completion["reason"] == "protected_recovery_merge_completed"
    assert completion["baseline_commit"] == baseline_commit
    assert completion["implementation_commit"] == preserved_commit
    assert completion["completion_source_event_id"].startswith("sha256:")
    assert observed["portal"].consume_calls == 1
    assert provider_hooks == []


def test_bridge_replays_queue_completion_before_owned_cleanup(
    tmp_path: Path,
) -> None:
    record, target, repo, target_root, *_rest = (
        _prepare_protected_preservation_successor_seed(tmp_path)
    )
    bridge, observed, queue, provider_hooks, factory_calls = (
        _protected_recovery_bridge(
            record=record,
            target=target,
            repo=repo,
            target_root=target_root,
            mode="queued_crash_after_completion",
        )
    )

    with pytest.raises(
        KeyboardInterrupt,
        match="fixture crash after exact queue completion",
    ):
        bridge.run_provider(target)
    first_portal = observed["portal"]
    [first_call] = first_portal.reconcile_calls
    recovery_worktree = Path(str(first_call["worktree_path"]))
    recovery_branch = str(first_call["branch_name"])
    assert recovery_worktree.exists()
    assert queue.get("request:protected-recovery").status == "completed"

    replay = bridge.run_provider(target)

    assert replay["accepted"] is True
    assert factory_calls == [target.task_alias, target.task_alias]
    assert observed["portal"].reconcile_calls == []
    assert observed["portal"].cleanup_calls
    assert not recovery_worktree.exists()
    assert subprocess.run(
        ["git", "rev-parse", f"refs/heads/{recovery_branch}"],
        cwd=repo,
        check=False,
        capture_output=True,
    ).returncode != 0
    assert provider_hooks == []


def test_bridge_adopts_repaired_terminal_queue_completion_without_provider(
    tmp_path: Path,
) -> None:
    record, target, repo, target_root, *_rest = (
        _prepare_protected_preservation_successor_seed(tmp_path)
    )
    bridge, observed, queue, provider_hooks, _factory_calls = (
        _protected_recovery_bridge(
            record=record,
            target=target,
            repo=repo,
            target_root=target_root,
            mode="queued_terminal",
        )
    )

    with pytest.raises(RuntimeError, match="terminalized") as interrupted:
        bridge.run_provider(target)
    assert not isinstance(interrupted.value, DatabasePortalBridgeError)
    first_portal = observed["portal"]
    [first_call] = first_portal.reconcile_calls
    recovery_worktree = Path(str(first_call["worktree_path"]))
    assert recovery_worktree.exists()
    assert queue.get("request:protected-recovery").status == "quarantined"

    _complete_protected_recovery_queue_request(
        bridge=bridge,
        target=target,
        queue=queue,
    )
    replay = bridge.run_provider(target)

    assert replay["accepted"] is True
    assert observed["portal"].reconcile_calls == []
    assert observed["portal"].cleanup_calls
    assert not recovery_worktree.exists()
    assert provider_hooks == []


@pytest.mark.parametrize(
    "forged_queue_state",
    ["pending", "completed-without-target"],
)
def test_bridge_rejects_forged_terminal_projection_without_queue_authority(
    tmp_path: Path,
    forged_queue_state: str,
) -> None:
    record, target, repo, target_root, *_rest = (
        _prepare_protected_preservation_successor_seed(tmp_path)
    )
    bridge, observed, queue, provider_hooks, factory_calls = (
        _protected_recovery_bridge(
            record=record,
            target=target,
            repo=repo,
            target_root=target_root,
            mode="queued_cancelled",
        )
    )

    with pytest.raises(RuntimeError, match="cancelled"):
        bridge.run_provider(target)
    first_portal = observed["portal"]
    [first_call] = first_portal.reconcile_calls
    recovery_worktree = Path(str(first_call["worktree_path"]))
    assert recovery_worktree.exists()
    assert queue.get("request:protected-recovery").status == "pending"
    paths = bridge._paths(target)
    projection = paths.task_projection.read_text(encoding="utf-8")
    paths.task_projection.write_text(
        projection.replace("- Status: ready", "- Status: completed").replace(
            "- Status: todo", "- Status: completed"
        ),
        encoding="utf-8",
    )
    if forged_queue_state == "completed-without-target":
        queue.get("request:protected-recovery").status = "completed"

    with pytest.raises(
        DatabasePortalBridgeError,
        match="lacks exact completed queue and target ancestry proof",
    ):
        bridge.run_provider(target)

    event_types = [
        str(event.get("type") or "")
        for event in bridge._verified_event_chain(paths)
    ]
    assert "task_completed" not in event_types
    assert "merge_reconciled" not in event_types
    assert recovery_worktree.exists()
    assert queue.get("request:protected-recovery").status == (
        "completed"
        if forged_queue_state == "completed-without-target"
        else "pending"
    )
    assert observed["portal"].cleanup_calls == []
    assert observed["portal"].reconcile_calls == []
    state = json.loads(paths.state.read_text(encoding="utf-8"))
    assert state["implementation_attempts"] == {"LGSWF-004": 2}
    assert state["implementation_attempts_by_cid"] == {"task:cid:004": 2}
    assert factory_calls == [target.task_alias, target.task_alias]
    assert provider_hooks == []


@pytest.mark.parametrize("mode", ["queued_cancelled", "queued_deadline"])
def test_bridge_queue_interruption_retains_same_attempt_authority(
    tmp_path: Path,
    mode: str,
) -> None:
    record, target, repo, target_root, *_rest = (
        _prepare_protected_preservation_successor_seed(tmp_path)
    )
    bridge, observed, queue, provider_hooks, _factory_calls = (
        _protected_recovery_bridge(
            record=record,
            target=target,
            repo=repo,
            target_root=target_root,
            mode=mode,
        )
    )

    with pytest.raises(RuntimeError) as interrupted:
        bridge.run_provider(target)

    assert not isinstance(interrupted.value, DatabasePortalBridgeError)
    [call] = observed["portal"].reconcile_calls
    assert Path(str(call["worktree_path"])).exists()
    assert queue.get("request:protected-recovery").status == "pending"
    state = json.loads(bridge._paths(target).state.read_text(encoding="utf-8"))
    assert state["implementation_attempts"] == {"LGSWF-004": 2}
    assert state["implementation_attempts_by_cid"] == {"task:cid:004": 2}
    assert target.status == "running"
    assert target.committed_phase == "claimed"
    assert provider_hooks == []


def test_database_attempt_heartbeat_loss_wins_when_callback_unwinds() -> None:
    renewal_failed = threading.Event()

    class HeartbeatHarness:
        _lease_heartbeat_interval_seconds = 0.001

        def __init__(self) -> None:
            self.renewals = 0

        def _attempt_claim(self, _attempt: object) -> object:
            return object()

        def _renew_attempt_lease(
            self,
            _attempt: object,
            *,
            claim: object,
        ) -> object:
            assert claim is not None
            self.renewals += 1
            if self.renewals > 1:
                renewal_failed.set()
                raise RuntimeError("fixture lease renewal lost")
            return claim

        def _protect_attempt_write(self, _attempt: object) -> None:
            pytest.fail("lost heartbeat accepted the callback result")

    harness = HeartbeatHarness()

    def interrupted_callback() -> dict[str, object]:
        assert renewal_failed.wait(1.0)
        raise RuntimeError(
            "protected preservation exact queue continuation reached its "
            "implementation timeout"
        )

    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="lost lease authority during execution",
    ) as lost:
        DatabaseImplementationDaemon._run_with_attempt_heartbeat(
            harness,  # type: ignore[arg-type]
            _attempt(),
            interrupted_callback,
        )
    assert isinstance(lost.value.__cause__, RuntimeError)
    assert str(lost.value.__cause__) == "fixture lease renewal lost"


@pytest.mark.parametrize("tamper", ["receipt", "ref", "ancestry"])
def test_bridge_rejects_tampered_protected_recovery_seed_before_factory(
    tmp_path: Path,
    tamper: str,
) -> None:
    (
        record,
        target,
        repo,
        target_root,
        baseline,
        preserved_commit,
        rescue_branch,
        seed,
    ) = _prepare_protected_preservation_successor_seed(tmp_path)
    if tamper == "receipt":
        seed["preserved_commit"] = baseline
    elif tamper == "ref":
        subprocess.run(
            ["git", "branch", "-f", rescue_branch, baseline],
            cwd=repo,
            check=True,
        )
    else:
        seed["baseline_commit"] = preserved_commit
        seed["receipt_id"] = _capacity_record_id(seed, "receipt_id")
    record.body["completion_receipt"]["protected_preservation_seed"] = seed
    factory_calls: list[str] = []

    def unexpected_factory(_paths: object, _alias: str) -> object:
        factory_calls.append("called")
        raise AssertionError("tampered seed reached portal factory")

    target_branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    queue = _ProtectedRecoveryQueue(repo, target_branch)
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=target_root,
        repository_root=repo,
        portal_factory=unexpected_factory,
        merge_queue=queue,
        merge_target_branch=target_branch,
        max_passes=1,
        max_task_attempts=4,
    )

    with pytest.raises(DatabasePortalBridgeError):
        bridge.run_provider(target)
    assert factory_calls == []


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


def test_validation_retry_seed_accepts_declared_outputs_in_different_order(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    commit, rescue_branch = _git_candidate_with_rescue_branch(repo)
    second_output = repo / "inventory" / "summary.json"
    second_output.write_text('{"summary":true}\n', encoding="utf-8")
    subprocess.run(["git", "add", "--", "inventory/summary.json"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "second candidate output"], cwd=repo, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(["git", "branch", "-f", rescue_branch, commit], cwd=repo, check=True)

    record = _record()
    record.outputs = (
        {"path": "inventory/summary.json"},
        {"path": "inventory/result.json"},
    )
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: _ValidationFailurePortal(
            paths,
            alias,
            commit=commit,
            rescue_branch=rescue_branch,
            changed_paths=("inventory/result.json", "inventory/summary.json"),
        ),
        repository_root=repo,
        max_passes=1,
        max_task_attempts=3,
    )
    source = _attempt()
    with pytest.raises(DatabasePortalValidationRetry) as caught:
        bridge.run_provider(source)

    successor = DatabaseTaskAttempt(
        attempt_id="attempt:002",
        claim_id="claim:002",
        task_cid=source.task_cid,
        task_alias=source.task_alias,
        attempt_number=2,
        owner_session_id=source.owner_session_id,
        fencing_token=source.fencing_token + 1,
        fence_epoch=source.fence_epoch + 1,
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
            "owner_session_id": successor.owner_session_id,
            "attempt_number": successor.attempt_number,
            "fencing_token": successor.fencing_token,
            "fence_epoch": successor.fence_epoch,
            "lease_id": successor.lease_id,
            "validation_retry_source_attempt_id": source.attempt_id,
            "validation_retry_seed": caught.value.retry_receipt,
        },
    }
    historical_claim_body = dict(record.body)
    record.revision += 1
    successor_bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "successor-attempts",
        portal_factory=lambda _paths, _alias: SimpleNamespace(
            run_once=lambda: {
                "implementation_result": {
                    "returncode": 1,
                    "reason": "stop_after_seed_inspection",
                }
            }
        ),
        repository_root=repo,
        max_passes=1,
        max_task_attempts=3,
    )
    with pytest.raises(DatabasePortalBridgeError, match="stop_after_seed_inspection"):
        successor_bridge.run_provider(successor)
    record.body = {
        **record.body,
        "completion_receipt": {
            "operation": "database_portal_terminal_failure",
            "attempt_id": successor.attempt_id,
        },
    }
    proof = successor_bridge.verify_validation_retry_successor_recovery(
        successor,
        record,
        historical_claim_body,
    )
    assert proof["schema"] == DATABASE_PORTAL_VALIDATION_RETRY_ORDER_REPAIR_SCHEMA
    assert proof["ordered_lists_differ"] is True
    assert proof["exact_output_set_verified"] is True
    assert proof["changed_paths"] == [
        "inventory/result.json",
        "inventory/summary.json",
    ]
    assert proof["scoped_outputs"] == [
        "inventory/summary.json",
        "inventory/result.json",
    ]


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
    repository_root = tmp_path / "repository"
    baseline_commit, baseline_tree, implementation_commit = (
        _git_completion_lineage(repository_root)
    )
    canonical_board = tmp_path / "canonical-board.md"
    canonical_board.write_text(
        "# Canonical\n\n## LGSWF-004 Authority\n\n- Status: ready\n",
        encoding="utf-8",
    )
    original = canonical_board.read_bytes()
    portals: list[_CompletingPortal] = []

    def factory(paths: object, alias: str) -> _CompletingPortal:
        portal = _CompletingPortal(
            paths,
            alias,
            baseline_commit=baseline_commit,
            implementation_commit=implementation_commit,
        )
        portals.append(portal)
        return portal

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=factory,
        repository_root=repository_root,
    )
    provider = bridge.run_provider(_attempt())
    effect = bridge.apply_effect(_attempt(), provider)
    validation = bridge.validate_effect(_attempt(), effect)
    replayed_effect = bridge.apply_effect(_attempt(), provider)
    replayed_validation = bridge.validate_effect(_attempt(), replayed_effect)

    assert provider["schema"] == DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA
    assert provider["accepted"] is True
    assert provider["provider"] == "PortalImplementationDaemon"
    assert provider["completion_authority"] == "DatabaseImplementationDaemon"
    assert provider["evidence_digest"].startswith("sha256:")
    assert "private provider payload" not in json.dumps(provider)
    assert "provider_payload" not in json.dumps(provider)
    assert effect["status"] == "applied"
    assert effect == replayed_effect
    assert validation["outcome"] == "passed"
    assert validation == replayed_validation
    assert validation["evidence_digest"] == provider["evidence_digest"]
    completion_binding = validation["portal_completion_binding"]
    assert completion_binding["schema"] == DATABASE_PORTAL_COMPLETION_BINDING_SCHEMA
    assert completion_binding["task_cid"] == _attempt().task_cid
    assert completion_binding["attempt_id"] == _attempt().attempt_id
    assert completion_binding["binding_id"] == provider["binding_id"]
    assert completion_binding["portal_receipt_id"] == provider["receipt_id"]
    assert completion_binding["evidence_digest"] == provider["evidence_digest"]
    assert completion_binding["baseline_commit"] == baseline_commit
    assert completion_binding["baseline_tree"] == baseline_tree
    assert completion_binding["implementation_commit"] == implementation_commit
    assert completion_binding["completion_event_id"].startswith("sha256:")
    assert completion_binding["receipt_id"] == _capacity_record_id(
        completion_binding,
        "receipt_id",
    )
    assert canonical_board.read_bytes() == original
    assert portals and portals[0].closed is True
    attempt_boards = list((tmp_path / "attempts").glob("*/task-projection.md"))
    assert len(attempt_boards) == 1
    assert "Projection authority: false" in attempt_boards[0].read_text(encoding="utf-8")


def test_bridge_projects_exact_vrif_benchmark_contract_without_expanding_scope(
    tmp_path: Path,
) -> None:
    outputs = (
        {
            "path": (
                "benchmarks/agent_supervisor/residual_intelligence/manifest.json"
            )
        },
        {
            "path": (
                "benchmarks/agent_supervisor/residual_intelligence/cases.jsonl"
            )
        },
        {"path": "test/api/residual_intelligence/test_benchmark.py"},
    )
    attempt = SimpleNamespace(
        task_cid="task:vrif:030",
        task_alias="VRIF-030",
        attempt_id="attempt:vrif:030",
        claim_id="claim:vrif:030",
        attempt_number=1,
        owner_session_id="session:vrif:030",
        fencing_token=1,
        fence_epoch=1,
        lease_id="lease:vrif:030",
    )
    record = SimpleNamespace(
        task_cid=attempt.task_cid,
        task_alias=attempt.task_alias,
        goal_cid="goal:vrif:release",
        plan_cid="plan:vrif:1",
        revision=31,
        priority="P0",
        dependencies=(),
        outputs=outputs,
        validations=(
            {
                "argv": [
                    "python",
                    "-m",
                    "pytest",
                    "-q",
                    "test/api/residual_intelligence/test_benchmark.py",
                ]
            },
        ),
        acceptance=({"criterion": "The frozen benchmark is exact"},),
        body={
            "objective": "Publish the owner-exact frozen benchmark",
            "completion": "auto",
            "track": "benchmark",
        },
    )
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: None,
    )

    projection = bridge._render_projection(attempt, record)

    assert projection.count("Scope expansion policy: exact") == 1
    assert "Root benchmark contract: owner-exact no-training freeze" in projection
    assert "_vrif_frozen_benchmark_contract" in projection
    assert "benchmark.build_frozen_benchmark_contract" in projection
    assert "base_frozen_bindings" in projection
    assert "test/api/residual_intelligence/test_goal_authority.py" in projection
    assert "benchmark_freeze" in projection
    assert "group_id, input_identity, input_disposition" in projection
    assert "exactly 96 cases" in projection
    assert "legacy Cartesian 384-case population" in projection
    assert "finish the candidate test_benchmark.py bytes first" in projection
    assert "materialize_vrif_frozen_benchmark.py" in projection
    assert "--baseline-commit <the resolved 40-hex commit> --write" in projection
    assert "an empty patch is a terminal implementation failure" in projection
    assert "self-consistency through load_frozen_benchmark is insufficient" in projection
    assert "independently reconstruct the owner base_frozen_bindings" in projection
    projected_outputs = next(
        line.removeprefix("- Outputs: ").split(", ")
        for line in projection.splitlines()
        if line.startswith("- Outputs: ")
    )
    assert projected_outputs == [item["path"] for item in outputs]


def test_bridge_projects_exact_vrif_root_report_contract_without_expanding_scope(
    tmp_path: Path,
) -> None:
    outputs = (
        {
            "path": (
                "docs/architecture/residual_intelligence_inventory/"
                "final_release_report.json"
            )
        },
        {
            "path": (
                "docs/architecture/residual_intelligence_inventory/"
                "final_release_report.md"
            )
        },
        {"path": "test/api/residual_intelligence/test_release_report.py"},
    )
    attempt = SimpleNamespace(
        task_cid="task:vrif:032",
        task_alias="VRIF-032",
        attempt_id="attempt:vrif:032",
        claim_id="claim:vrif:032",
        attempt_number=1,
        owner_session_id="session:vrif:032",
        fencing_token=1,
        fence_epoch=1,
        lease_id="lease:vrif:032",
    )
    record = SimpleNamespace(
        task_cid=attempt.task_cid,
        task_alias=attempt.task_alias,
        goal_cid="goal:vrif:root",
        plan_cid="plan:vrif:1",
        revision=15,
        priority="P0",
        dependencies=(),
        outputs=outputs,
        validations=(
            {
                "argv": [
                    "python",
                    "-m",
                    "pytest",
                    "-q",
                    "test/api/residual_intelligence/test_release_report.py",
                ]
            },
        ),
        acceptance=({"criterion": "The root completion gate is satisfied"},),
        body={
            "objective": "Publish the final root-gated release report",
            "completion": "auto",
            "track": "release",
        },
    )
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: None,
    )

    projection = bridge._render_projection(attempt, record)

    assert projection.count("Scope expansion policy: exact") == 1
    assert "Root completion contract: owner-exact VRIF terminal report" in projection
    assert (
        "producer_artifacts, files_symbols, corpus_rights_splits, "
        "architecture_tokenizer_checkpoint, proof_validation, drift, "
        "rollback_blocker_eligibility"
    ) in projection
    assert "_vrif_terminal_report_evidence" in projection
    assert "_vrif_release_report_markdown" in projection
    assert "replace substring-only checks" in projection
    assert "exact UTF-8 byte equality" in projection
    assert "derive end_tree and drift.evaluated_tree" in projection
    assert "modify all three declared outputs" in projection
    assert "test/api/residual_intelligence/test_goal_authority.py" in projection
    projected_outputs = next(
        line.removeprefix("- Outputs: ").split(", ")
        for line in projection.splitlines()
        if line.startswith("- Outputs: ")
    )
    assert projected_outputs == [item["path"] for item in outputs]


def test_bridge_accepts_independently_reconstructed_vrif_benchmark(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repository"
    baseline_commit, baseline_tree, implementation_commit = (
        _git_vrif_benchmark_lineage(repository_root)
    )
    record = _vrif_benchmark_record()
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: None,
        repository_root=repository_root,
    )

    bridge._verify_vrif_benchmark_acceptance(
        record=record,
        baseline_commit=baseline_commit,
        baseline_tree=baseline_tree,
        implementation_commit=implementation_commit,
    )


def test_bridge_rejects_self_consistent_but_not_owner_computed_vrif_benchmark(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repository"
    baseline_commit, baseline_tree, implementation_commit = (
        _git_vrif_benchmark_lineage(
            repository_root,
            self_consistent_wrong_binding=True,
        )
    )
    manifest, cases = load_frozen_benchmark(
        repository_root / _VRIF_BENCHMARK_OUTPUTS[0],
        repository_root / _VRIF_BENCHMARK_OUTPUTS[1],
    )
    assert manifest.benchmark_freeze["case_count"] == len(cases) == 96
    record = _vrif_benchmark_record()
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: None,
        repository_root=repository_root,
    )

    with pytest.raises(
        DatabasePortalBridgeError,
        match="owner-exact benchmark contract",
    ):
        bridge._verify_vrif_benchmark_acceptance(
            record=record,
            baseline_commit=baseline_commit,
            baseline_tree=baseline_tree,
            implementation_commit=implementation_commit,
        )


def test_bridge_accepts_typed_canonical_vrif_terminal_report(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repository"
    baseline_commit, baseline_tree, implementation_commit = (
        _git_vrif_terminal_lineage(repository_root)
    )
    record = _vrif_terminal_record()
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: None,
        repository_root=repository_root,
    )

    bridge._verify_vrif_terminal_acceptance(
        record=record,
        baseline_commit=baseline_commit,
        baseline_tree=baseline_tree,
        implementation_commit=implementation_commit,
    )


@pytest.mark.parametrize(
    ("fixture_options", "message"),
    (
        (
            {"noncanonical_markdown": True},
            "Markdown is not the owner-canonical report rendering",
        ),
        (
            {"wrong_baseline_tree": True},
            "exact Portal baseline tree",
        ),
    ),
)
def test_bridge_rejects_noncanonical_vrif_terminal_report(
    tmp_path: Path,
    fixture_options: dict[str, bool],
    message: str,
) -> None:
    repository_root = tmp_path / "repository"
    baseline_commit, baseline_tree, implementation_commit = (
        _git_vrif_terminal_lineage(repository_root, **fixture_options)
    )
    record = _vrif_terminal_record()
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: None,
        repository_root=repository_root,
    )

    with pytest.raises(DatabasePortalBridgeError, match=message):
        bridge._verify_vrif_terminal_acceptance(
            record=record,
            baseline_commit=baseline_commit,
            baseline_tree=baseline_tree,
            implementation_commit=implementation_commit,
        )


def test_bridge_calls_vrif_semantic_acceptance_at_effect_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = tmp_path / "repository"
    baseline_commit, baseline_tree, implementation_commit = (
        _git_completion_lineage(repository_root)
    )
    attempt = DatabaseTaskAttempt(
        **{
            **asdict(_attempt()),
            "task_alias": "VRIF-030",
        }
    )
    record = _vrif_benchmark_record(task_cid=attempt.task_cid)
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: _CompletingPortal(
            paths,
            alias,
            baseline_commit=baseline_commit,
            implementation_commit=implementation_commit,
        ),
        repository_root=repository_root,
    )
    provider = bridge.run_provider(attempt)
    observed: dict[str, object] = {}

    def verify_semantics(**kwargs: object) -> None:
        observed.update(kwargs)

    monkeypatch.setattr(
        bridge,
        "_verify_vrif_semantic_acceptance",
        verify_semantics,
    )

    effect = bridge.apply_effect(attempt, provider)

    assert effect["status"] == "applied"
    assert observed == {
        "attempt": attempt,
        "baseline_commit": baseline_commit,
        "baseline_tree": baseline_tree,
        "implementation_commit": implementation_commit,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("stale-tree", "stale Portal baseline tree"),
        ("non-ancestor", "unproven Portal commit lineage"),
    ),
)
def test_bridge_cached_effect_validation_rechecks_exact_git_lineage_before_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    repository_root = tmp_path / "repository"
    baseline_commit, _baseline_tree, implementation_commit = (
        _git_completion_lineage(repository_root)
    )
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: _CompletingPortal(
            paths,
            alias,
            baseline_commit=baseline_commit,
            implementation_commit=implementation_commit,
        ),
        repository_root=repository_root,
    )
    provider = bridge.run_provider(_attempt())
    cached_effect = dict(bridge.apply_effect(_attempt(), provider))
    binding = dict(cached_effect["portal_completion_binding"])
    if mutation == "stale-tree":
        cached_effect["baseline_tree"] = "f" * 40
        binding["baseline_tree"] = "f" * 40
    else:
        implementation_tree = subprocess.run(
            ["git", "rev-parse", f"{implementation_commit}^{{tree}}"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        nonancestor = subprocess.run(
            ["git", "commit-tree", implementation_tree],
            cwd=repository_root,
            check=True,
            input="synthetic unrelated candidate\n",
            capture_output=True,
            text=True,
        ).stdout.strip()
        cached_effect["implementation_commit"] = nonancestor
        binding["implementation_commit"] = nonancestor
    binding["receipt_id"] = _capacity_record_id(binding, "receipt_id")
    cached_effect["portal_completion_binding"] = binding
    semantic_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        bridge,
        "_verify_vrif_semantic_acceptance",
        lambda **kwargs: semantic_calls.append(dict(kwargs)),
    )

    with pytest.raises(DatabasePortalBridgeError, match=message):
        bridge.validate_effect(_attempt(), cached_effect)

    assert semantic_calls == []


@pytest.mark.parametrize(
    "tamper",
    [
        "wrong-task",
        "wrong-cid",
        "validation-failed",
        "malformed-baseline",
        "source-after-completion",
        "conflicting-completion-commit",
        "merge-failed",
        "queued-without-reconciliation",
        "queued-reconciliation-before-source",
    ],
)
def test_bridge_completion_lineage_requires_exact_precompletion_source_event(
    tmp_path: Path,
    tamper: str,
) -> None:
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: pytest.fail(
            "invalid terminal lineage reached provider dispatch"
        ),
    )
    record = bridge._record_for_attempt(bridge.task_source, _attempt())
    paths, binding = bridge._ensure_attempt_projection(_attempt(), record)
    projection = paths.task_projection.read_text(encoding="utf-8")
    paths.task_projection.write_text(
        projection.replace("- Status: ready", "- Status: completed"),
        encoding="utf-8",
    )
    source = {
        "task_id": (
            "LGSWF-999" if tamper == "wrong-task" else "LGSWF-004"
        ),
        "canonical_task_cid": (
            "task:cid:forged" if tamper == "wrong-cid" else "task:cid:004"
        ),
        "attempt": 1,
        "returncode": 0,
        "baseline_ref": (
            "not-a-commit" if tamper == "malformed-baseline" else "b" * 40
        ),
        "implementation_commit": "a" * 40,
        "validation_result": {
            "attempted": True,
            "passed": tamper != "validation-failed",
            "returncode": 1 if tamper == "validation-failed" else 0,
        },
        "merge_result": {
            "merged": tamper
            not in {
                "merge-failed",
                "queued-without-reconciliation",
                "queued-reconciliation-before-source",
            },
            "queued": tamper
            in {
                "queued-without-reconciliation",
                "queued-reconciliation-before-source",
            },
            "reason": (
                "merge_failed"
                if tamper == "merge-failed"
                else "merge_queued"
                if tamper
                in {
                    "queued-without-reconciliation",
                    "queued-reconciliation-before-source",
                }
                else "merged"
            ),
        },
    }
    completion = {
        "task_id": "LGSWF-004",
        "canonical_task_cid": "task:cid:004",
    }
    if tamper == "conflicting-completion-commit":
        completion["implementation_commit"] = "c" * 40
    if tamper == "queued-reconciliation-before-source":
        append_jsonl_event(
            paths.events,
            "merge_reconciled",
            {
                "task_id": "LGSWF-004",
                "canonical_task_cid": "task:cid:004",
                "implementation_commit": "a" * 40,
                "resolved": True,
                "merge_result": {
                    "merged": True,
                    "queued": False,
                    "reason": "merged",
                },
            },
        )
    if tamper != "source-after-completion":
        append_jsonl_event(paths.events, "implementation_finished", source)
    append_jsonl_event(paths.events, "task_completed", completion)
    if tamper == "source-after-completion":
        append_jsonl_event(paths.events, "implementation_finished", source)

    assert bridge._verify_projection(paths, binding)
    with pytest.raises(DatabasePortalBridgeError):
        bridge.run_provider(_attempt())


def _append_validated_no_change_completion_chain(
    paths: object,
    *,
    tamper: str = "",
) -> dict[str, object]:
    alias = "LGSWF-004"
    task_cid = "task:cid:004"
    task_key = "task/v1/current-authority-inventory"
    board_namespace = "task-projection.md"
    baseline = "b" * 40
    tree = "c" * 40
    branch = "implementation/lgswf-004-no-change"
    output = "inventory/result.json"
    validation_command = "/usr/bin/true"
    empty_fingerprint = "sha256:" + hashlib.sha256(b"[]").hexdigest()
    clean_fingerprint = "sha256:" + hashlib.sha256(b"").hexdigest()
    authority: dict[str, object] = {
        "board_namespace": board_namespace,
        "canonical_task_key": task_key,
        "declared_outputs": (output,),
        "projection_immutable_digest": "sha256:" + "1" * 64,
        "repository_scope": "",
        "repository_tree_id": tree,
        "task_contract_digest": "sha256:" + "2" * 64,
        "validation_commands": (validation_command,),
    }
    expected_findings = sorted(
        [
            [
                "empty_patch",
                "patch",
                "candidate diff contains no file changes",
                "",
            ],
            [
                "missing_required_field",
                "structure",
                "structured proposal requires operations",
                "",
            ],
            [
                "missing_required_field",
                "structure",
                "structured proposal requires patch_text",
                "",
            ],
        ]
    )
    policy_gate = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor/"
            "no-change-candidate-policy-gate@1"
        ),
        "attempted": True,
        "accepted": True,
        "reason": "empty_candidate_policy_admitted",
        "completion_mode": "allowed",
        "task_id": alias,
        "canonical_task_cid": task_cid,
        "proposal_id": "3" * 64,
        "policy_id": "4" * 64,
        "proposal_receipt_id": "5" * 64,
        "repository_tree_id": baseline,
        "repository_id": "repository:sha256:" + "6" * 64,
        "baseline_id": baseline,
        "context_id": task_cid,
        "accepted_plan_id": task_cid,
        "objective_id": task_cid,
        "replay_nonce": "7" * 64,
        "diff_digest": hashlib.sha256(b"[]").hexdigest(),
        "candidate_fingerprint": empty_fingerprint,
        "validation_plan_id": content_identity(
            {"command": [validation_command]}
        ),
        "expected_output_preflight_id": content_identity(
            {"outputs": [output]}
        ),
        "proposal_collection_error": "",
        "changed_paths": [],
        "proposal_accepted": False,
        "expected_findings": expected_findings,
        "actual_findings": expected_findings,
        "proof_authoritative": False,
        "completion_authoritative": False,
    }
    policy_gate["gate_id"] = content_identity(policy_gate)

    workspace = {
        "branch": branch,
        "errors": [],
        "head": baseline,
        "status_bytes": 0,
        "status_clean": True,
        "status_fingerprint": clean_fingerprint,
        "tree": tree,
        "verified": True,
    }

    def candidate_handoff(phase: str) -> dict[str, object]:
        return {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "validated-candidate-handoff-guard@1"
            ),
            "allowed": True,
            "phase": phase,
            "reasons": [],
            "task_id": alias,
            "attempt": 1,
            "baseline_ref": baseline,
            "baseline_commit": baseline,
            "expected_branch": branch,
            "implementation_commit": "",
            "expected_fingerprint": empty_fingerprint,
            "validated_fingerprint": empty_fingerprint,
            "current_fingerprint": empty_fingerprint,
            "candidate_entry_count": 0,
            "submodule_expansion_count": 0,
            "collection_error": "",
            "validated_workspace": dict(workspace),
            "workspace_before": dict(workspace),
            "workspace_after": dict(workspace),
            "final_tree": tree,
            "final_status_fingerprint": clean_fingerprint,
        }

    pre_commit_handoff = candidate_handoff("pre_commit")
    post_commit_handoff = candidate_handoff("post_commit")
    result = {
        "cache_hit": False,
        "command": validation_command,
        "ordinal": 0,
        "raw_command": validation_command,
        "returncode": 0,
        "timed_out": False,
        "validation_result_digest": "8" * 64,
    }
    command_binding = {
        "command_cid": content_identity(
            {
                "command": validation_command,
                "raw_command": validation_command,
                "ordinal": 0,
            }
        ),
        "validation_id": "declared:" + "9" * 64,
    }
    validation_plan_binding = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "deterministic-declared-validation-plan@1"
        ),
        "task_id": alias,
        "canonical_task_cid": task_cid,
        "repository_id": "repository:sha256:" + "6" * 64,
        "repository_tree_id": baseline,
        "graph_id": "a" * 64,
        "graph_version": "declared-validation-plan-v1",
        "command_count": 1,
        "commands": [command_binding],
        "changed_paths": [output],
    }
    validation_plan_binding["validation_plan_cid"] = content_identity(
        validation_plan_binding
    )
    validation = {
        "attempted": True,
        "passed": True,
        "returncode": 0,
        "target_commit": baseline,
        "cache_hits": 0,
        "cache_misses": 1,
        "results": [result],
        "selection": {
            "scope": "pre_merge",
            "changed_files": [],
            "selected_count": 1,
            "decisions": [
                {
                    "command": validation_command,
                    "selected": True,
                    "source": "declared",
                }
            ],
        },
        "validation_plan_binding": validation_plan_binding,
        "no_change_policy_gate": policy_gate,
        "proposal_gate": {
            "attempted": True,
            "accepted": False,
            "changed_paths": [],
            "completion_authoritative": False,
            "proof_authoritative": False,
            "reason": "empty_patch_reserved_for_no_change_gate",
            "reason_codes": ["empty_patch", "missing_required_field"],
            "proposal_id": policy_gate["proposal_id"],
            "policy_id": policy_gate["policy_id"],
            "receipt_id": policy_gate["proposal_receipt_id"],
            "repository_tree_id": baseline,
        },
        "candidate_binding": {
            "verified": True,
            "expected_fingerprint": empty_fingerprint,
            "current_fingerprint": empty_fingerprint,
            "reason": "validated_no_change_candidate",
            "validated_workspace": workspace,
        },
        "candidate_handoff": {
            "pre_commit": pre_commit_handoff,
            "post_commit": post_commit_handoff,
        },
    }
    cleanup_result = {
        "branch": branch,
        "cleaned": True,
        "deleted_branch": True,
        "finished_at": "2026-08-26T21:33:49+00:00",
        "lifecycle_finalize": {
            "fence": 5,
            "finalized": True,
            "reason": "pool_release_cleaned",
            "state": "terminal",
        },
        "pool_release": {"released": True},
        "pooled": True,
        "removed_worktree": False,
        "started_at": "2026-08-26T21:33:48+00:00",
        "submodule_cleanup": [],
        "worktree_path": "/tmp/disposable-no-change-worktree",
    }
    projection_path = "private/attempt/task-projection.md"
    projection_repo = "/tmp/disposable-no-change-repository"
    todo_result = {
        "already_completed_task_ids": [],
        "commit_result": {
            "commit": "d" * 40,
            "committed": True,
            "path": projection_path,
            "repo": projection_repo,
            "status": f"?? {projection_path}",
        },
        "completion_reason": "single_task",
        "completion_receipts": [
            {
                "board_namespace": board_namespace,
                "canonical_task_cid": task_cid,
                "canonical_task_key": task_key,
                "schema": (
                    "ipfs_accelerate_py.agent_supervisor."
                    "member_completion_receipt@1"
                ),
                "status": "succeeded",
                "task_id": alias,
            }
        ],
        "inserted_status_task_ids": [],
        "missing_status_task_ids": [],
        "missing_task_ids": [],
        "path": f"{projection_repo}/{projection_path}",
        "task_id": alias,
        "updated": True,
        "updated_checkbox_task_ids": [],
        "updated_task_ids": [alias],
    }
    no_change_guard = {
        "allowed": True,
        "reasons": [],
        "baseline_ref": baseline,
        "current_head": baseline,
        "expected_branch": branch,
        "current_branch": branch,
        "validated_changed_files": [],
        "no_change_policy_gate_id": policy_gate["gate_id"],
        "proposal_receipt_id": policy_gate["proposal_receipt_id"],
    }
    source = {
        "task_id": alias,
        "task_cid": task_cid,
        "canonical_task_cid": task_cid,
        "canonical_task_key": task_key,
        "board_namespace": board_namespace,
        "attempt": 1,
        "returncode": 0,
        "attempt_consumed": True,
        "provider_dispatched": False,
        "branch": branch,
        "baseline_ref": baseline,
        "implementation_commit": "",
        "validation_result": validation,
        "commit_result": {
            "committed": False,
            "reason": "no_changes",
            "no_change_guard": no_change_guard,
            "candidate_handoff_guard": post_commit_handoff,
        },
        "merge_result": {"merged": False, "reason": "not_attempted"},
        "cleanup_result": cleanup_result,
        "todo_update_result": todo_result,
        "board_completion": {
            "complete": True,
            "pending_merge": False,
            "reason": "validated_no_change_completion",
        },
    }
    if tamper == "implementation-commit":
        source["implementation_commit"] = "a" * 40
    elif tamper == "guard-head":
        no_change_guard["current_head"] = "c" * 40
    elif tamper == "guard-extra-field":
        no_change_guard["unsealed"] = True
    elif tamper == "policy-identity":
        policy_gate["gate_id"] = "sha256:" + "d" * 64
        no_change_guard["no_change_policy_gate_id"] = policy_gate["gate_id"]
    elif tamper == "policy-task":
        policy_gate["task_id"] = "LGSWF-999"
        policy_gate["gate_id"] = content_identity(
            {key: value for key, value in policy_gate.items() if key != "gate_id"}
        )
        no_change_guard["no_change_policy_gate_id"] = policy_gate["gate_id"]
    elif tamper == "proposal-receipt":
        validation["proposal_gate"]["receipt_id"] = "receipt:foreign"
    elif tamper == "candidate-fingerprint":
        validation["candidate_binding"]["current_fingerprint"] = (
            "sha256:" + "e" * 64
        )
    elif tamper == "candidate-handoff":
        validation["candidate_handoff"]["post_commit"]["allowed"] = False
    elif tamper == "merge":
        source["merge_result"] = {"merged": True, "reason": "merged"}
    elif tamper == "cleanup":
        cleanup_result["cleaned"] = False
    elif tamper == "board":
        source["board_completion"]["reason"] = "merged_into_target"
    elif tamper == "provider-type":
        source["provider_dispatched"] = "false"
    elif tamper == "unproven-bypass":
        validation["pre_dispatch_no_change"] = None
    elif tamper == "validation-cache":
        result["cache_hit"] = True
    elif tamper == "validation-command":
        result["command"] = "/usr/bin/false"
    elif tamper == "workspace-tree":
        workspace["tree"] = "e" * 40
    elif tamper == "authority-tree":
        authority["repository_tree_id"] = "e" * 40
    elif tamper == "projection-commit":
        todo_result["commit_result"]["commit"] = "not-a-commit"

    identity = {
        "board_namespace": board_namespace,
        "canonical_task_cid": task_cid,
        "canonical_task_key": task_key,
        "task_id": alias,
    }
    output_payload = {
        **identity,
        "completion_authoritative": False,
        "expected_paths": [output],
        "force_staged_paths": [],
        "issues": [],
        "passed": True,
        "proof_authoritative": False,
        "proposal_id": policy_gate["proposal_id"],
        "staged_paths": [],
    }
    preflight_output_payload = {
        **output_payload,
        "proposal_id": "",
        "reason": "validated_no_change_candidate",
    }
    proposal_event = {
        **identity,
        **{
            key: value
            for key, value in validation["proposal_gate"].items()
            if key != "reason"
        },
    }
    policy_event = {**identity, **policy_gate}
    candidate_event = {**identity, **validation["candidate_binding"]}
    protected_clear_event = {
        **identity,
        "attempt": 1,
        "reason": "post_validation_check_unchanged",
    }
    pre_handoff_event = {**identity, **pre_commit_handoff}
    post_handoff_event = {**identity, **post_commit_handoff}
    event_specs = [
        ("implementation_expected_outputs_checked", preflight_output_payload),
        ("implementation_expected_outputs_checked", output_payload),
        ("implementation_proposal_rejected", proposal_event),
        ("implementation_no_change_policy_validated", policy_event),
        ("implementation_candidate_binding_verified", candidate_event),
        (
            "implementation_protected_path_snapshot_cleared",
            protected_clear_event,
        ),
        ("implementation_candidate_handoff_verified", pre_handoff_event),
        ("implementation_candidate_handoff_verified", post_handoff_event),
        ("cleanup_finished", cleanup_result),
        ("todo_status_updated", {**todo_result, **identity}),
    ]
    if tamper == "missing-policy-event":
        event_specs.pop(3)
    elif tamper == "duplicate-policy-event":
        event_specs.insert(4, event_specs[3])
    elif tamper == "out-of-order":
        event_specs[6], event_specs[7] = event_specs[7], event_specs[6]
    elif tamper == "missing-preflight-output":
        event_specs.pop(0)
    for event_type, payload in event_specs:
        append_jsonl_event(paths.events, event_type, payload)
    append_jsonl_event(paths.events, "implementation_finished", source)
    append_jsonl_event(
        paths.events,
        "task_completed",
        identity,
    )
    return authority


def test_bridge_accepts_exact_validated_no_change_commit_lineage(
    tmp_path: Path,
) -> None:
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: pytest.fail(
            "validated no-change lineage reached provider dispatch"
        ),
    )
    record = bridge._record_for_attempt(bridge.task_source, _attempt())
    paths, _binding = bridge._ensure_attempt_projection(_attempt(), record)
    authority = _append_validated_no_change_completion_chain(paths)

    evidence = bridge._completion_event_evidence(
        paths,
        alias="LGSWF-004",
        task_cid="task:cid:004",
        validated_no_change_authority=authority,
    )

    assert evidence is not None
    assert evidence["baseline_commit"] == "b" * 40
    assert evidence["implementation_commit"] == "b" * 40
    assert evidence["completion_source_event_type"] == "implementation_finished"
    assert evidence["completion_source_portal_attempt"] == 1


def test_bridge_rejects_validated_no_change_without_current_task_authority(
    tmp_path: Path,
) -> None:
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: pytest.fail(
            "unauthorized no-change lineage reached provider dispatch"
        ),
    )
    record = bridge._record_for_attempt(bridge.task_source, _attempt())
    paths, _binding = bridge._ensure_attempt_projection(_attempt(), record)
    _append_validated_no_change_completion_chain(paths)

    with pytest.raises(
        DatabasePortalBridgeError,
        match="validated no-change completion lacks task authority",
    ):
        bridge._completion_event_evidence(
            paths,
            alias="LGSWF-004",
            task_cid="task:cid:004",
        )


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"no_change_completion": "allowed"}, True),
        ({"No-change completion": "allowed"}, True),
        ({"no change completion": " allowed "}, True),
        ({"no_change_completion": "Allowed"}, False),
        ({"no_change_completion": True}, False),
        (
            {
                "no_change_completion": "allowed",
                "no change completion": "allowed",
            },
            False,
        ),
        ({}, False),
    ],
)
def test_bridge_reads_one_exact_current_no_change_authority(
    body: dict[str, object],
    expected: bool,
) -> None:
    record = SimpleNamespace(body=body)

    assert (
        DatabasePortalExecutionBridge._record_allows_validated_no_change(record)
        is expected
    )


@pytest.mark.parametrize(
    "tamper",
    [
        "implementation-commit",
        "guard-head",
        "guard-extra-field",
        "policy-identity",
        "policy-task",
        "proposal-receipt",
        "candidate-fingerprint",
        "candidate-handoff",
        "merge",
        "cleanup",
        "board",
        "provider-type",
        "unproven-bypass",
        "validation-cache",
        "validation-command",
        "workspace-tree",
        "authority-tree",
        "projection-commit",
        "missing-policy-event",
        "duplicate-policy-event",
        "out-of-order",
        "missing-preflight-output",
    ],
)
def test_bridge_rejects_tampered_validated_no_change_commit_lineage(
    tmp_path: Path,
    tamper: str,
) -> None:
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: pytest.fail(
            "tampered no-change lineage reached provider dispatch"
        ),
    )
    record = bridge._record_for_attempt(bridge.task_source, _attempt())
    paths, _binding = bridge._ensure_attempt_projection(_attempt(), record)
    authority = _append_validated_no_change_completion_chain(
        paths,
        tamper=tamper,
    )

    with pytest.raises(
        DatabasePortalBridgeError,
        match="validated no-change completion",
    ):
        bridge._completion_event_evidence(
            paths,
            alias="LGSWF-004",
            task_cid="task:cid:004",
            validated_no_change_authority=authority,
        )


def test_bridge_requires_exact_validated_no_change_projection_target(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    output = repo / "inventory" / "result.json"
    output.parent.mkdir()
    output.write_text('{"passed":true}\n', encoding="utf-8")
    subprocess.run(["git", "add", "inventory/result.json"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=No Change Test",
            "-c",
            "user.email=no-change@example.invalid",
            "commit",
            "-qm",
            "baseline",
        ],
        cwd=repo,
        check=True,
    )
    baseline = subprocess.run(
        ["git", "rev-parse", "HEAD^{commit}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    projection = repo / "private" / "attempt" / "task-projection.md"
    projection.parent.mkdir(parents=True)
    projection_text = "## LGSWF-004\n- Status: completed\n"
    projection.write_text(projection_text, encoding="utf-8")
    projection_path = projection.relative_to(repo).as_posix()
    subprocess.run(["git", "add", projection_path], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=No Change Test",
            "-c",
            "user.email=no-change@example.invalid",
            "commit",
            "-qm",
            "private projection",
        ],
        cwd=repo,
        check=True,
    )
    projection_commit = subprocess.run(
        ["git", "rev-parse", "HEAD^{commit}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=repo / "private",
        portal_factory=lambda _paths, _alias: None,
        repository_root=repo,
        merge_target_branch="main",
    )
    authority = {
        "declared_outputs": ("inventory/result.json",),
        "repository_tree_id": tree,
    }
    completion = {
        "baseline_commit": baseline,
        "implementation_commit": baseline,
        "_source_validated_no_change": True,
        "_source_effect_tree": tree,
        "_source_projection_commit": projection_commit,
        "_source_projection_path": projection_path,
        "_source_projection_repo": str(repo),
        "_source_projection_absolute_path": str(projection),
    }

    observed_tree = bridge._require_validated_no_change_target(
        paths=SimpleNamespace(task_projection=projection),
        binding={"repository_tree_id": tree},
        authority=authority,
        completion=completion,
        projection_text=projection_text,
    )

    assert observed_tree == tree

    unrelated = repo / "unrelated.txt"
    unrelated.write_text("target moved\n", encoding="utf-8")
    subprocess.run(["git", "add", "unrelated.txt"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=No Change Test",
            "-c",
            "user.email=no-change@example.invalid",
            "commit",
            "-qm",
            "unrelated target movement",
        ],
        cwd=repo,
        check=True,
    )
    with pytest.raises(
        DatabasePortalBridgeError,
        match="exact projection commit",
    ):
        bridge._require_validated_no_change_target(
            paths=SimpleNamespace(task_projection=projection),
            binding={"repository_tree_id": tree},
            authority=authority,
            completion=completion,
            projection_text=projection_text,
        )


def _append_exact_callback_completion_chain(
    paths: object,
    *,
    tamper: str = "",
    projected_source: bool = False,
    terminal_tamper: str = "",
) -> None:
    alias = "LGSWF-004"
    task_cid = "task:cid:004"
    task_key = "task/v1/exact-callback"
    baseline = "b" * 40
    implementation = "a" * 40
    integration = "c" * 40
    request_id = "request:exact-callback"
    completion_task_cids = {alias: task_cid}
    task_source_identity = {
        "schema": "test.task-source-identity@1",
        "source_id": "source:exact-callback",
    }
    source_payload = {
            "task_id": alias,
            "canonical_task_cid": task_cid,
            "canonical_task_key": task_key,
            "board_namespace": "task-projection.md",
            "task_source_identity": task_source_identity,
            "attempt": 1,
            "returncode": 0,
            "attempt_consumed": True,
            "provider_dispatched": False,
            "branch": "candidate/exact-callback",
            "baseline_ref": baseline,
            "implementation_commit": implementation,
            "validation_result": {
                "attempted": True,
                "passed": True,
                "returncode": 0,
            },
            "merge_result": {
                "attempted": False,
                "merged": False,
                "queued": True,
                "reason": "merge_queued",
                "request_id": request_id,
                "branch": "candidate/exact-callback",
                "implementation_commit": implementation,
                "canonical_task_key": task_key,
                "canonical_task_cid": task_cid,
                "completion_task_cids": completion_task_cids,
                "target_repository_id": "repository:exact-callback",
                "target_branch": "main",
            },
            "board_completion": {
                "complete": False,
                "pending_merge": True,
                "reason": "merge_queued_awaiting_integration",
            },
        }
    if tamper == "source-returncode":
        source_payload["returncode"] = 7
    elif tamper == "source-attempt-consumed":
        source_payload["attempt_consumed"] = False
    elif tamper == "source-provider-type":
        source_payload["provider_dispatched"] = "false"
    elif tamper == "source-validation-shape":
        source_payload["validation_result"] = {
            "attempted": False,
            "passed": True,
            "returncode": 7,
        }
    elif tamper == "source-board-shape":
        source_payload.pop("board_completion")
    elif tamper == "source-extra-field":
        source_payload["unexpected_source_field"] = True
    source_event_type = "implementation_finished"
    if projected_source:
        validation_tree = "d" * 40
        append_jsonl_event(
            paths.events,
            "merge_candidate_enqueued",
            {
                "task_id": alias,
                "canonical_task_cid": task_cid,
                "canonical_task_key": task_key,
                "board_namespace": "task-projection.md",
                "task_source_identity": task_source_identity,
                "attempt": 1,
                "request_id": request_id,
                "branch": "candidate/exact-callback",
                "baseline_ref": baseline,
                "implementation_commit": implementation,
                "attempted": False,
                "merged": False,
                "queued": True,
                "reason": "merge_queued",
                "completion_task_cids": completion_task_cids,
                "target_repository_id": "repository:exact-callback",
                "target_branch": "main",
            },
        )
        enqueue = DatabasePortalExecutionBridge._verified_event_chain(paths)[
            -1
        ]
        provenance = {
            "schema": (
                "ipfs_accelerate_py.agent_supervisor."
                "merge-queue-synchronous-source@1"
            ),
            "request_id": request_id,
            "task_id": alias,
            "task_cid": task_cid,
            "canonical_task_key": task_key,
            "merge_candidate_enqueued_event_id": enqueue["event_id"],
            "portal_attempt": 1,
            "branch": "candidate/exact-callback",
            "baseline_ref": baseline,
            "implementation_commit": implementation,
            "validation_target_commit": implementation,
            "validation_target_tree": validation_tree,
            "validation_repository_tree_id": f"git-tree:{validation_tree}",
        }
        provenance["source_projection_id"] = content_identity(provenance)
        source_payload["attempt_consumed"] = False
        source_payload["provider_dispatched"] = False
        source_payload["reason"] = (
            "merge_queue_synchronous_source_projected"
        )
        source_payload["merge_queue_synchronous_source"] = provenance
        source_event_type = "worktree_reconciliation_candidate_queued"
    append_jsonl_event(
        paths.events,
        source_event_type,
        source_payload,
    )
    source = DatabasePortalExecutionBridge._verified_event_chain(paths)[-1]
    source_event_id = str(source["event_id"])
    member = {
        "board_namespace": "task-projection.md",
        "canonical_task_cid": task_cid,
        "canonical_task_key": task_key,
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "member_completion_receipt@1"
        ),
        "status": "succeeded",
        "task_id": alias,
    }
    receipt_evidence = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "merge-queue-callback-completion-receipt@1"
        ),
        "request_id": request_id,
        "completion_source_event_id": source_event_id,
        "integration_commit": integration,
        "completion_task_cids": completion_task_cids,
        "completion_receipts": [member],
    }
    receipt_evidence["receipt_id"] = content_identity(receipt_evidence)
    candidate_key = content_identity(
        {
            "schema": (
                "ipfs_accelerate_py.agent_supervisor."
                "merge-queue-reconciled-candidate@1"
            ),
            "task_id": alias,
            "task_cid": task_cid,
            "request_id": request_id,
            "baseline_ref": baseline,
            "implementation_commit": implementation,
            "completion_source_event_id": source_event_id,
        }
    )
    reconciliation = {
        "task_id": alias,
        "canonical_task_cid": task_cid,
        "canonical_task_key": task_key,
        "board_namespace": "task-projection.md",
        "task_source_identity": task_source_identity,
        "attempt": 1,
        "branch": "candidate/exact-callback",
        "request_id": request_id,
        "completion_source_event_id": source_event_id,
        "baseline_ref": baseline,
        "implementation_commit": implementation,
        "landed_commit": implementation,
        "merge_commit": integration,
        "target_commit": integration,
        "completion_task_cids": completion_task_cids,
        "reconciled_candidate_key": candidate_key,
        "resolved": True,
        "reason": "merge_queue_callback_completed",
        "merge_result": {
            "attempted": True,
            "merged": True,
            "queued": False,
            "reason": "merge_queue_callback_completed",
            "request_id": request_id,
            "merge_commit": integration,
            "target_commit": integration,
        },
        "integration_commit_proof": {
            "implementation_commit": implementation,
            "integration_commit": integration,
            "integration_ref": integration,
            "passed": True,
            "reasons": [],
            "target_branch": "main",
        },
        "post_merge_declared_output_invariant": {
            "checks": [],
            "missing_outputs": [],
            "mode": "repository_tree",
            "passed": True,
            "reason": "declared_outputs_tracked",
            "repository_ref": integration,
            "task_ids": [alias],
            "unsafe_outputs": [],
            "untracked_outputs": [],
        },
        "completion_receipt_evidence": receipt_evidence,
    }
    if tamper == "source-event":
        reconciliation["completion_source_event_id"] = "sha256:" + "d" * 64
    elif tamper == "request":
        reconciliation["request_id"] = "request:foreign"
    elif tamper == "baseline":
        reconciliation["baseline_ref"] = "d" * 40
    elif tamper == "proof":
        reconciliation["integration_commit_proof"]["passed"] = False
    elif tamper == "invariant":
        reconciliation["post_merge_declared_output_invariant"]["passed"] = False
    elif tamper == "receipt":
        reconciliation["completion_receipt_evidence"]["completion_receipts"][0][
            "canonical_task_cid"
        ] = "task:cid:foreign"
    elif tamper == "reconciliation-extra-field":
        reconciliation["unexpected_reconciliation_field"] = True
    elif tamper == "reconciliation-board-namespace":
        reconciliation["board_namespace"] = "foreign-board.md"
    elif tamper == "reconciliation-board-missing":
        reconciliation.pop("board_namespace")
    elif tamper == "reconciliation-task-source":
        reconciliation["task_source_identity"] = {
            "schema": "test.task-source-identity@1",
            "source_id": "source:foreign",
        }
    elif tamper == "reconciliation-task-source-missing":
        reconciliation.pop("task_source_identity")
    append_jsonl_event(paths.events, "merge_reconciled", reconciliation)
    if projected_source:
        terminal_merge = dict(source_payload["merge_result"])
        terminal_merge.update(
            {
                "attempted": True,
                "merged": True,
                "queued": False,
                "reason": "merged",
                "merge_commit": integration,
                "target_commit": integration,
            }
        )
        if terminal_tamper == "merge-commit":
            terminal_merge["merge_commit"] = "e" * 40
        elif terminal_tamper == "target-commit":
            terminal_merge["target_commit"] = "e" * 40
        elif terminal_tamper == "empty-target-commit":
            terminal_merge["target_commit"] = ""
        elif terminal_tamper == "missing-target-commit":
            terminal_merge.pop("target_commit")
        append_jsonl_event(
            paths.events,
            "implementation_finished",
            {
                "task_id": alias,
                "canonical_task_cid": task_cid,
                "canonical_task_key": task_key,
                "board_namespace": "task-projection.md",
                "task_source_identity": task_source_identity,
                "attempt": 1,
                "returncode": 0,
                "attempt_consumed": True,
                "provider_dispatched": False,
                "branch": "candidate/exact-callback",
                "baseline_ref": baseline,
                "implementation_commit": implementation,
                "validation_result": {
                    "attempted": True,
                    "passed": True,
                    "returncode": 0,
                },
                "merge_result": terminal_merge,
                "board_completion": {
                    "complete": True,
                    "pending_merge": False,
                    "reason": "merged_into_target",
                },
            },
        )
    append_jsonl_event(
        paths.events,
        "task_completed",
        {
            "task_id": alias,
            "canonical_task_cid": task_cid,
            "implementation_commit": implementation,
        },
    )


def test_bridge_completion_lineage_accepts_only_later_exact_queue_reconciliation(
    tmp_path: Path,
) -> None:
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: pytest.fail(
            "terminal queue lineage reached provider dispatch"
        ),
    )
    record = bridge._record_for_attempt(bridge.task_source, _attempt())
    paths, binding = bridge._ensure_attempt_projection(_attempt(), record)
    projection = paths.task_projection.read_text(encoding="utf-8")
    paths.task_projection.write_text(
        projection.replace("- Status: ready", "- Status: completed"),
        encoding="utf-8",
    )
    _append_exact_callback_completion_chain(paths)

    assert bridge._verify_projection(paths, binding)
    evidence = bridge._completion_event_evidence(
        paths,
        alias="LGSWF-004",
        task_cid="task:cid:004",
    )

    assert evidence is not None
    assert evidence["implementation_commit"] == "a" * 40
    assert evidence["baseline_commit"] == "b" * 40
    before = paths.events.read_bytes()
    assert bridge._completion_event_evidence(
        paths,
        alias="LGSWF-004",
        task_cid="task:cid:004",
    ) == evidence
    assert paths.events.read_bytes() == before


def test_bridge_accepts_exact_synchronous_source_with_terminal_confirmation(
    tmp_path: Path,
) -> None:
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: pytest.fail(
            "terminal callback lineage reached provider dispatch"
        ),
    )
    record = bridge._record_for_attempt(bridge.task_source, _attempt())
    paths, _binding = bridge._ensure_attempt_projection(_attempt(), record)
    _append_exact_callback_completion_chain(paths, projected_source=True)

    evidence = bridge._completion_event_evidence(
        paths,
        alias="LGSWF-004",
        task_cid="task:cid:004",
    )

    assert evidence is not None
    assert evidence["completion_source_event_type"] == (
        "implementation_finished"
    )
    assert evidence["implementation_commit"] == "a" * 40
    assert evidence["baseline_commit"] == "b" * 40


def test_bridge_accepts_historical_terminal_confirmation_without_target_commit(
    tmp_path: Path,
) -> None:
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: pytest.fail(
            "historical terminal callback lineage reached provider dispatch"
        ),
    )
    record = bridge._record_for_attempt(bridge.task_source, _attempt())
    paths, _binding = bridge._ensure_attempt_projection(_attempt(), record)
    _append_exact_callback_completion_chain(
        paths,
        projected_source=True,
        terminal_tamper="missing-target-commit",
    )

    evidence = bridge._completion_event_evidence(
        paths,
        alias="LGSWF-004",
        task_cid="task:cid:004",
    )

    assert evidence is not None
    assert evidence["completion_source_event_type"] == "implementation_finished"
    assert evidence["implementation_commit"] == "a" * 40
    assert evidence["baseline_commit"] == "b" * 40


@pytest.mark.parametrize(
    "terminal_tamper",
    ["merge-commit", "target-commit", "empty-target-commit"],
)
def test_bridge_rejects_terminal_confirmation_commit_mismatch(
    tmp_path: Path,
    terminal_tamper: str,
) -> None:
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: pytest.fail(
            "mismatched terminal callback lineage reached provider dispatch"
        ),
    )
    record = bridge._record_for_attempt(bridge.task_source, _attempt())
    paths, _binding = bridge._ensure_attempt_projection(_attempt(), record)
    _append_exact_callback_completion_chain(
        paths,
        projected_source=True,
        terminal_tamper=terminal_tamper,
    )

    with pytest.raises(
        DatabasePortalBridgeError,
        match="lacks one exact evaluated baseline",
    ):
        bridge._completion_event_evidence(
            paths,
            alias="LGSWF-004",
            task_cid="task:cid:004",
        )


@pytest.mark.parametrize(
    "tamper",
    [
        "source-event",
        "request",
        "baseline",
        "proof",
        "invariant",
        "receipt",
        "source-returncode",
        "source-attempt-consumed",
        "source-provider-type",
        "source-validation-shape",
        "source-board-shape",
        "source-extra-field",
        "reconciliation-extra-field",
        "reconciliation-board-namespace",
        "reconciliation-board-missing",
        "reconciliation-task-source",
        "reconciliation-task-source-missing",
    ],
)
def test_bridge_callback_reconciliation_rejects_tampered_binding(
    tmp_path: Path,
    tamper: str,
) -> None:
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: pytest.fail(
            "tampered callback lineage reached provider dispatch"
        ),
    )
    record = bridge._record_for_attempt(bridge.task_source, _attempt())
    paths, _binding = bridge._ensure_attempt_projection(_attempt(), record)
    _append_exact_callback_completion_chain(paths, tamper=tamper)

    with pytest.raises(
        DatabasePortalBridgeError,
        match="callback reconciliation binding is invalid",
    ):
        bridge._completion_event_evidence(
            paths,
            alias="LGSWF-004",
            task_cid="task:cid:004",
        )


def test_post_merge_completion_recovery_events_are_exact_and_idempotent(
    tmp_path: Path,
) -> None:
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: pytest.fail(
            "completion recovery dispatched a provider"
        ),
    )
    record = bridge._record_for_attempt(bridge.task_source, _attempt())
    paths, _binding = bridge._ensure_attempt_projection(_attempt(), record)

    first = bridge._ensure_post_merge_completion_recovery_events(
        paths,
        alias="LGSWF-004",
        task_cid="task:cid:004",
        request_id="request:repaired",
        baseline_commit="b" * 40,
        implementation_commit="a" * 40,
        seed_id="sha256:" + "c" * 64,
        recovery_evidence_id="sha256:" + "d" * 64,
    )
    before = paths.events.read_bytes()
    replay = bridge._ensure_post_merge_completion_recovery_events(
        paths,
        alias="LGSWF-004",
        task_cid="task:cid:004",
        request_id="request:repaired",
        baseline_commit="b" * 40,
        implementation_commit="a" * 40,
        seed_id="sha256:" + "c" * 64,
        recovery_evidence_id="sha256:" + "d" * 64,
    )

    assert paths.events.read_bytes() == before
    events = bridge._verified_event_chain(paths)
    assert [event["type"] for event in events] == [
        "worktree_reconciliation_candidate_queued",
        "merge_reconciled",
        "task_completed",
    ]
    assert events[0]["attempt_consumed"] is False
    assert events[0]["provider_dispatched"] is False
    assert events[0]["validation_result"]["passed"] is True
    assert events[0]["merge_result"]["request_id"] == "request:repaired"
    assert events[1]["merge_result"]["merged"] is True
    assert first == replay
    assert replay["implementation_commit"] == "a" * 40
    assert replay["baseline_commit"] == "b" * 40


def test_callback_integration_self_hashed_float_qualification_has_no_authority(
) -> None:
    daemon = object.__new__(DatabaseImplementationDaemon)
    candidate = "a" * 40
    baseline = "b" * 40
    integration = "c" * 40
    current = "d" * 40
    current_tree = "e" * 40
    task_cid = "task:cid:004"
    task_alias = "LGSWF-004"
    request_id = "request:callback-integration"
    train_receipt = {
        "acceptance_pending": False,
        "accepted": True,
        "callback_owned_integration": True,
        "commit_sha": candidate,
        "finished_at": 1787658878.9458497,
        "integrated": True,
        "merge_commit": integration,
        "merge_result": {
            "merged": True,
            "returncode": 0,
            "merge_commit": integration,
            "todo_update_result": {
                "completion_receipts": [
                    {
                        "task_id": task_alias,
                        "canonical_task_cid": task_cid,
                    }
                ]
            },
        },
        "merged": True,
        "request_id": request_id,
        "started_at": 1787658877.9862263,
        "status": "merged",
        "target_commit": integration,
        "task_id": task_alias,
    }
    train_json = json.dumps(
        train_receipt,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    qualification = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "post-merge-callback-integration-requalification@1"
        ),
        "task_ids": [task_alias],
        "task_cid": task_cid,
        "request_id": request_id,
        "candidate_commit": candidate,
        "baseline_commit": baseline,
        "integration_commit": integration,
        "source_event_id": "sha256:" + "1" * 64,
        "source_event_digest": "sha256:" + "2" * 64,
        "source_validation_result_digest": "sha256:" + "3" * 64,
        "queue_validation_proof_digest": "sha256:" + "4" * 64,
        "train_dedupe_key": "5" * 64,
        "train_receipt_id": (
            "sha256:" + hashlib.sha256(train_json.encode("utf-8")).hexdigest()
        ),
        "train_receipt": train_json,
        "current_target_commit": current,
        "current_target_tree": current_tree,
        "entries": [
            {
                "path": "inventory/result.json",
                "mode": "100644",
                "object_type": "blob",
                "object_id": "6" * 40,
            }
        ],
        "validation": [
            {
                "task_id": task_alias,
                "passed": True,
                "returncode": 0,
                "validation_result_digests": ["7" * 64],
                "command_count": 1,
                "log_sha256": "8" * 64,
            }
        ],
    }
    qualification["receipt_id"] = content_identity(qualification)

    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="source authority is invalid|authority is not configured",
    ):
        daemon._verified_post_merge_callback_integration_receipt(
            qualification,
            recovery_evidence={},
        )
    tampered = dict(qualification)
    tampered_receipt = dict(train_receipt)
    tampered_receipt["finished_at"] = 1787658879.0
    tampered["train_receipt"] = json.dumps(
        tampered_receipt,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    tampered["receipt_id"] = content_identity(
        {key: value for key, value in tampered.items() if key != "receipt_id"}
    )
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="callback integration receipt is invalid",
    ):
        daemon._verified_post_merge_callback_integration_receipt(
            tampered,
            recovery_evidence={},
        )


@pytest.mark.parametrize(
    ("parts", "accepted"),
    [
        (("state", "vrif_database_portal_attempts"), True),
        (
            (
                "state",
                "lane-2",
                "vrif_lane_2_database_portal_attempts",
            ),
            True,
        ),
        (
            (
                "state",
                "lane-2",
                "vrif_lane_3_database_portal_attempts",
            ),
            False,
        ),
    ],
)
def test_merge_train_recovery_binds_canonical_portal_attempt_root_shapes(
    tmp_path: Path,
    parts: tuple[str, ...],
    accepted: bool,
) -> None:
    daemon = object.__new__(DatabaseImplementationDaemon)
    daemon.require_real_execution = True
    daemon._lock = threading.RLock()
    daemon._merge_queue = None
    daemon._merge_repo_root = None
    daemon._merge_target_branch = ""
    daemon._merge_portal_attempt_root = None
    attempt_root = tmp_path.joinpath(*parts)

    if not accepted:
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="Portal attempt root is invalid",
        ):
            daemon.bind_merge_train_recovery(
                merge_queue=object(),
                repo_root=tmp_path,
                merge_target_branch="main",
                portal_attempt_root=attempt_root,
            )
        assert daemon._merge_queue is None
        return

    queue = object()
    daemon.bind_merge_train_recovery(
        merge_queue=queue,
        repo_root=tmp_path,
        merge_target_branch="main",
        portal_attempt_root=attempt_root,
    )
    assert daemon._merge_queue is queue
    assert daemon._merge_portal_attempt_root == attempt_root


def _callback_integration_authority_fixture(
    tmp_path: Path,
) -> tuple[
    DatabaseImplementationDaemon,
    dict[str, object],
    dict[str, object],
    Path,
    Path,
]:
    """Build one fully file/Git/binding-backed callback qualification."""

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
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
    output.parent.mkdir()
    output.write_text('{"generation":"baseline"}\n', encoding="utf-8")
    subprocess.run(["git", "add", "--", "inventory/result.json"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=repo, check=True)

    def git_text(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    baseline = git_text("rev-parse", "HEAD")
    output.write_text('{"generation":"candidate"}\n', encoding="utf-8")
    subprocess.run(["git", "add", "--", "inventory/result.json"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "candidate"], cwd=repo, check=True)
    candidate = git_text("rev-parse", "HEAD")
    candidate_tree = git_text("rev-parse", "HEAD^{tree}")
    subprocess.run(
        ["git", "commit", "--allow-empty", "-qm", "integration"],
        cwd=repo,
        check=True,
    )
    integration = git_text("rev-parse", "HEAD")
    subprocess.run(
        ["git", "commit", "--allow-empty", "-qm", "current target"],
        cwd=repo,
        check=True,
    )

    projection_bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=(
            tmp_path
            / "state"
            / "vrif_database_portal_attempts"
        ),
        portal_factory=lambda _paths, _alias: pytest.fail(
            "authority fixture dispatched a provider"
        ),
        repository_root=repo,
    )
    record = projection_bridge._record_for_attempt(
        projection_bridge.task_source,
        _attempt(),
    )
    paths, binding = projection_bridge._ensure_attempt_projection(
        _attempt(),
        record,
    )
    task_alias = str(binding["task_alias"])
    task_cid = str(binding["task_cid"])
    task_key = str(
        verify_database_portal_attempt_projection(
            paths.task_projection,
            expected_task_alias=task_alias,
            expected_task_cid=task_cid,
            allowed_root=paths.root,
        )["canonical_task_key"]
    )
    request_id = "request:callback-authority"
    validation_proof = {
        "attempted": True,
        "passed": True,
        "returncode": 0,
        "target_commit": candidate,
    }
    append_jsonl_event(
        paths.events,
        "implementation_finished",
        {
            "task_id": task_alias,
            "canonical_task_cid": task_cid,
            "canonical_task_key": task_key,
            "attempt": 1,
            "attempt_consumed": True,
            "provider_dispatched": True,
            "returncode": 0,
            "branch": "implementation/callback-authority",
            "baseline_ref": baseline,
            "implementation_commit": candidate,
            "validation_result": {
                "attempted": True,
                "passed": True,
                "returncode": 0,
            },
            "merge_result": {
                "attempted": False,
                "queued": True,
                "merged": False,
                "reason": "merge_queued",
                "request_id": request_id,
                "branch": "implementation/callback-authority",
                "implementation_commit": candidate,
                "canonical_task_key": task_key,
                "canonical_task_cid": task_cid,
                "completion_task_cids": {task_alias: task_cid},
                "target_repository_id": checkout_repository_id(repo),
                "target_branch": "main",
            },
            "board_completion": {
                "complete": False,
                "pending_merge": True,
                "reason": "merge_queued_awaiting_integration",
            },
        },
    )
    append_jsonl_event(
        paths.events,
        "task_completed",
        {
            "task_id": task_alias,
            "canonical_task_cid": task_cid,
            "reason": "task_became_completed",
            "completion_receipt_repair": False,
        },
    )

    target_repository_id = checkout_repository_id(repo)
    queue_dir = tmp_path / "merge-queue"
    completed_dir = queue_dir / "completed"
    receipt_dir = queue_dir / "train" / "receipts"
    completed_dir.mkdir(parents=True)
    receipt_dir.mkdir(parents=True)
    request_path = completed_dir / f"{request_id}.json"
    request = MergeRequest(
        request_id=request_id,
        branch_name="implementation/callback-authority",
        task_id=task_alias,
        priority="P0",
        lane_id="lane-2",
        enqueued_at=1.0,
        attempt=1,
        metadata={
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/merge-candidate@3"
            ),
            "target_binding_schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "merge-target-binding@1"
            ),
            "target_repository_id": target_repository_id,
            "target_branch": "main",
            "repo_root": str(repo),
            "task_header_prefix": "## ",
            "todo_path": str(paths.task_projection),
            "state_path": str(paths.state),
            "strategy_path": str(paths.strategy),
            "events_path": str(paths.events),
            "baseline_ref": baseline,
            "candidate_tree": candidate_tree,
            "implementation_commit": candidate,
            "completion_task_cids": {task_alias: task_cid},
            "validation_proof": validation_proof,
            "task": {
                "task_id": task_alias,
                "canonical_task_cid": task_cid,
                "canonical_task_key": task_key,
                "outputs": ["inventory/result.json"],
                "metadata": {
                    "database task cid": task_cid,
                    "canonical task cid": task_cid,
                    "canonical task key": task_key,
                    "projection authority": "false",
                    "database attempt id": binding["attempt_id"],
                    "database claim id": binding["claim_id"],
                },
            },
        },
        file_path=request_path,
        commit_sha=candidate,
        canonical_task_id=task_cid,
        canonical_task_key=task_key,
        status="completed",
    )
    request_path.write_text(
        json.dumps(request.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    check = {
        "exists": True,
        "path": "inventory/result.json",
        "reason": "declared_output_tracked",
        "repository": ".",
        "repository_ref": integration,
        "task_id": task_alias,
        "tracked": True,
        "tracked_path": "inventory/result.json",
    }
    member = {
        "board_namespace": "task-projection.md",
        "canonical_task_cid": task_cid,
        "canonical_task_key": task_key,
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "member_completion_receipt@1"
        ),
        "status": "succeeded",
        "task_id": task_alias,
    }
    todo = {
        "already_completed_task_ids": [],
        "commit_result": {"committed": False, "reason": "no_changes"},
        "completion_reason": "single_task",
        "completion_receipts": [member],
        "inserted_status_task_ids": [],
        "missing_status_task_ids": [],
        "missing_task_ids": [],
        "path": str(paths.task_projection),
        "task_id": task_alias,
        "updated": True,
        "updated_checkbox_task_ids": [],
        "updated_task_ids": [task_alias],
    }
    train_receipt = {
        "acceptance_pending": False,
        "accepted": True,
        "callback_owned_integration": True,
        "canonical_task_id": task_key,
        "commit_sha": candidate,
        "distributed_publication_admission": {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "distributed-lane-admission@1"
            ),
            "admitted": True,
            "distributed": False,
            "request_id": request_id,
            "status": "local",
        },
        "finished_at": 1787658878.9458497,
        "integrated": True,
        "merge_commit": integration,
        "merge_result": {
            "attempted": True,
            "merged": True,
            "returncode": 0,
            "merge_commit": integration,
            "target_branch": "main",
            "integration_commit_proof": {
                "implementation_commit": candidate,
                "integration_commit": integration,
                "integration_ref": integration,
                "passed": True,
                "reasons": [],
                "target_branch": "main",
            },
            "post_merge_declared_output_invariant": {
                "checks": [check],
                "missing_outputs": [],
                "mode": "repository_tree",
                "passed": True,
                "reason": "declared_outputs_tracked",
                "repository_ref": integration,
                "task_ids": [task_alias],
                "unsafe_outputs": [],
                "untracked_outputs": [],
            },
            "todo_update_result": todo,
        },
        "merged": True,
        "request_id": request_id,
        "started_at": 1787658877.9862263,
        "status": "merged",
        "target_branch": "main",
        "target_commit": integration,
        "task_id": task_alias,
    }
    train_json = json.dumps(
        train_receipt,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    train_path = receipt_dir / f"{request.dedupe_key}.json"
    train_path.write_text(json.dumps(train_receipt), encoding="utf-8")

    queue = SimpleNamespace(
        queue_dir=queue_dir,
        completed_dir=completed_dir,
        target_repository_id=target_repository_id,
        target_branch="main",
        require_target_binding=True,
        get=lambda observed_request_id: (
            request if observed_request_id == request_id else None
        ),
    )
    train = object.__new__(MergeTrain)
    train.queue = queue
    train.receipt_dir = receipt_dir
    source_bridge = object.__new__(DatabasePortalExecutionBridge)
    source_bridge.repository_root = repo
    source_bridge.merge_queue = queue
    source_bridge.merge_target_branch = "main"
    source = source_bridge._callback_integration_source_evidence(
        request,
        SimpleNamespace(paths=paths, binding=binding, task_status="blocked"),
        train=train,
    )
    assert source is not None
    assert source["train_receipt"] == train_json
    qualification: dict[str, object] = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "post-merge-callback-integration-requalification@1"
        ),
        **source,
        "validation": [
            {
                "task_id": task_alias,
                "passed": True,
                "returncode": 0,
                "validation_result_digests": ["7" * 64],
                "command_count": 1,
                "log_sha256": "8" * 64,
            }
        ],
    }
    qualification["receipt_id"] = content_identity(qualification)
    evidence: dict[str, object] = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "database-post-merge-callback-integration-recovery@1"
        ),
        "request_id": request_id,
        "task_cid": task_cid,
        "task_alias": task_alias,
        "candidate_commit": candidate,
        "source_attempt_id": binding["attempt_id"],
        "source_claim_id": binding["claim_id"],
        "source_lease_id": binding["lease_id"],
        "source_fencing_token": binding["fencing_token"],
        "source_fence_epoch": binding["fence_epoch"],
        "source_binding_id": binding["binding_id"],
        "source_projection_immutable_digest": binding[
            "projection_immutable_digest"
        ],
        "qualified_target_commit": source["current_target_commit"],
        "callback_requalification_receipt_id": qualification["receipt_id"],
        "callback_requalification_receipt": qualification,
    }
    daemon = object.__new__(DatabaseImplementationDaemon)
    daemon._merge_queue = queue
    daemon._merge_repo_root = repo
    daemon._merge_target_branch = "main"
    daemon._merge_portal_attempt_root = projection_bridge.attempt_root
    return daemon, qualification, evidence, train_path, repo


def test_callback_integration_authority_reloads_float_receipt_and_git(
    tmp_path: Path,
) -> None:
    daemon, qualification, evidence, _train_path, _repo = (
        _callback_integration_authority_fixture(tmp_path)
    )

    verified = daemon._verified_post_merge_callback_integration_receipt(
        qualification,
        recovery_evidence=evidence,
    )

    parsed = json.loads(str(verified["train_receipt"]))
    assert parsed["started_at"] == 1787658877.9862263
    assert parsed["finished_at"] == 1787658878.9458497


@pytest.mark.parametrize(
    "tamper",
    [
        "rehash",
        "train",
        "event",
        "git_blob",
        "projection_symlink",
        "train_ancestor_symlink",
    ],
)
def test_callback_integration_authority_rejects_rehashed_or_changed_sources(
    tmp_path: Path,
    tamper: str,
) -> None:
    daemon, qualification, evidence, train_path, repo = (
        _callback_integration_authority_fixture(tmp_path)
    )
    if tamper == "rehash":
        changed = {**qualification, "entries": [dict(qualification["entries"][0])]}
        changed["entries"][0]["object_id"] = "9" * 40
        changed.pop("receipt_id")
        changed["receipt_id"] = content_identity(changed)
        qualification = changed
    elif tamper == "train":
        receipt = json.loads(train_path.read_text(encoding="utf-8"))
        receipt["merge_result"]["integration_commit_proof"]["passed"] = False
        train_path.write_text(json.dumps(receipt), encoding="utf-8")
    elif tamper == "event":
        completed = daemon._merge_queue.get(str(qualification["request_id"]))
        events_path = Path(str(completed.metadata["events_path"]))
        append_jsonl_event(
            events_path,
            "implementation_finished",
            {
                "task_id": qualification["task_ids"][0],
                "canonical_task_cid": qualification["task_cid"],
                "canonical_task_key": completed.canonical_task_key,
                "attempt": 2,
                "attempt_consumed": True,
                "provider_dispatched": True,
                "returncode": 0,
                "baseline_ref": qualification["baseline_commit"],
                "implementation_commit": qualification["candidate_commit"],
                "validation_result": {
                    "attempted": True,
                    "passed": True,
                    "returncode": 0,
                },
                "merge_result": {
                    "queued": True,
                    "merged": False,
                    "reason": "merge_queued",
                    "request_id": qualification["request_id"],
                    "implementation_commit": qualification["candidate_commit"],
                    "completion_task_cids": {
                        qualification["task_ids"][0]: qualification["task_cid"]
                    },
                },
                "board_completion": {
                    "complete": False,
                    "pending_merge": True,
                    "reason": "merge_queued_awaiting_integration",
                },
            },
        )
    elif tamper == "projection_symlink":
        completed = daemon._merge_queue.get(str(qualification["request_id"]))
        attempt_dir = Path(str(completed.metadata["todo_path"])).parent
        moved_attempt_dir = tmp_path / "off-authority-attempt"
        attempt_dir.rename(moved_attempt_dir)
        attempt_dir.symlink_to(moved_attempt_dir, target_is_directory=True)
    elif tamper == "train_ancestor_symlink":
        train_dir = daemon._merge_queue.queue_dir / "train"
        moved_train_dir = tmp_path / "off-authority-train"
        train_dir.rename(moved_train_dir)
        train_dir.symlink_to(moved_train_dir, target_is_directory=True)
    else:
        output = repo / "inventory" / "result.json"
        output.write_text('{"generation":"tampered"}\n', encoding="utf-8")
        subprocess.run(
            ["git", "add", "--", "inventory/result.json"],
            cwd=repo,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-qm", "change declared blob"],
            cwd=repo,
            check=True,
        )

    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="source authority is invalid",
    ):
        daemon._verified_post_merge_callback_integration_receipt(
            qualification,
            recovery_evidence=evidence,
        )


def test_callback_integration_recovery_seed_is_closed_and_content_addressed(
) -> None:
    daemon = object.__new__(DatabaseImplementationDaemon)
    seed = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "database-post-merge-completion-recovery-seed@1"
        ),
        "task_cid": "task:cid:004",
        "task_alias": "LGSWF-004",
        "attempt_id": "attempt:001",
        "attempt_number": 1,
        "claim_id": "claim:001",
        "lease_id": "lease:001",
        "owner_session_id": "session:bridge",
        "fencing_token": 7,
        "fence_epoch": 3,
        "source_task_revision": 11,
        "request_id": "request:callback-integration",
        "candidate_commit": "a" * 40,
        "qualified_target_commit": "b" * 40,
        "qualification_kind": "callback_integration",
        "qualification_receipt_id": "baguqeeraqualification",
        "queue_source_attempt_id": "attempt:001",
        "queue_source_claim_id": "claim:001",
        "queue_source_lease_id": "lease:001",
        "queue_source_fencing_token": 7,
        "queue_source_fence_epoch": 3,
        "queue_source_binding_id": "sha256:" + "1" * 64,
        "queue_source_projection_immutable_digest": "sha256:" + "2" * 64,
        "recovery_evidence_id": "sha256:" + "3" * 64,
        "terminal_reason": (
            "Portal completion lacks one exact evaluated baseline"
        ),
    }
    seed["seed_id"] = daemon._database_portal_evidence_digest(seed)

    assert (
        daemon._verified_post_merge_completion_recovery_seed(seed)
        == seed
    )
    tampered = {**seed, "terminal_reason": "portal_execution_deferred"}
    tampered_without_id = {
        key: value for key, value in tampered.items() if key != "seed_id"
    }
    tampered["seed_id"] = daemon._database_portal_evidence_digest(
        tampered_without_id
    )
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="recovery seed is invalid",
    ):
        daemon._verified_post_merge_completion_recovery_seed(tampered)


def test_evaluated_baseline_terminal_is_callback_recovery_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = object.__new__(DatabaseImplementationDaemon)
    attempt = _attempt()
    reason = "Portal completion lacks one exact evaluated baseline"
    task = SimpleNamespace(
        body={
            "completion_receipt": {
                "operation": "database_portal_terminal_failure",
                "attempt_id": attempt.attempt_id,
                "reason": reason,
            }
        }
    )
    monkeypatch.setattr(
        daemon,
        "_post_merge_completion_recovery_was_consumed",
        lambda _attempt: False,
    )
    monkeypatch.setattr(
        daemon,
        "_is_post_merge_completion_target_generation_changed_terminal",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        daemon,
        "_terminal_portal_failure_reason",
        lambda _attempt: reason,
    )

    assert daemon._recoverable_post_merge_terminal_reason(reason) == reason
    assert daemon._is_portal_completion_evaluated_baseline_missing_terminal(
        attempt,
        task,
    )
    assert not daemon._is_post_merge_declared_outputs_missing_terminal(
        attempt,
        task,
    )
    exact_source = {
        "source_attempt_id": attempt.attempt_id,
        "source_claim_id": attempt.claim_id,
        "source_lease_id": attempt.lease_id,
        "source_fencing_token": attempt.fencing_token,
        "source_fence_epoch": attempt.fence_epoch,
    }
    assert daemon._post_merge_source_admitted(exact_source, attempt, task)
    assert not daemon._post_merge_source_admitted(
        {
            **exact_source,
            "source_attempt_id": "attempt:older",
            "source_claim_id": "claim:older",
            "source_lease_id": "lease:older",
            "source_fencing_token": attempt.fencing_token - 1,
            "source_fence_epoch": attempt.fence_epoch - 1,
        },
        attempt,
        task,
    )


def test_callback_integration_source_requires_exact_receipt_event_and_blobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = "a" * 40
    baseline = "b" * 40
    integration = "c" * 40
    current = "d" * 40
    candidate_tree = "e" * 40
    current_tree = "f" * 40
    blob = "1" * 40
    task_alias = "LGSWF-004"
    task_cid = "task:cid:004"
    task_key = "task/v1/exact-callback-source"
    request_id = "request:callback-integration"
    dedupe_key = "2" * 64
    output = "inventory/result.json"
    events_path = tmp_path / "portal-events.jsonl"
    receipt_path = tmp_path / "train-receipt.json"
    validation_proof = {
        "attempted": True,
        "passed": True,
        "returncode": 0,
        "target_commit": candidate,
    }
    source_event = {
        "type": "implementation_finished",
        "sequence": 15,
        "event_id": "sha256:" + "3" * 64,
        "task_id": task_alias,
        "canonical_task_cid": task_cid,
        "canonical_task_key": task_key,
        "attempt": 2,
        "attempt_consumed": True,
        "provider_dispatched": True,
        "returncode": 0,
        "baseline_ref": baseline,
        "implementation_commit": candidate,
        "validation_result": {
            "attempted": True,
            "passed": True,
            "returncode": 0,
        },
        "merge_result": {
            "queued": True,
            "merged": False,
            "reason": "merge_queued",
            "request_id": request_id,
            "implementation_commit": candidate,
            "completion_task_cids": {task_alias: task_cid},
        },
        "board_completion": {
            "complete": False,
            "pending_merge": True,
            "reason": "merge_queued_awaiting_integration",
        },
    }
    completion_event = {
        "type": "task_completed",
        "sequence": 22,
        "event_id": "sha256:" + "4" * 64,
        "task_id": task_alias,
        "canonical_task_cid": task_cid,
        "reason": "task_became_completed",
        "completion_receipt_repair": False,
    }
    check = {
        "exists": True,
        "path": output,
        "reason": "declared_output_tracked",
        "repository": ".",
        "repository_ref": integration,
        "task_id": task_alias,
        "tracked": True,
        "tracked_path": output,
    }
    member = {
        "board_namespace": "task-projection.md",
        "canonical_task_cid": task_cid,
        "canonical_task_key": task_key,
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "member_completion_receipt@1"
        ),
        "status": "succeeded",
        "task_id": task_alias,
    }
    todo = {
        "already_completed_task_ids": [],
        "commit_result": {"committed": False, "reason": "no_changes"},
        "completion_reason": "single_task",
        "completion_receipts": [member],
        "inserted_status_task_ids": [],
        "missing_status_task_ids": [],
        "missing_task_ids": [],
        "path": str(tmp_path / "task-projection.md"),
        "task_id": task_alias,
        "updated": True,
        "updated_checkbox_task_ids": [],
        "updated_task_ids": [task_alias],
    }
    receipt = {
        "acceptance_pending": False,
        "accepted": True,
        "callback_owned_integration": True,
        "canonical_task_id": task_key,
        "commit_sha": candidate,
        "distributed_publication_admission": {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "distributed-lane-admission@1"
            ),
            "admitted": True,
            "distributed": False,
            "request_id": request_id,
            "status": "local",
        },
        "finished_at": 2.5,
        "integrated": True,
        "merge_commit": integration,
        "merge_result": {
            "attempted": True,
            "merged": True,
            "returncode": 0,
            "merge_commit": integration,
            "target_branch": "main",
            "integration_commit_proof": {
                "implementation_commit": candidate,
                "integration_commit": integration,
                "integration_ref": integration,
                "passed": True,
                "reasons": [],
                "target_branch": "main",
            },
            "post_merge_declared_output_invariant": {
                "checks": [check],
                "missing_outputs": [],
                "mode": "repository_tree",
                "passed": True,
                "reason": "declared_outputs_tracked",
                "repository_ref": integration,
                "task_ids": [task_alias],
                "unsafe_outputs": [],
                "untracked_outputs": [],
            },
            "todo_update_result": todo,
        },
        "merged": True,
        "request_id": request_id,
        "started_at": 1.25,
        "status": "merged",
        "target_branch": "main",
        "target_commit": integration,
        "task_id": task_alias,
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    request = SimpleNamespace(
        request_id=request_id,
        task_id=task_alias,
        canonical_task_id=task_cid,
        canonical_task_key=task_key,
        canonical_identity=task_key,
        commit_sha=candidate,
        dedupe_key=dedupe_key,
        status="completed",
        metadata={
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/merge-candidate@3"
            ),
            "baseline_ref": baseline,
            "candidate_tree": candidate_tree,
            "implementation_commit": candidate,
            "events_path": str(events_path),
            "completion_task_cids": {task_alias: task_cid},
            "validation_proof": validation_proof,
            "task": {
                "task_id": task_alias,
                "canonical_task_cid": task_cid,
                "canonical_task_key": task_key,
                "outputs": [output],
            },
        },
    )
    bridge = object.__new__(DatabasePortalExecutionBridge)
    bridge.repository_root = tmp_path
    bridge.merge_queue = object()
    bridge.merge_target_branch = "main"
    bridge._verified_event_chain = lambda _paths: [
        source_event,
        completion_event,
    ]
    projection = SimpleNamespace(
        paths=SimpleNamespace(events=events_path),
        binding={},
        task_status="blocked",
    )

    def fake_git(
        argv: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        arguments = argv[1:]
        if arguments[:2] == ["rev-parse", "--verify"]:
            ref = arguments[2]
            observed = (
                current
                if ref.startswith("refs/heads/main")
                else current_tree
                if ref == f"{current}^{{tree}}"
                else candidate_tree
            )
            return subprocess.CompletedProcess(argv, 0, observed.encode() + b"\n", b"")
        if arguments[:3] == ["rev-list", "--parents", "-n"]:
            return subprocess.CompletedProcess(
                argv,
                0,
                f"{candidate} {baseline}\n".encode(),
                b"",
            )
        if arguments[0] == "merge-base":
            return subprocess.CompletedProcess(argv, 0, b"", b"")
        if arguments[0] == "ls-tree":
            raw = f"100644 blob {blob}\t{output}\0".encode()
            return subprocess.CompletedProcess(argv, 0, raw, b"")
        raise AssertionError(arguments)

    monkeypatch.setattr(subprocess, "run", fake_git)
    train = SimpleNamespace(
        _dedupe_key=lambda _canonical, _commit: dedupe_key,
        _read_receipt=lambda _key: receipt,
        _receipt_path=lambda _key: receipt_path,
    )

    source = bridge._callback_integration_source_evidence(
        request,
        projection,
        train=train,
    )

    assert source is not None
    assert source["integration_commit"] == integration
    assert source["entries"] == [
        {
            "path": output,
            "mode": "100644",
            "object_type": "blob",
            "object_id": blob,
        }
    ]
    assert isinstance(source["train_receipt"], str)

    modern_source_event = json.loads(json.dumps(source_event))
    modern_source_event.update(
        sequence=20,
        event_id="sha256:" + "5" * 64,
        provider_dispatched=False,
        board_completion={
            "complete": True,
            "pending_merge": False,
            "reason": "merged_into_target",
        },
    )
    modern_source_event["merge_result"] = {
        "attempted": True,
        "queued": False,
        "merged": True,
        "reason": "merged",
        "request_id": request_id,
        "implementation_commit": candidate,
        "completion_task_cids": {task_alias: task_cid},
        "merge_commit": integration,
        "target_repository_id": "repository:exact-callback",
        "target_branch": "main",
    }
    reconciliation_event = {
        "type": "merge_reconciled",
        "sequence": 19,
        "event_id": "sha256:" + "6" * 64,
        "task_id": task_alias,
        "canonical_task_cid": task_cid,
        "implementation_commit": candidate,
        "request_id": request_id,
        "merge_commit": integration,
        "target_commit": integration,
    }
    modern_completion_event = {
        **completion_event,
        "sequence": 22,
        "event_id": "sha256:" + "7" * 64,
    }
    bridge.merge_queue = SimpleNamespace(
        target_repository_id="repository:exact-callback"
    )
    bridge._verified_event_chain = lambda _paths: [
        reconciliation_event,
        modern_source_event,
        modern_completion_event,
    ]
    bridge._completion_event_evidence = lambda *_args, **_kwargs: {
        "completion_source_event_type": "implementation_finished",
        "completion_source_event_id": modern_source_event["event_id"],
        "completion_event_id": modern_completion_event["event_id"],
        "completion_source_portal_attempt": modern_source_event["attempt"],
        "baseline_commit": baseline,
        "implementation_commit": candidate,
    }

    modern_source = bridge._callback_integration_source_evidence(
        request,
        projection,
        train=train,
    )

    assert modern_source is not None
    assert modern_source["source_event_id"] == modern_source_event["event_id"]
    explicit_target = json.loads(json.dumps(modern_source_event))
    explicit_target["merge_result"]["target_commit"] = "9" * 40
    bridge._verified_event_chain = lambda _paths: [
        reconciliation_event,
        explicit_target,
        modern_completion_event,
    ]
    assert (
        bridge._callback_integration_source_evidence(
            request,
            projection,
            train=train,
        )
        is None
    )

    bridge.merge_queue = object()
    bridge._verified_event_chain = lambda _paths: [
        source_event,
        completion_event,
    ]
    del bridge._completion_event_evidence
    forged = json.loads(json.dumps(receipt))
    forged["merge_result"]["integration_commit_proof"]["passed"] = False
    forged_train = SimpleNamespace(
        _dedupe_key=lambda _canonical, _commit: dedupe_key,
        _read_receipt=lambda _key: forged,
        _receipt_path=lambda _key: receipt_path,
    )
    assert (
        bridge._callback_integration_source_evidence(
            request,
            projection,
            train=forged_train,
        )
        is None
    )


def test_callback_integration_source_accepts_only_exact_settled_quarantine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = "a" * 40
    baseline = "b" * 40
    integration = "c" * 40
    current = "d" * 40
    candidate_tree = "e" * 40
    current_tree = "f" * 40
    blob = "1" * 40
    task_alias = "VRIF-032"
    task_cid = "task:cid:004"
    task_key = "task/v1/settled-callback-source"
    request_id = "request-settled-callback"
    dedupe_key = "2" * 64
    output = (
        "docs/architecture/residual_intelligence_inventory/"
        "final_release_report.json"
    )
    events_path = tmp_path / "portal-events.jsonl"
    events_path.write_text("", encoding="utf-8")
    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir()
    validation_proof = {
        "attempted": True,
        "passed": True,
        "returncode": 0,
        "target_commit": candidate,
    }
    check = {
        "exists": True,
        "path": output,
        "reason": "declared_output_tracked",
        "repository": ".",
        "repository_ref": integration,
        "task_id": task_alias,
        "tracked": True,
        "tracked_path": output,
    }
    invariant = {
        "checks": [check],
        "missing_outputs": [],
        "mode": "repository_tree",
        "passed": True,
        "reason": "declared_outputs_tracked",
        "repository_ref": integration,
        "task_ids": [task_alias],
        "unsafe_outputs": [],
        "untracked_outputs": [],
    }
    proof = {
        "implementation_commit": candidate,
        "integration_commit": integration,
        "integration_ref": integration,
        "passed": True,
        "reasons": [],
        "target_branch": "main",
    }
    member = {
        "board_namespace": "task-projection.md",
        "canonical_task_cid": task_cid,
        "canonical_task_key": task_key,
        "schema": (
            "ipfs_accelerate_py.agent_supervisor." "member_completion_receipt@1"
        ),
        "status": "succeeded",
        "task_id": task_alias,
    }
    event_ids = [f"sha256:{digit * 64}" for digit in "345678"]
    enqueue = {
        "type": "merge_candidate_enqueued",
        "sequence": 1,
        "event_id": event_ids[0],
        "task_id": task_alias,
        "canonical_task_cid": task_cid,
        "canonical_task_key": task_key,
        "attempt": 1,
        "baseline_ref": baseline,
        "implementation_commit": candidate,
        "attempted": False,
        "queued": True,
        "merged": False,
        "reason": "merge_queued",
        "request_id": request_id,
        "completion_task_cids": {task_alias: task_cid},
    }
    provenance = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor." "merge-queue-synchronous-source@1"
        ),
        "request_id": request_id,
        "task_id": task_alias,
        "task_cid": task_cid,
        "canonical_task_key": task_key,
        "merge_candidate_enqueued_event_id": enqueue["event_id"],
        "portal_attempt": 1,
        "branch": "implementation/candidate",
        "baseline_ref": baseline,
        "implementation_commit": candidate,
        "validation_target_commit": candidate,
        "validation_target_tree": candidate_tree,
        "validation_repository_tree_id": f"git-tree:{candidate_tree}",
    }
    provenance["source_projection_id"] = content_identity(provenance)
    queued_merge = {
        "attempted": False,
        "queued": True,
        "merged": False,
        "reason": "merge_queued",
        "request_id": request_id,
        "branch": "implementation/candidate",
        "implementation_commit": candidate,
        "canonical_task_key": task_key,
        "canonical_task_cid": task_cid,
        "completion_task_cids": {task_alias: task_cid},
        "target_repository_id": "repository:settled-callback",
        "target_branch": "main",
    }
    projected = {
        "type": "worktree_reconciliation_candidate_queued",
        "sequence": 2,
        "event_id": event_ids[1],
        "task_id": task_alias,
        "canonical_task_cid": task_cid,
        "canonical_task_key": task_key,
        "attempt": 1,
        "attempt_consumed": False,
        "provider_dispatched": False,
        "returncode": 0,
        "branch": "implementation/candidate",
        "baseline_ref": baseline,
        "implementation_commit": candidate,
        "validation_result": {
            "attempted": True,
            "passed": True,
            "returncode": 0,
        },
        "merge_result": queued_merge,
        "board_completion": {
            "complete": False,
            "pending_merge": True,
            "reason": "merge_queued_awaiting_integration",
        },
        "reason": "merge_queue_synchronous_source_projected",
        "merge_queue_synchronous_source": provenance,
    }
    receipt_evidence = {
        "completion_receipts": [member],
    }
    reconciliation = {
        "type": "merge_reconciled",
        "sequence": 3,
        "event_id": event_ids[2],
        "task_id": task_alias,
        "canonical_task_cid": task_cid,
        "request_id": request_id,
        "completion_source_event_id": projected["event_id"],
        "integration_commit_proof": proof,
        "post_merge_declared_output_invariant": invariant,
        "completion_receipt_evidence": receipt_evidence,
    }
    completion_error = {
        "reason": "completion_receipt_binding_mismatch",
        "expected_task_cids": {task_alias: task_cid},
        "receipt_task_cids": {},
    }
    expected_lock_path = checkout_mutation_lock_path(tmp_path)
    todo_failure = {
        "completion_reason": "single_task",
        "lock_owner_branch": "",
        "lock_owner_lease_id": "lease:settled-callback-owner",
        "lock_owner_pid": 12345,
        "lock_owner_task_id": "",
        "lock_path": str(expected_lock_path),
        "reason": "checkout_mutation_lock_exists",
        "task_id": task_alias,
        "updated": False,
    }
    quarantine_merge = {
        "already_merged": False,
        "attempted": True,
        "branch": "implementation/candidate",
        "cleanup_result": {},
        "command": ["git", "merge"],
        "completion_receipt_error": completion_error,
        "deterministic_conflict_repair": {},
        "finished_at": 1.9,
        "generated_submodule_reconciliation": {},
        "identical_untracked_paths": [],
        "integration_commit_proof": proof,
        "integration_occurred": True,
        "main_worktree_path": str(tmp_path),
        "merge_commit": integration,
        "merge_reconciliation_receipt": {
            "recorded": True,
            "replayed": False,
            "event_id": reconciliation["event_id"],
        },
        "merged": False,
        "merged_gitlink_recording": {},
        "post_merge_declared_output_invariant": invariant,
        "reason": "merge_completion_receipt_invalid",
        "resolved_generated_conflicts": [],
        "restored_generated_dirty_overlap": [],
        "returncode": 2,
        "shared_worktree_path_scrub": {},
        "started_at": 1.3,
        "stderr": "",
        "stdout": "",
        "submodule_failure_rollback": {},
        "submodule_merge_results": [],
        "target_branch": "main",
        "target_commit": integration,
        "todo_update_result": todo_failure,
        "used_ephemeral_main_worktree": True,
    }
    quarantine = {
        "acceptance_pending": False,
        "accepted": False,
        "canonical_task_id": task_key,
        "commit_sha": candidate,
        "failure_count": 1,
        "finished_at": 2.0,
        "integrated": False,
        "max_attempts": 3,
        "merge_result": quarantine_merge,
        "merged": False,
        "reason": "merge_completion_receipt_invalid",
        "request_id": request_id,
        "retryable": False,
        "started_at": 1.25,
        "status": "quarantined",
        "target_branch": "main",
        "task_id": task_alias,
    }
    terminal_merge = {
        **queued_merge,
        "train_result": json.loads(json.dumps(quarantine)),
    }
    terminal = {
        "type": "implementation_finished",
        "sequence": 4,
        "event_id": event_ids[3],
        "task_id": task_alias,
        "canonical_task_cid": task_cid,
        "canonical_task_key": task_key,
        "attempt": 1,
        "attempt_consumed": True,
        "provider_dispatched": True,
        "returncode": 0,
        "branch": "implementation/candidate",
        "baseline_ref": baseline,
        "implementation_commit": candidate,
        "validation_result": {
            "attempted": True,
            "passed": True,
            "returncode": 0,
        },
        "merge_result": terminal_merge,
        "board_completion": {
            "complete": False,
            "pending_merge": True,
            "reason": "merge_queued_awaiting_integration",
        },
    }
    status_event = {
        "type": "todo_status_updated",
        "sequence": 5,
        "event_id": event_ids[4],
        "task_id": task_alias,
        "completion_reason": "merged_status_repair",
        "updated": True,
        "updated_task_ids": [task_alias],
        "missing_task_ids": [],
        "missing_status_task_ids": [],
        "completion_receipts": [member],
    }
    completion = {
        "type": "task_completed",
        "sequence": 6,
        "event_id": event_ids[5],
        "task_id": task_alias,
        "canonical_task_cid": task_cid,
        "reason": "task_became_completed",
        "completion_receipt_repair": False,
    }
    events = [
        enqueue,
        projected,
        reconciliation,
        terminal,
        status_event,
        completion,
    ]
    revival = {
        "at": 3.0,
        "previous_enqueued_at": 1.0,
        "previous_failure_count": 1,
        "previous_failure_reason": "merge_completion_receipt_invalid",
        "reason": (
            "merge train proved quarantined candidate already integrated "
            "into exact target"
        ),
    }
    settlement = {
        "already_merged": True,
        "canonical_task_id": task_key,
        "commit_sha": candidate,
        "distributed_publication_admission": {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/" "distributed-lane-admission@1"
            ),
            "admitted": True,
            "distributed": False,
            "request_id": request_id,
            "status": "local",
        },
        "finished_at": 4.5,
        "integrated": True,
        "merge_commit": integration,
        "merged": False,
        "mutation_short_circuited": True,
        "reason": "declared_outputs_already_on_target",
        "request_id": request_id,
        "started_at": 4.0,
        "status": "already_merged",
        "target_branch": "main",
        "target_commit": integration,
        "task_id": task_alias,
    }
    metadata = {
        "schema": "ipfs_accelerate_py/agent-supervisor/merge-candidate@3",
        "baseline_ref": baseline,
        "candidate_tree": candidate_tree,
        "implementation_commit": candidate,
        "events_path": str(events_path),
        "completion_task_cids": {task_alias: task_cid},
        "validation_proof": validation_proof,
        "task": {
            "task_id": task_alias,
            "canonical_task_cid": task_cid,
            "canonical_task_key": task_key,
            "outputs": [output],
        },
        "quarantine": quarantine,
        "revivals": [revival],
    }
    request = SimpleNamespace(
        request_id=request_id,
        task_id=task_alias,
        canonical_task_id=task_cid,
        canonical_task_key=task_key,
        canonical_identity=task_key,
        commit_sha=candidate,
        dedupe_key=dedupe_key,
        status="completed",
        attempt=1,
        failure_count=0,
        failure_reason="",
        enqueued_at=revival["at"],
        metadata=metadata,
    )

    def receipt_path(key: str) -> Path:
        return receipt_dir / f"{key}.json"

    receipt_path(dedupe_key).write_text(json.dumps(settlement), encoding="utf-8")
    receipt_path(f"quarantine-{request_id}").write_text(
        json.dumps(quarantine),
        encoding="utf-8",
    )
    train = SimpleNamespace(
        receipt_dir=receipt_dir,
        _dedupe_key=lambda _canonical, _candidate: dedupe_key,
        _read_receipt=lambda key: json.loads(
            receipt_path(key).read_text(encoding="utf-8")
        ),
        _receipt_path=receipt_path,
    )
    bridge = object.__new__(DatabasePortalExecutionBridge)
    bridge.repository_root = tmp_path
    bridge.merge_queue = SimpleNamespace(
        target_repository_id="repository:settled-callback"
    )
    bridge.merge_target_branch = "main"
    bridge._verified_event_chain = lambda _paths: events
    reconciliation_calls: list[tuple[object, object]] = []

    def exact_reconciliation(
        observed_reconciliation: object,
        observed_source: object,
        **_kwargs: object,
    ) -> bool:
        reconciliation_calls.append((observed_reconciliation, observed_source))
        return True

    bridge._exact_callback_reconciliation_for_completion_source = exact_reconciliation
    projection = SimpleNamespace(
        paths=SimpleNamespace(events=events_path),
        binding={},
        task_status="blocked",
    )

    def fake_git(
        argv: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        arguments = argv[1:]
        if arguments == ["rev-parse", "--git-common-dir"]:
            return subprocess.CompletedProcess(
                argv,
                0,
                f"{tmp_path / '.git'}\n",
                "",
            )
        if arguments[:2] == ["rev-parse", "--verify"]:
            ref = arguments[2]
            observed = (
                current
                if ref.startswith("refs/heads/main")
                else current_tree if ref == f"{current}^{{tree}}" else candidate_tree
            )
            return subprocess.CompletedProcess(argv, 0, observed.encode() + b"\n", b"")
        if arguments[:3] == ["rev-list", "--parents", "-n"]:
            return subprocess.CompletedProcess(
                argv,
                0,
                f"{candidate} {baseline}\n".encode(),
                b"",
            )
        if arguments[0] == "merge-base":
            return subprocess.CompletedProcess(argv, 0, b"", b"")
        if arguments[0] == "ls-tree":
            return subprocess.CompletedProcess(
                argv,
                0,
                f"100644 blob {blob}\t{output}\0".encode(),
                b"",
            )
        raise AssertionError(arguments)

    monkeypatch.setattr(subprocess, "run", fake_git)

    source = bridge._callback_integration_source_evidence(
        request,
        projection,
        train=train,
    )

    assert source is not None
    assert source["integration_commit"] == integration
    assert source["current_target_commit"] == current
    assert source["settled_integration_source"]["source_shape"] == (
        "settled_integrated_quarantine"
    )
    assert reconciliation_calls == [(reconciliation, projected)]

    qualification = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "post-merge-callback-integration-requalification@2"
        ),
        **source,
        "validation": [
            {
                "task_id": task_alias,
                "passed": True,
                "returncode": 0,
                "validation_result_digests": ["7" * 64],
                "command_count": 1,
                "log_sha256": "8" * 64,
            }
        ],
    }
    qualification["receipt_id"] = content_identity(qualification)
    assert (
        bridge._verified_post_merge_callback_integration_receipt(
            qualification,
            source=source,
        )
        == qualification
    )
    daemon = object.__new__(DatabaseImplementationDaemon)
    authority_calls: list[tuple[object, object]] = []
    daemon._verified_post_merge_callback_integration_source_authority = (
        lambda observed, evidence: authority_calls.append(
            (observed, evidence)
        )
    )
    verified = daemon._verified_post_merge_callback_integration_receipt(
        qualification,
        recovery_evidence={"source": "test"},
    )
    assert verified == qualification
    assert authority_calls == [(qualification, {"source": "test"})]

    entry = source["entries"][0]
    bound_identity = {
        "path": entry["path"],
        "index_mode": entry["mode"],
        "index_object_id": entry["object_id"],
        "worktree_mode": entry["mode"],
        "worktree_object_id": entry["object_id"],
    }
    generated_identity = {
        **bound_identity,
        "worktree_object_id": "9" * 40,
    }
    hygiene = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "post-merge-callback-validation-workspace-hygiene@1"
        ),
        "target_commit": source["current_target_commit"],
        "target_tree": source["current_target_tree"],
        "declared_entries": source["entries"],
        "pre_validation_identities": [bound_identity],
        "generated_identities": [generated_identity],
        "restored_identities": [bound_identity],
        "generated_dirty_paths": [output],
        "restoration_performed": True,
        "final_clean": True,
    }
    hygiene["hygiene_id"] = content_identity(hygiene)
    v3_qualification = {
        **qualification,
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "post-merge-callback-integration-requalification@3"
        ),
        "workspace_hygiene": hygiene,
    }
    v3_qualification.pop("receipt_id")
    v3_qualification["receipt_id"] = content_identity(v3_qualification)
    assert (
        bridge._verified_post_merge_callback_integration_receipt(
            v3_qualification,
            source=source,
        )
        == v3_qualification
    )
    assert (
        daemon._verified_post_merge_callback_integration_receipt(
            v3_qualification,
            recovery_evidence={"source": "v3-test"},
        )
        == v3_qualification
    )
    assert authority_calls[-1] == (v3_qualification, {"source": "v3-test"})

    for field in ("restored_identities", "generated_dirty_paths"):
        tampered_v3 = json.loads(json.dumps(v3_qualification))
        if field == "restored_identities":
            tampered_v3["workspace_hygiene"][field][0][
                "worktree_object_id"
            ] = "8" * 40
        else:
            tampered_v3["workspace_hygiene"][field] = [
                "test/api/residual_intelligence/test_release_report.py"
            ]
        tampered_hygiene = dict(tampered_v3["workspace_hygiene"])
        tampered_hygiene.pop("hygiene_id")
        tampered_v3["workspace_hygiene"]["hygiene_id"] = content_identity(
            tampered_hygiene
        )
        tampered_v3.pop("receipt_id")
        tampered_v3["receipt_id"] = content_identity(tampered_v3)
        assert (
            bridge._verified_post_merge_callback_integration_receipt(
                tampered_v3,
                source=source,
            )
            is None
        )
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="workspace hygiene is invalid",
        ):
            daemon._verified_post_merge_callback_integration_receipt(
                tampered_v3,
                recovery_evidence={"source": "tampered-v3"},
            )

    malformed_paths_v3 = json.loads(json.dumps(v3_qualification))
    malformed_paths_v3["workspace_hygiene"]["generated_dirty_paths"] = [{}]
    malformed_hygiene = dict(malformed_paths_v3["workspace_hygiene"])
    malformed_hygiene.pop("hygiene_id")
    malformed_paths_v3["workspace_hygiene"]["hygiene_id"] = content_identity(
        malformed_hygiene
    )
    malformed_paths_v3.pop("receipt_id")
    malformed_paths_v3["receipt_id"] = content_identity(malformed_paths_v3)
    assert (
        bridge._verified_post_merge_callback_integration_receipt(
            malformed_paths_v3,
            source=source,
        )
        is None
    )
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="workspace hygiene is invalid",
    ):
        daemon._verified_post_merge_callback_integration_receipt(
            malformed_paths_v3,
            recovery_evidence={"source": "malformed-paths-v3"},
        )

    tampered_qualification = json.loads(json.dumps(qualification))
    tampered_qualification["settled_integration_source"][
        "completion_event_digest"
    ] = "sha256:" + "9" * 64
    with pytest.raises(DatabaseImplementationAuthorityError):
        daemon._verified_post_merge_callback_integration_receipt(
            tampered_qualification,
            recovery_evidence={"source": "test"},
        )

    terminal_merge["train_result"]["merge_result"]["integration_commit_proof"][
        "passed"
    ] = False
    assert (
        bridge._callback_integration_source_evidence(
            request,
            projection,
            train=train,
        )
        is None
    )
    terminal_merge["train_result"] = json.loads(json.dumps(quarantine))
    status_event["completion_receipts"] = []
    assert (
        bridge._callback_integration_source_evidence(
            request,
            projection,
            train=train,
        )
        is None
    )


_VRIF_CALLBACK_REPORT_JSON = (
    "docs/architecture/residual_intelligence_inventory/final_release_report.json"
)
_VRIF_CALLBACK_REPORT_MARKDOWN = (
    "docs/architecture/residual_intelligence_inventory/final_release_report.md"
)
_VRIF_CALLBACK_REPORT_TEST = "test/api/residual_intelligence/test_release_report.py"


def _run_vrif_callback_hygiene_requalification(
    tmp_path: Path,
    mutate: object,
    *,
    cleanup_authoritative: bool = True,
    task_alias: str = "VRIF-032",
) -> tuple[dict[str, object] | None, Path, list[bytes], list[dict[str, str]]]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
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
    originals = {
        _VRIF_CALLBACK_REPORT_JSON: b'{"release":"bound"}\n',
        _VRIF_CALLBACK_REPORT_MARKDOWN: b"# Bound release\n",
        _VRIF_CALLBACK_REPORT_TEST: b"def test_bound_release():\n    assert True\n",
        "outside.txt": b"outside\n",
    }
    for path, payload in originals.items():
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "bound target"], cwd=repo, check=True)

    def git_text(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    head = git_text("rev-parse", "HEAD")
    tree = git_text("rev-parse", "HEAD^{tree}")
    entries: list[dict[str, str]] = []
    for path in (
        _VRIF_CALLBACK_REPORT_JSON,
        _VRIF_CALLBACK_REPORT_MARKDOWN,
        _VRIF_CALLBACK_REPORT_TEST,
    ):
        raw = subprocess.run(
            ["git", "ls-tree", "-z", head, "--", path],
            cwd=repo,
            check=True,
            capture_output=True,
        ).stdout
        metadata, observed_path = raw[:-1].split(b"\t", 1)
        mode, object_type, object_id = metadata.decode("ascii").split()
        assert observed_path.decode("utf-8") == path
        entries.append(
            {
                "path": path,
                "mode": mode,
                "object_type": object_type,
                "object_id": object_id,
            }
        )

    state_root = tmp_path / "state"
    logs = state_root / "logs"
    cleanup_statuses: list[bytes] = []
    queue = object()
    task_cid = f"task:{task_alias.lower()}"
    task = SimpleNamespace(
        task_id=task_alias,
        canonical_task_cid=task_cid,
        validation=("python -m pytest -q test/api/residual_intelligence/test_release_report.py",),
    )

    class Portal:
        merge_queue = queue
        repo_root = repo
        resolved_merge_target_branch = "main"

        @staticmethod
        def _load_tasks() -> list[SimpleNamespace]:
            return [task]

        @staticmethod
        def _run_validation_commands(
            worktree: Path,
            _task: object,
            log_path: Path,
            *,
            force_uncached: bool,
        ) -> dict[str, object]:
            assert force_uncached is True
            assert callable(mutate)
            mutate(worktree)
            log_path.write_text("uncached validation passed\n", encoding="utf-8")
            return {
                "passed": True,
                "returncode": 0,
                "results": [{"validation_result_digest": "7" * 64}],
            }

        @staticmethod
        def _run_checkout_mutation_transaction(
            *,
            callback: object,
            **_kwargs: object,
        ) -> dict[str, object]:
            assert callable(callback)
            return callback()

        @staticmethod
        def _cleanup_main_merge_workspace(
            worktree: Path,
            *,
            ephemeral: bool,
        ) -> dict[str, object]:
            assert ephemeral is True
            cleanup_statuses.append(
                subprocess.run(
                    [
                        "git",
                        "status",
                        "--porcelain=v1",
                        "-z",
                        "--untracked-files=all",
                    ],
                    cwd=worktree,
                    check=False,
                    capture_output=True,
                ).stdout
            )
            removed = subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree)],
                cwd=repo,
                check=False,
                capture_output=True,
            )
            return {
                "cleaned": removed.returncode == 0 and cleanup_authoritative
            }

        @staticmethod
        def close() -> None:
            return None

    bridge = object.__new__(DatabasePortalExecutionBridge)
    bridge.repository_root = repo
    bridge.merge_queue = queue
    bridge.merge_target_branch = "main"
    bridge.portal_factory = lambda _paths, _alias: Portal()
    bridge._load_post_merge_callback_integration_receipt = (
        lambda path, *, source: json.loads(path.read_text(encoding="utf-8"))
    )
    source = {
        "task_ids": [task_alias],
        "task_cid": task_cid,
        "train_receipt_id": "sha256:" + "1" * 64,
        "current_target_commit": head,
        "current_target_tree": tree,
        "integration_commit": head,
        "entries": entries,
        "settled_integration_source": {"source_shape": "test-settled-source"},
    }
    projection = SimpleNamespace(
        paths=SimpleNamespace(root=state_root, implementation_logs=logs)
    )
    receipt = bridge._requalify_callback_integration(
        source,
        request=SimpleNamespace(
            task_id=task_alias,
            canonical_task_id=task_cid,
        ),
        projection=projection,
    )
    assert git_text("rev-parse", "HEAD") == head
    assert git_text("rev-parse", "HEAD^{tree}") == tree
    for path, payload in originals.items():
        assert (repo / path).read_bytes() == payload
    return receipt, repo, cleanup_statuses, entries


def test_generic_settled_clean_callback_retains_v2_without_hygiene_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_identity_capture(*_args: object, **_kwargs: object) -> object:
        pytest.fail("generic V2 callback invoked V3 identity capture")

    monkeypatch.setattr(
        DatabasePortalExecutionBridge,
        "_callback_validation_output_identities",
        staticmethod(forbidden_identity_capture),
    )
    receipt, _repo, cleanup_statuses, _entries = (
        _run_vrif_callback_hygiene_requalification(
            tmp_path,
            lambda _worktree: None,
            task_alias="LGSWF-004",
        )
    )

    assert receipt is not None
    assert receipt["schema"] == (
        "ipfs_accelerate_py.agent_supervisor."
        "post-merge-callback-integration-requalification@2"
    )
    assert "workspace_hygiene" not in receipt
    assert cleanup_statuses == [b""]


def test_callback_v3_restores_only_vrif_report_regeneration(
    tmp_path: Path,
) -> None:
    def regenerate(worktree: Path) -> None:
        (worktree / _VRIF_CALLBACK_REPORT_JSON).write_text(
            '{"release":"fixture-generated"}\n',
            encoding="utf-8",
        )
        (worktree / _VRIF_CALLBACK_REPORT_MARKDOWN).write_text(
            "# Fixture-generated release\n",
            encoding="utf-8",
        )

    receipt, _repo, cleanup_statuses, entries = (
        _run_vrif_callback_hygiene_requalification(tmp_path, regenerate)
    )

    assert receipt is not None
    assert receipt["schema"] == (
        "ipfs_accelerate_py.agent_supervisor."
        "post-merge-callback-integration-requalification@3"
    )
    hygiene = receipt["workspace_hygiene"]
    assert hygiene["declared_entries"] == entries
    assert hygiene["generated_dirty_paths"] == sorted(
        [_VRIF_CALLBACK_REPORT_JSON, _VRIF_CALLBACK_REPORT_MARKDOWN]
    )
    assert hygiene["pre_validation_identities"] == hygiene[
        "restored_identities"
    ]
    assert hygiene["generated_identities"][2] == hygiene[
        "pre_validation_identities"
    ][2]
    assert cleanup_statuses == [b""]


@pytest.mark.parametrize(
    "mutated_path",
    [_VRIF_CALLBACK_REPORT_TEST, "outside.txt"],
)
def test_callback_v3_rejects_other_declared_or_out_of_scope_mutation(
    tmp_path: Path,
    mutated_path: str,
) -> None:
    receipt, _repo, cleanup_statuses, _entries = (
        _run_vrif_callback_hygiene_requalification(
            tmp_path,
            lambda worktree: (worktree / mutated_path).write_text(
                "validation mutation\n",
                encoding="utf-8",
            ),
        )
    )

    assert receipt is None
    assert cleanup_statuses and cleanup_statuses[0]


def test_callback_v3_rejects_restore_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_run = subprocess.run

    def no_op_restore(argv: object, *args: object, **kwargs: object) -> object:
        if isinstance(argv, list) and argv[:2] == ["git", "restore"]:
            return subprocess.CompletedProcess(argv, 0, b"", b"")
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", no_op_restore)
    receipt, _repo, cleanup_statuses, _entries = (
        _run_vrif_callback_hygiene_requalification(
            tmp_path,
            lambda worktree: (worktree / _VRIF_CALLBACK_REPORT_JSON).write_text(
                '{"release":"tampered-restore"}\n',
                encoding="utf-8",
            ),
        )
    )

    assert receipt is None
    assert cleanup_statuses and cleanup_statuses[0]


@pytest.mark.parametrize(
    "mutation_kind",
    ["staged", "untracked", "deleted", "mode_only"],
)
def test_callback_v3_rejects_non_content_workspace_mutation(
    tmp_path: Path,
    mutation_kind: str,
) -> None:
    def mutate(worktree: Path) -> None:
        report = worktree / _VRIF_CALLBACK_REPORT_JSON
        if mutation_kind == "staged":
            report.write_text('{"release":"staged"}\n', encoding="utf-8")
            subprocess.run(
                ["git", "add", "--", _VRIF_CALLBACK_REPORT_JSON],
                cwd=worktree,
                check=True,
            )
        elif mutation_kind == "untracked":
            (worktree / "validation-scratch.txt").write_text(
                "scratch\n",
                encoding="utf-8",
            )
        elif mutation_kind == "deleted":
            report.unlink()
        else:
            report.chmod(0o755)

    receipt, _repo, cleanup_statuses, _entries = (
        _run_vrif_callback_hygiene_requalification(tmp_path, mutate)
    )

    assert receipt is None
    assert cleanup_statuses and cleanup_statuses[0]


def test_callback_v3_rejects_cleanup_failure_after_clean_restore(
    tmp_path: Path,
) -> None:
    receipt, _repo, cleanup_statuses, _entries = (
        _run_vrif_callback_hygiene_requalification(
            tmp_path,
            lambda worktree: (worktree / _VRIF_CALLBACK_REPORT_JSON).write_text(
                '{"release":"cleanup-failure"}\n',
                encoding="utf-8",
            ),
            cleanup_authoritative=False,
        )
    )

    assert receipt is None
    assert cleanup_statuses == [b""]


@pytest.mark.parametrize(
    "status",
    [
        b"M  path\0",
        b"?? path\0",
        b" D path\0",
        b" T path\0",
        b"R  renamed\0path\0",
        b" M path\0 M path\0",
    ],
)
def test_callback_v3_porcelain_rejects_non_worktree_content_changes(
    status: bytes,
) -> None:
    assert (
        DatabasePortalExecutionBridge._callback_validation_generated_dirty_paths(
            status
        )
        is None
    )


def test_callback_integration_evidence_builds_dedicated_retry_cas_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = object.__new__(DatabaseImplementationDaemon)
    attempt = DatabaseTaskAttempt(
        attempt_id="attempt:callback",
        claim_id="claim:callback",
        task_cid="task:cid:004",
        task_alias="LGSWF-004",
        attempt_number=1,
        owner_session_id="session:callback",
        fencing_token=7,
        fence_epoch=3,
        lease_id="lease:callback",
        committed_phase="failed",
        status="failed",
        started_at_ms=1,
        finished_at_ms=100,
        revision=3,
        body={},
    )
    task = SimpleNamespace(
        task_cid=attempt.task_cid,
        task_alias=attempt.task_alias,
        status="blocked",
        revision=12,
        body={"completion_receipt": {"operation": "terminal"}},
    )
    candidate = "a" * 40
    integration = "b" * 40
    target = "c" * 40
    qualification = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "post-merge-callback-integration-requalification@1"
        ),
        "task_ids": [attempt.task_alias],
        "task_cid": attempt.task_cid,
        "request_id": "request:callback",
        "candidate_commit": candidate,
        "integration_commit": integration,
        "train_receipt_id": "sha256:" + "4" * 64,
        "current_target_commit": target,
    }
    qualification["receipt_id"] = content_identity(qualification)
    evidence = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "database-post-merge-callback-integration-recovery@1"
        ),
        "request_id": "request:callback",
        "task_cid": attempt.task_cid,
        "task_alias": attempt.task_alias,
        "candidate_commit": candidate,
        "source_attempt_id": attempt.attempt_id,
        "source_claim_id": attempt.claim_id,
        "source_lease_id": attempt.lease_id,
        "source_fencing_token": attempt.fencing_token,
        "source_fence_epoch": attempt.fence_epoch,
        "source_binding_id": "sha256:" + "1" * 64,
        "source_projection_immutable_digest": "sha256:" + "2" * 64,
        "qualified_target_commit": target,
        "callback_requalification_receipt_id": qualification["receipt_id"],
        "callback_requalification_receipt": qualification,
    }
    evidence["evidence_id"] = daemon._database_portal_evidence_digest(
        evidence
    )
    captured: dict[str, object] = {}

    class TaskSource:
        def get(self, task_cid: str) -> object | None:
            return task if task_cid == attempt.task_cid else None

        def record_queue_backoff_and_cas_status(
            self,
            **kwargs: object,
        ) -> object:
            captured.update(kwargs)
            return SimpleNamespace(
                to_dict=lambda: {
                    "status": "retrying",
                    "receipt": kwargs.get("receipt"),
                }
            )

    daemon._task_source = TaskSource()
    monkeypatch.setattr(daemon, "open", lambda: daemon)
    monkeypatch.setattr(
        daemon,
        "_require_execution_authority",
        lambda _operation: None,
    )
    monkeypatch.setattr(
        daemon,
        "_verified_post_merge_callback_integration_receipt",
        lambda raw, **_kwargs: dict(raw),
    )
    monkeypatch.setattr(
        daemon,
        "_automatic_claim_forbidden",
        lambda _task: False,
    )
    monkeypatch.setattr(
        daemon,
        "_post_merge_completion_crash_recovery_context",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        daemon,
        "_latest_failed_attempts",
        lambda: (attempt,),
    )
    monkeypatch.setattr(
        daemon,
        "_post_merge_source_admitted",
        lambda *_args: True,
    )
    monkeypatch.setattr(
        daemon,
        "_is_post_merge_declared_outputs_missing_terminal",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        daemon,
        "_is_portal_completion_evaluated_baseline_missing_terminal",
        lambda *_args: True,
    )
    monkeypatch.setattr(
        daemon,
        "_require_control_attempt_receipt",
        lambda *_args, **_kwargs: {"operation": "terminal"},
    )
    monkeypatch.setattr(
        daemon,
        "_post_merge_completion_recovery_source_terminal_reason",
        lambda *_args: "Portal completion lacks one exact evaluated baseline",
    )
    coordination = {
        "attempt_id": attempt.attempt_id,
        "claim_id": attempt.claim_id,
        "attempt_number": attempt.attempt_number,
    }
    monkeypatch.setattr(
        daemon,
        "_reconcile_failed_attempt_coordination",
        lambda _attempt: coordination,
    )
    monkeypatch.setattr(
        daemon,
        "_superseded_failed_attempt_reconciliation",
        lambda *_args, **_kwargs: None,
    )

    def execute_retry(
        _attempt: object,
        _coordination: object,
        callback: object,
        **_kwargs: object,
    ) -> dict[str, object]:
        cas_result = callback()
        return {
            "cas_result": cas_result,
            "queue_reused": False,
            "queue_receipt": {},
        }

    monkeypatch.setattr(
        daemon,
        "_execute_with_retry_transition_authority",
        execute_retry,
    )

    result = daemon.recover_blocked_post_merge_declared_outputs(evidence)

    transition = captured["receipt"]
    assert isinstance(transition, dict)
    assert transition["operation"] == (
        "database_post_merge_declared_outputs_callback_integration_recovery"
    )
    assert transition["source_integration_commit"] == integration
    assert transition["source_train_receipt_id"] == qualification[
        "train_receipt_id"
    ]
    seed = transition["post_merge_completion_recovery_seed"]
    assert seed["qualification_kind"] == "callback_integration"
    assert seed["terminal_reason"] == (
        "Portal completion lacks one exact evaluated baseline"
    )
    assert captured["status"] == "retrying"
    assert result["recovered"] is True
    assert result["write_count"] == 2

    retrying = SimpleNamespace(
        status="retrying",
        revision=int(transition["control_expected_revision"]) + 1,
        body={"completion_receipt": transition},
    )

    class ReplaySource:
        @staticmethod
        def get_queue_entry(task_cid: str) -> object | None:
            return (
                SimpleNamespace(reason=transition["queue_reason"])
                if task_cid == attempt.task_cid
                else None
            )

    daemon._task_source = ReplaySource()
    monkeypatch.setattr(
        daemon,
        "_post_merge_completion_terminal_receipt_from_history",
        lambda **_kwargs: {
            "reason": "Portal completion lacks one exact evaluated baseline"
        },
    )
    replay = daemon._verified_post_merge_declared_output_recovery_state(
        attempt,
        retrying,
        expected_evidence=evidence,
    )
    assert replay["qualification_kind"] == "callback_integration"
    assert replay["post_merge_completion_recovery_seed"] == seed


def test_post_merge_completion_recovery_seed_closes_without_portal_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _record()
    record.status = "in_progress"
    portal_calls: list[str] = []
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, alias: portal_calls.append(alias),
    )
    seed = {
        "request_id": "request:repaired",
        "baseline_commit": "b" * 40,
        "candidate_commit": "a" * 40,
        "seed_id": "sha256:" + "c" * 64,
        "recovery_evidence_id": "sha256:" + "d" * 64,
    }
    monkeypatch.setattr(
        bridge,
        "_post_merge_completion_recovery_seed_from_record",
        lambda **_kwargs: dict(seed),
    )

    receipt = bridge.run_provider(_attempt())

    assert portal_calls == []
    assert receipt["accepted"] is True
    assert receipt["completion_authority"] == "DatabaseImplementationDaemon"
    assert receipt["baseline_commit"] == "b" * 40
    assert receipt["implementation_commit"] == "a" * 40
    assert [
        event["type"]
        for event in bridge._verified_event_chain(bridge._paths(_attempt()))
    ] == [
        "worktree_reconciliation_candidate_queued",
        "merge_reconciled",
        "task_completed",
    ]


def test_post_merge_completion_recovery_never_repairs_bare_completion(
    tmp_path: Path,
) -> None:
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: None,
    )
    record = bridge._record_for_attempt(bridge.task_source, _attempt())
    paths, _binding = bridge._ensure_attempt_projection(_attempt(), record)
    append_jsonl_event(
        paths.events,
        "task_completed",
        {
            "task_id": "LGSWF-004",
            "canonical_task_cid": "task:cid:004",
        },
    )

    with pytest.raises(
        DatabasePortalBridgeError,
        match="cannot repair a bare completion",
    ):
        bridge._ensure_post_merge_completion_recovery_events(
            paths,
            alias="LGSWF-004",
            task_cid="task:cid:004",
            request_id="request:repaired",
            baseline_commit="b" * 40,
            implementation_commit="a" * 40,
            seed_id="sha256:" + "c" * 64,
            recovery_evidence_id="sha256:" + "d" * 64,
        )


@pytest.mark.parametrize("tamper", ["source-payload", "stage-order"])
def test_post_merge_completion_recovery_rejects_conflicting_partial_seed(
    tmp_path: Path,
    tamper: str,
) -> None:
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: None,
    )
    record = bridge._record_for_attempt(bridge.task_source, _attempt())
    paths, _binding = bridge._ensure_attempt_projection(_attempt(), record)
    seed_id = "sha256:" + "c" * 64
    if tamper == "source-payload":
        append_jsonl_event(
            paths.events,
            "worktree_reconciliation_candidate_queued",
            {
                "task_id": "LGSWF-004",
                "canonical_task_cid": "task:cid:004",
                "attempt": 1,
                "attempt_consumed": False,
                "provider_dispatched": False,
                "returncode": 0,
                "baseline_ref": "b" * 40,
                "implementation_commit": "a" * 40,
                "validation_result": {
                    "attempted": True,
                    "passed": True,
                    "returncode": 0,
                },
                "merge_result": {
                    "attempted": True,
                    "merged": False,
                    "queued": True,
                    "request_id": "request:forged",
                },
                "reason": "post_merge_completion_recovery_seed",
                "post_merge_completion_recovery_seed_id": seed_id,
            },
        )
    else:
        append_jsonl_event(
            paths.events,
            "merge_reconciled",
            {
                "task_id": "LGSWF-004",
                "canonical_task_cid": "task:cid:004",
                "attempt": 1,
                "implementation_commit": "a" * 40,
                "resolved": True,
                "post_merge_completion_recovery_seed_id": seed_id,
                "merge_result": {"merged": True},
            },
        )

    with pytest.raises(DatabasePortalBridgeError):
        bridge._ensure_post_merge_completion_recovery_events(
            paths,
            alias="LGSWF-004",
            task_cid="task:cid:004",
            request_id="request:repaired",
            baseline_commit="b" * 40,
            implementation_commit="a" * 40,
            seed_id=seed_id,
            recovery_evidence_id="sha256:" + "d" * 64,
        )
    assert len(bridge._verified_event_chain(paths)) == 1


def test_post_merge_completion_seed_admits_only_exact_shared_lane_source(
    tmp_path: Path,
) -> None:
    shared = tmp_path / "state"
    source_attempt_root = (
        shared / "lane-0" / "vrif_lane_0_database_portal_attempts"
    )
    consumer_attempt_root = (
        shared / "lane-3" / "vrif_lane_3_database_portal_attempts"
    )
    source_record = _record()
    source_record.status = "in_progress"
    source_bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(source_record),
        attempt_root=source_attempt_root,
        portal_factory=lambda _paths, _alias: None,
    )
    paths, binding = source_bridge._ensure_attempt_projection(
        _attempt(), source_record
    )
    [projected_task] = parse_task_text(
        paths.task_projection.read_text(encoding="utf-8"),
        path=paths.task_projection,
        task_header_prefix="## LGSWF-",
    )
    completion = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "post-merge-declared-output-completion@1"
        ),
        "status": "already_merged",
        "reason": "post_merge_declared_outputs_repaired",
        "candidate_commit": "a" * 40,
        "target_commit": "b" * 40,
        "repair_receipt": {},
    }
    metadata = {
        "schema": "ipfs_accelerate_py/agent-supervisor/merge-candidate@3",
        "target_binding_schema": (
            "ipfs_accelerate_py/agent-supervisor/merge-target-binding@1"
        ),
        "target_repository_id": "repository:test",
        "target_branch": "main",
        "implementation_commit": "a" * 40,
        "todo_path": str(paths.task_projection),
        "state_path": str(paths.state),
        "strategy_path": str(paths.strategy),
        "events_path": str(paths.events),
        "repo_root": str((tmp_path / "repo").absolute()),
        "task_header_prefix": "## LGSWF-",
        "task": asdict(projected_task),
        "completion_task_cids": {"LGSWF-004": "task:cid:004"},
        "false_positive_completion_reopen": {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "merge-queue-false-positive-completion-reopen@1"
            ),
            "reason": "declared_outputs_not_on_target",
            "train_receipt_id": "sha256:" + "c" * 64,
        },
        "completion": completion,
    }
    request = SimpleNamespace(
        request_id="request:shared-lane",
        task_id="LGSWF-004",
        canonical_task_id="task:cid:004",
        canonical_task_key=projected_task.canonical_task_key,
        commit_sha="a" * 40,
        status="completed",
        failure_reason="",
        metadata=metadata,
    )
    consumer = object.__new__(DatabasePortalExecutionBridge)
    consumer.task_source = _TaskSource(source_record)
    consumer.attempt_root = consumer_attempt_root.absolute()
    consumer.attempt_root.mkdir(parents=True)
    consumer.repository_root = (tmp_path / "repo").absolute()
    consumer.merge_target_branch = "main"
    consumer.task_header_prefix = "## LGSWF-"
    consumer.merge_queue = SimpleNamespace(
        target_repository_id="repository:test"
    )

    assert consumer._owned_post_merge_recovery_projection(request) is None
    admitted = consumer._owned_post_merge_recovery_projection(
        request,
        allowed_task_statuses=frozenset({"in_progress"}),
        allow_shared_lane_source=True,
    )
    assert admitted is not None
    assert admitted.binding["attempt_id"] == binding["attempt_id"]
    assert admitted.binding["binding_id"] == binding["binding_id"]

    mismatched_root = shared / "lane-1" / "vrif_lane_2_database_portal_attempts"
    mismatched_paths = DatabasePortalExecutionBridge(
        task_source=_TaskSource(source_record),
        attempt_root=mismatched_root,
        portal_factory=lambda _paths, _alias: None,
    )._ensure_attempt_projection(_attempt(), source_record)[0]
    malformed = SimpleNamespace(
        **{
            **request.__dict__,
            "metadata": {
                **metadata,
                "todo_path": str(mismatched_paths.task_projection),
                "state_path": str(mismatched_paths.state),
                "strategy_path": str(mismatched_paths.strategy),
                "events_path": str(mismatched_paths.events),
            },
        }
    )
    assert (
        consumer._owned_post_merge_recovery_projection(
            malformed,
            allowed_task_statuses=frozenset({"in_progress"}),
            allow_shared_lane_source=True,
        )
        is None
    )

    foreign_prefix_root = (
        shared / "lane-0" / "foreign_board_lane_0_database_portal_attempts"
    )
    foreign_prefix_paths = DatabasePortalExecutionBridge(
        task_source=_TaskSource(source_record),
        attempt_root=foreign_prefix_root,
        portal_factory=lambda _paths, _alias: None,
    )._ensure_attempt_projection(_attempt(), source_record)[0]
    foreign_prefix = SimpleNamespace(
        **{
            **request.__dict__,
            "metadata": {
                **metadata,
                "todo_path": str(foreign_prefix_paths.task_projection),
                "state_path": str(foreign_prefix_paths.state),
                "strategy_path": str(foreign_prefix_paths.strategy),
                "events_path": str(foreign_prefix_paths.events),
            },
        }
    )
    assert (
        consumer._owned_post_merge_recovery_projection(
            foreign_prefix,
            allowed_task_statuses=frozenset({"in_progress"}),
            allow_shared_lane_source=True,
        )
        is None
    )

    traversal_projection = (
        paths.root / ".." / paths.root.name / paths.task_projection.name
    )
    traversal = SimpleNamespace(
        **{
            **request.__dict__,
            "metadata": {
                **metadata,
                "todo_path": str(traversal_projection),
            },
        }
    )
    assert (
        consumer._owned_post_merge_recovery_projection(
            traversal,
            allowed_task_statuses=frozenset({"in_progress"}),
            allow_shared_lane_source=True,
        )
        is None
    )

    symlink_root = shared / "lane-2" / "vrif_lane_2_database_portal_attempts"
    symlink_root.parent.mkdir(parents=True)
    symlink_root.symlink_to(source_attempt_root, target_is_directory=True)
    linked_attempt = symlink_root / paths.root.name
    linked = SimpleNamespace(
        **{
            **request.__dict__,
            "metadata": {
                **metadata,
                "todo_path": str(linked_attempt / paths.task_projection.name),
                "state_path": str(linked_attempt / paths.state.name),
                "strategy_path": str(linked_attempt / paths.strategy.name),
                "events_path": str(linked_attempt / paths.events.name),
            },
        }
    )
    assert (
        consumer._owned_post_merge_recovery_projection(
            linked,
            allowed_task_statuses=frozenset({"in_progress"}),
            allow_shared_lane_source=True,
        )
        is None
    )


def test_post_merge_completion_recovery_priority_snapshot_is_exact(
    tmp_path: Path,
) -> None:
    del tmp_path
    bridge = object.__new__(DatabasePortalExecutionBridge)
    exact = SimpleNamespace(
        request_id="request-exact",
        canonical_task_id="task:cid:004",
    )
    calls: list[dict[str, object]] = []

    def completed_requests(**kwargs: object) -> tuple[object, ...]:
        calls.append(dict(kwargs))
        return (exact,)

    bridge.merge_queue = SimpleNamespace(completed_requests=completed_requests)

    assert bridge._priority_repaired_completion_requests(
        ("task:cid:004",)
    ) == (exact,)
    assert calls == [
        {
            "limit": 2,
            "completion_schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "post-merge-declared-output-completion@1"
            ),
            "completion_reason": "post_merge_declared_outputs_repaired",
            "canonical_task_id": "task:cid:004",
            "reopen_schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "merge-queue-false-positive-completion-reopen@1"
            ),
            "reopen_reason": "declared_outputs_not_on_target",
        },
        {
            "limit": 256,
            "canonical_task_id": "task:cid:004",
        },
    ]

    bridge.merge_queue.completed_requests = lambda **_kwargs: (exact, exact)
    assert bridge._priority_repaired_completion_requests(
        ("task:cid:004",)
    ) == ()


def test_post_merge_completion_recovery_priority_pages_cannot_pin_after_32(
) -> None:
    task_cids = tuple(f"task:cid:{index:03d}" for index in range(35))

    first, first_cursor = (
        DatabasePortalExecutionBridge._priority_recovery_task_cid_page(
            task_cids,
            after_task_cid="",
        )
    )
    second, second_cursor = (
        DatabasePortalExecutionBridge._priority_recovery_task_cid_page(
            task_cids,
            after_task_cid=first_cursor,
        )
    )
    wrapped, wrapped_cursor = (
        DatabasePortalExecutionBridge._priority_recovery_task_cid_page(
            task_cids,
            after_task_cid=second_cursor,
        )
    )

    assert first == task_cids[:32]
    assert second == task_cids[32:]
    assert wrapped == ()
    assert wrapped_cursor == ""
    restarted, _ = (
        DatabasePortalExecutionBridge._priority_recovery_task_cid_page(
            task_cids,
            after_task_cid=wrapped_cursor,
        )
    )
    assert restarted == first


def test_bridge_apply_effect_rejects_resealed_nonancestor_baseline(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repository"
    baseline_commit, _baseline_tree, implementation_commit = (
        _git_completion_lineage(repository_root)
    )
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: _CompletingPortal(
            paths,
            alias,
            baseline_commit=baseline_commit,
            implementation_commit=implementation_commit,
        ),
        repository_root=repository_root,
    )
    provider = dict(bridge.run_provider(_attempt()))
    subprocess.run(
        ["git", "checkout", "-q", baseline_commit],
        cwd=repository_root,
        check=True,
    )
    sibling = repository_root / "inventory" / "sibling.json"
    sibling.write_text('{"sibling":true}\n', encoding="utf-8")
    subprocess.run(
        ["git", "add", "--", "inventory/sibling.json"],
        cwd=repository_root,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-qm", "sibling"],
        cwd=repository_root,
        check=True,
    )
    nonancestor = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    evidence = dict(provider["portal_evidence"])
    evidence["baseline_commit"] = nonancestor
    provider["baseline_commit"] = nonancestor
    provider["portal_evidence"] = evidence
    provider["evidence_digest"] = _capacity_record_id(
        evidence,
        "__no_identity_field__",
    )
    provider["receipt_id"] = _capacity_record_id(provider, "receipt_id")

    with pytest.raises(
        DatabasePortalBridgeError,
        match="unproven Portal commit lineage",
    ):
        bridge.apply_effect(_attempt(), provider)


def test_bridge_apply_effect_replays_equal_commit_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = tmp_path / "repository"
    baseline_commit, _baseline_tree, implementation_commit = (
        _git_completion_lineage(repository_root)
    )
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: _CompletingPortal(
            paths,
            alias,
            baseline_commit=baseline_commit,
            implementation_commit=implementation_commit,
        ),
        repository_root=repository_root,
    )
    provider = dict(bridge.run_provider(_attempt()))
    evidence = dict(provider["portal_evidence"])
    evidence["implementation_commit"] = baseline_commit
    provider["implementation_commit"] = baseline_commit
    provider["portal_evidence"] = evidence
    provider["evidence_digest"] = _capacity_record_id(
        evidence,
        "__no_identity_field__",
    )
    provider["receipt_id"] = _capacity_record_id(provider, "receipt_id")
    observed: list[dict[str, object]] = []

    def revalidate(**kwargs: object) -> None:
        observed.append(dict(kwargs))

    monkeypatch.setattr(
        bridge,
        "_revalidate_equal_commit_completion",
        revalidate,
    )

    effect = bridge.apply_effect(_attempt(), provider)

    assert effect["implementation_commit"] == baseline_commit
    assert len(observed) == 1
    assert observed[0]["attempt"] == _attempt()
    assert observed[0]["evidence"] == evidence


@pytest.mark.parametrize(
    ("field", "tampered_value"),
    [
        ("receipt_id", "sha256:" + "f" * 64),
        ("execution_mode", "unbound-provider-mode"),
        ("completion_authority", "UntrustedCompleter"),
        ("baseline_commit", "c" * 40),
        ("implementation_commit", "b" * 40),
        ("completion_event_id", "sha256:" + "c" * 64),
    ],
)
def test_bridge_apply_effect_rejects_tampered_provider_receipt_joins(
    tmp_path: Path,
    field: str,
    tampered_value: str,
) -> None:
    repository_root = tmp_path / "repository"
    baseline_commit, _baseline_tree, implementation_commit = (
        _git_completion_lineage(repository_root)
    )
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: _CompletingPortal(
            paths,
            alias,
            baseline_commit=baseline_commit,
            implementation_commit=implementation_commit,
        ),
        repository_root=repository_root,
    )
    provider = dict(bridge.run_provider(_attempt()))
    provider[field] = tampered_value
    if field != "receipt_id":
        provider["receipt_id"] = _capacity_record_id(provider, "receipt_id")

    with pytest.raises(DatabasePortalBridgeError):
        bridge.apply_effect(_attempt(), provider)


@pytest.mark.parametrize(
    ("field", "tampered_value"),
    [
        ("effect_key", "portal:foreign:attempt"),
        ("binding_id", "sha256:" + "d" * 64),
        ("portal_receipt_id", "sha256:" + "e" * 64),
        ("baseline_commit", "c" * 40),
        ("baseline_tree", "d" * 40),
        ("implementation_commit", "b" * 40),
        ("completion_event_id", "sha256:" + "c" * 64),
    ],
)
def test_bridge_validate_effect_rejects_tampered_completion_binding_joins(
    tmp_path: Path,
    field: str,
    tampered_value: str,
) -> None:
    repository_root = tmp_path / "repository"
    baseline_commit, _baseline_tree, implementation_commit = (
        _git_completion_lineage(repository_root)
    )
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: _CompletingPortal(
            paths,
            alias,
            baseline_commit=baseline_commit,
            implementation_commit=implementation_commit,
        ),
        repository_root=repository_root,
    )
    provider = bridge.run_provider(_attempt())
    effect = dict(bridge.apply_effect(_attempt(), provider))
    effect[field] = tampered_value

    with pytest.raises(DatabasePortalBridgeError):
        bridge.validate_effect(_attempt(), effect)


@pytest.mark.parametrize(
    "field",
    [
        "binding_id",
        "portal_receipt_id",
        "evidence_digest",
        "baseline_commit",
        "baseline_tree",
        "implementation_commit",
        "completion_event_id",
    ],
)
def test_bridge_validate_effect_rejects_resealed_inner_binding_mismatch(
    tmp_path: Path,
    field: str,
) -> None:
    repository_root = tmp_path / "repository"
    baseline_commit, _baseline_tree, implementation_commit = (
        _git_completion_lineage(repository_root)
    )
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: _CompletingPortal(
            paths,
            alias,
            baseline_commit=baseline_commit,
            implementation_commit=implementation_commit,
        ),
        repository_root=repository_root,
    )
    provider = bridge.run_provider(_attempt())
    effect = dict(bridge.apply_effect(_attempt(), provider))
    binding = dict(effect["portal_completion_binding"])
    binding[field] = (
        "b" * 40
        if field
        in {"baseline_commit", "baseline_tree", "implementation_commit"}
        else "sha256:" + "f" * 64
    )
    binding["receipt_id"] = _capacity_record_id(binding, "receipt_id")
    effect["portal_completion_binding"] = binding

    with pytest.raises(DatabasePortalBridgeError):
        bridge.validate_effect(_attempt(), effect)


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
        "pkg/generated/",
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




def test_post_merge_recovery_is_inert_without_bound_queue(tmp_path: Path) -> None:
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: _CompletingPortal(paths, alias),
    )
    assert bridge.merge_queue is None
    assert bridge.recover_post_merge_declared_outputs(object()) is None




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
            match="recovery schema is invalid",
        ):
            daemon.recover_blocked_post_merge_declared_outputs(
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


def test_typed_output_rearm_requires_bound_post_merge_recovery() -> None:
    daemon = SimpleNamespace(
        task_source=SimpleNamespace(
            record_task_retry_cooldown=lambda **_kwargs: None,
        ),
        _post_merge_recovery_fn=None,
    )

    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="requires a bound post-merge recovery callback",
    ):
        DatabaseImplementationDaemon._rearm_blocked_tasks_with_outputs_on_head(
            daemon
        )


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
        recovery = daemon._run_post_merge_recovery()
        assert recovery["attempted"] is True
        assert recovery["recovered"] is False
        assert recovery["reason"] == "no_recoverable_post_merge_request"
        assert checkout_repository_id(repo) == daemon._merge_queue.target_repository_id
    finally:
        daemon.close()


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


def test_bridge_reopens_exact_false_declared_output_completion(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "reopen@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Reopen Test"],
        cwd=repo,
        check=True,
    )
    output = repo / "inventory" / "result.json"
    output.parent.mkdir(parents=True)
    output.write_text('{"version":"base"}\n', encoding="utf-8")
    subprocess.run(["git", "add", "inventory/result.json"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)
    target = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "switch", "-q", "-c", "implementation/candidate"],
        cwd=repo,
        check=True,
    )
    output.write_text('{"version":"candidate"}\n', encoding="utf-8")
    subprocess.run(["git", "add", "inventory/result.json"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "candidate"], cwd=repo, check=True)
    candidate = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(["git", "switch", "-q", "main"], cwd=repo, check=True)

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
    paths, _binding = seed_bridge._ensure_attempt_projection(_attempt(), record)
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
    request = queue.enqueue(
        branch_name="implementation/candidate",
        task_id="LGSWF-004",
        canonical_task_id="task:cid:004",
        canonical_task_key=str(projected_task.canonical_task_key),
        commit_sha=candidate,
        metadata={
            "schema": "ipfs_accelerate_py/agent-supervisor/merge-candidate@3",
            "target_binding_schema": (
                "ipfs_accelerate_py/agent-supervisor/merge-target-binding@1"
            ),
            "target_repository_id": repository_id,
            "target_branch": "main",
            "implementation_commit": candidate,
            "todo_path": str(paths.task_projection),
            "state_path": str(paths.state),
            "strategy_path": str(paths.strategy),
            "events_path": str(paths.events),
            "repo_root": str(repo.absolute()),
            "task_header_prefix": "## LGSWF-",
            "task": asdict(projected_task),
            "completion_task_cids": {"LGSWF-004": "task:cid:004"},
            "changed_submodule_paths": [],
        },
    )
    claimed = queue.claim_pending_request(
        request.request_id,
        consumer_id="merge-train:false-shortcut-fixture",
    )
    assert claimed is not None
    queue.complete(claimed)
    completed = queue.get(request.request_id)
    assert completed is not None and completed.status == "completed"
    train = MergeTrain(repo, queue, target_branch="main")
    assert train.portal_declared_outputs_match_target(completed) is False
    train_receipt = {
        "already_merged": True,
        "canonical_task_id": completed.canonical_identity,
        "commit_sha": candidate,
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
        "merge_commit": target,
        "merged": False,
        "mutation_short_circuited": True,
        "reason": "declared_outputs_already_on_target",
        "request_id": request.request_id,
        "started_at": 1.0,
        "status": "already_merged",
        "target_branch": "main",
        "target_commit": target,
        "task_id": "LGSWF-004",
    }
    train._write_receipt(completed.dedupe_key, train_receipt)
    (repo / "supervisor-fix.txt").write_text("fixed\n", encoding="utf-8")
    subprocess.run(["git", "add", "supervisor-fix.txt"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "advance target without declared outputs"],
        cwd=repo,
        check=True,
    )
    advanced_target = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert advanced_target != target
    portal_calls: list[str] = []
    checkout_mutation_calls: list[str] = []
    transaction_targets: list[str] = []

    class CheckoutAuthorityPortal:
        def __init__(self, alias: str) -> None:
            portal_calls.append(alias)
            self.merge_queue = queue
            self.repo_root = repo.absolute()
            self.resolved_merge_target_branch = "main"

        def _run_checkout_mutation_transaction(
            self,
            *,
            operation: str,
            callback: object,
            **_kwargs: object,
        ) -> dict[str, object]:
            checkout_mutation_calls.append(operation)
            assert callable(callback)
            # Simulate a target writer finishing immediately before this
            # transaction acquires the shared checkout lease.  Recovery must
            # capture and bind the newer target from inside the transaction.
            (repo / "pre-lock-target-advance.txt").write_text(
                "advanced before recovery lock\n",
                encoding="utf-8",
            )
            subprocess.run(
                ["git", "add", "pre-lock-target-advance.txt"],
                cwd=repo,
                check=True,
            )
            subprocess.run(
                ["git", "commit", "-qm", "advance before recovery lock"],
                cwd=repo,
                check=True,
            )
            transaction_targets.append(
                subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=repo,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            )
            return dict(callback())

        @staticmethod
        def close_event_runtime() -> None:
            return None

    bridge = DatabasePortalExecutionBridge(
        task_source=task_source,
        attempt_root=attempt_root,
        portal_factory=lambda _paths, alias: CheckoutAuthorityPortal(alias),
        repository_root=repo,
        merge_queue=queue,
        merge_target_branch="main",
        task_header_prefix="## LGSWF-",
    )

    class Authority:
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
            result = {
                **dict(source),
                "authorized": True,
                "task_status": "blocked",
            }
            result["authorization_id"] = (
                self._database_portal_evidence_digest(result)
            )
            return result

        @staticmethod
        def recover_blocked_post_merge_declared_outputs(
            _evidence: object,
        ) -> dict[str, object]:
            pytest.fail("false completion must reopen before database rearm")

    owned_projection = bridge._owned_post_merge_recovery_projection(completed)
    assert owned_projection is not None
    bridge._save_post_merge_recovery_cursors(
        {
            "completed_requests": "",
            "pending_requests": "9999999999999999999-stale-cursor",
            "processing_requests": "9999999999999999999-stale-cursor",
            "quarantined_requests": "9999999999999999999-stale-cursor",
        }
    )
    authority = Authority()

    missing_checkout_closed: list[bool] = []

    class MissingCheckoutAuthorityPortal:
        merge_queue = queue
        repo_root = repo.absolute()
        resolved_merge_target_branch = "main"

        @staticmethod
        def close_event_runtime() -> None:
            missing_checkout_closed.append(True)

    missing_checkout_bridge = DatabasePortalExecutionBridge(
        task_source=task_source,
        attempt_root=attempt_root,
        portal_factory=lambda _paths, _alias: MissingCheckoutAuthorityPortal(),
        repository_root=repo,
        merge_queue=queue,
        merge_target_branch="main",
        task_header_prefix="## LGSWF-",
    )
    with pytest.raises(
        DatabasePortalBridgeError,
        match="lacks checkout mutation authority",
    ):
        missing_checkout_bridge.recover_post_merge_declared_outputs(authority)
    unchanged = queue.get(request.request_id)
    assert unchanged is not None and unchanged.status == "completed"
    assert missing_checkout_closed == [True]

    result = bridge.recover_post_merge_declared_outputs(authority)

    assert result is not None
    assert result["schema"].endswith(
        "database-post-merge-declared-output-recovery@1"
    )
    assert result["recovered"] is False
    assert result["reason"] == "false_positive_completion_reopened"
    assert result["request_id"] == request.request_id
    assert result["candidate_commit"] == candidate
    assert transaction_targets
    assert transaction_targets == [result["target_commit"]]
    assert transaction_targets[0] != advanced_target
    assert result["write_count"] == 1
    assert checkout_mutation_calls == [
        "reopen_false_positive_merge_completion"
    ]
    recovery_cursors = bridge._load_post_merge_recovery_cursors()
    assert recovery_cursors["completed_requests"] == request.request_id
    assert recovery_cursors["pending_requests"] == ""
    assert recovery_cursors["processing_requests"] == ""
    assert recovery_cursors["quarantined_requests"] == ""
    reopened = queue.get(request.request_id)
    assert reopened is not None and reopened.status == "pending"
    assert reopened.metadata["false_positive_completion_reopen"]["reason"] == (
        "declared_outputs_not_on_target"
    )
    assert bridge._owned_post_merge_recovery_projection(reopened) is not None
    recovery_train = MergeTrain(repo, queue, target_branch="main")
    assert recovery_train._pending_request_is_integrated_quarantine_revival(
        reopened,
        allow_post_merge_declared_output_recovery=True,
    )
    assert recovery_train._quarantine_may_auto_recover(
        reopened,
        allow_post_merge_declared_output_recovery=True,
    )
    assert not recovery_train._quarantine_may_auto_recover(reopened)
    assert portal_calls == ["LGSWF-004"]
    assert subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip() == transaction_targets[0]


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_production_database_daemon_uses_portal_checkout_authority_for_false_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "production-reopen@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Production Reopen Test"],
        cwd=repo,
        check=True,
    )
    output = repo / "inventory" / "result.json"
    output.parent.mkdir(parents=True)
    output.write_text('{"version":"base"}\n', encoding="utf-8")
    subprocess.run(["git", "add", "inventory/result.json"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)
    target_branch = "test/production-false-completion"
    subprocess.run(
        ["git", "switch", "-q", "-c", target_branch],
        cwd=repo,
        check=True,
    )
    target = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "switch", "-q", "-c", "implementation/candidate"],
        cwd=repo,
        check=True,
    )
    output.write_text('{"version":"candidate"}\n', encoding="utf-8")
    subprocess.run(["git", "add", "inventory/result.json"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "candidate"], cwd=repo, check=True)
    candidate = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "switch", "-q", target_branch],
        cwd=repo,
        check=True,
    )

    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        owner_session_id="session:production-false-completion",
        authority_mode="embedded_exclusive",
        task_source_kind="duckdb",
        require_real_execution=True,
    )
    try:
        daemon.materialize_population(
            {
                "repository_tree_id": "tree:production-false-completion",
                "tasks": [
                    {
                        "task_cid": "task:cid:004",
                        "task_id": "LGSWF-004",
                        "goal_cid": "goal:inventory",
                        "status": "ready",
                        "priority": "P0",
                        "ordinal": 4,
                        "title": "Inventory",
                        "outputs": [{"path": "inventory/result.json"}],
                        "validations": [
                            {
                                "argv": [
                                    "python3",
                                    "-m",
                                    "pytest",
                                    "focused.py",
                                ]
                            }
                        ],
                        "acceptance": [
                            {"criterion": "Focused validation passes"}
                        ],
                        "objective": "Produce the current authority inventory",
                        "completion": "auto",
                        "track": "analysis",
                        "read_scope": ["ipfs_accelerate_py/agent_supervisor"],
                        "write_scope": ["inventory/result.json"],
                        "completion_contract": "Focused validation passes",
                    }
                ],
            }
        )
        failed = daemon.claim_next()
        assert failed is not None
        record = daemon.task_source.get_task(failed.task_cid)
        assert record is not None and record.status == "in_progress"

        attempt_root = tmp_path / "lane-0-attempts"
        seed_bridge = DatabasePortalExecutionBridge(
            task_source=daemon.task_source,
            attempt_root=attempt_root,
            portal_factory=lambda _paths, _alias: None,
            repository_root=repo,
            task_header_prefix="## LGSWF-",
        )
        paths, _binding = seed_bridge._ensure_attempt_projection(failed, record)
        [projected_task] = parse_task_text(
            paths.task_projection.read_text(encoding="utf-8"),
            path=paths.task_projection,
            task_header_prefix="## LGSWF-",
        )
        repository_id = checkout_repository_id(repo)
        queue = MergeQueue(
            tmp_path / "merge-queue",
            target_repository_id=repository_id,
            target_branch=target_branch,
            require_target_binding=True,
        )
        request = queue.enqueue(
            branch_name="implementation/candidate",
            task_id=failed.task_alias,
            canonical_task_id=failed.task_cid,
            canonical_task_key=str(projected_task.canonical_task_key),
            commit_sha=candidate,
            metadata={
                "schema": "ipfs_accelerate_py/agent-supervisor/merge-candidate@3",
                "target_binding_schema": (
                    "ipfs_accelerate_py/agent-supervisor/"
                    "merge-target-binding@1"
                ),
                "target_repository_id": repository_id,
                "target_branch": target_branch,
                "implementation_commit": candidate,
                "todo_path": str(paths.task_projection),
                "state_path": str(paths.state),
                "strategy_path": str(paths.strategy),
                "events_path": str(paths.events),
                "repo_root": str(repo.absolute()),
                "task_header_prefix": "## LGSWF-",
                "task": asdict(projected_task),
                "completion_task_cids": {failed.task_alias: failed.task_cid},
                "changed_submodule_paths": [],
            },
        )
        claimed = queue.claim_pending_request(
            request.request_id,
            consumer_id="merge-train:production-false-shortcut-fixture",
        )
        assert claimed is not None
        queue.complete(claimed)
        completed = queue.get(request.request_id)
        assert completed is not None and completed.status == "completed"
        train = MergeTrain(repo, queue, target_branch=target_branch)
        train._write_receipt(
            completed.dedupe_key,
            {
                "already_merged": True,
                "canonical_task_id": completed.canonical_identity,
                "commit_sha": candidate,
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
                "merge_commit": target,
                "merged": False,
                "mutation_short_circuited": True,
                "reason": "declared_outputs_already_on_target",
                "request_id": request.request_id,
                "started_at": 1.0,
                "status": "already_merged",
                "target_branch": target_branch,
                "target_commit": target,
                "task_id": failed.task_alias,
            },
        )

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
        assert not hasattr(daemon, "_run_checkout_mutation_transaction")

        lock_observations: list[bool] = []
        original_reopen = queue.reopen_false_positive_completion
        expected_lock_path = board_scoped_checkout_mutation_lock_path(
            repo,
            "production-false-completion",
        )

        def observe_reopen(
            current: object,
            *,
            completion_receipt: object,
        ) -> object:
            lock_observations.append(expected_lock_path.is_file())
            return original_reopen(
                current,
                completion_receipt=completion_receipt,
            )

        monkeypatch.setattr(
            queue,
            "reopen_false_positive_completion",
            observe_reopen,
        )
        portal_calls: list[str] = []

        def portal_factory(portal_paths: object, alias: str) -> object:
            portal_calls.append(alias)
            return PortalImplementationDaemon(
                todo_path=portal_paths.task_projection,
                state_path=portal_paths.state,
                strategy_path=portal_paths.strategy,
                events_path=portal_paths.events,
                repo_root=repo,
                task_header_prefix="## LGSWF-",
                merge_queue=queue,
                merge_target_branch=target_branch,
            )

        bridge = DatabasePortalExecutionBridge(
            task_source=daemon.task_source,
            attempt_root=attempt_root,
            portal_factory=portal_factory,
            repository_root=repo,
            merge_queue=queue,
            merge_target_branch=target_branch,
            task_header_prefix="## LGSWF-",
        )
        result = bridge.recover_post_merge_declared_outputs(daemon)

        assert result is not None
        assert result["reason"] == "false_positive_completion_reopened"
        assert result["write_count"] == 1
        assert portal_calls == [failed.task_alias]
        assert lock_observations == [True]
        assert not expected_lock_path.exists()
        reopened = queue.get(request.request_id)
        assert reopened is not None and reopened.status == "pending"
        blocked = daemon.task_source.get_task(failed.task_cid)
        assert blocked is not None and blocked.status == "blocked"
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
        repository_root=repo.resolve(),
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
        task_payload = asdict(projected_task)
        task_payload["canonical_task_cid"] = "task:cid:004"
        task_payload["canonical_task_key"] = str(
            projected_task.canonical_task_key
        )
        task_metadata = dict(task_payload.get("metadata") or {})
        task_metadata["database task cid"] = "task:cid:004"
        task_metadata["canonical task cid"] = "task:cid:004"
        task_metadata["canonical task key"] = str(
            projected_task.canonical_task_key
        )
        task_metadata["database attempt id"] = str(binding["attempt_id"])
        task_metadata["database claim id"] = str(binding["claim_id"])
        task_metadata["projection authority"] = "false"
        task_payload["metadata"] = task_metadata
        task_payload["canonical_task_key"] = str(
            binding.get("canonical_task_key") or projected_task.canonical_task_key
        )
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
            "repo_root": str(repo.resolve()),
            "task_header_prefix": "## LGSWF-",
            "task": task_payload,
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
            canonical_task_key=str(
                binding.get("canonical_task_key")
                or projected_task.canonical_task_key
            ),
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
            repository_root=repo.resolve(),
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
