"""Focused safety tests for database-authoritative Portal execution."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ContextCompilationReceipt,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import (
    ContextBudgetResolution,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DATABASE_PORTAL_CONSUMED_NO_PROGRESS_SCHEMA,
    DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA,
    DATABASE_PORTAL_RETRY_DEFERRAL_SCHEMA,
    DatabasePortalBridgeConsumedNoProgressError,
    DatabasePortalBridgeDeferred,
    DatabasePortalBridgeError,
    DatabasePortalExecutionBridge,
    database_portal_consumed_no_progress_fingerprint,
    database_portal_task_contract_digest,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DATASETS_AUTHORITATIVE_STATE_SCHEMA_REVISION,
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


def _attempt() -> DatabaseTaskAttempt:
    return DatabaseTaskAttempt(
        attempt_id="attempt:001",
        claim_id="claim:001",
        task_cid="task:cid:004",
        task_alias="LGSWF-004",
        attempt_number=1,
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
                    "canonical_task_key": "task/v1/current-authority-inventory",
                    "canonical_task_cid": "task:cid:004",
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
            event = json.loads(self.paths.events.read_text(encoding="utf-8"))
            event["canonical_task_cid"] = "task:cid:other"
            self.paths.events.write_text(
                json.dumps(event) + "\n",
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
