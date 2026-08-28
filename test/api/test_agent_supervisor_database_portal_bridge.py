"""Focused safety tests for database-authoritative Portal execution."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA,
    DatabasePortalBridgeDeferred,
    DatabasePortalBridgeError,
    DatabasePortalExecutionBridge,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DATASETS_AUTHORITATIVE_STATE_SCHEMA_REVISION,
    SEMANTIC_TRUTH_AUTHORITY_ENV,
    SEMANTIC_WRITER_POLICY_ENV,
    DatabaseImplementationAuthorityError,
    DatabaseImplementationDaemon,
    DatabaseTaskAttempt,
    PortalImplementationDaemon,
    parse_args,
    parse_task_text,
    task_declared_output_paths,
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
            "completion": "auto",
            "track": "analysis",
            "read_scope": ["ipfs_accelerate_py/agent_supervisor"],
            "write_scope": ["inventory/result.json"],
            "completion_contract": "Focused validation passes",
        },
    )


def _bridge_for_projection(tmp_path: Path) -> DatabasePortalExecutionBridge:
    return DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: object(),
    )


def _content_addressed_output(
    *,
    effect_id: str = "sha256:effect-001",
    declared_path: object = "inventory/result.json",
) -> dict[str, object]:
    return {
        "ordinal": 0,
        "path": effect_id,
        "effect": {
            "effect_id": effect_id,
            "declared_path": declared_path,
            "effect": "declared_output",
        },
    }


def test_projection_prefers_exact_nested_declared_output_path(
    tmp_path: Path,
) -> None:
    record = _record()
    record.outputs = (_content_addressed_output(),)

    projection = _bridge_for_projection(tmp_path)._render_projection(
        _attempt(), record
    )

    assert "- Outputs: inventory/result.json" in projection
    assert "- Outputs: sha256:effect-001" not in projection
    tasks = parse_task_text(
        projection,
        path=tmp_path / "task-projection.md",
        task_header_prefix="## LGSWF-",
    )
    assert len(tasks) == 1
    assert task_declared_output_paths(tasks[0]) == ("inventory/result.json",)


def test_projection_preserves_legacy_outer_output_path(tmp_path: Path) -> None:
    projection = _bridge_for_projection(tmp_path)._render_projection(
        _attempt(), _record()
    )

    assert "- Outputs: inventory/result.json" in projection


@pytest.mark.parametrize(
    "effect",
    (
        {
            "path": "outputs/LGSWF-006.json",
            "effect_id": "effect:LGSWF-006",
        },
        {
            "path": "outputs/LGSWF-006.json",
            "effect_id": "effect:LGSWF-006",
            "effect": "create",
        },
    ),
)
def test_projection_preserves_generic_nested_effect_id_output(
    tmp_path: Path,
    effect: dict[str, object],
) -> None:
    record = _record()
    record.outputs = (
        {
            "ordinal": 0,
            "path": "outputs/LGSWF-006.json",
            "effect": effect,
        },
    )

    projection = _bridge_for_projection(tmp_path)._render_projection(
        _attempt(), record
    )

    assert "- Outputs: outputs/LGSWF-006.json" in projection
    tasks = parse_task_text(
        projection,
        path=tmp_path / "task-projection.md",
        task_header_prefix="## LGSWF-",
    )
    assert len(tasks) == 1
    assert task_declared_output_paths(tasks[0]) == (
        "outputs/LGSWF-006.json",
    )


def test_projection_rejects_conflicting_legacy_outer_paths(tmp_path: Path) -> None:
    record = _record()
    record.outputs = (
        {"path": "inventory/result.json", "output": "inventory/other.json"},
    )

    with pytest.raises(DatabasePortalBridgeError, match="declarations conflict"):
        _bridge_for_projection(tmp_path)._render_projection(_attempt(), record)


@pytest.mark.parametrize(
    "output",
    [
        {
            **_content_addressed_output(),
            "output": "sha256:conflicting-outer-id",
        },
        {
            **_content_addressed_output(),
            "path": "sha256:conflicting-outer-id",
        },
        {
            **_content_addressed_output(),
            "effect": {
                "declared_path": "inventory/result.json",
                "effect": "declared_output",
            },
        },
        {
            **_content_addressed_output(),
            "effect": {
                "effect_id": "sha256:effect-001",
                "declared_path": "inventory/result.json",
                "effect": "declared_output",
                "authority": True,
            },
        },
        {
            **_content_addressed_output(),
            "effect": {
                "effect_id": "sha256:effect-001",
                "declared_path": "inventory/result.json",
                "effect": "write",
            },
        },
        {
            "effect_id": "sha256:effect-001",
            "declared_path": "inventory/result.json",
            "effect": "declared_output",
        },
        _content_addressed_output(declared_path=None),
        _content_addressed_output(declared_path="../inventory/result.json"),
        _content_addressed_output(declared_path="/inventory/result.json"),
        _content_addressed_output(declared_path="inventory\\result.json"),
        _content_addressed_output(declared_path="inventory//result.json"),
        _content_addressed_output(declared_path="inventory/result,other.json"),
        _content_addressed_output(declared_path="inventory/result.json\x85## SAWM-X"),
        _content_addressed_output(declared_path="inventory/result.json\u2028## SAWM-X"),
        _content_addressed_output(declared_path="inventory/result.json\u2029## SAWM-X"),
        _content_addressed_output(declared_path="none"),
        _content_addressed_output(declared_path="N/A"),
    ],
    ids=(
        "conflicting-outer-aliases",
        "effect-id-mismatch",
        "missing-effect-id",
        "open-effect-record",
        "wrong-effect-kind",
        "non-nested-declaration",
        "non-string-declared-path",
        "parent-traversal",
        "absolute-path",
        "backslash-path",
        "non-canonical-path",
        "projection-delimiter",
        "next-line-control",
        "unicode-line-separator",
        "unicode-paragraph-separator",
        "portal-none-sentinel",
        "portal-na-sentinel",
    ),
)
def test_projection_rejects_malformed_or_ambiguous_declared_output(
    tmp_path: Path,
    output: dict[str, object],
) -> None:
    record = _record()
    record.outputs = (output,)

    with pytest.raises(DatabasePortalBridgeError, match="declared-output"):
        _bridge_for_projection(tmp_path)._render_projection(_attempt(), record)


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
                "authority_mode": "legacy_markdown",
                "task_source_kind": "legacy-markdown",
                "failover_policy": "fail_closed",
                "explicit_legacy": True,
                "credential": "must-not-propagate",
            },
            sort_keys=True,
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


def test_bridge_honors_explicit_deferral_without_reason_keyword(
    tmp_path: Path,
) -> None:
    class DeferredPortal:
        def __init__(self) -> None:
            self.closed = False

        def run_once(self) -> dict[str, object]:
            return {
                "active_task_id": "",
                "implementation_result": {
                    "task_id": "LGSWF-004",
                    "returncode": 1,
                    "reason": "validation_project_dependency_preflight_failed",
                    "deferred": True,
                    "attempt_consumed": False,
                    "provider_call_allowed": False,
                },
            }

        def close_event_runtime(self) -> None:
            self.closed = True

    portals: list[DeferredPortal] = []

    def factory(_paths: object, _alias: str) -> DeferredPortal:
        portal = DeferredPortal()
        portals.append(portal)
        return portal

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=factory,
    )

    with pytest.raises(
        DatabasePortalBridgeDeferred,
        match="^validation_project_dependency_preflight_failed$",
    ):
        bridge.run_provider(_attempt())

    assert portals and portals[0].closed is True
    attempt_boards = list((tmp_path / "attempts").glob("*/task-projection.md"))
    assert len(attempt_boards) == 1
    projection = attempt_boards[0].read_text(encoding="utf-8")
    assert "- Status: ready" in projection
    assert "Projection authority: false" in projection


def _external_owner_recovery_result(*, owner: str) -> dict[str, object]:
    return {
        "blocked": True,
        "reason": "external_protected_checkout_recovery_required",
        "protected_checkout_recovery": {
            "required": True,
            "adopted": False,
            "blocked": True,
            "recovered": False,
            "reason": "external_protected_checkout_recovery_required",
            "protected_recovery_owner": owner,
            "lock_path": "/tmp/repository.lock",
        },
        "unchanged": True,
        "write_count": 0,
        "projection_delta": {},
        "implementation_result": None,
        "merge_reconciliation": [],
    }


def test_bridge_defers_exact_external_owner_recovery_without_settling(
    tmp_path: Path,
) -> None:
    class RecoveryBlockedPortal:
        def run_once(self) -> dict[str, object]:
            return _external_owner_recovery_result(
                owner="implementation_supervisor"
            )

        def close_event_runtime(self) -> None:
            return None

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: RecoveryBlockedPortal(),
    )

    with pytest.raises(
        DatabasePortalBridgeDeferred,
        match="^external_protected_checkout_recovery_required$",
    ):
        bridge.run_provider(_attempt())


def test_bridge_rejects_unbound_external_owner_recovery_as_terminal(
    tmp_path: Path,
) -> None:
    class UnboundRecoveryPortal:
        def run_once(self) -> dict[str, object]:
            return _external_owner_recovery_result(owner="")

        def close_event_runtime(self) -> None:
            return None

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: UnboundRecoveryPortal(),
    )

    with pytest.raises(DatabasePortalBridgeError) as captured:
        bridge.run_provider(_attempt())
    assert not isinstance(captured.value, DatabasePortalBridgeDeferred)
    assert str(captured.value) == (
        "external_protected_checkout_recovery_required"
    )


def test_external_owner_recovery_deferral_shape_is_closed() -> None:
    exact = _external_owner_recovery_result(
        owner="implementation_supervisor"
    )
    assert (
        DatabasePortalExecutionBridge._is_external_protected_recovery_deferral(
            exact
        )
        is True
    )

    variants: list[dict[str, object]] = []
    for field, value in (
        ("adopted", True),
        ("protected_recovery_owner", "implementation_daemon"),
        ("lock_path", ""),
    ):
        variant = json.loads(json.dumps(exact))
        variant["protected_checkout_recovery"][field] = value
        variants.append(variant)
    for field, value in (
        ("write_count", 1),
        ("projection_delta", {"changed": True}),
        ("merge_reconciliation", [{"status": "pending"}]),
    ):
        variant = json.loads(json.dumps(exact))
        variant[field] = value
        variants.append(variant)

    assert all(
        not DatabasePortalExecutionBridge._is_external_protected_recovery_deferral(
            variant
        )
        for variant in variants
    )


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
