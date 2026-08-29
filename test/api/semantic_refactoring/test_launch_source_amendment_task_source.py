"""Focused qualification for the Quack launch-source amendment read path."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    canonical_json_bytes,
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    TaskRecord,
    TaskSourceConflictError,
    TaskSourceIntegrityError,
    TaskSourceSnapshot,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.launch_source_amendment import (
    LAUNCH_SOURCE_FOREST_RECEIPT_SCHEMA,
    LaunchSourceAmendment,
    LaunchSourceAmendmentError,
    task_contract_set_cid,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
    DEFAULT_STATEMENT_TEMPLATES,
    QuackStateClient,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import (
    DETERMINISTIC_ONLY_EXECUTION_MODE,
    TaskExecutionRoutePolicy,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_database_task_source import (
    TypedDatabaseTaskSource,
    daemon_required_owner_operations,
)


def _canonical(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def _receipt_id(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _fixture() -> tuple[
    TaskRecord,
    TaskExecutionRoutePolicy,
    LaunchSourceAmendment,
    dict[str, Any],
    dict[str, Any],
]:
    bootstrap_head = "1" * 40
    bootstrap_tree = "2" * 40
    launch_head = "3" * 40
    launch_tree = "4" * 40
    plan_cid = content_identity({"plan": "SPAR-PLAN-R1"})
    task = TaskRecord(
        task_cid=content_identity({"task": "SPAR-001"}),
        task_alias="SPAR-001",
        goal_cid=content_identity({"goal": "SPAR-G011"}),
        plan_cid=plan_cid,
        objective_id="semantic-preserving-autonomous-remodularization-v1",
        ordinal=1,
        status="ready",
        revision=1,
        priority="critical",
        body={"description": "audit current authorities"},
    )
    snapshot = TaskSourceSnapshot(
        source_schema="test-launch-source@1",
        schema_version=1,
        plan_root_cid=plan_cid,
        repository_tree_id=bootstrap_tree,
        projection_cid=content_identity({"projection": "bootstrap"}),
        formal_plan_id=plan_cid,
        source_identity=content_identity({"source": "bootstrap"}),
        revision=1,
        event_cursor=1,
        goal_count=1,
        task_count=1,
        dependency_count=0,
        terminal=False,
        objective_count=1,
        plan_count=1,
    )
    policy = TaskExecutionRoutePolicy.seal(
        snapshot=snapshot,
        tasks=(task,),
        execution_modes={task.task_alias: DETERMINISTIC_ONLY_EXECUTION_MODE},
    )
    source_forest_body = {
        "source_head": launch_head,
        "nested_repositories": [],
        "cross_repository_writes": False,
    }
    source_forest = {
        **source_forest_body,
        "source_forest_root": _receipt_id(source_forest_body),
    }
    receipt_body = {
        "schema": LAUNCH_SOURCE_FOREST_RECEIPT_SCHEMA,
        "source_head": launch_head,
        "repository_tree": launch_tree,
        "source_forest_root": source_forest["source_forest_root"],
        "source_forest": source_forest,
    }
    forest_receipt = {**receipt_body, "receipt_id": _receipt_id(receipt_body)}
    amendment = LaunchSourceAmendment(
        board_namespace="semantic-preserving-autonomous-remodularization-v1",
        plan_alias="SPAR-PLAN-R1",
        bootstrap_receipt_id=content_identity({"bootstrap": "receipt"}),
        bootstrap_plan_root_cid=plan_cid,
        bootstrap_source_head=bootstrap_head,
        bootstrap_repository_tree_id=bootstrap_tree,
        launch_source_forest_receipt_id=forest_receipt["receipt_id"],
        launch_source_forest_root=source_forest["source_forest_root"],
        launch_source_forest_receipt=forest_receipt,
        launch_source_head=launch_head,
        launch_repository_tree_id=launch_tree,
        immutable_objectives_cid=content_identity({"immutable": "objectives"}),
        immutable_plan_cid=content_identity({"immutable": "plan"}),
        immutable_taskboard_cid=content_identity({"immutable": "taskboard"}),
        immutable_validator_cid=content_identity({"immutable": "validator"}),
        bootstrap_config_cid=content_identity({"config": "bootstrap"}),
        launch_config_cid=content_identity({"config": "launch"}),
        dependency_seal_cid=content_identity({"dependency": "seal"}),
        task_contract_set_cid=task_contract_set_cid((task,)),
        parent_plan_revision=1,
        amended_plan_revision=2,
    )
    plan_body = {
        "plan_cid": plan_cid,
        "repository_tree_id": bootstrap_tree,
        "source_head": bootstrap_head,
        "launch_source_amendment": amendment.to_dict(),
        "launch_source_amendment_id": amendment.amendment_id,
    }
    task_row = {
        "task_cid": task.task_cid,
        "task_alias": task.task_alias,
        "goal_cid": task.goal_cid,
        "plan_cid": task.plan_cid,
        "objective_id": task.objective_id,
        "ordinal": task.ordinal,
        "status": task.status,
        "revision": task.revision,
        "priority": task.priority,
        "identity_json": _canonical({"repository_tree_id": bootstrap_tree}),
        "body_json": _canonical(dict(task.body)),
        "dependencies_json": "[]",
        "outputs_json": "[]",
        "acceptance_json": "[]",
        "validations_json": "[]",
    }
    plan_row = {
        "plan_cid": plan_cid,
        "goal_cid": task.goal_cid,
        "plan_alias": amendment.plan_alias,
        "plan_status": "active",
        "plan_revision": 2,
        "plan_head_body_json": _canonical(plan_body),
        "revision_plan_cid": plan_cid,
        "revision_number": 2,
        "plan_revision_body_json": _canonical(plan_body),
        "revision_recorded_at": "2026-08-29T00:00:00Z",
    }
    return task, policy, amendment, task_row, plan_row


class _Client:
    def __init__(
        self,
        *,
        task_row: Mapping[str, Any],
        plan_row: Mapping[str, Any],
        changing_generation: bool = False,
    ) -> None:
        self.task_row = dict(task_row)
        self.plan_row = dict(plan_row)
        self.changing_generation = changing_generation
        self.generation_reads = 0
        self.plan_parameters: list[dict[str, Any]] = []

    def load_generation(self) -> SimpleNamespace:
        self.generation_reads += 1
        identity = (
            f"generation:{self.generation_reads}"
            if self.changing_generation
            else "generation:stable"
        )
        return SimpleNamespace(content_id=identity, revision=self.generation_reads)

    def execute(
        self,
        operation: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        if operation == "executor_control_snapshot":
            return (
                {
                    "objective_count": 1,
                    "goal_count": 1,
                    "plan_count": 1,
                    "task_count": 1,
                    "dependency_count": 0,
                    "event_watermark": 1,
                    "goals_json": "[]",
                    "plans_json": "[]",
                    "tasks_json": "[]",
                },
            )
        if operation == "executor_task_projection_page":
            return (self.task_row,)
        if operation == "executor_active_plan_revision_by_identity":
            supplied = dict(parameters or {})
            self.plan_parameters.append(supplied)
            assert supplied == {"plan_cid": self.plan_row["plan_cid"]}
            return (self.plan_row,)
        raise AssertionError(operation)


def _source(client: _Client, policy: TaskExecutionRoutePolicy) -> TypedDatabaseTaskSource:
    source = object.__new__(TypedDatabaseTaskSource)
    source._client = client  # type: ignore[attr-defined]
    source._closed = False  # type: ignore[attr-defined]
    source._execution_route_policy = policy  # type: ignore[attr-defined]
    source._asserted_launch_source_amendment = None  # type: ignore[attr-defined]
    return source


def test_active_plan_revision_operation_is_closed_and_granted() -> None:
    operation = "executor_active_plan_revision_by_identity"
    template = DEFAULT_STATEMENT_TEMPLATES[operation]

    assert template.parameter_names == ("plan_cid",)
    assert "INNER JOIN plan_revisions AS r" in template.sql
    assert "r.revision = p.revision" in template.sql
    assert "p.status = 'active'" in template.sql
    assert operation in daemon_required_owner_operations()


def test_launch_source_amendment_reads_exact_active_revision_and_contract_set() -> None:
    _task, policy, expected, task_row, plan_row = _fixture()
    client = _Client(task_row=task_row, plan_row=plan_row)

    observed = _source(client, policy).launch_source_amendment

    assert observed == expected
    assert client.plan_parameters == [{"plan_cid": policy.plan_root_cid}]


def test_launch_source_amendment_rejects_plan_head_revision_body_mismatch() -> None:
    _task, policy, _expected, task_row, plan_row = _fixture()
    plan_row["plan_revision_body_json"] = _canonical({"different": True})

    with pytest.raises(TaskSourceIntegrityError, match="head body differs"):
        _ = _source(_Client(task_row=task_row, plan_row=plan_row), policy).launch_source_amendment


def test_launch_source_amendment_rejects_task_contract_set_mismatch() -> None:
    _task, policy, expected, task_row, plan_row = _fixture()
    changed = replace(
        expected,
        task_contract_set_cid=content_identity({"wrong": "task-contract-set"}),
        amendment_id="",
    )
    body = json.loads(str(plan_row["plan_head_body_json"]))
    body["launch_source_amendment"] = changed.to_dict()
    body["launch_source_amendment_id"] = changed.amendment_id
    plan_row["plan_head_body_json"] = _canonical(body)
    plan_row["plan_revision_body_json"] = _canonical(body)

    with pytest.raises(TaskSourceIntegrityError, match="task contract set differs"):
        _ = _source(_Client(task_row=task_row, plan_row=plan_row), policy).launch_source_amendment


def test_launch_source_amendment_rejects_generation_drift() -> None:
    _task, policy, _expected, task_row, plan_row = _fixture()
    client = _Client(
        task_row=task_row,
        plan_row=plan_row,
        changing_generation=True,
    )

    with pytest.raises(TaskSourceConflictError, match="changed during bounded read"):
        _ = _source(client, policy).launch_source_amendment

    assert client.generation_reads == 8


def test_constructor_rejects_mismatched_cli_launch_source_assertion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _task, policy, authoritative, _task_row, _plan_row = _fixture()
    asserted = replace(
        authoritative,
        launch_config_cid=content_identity({"config": "substituted-cli"}),
        amendment_id="",
    )
    client = object.__new__(QuackStateClient)
    client._session = object()  # type: ignore[attr-defined]
    client._adapter = object()  # type: ignore[attr-defined]
    client._closed = False  # type: ignore[attr-defined]
    monkeypatch.setattr(
        TypedDatabaseTaskSource,
        "_validate_execution_route_policy_population",
        lambda _self: None,
    )
    monkeypatch.setattr(
        TypedDatabaseTaskSource,
        "launch_source_amendment",
        property(lambda _self: authoritative),
    )

    with pytest.raises(TaskSourceIntegrityError, match="asserted.*differs"):
        TypedDatabaseTaskSource(
            client,
            execution_route_policy=policy,
            launch_source_amendment=asserted.to_dict(),
            owns_client=False,
        )


def test_launch_source_amendment_schema_is_closed_and_rehashes() -> None:
    _task, _policy, amendment, _task_row, _plan_row = _fixture()
    unknown = amendment.to_dict()
    unknown["unexpected"] = True
    with pytest.raises(LaunchSourceAmendmentError, match="closed schema"):
        LaunchSourceAmendment.from_dict(unknown)

    duplicate = amendment.to_json()[:-1] + ',"schema":"duplicate"}'
    with pytest.raises(LaunchSourceAmendmentError, match="malformed or ambiguous"):
        LaunchSourceAmendment.from_json(duplicate)

    forged = amendment.to_dict()
    forged["launch_config_cid"] = content_identity({"config": "forged"})
    with pytest.raises(LaunchSourceAmendmentError, match="does not rehash"):
        LaunchSourceAmendment.from_dict(forged)


def test_launch_source_amendment_rejects_ambiguous_nested_json() -> None:
    _task, _policy, amendment, _task_row, _plan_row = _fixture()
    nonstring_key = amendment.to_dict()
    nonstring_key["launch_source_forest_receipt"]["source_forest"][1] = "bad"
    with pytest.raises(LaunchSourceAmendmentError, match="keys must be strings"):
        LaunchSourceAmendment.from_dict(nonstring_key)

    floating = amendment.to_dict()
    floating["launch_source_forest_receipt"]["source_forest"]["score"] = 1.0
    with pytest.raises(LaunchSourceAmendmentError, match="unsupported float"):
        LaunchSourceAmendment.from_dict(floating)


def test_launch_source_amendment_nested_values_are_immutable() -> None:
    _task, _policy, amendment, _task_row, _plan_row = _fixture()
    source_forest = amendment.launch_source_forest_receipt["source_forest"]

    with pytest.raises(TypeError):
        source_forest["source_head"] = "f" * 40
    with pytest.raises(AttributeError):
        source_forest["nested_repositories"].append("bad")


def test_task_contract_set_is_order_independent_and_rejects_duplicates() -> None:
    task, _policy, _amendment, _task_row, _plan_row = _fixture()
    sibling = replace(
        task,
        task_cid=content_identity({"task": "SPAR-002"}),
        task_alias="SPAR-002",
        ordinal=2,
    )
    assert task_contract_set_cid((task, sibling)) == task_contract_set_cid((sibling, task))
    with pytest.raises(LaunchSourceAmendmentError, match="empty or contains duplicate"):
        task_contract_set_cid(())
    with pytest.raises(LaunchSourceAmendmentError, match="empty or contains duplicate"):
        task_contract_set_cid((task, replace(sibling, task_alias=task.task_alias)))


def test_attempt_policy_composes_launch_and_historical_route() -> None:
    task, policy, amendment, _task_row, _plan_row = _fixture()
    binding = policy.binding_for_task(task).to_dict()

    first = amendment.attempt_policy_root(binding)
    advanced = replace(
        amendment,
        parent_plan_revision=2,
        amended_plan_revision=3,
        predecessor_amendment_id=amendment.amendment_id,
        amendment_id="",
    )

    assert first == amendment.attempt_policy_root(binding)
    assert advanced.same_launch_generation(amendment)
    assert advanced.attempt_policy_root(binding) != first


def test_unasserted_predecessor_board_does_not_acquire_amendment() -> None:
    _task, policy, _amendment, task_row, plan_row = _fixture()
    client = _Client(task_row=task_row, plan_row=plan_row)

    assert _source(client, policy).admitted_launch_source_amendment is None
    assert client.plan_parameters == []
