"""Offline-only EAAEF population projection for ``DatabaseTaskSource@1``.

This module translates one already compiled, fresh EAAEF population into the
public population shape consumed by :class:`DatabaseTaskSource`.  It neither
opens a database nor discovers an owner.  A caller may pass an already-open
task source to :func:`materialize_offline_eaaef_population`, but only while it
can attest that the exclusive owner is absent.

Historical status overlays are not an input.  The only admitted initial state
is 22 ``todo`` bootstrap tasks plus 94 ``blocked`` Plan-R2-held tasks.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

from .eaaef_reconciliation_lifecycle import (
    EAAEF_BOOTSTRAP_TASK_COUNT,
    EAAEF_GOAL_COUNT,
    EAAEF_GOAL_EDGE_COUNT,
    EAAEF_PLAN_R1_ALIAS,
    EAAEF_PLAN_R2_TASK_COUNT,
    EAAEF_TASK_COUNT,
    CompiledEAAEFPopulation,
    EAAEFReconciliationBlocked,
    EAAEFReconciliationIdentityError,
    _canonical_bytes,
    _cid,
    verify_compiled_eaaef_population_commitments,
)

EAAEF_OFFLINE_TASK_SOURCE_POPULATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-offline-task-source-population@1"
)
EAAEF_OFFLINE_TASK_SOURCE_MATERIALIZATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-offline-task-source-materialization@1"
)
DATABASE_TASK_SOURCE_INTERFACE: Final = "DatabaseTaskSource@1"

_EXPECTED_EXECUTION_CONTRACT_COUNTS: Final = {
    "task_dependencies": 270,
    "task_outputs": 415,
    "task_validations": 117,
    "task_acceptance": 116,
}
_TERMINAL_STATUSES: Final = frozenset(
    {"accepted", "cancelled", "complete", "completed", "done", "failed", "quarantined"}
)


@runtime_checkable
class OfflineDatabaseTaskSource(Protocol):
    """Narrow already-open sink used only before exclusive-owner birth."""

    INTERFACE: str

    def materialize(
        self,
        population: Mapping[str, Any],
        *,
        repository_tree_id: str = "",
        plan_root_cid: str = "",
    ) -> Mapping[str, Any]:
        """Materialize the translated population and return a typed receipt."""


def _plain(value: Any) -> Any:
    return json.loads(_canonical_bytes(value))


def _require_offline_inputs(
    *,
    owner_active: bool,
    historical_task_statuses: Mapping[str, str] | Sequence[Mapping[str, Any]] | None,
) -> None:
    if owner_active is not False:
        raise EAAEFReconciliationBlocked(
            "offline EAAEF population materialization requires the exclusive owner to be absent"
        )
    if historical_task_statuses is None:
        return
    if isinstance(historical_task_statuses, Mapping):
        imported = bool(historical_task_statuses)
    elif isinstance(historical_task_statuses, Sequence) and not isinstance(
        historical_task_statuses, (str, bytes, bytearray)
    ):
        imported = bool(historical_task_statuses)
    else:
        raise EAAEFReconciliationIdentityError("historical task status input is malformed")
    if imported:
        raise EAAEFReconciliationIdentityError(
            "historical task statuses cannot seed a fresh EAAEF population"
        )


def _translated_goals(
    population: CompiledEAAEFPopulation,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], str]:
    if len(population.goals) != EAAEF_GOAL_COUNT:
        raise EAAEFReconciliationIdentityError("compiled EAAEF goal count differs")
    goals: list[dict[str, Any]] = []
    goal_cids: set[str] = set()
    goal_aliases: set[str] = set()
    roots: list[str] = []
    for expected_ordinal, raw in enumerate(population.goals, start=1):
        if not isinstance(raw, Mapping):
            raise EAAEFReconciliationIdentityError("compiled EAAEF goal is not an object")
        source = dict(raw)
        body = source.get("body")
        identity = source.get("identity")
        if not isinstance(body, Mapping) or not isinstance(identity, Mapping):
            raise EAAEFReconciliationIdentityError("compiled EAAEF goal contract is absent")
        goal_cid = str(source.get("goal_cid") or "")
        goal_alias = str(source.get("goal_alias") or "")
        goal_spec = body.get("goal_spec")
        expected_cid = _cid(
            {
                "schema": "EAAEFFreshGoalIdentity@1",
                "goal_alias": goal_alias,
                "goal_spec": goal_spec,
                "board_cid": population.board_cid,
                "source_forest_root": population.source_forest_root,
            }
        )
        if (
            not goal_alias.startswith("EAAEF-G")
            or goal_cid != expected_cid
            or goal_cid in goal_cids
            or goal_alias in goal_aliases
            or source.get("ordinal") != expected_ordinal
            or source.get("status") != "open"
            or identity.get("goal_cid") != goal_cid
            or identity.get("goal_alias") != goal_alias
            or identity.get("board_cid") != population.board_cid
            or identity.get("source_forest_root") != population.source_forest_root
            or body.get("fresh_generation") is not True
            or body.get("historical_status_imported") is not False
        ):
            raise EAAEFReconciliationIdentityError(
                f"compiled EAAEF goal {goal_alias or expected_ordinal} differs"
            )
        parent_goal_cid = str(source.get("parent_goal_cid") or "")
        if not parent_goal_cid:
            roots.append(goal_alias)
        goal_cids.add(goal_cid)
        goal_aliases.add(goal_alias)
        row = _plain(body)
        row.update(
            {
                "goal_cid": goal_cid,
                "goal_alias": goal_alias,
                "title": str(source.get("title") or goal_alias),
                "objective_id": "objective:eaaef-root",
                "objective_alias": "ExternalAgentAutonomousExecutionFabric",
                "parent_goal_cid": parent_goal_cid,
                "ordinal": expected_ordinal,
                "status": "open",
                "compiled_identity": _plain(identity),
            }
        )
        goals.append(row)
    if roots != ["EAAEF-G000"]:
        raise EAAEFReconciliationIdentityError("compiled EAAEF root goal differs")
    if any(
        str(item.get("parent_goal_cid") or "") not in goal_cids
        for item in goals
        if item.get("parent_goal_cid")
    ):
        raise EAAEFReconciliationIdentityError("compiled EAAEF parent goal is absent")

    if len(population.goal_edges) != EAAEF_GOAL_EDGE_COUNT:
        raise EAAEFReconciliationIdentityError("compiled EAAEF goal edge count differs")
    edges: list[dict[str, str]] = []
    seen_edges: set[tuple[str, str, str]] = set()
    for raw in population.goal_edges:
        if not isinstance(raw, Mapping):
            raise EAAEFReconciliationIdentityError("compiled EAAEF goal edge is malformed")
        edge = {
            "parent_goal_cid": str(raw.get("parent_goal_cid") or ""),
            "child_goal_cid": str(raw.get("child_goal_cid") or ""),
            "edge_kind": str(raw.get("edge_kind") or ""),
        }
        identity = (edge["parent_goal_cid"], edge["child_goal_cid"], edge["edge_kind"])
        if (
            edge["parent_goal_cid"] not in goal_cids
            or edge["child_goal_cid"] not in goal_cids
            or edge["parent_goal_cid"] == edge["child_goal_cid"]
            or edge["edge_kind"] != "requires"
            or identity in seen_edges
        ):
            raise EAAEFReconciliationIdentityError("compiled EAAEF goal edge differs")
        seen_edges.add(identity)
        edges.append(edge)

    goal_population_cid = _cid(
        {
            "schema": "EAAEFFreshGoalPopulation@1",
            "goals": list(population.goals),
            "goal_edges": list(population.goal_edges),
            "plan_r1": population.plan_r1,
            "source_forest_root": population.source_forest_root,
        }
    )
    if population.goal_population_cid != goal_population_cid:
        raise EAAEFReconciliationIdentityError("compiled EAAEF goal population CID differs")
    root_cid = next(item["goal_cid"] for item in goals if item["goal_alias"] == "EAAEF-G000")
    return goals, edges, str(root_cid)


def _translated_plan(
    population: CompiledEAAEFPopulation,
    *,
    root_goal_cid: str,
) -> dict[str, Any]:
    if not isinstance(population.plan_r1, Mapping):
        raise EAAEFReconciliationIdentityError("compiled EAAEF R1 plan is absent")
    source = dict(population.plan_r1)
    body = source.get("body")
    if not isinstance(body, Mapping):
        raise EAAEFReconciliationIdentityError("compiled EAAEF R1 plan body is absent")
    expected_plan_cid = _cid(
        {
            "schema": "EAAEFFreshPlanIdentity@1",
            "plan_alias": EAAEF_PLAN_R1_ALIAS,
            "board_cid": population.board_cid,
            "source_forest_root": population.source_forest_root,
        }
    )
    if (
        population.plan_r1_cid != expected_plan_cid
        or source.get("plan_cid") != expected_plan_cid
        or source.get("plan_alias") != EAAEF_PLAN_R1_ALIAS
        or source.get("goal_cid") != root_goal_cid
        or source.get("status") != "active"
        or source.get("revision") != 1
        or source.get("semantic_root_cid") != population.source_forest_root
        or body.get("terminal_statuses_imported") != 0
        or body.get("fresh_generation") is not True
    ):
        raise EAAEFReconciliationIdentityError("compiled EAAEF R1 plan differs")
    row = _plain(body)
    row.update(
        {
            "plan_cid": expected_plan_cid,
            "plan_alias": EAAEF_PLAN_R1_ALIAS,
            "goal_cid": root_goal_cid,
            "status": "active",
            "revision": 1,
            "semantic_root_cid": population.source_forest_root,
        }
    )
    return row


def _translated_tasks(population: CompiledEAAEFPopulation) -> list[dict[str, Any]]:
    bootstrap = list(population.bootstrap_tasks)
    held = list(population.plan_r2_tasks)
    if len(bootstrap) != EAAEF_BOOTSTRAP_TASK_COUNT or len(held) != EAAEF_PLAN_R2_TASK_COUNT:
        raise EAAEFReconciliationIdentityError("compiled EAAEF 22+94 split differs")
    if population.task_count != EAAEF_TASK_COUNT:
        raise EAAEFReconciliationIdentityError("compiled EAAEF task count differs")
    status_counts = Counter(str(item.get("status") or "") for item in (*bootstrap, *held))
    if status_counts != {"todo": EAAEF_BOOTSTRAP_TASK_COUNT, "blocked": EAAEF_PLAN_R2_TASK_COUNT}:
        raise EAAEFReconciliationIdentityError("compiled EAAEF initial status split differs")
    if any(item.get("status") != "todo" for item in bootstrap) or any(
        item.get("status") != "blocked" for item in held
    ):
        raise EAAEFReconciliationIdentityError("compiled EAAEF population partitions differ")

    task_by_cid: dict[str, Mapping[str, Any]] = {}
    aliases: set[str] = set()
    for raw in (*bootstrap, *held):
        if not isinstance(raw, Mapping):
            raise EAAEFReconciliationIdentityError("compiled EAAEF task is not an object")
        task_cid = str(raw.get("task_cid") or "")
        task_alias = str(raw.get("task_alias") or "")
        body = raw.get("body")
        identity = raw.get("identity")
        if (
            not task_cid
            or task_cid in task_by_cid
            or not task_alias.startswith("EAAEF-")
            or task_alias in aliases
            or not isinstance(body, Mapping)
            or not isinstance(identity, Mapping)
            or raw.get("plan_cid") != population.plan_r1_cid
            or raw.get("revision") != 1
            or str(raw.get("status") or "").casefold() in _TERMINAL_STATUSES
            or body.get("fresh_generation") is not True
            or body.get("historical_status_imported") is not False
            or body.get("accepted_source_forest_root") != population.source_forest_root
            or identity.get("task_cid") != task_cid
            or identity.get("task_alias") != task_alias
            or identity.get("board_cid") != population.board_cid
            or identity.get("source_forest_root") != population.source_forest_root
        ):
            raise EAAEFReconciliationIdentityError(
                f"compiled EAAEF task {task_alias or task_cid or 'unknown'} differs"
            )
        task_by_cid[task_cid] = raw
        aliases.add(task_alias)

    dependencies_by_task: defaultdict[str, list[str]] = defaultdict(list)
    seen_dependencies: set[tuple[str, str, str]] = set()
    if len(population.dependencies) != _EXPECTED_EXECUTION_CONTRACT_COUNTS["task_dependencies"]:
        raise EAAEFReconciliationIdentityError("compiled EAAEF task dependency count differs")
    for raw in population.dependencies:
        if not isinstance(raw, Mapping):
            raise EAAEFReconciliationIdentityError("compiled EAAEF task dependency is malformed")
        task_cid = str(raw.get("task_cid") or "")
        dependency_cid = str(raw.get("dependency_task_cid") or "")
        kind = str(raw.get("kind") or "")
        identity = (task_cid, dependency_cid, kind)
        if (
            task_cid not in task_by_cid
            or dependency_cid not in task_by_cid
            or task_cid == dependency_cid
            or kind != "requires"
            or identity in seen_dependencies
        ):
            raise EAAEFReconciliationIdentityError("compiled EAAEF task dependency differs")
        seen_dependencies.add(identity)
        dependencies_by_task[task_cid].append(dependency_cid)

    rows: list[dict[str, Any]] = []
    for raw in (*bootstrap, *held):
        task_cid = str(raw["task_cid"])
        task_alias = str(raw["task_alias"])
        body = dict(raw["body"])
        declared_dependencies = sorted(str(item) for item in body.get("dependency_task_cids") or ())
        actual_dependencies = sorted(dependencies_by_task[task_cid])
        if declared_dependencies != actual_dependencies:
            raise EAAEFReconciliationIdentityError(
                f"compiled EAAEF task {task_alias} dependency projection differs"
            )
        output_paths = body.get("outputs")
        validations = body.get("validations")
        acceptance = body.get("acceptance")
        if (
            not isinstance(output_paths, list)
            or not output_paths
            or any(not isinstance(item, str) or not item for item in output_paths)
            or not isinstance(validations, list)
            or not validations
            or not isinstance(acceptance, list)
            or len(acceptance) != 1
            or not isinstance(acceptance[0], str)
            or not acceptance[0]
        ):
            raise EAAEFReconciliationIdentityError(
                f"compiled EAAEF task {task_alias} execution contract differs"
            )
        row = _plain(body)
        row.update(
            {
                "task_cid": task_cid,
                "task_alias": task_alias,
                "task_id": task_alias,
                "goal_cid": str(raw["goal_cid"]),
                "plan_cid": population.plan_r1_cid,
                "objective_id": "objective:eaaef-root",
                "ordinal": int(raw["ordinal"]),
                "status": str(raw["status"]),
                "priority": str(raw["priority"]),
                "compiled_revision": 1,
                "compiled_identity": _plain(raw["identity"]),
                "execution_contract_cid": str(raw["execution_contract_cid"]),
                "depends_on": actual_dependencies,
                "outputs": [
                    {
                        "path": path,
                        "effect_id": path,
                        "source_forest_root": population.source_forest_root,
                    }
                    for path in output_paths
                ],
                "validations": _plain(validations),
                "acceptance": list(acceptance),
            }
        )
        rows.append(row)

    observed_counts = {
        "task_dependencies": sum(len(item["depends_on"]) for item in rows),
        "task_outputs": sum(len(item["outputs"]) for item in rows),
        "task_validations": sum(len(item["validations"]) for item in rows),
        "task_acceptance": sum(len(item["acceptance"]) for item in rows),
    }
    if (
        population.execution_contract_counts != _EXPECTED_EXECUTION_CONTRACT_COUNTS
        or observed_counts != _EXPECTED_EXECUTION_CONTRACT_COUNTS
    ):
        raise EAAEFReconciliationIdentityError(
            "compiled EAAEF execution-contract population differs"
        )
    return rows


def translate_compiled_eaaef_population(
    population: CompiledEAAEFPopulation,
    *,
    current_board: Mapping[str, Any],
    current_forest: Mapping[str, Any],
    repo_root: str | Path | None = None,
    owner_active: bool,
    historical_task_statuses: Mapping[str, str]
    | Sequence[Mapping[str, Any]]
    | None = None,
) -> dict[str, Any]:
    """Build the exact fresh ``DatabaseTaskSource.materialize`` population."""

    _require_offline_inputs(
        owner_active=owner_active,
        historical_task_statuses=historical_task_statuses,
    )
    population = verify_compiled_eaaef_population_commitments(
        population,
        current_board=current_board,
        current_forest=current_forest,
        repo_root=repo_root,
    )
    goals, goal_edges, root_goal_cid = _translated_goals(population)
    plan = _translated_plan(population, root_goal_cid=root_goal_cid)
    tasks = _translated_tasks(population)
    projection = {
        "schema": EAAEF_OFFLINE_TASK_SOURCE_POPULATION_SCHEMA,
        "task_source_interface": DATABASE_TASK_SOURCE_INTERFACE,
        "board_namespace": "external-agent-autonomous-execution-fabric-v1",
        "board_cid": population.board_cid,
        "source_head": population.source_head,
        "source_tree": population.source_tree,
        "source_forest_root": population.source_forest_root,
        "repository_tree_id": population.source_tree,
        "plan_root_cid": population.plan_r1_cid,
        "population_cid": population.population_cid,
        "goal_population_cid": population.goal_population_cid,
        "execution_contract_population_cid": population.execution_contract_population_cid,
        "bootstrap_population_cid": population.bootstrap_population_cid,
        "held_plan_r2_population_cid": population.plan_r2_population_cid,
        "bootstrap_task_count": EAAEF_BOOTSTRAP_TASK_COUNT,
        "held_task_count": EAAEF_PLAN_R2_TASK_COUNT,
        "task_count": EAAEF_TASK_COUNT,
        "goal_count": EAAEF_GOAL_COUNT,
        "goal_edge_count": EAAEF_GOAL_EDGE_COUNT,
        "plan_count": 1,
        "task_status_counts": {
            "blocked": EAAEF_PLAN_R2_TASK_COUNT,
            "todo": EAAEF_BOOTSTRAP_TASK_COUNT,
        },
        "execution_contract_counts": dict(_EXPECTED_EXECUTION_CONTRACT_COUNTS),
        "terminal_statuses_imported": 0,
        "owner_absent_required": True,
        "provider_launch_allowed": False,
        "goals": goals,
        "goal_edges": goal_edges,
        "plans": [plan],
        "tasks": tasks,
    }
    projection["projection_cid"] = _cid(projection)
    return projection


def verify_translated_eaaef_population(
    translated: Mapping[str, Any],
    *,
    population: CompiledEAAEFPopulation,
    current_board: Mapping[str, Any],
    current_forest: Mapping[str, Any],
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify a caller-held translation against a fresh deterministic rebuild."""

    expected = translate_compiled_eaaef_population(
        population,
        current_board=current_board,
        current_forest=current_forest,
        repo_root=repo_root,
        owner_active=False,
        historical_task_statuses=None,
    )
    if _canonical_bytes(translated) != _canonical_bytes(expected):
        raise EAAEFReconciliationIdentityError(
            "offline EAAEF task-source population differs from its fresh compilation"
        )
    return expected


def materialize_offline_eaaef_population(
    task_source: OfflineDatabaseTaskSource,
    population: CompiledEAAEFPopulation,
    *,
    current_board: Mapping[str, Any],
    current_forest: Mapping[str, Any],
    repo_root: str | Path | None = None,
    owner_active: bool,
    historical_task_statuses: Mapping[str, str]
    | Sequence[Mapping[str, Any]]
    | None = None,
) -> dict[str, Any]:
    """Apply the fresh projection through an already-open offline task source.

    This is deliberately not the production reconciliation-owner adapter.  It
    proves only the bounded offline population write and never starts an owner,
    supervisor, or provider.
    """

    _require_offline_inputs(
        owner_active=owner_active,
        historical_task_statuses=historical_task_statuses,
    )
    if (
        not isinstance(task_source, OfflineDatabaseTaskSource)
        or getattr(task_source, "INTERFACE", "") != DATABASE_TASK_SOURCE_INTERFACE
    ):
        raise EAAEFReconciliationBlocked(
            "offline EAAEF materialization requires one already-open DatabaseTaskSource@1"
        )
    translated = translate_compiled_eaaef_population(
        population,
        current_board=current_board,
        current_forest=current_forest,
        repo_root=repo_root,
        owner_active=False,
        historical_task_statuses=None,
    )
    raw_receipt = task_source.materialize(
        translated,
        repository_tree_id=population.source_tree,
        plan_root_cid=population.plan_r1_cid,
    )
    if not isinstance(raw_receipt, Mapping):
        raise EAAEFReconciliationIdentityError("offline task-source receipt is malformed")
    task_cids = [str(item["task_cid"]) for item in translated["tasks"]]
    expected_receipt = {
        "schema": "ipfs_accelerate_py/agent-supervisor/database-task-source@1",
        "plan_root_cid": population.plan_r1_cid,
        "repository_tree_id": population.source_tree,
        "task_count": EAAEF_TASK_COUNT,
        "goal_count": EAAEF_GOAL_COUNT,
        "goal_edge_count": EAAEF_GOAL_EDGE_COUNT,
        "plan_count": 1,
        "task_cids": task_cids,
    }
    mismatched = sorted(
        field_name
        for field_name, expected_value in expected_receipt.items()
        if raw_receipt.get(field_name) != expected_value
    )
    if mismatched:
        raise EAAEFReconciliationIdentityError(
            "offline task-source receipt differs: " + ", ".join(mismatched)
        )
    receipt = {
        "schema": EAAEF_OFFLINE_TASK_SOURCE_MATERIALIZATION_SCHEMA,
        "task_source_interface": DATABASE_TASK_SOURCE_INTERFACE,
        "source_forest_root": population.source_forest_root,
        "population_cid": population.population_cid,
        "execution_contract_population_cid": population.execution_contract_population_cid,
        "translation_cid": translated["projection_cid"],
        "task_count": EAAEF_TASK_COUNT,
        "goal_count": EAAEF_GOAL_COUNT,
        "goal_edge_count": EAAEF_GOAL_EDGE_COUNT,
        "plan_count": 1,
        "task_status_counts": translated["task_status_counts"],
        "execution_contract_counts": translated["execution_contract_counts"],
        "terminal_statuses_imported": 0,
        "owner_absent_during_materialization": True,
        "owner_started": False,
        "provider_process_started": False,
        "task_source_receipt": _plain(dict(raw_receipt)),
        "qualification_status": "offline_population_only",
        "production_blockers": [
            "exclusive_typed_owner_adapter_not_bound",
            "owner_birth_not_observed",
            "signed_plan_r2_not_applied",
            "supervisor_not_launched",
        ],
    }
    receipt["receipt_cid"] = _cid(receipt)
    return receipt


__all__ = [
    "DATABASE_TASK_SOURCE_INTERFACE",
    "EAAEF_GOAL_EDGE_COUNT",
    "EAAEF_OFFLINE_TASK_SOURCE_MATERIALIZATION_SCHEMA",
    "EAAEF_OFFLINE_TASK_SOURCE_POPULATION_SCHEMA",
    "OfflineDatabaseTaskSource",
    "materialize_offline_eaaef_population",
    "translate_compiled_eaaef_population",
    "verify_translated_eaaef_population",
]
