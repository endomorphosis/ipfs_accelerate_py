"""Admit worldwide legal search goals and tasks through the native task source.

``goals.objectives.md`` and ``tasks.todo.md`` are bootstrap projections. Live
identities, dependencies, readiness and status live only in an isolated
``DatabaseTaskSource`` namespace. Typed goal contracts are derived from the
objective heap and linted before any database file is created. The root's ten
children are an explicit breadth admission: the native default of eight is
rejected rather than inherited.
"""
from __future__ import annotations

import fcntl
import inspect
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..objectives.adaptive_goal_refiner import GoalDebtKind, GoalQualityRecord
from ..objectives.goal_quality import lint_objective_typed_goals
from ..objectives.objective_tracker import build_objective_goal_quality_report
from .worldwide_legal_search_goals import build_typed_goals
from .worldwide_legal_search_runtime import build_population, identity

ADMISSION_SCHEMA = "worldwide-legal-search.native-admission@1"
ROOT_GOAL_ID = "WWL-G000"
ADMITTED_ROOT_BREADTH = 10
NATIVE_DEFAULT_BREADTH = 8
_NAMESPACE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_TASK_HEADER = re.compile(r"^##\s+(WWL-[0-9]+)\b")
_STATUS_LINE = re.compile(r"^-\s*Status:\s*(\S+)")
_PAGE_LIMIT = 1000
_COMPLETED_MARKDOWN = {"complete", "completed", "done"}


class WorldwideAdmissionError(ValueError):
    """The campaign cannot be admitted without rewriting native state."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _native_default_breadth() -> int:
    """Bind the campaign override to the real native default, not a comment."""
    default = inspect.signature(build_objective_goal_quality_report).parameters["default_max_breadth"].default
    if default != NATIVE_DEFAULT_BREADTH:
        raise WorldwideAdmissionError("native goal-quality default breadth changed and must be reviewed before admission")
    if ADMITTED_ROOT_BREADTH <= NATIVE_DEFAULT_BREADTH:
        raise WorldwideAdmissionError("admitted root breadth must exceed the native default")
    return int(default)


def _direct_children(plan: Mapping[str, Any]) -> tuple[str, ...]:
    children: list[str] = []
    for goal in plan.get("goals") or ():
        if not isinstance(goal, Mapping):
            raise WorldwideAdmissionError("plan goals must be objects")
        parents = goal.get("parent_ids") or ()
        if isinstance(parents, str) or not isinstance(parents, (list, tuple)):
            raise WorldwideAdmissionError("plan goal parents must be a list")
        if ROOT_GOAL_ID in parents:
            children.append(str(goal.get("id") or ""))
    return tuple(children)


def _ignored_markdown_statuses(task_markdown: str | None) -> dict[str, str]:
    """Record bootstrap statuses without letting them become native state."""
    if task_markdown is None:
        return {}
    if not isinstance(task_markdown, str):
        raise WorldwideAdmissionError("task markdown bootstrap must be text")
    if len(task_markdown.encode("utf-8")) > 4_194_304:
        raise WorldwideAdmissionError("task markdown bootstrap exceeds its size bound")
    ignored: dict[str, str] = {}
    task_id = ""
    for line in task_markdown.splitlines():
        header = _TASK_HEADER.match(line)
        if header:
            task_id = header.group(1)
            continue
        status = _STATUS_LINE.match(line)
        if task_id and status:
            ignored[task_id] = status.group(1)
            task_id = ""
    return dict(sorted(ignored.items()))


def _breadth_record(root: Any, *, max_breadth: int) -> GoalQualityRecord:
    return GoalQualityRecord(
        goal_id=root.goal_id,
        outcome=root.outcome,
        scope_ids=root.scope.include,
        assumption_ids=root.assumptions,
        non_goals=root.non_goals,
        acceptance_criteria=tuple(item.statement for item in root.acceptance_criteria),
        evidence_producer_ids=tuple(item.producer_id for item in root.evidence_producers),
        validation_ids=tuple(item.rule_id for item in root.validation_rules),
        freshness_horizon_seconds=root.freshness.max_age_seconds,
        resource_envelope={
            "max_wall_seconds": root.resources.max_wall_seconds,
            "max_tokens": root.resources.max_tokens,
            "max_artifacts": root.resources.max_artifacts,
            "max_parallelism": root.resources.max_parallelism,
            "max_scope_items": root.resources.max_scope_items,
        },
        refinement_budget={
            "max_rounds": root.refinement_budget.max_rounds,
            "max_children": root.refinement_budget.max_children,
            "max_depth": root.refinement_budget.max_depth,
            "max_debt_items": root.refinement_budget.max_debt_items,
            "max_tokens": root.refinement_budget.max_tokens,
        },
        unsupported_semantics=tuple(item.statement for item in root.unsupported_semantics),
        breadth=len(root.scope.include),
        max_breadth=max_breadth,
    )


def _require_root_breadth(plan: Mapping[str, Any], document: Any) -> tuple[str, ...]:
    native_default = _native_default_breadth()
    children = _direct_children(plan)
    if len(children) != ADMITTED_ROOT_BREADTH:
        raise WorldwideAdmissionError(
            f"root breadth is {len(children)}; admission requires the explicit ten-child breadth, not the native default of {native_default}"
        )
    root = document.goal_map().get(ROOT_GOAL_ID)
    if root is None:
        raise WorldwideAdmissionError("typed root goal is missing")
    if tuple(sorted(children)) != tuple(root.scope.include):
        raise WorldwideAdmissionError("typed root scope does not name the ten direct children")
    if root.refinement_budget.max_children != ADMITTED_ROOT_BREADTH:
        raise WorldwideAdmissionError("typed root refinement breadth does not explicitly admit ten children")
    if root.resources.max_scope_items < ADMITTED_ROOT_BREADTH:
        raise WorldwideAdmissionError("typed root resource scope is narrower than the admitted breadth")
    # The adaptive quality record is the native default of eight. Passing that
    # default must be debt; only the explicit ten-child bound is admissible.
    if GoalDebtKind.EXCESSIVE_BREADTH not in _breadth_record(root, max_breadth=native_default).debt:
        raise WorldwideAdmissionError("native default breadth unexpectedly accepted the ten-child root")
    if GoalDebtKind.EXCESSIVE_BREADTH in _breadth_record(root, max_breadth=ADMITTED_ROOT_BREADTH).debt:
        raise WorldwideAdmissionError("explicit ten-child breadth was rejected by goal quality")
    return tuple(sorted(children))


def _linted_goals(plan: Mapping[str, Any], goal_markdown: str, *, config_path: Path | str, python_executable: str):
    if not isinstance(goal_markdown, str) or not goal_markdown.strip():
        raise WorldwideAdmissionError("objective heap markdown is required as bootstrap input")
    document = build_typed_goals(plan, goal_markdown, config_path=config_path, python_executable=python_executable)
    reports = tuple(lint_objective_typed_goals(document))
    rejected = [report.goal_id for report in reports if report.debt or not report.accepted]
    if rejected or len(reports) != len(document.goals):
        raise WorldwideAdmissionError("native goal-quality lint rejected admission: " + ", ".join(rejected))
    return document, reports


def _population(plan: Mapping[str, Any], source_forest: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    repositories = {str(task.get("repository") or "") for task in plan.get("tasks") or ()}
    missing = sorted(repository for repository in repositories if repository not in source_forest)
    if missing:
        raise WorldwideAdmissionError("source forest is missing owning repositories: " + ", ".join(missing))
    try:
        population = build_population(plan, source_forest=source_forest)
        definition_id = identity(plan)
        forest = json.loads(_canonical(source_forest))
    except (TypeError, ValueError, KeyError) as exc:
        raise WorldwideAdmissionError("campaign plan cannot form stable native identities") from exc
    if len(population.get("plans") or ()) != 1:
        raise WorldwideAdmissionError("native population must have one plan head")
    # build_population always appends evidence_output. The native output key
    # is (task_cid, path), so a path listed twice is one effect, not two rows.
    for task in population["tasks"]:
        collapsed: list[dict[str, Any]] = []
        seen: dict[str, str] = {}
        for item in task["outputs"]:
            path = str(item.get("path") or "")
            effect_id = str(item.get("effect_id") or "")
            previous = seen.get(path)
            if previous is None:
                seen[path] = effect_id
                collapsed.append(dict(item))
                continue
            if previous != effect_id:
                raise WorldwideAdmissionError(
                    f"task {task['task_alias']} repeats output path {path} with a different effect"
                )
        task["outputs"] = collapsed
    population["plans"][0] = {
        **population["plans"][0],
        "plan_definition_id": definition_id,
        "board_namespace": str(plan["namespace"]),
        "admitted_source_forest": forest,
        "admitted_plan_root_cid": population["plan_root_cid"],
        "repository_tree_id": population["repository_tree_id"],
        "markdown_bootstrap_only": True,
    }
    return population, definition_id


def _load_tasks(source: Any) -> tuple[Any, ...]:
    page = source.list_tasks(limit=_PAGE_LIMIT)
    if page.next_cursor:
        raise WorldwideAdmissionError("native task population exceeds the bounded page")
    return tuple(page.tasks)


def _require_store_shape(source: Any, plan: Mapping[str, Any], tasks: tuple[Any, ...]) -> Any:
    snapshot = source.snapshot()
    if snapshot.task_count != len(plan["tasks"]) or len(tasks) != len(plan["tasks"]):
        raise WorldwideAdmissionError("native task count does not match the admitted plan")
    if snapshot.goal_count != len(plan["goals"]):
        raise WorldwideAdmissionError("native goal count does not match the admitted plan")
    expected_dependencies = sum(len(task.get("depends_on") or ()) for task in plan["tasks"])
    if snapshot.dependency_count != expected_dependencies:
        raise WorldwideAdmissionError("native dependency count does not match the admitted plan")
    edges = source.list_goal_edges(limit=_PAGE_LIMIT)
    expected_edges = sum(len(goal.get("parent_ids") or ()) for goal in plan["goals"])
    if len(edges) != expected_edges:
        raise WorldwideAdmissionError("native goal edges do not match the admitted plan")
    alias_set = {task.task_alias for task in tasks}
    if alias_set != {str(task["id"]) for task in plan["tasks"]}:
        raise WorldwideAdmissionError("native task aliases do not match the admitted plan")
    namespace = str(plan["namespace"])
    objective_id = "objective:" + namespace
    known_cids = {task.task_cid for task in tasks}
    for task in tasks:
        if task.objective_id != objective_id or task.body.get("board_namespace") != namespace:
            raise WorldwideAdmissionError("native task escaped its isolated namespace")
        if tuple(task.dependencies) and not set(task.dependencies) <= known_cids:
            raise WorldwideAdmissionError("native dependency is outside the admitted task set")
        source_task = task.body.get("source_plan_task")
        if not isinstance(source_task, Mapping) or str(source_task.get("id") or "") != task.task_alias:
            raise WorldwideAdmissionError("native task is missing its stable plan identity")
        goal = source.get_goal(task.goal_cid)
        if goal is None or str(goal.get("goal_alias") or "") != str(source_task.get("subgoal_id") or ""):
            raise WorldwideAdmissionError("native task is not bound to its subgoal identity")
    return snapshot


def _dependency_frontier(values: Any) -> tuple[str, ...]:
    """Native reads order dependency CIDs by identifier, not by plan order."""
    if isinstance(values, str) or not isinstance(values, (list, tuple)):
        raise WorldwideAdmissionError("native dependencies must be a list of task identities")
    return tuple(sorted(str(item) for item in values))


def _identity_mismatch(tasks: tuple[Any, ...], population: Mapping[str, Any]) -> str:
    expected = {str(item["task_alias"]): item for item in population["tasks"]}
    if {task.task_alias for task in tasks} != set(expected):
        return "task alias set"
    for task in tasks:
        item = expected[task.task_alias]
        observed = {
            "task_cid": task.task_cid,
            "goal_cid": task.goal_cid,
            "plan_cid": task.plan_cid,
            "dependencies": _dependency_frontier(task.dependencies),
            "objective_id": task.objective_id,
        }
        required = {
            "task_cid": item["task_cid"],
            "goal_cid": item["goal_cid"],
            "plan_cid": item["plan_cid"],
            "dependencies": _dependency_frontier(item["depends_on"]),
            "objective_id": item["objective_id"],
        }
        drifted = [field for field, value in required.items() if observed[field] != value]
        if drifted:
            return task.task_alias + " " + ",".join(drifted)
    return ""


def _same_identities(tasks: tuple[Any, ...], population: Mapping[str, Any]) -> bool:
    return _identity_mismatch(tasks, population) == ""


def _receipt(
    *,
    plan: Mapping[str, Any],
    population: Mapping[str, Any],
    document: Any,
    reports: tuple[Any, ...],
    children: tuple[str, ...],
    source: Any,
    tasks: tuple[Any, ...],
    snapshot: Any,
    admitted_forest: Mapping[str, Any],
    admitted_plan_root_cid: str,
    admitted_tree_id: str,
    caller_forest: Mapping[str, Any],
    ignored_statuses: Mapping[str, str],
    idempotent_replay: bool,
) -> dict[str, Any]:
    ready = [task.task_alias for task in source.ready_tasks(limit=_PAGE_LIMIT).tasks]
    ordered = sorted(tasks, key=lambda item: item.task_alias)
    return {
        "schema": ADMISSION_SCHEMA,
        "namespace": str(plan["namespace"]),
        "isolated_namespace": True,
        "plan_definition_id": str(population["plans"][0]["plan_definition_id"]),
        "plan_root_cid": admitted_plan_root_cid,
        "repository_tree_id": admitted_tree_id,
        "task_cids_by_alias": {task.task_alias: task.task_cid for task in ordered},
        "goal_count": int(snapshot.goal_count),
        "task_count": int(snapshot.task_count),
        "dependency_count": int(snapshot.dependency_count),
        "task_statuses": {task.task_alias: task.status for task in ordered},
        "task_revisions": {task.task_alias: int(task.revision) for task in ordered},
        "ready_task_ids": ready,
        "typed_goals_accepted": all(report.accepted and not report.debt for report in reports),
        "typed_goal_count": len(document.goals),
        "objective_heap_id": document.objective_heap_id,
        "root_goal_id": ROOT_GOAL_ID,
        "root_child_ids": list(children),
        "admitted_root_breadth": ADMITTED_ROOT_BREADTH,
        "native_default_breadth": NATIVE_DEFAULT_BREADTH,
        "default_breadth_rejects_root": True,
        "root_breadth_explicitly_admitted": True,
        "markdown_bootstrap_only": True,
        "markdown_is_execution_authority": False,
        "markdown_statuses_ignored": dict(ignored_statuses),
        "markdown_completed_task_ids": [alias for alias, status in ignored_statuses.items() if status.casefold() in _COMPLETED_MARKDOWN],
        "source_forest": dict(admitted_forest),
        "caller_source_forest_ignored": _canonical(caller_forest) != _canonical(admitted_forest),
        "idempotent_replay": idempotent_replay,
        "execution_completed": False,
        "completion_authority": False,
        "typed_goals": document.to_dict(),
    }


def admit_goals_and_tasks(
    plan: Mapping[str, Any],
    goal_markdown: str,
    *,
    database_path: Path | str,
    source_forest: Mapping[str, Any],
    config_path: Path | str,
    python_executable: str,
    task_markdown: str | None = None,
) -> dict[str, Any]:
    """Materialize one isolated namespace, or replay it without resetting state.

    Lint and the ten-child breadth gate run before the database file is
    created. A second call with the same plan definition returns the stored
    identities even when the observed source forest has moved. Markdown task
    statuses are reported and ignored.
    """
    if not isinstance(plan, Mapping) or plan.get("schema") != "worldwide-legal-search.plan@1":
        raise WorldwideAdmissionError("unsupported worldwide campaign plan")
    namespace = str(plan.get("namespace") or "")
    if _NAMESPACE.fullmatch(namespace) is None:
        raise WorldwideAdmissionError("campaign namespace is not an isolated identifier")
    if not isinstance(source_forest, Mapping) or isinstance(source_forest, (str, bytes)):
        raise WorldwideAdmissionError("source forest must be an object")
    ignored_statuses = _ignored_markdown_statuses(task_markdown)
    # Quality lint is an admission precondition. Opening DuckDB before this
    # point would leave an empty namespace behind a rejected contract.
    document, reports = _linted_goals(plan, goal_markdown, config_path=config_path, python_executable=python_executable)
    children = _require_root_breadth(plan, document)
    population, _definition_id = _population(plan, source_forest)
    path = Path(database_path)
    if path.exists() and path.is_dir():
        raise WorldwideAdmissionError("native task database path is a directory")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".admission.lock")
    from ..task_sources.database_task_source import DatabaseTaskSource

    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with DatabaseTaskSource(path, owner_id=namespace + ":wwl-admission") as source:
            stored = _load_tasks(source)
            if stored:
                stored_namespace = str(stored[0].body.get("board_namespace") or "")
                if stored_namespace != namespace:
                    raise WorldwideAdmissionError("existing native store belongs to a different isolated namespace")
                stored_plan = source.get_plan(stored[0].plan_cid)
                stored_body = stored_plan.get("body") if isinstance(stored_plan, Mapping) else None
                if not isinstance(stored_body, Mapping) or stored_body.get("plan_definition_id") != population["plans"][0]["plan_definition_id"]:
                    raise WorldwideAdmissionError("existing native store belongs to a different plan definition")
                admitted_forest = stored_body.get("admitted_source_forest")
                admitted_root = str(stored_body.get("admitted_plan_root_cid") or "")
                admitted_tree = str(stored_body.get("repository_tree_id") or "")
                if not isinstance(admitted_forest, Mapping) or not admitted_root or not admitted_tree:
                    raise WorldwideAdmissionError("existing native store is missing its admitted source binding")
                if not _same_identities(stored, population):
                    # Same plan bytes with a moved forest keep the original
                    # CIDs. Re-materializing would mint new identities and
                    # reset task revisions back to todo.
                    try:
                        replay_population = build_population(plan, source_forest=admitted_forest)
                    except (TypeError, ValueError, KeyError) as exc:
                        raise WorldwideAdmissionError("existing native identities do not match the admitted plan") from exc
                    mismatch = _identity_mismatch(stored, replay_population)
                    if mismatch:
                        raise WorldwideAdmissionError("existing native identities do not match the admitted plan: " + mismatch)
                source.plan_root_cid = admitted_root
                source.repository_tree_id = admitted_tree
                snapshot = _require_store_shape(source, plan, stored)
                return _receipt(
                    plan=plan, population=population, document=document, reports=reports, children=children,
                    source=source, tasks=stored, snapshot=snapshot, admitted_forest=admitted_forest,
                    admitted_plan_root_cid=admitted_root, admitted_tree_id=admitted_tree,
                    caller_forest=source_forest, ignored_statuses=ignored_statuses, idempotent_replay=True,
                )
            materialized = source.materialize(population)
            stored = _load_tasks(source)
            if int(materialized["task_count"]) != len(plan["tasks"]) or int(materialized["goal_count"]) != len(plan["goals"]):
                raise WorldwideAdmissionError("native materialization lost the exact task or goal population")
            snapshot = _require_store_shape(source, plan, stored)
            mismatch = _identity_mismatch(stored, population)
            if mismatch:
                raise WorldwideAdmissionError("native materialization did not retain stable identities: " + mismatch)
            receipt = _receipt(
                plan=plan, population=population, document=document, reports=reports, children=children,
                source=source, tasks=stored, snapshot=snapshot,
                admitted_forest=population["plans"][0]["admitted_source_forest"],
                admitted_plan_root_cid=str(population["plan_root_cid"]),
                admitted_tree_id=str(population["repository_tree_id"]),
                caller_forest=source_forest, ignored_statuses=ignored_statuses, idempotent_replay=False,
            )
            if any(status in {"complete", "completed", "done"} for status in receipt["task_statuses"].values()):
                raise WorldwideAdmissionError("initial admission applied a terminal markdown or plan status")
            return receipt
