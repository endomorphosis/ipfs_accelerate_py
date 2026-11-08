"""Admit worldwide legal search goals and bind the explicit Grok agent route.

``goals.objectives.md`` and ``tasks.todo.md`` are bootstrap projections. Live
identities, dependencies, readiness and status live only in an isolated
``DatabaseTaskSource`` namespace. Typed goal contracts are derived from the
objective heap and linted before any database file is created. The root's ten
children are an explicit breadth admission: the native default of eight is
rejected rather than inherited.

Provider execution uses ``ipfs_accelerate_py.llm_router``'s explicit agent
route: ``grok_cli`` / ``grok-4.7``, emergency policy
``grok_quota_auth_or_unavailable``, and Codex only after a typed pre-effect
failure on an unchanged workspace. This module decides and records that route.
The only process it launches is the admitted tiny live canary.
"""
from __future__ import annotations

import fcntl
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
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


ROUTE_SCHEMA = "worldwide-legal-search.explicit-agent-route@1"
TELEMETRY_SCHEMA = "worldwide-legal-search.provider-route-telemetry@1"
CANARY_ADMISSION_SCHEMA = "worldwide-legal-search.tiny-live-canary-admission@1"
CANARY_RESULT_SCHEMA = "worldwide-legal-search.tiny-live-canary@1"
ROUTER_NAME = "ipfs_accelerate_py.llm_router"
PRIMARY_PROVIDER = "grok_cli"
PRIMARY_MODEL = "grok-4.7"
FALLBACK_PROVIDER = "codex_cli"
FALLBACK_MODEL_ADMISSION = "resolve from admitted existing Codex configuration"
ROUTE_POLICY = "grok_quota_auth_or_unavailable"
_ROUTE_STAGES = frozenset({"implementation", "canary"})
_TYPED_PRE_EFFECT_KINDS = frozenset({
    "grok_quota_exhausted",
    "authentication_failure",
    "launch_failure",
})
_CANARY_PROMPT = "WWL-003 admitted tiny route canary\n"
_CANARY_TASK_ID = "WWL-003"
_CANARY_ATTEMPT = 1
CANONICAL_ROUTING = {
    "router": ROUTER_NAME,
    "primary_provider": PRIMARY_PROVIDER,
    "primary_model": PRIMARY_MODEL,
    "fallback_provider": FALLBACK_PROVIDER,
    "fallback_model": FALLBACK_MODEL_ADMISSION,
    "policy": ROUTE_POLICY,
    "fallback_conditions": (
        "typed quota exhaustion",
        "typed authentication failure",
        "typed launch unavailability",
    ),
    "unchanged_workspace_required": True,
    "unknown_effects_require_reconciliation": True,
    "mandatory_routine_codex_review": False,
}


class WorldwideRouteError(ValueError):
    """The campaign route cannot be bound or a fallback would be ambiguous."""


def _route_modules():
    from ... import llm_router
    from . import worldwide_legal_search_provider as provider

    if provider.MODEL != PRIMARY_MODEL or provider.POLICY != ROUTE_POLICY:
        raise WorldwideRouteError("provider adapter drifted from grok_cli/grok-4.7 emergency policy")
    if llm_router.GROK_QUOTA_AUTH_OR_UNAVAILABLE_AGENT_ROUTE_POLICY != ROUTE_POLICY:
        raise WorldwideRouteError("llm_router explicit agent policy is not the campaign emergency policy")
    return llm_router, provider


def _routing_view(plan: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, Mapping):
        raise WorldwideRouteError("campaign plan must be an object")
    routing = plan.get("routing")
    if not isinstance(routing, Mapping) or isinstance(routing, (str, bytes)):
        raise WorldwideRouteError("campaign routing must be an object")
    return dict(routing)


def require_canonical_routing(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless the plan names the canonical explicit agent route."""
    routing = _routing_view(plan)
    expected = dict(CANONICAL_ROUTING)
    expected["fallback_conditions"] = list(CANONICAL_ROUTING["fallback_conditions"])
    if set(routing) != set(expected) or any(routing[key] != expected[key] for key in expected):
        raise WorldwideRouteError("campaign routing is not the canonical grok_cli/grok-4.7 emergency Codex route")
    _route_modules()
    return expected


def snapshot_workspace(workspace: Path | str):
    """Hash the candidate workspace with the router's side-effect snapshot."""
    llm_router, _provider = _route_modules()
    return llm_router.snapshot_agent_cli_workspace(workspace)


def trusted_pre_effect_receipt(kind: str, *, returncode: int, activity: str, reason: str) -> str:
    """Serialize one body-free receipt through the canonical router helper."""
    llm_router, _provider = _route_modules()
    try:
        classification = llm_router.AgentCLIFailureClassification(
            llm_router.AgentCLIProviderFailureKind(kind),
            reason,
        )
        return llm_router.serialize_agent_cli_failure_receipt(
            classification,
            returncode=returncode,
            activity_state=llm_router.AgentCLIActivityState(activity),
        )
    except (ValueError, llm_router.LLMRouterError) as exc:
        raise WorldwideRouteError("trusted pre-effect receipt is not canonical") from exc


def bind_explicit_agent_route(
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    workspace: Path | str,
    task_id: str,
    attempt: int,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """Build the production runner argv. This does not launch a provider."""
    routing = require_canonical_routing(plan)
    _llm_router, provider = _route_modules()
    try:
        argv = provider.build_provider_argv(
            config,
            workspace=Path(workspace),
            task_id=task_id,
            attempt=attempt,
            python_executable=python_executable,
        )
    except (OSError, ValueError, KeyError) as exc:
        raise WorldwideRouteError(str(exc)) from exc
    primary = json.loads(argv[argv.index("--primary-command-json") + 1])
    if (
        argv[argv.index("--primary-provider") + 1] != "grok"
        or argv[argv.index("--fallback-provider") + 1] != "codex"
        or argv[argv.index("--fallback-policy") + 1] != ROUTE_POLICY
        or argv[argv.index("--probe-grok-model") + 1] != PRIMARY_MODEL
        or primary[primary.index("--model") + 1] != PRIMARY_MODEL
        or primary[primary.index("--mode") + 1] != "agent"
        or "--probe-route-readiness" not in argv
        or "--primary-unavailable-kind" in argv
        or "--dangerously-bypass-approvals-and-sandbox" in argv
    ):
        raise WorldwideRouteError("explicit agent argv drifted from the canonical grok_cli/grok-4.7 route")
    return {
        "schema": ROUTE_SCHEMA,
        "router": routing["router"],
        "primary_provider": PRIMARY_PROVIDER,
        "primary_model": PRIMARY_MODEL,
        "fallback_provider": FALLBACK_PROVIDER,
        "fallback_policy": ROUTE_POLICY,
        "fallback_model_admission": FALLBACK_MODEL_ADMISSION,
        "unchanged_workspace_required": True,
        "unknown_effects_require_reconciliation": True,
        "mandatory_routine_codex_review": False,
        "task_id": task_id,
        "attempt": attempt,
        "stage": "implementation",
        "argv": argv,
        "launched": False,
        "hosted_model_invocations": 0,
        "paid_provider_calls": 0,
        "completion_authority": False,
    }


def _workspace_unchanged(before: Any, after: Any) -> bool:
    return bool(
        getattr(before, "reliable", False)
        and getattr(after, "reliable", False)
        and getattr(before, "digest", "")
        and before.digest == after.digest
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = _canonical(payload).encode("utf-8")
    if len(encoded) > 16 * 1024:
        raise WorldwideRouteError("route telemetry exceeds its body-free bound")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def evaluate_emergency_fallback(
    workspace: Path | str,
    *,
    task_id: str,
    attempt: int,
    stage: str,
    primary_result: Any,
    workspace_before: Any,
    telemetry_path: Path | str,
    trusted_failure_receipt: str = "",
    primary_unavailable_kind: str | None = None,
) -> dict[str, Any]:
    """Apply the canonical pre-effect route and persist one telemetry record.

    Codex is selected only for typed quota, authentication, or launch failure
    before side effects, and only when the candidate digest is unchanged.
    Timeout, transport, malformed output, generic failure, started or unknown
    activity, an unreliable snapshot, and a mutated workspace stay on the
    primary attempt for reconciliation. This function does not launch a model.
    """
    llm_router, provider = _route_modules()
    if stage not in _ROUTE_STAGES or provider._TASK_RE.fullmatch(task_id) is None or type(attempt) is not int or attempt < 1:
        raise WorldwideRouteError("emergency fallback needs a native task, positive attempt, and known stage")
    if not isinstance(primary_result, llm_router.AgentCLIProviderResult):
        raise WorldwideRouteError("primary result must be the canonical agent CLI result")
    root = Path(workspace)
    after = snapshot_workspace(root)
    unchanged = _workspace_unchanged(workspace_before, after)
    try:
        decision = llm_router.route_agent_cli_failure(
            policy=ROUTE_POLICY,
            primary_provider="grok",
            fallback_provider="codex",
            primary_result=primary_result,
            workspace_before=workspace_before,
            workspace_after=after,
            trusted_failure_receipt=trusted_failure_receipt,
            primary_unavailable_kind=primary_unavailable_kind,
            receipt_binding={"task_id": task_id, "attempt": attempt, "stage": stage},
        )
    except llm_router.LLMRouterError as exc:
        raise WorldwideRouteError(str(exc)) from exc
    should_fallback = bool(decision.should_fallback and unchanged)
    if decision.should_fallback and not unchanged:
        reliable = bool(getattr(workspace_before, "reliable", False) and after.reliable and getattr(workspace_before, "digest", "") and after.digest)
        terminal = "side_effects_started" if reliable else "workspace_snapshot_unavailable"
    else:
        terminal = decision.terminal_reason
    kind = decision.classification.kind.value
    if should_fallback and kind not in _TYPED_PRE_EFFECT_KINDS:
        raise WorldwideRouteError("emergency fallback selected a non-typed failure")
    route_record = dict(decision.route_record) if should_fallback and decision.route_record is not None else None
    if route_record is not None:
        if route_record.get("route") != "fallback" or route_record.get("side_effects_started") is not False:
            raise WorldwideRouteError("canonical route record does not describe a pre-effect fallback")
        if route_record.get("completion_authority") is not False or route_record.get("fallback_policy") != ROUTE_POLICY:
            raise WorldwideRouteError("canonical route record drifted from the emergency policy")
    telemetry = {
        "schema": TELEMETRY_SCHEMA,
        "router": ROUTER_NAME,
        "task_id": task_id,
        "attempt": attempt,
        "stage": stage,
        "primary_provider": PRIMARY_PROVIDER,
        "primary_model": PRIMARY_MODEL,
        "fallback_provider": FALLBACK_PROVIDER,
        "fallback_policy": ROUTE_POLICY,
        "workspace_digest": after.digest if after.reliable else "",
        "workspace_unchanged": unchanged,
        "decision": "fallback" if should_fallback else "reconcile",
        "terminal_reason": terminal,
        "failure_kind": kind,
        "reason_code": decision.classification.reason_code,
        "side_effects_started": terminal == "side_effects_started",
        "effects_ambiguous": terminal in {"side_effects_started", "workspace_snapshot_unavailable"},
        "route_record": route_record,
        "hosted_model_invocations": 0,
        "paid_provider_calls": 0,
        "completion_authority": False,
    }
    _write_json(Path(telemetry_path), telemetry)
    return telemetry


def admitted_tiny_canary() -> dict[str, Any]:
    """The only live provider execution this binding is allowed to start."""
    prompt_digest = hashlib.sha256(_CANARY_PROMPT.encode("utf-8")).hexdigest()
    return {
        "schema": CANARY_ADMISSION_SCHEMA,
        "task_id": _CANARY_TASK_ID,
        "attempt": _CANARY_ATTEMPT,
        "stage": "canary",
        "prompt_sha256": prompt_digest,
        "max_prompt_bytes": 64,
        "primary_unavailable_kind": "launch_failure",
        "hosted_model_invocations": 0,
        "paid_provider_calls": 0,
        "probe_route_readiness": False,
        "completion_authority": False,
    }


def _canary_paths(state_root: Path) -> tuple[Path, Path, Path, Path]:
    root = state_root / "state" / "ptr_lane_worldwide_legal_search" / "provider_route_receipts" / _CANARY_TASK_ID
    return (
        root / f"provider-route-attempt-{_CANARY_ATTEMPT}.json",
        root / "tiny-live-canary.json",
        state_root / "canary" / "fallback.py",
        state_root / "canary" / "invoked",
    )


def run_admitted_tiny_canary(
    admission: Mapping[str, Any],
    *,
    workspace: Path | str,
    state_root: Path | str,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """Run one non-charging canary through the real fallback runner.

    Primary launch is the typed pre-dispatch ``launch_failure`` condition, so
    Grok is not started. The fallback process is a local script that checks
    the admitted prompt and writes a marker outside the candidate workspace.
    A second call, a different admission, a hosted probe, or a workspace
    change is refused.
    """
    expected = admitted_tiny_canary()
    if not isinstance(admission, Mapping) or _canonical(dict(admission)) != _canonical(expected):
        raise WorldwideRouteError("tiny live canary admission does not match the admitted contract")
    if len(_CANARY_PROMPT.encode("utf-8")) > int(expected["max_prompt_bytes"]):
        raise WorldwideRouteError("admitted canary prompt exceeds its byte bound")
    candidate = Path(workspace).expanduser().resolve(strict=True)
    state = Path(state_root).expanduser().resolve(strict=True)
    if not candidate.is_dir() or not state.is_dir() or candidate == state or candidate.is_relative_to(state) or state.is_relative_to(candidate):
        raise WorldwideRouteError("canary workspace and state root must be disjoint directories")
    receipt_path, envelope_path, script_path, marker_path = _canary_paths(state)
    if receipt_path.exists() or envelope_path.exists() or marker_path.exists():
        raise WorldwideRouteError("admitted tiny live canary already ran")
    python = str(python_executable or sys.executable)
    if Path(python).expanduser().is_absolute() is False:
        raise WorldwideRouteError("canary python must be an absolute interpreter")
    _llm_router, _provider = _route_modules()
    from .. import provider_fallback_runner
    from ..validation.validation_runtime import PROOF_REUSE_STATE_ROOT_ENV, PROVIDER_PROTECTED_STATE_ROOT_ENV

    script_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    script_path.write_text(
        "import pathlib, sys\n"
        f"expected = {_CANARY_PROMPT!r}\n"
        "prompt = sys.stdin.read()\n"
        "marker = pathlib.Path(sys.argv[1])\n"
        "if prompt != expected:\n"
        "    raise SystemExit(3)\n"
        "marker.write_text('admitted-tiny-canary\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    before = snapshot_workspace(candidate)
    if not before.reliable:
        raise WorldwideRouteError("canary workspace snapshot is unavailable")
    with tempfile.TemporaryDirectory(prefix="wwl-route-canary-home-") as private_home_text:
        private_home = Path(private_home_text)
        grok_home = private_home / "empty-grok"
        grok_home.mkdir(mode=0o700)
        env = os.environ.copy()
        env["HOME"] = str(private_home)
        env["GROK_HOME"] = str(grok_home)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        for name in (
            "CODEX_HOME",
            "IPFS_ACCELERATE_AGENT_VALIDATION_PATH",
            PROOF_REUSE_STATE_ROOT_ENV,
            PROVIDER_PROTECTED_STATE_ROOT_ENV,
            "XAI_API_KEY",
            "GROK_AUTH_PROVIDER_COMMAND",
            "IPFS_ACCELERATE_PY_XAI_API_KEY",
            "ipfs_accelerate_py_XAI_API_KEY",
            "IPFS_DATASETS_PY_XAI_API_KEY",
        ):
            env.pop(name, None)
        command = [
            python,
            str(Path(provider_fallback_runner.__file__).resolve()),
            "--workspace", str(candidate),
            "--primary-provider", "grok",
            "--fallback-provider", "codex",
            "--primary-command-json", "[]",
            "--fallback-command-json", json.dumps([python, str(script_path), str(marker_path)], separators=(",", ":")),
            "--fallback-policy", ROUTE_POLICY,
            "--primary-unavailable-kind", "launch_failure",
            "--route-receipt-path", str(receipt_path),
            "--route-task-id", _CANARY_TASK_ID,
            "--route-attempt", str(_CANARY_ATTEMPT),
            "--route-stage", "canary",
        ]
        if "--probe-route-readiness" in command or "--model" in command:
            raise WorldwideRouteError("tiny canary must not probe or pin a hosted model command")
        try:
            completed = subprocess.run(
                command,
                input=_CANARY_PROMPT,
                text=True,
                cwd=candidate,
                env=env,
                capture_output=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise WorldwideRouteError("admitted tiny live canary could not start") from exc
    after = snapshot_workspace(candidate)
    if completed.returncode != 0 or not marker_path.is_file() or marker_path.read_text(encoding="utf-8") != "admitted-tiny-canary\n":
        raise WorldwideRouteError("admitted tiny live canary did not complete on the local fallback: " + completed.stderr[-500:])
    if not _workspace_unchanged(before, after):
        raise WorldwideRouteError("admitted tiny live canary changed the candidate workspace")
    try:
        route_record = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorldwideRouteError("canary route telemetry is missing") from exc
    if (
        not isinstance(route_record, dict)
        or route_record.get("schema") != "ipfs_accelerate_py/provider-route@1"
        or route_record.get("route") != "fallback"
        or route_record.get("failure_kind") != "launch_failure"
        or route_record.get("fallback_policy") != ROUTE_POLICY
        or route_record.get("primary_provider") != "grok"
        or route_record.get("fallback_provider") != "codex"
        or route_record.get("side_effects_started") is not False
        or route_record.get("completion_authority") is not False
        or route_record.get("task_id") != _CANARY_TASK_ID
        or route_record.get("attempt") != _CANARY_ATTEMPT
        or route_record.get("stage") != "canary"
        or route_record.get("primary_returncode") is not None
    ):
        raise WorldwideRouteError("canary route telemetry is not a typed pre-effect fallback")
    envelope = {
        "schema": CANARY_RESULT_SCHEMA,
        "admission": expected,
        "router": ROUTER_NAME,
        "primary_provider": PRIMARY_PROVIDER,
        "primary_model": PRIMARY_MODEL,
        "fallback_provider": FALLBACK_PROVIDER,
        "fallback_policy": ROUTE_POLICY,
        "fallback_executable": "admitted-local-canary",
        "probe_route_readiness": False,
        "workspace_digest_before": before.digest,
        "workspace_digest_after": after.digest,
        "workspace_unchanged": True,
        "decision": "fallback",
        "terminal_reason": "fallback_selected",
        "failure_kind": "launch_failure",
        "side_effects_started": False,
        "effects_ambiguous": False,
        "route_record": route_record,
        "fallback_invoked": True,
        "hosted_model_invocations": 0,
        "paid_provider_calls": 0,
        "completion_authority": False,
    }
    _write_json(envelope_path, envelope)
    return envelope
