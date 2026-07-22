"""CLI bridge for reusable objective-graph backlog generation.

This package-level entry point scans an objective heap, appends missing-evidence
tasks, persists AST records, writes parallel bundle shards, and optionally
submits those bundles to a local task queue.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .objective_graph import (
    DEFAULT_DISCOVERY_OUTPUT_PATH,
    DEFAULT_OBJECTIVE_TASK_SUMMARY_PREFIX,
    DEFAULT_SURPLUS_FINDINGS_PER_GOAL,
    DEFAULT_SURPLUS_MIN_TERMS_PER_TODO,
    DEFAULT_TASK_PREFIX,
    generate_objective_todos,
    repo_relative_path,
    scan_objective_gaps,
    submit_bundle_tasks,
)
from .objective_tracker import (
    DEFAULT_GOAL_PREFIX,
    DEFAULT_ROOT_GOAL_TITLE,
    DEFAULT_TRACKING_DOCUMENT_TITLE,
    DEFAULT_ULTIMATE_GOAL,
    append_interoperability_goals,
    append_launch_readiness_goals,
    append_refinement_goals,
    deduplicate_interoperability_goals,
    ensure_objective_tracking_document,
    parse_root_evidence,
    reconcile_objective_goal_completion,
    write_objective_graph_artifact,
)

logger = logging.getLogger(__name__)


def _plan_value_dict(value: Any) -> dict[str, Any]:
    """Return a JSON-safe mapping for a structured planning value."""

    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return {str(key): item for key, item in payload.items()}
    if is_dataclass(value):
        payload = asdict(value)
        if isinstance(payload, dict):
            return payload
    raise TypeError(f"structured plan value must be mapping-like, got {type(value).__name__}")


def objective_record_plan_context(record: Any) -> dict[str, Any]:
    """Build the schema router input for one generated objective subgoal."""

    finding = record.finding
    validation = str(getattr(finding, "validation", "") or "").strip()
    validation_commands = [item.strip() for item in validation.split(";") if item.strip()]
    predicted_files = list(
        dict.fromkeys(
            str(item)
            for item in [
                *(getattr(finding, "predicted_files", ()) or ()),
                *(getattr(finding, "outputs", ()) or ()),
            ]
            if str(item).strip()
        )
    )
    predicted_symbols = list(
        dict.fromkeys(
            str(item)
            for item in (getattr(finding, "ast_symbols", ()) or ())
            if str(item).strip()
        )
    )
    return {
        "task_id": str(record.task_id),
        "goal_id": str(getattr(finding, "goal_id", "") or ""),
        "title": str(getattr(finding, "title", "") or ""),
        "summary": str(getattr(finding, "summary", "") or ""),
        "goal": str(getattr(finding, "goal", "") or getattr(finding, "title", "") or ""),
        "priority": str(getattr(finding, "priority", "") or ""),
        "track": str(getattr(finding, "track", "") or ""),
        "missing_evidence": [str(item) for item in getattr(finding, "missing_evidence", ()) or ()],
        "predicted_files": predicted_files,
        "predicted_symbols": predicted_symbols,
        "dependencies": [str(item) for item in getattr(finding, "parent_goal_ids", ()) or ()],
        "validation_commands": validation_commands,
        "validation_proof": validation_commands,
    }


def _evaluated_branch_dict(value: Any, evaluation: Any = None) -> dict[str, Any]:
    branch = getattr(value, "branch", None)
    if branch is None and isinstance(value, Mapping):
        branch = value.get("branch", value)
    if branch is None:
        # PlanEvaluation exposes selected/rejected PlanBranch values while
        # some consumers wrap them as EvaluatedPlanBranch. Support both
        # representations at this daemon boundary.
        branch = value
    branch_payload = _plan_value_dict(branch)
    branch_id = str(branch_payload.get("branch_id") or "")
    scores = getattr(evaluation, "scores", {}) if evaluation is not None else {}
    rationales = getattr(evaluation, "rationales", {}) if evaluation is not None else {}
    wrapped_score = (
        getattr(value, "score_millionths", None)
        if not isinstance(value, Mapping)
        else value.get("score_millionths")
    )
    wrapped_rationale = (
        getattr(value, "rationale", None)
        if not isinstance(value, Mapping)
        else value.get("rationale")
    )
    payload = {
        "branch": branch_payload,
        "score_millionths": int(wrapped_score if wrapped_score is not None else scores.get(branch_id, 0)),
        "rationale": list(
            wrapped_rationale if wrapped_rationale is not None else rationales.get(branch_id, ()) or ()
        ),
    }
    return payload


def plan_objective_records(
    records: Sequence[Any],
    *,
    branch_count: int = 3,
    router: Callable[..., Any] | None = None,
    fallback_planner: Callable[..., Any] | None = None,
    router_config: Any = None,
    evaluator: Callable[[Sequence[Any]], Any] | None = None,
    use_llm_router: bool = True,
) -> list[dict[str, Any]]:
    """Generate and deterministically select plan branches per ready record.

    Router failures are isolated to their record and use the deterministic
    planner.  Consequently a failed provider cannot prevent later ready work
    from receiving a selected implementation plan.
    """

    from .plan_evaluator import evaluate_plan_branches
    from .task_proposal_router import deterministic_plan_branches, generate_structured_plan_branches

    evaluate = evaluator or evaluate_plan_branches
    count = max(1, int(branch_count))
    decisions: list[dict[str, Any]] = []
    for record in records:
        context = objective_record_plan_context(record)
        route_error = ""
        used_fallback = False
        try:
            if use_llm_router:
                routed = generate_structured_plan_branches(
                    context,
                    router=router,
                    fallback_planner=fallback_planner,
                    config=router_config,
                    branch_count=count,
                )
                branches = tuple(routed.branches)
                used_fallback = bool(routed.used_fallback)
                route_error = str(routed.router_error or "")
            else:
                planner = fallback_planner or deterministic_plan_branches
                branches = tuple(planner(context, 1))
            evaluation = evaluate(branches)
            selected = _evaluated_branch_dict(evaluation.selected, evaluation)
            rejected = [_evaluated_branch_dict(item, evaluation) for item in evaluation.rejected]
            decisions.append(
                {
                    "task_id": context["task_id"],
                    "goal_id": context["goal_id"],
                    "source": (
                        "deterministic_fallback"
                        if used_fallback
                        else ("llm_router" if use_llm_router else "deterministic_planner")
                    ),
                    "used_fallback": used_fallback,
                    "router_error": route_error or None,
                    "evaluator_version": str(evaluation.evaluator_version),
                    "selected": selected,
                    "rejected": rejected,
                    "selection_rationale": selected["rationale"],
                }
            )
        except Exception as exc:  # one provider/record must never stall later ready work
            logger.warning("Plan routing failed for %s; using deterministic fallback: %s", context["task_id"], exc)
            try:
                branches = tuple(deterministic_plan_branches(context, branch_count=1))
                evaluation = evaluate(branches)
                selected = _evaluated_branch_dict(evaluation.selected, evaluation)
                decisions.append(
                    {
                        "task_id": context["task_id"],
                        "goal_id": context["goal_id"],
                        "source": "deterministic_fallback",
                        "used_fallback": True,
                        "router_error": str(exc),
                        "evaluator_version": str(evaluation.evaluator_version),
                        "selected": selected,
                        "rejected": [_evaluated_branch_dict(item, evaluation) for item in evaluation.rejected],
                        "selection_rationale": selected["rationale"],
                    }
                )
            except Exception as fallback_exc:
                # Preserve the failure as evidence and continue planning the
                # remaining records. A malformed injected fallback must not
                # turn one subgoal into a daemon-wide outage.
                logger.error("Deterministic plan fallback failed for %s: %s", context["task_id"], fallback_exc)
                decisions.append(
                    {
                        "task_id": context["task_id"],
                        "goal_id": context["goal_id"],
                        "source": "planning_error",
                        "used_fallback": True,
                        "router_error": str(exc),
                        "fallback_error": str(fallback_exc),
                        "evaluator_version": "",
                        "selected": None,
                        "rejected": [],
                        "selection_rationale": ["deterministic fallback could not produce a valid branch"],
                    }
                )
    return decisions


def persist_objective_plan_evaluations(
    path: Path,
    decisions: Sequence[Mapping[str, Any]],
    *,
    bundle_index_path: Path | None = None,
) -> None:
    """Persist decisions and project them into scheduler-visible bundle tasks."""

    retained: dict[str, dict[str, Any]] = {}
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            previous_items = previous.get("evaluations", []) if isinstance(previous, Mapping) else []
            if isinstance(previous_items, list):
                retained.update(
                    {
                        str(item.get("task_id") or ""): dict(item)
                        for item in previous_items
                        if isinstance(item, Mapping) and str(item.get("task_id") or "")
                    }
                )
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not retain prior plan decisions from %s: %s", path, exc)
    retained.update(
        {
            str(item.get("task_id") or ""): dict(item)
            for item in decisions
            if str(item.get("task_id") or "")
        }
    )
    ordered = [retained[task_id] for task_id in sorted(retained)]
    payload = {
        "schema": "ipfs_accelerate_py.agent_supervisor.objective_plan_evaluations@1",
        "evaluation_count": len(ordered),
        "evaluations": ordered,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if bundle_index_path is None or not bundle_index_path.exists():
        return
    try:
        bundle_payload = json.loads(bundle_index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not project plan decisions into %s: %s", bundle_index_path, exc)
        return
    if not isinstance(bundle_payload, dict):
        return
    by_task = {str(item.get("task_id") or ""): item for item in ordered}
    bundles = bundle_payload.get("bundles")
    if isinstance(bundles, Mapping):
        for info in bundles.values():
            if not isinstance(info, dict) or not isinstance(info.get("tasks"), list):
                continue
            for task in info["tasks"]:
                if not isinstance(task, dict):
                    continue
                decision = by_task.get(str(task.get("task_id") or ""))
                if decision is None:
                    continue
                task["plan_evaluation"] = decision
                selected = decision.get("selected")
                task["selected_plan_evaluation"] = selected
                task["selected_plan_branch"] = (
                    selected.get("branch") if isinstance(selected, Mapping) else None
                )
                task["rejected_plan_branches"] = decision.get("rejected", [])
                task["plan_selection_rationale"] = decision.get("selection_rationale", [])
    bundle_payload["plan_evaluation_path"] = str(path)
    bundle_payload["plan_evaluation_count"] = len(ordered)
    from .artifact_store import write_bundle_index_artifact

    write_bundle_index_artifact(bundle_index_path, bundle_payload)


def default_repo_root() -> Path:
    return Path.cwd()


def default_objective_path(repo_root: Path) -> Path:
    return repo_root / "implementation_plan" / "docs" / "23-virtual-ai-os-objective-goal-heap.md"


def default_todo_path(repo_root: Path) -> Path:
    return repo_root / "docs" / "AGENT_OBJECTIVE_TODO.md"


def default_state_root(repo_root: Path) -> Path:
    return repo_root / "data" / "agent_supervisor"


def split_csv(values: Iterable[str]) -> list[str]:
    items: list[str] = []
    for value in values:
        for raw in value.split(","):
            item = " ".join(raw.strip().split())
            if item:
                items.append(item)
    return items


def parse_goal_completion_todo_boards(
    specs: Iterable[str],
    *,
    repo_root: Path,
    default_task_prefix: str,
) -> list[tuple[Path, str]]:
    """Parse extra objective-completion board specs as ``path::task-prefix``."""

    boards: list[tuple[Path, str]] = []
    for raw_spec in specs:
        spec = str(raw_spec or "").strip()
        if not spec:
            continue
        if "::" in spec:
            raw_path, raw_prefix = spec.split("::", 1)
            prefix = raw_prefix.strip() or default_task_prefix
        else:
            raw_path = spec
            prefix = default_task_prefix
        path = Path(raw_path.strip())
        if not path.is_absolute():
            path = repo_root / path
        boards.append((path.resolve(), prefix))
    return boards


def discovery_fingerprints(discovery_dir: Path) -> set[str]:
    """Return previously filed objective-gap fingerprints from discovery files."""

    if not discovery_dir.exists():
        return set()
    fingerprints: set[str] = set()
    pattern = re.compile(r"^Fingerprint:\s*(\S+)\s*$", flags=re.MULTILINE)
    for path in discovery_dir.rglob("*objective-gap*.md"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        fingerprints.update(match.group(1) for match in pattern.finditer(text))
    return fingerprints


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate autonomous-agent todos from an objective goal heap")
    parser.add_argument("--repo-root", type=Path, default=default_repo_root())
    parser.add_argument("--objective-path", type=Path, default=None)
    parser.add_argument("--todo-path", type=Path, default=None)
    parser.add_argument("--discovery-dir", type=Path, default=None)
    parser.add_argument("--bundle-dir", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--graph-path", type=Path, default=None)
    parser.add_argument("--task-prefix", default=DEFAULT_TASK_PREFIX)
    parser.add_argument("--objective-summary-prefix", default=DEFAULT_OBJECTIVE_TASK_SUMMARY_PREFIX)
    parser.add_argument("--discovery-output-path", default=DEFAULT_DISCOVERY_OUTPUT_PATH)
    parser.add_argument("--depends-on", action="append", default=[])
    parser.add_argument("--seen-fingerprint", action="append", default=[])
    parser.add_argument(
        "--force-goal-id",
        action="append",
        default=[],
        help="Objective goal id to rescan even when an existing discovery fingerprint would suppress it.",
    )
    parser.add_argument("--repeat-existing", action="store_true", help="Do not suppress fingerprints already in discovery files")
    parser.add_argument("--max-findings", type=int, default=10)
    parser.add_argument(
        "--surplus-findings-per-goal",
        type=int,
        default=DEFAULT_SURPLUS_FINDINGS_PER_GOAL,
        help=(
            "Generate up to this many structured candidate todos per missing goal. "
            "The first candidate is the aggregate gap; additional candidates form multi-evidence batches "
            "so the vector index can bundle or merge related surplus work without creating tiny tasks."
        ),
    )
    parser.add_argument(
        "--surplus-min-terms-per-todo",
        type=int,
        default=DEFAULT_SURPLUS_MIN_TERMS_PER_TODO,
        help="Minimum missing-evidence terms for non-aggregate surplus todos when enough terms are available.",
    )
    parser.add_argument("--ensure-tracking-document", action="store_true")
    parser.add_argument("--ultimate-goal", default=DEFAULT_ULTIMATE_GOAL)
    parser.add_argument("--root-evidence", action="append", default=[])
    parser.add_argument("--goal-prefix", default=None)
    parser.add_argument("--root-goal-id", default=None)
    parser.add_argument("--root-goal-title", default=DEFAULT_ROOT_GOAL_TITLE)
    parser.add_argument("--tracking-document-title", default=DEFAULT_TRACKING_DOCUMENT_TITLE)
    parser.add_argument("--refine-objective-heap", action="store_true")
    parser.add_argument("--max-refinement-children", type=int, default=3)
    parser.add_argument("--max-refinement-depth", type=int, default=4)
    parser.add_argument(
        "--no-reconcile-goal-completion",
        action="store_true",
        help="Skip marking active goals completed when all required evidence is already present.",
    )
    parser.add_argument(
        "--objective-goal-completion-todo-board",
        action="append",
        default=[],
        help=(
            "Extra todo board that can keep objective goals open while referenced work is pending. "
            "Use 'path::TASK-' or 'path::## TASK-' and repeat for shared cross-track boards."
        ),
    )
    parser.add_argument(
        "--seed-interoperability-goals",
        action="store_true",
        help="Seed objective subgoals for cross-submodule interoperability and integration tests.",
    )
    parser.add_argument(
        "--interoperability-focus",
        action="append",
        default=[],
        help=(
            "Submodule path to pair with other submodules when seeding interoperability goals. "
            "If omitted, all submodule pairs are eligible."
        ),
    )
    parser.add_argument(
        "--interoperability-component-path",
        action="append",
        default=[],
        help="Repo-relative component path to include when seeding interoperability goals.",
    )
    parser.add_argument("--max-interoperability-goals", type=int, default=12)
    parser.add_argument(
        "--seed-launch-readiness-goals",
        action="store_true",
        help="Seed high-value launch-readiness goals for Swissknife, Hallucinate App, MCP servers, and Meta glasses.",
    )
    parser.add_argument("--max-launch-readiness-goals", type=int, default=8)
    parser.add_argument("--no-persist-ast-dataset", action="store_true")
    parser.add_argument(
        "--no-todo-vector-index",
        action="store_true",
        help="Skip writing the todo vector/AST index artifact.",
    )
    parser.add_argument(
        "--todo-vector-index-path",
        type=Path,
        default=None,
        help="Path for the todo vector/AST index artifact. Defaults to <bundle-dir>/todo_vector_index.json.",
    )
    parser.add_argument(
        "--generate-plan-branches",
        action="store_true",
        help=(
            "Generate multiple schema-validated branches for each new objective subgoal through llm_router. "
            "Without this flag the same scheduler artifact is populated by the deterministic planner."
        ),
    )
    parser.add_argument(
        "--plan-branch-count",
        type=int,
        default=3,
        help="Requested number of LLM plan alternatives per eligible subgoal (default: 3).",
    )
    parser.add_argument(
        "--plan-evaluation-path",
        type=Path,
        default=None,
        help="Selected and rejected plan artifact. Defaults to <state-root>/plan_evaluations.json.",
    )
    parser.add_argument(
        "--plan-router-provider",
        default=os.environ.get("IPFS_DATASETS_PY_LLM_PROVIDER", ""),
        help="Optional llm_router provider for structured planning.",
    )
    parser.add_argument(
        "--plan-router-model",
        default=os.environ.get("IPFS_DATASETS_PY_LLM_MODEL", "gpt-5.3-codex-spark"),
        help="llm_router model for structured planning.",
    )
    parser.add_argument("--plan-router-max-new-tokens", type=int, default=4096)
    parser.add_argument("--plan-router-timeout", type=int, default=300)
    parser.add_argument("--plan-router-temperature", type=float, default=0.1)
    parser.add_argument(
        "--plan-router-allow-local-fallback",
        action="store_true",
        help="Allow llm_router's local provider fallback before deterministic planning.",
    )
    parser.add_argument("--submit-bundles", action="store_true", help="Submit generated bundle shards to the local task queue")
    parser.add_argument("--queue-path", default=None)
    parser.add_argument("--queue-task-type", default="codex.todo_bundle")
    parser.add_argument("--queue-model-name", default="codex")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def run_objective_daemon(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    objective_path = (args.objective_path or default_objective_path(repo_root)).resolve()
    todo_path = (args.todo_path or default_todo_path(repo_root)).resolve()
    state_root = default_state_root(repo_root)
    discovery_dir = (args.discovery_dir or state_root / "discovery").resolve()
    bundle_dir = (args.bundle_dir or state_root / "objective_bundles").resolve()
    dataset_dir = (args.dataset_dir or state_root / "objective_datasets").resolve()
    graph_path = (getattr(args, "graph_path", None) or state_root / "objective_graph.json").resolve()

    seen_fingerprints = set(split_csv(args.seen_fingerprint))
    if not args.repeat_existing:
        seen_fingerprints.update(discovery_fingerprints(discovery_dir))

    tracking_created = False
    ensured_goal_ids: list[str] = []
    if getattr(args, "ensure_tracking_document", False):
        tracking = ensure_objective_tracking_document(
            objective_path,
            ultimate_goal=getattr(args, "ultimate_goal", DEFAULT_ULTIMATE_GOAL),
            root_evidence=parse_root_evidence(getattr(args, "root_evidence", [])),
            root_goal_id=getattr(args, "root_goal_id", None),
            goal_prefix=getattr(args, "goal_prefix", None) or DEFAULT_GOAL_PREFIX,
            root_goal_title=getattr(args, "root_goal_title", DEFAULT_ROOT_GOAL_TITLE),
            document_title=getattr(args, "tracking_document_title", DEFAULT_TRACKING_DOCUMENT_TITLE),
        )
        tracking_created = tracking.created
        ensured_goal_ids = tracking.appended_goal_ids

    deduplicated_interoperability_goal_ids: list[str] = []
    if objective_path.exists():
        deduplicated_interoperability_goal_ids = deduplicate_interoperability_goals(objective_path)

    seeded_interoperability_goal_ids: list[str] = []
    if getattr(args, "seed_interoperability_goals", False) and objective_path.exists():
        interoperability = append_interoperability_goals(
            objective_path,
            repo_root=repo_root,
            focus=getattr(args, "interoperability_focus", []) or (),
            component_paths=getattr(args, "interoperability_component_path", []) or (),
            max_goals=getattr(args, "max_interoperability_goals", 12),
            goal_prefix=getattr(args, "goal_prefix", None),
        )
        seeded_interoperability_goal_ids = interoperability.appended_goal_ids

    seeded_launch_readiness_goal_ids: list[str] = []
    if getattr(args, "seed_launch_readiness_goals", False) and objective_path.exists():
        launch_readiness = append_launch_readiness_goals(
            objective_path,
            repo_root=repo_root,
            max_goals=getattr(args, "max_launch_readiness_goals", 8),
            goal_prefix=getattr(args, "goal_prefix", None),
        )
        seeded_launch_readiness_goal_ids = launch_readiness.appended_goal_ids

    completed_goal_ids: list[str] = []
    objective_completed_goal_count = 0
    objective_completion_validation_results: dict[str, Any] = {}
    goal_completion_todo_boards = parse_goal_completion_todo_boards(
        getattr(args, "objective_goal_completion_todo_board", []) or [],
        repo_root=repo_root,
        default_task_prefix=args.task_prefix,
    )
    if not getattr(args, "no_reconcile_goal_completion", False) and objective_path.exists():
        completion = reconcile_objective_goal_completion(
            repo_root=repo_root,
            objective_path=objective_path,
            todo_path=todo_path,
            task_header_prefix=args.task_prefix,
            todo_boards=goal_completion_todo_boards,
        )
        completed_goal_ids = completion.completed_goal_ids
        objective_completed_goal_count = completion.completed_goal_count
        objective_completion_validation_results = completion.validation_results

    refined_goal_ids: list[str] = []
    if getattr(args, "refine_objective_heap", False) and objective_path.exists():
        refinement_findings = scan_objective_gaps(
            repo_root,
            objective_path=objective_path,
            max_findings=args.max_findings,
            seen_fingerprints=seen_fingerprints,
        )
        refinement = append_refinement_goals(
            objective_path,
            refinement_findings,
            max_children_per_finding=getattr(args, "max_refinement_children", 3),
            max_depth=getattr(args, "max_refinement_depth", 4),
            goal_prefix=getattr(args, "goal_prefix", None),
        )
        refined_goal_ids = refinement.appended_goal_ids
        if refined_goal_ids:
            seen_fingerprints.update(finding.fingerprint for finding in refinement_findings)

    records = generate_objective_todos(
        repo_root=repo_root,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        bundle_dir=bundle_dir,
        dataset_dir=dataset_dir,
        task_prefix=args.task_prefix,
        depends_on=split_csv(args.depends_on),
        max_findings=args.max_findings,
        seen_fingerprints=seen_fingerprints,
        force_goal_ids=[
            *ensured_goal_ids,
            *seeded_interoperability_goal_ids,
            *seeded_launch_readiness_goal_ids,
            *refined_goal_ids,
            *split_csv(getattr(args, "force_goal_id", []) or []),
        ],
        persist_ast_dataset=not args.no_persist_ast_dataset,
        write_todo_vector_index=not getattr(args, "no_todo_vector_index", False),
        todo_vector_index_path=getattr(args, "todo_vector_index_path", None),
        surplus_findings_per_goal=getattr(args, "surplus_findings_per_goal", DEFAULT_SURPLUS_FINDINGS_PER_GOAL),
        surplus_min_terms_per_todo=getattr(args, "surplus_min_terms_per_todo", DEFAULT_SURPLUS_MIN_TERMS_PER_TODO),
        summary_prefix=getattr(args, "objective_summary_prefix", DEFAULT_OBJECTIVE_TASK_SUMMARY_PREFIX),
        discovery_output_path=getattr(args, "discovery_output_path", DEFAULT_DISCOVERY_OUTPUT_PATH),
    )
    plan_evaluation_path = (
        getattr(args, "plan_evaluation_path", None) or state_root / "plan_evaluations.json"
    ).resolve()
    router_config = None
    if bool(getattr(args, "generate_plan_branches", False)):
        from .task_proposal_router import StructuredPlanRouterConfig

        router_config = StructuredPlanRouterConfig(
            repo_root=repo_root,
            provider=str(getattr(args, "plan_router_provider", "") or "") or None,
            model=str(getattr(args, "plan_router_model", "gpt-5.3-codex-spark")),
            branch_count=max(1, int(getattr(args, "plan_branch_count", 3))),
            max_new_tokens=int(getattr(args, "plan_router_max_new_tokens", 4096)),
            timeout_seconds=int(getattr(args, "plan_router_timeout", 300)),
            allow_local_fallback=bool(getattr(args, "plan_router_allow_local_fallback", False)),
            temperature=float(getattr(args, "plan_router_temperature", 0.1)),
        )
    plan_decisions = plan_objective_records(
        records,
        branch_count=max(1, int(getattr(args, "plan_branch_count", 3))),
        router_config=router_config,
        use_llm_router=bool(getattr(args, "generate_plan_branches", False)),
    )
    persist_objective_plan_evaluations(
        plan_evaluation_path,
        plan_decisions,
        bundle_index_path=bundle_dir / "index.json",
    )
    graph_payload = write_objective_graph_artifact(objective_path=objective_path, graph_path=graph_path)

    bundle_index_path = bundle_dir / "index.json"
    submitted_bundle_task_ids: list[str] = []
    if args.submit_bundles and bundle_index_path.exists():
        submitted_bundle_task_ids = submit_bundle_tasks(
            bundle_index_path,
            queue_path=args.queue_path,
            task_type=args.queue_task_type,
            model_name=args.queue_model_name,
        )

    payload = {
        "schema": "ipfs_accelerate_py.agent_supervisor.objective_daemon",
        "repo_root": str(repo_root),
        "objective_path": repo_relative_path(repo_root, objective_path),
        "todo_path": repo_relative_path(repo_root, todo_path),
        "discovery_dir": repo_relative_path(repo_root, discovery_dir),
        "bundle_index_path": repo_relative_path(repo_root, bundle_index_path),
        "todo_vector_index_path": repo_relative_path(
            repo_root,
            (getattr(args, "todo_vector_index_path", None) or bundle_dir / "todo_vector_index.json").resolve(),
        ),
        "dataset_dir": repo_relative_path(repo_root, dataset_dir),
        "graph_path": repo_relative_path(repo_root, graph_path),
        "plan_evaluation_path": repo_relative_path(repo_root, plan_evaluation_path),
        "plan_evaluation_count": len(plan_decisions),
        "plan_router_branch_count": max(1, int(getattr(args, "plan_branch_count", 3))),
        "plan_router_enabled": bool(getattr(args, "generate_plan_branches", False)),
        "plan_router_fallback_count": sum(1 for item in plan_decisions if item.get("used_fallback")),
        "plan_router_error_count": sum(1 for item in plan_decisions if item.get("router_error")),
        "tracking_document_created": tracking_created,
        "ensured_goal_ids": ensured_goal_ids,
        "deduplicated_interoperability_goal_ids": deduplicated_interoperability_goal_ids,
        "seeded_interoperability_goal_ids": seeded_interoperability_goal_ids,
        "seeded_launch_readiness_goal_ids": seeded_launch_readiness_goal_ids,
        "completed_goal_ids": completed_goal_ids,
        "goal_completion_todo_boards": [
            {
                "todo_path": repo_relative_path(repo_root, path),
                "task_prefix": prefix,
            }
            for path, prefix in goal_completion_todo_boards
        ],
        "objective_completion_validation_results": objective_completion_validation_results,
        "refined_goal_ids": refined_goal_ids,
        "objective_goal_count": graph_payload["goal_count"],
        "objective_active_goal_count": graph_payload["active_goal_count"],
        "objective_completed_goal_count": graph_payload.get("completed_goal_count", objective_completed_goal_count),
        "objective_heap_schedule_count": len(graph_payload.get("heap_schedule") or []),
        "generated_count": len(records),
        "surplus_findings_per_goal": getattr(args, "surplus_findings_per_goal", DEFAULT_SURPLUS_FINDINGS_PER_GOAL),
        "surplus_min_terms_per_todo": getattr(args, "surplus_min_terms_per_todo", DEFAULT_SURPLUS_MIN_TERMS_PER_TODO),
        "task_ids": [record.task_id for record in records],
        "discovery_paths": [repo_relative_path(repo_root, record.discovery_path) for record in records],
        "bundle_keys": sorted({record.finding.bundle_key for record in records}),
        "submitted_bundle_task_ids": submitted_bundle_task_ids,
    }
    logger.info("Objective daemon generated %s tasks", len(records))
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    payload = run_objective_daemon(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
