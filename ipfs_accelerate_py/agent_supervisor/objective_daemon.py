"""CLI bridge for reusable objective-graph backlog generation.

This package-level entry point scans an objective heap, appends missing-evidence
tasks, persists AST records, writes parallel bundle shards, and optionally
submits those bundles to a local task queue.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

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
    append_refinement_goals,
    deduplicate_interoperability_goals,
    ensure_objective_tracking_document,
    parse_root_evidence,
    reconcile_objective_goal_completion,
    write_objective_graph_artifact,
)

logger = logging.getLogger(__name__)


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

    completed_goal_ids: list[str] = []
    objective_completed_goal_count = 0
    objective_completion_validation_results: dict[str, Any] = {}
    if not getattr(args, "no_reconcile_goal_completion", False) and objective_path.exists():
        completion = reconcile_objective_goal_completion(
            repo_root=repo_root,
            objective_path=objective_path,
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
        "tracking_document_created": tracking_created,
        "ensured_goal_ids": ensured_goal_ids,
        "deduplicated_interoperability_goal_ids": deduplicated_interoperability_goal_ids,
        "seeded_interoperability_goal_ids": seeded_interoperability_goal_ids,
        "completed_goal_ids": completed_goal_ids,
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
