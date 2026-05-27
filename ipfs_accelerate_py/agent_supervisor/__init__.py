"""Autonomous agent supervisor helpers for objective-driven todo execution."""

from .dataset_store import DatasetArtifact, ObjectiveDatasetStore
from .objective_graph import (
    BundleWriteResult,
    ObjectiveFinding,
    ObjectiveGoal,
    ObjectiveTaskRecord,
    collect_ast_dataset_records,
    build_bundle_task_payloads,
    generate_objective_todos,
    goal_graph,
    persist_objective_ast_dataset,
    parse_goal_heap,
    plan_semantic_ast_bundles,
    scan_objective_gaps,
    submit_bundle_tasks,
    write_bundle_shards,
)
from .objective_tracker import (
    ObjectiveTrackingResult,
    append_refinement_goals,
    ensure_objective_tracking_document,
    fibonacci_priority,
    write_objective_graph_artifact,
)
from .todo_vector_index import (
    TodoIndexRecord,
    cluster_records,
    parse_todo_vector_records,
    write_todo_vector_index,
)

__all__ = [
    "BundleWriteResult",
    "BundleLaneSpec",
    "CodebaseFinding",
    "DatasetArtifact",
    "ObjectiveFinding",
    "ObjectiveGoal",
    "ObjectiveDatasetStore",
    "ObjectiveTrackingResult",
    "ObjectiveTaskRecord",
    "append_refinement_goals",
    "build_bundle_task_payloads",
    "build_merge_prompt",
    "build_objective_daemon_arg_parser",
    "collect_ast_dataset_records",
    "cluster_records",
    "generate_objective_todos",
    "ensure_objective_tracking_document",
    "fibonacci_priority",
    "goal_graph",
    "launch_bundle_lanes",
    "invoke_llm_resolver",
    "latest_failed_merge_event",
    "parse_goal_heap",
    "parse_todo_vector_records",
    "plan_semantic_ast_bundles",
    "plan_bundle_lanes",
    "persist_objective_ast_dataset",
    "record_codebase_scan_findings",
    "record_objective_backlog_findings",
    "record_retry_budget_findings",
    "resolver_payload",
    "run_backlog_refinery",
    "run_bundle_supervisor",
    "run_objective_daemon",
    "scan_codebase_findings",
    "scan_objective_gaps",
    "submit_bundle_tasks",
    "write_objective_graph_artifact",
    "write_bundle_shards",
    "write_bundle_lane_manifest",
    "write_todo_vector_index",
    "TodoIndexRecord",
]


def __getattr__(name: str):
    if name in {
        "CodebaseFinding",
        "record_codebase_scan_findings",
        "record_objective_backlog_findings",
        "record_retry_budget_findings",
        "run_backlog_refinery",
        "scan_codebase_findings",
    }:
        from . import backlog_refinery

        return getattr(backlog_refinery, name)
    if name in {
        "BundleLaneSpec",
        "launch_bundle_lanes",
        "plan_bundle_lanes",
        "run_bundle_supervisor",
        "write_bundle_lane_manifest",
    }:
        from . import bundle_supervisor

        return getattr(bundle_supervisor, name)
    if name in {"build_merge_prompt", "invoke_llm_resolver", "latest_failed_merge_event", "resolver_payload"}:
        from . import merge_resolver

        return getattr(merge_resolver, name)
    if name == "build_objective_daemon_arg_parser":
        from .objective_daemon import build_arg_parser

        return build_arg_parser
    if name == "run_objective_daemon":
        from .objective_daemon import run_objective_daemon

        return run_objective_daemon
    raise AttributeError(name)
