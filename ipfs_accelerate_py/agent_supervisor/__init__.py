"""Autonomous agent supervisor helpers for objective-driven todo execution."""

from .dataset_store import DatasetArtifact, ObjectiveDatasetStore
from .objective_graph import (
    BundleWriteResult,
    ObjectiveFinding,
    ObjectiveGoal,
    ObjectiveHeapRecord,
    ObjectiveTaskRecord,
    assign_goal_subgoal_packets,
    collect_ast_dataset_records,
    build_bundle_task_payloads,
    generate_objective_todos,
    goal_graph,
    objective_heap_schedule,
    persist_objective_ast_dataset,
    parse_goal_heap,
    plan_semantic_ast_bundles,
    scan_objective_gaps,
    submit_bundle_tasks,
    write_bundle_shards,
)
from .objective_tracker import (
    ObjectiveCompletionResult,
    ObjectiveTrackingResult,
    RepositoryComponent,
    append_interoperability_goals,
    append_refinement_goals,
    build_objective_thought_graph,
    discover_gitlink_paths,
    discover_gitmodule_paths,
    discover_repository_components,
    discover_submodule_paths,
    ensure_objective_tracking_document,
    fibonacci_priority,
    reconcile_objective_goal_completion,
    run_goal_validation,
    write_objective_graph_artifact,
)
from .todo_vector_index import (
    TodoIndexRecord,
    build_execution_packet,
    build_execution_packets,
    cluster_records,
    parse_todo_vector_records,
    write_todo_vector_index,
)

__all__ = [
    "BundleWriteResult",
    "BundleLaneSpec",
    "BootstrapPathSpec",
    "CodebaseFinding",
    "ConfiguredCodebaseScanRecorder",
    "ConfiguredObjectiveBacklogRecorder",
    "ConfiguredRetryBudgetRecorder",
    "DatasetArtifact",
    "ObjectiveFinding",
    "ObjectiveGoal",
    "ObjectiveHeapRecord",
    "ObjectiveDatasetStore",
    "ObjectiveCompletionResult",
    "ObjectiveTrackingResult",
    "ObjectiveTaskRecord",
    "RepositoryComponent",
    "append_interoperability_goals",
    "append_refinement_goals",
    "android_validation_command_needs_environment",
    "android_validation_environment_contract",
    "apply_environment_contract",
    "assign_goal_subgoal_packets",
    "build_bundle_task_payloads",
    "build_execution_packet",
    "build_execution_packets",
    "build_merge_prompt",
    "build_objective_daemon_arg_parser",
    "build_objective_thought_graph",
    "build_bootstrap_path_ensurer",
    "build_bootstrap_path_resolver",
    "build_runtime_environment_callback",
    "collect_ast_dataset_records",
    "cluster_records",
    "generate_objective_todos",
    "ensure_objective_tracking_document",
    "discover_gitlink_paths",
    "discover_gitmodule_paths",
    "discover_repository_components",
    "discover_submodule_paths",
    "csv_tuple",
    "default_llm_merge_resolver_command",
    "env_csv_tuple",
    "environment_assignment_prefix",
    "ensure_named_directories",
    "ensure_task_blocks_present",
    "enforce_android_validation_environment",
    "ensure_runtime_pythonpath",
    "fibonacci_priority",
    "goal_graph",
    "objective_heap_schedule",
    "launch_bundle_lanes",
    "invoke_llm_resolver",
    "latest_failed_merge_event",
    "parse_goal_heap",
    "parse_todo_vector_records",
    "plan_semantic_ast_bundles",
    "plan_bundle_lanes",
    "persist_objective_ast_dataset",
    "record_configured_codebase_scan_findings",
    "record_configured_objective_backlog_findings",
    "record_configured_retry_budget_findings",
    "record_codebase_scan_findings",
    "record_objective_backlog_findings",
    "record_retry_budget_findings",
    "reconcile_objective_goal_completion",
    "repo_relative_or_default",
    "resolve_and_ensure_bootstrap_paths",
    "resolve_bootstrap_paths",
    "resolve_append_only_markdown_conflicts",
    "resolver_payload",
    "run_backlog_refinery",
    "run_goal_validation",
    "run_bundle_supervisor",
    "run_objective_daemon",
    "scan_codebase_findings",
    "scan_objective_gaps",
    "submit_bundle_tasks",
    "write_objective_graph_artifact",
    "write_bundle_shards",
    "write_bundle_lane_manifest",
    "write_todo_vector_index",
    "unique_path_entries",
    "with_android_validation_environment",
    "with_default",
    "with_flag_default",
    "with_repeated_default",
    "TodoIndexRecord",
    "merge_append_only_markdown_sections",
    "parse_supervisor_track_spec",
    "run_supervisor_tracks",
    "SupervisorTrack",
    "build_task_proposal_prompt",
    "build_task_proposal_prompt_builder",
    "build_task_proposal_router_cli_config",
    "build_portal_implementation_daemon_from_args",
    "build_portal_implementation_supervisor_from_args",
    "build_daemon_refill_hooks",
    "build_daemon_codebase_scan_refill_callback",
    "build_daemon_objective_refill_callback",
    "build_daemon_retry_budget_refill_callback",
    "build_supervisor_refill_hooks",
    "build_supervisor_codebase_scan_refill_callback",
    "build_supervisor_objective_refill_callback",
    "build_supervisor_retry_budget_refill_callback",
    "build_supervisor_runtime_callbacks",
    "bootstrap_runtime_environment",
    "configure_daemon_logging",
    "configure_supervisor_logging",
    "apply_merge_resolver_environment",
    "apply_portal_implementation_daemon_defaults",
    "apply_portal_implementation_supervisor_defaults",
    "implementation_state_paths",
    "run_portal_implementation_daemon_loop",
    "run_portal_implementation_supervisor",
    "run_configured_portal_implementation_daemon",
    "run_configured_portal_implementation_supervisor",
    "run_task_proposal_router",
    "run_task_proposal_router_cli",
    "rewrite_validation_commands",
    "select_proposal_task",
    "DaemonLoopHook",
    "ImplementationDaemonRunContext",
    "ImplementationDaemonDefaults",
    "ImplementationSupervisorRunContext",
    "ImplementationSupervisorDefaults",
    "ObjectiveRefillDefaults",
    "CodebaseRefillDefaults",
    "SupervisorRunHook",
    "SupervisorRuntimeCallbacks",
    "TaskProposalRouterConfig",
    "TaskProposalRouterCliConfig",
    "TaskProposalRouterError",
    "task_metadata_lines",
]


def __getattr__(name: str):
    if name in {
        "CodebaseFinding",
        "ConfiguredCodebaseScanRecorder",
        "ConfiguredObjectiveBacklogRecorder",
        "ConfiguredRetryBudgetRecorder",
        "ensure_task_blocks_present",
        "record_configured_codebase_scan_findings",
        "record_configured_objective_backlog_findings",
        "record_configured_retry_budget_findings",
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
    if name in {"merge_append_only_markdown_sections", "resolve_append_only_markdown_conflicts"}:
        from . import merge_conflict_repair

        return getattr(merge_conflict_repair, name)
    if name == "build_objective_daemon_arg_parser":
        from .objective_daemon import build_arg_parser

        return build_arg_parser
    if name == "run_objective_daemon":
        from .objective_daemon import run_objective_daemon

        return run_objective_daemon
    if name in {
        "default_llm_merge_resolver_command",
        "android_validation_command_needs_environment",
        "android_validation_environment_contract",
        "apply_environment_contract",
        "BootstrapPathSpec",
        "build_bootstrap_path_ensurer",
        "build_bootstrap_path_resolver",
        "bootstrap_runtime_environment",
        "build_runtime_environment_callback",
        "csv_tuple",
        "env_csv_tuple",
        "environment_assignment_prefix",
        "ensure_named_directories",
        "enforce_android_validation_environment",
        "ensure_runtime_pythonpath",
        "repo_relative_or_default",
        "resolve_and_ensure_bootstrap_paths",
        "resolve_bootstrap_paths",
        "rewrite_validation_commands",
        "unique_path_entries",
        "with_android_validation_environment",
        "with_default",
        "with_flag_default",
        "with_repeated_default",
    }:
        from . import wrapper_utils

        return getattr(wrapper_utils, name)
    if name in {"parse_supervisor_track_spec", "run_supervisor_tracks", "SupervisorTrack"}:
        from . import multi_supervisor_runner

        if name == "parse_supervisor_track_spec":
            return multi_supervisor_runner.parse_track_spec
        return getattr(multi_supervisor_runner, name)
    if name in {
        "build_portal_implementation_supervisor_from_args",
        "build_supervisor_codebase_scan_refill_callback",
        "build_supervisor_objective_refill_callback",
        "build_supervisor_refill_hooks",
        "build_supervisor_retry_budget_refill_callback",
        "build_supervisor_runtime_callbacks",
        "configure_supervisor_logging",
        "apply_portal_implementation_supervisor_defaults",
        "run_portal_implementation_supervisor",
        "run_configured_portal_implementation_supervisor",
        "ImplementationSupervisorRunContext",
        "ImplementationSupervisorDefaults",
        "ObjectiveRefillDefaults",
        "CodebaseRefillDefaults",
        "SupervisorRunHook",
        "SupervisorRuntimeCallbacks",
    }:
        from . import implementation_supervisor_runner

        return getattr(implementation_supervisor_runner, name)
    if name in {
        "build_portal_implementation_daemon_from_args",
        "build_daemon_codebase_scan_refill_callback",
        "build_daemon_objective_refill_callback",
        "build_daemon_refill_hooks",
        "build_daemon_retry_budget_refill_callback",
        "configure_daemon_logging",
        "apply_merge_resolver_environment",
        "apply_portal_implementation_daemon_defaults",
        "implementation_state_paths",
        "run_portal_implementation_daemon_loop",
        "run_configured_portal_implementation_daemon",
        "DaemonLoopHook",
        "ImplementationDaemonRunContext",
        "ImplementationDaemonDefaults",
    }:
        from . import implementation_daemon_runner

        return getattr(implementation_daemon_runner, name)
    if name in {
        "build_task_proposal_prompt",
        "build_task_proposal_prompt_builder",
        "build_task_proposal_router_cli_config",
        "run_task_proposal_router",
        "run_task_proposal_router_cli",
        "select_proposal_task",
        "TaskProposalRouterConfig",
        "TaskProposalRouterCliConfig",
        "TaskProposalRouterError",
        "task_metadata_lines",
    }:
        from . import task_proposal_router

        return getattr(task_proposal_router, name)
    raise AttributeError(name)
