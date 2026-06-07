from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path.cwd()

from ..checkout_lock import (
    checkout_lock_metadata,
    checkout_lock_owner_is_active,
    checkout_mutation_lock_path,
)
from ..event_log import append_jsonl_event, repair_jsonl_event_log, unique_backup_path
from ..merge_conflict_repair import resolve_append_only_markdown_conflicts
from .core import ManagedDaemonSpec
from .implementation_daemon import (
    DEFAULT_TRACKS,
    TASK_HEADER_PREFIX,
    PortalImplementationDaemon,
    PortalTask,
    PortalTaskState,
    load_json_dict,
    normalize_relative_path_list,
    parse_timestamp,
    process_command_line,
    process_is_running,
    state_file_repair_reason,
    utc_now,
    write_json_atomic,
    write_text_atomic,
)
from .supervisor import worktree_phase_worker_status
from .supervisor_loop import SupervisorLoop, SupervisorLoopConfig, SupervisorLoopDecision
from .supervisor_runtime import RestartPolicy

logger = logging.getLogger("ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor")

RECOVERABLE_SUPERVISOR_LOOP_STATUSES = {"child_exited", "launch_failed", "max_restarts_reached"}
DEFAULT_OBJECTIVE_SURPLUS_FINDINGS_PER_GOAL = int(
    os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_SURPLUS_FINDINGS_PER_GOAL", "3")
)
DEFAULT_OBJECTIVE_SURPLUS_MIN_TERMS_PER_TODO = int(
    os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_SURPLUS_MIN_TERMS_PER_TODO", "3")
)
DEFAULT_WORKTREE_SCAN_CACHE_TTL_SECONDS = float(
    os.environ.get("IPFS_ACCELERATE_AGENT_WORKTREE_SCAN_CACHE_TTL_SECONDS", "900")
)


def split_csv_values(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    items: list[str] = []
    for value in values:
        for raw_item in str(value).split(","):
            item = " ".join(raw_item.strip().split())
            if item and item.lower() not in {"none", "n/a"} and item not in items:
                items.append(item)
    return tuple(items)


@dataclass
class PortalSupervisorConfig:
    todo_path: Path
    state_path: Path
    strategy_path: Path
    events_path: Path
    state_dir: Path
    stale_seconds: float = 1800.0
    check_interval: float = 60.0
    max_restarts: int = 10
    daemon_interval: float = 300.0
    task_prefix: str = TASK_HEADER_PREFIX
    state_prefix: str = "portal"
    reconciliation_only: bool = False
    implement: bool = False
    implementation_command: str = ""
    llm_merge_resolver_command: str = ""
    llm_merge_resolver_timeout_seconds: float | None = None
    implementation_timeout: float = 1800.0
    implementation_log_stall_seconds: float = 300.0
    use_ephemeral_worktree: bool = True
    worktree_root: Path | None = None
    worktree_submodule_paths: tuple[str, ...] = field(default_factory=tuple)
    worktree_reconciliation_enabled: bool = True
    worktree_reconciliation_max_merges: int = 1
    worktree_reconciliation_dry_run: bool = False
    worktree_reconciliation_preflight_enabled: bool = True
    worktree_scan_cache_enabled: bool = True
    worktree_scan_cache_ttl_seconds: float = DEFAULT_WORKTREE_SCAN_CACHE_TTL_SECONDS
    worktree_scan_cache_path: Path | None = None
    retry_budget_guardrail_enabled: bool = True
    retry_budget_discovery_dir: Path | None = None
    retry_budget_discovery_output_path: str = ""
    validation_retry_budget: int = 3
    merge_retry_budget: int = 3
    implementation_retry_budget: int = 3
    retry_budget_commit_outputs: bool = False
    retry_budget_commit_subject: str = "Agent: record retry-budget guardrail outputs"
    dependency_guardrail_enabled: bool = True
    dependency_guardrail_discovery_dir: Path | None = None
    dependency_guardrail_discovery_output_path: str = ""
    dependency_guardrail_max_findings: int = 5
    dependency_guardrail_commit_outputs: bool = False
    dependency_guardrail_commit_subject: str = "Agent: record dependency guardrail outputs"
    reconciliation_guardrail_enabled: bool = True
    reconciliation_guardrail_discovery_dir: Path | None = None
    reconciliation_guardrail_discovery_output_path: str = ""
    reconciliation_guardrail_max_findings: int = 3
    reconciliation_guardrail_commit_outputs: bool = False
    reconciliation_guardrail_commit_subject: str = "Agent: record reconciliation guardrail outputs"
    codebase_refill_enabled: bool = False
    codebase_scan_discovery_dir: Path | None = None
    codebase_scan_discovery_output_path: str = ""
    codebase_scan_min_open_tasks: int = 0
    codebase_scan_max_findings: int = 5
    codebase_scan_cooldown_seconds: int = 21600
    codebase_scan_depends_on: tuple[str, ...] = field(default_factory=tuple)
    codebase_scan_skip_prefixes: tuple[str, ...] = field(default_factory=tuple)
    codebase_defer_when_objective_refills: bool = True
    codebase_scan_commit_outputs: bool = False
    codebase_scan_commit_subject: str = "Agent: record supervisor codebase scan findings"
    objective_refill_enabled: bool = False
    objective_path: Path | None = None
    objective_graph_path: Path | None = None
    objective_bundle_dir: Path | None = None
    objective_dataset_dir: Path | None = None
    objective_discovery_dir: Path | None = None
    objective_discovery_output_path: str = ""
    objective_summary_prefix: str = ""
    objective_refine_goals: bool = True
    objective_reconcile_goal_completion: bool = True
    objective_seed_interoperability_goals: bool = False
    objective_interoperability_focus: tuple[str, ...] = field(default_factory=tuple)
    objective_interoperability_component_paths: tuple[str, ...] = field(default_factory=tuple)
    objective_max_interoperability_goals: int = 12
    objective_ensure_tracking_document: bool = False
    objective_ultimate_goal: str = ""
    objective_root_evidence: tuple[str, ...] = field(default_factory=tuple)
    objective_goal_prefix: str | None = None
    objective_root_goal_id: str | None = None
    objective_root_goal_title: str = ""
    objective_tracking_document_title: str = ""
    objective_scan_min_open_tasks: int = 0
    objective_scan_max_findings: int = 5
    objective_scan_cooldown_seconds: int = 21600
    objective_scan_depends_on: tuple[str, ...] = field(default_factory=tuple)
    objective_max_refinement_children: int = 3
    objective_max_refinement_depth: int = 4
    objective_persist_ast_dataset: bool = True
    objective_write_todo_vector_index: bool = True
    objective_todo_vector_index_path: Path | None = None
    objective_surplus_findings_per_goal: int = DEFAULT_OBJECTIVE_SURPLUS_FINDINGS_PER_GOAL
    objective_surplus_min_terms_per_todo: int = DEFAULT_OBJECTIVE_SURPLUS_MIN_TERMS_PER_TODO
    repo_root: Path = field(default_factory=Path.cwd)
    daemon_script_path: Path | None = None
    supervisor_script_path: Path | None = None


class AdoptedManagedDaemonProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode: int | None = None

    def poll(self) -> int | None:
        if process_is_running(self.pid):
            return None
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        if self.poll() is not None:
            return
        try:
            os.kill(self.pid, signal.SIGTERM)
            self.returncode = -signal.SIGTERM
        except ProcessLookupError:
            self.returncode = 0

    def kill(self) -> None:
        if self.poll() is not None:
            return
        try:
            os.kill(self.pid, signal.SIGKILL)
            self.returncode = -signal.SIGKILL
        except ProcessLookupError:
            self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        deadline = None if timeout is None else time.time() + timeout
        while True:
            polled = self.poll()
            if polled is not None:
                return polled
            if deadline is not None and time.time() >= deadline:
                raise subprocess.TimeoutExpired(cmd=["pid", str(self.pid)], timeout=timeout)
            time.sleep(0.2)


class PortalImplementationSupervisor:
    shared_supervisor_loop_class = SupervisorLoop
    shared_supervisor_loop_config_class = SupervisorLoopConfig
    shared_managed_daemon_spec_class = ManagedDaemonSpec

    def __init__(self, config: PortalSupervisorConfig) -> None:
        self.config = config
        self.restart_count = 0
        self.last_start_at: float | None = None

    def run_once(self) -> dict[str, Any]:
        event_log_repair = self.ensure_event_log_file()
        main_checkout_repair = self.repair_main_checkout_merge_state()
        worktree_reconciliation = self.reconcile_backlogged_worktrees()
        worktree_cleanup = self.cleanup_backlogged_worktrees()
        strategy_file_repair = self.ensure_strategy_file()
        state_file_repair = self.ensure_state_file()
        todo_board_repair = self.ensure_todo_board_for_refill()
        reconciliation_findings = self.record_reconciliation_guardrails(
            worktree_reconciliation,
            worktree_cleanup,
        )
        guardrail_releases = self.release_completed_guardrail_blocks()
        state = PortalTaskState.load(self.config.state_path)
        now_ts = time.time()
        stuck, reason = self.is_stuck(state, now_ts=now_ts)
        if stuck:
            retry_budget_findings = self.record_retry_budget_guardrails()
            dependency_findings = self.record_dependency_guardrails()
            strategy = self.rewrite_strategy(state, reason)
            state_repair = self.repair_blocked_progress_state(state, reason, now_ts=now_ts)
            return {
                "stuck": True,
                "reason": reason,
                "retry_budget_count": len(retry_budget_findings),
                "dependency_guardrail_count": len(dependency_findings),
                "reconciliation_guardrail_count": len(reconciliation_findings),
                "strategy_generation": int(strategy.get("generation", 0)),
                "active_task_id": state.active_task_id,
                "state_repair": state_repair,
                "event_log_repair": event_log_repair,
                "strategy_file_repair": strategy_file_repair,
                "state_file_repair": state_file_repair,
                "todo_board_repair": todo_board_repair,
                "main_checkout_repair": main_checkout_repair,
                "worktree_reconciliation": worktree_reconciliation,
                "worktree_cleanup": worktree_cleanup,
                "guardrail_unblock_count": len(guardrail_releases),
            }
        retry_budget_findings = self.record_retry_budget_guardrails()
        dependency_findings = self.record_dependency_guardrails()
        objective_payload = self.refill_objective_backlog()
        objective_generated_count = int(objective_payload.get("generated_count") or 0)
        objective_refined_goal_count = len(objective_payload.get("refined_goal_ids") or [])
        objective_seeded_goal_count = len(objective_payload.get("seeded_interoperability_goal_ids") or [])
        codebase_deferred_reason = ""
        if (
            self.config.codebase_defer_when_objective_refills
            and self.config.objective_refill_enabled
            and (objective_generated_count > 0 or objective_refined_goal_count > 0 or objective_seeded_goal_count > 0)
        ):
            codebase_findings = []
            codebase_deferred_reason = "objective_refill_produced_goal_work"
        else:
            codebase_findings = self.refill_codebase_backlog()
        self._record_event(
            "supervisor_check",
            {
                "stuck": False,
                "active_task_id": state.active_task_id,
                "completed_count": state.completed_count,
                "worktree_reconciliation_candidate_count": int(
                    worktree_reconciliation.get("candidate_count") or 0
                ),
                "worktree_reconciliation_processed_count": int(
                    worktree_reconciliation.get("processed_count") or 0
                ),
                "worktree_reconciliation_reconciled_count": int(
                    worktree_reconciliation.get("reconciled_count") or 0
                ),
                "worktree_reconciliation_preflight_blocked_count": int(
                    worktree_reconciliation.get("preflight_blocked_count") or 0
                ),
                "worktree_cleanup_removed_count": int(worktree_cleanup.get("removed_count") or 0),
                "worktree_cleanup_dirty_group_count": len(
                    worktree_cleanup.get("dirty_worktree_groups") or {}
                ),
                "retry_budget_count": len(retry_budget_findings),
                "dependency_guardrail_count": len(dependency_findings),
                "reconciliation_guardrail_count": len(reconciliation_findings),
                "guardrail_unblock_count": len(guardrail_releases),
                "objective_refill_count": objective_generated_count,
                "objective_refined_goal_count": objective_refined_goal_count,
                "objective_seeded_interoperability_goal_count": objective_seeded_goal_count,
                "codebase_refill_count": len(codebase_findings),
                "codebase_deferred_reason": codebase_deferred_reason,
            },
        )
        return {
            "stuck": False,
            "active_task_id": state.active_task_id,
            "completed_count": state.completed_count,
            "retry_budget_count": len(retry_budget_findings),
            "dependency_guardrail_count": len(dependency_findings),
            "reconciliation_guardrail_count": len(reconciliation_findings),
            "guardrail_unblock_count": len(guardrail_releases),
            "objective_refill_count": objective_generated_count,
            "objective_refined_goal_count": objective_refined_goal_count,
            "objective_seeded_interoperability_goal_count": objective_seeded_goal_count,
            "codebase_refill_count": len(codebase_findings),
            "codebase_deferred_reason": codebase_deferred_reason,
            "event_log_repair": event_log_repair,
            "strategy_file_repair": strategy_file_repair,
            "state_file_repair": state_file_repair,
            "todo_board_repair": todo_board_repair,
            "main_checkout_repair": main_checkout_repair,
            "worktree_reconciliation": worktree_reconciliation,
            "worktree_cleanup": worktree_cleanup,
        }

    def run_forever(self) -> None:
        self.ensure_event_log_file()
        self.repair_main_checkout_merge_state()
        self.ensure_managed_daemon_pid_file()
        while True:
            loop = self.shared_supervisor_loop_class(
                self.build_supervisor_loop_config(),
                watchdog_hook=self._supervisor_loop_watchdog_decision,
            )
            result = loop.run()
            self.restart_count = result.restart_count
            result_payload = {
                "status": result.status,
                "restart_count": result.restart_count,
                "last_exit_code": result.last_exit_code,
                "last_recycle_reason": result.last_recycle_reason,
                "last_run_id": result.last_run_id,
                "last_log_path": result.last_log_path,
            }
            self._record_event("supervisor_loop_finished", result_payload)
            if result.status not in RECOVERABLE_SUPERVISOR_LOOP_STATUSES:
                return

            try:
                recovery = self.run_once()
            except Exception as exc:
                recovery = {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                logger.warning("Supervisor recovery pass failed; restarting child loop anyway", exc_info=True)
                self._record_event(
                    "supervisor_loop_recovery_failed",
                    {
                        "loop_result": result_payload,
                        "recovery": recovery,
                    },
                )
            else:
                self._record_event(
                    "supervisor_loop_recovery_pass",
                    {
                        "loop_result": result_payload,
                        "recovery": recovery,
                    },
                )

            delay_seconds = self._supervisor_loop_recovery_delay_seconds()
            self._record_event(
                "supervisor_loop_restarting_after_recovery",
                {
                    "loop_result": result_payload,
                    "delay_seconds": delay_seconds,
                },
            )
            time.sleep(delay_seconds)

    def _supervisor_loop_recovery_delay_seconds(self) -> float:
        """Back off between outer loop recovery attempts without exceeding one check interval."""

        return max(5.0, min(float(self.config.check_interval), 60.0))

    def build_supervisor_loop_config(self) -> SupervisorLoopConfig:
        command = tuple(self._build_daemon_command())
        prefix = self.config.state_prefix
        spec = ManagedDaemonSpec(
            name=f"{prefix}-implementation-daemon",
            schema="ipfs_accelerate_py.agent_supervisor.todo_implementation_supervisor",
            repo_root=self.config.repo_root,
            daemon_dir=self.config.state_dir,
            runner=command,
            status_path=self.config.state_path,
            progress_path=self.config.state_path,
            result_log_path=self.config.events_path,
            task_board_path=self.config.todo_path,
            supervisor_status_path=self.config.state_dir / f"{prefix}_supervisor_status.json",
            supervisor_pid_path=self.config.state_dir / f"{prefix}_supervisor.pid",
            child_pid_path=self._managed_daemon_pid_path(),
            supervisor_out_path=self.config.state_dir / f"{prefix}_supervisor.out",
            ensure_status_path=self.config.state_dir / f"{prefix}_ensure_status.json",
            ensure_check_path=self.config.state_dir / f"{prefix}_ensure_check.json",
            supervisor_lock_path=self.config.state_dir / f"{prefix}_supervisor.lock",
            latest_log_path=self.config.state_dir / f"{prefix}_managed_daemon.latest.log",
            daemon_process_match_all=command,
            worktree_root=self.config.worktree_root,
        )
        return SupervisorLoopConfig(
            spec=spec,
            command=command,
            log_prefix=f"{prefix}_implementation_daemon",
            restart_policy=RestartPolicy(
                restart_backoff_seconds=max(0.0, float(self.config.check_interval)),
                fast_restart_backoff_seconds=min(2.0, max(0.0, float(self.config.check_interval))),
            ),
            heartbeat_seconds=max(0.01, float(self.config.check_interval)),
            poll_seconds=min(1.0, max(0.01, float(self.config.check_interval))),
            watchdog_stale_after_seconds=max(0.0, float(self.config.stale_seconds)),
            watchdog_startup_grace_seconds=max(30.0, float(self.config.check_interval) * 2.0),
            stop_grace_seconds=15.0,
            max_restarts=max(0, int(self.config.max_restarts)),
            status_static_fields={
                "todo_path": str(self.config.todo_path),
                "state_path": str(self.config.state_path),
                "task_prefix": self.config.task_prefix,
                "state_prefix": self.config.state_prefix,
                "worktree_no_child_stall_seconds": max(
                    0.0,
                    float(self.config.implementation_log_stall_seconds),
                ),
            },
        )

    def _supervisor_loop_watchdog_decision(
        self,
        _loop: SupervisorLoop,
        _child: Any,
        _current_status: dict[str, Any],
    ) -> SupervisorLoopDecision:
        self.ensure_event_log_file()
        main_checkout_repair = self.repair_main_checkout_merge_state()
        if main_checkout_repair.get("repaired"):
            return SupervisorLoopDecision.recycle(
                "main_checkout_merge_state_repaired",
                detail=main_checkout_repair,
            )
        worktree_reconciliation = self.reconcile_backlogged_worktrees()
        if worktree_reconciliation.get("reconciled_count"):
            self._record_event("backlogged_worktree_reconciliation", worktree_reconciliation)
        worktree_cleanup = self.cleanup_backlogged_worktrees()
        if worktree_cleanup.get("removed_count"):
            self._record_event("backlogged_worktree_cleanup", worktree_cleanup)
        reconciliation_findings = self.record_reconciliation_guardrails(
            worktree_reconciliation,
            worktree_cleanup,
        )
        state = PortalTaskState.load(self.config.state_path)
        stuck, reason = self.is_stuck(state, now_ts=time.time())
        if not stuck:
            self.ensure_strategy_file()
            self.ensure_state_file()
            self.ensure_todo_board_for_refill()
            self.release_completed_guardrail_blocks()
            self.record_retry_budget_guardrails()
            self.record_dependency_guardrails()
            self.refill_objective_backlog()
            self.refill_codebase_backlog()
            return SupervisorLoopDecision.keep_running()
        self.ensure_strategy_file()
        self.ensure_state_file()
        self.ensure_todo_board_for_refill()
        self.release_completed_guardrail_blocks()
        self.record_retry_budget_guardrails()
        self.record_dependency_guardrails()
        self.rewrite_strategy(state, reason)
        active_task_id = state.active_task_id
        self.repair_blocked_progress_state(state, reason, now_ts=time.time())
        return SupervisorLoopDecision.recycle(reason, detail={"active_task_id": active_task_id})

    def repair_main_checkout_merge_state(self) -> dict[str, Any]:
        """Resolve or abort an interrupted merge in the shared repository checkout."""

        repo_root = self.config.repo_root
        merge_head = self._git_merge_head(repo_root)
        unmerged_paths = self._git_unmerged_paths(repo_root)
        if not merge_head and not unmerged_paths:
            return {"attempted": False, "repaired": False, "reason": "clean", "path": str(repo_root)}

        lock_path = self._repo_merge_lock_path()
        lock_fd, lock_reason, existing_lock = self._try_acquire_checkout_lock(lock_path)
        if lock_fd is None:
            result: dict[str, Any] = {
                "attempted": True,
                "repaired": False,
                "path": str(repo_root),
                "merge_in_progress": bool(merge_head),
                "merge_head": merge_head,
                "initial_unmerged_paths": unmerged_paths,
                "status_short": self._git_status_short(repo_root),
                "reason": f"checkout_mutation_{lock_reason}",
                "lock_path": str(lock_path),
            }
            if existing_lock:
                result["lock_owner_pid"] = int(existing_lock.get("pid") or 0)
                result["lock_owner_task_id"] = str(existing_lock.get("task_id") or "")
                result["lock_owner_branch"] = str(existing_lock.get("branch") or "")
            self._record_event("main_checkout_merge_state_repair_deferred", result)
            return result

        self._write_checkout_lock_metadata(
            lock_fd,
            checkout_lock_metadata(
                kind="merge",
                repo_root=repo_root,
                task_id=self._active_task_id_for_lock(),
                branch="supervisor-main-checkout-repair",
                extra={"operation": "repair_main_checkout_merge_state", "started_at": utc_now()},
            ),
        )
        try:
            return self._repair_main_checkout_merge_state_locked(
                repo_root,
                merge_head=merge_head,
                unmerged_paths=unmerged_paths,
            )
        finally:
            try:
                if lock_path.exists():
                    lock_path.unlink()
            except OSError:
                logger.warning("Failed to remove checkout mutation lock %s", lock_path)

    def _repair_main_checkout_merge_state_locked(
        self,
        repo_root: Path,
        *,
        merge_head: str,
        unmerged_paths: list[str],
    ) -> dict[str, Any]:
        """Resolve or abort an interrupted merge after acquiring the checkout lock."""

        result: dict[str, Any] = {
            "attempted": True,
            "repaired": False,
            "path": str(repo_root),
            "merge_in_progress": bool(merge_head),
            "merge_head": merge_head,
            "initial_unmerged_paths": unmerged_paths,
            "status_short": self._git_status_short(repo_root),
        }
        if not merge_head:
            result["reason"] = "unmerged_paths_without_merge_head"
            self._record_event("main_checkout_merge_state_repair", result)
            return result

        deterministic_repair = self.repair_generated_main_checkout_conflicts(repo_root)
        if deterministic_repair:
            result["deterministic_conflict_repair"] = deterministic_repair
            if not self._git_unmerged_paths(repo_root):
                commit_result = self._commit_supervisor_resolved_merge(repo_root)
                result["commit_result"] = commit_result
                if commit_result.get("completed") or commit_result.get("reason") == "resolver_committed_merge":
                    result.update(
                        {
                            "repaired": True,
                            "reason": "deterministic_generated_markdown_conflict_repair",
                            "final_unmerged_paths": [],
                            "merge_in_progress_after": bool(self._git_merge_head(repo_root)),
                        }
                    )
                    self._record_event("main_checkout_merge_state_repair", result)
                    return result

        if self.config.llm_merge_resolver_command:
            llm_result = self._invoke_main_checkout_merge_resolver(
                repo_root,
                merge_head=merge_head,
                unmerged_paths=unmerged_paths,
            )
            result["llm_merge_resolver"] = self._compact_resolver_result(llm_result)
            if self._git_merge_head(repo_root):
                commit_result = self._commit_supervisor_resolved_merge(repo_root)
                result["commit_result"] = commit_result
                if commit_result.get("completed") or commit_result.get("reason") == "resolver_committed_merge":
                    result.update(
                        {
                            "repaired": True,
                            "reason": "llm_resolved_merge",
                            "final_unmerged_paths": self._git_unmerged_paths(repo_root),
                            "merge_in_progress_after": bool(self._git_merge_head(repo_root)),
                        }
                    )
                    self._record_event("main_checkout_merge_state_repair", result)
                    return result
            elif not self._git_unmerged_paths(repo_root):
                result.update(
                    {
                        "repaired": True,
                        "reason": "llm_resolver_completed_merge",
                        "final_unmerged_paths": [],
                        "merge_in_progress_after": False,
                    }
                )
                self._record_event("main_checkout_merge_state_repair", result)
                return result

        post_unmerged_paths = self._git_unmerged_paths(repo_root)
        post_merge_head = self._git_merge_head(repo_root)
        if post_merge_head:
            abort_result = self._abort_main_checkout_merge(repo_root)
            result["abort_result"] = abort_result
            result["repaired"] = bool(abort_result.get("aborted"))
            result["reason"] = (
                "merge_aborted_after_resolver_failed"
                if self.config.llm_merge_resolver_command
                else "merge_aborted_without_resolver"
            )
        else:
            result["reason"] = "merge_no_longer_in_progress"
            result["repaired"] = not post_unmerged_paths
        result["final_unmerged_paths"] = self._git_unmerged_paths(repo_root)
        result["merge_in_progress_after"] = bool(self._git_merge_head(repo_root))
        self._record_event("main_checkout_merge_state_repair", result)
        return result

    def _active_task_id_for_lock(self) -> str:
        try:
            return PortalTaskState.load(self.config.state_path).active_task_id
        except Exception:
            return ""

    def _repo_merge_lock_path(self) -> Path:
        return checkout_mutation_lock_path(self.config.repo_root)

    def _checkout_lock_owner_is_active(self, metadata: dict[str, Any]) -> bool:
        return checkout_lock_owner_is_active(
            metadata,
            expected_kind="merge",
            expected_repo_root=self.config.repo_root,
            process_command_line=process_command_line,
            process_is_running=process_is_running,
        )

    def _try_acquire_checkout_lock(self, lock_path: Path) -> tuple[int | None, str, dict[str, Any] | None]:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                return os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY), "acquired", None
            except FileExistsError:
                existing = load_json_dict(lock_path)
                if existing is not None and self._checkout_lock_owner_is_active(existing):
                    return None, "lock_exists", existing
                if not self._clear_stale_checkout_lock(lock_path, metadata=existing):
                    return None, "lock_cleanup_failed", existing
        existing = load_json_dict(lock_path)
        if existing is not None and self._checkout_lock_owner_is_active(existing):
            return None, "lock_exists", existing
        return None, "lock_unavailable", existing

    def _write_checkout_lock_metadata(self, lock_fd: int, metadata: dict[str, Any]) -> None:
        try:
            os.write(lock_fd, json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8"))
        finally:
            os.close(lock_fd)

    def _clear_stale_checkout_lock(self, lock_path: Path, *, metadata: dict[str, Any] | None) -> bool:
        moved_directory_path = ""
        try:
            if lock_path.is_dir():
                backup_path = unique_backup_path(lock_path, "directory-backup")
                lock_path.rename(backup_path)
                moved_directory_path = str(backup_path)
            else:
                lock_path.unlink()
        except FileNotFoundError:
            return True
        except OSError:
            logger.warning("Failed to remove stale checkout mutation lock %s", lock_path)
            return False
        event = {
            "lock_path": str(lock_path),
            "lock_owner_pid": int(metadata.get("pid") or 0) if metadata else 0,
            "task_id": str(metadata.get("task_id") or "") if metadata else "",
            "branch": str(metadata.get("branch") or "") if metadata else "",
        }
        if moved_directory_path:
            event["moved_directory_path"] = moved_directory_path
        self._record_event("checkout_mutation_lock_cleared", event)
        return True

    def repair_generated_main_checkout_conflicts(self, repo_root: Path) -> list[dict[str, object]]:
        """Resolve configured append-only generated markdown conflicts without LLM calls."""

        allowed_paths, allowed_dirs = self._append_only_markdown_conflict_targets()
        if not allowed_paths and not allowed_dirs:
            return []
        repairs = resolve_append_only_markdown_conflicts(
            repo_root=repo_root,
            allowed_paths=allowed_paths,
            allowed_dirs=allowed_dirs,
        )
        if repairs:
            self._record_event(
                "generated_markdown_conflict_repair",
                {
                    "repo_root": str(repo_root),
                    "results": repairs,
                },
            )
        return repairs

    def _append_only_markdown_conflict_targets(self) -> tuple[list[Path], list[Path]]:
        allowed_paths: list[Path] = []
        allowed_dirs: list[Path] = []
        if self.config.objective_path is not None:
            allowed_paths.append(self.config.objective_path)
        if self.config.objective_bundle_dir is not None:
            allowed_dirs.append(self.config.objective_bundle_dir)
        return allowed_paths, allowed_dirs

    def _invoke_main_checkout_merge_resolver(
        self,
        repo_root: Path,
        *,
        merge_head: str,
        unmerged_paths: list[str],
    ) -> dict[str, Any]:
        from ipfs_accelerate_py.agent_supervisor.merge_resolver import build_merge_prompt, invoke_llm_resolver

        target_branch = self._git_current_branch(repo_root) or "HEAD"
        active_task_id = ""
        try:
            active_task_id = PortalTaskState.load(self.config.state_path).active_task_id
        except Exception:
            active_task_id = ""
        merge_result = {
            "attempted": True,
            "merged": False,
            "returncode": 1,
            "branch": merge_head,
            "target_branch": target_branch,
            "command": ["git", "status", "--short"],
            "reason": "supervisor_main_checkout_merge_in_progress",
            "stdout": "\n".join(self._git_status_short(repo_root)),
            "stderr": "",
            "main_worktree_path": str(repo_root),
            "dirty_paths": unmerged_paths,
        }
        event = {
            "type": "supervisor_main_checkout_merge_repair",
            "task_id": active_task_id or self.config.state_prefix,
            "attempt": 0,
            "merge_result": merge_result,
        }
        payload = {
            "found": True,
            "task_id": active_task_id,
            "attempt": 0,
            "events_path": str(self.config.events_path),
            "repo_root": str(repo_root),
            "branch": merge_head,
            "target_branch": target_branch,
            "command": merge_result["command"],
            "reason": merge_result["reason"],
            "dirty_paths": unmerged_paths,
            "unmerged_paths": unmerged_paths,
            "prompt": build_merge_prompt(event=event, repo_root=repo_root),
        }
        return invoke_llm_resolver(
            payload,
            command_template=self.config.llm_merge_resolver_command,
            timeout_seconds=self.config.llm_merge_resolver_timeout_seconds,
        )

    @staticmethod
    def _compact_resolver_result(result: dict[str, Any]) -> dict[str, Any]:
        compact = dict(result)
        if "prompt" in compact:
            compact["prompt_chars"] = len(str(compact.pop("prompt") or ""))
        return compact

    def _commit_supervisor_resolved_merge(self, repo_root: Path) -> dict[str, Any]:
        unresolved = self._git_unmerged_paths(repo_root)
        if unresolved:
            return {
                "attempted": True,
                "completed": False,
                "reason": "unresolved_paths_remain",
                "unresolved_paths": unresolved,
            }
        if not self._git_merge_head(repo_root):
            return {
                "attempted": False,
                "completed": True,
                "reason": "resolver_committed_merge",
            }
        add = subprocess.run(
            ["git", "add", "-A"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if add.returncode != 0:
            return {
                "attempted": True,
                "completed": False,
                "reason": "stage_resolved_merge_failed",
                "returncode": add.returncode,
                "stdout": add.stdout[-4000:],
                "stderr": add.stderr[-4000:],
            }
        commit = subprocess.run(
            [
                "git",
                "-c",
                "user.name=Implementation Supervisor",
                "-c",
                "user.email=implementation-supervisor@example.invalid",
                "commit",
                "--no-edit",
            ],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        return {
            "attempted": True,
            "completed": commit.returncode == 0,
            "reason": "committed" if commit.returncode == 0 else "commit_failed",
            "returncode": commit.returncode,
            "stdout": commit.stdout[-4000:],
            "stderr": commit.stderr[-4000:],
        }

    def _abort_main_checkout_merge(self, repo_root: Path) -> dict[str, Any]:
        if not self._git_merge_head(repo_root):
            return {"attempted": False, "aborted": False, "reason": "no_merge_in_progress"}
        abort = subprocess.run(
            ["git", "merge", "--abort"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        return {
            "attempted": True,
            "aborted": abort.returncode == 0,
            "returncode": abort.returncode,
            "stdout": abort.stdout[-4000:],
            "stderr": abort.stderr[-4000:],
        }

    @staticmethod
    def _git_merge_head(repo_root: Path) -> str:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", "MERGE_HEAD"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return ""
        return result.stdout.strip()

    @staticmethod
    def _git_unmerged_paths(repo_root: Path) -> list[str]:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=U"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return []
        return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())

    @staticmethod
    def _git_status_short(repo_root: Path) -> list[str]:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return []
        return [line.rstrip() for line in result.stdout.splitlines() if line.strip()]

    @staticmethod
    def _git_current_branch(repo_root: Path) -> str:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return ""
        return result.stdout.strip()

    @staticmethod
    def _git_ref_commit(repo_root: Path, ref: str) -> str:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return ""
        return result.stdout.strip()

    def _worktree_scan_cache_path(self) -> Path:
        return self.config.worktree_scan_cache_path or (
            self.config.state_dir / f"{self.config.state_prefix}_worktree_scan_cache.json"
        )

    def _load_worktree_scan_cache(self) -> dict[str, Any]:
        if (
            not self.config.worktree_scan_cache_enabled
            or self.config.worktree_scan_cache_ttl_seconds <= 0
        ):
            return {"enabled": False, "entries": {}}
        payload = load_json_dict(self._worktree_scan_cache_path())
        entries = payload.get("entries") if isinstance(payload, dict) else {}
        if not isinstance(entries, dict):
            entries = {}
        return {
            "enabled": True,
            "version": 1,
            "entries": {
                str(key): dict(value)
                for key, value in entries.items()
                if isinstance(value, dict)
            },
        }

    def _write_worktree_scan_cache(self, cache: dict[str, Any]) -> bool:
        if not cache.get("enabled") or not cache.get("_changed"):
            return False
        entries = cache.get("entries")
        if not isinstance(entries, dict):
            entries = {}
        now = time.time()
        ttl = max(float(self.config.worktree_scan_cache_ttl_seconds), 0.0)
        max_age = max(ttl * 4, ttl + 3600.0)
        pruned_entries = {
            str(key): value
            for key, value in entries.items()
            if isinstance(value, dict)
            and now - float(value.get("updated_at_epoch") or 0.0) <= max_age
        }
        path = self._worktree_scan_cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            path,
            {
                "version": 1,
                "updated_at_epoch": now,
                "updated_at": utc_now(),
                "entries": pruned_entries,
            },
        )
        return True

    @staticmethod
    def _worktree_scan_cache_key(
        *,
        phase: str,
        path: Path,
        branch: str,
        head: str,
        target_signature: str,
    ) -> str:
        return sha1(
            json.dumps(
                {
                    "phase": phase,
                    "path": str(path),
                    "branch": branch,
                    "head": head,
                    "target_signature": target_signature,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    def _worktree_scan_cache_entry(
        self,
        cache: dict[str, Any],
        *,
        phase: str,
        path: Path,
        branch: str,
        head: str,
        target_signature: str,
    ) -> dict[str, Any] | None:
        if not cache.get("enabled"):
            return None
        key = self._worktree_scan_cache_key(
            phase=phase,
            path=path,
            branch=branch,
            head=head,
            target_signature=target_signature,
        )
        entries = cache.get("entries")
        entry = entries.get(key) if isinstance(entries, dict) else None
        if not isinstance(entry, dict):
            return None
        if time.time() - float(entry.get("updated_at_epoch") or 0.0) > float(
            self.config.worktree_scan_cache_ttl_seconds
        ):
            return None
        return dict(entry)

    def _store_worktree_scan_cache_entry(
        self,
        cache: dict[str, Any],
        *,
        phase: str,
        path: Path,
        branch: str,
        head: str,
        target_signature: str,
        classification: str,
        payload: Mapping[str, Any],
    ) -> None:
        if not cache.get("enabled"):
            return
        key = self._worktree_scan_cache_key(
            phase=phase,
            path=path,
            branch=branch,
            head=head,
            target_signature=target_signature,
        )
        entries = cache.setdefault("entries", {})
        if not isinstance(entries, dict):
            cache["entries"] = entries = {}
        now = time.time()
        entries[key] = {
            "phase": phase,
            "path": str(path),
            "branch": branch,
            "head": head,
            "target_signature": target_signature,
            "classification": classification,
            "payload": dict(payload),
            "updated_at_epoch": now,
            "updated_at": utc_now(),
        }
        cache["_changed"] = True

    def reconcile_backlogged_worktrees(self) -> dict[str, Any]:
        """Retry clean inactive implementation worktrees before cleanup."""

        if not self.config.worktree_reconciliation_enabled:
            return {"attempted": False, "reason": "worktree_reconciliation_disabled"}
        worktree_root = self.config.worktree_root
        if worktree_root is None:
            return {"attempted": False, "reason": "worktree_root_not_configured"}

        repo_root = self.config.repo_root
        records = self._git_worktree_records(repo_root)
        try:
            root_resolved = worktree_root.resolve()
        except OSError:
            root_resolved = worktree_root
        process_lines = self._list_process_commands()
        active_worktree = ""
        try:
            active_worktree = PortalTaskState.load(self.config.state_path).active_worktree_path
        except Exception:
            active_worktree = ""
        target_ref = self._git_current_branch(repo_root) or "HEAD"
        target_signature = self._git_ref_commit(repo_root, target_ref) or target_ref
        raw_main_status = self._main_status_for_worktree_reconciliation(repo_root, worktree_root)
        raw_main_dirty_evidence = (
            self._main_checkout_dirty_evidence(repo_root, raw_main_status)
            if raw_main_status
            else {}
        )
        main_status, main_dirty_evidence = self._filter_generated_main_checkout_status(
            raw_main_status,
            raw_main_dirty_evidence,
        )
        max_merges = max(0, int(self.config.worktree_reconciliation_max_merges))
        dry_run = bool(self.config.worktree_reconciliation_dry_run)
        scan_cache = self._load_worktree_scan_cache()
        scan_cache_hit_count = 0
        candidates: list[dict[str, Any]] = []
        processed: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        reconciliation_daemon: PortalImplementationDaemon | None = None

        for record in records:
            path_text = str(record.get("worktree") or "")
            if not path_text:
                continue
            path = Path(path_text)
            try:
                path_resolved = path.resolve()
                path_resolved.relative_to(root_resolved)
            except (OSError, ValueError):
                continue

            branch = str(record.get("branch") or "").removeprefix("refs/heads/")
            head = str(record.get("HEAD") or "")
            detail: dict[str, Any] = {"path": str(path), "branch": branch, "head": head}
            if active_worktree and path_resolved == Path(active_worktree).resolve():
                skipped.append({**detail, "reason": "active_state_worktree"})
                continue
            if any(str(path_resolved) in line for line in process_lines):
                skipped.append({**detail, "reason": "active_process"})
                continue
            cached_entry = self._worktree_scan_cache_entry(
                scan_cache,
                phase="reconciliation",
                path=path_resolved,
                branch=branch,
                head=head,
                target_signature=target_signature,
            )
            if cached_entry:
                classification = str(cached_entry.get("classification") or "")
                payload = dict(cached_entry.get("payload") or {})
                if classification == "skip":
                    skipped.append({**payload, "cached": True})
                    scan_cache_hit_count += 1
                    continue
                if classification == "candidate" and (dry_run or main_status):
                    candidate = {**payload, "cached": True}
                    candidates.append(candidate)
                    scan_cache_hit_count += 1
                    if not dry_run:
                        skipped.append({**candidate, "reason": "main_checkout_dirty", "status_short": main_status[:20]})
                    continue
            if not branch.startswith("implementation/"):
                skip = {**detail, "reason": "non_implementation_branch"}
                skipped.append(skip)
                self._store_worktree_scan_cache_entry(
                    scan_cache,
                    phase="reconciliation",
                    path=path_resolved,
                    branch=branch,
                    head=head,
                    target_signature=target_signature,
                    classification="skip",
                    payload=skip,
                )
                continue
            if not self._git_ref_exists(repo_root, branch):
                skip = {**detail, "reason": "implementation_branch_missing"}
                skipped.append(skip)
                self._store_worktree_scan_cache_entry(
                    scan_cache,
                    phase="reconciliation",
                    path=path_resolved,
                    branch=branch,
                    head=head,
                    target_signature=target_signature,
                    classification="skip",
                    payload=skip,
                )
                continue

            branch_merged = self._git_ref_is_ancestor(repo_root, branch, target_ref)
            head_merged = bool(head) and self._git_ref_is_ancestor(repo_root, head, target_ref)
            if branch_merged or head_merged:
                skip = {**detail, "reason": "already_merged_cleanup_pass"}
                skipped.append(skip)
                self._store_worktree_scan_cache_entry(
                    scan_cache,
                    phase="reconciliation",
                    path=path_resolved,
                    branch=branch,
                    head=head,
                    target_signature=target_signature,
                    classification="skip",
                    payload=skip,
                )
                continue
            dirty = self._git_status_short(path) if path.exists() else []
            if dirty:
                skip = {**detail, "reason": "dirty_worktree", "status_short": dirty[:20]}
                skipped.append(skip)
                self._store_worktree_scan_cache_entry(
                    scan_cache,
                    phase="reconciliation",
                    path=path_resolved,
                    branch=branch,
                    head=head,
                    target_signature=target_signature,
                    classification="skip",
                    payload=skip,
                )
                continue

            candidate = {**detail, "target_ref": target_ref}
            candidates.append(candidate)
            self._store_worktree_scan_cache_entry(
                scan_cache,
                phase="reconciliation",
                path=path_resolved,
                branch=branch,
                head=head,
                target_signature=target_signature,
                classification="candidate",
                payload=candidate,
            )
            if dry_run:
                continue
            if main_status:
                skipped.append({**candidate, "reason": "main_checkout_dirty", "status_short": main_status[:20]})
                continue
            if sum(1 for item in processed if item.get("merged")) >= max_merges:
                skipped.append({**candidate, "reason": "reconciliation_limit_reached"})
                continue

            preflight_result: dict[str, Any] = {}
            if self.config.worktree_reconciliation_preflight_enabled:
                preflight_result = self._preflight_worktree_reconciliation_merge(
                    repo_root,
                    target_ref=target_ref,
                    branch=branch,
                )
                if not preflight_result.get("mergeable", False):
                    processed.append(
                        {
                            **candidate,
                            "merged": False,
                            "preflight_result": preflight_result,
                            "merge_result": {
                                "attempted": False,
                                "merged": False,
                                "returncode": preflight_result.get("returncode"),
                                "branch": branch,
                                "target_ref": target_ref,
                                "reason": "preflight_merge_conflict",
                                "stdout": preflight_result.get("stdout", ""),
                                "stderr": preflight_result.get("stderr", ""),
                            },
                        }
                    )
                    continue

            if reconciliation_daemon is None:
                reconciliation_daemon = self._build_worktree_reconciliation_daemon()
            task = self._worktree_reconciliation_task(branch)
            merge_result = reconciliation_daemon._merge_branch_to_main(branch, task, 0)
            cleanup_result: dict[str, Any] = {}
            if merge_result.get("merged"):
                cleanup_result = reconciliation_daemon._cleanup_merged_worktree(path, branch)
            processed.append(
                {
                    **candidate,
                    "merged": bool(merge_result.get("merged")),
                    "merge_result": merge_result,
                    "cleanup_result": cleanup_result,
                }
            )

        result = {
            "attempted": True,
            "worktree_root": str(worktree_root),
            "target_ref": target_ref,
            "target_signature": target_signature,
            "dry_run": dry_run,
            "max_merges": max_merges,
            "main_checkout_dirty": bool(main_status),
            "main_status_short": main_status[:20],
            "main_dirty_evidence": main_dirty_evidence,
            "raw_main_checkout_dirty": bool(raw_main_status),
            "raw_main_status_short": raw_main_status[:20],
            "raw_main_dirty_evidence": raw_main_dirty_evidence,
            "candidate_count": len(candidates),
            "processed_count": len(processed),
            "reconciled_count": sum(1 for item in processed if item.get("merged")),
            "preflight_blocked_count": sum(
                1
                for item in processed
                if isinstance(item.get("preflight_result"), dict)
                and not item["preflight_result"].get("mergeable", False)
            ),
            "cleanup_count": sum(
                1
                for item in processed
                if isinstance(item.get("cleanup_result"), dict)
                and item["cleanup_result"].get("cleaned", False)
            ),
            "skipped_count": len(skipped),
            "candidates": candidates[:50],
            "processed": processed,
            "skipped": skipped[:50],
            "scan_cache_hit_count": scan_cache_hit_count,
            "scan_cache_written": self._write_worktree_scan_cache(scan_cache),
        }
        if processed:
            self._record_event("worktree_reconciliation", result)
        return result

    def _preflight_worktree_reconciliation_merge(
        self,
        repo_root: Path,
        *,
        target_ref: str,
        branch: str,
    ) -> dict[str, Any]:
        """Check branch mergeability without mutating the main checkout."""

        command = ["git", "merge-tree", "--write-tree", target_ref, branch]
        started_at = utc_now()
        result = subprocess.run(
            command,
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        payload: dict[str, Any] = {
            "attempted": True,
            "mergeable": result.returncode == 0,
            "returncode": result.returncode,
            "target_ref": target_ref,
            "branch": branch,
            "command": command,
            "started_at": started_at,
            "finished_at": utc_now(),
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
        }
        if result.returncode == 0:
            payload["tree"] = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
            return payload
        payload["reason"] = "preflight_merge_conflict"
        payload["conflict_paths"] = self._merge_tree_conflict_paths(output)
        return payload

    @staticmethod
    def _merge_tree_conflict_paths(output: str) -> list[str]:
        paths: list[str] = []
        for line in output.splitlines():
            path = ""
            if "Merge conflict in " in line:
                path = line.rsplit("Merge conflict in ", 1)[-1].strip()
            elif line.startswith("CONFLICT ") and " in " in line:
                path = line.rsplit(" in ", 1)[-1].strip()
            if path and path not in paths:
                paths.append(path)
        return paths

    def _main_checkout_dirty_evidence(self, repo_root: Path, status_lines: list[str]) -> dict[str, Any]:
        """Return bounded evidence for dirty main-checkout reconciliation blockers."""

        path_categories: dict[str, int] = {}
        status_paths: list[str] = []
        for line in status_lines:
            path = self._status_line_path(line)
            if path and path not in status_paths:
                status_paths.append(path)
            category = self._status_line_category(line)
            path_categories[category] = path_categories.get(category, 0) + 1
        evidence: dict[str, Any] = {
            "status_short": status_lines[:50],
            "status_paths": status_paths[:50],
            "path_categories": path_categories,
        }
        diff_stat = self._git_output(repo_root, ["diff", "--stat"], max_chars=4000)
        if diff_stat:
            evidence["diff_stat"] = diff_stat
        name_status = self._git_output(repo_root, ["diff", "--name-status"], max_chars=4000)
        if name_status:
            evidence["name_status"] = name_status
        staged_name_status = self._git_output(repo_root, ["diff", "--cached", "--name-status"], max_chars=4000)
        if staged_name_status:
            evidence["staged_name_status"] = staged_name_status
        submodule_summary = self._git_output(repo_root, ["submodule", "summary", "--files"], max_chars=4000)
        if submodule_summary:
            evidence["submodule_summary"] = submodule_summary
        untracked_paths = [
            self._status_line_path(line)
            for line in status_lines
            if line[:2] == "??" and self._status_line_path(line)
        ][:50]
        if untracked_paths:
            evidence["untracked_paths"] = untracked_paths
        return evidence

    def _generated_main_checkout_status_filters(self) -> tuple[list[str], list[str]]:
        """Return supervisor-generated dirty paths that should not block reconciliation."""

        from ipfs_accelerate_py.agent_supervisor.backlog_refinery import (
            generated_guardrail_status_filters,
        )

        discovery_dir = self._reconciliation_guardrail_discovery_dir()
        additional_paths = [
            path
            for path in (
                self.config.objective_path,
                self.config.objective_graph_path,
                self.config.objective_todo_vector_index_path,
            )
            if path is not None
        ]
        additional_prefixes = [
            path
            for path in (
                self.config.retry_budget_discovery_dir,
                self.config.dependency_guardrail_discovery_dir,
                self.config.reconciliation_guardrail_discovery_dir,
                self.config.codebase_scan_discovery_dir,
                self.config.objective_bundle_dir,
                self.config.objective_dataset_dir,
                self.config.objective_discovery_dir,
            )
            if path is not None
        ]
        return generated_guardrail_status_filters(
            todo_path=self.config.todo_path,
            discovery_dir=discovery_dir,
            repo_root=self.config.repo_root,
            additional_generated_paths=additional_paths,
            additional_generated_prefixes=additional_prefixes,
        )

    def _filter_generated_main_checkout_status(
        self,
        status_lines: list[str],
        evidence: Mapping[str, Any],
    ) -> tuple[list[str], dict[str, Any]]:
        """Filter deterministic supervisor output from main dirty evidence."""

        if not status_lines:
            return [], dict(evidence or {})
        from ipfs_accelerate_py.agent_supervisor.backlog_refinery import (
            filter_generated_main_checkout_evidence,
        )

        generated_paths, generated_prefixes = self._generated_main_checkout_status_filters()
        return filter_generated_main_checkout_evidence(
            status_short=status_lines,
            evidence=evidence,
            generated_paths=generated_paths,
            generated_prefixes=generated_prefixes,
        )

    @staticmethod
    def _status_line_category(line: str) -> str:
        code = line[:2]
        if code == "??":
            return "untracked"
        if "U" in code:
            return "unmerged"
        if "D" in code:
            return "deleted"
        if "R" in code:
            return "renamed"
        if "A" in code:
            return "added"
        if "M" in code:
            return "modified"
        if code.strip():
            return "other_dirty"
        return "clean"

    def _build_worktree_reconciliation_daemon(self) -> PortalImplementationDaemon:
        return PortalImplementationDaemon(
            todo_path=self.config.todo_path,
            state_path=self.config.state_path,
            strategy_path=self.config.strategy_path,
            events_path=self.config.events_path,
            repo_root=self.config.repo_root,
            task_header_prefix=self.config.task_prefix,
            implement=False,
            implementation_command=self.config.implementation_command,
            implementation_timeout=self.config.implementation_timeout,
            use_ephemeral_worktree=False,
            worktree_root=self.config.worktree_root,
            worktree_submodule_paths=self.config.worktree_submodule_paths,
            objective_path=self.config.objective_path,
            objective_bundle_dir=self.config.objective_bundle_dir,
            llm_merge_resolver_command=self.config.llm_merge_resolver_command,
            llm_merge_resolver_timeout_seconds=self.config.llm_merge_resolver_timeout_seconds,
        )

    def _reconciliation_guardrail_discovery_dir(self) -> Path:
        return (
            self.config.reconciliation_guardrail_discovery_dir
            or self.config.dependency_guardrail_discovery_dir
            or self.config.retry_budget_discovery_dir
            or self.config.state_dir.parent / "discovery"
        )

    def _main_status_for_worktree_reconciliation(self, repo_root: Path, worktree_root: Path) -> list[str]:
        status = self._git_status_short(repo_root)
        try:
            root_relative = worktree_root.resolve().relative_to(repo_root.resolve()).as_posix().rstrip("/")
        except (OSError, ValueError):
            return status
        if not root_relative:
            return status
        return [
            line
            for line in status
            if not self._status_line_targets_prefix(line, root_relative)
        ]

    @staticmethod
    def _status_line_targets_prefix(line: str, relative_prefix: str) -> bool:
        path_text = line[3:].strip() if len(line) > 3 else line.strip()
        if " -> " in path_text:
            path_text = path_text.split(" -> ", 1)[-1].strip()
        path_text = path_text.rstrip("/")
        return path_text == relative_prefix or path_text.startswith(f"{relative_prefix}/")

    @staticmethod
    def _worktree_reconciliation_task(branch: str) -> PortalTask:
        task_fragment = branch.removeprefix("implementation/").split("-attempt-", 1)[0].strip()
        task_id = task_fragment.upper() if task_fragment else "WORKTREE-RECONCILE"
        return PortalTask(
            task_id=task_id,
            title=f"Reconcile backlogged implementation branch {branch}",
            status="todo",
            completion="manual",
            priority="P2",
            track="ops",
        )

    def cleanup_backlogged_worktrees(self) -> dict[str, Any]:
        """Remove inactive implementation worktrees whose branches are already merged."""

        worktree_root = self.config.worktree_root
        if worktree_root is None:
            return {"attempted": False, "reason": "worktree_root_not_configured"}
        repo_root = self.config.repo_root
        prune = subprocess.run(
            ["git", "worktree", "prune"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        records = self._git_worktree_records(repo_root)
        try:
            root_resolved = worktree_root.resolve()
        except OSError:
            root_resolved = worktree_root
        process_lines = self._list_process_commands()
        active_worktree = ""
        try:
            active_worktree = PortalTaskState.load(self.config.state_path).active_worktree_path
        except Exception:
            active_worktree = ""
        target_ref = self._git_current_branch(repo_root) or "HEAD"
        target_signature = self._git_ref_commit(repo_root, target_ref) or target_ref
        scan_cache = self._load_worktree_scan_cache()
        scan_cache_hit_count = 0
        removed: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        dirty_evidence_sample_counts: dict[str, int] = {}

        for record in records:
            path_text = str(record.get("worktree") or "")
            if not path_text:
                continue
            path = Path(path_text)
            try:
                path_resolved = path.resolve()
                path_resolved.relative_to(root_resolved)
            except (OSError, ValueError):
                continue
            if active_worktree and path_resolved == Path(active_worktree).resolve():
                skipped.append({"path": str(path), "reason": "active_state_worktree"})
                continue
            if any(str(path_resolved) in line for line in process_lines):
                skipped.append({"path": str(path), "reason": "active_process"})
                continue

            branch = str(record.get("branch") or "").removeprefix("refs/heads/")
            head = str(record.get("HEAD") or "")
            cached_entry = self._worktree_scan_cache_entry(
                scan_cache,
                phase="cleanup",
                path=path_resolved,
                branch=branch,
                head=head,
                target_signature=target_signature,
            )
            if cached_entry:
                classification = str(cached_entry.get("classification") or "")
                payload = dict(cached_entry.get("payload") or {})
                if classification == "skip":
                    skipped.append({**payload, "cached": True})
                    scan_cache_hit_count += 1
                    continue
            branch_merged = bool(branch) and self._git_ref_is_ancestor(repo_root, branch, target_ref)
            head_merged = bool(head) and self._git_ref_is_ancestor(repo_root, head, target_ref)
            if not (branch_merged or head_merged):
                skip = {"path": str(path), "branch": branch, "reason": "not_merged"}
                skipped.append(skip)
                self._store_worktree_scan_cache_entry(
                    scan_cache,
                    phase="cleanup",
                    path=path_resolved,
                    branch=branch,
                    head=head,
                    target_signature=target_signature,
                    classification="skip",
                    payload=skip,
                )
                continue
            dirty = self._git_status_short(path) if path.exists() else []
            dirty_redundancy: dict[str, Any] = {}
            if dirty:
                redundant_dirty = self._redundant_dirty_worktree_status(path, dirty, target_ref)
                if redundant_dirty.get("redundant"):
                    dirty_redundancy = redundant_dirty
                    dirty = []
                else:
                    dirty_reason = self._dirty_redundancy_reason(redundant_dirty)
                    evidence: dict[str, Any] = {}
                    if dirty_evidence_sample_counts.get(dirty_reason, 0) < 20:
                        evidence = self._dirty_worktree_evidence(path, dirty)
                        dirty_evidence_sample_counts[dirty_reason] = (
                            dirty_evidence_sample_counts.get(dirty_reason, 0) + 1
                        )
                    skip = {
                        "path": str(path),
                        "branch": branch,
                        "reason": "dirty_worktree",
                        "status_short": dirty[:20],
                        "dirty_redundancy": redundant_dirty,
                        "dirty_evidence": evidence,
                    }
                    skipped.append(skip)
                    self._store_worktree_scan_cache_entry(
                        scan_cache,
                        phase="cleanup",
                        path=path_resolved,
                        branch=branch,
                        head=head,
                        target_signature=target_signature,
                        classification="skip",
                        payload=skip,
                    )
                    continue

            remove = subprocess.run(
                ["git", "worktree", "remove", "--force", str(path)],
                cwd=repo_root,
                text=True,
                capture_output=True,
                check=False,
            )
            branch_delete: dict[str, Any] = {}
            if remove.returncode == 0 and branch.startswith("implementation/") and branch_merged:
                delete = subprocess.run(
                    ["git", "branch", "-D", branch],
                    cwd=repo_root,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                branch_delete = {
                    "attempted": True,
                    "deleted": delete.returncode == 0,
                    "returncode": delete.returncode,
                    "stdout": delete.stdout[-4000:],
                    "stderr": delete.stderr[-4000:],
                }
            removed.append(
                {
                    "path": str(path),
                    "branch": branch,
                    "head": head,
                    "removed": remove.returncode == 0,
                    "returncode": remove.returncode,
                    "stdout": remove.stdout[-4000:],
                    "stderr": remove.stderr[-4000:],
                    "branch_delete": branch_delete,
                    "dirty_redundancy": dirty_redundancy,
                }
            )

        skip_summary = self._cleanup_skip_summary(skipped)
        result = {
            "attempted": True,
            "worktree_root": str(worktree_root),
            "target_ref": target_ref,
            "target_signature": target_signature,
            "prune_returncode": prune.returncode,
            "prune_stdout": prune.stdout[-4000:],
            "prune_stderr": prune.stderr[-4000:],
            "removed_count": sum(1 for item in removed if item.get("removed")),
            "skipped_count": len(skipped),
            "skipped_reason_counts": skip_summary["reason_counts"],
            "dirty_worktree_groups": skip_summary["dirty_worktree_groups"],
            "removed": removed,
            "skipped": skipped[:50],
            "scan_cache_hit_count": scan_cache_hit_count,
            "scan_cache_written": self._write_worktree_scan_cache(scan_cache),
        }
        if removed or skip_summary["dirty_worktree_groups"]:
            self._record_event("merged_worktree_cleanup", result)
        return result

    @staticmethod
    def _dirty_redundancy_reason(dirty_redundancy: dict[str, Any]) -> str:
        return str(dirty_redundancy.get("reason") or "dirty_worktree")

    def _dirty_worktree_evidence(self, worktree_path: Path, status_lines: list[str]) -> dict[str, Any]:
        """Return bounded evidence for dirty cleanup blockers without storing full patches."""

        evidence: dict[str, Any] = {
            "status_short": status_lines[:20],
        }
        diff_stat = self._git_output(worktree_path, ["diff", "--stat"], max_chars=4000)
        if diff_stat:
            evidence["diff_stat"] = diff_stat
        name_status = self._git_output(worktree_path, ["diff", "--name-status"], max_chars=4000)
        if name_status:
            evidence["name_status"] = name_status
        untracked_paths = [
            self._status_line_path(line)
            for line in status_lines
            if line[:2] == "??" and self._status_line_path(line)
        ][:20]
        if untracked_paths:
            evidence["untracked_paths"] = untracked_paths
        return evidence

    @staticmethod
    def _cleanup_skip_summary(skipped: list[dict[str, Any]]) -> dict[str, Any]:
        reason_counts: dict[str, int] = {}
        dirty_worktree_groups: dict[str, dict[str, Any]] = {}
        for item in skipped:
            reason = str(item.get("reason") or "unknown")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            if reason != "dirty_worktree":
                continue
            dirty_redundancy = item.get("dirty_redundancy") or {}
            dirty_reason = (
                str(dirty_redundancy.get("reason") or "dirty_worktree")
                if isinstance(dirty_redundancy, dict)
                else "dirty_worktree"
            )
            reason_key = f"dirty_worktree:{dirty_reason}"
            reason_counts[reason_key] = reason_counts.get(reason_key, 0) + 1
            group = dirty_worktree_groups.setdefault(
                dirty_reason,
                {
                    "count": 0,
                    "samples": [],
                },
            )
            group["count"] += 1
            if len(group["samples"]) < 20:
                group["samples"].append(
                    {
                        "branch": str(item.get("branch") or ""),
                        "path": str(item.get("path") or ""),
                        "status_short": list(item.get("status_short") or []),
                        "dirty_reason": dirty_reason,
                        "dirty_evidence": dict(item.get("dirty_evidence") or {}),
                    }
                )
        return {
            "reason_counts": reason_counts,
            "dirty_worktree_groups": dirty_worktree_groups,
        }

    @staticmethod
    def _git_output(cwd: Path, args: list[str], *, max_chars: int = 4000) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return ""
        return result.stdout[-max_chars:].strip()

    def _redundant_dirty_worktree_status(
        self,
        worktree_path: Path,
        status_lines: list[str],
        target_ref: str,
    ) -> dict[str, Any]:
        checked: list[dict[str, Any]] = []
        configured_submodule_deletion = False
        for line in status_lines:
            code = line[:2]
            relative = self._status_line_path(line)
            detail = {"status": code, "path": relative}
            if not relative:
                return {"redundant": False, "reason": "empty_status_path", "checked": checked}
            if self._status_line_is_configured_submodule_deletion(code, relative, target_ref):
                checked.append({**detail, "matches_target": True, "configured_submodule_deletion": True})
                configured_submodule_deletion = True
                continue
            if "D" in code or "?" in code.strip(" ?"):
                return {"redundant": False, "reason": "unsupported_status", "checked": [*checked, detail]}
            if code == "??" or "M" in code or "A" in code:
                if not self._worktree_file_matches_ref(worktree_path, relative, target_ref):
                    return {"redundant": False, "reason": "content_not_in_target", "checked": [*checked, detail]}
                checked.append({**detail, "matches_target": True})
                continue
            return {"redundant": False, "reason": "unsupported_status", "checked": [*checked, detail]}
        reason = (
            "configured_submodule_deletions_match_target"
            if configured_submodule_deletion
            else "all_dirty_paths_match_target"
        )
        return {"redundant": True, "reason": reason, "checked": checked}

    def _status_line_is_configured_submodule_deletion(
        self,
        code: str,
        relative: str,
        target_ref: str,
    ) -> bool:
        if code not in {" D", "D "}:
            return False
        relative = relative.rstrip("/")
        if not any(relative == path.rstrip("/") for path in self.config.worktree_submodule_paths):
            return False
        return self._target_ref_has_path(relative, target_ref)

    def _target_ref_has_path(self, relative: str, target_ref: str) -> bool:
        result = subprocess.run(
            ["git", "ls-tree", target_ref, "--", relative],
            cwd=self.config.repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.returncode == 0 and bool(result.stdout.strip())

    @staticmethod
    def _status_line_path(line: str) -> str:
        path_text = line[3:].strip() if len(line) > 3 else line.strip()
        if " -> " in path_text:
            path_text = path_text.split(" -> ", 1)[-1].strip()
        return path_text.rstrip("/")

    def _worktree_file_matches_ref(self, worktree_path: Path, relative: str, target_ref: str) -> bool:
        candidate = worktree_path / relative
        if not candidate.is_file():
            return False
        result = subprocess.run(
            ["git", "show", f"{target_ref}:{relative}"],
            cwd=self.config.repo_root,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return False
        try:
            return candidate.read_bytes() == result.stdout
        except OSError:
            return False

    @staticmethod
    def _git_worktree_records(repo_root: Path) -> list[dict[str, str]]:
        result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return []
        records: list[dict[str, str]] = []
        current: dict[str, str] = {}
        for line in result.stdout.splitlines():
            if not line.strip():
                if current:
                    records.append(current)
                    current = {}
                continue
            key, _, value = line.partition(" ")
            current[key] = value
        if current:
            records.append(current)
        return records

    @staticmethod
    def _git_ref_is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
        if not ancestor or not descendant:
            return False
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.returncode == 0

    @staticmethod
    def _git_ref_exists(repo_root: Path, ref: str) -> bool:
        if not ref:
            return False
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", ref],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.returncode == 0

    @staticmethod
    def _list_process_commands() -> list[str]:
        result = subprocess.run(
            ["ps", "-eo", "args="],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def ensure_todo_board_for_refill(self) -> dict[str, Any]:
        """Create an empty todo board when refill machinery is expected to populate it."""

        if self.config.todo_path.exists():
            if self.config.todo_path.is_dir():
                if not (
                    self.config.objective_refill_enabled
                    or self.config.codebase_refill_enabled
                    or self.config.reconciliation_guardrail_enabled
                ):
                    return {"created": False, "reason": "todo_path_is_directory", "path": str(self.config.todo_path)}
                backup_path = unique_backup_path(self.config.todo_path, "directory-backup")
                self.config.todo_path.rename(backup_path)
                self.config.todo_path.parent.mkdir(parents=True, exist_ok=True)
                write_text_atomic(self.config.todo_path, "# Agent Todos\n")
                result = {
                    "created": True,
                    "repaired": True,
                    "reason": "todo_path_was_directory",
                    "path": str(self.config.todo_path),
                    "backup_path": str(backup_path),
                }
                self._record_event("todo_board_repaired", result)
                return result
            try:
                self.config.todo_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                if not (
                    self.config.objective_refill_enabled
                    or self.config.codebase_refill_enabled
                    or self.config.reconciliation_guardrail_enabled
                ):
                    return {"created": False, "reason": "todo_text_decode_failed", "path": str(self.config.todo_path)}
                backup_path = unique_backup_path(self.config.todo_path, "invalid-text")
                self.config.todo_path.rename(backup_path)
                write_text_atomic(self.config.todo_path, "# Agent Todos\n")
                result = {
                    "created": True,
                    "repaired": True,
                    "reason": "todo_text_decode_failed",
                    "path": str(self.config.todo_path),
                    "backup_path": str(backup_path),
                }
                self._record_event("todo_board_repaired", result)
                return result
            except OSError as exc:
                return {
                    "created": False,
                    "reason": "todo_read_failed",
                    "path": str(self.config.todo_path),
                    "error": str(exc),
                }
            return {"created": False, "reason": "exists", "path": str(self.config.todo_path)}
        if not (
            self.config.objective_refill_enabled
            or self.config.codebase_refill_enabled
            or self.config.reconciliation_guardrail_enabled
        ):
            return {"created": False, "reason": "refill_disabled", "path": str(self.config.todo_path)}
        self.config.todo_path.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(self.config.todo_path, "# Agent Todos\n")
        result = {"created": True, "reason": "refill_enabled", "path": str(self.config.todo_path)}
        self._record_event("todo_board_created", result)
        return result

    def ensure_event_log_file(self) -> dict[str, Any]:
        """Repair malformed supervisor event-log storage before guardrails run."""

        result = repair_jsonl_event_log(self.config.events_path)
        if result.get("repaired"):
            append_jsonl_event(self.config.events_path, "event_log_repaired", result)
        return result

    def ensure_state_file(self) -> dict[str, Any]:
        """Repair malformed durable daemon state before supervisor checks it."""

        reason = state_file_repair_reason(self.config.state_path)
        if not reason or reason == "missing_state_file":
            return {"repaired": False, "reason": reason or "valid", "path": str(self.config.state_path)}
        PortalTaskState().save(self.config.state_path)
        result = {"repaired": True, "reason": reason, "path": str(self.config.state_path)}
        self._record_event("state_file_repaired", result)
        return result

    def ensure_strategy_file(self) -> dict[str, Any]:
        """Persist a valid strategy file before guardrail/refill work starts."""

        defaults = {
            "generation": 0,
            "focus_tracks": DEFAULT_TRACKS,
            "blocked_tasks": [],
            "deprioritized_tasks": [],
            "last_rewrite_at": "",
            "last_rewrite_reason": "",
        }
        reason = ""
        if not self.config.strategy_path.exists():
            strategy = defaults.copy()
            reason = "missing_strategy_file"
        else:
            payload = load_json_dict(self.config.strategy_path)
            if payload is None:
                strategy = defaults.copy()
                reason = "invalid_or_unreadable_strategy_file"
            else:
                strategy = {**defaults, **payload}
                normalized_blocked = (
                    [str(item) for item in strategy.get("blocked_tasks", []) if str(item).strip()]
                    if isinstance(strategy.get("blocked_tasks"), list)
                    else []
                )
                normalized_deprioritized = (
                    [str(item) for item in strategy.get("deprioritized_tasks", []) if str(item).strip()]
                    if isinstance(strategy.get("deprioritized_tasks"), list)
                    else []
                )
                normalized_focus = (
                    [str(item).strip().lower() for item in strategy.get("focus_tracks", []) if str(item).strip()]
                    if isinstance(strategy.get("focus_tracks"), list)
                    else DEFAULT_TRACKS
                )
                if (
                    normalized_blocked != strategy.get("blocked_tasks")
                    or normalized_deprioritized != strategy.get("deprioritized_tasks")
                    or normalized_focus != strategy.get("focus_tracks")
                ):
                    reason = "normalized_strategy_metadata"
                strategy["blocked_tasks"] = normalized_blocked
                strategy["deprioritized_tasks"] = normalized_deprioritized
                strategy["focus_tracks"] = normalized_focus or DEFAULT_TRACKS
        if not reason:
            return {"repaired": False, "reason": "valid", "path": str(self.config.strategy_path)}
        strategy["last_strategy_repair_at"] = utc_now()
        strategy["last_strategy_repair_reason"] = reason
        write_json_atomic(self.config.strategy_path, strategy)
        result = {"repaired": True, "reason": reason, "path": str(self.config.strategy_path)}
        self._record_event("strategy_file_repaired", result)
        return result

    def release_completed_guardrail_blocks(self) -> list[dict[str, Any]]:
        """Remove strategy blocks once their generated repair task is completed."""

        if not self.config.todo_path.exists() or not self.config.strategy_path.exists():
            return []
        from ipfs_accelerate_py.agent_supervisor.backlog_refinery import (
            release_completed_guardrail_blocks,
            task_id_prefix,
        )

        releases = release_completed_guardrail_blocks(
            todo_path=self.config.todo_path,
            strategy_path=self.config.strategy_path,
            task_prefix=task_id_prefix(self.config.task_prefix),
        )
        if releases:
            self._record_event(
                "guardrail_blocks_released",
                {
                    "released_count": len(releases),
                    "todo_path": str(self.config.todo_path),
                    "strategy_path": str(self.config.strategy_path),
                    "releases": releases,
                },
            )
        return releases

    def record_dependency_guardrails(self) -> list[dict[str, Any]]:
        """Convert impossible dependency metadata into ready repair tasks."""

        if not self.config.dependency_guardrail_enabled:
            return []
        if not self.config.todo_path.exists():
            return []

        from ipfs_accelerate_py.agent_supervisor.backlog_refinery import (
            record_dependency_guardrail_findings,
            task_id_prefix,
        )

        discovery_dir = (
            self.config.dependency_guardrail_discovery_dir
            or self.config.retry_budget_discovery_dir
            or self.config.state_dir.parent / "discovery"
        )
        discovery_output_path = self.config.dependency_guardrail_discovery_output_path
        if not discovery_output_path:
            try:
                discovery_output_path = discovery_dir.resolve().relative_to(self.config.repo_root.resolve()).as_posix()
            except ValueError:
                discovery_output_path = str(discovery_dir)
        findings = record_dependency_guardrail_findings(
            todo_path=self.config.todo_path,
            strategy_path=self.config.strategy_path,
            discovery_dir=discovery_dir,
            task_header_prefix_value=self.config.task_prefix,
            task_prefix=task_id_prefix(self.config.task_prefix),
            max_findings=self.config.dependency_guardrail_max_findings,
            discovery_output_path=discovery_output_path,
            commit_outputs=self.config.dependency_guardrail_commit_outputs,
            repo_root=self.config.repo_root,
            commit_subject=self.config.dependency_guardrail_commit_subject,
        )
        if findings:
            self._record_event(
                "dependency_guardrail",
                {
                    "generated_count": len(findings),
                    "todo_path": str(self.config.todo_path),
                    "discovery_dir": str(discovery_dir),
                    "findings": findings,
                },
            )
        return findings

    def record_reconciliation_guardrails(
        self,
        worktree_reconciliation: Mapping[str, Any],
        worktree_cleanup: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Convert blocked checkout/worktree cleanup into deliberate repair tasks."""

        if not self.config.reconciliation_guardrail_enabled:
            return []
        if not self.config.todo_path.exists():
            return []

        from ipfs_accelerate_py.agent_supervisor.backlog_refinery import (
            record_reconciliation_guardrail_findings,
            task_id_prefix,
        )

        discovery_dir = self._reconciliation_guardrail_discovery_dir()
        discovery_output_path = self.config.reconciliation_guardrail_discovery_output_path
        if not discovery_output_path:
            try:
                discovery_output_path = discovery_dir.resolve().relative_to(self.config.repo_root.resolve()).as_posix()
            except ValueError:
                discovery_output_path = str(discovery_dir)
        generated_paths, generated_prefixes = self._generated_main_checkout_status_filters()
        findings = record_reconciliation_guardrail_findings(
            todo_path=self.config.todo_path,
            strategy_path=self.config.strategy_path,
            discovery_dir=discovery_dir,
            reconciliation_result=worktree_reconciliation,
            cleanup_result=worktree_cleanup,
            task_prefix=task_id_prefix(self.config.task_prefix),
            max_findings=self.config.reconciliation_guardrail_max_findings,
            discovery_output_path=discovery_output_path,
            commit_outputs=self.config.reconciliation_guardrail_commit_outputs,
            repo_root=self.config.repo_root,
            commit_subject=self.config.reconciliation_guardrail_commit_subject,
            additional_generated_status_paths=generated_paths,
            additional_generated_status_prefixes=generated_prefixes,
        )
        if findings:
            self._record_event(
                "reconciliation_guardrail",
                {
                    "generated_count": len(findings),
                    "todo_path": str(self.config.todo_path),
                    "discovery_dir": str(discovery_dir),
                    "findings": findings,
                },
            )
        return findings

    def record_retry_budget_guardrails(self) -> list[dict[str, Any]]:
        """Convert repeated daemon blockers into follow-up work before another retry loop."""

        if not self.config.retry_budget_guardrail_enabled:
            return []
        if not self.config.todo_path.exists():
            return []

        from ipfs_accelerate_py.agent_supervisor.backlog_refinery import (
            record_retry_budget_findings,
            task_id_prefix,
        )

        discovery_dir = self.config.retry_budget_discovery_dir or self.config.state_dir.parent / "discovery"
        discovery_output_path = self.config.retry_budget_discovery_output_path
        if not discovery_output_path:
            try:
                discovery_output_path = discovery_dir.resolve().relative_to(self.config.repo_root.resolve()).as_posix()
            except ValueError:
                discovery_output_path = str(discovery_dir)
        findings = record_retry_budget_findings(
            todo_path=self.config.todo_path,
            events_path=self.config.state_dir / f"{self.config.state_prefix}_events.jsonl",
            strategy_path=self.config.strategy_path,
            discovery_dir=discovery_dir,
            task_header_prefix_value=self.config.task_prefix,
            task_prefix=task_id_prefix(self.config.task_prefix),
            validation_retry_budget=self.config.validation_retry_budget,
            merge_retry_budget=self.config.merge_retry_budget,
            implementation_retry_budget=self.config.implementation_retry_budget,
            discovery_output_path=discovery_output_path,
            commit_outputs=self.config.retry_budget_commit_outputs,
            repo_root=self.config.repo_root,
            commit_subject=self.config.retry_budget_commit_subject,
        )
        if findings:
            self._record_event(
                "retry_budget_guardrail",
                {
                    "generated_count": len(findings),
                    "todo_path": str(self.config.todo_path),
                    "discovery_dir": str(discovery_dir),
                    "findings": findings,
                },
            )
        return findings

    def refill_objective_backlog(self) -> dict[str, Any]:
        """Refine the objective heap and feed todos when the backlog is low or drained."""

        if not self.config.objective_refill_enabled:
            return {}

        from argparse import Namespace

        from ipfs_accelerate_py.agent_supervisor.backlog_refinery import (
            load_strategy,
            should_refill_backlog,
            task_id_prefix,
            write_json,
        )
        from ipfs_accelerate_py.agent_supervisor.objective_daemon import (
            default_objective_path,
            discovery_fingerprints,
            run_objective_daemon,
        )
        from ipfs_accelerate_py.agent_supervisor.objective_graph import (
            DEFAULT_DISCOVERY_OUTPUT_PATH,
            DEFAULT_OBJECTIVE_TASK_SUMMARY_PREFIX,
        )
        from ipfs_accelerate_py.agent_supervisor.objective_tracker import (
            DEFAULT_ROOT_GOAL_TITLE,
            DEFAULT_TRACKING_DOCUMENT_TITLE,
            DEFAULT_ULTIMATE_GOAL,
        )

        objective_path = self.config.objective_path or default_objective_path(self.config.repo_root)
        if not objective_path.exists() and not self.config.objective_ensure_tracking_document:
            return {}
        if not self.config.todo_path.exists():
            return {}

        todo_text = self.config.todo_path.read_text(encoding="utf-8")
        strategy = load_strategy(self.config.strategy_path)
        task_prefix = task_id_prefix(self.config.task_prefix)
        should_scan, mode, current_open, task_count = should_refill_backlog(
            todo_text=todo_text,
            state_path=self.config.state_path,
            strategy=strategy,
            last_scan_key="last_objective_goal_scan_at",
            last_drained_scan_task_count_key="last_drained_objective_goal_scan_task_count",
            task_prefix=task_prefix,
            min_open_tasks=self.config.objective_scan_min_open_tasks,
            cooldown_seconds=self.config.objective_scan_cooldown_seconds,
        )
        if not should_scan:
            return {}

        state_root = self.config.state_dir.parent
        discovery_dir = self.config.objective_discovery_dir or state_root / "discovery"
        bundle_dir = self.config.objective_bundle_dir or state_root / "objective_bundles"
        dataset_dir = self.config.objective_dataset_dir or state_root / "objective_datasets"
        graph_path = self.config.objective_graph_path or state_root / "objective_graph.json"
        discovery_output_path = self.config.objective_discovery_output_path
        if not discovery_output_path:
            try:
                discovery_output_path = discovery_dir.resolve().relative_to(self.config.repo_root.resolve()).as_posix()
            except ValueError:
                discovery_output_path = DEFAULT_DISCOVERY_OUTPUT_PATH

        seen_fingerprints = {
            str(item)
            for item in strategy.get("objective_goal_seen_fingerprints", [])
            if str(item).strip()
        }
        seen_fingerprints.update(discovery_fingerprints(discovery_dir))
        payload = run_objective_daemon(
            Namespace(
                repo_root=self.config.repo_root,
                objective_path=objective_path,
                todo_path=self.config.todo_path,
                discovery_dir=discovery_dir,
                bundle_dir=bundle_dir,
                dataset_dir=dataset_dir,
                graph_path=graph_path,
                task_prefix=task_prefix,
                objective_summary_prefix=(
                    self.config.objective_summary_prefix or DEFAULT_OBJECTIVE_TASK_SUMMARY_PREFIX
                ),
                discovery_output_path=discovery_output_path,
                depends_on=list(self.config.objective_scan_depends_on),
                seen_fingerprint=sorted(seen_fingerprints),
                repeat_existing=False,
                max_findings=self.config.objective_scan_max_findings,
                ensure_tracking_document=self.config.objective_ensure_tracking_document,
                ultimate_goal=self.config.objective_ultimate_goal or DEFAULT_ULTIMATE_GOAL,
                root_evidence=list(self.config.objective_root_evidence),
                goal_prefix=self.config.objective_goal_prefix,
                root_goal_id=self.config.objective_root_goal_id,
                root_goal_title=self.config.objective_root_goal_title or DEFAULT_ROOT_GOAL_TITLE,
                tracking_document_title=(
                    self.config.objective_tracking_document_title or DEFAULT_TRACKING_DOCUMENT_TITLE
                ),
                refine_objective_heap=self.config.objective_refine_goals,
                no_reconcile_goal_completion=not self.config.objective_reconcile_goal_completion,
                seed_interoperability_goals=self.config.objective_seed_interoperability_goals,
                interoperability_focus=list(self.config.objective_interoperability_focus),
                interoperability_component_path=list(
                    self.config.objective_interoperability_component_paths
                    or self.config.worktree_submodule_paths
                ),
                max_interoperability_goals=self.config.objective_max_interoperability_goals,
                max_refinement_children=self.config.objective_max_refinement_children,
                max_refinement_depth=self.config.objective_max_refinement_depth,
                no_persist_ast_dataset=not self.config.objective_persist_ast_dataset,
                no_todo_vector_index=not self.config.objective_write_todo_vector_index,
                todo_vector_index_path=self.config.objective_todo_vector_index_path,
                surplus_findings_per_goal=self.config.objective_surplus_findings_per_goal,
                surplus_min_terms_per_todo=self.config.objective_surplus_min_terms_per_todo,
                submit_bundles=False,
                queue_path=None,
                queue_task_type="codex.todo_bundle",
                queue_model_name="codex",
                log_level="INFO",
            )
        )

        strategy = load_strategy(self.config.strategy_path)
        strategy["last_objective_goal_scan_at"] = utc_now()
        strategy["last_objective_goal_scan_mode"] = mode
        if current_open == 0 or mode.endswith("drained_exhaustive"):
            strategy["last_drained_objective_goal_scan_task_count"] = task_count
        strategy["objective_goal_seen_fingerprints"] = sorted(discovery_fingerprints(discovery_dir))
        strategy["last_objective_refined_goal_ids"] = list(payload.get("refined_goal_ids") or [])
        strategy["last_objective_completed_goal_ids"] = list(payload.get("completed_goal_ids") or [])
        strategy["last_objective_completion_validation_results"] = dict(
            payload.get("objective_completion_validation_results") or {}
        )
        strategy["last_objective_seeded_interoperability_goal_ids"] = list(
            payload.get("seeded_interoperability_goal_ids") or []
        )
        strategy["last_objective_generated_task_ids"] = list(payload.get("task_ids") or [])
        strategy["last_objective_todo_vector_index_path"] = str(payload.get("todo_vector_index_path") or "")
        strategy["last_objective_surplus_findings_per_goal"] = int(
            payload.get("surplus_findings_per_goal") or DEFAULT_OBJECTIVE_SURPLUS_FINDINGS_PER_GOAL
        )
        strategy["last_objective_surplus_min_terms_per_todo"] = int(
            payload.get("surplus_min_terms_per_todo") or DEFAULT_OBJECTIVE_SURPLUS_MIN_TERMS_PER_TODO
        )
        strategy["last_objective_goal_count"] = int(payload.get("objective_goal_count") or 0)
        strategy["last_objective_active_goal_count"] = int(payload.get("objective_active_goal_count") or 0)
        strategy["last_objective_completed_goal_count"] = int(payload.get("objective_completed_goal_count") or 0)
        strategy["last_objective_heap_schedule_count"] = int(payload.get("objective_heap_schedule_count") or 0)
        write_json(self.config.strategy_path, strategy)

        if (
            payload.get("completed_goal_ids")
            or payload.get("seeded_interoperability_goal_ids")
            or payload.get("refined_goal_ids")
            or payload.get("task_ids")
        ):
            self._record_event(
                "objective_refill_scan",
                {
                    "mode": mode,
                    "objective_path": str(objective_path),
                    "completed_goal_ids": payload.get("completed_goal_ids") or [],
                    "seeded_interoperability_goal_ids": payload.get("seeded_interoperability_goal_ids") or [],
                    "refined_goal_ids": payload.get("refined_goal_ids") or [],
                    "task_ids": payload.get("task_ids") or [],
                    "objective_active_goal_count": payload.get("objective_active_goal_count") or 0,
                    "objective_completed_goal_count": payload.get("objective_completed_goal_count") or 0,
                    "objective_heap_schedule_count": payload.get("objective_heap_schedule_count") or 0,
                    "bundle_keys": payload.get("bundle_keys") or [],
                    "todo_vector_index_path": payload.get("todo_vector_index_path") or "",
                    "surplus_findings_per_goal": payload.get("surplus_findings_per_goal")
                    or DEFAULT_OBJECTIVE_SURPLUS_FINDINGS_PER_GOAL,
                    "surplus_min_terms_per_todo": payload.get("surplus_min_terms_per_todo")
                    or DEFAULT_OBJECTIVE_SURPLUS_MIN_TERMS_PER_TODO,
                },
            )
        return payload

    def refill_codebase_backlog(self) -> list[dict[str, Any]]:
        """Feed low or drained todo boards from a codebase/submodule scan."""

        if not self.config.codebase_refill_enabled:
            return []

        from ipfs_accelerate_py.agent_supervisor.backlog_refinery import (
            CODEBASE_SCAN_SKIP_PREFIXES,
            record_codebase_scan_findings,
            task_id_prefix,
        )

        discovery_dir = self.config.codebase_scan_discovery_dir or self.config.state_dir.parent / "discovery"
        discovery_output_path = self.config.codebase_scan_discovery_output_path
        if not discovery_output_path:
            try:
                discovery_output_path = discovery_dir.resolve().relative_to(self.config.repo_root.resolve()).as_posix()
            except ValueError:
                discovery_output_path = str(discovery_dir)
        try:
            findings = record_codebase_scan_findings(
                todo_path=self.config.todo_path,
                state_path=self.config.state_path,
                strategy_path=self.config.strategy_path,
                discovery_dir=discovery_dir,
                repo_root=self.config.repo_root,
                task_prefix=task_id_prefix(self.config.task_prefix),
                depends_on=self.config.codebase_scan_depends_on,
                min_open_tasks=self.config.codebase_scan_min_open_tasks,
                max_findings=self.config.codebase_scan_max_findings,
                cooldown_seconds=self.config.codebase_scan_cooldown_seconds,
                discovery_output_path=discovery_output_path,
                skip_prefixes=self.config.codebase_scan_skip_prefixes or CODEBASE_SCAN_SKIP_PREFIXES,
                commit_outputs=self.config.codebase_scan_commit_outputs,
                commit_subject=self.config.codebase_scan_commit_subject,
            )
        except Exception as exc:
            failure = {
                "todo_path": str(self.config.todo_path),
                "discovery_dir": str(discovery_dir),
                "repo_root": str(self.config.repo_root),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            logger.warning("Codebase backlog refill failed; leaving supervisor alive", exc_info=True)
            self._record_event("codebase_refill_failed", failure)
            return []
        if findings:
            self._record_event(
                "codebase_refill_scan",
                {
                    "generated_count": len(findings),
                    "todo_path": str(self.config.todo_path),
                    "discovery_dir": str(discovery_dir),
                    "findings": findings,
                },
            )
        return findings

    def _implementation_attempt_is_active(self, state: PortalTaskState, *, now_ts: float) -> bool:
        if not state.active_task_id or not state.implementation_in_progress:
            return False
        if state.last_implementation_task_id and state.last_implementation_task_id != state.active_task_id:
            return False
        started_at = parse_timestamp(state.last_implementation_started_at or state.active_phase_started_at)
        if started_at is None:
            return False
        finished_at = parse_timestamp(state.last_implementation_finished_at)
        if finished_at is not None and finished_at >= started_at:
            return False
        grace_seconds = max(30.0, float(self.config.check_interval) * 2.0)
        max_age_seconds = max(float(self.config.stale_seconds), float(self.config.implementation_timeout))
        return max(0.0, now_ts - started_at.timestamp()) <= max_age_seconds + grace_seconds

    def _implementation_log_stall_reason(self, state: PortalTaskState, *, now_ts: float) -> str:
        if not state.active_task_id or not state.implementation_in_progress:
            return ""
        threshold = max(0.0, float(self.config.implementation_log_stall_seconds))
        if threshold <= 0.0:
            return ""
        log_text = state.last_implementation_log_path or state.active_log_path
        if not log_text:
            return ""
        log_path = Path(log_text)
        if not log_path.is_absolute():
            log_path = self.config.repo_root / log_path
        try:
            stat = log_path.stat()
        except OSError:
            return ""
        age_seconds = max(0.0, now_ts - stat.st_mtime)
        if age_seconds <= threshold:
            return ""
        return (
            f"implementation log stalled for active task {state.active_task_id}: "
            f"{age_seconds:.0f}s without output in {log_path}"
        )

    def is_stuck(
        self,
        state: PortalTaskState,
        *,
        now_ts: float,
        ignore_progress_until_ts: float | None = None,
    ) -> tuple[bool, str]:
        worktree_phase_stall_reason = self._worktree_phase_without_worker_reason(state, now_ts=now_ts)
        if worktree_phase_stall_reason:
            return True, worktree_phase_stall_reason
        log_stall_reason = self._implementation_log_stall_reason(state, now_ts=now_ts)
        if log_stall_reason:
            return True, log_stall_reason
        if self._implementation_attempt_is_active(state, now_ts=now_ts):
            return False, ""
        heartbeat_age = self._age_seconds(state.heartbeat_at, now_ts)
        progress_age = self._age_seconds(state.last_progress_at, now_ts)
        stale = self.config.stale_seconds
        if state.active_task_id and heartbeat_age > stale:
            return True, f"heartbeat stale for active task {state.active_task_id}"
        if (
            state.active_task_id
            and state.active_phase in {"merge_reconciliation", "merge_resolver"}
            and heartbeat_age <= stale
        ):
            return False, ""
        if ignore_progress_until_ts is not None and now_ts < ignore_progress_until_ts:
            return False, ""
        if state.active_task_id and state.ready_count > 0 and progress_age > stale:
            return True, f"no progress on active task {state.active_task_id}"
        if (
            state.active_task_id
            and state.last_implementation_task_id == state.active_task_id
            and state.last_implementation_commit
            and state.last_merge_returncode not in (None, 0)
            and not state.last_merge_commit
        ):
            detail = state.last_merge_error or "merge failed without a merge commit"
            return True, f"unresolved merge failure on active task {state.active_task_id}: {detail}"
        return False, ""

    def _worktree_phase_without_worker_reason(self, state: PortalTaskState, *, now_ts: float) -> str:
        if not state.active_task_id:
            return ""
        threshold = max(30.0, min(120.0, float(self.config.check_interval) * 2.0))
        worker_status = worktree_phase_worker_status(
            {
                "active_phase": state.active_phase,
                "active_phase_started_at": state.active_phase_started_at,
            },
            self._read_managed_daemon_pid(),
            threshold,
            now=datetime.fromtimestamp(now_ts, tz=timezone.utc),
        )
        if not worker_status.get("stalled_without_active_worker"):
            return ""
        self._record_event(
            "worktree_phase_without_worker",
            {
                "active_task_id": state.active_task_id,
                "active_phase": state.active_phase,
                "active_phase_detail": state.active_phase_detail,
                "worker_status": worker_status,
            },
        )
        return (
            f"{state.active_phase} stalled for active task {state.active_task_id}: "
            f"no active worker for {worker_status.get('phase_age_seconds')}s"
        )

    def rewrite_strategy(self, state: PortalTaskState, reason: str) -> dict[str, Any]:
        strategy = self._load_strategy()
        active_track = state.active_task_track.strip().lower()
        focus_tracks = [str(item).lower() for item in strategy.get("focus_tracks", DEFAULT_TRACKS)]
        generation = int(strategy.get("generation", 0)) + 1
        deprioritized_tasks = list(dict.fromkeys([*strategy.get("deprioritized_tasks", []), state.active_task_id]))
        blocked_tasks = [str(item) for item in strategy.get("blocked_tasks", []) if str(item).strip()]

        if active_track and active_track in focus_tracks:
            focus_tracks = [track for track in focus_tracks if track != active_track] + [active_track]

        strategy.update(
            {
                "generation": generation,
                "focus_tracks": focus_tracks or DEFAULT_TRACKS,
                "blocked_tasks": blocked_tasks,
                "deprioritized_tasks": [task_id for task_id in deprioritized_tasks if task_id],
                "last_rewrite_at": utc_now(),
                "last_rewrite_reason": reason,
            }
        )
        write_json_atomic(self.config.strategy_path, strategy)
        self._record_event(
            "strategy_rewrite",
            {
                "reason": reason,
                "generation": generation,
                "active_task_id": state.active_task_id,
                "active_track": active_track,
            },
        )
        return strategy

    def repair_blocked_progress_state(
        self,
        state: PortalTaskState,
        reason: str,
        *,
        now_ts: float,
    ) -> dict[str, Any]:
        """Clear stale active-task state after strategy has recorded the blocker."""

        if not state.active_task_id:
            return {"repaired": False, "reason": "no_active_task"}
        if self._implementation_attempt_is_active(state, now_ts=now_ts) and "no active worker" not in reason:
            return {"repaired": False, "reason": "implementation_attempt_active"}
        if reason.startswith("implementation log stalled"):
            return {"repaired": False, "reason": "implementation_log_stalled"}

        previous = {
            "active_task_id": state.active_task_id,
            "active_task_title": state.active_task_title,
            "active_task_track": state.active_task_track,
            "active_task_started_at": state.active_task_started_at,
            "active_attempt": state.active_attempt,
            "active_phase": state.active_phase,
            "active_phase_detail": state.active_phase_detail,
            "active_log_path": state.active_log_path,
            "active_worktree_path": state.active_worktree_path,
            "active_branch": state.active_branch,
            "implementation_in_progress": state.implementation_in_progress,
        }
        repaired_at = utc_now()
        state.active_task_id = ""
        state.active_task_title = ""
        state.active_task_track = ""
        state.active_task_started_at = ""
        state.active_attempt = 0
        state.active_phase = ""
        state.active_phase_started_at = ""
        state.active_phase_detail = ""
        state.active_log_path = ""
        state.active_worktree_path = ""
        state.active_branch = ""
        state.implementation_in_progress = False
        state.recommended_task_id = ""
        state.recommended_actions = []
        state.heartbeat_at = repaired_at
        state.last_progress_at = repaired_at
        state.save(self.config.state_path)
        result = {
            "repaired": True,
            "reason": "stale_active_state",
            "stuck_reason": reason,
            "repaired_at": repaired_at,
            **previous,
        }
        self._record_event("blocked_progress_state_repaired", result)
        return result

    def _load_strategy(self) -> dict[str, Any]:
        defaults = {
            "generation": 0,
            "focus_tracks": DEFAULT_TRACKS,
            "blocked_tasks": [],
            "deprioritized_tasks": [],
            "last_rewrite_at": "",
            "last_rewrite_reason": "",
        }
        if not self.config.strategy_path.exists():
            write_json_atomic(self.config.strategy_path, defaults)
            return defaults
        payload = load_json_dict(self.config.strategy_path)
        if payload is None:
            logger.warning("Strategy file is missing or invalid JSON; using defaults: %s", self.config.strategy_path)
            repaired = {
                **defaults,
                "last_strategy_repair_at": utc_now(),
                "last_strategy_repair_reason": "invalid_or_unreadable_strategy_file",
            }
            write_json_atomic(self.config.strategy_path, repaired)
            self._record_event(
                "strategy_file_repaired",
                {
                    "repaired": True,
                    "reason": "invalid_or_unreadable_strategy_file",
                    "path": str(self.config.strategy_path),
                },
            )
            return repaired
        merged = {**defaults, **payload}
        merged["focus_tracks"] = (
            [str(item).strip().lower() for item in merged.get("focus_tracks", []) if str(item).strip()]
            if isinstance(merged.get("focus_tracks"), list)
            else DEFAULT_TRACKS
        )
        merged["blocked_tasks"] = (
            [str(item) for item in merged.get("blocked_tasks", []) if str(item).strip()]
            if isinstance(merged.get("blocked_tasks"), list)
            else []
        )
        merged["deprioritized_tasks"] = (
            [str(item) for item in merged.get("deprioritized_tasks", []) if str(item).strip()]
            if isinstance(merged.get("deprioritized_tasks"), list)
            else []
        )
        return merged

    def _start_daemon(self) -> subprocess.Popen[str]:
        self.ensure_managed_daemon_pid_file()
        command = self._build_daemon_command()
        process = subprocess.Popen(command, cwd=self.config.repo_root, text=True)
        write_text_atomic(self._managed_daemon_pid_path(), f"{process.pid}\n")
        return process

    def _terminate(self, process: subprocess.Popen[str] | AdoptedManagedDaemonProcess) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
        pid_path = self._managed_daemon_pid_path()
        if pid_path.exists():
            if pid_path.is_dir():
                backup_path = unique_backup_path(pid_path, "directory-backup")
                pid_path.rename(backup_path)
            else:
                pid_path.unlink()
        self._record_event("daemon_stop", {"returncode": process.returncode})

    def _build_daemon_command(self) -> list[str]:
        daemon_script_path = self.config.daemon_script_path
        if daemon_script_path is None:
            command = [sys.executable, "-m", "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon"]
        else:
            command = [sys.executable, str(daemon_script_path)]
        command.extend(
            [
                "--interval",
                str(self.config.daemon_interval),
                "--todo-path",
                str(self.config.todo_path),
                "--state-dir",
                str(self.config.state_dir),
                "--task-prefix",
                self.config.task_prefix,
                "--state-prefix",
                self.config.state_prefix,
            ]
        )
        if self.config.implement:
            command.append("--implement")
            command.extend(["--implementation-timeout", str(self.config.implementation_timeout)])
            if self.config.implementation_command:
                command.extend(["--implementation-command", self.config.implementation_command])
            if self.config.llm_merge_resolver_command:
                command.extend(["--llm-merge-resolver-command", self.config.llm_merge_resolver_command])
            if self.config.llm_merge_resolver_timeout_seconds is not None:
                command.extend(
                    [
                        "--llm-merge-resolver-timeout-seconds",
                        str(self.config.llm_merge_resolver_timeout_seconds),
                    ]
                )
            if not self.config.use_ephemeral_worktree:
                command.append("--no-ephemeral-worktree")
            if self.config.worktree_root is not None:
                command.extend(["--worktree-root", str(self.config.worktree_root)])
            for relative in self.config.worktree_submodule_paths:
                command.extend(["--worktree-submodule-path", relative])
            if self.config.objective_path is not None:
                command.extend(["--objective-path", str(self.config.objective_path)])
            if self.config.objective_bundle_dir is not None:
                command.extend(["--objective-bundle-dir", str(self.config.objective_bundle_dir)])
        return command

    def _managed_daemon_pid_path(self) -> Path:
        return self.config.state_dir / f"{self.config.state_prefix}_managed_daemon.pid"

    def _read_managed_daemon_pid(self) -> int | None:
        try:
            raw_pid = self._managed_daemon_pid_path().read_text(encoding="utf-8").strip()
            return int(raw_pid)
        except (OSError, ValueError):
            return None

    def ensure_managed_daemon_pid_file(self) -> dict[str, Any]:
        """Remove stale or malformed managed-daemon PID state before adoption."""

        pid_path = self._managed_daemon_pid_path()
        if not pid_path.exists():
            return {"repaired": False, "reason": "missing", "path": str(pid_path)}
        if pid_path.is_dir():
            backup_path = unique_backup_path(pid_path, "directory-backup")
            pid_path.rename(backup_path)
            result = {
                "repaired": True,
                "reason": "managed_pid_path_was_directory",
                "path": str(pid_path),
                "backup_path": str(backup_path),
            }
            self._record_event("managed_daemon_pid_file_repaired", result)
            return result
        try:
            raw_pid = pid_path.read_text(encoding="utf-8").strip()
            pid = int(raw_pid)
        except (OSError, UnicodeDecodeError, ValueError):
            try:
                backup_path = unique_backup_path(pid_path, "invalid-pid")
                pid_path.rename(backup_path)
                result = {
                    "repaired": True,
                    "reason": "invalid_managed_pid_file",
                    "path": str(pid_path),
                    "backup_path": str(backup_path),
                }
            except OSError as exc:
                result = {
                    "repaired": False,
                    "reason": "invalid_managed_pid_file_unrepairable",
                    "path": str(pid_path),
                    "error": str(exc),
                }
            if result.get("repaired"):
                self._record_event("managed_daemon_pid_file_repaired", result)
            return result
        if pid <= 0:
            try:
                backup_path = unique_backup_path(pid_path, "invalid-pid")
                pid_path.rename(backup_path)
                result = {
                    "repaired": True,
                    "reason": "invalid_managed_pid",
                    "path": str(pid_path),
                    "pid": pid,
                    "backup_path": str(backup_path),
                }
            except OSError as exc:
                result = {
                    "repaired": False,
                    "reason": "invalid_managed_pid_unrepairable",
                    "path": str(pid_path),
                    "pid": pid,
                    "error": str(exc),
                }
            if result.get("repaired"):
                self._record_event("managed_daemon_pid_file_repaired", result)
            return result
        if not process_is_running(pid):
            try:
                pid_path.unlink()
                result = {
                    "repaired": True,
                    "reason": "stale_managed_pid",
                    "path": str(pid_path),
                    "pid": pid,
                }
            except OSError as exc:
                result = {
                    "repaired": False,
                    "reason": "stale_managed_pid_unrepairable",
                    "path": str(pid_path),
                    "pid": pid,
                    "error": str(exc),
                }
            if result.get("repaired"):
                self._record_event("managed_daemon_pid_file_repaired", result)
            return result
        command_line = process_command_line(pid)
        if not self._managed_daemon_matches_command_line(command_line):
            try:
                pid_path.unlink()
                result = {
                    "repaired": True,
                    "reason": "managed_pid_command_mismatch",
                    "path": str(pid_path),
                    "pid": pid,
                }
            except OSError as exc:
                result = {
                    "repaired": False,
                    "reason": "managed_pid_command_mismatch_unrepairable",
                    "path": str(pid_path),
                    "pid": pid,
                    "error": str(exc),
                }
            if result.get("repaired"):
                self._record_event("managed_daemon_pid_file_repaired", result)
            return result
        return {"repaired": False, "reason": "active", "path": str(pid_path), "pid": pid}

    def _adopt_existing_daemon(self) -> AdoptedManagedDaemonProcess | None:
        pid_path = self._managed_daemon_pid_path()
        repair = self.ensure_managed_daemon_pid_file()
        if repair.get("repaired") or not pid_path.exists() or pid_path.is_dir():
            return None
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            try:
                pid_path.unlink()
            except OSError:
                pass
            return None
        if not process_is_running(pid):
            try:
                pid_path.unlink()
            except OSError:
                pass
            return None
        command_line = process_command_line(pid)
        if not self._managed_daemon_matches_command_line(command_line):
            try:
                pid_path.unlink()
            except OSError:
                pass
            return None
        return AdoptedManagedDaemonProcess(pid)

    def _managed_daemon_matches_command_line(self, command_line: str) -> bool:
        daemon_script_path = self.config.daemon_script_path
        daemon_fragment = (
            Path(daemon_script_path).name
            if daemon_script_path is not None
            else "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon"
        )
        required_fragments = [
            daemon_fragment,
            "--state-dir",
            str(self.config.state_dir),
            "--state-prefix",
            self.config.state_prefix,
            "--todo-path",
            str(self.config.todo_path),
        ]
        if not all(fragment in command_line for fragment in required_fragments):
            return False
        has_implement_flag = "--implement" in command_line
        if self.config.implement != has_implement_flag:
            return False
        return True

    def _record_event(self, event_type: str, payload: dict[str, Any]) -> None:
        append_jsonl_event(self.config.events_path, event_type, payload)

    @staticmethod
    def _age_seconds(timestamp: str, now_ts: float) -> float:
        if not timestamp:
            return float("inf")
        try:
            parsed = datetime.fromisoformat(timestamp)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return max(0.0, now_ts - parsed.timestamp())
        except ValueError:
            return float("inf")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervise the portal implementation backlog daemon")
    parser.add_argument("--once", action="store_true", help="Run one supervisor check and exit")
    parser.add_argument(
        "--todo-path",
        type=Path,
        default=Path("docs/211_SERVICE_NAVIGATION_PORTAL_TODO.md"),
        help="Machine-readable markdown backlog",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path("data/portal_implementation/state"),
        help="Portal daemon state directory",
    )
    parser.add_argument("--stale-seconds", type=float, default=1800.0)
    parser.add_argument("--check-interval", type=float, default=60.0)
    parser.add_argument("--max-restarts", type=int, default=10)
    parser.add_argument("--daemon-interval", type=float, default=300.0)
    parser.add_argument(
        "--task-prefix",
        default=TASK_HEADER_PREFIX,
        help="Markdown heading prefix for tasks, for example '## PORTAL-' or '## AGENT-'",
    )
    parser.add_argument(
        "--state-prefix",
        default="portal",
        help="State file prefix inside --state-dir",
    )
    implement_group = parser.add_mutually_exclusive_group()
    implement_group.add_argument(
        "--implement",
        dest="implement",
        action="store_true",
        help="Allow the managed daemon to invoke the implementation agent",
    )
    implement_group.add_argument(
        "--no-implement",
        dest="implement",
        action="store_false",
        help="Only supervise backlog state; do not let the managed daemon invoke the implementation agent",
    )
    parser.set_defaults(implement=False)
    parser.add_argument(
        "--implementation-command",
        default="",
        help="Command used by the daemon for implementation. Defaults to codex exec --full-auto.",
    )
    parser.add_argument(
        "--llm-merge-resolver-command",
        default=os.environ.get("IPFS_ACCELERATE_AGENT_LLM_MERGE_RESOLVER_COMMAND", ""),
        help=(
            "Command invoked with merge-conflict repair prompts on stdin. "
            "Passed to the managed daemon as IPFS_ACCELERATE_AGENT_LLM_MERGE_RESOLVER_COMMAND."
        ),
    )
    parser.add_argument(
        "--llm-merge-resolver-timeout-seconds",
        type=float,
        default=None,
        help=(
            "Timeout for the merge resolver subprocess. Passed to the managed daemon as "
            "IPFS_ACCELERATE_AGENT_LLM_MERGE_RESOLVER_TIMEOUT_SECONDS; defaults to that env var "
            "or 600 seconds; <=0 disables."
        ),
    )
    parser.add_argument(
        "--allow-reconciliation-only-llm-resolver",
        action="store_true",
        help=(
            "Allow --reconciliation-only passes to invoke the configured LLM merge resolver. "
            "By default reconciliation-only disables this to keep cleanup probes non-interactive."
        ),
    )
    parser.add_argument("--implementation-timeout", type=float, default=1800.0)
    parser.add_argument(
        "--implementation-log-stall-seconds",
        type=float,
        default=300.0,
        help="Recycle an active implementation attempt after this many seconds without log output; <=0 disables.",
    )
    parser.add_argument(
        "--no-ephemeral-worktree",
        action="store_true",
        help="Run implementation commands in the main checkout instead of isolated temporary git worktrees",
    )
    parser.add_argument(
        "--worktree-root",
        type=Path,
        default=None,
        help="Directory for temporary implementation worktrees",
    )
    parser.add_argument(
        "--daemon-script-path",
        type=Path,
        default=None,
        help="Python script used to launch the managed daemon instead of the package module.",
    )
    parser.add_argument(
        "--supervisor-script-path",
        type=Path,
        default=None,
        help="Python script used to relaunch this supervisor from external wrappers.",
    )
    parser.add_argument(
        "--worktree-submodule-path",
        action="append",
        default=[],
        help=(
            "Repo-relative submodule path to initialize and commit inside implementation worktrees. "
            "May be repeated or comma-separated."
        ),
    )
    parser.add_argument(
        "--reconciliation-only",
        action="store_true",
        help=(
            "Run only supervisor reconciliation/cleanup checks. This disables implementation, "
            "retry/dependency/reconciliation guardrail writes, and objective/codebase refill scans."
        ),
    )
    parser.add_argument(
        "--no-worktree-reconciliation",
        dest="worktree_reconciliation_enabled",
        action="store_false",
        help="Disable supervisor retry/cleanup reconciliation for clean inactive implementation worktrees.",
    )
    parser.set_defaults(worktree_reconciliation_enabled=True)
    parser.add_argument(
        "--worktree-reconciliation-max-merges",
        type=int,
        default=1,
        help="Maximum clean backlogged implementation branches to merge per supervisor pass.",
    )
    parser.add_argument(
        "--worktree-reconciliation-dry-run",
        action="store_true",
        help="Classify clean backlogged implementation worktrees without merging or removing them.",
    )
    parser.add_argument(
        "--no-worktree-reconciliation-preflight",
        dest="worktree_reconciliation_preflight_enabled",
        action="store_false",
        help=(
            "Disable non-mutating merge-tree preflight before merging a backlogged "
            "implementation branch into the main checkout."
        ),
    )
    parser.set_defaults(worktree_reconciliation_preflight_enabled=True)
    parser.add_argument(
        "--no-worktree-scan-cache",
        dest="worktree_scan_cache_enabled",
        action="store_false",
        help="Disable cached non-mutating worktree reconciliation/cleanup classifications.",
    )
    parser.set_defaults(worktree_scan_cache_enabled=True)
    parser.add_argument(
        "--worktree-scan-cache-ttl-seconds",
        type=float,
        default=DEFAULT_WORKTREE_SCAN_CACHE_TTL_SECONDS,
        help="Seconds to reuse cached non-mutating worktree scan classifications; <=0 disables the cache.",
    )
    parser.add_argument(
        "--worktree-scan-cache-path",
        type=Path,
        default=None,
        help="JSON cache path for non-mutating worktree scan classifications. Defaults to the supervisor state dir.",
    )
    parser.add_argument(
        "--no-retry-budget-guardrail",
        dest="retry_budget_guardrail_enabled",
        action="store_false",
        help="Disable conversion of repeated implementation, validation, or merge failures into follow-up tasks.",
    )
    parser.set_defaults(retry_budget_guardrail_enabled=True)
    parser.add_argument(
        "--retry-budget-discovery-dir",
        type=Path,
        default=None,
        help="Directory for retry-budget discovery reports. Defaults to a sibling discovery directory near state.",
    )
    parser.add_argument("--retry-budget-discovery-output-path", default="")
    parser.add_argument("--validation-retry-budget", type=int, default=3)
    parser.add_argument("--merge-retry-budget", type=int, default=3)
    parser.add_argument("--implementation-retry-budget", type=int, default=3)
    parser.add_argument(
        "--retry-budget-commit-outputs",
        action="store_true",
        help="Commit generated retry-budget todo/discovery outputs.",
    )
    parser.add_argument(
        "--retry-budget-commit-subject",
        default="Agent: record retry-budget guardrail outputs",
    )
    parser.add_argument(
        "--no-dependency-guardrail",
        dest="dependency_guardrail_enabled",
        action="store_false",
        help="Disable conversion of missing/self-referential dependencies into ready repair tasks.",
    )
    parser.set_defaults(dependency_guardrail_enabled=True)
    parser.add_argument("--dependency-guardrail-discovery-dir", type=Path, default=None)
    parser.add_argument("--dependency-guardrail-discovery-output-path", default="")
    parser.add_argument("--dependency-guardrail-max-findings", type=int, default=5)
    parser.add_argument(
        "--dependency-guardrail-commit-outputs",
        action="store_true",
        help="Commit generated dependency-guardrail todo/discovery outputs.",
    )
    parser.add_argument(
        "--dependency-guardrail-commit-subject",
        default="Agent: record dependency guardrail outputs",
    )
    parser.add_argument(
        "--no-reconciliation-guardrail",
        dest="reconciliation_guardrail_enabled",
        action="store_false",
        help="Disable conversion of blocked checkout/worktree reconciliation into ready cleanup tasks.",
    )
    parser.set_defaults(reconciliation_guardrail_enabled=True)
    parser.add_argument("--reconciliation-guardrail-discovery-dir", type=Path, default=None)
    parser.add_argument("--reconciliation-guardrail-discovery-output-path", default="")
    parser.add_argument("--reconciliation-guardrail-max-findings", type=int, default=3)
    parser.add_argument(
        "--reconciliation-guardrail-commit-outputs",
        action="store_true",
        help="Commit generated reconciliation-guardrail todo/discovery outputs.",
    )
    parser.add_argument(
        "--reconciliation-guardrail-commit-subject",
        default="Agent: record reconciliation guardrail outputs",
    )
    parser.add_argument(
        "--codebase-refill-scan",
        action="store_true",
        help="Append codebase-scan follow-up tasks when the supervised backlog is low or drained.",
    )
    parser.add_argument(
        "--codebase-scan-discovery-dir",
        type=Path,
        default=None,
        help="Directory for codebase-scan discovery reports. Defaults to a sibling discovery directory near state.",
    )
    parser.add_argument(
        "--codebase-scan-discovery-output-path",
        default="",
        help="Todo Outputs path used for generated codebase-scan tasks.",
    )
    parser.add_argument(
        "--codebase-scan-min-open-tasks",
        type=int,
        default=0,
        help="Run the refill scan when open tasks are at or below this count.",
    )
    parser.add_argument("--codebase-scan-max-findings", type=int, default=5)
    parser.add_argument("--codebase-scan-cooldown-seconds", type=int, default=21600)
    parser.add_argument(
        "--codebase-scan-depends-on",
        action="append",
        default=[],
        help="Task id dependency for generated codebase-scan tasks. May be repeated or comma-separated.",
    )
    parser.add_argument(
        "--codebase-scan-skip-prefix",
        action="append",
        default=[],
        help="Repo-relative path prefix to skip during codebase scans. May be repeated.",
    )
    parser.add_argument(
        "--codebase-scan-commit-outputs",
        action="store_true",
        help="Commit generated todo/discovery outputs after a supervisor codebase scan.",
    )
    parser.add_argument(
        "--allow-codebase-refill-with-objective-work",
        dest="codebase_defer_when_objective_refills",
        action="store_false",
        help="Allow codebase-scan refill in the same supervisor pass that objective refill creates goal work.",
    )
    parser.set_defaults(codebase_defer_when_objective_refills=True)
    parser.add_argument(
        "--codebase-scan-commit-subject",
        default="Agent: record supervisor codebase scan findings",
    )
    parser.add_argument(
        "--objective-refill-scan",
        action="store_true",
        help="Refine the objective heap and append objective-gap todos when the supervised backlog is low or drained.",
    )
    parser.add_argument(
        "--objective-path",
        type=Path,
        default=None,
        help="Objective goal heap markdown document. Defaults to implementation_plan/docs/23-virtual-ai-os-objective-goal-heap.md.",
    )
    parser.add_argument("--objective-graph-path", type=Path, default=None)
    parser.add_argument("--objective-bundle-dir", type=Path, default=None)
    parser.add_argument("--objective-dataset-dir", type=Path, default=None)
    parser.add_argument("--objective-discovery-dir", type=Path, default=None)
    parser.add_argument("--objective-discovery-output-path", default="")
    parser.add_argument("--objective-summary-prefix", default="")
    parser.add_argument(
        "--no-objective-goal-refinement",
        dest="objective_refine_goals",
        action="store_false",
        help="Generate todos from the objective heap without appending new subgoals.",
    )
    parser.set_defaults(objective_refine_goals=True)
    parser.add_argument(
        "--no-objective-goal-completion-reconcile",
        dest="objective_reconcile_goal_completion",
        action="store_false",
        help="Do not mark objective goals completed when their required evidence is already present.",
    )
    parser.set_defaults(objective_reconcile_goal_completion=True)
    parser.add_argument(
        "--objective-seed-interoperability-goals",
        action="store_true",
        help="Seed objective goals for cross-submodule interoperability and integration tests.",
    )
    parser.add_argument(
        "--objective-interoperability-focus",
        action="append",
        default=[],
        help=(
            "Submodule path to pair with other submodules for interoperability goal seeding. "
            "If omitted, all submodule pairs are eligible."
        ),
    )
    parser.add_argument(
        "--objective-interoperability-component-path",
        action="append",
        default=[],
        help=(
            "Repo-relative component path to include when seeding interoperability goals. "
            "Defaults to configured worktree submodule paths when omitted."
        ),
    )
    parser.add_argument("--objective-max-interoperability-goals", type=int, default=12)
    parser.add_argument(
        "--objective-ensure-tracking-document",
        action="store_true",
        help="Create the objective heap with a root goal if it does not exist.",
    )
    parser.add_argument("--objective-ultimate-goal", default="")
    parser.add_argument("--objective-root-evidence", action="append", default=[])
    parser.add_argument("--objective-goal-prefix", default=None)
    parser.add_argument("--objective-root-goal-id", default=None)
    parser.add_argument("--objective-root-goal-title", default="")
    parser.add_argument("--objective-tracking-document-title", default="")
    parser.add_argument("--objective-scan-min-open-tasks", type=int, default=0)
    parser.add_argument("--objective-scan-max-findings", type=int, default=5)
    parser.add_argument("--objective-scan-cooldown-seconds", type=int, default=21600)
    parser.add_argument(
        "--objective-scan-depends-on",
        action="append",
        default=[],
        help="Task id dependency for generated objective tasks. May be repeated or comma-separated.",
    )
    parser.add_argument("--objective-max-refinement-children", type=int, default=3)
    parser.add_argument("--objective-max-refinement-depth", type=int, default=4)
    parser.add_argument(
        "--objective-surplus-findings-per-goal",
        type=int,
        default=DEFAULT_OBJECTIVE_SURPLUS_FINDINGS_PER_GOAL,
        help=(
            "Generate surplus structured objective todos per missing goal. "
            "Additional candidates are vector-indexed and bundled with related work."
        ),
    )
    parser.add_argument(
        "--objective-surplus-min-terms-per-todo",
        type=int,
        default=DEFAULT_OBJECTIVE_SURPLUS_MIN_TERMS_PER_TODO,
        help="Minimum missing-evidence terms for non-aggregate objective surplus todos.",
    )
    parser.add_argument(
        "--objective-todo-vector-index-path",
        type=Path,
        default=None,
        help="Path for the objective todo vector/AST index. Defaults to <objective-bundle-dir>/todo_vector_index.json.",
    )
    parser.add_argument(
        "--no-objective-ast-dataset",
        dest="objective_persist_ast_dataset",
        action="store_false",
        help="Skip persisting the objective AST/evidence dataset while refilling.",
    )
    parser.set_defaults(objective_persist_ast_dataset=True)
    parser.add_argument(
        "--no-objective-todo-vector-index",
        dest="objective_write_todo_vector_index",
        action="store_false",
        help="Skip writing the objective todo vector/AST index while refilling.",
    )
    parser.set_defaults(objective_write_todo_vector_index=True)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args(argv)


def supervisor_config_from_args(
    args: argparse.Namespace,
    *,
    repo_root: Path | None = None,
    daemon_script_path: Path | None = None,
    supervisor_script_path: Path | None = None,
    worktree_submodule_paths: Any = None,
    state_path: Path | None = None,
    strategy_path: Path | None = None,
    events_path: Path | None = None,
) -> PortalSupervisorConfig:
    """Build a supervisor config from parsed CLI args with optional embedding overrides."""

    resolved_worktree_submodule_paths = (
        args.worktree_submodule_path if worktree_submodule_paths is None else worktree_submodule_paths
    )
    reconciliation_only = bool(args.reconciliation_only)
    implement = bool(args.implement and not reconciliation_only)
    llm_merge_resolver_command = args.llm_merge_resolver_command
    if reconciliation_only and not args.allow_reconciliation_only_llm_resolver:
        llm_merge_resolver_command = ""
    return PortalSupervisorConfig(
        todo_path=args.todo_path,
        state_path=state_path or args.state_dir / f"{args.state_prefix}_task_state.json",
        strategy_path=strategy_path or args.state_dir / f"{args.state_prefix}_strategy.json",
        events_path=events_path or args.state_dir / f"{args.state_prefix}_supervisor_events.jsonl",
        state_dir=args.state_dir,
        stale_seconds=args.stale_seconds,
        check_interval=args.check_interval,
        max_restarts=args.max_restarts,
        daemon_interval=args.daemon_interval,
        task_prefix=args.task_prefix,
        state_prefix=args.state_prefix,
        reconciliation_only=reconciliation_only,
        implement=implement,
        implementation_command=args.implementation_command,
        llm_merge_resolver_command=llm_merge_resolver_command,
        llm_merge_resolver_timeout_seconds=args.llm_merge_resolver_timeout_seconds,
        implementation_timeout=args.implementation_timeout,
        implementation_log_stall_seconds=args.implementation_log_stall_seconds,
        use_ephemeral_worktree=implement and not args.no_ephemeral_worktree,
        worktree_root=args.worktree_root,
        worktree_submodule_paths=normalize_relative_path_list(resolved_worktree_submodule_paths),
        worktree_reconciliation_enabled=args.worktree_reconciliation_enabled,
        worktree_reconciliation_max_merges=args.worktree_reconciliation_max_merges,
        worktree_reconciliation_dry_run=args.worktree_reconciliation_dry_run,
        worktree_reconciliation_preflight_enabled=args.worktree_reconciliation_preflight_enabled,
        worktree_scan_cache_enabled=args.worktree_scan_cache_enabled,
        worktree_scan_cache_ttl_seconds=args.worktree_scan_cache_ttl_seconds,
        worktree_scan_cache_path=args.worktree_scan_cache_path,
        retry_budget_guardrail_enabled=args.retry_budget_guardrail_enabled and not reconciliation_only,
        retry_budget_discovery_dir=args.retry_budget_discovery_dir,
        retry_budget_discovery_output_path=args.retry_budget_discovery_output_path,
        validation_retry_budget=args.validation_retry_budget,
        merge_retry_budget=args.merge_retry_budget,
        implementation_retry_budget=args.implementation_retry_budget,
        retry_budget_commit_outputs=args.retry_budget_commit_outputs,
        retry_budget_commit_subject=args.retry_budget_commit_subject,
        dependency_guardrail_enabled=args.dependency_guardrail_enabled and not reconciliation_only,
        dependency_guardrail_discovery_dir=args.dependency_guardrail_discovery_dir,
        dependency_guardrail_discovery_output_path=args.dependency_guardrail_discovery_output_path,
        dependency_guardrail_max_findings=args.dependency_guardrail_max_findings,
        dependency_guardrail_commit_outputs=args.dependency_guardrail_commit_outputs,
        dependency_guardrail_commit_subject=args.dependency_guardrail_commit_subject,
        reconciliation_guardrail_enabled=args.reconciliation_guardrail_enabled,
        reconciliation_guardrail_discovery_dir=args.reconciliation_guardrail_discovery_dir,
        reconciliation_guardrail_discovery_output_path=args.reconciliation_guardrail_discovery_output_path,
        reconciliation_guardrail_max_findings=args.reconciliation_guardrail_max_findings,
        reconciliation_guardrail_commit_outputs=args.reconciliation_guardrail_commit_outputs,
        reconciliation_guardrail_commit_subject=args.reconciliation_guardrail_commit_subject,
        codebase_refill_enabled=args.codebase_refill_scan and not reconciliation_only,
        codebase_scan_discovery_dir=args.codebase_scan_discovery_dir,
        codebase_scan_discovery_output_path=args.codebase_scan_discovery_output_path,
        codebase_scan_min_open_tasks=args.codebase_scan_min_open_tasks,
        codebase_scan_max_findings=args.codebase_scan_max_findings,
        codebase_scan_cooldown_seconds=args.codebase_scan_cooldown_seconds,
        codebase_scan_depends_on=split_csv_values(args.codebase_scan_depends_on),
        codebase_scan_skip_prefixes=tuple(args.codebase_scan_skip_prefix),
        codebase_defer_when_objective_refills=args.codebase_defer_when_objective_refills,
        codebase_scan_commit_outputs=args.codebase_scan_commit_outputs,
        codebase_scan_commit_subject=args.codebase_scan_commit_subject,
        objective_refill_enabled=args.objective_refill_scan and not reconciliation_only,
        objective_path=args.objective_path,
        objective_graph_path=args.objective_graph_path,
        objective_bundle_dir=args.objective_bundle_dir,
        objective_dataset_dir=args.objective_dataset_dir,
        objective_discovery_dir=args.objective_discovery_dir,
        objective_discovery_output_path=args.objective_discovery_output_path,
        objective_summary_prefix=args.objective_summary_prefix,
        objective_refine_goals=args.objective_refine_goals,
        objective_reconcile_goal_completion=args.objective_reconcile_goal_completion,
        objective_seed_interoperability_goals=args.objective_seed_interoperability_goals,
        objective_interoperability_focus=split_csv_values(args.objective_interoperability_focus),
        objective_interoperability_component_paths=split_csv_values(
            args.objective_interoperability_component_path
        ),
        objective_max_interoperability_goals=args.objective_max_interoperability_goals,
        objective_ensure_tracking_document=args.objective_ensure_tracking_document,
        objective_ultimate_goal=args.objective_ultimate_goal,
        objective_root_evidence=split_csv_values(args.objective_root_evidence),
        objective_goal_prefix=args.objective_goal_prefix,
        objective_root_goal_id=args.objective_root_goal_id,
        objective_root_goal_title=args.objective_root_goal_title,
        objective_tracking_document_title=args.objective_tracking_document_title,
        objective_scan_min_open_tasks=args.objective_scan_min_open_tasks,
        objective_scan_max_findings=args.objective_scan_max_findings,
        objective_scan_cooldown_seconds=args.objective_scan_cooldown_seconds,
        objective_scan_depends_on=split_csv_values(args.objective_scan_depends_on),
        objective_max_refinement_children=args.objective_max_refinement_children,
        objective_max_refinement_depth=args.objective_max_refinement_depth,
        objective_persist_ast_dataset=args.objective_persist_ast_dataset,
        objective_write_todo_vector_index=args.objective_write_todo_vector_index,
        objective_todo_vector_index_path=args.objective_todo_vector_index_path,
        objective_surplus_findings_per_goal=args.objective_surplus_findings_per_goal,
        objective_surplus_min_terms_per_todo=args.objective_surplus_min_terms_per_todo,
        repo_root=repo_root or REPO_ROOT,
        daemon_script_path=daemon_script_path if daemon_script_path is not None else args.daemon_script_path,
        supervisor_script_path=supervisor_script_path
        if supervisor_script_path is not None
        else args.supervisor_script_path,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    supervisor = PortalImplementationSupervisor(supervisor_config_from_args(args, repo_root=REPO_ROOT))
    if args.once:
        result = supervisor.run_once()
        logger.info("Portal implementation supervisor check complete: %s", result)
        return
    supervisor.run_forever()


if __name__ == "__main__":
    main()


TodoSupervisorConfig = PortalSupervisorConfig
TodoImplementationSupervisor = PortalImplementationSupervisor
