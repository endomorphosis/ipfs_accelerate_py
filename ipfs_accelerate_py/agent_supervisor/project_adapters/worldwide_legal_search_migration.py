"""Fenced native CAS correction of legacy worldwide validation argv encoding.

The task meaning, aliases, outputs, dependencies, lifecycle and execution
history stay intact. Only the accidental one-element shell-command encoding is
replaced by the exact argv already admitted in plan.json.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shlex
import sys
import tempfile
import time
from typing import Any, Callable, Mapping

from .worldwide_legal_search_runtime import CampaignConfig, atomic_json, config_from_args, identity, observed_forest, owner_lock, parser as runtime_parser, read_json


def candidate_population(projection: Mapping[str, Any], plan: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    expected = {task["id"]: task for task in plan["tasks"]}
    if {task["task_alias"] for task in projection["tasks"]} != set(expected):
        raise ValueError("migration population differs from the admitted plan")
    trees = {task["identity"]["repository_tree_id"] for task in projection["tasks"]}
    if len(trees) != 1:
        raise ValueError("migration requires the existing exact repository forest identity")
    changed, tasks = [], []
    for record in sorted(projection["tasks"], key=lambda item: item["ordinal"]):
        alias = record["task_alias"]
        wanted = expected[alias]["validation_argv"]
        prior = [item["argv"] for item in record["validations"]]
        legacy = [[shlex.join(argv)] for argv in wanted]
        if prior not in (wanted, legacy):
            raise ValueError("validation differs from both admitted argv and the exact legacy defect: " + alias)
        if prior != wanted:
            if record["status"] not in {"todo", "pending", "ready", "queued", "retrying", "blocked", "proposed", "admitted"}:
                raise ValueError("native lifecycle forbids amendment: " + alias + ":" + record["status"])
            changed.append(alias)
        task = {**copy.deepcopy(record["body"]), **{key: record[key] for key in ("task_cid", "task_alias", "goal_cid", "plan_cid", "objective_id", "ordinal", "status", "priority")},
            "depends_on": [item["dependency_task_cid"] for item in record["dependencies"]],
            "outputs": [copy.deepcopy(item["effect"]) for item in record["outputs"]],
            "acceptance": [copy.deepcopy(item["evidence_policy"]) for item in record["acceptance"]],
            "validations": [{"argv": list(argv), **copy.deepcopy(item["policy"])} for argv, item in zip(wanted, record["validations"], strict=True)]}
        tasks.append(task)
    goals = [{**copy.deepcopy(goal["body"]), **{key: goal[key] for key in ("goal_cid", "goal_alias", "objective_id", "parent_goal_cid", "ordinal", "title", "status")}} for goal in projection["goals"]]
    plans = [{**copy.deepcopy(item["body"]), **{key: item[key] for key in ("plan_cid", "goal_cid", "plan_alias", "status")}} for item in projection["plans"]]
    return {"objectives": goals, "goal_edges": copy.deepcopy(projection["goal_edges"]), "plans": plans, "tasks": tasks,
        "repository_tree_id": next(iter(trees))}, changed


def _registered_stopped_fence(config: CampaignConfig) -> dict[str, Any]:
    from .worldwide_legal_search_host_lifecycle import RegisteredSystemdLifecycle
    lifecycle = RegisteredSystemdLifecycle(repository=config.site_root, state_root=config.state_root, run_id=config.run_id)
    fence = lifecycle.probe()
    if fence["owner_alive"] or fence["main_pid"] != 0 or not lifecycle._cgroup_empty(fence["control_group"]):
        raise ValueError("registered supervisor and its entire cgroup must be stopped before migration")
    return {**fence, "cgroup_empty": True}


def operator_rearm(config: CampaignConfig, *, task_alias: str, attempt_id: str, clearance_path: Path,
                   acknowledge_paid_attempt: bool, apply: bool = False,
                   fence_probe: Callable[[CampaignConfig], Mapping[str, Any]] = _registered_stopped_fence) -> dict[str, Any]:
    """Explicit operator retry after genuine dispatch; never claim non-dispatch."""
    from .worldwide_legal_search_watchdog import read_owned_json
    from ..todo_daemon.implementation_daemon import DatabaseImplementationDaemon, canonical_json
    from ..merge.database_coordination import TASK_CLAIM_FAILURE_REARM_OPERATION
    if not acknowledge_paid_attempt:
        raise ValueError("operator retry must acknowledge the preserved, potentially paid provider attempt")
    with owner_lock(config):
        fence = dict(fence_probe(config))
        if fence.get("owner_alive") is not False or fence.get("main_pid") != 0 or fence.get("cgroup_empty") is not True:
            raise ValueError("operator retry requires the exact stopped owner fence")
        clearance = read_owned_json(clearance_path)
        if clearance.get("schema") != "implementation-protected-path-clearance-v1" or clearance.get("task_id") != task_alias:
            raise ValueError("operator retry requires the native clearance for this task")
        basis = {key: clearance[key] for key in ("incident", "before_head", "after_head", "approved_commits", "protected_path_history_unchanged",
            "shared_checkout_rollback_proof", "disposed_ephemeral_workspace_proof", "mirrored_ephemeral_workspace_proof")}
        if clearance.get("clearance_id") != "sha256:" + hashlib.sha256(canonical_json(basis).encode()).hexdigest():
            raise ValueError("native clearance integrity differs")
        from .worldwide_legal_search_runtime import git
        if clearance["after_head"] != git(config.site_root, "rev-parse", "HEAD"):
            raise ValueError("shared checkout changed after native clearance")
        paths = config.state_root / "native/wwl_database_portal_attempts"
        matches = [path for path in paths.glob("*/database-attempt-binding.json") if read_owned_json(path).get("attempt_id") == attempt_id]
        if len(matches) != 1 or clearance_path.resolve().parent != matches[0].parent.resolve():
            raise ValueError("clearance is outside the exact retained attempt directory")
        binding = read_owned_json(matches[0])
        if binding.get("task_alias") != task_alias or any((matches[0].parent / name).exists() for name in ("implementation-protected-path-active.json", "implementation-protected-path-incident.json")):
            raise ValueError("the exact native predecessor remains fenced")
        daemon = DatabaseImplementationDaemon(database_path=config.database, coordination_path=config.state_root / "coordination.duckdb",
            execution_path=config.state_root / "execution.duckdb", state_dir=config.state_root / "native", state_prefix="wwl",
            owner_session_id=config.run_id, authority_mode="embedded_exclusive", require_real_execution=True, task_prefix="WWL-").open()
        try:
            if daemon.list_running_attempts():
                raise ValueError("operator retry cannot replace a running native attempt")
            attempt = daemon.get_attempt(attempt_id)
            if attempt is None or attempt.task_alias != task_alias or attempt.status != "failed" or attempt.attempt_number >= config.max_task_attempts:
                raise ValueError("operator retry requires this failed attempt with remaining native budget")
            failures = [phase["body"] for phase in daemon.phase_history(attempt_id) if phase.get("phase") == "failed" and phase.get("body", {}).get("failure_kind") == "terminal_portal_bridge_error"]
            if len(failures) != 1:
                raise ValueError("native failed phase does not provide one exact failure receipt")
            failure = failures[0]
            if failure.get("settlement_id") != daemon.task_source.get_task(task_alias).body.get("completion_receipt", {}).get("settlement_id"):
                raise ValueError("current control receipt does not bind the failed native attempt")
            if not daemon._portal_failure_matches_recoverable_reason(failure, attempt, "implementation_protected_path_mutated"):
                raise ValueError("operator retry reason differs from the cleared protected incident")
            compact = {"operation": TASK_CLAIM_FAILURE_REARM_OPERATION, "settlement_id": failure["settlement_id"]}
            task = daemon.task_source.get_task(task_alias)
            historical = daemon.coordinator.observe_rearmed_failed_task_claim(daemon.coordinator.get_task_claim(attempt.claim_id), failure_receipt=failure)
            report = {"schema": "wwl.operator-protected-retry@1", "task_alias": task_alias, "attempt_id": attempt_id,
                "native_failure_settlement": failure["settlement_id"], "native_clearance": clearance["clearance_id"],
                "attempt_number_preserved": attempt.attempt_number, "paid_provider_attempt_acknowledged": True,
                "non_dispatch_claimed": False, "attempt_budget_reset": False, "completion_authority": False}
            if historical is not None and task.status == "retrying" and dict(task.body.get("completion_receipt", {})) == compact:
                return {**report, "applied": True, "replayed": True, "coordination_receipt": dict(historical)}
            if task.status != "blocked" or task.revision != int(failure["control_expected_revision"]) + 1:
                raise ValueError("operator retry must precede specification amendments and follow the exact block CAS")
            if not apply:
                return {**report, "applied": False, "expected_revision": task.revision}
            changed = daemon.task_source.rearm_blocked_task(task.task_cid, receipt=compact)
            coordinated = daemon.coordinator.rearm_failed_task(failure_receipt=failure, control_task_observation=changed.to_dict())
            if daemon.get_attempt(attempt_id).attempt_number != attempt.attempt_number or daemon.get_attempt(attempt_id).status != "failed":
                raise RuntimeError("operator retry changed immutable failed attempt history")
            output = {**report, "applied": True, "control_revision": changed.revision, "coordination_receipt": coordinated}
            atomic_json(config.state_root / "maintenance" / ("operator-retry-" + task_alias + ".json"), output)
            return output
        finally:
            daemon.close()


def operator_rearm_interrupted_provider(config: CampaignConfig, *, task_alias: str, attempt_id: str,
        watchdog_action_id: str, acknowledge_paid_attempt: bool, apply: bool = False,
        fence_probe: Callable[[CampaignConfig], Mapping[str, Any]] = _registered_stopped_fence,
        supersession_verifier: Callable[..., Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Explicit retry of one locally fenced, interrupted native provider.

    A failed watchdog operation is never relabeled recovered. Its later native
    fence receipt must prove custody of that same old PID/birth, while the
    canonical failed phase and final Portal event prove no accepted task effect.
    """
    from datetime import datetime
    from .worldwide_legal_search_watchdog import WorldwideWatchdogBridge, read_owned_json
    from ..todo_daemon.implementation_daemon import DatabaseImplementationDaemon
    from ..merge.database_coordination import TASK_CLAIM_FAILURE_REARM_OPERATION
    if not acknowledge_paid_attempt:
        raise ValueError("interrupted provider retry requires explicit paid-attempt acknowledgment")
    with owner_lock(config):
        fence = dict(fence_probe(config))
        if fence.get("owner_alive") is not False or fence.get("main_pid") != 0 or fence.get("cgroup_empty") is not True:
            raise ValueError("interrupted provider retry requires the stopped registered owner")
        verifier = supersession_verifier or WorldwideWatchdogBridge(repository=config.site_root, state_root=config.state_root, run_id=config.run_id).verify_host_supersession
        proof = dict(verifier(watchdog_action_id, expected_task_id=task_alias, expected_stage="provider", require_owner_stopped=True))
        request = proof.get("action", {})
        if proof.get("schema") != "wwl.watchdog.verified-host-supersession@1" or request.get("kind") != "recover" or request.get("task_id") != task_alias or request.get("stage") != "provider" or request.get("run_id") != config.run_id:
            raise ValueError("native watchdog interruption proof does not bind this task")
        roots = config.state_root / "native/wwl_database_portal_attempts"
        bindings = [path for path in roots.glob("*/database-attempt-binding.json") if read_owned_json(path).get("attempt_id") == attempt_id]
        if len(bindings) != 1:
            raise ValueError("interrupted provider has no exact retained native binding")
        binding = read_owned_json(bindings[0])
        event_path = bindings[0].parent / "portal-events.jsonl"
        if event_path.is_symlink() or event_path.stat().st_uid != __import__("os").getuid() or event_path.stat().st_size > 1_048_576:
            raise ValueError("interrupted provider event stream custody is invalid")
        raw_events = event_path.read_bytes()
        events = [json.loads(line) for line in raw_events.splitlines() if line.strip()]
        finished = [event for event in events if event.get("type") == "implementation_finished" and event.get("task_id") == task_alias]
        if not finished:
            raise ValueError("interrupted provider has no durable native terminal observation")
        last = finished[-1]
        if (last.get("provider_dispatched") is not True or last.get("exception_result", {}).get("exception_type") != "ProcessGroupCleanupUnverified"
                or last.get("validation_result", {}).get("attempted") is not False
                or last.get("commit_result", {}).get("committed") is not False
                or last.get("merge_result", {}).get("merged") is not False
                or last.get("board_completion", {}).get("complete") is not False):
            raise ValueError("native provider outcome is not an unaccepted interrupted implementation")
        finished_at = datetime.fromisoformat(last["timestamp"]).timestamp()
        issued_at = request.get("issued_at")
        if type(issued_at) not in (int, float) or not 0 <= finished_at - issued_at <= 120:
            raise ValueError("watchdog interruption does not precede this exact native terminal event")
        daemon = DatabaseImplementationDaemon(database_path=config.database, coordination_path=config.state_root / "coordination.duckdb",
            execution_path=config.state_root / "execution.duckdb", state_dir=config.state_root / "native", state_prefix="wwl",
            owner_session_id=config.run_id, authority_mode="embedded_exclusive", require_real_execution=True, task_prefix="WWL-").open()
        try:
            if daemon.list_running_attempts():
                raise ValueError("a running native attempt still owns execution")
            attempt = daemon.get_attempt(attempt_id)
            if (attempt is None or attempt.task_alias != task_alias or attempt.task_cid != binding.get("task_cid") or attempt.status != "failed"
                    or attempt.attempt_number >= config.max_task_attempts or attempt.started_at_ms / 1000 > issued_at):
                raise ValueError("native failed attempt identity or remaining retry budget differs")
            phases = daemon.phase_history(attempt_id)
            if any(phase.get("phase") in {"effect", "validation", "completed"} for phase in phases):
                raise ValueError("accepted native effects prevent interrupted-provider retry")
            failure = daemon._portal_failure_phase_receipt(attempt)
            if failure is None or not daemon._portal_failure_matches_recoverable_reason(failure, attempt, "portal_provider_failed"):
                raise ValueError("failed native phase does not bind the provider failure")
            task = daemon.task_source.get_task(task_alias)
            compact = {"operation": TASK_CLAIM_FAILURE_REARM_OPERATION, "settlement_id": failure["settlement_id"]}
            historical = daemon.coordinator.observe_rearmed_failed_task_claim(daemon.coordinator.get_task_claim(attempt.claim_id), failure_receipt=failure)
            report = {"schema": "wwl.operator-fenced-provider-retry@1", "evidence_kind": "fenced_interrupted_provider", "run_id": config.run_id,
                "task_alias": task_alias, "attempt_id": attempt_id, "watchdog_action_id": watchdog_action_id,
                "supersession_receipt": proof.get("supersession_receipt"), "native_failure_settlement": failure["settlement_id"],
                "native_terminal_event": last.get("event_id"), "native_events_sha256": hashlib.sha256(raw_events).hexdigest(),
                "attempt_number_preserved": attempt.attempt_number, "paid_provider_attempt_acknowledged": True,
                "possible_external_effects": "unverified", "non_dispatch_claimed": False, "attempt_budget_reset": False, "completion_authority": False}
            if historical is not None and task.status == "retrying" and dict(task.body.get("completion_receipt", {})) == compact:
                return {**report, "applied": True, "replayed": True, "coordination_receipt": dict(historical)}
            if task.status != "blocked" or task.revision != int(failure["control_expected_revision"]) + 1 or dict(task.body.get("completion_receipt", {})) != failure:
                raise ValueError("operator retry requires the unchanged exact blocked failure CAS")
            if not apply:
                return {**report, "applied": False, "expected_revision": task.revision}
            changed = daemon.task_source.rearm_blocked_task(task.task_cid, receipt=compact)
            coordinated = daemon.coordinator.rearm_failed_task(failure_receipt=failure, control_task_observation=changed.to_dict())
            retained = daemon.get_attempt(attempt_id)
            if retained.status != "failed" or retained.attempt_number != attempt.attempt_number or daemon.phase_history(attempt_id) != phases:
                raise RuntimeError("interrupted provider retry changed immutable native history")
            output = {**report, "applied": True, "control_revision": changed.revision, "coordination_receipt": coordinated}
            atomic_json(config.state_root / "maintenance" / ("operator-interrupted-retry-" + task_alias + ".json"), output)
            return output
        finally:
            daemon.close()



def operator_rearm_inflight_candidate(config: CampaignConfig, *, task_alias: str, attempt_id: str,
        candidate_request_path: Path, acknowledge_paid_attempt: bool, apply: bool = False,
        fence_probe: Callable[[CampaignConfig], Mapping[str, Any]] = _registered_stopped_fence) -> dict[str, Any]:
    """Retry an exact misclassified paid inflight attempt without refunding it."""
    from .worldwide_legal_search_watchdog import read_owned_json
    from .worldwide_legal_search_continuation import candidate_manifest
    from ..todo_daemon.database_portal_bridge import DatabasePortalExecutionBridge
    from ..todo_daemon.implementation_daemon import DatabaseImplementationDaemon
    from ..merge.database_coordination import TASK_CLAIM_FAILURE_REARM_OPERATION
    if not acknowledge_paid_attempt:
        raise ValueError("candidate retry requires explicit preserved paid-attempt acknowledgment")
    with owner_lock(config):
        fence = dict(fence_probe(config))
        if fence.get("owner_alive") is not False or fence.get("main_pid") != 0 or fence.get("cgroup_empty") is not True:
            raise ValueError("candidate retry requires exact stopped owner and drained cgroup")
        if not candidate_request_path.is_absolute() or candidate_request_path.resolve(strict=True) != candidate_request_path or not candidate_request_path.is_relative_to(config.state_root):
            raise ValueError("candidate retry request must be an exact owned campaign path")
        request = read_owned_json(candidate_request_path)
        if request.get("schema") != "wwl.native-candidate-continuation@1" or request.get("receipt_id") != identity({key:value for key,value in request.items() if key != "receipt_id"}) or request.get("run_id") != config.run_id or request.get("task_id") != task_alias or request.get("predecessor_attempt_id") != attempt_id:
            raise ValueError("candidate retry request identity differs")
        binding_path = config.state_root / "native/wwl_database_portal_attempts" / hashlib.sha256(attempt_id.encode()).hexdigest()[:24] / "database-attempt-binding.json"
        binding = DatabasePortalExecutionBridge._strict_binding(binding_path)
        if request.get("predecessor_binding_id") != binding["binding_id"] or request.get("database_task_cid") != binding["task_cid"]:
            raise ValueError("candidate retry does not bind the exact failed task projection")
        repo = config.site_root if next(task for task in read_json(config.plan_path)["tasks"] if task["id"] == task_alias)["repository"] == "site" else config.framework_root
        workspace = Path(request["worktree_path"])
        if request.get("repository_root") != str(repo) or workspace.resolve(strict=True) != workspace or not workspace.is_relative_to(config.state_root / "worktrees") or candidate_manifest(workspace,request["baseline_commit"],request["candidate_commit"]) != request["changed_files"]:
            raise ValueError("candidate retry retained bytes or repository custody differ")
        daemon = DatabaseImplementationDaemon(database_path=config.database, coordination_path=config.state_root / "coordination.duckdb",
            execution_path=config.state_root / "execution.duckdb", state_dir=config.state_root / "native", state_prefix="wwl",
            owner_session_id=config.run_id, authority_mode="embedded_exclusive", require_real_execution=True, max_task_attempts=config.max_task_attempts, task_prefix="WWL-").open()
        try:
            if daemon.list_running_attempts():
                raise ValueError("candidate retry cannot replace a running native attempt")
            attempt = daemon.get_attempt(attempt_id)
            if (attempt is None or attempt.task_alias != task_alias or attempt.task_cid != binding["task_cid"] or attempt.status != "failed"
                    or attempt.attempt_number >= config.max_task_attempts or request.get("predecessor_attempt_number") != attempt.attempt_number
                    or request.get("next_attempt_number") != attempt.attempt_number + 1):
                raise ValueError("candidate retry predecessor or remaining native budget differs")
            phases = daemon.phase_history(attempt_id)
            if any(phase.get("phase") in {"effect","validation","completed"} for phase in phases):
                raise ValueError("candidate retry predecessor has accepted effect phases")
            failure = daemon._portal_failure_phase_receipt(attempt)
            if failure is None or not daemon._portal_failure_matches_recoverable_reason(failure,attempt,"inflight_process"):
                raise ValueError("candidate retry failed reason is not exact native inflight misclassification")
            task = daemon.task_source.get_task(task_alias)
            compact = {"operation":TASK_CLAIM_FAILURE_REARM_OPERATION,"settlement_id":failure["settlement_id"]}
            historical = daemon.coordinator.observe_rearmed_failed_task_claim(daemon.coordinator.get_task_claim(attempt.claim_id),failure_receipt=failure)
            report = {"schema":"wwl.operator-inflight-candidate-retry@1","run_id":config.run_id,"task_alias":task_alias,"attempt_id":attempt_id,
                "native_failure_settlement":failure["settlement_id"],"candidate_admission_receipt":request["receipt_id"],"attempt_number_preserved":attempt.attempt_number,
                "paid_provider_attempt_acknowledged":True,"possible_external_effects":"unverified","non_dispatch_claimed":False,"attempt_budget_reset":False,"completion_authority":False}
            if historical is not None and task.status == "retrying" and dict(task.body.get("completion_receipt",{})) == compact:
                return {**report,"applied":True,"replayed":True,"coordination_receipt":dict(historical)}
            if task.status != "blocked" or task.revision != int(failure["control_expected_revision"]) + 1 or dict(task.body.get("completion_receipt",{})) != failure:
                raise ValueError("candidate retry requires the exact unchanged native blocked failure CAS")
            if not apply:
                return {**report,"applied":False,"expected_revision":task.revision}
            changed = daemon.task_source.rearm_blocked_task(task.task_cid,receipt=compact)
            coordinated = daemon.coordinator.rearm_failed_task(failure_receipt=failure,control_task_observation=changed.to_dict())
            retained = daemon.get_attempt(attempt_id)
            if retained.status != "failed" or retained.attempt_number != attempt.attempt_number or daemon.phase_history(attempt_id) != phases:
                raise RuntimeError("candidate retry changed immutable native predecessor history")
            output = {**report,"applied":True,"control_revision":changed.revision,"coordination_receipt":coordinated}
            atomic_json(config.state_root / "maintenance" / ("operator-inflight-candidate-retry-" + task_alias + ".json"),output)
            return output
        finally:
            daemon.close()


def migrate(config: CampaignConfig, *, apply: bool = False, fence_probe: Callable[[CampaignConfig], Mapping[str, Any]] = _registered_stopped_fence) -> dict[str, Any]:
    from ..planning import plan_revision_contracts as pc
    from ..task_sources.database_task_source import DatabaseTaskSource
    from ..task_sources.intent_repository import task_projection_spec_cid
    from ..task_sources.plan_revision_store import PlanRevisionApplyRequest, PlanRevisionStore
    from ..todo_daemon.implementation_daemon import DatabaseImplementationDaemon

    with owner_lock(config):
        fence = dict(fence_probe(config))
        if fence.get("owner_alive") is not False or fence.get("main_pid") != 0 or fence.get("cgroup_empty") is not True:
            raise ValueError("migration has no closed owner fence")
        if "observed_at" in fence:
            fence["observed_at_decimal"] = repr(fence.pop("observed_at"))
        # Read the native attempt ledger through its public owner API. No
        # provider callbacks are installed and no execution pass is called.
        daemon = DatabaseImplementationDaemon(database_path=config.database, coordination_path=config.state_root / "coordination.duckdb",
            execution_path=config.state_root / "execution.duckdb", state_dir=config.state_root / "native", state_prefix="wwl",
            owner_session_id=config.run_id, authority_mode="embedded_exclusive", require_real_execution=True, task_prefix="WWL-").open()
        try:
            if daemon.list_running_attempts():
                raise ValueError("retained native execution must settle or quarantine before task specifications can change")
        finally:
            daemon.close()
        sidecars = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (config.state_root / "coordination.duckdb", config.state_root / "execution.duckdb") if path.is_file()}
        plan = read_json(config.plan_path)
        with DatabaseTaskSource(config.database, owner_id=config.run_id + ":validation-migration") as source:
            before = copy.deepcopy(dict(source.plan_projection()))
            population, aliases = candidate_population(before, plan)
            if not aliases:
                return {"schema": "wwl.validation-argv-migration@1", "changed": False, "projection_cid": before["projection_cid"], "completion_authority": False}
            with tempfile.TemporaryDirectory(prefix="wwl-argv-migration-candidate-") as directory:
                with DatabaseTaskSource(Path(directory) / "candidate.duckdb", owner_id="wwl-migration-candidate") as candidate:
                    candidate.materialize(population)
                    after = copy.deepcopy(dict(candidate.plan_projection()))
            current = {task["task_cid"]: task for task in before["tasks"]}
            replacement = {task["task_cid"]: task for task in after["tasks"]}
            items = []
            for cid, record in current.items():
                desired = replacement[cid]
                left, right = copy.deepcopy(record), copy.deepcopy(desired)
                left["validations"] = right["validations"] = []
                if task_projection_spec_cid(left) != task_projection_spec_cid(right):
                    raise ValueError("migration would change semantics beyond validation argv: " + record["task_alias"])
                if record["task_alias"] in aliases:
                    items.append(pc.PlanDeltaItem(item_key="validation-argv:" + record["task_alias"], operation=pc.PlanDeltaOperation.AMEND_UNSTARTED_TASK,
                        target_cid=cid, expected_target_lifecycle=source._lifecycle_for_status(record["status"]),
                        expected_target_spec_revision=task_projection_spec_cid(record), before_digest=task_projection_spec_cid(record), after_record_cid=task_projection_spec_cid(desired),
                        effect_class=pc.DeltaEffectClass.MATERIALIZABLE_NOW, rationale="Correct adapter serialization to the exact existing admitted argv; criteria, lifecycle, budget and history are retained.",
                        expected_effects=("correct_validation_argv_encoding",), affected_task_cids=(cid,)))
            report = {"schema": "wwl.validation-argv-migration-scan@1", "run_id": config.run_id, "source_projection": before["projection_cid"],
                "candidate_projection": after["projection_cid"], "amended_task_ids": aliases, "semantics_preserved": True, "owner_fence": fence,
                "sidecar_sha256": sidecars, "plan_definition_id": identity(plan), "execution_pass_called": False, "completion_authority": False}
            if not apply:
                return {**report, "applied": False}
            active = [item for item in before["plans"] if item["status"] == "active"]
            if len(active) != 1:
                raise ValueError("migration requires exactly one active native predecessor plan")
            store = PlanRevisionStore(config.state_root / "plan-revisions")
            prior = store.get_active()
            base = active[0]["plan_cid"]
            record_id = pc.plan_revision_cid(report)
            policy = read_json(config.plan_path.with_name("watchdog.policy.json"))
            roots = pc.PlanAuthorityRoots(repository_id="repository:" + config.run_id, repository_root_cid=identity(observed_forest(config)),
                dirty_worktree_root=population["repository_tree_id"], task_source_id=identity(str(config.database)), task_source_revision=before["projection_cid"],
                policy_root=identity(policy), intent_ir_root=identity(plan), legal_ir_root=identity({"legal_plan": identity(plan)}),
                security_ir_root=identity(plan["execution"]), program_root=identity({"migration_interface": "wwl.validation-argv-migration@1"}),
                capability_catalog_root=identity({"native": "PlanRevisionStore@1", "effect": "correct_validation_argv_encoding"}),
                provider_catalog_root=identity(plan.get("routing", {})), usage_policy_root=identity({"provider_calls": 0}), configuration_root=identity(config.serializable()))
            delta = pc.PlanDelta(base_plan_root=base, base_plan_revision=prior.semantic_revision if prior else 1, request_cid=record_id, roots=roots,
                items=tuple(items), expected_effects=("correct_validation_argv_encoding",), deferred_item_keys=(),
                claimed_population_digest=identity([]), accepted_population_digest=identity([task["task_cid"] for task in before["tasks"] if task["status"] in {"done", "complete", "completed"}]),
                scan_receipt_cid=record_id, evidence_bundle_cid=record_id, admission_receipt_cid=record_id)
            population_digest = lambda kind, ids=(): pc.PlanPopulationDigest(kind=kind, member_cids=tuple(ids))
            revision = pc.PlanRevision(plan_root_cid=pc.plan_revision_cid({"base": base, "delta": delta.delta_cid}), semantic_revision=(prior.semantic_revision + 1) if prior else 2,
                parent_plan_root=base, origin=pc.PlanOrigin.STEER, roots=roots, request_cid=record_id, delta_cid=delta.delta_cid,
                scan_receipt_cid=record_id, query_plan_cid=record_id, evidence_bundle_cid=record_id, admission_receipt_cid=record_id, execution_plan_cid=record_id,
                goal_population=population_digest(pc.PopulationKind.RETAINED, [goal["goal_cid"] for goal in before["goals"]]),
                task_population=population_digest(pc.PopulationKind.RETAINED, current), added_population=population_digest(pc.PopulationKind.ADDED),
                superseded_population=population_digest(pc.PopulationKind.SUPERSEDED), retained_population=population_digest(pc.PopulationKind.RETAINED, current),
                deferred_population=population_digest(pc.PopulationKind.DEFERRED), claimed_population=population_digest(pc.PopulationKind.CLAIMED),
                completed_population=population_digest(pc.PopulationKind.COMPLETED, [task["task_cid"] for task in before["tasks"] if task["status"] in {"done", "complete", "completed"}]),
                blocked_population=population_digest(pc.PopulationKind.BLOCKED, [task["task_cid"] for task in before["tasks"] if task["status"] == "blocked"]),
                resource_contract=pc.PlanResourceContract(), provider_contract=pc.PlanProviderContract(provider_requirement=identity(plan.get("routing", {})), endpoint_policy_class="existing-admitted-route"),
                lease_contract=pc.PlanLeaseContract(), retry_contract=pc.PlanRetryContract(max_retries=config.max_task_attempts - 1),
                worktree_contract=pc.PlanWorktreeContract(policy="existing-isolated-worktrees", isolation_required=True),
                merge_strategy=pc.PlanMergeStrategy(kind=pc.MergeStrategyKind.SERIAL), conflict_contract=pc.PlanConflictContract(),
                completion_rule=pc.PlanCompletionRule(authority=pc.CompletionAuthority.VALIDATION_GATE), event_cursor=before["projection_cid"])
            # Re-observe the actual source at the final owner boundary. The
            # immutable request then fences every amendment by exact spec CID.
            if source.plan_projection()["projection_cid"] != before["projection_cid"] or dict(fence_probe(config)).get("owner_alive") is not False:
                raise ValueError("migration authority drifted before native apply")
            if prior is None:
                # The original campaign predated the append-only revision
                # journal. Adopt its *observed* metadata without a projection
                # backend; CREATE must never rematerialize an existing DB.
                adoption = {"schema": "wwl.existing-plan-adoption@1", "plan_root_cid": base,
                    "native_projection_cid": before["projection_cid"], "native_population_json": json.dumps(before, sort_keys=True, ensure_ascii=False),
                    "bootstrap_receipt_json": (config.state_root / "bootstrap-receipt.json").read_text() if (config.state_root / "bootstrap-receipt.json").is_file() else None,
                    "owner_fence": fence, "projection_materialization_performed": False, "completion_authority": False}
                adoption_id = pc.plan_revision_cid(adoption)
                baseline = replace(revision, plan_root_cid=base, parent_plan_root="", origin=pc.PlanOrigin.CREATE,
                    semantic_revision=1, delta_cid="", request_cid=adoption_id, scan_receipt_cid=adoption_id,
                    evidence_bundle_cid=adoption_id, admission_receipt_cid=adoption_id, execution_plan_cid="")
                store.apply(PlanRevisionApplyRequest(revision=baseline, observed_roots=roots,
                    idempotency_key="wwl-existing-native-plan:" + before["projection_cid"],
                    expected_effects=("adopt_existing_native_plan_metadata",), records={"existing_native_plan_adoption": adoption}))
                prior = store.get_active()
            receipt = store.apply_authorized(PlanRevisionApplyRequest(revision=revision, observed_roots=roots,
                idempotency_key="wwl-validation-argv:" + record_id, expected_effects=delta.expected_effects, delta=delta,
                goal_graph=population, duckdb_source=source, repository_tree_id=population["repository_tree_id"],
                fencing_token=max(task["revision"] for task in before["tasks"]) + 1, lease_id="native-owner-lock:" + config.run_id,
                expected_active_plan_root=prior.plan_root_cid if prior else "", expected_active_revision_cid=prior.revision_cid if prior else "", records={"validation_argv_migration_scan": report}))
            final = dict(source.plan_projection())
            for task in final["tasks"]:
                original = current[task["task_cid"]]
                if task["status"] != original["status"] or task["body"] != original["body"] or task["dependencies"] != original["dependencies"]:
                    raise RuntimeError("native migration unexpectedly changed retained lifecycle/history")
                if [item["argv"] for item in task["validations"]] != next(item["validation_argv"] for item in plan["tasks"] if item["id"] == task["task_alias"]):
                    raise RuntimeError("native migration failed exact argv verification")
        if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest for path, digest in sidecars.items()):
            raise RuntimeError("migration changed execution or coordination sidecars")
        output = {**report, "applied": True, "native_apply_receipt": receipt.to_dict(), "final_projection": final["projection_cid"], "native_plan_root": revision.plan_root_cid,
            "attempt_history_preserved": True, "candidate_workspaces_untouched": True}
        atomic_json(config.state_root / "migrations/validation-argv.json", output)
        return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--rearm-protected")
    parser.add_argument("--rearm-interrupted-provider")
    parser.add_argument("--rearm-inflight-candidate")
    parser.add_argument("--candidate-request", type=Path)
    parser.add_argument("--watchdog-action-id")
    parser.add_argument("--attempt-id")
    parser.add_argument("--clearance", type=Path)
    parser.add_argument("--acknowledge-paid-attempt", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = config_from_args(runtime_parser().parse_args(["status", "--config", str(args.config)]))
        if sum(bool(value) for value in (args.rearm_protected,args.rearm_interrupted_provider,args.rearm_inflight_candidate)) > 1:
            raise ValueError("choose exactly one native operator retry operation")
        if args.rearm_inflight_candidate:
            if not args.attempt_id or not args.candidate_request:
                raise ValueError("inflight candidate retry requires --attempt-id and --candidate-request")
            output = operator_rearm_inflight_candidate(config, task_alias=args.rearm_inflight_candidate, attempt_id=args.attempt_id,
                candidate_request_path=args.candidate_request, acknowledge_paid_attempt=args.acknowledge_paid_attempt, apply=args.apply)
        elif args.rearm_protected:
            if not args.attempt_id or not args.clearance:
                raise ValueError("operator retry requires --attempt-id and --clearance")
            output = operator_rearm(config, task_alias=args.rearm_protected, attempt_id=args.attempt_id,
                clearance_path=args.clearance, acknowledge_paid_attempt=args.acknowledge_paid_attempt, apply=args.apply)
        elif args.rearm_interrupted_provider:
            if not args.attempt_id or not args.watchdog_action_id:
                raise ValueError("interrupted provider retry requires --attempt-id and --watchdog-action-id")
            output = operator_rearm_interrupted_provider(config, task_alias=args.rearm_interrupted_provider, attempt_id=args.attempt_id,
                watchdog_action_id=args.watchdog_action_id, acknowledge_paid_attempt=args.acknowledge_paid_attempt, apply=args.apply)
        else:
            output = migrate(config, apply=args.apply)
        print(json.dumps(output, sort_keys=True, default=str))
        return 0
    except Exception as error:
        print(json.dumps({"schema": "wwl.validation-argv-migration-error@1", "error_class": type(error).__name__, "reason": str(error), "completion_authority": False}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
