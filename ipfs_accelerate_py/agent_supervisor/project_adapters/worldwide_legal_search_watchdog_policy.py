"""Explicit stopped-owner admission of the seven reviewed repair-scope paths.

This is an operator migration, never an automatically repairable policy edit.
All task specifications, acceptance rules, paid attempts and watchdog budgets
remain intact. Native plan authority records an evidence-only CAS revision.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

from .worldwide_legal_search_runtime import (
    CampaignConfig, atomic_json, config_from_args, git, identity, observed_forest,
    owner_lock, parser as runtime_parser, read_json,
)
from .worldwide_legal_search_migration import _registered_stopped_fence, candidate_population
from .worldwide_legal_search_watchdog import publish_once, read_owned_json

ADDED_PATHS = (
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon_runner.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/dependency_preflight_retry.py",
    "ipfs_accelerate_py/agent_supervisor/validation/project_dependency_preflight.py",
    "ipfs_accelerate_py/agent_supervisor/validation/validation_commands.py",
    "ipfs_accelerate_py/agent_supervisor/validation/validation_runtime.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/fenced_candidate_retirement.py",
)


def validate_scope_transition(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    """Admit only this exact addition; limits and policy semantics stay exact."""
    expected = copy.deepcopy(dict(before))
    if before.get("schema") != "worldwide-legal-search.watchdog-policy@1":
        raise ValueError("unsupported predecessor policy")
    paths = expected.get("repair_scope", {}).get("allowed_paths")
    if not isinstance(paths, list) or len(set(paths)) != len(paths) or set(paths) & set(ADDED_PATHS):
        raise ValueError("predecessor must exclude all seven reviewed scope additions")
    paths.extend(ADDED_PATHS)
    if expected != after:
        raise ValueError("policy migration permits only the exact seven reviewed source paths")


def stopped_fences(config: CampaignConfig) -> dict[str, Any]:
    supervisor = _registered_stopped_fence(config)
    command = subprocess.run(["systemctl", "--user", "show", "worldwide-legal-search-watchdog.service",
        "--property=Id,MainPID,ActiveState,SubState,ControlGroup,FragmentPath"],
        capture_output=True, text=True, timeout=10, check=False)
    fields = dict(line.split("=", 1) for line in command.stdout.splitlines() if "=" in line)
    from .worldwide_legal_search_host_lifecycle import RegisteredSystemdLifecycle
    lifecycle = RegisteredSystemdLifecycle(repository=config.site_root, state_root=config.state_root, run_id=config.run_id)
    fragment = Path(fields.get("FragmentPath") or "/missing-unit")
    expected_fragment = Path.home() / ".config/systemd/user/worldwide-legal-search-watchdog.service"
    if (command.returncode or fields.get("Id") != "worldwide-legal-search-watchdog.service"
            or fields.get("MainPID") != "0" or fields.get("ActiveState") not in {"inactive", "failed"}
            or fragment != expected_fragment or not fragment.is_file() or fragment.is_symlink()
            or fragment.stat().st_uid != os.getuid()
            or str(config.site_root / "implementation_plan/worldwide_legal_search/watchdog_loop.py") not in fragment.read_text()
            or str(config.state_root / "watchdog/state.json") not in fragment.read_text()
            or not lifecycle._cgroup_empty(fields.get("ControlGroup", ""))):
        raise ValueError("registered watchdog and its entire cgroup must be stopped")
    return {"supervisor": supervisor, "watchdog": fields, "both_stopped": True}


@contextlib.contextmanager
def watchdog_lock(config: CampaignConfig):
    path = config.state_root / "watchdog/state.json.lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        if os.fstat(descriptor).st_uid != os.getuid():
            raise ValueError("watchdog lock is not owned")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(descriptor)


def _native_policy_revision(config: CampaignConfig, before: Mapping[str, Any], after: Mapping[str, Any],
                            admission: Mapping[str, Any], *, apply: bool) -> dict[str, Any]:
    """Use the existing Native store; perform zero task specification writes."""
    from ..task_sources.plan_revision_store import PlanRevisionStore, PlanRevisionApplyRequest
    from ..task_sources.database_task_source import DatabaseTaskSource
    from ..planning import plan_revision_contracts as pc

    # Do not create a journal for campaigns which never had this authority.
    active_path = config.state_root / "plan-revisions/active.json"
    if not active_path.exists():
        return {"required": False, "reason": "no_existing_native_plan_revision_authority"}
    store = PlanRevisionStore(active_path.parent)
    active = store.get_active()
    if active is None or active.quarantined:
        raise ValueError("native plan authority is absent or quarantined")
    prior = store.load_revision(active.revision_cid)
    record_id = pc.plan_revision_cid(dict(admission))
    if prior.roots.policy_root == identity(after):
        if prior.request_cid != record_id:
            raise ValueError("native policy changed under a different admission")
        return {"required": True, "applied": True, "replayed": True,
                "plan_root_cid": active.plan_root_cid, "revision_cid": active.revision_cid}
    if prior.roots.policy_root != identity(before):
        raise ValueError("native plan policy root differs from the exact predecessor")
    if not apply:
        return {"required": True, "applied": False, "prior_plan_root": active.plan_root_cid,
                "prior_revision_cid": active.revision_cid, "prior_policy_root": prior.roots.policy_root}
    with DatabaseTaskSource(config.database, owner_id=config.run_id + ":watchdog-policy-migration") as source:
        projection = copy.deepcopy(dict(source.plan_projection()))
        population, corrected = candidate_population(projection, read_json(config.plan_path))
        if corrected:
            raise ValueError("policy migration cannot change validation specifications")
        roots = replace(prior.roots, policy_root=identity(after), task_source_revision=projection["projection_cid"],
            repository_root_cid=identity(observed_forest(config)), configuration_root=identity(config.serializable()))
        item = pc.PlanDeltaItem(item_key="reviewed-watchdog-repair-scope", operation=pc.PlanDeltaOperation.ATTACH_EVIDENCE,
            target_cid=active.plan_root_cid, expected_target_lifecycle=pc.LifecycleState.ADMITTED,
            expected_target_spec_revision=active.revision_cid, before_digest=identity(before), after_record_cid=record_id,
            effect_class=pc.DeltaEffectClass.EVIDENCE_ONLY, rationale="Operator-admitted seven exact supervisor repair paths; all task specifications and history are retained.",
            expected_effects=("admit_reviewed_watchdog_repair_scope",))
        delta = pc.PlanDelta(base_plan_root=active.plan_root_cid, base_plan_revision=active.semantic_revision,
            request_cid=record_id, roots=roots, items=(item,), expected_effects=item.expected_effects,
            scan_receipt_cid=record_id, evidence_bundle_cid=record_id, admission_receipt_cid=record_id)
        current_tasks = projection["tasks"]
        claimed = set(prior.claimed_population.member_cids) | {t["task_cid"] for t in current_tasks if t["status"] in {"claimed", "in_progress", "running", "settling"}}
        completed = set(prior.completed_population.member_cids) | {t["task_cid"] for t in current_tasks if t["status"] in {"completed", "complete", "done"}}
        revision = replace(prior, plan_root_cid=pc.plan_revision_cid({"base": active.plan_root_cid, "delta": delta.delta_cid}),
            semantic_revision=active.semantic_revision + 1, parent_plan_root=active.plan_root_cid,
            origin=pc.PlanOrigin.STEER, roots=roots, request_cid=record_id, delta_cid=delta.delta_cid,
            scan_receipt_cid=record_id, evidence_bundle_cid=record_id, admission_receipt_cid=record_id,
            execution_plan_cid=record_id, event_cursor=projection["projection_cid"],
            claimed_population=pc.PlanPopulationDigest(kind=pc.PopulationKind.CLAIMED, member_cids=tuple(sorted(claimed))),
            completed_population=pc.PlanPopulationDigest(kind=pc.PopulationKind.COMPLETED, member_cids=tuple(sorted(completed))))
        receipt = store.apply_authorized(PlanRevisionApplyRequest(revision=revision, observed_roots=roots,
            idempotency_key="wwl-watchdog-policy:" + record_id, expected_effects=delta.expected_effects, delta=delta,
            goal_graph=population, duckdb_source=source, repository_tree_id=population["repository_tree_id"],
            fencing_token=max(t["revision"] for t in current_tasks) + 1, lease_id="native-owner-lock:" + config.run_id,
            base_event_cursor=active.event_cursor, expected_active_plan_root=active.plan_root_cid,
            expected_active_revision_cid=active.revision_cid, records={"watchdog_policy_admission": dict(admission)}))
        final = dict(source.plan_projection())
        for key in ("tasks", "goals", "goal_edges"):
            if final[key] != projection[key]:
                raise RuntimeError("native policy migration changed retained " + key)
    return {"required": True, "applied": True, "receipt": receipt.to_dict(),
            "plan_root_cid": revision.plan_root_cid, "revision_cid": revision.revision_cid,
            "task_specifications_unchanged": True, "task_lifecycle_and_history_unchanged": True}


def migrate(config: CampaignConfig, *, previous_site_revision: str, apply: bool = False,
            fence_probe=stopped_fences) -> dict[str, Any]:
    if not previous_site_revision or previous_site_revision.startswith("-"):
        raise ValueError("an explicit committed predecessor site revision is required")
    relative = config.plan_path.with_name("watchdog.policy.json").relative_to(config.site_root).as_posix()
    prior_commit = git(config.site_root, "rev-parse", "--verify", previous_site_revision + "^{commit}")
    head = git(config.site_root, "rev-parse", "HEAD")
    subprocess.run(["git", "-C", str(config.site_root), "merge-base", "--is-ancestor", prior_commit, head], check=True, timeout=10)
    before = json.loads(git(config.site_root, "show", prior_commit + ":" + relative))
    after = read_owned_json(config.site_root / relative)
    if after != json.loads(git(config.site_root, "show", "HEAD:" + relative)):
        raise ValueError("new repair policy must be committed and reviewable")
    validate_scope_transition(before, after)
    state_path = config.state_root / "watchdog/state.json"
    with owner_lock(config), watchdog_lock(config):
        fences = dict(fence_probe(config))
        if fences.get("both_stopped") is not True:
            raise ValueError("both registered services must be stopped")
        state = read_owned_json(state_path)
        old_digest, new_digest = identity(before).removeprefix("sha256:"), identity(after).removeprefix("sha256:")
        if state.get("run_id") != config.run_id or state.get("policy_digest") not in {old_digest, new_digest}:
            raise ValueError("watchdog state differs from predecessor run or policy")
        old_state = {**copy.deepcopy(state), "policy_digest": old_digest}
        new_state = {**copy.deepcopy(state), "policy_digest": new_digest}
        admission = {"schema": "wwl.watchdog-policy-admission@1", "run_id": config.run_id,
            "previous_site_revision": prior_commit, "accepted_site_revision": head,
            "old_policy": before, "new_policy": after, "old_policy_id": identity(before), "new_policy_id": identity(after),
            "added_paths": list(ADDED_PATHS), "old_state_id": identity(old_state), "new_state_id": identity(new_state),
            "task_acceptance_routing_and_budgets_unchanged": True, "completion_authority": False}
        admission_id = identity(admission)
        directory = config.state_root / "control/policy-migrations" / admission_id.removeprefix("sha256:")
        native_scan = _native_policy_revision(config, before, after, admission, apply=False)
        if not apply:
            return {"schema": "wwl.watchdog-policy-migration@1", "applied": False, "admission": admission,
                    "admission_id": admission_id, "owner_fences": fences, "native_policy_revision": native_scan}
        # Journal first. A crash may resume only this exact state and admission.
        publish_once(directory / "admission.json", {**admission, "receipt_id": admission_id})
        native = _native_policy_revision(config, before, after, admission, apply=True)
        native_path = directory / "native-policy-revision.json"
        if native_path.exists():
            prior_native = read_owned_json(native_path)
            if any(prior_native.get(k) != native.get(k) for k in ("required", "plan_root_cid", "revision_cid")):
                raise ValueError("durable native policy revision changed")
            native = prior_native
        else:
            publish_once(native_path, native)
        if dict(fence_probe(config)).get("both_stopped") is not True or read_owned_json(state_path) != state:
            raise ValueError("service or watchdog state changed before final policy CAS")
        if state != new_state:
            atomic_json(state_path, new_state)
        if read_owned_json(state_path) != new_state:
            raise RuntimeError("watchdog policy CAS verification failed")
        result = {"schema": "wwl.watchdog-policy-migration@1", "applied": True, "admission_id": admission_id,
            "owner_fences": fences, "native_policy_revision": native, "watchdog_state_id": identity(new_state),
            "watchdog_budgets_and_task_set_preserved": True, "completion_authority": False}
        atomic_json(directory / "result.json", result)
        return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--previous-site-revision", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = config_from_args(runtime_parser().parse_args(["status", "--config", str(args.config)]))
        print(json.dumps(migrate(config, previous_site_revision=args.previous_site_revision, apply=args.apply), sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(json.dumps({"schema": "wwl.watchdog-policy-migration-error@1", "error": str(error), "completion_authority": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
