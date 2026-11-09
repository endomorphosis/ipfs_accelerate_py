"""Observe opted-in planned work that needs a real producer, never activate it.

This module emits maintenance evidence only. The existing repair dispatcher
owns worker limits, locks, holds and retry budgets; ordinary task validation
remains the only completion authority.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

_DONE = {"completed", "complete", "done"}
_RUNNABLE = {"pending", "todo", "ready", "blocked", "failed", "stuck"}
_ACTIVE = {"running", "in_progress", "in-progress"}
_RETRY_WAIT = {"attempt_retry_wait", "semantic_failure_wait", "gate_retry_wait", "gate_withheld", "resource_wait"}
_EXECUTION = {"deterministic_command", "validation", "gate_dependency_revalidation",
              "gate_revalidation", "gate_evidence_revalidation"}
# A gate recheck is the same withheld frontier. It must not drop the incident.
_FRONTIER_WAIT = _RETRY_WAIT | {"gate_dependency_revalidation", "gate_recovery_needed"}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _local(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError("expected a repository-relative planned-work path")
    path = (root / value).resolve()
    path.relative_to(root.resolve())
    if not path.is_file():
        raise ValueError(f"planned-work evidence is missing: {value}")
    return path


def _graph_errors(records: dict[str, dict[str, Any]], key: str, label: str) -> list[str]:
    """Validate references and detect cycles without a recursive depth limit."""
    errors = []
    dependencies: dict[str, set[str]] = {}
    for task_id, record in records.items():
        declared = record.get(key, [])
        if not isinstance(declared, list) or not all(isinstance(dep, str) and dep for dep in declared):
            errors.append(f"{label} dependencies must be a list of task IDs: {task_id}")
            declared = []
        if len(set(declared)) != len(declared):
            errors.append(f"{label} duplicate dependencies: {task_id}")
        unknown = sorted(set(declared) - records.keys())
        if unknown:
            errors.append(f"{label} unknown dependencies: {task_id} -> {', '.join(unknown)}")
        dependencies[task_id] = set(declared) & records.keys()
    dependents: dict[str, list[str]] = {task_id: [] for task_id in records}
    counts = {task_id: len(deps) for task_id, deps in dependencies.items()}
    for task_id, deps in dependencies.items():
        for dependency in deps:
            dependents[dependency].append(task_id)
    ready = [task_id for task_id, count in counts.items() if count == 0]
    while ready:
        for dependent in dependents[ready.pop()]:
            counts[dependent] -= 1
            if counts[dependent] == 0:
                ready.append(dependent)
    if any(counts.values()):
        errors.append(f"{label} dependency cycle blocks: " + ", ".join(sorted(k for k, n in counts.items() if n)))
    return errors


def _dependencies_done(task: dict[str, Any], tasks: dict[str, dict[str, Any]]) -> bool:
    dependencies = task.get("depends", [])
    return (isinstance(dependencies, list) and all(isinstance(dep, str) for dep in dependencies)
            and all(tasks.get(dep, {}).get("status") in _DONE for dep in dependencies))


def collect_gate_blockers(*, repo_root: Path, manifest: dict[str, Any],
                          lanes: list[dict[str, Any]], ready: list[str], now: float) -> dict[str, Any]:
    """Accept only native, source-current refusals, independently of poll phase."""
    from .gate_recovery import gate_evidence, gate_source_contract
    from .lane_resume import _command_contract, parse_deterministic_command
    from .supervisor_watchdog import parse_todo_task_fields
    proofs, reviews = [], []
    try:
        root = repo_root.resolve()
        todo = Path(manifest["todo_path"])
        todo = todo if todo.is_absolute() else root / todo
        fields = dict(parse_todo_task_fields(todo.read_text(),
                      task_header_prefix=manifest.get("task_header_prefix", "## TASK-")))
    except (OSError, ValueError, KeyError, TypeError):
        return {"proofs": proofs, "reviews": reviews}
    for lane in lanes:
        if not lane.get("process", {}).get("alive") or not lane["process"].get("identity_verified"):
            continue
        try:
            status = Path(lane["status_path"]).resolve()
            status.relative_to(root)
            attempt_path = status.with_name("lane_resume_attempts.json")
            gate_path = status.with_name("lane_resume_attempts.gates.json")
            ledger, gates = json.loads(attempt_path.read_text()), json.loads(gate_path.read_text())
            if not isinstance(ledger, dict):
                continue
            fixed = ledger.get("_validation_fixed_points", {})
            if not isinstance(gates, dict) or not isinstance(fixed, dict):
                continue
        except (OSError, ValueError, KeyError, TypeError):
            continue
        for task_id in ready:
            record = fixed.get(task_id, {})
            progress = record.get("execution_progress", {}) if isinstance(record, dict) else {}
            gate = gates.get(task_id, {})
            if (not isinstance(progress, dict) or not isinstance(gate, dict)
                    or progress.get("schema") != "agent-supervisor/execution-progress@1"
                    or progress.get("state") != "gate_withheld"
                    or type(progress.get("last_exit_code")) is not int or progress["last_exit_code"] != 75
                    or progress.get("last_stage") not in {"command", "validation"}
                    or type(progress.get("gate_refusals")) is not int or progress["gate_refusals"] < 1
                    or type(progress.get("finished_at_ms")) is not int
                    or not 0 < progress["finished_at_ms"] / 1000 <= now
                    or type(gate.get("unchanged_probes")) is not int or gate["unchanged_probes"] < 1):
                continue
            try:
                task = fields.get(task_id, {})
                command = parse_deterministic_command(task.get("deterministic_command") or "", repo_root=root)
                validation = parse_deterministic_command(task.get("validation") or "", repo_root=root)
                if not command or (task.get("validation") and not validation):
                    continue
                contract = _command_contract(task, command, validation, repo_root=root)
                fingerprint, evidence = gate_evidence(task_id, fields, repo_root=root, todo_path=todo,
                    log_path=status.with_name("supervisor.log"))
                source = gate_source_contract(evidence, contract)
                if (progress.get("contract") != contract or gate.get("fingerprint") != fingerprint
                        or gate.get("source_contract") != source):
                    reviews.append({"lane": lane["lane"], "task_id": task_id,
                                    "reason": "gate_refusal_evidence_not_current", "current_proof_accepted": False})
                    continue
                proofs.append({"lane": lane["lane"], "task_id": task_id, "contract": contract,
                    "gate_fingerprint": fingerprint, "gate_source_contract": source,
                    "withheld_stage": progress["last_stage"], "last_exit_code": 75,
                    "validation_exit_code": gate.get("last_validation_exit_code"),
                    "attempt_path": str(attempt_path), "gate_path": str(gate_path)})
            except (OSError, ValueError, KeyError, TypeError):
                reviews.append({"lane": lane["lane"], "task_id": task_id,
                                "reason": "gate_refusal_evidence_unreadable", "current_proof_accepted": False})
    return {"proofs": proofs, "reviews": reviews}


def _blocked_frontier(*, result: dict[str, Any], tasks: dict[str, dict[str, Any]],
                      lanes: list[dict[str, Any]], proofs: list[dict[str, Any]],
                      state: dict[str, Any], now: float, held: bool) -> bool:
    """Keep a semantic refusal clock while probes rotate; never mark tasks done."""
    key = "blocked_frontier_observation"
    if held or result["configuration_errors"]:
        state.pop(key, None)
        return False
    ready = sorted(task_id for task_id, task in tasks.items()
                   if task.get("status") in _RUNNABLE and _dependencies_done(task, tasks))
    needed = sorted(set(ready) - result["deferred_tasks"].keys())
    by_id = {p["task_id"]: p for p in proofs if isinstance(p, dict) and p.get("task_id") in needed
             and all(isinstance(p.get(k), str) and re.fullmatch(r"[0-9a-f]{64}", p[k])
                     for k in ("contract", "gate_fingerprint", "gate_source_contract"))}
    prior = state.get(key, {})
    base = result["contract_fingerprint"]
    real_execution = any(lane.get("phase") in _EXECUTION and lane.get("phase") != "gate_dependency_revalidation"
                         for lane in lanes)
    all_blocked = bool(needed) and set(needed) <= by_id.keys()
    if not all_blocked:
        # An actual retry gets its original bounded opportunity, without a new
        # grace period merely because its phase temporarily changed to running.
        if (real_execution and isinstance(prior, dict) and prior.get("contract") == base
                and type(prior.get("first_seen")) in (int, float)
                and math.isfinite(prior["first_seen"]) and 0 <= prior["first_seen"] <= now):
            result.update(state="blocked_frontier_deferred", maintenance_deferred=True,
                          blocked_task_ids=prior.get("blocked_task_ids", []),
                          first_seen=prior["first_seen"],
                          replan_required=now - prior["first_seen"] >= result["grace_seconds"])
            return True
        # A stopped lane contributes no proofs. Dropping the clock here makes
        # the next lease look like a new frontier and starts its grace over.
        if (not proofs and not real_execution and isinstance(prior, dict)
                and isinstance(prior.get("signature"), str) and len(prior["signature"]) == 64
                and type(prior.get("first_seen")) in (int, float)
                and not isinstance(prior.get("first_seen"), bool)
                and math.isfinite(prior["first_seen"]) and 0 <= prior["first_seen"] <= now):
            return False
        state.pop(key, None)
        return False
    signature = _digest([{name: by_id[task][name] for name in
                         ("task_id", "contract", "gate_fingerprint", "gate_source_contract")}
                         for task in needed])
    same = isinstance(prior, dict) and prior.get("contract") == base and prior.get("signature") == signature
    first = prior.get("first_seen", now) if same else now
    if type(first) not in (int, float) or not math.isfinite(first) or first > now:
        first = now
    state[key] = {"contract": base, "signature": signature, "first_seen": first,
                  "last_seen": now, "blocked_task_ids": needed}
    accepted_wait = set(needed) | result["deferred_tasks"].keys()
    unsafe = any(lane.get("active_task_id") and not (
                 lane.get("active_task_id") in accepted_wait and
                 lane.get("phase") in _FRONTIER_WAIT) for lane in lanes)
    due = now - first >= result["grace_seconds"]
    result.update(state="blocked_frontier_deferred" if unsafe else "plan_replan_required" if due else "blocked_frontier_grace",
                  blocked_task_ids=needed, blocked_frontier_proofs=[by_id[t] for t in needed],
                  first_seen=first, replan_required=due, maintenance_deferred=unsafe,
                  blockage_fingerprint=signature)
    if due and not unsafe:
        result["incidents"] = [{"kind": "planned_work_blocked_frontier", "evidence": {
            name: value for name, value in result.items() if name != "incidents"}}]
    return True


def observe_planned_work(*, policy: Any, repo_root: Path, tasks: dict[str, dict[str, Any]],
                         lanes: list[dict[str, Any]], state: dict[str, Any], now: float,
                         held: bool = False, gate_blockers: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Persist a stable grace clock and return evidence, without task writes."""
    if policy is None or isinstance(policy, dict) and policy.get("enabled", False) is False:
        state.pop("planned_work_observation", None)
        state.pop("blocked_frontier_observation", None)
        return {"enabled": False, "state": "disabled", "incidents": []}
    errors: list[str] = []
    if not isinstance(policy, dict) or policy.get("enabled") is not True:
        # An invalid explicit policy is maintenance work, never implicit opt-in.
        message = "planned_work_policy.enabled must be boolean"
        return {"enabled": False, "state": "configuration_error",
                "configuration_errors": [message],
                "incidents": [{"kind": "planned_work_policy_invalid", "evidence": {"reason": message}}]}
    grace = policy.get("grace_seconds", 60)
    if type(grace) is not int or not 1 <= grace <= 3600:
        errors.append("planned_work_policy.grace_seconds must be an integer in 1..3600")
        grace = 60
    for flag in ("continue_during_retry_wait", "materialize_independent_frontier"):
        if type(policy.get(flag, False)) is not bool:
            errors.append(f"planned_work_policy.{flag} must be boolean")
    independent = policy.get("materialize_independent_frontier", False) is True
    continue_wait = policy.get("continue_during_retry_wait", False) is True or independent
    goals: dict[str, dict[str, Any]] = {}
    input_hashes: dict[str, str] = {}
    try:
        path = _local(repo_root, policy.get("goals_path"))
        raw = path.read_bytes()
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("planned-work goals exceed 8 MiB")
        document = json.loads(raw)
        records = document.get("tasks") if isinstance(document, dict) else None
        if not isinstance(records, list) or not all(isinstance(t, dict) and isinstance(t.get("id"), str) for t in records):
            raise ValueError("planned-work goals require task objects with IDs")
        goals = {task["id"]: task for task in records}
        if len(goals) != len(records):
            raise ValueError("duplicate planned-work goal task IDs")
        input_hashes[policy["goals_path"]] = hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(str(exc))
    errors.extend(_graph_errors(tasks, "depends", "board"))
    errors.extend(_graph_errors(goals, "depends_on", "goal"))
    planned = sorted(task_id for task_id, task in tasks.items() if task.get("status") == "planned")
    frontier = [task_id for task_id in planned if _dependencies_done(tasks[task_id], tasks)]
    descriptions = []
    for task_id in planned:
        goal = goals.get(task_id, {})
        if not goal:
            errors.append(f"planned task missing from goals: {task_id}")
        elif (not isinstance(goal.get("depends_on", []), list)
              or not all(isinstance(dep, str) for dep in goal.get("depends_on", []))
              or not isinstance(tasks[task_id].get("depends", []), list)
              or not all(isinstance(dep, str) for dep in tasks[task_id].get("depends", []))
              or sorted(goal.get("depends_on", [])) != sorted(tasks[task_id].get("depends", []))):
            errors.append(f"planned task dependency mismatch: {task_id}")
        work = goal.get("work_package") if isinstance(goal.get("work_package"), dict) else {}
        descriptions.append({"task_id": task_id, "goal_id": goal.get("goal_id", ""),
                             "activation_requires": work.get("activation_requires", []),
                             "work_package": work})
    deferred: dict[str, dict[str, Any]] = {}
    declarations = policy.get("deferred_tasks", {})
    if not isinstance(declarations, dict):
        errors.append("planned_work_policy.deferred_tasks must be an object")
        declarations = {}
    for task_id, declaration in declarations.items():
        try:
            if task_id not in tasks or task_id not in goals:
                raise ValueError("unknown deferred task")
            if (not isinstance(declaration, dict) or declaration.get("kind") != "not_before"
                    or not isinstance(declaration.get("reason"), str) or not declaration["reason"].strip()):
                raise ValueError("deferred task requires not_before kind and explicit reason")
            stamp = declaration.get("not_before_utc")
            if not isinstance(stamp, str):
                raise ValueError("deferred task requires not_before_utc")
            parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
                raise ValueError("not_before_utc must be timezone-aware UTC")
            evidence = _local(repo_root, declaration.get("evidence_path"))
            if evidence.stat().st_size > 8 * 1024 * 1024:
                raise ValueError("deferred evidence exceeds 8 MiB")
            input_hashes[declaration["evidence_path"]] = hashlib.sha256(evidence.read_bytes()).hexdigest()
            if tasks[task_id].get("status") not in _DONE and now < parsed.timestamp():
                deferred[task_id] = {**declaration, "state": "waiting_external_gate"}
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"{task_id}: {exc}")
    materialize = sorted(set(frontier) - set(deferred))
    waiting_lanes = [lane for lane in lanes if continue_wait and lane.get("phase") in _RETRY_WAIT
                     and lane.get("active_task_id") in tasks]
    other_active = {str(lane["active_task_id"]) for lane in lanes if lane.get("active_task_id")
                    and lane not in waiting_lanes}
    waiting = {str(lane["active_task_id"]) for lane in waiting_lanes} - other_active
    waiting_evidence = [{"lane": lane.get("lane", ""), "task_id": lane["active_task_id"],
                         "phase": lane["phase"]} for lane in waiting_lanes
                        if lane["active_task_id"] in waiting]
    executing = [lane.get("lane", "") for lane in lanes if lane.get("phase") in _EXECUTION]
    runnable = sorted(task_id for task_id, task in tasks.items() if task.get("status") in _RUNNABLE
                      and task_id not in waiting
                      and task_id not in deferred
                      and _dependencies_done(task, tasks))
    active = {task_id for task_id, task in tasks.items() if task.get("status") in _ACTIVE and task_id not in waiting}
    active.update(other_active)
    contract = _digest({"policy": policy, "tasks": tasks, "input_hashes": input_hashes,
                        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    result = {"enabled": True, "state": "dependency_wait", "planned_task_ids": planned,
              "dependency_ready_planned_task_ids": frontier, "materialization_task_ids": materialize,
              "deferred_tasks": deferred, "work_packages": descriptions, "configuration_errors": errors,
              "runnable_task_ids": runnable, "active_task_ids": sorted(set(active)),
              "waiting_task_ids": sorted(waiting), "waiting_lanes": waiting_evidence,
              "executing_lanes": executing,
              "input_hashes": input_hashes, "contract_fingerprint": contract,
              "grace_seconds": grace, "incidents": [], "completion_authority": False}
    if _blocked_frontier(result=result, tasks=tasks, lanes=lanes, proofs=gate_blockers or [],
                         state=state, now=now, held=held):
        state.pop("planned_work_observation", None)
        return result
    # Both explicit flags permit continuation during native retry waits. The
    # independent-frontier flag also permits queued work, never an active command.
    if held or active or executing or (runnable and not independent) or not (materialize or errors):
        state.pop("planned_work_observation", None)
        result["state"] = ("held" if held else "execution_available" if runnable or active or executing
                           else "waiting_external_gate" if deferred else "dependency_wait" if planned else "complete")
        return result
    previous = state.get("planned_work_observation", {})
    first = previous.get("first_seen", now) if isinstance(previous, dict) and previous.get("contract") == contract else now
    if not isinstance(first, (float, int)) or isinstance(first, bool) or not math.isfinite(first) or first > now:
        first = now
    state["planned_work_observation"] = {"contract": contract, "first_seen": first, "last_seen": now}
    result.update(state="materialization_grace", first_seen=first)
    if now - first >= grace:
        result["state"] = "planned_work_unmaterialized"
        result["incidents"] = [{"kind": "planned_work_unmaterialized", "evidence": {
            key: value for key, value in result.items() if key != "incidents"}}]
    return result
