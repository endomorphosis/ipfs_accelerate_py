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
from typing import Any

_DONE = {"completed", "complete", "done"}
_RUNNABLE = {"pending", "todo", "ready", "blocked", "failed", "stuck"}
_ACTIVE = {"running", "in_progress", "in-progress"}


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


def observe_planned_work(*, policy: Any, repo_root: Path, tasks: dict[str, dict[str, Any]],
                         lanes: list[dict[str, Any]], state: dict[str, Any], now: float,
                         held: bool = False) -> dict[str, Any]:
    """Persist a stable grace clock and return evidence, without task writes."""
    if policy is None or isinstance(policy, dict) and policy.get("enabled", False) is False:
        state.pop("planned_work_observation", None)
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
    planned = sorted(task_id for task_id, task in tasks.items() if task.get("status") == "planned")
    frontier = [task_id for task_id in planned
                if all(tasks.get(dep, {}).get("status") in _DONE for dep in tasks[task_id].get("depends", []))]
    descriptions = []
    for task_id in planned:
        goal = goals.get(task_id, {})
        if not goal:
            errors.append(f"planned task missing from goals: {task_id}")
        elif (not isinstance(goal.get("depends_on", []), list)
              or not all(isinstance(dep, str) for dep in goal.get("depends_on", []))
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
            if task_id in planned and now < parsed.timestamp():
                deferred[task_id] = {**declaration, "state": "waiting_external_gate"}
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"{task_id}: {exc}")
    materialize = sorted(set(frontier) - set(deferred))
    runnable = sorted(task_id for task_id, task in tasks.items() if task.get("status") in _RUNNABLE
                      and all(tasks.get(dep, {}).get("status") in _DONE for dep in task.get("depends", [])))
    active = sorted(task_id for task_id, task in tasks.items() if task.get("status") in _ACTIVE)
    active.extend(str(lane["active_task_id"]) for lane in lanes if lane.get("active_task_id"))
    contract = _digest({"policy": policy, "tasks": tasks, "input_hashes": input_hashes,
                        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    result = {"enabled": True, "state": "dependency_wait", "planned_task_ids": planned,
              "dependency_ready_planned_task_ids": frontier, "materialization_task_ids": materialize,
              "deferred_tasks": deferred, "work_packages": descriptions, "configuration_errors": errors,
              "runnable_task_ids": runnable, "active_task_ids": sorted(set(active)),
              "input_hashes": input_hashes, "contract_fingerprint": contract,
              "grace_seconds": grace, "incidents": [], "completion_authority": False}
    if held or runnable or active or not (materialize or errors):
        state.pop("planned_work_observation", None)
        result["state"] = ("held" if held else "execution_available" if runnable or active
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
