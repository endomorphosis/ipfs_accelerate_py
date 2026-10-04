"""Derive bounded goal/task progress from an existing native world capture.

Counts describe only the observed task population. They never complete a goal,
evaluate its independent completion contract, or create another state store.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import json

from ..proof.formal_verification_contracts import content_identity
from .world_snapshot_contracts import parse_world_snapshot


SCHEMA = "supervisor-observed-intent-progress@1"
MAX_NODES = 512
MAX_DEPTH = 32
MAX_BYTES = 8_000_000


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _identity(value, field):
    if not isinstance(value, str) or not value or len(value.encode()) > 512:
        raise ValueError("invalid progress identity: " + field)
    return value


def project_intent_progress(capture: Mapping) -> dict:
    """Project exact parent links, selected tasks and current receipt identities.

    The capture's producer supplies coherent plan/completion relations from one
    transaction. This function verifies those content bindings; live freshness
    still belongs to the capture loader and its actual owner, not this view.
    Missing links or cycles produce an unavailable hierarchy with no counts.
    """
    material = _plain(capture)
    if len(json.dumps(material, sort_keys=True, separators=(",", ":")).encode()) > MAX_BYTES:
        raise ValueError("progress input exceeds its capture bound")
    claimed = material.pop("capture_cid", None)
    if material.get("schema") != "supervisor-intent-world-capture@1" or claimed != content_identity(material):
        raise ValueError("progress requires an intact native world capture")
    snapshot = parse_world_snapshot(material["snapshot"])
    plan, completion, context = (material[key] for key in (
        "plan_projection", "completion_projection", "planning_context"))
    for projection in (plan, completion):
        if projection["projection_cid"] != content_identity({
                key: value for key, value in projection.items() if key != "projection_cid"}):
            raise ValueError("progress projection identity differs")
    if (context["plan_projection_cid"] != plan["projection_cid"]
            or context["world_snapshot_cid"] != snapshot["snapshot_cid"]
            or context["event_watermark"] != completion["event_watermark"]
            or any(context[key] != plan[key] for key in ("goals", "goal_edges", "tasks"))
            or any(material.get(key) is not False or context.get(key) is not False
                   for key in ("execution_authority", "completion_authority"))):
        raise ValueError("progress capture relations or authority flags differ")
    population = material["component_payloads"][snapshot["components"]["task_population"]["cid"]]["material"]
    scope = population.get("scope")
    if scope not in {"selected-tasks", "all-intent-tasks"} or population.get("tasks") != plan["tasks"]:
        raise ValueError("progress task population scope differs")
    if max(len(plan["goals"]), len(plan["tasks"]), len(plan["goal_edges"])) > MAX_NODES:
        raise ValueError("progress relation count exceeds bound")
    goals = {row["goal_cid"]: row for row in plan["goals"]}
    tasks = {row["task_cid"]: row for row in plan["tasks"]}
    states = {row["task_cid"]: row for row in completion["task_states"]}
    if (len(goals) != len(plan["goals"]) or len(tasks) != len(plan["tasks"])
            or len(states) != len(completion["task_states"]) or set(states) != set(tasks)
            or (scope == "selected-tasks" and sorted(tasks) != population["task_cids"])
            or (scope == "all-intent-tasks" and population["task_cids"] != [])):
        raise ValueError("progress relations omit or duplicate task/goal identities")
    for cid, row in tasks.items():
        _identity(cid, "task_cid")
        if states[cid] != {key: row[key] for key in ("task_cid", "status", "revision")}:
            raise ValueError("progress task state differs from same-transaction completion state")
    receipts = {}
    for receipt in completion["completion_receipts"]:
        cid = receipt["task_cid"]
        if (cid not in tasks or cid in receipts
                or receipt["body"].get("revision") != tasks[cid]["revision"]
                or receipt["goal_cid"] != tasks[cid]["goal_cid"]):
            raise ValueError("progress receipt is not for the exact observed task revision")
        receipts[cid] = {key: receipt[key] for key in (
            "receipt_cid", "attempt_id", "claim_cid", "fencing_token",
            "validation_run_id", "evidence_digest",
        )}
    reasons, children, direct = set(), {cid: [] for cid in goals}, {cid: [] for cid in goals}
    parents = []
    for cid, goal in goals.items():
        _identity(cid, "goal_cid")
        parent = goal["parent_goal_cid"]
        if parent:
            parents.append({"parent_goal_cid": parent, "child_goal_cid": cid,
                            "source_field": "goals.parent_goal_cid"})
            if parent not in goals:
                reasons.add("missing_parent_goal")
            else:
                children[parent].append(cid)
    for cid, task in tasks.items():
        if task["goal_cid"] not in goals:
            reasons.add("missing_task_goal")
        else:
            direct[task["goal_cid"]].append(cid)
    explicit_edges = [{key: edge[key] for key in ("parent_goal_cid", "child_goal_cid", "edge_kind")}
                      for edge in plan["goal_edges"]]
    if any(edge[key] not in goals for edge in explicit_edges for key in ("parent_goal_cid", "child_goal_cid")):
        reasons.add("missing_declared_edge_goal")

    visited, visiting, descendants = set(), set(), {}
    def collect(cid, depth=0):
        if cid in visiting:
            reasons.add("cyclic_goal_hierarchy")
            return set()
        if depth > MAX_DEPTH:
            reasons.add("goal_hierarchy_depth_exceeds_bound")
            return set()
        if cid in visited:
            return descendants[cid]
        visiting.add(cid)
        selected = set(direct[cid])
        for child in sorted(children[cid]):
            selected.update(collect(child, depth + 1))
        visiting.remove(cid)
        visited.add(cid)
        descendants[cid] = selected
        return selected
    for cid in sorted(goals):
        collect(cid)

    projected_tasks = [{"task_cid": cid, "task_alias": task["task_alias"],
        "goal_cid": task["goal_cid"], "status": task["status"], "revision": task["revision"],
        "current_completion_receipt": receipts.get(cid),
        "dependencies": [{key: dependency[key] for key in ("dependency_task_cid", "kind")}
                         for dependency in task["dependencies"]],
        "completion_authority": False} for cid, task in sorted(tasks.items())]
    projected_goals = []
    for cid, goal in sorted(goals.items()):
        selected = descendants.get(cid, set())
        counts = None if reasons else {
            "observed_descendant_tasks": len(selected),
            "observed_status_counts": dict(sorted(Counter(tasks[item]["status"] for item in selected).items())),
            "current_completion_receipts": sum(item in receipts for item in selected),
            "completed_with_current_receipt": sum(tasks[item]["status"] == "completed" and item in receipts for item in selected),
        }
        title = str(goal["title"])
        projected_goals.append({"goal_cid": cid, "goal_alias": goal["goal_alias"],
            "parent_goal_cid": goal["parent_goal_cid"], "title": title[:512], "title_truncated": len(title) > 512,
            "status": goal["status"], "revision": goal["revision"],
            "child_goal_cids": sorted(children[cid]), "direct_task_cids": sorted(direct[cid]),
            "observed_counts": counts, "goal_completion_authority": "unresolved",
            "goal_contracts_evaluated": False, "completion_authority": False})
    value = {"schema": SCHEMA, "status": "unavailable" if reasons else "observed",
        "unavailable_reasons": sorted(reasons), "capture_cid": claimed,
        "world_snapshot_cid": snapshot["snapshot_cid"], "plan_projection_cid": plan["projection_cid"],
        "completion_projection_cid": completion["projection_cid"], "event_watermark": context["event_watermark"],
        "population_scope": scope, "task_population_complete": scope == "all-intent-tasks",
        "roots": sorted(cid for cid, row in goals.items() if not row["parent_goal_cid"]),
        "parent_edges": sorted(parents, key=lambda row: (row["parent_goal_cid"], row["child_goal_cid"])),
        "declared_goal_edges": explicit_edges, "goals": projected_goals, "tasks": projected_tasks,
        "goal_completion_authority": "unresolved", "goal_contracts_evaluated": False,
        "live_freshness_verified_here": False, "execution_authority": False,
        "completion_authority": False, "canonical_state_mutated": False}
    return {**value, "progress_cid": content_identity(value)}
