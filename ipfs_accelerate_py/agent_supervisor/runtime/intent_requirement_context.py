"""Task-specific views of checked candidate requirements for a coding worker.

The caller replays signed admission and exact source first. This view preserves
applicable atoms, source selectors, dependencies and global prohibitions. It
does not infer source semantics or confer policy or execution authority.
"""
from __future__ import annotations

import json

from ..core.multiformats_identity import cid_for_dag_json
from ..planning.intent_requirement_adapter import _native_statement
from ..proof.formal_verification_contracts import content_identity

SCHEMA = "supervisor-task-intent-requirements@1"
MAX_CONTEXT_BYTES = 131_072
MAX_REQUIREMENTS = 512
_AUTHORITY = {"source_semantics_verified": False, "semantic_alignment_verified": False,
    "proof_authority": False, "execution_authority": False, "completion_authority": False,
    "publication_authority": False, "scope_expansion_authority": False}


def build_task_intent_context(*, contract, graph, coverage, task_cid):
    """Project only independently checked bindings for one exact graph task."""
    tasks = [task for task in graph.tasks if task.task_cid == task_cid]
    if len(tasks) != 1 or coverage.get("accepted") is not True:
        raise ValueError("requirement context needs an exact covered graph task")
    task = tasks[0]
    if (coverage["contract_cid"] != cid_for_dag_json(contract)
            or coverage["graph_cid"] != graph.plan_root_cid):
        raise ValueError("requirement context source/graph coverage identity differs")
    ledger = contract["ledger"]
    requirements = {row["requirement_id"]: row for row in ledger["requirements"]}
    if len(requirements) > MAX_REQUIREMENTS:
        raise ValueError("worker requirement population exceeds its bound")
    groundings = {row["requirement_id"]: row for row in contract["requirements"]}
    bindings = {row["requirement_id"]: row for row in coverage["bindings"]}
    units = {row["unit_id"]: row for row in ledger["source_units"]}
    selected = []
    for key, binding in bindings.items():
        if task.task_key in binding["task_keys"]:
            if task_cid not in binding["task_cids"]:
                raise ValueError("task key and content identity disagree in requirement coverage")
            selected.append(key)
    if not selected:
        raise ValueError("admitted task has no bound source requirement")
    dependencies, pending = set(), list(selected)
    while pending:
        key = pending.pop()
        for dependency in groundings[key]["dependency_requirement_ids"]:
            if dependency not in dependencies and dependency not in selected:
                dependencies.add(dependency)
                pending.append(dependency)

    def record(key):
        requirement, grounding = requirements[key], groundings[key]
        unit = units[requirement["source_unit_id"]]
        return {"requirement_id": key, "kind": requirement["kind"],
            "modality": requirement["modality"], "representation": requirement["representation"],
            "semantic_support": requirement["semantic_support"],
            "interpretation_status": requirement["interpretation_status"],
            "source_unit": {name: unit[name] for name in (
                "unit_id", "start_char", "end_char", "start_byte", "end_byte", "text", "sha256")},
            "native_atom": _native_statement(ledger, requirement),
            "outputs": grounding["outputs"], "validation_keys": grounding["validation_keys"],
            "dependency_requirement_ids": grounding["dependency_requirement_ids"],
            "binding": bindings.get(key), **_AUTHORITY}

    prohibited = sorted(key for key, row in requirements.items() if row["modality"] == "prohibited")
    operations = []
    if "symbolic_operations" in contract:
        operations = [row for row in contract["symbolic_operations"]["operations"]
                      if row["task_key"] == task.task_key]
    value = {"schema": SCHEMA, "task_cid": task_cid, "task_id": task.task_key,
        "contract_cid": coverage["contract_cid"], "ledger_sha256": ledger["ledger_sha256"],
        "graph_cid": graph.content_id, "coverage_cid": content_identity(coverage),
        "source_path": contract["source_path"], "source_sha256": ledger["source"]["sha256"],
        "requirement_ids": sorted(selected), "requirements": [record(key) for key in sorted(selected)],
        "dependency_requirements": [record(key) for key in sorted(dependencies)],
        "global_prohibitions": [record(key) for key in prohibited], "operations": operations,
        "task_contract": {"dependency_task_cids": list(task.dependency_task_cids),
            "scope_paths": list(task.scope_paths), "outputs": [row.to_dict() for row in task.outputs],
            "validations": [row.to_dict() for row in task.validations],
            "acceptance": [row.to_dict() for row in task.acceptance]},
        "official_reward": None, "native_persistence_verified_here": False, **_AUTHORITY}
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    if len(raw.encode()) > MAX_CONTEXT_BYTES:
        raise ValueError("worker requirement context exceeds its byte bound")
    return {**value, "requirements_context_cid": content_identity(value)}


__all__ = ["SCHEMA", "build_task_intent_context"]
