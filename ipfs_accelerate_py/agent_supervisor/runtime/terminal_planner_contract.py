"""Typed decoding constraints projected from independently signed task inputs.

This is a generation hint, never an admission receipt. The owner verifies the
original signed manifest again after parsing the provider's unmodified output.
"""
from __future__ import annotations

import hashlib
import json

FIELD = "terminal_planner_task_contract"
SCHEMA = "terminal-planner-task-contract@1"
MAX_BYTES = 32_768
TASK_FIELDS = frozenset({"task_key", "scope_paths", "outputs", "validations",
                         "acceptance", "dependency_task_keys"})


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def validate_task_contract(value):
    from ipfs_accelerate_py.cli_runtime.grok_structured_output import _bounded_json
    value = json.loads(_bounded_json(value, MAX_BYTES))
    fields = {"schema", "request_cid", "scan_cid", "program_root", "policy_root", "manifest_sha256",
              "evidence_cids", "tasks", "contract_sha256", "execution_authority", "completion_authority"}
    if (not isinstance(value, dict) or set(value) != fields or value["schema"] != SCHEMA
            or value["execution_authority"] is not False or value["completion_authority"] is not False):
        raise ValueError("exact non-authoritative terminal task contract required")
    for key in ("request_cid", "scan_cid", "program_root", "policy_root"):
        if type(value[key]) is not str or not 1 <= len(value[key]) <= 256:
            raise ValueError("bounded terminal task contract root required")
    for key in ("manifest_sha256", "contract_sha256"):
        if (type(value[key]) is not str or len(value[key]) != 64
                or any(char not in "0123456789abcdef" for char in value[key])):
            raise ValueError("exact terminal task contract digest required")
    unsigned = {key: item for key, item in value.items() if key != "contract_sha256"}
    if hashlib.sha256(_json(unsigned).encode()).hexdigest() != value["contract_sha256"]:
        raise ValueError("terminal task contract digest differs")
    refs = value["evidence_cids"]
    if (not isinstance(refs, list) or not 1 <= len(refs) <= 16
            or any(type(item) is not str or not 1 <= len(item) <= 256 for item in refs)
            or len(set(refs)) != len(refs)):
        raise ValueError("exact terminal descriptive evidence population required")
    tasks = value["tasks"]
    if not isinstance(tasks, list) or len(tasks) != 1 or not isinstance(tasks[0], dict) or set(tasks[0]) != TASK_FIELDS:
        raise ValueError("one complete terminal task declaration required")
    task = tasks[0]
    if task["task_key"] != "TB-CODE-TASK" or task["dependency_task_keys"] != []:
        raise ValueError("single direct terminal task population required")
    scopes = task["scope_paths"]
    if (not isinstance(scopes, list) or not 1 <= len(scopes) <= 256
            or any(type(name) is not str for name in scopes) or len(set(scopes)) != len(scopes)):
        raise ValueError("bounded unique terminal scope required")
    from pathlib import PurePosixPath
    for name in scopes:
        if not name or len(name) > 1024 or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or "\\" in name or "\x00" in name:
            raise ValueError("relative terminal scope required")
    # Reuse the canonical proposal's typed row schemas. This is structural
    # validation only; the owner separately verifies signature and authority.
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import _proposal_schema
    from types import SimpleNamespace
    from jsonschema import Draft202012Validator
    schema = _proposal_schema(SimpleNamespace(budget=SimpleNamespace(
        max_goals=2, max_tasks=1, max_graph_depth=4, max_serialized_bytes=MAX_BYTES, max_provider_tokens=MAX_BYTES)))
    for field, definition in (("outputs", "output"), ("validations", "validation"), ("acceptance", "acceptance")):
        rows = task[field]
        if not isinstance(rows, list) or not 1 <= len(rows) <= 64:
            raise ValueError("bounded terminal task rows required")
        try:
            for row in rows:
                Draft202012Validator(schema["definitions"][definition]).validate(row)
        except Exception:
            raise ValueError("terminal task declaration row differs from canonical grammar") from None
    if any(row["path"] not in scopes for row in task["outputs"]):
        raise ValueError("terminal output outside declared scope")
    if any(not set(row["evidence_cids"]) <= set(refs) for row in task["acceptance"]):
        raise ValueError("terminal acceptance outside descriptive evidence")
    return value


def contract_from_prompt(prompt):
    from ipfs_accelerate_py.cli_runtime.grok_structured_output import _loads
    try:
        payload = _loads(prompt)
    except (ValueError, TypeError):
        if FIELD in prompt:
            raise ValueError("malformed terminal task contract prompt") from None
        return None
    if not isinstance(payload, dict) or FIELD not in payload:
        return None
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import PROMPT_GOAL_PROVIDER_REQUEST_SCHEMA
    if payload.get("schema") != PROMPT_GOAL_PROVIDER_REQUEST_SCHEMA:
        raise ValueError("terminal task contract requires canonical planning request")
    contract = validate_task_contract(payload[FIELD])
    core = payload.get("request_core", {})
    if not isinstance(core, dict) or any(contract[key] != core.get(key) for key in ("request_cid", "scan_cid", "program_root", "policy_root")):
        raise ValueError("terminal task contract differs from canonical request roots")
    handles = payload.get("evidence_handles", [])
    if not isinstance(handles, list) or contract["evidence_cids"] != [row.get("evidence_cid") for row in handles if isinstance(row, dict)]:
        raise ValueError("terminal task contract differs from descriptive evidence population")
    task = contract["tasks"][0]
    constraints = payload.get("constraints", {})
    if (not isinstance(constraints, dict) or task["scope_paths"] != constraints.get("allowed_paths")
            or [row["argv"] for row in task["validations"]] != constraints.get("validation_commands")):
        raise ValueError("terminal task contract differs from declared scope or commands")
    return contract


def bind_terminal_task_contract(prompt, *, prepared, maximum_bytes=262_144):
    """Owner-only projection after signature/source verification; no model input."""
    if prepared.get("planning_strategy", "direct") != "direct" or prepared.get("schema") != "terminal-indexed-public-preparation@1":
        return prompt
    from . import local_planning_admission as local
    from ipfs_accelerate_py.cli_runtime.grok_structured_output import _loads
    manifest, _profile, _sources = local._manifest(prepared["manifest"], initial=True)
    if (manifest.get("tasks") != [prepared.get("spec")]
            or prepared.get("request") != manifest["planning_inputs"]["request"]
            or prepared.get("scan") != manifest["planning_inputs"]["scan"]):
        raise ValueError("terminal task hints differ from independently signed preparation")
    payload = _loads(prompt)
    if not isinstance(payload, dict) or FIELD in payload:
        raise ValueError("unmodified terminal planner request required")
    spec = manifest["tasks"][0]
    contract = {
        "schema": SCHEMA, **manifest["planning_roots"], "policy_root": prepared["request"]["policy_root"],
        "manifest_sha256": hashlib.sha256(_json(manifest).encode()).hexdigest(),
        "evidence_cids": [row["evidence_cid"] for row in payload["evidence_handles"]],
        "tasks": [{"task_key": spec["task_key"], "scope_paths": spec["scope_paths"],
            "outputs": spec["outputs"], "acceptance": spec["acceptance"],
            "dependency_task_keys": spec["dependencies"],
            "validations": [{key: value for key, value in row.items() if key != "policy_cid"}
                            for row in spec["validations"]]}],
        "execution_authority": False, "completion_authority": False,
    }
    contract["contract_sha256"] = hashlib.sha256(_json(contract).encode()).hexdigest()
    payload[FIELD] = contract
    encoded = _json(payload)
    request_budget = prepared["request"]["budget"]
    limit = min(maximum_bytes, request_budget["max_serialized_bytes"], request_budget["max_prompt_tokens"] * 4)
    if len(encoded.encode()) > limit:
        raise ValueError("terminal task contract exceeds existing planner request budget")
    contract_from_prompt(encoded)
    return encoded


def constrain_native_schema(wire, contract):
    """Intersect the native schema with exact declaration values; never rewrite output."""
    contract = validate_task_contract(contract)
    task = contract["tasks"][0]
    wire = json.loads(_json(wire))
    task_properties = {key: {"const": value} for key, value in task.items()}
    task_properties.update(goal_key={"const": "TB-SUBGOAL"}, assumptions={"const": []}, risks={"const": []},
        resource_class={"const": "cpu-medium"}, fallback_behavior={"const": "fail_closed"},
        predicted_files={"items": {"enum": task["scope_paths"]}},
        evidence_cids={"items": {"enum": contract["evidence_cids"]}})
    wire["definitions"]["task"]["allOf"] = [{"properties": task_properties}]
    wire["definitions"]["goal"]["allOf"] = [{"properties": {
        "acceptance": {"const": task["acceptance"]}, "assumptions": {"const": []}, "risks": {"const": []},
        "dependency_goal_keys": {"const": []}, "scope_paths": {"items": {"enum": task["scope_paths"]}},
        "evidence_cids": {"items": {"enum": contract["evidence_cids"]}}}, "anyOf": [
            {"properties": {"goal_key": {"const": "TB-GOAL"}, "parent_goal_key": {"const": ""}}},
            {"properties": {"goal_key": {"const": "TB-SUBGOAL"}, "parent_goal_key": {"const": "TB-GOAL"}}},
        ]}]
    # Key uniqueness and root/child existence remain canonical graph checks;
    # the native branches constrain row values without changing those checks.
    wire["allOf"] = [{"properties": {
        "root_goal_key": {"const": "TB-GOAL"}, "goals": {"minItems": 2, "maxItems": 2},
        "tasks": {"minItems": 1, "maxItems": 1}, "unresolved_questions": {"const": []},
        "uncertainty_debt": {"const": []},
    }}]
    return wire
