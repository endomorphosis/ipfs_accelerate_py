"""Closed public task declarations; structural checks carry no semantic authority."""
from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
import re

SCHEMA = "terminal-public-task-profile@1"
# Version 2 is reserved for the independently developed structured-data profile.
MULTITASK_SCHEMA = "terminal-public-task-profile@3"
MAX_TASKS = 16
PROFILE = ".supervisor-task-profile.json"
INSTRUCTION = ".supervisor-instruction.md"
SMOKE = ".supervisor-public-smoke.py"
MAX_INPUTS = 252
MAX_CREATED_OUTPUTS = 32
MAX_OUTPUT_BYTES = 1_000_000
MAX_TOTAL_OUTPUT_BYTES = 4_000_000


def normalized_instruction(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("instruction must be UTF-8 text")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def instruction_sha256(text: str) -> str:
    return hashlib.sha256(normalized_instruction(text).encode("utf-8")).hexdigest()


def _path(value):
    if (not isinstance(value, str) or not value or len(value.encode("utf-8")) > 1024
            or PurePosixPath(value).is_absolute() or PurePosixPath(value).as_posix() != value
            or any(part in {".", "..", ".git", ".runtime"} or part.startswith(".supervisor")
                   for part in value.split("/"))
            or any(char in value for char in "\\*?[]\x00\n\r")
            or not value.isprintable()):
        raise ValueError("public task paths must be exact canonical nonreserved relative files")
    return value


def _validate_single_task_profile(profile: dict, *, instruction: str | None = None) -> dict:
    if (type(profile) is not dict or set(profile) != {"schema", "instruction_sha256", "input_paths", "outputs"}
            or profile.get("schema") != SCHEMA
            or not isinstance(profile.get("instruction_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", profile["instruction_sha256"]) is None):
        raise ValueError("unsupported public task profile")
    if instruction is not None and profile["instruction_sha256"] != instruction_sha256(instruction):
        raise ValueError("public task profile instruction digest differs")
    inputs, outputs = profile["input_paths"], profile["outputs"]
    if type(inputs) is not list or len(inputs) > MAX_INPUTS:
        raise ValueError("public task source inventory exceeds its bound")
    inputs = [_path(name) for name in inputs]
    if len(set(inputs)) != len(inputs):
        raise ValueError("public task input paths must be unique")
    if type(outputs) is not list or not 1 <= len(outputs) <= 64:
        raise ValueError("public task requires 1 to 64 exact outputs")
    checked = []
    for output in outputs:
        if type(output) is not dict or set(output) != {"path", "effect", "media_type"}:
            raise ValueError("public task output declaration differs")
        name = _path(output["path"])
        if (output["effect"] not in ("create", "modify")
                or (name in inputs) != (output["effect"] == "modify")
                or not isinstance(output["media_type"], str)
                or re.fullmatch(r"[a-z0-9.+-]+/[a-z0-9.+-]+", output["media_type"]) is None
                or len(output["media_type"]) > 128):
            raise ValueError("public task output effect or media type differs")
        checked.append(dict(output))
    output_names = [item["path"] for item in checked]
    if len(set(output_names)) != len(output_names):
        raise ValueError("public task outputs must be unique")
    created_count = sum(item["effect"] == "create" for item in checked)
    # Match native manifest admission before preparation mutates the workspace.
    if created_count > MAX_CREATED_OUTPUTS:
        raise ValueError("public task exceeds the native 32-created-output manifest bound")
    # Publication observes both the signed baseline (including three support
    # files) and all newly created outputs under the same native source limit.
    if created_count and len(inputs) + 3 + created_count > 256:
        raise ValueError("public task exceeds the native 256-source published manifest bound")
    paths = set(inputs) | set(output_names)
    if any(str(parent) in paths for name in paths for parent in PurePosixPath(name).parents if str(parent) != "."):
        raise ValueError("public task file paths cannot contain another declared file")
    # Native manifests without creations have a deliberately smaller bound.
    if not created_count and len(inputs) + 3 > 128:
        raise ValueError("modify-only public task exceeds the native 128-source manifest bound")
    canonical = {"schema": SCHEMA, "instruction_sha256": profile["instruction_sha256"],
                 "input_paths": sorted(inputs), "outputs": sorted(checked, key=lambda item: item["path"])}
    if len(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")) + 1 > 65536:
        raise ValueError("public task profile exceeds its 65536-byte bound")
    return canonical


def _identifier(value, noun: str) -> str:
    if type(value) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", value) is None:
        raise ValueError(f"public task requires a bounded exact {noun}")
    return value


def validate_task_profile(profile: dict, *, instruction: str | None = None) -> dict:
    """Canonicalize a closed declaration without granting execution authority."""
    if type(profile) is not dict or profile.get("schema") != MULTITASK_SCHEMA:
        return _validate_single_task_profile(profile, instruction=instruction)
    if set(profile) != {"schema", "instruction_sha256", "input_paths", "outputs",
                        "intent_requirement_contract_cid", "tasks"}:
        raise ValueError("unsupported public multi-task profile")
    base = _validate_single_task_profile(
        {name: SCHEMA if name == "schema" else profile[name]
         for name in ("schema", "instruction_sha256", "input_paths", "outputs")},
        instruction=instruction,
    )
    from ..core.multiformats_identity import validate_cid
    try:
        validate_cid(profile["intent_requirement_contract_cid"], codecs=("dag-json",))
    except (TypeError, ValueError) as exc:
        raise ValueError("public multi-task profile requires a canonical requirement-contract CID") from exc
    if type(profile["tasks"]) is not list or not 2 <= len(profile["tasks"]) <= MAX_TASKS:
        raise ValueError("public multi-task profile requires 2 to 16 reviewed tasks")
    tasks, owned_outputs = [], []
    for item in profile["tasks"]:
        if type(item) is not dict or set(item) != {
            "task_key", "operation_id", "output_paths", "dependencies", "validation_key", "criterion_key",
        }:
            raise ValueError("public multi-task binding fields differ")
        task = {name: _identifier(item[name], name)
                for name in ("task_key", "operation_id", "validation_key", "criterion_key")}
        if type(item["output_paths"]) is not list or not 1 <= len(item["output_paths"]) <= 64:
            raise ValueError("public multi-task binding needs bounded nonempty output ownership")
        output_paths = [_path(name) for name in item["output_paths"]]
        if len(output_paths) != len(set(output_paths)):
            raise ValueError("public multi-task output ownership must be unique")
        if type(item["dependencies"]) is not list or len(item["dependencies"]) > MAX_TASKS - 1:
            raise ValueError("public multi-task dependencies exceed their bound")
        dependencies = [_identifier(name, "dependency task key") for name in item["dependencies"]]
        if len(dependencies) != len(set(dependencies)):
            raise ValueError("public multi-task dependencies must be unique")
        task.update(output_paths=sorted(output_paths), dependencies=sorted(dependencies))
        tasks.append(task)
        owned_outputs.extend(output_paths)
    for name in ("task_key", "operation_id", "validation_key", "criterion_key"):
        if len({task[name] for task in tasks}) != len(tasks):
            raise ValueError(f"public multi-task {name} must be globally unique")
    if (len(owned_outputs) != len(set(owned_outputs))
            or set(owned_outputs) != {item["path"] for item in base["outputs"]}):
        raise ValueError("public multi-task output ownership must be disjoint and cover exactly all outputs")
    by_key = {task["task_key"]: task for task in tasks}
    for task in tasks:
        if task["task_key"] in task["dependencies"] or not set(task["dependencies"]) <= set(by_key):
            raise ValueError("public multi-task dependency is unknown or self-referential")
    active, visited = set(), set()

    def visit(key):
        if key in active:
            raise ValueError("public multi-task dependency cycle")
        if key not in visited:
            active.add(key)
            for dependency in by_key[key]["dependencies"]:
                visit(dependency)
            active.remove(key)
            visited.add(key)

    for key in by_key:
        visit(key)
    canonical = {**base, "schema": MULTITASK_SCHEMA,
                 "intent_requirement_contract_cid": profile["intent_requirement_contract_cid"],
                 "tasks": sorted(tasks, key=lambda item: item["task_key"])}
    if len(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")) + 1 > 65536:
        raise ValueError("public task profile exceeds its 65536-byte bound")
    return canonical


def validate_task_profile_contract(profile: dict, contract: dict, *, instruction: str | None = None) -> dict:
    """Join reviewed atomic requirements to exact administrative task ownership.

    This checks authored declarations, not the semantic fidelity of the source
    interpretation or successful execution of any task.
    """
    profile = validate_task_profile(profile, instruction=instruction)
    if profile["schema"] != MULTITASK_SCHEMA:
        raise ValueError("reviewed multi-task contract requires the explicit multi-task profile")
    if type(contract) is not dict:
        raise ValueError("public multi-task profile requires its reviewed requirement contract")
    from ..core.multiformats_identity import cid_for_dag_json
    from ..prompt.intent_plan_coverage import (
        INTENT_SYMBOLIC_REQUIREMENT_CONTRACT_SCHEMA, validate_intent_requirement_contract,
    )
    canonical = validate_intent_requirement_contract(contract, source_text=instruction)
    if (canonical["schema"] != INTENT_SYMBOLIC_REQUIREMENT_CONTRACT_SCHEMA
            or canonical["source_path"] != INSTRUCTION
            or cid_for_dag_json(canonical) != profile["intent_requirement_contract_cid"]):
        raise ValueError("public multi-task requirement contract schema, source or identity differs")
    source = "".join(unit["text"] for unit in canonical["ledger"]["source_units"])
    if instruction_sha256(source) != profile["instruction_sha256"]:
        raise ValueError("public multi-task requirement contract instruction digest differs")
    operations = {item["operation_id"]: item for item in canonical["symbolic_operations"]["operations"]}
    tasks = {item["operation_id"]: item for item in profile["tasks"]}
    if set(operations) != set(tasks):
        raise ValueError("public multi-task bindings must cover exactly the reviewed operations")
    req_operations = {matcher["requirement_id"]: op["operation_id"]
                      for op in operations.values() for matcher in op["matchers"]}
    groundings = {item["requirement_id"]: item for item in canonical["requirements"]}
    outputs = {item["path"]: item for item in profile["outputs"]}
    for op_id, task in tasks.items():
        operation = operations[op_id]
        required_dependencies = {req_operations[dependency]
            for matcher in operation["matchers"]
            for dependency in groundings[matcher["requirement_id"]]["dependency_requirement_ids"]}
        if set(operation["dependency_operation_ids"]) != required_dependencies:
            raise ValueError("public multi-task operation dependencies must equal reviewed requirement ordering")
        expected_dependencies = sorted(operations[dependency]["task_key"]
                                       for dependency in operation["dependency_operation_ids"])
        if (task["task_key"] != operation["task_key"]
                or [outputs[name] for name in task["output_paths"]] != operation["outputs"]
                or [task["validation_key"]] != operation["validation_keys"]
                or task["dependencies"] != expected_dependencies):
            raise ValueError("public multi-task binding differs from its exact reviewed operation")
    return canonical


def task_profile_bytes(profile: dict) -> bytes:
    return (json.dumps(validate_task_profile(profile), sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def task_profile_worker_inputs(profile: dict) -> list[str]:
    return [INSTRUCTION, SMOKE, PROFILE, *validate_task_profile(profile)["input_paths"]]


def task_profile_index_paths(profile: dict) -> list[str]:
    paths = validate_task_profile(profile)["input_paths"]
    if len(paths) > 64:
        raise ValueError("full indexed public task exceeds native 64-file vector bound")
    return paths


def task_profile_spec(profile: dict, *, policy_cid: str) -> dict:
    profile = validate_task_profile(profile)
    if profile["schema"] != SCHEMA:
        raise ValueError("singleton public task specification requires version 1")
    return {"task_key": "TB-CODE-TASK", "scope_paths": list(dict.fromkeys([
        *task_profile_worker_inputs(profile), *(item["path"] for item in profile["outputs"])])),
        "dependencies": [], "outputs": profile["outputs"],
        "validations": [{"validation_key": "public-structural-smoke", "argv": ["python3", "-I", "-B", SMOKE],
                         "cwd": ".", "expected_exit_codes": [0], "policy_cid": policy_cid}],
        "acceptance": [{"criterion_key": "declared-output-structure",
            "criterion": "Declared output files satisfy byte bounds and Python outputs parse; benchmark correctness remains unverified",
            "evidence_cids": [], "validation_keys": ["public-structural-smoke"]}]}


def task_profile_specs(profile: dict, *, policy_cid: str) -> list[dict]:
    """Produce deterministic native declarations; callers must join the contract."""
    profile = validate_task_profile(profile)
    if profile["schema"] == SCHEMA:
        return [task_profile_spec(profile, policy_cid=policy_cid)]
    outputs = {item["path"]: item for item in profile["outputs"]}
    return [{"task_key": task["task_key"], "scope_paths": list(dict.fromkeys([
        *task_profile_worker_inputs(profile), *task["output_paths"]])),
        "dependencies": task["dependencies"], "outputs": [outputs[name] for name in task["output_paths"]],
        "validations": [{"validation_key": task["validation_key"],
            "argv": ["python3", "-I", "-B", SMOKE, "--task", task["task_key"]],
            "cwd": ".", "expected_exit_codes": [0], "policy_cid": policy_cid}],
        "acceptance": [{"criterion_key": task["criterion_key"],
            "criterion": "Declared task output files satisfy byte bounds and Python outputs parse; benchmark correctness remains unverified",
            "evidence_cids": [], "validation_keys": [task["validation_key"]]}]}
        for task in profile["tasks"]]


def task_profile_smoke(profile: dict) -> str:
    profile = validate_task_profile(profile)
    outputs = profile["outputs"]
    script = '''"""Fixed structural output check; no candidate execution or semantic proof."""
import ast
import json
import os
from pathlib import Path
import stat

outputs = json.loads(''' + repr(json.dumps(outputs, sort_keys=True, separators=(",", ":"))) + ''')
root = Path.cwd().resolve(strict=True)
total = 0
for output in outputs:
    path = root / output["path"]
    for parent in (*path.parents, path):
        if parent == root or root in parent.parents:
            assert not parent.is_symlink(), "output traverses a symlink"
    assert path.resolve(strict=True) == path, "output path differs"
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        before = os.fstat(stream.fileno())
        assert stat.S_ISREG(before.st_mode), "output must be a regular file"
        assert before.st_size <= ''' + str(MAX_OUTPUT_BYTES) + ''', "output exceeds byte bound"
        raw = stream.read(''' + str(MAX_OUTPUT_BYTES + 1) + ''')
        after = os.fstat(stream.fileno())
    assert (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "output changed during validation"
    assert len(raw) == before.st_size, "output length changed"
    total += len(raw)
    assert total <= ''' + str(MAX_TOTAL_OUTPUT_BYTES) + ''', "output population exceeds byte bound"
    if output["media_type"] == "text/x-python" or path.suffix == ".py":
        ast.parse(raw, filename=output["path"])
'''
    if profile["schema"] == SCHEMA:
        return script
    by_path = {item["path"]: item for item in outputs}
    task_outputs = {task["task_key"]: [by_path[name] for name in task["output_paths"]]
                    for task in profile["tasks"]}
    selector = '''import sys
task_outputs = json.loads(''' + repr(json.dumps(task_outputs, sort_keys=True, separators=(",", ":"))) + ''')
assert len(sys.argv) == 3 and sys.argv[1] == "--task", "exact task selector required"
assert sys.argv[2] in task_outputs, "unknown declared task selector"
outputs = task_outputs[sys.argv[2]]
'''
    return script.replace("root = Path.cwd().resolve(strict=True)\n", selector + "root = Path.cwd().resolve(strict=True)\n", 1)
