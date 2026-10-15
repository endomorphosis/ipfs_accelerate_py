"""Closed public declarations and opt-in acceptance observations, not proofs."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import PurePosixPath
import re

SCHEMA = "terminal-public-task-profile@1"
DATA_SCHEMA = "terminal-public-task-profile@2"
MULTITASK_SCHEMA = "terminal-public-task-profile@3"
PUBLIC_ACCEPTANCE_SCHEMA = "terminal-public-task-profile@4"
PUBLIC_VALIDATION_PROFILE = "largest-eigenval-public@1"
PUBLIC_ACCEPTANCE_RESULT_SCHEMA = "terminal-public-acceptance-result@1"
PUBLIC_INSTRUCTION_SHA256 = "6fa938389f99648508e5a9772e0d5e95d1102738c68d1ff61bd65469d76c52ab"
PUBLIC_EVALUATOR_SHA256 = "70a12a80b9caa392b1f1f0439de853932c5ac61f1a2274b5de1ce93aebd4e139"
PUBLIC_CHECKS = ("structural", "eigenpair", "performance")
PUBLIC_VALIDATION_KEYS = ("public-structural-smoke", "public-eigenpair-behavior", "public-eigenvalue-performance")
MAX_TASKS = 16
PROFILE = ".supervisor-task-profile.json"
INSTRUCTION = ".supervisor-instruction.md"
SMOKE = ".supervisor-public-smoke.py"
MAX_INPUTS = 252
MAX_CREATED_OUTPUTS = 32
MAX_OUTPUT_BYTES = 1_000_000
MAX_TOTAL_OUTPUT_BYTES = 4_000_000
DATA_MEDIA_SUFFIXES = {"application/json": ".json", "application/x-ndjson": ".jsonl",
                      "application/xml": ".xml", "text/calendar": ".ics"}


# Kept as one producer-owned fragment so independent input replay and isolated
# output validation execute the same bounded, non-executing format checks.
DATA_VALIDATOR = '''
def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result

def _constant(value):
    raise ValueError("nonfinite JSON constant")

def _float(value):
    number = float(value)
    if number == float("inf") or number == -float("inf"):
        raise ValueError("nonfinite JSON number")
    return number

def _data_format(raw, media):
    text = raw.decode("utf-8")
    if media == "application/json":
        json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant, parse_float=_float)
    elif media == "application/x-ndjson":
        lines = text.splitlines()
        if not lines or any(not line.strip() for line in lines):
            raise ValueError("JSONL requires nonempty records")
        for line in lines:
            if type(json.loads(line, object_pairs_hook=_pairs, parse_constant=_constant, parse_float=_float)) is not dict:
                raise ValueError("JSONL records must be objects")
    elif media == "application/xml":
        from xml.etree import ElementTree
        if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
            raise ValueError("XML document types and entities are not permitted")
        try:
            ElementTree.fromstring(text)
        except ElementTree.ParseError as error:
            raise ValueError("XML syntax differs") from error
    elif media == "text/calendar":
        # Structural framing only; dates, scheduling, and timezone semantics
        # remain obligations for an independent task-specific validator.
        lines = text.replace("\\r\\n", "\\n").splitlines()
        if not lines or lines[0] != "BEGIN:VCALENDAR" or lines[-1] != "END:VCALENDAR":
            raise ValueError("calendar framing required")
        if "VERSION:2.0" not in lines or not any(line.startswith("PRODID:") for line in lines):
            raise ValueError("calendar headers required")
        depth = []
        for line in lines:
            if line.startswith("BEGIN:"):
                depth.append(line[6:])
            elif line.startswith("END:"):
                if not depth or depth.pop() != line[4:]:
                    raise ValueError("calendar component framing differs")
        if depth:
            raise ValueError("calendar component is unclosed")
'''


def validate_task_data(raw: bytes, media_type: str) -> None:
    if type(raw) is not bytes or len(raw) > 262144 or media_type not in DATA_MEDIA_SUFFIXES:
        raise ValueError("bounded closed task data format required")
    namespace = {"json": json}
    exec(DATA_VALIDATOR, namespace)
    namespace["_data_format"](raw, media_type)


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
    schema = profile.get("schema") if type(profile) is dict else None
    keys = {"schema", "instruction_sha256", "input_paths", "outputs"}
    if schema == DATA_SCHEMA:
        keys.add("data_inputs")
    if schema == PUBLIC_ACCEPTANCE_SCHEMA:
        keys.add("public_validation_profile")
    if (type(profile) is not dict or set(profile) != keys
            or schema not in {SCHEMA, DATA_SCHEMA, PUBLIC_ACCEPTANCE_SCHEMA}
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
    data_inputs = []
    if schema == DATA_SCHEMA:
        data = profile["data_inputs"]
        if type(data) is not list or len(data) > 61:
            raise ValueError("public task data exceeds the 61-input support bound")
        for item in data:
            if type(item) is not dict or set(item) != {"path", "media_type"}:
                raise ValueError("closed task data declaration required")
            name = _path(item["path"])
            media = item["media_type"]
            if (name not in inputs or type(media) is not str or media not in DATA_MEDIA_SUFFIXES
                    or PurePosixPath(name).suffix != DATA_MEDIA_SUFFIXES[media]):
                raise ValueError("task data must use its declared closed format and suffix")
            data_inputs.append(dict(item))
        if len({item["path"] for item in data_inputs}) != len(data_inputs):
            raise ValueError("task data declarations must be unique")
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
    if {item["path"] for item in data_inputs} & {item["path"] for item in checked}:
        raise ValueError("declared task data inputs are immutable")
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
    canonical = {"schema": schema, "instruction_sha256": profile["instruction_sha256"],
                 "input_paths": sorted(inputs), "outputs": sorted(checked, key=lambda item: item["path"])}
    if schema == DATA_SCHEMA:
        canonical["data_inputs"] = sorted(data_inputs, key=lambda item: item["path"])
    if schema == PUBLIC_ACCEPTANCE_SCHEMA:
        if (profile["public_validation_profile"] != PUBLIC_VALIDATION_PROFILE
                or profile["instruction_sha256"] != PUBLIC_INSTRUCTION_SHA256
                or canonical["input_paths"] != ["eigen.py", "eval.py"]
                or canonical["outputs"] != [{"path": "eigen.py", "effect": "modify", "media_type": "text/x-python"}]):
            raise ValueError("public acceptance profile requires its exact reviewed instruction, inputs and output")
        canonical["public_validation_profile"] = PUBLIC_VALIDATION_PROFILE
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
    profile = validate_task_profile(profile)
    data = {item["path"] for item in profile.get("data_inputs", [])}
    paths = [name for name in profile["input_paths"] if name not in data]
    if len(paths) > 64:
        raise ValueError("full indexed public task exceeds native 64-file vector bound")
    return paths


def validate_task_profile_public_inputs(profile: dict, *, public_evaluator: bytes) -> None:
    """Bind this opt-in checker to the reviewed public evaluator, never a verifier."""
    profile = validate_task_profile(profile)
    if (profile["schema"] != PUBLIC_ACCEPTANCE_SCHEMA or type(public_evaluator) is not bytes
            or hashlib.sha256(public_evaluator).hexdigest() != PUBLIC_EVALUATOR_SHA256):
        raise ValueError("public acceptance requires the exact reviewed public evaluator bytes")


def public_validation_checks(profile: dict, *, policy_cid: str) -> list[dict]:
    profile = validate_task_profile(profile)
    if profile["schema"] != PUBLIC_ACCEPTANCE_SCHEMA:
        raise ValueError("explicit public acceptance profile required")
    return [{"validation_key": key, "argv": ["python3", "-I", "-B", SMOKE, "--check", check],
             "cwd": ".", "expected_exit_codes": [0], "policy_cid": policy_cid}
            for key, check in zip(PUBLIC_VALIDATION_KEYS, PUBLIC_CHECKS)]


# Shared verbatim by the signed checker and its closed result validator. The
# 20% spread is an authored measurement-stability rule, not a hidden threshold.
PUBLIC_TIMING_POLICY = '''
def _timing_status(rows):
    for size in (2, 4, 6, 8, 10):
        selected = [row for row in rows if row["size"] == size]
        for key in ("candidate_median_seconds", "reference_median_seconds"):
            values = [row[key] for row in selected]
            if (max(values) - min(values)) / min(values) > 0.20:
                return "inconclusive", "timing_unstable"
    faster = [row["candidate_median_seconds"] < row["reference_median_seconds"] for row in rows]
    if all(faster):
        return "passed", "faster_than_reference"
    # Any consistently non-faster size contradicts "consistently runs faster".
    # A crossing within any size is measurement uncertainty, never a failure.
    for size in (2, 4, 6, 8, 10):
        selected = [row["candidate_median_seconds"] < row["reference_median_seconds"]
                    for row in rows if row["size"] == size]
        if any(selected) and not all(selected):
            return "inconclusive", "timing_unstable"
    return "failed", "not_faster_than_reference"
'''


def public_validation_result(value: object, *, check: str) -> dict:
    """Copy a bounded public observation; it grants no execution authority."""
    if (check not in PUBLIC_CHECKS or type(value) is not dict
            or set(value) != {"schema", "profile", "check", "status", "reason", "measurements"}
            or value["schema"] != PUBLIC_ACCEPTANCE_RESULT_SCHEMA
            or value["profile"] != PUBLIC_VALIDATION_PROFILE or value["check"] != check):
        raise ValueError("closed public acceptance result required")
    pairs = {("passed", "obligations_satisfied"), ("failed", "output_structure_invalid")} if check == "structural" else {
        ("passed", "obligations_satisfied"), ("failed", "candidate_behavior_invalid")}
    if check == "performance":
        pairs = {("passed", "faster_than_reference"), ("failed", "not_faster_than_reference"),
                 ("failed", "candidate_behavior_invalid"), ("inconclusive", "timing_unstable"),
                 ("inconclusive", "timing_unavailable")}
    if (type(value["status"]) is not str or type(value["reason"]) is not str
            or (value["status"], value["reason"]) not in pairs or type(value["measurements"]) is not list):
        raise ValueError("public acceptance status or reason differs")
    rows = value["measurements"]
    measured = check == "performance" and value["reason"] not in {"candidate_behavior_invalid", "timing_unavailable"}
    if not measured:
        if rows:
            raise ValueError("unmeasured public check cannot carry measurements")
        return {**value, "measurements": []}
    expected = [(size, batch) for size in (2, 4, 6, 8, 10) for batch in range(3)]
    if len(rows) != len(expected):
        raise ValueError("public timing requires all fixed size and batch measurements")
    for row, (size, batch) in zip(rows, expected):
        if (type(row) is not dict or set(row) != {"size", "batch", "candidate_median_seconds", "reference_median_seconds"}
                or type(row["size"]) is not int or row["size"] != size
                or type(row["batch"]) is not int or row["batch"] != batch):
            raise ValueError("public timing measurement population differs")
        for key in ("candidate_median_seconds", "reference_median_seconds"):
            number = row[key]
            if type(number) not in (int, float) or not math.isfinite(number) or not 0 < number <= 3600:
                raise ValueError("public timing measurements must be positive finite bounded numbers")
    namespace = {}
    exec(PUBLIC_TIMING_POLICY, namespace)
    if (value["status"], value["reason"]) != namespace["_timing_status"](rows):
        raise ValueError("public timing outcome differs from unrounded measurements")
    return {**value, "measurements": [dict(row) for row in rows]}


def task_profile_spec(profile: dict, *, policy_cid: str) -> dict:
    profile = validate_task_profile(profile)
    if profile["schema"] == MULTITASK_SCHEMA:
        raise ValueError("singleton public task specification excludes the multi-task profile")
    spec = {"task_key": "TB-CODE-TASK", "scope_paths": list(dict.fromkeys([
        *task_profile_worker_inputs(profile), *(item["path"] for item in profile["outputs"])])),
        "dependencies": [], "outputs": profile["outputs"],
        "validations": [{"validation_key": "public-structural-smoke", "argv": ["python3", "-I", "-B", SMOKE],
                         "cwd": ".", "expected_exit_codes": [0], "policy_cid": policy_cid}],
        "acceptance": [{"criterion_key": "declared-output-structure",
            "criterion": ("Declared output files satisfy byte bounds and Python outputs parse; benchmark correctness remains unverified"
                if profile["schema"] == SCHEMA else
                "Declared outputs satisfy byte bounds and selected Python, JSON, JSONL, XML or calendar structural syntax; benchmark correctness remains unverified"),
            "evidence_cids": [], "validation_keys": ["public-structural-smoke"]}]}
    if profile["schema"] == PUBLIC_ACCEPTANCE_SCHEMA:
        spec["validations"] = public_validation_checks(profile, policy_cid=policy_cid)
        criteria = (
            ("declared-output-structure", "Declared eigen.py is bounded, regular Python source and parses"),
            ("public-dominant-eigenpair", "Public-domain matrices produce finite nonzero dominant eigenpairs satisfying numpy.allclose with its public default tolerances"),
            ("public-faster-than-reference", "Unrounded median candidate time is strictly less than numpy reference time in all three 100-call batches for every public size; mixed direction or over 20 percent median spread is inconclusive"),
        )
        spec["acceptance"] = [{"criterion_key": key, "criterion": text, "evidence_cids": [],
            "validation_keys": [validation]} for (key, text), validation in zip(criteria, PUBLIC_VALIDATION_KEYS)]
    return spec


def task_profile_specs(profile: dict, *, policy_cid: str) -> list[dict]:
    """Produce deterministic native declarations; callers must join the contract."""
    profile = validate_task_profile(profile)
    if profile["schema"] != MULTITASK_SCHEMA:
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
    if profile["schema"] == PUBLIC_ACCEPTANCE_SCHEMA:
        return _public_acceptance_smoke(profile)
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
    if profile["schema"] == DATA_SCHEMA:
        script = script.replace("root = Path.cwd()", DATA_VALIDATOR + "\nroot = Path.cwd()")
        script += '''    if output["media_type"] in ("application/json", "application/x-ndjson", "application/xml", "text/calendar"):
        _data_format(raw, output["media_type"])
'''
    if profile["schema"] != MULTITASK_SCHEMA:
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


def _public_acceptance_smoke(profile: dict) -> str:
    structural = task_profile_smoke({key: SCHEMA if key == "schema" else value
        for key, value in profile.items() if key != "public_validation_profile"})
    header = '''"""Signed public acceptance observations, not hidden-verifier or proof authority.

Public instruction: dominant means greatest magnitude; real float64 square
matrices up to 10, complex results allowed, numpy.allclose(A@v, lambda*v).
Public eval: numpy.linalg.eig reference, sizes 2,4,6,8,10, 100 calls/median.
Authored checks add deterministic public-domain cases and three timing batches.
Public criterion: every unrounded candidate median must be strictly faster.
Mixed directions within a size or over 20% within-role spread is inconclusive.
No hidden thresholds are used. Printed rounded eval.py values are never parsed.
"""
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
'''
    constants = ("STRUCTURAL_SOURCE = " + repr(structural) + "\n"
        + "PROFILE = " + repr(PUBLIC_VALIDATION_PROFILE) + "\n"
        + "RESULT_SCHEMA = " + repr(PUBLIC_ACCEPTANCE_RESULT_SCHEMA) + "\n"
        + "EVALUATOR_SHA256 = " + repr(PUBLIC_EVALUATOR_SHA256) + "\n")
    body = '''
def _numerical_api():
    # Bind the numerical operations before importing the candidate. This is a
    # defensive check, not a proof against arbitrary code in the child process.
    import numpy as np
    return SimpleNamespace(array=np.array, asarray=np.asarray, allclose=np.allclose,
        isfinite=np.isfinite, abs=np.abs, argmax=np.argmax, max=np.max,
        diag=np.diag, eig=np.linalg.eig, median=np.median,
        rng=np.random.default_rng, float64=np.float64)

def _reference(api, matrix):
    values, vectors = api.eig(matrix)
    index = api.argmax(api.abs(values))
    return values[index], vectors[:, index]

def _behavior(candidate, api):
    rng = api.rng(314159)
    matrices = [api.array([[0., -2.], [2., 0.]]),
                api.diag(api.array([-9., 4., 1.])),
                api.array([[1., 10.], [0., 2.]])]
    matrices += [rng.normal(size=(size, size)).astype(api.float64)
                 for size in range(1, 11) for _ in range(3)]
    for matrix in matrices:
        argument = matrix.copy()
        value, vector = candidate(argument)
        value, vector = api.asarray(value), api.asarray(vector)
        assert value.shape == () and vector.shape == (len(matrix),)
        assert bool(api.isfinite(value)) and bool(api.isfinite(vector).all())
        assert api.allclose(argument, matrix, rtol=0, atol=0), "input changed"
        assert not api.allclose(vector, 0), "zero eigenvector"
        assert api.allclose(matrix @ vector, value * vector), "invalid eigenpair"
        expected, _ = _reference(api, matrix)
        assert api.allclose(api.abs(value), api.abs(expected)), "not dominant"

def _measure(candidate, api, clock):
    rows = []
    rng = api.rng(271828)
    for size in (2, 4, 6, 8, 10):
        warmup = rng.normal(size=(size, size)).astype(api.float64)
        candidate(warmup.copy())
        _reference(api, warmup.copy())
        for batch in range(3):
            measured = {"candidate": [], "reference": []}
            for index in range(100):
                matrix = rng.normal(size=(size, size)).astype(api.float64)
                order = ("candidate", "reference") if (batch + index) % 2 == 0 else ("reference", "candidate")
                for role in order:
                    argument = matrix.copy()
                    start = clock()
                    if role == "candidate":
                        candidate(argument)
                    else:
                        _reference(api, argument)
                    elapsed = clock() - start
                    if not math.isfinite(elapsed) or not 0 < elapsed <= 3600:
                        return []
                    measured[role].append(elapsed)
            rows.append({"size": size, "batch": batch,
                "candidate_median_seconds": float(api.median(measured["candidate"])),
                "reference_median_seconds": float(api.median(measured["reference"]))})
    return rows

def _check(check):
    result = {"schema": RESULT_SCHEMA, "profile": PROFILE, "check": check,
              "status": "failed", "reason": ("output_structure_invalid" if check == "structural" else "candidate_behavior_invalid"), "measurements": []}
    try:
        exec(STRUCTURAL_SOURCE, {})
        if check == "structural":
            result.update(status="passed", reason="obligations_satisfied")
            return result
        result["reason"] = "candidate_behavior_invalid"
        evaluator = Path("eval.py")
        assert not evaluator.is_symlink() and hashlib.sha256(evaluator.read_bytes()).hexdigest() == EVALUATOR_SHA256
        api = _numerical_api()
        clock = time.perf_counter
        spec = importlib.util.spec_from_file_location("_public_candidate", Path("eigen.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        candidate = module.find_dominant_eigenvalue_and_eigenvector
        _behavior(candidate, api)
        if check == "eigenpair":
            result.update(status="passed", reason="obligations_satisfied")
            return result
        rows = _measure(candidate, api, clock)
        if not rows:
            result.update(status="inconclusive", reason="timing_unavailable")
            return result
        status, reason = _timing_status(rows)
        result.update(status=status, reason=reason, measurements=rows)
    except Exception:
        # Candidate exceptions or filenames must not leak into result metadata.
        pass
    return result

def main():
    if len(sys.argv) != 3 or sys.argv[1] != "--check" or sys.argv[2] not in ("structural", "eigenpair", "performance"):
        raise SystemExit(2)
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        result = _check(sys.argv[2])
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    raise SystemExit({"passed": 0, "failed": 1, "inconclusive": 2}[result["status"]])

if __name__ == "__main__":
    main()
'''
    return header + constants + PUBLIC_TIMING_POLICY + body
