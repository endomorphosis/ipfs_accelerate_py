"""Closed public task declarations; structural checks carry no semantic authority."""
from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
import re

SCHEMA = "terminal-public-task-profile@1"
DATA_SCHEMA = "terminal-public-task-profile@2"
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


def validate_task_profile(profile: dict, *, instruction: str | None = None) -> dict:
    schema = profile.get("schema") if type(profile) is dict else None
    keys = {"schema", "instruction_sha256", "input_paths", "outputs"}
    if schema == DATA_SCHEMA:
        keys.add("data_inputs")
    if (type(profile) is not dict or set(profile) != keys
            or schema not in {SCHEMA, DATA_SCHEMA}
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
    if len(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")) + 1 > 65536:
        raise ValueError("public task profile exceeds its 65536-byte bound")
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


def task_profile_spec(profile: dict, *, policy_cid: str) -> dict:
    profile = validate_task_profile(profile)
    return {"task_key": "TB-CODE-TASK", "scope_paths": list(dict.fromkeys([
        *task_profile_worker_inputs(profile), *(item["path"] for item in profile["outputs"])])),
        "dependencies": [], "outputs": profile["outputs"],
        "validations": [{"validation_key": "public-structural-smoke", "argv": ["python3", "-I", "-B", SMOKE],
                         "cwd": ".", "expected_exit_codes": [0], "policy_cid": policy_cid}],
        "acceptance": [{"criterion_key": "declared-output-structure",
            "criterion": ("Declared output files satisfy byte bounds and Python outputs parse; benchmark correctness remains unverified"
                if profile["schema"] == SCHEMA else
                "Declared outputs satisfy byte bounds and selected Python, JSON, JSONL, XML or calendar structural syntax; benchmark correctness remains unverified"),
            "evidence_cids": [], "validation_keys": ["public-structural-smoke"]}]}


def task_profile_smoke(profile: dict) -> str:
    profile = validate_task_profile(profile)
    outputs = profile["outputs"]
    smoke = '''"""Fixed structural output check; no candidate execution or semantic proof."""
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
        smoke = smoke.replace("root = Path.cwd()", DATA_VALIDATOR + "\nroot = Path.cwd()")
        smoke += '''    if output["media_type"] in ("application/json", "application/x-ndjson", "application/xml", "text/calendar"):
        _data_format(raw, output["media_type"])
'''
    return smoke
