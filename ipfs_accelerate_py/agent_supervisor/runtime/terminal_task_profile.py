"""Closed public task declarations; structural checks carry no semantic authority."""
from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
import re

SCHEMA = "terminal-public-task-profile@1"
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


def validate_task_profile(profile: dict, *, instruction: str | None = None) -> dict:
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


def task_profile_bytes(profile: dict) -> bytes:
    return (json.dumps(validate_task_profile(profile), sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def task_profile_worker_inputs(profile: dict) -> list[str]:
    return [INSTRUCTION, SMOKE, PROFILE, *validate_task_profile(profile)["input_paths"]]


def task_profile_index_paths(profile: dict) -> list[str]:
    paths = validate_task_profile(profile)["input_paths"]
    if not paths:
        raise ValueError("full indexed public task requires actual source inputs; instruction-only indexing is unavailable")
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
            "criterion": "Declared output files satisfy byte bounds and Python outputs parse; benchmark correctness remains unverified",
            "evidence_cids": [], "validation_keys": ["public-structural-smoke"]}]}


def task_profile_smoke(profile: dict) -> str:
    outputs = validate_task_profile(profile)["outputs"]
    return '''"""Fixed structural output check; no candidate execution or semantic proof."""
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
