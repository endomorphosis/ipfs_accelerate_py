"""Task-bound context nominations outside the canonical intent task body.

An exact bundle digest belongs to the launch configuration. These nominations
select independently verified context artifacts and cannot alter provider,
validation, scheduling or completion authority. Keeping them outside task state
avoids a world snapshot containing its own yet-to-be-computed digest.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


SCHEMA = "supervisor-task-context-nominations@1"
KEYS = frozenset({
    "semantic context artifact", "semantic context sha256", "semantic context refresh",
    "world context artifact", "world context sha256", "world context repository",
    "code retrieval artifact", "code retrieval sha256",
})
MAX_BYTES = 131_072


def _normalize(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("context metadata must be a dictionary")
    result = {}
    for key, value in metadata.items():
        normalized = str(key).strip().lower().replace("_", " ")
        if (normalized not in KEYS or normalized in result or not isinstance(value, str)
                or not value or len(value.encode()) > 4096 or any(c in value for c in "\n\r\0")):
            raise ValueError("invalid or authority-bearing context nomination")
        result[normalized] = value
    for kind in ("semantic", "world", "code retrieval"):
        prefix = kind if kind == "code retrieval" else kind + " context"
        if bool(result.get(prefix + " artifact")) != bool(result.get(prefix + " sha256")):
            raise ValueError("context nomination requires both artifact and digest")
    return result


def write_task_context_bundle(*, repository: Path, prepared: list[dict], output: Path) -> dict:
    root, output = Path(repository).resolve(strict=True), Path(output).absolute()
    if (not 1 <= len(prepared) <= 16 or output.exists() or output.resolve() != output
            or not output.is_relative_to(root)):
        raise ValueError("a new repository-contained bundle and 1 to 16 tasks are required")
    tasks = []
    for item in prepared:
        if item.get("schema") != "supervisor-task-context-preparation@1":
            raise ValueError("native prepared task context required")
        tasks.append({"task_cid": item["task_cid"], "task_id": item["task_id"],
                      "metadata": _normalize(item["metadata"])})
    if len({item["task_cid"] for item in tasks}) != len(tasks) or len({item["task_id"] for item in tasks}) != len(tasks):
        raise ValueError("context bundle task identities must be unique")
    payload = {"schema": SCHEMA, "tasks": tasks, "completion_authority": False}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    if len(raw) > MAX_BYTES:
        raise ValueError("context bundle exceeds its byte bound")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(raw)
    return {"artifact": output.relative_to(root).as_posix(), "sha256": hashlib.sha256(raw).hexdigest()}


def load_task_context_nomination(*, repository: Path, artifact: str, expected_sha256: str,
                                task_cid: str, task_id: str) -> dict[str, str]:
    root = Path(repository).resolve(strict=True)
    relative = Path(artifact)
    path = root / relative
    if (relative.is_absolute() or ".." in relative.parts or relative.as_posix() != artifact
            or path.is_symlink() or not path.resolve().is_relative_to(root)):
        raise ValueError("context bundle escapes repository")
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("context bundle digest differs")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate context bundle key")
            result[key] = value
        return result

    payload = json.loads(raw, object_pairs_hook=unique)
    if (not isinstance(payload, dict) or set(payload) != {"schema", "tasks", "completion_authority"}
            or payload["schema"] != SCHEMA or payload["completion_authority"] is not False
            or not isinstance(payload["tasks"], list) or not 1 <= len(payload["tasks"]) <= 16):
        raise ValueError("invalid context bundle")
    selected, identities, aliases = None, set(), set()
    for task in payload["tasks"]:
        if (not isinstance(task, dict) or set(task) != {"task_cid", "task_id", "metadata"}
                or not isinstance(task["task_cid"], str) or not task["task_cid"]
                or not isinstance(task["task_id"], str) or not task["task_id"]
                or task["task_cid"] in identities or task["task_id"] in aliases):
            raise ValueError("invalid context bundle task identity")
        identities.add(task["task_cid"])
        aliases.add(task["task_id"])
        metadata = _normalize(task["metadata"])
        if task["task_cid"] == task_cid or task["task_id"] == task_id:
            if task["task_cid"] != task_cid or task["task_id"] != task_id:
                raise ValueError("context nomination has a foreign task identity")
            selected = metadata
    if selected is None:
        raise ValueError("context bundle does not contain the dispatched task")
    return selected
