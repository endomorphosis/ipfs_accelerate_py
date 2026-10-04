"""Task-bound context nominations outside the canonical intent task body.

An exact bundle digest belongs to the launch configuration. These nominations
select independently verified context artifacts and cannot alter provider,
validation, scheduling or completion authority. Keeping them outside task state
avoids a world snapshot containing its own yet-to-be-computed digest.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


SCHEMA = "supervisor-task-context-nominations@1"
SOURCE384_SCHEMA = "supervisor-task-context-nominations@2"
SOURCE384_REFERENCE_SCHEMA = "supervisor-task-context-nominations@3"
RECEIPT_REFERENCE_SCHEMA = "supervisor-source384-receipt-reference@1"
SOURCE384_RECEIPT_SCHEMAS = frozenset({"terminal-source384-repository-context@1",
    "terminal-source384-repository-context@2", "terminal-source384-repository-context@3"})
KEYS = frozenset({
    "semantic context artifact", "semantic context sha256", "semantic context refresh",
    "world context artifact", "world context sha256", "world context repository",
    "code retrieval artifact", "code retrieval sha256",
})
MAX_BYTES = 131_072
MAX_SOURCE384_RECEIPT_BYTES = 131_072
MAX_SOURCE384_REFERENCE_BYTES = 4096


def _raw(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate context bundle key")
        result[key] = value
    return result


def _source384_reference(reference, repository):
    """A nomination selects complete receipt bytes, never a truncated receipt."""
    keys = {"schema", "output", "receipt_sha256", "receipt_bytes"}
    if (type(reference) is not dict or set(reference) != keys
            or reference["schema"] != RECEIPT_REFERENCE_SCHEMA
            or type(reference["output"]) is not str or not reference["output"]
            or any(c in reference["output"] for c in "\n\r\0")
            or type(reference["receipt_sha256"]) is not str
            or len(reference["receipt_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in reference["receipt_sha256"])
            or type(reference["receipt_bytes"]) is not int
            or not 0 < reference["receipt_bytes"] <= MAX_SOURCE384_RECEIPT_BYTES
            or len(_raw(reference)) > MAX_SOURCE384_REFERENCE_BYTES):
        raise ValueError("invalid bounded Source384 receipt reference")
    output = Path(reference["output"])
    if (not output.is_absolute() or output.as_posix() != reference["output"]
            or ".." in output.parts or output.is_relative_to(repository)
            or repository.is_relative_to(output)):
        raise ValueError("canonical external Source384 receipt reference required")
    return reference


def _read_source384_reference(reference, repository):
    from .security_autoencoder_advisor import _read

    reference = _source384_reference(reference, repository)
    raw = _read(Path(reference["output"]) / "receipt.json", MAX_SOURCE384_RECEIPT_BYTES)
    if (len(raw) != reference["receipt_bytes"]
            or hashlib.sha256(raw).hexdigest() != reference["receipt_sha256"]):
        raise ValueError("Source384 referenced receipt bytes differ")
    receipt = json.loads(raw, object_pairs_hook=_unique)
    if (type(receipt) is not dict or receipt.get("schema") not in SOURCE384_RECEIPT_SCHEMAS
            or receipt.get("output") != reference["output"]
            or receipt.get("repository") != str(repository) or _raw(receipt) != raw):
        raise ValueError("Source384 referenced receipt envelope differs")
    return receipt


def _nominate_source384_receipt(receipt, repository):
    if (type(receipt) is not dict or receipt.get("schema") not in SOURCE384_RECEIPT_SCHEMAS):
        raise ValueError("a selected Source384 context receipt is required")
    raw = _raw(receipt)
    reference = _source384_reference(dict(schema=RECEIPT_REFERENCE_SCHEMA,
        output=receipt.get("output"), receipt_sha256=hashlib.sha256(raw).hexdigest(),
        receipt_bytes=len(raw)), repository)
    if _read_source384_reference(reference, repository) != receipt:
        raise ValueError("Source384 selected receipt differs from retained bytes")
    return reference


def _source384_receipt(receipt):
    """Bound the transport envelope; the canonical consumer verifies its contents."""
    if (not isinstance(receipt, dict)
            or receipt.get("schema") not in SOURCE384_RECEIPT_SCHEMAS):
        raise ValueError("a selected Source384 context receipt is required")
    try:
        raw = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid Source384 context receipt") from exc
    if len(raw) > 32_768:
        raise ValueError("Source384 context receipt exceeds its byte bound")
    return receipt


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
    source384 = any("source384_context" in item for item in prepared)
    for item in prepared:
        if item.get("schema") != "supervisor-task-context-preparation@1":
            raise ValueError("native prepared task context required")
        tasks.append({"task_cid": item["task_cid"], "task_id": item["task_id"],
                      "metadata": _normalize(item["metadata"])})
        if source384:
            tasks[-1]["source384_context"] = _nominate_source384_receipt(item.get("source384_context"), root)
    if len({item["task_cid"] for item in tasks}) != len(tasks) or len({item["task_id"] for item in tasks}) != len(tasks):
        raise ValueError("context bundle task identities must be unique")
    payload = {"schema": SOURCE384_REFERENCE_SCHEMA if source384 else SCHEMA,
               "tasks": tasks, "completion_authority": False}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(raw) > MAX_BYTES:
        raise ValueError("context bundle exceeds its byte bound")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(raw)
    return {"artifact": output.relative_to(root).as_posix(), "sha256": hashlib.sha256(raw).hexdigest()}


def load_task_context_nomination(*, repository: Path, artifact: str, expected_sha256: str,
                                task_cid: str, task_id: str, source384_timeout_seconds: float | None = None) -> dict[str, str]:
    return load_task_context_selection(repository=repository, artifact=artifact,
        expected_sha256=expected_sha256, task_cid=task_cid, task_id=task_id,
        **({"source384_timeout_seconds": source384_timeout_seconds} if source384_timeout_seconds is not None else {}))["metadata"]


def load_task_context_selection(*, repository: Path, artifact: str, expected_sha256: str,
                               task_cid: str, task_id: str, source384_timeout_seconds: float | None = None) -> dict:
    """Revalidate task-bound nominations and any explicitly selected Source384 receipt."""
    deadline = None
    if source384_timeout_seconds is not None:
        import math
        if (type(source384_timeout_seconds) not in (int, float) or not math.isfinite(source384_timeout_seconds)
                or not 0 < source384_timeout_seconds <= 90):
            raise ValueError("Source384 nomination timeout must be finite and in (0, 90]")
        deadline = time.monotonic() + source384_timeout_seconds
    result = read_task_context_historical_selection(repository=repository, artifact=artifact,
        expected_sha256=expected_sha256, task_cid=task_cid, task_id=task_id)
    if "source384_context" in result:
        from .source384_repository_context import validate_source384_context
        options = {}
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Source384 nomination deadline expired")
            options["timeout_seconds"] = remaining
        validate_source384_context(repository=Path(repository).resolve(strict=True),
                                   expected_receipt=result["source384_context"], **options)
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Source384 nomination deadline expired")
    return result


def read_task_context_historical_selection(*, repository: Path, artifact: str, expected_sha256: str,
                                         task_cid: str, task_id: str) -> dict:
    """Read the exact sealed envelope without granting current context validity.

    Only historical publication observations use this reader directly. Planning,
    launch, warm rebinding, and worker dispatch must use the live loaders above.
    The caller must independently verify completed publication before reporting
    a historical Source384 receipt; this reader establishes envelope identity only.
    """
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

    payload = json.loads(raw, object_pairs_hook=_unique)
    if (not isinstance(payload, dict) or set(payload) != {"schema", "tasks", "completion_authority"}
            or payload["schema"] not in (SCHEMA, SOURCE384_SCHEMA, SOURCE384_REFERENCE_SCHEMA)
            or payload["completion_authority"] is not False
            or not isinstance(payload["tasks"], list) or not 1 <= len(payload["tasks"]) <= 16):
        raise ValueError("invalid context bundle")
    source384 = payload["schema"] != SCHEMA
    referenced = payload["schema"] == SOURCE384_REFERENCE_SCHEMA
    task_keys = {"task_cid", "task_id", "metadata"} | ({"source384_context"} if source384 else set())
    selected, selected_receipt, identities, aliases = None, None, set(), set()
    for task in payload["tasks"]:
        if (not isinstance(task, dict) or set(task) != task_keys
                or not isinstance(task["task_cid"], str) or not task["task_cid"]
                or not isinstance(task["task_id"], str) or not task["task_id"]
                or task["task_cid"] in identities or task["task_id"] in aliases):
            raise ValueError("invalid context bundle task identity")
        identities.add(task["task_cid"])
        aliases.add(task["task_id"])
        metadata = _normalize(task["metadata"])
        receipt = ((_source384_reference(task["source384_context"], root) if referenced
                    else _source384_receipt(task["source384_context"])) if source384 else None)
        if task["task_cid"] == task_cid or task["task_id"] == task_id:
            if task["task_cid"] != task_cid or task["task_id"] != task_id:
                raise ValueError("context nomination has a foreign task identity")
            selected = metadata
            selected_receipt = receipt
    if selected is None:
        raise ValueError("context bundle does not contain the dispatched task")
    result = {"metadata": selected}
    if source384:
        result["source384_context"] = (_read_source384_reference(selected_receipt, root)
                                       if referenced else selected_receipt)
    return result
