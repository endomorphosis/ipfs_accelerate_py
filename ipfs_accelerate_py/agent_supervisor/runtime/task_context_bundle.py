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
FORMALIZATION_REFERENCE_SCHEMA = "supervisor-task-context-nominations@4"
LEARNED_FORMALIZATION_REFERENCE_SCHEMA = "supervisor-task-context-nominations@5"
RECEIPT_REFERENCE_SCHEMA = "supervisor-source384-receipt-reference@1"
NUMERICAL_REFERENCE_SCHEMA = "supervisor-numerical-receipt-reference@1"
LEARNED_NUMERICAL_REFERENCE_SCHEMA = "supervisor-learned-numerical-receipt-reference@1"
WORKER_EVIDENCE_SCHEMA = "supervisor-task-formalization-worker-evidence@1"
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
MAX_NUMERICAL_RECEIPT_BYTES = 131_072
MAX_WORKER_EVIDENCE_BYTES = 32_768
FORMALIZATION_KEYS = ("source384_context", "numerical_formalization", "numerical_selection_384")


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


def _numerical_reference(reference, repository):
    # Both nominations select complete, independently validated external
    # receipts. Reuse the path/digest/size boundary without sharing semantics.
    if type(reference) is not dict or reference.get("schema") != NUMERICAL_REFERENCE_SCHEMA:
        raise ValueError("invalid bounded numerical receipt reference")
    _source384_reference({**reference, "schema": RECEIPT_REFERENCE_SCHEMA}, repository)
    return reference


def _read_numerical_reference(reference, repository):
    from .security_autoencoder_advisor import _read
    from .numerical_formalization_context import SCHEMA as numerical_schema

    reference = _numerical_reference(reference, repository)
    raw = _read(Path(reference["output"]) / "receipt.json", MAX_NUMERICAL_RECEIPT_BYTES)
    if (len(raw) != reference["receipt_bytes"]
            or hashlib.sha256(raw).hexdigest() != reference["receipt_sha256"]):
        raise ValueError("numerical referenced receipt bytes differ")
    receipt = json.loads(raw, object_pairs_hook=_unique)
    if (type(receipt) is not dict or receipt.get("schema") != numerical_schema
            or "receipt_sha256" in receipt or receipt.get("output") != reference["output"]
            or receipt.get("repository") != str(repository) or _raw(receipt) != raw):
        raise ValueError("numerical referenced receipt envelope differs")
    return {**receipt, "receipt_sha256": reference["receipt_sha256"]}


def _nominate_numerical_receipt(receipt, repository):
    if type(receipt) is not dict:
        raise ValueError("a selected numerical context receipt is required")
    raw = _raw({key: value for key, value in receipt.items() if key != "receipt_sha256"})
    reference = _numerical_reference(dict(schema=NUMERICAL_REFERENCE_SCHEMA,
        output=receipt.get("output"), receipt_sha256=receipt.get("receipt_sha256"),
        receipt_bytes=len(raw)), repository)
    if _read_numerical_reference(reference, repository) != receipt:
        raise ValueError("numerical selected receipt differs from retained bytes")
    return reference


def _learned_numerical_reference(reference, repository):
    if type(reference) is not dict or reference.get("schema") != LEARNED_NUMERICAL_REFERENCE_SCHEMA:
        raise ValueError("invalid bounded learned numerical receipt reference")
    _source384_reference({**reference, "schema": RECEIPT_REFERENCE_SCHEMA}, repository)
    return reference


def _read_learned_numerical_reference(reference, repository):
    from .security_autoencoder_advisor import _read
    from .numerical_selection_context import SCHEMA as learned_schema

    reference = _learned_numerical_reference(reference, repository)
    raw = _read(Path(reference["output"]) / "receipt.json", MAX_NUMERICAL_RECEIPT_BYTES)
    if (len(raw) != reference["receipt_bytes"]
            or hashlib.sha256(raw).hexdigest() != reference["receipt_sha256"]):
        raise ValueError("learned numerical referenced receipt bytes differ")
    receipt = json.loads(raw, object_pairs_hook=_unique)
    if (type(receipt) is not dict or receipt.get("schema") != learned_schema
            or "receipt_sha256" in receipt or receipt.get("output") != reference["output"]
            or receipt.get("repository") != str(repository) or _raw(receipt) != raw):
        raise ValueError("learned numerical referenced receipt envelope differs")
    return {**receipt, "receipt_sha256": reference["receipt_sha256"]}


def _nominate_learned_numerical_receipt(receipt, repository):
    if type(receipt) is not dict:
        raise ValueError("a selected learned numerical context receipt is required")
    raw = _raw({key: value for key, value in receipt.items() if key != "receipt_sha256"})
    reference = _learned_numerical_reference(dict(schema=LEARNED_NUMERICAL_REFERENCE_SCHEMA,
        output=receipt.get("output"), receipt_sha256=receipt.get("receipt_sha256"),
        receipt_bytes=len(raw)), repository)
    if _read_learned_numerical_reference(reference, repository) != receipt:
        raise ValueError("learned numerical selected receipt differs from retained bytes")
    return reference


def _worker_evidence(selection, *, repository, artifact, expected_sha256, task_cid, task_id):
    """Project already verified owner summaries, never accept caller formulae."""
    contexts = []
    for key, kind, nominate in (
        ("source384_context", "source384-candidate-context", _nominate_source384_receipt),
        ("numerical_formalization", "numerical-formalization-context", _nominate_numerical_receipt),
        ("numerical_selection_384", "learned-numerical-selection-context", _nominate_learned_numerical_receipt),
    ):
        if key not in selection:
            continue
        receipt = selection[key]
        summary = receipt.get("summary")
        if type(summary) is not dict or len(_raw(summary)) > 8192:
            raise ValueError("bounded validated formalization summary required")
        contexts.append(dict(kind=kind, receipt_reference=nominate(receipt, repository),
            source_hashes_sha256=hashlib.sha256(_raw(receipt["source_hashes"])).hexdigest(),
            summary_sha256=hashlib.sha256(_raw(summary)).hexdigest(), summary=summary))
    if not contexts:
        return None
    evidence = dict(schema=WORKER_EVIDENCE_SCHEMA, repository=str(repository),
        task_cid=task_cid, task_id=task_id,
        context_bundle=dict(artifact=artifact, sha256=expected_sha256), contexts=contexts,
        proof_authority=False, formalization_authority=False, execution_authority=False,
        completion_authority=False, planning_authority=False, dispatch_authority=False,
        source_semantics_verified=False, whole_program_semantics_verified=False,
        source_obligations_discharged=0)
    raw = _raw(evidence)
    if len(raw) > MAX_WORKER_EVIDENCE_BYTES:
        raise ValueError("formalization worker evidence exceeds its byte bound")
    return json.loads(raw)


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
    formalization = any("numerical_formalization" in item for item in prepared)
    learned = any("numerical_selection_384" in item for item in prepared)
    for item in prepared:
        if item.get("schema") != "supervisor-task-context-preparation@1":
            raise ValueError("native prepared task context required")
        tasks.append({"task_cid": item["task_cid"], "task_id": item["task_id"],
                      "metadata": _normalize(item["metadata"])})
        if source384 and (not (formalization or learned) or "source384_context" in item):
            tasks[-1]["source384_context"] = _nominate_source384_receipt(item.get("source384_context"), root)
        if formalization and "numerical_formalization" in item:
            tasks[-1]["numerical_formalization"] = _nominate_numerical_receipt(item["numerical_formalization"], root)
        if learned and "numerical_selection_384" in item:
            tasks[-1]["numerical_selection_384"] = _nominate_learned_numerical_receipt(item["numerical_selection_384"], root)
    if len({item["task_cid"] for item in tasks}) != len(tasks) or len({item["task_id"] for item in tasks}) != len(tasks):
        raise ValueError("context bundle task identities must be unique")
    payload = {"schema": LEARNED_FORMALIZATION_REFERENCE_SCHEMA if learned else
               FORMALIZATION_REFERENCE_SCHEMA if formalization else
               SOURCE384_REFERENCE_SCHEMA if source384 else SCHEMA,
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
    """Validate selected owner receipts and expose their task-bound worker advice.

    ``metadata`` retains the legacy nomination API. ``worker_evidence`` exists
    only after current receipt/source validation and contains bounded owner
    summaries, never proof, scheduling, or completion authority.
    """
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
    root = Path(repository).resolve(strict=True)
    if "numerical_formalization" in result:
        from .numerical_formalization_context import validate_numerical_context
        validate_numerical_context(repository=root, expected_receipt=result["numerical_formalization"])
    if "numerical_selection_384" in result:
        from .numerical_selection_context import validate_numerical_selection_context
        validate_numerical_selection_context(repository=root, expected_receipt=result["numerical_selection_384"])
    sources = {}
    for key in FORMALIZATION_KEYS:
        if key in result:
            for path, digest in result[key]["source_hashes"].items():
                if path in sources and sources[path] != digest:
                    raise ValueError("selected formalization contexts disagree on source bytes")
                sources[path] = digest
    # The second owner may take time to replay. Recheck signed source bytes
    # after all owner validations, without re-running either neural inference
    # or expensive projection/metadata validation.
    for key, owner in (("source384_context", "source384"), ("numerical_formalization", "numerical"),
                       ("numerical_selection_384", "learned_numerical")):
        if key in result:
            if owner == "source384":
                from .source384_repository_context import _sources
            elif owner == "numerical":
                from .numerical_formalization_context import _sources
            else:
                from .numerical_selection_context import _sources
            _sources(root, result[key]["source_hashes"])
    if read_task_context_historical_selection(repository=root, artifact=artifact,
            expected_sha256=expected_sha256, task_cid=task_cid, task_id=task_id) != result:
        raise ValueError("selected context changed during worker validation")
    evidence = _worker_evidence(result, repository=root, artifact=artifact,
        expected_sha256=expected_sha256, task_cid=task_cid, task_id=task_id)
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Source384 nomination deadline expired")
    if evidence is not None:
        result["worker_evidence"] = evidence
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
            or payload["schema"] not in (SCHEMA, SOURCE384_SCHEMA, SOURCE384_REFERENCE_SCHEMA,
                                         FORMALIZATION_REFERENCE_SCHEMA, LEARNED_FORMALIZATION_REFERENCE_SCHEMA)
            or payload["completion_authority"] is not False
            or not isinstance(payload["tasks"], list) or not 1 <= len(payload["tasks"]) <= 16):
        raise ValueError("invalid context bundle")
    formalization = payload["schema"] in (FORMALIZATION_REFERENCE_SCHEMA, LEARNED_FORMALIZATION_REFERENCE_SCHEMA)
    learned = payload["schema"] == LEARNED_FORMALIZATION_REFERENCE_SCHEMA
    source384 = payload["schema"] in (SOURCE384_SCHEMA, SOURCE384_REFERENCE_SCHEMA)
    referenced = payload["schema"] in (SOURCE384_REFERENCE_SCHEMA, FORMALIZATION_REFERENCE_SCHEMA,
                                       LEARNED_FORMALIZATION_REFERENCE_SCHEMA)
    task_keys = {"task_cid", "task_id", "metadata"} | ({"source384_context"} if source384 else set())
    selected, selected_receipt, selected_numerical, identities, aliases = None, None, None, set(), set()
    selected_learned = None
    for task in payload["tasks"]:
        allowed_families = FORMALIZATION_KEYS if learned else FORMALIZATION_KEYS[:2]
        expected_keys = task_keys | ({key for key in allowed_families
                                     if key in task} if formalization and isinstance(task, dict) else set())
        if (not isinstance(task, dict) or set(task) != expected_keys
                or not isinstance(task["task_cid"], str) or not task["task_cid"]
                or not isinstance(task["task_id"], str) or not task["task_id"]
                or task["task_cid"] in identities or task["task_id"] in aliases):
            raise ValueError("invalid context bundle task identity")
        identities.add(task["task_cid"])
        aliases.add(task["task_id"])
        metadata = _normalize(task["metadata"])
        receipt = ((_source384_reference(task["source384_context"], root) if referenced
                    else _source384_receipt(task["source384_context"])) if "source384_context" in task else None)
        numerical = (_numerical_reference(task["numerical_formalization"], root)
                     if "numerical_formalization" in task else None)
        learned_reference = (_learned_numerical_reference(task["numerical_selection_384"], root)
                             if "numerical_selection_384" in task else None)
        if task["task_cid"] == task_cid or task["task_id"] == task_id:
            if task["task_cid"] != task_cid or task["task_id"] != task_id:
                raise ValueError("context nomination has a foreign task identity")
            selected = metadata
            selected_receipt = receipt
            selected_numerical = numerical
            selected_learned = learned_reference
    if selected is None:
        raise ValueError("context bundle does not contain the dispatched task")
    result = {"metadata": selected}
    if selected_receipt is not None:
        result["source384_context"] = (_read_source384_reference(selected_receipt, root)
                                       if referenced else selected_receipt)
    if selected_numerical is not None:
        result["numerical_formalization"] = _read_numerical_reference(selected_numerical, root)
    if selected_learned is not None:
        result["numerical_selection_384"] = _read_learned_numerical_reference(selected_learned, root)
    return result
