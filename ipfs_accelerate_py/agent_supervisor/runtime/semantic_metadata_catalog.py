"""Register an owner-prepared metadata view as an observational catalog.

The caller independently verifies the current native task/source and the view
codec. This bridge pins only that immutable artifact and its separately supplied
owner binding. Catalog nominations cannot establish freshness, proof, execution
or permission to omit a required fact. No database program enters model input.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from .doctor_candidate_runner import _directory, _read
from .supervisor_meta_index import SupervisorMetaIndex

SCHEMA = "supervisor-semantic-metadata-catalog@1"
OWNER_RECEIPT_SCHEMA = "supervisor-semantic-metadata-catalog-owner-receipt@1"
MAX_ARTIFACT_BYTES = 1_000_000
MAX_RECEIPT_BYTES = 65_536
MAX_EVIDENCE_IDS = 128
BINDING_FIELDS = frozenset({
    "repository_id", "tree_id", "source_scope_cid", "task_id", "task_cid",
    "task_revision", "context_cid", "native_prompt_sha256", "native_evidence_sha256",
})
DENIAL_FIELDS = frozenset({
    "freshness_authority", "proof_authority", "execution_authority",
    "completion_authority", "publication_authority", "required_fact_omission_authority",
})
RECEIPT_FIELDS = frozenset({
    "schema", "binding", "artifact_sha256", "artifact_bytes", "native_evidence_ids",
    "provider_calls", *DENIAL_FIELDS,
})


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _snapshot(value: Mapping, *, fields: frozenset[str]) -> dict:
    from ipfs_accelerate_py.cli_runtime.grok_structured_output import _bounded_json

    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("closed owner metadata fields required")
    # Capture values before any database operation; reject cycles/nonfinite data.
    return json.loads(_bounded_json(dict(value), MAX_RECEIPT_BYTES))


def _hash(value) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("exact metadata SHA256 required")


def _identifier(value) -> None:
    if (type(value) is not str or not 1 <= len(value.encode()) <= 1024
            or any(character.isspace() or not character.isprintable() for character in value)):
        raise ValueError("bounded opaque metadata identifier required")


def _binding(value: Mapping) -> dict:
    result = _snapshot(value, fields=BINDING_FIELDS)
    for key in BINDING_FIELDS - {"task_revision", "native_prompt_sha256", "native_evidence_sha256"}:
        _identifier(result[key])
    for key in ("native_prompt_sha256", "native_evidence_sha256"):
        _hash(result[key])
    if type(result["task_revision"]) is not int or not 1 <= result["task_revision"] <= 2**63 - 1:
        raise ValueError("positive native task revision required")
    return result


def _artifact_snapshot(path: Path, receipt: dict) -> tuple[int, ...]:
    parent = _directory(path.parent)
    try:
        parent_info = os.fstat(parent)
        if parent_info.st_uid != os.geteuid() or stat.S_IMODE(parent_info.st_mode) & 0o022:
            raise ValueError("owner-controlled metadata artifact directory required")
        raw, info = _read(parent, path.name)
    finally:
        os.close(parent)
    if (info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o222
            or len(raw) > MAX_ARTIFACT_BYTES or len(raw) != receipt["artifact_bytes"]
            or _sha(raw) != receipt["artifact_sha256"]):
        raise ValueError("immutable metadata artifact digest, size or owner differs")
    return tuple(getattr(info, name) for name in
                 ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"))


def _history_summary(value: Mapping) -> dict:
    if (not isinstance(value, Mapping) or value.get("completion_authority") is not False
            or value.get("authoritative") is not False
            or value.get("status") not in {"projected", "unconfigured", "unavailable", "not_requested"}):
        raise ValueError("observational DuckLake result required")
    result = {"status": value["status"], "completion_authority": False, "authoritative": False}
    # Drop error text and locators. Only closed counters or reason codes survive.
    reason = value.get("reason_code")
    if type(reason) is str and re.fullmatch(r"[a-zA-Z0-9_]{1,64}", reason):
        result["reason_code"] = reason
    for name in ("stored_catalogs", "stored_links", "stored_bindings"):
        if name in value:
            count = value[name]
            if type(count) is not int or not 0 <= count <= 2**63 - 1:
                raise ValueError("bounded DuckLake observation counter required")
            result[name] = count
    return result


def register_semantic_metadata_catalog(
    *, index: SupervisorMetaIndex, artifact: Path, owner_receipt: Mapping,
    expected_binding: Mapping, expected_native_evidence_ids: list[str] | tuple[str, ...],
    project_history: bool = True,
) -> dict:
    """Pin and register a nominated view, never authorize its model use.

    ``expected_binding`` and ``expected_native_evidence_ids`` must come from the
    independently verified current owner preparation, not the receipt being
    checked. The artifact is opaque to
    this bridge: the caller owns native view reconstruction and source freshness.
    The existing public catalog and batched-link APIs use separate transactions;
    a link failure may leave an inert catalog row but never a successful receipt.
    """
    if type(index) is not SupervisorMetaIndex or type(project_history) is not bool:
        raise TypeError("exact native meta-index and explicit projection mode required")
    receipt = _snapshot(owner_receipt, fields=RECEIPT_FIELDS)
    binding = _binding(receipt["binding"])
    expected = _binding(expected_binding)
    if (receipt["schema"] != OWNER_RECEIPT_SCHEMA or binding != expected
            or type(receipt["provider_calls"]) is not int or receipt["provider_calls"] != 0
            or any(receipt[field] is not False for field in DENIAL_FIELDS)):
        raise ValueError("owner metadata binding or authority differs")
    _hash(receipt["artifact_sha256"])
    if (type(receipt["artifact_bytes"]) is not int
            or not 1 <= receipt["artifact_bytes"] <= MAX_ARTIFACT_BYTES):
        raise ValueError("bounded metadata artifact size required")
    evidence = receipt["native_evidence_ids"]
    if type(evidence) is not list or not 1 <= len(evidence) <= MAX_EVIDENCE_IDS:
        raise ValueError("bounded native evidence identifiers required")
    for reference in evidence:
        _identifier(reference)
    if len(set(evidence)) != len(evidence):
        raise ValueError("duplicate native evidence identifier")
    if (type(expected_native_evidence_ids) not in {list, tuple}
            or not 1 <= len(expected_native_evidence_ids) <= MAX_EVIDENCE_IDS):
        raise ValueError("independently expected native evidence population required")
    expected_ids = list(expected_native_evidence_ids)
    for reference in expected_ids:
        _identifier(reference)
    if evidence != expected_ids:
        raise ValueError("native evidence population differs from verified owner input")
    path = Path(artifact)
    database = Path(index.duckdb_path)
    if (not path.is_absolute() or ".." in path.parts or not database.is_absolute()
            or ".." in database.parts or path == database or database.resolve().name == "control.duckdb"):
        raise ValueError("separate absolute owner artifact and metadata catalog required")
    before = _artifact_snapshot(path, receipt)
    from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes
    encoded_receipt = json.dumps(receipt, sort_keys=True, ensure_ascii=False,
                                separators=(",", ":"), allow_nan=False).encode()
    capsule_cid = cid_for_bytes(encoded_receipt)
    if _artifact_snapshot(path, receipt) != before:
        raise ValueError("metadata artifact changed before registration")
    catalog = index.register_catalog(kind="capsule", locator_ref=str(path),
        exclusive_owner="semantic_metadata_owner_prepared", repository_id=binding["repository_id"],
        tree_id=binding["tree_id"], project=False)
    if catalog.get("attach_permitted") is not False or catalog.get("completion_authority") is not False:
        raise ValueError("metadata catalog nomination cannot grant attachment or completion")
    records = [{"subject_kind": kind, "subject_ref": reference,
                "catalog_id": catalog["catalog_id"], "record_kind": record_kind,
                "record_ref": capsule_cid, "capsule_cid": capsule_cid}
               for kind, reference, record_kind in (
                   ("task_id", binding["task_id"], "semantic_metadata_task"),
                   ("record_cid", binding["task_cid"], "semantic_metadata_native_task"),
                   ("tree_id", binding["tree_id"], "semantic_metadata_tree"),
                   ("content_cid", binding["source_scope_cid"], "semantic_metadata_source_scope"),
                   ("capsule_cid", binding["context_cid"], "semantic_metadata_native_context"),
                   *(("record_cid", reference, "semantic_metadata_native_evidence") for reference in evidence),
               )]
    links = index.link_identities(records, project=False)
    if len(links) != len(records) or any(link.get("completion_authority") is not False for link in links):
        raise ValueError("metadata catalog links differ from observational nominations")
    if _artifact_snapshot(path, receipt) != before:
        raise ValueError("metadata artifact changed during registration")
    history = index.project_ducklake() if project_history else {
        "status": "not_requested", "authoritative": False, "completion_authority": False,
    }
    history_summary = _history_summary(history)
    if _artifact_snapshot(path, receipt) != before:
        raise ValueError("metadata artifact changed during history projection")
    return {"schema": SCHEMA, "status": "registered_observational", "capsule_cid": capsule_cid,
        "catalog_id": catalog["catalog_id"], "artifact_sha256": receipt["artifact_sha256"],
        "artifact_bytes": receipt["artifact_bytes"], "owner_receipt_sha256": _sha(encoded_receipt),
        "link_ids": [link["link_id"] for link in links], "link_count": len(links),
        "ducklake": history_summary, "provider_calls": 0,
        "current_source_verified_by_bridge": False, "model_use_authorized": False,
        **{field: False for field in DENIAL_FIELDS}}
