"""Public replay of signed inventory references without owner credentials.

The complete scan and optional conditional query are descriptive inputs. The
signed native task graph, outputs and pending checks remain authoritative.
Public replay does not reobserve a native model owner or evidence epoch.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from ..proof.formal_verification_contracts import content_identity

SCHEMA = "supervisor-codebase-inventory-context@1"
DECLARATION_SCHEMA = "supervisor-codebase-inventory-declaration@1"
WORKER_SCHEMA = "supervisor-codebase-inventory-worker-context@1"
MAX_CONTEXT_BYTES = 2 * 1024 * 1024
AUTHORITY_NAMES = frozenset({
    "source_semantics_verified", "runtime_behavior_verified", "proof_authority",
    "execution_authority", "completion_authority", "mutation_authority",
    "admission_authority", "authoritative_cache_eligible", "behavioral_satisfaction",
    "training_executed", "decoded_formulas_generated", "repository_code_executed",
    "source_execution_attested", "scan_execution_attested",
})


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _wire(value, *, max_bytes=MAX_CONTEXT_BYTES):
    stack, nodes = [(value, 0)], 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        _need(nodes <= 200_000 and depth <= 32, "bounded inventory context required")
        if type(item) is dict:
            _need(all(type(key) is str for key in item), "exact inventory context keys required")
            stack.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            stack.extend((child, depth + 1) for child in item)
        elif type(item) is int:
            _need(item.bit_length() <= 128, "bounded inventory integer required")
        else:
            _need(type(item) in {str, bool, type(None)}, "float-free inventory context required")
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                     allow_nan=False).encode()
    _need(len(raw) <= max_bytes, "inventory context byte bound exceeded")
    return raw


def validate_inventory_context(value, *, sources=None):
    """Verify the complete retained native identities; never freshness or proof."""
    from ipfs_datasets_py.logic.software_contracts.codebase_inventory_resume import (
        validate_codebase_scan_completion_refs,
    )
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
    from ..planning.codebase_inventory_evidence_context import _validate_refs

    value = json.loads(_wire(value))
    _need(type(value) is dict and set(value) == {"schema", "scan", "evidence", "authority"}
          and value["schema"] == SCHEMA, "closed inventory context profile required")
    _need(type(value["authority"]) is dict and set(value["authority"]) == AUTHORITY_NAMES
          and all(flag is False for flag in value["authority"].values()),
          "inventory context cannot acquire authority")
    scan = validate_codebase_scan_completion_refs(value["scan"])
    _need(_wire(scan) == _wire(value["scan"]), "complete native scan references differ")
    members = scan["members"]
    if sources is not None:
        _need(type(sources) is dict and set(sources) == {member["path"] for member in members},
              "signed baseline must contain the complete scan member population")
    evidence = value["evidence"]
    if evidence is not None:
        head = CodebaseHead.from_dict(scan["head"])
        evidence = _validate_refs(evidence, artifact_cid=evidence.get("artifact_cid"), head=head)
        identities = {member["source_key"]: member["entry_cid"] for member in members}
        _need(all(identities.get(row["source_key"]) == row["entry_cid"]
                  for row in evidence["entry_evidence"]), "evidence ledger belongs to a different scan member")
        _need(all(evidence["model"][key] == scan["model"][key] for key in evidence["model"]),
              "evidence query belongs to a different frozen model")
    return value


def inventory_declaration(context):
    """Bind every full ledger byte while keeping signed native task bodies small."""
    context = validate_inventory_context(context)
    scan = context["scan"]
    compact_scan = {key: scan[key] for key in ("schema", "root_cid", "completion_cid", "head", "head_cid",
        "membership_cid", "model", "pages", "coverage", "limits", "implementation", "authority")}
    compact_scan["member_paths"] = [member["path"] for member in scan["members"]]
    evidence = context["evidence"]
    if evidence is not None:
        evidence = {**{key: evidence[key] for key in ("schema", "artifact_cid", "head_cid", "scan_artifact_cid",
            "membership_cid", "model", "coverage", "query", "authority")},
            "refs_cid": content_identity(evidence), "entry_evidence_cid": content_identity(evidence["entry_evidence"])}
    return {"schema": DECLARATION_SCHEMA, "full_context_cid": content_identity(context),
        "scan": compact_scan, "evidence": evidence, "authority": dict(context["authority"])}


def validate_inventory_declaration(value, *, sources=None, context=None):
    """Strict signed reference shape, with optional full native ledger replay."""
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as resume
    from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured, validate_cid

    value = json.loads(_wire(value))
    _need(type(value) is dict and set(value) == {"schema", "full_context_cid", "scan", "evidence", "authority"}
          and value["schema"] == DECLARATION_SCHEMA, "closed signed inventory declaration required")
    validate_cid(value["full_context_cid"], codecs={"dag-json"})
    _need(type(value["authority"]) is dict and set(value["authority"]) == AUTHORITY_NAMES
          and all(flag is False for flag in value["authority"].values()), "inventory declaration cannot acquire authority")
    scan = value["scan"]
    _need(type(scan) is dict and set(scan) == {"schema", "root_cid", "completion_cid", "head", "head_cid",
        "membership_cid", "model", "pages", "coverage", "limits", "implementation", "authority", "member_paths"}
        and scan["schema"] == resume.REFS_SCHEMA, "closed completed-scan declaration required")
    head = CodebaseHead.from_dict(scan["head"])
    _need(_wire(head.to_dict()) == _wire(scan["head"]) and scan["head_cid"] == cid_for_structured(scan["head"]),
          "signed inventory head identity differs")
    for key in ("root_cid", "completion_cid", "membership_cid"):
        validate_cid(scan[key], codecs={"dag-json"})
    resume.CodebaseScanResumeLimits.from_dict(scan["limits"])
    resume._model_shape(scan["model"])
    resume._implementation_shape(scan["implementation"])
    resume._authority(scan["authority"])
    completion_body = {"schema": resume.COMPLETION_SCHEMA, "root_cid": scan["root_cid"],
        "head_cid": scan["head_cid"], "membership_cid": scan["membership_cid"],
        "model_artifact_cid": scan["model"]["artifact_cid"], "pages": scan["pages"],
        "coverage": scan["coverage"], "authority": scan["authority"]}
    resume.CodebaseScanResumeCompletion.from_dict(scan["completion_cid"], completion_body)
    names = scan["member_paths"]
    _need(type(names) is list and len(names) == scan["coverage"]["inventory_entries"] <= 1024
          and all(type(name) is str for name in names)
          and len(set(names)) == len(names), "complete unique signed member paths required")
    from . import local_planning_admission as local
    for name in names:
        local._path(name)
    _need(names == sorted(names, key=lambda name: name.encode("utf-8")), "native raw path order required")
    if sources is not None:
        _need(type(sources) is dict and set(sources) == set(names), "signed baseline must contain the complete scan member population")
    evidence = value["evidence"]
    if evidence is not None:
        _need(type(evidence) is dict and set(evidence) == {"schema", "artifact_cid", "head_cid", "scan_artifact_cid",
            "membership_cid", "model", "coverage", "query", "authority", "refs_cid", "entry_evidence_cid"},
            "closed conditional evidence declaration required")
        _need(evidence["schema"] == "codebase-inventory-evidence-advisory-refs@1", "conditional evidence schema differs")
        for key in ("artifact_cid", "scan_artifact_cid"):
            validate_cid(evidence[key], codecs={"raw"})
        for key in ("head_cid", "membership_cid", "refs_cid", "entry_evidence_cid"):
            validate_cid(evidence[key], codecs={"dag-json"})
        _need(evidence["head_cid"] == scan["head_cid"] and type(evidence["authority"]) is dict
              and set(evidence["authority"]) == AUTHORITY_NAMES
              and all(flag is False for flag in evidence["authority"].values()), "conditional declaration head or authority differs")
        model_keys = {"version_id", "variant_id", "artifact_cid", "contract_sha256", "state_sha256", "feature_space_sha256"}
        _need(type(evidence["model"]) is dict and set(evidence["model"]) == model_keys
              and all(evidence["model"][key] == scan["model"][key] for key in model_keys),
              "conditional declaration model differs")
        coverage_keys = {"inventory_entries", "inferred_rows", "evidence_entries", "evidence_matched_members",
                         "evidence_complete_absent_members", "evidence_unknown_members"}
        coverage = evidence["coverage"]
        _need(type(coverage) is dict and set(coverage) == coverage_keys
              and all(type(count) is int and count >= 0 for count in coverage.values())
              and coverage["inferred_rows"] <= coverage["inventory_entries"] <= 256
              and coverage["evidence_entries"] <= 256
              and sum(coverage[key] for key in ("evidence_matched_members", "evidence_complete_absent_members",
                    "evidence_unknown_members")) == coverage["inventory_entries"], "conditional coverage differs")
        query = evidence["query"]
        _need(type(query) is dict and set(query) == {"selector_cid", "inventory_cid", "epoch", "complete", "next_cursor", "page_cids"}
              and type(query["epoch"]) is int and 1 <= query["epoch"] < 2**63
              and type(query["complete"]) is bool and type(query["page_cids"]) is list
              and len(query["page_cids"]) <= 256, "closed exact conditional query scope required")
        for key in ("selector_cid", "inventory_cid"):
            validate_cid(query[key], codecs={"dag-json"})
        for page_cid in query["page_cids"]:
            validate_cid(page_cid, codecs={"dag-json"})
        _need((query["complete"] and query["next_cursor"] is None and query["page_cids"])
              or (not query["complete"] and (query["next_cursor"] is not None or not query["page_cids"])),
              "conditional query completeness differs")
        _need((query["complete"] and coverage["evidence_unknown_members"] == 0)
              or (not query["complete"] and coverage["evidence_complete_absent_members"] == 0),
              "partial conditional query cannot establish absence")
        if query["next_cursor"] is not None:
            from ipfs_datasets_py.duckdb_control.codebase_verification_queries import CodebaseVerificationQueryCursor
            cursor = CodebaseVerificationQueryCursor.from_dict(query["next_cursor"])
            _need(cursor.head_cid == scan["head_cid"] and cursor.inventory_cid == query["inventory_cid"]
                  and cursor.selector_cid == query["selector_cid"] and cursor.epoch == query["epoch"],
                  "conditional continuation lost its sealed query binding")
    if context is not None:
        validated = validate_inventory_context(context, sources=sources)
        _need(_wire(inventory_declaration(validated)) == _wire(value),
              "full inventory ledger differs from its signed declaration")
    return value


def build_inventory_worker_context(*, manifest_envelope, graph, receipt, task_cid,
                                   owner_identity, owner_profile_id, inventory_context):
    """Replay public signatures and the whole independently authored task graph."""
    from . import local_planning_admission as local

    profile = SimpleNamespace(identity_did=owner_identity, profile_id=owner_profile_id)
    manifest = local._verify_signature(manifest_envelope, profile)
    _need(manifest.get("schema") == local.INVENTORY_MANIFEST_SCHEMA,
          "inventory worker requires the explicit signed manifest profile")
    local._validate_local_manifest_declarations(manifest)
    native = local.PromptGoalGraph.from_dict(graph)
    signed = local._verify_signature(receipt, profile)
    expected = local._plain(local._planning_payload(native, manifest_envelope, manifest, profile, manifest["sources"]))
    _need(_wire(signed, max_bytes=local.MAX_PLANNING_RECEIPT_BYTES)
          == _wire(expected, max_bytes=local.MAX_PLANNING_RECEIPT_BYTES),
          "inventory worker signed full planning receipt differs")
    tasks = [task for task in native.tasks if task.task_cid == task_cid]
    _need(len(tasks) == 1, "inventory worker must select one signed native task")
    task = tasks[0]
    spec = next(spec for spec in manifest["tasks"] if spec["task_key"] == task.task_key)
    context = validate_inventory_context(inventory_context, sources=manifest["sources"])
    validate_inventory_declaration(manifest["codebase_inventory_context"], sources=manifest["sources"], context=context)
    scan = context["scan"]
    # Full root/complete page-chain records stay in the signed artifact. Render
    # a bounded reference projection, keeping all selected task source members.
    selected = [member for member in scan["members"] if member["path"] in spec["scope_paths"]]
    advisory = {key: scan[key] for key in ("schema", "root_cid", "completion_cid", "head", "head_cid",
        "membership_cid", "model", "pages", "coverage", "limits", "implementation", "authority")}
    advisory["selected_task_members"] = selected
    evidence = context["evidence"]
    result = {"schema": WORKER_SCHEMA, "task_cid": task_cid, "task_id": task.task_key,
        "manifest_cid": content_identity(manifest_envelope), "graph_cid": native.content_id,
        "planning_receipt_cid": content_identity(receipt),
        "codebase_inventory_context_cid": content_identity(context), "scan": advisory, "evidence": evidence,
        "administrator_task_cids": sorted(item.task_cid for item in native.tasks),
        "task_spec": spec, "dependency_task_cids": list(task.dependency_task_cids),
        "pending_requirements": expected["pending_requirements"], "pending_cid": expected["pending_cid"],
        "current_facts": [], "removed_task_cids": [], "runtime_requirements_preserved": True,
        "native_inventory_current_verified_here": False, "native_persistence_verified_here": False,
        "publication_authority": False, "scope_expansion_authority": False,
        "authority": dict(context["authority"])}
    result["context_cid"] = content_identity(result)
    _wire(result)
    return result


__all__ = ["SCHEMA", "DECLARATION_SCHEMA", "WORKER_SCHEMA", "validate_inventory_context",
           "inventory_declaration", "validate_inventory_declaration", "build_inventory_worker_context"]
