"""Focused DCR-052 non-authoritative transform-selection tests."""

from __future__ import annotations

from typing import Any

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.registry import (
    OperatorDescriptor,
    OperatorRegistry,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.sca_doctor_bridge import (
    DoctorFinding,
    DoctorFindingDisposition,
    DoctorTransformDisposition,
    select_deterministic_doctor_transform,
)


def _registry() -> OperatorRegistry:
    descriptor = OperatorDescriptor.from_mapping(
        {
            "operator_id": "doctor.schema-alias",
            "kind": "replace_exact_bytes",
            "input_schema": {
                "type": "object",
                "required": ["source_digest"],
                "properties": {"source_digest": "sha256"},
                "additional_properties": False,
            },
            "owner_root": "ipfs_accelerate",
            "write_scope": ["agent_supervisor/handler.py"],
            "before_predicates": ["schema_anchor"],
            "after_predicates": ["schema_registered"],
            "applicability_proofs": ["dcr052_proof"],
            "preview": {"kind": "metadata_only", "fields": ["source_digest"]},
            "inverse": {"kind": "restore_exact_before_bytes", "binding": "before_bytes"},
            "validation_commands": [["pytest", "handler.py"]],
        }
    )
    return OperatorRegistry(
        [descriptor], reviewed_manifest={descriptor.operator_id: descriptor.descriptor_id}
    )


def _finding() -> DoctorFinding:
    semantic_key = {
        "package": "accelerate",
        "operation": "logic.cec_prove",
        "direction": "request",
        "schema": "LogicRequest@1",
        "profile": "base",
        "transport": "loopback_mcp",
    }
    finding_id = content_identity(
        {
            "edge_id": "edge.schema",
            "semantic_key": semantic_key,
            "mismatch_class": "schema",
            "forest_id": "sha256:forest",
            "graph_cid": "sha256:graph",
            "epoch_cid": "sha256:epoch",
        }
    )
    return DoctorFinding(
        disposition=DoctorFindingDisposition.FINDING,
        reason_code="earliest_topological_nonpassing_edge",
        finding_id=finding_id,
        edge_id="edge.schema",
        semantic_key=semantic_key,
        mismatch_class="schema",
        forest_id="sha256:forest",
        graph_cid="sha256:graph",
        epoch_cid="sha256:epoch",
        findings_cid="sha256:epoch",
        evidence_cids=("sha256:source", "sha256:epoch", "sha256:edge"),
    )


def _cid(value: dict[str, Any], field: str) -> dict[str, Any]:
    return {**value, field: content_identity(value)}


def _inputs(registry: OperatorRegistry) -> dict[str, Any]:
    finding = _finding()
    descriptor = registry.enumerate()[0]
    roots = _cid(
        {
            "forest_id": finding.forest_id,
            "graph_cid": finding.graph_cid,
            "epoch_cid": finding.epoch_cid,
            "findings_cid": finding.findings_cid,
        },
        "roots_cid",
    )
    applicability = _cid(
        {
            "operator_ids": (descriptor.operator_id,),
            "finding_id": finding.finding_id,
            "edge_id": finding.edge_id,
            "owner_root": descriptor.owner_root,
            "relative_path": "agent_supervisor/handler.py",
            "source_slice_cid": "sha256:source",
            "predicate_cids": {"schema_anchor": "sha256:predicate"},
        },
        "applicability_cid",
    )
    impact = _cid(
        {
            "finding_id": finding.finding_id,
            "edge_id": finding.edge_id,
            "roots_cid": roots["roots_cid"],
            "impact_id": "sha256:impact",
        },
        "impact_cid",
    )
    proof = _cid(
        {
            "finding_id": finding.finding_id,
            "applicability_cid": applicability["applicability_cid"],
            "impact_cid": impact["impact_cid"],
            "proof_id": "sha256:proof",
        },
        "proof_cid",
    )
    report = registry.report()
    pin = _cid(
        {
            "issuer": "reviewed_external_policy",
            "policy_id": "policy.dcr052",
            "policy_digest": "sha256:policy",
            "registry_cid": report["registry_cid"],
            "descriptor_id": descriptor.descriptor_id,
            "finding_class": finding.mismatch_class,
        },
        "pin_cid",
    )
    return {
        "finding": finding,
        "registry": registry,
        "reviewed_registry_cid": report["registry_cid"],
        "policy_pin": pin,
        "applicability": applicability,
        "proof": proof,
        "impact": impact,
        "roots": roots,
    }


def test_exact_policy_pinned_registry_selects_one_non_authoritative_transform() -> None:
    result = select_deterministic_doctor_transform(**_inputs(_registry()))

    assert result.disposition is DoctorTransformDisposition.TRANSFORM
    assert result.operator_id == "doctor.schema-alias"
    report = result.to_dict()
    assert report["mutation_authorized"] is False
    assert report["model_call_count"] == report["provider_call_count"] == 0


def test_ambiguous_absent_stale_and_raw_inputs_fail_closed() -> None:
    absent = _inputs(_registry())
    absent["applicability"] = _cid(
        {key: value for key, value in absent["applicability"].items() if key != "applicability_cid"}
        | {"operator_ids": ("missing",)},
        "applicability_cid",
    )
    absent["proof"] = _cid(
        {
            "finding_id": absent["finding"].finding_id,
            "applicability_cid": absent["applicability"]["applicability_cid"],
            "impact_cid": absent["impact"]["impact_cid"],
            "proof_id": "sha256:proof",
        },
        "proof_cid",
    )
    assert (
        select_deterministic_doctor_transform(**absent).disposition
        is DoctorTransformDisposition.ABSTAINED
    )

    stale = _inputs(_registry())
    stale["roots"] = {**stale["roots"], "forest_id": "sha256:other"}
    assert (
        select_deterministic_doctor_transform(**stale).disposition
        is DoctorTransformDisposition.DEFERRED
    )

    raw = _inputs(_registry())
    raw["proof"] = {**raw["proof"], "source": "do not accept source text"}
    assert (
        select_deterministic_doctor_transform(**raw).disposition
        is DoctorTransformDisposition.REJECTED
    )

    deferred = _inputs(_registry())
    deferred["finding"] = DoctorFinding(DoctorFindingDisposition.DEFERRED)
    assert (
        select_deterministic_doctor_transform(**deferred).disposition
        is DoctorTransformDisposition.DEFERRED
    )
