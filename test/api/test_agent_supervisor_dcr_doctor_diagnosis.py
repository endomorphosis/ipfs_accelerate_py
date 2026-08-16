"""Focused DCR-051 earliest-edge diagnosis tests."""

from __future__ import annotations

import hashlib
from typing import Any

from ipfs_accelerate_py.agent_supervisor.control.default_doctor_factory import (
    DcrDoctorCompositionResult,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    canonical_json_bytes,
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.sca_doctor_bridge import (
    DoctorFindingDisposition,
    DoctorSourceSlice,
    diagnose_earliest_dcr_edge,
)


def _composition() -> DcrDoctorCompositionResult:
    return DcrDoctorCompositionResult(
        disposition="integration_pending",
        reason_codes=("live_witness_current",),
        identities={"checkout_forest": "sha256:forest"},
        binding_complete=True,
    )


def _graph(
    relations: tuple[str, ...] = ("defines_method_schema", "performs_effect"),
) -> dict[str, Any]:
    nodes = [{"id": f"n{index}"} for index in range(len(relations) + 1)]
    body = {
        "schema": "ipfs_accelerate_py/agent-supervisor/mcp-contract-graph@1",
        "interface": "McpContractGraph@1",
        "authoritative": False,
        "nodes": nodes,
        "edges": [
            {
                "id": f"e{index}",
                "source": f"n{index}",
                "target": f"n{index + 1}",
                "relation": relation,
            }
            for index, relation in enumerate(relations)
        ],
        "blockers": [],
    }
    return {
        **body,
        "graph_cid": content_identity(body),
        "canonical_bytes": canonical_json_bytes(body).decode("utf-8"),
    }


def _report(graph: dict[str, Any], findings: list[dict[str, Any]]) -> dict[str, Any]:
    body = {
        "schema": "ipfs_accelerate_py/agent-supervisor/mcp-contract-mismatch-analysis@1",
        "interface": "McpContractMismatchAnalysis@1",
        "authoritative": False,
        "graph_cid": graph["graph_cid"],
        "semantic_roots": {"semantic": "sha256:semantic"},
        "snapshot_roots": {"forest_id": "sha256:forest"},
        "dcr023_current_valid": True,
        "production_readiness": "nonpassing",
        "findings": findings,
    }
    return {**body, "findings_cid": content_identity(body)}


def _finding(edge_id: str, mismatch_class: str = "schema") -> dict[str, Any]:
    return {
        "edge_id": edge_id,
        "mismatch_class": mismatch_class,
        "status": "missing",
        "semantic_key": {
            "package": "accelerate",
            "operation": "logic.cec_prove",
            "direction": "request",
            "schema": "LogicRequest@1",
            "profile": "base",
            "transport": "loopback_mcp",
        },
    }


def _source(edge_id: str, source: bytes = b"def handler():\n    pass\n") -> DoctorSourceSlice:
    start, end = source.index(b"pass"), source.index(b"pass") + 4
    return DoctorSourceSlice(
        edge_id=edge_id,
        root_owner="ipfs_accelerate",
        relative_path="agent_supervisor/handler.py",
        source_bytes=source,
        source_sha256="sha256:" + hashlib.sha256(source).hexdigest(),
        span_start=start,
        span_end=end,
        span_sha256="sha256:" + hashlib.sha256(source[start:end]).hexdigest(),
    )


def _diagnose(
    monkeypatch: Any, graph: dict[str, Any], report: dict[str, Any], source: DoctorSourceSlice
):
    composition = _composition()
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.control.default_doctor_factory.inspect_dcr_doctor_composition",
        lambda binding: composition,
    )
    return diagnose_earliest_dcr_edge(
        composition_result=composition,
        composition_binding={"checkout_forest": {"binding": {"forest_id": "sha256:forest"}}},
        graph=graph,
        mismatch_report=report,
        source_slices=(source,),
    )


def test_earliest_relation_is_deterministic_and_relocation_stable(monkeypatch: Any) -> None:
    graph = _graph()
    report = _report(graph, [_finding("e1", "implementation"), _finding("e0", "schema")])
    result = _diagnose(monkeypatch, graph, report, _source("e0"))
    relocated = _diagnose(
        monkeypatch, graph, report, _source("e0", b"\n\ndef handler():\n    pass\n")
    )

    assert result.disposition is DoctorFindingDisposition.FINDING
    assert result.edge_id == "e0"
    assert result.finding_id == relocated.finding_id
    assert result.to_dict()["mutation_authorized"] is False


def test_tie_stale_epoch_root_and_source_all_fail_closed(monkeypatch: Any) -> None:
    graph = _graph(("expects_descriptor", "expects_descriptor"))
    tie = _diagnose(
        monkeypatch, graph, _report(graph, [_finding("e0"), _finding("e1")]), _source("e0")
    )
    assert tie.disposition is DoctorFindingDisposition.ABSTAINED

    report = _report(_graph(("defines_method_schema",)), [_finding("e0")])
    stale = {**report, "findings_cid": "sha256:stale"}
    assert (
        _diagnose(monkeypatch, _graph(("defines_method_schema",)), stale, _source("e0")).disposition
        is DoctorFindingDisposition.DEFERRED
    )

    root_drift = _report(_graph(("defines_method_schema",)), [_finding("e0")])
    root_drift["snapshot_roots"] = {"forest_id": "sha256:other"}
    root_drift["findings_cid"] = content_identity(
        {key: value for key, value in root_drift.items() if key != "findings_cid"}
    )
    assert (
        _diagnose(
            monkeypatch, _graph(("defines_method_schema",)), root_drift, _source("e0")
        ).disposition
        is DoctorFindingDisposition.DEFERRED
    )

    unsupported = _diagnose(
        monkeypatch,
        _graph(("defines_method_schema",)),
        _report(_graph(("defines_method_schema",)), [_finding("e0")]),
        _source("e1"),
    )
    assert unsupported.disposition is DoctorFindingDisposition.ABSTAINED


def test_synthetic_dcr050_result_cannot_cross_reinspection(monkeypatch: Any) -> None:
    graph = _graph(("defines_method_schema",))
    current = _composition()
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.control.default_doctor_factory.inspect_dcr_doctor_composition",
        lambda binding: current,
    )
    synthetic = DcrDoctorCompositionResult(
        disposition="integration_pending",
        reason_codes=(),
        identities={"checkout_forest": "sha256:forged"},
        binding_complete=True,
    )
    result = diagnose_earliest_dcr_edge(
        composition_result=synthetic,
        composition_binding={"checkout_forest": {"binding": {"forest_id": "sha256:forest"}}},
        graph=graph,
        mismatch_report=_report(graph, [_finding("e0")]),
        source_slices=(_source("e0"),),
    )
    assert result.disposition is DoctorFindingDisposition.DEFERRED


def test_transitional_dcr050_projection_cannot_be_diagnostic_evidence(monkeypatch: Any) -> None:
    graph = _graph(("defines_method_schema",))
    transitional = DcrDoctorCompositionResult(
        disposition="integration_pending",
        reason_codes=("transitional_self_attested_bindings_non_live",),
        identities={"checkout_forest": "sha256:forest"},
        binding_complete=True,
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.control.default_doctor_factory.inspect_dcr_doctor_composition",
        lambda binding: transitional,
    )
    result = diagnose_earliest_dcr_edge(
        composition_result=transitional,
        composition_binding={"checkout_forest": {"binding": {"forest_id": "sha256:forest"}}},
        graph=graph,
        mismatch_report=_report(graph, [_finding("e0")]),
        source_slices=(_source("e0"),),
    )
    assert result.disposition is DoctorFindingDisposition.DEFERRED
