"""Focused DCR-031 tests for non-proving MCP graph obligations."""

from __future__ import annotations

from typing import Any

from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_graph import (
    MCP_CONTRACT_GRAPH_INTERFACE,
    MCP_CONTRACT_GRAPH_SCHEMA,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    canonical_json_bytes,
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.proof.ir_integration import (
    DatasetsLogicIrDisposition,
    DatasetsLogicIrResult,
)
from ipfs_accelerate_py.agent_supervisor.proof.mcp_contract_obligations import (
    McpObligationFragment,
    McpObligationDisposition,
    McpObligationFamily,
    McpObligationUnsupportedReason,
    compile_dcr031_mcp_contract_obligations,
)


_RELATIONS = (
    "expects_descriptor",
    "binds_orb_idl",
    "defines_method_schema",
    "binds_mediator_route",
    "routes_to_observed_dispatcher",
    "dispatches_to_handler",
    "performs_effect",
    "emits_receipt_runtime_identity",
)


def _graph(*, blockers: list[dict[str, str]] | None = None) -> dict[str, Any]:
    nodes = [{"id": "node:" + str(index)} for index in range(len(_RELATIONS) + 1)]
    body = {
        "schema": MCP_CONTRACT_GRAPH_SCHEMA,
        "interface": MCP_CONTRACT_GRAPH_INTERFACE,
        "authoritative": False,
        "nodes": nodes,
        "edges": [
            {
                "id": "edge:" + relation,
                "source": "node:" + str(index),
                "target": "node:" + str(index + 1),
                "relation": relation,
                "authority_class": "observed_provider"
                if relation in {"performs_effect", "dispatches_to_handler"}
                else "registration",
            }
            for index, relation in enumerate(_RELATIONS)
        ],
        "blockers": blockers or [],
    }
    return {
        **body,
        "graph_cid": content_identity(body),
        "canonical_bytes": canonical_json_bytes(body).decode("utf-8"),
    }


def _candidate(graph_cid: str) -> DatasetsLogicIrResult:
    return DatasetsLogicIrResult(
        disposition=DatasetsLogicIrDisposition.NORMALIZED,
        reason_codes=("candidate_context_only", "zero_model_calls"),
        input_cids=tuple(sorted(("bafy-finding", "bafy-forest", "bafy-source", graph_cid))),
        module_binding={
            "content_digest": "module:sha256:fixture",
            "module": "ipfs_datasets_py.logic.ir_core.identity",
            "origin": "/fixture/identity.py",
            "version": "1.0.0",
        },
        normalized_ir={
            "input_schema": "ipfs_accelerate_py/agent-supervisor/datasets-logic-ir-input@1",
            "integration_status": "candidate_context_only",
            "identity": {
                "cid": "bafy-logic",
                "digest": "sha256:logic",
                "profile": "ir-canonical-identity-v1",
                "logic_ir_interface": "LogicIR@1",
            },
        },
    )


def test_compiler_binds_every_dcr021_edge_and_preserves_semantics() -> None:
    graph = _graph()
    result = compile_dcr031_mcp_contract_obligations(_candidate(graph["graph_cid"]), graph=graph)

    assert result.disposition is McpObligationDisposition.OPEN
    assert result.to_dict()["authoritative"] is False
    assert result.to_dict()["completion_authoritative"] is False
    assert result.to_dict()["model_call_count"] == 0
    by_edge = {item.edge_id: item for item in result.obligations if item.edge_id}
    assert set(by_edge) == {"edge:" + relation for relation in _RELATIONS}
    effect = by_edge["edge:performs_effect"]
    assert effect.direction == ("node:6", "node:7")
    assert effect.temporal_authority == "observed_provider"
    assert effect.effect_semantics == "observed_provider_effect_edge"
    assert effect.graph_cid == graph["graph_cid"]
    assert graph["graph_cid"] in effect.cid_bindings
    assert all(item.disposition is McpObligationDisposition.OPEN for item in by_edge.values())


def test_compiler_covers_profiles_a_to_f_with_typed_missing_backend_reasons() -> None:
    graph = _graph()
    result = compile_dcr031_mcp_contract_obligations(_candidate(graph["graph_cid"]), graph=graph)
    by_family = {}
    for obligation in result.obligations:
        by_family.setdefault(obligation.family, []).append(obligation)

    assert set(McpObligationFamily).issubset(by_family)
    assert any(
        item.family is McpObligationFamily.PROFILE_A
        and item.disposition is McpObligationDisposition.OPEN
        for item in by_family[McpObligationFamily.PROFILE_A]
    )
    for family in (
        McpObligationFamily.PROFILE_B,
        McpObligationFamily.PROFILE_C,
        McpObligationFamily.PROFILE_D,
        McpObligationFamily.PROFILE_E,
        McpObligationFamily.PROFILE_F,
    ):
        item = by_family[family][0]
        assert item.disposition is McpObligationDisposition.UNSUPPORTED
        assert (
            item.unsupported_reason
            is McpObligationUnsupportedReason.PROFILE_DECLARATION_DRAFT_NON_AUTHORITATIVE
        )
        assert item.profile_declaration["status"] == "draft_non_normative"
    assert by_family[McpObligationFamily.PROFILE_B][0].fragment is McpObligationFragment.CID
    assert by_family[McpObligationFamily.PROFILE_C][0].fragment is McpObligationFragment.DELEGATION
    assert by_family[McpObligationFamily.PROFILE_D][0].fragment is McpObligationFragment.POLICY
    assert by_family[McpObligationFamily.PROFILE_E][0].fragment is McpObligationFragment.TRANSPORT
    assert by_family[McpObligationFamily.PROFILE_F][0].fragment is McpObligationFragment.EVENT_DAG
    negotiation = by_family[McpObligationFamily.NEGOTIATION][0]
    assert (
        negotiation.unsupported_reason is McpObligationUnsupportedReason.GRAPH_SEMANTIC_EDGE_ABSENT
    )


def test_compiler_fails_closed_for_raw_candidate_or_tampered_graph() -> None:
    graph = _graph()
    raw = compile_dcr031_mcp_contract_obligations(
        {"disposition": "normalized_candidate"}, graph=graph
    )
    assert raw.disposition is McpObligationDisposition.INTEGRATION_PENDING
    assert raw.reason is McpObligationUnsupportedReason.DCR030_CANDIDATE_INVALID

    tampered = dict(graph)
    tampered["edges"] = []
    blocked = compile_dcr031_mcp_contract_obligations(
        _candidate(graph["graph_cid"]), graph=tampered
    )
    assert blocked.disposition is McpObligationDisposition.INTEGRATION_PENDING
    assert blocked.reason is McpObligationUnsupportedReason.DCR021_GRAPH_INVALID


def test_compiler_rejects_a_canonical_but_blocked_dcr021_graph() -> None:
    graph = _graph(blockers=[{"kind": "mandatory_consumer_unresolved"}])

    result = compile_dcr031_mcp_contract_obligations(_candidate(graph["graph_cid"]), graph=graph)

    assert result.disposition is McpObligationDisposition.INTEGRATION_PENDING
    assert result.reason is McpObligationUnsupportedReason.DCR021_GRAPH_BLOCKED
    assert result.obligations == ()
