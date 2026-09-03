"""PCPR-081: Accelerate-owned generic MCP-client demonstration."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.assurance.generic_mcp_client import (
    AUTHORIZED_PATH_PREFIXES,
    CLIENT_OPERATIONS,
    CLOSED_RELEASE_OUTCOMES,
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID,
    ESCALATION_ORDER,
    FORBIDDEN_CLIENT_OPERATIONS,
    FORBIDDEN_MCP_TOOL_NAMES,
    GenericMcpClient,
    HERMETIC_CANDIDATE_SUITES,
    INTERFACE,
    MCP_TOOL_NAMES,
    OPERATOR_BLOCKING_TASK_ID,
    OBJECTIVE_KIND,
    OutcomeProbe,
    PCPR_081_GOAL_ID,
    PCPR_081_TASK_ID,
    PINNED_CATALOG_CID,
    PINNED_CHAIN_CID,
    PINNED_CLIENT_CID,
    PINNED_CURRENT_ROOT_CID,
    PINNED_DOCUMENT_CID,
    PINNED_IDEA_DIGEST,
    PINNED_OBJECTIVE_CID,
    PINNED_PACK_CID,
    PINNED_PYTHON_CLIENT_CID,
    SCHEMA,
    SEALED_PATH,
    SEALED_PYTHON,
    AccelerateGenericMcpClientError,
    current_head_static_probes,
    path_is_authorized,
    pcpr_081_receipt_promotion,
    platform_generic_mcp_client_catalog,
    produce_generic_mcp_client,
    qualify_current_head_generic_mcp_client,
    qualify_generic_mcp_client,
    refuse_chain_cid_remint,
    refuse_database_edit,
    refuse_forbidden_client_operation,
    refuse_mcp_plus_plus,
    refuse_model_completion,
    refuse_non_canonical_objective,
    refuse_objective_cid_remint,
    refuse_pack_cid_remint,
    refuse_self_promotion,
    refuse_unauthorized_path,
    render_declared_client,
    verify_generic_mcp_client_files,
)


_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def test_closed_vocabularies_match_pcpr_081_requirements() -> None:
    assert PCPR_081_TASK_ID == "PCPR-081"
    assert PCPR_081_GOAL_ID == "PCPR-G800"
    assert INTERFACE == "AccelerateGenericMcpClient@1"
    assert SCHEMA == "ipfs_accelerate_py/assurance/generic-mcp-client@1"
    assert OBJECTIVE_KIND == "declared_generic_mcp_client_demonstration"
    assert OPERATOR_BLOCKING_TASK_ID == "pcpr-081-operator-live-generic-mcp-client"
    assert "release_candidate_qualified" in CLOSED_RELEASE_OUTCOMES
    assert "rnd_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert HERMETIC_CANDIDATE_SUITES[-1].endswith(
        "test_agent_supervisor_generic_mcp_client.py"
    )
    assert SEALED_PATH == "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
    assert SEALED_PYTHON == "/usr/bin/python3.12"
    assert ESCALATION_ORDER[0] == "exact receipt"
    assert ESCALATION_ORDER[-1] == "human decision"
    assert CLIENT_OPERATIONS[0] == "submit_objective"
    assert CLIENT_OPERATIONS[-1] == "retrieve_final_receipts"
    assert MCP_TOOL_NAMES[0] == "pcpr_submit_objective"
    assert "self_promote" in FORBIDDEN_CLIENT_OPERATIONS
    assert "pcpr_self_promote" in FORBIDDEN_MCP_TOOL_NAMES
    document = render_declared_client()
    assert document["applied"] is False
    assert document["live"] is False
    assert document["release_claim"] is False
    assert document["closed_release_outcome"] is None
    client = document["generic_mcp_client"]
    assert client["produced"] is True
    assert client["live"] is False
    assert client["hermetic"] is True
    assert client["change_kind"] == "generic_mcp_client_demonstration"
    assert client["transport"] == "mcp"
    assert client["jsonrpc"] == "2.0"
    assert client["protocol_version"] == "2024-11-05"
    assert client["mcp_plus_plus"] is False
    assert client["authority"] == "intent_not_authority"
    assert client["identity_matches_pcpr_060"] is True
    assert client["events_subscribed"] is True
    assert client["task_state_inspected"] is True
    assert client["candidate_evidence_submitted"] is True
    assert client["final_receipts_retrieved"] is True
    assert client["complete_high_level_objective_path"] is True
    assert client["forbidden_operations_refused"] is True
    assert client["model_assertion_completes_work"] is False
    assert document["duckdb_or_quack_state_written"] is False
    assert document["objective_cid"] == PINNED_OBJECTIVE_CID
    assert document["idea_digest"] == PINNED_IDEA_DIGEST
    assert document["context_pack"]["pack_cid"] == PINNED_PACK_CID
    assert document["storage"]["current_root_cid"] == PINNED_CURRENT_ROOT_CID
    assert document["final_receipt_chain"]["chain_cid"] == PINNED_CHAIN_CID
    assert document["python_external_client"]["client_cid"] == PINNED_PYTHON_CLIENT_CID
    assert client["client_cid"] == PINNED_CLIENT_CID
    assert document["document_cid"] == PINNED_DOCUMENT_CID
    assert document["operator_blocking_task"]["status"] == "typed_blocked"
    idempotency = client["idempotency"]
    assert idempotency["identical"] is True
    assert idempotency["second_attempt_applied"] is False
    assert idempotency["first_client_cid"] == PINNED_CLIENT_CID
    assert refuse_pack_cid_remint(PINNED_PACK_CID) == PINNED_PACK_CID
    assert refuse_objective_cid_remint(PINNED_OBJECTIVE_CID) == PINNED_OBJECTIVE_CID
    assert refuse_chain_cid_remint(PINNED_CHAIN_CID) == PINNED_CHAIN_CID
    with pytest.raises(AccelerateGenericMcpClientError, match="remints"):
        refuse_pack_cid_remint(
            "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        )
    with pytest.raises(AccelerateGenericMcpClientError, match="cannot complete"):
        refuse_model_completion("frontier_model")
    with pytest.raises(AccelerateGenericMcpClientError, match="not an authorized"):
        refuse_unauthorized_path("/etc/passwd")
    with pytest.raises(AccelerateGenericMcpClientError, match="DuckDB"):
        refuse_database_edit(edited=True)
    with pytest.raises(AccelerateGenericMcpClientError, match="forbidden"):
        refuse_forbidden_client_operation("self_promote")
    with pytest.raises(AccelerateGenericMcpClientError, match="promote"):
        refuse_self_promotion(promoted=True)
    with pytest.raises(AccelerateGenericMcpClientError, match="canonical"):
        refuse_non_canonical_objective("a different idea")
    with pytest.raises(AccelerateGenericMcpClientError, match="not MCP"):
        refuse_mcp_plus_plus(claimed=True)
    assert path_is_authorized(AUTHORIZED_PATH_PREFIXES[0]) is True
    mcp_client = GenericMcpClient()
    assert mcp_client.submit_objective(
        "Modify a typed formal-logic API while reusing unaffected proofs, "
        "selecting only impacted tests, rejecting stale-tree evidence, and "
        "producing a complete proof-carrying execution receipt."
    )["submitted"] is True
    assert mcp_client.receive_identity()["objective_cid"] == PINNED_OBJECTIVE_CID
    assert mcp_client.retrieve_final_receipts()["chain_cid"] == PINNED_CHAIN_CID
    refused = mcp_client.refuse_forbidden_tool("pcpr_self_promote")
    assert refused["refused"] is True
    assert refused["jsonrpc_error_code"] == -32001


def test_current_head_is_rnd_non_promoted_and_not_a_release() -> None:
    verdict = qualify_current_head_generic_mcp_client()
    assert verdict.promotion_status == "rnd_non_promoted"
    assert verdict.supervisor_disposition == "supervisor_non_promoted"
    assert verdict.closed_release_outcome is None
    assert verdict.release_claim is False
    assert verdict.completion_authoritative is False
    assert verdict.contracts_frozen is False
    assert verdict.duckdb_or_quack_state_written is False
    assert verdict.sibling_source_required is False
    assert verdict.live_application is False
    assert verdict.live_client is False
    assert verdict.operator_blocking_task == OPERATOR_BLOCKING_TASK_ID
    assert verdict.simulated_results_represented_as_live is False
    assert verdict.this_task_created_competing_authority is False
    assert verdict.promotion_status not in CLOSED_RELEASE_OUTCOMES
    assert verdict.verdict_cid.startswith("baguqeera")
    assert verdict.verdict_cid == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert verdict.pack_cid == PINNED_PACK_CID
    assert verdict.client_cid == PINNED_CLIENT_CID
    assert verdict.document_cid == PINNED_DOCUMENT_CID
    assert verdict.catalog_cid == PINNED_CATALOG_CID
    assert verdict.current_root_cid == PINNED_CURRENT_ROOT_CID
    assert verdict.chain_cid == PINNED_CHAIN_CID
    assert verdict.blockers == ()
    section = pcpr_081_receipt_promotion(verdict)
    assert section["promotion_status"] == "rnd_non_promoted"
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_static_probes_show_declared_client_constraints() -> None:
    probes = {item.probe_id: item for item in current_head_static_probes()}
    assert probes["client_files_match_generator"].present is True
    assert probes["pyproject_generic_mcp_client_table"].present is True
    assert probes["paths_remain_authorized"].present is True
    assert probes["canonical_objective_bytes_match"].present is True
    assert probes["objective_identity_matches_pcpr_060"].present is True
    assert probes["owner_pack_cid_matches_pin"].present is True
    assert probes["kit_current_root_bound_not_minted"].present is True
    assert probes["chain_cid_bound_not_minted"].present is True
    assert probes["mcp_transport_used"].present is True
    assert probes["jsonrpc_envelope_used"].present is True
    assert probes["canonical_service_shared"].present is True
    assert probes["client_is_intent_not_authority"].present is True
    assert probes["initialize_handshake_demonstrated"].present is True
    assert probes["tools_list_demonstrated"].present is True
    assert probes["submit_objective_demonstrated"].present is True
    assert probes["receive_identity_demonstrated"].present is True
    assert probes["subscribe_events_demonstrated"].present is True
    assert probes["inspect_task_state_demonstrated"].present is True
    assert probes["submit_candidate_evidence_demonstrated"].present is True
    assert probes["retrieve_final_receipts_demonstrated"].present is True
    assert probes["complete_high_level_objective_path_demonstrated"].present is True
    assert probes["forbidden_client_operations_refused"].present is True
    assert probes["idempotent_client_cid"].present is True
    assert probes["model_assertion_cannot_complete_work"].present is True
    assert probes["datasets_identity_reminted"].present is False
    assert probes["kit_identity_reminted"].present is False
    assert probes["chain_identity_reminted"].present is False
    assert probes["python_client_identity_reminted"].present is False
    assert probes["forbidden_client_operation_accepted"].present is False
    assert probes["database_edited"].present is False
    assert probes["client_self_promoted"].present is False
    assert probes["mcp_plus_plus_represented_as_generic_mcp"].present is False
    assert probes["live_application"].evidence_kind == "unavailable"
    assert probes["live_client"].evidence_kind == "unavailable"
    for probe in probes.values():
        assert probe.live is False
        assert probe.simulated_represented_as_live is False


def test_committed_files_match_generator() -> None:
    verified = verify_generic_mcp_client_files()
    assert verified["ok"] is True
    catalog = platform_generic_mcp_client_catalog()
    assert catalog["interface"] == "PlatformGenericMcpClient@1"
    assert catalog["release_claim"] is False
    assert catalog["closed_release_outcome"] is None
    assert catalog["live_application"] is False
    assert catalog["live_client"] is False
    assert catalog["pack_cid"] == PINNED_PACK_CID
    assert catalog["chain_cid"] == PINNED_CHAIN_CID
    assert catalog["client_cid"] == PINNED_CLIENT_CID
    assert catalog["catalog_cid"] == PINNED_CATALOG_CID
    assert catalog["components"]["ipfs_accelerate_py"]["status"] == "observed"
    pyproject = (_PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'interface = "AccelerateGenericMcpClient@1"' in pyproject
    plan = produce_generic_mcp_client()
    assert plan["produced"] is True
    assert plan["live"] is False
    assert tuple(plan["escalation_order"]) == ESCALATION_ORDER
    with pytest.raises(AccelerateGenericMcpClientError, match="cannot complete"):
        produce_generic_mcp_client(complete_from="frontier_model")
    with pytest.raises(AccelerateGenericMcpClientError, match="DuckDB"):
        produce_generic_mcp_client(database_edited=True)
    with pytest.raises(AccelerateGenericMcpClientError, match="forbidden"):
        produce_generic_mcp_client(forbidden_operation="terminalize_task")
    with pytest.raises(AccelerateGenericMcpClientError, match="not MCP"):
        produce_generic_mcp_client(mcp_plus_plus=True)


def test_simulated_live_probe_is_rejected() -> None:
    dummy = "baguqeerabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    with pytest.raises(AccelerateGenericMcpClientError, match="simulated"):
        qualify_generic_mcp_client(
            (
                OutcomeProbe(
                    probe_id="bogus",
                    present=False,
                    evidence_kind="simulated",
                    live=False,
                    simulated_represented_as_live=True,
                    reason="must fail",
                ),
            ),
            client_cid=dummy,
            document_cid=dummy,
            catalog_cid=dummy,
            chain_cid=dummy,
            pack_cid=dummy,
            current_root_cid=dummy,
            objective_cid=dummy,
            idea_digest_cid=dummy,
        )
    with pytest.raises(AccelerateGenericMcpClientError, match="measured_live"):
        qualify_generic_mcp_client(
            (
                OutcomeProbe(
                    probe_id="bogus",
                    present=True,
                    evidence_kind="measured",
                    live=True,
                    simulated_represented_as_live=False,
                    reason="must fail",
                ),
            ),
            client_cid=dummy,
            document_cid=dummy,
            catalog_cid=dummy,
            chain_cid=dummy,
            pack_cid=dummy,
            current_root_cid=dummy,
            objective_cid=dummy,
            idea_digest_cid=dummy,
        )
