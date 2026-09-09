"""PCPR-083: Accelerate-owned external-client authority-bypass proof."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.assurance.authority_bypass import (
    AUTHORIZED_PATH_PREFIXES,
    CLIENT_OPERATIONS,
    CLOSED_RELEASE_OUTCOMES,
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID,
    ESCALATION_ORDER,
    FORBIDDEN_CLIENT_OPERATIONS,
    HARD_ZERO_COUNTERS,
    HERMETIC_CANDIDATE_SUITES,
    INTERFACE,
    OBJECTIVE_KIND,
    OPERATOR_BLOCKING_TASK_ID,
    OutcomeProbe,
    PCPR_083_GOAL_ID,
    PCPR_083_TASK_ID,
    PINNED_CATALOG_CID,
    PINNED_CHAIN_CID,
    PINNED_CURRENT_ROOT_CID,
    PINNED_DOCUMENT_CID,
    PINNED_IDEA_DIGEST,
    PINNED_MCP_CLIENT_CID,
    PINNED_OBJECTIVE_CID,
    PINNED_PACK_CID,
    PINNED_PARITY_CID,
    PINNED_PROOF_CID,
    PINNED_PYTHON_CLIENT_CID,
    SCHEMA,
    SEALED_PATH,
    SEALED_PYTHON,
    AccelerateAuthorityBypassError,
    attempt_mcp_bypass,
    attempt_python_bypass,
    current_head_static_probes,
    hermetic_authority_bypass_trace,
    path_is_authorized,
    pcpr_083_receipt_promotion,
    platform_authority_bypass_catalog,
    produce_authority_bypass,
    qualify_authority_bypass,
    qualify_current_head_authority_bypass,
    refuse_bypass_acceptance,
    refuse_chain_cid_remint,
    refuse_database_edit,
    refuse_mcp_client_cid_remint,
    refuse_model_completion,
    refuse_non_canonical_objective,
    refuse_objective_cid_remint,
    refuse_pack_cid_remint,
    refuse_parity_cid_remint,
    refuse_python_client_cid_remint,
    refuse_self_promotion,
    refuse_unauthorized_effect,
    refuse_unauthorized_path,
    render_declared_proof,
    verify_authority_bypass_files,
    zero_hard_zero_counters,
)


_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def test_closed_vocabularies_match_pcpr_083_requirements() -> None:
    assert PCPR_083_TASK_ID == "PCPR-083"
    assert PCPR_083_GOAL_ID == "PCPR-G800"
    assert INTERFACE == "AccelerateAuthorityBypass@1"
    assert SCHEMA == "ipfs_accelerate_py/assurance/authority-bypass@1"
    assert OBJECTIVE_KIND == "declared_external_client_authority_bypass_proof"
    assert OPERATOR_BLOCKING_TASK_ID == (
        "pcpr-083-operator-live-external-client-authority-bypass"
    )
    assert "release_candidate_qualified" in CLOSED_RELEASE_OUTCOMES
    assert "rnd_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert HERMETIC_CANDIDATE_SUITES[-1].endswith(
        "test_agent_supervisor_authority_bypass.py"
    )
    assert SEALED_PATH == "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
    assert SEALED_PYTHON == "/usr/bin/python3.12"
    assert ESCALATION_ORDER[0] == "exact receipt"
    assert ESCALATION_ORDER[-1] == "human decision"
    assert CLIENT_OPERATIONS[0] == "submit_objective"
    assert CLIENT_OPERATIONS[-1] == "retrieve_final_receipts"
    assert FORBIDDEN_CLIENT_OPERATIONS[0] == "update_task"
    assert FORBIDDEN_CLIENT_OPERATIONS[-1] == "self_promote"
    assert "unauthorized_task_updates" in HARD_ZERO_COUNTERS
    document = render_declared_proof()
    assert document["applied"] is False
    assert document["live"] is False
    assert document["release_claim"] is False
    assert document["closed_release_outcome"] is None
    proof = document["authority_bypass"]
    assert proof["produced"] is True
    assert proof["live"] is False
    assert proof["hermetic"] is True
    assert proof["change_kind"] == "external_client_authority_bypass_proof"
    assert proof["python_transport"] == "python"
    assert proof["mcp_transport"] == "mcp"
    assert proof["canonical_service"] == "pcpr-canonical-supervisor-service"
    assert proof["authority"] == "intent_not_authority"
    assert proof["every_python_bypass_attempt_fails_typed"] is True
    assert proof["every_mcp_bypass_attempt_fails_typed"] is True
    assert proof["no_unauthorized_effect"] is True
    assert proof["hard_zero_counters_remain_zero"] is True
    assert proof["task_state_unchanged"] is True
    assert proof["current_root_unchanged"] is True
    assert proof["policy_pointer_unchanged"] is True
    assert proof["client_cids_distinct"] is True
    assert proof["client_self_promoted"] is False
    assert proof["model_assertion_completes_work"] is False
    assert document["duckdb_or_quack_state_written"] is False
    assert document["objective_cid"] == PINNED_OBJECTIVE_CID
    assert document["idea_digest"] == PINNED_IDEA_DIGEST
    assert document["context_pack"]["pack_cid"] == PINNED_PACK_CID
    assert document["storage"]["current_root_cid"] == PINNED_CURRENT_ROOT_CID
    assert document["final_receipt_chain"]["chain_cid"] == PINNED_CHAIN_CID
    assert document["python_external_client"]["client_cid"] == PINNED_PYTHON_CLIENT_CID
    assert document["generic_mcp_client"]["client_cid"] == PINNED_MCP_CLIENT_CID
    assert document["objective_identity_parity"]["parity_cid"] == PINNED_PARITY_CID
    assert PINNED_PYTHON_CLIENT_CID != PINNED_MCP_CLIENT_CID
    assert proof["proof_cid"] == PINNED_PROOF_CID
    assert document["document_cid"] == PINNED_DOCUMENT_CID
    assert document["operator_blocking_task"]["status"] == "typed_blocked"
    trace = hermetic_authority_bypass_trace()
    assert trace["every_python_bypass_attempt_fails_typed"] is True
    assert trace["every_mcp_bypass_attempt_fails_typed"] is True
    assert trace["unauthorized_effect"] is False
    assert trace["accepted_count"] == 0
    assert all(value == 0 for value in zero_hard_zero_counters().values())
    idempotency = proof["idempotency"]
    assert idempotency["identical"] is True
    assert idempotency["second_attempt_applied"] is False
    assert idempotency["first_proof_cid"] == PINNED_PROOF_CID
    assert refuse_pack_cid_remint(PINNED_PACK_CID) == PINNED_PACK_CID
    assert refuse_objective_cid_remint(PINNED_OBJECTIVE_CID) == PINNED_OBJECTIVE_CID
    assert refuse_chain_cid_remint(PINNED_CHAIN_CID) == PINNED_CHAIN_CID
    assert refuse_python_client_cid_remint(PINNED_PYTHON_CLIENT_CID) == PINNED_PYTHON_CLIENT_CID
    assert refuse_mcp_client_cid_remint(PINNED_MCP_CLIENT_CID) == PINNED_MCP_CLIENT_CID
    assert refuse_parity_cid_remint(PINNED_PARITY_CID) == PINNED_PARITY_CID
    with pytest.raises(AccelerateAuthorityBypassError, match="remints"):
        refuse_pack_cid_remint(
            "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        )
    with pytest.raises(AccelerateAuthorityBypassError, match="cannot complete"):
        refuse_model_completion("frontier_model")
    with pytest.raises(AccelerateAuthorityBypassError, match="not an authorized"):
        refuse_unauthorized_path("/etc/passwd")
    with pytest.raises(AccelerateAuthorityBypassError, match="DuckDB"):
        refuse_database_edit(edited=True)
    with pytest.raises(AccelerateAuthorityBypassError, match="promote"):
        refuse_self_promotion(promoted=True)
    with pytest.raises(AccelerateAuthorityBypassError, match="canonical"):
        refuse_non_canonical_objective("a different idea")
    with pytest.raises(AccelerateAuthorityBypassError, match="accepted"):
        refuse_bypass_acceptance(accepted=True)
    with pytest.raises(AccelerateAuthorityBypassError, match="unauthorized"):
        refuse_unauthorized_effect(observed=True)
    assert path_is_authorized(AUTHORIZED_PATH_PREFIXES[0]) is True
    python_refusal = attempt_python_bypass("update_task")
    assert python_refusal["refused"] is True
    mcp_refusal = attempt_mcp_bypass("self_promote")
    assert mcp_refusal["refused"] is True
    assert mcp_refusal["jsonrpc_error_code"] == -32001


def test_current_head_is_rnd_non_promoted_and_not_a_release() -> None:
    verdict = qualify_current_head_authority_bypass()
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
    assert verdict.proof_cid == PINNED_PROOF_CID
    assert verdict.document_cid == PINNED_DOCUMENT_CID
    assert verdict.catalog_cid == PINNED_CATALOG_CID
    assert verdict.current_root_cid == PINNED_CURRENT_ROOT_CID
    assert verdict.chain_cid == PINNED_CHAIN_CID
    assert verdict.python_client_cid == PINNED_PYTHON_CLIENT_CID
    assert verdict.mcp_client_cid == PINNED_MCP_CLIENT_CID
    assert verdict.parity_cid == PINNED_PARITY_CID
    assert verdict.blockers == ()
    section = pcpr_083_receipt_promotion(verdict)
    assert section["promotion_status"] == "rnd_non_promoted"
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_static_probes_show_declared_bypass_constraints() -> None:
    probes = {item.probe_id: item for item in current_head_static_probes()}
    assert probes["bypass_files_match_generator"].present is True
    assert probes["pyproject_authority_bypass_table"].present is True
    assert probes["paths_remain_authorized"].present is True
    assert probes["canonical_objective_bytes_match"].present is True
    assert probes["every_python_bypass_attempt_fails_typed"].present is True
    assert probes["every_mcp_bypass_attempt_fails_typed"].present is True
    assert probes["no_unauthorized_effect"].present is True
    assert probes["hard_zero_counters_remain_zero"].present is True
    assert probes["task_state_unchanged"].present is True
    assert probes["current_root_unchanged"].present is True
    assert probes["policy_pointer_unchanged"].present is True
    assert probes["owner_pack_cid_matches_pin"].present is True
    assert probes["kit_current_root_bound_not_minted"].present is True
    assert probes["chain_cid_bound_not_minted"].present is True
    assert probes["canonical_service_shared"].present is True
    assert probes["python_client_cid_bound_not_minted"].present is True
    assert probes["mcp_client_cid_bound_not_minted"].present is True
    assert probes["parity_cid_bound_not_minted"].present is True
    assert probes["client_cids_remain_distinct"].present is True
    assert probes["clients_are_intent_not_authority"].present is True
    assert probes["idempotent_proof_cid"].present is True
    assert probes["model_assertion_cannot_complete_work"].present is True
    assert probes["datasets_identity_reminted"].present is False
    assert probes["kit_identity_reminted"].present is False
    assert probes["chain_identity_reminted"].present is False
    assert probes["python_client_identity_reminted"].present is False
    assert probes["mcp_client_identity_reminted"].present is False
    assert probes["parity_identity_reminted"].present is False
    assert probes["bypass_attempt_accepted"].present is False
    assert probes["unauthorized_effect_observed"].present is False
    assert probes["hard_zero_counter_nonzero"].present is False
    assert probes["database_edited"].present is False
    assert probes["client_self_promoted"].present is False
    assert probes["live_application"].evidence_kind == "unavailable"
    assert probes["live_client"].evidence_kind == "unavailable"
    for probe in probes.values():
        assert probe.live is False
        assert probe.simulated_represented_as_live is False


def test_committed_files_match_generator() -> None:
    verified = verify_authority_bypass_files()
    assert verified["ok"] is True
    catalog = platform_authority_bypass_catalog()
    assert catalog["interface"] == "PlatformAuthorityBypass@1"
    assert catalog["release_claim"] is False
    assert catalog["closed_release_outcome"] is None
    assert catalog["live_application"] is False
    assert catalog["live_client"] is False
    assert catalog["pack_cid"] == PINNED_PACK_CID
    assert catalog["chain_cid"] == PINNED_CHAIN_CID
    assert catalog["proof_cid"] == PINNED_PROOF_CID
    assert catalog["catalog_cid"] == PINNED_CATALOG_CID
    assert catalog["components"]["ipfs_accelerate_py"]["status"] == "observed"
    pyproject = (_PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'interface = "AccelerateAuthorityBypass@1"' in pyproject
    plan = produce_authority_bypass()
    assert plan["produced"] is True
    assert plan["live"] is False
    assert tuple(plan["escalation_order"]) == ESCALATION_ORDER
    with pytest.raises(AccelerateAuthorityBypassError, match="cannot complete"):
        produce_authority_bypass(complete_from="frontier_model")
    with pytest.raises(AccelerateAuthorityBypassError, match="DuckDB"):
        produce_authority_bypass(database_edited=True)
    with pytest.raises(AccelerateAuthorityBypassError, match="accepted"):
        produce_authority_bypass(bypass_accepted=True)


def test_simulated_live_probe_is_rejected() -> None:
    dummy = "baguqeerabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    with pytest.raises(AccelerateAuthorityBypassError, match="simulated"):
        qualify_authority_bypass(
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
            proof_cid=dummy,
            document_cid=dummy,
            catalog_cid=dummy,
            chain_cid=dummy,
            pack_cid=dummy,
            current_root_cid=dummy,
            objective_cid=dummy,
            idea_digest_cid=dummy,
        )
    with pytest.raises(AccelerateAuthorityBypassError, match="measured_live"):
        qualify_authority_bypass(
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
            proof_cid=dummy,
            document_cid=dummy,
            catalog_cid=dummy,
            chain_cid=dummy,
            pack_cid=dummy,
            current_root_cid=dummy,
            objective_cid=dummy,
            idea_digest_cid=dummy,
        )
