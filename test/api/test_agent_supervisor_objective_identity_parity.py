"""PCPR-082: Accelerate-owned cross-client objective identity parity."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.assurance.objective_identity_parity import (
    AUTHORIZED_PATH_PREFIXES,
    CLIENT_OPERATIONS,
    CLOSED_RELEASE_OUTCOMES,
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID,
    ESCALATION_ORDER,
    HERMETIC_CANDIDATE_SUITES,
    INTERFACE,
    OBJECTIVE_KIND,
    OPERATOR_BLOCKING_TASK_ID,
    OutcomeProbe,
    PCPR_082_GOAL_ID,
    PCPR_082_TASK_ID,
    PINNED_CATALOG_CID,
    PINNED_CHAIN_CID,
    PINNED_CURRENT_ROOT_CID,
    PINNED_DOCUMENT_CID,
    PINNED_IDEA_DIGEST,
    PINNED_MCP_CLIENT_CID,
    PINNED_OBJECTIVE_CID,
    PINNED_PACK_CID,
    PINNED_PARITY_CID,
    PINNED_PYTHON_CLIENT_CID,
    SCHEMA,
    SEALED_PATH,
    SEALED_PYTHON,
    AccelerateObjectiveIdentityParityError,
    compare_client_identities,
    current_head_static_probes,
    path_is_authorized,
    pcpr_082_receipt_promotion,
    platform_objective_identity_parity_catalog,
    produce_objective_identity_parity,
    qualify_current_head_objective_identity_parity,
    qualify_objective_identity_parity,
    refuse_chain_cid_remint,
    refuse_database_edit,
    refuse_identity_mismatch,
    refuse_mcp_client_cid_remint,
    refuse_model_completion,
    refuse_non_canonical_objective,
    refuse_objective_cid_remint,
    refuse_pack_cid_remint,
    refuse_python_client_cid_remint,
    refuse_self_promotion,
    refuse_unauthorized_path,
    render_declared_parity,
    verify_objective_identity_parity_files,
)


_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def test_closed_vocabularies_match_pcpr_082_requirements() -> None:
    assert PCPR_082_TASK_ID == "PCPR-082"
    assert PCPR_082_GOAL_ID == "PCPR-G800"
    assert INTERFACE == "AccelerateObjectiveIdentityParity@1"
    assert SCHEMA == "ipfs_accelerate_py/assurance/objective-identity-parity@1"
    assert OBJECTIVE_KIND == "declared_cross_client_objective_identity_parity"
    assert OPERATOR_BLOCKING_TASK_ID == (
        "pcpr-082-operator-live-cross-client-objective-identity-parity"
    )
    assert "release_candidate_qualified" in CLOSED_RELEASE_OUTCOMES
    assert "rnd_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert HERMETIC_CANDIDATE_SUITES[-1].endswith(
        "test_agent_supervisor_objective_identity_parity.py"
    )
    assert SEALED_PATH == "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
    assert SEALED_PYTHON == "/usr/bin/python3.12"
    assert ESCALATION_ORDER[0] == "exact receipt"
    assert ESCALATION_ORDER[-1] == "human decision"
    assert CLIENT_OPERATIONS[0] == "submit_objective"
    assert CLIENT_OPERATIONS[-1] == "retrieve_final_receipts"
    document = render_declared_parity()
    assert document["applied"] is False
    assert document["live"] is False
    assert document["release_claim"] is False
    assert document["closed_release_outcome"] is None
    parity = document["objective_identity_parity"]
    assert parity["produced"] is True
    assert parity["live"] is False
    assert parity["hermetic"] is True
    assert parity["change_kind"] == "cross_client_objective_identity_parity"
    assert parity["python_transport"] == "python"
    assert parity["mcp_transport"] == "mcp"
    assert parity["canonical_service"] == "pcpr-canonical-supervisor-service"
    assert parity["authority"] == "intent_not_authority"
    assert parity["identity_matches_pcpr_060"] is True
    assert parity["identity_matches_across_clients"] is True
    assert parity["equivalent_goals"] is True
    assert parity["equivalent_tasks"] is True
    assert parity["equivalent_events"] is True
    assert parity["equivalent_candidate_evidence"] is True
    assert parity["equivalent_final_receipts"] is True
    assert parity["client_cids_distinct"] is True
    assert parity["model_assertion_completes_work"] is False
    assert document["duckdb_or_quack_state_written"] is False
    assert document["objective_cid"] == PINNED_OBJECTIVE_CID
    assert document["idea_digest"] == PINNED_IDEA_DIGEST
    assert document["context_pack"]["pack_cid"] == PINNED_PACK_CID
    assert document["storage"]["current_root_cid"] == PINNED_CURRENT_ROOT_CID
    assert document["final_receipt_chain"]["chain_cid"] == PINNED_CHAIN_CID
    assert document["python_external_client"]["client_cid"] == PINNED_PYTHON_CLIENT_CID
    assert document["generic_mcp_client"]["client_cid"] == PINNED_MCP_CLIENT_CID
    assert PINNED_PYTHON_CLIENT_CID != PINNED_MCP_CLIENT_CID
    assert parity["parity_cid"] == PINNED_PARITY_CID
    assert document["document_cid"] == PINNED_DOCUMENT_CID
    assert document["operator_blocking_task"]["status"] == "typed_blocked"
    comparison = compare_client_identities()
    assert comparison["matched"] is True
    assert comparison["same_objective_cid"] is True
    assert comparison["equivalent_events"] is True
    idempotency = parity["idempotency"]
    assert idempotency["identical"] is True
    assert idempotency["second_attempt_applied"] is False
    assert idempotency["first_parity_cid"] == PINNED_PARITY_CID
    assert refuse_pack_cid_remint(PINNED_PACK_CID) == PINNED_PACK_CID
    assert refuse_objective_cid_remint(PINNED_OBJECTIVE_CID) == PINNED_OBJECTIVE_CID
    assert refuse_chain_cid_remint(PINNED_CHAIN_CID) == PINNED_CHAIN_CID
    assert refuse_python_client_cid_remint(PINNED_PYTHON_CLIENT_CID) == PINNED_PYTHON_CLIENT_CID
    assert refuse_mcp_client_cid_remint(PINNED_MCP_CLIENT_CID) == PINNED_MCP_CLIENT_CID
    with pytest.raises(AccelerateObjectiveIdentityParityError, match="remints"):
        refuse_pack_cid_remint(
            "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        )
    with pytest.raises(AccelerateObjectiveIdentityParityError, match="cannot complete"):
        refuse_model_completion("frontier_model")
    with pytest.raises(AccelerateObjectiveIdentityParityError, match="not an authorized"):
        refuse_unauthorized_path("/etc/passwd")
    with pytest.raises(AccelerateObjectiveIdentityParityError, match="DuckDB"):
        refuse_database_edit(edited=True)
    with pytest.raises(AccelerateObjectiveIdentityParityError, match="promote"):
        refuse_self_promotion(promoted=True)
    with pytest.raises(AccelerateObjectiveIdentityParityError, match="canonical"):
        refuse_non_canonical_objective("a different idea")
    with pytest.raises(AccelerateObjectiveIdentityParityError, match="same objective"):
        refuse_identity_mismatch(matched=False)
    assert path_is_authorized(AUTHORIZED_PATH_PREFIXES[0]) is True


def test_current_head_is_rnd_non_promoted_and_not_a_release() -> None:
    verdict = qualify_current_head_objective_identity_parity()
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
    assert verdict.parity_cid == PINNED_PARITY_CID
    assert verdict.document_cid == PINNED_DOCUMENT_CID
    assert verdict.catalog_cid == PINNED_CATALOG_CID
    assert verdict.current_root_cid == PINNED_CURRENT_ROOT_CID
    assert verdict.chain_cid == PINNED_CHAIN_CID
    assert verdict.python_client_cid == PINNED_PYTHON_CLIENT_CID
    assert verdict.mcp_client_cid == PINNED_MCP_CLIENT_CID
    assert verdict.blockers == ()
    section = pcpr_082_receipt_promotion(verdict)
    assert section["promotion_status"] == "rnd_non_promoted"
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_static_probes_show_declared_parity_constraints() -> None:
    probes = {item.probe_id: item for item in current_head_static_probes()}
    assert probes["parity_files_match_generator"].present is True
    assert probes["pyproject_objective_identity_parity_table"].present is True
    assert probes["paths_remain_authorized"].present is True
    assert probes["canonical_objective_bytes_match"].present is True
    assert probes["python_and_mcp_submit_same_objective_bytes"].present is True
    assert probes["objective_identity_matches_across_clients"].present is True
    assert probes["equivalent_goals_tasks_events_evidence_receipts"].present is True
    assert probes["owner_pack_cid_matches_pin"].present is True
    assert probes["kit_current_root_bound_not_minted"].present is True
    assert probes["chain_cid_bound_not_minted"].present is True
    assert probes["canonical_service_shared"].present is True
    assert probes["python_client_cid_bound_not_minted"].present is True
    assert probes["mcp_client_cid_bound_not_minted"].present is True
    assert probes["client_cids_remain_distinct"].present is True
    assert probes["clients_are_intent_not_authority"].present is True
    assert probes["idempotent_parity_cid"].present is True
    assert probes["model_assertion_cannot_complete_work"].present is True
    assert probes["datasets_identity_reminted"].present is False
    assert probes["kit_identity_reminted"].present is False
    assert probes["chain_identity_reminted"].present is False
    assert probes["python_client_identity_reminted"].present is False
    assert probes["mcp_client_identity_reminted"].present is False
    assert probes["identity_mismatch_accepted"].present is False
    assert probes["database_edited"].present is False
    assert probes["client_self_promoted"].present is False
    assert probes["live_application"].evidence_kind == "unavailable"
    assert probes["live_client"].evidence_kind == "unavailable"
    for probe in probes.values():
        assert probe.live is False
        assert probe.simulated_represented_as_live is False


def test_committed_files_match_generator() -> None:
    verified = verify_objective_identity_parity_files()
    assert verified["ok"] is True
    catalog = platform_objective_identity_parity_catalog()
    assert catalog["interface"] == "PlatformObjectiveIdentityParity@1"
    assert catalog["release_claim"] is False
    assert catalog["closed_release_outcome"] is None
    assert catalog["live_application"] is False
    assert catalog["live_client"] is False
    assert catalog["pack_cid"] == PINNED_PACK_CID
    assert catalog["chain_cid"] == PINNED_CHAIN_CID
    assert catalog["parity_cid"] == PINNED_PARITY_CID
    assert catalog["catalog_cid"] == PINNED_CATALOG_CID
    assert catalog["components"]["ipfs_accelerate_py"]["status"] == "observed"
    pyproject = (_PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'interface = "AccelerateObjectiveIdentityParity@1"' in pyproject
    plan = produce_objective_identity_parity()
    assert plan["produced"] is True
    assert plan["live"] is False
    assert tuple(plan["escalation_order"]) == ESCALATION_ORDER
    with pytest.raises(AccelerateObjectiveIdentityParityError, match="cannot complete"):
        produce_objective_identity_parity(complete_from="frontier_model")
    with pytest.raises(AccelerateObjectiveIdentityParityError, match="DuckDB"):
        produce_objective_identity_parity(database_edited=True)
    with pytest.raises(AccelerateObjectiveIdentityParityError, match="same objective"):
        produce_objective_identity_parity(identity_matched=False)


def test_simulated_live_probe_is_rejected() -> None:
    dummy = "baguqeerabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    with pytest.raises(AccelerateObjectiveIdentityParityError, match="simulated"):
        qualify_objective_identity_parity(
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
            parity_cid=dummy,
            document_cid=dummy,
            catalog_cid=dummy,
            chain_cid=dummy,
            pack_cid=dummy,
            current_root_cid=dummy,
            objective_cid=dummy,
            idea_digest_cid=dummy,
        )
    with pytest.raises(AccelerateObjectiveIdentityParityError, match="measured_live"):
        qualify_objective_identity_parity(
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
            parity_cid=dummy,
            document_cid=dummy,
            catalog_cid=dummy,
            chain_cid=dummy,
            pack_cid=dummy,
            current_root_cid=dummy,
            objective_cid=dummy,
            idea_digest_cid=dummy,
        )
