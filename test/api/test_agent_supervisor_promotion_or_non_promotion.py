"""PCPR-094: Accelerate-owned objective-to-release candidate gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.assurance.promotion_or_non_promotion import (
    AUTHORIZED_PATH_PREFIXES,
    CLOSED_RELEASE_OUTCOMES,
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID,
    ESCALATION_ORDER,
    GATE_FAMILIES,
    GATES,
    HERMETIC_CANDIDATE_SUITES,
    INTERFACE,
    OBJECTIVE_KIND,
    OPERATOR_BLOCKING_TASK_ID,
    OutcomeProbe,
    PCPR_094_GOAL_ID,
    PCPR_094_TASK_ID,
    PINNED_AUDIT_PACKAGE_CID,
    PINNED_BYPASS_PROOF_CID,
    PINNED_CATALOG_CID,
    PINNED_CHAIN_CID,
    PINNED_CURRENT_ROOT_CID,
    PINNED_DOCUMENT_CID,
    PINNED_DECISION_CID,
    PINNED_GATE_RUN_CID,
    PINNED_IDEA_DIGEST,
    PINNED_MCP_CLIENT_CID,
    PINNED_OBJECTIVE_CID,
    PINNED_PACK_CID,
    PINNED_PARITY_CID,
    PINNED_PYTHON_CLIENT_CID,
    PINNED_TCB_INVENTORY_CID,
    PINNED_THREAT_MODEL_CID,
    SCHEMA,
    SEALED_PATH,
    SEALED_PYTHON,
    AcceleratePromotionOrNonPromotionError,
    all_gate_families_evaluated,
    current_head_static_probes,
    hermetic_gate_run_trace,
    path_is_authorized,
    pcpr_094_receipt_promotion,
    platform_promotion_or_non_promotion_catalog,
    produce_promotion_or_non_promotion,
    qualify_current_head_promotion_or_non_promotion,
    qualify_promotion_or_non_promotion,
    refuse_audit_package_cid_remint,
    refuse_closed_release_claim,
    refuse_gate_run_cid_remint,
    refuse_database_edit,
    refuse_hidden_evidence,
    refuse_live_pass,
    refuse_model_completion,
    refuse_non_canonical_objective,
    refuse_pack_cid_remint,
    refuse_self_promotion,
    refuse_tcb_inventory_cid_remint,
    refuse_threat_model_cid_remint,
    refuse_unauthorized_path,
    refuse_unbounded_gate,
    render_declared_package,
    reproduce_promotion_or_non_promotion,
    verify_promotion_or_non_promotion_files,
)


_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def test_closed_vocabularies_match_pcpr_094_requirements() -> None:
    assert PCPR_094_TASK_ID == "PCPR-094"
    assert PCPR_094_GOAL_ID == "PCPR-G900"
    assert INTERFACE == "AcceleratePromotionOrNonPromotionReceipt@1"
    assert SCHEMA == "ipfs_accelerate_py/assurance/promotion-or-non-promotion-receipt@1"
    assert OBJECTIVE_KIND == "declared_objective_to_release_promotion_or_non_promotion"
    assert OPERATOR_BLOCKING_TASK_ID == "pcpr-094-operator-live-promotion"
    assert "release_candidate_qualified" in CLOSED_RELEASE_OUTCOMES
    assert "rnd_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert HERMETIC_CANDIDATE_SUITES[-1].endswith(
        "test_agent_supervisor_promotion_or_non_promotion.py"
    )
    assert SEALED_PATH == "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
    assert SEALED_PYTHON == "/usr/bin/python3.12"
    assert ESCALATION_ORDER[0] == "exact receipt"
    assert ESCALATION_ORDER[-1] == "human decision"
    assert "supervisor" in GATE_FAMILIES
    assert "hard_safety" in GATE_FAMILIES
    assert all_gate_families_evaluated() is True
    assert len(GATES) == 8
    document = render_declared_package()
    assert document["applied"] is False
    assert document["live"] is False
    assert document["release_claim"] is False
    assert document["closed_release_outcome"] is None
    package = document["promotion_or_non_promotion"]
    assert package["produced"] is True
    assert package["live"] is False
    assert package["hermetic"] is True
    assert package["change_kind"] == "objective_to_release_promotion_or_non_promotion"
    assert package["canonical_service"] == "pcpr-canonical-supervisor-service"
    assert package["authority"] == "intent_not_authority"
    assert package["all_gate_families_evaluated"] is True
    assert package["all_gates_bounded"] is True
    assert package["no_hidden_failed_stale_partial_simulated_estimated_or_missing"] is True
    assert package["reproduction_commands_work"] is True
    assert package["promotion_produced"] is True
    assert package["promotion_deferred"] is False
    assert package["honest_non_promotion_produced"] is True
    assert package["decision"] == "honest_non_promotion"
    assert package["promotion_status"] == "rnd_non_promoted"
    assert package["closed_release_deferred"] is False
    assert package["gate_run_cid"] == PINNED_GATE_RUN_CID
    assert document["release_candidate_gate"]["gate_run_cid"] == PINNED_GATE_RUN_CID
    assert package["client_self_promoted"] is False
    assert package["any_live_pass"] is False
    assert package["model_assertion_completes_work"] is False
    assert document["duckdb_or_quack_state_written"] is False
    assert document["objective_cid"] == PINNED_OBJECTIVE_CID
    assert document["idea_digest"] == PINNED_IDEA_DIGEST
    assert document["context_pack"]["pack_cid"] == PINNED_PACK_CID
    assert document["storage"]["current_root_cid"] == PINNED_CURRENT_ROOT_CID
    assert document["final_receipt_chain"]["chain_cid"] == PINNED_CHAIN_CID
    assert document["python_external_client"]["client_cid"] == PINNED_PYTHON_CLIENT_CID
    assert document["generic_mcp_client"]["client_cid"] == PINNED_MCP_CLIENT_CID
    assert document["objective_identity_parity"]["parity_cid"] == PINNED_PARITY_CID
    assert document["authority_bypass"]["proof_cid"] == PINNED_BYPASS_PROOF_CID
    assert document["threat_model"]["threat_model_cid"] == PINNED_THREAT_MODEL_CID
    assert document["trusted_computing_base"]["tcb_inventory_cid"] == (
        PINNED_TCB_INVENTORY_CID
    )
    assert document["security_and_correctness_audit_package"]["audit_package_cid"] == (
        PINNED_AUDIT_PACKAGE_CID
    )
    assert PINNED_PYTHON_CLIENT_CID != PINNED_MCP_CLIENT_CID
    assert package["decision_cid"] == PINNED_DECISION_CID
    assert document["document_cid"] == PINNED_DOCUMENT_CID
    assert document["operator_blocking_task"]["status"] == "typed_blocked"
    trace = hermetic_gate_run_trace()
    assert trace["all_gates_bounded"] is True
    assert trace["all_gate_families_evaluated"] is True
    assert trace["no_hidden_failed_stale_partial_simulated_estimated_or_missing"] is True
    reproduction = reproduce_promotion_or_non_promotion()
    assert reproduction["ok"] is True
    assert reproduction["command_count"] == 8
    idempotency = package["idempotency"]
    assert idempotency["identical"] is True
    assert idempotency["second_attempt_applied"] is False
    assert idempotency["first_decision_cid"] == PINNED_DECISION_CID
    assert refuse_pack_cid_remint(PINNED_PACK_CID) == PINNED_PACK_CID
    assert refuse_threat_model_cid_remint(PINNED_THREAT_MODEL_CID) == (
        PINNED_THREAT_MODEL_CID
    )
    assert refuse_tcb_inventory_cid_remint(PINNED_TCB_INVENTORY_CID) == (
        PINNED_TCB_INVENTORY_CID
    )
    assert refuse_audit_package_cid_remint(PINNED_AUDIT_PACKAGE_CID) == (
        PINNED_AUDIT_PACKAGE_CID
    )
    assert refuse_gate_run_cid_remint(PINNED_GATE_RUN_CID) == PINNED_GATE_RUN_CID
    with pytest.raises(AcceleratePromotionOrNonPromotionError, match="remints"):
        refuse_pack_cid_remint(
            "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        )
    with pytest.raises(AcceleratePromotionOrNonPromotionError, match="cannot complete"):
        refuse_model_completion("frontier_model")
    with pytest.raises(AcceleratePromotionOrNonPromotionError, match="not an authorized"):
        refuse_unauthorized_path("/etc/passwd")
    with pytest.raises(AcceleratePromotionOrNonPromotionError, match="DuckDB"):
        refuse_database_edit(edited=True)
    with pytest.raises(AcceleratePromotionOrNonPromotionError, match="promote"):
        refuse_self_promotion(promoted=True)
    with pytest.raises(AcceleratePromotionOrNonPromotionError, match="canonical"):
        refuse_non_canonical_objective("a different idea")
    with pytest.raises(AcceleratePromotionOrNonPromotionError, match="closed PCPR"):
        refuse_closed_release_claim(claimed=True)
    with pytest.raises(AcceleratePromotionOrNonPromotionError, match="bounded"):
        refuse_unbounded_gate(unbounded=True)
    with pytest.raises(AcceleratePromotionOrNonPromotionError, match="hidden"):
        refuse_hidden_evidence(hidden=True)
    with pytest.raises(AcceleratePromotionOrNonPromotionError, match="live pass"):
        refuse_live_pass(claimed=True)
    assert path_is_authorized(AUTHORIZED_PATH_PREFIXES[0]) is True


def test_current_head_is_rnd_non_promoted_and_not_a_release() -> None:
    verdict = qualify_current_head_promotion_or_non_promotion()
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
    assert verdict.decision_cid == PINNED_DECISION_CID
    assert verdict.document_cid == PINNED_DOCUMENT_CID
    assert verdict.catalog_cid == PINNED_CATALOG_CID
    assert verdict.current_root_cid == PINNED_CURRENT_ROOT_CID
    assert verdict.chain_cid == PINNED_CHAIN_CID
    assert verdict.threat_model_cid == PINNED_THREAT_MODEL_CID
    assert verdict.tcb_inventory_cid == PINNED_TCB_INVENTORY_CID
    assert verdict.audit_package_cid == PINNED_AUDIT_PACKAGE_CID
    assert verdict.python_client_cid == PINNED_PYTHON_CLIENT_CID
    assert verdict.mcp_client_cid == PINNED_MCP_CLIENT_CID
    assert verdict.parity_cid == PINNED_PARITY_CID
    assert verdict.bypass_proof_cid == PINNED_BYPASS_PROOF_CID
    assert verdict.blockers == ()
    section = pcpr_094_receipt_promotion(verdict)
    assert section["promotion_status"] == "rnd_non_promoted"
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_static_probes_show_declared_gate_constraints() -> None:
    probes = {item.probe_id: item for item in current_head_static_probes()}
    assert probes["gate_files_match_generator"].present is True
    assert probes["pyproject_promotion_or_non_promotion_table"].present is True
    assert probes["paths_remain_authorized"].present is True
    assert probes["canonical_objective_bytes_match"].present is True
    assert probes["supervisor_gate_evaluated"].present is True
    assert probes["datasets_gate_evaluated"].present is True
    assert probes["kit_gate_evaluated"].present is True
    assert probes["accelerate_gate_evaluated"].present is True
    assert probes["packaging_gate_evaluated"].present is True
    assert probes["reference_workflow_gate_evaluated"].present is True
    assert probes["interoperability_gate_evaluated"].present is True
    assert probes["hard_safety_gate_evaluated"].present is True
    assert probes["all_gate_families_evaluated"].present is True
    assert probes["all_gates_bounded"].present is True
    assert probes["no_hidden_failed_stale_partial_simulated_estimated_or_missing"].present is True
    assert probes["reproduction_commands_work"].present is True
    assert probes["owner_pack_cid_matches_pin"].present is True
    assert probes["kit_current_root_bound_not_minted"].present is True
    assert probes["chain_cid_bound_not_minted"].present is True
    assert probes["python_client_cid_bound_not_minted"].present is True
    assert probes["mcp_client_cid_bound_not_minted"].present is True
    assert probes["parity_cid_bound_not_minted"].present is True
    assert probes["bypass_proof_cid_bound_not_minted"].present is True
    assert probes["threat_model_cid_bound_not_minted"].present is True
    assert probes["tcb_inventory_cid_bound_not_minted"].present is True
    assert probes["audit_package_cid_bound_not_minted"].present is True
    assert probes["promotion_produced"].present is True
    assert probes["honest_non_promotion_produced"].present is True
    assert probes["gate_run_cid_bound_not_minted"].present is True
    assert probes["idempotent_decision_cid"].present is True
    assert probes["model_assertion_cannot_complete_work"].present is True
    assert probes["datasets_identity_reminted"].present is False
    assert probes["kit_identity_reminted"].present is False
    assert probes["chain_identity_reminted"].present is False
    assert probes["python_client_identity_reminted"].present is False
    assert probes["mcp_client_identity_reminted"].present is False
    assert probes["parity_identity_reminted"].present is False
    assert probes["bypass_identity_reminted"].present is False
    assert probes["threat_model_identity_reminted"].present is False
    assert probes["tcb_inventory_identity_reminted"].present is False
    assert probes["audit_package_identity_reminted"].present is False
    assert probes["closed_release_claimed"].present is False
    assert probes["unbounded_gate_present"].present is False
    assert probes["hidden_evidence_present"].present is False
    assert probes["reproduction_commands_failed"].present is False
    assert probes["live_pass_claimed"].present is False
    assert probes["database_edited"].present is False
    assert probes["live_application"].evidence_kind == "unavailable"
    assert probes["live_client"].evidence_kind == "unavailable"
    for probe in probes.values():
        assert probe.live is False
        assert probe.simulated_represented_as_live is False


def test_committed_files_match_generator() -> None:
    verified = verify_promotion_or_non_promotion_files()
    assert verified["ok"] is True
    catalog = platform_promotion_or_non_promotion_catalog()
    assert catalog["interface"] == "PlatformPromotionOrNonPromotion@1"
    assert catalog["release_claim"] is False
    assert catalog["closed_release_outcome"] is None
    assert catalog["live_application"] is False
    assert catalog["live_client"] is False
    assert catalog["pack_cid"] == PINNED_PACK_CID
    assert catalog["chain_cid"] == PINNED_CHAIN_CID
    assert catalog["threat_model_cid"] == PINNED_THREAT_MODEL_CID
    assert catalog["tcb_inventory_cid"] == PINNED_TCB_INVENTORY_CID
    assert catalog["audit_package_cid"] == PINNED_AUDIT_PACKAGE_CID
    assert catalog["decision_cid"] == PINNED_DECISION_CID
    assert catalog["catalog_cid"] == PINNED_CATALOG_CID
    assert catalog["components"]["ipfs_accelerate_py"]["status"] == "observed"
    pyproject = (_PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'interface = "AcceleratePromotionOrNonPromotionReceipt@1"' in pyproject
    plan = produce_promotion_or_non_promotion()
    assert plan["produced"] is True
    assert plan["live"] is False
    assert plan["closed_release_deferred"] is False
    assert tuple(plan["escalation_order"]) == ESCALATION_ORDER
    with pytest.raises(AcceleratePromotionOrNonPromotionError, match="cannot complete"):
        produce_promotion_or_non_promotion(complete_from="frontier_model")
    with pytest.raises(AcceleratePromotionOrNonPromotionError, match="DuckDB"):
        produce_promotion_or_non_promotion(database_edited=True)
    with pytest.raises(AcceleratePromotionOrNonPromotionError, match="closed PCPR"):
        produce_promotion_or_non_promotion(closed_release_claimed=True)


def test_simulated_live_probe_is_rejected() -> None:
    dummy = "baguqeerabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    with pytest.raises(AcceleratePromotionOrNonPromotionError, match="simulated"):
        qualify_promotion_or_non_promotion(
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
            decision_cid=dummy,
            document_cid=dummy,
            catalog_cid=dummy,
            audit_package_cid=dummy,
            tcb_inventory_cid=dummy,
            threat_model_cid=dummy,
            chain_cid=dummy,
            pack_cid=dummy,
            current_root_cid=dummy,
            objective_cid=dummy,
            idea_digest_cid=dummy,
        )
    with pytest.raises(AcceleratePromotionOrNonPromotionError, match="measured_live"):
        qualify_promotion_or_non_promotion(
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
            decision_cid=dummy,
            document_cid=dummy,
            catalog_cid=dummy,
            audit_package_cid=dummy,
            tcb_inventory_cid=dummy,
            threat_model_cid=dummy,
            chain_cid=dummy,
            pack_cid=dummy,
            current_root_cid=dummy,
            objective_cid=dummy,
            idea_digest_cid=dummy,
        )
