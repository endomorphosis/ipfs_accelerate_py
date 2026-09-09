"""PCPR-090: Accelerate-owned objective-to-release threat model."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.assurance.threat_model import (
    AUTHORIZED_PATH_PREFIXES,
    CLOSED_RELEASE_OUTCOMES,
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID,
    ESCALATION_ORDER,
    G910_SURFACES,
    HERMETIC_CANDIDATE_SUITES,
    INTERFACE,
    OBJECTIVE_KIND,
    OPERATOR_BLOCKING_TASK_ID,
    OutcomeProbe,
    PCPR_090_GOAL_ID,
    PCPR_090_TASK_ID,
    PINNED_BYPASS_PROOF_CID,
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
    PINNED_THREAT_MODEL_CID,
    SCHEMA,
    SEALED_PATH,
    SEALED_PYTHON,
    THREATS,
    TRUSTED_PATH_STAGES,
    AccelerateThreatModelError,
    all_g910_surfaces_bounded,
    current_head_static_probes,
    hermetic_threat_model_trace,
    path_is_authorized,
    pcpr_090_receipt_promotion,
    platform_threat_model_catalog,
    produce_threat_model,
    qualify_current_head_threat_model,
    qualify_threat_model,
    refuse_audit_package_claim,
    refuse_bypass_proof_cid_remint,
    refuse_chain_cid_remint,
    refuse_closed_release_claim,
    refuse_database_edit,
    refuse_mcp_client_cid_remint,
    refuse_model_completion,
    refuse_non_canonical_objective,
    refuse_objective_cid_remint,
    refuse_pack_cid_remint,
    refuse_parity_cid_remint,
    refuse_python_client_cid_remint,
    refuse_self_promotion,
    refuse_tcb_inventory_claim,
    refuse_unauthorized_path,
    refuse_unbounded_threat,
    render_declared_model,
    verify_threat_model_files,
)


_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def test_closed_vocabularies_match_pcpr_090_requirements() -> None:
    assert PCPR_090_TASK_ID == "PCPR-090"
    assert PCPR_090_GOAL_ID == "PCPR-G900"
    assert INTERFACE == "AccelerateThreatModel@1"
    assert SCHEMA == "ipfs_accelerate_py/assurance/threat-model@1"
    assert OBJECTIVE_KIND == "declared_objective_to_release_threat_model"
    assert OPERATOR_BLOCKING_TASK_ID == "pcpr-090-operator-live-threat-model"
    assert "release_candidate_qualified" in CLOSED_RELEASE_OUTCOMES
    assert "rnd_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert HERMETIC_CANDIDATE_SUITES[-1].endswith(
        "test_agent_supervisor_threat_model.py"
    )
    assert SEALED_PATH == "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
    assert SEALED_PYTHON == "/usr/bin/python3.12"
    assert ESCALATION_ORDER[0] == "exact receipt"
    assert ESCALATION_ORDER[-1] == "human decision"
    assert TRUSTED_PATH_STAGES[0] == "objective_submission"
    assert TRUSTED_PATH_STAGES[-1] == "release_decision"
    assert "actors" in G910_SURFACES
    assert "builds" in G910_SURFACES
    assert all_g910_surfaces_bounded() is True
    assert len(THREATS) == 24
    document = render_declared_model()
    assert document["applied"] is False
    assert document["live"] is False
    assert document["release_claim"] is False
    assert document["closed_release_outcome"] is None
    model = document["threat_model"]
    assert model["produced"] is True
    assert model["live"] is False
    assert model["hermetic"] is True
    assert model["change_kind"] == "objective_to_release_threat_model"
    assert model["canonical_service"] == "pcpr-canonical-supervisor-service"
    assert model["authority"] == "intent_not_authority"
    assert model["all_g910_surfaces_bounded"] is True
    assert model["all_threats_bounded"] is True
    assert model["trusted_path_stages_covered"] is True
    assert model["tcb_inventory_deferred"] is True
    assert model["audit_package_deferred"] is True
    assert model["closed_release_deferred"] is True
    assert model["client_self_promoted"] is False
    assert model["model_assertion_completes_work"] is False
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
    assert PINNED_PYTHON_CLIENT_CID != PINNED_MCP_CLIENT_CID
    assert model["threat_model_cid"] == PINNED_THREAT_MODEL_CID
    assert document["document_cid"] == PINNED_DOCUMENT_CID
    assert document["operator_blocking_task"]["status"] == "typed_blocked"
    trace = hermetic_threat_model_trace()
    assert trace["all_threats_bounded"] is True
    assert trace["all_g910_surfaces_bounded"] is True
    assert trace["trusted_path_stages_covered"] is True
    assert trace["tcb_inventory_deferred"] is True
    idempotency = model["idempotency"]
    assert idempotency["identical"] is True
    assert idempotency["second_attempt_applied"] is False
    assert idempotency["first_threat_model_cid"] == PINNED_THREAT_MODEL_CID
    assert refuse_pack_cid_remint(PINNED_PACK_CID) == PINNED_PACK_CID
    assert refuse_objective_cid_remint(PINNED_OBJECTIVE_CID) == PINNED_OBJECTIVE_CID
    assert refuse_chain_cid_remint(PINNED_CHAIN_CID) == PINNED_CHAIN_CID
    assert refuse_python_client_cid_remint(PINNED_PYTHON_CLIENT_CID) == PINNED_PYTHON_CLIENT_CID
    assert refuse_mcp_client_cid_remint(PINNED_MCP_CLIENT_CID) == PINNED_MCP_CLIENT_CID
    assert refuse_parity_cid_remint(PINNED_PARITY_CID) == PINNED_PARITY_CID
    assert refuse_bypass_proof_cid_remint(PINNED_BYPASS_PROOF_CID) == PINNED_BYPASS_PROOF_CID
    with pytest.raises(AccelerateThreatModelError, match="remints"):
        refuse_pack_cid_remint(
            "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        )
    with pytest.raises(AccelerateThreatModelError, match="cannot complete"):
        refuse_model_completion("frontier_model")
    with pytest.raises(AccelerateThreatModelError, match="not an authorized"):
        refuse_unauthorized_path("/etc/passwd")
    with pytest.raises(AccelerateThreatModelError, match="DuckDB"):
        refuse_database_edit(edited=True)
    with pytest.raises(AccelerateThreatModelError, match="promote"):
        refuse_self_promotion(promoted=True)
    with pytest.raises(AccelerateThreatModelError, match="canonical"):
        refuse_non_canonical_objective("a different idea")
    with pytest.raises(AccelerateThreatModelError, match="PCPR-091"):
        refuse_tcb_inventory_claim(claimed=True)
    with pytest.raises(AccelerateThreatModelError, match="PCPR-092"):
        refuse_audit_package_claim(claimed=True)
    with pytest.raises(AccelerateThreatModelError, match="PCPR-093"):
        refuse_closed_release_claim(claimed=True)
    with pytest.raises(AccelerateThreatModelError, match="bounded"):
        refuse_unbounded_threat(unbounded=True)
    assert path_is_authorized(AUTHORIZED_PATH_PREFIXES[0]) is True


def test_current_head_is_rnd_non_promoted_and_not_a_release() -> None:
    verdict = qualify_current_head_threat_model()
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
    assert verdict.threat_model_cid == PINNED_THREAT_MODEL_CID
    assert verdict.document_cid == PINNED_DOCUMENT_CID
    assert verdict.catalog_cid == PINNED_CATALOG_CID
    assert verdict.current_root_cid == PINNED_CURRENT_ROOT_CID
    assert verdict.chain_cid == PINNED_CHAIN_CID
    assert verdict.python_client_cid == PINNED_PYTHON_CLIENT_CID
    assert verdict.mcp_client_cid == PINNED_MCP_CLIENT_CID
    assert verdict.parity_cid == PINNED_PARITY_CID
    assert verdict.bypass_proof_cid == PINNED_BYPASS_PROOF_CID
    assert verdict.blockers == ()
    section = pcpr_090_receipt_promotion(verdict)
    assert section["promotion_status"] == "rnd_non_promoted"
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_static_probes_show_declared_threat_model_constraints() -> None:
    probes = {item.probe_id: item for item in current_head_static_probes()}
    assert probes["threat_model_files_match_generator"].present is True
    assert probes["pyproject_threat_model_table"].present is True
    assert probes["paths_remain_authorized"].present is True
    assert probes["canonical_objective_bytes_match"].present is True
    assert probes["actors_bounded"].present is True
    assert probes["assets_bounded"].present is True
    assert probes["trust_boundaries_bounded"].present is True
    assert probes["state_owners_bounded"].present is True
    assert probes["canonicalization_bounded"].present is True
    assert probes["cids_bounded"].present is True
    assert probes["events_bounded"].present is True
    assert probes["leases_bounded"].present is True
    assert probes["fences_bounded"].present is True
    assert probes["unknown_outcomes_bounded"].present is True
    assert probes["confirmations_bounded"].present is True
    assert probes["proof_admission_bounded"].present is True
    assert probes["dependencies_bounded"].present is True
    assert probes["builds_bounded"].present is True
    assert probes["all_g910_surfaces_bounded"].present is True
    assert probes["all_threats_bounded"].present is True
    assert probes["trusted_path_stages_covered"].present is True
    assert probes["owner_pack_cid_matches_pin"].present is True
    assert probes["kit_current_root_bound_not_minted"].present is True
    assert probes["chain_cid_bound_not_minted"].present is True
    assert probes["python_client_cid_bound_not_minted"].present is True
    assert probes["mcp_client_cid_bound_not_minted"].present is True
    assert probes["parity_cid_bound_not_minted"].present is True
    assert probes["bypass_proof_cid_bound_not_minted"].present is True
    assert probes["tcb_inventory_deferred"].present is True
    assert probes["audit_package_deferred"].present is True
    assert probes["closed_release_deferred"].present is True
    assert probes["idempotent_threat_model_cid"].present is True
    assert probes["model_assertion_cannot_complete_work"].present is True
    assert probes["datasets_identity_reminted"].present is False
    assert probes["kit_identity_reminted"].present is False
    assert probes["chain_identity_reminted"].present is False
    assert probes["python_client_identity_reminted"].present is False
    assert probes["mcp_client_identity_reminted"].present is False
    assert probes["parity_identity_reminted"].present is False
    assert probes["bypass_identity_reminted"].present is False
    assert probes["tcb_inventory_claimed"].present is False
    assert probes["audit_package_claimed"].present is False
    assert probes["closed_release_claimed"].present is False
    assert probes["unbounded_threat_present"].present is False
    assert probes["database_edited"].present is False
    assert probes["live_application"].evidence_kind == "unavailable"
    assert probes["live_client"].evidence_kind == "unavailable"
    for probe in probes.values():
        assert probe.live is False
        assert probe.simulated_represented_as_live is False


def test_committed_files_match_generator() -> None:
    verified = verify_threat_model_files()
    assert verified["ok"] is True
    catalog = platform_threat_model_catalog()
    assert catalog["interface"] == "PlatformThreatModel@1"
    assert catalog["release_claim"] is False
    assert catalog["closed_release_outcome"] is None
    assert catalog["live_application"] is False
    assert catalog["live_client"] is False
    assert catalog["pack_cid"] == PINNED_PACK_CID
    assert catalog["chain_cid"] == PINNED_CHAIN_CID
    assert catalog["threat_model_cid"] == PINNED_THREAT_MODEL_CID
    assert catalog["catalog_cid"] == PINNED_CATALOG_CID
    assert catalog["components"]["ipfs_accelerate_py"]["status"] == "observed"
    pyproject = (_PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'interface = "AccelerateThreatModel@1"' in pyproject
    plan = produce_threat_model()
    assert plan["produced"] is True
    assert plan["live"] is False
    assert tuple(plan["escalation_order"]) == ESCALATION_ORDER
    with pytest.raises(AccelerateThreatModelError, match="cannot complete"):
        produce_threat_model(complete_from="frontier_model")
    with pytest.raises(AccelerateThreatModelError, match="DuckDB"):
        produce_threat_model(database_edited=True)
    with pytest.raises(AccelerateThreatModelError, match="PCPR-091"):
        produce_threat_model(tcb_claimed=True)


def test_simulated_live_probe_is_rejected() -> None:
    dummy = "baguqeerabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    with pytest.raises(AccelerateThreatModelError, match="simulated"):
        qualify_threat_model(
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
            threat_model_cid=dummy,
            document_cid=dummy,
            catalog_cid=dummy,
            chain_cid=dummy,
            pack_cid=dummy,
            current_root_cid=dummy,
            objective_cid=dummy,
            idea_digest_cid=dummy,
        )
    with pytest.raises(AccelerateThreatModelError, match="measured_live"):
        qualify_threat_model(
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
            threat_model_cid=dummy,
            document_cid=dummy,
            catalog_cid=dummy,
            chain_cid=dummy,
            pack_cid=dummy,
            current_root_cid=dummy,
            objective_cid=dummy,
            idea_digest_cid=dummy,
        )
