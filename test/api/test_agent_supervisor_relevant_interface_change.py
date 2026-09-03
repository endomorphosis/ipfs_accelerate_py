"""PCPR-068: Accelerate-owned relevant interface change."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.assurance.relevant_interface_change import (
    AUTHORIZED_PATH_PREFIXES,
    CLOSED_RELEASE_OUTCOMES,
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID,
    ESCALATION_ORDER,
    HERMETIC_CANDIDATE_SUITES,
    IMPACTED_CONE,
    INTERFACE,
    OPERATOR_BLOCKING_TASK_ID,
    OBJECTIVE_KIND,
    OutcomeProbe,
    PCPR_068_GOAL_ID,
    PCPR_068_TASK_ID,
    PINNED_CATALOG_CID,
    PINNED_CHANGE_CID,
    PINNED_CURRENT_ROOT_CID,
    PINNED_DOCUMENT_CID,
    PINNED_IDEA_DIGEST,
    PINNED_OBJECTIVE_CID,
    PINNED_PACK_CID,
    PINNED_PATCH_CID,
    PINNED_ROUTE_CID,
    PINNED_RUN_CID,
    PROTOCOL_INTERFACE,
    SCHEMA,
    SEALED_PATH,
    SEALED_PYTHON,
    RELEVANT_CHANGE_PATHS,
    STALE_IDENTITIES,
    AccelerateRelevantInterfaceChangeError,
    current_head_static_probes,
    path_is_authorized,
    pcpr_068_receipt_promotion,
    platform_relevant_interface_change_catalog,
    produce_relevant_interface_change,
    qualify_current_head_relevant_interface_change,
    qualify_relevant_interface_change,
    refuse_model_completion,
    refuse_pack_cid_remint,
    refuse_patch_cid_remint,
    refuse_relevant_interface_change,
    refuse_reuse_as_demonstrated,
    refuse_route_cid_remint,
    refuse_run_cid_remint,
    refuse_unauthorized_path,
    refuse_whole_plan_regeneration,
    render_declared_change,
    verify_relevant_interface_change_files,
)


_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def test_closed_vocabularies_match_pcpr_068_requirements() -> None:
    assert PCPR_068_TASK_ID == "PCPR-068"
    assert PCPR_068_GOAL_ID == "PCPR-G700"
    assert INTERFACE == "AccelerateRelevantInterfaceChange@1"
    assert SCHEMA == "ipfs_accelerate_py/assurance/relevant-interface-change@1"
    assert OBJECTIVE_KIND == "declared_relevant_interface_change"
    assert OPERATOR_BLOCKING_TASK_ID == (
        "pcpr-068-operator-live-relevant-interface-change"
    )
    assert PROTOCOL_INTERFACE == "LogicProviderProtocol@2"
    assert "release_candidate_qualified" in CLOSED_RELEASE_OUTCOMES
    assert "rnd_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert HERMETIC_CANDIDATE_SUITES[-1].endswith(
        "test_agent_supervisor_relevant_interface_change.py"
    )
    assert SEALED_PATH == "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
    assert SEALED_PYTHON == "/usr/bin/python3.12"
    assert ESCALATION_ORDER[0] == "exact receipt"
    assert ESCALATION_ORDER[3] == "selected tests"
    assert ESCALATION_ORDER[-1] == "human decision"
    document = render_declared_change()
    assert document["applied"] is False
    assert document["live"] is False
    assert document["release_claim"] is False
    assert document["closed_release_outcome"] is None
    change = document["relevant_interface_change"]
    assert change["produced"] is True
    assert change["live"] is False
    assert change["hermetic"] is True
    assert change["change_kind"] == "relevant_interface"
    assert change["relevant_interface_change"] is True
    assert change["adds_protocol_operation"] is True
    assert change["remints_protocol"] is False
    assert change["whole_plan_regeneration_required"] is False
    assert change["reuse_demonstrated"] is False
    assert change["eligible_reuse_preserved"] is True
    assert change["model_assertion_completes_work"] is False
    assert list(change["impacted_cone"]) == list(IMPACTED_CONE)
    assert list(change["stale_identities"]) == list(STALE_IDENTITIES)
    assert list(change["relevant_change_paths"]) == list(RELEVANT_CHANGE_PATHS)
    assert document["duckdb_or_quack_state_written"] is False
    assert document["objective_cid"] == PINNED_OBJECTIVE_CID
    assert document["idea_digest"] == PINNED_IDEA_DIGEST
    assert document["context_pack"]["pack_cid"] == PINNED_PACK_CID
    assert document["storage"]["current_root_cid"] == PINNED_CURRENT_ROOT_CID
    assert document["route"]["route_cid"] == PINNED_ROUTE_CID
    assert document["bounded_patch"]["patch_cid"] == PINNED_PATCH_CID
    assert document["selected_tests"]["run_cid"] == PINNED_RUN_CID
    assert change["change_cid"] == PINNED_CHANGE_CID
    assert document["document_cid"] == PINNED_DOCUMENT_CID
    assert document["operator_blocking_task"]["status"] == "typed_blocked"
    assert refuse_pack_cid_remint(PINNED_PACK_CID) == PINNED_PACK_CID
    assert refuse_route_cid_remint(PINNED_ROUTE_CID) == PINNED_ROUTE_CID
    assert refuse_patch_cid_remint(PINNED_PATCH_CID) == PINNED_PATCH_CID
    assert refuse_run_cid_remint(PINNED_RUN_CID) == PINNED_RUN_CID
    with pytest.raises(AccelerateRelevantInterfaceChangeError, match="remints"):
        refuse_pack_cid_remint(
            "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        )
    with pytest.raises(AccelerateRelevantInterfaceChangeError, match="cannot complete"):
        refuse_model_completion("frontier_model")
    with pytest.raises(AccelerateRelevantInterfaceChangeError, match="not an authorized"):
        refuse_unauthorized_path("/etc/passwd")
    with pytest.raises(AccelerateRelevantInterfaceChangeError, match="relevant"):
        refuse_relevant_interface_change(relevant=False)
    with pytest.raises(AccelerateRelevantInterfaceChangeError, match="whole-plan"):
        refuse_whole_plan_regeneration(required=True)
    with pytest.raises(AccelerateRelevantInterfaceChangeError, match="PCPR-069"):
        refuse_reuse_as_demonstrated(demonstrated=True)
    assert path_is_authorized(AUTHORIZED_PATH_PREFIXES[0]) is True


def test_current_head_is_rnd_non_promoted_and_not_a_release() -> None:
    verdict = qualify_current_head_relevant_interface_change()
    assert verdict.promotion_status == "rnd_non_promoted"
    assert verdict.supervisor_disposition == "supervisor_non_promoted"
    assert verdict.closed_release_outcome is None
    assert verdict.release_claim is False
    assert verdict.completion_authoritative is False
    assert verdict.contracts_frozen is False
    assert verdict.duckdb_or_quack_state_written is False
    assert verdict.sibling_source_required is False
    assert verdict.live_application is False
    assert verdict.live_reuse is False
    assert verdict.operator_blocking_task == OPERATOR_BLOCKING_TASK_ID
    assert verdict.simulated_results_represented_as_live is False
    assert verdict.this_task_created_competing_authority is False
    assert verdict.promotion_status not in CLOSED_RELEASE_OUTCOMES
    assert verdict.verdict_cid.startswith("baguqeera")
    assert verdict.verdict_cid == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert verdict.pack_cid == PINNED_PACK_CID
    assert verdict.change_cid == PINNED_CHANGE_CID
    assert verdict.document_cid == PINNED_DOCUMENT_CID
    assert verdict.catalog_cid == PINNED_CATALOG_CID
    assert verdict.current_root_cid == PINNED_CURRENT_ROOT_CID
    assert verdict.route_cid == PINNED_ROUTE_CID
    assert verdict.patch_cid == PINNED_PATCH_CID
    assert verdict.run_cid == PINNED_RUN_CID
    assert verdict.blockers == ()
    section = pcpr_068_receipt_promotion(verdict)
    assert section["promotion_status"] == "rnd_non_promoted"
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_static_probes_show_declared_change_constraints() -> None:
    probes = {item.probe_id: item for item in current_head_static_probes()}
    assert probes["change_files_match_generator"].present is True
    assert probes["pyproject_relevant_interface_change_table"].present is True
    assert probes["paths_remain_authorized"].present is True
    assert probes["owner_pack_cid_matches_pin"].present is True
    assert probes["kit_current_root_bound_not_minted"].present is True
    assert probes["route_cid_bound_not_minted"].present is True
    assert probes["patch_cid_bound_not_minted"].present is True
    assert probes["run_cid_bound_not_minted"].present is True
    assert probes["change_is_relevant_interface"].present is True
    assert probes["impacted_cone_nonempty"].present is True
    assert probes["protocol_identity_not_reminted"].present is True
    assert probes["protocol_operation_added"].present is True
    assert probes["stale_identities_recorded"].present is True
    assert probes["eligible_reuse_not_demonstrated"].present is True
    assert probes["model_assertion_cannot_complete_work"].present is True
    assert probes["datasets_identity_reminted"].present is False
    assert probes["kit_identity_reminted"].present is False
    assert probes["model_assertion_completed_work"].present is False
    assert probes["unrelated_change_accepted_as_relevant"].present is False
    assert probes["reuse_claimed_as_demonstrated"].present is False
    assert probes["live_application"].evidence_kind == "unavailable"
    assert probes["live_reuse"].evidence_kind == "unavailable"
    for probe in probes.values():
        assert probe.live is False
        assert probe.simulated_represented_as_live is False


def test_committed_files_match_generator() -> None:
    verified = verify_relevant_interface_change_files()
    assert verified["ok"] is True
    catalog = platform_relevant_interface_change_catalog()
    assert catalog["interface"] == "PlatformRelevantInterfaceChange@1"
    assert catalog["release_claim"] is False
    assert catalog["closed_release_outcome"] is None
    assert catalog["live_application"] is False
    assert catalog["live_reuse"] is False
    assert catalog["pack_cid"] == PINNED_PACK_CID
    assert catalog["change_cid"] == PINNED_CHANGE_CID
    assert catalog["catalog_cid"] == PINNED_CATALOG_CID
    assert catalog["components"]["ipfs_accelerate_py"]["status"] == "observed"
    pyproject = (_PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'interface = "AccelerateRelevantInterfaceChange@1"' in pyproject
    plan = produce_relevant_interface_change()
    assert plan["produced"] is True
    assert plan["live"] is False
    assert tuple(plan["escalation_order"]) == ESCALATION_ORDER
    with pytest.raises(AccelerateRelevantInterfaceChangeError, match="cannot complete"):
        produce_relevant_interface_change(complete_from="frontier_model")
    with pytest.raises(AccelerateRelevantInterfaceChangeError, match="relevant"):
        produce_relevant_interface_change(relevant=False)
    with pytest.raises(AccelerateRelevantInterfaceChangeError, match="PCPR-069"):
        produce_relevant_interface_change(reuse_demonstrated=True)


def test_simulated_live_probe_is_rejected() -> None:
    with pytest.raises(AccelerateRelevantInterfaceChangeError, match="simulated"):
        qualify_relevant_interface_change(
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
            change_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            document_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            catalog_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            pack_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            current_root_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            route_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            patch_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            run_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            objective_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            idea_digest_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
    with pytest.raises(AccelerateRelevantInterfaceChangeError, match="measured_live"):
        qualify_relevant_interface_change(
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
            change_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            document_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            catalog_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            pack_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            current_root_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            route_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            patch_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            run_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            objective_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            idea_digest_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
