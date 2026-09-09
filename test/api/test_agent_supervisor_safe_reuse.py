"""PCPR-067: Accelerate-owned eligible reuse admission."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.assurance.safe_reuse import (
    AUTHORIZED_PATH_PREFIXES,
    CLOSED_RELEASE_OUTCOMES,
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID,
    ELIGIBLE_REUSE,
    ESCALATION_ORDER,
    HERMETIC_CANDIDATE_SUITES,
    INTERFACE,
    OPERATOR_BLOCKING_TASK_ID,
    OBJECTIVE_KIND,
    OutcomeProbe,
    PCPR_066_CHANGE_CID,
    PCPR_067_GOAL_ID,
    PCPR_067_TASK_ID,
    PINNED_CATALOG_CID,
    PINNED_CURRENT_ROOT_CID,
    PINNED_DOCUMENT_CID,
    PINNED_IDEA_DIGEST,
    PINNED_OBJECTIVE_CID,
    PINNED_PACK_CID,
    PINNED_PATCH_CID,
    PINNED_REUSE_CID,
    PINNED_ROUTE_CID,
    PINNED_RUN_CID,
    PROTOCOL_INTERFACE,
    SCHEMA,
    SEALED_PATH,
    SEALED_PYTHON,
    AccelerateSafeReuseError,
    current_head_static_probes,
    path_is_authorized,
    pcpr_067_receipt_promotion,
    platform_safe_reuse_catalog,
    produce_safe_reuse,
    qualify_current_head_safe_reuse,
    qualify_safe_reuse,
    refuse_change_cid_remint,
    refuse_live_reuse,
    refuse_model_completion,
    refuse_pack_cid_remint,
    refuse_patch_cid_remint,
    refuse_relevant_interface_change,
    refuse_reuse_not_demonstrated,
    refuse_route_cid_remint,
    refuse_run_cid_remint,
    refuse_stale_reuse,
    refuse_unauthorized_path,
    refuse_whole_plan_regeneration,
    render_declared_reuse,
    verify_safe_reuse_files,
)


_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def test_closed_vocabularies_match_pcpr_067_requirements() -> None:
    assert PCPR_067_TASK_ID == "PCPR-067"
    assert PCPR_067_GOAL_ID == "PCPR-G700"
    assert INTERFACE == "AccelerateSafeReuse@1"
    assert SCHEMA == "ipfs_accelerate_py/assurance/safe-reuse@1"
    assert OBJECTIVE_KIND == "declared_safe_reuse"
    assert OPERATOR_BLOCKING_TASK_ID == "pcpr-067-operator-live-safe-reuse"
    assert PROTOCOL_INTERFACE == "LogicProviderProtocol@2"
    assert "release_candidate_qualified" in CLOSED_RELEASE_OUTCOMES
    assert "rnd_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert HERMETIC_CANDIDATE_SUITES[-1].endswith(
        "test_agent_supervisor_safe_reuse.py"
    )
    assert SEALED_PATH == "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
    assert SEALED_PYTHON == "/usr/bin/python3.12"
    assert ESCALATION_ORDER[0] == "exact receipt"
    assert ESCALATION_ORDER[3] == "selected tests"
    assert ESCALATION_ORDER[-1] == "human decision"
    document = render_declared_reuse()
    assert document["applied"] is False
    assert document["live"] is False
    assert document["release_claim"] is False
    assert document["closed_release_outcome"] is None
    reuse = document["safe_reuse"]
    assert reuse["produced"] is True
    assert reuse["live"] is False
    assert reuse["hermetic"] is True
    assert reuse["reuse_kind"] == "eligible_unrelated_change_reuse"
    assert reuse["relevant_interface_change"] is False
    assert reuse["whole_plan_regeneration_required"] is False
    assert reuse["reuse_demonstrated"] is True
    assert reuse["live_reuse"] is False
    assert reuse["eligible_reuse_preserved"] is True
    assert reuse["model_assertion_completes_work"] is False
    assert list(reuse["impacted_cone"]) == []
    assert list(reuse["stale_identities"]) == []
    assert list(reuse["eligible_reuse"]) == list(ELIGIBLE_REUSE)
    assert [item["identity"] for item in reuse["reused"]] == list(ELIGIBLE_REUSE)
    assert document["duckdb_or_quack_state_written"] is False
    assert document["objective_cid"] == PINNED_OBJECTIVE_CID
    assert document["idea_digest"] == PINNED_IDEA_DIGEST
    assert document["context_pack"]["pack_cid"] == PINNED_PACK_CID
    assert document["context_pack"]["reused"] is True
    assert document["storage"]["current_root_cid"] == PINNED_CURRENT_ROOT_CID
    assert document["route"]["route_cid"] == PINNED_ROUTE_CID
    assert document["bounded_patch"]["patch_cid"] == PINNED_PATCH_CID
    assert document["selected_tests"]["run_cid"] == PINNED_RUN_CID
    assert document["unrelated_state_change"]["change_cid"] == PCPR_066_CHANGE_CID
    assert reuse["reuse_cid"] == PINNED_REUSE_CID
    assert document["document_cid"] == PINNED_DOCUMENT_CID
    assert document["operator_blocking_task"]["status"] == "typed_blocked"
    assert refuse_pack_cid_remint(PINNED_PACK_CID) == PINNED_PACK_CID
    assert refuse_route_cid_remint(PINNED_ROUTE_CID) == PINNED_ROUTE_CID
    assert refuse_patch_cid_remint(PINNED_PATCH_CID) == PINNED_PATCH_CID
    assert refuse_run_cid_remint(PINNED_RUN_CID) == PINNED_RUN_CID
    assert refuse_change_cid_remint(PCPR_066_CHANGE_CID) == PCPR_066_CHANGE_CID
    with pytest.raises(AccelerateSafeReuseError, match="remints"):
        refuse_pack_cid_remint(
            "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        )
    with pytest.raises(AccelerateSafeReuseError, match="cannot complete"):
        refuse_model_completion("frontier_model")
    with pytest.raises(AccelerateSafeReuseError, match="not an authorized"):
        refuse_unauthorized_path("/etc/passwd")
    with pytest.raises(AccelerateSafeReuseError, match="relevant"):
        refuse_relevant_interface_change(relevant=True)
    with pytest.raises(AccelerateSafeReuseError, match="whole-plan"):
        refuse_whole_plan_regeneration(required=True)
    with pytest.raises(AccelerateSafeReuseError, match="must demonstrate"):
        refuse_reuse_not_demonstrated(demonstrated=False)
    with pytest.raises(AccelerateSafeReuseError, match="live reuse"):
        refuse_live_reuse(live=True)
    with pytest.raises(AccelerateSafeReuseError, match="stale"):
        refuse_stale_reuse(stale=["DatasetsContextPack@1"])
    assert path_is_authorized(AUTHORIZED_PATH_PREFIXES[0]) is True


def test_current_head_is_rnd_non_promoted_and_not_a_release() -> None:
    verdict = qualify_current_head_safe_reuse()
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
    assert verdict.reuse_cid == PINNED_REUSE_CID
    assert verdict.document_cid == PINNED_DOCUMENT_CID
    assert verdict.catalog_cid == PINNED_CATALOG_CID
    assert verdict.current_root_cid == PINNED_CURRENT_ROOT_CID
    assert verdict.route_cid == PINNED_ROUTE_CID
    assert verdict.patch_cid == PINNED_PATCH_CID
    assert verdict.run_cid == PINNED_RUN_CID
    assert verdict.change_cid == PCPR_066_CHANGE_CID
    assert verdict.blockers == ()
    section = pcpr_067_receipt_promotion(verdict)
    assert section["promotion_status"] == "rnd_non_promoted"
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_static_probes_show_declared_reuse_constraints() -> None:
    probes = {item.probe_id: item for item in current_head_static_probes()}
    assert probes["reuse_files_match_generator"].present is True
    assert probes["pyproject_safe_reuse_table"].present is True
    assert probes["paths_remain_authorized"].present is True
    assert probes["owner_pack_cid_matches_pin"].present is True
    assert probes["kit_current_root_bound_not_minted"].present is True
    assert probes["route_cid_bound_not_minted"].present is True
    assert probes["patch_cid_bound_not_minted"].present is True
    assert probes["run_cid_bound_not_minted"].present is True
    assert probes["change_cid_bound_not_minted"].present is True
    assert probes["impacted_cone_empty"].present is True
    assert probes["eligible_reuse_demonstrated_hermetically"].present is True
    assert probes["stale_identities_empty"].present is True
    assert probes["protocol_identity_not_reminted"].present is True
    assert probes["model_assertion_cannot_complete_work"].present is True
    assert probes["datasets_identity_reminted"].present is False
    assert probes["kit_identity_reminted"].present is False
    assert probes["model_assertion_completed_work"].present is False
    assert probes["relevant_interface_change_accepted"].present is False
    assert probes["live_reuse_claimed"].present is False
    assert probes["stale_reuse_accepted"].present is False
    assert probes["live_application"].evidence_kind == "unavailable"
    assert probes["live_reuse"].evidence_kind == "unavailable"
    for probe in probes.values():
        assert probe.live is False
        assert probe.simulated_represented_as_live is False


def test_committed_files_match_generator() -> None:
    verified = verify_safe_reuse_files()
    assert verified["ok"] is True
    catalog = platform_safe_reuse_catalog()
    assert catalog["interface"] == "PlatformSafeReuse@1"
    assert catalog["release_claim"] is False
    assert catalog["closed_release_outcome"] is None
    assert catalog["live_application"] is False
    assert catalog["live_reuse"] is False
    assert catalog["pack_cid"] == PINNED_PACK_CID
    assert catalog["reuse_cid"] == PINNED_REUSE_CID
    assert catalog["catalog_cid"] == PINNED_CATALOG_CID
    assert catalog["components"]["ipfs_accelerate_py"]["status"] == "observed"
    pyproject = (_PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'interface = "AccelerateSafeReuse@1"' in pyproject
    plan = produce_safe_reuse()
    assert plan["produced"] is True
    assert plan["live"] is False
    assert plan["reuse_demonstrated"] is True
    assert tuple(plan["escalation_order"]) == ESCALATION_ORDER
    with pytest.raises(AccelerateSafeReuseError, match="cannot complete"):
        produce_safe_reuse(complete_from="frontier_model")
    with pytest.raises(AccelerateSafeReuseError, match="relevant"):
        produce_safe_reuse(relevant=True)
    with pytest.raises(AccelerateSafeReuseError, match="must demonstrate"):
        produce_safe_reuse(reuse_demonstrated=False)
    with pytest.raises(AccelerateSafeReuseError, match="live reuse"):
        produce_safe_reuse(live_reuse=True)


def test_simulated_live_probe_is_rejected() -> None:
    dummy = "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    with pytest.raises(AccelerateSafeReuseError, match="simulated"):
        qualify_safe_reuse(
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
            reuse_cid=dummy,
            document_cid=dummy,
            catalog_cid=dummy,
            pack_cid=dummy,
            current_root_cid=dummy,
            route_cid=dummy,
            patch_cid=dummy,
            run_cid=dummy,
            change_cid=dummy,
            objective_cid=dummy,
            idea_digest_cid=dummy,
        )
    with pytest.raises(AccelerateSafeReuseError, match="measured_live"):
        qualify_safe_reuse(
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
            reuse_cid=dummy,
            document_cid=dummy,
            catalog_cid=dummy,
            pack_cid=dummy,
            current_root_cid=dummy,
            route_cid=dummy,
            patch_cid=dummy,
            run_cid=dummy,
            change_cid=dummy,
            objective_cid=dummy,
            idea_digest_cid=dummy,
        )
