"""PCPR-069: Accelerate-owned stale rejection and PlanDelta."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.assurance.stale_rejection import (
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
    PCPR_069_GOAL_ID,
    PCPR_069_TASK_ID,
    PINNED_CATALOG_CID,
    PINNED_CURRENT_ROOT_CID,
    PINNED_DELTA_CID,
    PINNED_DOCUMENT_CID,
    PINNED_IDEA_DIGEST,
    PINNED_INTERFACE_CID,
    PINNED_OBJECTIVE_CID,
    PINNED_PACK_CID,
    PINNED_PATCH_CID,
    PINNED_ROUTE_CID,
    PINNED_RUN_CID,
    PLAN_EPOCH,
    PROTOCOL_INTERFACE,
    REFILL_TASKS,
    SCHEMA,
    SEALED_PATH,
    SEALED_PYTHON,
    STALE_IDENTITIES,
    AccelerateStaleRejectionError,
    current_head_static_probes,
    path_is_authorized,
    pcpr_069_receipt_promotion,
    platform_stale_rejection_catalog,
    produce_stale_rejection_and_plan_delta,
    qualify_current_head_stale_rejection,
    qualify_stale_rejection,
    refuse_history_mutation,
    refuse_interface_cid_remint,
    refuse_model_completion,
    refuse_pack_cid_remint,
    refuse_patch_cid_remint,
    refuse_route_cid_remint,
    refuse_run_cid_remint,
    refuse_stale_as_current,
    refuse_unauthorized_path,
    refuse_unaffected_as_stale,
    refuse_whole_plan_regeneration,
    reject_stale_identities,
    render_declared_delta,
    verify_stale_rejection_files,
)


_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def test_closed_vocabularies_match_pcpr_069_requirements() -> None:
    assert PCPR_069_TASK_ID == "PCPR-069"
    assert PCPR_069_GOAL_ID == "PCPR-G700"
    assert INTERFACE == "AccelerateStaleRejectionAndPlanDelta@1"
    assert SCHEMA == "ipfs_accelerate_py/assurance/stale-rejection-and-plan-delta@1"
    assert OBJECTIVE_KIND == "declared_stale_rejection_and_plan_delta"
    assert OPERATOR_BLOCKING_TASK_ID == (
        "pcpr-069-operator-live-stale-rejection-and-plan-delta"
    )
    assert PROTOCOL_INTERFACE == "LogicProviderProtocol@2"
    assert "release_candidate_qualified" in CLOSED_RELEASE_OUTCOMES
    assert "rnd_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert HERMETIC_CANDIDATE_SUITES[-1].endswith(
        "test_agent_supervisor_stale_rejection.py"
    )
    assert SEALED_PATH == "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
    assert SEALED_PYTHON == "/usr/bin/python3.12"
    assert ESCALATION_ORDER[0] == "exact receipt"
    assert ESCALATION_ORDER[3] == "selected tests"
    assert ESCALATION_ORDER[-1] == "human decision"
    document = render_declared_delta()
    assert document["applied"] is False
    assert document["live"] is False
    assert document["release_claim"] is False
    assert document["closed_release_outcome"] is None
    delta = document["stale_rejection"]
    assert delta["produced"] is True
    assert delta["live"] is False
    assert delta["hermetic"] is True
    assert delta["change_kind"] == "stale_rejection_and_plan_delta"
    assert delta["stale_rejected"] is True
    assert delta["unaffected_completion_preserved"] is True
    assert delta["plan_delta_produced"] is True
    assert delta["epoch_incremented"] is True
    assert delta["plan_epoch"] == PLAN_EPOCH
    assert delta["history_mutated"] is False
    assert delta["whole_plan_regeneration_required"] is False
    assert delta["remints_protocol"] is False
    assert delta["adds_protocol_operation"] is False
    assert delta["model_assertion_completes_work"] is False
    assert list(delta["impacted_cone"]) == list(IMPACTED_CONE)
    assert list(delta["stale_identities"]) == list(STALE_IDENTITIES)
    assert list(delta["refill"]["task_ids"]) == list(REFILL_TASKS)
    assert delta["refill"]["within_bounds"] is True
    assert document["duckdb_or_quack_state_written"] is False
    assert document["objective_cid"] == PINNED_OBJECTIVE_CID
    assert document["idea_digest"] == PINNED_IDEA_DIGEST
    assert document["context_pack"]["pack_cid"] == PINNED_PACK_CID
    assert document["context_pack"]["rejected"] is True
    assert document["storage"]["current_root_cid"] == PINNED_CURRENT_ROOT_CID
    assert document["route"]["route_cid"] == PINNED_ROUTE_CID
    assert document["bounded_patch"]["patch_cid"] == PINNED_PATCH_CID
    assert document["selected_tests"]["run_cid"] == PINNED_RUN_CID
    assert delta["interface_cid"] == PINNED_INTERFACE_CID
    assert delta["delta_cid"] == PINNED_DELTA_CID
    assert document["document_cid"] == PINNED_DOCUMENT_CID
    assert document["operator_blocking_task"]["status"] == "typed_blocked"
    assert refuse_pack_cid_remint(PINNED_PACK_CID) == PINNED_PACK_CID
    assert refuse_route_cid_remint(PINNED_ROUTE_CID) == PINNED_ROUTE_CID
    assert refuse_patch_cid_remint(PINNED_PATCH_CID) == PINNED_PATCH_CID
    assert refuse_run_cid_remint(PINNED_RUN_CID) == PINNED_RUN_CID
    assert refuse_interface_cid_remint(PINNED_INTERFACE_CID) == PINNED_INTERFACE_CID
    with pytest.raises(AccelerateStaleRejectionError, match="remints"):
        refuse_pack_cid_remint(
            "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        )
    with pytest.raises(AccelerateStaleRejectionError, match="cannot complete"):
        refuse_model_completion("frontier_model")
    with pytest.raises(AccelerateStaleRejectionError, match="not an authorized"):
        refuse_unauthorized_path("/etc/passwd")
    with pytest.raises(AccelerateStaleRejectionError, match="admitted as current"):
        refuse_stale_as_current(
            identity=STALE_IDENTITIES[0], admitted_as_current=True
        )
    with pytest.raises(AccelerateStaleRejectionError, match="must be preserved"):
        refuse_unaffected_as_stale(
            identity="tests/unit/test_pcpr_017_solver_qualification.py",
            rejected=True,
        )
    with pytest.raises(AccelerateStaleRejectionError, match="whole-plan"):
        refuse_whole_plan_regeneration(required=True)
    with pytest.raises(AccelerateStaleRejectionError, match="history"):
        refuse_history_mutation(mutated=True)
    assert path_is_authorized(AUTHORIZED_PATH_PREFIXES[0]) is True
    rejected = reject_stale_identities()
    assert rejected["stale_rejected"] is True
    assert rejected["live"] is False


def test_current_head_is_rnd_non_promoted_and_not_a_release() -> None:
    verdict = qualify_current_head_stale_rejection()
    assert verdict.promotion_status == "rnd_non_promoted"
    assert verdict.supervisor_disposition == "supervisor_non_promoted"
    assert verdict.closed_release_outcome is None
    assert verdict.release_claim is False
    assert verdict.completion_authoritative is False
    assert verdict.contracts_frozen is False
    assert verdict.duckdb_or_quack_state_written is False
    assert verdict.sibling_source_required is False
    assert verdict.live_application is False
    assert verdict.live_plan_delta is False
    assert verdict.operator_blocking_task == OPERATOR_BLOCKING_TASK_ID
    assert verdict.simulated_results_represented_as_live is False
    assert verdict.this_task_created_competing_authority is False
    assert verdict.promotion_status not in CLOSED_RELEASE_OUTCOMES
    assert verdict.verdict_cid.startswith("baguqeera")
    assert verdict.verdict_cid == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert verdict.pack_cid == PINNED_PACK_CID
    assert verdict.delta_cid == PINNED_DELTA_CID
    assert verdict.document_cid == PINNED_DOCUMENT_CID
    assert verdict.catalog_cid == PINNED_CATALOG_CID
    assert verdict.current_root_cid == PINNED_CURRENT_ROOT_CID
    assert verdict.route_cid == PINNED_ROUTE_CID
    assert verdict.patch_cid == PINNED_PATCH_CID
    assert verdict.run_cid == PINNED_RUN_CID
    assert verdict.interface_cid == PINNED_INTERFACE_CID
    assert verdict.blockers == ()
    section = pcpr_069_receipt_promotion(verdict)
    assert section["promotion_status"] == "rnd_non_promoted"
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_static_probes_show_declared_delta_constraints() -> None:
    probes = {item.probe_id: item for item in current_head_static_probes()}
    assert probes["delta_files_match_generator"].present is True
    assert probes["pyproject_stale_rejection_table"].present is True
    assert probes["paths_remain_authorized"].present is True
    assert probes["owner_pack_cid_matches_pin"].present is True
    assert probes["kit_current_root_bound_not_minted"].present is True
    assert probes["route_cid_bound_not_minted"].present is True
    assert probes["patch_cid_bound_not_minted"].present is True
    assert probes["run_cid_bound_not_minted"].present is True
    assert probes["interface_cid_bound_not_minted"].present is True
    assert probes["stale_identities_rejected"].present is True
    assert probes["unaffected_completion_preserved"].present is True
    assert probes["plan_delta_produced"].present is True
    assert probes["plan_epoch_incremented"].present is True
    assert probes["refill_within_bounds"].present is True
    assert probes["history_not_mutated"].present is True
    assert probes["model_assertion_cannot_complete_work"].present is True
    assert probes["datasets_identity_reminted"].present is False
    assert probes["kit_identity_reminted"].present is False
    assert probes["model_assertion_completed_work"].present is False
    assert probes["stale_identity_admitted_as_current"].present is False
    assert probes["unaffected_marked_stale"].present is False
    assert probes["history_mutated"].present is False
    assert probes["live_application"].evidence_kind == "unavailable"
    assert probes["live_plan_delta"].evidence_kind == "unavailable"
    for probe in probes.values():
        assert probe.live is False
        assert probe.simulated_represented_as_live is False


def test_committed_files_match_generator() -> None:
    verified = verify_stale_rejection_files()
    assert verified["ok"] is True
    catalog = platform_stale_rejection_catalog()
    assert catalog["interface"] == "PlatformStaleRejectionAndPlanDelta@1"
    assert catalog["release_claim"] is False
    assert catalog["closed_release_outcome"] is None
    assert catalog["live_application"] is False
    assert catalog["live_plan_delta"] is False
    assert catalog["pack_cid"] == PINNED_PACK_CID
    assert catalog["delta_cid"] == PINNED_DELTA_CID
    assert catalog["catalog_cid"] == PINNED_CATALOG_CID
    assert catalog["components"]["ipfs_accelerate_py"]["status"] == "observed"
    pyproject = (_PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'interface = "AccelerateStaleRejectionAndPlanDelta@1"' in pyproject
    plan = produce_stale_rejection_and_plan_delta()
    assert plan["produced"] is True
    assert plan["live"] is False
    assert tuple(plan["escalation_order"]) == ESCALATION_ORDER
    with pytest.raises(AccelerateStaleRejectionError, match="cannot complete"):
        produce_stale_rejection_and_plan_delta(complete_from="frontier_model")
    with pytest.raises(AccelerateStaleRejectionError, match="admitted as current"):
        produce_stale_rejection_and_plan_delta(admit_stale_as_current=True)
    with pytest.raises(AccelerateStaleRejectionError, match="history"):
        produce_stale_rejection_and_plan_delta(history_mutated=True)


def test_simulated_live_probe_is_rejected() -> None:
    dummy = "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    with pytest.raises(AccelerateStaleRejectionError, match="simulated"):
        qualify_stale_rejection(
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
            delta_cid=dummy,
            document_cid=dummy,
            catalog_cid=dummy,
            pack_cid=dummy,
            current_root_cid=dummy,
            route_cid=dummy,
            patch_cid=dummy,
            run_cid=dummy,
            interface_cid=dummy,
            objective_cid=dummy,
            idea_digest_cid=dummy,
        )
    with pytest.raises(AccelerateStaleRejectionError, match="measured_live"):
        qualify_stale_rejection(
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
            delta_cid=dummy,
            document_cid=dummy,
            catalog_cid=dummy,
            pack_cid=dummy,
            current_root_cid=dummy,
            route_cid=dummy,
            patch_cid=dummy,
            run_cid=dummy,
            interface_cid=dummy,
            objective_cid=dummy,
            idea_digest_cid=dummy,
        )
