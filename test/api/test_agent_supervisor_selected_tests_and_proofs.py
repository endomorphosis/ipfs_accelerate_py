"""PCPR-065: Accelerate-owned selected tests and proofs."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.assurance.selected_tests_and_proofs import (
    AUTHORIZED_PATH_PREFIXES,
    CLOSED_RELEASE_OUTCOMES,
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID,
    ESCALATION_ORDER,
    FULL_VALIDATION_FALLBACK,
    HERMETIC_CANDIDATE_SUITES,
    INTERFACE,
    OPERATOR_BLOCKING_TASK_ID,
    OBJECTIVE_KIND,
    OutcomeProbe,
    PCPR_065_GOAL_ID,
    PCPR_065_TASK_ID,
    PINNED_CATALOG_CID,
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
    SELECTED_TESTS,
    AccelerateSelectedTestsAndProofsError,
    current_head_static_probes,
    path_is_authorized,
    pcpr_065_receipt_promotion,
    platform_selected_tests_and_proofs_catalog,
    produce_selected_tests_and_proofs,
    qualify_current_head_selected_tests_and_proofs,
    qualify_selected_tests_and_proofs,
    refuse_incomplete_selection_as_sufficient,
    refuse_model_completion,
    refuse_pack_cid_remint,
    refuse_patch_cid_remint,
    refuse_route_cid_remint,
    refuse_unauthorized_path,
    render_declared_run,
    verify_selected_tests_and_proofs_files,
)


_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def test_closed_vocabularies_match_pcpr_065_requirements() -> None:
    assert PCPR_065_TASK_ID == "PCPR-065"
    assert PCPR_065_GOAL_ID == "PCPR-G700"
    assert INTERFACE == "AccelerateSelectedTestsAndProofs@1"
    assert SCHEMA == "ipfs_accelerate_py/assurance/selected-tests-and-proofs@1"
    assert OBJECTIVE_KIND == "declared_selected_tests_and_proofs"
    assert OPERATOR_BLOCKING_TASK_ID == (
        "pcpr-065-operator-live-selected-tests-and-proofs"
    )
    assert PROTOCOL_INTERFACE == "LogicProviderProtocol@2"
    assert "release_candidate_qualified" in CLOSED_RELEASE_OUTCOMES
    assert "rnd_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert HERMETIC_CANDIDATE_SUITES[-1].endswith(
        "test_agent_supervisor_selected_tests_and_proofs.py"
    )
    assert SEALED_PATH == "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
    assert SEALED_PYTHON == "/usr/bin/python3.12"
    assert ESCALATION_ORDER[0] == "exact receipt"
    assert ESCALATION_ORDER[3] == "selected tests"
    assert ESCALATION_ORDER[4] == "incremental prover"
    assert ESCALATION_ORDER[-1] == "human decision"
    document = render_declared_run()
    assert document["applied"] is False
    assert document["live"] is False
    assert document["release_claim"] is False
    assert document["closed_release_outcome"] is None
    assert document["selected_tests_and_proofs"]["produced"] is True
    assert document["selected_tests_and_proofs"]["run"] is True
    assert document["selected_tests_and_proofs"]["live"] is False
    assert document["selected_tests_and_proofs"]["hermetic"] is True
    assert document["selected_tests_and_proofs"][
        "model_assertion_completes_work"
    ] is False
    assert document["selected_tests"]["run"] is True
    assert document["selected_tests"]["live"] is False
    assert document["selected_tests"]["incomplete_selection"] is True
    assert document["selected_tests"][
        "incomplete_selection_requires_full_validation"
    ] is True
    assert document["selected_tests"]["full_validation_fallback_required"] is True
    assert list(document["selected_tests"]["paths"]) == list(SELECTED_TESTS)
    assert list(document["selected_tests"]["full_validation_fallback"]) == list(
        FULL_VALIDATION_FALLBACK
    )
    assert document["incremental_prover"]["status"] == "unavailable"
    assert document["incremental_prover"]["evidence_kind"] == "unavailable"
    assert document["incremental_prover"]["live"] is False
    assert document["duckdb_or_quack_state_written"] is False
    assert document["objective_cid"] == PINNED_OBJECTIVE_CID
    assert document["idea_digest"] == PINNED_IDEA_DIGEST
    assert document["context_pack"]["pack_cid"] == PINNED_PACK_CID
    assert document["storage"]["current_root_cid"] == PINNED_CURRENT_ROOT_CID
    assert document["route"]["route_cid"] == PINNED_ROUTE_CID
    assert document["bounded_patch"]["patch_cid"] == PINNED_PATCH_CID
    assert document["selected_tests_and_proofs"]["run_cid"] == PINNED_RUN_CID
    assert document["document_cid"] == PINNED_DOCUMENT_CID
    assert document["operator_blocking_task"]["status"] == "typed_blocked"
    assert refuse_pack_cid_remint(PINNED_PACK_CID) == PINNED_PACK_CID
    assert refuse_route_cid_remint(PINNED_ROUTE_CID) == PINNED_ROUTE_CID
    assert refuse_patch_cid_remint(PINNED_PATCH_CID) == PINNED_PATCH_CID
    with pytest.raises(AccelerateSelectedTestsAndProofsError, match="remints"):
        refuse_pack_cid_remint(
            "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        )
    with pytest.raises(AccelerateSelectedTestsAndProofsError, match="cannot complete"):
        refuse_model_completion("frontier_model")
    with pytest.raises(AccelerateSelectedTestsAndProofsError, match="not an authorized"):
        refuse_unauthorized_path("/etc/passwd")
    with pytest.raises(
        AccelerateSelectedTestsAndProofsError, match="cannot count as selected-test"
    ):
        refuse_incomplete_selection_as_sufficient(
            incomplete=True,
            full_validation_required=False,
        )
    assert path_is_authorized(AUTHORIZED_PATH_PREFIXES[0]) is True


def test_current_head_is_rnd_non_promoted_and_not_a_release() -> None:
    verdict = qualify_current_head_selected_tests_and_proofs()
    assert verdict.promotion_status == "rnd_non_promoted"
    assert verdict.supervisor_disposition == "supervisor_non_promoted"
    assert verdict.closed_release_outcome is None
    assert verdict.release_claim is False
    assert verdict.completion_authoritative is False
    assert verdict.contracts_frozen is False
    assert verdict.duckdb_or_quack_state_written is False
    assert verdict.sibling_source_required is False
    assert verdict.live_application is False
    assert verdict.live_selected_tests is False
    assert verdict.live_prover is False
    assert verdict.operator_blocking_task == OPERATOR_BLOCKING_TASK_ID
    assert verdict.simulated_results_represented_as_live is False
    assert verdict.this_task_created_competing_authority is False
    assert verdict.promotion_status not in CLOSED_RELEASE_OUTCOMES
    assert verdict.verdict_cid.startswith("baguqeera")
    assert verdict.verdict_cid == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert verdict.pack_cid == PINNED_PACK_CID
    assert verdict.run_cid == PINNED_RUN_CID
    assert verdict.document_cid == PINNED_DOCUMENT_CID
    assert verdict.catalog_cid == PINNED_CATALOG_CID
    assert verdict.current_root_cid == PINNED_CURRENT_ROOT_CID
    assert verdict.route_cid == PINNED_ROUTE_CID
    assert verdict.patch_cid == PINNED_PATCH_CID
    assert verdict.blockers == ()
    section = pcpr_065_receipt_promotion(verdict)
    assert section["promotion_status"] == "rnd_non_promoted"
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_static_probes_show_declared_run_constraints() -> None:
    probes = {item.probe_id: item for item in current_head_static_probes()}
    assert probes["run_files_match_generator"].present is True
    assert probes["pyproject_selected_tests_and_proofs_table"].present is True
    assert probes["paths_remain_authorized"].present is True
    assert probes["owner_pack_cid_matches_pin"].present is True
    assert probes["kit_current_root_bound_not_minted"].present is True
    assert probes["route_cid_bound_not_minted"].present is True
    assert probes["patch_cid_bound_not_minted"].present is True
    assert probes["selected_tests_run_hermetically"].present is True
    assert probes["incomplete_selection_requires_full_validation"].present is True
    assert probes["full_validation_fallback_required"].present is True
    assert probes["incremental_prover_typed_unavailable"].present is True
    assert probes["model_assertion_cannot_complete_work"].present is True
    assert probes["datasets_identity_reminted"].present is False
    assert probes["kit_identity_reminted"].present is False
    assert probes["model_assertion_completed_work"].present is False
    assert probes["incomplete_selection_accepted_as_sufficient"].present is False
    assert probes["live_selected_tests"].evidence_kind == "unavailable"
    assert probes["live_prover"].evidence_kind == "unavailable"
    for probe in probes.values():
        assert probe.live is False
        assert probe.simulated_represented_as_live is False


def test_committed_files_match_generator() -> None:
    verified = verify_selected_tests_and_proofs_files()
    assert verified["ok"] is True
    catalog = platform_selected_tests_and_proofs_catalog()
    assert catalog["interface"] == "PlatformSelectedTestsAndProofs@1"
    assert catalog["release_claim"] is False
    assert catalog["closed_release_outcome"] is None
    assert catalog["live_selected_tests"] is False
    assert catalog["live_prover"] is False
    assert catalog["pack_cid"] == PINNED_PACK_CID
    assert catalog["run_cid"] == PINNED_RUN_CID
    assert catalog["catalog_cid"] == PINNED_CATALOG_CID
    assert catalog["components"]["ipfs_accelerate_py"]["status"] == "observed"
    pyproject = (_PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'interface = "AccelerateSelectedTestsAndProofs@1"' in pyproject
    plan = produce_selected_tests_and_proofs()
    assert plan["produced"] is True
    assert plan["run"] is True
    assert plan["live"] is False
    assert tuple(plan["escalation_order"]) == ESCALATION_ORDER
    with pytest.raises(AccelerateSelectedTestsAndProofsError, match="cannot complete"):
        produce_selected_tests_and_proofs(complete_from="frontier_model")
    with pytest.raises(
        AccelerateSelectedTestsAndProofsError, match="cannot count as selected-test"
    ):
        produce_selected_tests_and_proofs(
            incomplete_selection=True,
            full_validation_required=False,
        )


def test_simulated_live_probe_is_rejected() -> None:
    with pytest.raises(AccelerateSelectedTestsAndProofsError, match="simulated"):
        qualify_selected_tests_and_proofs(
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
            run_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            document_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            catalog_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            pack_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            current_root_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            route_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            patch_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            objective_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            idea_digest_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
    with pytest.raises(AccelerateSelectedTestsAndProofsError, match="measured_live"):
        qualify_selected_tests_and_proofs(
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
            run_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            document_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            catalog_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            pack_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            current_root_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            route_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            patch_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            objective_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            idea_digest_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
