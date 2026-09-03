"""PCPR-064: Accelerate-owned bounded PatchPlan."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.assurance.bounded_patch import (
    AUTHORIZED_PATH_PREFIXES,
    CLOSED_RELEASE_OUTCOMES,
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID,
    ESCALATION_ORDER,
    HERMETIC_CANDIDATE_SUITES,
    INTERFACE,
    OPERATOR_BLOCKING_TASK_ID,
    OBJECTIVE_KIND,
    OutcomeProbe,
    PCPR_064_GOAL_ID,
    PCPR_064_TASK_ID,
    PINNED_CATALOG_CID,
    PINNED_CURRENT_ROOT_CID,
    PINNED_DOCUMENT_CID,
    PINNED_IDEA_DIGEST,
    PINNED_OBJECTIVE_CID,
    PINNED_PACK_CID,
    PINNED_PATCH_CID,
    PINNED_ROUTE_CID,
    PROTOCOL_INTERFACE,
    SCHEMA,
    SEALED_PATH,
    SEALED_PYTHON,
    AccelerateBoundedPatchError,
    current_head_static_probes,
    path_is_authorized,
    pcpr_064_receipt_promotion,
    platform_bounded_patch_catalog,
    produce_bounded_patch,
    qualify_bounded_patch,
    qualify_current_head_bounded_patch,
    refuse_model_completion,
    refuse_pack_cid_remint,
    refuse_protocol_remint,
    refuse_route_cid_remint,
    refuse_unauthorized_path,
    render_declared_patch,
    verify_bounded_patch_files,
)


_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def test_closed_vocabularies_match_pcpr_064_requirements() -> None:
    assert PCPR_064_TASK_ID == "PCPR-064"
    assert PCPR_064_GOAL_ID == "PCPR-G700"
    assert INTERFACE == "AccelerateBoundedPatch@1"
    assert SCHEMA == "ipfs_accelerate_py/assurance/bounded-patch@1"
    assert OBJECTIVE_KIND == "declared_bounded_patch"
    assert OPERATOR_BLOCKING_TASK_ID == "pcpr-064-operator-live-bounded-patch"
    assert PROTOCOL_INTERFACE == "LogicProviderProtocol@2"
    assert "release_candidate_qualified" in CLOSED_RELEASE_OUTCOMES
    assert "rnd_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert HERMETIC_CANDIDATE_SUITES[-1].endswith(
        "test_agent_supervisor_bounded_patch.py"
    )
    assert SEALED_PATH == "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
    assert SEALED_PYTHON == "/usr/bin/python3.12"
    assert ESCALATION_ORDER[0] == "exact receipt"
    assert ESCALATION_ORDER[-1] == "human decision"
    document = render_declared_patch()
    assert document["applied"] is False
    assert document["live"] is False
    assert document["release_claim"] is False
    assert document["closed_release_outcome"] is None
    assert document["bounded_patch"]["produced"] is True
    assert document["bounded_patch"]["applied"] is True
    assert document["bounded_patch"]["live"] is False
    assert document["bounded_patch"]["hermetic"] is True
    assert document["bounded_patch"]["remints_protocol"] is False
    assert document["bounded_patch"]["adds_protocol_operation"] is False
    assert document["bounded_patch"]["model_assertion_completes_work"] is False
    assert document["bounded_patch"]["protocol_interface"] == PROTOCOL_INTERFACE
    assert document["selected_tests"]["run"] is False
    assert document["duckdb_or_quack_state_written"] is False
    assert document["objective_cid"] == PINNED_OBJECTIVE_CID
    assert document["idea_digest"] == PINNED_IDEA_DIGEST
    assert document["context_pack"]["pack_cid"] == PINNED_PACK_CID
    assert document["storage"]["current_root_cid"] == PINNED_CURRENT_ROOT_CID
    assert document["route"]["route_cid"] == PINNED_ROUTE_CID
    assert document["bounded_patch"]["patch_cid"] == PINNED_PATCH_CID
    assert document["document_cid"] == PINNED_DOCUMENT_CID
    assert document["operator_blocking_task"]["status"] == "typed_blocked"
    assert refuse_pack_cid_remint(PINNED_PACK_CID) == PINNED_PACK_CID
    assert refuse_route_cid_remint(PINNED_ROUTE_CID) == PINNED_ROUTE_CID
    with pytest.raises(AccelerateBoundedPatchError, match="remints"):
        refuse_pack_cid_remint(
            "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        )
    with pytest.raises(AccelerateBoundedPatchError, match="cannot complete"):
        refuse_model_completion("frontier_model")
    with pytest.raises(AccelerateBoundedPatchError, match="not an authorized"):
        refuse_unauthorized_path("/etc/passwd")
    with pytest.raises(AccelerateBoundedPatchError, match="must not add"):
        refuse_protocol_remint(
            protocol_interface=PROTOCOL_INTERFACE,
            adds_protocol_operation=True,
        )
    assert path_is_authorized(AUTHORIZED_PATH_PREFIXES[0]) is True


def test_current_head_is_rnd_non_promoted_and_not_a_release() -> None:
    verdict = qualify_current_head_bounded_patch()
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
    assert verdict.patch_cid == PINNED_PATCH_CID
    assert verdict.document_cid == PINNED_DOCUMENT_CID
    assert verdict.catalog_cid == PINNED_CATALOG_CID
    assert verdict.current_root_cid == PINNED_CURRENT_ROOT_CID
    assert verdict.route_cid == PINNED_ROUTE_CID
    assert verdict.blockers == ()
    section = pcpr_064_receipt_promotion(verdict)
    assert section["promotion_status"] == "rnd_non_promoted"
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_static_probes_show_declared_patch_constraints() -> None:
    probes = {item.probe_id: item for item in current_head_static_probes()}
    assert probes["patch_files_match_generator"].present is True
    assert probes["pyproject_bounded_patch_table"].present is True
    assert probes["paths_remain_authorized"].present is True
    assert probes["protocol_identity_not_reminted"].present is True
    assert probes["no_protocol_operation_added"].present is True
    assert probes["owner_pack_cid_matches_pin"].present is True
    assert probes["kit_current_root_bound_not_minted"].present is True
    assert probes["route_cid_bound_not_minted"].present is True
    assert probes["bounded_patch_produced"].present is True
    assert probes["selected_tests_deferred_to_pcpr_065"].present is True
    assert probes["model_assertion_cannot_complete_work"].present is True
    assert probes["datasets_identity_reminted"].present is False
    assert probes["kit_identity_reminted"].present is False
    assert probes["model_assertion_completed_work"].present is False
    assert probes["live_application"].evidence_kind == "unavailable"
    assert probes["live_prover"].evidence_kind == "unavailable"
    for probe in probes.values():
        assert probe.live is False
        assert probe.simulated_represented_as_live is False


def test_committed_files_match_generator() -> None:
    verified = verify_bounded_patch_files()
    assert verified["ok"] is True
    catalog = platform_bounded_patch_catalog()
    assert catalog["interface"] == "PlatformBoundedPatch@1"
    assert catalog["release_claim"] is False
    assert catalog["closed_release_outcome"] is None
    assert catalog["live_application"] is False
    assert catalog["pack_cid"] == PINNED_PACK_CID
    assert catalog["patch_cid"] == PINNED_PATCH_CID
    assert catalog["catalog_cid"] == PINNED_CATALOG_CID
    assert catalog["components"]["ipfs_accelerate_py"]["status"] == "observed"
    pyproject = (_PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'interface = "AccelerateBoundedPatch@1"' in pyproject
    plan = produce_bounded_patch()
    assert plan["produced"] is True
    assert plan["live"] is False
    assert tuple(plan["escalation_order"]) == ESCALATION_ORDER
    with pytest.raises(AccelerateBoundedPatchError, match="cannot complete"):
        produce_bounded_patch(complete_from="frontier_model")


def test_simulated_live_probe_is_rejected() -> None:
    with pytest.raises(AccelerateBoundedPatchError, match="simulated"):
        qualify_bounded_patch(
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
            patch_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            document_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            catalog_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            pack_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            current_root_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            route_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            objective_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            idea_digest_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
    with pytest.raises(AccelerateBoundedPatchError, match="measured_live"):
        qualify_bounded_patch(
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
            patch_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            document_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            catalog_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            pack_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            current_root_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            route_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            objective_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            idea_digest_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
