"""PCPR-063: Accelerate-owned deterministic-first route."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.assurance.deterministic_first_route import (
    AUTHORIZED_PATH_PREFIXES,
    CLOSED_RELEASE_OUTCOMES,
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID,
    ESCALATION_ORDER,
    HERMETIC_CANDIDATE_SUITES,
    INTERFACE,
    OPERATOR_BLOCKING_TASK_ID,
    OBJECTIVE_KIND,
    OutcomeProbe,
    PCPR_063_GOAL_ID,
    PCPR_063_TASK_ID,
    PINNED_CATALOG_CID,
    PINNED_CURRENT_ROOT_CID,
    PINNED_DOCUMENT_CID,
    PINNED_IDEA_DIGEST,
    PINNED_OBJECTIVE_CID,
    PINNED_PACK_CID,
    PINNED_ROUTE_CID,
    SCHEMA,
    SEALED_PATH,
    SEALED_PYTHON,
    AccelerateDeterministicFirstRouteError,
    current_head_static_probes,
    execute_deterministic_first_route,
    path_is_authorized,
    pcpr_063_receipt_promotion,
    platform_deterministic_first_route_catalog,
    qualify_current_head_deterministic_first_route,
    qualify_deterministic_first_route,
    refuse_model_completion,
    refuse_order_violation,
    refuse_pack_cid_remint,
    refuse_route_cid_remint,
    refuse_unauthorized_path,
    render_declared_route,
    verify_deterministic_first_route_files,
)


_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def test_closed_vocabularies_match_pcpr_063_requirements() -> None:
    assert PCPR_063_TASK_ID == "PCPR-063"
    assert PCPR_063_GOAL_ID == "PCPR-G700"
    assert INTERFACE == "AccelerateDeterministicFirstRoute@1"
    assert SCHEMA == "ipfs_accelerate_py/assurance/deterministic-first-route@1"
    assert OBJECTIVE_KIND == "declared_deterministic_first_route"
    assert OPERATOR_BLOCKING_TASK_ID == "pcpr-063-operator-live-deterministic-route"
    assert "release_candidate_qualified" in CLOSED_RELEASE_OUTCOMES
    assert "rnd_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert HERMETIC_CANDIDATE_SUITES[-1].endswith(
        "test_agent_supervisor_deterministic_first_route.py"
    )
    assert SEALED_PATH == "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
    assert SEALED_PYTHON == "/usr/bin/python3.12"
    assert ESCALATION_ORDER[0] == "exact receipt"
    assert ESCALATION_ORDER[-1] == "human decision"
    document = render_declared_route()
    assert document["applied"] is False
    assert document["live"] is False
    assert document["release_claim"] is False
    assert document["closed_release_outcome"] is None
    assert document["route"]["executed"] is True
    assert document["route"]["live"] is False
    assert document["route"]["hermetic"] is True
    assert document["route"]["model_assertion_completes_work"] is False
    assert document["route"]["completed_through"] == "schema_type_and_static_checks"
    assert document["route"]["next_authorized_stage"] == "selected_tests"
    assert document["bounded_patch"]["produced"] is False
    assert document["selected_tests"]["run"] is False
    assert document["duckdb_or_quack_state_written"] is False
    assert document["objective_cid"] == PINNED_OBJECTIVE_CID
    assert document["idea_digest"] == PINNED_IDEA_DIGEST
    assert document["context_pack"]["pack_cid"] == PINNED_PACK_CID
    assert document["storage"]["current_root_cid"] == PINNED_CURRENT_ROOT_CID
    assert document["route"]["route_cid"] == PINNED_ROUTE_CID
    assert document["document_cid"] == PINNED_DOCUMENT_CID
    assert document["operator_blocking_task"]["status"] == "typed_blocked"
    assert refuse_pack_cid_remint(PINNED_PACK_CID) == PINNED_PACK_CID
    assert refuse_route_cid_remint(PINNED_ROUTE_CID) == PINNED_ROUTE_CID
    with pytest.raises(AccelerateDeterministicFirstRouteError, match="remints"):
        refuse_pack_cid_remint(
            "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        )
    with pytest.raises(AccelerateDeterministicFirstRouteError, match="cannot complete"):
        refuse_model_completion("frontier_model")
    with pytest.raises(AccelerateDeterministicFirstRouteError, match="order violated"):
        refuse_order_violation(
            current="exact_receipt", attempted="medium_model"
        )
    with pytest.raises(AccelerateDeterministicFirstRouteError, match="not an authorized"):
        refuse_unauthorized_path("/etc/passwd")
    assert path_is_authorized(AUTHORIZED_PATH_PREFIXES[0]) is True


def test_current_head_is_rnd_non_promoted_and_not_a_release() -> None:
    verdict = qualify_current_head_deterministic_first_route()
    assert verdict.promotion_status == "rnd_non_promoted"
    assert verdict.supervisor_disposition == "supervisor_non_promoted"
    assert verdict.closed_release_outcome is None
    assert verdict.release_claim is False
    assert verdict.completion_authoritative is False
    assert verdict.contracts_frozen is False
    assert verdict.duckdb_or_quack_state_written is False
    assert verdict.sibling_source_required is False
    assert verdict.live_execution is False
    assert verdict.live_selected_tests is False
    assert verdict.live_prover is False
    assert verdict.operator_blocking_task == OPERATOR_BLOCKING_TASK_ID
    assert verdict.simulated_results_represented_as_live is False
    assert verdict.this_task_created_competing_authority is False
    assert verdict.promotion_status not in CLOSED_RELEASE_OUTCOMES
    assert verdict.verdict_cid.startswith("baguqeera")
    assert verdict.verdict_cid == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert verdict.pack_cid == PINNED_PACK_CID
    assert verdict.route_cid == PINNED_ROUTE_CID
    assert verdict.document_cid == PINNED_DOCUMENT_CID
    assert verdict.catalog_cid == PINNED_CATALOG_CID
    assert verdict.current_root_cid == PINNED_CURRENT_ROOT_CID
    assert verdict.blockers == ()
    section = pcpr_063_receipt_promotion(verdict)
    assert section["promotion_status"] == "rnd_non_promoted"
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_static_probes_show_declared_route_constraints() -> None:
    probes = {item.probe_id: item for item in current_head_static_probes()}
    assert probes["route_files_match_generator"].present is True
    assert probes["pyproject_deterministic_first_route_table"].present is True
    assert probes["escalation_order_enforced"].present is True
    assert probes["static_checks_executed_hermetically"].present is True
    assert probes["model_assertion_cannot_complete_work"].present is True
    assert probes["paths_remain_authorized"].present is True
    assert probes["owner_pack_cid_matches_pin"].present is True
    assert probes["kit_current_root_bound_not_minted"].present is True
    assert probes["bounded_patch_deferred_to_pcpr_064"].present is True
    assert probes["selected_tests_deferred_to_pcpr_065"].present is True
    assert probes["datasets_identity_reminted"].present is False
    assert probes["kit_identity_reminted"].present is False
    assert probes["model_assertion_completed_work"].present is False
    assert probes["live_execution"].evidence_kind == "unavailable"
    assert probes["live_prover"].evidence_kind == "unavailable"
    for probe in probes.values():
        assert probe.live is False
        assert probe.simulated_represented_as_live is False


def test_committed_files_match_generator() -> None:
    verified = verify_deterministic_first_route_files()
    assert verified["ok"] is True
    catalog = platform_deterministic_first_route_catalog()
    assert catalog["interface"] == "PlatformDeterministicFirstRoute@1"
    assert catalog["release_claim"] is False
    assert catalog["closed_release_outcome"] is None
    assert catalog["live_execution"] is False
    assert catalog["pack_cid"] == PINNED_PACK_CID
    assert catalog["route_cid"] == PINNED_ROUTE_CID
    assert catalog["catalog_cid"] == PINNED_CATALOG_CID
    assert catalog["components"]["ipfs_accelerate_py"]["status"] == "observed"
    pyproject = (_PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'interface = "AccelerateDeterministicFirstRoute@1"' in pyproject
    decision = execute_deterministic_first_route()
    assert decision["executed"] is True
    assert decision["live"] is False
    assert tuple(decision["escalation_order"]) == ESCALATION_ORDER
    with pytest.raises(AccelerateDeterministicFirstRouteError, match="start at"):
        execute_deterministic_first_route(start_at="medium_model")
    with pytest.raises(AccelerateDeterministicFirstRouteError, match="cannot complete"):
        execute_deterministic_first_route(complete_from="frontier_model")


def test_simulated_live_probe_is_rejected() -> None:
    with pytest.raises(AccelerateDeterministicFirstRouteError, match="simulated"):
        qualify_deterministic_first_route(
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
            route_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            document_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            catalog_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            pack_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            current_root_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            objective_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            idea_digest_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
    with pytest.raises(AccelerateDeterministicFirstRouteError, match="measured_live"):
        qualify_deterministic_first_route(
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
            route_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            document_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            catalog_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            pack_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            current_root_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            objective_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            idea_digest_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
