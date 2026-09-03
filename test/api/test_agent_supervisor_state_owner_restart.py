"""PCPR-070: Accelerate-owned authoritative state-owner restart."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.assurance.state_owner_restart import (
    AUTHORIZED_PATH_PREFIXES,
    CLOSED_RELEASE_OUTCOMES,
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID,
    ESCALATION_ORDER,
    HERMETIC_CANDIDATE_SUITES,
    INTERFACE,
    OPERATOR_BLOCKING_TASK_ID,
    OBJECTIVE_KIND,
    OutcomeProbe,
    PCPR_070_GOAL_ID,
    PCPR_070_TASK_ID,
    PINNED_CATALOG_CID,
    PINNED_CURRENT_ROOT_CID,
    PINNED_DELTA_CID,
    PINNED_DOCUMENT_CID,
    PINNED_IDEA_DIGEST,
    PINNED_INTERFACE_CID,
    PINNED_OBJECTIVE_CID,
    PINNED_PACK_CID,
    PINNED_PATCH_CID,
    PINNED_RESTART_CID,
    PINNED_ROUTE_CID,
    PINNED_RUN_CID,
    PLAN_EPOCH,
    PROTOCOL_INTERFACE,
    RECONSTRUCTED_IDENTITIES,
    SCHEMA,
    SEALED_PATH,
    SEALED_PYTHON,
    AccelerateStateOwnerRestartError,
    current_head_static_probes,
    path_is_authorized,
    pcpr_070_receipt_promotion,
    platform_state_owner_restart_catalog,
    produce_state_owner_restart,
    qualify_current_head_state_owner_restart,
    qualify_state_owner_restart,
    refuse_database_edit,
    refuse_delta_cid_remint,
    refuse_duplicate_restart_effect,
    refuse_interface_cid_remint,
    refuse_memory_reconstruction,
    refuse_model_completion,
    refuse_pack_cid_remint,
    refuse_patch_cid_remint,
    refuse_route_cid_remint,
    refuse_run_cid_remint,
    refuse_stale_fence_reuse,
    refuse_stale_lease_reuse,
    refuse_unauthorized_path,
    render_declared_restart,
    verify_state_owner_restart_files,
)


_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def test_closed_vocabularies_match_pcpr_070_requirements() -> None:
    assert PCPR_070_TASK_ID == "PCPR-070"
    assert PCPR_070_GOAL_ID == "PCPR-G700"
    assert INTERFACE == "AccelerateAuthoritativeStateOwnerRestart@1"
    assert SCHEMA == "ipfs_accelerate_py/assurance/authoritative-state-owner-restart@1"
    assert OBJECTIVE_KIND == "declared_authoritative_state_owner_restart"
    assert OPERATOR_BLOCKING_TASK_ID == (
        "pcpr-070-operator-live-authoritative-state-owner-restart"
    )
    assert PROTOCOL_INTERFACE == "LogicProviderProtocol@2"
    assert "release_candidate_qualified" in CLOSED_RELEASE_OUTCOMES
    assert "rnd_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert HERMETIC_CANDIDATE_SUITES[-1].endswith(
        "test_agent_supervisor_state_owner_restart.py"
    )
    assert SEALED_PATH == "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
    assert SEALED_PYTHON == "/usr/bin/python3.12"
    assert ESCALATION_ORDER[0] == "exact receipt"
    assert ESCALATION_ORDER[3] == "selected tests"
    assert ESCALATION_ORDER[-1] == "human decision"
    document = render_declared_restart()
    assert document["applied"] is False
    assert document["live"] is False
    assert document["release_claim"] is False
    assert document["closed_release_outcome"] is None
    restart = document["state_owner_restart"]
    assert restart["produced"] is True
    assert restart["live"] is False
    assert restart["hermetic"] is True
    assert restart["change_kind"] == "authoritative_state_owner_restart"
    assert restart["state_reconstructed"] is True
    assert restart["reconstructed_from"] == "durable_records"
    assert restart["reconstructed_from_memory"] is False
    assert restart["duplicate_effect_rejected"] is True
    assert restart["continuation_without_database_edits"] is True
    assert restart["database_edited"] is False
    assert restart["plan_epoch"] == PLAN_EPOCH
    assert restart["history_mutated"] is False
    assert restart["remints_protocol"] is False
    assert restart["model_assertion_completes_work"] is False
    assert list(restart["reconstructed_identities"]) == list(RECONSTRUCTED_IDENTITIES)
    assert document["duckdb_or_quack_state_written"] is False
    assert document["objective_cid"] == PINNED_OBJECTIVE_CID
    assert document["idea_digest"] == PINNED_IDEA_DIGEST
    assert document["context_pack"]["pack_cid"] == PINNED_PACK_CID
    assert document["context_pack"]["rejected"] is True
    assert document["context_pack"]["survives_restart"] is True
    assert document["storage"]["current_root_cid"] == PINNED_CURRENT_ROOT_CID
    assert document["route"]["route_cid"] == PINNED_ROUTE_CID
    assert document["bounded_patch"]["patch_cid"] == PINNED_PATCH_CID
    assert document["selected_tests"]["run_cid"] == PINNED_RUN_CID
    assert restart["interface_cid"] == PINNED_INTERFACE_CID
    assert restart["delta_cid"] == PINNED_DELTA_CID
    assert restart["restart_cid"] == PINNED_RESTART_CID
    assert document["document_cid"] == PINNED_DOCUMENT_CID
    assert document["operator_blocking_task"]["status"] == "typed_blocked"
    assert refuse_pack_cid_remint(PINNED_PACK_CID) == PINNED_PACK_CID
    assert refuse_route_cid_remint(PINNED_ROUTE_CID) == PINNED_ROUTE_CID
    assert refuse_patch_cid_remint(PINNED_PATCH_CID) == PINNED_PATCH_CID
    assert refuse_run_cid_remint(PINNED_RUN_CID) == PINNED_RUN_CID
    assert refuse_interface_cid_remint(PINNED_INTERFACE_CID) == PINNED_INTERFACE_CID
    assert refuse_delta_cid_remint(PINNED_DELTA_CID) == PINNED_DELTA_CID
    with pytest.raises(AccelerateStateOwnerRestartError, match="remints"):
        refuse_pack_cid_remint(
            "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        )
    with pytest.raises(AccelerateStateOwnerRestartError, match="cannot complete"):
        refuse_model_completion("frontier_model")
    with pytest.raises(AccelerateStateOwnerRestartError, match="not an authorized"):
        refuse_unauthorized_path("/etc/passwd")
    with pytest.raises(AccelerateStateOwnerRestartError, match="memory"):
        refuse_memory_reconstruction(from_memory=True)
    with pytest.raises(AccelerateStateOwnerRestartError, match="DuckDB"):
        refuse_database_edit(edited=True)
    with pytest.raises(AccelerateStateOwnerRestartError, match="duplicate"):
        refuse_duplicate_restart_effect(already_applied=True)
    with pytest.raises(AccelerateStateOwnerRestartError, match="stale lease"):
        refuse_stale_lease_reuse(stale=True, reused=True)
    with pytest.raises(AccelerateStateOwnerRestartError, match="stale fence"):
        refuse_stale_fence_reuse(stale=True, reused=True)
    assert path_is_authorized(AUTHORIZED_PATH_PREFIXES[0]) is True


def test_current_head_is_rnd_non_promoted_and_not_a_release() -> None:
    verdict = qualify_current_head_state_owner_restart()
    assert verdict.promotion_status == "rnd_non_promoted"
    assert verdict.supervisor_disposition == "supervisor_non_promoted"
    assert verdict.closed_release_outcome is None
    assert verdict.release_claim is False
    assert verdict.completion_authoritative is False
    assert verdict.contracts_frozen is False
    assert verdict.duckdb_or_quack_state_written is False
    assert verdict.sibling_source_required is False
    assert verdict.live_application is False
    assert verdict.live_restart is False
    assert verdict.operator_blocking_task == OPERATOR_BLOCKING_TASK_ID
    assert verdict.simulated_results_represented_as_live is False
    assert verdict.this_task_created_competing_authority is False
    assert verdict.promotion_status not in CLOSED_RELEASE_OUTCOMES
    assert verdict.verdict_cid.startswith("baguqeera")
    assert verdict.verdict_cid == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert verdict.pack_cid == PINNED_PACK_CID
    assert verdict.restart_cid == PINNED_RESTART_CID
    assert verdict.document_cid == PINNED_DOCUMENT_CID
    assert verdict.catalog_cid == PINNED_CATALOG_CID
    assert verdict.current_root_cid == PINNED_CURRENT_ROOT_CID
    assert verdict.route_cid == PINNED_ROUTE_CID
    assert verdict.patch_cid == PINNED_PATCH_CID
    assert verdict.run_cid == PINNED_RUN_CID
    assert verdict.interface_cid == PINNED_INTERFACE_CID
    assert verdict.delta_cid == PINNED_DELTA_CID
    assert verdict.blockers == ()
    section = pcpr_070_receipt_promotion(verdict)
    assert section["promotion_status"] == "rnd_non_promoted"
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_static_probes_show_declared_restart_constraints() -> None:
    probes = {item.probe_id: item for item in current_head_static_probes()}
    assert probes["restart_files_match_generator"].present is True
    assert probes["pyproject_state_owner_restart_table"].present is True
    assert probes["paths_remain_authorized"].present is True
    assert probes["owner_pack_cid_matches_pin"].present is True
    assert probes["kit_current_root_bound_not_minted"].present is True
    assert probes["route_cid_bound_not_minted"].present is True
    assert probes["patch_cid_bound_not_minted"].present is True
    assert probes["run_cid_bound_not_minted"].present is True
    assert probes["interface_cid_bound_not_minted"].present is True
    assert probes["delta_cid_bound_not_minted"].present is True
    assert probes["state_reconstructed_from_durable_records"].present is True
    assert probes["duplicate_restart_effect_rejected"].present is True
    assert probes["unexpired_lease_retained"].present is True
    assert probes["stale_lease_expired"].present is True
    assert probes["current_fence_retained"].present is True
    assert probes["stale_fence_expired"].present is True
    assert probes["owner_generation_incremented"].present is True
    assert probes["continuation_without_database_edits"].present is True
    assert probes["model_assertion_cannot_complete_work"].present is True
    assert probes["datasets_identity_reminted"].present is False
    assert probes["kit_identity_reminted"].present is False
    assert probes["delta_identity_reminted"].present is False
    assert probes["memory_reconstruction_accepted"].present is False
    assert probes["duplicate_restart_effect_accepted"].present is False
    assert probes["stale_lease_reused"].present is False
    assert probes["stale_fence_reused"].present is False
    assert probes["database_edited"].present is False
    assert probes["live_application"].evidence_kind == "unavailable"
    assert probes["live_restart"].evidence_kind == "unavailable"
    for probe in probes.values():
        assert probe.live is False
        assert probe.simulated_represented_as_live is False


def test_committed_files_match_generator() -> None:
    verified = verify_state_owner_restart_files()
    assert verified["ok"] is True
    catalog = platform_state_owner_restart_catalog()
    assert catalog["interface"] == "PlatformAuthoritativeStateOwnerRestart@1"
    assert catalog["release_claim"] is False
    assert catalog["closed_release_outcome"] is None
    assert catalog["live_application"] is False
    assert catalog["live_restart"] is False
    assert catalog["pack_cid"] == PINNED_PACK_CID
    assert catalog["restart_cid"] == PINNED_RESTART_CID
    assert catalog["catalog_cid"] == PINNED_CATALOG_CID
    assert catalog["components"]["ipfs_accelerate_py"]["status"] == "observed"
    pyproject = (_PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'interface = "AccelerateAuthoritativeStateOwnerRestart@1"' in pyproject
    plan = produce_state_owner_restart()
    assert plan["produced"] is True
    assert plan["live"] is False
    assert tuple(plan["escalation_order"]) == ESCALATION_ORDER
    with pytest.raises(AccelerateStateOwnerRestartError, match="cannot complete"):
        produce_state_owner_restart(complete_from="frontier_model")
    with pytest.raises(AccelerateStateOwnerRestartError, match="memory"):
        produce_state_owner_restart(from_memory=True)
    with pytest.raises(AccelerateStateOwnerRestartError, match="DuckDB"):
        produce_state_owner_restart(database_edited=True)
    with pytest.raises(AccelerateStateOwnerRestartError, match="duplicate"):
        produce_state_owner_restart(already_applied=True)


def test_simulated_live_probe_is_rejected() -> None:
    dummy = "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    with pytest.raises(AccelerateStateOwnerRestartError, match="simulated"):
        qualify_state_owner_restart(
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
            restart_cid=dummy,
            document_cid=dummy,
            catalog_cid=dummy,
            pack_cid=dummy,
            current_root_cid=dummy,
            route_cid=dummy,
            patch_cid=dummy,
            run_cid=dummy,
            interface_cid=dummy,
            delta_cid=dummy,
            objective_cid=dummy,
            idea_digest_cid=dummy,
        )
    with pytest.raises(AccelerateStateOwnerRestartError, match="measured_live"):
        qualify_state_owner_restart(
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
            restart_cid=dummy,
            document_cid=dummy,
            catalog_cid=dummy,
            pack_cid=dummy,
            current_root_cid=dummy,
            route_cid=dummy,
            patch_cid=dummy,
            run_cid=dummy,
            interface_cid=dummy,
            delta_cid=dummy,
            objective_cid=dummy,
            idea_digest_cid=dummy,
        )
