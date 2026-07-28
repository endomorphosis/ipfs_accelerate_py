"""ASI-170: frozen paired E2E population for endpoint-aware supervisor usage."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.supervisor_usage_rollout import (
    REQUIRED_CONSUMERS,
    REQUIRED_MODES,
    REQUIRED_SAFETY_INVARIANTS,
    REQUIRED_STAGES,
    REQUIRED_TOPOLOGIES,
    SUPERVISOR_USAGE_ROLLOUT_REQUIREMENT_ID,
    SupervisorStage,
    SupervisorUsageRolloutMode,
    TopologyKind,
    build_paired_report,
    mode_is_non_selecting,
    run_e2e_population,
)


def test_frozen_e2e_covers_every_stage_topology_mode_and_consumer() -> None:
    receipts = run_e2e_population(observation_label="e2e")
    stages = {r.stage for r in receipts}
    topologies = {r.topology for r in receipts}
    modes = {r.mode for r in receipts}
    consumers = {r.consumer_id for r in receipts}

    assert stages == set(REQUIRED_STAGES)
    assert topologies == set(REQUIRED_TOPOLOGIES)
    assert modes == set(REQUIRED_MODES)
    assert consumers == set(REQUIRED_CONSUMERS)
    assert len(receipts) == len(REQUIRED_STAGES)


def test_exact_endpoint_and_task_stage_attribution() -> None:
    receipts = run_e2e_population(observation_label="attrib")
    for receipt in receipts:
        assert receipt.endpoint_scope_id.startswith("scope_")
        assert receipt.task_id == f"task:{receipt.stage.value}"
        assert receipt.request_id
        assert receipt.stage in REQUIRED_STAGES
        # Reservation id is present or explicitly none for off/degraded paths.
        assert receipt.reservation_id


def test_off_observe_shadow_do_not_alter_legacy_selection() -> None:
    receipts = run_e2e_population(observation_label="nonselect")
    for receipt in receipts:
        if mode_is_non_selecting(receipt.mode):
            assert receipt.altered_execution is False
            if receipt.mode is SupervisorUsageRolloutMode.OFF:
                assert receipt.selected_binding == receipt.legacy_binding
                assert receipt.charged_requests == 0


def test_paired_report_passes_safety_invariants() -> None:
    report = build_paired_report(observation_label="e2e-paired")
    assert report.passed
    assert not report.failure_codes()
    assert set(report.safety_invariants_passed) == {
        inv.value for inv in REQUIRED_SAFETY_INVARIANTS
    }
    assert report.to_dict()["requirement_id"] == (
        SUPERVISOR_USAGE_ROLLOUT_REQUIREMENT_ID
    )
    assert report.to_dict()["authoritative"] is False
    assert report.to_dict()["completion_authoritative"] is False


def test_local_fallback_and_single_flight_stages_present() -> None:
    receipts = run_e2e_population(observation_label="special")
    by_stage = {r.stage: r for r in receipts}
    assert SupervisorStage.LOCAL_FALLBACK in by_stage
    assert SupervisorStage.SINGLE_FLIGHT in by_stage
    assert SupervisorStage.BATCH in by_stage
    local = by_stage[SupervisorStage.LOCAL_FALLBACK]
    assert local.topology is TopologyKind.LOCAL_DETERMINISTIC


def test_e2e_receipts_are_redacted() -> None:
    report = build_paired_report(observation_label="redact-e2e")
    blob = report.to_dict()
    text = str(blob).casefold()
    assert "sk-" not in text
    assert "bearer " not in text
    assert "https://api." not in text
    for receipt in report.e2e_receipts:
        payload = receipt.to_dict()
        assert "prompt" not in payload
        assert "output" not in payload
