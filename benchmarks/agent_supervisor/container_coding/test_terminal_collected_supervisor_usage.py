"""Authored durable populations exercise the production usage handoff only."""
from copy import deepcopy
import json

import pytest

from benchmarks.agent_supervisor.container_coding.terminal_collected_supervisor_usage import project_collected_supervisor_report
from benchmarks.agent_supervisor.container_coding.terminal_supervisor_usage_accounting import summarize_supervisor_usage
from benchmarks.agent_supervisor.container_coding.test_terminal_world_packet_attempt_collection import _population, _pins, _reseal_join


def project(declaration, census, verifiers=None, *, arm="baseline"):
    return project_collected_supervisor_report(declaration, census, verifiers,
        arm=arm, **_pins(declaration, census, verifiers))


@pytest.mark.parametrize("arm,total", [("baseline", 120), ("candidate", 80)])
def test_pinned_receipts_reach_existing_usage_population_without_other_arm(arm, total):
    declaration, census, verifiers = _population()
    report = project(declaration, census, verifiers, arm=arm)
    usage = summarize_supervisor_usage(report)
    assert usage["total_tokens"] == total
    assert usage["provider_calls"] == 1
    binding = report["world_packet_attempt_collection"]
    assert binding["declared_arm_slot_count"] == binding["retained_arm_receipt_count"] == 1
    assert binding["arm_slot_states"][0]["attempt_id"].startswith(arm)
    assert binding["comparison_gate"]["observed_efficiency_qualified"] is False  # Authored.
    assert binding["new_provider_dispatch"] is binding["official_reward_created"] is False
    assert binding["billing_total_verified"] is binding["completion_authority"] is False


def test_missing_receipt_is_unknown_instead_of_zero_or_an_omitted_slot():
    declaration, census, _ = _population()
    census["slots"][0].update(receipt=None, receipt_sha256=None, terminal=None, terminal_sha256=None)
    report = project(declaration, census)
    usage = summarize_supervisor_usage(report)
    assert report["provider_invocations"] == []
    assert report["unreceipted_provider_attempt"] is None
    assert usage["total_tokens"] is None
    binding = report["world_packet_attempt_collection"]
    assert binding["declared_arm_slot_count"] == binding["missing_arm_receipt_count"] == 1
    assert binding["arm_slot_states"][0]["dispatch_state"] == "unknown"
    assert binding["durable_population_gate"]["qualified"] is False


@pytest.mark.parametrize("marker", [True, None])
def test_global_unknown_calls_are_not_attributed_away_from_selected_arm(marker):
    declaration, census, _ = _population()
    census["unreceipted_provider_attempt"] = marker
    report = project(declaration, census, arm="candidate")
    usage = summarize_supervisor_usage(report)
    assert report["unreceipted_provider_attempt"] is marker
    assert usage["total_tokens"] is None
    assert usage["known_token_subtotals"]["total_tokens"] == 80


@pytest.mark.parametrize("census_complete", [False, None])
def test_incomplete_census_cannot_become_complete_receipted_cost(census_complete):
    declaration, census, _ = _population()
    census["supplied_attempt_census_complete"] = census_complete
    report = project(declaration, census)
    assert report["unreceipted_provider_attempt"] is None
    assert summarize_supervisor_usage(report)["total_tokens"] is None


def test_native_cost_can_be_known_while_missing_terminal_blocks_experiment_gate():
    declaration, census, verifiers = _population()
    census["slots"][0].update(terminal=None, terminal_sha256=None)
    report = project(declaration, census, verifiers)
    assert summarize_supervisor_usage(report)["total_tokens"] == 120
    assert report["world_packet_attempt_collection"]["durable_population_gate"]["qualified"] is False
    assert report["world_packet_attempt_collection"]["comparison_gate"]["criteria_met"] is False


def test_failed_provider_cost_survives_without_a_verified_output_or_reward():
    declaration, census, _ = _population()
    row = census["slots"][0]
    row["receipt"].update(status="failed", error_type="RuntimeError")
    row["terminal"].update(outcome="raised", error_type="RuntimeError")
    row["output_sha256"] = None
    _reseal_join(row)
    report = project(declaration, census)
    usage = summarize_supervisor_usage(report)
    assert usage["total_tokens"] == 120
    assert report["world_packet_attempt_collection"]["arm_slot_states"][0]["outcome"] == "failed"
    assert report["world_packet_attempt_collection"]["comparison_gate"]["observed_efficiency_qualified"] is False


def test_changed_other_arm_receipt_is_refused_before_selecting_requested_arm():
    declaration, census, verifiers = _population()
    pins = _pins(declaration, census, verifiers)
    census["slots"][1]["receipt"]["status"] = "failed"
    with pytest.raises(ValueError, match="slot_census_hash_mismatch"):
        project_collected_supervisor_report(declaration, census, verifiers, arm="baseline", **pins)


@pytest.mark.parametrize("arm", [None, True, "full", "other", []])
def test_arm_is_explicit_and_does_not_select_another_benchmark_route(arm):
    declaration, census, _ = _population()
    with pytest.raises(ValueError, match="explicit_collected_supervisor_arm_required"):
        project(declaration, census, arm=arm)


def test_projection_is_detached_and_raw_receipts_remain_private_accounting_inputs():
    declaration, census, verifiers = _population()
    original = deepcopy(census)
    report = project(declaration, census, verifiers)
    report["provider_invocations"][0]["private_observation"] = "retained actual receipt payload"
    assert census == original
    assert "retained actual receipt payload" not in json.dumps(summarize_supervisor_usage(report))


def test_projected_report_reaches_actual_harbor_measured_usage_function():
    from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import measured_usage
    declaration, census, verifiers = _population()
    report = project(declaration, census, verifiers)
    usage = measured_usage(report)
    assert usage["total_tokens"] == 120 and usage["harbor_usage_exportable"] is True
    assert usage["billing_total_verified"] is False
