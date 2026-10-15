"""Cheap authored owner-binding receipts constrain usage coverage only.

No native producer, subprocess, model, or benchmark verifier is constructed.
"""

from copy import deepcopy
import json

import pytest

import benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent as harbor
import benchmarks.agent_supervisor.container_coding.terminal_container_supervisor as terminal
import ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_usage as usage
from test.api.test_terminal_decision_usage import _attempt, _dialogue, _native


def _case():
    native = _native()
    dialogue = _dialogue(_attempt(native))
    owner = {
        "schema": usage.OWNER_BINDING_SCHEMA, "binding_id": "binding-1", "task_id": "fixture-task",
        "native_prompt_sha256": "a" * 64, "native_prompt_bytes": 321,
        "parent_artifact": ".runtime/semantic/worker-context.json", "parent_sha256": "b" * 64,
        "launch_policy_artifact": "/app/.runtime/semantic-decisions/policy.json",
        "launch_policy_sha256": "c" * 64,
        "nomination_artifact": "/opt/ipfs-supervisor/policies/explicit nomination.json",
        "nomination_sha256": "d" * 64, "policy_sha256": "e" * 64,
        "provider_called": False, "completion_authority": False,
    }
    dialogue.update(task_id=owner["task_id"], parent_sha256=owner["parent_sha256"],
                    policy_sha256=owner["policy_sha256"],
                    # The prepared prompt includes literal suffixes; matching
                    # must use worker_launch's original input, not this field.
                    native_prompt_sha256="f" * 64,
                    worker_launch={
                        "original_native_prompt_sha256": owner["native_prompt_sha256"],
                        "original_native_prompt_bytes": owner["native_prompt_bytes"],
                        "launch_policy_artifact": owner["launch_policy_artifact"],
                        "launch_policy_sha256": owner["launch_policy_sha256"],
                    })
    return {
        "provider_invocations": [native], "semantic_decision_dialogues": [dialogue],
        "semantic_decision_owner_bindings": [owner],
        "semantic_decision_dialogue_expected": True,
        "semantic_decision_owner_binding_expected": True,
        "semantic_decision_expected_task_id": owner["task_id"],
        "semantic_decision_nomination_sha256": owner["nomination_sha256"],
    }


def _measure(report):
    before = deepcopy(report)
    terminal._final_usage_accounting(report)
    assert {key: value for key, value in report.items() if key != "provider_usage_accounting"} == before
    final, measured = report["provider_usage_accounting"], harbor.measured_usage(report)
    assert final["report_provider_total_tokens"] == measured["total_tokens"]
    assert final["semantic_decision_coverage"] == measured["semantic_decision_coverage"]
    return final, measured


def _gap(report, reason):
    final, measured = _measure(report)
    coverage = final["semantic_decision_coverage"]
    assert final["reconciliation"]["known_provider_token_subtotal"] == 120
    assert measured["known_provider_token_subtotal"] == 120
    assert final["report_provider_total_tokens"] is None and measured["provider_calls"] is None
    assert coverage["owner_binding_coverage_complete"] is False
    assert coverage["authoritative_usage_coverage_complete"] is False
    assert coverage["provider_call_count_complete"] is False
    assert measured["all_invocations_receipted"] is False
    assert reason in {row["reason"] for row in coverage["issues"]}
    return coverage


@pytest.mark.parametrize("order", ["owner_first", "native_first", "dialogue_first"])
def test_owner_receipt_collection_deduplicates_exact_rows_without_an_extra_provider_charge(order):
    report = _case()
    owner = report.pop("semantic_decision_owner_bindings")[0]
    dialogue = report.pop("semantic_decision_dialogues")[0]
    native = report["provider_invocations"].pop()
    observations = {"owner_first": [owner, dialogue, native], "native_first": [native, owner, dialogue],
                    "dialogue_first": [dialogue, native, owner]}[order]
    for item in observations + [deepcopy(owner), deepcopy(dialogue)]:
        assert usage.collect_provider_usage_observation(report, item) is True
    assert len(report["semantic_decision_owner_bindings"]) == 1
    assert len(report["provider_invocations"]) == 1
    final, measured = _measure(report)
    coverage = final["semantic_decision_coverage"]
    assert coverage["owner_bindings_observed"] == coverage["owner_bound_dialogues_observed"] == 1
    assert coverage["owner_bindings_without_dialogue_count"] == 0
    assert coverage["owner_binding_coverage_complete"] is True
    assert measured["provider_calls"] == 1 and measured["total_tokens"] == 120


def test_owner_receipt_is_observation_only_and_never_an_additional_token_charge():
    report = _case()
    owner = report["semantic_decision_owner_bindings"][0]
    owner.update(usage={"input_tokens": 999999, "output_tokens": 999999},
                 provider_receipt={"native_invocation": _native(2)},
                 prompt="sensitive prompt", credential="sensitive secret")
    final, measured = _measure(report)
    assert measured["provider_calls"] == 1 and measured["total_tokens"] == 120
    assert "sensitive" not in json.dumps(final) and "sensitive" not in json.dumps(measured)


@pytest.mark.parametrize("field", [
    "task_id", "parent_sha256", "policy_sha256", "original_native_prompt_sha256",
    "original_native_prompt_bytes", "launch_policy_artifact", "launch_policy_sha256", "worker_launch",
])
def test_each_dialogue_must_match_owner_task_parent_policy_and_original_worker_input(field):
    report = _case()
    dialogue = report["semantic_decision_dialogues"][0]
    if field == "worker_launch":
        dialogue.pop(field)
    elif field in {"task_id", "parent_sha256", "policy_sha256"}:
        dialogue[field] = "foreign-task" if field == "task_id" else "0" * 64
    else:
        dialogue["worker_launch"][field] = (322 if field == "original_native_prompt_bytes" else
            "/foreign/policy.json" if field == "launch_policy_artifact" else "0" * 64)
    coverage = _gap(report, "dialogue_owner_binding_missing_or_foreign")
    assert coverage["owner_bound_dialogues_observed"] == 0


@pytest.mark.parametrize("field, value", [
    ("binding_id", []), ("native_prompt_bytes", True), ("native_prompt_bytes", 0),
    ("native_prompt_bytes", usage.MAX_NATIVE_PROMPT_BYTES + 1),
    ("parent_artifact", "../outside.json"), ("launch_policy_artifact", "relative.json"),
    ("nomination_artifact", "/owner/../nomination.json"), ("task_id", "\ud800"),
    ("nomination_sha256", "not-a-digest"), ("provider_called", True), ("completion_authority", True),
], ids=["invalid-id", "bool-size", "zero-size", "over-size", "parent-traversal", "relative-policy",
        "nomination-traversal", "invalid-utf8-task", "invalid-sha", "provider-authority", "completion-authority"])
def test_malformed_owner_receipt_keeps_native_subtotal_and_unknown_coverage(field, value):
    report = _case()
    report["semantic_decision_owner_bindings"][0][field] = value
    _gap(report, "invalid_owner_binding_receipt")


def test_oversized_owner_receipt_cannot_qualify_coverage():
    report = _case()
    report["semantic_decision_owner_bindings"][0]["extra"] = "x" * usage.MAX_OWNER_BINDING_BYTES
    _gap(report, "invalid_owner_binding_receipt")


@pytest.mark.parametrize("pin", ["semantic_decision_expected_task_id", "semantic_decision_nomination_sha256"])
def test_missing_independent_owner_pins_cannot_be_replaced_with_self_consistent_observations(pin):
    report = _case()
    report.pop(pin)
    _gap(report, "invalid_expected_owner_binding_pins")


@pytest.mark.parametrize("field, pin", [
    ("task_id", "semantic_decision_expected_task_id"),
    ("nomination_sha256", "semantic_decision_nomination_sha256"),
])
def test_owner_binding_must_match_independent_signed_nomination_and_admitted_task(field, pin):
    report = _case()
    owner = report["semantic_decision_owner_bindings"][0]
    owner[field] = "foreign-task" if field == "task_id" else "0" * 64
    if field == "task_id":
        report["semantic_decision_dialogues"][0]["task_id"] = owner[field]
    assert report[pin] != owner[field]
    _gap(report, "owner_binding_expected_pin_conflict")


def test_missing_owner_receipt_blocks_exact_total_even_with_complete_dialogue_native_usage():
    report = _case()
    report.pop("semantic_decision_owner_bindings")
    _gap(report, "expected_owner_binding_receipt_missing")


def test_changed_binding_identity_is_retained_as_conflicting_evidence():
    report = _case()
    owner = report.pop("semantic_decision_owner_bindings")[0]
    changed = deepcopy(owner)
    changed["native_prompt_bytes"] += 1
    for item in (owner, deepcopy(owner), changed):
        usage.collect_provider_usage_observation(report, item)
    assert len(report["semantic_decision_owner_bindings"]) == 2
    _gap(report, "owner_binding_identity_conflict")


def test_exact_duplicate_owner_records_in_external_report_are_one_observation():
    report = _case()
    report["semantic_decision_owner_bindings"].append(deepcopy(report["semantic_decision_owner_bindings"][0]))
    final, measured = _measure(report)
    assert final["semantic_decision_coverage"]["owner_bindings_observed"] == 1
    assert measured["provider_calls"] == 1 and measured["total_tokens"] == 120


def test_extra_owner_binding_cannot_establish_that_an_unobserved_child_made_no_calls():
    report = _case()
    extra = deepcopy(report["semantic_decision_owner_bindings"][0])
    extra.update(binding_id="binding-extra", native_prompt_sha256="0" * 64)
    report["semantic_decision_owner_bindings"].append(extra)
    coverage = _gap(report, "owner_binding_without_dialogue")
    assert coverage["owner_bound_dialogues_observed"] == 1
    assert coverage["owner_bindings_without_dialogue_count"] == 1


def test_distinct_owner_bindings_for_one_identical_dialogue_are_ambiguous():
    report = _case()
    extra = deepcopy(report["semantic_decision_owner_bindings"][0])
    extra["binding_id"] = "binding-extra"
    report["semantic_decision_owner_bindings"].append(extra)
    _gap(report, "ambiguous_dialogue_owner_binding")


def test_one_owner_binding_cannot_cover_two_separate_dialogue_sessions():
    report = _case()
    second = deepcopy(report["semantic_decision_dialogues"][0])
    second["session_id"] = "another-dialogue-session"
    second["attempts"] = [_attempt(_native(2), number=2)]
    second["attempts"][0]["provider_receipt"]["session_id"] = second["session_id"]
    report["semantic_decision_dialogues"].append(second)
    report["provider_invocations"].append(_native(2))
    final, measured = _measure(report)
    assert measured["known_provider_token_subtotal"] == 240
    assert measured["provider_calls"] is None and measured["total_tokens"] is None
    assert "owner_binding_reused_for_dialogues" in {
        row["reason"] for row in final["semantic_decision_coverage"]["issues"]}


def test_owner_receipt_without_dialogue_does_not_fabricate_zero_calls_or_zero_tokens():
    report = _case()
    report["provider_invocations"] = []
    report.pop("semantic_decision_dialogues")
    final, measured = _measure(report)
    assert measured["provider_calls"] is None and measured["total_tokens"] is None
    assert measured["known_provider_token_subtotal"] is None
    coverage = final["semantic_decision_coverage"]
    assert coverage["owner_bindings_without_dialogue_count"] == 1
    assert "expected_owner_bound_dialogue_missing" in {row["reason"] for row in coverage["issues"]}


def test_expected_owner_route_exposes_missing_coverage_without_any_receipts():
    report = {"semantic_decision_owner_binding_expected": True,
              "semantic_decision_expected_task_id": "fixture-task",
              "semantic_decision_nomination_sha256": "d" * 64}
    final, measured = _measure(report)
    assert measured["provider_calls"] is None and measured["total_tokens"] is None
    assert final["semantic_decision_coverage"]["owner_binding_coverage_complete"] is False


def test_prebound_dialogue_without_owner_selection_keeps_legacy_surface():
    report = _case()
    for key in ("semantic_decision_owner_binding_expected", "semantic_decision_owner_bindings",
                "semantic_decision_expected_task_id", "semantic_decision_nomination_sha256"):
        report.pop(key)
    final, measured = _measure(report)
    assert measured["provider_calls"] == 1 and measured["total_tokens"] == 120
    assert "owner_binding_expected" not in final["semantic_decision_coverage"]


def test_owner_task_key_uses_the_same_512_byte_bound_as_signed_owner_dispatch():
    report = _case()
    task = "t" * 300
    report["semantic_decision_expected_task_id"] = task
    report["semantic_decision_owner_bindings"][0]["task_id"] = task
    report["semantic_decision_dialogues"][0]["task_id"] = task
    final, measured = _measure(report)
    assert final["semantic_decision_coverage"]["owner_binding_coverage_complete"] is True
    assert measured["provider_calls"] == 1 and measured["total_tokens"] == 120
