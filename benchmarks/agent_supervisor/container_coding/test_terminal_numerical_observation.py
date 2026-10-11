"""Planner delivery joins observed from authored bytes, with no provider calls."""
from copy import deepcopy
import hashlib
import json
import os

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_numerical_observation as observer
from benchmarks.agent_supervisor.container_coding.terminal_numerical_policy import PROFILE
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import PROMPT_GOAL_PROVIDER_REQUEST_SCHEMA
from ipfs_accelerate_py.agent_supervisor.runtime import security_autoencoder_advisor as files


def wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def authored(tmp_path, *, arm="full", wrapped=False):
    state = tmp_path / "state"
    state.mkdir()
    summary = dict(schema=observer.SUMMARY_SCHEMA, profile=PROFILE, source_tree_id="authored-source-tree",
        pending_obligation_count=2, proof_authority=False, completion_authority=False,
        execution_authority=False, source_semantics_verified=False, whole_program_semantics_verified=False,
        source_obligations_discharged=0)
    numerical = dict(profile=PROFILE, summary=summary, reports=[{"sha256": "c" * 64}])
    numerical_raw = wire(numerical)
    numerical["receipt_sha256"] = digest(numerical_raw)
    if arm == "full":
        (state / "numerical-formalization").mkdir()
        (state / "numerical-formalization/receipt.json").write_bytes(numerical_raw)
        (state / "numerical-formalization-selection.json").write_bytes(wire({"profile": PROFILE}))
    payload = dict(schema=PROMPT_GOAL_PROVIDER_REQUEST_SCHEMA,
        constraints={"constraint_summaries": [deepcopy(summary)] if arm == "full" else []},
        public_instruction="PRIVATE_TEST_SENTINEL_DO_NOT_EXPORT: improve the authored function")
    if wrapped:
        payload = dict(schema="intent-plan-provider-request@1", graph_request=payload,
                       intent_contract={"authored": True})
    raw = wire(payload)
    (state / "planner-provider-request.json").write_bytes(raw)
    selected = dict(provider="grok_cli", model="grok-4.7", reasoning_effort="high")
    receipt = dict(schema="router-implementation-invocation@1", invocation_id="a" * 32,
        purpose="planning", status="provider_returned", router_calls=1, completion_authority=False,
        prompt_sha256=digest(raw), native_prompt_sha256=digest(raw), model_prompt_sha256="b" * 64,
        prompt_bytes=len(raw), native_prompt_bytes=len(raw), model_prompt_bytes=len(raw) + 100,
        **selected)
    report = dict(arm=arm, provider_profile=selected, provider_invocations=[dict(phase="planning", **receipt)],
        initial_context={"numerical_formalization": numerical} if arm == "full" else {},
        planning=dict(qualified=True, provider_calls=1, model_request_sha256=digest(raw),
                      execution_receipt=receipt, initial_indexed_context={"supplied_to_router": True}
                      if arm == "full" else None, **selected))
    return state, report, payload


def rebind_request(state, report, payload):
    raw = wire(payload)
    (state / "planner-provider-request.json").write_bytes(raw)
    report["planning"]["model_request_sha256"] = digest(raw)
    receipt = report["planning"]["execution_receipt"]
    receipt.update(prompt_sha256=digest(raw), native_prompt_sha256=digest(raw),
                   prompt_bytes=len(raw), native_prompt_bytes=len(raw))
    report["provider_invocations"] = [dict(phase="planning", **receipt)]


def observe(state, report):
    return observer.observe_numerical_planning(state=state, report=report, profile=PROFILE)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_exact_request_dispatch_join_exports_only_closed_metadata(tmp_path, arm, wrapped):
    state, report, _ = authored(tmp_path, arm=arm, wrapped=wrapped)
    result = observe(state, report)
    assert result["status"] == "verified", result
    assert result["request_checked"] and result["request_hash_join_verified"]
    assert result["planning_invocation_join_verified"]
    assert result["planner_request_sha256"] == digest((state / "planner-provider-request.json").read_bytes())
    assert result["planning_execution_receipt_sha256"] == digest(wire(report["planning"]["execution_receipt"]))
    assert result["model_prompt_sha256"] != result["planner_request_sha256"]
    assert result["model_summary_equality_verified"] is False
    assert result["delivery_boundary"] == "native_planner_request_submitted_to_router"
    assert result["live_planner_delivery_verified"] is (arm == "full")
    assert result["no_index_absence_verified"] is (arm == "no-index")
    assert result["numerical_summary_count"] == (1 if arm == "full" else 0)
    assert result["source_obligations_discharged"] == result["observer_provider_calls"] == 0
    for key in ("proof_authority", "execution_authority", "completion_authority", "source_semantics_verified",
                "worker_delivery_verified", "raw_prompts_exported", "provider_free"):
        assert result[key] is False
    assert "PRIVATE_TEST_SENTINEL" not in json.dumps(result)
    assert all(value is None or type(value) in (str, bool, int) for value in result.values())
    if arm == "full":
        assert result["exact_summary_match"]
        assert result["numerical_summary_sha256"] == digest(wire(report["initial_context"]["numerical_formalization"]["summary"]))
    else:
        assert not result["exact_summary_match"] and result["numerical_receipt_sha256"] is None


def test_legacy_opt_out_does_not_read_any_evidence():
    assert observer.observe_numerical_planning(state=object(), report=object(), profile=None) is None


@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_legacy_driver_does_not_invoke_observer(tmp_path, monkeypatch, arm):
    from benchmarks.agent_supervisor.container_coding.test_formalization_resource_profile import _simulate
    monkeypatch.setattr(observer, "observe_numerical_planning",
                        lambda **kwargs: pytest.fail("legacy driver read new numerical evidence"))
    report, creates, _, _, _, _ = _simulate(tmp_path, monkeypatch, arm=arm)
    assert creates and "error" not in report and "numerical_planning_observation" not in report


@pytest.mark.parametrize("mutation,reason,status", [
    ("missing_execution", "planning_execution_receipt_missing", "unavailable"),
    ("missing_hash", "planning_prompt_hashes_missing", "unavailable"),
    ("wrong_hash", "planning_request_hash_mismatch", "mismatch"),
    ("wrong_size", "planning_request_hash_mismatch", "mismatch"),
    ("unqualified", "successful_planner_dispatch_not_observed", "unavailable"),
    ("boolean_calls", "successful_planner_dispatch_not_observed", "unavailable"),
    ("provider_error", "successful_planner_dispatch_not_observed", "unavailable"),
    ("provider_drift", "planning_provider_selection_mismatch", "mismatch"),
    ("missing_invocations", "planning_invocation_unobserved", "unavailable"),
    ("conflicting_invocation", "planning_invocation_receipts_differ", "mismatch"),
    ("missing_index_delivery", "initial_index_delivery_missing", "unavailable"),
    ("receipt_changed", "numerical_receipt_file_mismatch", "mismatch"),
    ("selection_changed", "numerical_selection_mismatch", "mismatch"),
])
def test_partial_or_conflicting_evidence_never_claims_live_delivery(tmp_path, mutation, reason, status):
    state, report, _ = authored(tmp_path)
    planning = report["planning"]
    receipt = planning["execution_receipt"]
    if mutation == "missing_execution": del planning["execution_receipt"]
    elif mutation == "missing_hash": del receipt["model_prompt_sha256"]
    elif mutation == "wrong_hash": receipt["native_prompt_sha256"] = "0" * 64
    elif mutation == "wrong_size": receipt["prompt_bytes"] = True
    elif mutation == "unqualified": planning["qualified"] = False
    elif mutation == "boolean_calls": planning["provider_calls"] = True
    elif mutation == "provider_error": receipt["status"] = "provider_error"
    elif mutation == "provider_drift": report["provider_profile"]["provider"] = "codex_cli"
    elif mutation == "missing_invocations": report["provider_invocations"] = []
    elif mutation == "conflicting_invocation": report["provider_invocations"].append(dict(phase="planning", **{**receipt, "invocation_id": "e" * 32}))
    elif mutation == "missing_index_delivery": planning["initial_indexed_context"] = None
    elif mutation == "receipt_changed": (state / "numerical-formalization/receipt.json").write_text("{}")
    elif mutation == "selection_changed": (state / "numerical-formalization-selection.json").write_text("{}")
    result = observe(state, report)
    assert (result["status"], result["reason"]) == (status, reason)
    assert not result["live_planner_delivery_verified"] and not result["no_index_absence_verified"]


@pytest.mark.parametrize("mutation,reason", [
    ("changed", "numerical_summary_missing_or_different"),
    ("missing", "numerical_summary_missing_or_different"),
    ("duplicate", "numerical_summary_missing_or_different"),
    ("outside", "numerical_summary_outside_constraint_slot"),
    ("claims_authority", "numerical_summary_claims_authority"),
])
def test_summary_must_be_exact_and_in_the_actual_constraint_slot(tmp_path, mutation, reason):
    state, report, payload = authored(tmp_path)
    summaries = payload["constraints"]["constraint_summaries"]
    if mutation == "changed": summaries[0]["pending_obligation_count"] += 1
    elif mutation == "missing": summaries.clear()
    elif mutation == "duplicate": summaries.append(deepcopy(summaries[0]))
    elif mutation == "outside": payload["other"] = summaries.pop()
    else:
        report["initial_context"]["numerical_formalization"]["summary"]["proof_authority"] = True
        summaries[0]["proof_authority"] = True
    rebind_request(state, report, payload)
    result = observe(state, report)
    assert (result["status"], result["reason"]) == ("mismatch", reason)
    assert not result["live_planner_delivery_verified"]


@pytest.mark.parametrize("mutation", ["summary", "indexed", "selection", "receipt", "dangling_symlink"])
def test_no_index_requires_checked_absence_not_just_arm_label(tmp_path, mutation):
    state, report, payload = authored(tmp_path, arm="no-index")
    if mutation == "summary":
        payload["constraints"]["constraint_summaries"] = [{"schema": observer.SUMMARY_SCHEMA}]
        rebind_request(state, report, payload)
    elif mutation == "indexed": report["planning"]["initial_indexed_context"] = {"supplied_to_router": True}
    elif mutation == "selection": (state / "numerical-formalization-selection.json").write_text("{}")
    elif mutation == "dangling_symlink": (state / "numerical-formalization-selection.json").symlink_to(state / "missing")
    else:
        (state / "numerical-formalization").mkdir()
        (state / "numerical-formalization/receipt.json").write_text("{}")
    result = observe(state, report)
    assert (result["status"], result["reason"]) == ("mismatch", "no_index_numerical_context_present")
    assert not result["no_index_absence_verified"]


@pytest.mark.parametrize("mutation", ["missing", "symlink", "hardlink", "oversized", "duplicates", "nonfinite"])
def test_unsafe_or_ambiguous_request_cannot_establish_delivery(tmp_path, mutation):
    state, report, _ = authored(tmp_path)
    path = state / "planner-provider-request.json"
    if mutation == "missing": path.unlink()
    elif mutation == "oversized": path.write_bytes(b" " * (observer.MAX_REQUEST_BYTES + 1))
    elif mutation == "duplicates": path.write_text('{"schema":"one","schema":"two"}')
    elif mutation == "nonfinite": path.write_text('{"bad":NaN}')
    else:
        target = state / "other.json"
        path.rename(target)
        if mutation == "symlink": path.symlink_to(target)
        else: os.link(target, path)
    result = observe(state, report)
    assert result["status"] in {"mismatch", "unavailable"}
    assert not result["live_planner_delivery_verified"]


def test_request_replacement_during_observation_fails_closed(tmp_path, monkeypatch):
    state, report, _ = authored(tmp_path)
    read, request_reads = files._read, []
    def replaced(path, maximum):
        if path.name == "planner-provider-request.json":
            request_reads.append(path)
            if len(request_reads) == 2:
                path.write_text("{}")
        return read(path, maximum)
    monkeypatch.setattr(files, "_read", replaced)
    result = observe(state, report)
    assert (result["status"], result["reason"]) == ("mismatch", "planner_request_changed_during_observation")
    assert not result["live_planner_delivery_verified"]


@pytest.mark.parametrize("arm", ["full", "no-index"])
@pytest.mark.parametrize("status", ["verified", "mismatch", "unavailable"])
def test_driver_requires_selected_observation_before_creating_runtime(tmp_path, monkeypatch, arm, status):
    from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
    from benchmarks.agent_supervisor.container_coding.test_formalization_resource_profile import _simulate
    run, calls = driver.run, []
    def selected(**kwargs):
        return run(**kwargs, numerical_formalization_profile=PROFILE)
    def observed(**kwargs):
        calls.append(kwargs)
        assert kwargs["report"]["planning"]["qualified"] is True
        return {"schema": observer.SCHEMA, "status": status, "provider_free": False}
    monkeypatch.setattr(driver, "run", selected)
    monkeypatch.setattr(observer, "observe_numerical_planning", observed)
    report, creates, _, _, _, _ = _simulate(tmp_path, monkeypatch, arm=arm)
    assert len(calls) == 1 and calls[0]["profile"] == PROFILE
    assert report["numerical_planning_observation"]["status"] == status
    assert bool(creates) is (status == "verified")
    if status != "verified":
        assert report["error_phase"] == "planning" and not report["task_completed"]
        assert "verified planner delivery" in report["error"]["message"]
