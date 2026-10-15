"""Authored collector/join fixtures; exact native producer replay is tested separately."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
from ipfs_accelerate_py.agent_supervisor.context.context_compiler import render_context_capsule
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextCapsule
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import render_model_prompt

pytest_plugins = ("benchmarks.agent_supervisor.container_coding.test_terminal_context_audit",)


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def dialogue_case(compiled, monkeypatch):
    original = render_context_capsule(ContextCapsule.from_dict(json.loads(compiled["capsule"].read_text())))
    prepared_path = compiled["state"] / "prepared.json"
    prepared = json.loads(prepared_path.read_text())
    prepared["repository"] = str(compiled["state"].parent)
    prepared_path.write_text(json.dumps(prepared))
    receipts, attempts = [], []
    for generation in range(2):
        prompt = "authored dialogue input " + str(generation) + " λ\r\n"
        row = {**compiled["receipts"][0], "invocation_id": "round-" + str(generation),
            "provider": "codex_cli", "model": "pinned",
            "prompt_sha256": _sha(prompt), "native_prompt_sha256": _sha(prompt),
            "prompt_bytes": len(prompt.encode()), "native_prompt_bytes": len(prompt.encode())}
        model, _ = render_model_prompt(prompt=prompt, purpose="coding", workspace=Path(row["workspace"]))
        row.update(model_prompt_sha256=_sha(model), model_prompt_bytes=len(model.encode()))
        receipts.append(row)
        call = "call-" + str(generation)
        wrapper = {"schema": "router-semantic-decision-provider@1",
            "call_id": call, "session_id": "session", "provider": "codex_cli", "model": "pinned",
            "workspace": row["workspace"], "model_prompt_sha256": row["model_prompt_sha256"],
            "provider_called": True, "native_invocation": copy.deepcopy(row)}
        attempts.append({"call_id": call, "generation": generation, "status": "provider_returned",
            "prompt_sha256": row["prompt_sha256"], "prompt_bytes": row["prompt_bytes"],
            "model_prompt_sha256": row["model_prompt_sha256"], "model_prompt_bytes": row["model_prompt_bytes"],
            "provider_receipt_bound": True, "provider_receipt": wrapper})
    dialogue = {"schema": "supervisor-semantic-decision-dialogue@1", "session_id": "session",
        "task_id": "TB-CODE-TASK", "parent_sha256": "a" * 64, "policy_sha256": "b" * 64,
        "provider": "codex_cli", "model": "pinned", "workspace": receipts[0]["workspace"],
        "invocation_attempt_count": 2, "attempts": attempts,
        "worker_launch": {"original_native_prompt_sha256": _sha(original),
                          "original_native_prompt_bytes": len(original.encode())}}
    report = {"semantic_decision_dialogue_expected": True, "semantic_decision_dialogues": [dialogue]}
    calls = []

    def replay(**kwargs):
        assert kwargs["native_prompt"] == original
        calls.append(kwargs)
        observations = [{**{key: row[key] for key in (
            "generation", "prompt_sha256", "prompt_bytes", "model_prompt_sha256", "model_prompt_bytes")},
            "model_input_verified": True, "provider_receipt_binding_validated": True,
            "provider_dispatch_reported": True,
            "native_invocation_id_sha256": _sha(row["provider_receipt"]["native_invocation"]["invocation_id"])}
            for row in kwargs["dialogue"]["attempts"]]
        return {"schema": "terminal-semantic-decision-dialogue-input-audit@1",
            "all_recorded_round_inputs_verified": True, "verified_round_count": 2,
            "recorded_round_count": 2, "attempts": observations, "raw_prompts_exported": False}

    monkeypatch.setattr(audit, "_decision_auditor", lambda: replay)
    return {**compiled, "original": original, "receipts": receipts, "report": report, "calls": calls}


def _observe(case):
    return audit.audit_terminal_context(state=case["state"], receipts=case["receipts"],
        workspace_root=case["workspace_root"], decision_report=case["report"])


def test_each_round_joins_independent_native_receipt_without_claiming_component_census(dialogue_case):
    result = _observe(dialogue_case)
    assert result["all_observed_coding_inputs_verified"] is True
    assert result["semantic_decision_inputs"]["all_observed_dialogue_inputs_verified"] is True
    assert len(result["matches"]) == 2 and len(dialogue_case["calls"]) == 1
    assert all(row["model_input_verified"] for row in result["invocations"])
    assert result["rendered_input_census"]["observations"] == []
    assert {row["reason"] for row in result["rendered_input_census"]["unavailable"]} == {"dialogue_packet_census_not_supported"}
    assert dialogue_case["original"] not in json.dumps(result)
    assert "authored dialogue input" not in json.dumps(result)
    assert result["provider_calls"] == 0 and result["completion_authority"] is False


@pytest.mark.parametrize("alteration", ["model_hash", "provider", "workspace", "native_hash"])
def test_independently_collected_native_receipt_mismatch_withholds_coverage(dialogue_case, alteration):
    row = dialogue_case["receipts"][1]
    if alteration == "model_hash":
        row["model_prompt_sha256"] = "c" * 64
    elif alteration == "provider":
        row["provider"] = "foreign"
    elif alteration == "workspace":
        row["workspace"] += "-foreign"
    else:
        row["native_prompt_sha256"] = row["prompt_sha256"] = "d" * 64
    result = _observe(dialogue_case)
    assert result["all_observed_coding_inputs_verified"] is False
    assert result["semantic_decision_inputs"]["all_observed_dialogue_inputs_verified"] is False
    assert len(result["matches"]) == 1


def test_identical_dialogue_observations_deduplicate_but_conflicts_do_not(dialogue_case):
    dialogue = dialogue_case["report"]["semantic_decision_dialogues"][0]
    dialogue_case["report"]["semantic_decision_dialogues"].append(copy.deepcopy(dialogue))
    exact = _observe(dialogue_case)
    assert exact["all_observed_coding_inputs_verified"] is True
    assert exact["semantic_decision_inputs"]["observed_dialogue_count"] == 1
    dialogue_case["report"]["semantic_decision_dialogues"][-1]["status"] = "conflicting observation"
    conflicting = _observe(dialogue_case)
    assert conflicting["all_observed_coding_inputs_verified"] is False
    assert conflicting["semantic_decision_inputs"]["observed_dialogue_count"] == 2


def test_missing_expected_dialogue_prevents_complete_legacy_input_verdict(compiled):
    result = audit.audit_terminal_context(state=compiled["state"], receipts=compiled["receipts"],
        workspace_root=compiled["workspace_root"], decision_report={"semantic_decision_dialogue_expected": True})
    assert result["any_model_input_verified"] is True
    assert result["all_observed_coding_inputs_verified"] is False
    assert result["semantic_decision_inputs"]["all_observed_dialogue_inputs_verified"] is False


def test_expected_dialogue_without_native_receipts_still_reports_missing_coverage(compiled):
    result = audit.audit_terminal_context(state=compiled["state"], receipts=[],
        workspace_root=compiled["workspace_root"], decision_report={"semantic_decision_dialogue_expected": True})
    assert result["all_observed_coding_inputs_verified"] is False
    assert result["semantic_decision_inputs"]["receipt_binding_coverage"]["all_attempts_receipted"] is False


def test_collector_transmits_only_closed_dialogue_report_fields(compiled, monkeypatch):
    captured = []
    def run(*args, **kwargs):
        captured.append(json.loads(kwargs["input_bytes"]))
        return 0, json.dumps(audit._base()).encode()
    monkeypatch.setattr(audit, "_bounded_audit_child", run)
    report = {"semantic_decision_dialogue_expected": True, "private_unrelated_state": "do not forward"}
    audit.collect_terminal_context_audit(state=compiled["state"], receipts=compiled["receipts"],
        workspace_root=compiled["workspace_root"], decision_report=report, timeout_seconds=1)
    assert captured[0]["decision_report"] == {"semantic_decision_dialogue_expected": True}
    assert "do not forward" not in json.dumps(captured)


def test_dialogue_request_limit_refuses_before_subprocess(compiled, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("oversized audit request launched")
    monkeypatch.setattr(audit, "_bounded_audit_child", forbidden)
    result = audit.collect_terminal_context_audit(state=compiled["state"], receipts=compiled["receipts"],
        workspace_root=compiled["workspace_root"], decision_report={"semantic_decision_dialogues": [{"extra": "x" * 65536}]})
    assert result["reason"] == "audit_request_exceeds_bound"


def test_actual_owner_child_accepts_expected_dialogue_envelope(compiled):
    result = audit.collect_terminal_context_audit(state=compiled["state"], receipts=compiled["receipts"],
        workspace_root=compiled["workspace_root"], decision_report={"semantic_decision_dialogue_expected": True}, timeout_seconds=10)
    assert result["any_model_input_verified"] is True
    assert result["all_observed_coding_inputs_verified"] is False
    assert result["semantic_decision_inputs"]["all_observed_dialogue_inputs_verified"] is False


def test_actual_owner_child_loads_sibling_observer_for_unqualified_dialogue(compiled):
    native = render_context_capsule(ContextCapsule.from_dict(json.loads(compiled["capsule"].read_text())))
    prepared_path = compiled["state"] / "prepared.json"
    prepared = json.loads(prepared_path.read_text())
    prepared["repository"] = str(compiled["state"].parent)
    prepared_path.write_text(json.dumps(prepared))
    report = {"semantic_decision_dialogue_expected": True, "semantic_decision_dialogues": [{
        "schema": "supervisor-semantic-decision-dialogue@1", "session_id": "authored-incomplete",
        "worker_launch": {"original_native_prompt_sha256": _sha(native)}}]}
    result = audit.collect_terminal_context_audit(state=compiled["state"], receipts=compiled["receipts"],
        workspace_root=compiled["workspace_root"], decision_report=report, timeout_seconds=10)
    replay = result["semantic_decision_inputs"]["dialogues"][0]
    assert replay["schema"] == "terminal-semantic-decision-dialogue-input-audit@1"
    assert replay["reason"] == "invalid_dialogue_identity"
    assert result["all_observed_coding_inputs_verified"] is False


def test_driver_forwards_expected_dialogue_and_owner_pins_with_finalization_budget(compiled, monkeypatch):
    from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
    captured = []
    def collect(**kwargs):
        captured.append(kwargs)
        return audit._base()
    monkeypatch.setattr(audit, "collect_terminal_context_audit", collect)
    report = {"arm": "full", "provider_invocations": compiled["receipts"],
        "semantic_decision_dialogue_expected": True, "semantic_decision_owner_binding_expected": True,
        "semantic_decision_expected_task_id": "signed-task", "semantic_decision_nomination_sha256": "e" * 64}
    driver._final_context_audit(report, state=compiled["state"], deadline=time.monotonic() + 5)
    assert captured[0]["decision_report"] == {key: value for key, value in report.items() if key in audit.DECISION_REPORT_FIELDS}
    assert 0 <= captured[0]["timeout_seconds"] <= 4


def test_required_suffix_loaders_use_original_stdin_and_keep_order(monkeypatch):
    from benchmarks.agent_supervisor.container_coding import semantic_decision_input_audit as replay
    from ipfs_accelerate_py.agent_supervisor.runtime import doctor_residual_context, router_public_instruction
    calls = []
    original = "original native stdin with λ and spaces  \r\n"
    blocks = ("\r\nDoctor residual café  \r\n", "\nPublic requirements 雪  \n")
    launch = {}
    for index, (key, module, name) in enumerate((
        ("doctor_residual_context", doctor_residual_context, "load_doctor_residual_advisory"),
        ("public_instruction", router_public_instruction, "load_public_instruction"))):
        retained = {"artifact": "/owner/" + key + ".json", "artifact_sha256": str(index + 1) * 64,
            "task_cid": "task:fixture", "source_freshness_verified": True, "historical_replay": False}
        launch[key] = retained
        def load(_key=key, _index=index, _retained=retained, **kwargs):
            calls.append((_key, kwargs))
            return blocks[_index], {**_retained, "source_freshness_verified": False, "historical_replay": True}
        monkeypatch.setattr(module, name, load)
    suffix = "".join(blocks)
    launch.update(required_suffix_sha256=_sha(suffix), required_suffix_bytes=len(suffix.encode()))
    assert replay._suffix(launch=launch, native_prompt=original, repository=Path("/owner")) == suffix
    assert [key for key, _ in calls] == ["doctor_residual_context", "public_instruction"]
    assert all(kwargs["prompt"] == original and kwargs["require_current_source"] is False for _, kwargs in calls)
