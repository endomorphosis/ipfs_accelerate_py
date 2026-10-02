"""Explicit 384D startup selection preserves raw instructions and native admission."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as preparation
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import original, _proposal_json  # noqa: F401
from ipfs_accelerate_py.agent_supervisor.runtime import intent_384_advisor as advisor
from ipfs_accelerate_py.agent_supervisor.runtime import intent_advisor_selection as subject

PROBE = "the calculator must compute result; requires true; ensures result = old(left) + old(right) and returned."
PIN = "a" * 64


def _plan(state, prepared, monkeypatch):
    native_run = subprocess.run
    def version(argv, *args, **kwargs):
        if argv == ["codex", "--version"]:
            return subprocess.CompletedProcess(argv, 0, "codex-cli 0.158.0\n", "")
        return native_run(argv, *args, **kwargs)
    monkeypatch.setattr(subprocess, "run", version)
    prompts = []
    def provider(prompt, **kwargs):
        prompts.append(prompt)
        return dict(text=_proposal_json(prepared), observation={}, execution_receipt=None)
    result = preparation.plan(state=state, provider_callable=provider)
    assert result["qualified"] and result["provider_calls"] == 1 and len(prompts) == 1, result
    return result, prompts[0]


def _original_prompt(prepared):
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import build_prompt_goal_provider_request
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import PromptWorkflowRequest, DirectoryScanReceipt
    return build_prompt_goal_provider_request(PromptWorkflowRequest.from_dict(prepared["request"]),
        DirectoryScanReceipt.from_dict(prepared["scan"]), config=preparation._config(Path(prepared["repository"])),
        constraint_summaries=prepared["constraints"])


class TransportOwner:
    """Authored native control for transport; the optional real test uses weights."""
    def __init__(self):
        self.calls = []

    def report(self, source, **options):
        from ipfs_datasets_py.logic.intent_ir.formalize import action_contracts as codec
        raw = codec.source_to_target(source)
        binding = codec.bind_candidate_source(source, raw)
        report = dict(schema=advisor.REPORT_SCHEMA, status="source_supported_action_contract",
            source_sha256=advisor._sha(source.encode()), checkpoint_sha256=options["expected_sha256"],
            checkpoint_path=options["checkpoint_path"], snapshot_path=options["snapshot_path"],
            raw_candidate_ir=raw, native_intent_ir=binding["bound_candidate"]["document"], binding=binding,
            **advisor.FALSE)
        report["report_sha256"] = advisor._sha(advisor._wire(report))
        return report

    def prepare_intent_action_inference(self, source, **options):
        self.calls.append("prepare")
        return self.report(source, **options)

    def verify_intent_action_inference(self, report, source, **options):
        self.calls.append("verify")
        expected = self.report(source, **options)
        if report != expected:
            raise ValueError("numerical replay differs")
        return expected


@pytest.fixture
def selected(tmp_path, monkeypatch):
    config = dict(schema=advisor.CONFIG_SCHEMA, checkpoint_path="/models/intent.json",
        checkpoint_sha256=PIN, embedding_snapshot_path="/models/gte")
    path = tmp_path / "intent-config.json"
    path.write_text(json.dumps(config))
    owner = TransportOwner()
    monkeypatch.setattr(advisor, "_owner", lambda: owner)
    return path, config, owner


def saved(tmp_path, selected):
    config_path, _, _ = selected
    advice, selection, elapsed = subject.prepare_intent_384_selection(instruction=PROBE, config_path=config_path)
    assert advice["status"] == "semantic_candidate_advice" and elapsed > 0
    path = tmp_path / "saved-advice.json"
    path.write_bytes(advisor._wire(advice))
    return dict(path=path, expected_sha256=advisor._sha(path.read_bytes()), instruction=PROBE, selection=selection)


def test_disabled_selection_never_reads_configuration_or_imports_model(tmp_path, monkeypatch):
    monkeypatch.setattr(advisor, "_owner", lambda: pytest.fail("disabled model imported"))
    result, selection, elapsed = subject.prepare_intent_384_selection(
        instruction=PROBE, config_path=tmp_path / "missing", enabled=False)
    assert result["status"] == "disabled" and elapsed == 0
    assert selection == dict(schema=subject.SCHEMA, enabled=False, config_path=None, config_sha256=None)


def test_saved_candidate_replays_current_config_and_learned_contract_summary(tmp_path, selected):
    values = saved(tmp_path, selected)
    before_calls = len(selected[2].calls)
    loaded = subject.load_intent_384_selection(**values)
    assert selected[2].calls[before_calls:] == ["verify"]
    summary, checked = subject.intent_384_planner_summary(loaded, instruction=PROBE)
    payload = json.loads(summary)
    assert checked == loaded and payload["contract"]["equation"]["operator"] == "add"
    assert payload["mode"] == "shared_384_action_contract_candidate"
    assert PROBE not in summary and "embedding" not in summary and "source_text" not in summary
    assert all(payload[key] is False for key in advisor.FALSE)
    assert selected[0].read_bytes() and values["selection"]["config_sha256"] == advisor._sha(selected[0].read_bytes())


@pytest.mark.parametrize("change", ["missing", "malformed", "oversized", "symlink", "duplicate", "unknown_field", "wrong_hash"])
def test_invalid_config_is_optional_failure_without_model_call(tmp_path, selected, change):
    path, config, owner = selected
    if change == "missing": path.unlink()
    elif change == "malformed": path.write_text("{")
    elif change == "oversized": path.write_bytes(b" " * (subject.MAX_CONFIG_BYTES + 1))
    elif change == "symlink":
        target = tmp_path / "real-config.json"; path.rename(target); path.symlink_to(target)
    elif change == "duplicate": path.write_text('{"schema":"first","schema":"second"}')
    else:
        if change == "unknown_field": config["fallback_to_parser"] = True
        else: config["checkpoint_sha256"] = "main"
        path.write_text(json.dumps(config))
    result, _, _ = subject.prepare_intent_384_selection(instruction=PROBE, config_path=path)
    assert result["status"] == "fail_open_unavailable" and result["continue_planning"]
    assert owner.calls == [] and result["candidate_intent_ir"] is None


@pytest.mark.parametrize("change", ["config_bytes", "config_checkpoint", "sidecar_bytes", "sidecar_rehashed_candidate",
    "source", "selection_extra", "selection_digest", "selection_disabled", "sidecar_symlink"])
def test_changed_config_source_or_rehashed_prediction_cannot_supply_planner_advice(tmp_path, selected, change):
    values = saved(tmp_path, selected)
    if change == "config_bytes": selected[0].write_text(selected[0].read_text() + "\n")
    elif change == "config_checkpoint":
        config = deepcopy(selected[1]); config["checkpoint_path"] = "/models/other.json"
        selected[0].write_text(json.dumps(config))
    elif change == "sidecar_bytes": values["path"].write_bytes(values["path"].read_bytes() + b"\n")
    elif change == "sidecar_rehashed_candidate":
        advice = json.loads(values["path"].read_bytes())
        advice["raw_candidate_ir"]["document"]["invented_effect"] = True
        advice.pop("advice_sha256"); advisor._finish(advice)
        values["path"].write_bytes(advisor._wire(advice))
        values["expected_sha256"] = advisor._sha(values["path"].read_bytes())
    elif change == "source": values["instruction"] = PROBE.replace("+", "-")
    elif change == "selection_extra": values["selection"]["trust_saved"] = True
    elif change == "selection_digest": values["selection"]["config_sha256"] = "0" * 64
    elif change == "selection_disabled": values["selection"]["enabled"] = False
    else:
        path = values["path"]; target = tmp_path / "real-advice.json"
        path.rename(target); path.symlink_to(target)
    result = subject.load_intent_384_selection(**values)
    summary, _ = subject.intent_384_planner_summary(result, instruction=values["instruction"])
    assert result["status"] == "fail_open_unavailable" and result["continue_planning"] and summary is None
    assert result["instruction_sha256"] == advisor._sha(values["instruction"].encode())


@pytest.mark.parametrize("phase", ["prepare", "load"])
def test_configuration_is_rechecked_after_model_execution(tmp_path, selected, phase):
    path, _, owner = selected
    values = saved(tmp_path, selected) if phase == "load" else None
    verify = owner.verify_intent_action_inference
    def changed(*args, **kwargs):
        value = verify(*args, **kwargs)
        path.write_text(path.read_text() + "\n")
        return value
    owner.verify_intent_action_inference = changed
    result = (subject.load_intent_384_selection(**values) if phase == "load" else
        subject.prepare_intent_384_selection(instruction=PROBE, config_path=path)[0])
    assert result["status"] == "fail_open_unavailable" and result["report"] is None


def test_summary_over_budget_keeps_original_instruction_available(tmp_path, selected):
    advice = subject.load_intent_384_selection(**saved(tmp_path, selected))
    summary, failed = subject.intent_384_planner_summary(advice, instruction=PROBE, maximum_bytes=16)
    assert summary is None and failed["continue_planning"] and failed["raw_instruction_preserved"]


def test_configured_inference_precedes_declarations_and_reaches_original_planner(original, selected, monkeypatch):
    root, instruction, state = original
    instruction.write_text(PROBE)
    original_domains = preparation.local.local_planning_domain_declarations
    def domains(**kwargs):
        assert selected[2].calls == ["prepare", "verify"]
        return original_domains(**kwargs)
    monkeypatch.setattr(preparation.local, "local_planning_domain_declarations", domains)
    prepared = preparation.prepare(repository=root, instruction=instruction, state=state,
        intent_action_384_config=selected[0])
    assert prepared["intent_preplanning"]["before_goal_decomposition"]
    assert prepared["query"] == PROBE == (root / preparation.INSTRUCTION).read_text()
    assert prepared["request"]["intent_ir_root"] == preparation.local.content_identity(
        prepared["manifest"]["payload"]["planning_inputs"]["domain_declarations"]["intent"])
    result, prompt = _plan(state, prepared, monkeypatch)
    assert result["intent_preplanning"]["supplied_to_router"]
    summaries = json.loads(prompt)["constraints"]["constraint_summaries"]
    candidate = next(json.loads(value) for value in summaries if "shared_384_action_contract_candidate" in value)
    assert candidate["contract"]["equation"]["operator"] == "add" and not candidate["proof_authority"]


@pytest.mark.parametrize("change", ["missing_config", "stale_config", "disabled"])
def test_optional_route_failures_preserve_exact_original_planner_request(original, selected, monkeypatch, change):
    root, instruction, state = original
    instruction.write_text(PROBE)
    if change == "missing_config": selected[0].unlink()
    prepared = preparation.prepare(repository=root, instruction=instruction, state=state,
        intent_action_384_config=selected[0], disable_intent_autoencoder=change == "disabled")
    if change == "stale_config": selected[0].write_text(selected[0].read_text() + "\n")
    result, prompt = _plan(state, prepared, monkeypatch)
    assert prompt == _original_prompt(prepared)
    assert not result["intent_preplanning"]["supplied_to_router"]
    assert result["intent_preplanning"]["status"] == ("disabled" if change == "disabled" else "fail_open_unavailable")
    assert (root / preparation.INSTRUCTION).read_text() == PROBE


@pytest.mark.parametrize("other", ["intent_checkpoint_descriptor", "intent_projection_request", "enable_source_unit_autoencoder"])
def test_ambiguous_preplanning_routes_are_refused_before_repository_mutation(original, selected, other):
    root, instruction, state = original
    with pytest.raises(ValueError, match="one explicit Intent"):
        preparation.prepare(repository=root, instruction=instruction, state=state,
            intent_action_384_config=selected[0], **{other: True if other.startswith("enable") else selected[0]})
    assert not state.exists() and selected[2].calls == []


def test_real_published_checkpoint_enters_benchmark_preplanning_without_manual_advice(original, monkeypatch):
    path = os.environ.get("IR384_TEST_INTENT_ACTION_CONFIG")
    source = os.environ.get("IR384_TEST_INTENT_ACTION_INSTRUCTION")
    if not path or not source:
        pytest.skip("explicit published action checkpoint and source instruction required")
    root, instruction, state = original
    instruction.write_text(source)
    prepared = preparation.prepare(repository=root, instruction=instruction, state=state,
        intent_action_384_config=Path(path))
    assert prepared["intent_preplanning"]["status"] == "semantic_candidate_advice"
    result, prompt = _plan(state, prepared, monkeypatch)
    assert result["intent_preplanning"]["supplied_to_router"]
    assert "shared_384_action_contract_candidate" in prompt and prepared["query"] == source
    assert result["provider_calls"] == 1  # Controlled planner fixture, not a live provider benchmark.
