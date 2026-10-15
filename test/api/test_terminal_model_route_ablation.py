"""Explicit model-route selection preserves the admitted full-arm boundaries.

Authored process, owner, archive, and trial fixtures exercise the real driver,
adapter, and host configuration. No provider, Docker task, hidden verifier,
tokenizer discovery, or benchmark acceptance is exercised by these tests.
"""
from __future__ import annotations

import asyncio
import builtins
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from benchmarks.agent_supervisor.container_coding import (
    container_worker_deployment as deployment,
)
from benchmarks.agent_supervisor.container_coding import (
    full_supervisor_benchmark as benchmark,
)
from benchmarks.agent_supervisor.container_coding import (
    full_supervisor_harbor_agent as adapter,
)
from benchmarks.agent_supervisor.container_coding import (
    native_codex_baseline as native,
)
from benchmarks.agent_supervisor.container_coding import (
    terminal_container_supervisor as driver,
)
from harbor.models.agent.context import AgentContext
from harbor.models.job.config import JobConfig

from test.api import test_terminal_prompt_tokenizer_forwarding as driver_fixtures
from test.api.test_benchmark_prompt_options import _retained_trial
from test.api.test_terminal_adapter_deadline import Clock, Environment, _install
from test.api.test_terminal_prompt_tokenizer_forwarding import (
    OPTIONS,
    SEMANTIC_ARGUMENTS,
    SEMANTIC_OPTIONS,
    TOKEN_ARGUMENTS,
)
from test.api.test_terminal_timeout_configuration import (
    _authored_prepare_dataset,
    _timeout_runtime_archive,
)

supervisor = driver_fixtures.supervisor


@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_auto_policy_preserves_empty_deployment_arguments(arm):
    assert deployment.doctor_dispatch_arguments(arm=arm) == []
    assert deployment.doctor_dispatch_arguments(arm=arm, doctor_dispatch_policy="auto") == []


def test_model_only_policy_is_an_explicit_full_arm_argument():
    assert deployment.doctor_dispatch_arguments(arm="full", doctor_dispatch_policy="model-only") == [
        "--doctor-dispatch-policy", "model-only",
    ]


@pytest.mark.parametrize("arm,policy", [
    ("full", None), ("full", True), ("full", 0), ("full", []),
    ("full", ""), ("full", "AUTO"), ("full", "model_only"),
    ("no-index", "model-only"), ("unknown", "auto"), (None, "auto"),
])
def test_invalid_policy_refuses_all_entrypoints_before_identity_or_io(tmp_path, monkeypatch, arm, policy):
    def unexpected(*_args, **_kwargs):
        pytest.fail("invalid dispatch policy reached identity, filesystem, import, or process work")

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "harbor" or name.startswith("harbor."):
            unexpected()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(driver.os, "geteuid", unexpected)
    monkeypatch.setattr(benchmark.subprocess, "run", unexpected)
    unread = SimpleNamespace(resolve=unexpected)
    options = {"arm": arm, "doctor_dispatch_policy": policy}
    with pytest.raises(ValueError):
        deployment.doctor_dispatch_arguments(**options)
    with pytest.raises(ValueError):
        driver.run(instruction=unread, state=unread, **options)
    with pytest.raises(ValueError):
        adapter.FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name="gpt-5.6-sol",
                                    runtime_archive=str(tmp_path / "missing-archive"), **options)
    with pytest.raises(ValueError):
        benchmark.config_for(unread, tmp_path / "output", unread, **options)
    with pytest.raises(ValueError):
        benchmark.prepare(dataset=unread, output=tmp_path / "output", archive=unread, **options)
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize("route", ["doctor_candidate", "doctor_contract_candidate", "model_router"])
def test_default_full_driver_keeps_real_dispatch_choice_and_candidate_authority(supervisor, monkeypatch, route):
    state, planned, signed = supervisor
    calls = []
    doctor = {"route": route, "status": "residual" if route == "model_router" else "candidate_ready",
              "provider_calls": 0, "artifact": "/owner/candidate.json", "sha256": "b" * 64,
              "task_cid": "task"}

    def dispatch(**kwargs):
        calls.append(kwargs)
        return doctor

    monkeypatch.setattr(driver, "prepare_terminal_doctor_dispatch", dispatch)
    result = driver.run(instruction=Path("/fixture/instruction"), state=state, arm="full")
    assert result["task_completed"] is True
    assert len(calls) == len(planned) == len(signed) == 1
    assert calls[0] == {"repository": Path("/app"), "state": state, "admission": {},
                        "task_cid": "task", "contract_profile": "wsgi-header-controls@1"}
    assert result["doctor_dispatch_policy"] == "auto"
    assert result["doctor_dispatch_observation"] == {
        "policy": "auto", "selection_performed": True, "skip_reason": None,
    }
    assert result["doctor_dispatch"] == doctor
    assert result["implementation_route"] == route
    assert result["phases"]["doctor_seconds"] >= 0
    if route == "model_router":
        assert "--public-instruction-artifact" in signed[0]
        assert "--model" in signed[0]
    else:
        flag = "--doctor-candidate-artifact" if route == "doctor_candidate" else "--doctor-contract-artifact"
        assert flag in signed[0]
        assert "--model" not in signed[0]
        assert "--public-instruction-artifact" not in signed[0]


@pytest.mark.parametrize("advanced", [False, True])
def test_model_only_keeps_planning_context_admission_and_ordinary_signed_model_command(supervisor, monkeypatch, advanced):
    state, planned, signed = supervisor
    calls, observed = [], {}
    requirements = Path("/already-deployed/requirements.json")

    for name in ("prepare", "initial_context", "plan", "context"):
        original = getattr(driver.preparation, name)

        def wrapped(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            observed[_name] = kwargs
            return _original(*args, **kwargs)

        monkeypatch.setattr(driver.preparation, name, wrapped)

    verify = driver.verify_local_benchmark_admission

    def admission(*args, **kwargs):
        calls.append("admission")
        observed["admission"] = kwargs
        return verify(*args, **kwargs)

    monkeypatch.setattr(driver, "verify_local_benchmark_admission", admission)
    monkeypatch.setattr(driver, "prepare_terminal_doctor_dispatch",
                        lambda **_: pytest.fail("model-only must not perform Doctor selection"))
    implementation = driver.implementation_argv

    def command(**kwargs):
        calls.append("command")
        observed["command"] = kwargs
        return implementation(**kwargs)

    monkeypatch.setattr(driver, "implementation_argv", command)
    from ipfs_accelerate_py.agent_supervisor.runtime import router_public_instruction

    public = router_public_instruction.prepare_public_instruction_context

    def instruction(**kwargs):
        calls.append("public_instruction")
        observed["public_instruction"] = kwargs
        return public(**kwargs)

    monkeypatch.setattr(router_public_instruction, "prepare_public_instruction_context", instruction)
    open_owner = driver.open_existing_native_owner

    @contextmanager
    def owner(**kwargs):
        calls.append("owner")
        observed["owner"] = kwargs
        with open_owner(**kwargs) as current:
            yield current

    monkeypatch.setattr(driver, "open_existing_native_owner", owner)
    create = driver.AdmittedBenchmarkRuntime.create

    def runtime(*args, **kwargs):
        calls.append("signed_runtime")
        observed["runtime"] = kwargs
        return create(*args, **kwargs)

    monkeypatch.setattr(driver.AdmittedBenchmarkRuntime, "create", runtime)
    options = {**OPTIONS, **SEMANTIC_OPTIONS} if advanced else {}
    result = driver.run(instruction=Path("/fixture/instruction"), state=state, arm="full",
                        doctor_dispatch_policy="model-only", intent_requirement_contract=requirements,
                        timeout_seconds=300, **options)
    assert calls == ["prepare", "initial_context", "plan", "context", "admission", "command",
                     "public_instruction", "owner", "signed_runtime"]
    assert observed["prepare"]["intent_requirement_contract"] == requirements
    assert observed["prepare"]["max_total_agent_seconds"] == 300
    assert observed["admission"] == {"initial": True}
    assert observed["command"]["doctor"] is None
    assert observed["command"]["semantic_repository"] == Path("/app")
    assert observed["public_instruction"] == {
        "repository": Path("/app"), "admission": {}, "task_cid": "task",
        "source_path": driver.preparation.INSTRUCTION, "expected_source_sha256": "b" * 64,
    }
    assert observed["owner"]["execution_routes"] == {"TASK": driver.GROK_CODEX_EXECUTION_MODE}
    launch = observed["runtime"]
    assert launch["admission"] == {}
    assert launch["implement"] is True
    assert launch["context_bundle"] == {"fixture": True}
    assert launch["max_task_attempts"] == 1
    assert launch["candidate_runner_argv"] == (str(driver.VALIDATOR),)
    assert launch["refresh_context_on_completion"] is True
    assert len(planned) == len(signed) == 1
    assert signed[0][0] == str(driver.ROUTER)
    assert signed[0][signed[0].index("--model") + 1] == driver.preparation.MODEL
    assert signed[0][signed[0].index("--reasoning-effort") + 1] == driver.preparation.REASONING
    assert signed[0][signed[0].index("--semantic-repository") + 1] == "/app"
    assert signed[0][-6:] == ["--public-instruction-artifact", "/app/.runtime/literal-instruction.json",
                             "--public-instruction-sha256", "c" * 64, "--public-instruction-task-cid", "task"]
    assert not any(arg.startswith("--doctor-") for arg in signed[0])
    if advanced:
        start = signed[0].index("--prompt-tokenizer-artifact")
        assert signed[0][start:start + 6] == TOKEN_ARGUMENTS + ["--max-input-tokens", "4096"]
        start = signed[0].index("--semantic-slot-policy-revision")
        assert signed[0][start:start + 3] == SEMANTIC_ARGUMENTS
        assert planned[0][-6:] == TOKEN_ARGUMENTS + ["--max-input-tokens", "4096"]
    else:
        assert "--prompt-tokenizer-artifact" not in signed[0]
        assert "--semantic-slot-policy-revision" not in signed[0]
    assert "--semantic-context-projection" not in planned[0]
    assert "--doctor-dispatch-policy" not in signed[0]
    assert result["implementation_route"] == "model_router"
    assert result["doctor_dispatch_policy"] == "model-only"
    assert result["doctor_dispatch_observation"] == {
        "policy": "model-only", "selection_performed": False, "skip_reason": "explicit_model_only_ablation",
    }
    assert "doctor_dispatch" not in result
    assert "doctor_seconds" not in result["phases"]
    assert result["max_total_agent_seconds"] == 300
    assert result["reserved_cleanup_seconds"] == 40
    assert result["work_cutoff_seconds"] == 260
    assert result["remaining_processes"] == 0
    assert result["worker_cleanup_returncode"] == 0
    assert result["official_reward"] is None
    assert result["benchmark_advantage_claimed"] is False
    assert result["unreceipted_provider_attempt"]["phase"] == "coding"
    assert [row["phase"] for row in result["provider_invocations"]] == ["planning"]


@pytest.mark.parametrize("boundary", ["planning", "admission"])
def test_model_only_never_bypasses_failed_planning_or_admission(supervisor, monkeypatch, boundary):
    state, _, signed = supervisor
    monkeypatch.setattr(driver, "prepare_terminal_doctor_dispatch",
                        lambda **_: pytest.fail("Doctor selection must be skipped"))
    monkeypatch.setattr(driver, "open_existing_native_owner",
                        lambda **_: pytest.fail("invalid admission reached owner"))
    if boundary == "planning":
        monkeypatch.setattr(driver.preparation, "plan", lambda **_: {"qualified": False})
        monkeypatch.setattr(driver.preparation, "context", lambda **_: pytest.fail("unqualified plan built context"))
    else:
        def reject(*_args, **_kwargs):
            raise ValueError("authored admission drift")

        monkeypatch.setattr(driver, "verify_local_benchmark_admission", reject)
    result = driver.run(instruction=Path("/fixture/instruction"), state=state, arm="full",
                        doctor_dispatch_policy="model-only")
    assert result["task_completed"] is False
    assert "error" in result
    assert signed == []
    assert "implementation_route" not in result
    assert "doctor_dispatch" not in result


@pytest.mark.parametrize("policy,arguments", [("auto", []), ("model-only", ["--doctor-dispatch-policy", "model-only"])])
def test_driver_cli_forwards_policy_without_selecting_a_route(monkeypatch, policy, arguments):
    calls = []
    monkeypatch.setattr(sys, "argv", ["supervisor", "--instruction", "/fixture/input", "--state", "/fixture/state",
                                     "--arm", "full", *arguments])
    monkeypatch.setattr(driver, "run", lambda **kwargs: calls.append(kwargs) or {
        "task_completed": True, "arm": "full", "seconds": 0, "provider_invocations": [],
    })
    assert driver.main() == 0
    assert calls[0]["doctor_dispatch_policy"] == policy


@pytest.mark.parametrize("arm,policy", [("full", "auto"), ("full", "model-only"), ("no-index", "auto")])
def test_adapter_forwards_policy_with_existing_requirement_transport_and_budget(tmp_path, monkeypatch, arm, policy):
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "manifest.json").write_text(json.dumps({"archive_sha256": "authored", "learned_requirements": []}))
    clock = Clock()
    costs = {"instruction_upload": 7, "requirement_upload": 11, "public_input_capture": 3,
             "driver": 150, "report_export": 4, "admission_export": 3, "public_output_export": 2}
    environment = Environment(clock, costs=costs)
    _install(monkeypatch, clock, environment)
    options = {**OPTIONS, **SEMANTIC_OPTIONS} if arm == "full" else OPTIONS
    agent = adapter.FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name="gpt-5.6-sol",
        runtime_archive=str(archive), arm=arm, doctor_dispatch_policy=policy,
        intent_requirement_contract={"schema": "authored transport fixture"}, agent_timeout_seconds=300, **options)
    context = AgentContext()
    asyncio.run(agent.run("Literal public instruction", environment, context))
    assert environment.events == list(costs)
    assert len(environment.commands) == 1
    command = environment.commands[0]
    argv = command["argv"]
    if policy == "auto":
        assert "--doctor-dispatch-policy" not in argv
    else:
        assert argv[argv.index("--doctor-dispatch-policy") + 1] == policy
    assert argv[argv.index("--intent-requirement-contract") + 1] == adapter.ROOT + "/intent-requirements.json"
    assert argv[argv.index("--timeout-seconds") + 1] == "249"
    assert command["timeout_sec"] == 259
    assert context.metadata["doctor_dispatch_policy"] == policy
    assert context.metadata["adapter_deadline"]["budget_seconds"] == 300
    assert context.metadata["signed_admission_exported"] is True
    assert (agent.logs_dir / "instruction.md").read_text() == "Literal public instruction"


def test_host_model_only_config_adds_only_dispatch_policy_to_existing_matched_profile(tmp_path):
    default = benchmark.config_for(tmp_path, tmp_path / "out", tmp_path / "archive", "full")
    selected = benchmark.config_for(tmp_path, tmp_path / "out", tmp_path / "archive", "full",
                                    doctor_dispatch_policy="model-only")
    wire = JobConfig.model_validate(selected, extra="forbid").model_dump(mode="json")
    assert wire["agents"][0]["kwargs"] == {**default["agents"][0]["kwargs"], "doctor_dispatch_policy": "model-only"}
    assert "doctor_dispatch_policy" not in default["agents"][0]["kwargs"]
    assert wire["agents"][0]["override_timeout_sec"] == 600
    assert wire["agents"][0]["override_setup_timeout_sec"] == 1800
    for key in ("tasks", "environment", "verifier", "n_attempts", "n_concurrent_trials", "timeout_multiplier", "retry"):
        assert selected[key] == default[key]


@pytest.mark.parametrize("policy", ["auto", "model-only"])
def test_host_preparation_pins_requested_policy_without_claiming_selected_route(tmp_path, monkeypatch, policy):
    archive, _ = _timeout_runtime_archive(tmp_path)
    dataset = _authored_prepare_dataset(tmp_path)
    output = tmp_path / "prepared"
    calls = []

    def dry(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="Authored dry-run double.\n", stderr="")

    monkeypatch.setattr(benchmark.subprocess, "run", dry)
    result = benchmark.prepare(dataset=dataset, output=output, archive=archive, arm="full",
                               doctor_dispatch_policy=policy)
    assert len(calls) == 1 and calls[0][-1] == "--dry-run"
    assert result["doctor_dispatch_policy"] == policy
    assert result["provider_calls"] == 0
    assert "implementation_route" not in result
    prepared = json.loads((output / "preparation.json").read_text())
    config = json.loads((output / "config.json").read_text())
    assert prepared["doctor_dispatch_policy"] == policy
    assert benchmark._doctor_policy_observation(prepared, config["agents"][0]["kwargs"]) == {
        "doctor_dispatch_policy": policy, "doctor_dispatch_policy_valid": True,
        "doctor_dispatch_policy_config_unchanged": True,
    }


def _policy_trial(tmp_path, *, policy="auto", pin=True):
    output, prepared, record, trial = _retained_trial(tmp_path, options={})
    if pin:
        prepared["doctor_dispatch_policy"] = policy
    config = benchmark.config_for(Path(prepared["dataset"]), output, tmp_path / "archive", "full",
                                  doctor_dispatch_policy=policy)
    native._json(output / "config.json", config)
    prepared["config_sha256"] = native._hash(output / "config.json")
    native._json(output / "preparation.json", prepared)
    record["config"]["agent"] = config["agents"][0]
    native._json(trial / "result.json", record)
    return output, prepared, record, trial


@pytest.mark.parametrize("policy", ["auto", "model-only"])
@pytest.mark.parametrize("pin", [True, False])
def test_collector_distinguishes_policy_configuration_from_legacy_pin_currentness(tmp_path, policy, pin):
    output, _, _, _ = _policy_trial(tmp_path, policy=policy, pin=pin)
    original = (output / "preparation.json").read_bytes()
    result = benchmark.collect(output)
    for observed in (result, result["trials"][0]):
        assert observed["doctor_dispatch_policy"] == policy
        assert observed["doctor_dispatch_policy_valid"] is True
        assert observed["doctor_dispatch_policy_config_unchanged"] is (True if pin else None)
        assert "implementation_route" not in observed
    assert (output / "preparation.json").read_bytes() == original


@pytest.mark.parametrize("missing", ["config", "agent", "kwargs"])
def test_absent_actual_trial_config_cannot_confirm_pinned_auto_policy(tmp_path, missing):
    output, _, record, trial = _policy_trial(tmp_path)
    if missing == "config":
        record.pop("config")
    elif missing == "agent":
        record["config"].pop("agent")
    else:
        record["config"]["agent"].pop("kwargs")
    native._json(trial / "result.json", record)
    result = benchmark.collect(output)
    assert result["doctor_dispatch_policy_config_unchanged"] is True
    observed = result["trials"][0]
    assert observed["doctor_dispatch_policy"] is None
    assert observed["doctor_dispatch_policy_valid"] is None
    assert observed["doctor_dispatch_policy_config_unchanged"] is None


@pytest.mark.parametrize("malformed", [[], "", 0, {"doctor_dispatch_policy": True}, {"doctor_dispatch_policy": "unknown"}])
def test_malformed_actual_trial_policy_never_becomes_valid_auto(tmp_path, malformed):
    output, _, record, trial = _policy_trial(tmp_path)
    record["config"]["agent"]["kwargs"] = malformed
    native._json(trial / "result.json", record)
    observed = benchmark.collect(output)["trials"][0]
    assert observed["doctor_dispatch_policy"] is None
    assert observed["doctor_dispatch_policy_valid"] is False
    assert observed["doctor_dispatch_policy_config_unchanged"] is False


def test_collector_reports_actual_trial_policy_mismatch_separately_from_host_configuration(tmp_path):
    output, _, record, trial = _policy_trial(tmp_path)
    record["config"]["agent"]["kwargs"]["doctor_dispatch_policy"] = "model-only"
    native._json(trial / "result.json", record)
    result = benchmark.collect(output)
    assert result["doctor_dispatch_policy"] == "auto"
    assert result["doctor_dispatch_policy_config_unchanged"] is True
    assert result["trials"][0]["doctor_dispatch_policy"] == "model-only"
    assert result["trials"][0]["doctor_dispatch_policy_valid"] is True
    assert result["trials"][0]["doctor_dispatch_policy_config_unchanged"] is False


@pytest.mark.parametrize("change", ["mismatched_pin", "invalid_pin", "missing_config_policy"])
def test_executor_refuses_changed_policy_before_runtime_asset_reads_or_launch(tmp_path, monkeypatch, change):
    output, prepared, _, _ = _policy_trial(tmp_path, policy="model-only")
    if change == "mismatched_pin":
        prepared["doctor_dispatch_policy"] = "auto"
    elif change == "invalid_pin":
        prepared["doctor_dispatch_policy"] = None
    else:
        config = json.loads((output / "config.json").read_text())
        config["agents"][0]["kwargs"].pop("doctor_dispatch_policy")
        native._json(output / "config.json", config)
        prepared["config_sha256"] = native._hash(output / "config.json")
    native._json(output / "preparation.json", prepared)
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("Harbor launched"))
    with pytest.raises(ValueError, match="Doctor policy"):
        benchmark.execute(output)
    assert not (output / "invocation.json").exists()


@pytest.mark.parametrize("policy,arguments", [("auto", []), ("model-only", ["--doctor-dispatch-policy", "model-only"])])
def test_prepare_cli_forwards_policy_to_preparation(tmp_path, monkeypatch, capsys, policy, arguments):
    observed = {}
    monkeypatch.setattr(benchmark, "prepare", lambda **kwargs: observed.update(kwargs) or {"prepared": True})
    monkeypatch.setattr(sys, "argv", ["benchmark", "prepare", "--dataset", str(tmp_path / "dataset"),
                                     "--output", str(tmp_path / "output"), "--archive", str(tmp_path / "archive"),
                                     *arguments])
    benchmark.main()
    assert observed["doctor_dispatch_policy"] == policy
    assert json.loads(capsys.readouterr().out) == {"prepared": True}


@pytest.mark.parametrize("operation", ["execute", "collect"])
def test_existing_preparation_operations_cannot_override_retained_dispatch_policy(tmp_path, monkeypatch, capsys, operation):
    calls = []
    monkeypatch.setattr(benchmark, operation, lambda output: calls.append(output) or {"operation": operation})
    monkeypatch.setattr(sys, "argv", ["benchmark", operation, "--output", str(tmp_path / "output"),
                                     "--doctor-dispatch-policy", "model-only"])
    benchmark.main()
    assert calls == [tmp_path / "output"]
    assert json.loads(capsys.readouterr().out) == {"operation": operation}
