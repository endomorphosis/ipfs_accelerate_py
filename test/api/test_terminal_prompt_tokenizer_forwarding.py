"""Forward explicit child-local tokenizer inputs without provider/deployment I/O.

These tests exercise the actual command and supervisor/Harbor adapter boundaries
with authored process/owner fixtures. Asset staging, runtime availability,
provider tokenizer compatibility, and task qualification remain separate.
"""
from __future__ import annotations

import asyncio
import json
import shlex
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from benchmarks.agent_supervisor.container_coding import (
    full_supervisor_harbor_agent as adapter,
)
from benchmarks.agent_supervisor.container_coding import (
    terminal_container_supervisor as driver,
)
from benchmarks.agent_supervisor.container_coding import (
    terminal_doctor_dispatch as dispatch,
)
from harbor.models.agent.context import AgentContext

ARTIFACT = Path("/opt/ipfs-supervisor/models/explicit tokenizer/descriptor.json")
DIGEST = "a" * 64
TOKEN_ARGUMENTS = ["--prompt-tokenizer-artifact", str(ARTIFACT),
                   "--prompt-tokenizer-sha256", DIGEST]
OPTIONS = {"prompt_tokenizer_artifact": ARTIFACT, "prompt_tokenizer_sha256": DIGEST,
           "max_input_tokens": 4096}
V2 = "closed-typed-identifiers@2"
SEMANTIC_OPTIONS = {"semantic_slot_policy_revision": V2, "semantic_context_projection": True}
SEMANTIC_ARGUMENTS = ["--semantic-slot-policy-revision", V2, "--semantic-context-projection"]


def command(**changes):
    return dispatch.implementation_argv(**{
        "router": Path("/router"), "model": "gpt-5.6-sol", "reasoning": "high",
        "timeout": 30, "semantic_repository": None, "doctor": None, **changes,
    })


def test_default_command_remains_exact_without_tokenizer_flags():
    assert command() == ["/router", "--model", "gpt-5.6-sol", "--reasoning-effort", "high",
                         "--timeout", "30", "--max-output-tokens", "4096"]


@pytest.mark.parametrize("budget", [None, 4096])
def test_explicit_pair_and_budget_survive_semantic_and_residual_command_quoting(budget):
    residual = {"route": "model_router", "status": "residual", "residual_context": {
        "artifact": "/owner/residual context.json", "sha256": "b" * 64, "task_cid": "task"}}
    argv = command(semantic_repository=Path("/owner/canonical repository"), doctor=residual,
                   **{**OPTIONS, "max_input_tokens": budget})
    assert shlex.split(shlex.join(argv)) == argv
    expected = TOKEN_ARGUMENTS + (["--max-input-tokens", str(budget)] if budget is not None else [])
    assert argv[-len(expected):] == expected
    assert argv[argv.index("--doctor-residual-artifact") + 1] == residual["residual_context"]["artifact"]
    assert argv[argv.index("--semantic-repository") + 1] == "/owner/canonical repository"


INVALID_OPTIONS = [
    {"prompt_tokenizer_artifact": ARTIFACT},
    {"prompt_tokenizer_sha256": DIGEST},
    {"prompt_tokenizer_artifact": ARTIFACT, "prompt_tokenizer_sha256": ""},
    {"prompt_tokenizer_artifact": ARTIFACT, "prompt_tokenizer_sha256": "not-a-sha"},
    {"prompt_tokenizer_artifact": "relative/descriptor.json", "prompt_tokenizer_sha256": DIGEST},
    {"prompt_tokenizer_artifact": "/owner/../descriptor.json", "prompt_tokenizer_sha256": DIGEST},
    {"prompt_tokenizer_artifact": "https://foreign/descriptor.json", "prompt_tokenizer_sha256": DIGEST},
    {**OPTIONS, "max_input_tokens": True},
    {**OPTIONS, "max_input_tokens": 0},
    {**OPTIONS, "max_input_tokens": -1},
]


def test_budget_without_descriptor_reaches_automatic_child_counter():
    assert command(max_input_tokens=4096)[-2:] == ["--max-input-tokens", "4096"]


@pytest.mark.parametrize("options", INVALID_OPTIONS)
def test_invalid_binding_refuses_before_supervisor_uid_state_or_provider_access(monkeypatch, options):
    def forbidden():
        pytest.fail("invalid binding must refuse before supervisor identity/state checks")

    monkeypatch.setattr(driver.os, "geteuid", forbidden)
    with pytest.raises(ValueError):
        command(**options)
    with pytest.raises(ValueError):
        driver.run(instruction=Path("/fixture/instruction"),
                   state=Path("/fixture/state"), arm="no-index", **options)


@pytest.mark.parametrize("route,flag", [("doctor_candidate", "--doctor-candidate-artifact"),
                                      ("doctor_contract_candidate", "--doctor-contract-artifact")])
def test_provider_free_route_retains_default_and_refuses_direct_token_options(route, flag):
    doctor = {"route": route, "status": "candidate_ready", "provider_calls": 0,
              "artifact": "/owner/candidate.json", "sha256": "b" * 64, "task_cid": "task"}
    assert flag in command(doctor=doctor)
    assert "--model" not in command(doctor=doctor)
    with pytest.raises(ValueError, match="require router invocation"):
        command(doctor=doctor, **OPTIONS)


@pytest.mark.parametrize("projection", [False, True])
def test_coding_command_appends_opt_in_comparison_before_signed_launch_without_tokenizer_discovery(projection):
    options = {"semantic_slot_policy_revision": V2, "semantic_context_projection": projection}
    baseline = command(semantic_repository=Path("/owner/repository"))
    advanced = command(semantic_repository=Path("/owner/repository"), **options)
    expected = SEMANTIC_ARGUMENTS if projection else SEMANTIC_ARGUMENTS[:2]
    assert advanced == baseline + expected
    assert "--prompt-tokenizer-artifact" not in advanced
    assert shlex.split(shlex.join(advanced)) == advanced
    assert command(semantic_slot_policy_revision="legacy@1", semantic_context_projection=False) == command()


@pytest.mark.parametrize("doctor", [None, {"route": "doctor_candidate"}, {"route": "doctor_contract_candidate"}])
def test_direct_compact_command_refuses_missing_repository_or_provider_free_route(doctor):
    with pytest.raises(ValueError, match="require bound model coding"):
        command(doctor=doctor, **SEMANTIC_OPTIONS)
    if doctor is not None:
        with pytest.raises(ValueError, match="require bound model coding"):
            command(doctor=doctor, semantic_repository=Path("/owner/repository"), **SEMANTIC_OPTIONS)


@pytest.mark.parametrize("options", [
    {"semantic_slot_policy_revision": "future@99"},
    {"semantic_context_projection": True},
    {"semantic_context_projection": "yes"},
    {"semantic_slot_policy_revision": V2, "semantic_context_projection": 1},
])
def test_invalid_semantic_options_refuse_before_owner_checks(monkeypatch, options):
    def forbidden():
        pytest.fail("invalid options must refuse before supervisor identity checks")

    monkeypatch.setattr(driver.os, "geteuid", forbidden)
    with pytest.raises(ValueError):
        command(semantic_repository=Path("/owner/repository"), **options)
    with pytest.raises(ValueError):
        driver.run(instruction=Path("/fixture/instruction"), state=Path("/fixture/state"),
                   arm="full", **options)


def test_no_index_arm_refuses_advanced_semantic_options_before_owner_checks(monkeypatch):
    monkeypatch.setattr(driver.os, "geteuid", lambda: pytest.fail("no-index must refuse before owner checks"))
    with pytest.raises(ValueError, match="full indexed arm"):
        driver.run(instruction=Path("/fixture/instruction"), state=Path("/fixture/state"),
                   arm="no-index", **SEMANTIC_OPTIONS)


@pytest.fixture
def supervisor(tmp_path, monkeypatch):
    root = tmp_path / "deployed"
    state = root / "state/trial"
    planned, signed = [], []
    task = SimpleNamespace(task_cid="task", task_key="TASK", status="completed", revision=1)
    monkeypatch.setattr(driver, "ROOT", root)
    monkeypatch.setattr(driver.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(driver.os, "umask", lambda *_: None)
    monkeypatch.setattr(driver.signal, "signal", lambda *_: None)
    monkeypatch.setattr(driver.signal, "setitimer", lambda *_: None)
    monkeypatch.setattr(driver.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0))

    def prepare(**_):
        state.mkdir(parents=True)
        (state / "admission.json").write_text("{}")
        return {"intent_preplanning": {}}

    monkeypatch.setattr(driver.preparation, "prepare", prepare)
    monkeypatch.setattr(driver.preparation, "initial_context", lambda **_: {})
    monkeypatch.setattr(driver.preparation, "context", lambda **_: {"context_bundle": {"fixture": True}})

    class Process:
        returncode = 0

        def __init__(self, argv, **_):
            planned.append(argv)

        def communicate(self, _prompt, **_):
            return (json.dumps({"schema": "router-implementation-invocation@1", "invocation_id": "planner",
                                "external_container_boundary": {"fixture": True},
                                "status": "provider_returned", "native_rollout_usage": None}) + "\nplan", "")

    monkeypatch.setattr(driver.subprocess, "Popen", Process)

    def plan(*, provider_callable, **_):
        provider_callable("fixture prompt", repository=root, provider="codex_cli",
                          model=driver.preparation.MODEL, reasoning_effort="high", timeout=30,
                          max_new_tokens=2048, trace_path=state / "planner.log")
        return {"qualified": True}

    monkeypatch.setattr(driver.preparation, "plan", plan)
    monkeypatch.setattr(driver, "verify_local_benchmark_admission", lambda *_args, **_kwargs: {
        "graph": SimpleNamespace(tasks=[task]), "manifest": {"repository_cid": "fixture-repository",
        "sources": {driver.preparation.INSTRUCTION: {"sha256": "b" * 64}}}})
    from ipfs_accelerate_py.agent_supervisor.runtime import router_public_instruction

    monkeypatch.setattr(router_public_instruction, "prepare_public_instruction_context", lambda **_: {
        "artifact": "/app/.runtime/literal-instruction.json", "sha256": "c" * 64})

    @contextmanager
    def owner(**_):
        yield SimpleNamespace(server=object(), source=SimpleNamespace(get_task=lambda _: task))

    monkeypatch.setattr(driver, "open_existing_native_owner", owner)
    response = SimpleNamespace(to_dict=lambda: {"status": "succeeded"})

    def runtime(_state, **kwargs):
        signed.append(shlex.split(kwargs["implementation_command"]))
        return SimpleNamespace(state=state, start=lambda: response, stop=lambda: response,
                               observe=lambda: {}, close=lambda: None, profile=object(),
                               process=SimpleNamespace(snapshot=lambda _: SimpleNamespace(members=[])))

    monkeypatch.setattr(driver.AdmittedBenchmarkRuntime, "create", runtime)
    monkeypatch.setattr(driver, "_native_diagnostics", lambda _: {})
    monkeypatch.setattr(driver, "_refresh_completed_context", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(driver, "_final_context_audit", lambda *_args, **_kwargs: None)
    return state, planned, signed


@pytest.mark.parametrize("arm,route", [("full", "model_router"), ("no-index", "model_router"),
                                      ("full", "doctor_candidate")])
def test_actual_supervisor_forwards_planning_and_signed_coding_inputs(supervisor, monkeypatch, arm, route):
    state, planned, signed = supervisor
    doctor = {"route": route, "status": "candidate_ready" if route == "doctor_candidate" else "residual",
              "provider_calls": 0, "artifact": "/owner/candidate.json", "sha256": "b" * 64, "task_cid": "task"}
    monkeypatch.setattr(driver, "prepare_terminal_doctor_dispatch", lambda **_: doctor)
    result = driver.run(instruction=Path("/fixture/instruction"), state=state, arm=arm, **OPTIONS)
    assert result["task_completed"] is True
    assert len(planned) == len(signed) == 1
    assert planned[0][-6:] == TOKEN_ARGUMENTS + ["--max-input-tokens", "4096"]
    assert planned[0][planned[0].index("--purpose") + 1] == "planning"
    if route == "model_router":
        start = signed[0].index("--prompt-tokenizer-artifact")
        assert signed[0][start:start + 6] == TOKEN_ARGUMENTS + ["--max-input-tokens", "4096"]
        assert "--public-instruction-artifact" in signed[0]
    else:
        assert "--doctor-candidate-artifact" in signed[0]
        assert "--prompt-tokenizer-artifact" not in signed[0]


@pytest.mark.parametrize("route", ["model_router", "doctor_candidate", "doctor_contract_candidate"])
def test_actual_full_supervisor_limits_compact_options_to_signed_model_coding(supervisor, monkeypatch, route):
    state, planned, signed = supervisor
    doctor = {"route": route, "status": "residual" if route == "model_router" else "candidate_ready",
              "provider_calls": 0, "artifact": "/owner/candidate.json", "sha256": "b" * 64, "task_cid": "task"}
    monkeypatch.setattr(driver, "prepare_terminal_doctor_dispatch", lambda **_: doctor)
    result = driver.run(instruction=Path("/fixture/instruction"), state=state, arm="full",
                        **OPTIONS, **SEMANTIC_OPTIONS)
    assert result["task_completed"] is True
    assert len(planned) == len(signed) == 1
    assert planned[0][-6:] == TOKEN_ARGUMENTS + ["--max-input-tokens", "4096"]
    assert "--semantic-slot-policy-revision" not in planned[0]
    assert "--semantic-context-projection" not in planned[0]
    if route == "model_router":
        start = signed[0].index("--semantic-slot-policy-revision")
        assert signed[0][start:start + 3] == SEMANTIC_ARGUMENTS
        assert "--public-instruction-artifact" in signed[0]
    else:
        assert "--semantic-slot-policy-revision" not in signed[0]
        assert "--semantic-context-projection" not in signed[0]


def test_bound_planner_model_mismatch_refuses_before_worker_creation(supervisor, monkeypatch):
    state, planned, signed = supervisor

    def plan(*, provider_callable, **_):
        provider_callable("fixture prompt", repository=state.parent, provider="codex_cli",
                          model="different-model", reasoning_effort="high", timeout=30,
                          max_new_tokens=2048, trace_path=state / "planner.log")

    monkeypatch.setattr(driver.preparation, "plan", plan)
    result = driver.run(instruction=Path("/fixture/instruction"), state=state, arm="no-index", **OPTIONS)
    assert result["task_completed"] is False
    assert result["error"]["type"] == "ValueError"
    assert "model differs" in result["error"]["message"]
    assert planned == signed == []


@pytest.mark.parametrize("options", [[], TOKEN_ARGUMENTS + ["--max-input-tokens", "4096"]])
def test_supervisor_cli_forwards_pair_without_implicit_defaults(monkeypatch, options):
    calls = []
    monkeypatch.setattr(sys, "argv", ["supervisor", "--instruction", "/fixture/input",
                                     "--state", "/fixture/state", "--arm", "no-index", *options])
    monkeypatch.setattr(driver, "run", lambda **kwargs: calls.append(kwargs) or {
        "task_completed": True, "arm": "no-index", "seconds": 0, "provider_invocations": []})
    assert driver.main() == 0
    assert len(calls) == 1
    assert calls[0]["prompt_tokenizer_artifact"] == (ARTIFACT if options else None)
    assert calls[0]["prompt_tokenizer_sha256"] == (DIGEST if options else None)
    assert calls[0]["max_input_tokens"] == (4096 if options else None)


def test_supervisor_cli_forwards_explicit_compact_options_without_implicit_tokenizer(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "argv", ["supervisor", "--instruction", "/fixture/input",
                                     "--state", "/fixture/state", "--arm", "full", *SEMANTIC_ARGUMENTS])
    monkeypatch.setattr(driver, "run", lambda **kwargs: calls.append(kwargs) or {
        "task_completed": True, "arm": "full", "seconds": 0, "provider_invocations": []})
    assert driver.main() == 0
    assert calls[0]["semantic_slot_policy_revision"] == V2
    assert calls[0]["semantic_context_projection"] is True
    assert calls[0]["prompt_tokenizer_artifact"] is None
    assert calls[0]["prompt_tokenizer_sha256"] is None


@pytest.mark.parametrize("arm", ["full", "no-index"])
@pytest.mark.parametrize("options", [{}, OPTIONS, SEMANTIC_OPTIONS])
def test_actual_harbor_adapter_forwards_existing_container_path_without_asset_upload(tmp_path, monkeypatch, arm, options):
    (tmp_path / "manifest.json").write_text(json.dumps({"archive_sha256": "fixture", "learned_requirements": []}))
    uploads, commands = [], []

    class Environment:
        async def upload_file(self, _source, target):
            uploads.append(target)

        async def exec(self, **kwargs):
            commands.append(shlex.split(kwargs["command"]))
            return SimpleNamespace(stdout="", stderr="", return_code=0)

        async def download_file(self, _source, _target):
            raise FileNotFoundError("authored fixture has no remote report")

    async def observation(*_):
        return {"status": "authored"}

    monkeypatch.setattr(adapter, "capture_public_inputs", observation)
    monkeypatch.setattr(adapter, "export_public_outputs", observation)
    if options == SEMANTIC_OPTIONS and arm == "no-index":
        with pytest.raises(ValueError, match="full indexed arm"):
            adapter.FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name="gpt-5.6-sol",
                                       runtime_archive=str(tmp_path), arm=arm, **options)
        assert uploads == commands == []
        return
    agent = adapter.FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name="gpt-5.6-sol",
                                        runtime_archive=str(tmp_path), arm=arm, **options)
    context = AgentContext()
    asyncio.run(agent.run("literal public instruction", Environment(), context))
    assert len(commands) == 1
    if options == OPTIONS:
        start = commands[0].index("--prompt-tokenizer-artifact")
        assert commands[0][start:start + 6] == TOKEN_ARGUMENTS + ["--max-input-tokens", "4096"]
    else:
        assert "--prompt-tokenizer-artifact" not in commands[0]
        assert "--max-input-tokens" not in commands[0]
    if options == SEMANTIC_OPTIONS:
        start = commands[0].index("--semantic-slot-policy-revision")
        assert commands[0][start:start + 3] == SEMANTIC_ARGUMENTS
    else:
        assert "--semantic-slot-policy-revision" not in commands[0]
        assert "--semantic-context-projection" not in commands[0]
    assert uploads == [adapter.ROOT + "/instruction.md"]
    assert context.metadata["task_completed"] is False


@pytest.mark.parametrize("options", INVALID_OPTIONS[:8])
def test_harbor_constructor_rejects_bad_inputs_before_archive_discovery(tmp_path, options):
    with pytest.raises(ValueError):
        adapter.FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name="gpt-5.6-sol",
                                    runtime_archive=str(tmp_path / "unavailable"), arm="no-index", **options)
