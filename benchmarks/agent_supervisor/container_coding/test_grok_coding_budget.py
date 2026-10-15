"""Authored turn-bound qualification; no provider or benchmark task execution."""
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shlex
import sys
import tarfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import grok_coding_budget as budget
from benchmarks.agent_supervisor.container_coding import benchmark_controls as controls
from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
from benchmarks.agent_supervisor.container_coding import terminal_doctor_dispatch as doctor
from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import GROK_PROFILE, resolve_provider_profile
from benchmarks.agent_supervisor.container_coding.test_source_formalization_worker_deployment import _run as worker_run
from benchmarks.agent_supervisor.container_coding.test_terminal_semantic_metadata_policy import authored_task
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _supervisor_manifest


def inventory():
    root = Path(budget.__file__).resolve().parents[3]
    return {"files": [{"path": "source/" + path, "bytes": (root / path).stat().st_size,
        "sha256": hashlib.sha256((root / path).read_bytes()).hexdigest()} for path in budget.SOURCE_PATHS]}


def manifest():
    from benchmarks.agent_supervisor.container_coding import terminal_grok_deployment as grok
    value = _supervisor_manifest({**inventory(), "archive_sha256": "a" * 64, "learned_requirements": []})
    value["grok_cli_assets"] = grok.grok_binding()
    value["files"].append({"path": grok.GROK_PATH, "bytes": grok.GROK_BYTES,
        "sha256": grok.GROK_SHA256, "mode": 0o755})
    return value


def seal(config):
    selected = resolve_provider_profile(config["agents"][0]["kwargs"].get("provider_profile"))
    return controls.build_controls(config, task_input_sha256={"instruction.md": "a" * 64},
        task=benchmark.TASK, **{key: selected[key] for key in ("model", "reasoning_effort", "cli_version")})


def frozen_trial(tmp_path, turns=16):
    dataset, task = authored_task(tmp_path)
    output = tmp_path / "out"; output.mkdir()
    config = benchmark.config_for(dataset, output, tmp_path / "archive", "full",
        provider_profile=GROK_PROFILE, grok_coding_max_turns=turns)
    (output / "config.json").write_text(json.dumps(config))
    selected = resolve_provider_profile(GROK_PROFILE)
    prepared = {"arm": "full", "dataset": str(dataset), "prepared": True,
        "config_sha256": benchmark._hash(output / "config.json"), "task_input_sha256": benchmark._task_hashes(task),
        "provider_profile": GROK_PROFILE, **budget.grok_coding_selection(turns),
        **{key: selected[key] for key in ("model", "reasoning_effort", "cli_version")}}
    (output / "preparation.json").write_text(json.dumps(prepared))
    return output, prepared, config


def provider_free_report(turns=16, route="doctor_candidate"):
    return {"grok_coding_max_turns": turns, "implementation_route": route,
        "provider_invocations": [], "coding_control_application": {
            "schema": "terminal-coding-control-application@1", "status": "not_applicable_provider_free",
            "implementation_route": route, "requested": {"grok_coding_max_turns": turns},
            "model_coding_dispatched": False, "token_savings_established": False, "completion_authority": False}}


@pytest.mark.parametrize("value", [None, True, False, 0, -1, 129, 1.0, "16", [], {}])
def test_exact_integer_bounds(value):
    with pytest.raises(ValueError, match="integer"):
        budget.validate_grok_coding_max_turns(value, provider="grok_cli")


@pytest.mark.parametrize("arm", ["full", "no-index"])
@pytest.mark.parametrize("provider", [None, GROK_PROFILE])
def test_default_preserves_configuration_and_declaration(tmp_path, arm, provider):
    args = (tmp_path, tmp_path / "out", tmp_path / "archive", arm)
    original = benchmark.config_for(*args, provider_profile=provider)
    explicit = benchmark.config_for(*args, provider_profile=provider, grok_coding_max_turns=128)
    assert original == explicit
    assert "grok_coding_max_turns" not in explicit["agents"][0]["kwargs"]
    assert seal(original) == seal(explicit)
    assert seal(explicit)["schema"] == controls.SCHEMA
    budget.require_grok_coding_archive({}, 128)


@pytest.mark.parametrize("combined", [False, True])
def test_sealed_treatment_compares_independently(tmp_path, combined):
    options = dict(provider_profile=GROK_PROFILE)
    if combined:
        options.update(semantic_working_set="capsule-references@1", source_view_profile="identity@1",
            capsule_budget="one-capsule@1", source_edit_mode="mapped-span@2",
            planning_mode="declared-contract@1", numerical_formalization_profile="python-numerical-control-structure@3")
    args = (tmp_path, tmp_path / "out", tmp_path / "archive", "full")
    original = seal(benchmark.config_for(*args, **options))
    selected = seal(benchmark.config_for(*args, **options, grok_coding_max_turns=16))
    assert selected["schema"] == controls.GROK_CODING_SCHEMA
    assert controls.validate_controls(selected)
    assert set(selected["experimental_controls_sha256"]) == {
        *original.get("experimental_controls_sha256", {}), "grok_coding_max_turns"}
    observe = lambda value: dict(status="observed", configuration_unchanged=True, declared=value)
    assert controls.compare_controls(observe(original), observe(selected))["differences"] == [
        "experimental_controls_sha256.grok_coding_max_turns"]
    assert controls.compare_controls(observe(selected), observe(selected))["matches"] is True
    edited = deepcopy(selected); edited["experimental_controls_sha256"].pop("grok_coding_max_turns")
    assert not controls.validate_controls(edited)


@pytest.mark.parametrize("mutation", ["missing", "digest", "size", "duplicate"])
def test_archive_binds_every_producer(mutation):
    manifest = inventory()
    budget.require_grok_coding_archive(manifest, 16)
    if mutation == "missing": manifest["files"].pop()
    elif mutation == "digest": manifest["files"][0]["sha256"] = "a" * 64
    elif mutation == "size": manifest["files"][0]["bytes"] = True
    else: manifest["files"].append(manifest["files"][0])
    with pytest.raises(ValueError, match="newly qualified"):
        budget.require_grok_coding_archive(manifest, 16)


def test_codex_and_census_refuse_before_file_access(tmp_path):
    from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
    from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import FullSupervisorAgent
    with pytest.raises(ValueError, match="Grok coding"):
        benchmark.prepare(dataset=tmp_path / "absent", output=tmp_path / "out",
            archive=tmp_path / "absent", arm="full", grok_coding_max_turns=16)
    with pytest.raises(ValueError, match="metadata census"):
        driver.run(instruction=tmp_path / "absent", state=tmp_path / "state", arm="full",
            provider_profile=GROK_PROFILE, grok_coding_max_turns=16, metadata_census_only=True)
    with pytest.raises(ValueError, match="metadata census"):
        FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name="grok-4.7", runtime_archive="absent",
            provider_profile=GROK_PROFILE, grok_coding_max_turns=16, metadata_census_only=True)


@pytest.mark.parametrize("turns", [1, 16, 128])
def test_doctor_and_installed_worker_preserve_coding_bound(tmp_path, turns):
    argv = doctor.implementation_argv(router=Path("/router"), model="grok-4.7", reasoning="high",
        timeout=30, semantic_repository=None, doctor=None, provider="grok_cli", grok_coding_max_turns=turns)
    args = argv[1:] + ["--provider", "grok_cli"]
    result = worker_run(tmp_path, args)
    assert result.returncode == 0, result.stderr.decode()
    forwarded = json.loads(result.stdout)["argv"]
    if turns == 128:
        assert "--grok-coding-max-turns" not in argv + forwarded
    else:
        for command in (argv, forwarded):
            assert command.count("--grok-coding-max-turns") == 1
            assert command[command.index("--grok-coding-max-turns") + 1] == str(turns)


@pytest.mark.parametrize("extra", [["--purpose", "planning"], ["--provider", "codex_cli"],
    ["--preflight"], ["--semantic-metadata-census-only"], ["--grok-coding-max-turns", "0"]])
def test_worker_refuses_unsupported_selection_before_router(tmp_path, extra):
    result = worker_run(tmp_path, ["--provider", "grok_cli", "--model", "grok-4.7",
        "--grok-coding-max-turns", "16", *extra])
    assert result.returncode != 0 and not result.stdout


@pytest.mark.parametrize("where", ["preparation", "config"])
@pytest.mark.parametrize("mutation", ["remove", "change"])
def test_collect_and_execute_detect_treatment_drift(tmp_path, monkeypatch, where, mutation):
    output, prepared, config = frozen_trial(tmp_path)
    target = prepared if where == "preparation" else config["agents"][0]["kwargs"]
    if mutation == "remove": target.pop("grok_coding_max_turns")
    else: target["grok_coding_max_turns"] = 8
    (output / "config.json").write_text(json.dumps(config))
    prepared["config_sha256"] = benchmark._hash(output / "config.json")
    (output / "preparation.json").write_text(json.dumps(prepared))
    receipt = benchmark.collect(output)
    assert receipt["grok_coding_config_unchanged"] is False
    assert receipt["complete_single_trial_receipt"] is False
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *_a, **_k: pytest.fail("changed selection executed"))
    with pytest.raises(ValueError, match="prepared Grok coding turn bound changed"):
        benchmark.execute(output)


@pytest.mark.parametrize("policy,expected", [(None, False), (True, False), (128, False), (16, True)])
def test_collect_checks_actual_policy_without_losing_usage(tmp_path, policy, expected):
    output, prepared, config = frozen_trial(tmp_path)
    job = Path(config["jobs_dir"]) / config["job_name"]
    trial = job / "authored"; (trial / "agent").mkdir(parents=True)
    (trial / "result.json").write_text(json.dumps({"task_name": benchmark.TASK,
        "config": {"task": {"path": str(Path(prepared["dataset"]) / benchmark.TASK)}}}))
    (job / "result.json").write_text("{}")
    report = {"grok_coding_max_turns": 16, "provider_invocations": [{"phase": "coding", "purpose": "coding",
        "provider": "grok_cli", "provider_invocation_policy": {"max_turns": policy},
        "native_rollout_usage": {"usage": {"total_tokens": 33}}}]}
    (trial / "agent/supervisor-result.json").write_text(json.dumps(report))
    receipt = benchmark.collect(output)
    assert receipt["grok_coding_runtime_declared_matches"] is True
    assert receipt["grok_coding_runtime_policy_matches"] is expected
    assert receipt["complete_single_trial_receipt"] is expected
    assert receipt["trials"][0]["supervisor"] == report


def test_missing_coding_receipts_remain_unknown():
    assert budget.grok_coding_runtime_matches({}, 16) is None
    assert budget.grok_coding_runtime_matches({"provider_invocations": []}, 16) is None


@pytest.mark.parametrize("route", ["doctor_candidate", "doctor_contract_candidate"])
def test_explicit_provider_free_treatment_keeps_complete_receipt_without_policy_match(tmp_path, route):
    output, prepared, config = frozen_trial(tmp_path)
    job = Path(config["jobs_dir"]) / config["job_name"]
    trial = job / "authored"; (trial / "agent").mkdir(parents=True)
    (trial / "result.json").write_text(json.dumps({"task_name": benchmark.TASK,
        "config": {"task": {"path": str(Path(prepared["dataset"]) / benchmark.TASK)}}}))
    (job / "result.json").write_text("{}")
    report = provider_free_report(route=route)
    report["provider_invocations"] = [{"phase": "planning", "purpose": "planning", "provider": "grok_cli",
        "native_rollout_usage": {"usage": {"total_tokens": 33}}}]
    path = trial / "agent/supervisor-result.json"
    path.write_text(json.dumps(report))
    receipt = benchmark.collect(output)
    assert receipt["complete_single_trial_receipt"] is True
    assert receipt["grok_coding_runtime_policy_matches"] is None
    assert receipt["grok_coding_not_applicable_provider_free"] is True
    assert receipt["trials"][0]["supervisor"] == report
    report.pop("coding_control_application")
    path.write_text(json.dumps(report))
    absent = benchmark.collect(output)
    assert absent["complete_single_trial_receipt"] is False
    assert absent["grok_coding_runtime_policy_matches"] is None
    assert absent["grok_coding_not_applicable_provider_free"] is False


def test_mixed_provider_and_symbolic_trials_preserve_single_trial_completeness_limit(tmp_path):
    output, prepared, config = frozen_trial(tmp_path)
    job = Path(config["jobs_dir"]) / config["job_name"]
    reports = [provider_free_report(), {"grok_coding_max_turns": 16,
        "provider_invocations": [{"purpose": "coding", "phase": "coding", "provider": "grok_cli",
            "provider_invocation_policy": {"max_turns": 16}}]}]
    for index, report in enumerate(reports):
        trial = job / f"authored-{index}"; (trial / "agent").mkdir(parents=True)
        (trial / "result.json").write_text(json.dumps({"task_name": benchmark.TASK,
            "config": {"task": {"path": str(Path(prepared["dataset"]) / benchmark.TASK)}}}))
        (trial / "agent/supervisor-result.json").write_text(json.dumps(report))
    (job / "result.json").write_text("{}")
    receipt = benchmark.collect(output)
    assert receipt["trial_count"] == 2
    assert receipt["complete_single_trial_receipt"] is False
    assert receipt["grok_coding_runtime_declared_matches"] is True
    assert receipt["grok_coding_runtime_policy_matches"] is None
    assert receipt["grok_coding_not_applicable_provider_free"] is False
    assert [row["supervisor"] for row in receipt["trials"]] == reports


@pytest.mark.parametrize("mutation", ["missing_rows", "unreceipted", "unknown_row", "coding_row",
    "conflicting_row", "missing_application", "application_extra", "route", "requested", "bool_requested",
    "dispatched", "savings", "authority", "missing_declaration"])
def test_provider_free_nonapplication_requires_exact_positive_evidence(mutation):
    report = provider_free_report()
    assert budget.grok_coding_not_applicable_provider_free(report, 16) is True
    application = report["coding_control_application"]
    if mutation == "missing_rows": report.pop("provider_invocations")
    elif mutation == "unreceipted": report["unreceipted_provider_attempt"] = None
    elif mutation == "unknown_row": report["provider_invocations"] = [{}]
    elif mutation == "coding_row": report["provider_invocations"] = [{"purpose": "coding"}]
    elif mutation == "conflicting_row": report["provider_invocations"] = [{"phase": "coding", "purpose": "planning"}]
    elif mutation == "missing_application": report.pop("coding_control_application")
    elif mutation == "application_extra": application["authority"] = True
    elif mutation == "route": application["implementation_route"] = "doctor_contract_candidate"
    elif mutation == "requested": application["requested"]["grok_coding_max_turns"] = 8
    elif mutation == "bool_requested": application["requested"]["grok_coding_max_turns"] = True
    elif mutation == "dispatched": application["model_coding_dispatched"] = True
    elif mutation == "savings": application["token_savings_established"] = True
    elif mutation == "authority": application["completion_authority"] = True
    else: report.pop("grok_coding_max_turns")
    assert budget.grok_coding_not_applicable_provider_free(report, 16) is False


@pytest.mark.parametrize("other", [{}, "bad", {"phase": "planning"},
    {"phase": "planning", "purpose": "coding"}, {"phase": "coding", "purpose": "planning"},
    {"purpose": "coding", "provider": "grok_cli"}])
def test_malformed_or_unclassified_invocations_cannot_disappear(other):
    valid = {"phase": "coding", "purpose": "coding", "provider": "grok_cli",
        "provider_invocation_policy": {"max_turns": 16}}
    assert budget.grok_coding_runtime_matches({"provider_invocations": [valid, other]}, 16) is False
    assert budget.grok_coding_runtime_matches({"provider_invocations": [valid,
        {"phase": "planning", "purpose": "planning"}]}, 16) is True


@pytest.mark.parametrize("route", ["doctor_candidate", "doctor_contract_candidate"])
def test_symbolic_candidate_keeps_turn_treatment_unapplied(route):
    argv = doctor.implementation_argv(router=Path("/router"), model="grok-4.7", reasoning="high",
        timeout=30, semantic_repository=None, provider="grok_cli", grok_coding_max_turns=16,
        doctor={"route": route, "status": "candidate_ready", "provider_calls": 0,
            "artifact": "/fixture", "sha256": "a" * 64, "task_cid": "task"})
    assert "--grok-coding-max-turns" not in argv
    assert budget.grok_coding_runtime_matches({"provider_invocations": []}, 16) is None


@pytest.mark.parametrize("mutation", ["removed", "changed", "introduced"])
def test_harbor_frozen_value_cannot_change(tmp_path, mutation):
    from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import FullSupervisorAgent
    (tmp_path / "manifest.json").write_text(json.dumps(manifest()))
    agent = FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name="grok-4.7", runtime_archive=str(tmp_path),
        provider_profile=GROK_PROFILE, grok_coding_max_turns=128 if mutation == "introduced" else 16)
    if mutation == "removed": del agent.grok_coding_max_turns
    else: agent.grok_coding_max_turns = 16 if mutation == "introduced" else 8
    with pytest.raises(ValueError, match="prepared Grok"):
        agent._checked_grok_coding_max_turns()


@pytest.mark.parametrize("turns", [16, 128])
def test_dry_preparation_freezes_optional_bound(tmp_path, monkeypatch, turns):
    from benchmarks.agent_supervisor.container_coding import terminal_grok_deployment as grok
    dataset, _ = authored_task(tmp_path)
    archive = tmp_path / "archive"; archive.mkdir()
    with tarfile.open(archive / "runtime.tar.gz", "w:gz"): pass
    value = manifest(); value["archive_sha256"] = benchmark._hash(archive / "runtime.tar.gz")
    (archive / "manifest.json").write_text(json.dumps(value))
    monkeypatch.setattr(grok, "verify_grok_archive", lambda *_a, **_k: None)
    calls = []
    monkeypatch.setattr(benchmark.subprocess, "run", lambda argv, **kwargs:
        calls.append(argv) or SimpleNamespace(returncode=0, stdout="Authored dry run", stderr=""))
    prepared = benchmark.prepare(dataset=dataset, output=tmp_path / "out", archive=archive,
        arm="full", provider_profile=GROK_PROFILE, grok_coding_max_turns=turns)
    assert prepared["provider_calls"] == 0 and len(calls) == 1 and "--dry-run" in calls[0]
    config = json.loads((tmp_path / "out/config.json").read_text())
    if turns == 128:
        assert "grok_coding_max_turns" not in prepared
        assert "grok_coding_max_turns" not in config["agents"][0]["kwargs"]
    else:
        assert prepared["grok_coding_max_turns"] == config["agents"][0]["kwargs"]["grok_coding_max_turns"] == turns
        assert prepared["comparison_controls"]["schema"] == controls.GROK_CODING_SCHEMA


@pytest.mark.parametrize("turns", [16, 128])
@pytest.mark.parametrize("provider_free", [False, True])
def test_harbor_forwards_optional_bound_and_keeps_usage(tmp_path, monkeypatch, turns, provider_free):
    from harbor.models.agent.context import AgentContext
    from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as adapter
    (tmp_path / "manifest.json").write_text(json.dumps(manifest()))
    monkeypatch.setattr(adapter, "capture_public_inputs", AsyncMock(return_value={}))
    monkeypatch.setattr(adapter, "export_public_outputs", AsyncMock(return_value={}))
    calls = []
    class Environment:
        async def upload_file(self, *_): pass
        async def exec(self, **kwargs):
            calls.append(shlex.split(kwargs["command"]))
            return SimpleNamespace(return_code=0, stdout="", stderr="")
        async def download_file(self, source, target):
            report = {"task_completed": False, "seconds": 1, "phases": {},
                **budget.grok_coding_selection(turns), "provider_invocations": [{
                    "invocation_id": "authored", "phase": "coding", "purpose": "coding", "provider": "grok_cli",
                    "provider_invocation_policy": {"max_turns": turns},
                    "native_rollout_usage": {"usage": {"input_tokens": 2, "cached_input_tokens": 0,
                        "output_tokens": 1, "total_tokens": 3}}}]}
            if provider_free:
                invocation = report["provider_invocations"][0]
                invocation.update(phase="planning", purpose="planning")
                report.update(provider_free_report(turns))
                report["provider_invocations"] = [invocation]
            Path(target).write_text(json.dumps(report if source.endswith("-result.json") else {}))
    agent = adapter.FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name="grok-4.7",
        runtime_archive=str(tmp_path), provider_profile=GROK_PROFILE, grok_coding_max_turns=turns)
    context = AgentContext()
    asyncio.run(agent.run("Authored task", Environment(), context))
    assert len(calls) == 1 and context.metadata["usage"]["total_tokens"] == 3
    if turns == 128:
        assert "--grok-coding-max-turns" not in calls[0]
        assert not any(key.startswith("grok_coding_") for key in context.metadata)
    else:
        assert calls[0].count("--grok-coding-max-turns") == 1
        assert calls[0][calls[0].index("--grok-coding-max-turns") + 1] == str(turns)
        assert context.metadata["grok_coding_runtime_declared_matches"] is True
        assert context.metadata["grok_coding_runtime_policy_matches"] is (None if provider_free else True)
        assert context.metadata["grok_coding_not_applicable_provider_free"] is provider_free


@pytest.mark.parametrize("arm,route", [("full", "model_router"), ("no-index", "model_router"),
    ("full", "doctor_candidate"), ("full", "doctor_contract_candidate")])
def test_driver_binds_command_before_native_start_and_labels_symbolic_nonapplication(tmp_path, monkeypatch, arm, route):
    from benchmarks.agent_supervisor.container_coding import benchmark_provider_profile as profiles
    from test.api.test_terminal_doctor_dispatch import (
        test_driver_selects_before_owner_and_preserves_provider_accounting as exercise_driver,
    )
    selected = resolve_provider_profile(GROK_PROFILE)
    monkeypatch.setattr(profiles, "resolve_provider_profile", lambda _name=None: dict(selected))
    original = doctor.implementation_argv
    seen = []
    def capture(**kwargs):
        assert kwargs["provider"] == "grok_cli"
        assert kwargs["grok_coding_max_turns"] == 16
        command = original(**kwargs)
        seen.append(command)
        return command
    monkeypatch.setattr(doctor, "implementation_argv", capture)
    exercise_driver(tmp_path, monkeypatch, arm, route, False,
        "supervisor-semantic-router-input@1", controls={"grok_coding_max_turns": 16})
    assert len(seen) == 1
    assert ("--grok-coding-max-turns" in seen[0]) is (route == "model_router")


def test_prepare_cli_binds_value_and_later_operations_cannot_override(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(benchmark, "prepare", lambda **kwargs: calls.append(kwargs) or {})
    monkeypatch.setattr(sys, "argv", ["benchmark", "prepare", "--dataset", str(tmp_path),
        "--archive", str(tmp_path), "--output", str(tmp_path / "out"),
        "--provider-profile", GROK_PROFILE, "--grok-coding-max-turns", "16"])
    benchmark.main()
    assert calls[0]["grok_coding_max_turns"] == 16
    for operation in ("collect", "execute"):
        monkeypatch.setattr(sys, "argv", ["benchmark", operation, "--output", str(tmp_path),
            "--grok-coding-max-turns", "16"])
        with pytest.raises(SystemExit) as error:
            benchmark.main()
        assert error.value.code == 2
