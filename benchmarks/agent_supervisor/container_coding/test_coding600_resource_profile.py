"""Explicit provider allowance under unchanged enclosing and authority bounds."""
from copy import deepcopy
from pathlib import Path
import shlex
import sys

import pytest

from benchmarks.agent_supervisor.container_coding import benchmark_resource_profile as profiles
from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver


PROFILE = profiles.CODING600_SOURCE384_PROFILE


def test_explicit_coding_allowance_preserves_existing_profiles_and_outer_bounds():
    assert PROFILE == "source384-5cpu-20gib-coding600@1"
    prior = profiles.PLANNER180_20GIB_SOURCE384_PROFILE
    assert profiles.resource_environment(PROFILE) == profiles.resource_environment(prior)
    assert profiles.resource_environment(PROFILE)["override_memory_mb"] == 20480
    assert profiles.execution_budget(PROFILE) == profiles.execution_budget(prior)
    assert profiles.execution_budget(PROFILE)["driver_seconds"] == 900
    assert profiles.execution_budget(PROFILE)["cleanup_seconds"] == 60
    assert profiles.planner_timeout_seconds(PROFILE) == 180
    assert profiles.admission_environment(PROFILE) == profiles.admission_environment(prior)
    assert profiles.coding_timeout_seconds(PROFILE) == 600
    assert profiles.implementation_watchdog_seconds(PROFILE) == 660
    for selected in (None, *[p for p in profiles.PROFILES if p != PROFILE]):
        assert profiles.coding_timeout_seconds(selected) == 300
        assert profiles.implementation_watchdog_seconds(selected) == 360
    for function in (profiles.coding_timeout_seconds, profiles.implementation_watchdog_seconds):
        with pytest.raises(ValueError, match="unknown"):
            function("source384-unreviewed@1")


def test_driver_cli_passes_explicit_selection_without_changing_outer_timeout(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(sys, "argv", ["driver", "--instruction", "/authored/instruction",
        "--state", "/authored/state", "--arm", "full", "--resource-profile", PROFILE])
    def run(**kwargs):
        seen.append(kwargs)
        return dict(task_completed=False, arm="full", seconds=0, provider_invocations=[])
    monkeypatch.setattr(driver, "run", run)
    assert driver.main() == 1
    assert seen[0]["resource_profile"] == PROFILE
    assert seen[0]["timeout_seconds"] is None
    assert profiles.execution_budget(seen[0]["resource_profile"])["driver_seconds"] == 900
    assert "provider_invocations" in capsys.readouterr().out


@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_frozen_configuration_binds_coding_profile_but_common_controls_only_compare_outer_limits(tmp_path, arm):
    from harbor.models.job.config import JobConfig
    from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
    from benchmarks.agent_supervisor.container_coding.benchmark_controls import compare_controls, observe_controls
    from benchmarks.agent_supervisor.container_coding.test_benchmark_controls import observation, HASHES
    def config(selected):
        value = benchmark.config_for(tmp_path / "data", tmp_path / "out", tmp_path / "archive",
            arm, resource_profile=selected)
        return JobConfig.model_validate(value, extra="forbid").model_dump(mode="json")
    current = config(PROFILE)
    prior = config(profiles.PLANNER180_20GIB_SOURCE384_PROFILE)
    left, right = observation(current), observation(prior)
    assert current["agents"][0]["kwargs"]["resource_profile"] != prior["agents"][0]["kwargs"]["resource_profile"]
    assert left["declared"]["configuration_sha256"] != right["declared"]["configuration_sha256"]
    # The frozen @1 comparison deliberately excludes adapter-specific kwargs.
    # Equal outer resources are not evidence of equal provider coding budgets.
    assert compare_controls(left, right)["matches"] is True
    assert "runtime_enforcement_not_established" in compare_controls(left, right)["basis"]
    changed = deepcopy(current)
    changed["agents"][0]["kwargs"]["resource_profile"] = profiles.PLANNER180_20GIB_SOURCE384_PROFILE
    reobserved = observe_controls({"comparison_controls": left["declared"]}, changed,
        current_task_hashes=HASHES)
    assert reobserved["status"] == "mismatch"
    assert reobserved["configuration_unchanged"] is False


def test_native_signed_launch_binds_600_command_660_watchdog_and_900_lifetime(tmp_path, monkeypatch):
    from benchmarks.agent_supervisor.container_coding.native_quack_qualification import open_existing_native_owner
    from benchmarks.agent_supervisor.container_coding.terminal_doctor_dispatch import implementation_argv
    from ipfs_accelerate_py.agent_supervisor.control import profile_authority
    from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
    from ipfs_accelerate_py.agent_supervisor.runtime.local_planning_admission import verify_local_benchmark_admission
    from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE
    from test.integration.test_admitted_benchmark_runtime import _prepare_implementation_fixture
    monkeypatch.setattr(profile_authority, "_LIFECYCLE_REGISTRY_ROOT_OVERRIDE", tmp_path / "account")
    monkeypatch.delenv("IPFS_DATASETS_PROOF_RESOURCE_PROFILE", raising=False)
    prepared = _prepare_implementation_fixture(tmp_path / "task")
    verified = verify_local_benchmark_admission(prepared["admission"])
    command = implementation_argv(router=Path("/authored/router"), model="grok-4.7", reasoning="high",
        timeout=profiles.coding_timeout_seconds(PROFILE), semantic_repository=None, doctor=None)
    command += ["--provider", "grok_cli"]
    with open_existing_native_owner(database=Path(prepared["intent_database"]),
            checkout=Path(prepared["repository"]), state_dir=tmp_path / "owner",
            repository_id=verified["manifest"]["repository_cid"],
            execution_routes={prepared["task_id"]: GROK_CODEX_EXECUTION_MODE}) as owner:
        runtime = AdmittedBenchmarkRuntime.create(tmp_path / "launch", admission=prepared["admission"],
            server=owner.server, source=owner.source, implement=True,
            implementation_command=shlex.join(command), implementation_timeout_seconds=660,
            lifetime_seconds=900, max_task_attempts=1)
        try:
            assert runtime.manifest["implementation_timeout_seconds"] == 660
            assert runtime.manifest["lifetime_seconds"] == 900
            bound = runtime.profile.argv[runtime.profile.argv.index("--implementation-command") + 1]
            assert shlex.split(bound) == command
            assert command[command.index("--timeout") + 1] == "600"
            runtime._verify()
            runtime.implementation_timeout_seconds = 661
            with pytest.raises(ValueError):
                runtime._verify()
        finally:
            runtime.implementation_timeout_seconds = 660
            assert not runtime.process.snapshot(runtime.profile).members
            runtime.close()
