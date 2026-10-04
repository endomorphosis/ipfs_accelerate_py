"""The benchmark's worker watchdog is explicit and signed, never a global default."""
import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime


@pytest.mark.parametrize("value", [True, False, 0, -1, 301, 1.0, "90", float("nan")])
def test_invalid_implementation_budget_refused_before_owner_or_state(tmp_path, value):
    with pytest.raises(ValueError, match="explicit implementation timeout"):
        AdmittedBenchmarkRuntime.create(tmp_path / "launch", admission=None, server=None, source=None,
            implement=True, implementation_command="/usr/bin/true", implementation_timeout_seconds=value)
    assert not (tmp_path / "launch").exists()


def test_observation_only_launch_cannot_carry_implementation_budget(tmp_path):
    with pytest.raises(ValueError, match="explicit implementation timeout"):
        AdmittedBenchmarkRuntime.create(tmp_path / "launch", admission=None, server=None, source=None,
            implementation_timeout_seconds=90)
    assert not (tmp_path / "launch").exists()


def test_changed_watchdog_budget_refused_before_owner_reads():
    runtime = object.__new__(AdmittedBenchmarkRuntime)
    runtime.start_timeout_ms = None
    runtime.implementation_timeout_seconds = 300
    runtime.manifest = {"implementation_timeout_seconds": 301}
    with pytest.raises(ValueError, match="implementation timeout differs"):
        runtime._verify()


@pytest.mark.parametrize("value", [None, 90, 300])
def test_native_constructor_signs_explicit_worker_budget(tmp_path, scenario, value):
    # Real Git, signed local admission, native DuckDB/Quack owner and runtime
    # construction. No START, provider, Docker, or source-code execution.
    from test.api.test_intent_requirement_observation_native import _admit, _owner
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import parse_args
    case = _admit(scenario)
    with _owner(case, tmp_path) as owner:
        runtime = AdmittedBenchmarkRuntime.create(tmp_path / "runtime", admission=case["admission"],
            server=owner.server, source=owner.source, implement=True,
            implementation_command="/usr/bin/true", implementation_timeout_seconds=value)
        try:
            argv = runtime.manifest["argv"]
            options = argv[argv.index("ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor") + 1:]
            assert runtime.manifest.get("implementation_timeout_seconds") == value
            assert runtime.implementation_timeout_seconds == value
            assert parse_args(options).implementation_timeout == (1800 if value is None else value)
            if value is None:
                assert "--implementation-timeout" not in argv
            else:
                assert argv.count("--implementation-timeout") == 1
                assert argv[argv.index("--implementation-timeout") + 1] == str(value)
        finally:
            runtime.close()


from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: E402,F401
