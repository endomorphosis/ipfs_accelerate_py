"""Keep accepted CLI provenance and native entrypoint admission intact."""

from __future__ import annotations

import argparse
import inspect
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import process_security
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_supervisor as supervisor_module,
)


def test_scheduler_expansion_is_parsed_once_without_losing_launch_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = tmp_path / "scheduler.json"
    incoming = ["--scheduler-config", str(profile), "--check-interval", "17"]
    expanded = ["--stale-seconds", "801", "--check-interval", "23",
                "--check-interval", "17"]
    expansion_calls = []
    parse_calls = []
    native_parse = argparse.ArgumentParser.parse_args

    def expand(argv):
        expansion_calls.append(tuple(argv))
        return list(expanded), profile

    def parse(parser, args=None, namespace=None):
        parse_calls.append(tuple(args))
        return native_parse(parser, args, namespace)

    monkeypatch.setattr(supervisor_module, "expand_supervisor_scheduler_config_args", expand)
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse)
    monkeypatch.setattr(sys, "argv", ["foreign-launcher", "--check-interval", "999"])
    parsed = supervisor_module.parse_args(incoming)

    assert expansion_calls == [tuple(incoming)]
    assert parse_calls == [tuple(expanded)]
    assert parsed.scheduler_config == profile
    assert parsed.stale_seconds == 801
    assert parsed.check_interval == 17
    assert parsed._configured_board_live_context is None
    assert parsed._control_plane_reload_argv == tuple(incoming)
    assert parsed._accepted_launch_argv == tuple(incoming)


def _live_argv() -> list[str]:
    return [
        "--require-lgcvf-configured-board-live-seal",
        supervisor_module.LGCVF_CONFIGURED_BOARD_LIVE_CONFIG_PATH,
        "--configured-board-live-capsule-pin-json", "{}",
        "--configured-board-live-capsule-fd", "10",
        "--configured-board-live-admission-json", "{}",
        "--configured-board-live-native-launch-json", "{}",
        "--configured-board-live-native-fd", "11",
    ]


def test_verified_live_context_survives_parse_and_config_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = object()
    verified = []

    def verify(**kwargs):
        verified.append(kwargs)
        return context

    monkeypatch.setattr(supervisor_module, "verify_lgcvf_configured_board_live_context", verify)
    monkeypatch.setattr(supervisor_module, "database_program_from_cli_namespace", lambda _args: None)
    # Inspect the parser-to-config handoff; native profile admission has its
    # own descriptor/owner tests and must still run on the forwarded context.
    monkeypatch.setattr(supervisor_module, "PortalSupervisorConfig", SimpleNamespace)
    incoming = _live_argv()
    args = supervisor_module.parse_args(incoming)
    config = supervisor_module.supervisor_config_from_args(args, repo_root=tmp_path)

    assert len(verified) == 1
    assert verified[0]["capsule_descriptor"] == 10
    assert verified[0]["native_descriptor"] == 11
    assert args._configured_board_live_context is context
    assert config.configured_board_live_context is context
    assert config.accepted_launch_argv == tuple(incoming)
    assert config.control_plane_reload_argv == tuple(incoming)


@pytest.mark.parametrize("case", ["incomplete", "foreign_config", "mutable_scheduler", "invalid_context"])
def test_live_admission_refusal_is_preserved(case: str, monkeypatch: pytest.MonkeyPatch) -> None:
    incoming = _live_argv()
    calls = []

    def verify(**kwargs):
        calls.append(kwargs)
        raise ValueError("fixture admission refusal")

    monkeypatch.setattr(supervisor_module, "verify_lgcvf_configured_board_live_context", verify)
    if case == "incomplete":
        incoming = incoming[:-2]
    elif case == "foreign_config":
        incoming[1] = "foreign.json"
    elif case == "mutable_scheduler":
        incoming.extend(["--scheduler-config", "mutable.json"])
    with pytest.raises(SystemExit) as caught:
        supervisor_module.parse_args(incoming)
    assert caught.value.code == 2
    assert len(calls) == (1 if case == "invalid_context" else 0)


def test_empty_accepted_launch_is_distinct_from_no_launch_provenance(tmp_path: Path) -> None:
    args = supervisor_module.parse_args([])
    assert args._accepted_launch_argv == ()
    assert args._control_plane_reload_argv == ()
    config = supervisor_module.PortalSupervisorConfig(
        todo_path=tmp_path / "todo.md", state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json", events_path=tmp_path / "events.jsonl",
        state_dir=tmp_path,
    )
    assert config.accepted_launch_argv is None


def _stub_hardening(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> None:
    monkeypatch.setattr(process_security, "harden_state_authority_process", lambda: events.append("harden"))
    monkeypatch.setattr(process_security, "capture_state_authority_credentials", lambda: events.append("capture"))


def test_main_routes_native_child_marker_before_ordinary_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    events = []
    child_argv = ["--native-child-fixture"]
    _stub_hardening(monkeypatch, events)

    def forbidden_parse(_argv):
        pytest.fail("native child marker entered ordinary argument parsing")

    def child(argv):
        assert argv == child_argv
        events.append("child")
        return 78

    monkeypatch.setattr(supervisor_module, "parse_args", forbidden_parse)
    monkeypatch.setattr(supervisor_module, "_run_plan_bound_daemon_child", child)
    assert supervisor_module.main([supervisor_module.PLAN_BOUND_DAEMON_CHILD_MARKER, *child_argv]) == 78
    assert events == ["harden", "capture", "child"]


def test_main_parses_accepted_arguments_once_and_closes_supervisor(monkeypatch: pytest.MonkeyPatch) -> None:
    events = []
    incoming = ["--once", "--check-interval", "17"]
    native_parse = supervisor_module.parse_args
    _stub_hardening(monkeypatch, events)

    def parse(argv):
        assert argv == incoming
        events.append("parse")
        return native_parse(argv)

    def config(args, **_kwargs):
        assert args._accepted_launch_argv == tuple(incoming)
        events.append("config")
        return args

    class Supervisor:
        def __init__(self, args):
            assert args.check_interval == 17

        def run_once(self):
            events.append("run")
            return {}

        def close(self):
            events.append("close")

    monkeypatch.setattr(supervisor_module, "parse_args", parse)
    monkeypatch.setattr(supervisor_module, "supervisor_config_from_args", config)
    monkeypatch.setattr(supervisor_module, "PortalImplementationSupervisor", Supervisor)
    assert supervisor_module.main(incoming) == 0
    assert events == ["harden", "capture", "parse", "config", "run", "close"]


@pytest.mark.filterwarnings("ignore:.*found in sys.modules.*:RuntimeWarning")
def test_module_execution_defines_native_helpers_before_entering_main(monkeypatch: pytest.MonkeyPatch) -> None:
    events = []
    _stub_hardening(monkeypatch, events)

    class StopBeforeDispatch(Exception):
        pass

    def stop_at_parser(_parser, args=None, namespace=None):
        module_globals = inspect.currentframe().f_back.f_globals
        assert callable(module_globals["_require_current_released_wave_diff_barrier_locked"])
        for name in ("ORDINARY_IMPLEMENTATION_SUPERVISOR_BOOTSTRAP", "ORDINARY_IMPLEMENTATION_DAEMON_BOOTSTRAP"):
            assert "preload_sealed_native_dependency_from_environment" in module_globals[name]
        assert module_globals["TodoImplementationSupervisor"] is module_globals["PortalImplementationSupervisor"]
        raise StopBeforeDispatch

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", stop_at_parser)
    monkeypatch.setattr(sys, "argv", ["implementation-supervisor", "--once"])
    with pytest.raises(StopBeforeDispatch):
        runpy.run_module(supervisor_module.__name__, run_name="__main__", alter_sys=True)
    assert events == ["harden", "capture"]
