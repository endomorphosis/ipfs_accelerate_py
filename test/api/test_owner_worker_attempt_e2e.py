"""Installed owner monitor -> real local children -> private attempt census.

Container boundary admission and the worker cleanup command are authored. The
writer, monitor, child birth/wait, immutable files and collector are production
code. Router receipts below are authored accounting fixtures, not model calls.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from benchmarks.agent_supervisor.container_coding.container_worker_deployment import OWNER_MONITOR, OWNER_ENTRY
from benchmarks.agent_supervisor.container_coding import terminal_owner_worker_attempt_observation as census
from ipfs_accelerate_py.agent_supervisor.runtime import owner_worker_attempt as attempts
from ipfs_accelerate_py.agent_supervisor.runtime import source_formalization_context as guard


@pytest.fixture
def environment(tmp_path, monkeypatch):
    boundary = tmp_path / "container-boundary.json"
    boundary.write_text("authored container boundary")
    parent = tmp_path / "worker-observations"
    parent.mkdir(mode=0o750)
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    monkeypatch.setattr(guard, "_owner_boundary", lambda path: (
        dict(owner_uid=os.getuid(), worker_uid=os.getuid() + 1), "b" * 64))
    monkeypatch.setattr(guard, "_protected_directory", lambda *a, **kw: None)
    monkeypatch.setattr(census, "_protected_directory", lambda *a, **kw: None)
    descriptor = os.memfd_create("authored-owner-attempt-prompt", os.MFD_ALLOW_SEALING)
    prompt = b"Implement the complete original public requirement.\n"
    os.write(descriptor, prompt)
    fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, 15)
    # pread-based custody must preserve the caller's current fd position.
    position = os.lseek(descriptor, 5, os.SEEK_SET)
    yield boundary, workspace, parent, descriptor, prompt, position
    os.close(descriptor)


def _execute(environment, monkeypatch, *, code=7, fail=None):
    boundary, workspace, parent, descriptor, prompt, position = environment
    attempt = attempts.prepare_owner_worker_attempt(boundary=boundary, workspace=workspace,
        input_fd=descriptor, absolute_deadline=time.monotonic() + 30)
    assert os.lseek(descriptor, 0, os.SEEK_CUR) == position
    original_run = subprocess.run
    events = []

    def cleanup(command, **kwargs):
        assert command[-1] == "--cleanup"
        events.append("cleanup")
        return original_run([sys.executable, "-c", "raise SystemExit(1)" if fail == "cleanup" else "pass"], **kwargs)

    class Broker:
        def note_spawned(self, process):
            events.append("register")
            # Birth is durable before broker registration can fail.
            assert (parent / f"worker-attempt-{attempt.attempt_id}-spawned.json").is_file()
            if fail == "register":
                raise RuntimeError("authored broker registration failure")

        def close(self):
            events.append("close")
            if fail == "close":
                raise RuntimeError("authored broker close failure")

    with monkeypatch.context() as local:
        local.setattr(subprocess, "run", cleanup)
        local.setattr(signal, "signal", lambda *args: None)
        namespace = dict(command=[sys.executable, "-c", f"raise SystemExit({code})"],
            owner_worker_attempt=attempt, numerical_obligation_broker=Broker(),
            worker_source_broker=None)
        if fail == "spawn":
            def refused(*args, **kwargs):
                raise RuntimeError("authored interrupted process creation")
            # Cleanup must still execute a real child through the original
            # constructor; only the monitor's coding launch is refused.
            original_popen = subprocess.Popen
            def spawn(command, *args, **kwargs):
                if command[-1] == f"raise SystemExit({code})":
                    return refused()
                return original_popen(command, *args, **kwargs)
            local.setattr(subprocess, "Popen", spawn)
        expected = RuntimeError if fail in {"spawn", "register", "close"} else SystemExit
        with pytest.raises(expected) as stopped:
            exec(compile(OWNER_MONITOR, "<installed-owner-monitor>", "exec"), namespace)
        if expected is SystemExit:
            assert stopped.value.code == (1 if fail == "cleanup" else code)
        if fail != "spawn":
            assert namespace["process"].returncode is not None
    assert events[-2:] == ["cleanup", "close"]
    row = dict(schema="router-implementation-invocation@1", phase="coding", purpose="coding",
        invocation_id="authored-" + attempt.attempt_id, owner_worker_attempt_id=attempt.attempt_id,
        native_prompt_sha256=hashlib.sha256(prompt).hexdigest(), workspace=str(workspace),
        provider="codex_cli", status="failed" if code else "provider_returned", usage=None)
    return attempt, row


def test_two_children_each_need_a_distinct_receipt(environment, monkeypatch):
    first, row1 = _execute(environment, monkeypatch, code=7)
    second, row2 = _execute(environment, monkeypatch, code=0)
    assert first.attempt_id != second.attempt_id
    report = dict(owner_worker_attempt_census_expected=True, provider_invocations=[row1, row2])
    complete = census.collect(parent=environment[2], report=report)
    assert complete["all_attempts_receipted"] is True, complete
    # A successful first receipt cannot cover a later unreceipted birth.
    report["provider_invocations"] = [row1]
    missing = census.collect(parent=environment[2], report=report)
    assert missing["all_attempts_receipted"] is False, missing
    # Evidence is a census, not authoritative native token usage.
    assert row1["usage"] is row2["usage"] is None
    assert all(value["execution_authority"] is value["completion_authority"] is False
        for value in (complete, missing))
    exported = json.dumps(missing)
    assert environment[4].decode().strip() not in exported
    assert str(environment[1]) not in exported


@pytest.mark.parametrize("failure", ["register", "cleanup", "close"])
def test_monitor_persists_uncertain_or_failed_retirement(environment, monkeypatch, failure):
    attempt, row = _execute(environment, monkeypatch, fail=failure)
    retired = json.loads((environment[2] / f"worker-attempt-{attempt.attempt_id}-retired.json").read_bytes())
    assert retired["waited"] is (failure != "register")
    assert retired["cleanup_succeeded"] is (failure != "cleanup")
    assert retired["brokers_closed"] is (failure != "close")
    result = census.collect(parent=environment[2], report=dict(
        owner_worker_attempt_census_expected=True, provider_invocations=[row]))
    assert result["all_attempts_receipted"] is False, result


def test_process_creation_error_remains_unknown(environment, monkeypatch):
    attempt, _ = _execute(environment, monkeypatch, fail="spawn")
    assert not (environment[2] / f"worker-attempt-{attempt.attempt_id}-retired.json").exists()
    result = census.collect(parent=environment[2], report=dict(
        owner_worker_attempt_census_expected=True, provider_invocations=[]))
    assert result["all_attempts_receipted"] is False


def test_retirement_write_failure_preserves_worker_exit(environment, monkeypatch, capsys):
    def unavailable(*args, **kwargs):
        raise OSError("authored private retirement write failure")
    monkeypatch.setattr(attempts.OwnerWorkerAttempt, "finish", unavailable)
    attempt, row = _execute(environment, monkeypatch)
    assert not (environment[2] / f"worker-attempt-{attempt.attempt_id}-retired.json").exists()
    observation = json.loads(capsys.readouterr().out)
    assert observation["status"] == "unavailable"
    assert "authored private" not in json.dumps(observation)
    result = census.collect(parent=environment[2], report=dict(
        owner_worker_attempt_census_expected=True, provider_invocations=[row]))
    assert result["all_attempts_receipted"] is False


@pytest.mark.parametrize("arguments,selected", [
    ([], True), (["--purpose=coding"], True), (["--purpose", "planning"], False),
    (["--preflight"], False), (["--semantic-metadata-census-only"], False),
    (["--doctor-candidate-artifact=path"], False),
    (["--doctor-contract-artifact", "path"], False),
    (["--finite-repository-artifact=path"], False),
    (["--semantic-decision-policy-artifact=path"], False),
])
def test_installed_owner_selects_only_ordinary_coding(arguments, selected):
    from ipfs_accelerate_py.agent_supervisor.runtime.numerical_obligation_broker import _argument
    start = OWNER_ENTRY.index("owner_attempt_excluded=")
    stop = OWNER_ENTRY.index(" if owner_attempt_selected:", start)
    # First line starts at the name, whereas subsequent lines retain try indent.
    fragment = OWNER_ENTRY[start:stop].replace("\n ", "\n")
    namespace = dict(handoff=dict(argv=arguments), worker_arguments=arguments, _argument=_argument)
    exec(compile(fragment, "<installed-owner-attempt-selection>", "exec"), namespace)
    assert namespace["owner_attempt_selected"] is selected
