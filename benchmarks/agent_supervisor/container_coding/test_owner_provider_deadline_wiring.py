"""Installed owner deadline flow with real inert children and authored cleanup.

The generated monitor runs unchanged. The sudo cleanup command and deployment
boundary are authored seams; no Docker, provider, target code or database runs.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import pathlib
import signal
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding.container_worker_deployment import (
    OWNER_ENTRY, OWNER_MONITOR, WORKER_ENTRY,
)
from ipfs_accelerate_py.agent_supervisor.runtime import owner_provider_deadline as deadline_module
from ipfs_accelerate_py.agent_supervisor.runtime import source_formalization_context as formalization


def _monitor(monkeypatch, *, with_input=False, stage=None, inherited=True):
    actual_clock = time.monotonic_ns
    deadline = actual_clock() + (150_000_000 if stage == "during-wait" else 30_000_000_000)
    clock = {"now": None}
    if stage == "before-birth": clock["now"] = deadline
    monkeypatch.setattr(time, "monotonic_ns", lambda: actual_clock() if clock["now"] is None else clock["now"])
    monkeypatch.setattr(signal, "signal", lambda *_args: None)
    events, children, wait_options, communication_options, census = [], [], [], [], []
    original_popen = subprocess.Popen

    def popen(command, **kwargs):
        events.append("birth")
        child = original_popen(command, **kwargs)
        children.append(child)
        actual_wait, actual_communicate = child.wait, child.communicate
        def wait(**options):
            wait_options.append(dict(options))
            result = actual_wait(**options)
            if stage == "after-wait": clock["now"] = deadline
            return result
        def communicate(**options):
            communication_options.append(dict(options))
            return actual_communicate(**options)
        child.wait, child.communicate = wait, communicate
        return child

    def register(child):
        events.append("registered")
        assert child is children[0]
        if stage == "after-birth": clock["now"] = deadline

    def cleanup(command, **kwargs):
        assert command[-1] == "--cleanup" and kwargs["timeout"] == 15
        events.append("cleanup")
        assert all(child.poll() is not None for child in children)
        return SimpleNamespace(returncode=0)

    def finish(**flags):
        events.append("census")
        census.append(flags)

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(subprocess, "run", cleanup)
    attempt = SimpleNamespace(note_spawned=register, finish=finish)
    code = ("import time;time.sleep(60)" if stage in ("after-birth", "during-wait") else
        "import sys;sys.stdin.buffer.read();raise SystemExit(0)" if with_input else "raise SystemExit(0)")
    namespace = dict(command=[sys.executable, "-I", "-c", code],
        owner_worker_attempt=attempt, owner_candidate_round_deadline=None)
    if inherited: namespace["owner_provider_deadline_ns"] = deadline
    if with_input: namespace["owner_prompt_bytes"] = b"Exact retained owner prompt."
    error = None
    try:
        exec(compile(OWNER_MONITOR, "<installed-owner-deadline-monitor>", "exec"), namespace)
    except BaseException as caught:
        error = caught
    finally:
        # An assertion failure must still leave this authored child reaped.
        for child in children:
            if child.poll() is None: child.kill()
            child.wait(timeout=5)
            if child.stdin is not None: child.stdin.close()
    assert error is not None
    return SimpleNamespace(error=error, events=events, children=children,
        waits=wait_options, communications=communication_options, census=census,
        deadline=deadline)


@pytest.mark.parametrize("with_input", [False, True])
def test_actual_monitor_waits_with_remaining_original_budget_and_retires(monkeypatch, with_input):
    result = _monitor(monkeypatch, with_input=with_input)
    assert type(result.error) is SystemExit and result.error.code == 0
    assert len(result.children) == 1 and result.children[0].returncode == 0
    selected = result.communications[0] if with_input else result.waits[0]
    assert 0 < selected["timeout"] <= 30
    if with_input: assert selected["input"] == b"Exact retained owner prompt."
    assert result.census == [dict(waited=True, cleanup_succeeded=True, brokers_closed=True)]
    assert result.events == ["birth", "registered", "cleanup", "census"]


def test_expiry_before_birth_never_constructs_worker_but_preserves_cleanup(monkeypatch):
    result = _monitor(monkeypatch, stage="before-birth")
    assert type(result.error) is TimeoutError
    assert result.children == result.waits == result.communications == result.census == []
    assert result.events == ["cleanup"]


@pytest.mark.parametrize("with_input", [False, True])
def test_expiry_after_actual_birth_reaps_without_delivery_or_waited_success(monkeypatch, with_input):
    result = _monitor(monkeypatch, with_input=with_input, stage="after-birth")
    assert type(result.error) is TimeoutError
    assert len(result.children) == 1 and result.children[0].returncode is not None
    assert result.communications == []
    assert result.census == [dict(waited=False, cleanup_succeeded=True, brokers_closed=True)]
    assert result.events == ["birth", "registered", "cleanup", "census"]


def test_original_wait_timeout_reaps_real_blocked_worker_and_keeps_false_wait_census(monkeypatch):
    result = _monitor(monkeypatch, stage="during-wait")
    assert isinstance(result.error, subprocess.TimeoutExpired)
    assert len(result.children) == 1 and result.children[0].returncode is not None
    assert 0 < result.waits[0]["timeout"] <= .15
    assert result.census == [dict(waited=False, cleanup_succeeded=True, brokers_closed=True)]


def test_expiry_after_real_wait_cannot_turn_expired_deadline_into_success(monkeypatch):
    result = _monitor(monkeypatch, stage="after-wait")
    assert type(result.error) is TimeoutError
    assert len(result.children) == 1 and result.children[0].returncode == 0
    assert result.census == [dict(waited=False, cleanup_succeeded=True, brokers_closed=True)]


@pytest.mark.parametrize("with_input", [False, True])
def test_absent_inherited_deadline_keeps_existing_monitor_call_contract(monkeypatch, with_input):
    result = _monitor(monkeypatch, with_input=with_input, inherited=False)
    assert type(result.error) is SystemExit and result.error.code == 0
    selected = result.communications[0] if with_input else result.waits[0]
    assert "timeout" not in selected
    assert result.census == [dict(waited=True, cleanup_succeeded=True, brokers_closed=True)]


@pytest.mark.parametrize("spelling", ["--owner-provider-deadline-ns", "--owner-provider-deadline-ns=123"])
def test_owner_consumes_restriction_before_rejecting_caller_deadline_and_before_chmod(monkeypatch, tmp_path, spelling):
    seen = []
    inherited = time.monotonic_ns() + 30_000_000_000
    def consume(**kwargs):
        seen.append(kwargs)
        return inherited
    monkeypatch.setattr(deadline_module, "consume_owner_provider_deadline_handoff", consume)
    monkeypatch.setattr(sys, "argv", [deadline_module.OWNER_WRAPPER, spelling, "123"])
    start = OWNER_ENTRY.index("from ipfs_accelerate_py.agent_supervisor.runtime.owner_provider_deadline import")
    stop = OWNER_ENTRY.index("if cwd.resolve()!=cwd")
    with pytest.raises(SystemExit, match="generated by the owner"):
        exec(compile(OWNER_ENTRY[start:stop], "<installed-owner-private-deadline>", "exec"),
            dict(cwd=tmp_path, sys=sys))
    assert seen == [dict(command=list(sys.argv), workspace=tmp_path)]
    assert stop < OWNER_ENTRY.index("owner_workspace_provisioning=")


@pytest.mark.parametrize("timeout", ["1", "90", "1800", "nan"])
def test_inherited_deadline_limits_census_and_refresh_independently_of_fresh_wrapper_timeout(monkeypatch, timeout):
    inherited = time.monotonic_ns() + 25_000_000_000
    started = time.monotonic()
    namespace = dict(owner_candidate_round_started=started,
        owner_provider_deadline_ns=inherited, sys=sys, math=math)
    monkeypatch.setattr(sys, "argv", [deadline_module.OWNER_WRAPPER, "--timeout", timeout])
    start = OWNER_ENTRY.index("# The fixed lease includes")
    stop = OWNER_ENTRY.index("# Initial provisioning completes")
    exec(compile(OWNER_ENTRY[start:stop], "<installed-owner-original-budget>", "exec"), namespace)
    result = namespace["owner_candidate_round_deadline"]
    effective = inherited
    if timeout != "nan":
        numerator, denominator = (started + int(timeout)).as_integer_ratio()
        effective = min(effective, numerator * 1_000_000_000 // denominator)
    assert namespace["owner_provider_deadline_ns"] == effective
    assert result == math.nextafter(effective / 1_000_000_000, -math.inf)
    assert result * 1_000_000_000 <= effective <= inherited


@pytest.mark.parametrize("attempt,decision,accepted", [("a" * 32, False, True),
    (None, False, False), ("a" * 32, True, False)])
def test_worker_parser_and_forwarder_preserve_exact_integer_restriction_for_ordinary_owner(monkeypatch, attempt, decision, accepted):
    inherited = time.monotonic_ns() + 30_000_000_000
    arguments = ["worker-entry", "--provider", "codex_cli", "--model", "pinned",
        "--owner-provider-deadline-ns", str(inherited)]
    if attempt is not None: arguments += ["--owner-worker-attempt-id", attempt]
    monkeypatch.setattr(sys, "argv", arguments)
    start = WORKER_ENTRY.index("parser=argparse.ArgumentParser()")
    stop = WORKER_ENTRY.index("if args.provider!=selected_provider")
    namespace = dict(argparse=argparse, pathlib=pathlib, selected_provider="codex_cli", sys=sys)
    exec(compile(WORKER_ENTRY[start:stop], "<installed-worker-parser>", "exec"), namespace)
    namespace.update(artifact=Path("/authored/boundary.json"), digest="a" * 64,
        decision_options=decision)
    start = WORKER_ENTRY.index(" sys.argv=[sys.argv[0],'--provider'")
    stop = WORKER_ENTRY.index(" if args.budget_profile is not None:sys.argv+=", start)
    forwarding = textwrap.dedent(WORKER_ENTRY[start:stop])
    if accepted:
        exec(compile(forwarding, "<installed-worker-deadline-forwarding>", "exec"), namespace)
        index = sys.argv.index("--owner-provider-deadline-ns")
        assert sys.argv[index + 1] == str(inherited)
        assert sys.argv.count("--owner-provider-deadline-ns") == 1
    else:
        with pytest.raises(SystemExit):
            exec(compile(forwarding, "<installed-worker-deadline-forwarding>", "exec"), namespace)


def test_expired_inherited_deadline_refuses_prompt_capture_before_select_or_read(monkeypatch):
    import select
    descriptor = os.memfd_create("authored-expired-owner-prompt", os.MFD_CLOEXEC)
    os.write(descriptor, b"Retained original prompt.")
    os.lseek(descriptor, 0, os.SEEK_SET)
    def forbidden(*_args, **_kwargs):
        pytest.fail("expired original deadline entered prompt I/O")
    monkeypatch.setattr(select, "select", forbidden)
    monkeypatch.setattr(os, "read", forbidden)
    try:
        with pytest.raises(TimeoutError, match="capture deadline"):
            formalization.prepare_container_source_formalization_dispatch(argv=[], boundary=None,
                workspace=None, input_fd=descriptor, absolute_deadline=time.monotonic() - 1)
        assert os.lseek(descriptor, 0, os.SEEK_CUR) == 0
    finally:
        os.close(descriptor)


def test_prompt_capture_uses_remaining_inherited_budget_without_restart(monkeypatch):
    import select
    deadline = time.monotonic() + 15
    options = []
    descriptor = os.memfd_create("authored-owner-prompt", os.MFD_CLOEXEC)
    os.write(descriptor, b"Retained original prompt.")
    os.lseek(descriptor, 0, os.SEEK_SET)
    def no_ready(readers, writers, errors, timeout):
        assert readers == [descriptor] and writers == errors == []
        options.append(timeout)
        return [], [], []
    monkeypatch.setattr(select, "select", no_ready)
    try:
        with pytest.raises(TimeoutError, match="capture deadline"):
            formalization.prepare_container_source_formalization_dispatch(argv=[], boundary=None,
                workspace=None, input_fd=descriptor, absolute_deadline=deadline)
    finally:
        os.close(descriptor)
    assert len(options) == 1 and 0 < options[0] <= 15
