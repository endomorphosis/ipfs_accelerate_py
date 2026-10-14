"""Independent process and state-machine probes; no containers or providers."""
from __future__ import annotations

import json
import fcntl
import multiprocessing
import os
from pathlib import Path
import queue
import threading
import time

import pytest

from benchmarks.agent_supervisor.container_coding.terminal_provider_backoff import (
    BackoffStateError,
    ProviderBackoff,
)


PROVIDER = "codex_cli"


def _stop(reason="rate_limit", *, seconds=None, unbounded=False):
    result = {
        "schema": "terminal-provider-stop@1", "state": "rate_limited",
        "reason_code": reason, "observation_valid": True,
        "provider": PROVIDER, "model": "gpt-6.1-sol", "phase": "planning",
        "retry_after_seconds": seconds, "retry_authority": False,
        "automatic_replay_allowed": False, "task_completion_authority": False,
    }
    if unbounded:
        result["retry_after_unbounded"] = True
    return result


@pytest.fixture
def clock_gate(tmp_path):
    clock = {"wall": 10000.0, "mono": 100.0}
    gate = ProviderBackoff(tmp_path / "state", wall_clock=lambda: clock["wall"],
        monotonic=lambda: clock["mono"], jitter=lambda: 0, boot_id="review-boot")
    return clock, gate


def _advance(clock, seconds):
    clock["mono"] += seconds
    clock["wall"] += seconds


def _contender(directory, barrier, released, observations):
    permit = None
    try:
        gate = ProviderBackoff(Path(directory), wall_clock=lambda: 10000.0,
            monotonic=lambda: 10000.0, jitter=lambda: 0, boot_id="review-boot")
        barrier.wait(timeout=20)
        # Contended state locks may conservatively defer. Neither contention
        # nor an expired deadline permits a second active recovery probe.
        try:
            permit = gate.acquire(PROVIDER)
            observations.put((permit.allowed, permit.reason))
        except BackoffStateError as error:
            observations.put((False, str(error)))
        if permit is not None and permit.allowed:
            released.wait(timeout=20)
    except BaseException as error:
        observations.put((None, type(error).__name__))
    finally:
        if permit is not None:
            permit.close()


def test_32_processes_admit_one_recovery_and_abandonment_requires_review(clock_gate):
    clock, gate = clock_gate
    first = gate.acquire(PROVIDER)
    gate.settle(first, stop=_stop())
    first.close()
    _advance(clock, 9900)
    context = multiprocessing.get_context("forkserver")
    barrier = context.Barrier(32)
    released = context.Event()
    observations = context.Queue()
    children = [context.Process(target=_contender,
        args=(str(gate.directory), barrier, released, observations)) for _ in range(32)]
    outcomes = []
    try:
        for child in children:
            child.start()
        for _ in children:
            outcomes.append(observations.get(timeout=25))
        assert all(allowed is not None for allowed, _ in outcomes), outcomes
        assert sum(allowed for allowed, _ in outcomes) == 1, outcomes
        assert gate.acquire(PROVIDER).allowed is False
    except queue.Empty:
        pytest.fail("bounded concurrent admission probes did not settle")
    finally:
        released.set()
        for child in children:
            if child.pid is not None:
                child.join(timeout=5)
                if child.is_alive():
                    child.terminate()
                    child.join(timeout=5)
        observations.close()
        observations.join_thread()
    assert all(child.exitcode == 0 for child in children)
    # The OS lock is released on process exit, but that is not evidence the
    # already admitted native invocation had no side effects.
    stopped = gate.acquire(PROVIDER)
    assert stopped.allowed is False
    assert stopped.reason == "unsettled_recovery_probe"


@pytest.mark.parametrize("reason,unbounded", [
    ("billing", False), ("authentication", False), ("quota", False),
    ("rate_limit", True),
])
def test_older_transient_result_cannot_weaken_newer_manual_block(clock_gate, reason, unbounded):
    clock, gate = clock_gate
    older, newer = gate.acquire(PROVIDER), gate.acquire(PROVIDER)
    strong = gate.settle(newer, stop=_stop(reason, unbounded=unbounded))
    later = gate.settle(older, stop=_stop(seconds=1))
    assert strong["blocked"] is later["blocked"] is True
    assert later["reason"] == strong["reason"]
    _advance(clock, 8 * 86400)
    assert gate.acquire(PROVIDER).allowed is False
    gate.reset(PROVIDER)
    assert gate.acquire(PROVIDER).allowed is True


def test_stale_success_does_not_clear_newer_stop_and_deadline_only_extends(clock_gate):
    clock, gate = clock_gate
    old, initial = gate.acquire(PROVIDER), gate.acquire(PROVIDER)
    gate.settle(initial, stop=_stop(seconds=300))
    _advance(clock, 300)
    probe = gate.acquire(PROVIDER)
    assert probe.allowed and probe.recovery_probe
    try:
        gate.settle(old, stop=_stop(seconds=500))
        pending = gate.settle(probe, provider_succeeded=True)
        assert pending["recovery_required"] is True
        assert pending["retry_after_seconds"] == 500
        assert gate.acquire(PROVIDER).allowed is False
    finally:
        probe.close()
    _advance(clock, 500)
    next_probe = gate.acquire(PROVIDER)
    assert next_probe.allowed and next_probe.recovery_probe
    try:
        assert gate.settle(next_probe, provider_succeeded=True)["recovery_required"] is False
    finally:
        next_probe.close()


@pytest.mark.parametrize("success", [False, None, 1, "true", {}, []])
def test_unknown_or_nonboolean_recovery_success_never_reopens(clock_gate, success):
    clock, gate = clock_gate
    gate.settle(gate.acquire(PROVIDER), stop=_stop())
    _advance(clock, 60)
    probe = gate.acquire(PROVIDER)
    try:
        observed = gate.settle(probe, provider_succeeded=success)
        assert observed["blocked"] is True
        assert observed["reason"] == "recovery_unverified"
    finally:
        probe.close()
    assert gate.acquire(PROVIDER).allowed is False


def test_wait_rechecks_a_new_manual_block_without_holding_state_lock(clock_gate):
    clock, gate = clock_gate
    old = gate.acquire(PROVIDER)
    gate.settle(gate.acquire(PROVIDER), stop=_stop())
    _advance(clock, 59.5)
    waits = []
    def sleeper(seconds):
        waits.append(seconds)
        gate.settle(old, stop=_stop("authentication"))
        _advance(clock, seconds)
    denied = gate.acquire_with_wait(PROVIDER, max_wait_seconds=2, sleeper=sleeper)
    assert waits == [0.5]
    assert denied.allowed is False and denied.reason == "authentication"


def test_cancelled_wait_creates_no_probe_and_same_boot_ignores_wall_jump(clock_gate):
    clock, gate = clock_gate
    gate.settle(gate.acquire(PROVIDER), stop=_stop())
    clock["wall"] += 10**8
    assert gate.acquire(PROVIDER).retry_after_seconds == 60
    _advance(clock, 59)
    cancelled = threading.Event()
    def sleeper(seconds):
        _advance(clock, seconds)
        cancelled.set()
    denied = gate.acquire_with_wait(PROVIDER, max_wait_seconds=2,
        sleeper=sleeper, cancel_event=cancelled)
    assert denied.allowed is False and denied.reason == "cancelled_before_dispatch"
    state = json.loads((gate.directory / (PROVIDER + ".json")).read_text())
    assert state["probe_pending"] is False


@pytest.mark.parametrize("mutation", [
    lambda state: state.update(generation=True),
    lambda state: state.update(not_before_monotonic=float("inf")),
    lambda state: state.update(probe_pending="false"),
    lambda state: state.update(reason=[]),
    lambda state: state.update(provider="grok_cli"),
    lambda state: state.update(extra="unexpected"),
])
def test_corrupt_state_fails_closed_without_rewriting_evidence(clock_gate, mutation):
    _, gate = clock_gate
    gate.settle(gate.acquire(PROVIDER), stop=_stop())
    path = gate.directory / (PROVIDER + ".json")
    state = json.loads(path.read_text())
    mutation(state)
    path.write_text(json.dumps(state))
    original = path.read_bytes()
    with pytest.raises(BackoffStateError):
        gate.acquire(PROVIDER)
    assert path.read_bytes() == original


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "public_mode", "oversized", "duplicate"])
def test_unsafe_state_file_cannot_grant_dispatch(clock_gate, tmp_path, kind):
    _, gate = clock_gate
    gate.settle(gate.acquire(PROVIDER), stop=_stop())
    path = gate.directory / (PROVIDER + ".json")
    if kind in {"symlink", "hardlink"}:
        moved = tmp_path / "retained-state.json"
        path.rename(moved)
        if kind == "symlink":
            path.symlink_to(moved)
        else:
            path.hardlink_to(moved)
    elif kind == "public_mode":
        path.chmod(0o644)
    elif kind == "oversized":
        path.write_text("x" * 8193)
    else:
        text = path.read_text()
        path.write_text(text.rstrip()[:-1] + ',"blocked":false}')
    with pytest.raises((BackoffStateError, OSError)):
        gate.acquire(PROVIDER)


def test_settlement_is_one_use(clock_gate):
    _, gate = clock_gate
    first = gate.acquire(PROVIDER)
    gate.settle(first, stop=_stop())
    with pytest.raises(BackoffStateError):
        gate.settle(first, provider_succeeded=True)


def test_closed_probe_cannot_claim_success(clock_gate):
    clock, gate = clock_gate
    first = gate.acquire(PROVIDER)
    gate.settle(first, stop=_stop())
    _advance(clock, 60)
    probe = gate.acquire(PROVIDER)
    assert probe.allowed and probe.recovery_probe
    probe.close()
    with pytest.raises(BackoffStateError):
        gate.settle(probe, provider_succeeded=True)
    assert gate.acquire(PROVIDER).allowed is False


def test_failed_state_commit_blocks_fresh_gate_and_failed_reset_preserves_marker(clock_gate, monkeypatch):
    _, gate = clock_gate
    permit = gate.acquire(PROVIDER)
    def fail_write(*args):
        raise OSError("authored state commit failure")
    with monkeypatch.context() as patch:
        patch.setattr(gate, "_write", fail_write)
        with pytest.raises(OSError, match="authored state commit failure"):
            gate.settle(permit, stop=_stop())
    fresh = ProviderBackoff(gate.directory, boot_id="review-boot")
    denied = fresh.acquire(PROVIDER)
    assert denied.allowed is False and denied.reason == "settlement_unavailable"
    with monkeypatch.context() as patch:
        patch.setattr(fresh, "_write", fail_write)
        with pytest.raises(OSError, match="authored state commit failure"):
            fresh.reset(PROVIDER)
    assert fresh.acquire(PROVIDER).allowed is False
    fresh.reset(PROVIDER)
    assert fresh.acquire(PROVIDER).allowed is True


def test_contended_settlement_has_bounded_wait_and_persists_independent_blocker(clock_gate):
    _, gate = clock_gate
    permit = gate.acquire(PROVIDER)
    fd = os.open(gate.directory / (PROVIDER + ".lock"), os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    try:
        with pytest.raises(BackoffStateError, match="state_busy"):
            gate.settle(permit, stop=_stop("billing"))
        assert time.monotonic() - started < 5
    finally:
        os.close(fd)
    denied = gate.acquire(PROVIDER)
    assert denied.allowed is False and denied.reason == "settlement_unavailable"


@pytest.mark.parametrize("dispatched", [False, True])
def test_execute_collection_failure_preserves_unknown_postdispatch_state(tmp_path, monkeypatch, dispatched):
    from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
    output = tmp_path / "trial"
    output.mkdir()
    (output / "preparation.json").write_text("{}")
    launched = []
    failure = RuntimeError("authored collection unavailable")
    def execute_once(path, **kwargs):
        if dispatched:
            (path / "invocation.json").write_text('{"authored_invocation":true}')
            launched.append("one native attempt")
        raise failure
    monkeypatch.setattr(benchmark, "_execute_once", execute_once)
    directory = tmp_path / "gate"
    with pytest.raises(RuntimeError) as raised:
        benchmark.execute(output, provider_backoff_directory=directory)
    assert raised.value is failure
    assert len(launched) == int(dispatched)
    next_permit = ProviderBackoff(directory).acquire(PROVIDER)
    assert next_permit.allowed is (not dispatched)
    assert (output / "invocation.json").exists() is dispatched
    assert not (output / "receipt.json").exists()


def test_concurrent_failed_settlement_marker_survives_operator_reset(clock_gate, monkeypatch):
    _, reset_gate = clock_gate
    fallback_gate = ProviderBackoff(reset_gate.directory, boot_id="review-boot")
    failed_permit = fallback_gate.acquire(PROVIDER)
    reset_gate.settle(reset_gate.acquire(PROVIDER), stop=_stop())
    reset_paused = threading.Event()
    reset_continue = threading.Event()
    failure_started = threading.Event()
    marker_created = threading.Event()
    outcomes = []
    original_write = reset_gate._write
    original_open = fallback_gate._open
    def paused_write(provider, value):
        original_write(provider, value)
        reset_paused.set()
        if not reset_continue.wait(timeout=5):
            raise AssertionError("reset race fixture deadline")
    def failed_settle(*args, **kwargs):
        failure_started.set()
        raise OSError("authored failed settlement")
    def observed_open(path, flags):
        fd = original_open(path, flags)
        if path.suffix == ".unsettled":
            marker_created.set()
        return fd
    monkeypatch.setattr(reset_gate, "_write", paused_write)
    monkeypatch.setattr(fallback_gate, "_settle", failed_settle)
    monkeypatch.setattr(fallback_gate, "_open", observed_open)
    def reset():
        try:
            reset_gate.reset(PROVIDER)
            outcomes.append("reset")
        except BaseException as error:
            outcomes.append(type(error).__name__)
    def fail():
        try:
            fallback_gate.settle(failed_permit, stop=_stop("billing"))
        except OSError:
            outcomes.append("expected settlement error")
        except BaseException as error:
            outcomes.append(type(error).__name__)
    reset_thread = threading.Thread(target=reset)
    failure_thread = threading.Thread(target=fail)
    reset_thread.start()
    try:
        assert reset_paused.wait(timeout=5)
        failure_thread.start()
        assert failure_started.wait(timeout=5)
        # An uncoordinated marker writer finishes here and would be erased by
        # reset. A serialized writer waits until reset releases its custody.
        marker_created.wait(timeout=0.25)
    finally:
        reset_continue.set()
        reset_thread.join(timeout=5)
        if failure_thread.ident is not None:
            failure_thread.join(timeout=5)
    assert not reset_thread.is_alive() and not failure_thread.is_alive()
    assert sorted(outcomes) == ["expected settlement error", "reset"]
    assert marker_created.is_set()
    assert reset_gate.acquire(PROVIDER).allowed is False
