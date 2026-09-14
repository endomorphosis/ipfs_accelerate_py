"""Observation continues across normal returns without replaying failures."""

import threading
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.retained_observation import (
    run_retained_observation,
)
from ipfs_accelerate_py.agent_supervisor.runtime import retained_observation


def test_normal_return_retains_same_observer_thread_until_native_stop():
    stop, failed = threading.Event(), threading.Event()
    first = threading.Event()
    third = threading.Event()
    calls, errors = [], []

    def observe():
        calls.append(threading.get_ident())
        first.set()
        if len(calls) == 3:
            third.set()
            stop.set()

    thread = threading.Thread(target=run_retained_observation, kwargs={
        "observe_once": observe, "stop": stop, "failed": failed,
        "interval": 0.001, "on_error": errors.append,
    })
    thread.start()
    try:
        assert first.wait(5)
        assert third.wait(5)
    finally:
        stop.set()
        thread.join(5)
    assert not thread.is_alive()
    assert len(calls) == 3
    assert len(set(calls)) == 1
    assert not errors


@pytest.mark.parametrize("boundary", ["before_wait", "during_wait", "sample"])
def test_failure_event_prevents_further_observation(boundary):
    failed = threading.Event()
    calls = []
    if boundary == "before_wait":
        failed.set()

    def wait(_):
        if boundary == "during_wait":
            failed.set()
        return False

    def observe():
        calls.append("sample")
        failed.set()

    run_retained_observation(
        observe, failed=failed, interval=1,
        stop=SimpleNamespace(wait=wait, is_set=lambda: False),
        on_error=lambda exc: pytest.fail(str(exc)),
    )
    assert calls == (["sample"] if boundary == "sample" else [])


def test_observation_exception_is_reported_once_without_retry():
    error = RuntimeError("binding rejected")
    calls, errors = [], []

    def observe():
        calls.append("sample")
        raise error

    run_retained_observation(
        observe, failed=threading.Event(), interval=1,
        stop=SimpleNamespace(wait=lambda _: False, is_set=lambda: False),
        on_error=errors.append,
    )
    assert calls == ["sample"]
    assert errors == [error]


def test_stopped_owner_never_samples():
    stop = threading.Event()
    stop.set()
    run_retained_observation(
        lambda: pytest.fail("sampled stopped owner"), stop=stop,
        failed=threading.Event(), interval=1,
        on_error=lambda exc: pytest.fail(str(exc)),
    )


@pytest.mark.parametrize("interval", [0, -1, float("inf"), float("nan")])
def test_invalid_interval_cannot_create_busy_retry_loop(interval):
    with pytest.raises(ValueError, match="finite and positive"):
        run_retained_observation(
            lambda: pytest.fail("sampled"), stop=threading.Event(),
            failed=threading.Event(), interval=interval, on_error=lambda exc: None,
        )


@pytest.mark.parametrize("durations,expected_starts", [
    ([4, 4, 4], [10, 20, 30]),
    ([13, 13, 13], [10, 23, 36]),
    ([35, 1, 1], [10, 45, 55]),
])
def test_sample_work_consumes_interval_without_overlap_or_catchup(
    monkeypatch, durations, expected_starts,
):
    now = [0.0]
    starts, waits = [], []
    failed = threading.Event()
    monkeypatch.setattr(retained_observation, "monotonic", lambda: now[0])

    def wait(delay):
        assert delay >= 0
        waits.append(delay)
        now[0] += delay
        return False

    def observe():
        starts.append(now[0])
        now[0] += durations[len(starts) - 1]
        if len(starts) == len(durations):
            failed.set()

    run_retained_observation(
        observe, stop=SimpleNamespace(wait=wait, is_set=lambda: False),
        failed=failed, interval=10, on_error=lambda exc: pytest.fail(str(exc)),
    )
    assert starts == expected_starts
    assert len(waits) == len(durations)


@pytest.mark.parametrize("boundary", ["stop", "failed", "exception"])
def test_overrunning_sample_preserves_shutdown_and_no_retry(monkeypatch, boundary):
    now = [0.0]
    stopped = [False]
    failed = threading.Event()
    calls, errors = [], []
    monkeypatch.setattr(retained_observation, "monotonic", lambda: now[0])

    def observe():
        calls.append("one sample")
        now[0] += 100
        if boundary == "exception":
            raise RuntimeError("unverified owner")
        if boundary == "stop":
            stopped[0] = True
        else:
            failed.set()

    run_retained_observation(
        observe, failed=failed, interval=10, on_error=errors.append,
        stop=SimpleNamespace(wait=lambda _: stopped[0], is_set=lambda: stopped[0]),
    )
    assert calls == ["one sample"]
    assert len(errors) == (1 if boundary == "exception" else 0)
