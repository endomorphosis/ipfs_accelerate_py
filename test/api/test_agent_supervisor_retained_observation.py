"""Observation continues across normal returns without replaying failures."""

import threading
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.retained_observation import (
    run_retained_observation,
)


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
