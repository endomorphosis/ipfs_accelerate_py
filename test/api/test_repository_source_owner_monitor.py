"""Synchronous retention: authored receipts and one admitted native handler.

Authored owner/type/timing seams establish monitor control flow only. The
native case borrows the enclosing admitted runner's real native parent and
uses the existing lifecycle/broker fixture. Its container boundary and peer
identity remain authored; this is not separate-UID deployment qualification.
"""
import os
import socket
import threading
import time
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_monitor as monitor
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_broker as broker
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_protocol as protocol
from test.api.test_repository_source_owner_broker import message
from test.api.test_repository_source_owner_lifecycle import native_owner, native_parent, short_service_directory


def closed(*, acquired=True):
    return dict(schema=monitor.SCHEMA, state="closed", ready=False,
        cleanup_complete=True, cleanup_pending=False, observation_pending=False,
        admission_uncertain=False, retained_custody=False, cleanup_error=None,
        native_parent_acquired=acquired, native_parent_released=True if acquired else None,
        native_parent={"owner_pid": os.getpid()} if acquired else None)


def pending():
    return {**closed(), "state": "cleanup_pending", "cleanup_complete": False,
            "cleanup_pending": True, "retained_custody": True, "native_parent_released": False}


@pytest.fixture
def authored(monkeypatch):
    """Finite scripted outcomes; no scheduler or native capability is created."""
    state = SimpleNamespace(now=10., sleeps=[], calls=[], outcomes=[], final=closed(), sleep_errors=[])

    class Owner:
        def __init__(self):
            self._owner_pid = os.getpid()
        def close(self, *, timeout_seconds):
            state.calls.append(timeout_seconds)
            value = state.outcomes.pop(0) if state.outcomes else state.final
            if isinstance(value, BaseException):
                raise value
            return value

    def sleep(seconds):
        state.sleeps.append(seconds)
        if state.sleep_errors:
            error, elapsed = state.sleep_errors.pop(0)
            state.now += elapsed
            raise error
        state.now += seconds

    monkeypatch.setattr(monitor, "RepositorySourceOwnerLifecycle", Owner)
    monkeypatch.setattr(monitor, "_monotonic", lambda: state.now)
    monkeypatch.setattr(monitor, "_sleep", sleep)
    state.Owner, state.owner = Owner, Owner()
    return state


@pytest.mark.parametrize("acquired", [True, False])
def test_known_complete_receipt_returns_without_waiting(authored, acquired):
    authored.final = closed(acquired=acquired)
    assert monitor.retire_repository_source_owner(authored.owner) is authored.final
    assert authored.calls == [1.0] and authored.sleeps == []


def test_pending_retains_same_owner_and_uses_capped_backoff(authored):
    authored.outcomes = [pending() for _ in range(8)]
    result = monitor.retire_repository_source_owner(authored.owner)
    assert result is authored.final
    assert authored.calls == [1.0] * 9
    assert authored.sleeps == pytest.approx([.1, .2, .4, .8, 1., 1., 1., 1.])
    assert authored.now == pytest.approx(15.5)


@pytest.mark.parametrize("error", [RuntimeError("private context"), KeyboardInterrupt(), SystemExit(7)])
def test_cleanup_exception_or_signal_cannot_exit_before_later_completion(authored, error, capsys):
    authored.outcomes = [error, pending()]
    assert monitor.retire_repository_source_owner(authored.owner) is authored.final
    assert authored.calls == [1.0] * 3
    assert authored.sleeps == pytest.approx([.1, .2])
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize("error", [RuntimeError("private sleep"), KeyboardInterrupt(), SystemExit(7)])
def test_interrupted_backoff_finishes_remaining_pause_before_next_close(authored, error):
    authored.outcomes = [pending()]
    authored.sleep_errors = [(error, .025)]
    assert monitor.retire_repository_source_owner(authored.owner) is authored.final
    assert authored.calls == [1.0, 1.0]
    assert authored.sleeps == pytest.approx([.1, .075])
    assert authored.now == pytest.approx(10.1)


@pytest.mark.parametrize("field,value", [
    ("schema", "different@1"), ("state", "active"), ("ready", True), ("ready", 0),
    ("cleanup_complete", False), ("cleanup_complete", 1), ("cleanup_pending", True),
    ("cleanup_pending", 0), ("observation_pending", True), ("observation_pending", None),
    ("admission_uncertain", True), ("admission_uncertain", None),
    ("retained_custody", True), ("retained_custody", 0), ("cleanup_error", "failed"),
    ("native_parent_acquired", None), ("native_parent_acquired", 1),
    ("native_parent_released", False), ("native_parent_released", None),
    ("native_parent_released", 1),
])
def test_false_or_unknown_completion_facts_do_not_exit(authored, field, value):
    authored.outcomes = [{**closed(), field: value}]
    assert monitor.retire_repository_source_owner(authored.owner) is authored.final
    assert authored.calls == [1.0, 1.0] and authored.sleeps == pytest.approx([.1])


@pytest.mark.parametrize("field", [
    "schema", "state", "ready", "cleanup_complete", "cleanup_pending", "observation_pending",
    "admission_uncertain", "retained_custody", "cleanup_error", "native_parent_acquired",
    "native_parent_released",
])
def test_missing_completion_fact_does_not_exit(authored, field):
    receipt = closed()
    del receipt[field]
    authored.outcomes = [receipt]
    assert monitor.retire_repository_source_owner(authored.owner) is authored.final
    assert len(authored.calls) == 2


@pytest.mark.parametrize("malformed", [None, False, "closed", [], 0])
def test_malformed_receipt_does_not_exit(authored, malformed):
    authored.outcomes = [malformed]
    assert monitor.retire_repository_source_owner(authored.owner) is authored.final
    assert len(authored.calls) == 2


@pytest.mark.parametrize("change", [
    {"admission_uncertain": True}, {"native_parent_released": False},
    {"native_parent_released": True}, {"native_parent": {}},
])
def test_claimed_no_admission_requires_unambiguous_absence(authored, change):
    authored.outcomes = [{**closed(acquired=False), **change}]
    assert monitor.retire_repository_source_owner(authored.owner) is authored.final
    assert len(authored.calls) == 2


@pytest.mark.parametrize("field", ["native_parent", "native_parent_released"])
def test_no_admission_requires_explicit_null_fields(authored, field):
    receipt = closed(acquired=False)
    del receipt[field]
    authored.outcomes = [receipt]
    assert monitor.retire_repository_source_owner(authored.owner) is authored.final
    assert len(authored.calls) == 2


def test_dict_subclass_cannot_supply_completion(authored):
    class Receipt(dict):
        pass
    authored.outcomes = [Receipt(closed())]
    assert monitor.retire_repository_source_owner(authored.owner) is authored.final
    assert len(authored.calls) == 2


def test_wrong_owner_and_subclass_are_refused_without_close(authored):
    class Other(authored.Owner):
        pass
    for value in (None, object(), Other()):
        with pytest.raises(ValueError, match="exact_repository_source_owner_required"):
            monitor.retire_repository_source_owner(value)
    assert authored.calls == authored.sleeps == []


@pytest.mark.parametrize("owner_pid", [-1, None, True, "123"])
def test_foreign_or_invalid_process_owner_is_refused_without_close(authored, owner_pid):
    authored.owner._owner_pid = owner_pid
    with pytest.raises(ValueError, match="belongs_to_another_process"):
        monitor.retire_repository_source_owner(authored.owner)
    assert authored.calls == authored.sleeps == []


def test_only_existing_cleanup_is_called(authored):
    # The authored owner deliberately has no start/note_spawned/scheduler API.
    authored.outcomes = [RuntimeError("cleanup failed"), pending(), None]
    assert monitor.retire_repository_source_owner(authored.owner) is authored.final
    assert authored.calls == [1.0] * 4


def test_retention_is_attempted_once_after_closed_receipt_and_preserves_identity(authored, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_retirement as retention
    authored.outcomes = [pending()]
    calls = []
    retained = dict(schema=monitor.RETENTION_SCHEMA, status="retained", reason=None,
                    name="authored.json", sha256="a" * 64, bytes=42)
    def retain(owner):
        assert owner is authored.owner and authored.calls == [1.0, 1.0]
        calls.append(owner)
        return retained
    monkeypatch.setattr(retention, "retain_repository_source_owner_retirement", retain)
    result = monitor.retire_repository_source_owner(authored.owner)
    assert result is authored.final and result["retained_lifecycle_receipt"] is retained
    assert calls == [authored.owner] and authored.sleeps == pytest.approx([.1])


@pytest.mark.parametrize("error", [OSError("private full disk"), KeyboardInterrupt(), SystemExit(7)])
def test_retention_failure_cannot_retry_cleanup_or_mask_closed_disposition(authored, monkeypatch, error, capsys):
    from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_retirement as retention
    calls = []
    def retain(owner):
        calls.append(owner)
        raise error
    monkeypatch.setattr(retention, "retain_repository_source_owner_retirement", retain)
    result = monitor.retire_repository_source_owner(authored.owner)
    assert result is authored.final and result["cleanup_complete"] is True
    assert result["native_parent_released"] is True
    assert result["retained_lifecycle_receipt"] == dict(schema=monitor.RETENTION_SCHEMA,
        status="unavailable", reason="retention_failed", name=None, sha256=None, bytes=None)
    assert calls == [authored.owner] and authored.calls == [1.0] and authored.sleeps == []
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_retention_import_failure_cannot_prevent_existing_cleanup(authored, monkeypatch):
    import builtins
    original = builtins.__import__
    def unavailable(name, *args, **kwargs):
        if name == "repository_source_owner_retirement":
            assert authored.calls == [1.0]
            raise ImportError("authored unavailable exporter")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", unavailable)
    result = monitor.retire_repository_source_owner(authored.owner)
    assert result is authored.final and result["cleanup_complete"] is True
    assert result["retained_lifecycle_receipt"]["status"] == "unavailable"
    assert authored.calls == [1.0] and authored.sleeps == []


def test_real_handler_custody_is_retained_until_cleanup(native_owner, native_parent, monkeypatch):
    owner = native_owner
    entered, finish = threading.Event(), threading.Event()
    original = broker.mcp.RepositoryMCPServer.handle
    observations, handler_timeouts = [], []

    def blocked(self, request):
        entered.set()
        if not finish.wait(10):
            handler_timeouts.append(True)
        return original(self, request)

    def release_from_backoff(seconds):
        if not observations:
            observations.append(dict(parent_released=owner._lease.released,
                child_released=owner._broker._native.released, caller_released=native_parent.released,
                caller_cancelled=native_parent.cancelled, stop_requested=owner._stop.is_set()))
            finish.set()
        time.sleep(min(seconds, .01))

    monkeypatch.setattr(broker.mcp.RepositoryMCPServer, "handle", blocked)
    monkeypatch.setattr(monitor, "_sleep", release_from_backoff)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.connect(str(owner._broker.socket_path))
        protocol.send_frame(channel,
            protocol.make_request(owner._broker.descriptor["session_id"], message("ping")),
            deadline_monotonic=time.monotonic() + 2)
        try:
            assert entered.wait(2)
            result = monitor.retire_repository_source_owner(owner)
        finally:
            finish.set()
    assert result["cleanup_complete"] is True and result["native_parent_released"] is True
    assert observations == [dict(parent_released=False, child_released=False,
        caller_released=False, caller_cancelled=False, stop_requested=True)]
    assert handler_timeouts == []
    assert owner._lease.released and owner._broker._native.released
    assert not native_parent.released and not native_parent.cancelled
