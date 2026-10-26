"""Real capture/read/edit lifecycle; only the native index is authored."""
from types import SimpleNamespace
import threading

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_index as index
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_session as sessions
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_view as views
from ipfs_accelerate_py.agent_supervisor.runtime.code_source_view import CodeSourceView
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
    LeaseCancelledError, LeaseTimeoutError,
)
from test.api.test_repository_source_session import RecordingIndex, digest, source


def controlled(source, tmp_path, *, signal=None, deadline=None, index_class=RecordingIndex):
    def factory(path, **controls):
        assert controls == {key: value for key, value in (
            ("cancel_event", signal), ("deadline_monotonic", deadline)) if value is not None}
        return index_class(path)
    return sessions.RepositorySourceSession(source, tmp_path / "state", index_factory=factory,
        cancel_event=signal, deadline_monotonic=deadline)


@pytest.mark.parametrize("mode", ["cancel", "deadline"])
def test_stopped_session_cannot_enter_create_state_or_invoke_factory(tmp_path, mode, monkeypatch):
    signal = threading.Event()
    monkeypatch.setattr(index, "time", SimpleNamespace(monotonic=lambda: 50.0))
    if mode == "cancel": signal.set()
    with pytest.raises(LeaseCancelledError if mode == "cancel" else LeaseTimeoutError):
        controlled(tmp_path / "missing", tmp_path, signal=signal,
                   deadline=49.0 if mode == "deadline" else None)
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize("stage", ["create_repository_view", "load_repository_view", "restore_repository_view", "native_refresh"])
def test_cancelled_initial_capture_removes_candidate_and_closes_index(source, tmp_path, monkeypatch, stage):
    signal, owners = threading.Event(), []
    class Owner(RecordingIndex):
        def __init__(self, path):
            super().__init__(path)
            owners.append(self)
        def refresh(self, *args):
            receipt = super().refresh(*args)
            if stage == "native_refresh": signal.set()
            return receipt
    if stage != "native_refresh":
        original = getattr(views, stage)
        def stopped(*args, **kwargs):
            result = original(*args, **kwargs)
            signal.set()
            return result
        monkeypatch.setattr(views, stage, stopped)
    with pytest.raises(LeaseCancelledError):
        controlled(source, tmp_path, signal=signal, index_class=Owner)
    assert owners[0].closed
    assert not list((tmp_path / "state").glob("generation-*"))


@pytest.mark.parametrize("mode", ["cancel", "deadline"])
@pytest.mark.parametrize("operation", ["status", "read", "refresh"])
def test_public_operations_refuse_stopped_controls_without_returning_old_source(source, tmp_path, monkeypatch, mode, operation):
    signal, clock = threading.Event(), [1.0]
    monkeypatch.setattr(index, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    session = controlled(source, tmp_path, signal=signal, deadline=10.0)
    before = len(session._index.calls)
    if mode == "cancel": signal.set()
    else: clock[0] = 10.0
    try:
        with pytest.raises(LeaseCancelledError if mode == "cancel" else LeaseTimeoutError):
            if operation == "status": session.status()
            elif operation == "read": session.read(generation=1, path="calc.py")
            else: session.refresh(generation=1)
        assert len(session._index.calls) == before
        assert session.generation == 1
    finally:
        session.close()
    assert session._index.closed


def test_cancellation_during_retained_read_does_not_deliver_content(source, tmp_path, monkeypatch):
    signal = threading.Event()
    with controlled(source, tmp_path, signal=signal) as session:
        original = views.read_repository_file
        def stopped(*args, **kwargs):
            result = original(*args, **kwargs)
            signal.set()
            return result
        monkeypatch.setattr(views, "read_repository_file", stopped)
        with pytest.raises(LeaseCancelledError):
            session.read(generation=1, path="calc.py")


def edit(session, before):
    return session.replace_symbol(generation=1, path="calc.py", symbol="add",
        expected_sha256=digest(before), replacement="def add(value):\n    return value + 77\n")


def test_stop_after_preflight_never_calls_canonical_editor(source, tmp_path, monkeypatch):
    signal = threading.Event()
    before = (source / "calc.py").read_bytes()
    with controlled(source, tmp_path, signal=signal) as session:
        original = session._preflight_edit
        def stopped(*args):
            result = original(*args)
            signal.set()
            return result
        monkeypatch.setattr(session, "_preflight_edit", stopped)
        with pytest.raises(LeaseCancelledError): edit(session, before)
        assert (source / "calc.py").read_bytes() == before
        assert len(session._index.calls) == 1 and session.generation == 1
        assert not list(session.state_dir.glob("proposal-*"))


@pytest.mark.parametrize("failure", [False, True])
def test_stop_after_atomic_edit_retains_observed_or_uncertain_mutation(source, tmp_path, monkeypatch, failure):
    signal = threading.Event()
    before = (source / "calc.py").read_bytes()
    with controlled(source, tmp_path, signal=signal) as session:
        original = CodeSourceView.replace
        def stopped(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            if self.root == source:
                signal.set()
                if failure: raise OSError("authored postcommit failure")
            return result
        monkeypatch.setattr(CodeSourceView, "replace", stopped)
        scan = views.assert_repository_current
        def no_late_scan(*args):
            assert not signal.is_set(), "no full-tree scan after stopped postcommit control"
            return scan(*args)
        monkeypatch.setattr(views, "assert_repository_current", no_late_scan)
        result = edit(session, before)
        assert (source / "calc.py").read_bytes() != before
        assert result["refresh_required"] is True
        assert result["canonical_changed"] is (None if failure else True)
        assert result["canonical_change"] == ("uncertain" if failure else "observed")
        assert session.generation == 1 and len(session._index.calls) == 1
        assert not list(session.state_dir.glob("proposal-*"))
        assert len(list(session.state_dir.glob("generation-*"))) == 1


def test_deadline_after_native_refresh_refuses_new_generation(source, tmp_path, monkeypatch):
    clock = [1.0]
    monkeypatch.setattr(index, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    with controlled(source, tmp_path, deadline=10.0) as session:
        original = session._index.refresh
        def expired(*args):
            result = original(*args)
            clock[0] = 10.0
            return result
        monkeypatch.setattr(session._index, "refresh", expired)
        with pytest.raises(LeaseTimeoutError): session.refresh(generation=1)
        assert session.generation == 1 and session._refresh_required
        assert len(list(session.state_dir.glob("generation-*"))) == 1


def test_editor_checks_controls_after_preparation_before_atomic_replace(source, monkeypatch):
    before = (source / "calc.py").read_bytes()
    calls = []
    def checkpoint():
        calls.append(1)
        if len(calls) == 4: raise LeaseCancelledError("authored precommit stop")
    with pytest.raises(LeaseCancelledError):
        CodeSourceView(source).replace("calc.py", "add", digest(before),
            "def add(value):\n    return value + 77\n", checkpoint=checkpoint)
    assert len(calls) == 4
    assert (source / "calc.py").read_bytes() == before
    assert not list(source.glob(".source-view-*.tmp"))
