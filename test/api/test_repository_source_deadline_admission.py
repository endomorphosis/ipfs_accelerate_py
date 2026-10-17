"""Inherited monotonic deadlines; native cases borrow the admitted runner lease.

Local clock seams affect only adapter checkpoints, never the shared scheduler's
clock, policy or telemetry. Authored control objects below exercise arithmetic
without opening an index or representing an admitted native owner.
"""
from contextlib import contextmanager
import threading
import time
import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_index as indexes
from ipfs_accelerate_py.agent_supervisor.runtime import repository_context_compiler as adapters
from test.api.test_repository_source_parent_admission import native_parent


class Clock:
    def __init__(self, now=None):
        self.now = time.monotonic() if now is None else now

    def monotonic(self):
        return self.now


def _local_clock(monkeypatch, now=None):
    clock = Clock(now)
    monkeypatch.setattr(indexes, "time", clock)
    monkeypatch.setattr(adapters, "time", clock)
    return clock


@pytest.mark.parametrize("value", [True, False, "123", float("nan"), float("inf"),
    -float("inf"), object(), 10 ** 1000])
def test_invalid_deadline_refuses_before_native_state(tmp_path, value):
    state = tmp_path / "index"
    with pytest.raises(indexes.RepositorySourceIndexError, match="invalid_resource_deadline"):
        indexes.NativeRepositoryIndex(state, deadline_monotonic=value)
    assert not state.exists()


@pytest.mark.parametrize("value", [None, 123, 123.25, -1.0])
def test_deadline_shape_validation_preserves_exact_caller_value(value):
    assert indexes._validate_resource_deadline(value) is value


def test_expired_deadline_refuses_before_native_state(tmp_path, monkeypatch):
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseTimeoutError

    _local_clock(monkeypatch, now=123.0)
    with pytest.raises(LeaseTimeoutError):
        indexes.NativeRepositoryIndex(tmp_path / "index", deadline_monotonic=123.0)
    assert not (tmp_path / "index").exists()


def test_shared_checkpoint_prioritizes_cancellation_and_never_resets_deadline(monkeypatch):
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
        LeaseCancelledError, LeaseTimeoutError,
    )

    clock = _local_clock(monkeypatch, now=100.0)
    signal = threading.Event()
    indexes._check_resource_cancellation(signal, 101.0)
    clock.now = 101.0
    with pytest.raises(LeaseTimeoutError):
        indexes._check_resource_cancellation(signal, 101.0)
    signal.set()
    with pytest.raises(LeaseCancelledError):
        indexes._check_resource_cancellation(signal, 101.0)


@pytest.mark.parametrize("deadline,execution,admission", [(None, 30.0, 10.0),
    (105.0, 5.0, 5.0), (120.0, 20.0, 10.0)])
def test_authored_controls_bound_both_native_budgets_without_admission(
        monkeypatch, deadline, execution, admission):
    _local_clock(monkeypatch, now=100.0)
    authored = object.__new__(indexes.NativeRepositoryIndex)
    authored._scheduler = authored._parent_lease = authored._cancel_event = None
    authored._resource_cancellation = None
    authored._deadline_monotonic = deadline
    assert authored._controls(256) == dict(scheduler=None, parent_lease=None, cancel_event=None,
        timeout_seconds=execution, admission_timeout_seconds=admission, memory_mb=256)
    assert authored._controls(512, timeout_seconds=2.0, admission_timeout_seconds=1.0) == dict(
        scheduler=None, parent_lease=None, cancel_event=None, timeout_seconds=2.0,
        admission_timeout_seconds=1.0, memory_mb=512)
    assert "deadline_monotonic" not in authored._resource_options()


def _snapshot(tmp_path):
    from test.api.test_repository_source_index import snapshot

    _source, _package, original, _receipt, manifest = snapshot(tmp_path,
        files={"calc.py": "# Authored source, never imported.\ndef add(value):\n    return value + 1\n"})
    return original, manifest


def _watch_children(monkeypatch, parent):
    from ipfs_datasets_py.logic.software_contracts import codebase_resources

    children, calls = [], []
    acquire = codebase_resources.acquire_codebase_resources
    @contextmanager
    def observed(**kwargs):
        assert kwargs.get("scheduler") is None and kwargs["parent_lease"] is parent
        calls.append(dict(kwargs))
        with acquire(**kwargs) as lease:
            assert lease.parent_lease_id == parent.lease_id
            children.append(lease)
            yield lease
    monkeypatch.setattr(codebase_resources, "acquire_codebase_resources", observed)
    return children, calls


def test_native_startup_expiry_closes_opened_database_without_releasing_parent(
        tmp_path, native_parent, monkeypatch):
    import duckdb
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseTimeoutError

    clock = _local_clock(monkeypatch)
    deadline = clock.now + 120.0
    connect, opened = duckdb.connect, []
    def expired_connect(*args, **kwargs):
        connection = connect(*args, **kwargs)
        opened.append(connection)
        clock.now = deadline
        return connection
    monkeypatch.setattr(duckdb, "connect", expired_connect)
    with pytest.raises(LeaseTimeoutError):
        indexes.NativeRepositoryIndex(tmp_path / "index", parent_lease=native_parent,
                                      deadline_monotonic=deadline)
    assert len(opened) == 1
    with pytest.raises(duckdb.ConnectionException):
        opened[0].execute("SELECT 1")
    assert not native_parent.released and not native_parent.cancelled


@pytest.mark.parametrize("boundary", ["prepare", "join"])
def test_native_refresh_late_expiry_withholds_current_generation_and_stops_next_admission(
        tmp_path, native_parent, monkeypatch, boundary):
    clock = _local_clock(monkeypatch)
    deadline = clock.now + 120.0
    original, manifest = _snapshot(tmp_path)
    before = (original / "calc.py").read_bytes()
    children, calls = _watch_children(monkeypatch, native_parent)
    with indexes.NativeRepositoryIndex(tmp_path / "index", parent_lease=native_parent,
                                      deadline_monotonic=deadline) as owner:
        if boundary == "prepare":
            prepare = owner._index.prepare_current
            def expires(*args, **kwargs):
                result = prepare(*args, **kwargs)
                clock.now = deadline
                return result
            monkeypatch.setattr(owner._index, "prepare_current", expires)
        else:
            join = owner._join
            def expires(*args, **kwargs):
                result = join(*args, **kwargs)
                clock.now = deadline
                return result
            monkeypatch.setattr(owner, "_join", expires)
        with pytest.raises(indexes.RepositorySourceIndexError, match="native_index_refresh_failed"):
            owner.refresh(original, manifest)
        assert owner._current is None
        assert len(calls) == (1 if boundary == "prepare" else 2)
        assert children and all(child.released for child in children)
        assert (original / "calc.py").read_bytes() == before
    assert not native_parent.released and not native_parent.cancelled


def test_native_symbol_final_observation_expiry_withholds_rows(tmp_path, native_parent, monkeypatch):
    clock = _local_clock(monkeypatch)
    deadline = clock.now + 120.0
    original, manifest = _snapshot(tmp_path)
    with indexes.NativeRepositoryIndex(tmp_path / "index", parent_lease=native_parent,
                                      deadline_monotonic=deadline) as owner:
        owner.refresh(original, manifest)
        observe, observations = owner._index.observe_current, []
        def expires(*args, **kwargs):
            result = observe(*args, **kwargs)
            observations.append(kwargs)
            if len(observations) == 2:
                clock.now = deadline
            return result
        monkeypatch.setattr(owner._index, "observe_current", expires)
        with pytest.raises(indexes.RepositorySourceIndexError, match="native_index_query_failed"):
            owner.symbols(query="add")
        assert len(observations) == 2 and owner._current is None
    assert not native_parent.released and not native_parent.cancelled


def test_native_capsule_expiry_after_build_refuses_and_releases_only_child(
        tmp_path, native_parent, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_capsules import RepositoryCapsules

    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_SEMANTIC_TRUTH_AUTHORITY", "ipfs_datasets_py")
    clock = _local_clock(monkeypatch)
    deadline = clock.now + 120.0
    original, manifest = _snapshot(tmp_path)
    children, _calls = _watch_children(monkeypatch, native_parent)
    with indexes.NativeRepositoryIndex(tmp_path / "index", parent_lease=native_parent,
                                      deadline_monotonic=deadline) as owner:
        owner.refresh(original, manifest)
        query = RepositoryCapsules.query
        def expires(*args, **kwargs):
            result = query(*args, **kwargs)
            assert not children[-1].released
            clock.now = deadline
            return result
        monkeypatch.setattr(RepositoryCapsules, "query", expires)
        with pytest.raises(indexes.RepositorySourceIndexError, match="capsule_resources_unavailable"):
            owner.context_query("add")
        assert owner._capsules is None
        assert children and all(child.released for child in children)
    assert not native_parent.released and not native_parent.cancelled


@pytest.mark.parametrize("boundary", ["compile", "publication"])
def test_native_compiler_inherits_remaining_budget_and_rejects_late_result(
        tmp_path, native_parent, monkeypatch, boundary):
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_session import RepositorySourceSession
    from test.api.test_repository_context_compiler import compiler, compile_context, obligation
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseTimeoutError

    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_SEMANTIC_TRUTH_AUTHORITY", "ipfs_datasets_py")
    clock = _local_clock(monkeypatch)
    deadline = clock.now + 120.0
    source = tmp_path / "source"
    source.mkdir()
    (source / "calc.py").write_text("def add(value):\n    return value + 1\n")
    # Keep the session's real clock and all scheduler clocks untouched. The
    # native adapter's injected clock is advanced only at the compiler seam.
    with RepositorySourceSession(source, tmp_path / "session", parent_lease=native_parent,
                                 deadline_monotonic=deadline) as session:
        children, calls = _watch_children(monkeypatch, native_parent)
        owner = compiler()
        original_compile = owner.compile
        adapter = session.context_adapter(owner)
        original_adapter_compile = adapter._compile
        def reserve_last_two_seconds(*args, **kwargs):
            clock.now = deadline - 2.0
            return original_adapter_compile(*args, **kwargs)
        monkeypatch.setattr(adapter, "_compile", reserve_last_two_seconds)
        def expires(*args, **kwargs):
            result = original_compile(*args, **kwargs)
            assert not children[-1].released
            assert calls[-1]["memory_mb"] == 256 and calls[-1]["timeout_seconds"] == 2.0
            if boundary == "compile":
                clock.now = deadline
            return result
        monkeypatch.setattr(owner, "compile", expires)
        if boundary == "publication":
            save = adapter._save
            def expires_after_publication(*args, **kwargs):
                checkpoint = session._checkpoint
                count = 0
                def publication_checkpoint():
                    nonlocal count
                    count += 1
                    if count == 2:
                        clock.now = deadline
                    return checkpoint()
                with monkeypatch.context() as scoped:
                    scoped.setattr(session, "_checkpoint", publication_checkpoint)
                    return save(*args, **kwargs)
            monkeypatch.setattr(adapter, "_save", expires_after_publication)
        expected = adapters.RepositoryContextError if boundary == "compile" else LeaseTimeoutError
        with pytest.raises(expected):
            compile_context(adapter, (obligation("intent_ir"),), query="add")
        assert adapter._parents == {}
        assert children and all(child.released for child in children)
    assert not native_parent.released and not native_parent.cancelled
