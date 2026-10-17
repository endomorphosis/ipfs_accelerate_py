"""Borrowed native admission controls; native smoke requires an injected lease.

The plugin-provided ``admitted_native_parent`` must be an actual live datasets
lease obtained by the enclosing bounded runner. No test creates a scheduler or
changes telemetry/policy. Without that fixture native cases explicitly skip.
Authored factories below test only argument plumbing, never native admission.
"""
from contextlib import contextmanager
import json
import threading

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_index import (
    NativeRepositoryIndex, RepositorySourceIndexError,
)
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_session import RepositorySourceSession


@pytest.fixture
def native_parent(request):
    try:
        lease = request.getfixturevalue("admitted_native_parent")
    except pytest.FixtureLookupError:
        pytest.skip("native smoke requires an actual parent lease from the admitted runner")
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import ResourceLease

    assert isinstance(lease, ResourceLease)
    assert not lease.released and not lease.cancelled
    assert lease.memory_mb >= 512 and lease.cpu_slots >= 1 and lease.child_process_slots >= 1
    yield lease
    assert not lease.released, "consumer must not release the caller's parent"


@pytest.mark.parametrize("constructor", ["index", "session"])
@pytest.mark.parametrize("controls", [dict(parent_lease=object()), dict(scheduler=object()),
    dict(cancel_event=object()), dict(parent_lease=object(), scheduler=object())])
def test_invalid_native_controls_refuse_before_state_filesystem_changes(tmp_path, constructor, controls):
    state = tmp_path / "state"
    with pytest.raises(RepositorySourceIndexError, match="invalid_resource_controls"):
        if constructor == "index":
            NativeRepositoryIndex(state, **controls)
        else:
            RepositorySourceSession(tmp_path / "missing-source", state, **controls)
    assert not state.exists()


@pytest.mark.parametrize("constructor", ["index", "session"])
def test_already_cancelled_signal_refuses_before_state_filesystem_changes(tmp_path, constructor):
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseCancelledError

    signal = threading.Event()
    signal.set()
    state = tmp_path / "state"
    with pytest.raises(LeaseCancelledError):
        if constructor == "index":
            NativeRepositoryIndex(state, cancel_event=signal)
        else:
            RepositorySourceSession(tmp_path / "missing-source", state, cancel_event=signal)
    assert not state.exists()


def _no_capture_session(monkeypatch):
    # No native index, source capture, or admission is implied by this seam.
    monkeypatch.setattr(RepositorySourceSession, "_validate_locations", lambda self: None)
    monkeypatch.setattr(RepositorySourceSession, "_capture", lambda self: None)


def test_unconfigured_session_preserves_single_argument_custom_factory(tmp_path, monkeypatch):
    _no_capture_session(monkeypatch)
    calls = []
    class AuthoredIndex:
        def __init__(self, path):
            calls.append(path)
        def close(self):
            calls.append("closed")
    session = RepositorySourceSession(tmp_path / "source", tmp_path / "state", index_factory=AuthoredIndex)
    try:
        assert calls == [tmp_path / "state" / "index"]
    finally:
        session.close()
    assert calls[-1] == "closed"


def test_explicit_signal_is_forwarded_to_custom_factory_without_retry(tmp_path, monkeypatch):
    _no_capture_session(monkeypatch)
    signal = threading.Event()
    calls = []
    class AuthoredIndex:
        def __init__(self, path, **controls):
            calls.append(controls)
        def close(self):
            pass
    session = RepositorySourceSession(tmp_path / "source", tmp_path / "state",
                                      index_factory=AuthoredIndex, cancel_event=signal)
    try:
        assert calls == [{"cancel_event": signal}]
    finally:
        session.close()
    assert not signal.is_set()
    calls.clear()
    def old_factory(path):
        calls.append(path)
        raise AssertionError("unsupported explicit controls must never be dropped")
    with pytest.raises(TypeError):
        RepositorySourceSession(tmp_path / "other-source", tmp_path / "other-state",
                                index_factory=old_factory, cancel_event=signal)
    assert calls == []


def _workspace(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "calc.py").write_text("# Authored source, never imported.\ndef add(value):\n    return value + 1\n")
    return root


def test_native_refresh_query_capsules_and_compiler_borrow_exact_parent(tmp_path, native_parent, monkeypatch):
    from ipfs_datasets_py.logic.software_contracts import codebase_resources
    from test.api.test_repository_context_compiler import compiler, compile_context, obligation

    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_SEMANTIC_TRUTH_AUTHORITY", "ipfs_datasets_py")
    source = _workspace(tmp_path)
    signal, admissions = threading.Event(), []
    acquire = codebase_resources.acquire_codebase_resources
    @contextmanager
    def observed(**kwargs):
        assert kwargs.get("scheduler") is None
        assert kwargs["parent_lease"] is native_parent and kwargs["cancel_event"] is signal
        with acquire(**kwargs) as lease:
            assert lease.parent_lease_id == native_parent.lease_id
            admissions.append((lease, kwargs["memory_mb"]))
            yield lease
    monkeypatch.setattr(codebase_resources, "acquire_codebase_resources", observed)
    with RepositorySourceSession(source, tmp_path / "session",
                                 parent_lease=native_parent, cancel_event=signal) as session:
        assert session._index._parent_lease is native_parent
        assert session.symbols(generation=1, query="add")["symbols"]
        query = session.context_query(generation=1, query="add")
        assert query["candidates"]
        result = compile_context(session.context_adapter(compiler()), (obligation("intent_ir"),), query="add")
        assert result.capsule.authority["proof_authority"] is False
        assert any(memory == 512 for _lease, memory in admissions)
        assert any(memory == 256 for _lease, memory in admissions)
        assert all(lease.released for lease, _memory in admissions)
    assert admissions and all(lease.released for lease, _memory in admissions)
    assert not native_parent.released and not native_parent.cancelled and not signal.is_set()


@pytest.mark.parametrize("disposition", ["cancel", "release", "external"])
def test_native_parent_liveness_refuses_query_and_close_keeps_borrowed_ownership(
        tmp_path, native_parent, disposition):
    from ipfs_datasets_py.logic.software_contracts.codebase_resources import acquire_codebase_resources
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseCancelledError

    source, signal = _workspace(tmp_path), threading.Event()
    with acquire_codebase_resources(parent_lease=native_parent, memory_mb=512, timeout_seconds=10) as owned_parent:
        session = RepositorySourceSession(source, tmp_path / "session",
                                          parent_lease=owned_parent, cancel_event=signal)
        if disposition == "external":
            signal.set()
        else:
            getattr(owned_parent, disposition)()
        released = owned_parent.released
        try:
            with pytest.raises(LeaseCancelledError):
                session._index.symbols(query="add")
            assert session._index._current is None
        finally:
            session.close()
        assert owned_parent.released is released


def test_cancelled_actual_parent_refuses_constructor_without_state(tmp_path, native_parent):
    from ipfs_datasets_py.logic.software_contracts.codebase_resources import acquire_codebase_resources
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseCancelledError

    with acquire_codebase_resources(parent_lease=native_parent, memory_mb=512, timeout_seconds=10) as owned_parent:
        owned_parent.cancel()
        with pytest.raises(LeaseCancelledError):
            NativeRepositoryIndex(tmp_path / "index", parent_lease=owned_parent)
        assert not (tmp_path / "index").exists()
        assert not owned_parent.released


def test_external_cancellation_after_capsule_build_is_fenced(tmp_path, native_parent, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_capsules import RepositoryCapsules

    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_SEMANTIC_TRUTH_AUTHORITY", "ipfs_datasets_py")
    signal = threading.Event()
    with RepositorySourceSession(_workspace(tmp_path), tmp_path / "session",
                                 parent_lease=native_parent, cancel_event=signal) as session:
        original = RepositoryCapsules.query
        def cancelled(*args, **kwargs):
            response = original(*args, **kwargs)
            signal.set()
            return response
        monkeypatch.setattr(RepositoryCapsules, "query", cancelled)
        with pytest.raises(RepositorySourceIndexError, match="capsule_resources_unavailable"):
            session._index.context_query("add")
        assert session._index._capsules is None
    assert not native_parent.released


def test_native_in_process_mcp_initialization_and_tools_borrow_parent(tmp_path, native_parent, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_mcp as mcp
    from ipfs_datasets_py.logic.software_contracts import codebase_resources
    from test.api.test_repository_source_mcp import configuration, initialization

    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_SEMANTIC_TRUTH_AUTHORITY", "ipfs_datasets_py")
    config, _path, _digest = configuration(tmp_path)
    acquire, children = codebase_resources.acquire_codebase_resources, []
    @contextmanager
    def observed(**kwargs):
        assert kwargs.get("scheduler") is None and kwargs["parent_lease"] is native_parent
        with acquire(**kwargs) as lease:
            assert lease.parent_lease_id == native_parent.lease_id
            children.append(lease)
            yield lease
    monkeypatch.setattr(codebase_resources, "acquire_codebase_resources", observed)
    server = mcp.RepositoryMCPServer(config, parent_lease=native_parent)
    try:
        assert "result" in server.handle(initialization())
        assert server.session._index._parent_lease is native_parent
        server.handle(dict(jsonrpc="2.0", method="notifications/initialized"))
        def call(name, **arguments):
            response = server.handle(dict(jsonrpc="2.0", id=2, method="tools/call",
                params=dict(name=name, arguments=arguments)))
            assert response["result"]["isError"] is False
            return json.loads(response["result"]["content"][0]["text"])
        assert call("repository_status")["source_current"] is True
        assert call("repository_read", generation=1, path="calc.py")["representation"] == "compact"
        assert call("repository_context_query", generation=1, query="add")["candidates"]
        assert children and all(lease.released for lease in children)
    finally:
        server.close()
    assert not native_parent.released and not native_parent.cancelled
    from pathlib import Path

    audit = Path(config["audit_store"]).read_text()
    assert "lease_key" not in audit and "parent_lease" not in audit
