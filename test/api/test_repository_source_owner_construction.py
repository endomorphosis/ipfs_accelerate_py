"""Interrupted owner construction retains actual native cleanup custody.

Container manifest/Git admission and server-side worker ancestry are authored,
as in the existing owner fixture. Thread-publication and join failures are
explicit test proxies. Where work starts, it uses the real native caller parent,
real service thread and actual MCP server; no scheduler or parent is fabricated.
"""
import os
from pathlib import Path
import stat
import threading
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_broker as broker
from test.api.test_repository_source_owner_broker import policy_digest, short_service_directory
from test.api.test_repository_source_parent_admission import native_parent


class AuthoredStartInterruption(RuntimeError):
    pass


@pytest.fixture
def construction_case(tmp_path, monkeypatch, native_parent, short_service_directory):
    repository, workspace = tmp_path / "repository", tmp_path / "workspace"
    repository.mkdir()
    workspace.mkdir()
    source = workspace / "calc.py"
    source.write_bytes(b"# Authored source; never executed.\ndef add(value):\n    return value + 1\n")
    source.chmod(0o640)
    original = source.read_bytes(), stat.S_IMODE(source.stat().st_mode)
    boundary = short_service_directory / "boundary.json"
    boundary.write_text("authored container identity; not deployed")
    (short_service_directory / "worker-observations").mkdir(mode=0o750)
    monkeypatch.setattr(broker.guard, "_workspace", lambda *args: (repository, workspace))
    monkeypatch.setattr(broker.guard, "_owner_boundary", lambda path: (
        dict(owner_uid=os.getuid(), worker_uid=os.getuid() + 1,
             allowed_worktree_roots=[str(tmp_path)]), "b" * 64))
    monkeypatch.setattr(broker.guard, "_protected_directory", lambda *args, **kwargs: None)
    monkeypatch.setattr(broker.OwnerRepositoryMCPBroker, "_peer", lambda self, channel: None)
    yield dict(repository=repository, workspace=workspace, boundary=boundary,
        parent_lease=native_parent, expected_policy_sha256=policy_digest(native_parent),
        deadline_monotonic=time.monotonic() + 60)
    assert not native_parent.released
    assert (source.read_bytes(), stat.S_IMODE(source.stat().st_mode)) == original


def _prelaunch_failure(monkeypatch):
    owners = []

    def fail(self, **kwargs):
        owners.append(self)
        raise AuthoredStartInterruption("authored failure before thread launch")

    monkeypatch.setattr(broker.OwnerRepositoryMCPBroker, "_start_session", fail)
    return owners


def _unknown_cleanup():
    return dict(status="unavailable", initializations=None, tool_calls=None,
        successful_reads=None, compact_reads=None, canonical_edits=None,
        refreshes=None, errors=None, uncertain_edits=None, returned_bytes=None,
        store_cleanup_complete=False, artifacts_retained_after_invocation=None)


def test_preallocated_owner_guard_failure_is_known_to_precede_resource_creation(construction_case, monkeypatch):
    def invalid_workspace(*args):
        raise AuthoredStartInterruption("authored workspace guard refusal")

    def no_store(*args, **kwargs):
        pytest.fail("workspace guard failure reached private store creation or cleanup")

    monkeypatch.setattr(broker.guard, "_workspace", invalid_workspace)
    monkeypatch.setattr(broker.policy, "create_runtime_store", no_store)
    monkeypatch.setattr(broker.policy, "finish_runtime_store", no_store)
    owner = broker.OwnerRepositoryMCPBroker.__new__(broker.OwnerRepositoryMCPBroker)
    with pytest.raises(AuthoredStartInterruption):
        broker.OwnerRepositoryMCPBroker.__init__(owner, **construction_case)
    assert owner._resource_creation_started is False
    result = owner.close(timeout_seconds=0)
    assert result["cleanup_complete"] and result["service_stopped"]
    assert result["cleanup_pending"] is False
    assert result["native_child_acquired"] is False and result["native_child_released"] is None
    assert result["native_child_acquisition_uncertain"] is False
    assert result["prepared_mcp_usage"] is None


def test_prelaunch_incomplete_store_cleanup_retains_owner_for_retry(construction_case, monkeypatch):
    owners = _prelaunch_failure(monkeypatch)
    finish = broker.policy.finish_runtime_store
    calls = []

    def incomplete_once(binding, *, expected_identity):
        calls.append(expected_identity)
        if len(calls) == 1:
            return _unknown_cleanup()
        return finish(binding, expected_identity=expected_identity)

    monkeypatch.setattr(broker.policy, "finish_runtime_store", incomplete_once)
    with pytest.raises(broker.RepositoryOwnerConstructionError) as stopped:
        broker.OwnerRepositoryMCPBroker(**construction_case)
    owner = stopped.value.owner
    assert owners == [owner] and isinstance(stopped.value.__cause__, AuthoredStartInterruption)
    private = Path(owner._binding["invocation_dir"])
    try:
        assert private.exists() and not owner._thread_launch_attempted
        assert owner._native is None and owner._thread is None
        result = owner.close(timeout_seconds=0)
        assert result["cleanup_complete"] and not result["cleanup_pending"]
        assert result["session"] is None and result["descriptor_sha256"] is None
        assert result["native_child_acquired"] is False and result["native_child_released"] is None
        assert result["construction_error"] == "AuthoredStartInterruption"
        assert result["thread_launch_attempted"] is False and not private.exists()
        assert len(calls) == 2 and calls[0] == calls[1] == owner._store_identity
    finally:
        owner.close(timeout_seconds=0)


def test_prelaunch_successful_cleanup_reraises_original_interruption(construction_case, monkeypatch):
    owners = _prelaunch_failure(monkeypatch)
    with pytest.raises(AuthoredStartInterruption):
        broker.OwnerRepositoryMCPBroker(**construction_case)
    assert len(owners) == 1
    owner = owners[0]
    assert not Path(owner._binding["invocation_dir"]).exists()
    result = owner.close(timeout_seconds=0)
    assert result["cleanup_complete"] and result["service_stopped"]
    assert result["native_child_acquired"] is False and result["native_child_released"] is None


def test_prelaunch_cleanup_exception_retains_owner_and_error(construction_case, monkeypatch):
    owners = _prelaunch_failure(monkeypatch)
    finish = broker.policy.finish_runtime_store

    def failed_cleanup(*args, **kwargs):
        raise OSError("authored private cleanup interruption")

    monkeypatch.setattr(broker.policy, "finish_runtime_store", failed_cleanup)
    with pytest.raises(broker.RepositoryOwnerConstructionError) as stopped:
        broker.OwnerRepositoryMCPBroker(**construction_case)
    owner = stopped.value.owner
    assert owners == [owner]
    private = Path(owner._binding["invocation_dir"])
    try:
        assert private.exists()
        result = owner.close(timeout_seconds=0)
        assert result["cleanup_pending"] and not result["cleanup_complete"]
        assert result["cleanup_error"] == "OSError" and result["service_stopped"]
        assert result["native_child_acquired"] is False
    finally:
        # The close API retries only this known prelaunch store cleanup. Its
        # retained exception remains visible even after the store is removed.
        monkeypatch.setattr(broker.policy, "finish_runtime_store", finish)
        result = owner.close(timeout_seconds=0)
        assert not private.exists()
        assert result["cleanup_error"] == "OSError" and result["cleanup_pending"]


@pytest.mark.parametrize("disposition", ["unknown", "timeout", "cancelled"])
def test_native_child_admission_disposition_is_not_inferred_from_missing_handle(
        construction_case, native_parent, monkeypatch, disposition):
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import existing_resource_scheduler as existing
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
        LeaseCancelledError, LeaseTimeoutError,
    )

    original_open = existing.open_existing_scheduler
    opened, requests = [], []
    errors = {"unknown": RuntimeError("authored child acquire interruption"),
        "timeout": LeaseTimeoutError("authored child acquire timeout"),
        "cancelled": LeaseCancelledError("authored child acquire cancellation")}

    def open_selected(path, digest):
        selected = original_open(path, digest)
        assert selected is not native_parent._scheduler
        assert selected.state_path == native_parent._scheduler.state_path
        assert digest == construction_case["expected_policy_sha256"]
        opened.append(selected)

        def refuse_child(**kwargs):
            # Only this newly opened broker facade's child call is authored.
            # It never performs admission or mutates native state; the caller's
            # actual parent and its scheduler acquire implementation are intact.
            assert kwargs["parent_lease"].lease_id == native_parent.lease_id
            assert {key: kwargs[key] for key in broker.DEMAND} == broker.DEMAND
            requests.append(kwargs)
            raise errors[disposition]

        monkeypatch.setattr(selected, "acquire", refuse_child)
        return selected

    monkeypatch.setattr(existing, "open_existing_scheduler", open_selected)
    owner = broker.OwnerRepositoryMCPBroker(**construction_case)
    try:
        assert owner._ready.wait(3) and owner._done.wait(3)
        assert len(opened) == len(requests) == 1 and owner._scheduler is opened[0]
        result = owner.close()
        assert result["service_stopped"] and result["accounting_complete"]
        assert result["service_error"] == type(errors[disposition]).__name__
        assert owner._native is None and owner._server is None
        assert result["native_child"] is None and result["native_child_released"] is None
        assert result["prepared_mcp_usage"]["store_cleanup_complete"] is True
        assert not Path(owner._binding["invocation_dir"]).exists()
        assert not owner.socket_path.exists() and not owner.descriptor_path.exists()
        if disposition == "unknown":
            assert result["native_child_acquired"] is None
            assert result["native_child_acquisition_uncertain"] is True
            assert result["cleanup_pending"] and not result["cleanup_complete"]
        else:
            assert result["native_child_acquired"] is False
            assert result["native_child_acquisition_uncertain"] is False
            assert result["cleanup_complete"] and not result["cleanup_pending"]
        assert not native_parent.released and not native_parent.cancelled
    finally:
        owner.close()


@pytest.mark.parametrize("interruption", ["hidden_ident", "join_refused"])
def test_started_native_service_keeps_custody_through_thread_publication_failure(
        construction_case, native_parent, monkeypatch, interruption):
    close_entered, finish_close = threading.Event(), threading.Event()
    real_thread = threading.Thread
    real_close = broker.mcp.RepositoryMCPServer.close
    proxies = []

    def held_close(server):
        close_entered.set()
        assert finish_close.wait(15), "authored close gate was not released"
        return real_close(server)

    class AuthoredThreadPublication:
        def __init__(self, **kwargs):
            self.owner = kwargs["target"].__self__
            self.real = real_thread(**kwargs)
            self.recovered = False
            self.join_refusals = 0

        @property
        def ident(self):
            if interruption == "hidden_ident" and not self.recovered:
                return None
            return self.real.ident

        def start(self):
            self.real.start()
            assert self.owner._ready.wait(12), "native startup exceeded bounded wait"
            assert self.owner._service_error is None
            raise AuthoredStartInterruption("authored Thread.start publication interruption")

        def join(self, timeout=None):
            if interruption == "join_refused" and not self.recovered:
                self.join_refusals += 1
                raise RuntimeError("cannot join thread before it is started")
            return self.real.join(timeout)

        def is_alive(self):
            return self.real.is_alive()

    def thread_factory(*args, **kwargs):
        if kwargs.get("name") == "repository-mcp-reader":
            assert not args
            proxy = AuthoredThreadPublication(**kwargs)
            proxies.append(proxy)
            return proxy
        # Native lease heartbeat threads stay real and retain their normal
        # existing scheduler behavior; this proxy affects only publication.
        return real_thread(*args, **kwargs)

    monkeypatch.setattr(broker.mcp.RepositoryMCPServer, "close", held_close)
    monkeypatch.setattr(broker.threading, "Thread", thread_factory)
    try:
        with pytest.raises(broker.RepositoryOwnerConstructionError) as stopped:
            broker.OwnerRepositoryMCPBroker(**construction_case)
        owner = stopped.value.owner
        assert len(proxies) == 1 and proxies[0].owner is owner
        assert isinstance(stopped.value.__cause__, AuthoredStartInterruption)
        assert close_entered.wait(2), "interrupted service did not enter its owned close"
        private = Path(owner._binding["invocation_dir"])
        child = owner._native
        assert owner._thread_launch_attempted and proxies[0].real.is_alive()
        assert child.parent_lease_id == native_parent.lease_id
        assert not child.released and not native_parent.released and private.exists()
        assert not owner.socket_path.exists() and not owner.descriptor_path.exists()
        assert not owner.audit_path.exists()
        if interruption == "hidden_ident":
            result = owner.close(timeout_seconds=0)
            assert result["cleanup_pending"] and not result["service_stopped"]
            assert not result["native_child_released"] and private.exists()
        else:
            result = owner.close(timeout_seconds=0)
            assert result["cleanup_pending"] and not result["service_stopped"]
            assert result["join_error"] == "RuntimeError" and not result["native_child_released"]
            assert proxies[0].join_refusals >= 2
            assert not child.released and private.exists()
    finally:
        finish_close.set()
        for proxy in proxies:
            proxy.recovered = True
            result = proxy.owner.close()
            assert result["cleanup_complete"] and result["native_child_released"]
            assert result["join_error"] is None
            assert not proxy.real.is_alive() and proxy.owner._done.is_set()
            assert not Path(proxy.owner._binding["invocation_dir"]).exists()
            assert not native_parent.released
