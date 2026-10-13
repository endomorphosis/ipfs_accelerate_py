"""Live native claim admission for owner-held current-source retrieval.

Tasks are materialized and claimed through the real typed daemon and its
process-serialized coordinator. The Quack listener lifecycle and host worker
deployment metadata are authored seams; DuckDB, claim receipts, Unix grants,
retained source, and the allocated Git worktree are real. Owner-side reads do
not qualify a separate worker UID, automatic allocation, or compiled context.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import ProcessSerializedDatabaseCoordinator
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import OwnerLiveness
from ipfs_accelerate_py.agent_supervisor.runtime import current_source_task_admission as admission
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_inventory as sources
from ipfs_accelerate_py.agent_supervisor.runtime import source_formalization_context as boundary_guard
from ipfs_accelerate_py.agent_supervisor.runtime import source_lookup_worker_broker as broker
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import FakeQuackTransport, build_server
from ipfs_accelerate_py.agent_supervisor.task_sources import typed_state_owner as typed
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import canonical_json_bytes, content_identity
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import DatabaseTaskSource
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import open_duckdb_connection
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import QuackStateClient
from ipfs_accelerate_py.agent_supervisor.task_sources.state_owner_bootstrap import StateOwnerBootstrapCredentials
from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_database_task_source import (
    TypedDatabaseTaskSource, daemon_required_owner_command_operations,
    daemon_required_owner_operations,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationDaemon
from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes, cid_for_structured
from test.api.causal_federation.test_bootstrap_runtime import _capability, _migrate
from test.api.test_repository_source_inventory import _repository, _retain, _git


SOURCE = b"# Authored source must not execute.\nVALUE = 42\n"
PROMPT = "Read the host-admitted current source for the live native task."
ALIAS = "SOURCE-ADMISSION-T001"
CLAIM_FIELDS = ("attempt_id", "claim_id", "attempt_number", "lease_id",
                "owner_session_id", "fencing_token", "fence_epoch")


@dataclass(repr=False, eq=False)
class NativeClock:
    """Real wall clock with an initially inert, one-shot source writer control."""

    edit_path: Path
    reads_until_edit: int | None = None
    edited: bool = False

    def __call__(self):
        if self.reads_until_edit is not None:
            self.reads_until_edit -= 1
            if self.reads_until_edit == 0:
                self.edit_path.write_bytes(SOURCE + b"# late coordinator callback edit\n")
                self.reads_until_edit = None
                self.edited = True
        return time.time_ns() // 1_000_000


@dataclass(repr=False)
class NativeCase:
    """Never export private owner resources through an assertion repr."""

    canonical: Path
    workspace: Path
    retained: object
    server: object
    daemon: object
    source: object
    task_cid: str
    before: object
    attempt: object
    native_task: object
    capture_kwargs: dict
    daemon_grant_id: str
    handle: object
    binding: dict
    session: object
    factory: dict
    clock: NativeClock


@contextmanager
def _native_case(tmp_path, monkeypatch, *, defer_source_binding=False, source_files=None):
    canonical = _repository(tmp_path, {b"module.py": SOURCE} if source_files is None else source_files)
    retained = _retain(canonical, tmp_path,
        limits=sources.SourceInventoryLimits(frame_bytes=4096))
    workspace = tmp_path / "allocated"
    if not defer_source_binding:
        _git(canonical, "worktree", "add", "--detach", "-q", str(workspace))
    task_cid = content_identity({"fixture_task": ALIAS, "tree": retained.request.expected_tree})
    goal_cid = content_identity({"fixture_goal": ALIAS})
    database = tmp_path / "control.duckdb"
    materializer = DatabaseTaskSource(database)
    try:
        materializer.materialize({
            "repository_tree_id": "git-tree:" + retained.request.expected_tree,
            "objectives": [{"objective_id": "objective:source-admission",
                "objective_alias": "SOURCE-ADMISSION-O001", "goal_cid": goal_cid,
                "goal_alias": "SOURCE-ADMISSION-G001", "title": "Read inert source",
                "status": "open"}],
            "tasks": [{"task_cid": task_cid, "task_id": ALIAS, "goal_cid": goal_cid,
                "status": "ready", "priority": "P0", "ordinal": 1,
                "title": "Read the retained authored source", "completion": "auto",
                "outputs": [{"path": "module.py"}],
                "acceptance": [{"criterion": "Source retrieval remains read only"}]}],
        })
        assert materializer.get_task(task_cid).status == "ready"
    finally:
        materializer.close()
    # The trusted fixture protects the existing native authority before owner
    # admission; database creation alone follows the process umask.
    database.chmod(0o600)
    server = build_server(database_path=database, state_dir=tmp_path / "owner",
        repository_id=retained.request.repository_id, store_id="source-task-admission-tests",
        transport=FakeQuackTransport(), capability_probe=_capability, migrate=_migrate,
        connection_factory=open_duckdb_connection,
        owner_liveness_probe=lambda _birth: OwnerLiveness.DEAD,
        typed_command_socket_path=tmp_path / "native.sock")
    client = source = daemon = session = None
    control = tempfile.TemporaryDirectory(prefix="sta-")
    try:
        identity = server.start()
        client_id = "database-implementation-daemon:source-task-admission-tests"
        token, grant = server.issue_typed_client_grant_record(
            client_id=client_id, process_birth_id=identity.process_birth_id,
            allowed_operations=daemon_required_owner_operations(),
            allowed_command_operations=daemon_required_owner_command_operations(),
            peer_pid=os.getpid())
        monkeypatch.setenv(typed.TYPED_STATE_OWNER_SOCKET_ENV, str(server.typed_command_socket_path()))
        monkeypatch.setenv(typed.TYPED_STATE_OWNER_TOKEN_ENV, token)
        client = QuackStateClient(owner_id=client_id, store_id=identity.store_id,
            process_birth_id=identity.process_birth_id)
        client.attach(identity.listen_uri, server_id=identity.server_id)
        unsealed = TypedDatabaseTaskSource(client, owns_client=False)
        try:
            policy = unsealed.seal_execution_route_policy({ALIAS: GROK_CODEX_EXECUTION_MODE})
        finally:
            unsealed.close()
        source = TypedDatabaseTaskSource(client, execution_route_policy=policy)
        credentials = StateOwnerBootstrapCredentials(endpoint=identity.listen_uri,
            socket_path=str(server.typed_command_socket_path()), store_id=identity.store_id,
            server_id=identity.server_id, client_id=client_id,
            process_birth_id=identity.process_birth_id, token=token, execution_route_policy=policy)
        # Providers, effects, and validations are never dispatched by this fixture.
        # A real two-lane configuration selects the native serialized authority.
        shard = int(hashlib.sha256(ALIAS.encode()).hexdigest()[:8], 16) % 2
        clock = NativeClock(workspace / "module.py")
        lane_state = tmp_path / "daemon-lane"
        lane_state.mkdir(mode=0o700)
        daemon = DatabaseImplementationDaemon(database_path=database,
            coordination_path=tmp_path / "coordination.duckdb",
            state_dir=lane_state, state_prefix="source-admission",
            task_shard_count=2, task_shard_index=shard, authority_mode="quack",
            task_source_kind="duckdb", owner_session_id="session:source-task-admission-tests",
            process_instance_id=identity.process_birth_id, quack_uri=identity.listen_uri,
            task_source=source, close_task_source=False,
            state_owner_bootstrap_credentials=credentials, strict_task_sharding=True,
            max_task_attempts=2, lease_ms=120_000, require_real_execution=True,
            clock_ms=clock).open()
        assert type(daemon.coordinator) is ProcessSerializedDatabaseCoordinator
        assert daemon._task_belongs_to_shard(ALIAS)
        daemon.coordinator.database_path.chmod(0o600)
        daemon.execution_path.chmod(0o600)
        before = source.get_task(task_cid)
        attempt = daemon.claim_next()
        assert attempt is not None and attempt.task_cid == task_cid
        native_task = source.get_task(task_cid)
        receipt = native_task.body["completion_receipt"]
        assert native_task.status == "in_progress" and native_task.revision == before.revision + 2
        assert receipt["operation"] == "database_attempt_admitted"
        assert receipt["claim_phase_schema"] == typed.TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA
        assert {name: receipt[name] for name in CLAIM_FIELDS} == {
            name: getattr(attempt, name) for name in CLAIM_FIELDS}
        capture_kwargs = dict(server=server, attempt_authority=daemon.coordinator,
            task_cid=task_cid,
            expected_claim_identity={name: getattr(attempt, name) for name in CLAIM_FIELDS},
            peer_identity=(grant.peer_pid, grant.peer_uid, grant.peer_start_time_ticks),
            client_id=client_id)
        handle = admission.capture_native_source_task_admission(**capture_kwargs)
        binding = None
        if not defer_source_binding:
            binding = handle.bind_current_source_lookup(repository=workspace, cas=retained.cas,
                expected_inventory_cid=retained.ref.inventory_cid, request=retained.request,
                limits=retained.limits, context_cid=cid_for_structured({"host_context": task_cid}),
                allowed_paths=(b"module.py".hex(),), max_response_bytes=4096, max_slice_bytes=512)
        boundary = Path(control.name) / "container-boundary.json"
        boundary.write_bytes(b"authored host deployment metadata only\n")
        (boundary.parent / "worker-observations").mkdir(mode=0o750)
        boundary_sha = hashlib.sha256(boundary.read_bytes()).hexdigest()
        monkeypatch.setattr(boundary_guard, "_owner_boundary", lambda _path: (
            {"owner_uid": os.getuid(), "worker_uid": os.getuid() + 1}, boundary_sha))
        monkeypatch.setattr(boundary_guard, "_protected_directory", lambda *args, **kwargs: None)
        factory = dict(server=server, canonical_repository=canonical, workspace=workspace,
            boundary=boundary, prompt=PROMPT, timeout_seconds=90, task_admission=handle)
        if not defer_source_binding:
            session = broker.start_source_lookup_worker_broker(**factory)
        # Credentials and private grant objects stay local and never enter Case.
        del token, credentials, grant
        yield NativeCase(canonical, workspace, retained, server, daemon, source,
            task_cid, before, attempt, native_task, capture_kwargs,
            receipt["claim_process_attestation"]["grant_id"], handle, binding, session, factory, clock)
    finally:
        try:
            if session is not None:
                session.close()
        finally:
            try:
                if daemon is not None:
                    daemon.close()
            finally:
                try:
                    if source is not None:
                        source.close()
                    elif client is not None:
                        client.close()
                finally:
                    server.stop()
                    control.cleanup()


@pytest.fixture
def native(tmp_path, monkeypatch):
    with _native_case(tmp_path, monkeypatch) as case:
        yield case


def _payload(case):
    return dict(schema="supervisor-current-source-request@1",
        binding_cid=case.binding["binding_cid"], raw_path_hex=b"module.py".hex(),
        source_cid=cid_for_bytes(SOURCE), offset=0, max_bytes=512, continuation=None)


def _update_body(case, body):
    with case.server._lock:
        case.server._connection.execute("UPDATE tasks SET body_json = ? WHERE task_cid = ?",
            [canonical_json_bytes(body).decode(), case.task_cid])


def _assert_refused(case):
    with pytest.raises((ValueError, RuntimeError)):
        case.handle.require_current()
    with pytest.raises((ValueError, RuntimeError)):
        case.session._read_source(_payload(case))
    assert case.session._native_response_bytes == 0


def test_real_typed_claim_derives_full_body_and_retrieves_allocated_source(native):
    snapshot = native.handle.require_current()
    assert snapshot["task_cid"] == native.task_cid
    assert snapshot["task_alias"] == ALIAS
    assert snapshot["task_revision"] == native.native_task.revision
    assert snapshot["expected_task_body_cid"] == content_identity(dict(native.native_task.body))
    assert snapshot["task_content_read_only"] is True
    assert all(snapshot[name] is False for name in (
        "proof_authority", "execution_authority", "completion_authority",
        "allocation_ownership_verified"))
    assert native.retained.request.task_id != ALIAS
    assert native.binding["task_cid"] == native.task_cid
    assert native.binding["expected_task_body_cid"] == snapshot["expected_task_body_cid"]
    snapshot["task_revision"] += 100
    assert native.handle.require_current()["task_revision"] == native.native_task.revision
    claim = native.daemon.coordinator.get_task_claim(native.attempt.claim_id)
    lease = native.daemon.coordinator.protect_task_claim(claim)
    renewed = native.daemon.coordinator.renew(lease, lease_ms=120_000)
    assert renewed.lease_id == lease.lease_id
    assert native.handle.require_current()["task_revision"] == native.native_task.revision
    result = native.session._read_source(_payload(native))
    assert result["content"].encode() == SOURCE
    assert result["source_complete"] is True and result["working_source_verified"] is True
    assert result["completion_authority"] is False
    assert native.session._native_response_bytes > 0
    assert native.session._native_wire_bytes == native.session._native_response_bytes + 3
    assert native.daemon.provider_invocation_recorded(native.attempt.attempt_id,
        idempotency_key="provider:" + native.attempt.attempt_id) is None
    public = canonical_json_bytes(native.session.descriptor)
    assert "claim_process_attestation" not in public.decode()
    assert "token" not in public.decode() and str(native.server.config.database_path) not in public.decode()


def test_ready_native_task_with_a_claim_tuple_is_not_an_admitted_task(native):
    # Restore the materializer's actual pre-claim row, never an invented receipt.
    with native.server._lock:
        native.server._connection.execute(
            "UPDATE tasks SET status = 'ready', revision = ?, body_json = ? WHERE task_cid = ?",
            [native.before.revision, canonical_json_bytes(dict(native.before.body)).decode(), native.task_cid])
    with pytest.raises((ValueError, RuntimeError)):
        admission.capture_native_source_task_admission(**native.capture_kwargs)
    _assert_refused(native)


@pytest.mark.parametrize("change", ["attempt_id", "fencing_token"])
def test_host_alleged_claim_tuple_cannot_replace_actual_native_claim(native, change):
    kwargs = dict(native.capture_kwargs)
    claimed = deepcopy(kwargs["expected_claim_identity"])
    claimed[change] = "attempt:foreign" if change == "attempt_id" else claimed[change] + 1
    kwargs["expected_claim_identity"] = claimed
    with pytest.raises((ValueError, RuntimeError)):
        admission.capture_native_source_task_admission(**kwargs)
    assert native.handle.require_current()["task_cid"] == native.task_cid


@pytest.mark.parametrize("change", ["peer", "client"])
def test_actual_claim_attestation_requires_its_original_peer_and_client(native, change):
    kwargs = dict(native.capture_kwargs)
    if change == "peer":
        pid, uid, birth = kwargs["peer_identity"]
        kwargs["peer_identity"] = (pid, uid, birth + 1)
    else:
        kwargs["client_id"] = "database-implementation-daemon:foreign"
    with pytest.raises((ValueError, RuntimeError)):
        admission.capture_native_source_task_admission(**kwargs)
    assert native.handle.require_current()["task_cid"] == native.task_cid


@pytest.mark.parametrize("transition", ["released", "expired"])
def test_real_coordinator_lease_transition_invalidates_live_source_proxy(native, transition):
    coordinator = native.daemon.coordinator
    claim = coordinator.get_task_claim(native.attempt.claim_id)
    lease = coordinator.protect_task_claim(claim)
    if transition == "released":
        coordinator.release(lease, reason="authored test release")
    else:
        coordinator.expire_task_claim(claim, now_ms=lease.expires_at_ms)
    assert native.session._closed is False
    _assert_refused(native)


def test_revoked_real_daemon_grant_refuses_even_when_source_proxy_grant_is_live(native):
    gateway = native.server._command_gateway
    gateway.revoke_grant(native.daemon_grant_id)
    proxy = native.session._grant
    assert gateway._require_active_grant(proxy,
        peer_identity=(proxy.peer_pid, proxy.peer_uid, proxy.peer_start_time_ticks)).grant_id == proxy.grant_id
    _assert_refused(native)


@pytest.mark.parametrize("change", ["revision", "body", "phase"])
def test_native_revision_full_body_and_claim_phase_changes_refuse_old_admission(native, change):
    if change == "revision":
        with native.server._lock:
            native.server._connection.execute("UPDATE tasks SET revision = revision + 1 WHERE task_cid = ?", [native.task_cid])
    else:
        body = deepcopy(dict(native.native_task.body))
        if change == "body":
            body["title"] = "The selected task now has a different full body"
        else:
            body["completion_receipt"]["attempt_execution_phase"] = "context"
        _update_body(native, body)
    _assert_refused(native)


@pytest.mark.parametrize("change", ["service", "cas"])
def test_original_binding_objects_cannot_be_replaced_by_equivalent_looking_inputs(native, change, monkeypatch):
    service = native.server._command_gateway._current_source_lookup
    if change == "service":
        monkeypatch.setattr(native.server._command_gateway, "_current_source_lookup", object())
    else:
        replacement = ImmutableCAS(native.retained.cas.root)
        monkeypatch.setattr(service._store, "cas", replacement)
    with pytest.raises((ValueError, RuntimeError)):
        native.session._read_source(_payload(native))
    assert native.session._native_response_bytes == 0


def test_native_task_admission_does_not_authorize_a_different_allocation(native, monkeypatch):
    service = native.server._command_gateway._current_source_lookup
    assert service._native_source_task_admission is native.handle
    proxy_grant_id = native.session._grant.grant_id
    def no_proxy_grant(**kwargs):
        pytest.fail("refused task-admission factory issued a worker proxy grant")
    with monkeypatch.context() as controlled:
        controlled.setattr(native.server, "issue_typed_client_grant_record", no_proxy_grant)
        omitted = dict(native.factory)
        omitted.pop("task_admission")
        with pytest.raises((ValueError, RuntimeError)):
            broker.start_source_lookup_worker_broker(**omitted)
        serialized = dict(native.factory, task_admission=native.handle.require_current())
        with pytest.raises((ValueError, RuntimeError)):
            broker.start_source_lookup_worker_broker(**serialized)
        factory = dict(native.factory, workspace=native.canonical)
        with pytest.raises((ValueError, RuntimeError)):
            broker.start_source_lookup_worker_broker(**factory)
    assert service._native_source_task_admission is native.handle
    assert native.session._grant.grant_id == proxy_grant_id
    assert native.handle.require_current()["task_cid"] == native.task_cid
    assert native.session._read_source(_payload(native))["content"].encode() == SOURCE


def test_task_drift_after_actual_backend_response_refuses_and_retains_incurred_cost(native, monkeypatch):
    original = native.session._client.source_get
    def changed_after_read(payload):
        result = original(payload)
        body = deepcopy(dict(native.native_task.body))
        body["title"] = "Native task changed after the source RPC"
        _update_body(native, body)
        return result
    monkeypatch.setattr(native.session._client, "source_get", changed_after_read)
    with pytest.raises((ValueError, RuntimeError)):
        native.session._read_source(_payload(native))
    assert native.session._native_response_bytes > 0
    assert native.session._native_wire_bytes == native.session._native_response_bytes + 3
    with pytest.raises((ValueError, RuntimeError)):
        native.handle.require_current()


def test_closing_native_clock_callback_source_edit_cannot_escape_final_byte_checks(native, monkeypatch):
    original = native.session._client.source_get
    def arm_after_read(payload):
        response = original(payload)
        # The immediate post-response native fence reads the real clock before
        # and after its callback. Edit on the next read, in the closing fence.
        native.clock.reads_until_edit = 3
        return response
    monkeypatch.setattr(native.session._client, "source_get", arm_after_read)
    assert native.daemon.coordinator._clock_ms is native.clock
    with pytest.raises((ValueError, RuntimeError)):
        native.session._read_source(_payload(native))
    assert native.clock.edited is True
    assert native.session._native_response_bytes > 0
    assert native.session._native_wire_bytes == native.session._native_response_bytes + 3
