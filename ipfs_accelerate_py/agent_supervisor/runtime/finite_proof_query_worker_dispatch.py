"""Owner-mediated proof custody immediately before the literal coding Popen.

The public socket and context CID are routing information, not capabilities.
Authority comes from the connecting kernel peer's existing native claim and
grant. The owner retains its task and proof locks until the daemon reports the
actual child birth. No source observation runs after that birth.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import threading
import time
import uuid

from ..task_sources.control_plane_contracts import canonical_json_bytes
from ..task_sources import typed_state_owner as typed
from . import finite_repository_execution as execution

SOCKET_ENV = "IPFS_ACCELERATE_FINITE_PROOF_QUERY_DISPATCH_SOCKET"
CONTEXT_ENV = "IPFS_ACCELERATE_FINITE_PROOF_QUERY_CONTEXT_CID"
WRAPPER_ENV = "IPFS_ACCELERATE_FINITE_PROOF_QUERY_WRAPPER_TICKET"
SCHEMA = "supervisor-finite-proof-query-worker-dispatch@1"
MAX_BYTES = 65_536
MAX_COMMAND_PARTS = 32
REQUEST_TIMEOUT_SECONDS = 95.0
ACK_TIMEOUT_SECONDS = 5.0
_SEAL = object()
_BINDING_FIELDS = frozenset({"task_id", "database_task_cid", "database_attempt_id",
                           "database_claim_id", "database_attempt_number"})
_REQUEST_FIELDS = frozenset({"schema", "operation", "context_cid", "command", "worktree",
                           *_BINDING_FIELDS})
_ISOLATED_FIELDS = frozenset({"schema", "operation", "context_cid", "command", "worktree",
                            "wrapper_ticket"})


class FiniteProofQueryDispatchError(RuntimeError):
    """The exact native coding-child handoff failed closed."""


def _need(value, message):
    if not value:
        raise FiniteProofQueryDispatchError(message)


def _text(value, *, limit=4096):
    _need(type(value) is str and 0 < len(value.encode("utf-8")) <= limit
          and not any(char in value for char in ("\x00", "\n", "\r")),
          "bounded literal dispatch identity required")
    return value


def _command(value):
    _need(type(value) is list and 1 <= len(value) <= MAX_COMMAND_PARTS,
          "bounded literal dispatch command required")
    return [_text(item, limit=8192) for item in value]


def _send(channel, value):
    body = canonical_json_bytes(value)
    _need(2 <= len(body) <= MAX_BYTES, "dispatch frame exceeds its byte bound")
    channel.sendall(len(body).to_bytes(4, "big") + body)


def _exact(channel, length):
    chunks = []
    while length:
        chunk = channel.recv(length)
        _need(bool(chunk), "dispatch channel closed before acknowledgement")
        chunks.append(chunk)
        length -= len(chunk)
    return b"".join(chunks)


def _receive(channel):
    size = int.from_bytes(_exact(channel, 4), "big")
    _need(2 <= size <= MAX_BYTES, "dispatch frame size exceeds its byte bound")
    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result, "duplicate dispatch field refused")
            result[key] = value
        return result
    try:
        value = json.loads(_exact(channel, size).decode("utf-8"), object_pairs_hook=pairs,
                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (UnicodeError, ValueError) as error:
        raise FiniteProofQueryDispatchError("malformed dispatch frame refused") from error
    _need(type(value) is dict, "closed dispatch object required")
    return value


def _request(value):
    _need(type(value) is dict and set(value) == _REQUEST_FIELDS
          and value["schema"] == SCHEMA and value["operation"] == "prepare",
          "dispatch request differs from its closed schema")
    for name in _REQUEST_FIELDS - {"command", "database_attempt_number"}:
        _text(value[name])
    _command(value["command"])
    _need(type(value["database_attempt_number"]) is int
          and 1 <= value["database_attempt_number"] < (1 << 32),
          "native dispatch attempt ordinal required")
    return value


def _isolated_request(value):
    _need(type(value) is dict and set(value) == _ISOLATED_FIELDS
          and value["schema"] == SCHEMA and value["operation"] == "prepare_isolated",
          "isolated dispatch request differs from its closed schema")
    for name in _ISOLATED_FIELDS - {"command"}:
        _text(value[name])
    _command(value["command"])
    return value


def _kernel_identity(pid):
    start, parent, boot = typed._process_runtime_facts(pid)
    status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    uids = [line.split()[1:] for line in status.splitlines() if line.startswith("Uid:")]
    _need(len(uids) == 1 and len(uids[0]) == 4, "kernel child UID identity unavailable")
    uid = int(uids[0][1])
    command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\x00")
    _need(command and any(command), "live kernel child command unavailable")
    _need(typed._process_start_time_ticks(pid) == start, "kernel child PID identity changed")
    return {"pid": pid, "uid": uid, "parent_pid": parent, "start_time_ticks": start,
            "process_birth_id": typed._process_birth_content_id(pid, start, boot, parent),
            "command": [part.decode("utf-8", errors="strict") for part in command if part]}


def _require_wrapper_kernel_command(*, child, command, worktree):
    identity = _kernel_identity(child)
    argv = identity["command"]
    _need(identity["uid"] == os.geteuid() and argv.count(command[0]) == 1
          and argv[argv.index(command[0]):] == command
          and os.readlink(f"/proc/{child}/cwd") == worktree,
          "native wrapper kernel command, UID or allocated worktree differs")
    return identity


def _isolated_kernel_birth(*, process_pid, wrapper_pid, command, reported_pid=None):
    """Observe an actual UID1001 worker beneath this exact sudo Popen birth."""
    sudo = _kernel_identity(process_pid)
    _need(sudo["parent_pid"] == wrapper_pid, "isolated Popen child has another native wrapper")
    if reported_pid is None:
        candidates = list(Path("/proc").iterdir())
        _need(len(candidates) <= 16_384, "kernel process population exceeds isolated dispatch bound")
        pids = [int(path.name) for path in candidates if path.name.isdigit()]
    else:
        _need(type(reported_pid) is int and reported_pid > 0, "actual isolated worker PID required")
        pids = [reported_pid]
    matches = []
    for pid in pids:
        try:
            worker = _kernel_identity(pid)
            if worker["uid"] != 1001:
                continue
            # The kernel may expose the shebang interpreter before worker-entry.
            argv = worker["command"]
            marker = command[5]
            if marker not in argv or argv[argv.index(marker) + 1:] != command[6:]:
                continue
            chain, current, seen = [], worker, set()
            for _ in range(8):
                if current["pid"] in seen:
                    break
                seen.add(current["pid"])
                chain.append({key: current[key] for key in
                    ("pid", "uid", "parent_pid", "start_time_ticks", "process_birth_id")})
                if current["pid"] == process_pid:
                    matches.append({"worker": worker, "sudo": sudo, "parent_chain": chain})
                    break
                if current["parent_pid"] <= 0 or current["parent_pid"] == wrapper_pid:
                    break
                current = _kernel_identity(current["parent_pid"])
        except (OSError, UnicodeError, ValueError, RuntimeError):
            continue
    _need(len(matches) == 1, "exact live UID1001 worker birth not yet observed")
    return matches[0]


def _require_native_dispatch_claim(*, server, expected_population, request, peer_identity,
                                   client_id, store_id):
    """Detached native SQL and existing native grant checks, under owner lock.

    This helper alone does not qualify a runtime, bootstrap or proof owner.
    The sealed broker adds those conditions before accepting a handoff.
    """
    _request(request)
    _need(server.identity is not None and server.identity.store_id == store_id,
          "dispatch native store identity differs")
    actual = execution._physical_native(server._connection)
    selected = expected_population["selected_task_cids"]
    _need(type(selected) is list and len(selected) == 1
          and request["database_task_cid"] == selected[0],
          "dispatch selected task differs from its native reservation")
    before = expected_population["tasks"]
    _need(len(actual["tasks"]) == len(before)
          and actual["completion_rows"] == expected_population["completion_rows"],
          "dispatch full native task or prerequisite population changed")
    expected = {row["task_cid"]: row for row in before}
    _need(set(expected) == {row["task_cid"] for row in actual["tasks"]},
          "dispatch full native task identities changed")
    selected_row = None
    for row in actual["tasks"]:
        prior = expected[row["task_cid"]]
        if row["task_cid"] != selected[0]:
            _need(canonical_json_bytes(row) == canonical_json_bytes(prior),
                  "dispatch prerequisite native rows changed")
            continue
        selected_row = row
        _need(prior["status"] == "ready" and row["status"] == "in_progress"
              and row["revision"] == prior["revision"] + 2
              and row["task_alias"] == request["task_id"],
              "dispatch requires the exact native admitted claim revision")
        _need(all(canonical_json_bytes(row[name]) == canonical_json_bytes(prior[name])
                  for name in prior if name not in {"status", "revision", "updated_at", "body"}),
              "dispatch selected task semantics changed")
        old_body, body = dict(prior["body"]), dict(row["body"])
        old_body.pop("completion_receipt", None)
        receipt = body.pop("completion_receipt", None)
        _need(canonical_json_bytes(old_body) == canonical_json_bytes(body)
              and type(receipt) is dict, "dispatch selected native task body changed")
    _need(selected_row is not None, "dispatch selected native task absent")
    receipt = selected_row["body"]["completion_receipt"]
    identity = typed._validated_database_claim_identity(receipt)
    _need(receipt.get("operation") == "database_attempt_admitted"
          and receipt.get("claim_phase_schema") == typed.TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA
          and type(receipt.get("claimed_from_revision")) is int
          and receipt["claimed_from_revision"] == expected[selected[0]]["revision"]
          and type(receipt.get("admitted_from_revision")) is int
          and receipt["admitted_from_revision"] == selected_row["revision"] - 1
          and receipt.get("attempt_execution_phase") == "claimed"
          and type(receipt.get("attempt_execution_revision")) is int
          and receipt["attempt_execution_revision"] == 1
          and identity["claim_id"] == request["database_claim_id"]
          and identity["attempt_id"] == request["database_attempt_id"]
          and identity["attempt_number"] == request["database_attempt_number"],
          "dispatch native attempt or claim binding differs")
    attestation = receipt.get("claim_process_attestation")
    _need(type(attestation) is dict, "dispatch native claim process attestation absent")
    gateway = server._command_gateway
    _need(type(gateway) is typed.TypedStateOwnerGateway
          and gateway._connection is server._connection
          and gateway._transaction_lock is server._lock,
          "dispatch requires its exact native typed owner gateway")
    # Capability dictionary keys are private tokens. Only values are inspected;
    # neither tokens nor complete private credentials enter the protocol.
    with gateway._grants_lock:
        candidates = [grant for grant in gateway._grants.values()
                      if grant.grant_id == attestation.get("grant_id")]
    _need(len(candidates) == 1, "dispatch native claim grant absent or ambiguous")
    grant = gateway._require_active_grant(candidates[0], peer_identity=peer_identity)
    _need(grant.client_id == client_id and client_id.startswith("database-implementation-daemon:"),
          "dispatch native bootstrap client identity differs")
    typed._require_database_claim_process_attestation(receipt, grant=grant)
    return {"task_cid": selected[0], "task_revision": selected_row["revision"],
            "attempt_id": identity["attempt_id"], "claim_id": identity["claim_id"],
            "attempt_number": identity["attempt_number"],
            **{name: identity[name] for name in
               ("lease_id", "owner_session_id", "fencing_token", "fence_epoch")},
            "grant_id": grant.grant_id,
            "client_id": grant.client_id, "store_id": store_id,
            "process_birth_id": grant.process_birth_id, "peer_pid": grant.peer_pid,
            "peer_uid": grant.peer_uid, "peer_start_time_ticks": grant.peer_start_time_ticks,
            "grant_expires_at_ms": grant.expires_at}


def _stable_native_dispatch_claim(claim):
    return {key: value for key, value in claim.items() if key != "grant_expires_at_ms"}


def _same_native_dispatch_claim(current, captured):
    return (canonical_json_bytes(_stable_native_dispatch_claim(current))
            == canonical_json_bytes(_stable_native_dispatch_claim(captured))
            and type(current.get("grant_expires_at_ms")) is int
            and current["grant_expires_at_ms"] >= captured["grant_expires_at_ms"])


def _require_native_grant_ack_budget(*, server, claim):
    """Read the genuine current grant and require its full birth-ACK budget."""
    gateway = server._command_gateway
    with gateway._grants_lock:
        candidates = [grant for grant in gateway._grants.values()
                      if grant.grant_id == claim["grant_id"]]
    _need(len(candidates) == 1, "native dispatch grant absent or ambiguous at acknowledgement fence")
    # The gateway's lock is intentionally nonreentrant. Its owned getter
    # reacquires that lock and resolves the current record by stable grant ID.
    grant = gateway._require_active_grant(candidates[0], peer_identity=(
        claim["peer_pid"], claim["peer_uid"], claim["peer_start_time_ticks"]))
    _need(grant.client_id == claim["client_id"]
          and grant.process_birth_id == claim["process_birth_id"]
          and grant.expires_at >= claim["grant_expires_at_ms"]
          and grant.expires_at > time.time_ns() // 1_000_000
              + int(ACK_TIMEOUT_SECONDS * 1000),
          "native dispatch grant has no remaining birth acknowledgement budget")
    return grant.expires_at


class OwnerFiniteProofQueryDispatchBroker:
    """Exact live owner capability; serialized values and subclasses are inert."""

    def __init__(self, seal, *, scope, runtime):
        from ..entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
        from .finite_proof_query_execution import FrozenFiniteProofQueryExecutionClosure
        _need(seal is _SEAL and type(self) is OwnerFiniteProofQueryDispatchBroker
              and type(scope) is execution.FrozenFiniteRepositoryExecutionScope
              and type(runtime) is AdmittedBenchmarkRuntime and scope._runtime is runtime
              and runtime.finite_execution_scope is scope,
              "dispatch broker requires its exact bound native runtime and scope")
        closure = scope._proof_query_closure
        _need(type(closure) is FrozenFiniteProofQueryExecutionClosure
              and closure._scope is scope and closure._runtime is runtime,
              "dispatch broker requires its exact live native proof-query closure")
        closure._assert_scope(scope)
        state = Path(runtime.state)
        metadata = state.lstat()
        _need(state.is_absolute() and state.resolve() == state and stat.S_ISDIR(metadata.st_mode)
              and metadata.st_uid == os.geteuid() and not metadata.st_mode & 0o077,
              "dispatch socket requires an exact private owner runtime state directory")
        self._seal, self._scope, self._runtime, self._closure = seal, scope, runtime, closure
        self._server, self._catalog = scope._server, closure._catalog
        self._context = closure.material_binding["worker_context"]["context_cid"]
        self._population = scope.to_dict()["payload"]["native_population"]
        self._command = scope.to_dict()["payload"]["candidate"]["argv"]
        self._population_bytes = canonical_json_bytes(self._population)
        self._command_bytes = canonical_json_bytes(self._command)
        self._context_bytes = canonical_json_bytes(self._context)
        self._state_path = state
        self._state_identity = (metadata.st_dev, metadata.st_ino, metadata.st_uid,
                                stat.S_IMODE(metadata.st_mode))
        self.socket_path = state / ("proof-query-dispatch-" + uuid.uuid4().hex[:12] + ".sock")
        self._original_socket_path = self.socket_path
        _need(len(os.fsencode(self.socket_path)) < 108, "dispatch Unix socket path exceeds kernel bound")
        self._fields = (scope, runtime, closure, self._server, self._catalog,
                        self._server._connection, self._server._lock, self._catalog.store._lock,
                        runtime.state, runtime.client_id, runtime.owner_identity.store_id,
                        canonical_json_bytes(scope.to_dict()), canonical_json_bytes(closure.to_dict()))
        self._stop, self._lock = threading.Event(), threading.Lock()
        self._pending, self._spawned, self._closed = False, False, False
        self._wrapper, self._isolated_spawned = None, False
        self._observations, self._dropped_observations, self._close_receipt = [], 0, None
        self._claim_checked, self._runtime_lease_checked = False, False
        self._grant_ack_budget_checked = False
        self._attempt_fence_boundaries = []
        self._attempt_authority_path, self._attempt_authority_identity = None, None
        self._stage, self._boundary = "idle", "unparsed"
        self._birth_ack_pending = False
        from .finite_proof_query_worker_source_custody import capture_worker_source_custody_baseline
        from .finite_proof_query_admission import _budget
        self._worker_source_baseline = capture_worker_source_custody_baseline(
            scope=scope, checkpoint=_budget(closure._owner))
        self._allocation, self._worker_source_custody = None, None
        self._original_worker_source_baseline = self._worker_source_baseline
        self._bound_worker_allocation = None
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self._listener.bind(str(self.socket_path))
            os.chmod(self.socket_path, 0o600)
            self._socket_identity = (self.socket_path.stat().st_dev, self.socket_path.stat().st_ino)
            self._listener.listen(4)
            self._listener.settimeout(0.2)
            self._thread = threading.Thread(target=self._serve, daemon=True,
                                            name="finite-proof-query-coding-dispatch")
            self._thread.start()
        except BaseException:
            self._listener.close()
            self.socket_path.unlink(missing_ok=True)
            raise

    def daemon_environment(self):
        self._identity_current()
        return {SOCKET_ENV: str(self.socket_path), CONTEXT_ENV: self._context}

    @property
    def observations(self):
        return json.loads(canonical_json_bytes(self._observations))

    def _observe(self, value):
        if len(self._observations) < 64:
            self._observations.append(value)
        else:
            self._dropped_observations += 1

    def _identity_current(self):
        _need(type(self) is OwnerFiniteProofQueryDispatchBroker and self._seal is _SEAL and not self._closed,
              "exact active native dispatch broker required")
        scope, runtime, closure = self._scope, self._runtime, self._closure
        current = (scope, runtime, closure, scope._server, closure._catalog,
                   scope._server._connection, scope._server._lock, closure._catalog.store._lock,
                   runtime.state, runtime.client_id, runtime.owner_identity.store_id,
                   canonical_json_bytes(scope.to_dict()), canonical_json_bytes(closure.to_dict()))
        _need(len(current) == len(self._fields)
              and all(left is right if number < 8 else left == right
                  for number, (left, right) in enumerate(zip(current, self._fields)))
              and scope._runtime is runtime and runtime.finite_execution_scope is scope
              and closure._scope is scope and closure._runtime is runtime,
              "native dispatch broker inputs were rebound")
        payload, material = scope.to_dict()["payload"], closure.material_binding
        _need(canonical_json_bytes(self._population) == self._population_bytes
              == canonical_json_bytes(payload["native_population"])
              and canonical_json_bytes(self._command) == self._command_bytes
              == canonical_json_bytes(payload["candidate"]["argv"])
              and canonical_json_bytes(self._context) == self._context_bytes
              == canonical_json_bytes(material["worker_context"]["context_cid"])
              and self.socket_path == self._original_socket_path
              and Path(runtime.state) == self._state_path,
              "native dispatch cached signed inputs or socket path were rebound")
        state = self._state_path.lstat()
        _need(stat.S_ISDIR(state.st_mode) and self._state_path.resolve() == self._state_path
              and (state.st_dev, state.st_ino, state.st_uid, stat.S_IMODE(state.st_mode))
              == self._state_identity,
              "native dispatch owner state directory identity or mode changed")
        endpoint = self._original_socket_path.lstat()
        _need(stat.S_ISSOCK(endpoint.st_mode) and endpoint.st_uid == os.geteuid()
              and stat.S_IMODE(endpoint.st_mode) == 0o600
              and (endpoint.st_dev, endpoint.st_ino) == self._socket_identity,
              "native dispatch original socket inode, owner or private mode changed")
        scope._active()
        closure._assert_scope(scope)
        _need(self._worker_source_baseline is self._original_worker_source_baseline
              and (self._bound_worker_allocation is None
                   or (self._allocation is self._bound_worker_allocation[0]
                       and self._worker_source_custody is self._bound_worker_allocation[1])),
              "native dispatch source allocation custody was rebound")

    def _bootstrap_current(self, claim):
        receipts = self._runtime.bootstrap_receipts
        _need(type(receipts) is list and len(receipts) <= 16,
              "native bootstrap receipt population exceeds dispatch bound")
        matches = [item for item in receipts if type(item) is dict
                   and item.get("pid") == claim["peer_pid"]
                   and item.get("process_birth_id") == claim["process_birth_id"]
                   and item.get("client_id") == claim["client_id"]
                   and item.get("store_id") == claim["store_id"]
                   and item.get("server_id") == self._runtime.owner_identity.server_id
                   and item.get("configuration_root") == self._runtime.manifest_id
                   and item.get("route_policy_id") == self._runtime.route_policy.policy_id]
        _need(len(matches) == 1, "dispatch peer lacks its exact native child bootstrap receipt")

    def _run_lease_current(self):
        runtime = self._runtime
        protected = runtime.coordinator.protect_write(
            runtime.lease, expected_fencing_token=runtime.lease.fencing_token,
            expected_fence_epoch=runtime.lease.fence_epoch)
        _need(protected.resource_id == runtime.run_id
              and protected.owner_session_id == runtime.local_profile.identity_did
              and protected.expires_at_ms > time.time_ns() // 1_000_000
              + int(ACK_TIMEOUT_SECONDS * 1000),
              "native dispatch runtime lease lost its exact live owner or acknowledgement budget")
        self._run_lease_expiry = protected.expires_at_ms
        self._runtime_lease_checked = True

    def _attempt_authority(self, *, claim, daemon_peer):
        """Derive the genuine daemon sidecar; sender paths never select it."""
        from ..merge.database_coordination import ProcessSerializedDatabaseCoordinator
        from ..task_sources.board_control_plane import repo_resident_duckdb
        from ..todo_daemon.implementation_daemon import _database_daemon_quack_sidecar_paths

        runtime = self._runtime
        observed = _kernel_identity(daemon_peer[0])
        argv = observed["command"]
        module = "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon"
        _need(observed["uid"] == os.geteuid()
              and observed["start_time_ticks"] == daemon_peer[2]
              and observed["process_birth_id"] == claim["process_birth_id"]
              and argv[:4] == [runtime.manifest["argv"][0], "-P", "-m", module],
              "attempt authority requires the exact bootstrapped native daemon")

        def exact_option(name, expected):
            _need(argv.count(name) == 1
                  and not any(part.startswith(name + "=") for part in argv),
                  "attempt authority daemon option is missing or ambiguous")
            position = argv.index(name)
            _need(position + 1 < len(argv) and argv[position + 1] == expected,
                  "attempt authority daemon option differs from its owner launch")

        control = Path(self._server.config.database_path)
        _need(control.is_absolute() and control.resolve() == control
              and control.suffix.lower() in {".duckdb", ".ddb"},
              "attempt authority requires its exact canonical native control path")
        for name, expected in (
            ("--todo-path", str(control)), ("--state-dir", str(runtime.state / "run")),
            ("--state-prefix", "admitted"), ("--authority-mode", "quack"),
            ("--task-source-kind", "duckdb"), ("--task-shard-count", "1"),
            ("--task-shard-index", "0"), ("--state-owner-client-id", runtime.client_id),
            ("--quack-endpoint", runtime.owner_identity.listen_uri),
        ):
            exact_option(name, expected)
        _need(not any(part == name or part.startswith(name + "=")
                      for name in ("--database-path", "--coordination-path", "--execution-path")
                      for part in argv),
              "attempt authority forbids an unregistered daemon sidecar override")
        derived = Path(repo_resident_duckdb(
            control.with_name(f"{control.stem}.coordination.duckdb"), relocate=False)).absolute()
        execution_path = Path(repo_resident_duckdb(
            control.with_name(f"{control.stem}.execution.duckdb"), relocate=False)).absolute()
        path, _execution = _database_daemon_quack_sidecar_paths(
            control, coordination_path=derived, execution_path=execution_path)
        _need(path != runtime.coordinator.database_path,
              "attempt authority must differ from the native runtime run lease")
        metadata = self._attempt_authority_file_current(path)
        if self._attempt_authority_path is None:
            self._attempt_authority_path = path
            self._attempt_authority_identity = metadata
        _need(path == self._attempt_authority_path
              and metadata == self._attempt_authority_identity,
              "native attempt authority path or file custody was rebound")
        return ProcessSerializedDatabaseCoordinator(
            path, lock_timeout_seconds=ACK_TIMEOUT_SECONDS,
            require_existing_authority=True)

    def _attempt_authority_file_current(self, path):
        _need(path.is_absolute() and path.resolve() == path,
              "native attempt authority path is not canonical")
        parent = path.parent.lstat()
        _need(stat.S_ISDIR(parent.st_mode) and parent.st_uid == os.geteuid()
              and not parent.st_mode & 0o077,
              "native attempt authority requires its private owner directory")
        metadata = path.lstat()
        _need(stat.S_ISREG(metadata.st_mode) and metadata.st_uid == os.geteuid()
              and metadata.st_nlink == 1 and not metadata.st_mode & 0o002,
              "native attempt authority is not an owner single-link regular file")
        return (metadata.st_dev, metadata.st_ino, metadata.st_uid,
                metadata.st_nlink, stat.S_IMODE(metadata.st_mode))

    def _execute_with_attempt_fence(self, *, request, daemon_peer, callback):
        # Remote claim callbacks acquire COOR before SERVER. Release the
        # preliminary SERVER lookup, then use that same order for the handoff.
        with self._server._lock:
            self._identity_current()
            claim = _require_native_dispatch_claim(server=self._server,
                expected_population=self._population, request=request, peer_identity=daemon_peer,
                client_id=self._runtime.client_id, store_id=self._runtime.owner_identity.store_id)
            self._bootstrap_current(claim)
        coordinator = self._attempt_authority(claim=claim, daemon_peer=daemon_peer)
        identity = {name: claim[name] for name in ("task_cid", "claim_id", "attempt_id",
            "attempt_number", "lease_id", "owner_session_id", "fencing_token", "fence_epoch")}
        with coordinator:
            def fenced(lease):
                result = callback(claim, lease)
                _need(self._attempt_authority_file_current(self._attempt_authority_path)
                      == self._attempt_authority_identity,
                      "native attempt authority file changed during child acknowledgement")
                return result

            result = coordinator.execute_with_task_claim_fence(
                identity, fenced, minimum_remaining_ms=int(ACK_TIMEOUT_SECONDS * 1000))
        self._attempt_fence_boundaries.append(self._boundary)
        return result

    @staticmethod
    def _attempt_ack_budget(lease):
        # This immutable lease comes from the actual guarded native SQL
        # transaction. Same-fence renewal cannot change it until the ACK closes.
        _need(lease.expires_at_ms > time.time_ns() // 1_000_000
              + int(ACK_TIMEOUT_SECONDS * 1000),
              "native attempt lease has no full child acknowledgement budget")

    def _worktree_current(self, request, *, claim, peer_identity):
        runtime = self._runtime
        root = Path(runtime.manifest.get("worker_worktree_root") or runtime.state / "worktrees")
        path = Path(request["worktree"])
        _need(root.is_absolute() and root.resolve() == root and path.is_absolute()
              and path.resolve() == path and path != root and path.is_relative_to(root)
              and path.is_dir() and not path.is_symlink(),
              "dispatch requires an exact allocated native worktree")
        marker = path / ".git"
        metadata = marker.lstat()
        _need(stat.S_ISREG(metadata.st_mode) and metadata.st_uid == os.geteuid()
              and metadata.st_nlink == 1, "dispatch requires an owner allocated Git worktree marker")
        from .finite_proof_query_worktree_allocation import (
            FrozenFiniteProofQueryWorktreeAllocation,
            _capture_owner_finite_proof_query_worktree_allocation,
        )
        from .finite_proof_query_admission import _budget
        checkpoint = _budget(self._closure._owner)
        if self._allocation is None:
            self._allocation = _capture_owner_finite_proof_query_worktree_allocation(
                broker=self, request=request, claim=claim, peer_identity=peer_identity,
                checkpoint=checkpoint)
            self._worker_source_custody = self._worker_source_baseline.bind_allocation(
                allocation=self._allocation, checkpoint=checkpoint)
            self._bound_worker_allocation = (self._allocation, self._worker_source_custody)
        _need(type(self._allocation) is FrozenFiniteProofQueryWorktreeAllocation
              and self._allocation._scope is self._scope
              and self._allocation.material_binding["worktree_path"] == request["worktree"],
              "native dispatch allocation differs from its registered request")
        FrozenFiniteProofQueryWorktreeAllocation.require_current(
            self._allocation, checkpoint=checkpoint)

    def _serve(self):
        while not self._stop.is_set():
            try:
                channel, _ = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            # One bounded connection at a time. Unauthenticated peers never
            # acquire either native owner lock before the closed frame/peer check.
            with channel:
                channel.settimeout(ACK_TIMEOUT_SECONDS)
                try:
                    self._stage, self._boundary = "receive", "unparsed"
                    self._birth_ack_pending = False
                    self._handoff(channel)
                except Exception as error:
                    self._observe({"schema": SCHEMA, "status": "refused",
                                   "error_type": type(error).__name__, "worker_birth_observed": (
                                       self._isolated_spawned if self._boundary == "isolated_worker"
                                       else self._spawned),
                                   "boundary": self._boundary, "stage": self._stage,
                                   "message_sha256": hashlib.sha256(str(error).encode("utf-8")).hexdigest(),
                                   "unacknowledged_child_birth_possible": self._birth_ack_pending})
                    try:
                        _send(channel, {"schema": SCHEMA, "ok": False, "error": "native_dispatch_refused"})
                    except (OSError, RuntimeError):
                        pass

    def _handoff(self, channel):
        raw = _receive(channel)
        peer = typed._kernel_peer_identity(channel)
        _need(peer[1] == os.geteuid(), "dispatch peer UID differs from the native owner")
        if raw.get("operation") == "prepare_isolated":
            return self._isolated_handoff(channel, _isolated_request(raw), peer)
        request = _request(raw)
        self._boundary = "native_daemon_to_owner_wrapper"
        with self._lock:
            _need(not self._pending and not self._spawned, "native coding dispatch is already consumed")
            self._pending = True
        try:
            def close_handoff(guarded_claim, guarded_lease):
                with self._server._lock, self._catalog.store._lock:
                    self._identity_current()
                    self._stage = "native_claim"
                    _need(request["context_cid"] == self._context and request["command"] == self._command,
                          "dispatch command or public context differs from the signed native candidate")
                    claim = _require_native_dispatch_claim(server=self._server, expected_population=self._population,
                        request=request, peer_identity=peer, client_id=self._runtime.client_id,
                        store_id=self._runtime.owner_identity.store_id)
                    _need(_same_native_dispatch_claim(claim, guarded_claim),
                          "native claim differs from its guarded attempt authority")
                    self._claim_checked = True
                    self._bootstrap_current(claim)
                    self._stage = "native_allocation"
                    self._worktree_current(request, claim=claim, peer_identity=peer)
                    self._run_lease_current()
                    self._stage = "proof_observer"
                    self._closure.require_worker_dispatch_current(
                        scope=self._scope, worker_source_custody=self._worker_source_custody)
                    # All potentially observing work precedes this final detached
                    # native claim and identity recheck; locks remain held through ACK.
                    self._identity_current()
                    final = _require_native_dispatch_claim(server=self._server, expected_population=self._population,
                        request=request, peer_identity=typed._kernel_peer_identity(channel),
                        client_id=self._runtime.client_id, store_id=self._runtime.owner_identity.store_id)
                    _need(_same_native_dispatch_claim(final, claim),
                          "native dispatch claim changed during closing custody")
                    claim = final
                    self._bootstrap_current(final)
                    self._worktree_current(request, claim=final, peer_identity=peer)
                    self._run_lease_current()
                    self._stage = "detached_proof_close"
                    self._final_proof_close()
                    final = _require_native_dispatch_claim(server=self._server, expected_population=self._population,
                        request=request, peer_identity=typed._kernel_peer_identity(channel),
                        client_id=self._runtime.client_id, store_id=self._runtime.owner_identity.store_id)
                    _need(_same_native_dispatch_claim(final, claim),
                        "native dispatch claim or active grant changed after final proof custody")
                    claim = final
                    _need(self._run_lease_expiry > time.time_ns() // 1_000_000
                          + int(ACK_TIMEOUT_SECONDS * 1000), "native dispatch run lease expired during proof custody")
                    self._stage = "grant_ack_budget"
                    grant_ack_expiry = _require_native_grant_ack_budget(server=self._server, claim=claim)
                    self._grant_ack_budget_checked = True
                    nonce = uuid.uuid4().hex
                    self._stage = "actual_wrapper_birth_ack"
                    self._attempt_ack_budget(guarded_lease)
                    self._birth_ack_pending = True
                    _send(channel, {"schema": SCHEMA, "ok": True, "phase": "prepared", "handoff_id": nonce})
                    acknowledgement = _receive(channel)
                    _need(set(acknowledgement) in ({"schema", "operation", "handoff_id"},
                            {"schema", "operation", "handoff_id", "pid", "process_birth_id"})
                          and acknowledgement.get("schema") == SCHEMA
                          and acknowledgement.get("handoff_id") == nonce,
                          "dispatch child acknowledgement differs from its closed handoff")
                    if acknowledgement.get("operation") == "abort":
                        _need(set(acknowledgement) == {"schema", "operation", "handoff_id"},
                              "dispatch abort contains unexpected fields")
                        status = {"schema": SCHEMA, "status": "aborted", "worker_birth_observed": False}
                    else:
                        _need(acknowledgement.get("operation") == "spawned"
                              and type(acknowledgement.get("pid")) is int and acknowledgement["pid"] > 0,
                              "actual dispatch child PID required")
                        child = acknowledgement["pid"]
                        start, parent, boot = typed._process_runtime_facts(child)
                        birth = typed._process_birth_content_id(child, start, boot, parent)
                        _need(parent == peer[0] and birth == acknowledgement.get("process_birth_id"),
                              "dispatch child birth differs from its connecting native daemon")
                        observed = _require_wrapper_kernel_command(
                            child=child, command=self._command, worktree=request["worktree"])
                        _need(observed["process_birth_id"] == birth,
                              "native wrapper birth changed during kernel command observation")
                        # Source is intentionally never read here: a real child may
                        # already have performed its authorized candidate edit.
                        self._spawned = True
                        self._wrapper = {"ticket": nonce, "request": request, "claim": claim,
                                         "pid": child, "process_birth_id": birth,
                                         "start_time_ticks": start, "daemon_peer": peer}
                        status = {"schema": SCHEMA, "status": "spawned", "worker_birth_observed": True,
                                  "pid": child, "process_birth_id": birth, "daemon_pid": peer[0],
                                  "task_cid": claim["task_cid"], "task_revision": claim["task_revision"],
                                  "attempt_id": claim["attempt_id"], "claim_id": claim["claim_id"],
                                  "context_cid": self._context, "boundary": "native_daemon_to_owner_wrapper",
                                  "isolated_uid_observed": False, "proof_authority": False,
                                  "completion_authority": False, "convergence_proved": False}
                return status, nonce, grant_ack_expiry

            status, nonce, grant_ack_expiry = self._execute_with_attempt_fence(
                request=request, daemon_peer=peer, callback=close_handoff)
            # Release native attempt, owner and proof locks before success ACK;
            # daemon's on_started callback, which uses the same typed gateway.
            self._birth_ack_pending = False
            status.update(allocation_cid=self._allocation.material_binding["allocation_cid"],
                          native_worktree_lease_checked=True, grant_ACK_budget_checked=True,
                          independent_live_attempt_lease_checked=True,
                          grant_expires_at_ms=grant_ack_expiry)
            self._observe(status)
            _send(channel, {"schema": SCHEMA, "ok": True, "phase": "released", "handoff_id": nonce})
        finally:
            with self._lock:
                self._pending = False

    def _final_proof_close(self):
        """Detached original producer fence after every public observer callback."""
        from dataclasses import replace
        from .finite_proof_query_execution import FrozenFiniteProofQueryExecutionClosure
        from . import finite_proof_query_admission as admission
        from ..planning import finite_proof_query_join as join
        closure = self._closure
        checkpoint = admission._budget(closure._owner)
        # Calling the owned class producer avoids treating a replacement
        # instance observer's return as evidence of a successful close.
        FrozenFiniteProofQueryExecutionClosure._physical_close(
            closure, scope=self._scope, checkpoint=checkpoint, ready_population=False,
            worker_source_custody=self._worker_source_custody)
        join._require_frozen_proof_inventory(owner=replace(closure._owner, timeout_seconds=checkpoint()),
            verification_catalog=closure._catalog,
            closure=json.loads(closure._value)["indexed_plan"]["proof_query_closure"],
            checkpoint=checkpoint)

    def _isolated_handoff(self, channel, request, peer):
        self._boundary, self._stage = "isolated_worker", "registered_wrapper"
        with self._lock:
            _need(not self._pending and self._spawned and not self._isolated_spawned
                  and self._wrapper is not None, "registered native wrapper dispatch required")
            self._pending = True
        try:
            wrapper = self._wrapper
            def close_handoff(guarded_claim, guarded_lease):
                with self._server._lock, self._catalog.store._lock:
                    self._identity_current()
                    wrapper = self._wrapper
                    start, parent, boot = typed._process_runtime_facts(peer[0])
                    _need(peer[0] == wrapper["pid"] and peer[2] == wrapper["start_time_ticks"]
                          and parent == wrapper["daemon_peer"][0]
                          and typed._process_birth_content_id(peer[0], start, boot, parent)
                          == wrapper["process_birth_id"]
                          and request["wrapper_ticket"] == wrapper["ticket"]
                          and request["context_cid"] == self._context
                          and request["worktree"] == wrapper["request"]["worktree"],
                          "isolated dispatch differs from its registered exact wrapper birth")
                    expected_command = ["/usr/bin/sudo", "-n", "-u", "benchmarkworker", "--",
                        str(Path(self._command[0]).parent / "worker-entry"), *self._command[1:]]
                    _need(request["command"] == expected_command,
                          "isolated dispatch must use the exact native UID transition and worker command")
                    daemon_peer = wrapper["daemon_peer"]
                    _need(typed._process_start_time_ticks(daemon_peer[0]) == daemon_peer[2],
                          "registered wrapper's native claiming daemon is unavailable")
                    claim = _require_native_dispatch_claim(server=self._server, expected_population=self._population,
                        request=wrapper["request"], peer_identity=daemon_peer, client_id=self._runtime.client_id,
                        store_id=self._runtime.owner_identity.store_id)
                    _need(_same_native_dispatch_claim(claim, guarded_claim),
                          "native claim differs from its guarded attempt authority")
                    _need(_same_native_dispatch_claim(claim, wrapper["claim"]),
                          "delegated wrapper native claim changed")
                    self._bootstrap_current(claim)
                    self._stage = "native_allocation"
                    self._worktree_current(wrapper["request"], claim=claim, peer_identity=daemon_peer)
                    self._run_lease_current()
                    self._stage = "proof_observer"
                    self._closure.require_worker_dispatch_current(
                        scope=self._scope, worker_source_custody=self._worker_source_custody)
                    self._identity_current()
                    final = _require_native_dispatch_claim(server=self._server, expected_population=self._population,
                        request=wrapper["request"], peer_identity=daemon_peer, client_id=self._runtime.client_id,
                        store_id=self._runtime.owner_identity.store_id)
                    _need(_same_native_dispatch_claim(final, claim),
                          "isolated dispatch native claim changed during proof custody")
                    claim = final
                    self._bootstrap_current(final)
                    self._worktree_current(wrapper["request"], claim=final, peer_identity=daemon_peer)
                    self._run_lease_current()
                    self._stage = "detached_proof_close"
                    self._final_proof_close()
                    final = _require_native_dispatch_claim(server=self._server, expected_population=self._population,
                        request=wrapper["request"], peer_identity=daemon_peer, client_id=self._runtime.client_id,
                        store_id=self._runtime.owner_identity.store_id)
                    _need(_same_native_dispatch_claim(final, claim),
                        "isolated dispatch active native claim changed after final proof custody")
                    claim = final
                    _need(self._run_lease_expiry > time.time_ns() // 1_000_000
                          + int(ACK_TIMEOUT_SECONDS * 1000), "isolated dispatch run lease expired during proof custody")
                    self._stage = "grant_ack_budget"
                    grant_ack_expiry = _require_native_grant_ack_budget(server=self._server, claim=claim)
                    self._grant_ack_budget_checked = True
                    nonce = uuid.uuid4().hex
                    self._stage = "actual_isolated_birth_ack"
                    self._attempt_ack_budget(guarded_lease)
                    self._birth_ack_pending = True
                    _send(channel, {"schema": SCHEMA, "ok": True, "phase": "prepared", "handoff_id": nonce})
                    acknowledgement = _receive(channel)
                    _need(acknowledgement.get("schema") == SCHEMA
                          and acknowledgement.get("handoff_id") == nonce,
                          "isolated child acknowledgement differs from its native handoff")
                    if acknowledgement.get("operation") == "abort":
                        _need(set(acknowledgement) == {"schema", "operation", "handoff_id"},
                              "isolated dispatch abort contains unexpected fields")
                        status = {"schema": SCHEMA, "status": "aborted", "boundary": "isolated_worker",
                                  "worker_birth_observed": False}
                    else:
                        _need(set(acknowledgement) == {"schema", "operation", "handoff_id", "pid",
                              "process_birth_id", "worker_pid", "worker_process_birth_id"}
                              and acknowledgement["operation"] == "spawned"
                              and type(acknowledgement["pid"]) is int,
                              "actual isolated native child birth acknowledgement required")
                        observed = _isolated_kernel_birth(process_pid=acknowledgement["pid"], wrapper_pid=peer[0],
                            command=expected_command, reported_pid=acknowledgement["worker_pid"])
                        _need(observed["sudo"]["process_birth_id"] == acknowledgement["process_birth_id"]
                              and observed["worker"]["process_birth_id"]
                              == acknowledgement["worker_process_birth_id"],
                              "isolated worker acknowledgement differs from owner kernel observations")
                        self._isolated_spawned = True
                        status = {"schema": SCHEMA, "status": "spawned", "boundary": "isolated_worker",
                                  "worker_birth_observed": True, "isolated_uid_observed": True,
                                  "worker_pid": observed["worker"]["pid"], "worker_uid": 1001,
                                  "worker_process_birth_id": observed["worker"]["process_birth_id"],
                                  "sudo_pid": observed["sudo"]["pid"], "wrapper_pid": peer[0],
                                  "parent_chain": observed["parent_chain"], "task_cid": claim["task_cid"],
                                  "task_revision": claim["task_revision"], "attempt_id": claim["attempt_id"],
                                  "claim_id": claim["claim_id"], "context_cid": self._context,
                                  "proof_authority": False, "completion_authority": False,
                                  "convergence_proved": False}
                return status, nonce, grant_ack_expiry

            status, nonce, grant_ack_expiry = self._execute_with_attempt_fence(
                request=wrapper["request"], daemon_peer=wrapper["daemon_peer"], callback=close_handoff)
            self._birth_ack_pending = False
            status.update(allocation_cid=self._allocation.material_binding["allocation_cid"],
                          native_worktree_lease_checked=True, grant_ACK_budget_checked=True,
                          independent_live_attempt_lease_checked=True,
                          grant_expires_at_ms=grant_ack_expiry)
            self._observe(status)
            _send(channel, {"schema": SCHEMA, "ok": True, "phase": "released", "handoff_id": nonce})
        finally:
            with self._lock:
                self._pending = False

    def close_after_native_stop(self):
        """Close transport only after native STOP and isolated cleanup.

        Shutdown must remain available after source edits or cancellation; no
        proof, source or freshness condition is evaluated here.
        """
        _need(type(self) is OwnerFiniteProofQueryDispatchBroker and self._seal is _SEAL
              and self._scope._runtime is self._runtime, "exact native dispatch broker required for close")
        if self._closed:
            return json.loads(canonical_json_bytes(self._close_receipt))
        self._scope.require_close(self._runtime)
        _need(self._runtime._context_refresh_stopped(),
              "dispatch broker close requires native STOP and isolated cleanup")
        self._close_transport()
        self._close_receipt = self._closed_receipt(unlaunched=False)
        return json.loads(canonical_json_bytes(self._close_receipt))

    def close_unlaunched_construction(self):
        """Release transport after a construction failure before any native birth.

        This is not a STOP or worker-cleanup receipt. The enclosing native
        constructor remains responsible for its unlaunched run reservation.
        """
        _need(type(self) is OwnerFiniteProofQueryDispatchBroker and self._seal is _SEAL
              and self._scope._runtime is self._runtime and not self._scope._spawned
              and not self._spawned and not self._isolated_spawned and not self._pending
              and not getattr(self._runtime, "_children", ()),
              "unlaunched broker cleanup requires no native or delegated process birth")
        process, profile = getattr(self._runtime, "process", None), getattr(self._runtime, "profile", None)
        if process is not None and profile is not None:
            _need(not process.snapshot(profile).members,
                  "unlaunched broker cleanup found a native process tree")
        if not self._closed:
            self._close_transport()
            self._close_receipt = self._closed_receipt(unlaunched=True)
        return json.loads(canonical_json_bytes(self._close_receipt))

    def _closed_receipt(self, *, unlaunched):
        return {"schema": SCHEMA, "broker_closed": True, "socket_removed": True, "thread_joined": True,
                "native_STOP_receipt_expected": not unlaunched, "native_STOP_proved": False,
                "isolated_cleanup_checked": not unlaunched, "unlaunchedresources_disposed": unlaunched,
                "observations": self.observations, "observation_count": len(self._observations),
                "dropped_observation_count": self._dropped_observations,
                "independent_live_attempt_lease_checked": self._attempt_fence_boundaries == [
                    "native_daemon_to_owner_wrapper", "isolated_worker"],
                "native_attempt_fence_boundaries": list(self._attempt_fence_boundaries),
                "current_native_claim_and_active_grant_checked": self._claim_checked,
                "native_runtime_run_lease_checked": self._runtime_lease_checked,
                "native_grant_ACK_budget_checked": self._grant_ack_budget_checked,
                "native_worktree_lease_checked": self._allocation is not None,
                "proof_authority": False, "completion_authority": False,
                "task_omission_authority": False, "convergence_proved": False}

    def _close_transport(self):
        self._stop.set()
        self._listener.close()
        self._thread.join(timeout=ACK_TIMEOUT_SECONDS + 1)
        _need(not self._thread.is_alive(), "native dispatch handoff has not closed safely")
        # Cleanup names only the originally owned inode. Cached proof inputs,
        # public endpoint aliases and a stale source are not shutdown conditions.
        endpoint = self._original_socket_path
        _need(endpoint.parent.resolve() == endpoint.parent,
              "native dispatch original socket cleanup parent is not canonical")
        metadata = endpoint.lstat()
        _need((metadata.st_dev, metadata.st_ino) == self._socket_identity
              and stat.S_ISSOCK(metadata.st_mode) and metadata.st_uid == os.geteuid(),
              "native dispatch original socket identity changed before cleanup")
        endpoint.unlink()
        self._closed = True


def start_owner_finite_proof_query_dispatch_broker(*, scope, runtime):
    return OwnerFiniteProofQueryDispatchBroker(_SEAL, scope=scope, runtime=runtime)


class _DispatchGuard:
    def __init__(self, channel, handoff_id, *, isolated_command=None):
        self._channel, self._handoff_id, self._closed = channel, handoff_id, False
        self._isolated_command = isolated_command

    def child_environment(self):
        _need(not self._closed and self._isolated_command is None,
              "only a pending native wrapper birth has a delegated routing ticket")
        return {WRAPPER_ENV: self._handoff_id}

    def _finish(self, value):
        _need(not self._closed, "native dispatch guard already consumed")
        try:
            _send(self._channel, value)
            response = _receive(self._channel)
            _need(response.get("ok") is True
                  and response == {"schema": SCHEMA, "ok": True, "phase": "released",
                               "handoff_id": self._handoff_id},
                  "native dispatch locks were not acknowledged as released")
        finally:
            self._closed = True
            self._channel.close()

    def note_spawned(self, process):
        pid = process.pid
        _need(type(pid) is int and pid > 0, "actual native Popen child required")
        try:
            start, parent, boot = typed._process_runtime_facts(pid)
            _need(parent == os.getpid(), "dispatch child does not belong to the current daemon")
            birth = typed._process_birth_content_id(pid, start, boot, parent)
            response = {"schema": SCHEMA, "operation": "spawned", "handoff_id": self._handoff_id,
                        "pid": pid, "process_birth_id": birth}
            if self._isolated_command is not None:
                deadline = time.monotonic() + ACK_TIMEOUT_SECONDS - 1
                while True:
                    try:
                        observed = _isolated_kernel_birth(process_pid=pid, wrapper_pid=os.getpid(),
                                                         command=self._isolated_command)
                        break
                    except (OSError, RuntimeError):
                        _need(time.monotonic() < deadline,
                              "genuine isolated UID1001 child birth was not observed within its bound")
                        time.sleep(0.01)
                response.update(worker_pid=observed["worker"]["pid"],
                                worker_process_birth_id=observed["worker"]["process_birth_id"])
            self._finish(response)
        except BaseException:
            # A post-birth failure cannot safely turn into an ordinary aborted
            # prelaunch attempt. The caller must stop/reap this actual child.
            self._closed = True
            self._channel.close()
            raise

    def abort(self):
        if not self._closed:
            self._finish({"schema": SCHEMA, "operation": "abort", "handoff_id": self._handoff_id})


def require_finite_proof_query_worker_dispatch(*, command, worktree, environment, dispatch_binding):
    """Called after native env/FD normalization and immediately before Popen.

    Ordinary commands remain unaffected. Any proof-query argument makes all
    three context arguments, owner routing markers and native binding mandatory.
    The socket remains open through the actual Popen acknowledgement.
    """
    argv = _command([str(item) for item in command])
    flags = ("--finite-proof-query-context", "--finite-proof-query-sha256",
             "--finite-proof-query-context-cid")
    present = any(flag in argv or any(item.startswith(flag + "=") for item in argv) for flag in flags)
    if not present:
        return None
    values = {}
    for flag in flags:
        _need(argv.count(flag) == 1 and not any(item.startswith(flag + "=") for item in argv),
              "exact complete proof-query context arguments required at coding dispatch")
        position = argv.index(flag)
        _need(position + 1 < len(argv), "proof-query context argument value absent")
        values[flag] = _text(argv[position + 1])
    _need(type(dispatch_binding) is dict and _BINDING_FIELDS.issubset(dispatch_binding),
          "native database attempt authority is required at coding dispatch")
    context = _text(environment.get(CONTEXT_ENV))
    endpoint = Path(_text(environment.get(SOCKET_ENV)))
    _need(context == values[flags[2]] and endpoint.is_absolute()
          and endpoint.parent.resolve() == endpoint.parent,
          "native dispatch routing differs from the literal proof-query context")
    metadata = endpoint.lstat()
    _need(stat.S_ISSOCK(metadata.st_mode) and metadata.st_uid == os.geteuid()
          and stat.S_IMODE(metadata.st_mode) == 0o600,
          "exact private owner dispatch socket required")
    request = _request({"schema": SCHEMA, "operation": "prepare", "context_cid": context,
        "command": argv, "worktree": str(worktree),
        **{name: dispatch_binding[name] for name in _BINDING_FIELDS}})
    channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    channel.settimeout(REQUEST_TIMEOUT_SECONDS)
    try:
        channel.connect(str(endpoint))
        _need(typed._kernel_peer_identity(channel)[1] == os.geteuid(),
              "dispatch owner peer UID differs")
        _send(channel, request)
        response = _receive(channel)
        _need(set(response) == {"schema", "ok", "phase", "handoff_id"}
              and response["schema"] == SCHEMA and response["ok"] is True
              and response["phase"] == "prepared", "native owner refused coding dispatch")
        return _DispatchGuard(channel, _text(response["handoff_id"], limit=64))
    except BaseException:
        channel.close()
        raise


def require_finite_proof_query_isolated_worker_dispatch(*, command, worktree, environment):
    """The registered owner wrapper gates the literal sudo-to-worker Popen."""
    argv = _command([str(item) for item in command])
    if "--finite-proof-query-context" not in argv:
        _need(not any("finite-proof-query" in item for item in argv),
              "complete isolated proof-query context required")
        return None
    _need(argv[:5] == ["/usr/bin/sudo", "-n", "-u", "benchmarkworker", "--"],
          "exact isolated worker UID transition required")
    context = _text(environment.get(CONTEXT_ENV))
    endpoint = Path(_text(environment.get(SOCKET_ENV)))
    ticket = _text(environment.get(WRAPPER_ENV), limit=64)
    _need(endpoint.is_absolute() and endpoint.parent.resolve() == endpoint.parent,
          "exact native isolated dispatch endpoint required")
    metadata = endpoint.lstat()
    _need(stat.S_ISSOCK(metadata.st_mode) and metadata.st_uid == os.geteuid()
          and stat.S_IMODE(metadata.st_mode) == 0o600, "private isolated dispatch owner socket required")
    request = _isolated_request({"schema": SCHEMA, "operation": "prepare_isolated", "command": argv,
        "worktree": str(worktree), "context_cid": context, "wrapper_ticket": ticket})
    channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    channel.settimeout(REQUEST_TIMEOUT_SECONDS)
    try:
        channel.connect(str(endpoint))
        _need(typed._kernel_peer_identity(channel)[1] == os.geteuid(), "isolated dispatch owner UID differs")
        _send(channel, request)
        response = _receive(channel)
        _need(set(response) == {"schema", "ok", "phase", "handoff_id"}
              and response["schema"] == SCHEMA and response["ok"] is True
              and response["phase"] == "prepared", "native owner refused isolated coding dispatch")
        return _DispatchGuard(channel, _text(response["handoff_id"], limit=64), isolated_command=argv)
    except BaseException:
        channel.close()
        raise


__all__ = ["SOCKET_ENV", "CONTEXT_ENV", "WRAPPER_ENV", "SCHEMA", "FiniteProofQueryDispatchError",
           "OwnerFiniteProofQueryDispatchBroker", "start_owner_finite_proof_query_dispatch_broker",
           "require_finite_proof_query_worker_dispatch", "require_finite_proof_query_isolated_worker_dispatch"]
