"""Original-owner budget authorization for a checked native observer launch.

Clients import only the standard library. Owner-only native SQL/proof imports
are local to the broker; public CIDs, socket names and tickets confer no budget
or process authority. One reservation remains with the original scope until
its owner validates the actual observer receipt or conservatively aborts it.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import struct
import threading
import time
import uuid
import weakref

SOCKET_ENV = "IPFS_ACCELERATE_CHECKED_CONTEXT_DISPATCH_SOCKET"
CONTEXT_ENV = "IPFS_ACCELERATE_CHECKED_CONTEXT_DISPATCH_CID"
OWNER_ENV = "IPFS_ACCELERATE_CHECKED_CONTEXT_OWNER_BIRTH"
WRAPPER_ENV = "IPFS_ACCELERATE_CHECKED_CONTEXT_WRAPPER_TICKET"
OWNER_FLAG = "--checked-context-owner-scope-cid"
SCHEMA = "supervisor-checked-context-dispatch@1"
MAX_BYTES = 1_048_576
MAX_INPUT_BYTES = 524_288
MAX_COMMAND_PARTS = 64
REQUEST_TIMEOUT_SECONDS = 95.0
ACK_TIMEOUT_SECONDS = 5.0
_SEAL = object()
_BROKERS = {}
_GUARDS = weakref.WeakValueDictionary()
_REGISTRY_LOCK = threading.RLock()
_BINDING_FIELDS = frozenset(("task_id", "database_task_cid", "database_attempt_id",
                            "database_claim_id", "database_attempt_number"))
_REQUEST_FIELDS = frozenset(("schema", "operation", "context_cid", "command", "worktree", "input_text", *_BINDING_FIELDS))
_ISOLATED_FIELDS = frozenset(("schema", "operation", "context_cid", "command", "worktree", "wrapper_ticket"))
_CHECKED_FLAGS = ("--checked-context-artifact", "--checked-context-sha256", "--checked-context-task-cid")
_FALSE = dict.fromkeys(("proof_authority", "execution_authority", "completion_authority",
                        "provider_usage_measured", "convergence_proved"), False)


class CheckedContextDispatchError(RuntimeError):
    """A checked native launch was refused or its birth remains uncertain."""


def _need(value, message):
    if not value:
        raise CheckedContextDispatchError(message)


def _wire(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise CheckedContextDispatchError("finite bounded canonical JSON required") from error


def _text(value, *, limit=4096):
    _need(type(value) is str and 0 < len(value.encode("utf-8")) <= limit
          and not any(c in value for c in ("\x00", "\n", "\r")), "bounded literal dispatch identity required")
    return value


def _command(value):
    _need(type(value) is list and 1 <= len(value) <= MAX_COMMAND_PARTS, "bounded literal dispatch command required")
    return [_text(part, limit=8192) for part in value]


def _input(value):
    _need(type(value) is str, "complete actual native stdin must be a string")
    try:
        raw = value.encode("utf-8")
    except UnicodeError as error:
        raise CheckedContextDispatchError("native stdin must be UTF-8") from error
    _need(0 < len(raw) <= MAX_INPUT_BYTES, "complete native stdin exceeds dispatch bound")
    return raw


def _digest(value):
    _need(type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value), "exact SHA-256 required")
    return value


def _send(channel, value):
    body = _wire(value)
    _need(2 <= len(body) <= MAX_BYTES, "dispatch frame exceeds byte bound")
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
    _need(2 <= size <= MAX_BYTES, "dispatch frame size exceeds byte bound")
    raw = _exact(channel, size)
    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result, "duplicate dispatch field refused")
            result[key] = value
        return result
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except (UnicodeError, ValueError, RecursionError) as error:
        raise CheckedContextDispatchError("malformed dispatch frame refused") from error
    _need(type(value) is dict and _wire(value) == raw, "one canonical closed dispatch object required")
    return value


def _request(value):
    _need(type(value) is dict and set(value) == _REQUEST_FIELDS and value["schema"] == SCHEMA
          and value["operation"] == "prepare", "closed checked dispatch request required")
    for name in _REQUEST_FIELDS - {"command", "input_text", "database_attempt_number"}:
        _text(value[name])
    _command(value["command"]); _input(value["input_text"])
    _need(type(value["database_attempt_number"]) is int and 1 <= value["database_attempt_number"] < 2**32,
          "native attempt ordinal required")
    _owner_recipe(value["command"], value["context_cid"])
    return value


def _isolated_request(value):
    _need(type(value) is dict and set(value) == _ISOLATED_FIELDS and value["schema"] == SCHEMA
          and value["operation"] == "prepare_isolated", "closed isolated checked request required")
    for name in _ISOLATED_FIELDS - {"command"}:
        _text(value[name])
    _command(value["command"])
    return value


def _has_flag(command, flag):
    return any(part == flag or part.startswith(flag + "=") for part in command)


def _owner_recipe(command, context):
    command = _command(command)
    _need(command.count(OWNER_FLAG) == 1 and not any(part.startswith(OWNER_FLAG + "=") for part in command),
          "one exact checked owner scope discriminator required")
    position = command.index(OWNER_FLAG)
    _need(position + 1 < len(command) and command[position + 1] == context,
          "checked owner scope discriminator differs")
    _need(not any(_has_flag(command, flag) for flag in _CHECKED_FLAGS), "base command cannot preselect checked artifact")
    return command[:position] + command[position + 2:]


def _derived(command, base, context):
    command = _command(command); prefix = _owner_recipe(base, context)
    _need(len(command) == len(prefix) + 6 and command[:len(prefix)] == prefix
          and command[len(prefix)::2] == list(_CHECKED_FLAGS), "derived command differs from closed checked triplet recipe")
    artifact, digest, task = command[-5], command[-3], command[-1]
    _need(Path(artifact).is_absolute() and ".." not in Path(artifact).parts, "exact absolute checked artifact path required")
    _digest(digest); _text(task)
    return command


def _peer(channel, owner_marker=None):
    pid, uid, _gid = struct.unpack("3i", channel.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
    _need(pid > 0 and uid == os.geteuid(), "checked dispatch peer UID differs")
    if owner_marker is not None:
        identity = _owner_hint(owner_marker)
        _need(pid == identity["pid"] and _kernel_identity(pid,allow_parent_zero=True)["process_birth_id"] == identity["process_birth_id"],
              "socket peer differs from signed original owner birth")
    return pid, uid


def _owner_hint(value):
    _need(type(value) is str and len(value.encode("utf-8")) <= 256,"signed owner birth marker required")
    def pairs(items):
        result = {}
        for key,item in items:
            _need(key not in result,"duplicate original owner birth field")
            result[key] = item
        return result
    try: identity = json.loads(value,object_pairs_hook=pairs)
    except (ValueError,TypeError) as error: raise CheckedContextDispatchError("closed original owner birth required")from error
    _need(type(identity) is dict and set(identity) == {"pid","process_birth_id"}
          and type(identity["pid"]) is int and identity["pid"] > 0
          and type(identity["process_birth_id"]) is str and identity["process_birth_id"].startswith("birth:")
          and len(identity["process_birth_id"]) == 38 and _wire(identity).decode() == value,
          "canonical original owner birth marker required")
    return identity


def owner_dispatch_identity():
    """Public original owner hint; only the live kernel peer can satisfy it."""
    observed = _kernel_identity(os.getpid(),allow_parent_zero=True)
    return {"pid":observed["pid"],"process_birth_id":observed["process_birth_id"]}


def _kernel_identity(pid, *, allow_parent_zero=False):
    """Client-side kernel birth observation; imports no native owner modules."""
    _need(type(pid) is int and pid > 0, "actual Popen PID required")
    path = Path("/proc") / str(pid)
    raw = (path / "stat").read_text(encoding="utf-8")
    fields = raw[raw.rindex(")") + 1:].split()
    start, parent = int(fields[19]), int(fields[1])
    boot = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    _need(fields[0] != "Z" and start > 0 and (parent >= 0 if allow_parent_zero else parent > 0) and boot, "live kernel birth required")
    status = (path / "status").read_text(encoding="utf-8")
    uid = int(next(line for line in status.splitlines() if line.startswith("Uid:")).split()[2])
    # Same domain as typed_state_owner._process_birth_content_id.
    material = f"{pid}:{start}:{boot}:{parent}".encode()
    birth = "birth:" + hashlib.sha256(material).hexdigest()[:32]
    command = [part.decode("utf-8") for part in (path / "cmdline").read_bytes().split(b"\x00") if part]
    again = (path / "stat").read_text(encoding="utf-8")
    _need(int(again[again.rindex(")") + 1:].split()[19]) == start, "kernel PID was reused")
    return {"pid": pid, "uid": uid, "parent_pid": parent, "start_time_ticks": start,
            "process_birth_id": birth, "command": command}


def _endpoint(environment, context):
    _need(type(environment) is dict or hasattr(environment, "get"), "exact dispatch environment required")
    _need(environment.get(CONTEXT_ENV) == context, "checked routing context differs")
    endpoint = Path(_text(environment.get(SOCKET_ENV)))
    _need(endpoint.is_absolute() and endpoint.parent.resolve() == endpoint.parent, "canonical private dispatch endpoint required")
    parent, entry = endpoint.parent.lstat(), endpoint.lstat()
    _need(stat.S_ISDIR(parent.st_mode) and parent.st_uid == os.geteuid() and not parent.st_mode & 0o077
          and stat.S_ISSOCK(entry.st_mode) and entry.st_uid == os.geteuid()
          and stat.S_IMODE(entry.st_mode) == 0o600, "exact owner-private dispatch socket required")
    return endpoint


class _DispatchGuard:
    def __init__(self, channel, response, *, isolated=False):
        self._channel, self._handoff_id = channel, response["handoff_id"]
        self._command = tuple(response["command"])
        self._isolated, self._closed, self._pid = isolated, False, os.getpid()
        _need(type(self) is _DispatchGuard, "exact dispatch guard required")
        self._original = (channel,self._handoff_id,self._command,isolated,self._pid)
        with _REGISTRY_LOCK: _GUARDS[id(self)] = self

    def _assert(self):
        with _REGISTRY_LOCK:
            _need(type(self) is _DispatchGuard and _GUARDS.get(id(self)) is self
                  and not self._closed and os.getpid() == self._pid
                  and (self._channel,self._handoff_id,self._command,self._isolated,self._pid) == self._original,
                  "original unchanged pending dispatch guard required")

    @property
    def command(self):
        self._assert()
        return list(self._command)

    def child_environment(self):
        self._assert()
        _need(not self._isolated,"only original pending wrapper may receive routing ticket")
        return {WRAPPER_ENV: self._handoff_id}

    def _finish(self, value):
        self._assert()
        try:
            _send(self._channel, value)
            response = _receive(self._channel)
            _need(response == {"schema": SCHEMA, "ok": True, "phase": "released", "handoff_id": self._handoff_id},
                  "checked owner did not acknowledge release")
        finally:
            self._closed = True
            with _REGISTRY_LOCK: _GUARDS.pop(id(self),None)
            self._channel.close()

    def note_spawned(self, process):
        self._assert()
        try:
            observed = _kernel_identity(process.pid)
            _need(observed["parent_pid"] == os.getpid(), "Popen child belongs to another wrapper")
            self._finish({"schema": SCHEMA, "operation": "spawned", "handoff_id": self._handoff_id,
                          "pid": observed["pid"], "process_birth_id": observed["process_birth_id"]})
        except BaseException:
            self._closed = True
            with _REGISTRY_LOCK: _GUARDS.pop(id(self),None)
            self._channel.close()
            raise  # Caller must fence/reap any actual child; never refund.

    def abort(self):
        if not self._closed:
            self._finish({"schema": SCHEMA, "operation": "abort", "handoff_id": self._handoff_id})


def _connect(request, environment, *, isolated=False):
    endpoint = _endpoint(environment, request["context_cid"])
    channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    channel.settimeout(REQUEST_TIMEOUT_SECONDS)
    try:
        _owner_hint(environment.get(OWNER_ENV))
        channel.connect(str(endpoint)); _peer(channel,environment[OWNER_ENV]); _send(channel, request)
        response = _receive(channel)
        fields = {"schema", "ok", "phase", "handoff_id", "command", "input_sha256", "input_bytes"}
        _need(set(response) == fields and response["schema"] == SCHEMA and response["ok"] is True
              and response["phase"] == "prepared", "checked owner refused dispatch")
        _text(response["handoff_id"], limit=64); _digest(response["input_sha256"])
        _need(type(response["input_bytes"]) is int and 0 < response["input_bytes"] <= MAX_INPUT_BYTES,
              "prepared native stdin byte bound differs")
        if isolated:
            _need(response["command"] == request["command"], "isolated owner changed exact sudo command")
        else:
            _derived(response["command"], request["command"], request["context_cid"])
            _need(response["command"][-1] == request["database_task_cid"],"owner prepared another native task")
            raw = _input(request["input_text"])
            _need(response["input_sha256"] == hashlib.sha256(raw).hexdigest()
                  and response["input_bytes"] == len(raw), "owner prepared another native stdin")
        return _DispatchGuard(channel, response, isolated=isolated)
    except BaseException:
        channel.close(); raise


def require_checked_context_worker_dispatch(*, command, worktree, environment, dispatch_binding, input_text):
    argv = _command([str(part) for part in command])
    if not _has_flag(argv, OWNER_FLAG):
        _need(not environment.get(SOCKET_ENV) and not environment.get(CONTEXT_ENV) and not environment.get(OWNER_ENV),
              "signed checked dispatch routing cannot fall back to ordinary Popen")
        return None
    _need(type(dispatch_binding) is dict and _BINDING_FIELDS <= set(dispatch_binding), "native attempt binding required")
    context = _text(environment.get(CONTEXT_ENV)); _owner_recipe(argv, context)
    request = _request({"schema": SCHEMA, "operation": "prepare", "context_cid": context,
        "command": argv, "worktree": str(worktree), "input_text": input_text,
        **{name: dispatch_binding[name] for name in _BINDING_FIELDS}})
    return _connect(request, environment)


def require_checked_context_isolated_worker_dispatch(*, command, worktree, environment):
    argv = _command([str(part) for part in command])
    if not environment.get(SOCKET_ENV) and not environment.get(CONTEXT_ENV) and not environment.get(WRAPPER_ENV) and not environment.get(OWNER_ENV):
        return None
    _need(argv[:5] == ["/usr/bin/sudo", "-n", "-u", "benchmarkworker", "--"]
          and not _has_flag(argv, OWNER_FLAG), "exact isolated checked sudo command required")
    context = _text(environment.get(CONTEXT_ENV)); _text(environment.get(WRAPPER_ENV), limit=64)
    request = _isolated_request({"schema": SCHEMA, "operation": "prepare_isolated", "context_cid": context,
        "command": argv, "worktree": str(worktree), "wrapper_ticket": environment[WRAPPER_ENV]})
    return _connect(request, environment, isolated=True)


def _native_projection(broker, request, claim, checkpoint):
    """Rebuild the exact sealed Portal binding from the owner's current row."""
    from dataclasses import replace
    from ..task_sources.intent_repository import IntentRepository
    from ..task_sources.database_task_source import DatabaseTaskSource
    from ..task_sources.board_control_plane import infer_board_namespace
    from ..todo_daemon.database_portal_bridge import DatabasePortalExecutionBridge
    from ..todo_daemon.implementation_daemon import DatabaseImplementationDaemon, DatabaseTaskAttempt

    checkpoint()
    runtime = broker._runtime
    repository = Path(runtime.manifest["repository_root"])
    _head, raw = _owner_regular(repository / ".git" / "HEAD", checkpoint, bound=4096)
    prefix = "ref: refs/heads/"
    text = raw.decode("ascii").strip()
    _need(text.startswith(prefix) and len(text) > len(prefix),
          "native allocation requires the canonical named baseline branch")
    target = text[len(prefix):]
    source = DatabaseTaskSource(intent=IntentRepository(
        bound_connection=broker._server._connection, install_schema=False), install_schema=False)
    record = source.get_task(request["database_task_cid"])
    _need(record is not None and record.status == "in_progress"
          and record.revision == claim["task_revision"],
          "native allocation requires its current admitted task")
    receipt = record.body.get("completion_receipt", {})
    attempt = DatabaseTaskAttempt(task_cid=record.task_cid, task_alias=record.task_alias,
        **{key: receipt[key] for key in ("attempt_id", "claim_id", "attempt_number",
            "owner_session_id", "fencing_token", "fence_epoch", "lease_id")},
        committed_phase=receipt["attempt_execution_phase"], status="running",
        started_at_ms=0, revision=receipt["attempt_execution_revision"], body={})
    attempt = replace(attempt, body={"control_binding":
        DatabaseImplementationDaemon._control_claim_binding(attempt, record)})

    def no_provider(*_args):
        raise CheckedContextDispatchError("allocation custody cannot dispatch a provider")

    bridge = DatabasePortalExecutionBridge(task_source=source,
        attempt_root=Path(runtime.state) / "run" / "admitted_database_portal_attempts",
        portal_factory=no_provider, repo_root=repository,
        board_namespace=infer_board_namespace(merge_target_branch=target,
            todo_path=Path(broker._server.config.database_path), state_prefix="admitted"),
        merge_target_branch=target, task_header_prefix="## " + record.task_alias)
    paths = bridge._paths(attempt)
    bridge._seal_attempt_directory(paths, attempt_id=attempt.attempt_id, create=False)
    _owner_regular(paths.binding, checkpoint)
    binding = bridge._strict_binding(paths.binding)
    expected = bridge._binding(attempt, record, bridge._render_projection(attempt, record),
                               schema=binding["schema"])
    _need(binding == expected, "native allocation Portal binding differs from the native claim")
    _owner_regular(paths.task_projection, checkpoint)
    identity = bridge._prior_projection_identity(paths, binding)
    _need(identity["task_id"] == request["task_id"],
          "native allocation Portal identity differs from the native alias")
    return paths, binding, identity, target

def _owner_regular(path, checkpoint, *, bound=262144):
    checkpoint()
    path = Path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        _need(stat.S_ISREG(before.st_mode) and before.st_uid == os.geteuid()
              and before.st_nlink == 1 and 0 <= before.st_size <= bound,
              "native allocation requires bounded owner regular file")
        raw = bytearray()
        while len(raw) <= bound:
            chunk = os.read(fd, min(65536, bound + 1 - len(raw)))
            if not chunk: break
            raw.extend(chunk)
        def stamp(x):
            return (x.st_dev,x.st_ino,x.st_mode,x.st_uid,x.st_nlink,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
        _need(len(raw) == before.st_size and stamp(before) == stamp(os.fstat(fd)) == stamp(path.lstat()),
              "native allocation file changed during read")
        return stamp(before), bytes(raw)
    finally:
        os.close(fd)


class OwnerCheckedContextDispatchBroker:
    """One original owner/runtime session; copying, subclassing and forking refuse."""

    def __init__(self, seal, *, scope, runtime):
        from ..entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
        from .checked_observer_budget import CheckedObserverBudgetScope
        from . import finite_proof_query_worker_dispatch as native
        _need(seal is _SEAL and type(self) is OwnerCheckedContextDispatchBroker
              and type(scope) is CheckedObserverBudgetScope and type(runtime) is AdmittedBenchmarkRuntime
              and scope.bound_runtime is runtime, "exact original checked scope and native runtime required")
        runtime._verify()
        self._scope, self._runtime, self._native = scope, runtime, native
        self._server = runtime.server
        self._seal, self._pid = seal, os.getpid()
        self._owner_identity = owner_dispatch_identity()
        self._owner_wire = _wire(self._owner_identity).decode()
        self._context, self._material = scope.context_cid, _wire(scope.material_binding)
        self._manifest, self._manifest_id = _wire(runtime.manifest), runtime.manifest_id
        _need(runtime.manifest.get("checked_observer_budget_scope") == scope.material_binding,
              "signed runtime differs from checked scope binding")
        self._state_path = Path(runtime.state)
        state = self._state_path.lstat()
        _need(self._state_path.is_absolute() and self._state_path.resolve() == self._state_path
              and stat.S_ISDIR(state.st_mode) and state.st_uid == os.geteuid() and not state.st_mode & 0o077,
              "checked broker requires exact private owner state directory")
        self._state_identity = (state.st_dev,state.st_ino,state.st_uid,stat.S_IMODE(state.st_mode))
        self.socket_path = self._state_path / "checked-context.sock"
        self._original_socket_path = self.socket_path
        _need(len(os.fsencode(self.socket_path)) < 108, "checked socket path exceeds kernel limit")
        env_items = runtime.manifest["environment"]
        _need(type(env_items) is list and len({x[0]for x in env_items}) == len(env_items), "unique signed environment required")
        environment = dict(env_items)
        _need(environment.get(SOCKET_ENV) == str(self.socket_path) and environment.get(CONTEXT_ENV) == self._context and environment.get(OWNER_ENV) == self._owner_wire,
              "signed checked endpoint or context differs")
        self._fields = (scope,runtime,self._server,self._server._connection,self._server._lock,
                        runtime.owner_identity,runtime.client_id,self._manifest,self._material,self._context)
        with self._server._lock:
            self._population = native.execution._physical_native(self._server._connection)
            self._population["selected_task_cids"] = [scope.task_cid]
        self._population_bytes = _wire(self._population)
        selected = [x for x in self._population["tasks"] if x["task_cid"] == scope.task_cid]
        _need(len(selected) == 1 and selected[0]["status"] == "ready", "checked population must be captured before START claim")
        self._stop, self._lock = threading.Event(), threading.RLock()
        self._pending = self._consumed = self._wrapper_spawned = self._isolated_spawned = self._closed = False
        self._ambiguous = self._prepared_ack = False
        self._wrapper = self._prepared = self._allocation = None
        self._wrapper_wire = self._prepared_wire = None
        self._attempt_authority_path = self._attempt_authority_identity = None
        self._observations, self._attempt_fence_boundaries = [], []
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        created = None
        try:
            self._listener.bind(str(self.socket_path))
            entry = self.socket_path.lstat(); created = (entry.st_dev,entry.st_ino)
            os.chmod(self.socket_path,0o600); self._socket_identity = created
            self._listener.listen(4); self._listener.settimeout(0.2)
            with _REGISTRY_LOCK: _BROKERS[id(self)] = self
            self._thread = threading.Thread(target=self._serve,daemon=True,name="checked-context-owner-dispatch")
            self._thread.start()
        except BaseException:
            self._stop.set(); self._listener.close()
            with _REGISTRY_LOCK: _BROKERS.pop(id(self),None)
            if created is not None:
                entry = self.socket_path.lstat()
                _need(stat.S_ISSOCK(entry.st_mode) and (entry.st_dev,entry.st_ino) == created,
                      "new checked socket changed during failed construction")
                self.socket_path.unlink()
            raise

    def _identity_current(self):
        with _REGISTRY_LOCK:
            _need(type(self) is OwnerCheckedContextDispatchBroker and self._seal is _SEAL
                  and self._pid == os.getpid() and _BROKERS.get(id(self)) is self and not self._closed,
                  "original active checked broker required")
        _need(owner_dispatch_identity() == self._owner_identity,"original owner kernel birth changed")
        runtime, scope = self._runtime,self._scope
        current = (scope,runtime,runtime.server,runtime.server._connection,runtime.server._lock,
                   runtime.owner_identity,runtime.client_id,_wire(runtime.manifest),_wire(scope.material_binding),scope.context_cid)
        _need(all(a is b if n < 6 else a == b for n,(a,b)in enumerate(zip(current,self._fields)))
              and scope.bound_runtime is runtime and runtime.manifest_id == self._manifest_id
              and self.socket_path == self._original_socket_path and _wire(self._population) == self._population_bytes,
              "checked broker owner, manifest, scope or native baseline was rebound")
        _need((self._wrapper is None and self._wrapper_wire is None)or _wire(self._wrapper) == self._wrapper_wire,
              "registered wrapper birth or command was mutated")
        _need((self._prepared is None and self._prepared_wire is None)or _wire(self._prepared) == self._prepared_wire,
              "original checked preparation was mutated")
        state,entry = self._state_path.lstat(),self.socket_path.lstat()
        _need(self._state_path.resolve() == self._state_path
              and (state.st_dev,state.st_ino,state.st_uid,stat.S_IMODE(state.st_mode)) == self._state_identity
              and stat.S_ISSOCK(entry.st_mode) and entry.st_uid == os.geteuid()
              and stat.S_IMODE(entry.st_mode) == 0o600 and (entry.st_dev,entry.st_ino) == self._socket_identity,
              "checked original private endpoint changed")
        runtime._verify()

    def daemon_environment(self):
        self._identity_current()
        return {SOCKET_ENV:str(self.socket_path),CONTEXT_ENV:self._context,OWNER_ENV:self._owner_wire}

    @property
    def observations(self):
        with self._lock: return json.loads(_wire(self._observations))

    @property
    def worker_birth_observed(self): return self._isolated_spawned

    @property
    def wrapper_birth_observed(self): return self._wrapper_spawned

    @property
    def ambiguous_birth_possible(self): return self._ambiguous

    def _observe(self,value):
        with self._lock:
            _need(len(self._observations) < 64,"checked observation inventory exceeds bound")
            self._observations.append({**value,**_FALSE})

    def _claim(self,request,peer):
        native = self._native
        translated = {key:value for key,value in request.items()if key != "input_text"}
        translated.update(schema=native.SCHEMA,operation="prepare",command=_owner_recipe(request["command"],request["context_cid"]))
        return native._require_native_dispatch_claim(server=self._server,expected_population=self._population,
            request=translated,peer_identity=peer,client_id=self._runtime.client_id,store_id=self._runtime.owner_identity.store_id)

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

    def _worktree_current(self, request, claim, peer):
        from ..merge import worktree_lifecycle as lifecycle
        repository = Path(self._runtime.manifest["repository_root"])
        root = Path(self._runtime.manifest.get("worker_worktree_root") or self._runtime.state / "worktrees")
        path = Path(request["worktree"])
        _need(root.is_absolute() and root.resolve() == root and path.is_absolute() and path.resolve() == path
              and path != root and path.is_relative_to(root) and path.is_dir() and not path.is_symlink(),
              "exact allocated native worktree required")
        checkpoint = self._identity_current
        _owner_regular(path / ".git", checkpoint, bound=4096)
        paths,binding,identity,target = _native_projection(self,request,claim,checkpoint)
        store = lifecycle.WorktreeLifecycleStore(repo_root=repository,
            store_dir=repository / ".git" / "agent-worktree-lifecycle")
        _owner_regular(store.workspace_path_for(path),checkpoint)
        record = store._load_strict_workspace_record(path)
        prefix = (1 << 52) | (claim["attempt_number"] * (1 << 16))
        local_attempt = record.attempt - prefix
        birth = lifecycle.read_process_birth(peer[0])
        _need(record.state is lifecycle.WorkspaceLifecycleState.ACTIVE
              and record.record_id == record.compute_record_id() and record.task_id == request["task_id"]
              and record.canonical_task_cid == identity["canonical_task_cid"]
              and 1 <= local_attempt < (1 << 16) and record.attempt == prefix | local_attempt
              and record.workspace_path == str(path) and record.repo_root == str(repository)
              and record.state_dir == str(paths.root) and record.lane_id.startswith(str(paths.root) + ":")
              and record.merge_target == target and not record.terminal_reason
              and birth is not None and birth == record.owner and birth.start_time_ticks == peer[2]
              and os.stat("/proc/" + str(birth.pid)).st_uid == peer[1]
              and record.expires_at > time.time() + ACK_TIMEOUT_SECONDS,
              "worktree lifecycle task, lease or native owner birth differs")
        index = store.task_index_path_for(canonical_task_cid=record.canonical_task_cid,
                                         task_id=record.task_id,attempt=record.attempt)
        _owner_regular(index,checkpoint); store._require_exact_task_index(record,index_path=index)
        frozen = {"binding":binding,"portal_identity":identity,"lifecycle":record.to_dict()}
        if self._allocation is not None:
            before,after = json.loads(self._allocation),json.loads(_wire(frozen))
            old_record,new_record = before["lifecycle"],after["lifecycle"]
            for name in ("fence","updated_at","expires_at"):
                _need(new_record[name] >= old_record[name],"native worktree lease rolled back")
                old_record.pop(name); new_record.pop(name)
            _need(before == after,"native allocated worktree ownership changed")
        self._allocation = _wire(frozen)
        return frozen

    def _execute_with_attempt_fence(self,request,peer,callback):
        # Match native remote claim callbacks: COOR before SERVER.
        with self._server._lock:
            self._identity_current(); claim = self._claim(request,peer); self._bootstrap_current(claim)
        coordinator = self._attempt_authority(claim=claim,daemon_peer=peer)
        identity = {name:claim[name] for name in ("task_cid","claim_id","attempt_id","attempt_number",
                                               "lease_id","owner_session_id","fencing_token","fence_epoch")}
        with coordinator:
            def fenced(lease):
                result = callback(claim,lease)
                _need(self._attempt_authority_file_current(self._attempt_authority_path) == self._attempt_authority_identity,
                      "native attempt authority changed during acknowledgement")
                return result
            result = coordinator.execute_with_task_claim_fence(identity,fenced,
                minimum_remaining_ms=int(ACK_TIMEOUT_SECONDS * 1000))
        self._attempt_fence_boundaries.append(self._boundary)
        return result

    def _closing_authority(self, request, peer, captured, lease):
        self._identity_current()
        claim = self._claim(request,peer)
        _need(self._native._same_native_dispatch_claim(claim,captured),"native claim changed during checked preparation")
        self._bootstrap_current(claim); self._worktree_current(request,claim,peer); self._run_lease_current()
        self._native._require_native_grant_ack_budget(server=self._server,claim=claim)
        _need(lease.expires_at_ms > time.time_ns() // 1_000_000 + int(ACK_TIMEOUT_SECONDS * 1000),
              "native attempt has insufficient birth acknowledgement time")
        return claim

    def _prepared_response(self, nonce, command, raw):
        return {"schema":SCHEMA,"ok":True,"phase":"prepared","handoff_id":nonce,"command":command,
                "input_sha256":hashlib.sha256(raw).hexdigest(),"input_bytes":len(raw)}

    def _ack(self, channel, nonce):
        value = _receive(channel)
        _need(value.get("schema") == SCHEMA and value.get("handoff_id") == nonce,
              "checked acknowledgement belongs to another handoff")
        if value.get("operation") == "abort":
            _need(set(value) == {"schema","operation","handoff_id"},"closed abort acknowledgement required")
        else:
            _need(set(value) == {"schema","operation","handoff_id","pid","process_birth_id"}
                  and value.get("operation") == "spawned" and type(value["pid"]) is int and value["pid"] > 0,
                  "actual closed Popen birth acknowledgement required")
            _text(value["process_birth_id"])
        return value

    def _handoff(self,channel):
        raw = _receive(channel)
        peer = self._native.typed._kernel_peer_identity(channel)
        _need(peer[1] == os.geteuid(),"native checked peer UID differs")
        if raw.get("operation") == "prepare_isolated":
            return self._isolated_handoff(channel,_isolated_request(raw),peer)
        request = _request(raw)
        self._boundary = "native_daemon_to_owner_wrapper"
        with self._lock:
            _need(not self._pending and not self._consumed and not self._ambiguous,"checked session already consumed")
            self._pending = True
        try:
            def prepared(claim,lease):
                with self._server._lock:
                    self._identity_current()
                    _need(request["context_cid"] == self._context,"checked context differs from original scope")
                    claim = self._closing_authority(request,peer,claim,lease)
                    with self._lock: self._consumed = True
                    self._active_charged = True
                    result = self._scope.prepare_dispatch(native_prompt=request["input_text"],
                        worktree=Path(request["worktree"]),base_command=request["command"])
                    _need(type(result) is dict and result.get("schema") == "checked-observer-budget-dispatch@1"
                          and result.get("context_cid") == self._context,"original scope preparation required")
                    command = _derived(result["command"],request["command"],self._context)
                    _need(command[-1] == request["database_task_cid"],"derived checked artifact has another native task")
                    self._prepared = json.loads(_wire(result)); frozen = self._prepared_wire = _wire(self._prepared)
                    self._scope.require_current()
                    claim = self._closing_authority(request,peer,claim,lease)
                    _need(_wire(self._prepared) == frozen,"checked preparation mutated before authorization")
                    nonce = uuid.uuid4().hex
                    self._prepared_ack = True
                    _send(channel,self._prepared_response(nonce,command,_input(request["input_text"])))
                    ack = self._ack(channel,nonce)
                    if ack["operation"] == "abort":
                        self._retire(ambiguous=True)
                        status = {"status":"aborted","boundary":self._boundary,"worker_birth_observed":False}
                    else:
                        observed = self._native._require_wrapper_kernel_command(
                            child=ack["pid"],command=command,worktree=request["worktree"])
                        _need(observed["parent_pid"] == peer[0] and observed["process_birth_id"] == ack["process_birth_id"],
                              "wrapper birth differs from native daemon acknowledgement")
                        self._wrapper_spawned = True
                        self._wrapper = {"ticket":nonce,"request":request,"claim":claim,"pid":ack["pid"],
                            "process_birth_id":ack["process_birth_id"],"start_time_ticks":observed["start_time_ticks"],
                            "daemon_peer":peer,"command":command}
                        self._wrapper_wire = _wire(self._wrapper)
                        status = {"status":"spawned","boundary":self._boundary,"wrapper_birth_observed":True,
                            "worker_birth_observed":False,"pid":ack["pid"],"process_birth_id":ack["process_birth_id"]}
                return nonce,status
            nonce,status = self._execute_with_attempt_fence(request,peer,prepared)
            # No native callbacks before attempt/owner locks have been released.
            self._prepared_ack = False
            self._observe({"schema":SCHEMA,**status})
            _send(channel,{"schema":SCHEMA,"ok":True,"phase":"released","handoff_id":nonce})
        finally:
            with self._lock: self._pending = False

    def _isolated_handoff(self,channel,request,peer):
        self._boundary = "isolated_worker"
        with self._lock:
            _need(not self._pending and self._consumed and self._wrapper_spawned and not self._isolated_spawned
                  and not self._ambiguous and self._wrapper is not None,"registered original checked wrapper required")
            self._pending = True
        try:
            wrapper = self._wrapper
            def prepared(claim,lease):
                with self._server._lock:
                    self._identity_current(); observed = self._native._kernel_identity(peer[0])
                    _need(peer[0] == wrapper["pid"] and peer[2] == wrapper["start_time_ticks"]
                          and observed["process_birth_id"] == wrapper["process_birth_id"]
                          and observed["parent_pid"] == wrapper["daemon_peer"][0]
                          and request["wrapper_ticket"] == wrapper["ticket"] and request["context_cid"] == self._context
                          and request["worktree"] == wrapper["request"]["worktree"],"isolated requester differs from registered wrapper birth")
                    command = ["/usr/bin/sudo","-n","-u","benchmarkworker","--",
                        str(Path(wrapper["command"][0]).parent / "worker-entry"),*wrapper["command"][1:]]
                    _need(request["command"] == command,"isolated sudo argv differs from derived checked recipe")
                    _need(self._native._same_native_dispatch_claim(claim,wrapper["claim"]),"registered native claim changed")
                    self._active_charged = True
                    # Wrapper intentionally changed allocation modes before this boundary.
                    self._scope.capture_allocated(); self._scope.require_current()
                    self._closing_authority(wrapper["request"],wrapper["daemon_peer"],claim,lease)
                    nonce = uuid.uuid4().hex; self._prepared_ack = True
                    _send(channel,self._prepared_response(nonce,command,_input(wrapper["request"]["input_text"])))
                    ack = self._ack(channel,nonce)
                    if ack["operation"] == "abort":
                        self._retire(ambiguous=True)
                        status = {"status":"aborted","boundary":self._boundary,"worker_birth_observed":False}
                    else:
                        deadline = time.monotonic() + ACK_TIMEOUT_SECONDS - 1
                        while True:
                            try:
                                child = self._native._isolated_kernel_birth(process_pid=ack["pid"],wrapper_pid=peer[0],command=command)
                                break
                            except (OSError,RuntimeError):
                                _need(time.monotonic() < deadline,"live UID1001 worker birth not observed within bound")
                                time.sleep(0.01)
                        _need(child["sudo"]["process_birth_id"] == ack["process_birth_id"]
                              and os.readlink("/proc/"+str(child["worker"]["pid"])+"/cwd") == request["worktree"],
                              "isolated worker birth or allocated cwd differs")
                        self._isolated_spawned = True
                        status = {"status":"spawned","boundary":self._boundary,"worker_birth_observed":True,
                            "isolated_uid_observed":True,"worker_uid":1001,"worker_pid":child["worker"]["pid"],
                            "worker_process_birth_id":child["worker"]["process_birth_id"],"sudo_pid":ack["pid"],
                            "wrapper_pid":peer[0],"parent_chain":child["parent_chain"]}
                return nonce,status
            nonce,status = self._execute_with_attempt_fence(wrapper["request"],wrapper["daemon_peer"],prepared)
            self._prepared_ack = False
            self._observe({"schema":SCHEMA,**status})
            _send(channel,{"schema":SCHEMA,"ok":True,"phase":"released","handoff_id":nonce})
        finally:
            with self._lock: self._pending = False

    def _retire(self,*,ambiguous):
        with self._lock:
            self._consumed = True
            self._ambiguous = self._ambiguous or ambiguous
        self._scope.abort_pending()  # Original scope never refunds reserved history.

    def _serve(self):
        while not self._stop.is_set():
            try: channel,_ = self._listener.accept()
            except TimeoutError: continue
            except OSError: return
            with channel:
                channel.settimeout(ACK_TIMEOUT_SECONDS)
                self._active_charged = False
                try:
                    self._handoff(channel)
                except Exception as error:
                    if self._active_charged:
                        try: self._retire(ambiguous=self._prepared_ack or self._wrapper_spawned)
                        except Exception: self._ambiguous = True
                    self._observe({"schema":SCHEMA,"status":"refused","boundary":getattr(self,"_boundary","unparsed"),
                        "error_type":type(error).__name__,"message_sha256":hashlib.sha256(str(error).encode()).hexdigest(),
                        "unacknowledged_child_birth_possible":self._ambiguous,
                        "worker_birth_observed":self._isolated_spawned})
                    try: _send(channel,{"schema":SCHEMA,"ok":False,"error":"checked_dispatch_refused"})
                    except (OSError,RuntimeError): pass

    def _close_transport(self):
        self._stop.set(); self._listener.close()
        self._thread.join(timeout=ACK_TIMEOUT_SECONDS+1)
        _need(not self._thread.is_alive(),"checked handoff has not closed safely")
        entry = self._original_socket_path.lstat()
        _need(stat.S_ISSOCK(entry.st_mode) and entry.st_uid == os.geteuid()
              and (entry.st_dev,entry.st_ino) == self._socket_identity,
              "original checked socket changed before cleanup")
        self._original_socket_path.unlink(); self._closed = True
        with _REGISTRY_LOCK: _BROKERS.pop(id(self),None)

    def close_after_native_stop(self):
        _need(type(self) is OwnerCheckedContextDispatchBroker and self._seal is _SEAL and self._pid == os.getpid()
              and self._scope.bound_runtime is self._runtime,"original checked broker required for close")
        if self._closed: return dict(self._close_receipt)
        _need(self._runtime._context_refresh_stopped() and not self._pending,
              "checked broker close requires native STOP and isolated cleanup")
        self._scope.abort_pending()
        self._close_transport()
        self._close_receipt = {"schema":SCHEMA,"broker_closed":True,"socket_removed":True,"thread_joined":True,
            "isolated_cleanup_checked":True,"worker_birth_observed":self._isolated_spawned,
            "ambiguous_birth_possible":self._ambiguous,"observations":self.observations,**_FALSE}
        return dict(self._close_receipt)

    def can_close_unlaunched(self):
        _need(type(self) is OwnerCheckedContextDispatchBroker and self._seal is _SEAL
              and self._pid == os.getpid(),"original checked broker required")
        with self._lock:
            if self._pending or self._consumed or self._wrapper_spawned or self._isolated_spawned or self._ambiguous:
                return False
        if type(getattr(self._runtime,"_children",None)) is not list or self._runtime._children:
            return False
        process,profile = getattr(self._runtime,"process",None),getattr(self._runtime,"profile",None)
        return process is None or profile is None or not process.snapshot(profile).members

    def close_unlaunched_construction(self):
        _need(type(self) is OwnerCheckedContextDispatchBroker and self._seal is _SEAL and self._pid == os.getpid()
              and not self._pending and not self._consumed and not self._wrapper_spawned and not self._isolated_spawned
              and not getattr(self._runtime,"_children",()),"unlaunched checked close requires no native birth or reservation")
        process,profile = getattr(self._runtime,"process",None),getattr(self._runtime,"profile",None)
        if process is not None and profile is not None:
            _need(not process.snapshot(profile).members,"unlaunched checked close found a native process tree")
        if not self._closed:
            self._close_transport()
            self._close_receipt = {"schema":SCHEMA,"broker_closed":True,"socket_removed":True,
                "thread_joined":True,"unlaunchedresources_disposed":True,**_FALSE}
        return dict(self._close_receipt)

    close = close_after_native_stop


def start_owner_checked_context_dispatch_broker(*,scope,runtime):
    return OwnerCheckedContextDispatchBroker(_SEAL,scope=scope,runtime=runtime)


__all__ = ["SOCKET_ENV","CONTEXT_ENV","OWNER_ENV","owner_dispatch_identity","WRAPPER_ENV","OWNER_FLAG","SCHEMA","CheckedContextDispatchError",
    "OwnerCheckedContextDispatchBroker","start_owner_checked_context_dispatch_broker",
    "require_checked_context_worker_dispatch","require_checked_context_isolated_worker_dispatch"]
