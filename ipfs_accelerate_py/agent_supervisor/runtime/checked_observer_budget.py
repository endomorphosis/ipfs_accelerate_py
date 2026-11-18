"""Original owner custody and single-attempt input reservation for observation.

Importing this module loads only the standard library. The owner factory loads
the existing checked delivery and accounting implementations locally. A worker
must use the separate broker client; public scope identifiers are data and never
stand in for this original process-local object or native launch authority.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import hmac
import json
import os
from enum import Enum
from pathlib import Path
import secrets
import shlex
from threading import RLock

SCHEMA = "checked-observer-budget-scope@1"
OWNER_FLAG = "--checked-context-owner-scope-cid"
MAX_WIRE_BYTES = 8_388_608
_KEY = object()
AUTHORITY = {"worker_birth_established": False, "provider_usage_measured": False,
    "proof_authority": False, "execution_authority": False, "completion_authority": False,
    "global_task_budget_enforced": False, "serialized_live_authority": False,
    "full_task_satisfaction": False, "training_convergence_proved": False}
_ROUTE = {"purpose": "coding", "semantic_transport": "native",
    "intent_requirements_transport": "expanded", "code_retrieval_codec": "expanded",
    "code_retrieval_wire_delivery": "inline", "observe_model_input_only": True,
    "single_attempt": True, "provider_calls": 0}


class CheckedObserverBudgetError(ValueError):
    """A bound original or its conservative one-attempt budget is unavailable."""


def _need(value, message):
    if not value:
        raise CheckedObserverBudgetError(message)


def _wire(value):
    count = 0
    def check(item, depth=0):
        nonlocal count
        count += 1
        _need(depth <= 48 and count <= 200000, "bounded plain observer scope JSON required")
        if isinstance(item, Enum):
            check(item.value, depth + 1)
        elif type(item) is dict:
            _need(all(type(key) is str for key in item), "exact string scope keys required")
            for child in item.values(): check(child, depth + 1)
        elif type(item) in {list, tuple}:
            for child in item: check(child, depth + 1)
        else:
            _need(item is None or type(item) in {str, int, bool}, "exact closed scope scalar required: " + type(item).__name__)
    check(value)
    def native_enum(item):
        _need(isinstance(item, Enum), "unsupported scope serialization type")
        return item.value
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
            default=native_enum).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        raise CheckedObserverBudgetError("canonical UTF-8 scope JSON required") from error
    _need(len(raw) <= MAX_WIRE_BYTES, "observer scope exceeds byte bound")
    return raw


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _plain(value):
    return json.loads(_wire(value))


def _path(value):
    _need(type(value) is type(Path()) and value.is_absolute() and value.resolve(strict=True) == value,
          "exact canonical existing owner path required")
    return value


class CheckedObserverBudgetScope:
    """Sealed original owner; one reservation, no serialized reconstruction."""
    __slots__ = ("_apis", "_owner", "_inputs", "_input_wire", "_binding_wire", "_context_cid",
        "_manager", "_scope", "_runtime", "_runtime_wire", "_command_wire", "_pending",
        "_prepared_wire", "_allocated", "_phase", "_checking", "_closed", "_secret", "_seal", "_pid", "_lock")

    def __init__(self, *, apis, budget_owner, inputs, binding, manager, scope, _key=None):
        _need(_key is _KEY, "use the original owner scope factory")
        self._apis, self._owner, self._inputs = apis, budget_owner, inputs
        self._input_wire, self._binding_wire = _wire(inputs), _wire(binding)
        self._context_cid = apis["content_identity"](binding)
        self._manager, self._scope = manager, scope
        self._runtime = self._runtime_wire = self._command_wire = None
        self._pending = self._prepared_wire = self._allocated = None
        self._phase, self._checking, self._closed = "unbound", False, False
        self._secret, self._pid, self._lock = secrets.token_bytes(32), os.getpid(), RLock()
        self._seal_state()

    def __copy__(self):
        raise CheckedObserverBudgetError("original owner scope cannot be copied")

    def __deepcopy__(self, memo):
        raise CheckedObserverBudgetError("original owner scope cannot be copied")

    def _state(self):
        return _wire({"self": id(self), "pid": self._pid, "lock": id(self._lock),
            "owner": id(self._owner), "apis": {key: id(value) for key, value in self._apis.items()},
            "input_wire": self._input_wire.decode(), "binding_wire": self._binding_wire.decode(),
            "context_cid": self._context_cid, "manager": id(self._manager), "scope": id(self._scope),
            "check": id(self._scope["check"]), "runtime": id(self._runtime),
            "runtime_wire": self._runtime_wire.decode() if self._runtime_wire is not None else None,
            "command_wire": self._command_wire.decode() if self._command_wire is not None else None,
            "pending": id(self._pending), "prepared_wire": self._prepared_wire.decode() if self._prepared_wire is not None else None,
            "allocated": id(self._allocated), "phase": self._phase, "checking": self._checking, "closed": self._closed})

    def _seal_state(self):
        self._seal = hmac.new(self._secret, self._state(), hashlib.sha256).digest()

    def _identity(self, *, open_required=True):
        _need(type(self) is CheckedObserverBudgetScope and self._pid == os.getpid(), "original process-local owner scope required")
        _need(hmac.compare_digest(self._seal, hmac.new(self._secret, self._state(), hashlib.sha256).digest()),
              "sealed original owner scope changed")
        _need(not open_required or not self._closed, "owner scope is closed")

    @property
    def context_cid(self):
        with self._lock:
            self._identity()
            return self._context_cid

    @property
    def is_closed(self):
        """Original process-local lifecycle state, including after retirement."""
        with self._lock:
            self._identity(open_required=False)
            return self._closed

    @property
    def material_binding(self):
        with self._lock:
            self._identity()
            return json.loads(self._binding_wire)

    @property
    def task_cid(self):
        return self.material_binding["task_cid"]

    @property
    def bound_runtime(self):
        with self._lock:
            self._identity()
            return self._runtime

    @property
    def prepared_record(self):
        """Detached evidence only; never a replacement reservation or scope."""
        with self._lock:
            self._identity()
            _need(self._prepared_wire is not None, "no original coding dispatch has been prepared")
            return json.loads(self._prepared_wire)

    def _runtime_material(self, runtime):
        native = self._apis["native"]
        _need(type(runtime) is native.AdmittedBenchmarkRuntime, "exact native admitted runtime required")
        manifest = runtime.manifest
        _need(type(manifest) is dict and native._digest(manifest) == runtime.manifest_id
              and manifest.get("schema") == "isolated-admitted-supervisor-launch@1"
              and manifest.get("run_id") == runtime.run_id
              and manifest.get("state_root") == str(runtime.state)
              and manifest.get("repository_root") == str(runtime.repository)
              and _wire(manifest.get("checked_observer_budget_scope")) == self._binding_wire
              and _wire(manifest.get("local_admission")) == _wire(self._inputs["admission"])
              and _wire(runtime.admission) == _wire(self._inputs["admission"])
              and getattr(runtime, "checked_observer_budget_scope", None) is self,
              "native runtime differs from original signed scope material")
        return _wire({"self": id(runtime), "manifest": manifest, "manifest_id": runtime.manifest_id,
            "signature": runtime.signature, "run_id": runtime.run_id, "state": str(runtime.state),
            "repository": str(runtime.repository), "context_bundle": runtime.context_bundle,
            "objects": {key: id(getattr(runtime, key)) for key in
                ("server", "source", "process", "profile", "service", "coordinator", "local_profile", "owner_identity")}})

    def _command(self, base_command):
        _need(type(base_command) is list and 1 <= len(base_command) <= 64
              and all(type(part) is str and 0 < len(part.encode()) <= 8192 for part in base_command),
              "bounded exact literal base command required")
        _need(base_command[0] == "/opt/ipfs-supervisor/bin/router-worker", "original observer wrapper required")
        allowed = {"--model", "--reasoning-effort", "--timeout", "--max-output-tokens", "--purpose",
            "--semantic-repository", "--semantic-transport", "--intent-requirements-transport",
            "--code-retrieval-codec", "--code-retrieval-wire-delivery", "--public-instruction-artifact",
            "--public-instruction-sha256", "--public-instruction-task-cid", OWNER_FLAG}
        options, index = {}, 1
        while index < len(base_command):
            flag = base_command[index]
            _need(flag not in options and flag in allowed | {"--observe-model-input-only"}, "unique closed observer recipe required")
            if flag == "--observe-model-input-only": options[flag] = True; index += 1
            else:
                _need(index + 1 < len(base_command) and not base_command[index + 1].startswith("--"), "complete literal observer option required")
                options[flag] = base_command[index + 1]; index += 2
        public, repository = self._inputs["public_instruction"], self._inputs["native_repository"]
        required = {"--observe-model-input-only": True, OWNER_FLAG: self._context_cid,
            "--semantic-repository": repository, "--semantic-transport": "native",
            "--purpose": "coding", "--intent-requirements-transport": "expanded",
            "--code-retrieval-codec": "expanded", "--code-retrieval-wire-delivery": "inline",
            "--public-instruction-artifact": public["artifact"], "--public-instruction-sha256": public["sha256"],
            "--public-instruction-task-cid": public["task_cid"]}
        _need(all(options.get(key) == value for key, value in required.items()), "exact owner-bound native inline observation recipe required")
        return _plain(base_command)

    def require_current(self):
        with self._lock:
            self._identity()
            _need(not self._checking, "reentrant owner current check refused")
            _need(self._phase not in {"retired", "closed"}, "owner scope is retired")
            _need(_wire(self._inputs) == self._input_wire, "original owner scope inputs changed")
            binding = json.loads(self._binding_wire)
            self._owner._check()
            _need(_wire(self._owner.binding) == _wire(binding["budget_binding"])
                  and _wire(self._owner.policy.to_dict()) == _wire(binding["budget_policy"]), "original budget binding or policy changed")
            if self._runtime is not None:
                _need(self._runtime_material(self._runtime) == self._runtime_wire, "original immutable runtime binding changed")
            self._checking = True; self._seal_state()
            try:
                self._scope["check"]()
                if self._allocated is not None: self._allocated()
                if self._pending is not None: self._pending._current()
                _need(_wire(self._inputs) == self._input_wire, "owner originals changed during current check")
                if self._runtime is not None:
                    _need(self._runtime_material(self._runtime) == self._runtime_wire, "runtime changed during current check")
            finally:
                self._checking = False; self._seal_state()

    def bind_runtime(self, runtime, base_command):
        with self._lock:
            self.require_current()
            _need(self._phase == "unbound" and self._runtime is None, "owner scope binds exactly one original runtime")
            command = self._command(base_command)
            material = self._runtime_material(runtime)
            argv = runtime.manifest.get("argv")
            _need(type(argv) is list and argv.count("--implementation-command") == 1
                  and shlex.split(argv[argv.index("--implementation-command") + 1]) == command,
                  "signed native implementation command differs")
            self._runtime, self._runtime_wire, self._command_wire = runtime, material, _wire(command)
            self._phase = "bound"; self._seal_state()
            self.require_current()

    def prepare_dispatch(self, *, native_prompt, worktree, base_command):
        with self._lock:
            self.require_current()
            _need(self._phase == "bound" and self._pending is None, "one original coding dispatch per owner scope")
            _need(_wire(self._command(base_command)) == self._command_wire, "original base command changed")
            worktree = _path(worktree)
            _need(type(native_prompt) is str and 0 < len(native_prompt.encode()) <= 256000, "exact complete emitted native prompt required")
            self._phase = "preparing"; self._seal_state()
            checked, instructions, semantic = self._apis["checked"], self._apis["instructions"], self._apis["semantic"]
            public, boundary = self._inputs["public_instruction"], self._inputs["boundary"]
            repository = Path(self._inputs["native_repository"])
            guard = None
            try:
                descriptor = checked.prepare_checked_context_extension(repository=repository,
                    admission=self._inputs["admission"], task_cid=self.task_cid, native_prompt=native_prompt,
                    public_instruction=public, context_packet=self._inputs["context_packet"],
                    checked_binding=self._owner.binding, budget_policy=self._owner.policy.to_dict())
                instruction, inclusion = instructions.load_public_instruction(artifact=Path(public["artifact"]),
                    expected_sha256=public["sha256"], task_cid=self.task_cid, prompt=native_prompt,
                    workspace=worktree, repository=repository, requirements_transport="expanded")
                encoded = semantic.encode_semantic_router_prompt(prompt=native_prompt, repository=repository).receipt
                verification = {"schema": "semantic-router-source-verification@1",
                    **{key: encoded[key] for key in checked._SEMANTIC_FIELDS - {"schema"}}}
                suffix, preflight, guard = checked.load_checked_context_extension(
                    artifact=Path(descriptor["artifact"]), expected_sha256=descriptor["sha256"], task_cid=self.task_cid,
                    native_prompt=native_prompt, instruction_receipt=inclusion, semantic_verification=verification,
                    workspace=worktree, boundary=boundary, repository=repository)
                guard(); guard.close(); guard = None
                preparation = self._apis["worker"].prepare_checked_worker_budget_event(budget_owner=self._owner,
                    event_id="checked-observer:initial", native_prompt=native_prompt, worker_workspace=worktree,
                    router_prompt=native_prompt + instruction, checked_suffix=suffix, task_cid=self.task_cid,
                    instruction_receipt=inclusion, boundary=boundary, boundary_sha256=boundary["manifest_sha256"])
                self.require_current()
                self._pending = self._owner.reserve(preparation["event"])
                self._prepared_wire = _wire({"preparation": preparation, "preflight": preflight,
                    "descriptor": descriptor, "verification": verification})
                self._phase = "pending"; self._seal_state()
                self.require_current()
                command = list(base_command)
                index = command.index(OWNER_FLAG); del command[index:index + 2]
                command += ["--checked-context-artifact", descriptor["artifact"], "--checked-context-sha256",
                    descriptor["sha256"], "--checked-context-task-cid", self.task_cid]
                result = {"schema": "checked-observer-budget-dispatch@1", "command": command,
                    "preparation": preparation, "preflight": preflight, "context_cid": self._context_cid}
                result["dispatch_cid"] = self._apis["content_identity"](result)
                return _plain(result)
            except BaseException:
                if guard is not None: guard.close()
                if self._pending is not None:
                    self._pending.abort(); self._pending = None
                self._phase = "retired"; self._seal_state()
                raise

    def capture_allocated(self):
        with self._lock:
            self.require_current()
            _need(self._phase == "pending" and self._allocated is None, "one post-permission allocated capture required")
            record = json.loads(self._prepared_wire); preparation = record["preparation"]; basis = preparation["basis"]
            _, receipt, guard = self._apis["checked"].load_checked_context_extension(
                artifact=Path(record["descriptor"]["artifact"]), expected_sha256=record["descriptor"]["sha256"],
                task_cid=self.task_cid, native_prompt=basis["native_prompt"], instruction_receipt=basis["instruction_receipt"],
                semantic_verification=record["verification"], workspace=Path(basis["worker_workspace"]),
                boundary=self._inputs["boundary"], repository=Path(self._inputs["native_repository"]))
            try:
                _need(_wire(receipt) == _wire(record["preflight"]), "allocated inclusion changed during permission transition")
                guard()
            except BaseException:
                guard.close(); raise
            self._allocated = guard; self._seal_state()
            self.require_current()

    def commit_observation(self, receipt):
        with self._lock:
            self.require_current()
            _need(self._phase == "pending" and self._pending is not None and self._allocated is not None,
                  "original pending reservation and post-permission allocated custody required")
            record = json.loads(self._prepared_wire); preparation = record["preparation"]
            try:
                value = self._apis["observations"].validate_model_input_observation(receipt)
                _need(value["status"] == "observed" and value["schema"] == self._apis["observations"].CHECKED_SCHEMA,
                      "successful closed native checked observation required")
                # The original event is still reserved, so a prepare replay is
                # intentionally unavailable. Compare its immutable retained
                # bytes and reuse Z's exact native/suffix/live-budget joins.
                _need(self._pending._event == self._apis["budgets"]._wire(preparation["event"]), "original reserved coding event changed")
                self._apis["worker"]._join(value, preparation, binding=self._owner.binding, policy=self._owner.policy.to_dict())
                self.require_current()
                snapshot = self._pending.commit(observed_event=preparation["event"])
                self._pending = None; self._phase = "committed"; self._seal_state()
                self.require_current()
                raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
                result = {"schema": "checked-observer-budget-commit@1", "context_cid": self._context_cid,
                    "event_cid": preparation["event"]["event_cid"], "budget_snapshot": snapshot,
                    "observation_json": raw.decode(), "observation_sha256": _sha(raw), "observation_bytes": len(raw),
                    "local_reservation_committed": True, **AUTHORITY}
                result["commit_cid"] = self._apis["content_identity"](result)
                return result
            except BaseException:
                if self._pending is not None:
                    self._pending.abort(); self._pending = None
                self._phase = "retired"; self._seal_state()
                raise

    def abort_pending(self):
        """Nonrefunding retirement; available even after current-source drift."""
        with self._lock:
            self._identity()
            if self._pending is not None:
                self._pending.abort(); self._pending = None
            self._phase = "retired"; self._seal_state()

    def _finish(self):
        failure = None
        try:
            if self._pending is not None:
                self._pending.abort(); self._pending = None
        except BaseException as error:
            failure = error
        try:
            if self._allocated is not None: self._allocated.close()
        finally:
            self._allocated = None; self._phase = "closed"; self._closed = True; self._seal_state()
            # Shutdown is already constrained by the native STOP/no-launch
            # fence. Throwing a local retirement marker through the delivery
            # generator skips its publication check, while finally blocks
            # release every FD even when sources or proofs have drifted.
            marker = CheckedObserverBudgetError("owner scope retired after native shutdown fence")
            try: self._manager.__exit__(type(marker), marker, None)
            except BaseException as error: failure = failure or error
        if failure is not None: raise failure

    def close_unlaunched_construction(self, runtime=None):
        with self._lock:
            self._identity()
            if self._runtime is None:
                _need(runtime is None, "unbound scope cannot close another runtime")
            else:
                _need(runtime is self._runtime and self._runtime_material(runtime) == self._runtime_wire,
                      "original immutable unlaunched runtime required")
                _need(type(runtime._children) is list and not runtime._children, "unlaunched cleanup refuses any possible native child")
                broker = getattr(runtime, "checked_context_dispatch_broker", None)
                if broker is not None:
                    _need(broker.can_close_unlaunched() is True, "original broker must prove no pending or possible birth")
            _need(self._pending is None and self._phase in {"unbound", "bound"}, "attempted dispatch requires native STOP cleanup")
            self._finish()
            return {"schema": "checked-observer-budget-close@1", "context_cid": self._context_cid,
                "unlaunched": True, "retained_guards_closed": True, "publication_check_claimed": False, **AUTHORITY}

    def close_after_native_stop(self, runtime):
        with self._lock:
            self._identity()
            _need(runtime is self._runtime and self._runtime_material(runtime) == self._runtime_wire,
                  "original immutable stopped runtime required")
            _need(runtime._context_refresh_stopped() is True, "successful native fenced STOP and worker cleanup required")
            stop = runtime._context_refresh_stop_receipt
            _need(stop.succeeded is True and stop.data.get("old_tree_fenced") is True,
                  "original successful native STOP fence required")
            cleanup = stop.data.get("isolated_worker_cleanup")
            _need(isinstance(cleanup, Mapping) and type(cleanup.get("returncode")) is int
                  and cleanup["returncode"] == 0 and cleanup.get("single_worker") is True,
                  "exact successful isolated worker cleanup required")
            self._finish()
            return {"schema": "checked-observer-budget-close@1", "context_cid": self._context_cid,
                "unlaunched": False, "retained_guards_closed": True, "publication_check_claimed": False, **AUTHORITY}


def create_checked_observer_budget_scope(*, budget_owner, context_packet, admission,
        public_instruction, boundary, index, repository, expected_head, native_capsules,
        code_views, tool_policy, policy_roots, policy_observer, scheduler, capacity_control,
        basis=None, expansion_reference=None, retry_instruction="", **basis_fields):
    """Owner-only factory; retain current originals before any runtime callback."""
    from benchmarks.agent_supervisor.container_coding import terminal_semantic_budget_delivery as delivery
    from benchmarks.agent_supervisor.container_coding import terminal_checked_worker_budget as worker
    from . import checked_audit_context as checked, router_public_instruction as instructions
    from . import semantic_router_translation as semantic, model_input_observation_receipt as observations
    from ..entrypoints import admitted_benchmark_runtime as native
    from ..proof.formal_verification_contracts import content_identity
    budgets = delivery.budgets
    _need(type(budget_owner) is budgets.ContextBudgetOwner, "original live context budget owner required")
    budget_owner._check()
    _need(not budget_owner.snapshot()["records"], "single coding scope requires an unused original budget")
    _need(expansion_reference is None and retry_instruction == "", "single coding observer scope cannot select expansion or retry")
    _need((basis is None or type(basis) is dict) and not (basis is not None and basis_fields), "one exact delivery basis required")
    basis = basis_fields if basis is None else basis
    _need(type(context_packet) is str and 0 < len(context_packet.encode()) <= 256000, "complete bounded checked context packet required")
    verified = checked.local.verify_local_benchmark_admission(admission, initial=True)
    tasks = verified["graph"].tasks
    _need(len(tasks) == 1 and type(public_instruction) is dict and public_instruction.get("task_cid") == tasks[0].task_cid,
          "one original admitted task and instruction required")
    original_boundary = boundary
    boundary = checked._boundary(boundary)
    inputs = {"context_packet": context_packet, "admission": admission, "public_instruction": public_instruction,
        "boundary": original_boundary, "native_repository": verified["manifest"]["repository"]}
    binding = {"schema": SCHEMA, "scope_id": secrets.token_hex(16), "task_cid": tasks[0].task_cid,
        "task_id": tasks[0].task_key, "admission_cid": content_identity(admission),
        "budget_binding": budget_owner.binding, "budget_policy": budget_owner.policy.to_dict(),
        "context_sha256": _sha(context_packet.encode()), "context_bytes": len(context_packet.encode()),
        "public_instruction": _plain(public_instruction), "boundary": boundary, "route_profile": _ROUTE, **AUTHORITY}
    from contextlib import ExitStack
    import sys
    manager = ExitStack()
    try:
        from types import SimpleNamespace
        from . import checked_context_dispatch
        supervisor_root = Path(__file__).resolve().parents[1]
        transport_paths = [supervisor_root / "todo_daemon/supervisor_runtime.py",
            supervisor_root / "todo_daemon/implementation_daemon.py",
            Path(worker.__file__).resolve().with_name("container_worker_deployment.py")]
        producer, producer_guard = manager.enter_context(delivery.forecast.metadata._producer(
            [sys.modules[__name__], checked_context_dispatch, worker, checked, instructions, semantic, observations, native,
             *(SimpleNamespace(__file__=str(path)) for path in transport_paths)]))
        native_guard = checked._Custody()
        manager.callback(native_guard.close)
        _, _, original_instruction = checked._instruction(Path(inputs["native_repository"]),
            public_instruction, native_guard, owner_uid=boundary["owner_uid"])
        literal = native_guard.read(Path(inputs["native_repository"]), original_instruction["source_path"], checked.MAX_SOURCE_BYTES)
        _need(_sha(literal) == original_instruction["source_sha256"]
              and len(literal) == original_instruction["source_bytes"], "original signed instruction source differs")
        native_guard()
        scope = manager.enter_context(delivery._current_delivery_scope(budget_owner=budget_owner, context_packet=context_packet,
            index=index, repository=repository, expected_head=expected_head, native_capsules=native_capsules,
            code_views=code_views, tool_policy=tool_policy, policy_roots=policy_roots,
            policy_observer=policy_observer, scheduler=scheduler, capacity_control=capacity_control,
            expansion_reference=None, retry_instruction="", basis=basis))
        original_check = scope["check"]
        def check():
            producer_guard(); native_guard(); original_check(); native_guard(); producer_guard()
        scope = {**scope, "check": check}
        binding["owner_producer"] = producer
    except BaseException:
        manager.__exit__(*sys.exc_info())
        raise
    try:
        value = CheckedObserverBudgetScope(apis={"delivery": delivery, "worker": worker, "budgets": budgets,
            "checked": checked, "instructions": instructions, "semantic": semantic, "observations": observations,
            "native": native, "content_identity": content_identity}, budget_owner=budget_owner, inputs=inputs,
            binding=binding, manager=manager, scope=scope, _key=_KEY)
        value.require_current()
        return value
    except BaseException:
        manager.__exit__(*__import__("sys").exc_info())
        raise
