"""Opt-in current proof-query custody at the native supervisor launch boundary.

The live capability adds context and freshness conditions to the existing
finite worker reservation. Conditional proof records grant no facts, omitted
tasks or completion. A serialized binding cannot authorize a process. The
separate coding-child receiving boundary must enforce its own owner handoff.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import shlex
import threading
import time
import uuid

from ipfs_datasets_py.duckdb_control.codebase_verification_catalog import CodebaseVerificationCatalog
from ipfs_datasets_py.logic.software_contracts.content import canonical_dag_json_bytes, cid_for_structured

from ..planning import finite_proof_query_join as join
from ..planning.finite_integer_source_custody import capture_source_custody, _read as read_custody_file
from ..planning.repository_plan_preview import RepositoryPlanPreviewOwner
from ..task_sources.typed_database_task_source import TypedDatabaseTaskSource
from .quack_state_server import QuackStateServer, ServerLifecycle
from . import finite_proof_query_admission as admission_boundary
from . import finite_repository_admission as finite
from . import finite_repository_execution as execution

PROFILE = "finite-repository-one-ready-proof-query-bound-native-worker@1"
SCHEMA = "supervisor-finite-proof-query-execution-closure@1"
EXECUTION_SCHEMA = "supervisor-finite-proof-query-repository-execution-scope@1"
MAX_BYTES = admission_boundary.MAX_BYTES
_SEAL = object()
_OPERATION_SEAL = object()
_FALSE = {name: False for name in (
    "source_semantics_verified", "runtime_behavior_verified", "proof_authority",
    "code_proof_authority", "behavior_authority", "completion_authority",
    "task_omission_authority", "mutation_authority", "publication_authority",
    "production_activation", "authenticated_process_origin", "atomicity_attested",
    "training_executed", "model_inference_executed", "convergence_proved",
)}


class FiniteProofQueryExecutionError(ValueError):
    """The exact native proof-query receiving or launch closure was refused."""


class _ProofQueryStartRefused(Exception):
    """Abort receiving while retaining the original failed native control result."""

    def __init__(self, result):
        self.result = result


def _need(value, message):
    if not value:
        raise FiniteProofQueryExecutionError(message)


def _wire(value):
    return admission_boundary._wire(value)


def _plain(value):
    return json.loads(_wire(value))


def _same(left, right):
    return _wire(left) == _wire(right)


def _owner_matches(left, right):
    return (type(left) is RepositoryPlanPreviewOwner and type(right) is RepositoryPlanPreviewOwner
            and left.index is right.index and left.repository == right.repository
            and left.expected_head is right.expected_head and left.memory_mb == right.memory_mb)


def _physical_plans(server, task_plan_ids):
    """Complete bounded SQL rows; no signer, task projection or source callback."""
    fields = ("plan_cid", "goal_cid", "plan_alias", "status", "created_at", "updated_at", "revision", "body")
    rows = server._connection.execute(
        "SELECT plan_cid,goal_cid,plan_alias,status,created_at,updated_at,revision,body_json "
        "FROM plans ORDER BY plan_cid LIMIT 17").fetchall()
    _need(1 <= len(rows) <= 16 and {str(row[0]) for row in rows} == set(task_plan_ids),
          "complete native proof-query plan population differs from the original tasks")
    result = []
    for values in rows:
        row = {key: (json.loads(values[number]) if key == "body" else
                     int(values[number]) if key == "revision" else str(values[number] or ""))
               for number, key in enumerate(fields)}
        _need(type(row["body"]) is dict and row["revision"] > 0,
              "complete native plan body and revision required")
        result.append(row)
    return result


def _references(plans, value, checkpoint):
    """Match complete native references to both exact immutable admissions."""
    expected = (("finite_repository_admission_ref", finite.REFERENCE_SCHEMA, value["finite_admission"]),
                ("finite_proof_query_admission_ref", admission_boundary.REFERENCE_SCHEMA, value))
    witnesses, seen = [], set()
    for plan in plans:
        for name, schema, body in expected:
            raw = _wire(body)
            reference = plan["body"].get(name)
            _need(type(reference) is dict and set(reference) ==
                  {"schema", "path", "sha256", "bytes", "admission_cid", *finite._FALSE}
                  and reference["schema"] == schema and type(reference["bytes"]) is int
                  and reference["bytes"] == len(raw)
                  and reference["sha256"] == hashlib.sha256(raw).hexdigest()
                  and reference["admission_cid"] == cid_for_structured(body)
                  and all(reference[flag] is False for flag in finite._FALSE),
                  "native plan lacks the exact complete finite proof-query admission reference")
            witness, current = read_custody_file(Path(reference["path"]), role="admission:" + name,
                                               bound=MAX_BYTES, checkpoint=checkpoint)
            _need(current == raw, "native retained admission reference bytes differ")
            if str(witness.path) not in seen:
                seen.add(str(witness.path))
                witnesses.append(witness)
    return tuple(witnesses)


def _proof_witnesses(owner, closure, checkpoint):
    result, total = [], 0
    for name in ("projection", "verification", "applicability"):
        body, identity = closure[name], closure[name + "_cid"]
        _need(cid_for_structured(body) == identity, "complete selected native proof object differs")
        witness, raw = read_custody_file(owner.index.artifacts.path_for(identity),
            role="proof-query:" + identity, bound=MAX_BYTES, checkpoint=checkpoint)
        _need(raw == canonical_dag_json_bytes(body), "physical native proof bytes differ")
        total += len(raw)
        _need(total <= MAX_BYTES, "physical native proof closure exceeds its bound")
        result.append(witness)
    return tuple(result)


def _witness_fence(witnesses, checkpoint):
    for original in witnesses:
        current, _ = read_custody_file(original.path, role=original.role,
                                       bound=MAX_BYTES, checkpoint=checkpoint)
        _need(current == original, "selected native admission, proof or public context file changed")


@dataclass
class _ReceivingOperation:
    seal: object
    purpose: str
    runtime: object
    pid: int
    thread_id: int
    fields: tuple
    material_bytes: bytes
    checkpoint: object
    closed: bool = False
    exited: bool = False


class FrozenFiniteProofQueryExecutionClosure:
    """Exact private owner capability; dictionaries and subclasses are inert."""

    def __init__(self, seal, *, owner, value, catalog, server, source, candidate,
                 base_candidate, output, observer, material, witnesses):
        _need(seal is _SEAL, "proof-query execution closure must be prepared by its native owner")
        self._seal, self._original_owner, self._owner = seal, owner, owner
        self._value = _wire(value)
        self._catalog, self._server, self._source = catalog, server, source
        self._candidate, self._base_candidate = _wire(candidate), _wire(base_candidate)
        self._output, self._observer, self._material = output, observer, _wire(material)
        self._witnesses = tuple(witnesses)
        self._scope, self._runtime, self._operation = None, None, None
        self._receiving_entering = False

    @property
    def material_binding(self):
        return json.loads(self._material)

    def to_dict(self):
        return self.material_binding

    def _fields(self):
        owner = self._owner
        return (owner, owner.index, owner.repository, owner.expected_head,
                self._catalog, self._catalog.store, self._catalog.artifacts,
                self._server, self._server._connection, self._server._lock,
                self._source, self._source.execution_route_policy, self._scope,
                self._value, self._candidate, self._base_candidate, self._observer,
                owner.cancel_event, owner.parent_lease, owner.scheduler,
                _wire([type(owner.timeout_seconds).__name__, repr(owner.timeout_seconds), owner.memory_mb]),
                self._server.identity,
                _wire(self._server.identity.to_dict()),
                _wire(self._source.execution_route_policy.to_dict()))

    def _assert_scope(self, scope):
        _need(type(self) is FrozenFiniteProofQueryExecutionClosure and self._seal is _SEAL
              and type(scope) is execution.FrozenFiniteRepositoryExecutionScope
              and scope is self._scope and _owner_matches(self._original_owner, scope._owner)
              and scope._owner is self._owner and scope._server is self._server
              and scope._source is self._source
              and self._catalog.index is self._owner.index
              and self._catalog.store is self._owner.index.ingestor.store
              and self._catalog.artifacts is self._owner.index.artifacts,
              "exact same native finite/proof owner and bound execution capability required")
        scope._active()
        payload = scope.to_dict()["payload"]
        _need(payload["schema"] == EXECUTION_SCHEMA and payload["profile"] == PROFILE
              and _same(payload.get("proof_query_closure"), self.material_binding)
              and _same(scope._admission, json.loads(self._value)["finite_admission"]),
              "signed finite execution scope lost its complete proof-query binding")

    def _bind_scope(self, scope):
        """Call once after the existing scope has its signed material and custody."""
        _need(self._scope is None and type(scope) is execution.FrozenFiniteRepositoryExecutionScope
              and _owner_matches(self._original_owner, scope._owner)
              and scope._server is self._server and scope._source is self._source
              and hasattr(scope, "_custody") and hasattr(scope, "_fence")
              and _same(scope._admission, json.loads(self._value)["finite_admission"])
              and _same(scope.to_dict()["payload"]["candidate"]["descriptor"], json.loads(self._candidate)),
              "proof-query closure cannot bind a foreign, incomplete or already bound finite reservation")
        self._scope, self._owner = scope, scope._owner
        self._assert_scope(scope)
        self.require_detached(scope=scope)

    def extend_candidate_binding(self, base_binding):
        """Only extend the independently validated original finite command."""
        _need(type(self) is FrozenFiniteProofQueryExecutionClosure and self._seal is _SEAL
              and _wire(base_binding) == self._base_candidate,
              "proof-query context must extend the exact native finite candidate binding")
        context = self.material_binding["worker_context"]
        argv = [*base_binding["argv"], "--finite-proof-query-context", context["artifact"],
                "--finite-proof-query-sha256", context["sha256"],
                "--finite-proof-query-context-cid", context["context_cid"]]
        return {**_plain(base_binding), "argv": argv, "implementation_command": shlex.join(argv),
                "worker_context": context}

    def _checkpoint(self):
        operation = self._operation
        if operation is not None:
            return operation.checkpoint()
        _need(self._scope is not None, "proof-query receiving budget requires its live finite reservation")
        self._scope._active()
        return self._owner.timeout_seconds

    @contextmanager
    def _budget(self):
        if self._operation is not None:
            yield self._operation.checkpoint
            return
        checkpoint = admission_boundary._budget(self._owner)
        def current():
            if self._scope is not None:
                self._scope._active()
            return checkpoint()
        yield current

    def _physical_close(self, *, scope, checkpoint, ready_population,
                        worker_source_custody=None):
        value, binding = json.loads(self._value), self.material_binding
        _need(_same(self._server.identity.to_dict(), binding["native_owner_identity"])
              and _same(self._source.execution_route_policy.to_dict(), binding["execution_route_policy"]),
              "native proof-query task owner or execution route changed")
        plans = _physical_plans(self._server, {row["plan_cid"] for row in
                                scope.to_dict()["payload"]["native_population"]["tasks"]})
        _need(_same(plans, binding["native_plan_rows"]),
              "complete native proof-query plan rows changed after callbacks")
        _references(plans, value, checkpoint)
        def source_close():
            if ready_population:
                _need(worker_source_custody is None,
                      "parent receiving cannot use allocated-worker source custody")
                scope._physical_source()
            else:
                from .finite_proof_query_worker_source_custody import FrozenWorkerSourceCustody
                _need(type(worker_source_custody) is FrozenWorkerSourceCustody
                      and getattr(worker_source_custody, "_scope", None) is scope,
                      "coding dispatch requires its sealed native allocated-source custody")
                FrozenWorkerSourceCustody.require_current(worker_source_custody, checkpoint)
        source_close()
        scope._fence()
        if ready_population:
            scope._detached_fence()
        else:
            # Genuine claim/revision/process-grant checks belong to the trusted
            # broker. The selected task is now in progress, not prelaunch ready.
            execution._candidate_bytes(json.loads(self._candidate))
            if hasattr(scope, "_launcher"):
                from .candidate_execution import _root_file
                _need(_root_file(Path(scope._launcher["path"])) == scope._launcher["sha256"],
                      "proof-query worker launcher changed before coding dispatch")
        join._require_frozen_proof_inventory(owner=replace(self._owner, timeout_seconds=checkpoint()),
            verification_catalog=self._catalog,
            closure=value["indexed_plan"]["proof_query_closure"], checkpoint=checkpoint)
        _witness_fence(self._witnesses, checkpoint)
        source_close()
        checkpoint()

    def require_detached(self, *, scope):
        """Callback-free inventory, complete plan/reference and physical bytes."""
        self._assert_scope(scope)
        with self._budget() as checkpoint, self._server._lock, self._catalog.store._lock:
            self._physical_close(scope=scope, checkpoint=checkpoint, ready_population=True)

    def require_worker_dispatch_current(self, *, scope, worker_source_custody):
        """Trusted coding broker fence after its genuine claim/grant checks.

        The broker must retain both native owner locks through literal coding
        Popen and acknowledgement. This fence never reconstructs ready rows
        after the separately validated native claim has advanced their revision.
        """
        self._assert_scope(scope)
        with self._budget() as checkpoint, self._server._lock, self._catalog.store._lock:
            self._physical_close(scope=scope, checkpoint=checkpoint, ready_population=False,
                                 worker_source_custody=worker_source_custody)

    def require_prelaunch_current(self, *, scope):
        # The paired receiving entry performs the fresh Python/Lean walk once.
        # Repeated launch preparation calls close those exact owned inputs.
        self.require_detached(scope=scope)

    def bind_runtime(self, runtime):
        from ..entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
        _need(type(runtime) is AdmittedBenchmarkRuntime and self._scope is not None
              and runtime.finite_execution_scope is self._scope
              and self._scope._runtime is runtime and self._runtime in (None, runtime),
              "proof-query closure can bind only its exact native finite runtime")
        self._runtime = runtime
        if self._operation is not None:
            _need(self._operation.purpose == "create" and self._operation.runtime is None,
                  "proof-query construction runtime can bind only once")
            self._operation.runtime = runtime

    def require_runtime(self, *, scope, runtime, stopping=False):
        """Shutdown checks immutable launch ownership, never current proofs."""
        _need(type(self) is FrozenFiniteProofQueryExecutionClosure and self._seal is _SEAL
              and scope is self._scope and runtime is self._runtime
              and runtime.finite_execution_scope is scope
              and _same(runtime.manifest.get("finite_execution_scope"), scope.to_dict())
              and _same(scope.to_dict()["payload"].get("proof_query_closure"), self.material_binding),
              "native runtime lost its immutable proof-query launch binding")
        scope._active(allow_cancelled=stopping)
        if not stopping:
            self._assert_scope(scope)

    def _operation_guard(self, operation, *, closing=False):
        self._assert_scope(self._scope)
        _need(type(operation) is _ReceivingOperation and operation.seal is _OPERATION_SEAL
              and operation is self._operation and not operation.exited
              and operation.pid == os.getpid() and operation.thread_id == threading.get_ident()
              and all(left is right if number not in {2, 13, 14, 15, 20, 22, 23} else left == right
                      for number, (left, right) in enumerate(zip(self._fields(), operation.fields)))
              and len(self._fields()) == len(operation.fields)
              and operation.material_bytes == self._material and (not operation.closed or closing),
              "exact private proof-query receiving inputs changed or operation was reused")
        _need(operation.runtime is self._runtime,
              "proof-query receiving runtime identity changed")

    @contextmanager
    def _receiving_operation(self, *, purpose, runtime=None):
        """One fresh receiving entry; one close before constructor return/Popen."""
        self._assert_scope(self._scope)
        _need((purpose == "create" and runtime is None and self._runtime is None)
              or (purpose == "start" and runtime is self._runtime and runtime is not None),
              "exact unlaunched proof-query native construction or START required")
        _need(not self._scope._spawned and self._operation is None and not self._receiving_entering,
              "proof-query receiving operations cannot be nested or reused")
        self._receiving_entering = True
        checkpoint = admission_boundary._budget(self._owner)
        def current():
            self._scope._active()
            return checkpoint()
        operation = _ReceivingOperation(_OPERATION_SEAL, purpose, runtime, os.getpid(),
            threading.get_ident(), self._fields(), self._material, current)
        self._operation = operation
        try:
            self._operation_guard(operation)
            observed = admission_boundary.verify_current_finite_proof_query_admission(
                owner=replace(self._owner, timeout_seconds=current()), admission=json.loads(self._value),
                verification_catalog=self._catalog,
                output=self._output / ("receiving-" + purpose + "-" + uuid.uuid4().hex),
                policy_observer=self._observer)
            binding = self.material_binding
            _need(observed["admission_cid"] == binding["proof_query_admission_cid"]
                  and observed["proof_query_closure_cid"] == binding["proof_query_closure_cid"]
                  and cid_for_structured(observed["semantic_context"]) == binding["semantic_context_cid"],
                  "fresh receiving proof-query or finite semantics differs from the signed launch")
            self.require_detached(scope=self._scope)
            yield operation
            # A successfully started worker may already have edited source.
            # Exit checks only the consumed private closing capability.
            self._operation_guard(operation, closing=True)
            _need(operation.closed, "proof-query receiving operation has no consumed closing fence")
        finally:
            operation.exited = True
            self._operation = None
            self._receiving_entering = False

    def _close_receiving_operation(self, operation):
        self._operation_guard(operation)
        self.require_detached(scope=self._scope)
        self._operation_guard(operation)
        operation.closed = True

    def _finish_creation_receiving(self, runtime):
        if self._runtime is None:
            self.bind_runtime(runtime)
        operation = self._operation
        _need(operation is not None and operation.purpose == "create" and runtime is self._runtime,
              "exact proof-query native construction operation required")
        with self._server._lock, self._catalog.store._lock:
            self._close_receiving_operation(operation)

    @contextmanager
    def require_spawn_fence(self, *, scope, runtime):
        """Hold native proof-owner lock through final close and literal Popen.

        The trusted runtime must use ``with closure.require_spawn_fence(...)``
        around Popen and the immediate ``note_spawned`` calls. This is a
        supervisor launch boundary, not an arbitrary OS-write atomicity claim.
        """
        self.require_runtime(scope=scope, runtime=runtime)
        operation = self._operation
        _need(operation is not None and operation.purpose == "start" and not scope._spawned,
              "native Popen requires its exact open proof-query START operation")
        with self._server._lock, self._catalog.store._lock:
            self._close_receiving_operation(operation)
            yield

    def note_spawned(self, runtime):
        # The existing finite scope records cleanup duty immediately on birth.
        _need(runtime is self._runtime and self._scope._spawned,
              "proof-query process birth must follow the native finite cleanup record")
        operation = self._operation
        self._operation_guard(operation, closing=True)
        _need(operation.purpose == "start" and operation.closed,
              "native proof-query process birth lacks its consumed closing fence")


def prepare_finite_proof_query_execution_closure(*, owner, admission, verification_catalog,
        server, source, candidate, worker_context, output, policy_observer):
    """Authenticate complete current source/proof/task/public-worker inputs."""
    _need(type(owner) is RepositoryPlanPreviewOwner
          and type(verification_catalog) is CodebaseVerificationCatalog
          and verification_catalog.index is owner.index
          and verification_catalog.store is owner.index.ingestor.store
          and verification_catalog.artifacts is owner.index.artifacts
          and type(server) is QuackStateServer and server.lifecycle is ServerLifecycle.READY
          and type(source) is TypedDatabaseTaskSource and source.execution_route_policy is not None
          and callable(policy_observer), "exact live source, native proof/task owners and observer required")
    checkpoint = admission_boundary._budget(owner)
    value, verified = admission_boundary._received(admission)
    _need(verified["receipt"]["planning_permitted"] is True
          and value["finite_admission"]["local_admission"] is not None,
          "no-work finite proof-query admission cannot reserve worker execution")
    output = Path(output).absolute()
    _need(not output.exists() and output.resolve() == output and not output.is_relative_to(owner.repository),
          "fresh external native proof-query execution evidence directory required")
    candidate, worker_context = _plain(candidate), _plain(worker_context)
    output.mkdir(mode=0o700)
    custody = capture_source_custody(owner, checkpoint)
    current = admission_boundary.verify_current_finite_proof_query_admission(
        owner=replace(owner, timeout_seconds=checkpoint()), admission=value,
        verification_catalog=verification_catalog, output=output / "prepared-current",
        policy_observer=policy_observer)
    semantic = verified["semantic_context"]
    _need(current["admission_cid"] == cid_for_structured(value)
          and _same(current["semantic_context"], semantic),
          "current proof-query admission differs from its independently reconstructed finite context")
    population = execution._native_population(server, source, value["finite_admission"]["local_admission"],
        {**semantic, "finite_admission_cid": cid_for_structured(value["finite_admission"])})
    base_candidate = execution._candidate(candidate, semantic, population,
                                         cid_for_structured(value["finite_admission"]))
    _need(type(worker_context) is dict and set(worker_context) == {"artifact", "sha256", "context_cid",
              "finite_proof_query_admission_cid", "finite_proof_query_closure_cid",
              "finite_repository_candidate_cid", "task_cid", "task_revision"}
          and type(worker_context.get("artifact")) is str
          and type(worker_context.get("sha256")) is str and type(worker_context.get("context_cid")) is str,
          "exact prepared public proof-query worker context descriptor required")
    _need(worker_context["finite_proof_query_admission_cid"] == cid_for_structured(value)
          and worker_context["finite_proof_query_closure_cid"] == value["indexed_plan"]["proof_query_closure_cid"]
          and worker_context["finite_repository_candidate_cid"] == candidate["candidate_cid"]
          and worker_context["task_cid"] == candidate["task_cid"]
          and type(worker_context["task_revision"]) is int
          and worker_context["task_revision"] == candidate["task_revision"],
          "public proof-query context belongs to another admission, candidate or native task revision")
    context_witness, context_raw = read_custody_file(Path(worker_context["artifact"]),
        role="proof-query:public-worker-context", bound=MAX_BYTES, checkpoint=checkpoint)
    _need(hashlib.sha256(context_raw).hexdigest() == worker_context["sha256"],
          "public proof-query worker context bytes differ")
    # The separate public context owner authenticates its closed schema and
    # signatures. It must not read private index owners or profile signing keys.
    from . import finite_proof_query_worker_context as public
    from .finite_repository_candidate_runner import load_finite_repository_candidate
    candidate_body = load_finite_repository_candidate(artifact=Path(candidate["artifact"]),
                                                       expected_sha256=candidate["sha256"])
    public.load_finite_proof_query_worker_context(artifact=Path(worker_context["artifact"]),
        expected_sha256=worker_context["sha256"], expected_context_cid=worker_context["context_cid"],
        candidate=candidate_body)
    public.validate_finite_proof_query_worker_descriptor(descriptor=worker_context,
                                                        candidate=candidate_body)
    closure = value["indexed_plan"]["proof_query_closure"]
    with server._lock, verification_catalog.store._lock:
        plans = _physical_plans(server, {row["plan_cid"] for row in population["tasks"]})
        witnesses = (*_references(plans, value, checkpoint),
                     *_proof_witnesses(owner, closure, checkpoint), context_witness)
        custody.require_current(checkpoint)
        join._require_frozen_proof_inventory(owner=replace(owner, timeout_seconds=checkpoint()),
            verification_catalog=verification_catalog, closure=closure, checkpoint=checkpoint)
        _witness_fence(witnesses, checkpoint)
    from .finite_proof_query_worker_source_custody import PROFILE as worker_source_profile
    material = {"schema": SCHEMA, "profile": PROFILE,
        "boundary": "native-supervisor-receiving-and-before-Popen",
        "proof_query_admission": value, "proof_query_admission_cid": cid_for_structured(value),
        "finite_admission_cid": cid_for_structured(value["finite_admission"]),
        "proof_query_closure_cid": value["indexed_plan"]["proof_query_closure_cid"],
        "semantic_context_cid": cid_for_structured(semantic), "head": owner.expected_head.to_dict(),
        "administrator_task_cids": semantic["administrator_task_cids"], "candidate": candidate,
        "worker_context": worker_context, "native_plan_rows": plans,
        "coding_source_custody_profile": worker_source_profile,
        "native_owner_identity": server.identity.to_dict(),
        "execution_route_policy": source.execution_route_policy.to_dict(),
        "physical_files": [witness.material() for witness in witnesses],
        "model_selection": dict(join._MODEL_OFF), "conditional_evidence_is_context_only": True,
        "parent_proof_query_fence": True, "coding_child_proof_query_fence": False,
        "authority": dict(_FALSE)}
    material["closure_cid"] = cid_for_structured(material)
    prepared = FrozenFiniteProofQueryExecutionClosure(_SEAL, owner=owner, value=value,
        catalog=verification_catalog, server=server, source=source, candidate=candidate,
        base_candidate=base_candidate, output=output, observer=policy_observer,
        material=material, witnesses=witnesses)
    (output / "proof-query-execution-closure.json").write_bytes(_wire(material))
    _witness_fence(witnesses, checkpoint)
    join._require_frozen_proof_inventory(owner=replace(owner, timeout_seconds=checkpoint()),
        verification_catalog=verification_catalog, closure=closure, checkpoint=checkpoint)
    return prepared


@contextmanager
def reserve_finite_proof_query_execution(*, owner, admission, verification_catalog,
        server, source, candidate, worker_context, output, policy_observer,
        cpu_slots=4, memory_mb=4096, child_process_slots=8, admission_timeout_seconds=30,
        closure_output=None):
    """Add exact proof-query custody to existing lease/STOP/cleanup accounting."""
    output = Path(output).absolute()
    destination = Path(closure_output) if closure_output is not None else output.with_name(
        output.name + "-proof-query-closure")
    prepared = prepare_finite_proof_query_execution_closure(owner=owner, admission=admission,
        verification_catalog=verification_catalog, server=server, source=source, candidate=candidate,
        worker_context=worker_context, output=destination, policy_observer=policy_observer)
    with execution.reserve_finite_repository_execution(owner=owner,
            admission=json.loads(prepared._value)["finite_admission"], candidate=candidate,
            server=server, source=source, output=output, policy_observer=policy_observer,
            cpu_slots=cpu_slots, memory_mb=memory_mb, child_process_slots=child_process_slots,
            admission_timeout_seconds=admission_timeout_seconds, proof_query_closure=prepared) as scope:
        yield scope


__all__ = ["PROFILE", "SCHEMA", "EXECUTION_SCHEMA", "FiniteProofQueryExecutionError",
           "FrozenFiniteProofQueryExecutionClosure", "prepare_finite_proof_query_execution_closure",
           "reserve_finite_proof_query_execution"]
