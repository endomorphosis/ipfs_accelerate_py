"""Current-owner admission for an explicit preview-bound successor worker.

The worker manifest is a signed child of an immutable @6 admission and its
proposal companion. Application observations finish unlocked; a separate,
one-use native source/model close precedes installation or worker launch.
Observations are sequential and confer no proof or production authority.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, fields
import os
from pathlib import Path
import threading

from ..proof.formal_verification_contracts import content_identity
from ..prompt.plan_create_service import freeze_plan_create_input_snapshot
from ..planning.plan_revision_contracts import PlanAuthorityRoots
from ..task_sources.intent_repository import IntentRepository
from . import local_planning_admission as local
from . import codebase_preview_successor_admission as companion
from . import codebase_successor_dispatch_admission as successor
from .codebase_preview_successor_contract import (
    build_preview_successor_bridge,
    require_passive_repository_preview_inputs, require_passive_repository_preview_receipt,
)
from .codebase_successor_dispatch_context import AUTHORITY_NAMES, _wire


@dataclass(frozen=True)
class PreviewSuccessorBridgeInputs:
    """Private owner inputs; observers must finish before the task-owner lock."""

    request: object
    materials: object
    preview: object
    roots_observer: object
    checker_observer: object


def _input_wires(value):
    if type(value) is not PreviewSuccessorBridgeInputs:
        raise ValueError("exact private preview successor bridge inputs required")
    attributes = object.__getattribute__(value, "__dict__")
    if type(attributes) is not dict or set(attributes) != {item.name for item in fields(value)}:
        raise ValueError("private preview inputs cannot override instance behavior")
    require_passive_repository_preview_inputs(value.request, value.materials)
    require_passive_repository_preview_receipt(value.preview)
    if not callable(value.roots_observer) or not callable(value.checker_observer):
        raise ValueError("trusted live preview root and checker observers required")
    # The unaugmented snapshot binds every original material field. Its
    # serializer is permitted only after the exact passive-record guards.
    return (_wire(value.request.to_dict()), _wire(value.preview.to_dict()),
            _wire(freeze_plan_create_input_snapshot(value.request,
                materials=value.materials).to_dict()))


def capture_preview_successor_bridge_inputs(value):
    """Capture identity and callback-free material custody for one boundary."""
    wires = _input_wires(value)
    identities = (id(value), *(id(object.__getattribute__(value, item.name))
                              for item in fields(PreviewSuccessorBridgeInputs)))
    return identities, wires


def require_preview_successor_bridge_inputs(value, captured):
    if capture_preview_successor_bridge_inputs(value) != captured:
        raise ValueError("private preview owner inputs or complete material binding changed")


def _companion_arguments(*, bridge_inputs, **arguments):
    _input_wires(bridge_inputs)
    return {**arguments, **{name: object.__getattribute__(bridge_inputs, name)
                          for name in ("request", "materials", "preview", "roots_observer", "checker_observer")}}


class _PreviewSuccessorWorkerClosingFence:
    """Unlocked application preparation followed by a native-only close."""

    __slots__ = ("_prepare", "_close", "_prepared", "_consumed", "_pid", "_thread")

    def __init__(self, prepare, close):
        for name, value in (("_prepare", prepare), ("_close", close),
                ("_prepared", False), ("_consumed", False),
                ("_pid", os.getpid()), ("_thread", threading.get_ident())):
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        raise ValueError("private preview worker closing fence cannot be rebound")

    def _same_thread(self):
        if os.getpid() != self._pid or threading.get_ident() != self._thread:
            raise ValueError("preview worker closing fence changed process or thread")

    def prepare(self):
        self._same_thread()
        if self._prepared or self._consumed:
            raise ValueError("preview worker closing preparation already consumed")
        # A failed preparation cannot be retried on the same operation.
        object.__setattr__(self, "_consumed", True)
        self._prepare()
        object.__setattr__(self, "_prepared", True)
        object.__setattr__(self, "_consumed", False)

    def __call__(self):
        self._same_thread()
        if not self._prepared or self._consumed:
            raise ValueError("preview worker requires unused unlocked closing preparation")
        object.__setattr__(self, "_consumed", True)
        return self._close()


@contextmanager
def _current_preview_successor_worker_operation(*, worker_admission, bridge_inputs,
        selection, root, completion, index, repository, registry, **resources):
    """Fresh paired receiving; no application callbacks follow native close.

    The proposal companion's complete two-pass operation is consumed before
    yielding. The last native operation then spans the caller's work. Its
    preparation reobserves all application roots and full materials unlocked;
    its close checks native sources/model after those observations.
    """
    from .codebase_preview_successor_worker_context import validate_preview_successor_declaration

    captured_inputs = capture_preview_successor_bridge_inputs(bridge_inputs)
    worker_admission = local._plain(worker_admission)
    worker_wire = _wire(worker_admission)
    remaining = companion._guard(resources)
    remaining()
    verified = local.verify_local_benchmark_admission(worker_admission)
    if verified["manifest"]["schema"] != local.PREVIEW_SUCCESSOR_MANIFEST_SCHEMA:
        raise ValueError("explicit signed preview successor worker admission required")
    declaration = validate_preview_successor_declaration(
        verified["manifest"]["codebase_preview_successor_context"],
        manifest=verified["manifest"], graph=verified["graph"], profile=verified["profile"])
    parent, bridge = declaration["parent_admission"], declaration["signed_bridge"]
    payload = bridge["payload"]
    parent_wire, bridge_wire = _wire(parent), _wire(bridge)
    memory = resources.get("memory_mb", 1024)
    metadata_size = (len(worker_wire) + len(parent_wire) + len(bridge_wire)
                     + sum(len(raw) for raw in captured_inputs[1]))
    if (type(memory) is not int or not 1024 <= memory <= 4096
            or 248 * 1024 * 1024 + 4 * metadata_size > memory * 1024 * 1024 // 4):
        raise ValueError("preview worker signed metadata exceeds its bounded memory reservation")
    owner_arguments = dict(selection=selection, root=root, completion=completion,
        index=index, repository=repository, registry=registry)
    initial_resources = {**resources, "timeout_seconds": remaining()}
    initial = companion.verify_current_preview_successor_bridge(
        **_companion_arguments(bridge_inputs=bridge_inputs, bridge=bridge,
            admission=parent, **owner_arguments, **initial_resources))
    require_preview_successor_bridge_inputs(bridge_inputs, captured_inputs)
    pid, thread_id = os.getpid(), threading.get_ident()
    prepared = closed = False

    def custody():
        require_preview_successor_bridge_inputs(bridge_inputs, captured_inputs)
        if (os.getpid() != pid or threading.get_ident() != thread_id
                or _wire(worker_admission) != worker_wire
                or _wire(parent) != parent_wire or _wire(bridge) != bridge_wire):
            raise ValueError("preview worker signed inputs or operation thread changed")

    final_resources = {**resources, "timeout_seconds": remaining()}
    with successor._current_successor_admission_operation(admission=parent,
            **owner_arguments, **final_resources) as (observed, close_native):
        custody()
        observed_wire = _wire(observed)
        if (observed["admission_cid"] != declaration["parent_admission_cid"]
                or observed["successor_selection_cid"] != initial["successor_selection_cid"]):
            raise ValueError("preview worker final native selection changed")
        structural = companion._structural_context_from_current(index, observed)
        structural_wire = _wire(structural.to_dict())
        if _wire(structural.to_dict()["head"]) != _wire(initial["head"]):
            raise ValueError("preview worker final structural native head changed")
        receipt = {**local._plain(observed),
            "schema": "supervisor-current-preview-successor-worker-verification@1",
            "admission_cid": content_identity(worker_admission),
            "manifest_cid": verified["receipt"]["manifest_cid"],
            "graph_cid": verified["graph"].content_id,
            "plan_id": verified["receipt"]["plan_id"],
            "pending_cid": verified["receipt"]["pending_cid"],
            "parent_admission_cid": declaration["parent_admission_cid"],
            "signed_bridge_cid": declaration["signed_bridge_cid"],
            "bridge_cid": declaration["bridge_cid"],
            "codebase_preview_successor_context_cid": content_identity(declaration),
            "input_snapshot_cid": payload["input_snapshot"]["snapshot_cid"],
            "checker_policy_cid": payload["checker_policy_cid"],
            "worker_profile_integrated": True,
            "authority": {key: False for key in sorted(AUTHORITY_NAMES)}}
        receipt_wire = _wire(receipt)

        def passive():
            custody()
            if _wire(observed) != observed_wire or _wire(receipt) != receipt_wire:
                raise ValueError("preview worker observation or verification receipt changed")

        def observe_application():
            passive()
            remaining()
            current_roots = bridge_inputs.roots_observer(bridge_inputs.request)
            passive()
            if (type(current_roots) is not PlanAuthorityRoots
                    or _wire(current_roots.to_dict()) != _wire(bridge_inputs.request.roots.to_dict())):
                raise ValueError("complete live preview worker authority roots changed")
            current_checker = bridge_inputs.checker_observer(bridge_inputs.request)
            passive()
            if type(current_checker) is not str or current_checker != payload["checker_policy_cid"]:
                raise ValueError("live preview worker checker policy changed")
            current_structural = companion._structural_context_from_current(index, observed)
            if _wire(current_structural.to_dict()) != structural_wire:
                raise ValueError("preview worker current structural source manifest changed")
            expected = build_preview_successor_bridge(request=bridge_inputs.request,
                materials=bridge_inputs.materials, preview=bridge_inputs.preview,
                structural_context=current_structural, graph=verified["graph"],
                successor_declaration=verified["manifest"]["codebase_successor_context"],
                admission_cid=declaration["parent_admission_cid"],
                candidate_bindings=payload["candidate_bindings"],
                requirement_bindings=payload["requirement_bindings"],
                checker_policy_cid=payload["checker_policy_cid"],
                training_record_cid=selection.to_dict()["training_record_cid"])
            if _wire(expected) != _wire(payload):
                raise ValueError("preview worker complete live companion binding changed")
            passive()
            remaining()

        observe_application()

        def prepare():
            nonlocal prepared
            if prepared or closed:
                raise ValueError("preview worker application preparation already consumed")
            observe_application()
            prepared = True

        def close():
            nonlocal closed
            if not prepared or closed:
                raise ValueError("preview worker final close requires its own unused application preparation")
            passive()
            # Every observer, CAS read, compiler and signature callback has
            # already finished. This one-use native close is the last owner
            # freshness operation before commit/Popen.
            if _wire(close_native()) != observed_wire:
                raise ValueError("preview worker final source/model fence changed")
            passive()
            closed = True
            return receipt

        closing = _PreviewSuccessorWorkerClosingFence(prepare, close)
        yield receipt, closing
        if not closed:
            raise ValueError("preview worker operation requires a successful closing fence")
        passive()
    passive()


def author_current_preview_successor_worker_manifest(*, parent_admission, bridge,
        bridge_inputs, **current):
    """Sign a new child profile without changing the original proposal hashes."""
    from .codebase_preview_successor_worker_context import preview_successor_declaration

    captured_inputs = capture_preview_successor_bridge_inputs(bridge_inputs)
    parent_admission, bridge = local._plain(parent_admission), local._plain(bridge)
    with companion._current_preview_successor_bridge_operation(
            **_companion_arguments(bridge_inputs=bridge_inputs, admission=parent_admission,
                bridge=bridge, **current)) as (_, close):
        verified = local.verify_local_benchmark_admission(parent_admission)
        payload = {**local._plain(verified["manifest"]),
            "schema": local.PREVIEW_SUCCESSOR_MANIFEST_SCHEMA,
            "codebase_preview_successor_context": preview_successor_declaration(parent_admission, bridge)}
        local._validate_local_manifest_declarations(payload)
        manifest = local._signed(payload, verified["manifest"])
        local._manifest(manifest, initial=True)
        captured = _wire(manifest)
        close()
        require_preview_successor_bridge_inputs(bridge_inputs, captured_inputs)
        if _wire(manifest) != captured:
            raise ValueError("signed preview worker manifest changed during closing callbacks")
    require_preview_successor_bridge_inputs(bridge_inputs, captured_inputs)
    if _wire(manifest) != captured:
        raise ValueError("signed preview worker manifest changed during resource closure")
    return manifest


def admit_current_preview_successor_worker_plan(*, manifest, graph, bridge_inputs, **current):
    captured_inputs = capture_preview_successor_bridge_inputs(bridge_inputs)
    manifest = local._plain(manifest)
    if type(graph) is not local.PromptGoalGraph:
        raise ValueError("exact independently authored complete preview worker graph required")
    admission = local.admit_local_benchmark_plan(manifest=manifest,
        graph=local.PromptGoalGraph.from_dict(graph.to_dict()))
    admission_wire = _wire(local._plain(admission))
    verify_current_preview_successor_worker_admission(worker_admission=admission,
        bridge_inputs=bridge_inputs, **current)
    require_preview_successor_bridge_inputs(bridge_inputs, captured_inputs)
    if _wire(local._plain(admission)) != admission_wire:
        raise ValueError("produced preview worker admission changed during current-owner callbacks")
    return local._plain(admission)


def verify_current_preview_successor_worker_admission(**arguments):
    with _current_preview_successor_worker_operation(**arguments) as (receipt, close):
        close.prepare()
        close()
    return receipt


class _PreparedIntentPopulation:
    """Prepare signed rows without holding the target Intent writer lock.

    The private local materializer only needs these four writes and a fresh
    task lookup. Recording their exact builtin arguments permits full signing
    and application observation before the final native write transaction.
    Nothing in this staging object makes tasks schedulable.
    """

    _WRITES = frozenset({"upsert_objective", "upsert_goal", "upsert_plan", "upsert_task"})

    def __init__(self, intent):
        self.owner_id, self.session_id = intent.owner_id, intent.session_id
        self._intent = intent
        self.operations = []

    def get_task(self, cid):
        return self._intent.get_task(cid)

    def __getattr__(self, name):
        if name not in self._WRITES:
            raise AttributeError(name)

        def record(**arguments):
            if len(self.operations) >= 64:
                raise ValueError("preview worker staged native population exceeds its bound")
            self.operations.append({"method": name, "arguments": local._plain(arguments)})
        return record


def materialize_current_preview_successor_worker_plan(*, worker_admission,
        bridge_inputs, intent, **current):
    if type(intent) is not IntentRepository or intent.uses_bound_connection:
        raise ValueError("independently owned native preview worker transaction required")
    worker_admission = local._plain(worker_admission)
    admission_wire = _wire(worker_admission)
    # Enter and prepare all application callbacks outside the Intent writer
    # lock. Close/exit the current native owner INSIDE the transaction so any
    # final fence or cleanup failure still rolls back the complete population.
    with ExitStack() as receiving:
        observed, close = receiving.enter_context(_current_preview_successor_worker_operation(
            worker_admission=worker_admission, bridge_inputs=bridge_inputs, **current))
        staged = _PreparedIntentPopulation(intent)
        result = local._materialize_local_benchmark_plan(admission=worker_admission, intent=staged)
        expected = observed["administrator_task_cids"]
        staged_tasks = sorted(row["arguments"]["task_cid"] for row in staged.operations
                              if row["method"] == "upsert_task")
        if sorted(result["task_cids"]) != expected or staged_tasks != expected:
            raise ValueError("preview worker preparation omitted an independently authored task")
        result_wire, staged_wire = _wire(result), _wire(staged.operations)
        replay = local._plain(staged.operations)
        if (any(type(row) is not dict or set(row) != {"method", "arguments"}
                or row["method"] not in _PreparedIntentPopulation._WRITES
                or type(row["arguments"]) is not dict for row in replay)
                or _wire(replay) != staged_wire):
            raise ValueError("closed prepared preview worker native writes required")
        memory = current.get("memory_mb", 1024)
        if 248 * 1024 * 1024 + 4 * (len(admission_wire) + len(staged_wire)) > memory * 1024 * 1024 // 4:
            raise ValueError("preview worker staged population exceeds its bounded memory reservation")
        close.prepare()
        if (_wire(staged.operations) != staged_wire or _wire(result) != result_wire
                or _wire(worker_admission) != admission_wire
                or staged.owner_id != intent.owner_id or staged.session_id != intent.session_id):
            raise ValueError("preview worker prepared native population changed during callbacks")
        with intent._connection(write=True) as connection:
            with IntentRepository(bound_connection=connection, owner_id=intent.owner_id,
                    session_id=intent.session_id) as bound:
                if any(bound.get_task(cid) is not None for cid in expected):
                    raise ValueError("preview worker requires new native tasks at commit")
                for row in replay:
                    if _wire(staged.operations) != staged_wire or _wire(replay) != staged_wire:
                        raise ValueError("preview worker prepared dispatch changed during native callbacks")
                    getattr(bound, row["method"])(**row["arguments"])
                if (_wire(staged.operations) != staged_wire or _wire(result) != result_wire
                        or _wire(replay) != staged_wire or _wire(worker_admission) != admission_wire):
                    raise ValueError("preview worker signed population changed during native installation")
                close()
                receiving.close()
                if (_wire(worker_admission) != admission_wire or _wire(result) != result_wire
                        or _wire(staged.operations) != staged_wire or _wire(replay) != staged_wire):
                    raise ValueError("preview worker materialization changed before native commit")
    return {**result, "schema": "supervisor-current-preview-successor-worker-materialization@1",
        **{key: observed[key] for key in ("parent_admission_cid", "signed_bridge_cid", "bridge_cid",
            "codebase_preview_successor_context_cid", "successor_selection_cid", "source_delta_cid")},
        "administrator_task_population_preserved": True, "current_facts": [], "removed_task_cids": [],
        "runtime_requirements_preserved": True, "observed_current": True,
        "authority": {key: False for key in sorted(AUTHORITY_NAMES)}}


def prepare_current_preview_successor_worker_context(*, worker_admission,
        bridge_inputs, task_cid, source_path, expected_source_sha256, **current):
    from .router_public_instruction import prepare_public_instruction_context

    worker_admission = local._plain(worker_admission)
    with _current_preview_successor_worker_operation(worker_admission=worker_admission,
            bridge_inputs=bridge_inputs, **current) as (_, close):
        full = successor._full_context(current["selection"], current["root"],
            current["completion"], current["index"], current.get("evidence_record"))
        selected = prepare_public_instruction_context(repository=Path(current["repository"]),
            admission=worker_admission, task_cid=task_cid, source_path=source_path,
            expected_source_sha256=expected_source_sha256,
            codebase_inventory_context=full["inventory"], codebase_successor_context=full)
        captured = _wire(selected)
        close.prepare()
        close()
        if _wire(selected) != captured:
            raise ValueError("preview worker public descriptor changed during closing callbacks")
    if _wire(selected) != captured:
        raise ValueError("preview worker public descriptor changed during resource closure")
    return selected


def reserve_current_preview_successor_execution(*, selection, bridge_inputs, **kwargs):
    from .codebase_inventory_execution import reserve_inventory_execution

    return reserve_inventory_execution(successor_selection=selection,
        preview_bridge_inputs=bridge_inputs, **kwargs)


__all__ = ["PreviewSuccessorBridgeInputs", "author_current_preview_successor_worker_manifest",
    "admit_current_preview_successor_worker_plan", "verify_current_preview_successor_worker_admission",
    "materialize_current_preview_successor_worker_plan", "prepare_current_preview_successor_worker_context",
    "reserve_current_preview_successor_execution"]
