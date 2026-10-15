"""Signed proposal companions joining repository previews to successor inputs.

This is a separate profile, not an execution admission. Existing worker
parsers must reject the companion. Every use repeats native successor and
trusted application root/checker observations; no freshness token survives.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import os
from pathlib import Path
import threading
import time

from ..planning.plan_revision_contracts import PlanAuthorityRoots, PlanCreateRequest
from ..prompt.plan_create_service import (
    PlanCreateMaterials, PlanCreatePreviewReceipt, freeze_plan_create_input_snapshot,
)
from ..proof.formal_verification_contracts import content_identity
from . import local_planning_admission as local
from . import codebase_successor_dispatch_admission as successor
from .codebase_successor_dispatch_context import AUTHORITY_NAMES, _wire
from .codebase_preview_successor_contract import (
    bind_repository_preview_materials, build_preview_successor_bridge,
    require_passive_repository_preview_inputs, require_passive_repository_preview_receipt,
)


def _structural_context_from_current(index, observed):
    """Read the same owner's immutable manifest while its native fence is held."""
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import CodebaseIRManifest, RepositoryCodebaseIndex
    from ..planning.structural_codebase_context import StructuralCodebaseContext

    if type(index) is not RepositoryCodebaseIndex:
        raise ValueError("exact current native successor source owner required")
    head = CodebaseHead.from_dict(observed["head"])
    manifest = index.load(head.manifest_cid)
    if (type(manifest) is not CodebaseIRManifest or manifest.cid != head.manifest_cid
            or manifest.snapshot.repository_id != head.repository_id
            or manifest.snapshot.snapshot_cid != head.snapshot_cid
            or manifest.ast_revision_id != head.ast_revision_id):
        raise ValueError("preview structural manifest differs from current successor head")
    return StructuralCodebaseContext(head, manifest.semantic_state.state_cid, manifest.coverage)


def _guard(resources):
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
        LeaseCancelledError, LeaseTimeoutError,
    )
    duration = resources.get("timeout_seconds", 120.0)
    if (type(duration) not in {int, float} or not math.isfinite(duration)
            or not 0 < duration <= 600):
        raise ValueError("bounded preview bridge deadline required")
    signal = resources.get("cancel_event")
    if signal is not None and not callable(getattr(signal, "is_set", None)):
        raise ValueError("typed preview bridge cancellation signal required")
    deadline = time.monotonic() + duration

    def remaining():
        if signal is not None and signal.is_set():
            raise LeaseCancelledError("preview successor bridge cancelled")
        left = deadline - time.monotonic()
        if left <= 0:
            raise LeaseTimeoutError("preview successor bridge deadline exceeded")
        return left
    return remaining


@contextmanager
def _current_bridge_binding_scope(*, admission, request, materials, preview,
        candidate_bindings, requirement_bindings, checker_policy_cid,
        roots_observer, checker_observer, selection, root, completion, index,
        repository, registry, **resources):
    """Compose sequential owner observations, not an atomic multi-owner lock."""
    if (type(request) is not PlanCreateRequest or type(materials) is not PlanCreateMaterials
            or type(preview) is not PlanCreatePreviewReceipt
            or not callable(roots_observer) or not callable(checker_observer)):
        raise ValueError("typed preview inputs and trusted live observers required")
    remaining = _guard(resources)
    remaining()
    # Nested record subclasses may override serialization. Rehydrate detached
    # base records before the first native observation, preserving exact wires.
    request_value = local._plain(request.to_dict())
    preview_value = local._plain(preview.to_dict())
    request = PlanCreateRequest.from_dict(request_value)
    preview = PlanCreatePreviewReceipt.from_dict(preview_value)
    if (_wire(request.to_dict()) != _wire(request_value)
            or _wire(preview.to_dict()) != _wire(preview_value)):
        raise ValueError("canonical detached preview request and receipt required")
    admission = local._plain(admission)
    candidate_bindings = local._plain(candidate_bindings)
    requirement_bindings = local._plain(requirement_bindings)
    admission_wire = _wire(admission)
    request_wire, preview_wire = _wire(request.to_dict()), _wire(preview.to_dict())
    selection_wire = _wire({"artifact_cid": selection.artifact_cid, "value": selection.to_dict()})
    thread_id = threading.get_ident()
    process_id = os.getpid()
    closed = False

    def unchanged():
        require_passive_repository_preview_inputs(request, materials)
        require_passive_repository_preview_receipt(preview)
        if (os.getpid() != process_id or threading.get_ident() != thread_id or _wire(admission) != admission_wire
                or _wire(request.to_dict()) != request_wire or _wire(preview.to_dict()) != preview_wire
                or _wire({"artifact_cid": selection.artifact_cid, "value": selection.to_dict()}) != selection_wire):
            raise ValueError("preview bridge captured inputs or operation thread changed")
        remaining()

    current_resources = {**resources, "timeout_seconds": remaining()}
    with successor._current_successor_admission_operation(selection=selection, root=root,
            completion=completion, index=index, repository=repository, registry=registry,
            admission=admission, **current_resources) as (observed, close_successor):
        unchanged()
        verified = local.verify_local_benchmark_admission(admission)
        if request.repository_root != str(Path(repository).resolve(strict=True)):
            raise ValueError("preview repository differs from signed successor repository")
        structural = _structural_context_from_current(index, observed)
        structural_wire, observed_wire = _wire(structural.to_dict()), _wire(observed)
        declaration = verified["manifest"]["codebase_successor_context"]
        if declaration["selection_cid"] != selection.artifact_cid:
            raise ValueError("preview bridge selected another successor model lineage")

        def build(context):
            return build_preview_successor_bridge(request=request, materials=materials, preview=preview,
                structural_context=context, graph=verified["graph"], successor_declaration=declaration,
                admission_cid=content_identity(admission), candidate_bindings=candidate_bindings,
                requirement_bindings=requirement_bindings, checker_policy_cid=checker_policy_cid,
                training_record_cid=selection.to_dict()["training_record_cid"])

        payload = build(structural)
        payload_wire = _wire(payload)

        def check_binding():
            # Callback-free custody/material checks may follow a consumed
            # native fence. Fresh owner observations must precede that fence.
            unchanged()
            bound = bind_repository_preview_materials(materials, structural)
            snapshot = freeze_plan_create_input_snapshot(request, materials=bound)
            if (_wire(observed) != observed_wire or _wire(payload) != payload_wire
                    or _wire(snapshot.to_dict()) != _wire(payload["input_snapshot"])):
                raise ValueError("preview bridge receipt, declaration or complete material binding changed")
            unchanged()

        def observe(*, reload_manifest):
            unchanged()
            if _wire(observed) != observed_wire or _wire(payload) != payload_wire:
                raise ValueError("preview bridge observed receipt or declaration changed")
            roots = roots_observer(request)
            unchanged()
            if type(roots) is not PlanAuthorityRoots or _wire(roots.to_dict()) != _wire(request.roots.to_dict()):
                raise ValueError("complete live preview authority roots changed")
            checker = checker_observer(request)
            unchanged()
            if type(checker) is not str or checker != checker_policy_cid:
                raise ValueError("live preview checker policy changed")
            context = _structural_context_from_current(index, observed) if reload_manifest else structural
            if _wire(context.to_dict()) != structural_wire or _wire(build(context)) != payload_wire:
                raise ValueError("preview bridge source, model or complete material binding changed")
            if _wire(observed) != observed_wire or _wire(payload) != payload_wire:
                raise ValueError("preview bridge declaration changed during live observations")
            unchanged()

        observe(reload_manifest=True)

        def close_current():
            nonlocal closed
            if closed:
                raise ValueError("preview successor bridge operation already closed")
            observe(reload_manifest=True)
            close_successor()
            check_binding()
            closed = True
            return payload

        yield payload, verified, close_current
        if not closed:
            raise ValueError("preview successor bridge requires a successful closing fence")
        check_binding()
    check_binding()
    # The first owner's callbacks/cleanup may change application roots. A
    # fresh, bounded native operation observes all roots/checker/materials
    # again, then puts its native source/model close AFTER those callbacks.
    # No application observer, CAS read or signature work follows this final
    # close: only detached pure guards and the receiver's own exit custody.
    final_resources = {**resources, "timeout_seconds": remaining()}
    with successor._current_successor_admission_operation(selection=selection, root=root,
            completion=completion, index=index, repository=repository, registry=registry,
            admission=admission, **final_resources) as (final_observed, final_close):
        if _wire(final_observed) != observed_wire:
            raise ValueError("preview bridge final native observation changed")
        observed = final_observed
        observe(reload_manifest=True)
        final_close()
        check_binding()
    check_binding()


def author_current_preview_successor_bridge(*, admission, request, materials, preview,
        candidate_bindings, requirement_bindings, checker_policy_cid,
        roots_observer, checker_observer, **current):
    """Sign a proposal-only companion; neither install tasks nor launch work."""
    with _current_bridge_binding_scope(admission=admission, request=request, materials=materials,
            preview=preview, candidate_bindings=candidate_bindings, requirement_bindings=requirement_bindings,
            checker_policy_cid=checker_policy_cid, roots_observer=roots_observer,
            checker_observer=checker_observer, **current) as (payload, verified, close):
        bridge = local._signed(local._plain(payload), verified["manifest"])
        captured = _wire(bridge)
        if _wire(local._verify_signature(bridge, verified["profile"])) != _wire(payload):
            raise ValueError("signed preview bridge payload changed")
        close()
        if _wire(bridge) != captured:
            raise ValueError("signed preview bridge changed during closing callbacks")
    if _wire(bridge) != captured:
        raise ValueError("signed preview bridge changed during resource closure")
    return bridge


@contextmanager
def _current_preview_successor_bridge_operation(*, bridge, admission, request, materials, preview,
        roots_observer, checker_observer, **current):
    bridge = local._plain(bridge)
    captured = _wire(bridge)
    if set(bridge) != {"payload", "binding"} or type(bridge["payload"]) is not dict:
        raise ValueError("exact signed preview bridge envelope required")
    declared = bridge["payload"]
    with _current_bridge_binding_scope(admission=admission, request=request, materials=materials,
            preview=preview, candidate_bindings=declared.get("candidate_bindings"),
            requirement_bindings=declared.get("requirement_bindings"),
            checker_policy_cid=declared.get("checker_policy_cid"), roots_observer=roots_observer,
            checker_observer=checker_observer, **current) as (expected, verified, close_binding):
        if _wire(local._verify_signature(bridge, verified["profile"])) != _wire(expected):
            raise ValueError("signed preview bridge differs from complete current inputs")
        receipt = {"schema": "supervisor-current-preview-successor-bridge-verification@1",
            "signed_bridge_cid": content_identity(bridge), "bridge_cid": expected["bridge_cid"],
            "admission_cid": expected["admission_cid"],
            "input_snapshot_cid": expected["input_snapshot"]["snapshot_cid"],
            "administrator_task_cids": sorted(task.task_cid for task in verified["graph"].tasks),
            "head": local._plain(expected["structural_context"]["head"]),
            "successor_selection_cid": verified["manifest"]["codebase_successor_context"]["selection_cid"],
            "checker_policy_cid": expected["checker_policy_cid"], "current_facts": [], "removed_task_cids": [],
            "authority": {key: False for key in sorted(AUTHORITY_NAMES)}, "observed_current": True,
            "worker_profile_integrated": False}
        receipt_wire = _wire(receipt)
        closed = False

        def unchanged():
            if _wire(bridge) != captured or _wire(receipt) != receipt_wire:
                raise ValueError("preview bridge envelope or verification receipt changed")

        def close_current():
            nonlocal closed
            if closed:
                raise ValueError("preview bridge verification already closed")
            unchanged()
            close_binding()
            unchanged()
            closed = True
            return receipt

        yield receipt, close_current
        if not closed:
            raise ValueError("preview bridge verification requires a successful closing fence")
        unchanged()
    unchanged()


def verify_current_preview_successor_bridge(**arguments):
    """Reobserve current native inputs and all companion bindings on each use."""
    with _current_preview_successor_bridge_operation(**arguments) as (receipt, close):
        close()
    return receipt


__all__ = ["author_current_preview_successor_bridge", "verify_current_preview_successor_bridge"]
