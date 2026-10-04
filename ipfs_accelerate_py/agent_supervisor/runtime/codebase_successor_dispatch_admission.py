"""Fresh signed local tasks bound to an explicitly trained successor model.

The versioned declaration adds source/model lineage to the existing full-task
local contract. Learned features never remove tasks or acceptance checks. No
operation here trains, infers, promotes a model or establishes a CodeProof.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
from pathlib import Path
import time

from . import local_planning_admission as local
from . import codebase_inventory_evidence_admission as inventory
from .codebase_successor_dispatch_context import (
    AUTHORITY_NAMES, SCHEMA, _wire, successor_declaration, validate_successor_context,
)
from ..proof.formal_verification_contracts import content_identity
from ..task_sources.intent_repository import IntentRepository


def _full_context(selection, root, completion, index, evidence_record):
    from ipfs_datasets_py.logic.software_contracts.codebase_inventory_successor import load_codebase_source_delta

    source = load_codebase_source_delta(index.artifacts, selection.to_dict()["source_delta_cid"])
    return validate_successor_context({"schema": SCHEMA,
        "selection": {"artifact_cid": selection.artifact_cid, "value": selection.to_dict()},
        "source_delta": {"artifact_cid": source.artifact_cid, "value": source.to_dict()},
        "inventory": inventory._full_context(root, completion, evidence_record),
        "authority": {key: False for key in sorted(AUTHORITY_NAMES)}})


@contextmanager
def _current_scope(*, selection, root, completion, index, repository, registry, evidence_record=None,
                   verification_catalog=None, scheduler=None, parent_lease=None, cancel_event=None,
                   admission_timeout_seconds=30.0, timeout_seconds=120.0, memory_mb=1024):
    from ipfs_datasets_py.logic.software_contracts.codebase_inventory_resume import (
        CodebaseScanResumeRoot, CodebaseScanResumeCompletion,
    )
    from ipfs_datasets_py.logic.software_contracts.codebase_inventory_successor_model import CodebaseSuccessorScanRecord
    from ipfs_datasets_py.logic.software_contracts.codebase_inventory_successor_receiving import (
        _paired_current_codebase_successor_completion,
    )
    from ipfs_datasets_py.logic.software_contracts.codebase_resources import acquire_codebase_resources
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseCancelledError, LeaseTimeoutError

    if (type(selection) is not CodebaseSuccessorScanRecord or type(root) is not CodebaseScanResumeRoot
            or type(completion) is not CodebaseScanResumeCompletion):
        raise ValueError("exact selected successor and complete native scan required")
    if (type(timeout_seconds) not in {int, float} or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 600
            or type(admission_timeout_seconds) not in {int, float} or not math.isfinite(admission_timeout_seconds)
            or not 0 <= admission_timeout_seconds <= 600 or type(memory_mb) is not int or not 1024 <= memory_mb <= 4096):
        raise ValueError("bounded successor admission resources required")
    if cancel_event is not None and not callable(getattr(cancel_event, "is_set", None)):
        raise ValueError("typed successor cancellation signal required")
    if evidence_record is not None:
        from ipfs_datasets_py.logic.software_contracts.codebase_inventory_evidence import (
            CodebaseInventoryEvidenceRecord, validate_current_inventory_evidence,
        )
        if type(evidence_record) is not CodebaseInventoryEvidenceRecord or verification_catalog is None:
            raise ValueError("exact optional conditional evidence record and owner required")
    deadline = time.monotonic() + timeout_seconds
    record_wires = tuple(_wire(record.to_dict()) for record in (selection, root, completion))

    def remaining(signal=None):
        if ((cancel_event is not None and cancel_event.is_set()) or (signal is not None and signal.is_set())):
            raise LeaseCancelledError("successor admission cancelled")
        left = deadline - time.monotonic()
        if left <= 0:
            raise LeaseTimeoutError("successor admission deadline exceeded")
        return left

    with acquire_codebase_resources(scheduler=scheduler, parent_lease=parent_lease, cancel_event=cancel_event,
            timeout_seconds=min(admission_timeout_seconds, remaining()), memory_mb=memory_mb) as lease:
        signal = lease.combined_cancellation_signal(cancel_event)
        context = _full_context(selection, root, completion, index, evidence_record)
        declaration = successor_declaration(context)
        context_wire, declaration_wire = _wire(context), _wire(declaration)
        # Count actual bounded signed metadata beside the private receiver's
        # phased ceiling. This is serialized accounting, never RSS containment.
        if 248 * 1024 * 1024 + 2 * (len(context_wire) + len(declaration_wire)) > memory_mb * 1024 * 1024 // 4:
            raise ValueError("successor signed metadata exceeds memory reservation")

        def unchanged():
            if (tuple(_wire(record.to_dict()) for record in (selection, root, completion)) != record_wires
                    or _wire(context) != context_wire or _wire(declaration) != declaration_wire
                    or _wire(completion.advisory_refs(root)) != _wire(context["inventory"]["scan"])
                    or (evidence_record is not None and _wire(evidence_record.advisory_refs()) != _wire(context["inventory"]["evidence"]))):
                raise ValueError("successor admission inputs changed during owner callbacks")

        def evidence_current():
            if evidence_record is not None:
                if validate_current_inventory_evidence(evidence_record, index, repository, registry=registry,
                        verification_catalog=verification_catalog, parent_lease=lease, cancel_event=signal,
                        admission_timeout_seconds=min(admission_timeout_seconds, remaining(signal)),
                        timeout_seconds=remaining(signal), memory_mb=memory_mb) is not evidence_record:
                    raise ValueError("successor receiving selected another conditional evidence record")
            unchanged()
            remaining(signal)

        with _paired_current_codebase_successor_completion(selection, completion, index, repository,
                root=root, registry=registry, parent_lease=lease, cancel_event=signal,
                admission_timeout_seconds=min(admission_timeout_seconds, remaining(signal)),
                timeout_seconds=remaining(signal), memory_mb=memory_mb) as close_scan:
            evidence_current()

            def current():
                unchanged()
                evidence_current()
                if close_scan() is not completion:
                    raise ValueError("successor receiving selected another scan completion")
                unchanged()
                remaining(signal)

            yield context, declaration, current
        unchanged()
    remaining()
    if tuple(_wire(record.to_dict()) for record in (selection, root, completion)) != record_wires:
        raise ValueError("immutable successor records changed during resource closure")


def _bound_context(manifest, context, declaration):
    from .codebase_inventory_evidence_worker_context import inventory_declaration
    payload = manifest.get("payload", {})
    if (payload.get("schema") != local.SUCCESSOR_MANIFEST_SCHEMA
            or _wire(payload.get("codebase_successor_context")) != _wire(declaration)
            or _wire(payload.get("codebase_inventory_context")) != _wire(inventory_declaration(context["inventory"]))):
        raise ValueError("signed successor manifest differs from complete selected inputs")


def author_current_successor_manifest(selection, root, completion, index, repository, *, registry,
        profile_dir, lifecycle_dir, task_specs, planning_roots, planning_inputs=None, **resources):
    from .codebase_inventory_evidence_worker_context import inventory_declaration
    with _current_scope(selection=selection, root=root, completion=completion, index=index, repository=repository,
                        registry=registry, **resources) as (context, declaration, current):
        manifest = local.author_local_benchmark_manifest(repository=Path(repository), profile_dir=Path(profile_dir),
            lifecycle_dir=Path(lifecycle_dir), task_specs=task_specs, planning_roots=planning_roots,
            planning_inputs=planning_inputs, codebase_inventory_context=inventory_declaration(context["inventory"]),
            codebase_successor_context=declaration)
        _bound_context(manifest, context, declaration)
        local._manifest(manifest, initial=True)
        manifest_wire = _wire(manifest)
        current()
        if _wire(manifest) != manifest_wire:
            raise ValueError("signed successor manifest changed during closing callbacks")
    if _wire(manifest) != manifest_wire:
        raise ValueError("signed successor manifest changed during resource closure")
    return manifest


def admit_current_successor_plan(*, selection, root, completion, index, repository, registry, manifest, graph, **resources):
    manifest = local._plain(manifest)
    if type(graph) is not local.PromptGoalGraph:
        raise ValueError("exact independently authored complete task graph required")
    graph = local.PromptGoalGraph.from_dict(graph.to_dict())
    manifest_wire, graph_wire = _wire(manifest), _wire(graph.to_dict())
    with _current_scope(selection=selection, root=root, completion=completion, index=index, repository=repository,
                        registry=registry, **resources) as (context, declaration, current):
        _bound_context(manifest, context, declaration)
        admission = local.admit_local_benchmark_plan(graph=graph, manifest=manifest)
        local.verify_local_benchmark_admission(admission)
        admission_wire = _wire(local._plain(admission))
        current()
        if (_wire(manifest) != manifest_wire or _wire(graph.to_dict()) != graph_wire
                or _wire(local._plain(admission)) != admission_wire):
            raise ValueError("successor authored graph or admission changed during closing callbacks")
    if (_wire(manifest) != manifest_wire or _wire(graph.to_dict()) != graph_wire
            or _wire(local._plain(admission)) != admission_wire):
        raise ValueError("successor authored graph or admission changed during resource closure")
    return admission


@contextmanager
def _current_successor_admission_operation(*, selection, root, completion, index, repository, registry,
                                         admission, **resources):
    admission = local._plain(admission)
    captured = _wire(admission)
    with _current_scope(selection=selection, root=root, completion=completion, index=index, repository=repository,
                        registry=registry, **resources) as (context, declaration, current):
        _bound_context(admission.get("manifest", {}), context, declaration)
        verified = local.verify_local_benchmark_admission(admission)
        receipt = {"schema": "supervisor-current-successor-admission-verification@1",
            "admission_cid": content_identity(admission), "manifest_cid": verified["receipt"]["manifest_cid"],
            "graph_cid": verified["graph"].content_id, "plan_id": verified["receipt"]["plan_id"],
            "codebase_inventory_context_cid": declaration["inventory_context_cid"],
            "codebase_successor_context_cid": declaration["full_context_cid"],
            "successor_selection_cid": selection.artifact_cid, "source_delta_cid": declaration["source_delta_cid"],
            "root_cid": root.artifact_cid, "completion_cid": completion.artifact_cid,
            "head": declaration["current_head"], "previous_head": declaration["previous_head"],
            "administrator_task_cids": sorted(task.task_cid for task in verified["graph"].tasks),
            "pending_cid": verified["receipt"]["pending_cid"], "current_facts": [], "removed_task_cids": [],
            "runtime_requirements_preserved": True, "observed_current": True,
            "native_persistence_verified_here": False, "authority": dict(context["authority"])}
        receipt_wire = _wire(receipt)

        def close_current():
            if _wire(admission) != captured or _wire(receipt) != receipt_wire:
                raise ValueError("successor admission verification changed during callbacks")
            current()
            if _wire(admission) != captured or _wire(receipt) != receipt_wire:
                raise ValueError("successor admission verification changed during callbacks")
            return receipt
        yield receipt, close_current


def verify_current_successor_admission(*, selection, root, completion, index, repository, registry, admission, **resources):
    with _current_successor_admission_operation(selection=selection, root=root, completion=completion, index=index,
            repository=repository, registry=registry, admission=admission, **resources) as (receipt, close):
        close()
    return receipt


def materialize_current_successor_plan(*, selection, root, completion, index, repository, registry,
                                      admission, intent, **resources):
    if type(intent) is not IntentRepository or intent.uses_bound_connection:
        raise ValueError("independently owned native intent transaction required")
    admission = local._plain(admission)
    admission_wire = _wire(admission)
    with intent._connection(write=True) as connection:
        with _current_scope(selection=selection, root=root, completion=completion, index=index, repository=repository,
                            registry=registry, **resources) as (context, declaration, current):
            _bound_context(admission.get("manifest", {}), context, declaration)
            verified = local.verify_local_benchmark_admission(admission)
            expected = sorted(task.task_cid for task in verified["graph"].tasks)
            with IntentRepository(bound_connection=connection, owner_id=intent.owner_id, session_id=intent.session_id) as bound:
                result = local._materialize_local_benchmark_plan(admission=admission, intent=bound)
                if sorted(result["task_cids"]) != expected:
                    raise ValueError("native successor install omitted an authored task")
                local.verify_local_benchmark_admission(admission)
                result_wire = _wire(result)
                current()
                if _wire(admission) != admission_wire or _wire(result) != result_wire:
                    raise ValueError("successor native materialization changed during closing callbacks")
        if _wire(admission) != admission_wire or _wire(result) != result_wire:
            raise ValueError("successor native materialization changed during resource closure before commit")
    return {**result, "schema": "supervisor-current-successor-native-materialization@1",
        "codebase_inventory_context_cid": declaration["inventory_context_cid"],
        "codebase_successor_context_cid": declaration["full_context_cid"],
        "successor_selection_cid": selection.artifact_cid, "source_delta_cid": declaration["source_delta_cid"],
        "root_cid": root.artifact_cid, "completion_cid": completion.artifact_cid,
        "administrator_task_population_preserved": True, "current_facts": [], "removed_task_cids": [],
        "runtime_requirements_preserved": True, "observed_current": True, "authority": dict(context["authority"])}


def prepare_current_successor_worker_context(*, selection, root, completion, index, repository, registry,
        admission, task_cid, source_path, expected_source_sha256, **resources):
    from .router_public_instruction import prepare_public_instruction_context
    admission = local._plain(admission)
    with _current_scope(selection=selection, root=root, completion=completion, index=index, repository=repository,
                        registry=registry, **resources) as (context, declaration, current):
        _bound_context(admission.get("manifest", {}), context, declaration)
        selected = prepare_public_instruction_context(repository=Path(repository), admission=admission, task_cid=task_cid,
            source_path=source_path, expected_source_sha256=expected_source_sha256,
            codebase_inventory_context=context["inventory"], codebase_successor_context=context)
        selected_wire = _wire(selected)
        current()
        if _wire(selected) != selected_wire:
            raise ValueError("successor public worker descriptor changed during closing callbacks")
    if _wire(selected) != selected_wire:
        raise ValueError("successor public worker descriptor changed during resource closure")
    return selected


def reserve_current_successor_execution(*, selection, **kwargs):
    from .codebase_inventory_execution import reserve_inventory_execution
    return reserve_inventory_execution(successor_selection=selection, **kwargs)


__all__ = ["author_current_successor_manifest", "admit_current_successor_plan", "verify_current_successor_admission",
    "materialize_current_successor_plan", "prepare_current_successor_worker_context", "reserve_current_successor_execution"]
