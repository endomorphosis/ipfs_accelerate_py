"""Fresh completed inventory references in a full signed local task admission.

This additive profile delegates task authority to the existing native pending
contract. Scan features and conditional query rows never remove tasks or
acceptance checks. Owner/source/model checks are sequential observations, not
an atomic repository, model and evidence-owner snapshot.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
from pathlib import Path
import time

from . import local_planning_admission as local
from .codebase_inventory_evidence_worker_context import (
    AUTHORITY_NAMES, SCHEMA, _wire, inventory_declaration, validate_inventory_context,
)
from ..proof.formal_verification_contracts import content_identity
from ..task_sources.intent_repository import IntentRepository


def _full_context(root, completion, evidence_record=None):
    return validate_inventory_context({"schema": SCHEMA, "scan": completion.advisory_refs(root),
        "evidence": None if evidence_record is None else evidence_record.advisory_refs(),
        "authority": {key: False for key in sorted(AUTHORITY_NAMES)}})


@contextmanager
def _current_scope(*, root, completion, index, repository, registry, evidence_record=None,
                   verification_catalog=None, scheduler=None, parent_lease=None, cancel_event=None,
                   admission_timeout_seconds=30.0, timeout_seconds=120.0, memory_mb=1024):
    from ipfs_datasets_py.logic.software_contracts.codebase_inventory_resume import (
        CodebaseScanResumeRoot, CodebaseScanResumeCompletion,
    )
    from ipfs_datasets_py.logic.software_contracts.codebase_inventory_receiving import (
        _paired_current_codebase_scan_completion,
    )
    from ipfs_datasets_py.logic.software_contracts.codebase_resources import acquire_codebase_resources
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseCancelledError, LeaseTimeoutError

    if type(root) is not CodebaseScanResumeRoot or type(completion) is not CodebaseScanResumeCompletion:
        raise ValueError("exact native completed resumable scan required")
    if (type(timeout_seconds) not in {int, float} or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 600
            or type(admission_timeout_seconds) not in {int, float}
            or not math.isfinite(admission_timeout_seconds) or admission_timeout_seconds < 0
            or type(memory_mb) is not int or not 1024 <= memory_mb <= 4096):
        raise ValueError("bounded inventory admission resources required")
    if cancel_event is not None and not callable(getattr(cancel_event, "is_set", None)):
        raise ValueError("typed inventory cancellation signal required")
    deadline = time.monotonic() + timeout_seconds
    if evidence_record is not None:
        from ipfs_datasets_py.logic.software_contracts.codebase_inventory_evidence import (
            CodebaseInventoryEvidenceRecord, validate_current_inventory_evidence,
        )
        if type(evidence_record) is not CodebaseInventoryEvidenceRecord or verification_catalog is None:
            raise ValueError("exact optional evidence record and native query owner required")
    context = _full_context(root, completion, evidence_record)
    declaration = inventory_declaration(context)
    captured = _wire(context)
    declaration_wire = _wire(declaration)
    root_wire, completion_wire = _wire(root.to_dict()), _wire(completion.to_dict())

    def remaining(signal=None):
        if ((cancel_event is not None and cancel_event.is_set())
                or (signal is not None and signal.is_set())):
            raise LeaseCancelledError("inventory admission cancelled")
        left = deadline - time.monotonic()
        if left <= 0:
            raise LeaseTimeoutError("inventory admission deadline exceeded")
        return left

    def unchanged():
        if (_wire(root.to_dict()) != root_wire or _wire(completion.to_dict()) != completion_wire
                or _wire(context) != captured
                or _wire(declaration) != declaration_wire
                or _wire(completion.advisory_refs(root)) != _wire(context["scan"])
                or (evidence_record is not None and _wire(evidence_record.advisory_refs()) != _wire(context["evidence"]))):
            raise ValueError("inventory admission inputs changed during owner callbacks")

    with acquire_codebase_resources(scheduler=scheduler, parent_lease=parent_lease, cancel_event=cancel_event,
            timeout_seconds=min(admission_timeout_seconds, remaining()), memory_mb=memory_mb) as lease:
        signal = lease.combined_cancellation_signal(cancel_event)

        def evidence_current():
            if evidence_record is not None:
                if validate_current_inventory_evidence(evidence_record, index, repository, registry=registry,
                        verification_catalog=verification_catalog, parent_lease=lease, cancel_event=signal,
                        admission_timeout_seconds=min(admission_timeout_seconds, remaining(signal)),
                        timeout_seconds=remaining(signal), memory_mb=memory_mb) is not evidence_record:
                    raise ValueError("receiving validation selected different conditional evidence")
            unchanged()
            remaining(signal)

        with _paired_current_codebase_scan_completion(completion, index, repository,
                root=root, registry=registry, parent_lease=lease, cancel_event=signal,
                admission_timeout_seconds=min(admission_timeout_seconds, remaining(signal)),
                timeout_seconds=remaining(signal), memory_mb=memory_mb) as close_scan:
            evidence_current()

            def current():
                unchanged()
                # Optional evidence receiving can invoke owner callbacks. It
                # must finish before the completed scan's final native close.
                evidence_current()
                if close_scan() is not completion:
                    raise ValueError("receiving validation selected different scan completion")
                unchanged()
                remaining(signal)

            yield declaration, current
    remaining()


def _bound_context(manifest, context):
    if (manifest.get("payload", {}).get("schema") != local.INVENTORY_MANIFEST_SCHEMA
            or _wire(manifest["payload"].get("codebase_inventory_context")) != _wire(context)):
        raise ValueError("signed manifest differs from the complete current inventory context")


def author_current_inventory_manifest(root, completion, index, repository, *, registry,
        profile_dir, lifecycle_dir, task_specs, planning_roots, planning_inputs=None, **resources):
    """Sign independent full-task declarations with a fresh complete scan root."""
    with _current_scope(root=root, completion=completion, index=index, repository=repository,
                        registry=registry, **resources) as (context, current):
        manifest = local.author_local_benchmark_manifest(repository=Path(repository),
            profile_dir=Path(profile_dir), lifecycle_dir=Path(lifecycle_dir), task_specs=task_specs,
            planning_roots=planning_roots, planning_inputs=planning_inputs,
            codebase_inventory_context=context)
        _bound_context(manifest, context)
        local._manifest(manifest, initial=True)
        current()
    return manifest


def admit_current_inventory_plan(*, root, completion, index, repository, registry, manifest,
                                graph, **resources):
    """The original full native graph is admitted with every check pending."""
    manifest = local._plain(manifest)
    if type(graph) is not local.PromptGoalGraph:
        raise ValueError("exact native independently authored full task graph required")
    graph = local.PromptGoalGraph.from_dict(graph.to_dict())
    with _current_scope(root=root, completion=completion, index=index, repository=repository,
                        registry=registry, **resources) as (context, current):
        _bound_context(manifest, context)
        admission = local.admit_local_benchmark_plan(graph=graph, manifest=manifest)
        local.verify_local_benchmark_admission(admission)
        current()
    return admission


@contextmanager
def _current_inventory_admission_operation(*, root, completion, index, repository, registry,
                                         admission, **resources):
    """Private one-operation entry/close; never an exported freshness token."""
    admission = local._plain(admission)
    captured = _wire(admission)
    with _current_scope(root=root, completion=completion, index=index, repository=repository,
                        registry=registry, **resources) as (context, current):
        _bound_context(admission.get("manifest", {}), context)
        verified = local.verify_local_benchmark_admission(admission)
        receipt = {"schema": "supervisor-current-inventory-admission-verification@1",
            "admission_cid": content_identity(admission), "manifest_cid": verified["receipt"]["manifest_cid"],
            "graph_cid": verified["graph"].content_id, "plan_id": verified["receipt"]["plan_id"],
            "codebase_inventory_context_cid": context["full_context_cid"],
            "root_cid": context["scan"]["root_cid"], "completion_cid": context["scan"]["completion_cid"],
            "head": context["scan"]["head"],
            "administrator_task_cids": sorted(task.task_cid for task in verified["graph"].tasks),
            "pending_cid": verified["receipt"]["pending_cid"], "current_facts": [], "removed_task_cids": [],
            "runtime_requirements_preserved": True, "observed_current": True,
            "native_persistence_verified_here": False, "authority": dict(context["authority"])}
        receipt_wire = _wire(receipt)

        def close_current():
            if _wire(admission) != captured or _wire(receipt) != receipt_wire:
                raise ValueError("inventory verification inputs changed during owner callbacks")
            current()
            if _wire(admission) != captured or _wire(receipt) != receipt_wire:
                raise ValueError("inventory verification inputs changed during owner callbacks")
            return receipt

        yield receipt, close_current


def verify_current_inventory_admission(*, root, completion, index, repository, registry,
                                      admission, **resources):
    """Owner receiving gate; not a worker launch or native persistence receipt."""
    with _current_inventory_admission_operation(root=root, completion=completion, index=index,
            repository=repository, registry=registry, admission=admission, **resources) as (receipt, close):
        close()
    return receipt


def materialize_current_inventory_plan(*, root, completion, index, repository, registry,
                                      admission, intent, **resources):
    """Install every original pending task in one native rollback transaction."""
    if type(intent) is not IntentRepository or intent.uses_bound_connection:
        raise ValueError("independently owned native intent transaction required")
    admission = local._plain(admission)
    with intent._connection(write=True) as connection:
        with _current_scope(root=root, completion=completion, index=index, repository=repository,
                            registry=registry, **resources) as (context, current):
            _bound_context(admission.get("manifest", {}), context)
            verified = local.verify_local_benchmark_admission(admission)
            expected = sorted(task.task_cid for task in verified["graph"].tasks)
            with IntentRepository(bound_connection=connection, owner_id=intent.owner_id,
                                  session_id=intent.session_id) as bound:
                result = local._materialize_local_benchmark_plan(admission=admission, intent=bound)
                if sorted(result["task_cids"]) != expected:
                    raise ValueError("native inventory install omitted an administrator task")
                local.verify_local_benchmark_admission(admission)
                # The current receiving check happens before the enclosing
                # native transaction can commit, after signing/install hooks.
                current()
    return {**result, "schema": "supervisor-current-inventory-native-materialization@1",
        "codebase_inventory_context_cid": context["full_context_cid"],
        "root_cid": context["scan"]["root_cid"], "completion_cid": context["scan"]["completion_cid"],
        "administrator_task_population_preserved": True, "current_facts": [], "removed_task_cids": [],
        "runtime_requirements_preserved": True, "observed_current": True,
        "authority": dict(context["authority"])}


def prepare_current_inventory_worker_context(*, root, completion, index, repository, registry,
        admission, task_cid, source_path, expected_source_sha256, **resources):
    """Owner emits the existing worker artifact between native receiving fences."""
    from .router_public_instruction import prepare_public_instruction_context

    admission = local._plain(admission)
    with _current_scope(root=root, completion=completion, index=index, repository=repository,
                        registry=registry, **resources) as (context, current):
        _bound_context(admission.get("manifest", {}), context)
        selected = prepare_public_instruction_context(repository=Path(repository), admission=admission,
            task_cid=task_cid, source_path=source_path, expected_source_sha256=expected_source_sha256,
            codebase_inventory_context=_full_context(root, completion, resources.get("evidence_record")))
        current()
    return selected


__all__ = ["author_current_inventory_manifest", "admit_current_inventory_plan",
    "verify_current_inventory_admission", "materialize_current_inventory_plan",
    "prepare_current_inventory_worker_context"]
