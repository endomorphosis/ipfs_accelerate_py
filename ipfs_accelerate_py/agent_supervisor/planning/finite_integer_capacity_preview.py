"""Fresh finite planning with source-model checks and held native capacity.

The old finite preview remains unchanged. This additive application profile
freezes actual capacity and optional registered feature context before a second
real service evaluation. Neither model reconstruction, a mathematical model
theorem nor a feasible native schedule grants execution permission.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import stat
import time
from typing import Any

from ipfs_datasets_py.logic.software_contracts.content import canonical_dag_json_bytes, cid_for_structured
from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import IntegerOffsetContract
from ipfs_datasets_py.logic.intent_ir.schema import IntentIRDocument

from ..prompt.plan_create_service import PlanCreateMode, freeze_plan_create_input_snapshot
from . import finite_integer_codebase as matcher
from .finite_integer_plan_preview import (
    FiniteIntegerOperationCatalog, FiniteIntegerPlanPreviewError, _materials,
    finite_integer_intent_cid, finite_integer_prompt_cid, preview_finite_integer_plan,
)
from .plan_revision_contracts import PlanAuthorityRoots, PlanCreateRequest
from .repository_plan_preview import RepositoryPlanPreviewOwner
from .structural_codebase_context import structural_codebase_context

SCHEMA = "finite-integer-capacity-bound-plan-preview@2"
_FALSE = {name: False for name in (
    "source_semantics_verified", "runtime_behavior_verified", "behavior_authority",
    "proof_authority", "production_admitted", "execution_authority", "completion_authority",
    "mutation_authority", "omission_authority", "worker_launched", "convergence_proved",
)}


def _output_path(value, repository):
    output = Path(value)
    if (not output.is_absolute() or output.resolve() != output or output.exists()
            or not output.parent.is_dir() or output.is_relative_to(repository)):
        raise FiniteIntegerPlanPreviewError("fresh canonical output outside the tested repository required")
    return output


def _detached_artifact_closure(*, output, match, model, feature_context, index, policy, checkpoint,
        source_custody, feature_versions, model_registry):
    """Read owned byte identities after all live observation callbacks finish.

    Expected bytes come from the freshly produced immutable records, never
    from a new snapshot of mutable files. This does not attest process origin
    or transitive runtime dependencies.
    """
    rows = {}

    def add(path, size, sha):
        path = Path(path)
        expected = (size, sha)
        if path in rows and rows[path] != expected:
            raise FiniteIntegerPlanPreviewError("conflicting owned artifact identities")
        rows[path] = expected

    def literal(path, raw):
        add(path, len(raw), hashlib.sha256(raw).hexdigest())

    # Read immutable native version records before detached physical checks.
    # Retained lineage identities also close the original registry blob paths.
    for version in feature_versions:
        checkpoint()
        if matcher._json(model_registry.get_version(version["version_id"])) != version:
            raise FiniteIntegerPlanPreviewError("native feature ancestry version changed")
        descriptor = version["artifact"]
        add(model_registry.artifact_path(descriptor), descriptor["bytes"], descriptor["sha256"])
    source_custody.require_current(checkpoint)
    for record in (match["observation"], model.to_dict()):
        directory = Path(record["output"])
        for descriptor in record["artifacts"].values():
            add(descriptor["path"], descriptor["size_bytes"], descriptor["sha256"])
            # The operational model stores every artifact in the byte CAS;
            # finite observations keep their source and structured result.
            if record is not match["observation"]:
                add(index.artifacts.path_for(descriptor["cid"], source=True),
                    descriptor["size_bytes"], descriptor["sha256"])
        raw = canonical_dag_json_bytes(record)
        literal(directory / "result.json", raw)
        literal(index.artifacts.path_for(cid_for_structured(record)), raw)
    source = match["observation"]["artifacts"]["source"]
    add(index.artifacts.path_for(source["cid"], source=True), source["size_bytes"], source["sha256"])
    directories = [output, output / "finite-observation", output / "operational-model"]
    if feature_context is not None:
        directories.append(feature_context.output)
        literal(feature_context.output / "context.json", feature_context._material_bytes)
        for descriptor in feature_context.material_binding["artifacts"].values():
            add(feature_context.output / descriptor["relative_path"],
                descriptor["size_bytes"], descriptor["sha256"])
    for name in ("python", "lean"):
        descriptor = policy[name]
        add(descriptor["path"], descriptor["size_bytes"], descriptor["sha256"])
    for directory in directories:
        checkpoint()
        if (directory.is_symlink() or not directory.is_dir()
                or directory.resolve(strict=True) != directory):
            raise FiniteIntegerPlanPreviewError("owned output directory custody changed")
    for path, (expected_size, expected_sha) in rows.items():
        checkpoint()
        if (not path.is_absolute() or path.resolve(strict=True) != path or path.is_symlink()
                or not path.is_file() or type(expected_size) is not int
                or not 0 <= expected_size <= 64 * 1024**2):
            raise FiniteIntegerPlanPreviewError("owned artifact path or byte bound changed")
        digest, size = hashlib.sha256(), 0
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
                raise FiniteIntegerPlanPreviewError("owned artifact descriptor changed")
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                checkpoint()
                size += len(block)
                if size > expected_size:
                    raise FiniteIntegerPlanPreviewError("owned artifact size changed")
                digest.update(block)
            after = os.fstat(stream.fileno())
            current = path.stat(follow_symlinks=False)
        identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        if (size != expected_size or digest.hexdigest() != expected_sha
                or identity(before) != identity(after) or identity(after) != identity(current)
                or path.resolve(strict=True) != path or path.is_symlink()):
            raise FiniteIntegerPlanPreviewError("owned artifact bytes changed after live validation")
    checkpoint()


def preview_capacity_bound_finite_integer_plan(*, owner: RepositoryPlanPreviewOwner,
        request: PlanCreateRequest, intent_document: IntentIRDocument, source_text: str,
        operation_catalog: FiniteIntegerOperationCatalog, output, tool_policy: dict[str, Any],
        policy_observer: Callable[[PlanCreateRequest], PlanAuthorityRoots],
        feature_context=None, model_registry=None) -> dict[str, Any]:
    """Fresh observations, an operational-model theorem and real resource checks.

    Feature fitting is a separate explicit preparation operation. Selected
    frozen features are verified natively and remain advisory. No caller match,
    capacity snapshot, model score, theorem receipt or service factory is accepted.
    Held resources are released before this proposal is returned.
    """
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
        LeaseCancelledError, LeaseTimeoutError,
    )
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_model_lean import (
        prove_current_integer_offset_model, validate_current_integer_offset_model,
    )
    from .finite_integer_capacity import (
        FINITE_CAPACITY_PROFILE, CapacityBoundFiniteIntegerPlanCreateService,
        reserve_finite_integer_capacity,
    )
    from .codebase_feature_context import FrozenCodebaseFeatureContext, verify_current_context
    from .finite_integer_source_custody import capture_source_custody

    if (type(owner) is not RepositoryPlanPreviewOwner or type(request) is not PlanCreateRequest
            or type(intent_document) is not IntentIRDocument
            or type(operation_catalog) is not FiniteIntegerOperationCatalog
            or not callable(policy_observer) or owner.memory_mb < 1024
            or request.budget.max_model_calls != 0 or request.budget.max_tasks < 2
            or request.budget.max_goals < 2 or request.required_analysis_operations
            or request.optional_analysis_operations or request.required_logic_families
            or request.optional_logic_families):
        raise FiniteIntegerPlanPreviewError("exact model-off planning request and native owner required")
    query = matcher.prepare_finite_integer_query(intent_document=intent_document, source_text=source_text)
    if not query["supported"] or set(query["requirement_ids"]) != {
            matcher.TYPE_STATEMENT_ID, matcher.OFFSET_STATEMENT_ID}:
        raise FiniteIntegerPlanPreviewError("complete supported finite instruction required")
    contract = IntegerOffsetContract.from_dict(query["contract"])
    if (request.repository_root != str(owner.repository)
            or request.repository_id != owner.expected_head.repository_id
            or request.roots.repository_root_cid != owner.expected_head.snapshot_cid
            or request.roots.dirty_worktree_root != owner.expected_head.snapshot_cid
            or request.prompt_source_cid != finite_integer_prompt_cid(source_text)
            or request.roots.intent_ir_root != finite_integer_intent_cid(intent_document)
            or request.roots.capability_catalog_root != operation_catalog.cid
            or request.scope_paths != (contract.path,)
            or any((row.path, row.function_name, row.parameter) !=
                   (contract.path, contract.function_name, contract.parameter)
                   for row in operation_catalog.operations)):
        raise FiniteIntegerPlanPreviewError("exact prompt, IR, catalog, scope and source roots required")
    if (feature_context is None) != (model_registry is None):
        raise FiniteIntegerPlanPreviewError("feature context and native registry must be selected together")
    if feature_context is not None and type(feature_context) is not FrozenCodebaseFeatureContext:
        raise FiniteIntegerPlanPreviewError("exact native-prepared feature context required")
    output = _output_path(output, owner.repository)
    policy = matcher._json(tool_policy)
    deadline = time.monotonic() + min(owner.timeout_seconds, request.budget.max_latency_ms / 1000)

    def remaining():
        if owner.cancel_event is not None and owner.cancel_event.is_set():
            raise LeaseCancelledError("capacity-bound finite planning cancelled")
        left = deadline - time.monotonic()
        if left <= 0:
            raise LeaseTimeoutError("capacity-bound finite planning deadline exceeded")
        return left

    def current_owner():
        return replace(owner, timeout_seconds=remaining(),
            scheduler=None if owner.parent_lease is not None else owner.scheduler)

    def controls():
        return {"expected_head": owner.expected_head, "repository_id": owner.expected_head.repository_id,
            "scheduler": None if owner.parent_lease is not None else owner.scheduler, "parent_lease": owner.parent_lease,
            "cancel_event": owner.cancel_event, "timeout_seconds": remaining(), "memory_mb": owner.memory_mb}

    def check_feature():
        if feature_context is not None:
            verify_current_context(owner=current_owner(), registry=model_registry, context=feature_context)
            remaining()

    source_custody = capture_source_custody(owner, checkpoint=remaining)
    check_feature()
    feature_versions = (tuple(matcher._json(row["version"]) for row in
        feature_context.retained_artifacts.get("lineage", ())) if feature_context is not None else ())
    output.mkdir(mode=0o700)
    initial = preview_finite_integer_plan(owner=current_owner(), request=request,
        intent_document=intent_document, source_text=source_text, operation_catalog=operation_catalog,
        output=output / "finite-observation", tool_policy=policy, policy_observer=policy_observer)
    match = matcher._json(initial["match"])
    model = prove_current_integer_offset_model(owner.index, owner.repository,
        expected_head=owner.expected_head, contract=contract, tool_policy=policy,
        output=output / "operational-model", scheduler=None if owner.parent_lease is not None else owner.scheduler,
        parent_lease=owner.parent_lease, cancel_event=owner.cancel_event,
        timeout_seconds=remaining(), memory_mb=owner.memory_mb)

    def check_artifacts(*, verify_feature=False):
        remaining()
        matcher._check_observation(observation=match["observation"], index=owner.index,
            head=owner.expected_head, contract=contract, inputs=query["domain_inputs"],
            tool_policy=policy, output=output / "finite-observation")
        validate_current_integer_offset_model(model, owner.index, owner.repository,
            expected_head=owner.expected_head, contract=contract, tool_policy=policy,
            scheduler=None if owner.parent_lease is not None else owner.scheduler, parent_lease=owner.parent_lease,
            cancel_event=owner.cancel_event, timeout_seconds=remaining(), memory_mb=owner.memory_mb)
        # Advisory features have immutable source/context/version/checkpoint
        # inputs. Their complete byte closure is checked at every fence. Fresh
        # numerical replay at entry and after all callbacks is sufficient for
        # this profile; repeated inference at each root observation otherwise
        # consumes a correct native forecast's validity window.
        if verify_feature:
            check_feature()
        _detached_artifact_closure(output=output, match=match, model=model,
            feature_context=feature_context, index=owner.index, policy=policy, checkpoint=remaining,
            source_custody=source_custody, feature_versions=feature_versions, model_registry=model_registry)
        remaining()

    with structural_codebase_context(owner.index, owner.repository, **controls()) as context:
        if (context.to_dict() != match["structural_context"]
                or request.roots.program_root != context.semantic_state_cid):
            raise FiniteIntegerPlanPreviewError("native source changed after fresh observation")
        materials, bindings = _materials(match, operation_catalog, context)
    check_artifacts()
    materials.extra["operational_model_projection"] = model.to_dict()
    materials.extra["source_custody"] = source_custody.material_binding
    materials.extra["feature_context"] = (feature_context.material_binding if feature_context is not None
        else {"schema": "finite-plan-feature-selection@1", "mode": "model_off",
              "head": owner.expected_head.to_dict(), "behavior_authority": False, "proof_authority": False})
    feature_policy = {"schema": "finite-plan-immutable-advisory-feature-validation@1",
        "mode": feature_context.mode if feature_context is not None else "model_off",
        "numerical_verification_points": (["entry", "after_reservation_release"]
            if feature_context is not None and feature_context.mode != "model_off" else []),
        "between_verifications": "current_source_CAS_AST_and_all_frozen_context_lineage_bytes_at_every_fence",
        "final_order": "native_verification_then_detached_byte_closure",
        "behavior_authority": False, "proof_authority": False, "execution_authority": False}
    materials.extra["feature_validation_policy"] = feature_policy

    # Capacity is separate from source/proof sibling reservations. It respects
    # the caller's original native parent, and never reparents proof work into
    # a single-slot planned-worker reservation.
    with reserve_finite_integer_capacity(owner=replace(owner, timeout_seconds=remaining()),
            request=request, materials=materials) as capacity:
        materials.extra["finite_capacity_profile"] = FINITE_CAPACITY_PROFILE
        materials.extra["finite_capacity_binding"] = capacity.material_binding
        snapshot = freeze_plan_create_input_snapshot(request, materials=materials)

        def observe_roots(value):
            if value != request:
                raise FiniteIntegerPlanPreviewError("capacity service changed the exact request")
            with structural_codebase_context(owner.index, owner.repository, **controls()) as current:
                if current != context:
                    raise FiniteIntegerPlanPreviewError("native source changed during capacity preview")
                check_artifacts()
                roots = policy_observer(request)
                if type(roots) is not PlanAuthorityRoots:
                    raise FiniteIntegerPlanPreviewError("complete independently observed policy roots required")
                roots.require_current(request.roots)
                check_artifacts()
            check_artifacts()
            remaining()
            return roots

        service = CapacityBoundFiniteIntegerPlanCreateService(root_observer=observe_roots,
            operation_bindings=bindings, capacity=capacity)
        receipt = service.preview_create(request, mode=PlanCreateMode.DETERMINISTIC, materials=materials)
        if receipt.input_snapshot_cid != snapshot.snapshot_cid:
            raise FiniteIntegerPlanPreviewError("capacity service consumed a different frozen input")
        stages = {row.stage.value: row for row in receipt.stage_results}
        if any(not stages[name].passed for name in (
                "scan", "query", "evidence", "obligation", "candidate", "critique", "parallel_plan")):
            raise FiniteIntegerPlanPreviewError("capacity-bound native stage rejected: " + str(receipt.to_dict()))
        if receipt.admitted or not receipt.read_only or receipt.wrote_effects:
            raise FiniteIntegerPlanPreviewError("capacity feasibility cannot grant execution admission")
        diagnostics = service.diagnostics
        check_artifacts()
        with structural_codebase_context(owner.index, owner.repository, **controls()) as final_context:
            if final_context != context:
                raise FiniteIntegerPlanPreviewError("source drifted before returning capacity proposal")
            check_artifacts()
        check_artifacts()

        def record(value):
            return value.to_dict() if callable(getattr(value, "to_dict", None)) else value

        result = {"schema": SCHEMA, "profile": FINITE_CAPACITY_PROFILE,
            "original_finite_preview": initial, "preview": receipt.to_dict(),
            "input_snapshot": snapshot.to_dict(), "match": match,
            "operation_catalog": operation_catalog.to_dict(), "operation_catalog_cid": operation_catalog.cid,
            "operational_model": model.to_dict(), "feature_context": materials.extra["feature_context"],
            "feature_validation_policy": feature_policy,
            "source_custody": source_custody.material_binding,
            **{key: record(value) for key, value in diagnostics.items()},
            "planner_status": service.planner_status,
            "selected_task_ids": [row["task_id"] for row in diagnostics["candidate_plan"]["tasks"]],
            "declared_task_requirement_ids": initial["declared_task_requirement_ids"],
            "current_facts_count": len(match["current_facts"]), "scope": "explicit_finite_domain_only",
            "capacity_binding": capacity.material_binding, "reservation_released_on_return": True,
            "capacity_scope": "feasibility_during_live_reservation_only",
            "training_steps_during_preview": 0, "planning_model_calls": 0, **_FALSE}
        result["result_cid"] = cid_for_structured(result)
        result = matcher._json(result)
        remaining()
    # Return only after the native reservation's context has completed cleanup.
    check_artifacts(verify_feature=True)
    remaining()
    return result


__all__ = ["SCHEMA", "preview_capacity_bound_finite_integer_plan"]
