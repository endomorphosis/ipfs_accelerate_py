"""Opt-in native structural preview; existing signed tasks retain authority."""
from dataclasses import replace
import hashlib
import math
from pathlib import Path
import time

from ipfs_accelerate_py.agent_supervisor.planning.repository_plan_preview import (
    RepositoryPlanPreviewOwner, preview_repository_plan,
)
from ipfs_accelerate_py.agent_supervisor.planning.structural_codebase_context import structural_codebase_context


def _controls(owner, remaining):
    return {"repository_id": owner.expected_head.repository_id,
        "expected_head": owner.expected_head, "scheduler": owner.scheduler,
        "parent_lease": owner.parent_lease, "cancel_event": owner.cancel_event,
        "timeout_seconds": remaining(), "memory_mb": owner.memory_mb}


def _source_join(index, context, prepared):
    """Compare every signed source to captured CAS bytes without execution."""
    from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
    from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import cid_for_dag_json
    manifest = index.load(context.head.manifest_cid)
    entries = {entry.path: entry for entry in manifest.snapshot.entries}
    sources = prepared["manifest"]["payload"]["sources"]
    for path, expected in sources.items():
        entry = entries.get(path)
        if entry is None or entry.is_opaque or entry.source_cid is None:
            raise ValueError("signed source is absent or opaque in repository preview: " + path)
        raw = index.artifacts.get_bytes(entry.source_cid)
        if hashlib.sha256(raw).hexdigest() != expected["sha256"]:
            raise ValueError("signed source differs from native captured bytes: " + path)
    return {"schema": "terminal-administrative-source-correspondence@1",
        "administrative_source_root": content_identity({"schema": "supervisor-local-source-tree@1", "sources": sources}),
        "native_snapshot_cid": context.head.snapshot_cid,
        "native_manifest_cid": context.head.manifest_cid,
        "signed_manifest_cid": cid_for_dag_json(prepared["manifest"]),
        "signed_sources_cid": cid_for_dag_json(sources), "signed_source_count": len(sources),
        "scope": "captured_bytes_and_separately_verified_signed_baseline",
        "behavior_authority": False, "execution_authority": False}


def prepare_repository_preview(*, state, index, repository_id, operation_id,
        expected_head=None, scheduler=None, parent_lease=None, cancel_event=None,
        timeout_seconds=30, memory_mb=1024, limits=None):
    """Explicitly capture a signed symbolic preparation through its native owner.

    Caller-owned native catalog/scheduler remain ephemeral. The default
    256-entry/256KiB scan requires a 1GiB reservation including AST work.
    Preview and cache reads never invoke this mutating preparation operation.
    """
    from . import terminal_indexed_preparation as prep
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import CodebaseScanLimits
    if (type(timeout_seconds) not in {int, float} or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 90 or type(memory_mb) is not int
            or not 32 <= memory_mb <= 4096):
        raise ValueError("bounded repository preparation time and memory required")
    state = Path(state).resolve(strict=True)
    prepared = prep._load_prepared(state)
    if prep._planning_strategy(prepared.get("intent_requirement_contract")) != "intent_symbolic":
        raise ValueError("repository preview requires explicit symbolic operation preparation")
    started = time.monotonic()
    receipt = index.prepare_current(Path(prepared["repository"]), repository_id=repository_id,
        operation_id=operation_id, expected_head=expected_head,
        limits=limits if limits is not None else CodebaseScanLimits(max_entries=256, max_file_bytes=256 * 1024),
        scheduler=scheduler, parent_lease=parent_lease, cancel_event=cancel_event,
        timeout_seconds=timeout_seconds, memory_mb=memory_mb)
    if prep._load_prepared(state) != prepared:
        raise ValueError("signed preparation changed during repository capture")
    owner = RepositoryPlanPreviewOwner(index=index, repository=Path(prepared["repository"]),
        expected_head=receipt.head, scheduler=scheduler, parent_lease=parent_lease,
        cancel_event=cancel_event, timeout_seconds=timeout_seconds, memory_mb=memory_mb)
    remaining = lambda: timeout_seconds - (time.monotonic() - started)
    with structural_codebase_context(index, owner.repository, **_controls(owner, remaining)) as context:
        _source_join(index, context, prepared)
    return owner


def require_current_repository_preview(*, state, prepared, owner, timeout_seconds):
    """Reobserve the selected owner/source immediately before old admission."""
    from . import terminal_indexed_preparation as prep
    started = time.monotonic()
    remaining = lambda: min(timeout_seconds, owner.timeout_seconds) - (time.monotonic() - started)
    with structural_codebase_context(owner.index, owner.repository, **_controls(owner, remaining)) as context:
        if prep._load_prepared(state) != prepared:
            raise ValueError("signed preparation changed before administrative admission")
        _source_join(owner.index, context, prepared)


def materialize_repository_preview_administration(*, state, prepared, owner, admission, remaining):
    """Fence the existing task transaction after every administrative replay.

    The native transaction rolls back if the selected repository changes or
    cancellation occurs while legacy verification/materialization is running.
    Signed receipt and task semantics remain those of the administrative gate.
    """
    from contextlib import contextmanager
    from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository

    def fence():
        require_current_repository_preview(state=state, prepared=prepared,
            owner=owner, timeout_seconds=remaining())

    class GuardedIntentRepository(IntentRepository):
        @contextmanager
        def _connection(self, *, write=False):
            with super()._connection(write=write) as connection:
                if write:
                    fence()
                yield connection
                if write:
                    # Raising here reaches the native owner's rollback before
                    # its COMMIT; bound inner owners share this transaction.
                    fence()

    with GuardedIntentRepository(Path(state) / "intent.duckdb",
            lock_timeout_seconds=min(30.0, remaining())) as intent:
        return local.materialize_local_benchmark_plan(admission=admission, intent=intent)


def preview_prepared_repository_plan(*, state, prepared, owner, timeout_seconds):
    """Run real create-plan stages with complete source-bound proposal inputs."""
    from . import terminal_indexed_preparation as prep
    from ipfs_accelerate_py.agent_supervisor.planning.intent_requirement_adapter import build_intent_planning_materials
    from ipfs_accelerate_py.agent_supervisor.prompt.plan_create_service import (
        PlanCreateMaterials, plan_create_request_from_workflow,
    )
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import PromptWorkflowRequest
    if (type(owner) is not RepositoryPlanPreviewOwner
            or str(owner.repository) != prepared["repository"]):
        raise ValueError("explicit native preview owner must bind the prepared repository")
    started = time.monotonic()
    duration = min(timeout_seconds, owner.timeout_seconds)
    def remaining():
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseTimeoutError
        left = duration - (time.monotonic() - started)
        if left <= 0:
            raise LeaseTimeoutError("terminal repository preview deadline exceeded")
        return left
    selected = build_intent_planning_materials(prepared["intent_requirement_contract"], manifest=prepared["manifest"])
    if selected.current_facts or prep._load_prepared(state) != prepared:
        raise ValueError("complete unchanged signed administrative materials required")
    with structural_codebase_context(owner.index, owner.repository, **_controls(owner, remaining)) as context:
        correspondence = _source_join(owner.index, context, prepared)
        if correspondence["administrative_source_root"] != selected.current_root_id:
            raise ValueError("administrative source correspondence differs from selected materials")
        base = plan_create_request_from_workflow(PromptWorkflowRequest.from_dict(prepared["request"]),
            repository_id=owner.expected_head.repository_id,
            scope_paths=tuple(sorted({path for task in prepared["manifest"]["payload"]["tasks"] for path in task["scope_paths"]})))
        request = replace(base, roots=replace(base.roots,
            repository_root_cid=owner.expected_head.snapshot_cid,
            dirty_worktree_root=owner.expected_head.snapshot_cid,
            program_root=context.semantic_state_cid), budget=replace(base.budget, max_model_calls=0))
        bound = PlanCreateMaterials(
            intent=replace(selected.intent, current_root_id=owner.expected_head.snapshot_cid),
            producers=selected.producers, task_candidates=selected.task_candidates, predicates=selected.predicates,
            frozen_goal=replace(selected.frozen_goal, repository_tree_id=owner.expected_head.snapshot_cid),
            candidate_context=dict(selected.candidate_context),
            extra={"operation_contract": dict(selected.operation_contract), "source_root_correspondence": correspondence})
        def observe_policy(value):
            if prep._load_prepared(state) != prepared or value != request:
                raise ValueError("signed preparation or policy changed during repository preview")
            return request.roots
        result = preview_repository_plan(owner=replace(owner, timeout_seconds=remaining()),
            request=request, materials=bound, policy_observer=observe_policy)
        remaining()
    result["source_root_correspondence"] = correspondence
    result["selection_authority"] = "existing_administrative_manifest"
    return result
