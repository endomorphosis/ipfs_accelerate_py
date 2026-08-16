"""Durable, exact-resume prompt-to-run saga.

The service records a non-terminal run before each external effect and only
publishes ``RUNNING`` after materialization and lifecycle birth/adoption have
both supplied durable receipts.  Retrying an invocation reconstructs that
record and continues from its cursor instead of replaying effects.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any

from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import cid_for_dag_json

from .contracts import (
    ContinuationAction, InvocationStatus, RunHandle, RunHealth, RunState,
    SupervisorInvocationResult,
)
from .run_registry import RunExistsError
from .runtime_factory import CompleteLaunchPlan, RuntimeEffectError, StandardSupervisorRuntimeFactory


class PromptToRunError(RuntimeError):
    """Base typed failure for prompt-to-run orchestration."""


class PromptToRunUnavailableError(PromptToRunError):
    """A required effect implementation is unavailable."""


@dataclass(frozen=True)
class PromptToRunSaga:
    """One durable root-bound invocation and its exact continuation cursor."""

    complete_plan: CompleteLaunchPlan
    run_handle: RunHandle


def _cid(kind: str, payload: dict[str, Any]) -> str:
    return cid_for_dag_json({"kind": kind, **payload})


class SupervisorIntentService:
    """Starts or resumes a materialized task source and fenced lifecycle."""

    def __init__(self, *, factory: StandardSupervisorRuntimeFactory) -> None:
        self.factory = factory

    def _initial_handle(self, plan: CompleteLaunchPlan, *, adopt: bool = False) -> RunHandle:
        launch = plan.launch_plan
        now = int(time.time() * 1000)
        run_id = _cid("prompt-to-run", {"invocation": launch.invocation_cid, "plan": launch.launch_plan_cid})
        return RunHandle(
            run_id=run_id, run_revision=1,
            target_resolution_receipt_cid=launch.target_resolution_receipt_cid,
            invocation_cid=launch.invocation_cid,
            prompt_cid=_cid("prompt-ref", {"invocation": launch.invocation_cid}),
            workflow_cid="", scan_cid="", plan_cid=launch.launch_plan_cid,
            materialization_cid="", task_source_cid=plan.task_source_cid,
            task_source_revision_cid=plan.task_source_revision_cid,
            lifecycle_profile_cid=launch.lifecycle_profile_cid, process_cid="",
            objective_cid=plan.objective_cid, objective_revision_cid=plan.objective_revision_cid,
            lease_id="", fencing_generation=0,
            state=RunState.ADAPTING if adopt else RunState.MATERIALIZING,
            health=RunHealth.UNKNOWN, state_revision_cid="", health_revision_cid="",
            event_cursor="", continuation_action=(
                ContinuationAction.ADOPT if adopt else ContinuationAction.MATERIALIZE
            ),
            pending_approval_cid="", ambiguity_cid="", created_at_ms=now, updated_at_ms=now,
        )

    def _advance(self, current: RunHandle, **changes: Any) -> RunHandle:
        now = int(time.time() * 1000)
        next_handle = replace(current, run_revision=current.run_revision + 1, updated_at_ms=max(now, current.updated_at_ms), **changes)
        self.factory.registry.cas_update(next_handle, expected_revision=current.run_revision, expected_handle_cid=current.content_id, expected_semantic_id=current.semantic_id)
        return next_handle

    def _result(self, plan: CompleteLaunchPlan, handle: RunHandle, receipts: tuple[str, ...], *, adopted: bool = False) -> SupervisorInvocationResult:
        return SupervisorInvocationResult(
            invocation_cid=plan.launch_plan.invocation_cid,
            status=InvocationStatus.ADOPTED if adopted else InvocationStatus.RUNNING,
            target_resolution_receipt_cid=plan.launch_plan.target_resolution_receipt_cid,
            launch_plan_cid=plan.launch_plan_cid, run_handle=handle,
            reason_codes=("resumed" if receipts == () else "started",), questions=(),
            continuation_action=handle.continuation_action, effect_receipt_cids=receipts,
            event_cursor=handle.event_cursor, error_code="",
        )

    def run(self, complete_plan: CompleteLaunchPlan, *, adopt: bool = False) -> SupervisorInvocationResult:
        """Create or reconstruct a run; never claim success without effects."""
        if not isinstance(complete_plan, CompleteLaunchPlan):
            raise PromptToRunError("complete_plan must be a CompleteLaunchPlan")
        initial = self._initial_handle(complete_plan, adopt=adopt)
        try:
            self.factory.registry.create(initial, run_namespace="default", repository_id="prompt-runtime")
            current = initial
        except RunExistsError:
            current = self.factory.registry.reconstruct(initial.run_id)
        if current.state is RunState.RUNNING:
            return self._result(complete_plan, current, ())
        receipts: list[str] = []
        try:
            if current.continuation_action is ContinuationAction.MATERIALIZE:
                materialized = self.factory.invoke("materialize", complete_plan, current)
                receipts.append(materialized.receipt_cid)
                task_cid = str(materialized.values.get("task_source_cid") or current.task_source_cid)
                revision_cid = str(materialized.values.get("task_source_revision_cid") or current.task_source_revision_cid)
                if not task_cid or not revision_cid:
                    raise RuntimeEffectError("materialize receipt lacks durable task-source identities")
                current = self._advance(current, materialization_cid=materialized.receipt_cid, task_source_cid=task_cid, task_source_revision_cid=revision_cid, state=RunState.STARTING, continuation_action=ContinuationAction.START)
            if current.continuation_action is ContinuationAction.ADOPT:
                adopted = self.factory.invoke("adopt", complete_plan, current)
                receipts.append(adopted.receipt_cid)
                process_cid = str(adopted.values.get("process_cid") or "")
                lease_id = str(adopted.values.get("lease_id") or "")
                fencing = int(adopted.values.get("fencing_generation") or 0)
                state_revision = str(adopted.values.get("state_revision_cid") or adopted.receipt_cid)
                health_revision = str(adopted.values.get("health_revision_cid") or adopted.receipt_cid)
                cursor = str(adopted.values.get("event_cursor") or "lifecycle-adopted")
                if not process_cid or not lease_id or fencing < 1:
                    raise RuntimeEffectError("adopt receipt lacks process identity or fenced lease")
                current = self._advance(current, process_cid=process_cid, lease_id=lease_id, fencing_generation=fencing, state=RunState.RUNNING, health=RunHealth.HEALTHY, state_revision_cid=state_revision, health_revision_cid=health_revision, event_cursor=cursor, continuation_action=ContinuationAction.MONITOR)
                return self._result(complete_plan, current, tuple(receipts), adopted=True)
            if current.continuation_action is ContinuationAction.START:
                started = self.factory.invoke("start", complete_plan, current)
                receipts.append(started.receipt_cid)
                process_cid = str(started.values.get("process_cid") or "")
                lease_id = str(started.values.get("lease_id") or "")
                fencing = int(started.values.get("fencing_generation") or 0)
                state_revision = str(started.values.get("state_revision_cid") or started.receipt_cid)
                health_revision = str(started.values.get("health_revision_cid") or started.receipt_cid)
                cursor = str(started.values.get("event_cursor") or "lifecycle-started")
                if not process_cid or not lease_id or fencing < 1:
                    raise RuntimeEffectError("start receipt lacks process identity or fenced lease")
                current = self._advance(current, process_cid=process_cid, lease_id=lease_id, fencing_generation=fencing, state=RunState.RUNNING, health=RunHealth.HEALTHY, state_revision_cid=state_revision, health_revision_cid=health_revision, event_cursor=cursor, continuation_action=ContinuationAction.MONITOR)
            return self._result(complete_plan, current, tuple(receipts))
        except Exception as exc:
            if isinstance(exc, PromptToRunError):
                raise
            raise PromptToRunUnavailableError(str(exc)) from exc

    def start_or_resume(self, complete_plan: CompleteLaunchPlan) -> SupervisorInvocationResult:
        return self.run(complete_plan)

    def adopt_or_resume(self, complete_plan: CompleteLaunchPlan) -> SupervisorInvocationResult:
        return self.run(complete_plan, adopt=True)


__all__ = ["PromptToRunError", "PromptToRunSaga", "PromptToRunUnavailableError", "SupervisorIntentService"]
