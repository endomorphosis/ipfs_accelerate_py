"""Durable prompt-to-run saga and canonical objective-submission service.

The prompt-to-run path records a non-terminal run before each external effect
and only publishes ``RUNNING`` after materialization and lifecycle
birth/adoption have both supplied durable receipts.  Retrying an invocation
reconstructs that record and continues from its cursor instead of replaying
effects.

The objective-submission path is a thin adapter over the datasets-owned
``SupervisorObjectiveIntent`` / ``ObjectiveMaterializationReceipt`` contracts
and the existing content-addressed objective identity helper.  It does not
introduce a second planner, objective store, DuckDB writer, or admission path,
and callers never supply authoritative policy.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, replace
from typing import Any, Final, Mapping

from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import cid_for_dag_json

from .contracts import (
    ContinuationAction, InvocationStatus, RunHandle, RunHealth, RunState,
    SupervisorInvocationResult,
)
from .run_registry import RunExistsError
from .runtime_factory import CompleteLaunchPlan, RuntimeEffectError, StandardSupervisorRuntimeFactory


CANONICAL_OBJECTIVE_SUBMISSION_SERVICE: Final = "SupervisorIntentService@1"
CANONICAL_OBJECTIVE_SUBMISSION_ENTRYPOINT: Final = "submit_objective"

_OBJECTIVE_SUBMISSION_FORBIDDEN_KWARGS: Final = frozenset(
    {
        "authorization",
        "authorization_decision",
        "budget_profile",
        "budgets",
        "completion_authoritative",
        "dry_run",
        "duckdb",
        "ducklake",
        "effect_claims",
        "execution_authorization",
        "expected_effects",
        "fencing_epoch",
        "fencing_generation",
        "formal_plan",
        "goal_cids",
        "lease_id",
        "objective_cid",
        "objective_revision_cid",
        "partial_order",
        "plan",
        "plan_root_cid",
        "policy",
        "policy_document",
        "policy_id",
        "policy_revision",
        "quack_mutation",
        "receipt_id",
        "risk_class",
        "storage_authorization",
        "task_cids",
        "terminalize",
    }
)


class PromptToRunError(RuntimeError):
    """Base typed failure for prompt-to-run orchestration."""


class PromptToRunUnavailableError(PromptToRunError):
    """A required effect implementation is unavailable."""


class ObjectiveSubmissionError(RuntimeError):
    """Base typed failure for canonical objective submission."""


class ObjectiveSubmissionPolicyError(ObjectiveSubmissionError):
    """Caller attempted to supply authoritative policy or effect authority."""


class ObjectiveSubmissionContractError(ObjectiveSubmissionError):
    """Submitted intent failed the datasets-owned semantic contract."""


class ObjectiveSubmissionUnavailableError(ObjectiveSubmissionError):
    """A required submission dependency is unavailable."""


@dataclass(frozen=True)
class PromptToRunSaga:
    """One durable root-bound invocation and its exact continuation cursor."""

    complete_plan: CompleteLaunchPlan
    run_handle: RunHandle


def _cid(kind: str, payload: dict[str, Any]) -> str:
    return cid_for_dag_json({"kind": kind, **payload})


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _load_objective_contracts() -> tuple[Any, ...]:
    try:
        from ipfs_datasets_py.logic.intent_ir.schema import (
            OBJECTIVE_MATERIALIZATION_RECEIPT_FORBIDDEN_FIELDS,
            SUPERVISOR_OBJECTIVE_INTENT_FORBIDDEN_FIELDS,
            SUPERVISOR_OBJECTIVE_INTENT_SCHEMA,
            IntentIRValidationError,
            ObjectiveMaterializationReceipt,
            SupervisorObjectiveIntent,
            validate_objective_materialization_receipt,
            validate_supervisor_objective_intent,
        )
    except Exception as exc:  # pragma: no cover - dependency/capability gap
        raise ObjectiveSubmissionUnavailableError(
            "ipfs_datasets_py SupervisorObjectiveIntent contracts are unavailable"
        ) from exc
    return (
        OBJECTIVE_MATERIALIZATION_RECEIPT_FORBIDDEN_FIELDS,
        SUPERVISOR_OBJECTIVE_INTENT_FORBIDDEN_FIELDS,
        SUPERVISOR_OBJECTIVE_INTENT_SCHEMA,
        IntentIRValidationError,
        ObjectiveMaterializationReceipt,
        SupervisorObjectiveIntent,
        validate_objective_materialization_receipt,
        validate_supervisor_objective_intent,
    )


def _reject_authority_overrides(intent: Any, overrides: Mapping[str, Any]) -> None:
    (
        receipt_forbidden,
        intent_forbidden,
        *_rest,
    ) = _load_objective_contracts()
    forbidden = (
        set(intent_forbidden)
        | set(receipt_forbidden)
        | set(_OBJECTIVE_SUBMISSION_FORBIDDEN_KWARGS)
    )
    if overrides:
        hit = sorted(key for key in overrides if key in forbidden)
        if hit:
            raise ObjectiveSubmissionPolicyError(
                "objective submission rejects caller-supplied authority fields: "
                + ", ".join(hit)
            )
        unknown = sorted(key for key in overrides if key not in forbidden)
        if unknown:
            raise ObjectiveSubmissionPolicyError(
                "objective submission rejects unknown overrides: "
                + ", ".join(unknown)
            )
    if isinstance(intent, Mapping):
        hit = sorted(key for key in intent if key in forbidden)
        if hit:
            raise ObjectiveSubmissionPolicyError(
                "objective submission rejects caller-supplied authority fields: "
                + ", ".join(hit)
            )


def submit_objective(
    intent: Any,
    **overrides: Any,
) -> Any:
    """Validate a direct objective intent and return a materialization receipt.

    This is intentionally a thin naming adapter on the existing intent service,
    not a second objective subsystem.  Semantic validation stays with datasets;
    objective/revision identities reuse
    :func:`content_addressed_prompt_objective`.  The receipt is evidence only
    and never completes work, writes DuckDB, or accepts caller policy.
    """

    (
        _receipt_forbidden,
        _intent_forbidden,
        intent_schema,
        IntentIRValidationError,
        ObjectiveMaterializationReceipt,
        SupervisorObjectiveIntent,
        validate_objective_materialization_receipt,
        validate_supervisor_objective_intent,
    ) = _load_objective_contracts()
    from .objective_resolver import content_addressed_prompt_objective

    if overrides:
        _reject_authority_overrides(intent, overrides)
    elif isinstance(intent, Mapping):
        _reject_authority_overrides(intent, {})

    if isinstance(intent, str) or not isinstance(
        intent, (Mapping, SupervisorObjectiveIntent)
    ):
        raise ObjectiveSubmissionContractError(
            "submit_objective requires a SupervisorObjectiveIntent or mapping"
        )

    try:
        validated = validate_supervisor_objective_intent(intent)
    except IntentIRValidationError as exc:
        message = str(exc)
        lowered = message.lower()
        if "forbid" in lowered or "authoritative" in lowered:
            raise ObjectiveSubmissionPolicyError(message) from exc
        raise ObjectiveSubmissionContractError(message) from exc

    if validated.callers_supply_authoritative_policy:
        raise ObjectiveSubmissionPolicyError(
            "SupervisorObjectiveIntent must not supply authoritative policy"
        )

    intent_payload = validated.to_dict()
    intent_sha256 = hashlib.sha256(_canonical_json_bytes(intent_payload)).hexdigest()
    prompt_cid = cid_for_dag_json(
        {"schema": intent_schema, "intent": intent_payload}
    )
    objective_cid, objective_revision_cid, _pending_plan_cid = (
        content_addressed_prompt_objective(
            prompt_cid,
            repository_id=validated.repository_id,
        )
    )
    # pending_plan_cid is intentionally dropped: ObjectiveMaterializationReceipt
    # forbids plan / plan_root_cid fields and is not plan-admission authority.
    receipt_id = cid_for_dag_json(
        {
            "kind": "objective-materialization",
            "intent_id": validated.intent_id,
            "intent_sha256": intent_sha256,
            "objective_id": validated.intent_id,
            "objective_cid": objective_cid,
            "objective_revision_cid": objective_revision_cid,
        }
    )
    receipt = ObjectiveMaterializationReceipt(
        receipt_id=receipt_id,
        intent_id=validated.intent_id,
        intent_sha256=intent_sha256,
        objective_id=validated.intent_id,
        objective_cid=objective_cid,
        objective_revision_cid=objective_revision_cid,
    )
    return validate_objective_materialization_receipt(receipt)


class SupervisorIntentService:
    """Canonical intent service: objective submission and prompt-to-run saga."""

    def __init__(
        self, *, factory: StandardSupervisorRuntimeFactory | None = None
    ) -> None:
        self.factory = factory

    def submit_objective(self, intent: Any, **overrides: Any) -> Any:
        """Instance adapter for :func:`submit_objective`."""

        return submit_objective(intent, **overrides)

    def _require_factory(self) -> StandardSupervisorRuntimeFactory:
        if self.factory is None:
            raise PromptToRunUnavailableError(
                "runtime factory is required for prompt-to-run orchestration"
            )
        return self.factory

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
        factory = self._require_factory()
        now = int(time.time() * 1000)
        next_handle = replace(current, run_revision=current.run_revision + 1, updated_at_ms=max(now, current.updated_at_ms), **changes)
        factory.registry.cas_update(next_handle, expected_revision=current.run_revision, expected_handle_cid=current.content_id, expected_semantic_id=current.semantic_id)
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
        factory = self._require_factory()
        if not isinstance(complete_plan, CompleteLaunchPlan):
            raise PromptToRunError("complete_plan must be a CompleteLaunchPlan")
        initial = self._initial_handle(complete_plan, adopt=adopt)
        try:
            factory.registry.create(initial, run_namespace="default", repository_id="prompt-runtime")
            current = initial
        except RunExistsError:
            current = factory.registry.reconstruct(initial.run_id)
        if current.state is RunState.RUNNING:
            return self._result(complete_plan, current, ())
        receipts: list[str] = []
        try:
            if current.continuation_action is ContinuationAction.MATERIALIZE:
                materialized = factory.invoke("materialize", complete_plan, current)
                receipts.append(materialized.receipt_cid)
                task_cid = str(materialized.values.get("task_source_cid") or current.task_source_cid)
                revision_cid = str(materialized.values.get("task_source_revision_cid") or current.task_source_revision_cid)
                if not task_cid or not revision_cid:
                    raise RuntimeEffectError("materialize receipt lacks durable task-source identities")
                current = self._advance(current, materialization_cid=materialized.receipt_cid, task_source_cid=task_cid, task_source_revision_cid=revision_cid, state=RunState.STARTING, continuation_action=ContinuationAction.START)
            if current.continuation_action is ContinuationAction.ADOPT:
                adopted = factory.invoke("adopt", complete_plan, current)
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
                started = factory.invoke("start", complete_plan, current)
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


__all__ = [
    "CANONICAL_OBJECTIVE_SUBMISSION_ENTRYPOINT",
    "CANONICAL_OBJECTIVE_SUBMISSION_SERVICE",
    "ObjectiveSubmissionContractError",
    "ObjectiveSubmissionError",
    "ObjectiveSubmissionPolicyError",
    "ObjectiveSubmissionUnavailableError",
    "PromptToRunError",
    "PromptToRunSaga",
    "PromptToRunUnavailableError",
    "SupervisorIntentService",
    "submit_objective",
]
