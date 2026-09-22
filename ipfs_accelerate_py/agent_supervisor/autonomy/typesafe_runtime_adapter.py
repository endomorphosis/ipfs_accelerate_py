"""DecisionRuntime-side TypeSafe handoff.

Call TypeSafe *before* the provider-free meta-controller ``step``. Never use
TypeSafe receipts as execution-permit authority. ``DecisionRuntime.decide``
stays the only effect/permit spine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from ipfs_accelerate_py.agent_supervisor.autonomy.contracts import ResolutionCandidate
from ipfs_accelerate_py.agent_supervisor.autonomy.runtime import MetaControllerStep
from ipfs_accelerate_py.agent_supervisor.autonomy.typesafe_decision import (
    admit_after_typesafe_advice,
    prepare_step_candidates,
)
from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor import (
    DecisionQuestionAdvice,
)

TYPESAFE_EVIDENCE_PREFIX = "typesafe-advice-"


class TypesafeAuthorityError(PermissionError):
    """Raised when TypeSafe advice is offered as execution authority."""


@dataclass(frozen=True)
class TypesafeStepHandoff:
    step: MetaControllerStep
    advice: Optional[DecisionQuestionAdvice]
    authorizes_effect: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "authorizes_effect", False)

    @property
    def admitted(self) -> bool:
        return bool(self.step.admitted)

    @property
    def requires_decision_runtime(self) -> bool:
        return bool(self.step.requires_decision_runtime)

    def to_dict(self) -> dict[str, Any]:
        return {
            "admitted": self.admitted,
            "authorizes_effect": False,
            "requires_decision_runtime": self.requires_decision_runtime,
            "step_status": self.step.status.value,
            "selected_action": (
                self.step.decision.selected_action.value
                if self.step.decision.selected_action is not None
                else ""
            ),
            "advice": None if self.advice is None else self.advice.to_dict(),
        }


def typesafe_receipt_id(value: Any) -> str:
    if hasattr(value, "receipt_id"):
        return str(getattr(value, "receipt_id") or "")
    if isinstance(value, Mapping):
        return str(value.get("receipt_id") or "")
    return str(value or "")


def typesafe_authority(value: Any) -> str:
    if hasattr(value, "authority"):
        return str(getattr(value, "authority") or "")
    if isinstance(value, Mapping):
        return str(value.get("authority") or "")
    return ""


def assert_typesafe_not_execution_authority(evidence_receipts: Sequence[Any]) -> None:
    """Fail closed if TypeSafe is the only evidence or is labeled authoritative."""

    rows = tuple(evidence_receipts or ())
    if not rows:
        return
    typesafe_ids = []
    authoritative_typesafe = False
    for item in rows:
        receipt_id = typesafe_receipt_id(item)
        authority = typesafe_authority(item).casefold()
        if receipt_id.startswith(TYPESAFE_EVIDENCE_PREFIX):
            typesafe_ids.append(receipt_id)
            if authority in {"authoritative", "verified", "derived"}:
                authoritative_typesafe = True
    if authoritative_typesafe:
        raise TypesafeAuthorityError("typesafe advice cannot be execution authority")
    if typesafe_ids and len(typesafe_ids) == len(rows):
        raise TypesafeAuthorityError("typesafe advice cannot be the sole execution evidence")


class TypesafeSupervisorAdapter:
    """Advise, then ``step``. Implements ``DecisionRuntimeAdapter`` by denial.

    ``admit_with_decision_runtime`` never calls ``DecisionRuntime.decide`` from
    TypeSafe evidence. A downstream owner must supply kernel/SMT/plan receipts.
    """

    def __init__(
        self,
        meta: Any,
        *,
        privacy_class: str = "repository_private",
        timeout: float = 30.0,
    ) -> None:
        self._meta = meta
        self.privacy_class = privacy_class
        self.timeout = timeout

    def prepare_and_step(
        self,
        *,
        candidates: Sequence[ResolutionCandidate],
        context: Any,
        state: Mapping[str, Any],
        meaningful_change: bool = True,
    ) -> TypesafeStepHandoff:
        step, advice = admit_after_typesafe_advice(
            self._meta,
            candidates=candidates,
            context=context,
            state=state,
            privacy_class=self.privacy_class,
            timeout=self.timeout,
            meaningful_change=meaningful_change,
        )
        return TypesafeStepHandoff(step=step, advice=advice)

    def admit_with_decision_runtime(
        self,
        step: MetaControllerStep,
        *,
        runtime: Any = None,
        runtime_input: Any = None,
    ) -> object:
        if not step.admitted:
            raise TypesafeAuthorityError("step is not budget-admitted")
        evidence = ()
        if runtime_input is not None:
            evidence = tuple(getattr(runtime_input, "evidence_receipts", ()) or ())
        assert_typesafe_not_execution_authority(evidence)
        if runtime is None or runtime_input is None:
            raise TypesafeAuthorityError(
                "decision runtime input with non-advisory evidence is required"
            )
        return runtime.decide(runtime_input)

    def handle_wake(
        self,
        runtime: Any,
        event: Any,
        *,
        candidates: Sequence[ResolutionCandidate],
        context: Any,
        state: Mapping[str, Any],
        **wake_kwargs: Any,
    ) -> TypesafeWakeHandoff:
        """Advise, then run provider-free ``handle_wake`` on the filtered set.

        ``AutonomyRuntime.handle_wake`` still never calls a model.
        ``model_called`` on the cycle receipt must stay false.
        """

        from ipfs_accelerate_py.agent_supervisor.autonomy.runtime import (
            AutonomyWakeEvent,
        )

        bound = (
            event
            if isinstance(event, AutonomyWakeEvent)
            else AutonomyWakeEvent.from_runtime_wake(event)
        )
        merged = dict(state)
        if bound.lane_id and not merged.get("lane_id"):
            merged["lane_id"] = bound.lane_id
        if bound.subject_id and not merged.get("subject_id"):
            merged["subject_id"] = bound.subject_id
        controller = runtime.controller
        question = controller.next_unresolved_question()
        advice = None
        prepared: Sequence[ResolutionCandidate] = tuple(candidates)
        if question is not None:
            remote = bool(getattr(context, "remote_disclosure_permitted", False))
            try:
                prepared, _updated, advice = prepare_step_candidates(
                    controller.decision_graph,
                    question,
                    candidates,
                    state=merged,
                    privacy_class=self.privacy_class,
                    remote_disclosure_permitted=remote,
                    timeout=self.timeout,
                )
            except Exception:
                prepared, advice = tuple(candidates), None
        result = runtime.handle_wake(
            bound,
            candidates=tuple(prepared),
            context=context,
            typesafe_prepare=False,
            typesafe_state=merged,
            **wake_kwargs,
        )
        if getattr(result, "model_called", False):
            raise TypesafeAuthorityError("autonomy runtime must remain provider-free")
        return TypesafeWakeHandoff(result=result, advice=advice)


@dataclass(frozen=True)
class TypesafeWakeHandoff:
    result: Any
    advice: Optional[DecisionQuestionAdvice]
    authorizes_effect: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "authorizes_effect", False)

    @property
    def model_called(self) -> bool:
        return bool(getattr(self.result, "model_called", False))

    def to_dict(self) -> dict[str, Any]:
        return {
            "authorizes_effect": False,
            "model_called": self.model_called,
            "status": getattr(getattr(self.result, "status", None), "value", ""),
            "advice": None if self.advice is None else self.advice.to_dict(),
        }


def dispatch_autonomy_wake(
    runtime: Any,
    event: Any,
    *,
    candidates: Sequence[ResolutionCandidate],
    context: Any,
    state: Mapping[str, Any] | None = None,
    privacy_class: str = "repository_private",
    timeout: float = 30.0,
    **wake_kwargs: Any,
) -> TypesafeWakeHandoff:
    """Production wake entry: advise, then provider-free ``handle_wake``.

    ``AutonomyRuntime.handle_wake`` is unchanged and still never calls a model.
    """

    adapter = TypesafeSupervisorAdapter(
        runtime.controller,
        privacy_class=privacy_class,
        timeout=timeout,
    )
    return adapter.handle_wake(
        runtime,
        event,
        candidates=candidates,
        context=context,
        state=state or {},
        **wake_kwargs,
    )


__all__ = [
    "TYPESAFE_EVIDENCE_PREFIX",
    "TypesafeAuthorityError",
    "TypesafeStepHandoff",
    "TypesafeSupervisorAdapter",
    "TypesafeWakeHandoff",
    "assert_typesafe_not_execution_authority",
    "dispatch_autonomy_wake",
]
