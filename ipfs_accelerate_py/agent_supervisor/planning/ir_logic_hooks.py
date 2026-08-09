"""Planning hooks that enforce mandatory IR logic stages (DCR-035).

Every diagnose / plan / admit / apply / complete decision consults
:class:`~ipfs_accelerate_py.agent_supervisor.proof.ir_logic_application.RequiredLogicStageGate`
before the decision may pass.  The hooks:

* never swallow required-path exceptions;
* never treat partial stage coverage as pass;
* never promote UI bridge-only availability to production capability;
* never default-true safety or execution claims; and
* never grant execution when any required stage is empty, skipped,
  unsupported, unknown, errored, or import-failed.

The hooks are intentionally thin.  They bind planning decisions to the sealed
proof-side gate and return the content-addressed
:class:`~ipfs_accelerate_py.agent_supervisor.proof.ir_logic_application.IRApplicationResult`.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Final

from ..proof.ir_logic_application import (
    IR_APPLICATION_FAILED,
    IRApplicationResult,
    IRLogicApplicationError,
    LogicDecisionKind,
    LogicStageId,
    LogicStageObservation,
    POLICY_DECISION_KINDS,
    REQUIRED_LOGIC_STAGES,
    REQUIRED_LOGIC_STAGE_GATE_INTERFACE,
    RequiredLogicStageGate,
    StageDisposition,
    SurfaceKind,
    UNKNOWN_GATE_EVIDENCE_TERM,
    all_stages_passed,
    apply_ir_logic,
    require_gate_pass,
    stage_observation,
)


IR_LOGIC_HOOKS_INTERFACE: Final = "IRLogicHooks@1"
IR_LOGIC_HOOKS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/ir-logic-hooks@1"
)
IR_LOGIC_HOOKS_VERSION: Final = "1"
DCR_TASK_ID: Final = "DCR-035"

# Decision kinds that require an explicit execution claim after a full pass.
_EXECUTION_DECISIONS: Final[frozenset[str]] = frozenset(
    {
        LogicDecisionKind.APPLY.value,
        LogicDecisionKind.COMPLETE.value,
    }
)


class IRLogicHookError(IRLogicApplicationError):
    """Raised when a planning IR logic hook is misconfigured or fails closed."""


class HookPhase(str, Enum):
    """Closed hook phases aligned with policy decision kinds."""

    DIAGNOSE = "diagnose"
    PLAN = "plan"
    ADMIT = "admit"
    APPLY = "apply"
    COMPLETE = "complete"

    @classmethod
    def from_decision(cls, decision: LogicDecisionKind | str) -> "HookPhase":
        if isinstance(decision, LogicDecisionKind):
            return cls(decision.value)
        text = str(getattr(decision, "value", decision)).strip()
        try:
            return cls(text)
        except ValueError as exc:
            raise IRLogicHookError(
                f"unknown hook phase {text!r}; fail closed",
                reason_code="unknown_hook_phase",
            ) from exc


def _as_observations(
    value: Sequence[LogicStageObservation | Mapping[str, Any]] | None,
) -> tuple[LogicStageObservation, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise IRLogicHookError(
            "observations must be a sequence",
            reason_code="invalid_field_type",
        )
    parsed: list[LogicStageObservation] = []
    for item in value:
        if isinstance(item, LogicStageObservation):
            parsed.append(item)
        elif isinstance(item, Mapping):
            parsed.append(LogicStageObservation.from_dict(item))
        else:
            raise IRLogicHookError(
                "each observation must be a LogicStageObservation or mapping",
                reason_code="invalid_field_type",
            )
    return tuple(parsed)


def empty_surface_observation(
    stage: LogicStageId | str,
    *,
    detail: str = "empty logic surface",
    family: str = "",
) -> LogicStageObservation:
    """Observation for an empty required surface (cannot pass or grant)."""

    return stage_observation(
        stage,
        StageDisposition.EMPTY_SURFACE,
        surface_kind=SurfaceKind.EMPTY,
        reason_codes=("empty_surface",),
        detail=detail,
        family=family,
    )


def skipped_stage_observation(
    stage: LogicStageId | str,
    *,
    detail: str = "required stage skipped",
) -> LogicStageObservation:
    """Observation for a skipped required stage (cannot pass or grant)."""

    return stage_observation(
        stage,
        StageDisposition.SKIP,
        reason_codes=("skipped_stage",),
        detail=detail,
    )


def unsupported_semantics_observation(
    stage: LogicStageId | str,
    *,
    detail: str = "unsupported semantics",
    family: str = "",
) -> LogicStageObservation:
    """Observation for unsupported semantics (cannot pass or grant)."""

    return stage_observation(
        stage,
        StageDisposition.UNSUPPORTED,
        reason_codes=("unsupported_semantics",),
        detail=detail,
        family=family,
    )


def import_failure_observation(
    stage: LogicStageId | str,
    *,
    detail: str = "import failure",
    module_origin: str = "",
    exception_swallowed: bool = False,
) -> LogicStageObservation:
    """Observation for an import failure (cannot pass or grant)."""

    return stage_observation(
        stage,
        StageDisposition.IMPORT_FAILURE,
        surface_kind=SurfaceKind.IMPORT,
        reason_codes=("import_failure",),
        detail=detail,
        module_origin=module_origin,
        exception_swallowed=exception_swallowed,
    )


def bridge_only_observation(
    stage: LogicStageId | str,
    *,
    detail: str = "UI bridge-only projection",
    family: str = "ui_ir",
) -> LogicStageObservation:
    """Observation for a UI bridge-only projection (cannot pass or grant)."""

    return stage_observation(
        stage,
        StageDisposition.BRIDGE_ONLY,
        surface_kind=SurfaceKind.BRIDGE_ONLY,
        reason_codes=("ui_bridge_only_projection",),
        detail=detail,
        family=family,
    )


def unknown_observation(
    stage: LogicStageId | str,
    *,
    detail: str = "unknown stage outcome",
) -> LogicStageObservation:
    """Observation for an unknown stage outcome (cannot pass or grant)."""

    return stage_observation(
        stage,
        StageDisposition.UNKNOWN,
        surface_kind=SurfaceKind.UNKNOWN,
        reason_codes=("unknown_outcome",),
        detail=detail,
    )


def error_observation(
    stage: LogicStageId | str,
    *,
    detail: str = "stage error",
    exception_swallowed: bool = False,
) -> LogicStageObservation:
    """Observation for a stage error (cannot pass or grant)."""

    return stage_observation(
        stage,
        StageDisposition.ERROR,
        reason_codes=("stage_error",),
        detail=detail,
        exception_swallowed=exception_swallowed,
    )


def production_pass_observation(
    stage: LogicStageId | str,
    *,
    detail: str = "stage passed",
    family: str = "",
    module_origin: str = "",
) -> LogicStageObservation:
    """Observation for a production-path stage pass."""

    return stage_observation(
        stage,
        StageDisposition.PASS,
        surface_kind=SurfaceKind.PRODUCTION,
        reason_codes=("stage_passed",),
        detail=detail,
        family=family,
        module_origin=module_origin,
    )


@dataclass(frozen=True, slots=True)
class IRLogicHookReceipt:
    """One decision-hook consultation bound to an IR application result."""

    phase: HookPhase
    result: IRApplicationResult
    raised: bool = False

    @property
    def gate_passed(self) -> bool:
        return self.result.gate_passed

    @property
    def execution_granted(self) -> bool:
        return self.result.execution_granted

    @property
    def no_false_grant(self) -> bool:
        return self.result.no_false_grant

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": IR_LOGIC_HOOKS_SCHEMA,
            "interface": IR_LOGIC_HOOKS_INTERFACE,
            "phase": self.phase.value,
            "raised": self.raised,
            "gate_passed": self.gate_passed,
            "execution_granted": self.execution_granted,
            "no_false_grant": self.no_false_grant,
            "result": self.result.to_dict(),
        }


@dataclass
class IRLogicHooks:
    """Planning-side hooks for mandatory IR logic stages.

    Interface: ``IRLogicHooks@1``.

    Call :meth:`run` (or the phase helpers) with stage observations for the
    decision under consideration.  Failures never grant execution and never
    report a false pass.
    """

    INTERFACE: ClassVar[str] = IR_LOGIC_HOOKS_INTERFACE

    gate: RequiredLogicStageGate = field(default_factory=RequiredLogicStageGate)
    raise_on_failure: bool = True
    _history: list[IRLogicHookReceipt] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.gate, RequiredLogicStageGate):
            raise IRLogicHookError(
                "gate must be a RequiredLogicStageGate",
                reason_code="invalid_field_type",
            )
        if not isinstance(self.raise_on_failure, bool):
            raise IRLogicHookError(
                "raise_on_failure must be a boolean",
                reason_code="invalid_field_type",
            )

    @property
    def interface(self) -> str:
        return IR_LOGIC_HOOKS_INTERFACE

    @property
    def history(self) -> tuple[IRLogicHookReceipt, ...]:
        return tuple(self._history)

    def clear_history(self) -> None:
        self._history.clear()

    def run(
        self,
        decision: LogicDecisionKind | HookPhase | str,
        observations: Sequence[LogicStageObservation | Mapping[str, Any]],
        *,
        claim_execution: bool | None = None,
        raise_on_failure: bool | None = None,
    ) -> IRApplicationResult:
        """Consult the mandatory stage gate for one planning decision.

        Required-path exceptions are never swallowed: they re-raise as
        :class:`IRLogicHookError` with ``IR_APPLICATION_FAILED``.
        """

        phase = HookPhase.from_decision(
            decision.value if isinstance(decision, HookPhase) else decision
        )
        decision_kind = LogicDecisionKind(phase.value)
        should_raise = (
            self.raise_on_failure
            if raise_on_failure is None
            else bool(raise_on_failure)
        )

        if claim_execution is None:
            # Never default-true: apply/complete still require an explicit claim.
            claim = False
        else:
            claim = bool(claim_execution)

        try:
            obs = _as_observations(observations)
            result = apply_ir_logic(
                decision_kind,
                obs,
                claim_execution=claim,
                gate=self.gate,
            )
        except IRLogicApplicationError:
            raise
        except Exception as exc:  # noqa: BLE001 - required path must not swallow
            # Convert unexpected failures into a fail-closed gate result rather
            # than inventing a pass.  Re-raise when configured.
            failure = apply_ir_logic(
                decision_kind,
                (
                    error_observation(
                        LogicStageId.NORMALIZE,
                        detail=f"hook exception: {type(exc).__name__}: {exc}",
                        exception_swallowed=False,
                    ),
                ),
                claim_execution=False,
                gate=self.gate,
            )
            receipt = IRLogicHookReceipt(
                phase=phase, result=failure, raised=should_raise
            )
            self._history.append(receipt)
            if should_raise:
                raise IRLogicHookError(
                    f"IR logic hook for {phase.value} failed: {exc}",
                    reason_code=IR_APPLICATION_FAILED,
                    details={"phase": phase.value, "error": type(exc).__name__},
                ) from exc
            return failure

        receipt = IRLogicHookReceipt(
            phase=phase,
            result=result,
            raised=should_raise and not result.gate_passed,
        )
        self._history.append(receipt)

        if should_raise and not result.gate_passed:
            raise IRLogicHookError(
                result.detail or f"mandatory logic stages failed for {phase.value}",
                reason_code=IR_APPLICATION_FAILED,
                details={
                    "phase": phase.value,
                    "reason_codes": list(result.reason_codes),
                    "ran_stages": list(result.ran_stages),
                    "pass_stages": list(result.pass_stages),
                    "unknown_rows": list(result.unknown_rows),
                    "unsupported_rows": list(result.unsupported_rows),
                    "error_rows": list(result.error_rows),
                },
            )
        return result

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    def diagnose(
        self,
        observations: Sequence[LogicStageObservation | Mapping[str, Any]],
        *,
        raise_on_failure: bool | None = None,
    ) -> IRApplicationResult:
        return self.run(
            LogicDecisionKind.DIAGNOSE,
            observations,
            claim_execution=False,
            raise_on_failure=raise_on_failure,
        )

    def plan(
        self,
        observations: Sequence[LogicStageObservation | Mapping[str, Any]],
        *,
        raise_on_failure: bool | None = None,
    ) -> IRApplicationResult:
        return self.run(
            LogicDecisionKind.PLAN,
            observations,
            claim_execution=False,
            raise_on_failure=raise_on_failure,
        )

    def admit(
        self,
        observations: Sequence[LogicStageObservation | Mapping[str, Any]],
        *,
        raise_on_failure: bool | None = None,
    ) -> IRApplicationResult:
        return self.run(
            LogicDecisionKind.ADMIT,
            observations,
            claim_execution=False,
            raise_on_failure=raise_on_failure,
        )

    def apply(
        self,
        observations: Sequence[LogicStageObservation | Mapping[str, Any]],
        *,
        claim_execution: bool = False,
        raise_on_failure: bool | None = None,
    ) -> IRApplicationResult:
        return self.run(
            LogicDecisionKind.APPLY,
            observations,
            claim_execution=claim_execution,
            raise_on_failure=raise_on_failure,
        )

    def complete(
        self,
        observations: Sequence[LogicStageObservation | Mapping[str, Any]],
        *,
        claim_execution: bool = False,
        raise_on_failure: bool | None = None,
    ) -> IRApplicationResult:
        return self.run(
            LogicDecisionKind.COMPLETE,
            observations,
            claim_execution=claim_execution,
            raise_on_failure=raise_on_failure,
        )

    def require_execution(
        self,
        decision: LogicDecisionKind | HookPhase | str,
        observations: Sequence[LogicStageObservation | Mapping[str, Any]],
    ) -> IRApplicationResult:
        """Run an apply/complete decision that must both pass and grant.

        Non-execution decisions fail closed: they cannot grant execution.
        """

        phase = HookPhase.from_decision(
            decision.value if isinstance(decision, HookPhase) else decision
        )
        if phase.value not in _EXECUTION_DECISIONS:
            raise IRLogicHookError(
                f"decision {phase.value!r} cannot grant execution",
                reason_code=IR_APPLICATION_FAILED,
                details={"phase": phase.value},
            )
        result = self.run(
            phase,
            observations,
            claim_execution=True,
            raise_on_failure=True,
        )
        return require_gate_pass(result, require_execution=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": IR_LOGIC_HOOKS_SCHEMA,
            "interface": IR_LOGIC_HOOKS_INTERFACE,
            "version": IR_LOGIC_HOOKS_VERSION,
            "task_id": DCR_TASK_ID,
            "evidence_term": UNKNOWN_GATE_EVIDENCE_TERM,
            "gate_interface": REQUIRED_LOGIC_STAGE_GATE_INTERFACE,
            "required_stages": list(REQUIRED_LOGIC_STAGES),
            "policy_decision_kinds": list(POLICY_DECISION_KINDS),
            "raise_on_failure": self.raise_on_failure,
            "gate": self.gate.to_dict(),
            "history": [item.to_dict() for item in self._history],
            "acceptance": {
                "empty_surfaces_cannot_pass_or_grant": True,
                "skipped_stages_cannot_pass_or_grant": True,
                "unsupported_semantics_cannot_pass_or_grant": True,
                "import_failures_cannot_pass_or_grant": True,
                "ui_bridge_only_projections_cannot_pass_or_grant": True,
                "exception_swallowing_forbidden": True,
                "partial_stage_pass_forbidden": True,
                "default_true_safety_claims_forbidden": True,
                "no_false_grant": True,
            },
        }


def default_ir_logic_hooks(
    *,
    raise_on_failure: bool = True,
) -> IRLogicHooks:
    """Construct the default fail-closed IR logic hooks."""

    return IRLogicHooks(
        gate=RequiredLogicStageGate(),
        raise_on_failure=raise_on_failure,
    )


def consult_ir_logic_gate(
    decision: LogicDecisionKind | str,
    observations: Sequence[LogicStageObservation | Mapping[str, Any]],
    *,
    claim_execution: bool = False,
    raise_on_failure: bool = True,
) -> IRApplicationResult:
    """Module-level convenience for one mandatory gate consultation."""

    hooks = default_ir_logic_hooks(raise_on_failure=raise_on_failure)
    return hooks.run(
        decision,
        observations,
        claim_execution=claim_execution,
        raise_on_failure=raise_on_failure,
    )


def build_stage_map(
    overrides: Mapping[str, LogicStageObservation | Mapping[str, Any]] | None = None,
    *,
    default_disposition: StageDisposition | str = StageDisposition.PASS,
    default_surface: SurfaceKind | str = SurfaceKind.PRODUCTION,
) -> tuple[LogicStageObservation, ...]:
    """Build a full required-stage observation set with optional overrides.

    Missing stages receive the default disposition.  Callers that want a
    fail-closed missing-stage test should omit stages from the returned set
    rather than using this helper.
    """

    override_map: MutableMapping[str, LogicStageObservation] = {}
    if overrides:
        for key, value in overrides.items():
            stage_id = str(key).strip()
            if isinstance(value, LogicStageObservation):
                override_map[stage_id] = value
            elif isinstance(value, Mapping):
                payload = dict(value)
                payload.setdefault("stage", stage_id)
                override_map[stage_id] = LogicStageObservation.from_dict(payload)
            else:
                raise IRLogicHookError(
                    "stage override must be a LogicStageObservation or mapping",
                    reason_code="invalid_field_type",
                )

    observations: list[LogicStageObservation] = []
    for stage in REQUIRED_LOGIC_STAGES:
        if stage in override_map:
            observations.append(override_map[stage])
        else:
            disposition_value = (
                default_disposition.value
                if isinstance(default_disposition, StageDisposition)
                else str(default_disposition).strip()
            )
            observations.append(
                stage_observation(
                    stage,
                    default_disposition,
                    surface_kind=default_surface,
                    reason_codes=(
                        ("stage_passed",)
                        if disposition_value == StageDisposition.PASS.value
                        else ("stage_default",)
                    ),
                )
            )
    return tuple(observations)


__all__ = [
    "DCR_TASK_ID",
    "HookPhase",
    "IR_LOGIC_HOOKS_INTERFACE",
    "IR_LOGIC_HOOKS_SCHEMA",
    "IR_LOGIC_HOOKS_VERSION",
    "IRLogicHookError",
    "IRLogicHookReceipt",
    "IRLogicHooks",
    "all_stages_passed",
    "bridge_only_observation",
    "build_stage_map",
    "consult_ir_logic_gate",
    "default_ir_logic_hooks",
    "empty_surface_observation",
    "error_observation",
    "import_failure_observation",
    "production_pass_observation",
    "skipped_stage_observation",
    "unknown_observation",
    "unsupported_semantics_observation",
]
