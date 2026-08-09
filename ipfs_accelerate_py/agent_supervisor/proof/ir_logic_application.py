"""Enforce mandatory IR logic stages and fail closed on unknown (DCR-035).

Interfaces
----------
* ``RequiredLogicStageGate@1`` — fixed-order policy gate over the exact
  required logic-stage set for diagnose / plan / admit / apply / complete.
* ``IRApplicationResult@1`` — content-addressed application receipt that
  records required/ran/pass stage sets, unknown/unsupported/error rows, and
  the no-false-grant claim.

Normative rules (fail-closed)
-----------------------------
* Empty surfaces cannot pass or grant execution.
* Skipped stages cannot pass or grant execution.
* Unsupported semantics cannot pass or grant execution.
* Import failures cannot be swallowed and cannot pass or grant execution.
* UI bridge-only projections cannot pass or grant execution.
* Unknown, error, and partial-stage outcomes never promote to pass.
* Only the exact policy-required stage set, fully ran and fully passed, may
  allow a decision to pass; execution is granted only for apply/complete when
  that full pass holds.
* Default-true safety claims and exception swallowing on the required path are
  forbidden.

Evidence term: ``dcr/unknown-gate@1``.
Generated artifact:
``data/agent_supervisor/deterministic_contract_repair/logic-gate.json``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Final

from .formal_verification_contracts import (
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)


# ---------------------------------------------------------------------------
# Interfaces / schemas / constants
# ---------------------------------------------------------------------------

REQUIRED_LOGIC_STAGE_GATE_INTERFACE: Final = "RequiredLogicStageGate@1"
IR_APPLICATION_RESULT_INTERFACE: Final = "IRApplicationResult@1"

REQUIRED_LOGIC_STAGE_GATE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/required-logic-stage-gate@1"
)
IR_APPLICATION_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/ir-application-result@1"
)
LOGIC_STAGE_OBSERVATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/logic-stage-observation@1"
)
LOGIC_GATE_ARTIFACT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/logic-gate-artifact@1"
)

UNKNOWN_GATE_EVIDENCE_TERM: Final = "dcr/unknown-gate@1"
IR_APPLICATION_FAILED: Final = "IR_APPLICATION_FAILED"
CONTRACT_VERSION: Final[int] = 1
LOGIC_GATE_VERSION: Final = "1"
DCR_TASK_ID: Final = "DCR-035"
DCR_ARTIFACT_PATH: Final = (
    "data/agent_supervisor/deterministic_contract_repair/logic-gate.json"
)

DEFAULT_MAX_BYTES: Final[int] = 1_048_576
_MAX_FIELD_BYTES: Final[int] = 4_096
_MAX_REASON_CODES: Final[int] = 64
_MAX_ROWS: Final[int] = 4_096

# Closed pipeline stages produced by DCR-030..DCR-034.  Every decision that
# consults the gate requires this exact set to have both ran and passed.
REQUIRED_LOGIC_STAGES: Final[tuple[str, ...]] = (
    "normalize",
    "obligate",
    "route",
    "reconstruct",
    "cache",
)

# Policy decision kinds that must consult the mandatory stage gate.
POLICY_DECISION_KINDS: Final[tuple[str, ...]] = (
    "diagnose",
    "plan",
    "admit",
    "apply",
    "complete",
)

# Decisions that may grant execution when (and only when) every required stage
# has both ran and passed with no fail-closed rows.
_EXECUTION_DECISIONS: Final[frozenset[str]] = frozenset({"apply", "complete"})

# Dispositions that never contribute to the pass set and always block grant.
_BLOCKING_DISPOSITIONS: Final[frozenset[str]] = frozenset(
    {
        "fail",
        "skip",
        "unknown",
        "unsupported",
        "error",
        "empty_surface",
        "bridge_only",
        "import_failure",
        "partial",
    }
)


# ---------------------------------------------------------------------------
# Errors / enums
# ---------------------------------------------------------------------------


class IRLogicApplicationError(ContractValidationError):
    """Raised when IR logic application inputs violate a closed invariant."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = IR_APPLICATION_FAILED,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.details = dict(details or {})


class LogicDecisionKind(str, Enum):
    """Closed policy decisions that must consult the mandatory stage gate."""

    DIAGNOSE = "diagnose"
    PLAN = "plan"
    ADMIT = "admit"
    APPLY = "apply"
    COMPLETE = "complete"

    @property
    def may_grant_execution(self) -> bool:
        return self.value in _EXECUTION_DECISIONS


class LogicStageId(str, Enum):
    """Closed mandatory IR logic pipeline stages (DCR-030..DCR-034)."""

    NORMALIZE = "normalize"
    OBLIGATE = "obligate"
    ROUTE = "route"
    RECONSTRUCT = "reconstruct"
    CACHE = "cache"


class StageDisposition(str, Enum):
    """Closed stage outcome vocabulary (fail-closed outside ``pass``)."""

    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"
    ERROR = "error"
    EMPTY_SURFACE = "empty_surface"
    BRIDGE_ONLY = "bridge_only"
    IMPORT_FAILURE = "import_failure"
    PARTIAL = "partial"

    @property
    def is_pass(self) -> bool:
        return self is StageDisposition.PASS

    @property
    def blocks(self) -> bool:
        return self.value in _BLOCKING_DISPOSITIONS or not self.is_pass


class SurfaceKind(str, Enum):
    """How a stage surface was obtained."""

    PRODUCTION = "production"
    EMPTY = "empty"
    BRIDGE_ONLY = "bridge_only"
    FIXTURE = "fixture"
    IMPORT = "import"
    UNKNOWN = "unknown"


class GateDisposition(str, Enum):
    """Terminal gate outcome for one decision consultation."""

    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(
    value: Any,
    field_name: str,
    *,
    required: bool = True,
    maximum: int = _MAX_FIELD_BYTES,
) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise IRLogicApplicationError(
            f"{field_name} must be a string",
            reason_code="invalid_field_type",
            details={"field": field_name},
        )
    if required and not text:
        raise IRLogicApplicationError(
            f"{field_name} is required",
            reason_code="missing_required_field",
            details={"field": field_name},
        )
    if len(text.encode("utf-8")) > maximum:
        raise IRLogicApplicationError(
            f"{field_name} exceeds the {maximum}-byte limit",
            reason_code="field_too_large",
            details={"field": field_name},
        )
    if "\x00" in text:
        raise IRLogicApplicationError(
            f"{field_name} must not contain NUL",
            reason_code="invalid_field_value",
            details={"field": field_name},
        )
    return text


def _enum(value: Any, kind: type[Enum], field_name: str) -> Any:
    if isinstance(value, kind):
        return value
    try:
        return kind(str(getattr(value, "value", value)).strip())
    except (TypeError, ValueError) as exc:
        raise IRLogicApplicationError(
            f"{field_name} must be one of: "
            + ", ".join(item.value for item in kind),
            reason_code="invalid_enum",
            details={"field": field_name, "value": repr(value)},
        ) from exc


def _bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise IRLogicApplicationError(
            f"{field_name} must be a boolean",
            reason_code="invalid_field_type",
            details={"field": field_name},
        )
    return value


def _reason_codes(values: Sequence[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(
        values, Sequence
    ):
        raise IRLogicApplicationError(
            "reason_codes must be a sequence of strings",
            reason_code="invalid_field_type",
        )
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _text(item, "reason_code", required=True, maximum=256)
        if text not in seen:
            seen.add(text)
            cleaned.append(text)
    if len(cleaned) > _MAX_REASON_CODES:
        raise IRLogicApplicationError(
            "reason_codes exceeds bound",
            reason_code="bounds_exceeded",
        )
    return tuple(cleaned)


def _stage_ids(values: Sequence[str] | Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise IRLogicApplicationError(
            "stage set must be a sequence of stage ids",
            reason_code="invalid_field_type",
        )
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _text(item, "stage_id", required=True, maximum=128)
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    return tuple(ordered)


def _default_workspace() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parents[5], here.parents[4], Path.cwd()):
        marker = (
            candidate / "config" / "deterministic_contract_repair_services.json"
        )
        if marker.is_file():
            return candidate
        forest = (
            candidate
            / "data/agent_supervisor/deterministic_contract_repair/forest.json"
        )
        if forest.is_file():
            return candidate
    return Path.cwd()


def _resolve_relative(root: Path, relative: str) -> Path:
    return root.joinpath(*PurePosixPath(relative).parts)


# ---------------------------------------------------------------------------
# Stage observation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LogicStageObservation:
    """One observed outcome for a mandatory or attempted logic stage."""

    stage: LogicStageId | str
    disposition: StageDisposition | str
    surface_kind: SurfaceKind | str = SurfaceKind.PRODUCTION
    reason_codes: tuple[str, ...] = ()
    detail: str = ""
    family: str = ""
    module_origin: str = ""
    exception_swallowed: bool = False
    grants_execution_claim: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stage",
            _enum(self.stage, LogicStageId, "stage"),
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, StageDisposition, "disposition"),
        )
        object.__setattr__(
            self,
            "surface_kind",
            _enum(self.surface_kind, SurfaceKind, "surface_kind"),
        )
        object.__setattr__(
            self, "reason_codes", _reason_codes(self.reason_codes)
        )
        object.__setattr__(
            self,
            "detail",
            _text(self.detail, "detail", required=False),
        )
        object.__setattr__(
            self,
            "family",
            _text(self.family, "family", required=False),
        )
        object.__setattr__(
            self,
            "module_origin",
            _text(self.module_origin, "module_origin", required=False),
        )
        object.__setattr__(
            self,
            "exception_swallowed",
            _bool(self.exception_swallowed, "exception_swallowed"),
        )
        object.__setattr__(
            self,
            "grants_execution_claim",
            _bool(self.grants_execution_claim, "grants_execution_claim"),
        )
        # Normalize contradictory surface/disposition pairs fail-closed.
        disposition = self.disposition
        surface = self.surface_kind
        if surface is SurfaceKind.EMPTY and disposition is StageDisposition.PASS:
            object.__setattr__(self, "disposition", StageDisposition.EMPTY_SURFACE)
        if (
            surface is SurfaceKind.BRIDGE_ONLY
            and disposition is StageDisposition.PASS
        ):
            object.__setattr__(self, "disposition", StageDisposition.BRIDGE_ONLY)
        if self.exception_swallowed:
            object.__setattr__(self, "disposition", StageDisposition.ERROR)

    @property
    def stage_id(self) -> str:
        return (
            self.stage.value
            if isinstance(self.stage, LogicStageId)
            else str(self.stage)
        )

    @property
    def disposition_value(self) -> str:
        return (
            self.disposition.value
            if isinstance(self.disposition, StageDisposition)
            else str(self.disposition)
        )

    @property
    def is_pass(self) -> bool:
        return self.disposition is StageDisposition.PASS

    @property
    def is_blocking(self) -> bool:
        return not self.is_pass

    @property
    def is_unknown_row(self) -> bool:
        return self.disposition is StageDisposition.UNKNOWN

    @property
    def is_unsupported_row(self) -> bool:
        return self.disposition is StageDisposition.UNSUPPORTED

    @property
    def is_error_row(self) -> bool:
        return self.disposition in {
            StageDisposition.ERROR,
            StageDisposition.IMPORT_FAILURE,
            StageDisposition.FAIL,
        }

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": LOGIC_STAGE_OBSERVATION_SCHEMA,
            "stage": self.stage_id,
            "disposition": self.disposition_value,
            "surface_kind": (
                self.surface_kind.value
                if isinstance(self.surface_kind, SurfaceKind)
                else str(self.surface_kind)
            ),
            "reason_codes": list(self.reason_codes),
            "detail": self.detail,
            "family": self.family,
            "module_origin": self.module_origin,
            "exception_swallowed": self.exception_swallowed,
            "grants_execution_claim": self.grants_execution_claim,
        }

    def to_dict(self) -> dict[str, Any]:
        return dict(self._payload())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LogicStageObservation":
        if not isinstance(value, Mapping):
            raise IRLogicApplicationError(
                "stage observation must be an object",
                reason_code="invalid_field_type",
            )
        return cls(
            stage=value.get("stage", ""),
            disposition=value.get("disposition", StageDisposition.UNKNOWN),
            surface_kind=value.get("surface_kind", SurfaceKind.UNKNOWN),
            reason_codes=tuple(value.get("reason_codes") or ()),
            detail=str(value.get("detail") or ""),
            family=str(value.get("family") or ""),
            module_origin=str(value.get("module_origin") or ""),
            exception_swallowed=bool(value.get("exception_swallowed", False)),
            grants_execution_claim=bool(
                value.get("grants_execution_claim", False)
            ),
        )


def stage_observation(
    stage: LogicStageId | str,
    disposition: StageDisposition | str,
    *,
    surface_kind: SurfaceKind | str = SurfaceKind.PRODUCTION,
    reason_codes: Sequence[str] = (),
    detail: str = "",
    family: str = "",
    module_origin: str = "",
    exception_swallowed: bool = False,
    grants_execution_claim: bool = False,
) -> LogicStageObservation:
    """Construct a validated stage observation."""

    return LogicStageObservation(
        stage=stage,
        disposition=disposition,
        surface_kind=surface_kind,
        reason_codes=tuple(reason_codes),
        detail=detail,
        family=family,
        module_origin=module_origin,
        exception_swallowed=exception_swallowed,
        grants_execution_claim=grants_execution_claim,
    )


def all_stages_passed(
    *,
    reason_codes: Sequence[str] = ("stage_passed",),
    surface_kind: SurfaceKind | str = SurfaceKind.PRODUCTION,
) -> tuple[LogicStageObservation, ...]:
    """Return a complete pass observation set for every required stage."""

    return tuple(
        stage_observation(
            stage,
            StageDisposition.PASS,
            surface_kind=surface_kind,
            reason_codes=reason_codes,
        )
        for stage in REQUIRED_LOGIC_STAGES
    )


# ---------------------------------------------------------------------------
# Gate / application result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IRApplicationResult:
    """Content-addressed receipt of one IR logic application gate check.

    Interface: ``IRApplicationResult@1``.
    """

    INTERFACE: ClassVar[str] = IR_APPLICATION_RESULT_INTERFACE

    decision: LogicDecisionKind | str
    disposition: GateDisposition | str
    required_stages: tuple[str, ...]
    ran_stages: tuple[str, ...]
    pass_stages: tuple[str, ...]
    observations: tuple[LogicStageObservation, ...] = ()
    unknown_rows: tuple[dict[str, Any], ...] = ()
    unsupported_rows: tuple[dict[str, Any], ...] = ()
    error_rows: tuple[dict[str, Any], ...] = ()
    reason_codes: tuple[str, ...] = ()
    detail: str = ""
    gate_passed: bool = False
    execution_granted: bool = False
    no_false_grant: bool = True
    model_calls: int = 0
    evidence_term: str = UNKNOWN_GATE_EVIDENCE_TERM
    policy_version: str = LOGIC_GATE_VERSION
    task_id: str = DCR_TASK_ID

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "decision",
            _enum(self.decision, LogicDecisionKind, "decision"),
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, GateDisposition, "disposition"),
        )
        object.__setattr__(
            self, "required_stages", _stage_ids(self.required_stages)
        )
        object.__setattr__(self, "ran_stages", _stage_ids(self.ran_stages))
        object.__setattr__(self, "pass_stages", _stage_ids(self.pass_stages))
        if not isinstance(self.observations, tuple):
            object.__setattr__(
                self, "observations", tuple(self.observations or ())
            )
        for obs in self.observations:
            if not isinstance(obs, LogicStageObservation):
                raise IRLogicApplicationError(
                    "observations must contain LogicStageObservation values",
                    reason_code="invalid_field_type",
                )
        if len(self.observations) > _MAX_ROWS:
            raise IRLogicApplicationError(
                "observations exceed bound",
                reason_code="bounds_exceeded",
            )
        object.__setattr__(
            self, "unknown_rows", tuple(dict(row) for row in self.unknown_rows)
        )
        object.__setattr__(
            self,
            "unsupported_rows",
            tuple(dict(row) for row in self.unsupported_rows),
        )
        object.__setattr__(
            self, "error_rows", tuple(dict(row) for row in self.error_rows)
        )
        object.__setattr__(
            self, "reason_codes", _reason_codes(self.reason_codes)
        )
        object.__setattr__(
            self, "detail", _text(self.detail, "detail", required=False)
        )
        object.__setattr__(
            self, "gate_passed", _bool(self.gate_passed, "gate_passed")
        )
        object.__setattr__(
            self,
            "execution_granted",
            _bool(self.execution_granted, "execution_granted"),
        )
        object.__setattr__(
            self, "no_false_grant", _bool(self.no_false_grant, "no_false_grant")
        )
        if not isinstance(self.model_calls, int) or isinstance(
            self.model_calls, bool
        ):
            raise IRLogicApplicationError(
                "model_calls must be an integer",
                reason_code="invalid_field_type",
            )
        if self.model_calls != 0:
            raise IRLogicApplicationError(
                "model_calls must be zero for IR logic application",
                reason_code="model_calls_forbidden",
            )
        object.__setattr__(
            self,
            "evidence_term",
            _text(self.evidence_term, "evidence_term") or UNKNOWN_GATE_EVIDENCE_TERM,
        )
        object.__setattr__(
            self,
            "policy_version",
            _text(self.policy_version, "policy_version") or LOGIC_GATE_VERSION,
        )
        object.__setattr__(
            self, "task_id", _text(self.task_id, "task_id") or DCR_TASK_ID
        )
        # Integrity: execution may never be granted without a full pass.
        if self.execution_granted and not self.gate_passed:
            raise IRLogicApplicationError(
                "execution cannot be granted when the gate did not pass",
                reason_code=IR_APPLICATION_FAILED,
            )
        if self.execution_granted and not self.no_false_grant:
            raise IRLogicApplicationError(
                "execution cannot be granted when no_false_grant is false",
                reason_code=IR_APPLICATION_FAILED,
            )
        if self.gate_passed and set(self.pass_stages) != set(
            self.required_stages
        ):
            raise IRLogicApplicationError(
                "gate_passed requires pass_stages == required_stages",
                reason_code=IR_APPLICATION_FAILED,
            )
        if self.gate_passed and (
            self.unknown_rows or self.unsupported_rows or self.error_rows
        ):
            raise IRLogicApplicationError(
                "gate_passed forbids unknown/unsupported/error rows",
                reason_code=IR_APPLICATION_FAILED,
            )

    @property
    def interface(self) -> str:
        return IR_APPLICATION_RESULT_INTERFACE

    @property
    def schema(self) -> str:
        return IR_APPLICATION_RESULT_SCHEMA

    @property
    def decision_value(self) -> str:
        return (
            self.decision.value
            if isinstance(self.decision, LogicDecisionKind)
            else str(self.decision)
        )

    @property
    def disposition_value(self) -> str:
        return (
            self.disposition.value
            if isinstance(self.disposition, GateDisposition)
            else str(self.disposition)
        )

    @property
    def result_id(self) -> str:
        return content_identity(self._identity_payload())

    @property
    def canonical_digest(self) -> str:
        digest = (
            "sha256:"
            + __import__("hashlib")
            .sha256(self.to_canonical_bytes())
            .hexdigest()
        )
        return digest

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": IR_APPLICATION_RESULT_SCHEMA,
            "interface": IR_APPLICATION_RESULT_INTERFACE,
            "evidence_term": self.evidence_term,
            "task_id": self.task_id,
            "policy_version": self.policy_version,
            "contract_version": CONTRACT_VERSION,
            "decision": self.decision_value,
            "disposition": self.disposition_value,
            "required_stages": list(self.required_stages),
            "ran_stages": list(self.ran_stages),
            "pass_stages": list(self.pass_stages),
            "observations": [obs.to_dict() for obs in self.observations],
            "unknown_rows": list(self.unknown_rows),
            "unsupported_rows": list(self.unsupported_rows),
            "error_rows": list(self.error_rows),
            "reason_codes": list(self.reason_codes),
            "detail": self.detail,
            "gate_passed": self.gate_passed,
            "execution_granted": self.execution_granted,
            "no_false_grant": self.no_false_grant,
            "model_calls": self.model_calls,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        payload["result_id"] = self.result_id
        payload["canonical_digest"] = self.canonical_digest
        return payload

    def to_canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self._identity_payload())

    def verifies_identity(self) -> bool:
        return self.result_id == content_identity(self._identity_payload())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "IRApplicationResult":
        if not isinstance(value, Mapping):
            raise IRLogicApplicationError(
                "IR application result must be an object",
                reason_code="invalid_field_type",
            )
        observations = tuple(
            item
            if isinstance(item, LogicStageObservation)
            else LogicStageObservation.from_dict(item)
            for item in (value.get("observations") or ())
        )
        result = cls(
            decision=value.get("decision", LogicDecisionKind.DIAGNOSE),
            disposition=value.get("disposition", GateDisposition.FAILED),
            required_stages=tuple(value.get("required_stages") or ()),
            ran_stages=tuple(value.get("ran_stages") or ()),
            pass_stages=tuple(value.get("pass_stages") or ()),
            observations=observations,
            unknown_rows=tuple(value.get("unknown_rows") or ()),
            unsupported_rows=tuple(value.get("unsupported_rows") or ()),
            error_rows=tuple(value.get("error_rows") or ()),
            reason_codes=tuple(value.get("reason_codes") or ()),
            detail=str(value.get("detail") or ""),
            gate_passed=bool(value.get("gate_passed", False)),
            execution_granted=bool(value.get("execution_granted", False)),
            no_false_grant=bool(value.get("no_false_grant", True)),
            model_calls=int(value.get("model_calls", 0) or 0),
            evidence_term=str(
                value.get("evidence_term") or UNKNOWN_GATE_EVIDENCE_TERM
            ),
            policy_version=str(
                value.get("policy_version") or LOGIC_GATE_VERSION
            ),
            task_id=str(value.get("task_id") or DCR_TASK_ID),
        )
        claimed = str(value.get("result_id") or "")
        if claimed and claimed != result.result_id:
            raise IRLogicApplicationError(
                "result_id does not match content identity",
                reason_code="identity_mismatch",
            )
        return result


@dataclass(frozen=True, slots=True)
class RequiredLogicStageGate:
    """Fixed-order mandatory logic-stage gate (``RequiredLogicStageGate@1``).

    The gate never swallows required-path exceptions, never treats partial
    stage coverage as pass, never promotes bridge-only availability, and never
    accepts default-true safety claims.
    """

    INTERFACE: ClassVar[str] = REQUIRED_LOGIC_STAGE_GATE_INTERFACE

    required_stages: tuple[str, ...] = REQUIRED_LOGIC_STAGES
    policy_version: str = LOGIC_GATE_VERSION

    def __post_init__(self) -> None:
        stages = _stage_ids(self.required_stages)
        if not stages:
            raise IRLogicApplicationError(
                "required_stages must not be empty",
                reason_code="empty_required_stages",
            )
        # Policy requires the exact DCR-030..034 set; callers may not shrink it.
        missing = [
            stage for stage in REQUIRED_LOGIC_STAGES if stage not in stages
        ]
        if missing:
            raise IRLogicApplicationError(
                "required_stages is missing mandatory pipeline stages: "
                + ", ".join(missing),
                reason_code="incomplete_required_stages",
                details={"missing": missing},
            )
        extra = [stage for stage in stages if stage not in REQUIRED_LOGIC_STAGES]
        if extra:
            raise IRLogicApplicationError(
                "required_stages contains unknown stages: " + ", ".join(extra),
                reason_code="unknown_required_stage",
                details={"extra": extra},
            )
        # Preserve policy order.
        object.__setattr__(self, "required_stages", tuple(REQUIRED_LOGIC_STAGES))
        object.__setattr__(
            self,
            "policy_version",
            _text(self.policy_version, "policy_version") or LOGIC_GATE_VERSION,
        )

    @property
    def interface(self) -> str:
        return REQUIRED_LOGIC_STAGE_GATE_INTERFACE

    def evaluate(
        self,
        decision: LogicDecisionKind | str,
        observations: Sequence[LogicStageObservation | Mapping[str, Any]],
        *,
        claim_execution: bool = False,
    ) -> IRApplicationResult:
        """Evaluate stage observations for one policy decision.

        Fail-closed: any empty surface, skip, unsupported semantics, import
        failure, bridge-only projection, unknown, error, partial, swallowed
        exception, or missing required stage blocks pass and execution grant.
        """

        decision_kind = _enum(decision, LogicDecisionKind, "decision")
        parsed = self._parse_observations(observations)
        by_stage = self._index_by_stage(parsed)

        ran: list[str] = []
        passed: list[str] = []
        unknown_rows: list[dict[str, Any]] = []
        unsupported_rows: list[dict[str, Any]] = []
        error_rows: list[dict[str, Any]] = []
        reasons: list[str] = []
        details: list[str] = []

        for stage in self.required_stages:
            obs = by_stage.get(stage)
            if obs is None:
                reasons.append(f"missing_stage:{stage}")
                error_rows.append(
                    {
                        "stage": stage,
                        "disposition": StageDisposition.FAIL.value,
                        "reason": "required_stage_not_ran",
                    }
                )
                details.append(f"required stage {stage!r} did not run")
                continue

            ran.append(stage)
            row = {
                "stage": stage,
                "disposition": obs.disposition_value,
                "surface_kind": (
                    obs.surface_kind.value
                    if isinstance(obs.surface_kind, SurfaceKind)
                    else str(obs.surface_kind)
                ),
                "reason_codes": list(obs.reason_codes),
                "family": obs.family,
                "module_origin": obs.module_origin,
            }

            if obs.exception_swallowed:
                reasons.append(f"exception_swallowed:{stage}")
                error_rows.append({**row, "reason": "exception_swallowed"})
                details.append(
                    f"stage {stage!r} swallowed an exception on the required path"
                )
                continue

            if obs.surface_kind is SurfaceKind.EMPTY or (
                obs.disposition is StageDisposition.EMPTY_SURFACE
            ):
                reasons.append(f"empty_surface:{stage}")
                error_rows.append({**row, "reason": "empty_surface"})
                details.append(f"stage {stage!r} has an empty surface")
                continue

            if obs.surface_kind is SurfaceKind.BRIDGE_ONLY or (
                obs.disposition is StageDisposition.BRIDGE_ONLY
            ):
                reasons.append(f"bridge_only:{stage}")
                error_rows.append({**row, "reason": "ui_bridge_only_projection"})
                details.append(
                    f"stage {stage!r} is a UI bridge-only projection"
                )
                continue

            if obs.disposition is StageDisposition.SKIP:
                reasons.append(f"skipped_stage:{stage}")
                error_rows.append({**row, "reason": "skipped_stage"})
                details.append(f"stage {stage!r} was skipped")
                continue

            if obs.disposition is StageDisposition.UNSUPPORTED:
                reasons.append(f"unsupported_semantics:{stage}")
                unsupported_rows.append(
                    {**row, "reason": "unsupported_semantics"}
                )
                details.append(f"stage {stage!r} has unsupported semantics")
                continue

            if obs.disposition is StageDisposition.UNKNOWN:
                reasons.append(f"unknown:{stage}")
                unknown_rows.append({**row, "reason": "unknown_outcome"})
                details.append(f"stage {stage!r} outcome is unknown")
                continue

            if obs.disposition is StageDisposition.IMPORT_FAILURE:
                reasons.append(f"import_failure:{stage}")
                error_rows.append({**row, "reason": "import_failure"})
                details.append(f"stage {stage!r} import failed")
                continue

            if obs.disposition is StageDisposition.ERROR:
                reasons.append(f"error:{stage}")
                error_rows.append({**row, "reason": "stage_error"})
                details.append(f"stage {stage!r} errored")
                continue

            if obs.disposition is StageDisposition.PARTIAL:
                reasons.append(f"partial:{stage}")
                error_rows.append({**row, "reason": "partial_stage"})
                details.append(f"stage {stage!r} is only partial")
                continue

            if obs.disposition is StageDisposition.FAIL or not obs.is_pass:
                reasons.append(f"fail:{stage}")
                error_rows.append({**row, "reason": "stage_failed"})
                details.append(f"stage {stage!r} failed")
                continue

            # Only an explicit production pass reaches here.
            if obs.surface_kind not in {
                SurfaceKind.PRODUCTION,
                SurfaceKind.FIXTURE,
            }:
                # Fixture may exercise adapters but still cannot grant
                # production execution; track as non-pass for execution.
                if obs.surface_kind is SurfaceKind.UNKNOWN:
                    reasons.append(f"unknown_surface:{stage}")
                    unknown_rows.append({**row, "reason": "unknown_surface"})
                    details.append(f"stage {stage!r} has unknown surface")
                    continue

            passed.append(stage)

        # Extra non-required observations still contribute fail-closed rows.
        for obs in parsed:
            if obs.stage_id in self.required_stages:
                continue
            if obs.is_unknown_row:
                unknown_rows.append(
                    {
                        "stage": obs.stage_id,
                        "disposition": obs.disposition_value,
                        "reason": "unknown_extra_stage",
                    }
                )
            elif obs.is_unsupported_row:
                unsupported_rows.append(
                    {
                        "stage": obs.stage_id,
                        "disposition": obs.disposition_value,
                        "reason": "unsupported_extra_stage",
                    }
                )
            elif obs.is_error_row or obs.exception_swallowed:
                error_rows.append(
                    {
                        "stage": obs.stage_id,
                        "disposition": obs.disposition_value,
                        "reason": "error_extra_stage",
                    }
                )

        required_set = set(self.required_stages)
        pass_set = set(passed)
        ran_set = set(ran)
        full_pass = (
            pass_set == required_set
            and ran_set == required_set
            and not unknown_rows
            and not unsupported_rows
            and not error_rows
        )

        # Spurious execution claims from stages are never authority.
        spurious_grant = any(obs.grants_execution_claim for obs in parsed)
        if spurious_grant and not full_pass:
            reasons.append("spurious_execution_claim")
            details.append(
                "stage claimed execution grant without a full mandatory pass"
            )

        if not full_pass and IR_APPLICATION_FAILED not in reasons:
            reasons.insert(0, IR_APPLICATION_FAILED)

        gate_passed = full_pass
        execution_granted = False
        if gate_passed and decision_kind.may_grant_execution:
            # Explicit claim_execution is required for apply/complete grant;
            # the gate never default-true grants execution.
            if claim_execution:
                execution_granted = True
            else:
                # Decision may pass the gate without automatically granting.
                execution_granted = False
        elif claim_execution and not gate_passed:
            reasons.append("execution_claim_rejected")
            details.append(
                "execution claim rejected because the mandatory stage gate failed"
            )

        if gate_passed:
            disposition = GateDisposition.PASSED
            if "mandatory_stages_passed" not in reasons:
                reasons.append("mandatory_stages_passed")
            detail = (
                f"decision {decision_kind.value} passed mandatory logic stages"
            )
        elif unknown_rows or unsupported_rows:
            disposition = GateDisposition.BLOCKED
            detail = (
                f"decision {decision_kind.value} blocked: "
                + "; ".join(details[:8])
            )
        else:
            disposition = GateDisposition.FAILED
            detail = (
                f"decision {decision_kind.value} failed: "
                + "; ".join(details[:8])
            )

        # no_false_grant is true only when we did not grant without a full pass.
        no_false_grant = not execution_granted or gate_passed

        return IRApplicationResult(
            decision=decision_kind,
            disposition=disposition,
            required_stages=self.required_stages,
            ran_stages=tuple(ran),
            pass_stages=tuple(passed),
            observations=tuple(parsed),
            unknown_rows=tuple(unknown_rows),
            unsupported_rows=tuple(unsupported_rows),
            error_rows=tuple(error_rows),
            reason_codes=tuple(reasons),
            detail=detail,
            gate_passed=gate_passed,
            execution_granted=execution_granted,
            no_false_grant=no_false_grant,
            model_calls=0,
            evidence_term=UNKNOWN_GATE_EVIDENCE_TERM,
            policy_version=self.policy_version,
            task_id=DCR_TASK_ID,
        )

    def _parse_observations(
        self,
        observations: Sequence[LogicStageObservation | Mapping[str, Any]],
    ) -> list[LogicStageObservation]:
        if isinstance(observations, (str, bytes, bytearray)) or not isinstance(
            observations, Sequence
        ):
            raise IRLogicApplicationError(
                "observations must be a sequence",
                reason_code="invalid_field_type",
            )
        if len(observations) > _MAX_ROWS:
            raise IRLogicApplicationError(
                "observations exceed bound",
                reason_code="bounds_exceeded",
            )
        parsed: list[LogicStageObservation] = []
        for item in observations:
            if isinstance(item, LogicStageObservation):
                parsed.append(item)
            elif isinstance(item, Mapping):
                parsed.append(LogicStageObservation.from_dict(item))
            else:
                raise IRLogicApplicationError(
                    "each observation must be a LogicStageObservation or mapping",
                    reason_code="invalid_field_type",
                )
        return parsed

    def _index_by_stage(
        self, observations: Sequence[LogicStageObservation]
    ) -> dict[str, LogicStageObservation]:
        """Index by stage; conflicting duplicates fail closed to error."""

        index: dict[str, LogicStageObservation] = {}
        for obs in observations:
            existing = index.get(obs.stage_id)
            if existing is None:
                index[obs.stage_id] = obs
                continue
            # Prefer the more severe disposition (never upgrade to pass).
            if existing.is_pass and not obs.is_pass:
                index[obs.stage_id] = obs
            elif existing.is_pass and obs.is_pass:
                continue
            elif not existing.is_pass and obs.is_pass:
                continue
            else:
                # Two non-pass outcomes: keep the first but mark conflict.
                index[obs.stage_id] = LogicStageObservation(
                    stage=obs.stage,
                    disposition=StageDisposition.ERROR,
                    surface_kind=obs.surface_kind,
                    reason_codes=tuple(
                        dict.fromkeys(
                            list(existing.reason_codes)
                            + list(obs.reason_codes)
                            + ["duplicate_stage_conflict"]
                        )
                    ),
                    detail="conflicting observations for the same stage",
                    family=obs.family or existing.family,
                    module_origin=obs.module_origin or existing.module_origin,
                    exception_swallowed=(
                        existing.exception_swallowed or obs.exception_swallowed
                    ),
                    grants_execution_claim=False,
                )
        return index

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REQUIRED_LOGIC_STAGE_GATE_SCHEMA,
            "interface": REQUIRED_LOGIC_STAGE_GATE_INTERFACE,
            "required_stages": list(self.required_stages),
            "policy_version": self.policy_version,
            "policy_decision_kinds": list(POLICY_DECISION_KINDS),
            "execution_decisions": sorted(_EXECUTION_DECISIONS),
            "blocking_dispositions": sorted(_BLOCKING_DISPOSITIONS),
            "evidence_term": UNKNOWN_GATE_EVIDENCE_TERM,
            "task_id": DCR_TASK_ID,
            "contract_version": CONTRACT_VERSION,
        }


def apply_ir_logic(
    decision: LogicDecisionKind | str,
    observations: Sequence[LogicStageObservation | Mapping[str, Any]],
    *,
    claim_execution: bool = False,
    gate: RequiredLogicStageGate | None = None,
) -> IRApplicationResult:
    """Apply the mandatory IR logic stage gate for one policy decision.

    This is the sole public application entry point.  It never grants
    execution on empty surfaces, skipped stages, unsupported semantics,
    import failures, or UI bridge-only projections.
    """

    owner = gate or RequiredLogicStageGate()
    return owner.evaluate(
        decision,
        observations,
        claim_execution=claim_execution,
    )


def require_gate_pass(
    result: IRApplicationResult,
    *,
    require_execution: bool = False,
) -> IRApplicationResult:
    """Raise when a gate result is not a clean pass (optionally with grant)."""

    if not isinstance(result, IRApplicationResult):
        raise IRLogicApplicationError(
            "result must be an IRApplicationResult",
            reason_code="invalid_field_type",
        )
    if not result.gate_passed:
        raise IRLogicApplicationError(
            result.detail or "mandatory logic stage gate failed",
            reason_code=IR_APPLICATION_FAILED,
            details={
                "decision": result.decision_value,
                "reason_codes": list(result.reason_codes),
                "ran_stages": list(result.ran_stages),
                "pass_stages": list(result.pass_stages),
            },
        )
    if require_execution and not result.execution_granted:
        raise IRLogicApplicationError(
            "execution was not granted by the mandatory logic stage gate",
            reason_code=IR_APPLICATION_FAILED,
            details={"decision": result.decision_value},
        )
    return result


# ---------------------------------------------------------------------------
# Artifact materialization
# ---------------------------------------------------------------------------


def materialize_logic_gate_artifact(
    *,
    results: Sequence[IRApplicationResult] = (),
    notes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build the DCR-035 generated artifact projection."""

    note_list = list(
        notes
        or (
            "DCR-035 mandatory logic-stage gate.",
            "Empty surfaces, skipped stages, unsupported semantics, import "
            "failures, and UI bridge-only projections cannot pass or grant "
            "execution.",
            "Unknown and error rows block pass; no-false-grant is sealed.",
        )
    )
    gate = RequiredLogicStageGate()
    sealed_results = [item.to_dict() for item in results]
    return {
        "schema": LOGIC_GATE_ARTIFACT_SCHEMA,
        "interface": REQUIRED_LOGIC_STAGE_GATE_INTERFACE,
        "result_interface": IR_APPLICATION_RESULT_INTERFACE,
        "evidence_term": UNKNOWN_GATE_EVIDENCE_TERM,
        "task_id": DCR_TASK_ID,
        "contract_version": CONTRACT_VERSION,
        "policy_version": LOGIC_GATE_VERSION,
        "required_stages": list(REQUIRED_LOGIC_STAGES),
        "policy_decision_kinds": list(POLICY_DECISION_KINDS),
        "gate": gate.to_dict(),
        "results": sealed_results,
        "result_count": len(sealed_results),
        "acceptance": {
            "empty_surfaces_cannot_pass_or_grant": True,
            "skipped_stages_cannot_pass_or_grant": True,
            "unsupported_semantics_cannot_pass_or_grant": True,
            "import_failures_cannot_pass_or_grant": True,
            "ui_bridge_only_projections_cannot_pass_or_grant": True,
            "unknown_and_error_rows_block_pass": True,
            "no_false_grant": True,
            "partial_stage_pass_forbidden": True,
            "exception_swallowing_forbidden": True,
            "default_true_safety_claims_forbidden": True,
        },
        "notes": note_list,
    }


def write_logic_gate(
    destination: str | Path | None = None,
    *,
    artifact: Mapping[str, Any] | None = None,
    results: Sequence[IRApplicationResult] = (),
    repo_root: Path | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Path:
    """Atomically write the logic-gate artifact as canonical JSON."""

    root = Path(repo_root) if repo_root is not None else _default_workspace()
    payload = (
        dict(artifact)
        if artifact is not None
        else materialize_logic_gate_artifact(results=results)
    )
    data = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if len(data) > max_bytes:
        raise IRLogicApplicationError(
            f"artifact exceeds {max_bytes} bytes",
            reason_code="bounds_exceeded",
            details={"byte_length": len(data)},
        )
    if destination is None:
        path = _resolve_relative(root, DCR_ARTIFACT_PATH)
    else:
        path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return path


def load_logic_gate(
    source: str | Path | None = None,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Load and lightly revalidate a logic-gate artifact."""

    root = Path(repo_root) if repo_root is not None else _default_workspace()
    if source is None:
        path = _resolve_relative(root, DCR_ARTIFACT_PATH)
    else:
        path = Path(source)
    if not path.is_file():
        raise IRLogicApplicationError(
            f"logic-gate artifact missing: {path}",
            reason_code="artifact_missing",
            details={"path": str(path)},
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IRLogicApplicationError(
            f"logic-gate artifact is not valid JSON: {path}",
            reason_code="invalid_json",
            details={"path": str(path)},
        ) from exc
    if not isinstance(payload, dict):
        raise IRLogicApplicationError(
            "logic-gate artifact must be a JSON object",
            reason_code="invalid_json_root",
        )
    if payload.get("schema") != LOGIC_GATE_ARTIFACT_SCHEMA:
        raise IRLogicApplicationError(
            "logic-gate artifact schema mismatch",
            reason_code="schema_mismatch",
        )
    if payload.get("interface") != REQUIRED_LOGIC_STAGE_GATE_INTERFACE:
        raise IRLogicApplicationError(
            "logic-gate artifact interface mismatch",
            reason_code="interface_mismatch",
        )
    if payload.get("evidence_term") != UNKNOWN_GATE_EVIDENCE_TERM:
        raise IRLogicApplicationError(
            "logic-gate artifact evidence term mismatch",
            reason_code="evidence_term_mismatch",
        )
    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, Mapping) or acceptance.get(
        "no_false_grant"
    ) is not True:
        raise IRLogicApplicationError(
            "logic-gate artifact must seal no_false_grant",
            reason_code="acceptance_missing",
        )
    return payload


def ensure_logic_gate_artifact(
    *,
    repo_root: Path | None = None,
    force: bool = False,
) -> Path:
    """Ensure the declared DCR-035 artifact exists without unnecessary rewrites."""

    root = Path(repo_root) if repo_root is not None else _default_workspace()
    out = _resolve_relative(root, DCR_ARTIFACT_PATH)
    if out.is_file() and not force:
        try:
            load_logic_gate(out)
            return out
        except IRLogicApplicationError:
            pass
    return write_logic_gate(out, repo_root=root)


__all__ = [
    "CONTRACT_VERSION",
    "DCR_ARTIFACT_PATH",
    "DCR_TASK_ID",
    "DEFAULT_MAX_BYTES",
    "GateDisposition",
    "IR_APPLICATION_FAILED",
    "IR_APPLICATION_RESULT_INTERFACE",
    "IR_APPLICATION_RESULT_SCHEMA",
    "IRApplicationResult",
    "IRLogicApplicationError",
    "LOGIC_GATE_ARTIFACT_SCHEMA",
    "LOGIC_GATE_VERSION",
    "LOGIC_STAGE_OBSERVATION_SCHEMA",
    "LogicDecisionKind",
    "LogicStageId",
    "LogicStageObservation",
    "POLICY_DECISION_KINDS",
    "REQUIRED_LOGIC_STAGES",
    "REQUIRED_LOGIC_STAGE_GATE_INTERFACE",
    "REQUIRED_LOGIC_STAGE_GATE_SCHEMA",
    "RequiredLogicStageGate",
    "StageDisposition",
    "SurfaceKind",
    "UNKNOWN_GATE_EVIDENCE_TERM",
    "all_stages_passed",
    "apply_ir_logic",
    "ensure_logic_gate_artifact",
    "load_logic_gate",
    "materialize_logic_gate_artifact",
    "require_gate_pass",
    "stage_observation",
    "write_logic_gate",
]
