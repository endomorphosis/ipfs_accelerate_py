"""Production default planner factory (WPD-011).

Interface: ``DefaultPlannerFactory@1``

Binds the live formal planning stack so CLI/API defaults never ship with
empty injected slots:

* :class:`~.formal_plan_compiler.FormalPlanCompiler`
* :class:`~.formal_plan_validator.FormalPlanValidator`
* :class:`~.formal_replanner.FormalReplanner`
* :class:`~.adaptive_planner.AdaptivePlanner`
* optional :class:`~.proof_carrying_planner.ProofCarryingPlanner` handle

Fail-closed rules:

* Core compiler / validator / replanner / adaptive planner are always real
  instances that share the same compiler and validator.
* Optional provers are probed without importing provider modules or claiming
  proof success from package presence alone.
* Required optional provers that are absent yield disposition
  ``defer_capability`` — never silent success and never a forged proof-
  carrying planner that pretends backends are available.
* This module does not import ``todo_daemon`` (package DAG).  The disposition
  string matches :class:`ImplementationDisposition.DEFER_CAPABILITY`.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ..proof.formal_verification_contracts import content_identity
from .adaptive_planner import AdaptivePlanner
from .formal_plan_compiler import FormalPlanCompiler
from .formal_plan_validator import FormalPlanValidator, ValidationBounds
from .formal_replanner import FormalReplanner, ReplanLimits
from .proof_carrying_planner import (
    ProofCarryingPlanner,
    ProofCarryingPlannerConfig,
    WorkflowAdapters,
    WorkflowConfigurationError,
)


# ---------------------------------------------------------------------------
# Interface identity
# ---------------------------------------------------------------------------

DEFAULT_PLANNER_FACTORY_INTERFACE: Final[str] = "DefaultPlannerFactory@1"
DEFAULT_PLANNER_HANDLES_INTERFACE: Final[str] = "DefaultPlannerHandles@1"
DEFAULT_PLANNER_FACTORY_VERSION: Final[int] = 1

DEFAULT_PLANNER_HANDLES_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/default-planner-handles@1"
)
DEFAULT_PLANNER_CAPABILITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/default-planner-capability@1"
)
OPTIONAL_PROVER_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/optional-prover-record@1"
)

# Objective-heap evidence key for the WPD factories goal packet.
DEFAULT_PLANNER_FACTORY_EVIDENCE: Final[str] = "wpd/default-planner-factory@1"

# Domain-agnostic IR logic (Intent/Legal/Security/UI + AST/KG/vector)
IR_LOGIC_CAPABILITY_EVIDENCE: Final[str] = "wpd/ir-logic-application@1"

# Wire spelling shared with ImplementationDisposition.DEFER_CAPABILITY.
DEFER_CAPABILITY_DISPOSITION: Final[str] = "defer_capability"

ExecutableFinder = Callable[[str], str | None]
OptionalProverProbe = Callable[[str], bool | Mapping[str, Any] | None]


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class OptionalProverId(str, Enum):
    """Closed optional prover / kernel identities the factory may probe."""

    LEAN = "lean"
    Z3 = "z3"
    CVC5 = "cvc5"
    COQ = "coq"
    HAMMER = "hammer"


class OptionalProverStatus(str, Enum):
    """Closed outcomes for one optional prover probe."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"
    NOT_PROBED = "not_probed"
    FAILED = "failed"


class PlannerStackDisposition(str, Enum):
    """Factory-level disposition for the bound planner stack.

    * ``ready`` — core compiler/validator/replanner/adaptive planner are bound
      and no *required* optional prover is missing.
    * ``defer_capability`` — a required optional backend is unavailable; the
      factory must not claim proof success or authorize silent completion.
    """

    READY = "ready"
    DEFER_CAPABILITY = DEFER_CAPABILITY_DISPOSITION


# Default optional provers probed for inventory (never treated as proof success).
DEFAULT_OPTIONAL_PROVERS: Final[tuple[OptionalProverId, ...]] = (
    OptionalProverId.LEAN,
    OptionalProverId.Z3,
    OptionalProverId.CVC5,
    OptionalProverId.COQ,
)

# Executable candidates for PATH lookup (validation PATH is sealed; probes
# report absence rather than inventing tools).
_EXECUTABLE_CANDIDATES: Final[Mapping[OptionalProverId, tuple[str, ...]]] = {
    OptionalProverId.LEAN: ("lean",),
    OptionalProverId.Z3: ("z3",),
    OptionalProverId.CVC5: ("cvc5",),
    OptionalProverId.COQ: ("coqc", "coqtop"),
    OptionalProverId.HAMMER: ("hammer",),
}

_MAX_REASON_CODES: Final[int] = 64
_MAX_TEXT: Final[int] = 4_096


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DefaultPlannerFactoryError(RuntimeError):
    """Fail-closed rejection for an unsafe or incomplete planner factory run."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "default_planner_factory_error",
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code or "default_planner_factory_error")


class DefaultPlannerCapabilityError(DefaultPlannerFactoryError):
    """A required optional prover/backend is unavailable."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "defer_capability",
        missing_provers: Sequence[str] = (),
    ) -> None:
        super().__init__(message, reason_code=reason_code)
        self.missing_provers = tuple(str(item) for item in missing_provers if str(item))


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptionalProverRecord:
    """Body-free probe result for one optional prover."""

    prover_id: OptionalProverId
    status: OptionalProverStatus
    required: bool = False
    executable: str = ""
    reason_code: str = ""

    def __post_init__(self) -> None:
        prover = self.prover_id
        if not isinstance(prover, OptionalProverId):
            prover = OptionalProverId(str(prover))
        object.__setattr__(self, "prover_id", prover)
        status = self.status
        if not isinstance(status, OptionalProverStatus):
            status = OptionalProverStatus(str(status))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "required", bool(self.required))
        executable = str(self.executable or "").strip()
        if len(executable.encode("utf-8")) > _MAX_TEXT:
            raise DefaultPlannerFactoryError(
                "executable path exceeds its byte bound",
                reason_code="bounds_exceeded",
            )
        object.__setattr__(self, "executable", executable)
        reason = str(self.reason_code or "").strip()
        if len(reason.encode("utf-8")) > _MAX_TEXT:
            raise DefaultPlannerFactoryError(
                "reason_code exceeds its byte bound",
                reason_code="bounds_exceeded",
            )
        object.__setattr__(self, "reason_code", reason)

    @property
    def available(self) -> bool:
        return self.status is OptionalProverStatus.AVAILABLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": OPTIONAL_PROVER_RECORD_SCHEMA,
            "prover_id": self.prover_id.value,
            "status": self.status.value,
            "required": self.required,
            "executable": self.executable,
            "reason_code": self.reason_code,
            "available": self.available,
        }


@dataclass(frozen=True)
class ProofCarryingPlannerHandle:
    """Optional proof-carrying planner construction boundary.

    The handle is always constructed so callers can observe availability.
    :meth:`build` refuses to mint a planner that would silently succeed when
    required optional provers are missing.
    """

    available: bool
    missing_provers: tuple[str, ...] = ()
    disposition: PlannerStackDisposition = PlannerStackDisposition.READY
    reason_code: str = ""
    compiler: FormalPlanCompiler | None = field(default=None, repr=False)
    validator: FormalPlanValidator | None = field(default=None, repr=False)
    adapters: WorkflowAdapters | None = field(default=None, repr=False)
    config: ProofCarryingPlannerConfig | Mapping[str, Any] | None = field(
        default=None, repr=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "available", bool(self.available))
        missing = tuple(
            sorted({str(item).strip() for item in self.missing_provers if str(item).strip()})
        )
        object.__setattr__(self, "missing_provers", missing)
        disposition = self.disposition
        if not isinstance(disposition, PlannerStackDisposition):
            disposition = PlannerStackDisposition(str(disposition))
        if not self.available and disposition is PlannerStackDisposition.READY:
            disposition = PlannerStackDisposition.DEFER_CAPABILITY
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(
            self, "reason_code", str(self.reason_code or "").strip()
        )

    @property
    def claims_success(self) -> bool:
        """Whether this handle may claim proof-carrying readiness."""

        return self.available and self.disposition is PlannerStackDisposition.READY

    def build(
        self,
        source: Mapping[str, Any] | None = None,
        *,
        artifact_path: Path | str | None = None,
        state_path: Path | str | None = None,
        adapters: WorkflowAdapters | None = None,
        config: ProofCarryingPlannerConfig | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> ProofCarryingPlanner:
        """Construct a proof-carrying planner or fail closed with defer_capability."""

        if not self.available:
            raise DefaultPlannerCapabilityError(
                "proof-carrying planner deferred: required optional provers are "
                f"unavailable ({', '.join(self.missing_provers) or 'unspecified'})",
                reason_code=DEFER_CAPABILITY_DISPOSITION,
                missing_provers=self.missing_provers,
            )
        if self.compiler is None or self.validator is None:
            raise DefaultPlannerFactoryError(
                "proof-carrying handle is missing bound compiler/validator",
                reason_code="unbound_core_stack",
            )
        try:
            return ProofCarryingPlanner(
                source,
                artifact_path=artifact_path,
                state_path=state_path,
                adapters=adapters if adapters is not None else self.adapters,
                config=config if config is not None else self.config,
                compiler=self.compiler,
                validator=self.validator,
                **kwargs,
            )
        except WorkflowConfigurationError as exc:
            raise DefaultPlannerFactoryError(
                str(exc), reason_code="proof_carrying_configuration_error"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "missing_provers": list(self.missing_provers),
            "disposition": self.disposition.value,
            "reason_code": self.reason_code,
            "claims_success": self.claims_success,
        }


@dataclass(frozen=True)
class DefaultPlannerHandles:
    """Bound production planner stack returned by the factory.

    Interface: ``DefaultPlannerHandles@1``
    """

    compiler: FormalPlanCompiler
    validator: FormalPlanValidator
    replanner: FormalReplanner
    adaptive_planner: AdaptivePlanner
    optional_prover_records: tuple[OptionalProverRecord, ...] = ()
    proof_carrying_handle: ProofCarryingPlannerHandle | None = None
    disposition: PlannerStackDisposition = PlannerStackDisposition.READY
    reason_codes: tuple[str, ...] = ()
    factory_interface: str = DEFAULT_PLANNER_FACTORY_INTERFACE
    handles_interface: str = DEFAULT_PLANNER_HANDLES_INTERFACE
    factory_version: int = DEFAULT_PLANNER_FACTORY_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.compiler, FormalPlanCompiler):
            raise DefaultPlannerFactoryError("compiler must be FormalPlanCompiler")
        if not isinstance(self.validator, FormalPlanValidator):
            raise DefaultPlannerFactoryError("validator must be FormalPlanValidator")
        if not isinstance(self.replanner, FormalReplanner):
            raise DefaultPlannerFactoryError("replanner must be FormalReplanner")
        if not isinstance(self.adaptive_planner, AdaptivePlanner):
            raise DefaultPlannerFactoryError(
                "adaptive_planner must be AdaptivePlanner"
            )
        records = tuple(self.optional_prover_records or ())
        if any(not isinstance(item, OptionalProverRecord) for item in records):
            raise DefaultPlannerFactoryError(
                "optional_prover_records must contain OptionalProverRecord instances"
            )
        object.__setattr__(self, "optional_prover_records", records)
        disposition = self.disposition
        if not isinstance(disposition, PlannerStackDisposition):
            disposition = PlannerStackDisposition(str(disposition))
        object.__setattr__(self, "disposition", disposition)
        reasons = tuple(
            str(item).strip()
            for item in (self.reason_codes or ())
            if str(item).strip()
        )
        if len(reasons) > _MAX_REASON_CODES:
            raise DefaultPlannerFactoryError(
                "reason_codes exceeds its item bound",
                reason_code="bounds_exceeded",
            )
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(
            self,
            "factory_interface",
            str(self.factory_interface or DEFAULT_PLANNER_FACTORY_INTERFACE),
        )
        object.__setattr__(
            self,
            "handles_interface",
            str(self.handles_interface or DEFAULT_PLANNER_HANDLES_INTERFACE),
        )
        object.__setattr__(self, "factory_version", int(self.factory_version))

    @property
    def core_ready(self) -> bool:
        """Core formal stack is bound (independent of optional provers)."""

        return (
            self.compiler is not None
            and self.validator is not None
            and self.replanner is not None
            and self.adaptive_planner is not None
        )

    @property
    def missing_optional_provers(self) -> tuple[str, ...]:
        return tuple(
            item.prover_id.value
            for item in self.optional_prover_records
            if not item.available
        )

    @property
    def available_optional_provers(self) -> tuple[str, ...]:
        return tuple(
            item.prover_id.value
            for item in self.optional_prover_records
            if item.available
        )

    @property
    def missing_required_optional_provers(self) -> tuple[str, ...]:
        return tuple(
            item.prover_id.value
            for item in self.optional_prover_records
            if item.required and not item.available
        )

    @property
    def capability_complete(self) -> bool:
        """True only when every *required* optional prover is available."""

        return not self.missing_required_optional_provers

    @property
    def claims_success(self) -> bool:
        """Whether the factory may claim full stack readiness including provers.

        Missing required optional provers never produce silent success.  When
        optional provers were probed and any remain unavailable, success is
        also withheld so inventory gaps cannot be mistaken for proof readiness.
        """

        if not self.core_ready:
            return False
        if not self.capability_complete:
            return False
        if self.disposition is not PlannerStackDisposition.READY:
            return False
        # Probed optional inventory with gaps is never silent success.
        if self.optional_prover_records and self.missing_optional_provers:
            return False
        return True

    @property
    def defers_capability(self) -> bool:
        return self.disposition is PlannerStackDisposition.DEFER_CAPABILITY

    @property
    def optional_prover_status(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                item.prover_id.value: item.status.value
                for item in self.optional_prover_records
            }
        )

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    def _ir_logic_capability_projection(self) -> dict[str, Any]:
        """Domain-agnostic IR logic capability (planner/doctor/repair shared).

        Covers shared IR (intent/legal/security/ui) and structural IR
        (AST/knowledge graph/vector index) — not SCA-taskboard-specific.
        """
        try:
            from .ir_logic_consumers import probe_ir_logic_consumer_capability
            from ..proof.ir_integration import probe_ir_integration

            consumer = probe_ir_logic_consumer_capability()
            inventory = probe_ir_integration(domain="planner")
            return {
                "evidence_key": IR_LOGIC_CAPABILITY_EVIDENCE,
                **consumer,
                "inventory_passed": bool(inventory.get("passed")),
                "shared_ir_families": list(inventory.get("shared_ir_families") or []),
                "structural_ir_families": list(
                    inventory.get("structural_ir_families") or []
                ),
                "family_availability": {
                    name: bool((doc or {}).get("available"))
                    for name, doc in (inventory.get("families") or {}).items()
                },
                "consumer_hooks": {
                    name: bool((doc or {}).get("available"))
                    for name, doc in (inventory.get("consumers") or {}).items()
                },
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "evidence_key": IR_LOGIC_CAPABILITY_EVIDENCE,
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
                "grants_execution_authority": False,
            }

    def to_dict(self) -> dict[str, Any]:
        """Body-free durable projection of the bound stack."""

        proof = (
            self.proof_carrying_handle.to_dict()
            if self.proof_carrying_handle is not None
            else {
                "available": False,
                "missing_provers": [],
                "disposition": PlannerStackDisposition.DEFER_CAPABILITY.value,
                "reason_code": "proof_carrying_handle_unbound",
                "claims_success": False,
            }
        )
        return {
            "schema": DEFAULT_PLANNER_HANDLES_SCHEMA,
            "factory_interface": self.factory_interface,
            "handles_interface": self.handles_interface,
            "factory_version": self.factory_version,
            "core_ready": self.core_ready,
            "disposition": self.disposition.value,
            "capability_complete": self.capability_complete,
            "claims_success": self.claims_success,
            "defers_capability": self.defers_capability,
            "reason_codes": list(self.reason_codes),
            "optional_provers": [item.to_dict() for item in self.optional_prover_records],
            "missing_optional_provers": list(self.missing_optional_provers),
            "missing_required_optional_provers": list(
                self.missing_required_optional_provers
            ),
            "available_optional_provers": list(self.available_optional_provers),
            "proof_carrying_handle": proof,
            "components": {
                "compiler": type(self.compiler).__name__,
                "validator": type(self.validator).__name__,
                "replanner": type(self.replanner).__name__,
                "adaptive_planner": type(self.adaptive_planner).__name__,
            },
            "ir_logic": self._ir_logic_capability_projection(),
            "shared_stack": {
                "replanner_uses_factory_compiler": self.replanner.compiler
                is self.compiler,
                "replanner_uses_factory_validator": self.replanner.validator
                is self.validator,
            },
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prover_id(value: Any) -> OptionalProverId:
    if isinstance(value, OptionalProverId):
        return value
    key = str(getattr(value, "value", value) or "").strip().casefold()
    aliases = {
        "lean": OptionalProverId.LEAN,
        "lean4": OptionalProverId.LEAN,
        "z3": OptionalProverId.Z3,
        "cvc5": OptionalProverId.CVC5,
        "cvc4": OptionalProverId.CVC5,
        "coq": OptionalProverId.COQ,
        "coqc": OptionalProverId.COQ,
        "hammer": OptionalProverId.HAMMER,
    }
    if key not in aliases:
        raise DefaultPlannerFactoryError(
            f"unsupported optional prover id: {value!r}",
            reason_code="unsupported_optional_prover",
        )
    return aliases[key]


def _probe_executable(
    prover: OptionalProverId,
    *,
    which: ExecutableFinder,
) -> tuple[OptionalProverStatus, str, str]:
    candidates = _EXECUTABLE_CANDIDATES.get(prover, (prover.value,))
    for candidate in candidates:
        try:
            found = which(candidate)
        except BaseException as exc:  # noqa: BLE001 — probe must not abort factory
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return (
                OptionalProverStatus.FAILED,
                "",
                f"probe_failed:{type(exc).__name__}",
            )
        if found:
            return OptionalProverStatus.AVAILABLE, str(found), "executable_found"
    return (
        OptionalProverStatus.UNAVAILABLE,
        "",
        f"executable_not_found:{','.join(candidates)}",
    )


def _probe_optional_prover(
    prover: OptionalProverId,
    *,
    which: ExecutableFinder,
    custom_probes: Mapping[str, OptionalProverProbe] | None,
    required: bool,
) -> OptionalProverRecord:
    custom = None
    if custom_probes:
        custom = custom_probes.get(prover.value) or custom_probes.get(prover.name)
    if custom is not None:
        try:
            result = custom(prover.value)
        except BaseException as exc:  # noqa: BLE001
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return OptionalProverRecord(
                prover_id=prover,
                status=OptionalProverStatus.FAILED,
                required=required,
                reason_code=f"custom_probe_failed:{type(exc).__name__}",
            )
        if result is True:
            return OptionalProverRecord(
                prover_id=prover,
                status=OptionalProverStatus.AVAILABLE,
                required=required,
                reason_code="custom_probe_available",
            )
        if result is False or result is None:
            return OptionalProverRecord(
                prover_id=prover,
                status=OptionalProverStatus.UNAVAILABLE,
                required=required,
                reason_code="custom_probe_unavailable",
            )
        if isinstance(result, Mapping):
            status_raw = str(
                result.get("status") or result.get("health") or ""
            ).strip().casefold()
            if status_raw in {"available", "ok", "supported", "ready"}:
                status = OptionalProverStatus.AVAILABLE
            elif status_raw in {"degraded", "partial"}:
                status = OptionalProverStatus.DEGRADED
            elif status_raw in {"failed", "error"}:
                status = OptionalProverStatus.FAILED
            else:
                status = OptionalProverStatus.UNAVAILABLE
            return OptionalProverRecord(
                prover_id=prover,
                status=status,
                required=required,
                executable=str(result.get("executable") or result.get("path") or ""),
                reason_code=str(
                    result.get("reason_code") or result.get("reason") or "custom_probe"
                ),
            )
        return OptionalProverRecord(
            prover_id=prover,
            status=OptionalProverStatus.AVAILABLE,
            required=required,
            reason_code="custom_probe_truthy",
        )

    status, executable, reason = _probe_executable(prover, which=which)
    return OptionalProverRecord(
        prover_id=prover,
        status=status,
        required=required,
        executable=executable,
        reason_code=reason,
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class DefaultPlannerFactory:
    """Production composition root for formal planner defaults.

    Interface: ``DefaultPlannerFactory@1``

    Always binds compiler, validator, replanner, and adaptive planner.
    Optional provers are inventory-probed; required absences yield
    ``defer_capability`` rather than silent success.
    """

    INTERFACE: Final[str] = DEFAULT_PLANNER_FACTORY_INTERFACE
    VERSION: Final[int] = DEFAULT_PLANNER_FACTORY_VERSION

    def __init__(
        self,
        *,
        validation_bounds: ValidationBounds | Mapping[str, Any] | None = None,
        replan_limits: ReplanLimits | Mapping[str, Any] | None = None,
        max_adaptive_candidates: int = 32,
        optional_provers: Sequence[OptionalProverId | str] | None = None,
        require_optional_provers: Sequence[OptionalProverId | str] = (),
        which: ExecutableFinder | None = None,
        optional_prover_probes: Mapping[str, OptionalProverProbe] | None = None,
        prover_executor: Callable[[Mapping[str, Any]], Any] | None = None,
        verifier: Callable[[Mapping[str, Any]], Any] | None = None,
        proof_carrying_config: ProofCarryingPlannerConfig
        | Mapping[str, Any]
        | None = None,
        strict_unknown_semantics: bool = True,
        default_trace_bound: int = 16,
        require_proof_carrying: bool = False,
    ) -> None:
        if isinstance(max_adaptive_candidates, bool) or max_adaptive_candidates < 1:
            raise DefaultPlannerFactoryError(
                "max_adaptive_candidates must be a positive integer",
                reason_code="invalid_bounds",
            )
        if isinstance(default_trace_bound, bool) or default_trace_bound <= 0:
            raise DefaultPlannerFactoryError(
                "default_trace_bound must be a positive integer",
                reason_code="invalid_bounds",
            )
        self._validation_bounds = validation_bounds
        self._replan_limits = replan_limits
        self._max_adaptive_candidates = int(max_adaptive_candidates)
        provers = (
            DEFAULT_OPTIONAL_PROVERS
            if optional_provers is None
            else tuple(_prover_id(item) for item in optional_provers)
        )
        # Stable unique order.
        self._optional_provers = tuple(dict.fromkeys(provers))
        required = tuple(
            dict.fromkeys(_prover_id(item) for item in require_optional_provers)
        )
        self._require_optional_provers = required
        self._which = which or shutil.which
        self._optional_prover_probes = dict(optional_prover_probes or {})
        self._prover_executor = prover_executor
        self._verifier = verifier
        self._proof_carrying_config = proof_carrying_config
        self._strict_unknown_semantics = bool(strict_unknown_semantics)
        self._default_trace_bound = int(default_trace_bound)
        self._require_proof_carrying = bool(require_proof_carrying)
        self._last_handles: DefaultPlannerHandles | None = None

    @property
    def last_handles(self) -> DefaultPlannerHandles | None:
        return self._last_handles

    def probe_optional_provers(self) -> tuple[OptionalProverRecord, ...]:
        """Probe optional provers without constructing the planner stack."""

        required = set(self._require_optional_provers)
        # Proof-carrying readiness treats the configured optional set as
        # required when the caller demanded a proof-carrying handle.
        if self._require_proof_carrying:
            required.update(self._optional_provers)
        records: list[OptionalProverRecord] = []
        for prover in self._optional_provers:
            records.append(
                _probe_optional_prover(
                    prover,
                    which=self._which,
                    custom_probes=self._optional_prover_probes,
                    required=prover in required,
                )
            )
        # Required provers not already in the inventory set are still probed.
        for prover in self._require_optional_provers:
            if prover in self._optional_provers:
                continue
            records.append(
                _probe_optional_prover(
                    prover,
                    which=self._which,
                    custom_probes=self._optional_prover_probes,
                    required=True,
                )
            )
        return tuple(records)

    def build(self) -> DefaultPlannerHandles:
        """Bind the production formal planner stack."""

        records = self.probe_optional_provers()
        missing_required = tuple(
            item.prover_id.value
            for item in records
            if item.required and not item.available
        )
        any_available = any(item.available for item in records)

        compiler = FormalPlanCompiler(
            strict_unknown_semantics=self._strict_unknown_semantics,
            default_trace_bound=self._default_trace_bound,
        )
        validator = FormalPlanValidator(self._validation_bounds)

        verifier_available: bool | None
        if self._verifier is not None:
            verifier_available = True
        elif missing_required:
            verifier_available = False
        elif any_available:
            verifier_available = True
        else:
            # Optional inventory empty / all unavailable and none required:
            # do not pretend a verifier exists.
            verifier_available = False

        replanner = FormalReplanner(
            compiler=compiler,
            validator=validator,
            limits=self._replan_limits,
            verifier=self._verifier,
            verifier_available=verifier_available,
        )
        adaptive_planner = AdaptivePlanner(
            max_candidates=self._max_adaptive_candidates
        )

        reason_codes: list[str] = []
        disposition = PlannerStackDisposition.READY

        if missing_required:
            disposition = PlannerStackDisposition.DEFER_CAPABILITY
            reason_codes.append(
                "missing_required_optional_provers:"
                + ",".join(missing_required)
            )
        elif self._require_proof_carrying and not any_available:
            disposition = PlannerStackDisposition.DEFER_CAPABILITY
            reason_codes.append("proof_carrying_backends_unavailable")

        proof_available = (
            disposition is PlannerStackDisposition.READY
            and not missing_required
            and (self._prover_executor is not None or any_available)
        )
        # Proof-carrying is optional: only claim availability when backends
        # exist.  Absence never forges a successful handle.
        if self._require_proof_carrying and not proof_available:
            disposition = PlannerStackDisposition.DEFER_CAPABILITY
            if "proof_carrying_backends_unavailable" not in reason_codes:
                reason_codes.append("proof_carrying_backends_unavailable")

        missing_for_handle = tuple(
            item.prover_id.value for item in records if not item.available
        )
        adapters = WorkflowAdapters(prover_lane=self._prover_executor)
        proof_handle = ProofCarryingPlannerHandle(
            available=proof_available and disposition is PlannerStackDisposition.READY,
            missing_provers=() if proof_available else missing_for_handle,
            disposition=(
                PlannerStackDisposition.READY
                if proof_available and disposition is PlannerStackDisposition.READY
                else PlannerStackDisposition.DEFER_CAPABILITY
            ),
            reason_code=(
                "proof_carrying_ready"
                if proof_available and disposition is PlannerStackDisposition.READY
                else (
                    "missing_optional_provers"
                    if missing_for_handle
                    else "prover_executor_unbound"
                )
            ),
            compiler=compiler,
            validator=validator,
            adapters=adapters,
            config=self._proof_carrying_config,
        )

        handles = DefaultPlannerHandles(
            compiler=compiler,
            validator=validator,
            replanner=replanner,
            adaptive_planner=adaptive_planner,
            optional_prover_records=records,
            proof_carrying_handle=proof_handle,
            disposition=disposition,
            reason_codes=tuple(reason_codes),
        )
        # Hard invariant: never claim success when required provers are gone.
        if missing_required and handles.claims_success:
            raise DefaultPlannerFactoryError(
                "invariant violation: claims_success with missing required provers",
                reason_code="silent_success_forbidden",
            )
        if (
            handles.disposition is PlannerStackDisposition.DEFER_CAPABILITY
            and handles.claims_success
        ):
            raise DefaultPlannerFactoryError(
                "invariant violation: claims_success under defer_capability",
                reason_code="silent_success_forbidden",
            )
        self._last_handles = handles
        return handles


def build_default_planner_factory(
    *,
    validation_bounds: ValidationBounds | Mapping[str, Any] | None = None,
    replan_limits: ReplanLimits | Mapping[str, Any] | None = None,
    max_adaptive_candidates: int = 32,
    optional_provers: Sequence[OptionalProverId | str] | None = None,
    require_optional_provers: Sequence[OptionalProverId | str] = (),
    which: ExecutableFinder | None = None,
    optional_prover_probes: Mapping[str, OptionalProverProbe] | None = None,
    prover_executor: Callable[[Mapping[str, Any]], Any] | None = None,
    verifier: Callable[[Mapping[str, Any]], Any] | None = None,
    proof_carrying_config: ProofCarryingPlannerConfig | Mapping[str, Any] | None = None,
    strict_unknown_semantics: bool = True,
    default_trace_bound: int = 16,
    require_proof_carrying: bool = False,
) -> DefaultPlannerFactory:
    """Construct a production default planner factory."""

    return DefaultPlannerFactory(
        validation_bounds=validation_bounds,
        replan_limits=replan_limits,
        max_adaptive_candidates=max_adaptive_candidates,
        optional_provers=optional_provers,
        require_optional_provers=require_optional_provers,
        which=which,
        optional_prover_probes=optional_prover_probes,
        prover_executor=prover_executor,
        verifier=verifier,
        proof_carrying_config=proof_carrying_config,
        strict_unknown_semantics=strict_unknown_semantics,
        default_trace_bound=default_trace_bound,
        require_proof_carrying=require_proof_carrying,
    )


def build_default_planner_handles(**kwargs: Any) -> DefaultPlannerHandles:
    """One-shot helper that builds a factory and returns bound handles."""

    return build_default_planner_factory(**kwargs).build()


# ---------------------------------------------------------------------------
# DCR-060 composition-root readiness (non-authoritative)
# ---------------------------------------------------------------------------

# This intentionally lives beside the existing factory instead of becoming a
# second planner implementation.  It only admits *already constructed* formal
# planner handles after checking the cross-DCR identity surface.  In
# particular, it never creates Doctor/provider/network clients and it never
# calls a supplied service object.
DCR_PLANNER_COMPOSITION_INTERFACE: Final[str] = "DcrPlannerComposition@1"
DCR_PLANNER_COMPOSITION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-planner-composition@1"
)
DCR_PLANNER_SERVICE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-planner-service-receipt@1"
)
DCR_PLANNER_LIVE_EVIDENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-planner-live-evidence@1"
)


class DcrPlannerCompositionDisposition(str, Enum):
    """Closed readiness outcomes; none is an execution authorization."""

    READY = "ready_for_composition"
    INTEGRATION_PENDING = "integration_pending"
    DEFER_CAPABILITY = "defer_capability"
    REJECTED = "rejected"


class DcrPlannerServiceKind(str, Enum):
    """Only the composition services that may be referenced by DCR-060."""

    RECEIPT = "receipt_service"
    CANDIDATE = "deterministic_candidate_service"
    SCHEDULER = "deterministic_scheduler_service"


def _dcr060_cid(value: Any) -> str:
    """Keep DCR-060 identities on the canonical repository CID primitive."""

    return content_identity(value)


@dataclass(frozen=True)
class DcrPlannerSelfTestReceipt:
    """Content-bound, closed self-test result for one deterministic service.

    A bare boolean/package-presence check is deliberately not represented.
    This record is only a readiness input and grants no runtime authority.
    """

    service_kind: DcrPlannerServiceKind
    service_identity: str
    root_cids: tuple[str, ...]
    input_cids: tuple[str, ...]
    outcome: str = "passed"
    model_call_count: int = 0
    provider_call_count: int = 0
    network_call_count: int = 0
    swallowed_exception: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.service_kind, DcrPlannerServiceKind):
            raise DefaultPlannerFactoryError("DCR-060 service kind must be closed")
        for field_name in ("service_identity", "outcome"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise DefaultPlannerFactoryError(
                    f"DCR-060 {field_name} must be non-empty exact text"
                )
        for field_name in ("root_cids", "input_cids"):
            values = tuple(getattr(self, field_name) or ())
            if not values or any(
                not isinstance(item, str) or not item or item != item.strip()
                for item in values
            ):
                raise DefaultPlannerFactoryError(
                    f"DCR-060 {field_name} must be non-empty exact CIDs"
                )
            object.__setattr__(self, field_name, tuple(sorted(set(values))))
        if self.outcome != "passed":
            raise DefaultPlannerFactoryError("DCR-060 self-test outcome must be passed")
        if self.swallowed_exception is not False:
            raise DefaultPlannerFactoryError("DCR-060 self-test may not swallow exceptions")
        for field_name in (
            "model_call_count",
            "provider_call_count",
            "network_call_count",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value != 0:
                raise DefaultPlannerFactoryError(
                    f"DCR-060 self-test {field_name} must be exactly zero"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DCR_PLANNER_SERVICE_RECEIPT_SCHEMA,
            "kind": self.service_kind.value,
            "service_identity": self.service_identity,
            "root_cids": list(self.root_cids),
            "input_cids": list(self.input_cids),
            "outcome": self.outcome,
            "model_call_count": self.model_call_count,
            "provider_call_count": self.provider_call_count,
            "network_call_count": self.network_call_count,
            "swallowed_exception": self.swallowed_exception,
        }

    @property
    def receipt_cid(self) -> str:
        return _dcr060_cid(self.to_dict())


@dataclass(frozen=True)
class DcrPlannerServiceBinding:
    """A non-callable service identity and its exact self-test receipt."""

    service_kind: DcrPlannerServiceKind
    service_interface: str
    service_identity: str
    self_test: DcrPlannerSelfTestReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.service_kind, DcrPlannerServiceKind):
            raise DefaultPlannerFactoryError("DCR-060 service kind must be closed")
        for field_name in ("service_interface", "service_identity"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise DefaultPlannerFactoryError(
                    f"DCR-060 {field_name} must be non-empty exact text"
                )
        if callable(self.self_test) or not isinstance(
            self.self_test, DcrPlannerSelfTestReceipt
        ):
            raise DefaultPlannerFactoryError("DCR-060 typed self-test receipt is required")
        if (
            self.self_test.service_kind is not self.service_kind
            or self.self_test.service_identity != self.service_identity
        ):
            raise DefaultPlannerFactoryError("DCR-060 service/self-test identity mismatch")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.service_kind.value,
            "interface": self.service_interface,
            "service_identity": self.service_identity,
            "self_test_cid": self.self_test.receipt_cid,
        }

    @property
    def binding_cid(self) -> str:
        return _dcr060_cid(self.to_dict())


@dataclass(frozen=True)
class DcrPlannerRoots:
    """The exact cross-DCR roots a planner composition must preserve."""

    policy_cid: str
    forest_cid: str
    graph_cid: str
    findings_cid: str
    logic_candidate_cid: str
    proof_cache_cid: str
    operator_registry_cid: str

    def __post_init__(self) -> None:
        for field_name in (
            "policy_cid",
            "forest_cid",
            "graph_cid",
            "findings_cid",
            "logic_candidate_cid",
            "proof_cache_cid",
            "operator_registry_cid",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise DefaultPlannerFactoryError(
                    f"DCR-060 {field_name} must be non-empty exact text"
                )

    def to_dict(self) -> dict[str, str]:
        return {
            "policy_cid": self.policy_cid,
            "forest_cid": self.forest_cid,
            "graph_cid": self.graph_cid,
            "findings_cid": self.findings_cid,
            "logic_candidate_cid": self.logic_candidate_cid,
            "proof_cache_cid": self.proof_cache_cid,
            "operator_registry_cid": self.operator_registry_cid,
        }


@dataclass(frozen=True)
class DcrPlannerLiveEvidence:
    """Future DCR-050/053 live binding; typed but never execution authority."""

    doctor_binding_cid: str
    doctor_service_interface: str
    dcr053_receipt_cid: str
    roots: DcrPlannerRoots

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DCR_PLANNER_LIVE_EVIDENCE_SCHEMA,
            "doctor_binding_cid": self.doctor_binding_cid,
            "doctor_service_interface": self.doctor_service_interface,
            "dcr053_receipt_cid": self.dcr053_receipt_cid,
            "roots": self.roots.to_dict() if isinstance(self.roots, DcrPlannerRoots) else {},
        }

    @property
    def evidence_cid(self) -> str:
        return _dcr060_cid(self.to_dict())


@dataclass(frozen=True)
class DcrPlannerCompositionEvidence:
    """Strict typed inputs for DCR-060's non-authoritative adapter."""

    doctor_binding: Any
    doctor_service: Any
    logic_candidate: Any
    stage_gate: Any
    proof_cache_binding: Any
    operator_registry: Any
    roots: DcrPlannerRoots
    services: tuple[DcrPlannerServiceBinding, ...]
    live_evidence: DcrPlannerLiveEvidence | None = None


@dataclass(frozen=True)
class DcrPlannerCompositionResult:
    """Readiness projection.  It cannot authorize execution or completion."""

    disposition: DcrPlannerCompositionDisposition
    reason_codes: tuple[str, ...]
    roots: DcrPlannerRoots | None = None
    planner_handles: DefaultPlannerHandles | None = None
    planner_view_cid: str = ""
    model_call_count: int = 0
    provider_call_count: int = 0
    network_call_count: int = 0
    execution_authorized: bool = False
    completion_authorized: bool = False

    @property
    def integration_pending(self) -> bool:
        return self.disposition is DcrPlannerCompositionDisposition.INTEGRATION_PENDING

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": DCR_PLANNER_COMPOSITION_SCHEMA,
            "interface": DCR_PLANNER_COMPOSITION_INTERFACE,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "roots": self.roots.to_dict() if self.roots is not None else {},
            "planner_view_cid": self.planner_view_cid,
            "model_call_count": 0,
            "provider_call_count": 0,
            "network_call_count": 0,
            "execution_authorized": False,
            "completion_authorized": False,
        }
        return payload


def _dcr060_reason_set(
    handles: Any,
    evidence: Any,
) -> tuple[list[str], DcrPlannerRoots | None]:
    """Perform type and identity checks without calling arbitrary objects."""

    reasons: list[str] = []
    roots = evidence.roots if isinstance(evidence, DcrPlannerCompositionEvidence) else None
    if not isinstance(handles, DefaultPlannerHandles):
        reasons.append("typed_default_planner_handles_required")
    elif (
        not handles.core_ready
        or handles.disposition is not PlannerStackDisposition.READY
        or not handles.claims_success
        or handles.proof_carrying_handle is None
        or not handles.proof_carrying_handle.available
    ):
        reasons.append("default_planner_handles_not_proof_carrying_ready")
    if not isinstance(evidence, DcrPlannerCompositionEvidence):
        reasons.append("typed_dcr060_evidence_required")
        return reasons, None
    if not isinstance(roots, DcrPlannerRoots):
        reasons.append("typed_dcr060_roots_required")
        return reasons, None

    # Local imports defer cross-DCR module loading until this explicit
    # assessment; construction of the DefaultPlannerFactory stays cold.
    from ..proof.dcr_proof_cache import DcrProofCacheBinding
    from ..proof.ir_integration import DatasetsLogicIrDisposition, DatasetsLogicIrResult
    from ..proof.ir_logic_application import (
        IrLogicRequiredGateDisposition,
        IrLogicRequiredGateResult,
    )
    from ..autonomous_repair.operators.registry import (
        OPERATOR_REGISTRY_SCHEMA,
        OperatorRegistry,
    )

    # DefaultDoctorBinding/DeterministicDoctorService predate strict DCR-050
    # composition and are not authority inputs here.  Until its typed result
    # contract exists, any legacy objects are merely rejected/pending context.
    if evidence.doctor_binding is not None or evidence.doctor_service is not None:
        reasons.append("legacy_doctor_binding_cannot_satisfy_dcr050")
    reasons.append("integration_pending_dcr050_typed_composition_unavailable")

    candidate = evidence.logic_candidate
    if not isinstance(candidate, DatasetsLogicIrResult):
        reasons.append("typed_dcr030_logic_candidate_required")
    elif (
        candidate.disposition is not DatasetsLogicIrDisposition.NORMALIZED
        or candidate.model_call_count != 0
        or candidate.mutation_authorized
        or not candidate.module_binding
        or _dcr060_cid(candidate.to_dict()) != roots.logic_candidate_cid
    ):
        reasons.append("dcr030_logic_candidate_identity_or_disposition_invalid")

    gate = evidence.stage_gate
    if not isinstance(gate, IrLogicRequiredGateResult):
        reasons.append("typed_dcr035_stage_gate_required")
    elif (
        gate.disposition is not IrLogicRequiredGateDisposition.PASSING
        or gate.model_call_count != 0
        or gate.provider_call_count != 0
        or gate.execution_authorized
        or gate.completion_authorized
        or gate.required_identity_cids.get("dcr030") != roots.logic_candidate_cid
        or gate.required_identity_cids.get("dcr034") != roots.proof_cache_cid
    ):
        reasons.append("dcr035_stage_gate_identity_or_verdict_invalid")

    cache_binding = evidence.proof_cache_binding
    if not isinstance(cache_binding, DcrProofCacheBinding):
        reasons.append("typed_dcr034_cache_binding_required")
    elif (
        cache_binding.validation_reasons()
        or cache_binding.key_cid != roots.proof_cache_cid
        or cache_binding.policy_root != roots.policy_cid
        or cache_binding.dcr030_forest_root != roots.forest_cid
        or cache_binding.dcr031_obligation.graph_cid != roots.graph_cid
    ):
        reasons.append("dcr034_cache_binding_identity_or_receipt_invalid")

    registry = evidence.operator_registry
    if not isinstance(registry, OperatorRegistry):
        reasons.append("typed_dcr040_operator_registry_required")
    else:
        report = registry.report()
        report_body = dict(report)
        report_cid = report_body.pop("registry_cid", "")
        if (
            report.get("schema") != OPERATOR_REGISTRY_SCHEMA
            or report.get("authoritative") is not False
            or report.get("execution_authorized") is not False
            or report.get("model_call_count") != 0
            or report_cid != _dcr060_cid(report_body)
            or report_cid != roots.operator_registry_cid
        ):
            reasons.append("dcr040_operator_registry_identity_or_review_invalid")

    services = evidence.services
    if (
        not isinstance(services, tuple)
        or {item.service_kind for item in services if isinstance(item, DcrPlannerServiceBinding)}
        != set(DcrPlannerServiceKind)
        or len(services) != len(DcrPlannerServiceKind)
    ):
        reasons.append("closed_deterministic_service_bindings_required")
    else:
        expected_roots = set(roots.to_dict().values())
        for item in services:
            if not isinstance(item, DcrPlannerServiceBinding):
                reasons.append("typed_deterministic_service_binding_required")
                continue
            receipt = item.self_test
            if (
                receipt.receipt_cid != _dcr060_cid(receipt.to_dict())
                or not expected_roots.issubset(set(receipt.root_cids))
                or not set(receipt.input_cids).issuperset(
                    {roots.logic_candidate_cid, roots.findings_cid}
                )
                or "synthetic" in item.service_identity.lower()
                or "stub" in item.service_interface.lower()
            ):
                reasons.append("deterministic_service_self_test_or_identity_invalid")

    live = evidence.live_evidence
    if not isinstance(live, DcrPlannerLiveEvidence):
        reasons.append("integration_pending_dcr050_dcr053_live_evidence")
    elif live.roots != roots or not live.dcr053_receipt_cid:
        reasons.append("dcr053_live_evidence_identity_invalid")
    # This checkout intentionally has no typed DCR-053 live-receipt contract
    # yet.  A populated string in the transitional wrapper is not evidence,
    # so it must never unlock a planner view on its own.
    reasons.append("integration_pending_dcr053_typed_live_receipt_unavailable")
    return reasons, roots


def assess_dcr_planner_composition(
    handles: DefaultPlannerHandles,
    evidence: DcrPlannerCompositionEvidence,
) -> DcrPlannerCompositionResult:
    """Check whether existing planner handles may be exposed to DCR-060.

    The function is pure with respect to its inputs: it does not build a
    factory, run a self-test, execute a planner, access a network, or expose
    completion authority.  Invalid/pending inputs intentionally have no
    ``planner_view_cid`` and no retained handles.
    """

    reasons, roots = _dcr060_reason_set(handles, evidence)
    if reasons:
        disposition = (
            DcrPlannerCompositionDisposition.INTEGRATION_PENDING
            if all(
                code.startswith("integration_pending_")
                or code.endswith("_not_current_live")
                for code in reasons
            )
            else DcrPlannerCompositionDisposition.DEFER_CAPABILITY
        )
        return DcrPlannerCompositionResult(
            disposition=disposition,
            reason_codes=tuple(sorted(set(reasons))),
        )
    assert roots is not None
    # Handles are already checked as real planner components above.  The view
    # identity binds their existing durable projection and all DCR roots; it
    # is read-only metadata, never an evidence receipt or an authorization.
    view_cid = _dcr060_cid(
        {
            "schema": DCR_PLANNER_COMPOSITION_SCHEMA,
            "roots": roots.to_dict(),
            "planner_handles_cid": handles.content_id,
        }
    )
    return DcrPlannerCompositionResult(
        disposition=DcrPlannerCompositionDisposition.READY,
        reason_codes=(),
        roots=roots,
        planner_handles=handles,
        planner_view_cid=view_cid,
    )


__all__ = [
    "DEFER_CAPABILITY_DISPOSITION",
    "DEFAULT_OPTIONAL_PROVERS",
    "DEFAULT_PLANNER_CAPABILITY_SCHEMA",
    "DEFAULT_PLANNER_FACTORY_EVIDENCE",
    "DEFAULT_PLANNER_FACTORY_INTERFACE",
    "DEFAULT_PLANNER_FACTORY_VERSION",
    "DEFAULT_PLANNER_HANDLES_INTERFACE",
    "DEFAULT_PLANNER_HANDLES_SCHEMA",
    "DCR_PLANNER_COMPOSITION_INTERFACE",
    "DCR_PLANNER_COMPOSITION_SCHEMA",
    "DCR_PLANNER_LIVE_EVIDENCE_SCHEMA",
    "DCR_PLANNER_SERVICE_RECEIPT_SCHEMA",
    "DcrPlannerCompositionDisposition",
    "DcrPlannerCompositionEvidence",
    "DcrPlannerCompositionResult",
    "DcrPlannerLiveEvidence",
    "DcrPlannerRoots",
    "DcrPlannerSelfTestReceipt",
    "DcrPlannerServiceBinding",
    "DcrPlannerServiceKind",
    "OPTIONAL_PROVER_RECORD_SCHEMA",
    "DefaultPlannerCapabilityError",
    "DefaultPlannerFactory",
    "DefaultPlannerFactoryError",
    "DefaultPlannerHandles",
    "OptionalProverId",
    "OptionalProverRecord",
    "OptionalProverStatus",
    "PlannerStackDisposition",
    "ProofCarryingPlannerHandle",
    "build_default_planner_factory",
    "build_default_planner_handles",
    "assess_dcr_planner_composition",
]
