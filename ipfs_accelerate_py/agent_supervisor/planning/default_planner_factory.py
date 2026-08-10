"""Production default planner factory (WPD-011 / DCR-060).

Interface: ``DefaultPlannerFactory@1`` / ``PlannerCompositionRoot@1``

Binds the live formal planning stack so CLI/API defaults never ship with
empty injected slots:

* :class:`~.formal_plan_compiler.FormalPlanCompiler`
* :class:`~.formal_plan_validator.FormalPlanValidator`
* :class:`~.formal_replanner.FormalReplanner`
* :class:`~.adaptive_planner.AdaptivePlanner`
* optional :class:`~.proof_carrying_planner.ProofCarryingPlanner` handle

DCR-060 extends the same composition root with production Doctor, datasets
logic, candidate portfolio, planner-node scheduler, IR logic (proof) hooks,
and plan-admission receipt services.  Default handles exercise those real
services; a missing mandatory component is recorded unavailable and the
stack cannot mint planner-view evidence.

Fail-closed rules:

* Core compiler / validator / replanner / adaptive planner are always real
  instances that share the same compiler and validator.
* Optional provers are probed without importing provider modules or claiming
  proof success from package presence alone.
* Required optional provers that are absent yield disposition
  ``defer_capability`` — never silent success and never a forged proof-
  carrying planner that pretends backends are available.
* Mandatory DCR components that fail to bind never become synthetic success;
  planner-view evidence minting refuses the incomplete stack.
* Required planner IR hooks propagate typed failures; exception swallowing
  and synthetic capability probes are forbidden.
* This module does not import ``todo_daemon`` (package DAG).  The disposition
  string matches :class:`ImplementationDisposition.DEFER_CAPABILITY`.
* Cold import never loads LLM / remote model-provider surfaces.
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
PLANNER_COMPOSITION_ROOT_INTERFACE: Final[str] = "PlannerCompositionRoot@1"
DEFAULT_PLANNER_FACTORY_VERSION: Final[int] = 1
PLANNER_COMPOSITION_ROOT_VERSION: Final[int] = 1

DEFAULT_PLANNER_HANDLES_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/default-planner-handles@1"
)
DEFAULT_PLANNER_CAPABILITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/default-planner-capability@1"
)
OPTIONAL_PROVER_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/optional-prover-record@1"
)
PLANNER_COMPONENT_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planner-component-record@1"
)
PLANNER_VIEW_EVIDENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planner-view-evidence@1"
)
PLANNER_COMPOSITION_ROOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planner-composition-root@1"
)
PLANNER_NODE_SCHEDULER_INTERFACE: Final[str] = "PlannerNodeScheduler@1"

# Objective-heap evidence key for the WPD factories goal packet.
DEFAULT_PLANNER_FACTORY_EVIDENCE: Final[str] = "wpd/default-planner-factory@1"
# DCR-060 evidence term for production planner composition.
DCR_PLANNER_FACTORY_EVIDENCE: Final[str] = "dcr/planner-factory@1"
DCR_PLANNER_CAPABILITIES_REL: Final[str] = (
    "data/agent_supervisor/deterministic_contract_repair/planner-capabilities.json"
)

# Wire spelling shared with ImplementationDisposition.DEFER_CAPABILITY.
DEFER_CAPABILITY_DISPOSITION: Final[str] = "defer_capability"

# Mandatory DCR composition slots (effects of DCR-060).
MANDATORY_PLANNER_COMPONENTS: Final[tuple[str, ...]] = (
    "compiler",
    "validator",
    "replanner",
    "candidate_portfolio",
    "scheduler",
    "doctor",
    "logic",
    "proof",
    "receipt",
)

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


class PlannerComponentId(str, Enum):
    """Closed mandatory composition slots for DCR-060."""

    COMPILER = "compiler"
    VALIDATOR = "validator"
    REPLANNER = "replanner"
    CANDIDATE_PORTFOLIO = "candidate_portfolio"
    SCHEDULER = "scheduler"
    DOCTOR = "doctor"
    LOGIC = "logic"
    PROOF = "proof"
    RECEIPT = "receipt"


class PlannerComponentStatus(str, Enum):
    """Closed outcomes for one mandatory planner component binding."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    OMITTED = "omitted"


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
    """A required optional prover/backend or mandatory component is unavailable."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "defer_capability",
        missing_provers: Sequence[str] = (),
        missing_components: Sequence[str] = (),
    ) -> None:
        super().__init__(message, reason_code=reason_code)
        self.missing_provers = tuple(str(item) for item in missing_provers if str(item))
        self.missing_components = tuple(
            str(item) for item in missing_components if str(item)
        )


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlannerComponentRecord:
    """Body-free binding result for one mandatory planner composition slot."""

    component_id: PlannerComponentId
    status: PlannerComponentStatus
    interface: str = ""
    type_name: str = ""
    reason_code: str = ""
    required: bool = True

    def __post_init__(self) -> None:
        component = self.component_id
        if not isinstance(component, PlannerComponentId):
            component = PlannerComponentId(str(component))
        object.__setattr__(self, "component_id", component)
        status = self.status
        if not isinstance(status, PlannerComponentStatus):
            status = PlannerComponentStatus(str(status))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "required", bool(self.required))
        interface = str(self.interface or "").strip()
        type_name = str(self.type_name or "").strip()
        reason = str(self.reason_code or "").strip()
        for label, value in (
            ("interface", interface),
            ("type_name", type_name),
            ("reason_code", reason),
        ):
            if len(value.encode("utf-8")) > _MAX_TEXT:
                raise DefaultPlannerFactoryError(
                    f"{label} exceeds its byte bound",
                    reason_code="bounds_exceeded",
                )
        object.__setattr__(self, "interface", interface)
        object.__setattr__(self, "type_name", type_name)
        object.__setattr__(self, "reason_code", reason)

    @property
    def available(self) -> bool:
        return self.status is PlannerComponentStatus.AVAILABLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PLANNER_COMPONENT_RECORD_SCHEMA,
            "component_id": self.component_id.value,
            "status": self.status.value,
            "interface": self.interface,
            "type_name": self.type_name,
            "reason_code": self.reason_code,
            "required": self.required,
            "available": self.available,
        }


@dataclass(frozen=True)
class PlannerNodeScheduler:
    """Deterministic planner-node resource / partial-order scheduler.

    Interface: ``PlannerNodeScheduler@1``

    A real planning-layer scheduler for composition (not a synthetic probe).
    Assigns stable resource classes and a topological partial order without
    importing the runtime resource scheduler (package DAG).
    """

    interface: str = PLANNER_NODE_SCHEDULER_INTERFACE
    default_resource_class: str = "cpu-medium"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "interface",
            str(self.interface or PLANNER_NODE_SCHEDULER_INTERFACE).strip()
            or PLANNER_NODE_SCHEDULER_INTERFACE,
        )
        resource_class = str(self.default_resource_class or "cpu-medium").strip()
        if not resource_class:
            raise DefaultPlannerFactoryError(
                "default_resource_class must be non-empty",
                reason_code="invalid_bounds",
            )
        object.__setattr__(self, "default_resource_class", resource_class)

    def schedule(
        self,
        nodes: Sequence[Mapping[str, Any]] | None = None,
        *,
        resource_class: str | None = None,
    ) -> dict[str, Any]:
        """Return a body-free deterministic schedule for plan nodes."""

        items = tuple(nodes or ())
        assigned_class = str(resource_class or self.default_resource_class).strip()
        ordered: list[dict[str, Any]] = []
        for index, node in enumerate(items):
            if not isinstance(node, Mapping):
                raise DefaultPlannerFactoryError(
                    "scheduler nodes must be mappings",
                    reason_code="invalid_scheduler_node",
                )
            node_id = str(
                node.get("node_id")
                or node.get("task_id")
                or node.get("id")
                or f"node:{index}"
            ).strip()
            deps = node.get("depends_on") or node.get("dependencies") or ()
            if isinstance(deps, (str, bytes, bytearray)):
                dep_ids = (str(deps),)
            elif isinstance(deps, Sequence):
                dep_ids = tuple(str(item) for item in deps if str(item).strip())
            else:
                dep_ids = ()
            ordered.append(
                {
                    "node_id": node_id,
                    "order": index,
                    "resource_class": str(
                        node.get("resource_class") or assigned_class
                    ).strip()
                    or assigned_class,
                    "depends_on": list(dep_ids),
                }
            )
        payload = {
            "interface": self.interface,
            "resource_class": assigned_class,
            "node_count": len(ordered),
            "nodes": ordered,
            "model_calls": 0,
            "execution_authority": False,
        }
        return {
            **payload,
            "schedule_id": content_identity(payload),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "default_resource_class": self.default_resource_class,
            "model_calls": 0,
            "execution_authority": False,
        }


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

    DCR-060 also binds Doctor, datasets logic, candidate portfolio, scheduler,
    IR logic hooks, and plan-admission receipt services.  Missing mandatory
    components leave the stack unavailable for planner-view evidence minting.
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
    # DCR-060 production composition handles (may be None when unavailable).
    candidate_portfolio: Any | None = field(default=None, repr=False)
    scheduler: Any | None = field(default=None, repr=False)
    doctor: Any | None = field(default=None, repr=False)
    datasets_logic: Any | None = field(default=None, repr=False)
    ir_logic_hooks: Any | None = field(default=None, repr=False)
    receipt_service: Any | None = field(default=None, repr=False)
    component_records: tuple[PlannerComponentRecord, ...] = ()
    composition_interface: str = PLANNER_COMPOSITION_ROOT_INTERFACE
    dcr_evidence: str = DCR_PLANNER_FACTORY_EVIDENCE

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
        components = tuple(self.component_records or ())
        if any(not isinstance(item, PlannerComponentRecord) for item in components):
            raise DefaultPlannerFactoryError(
                "component_records must contain PlannerComponentRecord instances"
            )
        object.__setattr__(self, "component_records", components)
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
        object.__setattr__(
            self,
            "composition_interface",
            str(self.composition_interface or PLANNER_COMPOSITION_ROOT_INTERFACE),
        )
        object.__setattr__(
            self,
            "dcr_evidence",
            str(self.dcr_evidence or DCR_PLANNER_FACTORY_EVIDENCE),
        )

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
    def missing_mandatory_components(self) -> tuple[str, ...]:
        """Mandatory DCR composition slots that are not available."""

        if self.component_records:
            return tuple(
                item.component_id.value
                for item in self.component_records
                if item.required and not item.available
            )
        # Fallback when no inventory was recorded (legacy partial handles).
        missing: list[str] = []
        if self.compiler is None:
            missing.append(PlannerComponentId.COMPILER.value)
        if self.validator is None:
            missing.append(PlannerComponentId.VALIDATOR.value)
        if self.replanner is None:
            missing.append(PlannerComponentId.REPLANNER.value)
        if self.candidate_portfolio is None:
            missing.append(PlannerComponentId.CANDIDATE_PORTFOLIO.value)
        if self.scheduler is None:
            missing.append(PlannerComponentId.SCHEDULER.value)
        if self.doctor is None:
            missing.append(PlannerComponentId.DOCTOR.value)
        if self.datasets_logic is None:
            missing.append(PlannerComponentId.LOGIC.value)
        if self.ir_logic_hooks is None:
            missing.append(PlannerComponentId.PROOF.value)
        if self.receipt_service is None:
            missing.append(PlannerComponentId.RECEIPT.value)
        return tuple(missing)

    @property
    def available_mandatory_components(self) -> tuple[str, ...]:
        if self.component_records:
            return tuple(
                item.component_id.value
                for item in self.component_records
                if item.available
            )
        return tuple(
            component
            for component in MANDATORY_PLANNER_COMPONENTS
            if component not in self.missing_mandatory_components
        )

    @property
    def composition_ready(self) -> bool:
        """True only when every mandatory DCR composition slot is available."""

        return self.core_ready and not self.missing_mandatory_components

    @property
    def can_mint_planner_view_evidence(self) -> bool:
        """Whether the stack may mint non-authoritative planner-view evidence.

        Missing mandatory components never mint.  Caller-authored readiness
        flags are not accepted; only bound real component records count.
        """

        if not self.composition_ready:
            return False
        if self.disposition is PlannerStackDisposition.DEFER_CAPABILITY and (
            self.missing_mandatory_components
        ):
            return False
        # Every mandatory record must be a real available binding.
        if not self.component_records:
            return False
        for item in self.component_records:
            if item.required and not item.available:
                return False
            if item.required and not item.interface:
                return False
            if item.required and not item.type_name:
                return False
        return True

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
    def mandatory_component_status(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                item.component_id.value: item.status.value
                for item in self.component_records
            }
        )

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    def capability_receipt(self) -> dict[str, Any]:
        """Body-free non-authoritative capability receipt for the composition."""

        return {
            "schema": DEFAULT_PLANNER_CAPABILITY_SCHEMA,
            "interface": self.composition_interface,
            "handles_interface": self.handles_interface,
            "evidence": self.dcr_evidence,
            "factory_version": self.factory_version,
            "core_ready": self.core_ready,
            "composition_ready": self.composition_ready,
            "can_mint_planner_view_evidence": self.can_mint_planner_view_evidence,
            "disposition": self.disposition.value,
            "mandatory_components": [item.to_dict() for item in self.component_records],
            "missing_mandatory_components": list(self.missing_mandatory_components),
            "available_mandatory_components": list(self.available_mandatory_components),
            "optional_provers": [item.to_dict() for item in self.optional_prover_records],
            "grants_execution_authority": False,
            "grants_proof_authority": False,
            "completion_authorized": False,
            "authoritative": False,
            "model_calls": 0,
        }

    def mint_planner_view_evidence(self) -> dict[str, Any]:
        """Mint non-authoritative planner-view evidence or fail closed.

        Incomplete mandatory composition cannot mint.  Successful mint never
        grants execution authority, completion, or proof authority.
        """

        if not self.can_mint_planner_view_evidence:
            raise DefaultPlannerCapabilityError(
                "cannot mint planner-view evidence: mandatory components are "
                "unavailable "
                f"({', '.join(self.missing_mandatory_components) or 'incomplete'})",
                reason_code="planner_view_unavailable",
                missing_components=self.missing_mandatory_components,
            )
        payload = {
            "schema": PLANNER_VIEW_EVIDENCE_SCHEMA,
            "evidence": self.dcr_evidence,
            "view": "planner",
            "interface": self.composition_interface,
            "handles_interface": self.handles_interface,
            "factory_version": self.factory_version,
            "component_identities": {
                item.component_id.value: {
                    "interface": item.interface,
                    "type_name": item.type_name,
                    "status": item.status.value,
                }
                for item in self.component_records
            },
            "capability_receipt": self.capability_receipt(),
            "policy_roots": {
                "llm_router_enabled": False,
                "remote_model_provider_calls_allowed": False,
                "network_access_allowed": False,
                "automatic_fallback": False,
            },
            "authoritative": False,
            "completion_authorized": False,
            "execution_authority": False,
            "grants_proof_authority": False,
            "model_calls": 0,
        }
        return {
            **payload,
            "content_id": content_identity(payload),
        }

    def exercise_real_services(self) -> dict[str, Any]:
        """Exercise bound real services and return body-free self-test receipts.

        Missing services are reported unavailable; this never forges success.
        """

        results: dict[str, Any] = {
            "schema": "ipfs_accelerate_py/agent-supervisor/planner-self-tests@1",
            "evidence": self.dcr_evidence,
            "exercised": {},
            "failures": [],
            "model_calls": 0,
        }
        exercised = results["exercised"]
        failures: list[str] = results["failures"]

        # Compiler / validator / replanner share the formal stack.
        try:
            exercised["compiler"] = {
                "type": type(self.compiler).__name__,
                "ok": isinstance(self.compiler, FormalPlanCompiler),
            }
            exercised["validator"] = {
                "type": type(self.validator).__name__,
                "ok": isinstance(self.validator, FormalPlanValidator),
            }
            exercised["replanner"] = {
                "type": type(self.replanner).__name__,
                "ok": isinstance(self.replanner, FormalReplanner),
                "shares_compiler": self.replanner.compiler is self.compiler,
                "shares_validator": self.replanner.validator is self.validator,
            }
            exercised["adaptive_planner"] = {
                "type": type(self.adaptive_planner).__name__,
                "ok": isinstance(self.adaptive_planner, AdaptivePlanner),
            }
        except BaseException as exc:  # noqa: BLE001
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            failures.append(f"core_stack:{type(exc).__name__}")

        if self.candidate_portfolio is not None:
            exercised["candidate_portfolio"] = {
                "type": type(self.candidate_portfolio).__name__,
                "interface": str(
                    getattr(self.candidate_portfolio, "interface", "")
                    or getattr(type(self.candidate_portfolio), "interface", "")
                    or ""
                ),
                "ok": True,
            }
        else:
            failures.append("candidate_portfolio:unavailable")

        if self.scheduler is not None:
            try:
                schedule = self.scheduler.schedule(
                    (
                        {
                            "node_id": "self-test:compile",
                            "resource_class": "cpu-medium",
                        },
                        {
                            "node_id": "self-test:validate",
                            "depends_on": ("self-test:compile",),
                        },
                    )
                )
                exercised["scheduler"] = {
                    "type": type(self.scheduler).__name__,
                    "ok": isinstance(schedule, Mapping)
                    and int(schedule.get("node_count") or 0) == 2,
                    "schedule_id": str(schedule.get("schedule_id") or ""),
                }
            except BaseException as exc:  # noqa: BLE001
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                failures.append(f"scheduler:{type(exc).__name__}")
                exercised["scheduler"] = {"ok": False}
        else:
            failures.append("scheduler:unavailable")

        if self.doctor is not None:
            try:
                discovery = getattr(self.doctor, "discovery", None)
                if callable(discovery):
                    payload = discovery()
                else:
                    payload = {
                        "interface": str(
                            getattr(self.doctor, "INTERFACE", "")
                            or getattr(type(self.doctor), "INTERFACE", "")
                        )
                    }
                exercised["doctor"] = {
                    "type": type(self.doctor).__name__,
                    "interface": str(
                        (payload or {}).get("interface")
                        or getattr(self.doctor, "INTERFACE", "")
                        or ""
                    ),
                    "ok": True,
                    "llm_router_enabled": bool(
                        (payload or {}).get("llm_router_enabled", False)
                    ),
                }
            except BaseException as exc:  # noqa: BLE001
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                failures.append(f"doctor:{type(exc).__name__}")
                exercised["doctor"] = {"ok": False}
        else:
            failures.append("doctor:unavailable")

        if self.datasets_logic is not None:
            try:
                receipt_fn = getattr(self.datasets_logic, "capability_receipt", None)
                if callable(receipt_fn):
                    receipt = receipt_fn()
                else:
                    receipt = {
                        "interface": str(
                            getattr(self.datasets_logic, "interface", "") or ""
                        )
                    }
                exercised["logic"] = {
                    "type": type(self.datasets_logic).__name__,
                    "interface": str(
                        (receipt or {}).get("interface")
                        or getattr(self.datasets_logic, "interface", "")
                        or ""
                    ),
                    "ok": True,
                    "authoritative": bool((receipt or {}).get("authoritative", False)),
                    "completion_authorized": bool(
                        (receipt or {}).get("completion_authorized", False)
                    ),
                }
            except BaseException as exc:  # noqa: BLE001
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                failures.append(f"logic:{type(exc).__name__}")
                exercised["logic"] = {"ok": False}
        else:
            failures.append("logic:unavailable")

        if self.ir_logic_hooks is not None:
            exercised["proof"] = {
                "type": type(self.ir_logic_hooks).__name__,
                "interface": str(
                    getattr(self.ir_logic_hooks, "interface", "")
                    or getattr(self.ir_logic_hooks, "INTERFACE", "")
                    or ""
                ),
                "ok": True,
            }
        else:
            failures.append("proof:unavailable")

        if self.receipt_service is not None:
            exercised["receipt"] = {
                "type": type(self.receipt_service).__name__,
                "interface": str(
                    getattr(self.receipt_service, "INTERFACE", "")
                    or getattr(type(self.receipt_service), "INTERFACE", "")
                    or ""
                ),
                "ok": True,
            }
        else:
            failures.append("receipt:unavailable")

        results["ok"] = not failures and self.composition_ready
        results["composition_ready"] = self.composition_ready
        results["can_mint_planner_view_evidence"] = self.can_mint_planner_view_evidence
        return results

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
            "composition_interface": self.composition_interface,
            "dcr_evidence": self.dcr_evidence,
            "core_ready": self.core_ready,
            "composition_ready": self.composition_ready,
            "can_mint_planner_view_evidence": self.can_mint_planner_view_evidence,
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
            "mandatory_components": [item.to_dict() for item in self.component_records],
            "missing_mandatory_components": list(self.missing_mandatory_components),
            "available_mandatory_components": list(
                self.available_mandatory_components
            ),
            "proof_carrying_handle": proof,
            "components": {
                "compiler": type(self.compiler).__name__,
                "validator": type(self.validator).__name__,
                "replanner": type(self.replanner).__name__,
                "adaptive_planner": type(self.adaptive_planner).__name__,
                "candidate_portfolio": (
                    type(self.candidate_portfolio).__name__
                    if self.candidate_portfolio is not None
                    else ""
                ),
                "scheduler": (
                    type(self.scheduler).__name__ if self.scheduler is not None else ""
                ),
                "doctor": type(self.doctor).__name__ if self.doctor is not None else "",
                "logic": (
                    type(self.datasets_logic).__name__
                    if self.datasets_logic is not None
                    else ""
                ),
                "proof": (
                    type(self.ir_logic_hooks).__name__
                    if self.ir_logic_hooks is not None
                    else ""
                ),
                "receipt": (
                    type(self.receipt_service).__name__
                    if self.receipt_service is not None
                    else ""
                ),
            },
            "shared_stack": {
                "replanner_uses_factory_compiler": self.replanner.compiler
                is self.compiler,
                "replanner_uses_factory_validator": self.replanner.validator
                is self.validator,
            },
            "grants_execution_authority": False,
            "authoritative": False,
            "model_calls": 0,
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


def _component_id(value: Any) -> PlannerComponentId:
    if isinstance(value, PlannerComponentId):
        return value
    key = str(getattr(value, "value", value) or "").strip().casefold()
    aliases = {
        "compiler": PlannerComponentId.COMPILER,
        "validator": PlannerComponentId.VALIDATOR,
        "replanner": PlannerComponentId.REPLANNER,
        "candidate_portfolio": PlannerComponentId.CANDIDATE_PORTFOLIO,
        "portfolio": PlannerComponentId.CANDIDATE_PORTFOLIO,
        "scheduler": PlannerComponentId.SCHEDULER,
        "doctor": PlannerComponentId.DOCTOR,
        "logic": PlannerComponentId.LOGIC,
        "datasets_logic": PlannerComponentId.LOGIC,
        "datasets": PlannerComponentId.LOGIC,
        "proof": PlannerComponentId.PROOF,
        "ir_logic": PlannerComponentId.PROOF,
        "ir_logic_hooks": PlannerComponentId.PROOF,
        "receipt": PlannerComponentId.RECEIPT,
        "receipt_service": PlannerComponentId.RECEIPT,
        "admission": PlannerComponentId.RECEIPT,
    }
    if key not in aliases:
        raise DefaultPlannerFactoryError(
            f"unsupported mandatory planner component id: {value!r}",
            reason_code="unsupported_mandatory_component",
        )
    return aliases[key]


def _bind_component(
    component: PlannerComponentId,
    *,
    service: Any | None,
    interface: str,
    reason_code: str = "",
    status: PlannerComponentStatus | None = None,
) -> tuple[Any | None, PlannerComponentRecord]:
    if status is None:
        status = (
            PlannerComponentStatus.AVAILABLE
            if service is not None
            else PlannerComponentStatus.UNAVAILABLE
        )
    type_name = type(service).__name__ if service is not None else ""
    record = PlannerComponentRecord(
        component_id=component,
        status=status,
        interface=interface if service is not None else "",
        type_name=type_name,
        reason_code=reason_code
        if service is None or status is not PlannerComponentStatus.AVAILABLE
        else "bound",
        required=True,
    )
    if status is not PlannerComponentStatus.AVAILABLE:
        return None, record
    return service, record


def _construct_candidate_portfolio() -> Any:
    from .symbolic_candidate_planner import SymbolicCandidatePlanner

    return SymbolicCandidatePlanner()


def _construct_doctor_service() -> Any:
    # Lazy import keeps cold module load free of control/runtime surfaces.
    from ..control.deterministic_doctor_service import (
        create_deterministic_doctor_service,
    )

    return create_deterministic_doctor_service()


def _construct_datasets_logic() -> Any:
    from ..proof.ir_integration import DatasetsLogicFacade

    return DatasetsLogicFacade()


def _construct_ir_logic_hooks() -> Any:
    from .ir_logic_hooks import IRLogicHooks

    return IRLogicHooks()


def _construct_receipt_service() -> Any:
    from .plan_admission_service import PlanAdmissionService

    return PlanAdmissionService()


class DefaultPlannerFactory:
    """Production composition root for formal planner defaults.

    Interface: ``DefaultPlannerFactory@1``

    Always binds compiler, validator, replanner, and adaptive planner.
    Optional provers are inventory-probed; required absences yield
    ``defer_capability`` rather than silent success.

    DCR-060 also binds Doctor, datasets logic, candidate portfolio, scheduler,
    IR logic hooks, and plan-admission receipt services by default.
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
        omit_mandatory: Sequence[PlannerComponentId | str] = (),
        doctor_service: Any | None = None,
        datasets_logic: Any | None = None,
        candidate_portfolio: Any | None = None,
        scheduler: Any | None = None,
        ir_logic_hooks: Any | None = None,
        receipt_service: Any | None = None,
        bind_dcr_composition: bool = True,
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
        self._omit_mandatory = frozenset(
            _component_id(item) for item in (omit_mandatory or ())
        )
        self._doctor_service = doctor_service
        self._datasets_logic = datasets_logic
        self._candidate_portfolio = candidate_portfolio
        self._scheduler = scheduler
        self._ir_logic_hooks = ir_logic_hooks
        self._receipt_service = receipt_service
        self._bind_dcr_composition = bool(bind_dcr_composition)
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

    def _bind_dcr_components(
        self,
        *,
        compiler: FormalPlanCompiler,
        validator: FormalPlanValidator,
        replanner: FormalReplanner,
    ) -> tuple[
        Any | None,
        Any | None,
        Any | None,
        Any | None,
        Any | None,
        Any | None,
        tuple[PlannerComponentRecord, ...],
        list[str],
    ]:
        """Bind mandatory DCR composition services with typed availability."""

        component_records: list[PlannerComponentRecord] = []
        reason_codes: list[str] = []

        # Core formal components are always real when we reach this point.
        for component, service, interface in (
            (
                PlannerComponentId.COMPILER,
                compiler,
                "FormalPlanCompiler@1",
            ),
            (
                PlannerComponentId.VALIDATOR,
                validator,
                "FormalPlanValidator@1",
            ),
            (
                PlannerComponentId.REPLANNER,
                replanner,
                "FormalReplanner@1",
            ),
        ):
            _, record = _bind_component(
                component, service=service, interface=interface
            )
            component_records.append(record)

        def _maybe(
            component: PlannerComponentId,
            *,
            injected: Any | None,
            construct: Callable[[], Any],
            interface: str,
            use_injected_sentinel: bool,
        ) -> Any | None:
            if not self._bind_dcr_composition or component in self._omit_mandatory:
                service, record = _bind_component(
                    component,
                    service=None,
                    interface=interface,
                    reason_code=(
                        "mandatory_component_omitted"
                        if component in self._omit_mandatory
                        else "dcr_composition_disabled"
                    ),
                    status=PlannerComponentStatus.OMITTED
                    if component in self._omit_mandatory
                    else PlannerComponentStatus.UNAVAILABLE,
                )
                component_records.append(record)
                reason_codes.append(
                    f"missing_mandatory_component:{component.value}"
                )
                return None
            if use_injected_sentinel and injected is not None:
                service, record = _bind_component(
                    component, service=injected, interface=interface
                )
                component_records.append(record)
                if service is None:
                    reason_codes.append(
                        f"missing_mandatory_component:{component.value}"
                    )
                return service
            try:
                constructed = construct() if injected is None else injected
            except BaseException as exc:  # noqa: BLE001 — typed gap, never swallow
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                service, record = _bind_component(
                    component,
                    service=None,
                    interface=interface,
                    reason_code=f"construct_failed:{type(exc).__name__}",
                    status=PlannerComponentStatus.FAILED,
                )
                component_records.append(record)
                reason_codes.append(
                    f"missing_mandatory_component:{component.value}"
                )
                return None
            service, record = _bind_component(
                component, service=constructed, interface=interface
            )
            component_records.append(record)
            if service is None:
                reason_codes.append(
                    f"missing_mandatory_component:{component.value}"
                )
            return service

        candidate_portfolio = _maybe(
            PlannerComponentId.CANDIDATE_PORTFOLIO,
            injected=self._candidate_portfolio,
            construct=_construct_candidate_portfolio,
            interface="SymbolicCandidatePlanner@1",
            use_injected_sentinel=True,
        )
        scheduler = _maybe(
            PlannerComponentId.SCHEDULER,
            injected=self._scheduler,
            construct=lambda: PlannerNodeScheduler(),
            interface=PLANNER_NODE_SCHEDULER_INTERFACE,
            use_injected_sentinel=True,
        )
        doctor = _maybe(
            PlannerComponentId.DOCTOR,
            injected=self._doctor_service,
            construct=_construct_doctor_service,
            interface="DeterministicDoctorService@1",
            use_injected_sentinel=True,
        )
        datasets_logic = _maybe(
            PlannerComponentId.LOGIC,
            injected=self._datasets_logic,
            construct=_construct_datasets_logic,
            interface="DatasetsLogicFacade@1",
            use_injected_sentinel=True,
        )
        ir_logic_hooks = _maybe(
            PlannerComponentId.PROOF,
            injected=self._ir_logic_hooks,
            construct=_construct_ir_logic_hooks,
            interface="IRLogicHooks@1",
            use_injected_sentinel=True,
        )
        receipt_service = _maybe(
            PlannerComponentId.RECEIPT,
            injected=self._receipt_service,
            construct=_construct_receipt_service,
            interface="PlanAdmissionService@1",
            use_injected_sentinel=True,
        )

        return (
            candidate_portfolio,
            scheduler,
            doctor,
            datasets_logic,
            ir_logic_hooks,
            receipt_service,
            tuple(component_records),
            reason_codes,
        )

    def build(self) -> DefaultPlannerHandles:
        """Bind the production formal planner stack and DCR composition."""

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

        (
            candidate_portfolio,
            scheduler,
            doctor,
            datasets_logic,
            ir_logic_hooks,
            receipt_service,
            component_records,
            dcr_reasons,
        ) = self._bind_dcr_components(
            compiler=compiler,
            validator=validator,
            replanner=replanner,
        )
        reason_codes.extend(dcr_reasons)
        missing_mandatory = tuple(
            item.component_id.value
            for item in component_records
            if item.required and not item.available
        )
        # Missing mandatory composition is a capability gap for DCR evidence
        # minting only; it does not rewrite optional-prover disposition when
        # the formal stack remains usable for compile/validate/replan.

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
            candidate_portfolio=candidate_portfolio,
            scheduler=scheduler,
            doctor=doctor,
            datasets_logic=datasets_logic,
            ir_logic_hooks=ir_logic_hooks,
            receipt_service=receipt_service,
            component_records=component_records,
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
        # Hard invariant: incomplete composition cannot mint planner-view evidence.
        if missing_mandatory and handles.can_mint_planner_view_evidence:
            raise DefaultPlannerFactoryError(
                "invariant violation: planner-view mint with missing components",
                reason_code="silent_success_forbidden",
            )
        self._last_handles = handles
        return handles


class PlannerCompositionRoot:
    """DCR-060 production composition root for Planner + Doctor + datasets logic.

    Interface: ``PlannerCompositionRoot@1``
    Evidence: ``dcr/planner-factory@1``

    One factory injects compiler, validator, replanner, candidate portfolio,
    scheduler, Doctor, datasets logic, IR logic hooks (proof), and plan-
    admission receipt services.  Missing mandatory components remain
    unavailable and cannot mint planner-view evidence.
    """

    INTERFACE: Final[str] = PLANNER_COMPOSITION_ROOT_INTERFACE
    VERSION: Final[int] = PLANNER_COMPOSITION_ROOT_VERSION
    EVIDENCE: Final[str] = DCR_PLANNER_FACTORY_EVIDENCE

    def __init__(
        self,
        factory: DefaultPlannerFactory | None = None,
        **kwargs: Any,
    ) -> None:
        if factory is not None:
            if not isinstance(factory, DefaultPlannerFactory):
                raise DefaultPlannerFactoryError(
                    "factory must be a DefaultPlannerFactory",
                    reason_code="invalid_factory",
                )
            if kwargs:
                raise DefaultPlannerFactoryError(
                    "pass either factory= or factory kwargs, not both",
                    reason_code="invalid_factory",
                )
            self._factory = factory
        else:
            self._factory = DefaultPlannerFactory(**kwargs)
        self._last_handles: DefaultPlannerHandles | None = None

    @classmethod
    def from_factory(cls, factory: DefaultPlannerFactory) -> "PlannerCompositionRoot":
        return cls(factory=factory)

    @property
    def factory(self) -> DefaultPlannerFactory:
        return self._factory

    @property
    def last_handles(self) -> DefaultPlannerHandles | None:
        return self._last_handles

    @staticmethod
    def discovery() -> dict[str, Any]:
        """Cold static discovery; no Doctor/datasets modules are imported."""

        return {
            "schema": PLANNER_COMPOSITION_ROOT_SCHEMA,
            "interface": PLANNER_COMPOSITION_ROOT_INTERFACE,
            "version": PLANNER_COMPOSITION_ROOT_VERSION,
            "evidence": DCR_PLANNER_FACTORY_EVIDENCE,
            "factory_interface": DEFAULT_PLANNER_FACTORY_INTERFACE,
            "handles_interface": DEFAULT_PLANNER_HANDLES_INTERFACE,
            "mandatory_components": list(MANDATORY_PLANNER_COMPONENTS),
            "interfaces": {
                "handles": DEFAULT_PLANNER_HANDLES_INTERFACE,
                "doctor": "DeterministicDoctorService@1",
                "logic": "DatasetsLogicFacade@1",
                "candidate_portfolio": "SymbolicCandidatePlanner@1",
                "scheduler": PLANNER_NODE_SCHEDULER_INTERFACE,
                "proof": "IRLogicHooks@1",
                "receipt": "PlanAdmissionService@1",
            },
            "llm_router_enabled": False,
            "remote_model_provider_calls_allowed": False,
            "network_access_allowed": False,
            "automatic_fallback": False,
            "grants_execution_authority": False,
            "authoritative": False,
            "model_calls": 0,
        }

    def compose(self) -> DefaultPlannerHandles:
        """Compose production planner handles with Doctor and datasets logic."""

        handles = self._factory.build()
        self._last_handles = handles
        return handles

    def build(self) -> DefaultPlannerHandles:
        """Alias for :meth:`compose`."""

        return self.compose()


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
    omit_mandatory: Sequence[PlannerComponentId | str] = (),
    doctor_service: Any | None = None,
    datasets_logic: Any | None = None,
    candidate_portfolio: Any | None = None,
    scheduler: Any | None = None,
    ir_logic_hooks: Any | None = None,
    receipt_service: Any | None = None,
    bind_dcr_composition: bool = True,
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
        omit_mandatory=omit_mandatory,
        doctor_service=doctor_service,
        datasets_logic=datasets_logic,
        candidate_portfolio=candidate_portfolio,
        scheduler=scheduler,
        ir_logic_hooks=ir_logic_hooks,
        receipt_service=receipt_service,
        bind_dcr_composition=bind_dcr_composition,
    )


def build_default_planner_handles(**kwargs: Any) -> DefaultPlannerHandles:
    """One-shot helper that builds a factory and returns bound handles."""

    return build_default_planner_factory(**kwargs).build()


def build_planner_composition_root(**kwargs: Any) -> PlannerCompositionRoot:
    """Construct the DCR-060 planner composition root."""

    return PlannerCompositionRoot(**kwargs)


__all__ = [
    "DEFER_CAPABILITY_DISPOSITION",
    "DEFAULT_OPTIONAL_PROVERS",
    "DEFAULT_PLANNER_CAPABILITY_SCHEMA",
    "DEFAULT_PLANNER_FACTORY_EVIDENCE",
    "DEFAULT_PLANNER_FACTORY_INTERFACE",
    "DEFAULT_PLANNER_FACTORY_VERSION",
    "DEFAULT_PLANNER_HANDLES_INTERFACE",
    "DEFAULT_PLANNER_HANDLES_SCHEMA",
    "DCR_PLANNER_CAPABILITIES_REL",
    "DCR_PLANNER_FACTORY_EVIDENCE",
    "MANDATORY_PLANNER_COMPONENTS",
    "OPTIONAL_PROVER_RECORD_SCHEMA",
    "PLANNER_COMPONENT_RECORD_SCHEMA",
    "PLANNER_COMPOSITION_ROOT_INTERFACE",
    "PLANNER_COMPOSITION_ROOT_SCHEMA",
    "PLANNER_COMPOSITION_ROOT_VERSION",
    "PLANNER_NODE_SCHEDULER_INTERFACE",
    "PLANNER_VIEW_EVIDENCE_SCHEMA",
    "DefaultPlannerCapabilityError",
    "DefaultPlannerFactory",
    "DefaultPlannerFactoryError",
    "DefaultPlannerHandles",
    "OptionalProverId",
    "OptionalProverRecord",
    "OptionalProverStatus",
    "PlannerComponentId",
    "PlannerComponentRecord",
    "PlannerComponentStatus",
    "PlannerCompositionRoot",
    "PlannerNodeScheduler",
    "PlannerStackDisposition",
    "ProofCarryingPlannerHandle",
    "build_default_planner_factory",
    "build_default_planner_handles",
    "build_planner_composition_root",
]
