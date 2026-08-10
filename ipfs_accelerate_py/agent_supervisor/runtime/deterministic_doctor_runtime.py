"""Lazy production composition root for the deterministic Doctor.

The control service intentionally owns transport-neutral request/policy
contracts.  This higher layer owns the filesystem-facing work needed by a
normal ``inspect --checkout-root`` invocation:

* resolve one exact checkout against an explicit allowlist;
* build the existing canonical planning-analysis snapshot;
* enumerate policy-admitted sources and configured submodules without
  importing target code;
* compile repository diagnostics and bridge them to the deterministic Doctor
  contracts; and
* expose every later Doctor stage through a lazy, capability-reporting backend
  factory.

Import, construction, and :meth:`discovery` are cold.  In particular they do
not construct ``PlanningAnalysisFactory`` (which owns an index directory),
start a provider/process, open a database, or inspect a checkout.  Stage
imports and factories run only after an operation requests that stage.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..analysis.deterministic_doctor_contracts import (
    DeterministicDoctorRunReceipt,
    DoctorEvidenceSnapshot,
    DoctorOperation,
    DoctorRepairDisposition,
)
from ..autonomous_repair.contracts import DeterministicRepairDisposition
from ..control.deterministic_doctor_service import (
    DeterministicDoctorService,
    DoctorOperationRequest,
    DoctorOperationResult,
    DoctorServiceCapabilityCode,
    DoctorServiceSafetyError,
    DoctorStageBackends,
    create_deterministic_doctor_service,
)
from ..proof.formal_verification_contracts import content_identity
from ..validation.deterministic_doctor_policy import DeterministicDoctorPolicy

DETERMINISTIC_DOCTOR_BACKEND_FACTORY_INTERFACE: Final[str] = (
    "DeterministicDoctorBackendFactory@1"
)
DETERMINISTIC_DOCTOR_RUNTIME_INTERFACE: Final[str] = "DeterministicDoctorRuntime@1"
DETERMINISTIC_DOCTOR_RUNTIME_DISCOVERY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/runtime-discovery@1"
)
DETERMINISTIC_DOCTOR_RUNTIME_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/runtime-report@1"
)
DETERMINISTIC_DOCTOR_EVIDENCE_BUNDLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/runtime-evidence@1"
)

# DCR-053: bounded Doctor termination at a proved fixed point or typed
# abstention.  Interface DoctorFixedPoint@1 reuses DeterministicRepairDisposition@1.
DOCTOR_FIXED_POINT_INTERFACE: Final[str] = "DoctorFixedPoint@1"
DOCTOR_FIXED_POINT_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/runtime-fixed-point@1"
)
DOCTOR_FIXED_POINT_OBSERVATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/runtime-fixed-point-observation@1"
)
DEFAULT_DOCTOR_FIXED_POINT_BOUND: Final[int] = 8
MAX_DOCTOR_FIXED_POINT_BOUND: Final[int] = 32
DETERMINISTIC_REPAIR_DISPOSITION_INTERFACE: Final[str] = (
    "DeterministicRepairDisposition@1"
)
DCR_DOCTOR_FIXED_POINT_EVIDENCE: Final[str] = "dcr/doctor-fixed-point@1"
DCR_DOCTOR_FIXED_POINT_VERSION: Final[int] = 1
DEFAULT_DOCTOR_FIXED_POINT_PATH: Final[str] = (
    "data/agent_supervisor/deterministic_contract_repair/doctor-fixed-point.json"
)
DOCTOR_FIXED_POINT_CATALOG_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-fixed-point-catalog@1"
)

# The deterministic snapshot contract intentionally bounds direct blob
# references.  The complete path ledger remains in PlanningAnalysisView; this
# is only the bounded parser batch projected into DoctorEvidenceSnapshot.
MAX_DIAGNOSTIC_SOURCE_PATHS: Final[int] = 256
MAX_DIAGNOSTIC_SOURCE_BYTES: Final[int] = 16 * 1024 * 1024
MAX_DIAGNOSTIC_TOTAL_BYTES: Final[int] = 256 * 1024 * 1024

_MODEL_MODULE_ROOTS: Final[frozenset[str]] = frozenset(
    {
        "anthropic",
        "llm_router",
        "openai",
        "torch",
        "transformers",
    }
)
_NETWORK_MODULE_ROOTS: Final[frozenset[str]] = frozenset(
    {"aiohttp", "httpx", "requests"}
)


class DeterministicDoctorRuntimeError(RuntimeError):
    """Base runtime composition or exact-evidence failure."""

    def __init__(self, reason_code: str, message: str = "") -> None:
        self.reason_code = str(reason_code or "runtime_error")
        super().__init__(str(message or reason_code))


class DeterministicDoctorRuntimeSafetyError(
    DoctorServiceSafetyError, DeterministicDoctorRuntimeError
):
    """A model/network/target-import route was attempted in deterministic mode."""

    def __init__(self, reason_code: str, message: str = "") -> None:
        self.reason_code = str(reason_code or "runtime_safety_error")
        DoctorServiceSafetyError.__init__(self, str(message or reason_code))


class DoctorRuntimeStageUnavailable(DeterministicDoctorRuntimeError):
    """One requested lazy stage is unavailable with an actionable remedy."""

    def __init__(
        self,
        stage: DoctorRuntimeStage | str,
        reason_code: str,
        remediation: str,
        *,
        cause: BaseException | None = None,
    ) -> None:
        self.stage = (
            stage.value if isinstance(stage, DoctorRuntimeStage) else str(stage)
        )
        self.remediation = str(remediation)
        message = (
            f"stage {self.stage!r} is unavailable ({reason_code}); "
            f"{self.remediation}"
        )
        super().__init__(reason_code, message)
        self.__cause__ = cause


class DoctorRuntimeStage(str, Enum):
    """Closed lazy production pipeline."""

    EVIDENCE = "evidence"
    DIAGNOSE = "diagnose"
    RETRIEVE = "retrieve"
    TACTICIAN = "tactician"
    PROOF = "proof"
    SYNTHESIS_PREVIEW = "synthesis_preview"
    IMPACT = "impact"
    TRANSACTION = "transaction"
    FIXED_POINT = "fixed_point"


_STAGE_ORDER: Final[tuple[DoctorRuntimeStage, ...]] = tuple(DoctorRuntimeStage)
_STAGE_INTERFACES: Final[Mapping[DoctorRuntimeStage, str]] = MappingProxyType(
    {
        DoctorRuntimeStage.EVIDENCE: "PlanningAnalysisFactory@1",
        DoctorRuntimeStage.DIAGNOSE: "DoctorRepositoryDiagnostics@1",
        DoctorRuntimeStage.RETRIEVE: "DoctorRepairCandidateRetriever@1",
        DoctorRuntimeStage.TACTICIAN: "DeterministicDoctorTactician@1",
        DoctorRuntimeStage.PROOF: "DeterministicDoctorHammer@1",
        DoctorRuntimeStage.SYNTHESIS_PREVIEW: "DeterministicDoctorSynthesizer@1",
        DoctorRuntimeStage.IMPACT: "DeterministicDoctorImpact@1",
        DoctorRuntimeStage.TRANSACTION: "DeterministicDoctorTransaction@1",
        DoctorRuntimeStage.FIXED_POINT: "DeterministicDoctorFixedPoint@1",
    }
)
_STAGE_REMEDIATIONS: Final[Mapping[DoctorRuntimeStage, str]] = MappingProxyType(
    {
        DoctorRuntimeStage.EVIDENCE: (
            "install the repository-analysis package and bind an exact checkout"
        ),
        DoctorRuntimeStage.DIAGNOSE: (
            "install the inert AST adapters or inject a diagnose stage factory"
        ),
        DoctorRuntimeStage.RETRIEVE: (
            "bind deterministic candidate signals or inject a retrieval factory"
        ),
        DoctorRuntimeStage.TACTICIAN: (
            "bind the local deterministic tactician; model routes are forbidden"
        ),
        DoctorRuntimeStage.PROOF: (
            "install an approved digest-bound prover/toolchain or remain report-only"
        ),
        DoctorRuntimeStage.SYNTHESIS_PREVIEW: (
            "bind an admitted operator proposal and proof before synthesis preview"
        ),
        DoctorRuntimeStage.IMPACT: (
            "bind current program-graph/consumer evidence before impact closure"
        ),
        DoctorRuntimeStage.TRANSACTION: (
            "bind a real sandbox applicator and a control-plane permit/effect adapter"
        ),
        DoctorRuntimeStage.FIXED_POINT: (
            "bind transaction output and independently produced fixed-point evidence"
        ),
    }
)

# Production composition (DCR-050) requires these stage backends to be real
# callables, never empty slots or deferred placeholders.
MANDATORY_PRODUCTION_BACKENDS: Final[tuple[str, ...]] = (
    "diagnose",
    "plan",
    "retrieve",
    "tactician",
    "proof",
    "transaction",
)

# Lazy stages that may remain deferred until typed inputs arrive.  Deferred
# production stages report unavailable / abstain — never successful completion.
# DCR-053 binds a real DoctorFixedPoint@1 controller, so fixed_point is no
# longer an empty deferred slot (it still never claims mutation success without
# residual-free observations).
OPTIONAL_DEFERRED_BACKENDS: Final[tuple[str, ...]] = (
    "synthesis",
    "impact",
    "explain",
)

# Closed terminal dispositions emitted by DoctorFixedPoint@1 (DCR-053).
# Non-terminal intermediate steps never leave this closed set as a final result.
_DOCTOR_FIXED_POINT_TERMINAL_DISPOSITIONS: Final[
    frozenset[DeterministicRepairDisposition]
] = frozenset(
    {
        DeterministicRepairDisposition.PROVED_VALID,
        DeterministicRepairDisposition.REFUTED_REPAIRABLE,
        DeterministicRepairDisposition.ABSTAIN_REVIEW,
        DeterministicRepairDisposition.DEFER_CAPABILITY,
    }
)


def _bounded_fixed_point_int(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int = MAX_DOCTOR_FIXED_POINT_BOUND,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DeterministicDoctorRuntimeError(
            "invalid_fixed_point_bound",
            f"{name} must be an integer",
        )
    if value < minimum or value > maximum:
        raise DeterministicDoctorRuntimeError(
            "invalid_fixed_point_bound",
            f"{name} must be in [{minimum}, {maximum}]",
        )
    return value


def _compact_fixed_point_id(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value.strip()
    else:
        text = str(value).strip()
    if required and not text:
        raise DeterministicDoctorRuntimeError(
            "invalid_fixed_point_observation",
            f"{name} is required",
        )
    if text and any(character.isspace() for character in text):
        raise DeterministicDoctorRuntimeError(
            "invalid_fixed_point_observation",
            f"{name} must be a compact opaque identifier",
        )
    return text


def _zero_invocation_counter(value: Any, name: str) -> int:
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value != 0:
        raise DeterministicDoctorRuntimeSafetyError(
            "model_route_forbidden",
            f"{name} must be exactly zero under DoctorFixedPoint@1",
        )
    return 0


@dataclass(frozen=True)
class DoctorFixedPointObservation:
    """One bounded Doctor iteration observation (body-free, content-addressed).

    ``transition_measure`` is a non-negative residual measure: ``0`` means the
    residual is closed (proved fixed point).  A non-increasing measure without a
    state-hash change is not progress.
    """

    state_hash: str
    transition_measure: int = 0
    progress_key: str = ""
    residual_finding_ids: tuple[str, ...] = ()
    receipt_root: str = ""
    capability_available: bool = True
    repairable: bool = False
    model_invocation_count: int = 0
    provider_invocation_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "state_hash",
            _compact_fixed_point_id(self.state_hash, "state_hash"),
        )
        object.__setattr__(
            self,
            "transition_measure",
            _bounded_fixed_point_int(
                self.transition_measure,
                "transition_measure",
                minimum=0,
                maximum=2**31 - 1,
            ),
        )
        object.__setattr__(
            self,
            "progress_key",
            _compact_fixed_point_id(
                self.progress_key or self.state_hash,
                "progress_key",
                required=True,
            ),
        )
        residuals = tuple(
            _compact_fixed_point_id(item, "residual_finding_ids")
            for item in tuple(self.residual_finding_ids or ())
            if str(item or "").strip()
        )
        object.__setattr__(self, "residual_finding_ids", residuals)
        object.__setattr__(
            self,
            "receipt_root",
            _compact_fixed_point_id(
                self.receipt_root, "receipt_root", required=False
            ),
        )
        object.__setattr__(
            self, "capability_available", bool(self.capability_available)
        )
        object.__setattr__(self, "repairable", bool(self.repairable))
        object.__setattr__(
            self,
            "model_invocation_count",
            _zero_invocation_counter(
                self.model_invocation_count, "model_invocation_count"
            ),
        )
        object.__setattr__(
            self,
            "provider_invocation_count",
            _zero_invocation_counter(
                self.provider_invocation_count, "provider_invocation_count"
            ),
        )

    @property
    def residual_free(self) -> bool:
        return (
            self.transition_measure == 0
            and not self.residual_finding_ids
        )

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DOCTOR_FIXED_POINT_OBSERVATION_SCHEMA,
            "interface": DOCTOR_FIXED_POINT_INTERFACE,
            "state_hash": self.state_hash,
            "transition_measure": self.transition_measure,
            "progress_key": self.progress_key,
            "residual_finding_ids": list(self.residual_finding_ids),
            "receipt_root": self.receipt_root,
            "capability_available": self.capability_available,
            "repairable": self.repairable,
            "model_invocation_count": 0,
            "provider_invocation_count": 0,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | DoctorFixedPointObservation) -> (
        "DoctorFixedPointObservation"
    ):
        if isinstance(payload, DoctorFixedPointObservation):
            return payload
        if not isinstance(payload, Mapping):
            raise DeterministicDoctorRuntimeError(
                "invalid_fixed_point_observation",
                "observation must be a mapping",
            )
        residuals = payload.get("residual_finding_ids") or ()
        if isinstance(residuals, str):
            residual_ids: tuple[str, ...] = (residuals,)
        else:
            residual_ids = tuple(str(item) for item in residuals)
        return cls(
            state_hash=str(payload.get("state_hash") or ""),
            transition_measure=int(payload.get("transition_measure") or 0),
            progress_key=str(payload.get("progress_key") or ""),
            residual_finding_ids=residual_ids,
            receipt_root=str(payload.get("receipt_root") or ""),
            capability_available=bool(
                payload["capability_available"]
                if "capability_available" in payload
                else True
            ),
            repairable=bool(payload.get("repairable") or False),
            model_invocation_count=int(payload.get("model_invocation_count") or 0),
            provider_invocation_count=int(
                payload.get("provider_invocation_count") or 0
            ),
        )


@dataclass(frozen=True)
class DoctorFixedPointResult:
    """One stable Doctor termination receipt (DCR-053).

    Terminal results never authorize model/provider routes.  Only
    ``proved_valid`` may claim completion; every other terminal disposition is a
    typed abstention / deferral / refutation.
    """

    disposition: DeterministicRepairDisposition
    terminal: bool
    iteration: int
    bound: int
    state_hashes: tuple[str, ...] = ()
    transition_measures: tuple[int, ...] = ()
    repeated_keys: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    receipt_roots: tuple[str, ...] = ()
    residual_finding_ids: tuple[str, ...] = ()
    model_invocation_count: int = 0
    provider_invocation_count: int = 0
    explanation: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, DeterministicRepairDisposition):
            raise DeterministicDoctorRuntimeError(
                "invalid_fixed_point_disposition",
                "disposition must be DeterministicRepairDisposition",
            )
        if self.disposition not in _DOCTOR_FIXED_POINT_TERMINAL_DISPOSITIONS:
            raise DeterministicDoctorRuntimeError(
                "invalid_fixed_point_disposition",
                f"disposition {self.disposition.value!r} is not a DoctorFixedPoint terminal",
            )
        object.__setattr__(self, "terminal", bool(self.terminal))
        object.__setattr__(
            self,
            "iteration",
            _bounded_fixed_point_int(
                self.iteration, "iteration", minimum=0, maximum=MAX_DOCTOR_FIXED_POINT_BOUND
            ),
        )
        object.__setattr__(
            self,
            "bound",
            _bounded_fixed_point_int(
                self.bound, "bound", minimum=1, maximum=MAX_DOCTOR_FIXED_POINT_BOUND
            ),
        )
        object.__setattr__(
            self,
            "model_invocation_count",
            _zero_invocation_counter(
                self.model_invocation_count, "model_invocation_count"
            ),
        )
        object.__setattr__(
            self,
            "provider_invocation_count",
            _zero_invocation_counter(
                self.provider_invocation_count, "provider_invocation_count"
            ),
        )
        object.__setattr__(
            self, "state_hashes", tuple(str(item) for item in self.state_hashes or ())
        )
        object.__setattr__(
            self,
            "transition_measures",
            tuple(int(item) for item in self.transition_measures or ()),
        )
        object.__setattr__(
            self, "repeated_keys", tuple(str(item) for item in self.repeated_keys or ())
        )
        object.__setattr__(
            self, "reason_codes", tuple(str(item) for item in self.reason_codes or ())
        )
        object.__setattr__(
            self, "receipt_roots", tuple(str(item) for item in self.receipt_roots or ())
        )
        object.__setattr__(
            self,
            "residual_finding_ids",
            tuple(str(item) for item in self.residual_finding_ids or ()),
        )
        object.__setattr__(self, "explanation", str(self.explanation or ""))
        if self.claims_completion and self.disposition is not (
            DeterministicRepairDisposition.PROVED_VALID
        ):
            raise DeterministicDoctorRuntimeError(
                "claims_completion_forbidden",
                "only proved_valid may claim completion",
            )
        if self.may_call_model:
            raise DeterministicDoctorRuntimeSafetyError(
                "model_route_forbidden",
                "DoctorFixedPoint forbids model routes on every disposition",
            )

    @property
    def claims_completion(self) -> bool:
        return (
            self.terminal
            and self.disposition is DeterministicRepairDisposition.PROVED_VALID
        )

    @property
    def may_call_model(self) -> bool:
        return False

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": DOCTOR_FIXED_POINT_RESULT_SCHEMA,
            "disposition": self.disposition.value,
            "terminal": self.terminal,
            "iteration": self.iteration,
            "bound": self.bound,
            "state_hashes": list(self.state_hashes),
            "transition_measures": list(self.transition_measures),
            "repeated_keys": list(self.repeated_keys),
            "reason_codes": list(self.reason_codes),
            "receipt_roots": list(self.receipt_roots),
            "residual_finding_ids": list(self.residual_finding_ids),
        }

    @property
    def content_id(self) -> str:
        return content_identity(self._identity_payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DOCTOR_FIXED_POINT_RESULT_SCHEMA,
            "interface": DOCTOR_FIXED_POINT_INTERFACE,
            "disposition_interface": DETERMINISTIC_REPAIR_DISPOSITION_INTERFACE,
            "disposition": self.disposition.value,
            "terminal": self.terminal,
            "iteration": self.iteration,
            "bound": self.bound,
            "state_hashes": list(self.state_hashes),
            "transition_measures": list(self.transition_measures),
            "repeated_keys": list(self.repeated_keys),
            "reason_codes": list(self.reason_codes),
            "receipt_roots": list(self.receipt_roots),
            "residual_finding_ids": list(self.residual_finding_ids),
            "model_invocation_count": 0,
            "provider_invocation_count": 0,
            "claims_completion": self.claims_completion,
            "may_call_model": False,
            "automatic_fallback": False,
            "explanation": self.explanation,
            "content_id": self.content_id,
        }


class NoProgressGuard:
    """Detect cycles and identical no-progress attempts for DoctorFixedPoint@1.

    Conflict policy (DCR-053): repeated identical findings/proposals never
    trigger free retry, a weaker gate, or a model fallback.  The guard records
    state hashes, transition measures, and progress keys, and emits one stable
    typed disposition when progress stalls.
    """

    def __init__(self, *, bound: int = DEFAULT_DOCTOR_FIXED_POINT_BOUND) -> None:
        self._bound = _bounded_fixed_point_int(
            bound, "bound", minimum=1, maximum=MAX_DOCTOR_FIXED_POINT_BOUND
        )
        self._state_hashes: list[str] = []
        self._transition_measures: list[int] = []
        self._progress_keys: list[str] = []
        self._receipt_roots: list[str] = []
        self._repeated_keys: list[str] = []
        self._lock = threading.RLock()

    @property
    def bound(self) -> int:
        return self._bound

    @property
    def iteration(self) -> int:
        with self._lock:
            return len(self._state_hashes)

    @property
    def state_hashes(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._state_hashes)

    @property
    def transition_measures(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(self._transition_measures)

    @property
    def progress_keys(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._progress_keys)

    @property
    def receipt_roots(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._receipt_roots)

    @property
    def repeated_keys(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._repeated_keys)

    def reset(self) -> None:
        with self._lock:
            self._state_hashes.clear()
            self._transition_measures.clear()
            self._progress_keys.clear()
            self._receipt_roots.clear()
            self._repeated_keys.clear()

    def observe(
        self, observation: DoctorFixedPointObservation | Mapping[str, Any]
    ) -> dict[str, Any]:
        """Record one observation and classify progress / no-progress / bound."""

        obs = DoctorFixedPointObservation.from_mapping(observation)
        with self._lock:
            previous_hash = self._state_hashes[-1] if self._state_hashes else ""
            previous_measure = (
                self._transition_measures[-1] if self._transition_measures else None
            )
            previous_key = self._progress_keys[-1] if self._progress_keys else ""

            self._state_hashes.append(obs.state_hash)
            self._transition_measures.append(obs.transition_measure)
            self._progress_keys.append(obs.progress_key)
            if obs.receipt_root:
                self._receipt_roots.append(obs.receipt_root)

            repeated_state = bool(previous_hash) and previous_hash == obs.state_hash
            repeated_key = bool(previous_key) and previous_key == obs.progress_key
            measure_non_improving = (
                previous_measure is not None
                and obs.transition_measure >= previous_measure
            )
            cycle = obs.state_hash in self._state_hashes[:-1]
            if repeated_state or (repeated_key and measure_non_improving) or cycle:
                key = obs.progress_key or obs.state_hash
                if key not in self._repeated_keys:
                    self._repeated_keys.append(key)

            bound_exhausted = len(self._state_hashes) >= self._bound
            no_progress = bool(self._repeated_keys) and (
                repeated_state
                or cycle
                or (repeated_key and measure_non_improving)
            )
            return {
                "iteration": len(self._state_hashes),
                "bound": self._bound,
                "bound_exhausted": bound_exhausted,
                "no_progress": no_progress,
                "repeated_state": repeated_state,
                "repeated_key": repeated_key,
                "cycle": cycle,
                "measure_non_improving": measure_non_improving,
                "state_hashes": tuple(self._state_hashes),
                "transition_measures": tuple(self._transition_measures),
                "repeated_keys": tuple(self._repeated_keys),
                "receipt_roots": tuple(self._receipt_roots),
                "observation": obs,
            }

    def stable_disposition(
        self,
        observation: DoctorFixedPointObservation,
        *,
        no_progress: bool,
        bound_exhausted: bool,
    ) -> tuple[DeterministicRepairDisposition, tuple[str, ...], str]:
        """Return one stable typed disposition for a terminal no-progress case."""

        reasons: list[str] = []
        if not observation.capability_available:
            reasons.append("capability_unavailable")
            reasons.append("defer_capability")
            return (
                DeterministicRepairDisposition.DEFER_CAPABILITY,
                tuple(reasons),
                "Doctor deferred: required capability is unavailable; no model fallback",
            )
        if observation.repairable:
            reasons.append("refuted_repairable")
            if no_progress:
                reasons.append("no_progress")
            if bound_exhausted:
                reasons.append("fixed_point_bound_exhausted")
            return (
                DeterministicRepairDisposition.REFUTED_REPAIRABLE,
                tuple(reasons),
                "Doctor refuted residual as repairable; free retry and model fallback denied",
            )
        if no_progress:
            reasons.append("no_progress")
        if bound_exhausted:
            reasons.append("fixed_point_bound_exhausted")
        if not reasons:
            reasons.append("abstain_review")
        reasons.append("typed_abstention")
        return (
            DeterministicRepairDisposition.ABSTAIN_REVIEW,
            tuple(dict.fromkeys(reasons)),
            "Doctor abstained after no-progress; one stable typed disposition, zero model calls",
        )


class DoctorFixedPoint:
    """Bounded Doctor termination controller (DoctorFixedPoint@1).

    Consumes content-addressed iteration observations and terminates within the
    configured bound at:

    * ``proved_valid`` — residual-free fixed point;
    * ``refuted_repairable`` — residual remains but is typed as repairable;
    * ``abstain_review`` — no-progress / bound exhaustion without repair path;
    * ``defer_capability`` — required capability is unavailable.

    Never starts a model or provider route.  Repeated identical observations
    collapse to one stable terminal disposition (see :class:`NoProgressGuard`).
    """

    INTERFACE: ClassVar[str] = DOCTOR_FIXED_POINT_INTERFACE

    def __init__(
        self,
        *,
        bound: int = DEFAULT_DOCTOR_FIXED_POINT_BOUND,
        no_progress_guard: NoProgressGuard | None = None,
    ) -> None:
        self._bound = _bounded_fixed_point_int(
            bound, "bound", minimum=1, maximum=MAX_DOCTOR_FIXED_POINT_BOUND
        )
        self._guard = no_progress_guard or NoProgressGuard(bound=self._bound)
        if self._guard.bound != self._bound:
            # Keep guard and controller bound identical; do not silently weaken.
            self._guard = NoProgressGuard(bound=self._bound)
        self._terminal: DoctorFixedPointResult | None = None
        self._lock = threading.RLock()

    @property
    def bound(self) -> int:
        return self._bound

    @property
    def guard(self) -> NoProgressGuard:
        return self._guard

    @property
    def terminal_result(self) -> DoctorFixedPointResult | None:
        with self._lock:
            return self._terminal

    def reset(self) -> None:
        with self._lock:
            self._guard.reset()
            self._terminal = None

    def step(
        self,
        observation: DoctorFixedPointObservation | Mapping[str, Any],
    ) -> DoctorFixedPointResult:
        """Advance one iteration or return the already-stable terminal result."""

        with self._lock:
            if self._terminal is not None:
                return self._terminal
            obs = DoctorFixedPointObservation.from_mapping(observation)
            progress = self._guard.observe(obs)
            result = self._classify(obs, progress)
            if result.terminal:
                self._terminal = result
            return result

    def run(
        self,
        observations: Sequence[DoctorFixedPointObservation | Mapping[str, Any]],
        *,
        reset: bool = True,
    ) -> DoctorFixedPointResult:
        """Consume a finite observation sequence and return the terminal result.

        Always terminates: either a residual-free fixed point, a typed
        no-progress disposition, or bound exhaustion.  Empty input is a stable
        typed abstention (never a model call).
        """

        with self._lock:
            if reset:
                self.reset()
            if self._terminal is not None:
                return self._terminal
            if not observations:
                result = DoctorFixedPointResult(
                    disposition=DeterministicRepairDisposition.ABSTAIN_REVIEW,
                    terminal=True,
                    iteration=0,
                    bound=self._bound,
                    reason_codes=("empty_observation_stream", "typed_abstention"),
                    explanation=(
                        "DoctorFixedPoint received no observations; typed abstention "
                        "with zero model/provider calls"
                    ),
                )
                self._terminal = result
                return result
            last: DoctorFixedPointResult | None = None
            for item in observations:
                last = self.step(item)
                if last.terminal:
                    return last
            # Defensive: step() always terminals by bound; if a caller supplied
            # fewer residual-bearing steps than the bound without closing, seal.
            assert last is not None
            if not last.terminal:
                sealed = DoctorFixedPointResult(
                    disposition=DeterministicRepairDisposition.ABSTAIN_REVIEW,
                    terminal=True,
                    iteration=last.iteration,
                    bound=self._bound,
                    state_hashes=last.state_hashes,
                    transition_measures=last.transition_measures,
                    repeated_keys=last.repeated_keys,
                    reason_codes=tuple(last.reason_codes)
                    + ("observation_stream_ended", "typed_abstention"),
                    receipt_roots=last.receipt_roots,
                    residual_finding_ids=last.residual_finding_ids,
                    explanation=(
                        "observation stream ended before residual-free fixed point"
                    ),
                )
                self._terminal = sealed
                return sealed
            return last

    def _classify(
        self,
        obs: DoctorFixedPointObservation,
        progress: Mapping[str, Any],
    ) -> DoctorFixedPointResult:
        iteration = int(progress["iteration"])
        bound = int(progress["bound"])
        state_hashes = tuple(progress["state_hashes"])
        transition_measures = tuple(progress["transition_measures"])
        repeated_keys = tuple(progress["repeated_keys"])
        receipt_roots = tuple(progress["receipt_roots"])
        no_progress = bool(progress["no_progress"])
        bound_exhausted = bool(progress["bound_exhausted"])

        if obs.residual_free:
            return DoctorFixedPointResult(
                disposition=DeterministicRepairDisposition.PROVED_VALID,
                terminal=True,
                iteration=iteration,
                bound=bound,
                state_hashes=state_hashes,
                transition_measures=transition_measures,
                repeated_keys=repeated_keys,
                reason_codes=("proved_valid", "fixed_point_reached"),
                receipt_roots=receipt_roots,
                residual_finding_ids=(),
                explanation="Doctor reached a residual-free proved fixed point",
            )

        if no_progress or bound_exhausted or not obs.capability_available:
            disposition, reasons, explanation = self._guard.stable_disposition(
                obs,
                no_progress=no_progress or bound_exhausted,
                bound_exhausted=bound_exhausted,
            )
            return DoctorFixedPointResult(
                disposition=disposition,
                terminal=True,
                iteration=iteration,
                bound=bound,
                state_hashes=state_hashes,
                transition_measures=transition_measures,
                repeated_keys=repeated_keys,
                reason_codes=reasons,
                receipt_roots=receipt_roots,
                residual_finding_ids=obs.residual_finding_ids,
                explanation=explanation,
            )

        # Intermediate residual that still has room under the bound.
        return DoctorFixedPointResult(
            disposition=DeterministicRepairDisposition.ABSTAIN_REVIEW,
            terminal=False,
            iteration=iteration,
            bound=bound,
            state_hashes=state_hashes,
            transition_measures=transition_measures,
            repeated_keys=repeated_keys,
            reason_codes=("iteration_open", "residual_present"),
            receipt_roots=receipt_roots,
            residual_finding_ids=obs.residual_finding_ids,
            explanation="Doctor iteration open; residual remains under bound",
        )


@dataclass(frozen=True)
class DoctorStageCapability:
    """Static/lazy capability state for one Doctor stage."""

    stage: DoctorRuntimeStage
    interface: str
    declared: bool = True
    loaded: bool = False
    available: bool | None = None
    reason_code: str = "not_probed"
    remediation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "interface": self.interface,
            "declared": self.declared,
            "loaded": self.loaded,
            "available": self.available,
            "reason_code": self.reason_code,
            "remediation": self.remediation,
        }


@dataclass(frozen=True)
class DoctorSourceInventoryEntry:
    """Body-free exact source enumeration entry."""

    path: str
    content_digest: str
    coverage_kind: str
    byte_count: int
    root_kind: str = "primary"
    git_object_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "content_digest": self.content_digest,
            "coverage_kind": self.coverage_kind,
            "byte_count": self.byte_count,
            "root_kind": self.root_kind,
            "git_object_id": self.git_object_id,
        }


@dataclass(frozen=True)
class DeterministicDoctorEvidenceBundle:
    """Exact checkout evidence plus the checked Doctor schema bridge."""

    checkout_root: str
    analysis_view: Any
    diagnostic_snapshot: Any
    snapshot: DoctorEvidenceSnapshot
    findings: tuple[Any, ...]
    source_inventory: tuple[DoctorSourceInventoryEntry, ...]
    diagnostic_source_paths: tuple[str, ...]
    submodule_closure: tuple[Mapping[str, Any], ...]
    bridge_id: str
    notes: tuple[str, ...] = ()

    @property
    def evidence_id(self) -> str:
        return content_identity(
            {
                "schema": DETERMINISTIC_DOCTOR_EVIDENCE_BUNDLE_SCHEMA,
                "checkout_root": self.checkout_root,
                "analysis_view_id": self.analysis_view.view_cid,
                "snapshot_id": self.snapshot.snapshot_id,
                "snapshot_content_id": self.snapshot.content_id,
                "source_inventory": [item.to_dict() for item in self.source_inventory],
                "submodule_closure": list(self.submodule_closure),
                "bridge_id": self.bridge_id,
                "notes": list(self.notes),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DETERMINISTIC_DOCTOR_EVIDENCE_BUNDLE_SCHEMA,
            "interface": DETERMINISTIC_DOCTOR_RUNTIME_INTERFACE,
            "evidence_id": self.evidence_id,
            "checkout_root": self.checkout_root,
            "analysis_view_id": self.analysis_view.view_cid,
            "analysis_completeness": self.analysis_view.completeness,
            "snapshot": self.snapshot.to_dict(),
            "finding_ids": [
                str(getattr(item, "finding_id", "") or getattr(item, "content_id", ""))
                for item in self.findings
            ],
            "source_inventory": [item.to_dict() for item in self.source_inventory],
            "diagnostic_source_paths": list(self.diagnostic_source_paths),
            "submodule_closure": [dict(item) for item in self.submodule_closure],
            "bridge_id": self.bridge_id,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class DeterministicDoctorRuntimeReport:
    """One service result enriched with body-free production stage evidence."""

    result: DoctorOperationResult
    evidence: DeterministicDoctorEvidenceBundle | None = None
    stage_receipts: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DETERMINISTIC_DOCTOR_RUNTIME_REPORT_SCHEMA,
            "interface": DETERMINISTIC_DOCTOR_RUNTIME_INTERFACE,
            "result": self.result.to_dict(),
            "evidence": self.evidence.to_dict() if self.evidence is not None else None,
            "stage_receipts": {
                str(key): dict(value)
                for key, value in sorted(self.stage_receipts.items())
            },
        }


def _import_symbol(module_name: str, symbol: str) -> Any:
    module = importlib.import_module(module_name)
    return getattr(module, symbol)


def _default_stage_loaders() -> dict[DoctorRuntimeStage, Callable[[], Any]]:
    """Return closures only; importing this module imports no stage provider."""

    return {
        DoctorRuntimeStage.EVIDENCE: lambda: _import_symbol(
            "ipfs_accelerate_py.agent_supervisor.analysis.planning_analysis_factory",
            "PlanningAnalysisFactory",
        ),
        DoctorRuntimeStage.DIAGNOSE: lambda: _import_symbol(
            "ipfs_accelerate_py.agent_supervisor.analysis.doctor_repository_diagnostics",
            "diagnose_repository",
        ),
        DoctorRuntimeStage.RETRIEVE: lambda: _import_symbol(
            "ipfs_accelerate_py.agent_supervisor.analysis.doctor_repair_candidate_retrieval",
            "DoctorRepairCandidateRetriever",
        ),
        DoctorRuntimeStage.TACTICIAN: lambda: _import_symbol(
            "ipfs_accelerate_py.agent_supervisor.planning.deterministic_doctor_tactician",
            "DeterministicDoctorTactician",
        ),
        DoctorRuntimeStage.PROOF: lambda: _import_symbol(
            "ipfs_accelerate_py.agent_supervisor.proof.deterministic_doctor_hammer",
            "DeterministicDoctorHammer",
        ),
        DoctorRuntimeStage.SYNTHESIS_PREVIEW: lambda: _import_symbol(
            "ipfs_accelerate_py.agent_supervisor.planning.deterministic_doctor_synthesis",
            "create_deterministic_doctor_synthesizer",
        ),
        DoctorRuntimeStage.IMPACT: lambda: _import_symbol(
            "ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_impact",
            "DeterministicDoctorImpactAnalyzer",
        ),
        DoctorRuntimeStage.TRANSACTION: lambda: _import_symbol(
            "ipfs_accelerate_py.agent_supervisor.planning.deterministic_doctor_transaction",
            "DeterministicDoctorTransaction",
        ),
        DoctorRuntimeStage.FIXED_POINT: lambda: _import_symbol(
            "ipfs_accelerate_py.agent_supervisor.validation.deterministic_doctor_fixed_point",
            "DeterministicDoctorFixedPointValidator",
        ),
    }


class DeterministicDoctorBackendFactory:
    """Thread-safe lazy stage registry with deterministic-route enforcement."""

    INTERFACE: ClassVar[str] = DETERMINISTIC_DOCTOR_BACKEND_FACTORY_INTERFACE

    def __init__(
        self,
        *,
        stage_factories: Mapping[
            DoctorRuntimeStage | str, Callable[[], Any] | Any
        ]
        | None = None,
        deterministic: bool = True,
    ) -> None:
        loaders = _default_stage_loaders()
        for raw_stage, loader in dict(stage_factories or {}).items():
            stage = (
                raw_stage
                if isinstance(raw_stage, DoctorRuntimeStage)
                else DoctorRuntimeStage(str(raw_stage))
            )
            loaders[stage] = loader if callable(loader) else lambda value=loader: value
        self._loaders = loaders
        self._instances: dict[DoctorRuntimeStage, Any] = {}
        self._failures: dict[DoctorRuntimeStage, DoctorRuntimeStageUnavailable] = {}
        self._deterministic = bool(deterministic)
        self._lock = threading.RLock()

    @staticmethod
    def discovery() -> dict[str, Any]:
        """Return the declared graph without importing or probing a stage."""

        return {
            "schema": DETERMINISTIC_DOCTOR_RUNTIME_DISCOVERY_SCHEMA,
            "interface": DETERMINISTIC_DOCTOR_BACKEND_FACTORY_INTERFACE,
            "runtime_interface": DETERMINISTIC_DOCTOR_RUNTIME_INTERFACE,
            "stages": [
                DoctorStageCapability(
                    stage=stage,
                    interface=_STAGE_INTERFACES[stage],
                    remediation=_STAGE_REMEDIATIONS[stage],
                ).to_dict()
                for stage in _STAGE_ORDER
            ],
            "deterministic": True,
            "model_routes_allowed": False,
            "network_routes_allowed": False,
            "providers_started": False,
            "processes_started": False,
            "database_opened": False,
        }

    def capabilities(self) -> tuple[DoctorStageCapability, ...]:
        """Return current state without causing a probe or load."""

        with self._lock:
            rows: list[DoctorStageCapability] = []
            for stage in _STAGE_ORDER:
                failure = self._failures.get(stage)
                loaded = stage in self._instances
                rows.append(
                    DoctorStageCapability(
                        stage=stage,
                        interface=_STAGE_INTERFACES[stage],
                        loaded=loaded,
                        available=False if failure else True if loaded else None,
                        reason_code=(
                            failure.reason_code
                            if failure is not None
                            else "available"
                            if loaded
                            else "not_probed"
                        ),
                        remediation=(
                            failure.remediation
                            if failure is not None
                            else _STAGE_REMEDIATIONS[stage]
                        ),
                    )
                )
            return tuple(rows)

    @property
    def loaded_stages(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(stage.value for stage in _STAGE_ORDER if stage in self._instances)

    def get(self, stage: DoctorRuntimeStage | str) -> Any:
        selected = (
            stage if isinstance(stage, DoctorRuntimeStage) else DoctorRuntimeStage(stage)
        )
        with self._lock:
            if selected in self._instances:
                return self._instances[selected]
            if selected in self._failures:
                raise self._failures[selected]
            loader = self._loaders.get(selected)
            if loader is None:
                failure = DoctorRuntimeStageUnavailable(
                    selected,
                    "stage_factory_missing",
                    _STAGE_REMEDIATIONS[selected],
                )
                self._failures[selected] = failure
                raise failure
            before_modules = frozenset(sys.modules)
            try:
                instance = loader()
                self._assert_safe_route(selected, instance, before_modules)
            except DeterministicDoctorRuntimeSafetyError:
                raise
            except Exception as exc:
                failure = DoctorRuntimeStageUnavailable(
                    selected,
                    "stage_dependency_unavailable",
                    _STAGE_REMEDIATIONS[selected],
                    cause=exc,
                )
                self._failures[selected] = failure
                raise failure from exc
            self._instances[selected] = instance
            return instance

    def _assert_safe_route(
        self,
        stage: DoctorRuntimeStage,
        instance: Any,
        before_modules: frozenset[str],
    ) -> None:
        if not self._deterministic:
            return
        if bool(getattr(instance, "uses_model", False)) or bool(
            getattr(instance, "model_route", False)
        ):
            raise DeterministicDoctorRuntimeSafetyError(
                "model_route_forbidden",
                f"stage {stage.value!r} declared a model route",
            )
        if bool(getattr(instance, "uses_network", False)) or bool(
            getattr(instance, "network_route", False)
        ):
            raise DeterministicDoctorRuntimeSafetyError(
                "network_route_forbidden",
                f"stage {stage.value!r} declared a network route",
            )
        added_roots = {
            name.split(".", 1)[0] for name in set(sys.modules).difference(before_modules)
        }
        model = sorted(added_roots & _MODEL_MODULE_ROOTS)
        network = sorted(added_roots & _NETWORK_MODULE_ROOTS)
        if model:
            raise DeterministicDoctorRuntimeSafetyError(
                "model_route_forbidden",
                f"stage {stage.value!r} loaded model modules: {model}",
            )
        if network:
            raise DeterministicDoctorRuntimeSafetyError(
                "network_route_forbidden",
                f"stage {stage.value!r} loaded network modules: {network}",
            )


def _canonical_checkout(
    checkout_root: str | os.PathLike[str],
    repository_allowlist: Sequence[str | os.PathLike[str]],
) -> Path:
    try:
        root = Path(checkout_root).expanduser().resolve(strict=True)
    except OSError as exc:
        raise DeterministicDoctorRuntimeError(
            "checkout_unavailable", f"checkout root is unavailable: {checkout_root}"
        ) from exc
    if not root.is_dir():
        raise DeterministicDoctorRuntimeError(
            "checkout_not_directory", "checkout root must be a directory"
        )
    allowed: set[Path] = set()
    for candidate in repository_allowlist:
        try:
            allowed.add(Path(candidate).expanduser().resolve(strict=True))
        except OSError as exc:
            raise DeterministicDoctorRuntimeError(
                "allowlist_root_unavailable",
                f"allowlisted repository root is unavailable: {candidate}",
            ) from exc
    if root not in allowed:
        raise DeterministicDoctorRuntimeError(
            "checkout_not_allowlisted",
            "checkout root is not one of the explicit exact repository roots",
        )
    return root


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _submodule_records(entries: Sequence[Any]) -> tuple[Mapping[str, Any], ...]:
    records: list[Mapping[str, Any]] = []

    def visit(entry: Any, prefix: str = "") -> None:
        local = str(getattr(entry, "path", "") or "")
        joined = str(PurePosixPath(prefix, local)) if prefix else local
        records.append(
            MappingProxyType(
                {
                    "path": joined,
                    "commit_id": str(getattr(entry, "commit_id", "") or ""),
                    "depth": int(getattr(entry, "depth", 0)),
                    "available": bool(getattr(entry, "available", False)),
                    "reason_code": str(
                        getattr(entry, "reason_code", "configured_submodule")
                    ),
                }
            )
        )
        for child in tuple(getattr(entry, "nested", ()) or ()):
            visit(child, joined)

    for root_entry in entries:
        visit(root_entry)
    return tuple(sorted(records, key=lambda item: str(item["path"])))


class DeterministicDoctorRuntime:
    """Lazy production deterministic-Doctor runtime for one exact checkout."""

    INTERFACE: ClassVar[str] = DETERMINISTIC_DOCTOR_RUNTIME_INTERFACE

    def __init__(
        self,
        *,
        checkout_root: str | os.PathLike[str],
        repository_allowlist: Sequence[str | os.PathLike[str]] | None = None,
        policy: DeterministicDoctorPolicy | Mapping[str, Any] | None = None,
        backend_factory: DeterministicDoctorBackendFactory | None = None,
        control_service: Any | None = None,
        receipt_store: Any | None = None,
        scope_policy: Any | None = None,
        index_root: str | os.PathLike[str] | None = None,
        deterministic: bool = True,
        fixed_point_bound: int | None = None,
        fixed_point: DoctorFixedPoint | None = None,
    ) -> None:
        allowlist = tuple(repository_allowlist or (checkout_root,))
        self.checkout_root = _canonical_checkout(checkout_root, allowlist)
        self.repository_allowlist = tuple(
            sorted(str(Path(item).expanduser().resolve(strict=True)) for item in allowlist)
        )
        self._policy = policy
        self._factory = backend_factory or DeterministicDoctorBackendFactory(
            deterministic=deterministic
        )
        self._control_service = control_service
        self._scope_policy = scope_policy
        self._index_root = str(Path(index_root).resolve(strict=False)) if index_root else None
        self._deterministic = bool(deterministic)
        self._analysis_factory: Any | None = None
        self._evidence: DeterministicDoctorEvidenceBundle | None = None
        self._stage_receipts: dict[str, Mapping[str, Any]] = {}
        self._lock = threading.RLock()
        bound = (
            DEFAULT_DOCTOR_FIXED_POINT_BOUND
            if fixed_point_bound is None
            else _bounded_fixed_point_int(
                fixed_point_bound,
                "fixed_point_bound",
                minimum=1,
                maximum=MAX_DOCTOR_FIXED_POINT_BOUND,
            )
        )
        if policy is not None:
            try:
                policy_bounds = getattr(policy, "resource_bounds", None)
                if policy_bounds is None and isinstance(policy, Mapping):
                    raw_bounds = policy.get("resource_bounds") or policy.get("limits")
                    if isinstance(raw_bounds, Mapping):
                        bound = _bounded_fixed_point_int(
                            int(
                                raw_bounds.get("max_fixed_point_iterations", bound)
                            ),
                            "fixed_point_bound",
                            minimum=1,
                            maximum=MAX_DOCTOR_FIXED_POINT_BOUND,
                        )
                elif policy_bounds is not None and fixed_point_bound is None:
                    bound = _bounded_fixed_point_int(
                        int(
                            getattr(
                                policy_bounds,
                                "max_fixed_point_iterations",
                                bound,
                            )
                        ),
                        "fixed_point_bound",
                        minimum=1,
                        maximum=MAX_DOCTOR_FIXED_POINT_BOUND,
                    )
            except (TypeError, ValueError, DeterministicDoctorRuntimeError):
                # Keep the explicit/default bound; policy may be a partial map.
                pass
        self._fixed_point = fixed_point or DoctorFixedPoint(bound=bound)
        # Production composition (DCR-050): mandatory backends are real stage
        # adapters bound at construction.  Optional later stages may defer with
        # typed unavailability — never empty slots or silent success.
        self._service: DeterministicDoctorService = create_deterministic_doctor_service(
            policy=policy,
            receipt_store=receipt_store,
            control_service=control_service,
            backends=DoctorStageBackends(
                diagnose=self._diagnose_backend,
                plan=self._plan_backend,
                retrieve=self._retrieve_backend,
                tactician=self._tactician_backend,
                proof=self._proof_backend,
                transaction=self._transaction_backend,
                synthesis=self._deferred_stage_backend(
                    DoctorRuntimeStage.SYNTHESIS_PREVIEW
                ),
                impact=self._deferred_stage_backend(DoctorRuntimeStage.IMPACT),
                fixed_point=self._fixed_point_backend,
            ),
        )
        self._composition_handles: Mapping[str, Any] | None = None

    @staticmethod
    def discovery() -> dict[str, Any]:
        """Cold static discovery; no checkout or provider is touched."""

        manifest = DeterministicDoctorBackendFactory.discovery()
        return {
            **manifest,
            "interface": DETERMINISTIC_DOCTOR_RUNTIME_INTERFACE,
            "backend_factory_interface": DETERMINISTIC_DOCTOR_BACKEND_FACTORY_INTERFACE,
            "service_interface": DeterministicDoctorService.INTERFACE,
        }

    @property
    def service(self) -> DeterministicDoctorService:
        return self._service

    @property
    def backend_factory(self) -> DeterministicDoctorBackendFactory:
        return self._factory

    @property
    def evidence(self) -> DeterministicDoctorEvidenceBundle | None:
        return self._evidence

    @property
    def composition_handles(self) -> Mapping[str, Any] | None:
        """Optional production composition handles attached by the factory."""

        return self._composition_handles

    @property
    def fixed_point(self) -> DoctorFixedPoint:
        """Bounded DoctorFixedPoint@1 termination controller (DCR-053)."""

        return self._fixed_point

    @property
    def no_progress_guard(self) -> NoProgressGuard:
        """Shared no-progress / cycle guard for the fixed-point controller."""

        return self._fixed_point.guard

    def evaluate_fixed_point(
        self,
        observations: Sequence[DoctorFixedPointObservation | Mapping[str, Any]]
        | DoctorFixedPointObservation
        | Mapping[str, Any]
        | None = None,
        *,
        reset: bool = True,
    ) -> DoctorFixedPointResult:
        """Terminate at a proved fixed point or one stable typed abstention.

        Always returns within the configured bound with zero model/provider
        invocations.  Empty or missing observations yield a typed abstention.
        """

        if observations is None:
            sequence: tuple[DoctorFixedPointObservation | Mapping[str, Any], ...] = ()
        elif isinstance(observations, DoctorFixedPointObservation):
            sequence = (observations,)
        elif isinstance(observations, Mapping):
            # A bare observation mapping (has state_hash) vs a list-like payload.
            if "state_hash" in observations and "observations" not in observations:
                sequence = (observations,)
            else:
                raw = observations.get("observations") or ()
                sequence = tuple(raw)  # type: ignore[arg-type]
        else:
            sequence = tuple(observations)
        result = self._fixed_point.run(sequence, reset=reset)
        self._stage_receipts[DoctorRuntimeStage.FIXED_POINT.value] = {
            "status": "completed" if result.claims_completion else "terminal",
            "reason_code": (
                result.reason_codes[0] if result.reason_codes else result.disposition.value
            ),
            "disposition": result.disposition.value,
            "terminal": result.terminal,
            "iteration": result.iteration,
            "bound": result.bound,
            "model_invocation_count": 0,
            "provider_invocation_count": 0,
            "claims_completion": result.claims_completion,
            "may_call_model": False,
            "content_id": result.content_id,
        }
        return result

    def attach_composition_handles(self, handles: Mapping[str, Any]) -> None:
        """Attach body-free production composition handles (idempotent)."""

        if not isinstance(handles, Mapping):
            raise DeterministicDoctorRuntimeError(
                "invalid_composition_handles",
                "composition handles must be a mapping",
            )
        self._composition_handles = MappingProxyType(dict(handles))

    def mandatory_backends_bound(self) -> tuple[str, ...]:
        """Return mandatory backends that are non-empty and non-deferred."""

        available = set(self._service.backends_available)
        bound: list[str] = []
        for name in MANDATORY_PRODUCTION_BACKENDS:
            backend = getattr(self._service._backends, name, None)  # noqa: SLF001
            if backend is None or name not in available:
                continue
            if bool(getattr(backend, "doctor_deferred_backend", False)):
                continue
            bound.append(name)
        return tuple(bound)

    def assert_mandatory_backends_production_ready(self) -> None:
        """Fail closed when any mandatory backend is empty or deferred."""

        bound = set(self.mandatory_backends_bound())
        missing = [name for name in MANDATORY_PRODUCTION_BACKENDS if name not in bound]
        if missing:
            raise DeterministicDoctorRuntimeError(
                "mandatory_backend_unavailable",
                "mandatory production backends are empty or deferred: "
                + ", ".join(missing),
            )

    def capability_graph(self) -> dict[str, Any]:
        """Report current lazy state without loading an unrequested stage."""

        return {
            "interface": DETERMINISTIC_DOCTOR_BACKEND_FACTORY_INTERFACE,
            "stages": [item.to_dict() for item in self._factory.capabilities()],
            "loaded_stages": list(self._factory.loaded_stages),
            "mandatory_backends": list(MANDATORY_PRODUCTION_BACKENDS),
            "mandatory_backends_bound": list(self.mandatory_backends_bound()),
            "optional_deferred_backends": list(OPTIONAL_DEFERRED_BACKENDS),
            "fixed_point_interface": DOCTOR_FIXED_POINT_INTERFACE,
            "fixed_point_bound": self._fixed_point.bound,
            "disposition_interface": DETERMINISTIC_REPAIR_DISPOSITION_INTERFACE,
            "providers_started": False,
            "network_routes_allowed": False,
            "model_routes_allowed": False,
        }

    def inspect(self, **kwargs: Any) -> DeterministicDoctorRuntimeReport:
        return self.execute(DoctorOperation.INSPECT.value, **kwargs)

    def plan(self, **kwargs: Any) -> DeterministicDoctorRuntimeReport:
        return self.execute(DoctorOperation.PLAN.value, **kwargs)

    def execute(
        self,
        request: DoctorOperationRequest | Mapping[str, Any] | str,
        **kwargs: Any,
    ) -> DeterministicDoctorRuntimeReport:
        """Execute through the shared control service with runtime-built evidence."""

        if isinstance(request, DoctorOperationRequest):
            payload = request.to_dict()
            payload.pop("schema", None)
            payload.pop("content_id", None)
            payload.update(kwargs)
        elif isinstance(request, Mapping):
            payload = {**dict(request), **kwargs}
        else:
            payload = {"operation": str(request), **kwargs}
        operation = str(payload.get("operation") or "")
        if self._deterministic and (
            payload.get("network_access")
            or payload.get("llm_router_invoked")
            or payload.get("remote_model_provider_invoked")
            or int(payload.get("model_invocation_count") or 0)
            or int(payload.get("provider_invocation_count") or 0)
        ):
            raise DeterministicDoctorRuntimeSafetyError(
                "deterministic_route_forbidden",
                "deterministic Doctor forbids model/provider/network routes",
            )

        # Report/plan operations can be invoked without caller-authored JSON.
        if operation in {
            DoctorOperation.INSPECT.value,
            DoctorOperation.EXPLAIN.value,
            DoctorOperation.PLAN.value,
            DoctorOperation.REPAIR.value,
        } and "snapshot" not in payload:
            evidence = self.build_evidence()
            payload["snapshot"] = evidence.snapshot.to_dict()
            payload.setdefault("roots", evidence.snapshot.roots.to_dict())
            payload.setdefault(
                "finding_ids",
                tuple(
                    str(
                        getattr(item, "finding_id", "")
                        or getattr(item, "content_id", "")
                    )
                    for item in evidence.findings
                ),
            )
        result = self._service.execute(payload)
        return DeterministicDoctorRuntimeReport(
            result=result,
            evidence=self._evidence,
            stage_receipts=MappingProxyType(dict(self._stage_receipts)),
        )

    def build_evidence(self, *, refresh: bool = False) -> DeterministicDoctorEvidenceBundle:
        """Build or return exact body-free evidence for the bound checkout."""

        with self._lock:
            if self._evidence is not None and not refresh:
                return self._evidence
            analysis_factory_class = self._factory.get(DoctorRuntimeStage.EVIDENCE)
            if self._analysis_factory is None:
                self._analysis_factory = analysis_factory_class(
                    repository_allowlist=self.repository_allowlist,
                    index_root=self._index_root,
                    scope_policy=self._scope_policy,
                    optional_providers={},
                    build_index=True,
                )
            view = self._analysis_factory.analyze(self.checkout_root)
            inventory = self._enumerate_sources(view)
            notes: list[str] = ["exact_checkout_stability_verified"]
            diagnostic_paths_list: list[str] = []
            diagnostic_bytes = 0
            bounded_sources = False
            for item in inventory:
                if item.coverage_kind not in {"semantic_ast", "structured_data"}:
                    continue
                if item.byte_count > MAX_DIAGNOSTIC_SOURCE_BYTES:
                    bounded_sources = True
                    continue
                if (
                    len(diagnostic_paths_list) >= MAX_DIAGNOSTIC_SOURCE_PATHS
                    or diagnostic_bytes + item.byte_count
                    > MAX_DIAGNOSTIC_TOTAL_BYTES
                ):
                    bounded_sources = True
                    continue
                diagnostic_paths_list.append(item.path)
                diagnostic_bytes += item.byte_count
            diagnostic_paths = tuple(diagnostic_paths_list)
            if bounded_sources:
                notes.append("diagnostic_source_bound_reached")
            diag_snapshot, snapshot, findings, bridge_id = self._compile_diagnostics(
                view, diagnostic_paths, inventory
            )
            closure = _submodule_records(view.submodule_closure)
            bundle = DeterministicDoctorEvidenceBundle(
                checkout_root=str(self.checkout_root),
                analysis_view=view,
                diagnostic_snapshot=diag_snapshot,
                snapshot=snapshot,
                findings=findings,
                source_inventory=inventory,
                diagnostic_source_paths=diagnostic_paths,
                submodule_closure=closure,
                bridge_id=bridge_id,
                notes=tuple(notes),
            )
            self._evidence = bundle
            self._stage_receipts[DoctorRuntimeStage.EVIDENCE.value] = {
                "status": "completed",
                "evidence_id": bundle.evidence_id,
                "analysis_view_id": view.view_cid,
                "source_count": len(inventory),
                "submodule_count": len(closure),
            }
            self._stage_receipts[DoctorRuntimeStage.DIAGNOSE.value] = {
                "status": "completed",
                "snapshot_id": snapshot.snapshot_id,
                "finding_count": len(findings),
                "bridge_id": bridge_id,
            }
            return bundle

    def _enumerate_sources(
        self, view: Any
    ) -> tuple[DoctorSourceInventoryEntry, ...]:
        from ..analysis.repository_snapshot import CoverageKind, EntryKind

        rows: dict[str, DoctorSourceInventoryEntry] = {}

        def add_snapshot(snapshot: Any, prefix: str, root_kind: str) -> None:
            for disposition in snapshot.dispositions:
                if disposition.kind is CoverageKind.EXCLUDED:
                    continue
                if disposition.entry_kind is not EntryKind.REGULAR:
                    continue
                relative = (
                    str(PurePosixPath(prefix, disposition.path))
                    if prefix
                    else disposition.path
                )
                candidate = self.checkout_root.joinpath(*PurePosixPath(relative).parts)
                try:
                    if candidate.is_symlink() or not candidate.is_file():
                        continue
                    payload = candidate.read_bytes()
                except OSError as exc:
                    raise DeterministicDoctorRuntimeError(
                        "source_became_unreadable",
                        f"admitted source became unreadable: {relative}",
                    ) from exc
                digest = _sha256(payload)
                if disposition.content_digest and digest != disposition.content_digest:
                    raise DeterministicDoctorRuntimeError(
                        "source_identity_mismatch",
                        f"admitted source changed after snapshot: {relative}",
                    )
                rows[relative] = DoctorSourceInventoryEntry(
                    path=relative,
                    content_digest=digest,
                    coverage_kind=disposition.kind.value,
                    byte_count=len(payload),
                    root_kind=root_kind,
                    git_object_id=disposition.git_object_id,
                )

        add_snapshot(view.sca_snapshot, "", "primary")

        # A primary snapshot records gitlinks but not child source ledgers.
        # Only recursively configured and materialized submodules are expanded.
        from ..analysis.repository_snapshot import build_repository_snapshot

        def add_submodule(entry: Any, prefix: str = "") -> None:
            local = str(getattr(entry, "path", "") or "")
            joined = str(PurePosixPath(prefix, local)) if prefix else local
            if bool(getattr(entry, "available", False)):
                child_root = self.checkout_root.joinpath(*PurePosixPath(joined).parts)
                try:
                    child_snapshot = build_repository_snapshot(
                        child_root,
                        scope_policy=self._analysis_factory.scope_policy,
                        allow_dirty_analysis=True,
                    )
                    add_snapshot(child_snapshot, joined, "submodule")
                except Exception as exc:
                    raise DeterministicDoctorRuntimeError(
                        "submodule_inventory_unavailable",
                        f"configured submodule could not be inventoried: {joined}",
                    ) from exc
            for child in tuple(getattr(entry, "nested", ()) or ()):
                add_submodule(child, joined)

        for root_entry in view.submodule_closure:
            add_submodule(root_entry)
        return tuple(rows[path] for path in sorted(rows))

    def _compile_diagnostics(
        self,
        view: Any,
        diagnostic_paths: Sequence[str],
        inventory: Sequence[DoctorSourceInventoryEntry],
    ) -> tuple[Any, DoctorEvidenceSnapshot, tuple[Any, ...], str]:
        diagnose = self._factory.get(DoctorRuntimeStage.DIAGNOSE)
        diagnostics_module = importlib.import_module(
            "ipfs_accelerate_py.agent_supervisor.analysis.doctor_repository_diagnostics"
        )
        roots = view.reasoning_snapshot.roots
        diag_roots = diagnostics_module.DoctorAuthorityRoots(
            repository_id=roots.repository_id,
            forest_id=roots.forest_id,
            tree_id=roots.tree_id,
            overlay_id=roots.overlay_id,
            file_root_id=view.sca_snapshot.snapshot_id,
            blob_root_id=content_identity(
                {
                    "snapshot_id": view.sca_snapshot.snapshot_id,
                    "paths": list(diagnostic_paths),
                }
            ),
            parser_id=roots.parser_root,
            config_id=roots.scope_policy_id,
            toolchain_id=roots.toolchain_root,
            policy_id=roots.policy_root,
            ast_index_id=roots.ast_root,
            symbol_index_id=roots.index_root,
            import_graph_id=roots.program_behavior_root,
            dependency_graph_id=roots.program_behavior_root,
            evidence_graph_id=view.view_cid,
            corpus_root_id=view.sca_snapshot.snapshot_id,
        )
        digest_by_path = {item.path: item.content_digest for item in inventory}
        byte_count_by_path = {item.path: item.byte_count for item in inventory}
        # Production composition (DCR-050): empty source bytes are unavailable,
        # not successful.  Load exact admitted bytes from the checkout inventory.
        source_units_list: list[Any] = []
        for path in diagnostic_paths:
            candidate = self.checkout_root.joinpath(*PurePosixPath(path).parts)
            try:
                if candidate.is_symlink() or not candidate.is_file():
                    raise DeterministicDoctorRuntimeError(
                        "source_became_unreadable",
                        f"admitted diagnostic source is not a regular file: {path}",
                    )
                payload = candidate.read_bytes()
            except OSError as exc:
                raise DeterministicDoctorRuntimeError(
                    "source_became_unreadable",
                    f"admitted diagnostic source became unreadable: {path}",
                ) from exc
            if not payload:
                # Empty bodies cannot establish production diagnostic evidence.
                raise DeterministicDoctorRuntimeError(
                    "empty_source_unavailable",
                    f"admitted diagnostic source has empty bytes: {path}",
                )
            digest = _sha256(payload)
            expected = digest_by_path.get(path, "")
            if expected and digest != expected:
                raise DeterministicDoctorRuntimeError(
                    "source_identity_mismatch",
                    f"admitted source changed after snapshot: {path}",
                )
            if byte_count_by_path.get(path, -1) not in {-1, len(payload)}:
                raise DeterministicDoctorRuntimeError(
                    "source_identity_mismatch",
                    f"admitted source size drifted after inventory: {path}",
                )
            source_units_list.append(
                diagnostics_module.DoctorSourceUnit(
                    path=path,
                    source_bytes=payload,
                    blob_identity=digest,
                )
            )
        source_units = tuple(source_units_list)
        diag_snapshot = diagnose(
            sources=source_units,
            repository_root=str(self.checkout_root),
            authority_roots=diag_roots,
            policy={
                "max_paths": max(1, len(source_units)),
                "max_source_bytes": MAX_DIAGNOSTIC_SOURCE_BYTES,
                "max_total_bytes": MAX_DIAGNOSTIC_TOTAL_BYTES,
                "open_frontiers": tuple(view.open_frontier_ids),
            },
            claimed_tree_id=roots.tree_id,
        )
        bridge_class = _import_symbol(
            "ipfs_accelerate_py.agent_supervisor.analysis.doctor_contract_adapters",
            "DiagnosisObligationBridge",
        )
        bridge = bridge_class.from_diagnostic_snapshot(
            diag_snapshot,
            require_repository_id=roots.repository_id,
            notes=("runtime_exact_checkout",),
        )
        if bridge.snapshot_bridge is None:  # pragma: no cover - constructor invariant
            raise DeterministicDoctorRuntimeError(
                "snapshot_bridge_missing", "diagnostic bridge omitted the snapshot"
            )
        snapshot = bridge.snapshot_bridge.materialize_deterministic()
        findings = bridge.snapshot_bridge.materialize_finding_deterministics()
        return diag_snapshot, snapshot, findings, bridge.content_id

    def _diagnose_backend(
        self,
        request: DoctorOperationRequest,
        *,
        policy: DeterministicDoctorPolicy,
        policy_decision: Any,
    ) -> DeterministicDoctorRunReceipt:
        del policy_decision
        evidence = self.build_evidence()
        snapshot = evidence.snapshot
        return DeterministicDoctorRunReceipt(
            roots=snapshot.roots,
            receipt_id=content_identity(
                {
                    "runtime": self.INTERFACE,
                    "operation": DoctorOperation.INSPECT.value,
                    "request_id": request.request_id,
                    "evidence_id": evidence.evidence_id,
                }
            ),
            operation=DoctorOperation.INSPECT,
            mode=request.mode,
            disposition=DoctorRepairDisposition.SUPPORTED,
            snapshot_id=snapshot.snapshot_id,
            incident_id=request.incident_cid(),
            network_denied=True,
            secrets_inherited=False,
            reason_codes=(
                "runtime_exact_evidence",
                "optional_providers_not_required",
            ),
            invalidation_refs=snapshot.invalidation_refs,
            resource_bounds=policy.resource_bounds,
        )

    def _plan_backend(
        self,
        request: DoctorOperationRequest,
        *,
        policy: DeterministicDoctorPolicy,
        policy_decision: Any,
    ) -> DoctorOperationResult:
        """Lazily wire analytical planning stages, then abstain if inputs are open."""

        del policy
        evidence = self.build_evidence()
        receipts: dict[str, Mapping[str, Any]] = {}
        unavailable: list[str] = []
        for stage in (
            DoctorRuntimeStage.RETRIEVE,
            DoctorRuntimeStage.TACTICIAN,
            DoctorRuntimeStage.PROOF,
            DoctorRuntimeStage.SYNTHESIS_PREVIEW,
            DoctorRuntimeStage.IMPACT,
        ):
            try:
                self._factory.get(stage)
                receipts[stage.value] = {
                    "status": "wired",
                    "reason_code": "awaiting_typed_stage_inputs",
                    "remediation": _STAGE_REMEDIATIONS[stage],
                }
            except DoctorRuntimeStageUnavailable as exc:
                unavailable.append(stage.value)
                receipts[stage.value] = {
                    "status": "unavailable",
                    "reason_code": exc.reason_code,
                    "remediation": exc.remediation,
                }
        self._stage_receipts.update(receipts)
        reasons = (
            DoctorServiceCapabilityCode.STAGE_BACKEND_MISSING.value,
            "plan_inputs_deferred",
            *(f"stage_unavailable:{name}" for name in unavailable),
        )
        return DoctorOperationResult(
            request_id=request.request_id,
            operation=DoctorOperation.PLAN.value,
            mode=request.mode,
            disposition=DoctorRepairDisposition.ABSTAIN,
            incident_id=request.incident_cid(),
            read_only=True,
            policy_decision=policy_decision,
            reason_codes=reasons,
            explanation=(
                "planning stages are lazily wired; typed retrieval/proof/operator "
                "inputs are required before an admitted plan can be materialized"
            ),
            changed=False,
            status={
                "snapshot_id": evidence.snapshot.snapshot_id,
                "capability_graph": self.capability_graph(),
                "automatic_fallback": False,
            },
            stage_refs={
                name: str(value.get("reason_code", ""))
                for name, value in receipts.items()
            },
        )

    def _deferred_stage_backend(
        self, stage: DoctorRuntimeStage
    ) -> Callable[..., DoctorOperationResult]:
        def backend(
            request: DoctorOperationRequest,
            *,
            policy: DeterministicDoctorPolicy,
            policy_decision: Any,
        ) -> DoctorOperationResult:
            del policy
            try:
                self._factory.get(stage)
                # Typed inputs still open: unavailable, never successful.
                reason = "stage_unavailable_awaiting_typed_inputs"
                remediation = _STAGE_REMEDIATIONS[stage]
            except DoctorRuntimeStageUnavailable as exc:
                reason = exc.reason_code
                remediation = exc.remediation
            self._stage_receipts[stage.value] = {
                "status": "unavailable",
                "reason_code": reason,
                "remediation": remediation,
            }
            return DoctorOperationResult(
                request_id=request.request_id,
                operation=request.operation,
                mode=request.mode,
                disposition=DoctorRepairDisposition.ABSTAIN,
                incident_id=request.incident_cid(),
                read_only=request.is_read_only,
                policy_decision=policy_decision,
                reason_codes=(
                    DoctorServiceCapabilityCode.CAPABILITY_UNAVAILABLE.value,
                    reason,
                ),
                explanation=(
                    f"{stage.value} unavailable (not successful): {remediation}"
                ),
                changed=False,
                status={
                    "stage": stage.value,
                    "automatic_fallback": False,
                    "production_success": False,
                },
                stage_refs={stage.value: reason},
            )

        # Mark optional deferred adapters so production composition can reject
        # them as mandatory-backend candidates.
        setattr(backend, "doctor_deferred_backend", True)
        setattr(backend, "doctor_stage_name", stage.value)
        return backend

    def _mandatory_stage_backend(
        self, stage: DoctorRuntimeStage
    ) -> Callable[..., DoctorOperationResult]:
        """Bind a mandatory stage: wire the class, abstain only on open inputs."""

        def backend(
            request: DoctorOperationRequest,
            *,
            policy: DeterministicDoctorPolicy,
            policy_decision: Any,
        ) -> DoctorOperationResult:
            del policy
            try:
                self._factory.get(stage)
                reason = "awaiting_typed_stage_inputs"
                remediation = _STAGE_REMEDIATIONS[stage]
                status = "wired"
            except DoctorRuntimeStageUnavailable as exc:
                # Dependency gap is typed unavailability, never empty success.
                reason = exc.reason_code
                remediation = exc.remediation
                status = "unavailable"
            self._stage_receipts[stage.value] = {
                "status": status,
                "reason_code": reason,
                "remediation": remediation,
            }
            return DoctorOperationResult(
                request_id=request.request_id,
                operation=request.operation,
                mode=request.mode,
                disposition=DoctorRepairDisposition.ABSTAIN,
                incident_id=request.incident_cid(),
                read_only=request.is_read_only,
                policy_decision=policy_decision,
                reason_codes=(
                    DoctorServiceCapabilityCode.CAPABILITY_UNAVAILABLE.value
                    if status == "unavailable"
                    else DoctorServiceCapabilityCode.STAGE_BACKEND_MISSING.value,
                    reason,
                ),
                explanation=(
                    f"{stage.value} bound; {remediation}"
                    if status == "wired"
                    else f"{stage.value} unavailable: {remediation}"
                ),
                changed=False,
                status={
                    "stage": stage.value,
                    "automatic_fallback": False,
                    "production_success": False,
                    "mandatory": True,
                    "deferred": False,
                },
                stage_refs={stage.value: reason},
            )

        setattr(backend, "doctor_deferred_backend", False)
        setattr(backend, "doctor_stage_name", stage.value)
        setattr(backend, "doctor_mandatory_backend", True)
        return backend

    def _retrieve_backend(
        self,
        request: DoctorOperationRequest,
        *,
        policy: DeterministicDoctorPolicy,
        policy_decision: Any,
    ) -> DoctorOperationResult:
        return self._mandatory_stage_backend(DoctorRuntimeStage.RETRIEVE)(
            request, policy=policy, policy_decision=policy_decision
        )

    def _tactician_backend(
        self,
        request: DoctorOperationRequest,
        *,
        policy: DeterministicDoctorPolicy,
        policy_decision: Any,
    ) -> DoctorOperationResult:
        return self._mandatory_stage_backend(DoctorRuntimeStage.TACTICIAN)(
            request, policy=policy, policy_decision=policy_decision
        )

    def _proof_backend(
        self,
        request: DoctorOperationRequest,
        *,
        policy: DeterministicDoctorPolicy,
        policy_decision: Any,
    ) -> DoctorOperationResult:
        return self._mandatory_stage_backend(DoctorRuntimeStage.PROOF)(
            request, policy=policy, policy_decision=policy_decision
        )

    def _transaction_backend(
        self,
        request: DoctorOperationRequest,
        *,
        policy: DeterministicDoctorPolicy,
        policy_decision: Any,
    ) -> DoctorOperationResult:
        # Loading the class proves wiring.  Mutation still needs real adapters
        # and the service's control dependency — never claim success without them.
        del policy
        try:
            self._factory.get(DoctorRuntimeStage.TRANSACTION)
            wired = True
            reason = "awaiting_typed_stage_inputs"
            remediation = _STAGE_REMEDIATIONS[DoctorRuntimeStage.TRANSACTION]
        except DoctorRuntimeStageUnavailable as exc:
            wired = False
            reason = exc.reason_code
            remediation = exc.remediation
        if self._control_service is None and wired:
            reason = "control_service_required"
            remediation = (
                "bind a control-plane permit/effect adapter before transaction apply"
            )
            wired = False
        status = "wired" if wired else "unavailable"
        self._stage_receipts[DoctorRuntimeStage.TRANSACTION.value] = {
            "status": status,
            "reason_code": reason,
            "remediation": remediation,
        }
        return DoctorOperationResult(
            request_id=request.request_id,
            operation=request.operation,
            mode=request.mode,
            disposition=DoctorRepairDisposition.ABSTAIN,
            incident_id=request.incident_cid(),
            read_only=request.is_read_only,
            policy_decision=policy_decision,
            reason_codes=(
                DoctorServiceCapabilityCode.CAPABILITY_UNAVAILABLE.value,
                reason,
            ),
            explanation=f"transaction {status}: {remediation}",
            changed=False,
            status={
                "stage": DoctorRuntimeStage.TRANSACTION.value,
                "automatic_fallback": False,
                "production_success": False,
                "mandatory": True,
                "deferred": False,
            },
            stage_refs={DoctorRuntimeStage.TRANSACTION.value: reason},
        )

    def _fixed_point_backend(
        self,
        request: DoctorOperationRequest,
        *,
        policy: DeterministicDoctorPolicy,
        policy_decision: Any,
    ) -> DoctorOperationResult:
        """Bound DoctorFixedPoint@1 termination (DCR-053).

        When the request carries body-free fixed-point observations under
        ``context['fixed_point_observations']`` / ``context['observations']``,
        evaluate them through :meth:`evaluate_fixed_point`.  Otherwise return a
        typed abstention: the stage is wired and zero-model, never a free
        retry or provider fallback.
        """

        del policy, policy_decision
        context = getattr(request, "context", None)
        observations: Sequence[Any] | None = None
        if isinstance(context, Mapping):
            raw = context.get("fixed_point_observations")
            if raw is None:
                raw = context.get("observations")
            if isinstance(raw, (list, tuple)):
                observations = tuple(raw)
            elif isinstance(raw, Mapping) or isinstance(
                raw, DoctorFixedPointObservation
            ):
                observations = (raw,)
        if observations is None:
            result = self.evaluate_fixed_point(())
            reason = "awaiting_typed_stage_inputs"
            explanation = (
                "fixed_point wired; supply content-addressed observations to "
                "terminate at proved_valid or a stable typed abstention"
            )
        else:
            result = self.evaluate_fixed_point(observations)
            reason = (
                result.reason_codes[0]
                if result.reason_codes
                else result.disposition.value
            )
            explanation = result.explanation or (
                f"fixed_point terminal: {result.disposition.value}"
            )
        # Map public DeterministicRepairDisposition onto DoctorRepairDisposition
        # for the service-layer envelope without claiming mutation success.
        if result.disposition is DeterministicRepairDisposition.PROVED_VALID:
            doctor_disposition = DoctorRepairDisposition.SUPPORTED
        elif result.disposition is DeterministicRepairDisposition.DEFER_CAPABILITY:
            doctor_disposition = DoctorRepairDisposition.ABSTAIN
        else:
            doctor_disposition = DoctorRepairDisposition.ABSTAIN
        return DoctorOperationResult(
            request_id=request.request_id,
            operation=request.operation,
            mode=request.mode,
            disposition=doctor_disposition,
            incident_id=request.incident_cid(),
            read_only=True,
            policy_decision=None,
            reason_codes=(
                result.disposition.value,
                *result.reason_codes,
                "runtime_model_calls_0",
            ),
            explanation=explanation,
            changed=False,
            status={
                "stage": DoctorRuntimeStage.FIXED_POINT.value,
                "automatic_fallback": False,
                "production_success": result.claims_completion,
                "mandatory": False,
                "deferred": False,
                "terminal": result.terminal,
                "iteration": result.iteration,
                "bound": result.bound,
                "content_id": result.content_id,
            },
            stage_refs={DoctorRuntimeStage.FIXED_POINT.value: reason},
        )


def create_deterministic_doctor_runtime(
    checkout_root: str | os.PathLike[str],
    **kwargs: Any,
) -> DeterministicDoctorRuntime:
    """Create a cold runtime bound to one explicit checkout."""

    return DeterministicDoctorRuntime(checkout_root=checkout_root, **kwargs)


def materialize_doctor_fixed_point(
    *,
    observations: Sequence[DoctorFixedPointObservation | Mapping[str, Any]]
    | None = None,
    bound: int = DEFAULT_DOCTOR_FIXED_POINT_BOUND,
    destination: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Materialize doctor-fixed-point.json evidence for DCR-053."""

    controller = DoctorFixedPoint(bound=bound)
    if observations is None:
        # Default fixture: residual-free fixed point (proved_valid).
        observations = (
            DoctorFixedPointObservation(
                state_hash="state:dcr053-fixed",
                transition_measure=0,
                progress_key="key:dcr053-fixed",
                receipt_root="receipt:dcr053-fixed",
            ),
        )
    result = controller.run(observations)
    payload = {
        "schema": DOCTOR_FIXED_POINT_CATALOG_SCHEMA,
        "interface": DOCTOR_FIXED_POINT_INTERFACE,
        "disposition_interface": DETERMINISTIC_REPAIR_DISPOSITION_INTERFACE,
        "evidence_id": DCR_DOCTOR_FIXED_POINT_EVIDENCE,
        "version": DCR_DOCTOR_FIXED_POINT_VERSION,
        "bound": bound,
        "result": result.to_dict(),
        "observations": [
            (
                item.to_dict()
                if isinstance(item, DoctorFixedPointObservation)
                else DoctorFixedPointObservation.from_mapping(item).to_dict()
            )
            for item in observations
        ],
        "runtime_model_calls": 0,
        "provider_invocation_count": 0,
    }
    root = Path(repo_root).resolve() if repo_root is not None else Path.cwd()
    path = (
        Path(destination)
        if destination is not None
        else root.joinpath(*PurePosixPath(DEFAULT_DOCTOR_FIXED_POINT_PATH).parts)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


__all__ = [
    "DCR_DOCTOR_FIXED_POINT_EVIDENCE",
    "DCR_DOCTOR_FIXED_POINT_VERSION",
    "DEFAULT_DOCTOR_FIXED_POINT_BOUND",
    "DEFAULT_DOCTOR_FIXED_POINT_PATH",
    "DETERMINISTIC_DOCTOR_BACKEND_FACTORY_INTERFACE",
    "DETERMINISTIC_DOCTOR_EVIDENCE_BUNDLE_SCHEMA",
    "DETERMINISTIC_DOCTOR_RUNTIME_DISCOVERY_SCHEMA",
    "DETERMINISTIC_DOCTOR_RUNTIME_INTERFACE",
    "DETERMINISTIC_DOCTOR_RUNTIME_REPORT_SCHEMA",
    "DETERMINISTIC_REPAIR_DISPOSITION_INTERFACE",
    "DOCTOR_FIXED_POINT_CATALOG_SCHEMA",
    "DOCTOR_FIXED_POINT_INTERFACE",
    "DOCTOR_FIXED_POINT_OBSERVATION_SCHEMA",
    "DOCTOR_FIXED_POINT_RESULT_SCHEMA",
    "MAX_DOCTOR_FIXED_POINT_BOUND",
    "DeterministicDoctorBackendFactory",
    "DeterministicDoctorEvidenceBundle",
    "DeterministicDoctorRuntime",
    "DeterministicDoctorRuntimeError",
    "DeterministicDoctorRuntimeReport",
    "DeterministicDoctorRuntimeSafetyError",
    "DoctorFixedPoint",
    "DoctorFixedPointObservation",
    "DoctorFixedPointResult",
    "DoctorRuntimeStage",
    "DoctorRuntimeStageUnavailable",
    "DoctorSourceInventoryEntry",
    "DoctorStageCapability",
    "NoProgressGuard",
    "create_deterministic_doctor_runtime",
    "materialize_doctor_fixed_point",
]
