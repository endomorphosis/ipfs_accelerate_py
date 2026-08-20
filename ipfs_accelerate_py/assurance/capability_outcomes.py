"""FACP-025: Migrate Accelerate capability and inference outcomes.

Probe-backed capability / inference adapter for inventoried Accelerate seams.

Fail-closed invariants:

* Non-CPU routing requires **current** capability probe evidence (identity +
  freshness). Simulated / fixture / declared / stale probes cannot admit
  cuda/openvino/webgpu/… production routes.
* CPU remains the baseline route and does not require non-CPU probe receipts.
* Simulation namespaces (mock hardware, mock handlers, compat mock IPFS, …)
  remain selectable **only** under explicit test mode; production resolution
  yields typed ``Unavailable`` / ``Rejected``.
* Inference returns observed or delegated evidence, ``Unknown``,
  ``Unavailable``, or ``Failed`` — never invented success. Mock/fallback
  ``success=True`` stubs are compatibility-refused.

Cold import is hermetic: no network, installer, hardware probe, or process
mutation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Final, Literal, Mapping, Optional, Sequence

TASK_ID: Final[str] = "FACP-025"
GOAL_ID: Final[str] = "FACP-G220"
BUNDLE: Final[str] = "facp/migration/accelerate-outcomes"
EVIDENCE_ID: Final[str] = "facp/accelerate-outcomes@1"
INTERFACE: Final[str] = "AccelerateCapabilityOutcomes@1"
SCHEMA: Final[str] = "ipfs_accelerate_py/assurance/capability-outcomes@1"
FCA_VOCABULARY_SCHEMA: Final[str] = "facp/formal-claim-algebra-v1@1"
UNSAFE_PROMOTION: Final[bool] = False

# Explicit test-mode admission (simulation namespace selection).
EXPLICIT_TEST_MODE_ENV: Final[str] = "IPFS_ACCELERATE_EXPLICIT_TEST_MODE"

ClosedOutcome = Literal[
    "Unavailable",
    "Rejected",
    "Simulated",
    "Attempted",
    "Unknown",
    "Observed",
    "Verified",
    "Failed",
    "Compensated",
]

CLOSED_OUTCOMES: Final[frozenset[str]] = frozenset(
    {
        "Unavailable",
        "Rejected",
        "Simulated",
        "Attempted",
        "Unknown",
        "Observed",
        "Verified",
        "Failed",
        "Compensated",
    }
)

# Inference success dispositions: Observed with admitted observation / delegation.
_INFERENCE_SUCCESS_OUTCOMES: Final[frozenset[str]] = frozenset({"Observed", "Verified"})
_INFERENCE_SUCCESS_CODES: Final[frozenset[str]] = frozenset(
    {
        "inference_observed",
        "inference_delegated",
        "effect_observed",
        "verified_admitted",
    }
)

# Backends that may route without non-CPU capability probe evidence.
CPU_BACKENDS: Final[frozenset[str]] = frozenset(
    {"cpu", "cpu:0", "local-cpu", "host-cpu"}
)

NON_CPU_BACKEND_FAMILIES: Final[frozenset[str]] = frozenset(
    {
        "cuda",
        "rocm",
        "openvino",
        "webgpu",
        "metal",
        "qualcomm",
        "mps",
        "xpu",
        "habana",
        "tpu",
    }
)

_WEAK_ORIGINS: Final[frozenset[str]] = frozenset(
    {"absent", "declared", "fixture", "simulated"}
)
_OBSERVABLE_ORIGINS: Final[frozenset[str]] = frozenset(
    {"hermetic_observed", "live_observed"}
)
_CURRENT_FRESHNESS: Final[frozenset[str]] = frozenset({"current"})
_STALE_FRESHNESS: Final[frozenset[str]] = frozenset(
    {"stale", "superseded", "withdrawn", "missing"}
)
_VALID_INTEGRITY: Final[frozenset[str]] = frozenset(
    {"digest_valid", "signature_valid"}
)

_REQUIRED_PROBE_EVIDENCE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "probe_identity",
        "probe_freshness_receipt",
    }
)

DIMENSION_ORDER: Final[tuple[str, ...]] = (
    "origin",
    "integrity",
    "authority",
    "policy",
    "proof",
    "freshness",
    "effect",
    "environment",
    "review",
)

# Inventoried capability / inference / simulation seams owned by FACP-025.
INVENTORIED_CAPABILITY_SITES: Final[tuple[Mapping[str, Any], ...]] = (
    {
        "defect_id": "defect:accelerate-mock-worker-cuda-true",
        "seed_id": "seed:mock-worker-cuda-true",
        "category": "mock_worker",
        "fca": "reject_simulated_to_live_promotion",
        "symbol": "MockWorker.test_hardware",
    },
    {
        "defect_id": "defect:accelerate-test-hardware-hardcoded-support",
        "seed_id": "seed:hardcoded-hwtest-true",
        "category": "success_support_field",
        "fca": "require_probe_backed_capability",
        "symbol": "ipfs_accelerate_py.test_hardware",
    },
    {
        "defect_id": "defect:accelerate-mock-handler-labeled-real",
        "seed_id": "seed:mock-handler-real-label",
        "category": "mock_handler",
        "fca": "reject_simulated_to_live_promotion",
        "symbol": "ipfs_accelerate_py._create_mock_handler",
    },
    {
        "defect_id": "defect:accelerate-api-available-default-true",
        "seed_id": "seed:api-available-default-true",
        "category": "success_support_field",
        "fca": "require_probe_backed_capability",
        "symbol": "api endpoint registration",
    },
    {
        "defect_id": "defect:accelerate-legacy-mock-hardware-detection",
        "seed_id": "seed:legacy-mock-hardware",
        "category": "mock_hardware",
        "fca": "isolate_simulation_namespace",
        "symbol": "ipfs_accelerate_py._create_mock_hardware_detection",
    },
    {
        "defect_id": "defect:accelerate-cuda-utils-mock-device-available",
        "seed_id": "seed:cuda-mock-available-true",
        "category": "mock_hardware",
        "fca": "require_probe_backed_capability",
        "symbol": "CudaUtils.create_cuda_mock_implementation",
    },
    {
        "defect_id": "defect:accelerate-skillset-mock-model-fallback",
        "seed_id": "seed:skillset-mock-fallback-success",
        "category": "inference_outcome",
        "fca": "require_observed_outcome",
        "symbol": "hf_t5.create_mock_model",
    },
    {
        "defect_id": "defect:accelerate-fallback-compat-mock-namespaces",
        "seed_id": "seed:compat-mock-namespace",
        "category": "fallback_namespace",
        "fca": "isolate_simulation_namespace",
        "symbol": "mcp_server.tools.ipfs._load_mock_ipfs_client",
    },
    {
        "defect_id": "defect:accelerate-legacy-mock-inference-handler",
        "seed_id": "seed:legacy-mock-handler",
        "category": "mock_handler",
        "fca": "reject_simulated_to_live_promotion",
        "symbol": "ipfs_accelerate_py_legacy._create_mock_handler",
    },
    {
        "defect_id": "defect:accelerate-cuda-utils-magicmock-torch",
        "seed_id": "seed:cuda-utils-magicmock-torch",
        "category": "mock_hardware",
        "fca": "require_probe_backed_capability",
        "symbol": "cuda_utils torch MagicMock substitution",
    },
    {
        "defect_id": "defect:accelerate-worker-hwtest-hardcoded-true",
        "seed_id": "seed:worker-hwtest-hardcoded-true",
        "category": "success_support_field",
        "fca": "require_probe_backed_capability",
        "symbol": "worker.Worker.test_hardware",
    },
    {
        "defect_id": "defect:accelerate-ai-model-server-mock-classify",
        "seed_id": "seed:ai-model-server-mock-classify",
        "category": "inference_outcome",
        "fca": "require_observed_outcome",
        "symbol": "mcp ai_model_server mock generate/classify/embed",
    },
)

SIMULATION_NAMESPACES: Final[frozenset[str]] = frozenset(
    {
        "mock_hardware",
        "mock_worker",
        "mock_handler",
        "mock_ipfs",
        "compat_mock_ipfs",
        "cuda_mock_implementation",
        "skillset_mock_model",
        "ai_model_server_mock",
    }
)

FORBIDDEN_LEGACY_SUCCESS_FIELDS: Final[frozenset[str]] = frozenset(
    {"success", "ok", "passed", "production_supported", "available", "supported"}
)


class CapabilityOutcomeError(ValueError):
    """Malformed Accelerate capability / inference outcome construction."""


class RoutingDisposition(str, Enum):
    """Result of backend route admission."""

    ADMITTED = "admitted"
    CPU_BASELINE = "cpu_baseline"
    REJECTED_MISSING_PROBE = "rejected_missing_probe"
    REJECTED_STALE_PROBE = "rejected_stale_probe"
    REJECTED_WEAK_ORIGIN = "rejected_weak_origin"
    REJECTED_SIMULATED = "rejected_simulated"
    UNAVAILABLE = "unavailable"


class SimulationSelection(str, Enum):
    """Simulation-namespace selection disposition."""

    SELECTED_TEST_MODE = "selected_test_mode"
    REFUSED_PRODUCTION = "refused_production"
    UNKNOWN_NAMESPACE = "unknown_namespace"


@dataclass(frozen=True)
class EvidenceEnvelope:
    """Nine-dimension FCA evidence product (weakest defaults)."""

    origin: str = "absent"
    integrity: str = "unchecked"
    authority: str = "unchecked"
    policy: str = "unchecked"
    proof: str = "none"
    freshness: str = "stale"
    effect: str = "not_started"
    environment: str = "hermetic"
    review: str = "unreviewed"

    def to_mapping(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in DIMENSION_ORDER}

    def with_overrides(self, **overrides: str) -> "EvidenceEnvelope":
        data = self.to_mapping()
        data.update(overrides)
        return EvidenceEnvelope(**data)


@dataclass(frozen=True)
class CapabilityProbeEvidence:
    """Current capability probe evidence for one backend / device family.

    Identity binds the probe to a concrete backend. Freshness and origin gate
    non-CPU production routing. Simulated / fixture / declared probes never
    satisfy the current-capability conjunction.
    """

    backend: str
    probe_identity: str
    available: bool = False
    origin: str = "absent"
    freshness: str = "stale"
    integrity: str = "unchecked"
    authority: str = "unchecked"
    policy: str = "unchecked"
    proof: str = "none"
    effect: str = "not_started"
    environment: str = "hermetic"
    review: str = "unreviewed"
    probe_freshness_receipt: Optional[str] = None
    issued_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    evidence_bag: Mapping[str, str] = field(default_factory=dict)
    attributes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise CapabilityOutcomeError("backend must be a non-empty string")
        if not isinstance(self.probe_identity, str) or not self.probe_identity.strip():
            raise CapabilityOutcomeError("probe_identity must be a non-empty string")
        if not isinstance(self.available, bool):
            raise CapabilityOutcomeError("available must be a boolean")
        for map_name in ("evidence_bag", "attributes"):
            raw = getattr(self, map_name)
            if isinstance(raw, MappingProxyType):
                continue
            if not isinstance(raw, Mapping):
                raise CapabilityOutcomeError(f"{map_name} must be a mapping")
            object.__setattr__(
                self,
                map_name,
                MappingProxyType({str(k): str(v) for k, v in raw.items()}),
            )
        for ts_name in ("issued_at", "expires_at"):
            ts = getattr(self, ts_name)
            if ts is not None and (
                not isinstance(ts, datetime) or ts.tzinfo is None
            ):
                raise CapabilityOutcomeError(
                    f"{ts_name} must be timezone-aware datetime when set"
                )

    def to_envelope(self) -> EvidenceEnvelope:
        return EvidenceEnvelope(
            origin=self.origin,
            integrity=self.integrity,
            authority=self.authority,
            policy=self.policy,
            proof=self.proof,
            freshness=self.freshness,
            effect=self.effect,
            environment=self.environment,
            review=self.review,
        )

    def with_overrides(self, **overrides: Any) -> "CapabilityProbeEvidence":
        return replace(self, **overrides)


@dataclass(frozen=True)
class CapabilityOutcome:
    """Typed closed-outcome result for capability routing or inference.

    ``ok`` is true only for Observed/Verified with an admitted success code.
    Invented mock success, Attempted, Unknown, Unavailable, Simulated,
    Rejected, and Failed are never success dispositions.
    """

    outcome: ClosedOutcome
    code: str
    message: str
    operation: str
    envelope: EvidenceEnvelope = field(default_factory=EvidenceEnvelope)
    evidence: frozenset[str] = field(default_factory=frozenset)
    backend: Optional[str] = None
    details: Mapping[str, Any] = field(default_factory=dict)
    unsafe_promotion: bool = False
    routing_disposition: Optional[str] = None

    def __post_init__(self) -> None:
        if self.outcome not in CLOSED_OUTCOMES:
            raise CapabilityOutcomeError(f"unknown closed outcome: {self.outcome!r}")
        if self.unsafe_promotion:
            raise CapabilityOutcomeError(
                "unsafe_promotion must remain False on Accelerate capability outcomes"
            )
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))
        object.__setattr__(self, "evidence", frozenset(self.evidence))

    @property
    def ok(self) -> bool:
        return (
            self.outcome in _INFERENCE_SUCCESS_OUTCOMES
            and self.code in _INFERENCE_SUCCESS_CODES
        )

    @property
    def is_success_disposition(self) -> bool:
        return self.ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "code": self.code,
            "message": self.message,
            "operation": self.operation,
            "ok": self.ok,
            "backend": self.backend,
            "envelope": self.envelope.to_mapping(),
            "evidence": sorted(self.evidence),
            "details": dict(self.details),
            "unsafe_promotion": self.unsafe_promotion,
            "routing_disposition": self.routing_disposition,
            "task_id": TASK_ID,
            "evidence_id": EVIDENCE_ID,
            "fca_vocabulary_schema": FCA_VOCABULARY_SCHEMA,
            "schema": SCHEMA,
        }

    def to_legacy_compat_dict(self) -> dict[str, Any]:
        """Project to a dict-shaped compatibility surface without inventing success."""
        payload: dict[str, Any] = {
            "status": "success" if self.ok else self.outcome.lower(),
            "outcome": self.outcome,
            "code": self.code,
            "message": self.message,
            "operation": self.operation,
            "ok": self.ok,
            "disposition": "success" if self.ok else "non_success",
        }
        if self.backend is not None:
            payload["backend"] = self.backend
        for key in ("model", "endpoint", "namespace", "observation_id", "receipt_id"):
            if key in self.details:
                payload[key] = self.details[key]
        return payload


@dataclass(frozen=True)
class RouteDecision:
    """Backend routing decision after capability-evidence admission."""

    admitted: bool
    backend: Optional[str]
    disposition: RoutingDisposition
    outcome: CapabilityOutcome
    reason_codes: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.admitted and self.outcome.outcome not in {
            "Unavailable",
            "Rejected",
            "Failed",
            "Simulated",
            "Unknown",
        }


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def normalize_backend(backend: str | None) -> str:
    """Normalize a backend / device string to a comparable token."""
    if backend is None:
        return ""
    text = str(backend).strip().lower()
    if not text:
        return ""
    # Strip device ordinal: cuda:0 -> cuda, openvino:CPU -> openvino
    if ":" in text:
        family, _, rest = text.partition(":")
        if family in NON_CPU_BACKEND_FAMILIES or family == "cpu":
            return family if family != "cpu" else (
                text if text in CPU_BACKENDS else "cpu"
            )
        # Keep composite tokens that are not ordinal device ids.
        if rest.isdigit() or rest in {"0", "1", "2", "3"}:
            return family
    return text


def is_cpu_backend(backend: str | None) -> bool:
    token = normalize_backend(backend)
    return token in CPU_BACKENDS or token == "cpu"


def is_non_cpu_backend(backend: str | None) -> bool:
    token = normalize_backend(backend)
    if not token or is_cpu_backend(token):
        return False
    if token in NON_CPU_BACKEND_FAMILIES:
        return True
    for family in NON_CPU_BACKEND_FAMILIES:
        if token.startswith(family):
            return True
    return False


def is_explicit_test_mode(
    *,
    explicit_test_mode: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return True only when explicit test mode is opted in.

    Production callers must pass ``explicit_test_mode=True`` or set
    ``IPFS_ACCELERATE_EXPLICIT_TEST_MODE`` to a truthy value (1/true/yes/on).
    Presence of ``PYTEST_CURRENT_TEST`` alone is **not** sufficient — simulation
    selection requires an explicit opt-in.
    """
    if explicit_test_mode is True:
        return True
    if explicit_test_mode is False:
        return False
    env = environ if environ is not None else os.environ
    raw = str(env.get(EXPLICIT_TEST_MODE_ENV, "") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def current_capability_probe(
    backend: str,
    *,
    probe_identity: str,
    available: bool = True,
    origin: str = "live_observed",
    freshness: str = "current",
    integrity: str = "digest_valid",
    authority: str = "valid",
    policy: str = "allowed",
    environment: str = "live",
    probe_freshness_receipt: str | None = None,
    issued_at: datetime | None = None,
    expires_at: datetime | None = None,
    evidence_bag: Mapping[str, str] | None = None,
    attributes: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> CapabilityProbeEvidence:
    """Build a current live capability probe evidence product."""
    now = now or _utcnow()
    receipt = probe_freshness_receipt or f"probe-receipt:{normalize_backend(backend)}"
    bag = {
        "probe_identity": probe_identity,
        "probe_freshness_receipt": receipt,
        **dict(evidence_bag or {}),
    }
    return CapabilityProbeEvidence(
        backend=normalize_backend(backend) or backend,
        probe_identity=probe_identity,
        available=available,
        origin=origin,
        freshness=freshness,
        integrity=integrity,
        authority=authority,
        policy=policy,
        proof="none",
        effect="observed",
        environment=environment,
        review="machine_reviewed",
        probe_freshness_receipt=receipt,
        issued_at=issued_at or now,
        expires_at=expires_at,
        evidence_bag=bag,
        attributes=dict(attributes or {}),
    )


def _probe_receipt_current(
    probe: CapabilityProbeEvidence, *, now: datetime
) -> bool:
    if probe.expires_at is not None and probe.expires_at <= now:
        return False
    if probe.freshness not in _CURRENT_FRESHNESS:
        return False
    return True


def is_current_capability_evidence(
    probe: CapabilityProbeEvidence | None,
    *,
    now: datetime | None = None,
) -> bool:
    """True when probe evidence can admit non-CPU production routing."""
    if probe is None:
        return False
    now = now or _utcnow()
    if not probe.available:
        return False
    if probe.origin in _WEAK_ORIGINS:
        return False
    if probe.origin not in _OBSERVABLE_ORIGINS:
        return False
    if not _probe_receipt_current(probe, now=now):
        return False
    if probe.freshness in _STALE_FRESHNESS:
        return False
    if not probe.probe_identity.strip():
        return False
    if not probe.probe_freshness_receipt:
        return False
    bag_keys = set(probe.evidence_bag.keys())
    bag_keys.add("probe_identity")
    bag_keys.add("probe_freshness_receipt")
    if not _REQUIRED_PROBE_EVIDENCE_KEYS.issubset(bag_keys):
        return False
    if probe.integrity not in _VALID_INTEGRITY and probe.origin == "live_observed":
        # Hermetic probes may carry structurally_valid; live requires digest/signature.
        return False
    return True


def assess_capability_probe(
    probe: CapabilityProbeEvidence | None,
    *,
    backend: str | None = None,
    now: datetime | None = None,
) -> CapabilityOutcome:
    """Assess whether capability probe evidence is current for routing."""
    now = now or _utcnow()
    target = normalize_backend(backend or (probe.backend if probe else None))

    if probe is None:
        return CapabilityOutcome(
            outcome="Unavailable",
            code="capability_probe_missing",
            message=(
                f"non-CPU backend {target!r} requires current capability probe "
                "evidence; none was provided"
            ),
            operation="capability_route",
            backend=target or None,
            envelope=EvidenceEnvelope(),
            details={
                "fallback_success_forbidden": True,
                "requires_current_capability_evidence": True,
            },
            routing_disposition=RoutingDisposition.REJECTED_MISSING_PROBE.value,
        )

    if probe.origin in {"simulated", "fixture"}:
        return CapabilityOutcome(
            outcome="Simulated",
            code="capability_probe_simulated",
            message=(
                f"capability probe for {probe.backend!r} originates as "
                f"{probe.origin}; cannot admit non-CPU production routing"
            ),
            operation="capability_route",
            backend=probe.backend,
            envelope=probe.to_envelope(),
            evidence=frozenset({"probe_identity"}),
            details={
                "probe_identity": probe.probe_identity,
                "origin": probe.origin,
                "production_routing_forbidden": True,
            },
            routing_disposition=RoutingDisposition.REJECTED_SIMULATED.value,
        )

    if probe.origin in _WEAK_ORIGINS:
        return CapabilityOutcome(
            outcome="Rejected",
            code="capability_probe_weak_origin",
            message=(
                f"capability probe origin {probe.origin!r} cannot admit "
                f"non-CPU routing for {probe.backend!r}"
            ),
            operation="capability_route",
            backend=probe.backend,
            envelope=probe.to_envelope(),
            details={"origin": probe.origin, "probe_identity": probe.probe_identity},
            routing_disposition=RoutingDisposition.REJECTED_WEAK_ORIGIN.value,
        )

    if not _probe_receipt_current(probe, now=now) or probe.freshness in _STALE_FRESHNESS:
        return CapabilityOutcome(
            outcome="Rejected",
            code="capability_probe_stale",
            message=(
                f"capability probe for {probe.backend!r} is not current "
                f"(freshness={probe.freshness!r})"
            ),
            operation="capability_route",
            backend=probe.backend,
            envelope=probe.to_envelope().with_overrides(freshness="stale"),
            evidence=frozenset({"probe_identity"}),
            details={
                "probe_identity": probe.probe_identity,
                "freshness": probe.freshness,
                "expires_at": probe.expires_at.isoformat() if probe.expires_at else None,
            },
            routing_disposition=RoutingDisposition.REJECTED_STALE_PROBE.value,
        )

    if not probe.available:
        return CapabilityOutcome(
            outcome="Unavailable",
            code="capability_unavailable",
            message=f"capability probe reports {probe.backend!r} unavailable",
            operation="capability_route",
            backend=probe.backend,
            envelope=probe.to_envelope(),
            evidence=frozenset({"probe_identity", "probe_freshness_receipt"}),
            details={"available": False, "probe_identity": probe.probe_identity},
            routing_disposition=RoutingDisposition.UNAVAILABLE.value,
        )

    if not is_current_capability_evidence(probe, now=now):
        return CapabilityOutcome(
            outcome="Rejected",
            code="capability_probe_incomplete",
            message=(
                f"capability probe for {probe.backend!r} fails the current "
                "capability evidence conjunction"
            ),
            operation="capability_route",
            backend=probe.backend,
            envelope=probe.to_envelope(),
            details={
                "probe_identity": probe.probe_identity,
                "required_evidence": sorted(_REQUIRED_PROBE_EVIDENCE_KEYS),
            },
            routing_disposition=RoutingDisposition.REJECTED_MISSING_PROBE.value,
        )

    return CapabilityOutcome(
        outcome="Observed",
        code="effect_observed",
        message=f"current capability evidence admits backend {probe.backend!r}",
        operation="capability_route",
        backend=probe.backend,
        envelope=probe.to_envelope().with_overrides(
            freshness="current",
            effect="observed",
        ),
        evidence=frozenset({"probe_identity", "probe_freshness_receipt"}),
        details={
            "probe_identity": probe.probe_identity,
            "probe_freshness_receipt": probe.probe_freshness_receipt,
            "available": True,
        },
        routing_disposition=RoutingDisposition.ADMITTED.value,
    )


def route_backend(
    backend: str,
    *,
    probe: CapabilityProbeEvidence | None = None,
    now: datetime | None = None,
) -> RouteDecision:
    """Admit a backend route under probe-backed capability rules.

    CPU routes are admitted as the baseline without non-CPU probe evidence.
    Non-CPU routes require current capability probe evidence.
    """
    now = now or _utcnow()
    token = normalize_backend(backend)
    if not token:
        outcome = CapabilityOutcome(
            outcome="Unavailable",
            code="backend_unspecified",
            message="backend route requires a non-empty backend identifier",
            operation="capability_route",
            envelope=EvidenceEnvelope(),
            details={"fallback_success_forbidden": True},
            routing_disposition=RoutingDisposition.UNAVAILABLE.value,
        )
        return RouteDecision(
            admitted=False,
            backend=None,
            disposition=RoutingDisposition.UNAVAILABLE,
            outcome=outcome,
            reason_codes=("backend_unspecified",),
        )

    if is_cpu_backend(token):
        outcome = CapabilityOutcome(
            outcome="Observed",
            code="effect_observed",
            message=f"CPU baseline route admitted for {token!r} without non-CPU probe",
            operation="capability_route",
            backend=token,
            envelope=EvidenceEnvelope(
                origin="declared",
                freshness="current",
                effect="observed",
                environment="hermetic",
                integrity="structurally_valid",
                authority="valid",
                policy="allowed",
                review="machine_reviewed",
            ),
            evidence=frozenset({"cpu_baseline"}),
            details={
                "cpu_baseline": True,
                "non_cpu_probe_required": False,
            },
            routing_disposition=RoutingDisposition.CPU_BASELINE.value,
        )
        return RouteDecision(
            admitted=True,
            backend=token,
            disposition=RoutingDisposition.CPU_BASELINE,
            outcome=outcome,
            reason_codes=("cpu_baseline_admitted",),
        )

    if not is_non_cpu_backend(token):
        # Unknown accelerator family: fail closed unless probe explicitly covers it.
        if probe is None or normalize_backend(probe.backend) != token:
            outcome = CapabilityOutcome(
                outcome="Unavailable",
                code="backend_family_unrecognized",
                message=(
                    f"backend {token!r} is not a recognized CPU or non-CPU family "
                    "and has no matching current probe"
                ),
                operation="capability_route",
                backend=token,
                envelope=EvidenceEnvelope(),
                details={"fallback_success_forbidden": True},
                routing_disposition=RoutingDisposition.UNAVAILABLE.value,
            )
            return RouteDecision(
                admitted=False,
                backend=None,
                disposition=RoutingDisposition.UNAVAILABLE,
                outcome=outcome,
                reason_codes=("backend_family_unrecognized",),
            )

    assessed = assess_capability_probe(probe, backend=token, now=now)
    if assessed.routing_disposition == RoutingDisposition.ADMITTED.value and assessed.ok:
        # Probe backend must match requested route family.
        if probe is not None and normalize_backend(probe.backend) != token:
            mismatch = CapabilityOutcome(
                outcome="Rejected",
                code="capability_probe_backend_mismatch",
                message=(
                    f"probe backend {probe.backend!r} does not match requested "
                    f"route {token!r}"
                ),
                operation="capability_route",
                backend=token,
                envelope=probe.to_envelope(),
                details={
                    "requested": token,
                    "probe_backend": probe.backend,
                },
                routing_disposition=RoutingDisposition.REJECTED_MISSING_PROBE.value,
            )
            return RouteDecision(
                admitted=False,
                backend=None,
                disposition=RoutingDisposition.REJECTED_MISSING_PROBE,
                outcome=mismatch,
                reason_codes=("capability_probe_backend_mismatch",),
            )
        return RouteDecision(
            admitted=True,
            backend=token,
            disposition=RoutingDisposition.ADMITTED,
            outcome=assessed,
            reason_codes=(
                "non_cpu_routing_requires_current_capability_evidence",
                "current_capability_evidence_admitted",
            ),
        )

    disposition = RoutingDisposition(
        assessed.routing_disposition or RoutingDisposition.UNAVAILABLE.value
    )
    return RouteDecision(
        admitted=False,
        backend=None,
        disposition=disposition,
        outcome=assessed,
        reason_codes=(assessed.code, "non_cpu_routing_requires_current_capability_evidence"),
    )


def select_simulation_namespace(
    namespace: str,
    *,
    explicit_test_mode: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> CapabilityOutcome:
    """Select a simulation namespace only under explicit test mode.

    Outside explicit test mode the selection is refused with typed
    ``Unavailable`` (compatibility / production resolution fail-closed).
    """
    name = str(namespace or "").strip().lower()
    if not name:
        return CapabilityOutcome(
            outcome="Unavailable",
            code="simulation_namespace_unspecified",
            message="simulation namespace selection requires a namespace id",
            operation="simulation_select",
            envelope=EvidenceEnvelope(origin="absent"),
            details={"fallback_success_forbidden": True},
        )

    if name not in SIMULATION_NAMESPACES:
        return CapabilityOutcome(
            outcome="Rejected",
            code="simulation_namespace_unknown",
            message=f"unknown simulation namespace {name!r}",
            operation="simulation_select",
            envelope=EvidenceEnvelope(origin="declared"),
            details={
                "namespace": name,
                "known_namespaces": sorted(SIMULATION_NAMESPACES),
                "selection": SimulationSelection.UNKNOWN_NAMESPACE.value,
            },
        )

    if not is_explicit_test_mode(
        explicit_test_mode=explicit_test_mode, environ=environ
    ):
        return CapabilityOutcome(
            outcome="Unavailable",
            code="simulation_requires_explicit_test_mode",
            message=(
                f"simulation namespace {name!r} is selectable only under explicit "
                "test mode; production resolution is Unavailable"
            ),
            operation="simulation_select",
            envelope=EvidenceEnvelope(origin="simulated", effect="not_started"),
            details={
                "namespace": name,
                "explicit_test_mode": False,
                "selection": SimulationSelection.REFUSED_PRODUCTION.value,
                "compatibility_refusal": True,
                "fallback_success_forbidden": True,
            },
        )

    return CapabilityOutcome(
        outcome="Simulated",
        code="simulation_selected_test_mode",
        message=(
            f"simulation namespace {name!r} selected under explicit test mode; "
            "not production-supported"
        ),
        operation="simulation_select",
        envelope=EvidenceEnvelope(
            origin="simulated",
            effect="started",
            environment="hermetic",
            freshness="current",
            review="machine_reviewed",
        ),
        evidence=frozenset({"explicit_test_mode"}),
        details={
            "namespace": name,
            "explicit_test_mode": True,
            "selection": SimulationSelection.SELECTED_TEST_MODE.value,
            "production_supported": False,
        },
    )


def begin_inference_attempt(
    *,
    backend: str = "cpu",
    model: str | None = None,
    origin: str = "declared",
    details: Mapping[str, Any] | None = None,
) -> CapabilityOutcome:
    """Record that inference was attempted but not yet observed."""
    env_origin = origin if origin in (_WEAK_ORIGINS | _OBSERVABLE_ORIGINS) else "declared"
    return CapabilityOutcome(
        outcome="Attempted",
        code="inference_attempted",
        message="inference started; independent observation pending",
        operation="inference",
        backend=normalize_backend(backend) or backend,
        envelope=EvidenceEnvelope(effect="started", origin=env_origin),
        evidence=frozenset(),
        details={
            "attempt_evidenced": True,
            "model": model,
            **dict(details or {}),
        },
    )


def bind_inference_observation(
    attempt: CapabilityOutcome,
    *,
    observation_present: bool,
    observation_id: str | None = None,
    admission_token: str | None = None,
    origin: str = "hermetic_observed",
    integrity: str = "digest_valid",
    delegated: bool = False,
    details: Mapping[str, Any] | None = None,
) -> CapabilityOutcome:
    """Bind an independent (or delegated) observation to an inference attempt.

    Without observation evidence the result is ``Unknown`` — never success.
    """
    if attempt.outcome not in {"Attempted", "Unknown", "Observed"}:
        return CapabilityOutcome(
            outcome="Failed",
            code="observation_requires_attempt",
            message=(
                "bind_inference_observation requires a prior Attempted/Unknown/"
                f"Observed outcome, got {attempt.outcome!r}"
            ),
            operation="inference",
            backend=attempt.backend,
            envelope=attempt.envelope,
            evidence=attempt.evidence,
            details={"prior_outcome": attempt.outcome, **dict(details or {})},
        )

    if not observation_present or not observation_id:
        return CapabilityOutcome(
            outcome="Unknown",
            code="inference_unobserved",
            message=(
                "inference was attempted but no independent observation is "
                "available; not success"
            ),
            operation="inference",
            backend=attempt.backend,
            envelope=attempt.envelope.with_overrides(effect="externally_unknown"),
            evidence=attempt.evidence,
            details={
                "attempt_evidenced": True,
                "observation_present": False,
                "success_forbidden_without_observation": True,
                **dict(details or {}),
            },
        )

    if origin in _WEAK_ORIGINS:
        return CapabilityOutcome(
            outcome="Failed",
            code="weak_origin_cannot_observe_inference",
            message=f"origin {origin!r} cannot bind an inference observation",
            operation="inference",
            backend=attempt.backend,
            envelope=attempt.envelope.with_overrides(origin=origin, effect="started"),
            evidence=attempt.evidence,
            details={"origin": origin, **dict(details or {})},
        )

    evidence = set(attempt.evidence)
    evidence.add("independent_effect_observation")
    if admission_token:
        evidence.add("admission_token")
    if delegated:
        evidence.add("delegated_receipt")

    code = "inference_delegated" if delegated else "inference_observed"
    return CapabilityOutcome(
        outcome="Observed",
        code=code,
        message=(
            "inference observed via delegated receipt"
            if delegated
            else "inference observed via independent evidence"
        ),
        operation="inference",
        backend=attempt.backend,
        envelope=EvidenceEnvelope(
            origin=origin if origin in _OBSERVABLE_ORIGINS else "hermetic_observed",
            integrity=integrity if integrity in _VALID_INTEGRITY else "unchecked",
            authority="valid",
            policy="allowed",
            proof="none",
            freshness="current",
            effect="observed",
            environment="hermetic" if origin == "hermetic_observed" else "live",
            review="machine_reviewed",
        ),
        evidence=frozenset(evidence),
        details={
            "observation_id": observation_id,
            "admission_token": admission_token,
            "delegated": delegated,
            **dict(details or {}),
        },
    )


def validate_delegated_inference_receipt(
    receipt: Mapping[str, Any],
    *,
    backend: str = "cpu",
    model: str | None = None,
) -> CapabilityOutcome:
    """Validate a delegated inference receipt without inventing success."""
    if not receipt:
        return CapabilityOutcome(
            outcome="Unavailable",
            code="delegated_receipt_missing",
            message="delegated inference receipt is missing",
            operation="inference",
            backend=normalize_backend(backend) or backend,
            envelope=EvidenceEnvelope(),
            details={"receipt_present": False, "model": model},
        )

    if receipt.get("revoked") is True or receipt.get("authority") == "revoked":
        return CapabilityOutcome(
            outcome="Failed",
            code="delegated_receipt_revoked",
            message="delegated inference receipt authority is revoked",
            operation="inference",
            backend=normalize_backend(backend) or backend,
            envelope=EvidenceEnvelope(authority="revoked", effect="failed"),
            details={"receipt_id": receipt.get("receipt_id"), "model": model},
        )

    has_observation = bool(receipt.get("independent_effect_observation"))
    has_signature = bool(
        receipt.get("signed_receipt") or receipt.get("signature_valid")
    )
    if not has_observation:
        return CapabilityOutcome(
            outcome="Unknown",
            code="delegated_receipt_unobserved",
            message="delegated inference receipt lacks independent effect observation",
            operation="inference",
            backend=normalize_backend(backend) or backend,
            envelope=EvidenceEnvelope(
                effect="externally_unknown",
                integrity="structurally_valid" if has_signature else "unchecked",
            ),
            details={"receipt_id": receipt.get("receipt_id"), "model": model},
        )

    if not has_signature:
        return CapabilityOutcome(
            outcome="Unknown",
            code="delegated_receipt_unsigned",
            message="delegated inference receipt observation is unsigned",
            operation="inference",
            backend=normalize_backend(backend) or backend,
            envelope=EvidenceEnvelope(effect="externally_unknown", integrity="unchecked"),
            evidence=frozenset({"independent_effect_observation"}),
            details={"receipt_id": receipt.get("receipt_id"), "model": model},
        )

    attempt = begin_inference_attempt(
        backend=backend,
        model=model or receipt.get("model"),  # type: ignore[arg-type]
    )
    return bind_inference_observation(
        attempt,
        observation_present=True,
        observation_id=str(receipt.get("receipt_id") or "delegated-inference"),
        admission_token=str(receipt.get("admission_token") or "admission:inference"),
        origin="live_observed"
        if receipt.get("environment") == "live"
        else "hermetic_observed",
        integrity="signature_valid",
        delegated=True,
        details={"receipt_id": receipt.get("receipt_id"), "model": model},
    )


def resolve_inference_outcome(
    *,
    backend: str = "cpu",
    model: str | None = None,
    probe: CapabilityProbeEvidence | None = None,
    backend_available: bool | None = None,
    attempt_evidenced: bool = False,
    observation_present: bool = False,
    observation_id: str | None = None,
    admission_token: str | None = None,
    delegated_receipt: Mapping[str, Any] | None = None,
    simulated: bool = False,
    mock_handler: bool = False,
    error: str | None = None,
    explicit_test_mode: bool | None = None,
    now: datetime | None = None,
    details: Mapping[str, Any] | None = None,
) -> CapabilityOutcome:
    """Resolve an inference outcome under require_observed_outcome.

    Never invents success. Returns Observed (local or delegated), Unknown,
    Unavailable, or Failed — or Simulated only when an explicit test-mode
    simulation path is selected (still non-success).
    """
    now = now or _utcnow()
    token = normalize_backend(backend) or "cpu"
    extra = dict(details or {})
    if model is not None:
        extra.setdefault("model", model)

    if error:
        return CapabilityOutcome(
            outcome="Failed",
            code="inference_failed",
            message=error,
            operation="inference",
            backend=token,
            envelope=EvidenceEnvelope(effect="failed", origin="declared"),
            details={**extra, "error": error},
        )

    if delegated_receipt is not None:
        return validate_delegated_inference_receipt(
            delegated_receipt,
            backend=token,
            model=model,
        )

    if simulated or mock_handler:
        if is_explicit_test_mode(explicit_test_mode=explicit_test_mode):
            return CapabilityOutcome(
                outcome="Simulated",
                code="inference_simulated_test_mode",
                message=(
                    "simulated/mock inference selected under explicit test mode; "
                    "not production success"
                ),
                operation="inference",
                backend=token,
                envelope=EvidenceEnvelope(
                    origin="simulated",
                    effect="started",
                    environment="hermetic",
                ),
                evidence=frozenset({"explicit_test_mode"}),
                details={
                    **extra,
                    "simulated": True,
                    "mock_handler": mock_handler,
                    "production_supported": False,
                },
            )
        return CapabilityOutcome(
            outcome="Unavailable",
            code="inference_simulation_refused_outside_test_mode",
            message=(
                "simulated/mock inference is selectable only under explicit test "
                "mode; production inference is Unavailable"
            ),
            operation="inference",
            backend=token,
            envelope=EvidenceEnvelope(origin="simulated", effect="not_started"),
            details={
                **extra,
                "simulated": True,
                "mock_handler": mock_handler,
                "compatibility_refusal": True,
                "fallback_success_forbidden": True,
            },
        )

    # Non-CPU inference requires current capability evidence before observation.
    if is_non_cpu_backend(token):
        route = route_backend(token, probe=probe, now=now)
        if not route.admitted:
            # Preserve Unavailable/Rejected/Simulated from routing; map to
            # inference operation while never inventing success.
            routed = route.outcome
            return CapabilityOutcome(
                outcome=routed.outcome
                if routed.outcome in {"Unavailable", "Rejected", "Simulated", "Failed"}
                else "Unavailable",
                code=routed.code,
                message=routed.message,
                operation="inference",
                backend=token,
                envelope=routed.envelope,
                evidence=routed.evidence,
                details={**dict(routed.details), **extra},
                routing_disposition=routed.routing_disposition,
            )

    if backend_available is False:
        return CapabilityOutcome(
            outcome="Unavailable",
            code="inference_backend_unavailable",
            message=f"inference backend {token!r} is unavailable",
            operation="inference",
            backend=token,
            envelope=EvidenceEnvelope(effect="not_started", origin="absent"),
            details={**extra, "fallback_success_forbidden": True},
        )

    if observation_present:
        attempt = begin_inference_attempt(backend=token, model=model, details=extra)
        return bind_inference_observation(
            attempt,
            observation_present=True,
            observation_id=observation_id or "inference-observation",
            admission_token=admission_token or "admission:inference",
            details=extra,
        )

    if attempt_evidenced:
        # Attempted-but-unobserved → Unknown for the inference surface called out
        # by acceptance (Observed/delegated, Unknown, Unavailable, Failed).
        attempt = begin_inference_attempt(backend=token, model=model, details=extra)
        return bind_inference_observation(
            attempt,
            observation_present=False,
            details=extra,
        )

    return CapabilityOutcome(
        outcome="Unavailable",
        code="inference_evidence_absent",
        message=(
            "inference lacks attempt, observation, or delegated receipt evidence; "
            "not success"
        ),
        operation="inference",
        backend=token,
        envelope=EvidenceEnvelope(),
        details={**extra, "fallback_success_forbidden": True},
    )


def refuse_compatibility_success(
    legacy: Mapping[str, Any],
    *,
    operation: str = "inference",
    backend: str | None = None,
) -> CapabilityOutcome:
    """Refuse legacy success / available / supported stubs without evidence.

    Compatibility projection **preserves non-success disposition**: a legacy
    ``success=True`` / ``implementation_type=REAL`` mock never becomes
    Observed. Simulated origins map to refusal outside test mode semantics
    (Unavailable) or Simulated classification when explicitly marked and the
    caller already opted into test-mode selection elsewhere.
    """
    token = normalize_backend(
        backend or legacy.get("backend") or legacy.get("device") or "cpu"
    )
    details: dict[str, Any] = {
        "legacy_keys": sorted(str(k) for k in legacy.keys()),
        "compatibility_projection": True,
        "compatibility_refusal": True,
    }

    note = str(legacy.get("note") or legacy.get("message") or "").lower()
    impl = str(legacy.get("implementation_type") or "").upper()
    mockish = (
        legacy.get("mock") is True
        or legacy.get("simulated") is True
        or impl == "MOCK"
        or "mock" in note
        or "simulated" in note
        or str(legacy.get("origin") or "").lower() in {"mock", "simulated", "fixture"}
        or bool(legacy.get("create_mock_model"))
    )

    claims_success = (
        legacy.get("status") == "success"
        or any(legacy.get(k) is True for k in FORBIDDEN_LEGACY_SUCCESS_FIELDS)
        or (impl == "REAL" and mockish)
    )

    observation_present = (
        legacy.get("independent_effect_observation") is True
        or legacy.get("observation_present") is True
        or legacy.get("durable_effect") is True
    )
    attempt_evidenced = legacy.get("attempt_evidenced") is True

    if legacy.get("backend_available") is False:
        return CapabilityOutcome(
            outcome="Unavailable",
            code="legacy_backend_unavailable",
            message="compatibility refusal: missing backend is Unavailable",
            operation=operation,
            backend=token or None,
            envelope=EvidenceEnvelope(effect="not_started", origin="absent"),
            details={**details, "claims_success_clamped": bool(claims_success)},
        )

    if mockish and claims_success:
        return CapabilityOutcome(
            outcome="Unavailable",
            code="legacy_simulated_success_refused",
            message=(
                "compatibility refusal: simulated/mock success cannot project to "
                "Observed/Verified; production disposition is Unavailable"
            ),
            operation=operation,
            backend=token or None,
            envelope=EvidenceEnvelope(origin="simulated", effect="not_started"),
            details={**details, "claims_success_clamped": True, "mockish": True},
        )

    if claims_success and not observation_present:
        if attempt_evidenced:
            return CapabilityOutcome(
                outcome="Unknown",
                code="legacy_success_without_observation",
                message=(
                    "compatibility refusal: legacy success without observation "
                    "is Unknown, not success"
                ),
                operation=operation,
                backend=token or None,
                envelope=EvidenceEnvelope(effect="externally_unknown", origin="declared"),
                details={**details, "claims_success_clamped": True},
            )
        return CapabilityOutcome(
            outcome="Unavailable",
            code="legacy_success_without_effect",
            message=(
                "compatibility refusal: legacy success without durable effect "
                "is Unavailable"
            ),
            operation=operation,
            backend=token or None,
            envelope=EvidenceEnvelope(effect="not_started", origin="absent"),
            details={**details, "claims_success_clamped": True},
        )

    if legacy.get("status") == "error" or legacy.get("success") is False:
        return CapabilityOutcome(
            outcome="Failed",
            code="legacy_explicit_failure",
            message="compatibility projection: legacy failure retained as Failed",
            operation=operation,
            backend=token or None,
            envelope=EvidenceEnvelope(effect="failed", origin="declared"),
            details=details,
        )

    if claims_success and observation_present:
        return CapabilityOutcome(
            outcome="Observed",
            code="inference_observed",
            message="compatibility projection: observation-backed legacy success",
            operation=operation,
            backend=token or None,
            envelope=EvidenceEnvelope(
                origin="hermetic_observed",
                integrity="digest_valid",
                authority="valid",
                policy="allowed",
                effect="observed",
                freshness="current",
            ),
            evidence=frozenset({"independent_effect_observation", "admission_token"}),
            details=details,
        )

    return CapabilityOutcome(
        outcome="Unavailable",
        code="legacy_unclassified_non_success",
        message="compatibility refusal: unclassified legacy result is Unavailable",
        operation=operation,
        backend=token or None,
        envelope=EvidenceEnvelope(),
        details=details,
    )


# Alias used by some call sites / tests.
project_compatibility = refuse_compatibility_success


__all__ = [
    "TASK_ID",
    "GOAL_ID",
    "BUNDLE",
    "EVIDENCE_ID",
    "INTERFACE",
    "SCHEMA",
    "FCA_VOCABULARY_SCHEMA",
    "UNSAFE_PROMOTION",
    "EXPLICIT_TEST_MODE_ENV",
    "CLOSED_OUTCOMES",
    "CPU_BACKENDS",
    "NON_CPU_BACKEND_FAMILIES",
    "INVENTORIED_CAPABILITY_SITES",
    "SIMULATION_NAMESPACES",
    "FORBIDDEN_LEGACY_SUCCESS_FIELDS",
    "CapabilityOutcomeError",
    "RoutingDisposition",
    "SimulationSelection",
    "EvidenceEnvelope",
    "CapabilityProbeEvidence",
    "CapabilityOutcome",
    "RouteDecision",
    "normalize_backend",
    "is_cpu_backend",
    "is_non_cpu_backend",
    "is_explicit_test_mode",
    "current_capability_probe",
    "is_current_capability_evidence",
    "assess_capability_probe",
    "route_backend",
    "select_simulation_namespace",
    "begin_inference_attempt",
    "bind_inference_observation",
    "validate_delegated_inference_receipt",
    "resolve_inference_outcome",
    "refuse_compatibility_success",
    "project_compatibility",
]
