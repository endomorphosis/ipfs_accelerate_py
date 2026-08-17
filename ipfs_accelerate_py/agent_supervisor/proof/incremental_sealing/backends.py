"""Proof-backend capability probing and recursion admission (IPS-029).

Capability is executable evidence, not a documentation flag.  Every probe
records operational prove/verify, signature, direct-computation, aggregation,
recursive-verification, resource, timeout, and cancellation dimensions without
installation or setup side effects.

Recursive self-verification is admitted only when a bounded prove-and-verify
probe succeeds against preconfigured test-only material.  Absent, failed, or
inconclusive recursive probes leave ``recursive_verification`` explicitly
``False`` and select Merkleized manifest aggregation.

Interfaces: ``ProofBackendCapability``, ``BackendCapabilityRegistry``,
``probe_backend_capability``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, Protocol, runtime_checkable

BACKEND_CAPABILITY_EVIDENCE: Final[str] = "ips/backend-capability-matrix@1"
RECURSION_PROBE_EVIDENCE: Final[str] = "ips/recursion-probe@1"
CAPABILITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "proof-backend-capability@1"
)
MATRIX_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "backend-capability-matrix@1"
)
RECURSION_PROBE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "recursion-probe@1"
)

# Closed known backend identifiers.  Anything else is typed unknown/rejected.
KNOWN_BACKEND_IDS: Final[frozenset[str]] = frozenset(
    {
        "groth16",
        "provekit",
        "simulated",
        "integrity",
        "signed_receipt",
        "merkle_manifest",
    }
)

# Trust-baseline backend disposition labels (documentation alignment only;
# operational truth always comes from a live probe).
TRUST_BASELINE_BACKEND_DECISIONS: Final[Mapping[str, str]] = {
    "existing_recursive_backend": "unsupported",
    "groth16": "bounded_declared_computation_only",
    "provekit": "optional_capability_unavailable_is_typed",
    "simulated": "production_seal_forbidden",
    "unknown": "rejected",
}

DEFAULT_RECURSION_PROBE_TIMEOUT_SECONDS: Final[float] = 2.0
DEFAULT_RECURSION_PROBE_MAX_STEPS: Final[int] = 4
DEFAULT_RECURSION_PROBE_MAX_BYTES: Final[int] = 65_536

# Preconfigured test-only material.  Never production keys or circuits.
TEST_ONLY_RECURSION_MATERIAL_ID: Final[str] = (
    "ips/recursion-probe/test-only-material@1"
)
TEST_ONLY_CIRCUIT_ID: Final[str] = "circuit:ips-recursion-probe-test-only@1"
TEST_ONLY_PUBLIC_INPUT: Final[bytes] = (
    b"ips-recursion-probe-public-input-v1\n"
    b"not-for-production\n"
)
TEST_ONLY_WITNESS: Final[bytes] = (
    b"ips-recursion-probe-witness-v1\n"
    b"test-only-secret-never-export\n"
)
TEST_ONLY_CHILD_STATEMENT: Final[bytes] = (
    b"ips-recursion-probe-child-statement-v1\n"
)

_PROVEKIT_EXECUTABLE_NAMES: Final[tuple[str, ...]] = ("provekit-cli", "provekit")


class BackendCapabilityError(ValueError):
    """Fail-closed backend capability contract violation."""


class BackendAvailabilityStatus(str, Enum):
    """Closed operational availability vocabulary."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"
    SIMULATED_ONLY = "simulated_only"


class AggregationDisposition(str, Enum):
    """How aggregation must be labeled after the capability probe."""

    # Explicit plan label for the non-recursive path.
    MANIFEST_AGGREGATION = "manifest_aggregation"
    RECURSIVE_VERIFICATION = "recursive_verification"


class RecursionProbeVerdict(str, Enum):
    """Closed outcomes of the bounded recursive prove-and-verify probe."""

    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    NOT_ATTEMPTED = "not_attempted"
    UNSUPPORTED = "unsupported"
    TIMEOUT = "timeout"
    ERROR = "error"


class CapabilityReasonCode(str, Enum):
    """Stable reason codes for typed capability failures and dispositions."""

    UNKNOWN_BACKEND = "unknown_backend"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    SIMULATED_PRODUCTION_FORBIDDEN = "simulated_production_forbidden"
    RECURSION_NOT_DEMONSTRATED = "recursion_not_demonstrated"
    RECURSION_PROBE_FAILED = "recursion_probe_failed"
    RECURSION_PROBE_INCONCLUSIVE = "recursion_probe_inconclusive"
    RECURSION_PROBE_TIMEOUT = "recursion_probe_timeout"
    RECURSION_PROBE_ERROR = "recursion_probe_error"
    RECURSION_PROBE_PASSED = "recursion_probe_passed"
    OPERATIONAL = "operational"
    INTEGRITY_ONLY = "integrity_only"
    SIGNATURE_STRUCTURAL = "signature_structural"
    BOUNDED_DECLARED_COMPUTATION = "bounded_declared_computation"
    OPTIONAL_CAPABILITY_UNAVAILABLE = "optional_capability_unavailable"


@dataclass(frozen=True, slots=True)
class RecursionProbeMaterial:
    """Preconfigured test-only material for the bounded recursion probe.

    Production key material is never accepted.  ``test_only`` must remain
    ``True``; probes reject any attempt to smuggle production designations.
    """

    material_id: str = TEST_ONLY_RECURSION_MATERIAL_ID
    circuit_id: str = TEST_ONLY_CIRCUIT_ID
    public_input: bytes = TEST_ONLY_PUBLIC_INPUT
    witness: bytes = TEST_ONLY_WITNESS
    child_statement: bytes = TEST_ONLY_CHILD_STATEMENT
    test_only: bool = True
    max_steps: int = DEFAULT_RECURSION_PROBE_MAX_STEPS
    max_bytes: int = DEFAULT_RECURSION_PROBE_MAX_BYTES
    timeout_seconds: float = DEFAULT_RECURSION_PROBE_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if self.test_only is not True:
            raise BackendCapabilityError(
                "recursion probe material must be test_only=True"
            )
        if not isinstance(self.material_id, str) or not self.material_id.strip():
            raise BackendCapabilityError("material_id must be a non-empty string")
        if not isinstance(self.circuit_id, str) or not self.circuit_id.strip():
            raise BackendCapabilityError("circuit_id must be a non-empty string")
        for name in ("public_input", "witness", "child_statement"):
            value = getattr(self, name)
            if not isinstance(value, (bytes, bytearray)):
                raise BackendCapabilityError(f"{name} must be bytes")
            if len(value) == 0:
                raise BackendCapabilityError(f"{name} must be non-empty")
            if len(value) > self.max_bytes:
                raise BackendCapabilityError(f"{name} exceeds max_bytes bound")
        if (
            isinstance(self.max_steps, bool)
            or not isinstance(self.max_steps, int)
            or self.max_steps < 2
        ):
            raise BackendCapabilityError("max_steps must be an int >= 2")
        if (
            isinstance(self.max_bytes, bool)
            or not isinstance(self.max_bytes, int)
            or self.max_bytes < 1
        ):
            raise BackendCapabilityError("max_bytes must be a positive int")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or float(self.timeout_seconds) <= 0
        ):
            raise BackendCapabilityError("timeout_seconds must be a positive number")
        object.__setattr__(self, "material_id", self.material_id.strip())
        object.__setattr__(self, "circuit_id", self.circuit_id.strip())
        object.__setattr__(self, "public_input", bytes(self.public_input))
        object.__setattr__(self, "witness", bytes(self.witness))
        object.__setattr__(self, "child_statement", bytes(self.child_statement))
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))

    def public_input_digest(self) -> str:
        return f"sha256:{hashlib.sha256(self.public_input).hexdigest()}"

    def to_canonical(self) -> dict[str, Any]:
        return {
            "material_id": self.material_id,
            "circuit_id": self.circuit_id,
            "public_input_digest": self.public_input_digest(),
            "child_statement_digest": (
                f"sha256:{hashlib.sha256(self.child_statement).hexdigest()}"
            ),
            "test_only": True,
            "max_steps": self.max_steps,
            "max_bytes": self.max_bytes,
            "timeout_seconds": self.timeout_seconds,
            # Witness bytes are never exported; only a designation flag.
            "witness_present": True,
            "witness_exported": False,
        }


@dataclass(frozen=True, slots=True)
class RecursionProbeArtifact:
    """Hermetic artifact produced by a prove step of the recursion probe."""

    kind: str
    proof_bytes: bytes
    public_input_digest: str
    circuit_id: str
    test_only: bool = True
    child_root: str | None = None

    def __post_init__(self) -> None:
        if self.test_only is not True:
            raise BackendCapabilityError("recursion probe artifacts must be test_only")
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise BackendCapabilityError("artifact kind must be a non-empty string")
        if not isinstance(self.proof_bytes, (bytes, bytearray)):
            raise BackendCapabilityError("proof_bytes must be bytes")
        if not self.proof_bytes:
            raise BackendCapabilityError("proof_bytes must be non-empty")
        if not isinstance(self.public_input_digest, str) or not self.public_input_digest:
            raise BackendCapabilityError("public_input_digest is required")
        if not isinstance(self.circuit_id, str) or not self.circuit_id.strip():
            raise BackendCapabilityError("circuit_id is required")
        object.__setattr__(self, "kind", self.kind.strip())
        object.__setattr__(self, "proof_bytes", bytes(self.proof_bytes))
        object.__setattr__(self, "circuit_id", self.circuit_id.strip())


@dataclass(frozen=True, slots=True)
class RecursionProbeResult:
    """Executable evidence for recursive self-verification admission."""

    schema: str
    verdict: RecursionProbeVerdict
    attempted: bool
    passed: bool
    prove_ok: bool
    verify_ok: bool
    recursive_prove_ok: bool
    recursive_verify_ok: bool
    bounded: bool
    test_only_material: bool
    material_id: str
    steps_executed: int
    duration_ms: int
    reason_code: str
    message: str
    child_root: str | None = None
    aggregate_root: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "verdict",
            RecursionProbeVerdict(str(getattr(self.verdict, "value", self.verdict))),
        )
        if self.passed and self.verdict is not RecursionProbeVerdict.PASSED:
            raise BackendCapabilityError("passed recursion probe requires PASSED verdict")
        if self.passed and not self.attempted:
            raise BackendCapabilityError("passed recursion probe must have been attempted")
        if self.passed and not (
            self.prove_ok
            and self.verify_ok
            and self.recursive_prove_ok
            and self.recursive_verify_ok
            and self.bounded
            and self.test_only_material
        ):
            raise BackendCapabilityError(
                "passed recursion probe requires full bounded prove-and-verify success "
                "on test-only material"
            )
        if type(self.passed) is not bool or type(self.attempted) is not bool:
            raise BackendCapabilityError("attempted/passed must be booleans")

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_subset": RECURSION_PROBE_EVIDENCE,
            "verdict": self.verdict.value,
            "attempted": self.attempted,
            "passed": self.passed,
            "prove_ok": self.prove_ok,
            "verify_ok": self.verify_ok,
            "recursive_prove_ok": self.recursive_prove_ok,
            "recursive_verify_ok": self.recursive_verify_ok,
            "bounded": self.bounded,
            "test_only_material": self.test_only_material,
            "material_id": self.material_id,
            "steps_executed": self.steps_executed,
            "duration_ms": self.duration_ms,
            "reason_code": self.reason_code,
            "message": self.message,
            "child_root": self.child_root,
            "aggregate_root": self.aggregate_root,
            # Never export witness or proving-key material.
            "witness_exported": False,
            "proving_key_exported": False,
        }


@runtime_checkable
class RecursiveProbeBackend(Protocol):
    """Optional adapter that can execute the bounded recursion probe.

    Implementations must operate only on preconfigured test-only material and
    must not install tools, generate production keys, or touch the network.
    """

    def prove_child(
        self, material: RecursionProbeMaterial
    ) -> RecursionProbeArtifact:
        """Prove the child statement under test-only material."""

    def verify_child(
        self, artifact: RecursionProbeArtifact, material: RecursionProbeMaterial
    ) -> bool:
        """Verify a child artifact; return True only on cryptographic/structural success."""

    def prove_recursive(
        self,
        child: RecursionProbeArtifact,
        material: RecursionProbeMaterial,
    ) -> RecursionProbeArtifact:
        """Prove an aggregate that binds verified-child validity under test-only keys."""

    def verify_recursive(
        self, artifact: RecursionProbeArtifact, material: RecursionProbeMaterial
    ) -> bool:
        """Verify the recursive aggregate artifact."""


@dataclass(frozen=True, slots=True)
class ProofBackendCapability:
    """Closed capability report for one proof backend."""

    schema: str
    backend_id: str
    status: BackendAvailabilityStatus
    can_prove: bool
    can_verify: bool
    can_sign: bool
    can_direct_computation: bool
    can_aggregate: bool
    recursive_verification: bool
    supports_resource_limits: bool
    supports_timeout: bool
    supports_cancellation: bool
    aggregation_disposition: AggregationDisposition
    production_seal_allowed: bool
    reason_code: str
    message: str
    recursion_probe: RecursionProbeResult
    trust_baseline_decision: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            BackendAvailabilityStatus(str(getattr(self.status, "value", self.status))),
        )
        object.__setattr__(
            self,
            "aggregation_disposition",
            AggregationDisposition(
                str(
                    getattr(
                        self.aggregation_disposition,
                        "value",
                        self.aggregation_disposition,
                    )
                )
            ),
        )
        if not isinstance(self.backend_id, str) or not self.backend_id.strip():
            raise BackendCapabilityError("backend_id must be a non-empty string")
        object.__setattr__(self, "backend_id", self.backend_id.strip())

        bool_fields = (
            "can_prove",
            "can_verify",
            "can_sign",
            "can_direct_computation",
            "can_aggregate",
            "recursive_verification",
            "supports_resource_limits",
            "supports_timeout",
            "supports_cancellation",
            "production_seal_allowed",
        )
        for name in bool_fields:
            if type(getattr(self, name)) is not bool:
                raise BackendCapabilityError(f"{name} must be a boolean")

        # Fail closed: recursion requires an executable passed probe.
        if self.recursive_verification:
            if not self.recursion_probe.passed:
                raise BackendCapabilityError(
                    "recursive_verification requires a passed recursion probe"
                )
            if self.aggregation_disposition is not AggregationDisposition.RECURSIVE_VERIFICATION:
                raise BackendCapabilityError(
                    "recursive_verification requires recursive aggregation disposition"
                )
            if self.status is not BackendAvailabilityStatus.AVAILABLE:
                raise BackendCapabilityError(
                    "recursive_verification requires an available backend"
                )
            if not self.production_seal_allowed:
                raise BackendCapabilityError(
                    "recursive_verification cannot be admitted when production seals "
                    "are forbidden for this backend"
                )
        else:
            # Explicit non-recursion path.
            if (
                self.aggregation_disposition
                is AggregationDisposition.RECURSIVE_VERIFICATION
            ):
                raise BackendCapabilityError(
                    "aggregation disposition recursive_verification requires "
                    "recursive_verification=True"
                )

        if self.status is BackendAvailabilityStatus.UNKNOWN:
            if any(
                getattr(self, name)
                for name in (
                    "can_prove",
                    "can_verify",
                    "can_sign",
                    "can_direct_computation",
                    "can_aggregate",
                    "recursive_verification",
                    "production_seal_allowed",
                )
            ):
                raise BackendCapabilityError(
                    "unknown backends must not claim operational capabilities"
                )

        if self.status is BackendAvailabilityStatus.SIMULATED_ONLY:
            if self.production_seal_allowed or self.recursive_verification:
                raise BackendCapabilityError(
                    "simulated backends forbid production seals and recursion"
                )

        if not isinstance(self.metadata, Mapping):
            raise BackendCapabilityError("metadata must be a mapping")

    @property
    def available(self) -> bool:
        return self.status is BackendAvailabilityStatus.AVAILABLE

    @property
    def unknown(self) -> bool:
        return self.status is BackendAvailabilityStatus.UNKNOWN

    @property
    def unavailable(self) -> bool:
        return self.status is BackendAvailabilityStatus.UNAVAILABLE

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_subset": BACKEND_CAPABILITY_EVIDENCE,
            "backend_id": self.backend_id,
            "status": self.status.value,
            "can_prove": self.can_prove,
            "can_verify": self.can_verify,
            "can_sign": self.can_sign,
            "can_direct_computation": self.can_direct_computation,
            "can_aggregate": self.can_aggregate,
            "recursive_verification": self.recursive_verification,
            "supports_resource_limits": self.supports_resource_limits,
            "supports_timeout": self.supports_timeout,
            "supports_cancellation": self.supports_cancellation,
            "aggregation_disposition": self.aggregation_disposition.value,
            "production_seal_allowed": self.production_seal_allowed,
            "reason_code": self.reason_code,
            "message": self.message,
            "trust_baseline_decision": self.trust_baseline_decision,
            "recursion_probe": self.recursion_probe.to_canonical(),
            "metadata": dict(self.metadata),
        }

    def to_canonical_json(self) -> str:
        return json.dumps(
            self.to_canonical(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )


@dataclass(frozen=True, slots=True)
class BackendCapabilityMatrix:
    """Closed multi-backend capability matrix (``ips/backend-capability-matrix@1``)."""

    schema: str
    capabilities: tuple[ProofBackendCapability, ...]
    any_recursive_verification: bool
    aggregation_disposition: AggregationDisposition
    probed_backend_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "aggregation_disposition",
            AggregationDisposition(
                str(
                    getattr(
                        self.aggregation_disposition,
                        "value",
                        self.aggregation_disposition,
                    )
                )
            ),
        )
        recursive = tuple(
            item for item in self.capabilities if item.recursive_verification
        )
        computed = bool(recursive)
        if self.any_recursive_verification is not computed:
            raise BackendCapabilityError(
                "any_recursive_verification must match capability entries"
            )
        if computed:
            if (
                self.aggregation_disposition
                is not AggregationDisposition.RECURSIVE_VERIFICATION
            ):
                raise BackendCapabilityError(
                    "matrix with recursive backends must use recursive disposition"
                )
        else:
            if (
                self.aggregation_disposition
                is AggregationDisposition.RECURSIVE_VERIFICATION
            ):
                raise BackendCapabilityError(
                    "matrix without recursive backends must use manifest aggregation"
                )

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_subset": BACKEND_CAPABILITY_EVIDENCE,
            "probed_backend_ids": list(self.probed_backend_ids),
            "any_recursive_verification": self.any_recursive_verification,
            "aggregation_disposition": self.aggregation_disposition.value,
            "unsupported_recursion_fallback": (
                AggregationDisposition.MANIFEST_AGGREGATION.value
            ),
            "capabilities": [item.to_canonical() for item in self.capabilities],
            "trust_baseline_backend_decisions": dict(TRUST_BASELINE_BACKEND_DECISIONS),
        }


def _not_attempted_probe(
    *,
    reason_code: str,
    message: str,
    verdict: RecursionProbeVerdict = RecursionProbeVerdict.NOT_ATTEMPTED,
    material_id: str = TEST_ONLY_RECURSION_MATERIAL_ID,
) -> RecursionProbeResult:
    return RecursionProbeResult(
        schema=RECURSION_PROBE_SCHEMA,
        verdict=verdict,
        attempted=False,
        passed=False,
        prove_ok=False,
        verify_ok=False,
        recursive_prove_ok=False,
        recursive_verify_ok=False,
        bounded=True,
        test_only_material=True,
        material_id=material_id,
        steps_executed=0,
        duration_ms=0,
        reason_code=reason_code,
        message=message,
    )


def _failed_probe(
    *,
    verdict: RecursionProbeVerdict,
    reason_code: str,
    message: str,
    material: RecursionProbeMaterial,
    prove_ok: bool = False,
    verify_ok: bool = False,
    recursive_prove_ok: bool = False,
    recursive_verify_ok: bool = False,
    steps_executed: int = 0,
    duration_ms: int = 0,
    child_root: str | None = None,
    aggregate_root: str | None = None,
) -> RecursionProbeResult:
    return RecursionProbeResult(
        schema=RECURSION_PROBE_SCHEMA,
        verdict=verdict,
        attempted=True,
        passed=False,
        prove_ok=prove_ok,
        verify_ok=verify_ok,
        recursive_prove_ok=recursive_prove_ok,
        recursive_verify_ok=recursive_verify_ok,
        bounded=True,
        test_only_material=True,
        material_id=material.material_id,
        steps_executed=steps_executed,
        duration_ms=duration_ms,
        reason_code=reason_code,
        message=message,
        child_root=child_root,
        aggregate_root=aggregate_root,
    )


def _passed_probe(
    *,
    material: RecursionProbeMaterial,
    steps_executed: int,
    duration_ms: int,
    child_root: str,
    aggregate_root: str,
) -> RecursionProbeResult:
    return RecursionProbeResult(
        schema=RECURSION_PROBE_SCHEMA,
        verdict=RecursionProbeVerdict.PASSED,
        attempted=True,
        passed=True,
        prove_ok=True,
        verify_ok=True,
        recursive_prove_ok=True,
        recursive_verify_ok=True,
        bounded=True,
        test_only_material=True,
        material_id=material.material_id,
        steps_executed=steps_executed,
        duration_ms=duration_ms,
        reason_code=CapabilityReasonCode.RECURSION_PROBE_PASSED.value,
        message=(
            "bounded prove-and-verify recursion probe passed on preconfigured "
            "test-only material"
        ),
        child_root=child_root,
        aggregate_root=aggregate_root,
    )


def run_bounded_recursion_probe(
    backend: RecursiveProbeBackend,
    material: RecursionProbeMaterial | None = None,
    *,
    monotonic: Callable[[], float] | None = None,
) -> RecursionProbeResult:
    """Execute a bounded prove-and-verify recursion probe.

    Steps (hard-capped by ``material.max_steps`` and ``timeout_seconds``):

    1. prove child under test-only material
    2. verify child
    3. prove recursive aggregate binding the verified child
    4. verify recursive aggregate

    Any missing success, timeout, oversize artifact, non-test-only material, or
    raised exception yields a non-passing typed result.  Recursion is never
    inferred from documentation or mere availability.
    """

    checked = material or RecursionProbeMaterial()
    if checked.test_only is not True:
        return _failed_probe(
            verdict=RecursionProbeVerdict.FAILED,
            reason_code=CapabilityReasonCode.RECURSION_PROBE_FAILED.value,
            message="recursion probe rejected non-test-only material",
            material=checked,
        )

    clock = monotonic or time.monotonic
    started = clock()
    steps = 0
    child_root: str | None = None
    aggregate_root: str | None = None

    def _elapsed_ms() -> int:
        return max(0, int((clock() - started) * 1000))

    def _timed_out() -> bool:
        return (clock() - started) > checked.timeout_seconds

    try:
        if _timed_out():
            return _failed_probe(
                verdict=RecursionProbeVerdict.TIMEOUT,
                reason_code=CapabilityReasonCode.RECURSION_PROBE_TIMEOUT.value,
                message="recursion probe timed out before prove_child",
                material=checked,
                duration_ms=_elapsed_ms(),
            )

        child = backend.prove_child(checked)
        steps += 1
        if steps > checked.max_steps:
            return _failed_probe(
                verdict=RecursionProbeVerdict.FAILED,
                reason_code=CapabilityReasonCode.RECURSION_PROBE_FAILED.value,
                message="recursion probe exceeded max_steps during prove_child",
                material=checked,
                steps_executed=steps,
                duration_ms=_elapsed_ms(),
            )
        if child.test_only is not True:
            return _failed_probe(
                verdict=RecursionProbeVerdict.FAILED,
                reason_code=CapabilityReasonCode.RECURSION_PROBE_FAILED.value,
                message="child artifact must be test_only",
                material=checked,
                prove_ok=False,
                steps_executed=steps,
                duration_ms=_elapsed_ms(),
            )
        if len(child.proof_bytes) > checked.max_bytes:
            return _failed_probe(
                verdict=RecursionProbeVerdict.FAILED,
                reason_code=CapabilityReasonCode.RECURSION_PROBE_FAILED.value,
                message="child proof exceeds max_bytes bound",
                material=checked,
                prove_ok=False,
                steps_executed=steps,
                duration_ms=_elapsed_ms(),
            )
        if child.circuit_id != checked.circuit_id:
            return _failed_probe(
                verdict=RecursionProbeVerdict.FAILED,
                reason_code=CapabilityReasonCode.RECURSION_PROBE_FAILED.value,
                message="child artifact circuit_id does not match test-only material",
                material=checked,
                prove_ok=False,
                steps_executed=steps,
                duration_ms=_elapsed_ms(),
            )
        child_root = (
            f"sha256:{hashlib.sha256(child.proof_bytes).hexdigest()}"
        )
        prove_ok = True

        if _timed_out():
            return _failed_probe(
                verdict=RecursionProbeVerdict.TIMEOUT,
                reason_code=CapabilityReasonCode.RECURSION_PROBE_TIMEOUT.value,
                message="recursion probe timed out before verify_child",
                material=checked,
                prove_ok=True,
                steps_executed=steps,
                duration_ms=_elapsed_ms(),
                child_root=child_root,
            )

        child_ok = backend.verify_child(child, checked) is True
        steps += 1
        if not child_ok:
            return _failed_probe(
                verdict=RecursionProbeVerdict.FAILED,
                reason_code=CapabilityReasonCode.RECURSION_PROBE_FAILED.value,
                message="child verify failed during recursion probe",
                material=checked,
                prove_ok=True,
                verify_ok=False,
                steps_executed=steps,
                duration_ms=_elapsed_ms(),
                child_root=child_root,
            )
        verify_ok = True

        if _timed_out():
            return _failed_probe(
                verdict=RecursionProbeVerdict.TIMEOUT,
                reason_code=CapabilityReasonCode.RECURSION_PROBE_TIMEOUT.value,
                message="recursion probe timed out before prove_recursive",
                material=checked,
                prove_ok=True,
                verify_ok=True,
                steps_executed=steps,
                duration_ms=_elapsed_ms(),
                child_root=child_root,
            )

        aggregate = backend.prove_recursive(child, checked)
        steps += 1
        if steps > checked.max_steps:
            return _failed_probe(
                verdict=RecursionProbeVerdict.FAILED,
                reason_code=CapabilityReasonCode.RECURSION_PROBE_FAILED.value,
                message="recursion probe exceeded max_steps during prove_recursive",
                material=checked,
                prove_ok=True,
                verify_ok=True,
                steps_executed=steps,
                duration_ms=_elapsed_ms(),
                child_root=child_root,
            )
        if aggregate.test_only is not True:
            return _failed_probe(
                verdict=RecursionProbeVerdict.FAILED,
                reason_code=CapabilityReasonCode.RECURSION_PROBE_FAILED.value,
                message="recursive artifact must be test_only",
                material=checked,
                prove_ok=True,
                verify_ok=True,
                recursive_prove_ok=False,
                steps_executed=steps,
                duration_ms=_elapsed_ms(),
                child_root=child_root,
            )
        if len(aggregate.proof_bytes) > checked.max_bytes:
            return _failed_probe(
                verdict=RecursionProbeVerdict.FAILED,
                reason_code=CapabilityReasonCode.RECURSION_PROBE_FAILED.value,
                message="recursive proof exceeds max_bytes bound",
                material=checked,
                prove_ok=True,
                verify_ok=True,
                recursive_prove_ok=False,
                steps_executed=steps,
                duration_ms=_elapsed_ms(),
                child_root=child_root,
            )
        if aggregate.kind != "recursive":
            return _failed_probe(
                verdict=RecursionProbeVerdict.FAILED,
                reason_code=CapabilityReasonCode.RECURSION_PROBE_FAILED.value,
                message="recursive artifact kind must be 'recursive'",
                material=checked,
                prove_ok=True,
                verify_ok=True,
                recursive_prove_ok=False,
                steps_executed=steps,
                duration_ms=_elapsed_ms(),
                child_root=child_root,
            )
        if not aggregate.child_root or not hmac.compare_digest(
            str(aggregate.child_root), str(child_root)
        ):
            return _failed_probe(
                verdict=RecursionProbeVerdict.FAILED,
                reason_code=CapabilityReasonCode.RECURSION_PROBE_FAILED.value,
                message="recursive artifact must bind the verified child root",
                material=checked,
                prove_ok=True,
                verify_ok=True,
                recursive_prove_ok=False,
                steps_executed=steps,
                duration_ms=_elapsed_ms(),
                child_root=child_root,
            )
        aggregate_root = (
            f"sha256:{hashlib.sha256(aggregate.proof_bytes).hexdigest()}"
        )
        recursive_prove_ok = True

        if _timed_out():
            return _failed_probe(
                verdict=RecursionProbeVerdict.TIMEOUT,
                reason_code=CapabilityReasonCode.RECURSION_PROBE_TIMEOUT.value,
                message="recursion probe timed out before verify_recursive",
                material=checked,
                prove_ok=True,
                verify_ok=True,
                recursive_prove_ok=True,
                steps_executed=steps,
                duration_ms=_elapsed_ms(),
                child_root=child_root,
                aggregate_root=aggregate_root,
            )

        recursive_ok = backend.verify_recursive(aggregate, checked) is True
        steps += 1
        if not recursive_ok:
            return _failed_probe(
                verdict=RecursionProbeVerdict.FAILED,
                reason_code=CapabilityReasonCode.RECURSION_PROBE_FAILED.value,
                message="recursive verify failed during recursion probe",
                material=checked,
                prove_ok=True,
                verify_ok=True,
                recursive_prove_ok=True,
                recursive_verify_ok=False,
                steps_executed=steps,
                duration_ms=_elapsed_ms(),
                child_root=child_root,
                aggregate_root=aggregate_root,
            )

        if steps > checked.max_steps:
            return _failed_probe(
                verdict=RecursionProbeVerdict.FAILED,
                reason_code=CapabilityReasonCode.RECURSION_PROBE_FAILED.value,
                message="recursion probe exceeded max_steps after verify_recursive",
                material=checked,
                prove_ok=True,
                verify_ok=True,
                recursive_prove_ok=True,
                recursive_verify_ok=True,
                steps_executed=steps,
                duration_ms=_elapsed_ms(),
                child_root=child_root,
                aggregate_root=aggregate_root,
            )

        return _passed_probe(
            material=checked,
            steps_executed=steps,
            duration_ms=_elapsed_ms(),
            child_root=child_root or "",
            aggregate_root=aggregate_root or "",
        )
    except Exception as exc:  # noqa: BLE001 - fail closed on any probe fault
        return _failed_probe(
            verdict=RecursionProbeVerdict.ERROR,
            reason_code=CapabilityReasonCode.RECURSION_PROBE_ERROR.value,
            message=f"recursion probe error: {type(exc).__name__}: {exc}",
            material=checked,
            prove_ok=False,
            steps_executed=steps,
            duration_ms=_elapsed_ms(),
            child_root=child_root,
            aggregate_root=aggregate_root,
        )


@dataclass
class HermeticTestOnlyRecursiveBackend:
    """In-process recursive probe backend using only preconfigured test material.

    Intended for tests and hermetic demonstration.  Uses HMAC-style digests over
    test-only public inputs and witnesses; never production keys.
    """

    backend_id: str = "hermetic-test-only-recursive"

    def _child_mac(self, material: RecursionProbeMaterial) -> bytes:
        key = hashlib.sha256(
            b"ips-recursion-probe-test-only-key\n" + material.witness
        ).digest()
        return hmac.new(
            key,
            material.child_statement + material.public_input + material.circuit_id.encode(),
            hashlib.sha256,
        ).digest()

    def prove_child(
        self, material: RecursionProbeMaterial
    ) -> RecursionProbeArtifact:
        proof = self._child_mac(material)
        return RecursionProbeArtifact(
            kind="child",
            proof_bytes=proof,
            public_input_digest=material.public_input_digest(),
            circuit_id=material.circuit_id,
            test_only=True,
        )

    def verify_child(
        self, artifact: RecursionProbeArtifact, material: RecursionProbeMaterial
    ) -> bool:
        if artifact.kind != "child" or artifact.test_only is not True:
            return False
        if artifact.circuit_id != material.circuit_id:
            return False
        if artifact.public_input_digest != material.public_input_digest():
            return False
        expected = self._child_mac(material)
        return hmac.compare_digest(artifact.proof_bytes, expected)

    def prove_recursive(
        self,
        child: RecursionProbeArtifact,
        material: RecursionProbeMaterial,
    ) -> RecursionProbeArtifact:
        if not self.verify_child(child, material):
            raise BackendCapabilityError("cannot recurse over unverified child")
        child_root = f"sha256:{hashlib.sha256(child.proof_bytes).hexdigest()}"
        key = hashlib.sha256(
            b"ips-recursion-probe-test-only-recursive-key\n" + material.witness
        ).digest()
        proof = hmac.new(
            key,
            child_root.encode()
            + material.public_input
            + material.circuit_id.encode()
            + b"recursive",
            hashlib.sha256,
        ).digest()
        return RecursionProbeArtifact(
            kind="recursive",
            proof_bytes=proof,
            public_input_digest=material.public_input_digest(),
            circuit_id=material.circuit_id,
            test_only=True,
            child_root=child_root,
        )

    def verify_recursive(
        self, artifact: RecursionProbeArtifact, material: RecursionProbeMaterial
    ) -> bool:
        if artifact.kind != "recursive" or artifact.test_only is not True:
            return False
        if artifact.circuit_id != material.circuit_id:
            return False
        if not artifact.child_root:
            return False
        key = hashlib.sha256(
            b"ips-recursion-probe-test-only-recursive-key\n" + material.witness
        ).digest()
        expected = hmac.new(
            key,
            artifact.child_root.encode()
            + material.public_input
            + material.circuit_id.encode()
            + b"recursive",
            hashlib.sha256,
        ).digest()
        return hmac.compare_digest(artifact.proof_bytes, expected)


def _module_spec_present(module: str) -> bool:
    """Locate a module without importing provider packages (no side effects)."""

    import importlib.machinery

    parts = [part for part in str(module).split(".") if part]
    if not parts:
        return False
    spec = None
    parent = ""
    for index, part in enumerate(parts):
        name = part if not parent else f"{parent}.{part}"
        path = None if spec is None else getattr(spec, "submodule_search_locations", None)
        if index == 0:
            spec = importlib.machinery.PathFinder.find_spec(name)
        else:
            if not path:
                return False
            spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None:
            return False
        parent = name
    return True


def _which_first(names: Sequence[str], which: Callable[[str], str | None]) -> str | None:
    for name in names:
        found = which(name)
        if found:
            return found
    return None


def _trust_decision_for(backend_id: str) -> str:
    if backend_id not in KNOWN_BACKEND_IDS:
        return TRUST_BASELINE_BACKEND_DECISIONS["unknown"]
    if backend_id == "groth16":
        return TRUST_BASELINE_BACKEND_DECISIONS["groth16"]
    if backend_id == "provekit":
        return TRUST_BASELINE_BACKEND_DECISIONS["provekit"]
    if backend_id == "simulated":
        return TRUST_BASELINE_BACKEND_DECISIONS["simulated"]
    if backend_id in {"integrity", "signed_receipt", "merkle_manifest"}:
        return "manifest_or_integrity_only"
    return TRUST_BASELINE_BACKEND_DECISIONS["existing_recursive_backend"]


def _disposition_for(recursive: bool) -> AggregationDisposition:
    if recursive:
        return AggregationDisposition.RECURSIVE_VERIFICATION
    return AggregationDisposition.MANIFEST_AGGREGATION


def _capability(
    *,
    backend_id: str,
    status: BackendAvailabilityStatus,
    can_prove: bool,
    can_verify: bool,
    can_sign: bool,
    can_direct_computation: bool,
    can_aggregate: bool,
    recursive_verification: bool,
    supports_resource_limits: bool,
    supports_timeout: bool,
    supports_cancellation: bool,
    production_seal_allowed: bool,
    reason_code: str,
    message: str,
    recursion_probe: RecursionProbeResult,
    metadata: Mapping[str, Any] | None = None,
) -> ProofBackendCapability:
    return ProofBackendCapability(
        schema=CAPABILITY_SCHEMA,
        backend_id=backend_id,
        status=status,
        can_prove=can_prove,
        can_verify=can_verify,
        can_sign=can_sign,
        can_direct_computation=can_direct_computation,
        can_aggregate=can_aggregate,
        recursive_verification=recursive_verification,
        supports_resource_limits=supports_resource_limits,
        supports_timeout=supports_timeout,
        supports_cancellation=supports_cancellation,
        aggregation_disposition=_disposition_for(recursive_verification),
        production_seal_allowed=production_seal_allowed,
        reason_code=reason_code,
        message=message,
        recursion_probe=recursion_probe,
        trust_baseline_decision=_trust_decision_for(backend_id),
        metadata=dict(metadata or {}),
    )


def _unknown_capability(backend_id: str) -> ProofBackendCapability:
    probe = _not_attempted_probe(
        reason_code=CapabilityReasonCode.UNKNOWN_BACKEND.value,
        message="unknown backend; recursion probe not attempted",
        verdict=RecursionProbeVerdict.UNSUPPORTED,
    )
    return _capability(
        backend_id=backend_id,
        status=BackendAvailabilityStatus.UNKNOWN,
        can_prove=False,
        can_verify=False,
        can_sign=False,
        can_direct_computation=False,
        can_aggregate=False,
        recursive_verification=False,
        supports_resource_limits=False,
        supports_timeout=False,
        supports_cancellation=False,
        production_seal_allowed=False,
        reason_code=CapabilityReasonCode.UNKNOWN_BACKEND.value,
        message=(
            f"unknown proof backend {backend_id!r}; rejected by closed registry "
            f"(known={sorted(KNOWN_BACKEND_IDS)})"
        ),
        recursion_probe=probe,
    )


def _unavailable_capability(
    backend_id: str,
    *,
    reason_code: str,
    message: str,
    metadata: Mapping[str, Any] | None = None,
) -> ProofBackendCapability:
    probe = _not_attempted_probe(
        reason_code=reason_code,
        message="backend unavailable; recursion probe not attempted",
        verdict=RecursionProbeVerdict.UNSUPPORTED,
    )
    return _capability(
        backend_id=backend_id,
        status=BackendAvailabilityStatus.UNAVAILABLE,
        can_prove=False,
        can_verify=False,
        can_sign=False,
        can_direct_computation=False,
        can_aggregate=False,
        recursive_verification=False,
        supports_resource_limits=False,
        supports_timeout=False,
        supports_cancellation=False,
        production_seal_allowed=False,
        reason_code=reason_code,
        message=message,
        recursion_probe=probe,
        metadata=metadata,
    )


def _evaluate_recursion(
    *,
    backend_id: str,
    recursive_backend: RecursiveProbeBackend | None,
    material: RecursionProbeMaterial,
    allow_recursion_probe: bool,
) -> RecursionProbeResult:
    if not allow_recursion_probe:
        return _not_attempted_probe(
            reason_code=CapabilityReasonCode.RECURSION_NOT_DEMONSTRATED.value,
            message=(
                f"recursion probe disabled for backend {backend_id!r}; "
                "recursive_verification remains explicitly false"
            ),
            verdict=RecursionProbeVerdict.NOT_ATTEMPTED,
            material_id=material.material_id,
        )
    if recursive_backend is None:
        return _not_attempted_probe(
            reason_code=CapabilityReasonCode.RECURSION_NOT_DEMONSTRATED.value,
            message=(
                f"no recursive probe adapter configured for backend {backend_id!r}; "
                "selecting Merkleized manifest aggregation"
            ),
            verdict=RecursionProbeVerdict.UNSUPPORTED,
            material_id=material.material_id,
        )
    return run_bounded_recursion_probe(recursive_backend, material)


def probe_backend_capability(
    backend_id: str,
    *,
    recursive_backend: RecursiveProbeBackend | None = None,
    material: RecursionProbeMaterial | None = None,
    allow_recursion_probe: bool = True,
    which: Callable[[str], str | None] | None = None,
    availability_overrides: Mapping[str, bool] | None = None,
) -> ProofBackendCapability:
    """Probe one backend and admit recursion only when demonstrated.

    Parameters
    ----------
    backend_id:
        Closed backend identifier.  Unknown IDs fail typed as ``unknown``.
    recursive_backend:
        Optional adapter implementing the bounded prove-and-verify recursion
        probe against preconfigured test-only material.  Absence leaves
        ``recursive_verification`` explicitly ``False``.
    material:
        Preconfigured test-only material.  Defaults to the fixed IPS probe
        vectors; production designations are rejected.
    allow_recursion_probe:
        When ``False``, recursion is never attempted and remains ``False``.
    which:
        Optional executable lookup (defaults to ``shutil.which``).  Used only
        for optional ProveKit discovery; never installs tools.
    availability_overrides:
        Test/injection map of backend_id -> available bool.  Does not enable
        recursion by itself.
    """

    if not isinstance(backend_id, str) or not backend_id.strip():
        raise BackendCapabilityError("backend_id must be a non-empty string")
    backend_id = backend_id.strip()
    checked_material = material or RecursionProbeMaterial()
    which_fn = which or shutil.which
    overrides = dict(availability_overrides or {})

    if backend_id not in KNOWN_BACKEND_IDS:
        return _unknown_capability(backend_id)

    # --- simulated: discoverable but production-forbidden --------------------
    if backend_id == "simulated":
        probe = _not_attempted_probe(
            reason_code=CapabilityReasonCode.SIMULATED_PRODUCTION_FORBIDDEN.value,
            message=(
                "simulated backend never admits recursive verification or "
                "production seals"
            ),
            verdict=RecursionProbeVerdict.UNSUPPORTED,
            material_id=checked_material.material_id,
        )
        return _capability(
            backend_id=backend_id,
            status=BackendAvailabilityStatus.SIMULATED_ONLY,
            can_prove=True,
            can_verify=True,
            can_sign=False,
            can_direct_computation=False,
            can_aggregate=False,
            recursive_verification=False,
            supports_resource_limits=True,
            supports_timeout=True,
            supports_cancellation=True,
            production_seal_allowed=False,
            reason_code=CapabilityReasonCode.SIMULATED_PRODUCTION_FORBIDDEN.value,
            message=(
                "simulated backend is operational only as a non-cryptographic lane; "
                "production seals and recursive verification are forbidden"
            ),
            recursion_probe=probe,
            metadata={"simulated": True},
        )

    # --- integrity / merkle manifest / signed receipt ------------------------
    if backend_id == "integrity":
        probe = _evaluate_recursion(
            backend_id=backend_id,
            recursive_backend=None,
            material=checked_material,
            allow_recursion_probe=False,
        )
        return _capability(
            backend_id=backend_id,
            status=BackendAvailabilityStatus.AVAILABLE,
            can_prove=False,
            can_verify=True,
            can_sign=False,
            can_direct_computation=False,
            can_aggregate=False,
            recursive_verification=False,
            supports_resource_limits=True,
            supports_timeout=True,
            supports_cancellation=True,
            production_seal_allowed=True,
            reason_code=CapabilityReasonCode.INTEGRITY_ONLY.value,
            message="integrity backend verifies digests/CIDs only; no recursion",
            recursion_probe=probe,
        )

    if backend_id == "signed_receipt":
        probe = _evaluate_recursion(
            backend_id=backend_id,
            recursive_backend=None,
            material=checked_material,
            allow_recursion_probe=False,
        )
        return _capability(
            backend_id=backend_id,
            status=BackendAvailabilityStatus.AVAILABLE,
            can_prove=False,
            can_verify=True,
            can_sign=True,
            can_direct_computation=False,
            can_aggregate=False,
            recursive_verification=False,
            supports_resource_limits=True,
            supports_timeout=True,
            supports_cancellation=True,
            production_seal_allowed=True,
            reason_code=CapabilityReasonCode.SIGNATURE_STRUCTURAL.value,
            message=(
                "signed-receipt backend verifies allowlisted signatures; "
                "does not prove direct execution or recursion"
            ),
            recursion_probe=probe,
        )

    if backend_id == "merkle_manifest":
        probe = _evaluate_recursion(
            backend_id=backend_id,
            recursive_backend=None,
            material=checked_material,
            allow_recursion_probe=False,
        )
        return _capability(
            backend_id=backend_id,
            status=BackendAvailabilityStatus.AVAILABLE,
            can_prove=False,
            can_verify=True,
            can_sign=False,
            can_direct_computation=False,
            can_aggregate=True,
            recursive_verification=False,
            supports_resource_limits=True,
            supports_timeout=True,
            supports_cancellation=True,
            production_seal_allowed=True,
            reason_code=CapabilityReasonCode.OPERATIONAL.value,
            message=(
                "merkle_manifest backend aggregates via individually verified leaves "
                "and a Merkle completeness commitment; recursive_verification=false"
            ),
            recursion_probe=probe,
            metadata={
                "aggregation_mode": AggregationDisposition.MANIFEST_AGGREGATION.value
            },
        )

    # --- provekit: optional; unavailable is typed ----------------------------
    if backend_id == "provekit":
        if backend_id in overrides:
            available = overrides[backend_id] is True
            executable = "override" if available else ""
        else:
            executable = _which_first(_PROVEKIT_EXECUTABLE_NAMES, which_fn) or ""
            available = bool(executable)
        if not available:
            return _unavailable_capability(
                backend_id,
                reason_code=CapabilityReasonCode.OPTIONAL_CAPABILITY_UNAVAILABLE.value,
                message=(
                    "provekit optional capability unavailable; typed unavailable "
                    "(no installation or setup performed)"
                ),
                metadata={"executable": ""},
            )
        # Available still does not imply recursion without the probe.
        if recursive_backend is not None and allow_recursion_probe:
            probe = run_bounded_recursion_probe(recursive_backend, checked_material)
            recursive = probe.passed is True
        else:
            probe = _not_attempted_probe(
                reason_code=CapabilityReasonCode.RECURSION_NOT_DEMONSTRATED.value,
                message=(
                    "provekit present but recursive self-verification not demonstrated; "
                    "manifest aggregation selected"
                ),
                verdict=RecursionProbeVerdict.UNSUPPORTED,
                material_id=checked_material.material_id,
            )
            recursive = False
        return _capability(
            backend_id=backend_id,
            status=BackendAvailabilityStatus.AVAILABLE,
            can_prove=True,
            can_verify=True,
            can_sign=False,
            can_direct_computation=True,
            can_aggregate=True,
            recursive_verification=recursive,
            supports_resource_limits=True,
            supports_timeout=True,
            supports_cancellation=True,
            production_seal_allowed=True,
            reason_code=(
                CapabilityReasonCode.RECURSION_PROBE_PASSED.value
                if recursive
                else CapabilityReasonCode.RECURSION_NOT_DEMONSTRATED.value
            ),
            message=(
                "provekit operational; recursive_verification admitted"
                if recursive
                else (
                    "provekit operational for bounded declared computation; "
                    "recursive_verification explicitly false without successful "
                    "bounded prove-and-verify probe"
                )
            ),
            recursion_probe=probe,
            metadata={"executable": executable},
        )

    # --- groth16: bounded declared computation; recursion only via probe -----
    if backend_id == "groth16":
        if backend_id in overrides:
            available = overrides[backend_id] is True
            surface_present = available
        else:
            surface_present = _module_spec_present(
                "ipfs_datasets_py.ipfs_datasets_py.logic.zkp.backends.groth16"
            ) or _module_spec_present("ipfs_datasets_py.logic.zkp.backends.groth16")
            available = surface_present
        if not available:
            return _unavailable_capability(
                backend_id,
                reason_code=CapabilityReasonCode.BACKEND_UNAVAILABLE.value,
                message=(
                    "groth16 backend surface unavailable; typed unavailable "
                    "(no installation or setup performed)"
                ),
                metadata={"surface_present": False},
            )

        if recursive_backend is not None and allow_recursion_probe:
            probe = run_bounded_recursion_probe(recursive_backend, checked_material)
            recursive = probe.passed is True
        else:
            probe = _not_attempted_probe(
                reason_code=CapabilityReasonCode.RECURSION_NOT_DEMONSTRATED.value,
                message=(
                    "groth16 available for bounded declared computation only; "
                    "recursive self-verification not demonstrated "
                    f"(trust baseline: {TRUST_BASELINE_BACKEND_DECISIONS['existing_recursive_backend']})"
                ),
                verdict=RecursionProbeVerdict.UNSUPPORTED,
                material_id=checked_material.material_id,
            )
            recursive = False

        return _capability(
            backend_id=backend_id,
            status=BackendAvailabilityStatus.AVAILABLE,
            can_prove=True,
            can_verify=True,
            can_sign=False,
            can_direct_computation=True,
            can_aggregate=True,
            recursive_verification=recursive,
            supports_resource_limits=True,
            supports_timeout=True,
            supports_cancellation=True,
            production_seal_allowed=True,
            reason_code=(
                CapabilityReasonCode.RECURSION_PROBE_PASSED.value
                if recursive
                else CapabilityReasonCode.BOUNDED_DECLARED_COMPUTATION.value
            ),
            message=(
                "groth16 recursive_verification admitted after bounded prove-and-verify probe"
                if recursive
                else (
                    "groth16 operational for bounded declared computation; "
                    "recursive_verification explicitly false; "
                    "aggregation disposition is manifest_aggregation"
                )
            ),
            recursion_probe=probe,
            metadata={
                "surface_present": surface_present,
                "bounded_declared_computation_only": not recursive,
            },
        )

    # Defensive: known set should have been handled above.
    return _unknown_capability(backend_id)


class BackendCapabilityRegistry:
    """Registry that probes and caches closed backend capability reports."""

    def __init__(
        self,
        *,
        backend_ids: Sequence[str] | None = None,
        recursive_backends: Mapping[str, RecursiveProbeBackend] | None = None,
        material: RecursionProbeMaterial | None = None,
        which: Callable[[str], str | None] | None = None,
        availability_overrides: Mapping[str, bool] | None = None,
        allow_recursion_probe: bool = True,
    ) -> None:
        self._backend_ids = tuple(
            sorted({str(item).strip() for item in (backend_ids or sorted(KNOWN_BACKEND_IDS)) if str(item).strip()})
        )
        if not self._backend_ids:
            raise BackendCapabilityError("registry requires at least one backend id")
        self._recursive_backends = dict(recursive_backends or {})
        self._material = material or RecursionProbeMaterial()
        self._which = which
        self._availability_overrides = dict(availability_overrides or {})
        self._allow_recursion_probe = bool(allow_recursion_probe)
        self._cache: dict[str, ProofBackendCapability] = {}

    @property
    def backend_ids(self) -> tuple[str, ...]:
        return self._backend_ids

    def clear_cache(self) -> None:
        self._cache.clear()

    def probe(
        self,
        backend_id: str,
        *,
        refresh: bool = False,
    ) -> ProofBackendCapability:
        key = str(backend_id).strip()
        if not refresh and key in self._cache:
            return self._cache[key]
        capability = probe_backend_capability(
            key,
            recursive_backend=self._recursive_backends.get(key),
            material=self._material,
            allow_recursion_probe=self._allow_recursion_probe,
            which=self._which,
            availability_overrides=self._availability_overrides,
        )
        self._cache[key] = capability
        return capability

    def probe_all(self, *, refresh: bool = False) -> tuple[ProofBackendCapability, ...]:
        return tuple(self.probe(backend_id, refresh=refresh) for backend_id in self._backend_ids)

    def get(self, backend_id: str) -> ProofBackendCapability | None:
        return self._cache.get(str(backend_id).strip())

    def require(self, backend_id: str) -> ProofBackendCapability:
        capability = self.probe(backend_id)
        if capability.unknown:
            raise BackendCapabilityError(
                f"unknown backend {backend_id!r}: {capability.message}"
            )
        if capability.unavailable:
            raise BackendCapabilityError(
                f"unavailable backend {backend_id!r}: {capability.message}"
            )
        return capability

    def recursive_verification_admitted(self, backend_id: str) -> bool:
        return self.probe(backend_id).recursive_verification is True

    def matrix(self, *, refresh: bool = False) -> BackendCapabilityMatrix:
        capabilities = self.probe_all(refresh=refresh)
        any_recursive = any(item.recursive_verification for item in capabilities)
        return BackendCapabilityMatrix(
            schema=MATRIX_SCHEMA,
            capabilities=capabilities,
            any_recursive_verification=any_recursive,
            aggregation_disposition=_disposition_for(any_recursive),
            probed_backend_ids=tuple(item.backend_id for item in capabilities),
        )

    def to_canonical_matrix(self, *, refresh: bool = False) -> dict[str, Any]:
        return self.matrix(refresh=refresh).to_canonical()


def closed_known_backend_ids() -> frozenset[str]:
    return KNOWN_BACKEND_IDS


def closed_capability_reason_codes() -> frozenset[str]:
    return frozenset(item.value for item in CapabilityReasonCode)


def closed_aggregation_dispositions() -> frozenset[str]:
    return frozenset(item.value for item in AggregationDisposition)


__all__ = (
    "BACKEND_CAPABILITY_EVIDENCE",
    "RECURSION_PROBE_EVIDENCE",
    "CAPABILITY_SCHEMA",
    "MATRIX_SCHEMA",
    "RECURSION_PROBE_SCHEMA",
    "KNOWN_BACKEND_IDS",
    "TRUST_BASELINE_BACKEND_DECISIONS",
    "TEST_ONLY_RECURSION_MATERIAL_ID",
    "TEST_ONLY_CIRCUIT_ID",
    "DEFAULT_RECURSION_PROBE_TIMEOUT_SECONDS",
    "DEFAULT_RECURSION_PROBE_MAX_STEPS",
    "DEFAULT_RECURSION_PROBE_MAX_BYTES",
    "BackendCapabilityError",
    "BackendAvailabilityStatus",
    "AggregationDisposition",
    "RecursionProbeVerdict",
    "CapabilityReasonCode",
    "RecursionProbeMaterial",
    "RecursionProbeArtifact",
    "RecursionProbeResult",
    "RecursiveProbeBackend",
    "HermeticTestOnlyRecursiveBackend",
    "ProofBackendCapability",
    "BackendCapabilityMatrix",
    "BackendCapabilityRegistry",
    "probe_backend_capability",
    "run_bounded_recursion_probe",
    "closed_known_backend_ids",
    "closed_capability_reason_codes",
    "closed_aggregation_dispositions",
)
