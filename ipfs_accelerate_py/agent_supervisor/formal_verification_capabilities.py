"""Import-safe runtime discovery for formal-verification capabilities.

The supervisor uses this module to decide which proof *routes may be
attempted*.  Discovery is intentionally weaker than verification: probes do
not execute a prover, load model weights, generate a circuit proof, or claim
that any proposition has been proved.

All optional integrations are inspected through package metadata, executable
lookup, environment configuration, and bounded filesystem checks.  Provider
modules are never imported, which keeps importing :mod:`agent_supervisor`
safe in minimal installations.
"""

from __future__ import annotations

import importlib.metadata
import importlib.machinery
import json
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


FORMAL_VERIFICATION_CAPABILITY_SCHEMA_VERSION = (
    "ipfs_accelerate_py/agent-supervisor/formal-verification-capabilities@1"
)
FORMAL_VERIFICATION_CAPABILITY_REPORT_VERSION = 1
PROOF_PROVIDER_CAPABILITY_SCHEMA_VERSION = (
    "ipfs_accelerate_py/agent-supervisor/proof-provider-capability@1"
)
DEFAULT_CAPABILITY_CACHE_TTL_SECONDS = 300.0
DEFAULT_CAPABILITY_PROBE_TIMEOUT_SECONDS = 2.0
DEFAULT_CAPABILITY_PROBE_MAX_CHECKS = 96


class CapabilityHealth(str, Enum):
    """Truthful readiness states for one independently probed dependency."""

    SIMULATED = "simulated"
    CONFIGURED = "configured"
    AVAILABLE = "available"
    VERIFIED = "verified"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    DISABLED = "disabled"


class CapabilityDimension(str, Enum):
    """The independent dependency dimensions represented in every report."""

    PROVIDER = "provider"
    EXECUTABLE = "executable"
    PACKAGE = "package"
    MODEL = "model"
    CIRCUIT = "circuit"
    OPTIONAL_DEPENDENCY = "optional_dependency"


class ProofProviderOperation(str, Enum):
    """Operations exposed by version 1 of the optional provider protocol."""

    CAPABILITY = "capability"
    TRANSLATE = "translate"
    PROVE = "prove"
    RECONSTRUCT = "reconstruct"
    VERIFY = "verify"
    ATTEST = "attest"


class ProofProviderIsolation(str, Enum):
    """Execution boundaries a provider advertises support for."""

    IN_PROCESS = "in_process"
    SUBPROCESS = "subprocess"


_PROVIDER_OPERATION_ORDER = tuple(ProofProviderOperation)


@dataclass(frozen=True)
class ProofProviderCapability:
    """Versioned operation-level routing information for one provider.

    This descriptor is deliberately not proof evidence.  In particular,
    ``proof_attempted`` and ``proof_success`` are fixed to false just as they
    are on the broader dependency report above.
    """

    provider_id: str
    provider_version: str
    protocol_versions: tuple[int, ...] = (1,)
    operations: tuple[ProofProviderOperation | str, ...] = (
        ProofProviderOperation.CAPABILITY,
    )
    isolation: tuple[ProofProviderIsolation | str, ...] = (
        ProofProviderIsolation.IN_PROCESS,
    )
    network_access_required: bool = False
    resource_limits_supported: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = PROOF_PROVIDER_CAPABILITY_SCHEMA_VERSION
    proof_attempted: bool = field(default=False, init=False)
    proof_success: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        provider_id = str(self.provider_id).strip()
        provider_version = str(self.provider_version).strip()
        if not provider_id:
            raise ValueError("proof-provider provider_id must not be empty")
        if not provider_version:
            raise ValueError("proof-provider provider_version must not be empty")
        if self.schema_version != PROOF_PROVIDER_CAPABILITY_SCHEMA_VERSION:
            raise ValueError("unsupported proof-provider capability schema")

        versions: list[int] = []
        for raw_version in self.protocol_versions:
            if isinstance(raw_version, bool) or not isinstance(raw_version, int):
                raise ValueError("proof-provider protocol versions must be integers")
            if raw_version < 1:
                raise ValueError("proof-provider protocol versions must be positive")
            if raw_version not in versions:
                versions.append(raw_version)
        if not versions:
            raise ValueError("at least one proof-provider protocol version is required")

        operations = {
            ProofProviderOperation(str(getattr(operation, "value", operation)))
            for operation in self.operations
        }
        if ProofProviderOperation.CAPABILITY not in operations:
            raise ValueError("proof providers must support the capability operation")
        isolation = {
            ProofProviderIsolation(str(getattr(mode, "value", mode)))
            for mode in self.isolation
        }
        if not isolation:
            raise ValueError("at least one proof-provider isolation mode is required")
        if not isinstance(self.network_access_required, bool):
            raise ValueError("network_access_required must be a boolean")
        if not isinstance(self.resource_limits_supported, bool):
            raise ValueError("resource_limits_supported must be a boolean")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("proof-provider capability metadata must be a mapping")
        try:
            metadata = json.loads(
                json.dumps(
                    dict(self.metadata),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                "proof-provider capability metadata must contain strict JSON values"
            ) from exc

        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "provider_version", provider_version)
        object.__setattr__(self, "protocol_versions", tuple(sorted(versions)))
        object.__setattr__(
            self,
            "operations",
            tuple(operation for operation in _PROVIDER_OPERATION_ORDER if operation in operations),
        )
        object.__setattr__(
            self,
            "isolation",
            tuple(mode for mode in ProofProviderIsolation if mode in isolation),
        )
        object.__setattr__(self, "metadata", metadata)

    def supports(
        self,
        operation: ProofProviderOperation | str,
        *,
        protocol_version: int = 1,
        isolation: ProofProviderIsolation | str | None = None,
    ) -> bool:
        """Return whether this descriptor can route the requested call."""

        try:
            normalized_operation = ProofProviderOperation(
                str(getattr(operation, "value", operation))
            )
            normalized_isolation = (
                None
                if isolation is None
                else ProofProviderIsolation(str(getattr(isolation, "value", isolation)))
            )
        except ValueError:
            return False
        return (
            protocol_version in self.protocol_versions
            and normalized_operation in self.operations
            and (
                normalized_isolation is None
                or normalized_isolation in self.isolation
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "protocol_versions": list(self.protocol_versions),
            "operations": [operation.value for operation in self.operations],
            "isolation": [mode.value for mode in self.isolation],
            "network_access_required": self.network_access_required,
            "resource_limits_supported": self.resource_limits_supported,
            "metadata": dict(self.metadata),
            "proof_attempted": False,
            "proof_success": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProofProviderCapability":
        """Validate and decode a provider capability response."""

        if not isinstance(payload, Mapping):
            raise ValueError("proof-provider capability must be an object")
        return cls(
            schema_version=str(payload.get("schema_version", "")),
            provider_id=str(payload.get("provider_id", "")),
            provider_version=str(payload.get("provider_version", "")),
            protocol_versions=tuple(payload.get("protocol_versions") or ()),
            operations=tuple(payload.get("operations") or ()),
            isolation=tuple(payload.get("isolation") or ()),
            network_access_required=payload.get("network_access_required", False),
            resource_limits_supported=payload.get("resource_limits_supported", False),
            metadata=payload.get("metadata", {}),
        )


# A concise compatibility spelling useful to provider implementations.
ProviderCapabilities = ProofProviderCapability


_DIMENSION_ORDER = tuple(CapabilityDimension)


@dataclass(frozen=True)
class CapabilityHealthCheck:
    """Health of one named dependency without any proof-success assertion."""

    dimension: CapabilityDimension | str
    name: str
    status: CapabilityHealth | str
    reason: str
    required: bool = False
    version: str | None = None
    location: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    proof_attempted: bool = field(default=False, init=False)
    proof_success: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dimension",
            CapabilityDimension(str(getattr(self.dimension, "value", self.dimension))),
        )
        object.__setattr__(
            self,
            "status",
            CapabilityHealth(str(getattr(self.status, "value", self.status))),
        )
        name = str(self.name).strip()
        reason = str(self.reason).strip()
        if not name:
            raise ValueError("capability health-check name must not be empty")
        if not reason:
            raise ValueError("capability health-check reason must not be empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def available(self) -> bool:
        """Return true when a dependency is available for use."""

        return self.status in {
            CapabilityHealth.AVAILABLE,
            CapabilityHealth.VERIFIED,
        }

    @property
    def verified(self) -> bool:
        """Return true only for evidence-backed production health."""

        return self.status is CapabilityHealth.VERIFIED

    @property
    def discovered(self) -> bool:
        """Return true when the dependency exists, including degraded forms."""

        return self.status in {
            CapabilityHealth.SIMULATED,
            CapabilityHealth.CONFIGURED,
            CapabilityHealth.AVAILABLE,
            CapabilityHealth.VERIFIED,
            CapabilityHealth.DEGRADED,
            CapabilityHealth.DISABLED,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "name": self.name,
            "status": self.status.value,
            "reason": self.reason,
            "required": self.required,
            "version": self.version,
            "location": self.location,
            "metadata": dict(self.metadata),
            "proof_attempted": False,
            "proof_success": False,
        }


@dataclass(frozen=True)
class FormalVerificationProviderCapability:
    """Capability matrix row for one formal-logic provider family."""

    provider_id: str
    display_name: str
    status: CapabilityHealth | str
    checks: tuple[CapabilityHealthCheck, ...]
    reason: str
    proof_attempted: bool = field(default=False, init=False)
    proof_success: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            CapabilityHealth(str(getattr(self.status, "value", self.status))),
        )
        provider_id = str(self.provider_id).strip()
        display_name = str(self.display_name).strip()
        reason = str(self.reason).strip()
        if not provider_id or not display_name or not reason:
            raise ValueError("provider_id, display_name, and reason must not be empty")
        if not self.checks:
            raise ValueError(f"provider {provider_id!r} must contain health checks")
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "checks", tuple(self.checks))

        provider_checks = self.health_for(CapabilityDimension.PROVIDER)
        if not provider_checks:
            raise ValueError(f"provider {provider_id!r} has no provider health check")

    def health_for(
        self, dimension: CapabilityDimension | str
    ) -> tuple[CapabilityHealthCheck, ...]:
        normalized = CapabilityDimension(str(getattr(dimension, "value", dimension)))
        return tuple(check for check in self.checks if check.dimension is normalized)

    @property
    def provider_health(self) -> tuple[CapabilityHealthCheck, ...]:
        return self.health_for(CapabilityDimension.PROVIDER)

    @property
    def executable_health(self) -> tuple[CapabilityHealthCheck, ...]:
        return self.health_for(CapabilityDimension.EXECUTABLE)

    @property
    def package_health(self) -> tuple[CapabilityHealthCheck, ...]:
        return self.health_for(CapabilityDimension.PACKAGE)

    @property
    def model_health(self) -> tuple[CapabilityHealthCheck, ...]:
        return self.health_for(CapabilityDimension.MODEL)

    @property
    def circuit_health(self) -> tuple[CapabilityHealthCheck, ...]:
        return self.health_for(CapabilityDimension.CIRCUIT)

    @property
    def optional_dependency_health(self) -> tuple[CapabilityHealthCheck, ...]:
        return self.health_for(CapabilityDimension.OPTIONAL_DEPENDENCY)

    @property
    def available(self) -> bool:
        return self.status in {
            CapabilityHealth.AVAILABLE,
            CapabilityHealth.VERIFIED,
        }

    @property
    def production_ready(self) -> bool:
        """Return whether bounded verification evidence promoted this provider."""

        return self.status is CapabilityHealth.VERIFIED

    def to_dict(self) -> dict[str, Any]:
        health = {
            dimension.value: [
                check.to_dict() for check in self.health_for(dimension)
            ]
            for dimension in _DIMENSION_ORDER
        }
        return {
            "provider_id": self.provider_id,
            "display_name": self.display_name,
            "status": self.status.value,
            "reason": self.reason,
            "health": health,
            "proof_attempted": False,
            "proof_success": False,
        }


@dataclass(frozen=True)
class FormalVerificationCapabilityReport:
    """Versioned, immutable snapshot used for routing proof work."""

    providers: tuple[FormalVerificationProviderCapability, ...]
    generated_at: str
    duration_seconds: float
    probe_count: int
    bounded: bool = True
    cache_ttl_seconds: float = DEFAULT_CAPABILITY_CACHE_TTL_SECONDS
    schema_version: str = FORMAL_VERIFICATION_CAPABILITY_SCHEMA_VERSION
    report_version: int = FORMAL_VERIFICATION_CAPABILITY_REPORT_VERSION
    proof_attempted: bool = field(default=False, init=False)
    proof_success: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        providers = tuple(self.providers)
        ids = [provider.provider_id for provider in providers]
        if len(ids) != len(set(ids)):
            raise ValueError("formal-verification provider ids must be unique")
        if self.report_version < 1:
            raise ValueError("report_version must be positive")
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative")
        if self.probe_count < 0:
            raise ValueError("probe_count must be non-negative")
        object.__setattr__(self, "providers", providers)

    @property
    def capabilities(self) -> Mapping[str, FormalVerificationProviderCapability]:
        return {provider.provider_id: provider for provider in self.providers}

    @property
    def overall_status(self) -> CapabilityHealth:
        if not self.providers:
            return CapabilityHealth.UNAVAILABLE
        statuses = tuple(provider.status for provider in self.providers)
        if all(
            status in {CapabilityHealth.AVAILABLE, CapabilityHealth.VERIFIED}
            for status in statuses
        ):
            return CapabilityHealth.AVAILABLE
        if all(status is CapabilityHealth.UNAVAILABLE for status in statuses):
            return CapabilityHealth.UNAVAILABLE
        return CapabilityHealth.DEGRADED

    def provider(self, provider_id: str) -> FormalVerificationProviderCapability:
        try:
            return self.capabilities[str(provider_id)]
        except KeyError as exc:
            raise KeyError(f"unknown formal-verification provider: {provider_id}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "report_version": self.report_version,
            "generated_at": self.generated_at,
            "duration_seconds": self.duration_seconds,
            "probe_count": self.probe_count,
            "bounded": self.bounded,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "overall_status": self.overall_status.value,
            "proof_attempted": False,
            "proof_success": False,
            "providers": {
                provider.provider_id: provider.to_dict()
                for provider in self.providers
            },
        }


@dataclass(frozen=True)
class FormalVerificationProbeConfig:
    """Resource and configuration limits for one capability snapshot."""

    cache_ttl_seconds: float = DEFAULT_CAPABILITY_CACHE_TTL_SECONDS
    timeout_seconds: float = DEFAULT_CAPABILITY_PROBE_TIMEOUT_SECONDS
    max_checks: int = DEFAULT_CAPABILITY_PROBE_MAX_CHECKS
    spacy_model: str = "en_core_web_sm"
    leanstral_model_path: str | None = None
    groth16_artifacts_path: str | None = None
    provekit_artifacts_path: str | None = None

    def __post_init__(self) -> None:
        if self.cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must be non-negative")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_checks < 1:
            raise ValueError("max_checks must be positive")
        if not str(self.spacy_model).strip():
            raise ValueError("spacy_model must not be empty")


PackageFinder = Callable[[str], Any]
ExecutableFinder = Callable[[str], str | None]
DistributionVersionFinder = Callable[[str], str]


def _find_spec_without_import(module: str) -> Any:
    """Resolve a dotted module with ``PathFinder`` without importing parents."""

    path: Sequence[str] | None = None
    parts = str(module).split(".")
    if not parts or any(not part for part in parts):
        return None
    spec: Any = None
    for index in range(len(parts)):
        qualified_name = ".".join(parts[: index + 1])
        spec = importlib.machinery.PathFinder.find_spec(qualified_name, path)
        if spec is None:
            return None
        if index < len(parts) - 1:
            locations = spec.submodule_search_locations
            if locations is None:
                return None
            path = tuple(str(location) for location in locations)
    return spec


class FormalVerificationCapabilityProbe:
    """Bounded and cacheable formal-verification discovery service.

    ``find_spec`` and ``which`` are injectable so callers can test deployment
    matrices without installing heavyweight optional dependencies.
    """

    def __init__(
        self,
        config: FormalVerificationProbeConfig | None = None,
        *,
        find_spec: PackageFinder | None = None,
        which: ExecutableFinder | None = None,
        distribution_version: DistributionVersionFinder | None = None,
        environ: Mapping[str, str] | None = None,
        monotonic: Callable[[], float] | None = None,
        wall_clock: Callable[[], float] | None = None,
    ) -> None:
        self.config = config or FormalVerificationProbeConfig()
        self._find_spec = find_spec or _find_spec_without_import
        self._which = which or shutil.which
        self._distribution_version = (
            distribution_version or importlib.metadata.version
        )
        self._environ = environ if environ is not None else os.environ
        self._monotonic = monotonic or time.monotonic
        self._wall_clock = wall_clock or time.time
        self._cache_lock = threading.Lock()
        self._cached: tuple[float, FormalVerificationCapabilityReport] | None = None
        self._started = 0.0
        self._deadline = 0.0
        self._probe_count = 0

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cached = None

    def probe(
        self, *, force_refresh: bool = False
    ) -> FormalVerificationCapabilityReport:
        """Return a cached snapshot or perform one bounded metadata probe."""

        with self._cache_lock:
            now = self._monotonic()
            if (
                not force_refresh
                and self._cached is not None
                and self.config.cache_ttl_seconds > 0
                and now - self._cached[0] < self.config.cache_ttl_seconds
            ):
                return self._cached[1]

            self._started = now
            self._deadline = now + self.config.timeout_seconds
            self._probe_count = 0
            providers = (
                self._probe_hammer(),
                self._probe_tdfol(),
                self._probe_external_provers(),
                self._probe_lean(),
                self._probe_leanstral(),
                self._probe_frame_logic(),
                self._probe_knowledge_graphs(),
                self._probe_zkp_backends(),
            )
            finished = self._monotonic()
            report = FormalVerificationCapabilityReport(
                providers=providers,
                generated_at=datetime.fromtimestamp(
                    self._wall_clock(), tz=timezone.utc
                )
                .isoformat()
                .replace("+00:00", "Z"),
                duration_seconds=max(0.0, finished - self._started),
                probe_count=self._probe_count,
                cache_ttl_seconds=self.config.cache_ttl_seconds,
            )
            self._cached = (finished, report)
            return report

    # -- primitive checks -------------------------------------------------

    def _budget_reason(self) -> str | None:
        if self._probe_count >= self.config.max_checks:
            return (
                f"probe check limit ({self.config.max_checks}) exhausted; "
                "health was not inspected"
            )
        if self._monotonic() >= self._deadline:
            return (
                f"probe time budget ({self.config.timeout_seconds:g}s) exhausted; "
                "health was not inspected"
            )
        self._probe_count += 1
        return None

    def _not_inspected(
        self,
        dimension: CapabilityDimension,
        name: str,
        reason: str,
        *,
        required: bool = False,
    ) -> CapabilityHealthCheck:
        return CapabilityHealthCheck(
            dimension=dimension,
            name=name,
            status=CapabilityHealth.UNAVAILABLE,
            reason=reason,
            required=required,
            metadata={"probe_limited": True},
        )

    def _bounded_call(
        self,
        function: Callable[..., Any],
        *args: Any,
    ) -> tuple[bool, Any, BaseException | None]:
        """Run a metadata operation without letting it exceed the report budget.

        Python cannot safely terminate an arbitrary import hook.  A timed-out
        hook therefore finishes, if ever, on a daemon thread and cannot delay
        process shutdown or mutate the capability report.
        """

        remaining = self._deadline - self._monotonic()
        if remaining <= 0:
            return False, None, TimeoutError("capability probe time budget exhausted")

        outcome: list[tuple[bool, Any, BaseException | None]] = []

        def invoke() -> None:
            try:
                outcome.append((True, function(*args), None))
            except BaseException as exc:  # keep broken third-party hooks isolated
                outcome.append((True, None, exc))

        worker = threading.Thread(
            target=invoke,
            name="formal-capability-metadata-probe",
            daemon=True,
        )
        worker.start()
        worker.join(remaining)
        if worker.is_alive():
            return False, None, TimeoutError(
                f"metadata operation exceeded {remaining:g}s remaining probe budget"
            )
        if not outcome:
            return False, None, RuntimeError("metadata operation returned no outcome")
        return outcome[0]

    def _package(
        self,
        module: str,
        *,
        name: str | None = None,
        distribution: str | None = None,
        required: bool = False,
        dimension: CapabilityDimension = CapabilityDimension.PACKAGE,
        missing_reason: str | None = None,
    ) -> CapabilityHealthCheck:
        label = name or module
        limited = self._budget_reason()
        if limited:
            return self._not_inspected(
                dimension, label, limited, required=required
            )
        completed, spec, error = self._bounded_call(self._find_spec, module)
        if not completed:
            return CapabilityHealthCheck(
                dimension=dimension,
                name=label,
                status=CapabilityHealth.UNAVAILABLE,
                reason=f"Python package discovery for {module!r} timed out safely: {error}",
                required=required,
                metadata={"module": module, "probe_limited": True},
            )
        if error is not None:
            return CapabilityHealthCheck(
                dimension=dimension,
                name=label,
                status=CapabilityHealth.UNAVAILABLE,
                reason=(
                    f"Python package discovery for {module!r} failed safely: "
                    f"{type(error).__name__}: {error}"
                ),
                required=required,
                metadata={"module": module},
            )
        if spec is None:
            return CapabilityHealthCheck(
                dimension=dimension,
                name=label,
                status=CapabilityHealth.UNAVAILABLE,
                reason=missing_reason or f"Python package {module!r} is not installed",
                required=required,
                metadata={"module": module},
            )

        version: str | None = None
        if distribution:
            version_completed, raw_version, version_error = self._bounded_call(
                self._distribution_version, distribution
            )
            if version_completed and version_error is None:
                version = str(raw_version)
        location = getattr(spec, "origin", None)
        return CapabilityHealthCheck(
            dimension=dimension,
            name=label,
            status=CapabilityHealth.AVAILABLE,
            reason=(
                f"Python module {module!r} is discoverable; it was not imported "
                "or executed by this probe"
            ),
            required=required,
            version=version,
            location=str(location) if location else None,
            metadata={"module": module},
        )

    def _executable(
        self,
        name: str,
        candidates: Sequence[str],
        *,
        required: bool = False,
        explicit_env: Sequence[str] = (),
        missing_reason: str | None = None,
    ) -> CapabilityHealthCheck:
        limited = self._budget_reason()
        if limited:
            return self._not_inspected(
                CapabilityDimension.EXECUTABLE, name, limited, required=required
            )

        for env_name in explicit_env:
            configured = str(self._environ.get(env_name, "")).strip()
            if not configured:
                continue
            path = Path(configured).expanduser()
            completed, usable, error = self._bounded_call(
                lambda: path.is_file() and os.access(path, os.X_OK)
            )
            if not completed:
                return CapabilityHealthCheck(
                    dimension=CapabilityDimension.EXECUTABLE,
                    name=name,
                    status=CapabilityHealth.UNAVAILABLE,
                    reason=f"configured executable inspection timed out safely: {error}",
                    required=required,
                    location=str(path),
                    metadata={
                        "environment_variable": env_name,
                        "probe_limited": True,
                    },
                )
            if error is not None:
                usable = False
            if usable:
                return CapabilityHealthCheck(
                    dimension=CapabilityDimension.EXECUTABLE,
                    name=name,
                    status=CapabilityHealth.AVAILABLE,
                    reason=(
                        f"executable configured by {env_name}; it was not invoked"
                    ),
                    required=required,
                    location=str(path),
                    metadata={"environment_variable": env_name},
                )
            return CapabilityHealthCheck(
                dimension=CapabilityDimension.EXECUTABLE,
                name=name,
                status=CapabilityHealth.UNAVAILABLE,
                reason=(
                    f"{env_name} points to {configured!r}, which is not an "
                    "executable file"
                ),
                required=required,
                location=configured,
                metadata={"environment_variable": env_name},
            )

        for candidate in candidates:
            completed, found, error = self._bounded_call(self._which, candidate)
            if not completed:
                return CapabilityHealthCheck(
                    dimension=CapabilityDimension.EXECUTABLE,
                    name=name,
                    status=CapabilityHealth.UNAVAILABLE,
                    reason=f"executable discovery timed out safely: {error}",
                    required=required,
                    metadata={"probe_limited": True},
                )
            if error is not None:
                return CapabilityHealthCheck(
                    dimension=CapabilityDimension.EXECUTABLE,
                    name=name,
                    status=CapabilityHealth.UNAVAILABLE,
                    reason=(
                        f"executable discovery failed safely: "
                        f"{type(error).__name__}: {error}"
                    ),
                    required=required,
                )
            if found:
                return CapabilityHealthCheck(
                    dimension=CapabilityDimension.EXECUTABLE,
                    name=name,
                    status=CapabilityHealth.AVAILABLE,
                    reason=(
                        f"executable {candidate!r} is on PATH; it was not invoked"
                    ),
                    required=required,
                    location=str(found),
                    metadata={"candidate": candidate},
                )

        return CapabilityHealthCheck(
            dimension=CapabilityDimension.EXECUTABLE,
            name=name,
            status=CapabilityHealth.UNAVAILABLE,
            reason=missing_reason
            or f"no prover executable found on PATH ({', '.join(candidates)})",
            required=required,
            metadata={"candidates": list(candidates)},
        )

    def _configured_artifact(
        self,
        dimension: CapabilityDimension,
        name: str,
        configured: str | None,
        *,
        env_names: Sequence[str],
        required_files: Sequence[str] = (),
        required_pattern_groups: Sequence[Sequence[str]] = (),
        missing_reason: str,
        required: bool = False,
    ) -> CapabilityHealthCheck:
        limited = self._budget_reason()
        if limited:
            return self._not_inspected(dimension, name, limited, required=required)

        source = "probe configuration"
        raw = str(configured or "").strip()
        if not raw:
            for env_name in env_names:
                raw = str(self._environ.get(env_name, "")).strip()
                if raw:
                    source = env_name
                    break
        if not raw:
            return CapabilityHealthCheck(
                dimension=dimension,
                name=name,
                status=CapabilityHealth.UNAVAILABLE,
                reason=missing_reason,
                required=required,
                metadata={"configuration_keys": list(env_names)},
            )

        path = Path(raw).expanduser()

        def inspect_artifact() -> tuple[bool, list[str], list[list[str]]]:
            exists = path.exists()
            missing_exact: list[str] = []
            missing_groups: list[list[str]] = []
            if not exists:
                return False, missing_exact, missing_groups
            if path.is_file():
                if required_files or required_pattern_groups:
                    filename = path.name
                    missing_exact = [
                        required_name
                        for required_name in required_files
                        if filename != required_name
                    ]
                    missing_groups = [
                        list(patterns)
                        for patterns in required_pattern_groups
                        if not any(path.match(pattern) for pattern in patterns)
                    ]
                return True, missing_exact, missing_groups
            missing_exact = [
                filename
                for filename in required_files
                if not (path / filename).is_file()
            ]
            for patterns in required_pattern_groups:
                if not any(
                    any(candidate.is_file() for candidate in path.glob(pattern))
                    for pattern in patterns
                ):
                    missing_groups.append(list(patterns))
            return True, missing_exact, missing_groups

        completed, inspection, error = self._bounded_call(inspect_artifact)
        if not completed:
            return CapabilityHealthCheck(
                dimension=dimension,
                name=name,
                status=CapabilityHealth.UNAVAILABLE,
                reason=f"configured artifact inspection timed out safely: {error}",
                required=required,
                location=str(path),
                metadata={"probe_limited": True},
            )
        if error is not None:
            return CapabilityHealthCheck(
                dimension=dimension,
                name=name,
                status=CapabilityHealth.UNAVAILABLE,
                reason=(
                    f"could not inspect configured artifact {path}: "
                    f"{type(error).__name__}: {error}"
                ),
                required=required,
                location=str(path),
            )
        exists, missing, missing_groups = inspection
        if not exists:
            return CapabilityHealthCheck(
                dimension=dimension,
                name=name,
                status=CapabilityHealth.UNAVAILABLE,
                reason=f"{source} points to missing artifact path {path}",
                required=required,
                location=str(path),
            )
        if missing or missing_groups:
            missing_descriptions = list(missing)
            missing_descriptions.extend(
                "one of [" + ", ".join(patterns) + "]"
                for patterns in missing_groups
            )
            return CapabilityHealthCheck(
                dimension=dimension,
                name=name,
                status=CapabilityHealth.DEGRADED,
                reason=(
                    f"artifact directory exists but required files are missing: "
                    f"{', '.join(missing_descriptions)}"
                ),
                required=required,
                location=str(path),
                metadata={
                    "missing_files": missing,
                    "missing_pattern_groups": missing_groups,
                },
            )
        return CapabilityHealthCheck(
            dimension=dimension,
            name=name,
            status=CapabilityHealth.CONFIGURED,
            reason=(
                f"artifact path configured by {source} exists; contents were not "
                "loaded or cryptographically verified"
            ),
            required=required,
            location=str(path),
        )

    @staticmethod
    def _provider_check(
        name: str, status: CapabilityHealth, reason: str
    ) -> CapabilityHealthCheck:
        return CapabilityHealthCheck(
            dimension=CapabilityDimension.PROVIDER,
            name=name,
            status=status,
            reason=reason,
            required=True,
        )

    @staticmethod
    def _component(
        provider_id: str,
        display_name: str,
        status: CapabilityHealth,
        reason: str,
        checks: Sequence[CapabilityHealthCheck],
    ) -> FormalVerificationProviderCapability:
        return FormalVerificationProviderCapability(
            provider_id=provider_id,
            display_name=display_name,
            status=status,
            reason=reason,
            checks=tuple(checks),
        )

    # -- provider families ------------------------------------------------

    def _probe_hammer(self) -> FormalVerificationProviderCapability:
        package = self._package(
            "ipfs_datasets_py.logic.hammers",
            name="hammer",
            required=True,
            missing_reason="Hammer Python provider package is not importable",
        )
        eprover = self._executable("E prover", ("eprover", "eprover-ho"))
        vampire = self._executable("Vampire", ("vampire",))
        z3 = self._executable("Z3 CLI", ("z3",))
        cvc5 = self._executable("CVC5 CLI", ("cvc5",))
        model = self._package(
            "ipfs_datasets_py.logic.hammers.learned_selector",
            name="built-in graph selector artifact",
            dimension=CapabilityDimension.MODEL,
            missing_reason="Hammer graph-selector model definition is unavailable",
        )
        engines = (eprover, vampire, z3, cvc5)
        if not package.available:
            status = CapabilityHealth.UNAVAILABLE
            reason = package.reason
        elif any(check.available for check in engines):
            status = CapabilityHealth.AVAILABLE
            reason = "Hammer provider and at least one external proving engine are discoverable"
        else:
            status = CapabilityHealth.DEGRADED
            reason = (
                "Hammer translation and reconstruction are discoverable, but no "
                "ATP/SMT executable is available"
            )
        return self._component(
            "hammer",
            "Hammer",
            status,
            reason,
            (
                self._provider_check("hammer", status, reason),
                package,
                model,
                *engines,
            ),
        )

    def _probe_tdfol(self) -> FormalVerificationProviderCapability:
        package = self._package(
            "ipfs_datasets_py.logic.TDFOL",
            name="tdfol",
            required=True,
            missing_reason="TDFOL Python provider package is not importable",
        )
        spacy = self._package(
            "spacy",
            name="spaCy",
            distribution="spacy",
            dimension=CapabilityDimension.OPTIONAL_DEPENDENCY,
            missing_reason=(
                "spaCy is not installed; TDFOL natural-language preprocessing "
                "is limited to non-spaCy paths"
            ),
        )
        spacy_model = self._package(
            self.config.spacy_model,
            name=f"spaCy model {self.config.spacy_model}",
            dimension=CapabilityDimension.MODEL,
            missing_reason=(
                f"spaCy model weights {self.config.spacy_model!r} are not installed; "
                "model-assisted TDFOL parsing is unavailable"
            ),
        )
        if not package.available:
            status = CapabilityHealth.UNAVAILABLE
            reason = package.reason
        elif not spacy.available or not spacy_model.available:
            status = CapabilityHealth.DEGRADED
            reason = (
                "TDFOL symbolic logic is available, but optional natural-language "
                "model support is incomplete"
            )
        else:
            status = CapabilityHealth.AVAILABLE
            reason = "TDFOL core and optional natural-language dependencies are discoverable"
        return self._component(
            "tdfol",
            "Temporal Deontic First-Order Logic",
            status,
            reason,
            (
                self._provider_check("tdfol", status, reason),
                package,
                spacy_model,
                spacy,
            ),
        )

    def _probe_external_provers(self) -> FormalVerificationProviderCapability:
        provider_package = self._package(
            "ipfs_datasets_py.logic.external_provers",
            name="external prover bridge",
            required=True,
            missing_reason="external prover Python bridge package is not importable",
        )
        z3_binding = self._package(
            "z3",
            name="Z3 Python binding",
            distribution="z3-solver",
            missing_reason=(
                "Z3 Python bindings are not installed; the Z3 bridge is unavailable"
            ),
        )
        cvc5_binding = self._package(
            "cvc5",
            name="CVC5 Python binding",
            distribution="cvc5",
            missing_reason=(
                "CVC5 Python bindings are not installed; the CVC5 bridge is unavailable"
            ),
        )
        symbolicai = self._package(
            "symai",
            name="SymbolicAI",
            distribution="symbolicai",
            dimension=CapabilityDimension.OPTIONAL_DEPENDENCY,
            missing_reason=(
                "SymbolicAI is not installed; neural external-prover routing is unavailable"
            ),
        )
        executables = (
            self._executable("Z3 CLI", ("z3",)),
            self._executable("CVC5 CLI", ("cvc5",)),
            self._executable("E prover", ("eprover", "eprover-ho")),
            self._executable("Vampire", ("vampire",)),
            self._executable("Coq", ("coqtop",)),
        )
        engines_available = (
            z3_binding.available
            or cvc5_binding.available
            or any(check.available for check in executables)
        )
        if not provider_package.available:
            status = CapabilityHealth.UNAVAILABLE
            reason = provider_package.reason
        elif engines_available:
            status = CapabilityHealth.AVAILABLE
            reason = "external prover bridge and at least one solver are discoverable"
        else:
            status = CapabilityHealth.DEGRADED
            reason = (
                "external prover routing code is discoverable, but Python bindings "
                "and prover executables are unavailable"
            )
        return self._component(
            "external_provers",
            "External Provers",
            status,
            reason,
            (
                self._provider_check("external_provers", status, reason),
                provider_package,
                z3_binding,
                cvc5_binding,
                symbolicai,
                *executables,
            ),
        )

    def _probe_lean(self) -> FormalVerificationProviderCapability:
        package = self._package(
            "ipfs_datasets_py.logic.external_provers.interactive.lean_prover_bridge",
            name="Lean bridge",
            required=True,
            missing_reason="Lean Python bridge package is not importable",
        )
        executable = self._executable(
            "Lean 4",
            ("lean", "lake"),
            required=True,
            explicit_env=("LEAN_BINARY", "IPFS_DATASETS_LEAN_BINARY"),
            missing_reason=(
                "Lean prover executable is not installed or configured; no Lean "
                "kernel check can run"
            ),
        )
        if not package.available:
            status = CapabilityHealth.UNAVAILABLE
            reason = package.reason
        elif not executable.available:
            status = CapabilityHealth.UNAVAILABLE
            reason = executable.reason
        else:
            status = CapabilityHealth.AVAILABLE
            reason = "Lean bridge and Lean 4 executable are discoverable"
        return self._component(
            "lean",
            "Lean 4",
            status,
            reason,
            (
                self._provider_check("lean", status, reason),
                package,
                executable,
            ),
        )

    def _probe_leanstral(self) -> FormalVerificationProviderCapability:
        adapter = self._package(
            "ipfs_accelerate_py.agent_supervisor.leanstral_proof_provider",
            name="Leanstral proof-provider adapter",
            dimension=CapabilityDimension.PROVIDER,
            required=True,
            missing_reason=(
                "the capability-isolated Leanstral proof-provider adapter is unavailable"
            ),
        )
        model_service = self._package(
            "ipfs_accelerate_py.llm_router",
            name="llm_router model service",
            dimension=CapabilityDimension.PROVIDER,
            required=True,
            missing_reason=(
                "llm_router is unavailable; no isolated Leanstral model route can run"
            ),
        )
        package = self._package(
            "ipfs_datasets_py.logic.modal.leanstral",
            name="Leanstral integration",
            missing_reason=(
                "Leanstral legal-modal integration is unavailable; the isolated "
                "provider remains importable but legal-modal prompt construction "
                "is degraded"
            ),
        )
        codec = self._package(
            "ipfs_datasets_py.logic.modal.codec",
            name="legal modal codec",
            missing_reason=(
                "legal modal codec is unavailable; Leanstral drafts cannot be "
                "bound to codec-produced canonical source"
            ),
        )
        spacy = self._package(
            "spacy",
            name="spaCy",
            distribution="spacy",
            dimension=CapabilityDimension.OPTIONAL_DEPENDENCY,
            missing_reason=(
                "spaCy is not installed; Leanstral legal-language preprocessing "
                "is degraded"
            ),
        )
        spacy_model = self._package(
            self.config.spacy_model,
            name=f"spaCy model {self.config.spacy_model}",
            dimension=CapabilityDimension.MODEL,
            missing_reason=(
                f"spaCy model weights {self.config.spacy_model!r} are unavailable; "
                "Leanstral legal-language preprocessing is degraded"
            ),
        )
        transformers = self._package(
            "transformers",
            name="Transformers",
            distribution="transformers",
            dimension=CapabilityDimension.OPTIONAL_DEPENDENCY,
            missing_reason=(
                "Transformers is not installed; local Leanstral inference is unavailable"
            ),
        )
        torch = self._package(
            "torch",
            name="PyTorch",
            distribution="torch",
            dimension=CapabilityDimension.OPTIONAL_DEPENDENCY,
            missing_reason=(
                "PyTorch is not installed; local Leanstral model execution is unavailable"
            ),
        )
        model = self._configured_artifact(
            CapabilityDimension.MODEL,
            "Leanstral model weights",
            self.config.leanstral_model_path,
            env_names=(
                "IPFS_DATASETS_LEANSTRAL_MODEL_PATH",
                "LEANSTRAL_MODEL_PATH",
            ),
            missing_reason=(
                "Leanstral model weights are not configured; set "
                "IPFS_DATASETS_LEANSTRAL_MODEL_PATH for local inference or inject "
                "an explicitly managed provider callback"
            ),
            required_files=("config.json",),
            required_pattern_groups=(
                ("*.safetensors", "pytorch_model*.bin", "*.gguf", "*.pt"),
            ),
        )
        lean = self._executable(
            "Lean 4",
            ("lean", "lake"),
            explicit_env=("LEAN_BINARY", "IPFS_DATASETS_LEAN_BINARY"),
            missing_reason=(
                "Lean executable is unavailable; Leanstral proposals cannot receive "
                "a local kernel check"
            ),
        )
        inference_ready = (
            adapter.available
            and model_service.available
            and package.available
            and codec.available
            and spacy.available
            and spacy_model.available
            and model.available
            and transformers.available
            and torch.available
        )
        if inference_ready and lean.available:
            status = CapabilityHealth.AVAILABLE
            reason = (
                "isolated Leanstral inference dependencies and the separate Lean "
                "kernel-checking toolchain are discoverable"
            )
        else:
            status = CapabilityHealth.DEGRADED
            reason = (
                "the Leanstral provider is available only as a degraded capability; "
                "one or more model-service, legal-modal, spaCy, codec, model, or "
                "separate kernel dependencies are unavailable"
            )
        return self._component(
            "leanstral",
            "Leanstral",
            status,
            reason,
            (
                self._provider_check("leanstral", status, reason),
                adapter,
                model_service,
                package,
                codec,
                model,
                lean,
                spacy_model,
                spacy,
                transformers,
                torch,
            ),
        )

    def _probe_frame_logic(self) -> FormalVerificationProviderCapability:
        package = self._package(
            "ipfs_datasets_py.logic.flogic",
            name="frame logic",
            required=True,
            missing_reason="frame-logic Python package is not importable",
        )
        ergo = self._executable(
            "ErgoAI",
            ("runErgo.sh", "runergo"),
            explicit_env=("ERGOAI_BINARY",),
            missing_reason=(
                "ErgoAI prover executable is unavailable; frame logic is limited "
                "to its in-memory structural/simulation mode"
            ),
        )
        if not package.available:
            status = CapabilityHealth.UNAVAILABLE
            reason = package.reason
        elif not ergo.available:
            status = CapabilityHealth.DEGRADED
            reason = (
                "frame-logic structures are available, but theorem queries cannot "
                "use ErgoAI"
            )
        else:
            status = CapabilityHealth.AVAILABLE
            reason = "frame-logic package and ErgoAI executable are discoverable"
        return self._component(
            "frame_logic",
            "Frame Logic (ErgoAI)",
            status,
            reason,
            (
                self._provider_check("frame_logic", status, reason),
                package,
                ergo,
            ),
        )

    def _probe_knowledge_graphs(self) -> FormalVerificationProviderCapability:
        package = self._package(
            "ipfs_datasets_py.knowledge_graphs",
            name="knowledge graphs",
            required=True,
            missing_reason="knowledge-graph Python package is not importable",
        )
        spacy = self._package(
            "spacy",
            name="spaCy",
            distribution="spacy",
            dimension=CapabilityDimension.OPTIONAL_DEPENDENCY,
            missing_reason=(
                "spaCy is not installed; knowledge-graph extraction uses rule-based "
                "fallbacks only"
            ),
        )
        transformers = self._package(
            "transformers",
            name="Transformers",
            distribution="transformers",
            dimension=CapabilityDimension.OPTIONAL_DEPENDENCY,
            missing_reason=(
                "Transformers is not installed; neural entity/relation extraction "
                "is unavailable"
            ),
        )
        networkx = self._package(
            "networkx",
            name="NetworkX",
            distribution="networkx",
            dimension=CapabilityDimension.OPTIONAL_DEPENDENCY,
            missing_reason=(
                "NetworkX is not installed; graph algorithms requiring it are unavailable"
            ),
        )
        spacy_model = self._package(
            self.config.spacy_model,
            name=f"spaCy model {self.config.spacy_model}",
            dimension=CapabilityDimension.MODEL,
            missing_reason=(
                f"spaCy model weights {self.config.spacy_model!r} are not installed; "
                "model-assisted entity and frame extraction is unavailable"
            ),
        )
        if not package.available:
            status = CapabilityHealth.UNAVAILABLE
            reason = package.reason
        elif not spacy.available or not spacy_model.available:
            status = CapabilityHealth.DEGRADED
            reason = (
                "knowledge-graph core is discoverable, but optional NLP/model "
                "enhancements are incomplete"
            )
        else:
            status = CapabilityHealth.AVAILABLE
            reason = "knowledge-graph core and spaCy model support are discoverable"
        return self._component(
            "knowledge_graphs",
            "Knowledge Graphs",
            status,
            reason,
            (
                self._provider_check("knowledge_graphs", status, reason),
                package,
                spacy_model,
                spacy,
                transformers,
                networkx,
            ),
        )

    def _probe_zkp_backends(self) -> FormalVerificationProviderCapability:
        package = self._package(
            "ipfs_datasets_py.logic.zkp.backends",
            name="ZKP backend registry",
            required=True,
            missing_reason="ZKP backend registry package is not importable",
        )
        circuits_package = self._package(
            "ipfs_datasets_py.logic.zkp.circuits",
            name="logic circuit definitions",
            dimension=CapabilityDimension.CIRCUIT,
            required=True,
            missing_reason="versioned logic circuit definitions are unavailable",
        )
        simulated = CapabilityHealthCheck(
            dimension=CapabilityDimension.PROVIDER,
            name="simulated ZKP backend",
            status=CapabilityHealth.DEGRADED
            if package.available
            else CapabilityHealth.UNAVAILABLE,
            reason=(
                "simulated backend is discoverable but is educational and "
                "non-cryptographic; it cannot satisfy proof assurance"
                if package.available
                else "simulated ZKP backend package is not discoverable"
            ),
            required=False,
            metadata={
                "cryptographic": False,
                "backend_health": CapabilityHealth.SIMULATED.value,
            },
        )
        groth16 = self._executable(
            "Groth16 Rust backend",
            ("groth16",),
            explicit_env=("IPFS_DATASETS_GROTH16_BINARY", "GROTH16_BINARY"),
            missing_reason=(
                "Groth16 Rust prover executable is not built or configured"
            ),
        )
        provekit = self._executable(
            "ProveKit CLI",
            ("provekit-cli", "provekit"),
            explicit_env=("IPFS_DATASETS_PROVEKIT_BINARY", "PROVEKIT_CLI"),
            missing_reason="ProveKit prover executable is not installed or configured",
        )
        groth16_artifacts = self._configured_artifact(
            CapabilityDimension.CIRCUIT,
            "Groth16 setup artifacts",
            self.config.groth16_artifacts_path,
            env_names=(
                "IPFS_DATASETS_GROTH16_ARTIFACTS_DIR",
                "GROTH16_ARTIFACTS_DIR",
            ),
            required_files=("proving_key.bin", "verifying_key.bin"),
            missing_reason=(
                "Groth16 proving/verifying circuit artifacts are not configured"
            ),
        )
        provekit_artifacts = self._configured_artifact(
            CapabilityDimension.CIRCUIT,
            "ProveKit circuit artifacts",
            self.config.provekit_artifacts_path,
            env_names=(
                "IPFS_DATASETS_PROVEKIT_ARTIFACTS_DIR",
                "PROVEKIT_ARTIFACTS_DIR",
            ),
            missing_reason=(
                "ProveKit prover/verifier circuit artifacts are not configured"
            ),
            required_pattern_groups=(
                ("*.pkp", "*prover*.key", "prover_key*"),
                ("*.pkv", "*verifier*.key", "verifier_key*"),
            ),
        )
        py_ecc = self._package(
            "py_ecc",
            name="py-ecc",
            distribution="py-ecc",
            dimension=CapabilityDimension.OPTIONAL_DEPENDENCY,
            missing_reason=(
                "py-ecc Python bindings are not installed; Python-side BN254 "
                "operations are unavailable"
            ),
        )
        configured_states = {
            CapabilityHealth.CONFIGURED,
            CapabilityHealth.AVAILABLE,
            CapabilityHealth.VERIFIED,
        }
        groth16_ready = (
            groth16.available and groth16_artifacts.status in configured_states
        )
        provekit_ready = (
            provekit.available and provekit_artifacts.status in configured_states
        )
        if not package.available or not circuits_package.available:
            status = CapabilityHealth.UNAVAILABLE
            reason = (
                package.reason if not package.available else circuits_package.reason
            )
        elif groth16_ready or provekit_ready:
            status = CapabilityHealth.AVAILABLE
            reason = "at least one cryptographic ZKP backend and its artifacts are discoverable"
        else:
            status = CapabilityHealth.DEGRADED
            reason = (
                "ZKP interfaces and simulated backend are discoverable, but no real "
                "backend has both an executable and circuit artifacts"
            )
        real_backend_checks = (
            CapabilityHealthCheck(
                dimension=CapabilityDimension.PROVIDER,
                name="Groth16 backend",
                status=CapabilityHealth.AVAILABLE
                if groth16_ready
                else CapabilityHealth.UNAVAILABLE,
                reason=(
                    "Groth16 executable and setup artifacts are available but "
                    "production self-tests have not been supplied to this metadata probe"
                    if groth16_ready
                    else "Groth16 requires both its Rust executable and setup artifacts"
                ),
                metadata={"cryptographic": True, "production_verified": False},
            ),
            CapabilityHealthCheck(
                dimension=CapabilityDimension.PROVIDER,
                name="ProveKit backend",
                status=CapabilityHealth.AVAILABLE
                if provekit_ready
                else CapabilityHealth.UNAVAILABLE,
                reason=(
                    "ProveKit executable and circuit artifacts are available but "
                    "production self-tests have not been supplied to this metadata probe"
                    if provekit_ready
                    else "ProveKit requires both its CLI and prepared circuit artifacts"
                ),
                metadata={"cryptographic": True, "production_verified": False},
            ),
        )
        return self._component(
            "zkp_backends",
            "Zero-Knowledge Proof Backends",
            status,
            reason,
            (
                self._provider_check("zkp_backends", status, reason),
                simulated,
                *real_backend_checks,
                package,
                groth16,
                provekit,
                circuits_package,
                groth16_artifacts,
                provekit_artifacts,
                py_ecc,
            ),
        )


_DEFAULT_PROBE = FormalVerificationCapabilityProbe()


def probe_formal_verification_capabilities(
    *,
    force_refresh: bool = False,
) -> FormalVerificationCapabilityReport:
    """Return the process-wide cached formal-verification capability report."""

    return _DEFAULT_PROBE.probe(force_refresh=force_refresh)


def clear_formal_verification_capability_cache() -> None:
    """Clear the process-wide report cache, primarily for tests and operators."""

    _DEFAULT_PROBE.clear_cache()


__all__ = [
    "FORMAL_VERIFICATION_CAPABILITY_SCHEMA_VERSION",
    "FORMAL_VERIFICATION_CAPABILITY_REPORT_VERSION",
    "PROOF_PROVIDER_CAPABILITY_SCHEMA_VERSION",
    "DEFAULT_CAPABILITY_CACHE_TTL_SECONDS",
    "DEFAULT_CAPABILITY_PROBE_TIMEOUT_SECONDS",
    "DEFAULT_CAPABILITY_PROBE_MAX_CHECKS",
    "CapabilityHealth",
    "CapabilityDimension",
    "CapabilityHealthCheck",
    "ProofProviderOperation",
    "ProofProviderIsolation",
    "ProofProviderCapability",
    "ProviderCapabilities",
    "FormalVerificationProviderCapability",
    "FormalVerificationCapabilityReport",
    "FormalVerificationProbeConfig",
    "FormalVerificationCapabilityProbe",
    "probe_formal_verification_capabilities",
    "clear_formal_verification_capability_cache",
]
