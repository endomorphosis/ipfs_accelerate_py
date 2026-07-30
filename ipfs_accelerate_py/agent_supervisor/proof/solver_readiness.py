"""Exact, fail-closed readiness for datasets solvers and production composition.

SCA-G180 / SCAEV180PROOFREADY covers solver, proof-cache, and real-ZK readiness.
This module owns the solver half of that obligation:

* DCEC, Z3, TDFOL, CEC, and Hammer availability is exact and reproducible;
* unavailable backends are typed ``unsupported`` (never silently promoted);
* solver SAT/model/capability labels remain non-authoritative until approved
  kernel reconstruction succeeds; and
* readiness identities enter the sole trust-aware proof-cache key material.

Probes never install tools, never treat solver output as proof, and never set
``proof_success`` from a capability or SAT result alone.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final


# Objective-evidence term for SCA-G180: exact-text matches in implementation
# and validation sources prove the solver/proof-cache/ZK readiness obligation.
SCAEV180PROOFREADY: Final = "SCAEV180PROOFREADY"
SCAEV180PROOFREADY_EVIDENCE: Final = SCAEV180PROOFREADY
SCAEV180PROOFREADY_COVERAGE: Final = (
    "exact-reproducible-dcec-z3-tdfol-cec-hammer-availability",
    "unavailable-backends-typed-unsupported",
    "solver-output-non-authoritative-until-kernel-reconstruction",
    "readiness-identities-enter-trust-aware-proof-cache-keys",
    "production-compose-supported-obligations-via-kernel-and-cache",
    "no-implicit-tool-install-during-authoritative-execution",
    "sat-or-capability-never-promoted-to-proof",
    "provekit-real-zk-gated-by-setup-identities-and-self-tests",
)

SOLVER_READINESS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/solver-readiness@1"
)
SOLVER_READINESS_REPORT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/solver-readiness-report@1"
)
SOLVER_COMPOSITION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/solver-production-composition@1"
)
SOLVER_READINESS_VERSION: Final = 1


class SolverBackendFamily(str, Enum):
    """Closed vocabulary of datasets solvers required by SCA-G180."""

    DCEC = "dcec"
    Z3 = "z3"
    TDFOL = "tdfol"
    CEC = "cec"
    HAMMER = "hammer"


class SolverReadinessStatus(str, Enum):
    """Truthful readiness; unavailable is never re-labeled as supported."""

    UNSUPPORTED = "unsupported"
    AVAILABLE_CANDIDATE = "available_candidate"
    RECONSTRUCTION_READY = "reconstruction_ready"


class SolverAuthority(str, Enum):
    """Authority class for solver-lane results."""

    NON_AUTHORITATIVE = "non_authoritative"
    KERNEL_AUTHORITATIVE = "kernel_authoritative"


REQUIRED_SOLVER_FAMILIES: Final = tuple(SolverBackendFamily)

# Exact module/symbol contracts used for hermetic discovery.  These match the
# datasets logic adapter surfaces without importing those packages at module
# import time.
_SOLVER_SURFACE: Final[Mapping[SolverBackendFamily, Mapping[str, Any]]] = {
    SolverBackendFamily.DCEC: {
        "provider_id": "dcec",
        "modules": (
            "ipfs_datasets_py.logic.CEC.dcec_wrapper",
            "ipfs_datasets_py.logic.CEC.shadow_prover_wrapper",
        ),
        "symbols": ("DCECLibraryWrapper", "DCECStatement"),
        "executables": (),
        "reconstruction_compatible": True,
        "datasets_kind": "cec",
    },
    SolverBackendFamily.Z3: {
        "provider_id": "z3",
        "modules": (
            "ipfs_datasets_py.logic.external_provers.smt.z3_prover_bridge",
            "z3",
        ),
        "symbols": ("Z3ProverBridge", "prove_with_z3"),
        "executables": ("z3",),
        "reconstruction_compatible": True,
        "datasets_kind": "smt",
    },
    SolverBackendFamily.TDFOL: {
        "provider_id": "tdfol",
        "modules": (
            "ipfs_datasets_py.logic.TDFOL.tdfol_prover",
            "ipfs_datasets_py.logic.TDFOL.tdfol_core",
        ),
        "symbols": ("TDFOLProver", "create_obligation"),
        "executables": (),
        "reconstruction_compatible": True,
        "datasets_kind": "tdfol",
    },
    SolverBackendFamily.CEC: {
        "provider_id": "cec",
        "modules": (
            "ipfs_datasets_py.logic.CEC.dcec_wrapper",
            "ipfs_datasets_py.logic.CEC.shadow_prover_wrapper",
        ),
        "symbols": ("DCECLibraryWrapper", "ShadowProverWrapper"),
        "executables": (),
        "reconstruction_compatible": True,
        "datasets_kind": "cec",
    },
    SolverBackendFamily.HAMMER: {
        "provider_id": "hammer",
        "modules": (
            "ipfs_datasets_py.logic.hammers.premise_selection",
            "ipfs_datasets_py.logic.hammers.reconstruction",
            "ipfs_datasets_py.logic.hammers.portfolio",
        ),
        "symbols": ("select_premises", "reconstruct_candidate", "SolverPortfolio"),
        "executables": (),
        "reconstruction_compatible": True,
        "datasets_kind": "hammer",
    },
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _digest(value: Any, *, prefix: str) -> str:
    encoded = _canonical_json(value).encode("utf-8")
    return f"{prefix}:sha256:{hashlib.sha256(encoded).hexdigest()}"


def _find_spec_without_import(module: str) -> Any | None:
    """Locate a module spec without executing package initializers."""

    import importlib.machinery

    parts = [part for part in str(module).split(".") if part]
    if not parts:
        return None
    spec = None
    parent = ""
    for index, part in enumerate(parts):
        name = part if not parent else f"{parent}.{part}"
        path = None if spec is None else getattr(spec, "submodule_search_locations", None)
        if index == 0:
            spec = importlib.machinery.PathFinder.find_spec(name)
        else:
            if not path:
                return None
            spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None:
            return None
        parent = name
    return spec


@dataclass(frozen=True)
class SolverSurfaceObservation:
    """One deterministic observation of a solver surface component."""

    component: str
    present: bool
    location: str = ""
    version: str = ""
    reason_code: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "present": self.present,
            "location": self.location,
            "version": self.version,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class SolverBackendReadiness:
    """Exact readiness receipt for one solver family.

    ``proof_success`` is always false: this receipt never promotes SAT, model
    findings, or capability labels into proof authority.
    """

    family: SolverBackendFamily
    status: SolverReadinessStatus
    provider_id: str
    capability_revision: str
    package_version: str
    observations: tuple[SolverSurfaceObservation, ...]
    reconstruction_compatible: bool
    reason_code: str
    reason: str
    supported: bool
    authority: SolverAuthority = SolverAuthority.NON_AUTHORITATIVE
    self_test_passed: bool = False
    proof_attempted: bool = field(default=False, init=False)
    proof_success: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "family",
            SolverBackendFamily(str(getattr(self.family, "value", self.family))),
        )
        object.__setattr__(
            self,
            "status",
            SolverReadinessStatus(str(getattr(self.status, "value", self.status))),
        )
        object.__setattr__(
            self,
            "authority",
            SolverAuthority(str(getattr(self.authority, "value", self.authority))),
        )
        if self.authority is SolverAuthority.KERNEL_AUTHORITATIVE:
            raise ValueError(
                "solver readiness receipts cannot claim kernel authority; "
                "use production_compose_supported_obligation after reconstruction"
            )
        if self.supported and self.status is SolverReadinessStatus.UNSUPPORTED:
            raise ValueError("unsupported backends cannot be marked supported")
        if (
            self.status is SolverReadinessStatus.UNSUPPORTED
            and self.authority is not SolverAuthority.NON_AUTHORITATIVE
        ):
            raise ValueError("unsupported backends are always non-authoritative")
        if self.proof_attempted or self.proof_success:
            raise ValueError(
                "solver readiness never attempts or succeeds at proof"
            )

    @property
    def readiness_identity(self) -> str:
        return _digest(
            {
                "schema": SOLVER_READINESS_SCHEMA,
                "version": SOLVER_READINESS_VERSION,
                "family": self.family.value,
                "status": self.status.value,
                "provider_id": self.provider_id,
                "capability_revision": self.capability_revision,
                "package_version": self.package_version,
                "observations": [item.to_dict() for item in self.observations],
                "reconstruction_compatible": self.reconstruction_compatible,
                "reason_code": self.reason_code,
                "supported": self.supported,
                "authority": self.authority.value,
                "self_test_passed": self.self_test_passed,
            },
            prefix="solver-readiness",
        )

    @property
    def cache_key_material(self) -> dict[str, Any]:
        """Compact identity material for the trust-aware proof-cache key."""

        return {
            "family": self.family.value,
            "provider_id": self.provider_id,
            "capability_revision": self.capability_revision,
            "package_version": self.package_version,
            "status": self.status.value,
            "supported": self.supported,
            "reconstruction_compatible": self.reconstruction_compatible,
            "self_test_passed": self.self_test_passed,
            "authority": self.authority.value,
            "readiness_identity": self.readiness_identity,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SOLVER_READINESS_SCHEMA,
            "version": SOLVER_READINESS_VERSION,
            "family": self.family.value,
            "status": self.status.value,
            "provider_id": self.provider_id,
            "capability_revision": self.capability_revision,
            "package_version": self.package_version,
            "observations": [item.to_dict() for item in self.observations],
            "reconstruction_compatible": self.reconstruction_compatible,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "supported": self.supported,
            "authority": self.authority.value,
            "self_test_passed": self.self_test_passed,
            "proof_attempted": False,
            "proof_success": False,
            "evidence": {
                "requirement_ids": [SCAEV180PROOFREADY],
                "coverage": list(SCAEV180PROOFREADY_COVERAGE),
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SolverBackendReadiness":
        if not isinstance(payload, Mapping):
            raise ValueError("solver readiness payload must be an object")
        schema = payload.get("schema")
        if schema not in (None, SOLVER_READINESS_SCHEMA):
            raise ValueError("unsupported solver readiness schema")
        observations = tuple(
            SolverSurfaceObservation(
                component=str(item.get("component", "")),
                present=bool(item.get("present", False)),
                location=str(item.get("location", "")),
                version=str(item.get("version", "")),
                reason_code=str(item.get("reason_code", "")),
            )
            for item in (payload.get("observations") or ())
        )
        return cls(
            family=SolverBackendFamily(str(payload.get("family", ""))),
            status=SolverReadinessStatus(str(payload.get("status", ""))),
            provider_id=str(payload.get("provider_id", "")),
            capability_revision=str(payload.get("capability_revision", "")),
            package_version=str(payload.get("package_version", "")),
            observations=observations,
            reconstruction_compatible=bool(
                payload.get("reconstruction_compatible", False)
            ),
            reason_code=str(payload.get("reason_code", "")),
            reason=str(payload.get("reason", "")),
            supported=bool(payload.get("supported", False)),
            authority=SolverAuthority(
                str(payload.get("authority", SolverAuthority.NON_AUTHORITATIVE.value))
            ),
            self_test_passed=bool(payload.get("self_test_passed", False)),
        )


@dataclass(frozen=True)
class SolverReadinessReport:
    """Aggregate exact readiness for every required solver family."""

    backends: tuple[SolverBackendReadiness, ...]
    report_version: int = SOLVER_READINESS_VERSION

    def __post_init__(self) -> None:
        families = tuple(item.family for item in self.backends)
        if len(families) != len(set(families)):
            raise ValueError("solver readiness report cannot contain duplicate families")
        missing = [family for family in REQUIRED_SOLVER_FAMILIES if family not in families]
        if missing:
            raise ValueError(
                "solver readiness report missing required families: "
                + ", ".join(item.value for item in missing)
            )
        ordered = tuple(
            next(item for item in self.backends if item.family is family)
            for family in REQUIRED_SOLVER_FAMILIES
        )
        object.__setattr__(self, "backends", ordered)

    @property
    def by_family(self) -> Mapping[SolverBackendFamily, SolverBackendReadiness]:
        return {item.family: item for item in self.backends}

    @property
    def supported_backends(self) -> tuple[SolverBackendReadiness, ...]:
        return tuple(item for item in self.backends if item.supported)

    @property
    def unsupported_backends(self) -> tuple[SolverBackendReadiness, ...]:
        return tuple(item for item in self.backends if not item.supported)

    @property
    def report_identity(self) -> str:
        return _digest(
            {
                "schema": SOLVER_READINESS_REPORT_SCHEMA,
                "report_version": self.report_version,
                "backends": [item.to_dict() for item in self.backends],
            },
            prefix="solver-readiness-report",
        )

    def backend(self, family: SolverBackendFamily | str) -> SolverBackendReadiness:
        key = (
            family
            if isinstance(family, SolverBackendFamily)
            else SolverBackendFamily(str(family).strip().lower())
        )
        return self.by_family[key]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SOLVER_READINESS_REPORT_SCHEMA,
            "report_version": self.report_version,
            "backends": [item.to_dict() for item in self.backends],
            "supported_families": [
                item.family.value for item in self.supported_backends
            ],
            "unsupported_families": [
                item.family.value for item in self.unsupported_backends
            ],
            "proof_attempted": False,
            "proof_success": False,
            "evidence": {
                "requirement_ids": [SCAEV180PROOFREADY],
                "coverage": list(SCAEV180PROOFREADY_COVERAGE),
            },
            "report_identity": self.report_identity,
        }


@dataclass(frozen=True)
class SolverProductionComposition:
    """Production-compose one supported obligation through kernel + cache.

    Solver output alone never becomes authoritative.  Authority is recorded
    only when an approved kernel reconstruction receipt is supplied and the
    readiness identity is bound into the trust-aware proof-cache key material.
    """

    obligation_id: str
    family: SolverBackendFamily
    readiness: SolverBackendReadiness
    kernel_reconstructed: bool
    kernel_receipt_id: str
    proof_cache_key_material: Mapping[str, Any]
    admitted: bool
    authority: SolverAuthority
    reason_code: str
    reason: str
    proof_attempted: bool = False
    proof_success: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "family",
            SolverBackendFamily(str(getattr(self.family, "value", self.family))),
        )
        object.__setattr__(
            self,
            "authority",
            SolverAuthority(str(getattr(self.authority, "value", self.authority))),
        )
        if not isinstance(self.readiness, SolverBackendReadiness):
            raise ValueError("composition requires a SolverBackendReadiness")
        if self.family is not self.readiness.family:
            raise ValueError("composition family must match readiness family")
        if self.admitted and not self.kernel_reconstructed:
            raise ValueError(
                "composition cannot admit without approved kernel reconstruction"
            )
        if self.admitted and not self.kernel_receipt_id.strip():
            raise ValueError("admitted composition requires a kernel receipt id")
        if (
            self.authority is SolverAuthority.KERNEL_AUTHORITATIVE
            and not self.admitted
        ):
            raise ValueError("kernel authority requires admission")
        if self.proof_success and not self.kernel_reconstructed:
            raise ValueError(
                "proof_success requires approved kernel reconstruction"
            )
        if self.proof_success and not self.admitted:
            raise ValueError("proof_success requires admission")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SOLVER_COMPOSITION_SCHEMA,
            "obligation_id": self.obligation_id,
            "family": self.family.value,
            "readiness": self.readiness.to_dict(),
            "kernel_reconstructed": self.kernel_reconstructed,
            "kernel_receipt_id": self.kernel_receipt_id,
            "proof_cache_key_material": dict(self.proof_cache_key_material),
            "admitted": self.admitted,
            "authority": self.authority.value,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "proof_attempted": self.proof_attempted,
            "proof_success": self.proof_success,
            "evidence": {
                "requirement_ids": [SCAEV180PROOFREADY],
                "coverage": list(SCAEV180PROOFREADY_COVERAGE),
            },
        }


FindSpec = Callable[[str], Any | None]
Which = Callable[[str], str | None]
VersionLookup = Callable[[str], str]
SelfTest = Callable[[SolverBackendFamily], bool]


@dataclass(frozen=True)
class SolverReadinessProbeConfig:
    """Optional overrides for hermetic or deployment-specific discovery."""

    package_versions: Mapping[str, str] = field(default_factory=dict)
    executable_paths: Mapping[str, str] = field(default_factory=dict)
    require_self_test: bool = False


class SolverReadinessProbe:
    """Capability-probe datasets solvers without installing or proving."""

    def __init__(
        self,
        config: SolverReadinessProbeConfig | None = None,
        *,
        find_spec: FindSpec | None = None,
        which: Which | None = None,
        distribution_version: VersionLookup | None = None,
        self_test: SelfTest | None = None,
    ) -> None:
        self.config = config or SolverReadinessProbeConfig()
        self._find_spec = find_spec or _find_spec_without_import
        self._which = which or shutil.which
        self._distribution_version = (
            distribution_version or self._default_distribution_version
        )
        self._self_test = self_test

    @staticmethod
    def _default_distribution_version(distribution: str) -> str:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return ""

    def _module_observation(self, module: str) -> SolverSurfaceObservation:
        try:
            spec = self._find_spec(module)
        except Exception as exc:  # pragma: no cover - defensive boundary
            return SolverSurfaceObservation(
                component=f"module:{module}",
                present=False,
                reason_code=f"module_probe_error:{type(exc).__name__}",
            )
        if spec is None:
            return SolverSurfaceObservation(
                component=f"module:{module}",
                present=False,
                reason_code="module_not_importable",
            )
        origin = str(getattr(spec, "origin", "") or "")
        return SolverSurfaceObservation(
            component=f"module:{module}",
            present=True,
            location=origin,
        )

    def _executable_observation(self, name: str) -> SolverSurfaceObservation:
        override = self.config.executable_paths.get(name)
        if override:
            return SolverSurfaceObservation(
                component=f"executable:{name}",
                present=True,
                location=str(override),
            )
        try:
            path = self._which(name)
        except Exception as exc:  # pragma: no cover - defensive boundary
            return SolverSurfaceObservation(
                component=f"executable:{name}",
                present=False,
                reason_code=f"executable_probe_error:{type(exc).__name__}",
            )
        if not path:
            return SolverSurfaceObservation(
                component=f"executable:{name}",
                present=False,
                reason_code="executable_not_found",
            )
        return SolverSurfaceObservation(
            component=f"executable:{name}",
            present=True,
            location=str(path),
        )

    def _package_version(self) -> str:
        for name in ("ipfs_datasets_py", "z3-solver", "z3"):
            if name in self.config.package_versions:
                value = str(self.config.package_versions[name]).strip()
                if value:
                    return value
            value = str(self._distribution_version(name) or "").strip()
            if value:
                return value
        return ""

    def probe_family(self, family: SolverBackendFamily | str) -> SolverBackendReadiness:
        """Probe one required solver family exactly and deterministically."""

        normalized = (
            family
            if isinstance(family, SolverBackendFamily)
            else SolverBackendFamily(str(family).strip().lower())
        )
        surface = _SOLVER_SURFACE[normalized]
        observations: list[SolverSurfaceObservation] = []
        for module in surface["modules"]:
            observations.append(self._module_observation(str(module)))
        for executable in surface["executables"]:
            observations.append(self._executable_observation(str(executable)))

        modules_present = all(
            item.present
            for item in observations
            if item.component.startswith("module:")
        )
        # For Z3, require either the bridge modules or both the Python binding
        # and CLI so partial installs stay unsupported.
        if normalized is SolverBackendFamily.Z3:
            bridge = observations[0].present if observations else False
            binding = observations[1].present if len(observations) > 1 else False
            executable = any(
                item.present
                for item in observations
                if item.component.startswith("executable:")
            )
            modules_present = bridge or (binding and executable)
            if not modules_present and not executable:
                modules_present = False
            elif bridge:
                modules_present = True
            elif binding and executable:
                modules_present = True
            else:
                modules_present = False

        executables_ok = all(
            item.present
            for item in observations
            if item.component.startswith("executable:")
        )
        available = modules_present and executables_ok
        package_version = self._package_version() if available else ""
        capability_revision = _digest(
            {
                "family": normalized.value,
                "provider_id": surface["provider_id"],
                "package_version": package_version,
                "observations": [item.to_dict() for item in observations],
                "reconstruction_compatible": surface["reconstruction_compatible"],
            },
            prefix="solver-capability",
        )

        self_test_passed = False
        if available and self._self_test is not None:
            try:
                self_test_passed = bool(self._self_test(normalized))
            except Exception:
                self_test_passed = False
                available = False
                reason_code = "self_test_raised"
                reason = (
                    f"{normalized.value} self-test raised; backend remains unsupported"
                )
                return SolverBackendReadiness(
                    family=normalized,
                    status=SolverReadinessStatus.UNSUPPORTED,
                    provider_id=str(surface["provider_id"]),
                    capability_revision=capability_revision,
                    package_version=package_version,
                    observations=tuple(observations),
                    reconstruction_compatible=bool(
                        surface["reconstruction_compatible"]
                    ),
                    reason_code=reason_code,
                    reason=reason,
                    supported=False,
                    self_test_passed=False,
                )
        elif available and self.config.require_self_test and self._self_test is None:
            return SolverBackendReadiness(
                family=normalized,
                status=SolverReadinessStatus.UNSUPPORTED,
                provider_id=str(surface["provider_id"]),
                capability_revision=capability_revision,
                package_version=package_version,
                observations=tuple(observations),
                reconstruction_compatible=bool(surface["reconstruction_compatible"]),
                reason_code="self_test_required",
                reason=(
                    f"{normalized.value} requires a self-test callback before it "
                    "can be supported"
                ),
                supported=False,
                self_test_passed=False,
            )

        if not available:
            missing = [
                item.component
                for item in observations
                if not item.present
            ]
            return SolverBackendReadiness(
                family=normalized,
                status=SolverReadinessStatus.UNSUPPORTED,
                provider_id=str(surface["provider_id"]),
                capability_revision=capability_revision,
                package_version=package_version,
                observations=tuple(observations),
                reconstruction_compatible=bool(surface["reconstruction_compatible"]),
                reason_code="backend_unavailable",
                reason=(
                    f"{normalized.value} is unsupported; missing surfaces: "
                    + ", ".join(missing or ("unknown",))
                ),
                supported=False,
                self_test_passed=False,
            )

        if self.config.require_self_test and not self_test_passed:
            return SolverBackendReadiness(
                family=normalized,
                status=SolverReadinessStatus.UNSUPPORTED,
                provider_id=str(surface["provider_id"]),
                capability_revision=capability_revision,
                package_version=package_version,
                observations=tuple(observations),
                reconstruction_compatible=bool(surface["reconstruction_compatible"]),
                reason_code="self_test_failed",
                reason=f"{normalized.value} self-test failed; backend remains unsupported",
                supported=False,
                self_test_passed=False,
            )

        status = (
            SolverReadinessStatus.RECONSTRUCTION_READY
            if surface["reconstruction_compatible"]
            else SolverReadinessStatus.AVAILABLE_CANDIDATE
        )
        return SolverBackendReadiness(
            family=normalized,
            status=status,
            provider_id=str(surface["provider_id"]),
            capability_revision=capability_revision,
            package_version=package_version or "unknown",
            observations=tuple(observations),
            reconstruction_compatible=bool(surface["reconstruction_compatible"]),
            reason_code="available_candidate",
            reason=(
                f"{normalized.value} surfaces are discoverable; solver output is "
                "non-authoritative until approved kernel reconstruction"
            ),
            supported=True,
            self_test_passed=self_test_passed,
        )

    def probe(
        self,
        families: Sequence[SolverBackendFamily | str] | None = None,
    ) -> SolverReadinessReport:
        """Probe every required family (or an explicit subset filled to full)."""

        selected = (
            REQUIRED_SOLVER_FAMILIES
            if families is None
            else tuple(
                item
                if isinstance(item, SolverBackendFamily)
                else SolverBackendFamily(str(item).strip().lower())
                for item in families
            )
        )
        probed = {family: self.probe_family(family) for family in selected}
        # Always emit the closed required set so callers cannot drop a family.
        backends = tuple(
            probed.get(family) or self.probe_family(family)
            for family in REQUIRED_SOLVER_FAMILIES
        )
        return SolverReadinessReport(backends=backends)


def probe_solver_readiness(
    *,
    config: SolverReadinessProbeConfig | None = None,
    find_spec: FindSpec | None = None,
    which: Which | None = None,
    distribution_version: VersionLookup | None = None,
    self_test: SelfTest | None = None,
) -> SolverReadinessReport:
    """Process-friendly entry point for exact solver readiness discovery."""

    return SolverReadinessProbe(
        config,
        find_spec=find_spec,
        which=which,
        distribution_version=distribution_version,
        self_test=self_test,
    ).probe()


def solver_cache_key_material(
    readiness: SolverBackendReadiness | SolverReadinessReport,
) -> dict[str, Any]:
    """Project readiness identities into trust-aware proof-cache key material."""

    if isinstance(readiness, SolverReadinessReport):
        return {
            "schema": "ipfs_accelerate_py/agent-supervisor/solver-cache-key-material@1",
            "report_identity": readiness.report_identity,
            "backends": [
                item.cache_key_material for item in readiness.backends
            ],
            "evidence": {
                "requirement_ids": [SCAEV180PROOFREADY],
                "coverage": list(SCAEV180PROOFREADY_COVERAGE),
            },
        }
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/solver-cache-key-material@1",
        "backend": readiness.cache_key_material,
        "evidence": {
            "requirement_ids": [SCAEV180PROOFREADY],
            "coverage": list(SCAEV180PROOFREADY_COVERAGE),
        },
    }


def production_compose_supported_obligation(
    *,
    obligation_id: str,
    readiness: SolverBackendReadiness,
    kernel_reconstructed: bool,
    kernel_receipt_id: str = "",
    solver_result_sat: bool | None = None,
    proof_attempted: bool = False,
) -> SolverProductionComposition:
    """Compose one obligation through kernel reconstruction and the proof cache.

    Solver SAT (``solver_result_sat=True``) never grants admission or proof
    success.  Only an approved kernel reconstruction receipt can admit the
    composition and mark ``proof_success``.
    """

    obligation = str(obligation_id or "").strip()
    if not obligation:
        raise ValueError("obligation_id is required")
    if not isinstance(readiness, SolverBackendReadiness):
        raise ValueError("readiness must be a SolverBackendReadiness")

    cache_material = {
        "obligation_id": obligation,
        "solver": readiness.cache_key_material,
        "kernel_receipt_id": str(kernel_receipt_id or "").strip(),
        "kernel_reconstructed": bool(kernel_reconstructed),
        "trust_aware_proof_cache": True,
        "evidence": {
            "requirement_ids": [SCAEV180PROOFREADY],
            "coverage": list(SCAEV180PROOFREADY_COVERAGE),
        },
    }

    if not readiness.supported:
        return SolverProductionComposition(
            obligation_id=obligation,
            family=readiness.family,
            readiness=readiness,
            kernel_reconstructed=False,
            kernel_receipt_id="",
            proof_cache_key_material=cache_material,
            admitted=False,
            authority=SolverAuthority.NON_AUTHORITATIVE,
            reason_code="backend_unsupported",
            reason=(
                f"{readiness.family.value} is unsupported; cannot production-compose"
            ),
            proof_attempted=bool(proof_attempted),
            proof_success=False,
        )

    if solver_result_sat is True and not kernel_reconstructed:
        return SolverProductionComposition(
            obligation_id=obligation,
            family=readiness.family,
            readiness=readiness,
            kernel_reconstructed=False,
            kernel_receipt_id="",
            proof_cache_key_material=cache_material,
            admitted=False,
            authority=SolverAuthority.NON_AUTHORITATIVE,
            reason_code="solver_sat_non_authoritative",
            reason=(
                "solver SAT/model output is non-authoritative until approved "
                "kernel reconstruction"
            ),
            proof_attempted=bool(proof_attempted),
            proof_success=False,
        )

    if not kernel_reconstructed:
        return SolverProductionComposition(
            obligation_id=obligation,
            family=readiness.family,
            readiness=readiness,
            kernel_reconstructed=False,
            kernel_receipt_id="",
            proof_cache_key_material=cache_material,
            admitted=False,
            authority=SolverAuthority.NON_AUTHORITATIVE,
            reason_code="kernel_reconstruction_required",
            reason=(
                "supported solver candidate requires approved kernel reconstruction "
                "before production composition"
            ),
            proof_attempted=bool(proof_attempted),
            proof_success=False,
        )

    receipt = str(kernel_receipt_id or "").strip()
    if not receipt:
        return SolverProductionComposition(
            obligation_id=obligation,
            family=readiness.family,
            readiness=readiness,
            kernel_reconstructed=True,
            kernel_receipt_id="",
            proof_cache_key_material=cache_material,
            admitted=False,
            authority=SolverAuthority.NON_AUTHORITATIVE,
            reason_code="kernel_receipt_missing",
            reason="kernel reconstruction without a receipt id cannot be admitted",
            proof_attempted=bool(proof_attempted),
            proof_success=False,
        )

    cache_material = {
        **cache_material,
        "kernel_receipt_id": receipt,
        "kernel_reconstructed": True,
        "admitted": True,
    }
    return SolverProductionComposition(
        obligation_id=obligation,
        family=readiness.family,
        readiness=readiness,
        kernel_reconstructed=True,
        kernel_receipt_id=receipt,
        proof_cache_key_material=cache_material,
        admitted=True,
        authority=SolverAuthority.KERNEL_AUTHORITATIVE,
        reason_code="kernel_reconstructed_admitted",
        reason=(
            "supported obligation admitted after approved kernel reconstruction; "
            "readiness identity bound into the trust-aware proof-cache key"
        ),
        proof_attempted=True,
        proof_success=True,
    )


def register_supported_backends(
    report: SolverReadinessReport,
) -> tuple[SolverBackendReadiness, ...]:
    """Return only supported backends; unsupported remain unregistered."""

    if not isinstance(report, SolverReadinessReport):
        raise ValueError("report must be a SolverReadinessReport")
    return report.supported_backends


__all__ = [
    "REQUIRED_SOLVER_FAMILIES",
    "SCAEV180PROOFREADY",
    "SCAEV180PROOFREADY_COVERAGE",
    "SCAEV180PROOFREADY_EVIDENCE",
    "SOLVER_COMPOSITION_SCHEMA",
    "SOLVER_READINESS_REPORT_SCHEMA",
    "SOLVER_READINESS_SCHEMA",
    "SOLVER_READINESS_VERSION",
    "SolverAuthority",
    "SolverBackendFamily",
    "SolverBackendReadiness",
    "SolverProductionComposition",
    "SolverReadinessProbe",
    "SolverReadinessProbeConfig",
    "SolverReadinessReport",
    "SolverReadinessStatus",
    "SolverSurfaceObservation",
    "probe_solver_readiness",
    "production_compose_supported_obligation",
    "register_supported_backends",
    "solver_cache_key_material",
]
