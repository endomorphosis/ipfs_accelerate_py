"""Closed, lazy capability matrix for program-analysis optional surfaces.

This module discovers whether :mod:`ipfs_datasets_py` can currently support
strict CID helpers, GraphRAG/knowledge-graph queries, ``ir_core`` claims and
protocols, cvc5/z3 compiler bindings and executables, AST producers, and ZKP
backends/circuits.

Discovery is intentionally weaker than verification:

* constructing the provider and reading its closed matrix declaration never
  imports optional packages;
* package presence alone is never treated as a capability proof — required
  callables and method signatures must be present and compatible;
* simulated or fallback ZKP backends may be reported as diagnostics but never
  as cryptographic authority;
* legacy pseudo-CIDs, unhealthy providers, incompatible schemas, and unbounded
  probe outputs are rejected; and
* current environment observations (for example cvc5 available, z3 absent, or
  a limited crypto-exchange AST extractor) are probe diagnostics, never
  hard-coded constants.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import inspect
import json
import math
import os
import shutil
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final


PROGRAM_ANALYSIS_CAPABILITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "ipfs-datasets-program-analysis-capability@1"
)
PROGRAM_ANALYSIS_CAPABILITY_REPORT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "ipfs-datasets-program-analysis-capability-report@1"
)
PROGRAM_ANALYSIS_CAPABILITY_REPORT_VERSION: Final = 1
IPFS_DATASETS_PROGRAM_ANALYSIS_PROVIDER_ID: Final = (
    "ipfs_datasets_py.program_analysis"
)
IPFS_DATASETS_PROGRAM_ANALYSIS_PROVIDER_VERSION: Final = "1.0.0"

DEFAULT_OPTIONAL_ROOT: Final = "ipfs_datasets_py"
DEFAULT_PROBE_TIMEOUT_SECONDS: Final = 2.0
DEFAULT_PROBE_CACHE_TTL_SECONDS: Final = 300.0
DEFAULT_PROBE_MAX_CHECKS: Final = 96
DEFAULT_MAX_REPORT_BYTES: Final = 128 * 1024
DEFAULT_MAX_REASON_BYTES: Final = 1024
DEFAULT_MAX_METADATA_BYTES: Final = 4096
DEFAULT_MAX_SIGNATURE_DEPTH: Final = 8

# Closed capability families.  Order is part of the public matrix identity.
CAPABILITY_FAMILY_ORDER: Final = (
    "strict_cid",
    "graphrag",
    "ir_core",
    "solvers",
    "ast_producers",
    "zkp",
)

_IR_CORE_CLAIM_EXPORTS: Final = (
    "IRClaim",
    "IRAssumption",
    "IRObligation",
    "ClaimValidationError",
    "IR_CLAIM_SCHEMA_VERSION",
)
_IR_CORE_PROTOCOL_EXPORTS: Final = (
    "AuthorityKind",
    "QueryKind",
    "AttemptStatus",
    "ProtocolValidationError",
    "AuthorityMismatchError",
    "BACKEND_CAPABILITIES_SCHEMA_VERSION",
)
_AST_EXTRACTOR_METHODS: Final = ("extract_from_source",)

# Legacy / pseudo identity shapes that strict CIDv1 helpers must reject.
# Only include forms that a correct validator is required to refuse; never
# include well-formed CIDv1 strings (those are valid identities, not pseudo).
_PSEUDO_CID_SAMPLES: Final = (
    "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG",  # CIDv0
    "cid:not-a-cid",
    "sha256:deadbeef",
    "tree:sha256:111",
    "repo:fixture",
    "BAGAAQEERA",  # non-canonical case / truncated
    "",
)

PackageFinder = Callable[[str], Any]
ExecutableFinder = Callable[[str], str | None]
ModuleImporter = Callable[[str], Any]
Clock = Callable[[], float]


class ProgramAnalysisCapabilityError(ValueError):
    """A capability declaration, probe config, or report violates the contract."""


class CapabilityFamily(str, Enum):
    """Closed vocabulary of program-analysis capability families."""

    STRICT_CID = "strict_cid"
    GRAPHRAG = "graphrag"
    IR_CORE = "ir_core"
    SOLVERS = "solvers"
    AST_PRODUCERS = "ast_producers"
    ZKP = "zkp"


class CapabilityProbeStatus(str, Enum):
    """Truthful readiness of one family or named surface after a probe."""

    LAZY = "lazy"
    AVAILABLE = "available"
    PARTIAL = "partial"
    DEGRADED = "degraded"
    SIMULATED = "simulated"
    UNAVAILABLE = "unavailable"
    INCOMPATIBLE = "incompatible"
    UNHEALTHY = "unhealthy"
    TIMED_OUT = "timed_out"
    REJECTED = "rejected"


class CapabilityAuthority(str, Enum):
    """What, if anything, a positive family result may authorize."""

    NONE = "none"
    DIAGNOSTIC = "diagnostic"
    BOUNDED_QUERY = "bounded_query"
    STRICT_IDENTITY = "strict_identity"
    IR_DECLARATION = "ir_declaration"
    SOLVER_CANDIDATE = "solver_candidate"
    AST_OBSERVATION = "ast_observation"
    # Cryptographic ZKP authority is never granted by discovery alone.
    ZKP_DIAGNOSTIC = "zkp_diagnostic"


class CapabilityReasonCode(str, Enum):
    """Stable machine-readable probe outcomes."""

    NOT_PROBED = "not_probed"
    AVAILABLE = "available"
    PARTIAL = "partial"
    DEGRADED = "degraded"
    PACKAGE_MISSING = "package_missing"
    PACKAGE_PRESENCE_ONLY = "package_presence_only_rejected"
    CALLABLE_MISSING = "callable_missing"
    SIGNATURE_INCOMPATIBLE = "signature_incompatible"
    SCHEMA_INCOMPATIBLE = "schema_incompatible"
    EXECUTABLE_MISSING = "executable_missing"
    BINDING_MISSING = "binding_missing"
    COMPILER_MISSING = "compiler_missing"
    PSEUDO_CID_ACCEPTED = "pseudo_cid_accepted"
    SIMULATED_ZKP_ONLY = "simulated_zkp_only"
    SIMULATED_ZKP_AUTHORITY_REJECTED = "simulated_zkp_authority_rejected"
    FALLBACK_ZKP_REJECTED = "fallback_zkp_rejected"
    UNHEALTHY_PROVIDER = "unhealthy_provider"
    UNBOUNDED_OUTPUT = "unbounded_output"
    PROBE_TIMEOUT = "probe_timeout"
    PROBE_LIMIT = "probe_limit"
    IMPORT_FAILED = "import_failed"
    CURRENT_DIAGNOSTIC = "current_diagnostic"


_AVAILABLE_STATUSES = frozenset(
    {
        CapabilityProbeStatus.AVAILABLE,
        CapabilityProbeStatus.PARTIAL,
        CapabilityProbeStatus.DEGRADED,
        CapabilityProbeStatus.SIMULATED,
    }
)


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
            resolved_locations = tuple(str(location) for location in locations)
            package_name = parts[index]
            nested_locations = tuple(
                str(candidate)
                for location in resolved_locations
                if Path(location).name == package_name
                for candidate in (Path(location) / package_name,)
                if candidate.is_dir() and (candidate / "__init__.py").is_file()
            )
            path = tuple(dict.fromkeys((*nested_locations, *resolved_locations)))
    return spec


def _canonical_json_bytes(value: Any, *, name: str = "value") -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProgramAnalysisCapabilityError(
            f"{name} must be strict JSON-serializable"
        ) from exc


def _content_id(value: Any, *, name: str) -> str:
    digest = hashlib.sha256(_canonical_json_bytes(value, name=name)).hexdigest()
    return f"{name}:sha256:{digest}"


def _text(value: Any, name: str, *, max_bytes: int = DEFAULT_MAX_REASON_BYTES) -> str:
    if not isinstance(value, str):
        raise ProgramAnalysisCapabilityError(f"{name} must be a string")
    result = value.strip()
    if not result:
        raise ProgramAnalysisCapabilityError(f"{name} must not be empty")
    if "\x00" in result:
        raise ProgramAnalysisCapabilityError(f"{name} contains a NUL byte")
    if len(result.encode("utf-8")) > max_bytes:
        raise ProgramAnalysisCapabilityError(f"{name} exceeds {max_bytes} bytes")
    return result


def _positive_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ProgramAnalysisCapabilityError(f"{name} must be finite and positive")
    return float(value)


def _non_negative_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ProgramAnalysisCapabilityError(
            f"{name} must be finite and non-negative"
        )
    return float(value)


def _positive_int(value: Any, name: str, *, maximum: int = 10_000_000) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value > maximum
    ):
        raise ProgramAnalysisCapabilityError(
            f"{name} must be an integer between 1 and {maximum}"
        )
    return value


def _bounded_metadata(
    value: Mapping[str, Any] | None,
    *,
    max_bytes: int = DEFAULT_MAX_METADATA_BYTES,
) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ProgramAnalysisCapabilityError("metadata must be a mapping")
    raw = dict(value)
    encoded = _canonical_json_bytes(raw, name="metadata")
    if len(encoded) > max_bytes:
        raise ProgramAnalysisCapabilityError(
            f"metadata exceeds {max_bytes} bytes"
        )
    return json.loads(encoded.decode("utf-8"))


def _signature_params(func: Any) -> frozenset[str]:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError) as exc:
        raise ProgramAnalysisCapabilityError(
            f"callable signature is uninspectable: {exc}"
        ) from exc
    return frozenset(signature.parameters)


def _has_required_params(func: Any, required: Sequence[str]) -> bool:
    params = _signature_params(func)
    # Methods may bind self; require every declared parameter name.
    return all(name in params for name in required)


def _callable_compatible(
    owner: Any,
    name: str,
    *,
    required_params: Sequence[str] = (),
    allow_missing: bool = False,
) -> tuple[bool, str]:
    target = getattr(owner, name, None)
    if target is None:
        if allow_missing:
            return False, f"{name} is absent"
        return False, f"required callable {name!r} is missing"
    if not callable(target):
        return False, f"{name!r} exists but is not callable"
    if required_params and not _has_required_params(target, required_params):
        return (
            False,
            f"{name!r} signature is incompatible; expected parameters "
            f"{list(required_params)}",
        )
    return True, f"{name} signature is compatible"


@dataclass(frozen=True)
class ProgramAnalysisProbeConfig:
    """Resource limits and injectable roots for one capability snapshot."""

    timeout_seconds: float = DEFAULT_PROBE_TIMEOUT_SECONDS
    cache_ttl_seconds: float = DEFAULT_PROBE_CACHE_TTL_SECONDS
    max_checks: int = DEFAULT_PROBE_MAX_CHECKS
    max_report_bytes: int = DEFAULT_MAX_REPORT_BYTES
    optional_root: str = DEFAULT_OPTIONAL_ROOT
    families: tuple[CapabilityFamily, ...] = tuple(
        CapabilityFamily(item) for item in CAPABILITY_FAMILY_ORDER
    )
    allow_simulated_zkp_authority: bool = False
    run_strict_cid_canary: bool = True
    groth16_artifacts_path: str | None = None
    provekit_artifacts_path: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "timeout_seconds", _positive_number(self.timeout_seconds, "timeout_seconds")
        )
        object.__setattr__(
            self,
            "cache_ttl_seconds",
            _non_negative_number(self.cache_ttl_seconds, "cache_ttl_seconds"),
        )
        object.__setattr__(
            self, "max_checks", _positive_int(self.max_checks, "max_checks", maximum=10_000)
        )
        object.__setattr__(
            self,
            "max_report_bytes",
            _positive_int(self.max_report_bytes, "max_report_bytes"),
        )
        root = _text(self.optional_root, "optional_root", max_bytes=256)
        object.__setattr__(self, "optional_root", root)
        if not isinstance(self.allow_simulated_zkp_authority, bool):
            raise ProgramAnalysisCapabilityError(
                "allow_simulated_zkp_authority must be a boolean"
            )
        if self.allow_simulated_zkp_authority:
            # Simulated ZKP must never become authority through configuration.
            raise ProgramAnalysisCapabilityError(
                "simulated ZKP authority is permanently rejected"
            )
        if not isinstance(self.run_strict_cid_canary, bool):
            raise ProgramAnalysisCapabilityError(
                "run_strict_cid_canary must be a boolean"
            )
        try:
            families = tuple(
                CapabilityFamily(str(getattr(item, "value", item)))
                for item in self.families
            )
        except ValueError as exc:
            raise ProgramAnalysisCapabilityError(
                f"unknown capability families in config: {exc}"
            ) from exc
        if not families:
            raise ProgramAnalysisCapabilityError("at least one family is required")
        unknown = {item.value for item in families} - set(CAPABILITY_FAMILY_ORDER)
        if unknown:
            raise ProgramAnalysisCapabilityError(
                f"unknown capability families: {sorted(unknown)}"
            )
        # Preserve closed matrix order and drop duplicates.
        ordered = tuple(
            family
            for name in CAPABILITY_FAMILY_ORDER
            for family in (CapabilityFamily(name),)
            if family in families
        )
        object.__setattr__(self, "families", ordered)


@dataclass(frozen=True)
class CapabilitySurface:
    """One named dependency/surface inside a capability family."""

    surface_id: str
    status: CapabilityProbeStatus
    reason_code: CapabilityReasonCode
    reason: str
    authority: CapabilityAuthority = CapabilityAuthority.NONE
    required: bool = False
    version: str | None = None
    location: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    proof_attempted: bool = field(default=False, init=False)
    proof_success: bool = field(default=False, init=False)
    completion_authority: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "surface_id",
            _text(self.surface_id, "surface_id", max_bytes=256),
        )
        object.__setattr__(
            self,
            "status",
            CapabilityProbeStatus(str(getattr(self.status, "value", self.status))),
        )
        object.__setattr__(
            self,
            "reason_code",
            CapabilityReasonCode(
                str(getattr(self.reason_code, "value", self.reason_code))
            ),
        )
        object.__setattr__(
            self,
            "reason",
            _text(self.reason, "reason", max_bytes=DEFAULT_MAX_REASON_BYTES),
        )
        object.__setattr__(
            self,
            "authority",
            CapabilityAuthority(str(getattr(self.authority, "value", self.authority))),
        )
        if self.authority is CapabilityAuthority.ZKP_DIAGNOSTIC and self.status in {
            CapabilityProbeStatus.AVAILABLE,
        }:
            # ZKP discovery never upgrades to cryptographic authority.
            pass
        if self.authority not in {
            CapabilityAuthority.NONE,
            CapabilityAuthority.DIAGNOSTIC,
            CapabilityAuthority.BOUNDED_QUERY,
            CapabilityAuthority.STRICT_IDENTITY,
            CapabilityAuthority.IR_DECLARATION,
            CapabilityAuthority.SOLVER_CANDIDATE,
            CapabilityAuthority.AST_OBSERVATION,
            CapabilityAuthority.ZKP_DIAGNOSTIC,
        }:
            raise ProgramAnalysisCapabilityError("unsupported capability authority")
        # Permanent rejects for simulated ZKP being treated as authority.
        if (
            self.authority is not CapabilityAuthority.ZKP_DIAGNOSTIC
            and self.authority is not CapabilityAuthority.NONE
            and self.authority is not CapabilityAuthority.DIAGNOSTIC
            and self.status is CapabilityProbeStatus.SIMULATED
        ):
            raise ProgramAnalysisCapabilityError(
                "simulated surfaces cannot carry non-diagnostic authority"
            )
        object.__setattr__(
            self, "metadata", MappingProxyType(_bounded_metadata(dict(self.metadata)))
        )
        if self.version is not None:
            object.__setattr__(
                self, "version", _text(self.version, "version", max_bytes=128)
            )
        if self.location is not None:
            object.__setattr__(
                self, "location", _text(self.location, "location", max_bytes=1024)
            )
        object.__setattr__(self, "proof_attempted", False)
        object.__setattr__(self, "proof_success", False)
        object.__setattr__(self, "completion_authority", False)

    @property
    def usable(self) -> bool:
        return self.status in _AVAILABLE_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "status": self.status.value,
            "reason_code": self.reason_code.value,
            "reason": self.reason,
            "authority": self.authority.value,
            "required": self.required,
            "version": self.version,
            "location": self.location,
            "metadata": dict(self.metadata),
            "proof_attempted": False,
            "proof_success": False,
            "completion_authority": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CapabilitySurface":
        if not isinstance(payload, Mapping):
            raise ProgramAnalysisCapabilityError("surface payload must be an object")
        return cls(
            surface_id=payload.get("surface_id", ""),
            status=payload.get("status", CapabilityProbeStatus.UNAVAILABLE),
            reason_code=payload.get(
                "reason_code", CapabilityReasonCode.NOT_PROBED
            ),
            reason=payload.get("reason", "missing reason"),
            authority=payload.get("authority", CapabilityAuthority.NONE),
            required=bool(payload.get("required", False)),
            version=payload.get("version"),
            location=payload.get("location"),
            metadata=payload.get("metadata") or {},
        )


@dataclass(frozen=True)
class CapabilityFamilyReport:
    """Probe result for one closed capability family."""

    family: CapabilityFamily
    status: CapabilityProbeStatus
    reason_code: CapabilityReasonCode
    reason: str
    authority: CapabilityAuthority
    surfaces: tuple[CapabilitySurface, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    proof_attempted: bool = field(default=False, init=False)
    proof_success: bool = field(default=False, init=False)
    completion_authority: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "family", CapabilityFamily(str(getattr(self.family, "value", self.family)))
        )
        object.__setattr__(
            self,
            "status",
            CapabilityProbeStatus(str(getattr(self.status, "value", self.status))),
        )
        object.__setattr__(
            self,
            "reason_code",
            CapabilityReasonCode(
                str(getattr(self.reason_code, "value", self.reason_code))
            ),
        )
        object.__setattr__(
            self, "reason", _text(self.reason, "reason", max_bytes=DEFAULT_MAX_REASON_BYTES)
        )
        object.__setattr__(
            self,
            "authority",
            CapabilityAuthority(str(getattr(self.authority, "value", self.authority))),
        )
        if not self.surfaces:
            raise ProgramAnalysisCapabilityError(
                f"family {self.family.value!r} must report at least one surface"
            )
        object.__setattr__(self, "surfaces", tuple(self.surfaces))
        object.__setattr__(
            self, "metadata", MappingProxyType(_bounded_metadata(dict(self.metadata)))
        )
        object.__setattr__(self, "proof_attempted", False)
        object.__setattr__(self, "proof_success", False)
        object.__setattr__(self, "completion_authority", False)

    @property
    def family_id(self) -> str:
        return self.family.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family.value,
            "status": self.status.value,
            "reason_code": self.reason_code.value,
            "reason": self.reason,
            "authority": self.authority.value,
            "surfaces": [surface.to_dict() for surface in self.surfaces],
            "metadata": dict(self.metadata),
            "proof_attempted": False,
            "proof_success": False,
            "completion_authority": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CapabilityFamilyReport":
        if not isinstance(payload, Mapping):
            raise ProgramAnalysisCapabilityError("family payload must be an object")
        surfaces = payload.get("surfaces") or ()
        if not isinstance(surfaces, Sequence) or isinstance(surfaces, (str, bytes)):
            raise ProgramAnalysisCapabilityError("surfaces must be a sequence")
        return cls(
            family=payload.get("family", CapabilityFamily.STRICT_CID),
            status=payload.get("status", CapabilityProbeStatus.UNAVAILABLE),
            reason_code=payload.get(
                "reason_code", CapabilityReasonCode.NOT_PROBED
            ),
            reason=payload.get("reason", "missing reason"),
            authority=payload.get("authority", CapabilityAuthority.NONE),
            surfaces=tuple(CapabilitySurface.from_dict(item) for item in surfaces),
            metadata=payload.get("metadata") or {},
        )


@dataclass(frozen=True)
class ProgramAnalysisCapabilityMatrix:
    """Closed matrix declaration before any optional import or probe."""

    schema_version: str = PROGRAM_ANALYSIS_CAPABILITY_SCHEMA
    provider_id: str = IPFS_DATASETS_PROGRAM_ANALYSIS_PROVIDER_ID
    provider_version: str = IPFS_DATASETS_PROGRAM_ANALYSIS_PROVIDER_VERSION
    families: tuple[CapabilityFamily, ...] = tuple(
        CapabilityFamily(item) for item in CAPABILITY_FAMILY_ORDER
    )
    lazy_import: bool = True
    probed: bool = False
    imported: bool = False
    completion_authority: bool = False
    proof_attempted: bool = False
    proof_success: bool = False
    non_authoritative: bool = True

    def __post_init__(self) -> None:
        if self.schema_version != PROGRAM_ANALYSIS_CAPABILITY_SCHEMA:
            raise ProgramAnalysisCapabilityError(
                "unsupported program-analysis capability schema"
            )
        object.__setattr__(
            self,
            "provider_id",
            _text(self.provider_id, "provider_id", max_bytes=256),
        )
        object.__setattr__(
            self,
            "provider_version",
            _text(self.provider_version, "provider_version", max_bytes=128),
        )
        families = tuple(
            CapabilityFamily(str(getattr(item, "value", item)))
            for item in self.families
        )
        expected = tuple(CapabilityFamily(item) for item in CAPABILITY_FAMILY_ORDER)
        if families != expected:
            raise ProgramAnalysisCapabilityError(
                "capability matrix families must be the closed ordered vocabulary"
            )
        object.__setattr__(self, "families", families)
        if self.lazy_import is not True:
            raise ProgramAnalysisCapabilityError("lazy_import must be true")
        if self.completion_authority or self.proof_attempted or self.proof_success:
            raise ProgramAnalysisCapabilityError(
                "capability matrix cannot claim completion or proof authority"
            )
        if self.non_authoritative is not True:
            raise ProgramAnalysisCapabilityError(
                "capability matrix is permanently non-authoritative"
            )
        object.__setattr__(self, "probed", bool(self.probed))
        object.__setattr__(self, "imported", bool(self.imported))

    @property
    def matrix_id(self) -> str:
        return _content_id(self.to_dict(), name="program-analysis-capability-matrix")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "families": [family.value for family in self.families],
            "lazy_import": True,
            "probed": self.probed,
            "imported": self.imported,
            "completion_authority": False,
            "proof_attempted": False,
            "proof_success": False,
            "non_authoritative": True,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProgramAnalysisCapabilityMatrix":
        if not isinstance(payload, Mapping):
            raise ProgramAnalysisCapabilityError("matrix payload must be an object")
        return cls(
            schema_version=payload.get(
                "schema_version", PROGRAM_ANALYSIS_CAPABILITY_SCHEMA
            ),
            provider_id=payload.get(
                "provider_id", IPFS_DATASETS_PROGRAM_ANALYSIS_PROVIDER_ID
            ),
            provider_version=payload.get(
                "provider_version", IPFS_DATASETS_PROGRAM_ANALYSIS_PROVIDER_VERSION
            ),
            families=tuple(payload.get("families") or CAPABILITY_FAMILY_ORDER),
            lazy_import=payload.get("lazy_import", True),
            probed=bool(payload.get("probed", False)),
            imported=bool(payload.get("imported", False)),
            completion_authority=bool(payload.get("completion_authority", False)),
            proof_attempted=bool(payload.get("proof_attempted", False)),
            proof_success=bool(payload.get("proof_success", False)),
            non_authoritative=payload.get("non_authoritative", True),
        )


@dataclass(frozen=True)
class ProgramAnalysisCapabilityReport:
    """Bounded snapshot of every closed family after a real probe."""

    schema_version: str = PROGRAM_ANALYSIS_CAPABILITY_REPORT_SCHEMA
    report_version: int = PROGRAM_ANALYSIS_CAPABILITY_REPORT_VERSION
    provider_id: str = IPFS_DATASETS_PROGRAM_ANALYSIS_PROVIDER_ID
    provider_version: str = IPFS_DATASETS_PROGRAM_ANALYSIS_PROVIDER_VERSION
    generated_at: str = ""
    duration_seconds: float = 0.0
    probe_count: int = 0
    bounded: bool = True
    overall_status: CapabilityProbeStatus = CapabilityProbeStatus.UNAVAILABLE
    families: tuple[CapabilityFamilyReport, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    matrix: ProgramAnalysisCapabilityMatrix = field(
        default_factory=ProgramAnalysisCapabilityMatrix
    )
    completion_authority: bool = field(default=False, init=False)
    proof_attempted: bool = field(default=False, init=False)
    proof_success: bool = field(default=False, init=False)
    non_authoritative: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        if self.schema_version != PROGRAM_ANALYSIS_CAPABILITY_REPORT_SCHEMA:
            raise ProgramAnalysisCapabilityError(
                "unsupported program-analysis capability report schema"
            )
        if self.report_version != PROGRAM_ANALYSIS_CAPABILITY_REPORT_VERSION:
            raise ProgramAnalysisCapabilityError(
                "unsupported program-analysis capability report version"
            )
        object.__setattr__(
            self,
            "provider_id",
            _text(self.provider_id, "provider_id", max_bytes=256),
        )
        object.__setattr__(
            self,
            "provider_version",
            _text(self.provider_version, "provider_version", max_bytes=128),
        )
        if not self.generated_at:
            object.__setattr__(
                self,
                "generated_at",
                datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            )
        else:
            object.__setattr__(
                self,
                "generated_at",
                _text(self.generated_at, "generated_at", max_bytes=64),
            )
        object.__setattr__(
            self,
            "duration_seconds",
            _non_negative_number(self.duration_seconds, "duration_seconds"),
        )
        if isinstance(self.probe_count, bool) or not isinstance(self.probe_count, int) or self.probe_count < 0:
            raise ProgramAnalysisCapabilityError(
                "probe_count must be a non-negative integer"
            )
        if self.bounded is not True:
            raise ProgramAnalysisCapabilityError("capability reports must be bounded")
        object.__setattr__(
            self,
            "overall_status",
            CapabilityProbeStatus(
                str(getattr(self.overall_status, "value", self.overall_status))
            ),
        )
        expected = tuple(CapabilityFamily(item) for item in CAPABILITY_FAMILY_ORDER)
        families = tuple(self.families)
        if tuple(item.family for item in families) != expected:
            raise ProgramAnalysisCapabilityError(
                "report must cover the closed ordered capability matrix exactly once"
            )
        object.__setattr__(self, "families", families)
        object.__setattr__(
            self,
            "diagnostics",
            MappingProxyType(_bounded_metadata(dict(self.diagnostics))),
        )
        if not isinstance(self.matrix, ProgramAnalysisCapabilityMatrix):
            raise ProgramAnalysisCapabilityError("matrix must be a capability matrix")
        object.__setattr__(self, "completion_authority", False)
        object.__setattr__(self, "proof_attempted", False)
        object.__setattr__(self, "proof_success", False)
        object.__setattr__(self, "non_authoritative", True)

    @property
    def report_id(self) -> str:
        return _content_id(self.to_dict(), name="program-analysis-capability-report")

    def family(self, name: CapabilityFamily | str) -> CapabilityFamilyReport:
        key = CapabilityFamily(str(getattr(name, "value", name)))
        for item in self.families:
            if item.family is key:
                return item
        raise KeyError(f"unknown capability family: {key.value}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "report_version": self.report_version,
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "generated_at": self.generated_at,
            "duration_seconds": self.duration_seconds,
            "probe_count": self.probe_count,
            "bounded": True,
            "overall_status": self.overall_status.value,
            "families": {
                item.family.value: item.to_dict() for item in self.families
            },
            "diagnostics": dict(self.diagnostics),
            "matrix": self.matrix.to_dict(),
            "completion_authority": False,
            "proof_attempted": False,
            "proof_success": False,
            "non_authoritative": True,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProgramAnalysisCapabilityReport":
        if not isinstance(payload, Mapping):
            raise ProgramAnalysisCapabilityError("report payload must be an object")
        raw_families = payload.get("families")
        if isinstance(raw_families, Mapping):
            ordered = tuple(
                CapabilityFamilyReport.from_dict(raw_families[name])
                for name in CAPABILITY_FAMILY_ORDER
            )
        elif isinstance(raw_families, Sequence) and not isinstance(
            raw_families, (str, bytes)
        ):
            ordered = tuple(
                CapabilityFamilyReport.from_dict(item) for item in raw_families
            )
        else:
            raise ProgramAnalysisCapabilityError("families must be a map or sequence")
        matrix_payload = payload.get("matrix") or {}
        return cls(
            schema_version=payload.get(
                "schema_version", PROGRAM_ANALYSIS_CAPABILITY_REPORT_SCHEMA
            ),
            report_version=int(
                payload.get("report_version", PROGRAM_ANALYSIS_CAPABILITY_REPORT_VERSION)
            ),
            provider_id=payload.get(
                "provider_id", IPFS_DATASETS_PROGRAM_ANALYSIS_PROVIDER_ID
            ),
            provider_version=payload.get(
                "provider_version", IPFS_DATASETS_PROGRAM_ANALYSIS_PROVIDER_VERSION
            ),
            generated_at=payload.get("generated_at", ""),
            duration_seconds=payload.get("duration_seconds", 0.0),
            probe_count=payload.get("probe_count", 0),
            bounded=payload.get("bounded", True),
            overall_status=payload.get(
                "overall_status", CapabilityProbeStatus.UNAVAILABLE
            ),
            families=ordered,
            diagnostics=payload.get("diagnostics") or {},
            matrix=ProgramAnalysisCapabilityMatrix.from_dict(matrix_payload),
        )


def declare_program_analysis_capability_matrix(
    *,
    probed: bool = False,
    imported: bool = False,
) -> ProgramAnalysisCapabilityMatrix:
    """Return the closed matrix without importing optional packages."""

    return ProgramAnalysisCapabilityMatrix(probed=probed, imported=imported)


def inspect_program_analysis_capability_matrix(
    payload: Mapping[str, Any] | None = None,
) -> ProgramAnalysisCapabilityMatrix:
    """Pure inspection of a matrix declaration from canonical metadata."""

    if payload is None:
        return declare_program_analysis_capability_matrix()
    return ProgramAnalysisCapabilityMatrix.from_dict(payload)


class ProgramAnalysisCapabilityProbe:
    """Bounded, cacheable probe for the program-analysis capability matrix.

    Injection points (``find_spec``, ``which``, ``importer``, clocks) let tests
    exercise unavailable, incompatible, partial, and timeout paths without
    installing optional toolchains.
    """

    def __init__(
        self,
        config: ProgramAnalysisProbeConfig | None = None,
        *,
        find_spec: PackageFinder | None = None,
        which: ExecutableFinder | None = None,
        importer: ModuleImporter | None = None,
        environ: Mapping[str, str] | None = None,
        monotonic: Clock | None = None,
        wall_clock: Clock | None = None,
    ) -> None:
        self.config = config or ProgramAnalysisProbeConfig()
        self._find_spec = find_spec or _find_spec_without_import
        self._which = which or shutil.which
        self._importer = importer or importlib.import_module
        self._environ = environ if environ is not None else os.environ
        self._monotonic = monotonic or time.monotonic
        self._wall_clock = wall_clock or time.time
        self._cache_lock = threading.Lock()
        self._cached: tuple[float, ProgramAnalysisCapabilityReport] | None = None
        self._started = 0.0
        self._deadline = 0.0
        self._probe_count = 0
        self._import_attempted = False
        self._any_import_succeeded = False

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cached = None

    def matrix(self) -> ProgramAnalysisCapabilityMatrix:
        """Closed matrix declaration; never imports optional packages."""

        return declare_program_analysis_capability_matrix(
            probed=False, imported=False
        )

    def probe(
        self, *, force_refresh: bool = False
    ) -> ProgramAnalysisCapabilityReport:
        """Return a cached snapshot or perform one bounded real probe."""

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
            self._import_attempted = False
            self._any_import_succeeded = False

            family_reports = (
                self._probe_strict_cid(),
                self._probe_graphrag(),
                self._probe_ir_core(),
                self._probe_solvers(),
                self._probe_ast_producers(),
                self._probe_zkp(),
            )
            finished = self._monotonic()
            overall = self._overall_status(family_reports)
            diagnostics = {
                "import_attempted": self._import_attempted,
                "import_succeeded": self._any_import_succeeded,
                "probe_limited": any(
                    surface.reason_code
                    in {
                        CapabilityReasonCode.PROBE_TIMEOUT,
                        CapabilityReasonCode.PROBE_LIMIT,
                    }
                    for family in family_reports
                    for surface in family.surfaces
                ),
                "observations_are_diagnostics": True,
                "package_presence_is_not_capability": True,
                "simulated_zkp_authority": False,
                "completion_authority": False,
            }
            report = ProgramAnalysisCapabilityReport(
                generated_at=datetime.fromtimestamp(
                    self._wall_clock(), tz=timezone.utc
                )
                .isoformat()
                .replace("+00:00", "Z"),
                duration_seconds=max(0.0, finished - self._started),
                probe_count=self._probe_count,
                overall_status=overall,
                families=family_reports,
                diagnostics=diagnostics,
                matrix=declare_program_analysis_capability_matrix(
                    probed=True, imported=self._any_import_succeeded
                ),
            )
            encoded = _canonical_json_bytes(report.to_dict(), name="report")
            if len(encoded) > self.config.max_report_bytes:
                raise ProgramAnalysisCapabilityError(
                    f"capability report exceeds {self.config.max_report_bytes} bytes"
                )
            self._cached = (finished, report)
            return report

    # -- budget helpers ---------------------------------------------------

    def _budget_reason(self) -> CapabilityReasonCode | None:
        if self._probe_count >= self.config.max_checks:
            return CapabilityReasonCode.PROBE_LIMIT
        if self._monotonic() >= self._deadline:
            return CapabilityReasonCode.PROBE_TIMEOUT
        self._probe_count += 1
        return None

    def _bounded_call(
        self,
        function: Callable[..., Any],
        *args: Any,
        timeout_seconds: float | None = None,
    ) -> tuple[bool, Any, BaseException | None]:
        remaining = self._deadline - self._monotonic()
        if timeout_seconds is not None:
            remaining = min(remaining, float(timeout_seconds))
        if remaining <= 0:
            return False, None, TimeoutError("capability probe time budget exhausted")

        outcome: list[tuple[bool, Any, BaseException | None]] = []

        def invoke() -> None:
            try:
                outcome.append((True, function(*args), None))
            except BaseException as exc:  # isolate third-party hooks
                outcome.append((True, None, exc))

        worker = threading.Thread(
            target=invoke,
            name="program-analysis-capability-probe",
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

    def _discover_package(
        self, module: str, *, required: bool = True
    ) -> CapabilitySurface:
        limited = self._budget_reason()
        if limited is CapabilityReasonCode.PROBE_TIMEOUT:
            return CapabilitySurface(
                surface_id=f"package:{module}",
                status=CapabilityProbeStatus.TIMED_OUT,
                reason_code=limited,
                reason=f"package discovery for {module!r} timed out before inspection",
                required=required,
                metadata={"module": module, "probe_limited": True},
            )
        if limited is CapabilityReasonCode.PROBE_LIMIT:
            return CapabilitySurface(
                surface_id=f"package:{module}",
                status=CapabilityProbeStatus.UNAVAILABLE,
                reason_code=limited,
                reason=f"package discovery for {module!r} skipped; probe check limit exhausted",
                required=required,
                metadata={"module": module, "probe_limited": True},
            )
        completed, spec, error = self._bounded_call(self._find_spec, module)
        if not completed:
            return CapabilitySurface(
                surface_id=f"package:{module}",
                status=CapabilityProbeStatus.TIMED_OUT,
                reason_code=CapabilityReasonCode.PROBE_TIMEOUT,
                reason=f"package discovery for {module!r} timed out safely: {error}",
                required=required,
                metadata={"module": module, "probe_limited": True},
            )
        if error is not None:
            return CapabilitySurface(
                surface_id=f"package:{module}",
                status=CapabilityProbeStatus.UNAVAILABLE,
                reason_code=CapabilityReasonCode.IMPORT_FAILED,
                reason=(
                    f"package discovery for {module!r} failed safely: "
                    f"{type(error).__name__}: {error}"
                ),
                required=required,
                metadata={"module": module},
            )
        if spec is None:
            return CapabilitySurface(
                surface_id=f"package:{module}",
                status=CapabilityProbeStatus.UNAVAILABLE,
                reason_code=CapabilityReasonCode.PACKAGE_MISSING,
                reason=f"Python package {module!r} is not discoverable",
                required=required,
                metadata={"module": module},
            )
        origin = getattr(spec, "origin", None)
        return CapabilitySurface(
            surface_id=f"package:{module}",
            status=CapabilityProbeStatus.AVAILABLE,
            reason_code=CapabilityReasonCode.PACKAGE_PRESENCE_ONLY,
            reason=(
                f"package {module!r} is discoverable; presence alone is not a "
                "capability proof"
            ),
            authority=CapabilityAuthority.DIAGNOSTIC,
            required=required,
            location=str(origin) if origin else None,
            metadata={
                "module": module,
                "package_presence_is_not_capability": True,
            },
        )

    def _import_module(self, module: str) -> tuple[Any | None, CapabilitySurface]:
        presence = self._discover_package(module, required=True)
        if presence.status is not CapabilityProbeStatus.AVAILABLE:
            return None, presence
        limited = self._budget_reason()
        if limited is not None:
            status = (
                CapabilityProbeStatus.TIMED_OUT
                if limited is CapabilityReasonCode.PROBE_TIMEOUT
                else CapabilityProbeStatus.UNAVAILABLE
            )
            return None, CapabilitySurface(
                surface_id=f"import:{module}",
                status=status,
                reason_code=limited,
                reason=f"import of {module!r} skipped due to probe budget",
                required=True,
                metadata={"module": module, "probe_limited": True},
            )
        self._import_attempted = True
        completed, module_obj, error = self._bounded_call(self._importer, module)
        if not completed:
            return None, CapabilitySurface(
                surface_id=f"import:{module}",
                status=CapabilityProbeStatus.TIMED_OUT,
                reason_code=CapabilityReasonCode.PROBE_TIMEOUT,
                reason=f"import of {module!r} timed out safely: {error}",
                required=True,
                metadata={"module": module, "probe_limited": True},
            )
        if error is not None:
            return None, CapabilitySurface(
                surface_id=f"import:{module}",
                status=CapabilityProbeStatus.UNAVAILABLE,
                reason_code=CapabilityReasonCode.IMPORT_FAILED,
                reason=(
                    f"import of {module!r} failed: "
                    f"{type(error).__name__}: {error}"
                ),
                required=True,
                metadata={"module": module},
            )
        self._any_import_succeeded = True
        return module_obj, CapabilitySurface(
            surface_id=f"import:{module}",
            status=CapabilityProbeStatus.AVAILABLE,
            reason_code=CapabilityReasonCode.AVAILABLE,
            reason=f"module {module!r} imported for signature inspection",
            authority=CapabilityAuthority.DIAGNOSTIC,
            required=True,
            location=getattr(module_obj, "__file__", None),
            metadata={"module": module},
        )

    def _executable_surface(
        self,
        name: str,
        candidates: Sequence[str],
        *,
        env_names: Sequence[str] = (),
        required: bool = False,
    ) -> CapabilitySurface:
        limited = self._budget_reason()
        if limited is not None:
            status = (
                CapabilityProbeStatus.TIMED_OUT
                if limited is CapabilityReasonCode.PROBE_TIMEOUT
                else CapabilityProbeStatus.UNAVAILABLE
            )
            return CapabilitySurface(
                surface_id=f"executable:{name}",
                status=status,
                reason_code=limited,
                reason=f"executable lookup for {name!r} skipped due to probe budget",
                required=required,
                metadata={"probe_limited": True, "candidates": list(candidates)},
            )
        for env_name in env_names:
            configured = str(self._environ.get(env_name, "") or "").strip()
            if configured:
                path = Path(configured)
                if path.is_file() and os.access(path, os.X_OK):
                    return CapabilitySurface(
                        surface_id=f"executable:{name}",
                        status=CapabilityProbeStatus.AVAILABLE,
                        reason_code=CapabilityReasonCode.AVAILABLE,
                        reason=f"{name} executable configured via {env_name}",
                        authority=CapabilityAuthority.SOLVER_CANDIDATE,
                        required=required,
                        location=str(path),
                        metadata={"source": "env", "env": env_name},
                    )
        for candidate in candidates:
            completed, path, error = self._bounded_call(self._which, candidate)
            if not completed:
                return CapabilitySurface(
                    surface_id=f"executable:{name}",
                    status=CapabilityProbeStatus.TIMED_OUT,
                    reason_code=CapabilityReasonCode.PROBE_TIMEOUT,
                    reason=f"executable lookup for {name!r} timed out: {error}",
                    required=required,
                    metadata={"probe_limited": True},
                )
            if error is None and path:
                return CapabilitySurface(
                    surface_id=f"executable:{name}",
                    status=CapabilityProbeStatus.AVAILABLE,
                    reason_code=CapabilityReasonCode.AVAILABLE,
                    reason=f"{name} executable discovered at {path}",
                    authority=CapabilityAuthority.SOLVER_CANDIDATE,
                    required=required,
                    location=str(path),
                    metadata={"source": "path", "candidate": candidate},
                )
        return CapabilitySurface(
            surface_id=f"executable:{name}",
            status=CapabilityProbeStatus.UNAVAILABLE,
            reason_code=CapabilityReasonCode.EXECUTABLE_MISSING,
            reason=f"{name} executable is not installed or configured",
            required=required,
            metadata={"candidates": list(candidates)},
        )

    def _surface_from_callables(
        self,
        surface_id: str,
        owner: Any,
        required: Sequence[tuple[str, Sequence[str]]],
        *,
        authority: CapabilityAuthority,
        schema_attr: str | None = None,
        expected_schema_prefix: str | None = None,
    ) -> CapabilitySurface:
        missing: list[str] = []
        incompatible: list[str] = []
        for name, params in required:
            ok, detail = _callable_compatible(owner, name, required_params=params)
            if not ok:
                if "signature is incompatible" in detail:
                    incompatible.append(detail)
                else:
                    missing.append(detail)
        if missing:
            return CapabilitySurface(
                surface_id=surface_id,
                status=CapabilityProbeStatus.UNAVAILABLE,
                reason_code=CapabilityReasonCode.CALLABLE_MISSING,
                reason="; ".join(missing),
                authority=CapabilityAuthority.NONE,
                required=True,
            )
        if incompatible:
            return CapabilitySurface(
                surface_id=surface_id,
                status=CapabilityProbeStatus.INCOMPATIBLE,
                reason_code=CapabilityReasonCode.SIGNATURE_INCOMPATIBLE,
                reason="; ".join(incompatible),
                authority=CapabilityAuthority.NONE,
                required=True,
            )
        if schema_attr is not None:
            schema_value = getattr(owner, schema_attr, None)
            if not isinstance(schema_value, str) or not schema_value.strip():
                return CapabilitySurface(
                    surface_id=surface_id,
                    status=CapabilityProbeStatus.INCOMPATIBLE,
                    reason_code=CapabilityReasonCode.SCHEMA_INCOMPATIBLE,
                    reason=f"{schema_attr} is missing or empty",
                    required=True,
                )
            if expected_schema_prefix and not schema_value.startswith(
                expected_schema_prefix
            ):
                return CapabilitySurface(
                    surface_id=surface_id,
                    status=CapabilityProbeStatus.INCOMPATIBLE,
                    reason_code=CapabilityReasonCode.SCHEMA_INCOMPATIBLE,
                    reason=(
                        f"{schema_attr}={schema_value!r} is incompatible with "
                        f"prefix {expected_schema_prefix!r}"
                    ),
                    required=True,
                )
        return CapabilitySurface(
            surface_id=surface_id,
            status=CapabilityProbeStatus.AVAILABLE,
            reason_code=CapabilityReasonCode.AVAILABLE,
            reason=f"{surface_id} callables and signatures are compatible",
            authority=authority,
            required=True,
            metadata={
                "callables": [name for name, _ in required],
                **(
                    {schema_attr: getattr(owner, schema_attr)}
                    if schema_attr is not None
                    else {}
                ),
            },
        )

    def _family(
        self,
        family: CapabilityFamily,
        surfaces: Sequence[CapabilitySurface],
        *,
        authority: CapabilityAuthority,
        available_when: Callable[[Sequence[CapabilitySurface]], bool] | None = None,
        partial_when: Callable[[Sequence[CapabilitySurface]], bool] | None = None,
    ) -> CapabilityFamilyReport:
        surfaces_t = tuple(surfaces)
        if any(s.status is CapabilityProbeStatus.TIMED_OUT for s in surfaces_t):
            status = CapabilityProbeStatus.TIMED_OUT
            reason_code = CapabilityReasonCode.PROBE_TIMEOUT
            reason = f"{family.value} probe timed out under the configured budget"
            authority_out = CapabilityAuthority.NONE
        elif any(
            s.status is CapabilityProbeStatus.INCOMPATIBLE for s in surfaces_t
        ):
            status = CapabilityProbeStatus.INCOMPATIBLE
            reason_code = CapabilityReasonCode.SIGNATURE_INCOMPATIBLE
            reason = f"{family.value} has incompatible callables or schemas"
            authority_out = CapabilityAuthority.NONE
        elif any(
            s.reason_code is CapabilityReasonCode.UNBOUNDED_OUTPUT for s in surfaces_t
        ):
            status = CapabilityProbeStatus.REJECTED
            reason_code = CapabilityReasonCode.UNBOUNDED_OUTPUT
            reason = f"{family.value} rejected unbounded probe output"
            authority_out = CapabilityAuthority.NONE
        elif any(
            s.reason_code
            in {
                CapabilityReasonCode.SIMULATED_ZKP_AUTHORITY_REJECTED,
                CapabilityReasonCode.FALLBACK_ZKP_REJECTED,
                CapabilityReasonCode.PSEUDO_CID_ACCEPTED,
            }
            for s in surfaces_t
        ):
            rejected = next(
                s
                for s in surfaces_t
                if s.reason_code
                in {
                    CapabilityReasonCode.SIMULATED_ZKP_AUTHORITY_REJECTED,
                    CapabilityReasonCode.FALLBACK_ZKP_REJECTED,
                    CapabilityReasonCode.PSEUDO_CID_ACCEPTED,
                }
            )
            status = CapabilityProbeStatus.REJECTED
            reason_code = rejected.reason_code
            reason = rejected.reason
            authority_out = CapabilityAuthority.NONE
        elif available_when and available_when(surfaces_t):
            if partial_when and partial_when(surfaces_t):
                status = CapabilityProbeStatus.PARTIAL
                reason_code = CapabilityReasonCode.PARTIAL
                reason = f"{family.value} is only partially available"
            elif any(s.status is CapabilityProbeStatus.DEGRADED for s in surfaces_t):
                status = CapabilityProbeStatus.DEGRADED
                reason_code = CapabilityReasonCode.DEGRADED
                reason = f"{family.value} is degraded"
            elif any(s.status is CapabilityProbeStatus.SIMULATED for s in surfaces_t):
                status = CapabilityProbeStatus.SIMULATED
                reason_code = CapabilityReasonCode.SIMULATED_ZKP_ONLY
                reason = f"{family.value} exposes only simulated surfaces"
                authority_out = CapabilityAuthority.ZKP_DIAGNOSTIC
            else:
                status = CapabilityProbeStatus.AVAILABLE
                reason_code = CapabilityReasonCode.AVAILABLE
                reason = f"{family.value} surfaces are currently available"
            if status is not CapabilityProbeStatus.SIMULATED:
                authority_out = authority
            else:
                authority_out = CapabilityAuthority.ZKP_DIAGNOSTIC
        elif any(s.status is CapabilityProbeStatus.UNHEALTHY for s in surfaces_t):
            status = CapabilityProbeStatus.UNHEALTHY
            reason_code = CapabilityReasonCode.UNHEALTHY_PROVIDER
            reason = f"{family.value} provider is unhealthy"
            authority_out = CapabilityAuthority.NONE
        else:
            status = CapabilityProbeStatus.UNAVAILABLE
            reason_code = CapabilityReasonCode.PACKAGE_MISSING
            reason = f"{family.value} is unavailable in the current environment"
            authority_out = CapabilityAuthority.NONE
        return CapabilityFamilyReport(
            family=family,
            status=status,
            reason_code=reason_code,
            reason=reason,
            authority=authority_out,
            surfaces=surfaces_t,
            metadata={"family": family.value},
        )

    def _overall_status(
        self, families: Sequence[CapabilityFamilyReport]
    ) -> CapabilityProbeStatus:
        statuses = {item.status for item in families}
        if CapabilityProbeStatus.TIMED_OUT in statuses:
            return CapabilityProbeStatus.TIMED_OUT
        if CapabilityProbeStatus.INCOMPATIBLE in statuses:
            return CapabilityProbeStatus.INCOMPATIBLE
        if CapabilityProbeStatus.REJECTED in statuses:
            return CapabilityProbeStatus.REJECTED
        if CapabilityProbeStatus.UNHEALTHY in statuses:
            return CapabilityProbeStatus.UNHEALTHY
        if statuses <= {CapabilityProbeStatus.AVAILABLE}:
            return CapabilityProbeStatus.AVAILABLE
        if CapabilityProbeStatus.AVAILABLE in statuses or CapabilityProbeStatus.PARTIAL in statuses:
            return CapabilityProbeStatus.PARTIAL
        if CapabilityProbeStatus.DEGRADED in statuses or CapabilityProbeStatus.SIMULATED in statuses:
            return CapabilityProbeStatus.DEGRADED
        return CapabilityProbeStatus.UNAVAILABLE

    # -- family probes ----------------------------------------------------

    def _probe_strict_cid(self) -> CapabilityFamilyReport:
        root = self.config.optional_root
        module_name = f"{root}.utils.cid_utils"
        module, import_surface = self._import_module(module_name)
        surfaces: list[CapabilitySurface] = [import_surface]
        if module is None:
            return self._family(
                CapabilityFamily.STRICT_CID,
                surfaces,
                authority=CapabilityAuthority.STRICT_IDENTITY,
                available_when=lambda _: False,
            )

        # Required callables with parameter names that must be present.
        required = [
            ("canonical_dag_json_bytes", ("obj",)),
            ("cid_for_bytes", ("data",)),
            ("cid_for_dag_json", ("obj",)),
            ("cid_for_obj", ("obj",)),
            ("validate_cid", ("value",)),
        ]
        callables = self._surface_from_callables(
            "strict_cid.callables",
            module,
            required,
            authority=CapabilityAuthority.STRICT_IDENTITY,
        )
        surfaces.append(callables)

        multi = self._discover_package("multiformats", required=True)
        # Presence of multiformats is diagnostic; cid_utils must actually work.
        if multi.status is CapabilityProbeStatus.AVAILABLE:
            multi = CapabilitySurface(
                surface_id=multi.surface_id,
                status=CapabilityProbeStatus.AVAILABLE,
                reason_code=CapabilityReasonCode.CURRENT_DIAGNOSTIC,
                reason=(
                    "multiformats is discoverable; strict CID helpers still "
                    "require a functional canary"
                ),
                authority=CapabilityAuthority.DIAGNOSTIC,
                required=True,
                location=multi.location,
                metadata=dict(multi.metadata),
            )
        surfaces.append(multi)

        if (
            self.config.run_strict_cid_canary
            and callables.status is CapabilityProbeStatus.AVAILABLE
        ):
            canary = self._strict_cid_canary(module)
            surfaces.append(canary)

        def available(items: Sequence[CapabilitySurface]) -> bool:
            by_id = {item.surface_id: item for item in items}
            call = by_id.get("strict_cid.callables")
            canary = by_id.get("strict_cid.canary")
            if call is None or call.status is not CapabilityProbeStatus.AVAILABLE:
                return False
            if canary is not None:
                return canary.status is CapabilityProbeStatus.AVAILABLE
            return True

        return self._family(
            CapabilityFamily.STRICT_CID,
            surfaces,
            authority=CapabilityAuthority.STRICT_IDENTITY,
            available_when=available,
        )

    def _strict_cid_canary(self, module: Any) -> CapabilitySurface:
        limited = self._budget_reason()
        if limited is not None:
            status = (
                CapabilityProbeStatus.TIMED_OUT
                if limited is CapabilityReasonCode.PROBE_TIMEOUT
                else CapabilityProbeStatus.UNAVAILABLE
            )
            return CapabilitySurface(
                surface_id="strict_cid.canary",
                status=status,
                reason_code=limited,
                reason="strict CID canary skipped due to probe budget",
                required=True,
                metadata={"probe_limited": True},
            )

        def run() -> dict[str, Any]:
            payload = {"program_analysis": "cid-canary", "n": 1}
            cid = module.cid_for_dag_json(payload)
            validated = module.validate_cid(cid)
            if validated != cid:
                raise ProgramAnalysisCapabilityError(
                    "validate_cid did not round-trip cid_for_dag_json"
                )
            accepted_pseudo: list[str] = []
            for sample in _PSEUDO_CID_SAMPLES:
                if sample == "":
                    # Empty string is invalid input; validators may raise TypeError
                    # or ValueError. Either is acceptable rejection.
                    try:
                        module.validate_cid(sample)
                    except Exception:
                        continue
                    accepted_pseudo.append("<empty>")
                    continue
                try:
                    module.validate_cid(sample)
                except Exception:
                    continue
                accepted_pseudo.append(sample)
            return {
                "cid": cid,
                "accepted_pseudo": accepted_pseudo,
                "cid_bytes": len(str(cid).encode("utf-8")),
            }

        completed, result, error = self._bounded_call(run)
        if not completed:
            return CapabilitySurface(
                surface_id="strict_cid.canary",
                status=CapabilityProbeStatus.TIMED_OUT,
                reason_code=CapabilityReasonCode.PROBE_TIMEOUT,
                reason=f"strict CID canary timed out: {error}",
                required=True,
                metadata={"probe_limited": True},
            )
        if error is not None:
            return CapabilitySurface(
                surface_id="strict_cid.canary",
                status=CapabilityProbeStatus.UNHEALTHY,
                reason_code=CapabilityReasonCode.UNHEALTHY_PROVIDER,
                reason=(
                    f"strict CID canary failed: {type(error).__name__}: {error}"
                ),
                required=True,
            )
        assert isinstance(result, Mapping)
        if result.get("accepted_pseudo"):
            return CapabilitySurface(
                surface_id="strict_cid.canary",
                status=CapabilityProbeStatus.REJECTED,
                reason_code=CapabilityReasonCode.PSEUDO_CID_ACCEPTED,
                reason=(
                    "strict CID helpers accepted legacy or pseudo CID samples: "
                    + ", ".join(result["accepted_pseudo"])
                ),
                required=True,
                metadata={"accepted_pseudo": list(result["accepted_pseudo"])},
            )
        cid = str(result.get("cid") or "")
        if not cid or len(cid.encode("utf-8")) > 256:
            return CapabilitySurface(
                surface_id="strict_cid.canary",
                status=CapabilityProbeStatus.REJECTED,
                reason_code=CapabilityReasonCode.UNBOUNDED_OUTPUT,
                reason="strict CID canary returned an empty or oversized CID",
                required=True,
            )
        return CapabilitySurface(
            surface_id="strict_cid.canary",
            status=CapabilityProbeStatus.AVAILABLE,
            reason_code=CapabilityReasonCode.AVAILABLE,
            reason="strict CID helpers mint and validate CIDv1 DAG-JSON identities",
            authority=CapabilityAuthority.STRICT_IDENTITY,
            required=True,
            metadata={
                "sample_cid_prefix": cid[:16],
                "rejected_pseudo_cid_samples": len(_PSEUDO_CID_SAMPLES),
            },
        )

    def _probe_graphrag(self) -> CapabilityFamilyReport:
        root = self.config.optional_root
        candidates = (
            f"{root}.search.graphrag_integration.graphrag_integration",
            f"{root}.search.graph_query",
            f"{root}.knowledge_graphs",
        )
        surfaces: list[CapabilitySurface] = []
        imported_any = False
        query_surface: CapabilitySurface | None = None

        for module_name in candidates:
            module, import_surface = self._import_module(module_name)
            surfaces.append(import_surface)
            if module is None:
                continue
            imported_any = True
            # Prefer a query-capable class with a bounded signature.
            for attr_name in (
                "GraphRAGQueryEngine",
                "GraphRAGIntegration",
                "GraphQueryEngine",
                "query",
            ):
                target = getattr(module, attr_name, None)
                if target is None:
                    continue
                if attr_name == "query" and callable(target):
                    ok, detail = _callable_compatible(
                        module, "query", required_params=()
                    )
                    query_surface = CapabilitySurface(
                        surface_id=f"graphrag.query:{module_name}",
                        status=(
                            CapabilityProbeStatus.AVAILABLE
                            if ok
                            else CapabilityProbeStatus.INCOMPATIBLE
                        ),
                        reason_code=(
                            CapabilityReasonCode.AVAILABLE
                            if ok
                            else CapabilityReasonCode.SIGNATURE_INCOMPATIBLE
                        ),
                        reason=detail if ok else detail,
                        authority=(
                            CapabilityAuthority.BOUNDED_QUERY
                            if ok
                            else CapabilityAuthority.NONE
                        ),
                        required=False,
                        metadata={"module": module_name, "symbol": "query"},
                    )
                    surfaces.append(query_surface)
                    break
                if inspect.isclass(target):
                    method = getattr(target, "query", None)
                    if method is None:
                        # GraphRAGIntegration may expose other retrieval APIs.
                        method = getattr(target, "explain_trace", None)
                        method_name = "explain_trace"
                    else:
                        method_name = "query"
                    if method is None:
                        continue
                    try:
                        params = _signature_params(method)
                    except ProgramAnalysisCapabilityError as exc:
                        surfaces.append(
                            CapabilitySurface(
                                surface_id=f"graphrag.{attr_name}",
                                status=CapabilityProbeStatus.INCOMPATIBLE,
                                reason_code=CapabilityReasonCode.SIGNATURE_INCOMPATIBLE,
                                reason=str(exc),
                                required=False,
                            )
                        )
                        continue
                    # Reject clearly unbounded APIs that lack any limit knobs when
                    # the method is named query.
                    has_bounds = bool(
                        params
                        & {
                            "top_k",
                            "max_results",
                            "limit",
                            "max_nodes_visited",
                            "max_edges_traversed",
                            "max_graph_hops",
                        }
                    )
                    if method_name == "query" and not has_bounds:
                        surfaces.append(
                            CapabilitySurface(
                                surface_id=f"graphrag.{attr_name}.query",
                                status=CapabilityProbeStatus.REJECTED,
                                reason_code=CapabilityReasonCode.UNBOUNDED_OUTPUT,
                                reason=(
                                    f"{attr_name}.query lacks explicit bound "
                                    "parameters (top_k/limit/max_*)"
                                ),
                                required=False,
                                metadata={"parameters": sorted(params)},
                            )
                        )
                        continue
                    query_surface = CapabilitySurface(
                        surface_id=f"graphrag.{attr_name}.{method_name}",
                        status=CapabilityProbeStatus.AVAILABLE,
                        reason_code=CapabilityReasonCode.AVAILABLE,
                        reason=(
                            f"{attr_name}.{method_name} is present with "
                            f"{'bounded' if has_bounds or method_name != 'query' else 'diagnostic'} parameters"
                        ),
                        authority=CapabilityAuthority.BOUNDED_QUERY,
                        required=False,
                        metadata={
                            "module": module_name,
                            "symbol": attr_name,
                            "method": method_name,
                            "bounded": has_bounds or method_name != "query",
                            "parameters": sorted(params)[:32],
                        },
                    )
                    surfaces.append(query_surface)
                    break
            if query_surface is not None and query_surface.status is CapabilityProbeStatus.AVAILABLE:
                break

        if not surfaces:
            surfaces.append(
                CapabilitySurface(
                    surface_id="graphrag.missing",
                    status=CapabilityProbeStatus.UNAVAILABLE,
                    reason_code=CapabilityReasonCode.PACKAGE_MISSING,
                    reason="no GraphRAG or knowledge-graph modules are discoverable",
                    required=True,
                )
            )

        def available(items: Sequence[CapabilitySurface]) -> bool:
            return any(
                item.authority is CapabilityAuthority.BOUNDED_QUERY
                and item.status is CapabilityProbeStatus.AVAILABLE
                for item in items
            )

        def partial(items: Sequence[CapabilitySurface]) -> bool:
            imported = any(
                item.surface_id.startswith("import:")
                and item.status is CapabilityProbeStatus.AVAILABLE
                for item in items
            )
            query_ok = available(items)
            return imported and not query_ok

        # If modules imported but no usable query API, mark degraded/partial.
        if imported_any and query_surface is None:
            surfaces.append(
                CapabilitySurface(
                    surface_id="graphrag.query_api",
                    status=CapabilityProbeStatus.DEGRADED,
                    reason_code=CapabilityReasonCode.DEGRADED,
                    reason=(
                        "GraphRAG/knowledge-graph packages imported but no "
                        "compatible query surface was found"
                    ),
                    required=True,
                )
            )

        return self._family(
            CapabilityFamily.GRAPHRAG,
            surfaces,
            authority=CapabilityAuthority.BOUNDED_QUERY,
            available_when=lambda items: available(items) or partial(items),
            partial_when=partial,
        )

    def _probe_ir_core(self) -> CapabilityFamilyReport:
        root = self.config.optional_root
        claims_name = f"{root}.logic.ir_core.claims"
        protocols_name = f"{root}.logic.ir_core.protocols"
        surfaces: list[CapabilitySurface] = []

        claims_mod, claims_import = self._import_module(claims_name)
        surfaces.append(claims_import)
        if claims_mod is not None:
            missing = [
                name
                for name in _IR_CORE_CLAIM_EXPORTS
                if not hasattr(claims_mod, name)
            ]
            if missing:
                surfaces.append(
                    CapabilitySurface(
                        surface_id="ir_core.claims.exports",
                        status=CapabilityProbeStatus.INCOMPATIBLE,
                        reason_code=CapabilityReasonCode.SCHEMA_INCOMPATIBLE,
                        reason=f"ir_core.claims missing exports: {missing}",
                        required=True,
                    )
                )
            else:
                schema = getattr(claims_mod, "IR_CLAIM_SCHEMA_VERSION", "")
                if not isinstance(schema, str) or "claim" not in schema:
                    surfaces.append(
                        CapabilitySurface(
                            surface_id="ir_core.claims.schema",
                            status=CapabilityProbeStatus.INCOMPATIBLE,
                            reason_code=CapabilityReasonCode.SCHEMA_INCOMPATIBLE,
                            reason=f"unexpected IR claim schema version: {schema!r}",
                            required=True,
                        )
                    )
                else:
                    surfaces.append(
                        CapabilitySurface(
                            surface_id="ir_core.claims",
                            status=CapabilityProbeStatus.AVAILABLE,
                            reason_code=CapabilityReasonCode.AVAILABLE,
                            reason="ir_core claims exports and schema are compatible",
                            authority=CapabilityAuthority.IR_DECLARATION,
                            required=True,
                            metadata={
                                "schema_version": schema,
                                "exports": list(_IR_CORE_CLAIM_EXPORTS),
                            },
                        )
                    )

        protocols_mod, protocols_import = self._import_module(protocols_name)
        surfaces.append(protocols_import)
        if protocols_mod is not None:
            missing = [
                name
                for name in _IR_CORE_PROTOCOL_EXPORTS
                if not hasattr(protocols_mod, name)
            ]
            if missing:
                surfaces.append(
                    CapabilitySurface(
                        surface_id="ir_core.protocols.exports",
                        status=CapabilityProbeStatus.INCOMPATIBLE,
                        reason_code=CapabilityReasonCode.SCHEMA_INCOMPATIBLE,
                        reason=f"ir_core.protocols missing exports: {missing}",
                        required=True,
                    )
                )
            else:
                authority_kind = getattr(protocols_mod, "AuthorityKind", None)
                # Authority kinds must remain non-hierarchical / closed.
                try:
                    values = {
                        str(getattr(item, "value", item))
                        for item in authority_kind
                    }
                except Exception:
                    values = set()
                if "theorem_proof" not in values:
                    surfaces.append(
                        CapabilitySurface(
                            surface_id="ir_core.protocols.authority",
                            status=CapabilityProbeStatus.INCOMPATIBLE,
                            reason_code=CapabilityReasonCode.SCHEMA_INCOMPATIBLE,
                            reason="AuthorityKind lacks theorem_proof",
                            required=True,
                        )
                    )
                else:
                    surfaces.append(
                        CapabilitySurface(
                            surface_id="ir_core.protocols",
                            status=CapabilityProbeStatus.AVAILABLE,
                            reason_code=CapabilityReasonCode.AVAILABLE,
                            reason="ir_core protocols exports are compatible",
                            authority=CapabilityAuthority.IR_DECLARATION,
                            required=True,
                            metadata={
                                "authority_kinds": sorted(values),
                                "exports": list(_IR_CORE_PROTOCOL_EXPORTS),
                            },
                        )
                    )

        def available(items: Sequence[CapabilitySurface]) -> bool:
            ids = {
                item.surface_id
                for item in items
                if item.status is CapabilityProbeStatus.AVAILABLE
            }
            return "ir_core.claims" in ids and "ir_core.protocols" in ids

        def partial(items: Sequence[CapabilitySurface]) -> bool:
            ids = {
                item.surface_id
                for item in items
                if item.status is CapabilityProbeStatus.AVAILABLE
            }
            return bool(ids & {"ir_core.claims", "ir_core.protocols"}) and not (
                "ir_core.claims" in ids and "ir_core.protocols" in ids
            )

        return self._family(
            CapabilityFamily.IR_CORE,
            surfaces,
            authority=CapabilityAuthority.IR_DECLARATION,
            available_when=lambda items: available(items) or partial(items),
            partial_when=partial,
        )

    def _probe_solvers(self) -> CapabilityFamilyReport:
        root = self.config.optional_root
        surfaces: list[CapabilitySurface] = []

        # Python compiler/bindings
        for binding, module_name, env in (
            ("cvc5", "cvc5", ("CVC5_BINARY", "IPFS_DATASETS_CVC5_BINARY")),
            ("z3", "z3", ("Z3_BINARY", "IPFS_DATASETS_Z3_BINARY")),
        ):
            package = self._discover_package(module_name, required=False)
            if package.status is CapabilityProbeStatus.AVAILABLE:
                # Reject package-presence-only: require import + a usable API.
                mod, import_surface = self._import_module(module_name)
                surfaces.append(import_surface)
                if mod is not None:
                    # cvc5 exposes Solver; z3 exposes Solver / parse_smt2_string.
                    marker = None
                    for attr in ("Solver", "solver", "parse_smt2_string", "Context"):
                        if hasattr(mod, attr):
                            marker = attr
                            break
                    if marker is None:
                        surfaces.append(
                            CapabilitySurface(
                                surface_id=f"solver.binding.{binding}",
                                status=CapabilityProbeStatus.INCOMPATIBLE,
                                reason_code=CapabilityReasonCode.SIGNATURE_INCOMPATIBLE,
                                reason=(
                                    f"{binding} Python package imported but exposes "
                                    "no recognized solver API"
                                ),
                                required=False,
                                metadata={"module": module_name},
                            )
                        )
                    else:
                        surfaces.append(
                            CapabilitySurface(
                                surface_id=f"solver.binding.{binding}",
                                status=CapabilityProbeStatus.AVAILABLE,
                                reason_code=CapabilityReasonCode.CURRENT_DIAGNOSTIC,
                                reason=(
                                    f"{binding} Python binding is importable and "
                                    f"exposes {marker}"
                                ),
                                authority=CapabilityAuthority.SOLVER_CANDIDATE,
                                required=False,
                                version=str(getattr(mod, "__version__", "") or "")
                                or None,
                                metadata={
                                    "module": module_name,
                                    "api_marker": marker,
                                    "observation": True,
                                },
                            )
                        )
            else:
                package = CapabilitySurface(
                    surface_id=f"solver.binding.{binding}",
                    status=package.status,
                    reason_code=(
                        CapabilityReasonCode.BINDING_MISSING
                        if package.reason_code
                        is CapabilityReasonCode.PACKAGE_MISSING
                        else package.reason_code
                    ),
                    reason=f"{binding} Python binding is not available",
                    required=False,
                    metadata={"module": module_name},
                )
                surfaces.append(package)

            exe = self._executable_surface(
                binding,
                (binding,),
                env_names=env,
                required=False,
            )
            # Rewrite surface id for stability.
            surfaces.append(
                CapabilitySurface(
                    surface_id=f"solver.executable.{binding}",
                    status=exe.status,
                    reason_code=(
                        CapabilityReasonCode.EXECUTABLE_MISSING
                        if exe.reason_code
                        is CapabilityReasonCode.EXECUTABLE_MISSING
                        else exe.reason_code
                    ),
                    reason=exe.reason,
                    authority=exe.authority,
                    required=False,
                    location=exe.location,
                    metadata={**dict(exe.metadata), "solver": binding},
                )
            )

        # Compiler bridges inside ipfs_datasets_py (optional diagnostics).
        for bridge_name, module_name in (
            (
                "cvc5_bridge",
                f"{root}.logic.external_provers.smt.cvc5_prover_bridge",
            ),
            (
                "z3_bridge",
                f"{root}.logic.external_provers.smt.z3_prover_bridge",
            ),
        ):
            presence = self._discover_package(module_name, required=False)
            if presence.status is CapabilityProbeStatus.AVAILABLE:
                surfaces.append(
                    CapabilitySurface(
                        surface_id=f"solver.compiler.{bridge_name}",
                        status=CapabilityProbeStatus.AVAILABLE,
                        reason_code=CapabilityReasonCode.CURRENT_DIAGNOSTIC,
                        reason=(
                            f"{bridge_name} module is discoverable; runtime "
                            "availability still depends on binding/executable probes"
                        ),
                        authority=CapabilityAuthority.DIAGNOSTIC,
                        required=False,
                        location=presence.location,
                        metadata={
                            "module": module_name,
                            "package_presence_is_not_capability": True,
                        },
                    )
                )
            else:
                surfaces.append(
                    CapabilitySurface(
                        surface_id=f"solver.compiler.{bridge_name}",
                        status=presence.status,
                        reason_code=(
                            CapabilityReasonCode.COMPILER_MISSING
                            if presence.reason_code
                            is CapabilityReasonCode.PACKAGE_MISSING
                            else presence.reason_code
                        ),
                        reason=f"{bridge_name} is not discoverable",
                        required=False,
                        metadata={"module": module_name},
                    )
                )

        def available(items: Sequence[CapabilitySurface]) -> bool:
            return any(
                item.surface_id.startswith("solver.")
                and item.status is CapabilityProbeStatus.AVAILABLE
                and item.authority
                in {
                    CapabilityAuthority.SOLVER_CANDIDATE,
                    CapabilityAuthority.DIAGNOSTIC,
                }
                for item in items
            )

        def partial(items: Sequence[CapabilitySurface]) -> bool:
            available_ids = {
                item.surface_id
                for item in items
                if item.status is CapabilityProbeStatus.AVAILABLE
            }
            # Partial when some but not all of cvc5/z3 binding+executable exist.
            cvc5_ok = any("cvc5" in sid for sid in available_ids)
            z3_ok = any("z3" in sid and "cvc5" not in sid for sid in available_ids)
            return cvc5_ok != z3_ok or (
                bool(available_ids)
                and not (
                    any("binding.cvc5" in sid for sid in available_ids)
                    and any("binding.z3" in sid for sid in available_ids)
                    and any("executable.cvc5" in sid for sid in available_ids)
                    and any("executable.z3" in sid for sid in available_ids)
                )
            )

        return self._family(
            CapabilityFamily.SOLVERS,
            surfaces,
            authority=CapabilityAuthority.SOLVER_CANDIDATE,
            available_when=available,
            partial_when=partial,
        )

    def _probe_ast_producers(self) -> CapabilityFamilyReport:
        root = self.config.optional_root
        module_name = (
            f"{root}.logic.security_models.crypto_exchange.extractors."
            "python_ast_extractor"
        )
        surfaces: list[CapabilitySurface] = []
        module, import_surface = self._import_module(module_name)
        surfaces.append(import_surface)
        if module is None:
            return self._family(
                CapabilityFamily.AST_PRODUCERS,
                surfaces,
                authority=CapabilityAuthority.AST_OBSERVATION,
                available_when=lambda _: False,
            )

        extractor_cls = getattr(module, "PythonASTExtractor", None)
        if extractor_cls is None or not inspect.isclass(extractor_cls):
            surfaces.append(
                CapabilitySurface(
                    surface_id="ast.python_ast_extractor",
                    status=CapabilityProbeStatus.INCOMPATIBLE,
                    reason_code=CapabilityReasonCode.CALLABLE_MISSING,
                    reason="PythonASTExtractor class is missing",
                    required=True,
                )
            )
        else:
            method = getattr(extractor_cls, "extract_from_source", None)
            if method is None or not callable(method):
                surfaces.append(
                    CapabilitySurface(
                        surface_id="ast.python_ast_extractor.extract_from_source",
                        status=CapabilityProbeStatus.INCOMPATIBLE,
                        reason_code=CapabilityReasonCode.SIGNATURE_INCOMPATIBLE,
                        reason="extract_from_source is missing or not callable",
                        required=True,
                    )
                )
            else:
                try:
                    params = _signature_params(method)
                except ProgramAnalysisCapabilityError as exc:
                    surfaces.append(
                        CapabilitySurface(
                            surface_id="ast.python_ast_extractor.extract_from_source",
                            status=CapabilityProbeStatus.INCOMPATIBLE,
                            reason_code=CapabilityReasonCode.SIGNATURE_INCOMPATIBLE,
                            reason=str(exc),
                            required=True,
                        )
                    )
                else:
                    if "source" not in params:
                        surfaces.append(
                            CapabilitySurface(
                                surface_id="ast.python_ast_extractor.extract_from_source",
                                status=CapabilityProbeStatus.INCOMPATIBLE,
                                reason_code=CapabilityReasonCode.SIGNATURE_INCOMPATIBLE,
                                reason=(
                                    "extract_from_source signature incompatible; "
                                    "expected parameter 'source'"
                                ),
                                required=True,
                                metadata={"parameters": sorted(params)},
                            )
                        )
                    else:
                        # Limited domain is a diagnostic, not a constant claim of
                        # general program-AST coverage.
                        surfaces.append(
                            CapabilitySurface(
                                surface_id="ast.python_ast_extractor",
                                status=CapabilityProbeStatus.PARTIAL,
                                reason_code=CapabilityReasonCode.PARTIAL,
                                reason=(
                                    "crypto-exchange PythonASTExtractor is available "
                                    "with a compatible extract_from_source signature; "
                                    "coverage is limited to that domain"
                                ),
                                authority=CapabilityAuthority.AST_OBSERVATION,
                                required=True,
                                metadata={
                                    "module": module_name,
                                    "domain": "crypto_exchange",
                                    "limited": True,
                                    "methods": list(_AST_EXTRACTOR_METHODS),
                                    "parameters": sorted(params)[:32],
                                    "observation": True,
                                },
                            )
                        )

        def available(items: Sequence[CapabilitySurface]) -> bool:
            return any(
                item.surface_id.startswith("ast.")
                and item.status
                in {
                    CapabilityProbeStatus.AVAILABLE,
                    CapabilityProbeStatus.PARTIAL,
                }
                for item in items
            )

        def partial(items: Sequence[CapabilitySurface]) -> bool:
            return any(
                item.status is CapabilityProbeStatus.PARTIAL for item in items
            )

        return self._family(
            CapabilityFamily.AST_PRODUCERS,
            surfaces,
            authority=CapabilityAuthority.AST_OBSERVATION,
            available_when=available,
            partial_when=partial,
        )

    def _probe_zkp(self) -> CapabilityFamilyReport:
        root = self.config.optional_root
        backends_name = f"{root}.logic.zkp.backends"
        circuits_name = f"{root}.logic.zkp.circuits"
        surfaces: list[CapabilitySurface] = []

        backends_mod, backends_import = self._import_module(backends_name)
        surfaces.append(backends_import)
        circuits_mod, circuits_import = self._import_module(circuits_name)
        surfaces.append(circuits_import)

        simulated_only = False
        crypto_ready = False

        if backends_mod is not None:
            get_backend = getattr(backends_mod, "get_backend", None)
            metadata = getattr(backends_mod, "_BACKEND_METADATA", None)
            if get_backend is None or not callable(get_backend):
                surfaces.append(
                    CapabilitySurface(
                        surface_id="zkp.backends.get_backend",
                        status=CapabilityProbeStatus.INCOMPATIBLE,
                        reason_code=CapabilityReasonCode.SIGNATURE_INCOMPATIBLE,
                        reason="get_backend is missing or not callable",
                        required=True,
                    )
                )
            else:
                try:
                    params = _signature_params(get_backend)
                except ProgramAnalysisCapabilityError as exc:
                    surfaces.append(
                        CapabilitySurface(
                            surface_id="zkp.backends.get_backend",
                            status=CapabilityProbeStatus.INCOMPATIBLE,
                            reason_code=CapabilityReasonCode.SIGNATURE_INCOMPATIBLE,
                            reason=str(exc),
                            required=True,
                        )
                    )
                else:
                    if "backend" not in params:
                        surfaces.append(
                            CapabilitySurface(
                                surface_id="zkp.backends.get_backend",
                                status=CapabilityProbeStatus.INCOMPATIBLE,
                                reason_code=CapabilityReasonCode.SIGNATURE_INCOMPATIBLE,
                                reason=(
                                    "get_backend signature incompatible; "
                                    "expected parameter 'backend'"
                                ),
                                required=True,
                            )
                        )
                    else:
                        backend_ids = (
                            sorted(metadata.keys())
                            if isinstance(metadata, Mapping)
                            else []
                        )
                        surfaces.append(
                            CapabilitySurface(
                                surface_id="zkp.backends.registry",
                                status=CapabilityProbeStatus.AVAILABLE,
                                reason_code=CapabilityReasonCode.CURRENT_DIAGNOSTIC,
                                reason=(
                                    "ZKP backend registry is importable; "
                                    "authority requires a non-simulated backend"
                                ),
                                authority=CapabilityAuthority.ZKP_DIAGNOSTIC,
                                required=True,
                                metadata={
                                    "backends": backend_ids,
                                    "default_is_simulated": True,
                                },
                            )
                        )
                        if "simulated" in backend_ids or not backend_ids:
                            simulated_only = True
                            surfaces.append(
                                CapabilitySurface(
                                    surface_id="zkp.backends.simulated",
                                    status=CapabilityProbeStatus.SIMULATED,
                                    reason_code=CapabilityReasonCode.SIMULATED_ZKP_ONLY,
                                    reason=(
                                        "simulated ZKP backend is present for "
                                        "diagnostics only and cannot satisfy "
                                        "cryptographic authority"
                                    ),
                                    authority=CapabilityAuthority.ZKP_DIAGNOSTIC,
                                    required=False,
                                    metadata={
                                        "cryptographic": False,
                                        "authority": False,
                                    },
                                )
                            )
                            # Explicit permanent rejection of simulated authority.
                            surfaces.append(
                                CapabilitySurface(
                                    surface_id="zkp.authority.simulated",
                                    status=CapabilityProbeStatus.REJECTED,
                                    reason_code=(
                                        CapabilityReasonCode.SIMULATED_ZKP_AUTHORITY_REJECTED
                                    ),
                                    reason=(
                                        "simulated/fallback ZKP authority is permanently rejected"
                                    ),
                                    authority=CapabilityAuthority.NONE,
                                    required=True,
                                    metadata={"cryptographic": False},
                                )
                            )

            # Cryptographic backends require executable + artifacts, not presence.
            groth16_exe = self._executable_surface(
                "groth16",
                ("groth16",),
                env_names=("IPFS_DATASETS_GROTH16_BINARY", "GROTH16_BINARY"),
                required=False,
            )
            provekit_exe = self._executable_surface(
                "provekit",
                ("provekit-cli", "provekit"),
                env_names=("IPFS_DATASETS_PROVEKIT_BINARY", "PROVEKIT_CLI"),
                required=False,
            )
            surfaces.append(
                CapabilitySurface(
                    surface_id="zkp.backend.groth16.executable",
                    status=groth16_exe.status,
                    reason_code=groth16_exe.reason_code,
                    reason=groth16_exe.reason,
                    authority=CapabilityAuthority.ZKP_DIAGNOSTIC,
                    required=False,
                    location=groth16_exe.location,
                    metadata={**dict(groth16_exe.metadata), "cryptographic": True},
                )
            )
            surfaces.append(
                CapabilitySurface(
                    surface_id="zkp.backend.provekit.executable",
                    status=provekit_exe.status,
                    reason_code=provekit_exe.reason_code,
                    reason=provekit_exe.reason,
                    authority=CapabilityAuthority.ZKP_DIAGNOSTIC,
                    required=False,
                    location=provekit_exe.location,
                    metadata={**dict(provekit_exe.metadata), "cryptographic": True},
                )
            )
            groth16_artifacts = self._artifact_dir_surface(
                "zkp.backend.groth16.artifacts",
                self.config.groth16_artifacts_path,
                env_names=(
                    "IPFS_DATASETS_GROTH16_ARTIFACTS_DIR",
                    "GROTH16_ARTIFACTS_DIR",
                ),
                required_files=("proving_key.bin", "verifying_key.bin"),
            )
            provekit_artifacts = self._artifact_dir_surface(
                "zkp.backend.provekit.artifacts",
                self.config.provekit_artifacts_path,
                env_names=(
                    "IPFS_DATASETS_PROVEKIT_ARTIFACTS_DIR",
                    "PROVEKIT_ARTIFACTS_DIR",
                ),
                required_files=(),
            )
            surfaces.extend((groth16_artifacts, provekit_artifacts))
            crypto_ready = (
                groth16_exe.status is CapabilityProbeStatus.AVAILABLE
                and groth16_artifacts.status is CapabilityProbeStatus.AVAILABLE
            ) or (
                provekit_exe.status is CapabilityProbeStatus.AVAILABLE
                and provekit_artifacts.status is CapabilityProbeStatus.AVAILABLE
            )
            if crypto_ready:
                surfaces.append(
                    CapabilitySurface(
                        surface_id="zkp.backend.cryptographic",
                        status=CapabilityProbeStatus.AVAILABLE,
                        reason_code=CapabilityReasonCode.CURRENT_DIAGNOSTIC,
                        reason=(
                            "at least one cryptographic ZKP backend executable "
                            "and artifacts are discoverable; production self-tests "
                            "are outside this discovery probe"
                        ),
                        authority=CapabilityAuthority.ZKP_DIAGNOSTIC,
                        required=False,
                        metadata={
                            "cryptographic": True,
                            "production_verified": False,
                        },
                    )
                )
            elif simulated_only:
                # Already recorded rejection of simulated authority.
                pass

        if circuits_mod is not None:
            # circuits package must exist; treat import success as diagnostic.
            surfaces.append(
                CapabilitySurface(
                    surface_id="zkp.circuits",
                    status=CapabilityProbeStatus.AVAILABLE,
                    reason_code=CapabilityReasonCode.CURRENT_DIAGNOSTIC,
                    reason="ZKP circuits module imported for discovery",
                    authority=CapabilityAuthority.ZKP_DIAGNOSTIC,
                    required=True,
                    location=getattr(circuits_mod, "__file__", None),
                    metadata={"module": circuits_name},
                )
            )
        elif backends_mod is not None:
            surfaces.append(
                CapabilitySurface(
                    surface_id="zkp.circuits",
                    status=CapabilityProbeStatus.UNAVAILABLE,
                    reason_code=CapabilityReasonCode.PACKAGE_MISSING,
                    reason="ZKP circuits package is not importable",
                    required=True,
                )
            )

        def available(items: Sequence[CapabilitySurface]) -> bool:
            return any(
                item.surface_id
                in {
                    "zkp.backends.registry",
                    "zkp.backend.cryptographic",
                    "zkp.backends.simulated",
                    "zkp.circuits",
                }
                and item.status
                in {
                    CapabilityProbeStatus.AVAILABLE,
                    CapabilityProbeStatus.SIMULATED,
                    CapabilityProbeStatus.PARTIAL,
                    CapabilityProbeStatus.DEGRADED,
                }
                for item in items
            )

        def partial(items: Sequence[CapabilitySurface]) -> bool:
            has_sim = any(
                item.surface_id == "zkp.backends.simulated"
                and item.status is CapabilityProbeStatus.SIMULATED
                for item in items
            )
            has_crypto = any(
                item.surface_id == "zkp.backend.cryptographic"
                and item.status is CapabilityProbeStatus.AVAILABLE
                for item in items
            )
            return has_sim and not has_crypto

        # Special-case: simulated authority rejection should not make the whole
        # family REJECTED if we only mean authority is rejected.  Reclassify
        # family via custom logic when only simulated is present.
        report = self._family(
            CapabilityFamily.ZKP,
            surfaces,
            authority=CapabilityAuthority.ZKP_DIAGNOSTIC,
            available_when=available,
            partial_when=partial,
        )
        if (
            report.status is CapabilityProbeStatus.REJECTED
            and report.reason_code
            is CapabilityReasonCode.SIMULATED_ZKP_AUTHORITY_REJECTED
        ):
            # Authority is rejected, but the family itself is a simulated diagnostic.
            return CapabilityFamilyReport(
                family=CapabilityFamily.ZKP,
                status=CapabilityProbeStatus.SIMULATED
                if not crypto_ready
                else CapabilityProbeStatus.PARTIAL,
                reason_code=(
                    CapabilityReasonCode.SIMULATED_ZKP_ONLY
                    if not crypto_ready
                    else CapabilityReasonCode.PARTIAL
                ),
                reason=(
                    "ZKP surfaces are discoverable; simulated backends remain "
                    "diagnostic-only and cannot authorize ZK receipts"
                    if not crypto_ready
                    else "cryptographic ZKP backends are only partially configured"
                ),
                authority=CapabilityAuthority.ZKP_DIAGNOSTIC,
                surfaces=report.surfaces,
                metadata={
                    "simulated_authority": False,
                    "cryptographic_ready": crypto_ready,
                },
            )
        return report

    def _artifact_dir_surface(
        self,
        surface_id: str,
        configured: str | None,
        *,
        env_names: Sequence[str],
        required_files: Sequence[str],
    ) -> CapabilitySurface:
        limited = self._budget_reason()
        if limited is not None:
            status = (
                CapabilityProbeStatus.TIMED_OUT
                if limited is CapabilityReasonCode.PROBE_TIMEOUT
                else CapabilityProbeStatus.UNAVAILABLE
            )
            return CapabilitySurface(
                surface_id=surface_id,
                status=status,
                reason_code=limited,
                reason=f"artifact inspection for {surface_id} skipped due to probe budget",
                required=False,
                metadata={"probe_limited": True},
            )
        path_raw = (configured or "").strip()
        if not path_raw:
            for env_name in env_names:
                path_raw = str(self._environ.get(env_name, "") or "").strip()
                if path_raw:
                    break
        if not path_raw:
            return CapabilitySurface(
                surface_id=surface_id,
                status=CapabilityProbeStatus.UNAVAILABLE,
                reason_code=CapabilityReasonCode.PACKAGE_MISSING,
                reason=f"{surface_id} artifacts are not configured",
                required=False,
            )
        path = Path(path_raw)
        if not path.is_dir():
            return CapabilitySurface(
                surface_id=surface_id,
                status=CapabilityProbeStatus.UNAVAILABLE,
                reason_code=CapabilityReasonCode.PACKAGE_MISSING,
                reason=f"{surface_id} artifact path is not a directory: {path}",
                required=False,
                location=str(path),
            )
        missing = [name for name in required_files if not (path / name).is_file()]
        if missing:
            return CapabilitySurface(
                surface_id=surface_id,
                status=CapabilityProbeStatus.UNAVAILABLE,
                reason_code=CapabilityReasonCode.PACKAGE_MISSING,
                reason=f"{surface_id} missing required files: {missing}",
                required=False,
                location=str(path),
                metadata={"missing": missing},
            )
        return CapabilitySurface(
            surface_id=surface_id,
            status=CapabilityProbeStatus.AVAILABLE,
            reason_code=CapabilityReasonCode.CURRENT_DIAGNOSTIC,
            reason=f"{surface_id} artifacts directory is present",
            authority=CapabilityAuthority.ZKP_DIAGNOSTIC,
            required=False,
            location=str(path),
            metadata={"required_files": list(required_files)},
        )


class IpfsDatasetsProgramAnalysisProvider:
    """Supervisor-facing facade for the program-analysis capability matrix.

    Construction and matrix declaration are cold-import safe.  A probe is the
    only path that may import optional packages, and every result remains
    non-authoritative for completion and proof.
    """

    def __init__(
        self,
        config: ProgramAnalysisProbeConfig | None = None,
        *,
        probe: ProgramAnalysisCapabilityProbe | None = None,
        find_spec: PackageFinder | None = None,
        which: ExecutableFinder | None = None,
        importer: ModuleImporter | None = None,
        environ: Mapping[str, str] | None = None,
        monotonic: Clock | None = None,
        wall_clock: Clock | None = None,
    ) -> None:
        self.config = config or ProgramAnalysisProbeConfig()
        self._probe = probe or ProgramAnalysisCapabilityProbe(
            self.config,
            find_spec=find_spec,
            which=which,
            importer=importer,
            environ=environ,
            monotonic=monotonic,
            wall_clock=wall_clock,
        )

    @property
    def provider_id(self) -> str:
        return IPFS_DATASETS_PROGRAM_ANALYSIS_PROVIDER_ID

    def capabilities(self) -> ProgramAnalysisCapabilityMatrix:
        """Return the closed lazy matrix without probing or importing."""

        return self._probe.matrix()

    def capability(self) -> ProgramAnalysisCapabilityMatrix:
        return self.capabilities()

    def probe_capabilities(
        self, *, force_refresh: bool = False
    ) -> ProgramAnalysisCapabilityReport:
        """Run (or reuse) a bounded real capability probe."""

        return self._probe.probe(force_refresh=force_refresh)

    def clear_cache(self) -> None:
        self._probe.clear_cache()


_DEFAULT_PROVIDER = IpfsDatasetsProgramAnalysisProvider()


def probe_program_analysis_capabilities(
    *,
    force_refresh: bool = False,
) -> ProgramAnalysisCapabilityReport:
    """Process-wide cached program-analysis capability report."""

    return _DEFAULT_PROVIDER.probe_capabilities(force_refresh=force_refresh)


def clear_program_analysis_capability_cache() -> None:
    """Clear the process-wide capability cache (tests/operators)."""

    _DEFAULT_PROVIDER.clear_cache()


__all__ = [
    "PROGRAM_ANALYSIS_CAPABILITY_SCHEMA",
    "PROGRAM_ANALYSIS_CAPABILITY_REPORT_SCHEMA",
    "PROGRAM_ANALYSIS_CAPABILITY_REPORT_VERSION",
    "IPFS_DATASETS_PROGRAM_ANALYSIS_PROVIDER_ID",
    "IPFS_DATASETS_PROGRAM_ANALYSIS_PROVIDER_VERSION",
    "CAPABILITY_FAMILY_ORDER",
    "DEFAULT_OPTIONAL_ROOT",
    "DEFAULT_PROBE_TIMEOUT_SECONDS",
    "DEFAULT_PROBE_CACHE_TTL_SECONDS",
    "DEFAULT_PROBE_MAX_CHECKS",
    "DEFAULT_MAX_REPORT_BYTES",
    "ProgramAnalysisCapabilityError",
    "CapabilityFamily",
    "CapabilityProbeStatus",
    "CapabilityAuthority",
    "CapabilityReasonCode",
    "ProgramAnalysisProbeConfig",
    "CapabilitySurface",
    "CapabilityFamilyReport",
    "ProgramAnalysisCapabilityMatrix",
    "ProgramAnalysisCapabilityReport",
    "ProgramAnalysisCapabilityProbe",
    "IpfsDatasetsProgramAnalysisProvider",
    "declare_program_analysis_capability_matrix",
    "inspect_program_analysis_capability_matrix",
    "probe_program_analysis_capabilities",
    "clear_program_analysis_capability_cache",
]
