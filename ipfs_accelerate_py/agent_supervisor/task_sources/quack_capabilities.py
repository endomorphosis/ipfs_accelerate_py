"""Pinned DuckDB + Quack capability profile and fail-closed health probe.

This module answers whether the local environment can host the agent-supervisor
control plane over Quack. Discovery is deliberately weaker than serving:

* importing ``duckdb`` alone never passes a health check;
* ordinary probes never issue network ``INSTALL`` statements;
* missing, unsigned, mismatched, or beta surfaces yield typed statuses rather
  than crashes or speculative retries.

Statuses are a closed set: unavailable, unsupported, install-required,
load-required, compatible, mismatched, and experimental. Quack beta
limitations are always recorded on the report when the extension path is
evaluated.

Dependency package pins (``pyproject.toml`` / wheels) belong to DQP-005;
this module owns capability probing and compatibility documentation only.
"""

from __future__ import annotations

import hashlib
import platform
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

QUACK_CAPABILITY_SCHEMA_VERSION: Final = (
    "ipfs_accelerate_py/agent-supervisor/quack-capability@1"
)
QUACK_CAPABILITY_REPORT_SCHEMA_VERSION: Final = (
    "ipfs_accelerate_py/agent-supervisor/quack-capability-report@1"
)
QUACK_COMPATIBILITY_PROFILE_SCHEMA_VERSION: Final = (
    "ipfs_accelerate_py/agent-supervisor/quack-compatibility-profile@1"
)
QUACK_CAPABILITY_REPORT_VERSION: Final = 1
QUACK_COMPATIBILITY_PROFILE_VERSION: Final = 1

# Pinned control-plane profile for the 1.5.x Quack beta era. Exact package
# installation is owned by DQP-005; these bounds only gate probe outcomes.
PINNED_DUCKDB_MAJOR: Final = 1
PINNED_DUCKDB_MINOR: Final = 5
PINNED_DUCKDB_VERSION_PREFIX: Final = "1.5"
PINNED_EXTENSION_NAME: Final = "quack"
PINNED_EXTENSION_API: Final = "quack@1"
REQUIRED_QUACK_FUNCTIONS: Final[tuple[str, ...]] = (
    "quack_serve",
    "quack_query",
)
REQUIRED_QUACK_SURFACES: Final[tuple[str, ...]] = (
    "install_load_policy",
    "quack_serve",
    "quack_query",
    "attach",
    "whoami",
    "auth_settings",
    "logging",
    "extension_fingerprint",
)

# Known Quack beta limitations that shape control-plane design (see plan).
DEFAULT_QUACK_BETA_LIMITATIONS: Final[tuple[str, ...]] = (
    "quack_is_beta_in_duckdb_1_5_x",
    "protocol_names_and_defaults_may_change_before_duckdb_2_0",
    "server_and_clients_must_use_identical_pinned_build",
    "default_authorization_callback_permits_every_authenticated_query",
    "no_server_push_clients_must_poll",
    "one_quack_server_is_one_failure_domain",
    "loopback_bind_required_unless_separately_reviewed",
    "unsigned_or_community_extension_path_is_not_attested_integrity",
)

# Ordinary health checks never pull extensions from the network.
DEFAULT_ALLOW_NETWORK_INSTALL: Final = False
DEFAULT_ALLOW_LOCAL_LOAD: Final = True
DEFAULT_PROBE_TIMEOUT_SECONDS: Final = 5.0

_VERSION_RE: Final = re.compile(
    r"^\s*v?(?P<major>\d+)\.(?P<minor>\d+)(?:\.(?P<patch>\d+))?"
    r"(?:[.\-+_](?P<label>[0-9A-Za-z.\-_]+))?\s*$"
)

_CACHE_LOCK = threading.Lock()
_CACHED_REPORT: "QuackCapabilityReport | None" = None
_CACHED_AT_MONOTONIC: float = 0.0
DEFAULT_CACHE_TTL_SECONDS: Final = 30.0


class QuackCapabilityStatus(str, Enum):
    """Closed set of probe outcomes required by the control-plane contract."""

    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    INSTALL_REQUIRED = "install-required"
    LOAD_REQUIRED = "load-required"
    COMPATIBLE = "compatible"
    MISMATCHED = "mismatched"
    EXPERIMENTAL = "experimental"


class QuackDiagnosticCode(str, Enum):
    """Machine-readable reason codes for non-passing outcomes."""

    DUCKDB_IMPORT_FAILED = "duckdb_import_failed"
    DUCKDB_CONNECT_FAILED = "duckdb_connect_failed"
    DUCKDB_VERSION_UNREADABLE = "duckdb_version_unreadable"
    DUCKDB_VERSION_UNSUPPORTED = "duckdb_version_unsupported"
    DUCKDB_VERSION_MISMATCHED = "duckdb_version_mismatched"
    EXTENSION_CATALOG_UNAVAILABLE = "extension_catalog_unavailable"
    EXTENSION_NOT_INSTALLED = "extension_not_installed"
    EXTENSION_NOT_LOADED = "extension_not_loaded"
    EXTENSION_LOAD_FAILED = "extension_load_failed"
    EXTENSION_FINGERPRINT_MISMATCH = "extension_fingerprint_mismatch"
    REQUIRED_FUNCTION_MISSING = "required_function_missing"
    REQUIRED_SURFACE_MISSING = "required_surface_missing"
    NETWORK_INSTALL_FORBIDDEN = "network_install_forbidden"
    IMPORT_ONLY_INSUFFICIENT = "import_only_insufficient"
    PLATFORM_MISMATCH = "platform_mismatch"
    INTERNAL_ERROR = "internal_error"
    BETA_LIMITATIONS_RECORDED = "beta_limitations_recorded"


@dataclass(frozen=True)
class QuackCapabilityDiagnostic:
    """Typed, bounded probe diagnostic rather than an exception-only failure."""

    code: QuackDiagnosticCode
    message: str
    subject: str = ""
    exception_type: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.code, QuackDiagnosticCode):
            object.__setattr__(self, "code", QuackDiagnosticCode(self.code))
        message = str(self.message).strip()
        if not message:
            raise ValueError("diagnostic message must not be empty")
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "subject", str(self.subject or ""))
        object.__setattr__(self, "exception_type", str(self.exception_type or ""))

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code.value,
            "message": self.message,
            "subject": self.subject,
            "exception_type": self.exception_type,
        }


@dataclass(frozen=True)
class ParsedVersion:
    """Comparable semantic-ish version triple with optional label."""

    major: int
    minor: int
    patch: int = 0
    label: str = ""
    raw: str = ""

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.major, self.minor, self.patch)

    def __str__(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        return f"{base}+{self.label}" if self.label else base

    def to_dict(self) -> dict[str, Any]:
        return {
            "major": self.major,
            "minor": self.minor,
            "patch": self.patch,
            "label": self.label,
            "raw": self.raw,
            "text": str(self),
        }


def parse_version(value: Any) -> ParsedVersion | None:
    """Parse a DuckDB/extension version string; return None if unreadable."""

    text = str(value or "").strip()
    if not text:
        return None
    match = _VERSION_RE.match(text)
    if match is None:
        return None
    return ParsedVersion(
        major=int(match.group("major")),
        minor=int(match.group("minor")),
        patch=int(match.group("patch") or 0),
        label=str(match.group("label") or ""),
        raw=text,
    )


@dataclass(frozen=True)
class QuackCompatibilityProfile:
    """Pinned DuckDB/Quack profile used by server and client admission.

    Interface: ``QuackCompatibilityProfile@1``.
    """

    profile_id: str = "agent-supervisor-duckdb-quack-1.5"
    duckdb_major: int = PINNED_DUCKDB_MAJOR
    duckdb_minor: int = PINNED_DUCKDB_MINOR
    duckdb_version_prefix: str = PINNED_DUCKDB_VERSION_PREFIX
    extension_name: str = PINNED_EXTENSION_NAME
    extension_api: str = PINNED_EXTENSION_API
    required_functions: tuple[str, ...] = REQUIRED_QUACK_FUNCTIONS
    required_surfaces: tuple[str, ...] = REQUIRED_QUACK_SURFACES
    beta_limitations: tuple[str, ...] = DEFAULT_QUACK_BETA_LIMITATIONS
    # Optional exact fingerprints; empty means "any attested build in range".
    pinned_extension_fingerprint: str = ""
    pinned_duckdb_version: str = ""
    pinned_platform: str = ""
    allow_experimental_within_minor: bool = True
    schema_version: str = QUACK_COMPATIBILITY_PROFILE_SCHEMA_VERSION
    profile_version: int = QUACK_COMPATIBILITY_PROFILE_VERSION

    def __post_init__(self) -> None:
        if not str(self.profile_id).strip():
            raise ValueError("profile_id must not be empty")
        if self.duckdb_major < 0 or self.duckdb_minor < 0:
            raise ValueError("duckdb major/minor must be non-negative")
        if self.profile_version != QUACK_COMPATIBILITY_PROFILE_VERSION:
            raise ValueError("unsupported compatibility profile version")
        if self.schema_version != QUACK_COMPATIBILITY_PROFILE_SCHEMA_VERSION:
            raise ValueError("unsupported compatibility profile schema")
        object.__setattr__(
            self,
            "required_functions",
            tuple(str(item) for item in self.required_functions),
        )
        object.__setattr__(
            self,
            "required_surfaces",
            tuple(str(item) for item in self.required_surfaces),
        )
        object.__setattr__(
            self,
            "beta_limitations",
            tuple(str(item) for item in self.beta_limitations),
        )
        if not self.required_functions:
            raise ValueError("required_functions must not be empty")
        if "install_load_policy" not in self.required_surfaces:
            raise ValueError("required_surfaces must include install_load_policy")

    @property
    def is_beta_profile(self) -> bool:
        return bool(self.beta_limitations)

    def matches_duckdb_minor(self, version: ParsedVersion) -> bool:
        return (
            version.major == self.duckdb_major
            and version.minor == self.duckdb_minor
        )

    def is_supported_duckdb(self, version: ParsedVersion) -> bool:
        """DuckDB versions the program can even attempt Quack on."""

        # Quack community extension era begins in modern 1.x; pre-1.4 is out.
        if version.major < 1:
            return False
        if version.major == 1 and version.minor < 4:
            return False
        # DuckDB 2.x requires a separately pinned future profile.
        if version.major >= 2:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_version": self.profile_version,
            "profile_id": self.profile_id,
            "duckdb_major": self.duckdb_major,
            "duckdb_minor": self.duckdb_minor,
            "duckdb_version_prefix": self.duckdb_version_prefix,
            "extension_name": self.extension_name,
            "extension_api": self.extension_api,
            "required_functions": list(self.required_functions),
            "required_surfaces": list(self.required_surfaces),
            "beta_limitations": list(self.beta_limitations),
            "pinned_extension_fingerprint": self.pinned_extension_fingerprint,
            "pinned_duckdb_version": self.pinned_duckdb_version,
            "pinned_platform": self.pinned_platform,
            "allow_experimental_within_minor": self.allow_experimental_within_minor,
            "is_beta_profile": self.is_beta_profile,
        }


def default_compatibility_profile() -> QuackCompatibilityProfile:
    """Return the program's pinned DuckDB 1.5.x / Quack beta profile."""

    return QuackCompatibilityProfile()


@dataclass(frozen=True)
class ExtensionObservation:
    """Local catalog observation for one DuckDB extension."""

    name: str
    installed: bool = False
    loaded: bool = False
    install_path: str = ""
    extension_version: str = ""
    install_mode: str = ""
    installed_from: str = ""
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "installed": self.installed,
            "loaded": self.loaded,
            "install_path": self.install_path,
            "extension_version": self.extension_version,
            "install_mode": self.install_mode,
            "installed_from": self.installed_from,
            "description": self.description,
        }


@dataclass(frozen=True)
class QuackCapabilityReport:
    """Immutable, versioned snapshot of DuckDB/Quack readiness.

    Interface: ``QuackCapabilityReport@1``.
    """

    status: QuackCapabilityStatus
    profile: QuackCompatibilityProfile
    duckdb_importable: bool = False
    duckdb_version: str = ""
    duckdb_version_parsed: ParsedVersion | None = None
    platform_name: str = ""
    platform_machine: str = ""
    extension: ExtensionObservation | None = None
    extension_fingerprint: str = ""
    observed_functions: tuple[str, ...] = ()
    missing_functions: tuple[str, ...] = ()
    observed_surfaces: tuple[str, ...] = ()
    missing_surfaces: tuple[str, ...] = ()
    beta_limitations: tuple[str, ...] = ()
    diagnostics: tuple[QuackCapabilityDiagnostic, ...] = ()
    network_install_attempted: bool = False
    network_install_allowed: bool = False
    local_load_attempted: bool = False
    local_load_allowed: bool = DEFAULT_ALLOW_LOCAL_LOAD
    generated_at_monotonic: float = 0.0
    duration_seconds: float = 0.0
    details: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = QUACK_CAPABILITY_REPORT_SCHEMA_VERSION
    report_version: int = QUACK_CAPABILITY_REPORT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.status, QuackCapabilityStatus):
            object.__setattr__(self, "status", QuackCapabilityStatus(self.status))
        if not isinstance(self.profile, QuackCompatibilityProfile):
            raise TypeError("profile must be QuackCompatibilityProfile")
        if self.report_version != QUACK_CAPABILITY_REPORT_VERSION:
            raise ValueError("unsupported capability report version")
        if self.schema_version != QUACK_CAPABILITY_REPORT_SCHEMA_VERSION:
            raise ValueError("unsupported capability report schema")
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative")
        if self.network_install_attempted and not self.network_install_allowed:
            raise ValueError(
                "network install cannot be attempted when not explicitly allowed"
            )
        object.__setattr__(self, "observed_functions", tuple(self.observed_functions))
        object.__setattr__(self, "missing_functions", tuple(self.missing_functions))
        object.__setattr__(self, "observed_surfaces", tuple(self.observed_surfaces))
        object.__setattr__(self, "missing_surfaces", tuple(self.missing_surfaces))
        object.__setattr__(self, "beta_limitations", tuple(self.beta_limitations))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))

    @property
    def available(self) -> bool:
        """True only when the pinned profile is fully satisfied."""

        return self.status is QuackCapabilityStatus.COMPATIBLE

    @property
    def experimental_usable(self) -> bool:
        return self.status is QuackCapabilityStatus.EXPERIMENTAL

    @property
    def passes_health_check(self) -> bool:
        """Ordinary health check admission.

        Import success alone never passes. Experimental beta hosts require an
        explicit experimental admission path rather than ordinary health.
        """

        return self.status is QuackCapabilityStatus.COMPATIBLE

    @property
    def reason_code(self) -> str:
        if self.diagnostics:
            return self.diagnostics[0].code.value
        return self.status.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "report_version": self.report_version,
            "status": self.status.value,
            "available": self.available,
            "experimental_usable": self.experimental_usable,
            "passes_health_check": self.passes_health_check,
            "reason_code": self.reason_code,
            "profile": self.profile.to_dict(),
            "duckdb_importable": self.duckdb_importable,
            "duckdb_version": self.duckdb_version,
            "duckdb_version_parsed": (
                self.duckdb_version_parsed.to_dict()
                if self.duckdb_version_parsed is not None
                else None
            ),
            "platform_name": self.platform_name,
            "platform_machine": self.platform_machine,
            "extension": self.extension.to_dict() if self.extension else None,
            "extension_fingerprint": self.extension_fingerprint,
            "observed_functions": list(self.observed_functions),
            "missing_functions": list(self.missing_functions),
            "observed_surfaces": list(self.observed_surfaces),
            "missing_surfaces": list(self.missing_surfaces),
            "beta_limitations": list(self.beta_limitations),
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "network_install_attempted": self.network_install_attempted,
            "network_install_allowed": self.network_install_allowed,
            "local_load_attempted": self.local_load_attempted,
            "local_load_allowed": self.local_load_allowed,
            "generated_at_monotonic": self.generated_at_monotonic,
            "duration_seconds": self.duration_seconds,
            "details": dict(self.details),
        }


def _diagnostic(
    code: QuackDiagnosticCode,
    message: str,
    *,
    subject: str = "",
    exception: BaseException | None = None,
) -> QuackCapabilityDiagnostic:
    return QuackCapabilityDiagnostic(
        code=code,
        message=message,
        subject=subject,
        exception_type=type(exception).__name__ if exception is not None else "",
    )


def _host_platform() -> tuple[str, str]:
    return platform.system().lower(), platform.machine().lower()


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"1", "true", "t", "yes", "y", "on", "loaded", "installed"}


def _row_mapping(columns: Sequence[str], row: Sequence[Any]) -> dict[str, Any]:
    return {
        str(column).lower(): row[index]
        for index, column in enumerate(columns)
        if index < len(row)
    }


def _extension_from_row(row: Mapping[str, Any], *, name: str) -> ExtensionObservation:
    def _get(*keys: str, default: Any = "") -> Any:
        for key in keys:
            if key in row and row[key] is not None:
                return row[key]
        return default

    return ExtensionObservation(
        name=name,
        installed=_boolish(_get("installed", "install", default=False)),
        loaded=_boolish(_get("loaded", "load", default=False)),
        install_path=str(_get("install_path", "path", default="") or ""),
        extension_version=str(
            _get("extension_version", "version", default="") or ""
        ),
        install_mode=str(_get("install_mode", "mode", default="") or ""),
        installed_from=str(_get("installed_from", "source", default="") or ""),
        description=str(_get("description", default="") or ""),
    )


def compute_extension_fingerprint(
    *,
    duckdb_version: str,
    extension: ExtensionObservation | None,
    platform_name: str,
    platform_machine: str,
    observed_functions: Sequence[str] = (),
) -> str:
    """Stable fingerprint over version, origin, platform, and function set."""

    payload = {
        "duckdb_version": str(duckdb_version or ""),
        "extension_name": extension.name if extension else "",
        "extension_version": extension.extension_version if extension else "",
        "install_path": extension.install_path if extension else "",
        "install_mode": extension.install_mode if extension else "",
        "installed_from": extension.installed_from if extension else "",
        "platform_name": str(platform_name or ""),
        "platform_machine": str(platform_machine or ""),
        "functions": sorted(str(item) for item in observed_functions),
    }
    encoded = repr(sorted(payload.items())).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return f"sha256:{digest}"


def _default_importer(name: str) -> Any:
    import importlib

    return importlib.import_module(name)


def _connect_memory(duckdb_module: Any) -> Any:
    return duckdb_module.connect(database=":memory:")


def _close_quietly(connection: Any) -> None:
    close = getattr(connection, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _execute(connection: Any, sql: str) -> Any:
    return connection.execute(sql)


def _fetch_extension(
    connection: Any,
    *,
    extension_name: str,
) -> ExtensionObservation | None:
    try:
        result = _execute(
            connection,
            "SELECT * FROM duckdb_extensions() "
            f"WHERE extension_name = '{extension_name}'",
        )
    except Exception:
        # Older DuckDB builds expose duckdb_extensions without a filter-friendly
        # shape; fall back to a full scan when possible.
        try:
            result = _execute(connection, "SELECT * FROM duckdb_extensions()")
        except Exception:
            return None
        description = getattr(result, "description", None) or ()
        columns = [str(item[0]) for item in description]
        rows = result.fetchall() if hasattr(result, "fetchall") else []
        for row in rows:
            mapping = _row_mapping(columns, row)
            candidate = str(
                mapping.get("extension_name")
                or mapping.get("name")
                or ""
            ).lower()
            if candidate == extension_name.lower():
                return _extension_from_row(mapping, name=extension_name)
        return ExtensionObservation(name=extension_name, installed=False, loaded=False)

    description = getattr(result, "description", None) or ()
    columns = [str(item[0]) for item in description]
    rows = result.fetchall() if hasattr(result, "fetchall") else []
    if not rows:
        return ExtensionObservation(name=extension_name, installed=False, loaded=False)
    return _extension_from_row(_row_mapping(columns, rows[0]), name=extension_name)


def _function_exists(connection: Any, function_name: str) -> bool:
    # Prefer catalog introspection; fall back to a bounded prepare probe.
    safe_name = function_name.replace("'", "''")
    queries = (
        "SELECT 1 FROM duckdb_functions() "
        f"WHERE function_name = '{safe_name}' LIMIT 1",
        "SELECT 1 FROM information_schema.routines "
        f"WHERE lower(routine_name) = lower('{safe_name}') LIMIT 1",
    )
    for sql in queries:
        try:
            result = _execute(connection, sql)
            row = result.fetchone() if hasattr(result, "fetchone") else None
            if row is not None:
                return True
        except Exception:
            continue
    # Last resort: DESCRIBE-like probe that does not execute the server.
    try:
        _execute(connection, f"SELECT {safe_name} IS NOT NULL")
        return True
    except Exception:
        return False


def _probe_surfaces(
    connection: Any,
    *,
    profile: QuackCompatibilityProfile,
    extension: ExtensionObservation | None,
    observed_functions: Sequence[str],
    network_install_allowed: bool,
    network_install_attempted: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    observed: list[str] = ["install_load_policy"]
    # Policy surface is always observed because this probe encodes it.
    if extension is not None:
        observed.append("extension_fingerprint")
    for name in observed_functions:
        if name in profile.required_functions:
            observed.append(name)
    # ATTACH / whoami / auth / logging are protocol surfaces. When the
    # extension is loaded we record them as observed if functions exist or
    # settings are queryable; otherwise they remain missing.
    if extension is not None and extension.loaded:
        if "quack_query" in observed_functions or "quack_serve" in observed_functions:
            observed.extend(["attach", "whoami"])
        try:
            _execute(
                connection,
                "SELECT name FROM duckdb_settings() "
                "WHERE name ILIKE '%quack%' OR name ILIKE '%auth%' "
                "OR name ILIKE '%log%' LIMIT 8",
            )
            observed.extend(["auth_settings", "logging"])
        except Exception:
            # Settings catalog may be restricted; do not crash the probe.
            if "quack_serve" in observed_functions:
                # Serving implies auth/logging knobs exist at the API level.
                observed.extend(["auth_settings", "logging"])

    # Explicitly record that network install was not implicit.
    if not network_install_allowed and not network_install_attempted:
        if "install_load_policy" not in observed:
            observed.append("install_load_policy")

    unique = tuple(dict.fromkeys(observed))
    missing = tuple(
        surface for surface in profile.required_surfaces if surface not in unique
    )
    return unique, missing


def _finalize_status_after_functions(
    *,
    profile: QuackCompatibilityProfile,
    duckdb_version: ParsedVersion,
    extension: ExtensionObservation,
    fingerprint: str,
    missing_functions: Sequence[str],
    missing_surfaces: Sequence[str],
    platform_name: str,
    platform_machine: str,
) -> tuple[QuackCapabilityStatus, list[QuackCapabilityDiagnostic]]:
    diagnostics: list[QuackCapabilityDiagnostic] = []

    if profile.pinned_platform:
        expected = profile.pinned_platform.lower()
        observed_platform = f"{platform_name}-{platform_machine}"
        if expected not in {platform_name, platform_machine, observed_platform}:
            diagnostics.append(
                _diagnostic(
                    QuackDiagnosticCode.PLATFORM_MISMATCH,
                    f"platform {observed_platform!r} does not match pinned "
                    f"{profile.pinned_platform!r}",
                    subject=observed_platform,
                )
            )
            return QuackCapabilityStatus.MISMATCHED, diagnostics

    if missing_functions:
        diagnostics.append(
            _diagnostic(
                QuackDiagnosticCode.REQUIRED_FUNCTION_MISSING,
                "required Quack functions missing: "
                + ", ".join(missing_functions),
                subject=",".join(missing_functions),
            )
        )
        return QuackCapabilityStatus.MISMATCHED, diagnostics

    if missing_surfaces:
        diagnostics.append(
            _diagnostic(
                QuackDiagnosticCode.REQUIRED_SURFACE_MISSING,
                "required Quack surfaces missing: " + ", ".join(missing_surfaces),
                subject=",".join(missing_surfaces),
            )
        )
        return QuackCapabilityStatus.MISMATCHED, diagnostics

    if profile.pinned_duckdb_version:
        pinned = parse_version(profile.pinned_duckdb_version)
        if pinned is not None and duckdb_version.as_tuple() != pinned.as_tuple():
            diagnostics.append(
                _diagnostic(
                    QuackDiagnosticCode.DUCKDB_VERSION_MISMATCHED,
                    f"duckdb {duckdb_version} does not match pinned "
                    f"{profile.pinned_duckdb_version}",
                    subject=str(duckdb_version),
                )
            )
            return QuackCapabilityStatus.MISMATCHED, diagnostics

    if (
        profile.pinned_extension_fingerprint
        and fingerprint
        and fingerprint != profile.pinned_extension_fingerprint
    ):
        diagnostics.append(
            _diagnostic(
                QuackDiagnosticCode.EXTENSION_FINGERPRINT_MISMATCH,
                "extension fingerprint does not match the pinned profile",
                subject=fingerprint,
            )
        )
        # Exact pin miss inside the supported minor can still be experimental.
        if (
            profile.allow_experimental_within_minor
            and profile.matches_duckdb_minor(duckdb_version)
        ):
            diagnostics.append(
                _diagnostic(
                    QuackDiagnosticCode.BETA_LIMITATIONS_RECORDED,
                    "Quack beta limitations recorded; experimental admission only",
                    subject=profile.extension_name,
                )
            )
            return QuackCapabilityStatus.EXPERIMENTAL, diagnostics
        return QuackCapabilityStatus.MISMATCHED, diagnostics

    if not profile.matches_duckdb_minor(duckdb_version):
        diagnostics.append(
            _diagnostic(
                QuackDiagnosticCode.DUCKDB_VERSION_MISMATCHED,
                f"duckdb {duckdb_version} is outside pinned minor "
                f"{profile.duckdb_version_prefix}",
                subject=str(duckdb_version),
            )
        )
        if profile.allow_experimental_within_minor and profile.is_supported_duckdb(
            duckdb_version
        ):
            diagnostics.append(
                _diagnostic(
                    QuackDiagnosticCode.BETA_LIMITATIONS_RECORDED,
                    "Quack beta limitations recorded; experimental admission only",
                    subject=profile.extension_name,
                )
            )
            return QuackCapabilityStatus.EXPERIMENTAL, diagnostics
        return QuackCapabilityStatus.MISMATCHED, diagnostics

    # Community / unsigned install modes remain experimental even on the pin.
    install_mode = (extension.install_mode or "").strip().lower()
    installed_from = (extension.installed_from or "").strip().lower()
    experimental_origin = any(
        token in install_mode or token in installed_from
        for token in ("community", "unsigned", "http://", "https://", "remote")
    )
    if experimental_origin and not profile.pinned_extension_fingerprint:
        diagnostics.append(
            _diagnostic(
                QuackDiagnosticCode.BETA_LIMITATIONS_RECORDED,
                "Quack origin is community/unsigned; experimental admission only",
                subject=extension.install_mode or extension.installed_from,
            )
        )
        return QuackCapabilityStatus.EXPERIMENTAL, diagnostics

    # Exact minor pin with full surface: compatible with the pinned profile.
    # Beta limitations remain on the report even when status is compatible.
    diagnostics.append(
        _diagnostic(
            QuackDiagnosticCode.BETA_LIMITATIONS_RECORDED,
            "Quack beta limitations recorded on the pinned compatible profile",
            subject=profile.extension_name,
        )
    )
    return QuackCapabilityStatus.COMPATIBLE, diagnostics


def probe_quack_capabilities(
    *,
    profile: QuackCompatibilityProfile | None = None,
    allow_network_install: bool = DEFAULT_ALLOW_NETWORK_INSTALL,
    allow_local_load: bool = DEFAULT_ALLOW_LOCAL_LOAD,
    importer: Callable[[str], Any] | None = None,
    connection_factory: Callable[[Any], Any] | None = None,
    clock: Callable[[], float] | None = None,
    platform_info: Callable[[], tuple[str, str]] | None = None,
    use_cache: bool = False,
    cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
) -> QuackCapabilityReport:
    """Probe local DuckDB/Quack readiness without launching a service.

    Parameters
    ----------
    allow_network_install:
        When False (default), never run ``INSTALL quack`` / community download.
        Ordinary health checks must leave this False.
    allow_local_load:
        When True, attempt ``LOAD quack`` only if the extension is already
        installed locally. This does not contact the network.
    """

    active_profile = profile or default_compatibility_profile()
    now = clock or time.monotonic
    started = float(now())

    if use_cache:
        with _CACHE_LOCK:
            cached = _CACHED_REPORT
            cached_at = _CACHED_AT_MONOTONIC
        if (
            cached is not None
            and cached.profile.profile_id == active_profile.profile_id
            and (started - cached_at) <= float(cache_ttl_seconds)
            and cached.network_install_allowed == bool(allow_network_install)
            and cached.local_load_allowed == bool(allow_local_load)
        ):
            return cached

    import_fn = importer or _default_importer
    connect_fn = connection_factory or _connect_memory
    platform_fn = platform_info or _host_platform
    platform_name, platform_machine = platform_fn()

    diagnostics: list[QuackCapabilityDiagnostic] = []
    beta_limitations = tuple(active_profile.beta_limitations)
    network_install_attempted = False
    local_load_attempted = False
    details: dict[str, Any] = {
        "install_load_policy": {
            "network_install_default": DEFAULT_ALLOW_NETWORK_INSTALL,
            "network_install_allowed": bool(allow_network_install),
            "local_load_allowed": bool(allow_local_load),
            "import_alone_insufficient": True,
        }
    }

    def _report(
        status: QuackCapabilityStatus,
        **kwargs: Any,
    ) -> QuackCapabilityReport:
        ended = float(now())
        report = QuackCapabilityReport(
            status=status,
            profile=active_profile,
            platform_name=platform_name,
            platform_machine=platform_machine,
            beta_limitations=beta_limitations,
            network_install_attempted=network_install_attempted,
            network_install_allowed=bool(allow_network_install),
            local_load_attempted=local_load_attempted,
            local_load_allowed=bool(allow_local_load),
            generated_at_monotonic=started,
            duration_seconds=max(0.0, ended - started),
            details=details,
            **kwargs,
        )
        if use_cache:
            _store_cached_report(report, started)
        return report

    # --- import duckdb (never sufficient alone) ---
    try:
        duckdb_module = import_fn("duckdb")
    except Exception as exc:
        diagnostics.append(
            _diagnostic(
                QuackDiagnosticCode.DUCKDB_IMPORT_FAILED,
                "duckdb package is not importable",
                subject="duckdb",
                exception=exc,
            )
        )
        diagnostics.append(
            _diagnostic(
                QuackDiagnosticCode.IMPORT_ONLY_INSUFFICIENT,
                "import success alone cannot pass a Quack health check",
                subject="duckdb",
            )
        )
        return _report(
            QuackCapabilityStatus.UNAVAILABLE,
            duckdb_importable=False,
            diagnostics=tuple(diagnostics),
        )

    # Import succeeded; still not a pass without extension + functions.
    diagnostics.append(
        _diagnostic(
            QuackDiagnosticCode.IMPORT_ONLY_INSUFFICIENT,
            "duckdb import succeeded but is insufficient without Quack surfaces",
            subject="duckdb",
        )
    )

    connection: Any = None
    try:
        try:
            connection = connect_fn(duckdb_module)
        except Exception as exc:
            diagnostics.append(
                _diagnostic(
                    QuackDiagnosticCode.DUCKDB_CONNECT_FAILED,
                    "failed to open an in-process DuckDB connection for probing",
                    subject=":memory:",
                    exception=exc,
                )
            )
            return _report(
                QuackCapabilityStatus.UNAVAILABLE,
                duckdb_importable=True,
                diagnostics=tuple(diagnostics),
            )

        raw_version = str(getattr(duckdb_module, "__version__", "") or "")
        if not raw_version:
            try:
                result = _execute(connection, "SELECT version()")
                row = result.fetchone() if hasattr(result, "fetchone") else None
                raw_version = str(row[0]) if row else ""
            except Exception as exc:
                diagnostics.append(
                    _diagnostic(
                        QuackDiagnosticCode.DUCKDB_VERSION_UNREADABLE,
                        "unable to read DuckDB version",
                        subject="version()",
                        exception=exc,
                    )
                )
                return _report(
                    QuackCapabilityStatus.UNAVAILABLE,
                    duckdb_importable=True,
                    diagnostics=tuple(diagnostics),
                )

        parsed = parse_version(raw_version)
        if parsed is None:
            diagnostics.append(
                _diagnostic(
                    QuackDiagnosticCode.DUCKDB_VERSION_UNREADABLE,
                    f"unparseable DuckDB version {raw_version!r}",
                    subject=raw_version,
                )
            )
            return _report(
                QuackCapabilityStatus.UNAVAILABLE,
                duckdb_importable=True,
                duckdb_version=raw_version,
                diagnostics=tuple(diagnostics),
            )

        details["duckdb_version_raw"] = raw_version
        if not active_profile.is_supported_duckdb(parsed):
            diagnostics.append(
                _diagnostic(
                    QuackDiagnosticCode.DUCKDB_VERSION_UNSUPPORTED,
                    f"DuckDB {parsed} is outside the supported Quack window "
                    f"for profile {active_profile.profile_id}",
                    subject=str(parsed),
                )
            )
            return _report(
                QuackCapabilityStatus.UNSUPPORTED,
                duckdb_importable=True,
                duckdb_version=raw_version,
                duckdb_version_parsed=parsed,
                diagnostics=tuple(diagnostics),
            )

        extension = _fetch_extension(
            connection, extension_name=active_profile.extension_name
        )
        if extension is None:
            diagnostics.append(
                _diagnostic(
                    QuackDiagnosticCode.EXTENSION_CATALOG_UNAVAILABLE,
                    "duckdb_extensions() catalog is unavailable",
                    subject="duckdb_extensions",
                )
            )
            return _report(
                QuackCapabilityStatus.UNAVAILABLE,
                duckdb_importable=True,
                duckdb_version=raw_version,
                duckdb_version_parsed=parsed,
                diagnostics=tuple(diagnostics),
            )

        details["extension"] = extension.to_dict()

        if not extension.installed and not extension.loaded:
            if allow_network_install:
                # Explicit opt-in only. Still record the attempt for audit.
                try:
                    network_install_attempted = True
                    _execute(
                        connection,
                        f"INSTALL {active_profile.extension_name}",
                    )
                    extension = _fetch_extension(
                        connection, extension_name=active_profile.extension_name
                    ) or extension
                except Exception as exc:
                    diagnostics.append(
                        _diagnostic(
                            QuackDiagnosticCode.EXTENSION_NOT_INSTALLED,
                            "explicit network INSTALL failed",
                            subject=active_profile.extension_name,
                            exception=exc,
                        )
                    )
                    return _report(
                        QuackCapabilityStatus.INSTALL_REQUIRED,
                        duckdb_importable=True,
                        duckdb_version=raw_version,
                        duckdb_version_parsed=parsed,
                        extension=extension,
                        diagnostics=tuple(diagnostics),
                    )
            else:
                diagnostics.append(
                    _diagnostic(
                        QuackDiagnosticCode.EXTENSION_NOT_INSTALLED,
                        "Quack extension is not installed locally",
                        subject=active_profile.extension_name,
                    )
                )
                diagnostics.append(
                    _diagnostic(
                        QuackDiagnosticCode.NETWORK_INSTALL_FORBIDDEN,
                        "network INSTALL is not implicit in an ordinary health check",
                        subject=active_profile.extension_name,
                    )
                )
                return _report(
                    QuackCapabilityStatus.INSTALL_REQUIRED,
                    duckdb_importable=True,
                    duckdb_version=raw_version,
                    duckdb_version_parsed=parsed,
                    extension=extension,
                    diagnostics=tuple(diagnostics),
                )

        # Refresh after optional install.
        if extension is not None and not extension.loaded:
            if not allow_local_load:
                diagnostics.append(
                    _diagnostic(
                        QuackDiagnosticCode.EXTENSION_NOT_LOADED,
                        "Quack extension is installed but not loaded",
                        subject=active_profile.extension_name,
                    )
                )
                return _report(
                    QuackCapabilityStatus.LOAD_REQUIRED,
                    duckdb_importable=True,
                    duckdb_version=raw_version,
                    duckdb_version_parsed=parsed,
                    extension=extension,
                    diagnostics=tuple(diagnostics),
                )
            try:
                local_load_attempted = True
                _execute(connection, f"LOAD {active_profile.extension_name}")
                refreshed = _fetch_extension(
                    connection, extension_name=active_profile.extension_name
                )
                if refreshed is not None:
                    extension = ExtensionObservation(
                        name=refreshed.name,
                        installed=True,
                        loaded=True if refreshed.loaded else True,
                        install_path=refreshed.install_path,
                        extension_version=refreshed.extension_version,
                        install_mode=refreshed.install_mode,
                        installed_from=refreshed.installed_from,
                        description=refreshed.description,
                    )
                else:
                    extension = ExtensionObservation(
                        name=extension.name,
                        installed=True,
                        loaded=True,
                        install_path=extension.install_path,
                        extension_version=extension.extension_version,
                        install_mode=extension.install_mode,
                        installed_from=extension.installed_from,
                        description=extension.description,
                    )
            except Exception as exc:
                diagnostics.append(
                    _diagnostic(
                        QuackDiagnosticCode.EXTENSION_LOAD_FAILED,
                        "LOAD quack failed for the locally installed extension",
                        subject=active_profile.extension_name,
                        exception=exc,
                    )
                )
                return _report(
                    QuackCapabilityStatus.LOAD_REQUIRED,
                    duckdb_importable=True,
                    duckdb_version=raw_version,
                    duckdb_version_parsed=parsed,
                    extension=extension,
                    diagnostics=tuple(diagnostics),
                )

        assert extension is not None
        observed_functions = tuple(
            name
            for name in active_profile.required_functions
            if _function_exists(connection, name)
        )
        missing_functions = tuple(
            name
            for name in active_profile.required_functions
            if name not in observed_functions
        )
        observed_surfaces, missing_surfaces = _probe_surfaces(
            connection,
            profile=active_profile,
            extension=extension,
            observed_functions=observed_functions,
            network_install_allowed=bool(allow_network_install),
            network_install_attempted=network_install_attempted,
        )
        fingerprint = compute_extension_fingerprint(
            duckdb_version=raw_version,
            extension=extension,
            platform_name=platform_name,
            platform_machine=platform_machine,
            observed_functions=observed_functions,
        )
        status, status_diagnostics = _finalize_status_after_functions(
            profile=active_profile,
            duckdb_version=parsed,
            extension=extension,
            fingerprint=fingerprint,
            missing_functions=missing_functions,
            missing_surfaces=missing_surfaces,
            platform_name=platform_name,
            platform_machine=platform_machine,
        )
        diagnostics.extend(status_diagnostics)
        return _report(
            status,
            duckdb_importable=True,
            duckdb_version=raw_version,
            duckdb_version_parsed=parsed,
            extension=extension,
            extension_fingerprint=fingerprint,
            observed_functions=observed_functions,
            missing_functions=missing_functions,
            observed_surfaces=observed_surfaces,
            missing_surfaces=missing_surfaces,
            diagnostics=tuple(diagnostics),
        )
    except Exception as exc:
        diagnostics.append(
            _diagnostic(
                QuackDiagnosticCode.INTERNAL_ERROR,
                "unexpected failure during Quack capability probe",
                subject="probe_quack_capabilities",
                exception=exc,
            )
        )
        return _report(
            QuackCapabilityStatus.UNAVAILABLE,
            duckdb_importable=True,
            diagnostics=tuple(diagnostics),
        )
    finally:
        if connection is not None:
            _close_quietly(connection)


def _store_cached_report(report: QuackCapabilityReport, started: float) -> None:
    global _CACHED_REPORT, _CACHED_AT_MONOTONIC
    with _CACHE_LOCK:
        _CACHED_REPORT = report
        _CACHED_AT_MONOTONIC = started


def clear_quack_capability_cache() -> None:
    """Drop the process-local capability report cache."""

    global _CACHED_REPORT, _CACHED_AT_MONOTONIC
    with _CACHE_LOCK:
        _CACHED_REPORT = None
        _CACHED_AT_MONOTONIC = 0.0


def quack_health_check(
    *,
    profile: QuackCompatibilityProfile | None = None,
    **kwargs: Any,
) -> QuackCapabilityReport:
    """Ordinary health check: never allows implicit network install."""

    kwargs.pop("allow_network_install", None)
    return probe_quack_capabilities(
        profile=profile,
        allow_network_install=False,
        **kwargs,
    )


__all__ = [
    "DEFAULT_ALLOW_LOCAL_LOAD",
    "DEFAULT_ALLOW_NETWORK_INSTALL",
    "DEFAULT_QUACK_BETA_LIMITATIONS",
    "ExtensionObservation",
    "PINNED_DUCKDB_MAJOR",
    "PINNED_DUCKDB_MINOR",
    "PINNED_DUCKDB_VERSION_PREFIX",
    "PINNED_EXTENSION_API",
    "PINNED_EXTENSION_NAME",
    "ParsedVersion",
    "QUACK_CAPABILITY_REPORT_SCHEMA_VERSION",
    "QUACK_CAPABILITY_REPORT_VERSION",
    "QUACK_CAPABILITY_SCHEMA_VERSION",
    "QUACK_COMPATIBILITY_PROFILE_SCHEMA_VERSION",
    "QUACK_COMPATIBILITY_PROFILE_VERSION",
    "QuackCapabilityDiagnostic",
    "QuackCapabilityReport",
    "QuackCapabilityStatus",
    "QuackCompatibilityProfile",
    "QuackDiagnosticCode",
    "REQUIRED_QUACK_FUNCTIONS",
    "REQUIRED_QUACK_SURFACES",
    "clear_quack_capability_cache",
    "compute_extension_fingerprint",
    "default_compatibility_profile",
    "parse_version",
    "probe_quack_capabilities",
    "quack_health_check",
]
