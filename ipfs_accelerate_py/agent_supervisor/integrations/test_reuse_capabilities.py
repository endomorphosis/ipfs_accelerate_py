"""Cold, bounded capability discovery for proof-backed test reuse.

This module reports whether optional test-reuse integrations are discoverable.
It deliberately does *not* import an optional provider, connect to IPFS or a
prover, start a daemon, install a package, create a cache, or validate a proof.
Consequently an ``available`` fact means only that the configured cold
prerequisites were found; the reuse decision must still validate every receipt
and certificate locally.

All capabilities in this report are optional.  A non-available fact therefore
means "execute the test" and is never, by itself, a startup error or a reason to
skip the test.
"""

from __future__ import annotations

import importlib.machinery
import math
import os
import shutil
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

TEST_REUSE_CAPABILITY_REPORT_SCHEMA = "TestReuseCapabilityReport@1"
TEST_REUSE_CAPABILITY_SCHEMA_VERSION = TEST_REUSE_CAPABILITY_REPORT_SCHEMA
TEST_REUSE_CAPABILITY_REPORT_VERSION = 1
DEFAULT_TEST_REUSE_CAPABILITY_TIMEOUT_SECONDS = 0.5
DEFAULT_TEST_REUSE_CAPABILITY_MAX_CHECKS = 48


class TestReuseCapabilityStatus(str, Enum):
    """Exhaustive outcome of one cold capability probe."""

    AVAILABLE = "available"
    DISABLED = "disabled"
    MISSING = "missing"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


class TestReuseCapabilityName(str, Enum):
    """Stable capability identifiers in report order."""

    MULTIFORMATS = "multiformats"
    DATASETS_ZK = "datasets_zk"
    GROTH16 = "groth16"
    PROVEKIT = "provekit"
    CACHE = "cache"
    IPFS = "ipfs"
    LOCAL_VERIFIER = "local_verifier"


class TestReuseCapabilityEvidenceKind(str, Enum):
    """Cold evidence sources; none execute the discovered component."""

    IMPORT_SPEC = "import_spec"
    EXECUTABLE = "executable"
    CONFIGURED_PATH = "configured_path"
    BACKEND_REGISTRY = "backend_registry"
    CAPABILITY_METADATA = "capability_metadata"
    CONFIGURATION = "configuration"


_CAPABILITY_ORDER = tuple(TestReuseCapabilityName)
_DISABLED_VALUES = frozenset({"0", "disabled", "false", "no", "off"})
_ENABLED_VALUES = frozenset({"1", "enabled", "on", "true", "yes"})
_METADATA_FIELDS = (
    "schema_version",
    "schema",
    "compatible",
    "enabled",
    "status",
    "reason_code",
    "available",
)
_REGISTRY_FIELDS = (
    "enabled",
    "api_version",
    "version",
    "compatible",
    "available",
)
_MAX_REASON_CODE_LENGTH = 128


@dataclass(frozen=True)
class TestReuseCapabilityEvidence:
    """One bounded, serialization-safe observation made by the probe."""

    kind: TestReuseCapabilityEvidenceKind
    subject: str
    present: bool | None
    compatible: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, TestReuseCapabilityEvidenceKind):
            object.__setattr__(self, "kind", TestReuseCapabilityEvidenceKind(self.kind))
        subject = str(self.subject).strip()
        if not subject:
            raise ValueError("capability evidence subject must not be empty")
        if self.present is not None and not isinstance(self.present, bool):
            raise ValueError("capability evidence present must be bool or None")
        if self.compatible is not None and not isinstance(self.compatible, bool):
            raise ValueError("capability evidence compatible must be bool or None")
        object.__setattr__(self, "subject", subject)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "subject": self.subject,
            "present": self.present,
            "compatible": self.compatible,
        }


@dataclass(frozen=True)
class TestReuseCapability:
    """Typed cold fact for one optional test-reuse integration."""

    capability_id: TestReuseCapabilityName
    status: TestReuseCapabilityStatus
    reason_code: str
    evidence: tuple[TestReuseCapabilityEvidence, ...] = ()
    optional: bool = field(default=True, init=False)
    blocking: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.capability_id, TestReuseCapabilityName):
            object.__setattr__(self, "capability_id", TestReuseCapabilityName(self.capability_id))
        if not isinstance(self.status, TestReuseCapabilityStatus):
            object.__setattr__(self, "status", TestReuseCapabilityStatus(self.status))
        reason_code = str(self.reason_code).strip()
        if not reason_code:
            raise ValueError("capability reason_code must not be empty")
        evidence = tuple(self.evidence)
        if any(not isinstance(item, TestReuseCapabilityEvidence) for item in evidence):
            raise ValueError("capability evidence must contain typed evidence facts")
        object.__setattr__(self, "reason_code", reason_code)
        object.__setattr__(self, "evidence", evidence)

    @property
    def name(self) -> str:
        return self.capability_id.value

    @property
    def available(self) -> bool:
        return self.status is TestReuseCapabilityStatus.AVAILABLE

    @property
    def non_blocking(self) -> bool:
        return True

    @property
    def test_action(self) -> str:
        """Unavailable optional infrastructure always falls through to RUN."""

        return "continue" if self.available else "run"

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id.value,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "available": self.available,
            "optional": True,
            "blocking": False,
            "test_action": self.test_action,
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True)
class TestReuseCapabilityReport:
    """Deterministic versioned snapshot of all optional reuse capabilities."""

    capabilities: tuple[TestReuseCapability, ...]
    probe_count: int
    mode: str | None = None
    schema_version: str = TEST_REUSE_CAPABILITY_REPORT_SCHEMA
    report_version: int = TEST_REUSE_CAPABILITY_REPORT_VERSION
    bounded: bool = field(default=True, init=False)
    lazy: bool = field(default=True, init=False)
    side_effect_free: bool = field(default=True, init=False)
    network_attempted: bool = field(default=False, init=False)
    daemon_started: bool = field(default=False, init=False)
    cache_created: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        capabilities = tuple(self.capabilities)
        ids = tuple(item.capability_id for item in capabilities)
        if ids != _CAPABILITY_ORDER:
            raise ValueError("test-reuse capabilities must appear once in stable contract order")
        if (
            isinstance(self.probe_count, bool)
            or not isinstance(self.probe_count, int)
            or self.probe_count < 0
        ):
            raise ValueError("probe_count must be a non-negative integer")
        mode = str(self.mode).strip().lower() if self.mode is not None else None
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "mode", mode or None)

    @property
    def facts(self) -> Mapping[str, TestReuseCapability]:
        return {item.capability_id.value: item for item in self.capabilities}

    @property
    def all_optional(self) -> bool:
        return all(item.optional and not item.blocking for item in self.capabilities)

    @property
    def unavailable_is_non_blocking(self) -> bool:
        return all(
            item.available or (not item.blocking and item.test_action == "run")
            for item in self.capabilities
        )

    def capability(self, capability_id: TestReuseCapabilityName | str) -> TestReuseCapability:
        key = (
            capability_id.value
            if isinstance(capability_id, TestReuseCapabilityName)
            else str(capability_id)
        )
        try:
            return self.facts[key]
        except KeyError as exc:
            raise KeyError(f"unknown test-reuse capability: {key}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "report_version": self.report_version,
            "mode": self.mode,
            "probe_count": self.probe_count,
            "bounded": True,
            "lazy": True,
            "side_effect_free": True,
            "network_attempted": False,
            "daemon_started": False,
            "cache_created": False,
            "all_optional": self.all_optional,
            "unavailable_is_non_blocking": self.unavailable_is_non_blocking,
            "capabilities": {
                item.capability_id.value: item.to_dict() for item in self.capabilities
            },
        }


@dataclass(frozen=True)
class TestReuseCapabilityProbeConfig:
    """Strict bounds and optional cold configuration paths for one snapshot."""

    timeout_seconds: float = DEFAULT_TEST_REUSE_CAPABILITY_TIMEOUT_SECONDS
    max_checks: int = DEFAULT_TEST_REUSE_CAPABILITY_MAX_CHECKS
    disabled_capabilities: frozenset[TestReuseCapabilityName | str] = frozenset()
    groth16_artifacts_path: str | None = None
    provekit_artifacts_path: str | None = None
    cache_path: str | None = None
    local_verifier_key_path: str | None = None
    local_verifier_circuit_path: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(float(self.timeout_seconds))
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        if (
            isinstance(self.max_checks, bool)
            or not isinstance(self.max_checks, int)
            or self.max_checks < 1
        ):
            raise ValueError("max_checks must be a positive integer")
        disabled = frozenset(
            item
            if isinstance(item, TestReuseCapabilityName)
            else TestReuseCapabilityName(str(item))
            for item in self.disabled_capabilities
        )
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))
        object.__setattr__(self, "disabled_capabilities", disabled)
        for name in (
            "groth16_artifacts_path",
            "provekit_artifacts_path",
            "cache_path",
            "local_verifier_key_path",
            "local_verifier_circuit_path",
        ):
            value = getattr(self, name)
            if value is not None:
                normalized = str(value).strip()
                if not normalized:
                    raise ValueError(f"{name} must not be empty when configured")
                object.__setattr__(self, name, normalized)


PackageFinder = Callable[[str], Any]
ExecutableFinder = Callable[[str], str | None]
PathPredicate = Callable[[str], bool]


def _find_spec_without_import(module: str) -> Any:
    """Resolve dotted modules without executing any package initializer."""

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
            resolved = tuple(str(location) for location in locations)
            package_name = parts[index]
            nested = tuple(
                str(candidate)
                for location in resolved
                for candidate in (Path(location) / package_name,)
                if Path(location).name == package_name
                and candidate.is_dir()
                and (candidate / "__init__.py").is_file()
            )
            path = tuple(dict.fromkeys((*nested, *resolved)))
    return spec


@dataclass
class _ProbeBudget:
    timeout_seconds: float
    max_checks: int
    monotonic: Callable[[], float]
    started: float = field(init=False)
    deadline: float = field(init=False)
    count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.started = self.monotonic()
        self.deadline = self.started + self.timeout_seconds

    def call(
        self, function: Callable[..., Any], *args: Any
    ) -> tuple[TestReuseCapabilityStatus, Any]:
        if self.count >= self.max_checks or self.monotonic() >= self.deadline:
            return TestReuseCapabilityStatus.UNKNOWN, "probe_budget_exhausted"
        self.count += 1
        remaining = self.deadline - self.monotonic()
        if remaining <= 0:
            return TestReuseCapabilityStatus.UNKNOWN, "probe_budget_exhausted"

        outcome: list[tuple[bool, Any]] = []

        def invoke() -> None:
            try:
                outcome.append((True, function(*args)))
            except BaseException:
                # Optional discovery hooks are an untrusted boundary.  Do not
                # leak their exception text into a deterministic report.
                outcome.append((False, None))

        worker = threading.Thread(
            target=invoke,
            name="test-reuse-cold-capability-probe",
            daemon=True,
        )
        worker.start()
        worker.join(remaining)
        if worker.is_alive():
            return TestReuseCapabilityStatus.UNKNOWN, "probe_timed_out"
        if not outcome or not outcome[0][0]:
            return TestReuseCapabilityStatus.UNKNOWN, "probe_failed"
        return TestReuseCapabilityStatus.AVAILABLE, outcome[0][1]


@dataclass(frozen=True)
class _Check:
    status: TestReuseCapabilityStatus
    reason_code: str
    evidence: tuple[TestReuseCapabilityEvidence, ...]


@dataclass(frozen=True)
class _MappingEntrySnapshot:
    """Bounded projection of one untrusted mapping entry."""

    key_present: bool
    value_present: bool
    value_is_mapping: bool
    values: tuple[tuple[str, Any], ...] = ()


def _snapshot_mapping_entry(
    source: Mapping[str, Any], key: str, fields: Sequence[str]
) -> _MappingEntrySnapshot:
    """Read one provider entry in the worker used by :class:`_ProbeBudget`.

    Only the fixed contract fields are retained.  In particular, the probe
    never iterates or copies a provider registry and never allows a custom
    mapping implementation to escape the timeout/error boundary.
    """

    try:
        raw = source[key]
    except KeyError:
        return _MappingEntrySnapshot(False, False, False)
    if not isinstance(raw, Mapping):
        return _MappingEntrySnapshot(True, raw is not None, False)

    values: list[tuple[str, Any]] = []
    for field_name in fields:
        try:
            value = raw[field_name]
        except KeyError:
            continue
        values.append((field_name, value))
    return _MappingEntrySnapshot(True, True, True, tuple(values))


def _read_environment_entry(source: Mapping[str, str], key: str) -> str:
    """Read one environment value without coercing provider-owned objects."""

    try:
        raw = source[key]
    except KeyError:
        return ""
    if type(raw) is not str:
        raise TypeError("environment values must be strings")
    return raw.strip()


def _metadata_reason(values: Mapping[str, Any], fallback: str) -> str:
    """Return a bounded, serialization-safe provider reason code."""

    value = values.get("reason_code")
    if type(value) is not str:
        return fallback
    normalized = value.strip()
    if not normalized or len(normalized) > _MAX_REASON_CODE_LENGTH:
        return fallback
    if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
        return fallback
    return normalized


class TestReuseCapabilityProbe:
    """Perform one cold snapshot; construction itself performs no discovery."""

    def __init__(
        self,
        config: TestReuseCapabilityProbeConfig | None = None,
        *,
        find_spec: PackageFinder | None = None,
        which: ExecutableFinder | None = None,
        path_is_file: PathPredicate | None = None,
        path_is_dir: PathPredicate | None = None,
        environ: Mapping[str, str] | None = None,
        backend_registry: Mapping[str, Any] | None = None,
        capability_metadata: Mapping[str, Any] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self.config = config if config is not None else TestReuseCapabilityProbeConfig()
        self._find_spec = find_spec if find_spec is not None else _find_spec_without_import
        self._which = which if which is not None else shutil.which
        self._path_is_file = (
            path_is_file if path_is_file is not None else lambda value: Path(value).is_file()
        )
        self._path_is_dir = (
            path_is_dir if path_is_dir is not None else lambda value: Path(value).is_dir()
        )
        self._environ = environ if environ is not None else os.environ
        # Plain dictionaries and the process environment have local, constant
        # time access.  Other Mapping implementations are provider boundaries
        # and must be read through the probe budget.
        self._trusted_environ = environ is None or type(environ) is dict
        # Retain injected mappings by reference.  Copying or testing their
        # truthiness can execute user-defined discovery during construction,
        # which would violate the probe's lazy boundary.
        self._backend_registry = backend_registry
        self._metadata = capability_metadata
        self._monotonic = monotonic if monotonic is not None else time.monotonic

    def probe(self) -> TestReuseCapabilityReport:
        """Return one bounded report without retaining or creating a cache."""

        budget = _ProbeBudget(self.config.timeout_seconds, self.config.max_checks, self._monotonic)
        mode_status, mode_value = self._environment_value(
            "IPFS_TEST_PROOF_REUSE_MODE", budget
        )
        if mode_status is TestReuseCapabilityStatus.UNKNOWN:
            facts = tuple(
                self._fact(
                    capability_id,
                    mode_status,
                    mode_value,
                    (
                        TestReuseCapabilityEvidence(
                            TestReuseCapabilityEvidenceKind.CONFIGURATION,
                            "IPFS_TEST_PROOF_REUSE_MODE",
                            None,
                        ),
                    ),
                )
                for capability_id in _CAPABILITY_ORDER
            )
            return TestReuseCapabilityReport(facts, probe_count=budget.count)

        mode = mode_value.lower() or None
        if mode in _DISABLED_VALUES:
            facts = tuple(
                self._fact(
                    capability_id,
                    TestReuseCapabilityStatus.DISABLED,
                    "proof_reuse_disabled",
                    (
                        TestReuseCapabilityEvidence(
                            TestReuseCapabilityEvidenceKind.CONFIGURATION,
                            "IPFS_TEST_PROOF_REUSE_MODE",
                            True,
                        ),
                    ),
                )
                for capability_id in _CAPABILITY_ORDER
            )
            return TestReuseCapabilityReport(facts, probe_count=budget.count, mode=mode)

        facts = tuple(self._probe_one(item, budget) for item in _CAPABILITY_ORDER)
        return TestReuseCapabilityReport(facts, probe_count=budget.count, mode=mode)

    def _probe_one(
        self, capability_id: TestReuseCapabilityName, budget: _ProbeBudget
    ) -> TestReuseCapability:
        if capability_id in self.config.disabled_capabilities:
            return self._fact(
                capability_id,
                TestReuseCapabilityStatus.DISABLED,
                "capability_disabled",
                (),
            )
        environment_disabled = self._env_disabled(capability_id, budget)
        if isinstance(environment_disabled, _Check):
            return self._fact(
                capability_id,
                environment_disabled.status,
                environment_disabled.reason_code,
                environment_disabled.evidence,
            )
        if environment_disabled:
            return self._fact(
                capability_id,
                TestReuseCapabilityStatus.DISABLED,
                "capability_disabled",
                (),
            )
        override = self._override(capability_id, budget)
        if override is not None:
            return override

        probes: dict[TestReuseCapabilityName, Callable[[_ProbeBudget], _Check]] = {
            TestReuseCapabilityName.MULTIFORMATS: self._probe_multiformats,
            TestReuseCapabilityName.DATASETS_ZK: self._probe_datasets_zk,
            TestReuseCapabilityName.GROTH16: self._probe_groth16,
            TestReuseCapabilityName.PROVEKIT: self._probe_provekit,
            TestReuseCapabilityName.CACHE: self._probe_cache,
            TestReuseCapabilityName.IPFS: self._probe_ipfs,
            TestReuseCapabilityName.LOCAL_VERIFIER: self._probe_local_verifier,
        }
        check = probes[capability_id](budget)
        return self._fact(capability_id, check.status, check.reason_code, check.evidence)

    @staticmethod
    def _fact(
        capability_id: TestReuseCapabilityName,
        status: TestReuseCapabilityStatus,
        reason_code: str,
        evidence: tuple[TestReuseCapabilityEvidence, ...],
    ) -> TestReuseCapability:
        return TestReuseCapability(capability_id, status, reason_code, evidence)

    def _environment_value(
        self, environment_name: str, budget: _ProbeBudget
    ) -> tuple[TestReuseCapabilityStatus, str]:
        if self._trusted_environ:
            try:
                return (
                    TestReuseCapabilityStatus.AVAILABLE,
                    _read_environment_entry(self._environ, environment_name),
                )
            except BaseException:
                return TestReuseCapabilityStatus.UNKNOWN, "probe_failed"
        status, value = budget.call(
            _read_environment_entry,
            self._environ,
            environment_name,
        )
        return status, str(value)

    def _env_disabled(
        self, capability_id: TestReuseCapabilityName, budget: _ProbeBudget
    ) -> bool | _Check:
        label = capability_id.value.upper()
        disable_name = f"IPFS_TEST_PROOF_REUSE_DISABLE_{label}"
        enabled_name = f"IPFS_TEST_PROOF_REUSE_{label}_ENABLED"
        status, disable = self._environment_value(disable_name, budget)
        if status is TestReuseCapabilityStatus.UNKNOWN:
            return _Check(
                status,
                disable,
                (
                    TestReuseCapabilityEvidence(
                        TestReuseCapabilityEvidenceKind.CONFIGURATION,
                        disable_name,
                        None,
                    ),
                ),
            )
        status, enabled = self._environment_value(enabled_name, budget)
        if status is TestReuseCapabilityStatus.UNKNOWN:
            return _Check(
                status,
                enabled,
                (
                    TestReuseCapabilityEvidence(
                        TestReuseCapabilityEvidenceKind.CONFIGURATION,
                        enabled_name,
                        None,
                    ),
                ),
            )
        disable = disable.lower()
        enabled = enabled.lower()
        return disable in _ENABLED_VALUES or enabled in _DISABLED_VALUES

    def _override(
        self, capability_id: TestReuseCapabilityName, budget: _ProbeBudget
    ) -> TestReuseCapability | None:
        if self._metadata is None:
            return None

        status, snapshot = budget.call(
            _snapshot_mapping_entry,
            self._metadata,
            capability_id.value,
            _METADATA_FIELDS,
        )
        if status is TestReuseCapabilityStatus.UNKNOWN:
            return self._fact(
                capability_id,
                status,
                str(snapshot),
                (
                    TestReuseCapabilityEvidence(
                        TestReuseCapabilityEvidenceKind.CAPABILITY_METADATA,
                        capability_id.value,
                        None,
                    ),
                ),
            )
        if not snapshot.key_present:
            return None

        evidence = (
            TestReuseCapabilityEvidence(
                TestReuseCapabilityEvidenceKind.CAPABILITY_METADATA,
                capability_id.value,
                snapshot.value_present,
            ),
        )
        if not snapshot.value_is_mapping:
            return self._fact(
                capability_id,
                TestReuseCapabilityStatus.INCOMPATIBLE,
                "capability_metadata_incompatible",
                evidence,
            )
        values = dict(snapshot.values)
        schema = values.get("schema_version", values.get("schema"))
        if schema is not None and (
            type(schema) is not str
            or schema
            not in ("TestReuseCapability@1", TEST_REUSE_CAPABILITY_REPORT_SCHEMA)
        ):
            return self._fact(
                capability_id,
                TestReuseCapabilityStatus.INCOMPATIBLE,
                "capability_metadata_incompatible",
                evidence,
            )
        if "compatible" in values and not isinstance(values["compatible"], bool):
            return self._fact(
                capability_id,
                TestReuseCapabilityStatus.INCOMPATIBLE,
                "capability_metadata_incompatible",
                evidence,
            )
        if values.get("compatible") is False:
            return self._fact(
                capability_id,
                TestReuseCapabilityStatus.INCOMPATIBLE,
                "capability_incompatible",
                evidence,
            )
        if "enabled" in values and not isinstance(values["enabled"], bool):
            return self._fact(
                capability_id,
                TestReuseCapabilityStatus.INCOMPATIBLE,
                "capability_metadata_incompatible",
                evidence,
            )
        if values.get("enabled") is False:
            return self._fact(
                capability_id,
                TestReuseCapabilityStatus.DISABLED,
                "capability_disabled",
                evidence,
            )
        status_value = values.get("status")
        if status_value is not None:
            if type(status_value) is not str:
                return self._fact(
                    capability_id,
                    TestReuseCapabilityStatus.INCOMPATIBLE,
                    "capability_metadata_incompatible",
                    evidence,
                )
            try:
                status = TestReuseCapabilityStatus(status_value.lower())
            except ValueError:
                status = TestReuseCapabilityStatus.INCOMPATIBLE
            return self._fact(
                capability_id,
                status,
                _metadata_reason(values, f"capability_{status.value}"),
                evidence,
            )
        if "available" in values and not isinstance(values["available"], bool):
            return self._fact(
                capability_id,
                TestReuseCapabilityStatus.INCOMPATIBLE,
                "capability_metadata_incompatible",
                evidence,
            )
        if values.get("available") is True:
            return self._fact(
                capability_id,
                TestReuseCapabilityStatus.AVAILABLE,
                _metadata_reason(values, "cold_metadata_available"),
                evidence,
            )
        if values.get("available") is False:
            return self._fact(
                capability_id,
                TestReuseCapabilityStatus.MISSING,
                _metadata_reason(values, "cold_metadata_missing"),
                evidence,
            )
        return self._fact(
            capability_id,
            TestReuseCapabilityStatus.UNKNOWN,
            "capability_metadata_indeterminate",
            evidence,
        )

    def _module(self, module: str, budget: _ProbeBudget) -> _Check:
        status, value = budget.call(self._find_spec, module)
        if status is TestReuseCapabilityStatus.UNKNOWN:
            return _Check(
                status,
                str(value),
                (
                    TestReuseCapabilityEvidence(
                        TestReuseCapabilityEvidenceKind.IMPORT_SPEC,
                        module,
                        None,
                    ),
                ),
            )
        present = value is not None
        return _Check(
            TestReuseCapabilityStatus.AVAILABLE if present else TestReuseCapabilityStatus.MISSING,
            "import_spec_available" if present else "import_spec_missing",
            (
                TestReuseCapabilityEvidence(
                    TestReuseCapabilityEvidenceKind.IMPORT_SPEC, module, present
                ),
            ),
        )

    def _executable(
        self,
        candidates: Sequence[str],
        environment_names: Sequence[str],
        budget: _ProbeBudget,
    ) -> _Check:
        for environment_name in environment_names:
            status, configured = self._environment_value(environment_name, budget)
            if status is TestReuseCapabilityStatus.UNKNOWN:
                return _Check(
                    status,
                    configured,
                    (
                        TestReuseCapabilityEvidence(
                            TestReuseCapabilityEvidenceKind.CONFIGURATION,
                            environment_name,
                            None,
                        ),
                    ),
                )
            if not configured:
                continue
            status, value = budget.call(self._path_is_file, configured)
            if status is TestReuseCapabilityStatus.UNKNOWN:
                return _Check(
                    status,
                    str(value),
                    (
                        TestReuseCapabilityEvidence(
                            TestReuseCapabilityEvidenceKind.CONFIGURED_PATH,
                            environment_name,
                            None,
                        ),
                    ),
                )
            return _Check(
                TestReuseCapabilityStatus.AVAILABLE if value else TestReuseCapabilityStatus.MISSING,
                "configured_executable_available" if value else "configured_executable_missing",
                (
                    TestReuseCapabilityEvidence(
                        TestReuseCapabilityEvidenceKind.CONFIGURED_PATH,
                        environment_name,
                        bool(value),
                    ),
                ),
            )

        evidence: list[TestReuseCapabilityEvidence] = []
        for candidate in candidates:
            status, value = budget.call(self._which, candidate)
            if status is TestReuseCapabilityStatus.UNKNOWN:
                evidence.append(
                    TestReuseCapabilityEvidence(
                        TestReuseCapabilityEvidenceKind.EXECUTABLE, candidate, None
                    )
                )
                return _Check(status, str(value), tuple(evidence))
            present = bool(value)
            evidence.append(
                TestReuseCapabilityEvidence(
                    TestReuseCapabilityEvidenceKind.EXECUTABLE, candidate, present
                )
            )
            if present:
                return _Check(
                    TestReuseCapabilityStatus.AVAILABLE,
                    "executable_available",
                    tuple(evidence),
                )
        return _Check(
            TestReuseCapabilityStatus.MISSING,
            "executable_missing",
            tuple(evidence),
        )

    def _configured_directory(
        self,
        configured: str | None,
        environment_names: Sequence[str],
        budget: _ProbeBudget,
    ) -> _Check | None:
        subject = "probe_config"
        value = configured
        if value is None:
            for environment_name in environment_names:
                status, candidate = self._environment_value(environment_name, budget)
                if status is TestReuseCapabilityStatus.UNKNOWN:
                    return _Check(
                        status,
                        candidate,
                        (
                            TestReuseCapabilityEvidence(
                                TestReuseCapabilityEvidenceKind.CONFIGURATION,
                                environment_name,
                                None,
                            ),
                        ),
                    )
                if candidate:
                    value, subject = candidate, environment_name
                    break
        if value is None:
            return None
        status, present = budget.call(self._path_is_dir, value)
        if status is TestReuseCapabilityStatus.UNKNOWN:
            return _Check(
                status,
                str(present),
                (
                    TestReuseCapabilityEvidence(
                        TestReuseCapabilityEvidenceKind.CONFIGURED_PATH,
                        subject,
                        None,
                    ),
                ),
            )
        return _Check(
            TestReuseCapabilityStatus.AVAILABLE if present else TestReuseCapabilityStatus.MISSING,
            "configured_path_available" if present else "configured_path_missing",
            (
                TestReuseCapabilityEvidence(
                    TestReuseCapabilityEvidenceKind.CONFIGURED_PATH,
                    subject,
                    bool(present),
                ),
            ),
        )

    def _registry(self, backend_id: str, budget: _ProbeBudget) -> _Check | None:
        if self._backend_registry is None:
            return None

        status, snapshot = budget.call(
            _snapshot_mapping_entry,
            self._backend_registry,
            backend_id,
            _REGISTRY_FIELDS,
        )
        if status is TestReuseCapabilityStatus.UNKNOWN:
            return _Check(
                status,
                str(snapshot),
                (
                    TestReuseCapabilityEvidence(
                        TestReuseCapabilityEvidenceKind.BACKEND_REGISTRY,
                        backend_id,
                        None,
                    ),
                ),
            )
        evidence = (
            TestReuseCapabilityEvidence(
                TestReuseCapabilityEvidenceKind.BACKEND_REGISTRY,
                backend_id,
                snapshot.key_present,
            ),
        )
        if not snapshot.key_present:
            return _Check(
                TestReuseCapabilityStatus.MISSING,
                "backend_not_registered",
                evidence,
            )
        if not snapshot.value_is_mapping:
            return _Check(
                TestReuseCapabilityStatus.INCOMPATIBLE,
                "backend_registry_incompatible",
                evidence,
            )
        values = dict(snapshot.values)
        if "enabled" in values and not isinstance(values["enabled"], bool):
            return _Check(
                TestReuseCapabilityStatus.INCOMPATIBLE,
                "backend_registry_incompatible",
                evidence,
            )
        if values.get("enabled") is False:
            return _Check(TestReuseCapabilityStatus.DISABLED, "backend_disabled", evidence)
        if "compatible" in values and not isinstance(values["compatible"], bool):
            return _Check(
                TestReuseCapabilityStatus.INCOMPATIBLE,
                "backend_registry_incompatible",
                evidence,
            )
        api_version = values.get("api_version", values.get("version"))
        supported_api_version = (
            type(api_version) in (int, float)
            and api_version == 1
        ) or (
            type(api_version) is str
            and api_version in ("1", "1.0")
        )
        if values.get("compatible") is False or (
            api_version is not None and not supported_api_version
        ):
            return _Check(
                TestReuseCapabilityStatus.INCOMPATIBLE,
                "backend_registry_incompatible",
                evidence,
            )
        available = values.get("available", True)
        if not isinstance(available, bool):
            return _Check(
                TestReuseCapabilityStatus.INCOMPATIBLE,
                "backend_registry_incompatible",
                evidence,
            )
        if available is not True:
            return _Check(TestReuseCapabilityStatus.MISSING, "backend_unavailable", evidence)
        return _Check(
            TestReuseCapabilityStatus.AVAILABLE,
            "backend_registered",
            evidence,
        )

    @staticmethod
    def _combine(*checks: _Check, missing_reason: str) -> _Check:
        evidence = tuple(item for check in checks for item in check.evidence)
        for status in (
            TestReuseCapabilityStatus.UNKNOWN,
            TestReuseCapabilityStatus.INCOMPATIBLE,
            TestReuseCapabilityStatus.DISABLED,
            TestReuseCapabilityStatus.MISSING,
        ):
            match = next((check for check in checks if check.status is status), None)
            if match is not None:
                reason = (
                    match.reason_code
                    if status is not TestReuseCapabilityStatus.MISSING
                    else missing_reason
                )
                return _Check(status, reason, evidence)
        return _Check(
            TestReuseCapabilityStatus.AVAILABLE,
            "cold_prerequisites_available",
            evidence,
        )

    def _probe_multiformats(self, budget: _ProbeBudget) -> _Check:
        return self._module("multiformats", budget)

    def _probe_datasets_zk(self, budget: _ProbeBudget) -> _Check:
        return self._module("ipfs_datasets_py.logic.zkp", budget)

    def _probe_groth16(self, budget: _ProbeBudget) -> _Check:
        registry = self._registry("groth16", budget)
        if registry is not None and registry.status is not TestReuseCapabilityStatus.AVAILABLE:
            return registry
        module = self._module("ipfs_datasets_py.logic.zkp.backends.groth16", budget)
        executable = self._executable(
            ("groth16",),
            ("IPFS_DATASETS_GROTH16_BINARY", "GROTH16_BINARY"),
            budget,
        )
        artifacts = self._configured_directory(
            self.config.groth16_artifacts_path,
            ("IPFS_DATASETS_GROTH16_ARTIFACTS_DIR", "GROTH16_ARTIFACTS_DIR"),
            budget,
        )
        return self._combine(
            *(item for item in (registry, module, executable, artifacts) if item is not None),
            missing_reason="groth16_unavailable",
        )

    def _probe_provekit(self, budget: _ProbeBudget) -> _Check:
        registry = self._registry("provekit", budget)
        if registry is not None and registry.status is not TestReuseCapabilityStatus.AVAILABLE:
            return registry
        module = self._module("ipfs_datasets_py.logic.zkp.backends.provekit", budget)
        executable = self._executable(
            ("provekit-cli", "provekit"),
            ("IPFS_DATASETS_PROVEKIT_BINARY", "PROVEKIT_CLI"),
            budget,
        )
        artifacts = self._configured_directory(
            self.config.provekit_artifacts_path,
            ("IPFS_DATASETS_PROVEKIT_ARTIFACTS_DIR", "PROVEKIT_ARTIFACTS_DIR"),
            budget,
        )
        return self._combine(
            *(item for item in (registry, module, executable, artifacts) if item is not None),
            missing_reason="provekit_unavailable",
        )

    def _probe_cache(self, budget: _ProbeBudget) -> _Check:
        module = self._module(
            "ipfs_accelerate_py.agent_supervisor.proof.mcp_contract_proof_cache",
            budget,
        )
        configured = self._configured_directory(
            self.config.cache_path,
            ("IPFS_TEST_PROOF_REUSE_CACHE_DIR",),
            budget,
        )
        return self._combine(
            *(item for item in (module, configured) if item is not None),
            missing_reason="cache_unavailable",
        )

    def _probe_ipfs(self, budget: _ProbeBudget) -> _Check:
        module = self._module("ipfshttpclient", budget)
        if module.status is TestReuseCapabilityStatus.UNKNOWN:
            return module
        if module.status is TestReuseCapabilityStatus.AVAILABLE:
            return _Check(
                TestReuseCapabilityStatus.AVAILABLE,
                "ipfs_client_discoverable",
                module.evidence,
            )
        executable = self._executable(("ipfs",), (), budget)
        evidence = (*module.evidence, *executable.evidence)
        if executable.status is TestReuseCapabilityStatus.UNKNOWN:
            return _Check(
                TestReuseCapabilityStatus.UNKNOWN,
                executable.reason_code,
                evidence,
            )
        if executable.status is TestReuseCapabilityStatus.AVAILABLE:
            return _Check(
                TestReuseCapabilityStatus.AVAILABLE,
                "ipfs_client_discoverable",
                evidence,
            )
        return _Check(TestReuseCapabilityStatus.MISSING, "ipfs_unavailable", evidence)

    def _configured_file(
        self,
        configured: str | None,
        environment_name: str,
        budget: _ProbeBudget,
    ) -> _Check | None:
        value = configured
        subject = "probe_config"
        if value is None:
            status, candidate = self._environment_value(environment_name, budget)
            if status is TestReuseCapabilityStatus.UNKNOWN:
                return _Check(
                    status,
                    candidate,
                    (
                        TestReuseCapabilityEvidence(
                            TestReuseCapabilityEvidenceKind.CONFIGURATION,
                            environment_name,
                            None,
                        ),
                    ),
                )
            if candidate:
                value, subject = candidate, environment_name
        if value is None:
            return None
        status, present = budget.call(self._path_is_file, value)
        evidence = (
            TestReuseCapabilityEvidence(
                TestReuseCapabilityEvidenceKind.CONFIGURED_PATH,
                subject,
                None if status is TestReuseCapabilityStatus.UNKNOWN else bool(present),
            ),
        )
        if status is TestReuseCapabilityStatus.UNKNOWN:
            return _Check(status, str(present), evidence)
        return _Check(
            TestReuseCapabilityStatus.AVAILABLE if present else TestReuseCapabilityStatus.MISSING,
            "configured_path_available" if present else "configured_path_missing",
            evidence,
        )

    def _probe_local_verifier(self, budget: _ProbeBudget) -> _Check:
        key = self._configured_file(
            self.config.local_verifier_key_path,
            "IPFS_TEST_PROOF_REUSE_VERIFIER_KEY",
            budget,
        )
        circuit = self._configured_file(
            self.config.local_verifier_circuit_path,
            "IPFS_TEST_PROOF_REUSE_VERIFIER_CIRCUIT",
            budget,
        )
        if key is None or circuit is None:
            evidence = tuple(
                TestReuseCapabilityEvidence(
                    TestReuseCapabilityEvidenceKind.CONFIGURATION,
                    subject,
                    False,
                )
                for item, subject in (
                    (key, "IPFS_TEST_PROOF_REUSE_VERIFIER_KEY"),
                    (circuit, "IPFS_TEST_PROOF_REUSE_VERIFIER_CIRCUIT"),
                )
                if item is None
            )
            return _Check(
                TestReuseCapabilityStatus.MISSING,
                "local_verifier_not_configured",
                evidence,
            )
        module = self._module("ipfs_datasets_py.logic.zkp.zkp_verifier", budget)
        return self._combine(
            module,
            key,
            circuit,
            missing_reason="local_verifier_unavailable",
        )


def probe_test_reuse_capabilities(
    config: TestReuseCapabilityProbeConfig | None = None,
    **probe_dependencies: Any,
) -> TestReuseCapabilityReport:
    """Convenience entry point for one uncached cold capability snapshot.

    Dependency keywords are intentionally injectable for deterministic tests and
    embedding applications.  They are the same keywords accepted by
    :class:`TestReuseCapabilityProbe`.
    """

    return TestReuseCapabilityProbe(config, **probe_dependencies).probe()


__all__ = [
    "DEFAULT_TEST_REUSE_CAPABILITY_MAX_CHECKS",
    "DEFAULT_TEST_REUSE_CAPABILITY_TIMEOUT_SECONDS",
    "TEST_REUSE_CAPABILITY_REPORT_SCHEMA",
    "TEST_REUSE_CAPABILITY_REPORT_VERSION",
    "TEST_REUSE_CAPABILITY_SCHEMA_VERSION",
    "TestReuseCapability",
    "TestReuseCapabilityEvidence",
    "TestReuseCapabilityEvidenceKind",
    "TestReuseCapabilityName",
    "TestReuseCapabilityProbe",
    "TestReuseCapabilityProbeConfig",
    "TestReuseCapabilityReport",
    "TestReuseCapabilityStatus",
    "probe_test_reuse_capabilities",
]
