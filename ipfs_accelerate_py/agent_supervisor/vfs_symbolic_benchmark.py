"""Deterministic benchmark for the symbolic-first VFS assurance pipeline.

The benchmark is a measurement and replay boundary.  Producers run the
scanner, caches, symbolic checker, scheduler, and packet compiler, then freeze
their counters in :class:`SymbolicBenchmarkObservation`.  This module never
contacts a provider and never promotes a result.  It validates the complete
observation population and deterministically recomputes a conservative report.

Repository bodies, prompts, provider responses, and proof payloads are
deliberately excluded.  Content identities, counters, and bounded evidence
references are sufficient to establish coverage and parity without turning the
benchmark artifact into another repository-context packet.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Any, Final


SYMBOLIC_BENCHMARK_VERSION: Final = 1
SYMBOLIC_BENCHMARK_EVIDENCE_SCHEMA: Final = (
    "vfs/symbolic-efficiency-benchmark@1"
)
SYMBOLIC_BENCHMARK_OBSERVATION_SCHEMA: Final = (
    "vfs/symbolic-efficiency-observation@1"
)
SYMBOLIC_BENCHMARK_POPULATION_SCHEMA: Final = (
    "vfs/symbolic-efficiency-population@1"
)

DETERMINISTIC_STAGE_NAMES: Final[tuple[str, ...]] = (
    "inventory",
    "scan",
    "parse",
    "identity",
    "graph",
    "contract",
    "cache",
    "proof",
)
REQUIRED_CACHE_STAGES: Final[tuple[str, ...]] = (
    "ast",
    "graph",
    "contract",
    "proof",
)
REQUIRED_SCAN_MODES: Final[tuple[str, ...]] = (
    "cold",
    "warm",
    "exact",
    "delta",
)

MAX_COUNTER: Final = 10**18
MAX_COLLECTION_ITEMS: Final = 100_000
MAX_OBSERVATIONS: Final = 10_000
MAX_OBSERVATION_BYTES: Final = 2 * 1024 * 1024
MAX_POPULATION_BYTES: Final = 32 * 1024 * 1024
MAX_REPORT_BYTES: Final = 2 * 1024 * 1024
_CONTENT_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMPACT_CODE = re.compile(r"^[a-z][a-z0-9_.:/@+-]{0,191}$")


class SymbolicBenchmarkError(ValueError):
    """A symbolic benchmark object is malformed, incomplete, or detached."""


class ScanMode(str, Enum):
    """The required scan/reuse population."""

    COLD = "cold"
    WARM = "warm"
    EXACT = "exact"
    DELTA = "delta"


class FindingTruth(str, Enum):
    """Expected and observed seeded-finding classification."""

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


class GateStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    INSUFFICIENT_SAMPLES = "insufficient-samples"


class BenchmarkConclusion(str, Enum):
    """A measurement conclusion, intentionally not a promotion decision."""

    PASSED = "passed"
    FAILED = "failed"
    INSUFFICIENT_SAMPLES = "insufficient-samples"


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _plain(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SymbolicBenchmarkError(
            "benchmark data must be canonical JSON"
        ) from exc


def _identity(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _load_json(value: str | bytes | bytearray, name: str) -> Mapping[str, Any]:
    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise SymbolicBenchmarkError(f"{name} contains duplicate keys")
            result[key] = item
        return result

    try:
        parsed = json.loads(value, object_pairs_hook=reject_pairs)
    except SymbolicBenchmarkError:
        raise
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise SymbolicBenchmarkError(f"{name} is not valid JSON") from exc
    if not isinstance(parsed, Mapping):
        raise SymbolicBenchmarkError(f"{name} must be a JSON object")
    return parsed


def _text(value: Any, name: str, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SymbolicBenchmarkError(f"{name} must be non-empty text")
    result = value.strip()
    if "\x00" in result or len(result.encode("utf-8")) > maximum:
        raise SymbolicBenchmarkError(
            f"{name} is unsafe or exceeds its {maximum}-byte bound"
        )
    return result


def _code(value: Any, name: str) -> str:
    result = _text(value, name, 192).lower()
    if not _COMPACT_CODE.fullmatch(result):
        raise SymbolicBenchmarkError(f"{name} must be a compact code")
    return result


def _content_id(value: Any, name: str) -> str:
    result = _text(value, name, 71).lower()
    if not _CONTENT_ID.fullmatch(result):
        raise SymbolicBenchmarkError(f"{name} must be a sha256 content ID")
    return result


def _count(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int = MAX_COUNTER,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        raise SymbolicBenchmarkError(
            f"{name} must be an integer from {minimum} through {maximum}"
        )
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise SymbolicBenchmarkError(f"{name} must be boolean")
    return value


def _enum(value: Any, enum_type: type[Enum], name: str) -> Enum:
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except (TypeError, ValueError) as exc:
        raise SymbolicBenchmarkError(f"{name} is unsupported") from exc


def _items(
    value: Sequence[Any],
    name: str,
    *,
    allow_empty: bool = True,
) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise SymbolicBenchmarkError(f"{name} must be a sequence")
    result = tuple(value)
    if (not allow_empty and not result) or len(result) > MAX_COLLECTION_ITEMS:
        raise SymbolicBenchmarkError(f"{name} has an invalid item count")
    return result


def _codes(
    value: Sequence[Any],
    name: str,
    *,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    result = tuple(_code(item, f"{name} item") for item in _items(
        value, name, allow_empty=allow_empty
    ))
    if len(result) != len(set(result)):
        raise SymbolicBenchmarkError(f"{name} contains duplicates")
    return tuple(sorted(result))


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SymbolicBenchmarkError(f"{name} must be an object")
    return value


@dataclass(frozen=True)
class FixtureIdentity:
    """Complete identity of one immutable benchmark fixture."""

    fixture_id: str
    fixture_revision: str
    repository_id: str
    forest_id: str
    dirty_overlay_id: str
    inventory_policy_id: str
    inventory_policy_revision: str
    seeded_findings_id: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _text(getattr(self, name), name))

    @property
    def identity_id(self) -> str:
        return _identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FixtureIdentity":
        return cls(**{
            name: value[name] for name in cls.__dataclass_fields__
        })


@dataclass(frozen=True)
class ToolchainIdentity:
    """Versions that participate in deterministic cache and proof keys."""

    scanner_id: str
    parser_id: str
    analyzer_id: str
    graph_schema_id: str
    resolver_id: str
    contract_schema_id: str
    prover_id: str
    proof_circuit_id: str
    cache_schema_id: str
    packet_schema_id: str
    provider_id: str
    provider_revision: str
    tokenizer_id: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _text(getattr(self, name), name))

    @property
    def identity_id(self) -> str:
        return _identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ToolchainIdentity":
        return cls(**{
            name: value[name] for name in cls.__dataclass_fields__
        })


@dataclass(frozen=True)
class BenchmarkProfile:
    """Frozen sufficiency, context, latency, and resource ceilings."""

    profile_id: str
    profile_revision: str
    minimum_samples_per_mode: int = 3
    minimum_packet_pairs: int = 3
    minimum_provider_reduction_basis_points: int = 8_000
    packet_input_budget_bytes: int = 16_384
    max_counterexample_time_ns: int = 10_000_000_000
    max_wall_time_ns: int = 60_000_000_000
    max_cpu_time_ns: int = 60_000_000_000
    max_peak_rss_bytes: int = 2 * 1024 * 1024 * 1024
    max_process_count: int = 16
    max_disk_growth_bytes: int = 1024 * 1024 * 1024
    max_artifact_bytes: int = 1024 * 1024 * 1024
    minimum_idle_observation_ns: int = 1_000_000_000
    max_idle_cpu_millionths: int = 20_000
    max_idle_write_operations: int = 0
    max_idle_write_bytes: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_id", _text(
            self.profile_id, "profile_id"
        ))
        object.__setattr__(self, "profile_revision", _text(
            self.profile_revision, "profile_revision"
        ))
        for name in self.__dataclass_fields__:
            if name in {"profile_id", "profile_revision"}:
                continue
            minimum = 1 if name in {
                "minimum_samples_per_mode",
                "minimum_packet_pairs",
                "packet_input_budget_bytes",
                "max_counterexample_time_ns",
                "max_wall_time_ns",
                "max_cpu_time_ns",
                "max_peak_rss_bytes",
                "max_process_count",
                "max_disk_growth_bytes",
                "max_artifact_bytes",
                "minimum_idle_observation_ns",
            } else 0
            object.__setattr__(
                self,
                name,
                _count(getattr(self, name), name, minimum=minimum),
            )
        if self.minimum_provider_reduction_basis_points > 10_000:
            raise SymbolicBenchmarkError(
                "minimum_provider_reduction_basis_points exceeds 10000"
            )
        if self.max_idle_cpu_millionths > 1_000_000:
            raise SymbolicBenchmarkError(
                "max_idle_cpu_millionths exceeds 1000000"
            )

    @property
    def identity_id(self) -> str:
        return _identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BenchmarkProfile":
        return cls(**{
            name: value[name] for name in cls.__dataclass_fields__
        })


@dataclass(frozen=True)
class InventoryMeasurement:
    observed_paths: int
    emitted_paths: int
    included_paths: int
    excluded_paths: int
    omitted_paths: int
    exhaustive: bool
    unexplained_gap_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "observed_paths",
            "emitted_paths",
            "included_paths",
            "excluded_paths",
            "omitted_paths",
        ):
            object.__setattr__(self, name, _count(getattr(self, name), name))
        object.__setattr__(
            self,
            "exhaustive",
            _boolean(self.exhaustive, "exhaustive"),
        )
        object.__setattr__(
            self,
            "unexplained_gap_codes",
            _codes(self.unexplained_gap_codes, "unexplained_gap_codes"),
        )
        if self.emitted_paths + self.omitted_paths != self.observed_paths:
            raise SymbolicBenchmarkError(
                "inventory emitted plus omitted must equal observed"
            )
        if self.included_paths + self.excluded_paths != self.emitted_paths:
            raise SymbolicBenchmarkError(
                "inventory included plus excluded must equal emitted"
            )
        if self.exhaustive and (
            self.omitted_paths or self.unexplained_gap_codes
        ):
            raise SymbolicBenchmarkError(
                "exhaustive inventory cannot contain unexplained gaps"
            )

    @property
    def complete(self) -> bool:
        return (
            self.exhaustive
            and self.omitted_paths == 0
            and not self.unexplained_gap_codes
        )

    def to_dict(self) -> dict[str, Any]:
        return _plain({
            name: getattr(self, name) for name in self.__dataclass_fields__
        })

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InventoryMeasurement":
        return cls(**{
            name: value.get(name, ())
            if name == "unexplained_gap_codes"
            else value[name]
            for name in cls.__dataclass_fields__
        })


@dataclass(frozen=True)
class CacheMeasurement:
    stage: str
    lookups: int
    hits: int
    reused_artifacts: int
    reused_bytes: int
    produced_artifacts: int
    produced_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", _code(self.stage, "cache stage"))
        if self.stage not in REQUIRED_CACHE_STAGES:
            raise SymbolicBenchmarkError("unsupported cache stage")
        for name in self.__dataclass_fields__:
            if name == "stage":
                continue
            object.__setattr__(self, name, _count(getattr(self, name), name))
        if self.hits > self.lookups:
            raise SymbolicBenchmarkError("cache hits exceed lookups")
        if self.reused_artifacts > self.hits:
            raise SymbolicBenchmarkError(
                "reused artifacts exceed cache hits"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CacheMeasurement":
        return cls(**{
            name: value[name] for name in cls.__dataclass_fields__
        })


@dataclass(frozen=True)
class InvalidationMeasurement:
    changed_source_ids: tuple[str, ...]
    expected_invalidated_ids: tuple[str, ...]
    actual_invalidated_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                name,
                _codes(getattr(self, name), name),
            )
        if not self.changed_source_ids:
            raise SymbolicBenchmarkError(
                "delta invalidation needs at least one changed source"
            )

    @property
    def false_invalidations(self) -> tuple[str, ...]:
        return tuple(sorted(
            set(self.actual_invalidated_ids)
            - set(self.expected_invalidated_ids)
        ))

    @property
    def missed_invalidations(self) -> tuple[str, ...]:
        return tuple(sorted(
            set(self.expected_invalidated_ids)
            - set(self.actual_invalidated_ids)
        ))

    @property
    def precise(self) -> bool:
        return not self.false_invalidations and not self.missed_invalidations

    def to_dict(self) -> dict[str, Any]:
        return _plain({
            name: getattr(self, name) for name in self.__dataclass_fields__
        })

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InvalidationMeasurement":
        return cls(**{
            name: tuple(value[name]) for name in cls.__dataclass_fields__
        })


@dataclass(frozen=True)
class FindingMeasurement:
    seed_id: str
    expected_truth: FindingTruth
    observed_truth: FindingTruth
    evidence_ids: tuple[str, ...]
    time_to_counterexample_ns: int | None = None
    counterexample_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed_id", _code(self.seed_id, "seed_id"))
        object.__setattr__(
            self,
            "expected_truth",
            _enum(self.expected_truth, FindingTruth, "expected_truth"),
        )
        object.__setattr__(
            self,
            "observed_truth",
            _enum(self.observed_truth, FindingTruth, "observed_truth"),
        )
        object.__setattr__(
            self,
            "evidence_ids",
            _codes(self.evidence_ids, "evidence_ids", allow_empty=False),
        )
        if self.counterexample_id is not None:
            object.__setattr__(
                self,
                "counterexample_id",
                _text(self.counterexample_id, "counterexample_id"),
            )
        if self.time_to_counterexample_ns is not None:
            object.__setattr__(
                self,
                "time_to_counterexample_ns",
                _count(
                    self.time_to_counterexample_ns,
                    "time_to_counterexample_ns",
                ),
            )
        if self.observed_truth is FindingTruth.TRUE:
            if (
                self.counterexample_id is None
                or self.time_to_counterexample_ns is None
            ):
                raise SymbolicBenchmarkError(
                    "true findings require a timed counterexample"
                )
        elif (
            self.counterexample_id is not None
            or self.time_to_counterexample_ns is not None
        ):
            raise SymbolicBenchmarkError(
                "non-true findings cannot claim a counterexample"
            )

    @property
    def covered(self) -> bool:
        return self.expected_truth is self.observed_truth

    def to_dict(self) -> dict[str, Any]:
        return _plain({
            name: getattr(self, name) for name in self.__dataclass_fields__
        })

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FindingMeasurement":
        return cls(
            seed_id=value["seed_id"],
            expected_truth=value["expected_truth"],
            observed_truth=value["observed_truth"],
            evidence_ids=tuple(value["evidence_ids"]),
            time_to_counterexample_ns=value.get(
                "time_to_counterexample_ns"
            ),
            counterexample_id=value.get("counterexample_id"),
        )


@dataclass(frozen=True)
class TaskMeasurement:
    candidate_findings: int
    eligible_findings: int
    emitted_tasks: int
    deduplicated_findings: int
    duplicate_group_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "candidate_findings",
            "eligible_findings",
            "emitted_tasks",
            "deduplicated_findings",
        ):
            object.__setattr__(self, name, _count(getattr(self, name), name))
        object.__setattr__(
            self,
            "duplicate_group_ids",
            _codes(self.duplicate_group_ids, "duplicate_group_ids"),
        )
        if self.eligible_findings > self.candidate_findings:
            raise SymbolicBenchmarkError(
                "eligible findings exceed candidates"
            )
        if (
            self.emitted_tasks + self.deduplicated_findings
            != self.eligible_findings
        ):
            raise SymbolicBenchmarkError(
                "task yield and deduplication do not close eligible findings"
            )
        if bool(self.deduplicated_findings) != bool(
            self.duplicate_group_ids
        ):
            raise SymbolicBenchmarkError(
                "deduplicated findings require duplicate group identities"
            )

    def to_dict(self) -> dict[str, Any]:
        return _plain({
            name: getattr(self, name) for name in self.__dataclass_fields__
        })

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskMeasurement":
        return cls(
            candidate_findings=value["candidate_findings"],
            eligible_findings=value["eligible_findings"],
            emitted_tasks=value["emitted_tasks"],
            deduplicated_findings=value["deduplicated_findings"],
            duplicate_group_ids=tuple(value.get("duplicate_group_ids", ())),
        )


@dataclass(frozen=True)
class ProviderPacketMeasurement:
    """A paired bounded-baseline/compact-packet input observation."""

    pair_id: str
    baseline_context_bound_bytes: int
    baseline_input_bytes: int
    baseline_input_tokens: int
    packet_input_bytes: int
    packet_input_tokens: int
    baseline_required_evidence_ids: tuple[str, ...]
    packet_evidence_ids: tuple[str, ...]
    baseline_seed_coverage_ids: tuple[str, ...]
    packet_seed_coverage_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair_id", _code(self.pair_id, "pair_id"))
        for name in (
            "baseline_context_bound_bytes",
            "baseline_input_bytes",
            "baseline_input_tokens",
            "packet_input_bytes",
            "packet_input_tokens",
        ):
            object.__setattr__(
                self,
                name,
                _count(getattr(self, name), name, minimum=1),
            )
        if self.baseline_input_bytes > self.baseline_context_bound_bytes:
            raise SymbolicBenchmarkError(
                "baseline input exceeds its declared repository-context bound"
            )
        for name in (
            "baseline_required_evidence_ids",
            "packet_evidence_ids",
            "baseline_seed_coverage_ids",
            "packet_seed_coverage_ids",
        ):
            object.__setattr__(
                self,
                name,
                _codes(getattr(self, name), name, allow_empty=False),
            )

    @property
    def evidence_preserved(self) -> bool:
        return set(self.baseline_required_evidence_ids).issubset(
            self.packet_evidence_ids
        )

    @property
    def seeded_coverage_preserved(self) -> bool:
        return set(self.baseline_seed_coverage_ids).issubset(
            self.packet_seed_coverage_ids
        )

    def to_dict(self) -> dict[str, Any]:
        return _plain({
            name: getattr(self, name) for name in self.__dataclass_fields__
        })

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "ProviderPacketMeasurement":
        return cls(**{
            name: tuple(value[name])
            if name.endswith("_ids")
            else value[name]
            for name in cls.__dataclass_fields__
        })


@dataclass(frozen=True)
class ResourceMeasurement:
    wall_time_ns: int
    cpu_time_ns: int
    peak_rss_bytes: int
    peak_process_count: int
    disk_bytes_before: int
    disk_bytes_after: int
    artifact_bytes: int
    idle_observation_ns: int
    idle_cpu_time_ns: int
    idle_write_operations: int
    idle_write_bytes: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            minimum = 1 if name in {
                "wall_time_ns",
                "peak_process_count",
                "idle_observation_ns",
            } else 0
            object.__setattr__(
                self,
                name,
                _count(getattr(self, name), name, minimum=minimum),
            )

    @property
    def disk_growth_bytes(self) -> int:
        return max(0, self.disk_bytes_after - self.disk_bytes_before)

    @property
    def idle_cpu_millionths(self) -> int:
        return (
            self.idle_cpu_time_ns * 1_000_000
        ) // self.idle_observation_ns

    def to_dict(self) -> dict[str, Any]:
        result = {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }
        result["disk_growth_bytes"] = self.disk_growth_bytes
        result["idle_cpu_millionths"] = self.idle_cpu_millionths
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResourceMeasurement":
        result = cls(**{
            name: value[name] for name in cls.__dataclass_fields__
        })
        if value.get("disk_growth_bytes", result.disk_growth_bytes) != (
            result.disk_growth_bytes
        ):
            raise SymbolicBenchmarkError("disk growth metric mismatch")
        if value.get(
            "idle_cpu_millionths", result.idle_cpu_millionths
        ) != result.idle_cpu_millionths:
            raise SymbolicBenchmarkError("idle CPU metric mismatch")
        return result


@dataclass(frozen=True)
class SymbolicBenchmarkObservation:
    """One measured scan, including its paired provider-input packet."""

    mode: ScanMode
    sample_index: int
    fixture: FixtureIdentity
    toolchain: ToolchainIdentity
    profile_id: str
    profile_revision: str
    inventory: InventoryMeasurement
    caches: tuple[CacheMeasurement, ...]
    deterministic_stage_llm_calls: tuple[tuple[str, int], ...]
    findings: tuple[FindingMeasurement, ...]
    invalidation: InvalidationMeasurement | None
    tasks: TaskMeasurement
    packet: ProviderPacketMeasurement
    resources: ResourceMeasurement
    source_receipt_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "mode", _enum(self.mode, ScanMode, "scan mode")
        )
        object.__setattr__(
            self,
            "sample_index",
            _count(self.sample_index, "sample_index", minimum=1),
        )
        if not isinstance(self.fixture, FixtureIdentity):
            raise SymbolicBenchmarkError("fixture identity is invalid")
        if not isinstance(self.toolchain, ToolchainIdentity):
            raise SymbolicBenchmarkError("toolchain identity is invalid")
        object.__setattr__(
            self, "profile_id", _text(self.profile_id, "profile_id")
        )
        object.__setattr__(
            self,
            "profile_revision",
            _text(self.profile_revision, "profile_revision"),
        )
        if not isinstance(self.inventory, InventoryMeasurement):
            raise SymbolicBenchmarkError("inventory measurement is invalid")
        caches = _items(self.caches, "caches", allow_empty=False)
        if any(not isinstance(item, CacheMeasurement) for item in caches):
            raise SymbolicBenchmarkError("cache measurement is invalid")
        by_stage = {item.stage: item for item in caches}
        if set(by_stage) != set(REQUIRED_CACHE_STAGES):
            raise SymbolicBenchmarkError(
                "AST, graph, contract, and proof cache measurements are required"
            )
        object.__setattr__(
            self,
            "caches",
            tuple(by_stage[stage] for stage in REQUIRED_CACHE_STAGES),
        )
        calls: dict[str, int] = {}
        for raw in _items(
            self.deterministic_stage_llm_calls,
            "deterministic_stage_llm_calls",
            allow_empty=False,
        ):
            if (
                not isinstance(raw, (tuple, list))
                or len(raw) != 2
            ):
                raise SymbolicBenchmarkError(
                    "stage call measurement must be a name/count pair"
                )
            stage = _code(raw[0], "deterministic stage")
            if stage in calls:
                raise SymbolicBenchmarkError(
                    "duplicate deterministic stage measurement"
                )
            calls[stage] = _count(raw[1], "deterministic stage LLM calls")
        if set(calls) != set(DETERMINISTIC_STAGE_NAMES):
            raise SymbolicBenchmarkError(
                "all deterministic stages must record LLM call counts"
            )
        object.__setattr__(
            self,
            "deterministic_stage_llm_calls",
            tuple((stage, calls[stage]) for stage in DETERMINISTIC_STAGE_NAMES),
        )
        findings = _items(self.findings, "findings", allow_empty=False)
        if any(not isinstance(item, FindingMeasurement) for item in findings):
            raise SymbolicBenchmarkError("finding measurement is invalid")
        seed_ids = [item.seed_id for item in findings]
        if len(seed_ids) != len(set(seed_ids)):
            raise SymbolicBenchmarkError("duplicate seeded finding")
        expected = {item.expected_truth for item in findings}
        if expected != set(FindingTruth):
            raise SymbolicBenchmarkError(
                "true, false, and unknown seeded findings are required"
            )
        object.__setattr__(
            self, "findings", tuple(sorted(findings, key=lambda item: item.seed_id))
        )
        if self.mode is ScanMode.DELTA:
            if not isinstance(self.invalidation, InvalidationMeasurement):
                raise SymbolicBenchmarkError(
                    "delta scans require invalidation measurement"
                )
        elif self.invalidation is not None:
            raise SymbolicBenchmarkError(
                "only delta scans may record invalidation"
            )
        if not isinstance(self.tasks, TaskMeasurement):
            raise SymbolicBenchmarkError("task measurement is invalid")
        if not isinstance(self.packet, ProviderPacketMeasurement):
            raise SymbolicBenchmarkError("packet measurement is invalid")
        if not isinstance(self.resources, ResourceMeasurement):
            raise SymbolicBenchmarkError("resource measurement is invalid")
        object.__setattr__(
            self,
            "source_receipt_ids",
            _codes(
                self.source_receipt_ids,
                "source_receipt_ids",
                allow_empty=False,
            ),
        )

    @property
    def observation_id(self) -> str:
        return _identity(self.to_dict(include_observation_id=False))

    @property
    def total_llm_calls(self) -> int:
        return sum(count for _, count in self.deterministic_stage_llm_calls)

    def to_dict(
        self, *, include_observation_id: bool = True
    ) -> dict[str, Any]:
        payload = {
            "schema": SYMBOLIC_BENCHMARK_OBSERVATION_SCHEMA,
            "version": SYMBOLIC_BENCHMARK_VERSION,
            "mode": self.mode.value,
            "sample_index": self.sample_index,
            "fixture": self.fixture.to_dict(),
            "toolchain": self.toolchain.to_dict(),
            "profile_id": self.profile_id,
            "profile_revision": self.profile_revision,
            "inventory": self.inventory.to_dict(),
            "caches": [item.to_dict() for item in self.caches],
            "deterministic_stage_llm_calls": [
                [stage, count]
                for stage, count in self.deterministic_stage_llm_calls
            ],
            "findings": [item.to_dict() for item in self.findings],
            "invalidation": (
                None if self.invalidation is None
                else self.invalidation.to_dict()
            ),
            "tasks": self.tasks.to_dict(),
            "packet": self.packet.to_dict(),
            "resources": self.resources.to_dict(),
            "source_receipt_ids": list(self.source_receipt_ids),
        }
        if include_observation_id:
            payload["observation_id"] = self.observation_id
        return payload

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "SymbolicBenchmarkObservation":
        if (
            value.get("schema") != SYMBOLIC_BENCHMARK_OBSERVATION_SCHEMA
            or value.get("version") != SYMBOLIC_BENCHMARK_VERSION
        ):
            raise SymbolicBenchmarkError("unsupported observation schema")
        result = cls(
            mode=value["mode"],
            sample_index=value["sample_index"],
            fixture=FixtureIdentity.from_dict(_mapping(
                value["fixture"], "fixture"
            )),
            toolchain=ToolchainIdentity.from_dict(_mapping(
                value["toolchain"], "toolchain"
            )),
            profile_id=value["profile_id"],
            profile_revision=value["profile_revision"],
            inventory=InventoryMeasurement.from_dict(_mapping(
                value["inventory"], "inventory"
            )),
            caches=tuple(
                CacheMeasurement.from_dict(_mapping(item, "cache"))
                for item in value["caches"]
            ),
            deterministic_stage_llm_calls=tuple(
                tuple(item) for item in value["deterministic_stage_llm_calls"]
            ),
            findings=tuple(
                FindingMeasurement.from_dict(_mapping(item, "finding"))
                for item in value["findings"]
            ),
            invalidation=(
                None
                if value.get("invalidation") is None
                else InvalidationMeasurement.from_dict(_mapping(
                    value["invalidation"], "invalidation"
                ))
            ),
            tasks=TaskMeasurement.from_dict(_mapping(
                value["tasks"], "tasks"
            )),
            packet=ProviderPacketMeasurement.from_dict(_mapping(
                value["packet"], "packet"
            )),
            resources=ResourceMeasurement.from_dict(_mapping(
                value["resources"], "resources"
            )),
            source_receipt_ids=tuple(value["source_receipt_ids"]),
        )
        if value.get(
            "observation_id", result.observation_id
        ) != result.observation_id:
            raise SymbolicBenchmarkError("observation ID mismatch")
        if len(_canonical_bytes(result.to_dict())) > MAX_OBSERVATION_BYTES:
            raise SymbolicBenchmarkError("observation exceeds byte bound")
        return result

    def to_json(self) -> str:
        return _canonical_bytes(self.to_dict()).decode("utf-8")

    @classmethod
    def from_json(
        cls, value: str | bytes | bytearray
    ) -> "SymbolicBenchmarkObservation":
        return cls.from_dict(_load_json(value, "symbolic observation"))


@dataclass(frozen=True)
class SymbolicBenchmarkPopulation:
    """The complete observation population and its frozen profile."""

    profile: BenchmarkProfile
    observations: tuple[SymbolicBenchmarkObservation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.profile, BenchmarkProfile):
            raise SymbolicBenchmarkError("benchmark profile is invalid")
        observations = _items(
            self.observations, "observations", allow_empty=False
        )
        if len(observations) > MAX_OBSERVATIONS:
            raise SymbolicBenchmarkError("too many benchmark observations")
        if any(
            not isinstance(item, SymbolicBenchmarkObservation)
            for item in observations
        ):
            raise SymbolicBenchmarkError("observation population is invalid")
        ids = [item.observation_id for item in observations]
        if len(ids) != len(set(ids)):
            raise SymbolicBenchmarkError("duplicate benchmark observation")
        pairs = [item.packet.pair_id for item in observations]
        if len(pairs) != len(set(pairs)):
            raise SymbolicBenchmarkError("duplicate provider packet pair")
        coordinates = [
            (
                item.fixture.identity_id,
                item.toolchain.identity_id,
                item.mode.value,
                item.sample_index,
            )
            for item in observations
        ]
        if len(coordinates) != len(set(coordinates)):
            raise SymbolicBenchmarkError(
                "duplicate fixture/toolchain/mode/sample coordinate"
            )
        for item in observations:
            if (
                item.profile_id != self.profile.profile_id
                or item.profile_revision != self.profile.profile_revision
            ):
                raise SymbolicBenchmarkError(
                    "observation is detached from benchmark profile"
                )
            if item.packet.packet_input_bytes > (
                self.profile.packet_input_budget_bytes
            ):
                raise SymbolicBenchmarkError(
                    "compact packet exceeds profile input budget"
                )
        object.__setattr__(
            self,
            "observations",
            tuple(sorted(observations, key=lambda item: item.observation_id)),
        )
        if len(_canonical_bytes(self.to_dict())) > MAX_POPULATION_BYTES:
            raise SymbolicBenchmarkError("benchmark population exceeds byte bound")

    @property
    def population_id(self) -> str:
        return _identity(self.to_dict(include_population_id=False))

    def to_dict(
        self, *, include_population_id: bool = True
    ) -> dict[str, Any]:
        payload = {
            "schema": SYMBOLIC_BENCHMARK_POPULATION_SCHEMA,
            "version": SYMBOLIC_BENCHMARK_VERSION,
            "profile": self.profile.to_dict(),
            "observations": [
                item.to_dict() for item in self.observations
            ],
        }
        if include_population_id:
            payload["population_id"] = self.population_id
        return payload

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "SymbolicBenchmarkPopulation":
        if (
            value.get("schema") != SYMBOLIC_BENCHMARK_POPULATION_SCHEMA
            or value.get("version") != SYMBOLIC_BENCHMARK_VERSION
        ):
            raise SymbolicBenchmarkError("unsupported population schema")
        result = cls(
            profile=BenchmarkProfile.from_dict(_mapping(
                value["profile"], "profile"
            )),
            observations=tuple(
                SymbolicBenchmarkObservation.from_dict(_mapping(
                    item, "observation"
                ))
                for item in value["observations"]
            ),
        )
        if value.get("population_id", result.population_id) != (
            result.population_id
        ):
            raise SymbolicBenchmarkError("population ID mismatch")
        return result

    def to_json(self) -> str:
        return _canonical_bytes(self.to_dict()).decode("utf-8")

    @classmethod
    def from_json(
        cls, value: str | bytes | bytearray
    ) -> "SymbolicBenchmarkPopulation":
        return cls.from_dict(_load_json(value, "symbolic population"))


@dataclass(frozen=True)
class BenchmarkGate:
    name: str
    status: GateStatus
    observed: str
    requirement: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _code(self.name, "gate name"))
        object.__setattr__(
            self, "status", _enum(self.status, GateStatus, "gate status")
        )
        object.__setattr__(
            self, "observed", _text(self.observed, "gate observed", 2048)
        )
        object.__setattr__(
            self,
            "requirement",
            _text(self.requirement, "gate requirement", 2048),
        )

    def to_dict(self) -> dict[str, Any]:
        return _plain({
            name: getattr(self, name) for name in self.__dataclass_fields__
        })


def _median_fraction(values: Sequence[int]) -> Fraction:
    if not values:
        return Fraction(0, 1)
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return Fraction(ordered[middle], 1)
    return Fraction(ordered[middle - 1] + ordered[middle], 2)


def _fraction_dict(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _reduction_basis_points(baseline: Fraction, candidate: Fraction) -> int:
    if baseline <= 0 or candidate >= baseline:
        return 0
    reduction = (baseline - candidate) / baseline
    return min(10_000, (reduction.numerator * 10_000) // reduction.denominator)


@dataclass(frozen=True)
class SymbolicEfficiencyBenchmarkReport:
    """Replay-derived evidence.  It has no completion or promotion authority."""

    population_id: str
    profile_identity_id: str
    fixture_identity_ids: tuple[str, ...]
    toolchain_identity_ids: tuple[str, ...]
    sample_counts_by_mode: tuple[tuple[str, int], ...]
    observation_count: int
    inventory_observed_paths: int
    inventory_emitted_paths: int
    inventory_included_paths: int
    inventory_excluded_paths: int
    inventory_omitted_paths: int
    inventory_complete_observations: int
    cache_lookups_by_stage: tuple[tuple[str, int], ...]
    cache_hits_by_stage: tuple[tuple[str, int], ...]
    cache_reused_artifacts_by_stage: tuple[tuple[str, int], ...]
    cache_reused_bytes_by_stage: tuple[tuple[str, int], ...]
    invalidation_expected_count: int
    invalidation_actual_count: int
    invalidation_false_positive_count: int
    invalidation_false_negative_count: int
    seeded_expected_by_truth: tuple[tuple[str, int], ...]
    seeded_covered_by_truth: tuple[tuple[str, int], ...]
    counterexample_count: int
    median_counterexample_time_ns: dict[str, int]
    artifact_bytes: int
    wall_time_ns: int
    cpu_time_ns: int
    scan_wall_time_ns_by_mode: tuple[tuple[str, int], ...]
    scan_cpu_time_ns_by_mode: tuple[tuple[str, int], ...]
    peak_rss_bytes: int
    peak_process_count: int
    disk_growth_bytes: int
    idle_cpu_time_ns: int
    idle_write_operations: int
    idle_write_bytes: int
    candidate_findings: int
    eligible_findings: int
    emitted_tasks: int
    deduplicated_findings: int
    provider_pair_count: int
    median_baseline_input_bytes: dict[str, int]
    median_packet_input_bytes: dict[str, int]
    median_baseline_input_tokens: dict[str, int]
    median_packet_input_tokens: dict[str, int]
    provider_byte_reduction_basis_points: int
    provider_token_reduction_basis_points: int
    deterministic_llm_calls: int
    gates: tuple[BenchmarkGate, ...]
    conclusion: BenchmarkConclusion
    failure_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _content_id(self.population_id, "population_id")
        _content_id(self.profile_identity_id, "profile_identity_id")
        object.__setattr__(
            self,
            "fixture_identity_ids",
            tuple(sorted(_content_id(
                item, "fixture identity"
            ) for item in self.fixture_identity_ids)),
        )
        object.__setattr__(
            self,
            "toolchain_identity_ids",
            tuple(sorted(_content_id(
                item, "toolchain identity"
            ) for item in self.toolchain_identity_ids)),
        )
        gates = tuple(self.gates)
        if any(not isinstance(item, BenchmarkGate) for item in gates):
            raise SymbolicBenchmarkError("invalid benchmark gate")
        if len({item.name for item in gates}) != len(gates):
            raise SymbolicBenchmarkError("duplicate benchmark gate")
        object.__setattr__(
            self, "gates", tuple(sorted(gates, key=lambda item: item.name))
        )
        object.__setattr__(
            self,
            "conclusion",
            _enum(self.conclusion, BenchmarkConclusion, "conclusion"),
        )
        object.__setattr__(
            self,
            "failure_codes",
            _codes(self.failure_codes, "failure_codes"),
        )

    @property
    def report_id(self) -> str:
        return _identity(self.to_dict(include_report_id=False))

    @property
    def passed(self) -> bool:
        return self.conclusion is BenchmarkConclusion.PASSED

    @property
    def authoritative(self) -> bool:
        return False

    @property
    def completion_authoritative(self) -> bool:
        return False

    @property
    def promotion_authoritative(self) -> bool:
        return False

    def gate(self, name: str) -> BenchmarkGate:
        normalized = _code(name, "gate name")
        for gate in self.gates:
            if gate.name == normalized:
                return gate
        raise KeyError(normalized)

    def to_dict(self, *, include_report_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": SYMBOLIC_BENCHMARK_EVIDENCE_SCHEMA,
            "version": SYMBOLIC_BENCHMARK_VERSION,
            **{
                name: _plain(getattr(self, name))
                for name in self.__dataclass_fields__
            },
            "authoritative": False,
            "completion_authoritative": False,
            "promotion_authoritative": False,
        }
        if include_report_id:
            payload["report_id"] = self.report_id
        return payload

    def to_json(self) -> str:
        return _canonical_bytes(self.to_dict()).decode("utf-8")

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        population: SymbolicBenchmarkPopulation,
    ) -> "SymbolicEfficiencyBenchmarkReport":
        replayed = evaluate_symbolic_efficiency(population)
        if _canonical_bytes(value) != _canonical_bytes(replayed.to_dict()):
            raise SymbolicBenchmarkError(
                "report does not match complete population replay"
            )
        return replayed

    @classmethod
    def from_json(
        cls,
        value: str | bytes | bytearray,
        *,
        population: SymbolicBenchmarkPopulation,
    ) -> "SymbolicEfficiencyBenchmarkReport":
        return cls.from_dict(
            _load_json(value, "symbolic efficiency report"),
            population=population,
        )


def _gate(
    name: str,
    passed: bool,
    observed: str,
    requirement: str,
    *,
    insufficient: bool = False,
) -> BenchmarkGate:
    status = (
        GateStatus.INSUFFICIENT_SAMPLES
        if insufficient
        else GateStatus.PASSED if passed else GateStatus.FAILED
    )
    return BenchmarkGate(name, status, observed, requirement)


def evaluate_symbolic_efficiency(
    population: SymbolicBenchmarkPopulation,
) -> SymbolicEfficiencyBenchmarkReport:
    """Recompute all VFS-G121 gates from the closed observation population."""

    if not isinstance(population, SymbolicBenchmarkPopulation):
        raise SymbolicBenchmarkError(
            "population must be SymbolicBenchmarkPopulation"
        )
    observations = population.observations
    profile = population.profile
    by_mode = Counter(item.mode.value for item in observations)
    sufficient_modes = all(
        by_mode[mode] >= profile.minimum_samples_per_mode
        for mode in REQUIRED_SCAN_MODES
    )
    sufficient_pairs = (
        len(observations) >= profile.minimum_packet_pairs
    )

    gates: list[BenchmarkGate] = []
    gates.append(_gate(
        "sample-sufficiency",
        sufficient_modes and sufficient_pairs,
        ",".join(f"{mode}={by_mode[mode]}" for mode in REQUIRED_SCAN_MODES),
        (
            f">={profile.minimum_samples_per_mode} per mode and "
            f">={profile.minimum_packet_pairs} packet pairs"
        ),
        insufficient=not (sufficient_modes and sufficient_pairs),
    ))
    inventory_complete = all(item.inventory.complete for item in observations)
    gates.append(_gate(
        "inventory-completeness",
        inventory_complete,
        f"{sum(item.inventory.complete for item in observations)}/"
        f"{len(observations)} complete",
        "every inventory is exhaustive with zero unexplained omissions",
    ))
    deterministic_calls = sum(
        item.total_llm_calls for item in observations
    )
    gates.append(_gate(
        "deterministic-zero-llm",
        deterministic_calls == 0,
        str(deterministic_calls),
        "zero LLM calls in inventory/scan/parse/identity/graph/contract/cache/proof",
    ))

    caches = [
        (item.mode, cache)
        for item in observations
        for cache in item.caches
    ]
    exact_cache = [
        cache for mode, cache in caches if mode is ScanMode.EXACT
    ]
    warm_cache = [
        cache for mode, cache in caches if mode is ScanMode.WARM
    ]
    delta_cache = [
        cache for mode, cache in caches if mode is ScanMode.DELTA
    ]
    exact_reuse = bool(exact_cache) and all(
        cache.lookups > 0
        and cache.hits == cache.lookups
        and cache.reused_artifacts == cache.hits
        for cache in exact_cache
    )
    warm_reuse = bool(warm_cache) and all(
        cache.lookups > 0 and cache.hits > 0
        for cache in warm_cache
    )
    delta_reuse = bool(delta_cache) and all(
        cache.lookups > 0 and cache.hits > 0
        for cache in delta_cache
    )
    gates.append(_gate(
        "cache-reuse",
        exact_reuse and warm_reuse and delta_reuse,
        (
            f"exact={int(exact_reuse)},warm={int(warm_reuse)},"
            f"delta={int(delta_reuse)}"
        ),
        "AST/graph/contract/proof exact hits and warm/delta reuse",
    ))

    invalidations = [
        item.invalidation
        for item in observations
        if item.mode is ScanMode.DELTA
    ]
    precise_invalidation = bool(invalidations) and all(
        item is not None and item.precise for item in invalidations
    )
    gates.append(_gate(
        "invalidation-precision",
        precise_invalidation,
        (
            f"false-positive={sum(len(item.false_invalidations) for item in invalidations if item)},"
            f"false-negative={sum(len(item.missed_invalidations) for item in invalidations if item)}"
        ),
        "actual invalidation closure equals expected transitive closure",
    ))

    findings = [
        finding
        for observation in observations
        for finding in observation.findings
    ]
    finding_coverage = all(item.covered for item in findings)
    gates.append(_gate(
        "seeded-finding-coverage",
        finding_coverage,
        f"{sum(item.covered for item in findings)}/{len(findings)}",
        "all seeded true/false/unknown classifications match",
    ))
    counterexamples = [
        item.time_to_counterexample_ns
        for item in findings
        if item.expected_truth is FindingTruth.TRUE
        and item.time_to_counterexample_ns is not None
    ]
    counterexample_latency = (
        len(counterexamples)
        == sum(
            item.expected_truth is FindingTruth.TRUE for item in findings
        )
        and all(
            item <= profile.max_counterexample_time_ns
            for item in counterexamples
        )
    )
    gates.append(_gate(
        "counterexample-latency",
        counterexample_latency,
        (
            f"max={max(counterexamples) if counterexamples else 0}ns,"
            f"count={len(counterexamples)}"
        ),
        f"every true seed <= {profile.max_counterexample_time_ns}ns",
    ))

    tasks_close = all(
        item.tasks.emitted_tasks + item.tasks.deduplicated_findings
        == item.tasks.eligible_findings
        for item in observations
    )
    gates.append(_gate(
        "task-yield-deduplication",
        tasks_close,
        (
            f"eligible={sum(item.tasks.eligible_findings for item in observations)},"
            f"tasks={sum(item.tasks.emitted_tasks for item in observations)},"
            f"deduplicated={sum(item.tasks.deduplicated_findings for item in observations)}"
        ),
        "emitted tasks plus same-root deduplications close eligible findings",
    ))

    packet_parity = all(
        item.packet.evidence_preserved
        and item.packet.seeded_coverage_preserved
        for item in observations
    )
    gates.append(_gate(
        "packet-evidence-parity",
        packet_parity,
        f"{sum(item.packet.evidence_preserved and item.packet.seeded_coverage_preserved for item in observations)}/{len(observations)}",
        "compact packet preserves required evidence and baseline seeded coverage",
    ))
    median_baseline_bytes = _median_fraction([
        item.packet.baseline_input_bytes for item in observations
    ])
    median_packet_bytes = _median_fraction([
        item.packet.packet_input_bytes for item in observations
    ])
    median_baseline_tokens = _median_fraction([
        item.packet.baseline_input_tokens for item in observations
    ])
    median_packet_tokens = _median_fraction([
        item.packet.packet_input_tokens for item in observations
    ])
    byte_reduction = _reduction_basis_points(
        median_baseline_bytes, median_packet_bytes
    )
    token_reduction = _reduction_basis_points(
        median_baseline_tokens, median_packet_tokens
    )
    # Provider medians are promotion evidence only when the entire paired
    # benchmark design is sufficiently sampled, including every scan mode.
    reduction_insufficient = not (sufficient_modes and sufficient_pairs)
    gates.append(_gate(
        "provider-byte-reduction",
        byte_reduction
        >= profile.minimum_provider_reduction_basis_points,
        f"{byte_reduction} basis points",
        f">={profile.minimum_provider_reduction_basis_points} basis points",
        insufficient=reduction_insufficient,
    ))
    gates.append(_gate(
        "provider-token-reduction",
        token_reduction
        >= profile.minimum_provider_reduction_basis_points,
        f"{token_reduction} basis points",
        f">={profile.minimum_provider_reduction_basis_points} basis points",
        insufficient=reduction_insufficient,
    ))

    resource_passed = all(
        item.resources.wall_time_ns <= profile.max_wall_time_ns
        and item.resources.cpu_time_ns <= profile.max_cpu_time_ns
        and item.resources.peak_rss_bytes <= profile.max_peak_rss_bytes
        and item.resources.peak_process_count <= profile.max_process_count
        and item.resources.disk_growth_bytes
        <= profile.max_disk_growth_bytes
        and item.resources.artifact_bytes <= profile.max_artifact_bytes
        for item in observations
    )
    gates.append(_gate(
        "resource-ceilings",
        resource_passed,
        (
            f"peak-rss={max(item.resources.peak_rss_bytes for item in observations)},"
            f"peak-processes={max(item.resources.peak_process_count for item in observations)},"
            f"artifact-bytes={sum(item.resources.artifact_bytes for item in observations)}"
        ),
        "every observation is within the frozen CPU/RSS/process/disk/artifact ceilings",
    ))
    idle_passed = all(
        item.resources.idle_observation_ns
        >= profile.minimum_idle_observation_ns
        and item.resources.idle_cpu_millionths
        <= profile.max_idle_cpu_millionths
        and item.resources.idle_write_operations
        <= profile.max_idle_write_operations
        and item.resources.idle_write_bytes
        <= profile.max_idle_write_bytes
        for item in observations
    )
    gates.append(_gate(
        "idle-quiescence",
        idle_passed,
        (
            f"max-cpu-millionths={max(item.resources.idle_cpu_millionths for item in observations)},"
            f"writes={sum(item.resources.idle_write_operations for item in observations)},"
            f"write-bytes={sum(item.resources.idle_write_bytes for item in observations)}"
        ),
        "idle window meets duration with CPU and writes below profile ceilings",
    ))

    hard_failures = sorted(
        gate.name for gate in gates if gate.status is GateStatus.FAILED
    )
    insufficient = any(
        gate.status is GateStatus.INSUFFICIENT_SAMPLES for gate in gates
    )
    if hard_failures:
        conclusion = BenchmarkConclusion.FAILED
    elif insufficient:
        conclusion = BenchmarkConclusion.INSUFFICIENT_SAMPLES
    else:
        conclusion = BenchmarkConclusion.PASSED

    cache_lookups = Counter()
    cache_hits = Counter()
    cache_reused = Counter()
    cache_bytes = Counter()
    for _, item in caches:
        cache_lookups[item.stage] += item.lookups
        cache_hits[item.stage] += item.hits
        cache_reused[item.stage] += item.reused_artifacts
        cache_bytes[item.stage] += item.reused_bytes
    expected_truth = Counter(item.expected_truth.value for item in findings)
    covered_truth = Counter(
        item.expected_truth.value for item in findings if item.covered
    )
    invalidation_expected = sum(
        len(item.expected_invalidated_ids)
        for item in invalidations if item is not None
    )
    invalidation_actual = sum(
        len(item.actual_invalidated_ids)
        for item in invalidations if item is not None
    )
    false_positive = sum(
        len(item.false_invalidations)
        for item in invalidations if item is not None
    )
    false_negative = sum(
        len(item.missed_invalidations)
        for item in invalidations if item is not None
    )
    counterexample_median = _median_fraction(counterexamples)

    report = SymbolicEfficiencyBenchmarkReport(
        population_id=population.population_id,
        profile_identity_id=profile.identity_id,
        fixture_identity_ids=tuple(sorted({
            item.fixture.identity_id for item in observations
        })),
        toolchain_identity_ids=tuple(sorted({
            item.toolchain.identity_id for item in observations
        })),
        sample_counts_by_mode=tuple(
            (mode, by_mode[mode]) for mode in REQUIRED_SCAN_MODES
        ),
        observation_count=len(observations),
        inventory_observed_paths=sum(
            item.inventory.observed_paths for item in observations
        ),
        inventory_emitted_paths=sum(
            item.inventory.emitted_paths for item in observations
        ),
        inventory_included_paths=sum(
            item.inventory.included_paths for item in observations
        ),
        inventory_excluded_paths=sum(
            item.inventory.excluded_paths for item in observations
        ),
        inventory_omitted_paths=sum(
            item.inventory.omitted_paths for item in observations
        ),
        inventory_complete_observations=sum(
            item.inventory.complete for item in observations
        ),
        cache_lookups_by_stage=tuple(
            (stage, cache_lookups[stage]) for stage in REQUIRED_CACHE_STAGES
        ),
        cache_hits_by_stage=tuple(
            (stage, cache_hits[stage]) for stage in REQUIRED_CACHE_STAGES
        ),
        cache_reused_artifacts_by_stage=tuple(
            (stage, cache_reused[stage]) for stage in REQUIRED_CACHE_STAGES
        ),
        cache_reused_bytes_by_stage=tuple(
            (stage, cache_bytes[stage]) for stage in REQUIRED_CACHE_STAGES
        ),
        invalidation_expected_count=invalidation_expected,
        invalidation_actual_count=invalidation_actual,
        invalidation_false_positive_count=false_positive,
        invalidation_false_negative_count=false_negative,
        seeded_expected_by_truth=tuple(
            (truth.value, expected_truth[truth.value])
            for truth in FindingTruth
        ),
        seeded_covered_by_truth=tuple(
            (truth.value, covered_truth[truth.value])
            for truth in FindingTruth
        ),
        counterexample_count=len(counterexamples),
        median_counterexample_time_ns=_fraction_dict(counterexample_median),
        artifact_bytes=sum(
            item.resources.artifact_bytes for item in observations
        ),
        wall_time_ns=sum(
            item.resources.wall_time_ns for item in observations
        ),
        cpu_time_ns=sum(
            item.resources.cpu_time_ns for item in observations
        ),
        scan_wall_time_ns_by_mode=tuple(
            (
                mode,
                sum(
                    item.resources.wall_time_ns
                    for item in observations
                    if item.mode.value == mode
                ),
            )
            for mode in REQUIRED_SCAN_MODES
        ),
        scan_cpu_time_ns_by_mode=tuple(
            (
                mode,
                sum(
                    item.resources.cpu_time_ns
                    for item in observations
                    if item.mode.value == mode
                ),
            )
            for mode in REQUIRED_SCAN_MODES
        ),
        peak_rss_bytes=max(
            item.resources.peak_rss_bytes for item in observations
        ),
        peak_process_count=max(
            item.resources.peak_process_count for item in observations
        ),
        disk_growth_bytes=sum(
            item.resources.disk_growth_bytes for item in observations
        ),
        idle_cpu_time_ns=sum(
            item.resources.idle_cpu_time_ns for item in observations
        ),
        idle_write_operations=sum(
            item.resources.idle_write_operations for item in observations
        ),
        idle_write_bytes=sum(
            item.resources.idle_write_bytes for item in observations
        ),
        candidate_findings=sum(
            item.tasks.candidate_findings for item in observations
        ),
        eligible_findings=sum(
            item.tasks.eligible_findings for item in observations
        ),
        emitted_tasks=sum(
            item.tasks.emitted_tasks for item in observations
        ),
        deduplicated_findings=sum(
            item.tasks.deduplicated_findings for item in observations
        ),
        provider_pair_count=len(observations),
        median_baseline_input_bytes=_fraction_dict(median_baseline_bytes),
        median_packet_input_bytes=_fraction_dict(median_packet_bytes),
        median_baseline_input_tokens=_fraction_dict(median_baseline_tokens),
        median_packet_input_tokens=_fraction_dict(median_packet_tokens),
        provider_byte_reduction_basis_points=byte_reduction,
        provider_token_reduction_basis_points=token_reduction,
        deterministic_llm_calls=deterministic_calls,
        gates=tuple(gates),
        conclusion=conclusion,
        failure_codes=tuple(hard_failures),
    )
    if len(_canonical_bytes(report.to_dict())) > MAX_REPORT_BYTES:
        raise SymbolicBenchmarkError("benchmark report exceeds byte bound")
    return report


def build_symbolic_efficiency_report(
    observations: Sequence[SymbolicBenchmarkObservation]
    | SymbolicBenchmarkPopulation,
    *,
    profile: BenchmarkProfile | None = None,
) -> SymbolicEfficiencyBenchmarkReport:
    """Convenience entry point that still closes and identifies the population."""

    if isinstance(observations, SymbolicBenchmarkPopulation):
        if profile is not None and profile != observations.profile:
            raise SymbolicBenchmarkError(
                "profile conflicts with the closed population"
            )
        population = observations
    else:
        if profile is None:
            raise SymbolicBenchmarkError(
                "profile is required with raw observations"
            )
        population = SymbolicBenchmarkPopulation(
            profile=profile,
            observations=tuple(observations),
        )
    return evaluate_symbolic_efficiency(population)


def verify_symbolic_efficiency_report(
    report: SymbolicEfficiencyBenchmarkReport,
    population: SymbolicBenchmarkPopulation,
) -> bool:
    """Return whether a report exactly matches deterministic replay."""

    if not isinstance(report, SymbolicEfficiencyBenchmarkReport):
        return False
    try:
        replayed = evaluate_symbolic_efficiency(population)
    except SymbolicBenchmarkError:
        return False
    return _canonical_bytes(report.to_dict()) == _canonical_bytes(
        replayed.to_dict()
    )


__all__ = [
    "BenchmarkConclusion",
    "BenchmarkGate",
    "BenchmarkProfile",
    "CacheMeasurement",
    "DETERMINISTIC_STAGE_NAMES",
    "FindingMeasurement",
    "FindingTruth",
    "FixtureIdentity",
    "GateStatus",
    "InvalidationMeasurement",
    "InventoryMeasurement",
    "ProviderPacketMeasurement",
    "REQUIRED_CACHE_STAGES",
    "REQUIRED_SCAN_MODES",
    "ResourceMeasurement",
    "SYMBOLIC_BENCHMARK_EVIDENCE_SCHEMA",
    "SYMBOLIC_BENCHMARK_OBSERVATION_SCHEMA",
    "SYMBOLIC_BENCHMARK_POPULATION_SCHEMA",
    "ScanMode",
    "SymbolicBenchmarkError",
    "SymbolicBenchmarkObservation",
    "SymbolicBenchmarkPopulation",
    "SymbolicEfficiencyBenchmarkReport",
    "TaskMeasurement",
    "ToolchainIdentity",
    "build_symbolic_efficiency_report",
    "evaluate_symbolic_efficiency",
    "verify_symbolic_efficiency_report",
]
