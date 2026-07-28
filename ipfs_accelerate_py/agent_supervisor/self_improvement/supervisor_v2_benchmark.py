"""Closed generation-2 supervisor benchmark and causal baseline.

This module is a measurement boundary, not a supervisor or rollout policy.  It
freezes a paired fixture population, adapts generation-1 efficiency receipts
into a smaller causal observation, and deterministically recomputes a report.

The persisted objects intentionally contain identities, counters, durations,
statuses, and content digests only.  Prompts, source bodies, decoded provider
output, patches, and artifact graphs are outside this contract.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Final


V2_BENCHMARK_CONTRACT_VERSION: Final = 1
V2_BENCHMARK_CORPUS_VERSION: Final = "generation-2@1"
V2_CAUSAL_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/v2-causal-receipt@1"
)
V2_PAIRED_CASE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/v2-paired-benchmark-case@1"
)
V2_PAIRED_CORPUS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/v2-paired-benchmark-corpus@1"
)
V2_BENCHMARK_REPORT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/v2-benchmark-report@1"
)

# Producer-owned evidence identity referenced by ASI-G200.  A string constant
# is used to match the repository's existing objective-evidence vocabulary.
V2_PAIRED_BASELINE_REQUIREMENT_ID: Final = (
    "278718548022626862973862211376400215092"
)
V2_PAIRED_BASELINE_GOAL_ID: Final = "ASI-G200"

# The benchmark envelope is intentionally literal rather than derived from the
# current checkout.  Changing any identity requires a corpus-version change.
V2_FROZEN_REPOSITORY_ID: Final = "repository:supervisor-v2-benchmark@1"
V2_FROZEN_TREE_ID: Final = (
    "sha256:f993b0b7bda1c3c13bcf4a352f45c286f72d2e40783ff79ac75c50fd991ed7d1"
)
V2_FROZEN_OBJECTIVE_ID: Final = V2_PAIRED_BASELINE_GOAL_ID
V2_FROZEN_OBJECTIVE_REVISION: Final = (
    "sha256:51279ffa477f90486f0eee8ef6209465a3089c053797296df6b0d83949b689a4"
)
V2_FROZEN_PROVIDER_ID: Final = "provider:deterministic-fixture@1"
V2_FROZEN_PROVIDER_REVISION: Final = (
    "sha256:f8617cff2880336f71a02089f1bb12691ef8d4b49fba549e88cf1e82625e01d6"
)
V2_FROZEN_CAPABILITY_ID: Final = "capability:supervisor-v2-measurement@1"
V2_FROZEN_CAPABILITY_REVISION: Final = (
    "sha256:13ad5b4964e4402d48638a8c07520f32b0c1d3ff1e7869ed3b38f9b7bd4b1e53"
)
V2_FROZEN_POLICY_ID: Final = "policy:supervisor-v2-benchmark@1"
V2_FROZEN_POLICY_REVISION: Final = (
    "sha256:e9f9146533b39e99ceb8ca1e41a6ce92765dc4a3a627f1737d4951e52e4dca54"
)
V2_FROZEN_CORPUS_ID: Final = (
    "sha256:5e8a8bea7db6353aa12aed6c7ddf185ab45b7a74ef94da7cc21526c52f1f4c8e"
)
V2_CAUSAL_BASELINE_REPORT_ID: Final = (
    "sha256:f13bb78bf21c6afdb54da314e7eedcfd88e42aa409891f2be108def76b877671"
)

MAX_V2_RECEIPT_BYTES: Final = 65_536
MAX_V2_CORPUS_BYTES: Final = 2 * 1024 * 1024
MAX_V2_REPORT_BYTES: Final = 262_144
MAX_V2_STAGES: Final = 32
MAX_V2_CRITERIA: Final = 128
MAX_V2_CAUSES: Final = 16
MAX_V2_COUNTER: Final = 10**15
MAX_V2_DURATION_MS: Final = 31 * 24 * 60 * 60 * 1000
MAX_V2_ARTIFACT_COUNT: Final = 256
MAX_V2_ARTIFACT_BYTES: Final = 4 * 1024 * 1024
MAX_DRAINED_IDLE_CPU_MILLI_PERCENT: Final = 2_000
MIN_DRAINED_OBSERVATION_MS: Final = 10 * 60 * 1000

_CONTENT_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_CODE = re.compile(r"^[a-z][a-z0-9_.:/@-]{0,191}$")
_FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "prompt",
        "prompts",
        "source_body",
        "source_bodies",
        "decoded_output",
        "decoded_outputs",
        "patch",
        "patches",
        "artifact_graph",
        "artifact_graphs",
        "nested_artifact_graph",
    }
)


class V2BenchmarkValidationError(ValueError):
    """A generation-2 benchmark object is malformed or detached."""


class V2FixtureKind(str, Enum):
    """The exact closed generation-2 paired fixture taxonomy."""

    COLD = "cold"
    WARM = "warm"
    BROAD_GOAL = "broad-goal"
    MALFORMED_OUTPUT = "malformed-output"
    CONTRADICTORY_INPUT = "contradictory-input"
    STALE_CACHE = "stale-cache"
    UNAVAILABLE_PROVIDER = "unavailable-provider"
    INDEPENDENT_LANE = "independent-lane"
    CONFLICTING_LANE = "conflicting-lane"
    FAILED_VALIDATION = "failed-validation"
    RESTART = "restart"
    DRAINED_BOARD = "drained-board"
    ARTIFACT_PRESSURE = "artifact-pressure"
    UNTRUSTED_REPOSITORY = "untrusted-repository"


REQUIRED_V2_FIXTURE_KINDS: Final[tuple[V2FixtureKind, ...]] = tuple(
    V2FixtureKind
)


class V2BenchmarkArm(str, Enum):
    BASELINE = "baseline"
    CANDIDATE = "candidate"


class V2GateStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_REQUIRED = "not-required"


V2_NON_COMPENSABLE_SAFETY_GATES: Final[tuple[str, ...]] = (
    "authority",
    "terminal-criteria",
    "validation",
    "proof",
    "merge",
    "persistence",
    "cache-authority",
    "restart-recovery",
    "idle-board",
    "artifact-bounds",
    "repository-trust",
)

_EXPECTED_TERMINAL_OUTCOME: Final[dict[V2FixtureKind, str]] = {
    V2FixtureKind.COLD: "accepted",
    V2FixtureKind.WARM: "accepted",
    V2FixtureKind.BROAD_GOAL: "rejected",
    V2FixtureKind.MALFORMED_OUTPUT: "rejected",
    V2FixtureKind.CONTRADICTORY_INPUT: "rejected",
    V2FixtureKind.STALE_CACHE: "accepted",
    V2FixtureKind.UNAVAILABLE_PROVIDER: "degraded",
    V2FixtureKind.INDEPENDENT_LANE: "accepted",
    V2FixtureKind.CONFLICTING_LANE: "rejected",
    V2FixtureKind.FAILED_VALIDATION: "rejected",
    V2FixtureKind.RESTART: "accepted",
    V2FixtureKind.DRAINED_BOARD: "idle",
    V2FixtureKind.ARTIFACT_PRESSURE: "rejected",
    V2FixtureKind.UNTRUSTED_REPOSITORY: "rejected",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            _jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise V2BenchmarkValidationError(
            "benchmark data must be canonical JSON"
        ) from exc


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _fixture_digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _text(value: Any, name: str, *, maximum: int = 512) -> str:
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str) or not value.strip():
        raise V2BenchmarkValidationError(f"{name} must be non-empty text")
    result = value.strip()
    if "\x00" in result or len(result.encode("utf-8")) > maximum:
        raise V2BenchmarkValidationError(
            f"{name} is unsafe or exceeds its {maximum}-byte bound"
        )
    return result


def _code(value: Any, name: str) -> str:
    result = _text(value, name, maximum=192).lower()
    if not _CODE.fullmatch(result):
        raise V2BenchmarkValidationError(f"{name} must be a compact code")
    return result


def _content_id(value: Any, name: str) -> str:
    result = _text(value, name, maximum=71).lower()
    if not _CONTENT_ID.fullmatch(result):
        raise V2BenchmarkValidationError(
            f"{name} must be a sha256 content ID"
        )
    return result


def _integer(
    value: Any,
    name: str,
    *,
    maximum: int = MAX_V2_COUNTER,
    minimum: int = 0,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        raise V2BenchmarkValidationError(
            f"{name} must be an integer from {minimum} through {maximum}"
        )
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise V2BenchmarkValidationError(f"{name} must be a boolean")
    return value


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    raw = getattr(value, "value", value)
    try:
        return enum_type(str(raw))
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise V2BenchmarkValidationError(
            f"{name} must be one of: {allowed}"
        ) from exc


def _codes(
    values: Sequence[Any],
    name: str,
    *,
    maximum: int,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise V2BenchmarkValidationError(f"{name} must be a sequence")
    if len(values) > maximum:
        raise V2BenchmarkValidationError(
            f"{name} exceeds its {maximum}-item bound"
        )
    result = tuple(sorted(_code(value, name) for value in values))
    if len(result) != len(set(result)):
        raise V2BenchmarkValidationError(f"{name} must be unique")
    return result


def _ordered_codes(
    values: Sequence[Any],
    name: str,
    *,
    maximum: int,
) -> tuple[str, ...]:
    """Validate compact identifiers while preserving meaningful pair order."""

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise V2BenchmarkValidationError(f"{name} must be a sequence")
    if len(values) > maximum:
        raise V2BenchmarkValidationError(
            f"{name} exceeds its {maximum}-item bound"
        )
    result = tuple(_code(value, name) for value in values)
    if len(result) != len(set(result)):
        raise V2BenchmarkValidationError(f"{name} must be unique")
    return result


def _strict_keys(
    payload: Mapping[str, Any],
    allowed: set[str],
    *,
    name: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise V2BenchmarkValidationError(f"{name} must be an object")
    extras = sorted(set(payload) - allowed)
    missing = sorted(allowed - set(payload))
    if extras or missing:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extras:
            details.append("unexpected " + ", ".join(extras))
        raise V2BenchmarkValidationError(
            f"{name} has invalid fields: {'; '.join(details)}"
        )


def _reject_forbidden_payload(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _FORBIDDEN_PAYLOAD_KEYS or normalized.endswith(
                "_body"
            ):
                raise V2BenchmarkValidationError(
                    f"benchmark payload cannot contain {key!r}"
                )
            _reject_forbidden_payload(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_forbidden_payload(item)


def _load_json(value: str | bytes | bytearray, *, name: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise V2BenchmarkValidationError(
                    f"{name} JSON contains duplicate object keys"
                )
            result[key] = item
        return result

    try:
        if isinstance(value, (bytes, bytearray)):
            value = bytes(value).decode("utf-8")
        if not isinstance(value, str):
            raise V2BenchmarkValidationError(f"{name} JSON must be text")
        result = json.loads(value, object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V2BenchmarkValidationError(f"{name} JSON is invalid") from exc
    _reject_forbidden_payload(result)
    return result


@dataclass(frozen=True)
class V2FrozenIdentity:
    """All semantic identities required to interpret one observation."""

    repository_id: str
    tree_id: str
    objective_id: str
    objective_revision: str
    provider_id: str
    provider_revision: str
    capability_id: str
    capability_revision: str
    policy_id: str
    policy_revision: str
    fault_id: str
    observation_id: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), name, maximum=256),
            )

    @property
    def pairing_identity(self) -> tuple[str, ...]:
        return tuple(
            getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "observation_id"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "V2FrozenIdentity":
        allowed = set(cls.__dataclass_fields__)
        _strict_keys(payload, allowed, name="v2 frozen identity")
        return cls(**{name: payload[name] for name in allowed})


@dataclass(frozen=True)
class V2StageLatency:
    """One compact stage-duration join."""

    stage: str
    latency_ms: int
    invocation_count: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", _code(self.stage, "stage"))
        object.__setattr__(
            self,
            "latency_ms",
            _integer(
                self.latency_ms,
                "latency_ms",
                maximum=MAX_V2_DURATION_MS,
            ),
        )
        object.__setattr__(
            self,
            "invocation_count",
            _integer(
                self.invocation_count,
                "invocation_count",
                minimum=1,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "latency_ms": self.latency_ms,
            "invocation_count": self.invocation_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "V2StageLatency":
        allowed = {"stage", "latency_ms", "invocation_count"}
        _strict_keys(payload, allowed, name="v2 stage latency")
        return cls(**{name: payload[name] for name in allowed})


@dataclass(frozen=True)
class V2CausalMetrics:
    """Bounded joined observation for one benchmark arm.

    Artifact and provider data are summaries.  Criterion values are compact
    stable identifiers, never the criterion or evidence bodies.
    """

    stage_latencies: tuple[V2StageLatency, ...]
    elapsed_ms: int
    queue_delay_ms: int
    provider_input_tokens: int
    provider_output_tokens: int
    provider_reused_input_tokens: int
    cache_lookup_count: int
    cache_hit_count: int
    cache_reused_bytes: int
    retry_count: int
    retry_input_tokens: int
    retry_output_tokens: int
    validation_status: V2GateStatus
    validation_latency_ms: int
    proof_status: V2GateStatus
    proof_latency_ms: int
    merge_status: V2GateStatus
    merge_latency_ms: int
    persistence_status: V2GateStatus
    persistence_latency_ms: int
    persistence_write_count: int
    persistence_bytes: int
    idle_cpu_milli_percent: int
    idle_observation_ms: int
    artifact_count: int
    artifact_bytes: int
    artifact_manifest_digest: str
    expected_terminal_outcome: str
    terminal_outcome: str
    required_criterion_ids: tuple[str, ...]
    terminal_accepted_criterion_ids: tuple[str, ...]
    authority_violation_count: int = 0
    false_completion_count: int = 0
    stale_authoritative_cache_hit_count: int = 0
    escaped_validation_failure_count: int = 0
    escaped_proof_failure_count: int = 0
    merge_safety_violation_count: int = 0
    persistence_loss_count: int = 0
    restart_inconsistency_count: int = 0
    unbounded_artifact_count: int = 0
    untrusted_repository_mutation_count: int = 0

    def __post_init__(self) -> None:
        stages: list[V2StageLatency] = []
        if (
            isinstance(self.stage_latencies, (str, bytes))
            or not isinstance(self.stage_latencies, Sequence)
            or len(self.stage_latencies) > MAX_V2_STAGES
        ):
            raise V2BenchmarkValidationError(
                "stage_latencies must be a bounded sequence"
            )
        for item in self.stage_latencies:
            if isinstance(item, Mapping):
                item = V2StageLatency.from_dict(item)
            if not isinstance(item, V2StageLatency):
                raise V2BenchmarkValidationError(
                    "stage_latencies must contain V2StageLatency values"
                )
            stages.append(item)
        stages.sort(key=lambda item: item.stage)
        if len({item.stage for item in stages}) != len(stages):
            raise V2BenchmarkValidationError(
                "stage_latencies must have unique stage names"
            )
        object.__setattr__(self, "stage_latencies", tuple(stages))

        for name in (
            "elapsed_ms",
            "queue_delay_ms",
            "validation_latency_ms",
            "proof_latency_ms",
            "merge_latency_ms",
            "persistence_latency_ms",
            "idle_observation_ms",
        ):
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    name,
                    maximum=MAX_V2_DURATION_MS,
                ),
            )
        if self.queue_delay_ms > self.elapsed_ms:
            raise V2BenchmarkValidationError(
                "queue_delay_ms cannot exceed elapsed_ms"
            )
        if any(item.latency_ms > self.elapsed_ms for item in stages):
            raise V2BenchmarkValidationError(
                "stage latency cannot exceed elapsed_ms"
            )

        for name in (
            "provider_input_tokens",
            "provider_output_tokens",
            "provider_reused_input_tokens",
            "cache_lookup_count",
            "cache_hit_count",
            "cache_reused_bytes",
            "retry_count",
            "retry_input_tokens",
            "retry_output_tokens",
            "persistence_write_count",
            "persistence_bytes",
            "artifact_count",
            "artifact_bytes",
            "authority_violation_count",
            "false_completion_count",
            "stale_authoritative_cache_hit_count",
            "escaped_validation_failure_count",
            "escaped_proof_failure_count",
            "merge_safety_violation_count",
            "persistence_loss_count",
            "restart_inconsistency_count",
            "unbounded_artifact_count",
            "untrusted_repository_mutation_count",
        ):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "idle_cpu_milli_percent",
            _integer(
                self.idle_cpu_milli_percent,
                "idle_cpu_milli_percent",
                maximum=100_000,
            ),
        )
        if self.provider_reused_input_tokens > self.provider_input_tokens:
            raise V2BenchmarkValidationError(
                "provider_reused_input_tokens cannot exceed input tokens"
            )
        if self.cache_hit_count > self.cache_lookup_count:
            raise V2BenchmarkValidationError(
                "cache_hit_count cannot exceed cache_lookup_count"
            )
        if self.retry_input_tokens > self.provider_input_tokens:
            raise V2BenchmarkValidationError(
                "retry_input_tokens cannot exceed provider input tokens"
            )
        if self.retry_output_tokens > self.provider_output_tokens:
            raise V2BenchmarkValidationError(
                "retry_output_tokens cannot exceed provider output tokens"
            )

        for name in (
            "validation_status",
            "proof_status",
            "merge_status",
            "persistence_status",
        ):
            object.__setattr__(
                self,
                name,
                _enum(getattr(self, name), V2GateStatus, name),
            )
        for status_name, latency_name in (
            ("validation_status", "validation_latency_ms"),
            ("proof_status", "proof_latency_ms"),
            ("merge_status", "merge_latency_ms"),
            ("persistence_status", "persistence_latency_ms"),
        ):
            if (
                getattr(self, status_name) is V2GateStatus.NOT_REQUIRED
                and getattr(self, latency_name)
            ):
                raise V2BenchmarkValidationError(
                    f"{latency_name} must be zero when work is not required"
                )

        object.__setattr__(
            self,
            "artifact_manifest_digest",
            _content_id(
                self.artifact_manifest_digest, "artifact_manifest_digest"
            ),
        )
        for name in ("expected_terminal_outcome", "terminal_outcome"):
            object.__setattr__(
                self, name, _code(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "required_criterion_ids",
            _codes(
                self.required_criterion_ids,
                "required_criterion_ids",
                maximum=MAX_V2_CRITERIA,
            ),
        )
        object.__setattr__(
            self,
            "terminal_accepted_criterion_ids",
            _codes(
                self.terminal_accepted_criterion_ids,
                "terminal_accepted_criterion_ids",
                maximum=MAX_V2_CRITERIA,
            ),
        )
        if not self.required_criterion_ids:
            raise V2BenchmarkValidationError(
                "each observation requires at least one terminal criterion"
            )
        if not set(self.terminal_accepted_criterion_ids).issubset(
            self.required_criterion_ids
        ):
            raise V2BenchmarkValidationError(
                "terminal accepted criteria must be drawn from the frozen "
                "required criterion population"
            )

    @property
    def stage_latency_ms(self) -> int:
        return sum(item.latency_ms for item in self.stage_latencies)

    @property
    def terminal_criteria_complete(self) -> bool:
        return set(self.required_criterion_ids).issubset(
            self.terminal_accepted_criterion_ids
        )

    @property
    def artifact_bounds_passed(self) -> bool:
        return (
            self.artifact_count <= MAX_V2_ARTIFACT_COUNT
            and self.artifact_bytes <= MAX_V2_ARTIFACT_BYTES
            and self.unbounded_artifact_count == 0
        )

    def safety_gates(
        self, fixture_kind: V2FixtureKind | str
    ) -> dict[str, bool]:
        kind = _enum(fixture_kind, V2FixtureKind, "fixture_kind")
        accepted = self.terminal_outcome == "accepted"
        return {
            "authority": (
                self.authority_violation_count == 0
                and self.false_completion_count == 0
            ),
            "terminal-criteria": (
                self.terminal_outcome == self.expected_terminal_outcome
                and self.terminal_criteria_complete
            ),
            "validation": (
                self.escaped_validation_failure_count == 0
                and (
                    not accepted
                    or self.validation_status is V2GateStatus.PASSED
                )
            ),
            "proof": (
                self.escaped_proof_failure_count == 0
                and (
                    not accepted
                    or self.proof_status
                    in {V2GateStatus.PASSED, V2GateStatus.NOT_REQUIRED}
                )
            ),
            "merge": (
                self.merge_safety_violation_count == 0
                and (
                    not accepted
                    or self.merge_status
                    in {V2GateStatus.PASSED, V2GateStatus.NOT_REQUIRED}
                )
            ),
            "persistence": (
                self.persistence_loss_count == 0
                and self.persistence_status is V2GateStatus.PASSED
            ),
            "cache-authority": (
                self.stale_authoritative_cache_hit_count == 0
            ),
            "restart-recovery": (
                self.restart_inconsistency_count == 0
            ),
            "idle-board": (
                kind is not V2FixtureKind.DRAINED_BOARD
                or (
                    self.idle_observation_ms >= MIN_DRAINED_OBSERVATION_MS
                    and self.idle_cpu_milli_percent
                    < MAX_DRAINED_IDLE_CPU_MILLI_PERCENT
                    and self.persistence_write_count == 0
                )
            ),
            "artifact-bounds": self.artifact_bounds_passed,
            "repository-trust": (
                self.untrusted_repository_mutation_count == 0
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            name: _jsonable(getattr(self, name))
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "V2CausalMetrics":
        _reject_forbidden_payload(payload)
        allowed = set(cls.__dataclass_fields__)
        _strict_keys(payload, allowed, name="v2 causal metrics")
        values = {name: payload[name] for name in allowed}
        values["stage_latencies"] = tuple(
            V2StageLatency.from_dict(item)
            for item in values["stage_latencies"]
        )
        return cls(**values)


@dataclass(frozen=True)
class V2CausalReceipt:
    """Compact content-addressed observation for one fixture arm."""

    fixture_kind: V2FixtureKind
    arm: V2BenchmarkArm
    identity: V2FrozenIdentity
    input_id: str
    metrics: V2CausalMetrics
    source_receipt_ids: tuple[str, ...] = ()
    causal_parent_ids: tuple[str, ...] = ()
    intervention_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fixture_kind",
            _enum(self.fixture_kind, V2FixtureKind, "fixture_kind"),
        )
        object.__setattr__(
            self, "arm", _enum(self.arm, V2BenchmarkArm, "arm")
        )
        identity = self.identity
        if isinstance(identity, Mapping):
            identity = V2FrozenIdentity.from_dict(identity)
        if not isinstance(identity, V2FrozenIdentity):
            raise V2BenchmarkValidationError(
                "identity must be V2FrozenIdentity"
            )
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "input_id", _content_id(self.input_id, "input_id"))
        metrics = self.metrics
        if isinstance(metrics, Mapping):
            metrics = V2CausalMetrics.from_dict(metrics)
        if not isinstance(metrics, V2CausalMetrics):
            raise V2BenchmarkValidationError(
                "metrics must be V2CausalMetrics"
            )
        if metrics.expected_terminal_outcome != _EXPECTED_TERMINAL_OUTCOME[
            self.fixture_kind
        ]:
            raise V2BenchmarkValidationError(
                "expected terminal outcome is not frozen for fixture kind"
            )
        object.__setattr__(self, "metrics", metrics)
        for name, maximum in (
            ("source_receipt_ids", MAX_V2_CAUSES),
            ("causal_parent_ids", MAX_V2_CAUSES),
            ("intervention_ids", MAX_V2_CAUSES),
        ):
            object.__setattr__(
                self,
                name,
                _codes(getattr(self, name), name, maximum=maximum),
            )
        if not self.intervention_ids:
            raise V2BenchmarkValidationError(
                "causal receipts require a frozen intervention identity"
            )
        if self.arm is V2BenchmarkArm.BASELINE and self.causal_parent_ids:
            raise V2BenchmarkValidationError(
                "baseline receipts cannot have a paired causal parent"
            )
        if len(self.canonical_bytes()) > MAX_V2_RECEIPT_BYTES:
            raise V2BenchmarkValidationError(
                "v2 causal receipt exceeds its serialized byte bound"
            )

    @property
    def receipt_id(self) -> str:
        return _digest(self.to_dict())

    @property
    def safety_gates(self) -> dict[str, bool]:
        return self.metrics.safety_gates(self.fixture_kind)

    @property
    def safety_passed(self) -> bool:
        return all(self.safety_gates.values())

    def to_dict(self, *, include_receipt_id: bool = False) -> dict[str, Any]:
        payload = {
            "schema": V2_CAUSAL_RECEIPT_SCHEMA,
            "contract_version": V2_BENCHMARK_CONTRACT_VERSION,
            "corpus_version": V2_BENCHMARK_CORPUS_VERSION,
            "fixture_kind": self.fixture_kind.value,
            "arm": self.arm.value,
            "identity": self.identity.to_dict(),
            "input_id": self.input_id,
            "metrics": self.metrics.to_dict(),
            "source_receipt_ids": list(self.source_receipt_ids),
            "causal_parent_ids": list(self.causal_parent_ids),
            "intervention_ids": list(self.intervention_ids),
        }
        if include_receipt_id:
            payload["receipt_id"] = self.receipt_id
        return payload

    def canonical_bytes(self) -> bytes:
        return _canonical_json(self.to_dict()).encode("utf-8")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "V2CausalReceipt":
        _reject_forbidden_payload(payload)
        allowed = {
            "schema",
            "contract_version",
            "corpus_version",
            "fixture_kind",
            "arm",
            "identity",
            "input_id",
            "metrics",
            "source_receipt_ids",
            "causal_parent_ids",
            "intervention_ids",
            "receipt_id",
        }
        required = allowed - {"receipt_id"}
        _strict_keys(
            payload,
            required if "receipt_id" not in payload else allowed,
            name="v2 causal receipt",
        )
        if payload["schema"] != V2_CAUSAL_RECEIPT_SCHEMA:
            raise V2BenchmarkValidationError(
                "unsupported v2 causal receipt schema"
            )
        if payload["contract_version"] != V2_BENCHMARK_CONTRACT_VERSION:
            raise V2BenchmarkValidationError(
                "unsupported v2 benchmark contract version"
            )
        if payload["corpus_version"] != V2_BENCHMARK_CORPUS_VERSION:
            raise V2BenchmarkValidationError(
                "unsupported v2 benchmark corpus version"
            )
        result = cls(
            fixture_kind=payload["fixture_kind"],
            arm=payload["arm"],
            identity=V2FrozenIdentity.from_dict(payload["identity"]),
            input_id=payload["input_id"],
            metrics=V2CausalMetrics.from_dict(payload["metrics"]),
            source_receipt_ids=tuple(payload["source_receipt_ids"]),
            causal_parent_ids=tuple(payload["causal_parent_ids"]),
            intervention_ids=tuple(payload["intervention_ids"]),
        )
        if payload.get("receipt_id", result.receipt_id) != result.receipt_id:
            raise V2BenchmarkValidationError(
                "v2 causal receipt identity does not match payload"
            )
        return result

    @classmethod
    def from_json(
        cls, value: str | bytes | bytearray
    ) -> "V2CausalReceipt":
        return cls.from_dict(_load_json(value, name="v2 causal receipt"))


def _fixture_id(kind: V2FixtureKind) -> str:
    return f"fixture:supervisor-v2:{kind.value}@1"


def _fault_id(kind: V2FixtureKind) -> str:
    return f"fault:supervisor-v2:{kind.value}@1"


def _observation_id(kind: V2FixtureKind, arm: V2BenchmarkArm) -> str:
    return f"observation:supervisor-v2:{kind.value}:{arm.value}@1"


@dataclass(frozen=True)
class V2PairedBenchmarkCase:
    """Baseline and candidate observations for exactly the same fixture."""

    fixture_id: str
    fixture_kind: V2FixtureKind
    fixture_revision: str
    baseline: V2CausalReceipt
    candidate: V2CausalReceipt

    def __post_init__(self) -> None:
        kind = _enum(self.fixture_kind, V2FixtureKind, "fixture_kind")
        object.__setattr__(self, "fixture_kind", kind)
        if self.fixture_id != _fixture_id(kind):
            raise V2BenchmarkValidationError(
                "fixture_id is not the frozen identity for fixture kind"
            )
        object.__setattr__(
            self,
            "fixture_revision",
            _text(self.fixture_revision, "fixture_revision", maximum=64),
        )
        if self.fixture_revision != V2_BENCHMARK_CORPUS_VERSION:
            raise V2BenchmarkValidationError(
                "fixture revision does not match frozen corpus version"
            )
        for name, arm in (
            ("baseline", V2BenchmarkArm.BASELINE),
            ("candidate", V2BenchmarkArm.CANDIDATE),
        ):
            value = getattr(self, name)
            if isinstance(value, Mapping):
                value = V2CausalReceipt.from_dict(value)
            if not isinstance(value, V2CausalReceipt):
                raise V2BenchmarkValidationError(
                    f"{name} must be V2CausalReceipt"
                )
            if value.fixture_kind is not kind or value.arm is not arm:
                raise V2BenchmarkValidationError(
                    f"{name} receipt is detached from its fixture arm"
                )
            object.__setattr__(self, name, value)
        if self.baseline.input_id != self.candidate.input_id:
            raise V2BenchmarkValidationError(
                "paired receipts must freeze the same input identity"
            )
        if (
            self.baseline.identity.pairing_identity
            != self.candidate.identity.pairing_identity
        ):
            raise V2BenchmarkValidationError(
                "paired receipts must freeze identical semantic identities"
            )
        if self.baseline.identity.fault_id != _fault_id(kind):
            raise V2BenchmarkValidationError(
                "fixture fault identity is not frozen"
            )
        for receipt, arm in (
            (self.baseline, V2BenchmarkArm.BASELINE),
            (self.candidate, V2BenchmarkArm.CANDIDATE),
        ):
            if receipt.identity.observation_id != _observation_id(kind, arm):
                raise V2BenchmarkValidationError(
                    "fixture observation identity is not frozen"
                )
        if self.candidate.causal_parent_ids != (self.baseline.receipt_id,):
            raise V2BenchmarkValidationError(
                "candidate must causally reference its paired baseline"
            )
        if len(self.canonical_bytes()) > 2 * MAX_V2_RECEIPT_BYTES:
            raise V2BenchmarkValidationError(
                "v2 paired case exceeds its serialized byte bound"
            )

    @property
    def case_id(self) -> str:
        return _digest(self.to_dict())

    @property
    def pair_integrity_passed(self) -> bool:
        return True

    def to_dict(self, *, include_case_id: bool = False) -> dict[str, Any]:
        payload = {
            "schema": V2_PAIRED_CASE_SCHEMA,
            "fixture_id": self.fixture_id,
            "fixture_kind": self.fixture_kind.value,
            "fixture_revision": self.fixture_revision,
            "baseline": self.baseline.to_dict(include_receipt_id=True),
            "candidate": self.candidate.to_dict(include_receipt_id=True),
        }
        if include_case_id:
            payload["case_id"] = self.case_id
        return payload

    def canonical_bytes(self) -> bytes:
        return _canonical_json(self.to_dict()).encode("utf-8")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "V2PairedBenchmarkCase":
        _reject_forbidden_payload(payload)
        allowed = {
            "schema",
            "fixture_id",
            "fixture_kind",
            "fixture_revision",
            "baseline",
            "candidate",
            "case_id",
        }
        required = allowed - {"case_id"}
        _strict_keys(
            payload,
            required if "case_id" not in payload else allowed,
            name="v2 paired benchmark case",
        )
        if payload["schema"] != V2_PAIRED_CASE_SCHEMA:
            raise V2BenchmarkValidationError(
                "unsupported v2 paired benchmark case schema"
            )
        result = cls(
            fixture_id=payload["fixture_id"],
            fixture_kind=payload["fixture_kind"],
            fixture_revision=payload["fixture_revision"],
            baseline=V2CausalReceipt.from_dict(payload["baseline"]),
            candidate=V2CausalReceipt.from_dict(payload["candidate"]),
        )
        if payload.get("case_id", result.case_id) != result.case_id:
            raise V2BenchmarkValidationError(
                "v2 paired case identity does not match payload"
            )
        return result


@dataclass(frozen=True)
class V2PairedBenchmarkCorpus:
    """Immutable, non-narrowable generation-2 paired population."""

    cases: tuple[V2PairedBenchmarkCase, ...]
    corpus_version: str = V2_BENCHMARK_CORPUS_VERSION
    requirement_id: str = V2_PAIRED_BASELINE_REQUIREMENT_ID

    def __post_init__(self) -> None:
        if self.corpus_version != V2_BENCHMARK_CORPUS_VERSION:
            raise V2BenchmarkValidationError(
                "unsupported v2 paired corpus version"
            )
        if self.requirement_id != V2_PAIRED_BASELINE_REQUIREMENT_ID:
            raise V2BenchmarkValidationError(
                "v2 paired corpus requirement identity is not frozen"
            )
        normalized: list[V2PairedBenchmarkCase] = []
        if isinstance(self.cases, (str, bytes)) or not isinstance(
            self.cases, Sequence
        ):
            raise V2BenchmarkValidationError("cases must be a sequence")
        for value in self.cases:
            if isinstance(value, Mapping):
                value = V2PairedBenchmarkCase.from_dict(value)
            if not isinstance(value, V2PairedBenchmarkCase):
                raise V2BenchmarkValidationError(
                    "cases must contain V2PairedBenchmarkCase values"
                )
            normalized.append(value)
        by_kind = {item.fixture_kind: item for item in normalized}
        if (
            len(by_kind) != len(normalized)
            or set(by_kind) != set(REQUIRED_V2_FIXTURE_KINDS)
        ):
            raise V2BenchmarkValidationError(
                "v2 fixture population is closed and cannot be narrowed, "
                "duplicated, or widened"
            )
        normalized = [by_kind[kind] for kind in REQUIRED_V2_FIXTURE_KINDS]
        object.__setattr__(self, "cases", tuple(normalized))
        if len(self.canonical_bytes()) > MAX_V2_CORPUS_BYTES:
            raise V2BenchmarkValidationError(
                "v2 paired corpus exceeds its serialized byte bound"
            )

    @property
    def fixture_population_ids(self) -> tuple[str, ...]:
        return tuple(item.fixture_id for item in self.cases)

    @property
    def corpus_id(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self, *, include_corpus_id: bool = False) -> dict[str, Any]:
        payload = {
            "schema": V2_PAIRED_CORPUS_SCHEMA,
            "contract_version": V2_BENCHMARK_CONTRACT_VERSION,
            "corpus_version": self.corpus_version,
            "requirement_id": self.requirement_id,
            "fixture_population_ids": list(self.fixture_population_ids),
            "cases": [item.to_dict(include_case_id=True) for item in self.cases],
        }
        if include_corpus_id:
            payload["corpus_id"] = self.corpus_id
        return payload

    def canonical_bytes(self) -> bytes:
        return _canonical_json(self.to_dict()).encode("utf-8")

    def to_json(self, *, include_corpus_id: bool = True) -> str:
        return _canonical_json(self.to_dict(include_corpus_id=include_corpus_id))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "V2PairedBenchmarkCorpus":
        _reject_forbidden_payload(payload)
        allowed = {
            "schema",
            "contract_version",
            "corpus_version",
            "requirement_id",
            "fixture_population_ids",
            "cases",
            "corpus_id",
        }
        required = allowed - {"corpus_id"}
        _strict_keys(
            payload,
            required if "corpus_id" not in payload else allowed,
            name="v2 paired benchmark corpus",
        )
        if payload["schema"] != V2_PAIRED_CORPUS_SCHEMA:
            raise V2BenchmarkValidationError(
                "unsupported v2 paired benchmark corpus schema"
            )
        if payload["contract_version"] != V2_BENCHMARK_CONTRACT_VERSION:
            raise V2BenchmarkValidationError(
                "unsupported v2 benchmark contract version"
            )
        result = cls(
            cases=tuple(
                V2PairedBenchmarkCase.from_dict(item)
                for item in payload["cases"]
            ),
            corpus_version=payload["corpus_version"],
            requirement_id=payload["requirement_id"],
        )
        if tuple(payload["fixture_population_ids"]) != (
            result.fixture_population_ids
        ):
            raise V2BenchmarkValidationError(
                "fixture population identity does not match cases"
            )
        if payload.get("corpus_id", result.corpus_id) != result.corpus_id:
            raise V2BenchmarkValidationError(
                "v2 paired corpus identity does not match payload"
            )
        return result

    @classmethod
    def from_json(
        cls, value: str | bytes | bytearray
    ) -> "V2PairedBenchmarkCorpus":
        return cls.from_dict(_load_json(value, name="v2 paired corpus"))


@dataclass(frozen=True)
class V2AggregateMetrics:
    """Additive arm totals used for causal deltas without composite scoring."""

    stage_latency_ms: int
    elapsed_ms: int
    queue_delay_ms: int
    provider_input_tokens: int
    provider_output_tokens: int
    provider_reused_input_tokens: int
    cache_lookup_count: int
    cache_hit_count: int
    cache_reused_bytes: int
    retry_count: int
    retry_input_tokens: int
    retry_output_tokens: int
    validation_latency_ms: int
    proof_latency_ms: int
    merge_latency_ms: int
    persistence_latency_ms: int
    persistence_write_count: int
    persistence_bytes: int
    idle_cpu_milli_percent: int
    artifact_count: int
    artifact_bytes: int
    terminal_required_criteria: int
    terminal_accepted_criteria: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    name,
                    maximum=MAX_V2_COUNTER,
                ),
            )

    def to_dict(self) -> dict[str, int]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "V2AggregateMetrics":
        allowed = set(cls.__dataclass_fields__)
        _strict_keys(payload, allowed, name="v2 aggregate metrics")
        return cls(**{name: payload[name] for name in allowed})


def _aggregate(
    receipts: Sequence[V2CausalReceipt],
) -> V2AggregateMetrics:
    fields = {
        "stage_latency_ms": sum(
            item.metrics.stage_latency_ms for item in receipts
        ),
        "elapsed_ms": sum(item.metrics.elapsed_ms for item in receipts),
        "queue_delay_ms": sum(
            item.metrics.queue_delay_ms for item in receipts
        ),
        "provider_input_tokens": sum(
            item.metrics.provider_input_tokens for item in receipts
        ),
        "provider_output_tokens": sum(
            item.metrics.provider_output_tokens for item in receipts
        ),
        "provider_reused_input_tokens": sum(
            item.metrics.provider_reused_input_tokens for item in receipts
        ),
        "cache_lookup_count": sum(
            item.metrics.cache_lookup_count for item in receipts
        ),
        "cache_hit_count": sum(
            item.metrics.cache_hit_count for item in receipts
        ),
        "cache_reused_bytes": sum(
            item.metrics.cache_reused_bytes for item in receipts
        ),
        "retry_count": sum(item.metrics.retry_count for item in receipts),
        "retry_input_tokens": sum(
            item.metrics.retry_input_tokens for item in receipts
        ),
        "retry_output_tokens": sum(
            item.metrics.retry_output_tokens for item in receipts
        ),
        "validation_latency_ms": sum(
            item.metrics.validation_latency_ms for item in receipts
        ),
        "proof_latency_ms": sum(
            item.metrics.proof_latency_ms for item in receipts
        ),
        "merge_latency_ms": sum(
            item.metrics.merge_latency_ms for item in receipts
        ),
        "persistence_latency_ms": sum(
            item.metrics.persistence_latency_ms for item in receipts
        ),
        "persistence_write_count": sum(
            item.metrics.persistence_write_count for item in receipts
        ),
        "persistence_bytes": sum(
            item.metrics.persistence_bytes for item in receipts
        ),
        "idle_cpu_milli_percent": sum(
            item.metrics.idle_cpu_milli_percent for item in receipts
        ),
        "artifact_count": sum(
            item.metrics.artifact_count for item in receipts
        ),
        "artifact_bytes": sum(
            item.metrics.artifact_bytes for item in receipts
        ),
        "terminal_required_criteria": sum(
            len(item.metrics.required_criterion_ids) for item in receipts
        ),
        "terminal_accepted_criteria": sum(
            len(item.metrics.terminal_accepted_criterion_ids)
            for item in receipts
        ),
    }
    return V2AggregateMetrics(**fields)


@dataclass(frozen=True)
class V2BenchmarkReport:
    """Deterministic replay result over the complete paired corpus."""

    corpus_id: str
    fixture_population_ids: tuple[str, ...]
    case_ids: tuple[str, ...]
    baseline_receipt_ids: tuple[str, ...]
    candidate_receipt_ids: tuple[str, ...]
    baseline: V2AggregateMetrics
    candidate: V2AggregateMetrics
    candidate_minus_baseline: Mapping[str, int]
    gate_failures: Mapping[str, tuple[str, ...]]
    population_complete: bool
    baseline_candidate_paired: bool
    non_compensable_safety_passed: bool
    passed: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "corpus_id", _content_id(self.corpus_id, "corpus_id"))
        for name in (
            "fixture_population_ids",
            "case_ids",
            "baseline_receipt_ids",
            "candidate_receipt_ids",
        ):
            object.__setattr__(
                self,
                name,
                _ordered_codes(
                    getattr(self, name),
                    name,
                    maximum=len(REQUIRED_V2_FIXTURE_KINDS),
                ),
            )
        expected_population = tuple(
            _fixture_id(kind) for kind in REQUIRED_V2_FIXTURE_KINDS
        )
        calculated_population_complete = (
            self.fixture_population_ids == expected_population
            and len(self.case_ids) == len(expected_population)
            and len(self.baseline_receipt_ids) == len(expected_population)
            and len(self.candidate_receipt_ids) == len(expected_population)
        )
        for name in ("baseline", "candidate"):
            value = getattr(self, name)
            if isinstance(value, Mapping):
                value = V2AggregateMetrics.from_dict(value)
            if not isinstance(value, V2AggregateMetrics):
                raise V2BenchmarkValidationError(
                    f"{name} must be V2AggregateMetrics"
                )
            object.__setattr__(self, name, value)
        expected_delta = {
            name: getattr(self.candidate, name) - getattr(self.baseline, name)
            for name in self.baseline.__dataclass_fields__
        }
        if dict(self.candidate_minus_baseline) != expected_delta:
            raise V2BenchmarkValidationError(
                "causal delta does not match paired aggregate metrics"
            )
        object.__setattr__(
            self, "candidate_minus_baseline", expected_delta
        )
        if set(self.gate_failures) != set(V2_NON_COMPENSABLE_SAFETY_GATES):
            raise V2BenchmarkValidationError(
                "report must contain every non-compensable safety gate"
            )
        normalized_failures: dict[str, tuple[str, ...]] = {}
        population = set(self.fixture_population_ids)
        for name in V2_NON_COMPENSABLE_SAFETY_GATES:
            values = _codes(
                self.gate_failures[name],
                f"gate_failures.{name}",
                maximum=len(REQUIRED_V2_FIXTURE_KINDS),
            )
            if not set(values).issubset(population):
                raise V2BenchmarkValidationError(
                    "gate failures must reference the frozen population"
                )
            normalized_failures[name] = values
        object.__setattr__(self, "gate_failures", normalized_failures)
        for name in (
            "population_complete",
            "baseline_candidate_paired",
            "non_compensable_safety_passed",
            "passed",
        ):
            object.__setattr__(
                self, name, _boolean(getattr(self, name), name)
            )
        calculated_safety = not any(normalized_failures.values())
        calculated_passed = (
            calculated_population_complete
            and self.baseline_candidate_paired
            and calculated_safety
        )
        if self.population_complete is not calculated_population_complete:
            raise V2BenchmarkValidationError(
                "population completeness claim does not match the exact "
                "closed paired population"
            )
        if self.non_compensable_safety_passed is not calculated_safety:
            raise V2BenchmarkValidationError(
                "non-compensable safety claim does not match gate failures"
            )
        if self.passed is not calculated_passed:
            raise V2BenchmarkValidationError(
                "report pass claim does not match mandatory gates"
            )
        if len(self.canonical_bytes()) > MAX_V2_REPORT_BYTES:
            raise V2BenchmarkValidationError(
                "v2 benchmark report exceeds its byte bound"
            )

    @property
    def report_id(self) -> str:
        return _digest(self.to_dict())

    @property
    def evidence_claim_ids(self) -> tuple[str, ...]:
        return (
            (V2_PAIRED_BASELINE_REQUIREMENT_ID,)
            if self.passed
            else ()
        )

    def to_dict(self, *, include_report_id: bool = False) -> dict[str, Any]:
        payload = {
            "schema": V2_BENCHMARK_REPORT_SCHEMA,
            "contract_version": V2_BENCHMARK_CONTRACT_VERSION,
            "corpus_version": V2_BENCHMARK_CORPUS_VERSION,
            "corpus_id": self.corpus_id,
            "fixture_population_ids": list(self.fixture_population_ids),
            "case_ids": list(self.case_ids),
            "baseline_receipt_ids": list(self.baseline_receipt_ids),
            "candidate_receipt_ids": list(self.candidate_receipt_ids),
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
            "candidate_minus_baseline": dict(self.candidate_minus_baseline),
            "gate_failures": {
                name: list(self.gate_failures[name])
                for name in V2_NON_COMPENSABLE_SAFETY_GATES
            },
            "population_complete": self.population_complete,
            "baseline_candidate_paired": self.baseline_candidate_paired,
            "non_compensable_safety_passed": (
                self.non_compensable_safety_passed
            ),
            "passed": self.passed,
            "evidence_claim_ids": list(self.evidence_claim_ids),
        }
        if include_report_id:
            payload["report_id"] = self.report_id
        return payload

    def canonical_bytes(self) -> bytes:
        return _canonical_json(self.to_dict()).encode("utf-8")

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        corpus: V2PairedBenchmarkCorpus,
    ) -> "V2BenchmarkReport":
        """Restore only by independently replaying the supplied corpus."""

        _reject_forbidden_payload(payload)
        expected = build_v2_benchmark_report(corpus)
        allowed = set(expected.to_dict(include_report_id=True))
        required = allowed - {"report_id"}
        _strict_keys(
            payload,
            required if "report_id" not in payload else allowed,
            name="v2 benchmark report",
        )
        if payload["schema"] != V2_BENCHMARK_REPORT_SCHEMA:
            raise V2BenchmarkValidationError(
                "unsupported v2 benchmark report schema"
            )
        if payload["contract_version"] != V2_BENCHMARK_CONTRACT_VERSION:
            raise V2BenchmarkValidationError(
                "unsupported v2 benchmark contract version"
            )
        if payload["corpus_version"] != V2_BENCHMARK_CORPUS_VERSION:
            raise V2BenchmarkValidationError(
                "unsupported v2 benchmark corpus version"
            )
        actual = dict(payload)
        claimed_id = actual.pop("report_id", expected.report_id)
        if actual != expected.to_dict():
            raise V2BenchmarkValidationError(
                "persisted report does not match deterministic corpus replay"
            )
        if claimed_id != expected.report_id:
            raise V2BenchmarkValidationError(
                "v2 benchmark report identity does not match replay"
            )
        return expected

    @classmethod
    def from_json(
        cls,
        value: str | bytes | bytearray,
        *,
        corpus: V2PairedBenchmarkCorpus,
    ) -> "V2BenchmarkReport":
        return cls.from_dict(
            _load_json(value, name="v2 benchmark report"),
            corpus=corpus,
        )


def build_v2_benchmark_report(
    corpus: V2PairedBenchmarkCorpus | Mapping[str, Any],
) -> V2BenchmarkReport:
    """Recompute all joins and non-compensable gates from source receipts."""

    if isinstance(corpus, Mapping):
        corpus = V2PairedBenchmarkCorpus.from_dict(corpus)
    if not isinstance(corpus, V2PairedBenchmarkCorpus):
        raise V2BenchmarkValidationError(
            "corpus must be V2PairedBenchmarkCorpus"
        )
    baseline = tuple(item.baseline for item in corpus.cases)
    candidate = tuple(item.candidate for item in corpus.cases)
    baseline_total = _aggregate(baseline)
    candidate_total = _aggregate(candidate)
    delta = {
        name: getattr(candidate_total, name) - getattr(baseline_total, name)
        for name in baseline_total.__dataclass_fields__
    }
    failures: dict[str, tuple[str, ...]] = {}
    for gate in V2_NON_COMPENSABLE_SAFETY_GATES:
        # Both arms must be safe.  A candidate cannot appear better by pairing
        # against an invalid or selectively omitted baseline.
        failures[gate] = tuple(
            item.fixture_id
            for item in corpus.cases
            if not item.baseline.safety_gates[gate]
            or not item.candidate.safety_gates[gate]
        )
    expected_population = tuple(
        _fixture_id(kind) for kind in REQUIRED_V2_FIXTURE_KINDS
    )
    population_complete = corpus.fixture_population_ids == expected_population
    paired = all(item.pair_integrity_passed for item in corpus.cases)
    safety_passed = not any(failures.values())
    return V2BenchmarkReport(
        corpus_id=corpus.corpus_id,
        fixture_population_ids=corpus.fixture_population_ids,
        case_ids=tuple(item.case_id for item in corpus.cases),
        baseline_receipt_ids=tuple(
            item.baseline.receipt_id for item in corpus.cases
        ),
        candidate_receipt_ids=tuple(
            item.candidate.receipt_id for item in corpus.cases
        ),
        baseline=baseline_total,
        candidate=candidate_total,
        candidate_minus_baseline=delta,
        gate_failures=failures,
        population_complete=population_complete,
        baseline_candidate_paired=paired,
        non_compensable_safety_passed=safety_passed,
        passed=population_complete and paired and safety_passed,
    )


def verify_v2_benchmark_report(
    report: V2BenchmarkReport | Mapping[str, Any],
    corpus: V2PairedBenchmarkCorpus,
) -> V2BenchmarkReport:
    """Fail closed unless ``report`` is the exact deterministic replay."""

    if isinstance(report, V2BenchmarkReport):
        payload = report.to_dict(include_report_id=True)
    elif isinstance(report, Mapping):
        payload = report
    else:
        raise V2BenchmarkValidationError(
            "report must be V2BenchmarkReport or an object"
        )
    return V2BenchmarkReport.from_dict(payload, corpus=corpus)


def replay_v2_benchmark(
    corpus: (
        V2PairedBenchmarkCorpus
        | Mapping[str, Any]
        | str
        | bytes
        | bytearray
    ),
    *,
    expected_report: V2BenchmarkReport | Mapping[str, Any] | None = None,
) -> V2BenchmarkReport:
    """Restore the closed corpus and deterministically replay its report.

    Supplying ``expected_report`` turns replay into verification; altered
    totals, identities, gates, or report IDs then fail closed.
    """

    if isinstance(corpus, (str, bytes, bytearray)):
        normalized = V2PairedBenchmarkCorpus.from_json(corpus)
    elif isinstance(corpus, Mapping):
        normalized = V2PairedBenchmarkCorpus.from_dict(corpus)
    elif isinstance(corpus, V2PairedBenchmarkCorpus):
        normalized = corpus
    else:
        raise V2BenchmarkValidationError(
            "corpus must be a v2 paired corpus or its serialized payload"
        )
    report = build_v2_benchmark_report(normalized)
    if expected_report is not None:
        return verify_v2_benchmark_report(expected_report, normalized)
    return report


def _v1_status(value: Any) -> V2GateStatus:
    raw = str(getattr(value, "value", value)).replace("_", "-")
    if raw == "passed":
        return V2GateStatus.PASSED
    if raw == "failed":
        return V2GateStatus.FAILED
    return V2GateStatus.NOT_REQUIRED


def adapt_v1_efficiency_receipt(
    receipt: Any,
    *,
    fixture_kind: V2FixtureKind | str,
    arm: V2BenchmarkArm | str,
    identity: V2FrozenIdentity,
    input_id: str,
    expected_terminal_outcome: str | None = None,
    terminal_outcome: str | None = None,
    terminal_criterion_ids: Sequence[str] | None = None,
    merge_status: V2GateStatus | str = V2GateStatus.NOT_REQUIRED,
    merge_latency_ms: int = 0,
    persistence_status: V2GateStatus | str = V2GateStatus.PASSED,
    persistence_latency_ms: int = 0,
    persistence_write_count: int = 0,
    persistence_bytes: int = 0,
    idle_cpu_milli_percent: int = 0,
    idle_observation_ms: int = 0,
    causal_parent_ids: Sequence[str] = (),
    intervention_ids: Sequence[str] = ("intervention:v1-adapter@1",),
) -> V2CausalReceipt:
    """Adapt an existing v1 efficiency receipt without copying its bodies.

    The adapter is intentionally structural so this module does not change the
    v1 contract.  A duck-typed object must expose the public fields of
    :class:`supervisor_efficiency_metrics.EfficiencyReceipt`.
    """

    kind = _enum(fixture_kind, V2FixtureKind, "fixture_kind")
    normalized_arm = _enum(arm, V2BenchmarkArm, "arm")
    required = (
        tuple(terminal_criterion_ids)
        if terminal_criterion_ids is not None
        else (f"criterion:v2:{kind.value}:terminal",)
    )
    try:
        stages = tuple(
            V2StageLatency(
                stage=str(getattr(item.stage, "value", item.stage)),
                latency_ms=item.latency_ms,
                invocation_count=item.invocation_count,
            )
            for item in receipt.stages
        )
        cache_observations = tuple(receipt.cache_observations)
        retries = tuple(receipt.retries)
        terminal_accepted = bool(receipt.accepted)
        actual_outcome = str(
            getattr(receipt.terminal.outcome, "value", receipt.terminal.outcome)
        )
        source_id = str(receipt.receipt_id)
    except (AttributeError, TypeError) as exc:
        raise V2BenchmarkValidationError(
            "v1 receipt does not expose the efficiency receipt contract"
        ) from exc
    expected = expected_terminal_outcome or _EXPECTED_TERMINAL_OUTCOME[kind]
    actual = terminal_outcome or (
        "accepted" if terminal_accepted else actual_outcome
    )
    metrics = V2CausalMetrics(
        stage_latencies=stages,
        elapsed_ms=receipt.elapsed_ms,
        queue_delay_ms=receipt.queue_delay_ms,
        provider_input_tokens=receipt.tokens.input_tokens,
        provider_output_tokens=receipt.tokens.output_tokens,
        provider_reused_input_tokens=receipt.tokens.reused_tokens,
        cache_lookup_count=len(cache_observations),
        cache_hit_count=sum(
            str(getattr(item.disposition, "value", item.disposition)) == "hit"
            for item in cache_observations
        ),
        cache_reused_bytes=sum(item.bytes_reused for item in cache_observations),
        retry_count=len(retries),
        retry_input_tokens=sum(item.tokens.input_tokens for item in retries),
        retry_output_tokens=sum(item.tokens.output_tokens for item in retries),
        validation_status=_v1_status(receipt.validation.status),
        validation_latency_ms=receipt.validation.duration_ms,
        proof_status=_v1_status(receipt.proof.status),
        proof_latency_ms=receipt.proof.duration_ms,
        merge_status=merge_status,
        merge_latency_ms=merge_latency_ms,
        persistence_status=persistence_status,
        persistence_latency_ms=persistence_latency_ms,
        persistence_write_count=persistence_write_count,
        persistence_bytes=persistence_bytes,
        idle_cpu_milli_percent=idle_cpu_milli_percent,
        idle_observation_ms=idle_observation_ms,
        artifact_count=len(receipt.artifacts),
        artifact_bytes=sum(item.byte_count for item in receipt.artifacts),
        artifact_manifest_digest=_digest(
            tuple(
                (item.reference_id, item.digest, item.byte_count)
                for item in receipt.artifacts
            )
        ),
        expected_terminal_outcome=expected,
        terminal_outcome=actual,
        required_criterion_ids=required,
        terminal_accepted_criterion_ids=required,
    )
    return V2CausalReceipt(
        fixture_kind=kind,
        arm=normalized_arm,
        identity=identity,
        input_id=input_id,
        metrics=metrics,
        source_receipt_ids=(source_id,),
        causal_parent_ids=tuple(causal_parent_ids),
        intervention_ids=tuple(intervention_ids),
    )


def _default_identity(
    kind: V2FixtureKind, arm: V2BenchmarkArm
) -> V2FrozenIdentity:
    return V2FrozenIdentity(
        repository_id=V2_FROZEN_REPOSITORY_ID,
        tree_id=V2_FROZEN_TREE_ID,
        objective_id=V2_FROZEN_OBJECTIVE_ID,
        objective_revision=V2_FROZEN_OBJECTIVE_REVISION,
        provider_id=V2_FROZEN_PROVIDER_ID,
        provider_revision=V2_FROZEN_PROVIDER_REVISION,
        capability_id=V2_FROZEN_CAPABILITY_ID,
        capability_revision=V2_FROZEN_CAPABILITY_REVISION,
        policy_id=V2_FROZEN_POLICY_ID,
        policy_revision=V2_FROZEN_POLICY_REVISION,
        fault_id=_fault_id(kind),
        observation_id=_observation_id(kind, arm),
    )


def _base_metrics(
    kind: V2FixtureKind, arm: V2BenchmarkArm
) -> V2CausalMetrics:
    accepted = _EXPECTED_TERMINAL_OUTCOME[kind]
    candidate = arm is V2BenchmarkArm.CANDIDATE
    no_provider = kind in {
        V2FixtureKind.UNAVAILABLE_PROVIDER,
        V2FixtureKind.DRAINED_BOARD,
        V2FixtureKind.UNTRUSTED_REPOSITORY,
    }
    input_tokens = 0 if no_provider else (2_400 if candidate else 4_000)
    output_tokens = 0 if no_provider else (360 if candidate else 600)
    warm = kind in {V2FixtureKind.WARM, V2FixtureKind.RESTART}
    reused_tokens = (
        (1_800 if candidate else 2_400)
        if warm
        else 0
    )
    retry_count = (
        1
        if kind in {
            V2FixtureKind.MALFORMED_OUTPUT,
            V2FixtureKind.FAILED_VALIDATION,
            V2FixtureKind.RESTART,
        }
        else 0
    )
    retry_input = (360 if candidate else 1_000) if retry_count else 0
    retry_output = (60 if candidate else 180) if retry_count else 0
    elapsed = 8_000 if candidate else 12_000
    if no_provider:
        elapsed = 250 if candidate else 500
    if kind is V2FixtureKind.DRAINED_BOARD:
        elapsed = MIN_DRAINED_OBSERVATION_MS
    is_terminal_accept = accepted == "accepted"
    validation = (
        V2GateStatus.PASSED
        if is_terminal_accept
        else (
            V2GateStatus.FAILED
            if kind is V2FixtureKind.FAILED_VALIDATION
            else V2GateStatus.NOT_REQUIRED
        )
    )
    proof = (
        V2GateStatus.PASSED
        if kind in {V2FixtureKind.BROAD_GOAL, V2FixtureKind.RESTART}
        else V2GateStatus.NOT_REQUIRED
    )
    merge = (
        V2GateStatus.PASSED
        if is_terminal_accept
        else V2GateStatus.NOT_REQUIRED
    )
    stages = (
        ()
        if no_provider
        else (
            V2StageLatency("analysis", 900 if candidate else 1_400),
            V2StageLatency("inference", 2_400 if candidate else 4_000),
            V2StageLatency("validation", 1_100 if candidate else 2_000),
        )
    )
    criterion = (f"criterion:v2:{kind.value}:terminal",)
    artifact_count = 1 if is_terminal_accept else 0
    artifact_bytes = 768 if candidate else 1_024
    if not artifact_count:
        artifact_bytes = 0
    return V2CausalMetrics(
        stage_latencies=stages,
        elapsed_ms=elapsed,
        queue_delay_ms=0 if no_provider else (400 if candidate else 1_000),
        provider_input_tokens=input_tokens,
        provider_output_tokens=output_tokens,
        provider_reused_input_tokens=reused_tokens,
        cache_lookup_count=1 if warm else 0,
        cache_hit_count=1 if warm else 0,
        cache_reused_bytes=(4_096 if candidate else 2_048) if warm else 0,
        retry_count=retry_count,
        retry_input_tokens=retry_input,
        retry_output_tokens=retry_output,
        validation_status=validation,
        validation_latency_ms=(
            (1_100 if candidate else 2_000)
            if validation is not V2GateStatus.NOT_REQUIRED
            else 0
        ),
        proof_status=proof,
        proof_latency_ms=(
            (500 if candidate else 900)
            if proof is not V2GateStatus.NOT_REQUIRED
            else 0
        ),
        merge_status=merge,
        merge_latency_ms=(
            (450 if candidate else 800)
            if merge is not V2GateStatus.NOT_REQUIRED
            else 0
        ),
        persistence_status=V2GateStatus.PASSED,
        persistence_latency_ms=30 if candidate else 60,
        persistence_write_count=(
            0 if kind is V2FixtureKind.DRAINED_BOARD else 1
        ),
        persistence_bytes=(
            0 if kind is V2FixtureKind.DRAINED_BOARD else 512
        ),
        idle_cpu_milli_percent=(
            (700 if candidate else 1_400)
            if kind is V2FixtureKind.DRAINED_BOARD
            else 0
        ),
        idle_observation_ms=(
            MIN_DRAINED_OBSERVATION_MS
            if kind is V2FixtureKind.DRAINED_BOARD
            else 0
        ),
        artifact_count=artifact_count,
        artifact_bytes=artifact_bytes,
        artifact_manifest_digest=_fixture_digest(
            f"v2:{kind.value}:{arm.value}:artifact-manifest"
        ),
        expected_terminal_outcome=accepted,
        terminal_outcome=accepted,
        required_criterion_ids=criterion,
        terminal_accepted_criterion_ids=criterion,
    )


def build_frozen_v2_paired_corpus() -> V2PairedBenchmarkCorpus:
    """Return the canonical deterministic generation-2 baseline corpus."""

    cases: list[V2PairedBenchmarkCase] = []
    for kind in REQUIRED_V2_FIXTURE_KINDS:
        input_id = _fixture_digest(f"supervisor-v2-input:{kind.value}@1")
        baseline = V2CausalReceipt(
            fixture_kind=kind,
            arm=V2BenchmarkArm.BASELINE,
            identity=_default_identity(kind, V2BenchmarkArm.BASELINE),
            input_id=input_id,
            metrics=_base_metrics(kind, V2BenchmarkArm.BASELINE),
            intervention_ids=("intervention:generation-1-control@1",),
        )
        candidate = V2CausalReceipt(
            fixture_kind=kind,
            arm=V2BenchmarkArm.CANDIDATE,
            identity=_default_identity(kind, V2BenchmarkArm.CANDIDATE),
            input_id=input_id,
            metrics=_base_metrics(kind, V2BenchmarkArm.CANDIDATE),
            causal_parent_ids=(baseline.receipt_id,),
            intervention_ids=("intervention:generation-2-candidate@1",),
        )
        cases.append(
            V2PairedBenchmarkCase(
                fixture_id=_fixture_id(kind),
                fixture_kind=kind,
                fixture_revision=V2_BENCHMARK_CORPUS_VERSION,
                baseline=baseline,
                candidate=candidate,
            )
        )
    corpus = V2PairedBenchmarkCorpus(cases=tuple(cases))
    if corpus.corpus_id != V2_FROZEN_CORPUS_ID:
        raise V2BenchmarkValidationError(
            "canonical v2 corpus drifted without a version change"
        )
    report = build_v2_benchmark_report(corpus)
    if report.report_id != V2_CAUSAL_BASELINE_REPORT_ID:
        raise V2BenchmarkValidationError(
            "canonical v2 causal baseline drifted without a version change"
        )
    return corpus


def replace_v2_candidate_metrics(
    corpus: V2PairedBenchmarkCorpus,
    fixture_kind: V2FixtureKind | str,
    **changes: Any,
) -> V2PairedBenchmarkCorpus:
    """Create a validated test/canary corpus with one candidate observation.

    This helper preserves every frozen identity and the full population.  It
    exists so fault injection cannot accidentally turn into denominator
    narrowing.
    """

    kind = _enum(fixture_kind, V2FixtureKind, "fixture_kind")
    cases: list[V2PairedBenchmarkCase] = []
    for case in corpus.cases:
        if case.fixture_kind is not kind:
            cases.append(case)
            continue
        metrics = replace(case.candidate.metrics, **changes)
        candidate = replace(case.candidate, metrics=metrics)
        cases.append(replace(case, candidate=candidate))
    return V2PairedBenchmarkCorpus(cases=tuple(cases))


# Discoverable compatibility names; package-level exports are intentionally
# deferred to the public-API task.
FixtureKind = V2FixtureKind
BenchmarkArm = V2BenchmarkArm
PairedBenchmarkCase = V2PairedBenchmarkCase
PairedBenchmarkCorpus = V2PairedBenchmarkCorpus
CausalReceipt = V2CausalReceipt
CausalMetrics = V2CausalMetrics
V2BenchmarkFixture = V2PairedBenchmarkCase
V2BenchmarkCorpus = V2PairedBenchmarkCorpus
SupervisorV2BenchmarkCorpus = V2PairedBenchmarkCorpus
FrozenBenchmarkIdentities = V2FrozenIdentity
REQUIRED_V2_BENCHMARK_FIXTURES = REQUIRED_V2_FIXTURE_KINDS
V2_REQUIRED_FIXTURE_KINDS = REQUIRED_V2_FIXTURE_KINDS
build_v2_paired_corpus = build_frozen_v2_paired_corpus
build_v2_benchmark_corpus = build_frozen_v2_paired_corpus
build_paired_benchmark_report = build_v2_benchmark_report


__all__ = [
    "BenchmarkArm",
    "CausalMetrics",
    "CausalReceipt",
    "FixtureKind",
    "FrozenBenchmarkIdentities",
    "MAX_DRAINED_IDLE_CPU_MILLI_PERCENT",
    "MAX_V2_ARTIFACT_BYTES",
    "MAX_V2_ARTIFACT_COUNT",
    "MAX_V2_CORPUS_BYTES",
    "MAX_V2_RECEIPT_BYTES",
    "MIN_DRAINED_OBSERVATION_MS",
    "PairedBenchmarkCase",
    "PairedBenchmarkCorpus",
    "REQUIRED_V2_BENCHMARK_FIXTURES",
    "REQUIRED_V2_FIXTURE_KINDS",
    "V2AggregateMetrics",
    "V2BenchmarkArm",
    "V2BenchmarkCorpus",
    "V2BenchmarkFixture",
    "V2BenchmarkReport",
    "V2BenchmarkValidationError",
    "V2CausalMetrics",
    "V2CausalReceipt",
    "V2FixtureKind",
    "V2FrozenIdentity",
    "V2GateStatus",
    "V2PairedBenchmarkCase",
    "V2PairedBenchmarkCorpus",
    "V2StageLatency",
    "V2_BENCHMARK_CONTRACT_VERSION",
    "V2_BENCHMARK_CORPUS_VERSION",
    "V2_BENCHMARK_REPORT_SCHEMA",
    "V2_CAUSAL_BASELINE_REPORT_ID",
    "V2_CAUSAL_RECEIPT_SCHEMA",
    "V2_FROZEN_CAPABILITY_ID",
    "V2_FROZEN_CAPABILITY_REVISION",
    "V2_FROZEN_CORPUS_ID",
    "V2_FROZEN_OBJECTIVE_ID",
    "V2_FROZEN_OBJECTIVE_REVISION",
    "V2_FROZEN_POLICY_ID",
    "V2_FROZEN_POLICY_REVISION",
    "V2_FROZEN_PROVIDER_ID",
    "V2_FROZEN_PROVIDER_REVISION",
    "V2_FROZEN_REPOSITORY_ID",
    "V2_FROZEN_TREE_ID",
    "V2_NON_COMPENSABLE_SAFETY_GATES",
    "V2_PAIRED_BASELINE_GOAL_ID",
    "V2_PAIRED_BASELINE_REQUIREMENT_ID",
    "V2_PAIRED_CASE_SCHEMA",
    "V2_PAIRED_CORPUS_SCHEMA",
    "V2_REQUIRED_FIXTURE_KINDS",
    "SupervisorV2BenchmarkCorpus",
    "adapt_v1_efficiency_receipt",
    "build_frozen_v2_paired_corpus",
    "build_paired_benchmark_report",
    "build_v2_benchmark_report",
    "build_v2_benchmark_corpus",
    "build_v2_paired_corpus",
    "replace_v2_candidate_metrics",
    "replay_v2_benchmark",
    "verify_v2_benchmark_report",
]
