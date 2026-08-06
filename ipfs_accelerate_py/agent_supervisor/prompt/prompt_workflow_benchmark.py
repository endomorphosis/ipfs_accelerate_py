"""Frozen paired/adversarial/chaos benchmark for prompt bootstrap and rescue.

This module is deliberately provider-free.  It freezes prompt/repository
fixtures as producer receipts, recomputes every parity/safety metric, and
never grants mutation, completion, or process authority.  Production promotion
must replace smoke identities with live receipts; this closed population is a
local conformance fixture only.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Final


PROMPT_WORKFLOW_BENCHMARK_VERSION: Final = 1
PROMPT_WORKFLOW_PRODUCER_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/prompt-workflow-producer-receipt@1"
)
PROMPT_WORKFLOW_BENCHMARK_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/prompt-workflow-benchmark@1"
)
PROMPT_WORKFLOW_GATE_REPORT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/prompt-workflow-gate-report@1"
)
PROMPT_WORKFLOW_BENCHMARK_REQUIREMENT_ID: Final = (
    "asi-159:prompt-workflow-paired-adversarial-chaos-gate"
)
MAX_RECEIPTS: Final = 100_000
MAX_COUNTER: Final = 10**15
MAX_REPORT_BYTES: Final = 8 * 1024 * 1024

_CONTENT_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_CODE = re.compile(r"^[a-z][a-z0-9_.:/@-]{0,191}$")


class PromptWorkflowBenchmarkError(ValueError):
    """Benchmark source evidence is malformed, incomplete, or inconsistent."""


class PlanningMode(str, Enum):
    DETERMINISTIC = "deterministic"
    MODEL = "model"


class TaskSourceBackend(str, Enum):
    MARKDOWN = "markdown"
    DUCKDB = "duckdb"
    BOTH = "both"


class TransportSurface(str, Enum):
    PYTHON = "python"
    CLI = "cli"
    SCRIPT = "script"
    MCP = "mcp"


class ChaosBoundary(str, Enum):
    """Every materialization/lifecycle/rescue intent-effect-receipt boundary."""

    MATERIALIZE_BEFORE_INTENT = "materialize-before-intent"
    MATERIALIZE_AFTER_INTENT = "materialize-after-intent"
    MATERIALIZE_BEFORE_EFFECT = "materialize-before-effect"
    MATERIALIZE_AFTER_EFFECT = "materialize-after-effect"
    MATERIALIZE_BEFORE_RECEIPT = "materialize-before-receipt"
    MATERIALIZE_AFTER_RECEIPT = "materialize-after-receipt"
    LIFECYCLE_BEFORE_INTENT = "lifecycle-before-intent"
    LIFECYCLE_AFTER_INTENT = "lifecycle-after-intent"
    LIFECYCLE_BEFORE_EFFECT = "lifecycle-before-effect"
    LIFECYCLE_AFTER_EFFECT = "lifecycle-after-effect"
    LIFECYCLE_BEFORE_RECEIPT = "lifecycle-before-receipt"
    LIFECYCLE_AFTER_RECEIPT = "lifecycle-after-receipt"
    RESCUE_BEFORE_INTENT = "rescue-before-intent"
    RESCUE_AFTER_INTENT = "rescue-after-intent"
    RESCUE_BEFORE_EFFECT = "rescue-before-effect"
    RESCUE_AFTER_EFFECT = "rescue-after-effect"
    RESCUE_BEFORE_RECEIPT = "rescue-before-receipt"
    RESCUE_AFTER_RECEIPT = "rescue-after-receipt"


class AdversarialFixture(str, Enum):
    PROMPT_INJECTION = "prompt-injection"
    REPOSITORY_INJECTION = "repository-injection"
    PATH_ESCAPE = "path-escape"
    SYMLINK_ESCAPE = "symlink-escape"
    SECRET_LEAK = "secret-leak"
    FORGED_CID = "forged-cid"
    SCHEMA_DOWNGRADE = "schema-downgrade"
    SQL_INJECTION = "sql-injection"
    PID_REUSE = "pid-reuse"
    PROCESS_ESCAPE = "process-escape"
    POLICY_WEAKENING = "policy-weakening"
    AUTHORIZATION_BYPASS = "authorization-bypass"
    PERMIT_FORGERY = "permit-forgery"
    COMPLETION_FORGERY = "completion-forgery"
    MANDATORY_EVIDENCE_OMISSION = "mandatory-evidence-omission"
    STALE_PREVIEW = "stale-preview"
    CROSS_REPOSITORY_REPLAY = "cross-repository-replay"
    SHELL_RESCUE_PROPOSAL = "shell-rescue-proposal"


class OptionalDependency(str, Enum):
    DUCKDB = "duckdb"
    LLM_ROUTER = "llm-router"
    MCP = "mcp"
    IPFS_DATASETS = "ipfs-datasets"
    MODEL_CREDENTIALS = "model-credentials"


class FaultOutcome(str, Enum):
    RESUME = "resume"
    COMPENSATE = "compensate"
    QUARANTINE = "quarantine"


class TerminalOutcome(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DENIED = "denied"
    DEGRADED = "degraded"
    FAIL_CLOSED = "fail-closed"
    QUARANTINED = "quarantined"
    HEALTHY = "healthy"


REQUIRED_TASK_SOURCES: Final[tuple[TaskSourceBackend, ...]] = tuple(
    TaskSourceBackend
)
REQUIRED_PLANNING_MODES: Final[tuple[PlanningMode, ...]] = tuple(PlanningMode)
REQUIRED_TRANSPORTS: Final[tuple[TransportSurface, ...]] = tuple(
    TransportSurface
)
REQUIRED_CHAOS_BOUNDARIES: Final[tuple[ChaosBoundary, ...]] = tuple(
    ChaosBoundary
)
REQUIRED_ADVERSARIAL_FIXTURES: Final[tuple[AdversarialFixture, ...]] = tuple(
    AdversarialFixture
)
REQUIRED_OPTIONAL_DEPENDENCIES: Final[tuple[OptionalDependency, ...]] = tuple(
    OptionalDependency
)

SAFE_ADVERSARIAL_TERMINALS: Final[frozenset[str]] = frozenset(
    {
        TerminalOutcome.REJECTED.value,
        TerminalOutcome.DENIED.value,
        TerminalOutcome.DEGRADED.value,
        TerminalOutcome.FAIL_CLOSED.value,
        TerminalOutcome.QUARANTINED.value,
    }
)
SAFE_CHAOS_TERMINALS: Final[frozenset[str]] = frozenset(
    {
        TerminalOutcome.HEALTHY.value,
        TerminalOutcome.QUARANTINED.value,
        TerminalOutcome.DEGRADED.value,
        FaultOutcome.RESUME.value,
        FaultOutcome.COMPENSATE.value,
        FaultOutcome.QUARANTINE.value,
    }
)


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
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
        raise PromptWorkflowBenchmarkError(
            "benchmark evidence must be canonical JSON"
        ) from exc


def _identity(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _load_json(value: str | bytes | bytearray, name: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise PromptWorkflowBenchmarkError(
                    f"{name} contains duplicate JSON key {key!r}"
                )
            result[key] = item
        return result

    try:
        if isinstance(value, (bytes, bytearray)):
            value = bytes(value).decode("utf-8")
        if not isinstance(value, str):
            raise PromptWorkflowBenchmarkError(f"{name} must be JSON text")
        return json.loads(value, object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PromptWorkflowBenchmarkError(f"{name} is invalid JSON") from exc


def _text(value: Any, name: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PromptWorkflowBenchmarkError(
            f"{name} must be non-empty canonical text"
        )
    if "\x00" in value or len(value.encode("utf-8")) > maximum:
        raise PromptWorkflowBenchmarkError(f"{name} is unsafe or too large")
    return value


def _code(value: Any, name: str) -> str:
    result = _text(str(getattr(value, "value", value)), name, maximum=192)
    if not _CODE.fullmatch(result):
        raise PromptWorkflowBenchmarkError(f"{name} must be a compact code")
    return result


def _content_id(value: Any, name: str) -> str:
    result = _text(value, name, maximum=71)
    if not _CONTENT_ID.fullmatch(result):
        raise PromptWorkflowBenchmarkError(
            f"{name} must be a lowercase sha256 content ID"
        )
    return result


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > MAX_COUNTER
    ):
        raise PromptWorkflowBenchmarkError(
            f"{name} must be an integer from {minimum} through {MAX_COUNTER}"
        )
    return value


def _enum(value: Any, kind: type[Enum], name: str) -> Any:
    if isinstance(value, kind):
        return value
    try:
        return kind(str(getattr(value, "value", value)))
    except (TypeError, ValueError) as exc:
        raise PromptWorkflowBenchmarkError(f"invalid {name}") from exc


def _ids(values: Sequence[Any], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise PromptWorkflowBenchmarkError(f"{name} must be a sequence")
    result = tuple(sorted(_text(str(v), name, maximum=512) for v in values))
    if len(result) != len(set(result)):
        raise PromptWorkflowBenchmarkError(f"{name} must be unique")
    return result


@dataclass(frozen=True)
class FrozenPromptFixtureIdentity:
    """Exact identities shared by every paired path over one frozen fixture."""

    repository_id: str
    tree_id: str
    prompt_fixture_id: str
    prompt_cid: str
    scan_cid: str
    plan_root_cid: str
    objective_id: str
    objective_revision: str
    policy_id: str
    policy_revision: str
    capability_id: str
    capability_revision: str
    partition_id: str

    def __post_init__(self) -> None:
        for name in (
            "repository_id",
            "tree_id",
            "prompt_fixture_id",
            "objective_id",
            "objective_revision",
            "policy_id",
            "policy_revision",
            "capability_id",
            "capability_revision",
            "partition_id",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, maximum=512)
            )
        for name in ("prompt_cid", "scan_cid", "plan_root_cid"):
            object.__setattr__(
                self, name, _content_id(getattr(self, name), name)
            )

    @property
    def identity_id(self) -> str:
        return _identity(self.to_dict())

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenPromptFixtureIdentity":
        if set(value) != set(cls.__dataclass_fields__):
            raise PromptWorkflowBenchmarkError("invalid frozen identity fields")
        return cls(**dict(value))


@dataclass(frozen=True)
class PromptWorkflowMetrics:
    """Bounded counters derived only from producer receipts."""

    admitted_task_cids: tuple[str, ...]
    ready_task_cids: tuple[str, ...]
    accepted_effect_ids: tuple[str, ...]
    terminal_result: str
    model_calls: int
    provider_input_tokens: int
    provider_output_tokens: int
    retries: int
    storage_bytes: int
    process_count: int
    materialization_latency_ms: int
    recovery_latency_ms: int
    secret_bytes_emitted: int = 0
    escape_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "admitted_task_cids",
            _ids(self.admitted_task_cids, "admitted_task_cids"),
        )
        object.__setattr__(
            self, "ready_task_cids", _ids(self.ready_task_cids, "ready_task_cids")
        )
        object.__setattr__(
            self,
            "accepted_effect_ids",
            _ids(self.accepted_effect_ids, "accepted_effect_ids"),
        )
        object.__setattr__(
            self, "terminal_result", _code(self.terminal_result, "terminal_result")
        )
        for name in (
            "model_calls",
            "provider_input_tokens",
            "provider_output_tokens",
            "retries",
            "storage_bytes",
            "process_count",
            "materialization_latency_ms",
            "recovery_latency_ms",
            "secret_bytes_emitted",
            "escape_count",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if not set(self.ready_task_cids).issubset(self.admitted_task_cids):
            raise PromptWorkflowBenchmarkError(
                "ready tasks must be a subset of admitted tasks"
            )

    @property
    def total_tokens(self) -> int:
        return self.provider_input_tokens + self.provider_output_tokens

    def to_dict(self) -> dict[str, Any]:
        payload = {
            name: _plain(getattr(self, name))
            for name in self.__dataclass_fields__
        }
        payload["total_tokens"] = self.total_tokens
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PromptWorkflowMetrics":
        fields = set(cls.__dataclass_fields__)
        if not fields.issubset(value) or set(value).difference(
            fields | {"total_tokens"}
        ):
            raise PromptWorkflowBenchmarkError("invalid metrics fields")
        return cls(**{name: value[name] for name in fields})


@dataclass(frozen=True)
class PromptWorkflowProducerReceipt:
    """One immutable observation over a frozen fixture path."""

    identity: FrozenPromptFixtureIdentity
    planning_mode: PlanningMode | str
    task_source: TaskSourceBackend | str
    transport: TransportSurface | str
    metrics: PromptWorkflowMetrics
    source_receipt_ids: tuple[str, ...]
    adversarial_fixture: AdversarialFixture | str | None = None
    chaos_boundary: ChaosBoundary | str | None = None
    fault_outcome: FaultOutcome | str | None = None
    optional_dependency: OptionalDependency | str | None = None
    degraded_local: bool = False
    deterministic_replay_id: str = ""
    lazy_discovery: bool = True
    projection_cid: str = ""
    run_cid: str = ""
    rescue_plan_cid: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.identity, FrozenPromptFixtureIdentity):
            raise PromptWorkflowBenchmarkError("identity is not frozen")
        object.__setattr__(
            self,
            "planning_mode",
            _enum(self.planning_mode, PlanningMode, "planning_mode"),
        )
        object.__setattr__(
            self,
            "task_source",
            _enum(self.task_source, TaskSourceBackend, "task_source"),
        )
        object.__setattr__(
            self, "transport", _enum(self.transport, TransportSurface, "transport")
        )
        if not isinstance(self.metrics, PromptWorkflowMetrics):
            raise PromptWorkflowBenchmarkError(
                "metrics must be PromptWorkflowMetrics"
            )
        object.__setattr__(
            self,
            "source_receipt_ids",
            _ids(self.source_receipt_ids, "source_receipt_ids"),
        )
        if not self.source_receipt_ids:
            raise PromptWorkflowBenchmarkError(
                "producer observation requires source receipts"
            )
        if self.adversarial_fixture is not None:
            object.__setattr__(
                self,
                "adversarial_fixture",
                _enum(
                    self.adversarial_fixture,
                    AdversarialFixture,
                    "adversarial_fixture",
                ),
            )
        if self.chaos_boundary is not None:
            object.__setattr__(
                self,
                "chaos_boundary",
                _enum(self.chaos_boundary, ChaosBoundary, "chaos_boundary"),
            )
        if self.fault_outcome is not None:
            object.__setattr__(
                self,
                "fault_outcome",
                _enum(self.fault_outcome, FaultOutcome, "fault_outcome"),
            )
        if self.optional_dependency is not None:
            object.__setattr__(
                self,
                "optional_dependency",
                _enum(
                    self.optional_dependency,
                    OptionalDependency,
                    "optional_dependency",
                ),
            )
        if not isinstance(self.degraded_local, bool) or not isinstance(
            self.lazy_discovery, bool
        ):
            raise PromptWorkflowBenchmarkError(
                "degraded_local and lazy_discovery must be boolean"
            )
        if self.degraded_local:
            object.__setattr__(
                self,
                "deterministic_replay_id",
                _content_id(
                    self.deterministic_replay_id, "deterministic_replay_id"
                ),
            )
        elif self.deterministic_replay_id:
            object.__setattr__(
                self,
                "deterministic_replay_id",
                _content_id(
                    self.deterministic_replay_id, "deterministic_replay_id"
                ),
            )
        for name in ("projection_cid", "run_cid", "rescue_plan_cid"):
            value = getattr(self, name)
            if value:
                object.__setattr__(self, name, _content_id(value, name))
        # Exactly one intervention class per receipt keeps the population closed.
        interventions = sum(
            1
            for item in (
                self.adversarial_fixture,
                self.chaos_boundary,
                self.optional_dependency,
            )
            if item is not None
        )
        if interventions > 1:
            raise PromptWorkflowBenchmarkError(
                "producer receipt may carry at most one intervention class"
            )
        if self.chaos_boundary is not None and self.fault_outcome is None:
            raise PromptWorkflowBenchmarkError(
                "chaos boundary requires a fault outcome"
            )

    @property
    def receipt_id(self) -> str:
        return _identity(self.to_dict(include_receipt_id=False))

    @property
    def is_paired_path(self) -> bool:
        return (
            self.adversarial_fixture is None
            and self.chaos_boundary is None
            and self.optional_dependency is None
        )

    def to_dict(self, *, include_receipt_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": PROMPT_WORKFLOW_PRODUCER_RECEIPT_SCHEMA,
            "version": PROMPT_WORKFLOW_BENCHMARK_VERSION,
            "identity": self.identity.to_dict(),
            "planning_mode": self.planning_mode.value,
            "task_source": self.task_source.value,
            "transport": self.transport.value,
            "metrics": self.metrics.to_dict(),
            "source_receipt_ids": list(self.source_receipt_ids),
            "adversarial_fixture": (
                self.adversarial_fixture.value
                if self.adversarial_fixture is not None
                else None
            ),
            "chaos_boundary": (
                self.chaos_boundary.value
                if self.chaos_boundary is not None
                else None
            ),
            "fault_outcome": (
                self.fault_outcome.value if self.fault_outcome is not None else None
            ),
            "optional_dependency": (
                self.optional_dependency.value
                if self.optional_dependency is not None
                else None
            ),
            "degraded_local": self.degraded_local,
            "deterministic_replay_id": self.deterministic_replay_id,
            "lazy_discovery": self.lazy_discovery,
            "projection_cid": self.projection_cid,
            "run_cid": self.run_cid,
            "rescue_plan_cid": self.rescue_plan_cid,
        }
        if include_receipt_id:
            payload["receipt_id"] = self.receipt_id
        return payload

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "PromptWorkflowProducerReceipt":
        allowed = {
            "schema",
            "version",
            "receipt_id",
            "identity",
            "planning_mode",
            "task_source",
            "transport",
            "metrics",
            "source_receipt_ids",
            "adversarial_fixture",
            "chaos_boundary",
            "fault_outcome",
            "optional_dependency",
            "degraded_local",
            "deterministic_replay_id",
            "lazy_discovery",
            "projection_cid",
            "run_cid",
            "rescue_plan_cid",
        }
        if set(value).difference(allowed):
            raise PromptWorkflowBenchmarkError("unknown producer receipt fields")
        if (
            value.get("schema") != PROMPT_WORKFLOW_PRODUCER_RECEIPT_SCHEMA
            or value.get("version") != PROMPT_WORKFLOW_BENCHMARK_VERSION
        ):
            raise PromptWorkflowBenchmarkError(
                "unsupported producer receipt schema"
            )
        result = cls(
            identity=FrozenPromptFixtureIdentity.from_dict(value["identity"]),
            planning_mode=value["planning_mode"],
            task_source=value["task_source"],
            transport=value["transport"],
            metrics=PromptWorkflowMetrics.from_dict(value["metrics"]),
            source_receipt_ids=tuple(value["source_receipt_ids"]),
            adversarial_fixture=value.get("adversarial_fixture"),
            chaos_boundary=value.get("chaos_boundary"),
            fault_outcome=value.get("fault_outcome"),
            optional_dependency=value.get("optional_dependency"),
            degraded_local=value.get("degraded_local", False),
            deterministic_replay_id=value.get("deterministic_replay_id", ""),
            lazy_discovery=value.get("lazy_discovery", True),
            projection_cid=value.get("projection_cid", ""),
            run_cid=value.get("run_cid", ""),
            rescue_plan_cid=value.get("rescue_plan_cid", ""),
        )
        if value.get("receipt_id", result.receipt_id) != result.receipt_id:
            raise PromptWorkflowBenchmarkError("producer receipt ID mismatch")
        return result

    def to_json(self) -> str:
        return _canonical_bytes(self.to_dict()).decode("utf-8")

    @classmethod
    def from_json(
        cls, value: str | bytes | bytearray
    ) -> "PromptWorkflowProducerReceipt":
        return cls.from_dict(_load_json(value, "producer receipt"))


@dataclass(frozen=True)
class PromptWorkflowBenchmark:
    """The complete closed source population; no selected-case evaluation."""

    receipts: tuple[PromptWorkflowProducerReceipt, ...]
    requirement_id: str = PROMPT_WORKFLOW_BENCHMARK_REQUIREMENT_ID

    def __post_init__(self) -> None:
        receipts = tuple(self.receipts)
        if not receipts or len(receipts) > MAX_RECEIPTS:
            raise PromptWorkflowBenchmarkError(
                "benchmark receipt population is empty or unbounded"
            )
        if any(
            not isinstance(item, PromptWorkflowProducerReceipt)
            for item in receipts
        ):
            raise PromptWorkflowBenchmarkError(
                "benchmark contains non-producer receipts"
            )
        receipt_ids = [item.receipt_id for item in receipts]
        if len(receipt_ids) != len(set(receipt_ids)):
            raise PromptWorkflowBenchmarkError("duplicate producer receipt")
        object.__setattr__(
            self, "receipts", tuple(sorted(receipts, key=lambda r: r.receipt_id))
        )
        if self.requirement_id != PROMPT_WORKFLOW_BENCHMARK_REQUIREMENT_ID:
            raise PromptWorkflowBenchmarkError("wrong benchmark requirement")

    @property
    def benchmark_id(self) -> str:
        return _identity(self.to_dict(include_benchmark_id=False))

    def to_dict(self, *, include_benchmark_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": PROMPT_WORKFLOW_BENCHMARK_SCHEMA,
            "version": PROMPT_WORKFLOW_BENCHMARK_VERSION,
            "requirement_id": self.requirement_id,
            "receipts": [item.to_dict() for item in self.receipts],
        }
        if include_benchmark_id:
            payload["benchmark_id"] = self.benchmark_id
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PromptWorkflowBenchmark":
        allowed = {
            "schema",
            "version",
            "requirement_id",
            "receipts",
            "benchmark_id",
        }
        if set(value).difference(allowed):
            raise PromptWorkflowBenchmarkError("unknown benchmark fields")
        if (
            value.get("schema") != PROMPT_WORKFLOW_BENCHMARK_SCHEMA
            or value.get("version") != PROMPT_WORKFLOW_BENCHMARK_VERSION
        ):
            raise PromptWorkflowBenchmarkError("unsupported benchmark schema")
        result = cls(
            receipts=tuple(
                PromptWorkflowProducerReceipt.from_dict(item)
                for item in value["receipts"]
            ),
            requirement_id=value["requirement_id"],
        )
        if value.get("benchmark_id", result.benchmark_id) != result.benchmark_id:
            raise PromptWorkflowBenchmarkError("benchmark ID mismatch")
        return result

    def to_json(self) -> str:
        return _canonical_bytes(self.to_dict()).decode("utf-8")

    @classmethod
    def from_json(
        cls, value: str | bytes | bytearray
    ) -> "PromptWorkflowBenchmark":
        return cls.from_dict(_load_json(value, "prompt workflow benchmark"))


@dataclass(frozen=True)
class PromptWorkflowGateReport:
    """Recomputed gate result.  This report has no execution authority."""

    benchmark_id: str
    fixture_count: int
    receipt_count: int
    paired_path_count: int
    admitted_task_cid_count: int
    ready_task_cid_count: int
    accepted_effect_count: int
    model_calls: int
    total_tokens: int
    retries: int
    storage_bytes: int
    process_count: int
    task_sources_passed: tuple[str, ...]
    planning_modes_passed: tuple[str, ...]
    transports_passed: tuple[str, ...]
    adversarial_fixtures_passed: tuple[str, ...]
    chaos_boundaries_passed: tuple[str, ...]
    optional_dependencies_passed: tuple[str, ...]
    task_cid_parity_passed: bool
    ready_set_parity_passed: bool
    effect_parity_passed: bool
    terminal_parity_passed: bool
    transport_parity_passed: bool
    adversarial_passed: bool
    chaos_passed: bool
    bounds_passed: bool
    secret_hygiene_passed: bool
    deterministic_degraded_passed: bool
    lazy_discovery_passed: bool
    passed: bool
    failure_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _content_id(self.benchmark_id, "benchmark_id")
        object.__setattr__(
            self, "failure_codes", tuple(sorted(set(self.failure_codes)))
        )
        encoded = _canonical_bytes(self.to_dict(include_report_id=False))
        if len(encoded) > MAX_REPORT_BYTES:
            raise PromptWorkflowBenchmarkError("gate report exceeds byte bound")

    @property
    def report_id(self) -> str:
        return _identity(self.to_dict(include_report_id=False))

    @property
    def authoritative(self) -> bool:
        return False

    @property
    def completion_authoritative(self) -> bool:
        return False

    def to_dict(self, *, include_report_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": PROMPT_WORKFLOW_GATE_REPORT_SCHEMA,
            "version": PROMPT_WORKFLOW_BENCHMARK_VERSION,
            **{
                name: _plain(getattr(self, name))
                for name in self.__dataclass_fields__
            },
            "authoritative": False,
            "completion_authoritative": False,
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
        benchmark: PromptWorkflowBenchmark,
    ) -> "PromptWorkflowGateReport":
        replayed = recompute_prompt_workflow_gate(benchmark)
        if _canonical_bytes(value) != _canonical_bytes(replayed.to_dict()):
            raise PromptWorkflowBenchmarkError(
                "gate report does not match producer receipt replay"
            )
        return replayed

    @classmethod
    def from_json(
        cls,
        value: str | bytes | bytearray,
        *,
        benchmark: PromptWorkflowBenchmark,
    ) -> "PromptWorkflowGateReport":
        return cls.from_dict(
            _load_json(value, "prompt workflow gate report"),
            benchmark=benchmark,
        )


def recompute_prompt_workflow_gate(
    benchmark: PromptWorkflowBenchmark,
) -> PromptWorkflowGateReport:
    """Replay the complete paired/adversarial/chaos population from receipts."""

    if not isinstance(benchmark, PromptWorkflowBenchmark):
        raise PromptWorkflowBenchmarkError(
            "benchmark must be PromptWorkflowBenchmark"
        )

    paired = [r for r in benchmark.receipts if r.is_paired_path]
    adversarial = [
        r for r in benchmark.receipts if r.adversarial_fixture is not None
    ]
    chaos = [r for r in benchmark.receipts if r.chaos_boundary is not None]
    degraded = [
        r
        for r in benchmark.receipts
        if r.optional_dependency is not None or r.degraded_local
    ]
    failures: list[str] = []

    # ---- Paired backend / planning / transport parity ----
    fixtures: dict[str, list[PromptWorkflowProducerReceipt]] = {}
    for receipt in paired:
        fixtures.setdefault(receipt.identity.identity_id, []).append(receipt)

    task_sources_passed: set[str] = set()
    planning_modes_passed: set[str] = set()
    transports_passed: set[str] = set()
    task_cid_parity = True
    ready_parity = True
    effect_parity = True
    terminal_parity = True
    transport_parity = True

    for identity_id, population in fixtures.items():
        by_key = {
            (r.planning_mode, r.task_source, r.transport): r for r in population
        }
        if len(by_key) != len(population):
            failures.append(f"duplicate-paired-observation:{identity_id}")
        expected_keys = {
            (mode, source, transport)
            for mode in REQUIRED_PLANNING_MODES
            for source in REQUIRED_TASK_SOURCES
            for transport in REQUIRED_TRANSPORTS
        }
        missing = expected_keys.difference(by_key)
        if missing:
            failures.append(f"missing-paired-path:{identity_id}")
            continue
        reference = by_key[
            (
                PlanningMode.DETERMINISTIC,
                TaskSourceBackend.MARKDOWN,
                TransportSurface.PYTHON,
            )
        ]
        for receipt in population:
            task_sources_passed.add(receipt.task_source.value)
            planning_modes_passed.add(receipt.planning_mode.value)
            transports_passed.add(receipt.transport.value)
            if (
                receipt.metrics.admitted_task_cids
                != reference.metrics.admitted_task_cids
            ):
                task_cid_parity = False
            if (
                receipt.metrics.ready_task_cids
                != reference.metrics.ready_task_cids
            ):
                ready_parity = False
            if (
                receipt.metrics.accepted_effect_ids
                != reference.metrics.accepted_effect_ids
            ):
                effect_parity = False
            if (
                receipt.metrics.terminal_result
                != reference.metrics.terminal_result
            ):
                terminal_parity = False
            if receipt.metrics.escape_count or receipt.metrics.secret_bytes_emitted:
                transport_parity = False
                failures.append(
                    f"paired-escape:{identity_id}:{receipt.transport.value}"
                )
            # Model path may spend model calls; deterministic must not.
            if (
                receipt.planning_mode is PlanningMode.DETERMINISTIC
                and receipt.metrics.model_calls != 0
            ):
                failures.append(
                    f"deterministic-model-call:{identity_id}:{receipt.transport.value}"
                )
            if (
                receipt.planning_mode is PlanningMode.MODEL
                and receipt.metrics.model_calls < 1
            ):
                failures.append(
                    f"model-path-without-call:{identity_id}:{receipt.transport.value}"
                )

    if not task_cid_parity:
        failures.append("task-cid-parity")
    if not ready_parity:
        failures.append("ready-set-parity")
    if not effect_parity:
        failures.append("effect-parity")
    if not terminal_parity:
        failures.append("terminal-parity")
    if not transport_parity and "transport-parity" not in failures:
        # already recorded paired escapes when present
        pass
    if set(task_sources_passed) != {item.value for item in REQUIRED_TASK_SOURCES}:
        failures.append("incomplete-task-source-coverage")
    if set(planning_modes_passed) != {
        item.value for item in REQUIRED_PLANNING_MODES
    }:
        failures.append("incomplete-planning-mode-coverage")
    if set(transports_passed) != {item.value for item in REQUIRED_TRANSPORTS}:
        failures.append("incomplete-transport-coverage")

    # ---- Adversarial: zero escapes, fail-closed terminals ----
    fixture_counts = {fixture: 0 for fixture in REQUIRED_ADVERSARIAL_FIXTURES}
    fixture_passed: set[str] = set()
    for receipt in adversarial:
        fixture = receipt.adversarial_fixture
        assert fixture is not None
        fixture_counts[fixture] += 1
        terminal = receipt.metrics.terminal_result
        if (
            receipt.metrics.escape_count
            or receipt.metrics.secret_bytes_emitted
            or terminal not in SAFE_ADVERSARIAL_TERMINALS
        ):
            failures.append(f"adversarial-escape:{fixture.value}")
        else:
            fixture_passed.add(fixture.value)
    for fixture, count in fixture_counts.items():
        if not count:
            failures.append(f"missing-adversarial-fixture:{fixture.value}")
    adversarial_passed = set(fixture_passed) == {
        item.value for item in REQUIRED_ADVERSARIAL_FIXTURES
    } and not any(code.startswith("adversarial-escape:") for code in failures)

    # ---- Chaos: every boundary has deterministic resume/compensate/quarantine ----
    boundary_counts = {boundary: 0 for boundary in REQUIRED_CHAOS_BOUNDARIES}
    boundary_passed: set[str] = set()
    for receipt in chaos:
        boundary = receipt.chaos_boundary
        assert boundary is not None
        boundary_counts[boundary] += 1
        outcome = (
            receipt.fault_outcome.value
            if receipt.fault_outcome is not None
            else ""
        )
        terminal = receipt.metrics.terminal_result
        if (
            receipt.metrics.escape_count
            or outcome not in {item.value for item in FaultOutcome}
            or (
                terminal not in SAFE_CHAOS_TERMINALS
                and outcome not in SAFE_CHAOS_TERMINALS
            )
        ):
            failures.append(f"chaos-escape:{boundary.value}")
        else:
            boundary_passed.add(boundary.value)
    for boundary, count in boundary_counts.items():
        if not count:
            failures.append(f"missing-chaos-boundary:{boundary.value}")
    chaos_passed = set(boundary_passed) == {
        item.value for item in REQUIRED_CHAOS_BOUNDARIES
    } and not any(code.startswith("chaos-escape:") for code in failures)

    # ---- Optional dependency degradation ----
    dep_counts = {dep: 0 for dep in REQUIRED_OPTIONAL_DEPENDENCIES}
    dep_passed: set[str] = set()
    for receipt in degraded:
        dep = receipt.optional_dependency
        if dep is not None:
            dep_counts[dep] += 1
        if (
            receipt.metrics.escape_count
            or not receipt.degraded_local
            or not receipt.deterministic_replay_id
            or receipt.metrics.terminal_result
            not in {
                TerminalOutcome.DEGRADED.value,
                TerminalOutcome.FAIL_CLOSED.value,
                TerminalOutcome.ACCEPTED.value,
                TerminalOutcome.HEALTHY.value,
            }
        ):
            failures.append(
                "optional-dependency-escape:"
                + (dep.value if dep is not None else "unknown")
            )
        elif dep is not None:
            dep_passed.add(dep.value)
    for dep, count in dep_counts.items():
        if not count:
            failures.append(f"missing-optional-dependency:{dep.value}")
    deterministic_degraded_passed = set(dep_passed) == {
        item.value for item in REQUIRED_OPTIONAL_DEPENDENCIES
    } and not any(
        code.startswith("optional-dependency-escape:") for code in failures
    )
    if not deterministic_degraded_passed and not any(
        code.startswith("optional-dependency")
        or code.startswith("deterministic-local")
        for code in failures
    ):
        failures.append("deterministic-local-degraded-operation")

    # ---- Bounds and hygiene ----
    metrics = [r.metrics for r in benchmark.receipts]
    model_calls = sum(m.model_calls for m in metrics)
    total_tokens = sum(m.total_tokens for m in metrics)
    retries = sum(m.retries for m in metrics)
    storage_bytes = sum(m.storage_bytes for m in metrics)
    process_count = sum(m.process_count for m in metrics)
    secret_bytes = sum(m.secret_bytes_emitted for m in metrics)
    bounds_passed = (
        model_calls <= 10_000
        and total_tokens <= 50_000_000
        and retries <= 10_000
        and storage_bytes <= 512 * 1024 * 1024
        and process_count <= 1_000
    )
    if not bounds_passed:
        failures.append("resource-bounds-exceeded")
    secret_hygiene_passed = secret_bytes == 0
    if not secret_hygiene_passed:
        failures.append("secret-hygiene")
    lazy_passed = all(r.lazy_discovery for r in benchmark.receipts)
    if not lazy_passed:
        failures.append("eager-optional-discovery")

    # Aggregate admitted/ready/effect counts from the reference paired path only.
    reference_receipts = [
        r
        for r in paired
        if r.planning_mode is PlanningMode.DETERMINISTIC
        and r.task_source is TaskSourceBackend.MARKDOWN
        and r.transport is TransportSurface.PYTHON
    ]
    admitted = set()
    ready = set()
    effects = set()
    for receipt in reference_receipts:
        admitted.update(receipt.metrics.admitted_task_cids)
        ready.update(receipt.metrics.ready_task_cids)
        effects.update(receipt.metrics.accepted_effect_ids)

    passed = not failures
    return PromptWorkflowGateReport(
        benchmark_id=benchmark.benchmark_id,
        fixture_count=len(fixtures),
        receipt_count=len(benchmark.receipts),
        paired_path_count=len(paired),
        admitted_task_cid_count=len(admitted),
        ready_task_cid_count=len(ready),
        accepted_effect_count=len(effects),
        model_calls=model_calls,
        total_tokens=total_tokens,
        retries=retries,
        storage_bytes=storage_bytes,
        process_count=process_count,
        task_sources_passed=tuple(sorted(task_sources_passed)),
        planning_modes_passed=tuple(sorted(planning_modes_passed)),
        transports_passed=tuple(sorted(transports_passed)),
        adversarial_fixtures_passed=tuple(sorted(fixture_passed)),
        chaos_boundaries_passed=tuple(sorted(boundary_passed)),
        optional_dependencies_passed=tuple(sorted(dep_passed)),
        task_cid_parity_passed=task_cid_parity,
        ready_set_parity_passed=ready_parity,
        effect_parity_passed=effect_parity,
        terminal_parity_passed=terminal_parity,
        transport_parity_passed=transport_parity
        and set(transports_passed)
        == {item.value for item in REQUIRED_TRANSPORTS},
        adversarial_passed=adversarial_passed,
        chaos_passed=chaos_passed,
        bounds_passed=bounds_passed,
        secret_hygiene_passed=secret_hygiene_passed,
        deterministic_degraded_passed=deterministic_degraded_passed,
        lazy_discovery_passed=lazy_passed,
        passed=passed,
        failure_codes=tuple(sorted(set(failures))),
    )


def verify_prompt_workflow_gate_report(
    report: PromptWorkflowGateReport,
    benchmark: PromptWorkflowBenchmark,
) -> bool:
    if not isinstance(report, PromptWorkflowGateReport):
        return False
    replayed = recompute_prompt_workflow_gate(benchmark)
    return _canonical_bytes(report.to_dict()) == _canonical_bytes(
        replayed.to_dict()
    )


def _cid(label: str) -> str:
    return _identity({"frozen-prompt-workflow": label})


def build_frozen_prompt_workflow_benchmark(
    *,
    observation_label: str = "qualification",
    tree_id: str = "sha256:frozen-prompt-workflow-tree",
) -> PromptWorkflowBenchmark:
    """Build the deterministic closed smoke population for ASI-159 gates.

    This is a local conformance fixture, not production evidence.  Promotion
    requires a later separate fresh-root evaluation with live receipts.
    """

    label = _code(observation_label, "observation_label")
    task_a = _cid("task:bootstrap-contracts")
    task_b = _cid("task:bootstrap-materialize")
    task_c = _cid("task:bootstrap-lifecycle")
    admitted = (task_a, task_b, task_c)
    ready = (task_a,)
    effects = (
        _cid("effect:materialize-plan"),
        _cid("effect:start-supervisor"),
    )
    identity = FrozenPromptFixtureIdentity(
        repository_id="repository:prompt-workflow-benchmark@1",
        tree_id=tree_id,
        prompt_fixture_id="prompt-fixture:improve-retry-recovery@1",
        prompt_cid=_cid(f"prompt:{label}"),
        scan_cid=_cid(f"scan:{label}"),
        plan_root_cid=_cid(f"plan:{label}"),
        objective_id="ASI-G470",
        objective_revision="sha256:frozen-prompt-objective",
        policy_id="policy:prompt-workflow-rollout@1",
        policy_revision="sha256:frozen-prompt-policy",
        capability_id="capability:prompt-workflow-local@1",
        capability_revision="sha256:frozen-prompt-capability",
        partition_id="partition:frozen-prompt-workflow@1",
    )
    projection_cid = _cid(f"projection:{label}")
    run_cid = _cid(f"run:{label}")

    def paired_metrics(
        *, planning_mode: PlanningMode
    ) -> PromptWorkflowMetrics:
        model_calls = 1 if planning_mode is PlanningMode.MODEL else 0
        input_tokens = 120 if planning_mode is PlanningMode.MODEL else 0
        output_tokens = 40 if planning_mode is PlanningMode.MODEL else 0
        return PromptWorkflowMetrics(
            admitted_task_cids=admitted,
            ready_task_cids=ready,
            accepted_effect_ids=effects,
            terminal_result=TerminalOutcome.ACCEPTED.value,
            model_calls=model_calls,
            provider_input_tokens=input_tokens,
            provider_output_tokens=output_tokens,
            retries=0,
            storage_bytes=8_192,
            process_count=1,
            materialization_latency_ms=25,
            recovery_latency_ms=0,
            secret_bytes_emitted=0,
            escape_count=0,
        )

    receipts: list[PromptWorkflowProducerReceipt] = []
    for planning_mode in REQUIRED_PLANNING_MODES:
        for task_source in REQUIRED_TASK_SOURCES:
            for transport in REQUIRED_TRANSPORTS:
                receipts.append(
                    PromptWorkflowProducerReceipt(
                        identity=identity,
                        planning_mode=planning_mode,
                        task_source=task_source,
                        transport=transport,
                        metrics=paired_metrics(planning_mode=planning_mode),
                        source_receipt_ids=(
                            _cid(
                                f"source:{label}:{planning_mode.value}:"
                                f"{task_source.value}:{transport.value}"
                            ),
                        ),
                        projection_cid=projection_cid,
                        run_cid=run_cid,
                        lazy_discovery=True,
                    )
                )

    adversarial_metrics = PromptWorkflowMetrics(
        admitted_task_cids=(),
        ready_task_cids=(),
        accepted_effect_ids=(),
        terminal_result=TerminalOutcome.REJECTED.value,
        model_calls=0,
        provider_input_tokens=0,
        provider_output_tokens=0,
        retries=0,
        storage_bytes=256,
        process_count=0,
        materialization_latency_ms=1,
        recovery_latency_ms=0,
    )
    for fixture in REQUIRED_ADVERSARIAL_FIXTURES:
        terminal = TerminalOutcome.REJECTED.value
        if fixture in {
            AdversarialFixture.AUTHORIZATION_BYPASS,
            AdversarialFixture.PERMIT_FORGERY,
        }:
            terminal = TerminalOutcome.DENIED.value
        elif fixture is AdversarialFixture.SECRET_LEAK:
            terminal = TerminalOutcome.FAIL_CLOSED.value
        elif fixture is AdversarialFixture.SHELL_RESCUE_PROPOSAL:
            terminal = TerminalOutcome.QUARANTINED.value
        receipts.append(
            PromptWorkflowProducerReceipt(
                identity=identity,
                planning_mode=PlanningMode.DETERMINISTIC,
                task_source=TaskSourceBackend.BOTH,
                transport=TransportSurface.PYTHON,
                metrics=replace(
                    adversarial_metrics, terminal_result=terminal
                ),
                source_receipt_ids=(
                    _cid(f"source:{label}:adversarial:{fixture.value}"),
                ),
                adversarial_fixture=fixture,
                lazy_discovery=True,
            )
        )

    fault_cycle = (
        FaultOutcome.RESUME,
        FaultOutcome.COMPENSATE,
        FaultOutcome.QUARANTINE,
    )
    for index, boundary in enumerate(REQUIRED_CHAOS_BOUNDARIES):
        outcome = fault_cycle[index % len(fault_cycle)]
        terminal = (
            TerminalOutcome.HEALTHY.value
            if outcome is not FaultOutcome.QUARANTINE
            else TerminalOutcome.QUARANTINED.value
        )
        receipts.append(
            PromptWorkflowProducerReceipt(
                identity=identity,
                planning_mode=PlanningMode.DETERMINISTIC,
                task_source=TaskSourceBackend.BOTH,
                transport=TransportSurface.PYTHON,
                metrics=PromptWorkflowMetrics(
                    admitted_task_cids=admitted,
                    ready_task_cids=ready,
                    accepted_effect_ids=(
                        effects if outcome is not FaultOutcome.QUARANTINE else ()
                    ),
                    terminal_result=terminal,
                    model_calls=0,
                    provider_input_tokens=0,
                    provider_output_tokens=0,
                    retries=1 if outcome is FaultOutcome.RESUME else 0,
                    storage_bytes=4_096,
                    process_count=1 if outcome is not FaultOutcome.QUARANTINE else 0,
                    materialization_latency_ms=10,
                    recovery_latency_ms=15,
                ),
                source_receipt_ids=(
                    _cid(f"source:{label}:chaos:{boundary.value}"),
                ),
                chaos_boundary=boundary,
                fault_outcome=outcome,
                run_cid=run_cid if outcome is not FaultOutcome.QUARANTINE else "",
                rescue_plan_cid=(
                    _cid(f"rescue:{label}:{boundary.value}")
                    if "rescue" in boundary.value
                    else ""
                ),
                lazy_discovery=True,
            )
        )

    for dependency in REQUIRED_OPTIONAL_DEPENDENCIES:
        terminal = (
            TerminalOutcome.DEGRADED.value
            if dependency
            in {
                OptionalDependency.DUCKDB,
                OptionalDependency.LLM_ROUTER,
                OptionalDependency.MODEL_CREDENTIALS,
            }
            else TerminalOutcome.FAIL_CLOSED.value
        )
        # MCP/discovery absence is explicit fail-closed; DuckDB can fall back.
        if dependency is OptionalDependency.DUCKDB:
            terminal = TerminalOutcome.DEGRADED.value
        elif dependency is OptionalDependency.MCP:
            terminal = TerminalOutcome.FAIL_CLOSED.value
        receipts.append(
            PromptWorkflowProducerReceipt(
                identity=identity,
                planning_mode=PlanningMode.DETERMINISTIC,
                task_source=(
                    TaskSourceBackend.MARKDOWN
                    if dependency is OptionalDependency.DUCKDB
                    else TaskSourceBackend.BOTH
                ),
                transport=(
                    TransportSurface.PYTHON
                    if dependency is OptionalDependency.MCP
                    else TransportSurface.PYTHON
                ),
                metrics=PromptWorkflowMetrics(
                    admitted_task_cids=admitted
                    if terminal != TerminalOutcome.FAIL_CLOSED.value
                    else (),
                    ready_task_cids=ready
                    if terminal != TerminalOutcome.FAIL_CLOSED.value
                    else (),
                    accepted_effect_ids=()
                    if terminal == TerminalOutcome.FAIL_CLOSED.value
                    else effects[:1],
                    terminal_result=terminal,
                    model_calls=0,
                    provider_input_tokens=0,
                    provider_output_tokens=0,
                    retries=0,
                    storage_bytes=1_024,
                    process_count=0,
                    materialization_latency_ms=5,
                    recovery_latency_ms=0,
                ),
                source_receipt_ids=(
                    _cid(f"source:{label}:degraded:{dependency.value}"),
                ),
                optional_dependency=dependency,
                degraded_local=True,
                deterministic_replay_id=_cid(
                    f"local-replay:{dependency.value}:{terminal}"
                ),
                lazy_discovery=True,
            )
        )

    return PromptWorkflowBenchmark(tuple(receipts))


__all__ = (
    "AdversarialFixture",
    "ChaosBoundary",
    "FaultOutcome",
    "FrozenPromptFixtureIdentity",
    "OptionalDependency",
    "PROMPT_WORKFLOW_BENCHMARK_REQUIREMENT_ID",
    "PROMPT_WORKFLOW_BENCHMARK_SCHEMA",
    "PROMPT_WORKFLOW_BENCHMARK_VERSION",
    "PROMPT_WORKFLOW_GATE_REPORT_SCHEMA",
    "PROMPT_WORKFLOW_PRODUCER_RECEIPT_SCHEMA",
    "PlanningMode",
    "PromptWorkflowBenchmark",
    "PromptWorkflowBenchmarkError",
    "PromptWorkflowGateReport",
    "PromptWorkflowMetrics",
    "PromptWorkflowProducerReceipt",
    "REQUIRED_ADVERSARIAL_FIXTURES",
    "REQUIRED_CHAOS_BOUNDARIES",
    "REQUIRED_OPTIONAL_DEPENDENCIES",
    "REQUIRED_PLANNING_MODES",
    "REQUIRED_TASK_SOURCES",
    "REQUIRED_TRANSPORTS",
    "TaskSourceBackend",
    "TerminalOutcome",
    "TransportSurface",
    "build_frozen_prompt_workflow_benchmark",
    "recompute_prompt_workflow_gate",
    "verify_prompt_workflow_gate_report",
)
