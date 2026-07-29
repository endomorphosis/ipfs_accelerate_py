"""Replayable proof-dependency scaling benchmark for the decision runtime.

This module is deliberately provider-free.  It consumes producer-owned
receipts, freezes paired observations, and recomputes every reported metric.
It does not estimate tokens, inspect a checkout, resolve optional providers,
or grant execution/completion authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Final


DECISION_RUNTIME_BENCHMARK_VERSION: Final = 1
DECISION_RUNTIME_PRODUCER_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/decision-runtime-producer-receipt@1"
)
DECISION_RUNTIME_BENCHMARK_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/decision-runtime-benchmark@1"
)
PROOF_DEPENDENCY_SCALING_REPORT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/proof-dependency-scaling-report@1"
)
PROOF_DEPENDENCY_SCALING_REQUIREMENT_ID: Final = (
    "asi-139:proof-dependency-context-scaling"
)
MINIMUM_IRRELEVANT_SCALE_FACTOR: Final = 10
MAX_RECEIPTS: Final = 100_000
MAX_COUNTER: Final = 10**15
MAX_REPORT_BYTES: Final = 8 * 1024 * 1024

_CONTENT_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_CODE = re.compile(r"^[a-z][a-z0-9_.:/@-]{0,191}$")


class DecisionRuntimeBenchmarkError(ValueError):
    """Benchmark source evidence is malformed, incomplete, or inconsistent."""


class DecisionRuntimePath(str, Enum):
    CURRENT = "current"
    PROOF_DIRECTED = "proof-directed"


class IrrelevantCorpus(str, Enum):
    LEGAL = "legal-corpus"
    CODEBASE = "codebase"
    SKILLCENTER_ROWS = "skillcenter-rows"
    SKILLCENTER_GRAPH = "skillcenter-graph"
    CONVERSATION = "conversation-history"


class AdversarialFixture(str, Enum):
    FORGED_CID = "forged-cid"
    CANONICALIZATION = "canonicalization"
    SCHEMA = "schema"
    STALE_ROOT = "stale-root"
    CROSS_PARTITION = "cross-partition"
    PROMPT_INJECTION = "prompt-injection"
    POISONED_EMBEDDING = "poisoned-embedding"
    INAPPLICABLE_LAW = "inapplicable-law"
    LEGAL_CONFLICT = "legal-conflict"
    SECURITY_IR_DENY = "securityir-deny"
    SECURITY_IR_UNKNOWN = "securityir-unknown"
    INTENT_AUTHORITY_CONFUSION = "intent-authority-confusion"
    DIRTY_FILE = "dirty-file"
    CHANGED_TOOL_ARGUMENT = "changed-tool-argument"
    STALE_LEASE = "stale-lease"
    PROOF_REPLAY = "proof-replay"
    GRAPH_TRUNCATION = "graph-truncation"
    RECOVERY = "recovery"
    PATH_ESCAPE = "path-escape"
    EFFECT_ESCAPE = "effect-escape"
    MANDATORY_OMISSION = "mandatory-omission"


REQUIRED_IRRELEVANT_CORPORA: Final[tuple[IrrelevantCorpus, ...]] = tuple(
    IrrelevantCorpus
)
REQUIRED_ADVERSARIAL_FIXTURES: Final[tuple[AdversarialFixture, ...]] = tuple(
    AdversarialFixture
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
    if hasattr(value, "to_record") and callable(value.to_record):
        return _plain(value.to_record())
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
        raise DecisionRuntimeBenchmarkError(
            "benchmark evidence must be canonical JSON"
        ) from exc


def _identity(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _load_json(value: str | bytes | bytearray, name: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise DecisionRuntimeBenchmarkError(
                    f"{name} contains duplicate JSON key {key!r}"
                )
            result[key] = item
        return result

    try:
        if isinstance(value, (bytes, bytearray)):
            value = bytes(value).decode("utf-8")
        if not isinstance(value, str):
            raise DecisionRuntimeBenchmarkError(f"{name} must be JSON text")
        return json.loads(value, object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DecisionRuntimeBenchmarkError(f"{name} is invalid JSON") from exc


def _text(value: Any, name: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise DecisionRuntimeBenchmarkError(
            f"{name} must be non-empty canonical text"
        )
    if "\x00" in value or len(value.encode("utf-8")) > maximum:
        raise DecisionRuntimeBenchmarkError(f"{name} is unsafe or too large")
    return value


def _code(value: Any, name: str) -> str:
    result = _text(str(getattr(value, "value", value)), name, maximum=192)
    if not _CODE.fullmatch(result):
        raise DecisionRuntimeBenchmarkError(f"{name} must be a compact code")
    return result


def _content_id(value: Any, name: str) -> str:
    result = _text(value, name, maximum=71)
    if not _CONTENT_ID.fullmatch(result):
        raise DecisionRuntimeBenchmarkError(
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
        raise DecisionRuntimeBenchmarkError(
            f"{name} must be an integer from {minimum} through {MAX_COUNTER}"
        )
    return value


def _enum(value: Any, kind: type[Enum], name: str) -> Any:
    if isinstance(value, kind):
        return value
    try:
        return kind(str(getattr(value, "value", value)))
    except (TypeError, ValueError) as exc:
        raise DecisionRuntimeBenchmarkError(f"invalid {name}") from exc


def _ids(values: Sequence[Any], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise DecisionRuntimeBenchmarkError(f"{name} must be a sequence")
    result = tuple(sorted(_text(str(v), name, maximum=512) for v in values))
    if len(result) != len(set(result)):
        raise DecisionRuntimeBenchmarkError(f"{name} must be unique")
    return result


@dataclass(frozen=True)
class FrozenDecisionIdentity:
    """Exact identities shared by both live paths and every scale ablation."""

    repository_id: str
    tree_id: str
    decision_request_id: str
    objective_id: str
    objective_revision: str
    policy_id: str
    policy_revision: str
    capability_id: str
    capability_revision: str
    provider_id: str
    provider_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    partition_id: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self, name, _text(getattr(self, name), name, maximum=512)
            )

    @property
    def identity_id(self) -> str:
        return _identity(self.to_dict())

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenDecisionIdentity":
        if set(value) != set(cls.__dataclass_fields__):
            raise DecisionRuntimeBenchmarkError("invalid frozen identity fields")
        return cls(**dict(value))


@dataclass(frozen=True)
class CorpusScale:
    """Independent scale intervention over irrelevant inputs."""

    legal_corpus: int = 1
    codebase: int = 1
    skillcenter_rows: int = 1
    skillcenter_graph: int = 1
    conversation_history: int = 1

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self, name, _integer(getattr(self, name), name, minimum=1)
            )

    @property
    def intervention(self) -> IrrelevantCorpus | None:
        changed = [
            kind
            for kind, value in self.by_kind.items()
            if value != 1
        ]
        if not changed:
            return None
        if len(changed) != 1:
            raise DecisionRuntimeBenchmarkError(
                "irrelevant corpora must be scaled independently"
            )
        return changed[0]

    @property
    def by_kind(self) -> Mapping[IrrelevantCorpus, int]:
        return {
            IrrelevantCorpus.LEGAL: self.legal_corpus,
            IrrelevantCorpus.CODEBASE: self.codebase,
            IrrelevantCorpus.SKILLCENTER_ROWS: self.skillcenter_rows,
            IrrelevantCorpus.SKILLCENTER_GRAPH: self.skillcenter_graph,
            IrrelevantCorpus.CONVERSATION: self.conversation_history,
        }

    @property
    def scale_id(self) -> str:
        return _identity(self.to_dict())

    def to_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CorpusScale":
        if set(value) != set(cls.__dataclass_fields__):
            raise DecisionRuntimeBenchmarkError("invalid corpus scale fields")
        return cls(**dict(value))


@dataclass(frozen=True)
class DecisionRuntimeMetrics:
    """Counters joined from the context/runtime/cache/proof producer receipts."""

    provider_input_tokens: int
    provider_output_tokens: int
    provider_reused_input_tokens: int
    mandatory_closure_nodes: int
    mandatory_closure_bytes: int
    total_corpus_nodes: int
    total_corpus_bytes: int
    cache_lookups: int
    cache_hits: int
    cache_reused_bytes: int
    invalidation_expected: int
    invalidation_actual: int
    invalidation_true_positive: int
    first_valid_plan: bool
    retries: int
    proof_cost: int
    validation_cost: int
    declared_effect_ids: tuple[str, ...]
    observed_effect_ids: tuple[str, ...]
    terminal_result: str
    index_metadata_bytes: int = 0

    def __post_init__(self) -> None:
        for name in (
            "provider_input_tokens",
            "provider_output_tokens",
            "provider_reused_input_tokens",
            "mandatory_closure_nodes",
            "mandatory_closure_bytes",
            "total_corpus_nodes",
            "total_corpus_bytes",
            "cache_lookups",
            "cache_hits",
            "cache_reused_bytes",
            "invalidation_expected",
            "invalidation_actual",
            "invalidation_true_positive",
            "retries",
            "proof_cost",
            "validation_cost",
            "index_metadata_bytes",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if not isinstance(self.first_valid_plan, bool):
            raise DecisionRuntimeBenchmarkError(
                "first_valid_plan must be boolean"
            )
        object.__setattr__(
            self,
            "declared_effect_ids",
            _ids(self.declared_effect_ids, "declared_effect_ids"),
        )
        object.__setattr__(
            self,
            "observed_effect_ids",
            _ids(self.observed_effect_ids, "observed_effect_ids"),
        )
        object.__setattr__(
            self, "terminal_result", _code(self.terminal_result, "terminal_result")
        )
        if self.cache_hits > self.cache_lookups:
            raise DecisionRuntimeBenchmarkError("cache hits exceed lookups")
        if self.invalidation_true_positive > min(
            self.invalidation_expected, self.invalidation_actual
        ):
            raise DecisionRuntimeBenchmarkError(
                "invalidation true positives exceed either population"
            )

    @property
    def invalidation_false_positive(self) -> int:
        return self.invalidation_actual - self.invalidation_true_positive

    @property
    def invalidation_false_negative(self) -> int:
        return self.invalidation_expected - self.invalidation_true_positive

    @property
    def invalidation_exact(self) -> bool:
        return (
            self.invalidation_false_positive == 0
            and self.invalidation_false_negative == 0
        )

    @property
    def cache_reuse_millionths(self) -> int:
        if not self.cache_lookups:
            return 1_000_000
        return self.cache_hits * 1_000_000 // self.cache_lookups

    def to_dict(self) -> dict[str, Any]:
        payload = {
            name: _plain(getattr(self, name))
            for name in self.__dataclass_fields__
        }
        payload.update(
            invalidation_false_positive=self.invalidation_false_positive,
            invalidation_false_negative=self.invalidation_false_negative,
            invalidation_exact=self.invalidation_exact,
            cache_reuse_millionths=self.cache_reuse_millionths,
        )
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DecisionRuntimeMetrics":
        fields = set(cls.__dataclass_fields__)
        if not fields.issubset(value) or set(value).difference(
            fields
            | {
                "invalidation_false_positive",
                "invalidation_false_negative",
                "invalidation_exact",
                "cache_reuse_millionths",
            }
        ):
            raise DecisionRuntimeBenchmarkError("invalid metrics fields")
        return cls(**{name: value[name] for name in fields})


@dataclass(frozen=True)
class DecisionRuntimeProducerReceipt:
    """One immutable observation derived from producer receipt identities."""

    identity: FrozenDecisionIdentity
    path: DecisionRuntimePath
    scale: CorpusScale
    metrics: DecisionRuntimeMetrics
    mandatory_closure_id: str
    context_id: str
    source_receipt_ids: tuple[str, ...]
    adversarial_fixture: AdversarialFixture | None = None
    escape_count: int = 0
    degraded_local: bool = False
    deterministic_replay_id: str = ""
    lazy_discovery: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.identity, FrozenDecisionIdentity):
            raise DecisionRuntimeBenchmarkError("identity is not frozen")
        object.__setattr__(
            self, "path", _enum(self.path, DecisionRuntimePath, "path")
        )
        if not isinstance(self.scale, CorpusScale):
            raise DecisionRuntimeBenchmarkError("scale must be CorpusScale")
        if not isinstance(self.metrics, DecisionRuntimeMetrics):
            raise DecisionRuntimeBenchmarkError(
                "metrics must be DecisionRuntimeMetrics"
            )
        object.__setattr__(
            self,
            "mandatory_closure_id",
            _content_id(self.mandatory_closure_id, "mandatory_closure_id"),
        )
        object.__setattr__(
            self, "context_id", _content_id(self.context_id, "context_id")
        )
        object.__setattr__(
            self,
            "source_receipt_ids",
            _ids(self.source_receipt_ids, "source_receipt_ids"),
        )
        if not self.source_receipt_ids:
            raise DecisionRuntimeBenchmarkError(
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
        object.__setattr__(
            self, "escape_count", _integer(self.escape_count, "escape_count")
        )
        if not isinstance(self.degraded_local, bool) or not isinstance(
            self.lazy_discovery, bool
        ):
            raise DecisionRuntimeBenchmarkError(
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
        # Validate the independent intervention immediately.
        intervention = self.scale.intervention
        if intervention is not None and (
            self.scale.by_kind[intervention] < MINIMUM_IRRELEVANT_SCALE_FACTOR
        ):
            raise DecisionRuntimeBenchmarkError(
                "scaled irrelevant corpus must grow by at least 10x"
            )

    @property
    def receipt_id(self) -> str:
        return _identity(self.to_dict(include_receipt_id=False))

    def to_dict(self, *, include_receipt_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": DECISION_RUNTIME_PRODUCER_RECEIPT_SCHEMA,
            "version": DECISION_RUNTIME_BENCHMARK_VERSION,
            "identity": self.identity.to_dict(),
            "path": self.path.value,
            "scale": self.scale.to_dict(),
            "metrics": self.metrics.to_dict(),
            "mandatory_closure_id": self.mandatory_closure_id,
            "context_id": self.context_id,
            "source_receipt_ids": list(self.source_receipt_ids),
            "adversarial_fixture": (
                self.adversarial_fixture.value
                if self.adversarial_fixture is not None
                else None
            ),
            "escape_count": self.escape_count,
            "degraded_local": self.degraded_local,
            "deterministic_replay_id": self.deterministic_replay_id,
            "lazy_discovery": self.lazy_discovery,
        }
        if include_receipt_id:
            payload["receipt_id"] = self.receipt_id
        return payload

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "DecisionRuntimeProducerReceipt":
        allowed = {
            "schema",
            "version",
            "receipt_id",
            "identity",
            "path",
            "scale",
            "metrics",
            "mandatory_closure_id",
            "context_id",
            "source_receipt_ids",
            "adversarial_fixture",
            "escape_count",
            "degraded_local",
            "deterministic_replay_id",
            "lazy_discovery",
        }
        if set(value).difference(allowed):
            raise DecisionRuntimeBenchmarkError("unknown producer receipt fields")
        if (
            value.get("schema") != DECISION_RUNTIME_PRODUCER_RECEIPT_SCHEMA
            or value.get("version") != DECISION_RUNTIME_BENCHMARK_VERSION
        ):
            raise DecisionRuntimeBenchmarkError(
                "unsupported producer receipt schema"
            )
        result = cls(
            identity=FrozenDecisionIdentity.from_dict(value["identity"]),
            path=value["path"],
            scale=CorpusScale.from_dict(value["scale"]),
            metrics=DecisionRuntimeMetrics.from_dict(value["metrics"]),
            mandatory_closure_id=value["mandatory_closure_id"],
            context_id=value["context_id"],
            source_receipt_ids=tuple(value["source_receipt_ids"]),
            adversarial_fixture=value.get("adversarial_fixture"),
            escape_count=value.get("escape_count", 0),
            degraded_local=value.get("degraded_local", False),
            deterministic_replay_id=value.get("deterministic_replay_id", ""),
            lazy_discovery=value.get("lazy_discovery", True),
        )
        if value.get("receipt_id", result.receipt_id) != result.receipt_id:
            raise DecisionRuntimeBenchmarkError("producer receipt ID mismatch")
        return result

    def to_json(self) -> str:
        return _canonical_bytes(self.to_dict()).decode("utf-8")

    @classmethod
    def from_json(
        cls, value: str | bytes | bytearray
    ) -> "DecisionRuntimeProducerReceipt":
        return cls.from_dict(_load_json(value, "producer receipt"))


@dataclass(frozen=True)
class DecisionRuntimeBenchmark:
    """The complete closed source population; no selected-case evaluation."""

    receipts: tuple[DecisionRuntimeProducerReceipt, ...]
    requirement_id: str = PROOF_DEPENDENCY_SCALING_REQUIREMENT_ID

    def __post_init__(self) -> None:
        receipts = tuple(self.receipts)
        if not receipts or len(receipts) > MAX_RECEIPTS:
            raise DecisionRuntimeBenchmarkError(
                "benchmark receipt population is empty or unbounded"
            )
        if any(
            not isinstance(item, DecisionRuntimeProducerReceipt)
            for item in receipts
        ):
            raise DecisionRuntimeBenchmarkError(
                "benchmark contains non-producer receipts"
            )
        receipt_ids = [item.receipt_id for item in receipts]
        if len(receipt_ids) != len(set(receipt_ids)):
            raise DecisionRuntimeBenchmarkError("duplicate producer receipt")
        object.__setattr__(
            self, "receipts", tuple(sorted(receipts, key=lambda r: r.receipt_id))
        )
        if self.requirement_id != PROOF_DEPENDENCY_SCALING_REQUIREMENT_ID:
            raise DecisionRuntimeBenchmarkError("wrong benchmark requirement")

    @property
    def benchmark_id(self) -> str:
        return _identity(self.to_dict(include_benchmark_id=False))

    def to_dict(self, *, include_benchmark_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": DECISION_RUNTIME_BENCHMARK_SCHEMA,
            "version": DECISION_RUNTIME_BENCHMARK_VERSION,
            "requirement_id": self.requirement_id,
            "receipts": [item.to_dict() for item in self.receipts],
        }
        if include_benchmark_id:
            payload["benchmark_id"] = self.benchmark_id
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DecisionRuntimeBenchmark":
        allowed = {
            "schema",
            "version",
            "requirement_id",
            "receipts",
            "benchmark_id",
        }
        if set(value).difference(allowed):
            raise DecisionRuntimeBenchmarkError("unknown benchmark fields")
        if (
            value.get("schema") != DECISION_RUNTIME_BENCHMARK_SCHEMA
            or value.get("version") != DECISION_RUNTIME_BENCHMARK_VERSION
        ):
            raise DecisionRuntimeBenchmarkError("unsupported benchmark schema")
        result = cls(
            receipts=tuple(
                DecisionRuntimeProducerReceipt.from_dict(item)
                for item in value["receipts"]
            ),
            requirement_id=value["requirement_id"],
        )
        if value.get("benchmark_id", result.benchmark_id) != result.benchmark_id:
            raise DecisionRuntimeBenchmarkError("benchmark ID mismatch")
        return result

    def to_json(self) -> str:
        return _canonical_bytes(self.to_dict()).decode("utf-8")

    @classmethod
    def from_json(
        cls, value: str | bytes | bytearray
    ) -> "DecisionRuntimeBenchmark":
        return cls.from_dict(_load_json(value, "decision runtime benchmark"))


def _correlation(xs: Sequence[int], ys: Sequence[int]) -> int:
    """Return absolute Pearson correlation as integer millionths."""

    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return 0
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum(
        (x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)
    )
    x_norm = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_norm = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if not x_norm or not y_norm:
        return 0
    return min(1_000_000, round(abs(numerator / (x_norm * y_norm)) * 1_000_000))


@dataclass(frozen=True)
class ProofDependencyScalingReport:
    """Recomputed causal result.  This report has no execution authority."""

    benchmark_id: str
    decision_count: int
    receipt_count: int
    current_provider_tokens: int
    proof_directed_provider_tokens: int
    mandatory_closure_nodes: int
    mandatory_closure_bytes: int
    total_corpus_nodes: int
    total_corpus_bytes: int
    cache_lookups: int
    cache_hits: int
    cache_reused_bytes: int
    first_valid_plans: int
    retries: int
    proof_cost: int
    validation_cost: int
    invalidation_false_positives: int
    invalidation_false_negatives: int
    closure_token_correlation_millionths: int
    corpus_token_correlation_millionths: int
    scale_dimensions_passed: tuple[str, ...]
    adversarial_fixtures_passed: tuple[str, ...]
    context_scaling_passed: bool
    cache_reuse_passed: bool
    invalidation_precision_passed: bool
    effect_parity_passed: bool
    terminal_parity_passed: bool
    deterministic_degraded_passed: bool
    lazy_discovery_passed: bool
    passed: bool
    failure_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _content_id(self.benchmark_id, "benchmark_id")
        object.__setattr__(
            self, "failure_codes", tuple(sorted(set(self.failure_codes)))
        )

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
            "schema": PROOF_DEPENDENCY_SCALING_REPORT_SCHEMA,
            "version": DECISION_RUNTIME_BENCHMARK_VERSION,
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
        benchmark: DecisionRuntimeBenchmark,
    ) -> "ProofDependencyScalingReport":
        """Restore only by replaying the complete producer population."""

        replayed = recompute_proof_dependency_scaling(benchmark)
        if _canonical_bytes(value) != _canonical_bytes(replayed.to_dict()):
            raise DecisionRuntimeBenchmarkError(
                "scaling report does not match producer receipt replay"
            )
        return replayed

    @classmethod
    def from_json(
        cls,
        value: str | bytes | bytearray,
        *,
        benchmark: DecisionRuntimeBenchmark,
    ) -> "ProofDependencyScalingReport":
        return cls.from_dict(
            _load_json(value, "proof dependency scaling report"),
            benchmark=benchmark,
        )


def recompute_proof_dependency_scaling(
    benchmark: DecisionRuntimeBenchmark,
) -> ProofDependencyScalingReport:
    """Replay the complete paired/adversarial population from source receipts."""

    if not isinstance(benchmark, DecisionRuntimeBenchmark):
        raise DecisionRuntimeBenchmarkError(
            "benchmark must be DecisionRuntimeBenchmark"
        )
    normal = [r for r in benchmark.receipts if r.adversarial_fixture is None]
    adversarial = [
        r for r in benchmark.receipts if r.adversarial_fixture is not None
    ]
    failures: list[str] = []
    groups: dict[str, list[DecisionRuntimeProducerReceipt]] = {}
    for receipt in normal:
        groups.setdefault(receipt.identity.identity_id, []).append(receipt)

    dimensions_passed: set[str] = set()
    effect_parity = True
    terminal_parity = True
    for identity_id, population in groups.items():
        by_key = {(r.path, r.scale.scale_id): r for r in population}
        baselines = {
            r.path: r
            for r in population
            if r.scale.intervention is None
        }
        if set(baselines) != set(DecisionRuntimePath):
            failures.append(f"missing-paired-baseline:{identity_id}")
            continue
        current = baselines[DecisionRuntimePath.CURRENT]
        proof = baselines[DecisionRuntimePath.PROOF_DIRECTED]
        if (
            current.metrics.declared_effect_ids
            != proof.metrics.declared_effect_ids
            or current.metrics.observed_effect_ids
            != proof.metrics.observed_effect_ids
        ):
            effect_parity = False
        if current.metrics.terminal_result != proof.metrics.terminal_result:
            terminal_parity = False
        for corpus in REQUIRED_IRRELEVANT_CORPORA:
            candidates = [
                r for r in population
                if r.scale.intervention is corpus
                and r.scale.by_kind[corpus] >= MINIMUM_IRRELEVANT_SCALE_FACTOR
            ]
            if {r.path for r in candidates} != set(DecisionRuntimePath):
                failures.append(
                    f"missing-independent-scale:{identity_id}:{corpus.value}"
                )
                continue
            valid = True
            for grown in candidates:
                base = baselines[grown.path]
                if (
                    grown.metrics.total_corpus_nodes
                    < base.metrics.total_corpus_nodes
                    * MINIMUM_IRRELEVANT_SCALE_FACTOR
                    or grown.metrics.total_corpus_bytes
                    < base.metrics.total_corpus_bytes
                    * MINIMUM_IRRELEVANT_SCALE_FACTOR
                    or grown.mandatory_closure_id
                    != base.mandatory_closure_id
                    or grown.metrics.mandatory_closure_nodes
                    != base.metrics.mandatory_closure_nodes
                    or grown.metrics.mandatory_closure_bytes
                    != base.metrics.mandatory_closure_bytes
                    or grown.metrics.declared_effect_ids
                    != base.metrics.declared_effect_ids
                    or grown.metrics.observed_effect_ids
                    != base.metrics.observed_effect_ids
                    or grown.metrics.terminal_result
                    != base.metrics.terminal_result
                ):
                    valid = False
                # The proof-directed provider context must not follow irrelevant
                # corpus growth.  Current-path growth is measured, not trusted.
                if (
                    grown.path is DecisionRuntimePath.PROOF_DIRECTED
                    and grown.metrics.provider_input_tokens
                    != base.metrics.provider_input_tokens
                ):
                    valid = False
            if valid:
                dimensions_passed.add(corpus.value)
            else:
                failures.append(
                    f"context-follows-irrelevant-corpus:{identity_id}:{corpus.value}"
                )
        if len(by_key) != len(population):
            failures.append(f"duplicate-scale-observation:{identity_id}")

    fixture_counts = {fixture: 0 for fixture in REQUIRED_ADVERSARIAL_FIXTURES}
    fixture_passed: set[str] = set()
    for receipt in adversarial:
        fixture = receipt.adversarial_fixture
        assert fixture is not None
        fixture_counts[fixture] += 1
        if receipt.escape_count or receipt.metrics.terminal_result not in {
            "denied",
            "rejected",
            "degraded",
            "fail-closed",
        }:
            failures.append(f"adversarial-escape:{fixture.value}")
        else:
            fixture_passed.add(fixture.value)
    for fixture, count in fixture_counts.items():
        if not count:
            failures.append(f"missing-adversarial-fixture:{fixture.value}")

    degraded = [r for r in benchmark.receipts if r.degraded_local]
    degraded_passed = bool(degraded) and all(
        r.escape_count == 0 and bool(r.deterministic_replay_id) for r in degraded
    )
    if not degraded_passed:
        failures.append("deterministic-local-degraded-operation")
    lazy_passed = all(r.lazy_discovery for r in benchmark.receipts)
    if not lazy_passed:
        failures.append("eager-optional-discovery")

    proof_receipts = [
        r for r in normal if r.path is DecisionRuntimePath.PROOF_DIRECTED
    ]
    metrics = [r.metrics for r in benchmark.receipts]
    invalidation_fp = sum(m.invalidation_false_positive for m in metrics)
    invalidation_fn = sum(m.invalidation_false_negative for m in metrics)
    invalidation_passed = invalidation_fp == 0 and invalidation_fn == 0
    if not invalidation_passed:
        failures.append("imprecise-invalidation")
    warm_receipts = [
        r
        for r in proof_receipts
        if r.scale.intervention is None
    ]
    cache_passed = bool(warm_receipts) and all(
        r.metrics.cache_lookups > 0
        and r.metrics.cache_hits == r.metrics.cache_lookups
        and r.metrics.cache_reused_bytes >= r.metrics.mandatory_closure_bytes
        and r.metrics.provider_reused_input_tokens > 0
        for r in warm_receipts
    )
    if not cache_passed:
        failures.append("exact-warm-cache-reuse-missing")
    if not effect_parity:
        failures.append("effect-parity")
    if not terminal_parity:
        failures.append("terminal-parity")

    closure_tokens = _correlation(
        [r.metrics.mandatory_closure_bytes for r in proof_receipts],
        [r.metrics.provider_input_tokens for r in proof_receipts],
    )
    corpus_tokens = _correlation(
        [r.metrics.total_corpus_bytes for r in proof_receipts],
        [r.metrics.provider_input_tokens for r in proof_receipts],
    )
    context_passed = (
        set(dimensions_passed)
        == {item.value for item in REQUIRED_IRRELEVANT_CORPORA}
        and closure_tokens >= 900_000
        and corpus_tokens <= 500_000
        and closure_tokens > corpus_tokens
    )
    if not context_passed:
        failures.append("proof-closure-scaling-gate")

    passed = not failures
    return ProofDependencyScalingReport(
        benchmark_id=benchmark.benchmark_id,
        decision_count=len(groups),
        receipt_count=len(benchmark.receipts),
        current_provider_tokens=sum(
            m.provider_input_tokens + m.provider_output_tokens
            for r in normal
            if r.path is DecisionRuntimePath.CURRENT
            for m in (r.metrics,)
        ),
        proof_directed_provider_tokens=sum(
            m.provider_input_tokens + m.provider_output_tokens
            for r in proof_receipts
            for m in (r.metrics,)
        ),
        mandatory_closure_nodes=sum(m.mandatory_closure_nodes for m in metrics),
        mandatory_closure_bytes=sum(m.mandatory_closure_bytes for m in metrics),
        total_corpus_nodes=sum(m.total_corpus_nodes for m in metrics),
        total_corpus_bytes=sum(m.total_corpus_bytes for m in metrics),
        cache_lookups=sum(m.cache_lookups for m in metrics),
        cache_hits=sum(m.cache_hits for m in metrics),
        cache_reused_bytes=sum(m.cache_reused_bytes for m in metrics),
        first_valid_plans=sum(bool(m.first_valid_plan) for m in metrics),
        retries=sum(m.retries for m in metrics),
        proof_cost=sum(m.proof_cost for m in metrics),
        validation_cost=sum(m.validation_cost for m in metrics),
        invalidation_false_positives=invalidation_fp,
        invalidation_false_negatives=invalidation_fn,
        closure_token_correlation_millionths=closure_tokens,
        corpus_token_correlation_millionths=corpus_tokens,
        scale_dimensions_passed=tuple(sorted(dimensions_passed)),
        adversarial_fixtures_passed=tuple(sorted(fixture_passed)),
        context_scaling_passed=context_passed,
        cache_reuse_passed=cache_passed,
        invalidation_precision_passed=invalidation_passed,
        effect_parity_passed=effect_parity,
        terminal_parity_passed=terminal_parity,
        deterministic_degraded_passed=degraded_passed,
        lazy_discovery_passed=lazy_passed,
        passed=passed,
        failure_codes=tuple(sorted(set(failures))),
    )


def build_proof_dependency_scaling_report(
    receipts: Sequence[DecisionRuntimeProducerReceipt]
    | DecisionRuntimeBenchmark,
) -> ProofDependencyScalingReport:
    benchmark = (
        receipts
        if isinstance(receipts, DecisionRuntimeBenchmark)
        else DecisionRuntimeBenchmark(tuple(receipts))
    )
    return recompute_proof_dependency_scaling(benchmark)


def verify_proof_dependency_scaling_report(
    report: ProofDependencyScalingReport,
    benchmark: DecisionRuntimeBenchmark,
) -> bool:
    if not isinstance(report, ProofDependencyScalingReport):
        return False
    replayed = recompute_proof_dependency_scaling(benchmark)
    return _canonical_bytes(report.to_dict()) == _canonical_bytes(
        replayed.to_dict()
    )


def producer_receipt_from_records(
    *,
    identity: FrozenDecisionIdentity,
    path: DecisionRuntimePath | str,
    scale: CorpusScale,
    context_receipt: Any,
    runtime_receipt: Any,
    effect_receipt: Any,
    cache_receipt: Any,
    invalidation_receipt: Any,
    plan_receipt: Any,
    proof_receipt: Any,
    validation_receipt: Any,
    total_corpus_nodes: int,
    total_corpus_bytes: int,
    source_receipt_ids: Sequence[str],
    **options: Any,
) -> DecisionRuntimeProducerReceipt:
    """Join typed producer records without accepting caller aggregate metrics.

    The adapter intentionally uses a small structural protocol so generation-1
    and generation-2 receipt implementations can participate without imports.
    Missing fields fail closed.
    """

    def get(obj: Any, *names: str, default: Any = None) -> Any:
        for name in names:
            if isinstance(obj, Mapping) and name in obj:
                return obj[name]
            if hasattr(obj, name):
                return getattr(obj, name)
        if default is not None:
            return default
        raise DecisionRuntimeBenchmarkError(
            "producer receipt missing " + "/".join(names)
        )

    declared = get(effect_receipt, "declared_effect_ids", "expected_effect_ids")
    observed = get(effect_receipt, "observed_effect_ids")
    invalidated = set(get(invalidation_receipt, "invalidated_ids", default=()))
    expected = set(
        get(invalidation_receipt, "expected_invalidated_ids", default=())
    )
    metrics = DecisionRuntimeMetrics(
        provider_input_tokens=get(
            context_receipt, "provider_input_tokens", "complete_input_tokens"
        ),
        provider_output_tokens=get(
            runtime_receipt, "provider_output_tokens", default=0
        ),
        provider_reused_input_tokens=get(
            cache_receipt, "reused_input_tokens", default=0
        ),
        mandatory_closure_nodes=get(
            context_receipt, "mandatory_closure_nodes", "mandatory_node_count"
        ),
        mandatory_closure_bytes=get(
            context_receipt, "mandatory_closure_bytes", "mandatory_bytes"
        ),
        total_corpus_nodes=total_corpus_nodes,
        total_corpus_bytes=total_corpus_bytes,
        cache_lookups=get(cache_receipt, "lookup_count", default=1),
        cache_hits=get(cache_receipt, "hit_count", default=0),
        cache_reused_bytes=get(cache_receipt, "reused_bytes", default=0),
        invalidation_expected=len(expected),
        invalidation_actual=len(invalidated),
        invalidation_true_positive=len(expected.intersection(invalidated)),
        first_valid_plan=bool(
            get(plan_receipt, "first_valid_plan", "admitted", default=False)
        ),
        retries=get(context_receipt, "retry_count", default=0),
        proof_cost=get(proof_receipt, "cost", "latency_ms", default=0),
        validation_cost=get(
            validation_receipt, "cost", "latency_ms", default=0
        ),
        declared_effect_ids=tuple(declared),
        observed_effect_ids=tuple(observed),
        terminal_result=get(runtime_receipt, "terminal_result", "outcome"),
        index_metadata_bytes=get(
            context_receipt, "index_metadata_bytes", default=0
        ),
    )
    return DecisionRuntimeProducerReceipt(
        identity=identity,
        path=path,
        scale=scale,
        metrics=metrics,
        mandatory_closure_id=get(
            context_receipt, "mandatory_closure_id", "closure_id"
        ),
        context_id=get(context_receipt, "context_id"),
        source_receipt_ids=tuple(source_receipt_ids),
        **options,
    )


def build_frozen_decision_runtime_benchmark(
    *,
    observation_label: str = "qualification",
    tree_id: str = "sha256:frozen-proof-runtime-tree",
) -> DecisionRuntimeBenchmark:
    """Build the deterministic closed smoke population.

    This is a local conformance fixture, not production evidence.  Production
    promotion must replace every source ID with receipts from both live paths.
    """

    label = _code(observation_label, "observation_label")
    identity = FrozenDecisionIdentity(
        repository_id="repository:proof-runtime-benchmark@1",
        tree_id=tree_id,
        decision_request_id="decision:frozen-context-scaling@1",
        objective_id="ASI-G360",
        objective_revision="sha256:frozen-objective",
        policy_id="policy:proof-runtime-rollout@1",
        policy_revision="sha256:frozen-policy",
        capability_id="capability:proof-runtime-local@1",
        capability_revision="sha256:frozen-capability",
        provider_id="provider:deterministic-local@1",
        provider_revision="sha256:frozen-provider",
        tokenizer_id="tokenizer:deterministic-bytes@1",
        tokenizer_revision="sha256:frozen-tokenizer",
        partition_id="partition:frozen-proof-runtime@1",
    )
    effects = ("effect:emit-decision-receipt",)

    def measurement(
        *,
        path: DecisionRuntimePath,
        scale: CorpusScale,
        closure_nodes: int,
        closure_bytes: int,
        proof_tokens: int,
    ) -> DecisionRuntimeMetrics:
        intervention = scale.intervention
        factor = scale.by_kind[intervention] if intervention else 1
        corpus_nodes = 100 * factor if intervention else 100
        corpus_bytes = 10_000 * factor if intervention else 10_000
        input_tokens = (
            proof_tokens + 40 + (factor * 25 if intervention else 0)
            if path is DecisionRuntimePath.CURRENT
            else proof_tokens
        )
        return DecisionRuntimeMetrics(
            provider_input_tokens=input_tokens,
            provider_output_tokens=20,
            provider_reused_input_tokens=60,
            mandatory_closure_nodes=closure_nodes,
            mandatory_closure_bytes=closure_bytes,
            total_corpus_nodes=corpus_nodes,
            total_corpus_bytes=corpus_bytes,
            cache_lookups=1,
            cache_hits=1,
            cache_reused_bytes=closure_bytes,
            invalidation_expected=1,
            invalidation_actual=1,
            invalidation_true_positive=1,
            first_valid_plan=True,
            retries=0,
            proof_cost=10,
            validation_cost=10,
            declared_effect_ids=effects,
            observed_effect_ids=effects,
            terminal_result="accepted",
            index_metadata_bytes=64 if intervention else 32,
        )

    scales = [CorpusScale()]
    for kind in REQUIRED_IRRELEVANT_CORPORA:
        values = {name: 1 for name in CorpusScale.__dataclass_fields__}
        field_name = {
            IrrelevantCorpus.LEGAL: "legal_corpus",
            IrrelevantCorpus.CODEBASE: "codebase",
            IrrelevantCorpus.SKILLCENTER_ROWS: "skillcenter_rows",
            IrrelevantCorpus.SKILLCENTER_GRAPH: "skillcenter_graph",
            IrrelevantCorpus.CONVERSATION: "conversation_history",
        }[kind]
        values[field_name] = MINIMUM_IRRELEVANT_SCALE_FACTOR
        scales.append(CorpusScale(**values))

    second_identity = FrozenDecisionIdentity(
        **{
            **identity.to_dict(),
            "decision_request_id": "decision:frozen-context-scaling-large@1",
        }
    )
    decisions = (
        (identity, 8, 800, 80),
        (second_identity, 16, 1_600, 160),
    )
    receipts: list[DecisionRuntimeProducerReceipt] = []
    for frozen, closure_nodes, closure_bytes, proof_tokens in decisions:
        closure_id = _identity(
            {
                "decision_request_id": frozen.decision_request_id,
                "nodes": closure_nodes,
                "bytes": closure_bytes,
            }
        )
        for path in DecisionRuntimePath:
            for scale in scales:
                receipts.append(
                    DecisionRuntimeProducerReceipt(
                        identity=frozen,
                        path=path,
                        scale=scale,
                        metrics=measurement(
                            path=path,
                            scale=scale,
                            closure_nodes=closure_nodes,
                            closure_bytes=closure_bytes,
                            proof_tokens=proof_tokens,
                        ),
                        mandatory_closure_id=closure_id,
                        context_id=_identity(
                            {
                                "label": label,
                                "decision": frozen.decision_request_id,
                                "path": path.value,
                                "scale": scale.to_dict(),
                                "closure": closure_id,
                            }
                        ),
                        source_receipt_ids=(
                            _identity(
                                {
                                    "producer": label,
                                    "decision": frozen.decision_request_id,
                                    "path": path.value,
                                    "scale": scale.to_dict(),
                                }
                            ),
                        ),
                    )
                )
    base_metrics = measurement(
        path=DecisionRuntimePath.PROOF_DIRECTED,
        scale=CorpusScale(),
        closure_nodes=8,
        closure_bytes=800,
        proof_tokens=80,
    )
    closure_id = _identity(
        {
            "decision_request_id": identity.decision_request_id,
            "nodes": 8,
            "bytes": 800,
        }
    )
    for fixture in REQUIRED_ADVERSARIAL_FIXTURES:
        degraded = fixture is AdversarialFixture.RECOVERY
        adversarial_metrics = replace(
            base_metrics,
            first_valid_plan=False,
            declared_effect_ids=(),
            observed_effect_ids=(),
            terminal_result="rejected",
        )
        receipts.append(
            DecisionRuntimeProducerReceipt(
                identity=identity,
                path=DecisionRuntimePath.PROOF_DIRECTED,
                scale=CorpusScale(),
                metrics=adversarial_metrics,
                mandatory_closure_id=closure_id,
                context_id=_identity(
                    {"label": label, "adversarial": fixture.value}
                ),
                source_receipt_ids=(
                    _identity(
                        {"producer": label, "adversarial": fixture.value}
                    ),
                ),
                adversarial_fixture=fixture,
                escape_count=0,
                degraded_local=degraded,
                deterministic_replay_id=(
                    _identity(
                        {"local-replay": fixture.value, "result": "fail-closed"}
                    )
                    if degraded
                    else ""
                ),
                lazy_discovery=True,
            )
        )
    return DecisionRuntimeBenchmark(tuple(receipts))


__all__ = (
    "AdversarialFixture",
    "CorpusScale",
    "DECISION_RUNTIME_BENCHMARK_SCHEMA",
    "DECISION_RUNTIME_BENCHMARK_VERSION",
    "DECISION_RUNTIME_PRODUCER_RECEIPT_SCHEMA",
    "DecisionRuntimeBenchmark",
    "DecisionRuntimeBenchmarkError",
    "DecisionRuntimeMetrics",
    "DecisionRuntimePath",
    "DecisionRuntimeProducerReceipt",
    "FrozenDecisionIdentity",
    "IrrelevantCorpus",
    "MINIMUM_IRRELEVANT_SCALE_FACTOR",
    "PROOF_DEPENDENCY_SCALING_REPORT_SCHEMA",
    "PROOF_DEPENDENCY_SCALING_REQUIREMENT_ID",
    "ProofDependencyScalingReport",
    "REQUIRED_ADVERSARIAL_FIXTURES",
    "REQUIRED_IRRELEVANT_CORPORA",
    "build_proof_dependency_scaling_report",
    "build_frozen_decision_runtime_benchmark",
    "producer_receipt_from_records",
    "recompute_proof_dependency_scaling",
    "verify_proof_dependency_scaling_report",
)
