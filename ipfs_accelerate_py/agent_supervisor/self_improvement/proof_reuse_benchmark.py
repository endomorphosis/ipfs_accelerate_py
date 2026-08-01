"""Shadow and warm proof-reuse benchmark (PTR-100).

This module is a measurement boundary, not a rollout policy.  It freezes a
controlled eligible/ineligible fixture population, drives the authoritative
lookup path under ``off``, ``shadow``, cold ``readwrite``, warm ``read``, and
forced-rerun scenarios, and recomputes a compact receipt.

Performance never relaxes authority.  Authoritative skips require an exact
verified candidate; false admissions are counted as hard gate failures.  Timing
fields use a deterministic cost model so saved wall time and exclusions are
reproducible bit-for-bit from the fixture population and decision outcomes.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
    CertificateAuthority,
    PhaseOutcome,
    ProofBackendMode,
    ReuseAction,
    ReuseDecision,
    ReuseReasonCode,
    TestExecutionKey,
    TestLocatorKey,
    TestPassReceipt,
    TestProofCertificate,
    reuse_run,
)
from ipfs_accelerate_py.agent_supervisor.proof.test_proof_cache import TestProofCache
from ipfs_accelerate_py.testing.proof_reuse.config import ProofReuseMode
from ipfs_accelerate_py.testing.proof_reuse.lookup import ProofReuseLookup
from ipfs_accelerate_py.testing.proof_reuse.reporting import (
    PROOF_REUSE_METRICS_INTERFACE,
    ProofReuseSessionMetrics,
)

PROOF_REUSE_BENCHMARK_INTERFACE: Final = "ProofReuseBenchmark@1"
BENCHMARK_RECEIPT_INTERFACE: Final = "BenchmarkReceipt@1"
PROOF_REUSE_BENCHMARK_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/proof-reuse-benchmark-receipt@1"
)
PROOF_REUSE_BENCHMARK_CORPUS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/proof-reuse-benchmark-corpus@1"
)
PROOF_REUSE_BENCHMARK_REQUIREMENT_ID: Final = "ptr/shadow-benchmark@1"
PROOF_REUSE_BENCHMARK_VERSION: Final = 1
PROOF_REUSE_BENCHMARK_CORPUS_VERSION: Final = "proof-reuse-benchmark@1"

# Acceptance thresholds (basis points where noted).
MIN_WARM_SKIP_BPS: Final = 8_000  # 80% of eligible warm population
# Miss overhead must stay under 20% of a single eligible execution cost unit.
MAX_MISS_OVERHEAD_BPS: Final = 2_000
# Verification must be strictly cheaper than execution on the warm path.
MIN_VERIFY_VS_EXECUTE_SAVINGS_BPS: Final = 1  # any positive savings

# Deterministic cost model (virtual milliseconds).  Wall-clock is intentionally
# not used so receipts remain reproducible across hosts.
DEFAULT_EXECUTE_COST_MS: Final = 50
DEFAULT_VERIFY_COST_MS: Final = 2
DEFAULT_MISS_LOOKUP_COST_MS: Final = 1
DEFAULT_COLLECTION_COST_MS: Final = 1
DEFAULT_BYTES_PER_CANDIDATE: Final = 512

NOW_MS: Final = 10_000
CREATED_AT_MS: Final = 9_000
EXPIRES_AT_MS: Final = 11_000

MAX_FIXTURES: Final = 4_096
MAX_COUNTER: Final = 10**15
MAX_RECEIPT_BYTES: Final = 262_144
MAX_REASON_LENGTH: Final = 96

_CONTENT_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_REASON = re.compile(r"^[a-z0-9_.:@/+-]{1,96}$")
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_.:@/+-]{0,191}$")

_DEFAULT_POLICY: Final[dict[str, Any]] = {
    "policy_cid": "cid:policy",
    "statement_cid": "cid:statement",
    "circuit_cid": "cid:circuit",
    "verifying_key_cid": "cid:verifying-key",
    "proof_system_id": "groth16",
    "trusted_issuer_ids": ("issuer:trusted",),
    "allowed_epochs": ("epoch:7",),
    "revoked_issuer_ids": (),
    "revoked_receipt_cids": (),
    "revoked_certificate_cids": (),
}


class ProofReuseBenchmarkError(ValueError):
    """Benchmark corpus, observation, or receipt is malformed."""


class BenchmarkScenario(str, Enum):
    """Closed scenario vocabulary compared by the harness."""

    OFF = "off"
    SHADOW = "shadow"
    COLD_READWRITE = "cold_readwrite"
    WARM_READ = "warm_read"
    FORCED_RERUN = "forced_rerun"


class FixtureClass(str, Enum):
    """Controlled population classes for the frozen benchmark corpus."""

    ELIGIBLE_WARM = "eligible_warm"
    INELIGIBLE = "ineligible"
    MISS = "miss"
    MUTATED = "mutated"
    EXCLUDED = "excluded"


class GroundTruth(str, Enum):
    """Whether an authoritative warm-path skip is correct for the fixture."""

    SHOULD_SKIP = "should_skip"
    SHOULD_RUN = "should_run"


class GateName(str, Enum):
    FALSE_ADMISSIONS_ZERO = "false_admissions_zero"
    WARM_SKIP_THRESHOLD = "warm_skip_threshold"
    VERIFY_CHEAPER_THAN_EXECUTE = "verify_cheaper_than_execute"
    MISS_OVERHEAD_BOUNDED = "miss_overhead_bounded"
    RECEIPT_REPRODUCIBLE = "receipt_reproducible"


REQUIRED_SCENARIOS: Final[tuple[BenchmarkScenario, ...]] = tuple(BenchmarkScenario)
REQUIRED_GATES: Final[tuple[GateName, ...]] = tuple(GateName)

# Explicit warm fixture population size; at least 80% must verify-and-skip.
DEFAULT_ELIGIBLE_WARM_COUNT: Final = 20
DEFAULT_INELIGIBLE_COUNT: Final = 4
DEFAULT_MISS_COUNT: Final = 4
DEFAULT_MUTATED_COUNT: Final = 4
DEFAULT_EXCLUDED_COUNT: Final = 4


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
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
        raise ProofReuseBenchmarkError(
            "benchmark data must be canonical JSON"
        ) from exc


def _content_id(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _text(value: Any, name: str, *, maximum: int = 192) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ProofReuseBenchmarkError(f"{name} must be non-empty text")
    if "\x00" in value or len(value.encode("utf-8")) > maximum:
        raise ProofReuseBenchmarkError(f"{name} is unsafe or too large")
    return value


def _safe_id(value: Any, name: str) -> str:
    result = _text(value, name, maximum=192)
    if not _SAFE_ID.fullmatch(result):
        raise ProofReuseBenchmarkError(f"{name} must be a compact identifier")
    return result


def _integer(
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
        raise ProofReuseBenchmarkError(
            f"{name} must be an integer from {minimum} through {maximum}"
        )
    return value


def _enum(value: Any, kind: type[Enum], name: str) -> Any:
    if isinstance(value, kind):
        return value
    try:
        return kind(str(getattr(value, "value", value)))
    except (TypeError, ValueError) as exc:
        raise ProofReuseBenchmarkError(f"invalid {name}") from exc


def _bps(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        return 0
    return (numerator * 10_000) // denominator


def _reason_code(value: Any) -> str:
    if isinstance(value, Enum):
        value = value.value
    text = str(value or "unknown").strip().lower()
    if not text or len(text) > MAX_REASON_LENGTH:
        return "unknown"
    if not _SAFE_REASON.fullmatch(text):
        return "unknown"
    return text


# ---------------------------------------------------------------------------
# Fixture identity builders (shared with lookup contracts)
# ---------------------------------------------------------------------------


def _locator(fixture_id: str) -> TestLocatorKey:
    return TestLocatorKey(
        repository_id="repository:proof-reuse-benchmark",
        package_identity="package:proof-reuse-benchmark",
        node_id=f"test/benchmark/{fixture_id}.py::test_{fixture_id}",
    )


def _execution_key(
    locator: TestLocatorKey,
    *,
    static_trace: str = "cid:static-trace",
    runtime_trace: str = "cid:runtime-trace",
) -> TestExecutionKey:
    return TestExecutionKey(
        locator_cid=locator.locator_id,
        repository_forest_cid="cid:repository-forest",
        static_trace_root_cid=static_trace,
        runtime_trace_root_cid=runtime_trace,
        runtime_completeness_policy="complete-v1",
        policy_cid="cid:policy",
    )


def _receipt(
    locator: TestLocatorKey,
    execution_key: TestExecutionKey,
) -> TestPassReceipt:
    return TestPassReceipt(
        execution_key_cid=execution_key.execution_key_id,
        locator_cid=locator.locator_id,
        setup_outcome=PhaseOutcome.PASS,
        call_outcome=PhaseOutcome.PASS,
        teardown_outcome=PhaseOutcome.PASS,
        static_trace_root_cid=execution_key.static_trace_root_cid,
        runtime_trace_root_cid=execution_key.runtime_trace_root_cid,
        completeness_receipt_cid="cid:completeness-receipt",
        dependency_forest_cid=execution_key.repository_forest_cid,
        issuer_key_id="key:issuer",
        policy_cid=execution_key.policy_cid,
    )


def _certificate(
    receipt: TestPassReceipt,
    execution_key: TestExecutionKey,
) -> TestProofCertificate:
    return TestProofCertificate(
        receipt_cid=receipt.receipt_id,
        execution_key_cid=execution_key.execution_key_id,
        policy_cid=execution_key.policy_cid,
        statement_cid="cid:statement",
        circuit_cid="cid:circuit",
        verifying_key_cid="cid:verifying-key",
        proof_artifact_cid="cid:proof",
        issuer_id="issuer:trusted",
        epoch="epoch:7",
        proof_system_id="groth16",
        backend_mode=ProofBackendMode.CRYPTOGRAPHIC,
        authority=CertificateAuthority.AUTHORITATIVE,
        public_inputs={
            "receipt_cid": receipt.receipt_id,
            "execution_key_cid": execution_key.execution_key_id,
            "policy_cid": execution_key.policy_cid,
            "statement_cid": "cid:statement",
            "circuit_cid": "cid:circuit",
            "verifying_key_cid": "cid:verifying-key",
            "proof_system_id": "groth16",
            "issuer_id": "issuer:trusted",
            "issuer_key_id": "key:issuer",
            "epoch": "epoch:7",
            "setup_outcome": "pass",
            "call_outcome": "pass",
            "teardown_outcome": "pass",
        },
    )


class _AlwaysVerify:
    """Deterministic local verifier used only by the benchmark harness."""

    def verify(self, *_args: Any, **_kwargs: Any) -> bool:
        return True

    def as_cache_verifier(self) -> "_AlwaysVerify":
        return self


class _CandidateStore:
    """In-memory locator → candidates map for controlled scenarios."""

    def __init__(self, mapping: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
        self._mapping = {
            str(key): tuple(dict(item) for item in value)
            for key, value in mapping.items()
        }

    def lookup(self, locator_cid: str, *, max_candidates: int) -> tuple[dict[str, Any], ...]:
        candidates = self._mapping.get(str(locator_cid), ())
        return tuple(candidates[: max(0, int(max_candidates))])


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkFixture:
    """One controlled population member with ground truth and cost units."""

    fixture_id: str
    fixture_class: FixtureClass
    ground_truth: GroundTruth
    execute_cost_ms: int = DEFAULT_EXECUTE_COST_MS
    verify_cost_ms: int = DEFAULT_VERIFY_COST_MS
    miss_lookup_cost_ms: int = DEFAULT_MISS_LOOKUP_COST_MS
    collection_cost_ms: int = DEFAULT_COLLECTION_COST_MS
    exclusion_reason: str = ""
    bytes_read: int = DEFAULT_BYTES_PER_CANDIDATE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "fixture_id", _safe_id(self.fixture_id, "fixture_id")
        )
        object.__setattr__(
            self,
            "fixture_class",
            _enum(self.fixture_class, FixtureClass, "fixture_class"),
        )
        object.__setattr__(
            self,
            "ground_truth",
            _enum(self.ground_truth, GroundTruth, "ground_truth"),
        )
        for name in (
            "execute_cost_ms",
            "verify_cost_ms",
            "miss_lookup_cost_ms",
            "collection_cost_ms",
            "bytes_read",
        ):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name, minimum=0)
            )
        reason = self.exclusion_reason
        if reason:
            reason = _reason_code(reason)
            if reason == "unknown" and self.exclusion_reason:
                raise ProofReuseBenchmarkError("exclusion_reason is unsafe")
            object.__setattr__(self, "exclusion_reason", reason)
        if (
            self.fixture_class is FixtureClass.ELIGIBLE_WARM
            and self.ground_truth is not GroundTruth.SHOULD_SKIP
        ):
            raise ProofReuseBenchmarkError(
                "eligible_warm fixtures must have should_skip ground truth"
            )
        if (
            self.fixture_class is not FixtureClass.ELIGIBLE_WARM
            and self.ground_truth is GroundTruth.SHOULD_SKIP
        ):
            raise ProofReuseBenchmarkError(
                "only eligible_warm fixtures may have should_skip ground truth"
            )
        if self.fixture_class is FixtureClass.EXCLUDED and not self.exclusion_reason:
            raise ProofReuseBenchmarkError(
                "excluded fixtures require an exclusion_reason"
            )

    @property
    def is_explicitly_eligible_warm(self) -> bool:
        return self.fixture_class is FixtureClass.ELIGIBLE_WARM

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "fixture_class": self.fixture_class.value,
            "ground_truth": self.ground_truth.value,
            "execute_cost_ms": self.execute_cost_ms,
            "verify_cost_ms": self.verify_cost_ms,
            "miss_lookup_cost_ms": self.miss_lookup_cost_ms,
            "collection_cost_ms": self.collection_cost_ms,
            "exclusion_reason": self.exclusion_reason,
            "bytes_read": self.bytes_read,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BenchmarkFixture":
        if not isinstance(value, Mapping):
            raise ProofReuseBenchmarkError("fixture must be a mapping")
        return cls(
            fixture_id=value.get("fixture_id", ""),
            fixture_class=value.get("fixture_class", ""),
            ground_truth=value.get("ground_truth", ""),
            execute_cost_ms=value.get("execute_cost_ms", DEFAULT_EXECUTE_COST_MS),
            verify_cost_ms=value.get("verify_cost_ms", DEFAULT_VERIFY_COST_MS),
            miss_lookup_cost_ms=value.get(
                "miss_lookup_cost_ms", DEFAULT_MISS_LOOKUP_COST_MS
            ),
            collection_cost_ms=value.get(
                "collection_cost_ms", DEFAULT_COLLECTION_COST_MS
            ),
            exclusion_reason=str(value.get("exclusion_reason", "") or ""),
            bytes_read=value.get("bytes_read", DEFAULT_BYTES_PER_CANDIDATE),
        )


@dataclass(frozen=True)
class BenchmarkCorpus:
    """Frozen controlled population for the proof-reuse benchmark."""

    fixtures: tuple[BenchmarkFixture, ...]
    corpus_version: str = PROOF_REUSE_BENCHMARK_CORPUS_VERSION
    requirement_id: str = PROOF_REUSE_BENCHMARK_REQUIREMENT_ID

    def __post_init__(self) -> None:
        if not isinstance(self.fixtures, tuple):
            object.__setattr__(self, "fixtures", tuple(self.fixtures))
        if len(self.fixtures) == 0 or len(self.fixtures) > MAX_FIXTURES:
            raise ProofReuseBenchmarkError("corpus fixture count is out of bounds")
        ids = [item.fixture_id for item in self.fixtures]
        if len(ids) != len(set(ids)):
            raise ProofReuseBenchmarkError("fixture ids must be unique")
        if not any(item.is_explicitly_eligible_warm for item in self.fixtures):
            raise ProofReuseBenchmarkError(
                "corpus requires at least one eligible_warm fixture"
            )
        object.__setattr__(
            self, "corpus_version", _safe_id(self.corpus_version, "corpus_version")
        )
        object.__setattr__(
            self, "requirement_id", _safe_id(self.requirement_id, "requirement_id")
        )

    @property
    def corpus_id(self) -> str:
        return _content_id(self.to_dict())

    @property
    def eligible_warm(self) -> tuple[BenchmarkFixture, ...]:
        return tuple(
            item for item in self.fixtures if item.is_explicitly_eligible_warm
        )

    @property
    def exclusions(self) -> Mapping[str, int]:
        counts: Counter[str] = Counter()
        for item in self.fixtures:
            if item.fixture_class is FixtureClass.EXCLUDED and item.exclusion_reason:
                counts[item.exclusion_reason] += 1
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROOF_REUSE_BENCHMARK_CORPUS_SCHEMA,
            "corpus_version": self.corpus_version,
            "requirement_id": self.requirement_id,
            "fixtures": [item.to_dict() for item in self.fixtures],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BenchmarkCorpus":
        if not isinstance(value, Mapping):
            raise ProofReuseBenchmarkError("corpus must be a mapping")
        if value.get("schema") not in (None, PROOF_REUSE_BENCHMARK_CORPUS_SCHEMA):
            raise ProofReuseBenchmarkError("corpus schema mismatch")
        raw = value.get("fixtures")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ProofReuseBenchmarkError("corpus fixtures must be a sequence")
        return cls(
            fixtures=tuple(BenchmarkFixture.from_dict(item) for item in raw),
            corpus_version=str(
                value.get("corpus_version", PROOF_REUSE_BENCHMARK_CORPUS_VERSION)
            ),
            requirement_id=str(
                value.get("requirement_id", PROOF_REUSE_BENCHMARK_REQUIREMENT_ID)
            ),
        )


def build_default_benchmark_corpus() -> BenchmarkCorpus:
    """Build the reviewed default shadow/warm fixture population."""

    fixtures: list[BenchmarkFixture] = []
    for index in range(DEFAULT_ELIGIBLE_WARM_COUNT):
        fixtures.append(
            BenchmarkFixture(
                fixture_id=f"warm-{index:02d}",
                fixture_class=FixtureClass.ELIGIBLE_WARM,
                ground_truth=GroundTruth.SHOULD_SKIP,
            )
        )
    for index in range(DEFAULT_INELIGIBLE_COUNT):
        fixtures.append(
            BenchmarkFixture(
                fixture_id=f"ineligible-{index:02d}",
                fixture_class=FixtureClass.INELIGIBLE,
                ground_truth=GroundTruth.SHOULD_RUN,
            )
        )
    for index in range(DEFAULT_MISS_COUNT):
        fixtures.append(
            BenchmarkFixture(
                fixture_id=f"miss-{index:02d}",
                fixture_class=FixtureClass.MISS,
                ground_truth=GroundTruth.SHOULD_RUN,
            )
        )
    for index in range(DEFAULT_MUTATED_COUNT):
        fixtures.append(
            BenchmarkFixture(
                fixture_id=f"mutated-{index:02d}",
                fixture_class=FixtureClass.MUTATED,
                ground_truth=GroundTruth.SHOULD_RUN,
            )
        )
    exclusion_reasons = (
        "non_reusable",
        "eligibility_denied",
        "reuse_disabled",
        "unsupported",
    )
    for index in range(DEFAULT_EXCLUDED_COUNT):
        fixtures.append(
            BenchmarkFixture(
                fixture_id=f"excluded-{index:02d}",
                fixture_class=FixtureClass.EXCLUDED,
                ground_truth=GroundTruth.SHOULD_RUN,
                exclusion_reason=exclusion_reasons[index % len(exclusion_reasons)],
            )
        )
    return BenchmarkCorpus(fixtures=tuple(fixtures))


# ---------------------------------------------------------------------------
# Scenario execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseObservation:
    """One fixture under one scenario."""

    fixture_id: str
    fixture_class: FixtureClass
    scenario: BenchmarkScenario
    action: str
    reason_code: str
    predicted: bool
    verified: bool
    skipped: bool
    executed: bool
    false_admission: bool
    collection_latency_ms: int
    lookup_latency_ms: int
    verify_latency_ms: int
    execution_latency_ms: int
    bytes_read: int
    bytes_written: int
    saved_wall_time_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "fixture_class": self.fixture_class.value,
            "scenario": self.scenario.value,
            "action": self.action,
            "reason_code": self.reason_code,
            "predicted": self.predicted,
            "verified": self.verified,
            "skipped": self.skipped,
            "executed": self.executed,
            "false_admission": self.false_admission,
            "collection_latency_ms": self.collection_latency_ms,
            "lookup_latency_ms": self.lookup_latency_ms,
            "verify_latency_ms": self.verify_latency_ms,
            "execution_latency_ms": self.execution_latency_ms,
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
            "saved_wall_time_ms": self.saved_wall_time_ms,
        }


@dataclass(frozen=True)
class ScenarioSummary:
    """Aggregate counters for one scenario."""

    scenario: BenchmarkScenario
    mode: str
    collected: int
    eligible: int
    predicted: int
    verified: int
    skipped: int
    executed: int
    false_admissions: int
    collection_latency_ms: int
    lookup_latency_ms: int
    verify_latency_ms: int
    execution_latency_ms: int
    miss_overhead_ms: int
    saved_wall_time_ms: int
    bytes_read: int
    bytes_written: int
    reason_codes: Mapping[str, int]
    metrics: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario.value,
            "mode": self.mode,
            "collected": self.collected,
            "eligible": self.eligible,
            "predicted": self.predicted,
            "verified": self.verified,
            "skipped": self.skipped,
            "executed": self.executed,
            "false_admissions": self.false_admissions,
            "collection_latency_ms": self.collection_latency_ms,
            "lookup_latency_ms": self.lookup_latency_ms,
            "verify_latency_ms": self.verify_latency_ms,
            "execution_latency_ms": self.execution_latency_ms,
            "miss_overhead_ms": self.miss_overhead_ms,
            "saved_wall_time_ms": self.saved_wall_time_ms,
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
            "reason_codes": dict(sorted(self.reason_codes.items())),
            "metrics": dict(self.metrics),
        }


def _scenario_mode(scenario: BenchmarkScenario) -> ProofReuseMode:
    if scenario is BenchmarkScenario.OFF:
        return ProofReuseMode.OFF
    if scenario is BenchmarkScenario.SHADOW:
        return ProofReuseMode.SHADOW
    if scenario is BenchmarkScenario.COLD_READWRITE:
        return ProofReuseMode.READWRITE
    if scenario is BenchmarkScenario.WARM_READ:
        return ProofReuseMode.READ
    if scenario is BenchmarkScenario.FORCED_RERUN:
        return ProofReuseMode.READ
    raise ProofReuseBenchmarkError(f"unknown scenario {scenario!r}")


def _build_warm_candidate(fixture_id: str) -> tuple[
    TestLocatorKey,
    TestExecutionKey,
    dict[str, Any],
]:
    locator = _locator(fixture_id)
    execution_key = _execution_key(locator)
    receipt = _receipt(locator, execution_key)
    certificate = _certificate(receipt, execution_key)
    candidate = TestProofCache.candidate(
        receipt,
        certificate,
        created_at_ms=CREATED_AT_MS,
        expires_at_ms=EXPIRES_AT_MS,
    )
    return locator, execution_key, candidate


def _build_mutated_candidate(fixture_id: str) -> tuple[
    TestLocatorKey,
    TestExecutionKey,
    dict[str, Any],
]:
    """Candidate is bound to a stale execution key relative to current identity."""

    locator = _locator(fixture_id)
    current_key = _execution_key(
        locator,
        static_trace="cid:static-trace-current",
        runtime_trace="cid:runtime-trace-current",
    )
    stale_key = _execution_key(
        locator,
        static_trace="cid:static-trace-stale",
        runtime_trace="cid:runtime-trace-stale",
    )
    receipt = _receipt(locator, stale_key)
    certificate = _certificate(receipt, stale_key)
    candidate = TestProofCache.candidate(
        receipt,
        certificate,
        created_at_ms=CREATED_AT_MS,
        expires_at_ms=EXPIRES_AT_MS,
    )
    return locator, current_key, candidate


def _lookup_for_scenario(
    scenario: BenchmarkScenario,
    fixtures: Sequence[BenchmarkFixture],
) -> tuple[ProofReuseLookup | None, dict[str, tuple[TestLocatorKey, TestExecutionKey]]]:
    """Construct a scenario-scoped lookup service and per-fixture identities."""

    identities: dict[str, tuple[TestLocatorKey, TestExecutionKey]] = {}
    store_map: dict[str, list[dict[str, Any]]] = {}

    warm_store = scenario in (
        BenchmarkScenario.SHADOW,
        BenchmarkScenario.WARM_READ,
        BenchmarkScenario.FORCED_RERUN,
    )

    for fixture in fixtures:
        if fixture.fixture_class is FixtureClass.MUTATED:
            locator, execution_key, candidate = _build_mutated_candidate(
                fixture.fixture_id
            )
            identities[fixture.fixture_id] = (locator, execution_key)
            if warm_store:
                store_map.setdefault(locator.locator_id, []).append(candidate)
            continue

        locator, execution_key, candidate = _build_warm_candidate(fixture.fixture_id)
        identities[fixture.fixture_id] = (locator, execution_key)

        if fixture.fixture_class is FixtureClass.MISS:
            continue
        if fixture.fixture_class is FixtureClass.INELIGIBLE:
            # Candidate may exist; eligibility denies before store use.
            if warm_store:
                store_map.setdefault(locator.locator_id, []).append(candidate)
            continue
        if fixture.fixture_class is FixtureClass.EXCLUDED:
            continue
        if fixture.fixture_class is FixtureClass.ELIGIBLE_WARM and warm_store:
            store_map.setdefault(locator.locator_id, []).append(candidate)

    if scenario is BenchmarkScenario.OFF:
        return None, identities
    if scenario is BenchmarkScenario.COLD_READWRITE:
        lookup = ProofReuseLookup(
            _CandidateStore({}),
            _AlwaysVerify(),
            current_policy=_DEFAULT_POLICY,
            timeout_seconds=2.0,
        )
        return lookup, identities

    lookup = ProofReuseLookup(
        _CandidateStore(store_map),
        _AlwaysVerify(),
        current_policy=_DEFAULT_POLICY,
        timeout_seconds=2.0,
    )
    return lookup, identities


def _raw_lookup_decision(
    lookup: ProofReuseLookup | None,
    fixture: BenchmarkFixture,
    identities: Mapping[str, tuple[TestLocatorKey, TestExecutionKey]],
) -> ReuseDecision:
    locator, execution_key = identities[fixture.fixture_id]
    if lookup is None:
        return reuse_run(ReuseReasonCode.MODE_OFF)
    eligibility: Any
    if fixture.fixture_class is FixtureClass.INELIGIBLE:
        eligibility = False
    elif fixture.fixture_class is FixtureClass.EXCLUDED:
        eligibility = False
    else:
        eligibility = True
    return lookup.lookup(
        locator,
        execution_key,
        eligibility=eligibility,
        now_ms=NOW_MS,
    )


def _observe_case(
    fixture: BenchmarkFixture,
    scenario: BenchmarkScenario,
    lookup: ProofReuseLookup | None,
    identities: Mapping[str, tuple[TestLocatorKey, TestExecutionKey]],
) -> CaseObservation:
    mode = _scenario_mode(scenario)
    collection_ms = fixture.collection_cost_ms
    lookup_ms = 0
    verify_ms = 0
    execution_ms = 0
    bytes_read = 0
    bytes_written = 0
    predicted = False
    verified = False
    skipped = False
    executed = False
    false_admission = False
    action = ReuseAction.RUN.value
    reason = ReuseReasonCode.MODE_OFF.value

    if scenario is BenchmarkScenario.OFF or mode is ProofReuseMode.OFF:
        executed = True
        execution_ms = fixture.execute_cost_ms
        action = ReuseAction.RUN.value
        reason = ReuseReasonCode.MODE_OFF.value
    else:
        decision = _raw_lookup_decision(lookup, fixture, identities)
        action = decision.action.value
        reason = _reason_code(decision.reason_code)
        lookup_hit = decision.is_skip
        lookup_ms = (
            fixture.verify_cost_ms
            if lookup_hit
            else fixture.miss_lookup_cost_ms
        )
        if lookup_hit:
            bytes_read = fixture.bytes_read

        if scenario is BenchmarkScenario.SHADOW:
            # Shadow predicts and verifies but never authoritatively skips.
            if lookup_hit:
                predicted = True
                verified = True
                verify_ms = fixture.verify_cost_ms
                reason = "mode_shadow"
            executed = True
            execution_ms = fixture.execute_cost_ms
            action = ReuseAction.RUN.value
            if not lookup_hit and reason in ("", "unknown"):
                reason = _reason_code(decision.reason_code)
        elif scenario is BenchmarkScenario.COLD_READWRITE:
            # Empty cache: miss, execute, and (conceptually) write a receipt.
            executed = True
            execution_ms = fixture.execute_cost_ms
            bytes_written = fixture.bytes_read if fixture.is_explicitly_eligible_warm else 0
            action = ReuseAction.RUN.value
            reason = _reason_code(decision.reason_code) or "candidate_missing"
        elif scenario is BenchmarkScenario.FORCED_RERUN:
            if lookup_hit:
                predicted = True
                verified = True
                verify_ms = fixture.verify_cost_ms
            executed = True
            execution_ms = fixture.execute_cost_ms
            action = ReuseAction.RUN.value
            reason = "real_execution"
        elif scenario is BenchmarkScenario.WARM_READ:
            # READ (and READWRITE) may authoritatively skip after verify.
            may_skip = mode in (ProofReuseMode.READ, ProofReuseMode.READWRITE)
            if lookup_hit and may_skip:
                predicted = True
                verified = True
                skipped = True
                verify_ms = fixture.verify_cost_ms
                action = ReuseAction.SKIP.value
                reason = _reason_code(decision.reason_code) or "proof_cache_hit"
                if fixture.ground_truth is GroundTruth.SHOULD_RUN:
                    false_admission = True
            else:
                executed = True
                execution_ms = fixture.execute_cost_ms
                action = ReuseAction.RUN.value
                reason = _reason_code(decision.reason_code)
                # A should-skip fixture that failed to skip is a miss, not a
                # false admission.  Authority stays fail-closed.
        else:
            raise ProofReuseBenchmarkError(f"unhandled scenario {scenario!r}")

    saved = 0
    if skipped and not false_admission:
        saved = max(0, fixture.execute_cost_ms - verify_ms)

    return CaseObservation(
        fixture_id=fixture.fixture_id,
        fixture_class=fixture.fixture_class,
        scenario=scenario,
        action=action,
        reason_code=reason,
        predicted=predicted,
        verified=verified,
        skipped=skipped,
        executed=executed,
        false_admission=false_admission,
        collection_latency_ms=collection_ms,
        lookup_latency_ms=lookup_ms,
        verify_latency_ms=verify_ms,
        execution_latency_ms=execution_ms,
        bytes_read=bytes_read,
        bytes_written=bytes_written,
        saved_wall_time_ms=saved,
    )


def _summarize_scenario(
    scenario: BenchmarkScenario,
    observations: Sequence[CaseObservation],
    *,
    eligible_count: int,
) -> ScenarioSummary:
    metrics = ProofReuseSessionMetrics()
    reasons: Counter[str] = Counter()
    collection = lookup = verify = execution = 0
    miss_overhead = 0
    saved = 0
    bytes_read = bytes_written = 0
    predicted = verified = skipped = executed = false_admissions = 0

    for item in observations:
        collection += item.collection_latency_ms
        lookup += item.lookup_latency_ms
        verify += item.verify_latency_ms
        execution += item.execution_latency_ms
        saved += item.saved_wall_time_ms
        bytes_read += item.bytes_read
        bytes_written += item.bytes_written
        reasons[item.reason_code] += 1
        if item.predicted:
            predicted += 1
            metrics.predicted(reason_code=item.reason_code)
        if item.verified:
            verified += 1
            metrics.verified(
                reason_code=item.reason_code,
                latency_ms=item.verify_latency_ms,
                bytes_read=item.bytes_read,
            )
        if item.skipped:
            skipped += 1
            metrics.skipped(reason_code=item.reason_code)
        if item.executed:
            executed += 1
            metrics.executed(
                reason_code=item.reason_code,
                latency_ms=item.execution_latency_ms,
                bytes_written=item.bytes_written,
            )
        if item.false_admission:
            false_admissions += 1
        if (
            item.fixture_class is FixtureClass.MISS
            or (
                scenario is BenchmarkScenario.COLD_READWRITE
                and item.executed
            )
        ):
            miss_overhead += item.collection_latency_ms + item.lookup_latency_ms

    snapshot = metrics.snapshot()
    return ScenarioSummary(
        scenario=scenario,
        mode=_scenario_mode(scenario).value,
        collected=len(observations),
        eligible=eligible_count,
        predicted=predicted,
        verified=verified,
        skipped=skipped,
        executed=executed,
        false_admissions=false_admissions,
        collection_latency_ms=collection,
        lookup_latency_ms=lookup,
        verify_latency_ms=verify,
        execution_latency_ms=execution,
        miss_overhead_ms=miss_overhead,
        saved_wall_time_ms=saved,
        bytes_read=bytes_read,
        bytes_written=bytes_written,
        reason_codes=dict(sorted(reasons.items())),
        metrics=snapshot.to_dict(),
    )


# ---------------------------------------------------------------------------
# Receipt and gates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateResult:
    name: GateName
    passed: bool
    detail: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "passed": self.passed,
            "detail": dict(sorted(_plain(self.detail).items()))
            if isinstance(self.detail, Mapping)
            else {},
        }


@dataclass(frozen=True)
class ProofReuseBenchmarkReceipt:
    """Immutable, JSON-safe benchmark receipt (``BenchmarkReceipt@1``)."""

    interface: str = BENCHMARK_RECEIPT_INTERFACE
    benchmark_interface: str = PROOF_REUSE_BENCHMARK_INTERFACE
    metrics_interface: str = PROOF_REUSE_METRICS_INTERFACE
    schema: str = PROOF_REUSE_BENCHMARK_RECEIPT_SCHEMA
    version: int = PROOF_REUSE_BENCHMARK_VERSION
    requirement_id: str = PROOF_REUSE_BENCHMARK_REQUIREMENT_ID
    corpus_id: str = ""
    corpus_version: str = PROOF_REUSE_BENCHMARK_CORPUS_VERSION
    false_admissions: int = 0
    warm_eligible_count: int = 0
    warm_verified_skips: int = 0
    warm_skip_bps: int = 0
    verify_latency_ms: int = 0
    execution_latency_ms: int = 0
    miss_overhead_ms: int = 0
    max_miss_overhead_ms: int = 0
    saved_wall_time_ms: int = 0
    exclusions: Mapping[str, int] = field(default_factory=dict)
    scenario_summaries: tuple[ScenarioSummary, ...] = ()
    gates: tuple[GateResult, ...] = ()
    passed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "false_admissions", _integer(self.false_admissions, "false_admissions")
        )
        object.__setattr__(
            self,
            "warm_eligible_count",
            _integer(self.warm_eligible_count, "warm_eligible_count"),
        )
        object.__setattr__(
            self,
            "warm_verified_skips",
            _integer(self.warm_verified_skips, "warm_verified_skips"),
        )
        object.__setattr__(
            self, "warm_skip_bps", _integer(self.warm_skip_bps, "warm_skip_bps")
        )
        for name in (
            "verify_latency_ms",
            "execution_latency_ms",
            "miss_overhead_ms",
            "max_miss_overhead_ms",
            "saved_wall_time_ms",
        ):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name, minimum=0)
            )
        exclusions = {
            _reason_code(key): _integer(count, "exclusion_count")
            for key, count in dict(self.exclusions).items()
            if _reason_code(key) != "unknown"
        }
        object.__setattr__(self, "exclusions", dict(sorted(exclusions.items())))
        if not isinstance(self.scenario_summaries, tuple):
            object.__setattr__(
                self, "scenario_summaries", tuple(self.scenario_summaries)
            )
        if not isinstance(self.gates, tuple):
            object.__setattr__(self, "gates", tuple(self.gates))
        if not isinstance(self.passed, bool):
            raise ProofReuseBenchmarkError("passed must be a boolean")

    @property
    def receipt_id(self) -> str:
        return _content_id(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "benchmark_interface": self.benchmark_interface,
            "metrics_interface": self.metrics_interface,
            "version": self.version,
            "requirement_id": self.requirement_id,
            "corpus_id": self.corpus_id,
            "corpus_version": self.corpus_version,
            "false_admissions": self.false_admissions,
            "warm_eligible_count": self.warm_eligible_count,
            "warm_verified_skips": self.warm_verified_skips,
            "warm_skip_bps": self.warm_skip_bps,
            "verify_latency_ms": self.verify_latency_ms,
            "execution_latency_ms": self.execution_latency_ms,
            "miss_overhead_ms": self.miss_overhead_ms,
            "max_miss_overhead_ms": self.max_miss_overhead_ms,
            "saved_wall_time_ms": self.saved_wall_time_ms,
            "exclusions": dict(sorted(self.exclusions.items())),
            "scenario_summaries": [
                item.to_dict() for item in self.scenario_summaries
            ],
            "gates": [item.to_dict() for item in self.gates],
            "passed": self.passed,
        }

    def to_json(self) -> str:
        return _canonical_bytes(self.to_dict()).decode("utf-8")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProofReuseBenchmarkReceipt":
        if not isinstance(value, Mapping):
            raise ProofReuseBenchmarkError("receipt must be a mapping")
        if value.get("schema") != PROOF_REUSE_BENCHMARK_RECEIPT_SCHEMA:
            raise ProofReuseBenchmarkError("receipt schema mismatch")
        summaries = []
        for raw in value.get("scenario_summaries", ()):
            if not isinstance(raw, Mapping):
                raise ProofReuseBenchmarkError("scenario summary must be a mapping")
            summaries.append(
                ScenarioSummary(
                    scenario=_enum(raw.get("scenario"), BenchmarkScenario, "scenario"),
                    mode=str(raw.get("mode", "")),
                    collected=_integer(raw.get("collected", 0), "collected"),
                    eligible=_integer(raw.get("eligible", 0), "eligible"),
                    predicted=_integer(raw.get("predicted", 0), "predicted"),
                    verified=_integer(raw.get("verified", 0), "verified"),
                    skipped=_integer(raw.get("skipped", 0), "skipped"),
                    executed=_integer(raw.get("executed", 0), "executed"),
                    false_admissions=_integer(
                        raw.get("false_admissions", 0), "false_admissions"
                    ),
                    collection_latency_ms=_integer(
                        raw.get("collection_latency_ms", 0), "collection_latency_ms"
                    ),
                    lookup_latency_ms=_integer(
                        raw.get("lookup_latency_ms", 0), "lookup_latency_ms"
                    ),
                    verify_latency_ms=_integer(
                        raw.get("verify_latency_ms", 0), "verify_latency_ms"
                    ),
                    execution_latency_ms=_integer(
                        raw.get("execution_latency_ms", 0), "execution_latency_ms"
                    ),
                    miss_overhead_ms=_integer(
                        raw.get("miss_overhead_ms", 0), "miss_overhead_ms"
                    ),
                    saved_wall_time_ms=_integer(
                        raw.get("saved_wall_time_ms", 0), "saved_wall_time_ms"
                    ),
                    bytes_read=_integer(raw.get("bytes_read", 0), "bytes_read"),
                    bytes_written=_integer(
                        raw.get("bytes_written", 0), "bytes_written"
                    ),
                    reason_codes=dict(raw.get("reason_codes") or {}),
                    metrics=dict(raw.get("metrics") or {}),
                )
            )
        gates = []
        for raw in value.get("gates", ()):
            if not isinstance(raw, Mapping):
                raise ProofReuseBenchmarkError("gate must be a mapping")
            gates.append(
                GateResult(
                    name=_enum(raw.get("name"), GateName, "gate name"),
                    passed=bool(raw.get("passed")),
                    detail=dict(raw.get("detail") or {}),
                )
            )
        return cls(
            interface=str(value.get("interface", BENCHMARK_RECEIPT_INTERFACE)),
            benchmark_interface=str(
                value.get("benchmark_interface", PROOF_REUSE_BENCHMARK_INTERFACE)
            ),
            metrics_interface=str(
                value.get("metrics_interface", PROOF_REUSE_METRICS_INTERFACE)
            ),
            schema=str(value.get("schema", PROOF_REUSE_BENCHMARK_RECEIPT_SCHEMA)),
            version=_integer(value.get("version", PROOF_REUSE_BENCHMARK_VERSION), "version"),
            requirement_id=str(
                value.get("requirement_id", PROOF_REUSE_BENCHMARK_REQUIREMENT_ID)
            ),
            corpus_id=str(value.get("corpus_id", "")),
            corpus_version=str(
                value.get("corpus_version", PROOF_REUSE_BENCHMARK_CORPUS_VERSION)
            ),
            false_admissions=_integer(
                value.get("false_admissions", 0), "false_admissions"
            ),
            warm_eligible_count=_integer(
                value.get("warm_eligible_count", 0), "warm_eligible_count"
            ),
            warm_verified_skips=_integer(
                value.get("warm_verified_skips", 0), "warm_verified_skips"
            ),
            warm_skip_bps=_integer(value.get("warm_skip_bps", 0), "warm_skip_bps"),
            verify_latency_ms=_integer(
                value.get("verify_latency_ms", 0), "verify_latency_ms"
            ),
            execution_latency_ms=_integer(
                value.get("execution_latency_ms", 0), "execution_latency_ms"
            ),
            miss_overhead_ms=_integer(
                value.get("miss_overhead_ms", 0), "miss_overhead_ms"
            ),
            max_miss_overhead_ms=_integer(
                value.get("max_miss_overhead_ms", 0), "max_miss_overhead_ms"
            ),
            saved_wall_time_ms=_integer(
                value.get("saved_wall_time_ms", 0), "saved_wall_time_ms"
            ),
            exclusions=dict(value.get("exclusions") or {}),
            scenario_summaries=tuple(summaries),
            gates=tuple(gates),
            passed=bool(value.get("passed")),
        )

    @classmethod
    def from_json(cls, text: str | bytes | bytearray) -> "ProofReuseBenchmarkReceipt":
        if isinstance(text, (bytes, bytearray)):
            text = bytes(text).decode("utf-8")
        try:
            payload = json.loads(text)
        except (TypeError, ValueError, UnicodeDecodeError) as exc:
            raise ProofReuseBenchmarkError("receipt JSON is invalid") from exc
        return cls.from_dict(payload)


def evaluate_benchmark_gates(
    *,
    false_admissions: int,
    warm_eligible_count: int,
    warm_verified_skips: int,
    warm_skip_bps: int,
    verify_latency_ms: int,
    execution_latency_ms: int,
    miss_overhead_ms: int,
    max_miss_overhead_ms: int,
    receipt_reproducible: bool,
) -> tuple[GateResult, ...]:
    """Evaluate the closed PTR-100 acceptance gates."""

    return (
        GateResult(
            name=GateName.FALSE_ADMISSIONS_ZERO,
            passed=false_admissions == 0,
            detail={"false_admissions": false_admissions},
        ),
        GateResult(
            name=GateName.WARM_SKIP_THRESHOLD,
            passed=(
                warm_eligible_count > 0
                and warm_skip_bps >= MIN_WARM_SKIP_BPS
                and warm_verified_skips >= ((warm_eligible_count * MIN_WARM_SKIP_BPS) // 10_000)
            ),
            detail={
                "warm_eligible_count": warm_eligible_count,
                "warm_verified_skips": warm_verified_skips,
                "warm_skip_bps": warm_skip_bps,
                "min_warm_skip_bps": MIN_WARM_SKIP_BPS,
            },
        ),
        GateResult(
            name=GateName.VERIFY_CHEAPER_THAN_EXECUTE,
            passed=(
                verify_latency_ms >= 0
                and execution_latency_ms > 0
                and verify_latency_ms < execution_latency_ms
            ),
            detail={
                "verify_latency_ms": verify_latency_ms,
                "execution_latency_ms": execution_latency_ms,
            },
        ),
        GateResult(
            name=GateName.MISS_OVERHEAD_BOUNDED,
            passed=miss_overhead_ms <= max_miss_overhead_ms,
            detail={
                "miss_overhead_ms": miss_overhead_ms,
                "max_miss_overhead_ms": max_miss_overhead_ms,
                "max_miss_overhead_bps": MAX_MISS_OVERHEAD_BPS,
            },
        ),
        GateResult(
            name=GateName.RECEIPT_REPRODUCIBLE,
            passed=bool(receipt_reproducible),
            detail={"receipt_reproducible": bool(receipt_reproducible)},
        ),
    )


@dataclass
class ProofReuseBenchmark:
    """Orchestrates the closed shadow/warm proof-reuse measurement population."""

    interface: str = PROOF_REUSE_BENCHMARK_INTERFACE
    corpus: BenchmarkCorpus = field(default_factory=build_default_benchmark_corpus)

    def run(self) -> ProofReuseBenchmarkReceipt:
        """Execute every required scenario and return an immutable receipt."""

        corpus = self.corpus
        eligible = corpus.eligible_warm
        eligible_count = len(eligible)
        if eligible_count <= 0:
            raise ProofReuseBenchmarkError("eligible warm population is empty")

        scenario_summaries: list[ScenarioSummary] = []
        all_observations: list[CaseObservation] = []
        total_false = 0

        for scenario in REQUIRED_SCENARIOS:
            lookup, identities = _lookup_for_scenario(scenario, corpus.fixtures)
            observations = tuple(
                _observe_case(fixture, scenario, lookup, identities)
                for fixture in corpus.fixtures
            )
            all_observations.extend(observations)
            summary = _summarize_scenario(
                scenario,
                observations,
                eligible_count=eligible_count,
            )
            scenario_summaries.append(summary)
            total_false += summary.false_admissions

        warm_summary = next(
            item
            for item in scenario_summaries
            if item.scenario is BenchmarkScenario.WARM_READ
        )
        off_summary = next(
            item
            for item in scenario_summaries
            if item.scenario is BenchmarkScenario.OFF
        )
        cold_summary = next(
            item
            for item in scenario_summaries
            if item.scenario is BenchmarkScenario.COLD_READWRITE
        )

        warm_cases = [
            item
            for item in all_observations
            if item.scenario is BenchmarkScenario.WARM_READ
            and item.fixture_class is FixtureClass.ELIGIBLE_WARM
        ]
        warm_verified_skips = sum(
            1 for item in warm_cases if item.verified and item.skipped
        )
        warm_skip_bps = _bps(warm_verified_skips, eligible_count)

        # Warm-path verify cost for skipped eligible fixtures vs full execution
        # cost of the same population under the off baseline.
        verify_latency_ms = sum(item.verify_latency_ms for item in warm_cases if item.skipped)
        execution_latency_ms = sum(
            item.execute_cost_ms for item in corpus.eligible_warm
        )
        if execution_latency_ms <= 0:
            # Fall back to off-scenario execution total for the eligible set.
            execution_latency_ms = off_summary.execution_latency_ms

        # Miss overhead: collection+lookup for miss-class and cold empty-cache
        # paths, bounded against a single eligible execution cost unit times
        # the miss population (or cold collected count).
        miss_overhead_ms = cold_summary.miss_overhead_ms
        miss_population = max(
            1,
            sum(
                1
                for item in corpus.fixtures
                if item.fixture_class is FixtureClass.MISS
            ),
        )
        # Bound: per-item miss overhead ≤ MAX_MISS_OVERHEAD_BPS of one execute.
        per_item_cap = max(
            1,
            (DEFAULT_EXECUTE_COST_MS * MAX_MISS_OVERHEAD_BPS) // 10_000,
        )
        # Cold path measures overhead across the whole collected set.
        max_miss_overhead_ms = per_item_cap * max(cold_summary.collected, miss_population)

        saved_wall_time_ms = warm_summary.saved_wall_time_ms
        exclusions = dict(corpus.exclusions)

        # Provisional gates without the reproducibility self-check; we re-run
        # once and compare receipt ids for the final gate.
        provisional_gates = evaluate_benchmark_gates(
            false_admissions=total_false,
            warm_eligible_count=eligible_count,
            warm_verified_skips=warm_verified_skips,
            warm_skip_bps=warm_skip_bps,
            verify_latency_ms=verify_latency_ms
            if verify_latency_ms > 0
            else warm_summary.verify_latency_ms,
            execution_latency_ms=execution_latency_ms,
            miss_overhead_ms=miss_overhead_ms,
            max_miss_overhead_ms=max_miss_overhead_ms,
            receipt_reproducible=True,
        )
        provisional = ProofReuseBenchmarkReceipt(
            corpus_id=corpus.corpus_id,
            corpus_version=corpus.corpus_version,
            requirement_id=corpus.requirement_id,
            false_admissions=total_false,
            warm_eligible_count=eligible_count,
            warm_verified_skips=warm_verified_skips,
            warm_skip_bps=warm_skip_bps,
            verify_latency_ms=(
                verify_latency_ms
                if verify_latency_ms > 0
                else warm_summary.verify_latency_ms
            ),
            execution_latency_ms=execution_latency_ms,
            miss_overhead_ms=miss_overhead_ms,
            max_miss_overhead_ms=max_miss_overhead_ms,
            saved_wall_time_ms=saved_wall_time_ms,
            exclusions=exclusions,
            scenario_summaries=tuple(scenario_summaries),
            gates=provisional_gates,
            passed=all(gate.passed for gate in provisional_gates),
        )

        # Second independent run must produce an identical receipt body.
        twin = self._run_once()
        reproducible = (
            twin.to_dict() == provisional.to_dict()
            and twin.receipt_id == provisional.receipt_id
        )
        final_gates = evaluate_benchmark_gates(
            false_admissions=total_false,
            warm_eligible_count=eligible_count,
            warm_verified_skips=warm_verified_skips,
            warm_skip_bps=warm_skip_bps,
            verify_latency_ms=provisional.verify_latency_ms,
            execution_latency_ms=execution_latency_ms,
            miss_overhead_ms=miss_overhead_ms,
            max_miss_overhead_ms=max_miss_overhead_ms,
            receipt_reproducible=reproducible,
        )
        return ProofReuseBenchmarkReceipt(
            corpus_id=corpus.corpus_id,
            corpus_version=corpus.corpus_version,
            requirement_id=corpus.requirement_id,
            false_admissions=total_false,
            warm_eligible_count=eligible_count,
            warm_verified_skips=warm_verified_skips,
            warm_skip_bps=warm_skip_bps,
            verify_latency_ms=provisional.verify_latency_ms,
            execution_latency_ms=execution_latency_ms,
            miss_overhead_ms=miss_overhead_ms,
            max_miss_overhead_ms=max_miss_overhead_ms,
            saved_wall_time_ms=saved_wall_time_ms,
            exclusions=exclusions,
            scenario_summaries=tuple(scenario_summaries),
            gates=final_gates,
            passed=all(gate.passed for gate in final_gates),
        )

    def _run_once(self) -> ProofReuseBenchmarkReceipt:
        """Single-pass measurement used for the reproducibility self-check."""

        corpus = self.corpus
        eligible_count = len(corpus.eligible_warm)
        scenario_summaries: list[ScenarioSummary] = []
        all_observations: list[CaseObservation] = []
        total_false = 0
        for scenario in REQUIRED_SCENARIOS:
            lookup, identities = _lookup_for_scenario(scenario, corpus.fixtures)
            observations = tuple(
                _observe_case(fixture, scenario, lookup, identities)
                for fixture in corpus.fixtures
            )
            all_observations.extend(observations)
            summary = _summarize_scenario(
                scenario,
                observations,
                eligible_count=eligible_count,
            )
            scenario_summaries.append(summary)
            total_false += summary.false_admissions

        warm_summary = next(
            item
            for item in scenario_summaries
            if item.scenario is BenchmarkScenario.WARM_READ
        )
        off_summary = next(
            item
            for item in scenario_summaries
            if item.scenario is BenchmarkScenario.OFF
        )
        cold_summary = next(
            item
            for item in scenario_summaries
            if item.scenario is BenchmarkScenario.COLD_READWRITE
        )
        warm_cases = [
            item
            for item in all_observations
            if item.scenario is BenchmarkScenario.WARM_READ
            and item.fixture_class is FixtureClass.ELIGIBLE_WARM
        ]
        warm_verified_skips = sum(
            1 for item in warm_cases if item.verified and item.skipped
        )
        warm_skip_bps = _bps(warm_verified_skips, eligible_count)
        verify_latency_ms = sum(
            item.verify_latency_ms for item in warm_cases if item.skipped
        )
        execution_latency_ms = sum(
            item.execute_cost_ms for item in corpus.eligible_warm
        ) or off_summary.execution_latency_ms
        miss_overhead_ms = cold_summary.miss_overhead_ms
        miss_population = max(
            1,
            sum(
                1
                for item in corpus.fixtures
                if item.fixture_class is FixtureClass.MISS
            ),
        )
        per_item_cap = max(
            1,
            (DEFAULT_EXECUTE_COST_MS * MAX_MISS_OVERHEAD_BPS) // 10_000,
        )
        max_miss_overhead_ms = per_item_cap * max(
            cold_summary.collected, miss_population
        )
        gates = evaluate_benchmark_gates(
            false_admissions=total_false,
            warm_eligible_count=eligible_count,
            warm_verified_skips=warm_verified_skips,
            warm_skip_bps=warm_skip_bps,
            verify_latency_ms=verify_latency_ms or warm_summary.verify_latency_ms,
            execution_latency_ms=execution_latency_ms,
            miss_overhead_ms=miss_overhead_ms,
            max_miss_overhead_ms=max_miss_overhead_ms,
            receipt_reproducible=True,
        )
        return ProofReuseBenchmarkReceipt(
            corpus_id=corpus.corpus_id,
            corpus_version=corpus.corpus_version,
            requirement_id=corpus.requirement_id,
            false_admissions=total_false,
            warm_eligible_count=eligible_count,
            warm_verified_skips=warm_verified_skips,
            warm_skip_bps=warm_skip_bps,
            verify_latency_ms=verify_latency_ms or warm_summary.verify_latency_ms,
            execution_latency_ms=execution_latency_ms,
            miss_overhead_ms=miss_overhead_ms,
            max_miss_overhead_ms=max_miss_overhead_ms,
            saved_wall_time_ms=warm_summary.saved_wall_time_ms,
            exclusions=dict(corpus.exclusions),
            scenario_summaries=tuple(scenario_summaries),
            gates=gates,
            passed=all(gate.passed for gate in gates),
        )


def run_proof_reuse_benchmark(
    corpus: BenchmarkCorpus | None = None,
) -> ProofReuseBenchmarkReceipt:
    """Convenience entry point used by tests and operator tooling."""

    benchmark = ProofReuseBenchmark(
        corpus=corpus if corpus is not None else build_default_benchmark_corpus()
    )
    return benchmark.run()


def verify_benchmark_receipt(
    receipt: ProofReuseBenchmarkReceipt | Mapping[str, Any],
    *,
    corpus: BenchmarkCorpus | None = None,
) -> bool:
    """Recompute the benchmark and require an exact receipt match."""

    if isinstance(receipt, Mapping):
        receipt = ProofReuseBenchmarkReceipt.from_dict(receipt)
    if not isinstance(receipt, ProofReuseBenchmarkReceipt):
        raise ProofReuseBenchmarkError("receipt type is unsupported")
    recomputed = run_proof_reuse_benchmark(corpus=corpus)
    if corpus is not None and recomputed.corpus_id != receipt.corpus_id:
        # Allow verifying a custom corpus receipt against the same corpus only.
        pass
    if corpus is None and receipt.corpus_id != recomputed.corpus_id:
        return False
    if corpus is not None:
        expected = ProofReuseBenchmark(corpus=corpus).run()
        return expected.to_dict() == receipt.to_dict()
    return recomputed.to_dict() == receipt.to_dict()


__all__ = [
    "BENCHMARK_RECEIPT_INTERFACE",
    "DEFAULT_ELIGIBLE_WARM_COUNT",
    "DEFAULT_EXECUTE_COST_MS",
    "DEFAULT_VERIFY_COST_MS",
    "MAX_MISS_OVERHEAD_BPS",
    "MIN_WARM_SKIP_BPS",
    "PROOF_REUSE_BENCHMARK_CORPUS_SCHEMA",
    "PROOF_REUSE_BENCHMARK_CORPUS_VERSION",
    "PROOF_REUSE_BENCHMARK_INTERFACE",
    "PROOF_REUSE_BENCHMARK_RECEIPT_SCHEMA",
    "PROOF_REUSE_BENCHMARK_REQUIREMENT_ID",
    "PROOF_REUSE_BENCHMARK_VERSION",
    "PROOF_REUSE_METRICS_INTERFACE",
    "REQUIRED_GATES",
    "REQUIRED_SCENARIOS",
    "BenchmarkCorpus",
    "BenchmarkFixture",
    "BenchmarkScenario",
    "CaseObservation",
    "FixtureClass",
    "GateName",
    "GateResult",
    "GroundTruth",
    "ProofReuseBenchmark",
    "ProofReuseBenchmarkError",
    "ProofReuseBenchmarkReceipt",
    "ScenarioSummary",
    "build_default_benchmark_corpus",
    "evaluate_benchmark_gates",
    "run_proof_reuse_benchmark",
    "verify_benchmark_receipt",
]
