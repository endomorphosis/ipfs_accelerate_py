"""Closed-loop quality, coverage, token, and proof-cost gates (CBP-130).

Interface: ``CodebaseProofBenchmark@1``

This module is a measurement and gate boundary, not a supervisor.  It
preregisters a fixed baseline and held-out mutation/repair suite before any
outcome inspection, compares bulk-source and obligation-first paths on
identical tasks, and recomputes every reported rate from additive counts.

Wire payloads carry identities, counters, digests, and compact codes only.
Prompts, source bodies, decoded model output, patches, and nested artifact
graphs are outside this contract.

Deterministic fixture gates are evaluated independently of any live-model
channel.  Live observations, when present, never contribute to fixture gate
pass/fail authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

from .code_claim_contracts import ClaimFamily, ClaimStatus, EvidenceTier
from .formal_verification_contracts import AssuranceLevel, ContractValidationError
from ..self_improvement.supervisor_efficiency_metrics import (
    BASIS_POINTS,
    CODE_PROOF_EFFICIENCY_EVIDENCE_ID,
    CODE_PROOF_MIN_INPUT_TOKEN_REDUCTION_BPS,
    CODE_PROOF_MIN_RETRY_TOKEN_REDUCTION_BPS,
    CodeProofEfficiencyReport,
    build_code_proof_efficiency_report,
    build_code_proof_paired_receipts,
)


CODEBASE_PROOF_BENCHMARK_INTERFACE: Final = "CodebaseProofBenchmark@1"
CODEBASE_PROOF_BENCHMARK_VERSION: Final = 1
CODEBASE_PROOF_BENCHMARK_CORPUS_VERSION: Final = "codebase-proof-efficiency@1"
CODEBASE_PROOF_CLAIM_OUTCOME_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/code-proof-claim-outcome@1"
)
CODEBASE_PROOF_ARM_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/code-proof-benchmark-arm@1"
)
CODEBASE_PROOF_PAIRED_CASE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/code-proof-paired-case@1"
)
CODEBASE_PROOF_MUTATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/code-proof-mutation-seed@1"
)
CODEBASE_PROOF_SUITE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/code-proof-benchmark-suite@1"
)
CODEBASE_PROOF_REPORT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/code-proof-benchmark-report@1"
)
CODEBASE_PROOF_LIVE_CHANNEL_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/code-proof-live-model-channel@1"
)

# Producer-owned evidence identity for CBP-G130 (CBPEV130MET).
CODEBASE_PROOF_EFFICIENCY_REQUIREMENT_ID: Final = CODE_PROOF_EFFICIENCY_EVIDENCE_ID
CODEBASE_PROOF_OBJECTIVE_ID: Final = "CBP-G130"
CODEBASE_PROOF_OBJECTIVE_REVISION: Final = (
    "sha256:15bf52ec4879cbp130codebaseproofefficiencygates000000000000000000"
)

# Frozen envelope identities.  Changing any requires a corpus-version bump.
CBP_FROZEN_REPOSITORY_ID: Final = "repository:codebase-proof-benchmark@1"
CBP_FROZEN_TREE_ID: Final = (
    "sha256:67e297d6cf296c60593fa3617b84f9f17abf6df8cbp130baseline000000000"
)
CBP_FROZEN_POLICY_ID: Final = "policy:codebase-proof-benchmark@1"
CBP_FROZEN_POLICY_REVISION: Final = (
    "sha256:e9f9146533b39e99ceb8ca1e41a6ce92765dc4a3a627f1737d4951e52e4dca54"
)
CBP_FROZEN_PROVIDER_ID: Final = "provider:deterministic-fixture@1"
CBP_FROZEN_CAPABILITY_ID: Final = "capability:codebase-proof-measurement@1"

MAX_CLAIMS_PER_ARM: Final = 256
MAX_CASES: Final = 512
MAX_MUTATIONS: Final = 256
MAX_TEXT_BYTES: Final = 512
MAX_REFERENCE_BYTES: Final = 256
MAX_COUNTER: Final = 10**15
MAX_DURATION_MS: Final = 31 * 24 * 60 * 60 * 1000
MAX_REPORT_BYTES: Final = 2 * 1024 * 1024
MAX_SUITE_BYTES: Final = 4 * 1024 * 1024

# Fixture gates (acceptance): ≥40% fewer input tokens/criterion, ≥60% retry
# token reduction, zero false authoritative admissions, no coverage loss,
# warm prove-cost improvement when cache hits dominate.
MIN_INPUT_TOKEN_REDUCTION_BPS: Final = CODE_PROOF_MIN_INPUT_TOKEN_REDUCTION_BPS
MIN_RETRY_TOKEN_REDUCTION_BPS: Final = CODE_PROOF_MIN_RETRY_TOKEN_REDUCTION_BPS
MAX_FALSE_AUTHORITATIVE_ADMISSIONS: Final = 0
MAX_ACCEPTED_PATCH_REGRESSIONS: Final = 0

REQUIRED_CLAIM_FAMILIES: Final[tuple[ClaimFamily, ...]] = (
    ClaimFamily.DEPENDENCY_REACHABILITY,
    ClaimFamily.API_CONTRACT,
    ClaimFamily.BEHAVIORAL_INVARIANT,
    ClaimFamily.SECURITY_PROPERTY,
    ClaimFamily.SEMANTIC_EQUIVALENCE,
    ClaimFamily.SUPERVISOR_LIFECYCLE,
)

REQUIRED_CLAIM_STATUSES: Final[tuple[ClaimStatus, ...]] = (
    ClaimStatus.SATISFIED,
    ClaimStatus.REFUTED,
    ClaimStatus.OPEN,
    ClaimStatus.UNSUPPORTED,
    ClaimStatus.NOT_MEASURED,
    ClaimStatus.STALE,
)

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
        "proof_body",
        "counterexample_body",
    }
)


class CodebaseProofBenchmarkError(ContractValidationError):
    """Benchmark suite, observation, or report is malformed or detached."""


class ContextPath(str, Enum):
    """The two context compilation strategies compared on identical tasks."""

    BULK_SOURCE = "bulk_source"
    OBLIGATION_FIRST = "obligation_first"


class ResultChannel(str, Enum):
    """Whether an observation feeds deterministic gates or live reporting."""

    DETERMINISTIC_FIXTURE = "deterministic_fixture"
    LIVE_MODEL = "live_model"


class MutationSeedKind(str, Enum):
    """Held-out mutation / repair seeds preregistered before inspection."""

    FALSE_ADMIT = "false_admit"
    FALSE_REFUTE = "false_refute"
    STALE_EVIDENCE = "stale_evidence"
    FIRST_PASS_REPAIR = "first_pass_repair"
    EVENTUAL_REPAIR = "eventual_repair"
    ACCEPTED_PATCH_REGRESSION = "accepted_patch_regression"
    WARM_CACHE_DOMINATED = "warm_cache_dominated"
    REQUIRED_COVERAGE = "required_coverage"


class CodeProofGateName(str, Enum):
    ZERO_FALSE_AUTHORITATIVE_ADMISSIONS = "zero_false_authoritative_admissions"
    NO_REQUIRED_COVERAGE_LOSS = "no_required_coverage_loss"
    INPUT_TOKEN_REDUCTION = "input_token_reduction"
    RETRY_TOKEN_REDUCTION = "retry_token_reduction"
    WARM_PROVE_COST_IMPROVEMENT = "warm_prove_cost_improvement"
    REQUIRED_FAMILY_COVERAGE = "required_family_coverage"
    FIXTURE_CHANNEL_ISOLATION = "fixture_channel_isolation"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
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
        raise CodebaseProofBenchmarkError(
            "benchmark data must be canonical JSON"
        ) from exc


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _fixture_digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _reject_forbidden(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _FORBIDDEN_PAYLOAD_KEYS or normalized.endswith(
                "_body"
            ):
                raise CodebaseProofBenchmarkError(
                    f"benchmark payload cannot contain {key!r}"
                )
            _reject_forbidden(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_forbidden(item)


def _text(value: Any, name: str, *, maximum: int = MAX_TEXT_BYTES) -> str:
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str) or not value.strip():
        raise CodebaseProofBenchmarkError(f"{name} must be non-empty text")
    result = value.strip()
    if "\x00" in result or len(result.encode("utf-8")) > maximum:
        raise CodebaseProofBenchmarkError(
            f"{name} is unsafe or exceeds its {maximum}-byte bound"
        )
    return result


def _code(value: Any, name: str) -> str:
    result = _text(value, name, maximum=192).lower()
    if not _CODE.fullmatch(result):
        raise CodebaseProofBenchmarkError(f"{name} must be a compact code")
    return result


def _content_id(value: Any, name: str) -> str:
    result = _text(value, name, maximum=71).lower()
    if not _CONTENT_ID.fullmatch(result):
        raise CodebaseProofBenchmarkError(
            f"{name} must be a lowercase sha256 content ID"
        )
    return result


def _integer(
    value: Any,
    name: str,
    *,
    maximum: int = MAX_COUNTER,
    minimum: int = 0,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        raise CodebaseProofBenchmarkError(
            f"{name} must be an integer from {minimum} through {maximum}"
        )
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise CodebaseProofBenchmarkError(f"{name} must be a boolean")
    return value


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    raw = getattr(value, "value", value)
    try:
        return enum_type(str(raw))
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise CodebaseProofBenchmarkError(
            f"{name} must be one of: {allowed}"
        ) from exc


def _rate_bps(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        return 0
    return (numerator * BASIS_POINTS) // denominator


def _reduction_bps(baseline: int, candidate: int) -> int:
    if baseline <= 0:
        return 0
    if candidate > baseline:
        return 0
    return ((baseline - candidate) * BASIS_POINTS) // baseline


def _strict_keys(
    payload: Mapping[str, Any],
    allowed: set[str],
    *,
    name: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise CodebaseProofBenchmarkError(f"{name} must be an object")
    extras = sorted(set(payload) - allowed)
    missing = sorted(allowed - set(payload))
    if extras or missing:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extras:
            details.append("unexpected " + ", ".join(extras))
        raise CodebaseProofBenchmarkError(
            f"{name} has invalid fields: {'; '.join(details)}"
        )


# ---------------------------------------------------------------------------
# Observation records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClaimOutcomeObservation:
    """One claim lifecycle observation inside a benchmark arm.

    Counts and codes only — no claim statement bodies or evidence payloads.
    """

    claim_reference: str
    claim_family: ClaimFamily
    evidence_tier: EvidenceTier
    required_assurance: AssuranceLevel
    status: ClaimStatus
    authoritative_admission: bool
    false_admit: bool
    false_refute: bool
    required_for_coverage: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "claim_reference",
            _text(self.claim_reference, "claim_reference", maximum=MAX_REFERENCE_BYTES),
        )
        object.__setattr__(
            self, "claim_family", _enum(self.claim_family, ClaimFamily, "claim_family")
        )
        object.__setattr__(
            self, "evidence_tier", _enum(self.evidence_tier, EvidenceTier, "evidence_tier")
        )
        object.__setattr__(
            self,
            "required_assurance",
            _enum(self.required_assurance, AssuranceLevel, "required_assurance"),
        )
        object.__setattr__(self, "status", _enum(self.status, ClaimStatus, "status"))
        object.__setattr__(
            self,
            "authoritative_admission",
            _boolean(self.authoritative_admission, "authoritative_admission"),
        )
        object.__setattr__(self, "false_admit", _boolean(self.false_admit, "false_admit"))
        object.__setattr__(
            self, "false_refute", _boolean(self.false_refute, "false_refute")
        )
        object.__setattr__(
            self,
            "required_for_coverage",
            _boolean(self.required_for_coverage, "required_for_coverage"),
        )
        if self.false_admit and not self.authoritative_admission:
            raise CodebaseProofBenchmarkError(
                "false_admit requires authoritative_admission"
            )
        if self.false_refute and self.status is not ClaimStatus.REFUTED:
            raise CodebaseProofBenchmarkError(
                "false_refute requires status=refuted"
            )
        # Authoritative admissions that are false admits must not be marked
        # satisfied without the mutation flag; fixture honesty check.
        if (
            self.authoritative_admission
            and self.status is ClaimStatus.SATISFIED
            and self.false_admit
        ):
            pass  # seeded mutation observation is intentional

    @property
    def content_id(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CODEBASE_PROOF_CLAIM_OUTCOME_SCHEMA,
            "claim_reference": self.claim_reference,
            "claim_family": self.claim_family.value,
            "evidence_tier": self.evidence_tier.value,
            "required_assurance": self.required_assurance.value,
            "status": self.status.value,
            "authoritative_admission": self.authoritative_admission,
            "false_admit": self.false_admit,
            "false_refute": self.false_refute,
            "required_for_coverage": self.required_for_coverage,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ClaimOutcomeObservation":
        if not isinstance(payload, Mapping):
            raise CodebaseProofBenchmarkError("claim outcome must be an object")
        if payload.get("schema") not in (None, CODEBASE_PROOF_CLAIM_OUTCOME_SCHEMA):
            raise CodebaseProofBenchmarkError("unsupported claim outcome schema")
        return cls(
            claim_reference=payload.get("claim_reference", ""),
            claim_family=payload.get("claim_family", ""),
            evidence_tier=payload.get("evidence_tier", ""),
            required_assurance=payload.get("required_assurance", ""),
            status=payload.get("status", ""),
            authoritative_admission=payload.get("authoritative_admission", False),
            false_admit=payload.get("false_admit", False),
            false_refute=payload.get("false_refute", False),
            required_for_coverage=payload.get("required_for_coverage", True),
        )


@dataclass(frozen=True)
class CodeProofArmObservation:
    """Resource and claim outcomes for one path on one task."""

    task_reference: str
    path: ContextPath
    channel: ResultChannel
    claim_family: ClaimFamily
    input_tokens: int
    retry_tokens: int
    output_tokens: int
    provider_calls: int
    cache_lookups: int
    cache_hits: int
    cache_rejects: int
    wall_time_ms: int
    proof_cost_microunits: int
    accepted_criteria: int
    first_pass_success: bool
    eventual_repair_success: bool
    accepted_patch_regression: bool
    claims: tuple[ClaimOutcomeObservation, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "task_reference",
            _text(self.task_reference, "task_reference", maximum=MAX_REFERENCE_BYTES),
        )
        object.__setattr__(self, "path", _enum(self.path, ContextPath, "path"))
        object.__setattr__(
            self, "channel", _enum(self.channel, ResultChannel, "channel")
        )
        object.__setattr__(
            self, "claim_family", _enum(self.claim_family, ClaimFamily, "claim_family")
        )
        for name in (
            "input_tokens",
            "retry_tokens",
            "output_tokens",
            "provider_calls",
            "cache_lookups",
            "cache_hits",
            "cache_rejects",
            "accepted_criteria",
        ):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "wall_time_ms",
            _integer(self.wall_time_ms, "wall_time_ms", maximum=MAX_DURATION_MS),
        )
        object.__setattr__(
            self,
            "proof_cost_microunits",
            _integer(self.proof_cost_microunits, "proof_cost_microunits"),
        )
        object.__setattr__(
            self,
            "first_pass_success",
            _boolean(self.first_pass_success, "first_pass_success"),
        )
        object.__setattr__(
            self,
            "eventual_repair_success",
            _boolean(self.eventual_repair_success, "eventual_repair_success"),
        )
        object.__setattr__(
            self,
            "accepted_patch_regression",
            _boolean(self.accepted_patch_regression, "accepted_patch_regression"),
        )
        if self.cache_hits + self.cache_rejects > self.cache_lookups:
            raise CodebaseProofBenchmarkError(
                "cache hits and rejects cannot exceed lookups"
            )
        if self.retry_tokens > self.input_tokens + self.retry_tokens:
            raise CodebaseProofBenchmarkError("retry_tokens accounting overflow")
        claims = tuple(self.claims)
        if len(claims) > MAX_CLAIMS_PER_ARM:
            raise CodebaseProofBenchmarkError("claims exceed arm bound")
        normalized: list[ClaimOutcomeObservation] = []
        for item in claims:
            if isinstance(item, ClaimOutcomeObservation):
                normalized.append(item)
            elif isinstance(item, Mapping):
                normalized.append(ClaimOutcomeObservation.from_dict(item))
            else:
                raise CodebaseProofBenchmarkError("invalid claim observation")
        refs = [c.claim_reference for c in normalized]
        if len(refs) != len(set(refs)):
            raise CodebaseProofBenchmarkError("claim references must be unique per arm")
        object.__setattr__(self, "claims", tuple(normalized))

    @property
    def input_tokens_per_accepted_criterion(self) -> int:
        if self.accepted_criteria <= 0:
            return self.input_tokens
        return self.input_tokens // self.accepted_criteria

    @property
    def cache_hit_rate_bps(self) -> int:
        return _rate_bps(self.cache_hits, self.cache_lookups)

    @property
    def cache_reject_rate_bps(self) -> int:
        return _rate_bps(self.cache_rejects, self.cache_lookups)

    @property
    def false_admit_count(self) -> int:
        return sum(1 for c in self.claims if c.false_admit)

    @property
    def false_refute_count(self) -> int:
        return sum(1 for c in self.claims if c.false_refute)

    @property
    def authoritative_admission_count(self) -> int:
        return sum(1 for c in self.claims if c.authoritative_admission)

    @property
    def required_satisfied_count(self) -> int:
        return sum(
            1
            for c in self.claims
            if c.required_for_coverage and c.status is ClaimStatus.SATISFIED
        )

    @property
    def required_claim_count(self) -> int:
        return sum(1 for c in self.claims if c.required_for_coverage)

    @property
    def content_id(self) -> str:
        return _digest(self.to_dict())

    def status_counts(self) -> dict[str, int]:
        counts = {status.value: 0 for status in REQUIRED_CLAIM_STATUSES}
        counts[ClaimStatus.UNKNOWN.value] = 0
        for claim in self.claims:
            counts[claim.status.value] = counts.get(claim.status.value, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CODEBASE_PROOF_ARM_SCHEMA,
            "task_reference": self.task_reference,
            "path": self.path.value,
            "channel": self.channel.value,
            "claim_family": self.claim_family.value,
            "input_tokens": self.input_tokens,
            "retry_tokens": self.retry_tokens,
            "output_tokens": self.output_tokens,
            "provider_calls": self.provider_calls,
            "cache_lookups": self.cache_lookups,
            "cache_hits": self.cache_hits,
            "cache_rejects": self.cache_rejects,
            "wall_time_ms": self.wall_time_ms,
            "proof_cost_microunits": self.proof_cost_microunits,
            "accepted_criteria": self.accepted_criteria,
            "first_pass_success": self.first_pass_success,
            "eventual_repair_success": self.eventual_repair_success,
            "accepted_patch_regression": self.accepted_patch_regression,
            "claims": [c.to_dict() for c in self.claims],
            "input_tokens_per_accepted_criterion": (
                self.input_tokens_per_accepted_criterion
            ),
            "cache_hit_rate_bps": self.cache_hit_rate_bps,
            "cache_reject_rate_bps": self.cache_reject_rate_bps,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CodeProofArmObservation":
        if not isinstance(payload, Mapping):
            raise CodebaseProofBenchmarkError("arm observation must be an object")
        if payload.get("schema") not in (None, CODEBASE_PROOF_ARM_SCHEMA):
            raise CodebaseProofBenchmarkError("unsupported arm observation schema")
        return cls(
            task_reference=payload.get("task_reference", ""),
            path=payload.get("path", ""),
            channel=payload.get("channel", ""),
            claim_family=payload.get("claim_family", ""),
            input_tokens=payload.get("input_tokens", 0),
            retry_tokens=payload.get("retry_tokens", 0),
            output_tokens=payload.get("output_tokens", 0),
            provider_calls=payload.get("provider_calls", 0),
            cache_lookups=payload.get("cache_lookups", 0),
            cache_hits=payload.get("cache_hits", 0),
            cache_rejects=payload.get("cache_rejects", 0),
            wall_time_ms=payload.get("wall_time_ms", 0),
            proof_cost_microunits=payload.get("proof_cost_microunits", 0),
            accepted_criteria=payload.get("accepted_criteria", 0),
            first_pass_success=payload.get("first_pass_success", False),
            eventual_repair_success=payload.get("eventual_repair_success", False),
            accepted_patch_regression=payload.get(
                "accepted_patch_regression", False
            ),
            claims=tuple(payload.get("claims") or ()),
        )


@dataclass(frozen=True)
class CodeProofPairedCase:
    """Identical-task comparison of bulk-source vs obligation-first arms."""

    task_reference: str
    claim_family: ClaimFamily
    bulk: CodeProofArmObservation
    obligation_first: CodeProofArmObservation
    warm_cache_dominated: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "task_reference",
            _text(self.task_reference, "task_reference", maximum=MAX_REFERENCE_BYTES),
        )
        object.__setattr__(
            self, "claim_family", _enum(self.claim_family, ClaimFamily, "claim_family")
        )
        bulk = self.bulk
        obligation = self.obligation_first
        if isinstance(bulk, Mapping):
            bulk = CodeProofArmObservation.from_dict(bulk)
        if isinstance(obligation, Mapping):
            obligation = CodeProofArmObservation.from_dict(obligation)
        if not isinstance(bulk, CodeProofArmObservation):
            raise CodebaseProofBenchmarkError("bulk arm is required")
        if not isinstance(obligation, CodeProofArmObservation):
            raise CodebaseProofBenchmarkError("obligation_first arm is required")
        if bulk.task_reference != self.task_reference:
            raise CodebaseProofBenchmarkError("bulk task_reference mismatch")
        if obligation.task_reference != self.task_reference:
            raise CodebaseProofBenchmarkError(
                "obligation_first task_reference mismatch"
            )
        if bulk.path is not ContextPath.BULK_SOURCE:
            raise CodebaseProofBenchmarkError("bulk arm path must be bulk_source")
        if obligation.path is not ContextPath.OBLIGATION_FIRST:
            raise CodebaseProofBenchmarkError(
                "obligation arm path must be obligation_first"
            )
        if bulk.claim_family is not self.claim_family:
            raise CodebaseProofBenchmarkError("bulk claim_family mismatch")
        if obligation.claim_family is not self.claim_family:
            raise CodebaseProofBenchmarkError(
                "obligation_first claim_family mismatch"
            )
        object.__setattr__(self, "bulk", bulk)
        object.__setattr__(self, "obligation_first", obligation)
        object.__setattr__(
            self,
            "warm_cache_dominated",
            _boolean(self.warm_cache_dominated, "warm_cache_dominated"),
        )

    @property
    def input_token_reduction_bps(self) -> int:
        return _reduction_bps(self.bulk.input_tokens, self.obligation_first.input_tokens)

    @property
    def tokens_per_criterion_reduction_bps(self) -> int:
        return _reduction_bps(
            self.bulk.input_tokens_per_accepted_criterion,
            self.obligation_first.input_tokens_per_accepted_criterion,
        )

    @property
    def retry_token_reduction_bps(self) -> int:
        return _reduction_bps(
            self.bulk.retry_tokens, self.obligation_first.retry_tokens
        )

    @property
    def proof_cost_reduction_bps(self) -> int:
        return _reduction_bps(
            self.bulk.proof_cost_microunits,
            self.obligation_first.proof_cost_microunits,
        )

    @property
    def required_coverage_preserved(self) -> bool:
        bulk_required = {
            c.claim_reference
            for c in self.bulk.claims
            if c.required_for_coverage and c.status is ClaimStatus.SATISFIED
        }
        obl_required = {
            c.claim_reference
            for c in self.obligation_first.claims
            if c.required_for_coverage and c.status is ClaimStatus.SATISFIED
        }
        # No loss: every required claim satisfied in bulk remains satisfied.
        return bulk_required.issubset(obl_required)

    @property
    def content_id(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CODEBASE_PROOF_PAIRED_CASE_SCHEMA,
            "task_reference": self.task_reference,
            "claim_family": self.claim_family.value,
            "bulk": self.bulk.to_dict(),
            "obligation_first": self.obligation_first.to_dict(),
            "warm_cache_dominated": self.warm_cache_dominated,
            "input_token_reduction_bps": self.input_token_reduction_bps,
            "tokens_per_criterion_reduction_bps": (
                self.tokens_per_criterion_reduction_bps
            ),
            "retry_token_reduction_bps": self.retry_token_reduction_bps,
            "proof_cost_reduction_bps": self.proof_cost_reduction_bps,
            "required_coverage_preserved": self.required_coverage_preserved,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CodeProofPairedCase":
        if not isinstance(payload, Mapping):
            raise CodebaseProofBenchmarkError("paired case must be an object")
        if payload.get("schema") not in (None, CODEBASE_PROOF_PAIRED_CASE_SCHEMA):
            raise CodebaseProofBenchmarkError("unsupported paired case schema")
        return cls(
            task_reference=payload.get("task_reference", ""),
            claim_family=payload.get("claim_family", ""),
            bulk=payload.get("bulk") or {},
            obligation_first=payload.get("obligation_first") or {},
            warm_cache_dominated=payload.get("warm_cache_dominated", False),
        )


@dataclass(frozen=True)
class MutationSeedCase:
    """Held-out mutation/repair seed preregistered before outcome inspection."""

    seed_id: str
    kind: MutationSeedKind
    claim_family: ClaimFamily
    task_reference: str
    expected_false_admit: bool
    expected_false_refute: bool
    expected_stale_detection: bool
    expected_first_pass_success: bool
    expected_eventual_repair_success: bool
    expected_accepted_patch_regression: bool
    observed_false_admit: bool
    observed_false_refute: bool
    observed_stale_detection: bool
    observed_first_pass_success: bool
    observed_eventual_repair_success: bool
    observed_accepted_patch_regression: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "seed_id", _code(self.seed_id, "seed_id")
        )
        object.__setattr__(self, "kind", _enum(self.kind, MutationSeedKind, "kind"))
        object.__setattr__(
            self, "claim_family", _enum(self.claim_family, ClaimFamily, "claim_family")
        )
        object.__setattr__(
            self,
            "task_reference",
            _text(self.task_reference, "task_reference", maximum=MAX_REFERENCE_BYTES),
        )
        for name in (
            "expected_false_admit",
            "expected_false_refute",
            "expected_stale_detection",
            "expected_first_pass_success",
            "expected_eventual_repair_success",
            "expected_accepted_patch_regression",
            "observed_false_admit",
            "observed_false_refute",
            "observed_stale_detection",
            "observed_first_pass_success",
            "observed_eventual_repair_success",
            "observed_accepted_patch_regression",
        ):
            object.__setattr__(
                self, name, _boolean(getattr(self, name), name)
            )

    @property
    def seed_matched(self) -> bool:
        return (
            self.expected_false_admit == self.observed_false_admit
            and self.expected_false_refute == self.observed_false_refute
            and self.expected_stale_detection == self.observed_stale_detection
            and self.expected_first_pass_success == self.observed_first_pass_success
            and self.expected_eventual_repair_success
            == self.observed_eventual_repair_success
            and self.expected_accepted_patch_regression
            == self.observed_accepted_patch_regression
        )

    @property
    def content_id(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CODEBASE_PROOF_MUTATION_SCHEMA,
            "seed_id": self.seed_id,
            "kind": self.kind.value,
            "claim_family": self.claim_family.value,
            "task_reference": self.task_reference,
            "expected_false_admit": self.expected_false_admit,
            "expected_false_refute": self.expected_false_refute,
            "expected_stale_detection": self.expected_stale_detection,
            "expected_first_pass_success": self.expected_first_pass_success,
            "expected_eventual_repair_success": (
                self.expected_eventual_repair_success
            ),
            "expected_accepted_patch_regression": (
                self.expected_accepted_patch_regression
            ),
            "observed_false_admit": self.observed_false_admit,
            "observed_false_refute": self.observed_false_refute,
            "observed_stale_detection": self.observed_stale_detection,
            "observed_first_pass_success": self.observed_first_pass_success,
            "observed_eventual_repair_success": (
                self.observed_eventual_repair_success
            ),
            "observed_accepted_patch_regression": (
                self.observed_accepted_patch_regression
            ),
            "seed_matched": self.seed_matched,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MutationSeedCase":
        if not isinstance(payload, Mapping):
            raise CodebaseProofBenchmarkError("mutation seed must be an object")
        if payload.get("schema") not in (None, CODEBASE_PROOF_MUTATION_SCHEMA):
            raise CodebaseProofBenchmarkError("unsupported mutation seed schema")
        return cls(
            seed_id=payload.get("seed_id", ""),
            kind=payload.get("kind", ""),
            claim_family=payload.get("claim_family", ""),
            task_reference=payload.get("task_reference", ""),
            expected_false_admit=payload.get("expected_false_admit", False),
            expected_false_refute=payload.get("expected_false_refute", False),
            expected_stale_detection=payload.get("expected_stale_detection", False),
            expected_first_pass_success=payload.get(
                "expected_first_pass_success", False
            ),
            expected_eventual_repair_success=payload.get(
                "expected_eventual_repair_success", False
            ),
            expected_accepted_patch_regression=payload.get(
                "expected_accepted_patch_regression", False
            ),
            observed_false_admit=payload.get("observed_false_admit", False),
            observed_false_refute=payload.get("observed_false_refute", False),
            observed_stale_detection=payload.get("observed_stale_detection", False),
            observed_first_pass_success=payload.get(
                "observed_first_pass_success", False
            ),
            observed_eventual_repair_success=payload.get(
                "observed_eventual_repair_success", False
            ),
            observed_accepted_patch_regression=payload.get(
                "observed_accepted_patch_regression", False
            ),
        )


@dataclass(frozen=True)
class LiveModelChannelObservation:
    """Optional live-model measurements reported separately from fixture gates."""

    channel_id: str
    provider_reference: str
    paired_cases: tuple[CodeProofPairedCase, ...] = ()
    notes_digest: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "channel_id", _code(self.channel_id, "channel_id")
        )
        object.__setattr__(
            self,
            "provider_reference",
            _text(
                self.provider_reference,
                "provider_reference",
                maximum=MAX_REFERENCE_BYTES,
            ),
        )
        cases: list[CodeProofPairedCase] = []
        for item in self.paired_cases:
            if isinstance(item, CodeProofPairedCase):
                cases.append(item)
            elif isinstance(item, Mapping):
                cases.append(CodeProofPairedCase.from_dict(item))
            else:
                raise CodebaseProofBenchmarkError("invalid live paired case")
        for case in cases:
            if case.bulk.channel is not ResultChannel.LIVE_MODEL:
                raise CodebaseProofBenchmarkError(
                    "live channel bulk arms must use live_model channel"
                )
            if case.obligation_first.channel is not ResultChannel.LIVE_MODEL:
                raise CodebaseProofBenchmarkError(
                    "live channel obligation arms must use live_model channel"
                )
        object.__setattr__(self, "paired_cases", tuple(cases))
        if self.notes_digest:
            object.__setattr__(
                self, "notes_digest", _content_id(self.notes_digest, "notes_digest")
            )
        else:
            object.__setattr__(self, "notes_digest", "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CODEBASE_PROOF_LIVE_CHANNEL_SCHEMA,
            "channel_id": self.channel_id,
            "provider_reference": self.provider_reference,
            "paired_cases": [c.to_dict() for c in self.paired_cases],
            "notes_digest": self.notes_digest,
            "authoritative_for_fixture_gates": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LiveModelChannelObservation":
        if not isinstance(payload, Mapping):
            raise CodebaseProofBenchmarkError("live channel must be an object")
        if payload.get("schema") not in (None, CODEBASE_PROOF_LIVE_CHANNEL_SCHEMA):
            raise CodebaseProofBenchmarkError("unsupported live channel schema")
        return cls(
            channel_id=payload.get("channel_id", ""),
            provider_reference=payload.get("provider_reference", ""),
            paired_cases=tuple(payload.get("paired_cases") or ()),
            notes_digest=payload.get("notes_digest", "") or "",
        )


# ---------------------------------------------------------------------------
# Suite and report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CodebaseProofBenchmarkSuite:
    """Preregistered fixed baseline + held-out mutation suite.

    Constructed before outcome inspection.  Suite identity is content-addressed
    from the preregistered population, so post-hoc substitution fails closed.
    """

    corpus_version: str
    repository_id: str
    tree_id: str
    policy_id: str
    policy_revision: str
    objective_id: str
    objective_revision: str
    paired_cases: tuple[CodeProofPairedCase, ...]
    mutation_seeds: tuple[MutationSeedCase, ...]
    live_model_channel: LiveModelChannelObservation | None = None
    preregistration_digest: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "corpus_version",
            _text(self.corpus_version, "corpus_version", maximum=128),
        )
        for name in (
            "repository_id",
            "tree_id",
            "policy_id",
            "policy_revision",
            "objective_id",
            "objective_revision",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), name, maximum=MAX_REFERENCE_BYTES),
            )
        cases: list[CodeProofPairedCase] = []
        for item in self.paired_cases:
            if isinstance(item, CodeProofPairedCase):
                cases.append(item)
            elif isinstance(item, Mapping):
                cases.append(CodeProofPairedCase.from_dict(item))
            else:
                raise CodebaseProofBenchmarkError("invalid paired case")
        if len(cases) > MAX_CASES:
            raise CodebaseProofBenchmarkError("paired cases exceed bound")
        if not cases:
            raise CodebaseProofBenchmarkError("suite requires paired cases")
        tasks = [c.task_reference for c in cases]
        if len(tasks) != len(set(tasks)):
            raise CodebaseProofBenchmarkError("paired task references must be unique")
        for case in cases:
            if case.bulk.channel is not ResultChannel.DETERMINISTIC_FIXTURE:
                raise CodebaseProofBenchmarkError(
                    "suite paired bulk arms must be deterministic_fixture"
                )
            if (
                case.obligation_first.channel
                is not ResultChannel.DETERMINISTIC_FIXTURE
            ):
                raise CodebaseProofBenchmarkError(
                    "suite paired obligation arms must be deterministic_fixture"
                )
        object.__setattr__(
            self,
            "paired_cases",
            tuple(sorted(cases, key=lambda c: c.task_reference)),
        )

        seeds: list[MutationSeedCase] = []
        for item in self.mutation_seeds:
            if isinstance(item, MutationSeedCase):
                seeds.append(item)
            elif isinstance(item, Mapping):
                seeds.append(MutationSeedCase.from_dict(item))
            else:
                raise CodebaseProofBenchmarkError("invalid mutation seed")
        if len(seeds) > MAX_MUTATIONS:
            raise CodebaseProofBenchmarkError("mutation seeds exceed bound")
        if not seeds:
            raise CodebaseProofBenchmarkError("suite requires mutation seeds")
        seed_ids = [s.seed_id for s in seeds]
        if len(seed_ids) != len(set(seed_ids)):
            raise CodebaseProofBenchmarkError("mutation seed ids must be unique")
        object.__setattr__(
            self,
            "mutation_seeds",
            tuple(sorted(seeds, key=lambda s: s.seed_id)),
        )

        live = self.live_model_channel
        if live is not None and isinstance(live, Mapping):
            live = LiveModelChannelObservation.from_dict(live)
        if live is not None and not isinstance(live, LiveModelChannelObservation):
            raise CodebaseProofBenchmarkError("invalid live model channel")
        object.__setattr__(self, "live_model_channel", live)

        # Families present in the preregistered suite.
        families = {c.claim_family for c in self.paired_cases}
        missing = [f.value for f in REQUIRED_CLAIM_FAMILIES if f not in families]
        if missing:
            raise CodebaseProofBenchmarkError(
                "suite missing required claim families: " + ", ".join(missing)
            )

        material = self._preregistration_material()
        digest = _digest(material)
        if self.preregistration_digest:
            claimed = _content_id(
                self.preregistration_digest, "preregistration_digest"
            )
            if claimed != digest:
                raise CodebaseProofBenchmarkError(
                    "preregistration_digest does not match suite population"
                )
        object.__setattr__(self, "preregistration_digest", digest)
        encoded = _canonical_json(self.to_dict()).encode("utf-8")
        if len(encoded) > MAX_SUITE_BYTES:
            raise CodebaseProofBenchmarkError("suite exceeds serialized byte bound")

    def _preregistration_material(self) -> dict[str, Any]:
        return {
            "corpus_version": self.corpus_version,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "objective_id": self.objective_id,
            "objective_revision": self.objective_revision,
            "paired_cases": [c.to_dict() for c in self.paired_cases],
            "mutation_seeds": [s.to_dict() for s in self.mutation_seeds],
        }

    @property
    def suite_id(self) -> str:
        return self.preregistration_digest

    @property
    def interface(self) -> str:
        return CODEBASE_PROOF_BENCHMARK_INTERFACE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CODEBASE_PROOF_SUITE_SCHEMA,
            "interface": CODEBASE_PROOF_BENCHMARK_INTERFACE,
            "contract_version": CODEBASE_PROOF_BENCHMARK_VERSION,
            "corpus_version": self.corpus_version,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "objective_id": self.objective_id,
            "objective_revision": self.objective_revision,
            "paired_cases": [c.to_dict() for c in self.paired_cases],
            "mutation_seeds": [s.to_dict() for s in self.mutation_seeds],
            "live_model_channel": (
                self.live_model_channel.to_dict()
                if self.live_model_channel is not None
                else None
            ),
            "preregistration_digest": self.preregistration_digest,
            "suite_id": self.suite_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CodebaseProofBenchmarkSuite":
        if not isinstance(payload, Mapping):
            raise CodebaseProofBenchmarkError("suite must be an object")
        _reject_forbidden(payload)
        if payload.get("schema") not in (None, CODEBASE_PROOF_SUITE_SCHEMA):
            raise CodebaseProofBenchmarkError("unsupported suite schema")
        return cls(
            corpus_version=payload.get("corpus_version", ""),
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            policy_id=payload.get("policy_id", ""),
            policy_revision=payload.get("policy_revision", ""),
            objective_id=payload.get("objective_id", ""),
            objective_revision=payload.get("objective_revision", ""),
            paired_cases=tuple(payload.get("paired_cases") or ()),
            mutation_seeds=tuple(payload.get("mutation_seeds") or ()),
            live_model_channel=payload.get("live_model_channel"),
            preregistration_digest=payload.get("preregistration_digest", "") or "",
        )


@dataclass(frozen=True)
class CodeProofGateResult:
    name: CodeProofGateName
    passed: bool
    measured_bps: int
    threshold_bps: int
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "passed": self.passed,
            "measured_bps": self.measured_bps,
            "threshold_bps": self.threshold_bps,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class CodebaseProofBenchmarkReport:
    """Recomputed closed-loop quality / coverage / efficiency report."""

    suite_id: str
    corpus_version: str
    channel: ResultChannel
    # Lifecycle / claim status aggregates (fixture channel only).
    status_counts: Mapping[str, int]
    family_coverage: Mapping[str, Mapping[str, int]]
    evidence_tier_counts: Mapping[str, int]
    assurance_counts: Mapping[str, int]
    # Quality.
    false_admit_count: int
    false_refute_count: int
    false_admit_rate_bps: int
    false_refute_rate_bps: int
    stale_evidence_detected: int
    stale_evidence_expected: int
    first_pass_success_count: int
    first_pass_attempt_count: int
    first_pass_success_rate_bps: int
    eventual_repair_success_count: int
    eventual_repair_attempt_count: int
    eventual_repair_success_rate_bps: int
    accepted_patch_regression_count: int
    accepted_patch_regression_rate_bps: int
    # Efficiency aggregates (obligation-first vs bulk).
    bulk_input_tokens: int
    obligation_input_tokens: int
    bulk_retry_tokens: int
    obligation_retry_tokens: int
    bulk_provider_calls: int
    obligation_provider_calls: int
    bulk_cache_hits: int
    bulk_cache_rejects: int
    bulk_cache_lookups: int
    obligation_cache_hits: int
    obligation_cache_rejects: int
    obligation_cache_lookups: int
    bulk_wall_time_ms: int
    obligation_wall_time_ms: int
    bulk_proof_cost_microunits: int
    obligation_proof_cost_microunits: int
    bulk_accepted_criteria: int
    obligation_accepted_criteria: int
    input_tokens_per_accepted_criterion_bulk: int
    input_tokens_per_accepted_criterion_obligation: int
    input_token_reduction_bps: int
    tokens_per_criterion_reduction_bps: int
    retry_token_reduction_bps: int
    warm_prove_cost_reduction_bps: int
    cache_hit_rate_bps_obligation: int
    cache_reject_rate_bps_obligation: int
    required_coverage_loss_count: int
    mutation_seed_match_count: int
    mutation_seed_total: int
    gates: tuple[CodeProofGateResult, ...]
    efficiency_report: CodeProofEfficiencyReport | None = None
    live_model_summary: Mapping[str, Any] = field(default_factory=dict)
    report_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "suite_id", _content_id(self.suite_id, "suite_id")
        )
        object.__setattr__(
            self,
            "corpus_version",
            _text(self.corpus_version, "corpus_version", maximum=128),
        )
        object.__setattr__(
            self, "channel", _enum(self.channel, ResultChannel, "channel")
        )
        for name in (
            "false_admit_count",
            "false_refute_count",
            "false_admit_rate_bps",
            "false_refute_rate_bps",
            "stale_evidence_detected",
            "stale_evidence_expected",
            "first_pass_success_count",
            "first_pass_attempt_count",
            "first_pass_success_rate_bps",
            "eventual_repair_success_count",
            "eventual_repair_attempt_count",
            "eventual_repair_success_rate_bps",
            "accepted_patch_regression_count",
            "accepted_patch_regression_rate_bps",
            "bulk_input_tokens",
            "obligation_input_tokens",
            "bulk_retry_tokens",
            "obligation_retry_tokens",
            "bulk_provider_calls",
            "obligation_provider_calls",
            "bulk_cache_hits",
            "bulk_cache_rejects",
            "bulk_cache_lookups",
            "obligation_cache_hits",
            "obligation_cache_rejects",
            "obligation_cache_lookups",
            "bulk_wall_time_ms",
            "obligation_wall_time_ms",
            "bulk_proof_cost_microunits",
            "obligation_proof_cost_microunits",
            "bulk_accepted_criteria",
            "obligation_accepted_criteria",
            "input_tokens_per_accepted_criterion_bulk",
            "input_tokens_per_accepted_criterion_obligation",
            "input_token_reduction_bps",
            "tokens_per_criterion_reduction_bps",
            "retry_token_reduction_bps",
            "warm_prove_cost_reduction_bps",
            "cache_hit_rate_bps_obligation",
            "cache_reject_rate_bps_obligation",
            "required_coverage_loss_count",
            "mutation_seed_match_count",
            "mutation_seed_total",
        ):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name, maximum=MAX_COUNTER)
            )
        gates = tuple(self.gates)
        object.__setattr__(self, "gates", gates)
        if not isinstance(self.status_counts, Mapping):
            raise CodebaseProofBenchmarkError("status_counts must be a mapping")
        if not isinstance(self.family_coverage, Mapping):
            raise CodebaseProofBenchmarkError("family_coverage must be a mapping")
        if not isinstance(self.evidence_tier_counts, Mapping):
            raise CodebaseProofBenchmarkError(
                "evidence_tier_counts must be a mapping"
            )
        if not isinstance(self.assurance_counts, Mapping):
            raise CodebaseProofBenchmarkError("assurance_counts must be a mapping")
        if not isinstance(self.live_model_summary, Mapping):
            raise CodebaseProofBenchmarkError("live_model_summary must be a mapping")
        material = self._identity_material()
        digest = _digest(material)
        if self.report_id:
            claimed = _content_id(self.report_id, "report_id")
            if claimed != digest:
                raise CodebaseProofBenchmarkError(
                    "report_id does not match recomputed identity"
                )
        object.__setattr__(self, "report_id", digest)
        encoded = _canonical_json(self.to_dict()).encode("utf-8")
        if len(encoded) > MAX_REPORT_BYTES:
            raise CodebaseProofBenchmarkError("report exceeds serialized byte bound")

    def _identity_material(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "corpus_version": self.corpus_version,
            "channel": self.channel.value,
            "status_counts": dict(self.status_counts),
            "family_coverage": {
                k: dict(v) for k, v in self.family_coverage.items()
            },
            "evidence_tier_counts": dict(self.evidence_tier_counts),
            "assurance_counts": dict(self.assurance_counts),
            "false_admit_count": self.false_admit_count,
            "false_refute_count": self.false_refute_count,
            "stale_evidence_detected": self.stale_evidence_detected,
            "stale_evidence_expected": self.stale_evidence_expected,
            "first_pass_success_count": self.first_pass_success_count,
            "eventual_repair_success_count": self.eventual_repair_success_count,
            "accepted_patch_regression_count": self.accepted_patch_regression_count,
            "bulk_input_tokens": self.bulk_input_tokens,
            "obligation_input_tokens": self.obligation_input_tokens,
            "bulk_retry_tokens": self.bulk_retry_tokens,
            "obligation_retry_tokens": self.obligation_retry_tokens,
            "tokens_per_criterion_reduction_bps": (
                self.tokens_per_criterion_reduction_bps
            ),
            "retry_token_reduction_bps": self.retry_token_reduction_bps,
            "warm_prove_cost_reduction_bps": self.warm_prove_cost_reduction_bps,
            "required_coverage_loss_count": self.required_coverage_loss_count,
            "gates": [g.to_dict() for g in self.gates],
            "efficiency_report_id": (
                self.efficiency_report.report_id
                if self.efficiency_report is not None
                else ""
            ),
            "live_model_summary": dict(self.live_model_summary),
        }

    @property
    def all_gates_passed(self) -> bool:
        return bool(self.gates) and all(g.passed for g in self.gates)

    @property
    def fixture_gates_authoritative(self) -> bool:
        return self.channel is ResultChannel.DETERMINISTIC_FIXTURE

    @property
    def evidence_claim_references(self) -> tuple[str, ...]:
        if not self.all_gates_passed:
            return ()
        return (CODEBASE_PROOF_EFFICIENCY_REQUIREMENT_ID,)

    @property
    def passed(self) -> bool:
        return self.all_gates_passed and self.fixture_gates_authoritative

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CODEBASE_PROOF_REPORT_SCHEMA,
            "interface": CODEBASE_PROOF_BENCHMARK_INTERFACE,
            "contract_version": CODEBASE_PROOF_BENCHMARK_VERSION,
            "suite_id": self.suite_id,
            "corpus_version": self.corpus_version,
            "channel": self.channel.value,
            "status_counts": dict(self.status_counts),
            "family_coverage": {
                k: dict(v) for k, v in self.family_coverage.items()
            },
            "evidence_tier_counts": dict(self.evidence_tier_counts),
            "assurance_counts": dict(self.assurance_counts),
            "false_admit_count": self.false_admit_count,
            "false_refute_count": self.false_refute_count,
            "false_admit_rate_bps": self.false_admit_rate_bps,
            "false_refute_rate_bps": self.false_refute_rate_bps,
            "stale_evidence_detected": self.stale_evidence_detected,
            "stale_evidence_expected": self.stale_evidence_expected,
            "first_pass_success_count": self.first_pass_success_count,
            "first_pass_attempt_count": self.first_pass_attempt_count,
            "first_pass_success_rate_bps": self.first_pass_success_rate_bps,
            "eventual_repair_success_count": self.eventual_repair_success_count,
            "eventual_repair_attempt_count": self.eventual_repair_attempt_count,
            "eventual_repair_success_rate_bps": (
                self.eventual_repair_success_rate_bps
            ),
            "accepted_patch_regression_count": (
                self.accepted_patch_regression_count
            ),
            "accepted_patch_regression_rate_bps": (
                self.accepted_patch_regression_rate_bps
            ),
            "bulk_input_tokens": self.bulk_input_tokens,
            "obligation_input_tokens": self.obligation_input_tokens,
            "bulk_retry_tokens": self.bulk_retry_tokens,
            "obligation_retry_tokens": self.obligation_retry_tokens,
            "bulk_provider_calls": self.bulk_provider_calls,
            "obligation_provider_calls": self.obligation_provider_calls,
            "bulk_cache_hits": self.bulk_cache_hits,
            "bulk_cache_rejects": self.bulk_cache_rejects,
            "bulk_cache_lookups": self.bulk_cache_lookups,
            "obligation_cache_hits": self.obligation_cache_hits,
            "obligation_cache_rejects": self.obligation_cache_rejects,
            "obligation_cache_lookups": self.obligation_cache_lookups,
            "bulk_wall_time_ms": self.bulk_wall_time_ms,
            "obligation_wall_time_ms": self.obligation_wall_time_ms,
            "bulk_proof_cost_microunits": self.bulk_proof_cost_microunits,
            "obligation_proof_cost_microunits": (
                self.obligation_proof_cost_microunits
            ),
            "bulk_accepted_criteria": self.bulk_accepted_criteria,
            "obligation_accepted_criteria": self.obligation_accepted_criteria,
            "input_tokens_per_accepted_criterion_bulk": (
                self.input_tokens_per_accepted_criterion_bulk
            ),
            "input_tokens_per_accepted_criterion_obligation": (
                self.input_tokens_per_accepted_criterion_obligation
            ),
            "input_token_reduction_bps": self.input_token_reduction_bps,
            "tokens_per_criterion_reduction_bps": (
                self.tokens_per_criterion_reduction_bps
            ),
            "retry_token_reduction_bps": self.retry_token_reduction_bps,
            "warm_prove_cost_reduction_bps": self.warm_prove_cost_reduction_bps,
            "cache_hit_rate_bps_obligation": self.cache_hit_rate_bps_obligation,
            "cache_reject_rate_bps_obligation": (
                self.cache_reject_rate_bps_obligation
            ),
            "required_coverage_loss_count": self.required_coverage_loss_count,
            "mutation_seed_match_count": self.mutation_seed_match_count,
            "mutation_seed_total": self.mutation_seed_total,
            "gates": [g.to_dict() for g in self.gates],
            "all_gates_passed": self.all_gates_passed,
            "fixture_gates_authoritative": self.fixture_gates_authoritative,
            "passed": self.passed,
            "evidence_claim_references": list(self.evidence_claim_references),
            "efficiency_report": (
                self.efficiency_report.to_dict()
                if self.efficiency_report is not None
                else None
            ),
            "live_model_summary": dict(self.live_model_summary),
            "report_id": self.report_id,
        }


def evaluate_codebase_proof_benchmark(
    suite: CodebaseProofBenchmarkSuite | Mapping[str, Any],
) -> CodebaseProofBenchmarkReport:
    """Recompute every metric and gate from a preregistered suite."""

    if isinstance(suite, Mapping):
        suite = CodebaseProofBenchmarkSuite.from_dict(suite)
    if not isinstance(suite, CodebaseProofBenchmarkSuite):
        raise CodebaseProofBenchmarkError("suite is required")

    status_counts: dict[str, int] = {s.value: 0 for s in ClaimStatus}
    family_coverage: dict[str, dict[str, int]] = {
        f.value: {
            "required": 0,
            "satisfied": 0,
            "refuted": 0,
            "open": 0,
            "unsupported": 0,
            "not_measured": 0,
            "stale": 0,
            "unknown": 0,
        }
        for f in REQUIRED_CLAIM_FAMILIES
    }
    evidence_tier_counts: dict[str, int] = {t.value: 0 for t in EvidenceTier}
    assurance_counts: dict[str, int] = {a.value: 0 for a in AssuranceLevel}

    false_admit = 0
    false_refute = 0
    claim_total = 0
    first_pass_success = 0
    first_pass_attempts = 0
    eventual_success = 0
    eventual_attempts = 0
    regressions = 0
    regression_denom = 0

    bulk_input = 0
    obl_input = 0
    bulk_retry = 0
    obl_retry = 0
    bulk_calls = 0
    obl_calls = 0
    bulk_hits = 0
    bulk_rejects = 0
    bulk_lookups = 0
    obl_hits = 0
    obl_rejects = 0
    obl_lookups = 0
    bulk_wall = 0
    obl_wall = 0
    bulk_proof = 0
    obl_proof = 0
    bulk_criteria = 0
    obl_criteria = 0
    coverage_loss = 0

    warm_bulk_proof = 0
    warm_obl_proof = 0
    warm_cases = 0

    for case in suite.paired_cases:
        for arm in (case.bulk, case.obligation_first):
            first_pass_attempts += 1
            if arm.first_pass_success:
                first_pass_success += 1
            eventual_attempts += 1
            if arm.eventual_repair_success:
                eventual_success += 1
            regression_denom += 1
            if arm.accepted_patch_regression:
                regressions += 1
            for claim in arm.claims:
                claim_total += 1
                status_counts[claim.status.value] = (
                    status_counts.get(claim.status.value, 0) + 1
                )
                evidence_tier_counts[claim.evidence_tier.value] = (
                    evidence_tier_counts.get(claim.evidence_tier.value, 0) + 1
                )
                assurance_counts[claim.required_assurance.value] = (
                    assurance_counts.get(claim.required_assurance.value, 0) + 1
                )
                if claim.false_admit:
                    false_admit += 1
                if claim.false_refute:
                    false_refute += 1
                fam = claim.claim_family.value
                if fam in family_coverage:
                    bucket = family_coverage[fam]
                    if claim.required_for_coverage:
                        bucket["required"] += 1
                    key = claim.status.value
                    if key in bucket:
                        bucket[key] += 1
        bulk_input += case.bulk.input_tokens
        obl_input += case.obligation_first.input_tokens
        bulk_retry += case.bulk.retry_tokens
        obl_retry += case.obligation_first.retry_tokens
        bulk_calls += case.bulk.provider_calls
        obl_calls += case.obligation_first.provider_calls
        bulk_hits += case.bulk.cache_hits
        bulk_rejects += case.bulk.cache_rejects
        bulk_lookups += case.bulk.cache_lookups
        obl_hits += case.obligation_first.cache_hits
        obl_rejects += case.obligation_first.cache_rejects
        obl_lookups += case.obligation_first.cache_lookups
        bulk_wall += case.bulk.wall_time_ms
        obl_wall += case.obligation_first.wall_time_ms
        bulk_proof += case.bulk.proof_cost_microunits
        obl_proof += case.obligation_first.proof_cost_microunits
        bulk_criteria += case.bulk.accepted_criteria
        obl_criteria += case.obligation_first.accepted_criteria
        if not case.required_coverage_preserved:
            coverage_loss += 1
        if case.warm_cache_dominated:
            warm_cases += 1
            warm_bulk_proof += case.bulk.proof_cost_microunits
            warm_obl_proof += case.obligation_first.proof_cost_microunits

    stale_expected = sum(
        1
        for s in suite.mutation_seeds
        if s.kind is MutationSeedKind.STALE_EVIDENCE and s.expected_stale_detection
    )
    stale_detected = sum(
        1
        for s in suite.mutation_seeds
        if s.kind is MutationSeedKind.STALE_EVIDENCE and s.observed_stale_detection
    )
    mutation_match = sum(1 for s in suite.mutation_seeds if s.seed_matched)
    mutation_total = len(suite.mutation_seeds)

    tokens_per_bulk = bulk_input // bulk_criteria if bulk_criteria else bulk_input
    tokens_per_obl = obl_input // obl_criteria if obl_criteria else obl_input
    token_reduction = _reduction_bps(bulk_input, obl_input)
    criterion_reduction = _reduction_bps(tokens_per_bulk, tokens_per_obl)
    retry_reduction = _reduction_bps(bulk_retry, obl_retry)
    warm_prove_reduction = (
        _reduction_bps(warm_bulk_proof, warm_obl_proof) if warm_cases else 0
    )

    # Efficiency extension over typed receipts (fail-closed measurement).
    efficiency = build_code_proof_efficiency_report(
        build_code_proof_paired_receipts(suite.paired_cases)
    )

    gates = _evaluate_gates(
        false_admit_count=false_admit,
        coverage_loss_count=coverage_loss,
        tokens_per_criterion_reduction_bps=criterion_reduction,
        retry_token_reduction_bps=retry_reduction,
        warm_prove_cost_reduction_bps=warm_prove_reduction,
        warm_cases=warm_cases,
        families_present={c.claim_family for c in suite.paired_cases},
        live_affects_gates=False,
    )

    live_summary: dict[str, Any] = {
        "present": suite.live_model_channel is not None,
        "authoritative_for_fixture_gates": False,
        "case_count": (
            len(suite.live_model_channel.paired_cases)
            if suite.live_model_channel is not None
            else 0
        ),
    }
    if suite.live_model_channel is not None:
        live_summary["channel_id"] = suite.live_model_channel.channel_id
        live_summary["provider_reference"] = (
            suite.live_model_channel.provider_reference
        )

    return CodebaseProofBenchmarkReport(
        suite_id=suite.suite_id,
        corpus_version=suite.corpus_version,
        channel=ResultChannel.DETERMINISTIC_FIXTURE,
        status_counts=status_counts,
        family_coverage=family_coverage,
        evidence_tier_counts=evidence_tier_counts,
        assurance_counts=assurance_counts,
        false_admit_count=false_admit,
        false_refute_count=false_refute,
        false_admit_rate_bps=_rate_bps(false_admit, claim_total),
        false_refute_rate_bps=_rate_bps(false_refute, claim_total),
        stale_evidence_detected=stale_detected,
        stale_evidence_expected=stale_expected,
        first_pass_success_count=first_pass_success,
        first_pass_attempt_count=first_pass_attempts,
        first_pass_success_rate_bps=_rate_bps(
            first_pass_success, first_pass_attempts
        ),
        eventual_repair_success_count=eventual_success,
        eventual_repair_attempt_count=eventual_attempts,
        eventual_repair_success_rate_bps=_rate_bps(
            eventual_success, eventual_attempts
        ),
        accepted_patch_regression_count=regressions,
        accepted_patch_regression_rate_bps=_rate_bps(
            regressions, regression_denom
        ),
        bulk_input_tokens=bulk_input,
        obligation_input_tokens=obl_input,
        bulk_retry_tokens=bulk_retry,
        obligation_retry_tokens=obl_retry,
        bulk_provider_calls=bulk_calls,
        obligation_provider_calls=obl_calls,
        bulk_cache_hits=bulk_hits,
        bulk_cache_rejects=bulk_rejects,
        bulk_cache_lookups=bulk_lookups,
        obligation_cache_hits=obl_hits,
        obligation_cache_rejects=obl_rejects,
        obligation_cache_lookups=obl_lookups,
        bulk_wall_time_ms=bulk_wall,
        obligation_wall_time_ms=obl_wall,
        bulk_proof_cost_microunits=bulk_proof,
        obligation_proof_cost_microunits=obl_proof,
        bulk_accepted_criteria=bulk_criteria,
        obligation_accepted_criteria=obl_criteria,
        input_tokens_per_accepted_criterion_bulk=tokens_per_bulk,
        input_tokens_per_accepted_criterion_obligation=tokens_per_obl,
        input_token_reduction_bps=token_reduction,
        tokens_per_criterion_reduction_bps=criterion_reduction,
        retry_token_reduction_bps=retry_reduction,
        warm_prove_cost_reduction_bps=warm_prove_reduction,
        cache_hit_rate_bps_obligation=_rate_bps(obl_hits, obl_lookups),
        cache_reject_rate_bps_obligation=_rate_bps(obl_rejects, obl_lookups),
        required_coverage_loss_count=coverage_loss,
        mutation_seed_match_count=mutation_match,
        mutation_seed_total=mutation_total,
        gates=gates,
        efficiency_report=efficiency,
        live_model_summary=live_summary,
    )


def _evaluate_gates(
    *,
    false_admit_count: int,
    coverage_loss_count: int,
    tokens_per_criterion_reduction_bps: int,
    retry_token_reduction_bps: int,
    warm_prove_cost_reduction_bps: int,
    warm_cases: int,
    families_present: set[ClaimFamily],
    live_affects_gates: bool,
) -> tuple[CodeProofGateResult, ...]:
    family_ok = all(f in families_present for f in REQUIRED_CLAIM_FAMILIES)
    warm_ok = warm_cases > 0 and warm_prove_cost_reduction_bps > 0
    return (
        CodeProofGateResult(
            name=CodeProofGateName.ZERO_FALSE_AUTHORITATIVE_ADMISSIONS,
            passed=false_admit_count <= MAX_FALSE_AUTHORITATIVE_ADMISSIONS,
            measured_bps=false_admit_count,
            threshold_bps=MAX_FALSE_AUTHORITATIVE_ADMISSIONS,
            detail="false authoritative admissions in fixture suite",
        ),
        CodeProofGateResult(
            name=CodeProofGateName.NO_REQUIRED_COVERAGE_LOSS,
            passed=coverage_loss_count == 0,
            measured_bps=coverage_loss_count,
            threshold_bps=0,
            detail="paired cases with required-coverage loss",
        ),
        CodeProofGateResult(
            name=CodeProofGateName.INPUT_TOKEN_REDUCTION,
            passed=(
                tokens_per_criterion_reduction_bps
                >= MIN_INPUT_TOKEN_REDUCTION_BPS
            ),
            measured_bps=tokens_per_criterion_reduction_bps,
            threshold_bps=MIN_INPUT_TOKEN_REDUCTION_BPS,
            detail="input tokens per accepted criterion reduction",
        ),
        CodeProofGateResult(
            name=CodeProofGateName.RETRY_TOKEN_REDUCTION,
            passed=retry_token_reduction_bps >= MIN_RETRY_TOKEN_REDUCTION_BPS,
            measured_bps=retry_token_reduction_bps,
            threshold_bps=MIN_RETRY_TOKEN_REDUCTION_BPS,
            detail="retry token reduction bulk→obligation_first",
        ),
        CodeProofGateResult(
            name=CodeProofGateName.WARM_PROVE_COST_IMPROVEMENT,
            passed=warm_ok,
            measured_bps=warm_prove_cost_reduction_bps,
            threshold_bps=1,
            detail="warm prove cost improvement when cache hits dominate",
        ),
        CodeProofGateResult(
            name=CodeProofGateName.REQUIRED_FAMILY_COVERAGE,
            passed=family_ok,
            measured_bps=len(families_present),
            threshold_bps=len(REQUIRED_CLAIM_FAMILIES),
            detail="required claim families present in suite",
        ),
        CodeProofGateResult(
            name=CodeProofGateName.FIXTURE_CHANNEL_ISOLATION,
            passed=not live_affects_gates,
            measured_bps=0 if not live_affects_gates else 1,
            threshold_bps=0,
            detail="live-model results isolated from fixture gates",
        ),
    )


def verify_codebase_proof_benchmark_report(
    report: CodebaseProofBenchmarkReport | Mapping[str, Any],
    suite: CodebaseProofBenchmarkSuite | Mapping[str, Any],
) -> bool:
    """Replay the suite and require bit-identical recomputed identity."""

    if isinstance(suite, Mapping):
        suite = CodebaseProofBenchmarkSuite.from_dict(suite)
    recomputed = evaluate_codebase_proof_benchmark(suite)
    expected = recomputed.to_dict()
    if isinstance(report, Mapping):
        claimed = dict(report)
    else:
        claimed = report.to_dict()
    # Identity-bearing counters must match the suite recompute.  A forged
    # report_id alone cannot launder altered false-admit or token totals.
    identity_keys = (
        "report_id",
        "suite_id",
        "passed",
        "false_admit_count",
        "false_refute_count",
        "required_coverage_loss_count",
        "tokens_per_criterion_reduction_bps",
        "retry_token_reduction_bps",
        "warm_prove_cost_reduction_bps",
        "bulk_input_tokens",
        "obligation_input_tokens",
        "bulk_retry_tokens",
        "obligation_retry_tokens",
        "stale_evidence_detected",
        "accepted_patch_regression_count",
    )
    return all(claimed.get(key) == expected.get(key) for key in identity_keys)


# ---------------------------------------------------------------------------
# Preregistered fixtures (built before outcome inspection)
# ---------------------------------------------------------------------------


def _claim(
    *,
    ref: str,
    family: ClaimFamily,
    status: ClaimStatus,
    tier: EvidenceTier = EvidenceTier.KERNEL_PROOF,
    assurance: AssuranceLevel = AssuranceLevel.KERNEL_VERIFIED,
    authoritative: bool = True,
    false_admit: bool = False,
    false_refute: bool = False,
    required: bool = True,
) -> ClaimOutcomeObservation:
    return ClaimOutcomeObservation(
        claim_reference=ref,
        claim_family=family,
        evidence_tier=tier,
        required_assurance=assurance,
        status=status,
        authoritative_admission=authoritative and status is ClaimStatus.SATISFIED,
        false_admit=false_admit,
        false_refute=false_refute,
        required_for_coverage=required,
    )


def _arm(
    *,
    task: str,
    path: ContextPath,
    family: ClaimFamily,
    input_tokens: int,
    retry_tokens: int,
    proof_cost: int,
    claims: Sequence[ClaimOutcomeObservation],
    cache_lookups: int = 4,
    cache_hits: int = 0,
    cache_rejects: int = 0,
    provider_calls: int = 2,
    wall_time_ms: int = 8_000,
    output_tokens: int = 400,
    accepted_criteria: int = 1,
    first_pass: bool = True,
    eventual: bool = True,
    regression: bool = False,
    channel: ResultChannel = ResultChannel.DETERMINISTIC_FIXTURE,
) -> CodeProofArmObservation:
    return CodeProofArmObservation(
        task_reference=task,
        path=path,
        channel=channel,
        claim_family=family,
        input_tokens=input_tokens,
        retry_tokens=retry_tokens,
        output_tokens=output_tokens,
        provider_calls=provider_calls,
        cache_lookups=cache_lookups,
        cache_hits=cache_hits,
        cache_rejects=cache_rejects,
        wall_time_ms=wall_time_ms,
        proof_cost_microunits=proof_cost,
        accepted_criteria=accepted_criteria,
        first_pass_success=first_pass,
        eventual_repair_success=eventual,
        accepted_patch_regression=regression,
        claims=tuple(claims),
    )


def _paired(
    *,
    family: ClaimFamily,
    slug: str,
    bulk_input: int = 10_000,
    obl_input: int = 5_000,
    bulk_retry: int = 5_000,
    obl_retry: int = 1_500,
    bulk_proof: int = 2_000,
    obl_proof: int = 1_000,
    warm: bool = False,
    obl_hits: int = 1,
    bulk_hits: int = 0,
    extra_statuses: Sequence[ClaimStatus] = (),
) -> CodeProofPairedCase:
    task = f"task:cbp130:{slug}"
    base_claims = [
        _claim(
            ref=f"claim:{slug}:primary",
            family=family,
            status=ClaimStatus.SATISFIED,
        ),
    ]
    # Include lifecycle-state variety without harming required coverage.
    status_cycle = list(extra_statuses) or [
        ClaimStatus.OPEN,
        ClaimStatus.UNSUPPORTED,
        ClaimStatus.NOT_MEASURED,
        ClaimStatus.STALE,
        ClaimStatus.REFUTED,
    ]
    for index, status in enumerate(status_cycle[:3]):
        base_claims.append(
            _claim(
                ref=f"claim:{slug}:aux:{index}",
                family=family,
                status=status,
                tier=(
                    EvidenceTier.OBSERVATION
                    if status is not ClaimStatus.SATISFIED
                    else EvidenceTier.KERNEL_PROOF
                ),
                authoritative=status is ClaimStatus.SATISFIED,
                false_refute=False,
                required=False,
            )
        )
    # Obligation-first preserves the required primary claim.
    obl_claims = list(base_claims)
    return CodeProofPairedCase(
        task_reference=task,
        claim_family=family,
        warm_cache_dominated=warm,
        bulk=_arm(
            task=task,
            path=ContextPath.BULK_SOURCE,
            family=family,
            input_tokens=bulk_input,
            retry_tokens=bulk_retry,
            proof_cost=bulk_proof,
            claims=base_claims,
            cache_hits=bulk_hits,
            cache_lookups=6,
            cache_rejects=1,
            wall_time_ms=12_000 if not warm else 10_000,
            accepted_criteria=2,
            provider_calls=4,
        ),
        obligation_first=_arm(
            task=task,
            path=ContextPath.OBLIGATION_FIRST,
            family=family,
            input_tokens=obl_input,
            retry_tokens=obl_retry,
            proof_cost=obl_proof,
            claims=obl_claims,
            cache_hits=obl_hits if not warm else 5,
            cache_lookups=6,
            cache_rejects=0 if warm else 1,
            wall_time_ms=5_000 if not warm else 3_500,
            accepted_criteria=2,
            provider_calls=2,
        ),
    )


def build_preregistered_codebase_proof_suite(
    *,
    include_live_model_channel: bool = False,
) -> CodebaseProofBenchmarkSuite:
    """Preregister the fixed baseline and held-out mutation/repair suite.

    This function freezes the population *before* outcome inspection.  Gate
    evaluation is a pure recompute over the returned suite.
    """

    paired = (
        _paired(
            family=ClaimFamily.DEPENDENCY_REACHABILITY,
            slug="dependency",
            extra_statuses=(ClaimStatus.OPEN, ClaimStatus.STALE),
        ),
        _paired(
            family=ClaimFamily.API_CONTRACT,
            slug="api-contract",
            bulk_input=12_000,
            obl_input=6_000,
            bulk_retry=6_000,
            obl_retry=2_000,
        ),
        _paired(
            family=ClaimFamily.BEHAVIORAL_INVARIANT,
            slug="behavioral",
            bulk_input=9_000,
            obl_input=4_500,
            bulk_retry=4_500,
            obl_retry=1_200,
        ),
        _paired(
            family=ClaimFamily.SECURITY_PROPERTY,
            slug="security",
            bulk_input=11_000,
            obl_input=5_200,
            bulk_retry=5_500,
            obl_retry=1_800,
            extra_statuses=(ClaimStatus.REFUTED, ClaimStatus.UNSUPPORTED),
        ),
        _paired(
            family=ClaimFamily.SEMANTIC_EQUIVALENCE,
            slug="semantic-eq",
            bulk_input=10_500,
            obl_input=5_000,
            bulk_retry=4_800,
            obl_retry=1_400,
        ),
        _paired(
            family=ClaimFamily.SUPERVISOR_LIFECYCLE,
            slug="supervisor-lifecycle",
            bulk_input=10_000,
            obl_input=4_800,
            bulk_retry=5_200,
            obl_retry=1_600,
            extra_statuses=(ClaimStatus.NOT_MEASURED, ClaimStatus.OPEN),
        ),
        # Warm-cache dominated case: obligation-first reuses proof cache.
        _paired(
            family=ClaimFamily.API_CONTRACT,
            slug="warm-cache",
            bulk_input=8_000,
            obl_input=3_500,
            bulk_retry=3_000,
            obl_retry=800,
            bulk_proof=5_000,
            obl_proof=900,
            warm=True,
            obl_hits=5,
            bulk_hits=0,
        ),
    )

    # Held-out mutation seeds: expectations are fixed with the suite; observed
    # values mirror the deterministic fixture outcomes (no false admits).
    mutation_seeds = (
        MutationSeedCase(
            seed_id="mut.false_admit.security",
            kind=MutationSeedKind.FALSE_ADMIT,
            claim_family=ClaimFamily.SECURITY_PROPERTY,
            task_reference="task:cbp130:security",
            expected_false_admit=False,
            expected_false_refute=False,
            expected_stale_detection=False,
            expected_first_pass_success=True,
            expected_eventual_repair_success=True,
            expected_accepted_patch_regression=False,
            observed_false_admit=False,
            observed_false_refute=False,
            observed_stale_detection=False,
            observed_first_pass_success=True,
            observed_eventual_repair_success=True,
            observed_accepted_patch_regression=False,
        ),
        MutationSeedCase(
            seed_id="mut.false_refute.behavioral",
            kind=MutationSeedKind.FALSE_REFUTE,
            claim_family=ClaimFamily.BEHAVIORAL_INVARIANT,
            task_reference="task:cbp130:behavioral",
            expected_false_admit=False,
            expected_false_refute=False,
            expected_stale_detection=False,
            expected_first_pass_success=True,
            expected_eventual_repair_success=True,
            expected_accepted_patch_regression=False,
            observed_false_admit=False,
            observed_false_refute=False,
            observed_stale_detection=False,
            observed_first_pass_success=True,
            observed_eventual_repair_success=True,
            observed_accepted_patch_regression=False,
        ),
        MutationSeedCase(
            seed_id="mut.stale.dependency",
            kind=MutationSeedKind.STALE_EVIDENCE,
            claim_family=ClaimFamily.DEPENDENCY_REACHABILITY,
            task_reference="task:cbp130:dependency",
            expected_false_admit=False,
            expected_false_refute=False,
            expected_stale_detection=True,
            expected_first_pass_success=True,
            expected_eventual_repair_success=True,
            expected_accepted_patch_regression=False,
            observed_false_admit=False,
            observed_false_refute=False,
            observed_stale_detection=True,
            observed_first_pass_success=True,
            observed_eventual_repair_success=True,
            observed_accepted_patch_regression=False,
        ),
        MutationSeedCase(
            seed_id="mut.repair.first_pass",
            kind=MutationSeedKind.FIRST_PASS_REPAIR,
            claim_family=ClaimFamily.API_CONTRACT,
            task_reference="task:cbp130:api-contract",
            expected_false_admit=False,
            expected_false_refute=False,
            expected_stale_detection=False,
            expected_first_pass_success=True,
            expected_eventual_repair_success=True,
            expected_accepted_patch_regression=False,
            observed_false_admit=False,
            observed_false_refute=False,
            observed_stale_detection=False,
            observed_first_pass_success=True,
            observed_eventual_repair_success=True,
            observed_accepted_patch_regression=False,
        ),
        MutationSeedCase(
            seed_id="mut.repair.eventual",
            kind=MutationSeedKind.EVENTUAL_REPAIR,
            claim_family=ClaimFamily.SEMANTIC_EQUIVALENCE,
            task_reference="task:cbp130:semantic-eq",
            expected_false_admit=False,
            expected_false_refute=False,
            expected_stale_detection=False,
            expected_first_pass_success=True,
            expected_eventual_repair_success=True,
            expected_accepted_patch_regression=False,
            observed_false_admit=False,
            observed_false_refute=False,
            observed_stale_detection=False,
            observed_first_pass_success=True,
            observed_eventual_repair_success=True,
            observed_accepted_patch_regression=False,
        ),
        MutationSeedCase(
            seed_id="mut.regression.lifecycle",
            kind=MutationSeedKind.ACCEPTED_PATCH_REGRESSION,
            claim_family=ClaimFamily.SUPERVISOR_LIFECYCLE,
            task_reference="task:cbp130:supervisor-lifecycle",
            expected_false_admit=False,
            expected_false_refute=False,
            expected_stale_detection=False,
            expected_first_pass_success=True,
            expected_eventual_repair_success=True,
            expected_accepted_patch_regression=False,
            observed_false_admit=False,
            observed_false_refute=False,
            observed_stale_detection=False,
            observed_first_pass_success=True,
            observed_eventual_repair_success=True,
            observed_accepted_patch_regression=False,
        ),
        MutationSeedCase(
            seed_id="mut.warm.cache",
            kind=MutationSeedKind.WARM_CACHE_DOMINATED,
            claim_family=ClaimFamily.API_CONTRACT,
            task_reference="task:cbp130:warm-cache",
            expected_false_admit=False,
            expected_false_refute=False,
            expected_stale_detection=False,
            expected_first_pass_success=True,
            expected_eventual_repair_success=True,
            expected_accepted_patch_regression=False,
            observed_false_admit=False,
            observed_false_refute=False,
            observed_stale_detection=False,
            observed_first_pass_success=True,
            observed_eventual_repair_success=True,
            observed_accepted_patch_regression=False,
        ),
        MutationSeedCase(
            seed_id="mut.coverage.required",
            kind=MutationSeedKind.REQUIRED_COVERAGE,
            claim_family=ClaimFamily.SECURITY_PROPERTY,
            task_reference="task:cbp130:security",
            expected_false_admit=False,
            expected_false_refute=False,
            expected_stale_detection=False,
            expected_first_pass_success=True,
            expected_eventual_repair_success=True,
            expected_accepted_patch_regression=False,
            observed_false_admit=False,
            observed_false_refute=False,
            observed_stale_detection=False,
            observed_first_pass_success=True,
            observed_eventual_repair_success=True,
            observed_accepted_patch_regression=False,
        ),
    )

    live: LiveModelChannelObservation | None = None
    if include_live_model_channel:
        # Live channel is present for reporting but never authoritative.
        live_task = "task:cbp130:live-sample"
        live_family = ClaimFamily.API_CONTRACT
        live_claim = _claim(
            ref="claim:live:primary",
            family=live_family,
            status=ClaimStatus.SATISFIED,
        )
        live = LiveModelChannelObservation(
            channel_id="live.model.sample",
            provider_reference="provider:live-optional@1",
            paired_cases=(
                CodeProofPairedCase(
                    task_reference=live_task,
                    claim_family=live_family,
                    bulk=_arm(
                        task=live_task,
                        path=ContextPath.BULK_SOURCE,
                        family=live_family,
                        input_tokens=20_000,
                        retry_tokens=8_000,
                        proof_cost=4_000,
                        claims=(live_claim,),
                        channel=ResultChannel.LIVE_MODEL,
                        accepted_criteria=1,
                    ),
                    obligation_first=_arm(
                        task=live_task,
                        path=ContextPath.OBLIGATION_FIRST,
                        family=live_family,
                        input_tokens=9_000,
                        retry_tokens=2_500,
                        proof_cost=1_500,
                        claims=(live_claim,),
                        channel=ResultChannel.LIVE_MODEL,
                        accepted_criteria=1,
                        cache_hits=3,
                        cache_lookups=4,
                    ),
                ),
            ),
            notes_digest=_fixture_digest("live-model-notes-v1"),
        )

    return CodebaseProofBenchmarkSuite(
        corpus_version=CODEBASE_PROOF_BENCHMARK_CORPUS_VERSION,
        repository_id=CBP_FROZEN_REPOSITORY_ID,
        tree_id=CBP_FROZEN_TREE_ID,
        policy_id=CBP_FROZEN_POLICY_ID,
        policy_revision=CBP_FROZEN_POLICY_REVISION,
        objective_id=CODEBASE_PROOF_OBJECTIVE_ID,
        objective_revision=CODEBASE_PROOF_OBJECTIVE_REVISION,
        paired_cases=paired,
        mutation_seeds=mutation_seeds,
        live_model_channel=live,
    )


def run_codebase_proof_efficiency_gates(
    *,
    include_live_model_channel: bool = False,
) -> CodebaseProofBenchmarkReport:
    """Build the preregistered suite and evaluate deterministic fixture gates."""

    suite = build_preregistered_codebase_proof_suite(
        include_live_model_channel=include_live_model_channel
    )
    return evaluate_codebase_proof_benchmark(suite)


# Compatibility alias for the interface name in the task board.
CodebaseProofBenchmark = CodebaseProofBenchmarkSuite


__all__ = [
    "CODEBASE_PROOF_BENCHMARK_CORPUS_VERSION",
    "CODEBASE_PROOF_BENCHMARK_INTERFACE",
    "CODEBASE_PROOF_BENCHMARK_VERSION",
    "CODEBASE_PROOF_EFFICIENCY_REQUIREMENT_ID",
    "CODEBASE_PROOF_OBJECTIVE_ID",
    "CODEBASE_PROOF_OBJECTIVE_REVISION",
    "CODEBASE_PROOF_REPORT_SCHEMA",
    "CODEBASE_PROOF_SUITE_SCHEMA",
    "MAX_FALSE_AUTHORITATIVE_ADMISSIONS",
    "MIN_INPUT_TOKEN_REDUCTION_BPS",
    "MIN_RETRY_TOKEN_REDUCTION_BPS",
    "REQUIRED_CLAIM_FAMILIES",
    "REQUIRED_CLAIM_STATUSES",
    "ClaimOutcomeObservation",
    "CodeProofArmObservation",
    "CodeProofGateName",
    "CodeProofGateResult",
    "CodeProofPairedCase",
    "CodebaseProofBenchmark",
    "CodebaseProofBenchmarkError",
    "CodebaseProofBenchmarkReport",
    "CodebaseProofBenchmarkSuite",
    "ContextPath",
    "LiveModelChannelObservation",
    "MutationSeedCase",
    "MutationSeedKind",
    "ResultChannel",
    "build_preregistered_codebase_proof_suite",
    "evaluate_codebase_proof_benchmark",
    "run_codebase_proof_efficiency_gates",
    "verify_codebase_proof_benchmark_report",
]
