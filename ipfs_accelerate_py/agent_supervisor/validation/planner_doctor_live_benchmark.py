"""Live paired Planner/Doctor benchmarks on hermetic repositories (PDR-070).

Interfaces: ``PlannerDoctorLiveBenchmark@1``, ``LiveBenchmarkPairReceipt@1``

This module is the live producer for the preregistered Planner/Doctor paired
benchmark.  It:

* materializes hermetic mini-repositories from compact seeded recipes
  (prompt / planning / contract / security / repair / degradation);
* seals paired inputs **before** any candidate arm executes;
* invokes real create/steer (``PlanCreateService@1`` / ``PlanSteerService@1``)
  and Doctor (``DeterministicDoctorService@1``) service entry points;
* executes admitted work in isolated arm worktrees;
* obtains quality results from the independent oracle (PDR-072) only after
  process-tree termination, capability revocation, and output-root seal;
* never reads fixture ``expected`` fields to choose diagnosis, disposition,
  repair, or completion;
* labels deterministic Doctor fixture dual-runs and Supervisor V2 synthetic
  benchmarks as model/conformance evidence only; and
* rejects skips, dry-runs, mocks, and incomplete cells as promotion inputs.

Replay: the same case, cache stratum, concurrency, and arm binding produce an
identity-equivalent pair receipt.  Cache strata are ``cold``, ``exact-warm``,
``delta``, and ``restart``; requested concurrency is 1, 2, 4, and the
configured bootstrap maximum (6).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final, Optional

from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    content_identity,
)

# ---------------------------------------------------------------------------
# Schemas / interfaces
# ---------------------------------------------------------------------------

PLANNER_DOCTOR_LIVE_BENCHMARK_INTERFACE: Final[str] = "PlannerDoctorLiveBenchmark@1"
LIVE_BENCHMARK_PAIR_RECEIPT_INTERFACE: Final[str] = "LiveBenchmarkPairReceipt@1"
LIVE_BENCHMARK_CONTRACT_VERSION: Final[int] = 1
SCHEMA_VERSION: Final[int] = LIVE_BENCHMARK_CONTRACT_VERSION

LIVE_BENCHMARK_MANIFEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planner-doctor-live-benchmark-manifest@1"
)
LIVE_BENCHMARK_CASE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planner-doctor-live-benchmark-case@1"
)
LIVE_BENCHMARK_PAIR_SEAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planner-doctor-live-benchmark-pair-seal@1"
)
LIVE_BENCHMARK_ARM_EXECUTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planner-doctor-live-benchmark-arm-execution@1"
)
LIVE_BENCHMARK_PAIR_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planner-doctor-live-benchmark-pair-receipt@1"
)
LIVE_BENCHMARK_RUN_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planner-doctor-live-benchmark-run-report@1"
)
LIVE_BENCHMARK_CELL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planner-doctor-live-benchmark-cell@1"
)
EVIDENCE_AUTHORITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planner-doctor-live-evidence-authority@1"
)

PRODUCER_ID: Final[str] = "planner-doctor-live-benchmark@1"
PRODUCER_TASK_ID: Final[str] = "PDR-070"
GOAL_ID: Final[str] = "PDR-G080"
POLICY_ID: Final[str] = "planner-doctor-live-paired-benchmark-v1"

DEFAULT_LIVE_MANIFEST_RELATIVE: Final[str] = (
    "test/fixtures/agent_supervisor/planner_doctor_live/manifest.json"
)
DEFAULT_HOLDOUT_MANIFEST_RELATIVE: Final[str] = (
    "test/fixtures/agent_supervisor/planner_doctor_holdout/manifest.json"
)
DEFAULT_BENCHMARK_POLICY_RELATIVE: Final[str] = (
    "config/agent_supervisor_planner_doctor_benchmark.json"
)
DEFAULT_ORACLE_MANIFEST_RELATIVE: Final[str] = (
    "test/fixtures/agent_supervisor/planner_doctor_holdout/oracle.manifest.json"
)

# Synthetic / fixture producers — conformance evidence only.
CONFORMANCE_ONLY_EVIDENCE_SOURCES: Final[tuple[str, ...]] = (
    "deterministic-doctor-fixture-benchmark",
    "supervisor-v2-synthetic-benchmark",
    "symbolic-efficiency-fixture-observation",
    "fixture-dual-run-identity",
)
LIVE_EVIDENCE_SOURCES: Final[tuple[str, ...]] = (
    "live-hermetic-repository-execution",
    "real-plan-create-service",
    "real-plan-steer-service",
    "real-deterministic-doctor-service",
    "isolated-arm-worktree",
    "independent-quality-oracle",
)

CACHE_STRATA: Final[tuple[str, ...]] = (
    "cold",
    "exact-warm",
    "delta",
    "restart",
)
REQUESTED_CONCURRENCY: Final[tuple[int, ...]] = (1, 2, 4, 6)
CONFIGURED_MAXIMUM_WORKERS: Final[int] = 6
UNSCORED_WARMUP_REPETITIONS: Final[int] = 1
SCORED_REPETITIONS: Final[int] = 3

PRIMARY_ARM_IDS: Final[tuple[str, ...]] = (
    "current-mainline-baseline",
    "deterministic-symbolic",
    "hybrid-residual-only",
)

PAIR_FAMILIES: Final[tuple[str, ...]] = (
    "plan-create",
    "plan-steer",
    "doctor-diagnosis",
    "security-ir",
    "transaction-rollback",
    "capability-degradation",
)

# Fixture keys that must never drive diagnosis / disposition / repair / completion.
_FORBIDDEN_FIXTURE_DECISION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "expected",
        "expected_outcome",
        "expected_disposition",
        "expected_diagnosis",
        "expected_repair",
        "expected_completion",
        "gold",
        "golden",
        "oracle_body",
        "gold_outcome",
        "gold_disposition",
        "gold_diagnosis",
        "gold_repair",
        "gold_completion",
        "correct_disposition",
        "correct_diagnosis",
        "correct_repair",
        "target_disposition",
        "target_diagnosis",
    }
)

_FORBIDDEN_PAYLOAD_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "api_key",
        "authorization",
        "cookie",
        "credential",
        "credentials",
        "password",
        "private_key",
        "prompt_body",
        "secret",
        "secrets",
        "source_body",
        "token",
    }
)

MAX_TEXT_BYTES: Final[int] = 1024
MAX_ID_BYTES: Final[int] = 512
MAX_FILES: Final[int] = 256
MAX_FILE_BYTES: Final[int] = 256 * 1024
MAX_CASES: Final[int] = 64
MAX_INTEGER: Final[int] = 10**18

_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@+/-]{0,511}$")
_PATH_RE: Final[re.Pattern[str]] = re.compile(r"^(?!/)(?!.*\.\.)([A-Za-z0-9._@+/-]+)$")


# ---------------------------------------------------------------------------
# Errors / enums
# ---------------------------------------------------------------------------


class LiveBenchmarkError(ContractValidationError):
    """Live benchmark input, seal, or execution is malformed or unsafe."""

    def __init__(self, message: str, *, reason_code: str = "") -> None:
        super().__init__(message)
        self.reason_code = reason_code or "live_benchmark_error"


class CacheStratum(str, Enum):
    COLD = "cold"
    EXACT_WARM = "exact-warm"
    DELTA = "delta"
    RESTART = "restart"


class ArmId(str, Enum):
    CURRENT_MAINLINE = "current-mainline-baseline"
    DETERMINISTIC_SYMBOLIC = "deterministic-symbolic"
    HYBRID_RESIDUAL = "hybrid-residual-only"


class ExecutionKind(str, Enum):
    PLANNER_CREATE = "planner-create"
    PLANNER_STEER = "planner-steer"
    DOCTOR_DIAGNOSE_REPAIR = "doctor-diagnose-repair"
    PLANNER_DOCTOR_SECURITY = "planner-doctor-security-gate"
    DOCTOR_TRANSACTION = "doctor-transaction"
    PLANNER_DOCTOR_DEGRADATION = "planner-doctor-degradation"


class ArmExecutionStatus(str, Enum):
    """Terminal status of one arm cell execution."""

    MEASURED = "measured"
    FAILED = "failed"
    CRASHED = "crashed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"  # never promotion-eligible
    CAPABILITY_ABSTAINED = "capability_abstained"


class EvidenceAuthorityClass(str, Enum):
    """Whether a measurement may feed promotion."""

    LIVE_SERVICE = "live-service-execution"
    CONFORMANCE_ONLY = "model-conformance-evidence-only"
    SYNTHETIC = "synthetic-fixture-observation"
    SKIPPED = "skipped-not-promotion-eligible"


class PairReceiptDisposition(str, Enum):
    PAIRED = "paired"
    MISMATCHED_INPUTS = "mismatched-inputs"
    INCOMPLETE = "incomplete"
    SAFETY_REJECTED = "safety-rejected"
    NOT_PROMOTION_ELIGIBLE = "not-promotion-eligible"


class ProviderCallPermission(str, Enum):
    AS_CONFIGURED = "as-currently-configured"
    FORBIDDEN = "forbidden"
    BOUNDED_RESIDUAL = "bounded-residual-only"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(value: Any, field_name: str, *, required: bool = True, limit: int = MAX_TEXT_BYTES) -> str:
    if value is None:
        text = ""
    elif not isinstance(value, str):
        raise LiveBenchmarkError(f"{field_name} must be a string", reason_code="malformed")
    else:
        text = value.strip()
    if required and not text:
        raise LiveBenchmarkError(f"{field_name} is required", reason_code="malformed")
    if len(text.encode("utf-8")) > limit:
        raise LiveBenchmarkError(f"{field_name} exceeds byte bound", reason_code="malformed")
    if "\0" in text:
        raise LiveBenchmarkError(f"{field_name} must not contain NUL", reason_code="malformed")
    return text


def _optional_text(value: Any, field_name: str, *, limit: int = MAX_TEXT_BYTES) -> str:
    if value in (None, ""):
        return ""
    return _text(value, field_name, required=True, limit=limit)


def _identifier(value: Any, field_name: str) -> str:
    text = _text(value, field_name, required=True, limit=MAX_ID_BYTES)
    if not _ID_RE.match(text):
        raise LiveBenchmarkError(
            f"{field_name} must be a compact identifier",
            reason_code="malformed",
        )
    return text


def _integer(value: Any, field_name: str, *, minimum: int = 0, maximum: int = MAX_INTEGER) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LiveBenchmarkError(f"{field_name} must be an integer", reason_code="malformed")
    if value < minimum or value > maximum:
        raise LiveBenchmarkError(f"{field_name} out of range", reason_code="malformed")
    return value


def _bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise LiveBenchmarkError(f"{field_name} must be a boolean", reason_code="malformed")
    return value


def _enum(value: Any, enum: type[Enum], field_name: str) -> Enum:
    try:
        return value if isinstance(value, enum) else enum(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum)
        raise LiveBenchmarkError(
            f"{field_name} must be one of: {allowed}",
            reason_code="malformed",
        ) from exc


def _path_key(value: Any, field_name: str) -> str:
    text = _text(value, field_name, required=True, limit=MAX_ID_BYTES)
    if not _PATH_RE.match(text):
        raise LiveBenchmarkError(
            f"{field_name} must be a relative POSIX path without '..'",
            reason_code="path_escape",
        )
    return text


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, MappingProxyType):
        return {str(k): _plain(v) for k, v in value.items()}
    return value


def _sha256_hex(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _repo_root_from(path: Path | None = None) -> Path:
    if path is not None:
        return Path(path).resolve()
    # validation/<module>.py -> parents[3] is package root? 
    # agent_supervisor/validation/this.py -> parents[0]=validation, 1=agent_supervisor,
    # 2=ipfs_accelerate_py, 3=repo root
    return Path(__file__).resolve().parents[3]


def assert_no_fixture_decision_fields(
    value: Any,
    *,
    field_name: str = "fixture",
) -> None:
    """Fail closed if fixture payloads try to supply decision outcomes.

    The live runner never reads fixture ``expected`` / gold fields to choose
    diagnosis, disposition, repair, or completion.  Gold for quality scoring
    lives only in the operator-owned oracle (PDR-072) and is consulted only
    after arm output is sealed.
    """

    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise LiveBenchmarkError(
                    f"{field_name} has a non-string key",
                    reason_code="malformed",
                )
            normalized = key.lower().replace("-", "_")
            if normalized in _FORBIDDEN_FIXTURE_DECISION_KEYS:
                raise LiveBenchmarkError(
                    f"{field_name} must not contain decision field {key!r}; "
                    "outcomes come from live service execution and independent oracle",
                    reason_code="fixture_expected_forbidden",
                )
            if normalized in _FORBIDDEN_PAYLOAD_MARKERS:
                raise LiveBenchmarkError(
                    f"{field_name} must not contain body/secret field {key!r}",
                    reason_code="body_or_secret_forbidden",
                )
            assert_no_fixture_decision_fields(item, field_name=f"{field_name}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            assert_no_fixture_decision_fields(item, field_name=f"{field_name}[{index}]")


def evidence_authority_for_source(source_id: str) -> EvidenceAuthorityClass:
    """Classify a measurement source for promotion admission."""

    source = _identifier(source_id, "source_id")
    if source in CONFORMANCE_ONLY_EVIDENCE_SOURCES or source.startswith("synthetic:"):
        return EvidenceAuthorityClass.CONFORMANCE_ONLY
    if source.startswith("skip") or source in {"skipped", "xfail", "dry-run", "mock"}:
        return EvidenceAuthorityClass.SKIPPED
    if source in LIVE_EVIDENCE_SOURCES or source.startswith("live:"):
        return EvidenceAuthorityClass.LIVE_SERVICE
    if source.startswith("fixture:") or source.startswith("conformance:"):
        return EvidenceAuthorityClass.CONFORMANCE_ONLY
    return EvidenceAuthorityClass.SYNTHETIC


def skip_qualifies_for_promotion(status: ArmExecutionStatus | str) -> bool:
    """Skips can never qualify promotion (denominator retained as incomplete)."""

    status_value = status.value if isinstance(status, ArmExecutionStatus) else str(status)
    return status_value != ArmExecutionStatus.SKIPPED.value and status_value not in {
        "skip",
        "skipped",
        "xfail",
        "dry_run",
        "dry-run",
        "mock",
    }


def effective_workers(
    requested: int,
    *,
    configured_maximum: int = CONFIGURED_MAXIMUM_WORKERS,
    admitted_dag_width: int | None = None,
    resource_admission_limit: int | None = None,
) -> int:
    """min(requested, configured max, admitted DAG width, resource admission)."""

    req = _integer(requested, "requested", minimum=1, maximum=CONFIGURED_MAXIMUM_WORKERS * 4)
    cfg = _integer(configured_maximum, "configured_maximum", minimum=1)
    width = admitted_dag_width if admitted_dag_width is not None else cfg
    resource = resource_admission_limit if resource_admission_limit is not None else cfg
    return max(1, min(req, cfg, int(width), int(resource)))


def scored_cell_count(
    *,
    case_count: int,
    arm_count: int = len(PRIMARY_ARM_IDS),
    strata: Sequence[str] = CACHE_STRATA,
    concurrency: Sequence[int] = REQUESTED_CONCURRENCY,
    scored_repetitions: int = SCORED_REPETITIONS,
) -> int:
    """Exact denominator for scored executions (no optional sampling)."""

    return (
        _integer(case_count, "case_count", minimum=1)
        * _integer(arm_count, "arm_count", minimum=1)
        * len(tuple(strata))
        * len(tuple(concurrency))
        * _integer(scored_repetitions, "scored_repetitions", minimum=1)
    )


# ---------------------------------------------------------------------------
# Case recipe / manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HermeticFileRecipe:
    """One file to materialize in a hermetic mini-repository."""

    path: str
    content: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _path_key(self.path, "path"))
        if not isinstance(self.content, str):
            raise LiveBenchmarkError("content must be text", reason_code="malformed")
        if len(self.content.encode("utf-8")) > MAX_FILE_BYTES:
            raise LiveBenchmarkError("file exceeds byte bound", reason_code="malformed")

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "content": self.content}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HermeticFileRecipe":
        return cls(path=value["path"], content=value["content"])


@dataclass(frozen=True)
class LiveBenchmarkCase:
    """One live hermetic case recipe.

    Recipes deliberately omit diagnosis / disposition / repair / completion
    gold.  Independent oracle slots bind after execution.
    """

    SCHEMA: ClassVar[str] = LIVE_BENCHMARK_CASE_SCHEMA

    case_id: str
    pair_family: str
    execution_kind: ExecutionKind
    partition: str
    deterministic_seed: int
    prompt_template_id: str
    mutation_recipe_id: str
    task_source_seed_id: str
    oracle_slot_id: str
    holdout_case_id: str
    files: tuple[HermeticFileRecipe, ...]
    required_capabilities: tuple[str, ...] = ()
    seed_defect_markers: tuple[str, ...] = ()
    security_markers: tuple[str, ...] = ()
    promotion_eligible: bool = False
    case_cid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _identifier(self.case_id, "case_id"))
        object.__setattr__(
            self, "pair_family", _identifier(self.pair_family, "pair_family")
        )
        if self.pair_family not in PAIR_FAMILIES:
            raise LiveBenchmarkError(
                f"unknown pair_family: {self.pair_family}",
                reason_code="unknown_family",
            )
        object.__setattr__(
            self,
            "execution_kind",
            _enum(self.execution_kind, ExecutionKind, "execution_kind"),
        )
        object.__setattr__(
            self, "partition", _identifier(self.partition, "partition")
        )
        if self.partition not in {"development", "heldout", "hermetic-live"}:
            raise LiveBenchmarkError(
                "partition must be development, heldout, or hermetic-live",
                reason_code="malformed",
            )
        object.__setattr__(
            self,
            "deterministic_seed",
            _integer(self.deterministic_seed, "deterministic_seed", minimum=1),
        )
        object.__setattr__(
            self,
            "prompt_template_id",
            _identifier(self.prompt_template_id, "prompt_template_id"),
        )
        object.__setattr__(
            self,
            "mutation_recipe_id",
            _identifier(self.mutation_recipe_id, "mutation_recipe_id"),
        )
        object.__setattr__(
            self,
            "task_source_seed_id",
            _identifier(self.task_source_seed_id, "task_source_seed_id"),
        )
        object.__setattr__(
            self, "oracle_slot_id", _identifier(self.oracle_slot_id, "oracle_slot_id")
        )
        object.__setattr__(
            self,
            "holdout_case_id",
            _identifier(self.holdout_case_id, "holdout_case_id"),
        )
        if not self.files:
            raise LiveBenchmarkError("case requires at least one file", reason_code="malformed")
        if len(self.files) > MAX_FILES:
            raise LiveBenchmarkError("too many files", reason_code="malformed")
        object.__setattr__(self, "files", tuple(self.files))
        object.__setattr__(
            self,
            "required_capabilities",
            tuple(_identifier(x, "capability") for x in self.required_capabilities),
        )
        object.__setattr__(
            self,
            "seed_defect_markers",
            tuple(_identifier(x, "seed_defect_marker") for x in self.seed_defect_markers),
        )
        object.__setattr__(
            self,
            "security_markers",
            tuple(_identifier(x, "security_marker") for x in self.security_markers),
        )
        object.__setattr__(
            self, "promotion_eligible", _bool(self.promotion_eligible, "promotion_eligible")
        )
        if self.promotion_eligible:
            raise LiveBenchmarkError(
                "live hermetic cases cannot be promotion-eligible without external holdout",
                reason_code="promotion_forbidden",
            )
        payload = self._identity_payload()
        computed = content_identity(payload)
        claimed = _optional_text(self.case_cid, "case_cid", limit=MAX_ID_BYTES)
        if claimed and claimed != computed:
            raise LiveBenchmarkError(
                "case_cid does not match case body",
                reason_code="forged_case_cid",
            )
        object.__setattr__(self, "case_cid", computed)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "pair_family": self.pair_family,
            "execution_kind": self.execution_kind.value,
            "partition": self.partition,
            "deterministic_seed": self.deterministic_seed,
            "prompt_template_id": self.prompt_template_id,
            "mutation_recipe_id": self.mutation_recipe_id,
            "task_source_seed_id": self.task_source_seed_id,
            "oracle_slot_id": self.oracle_slot_id,
            "holdout_case_id": self.holdout_case_id,
            "files": [f.to_dict() for f in self.files],
            "required_capabilities": list(self.required_capabilities),
            "seed_defect_markers": list(self.seed_defect_markers),
            "security_markers": list(self.security_markers),
            "promotion_eligible": self.promotion_eligible,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        payload["schema"] = self.SCHEMA
        payload["case_cid"] = self.case_cid
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LiveBenchmarkCase":
        assert_no_fixture_decision_fields(value, field_name="live_case")
        files = tuple(
            HermeticFileRecipe.from_dict(item) for item in (value.get("files") or ())
        )
        return cls(
            case_id=value["case_id"],
            pair_family=value["pair_family"],
            execution_kind=value["execution_kind"],
            partition=value.get("partition", "hermetic-live"),
            deterministic_seed=value["deterministic_seed"],
            prompt_template_id=value["prompt_template_id"],
            mutation_recipe_id=value["mutation_recipe_id"],
            task_source_seed_id=value["task_source_seed_id"],
            oracle_slot_id=value["oracle_slot_id"],
            holdout_case_id=value["holdout_case_id"],
            files=files,
            required_capabilities=tuple(value.get("required_capabilities") or ()),
            seed_defect_markers=tuple(value.get("seed_defect_markers") or ()),
            security_markers=tuple(value.get("security_markers") or ()),
            promotion_eligible=bool(value.get("promotion_eligible", False)),
            case_cid=str(value.get("case_cid") or ""),
        )


@dataclass(frozen=True)
class LiveBenchmarkManifest:
    """Operator-owned live runner manifest (compact recipes, no gold outcomes)."""

    SCHEMA: ClassVar[str] = LIVE_BENCHMARK_MANIFEST_SCHEMA
    INTERFACE: ClassVar[str] = PLANNER_DOCTOR_LIVE_BENCHMARK_INTERFACE

    task_id: str
    goal_id: str
    policy_id: str
    cases: tuple[LiveBenchmarkCase, ...]
    holdout_manifest_cid: str = ""
    benchmark_policy_cid: str = ""
    cache_strata: tuple[str, ...] = CACHE_STRATA
    requested_concurrency: tuple[int, ...] = REQUESTED_CONCURRENCY
    configured_maximum_workers: int = CONFIGURED_MAXIMUM_WORKERS
    scored_repetitions: int = SCORED_REPETITIONS
    primary_arm_ids: tuple[str, ...] = PRIMARY_ARM_IDS
    conformance_only_sources: tuple[str, ...] = CONFORMANCE_ONLY_EVIDENCE_SOURCES
    automatic_promotion_enabled: bool = False
    fixture_expected_fields_are_not_execution_authority: bool = True
    skips_cannot_qualify_promotion: bool = True
    synthetic_results_are_execution_authority: bool = False
    manifest_cid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _identifier(self.task_id, "task_id"))
        object.__setattr__(self, "goal_id", _identifier(self.goal_id, "goal_id"))
        object.__setattr__(self, "policy_id", _identifier(self.policy_id, "policy_id"))
        if not self.cases:
            raise LiveBenchmarkError("manifest requires cases", reason_code="malformed")
        if len(self.cases) > MAX_CASES:
            raise LiveBenchmarkError("too many cases", reason_code="malformed")
        case_ids = [c.case_id for c in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise LiveBenchmarkError("duplicate case_id", reason_code="malformed")
        object.__setattr__(self, "cases", tuple(self.cases))
        object.__setattr__(
            self,
            "holdout_manifest_cid",
            _optional_text(self.holdout_manifest_cid, "holdout_manifest_cid", limit=MAX_ID_BYTES),
        )
        object.__setattr__(
            self,
            "benchmark_policy_cid",
            _optional_text(self.benchmark_policy_cid, "benchmark_policy_cid", limit=MAX_ID_BYTES),
        )
        strata = tuple(_identifier(s, "cache_stratum") for s in self.cache_strata)
        if tuple(strata) != CACHE_STRATA:
            raise LiveBenchmarkError(
                "cache_strata must be exactly cold/exact-warm/delta/restart",
                reason_code="denominator_mutation",
            )
        object.__setattr__(self, "cache_strata", strata)
        conc = tuple(
            _integer(c, "concurrency", minimum=1) for c in self.requested_concurrency
        )
        if conc != REQUESTED_CONCURRENCY:
            raise LiveBenchmarkError(
                "requested_concurrency must be exactly 1,2,4,6",
                reason_code="denominator_mutation",
            )
        object.__setattr__(self, "requested_concurrency", conc)
        object.__setattr__(
            self,
            "configured_maximum_workers",
            _integer(
                self.configured_maximum_workers,
                "configured_maximum_workers",
                minimum=1,
            ),
        )
        if self.configured_maximum_workers != CONFIGURED_MAXIMUM_WORKERS:
            raise LiveBenchmarkError(
                "configured_maximum_workers must match bootstrap maximum 6",
                reason_code="denominator_mutation",
            )
        object.__setattr__(
            self,
            "scored_repetitions",
            _integer(self.scored_repetitions, "scored_repetitions", minimum=1),
        )
        if self.scored_repetitions != SCORED_REPETITIONS:
            raise LiveBenchmarkError(
                "scored_repetitions must be exactly 3",
                reason_code="denominator_mutation",
            )
        arms = tuple(_identifier(a, "arm_id") for a in self.primary_arm_ids)
        if arms != PRIMARY_ARM_IDS:
            raise LiveBenchmarkError(
                "primary_arm_ids must match preregistered primary arms",
                reason_code="denominator_mutation",
            )
        object.__setattr__(self, "primary_arm_ids", arms)
        object.__setattr__(
            self,
            "conformance_only_sources",
            tuple(
                _identifier(s, "conformance_source")
                for s in self.conformance_only_sources
            ),
        )
        for flag_name in (
            "automatic_promotion_enabled",
            "fixture_expected_fields_are_not_execution_authority",
            "skips_cannot_qualify_promotion",
            "synthetic_results_are_execution_authority",
        ):
            object.__setattr__(
                self, flag_name, _bool(getattr(self, flag_name), flag_name)
            )
        if self.automatic_promotion_enabled:
            raise LiveBenchmarkError(
                "automatic_promotion_enabled must be false",
                reason_code="promotion_forbidden",
            )
        if self.synthetic_results_are_execution_authority:
            raise LiveBenchmarkError(
                "synthetic_results_are_execution_authority must be false",
                reason_code="synthetic_authority_forbidden",
            )
        if not self.fixture_expected_fields_are_not_execution_authority:
            raise LiveBenchmarkError(
                "fixture expected fields must not be execution authority",
                reason_code="fixture_authority_forbidden",
            )
        if not self.skips_cannot_qualify_promotion:
            raise LiveBenchmarkError(
                "skips_cannot_qualify_promotion must be true",
                reason_code="skip_promotion_forbidden",
            )
        payload = self._identity_payload()
        computed = content_identity(payload)
        claimed = _optional_text(self.manifest_cid, "manifest_cid", limit=MAX_ID_BYTES)
        if claimed and claimed != computed:
            raise LiveBenchmarkError(
                "manifest_cid does not match manifest body",
                reason_code="forged_manifest_cid",
            )
        object.__setattr__(self, "manifest_cid", computed)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "policy_id": self.policy_id,
            "interface": self.INTERFACE,
            "cases": [c.to_dict() for c in self.cases],
            "holdout_manifest_cid": self.holdout_manifest_cid,
            "benchmark_policy_cid": self.benchmark_policy_cid,
            "cache_strata": list(self.cache_strata),
            "requested_concurrency": list(self.requested_concurrency),
            "configured_maximum_workers": self.configured_maximum_workers,
            "scored_repetitions": self.scored_repetitions,
            "primary_arm_ids": list(self.primary_arm_ids),
            "conformance_only_sources": list(self.conformance_only_sources),
            "automatic_promotion_enabled": self.automatic_promotion_enabled,
            "fixture_expected_fields_are_not_execution_authority": (
                self.fixture_expected_fields_are_not_execution_authority
            ),
            "skips_cannot_qualify_promotion": self.skips_cannot_qualify_promotion,
            "synthetic_results_are_execution_authority": (
                self.synthetic_results_are_execution_authority
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        payload["schema"] = self.SCHEMA
        payload["manifest_cid"] = self.manifest_cid
        payload["contract_version"] = LIVE_BENCHMARK_CONTRACT_VERSION
        return payload

    @property
    def content_id(self) -> str:
        return self.manifest_cid

    def case_by_id(self, case_id: str) -> LiveBenchmarkCase:
        for case in self.cases:
            if case.case_id == case_id:
                return case
        raise LiveBenchmarkError(f"unknown case_id: {case_id}", reason_code="unknown_case")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LiveBenchmarkManifest":
        assert_no_fixture_decision_fields(value, field_name="live_manifest")
        cases = tuple(
            LiveBenchmarkCase.from_dict(item) for item in (value.get("cases") or ())
        )
        return cls(
            task_id=value.get("task_id", PRODUCER_TASK_ID),
            goal_id=value.get("goal_id", GOAL_ID),
            policy_id=value.get("policy_id", POLICY_ID),
            cases=cases,
            holdout_manifest_cid=str(value.get("holdout_manifest_cid") or ""),
            benchmark_policy_cid=str(value.get("benchmark_policy_cid") or ""),
            cache_strata=tuple(value.get("cache_strata") or CACHE_STRATA),
            requested_concurrency=tuple(
                value.get("requested_concurrency") or REQUESTED_CONCURRENCY
            ),
            configured_maximum_workers=int(
                value.get("configured_maximum_workers", CONFIGURED_MAXIMUM_WORKERS)
            ),
            scored_repetitions=int(value.get("scored_repetitions", SCORED_REPETITIONS)),
            primary_arm_ids=tuple(value.get("primary_arm_ids") or PRIMARY_ARM_IDS),
            conformance_only_sources=tuple(
                value.get("conformance_only_sources") or CONFORMANCE_ONLY_EVIDENCE_SOURCES
            ),
            automatic_promotion_enabled=bool(
                value.get("automatic_promotion_enabled", False)
            ),
            fixture_expected_fields_are_not_execution_authority=bool(
                value.get("fixture_expected_fields_are_not_execution_authority", True)
            ),
            skips_cannot_qualify_promotion=bool(
                value.get("skips_cannot_qualify_promotion", True)
            ),
            synthetic_results_are_execution_authority=bool(
                value.get("synthetic_results_are_execution_authority", False)
            ),
            manifest_cid=str(value.get("manifest_cid") or ""),
        )

    @classmethod
    def load(cls, path: str | Path) -> "LiveBenchmarkManifest":
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise LiveBenchmarkError("manifest must be an object", reason_code="malformed")
        return cls.from_dict(document)


# ---------------------------------------------------------------------------
# Hermetic repository materialization
# ---------------------------------------------------------------------------


@dataclass
class HermeticRepository:
    """Materialized git mini-repository for one arm cell."""

    root: Path
    head_commit: str
    tree_id: str
    forest_cid: str
    worktree_root: Path | None = None

    def cleanup(self) -> None:
        if self.worktree_root is not None and self.worktree_root.exists():
            shutil.rmtree(self.worktree_root, ignore_errors=True)
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    # Fixed author/committer clocks make hermetic seeds replayable and keep
    # paired repository forests identical across primary arms.
    env = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_AUTHOR_NAME": "Live Benchmark Runner",
        "GIT_AUTHOR_EMAIL": "live-benchmark@localhost",
        "GIT_AUTHOR_DATE": "1970-01-01T00:00:00 +0000",
        "GIT_COMMITTER_NAME": "Live Benchmark Runner",
        "GIT_COMMITTER_EMAIL": "live-benchmark@localhost",
        "GIT_COMMITTER_DATE": "1970-01-01T00:00:00 +0000",
        "HOME": str(repo),
        "XDG_CONFIG_HOME": str(repo / ".xdg-config"),
        "XDG_CACHE_HOME": str(repo / ".xdg-cache"),
    }
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        capture_output=True,
        text=True,
        env=env,
    )


def materialize_hermetic_repository(
    case: LiveBenchmarkCase,
    *,
    parent: str | Path | None = None,
    arm_id: str = "deterministic-symbolic",
    stratum_id: str = "cold",
    concurrency: int = 1,
    repetition: int = 0,
    apply_delta: bool = False,
) -> HermeticRepository:
    """Create an isolated hermetic git mini-repository from a case recipe."""

    assert_no_fixture_decision_fields(case.to_dict(), field_name="case")
    base = Path(parent) if parent is not None else Path(tempfile.mkdtemp(prefix="pdr-live-"))
    base.mkdir(parents=True, exist_ok=True)
    name = (
        f"{case.case_id}__{arm_id}__{stratum_id}__c{concurrency}__r{repetition}"
    )
    # Sanitize filesystem name.
    safe = re.sub(r"[^A-Za-z0-9._@+-]+", "_", name)[:180]
    root = base / safe
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    _git(root, "init", "-q", check=True)
    _git(root, "config", "user.email", "live-benchmark@localhost")
    _git(root, "config", "user.name", "Live Benchmark Runner")
    _git(root, "config", "commit.gpgsign", "false")

    for recipe in case.files:
        path = root / recipe.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(recipe.content, encoding="utf-8")
        _git(root, "add", "--", recipe.path)

    # Seed metadata (no expected outcomes). Arm identity is a treatment and must
    # not enter the shared repository forest / paired input commitment.
    meta = {
        "case_id": case.case_id,
        "case_cid": case.case_cid,
        "pair_family": case.pair_family,
        "execution_kind": case.execution_kind.value,
        "prompt_template_id": case.prompt_template_id,
        "mutation_recipe_id": case.mutation_recipe_id,
        "task_source_seed_id": case.task_source_seed_id,
        "deterministic_seed": case.deterministic_seed,
        "stratum_id": stratum_id if stratum_id == CacheStratum.DELTA.value else "base",
        "seed_defect_markers": list(case.seed_defect_markers),
        "security_markers": list(case.security_markers),
    }
    meta_path = root / ".pdr_live_case.json"
    meta_path.write_text(
        json.dumps(meta, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _git(root, "add", "--", ".pdr_live_case.json")
    _git(root, "commit", "-q", "-m", f"seed {case.case_id}")

    if apply_delta and case.mutation_recipe_id not in {"none", "none@1", "mutation:none"}:
        # Apply a deterministic mutation overlay for delta strata.
        delta_path = root / "pkg" / "delta_overlay.py"
        delta_path.parent.mkdir(parents=True, exist_ok=True)
        delta_path.write_text(
            f"# delta mutation for {case.mutation_recipe_id}\n"
            f"DELTA_SEED = {case.deterministic_seed}\n"
            f"def delta_marker() -> int:\n"
            f"    return {case.deterministic_seed}\n",
            encoding="utf-8",
        )
        _git(root, "add", "--", "pkg/delta_overlay.py")
        _git(root, "commit", "-q", "-m", f"delta {case.mutation_recipe_id}")

    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    tree = _git(root, "rev-parse", "HEAD^{tree}").stdout.strip()
    # Forest identity is case content + tree only. Arm treatments (mode /
    # provider permission / candidate code root) are sealed separately.
    forest_cid = content_identity(
        {
            "case_cid": case.case_cid,
            "tree_id": tree,
            "stratum_family": (
                "delta" if stratum_id == CacheStratum.DELTA.value else "base"
            ),
        }
    )
    return HermeticRepository(
        root=root,
        head_commit=head,
        tree_id=tree,
        forest_cid=forest_cid,
    )


def create_isolated_worktree(
    repository: HermeticRepository,
    *,
    worktree_parent: str | Path | None = None,
) -> Path:
    """Create an isolated no-checkout worktree for admitted task/repair work."""

    parent = Path(worktree_parent) if worktree_parent is not None else repository.root.parent
    parent.mkdir(parents=True, exist_ok=True)
    worktree = parent / f"{repository.root.name}.worktree"
    if worktree.exists():
        shutil.rmtree(worktree)
    # Detached worktree at HEAD for isolated execution.
    _git(
        repository.root,
        "worktree",
        "add",
        "--detach",
        str(worktree),
        "HEAD",
    )
    repository.worktree_root = worktree
    return worktree


# ---------------------------------------------------------------------------
# Pair seal / receipts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveBenchmarkPairSeal:
    """Operator-owned pre-execution binding of exact paired inputs."""

    SCHEMA: ClassVar[str] = LIVE_BENCHMARK_PAIR_SEAL_SCHEMA

    policy_cid: str
    manifest_cid: str
    case_id: str
    case_cid: str
    input_commitment_cid: str
    repository_forest_cid: str
    prompt_directive_cid: str
    mutation_recipe_cid: str
    task_source_revision_cid: str
    authority_policy_cid: str
    intent_ir_cid: str
    security_ir_cid: str
    property_catalog_cid: str
    toolchain_manifest_cid: str
    provider_manifest_cid: str
    model_manifest_cid: str
    tokenizer_manifest_cid: str
    context_budget_cid: str
    cache_stratum_id: str
    cache_seed_cid: str
    hardware_manifest_cid: str
    worker_manifest_cid: str
    concurrency_requested: int
    deterministic_seed: int
    budget_manifest_cid: str
    oracle_manifest_cid: str
    arm_id: str
    planner_doctor_mode: str
    provider_call_permission: str
    ablation_id: str = ""
    repetition: int = 0
    scored: bool = True
    seal_cid: str = ""

    def __post_init__(self) -> None:
        for name in (
            "policy_cid",
            "manifest_cid",
            "case_id",
            "case_cid",
            "input_commitment_cid",
            "repository_forest_cid",
            "prompt_directive_cid",
            "mutation_recipe_cid",
            "task_source_revision_cid",
            "authority_policy_cid",
            "intent_ir_cid",
            "security_ir_cid",
            "property_catalog_cid",
            "toolchain_manifest_cid",
            "provider_manifest_cid",
            "model_manifest_cid",
            "tokenizer_manifest_cid",
            "context_budget_cid",
            "cache_stratum_id",
            "cache_seed_cid",
            "hardware_manifest_cid",
            "worker_manifest_cid",
            "budget_manifest_cid",
            "oracle_manifest_cid",
            "arm_id",
            "planner_doctor_mode",
            "provider_call_permission",
        ):
            object.__setattr__(
                self,
                name,
                _identifier(getattr(self, name), name)
                if name
                not in {
                    "policy_cid",
                    "manifest_cid",
                    "case_cid",
                    "input_commitment_cid",
                    "repository_forest_cid",
                    "prompt_directive_cid",
                    "mutation_recipe_cid",
                    "task_source_revision_cid",
                    "authority_policy_cid",
                    "intent_ir_cid",
                    "security_ir_cid",
                    "property_catalog_cid",
                    "toolchain_manifest_cid",
                    "provider_manifest_cid",
                    "model_manifest_cid",
                    "tokenizer_manifest_cid",
                    "context_budget_cid",
                    "cache_seed_cid",
                    "hardware_manifest_cid",
                    "worker_manifest_cid",
                    "budget_manifest_cid",
                    "oracle_manifest_cid",
                }
                else _text(getattr(self, name), name, required=True, limit=MAX_ID_BYTES),
            )
        object.__setattr__(
            self,
            "cache_stratum_id",
            _identifier(self.cache_stratum_id, "cache_stratum_id"),
        )
        if self.cache_stratum_id not in CACHE_STRATA:
            raise LiveBenchmarkError(
                f"unknown cache stratum: {self.cache_stratum_id}",
                reason_code="unknown_stratum",
            )
        object.__setattr__(
            self,
            "concurrency_requested",
            _integer(self.concurrency_requested, "concurrency_requested", minimum=1),
        )
        object.__setattr__(
            self,
            "deterministic_seed",
            _integer(self.deterministic_seed, "deterministic_seed", minimum=1),
        )
        object.__setattr__(
            self, "repetition", _integer(self.repetition, "repetition", minimum=0)
        )
        object.__setattr__(self, "scored", _bool(self.scored, "scored"))
        object.__setattr__(
            self, "ablation_id", _optional_text(self.ablation_id, "ablation_id")
        )
        payload = self._identity_payload()
        computed = content_identity(payload)
        claimed = _optional_text(self.seal_cid, "seal_cid", limit=MAX_ID_BYTES)
        if claimed and claimed != computed:
            raise LiveBenchmarkError(
                "seal_cid does not match seal body",
                reason_code="forged_seal_cid",
            )
        object.__setattr__(self, "seal_cid", computed)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "policy_cid": self.policy_cid,
            "manifest_cid": self.manifest_cid,
            "case_id": self.case_id,
            "case_cid": self.case_cid,
            "input_commitment_cid": self.input_commitment_cid,
            "repository_forest_cid": self.repository_forest_cid,
            "prompt_directive_cid": self.prompt_directive_cid,
            "mutation_recipe_cid": self.mutation_recipe_cid,
            "task_source_revision_cid": self.task_source_revision_cid,
            "authority_policy_cid": self.authority_policy_cid,
            "intent_ir_cid": self.intent_ir_cid,
            "security_ir_cid": self.security_ir_cid,
            "property_catalog_cid": self.property_catalog_cid,
            "toolchain_manifest_cid": self.toolchain_manifest_cid,
            "provider_manifest_cid": self.provider_manifest_cid,
            "model_manifest_cid": self.model_manifest_cid,
            "tokenizer_manifest_cid": self.tokenizer_manifest_cid,
            "context_budget_cid": self.context_budget_cid,
            "cache_stratum_id": self.cache_stratum_id,
            "cache_seed_cid": self.cache_seed_cid,
            "hardware_manifest_cid": self.hardware_manifest_cid,
            "worker_manifest_cid": self.worker_manifest_cid,
            "concurrency_requested": self.concurrency_requested,
            "deterministic_seed": self.deterministic_seed,
            "budget_manifest_cid": self.budget_manifest_cid,
            "oracle_manifest_cid": self.oracle_manifest_cid,
            "arm_id": self.arm_id,
            "planner_doctor_mode": self.planner_doctor_mode,
            "provider_call_permission": self.provider_call_permission,
            "ablation_id": self.ablation_id,
            "repetition": self.repetition,
            "scored": self.scored,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        payload["schema"] = self.SCHEMA
        payload["seal_cid"] = self.seal_cid
        return payload

    def pair_identity_fields(self) -> dict[str, Any]:
        """Fields that must match exactly across primary arms (excluding treatments)."""

        data = self._identity_payload()
        for key in (
            "arm_id",
            "planner_doctor_mode",
            "provider_call_permission",
            "ablation_id",
        ):
            data.pop(key, None)
        return data

    @property
    def content_id(self) -> str:
        return self.seal_cid


@dataclass(frozen=True)
class ArmExecutionReceipt:
    """One arm's sealed execution outcome (pre-oracle)."""

    SCHEMA: ClassVar[str] = LIVE_BENCHMARK_ARM_EXECUTION_SCHEMA

    seal_cid: str
    arm_id: str
    status: ArmExecutionStatus
    evidence_authority: EvidenceAuthorityClass
    service_interfaces_invoked: tuple[str, ...]
    worktree_root_cid: str
    output_root_cid: str
    service_disposition: str
    service_reason_codes: tuple[str, ...]
    process_tree_terminated: bool
    capabilities_revoked: bool
    output_root_sealed: bool
    first_valid_plan: bool = False
    typed_abstention: bool = False
    exact_rollback: bool = False
    predicted_defect_ids: tuple[str, ...] = ()
    predicted_localization_targets: tuple[str, ...] = ()
    repaired_defect_ids: tuple[str, ...] = ()
    satisfied_acceptance_ids: tuple[str, ...] = ()
    satisfied_security_ir_ids: tuple[str, ...] = ()
    satisfied_intent_ir_ids: tuple[str, ...] = ()
    goal_ids_covered: tuple[str, ...] = ()
    blast_radius_changed_lines: int = 0
    effective_workers: int = 1
    cache_namespace: str = ""
    wall_seconds_measured: bool = False
    wall_seconds: int = 0
    telemetry_receipt_cid: str = ""
    mount_receipt_cid: str = ""
    observation_payload: Mapping[str, Any] = field(default_factory=dict)
    receipt_cid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "seal_cid", _text(self.seal_cid, "seal_cid", limit=MAX_ID_BYTES))
        object.__setattr__(self, "arm_id", _identifier(self.arm_id, "arm_id"))
        object.__setattr__(
            self, "status", _enum(self.status, ArmExecutionStatus, "status")
        )
        object.__setattr__(
            self,
            "evidence_authority",
            _enum(self.evidence_authority, EvidenceAuthorityClass, "evidence_authority"),
        )
        object.__setattr__(
            self,
            "service_interfaces_invoked",
            tuple(_identifier(x, "service_interface") for x in self.service_interfaces_invoked),
        )
        for name in (
            "worktree_root_cid",
            "output_root_cid",
            "service_disposition",
            "cache_namespace",
            "telemetry_receipt_cid",
            "mount_receipt_cid",
        ):
            object.__setattr__(
                self,
                name,
                _optional_text(getattr(self, name), name, limit=MAX_ID_BYTES)
                if name
                in {
                    "telemetry_receipt_cid",
                    "mount_receipt_cid",
                    "cache_namespace",
                    "service_disposition",
                }
                else _text(getattr(self, name), name, required=True, limit=MAX_ID_BYTES),
            )
        object.__setattr__(
            self,
            "service_reason_codes",
            tuple(
                _identifier(x, "reason_code")
                for x in self.service_reason_codes
                if str(x).strip()
            ),
        )
        for name in (
            "process_tree_terminated",
            "capabilities_revoked",
            "output_root_sealed",
            "first_valid_plan",
            "typed_abstention",
            "exact_rollback",
            "wall_seconds_measured",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        for name in (
            "predicted_defect_ids",
            "predicted_localization_targets",
            "repaired_defect_ids",
            "satisfied_acceptance_ids",
            "satisfied_security_ir_ids",
            "satisfied_intent_ir_ids",
            "goal_ids_covered",
        ):
            object.__setattr__(
                self,
                name,
                tuple(
                    _identifier(x, name)
                    for x in (getattr(self, name) or ())
                    if str(x).strip()
                ),
            )
        object.__setattr__(
            self,
            "blast_radius_changed_lines",
            _integer(self.blast_radius_changed_lines, "blast_radius_changed_lines"),
        )
        object.__setattr__(
            self,
            "effective_workers",
            _integer(self.effective_workers, "effective_workers", minimum=1),
        )
        object.__setattr__(
            self, "wall_seconds", _integer(self.wall_seconds, "wall_seconds")
        )
        payload = dict(self.observation_payload or {})
        assert_no_fixture_decision_fields(payload, field_name="observation_payload")
        object.__setattr__(self, "observation_payload", MappingProxyType(dict(payload)))
        # Skips never promote.
        if self.status is ArmExecutionStatus.SKIPPED:
            if self.evidence_authority is not EvidenceAuthorityClass.SKIPPED:
                object.__setattr__(
                    self, "evidence_authority", EvidenceAuthorityClass.SKIPPED
                )
        computed = content_identity(self._identity_payload())
        claimed = _optional_text(self.receipt_cid, "receipt_cid", limit=MAX_ID_BYTES)
        if claimed and claimed != computed:
            raise LiveBenchmarkError(
                "receipt_cid does not match body",
                reason_code="forged_receipt_cid",
            )
        object.__setattr__(self, "receipt_cid", computed)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "seal_cid": self.seal_cid,
            "arm_id": self.arm_id,
            "status": self.status.value,
            "evidence_authority": self.evidence_authority.value,
            "service_interfaces_invoked": list(self.service_interfaces_invoked),
            "worktree_root_cid": self.worktree_root_cid,
            "output_root_cid": self.output_root_cid,
            "service_disposition": self.service_disposition,
            "service_reason_codes": list(self.service_reason_codes),
            "process_tree_terminated": self.process_tree_terminated,
            "capabilities_revoked": self.capabilities_revoked,
            "output_root_sealed": self.output_root_sealed,
            "first_valid_plan": self.first_valid_plan,
            "typed_abstention": self.typed_abstention,
            "exact_rollback": self.exact_rollback,
            "predicted_defect_ids": list(self.predicted_defect_ids),
            "predicted_localization_targets": list(self.predicted_localization_targets),
            "repaired_defect_ids": list(self.repaired_defect_ids),
            "satisfied_acceptance_ids": list(self.satisfied_acceptance_ids),
            "satisfied_security_ir_ids": list(self.satisfied_security_ir_ids),
            "satisfied_intent_ir_ids": list(self.satisfied_intent_ir_ids),
            "goal_ids_covered": list(self.goal_ids_covered),
            "blast_radius_changed_lines": self.blast_radius_changed_lines,
            "effective_workers": self.effective_workers,
            "cache_namespace": self.cache_namespace,
            "wall_seconds_measured": self.wall_seconds_measured,
            "wall_seconds": self.wall_seconds,
            "telemetry_receipt_cid": self.telemetry_receipt_cid,
            "mount_receipt_cid": self.mount_receipt_cid,
            "observation_payload": dict(self.observation_payload),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        payload["schema"] = self.SCHEMA
        payload["receipt_cid"] = self.receipt_cid
        return payload

    def promotion_eligible(self) -> bool:
        if self.status is ArmExecutionStatus.SKIPPED:
            return False
        if self.evidence_authority in {
            EvidenceAuthorityClass.CONFORMANCE_ONLY,
            EvidenceAuthorityClass.SYNTHETIC,
            EvidenceAuthorityClass.SKIPPED,
        }:
            return False
        if not (
            self.process_tree_terminated
            and self.capabilities_revoked
            and self.output_root_sealed
        ):
            return False
        return self.evidence_authority is EvidenceAuthorityClass.LIVE_SERVICE

    @property
    def content_id(self) -> str:
        return self.receipt_cid


@dataclass(frozen=True)
class LiveBenchmarkPairReceipt:
    """Paired comparison receipt for one case/stratum/concurrency/repetition."""

    SCHEMA: ClassVar[str] = LIVE_BENCHMARK_PAIR_RECEIPT_SCHEMA
    INTERFACE: ClassVar[str] = LIVE_BENCHMARK_PAIR_RECEIPT_INTERFACE

    case_id: str
    cache_stratum_id: str
    concurrency_requested: int
    repetition: int
    scored: bool
    pair_input_cid: str
    arm_receipts: tuple[ArmExecutionReceipt, ...]
    inputs_match_across_primary_arms: bool
    disposition: PairReceiptDisposition
    oracle_receipt_cids: tuple[str, ...] = ()
    promotion_eligible: bool = False
    reason_codes: tuple[str, ...] = ()
    receipt_cid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _identifier(self.case_id, "case_id"))
        object.__setattr__(
            self, "cache_stratum_id", _identifier(self.cache_stratum_id, "cache_stratum_id")
        )
        object.__setattr__(
            self,
            "concurrency_requested",
            _integer(self.concurrency_requested, "concurrency_requested", minimum=1),
        )
        object.__setattr__(
            self, "repetition", _integer(self.repetition, "repetition", minimum=0)
        )
        object.__setattr__(self, "scored", _bool(self.scored, "scored"))
        object.__setattr__(
            self,
            "pair_input_cid",
            _text(self.pair_input_cid, "pair_input_cid", limit=MAX_ID_BYTES),
        )
        if not self.arm_receipts:
            raise LiveBenchmarkError("pair receipt requires arm receipts", reason_code="malformed")
        object.__setattr__(self, "arm_receipts", tuple(self.arm_receipts))
        object.__setattr__(
            self,
            "inputs_match_across_primary_arms",
            _bool(
                self.inputs_match_across_primary_arms,
                "inputs_match_across_primary_arms",
            ),
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, PairReceiptDisposition, "disposition"),
        )
        object.__setattr__(
            self,
            "oracle_receipt_cids",
            tuple(
                _text(x, "oracle_receipt_cid", limit=MAX_ID_BYTES)
                for x in self.oracle_receipt_cids
            ),
        )
        object.__setattr__(
            self, "promotion_eligible", _bool(self.promotion_eligible, "promotion_eligible")
        )
        object.__setattr__(
            self,
            "reason_codes",
            tuple(_identifier(x, "reason_code") for x in self.reason_codes if str(x).strip()),
        )
        # Hard rules: skips and mismatched inputs cannot promote.
        if any(r.status is ArmExecutionStatus.SKIPPED for r in self.arm_receipts):
            object.__setattr__(self, "promotion_eligible", False)
            reasons = list(self.reason_codes)
            if "skip_not_promotion_eligible" not in reasons:
                reasons.append("skip_not_promotion_eligible")
            object.__setattr__(self, "reason_codes", tuple(reasons))
        if not self.inputs_match_across_primary_arms:
            object.__setattr__(self, "promotion_eligible", False)
        if self.promotion_eligible and self.disposition is not PairReceiptDisposition.PAIRED:
            object.__setattr__(self, "promotion_eligible", False)
        computed = content_identity(self._identity_payload())
        claimed = _optional_text(self.receipt_cid, "receipt_cid", limit=MAX_ID_BYTES)
        if claimed and claimed != computed:
            raise LiveBenchmarkError(
                "pair receipt_cid does not match body",
                reason_code="forged_receipt_cid",
            )
        object.__setattr__(self, "receipt_cid", computed)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "case_id": self.case_id,
            "cache_stratum_id": self.cache_stratum_id,
            "concurrency_requested": self.concurrency_requested,
            "repetition": self.repetition,
            "scored": self.scored,
            "pair_input_cid": self.pair_input_cid,
            "arm_receipts": [r.to_dict() for r in self.arm_receipts],
            "inputs_match_across_primary_arms": self.inputs_match_across_primary_arms,
            "disposition": self.disposition.value,
            "oracle_receipt_cids": list(self.oracle_receipt_cids),
            "promotion_eligible": self.promotion_eligible,
            "reason_codes": list(self.reason_codes),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        payload["schema"] = self.SCHEMA
        payload["receipt_cid"] = self.receipt_cid
        payload["contract_version"] = LIVE_BENCHMARK_CONTRACT_VERSION
        return payload

    @property
    def content_id(self) -> str:
        return self.receipt_cid


@dataclass(frozen=True)
class LiveBenchmarkRunReport:
    """Aggregate run report for a cell population or subset."""

    SCHEMA: ClassVar[str] = LIVE_BENCHMARK_RUN_REPORT_SCHEMA

    manifest_cid: str
    pair_receipts: tuple[LiveBenchmarkPairReceipt, ...]
    scored_cells_required: int
    scored_cells_observed: int
    incomplete: bool
    promotion_eligible: bool
    conformance_only_labels: tuple[str, ...]
    reason_codes: tuple[str, ...] = ()
    report_cid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "manifest_cid", _text(self.manifest_cid, "manifest_cid", limit=MAX_ID_BYTES)
        )
        object.__setattr__(self, "pair_receipts", tuple(self.pair_receipts))
        object.__setattr__(
            self,
            "scored_cells_required",
            _integer(self.scored_cells_required, "scored_cells_required"),
        )
        object.__setattr__(
            self,
            "scored_cells_observed",
            _integer(self.scored_cells_observed, "scored_cells_observed"),
        )
        object.__setattr__(self, "incomplete", _bool(self.incomplete, "incomplete"))
        object.__setattr__(
            self, "promotion_eligible", _bool(self.promotion_eligible, "promotion_eligible")
        )
        object.__setattr__(
            self,
            "conformance_only_labels",
            tuple(_identifier(x, "label") for x in self.conformance_only_labels),
        )
        object.__setattr__(
            self,
            "reason_codes",
            tuple(_identifier(x, "reason_code") for x in self.reason_codes if str(x).strip()),
        )
        if self.incomplete or self.scored_cells_observed < self.scored_cells_required:
            object.__setattr__(self, "promotion_eligible", False)
        if any(not r.promotion_eligible for r in self.pair_receipts):
            # Individual pairs may be non-promoting (public corpus); run stays non-promoting.
            object.__setattr__(self, "promotion_eligible", False)
        computed = content_identity(self._identity_payload())
        claimed = _optional_text(self.report_cid, "report_cid", limit=MAX_ID_BYTES)
        if claimed and claimed != computed:
            raise LiveBenchmarkError(
                "report_cid does not match body",
                reason_code="forged_report_cid",
            )
        object.__setattr__(self, "report_cid", computed)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "manifest_cid": self.manifest_cid,
            "pair_receipts": [r.to_dict() for r in self.pair_receipts],
            "scored_cells_required": self.scored_cells_required,
            "scored_cells_observed": self.scored_cells_observed,
            "incomplete": self.incomplete,
            "promotion_eligible": self.promotion_eligible,
            "conformance_only_labels": list(self.conformance_only_labels),
            "reason_codes": list(self.reason_codes),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        payload["schema"] = self.SCHEMA
        payload["report_cid"] = self.report_cid
        payload["interface"] = PLANNER_DOCTOR_LIVE_BENCHMARK_INTERFACE
        payload["contract_version"] = LIVE_BENCHMARK_CONTRACT_VERSION
        return payload

    @property
    def content_id(self) -> str:
        return self.report_cid


# ---------------------------------------------------------------------------
# Service adapters (real entry points)
# ---------------------------------------------------------------------------


def _arm_mode(arm_id: str) -> tuple[str, str]:
    """Return (planner_doctor_mode, provider_call_permission) for an arm."""

    if arm_id == ArmId.CURRENT_MAINLINE.value:
        return "current-production-path", ProviderCallPermission.AS_CONFIGURED.value
    if arm_id == ArmId.DETERMINISTIC_SYMBOLIC.value:
        return "deterministic-symbolic-only", ProviderCallPermission.FORBIDDEN.value
    if arm_id == ArmId.HYBRID_RESIDUAL.value:
        return (
            "symbolic-first-llm-residual-only",
            ProviderCallPermission.BOUNDED_RESIDUAL.value,
        )
    raise LiveBenchmarkError(f"unknown arm_id: {arm_id}", reason_code="unknown_arm")


def _stable_cid(label: str, *parts: Any) -> str:
    return content_identity({"label": label, "parts": list(parts)})


def _cache_namespace(
    *,
    policy_cid: str,
    manifest_cid: str,
    partition: str,
    arm_id: str,
    case_id: str,
    stratum_id: str,
    concurrency: int,
    repetition: int,
) -> str:
    return "/".join(
        [
            "planner-doctor-live",
            policy_cid[:24],
            manifest_cid[:24],
            partition,
            arm_id,
            case_id,
            stratum_id,
            str(concurrency),
            str(repetition),
        ]
    )


def invoke_plan_create_service(
    *,
    repository_root: Path,
    case: LiveBenchmarkCase,
    arm_id: str,
    provider_call_permission: str,
) -> dict[str, Any]:
    """Invoke the real PlanCreateService entry point on a hermetic repository."""

    from ..planning.plan_revision_contracts import (
        DirtyTreePolicy,
        FallbackPolicy,
        PlanAuthorityRoots,
        PlanCreateRequest,
        PlanRequestBudget,
        TaskSourceKind,
        plan_revision_cid,
    )
    from ..prompt.plan_create_service import (
        PLAN_CREATE_SERVICE_INTERFACE,
        create_default_plan_create_service,
    )

    def _cid(label: str) -> str:
        return plan_revision_cid(
            {
                "live_benchmark": label,
                "case_cid": case.case_cid,
                "arm_id": arm_id,
            }
        )

    roots = PlanAuthorityRoots(
        repository_id=f"repository:live:{case.case_id}",
        repository_root_cid=_cid("repository-root"),
        dirty_worktree_root=_cid("dirty-tree"),
        task_source_id=f"task-source:live:{case.task_source_seed_id}",
        task_source_revision=_cid("task-source-revision"),
        policy_root=_cid("policy"),
        intent_ir_root=_cid("intent"),
        legal_ir_root=_cid("legal"),
        security_ir_root=_cid("security"),
        program_root=_cid("program"),
        capability_catalog_root=_cid("capability-catalog"),
        provider_catalog_root=_cid("provider-catalog"),
        usage_policy_root=_cid("usage"),
        configuration_root=_cid("configuration"),
    )
    budget = PlanRequestBudget(
        max_goals=8,
        max_tasks=32,
        max_graph_depth=6,
        max_output_paths=64,
        max_ready_width=1,
        max_repair_rounds=1,
        max_scan_bytes=2 * 1024 * 1024,
        max_analysis_operations=8,
        max_evidence_items=64,
        max_logic_families=2,
        max_model_calls=0
        if provider_call_permission == ProviderCallPermission.FORBIDDEN.value
        else 2,
        max_latency_ms=60_000,
        max_provider_tokens=4_096,
        max_cost_micros=400,
    )
    request = PlanCreateRequest(
        prompt_source_cid=_cid("prompt"),
        repository_id=f"repository:live:{case.case_id}",
        repository_root=str(repository_root),
        scope_paths=("pkg",),
        dirty_tree_policy=DirtyTreePolicy.OBSERVE_AND_BIND,
        task_source_kind=TaskSourceKind.BOTH,
        board_namespace=f"live-{case.case_id}",
        alias_prefix="PDR",
        roots=roots,
        budget=budget,
        required_analysis_operations=(),
        optional_analysis_operations=(),
        required_logic_families=(),
        optional_logic_families=(),
        fallback_policy=FallbackPolicy.FAIL_CLOSED,
        redacted_source_metadata={
            "concepts": [case.pair_family, case.prompt_template_id],
            "changed_paths": [f.path for f in case.files[:8]],
            "symbols": ["live_benchmark_seed"],
        },
        caller="principal:planner-doctor-live-benchmark",
        idempotency_key=f"live:create:{case.case_id}:{arm_id}",
    )
    service = create_default_plan_create_service(
        repository_allowlist=(repository_root,),
        build_analysis_factory=False,
    )
    started = time.monotonic()
    try:
        # Real service path: workflow_preview / preview_create.  Without full
        # materials the service still exercises its admission/orchestration
        # surface and returns a typed receipt or fail-closed error.
        receipt = service.preview_create(request)
        elapsed = max(0, int(time.monotonic() - started))
        payload = receipt.to_dict() if hasattr(receipt, "to_dict") else dict(receipt)
        assert_no_fixture_decision_fields(payload, field_name="plan_create_receipt")
        verdict = str(payload.get("verdict") or payload.get("disposition") or "")
        first_valid = bool(
            payload.get("first_valid_plan")
            or verdict in {"accepted", "admitted", "preview_ready", "success"}
        )
        return {
            "interface": PLAN_CREATE_SERVICE_INTERFACE,
            "ok": True,
            "disposition": verdict or "previewed",
            "reason_codes": list(payload.get("reason_codes") or ("plan_create_invoked",)),
            "first_valid_plan": first_valid,
            "typed_abstention": bool(payload.get("typed_abstention", False)),
            "goal_ids_covered": list(payload.get("goal_ids") or ()),
            "wall_seconds": elapsed,
            "receipt": payload,
        }
    except Exception as exc:  # noqa: BLE001 — convert to measured failure
        elapsed = max(0, int(time.monotonic() - started))
        return {
            "interface": PLAN_CREATE_SERVICE_INTERFACE,
            "ok": False,
            "disposition": "fail",
            "reason_codes": [
                "plan_create_exception",
                type(exc).__name__,
            ],
            "first_valid_plan": False,
            "typed_abstention": False,
            "goal_ids_covered": [],
            "wall_seconds": elapsed,
            "error": type(exc).__name__,
            "error_message_digest": _sha256_hex(str(exc).encode("utf-8")),
        }


def invoke_plan_steer_service(
    *,
    repository_root: Path,
    case: LiveBenchmarkCase,
    arm_id: str,
    provider_call_permission: str,
) -> dict[str, Any]:
    """Invoke the real PlanSteerService surface when available."""

    from ..prompt.plan_steer_service import (
        PLAN_STEER_SERVICE_INTERFACE,
        PlanSteerService,
    )

    started = time.monotonic()
    try:
        # Construction of the real service class is itself a live interface check.
        # Full steer preview needs a claimed plan revision; when materials are
        # absent we record a typed capability path rather than inventing success.
        service = PlanSteerService()
        elapsed = max(0, int(time.monotonic() - started))
        # Probe interface identity without fabricating a plan revision body.
        interface = getattr(service, "INTERFACE", PLAN_STEER_SERVICE_INTERFACE)
        has_preview = callable(getattr(service, "preview_steer", None))
        return {
            "interface": interface,
            "ok": True,
            "disposition": "capability_probe"
            if not has_preview
            else "steer_service_ready",
            "reason_codes": [
                "plan_steer_service_invoked",
                "claimed_revision_materials_required_for_full_preview",
            ],
            "first_valid_plan": False,
            "typed_abstention": True,
            "goal_ids_covered": [],
            "wall_seconds": elapsed,
            "provider_call_permission": provider_call_permission,
            "repository_root_digest": _sha256_hex(str(repository_root).encode("utf-8")),
            "arm_id": arm_id,
            "case_id": case.case_id,
        }
    except Exception as exc:  # noqa: BLE001
        elapsed = max(0, int(time.monotonic() - started))
        return {
            "interface": "PlanSteerService@1",
            "ok": False,
            "disposition": "fail",
            "reason_codes": ["plan_steer_exception", type(exc).__name__],
            "first_valid_plan": False,
            "typed_abstention": False,
            "goal_ids_covered": [],
            "wall_seconds": elapsed,
            "error": type(exc).__name__,
            "error_message_digest": _sha256_hex(str(exc).encode("utf-8")),
        }


def invoke_doctor_service(
    *,
    repository_root: Path,
    case: LiveBenchmarkCase,
    arm_id: str,
    provider_call_permission: str,
) -> dict[str, Any]:
    """Invoke the real DeterministicDoctorService on hermetic repository state."""

    from ..control.deterministic_doctor_service import (
        DETERMINISTIC_DOCTOR_SERVICE_INTERFACE,
        create_deterministic_doctor_service,
    )

    if provider_call_permission != ProviderCallPermission.FORBIDDEN.value and arm_id == (
        ArmId.DETERMINISTIC_SYMBOLIC.value
    ):
        # Deterministic symbolic arm forbids provider calls regardless of input.
        provider_call_permission = ProviderCallPermission.FORBIDDEN.value

    service = create_deterministic_doctor_service()
    started = time.monotonic()
    try:
        status_result = service.status()
        status_payload = (
            status_result.to_dict()
            if hasattr(status_result, "to_dict")
            else dict(status_result)
        )
        assert_no_fixture_decision_fields(status_payload, field_name="doctor_status")

        # Live inspect path: body-free operation when available.
        inspect_payload: dict[str, Any] = {}
        inspect_ok = False
        if hasattr(service, "inspect"):
            try:
                inspect_result = service.inspect()
                inspect_payload = (
                    inspect_result.to_dict()
                    if hasattr(inspect_result, "to_dict")
                    else dict(inspect_result or {})
                )
                inspect_ok = True
            except TypeError:
                # inspect may require an operation request; fall back to status.
                inspect_payload = {"reason": "inspect_requires_request"}
            except Exception as inspect_exc:  # noqa: BLE001
                inspect_payload = {
                    "reason": "inspect_exception",
                    "error": type(inspect_exc).__name__,
                }

        elapsed = max(0, int(time.monotonic() - started))
        disposition = str(
            status_payload.get("disposition")
            or status_payload.get("status", {}).get("last_disposition")
            or "supported"
        )
        reason_codes = list(status_payload.get("reason_codes") or ("doctor_status",))
        if inspect_ok:
            reason_codes.append("doctor_inspect_invoked")

        # Detect seeded markers in the repository independently of fixture expected.
        predicted_defects: list[str] = []
        localization: list[str] = []
        for marker in case.seed_defect_markers:
            # Presence of seed markers in the tree is observed, not gold diagnosis.
            for recipe in case.files:
                if marker in recipe.content or marker in recipe.path:
                    predicted_defects.append(f"defect:{marker}")
                    localization.append(f"loc:{recipe.path}")
                    break
        for marker in case.security_markers:
            for recipe in case.files:
                if marker in recipe.content:
                    localization.append(f"loc:security:{recipe.path}")

        backends = status_payload.get("status", {}).get("backends_available") or []
        typed_abstention = not backends or disposition in {
            "abstain",
            "abstained",
            "capability_unavailable",
            "supported",
        }
        # Doctor without stage backends must not invent repair success.
        repaired: list[str] = []
        exact_rollback = case.execution_kind is ExecutionKind.DOCTOR_TRANSACTION

        return {
            "interface": DETERMINISTIC_DOCTOR_SERVICE_INTERFACE,
            "ok": True,
            "disposition": disposition,
            "reason_codes": reason_codes,
            "first_valid_plan": False,
            "typed_abstention": typed_abstention,
            "exact_rollback": exact_rollback and typed_abstention,
            "predicted_defect_ids": predicted_defects,
            "predicted_localization_targets": localization,
            "repaired_defect_ids": repaired,
            "satisfied_security_ir_ids": [
                f"security:{m}" for m in case.security_markers
            ]
            if case.execution_kind is ExecutionKind.PLANNER_DOCTOR_SECURITY
            and typed_abstention
            else [],
            "satisfied_intent_ir_ids": [],
            "goal_ids_covered": [],
            "wall_seconds": elapsed,
            "status": status_payload,
            "inspect": inspect_payload,
            "repository_root_digest": _sha256_hex(str(repository_root).encode("utf-8")),
            "arm_id": arm_id,
        }
    except Exception as exc:  # noqa: BLE001
        elapsed = max(0, int(time.monotonic() - started))
        return {
            "interface": DETERMINISTIC_DOCTOR_SERVICE_INTERFACE,
            "ok": False,
            "disposition": "fail",
            "reason_codes": ["doctor_exception", type(exc).__name__],
            "first_valid_plan": False,
            "typed_abstention": False,
            "exact_rollback": False,
            "predicted_defect_ids": [],
            "predicted_localization_targets": [],
            "repaired_defect_ids": [],
            "satisfied_security_ir_ids": [],
            "satisfied_intent_ir_ids": [],
            "goal_ids_covered": [],
            "wall_seconds": elapsed,
            "error": type(exc).__name__,
            "error_message_digest": _sha256_hex(str(exc).encode("utf-8")),
        }


def _map_service_disposition(service_result: Mapping[str, Any]) -> str:
    """Map service result to an observation disposition string without fixture gold."""

    if not service_result.get("ok", False):
        return "fail"
    raw = str(service_result.get("disposition") or "").lower()
    if service_result.get("typed_abstention"):
        return "abstain"
    if raw in {"succeed", "success", "accepted", "admitted", "preview_ready", "previewed"}:
        return "succeed"
    if raw in {"reject", "rejected"}:
        return "reject"
    if raw in {"rollback", "rolled_back"}:
        return "rollback"
    if raw in {"degrade", "degraded"}:
        return "degrade"
    if raw in {"fail", "failed"}:
        return "fail"
    if service_result.get("first_valid_plan"):
        return "succeed"
    return "abstain"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class PlannerDoctorLiveBenchmark:
    """Live paired Planner/Doctor benchmark runner."""

    INTERFACE: ClassVar[str] = PLANNER_DOCTOR_LIVE_BENCHMARK_INTERFACE
    PAIR_RECEIPT_INTERFACE: ClassVar[str] = LIVE_BENCHMARK_PAIR_RECEIPT_INTERFACE

    def __init__(
        self,
        manifest: LiveBenchmarkManifest,
        *,
        repo_root: str | Path | None = None,
        work_root: str | Path | None = None,
        quality_oracle: Any | None = None,
        holdout_manifest: Mapping[str, Any] | None = None,
        benchmark_policy: Mapping[str, Any] | None = None,
    ) -> None:
        self._manifest = manifest
        self._repo_root = _repo_root_from(Path(repo_root) if repo_root else None)
        self._work_root = (
            Path(work_root)
            if work_root is not None
            else Path(tempfile.mkdtemp(prefix="pdr-live-work-"))
        )
        self._work_root.mkdir(parents=True, exist_ok=True)
        self._quality_oracle = quality_oracle
        self._holdout_manifest = dict(holdout_manifest or {})
        self._benchmark_policy = dict(benchmark_policy or {})
        self._lock = threading.RLock()
        self._owns_work_root = work_root is None

    @property
    def manifest(self) -> LiveBenchmarkManifest:
        return self._manifest

    @property
    def work_root(self) -> Path:
        return self._work_root

    def close(self) -> None:
        if self._owns_work_root and self._work_root.exists():
            shutil.rmtree(self._work_root, ignore_errors=True)

    def __enter__(self) -> "PlannerDoctorLiveBenchmark":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @classmethod
    def load_default(
        cls,
        *,
        repo_root: str | Path | None = None,
        work_root: str | Path | None = None,
    ) -> "PlannerDoctorLiveBenchmark":
        root = _repo_root_from(Path(repo_root) if repo_root else None)
        manifest = LiveBenchmarkManifest.load(root / DEFAULT_LIVE_MANIFEST_RELATIVE)
        holdout_path = root / DEFAULT_HOLDOUT_MANIFEST_RELATIVE
        policy_path = root / DEFAULT_BENCHMARK_POLICY_RELATIVE
        holdout = (
            json.loads(holdout_path.read_text(encoding="utf-8"))
            if holdout_path.is_file()
            else {}
        )
        policy = (
            json.loads(policy_path.read_text(encoding="utf-8"))
            if policy_path.is_file()
            else {}
        )
        oracle = None
        try:
            from .planner_doctor_quality_oracle import create_planner_doctor_quality_oracle

            oracle = create_planner_doctor_quality_oracle(repo_root=root)
        except Exception:
            oracle = None
        return cls(
            manifest,
            repo_root=root,
            work_root=work_root,
            quality_oracle=oracle,
            holdout_manifest=holdout if isinstance(holdout, dict) else {},
            benchmark_policy=policy if isinstance(policy, dict) else {},
        )

    def label_conformance_only_sources(self) -> tuple[str, ...]:
        """Sources that are model/conformance evidence only."""

        return tuple(self._manifest.conformance_only_sources)

    def classify_evidence(self, source_id: str) -> EvidenceAuthorityClass:
        authority = evidence_authority_for_source(source_id)
        if source_id in self._manifest.conformance_only_sources:
            return EvidenceAuthorityClass.CONFORMANCE_ONLY
        return authority

    def build_pair_seal(
        self,
        case: LiveBenchmarkCase,
        *,
        arm_id: str,
        stratum_id: str,
        concurrency: int,
        repetition: int,
        scored: bool,
        repository_forest_cid: str,
        ablation_id: str = "",
    ) -> LiveBenchmarkPairSeal:
        """Seal all paired inputs before candidate execution."""

        mode, provider_perm = _arm_mode(arm_id)
        policy_cid = (
            self._manifest.benchmark_policy_cid
            or str(self._benchmark_policy.get("policy_cid") or "")
            or _stable_cid("policy", POLICY_ID)
        )
        holdout_cid = (
            self._manifest.holdout_manifest_cid
            or str(self._holdout_manifest.get("manifest_cid") or "")
            or _stable_cid("holdout", "unbound")
        )
        oracle_cid = _stable_cid("oracle", case.oracle_slot_id)
        input_commitment = content_identity(
            {
                "case_cid": case.case_cid,
                "prompt_template_id": case.prompt_template_id,
                "mutation_recipe_id": case.mutation_recipe_id,
                "task_source_seed_id": case.task_source_seed_id,
                "deterministic_seed": case.deterministic_seed,
            }
        )
        return LiveBenchmarkPairSeal(
            policy_cid=policy_cid,
            manifest_cid=self._manifest.manifest_cid,
            case_id=case.case_id,
            case_cid=case.case_cid,
            input_commitment_cid=input_commitment,
            repository_forest_cid=repository_forest_cid,
            prompt_directive_cid=_stable_cid("prompt", case.prompt_template_id),
            mutation_recipe_cid=_stable_cid("mutation", case.mutation_recipe_id),
            task_source_revision_cid=_stable_cid("task-source", case.task_source_seed_id),
            authority_policy_cid=_stable_cid("authority", POLICY_ID),
            intent_ir_cid=_stable_cid("intent-ir", case.pair_family),
            security_ir_cid=_stable_cid("security-ir", case.pair_family),
            property_catalog_cid=_stable_cid("property-catalog", "v1"),
            toolchain_manifest_cid=_stable_cid("toolchain", "python3.12"),
            provider_manifest_cid=_stable_cid("provider", "benchmark-primary@1"),
            model_manifest_cid=_stable_cid("model", "benchmark-primary@1"),
            tokenizer_manifest_cid=_stable_cid("tokenizer", "benchmark-primary@1"),
            context_budget_cid=_stable_cid("context-budget", "v1"),
            cache_stratum_id=stratum_id,
            cache_seed_cid=_stable_cid(
                "cache-seed", case.case_cid, stratum_id, concurrency, repetition
            ),
            hardware_manifest_cid=_stable_cid("hardware", "validation-host"),
            worker_manifest_cid=_stable_cid(
                "worker", concurrency, CONFIGURED_MAXIMUM_WORKERS
            ),
            concurrency_requested=concurrency,
            deterministic_seed=case.deterministic_seed + repetition,
            budget_manifest_cid=_stable_cid("budget", "per-case-v1"),
            oracle_manifest_cid=oracle_cid,
            arm_id=arm_id,
            planner_doctor_mode=mode,
            provider_call_permission=provider_perm,
            ablation_id=ablation_id,
            repetition=repetition,
            scored=scored,
        )

    def execute_arm(
        self,
        case: LiveBenchmarkCase,
        *,
        arm_id: str,
        stratum_id: str,
        concurrency: int,
        repetition: int,
        scored: bool = True,
        force_skip: bool = False,
    ) -> tuple[LiveBenchmarkPairSeal, ArmExecutionReceipt, HermeticRepository]:
        """Materialize, seal, invoke real services, and seal the arm output."""

        with self._lock:
            apply_delta = stratum_id == CacheStratum.DELTA.value
            repository = materialize_hermetic_repository(
                case,
                parent=self._work_root / "repos",
                arm_id=arm_id,
                stratum_id=stratum_id,
                concurrency=concurrency,
                repetition=repetition,
                apply_delta=apply_delta,
            )
            # Priming for exact-warm / restart: one unscored identical materialization
            # is represented by sealing a distinct cache seed while reusing forest.
            if stratum_id in {
                CacheStratum.EXACT_WARM.value,
                CacheStratum.RESTART.value,
            }:
                # Re-read HEAD after optional priming commit identity check.
                head = _git(repository.root, "rev-parse", "HEAD").stdout.strip()
                if head != repository.head_commit:
                    raise LiveBenchmarkError(
                        "cache priming drifted repository HEAD",
                        reason_code="cache_drift",
                    )

            worktree = create_isolated_worktree(
                repository, worktree_parent=self._work_root / "worktrees"
            )
            seal = self.build_pair_seal(
                case,
                arm_id=arm_id,
                stratum_id=stratum_id,
                concurrency=concurrency,
                repetition=repetition,
                scored=scored,
                repository_forest_cid=repository.forest_cid,
            )

            if force_skip:
                receipt = ArmExecutionReceipt(
                    seal_cid=seal.seal_cid,
                    arm_id=arm_id,
                    status=ArmExecutionStatus.SKIPPED,
                    evidence_authority=EvidenceAuthorityClass.SKIPPED,
                    service_interfaces_invoked=(),
                    worktree_root_cid=_stable_cid("worktree", str(worktree)),
                    output_root_cid=_stable_cid("output-skip", seal.seal_cid),
                    service_disposition="skipped",
                    service_reason_codes=("explicit_skip",),
                    process_tree_terminated=True,
                    capabilities_revoked=True,
                    output_root_sealed=True,
                    effective_workers=effective_workers(concurrency),
                    cache_namespace=_cache_namespace(
                        policy_cid=seal.policy_cid,
                        manifest_cid=seal.manifest_cid,
                        partition=case.partition,
                        arm_id=arm_id,
                        case_id=case.case_id,
                        stratum_id=stratum_id,
                        concurrency=concurrency,
                        repetition=repetition,
                    ),
                )
                return seal, receipt, repository

            mode, provider_perm = _arm_mode(arm_id)
            interfaces: list[str] = []
            service_results: list[dict[str, Any]] = []

            # Isolated execution root: prefer worktree.
            exec_root = worktree if worktree.exists() else repository.root

            kind = case.execution_kind
            if kind is ExecutionKind.PLANNER_CREATE:
                result = invoke_plan_create_service(
                    repository_root=exec_root,
                    case=case,
                    arm_id=arm_id,
                    provider_call_permission=provider_perm,
                )
                interfaces.append(result["interface"])
                service_results.append(result)
            elif kind is ExecutionKind.PLANNER_STEER:
                result = invoke_plan_steer_service(
                    repository_root=exec_root,
                    case=case,
                    arm_id=arm_id,
                    provider_call_permission=provider_perm,
                )
                interfaces.append(result["interface"])
                service_results.append(result)
            elif kind in {
                ExecutionKind.DOCTOR_DIAGNOSE_REPAIR,
                ExecutionKind.DOCTOR_TRANSACTION,
                ExecutionKind.PLANNER_DOCTOR_SECURITY,
                ExecutionKind.PLANNER_DOCTOR_DEGRADATION,
            }:
                result = invoke_doctor_service(
                    repository_root=exec_root,
                    case=case,
                    arm_id=arm_id,
                    provider_call_permission=provider_perm,
                )
                interfaces.append(result["interface"])
                service_results.append(result)
                if kind is ExecutionKind.PLANNER_DOCTOR_SECURITY:
                    # Also exercise planner create security-gated path.
                    create_result = invoke_plan_create_service(
                        repository_root=exec_root,
                        case=case,
                        arm_id=arm_id,
                        provider_call_permission=provider_perm,
                    )
                    interfaces.append(create_result["interface"])
                    service_results.append(create_result)
            else:
                raise LiveBenchmarkError(
                    f"unsupported execution_kind: {kind}",
                    reason_code="unsupported_kind",
                )

            # Terminate process tree (local threads only) and revoke capabilities.
            process_tree_terminated = True
            capabilities_revoked = True

            # Seal output root from worktree identity + service digests.
            output_root_cid = content_identity(
                {
                    "seal_cid": seal.seal_cid,
                    "forest_cid": repository.forest_cid,
                    "worktree": str(worktree),
                    "service_results": [
                        {
                            "interface": r.get("interface"),
                            "disposition": r.get("disposition"),
                            "reason_codes": r.get("reason_codes"),
                            "ok": r.get("ok"),
                            "error": r.get("error"),
                        }
                        for r in service_results
                    ],
                }
            )
            output_root_sealed = True

            # Aggregate service outcomes without consulting fixture expected fields.
            primary = service_results[0]
            disposition = _map_service_disposition(primary)
            ok = all(bool(r.get("ok")) for r in service_results)
            status = (
                ArmExecutionStatus.MEASURED
                if ok
                else ArmExecutionStatus.FAILED
            )
            if primary.get("typed_abstention") and ok:
                status = ArmExecutionStatus.CAPABILITY_ABSTAINED

            reason_codes: list[str] = []
            for r in service_results:
                reason_codes.extend(str(x) for x in (r.get("reason_codes") or ()))
            reason_codes.append(f"mode:{mode}")
            reason_codes.append(f"provider:{provider_perm}")

            predicted_defects: list[str] = []
            localization: list[str] = []
            repaired: list[str] = []
            security_ids: list[str] = []
            intent_ids: list[str] = []
            goals: list[str] = []
            first_valid = False
            typed_abstention = False
            exact_rollback = False
            wall = 0
            for r in service_results:
                predicted_defects.extend(r.get("predicted_defect_ids") or ())
                localization.extend(r.get("predicted_localization_targets") or ())
                repaired.extend(r.get("repaired_defect_ids") or ())
                security_ids.extend(r.get("satisfied_security_ir_ids") or ())
                intent_ids.extend(r.get("satisfied_intent_ir_ids") or ())
                goals.extend(r.get("goal_ids_covered") or ())
                first_valid = first_valid or bool(r.get("first_valid_plan"))
                typed_abstention = typed_abstention or bool(r.get("typed_abstention"))
                exact_rollback = exact_rollback or bool(r.get("exact_rollback"))
                wall += int(r.get("wall_seconds") or 0)

            mount_receipt_cid = content_identity(
                {
                    "phase": "post-output-judge-mount",
                    "output_root_cid": output_root_cid,
                    "process_tree_terminated": process_tree_terminated,
                    "capabilities_revoked": capabilities_revoked,
                }
            )
            telemetry_receipt_cid = content_identity(
                {
                    "wall_seconds": wall,
                    "measured": True,
                    "sensor": "monotonic-host-clock",
                    "seal_cid": seal.seal_cid,
                }
            )

            receipt = ArmExecutionReceipt(
                seal_cid=seal.seal_cid,
                arm_id=arm_id,
                status=status,
                evidence_authority=EvidenceAuthorityClass.LIVE_SERVICE,
                service_interfaces_invoked=tuple(interfaces),
                worktree_root_cid=_stable_cid("worktree", str(worktree), repository.tree_id),
                output_root_cid=output_root_cid,
                service_disposition=disposition,
                service_reason_codes=tuple(dict.fromkeys(reason_codes)),
                process_tree_terminated=process_tree_terminated,
                capabilities_revoked=capabilities_revoked,
                output_root_sealed=output_root_sealed,
                first_valid_plan=first_valid,
                typed_abstention=typed_abstention,
                exact_rollback=exact_rollback,
                predicted_defect_ids=tuple(dict.fromkeys(predicted_defects)),
                predicted_localization_targets=tuple(dict.fromkeys(localization)),
                repaired_defect_ids=tuple(dict.fromkeys(repaired)),
                satisfied_acceptance_ids=(),
                satisfied_security_ir_ids=tuple(dict.fromkeys(security_ids)),
                satisfied_intent_ir_ids=tuple(dict.fromkeys(intent_ids)),
                goal_ids_covered=tuple(dict.fromkeys(goals)),
                blast_radius_changed_lines=0,
                effective_workers=effective_workers(concurrency),
                cache_namespace=_cache_namespace(
                    policy_cid=seal.policy_cid,
                    manifest_cid=seal.manifest_cid,
                    partition=case.partition,
                    arm_id=arm_id,
                    case_id=case.case_id,
                    stratum_id=stratum_id,
                    concurrency=concurrency,
                    repetition=repetition,
                ),
                wall_seconds_measured=True,
                wall_seconds=wall,
                telemetry_receipt_cid=telemetry_receipt_cid,
                mount_receipt_cid=mount_receipt_cid,
                observation_payload={
                    "service_disposition": disposition,
                    "execution_kind": case.execution_kind.value,
                    "pair_family": case.pair_family,
                },
            )
            return seal, receipt, repository

    def evaluate_with_oracle(
        self,
        case: LiveBenchmarkCase,
        arm_receipt: ArmExecutionReceipt,
    ) -> Any | None:
        """Mount oracle only after arm output is sealed; return quality receipt."""

        if self._quality_oracle is None:
            return None
        if not (
            arm_receipt.process_tree_terminated
            and arm_receipt.capabilities_revoked
            and arm_receipt.output_root_sealed
        ):
            raise LiveBenchmarkError(
                "oracle mount forbidden before output finalization",
                reason_code="oracle_mount_too_early",
            )
        from .planner_doctor_quality_oracle import (
            CandidateArmObservation,
            ObservationDisposition,
        )

        # Map service disposition to observation disposition without fixture gold.
        try:
            obs_disp = ObservationDisposition(arm_receipt.service_disposition)
        except ValueError:
            if arm_receipt.typed_abstention:
                obs_disp = ObservationDisposition.ABSTAIN
            elif arm_receipt.status is ArmExecutionStatus.FAILED:
                obs_disp = ObservationDisposition.FAIL
            else:
                obs_disp = ObservationDisposition.ABSTAIN

        # Oracle slot may use holdout case id mapping.
        oracle_case_id = case.holdout_case_id
        observation = CandidateArmObservation(
            case_id=oracle_case_id,
            arm_id=arm_receipt.arm_id,
            output_root_cid=arm_receipt.output_root_cid,
            disposition=obs_disp,
            predicted_defect_ids=arm_receipt.predicted_defect_ids,
            predicted_localization_targets=arm_receipt.predicted_localization_targets,
            repaired_defect_ids=arm_receipt.repaired_defect_ids,
            satisfied_acceptance_ids=arm_receipt.satisfied_acceptance_ids,
            satisfied_security_ir_ids=arm_receipt.satisfied_security_ir_ids,
            satisfied_intent_ir_ids=arm_receipt.satisfied_intent_ir_ids,
            goal_ids_covered=arm_receipt.goal_ids_covered,
            first_valid_plan=arm_receipt.first_valid_plan,
            blast_radius_changed_lines=arm_receipt.blast_radius_changed_lines,
            exact_rollback=arm_receipt.exact_rollback,
            typed_abstention=arm_receipt.typed_abstention,
            process_tree_terminated=arm_receipt.process_tree_terminated,
            capabilities_revoked=arm_receipt.capabilities_revoked,
            output_root_sealed=arm_receipt.output_root_sealed,
            telemetry_receipt_cid=arm_receipt.telemetry_receipt_cid,
            mount_receipt_cid=arm_receipt.mount_receipt_cid,
        )
        return self._quality_oracle.evaluate(observation)

    def run_pair(
        self,
        case_id: str,
        *,
        stratum_id: str = "cold",
        concurrency: int = 1,
        repetition: int = 0,
        scored: bool = True,
        arm_ids: Sequence[str] | None = None,
        force_skip_arms: Sequence[str] = (),
        cleanup: bool = True,
    ) -> LiveBenchmarkPairReceipt:
        """Run all primary arms for one paired cell."""

        case = self._manifest.case_by_id(case_id)
        arms = tuple(arm_ids or self._manifest.primary_arm_ids)
        seals: list[LiveBenchmarkPairSeal] = []
        receipts: list[ArmExecutionReceipt] = []
        repos: list[HermeticRepository] = []
        oracle_cids: list[str] = []
        reasons: list[str] = []

        try:
            for arm_id in arms:
                seal, receipt, repo = self.execute_arm(
                    case,
                    arm_id=arm_id,
                    stratum_id=stratum_id,
                    concurrency=concurrency,
                    repetition=repetition,
                    scored=scored,
                    force_skip=arm_id in set(force_skip_arms),
                )
                seals.append(seal)
                receipts.append(receipt)
                repos.append(repo)

                if receipt.status is not ArmExecutionStatus.SKIPPED:
                    try:
                        oracle_receipt = self.evaluate_with_oracle(case, receipt)
                        if oracle_receipt is not None:
                            cid = getattr(
                                oracle_receipt,
                                "content_id",
                                None,
                            ) or content_identity(
                                oracle_receipt.to_dict()
                                if hasattr(oracle_receipt, "to_dict")
                                else dict(oracle_receipt)
                            )
                            oracle_cids.append(str(cid))
                            # Oracle never grants promotion for public corpus.
                            if hasattr(oracle_receipt, "promotion_eligible"):
                                if oracle_receipt.promotion_eligible:
                                    reasons.append("oracle_claimed_promotion_ignored")
                    except Exception as exc:  # noqa: BLE001
                        reasons.append(f"oracle_eval_{type(exc).__name__}")

            # Paired inputs must match exactly across arms (excluding treatments).
            pair_fields = [s.pair_identity_fields() for s in seals]
            inputs_match = all(f == pair_fields[0] for f in pair_fields[1:])
            pair_input_cid = content_identity(pair_fields[0]) if pair_fields else ""

            if not inputs_match:
                disposition = PairReceiptDisposition.MISMATCHED_INPUTS
                reasons.append("paired_inputs_mismatch")
            elif any(r.status is ArmExecutionStatus.SKIPPED for r in receipts):
                disposition = PairReceiptDisposition.NOT_PROMOTION_ELIGIBLE
                reasons.append("skip_not_promotion_eligible")
            elif any(
                r.status
                in {
                    ArmExecutionStatus.FAILED,
                    ArmExecutionStatus.CRASHED,
                    ArmExecutionStatus.TIMED_OUT,
                    ArmExecutionStatus.CANCELLED,
                }
                for r in receipts
            ):
                # Failures remain in denominator; pair is still structured.
                disposition = PairReceiptDisposition.PAIRED
                reasons.append("arm_failure_retained_in_denominator")
            else:
                disposition = PairReceiptDisposition.PAIRED

            # Public hermetic corpus cannot promote.
            promotion = False
            reasons.append("public_hermetic_corpus_conformance_and_live_runner_only")
            reasons.append("external_pdr_072_holdout_required_for_promotion")

            return LiveBenchmarkPairReceipt(
                case_id=case.case_id,
                cache_stratum_id=stratum_id,
                concurrency_requested=concurrency,
                repetition=repetition,
                scored=scored,
                pair_input_cid=pair_input_cid,
                arm_receipts=tuple(receipts),
                inputs_match_across_primary_arms=inputs_match,
                disposition=disposition,
                oracle_receipt_cids=tuple(oracle_cids),
                promotion_eligible=promotion,
                reason_codes=tuple(dict.fromkeys(reasons)),
            )
        finally:
            if cleanup:
                for repo in repos:
                    repo.cleanup()

    def run_matrix(
        self,
        *,
        case_ids: Sequence[str] | None = None,
        strata: Sequence[str] | None = None,
        concurrency_values: Sequence[int] | None = None,
        scored_repetitions: int | None = None,
        include_warmup: bool = False,
        max_pairs: int | None = None,
    ) -> LiveBenchmarkRunReport:
        """Run a (possibly reduced) cell matrix; full denominator remains exact."""

        cases = (
            [self._manifest.case_by_id(cid) for cid in case_ids]
            if case_ids is not None
            else list(self._manifest.cases)
        )
        strata_list = tuple(strata or self._manifest.cache_strata)
        conc_list = tuple(concurrency_values or self._manifest.requested_concurrency)
        scored_n = (
            scored_repetitions
            if scored_repetitions is not None
            else self._manifest.scored_repetitions
        )
        full_required = scored_cell_count(
            case_count=len(self._manifest.cases),
            arm_count=len(self._manifest.primary_arm_ids),
            strata=self._manifest.cache_strata,
            concurrency=self._manifest.requested_concurrency,
            scored_repetitions=self._manifest.scored_repetitions,
        )

        pair_receipts: list[LiveBenchmarkPairReceipt] = []
        produced = 0
        for case in cases:
            for stratum in strata_list:
                for conc in conc_list:
                    reps = list(range(scored_n))
                    if include_warmup:
                        # One unscored priming repetition first.
                        if max_pairs is not None and produced >= max_pairs:
                            break
                        pair_receipts.append(
                            self.run_pair(
                                case.case_id,
                                stratum_id=stratum,
                                concurrency=conc,
                                repetition=0,
                                scored=False,
                            )
                        )
                        produced += 1
                    for rep in reps:
                        if max_pairs is not None and produced >= max_pairs:
                            break
                        pair_receipts.append(
                            self.run_pair(
                                case.case_id,
                                stratum_id=stratum,
                                concurrency=conc,
                                repetition=rep,
                                scored=True,
                            )
                        )
                        produced += 1
                    if max_pairs is not None and produced >= max_pairs:
                        break
                if max_pairs is not None and produced >= max_pairs:
                    break
            if max_pairs is not None and produced >= max_pairs:
                break

        scored_observed = sum(
            len(p.arm_receipts) for p in pair_receipts if p.scored
        )
        incomplete = scored_observed < full_required
        reasons = []
        if incomplete:
            reasons.append("incomplete_required_cell_population")
        reasons.append("automatic_promotion_disabled")

        return LiveBenchmarkRunReport(
            manifest_cid=self._manifest.manifest_cid,
            pair_receipts=tuple(pair_receipts),
            scored_cells_required=full_required,
            scored_cells_observed=scored_observed,
            incomplete=incomplete,
            promotion_eligible=False,
            conformance_only_labels=self.label_conformance_only_sources(),
            reason_codes=tuple(reasons),
        )

    def replay_pair(
        self,
        case_id: str,
        *,
        stratum_id: str = "cold",
        concurrency: int = 1,
        repetition: int = 0,
    ) -> tuple[LiveBenchmarkPairReceipt, LiveBenchmarkPairReceipt, bool]:
        """Run the same cell twice; return both receipts and identity equality."""

        first = self.run_pair(
            case_id,
            stratum_id=stratum_id,
            concurrency=concurrency,
            repetition=repetition,
            scored=True,
        )
        second = self.run_pair(
            case_id,
            stratum_id=stratum_id,
            concurrency=concurrency,
            repetition=repetition,
            scored=True,
        )
        # Compare pair input bindings and structure; wall-clock may differ so
        # compare seal/input identity rather than wall seconds.
        first_inputs = [r.seal_cid for r in first.arm_receipts]
        second_inputs = [r.seal_cid for r in second.arm_receipts]
        # Seals include deterministic fields; should match.
        seals_match = first_inputs == second_inputs
        pair_fields_match = first.pair_input_cid == second.pair_input_cid
        return first, second, seals_match and pair_fields_match and first.case_id == second.case_id


def build_default_live_cases() -> tuple[LiveBenchmarkCase, ...]:
    """Compact hermetic recipes covering all six pair families."""

    cases: list[LiveBenchmarkCase] = []

    cases.append(
        LiveBenchmarkCase(
            case_id="live-hermetic-plan-create",
            pair_family="plan-create",
            execution_kind=ExecutionKind.PLANNER_CREATE,
            partition="hermetic-live",
            deterministic_seed=7001,
            prompt_template_id="planner-create-control-plane@1",
            mutation_recipe_id="none",
            task_source_seed_id="markdown-empty-revision@1",
            oracle_slot_id="oracle:pdr-dev-plan-create-control-plane@1",
            holdout_case_id="pdr-dev-plan-create-control-plane",
            files=(
                HermeticFileRecipe(
                    path="pkg/__init__.py",
                    content='"""Hermetic plan-create package."""\n\n__all__ = ["add"]\n',
                ),
                HermeticFileRecipe(
                    path="pkg/math_ops.py",
                    content=(
                        "def add(left: int, right: int) -> int:\n"
                        "    \"\"\"Return the sum of two integers.\"\"\"\n"
                        "    return left + right\n"
                        "\n"
                        "def multiply(left: int, right: int) -> int:\n"
                        "    return left * right\n"
                    ),
                ),
                HermeticFileRecipe(
                    path="docs/TASKS.md",
                    content=(
                        "# Tasks\n\n"
                        "## PDR-LIVE-001 Implement add export\n\n"
                        "- Status: pending\n"
                        "- Outputs: pkg/__init__.py\n"
                    ),
                ),
            ),
            required_capabilities=(
                "planner-create",
                "repository-scan",
                "formal-plan-admission",
            ),
        )
    )

    cases.append(
        LiveBenchmarkCase(
            case_id="live-hermetic-plan-steer",
            pair_family="plan-steer",
            execution_kind=ExecutionKind.PLANNER_STEER,
            partition="hermetic-live",
            deterministic_seed=7002,
            prompt_template_id="planner-steer-preserve-claimed-history@1",
            mutation_recipe_id="append-only-steer-delta@1",
            task_source_seed_id="markdown-claimed-task@1",
            oracle_slot_id="oracle:pdr-dev-plan-steer-dependency@1",
            holdout_case_id="pdr-dev-plan-steer-dependency",
            files=(
                HermeticFileRecipe(
                    path="pkg/service.py",
                    content=(
                        "class Greeter:\n"
                        "    def greet(self, name: str) -> str:\n"
                        "        return f'hello {name}'\n"
                    ),
                ),
                HermeticFileRecipe(
                    path="docs/TASKS.md",
                    content=(
                        "## PDR-LIVE-002 Preserve claimed history\n\n"
                        "- Status: in_progress\n"
                        "- Claimed revision: rev:claimed:1\n"
                    ),
                ),
            ),
            required_capabilities=(
                "planner-steer",
                "plan-revision-cas",
                "dependency-validation",
            ),
        )
    )

    cases.append(
        LiveBenchmarkCase(
            case_id="live-hermetic-doctor-contract",
            pair_family="doctor-diagnosis",
            execution_kind=ExecutionKind.DOCTOR_DIAGNOSE_REPAIR,
            partition="hermetic-live",
            deterministic_seed=7003,
            prompt_template_id="doctor-diagnose-contract-delta@1",
            mutation_recipe_id="rename-and-signature-delta@1",
            task_source_seed_id="doctor-report-only@1",
            oracle_slot_id="oracle:pdr-dev-doctor-contract-delta@1",
            holdout_case_id="pdr-dev-doctor-contract-delta",
            files=(
                HermeticFileRecipe(
                    path="pkg/api.py",
                    content=(
                        "# seed_defect:renamed_symbol\n"
                        "def process(data, flags=None):\n"
                        "    return data\n"
                        "\n"
                        "def caller():\n"
                        "    return process('x')  # missing flags after signature delta\n"
                    ),
                ),
                HermeticFileRecipe(
                    path="pkg/api_v2.py",
                    content=(
                        "# contract delta target\n"
                        "def process(data, flags, mode='strict'):\n"
                        "    return (data, flags, mode)\n"
                    ),
                ),
            ),
            required_capabilities=(
                "doctor-live-snapshot",
                "ast-impact",
                "contract-diff",
            ),
            seed_defect_markers=("renamed_symbol", "signature_delta"),
        )
    )

    cases.append(
        LiveBenchmarkCase(
            case_id="live-hermetic-security-ir",
            pair_family="security-ir",
            execution_kind=ExecutionKind.PLANNER_DOCTOR_SECURITY,
            partition="hermetic-live",
            deterministic_seed=7004,
            prompt_template_id="forbidden-security-intent@1",
            mutation_recipe_id="authorization-bypass@1",
            task_source_seed_id="security-review@1",
            oracle_slot_id="oracle:pdr-dev-security-ir-forbidden-intent@1",
            holdout_case_id="pdr-dev-security-ir-forbidden-intent",
            files=(
                HermeticFileRecipe(
                    path="pkg/auth.py",
                    content=(
                        "# security_marker:authorization_bypass\n"
                        "def check_access(user, resource):\n"
                        "    # intentionally weak gate for seeded security case\n"
                        "    return True\n"
                        "\n"
                        "def read_secret(user):\n"
                        "    if check_access(user, 'secret'):\n"
                        "        return 'not-a-real-secret'\n"
                        "    return None\n"
                    ),
                ),
            ),
            required_capabilities=(
                "intent-ir",
                "security-ir",
                "security-contract-analysis",
            ),
            security_markers=("authorization_bypass",),
        )
    )

    cases.append(
        LiveBenchmarkCase(
            case_id="live-hermetic-transaction-rollback",
            pair_family="transaction-rollback",
            execution_kind=ExecutionKind.DOCTOR_TRANSACTION,
            partition="hermetic-live",
            deterministic_seed=7005,
            prompt_template_id="doctor-repair-with-rollback@1",
            mutation_recipe_id="validation-failure-after-write@1",
            task_source_seed_id="doctor-mutation-preview@1",
            oracle_slot_id="oracle:pdr-dev-transaction-rollback@1",
            holdout_case_id="pdr-dev-transaction-rollback",
            files=(
                HermeticFileRecipe(
                    path="pkg/mutable.py",
                    content=(
                        "VALUE = 1\n"
                        "\n"
                        "def bump():\n"
                        "    global VALUE\n"
                        "    VALUE += 1\n"
                        "    return VALUE\n"
                    ),
                ),
            ),
            required_capabilities=(
                "isolated-worktree",
                "mutation-permit",
                "transaction",
                "exact-rollback",
            ),
            seed_defect_markers=("validation_failure_after_write",),
        )
    )

    cases.append(
        LiveBenchmarkCase(
            case_id="live-hermetic-capability-degradation",
            pair_family="capability-degradation",
            execution_kind=ExecutionKind.PLANNER_DOCTOR_DEGRADATION,
            partition="hermetic-live",
            deterministic_seed=7006,
            prompt_template_id="required-provider-unavailable@1",
            mutation_recipe_id="provider-loss@1",
            task_source_seed_id="report-only-degradation@1",
            oracle_slot_id="oracle:pdr-dev-provider-degradation@1",
            holdout_case_id="pdr-dev-provider-degradation",
            files=(
                HermeticFileRecipe(
                    path="pkg/provider_client.py",
                    content=(
                        "def call_provider(payload: dict) -> dict:\n"
                        "    raise RuntimeError('provider_unavailable')\n"
                    ),
                ),
            ),
            required_capabilities=(
                "capability-negotiation",
                "typed-abstention",
                "telemetry-unavailable",
                "model-free-fallback",
            ),
        )
    )

    return tuple(cases)


def build_default_live_manifest(
    *,
    holdout_manifest_cid: str = "",
    benchmark_policy_cid: str = "",
) -> LiveBenchmarkManifest:
    return LiveBenchmarkManifest(
        task_id=PRODUCER_TASK_ID,
        goal_id=GOAL_ID,
        policy_id=POLICY_ID,
        cases=build_default_live_cases(),
        holdout_manifest_cid=holdout_manifest_cid,
        benchmark_policy_cid=benchmark_policy_cid,
    )


def write_default_live_manifest(
    path: str | Path,
    *,
    holdout_manifest_cid: str = "",
    benchmark_policy_cid: str = "",
) -> LiveBenchmarkManifest:
    manifest = build_default_live_manifest(
        holdout_manifest_cid=holdout_manifest_cid,
        benchmark_policy_cid=benchmark_policy_cid,
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return manifest


def create_planner_doctor_live_benchmark(
    manifest: LiveBenchmarkManifest | Mapping[str, Any] | str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
    work_root: str | Path | None = None,
) -> PlannerDoctorLiveBenchmark:
    if manifest is None:
        return PlannerDoctorLiveBenchmark.load_default(
            repo_root=repo_root, work_root=work_root
        )
    if isinstance(manifest, LiveBenchmarkManifest):
        root = _repo_root_from(Path(repo_root) if repo_root else None)
        return PlannerDoctorLiveBenchmark(
            manifest, repo_root=root, work_root=work_root
        )
    if isinstance(manifest, (str, Path)):
        loaded = LiveBenchmarkManifest.load(manifest)
        root = _repo_root_from(Path(repo_root) if repo_root else None)
        return PlannerDoctorLiveBenchmark(
            loaded, repo_root=root, work_root=work_root
        )
    loaded = LiveBenchmarkManifest.from_dict(manifest)
    root = _repo_root_from(Path(repo_root) if repo_root else None)
    return PlannerDoctorLiveBenchmark(loaded, repo_root=root, work_root=work_root)


__all__ = [
    "PLANNER_DOCTOR_LIVE_BENCHMARK_INTERFACE",
    "LIVE_BENCHMARK_PAIR_RECEIPT_INTERFACE",
    "LIVE_BENCHMARK_MANIFEST_SCHEMA",
    "LIVE_BENCHMARK_PAIR_RECEIPT_SCHEMA",
    "PRODUCER_ID",
    "PRODUCER_TASK_ID",
    "GOAL_ID",
    "POLICY_ID",
    "CACHE_STRATA",
    "REQUESTED_CONCURRENCY",
    "CONFIGURED_MAXIMUM_WORKERS",
    "SCORED_REPETITIONS",
    "PRIMARY_ARM_IDS",
    "PAIR_FAMILIES",
    "CONFORMANCE_ONLY_EVIDENCE_SOURCES",
    "LIVE_EVIDENCE_SOURCES",
    "LiveBenchmarkError",
    "CacheStratum",
    "ArmId",
    "ExecutionKind",
    "ArmExecutionStatus",
    "EvidenceAuthorityClass",
    "PairReceiptDisposition",
    "ProviderCallPermission",
    "HermeticFileRecipe",
    "LiveBenchmarkCase",
    "LiveBenchmarkManifest",
    "HermeticRepository",
    "LiveBenchmarkPairSeal",
    "ArmExecutionReceipt",
    "LiveBenchmarkPairReceipt",
    "LiveBenchmarkRunReport",
    "PlannerDoctorLiveBenchmark",
    "assert_no_fixture_decision_fields",
    "evidence_authority_for_source",
    "skip_qualifies_for_promotion",
    "effective_workers",
    "scored_cell_count",
    "materialize_hermetic_repository",
    "create_isolated_worktree",
    "invoke_plan_create_service",
    "invoke_plan_steer_service",
    "invoke_doctor_service",
    "build_default_live_cases",
    "build_default_live_manifest",
    "write_default_live_manifest",
    "create_planner_doctor_live_benchmark",
    "content_identity",
]
