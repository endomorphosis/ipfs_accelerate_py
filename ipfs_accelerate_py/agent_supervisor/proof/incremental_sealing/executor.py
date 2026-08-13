"""Execute incremental plans with fresh cache verification (IPS-035).

Every reusable candidate is fetched, rehash-checked, and cryptographically or
signature verified under the current admission policy before reuse.  There is
no cache fast path that bypasses verification.  Invalidated and added units
are proved, verified, and admitted; tombstones/invalidations are recorded for
rejected or removed material.

Reused and newly proved sets must exactly cover the plan's required units.
Stale, poisoned, corrupt, mismatched, and simulated evidence is rejected.
Cancellation and unavailable outcomes never proceed to aggregation.

Interfaces: ``IncrementalPlanExecutor``, ``IncrementalProofResult``,
``execute_incremental_plan``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.admission import (
    AdmissionDecision,
    AdmissionError,
    AdmissionPolicy,
    CacheAdmissionRecord,
    EvidenceCandidate,
    EvidenceVerifier,
    VerifierFn,
    issue_cache_admission_record,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.planner import (
    IncrementalProofPlan,
    UnitPlanKind,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.process_control import (
    CancellationToken,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.scheduling import (
    AdmissionVerdict,
    ProofResourcePolicy,
    ProofWorkItem,
    ProofWorkScheduler,
    WorkClass,
)
from ipfs_datasets_py.logic.zkp.incremental_sealing.evidence import (
    ProofMode,
    ProofTerminalStatus,
)

EVIDENCE_SUBSET: Final[str] = "ips/incremental-execution@1"
CACHE_REVERIFY_EVIDENCE_SUBSET: Final[str] = "ips/cache-reverification@1"
RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "incremental-proof-result@1"
)
UNIT_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "unit-execution-record@1"
)
REVERIFY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "cache-reverification@1"
)

# Terminal prove statuses that block aggregation (never success).
_BLOCKING_PROVE_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "cancelled",
        "unavailable",
        "timeout",
        "failed",
        "proof_failed",
        "invalid",
        "unknown",
        "ambiguous",
        "simulated",
        "disproved",
        "verification_failed",
    }
)


class ExecutorError(ValueError):
    """Fail-closed incremental plan execution contract violation."""


class ExecutionStatus(str, Enum):
    """Closed overall plan-execution outcomes."""

    COMPLETED = "completed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    INCOMPLETE = "incomplete"


class UnitDisposition(str, Enum):
    """Per-unit execution disposition."""

    REUSED = "reused"
    NEWLY_PROVED = "newly_proved"
    REMOVED = "removed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    TOMBSTONED = "tombstoned"


class ExecutionReasonCode(str, Enum):
    """Stable reason codes for unit and plan outcomes."""

    REUSED_AFTER_FRESH_VERIFICATION = "reused_after_fresh_verification"
    NEWLY_PROVED_AND_ADMITTED = "newly_proved_and_admitted"
    REMOVED = "removed"
    MISSING_CANDIDATE = "missing_candidate"
    MISSING_PROVE_PAYLOAD = "missing_prove_payload"
    STALE_EVIDENCE = "stale_evidence"
    POISONED_EVIDENCE = "poisoned_evidence"
    CORRUPT_EVIDENCE = "corrupt_evidence"
    CACHE_KEY_MISMATCH = "cache_key_mismatch"
    PUBLIC_INPUT_MISMATCH = "public_input_mismatch"
    SIMULATED_EVIDENCE = "simulated_evidence"
    REHASH_FAILED = "rehash_failed"
    ADMISSION_REJECTED = "admission_rejected"
    PROVE_CANCELLED = "prove_cancelled"
    PROVE_UNAVAILABLE = "prove_unavailable"
    PROVE_FAILED = "prove_failed"
    PROVE_TIMEOUT = "prove_timeout"
    RESOURCE_UNAVAILABLE = "resource_unavailable"
    PLAN_CANCELLED = "plan_cancelled"
    COVERAGE_INCOMPLETE = "coverage_incomplete"
    MALFORMED_INPUT = "malformed_input"


def closed_execution_statuses() -> frozenset[str]:
    return frozenset(item.value for item in ExecutionStatus)


def closed_unit_dispositions() -> frozenset[str]:
    return frozenset(item.value for item in UnitDisposition)


def closed_execution_reason_codes() -> frozenset[str]:
    return frozenset(item.value for item in ExecutionReasonCode)


def _sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _require_nonempty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExecutorError(f"{field_name} must be a non-empty string")
    return value.strip()


def _require_bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise ExecutorError(f"{field_name} must be a boolean")
    return value


def _require_nonneg_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExecutorError(f"{field_name} must be a non-negative int")
    return value


def _sorted_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


def _constant_time_equal(left: str, right: str) -> bool:
    """Length-safe constant-time equality for digests and cache keys."""

    if not isinstance(left, str) or not isinstance(right, str):
        return False
    if len(left) != len(right):
        return False
    return hmac.compare_digest(left, right)


def _coerce_proof_mode(value: ProofMode | str) -> ProofMode:
    if isinstance(value, ProofMode):
        return value
    if not isinstance(value, str) or not value.strip():
        raise ExecutorError("proof_mode must be a closed ProofMode string")
    try:
        return ProofMode(value.strip())
    except ValueError as exc:
        raise ExecutorError(f"unknown proof_mode {value!r}") from exc


def _coerce_terminal_status(value: ProofTerminalStatus | str) -> ProofTerminalStatus:
    if isinstance(value, ProofTerminalStatus):
        return value
    if not isinstance(value, str) or not value.strip():
        raise ExecutorError("terminal_status must be a closed status string")
    try:
        return ProofTerminalStatus(value.strip())
    except ValueError as exc:
        raise ExecutorError(f"unknown terminal_status {value!r}") from exc


@dataclass(frozen=True, slots=True)
class CacheCandidatePayload:
    """Immutable cache candidate hint that still requires fresh verification.

    Kit (or an in-memory test store) may supply this material.  Presence never
    authorizes reuse; only accelerate rehash + admission may.
    """

    unit_id: str
    cache_key: str
    artifact_cid: str
    artifact_bytes: bytes
    expected_digest: str
    evidence: Any
    proof_system_id: str
    public_input_cid: str
    proof_object_cid: str = "n/a"
    proof_mode: ProofMode | str = ProofMode.INTEGRITY_ONLY
    terminal_status: ProofTerminalStatus | str = (
        ProofTerminalStatus.INTEGRITY_VERIFIED
    )
    observed_public_input_cid: str | None = None
    recomputed_cache_key: str | None = None
    stale: bool = False
    poisoned: bool = False
    corrupt: bool = False
    simulated: bool = False
    cache_key_mismatch: bool = False
    logical_epoch: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "unit_id", _require_nonempty_str(self.unit_id, "unit_id")
        )
        object.__setattr__(
            self, "cache_key", _require_nonempty_str(self.cache_key, "cache_key")
        )
        object.__setattr__(
            self,
            "artifact_cid",
            _require_nonempty_str(self.artifact_cid, "artifact_cid"),
        )
        if not isinstance(self.artifact_bytes, (bytes, bytearray)):
            raise ExecutorError("artifact_bytes must be bytes")
        object.__setattr__(self, "artifact_bytes", bytes(self.artifact_bytes))
        object.__setattr__(
            self,
            "expected_digest",
            _require_nonempty_str(self.expected_digest, "expected_digest"),
        )
        object.__setattr__(
            self,
            "proof_system_id",
            _require_nonempty_str(self.proof_system_id, "proof_system_id"),
        )
        object.__setattr__(
            self,
            "public_input_cid",
            _require_nonempty_str(self.public_input_cid, "public_input_cid"),
        )
        object.__setattr__(
            self,
            "proof_object_cid",
            str(self.proof_object_cid or "n/a").strip() or "n/a",
        )
        object.__setattr__(self, "proof_mode", _coerce_proof_mode(self.proof_mode))
        object.__setattr__(
            self, "terminal_status", _coerce_terminal_status(self.terminal_status)
        )
        for flag in (
            "stale",
            "poisoned",
            "corrupt",
            "simulated",
            "cache_key_mismatch",
        ):
            object.__setattr__(self, flag, _require_bool(getattr(self, flag), flag))
        object.__setattr__(
            self, "logical_epoch", _require_nonneg_int(self.logical_epoch, "logical_epoch")
        )
        if not isinstance(self.metadata, Mapping):
            raise ExecutorError("metadata must be a mapping")
        if self.recomputed_cache_key is not None:
            object.__setattr__(
                self,
                "recomputed_cache_key",
                _require_nonempty_str(
                    self.recomputed_cache_key, "recomputed_cache_key"
                ),
            )

    def observed_digest(self) -> str:
        return _sha256_hex(self.artifact_bytes)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": REVERIFY_SCHEMA,
            "evidence_subset": CACHE_REVERIFY_EVIDENCE_SUBSET,
            "unit_id": self.unit_id,
            "cache_key": self.cache_key,
            "artifact_cid": self.artifact_cid,
            "artifact_byte_length": len(self.artifact_bytes),
            "expected_digest": self.expected_digest,
            "observed_digest": self.observed_digest(),
            "proof_system_id": self.proof_system_id,
            "public_input_cid": self.public_input_cid,
            "proof_object_cid": self.proof_object_cid,
            "proof_mode": self.proof_mode.value,
            "terminal_status": self.terminal_status.value,
            "stale": self.stale,
            "poisoned": self.poisoned,
            "corrupt": self.corrupt,
            "simulated": self.simulated,
            "cache_key_mismatch": self.cache_key_mismatch,
            "requires_fresh_verification": True,
            "logical_epoch": self.logical_epoch,
        }


@dataclass(frozen=True, slots=True)
class FreshProvePayload:
    """Closed material for proving an invalidated or added unit.

    ``prove_status`` is the pre-admission prove outcome.  Only ``proved`` may
    continue to verification and cache admission.
    """

    unit_id: str
    evidence: Any
    proof_system_id: str
    public_input_cid: str
    proof_object_cid: str = "n/a"
    proof_mode: ProofMode | str = ProofMode.INTEGRITY_ONLY
    terminal_status: ProofTerminalStatus | str = (
        ProofTerminalStatus.INTEGRITY_VERIFIED
    )
    prove_status: str = "proved"
    expected_digest: str | None = None
    observed_digest: str | None = None
    observed_public_input_cid: str | None = None
    cpu: int = 1
    memory_mb: int = 64
    gpu: int = 0
    simulated_gpu: bool = False
    logical_epoch: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "unit_id", _require_nonempty_str(self.unit_id, "unit_id")
        )
        object.__setattr__(
            self,
            "proof_system_id",
            _require_nonempty_str(self.proof_system_id, "proof_system_id"),
        )
        object.__setattr__(
            self,
            "public_input_cid",
            _require_nonempty_str(self.public_input_cid, "public_input_cid"),
        )
        object.__setattr__(
            self,
            "proof_object_cid",
            str(self.proof_object_cid or "n/a").strip() or "n/a",
        )
        object.__setattr__(self, "proof_mode", _coerce_proof_mode(self.proof_mode))
        object.__setattr__(
            self, "terminal_status", _coerce_terminal_status(self.terminal_status)
        )
        object.__setattr__(
            self,
            "prove_status",
            _require_nonempty_str(self.prove_status, "prove_status").casefold(),
        )
        object.__setattr__(self, "cpu", _require_nonneg_int(self.cpu, "cpu"))
        object.__setattr__(
            self, "memory_mb", _require_nonneg_int(self.memory_mb, "memory_mb")
        )
        object.__setattr__(self, "gpu", _require_nonneg_int(self.gpu, "gpu"))
        object.__setattr__(
            self, "simulated_gpu", _require_bool(self.simulated_gpu, "simulated_gpu")
        )
        object.__setattr__(
            self, "logical_epoch", _require_nonneg_int(self.logical_epoch, "logical_epoch")
        )
        if not isinstance(self.metadata, Mapping):
            raise ExecutorError("metadata must be a mapping")


@dataclass(frozen=True, slots=True)
class UnitExecutionRecord:
    """Per-unit execution result with exact disposition and admission state."""

    unit_id: str
    disposition: UnitDisposition
    reason_code: str
    message: str
    freshly_verified: bool
    ready_for_aggregation: bool
    admission: AdmissionDecision | None = None
    cache_admission_record: CacheAdmissionRecord | None = None
    reverify_digest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "unit_id", _require_nonempty_str(self.unit_id, "unit_id")
        )
        if not isinstance(self.disposition, UnitDisposition):
            object.__setattr__(
                self, "disposition", UnitDisposition(str(self.disposition))
            )
        object.__setattr__(
            self,
            "reason_code",
            _require_nonempty_str(self.reason_code, "reason_code"),
        )
        object.__setattr__(self, "message", str(self.message))
        object.__setattr__(
            self,
            "freshly_verified",
            _require_bool(self.freshly_verified, "freshly_verified"),
        )
        object.__setattr__(
            self,
            "ready_for_aggregation",
            _require_bool(self.ready_for_aggregation, "ready_for_aggregation"),
        )
        if self.ready_for_aggregation and self.disposition not in {
            UnitDisposition.REUSED,
            UnitDisposition.NEWLY_PROVED,
            UnitDisposition.REMOVED,
        }:
            raise ExecutorError(
                "only reused, newly_proved, or removed units may be ready "
                "for aggregation"
            )
        if self.disposition in {
            UnitDisposition.REUSED,
            UnitDisposition.NEWLY_PROVED,
        }:
            if self.cache_admission_record is None or not self.freshly_verified:
                raise ExecutorError(
                    f"{self.disposition.value} units require fresh verification "
                    "and a CacheAdmissionRecord"
                )
            if not self.ready_for_aggregation:
                raise ExecutorError(
                    f"{self.disposition.value} success requires "
                    "ready_for_aggregation=True"
                )
        if self.disposition in {
            UnitDisposition.CANCELLED,
            UnitDisposition.UNAVAILABLE,
            UnitDisposition.REJECTED,
            UnitDisposition.FAILED,
        }:
            if self.ready_for_aggregation:
                raise ExecutorError(
                    f"{self.disposition.value} units cannot proceed to aggregation"
                )
            if self.cache_admission_record is not None:
                raise ExecutorError(
                    f"{self.disposition.value} units must not carry admission records"
                )

    def to_canonical(self) -> dict[str, Any]:
        admission_payload = None
        if self.admission is not None:
            admission_payload = self.admission.to_canonical()
        record_payload = None
        if self.cache_admission_record is not None:
            record_payload = self.cache_admission_record.to_canonical()
        return {
            "schema": UNIT_RECORD_SCHEMA,
            "unit_id": self.unit_id,
            "disposition": self.disposition.value,
            "reason_code": self.reason_code,
            "message": self.message,
            "freshly_verified": self.freshly_verified,
            "ready_for_aggregation": self.ready_for_aggregation,
            "reverify_digest": self.reverify_digest,
            "admission": admission_payload,
            "cache_admission_record": record_payload,
        }


@dataclass(frozen=True, slots=True)
class IncrementalProofResult:
    """Closed result of executing one incremental plan."""

    schema: str
    evidence_subset: str
    status: ExecutionStatus
    plan_cid: str
    reused_unit_ids: tuple[str, ...]
    newly_proved_unit_ids: tuple[str, ...]
    removed_unit_ids: tuple[str, ...]
    rejected_unit_ids: tuple[str, ...]
    cancelled_unit_ids: tuple[str, ...]
    unavailable_unit_ids: tuple[str, ...]
    failed_unit_ids: tuple[str, ...]
    required_unit_ids: tuple[str, ...]
    unit_records: tuple[UnitExecutionRecord, ...]
    admissions: tuple[CacheAdmissionRecord, ...]
    tombstones: tuple[str, ...]
    ready_for_aggregation: bool
    coverage_complete: bool
    message: str
    cache_reverification_subset: str = CACHE_REVERIFY_EVIDENCE_SUBSET

    def __post_init__(self) -> None:
        if not isinstance(self.status, ExecutionStatus):
            object.__setattr__(self, "status", ExecutionStatus(str(self.status)))
        object.__setattr__(
            self, "reused_unit_ids", _sorted_unique(self.reused_unit_ids)
        )
        object.__setattr__(
            self, "newly_proved_unit_ids", _sorted_unique(self.newly_proved_unit_ids)
        )
        object.__setattr__(
            self, "removed_unit_ids", _sorted_unique(self.removed_unit_ids)
        )
        object.__setattr__(
            self, "rejected_unit_ids", _sorted_unique(self.rejected_unit_ids)
        )
        object.__setattr__(
            self, "cancelled_unit_ids", _sorted_unique(self.cancelled_unit_ids)
        )
        object.__setattr__(
            self, "unavailable_unit_ids", _sorted_unique(self.unavailable_unit_ids)
        )
        object.__setattr__(
            self, "failed_unit_ids", _sorted_unique(self.failed_unit_ids)
        )
        object.__setattr__(
            self, "required_unit_ids", _sorted_unique(self.required_unit_ids)
        )
        object.__setattr__(self, "tombstones", _sorted_unique(self.tombstones))
        object.__setattr__(
            self,
            "ready_for_aggregation",
            _require_bool(self.ready_for_aggregation, "ready_for_aggregation"),
        )
        object.__setattr__(
            self,
            "coverage_complete",
            _require_bool(self.coverage_complete, "coverage_complete"),
        )
        # Fail-closed invariants: aggregation is gated on exact coverage only.
        if self.ready_for_aggregation:
            if self.status is not ExecutionStatus.COMPLETED:
                raise ExecutorError(
                    "ready_for_aggregation requires status=completed"
                )
            if not self.coverage_complete:
                raise ExecutorError(
                    "ready_for_aggregation requires complete coverage"
                )
            if self.cancelled_unit_ids or self.unavailable_unit_ids:
                raise ExecutorError(
                    "cancelled/unavailable outcomes cannot proceed to aggregation"
                )
            if self.rejected_unit_ids or self.failed_unit_ids:
                raise ExecutorError(
                    "rejected/failed required units cannot proceed to aggregation"
                )
            covered = set(self.reused_unit_ids) | set(self.newly_proved_unit_ids)
            if covered != set(self.required_unit_ids):
                raise ExecutorError(
                    "reused and newly_proved sets must exactly cover required units"
                )
            if set(self.reused_unit_ids) & set(self.newly_proved_unit_ids):
                raise ExecutorError(
                    "reused and newly_proved sets must be disjoint"
                )
            for record in self.unit_records:
                if record.unit_id not in self.required_unit_ids:
                    continue
                if record.disposition not in {
                    UnitDisposition.REUSED,
                    UnitDisposition.NEWLY_PROVED,
                }:
                    raise ExecutorError(
                        "ready_for_aggregation requires every required unit to be "
                        "reused or newly_proved"
                    )
                if (
                    not record.freshly_verified
                    or record.cache_admission_record is None
                    or not record.ready_for_aggregation
                ):
                    raise ExecutorError(
                        "ready_for_aggregation requires fresh verified admission "
                        f"for required unit {record.unit_id!r}"
                    )
        if self.cancelled_unit_ids and self.ready_for_aggregation:
            raise ExecutorError("cancelled outcomes cannot proceed to aggregation")
        if self.unavailable_unit_ids and self.ready_for_aggregation:
            raise ExecutorError("unavailable outcomes cannot proceed to aggregation")

    @property
    def success(self) -> bool:
        return (
            self.status is ExecutionStatus.COMPLETED
            and self.ready_for_aggregation
            and self.coverage_complete
        )

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_subset": self.evidence_subset,
            "cache_reverification_subset": self.cache_reverification_subset,
            "status": self.status.value,
            "plan_cid": self.plan_cid,
            "reused_unit_ids": list(self.reused_unit_ids),
            "newly_proved_unit_ids": list(self.newly_proved_unit_ids),
            "removed_unit_ids": list(self.removed_unit_ids),
            "rejected_unit_ids": list(self.rejected_unit_ids),
            "cancelled_unit_ids": list(self.cancelled_unit_ids),
            "unavailable_unit_ids": list(self.unavailable_unit_ids),
            "failed_unit_ids": list(self.failed_unit_ids),
            "required_unit_ids": list(self.required_unit_ids),
            "unit_records": [item.to_canonical() for item in self.unit_records],
            "admissions": [item.to_canonical() for item in self.admissions],
            "tombstones": list(self.tombstones),
            "ready_for_aggregation": self.ready_for_aggregation,
            "coverage_complete": self.coverage_complete,
            "success": self.success,
            "message": self.message,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_canonical())


def plan_required_unit_ids(plan: IncrementalProofPlan) -> tuple[str, ...]:
    """Return the exact set of units that must be reused or newly proved."""

    if not isinstance(plan, IncrementalProofPlan):
        raise ExecutorError("plan must be IncrementalProofPlan")
    required: set[str] = set()
    required.update(plan.reusable_unit_ids)
    required.update(plan.invalidated_unit_ids)
    required.update(plan.added_unit_ids)
    # Reject-reuse / incomplete-key units still require evidence for the seal.
    for item in plan.units:
        if item.kind is UnitPlanKind.REMOVE:
            continue
        if item.kind in {
            UnitPlanKind.REUSE,
            UnitPlanKind.REPROVE,
            UnitPlanKind.PROVE_NEW,
            UnitPlanKind.REJECT_REUSE,
        }:
            required.add(item.unit_id)
    return _sorted_unique(tuple(required))


def plan_prove_unit_ids(plan: IncrementalProofPlan) -> tuple[str, ...]:
    """Units that must be freshly proved (not reuse candidates)."""

    if not isinstance(plan, IncrementalProofPlan):
        raise ExecutorError("plan must be IncrementalProofPlan")
    prove: set[str] = set(plan.invalidated_unit_ids) | set(plan.added_unit_ids)
    for item in plan.units:
        if item.kind in {
            UnitPlanKind.REPROVE,
            UnitPlanKind.PROVE_NEW,
            UnitPlanKind.REJECT_REUSE,
        }:
            prove.add(item.unit_id)
    # Never prove a unit that is still planned for pure reuse.
    prove -= set(plan.reusable_unit_ids)
    return _sorted_unique(tuple(prove))


class IncrementalPlanExecutor:
    """Execute an :class:`IncrementalProofPlan` under verified admission.

    Dependencies are injectable so hermetic tests never require a live kit
    store or external prover.  Every reuse path still runs rehash + admission.
    """

    def __init__(
        self,
        *,
        candidates: Mapping[str, CacheCandidatePayload] | None = None,
        prove_payloads: Mapping[str, FreshProvePayload] | None = None,
        admission_policy: AdmissionPolicy | None = None,
        verifier: VerifierFn | None = None,
        resource_policy: ProofResourcePolicy | None = None,
        cancellation: CancellationToken | None = None,
        candidate_lookup: Callable[[str], CacheCandidatePayload | None] | None = None,
        prove_lookup: Callable[[str], FreshProvePayload | None] | None = None,
        on_tombstone: Callable[[str, str], None] | None = None,
        on_admission: Callable[[CacheAdmissionRecord], None] | None = None,
    ) -> None:
        self._candidates: MutableMapping[str, CacheCandidatePayload] = dict(
            candidates or {}
        )
        self._prove_payloads: MutableMapping[str, FreshProvePayload] = dict(
            prove_payloads or {}
        )
        self._admission_policy = admission_policy or AdmissionPolicy()
        self._verifier_fn = verifier
        self._evidence_verifier = EvidenceVerifier(
            policy=self._admission_policy, verifier=verifier
        )
        self._resource_policy = resource_policy or ProofResourcePolicy()
        self._cancellation = cancellation or CancellationToken()
        self._candidate_lookup = candidate_lookup
        self._prove_lookup = prove_lookup
        self._on_tombstone = on_tombstone
        self._on_admission = on_admission
        self._tombstones: list[tuple[str, str]] = []
        self._admissions: list[CacheAdmissionRecord] = []

    @property
    def tombstones(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._tombstones)

    @property
    def admissions(self) -> tuple[CacheAdmissionRecord, ...]:
        return tuple(self._admissions)

    def execute(
        self,
        plan: IncrementalProofPlan,
        resource_policy: ProofResourcePolicy | None = None,
    ) -> IncrementalProofResult:
        return self.execute_incremental_plan(plan, resource_policy)

    def execute_incremental_plan(
        self,
        plan: IncrementalProofPlan,
        resource_policy: ProofResourcePolicy | None = None,
    ) -> IncrementalProofResult:
        if not isinstance(plan, IncrementalProofPlan):
            raise ExecutorError("plan must be IncrementalProofPlan")
        if not plan.complete:
            raise ExecutorError("incomplete plans cannot be executed")

        policy = resource_policy or self._resource_policy
        if not isinstance(policy, ProofResourcePolicy):
            raise ExecutorError("resource_policy must be ProofResourcePolicy")

        plan_cid = plan.plan_cid()
        required = plan_required_unit_ids(plan)
        prove_ids = plan_prove_unit_ids(plan)
        reuse_ids = _sorted_unique(plan.reusable_unit_ids)
        removed_ids = _sorted_unique(plan.removed_unit_ids)

        records: list[UnitExecutionRecord] = []

        # Cooperative plan-level cancellation before any work.
        if self._cancellation.cancelled:
            for unit_id in required:
                records.append(
                    self._blocking_record(
                        unit_id,
                        UnitDisposition.CANCELLED,
                        ExecutionReasonCode.PLAN_CANCELLED,
                        f"plan cancelled before execution: {self._cancellation.reason}",
                    )
                )
            for unit_id in removed_ids:
                records.append(self._remove_record(unit_id))
            return self._finalize(
                plan_cid=plan_cid,
                required=required,
                records=records,
                message="plan cancelled; aggregation blocked",
            )

        # Removals first — record tombstones, never prove/reuse.
        for unit_id in removed_ids:
            records.append(self._remove_record(unit_id))
            self._record_tombstone(unit_id, "removed")

        # Fresh cache verification for planned reuse units.
        for unit_id in reuse_ids:
            if self._cancellation.cancelled:
                records.append(
                    self._blocking_record(
                        unit_id,
                        UnitDisposition.CANCELLED,
                        ExecutionReasonCode.PLAN_CANCELLED,
                        "cancelled during reuse verification",
                    )
                )
                continue
            records.append(self._execute_reuse(unit_id))

        # Prove invalidated / added / reject-reuse units under resource policy.
        scheduler = ProofWorkScheduler(policy)
        for unit_id in prove_ids:
            if self._cancellation.cancelled:
                records.append(
                    self._blocking_record(
                        unit_id,
                        UnitDisposition.CANCELLED,
                        ExecutionReasonCode.PLAN_CANCELLED,
                        "cancelled during prove",
                    )
                )
                continue
            records.append(self._execute_prove(unit_id, scheduler))

        return self._finalize(
            plan_cid=plan_cid,
            required=required,
            records=records,
            message="incremental plan execution finished",
        )

    def _lookup_candidate(self, unit_id: str) -> CacheCandidatePayload | None:
        if unit_id in self._candidates:
            return self._candidates[unit_id]
        if self._candidate_lookup is not None:
            return self._candidate_lookup(unit_id)
        return None

    def _lookup_prove(self, unit_id: str) -> FreshProvePayload | None:
        if unit_id in self._prove_payloads:
            return self._prove_payloads[unit_id]
        if self._prove_lookup is not None:
            return self._prove_lookup(unit_id)
        return None

    def _execute_reuse(self, unit_id: str) -> UnitExecutionRecord:
        candidate = self._lookup_candidate(unit_id)
        if candidate is None:
            self._record_tombstone(unit_id, "missing_candidate")
            return self._blocking_record(
                unit_id,
                UnitDisposition.REJECTED,
                ExecutionReasonCode.MISSING_CANDIDATE,
                "no cache candidate for planned reuse unit",
            )
        if candidate.unit_id != unit_id:
            self._record_tombstone(unit_id, "cache_key_mismatch")
            return self._blocking_record(
                unit_id,
                UnitDisposition.REJECTED,
                ExecutionReasonCode.CACHE_KEY_MISMATCH,
                (
                    f"candidate unit_id {candidate.unit_id!r} does not match "
                    f"planned reuse unit {unit_id!r}"
                ),
            )

        # Fault markers observed before crypto admission.  Presence of a kit
        # candidate never authorizes reuse; every marker fails closed here.
        if candidate.stale:
            self._record_tombstone(unit_id, "stale")
            return self._blocking_record(
                unit_id,
                UnitDisposition.REJECTED,
                ExecutionReasonCode.STALE_EVIDENCE,
                "stale cache candidate rejected; fresh verification required",
            )
        if candidate.poisoned:
            self._record_tombstone(unit_id, "poisoned")
            return self._blocking_record(
                unit_id,
                UnitDisposition.REJECTED,
                ExecutionReasonCode.POISONED_EVIDENCE,
                "poisoned cache candidate quarantined",
            )
        if candidate.simulated or candidate.proof_mode is ProofMode.SIMULATED:
            self._record_tombstone(unit_id, "simulated")
            return self._blocking_record(
                unit_id,
                UnitDisposition.REJECTED,
                ExecutionReasonCode.SIMULATED_EVIDENCE,
                "simulated evidence cannot be reused under production policy",
            )
        if candidate.corrupt:
            self._record_tombstone(unit_id, "corrupt")
            return self._blocking_record(
                unit_id,
                UnitDisposition.REJECTED,
                ExecutionReasonCode.CORRUPT_EVIDENCE,
                "corrupt candidate bytes rejected",
            )

        observed = candidate.observed_digest()
        if not _constant_time_equal(observed, candidate.expected_digest):
            self._record_tombstone(unit_id, "rehash_failed")
            return self._blocking_record(
                unit_id,
                UnitDisposition.REJECTED,
                ExecutionReasonCode.REHASH_FAILED,
                (
                    f"rehash mismatch: observed {observed!r} != "
                    f"expected {candidate.expected_digest!r}"
                ),
            )

        recomputed_key = candidate.recomputed_cache_key or candidate.cache_key
        if candidate.cache_key_mismatch or not _constant_time_equal(
            recomputed_key, candidate.cache_key
        ):
            self._record_tombstone(unit_id, "cache_key_mismatch")
            return self._blocking_record(
                unit_id,
                UnitDisposition.REJECTED,
                ExecutionReasonCode.CACHE_KEY_MISMATCH,
                "recomputed cache key does not match candidate key",
            )

        observed_pi = candidate.observed_public_input_cid
        if observed_pi is not None and observed_pi != candidate.public_input_cid:
            self._record_tombstone(unit_id, "public_input_mismatch")
            return self._blocking_record(
                unit_id,
                UnitDisposition.REJECTED,
                ExecutionReasonCode.PUBLIC_INPUT_MISMATCH,
                (
                    f"public-input mismatch: expected "
                    f"{candidate.public_input_cid!r}, observed {observed_pi!r}"
                ),
            )

        # No fast path: always run admission verification.
        try:
            decision = self._evidence_verifier.verify_for_admission(
                EvidenceCandidate(
                    evidence=candidate.evidence,
                    proof_system_id=candidate.proof_system_id,
                    public_input_cid=candidate.public_input_cid,
                    proof_unit_id=unit_id,
                    proof_object_cid=candidate.proof_object_cid,
                    required_for_seal=True,
                    proof_mode=candidate.proof_mode,
                    terminal_status=candidate.terminal_status,
                    expected_digest=candidate.expected_digest,
                    observed_digest=observed,
                    observed_public_input_cid=observed_pi,
                    logical_epoch=candidate.logical_epoch,
                    metadata=dict(candidate.metadata),
                )
            )
        except (AdmissionError, ExecutorError, TypeError, ValueError) as exc:
            self._record_tombstone(unit_id, "malformed")
            return self._blocking_record(
                unit_id,
                UnitDisposition.REJECTED,
                ExecutionReasonCode.MALFORMED_INPUT,
                f"reuse verification failed closed: {exc}",
            )

        reverify_digest = _sha256_hex(
            _canonical_json(
                {
                    "unit_id": unit_id,
                    "cache_key": candidate.cache_key,
                    "observed_digest": observed,
                    "artifact_cid": candidate.artifact_cid,
                    "admission_admitted": decision.admitted,
                    "verification_digest": decision.verification_digest,
                }
            )
        )

        if not decision.admitted:
            self._record_tombstone(
                unit_id, decision.reason_code or "admission_rejected"
            )
            return UnitExecutionRecord(
                unit_id=unit_id,
                disposition=UnitDisposition.REJECTED,
                reason_code=ExecutionReasonCode.ADMISSION_REJECTED.value,
                message=(
                    f"fresh cache verification rejected reuse: "
                    f"{decision.reason_code}: {decision.message}"
                ),
                freshly_verified=False,
                ready_for_aggregation=False,
                admission=decision,
                cache_admission_record=None,
                reverify_digest=reverify_digest,
            )

        record = issue_cache_admission_record(decision)
        self._record_admission(record)
        return UnitExecutionRecord(
            unit_id=unit_id,
            disposition=UnitDisposition.REUSED,
            reason_code=ExecutionReasonCode.REUSED_AFTER_FRESH_VERIFICATION.value,
            message="candidate freshly rehashed and admitted for reuse",
            freshly_verified=True,
            ready_for_aggregation=True,
            admission=decision,
            cache_admission_record=record,
            reverify_digest=reverify_digest,
        )

    def _execute_prove(
        self,
        unit_id: str,
        scheduler: ProofWorkScheduler,
    ) -> UnitExecutionRecord:
        payload = self._lookup_prove(unit_id)
        if payload is None:
            return self._blocking_record(
                unit_id,
                UnitDisposition.UNAVAILABLE,
                ExecutionReasonCode.MISSING_PROVE_PAYLOAD,
                "prove payload unavailable for required unit",
            )
        if payload.unit_id != unit_id:
            self._record_tombstone(unit_id, "malformed")
            return self._blocking_record(
                unit_id,
                UnitDisposition.REJECTED,
                ExecutionReasonCode.MALFORMED_INPUT,
                (
                    f"prove payload unit_id {payload.unit_id!r} does not match "
                    f"required unit {unit_id!r}"
                ),
            )

        work = ProofWorkItem(
            work_id=f"prove:{unit_id}",
            unit_id=unit_id,
            work_class=WorkClass.SMALL_INDEPENDENT,
            cpu=payload.cpu,
            memory_mb=payload.memory_mb,
            gpu=payload.gpu,
            simulated_gpu=payload.simulated_gpu,
        )
        verdict = scheduler.admit(work)
        if verdict is AdmissionVerdict.UNAVAILABLE:
            return self._blocking_record(
                unit_id,
                UnitDisposition.UNAVAILABLE,
                ExecutionReasonCode.RESOURCE_UNAVAILABLE,
                "resource policy marked prove work unavailable",
            )
        if verdict is AdmissionVerdict.WAIT:
            # Hermetic executor is single-wave: WAIT with no concurrent release
            # is treated as unavailable rather than spinning.
            return self._blocking_record(
                unit_id,
                UnitDisposition.UNAVAILABLE,
                ExecutionReasonCode.RESOURCE_UNAVAILABLE,
                "resource policy deferred prove work without capacity",
            )

        try:
            # Cooperative cancellation after resource admission still fences
            # late prove material from cache admission and aggregation.
            if self._cancellation.cancelled:
                return self._blocking_record(
                    unit_id,
                    UnitDisposition.CANCELLED,
                    ExecutionReasonCode.PLAN_CANCELLED,
                    "cancelled after resource admission; late output fenced",
                )

            status = payload.prove_status
            if status == "cancelled":
                return self._blocking_record(
                    unit_id,
                    UnitDisposition.CANCELLED,
                    ExecutionReasonCode.PROVE_CANCELLED,
                    "prove cancelled; late output cannot be admitted",
                )
            if status == "unavailable":
                return self._blocking_record(
                    unit_id,
                    UnitDisposition.UNAVAILABLE,
                    ExecutionReasonCode.PROVE_UNAVAILABLE,
                    "prove backend unavailable",
                )
            if status == "timeout":
                return self._blocking_record(
                    unit_id,
                    UnitDisposition.FAILED,
                    ExecutionReasonCode.PROVE_TIMEOUT,
                    "prove timed out; required unit not satisfied",
                )
            if status == "simulated" or payload.proof_mode is ProofMode.SIMULATED:
                self._record_tombstone(unit_id, "simulated")
                return self._blocking_record(
                    unit_id,
                    UnitDisposition.REJECTED,
                    ExecutionReasonCode.SIMULATED_EVIDENCE,
                    "simulated prove outcome cannot be admitted",
                )
            if status in _BLOCKING_PROVE_STATUSES or status != "proved":
                return self._blocking_record(
                    unit_id,
                    UnitDisposition.FAILED,
                    ExecutionReasonCode.PROVE_FAILED,
                    f"prove did not succeed: status={status!r}",
                )

            observed_pi = payload.observed_public_input_cid
            if (
                observed_pi is not None
                and observed_pi != payload.public_input_cid
            ):
                self._record_tombstone(unit_id, "public_input_mismatch")
                return self._blocking_record(
                    unit_id,
                    UnitDisposition.REJECTED,
                    ExecutionReasonCode.PUBLIC_INPUT_MISMATCH,
                    (
                        f"public-input mismatch: expected "
                        f"{payload.public_input_cid!r}, observed {observed_pi!r}"
                    ),
                )

            decision = self._evidence_verifier.verify_for_admission(
                EvidenceCandidate(
                    evidence=payload.evidence,
                    proof_system_id=payload.proof_system_id,
                    public_input_cid=payload.public_input_cid,
                    proof_unit_id=unit_id,
                    proof_object_cid=payload.proof_object_cid,
                    required_for_seal=True,
                    proof_mode=payload.proof_mode,
                    terminal_status=payload.terminal_status,
                    expected_digest=payload.expected_digest,
                    observed_digest=payload.observed_digest,
                    observed_public_input_cid=observed_pi,
                    logical_epoch=payload.logical_epoch,
                    metadata=dict(payload.metadata),
                )
            )

            if not decision.admitted:
                self._record_tombstone(
                    unit_id, decision.reason_code or "admission_rejected"
                )
                return UnitExecutionRecord(
                    unit_id=unit_id,
                    disposition=UnitDisposition.REJECTED,
                    reason_code=ExecutionReasonCode.ADMISSION_REJECTED.value,
                    message=(
                        f"new proof rejected by admission: "
                        f"{decision.reason_code}: {decision.message}"
                    ),
                    freshly_verified=False,
                    ready_for_aggregation=False,
                    admission=decision,
                    cache_admission_record=None,
                )

            record = issue_cache_admission_record(decision)
            self._record_admission(record)
            return UnitExecutionRecord(
                unit_id=unit_id,
                disposition=UnitDisposition.NEWLY_PROVED,
                reason_code=ExecutionReasonCode.NEWLY_PROVED_AND_ADMITTED.value,
                message="unit freshly proved, verified, and admitted",
                freshly_verified=True,
                ready_for_aggregation=True,
                admission=decision,
                cache_admission_record=record,
            )
        except (AdmissionError, ExecutorError, TypeError, ValueError) as exc:
            return self._blocking_record(
                unit_id,
                UnitDisposition.FAILED,
                ExecutionReasonCode.MALFORMED_INPUT,
                f"prove/admission failed closed: {exc}",
            )
        finally:
            scheduler.release(work.work_id)

    def _remove_record(self, unit_id: str) -> UnitExecutionRecord:
        return UnitExecutionRecord(
            unit_id=unit_id,
            disposition=UnitDisposition.REMOVED,
            reason_code=ExecutionReasonCode.REMOVED.value,
            message="unit removed from required manifest",
            freshly_verified=False,
            ready_for_aggregation=True,
            admission=None,
            cache_admission_record=None,
        )

    def _blocking_record(
        self,
        unit_id: str,
        disposition: UnitDisposition,
        reason: ExecutionReasonCode,
        message: str,
    ) -> UnitExecutionRecord:
        return UnitExecutionRecord(
            unit_id=unit_id,
            disposition=disposition,
            reason_code=reason.value,
            message=message,
            freshly_verified=False,
            ready_for_aggregation=False,
            admission=None,
            cache_admission_record=None,
        )

    def _record_tombstone(self, unit_id: str, reason: str) -> None:
        self._tombstones.append((unit_id, reason))
        if self._on_tombstone is not None:
            self._on_tombstone(unit_id, reason)

    def _record_admission(self, record: CacheAdmissionRecord) -> None:
        self._admissions.append(record)
        if self._on_admission is not None:
            self._on_admission(record)

    def _finalize(
        self,
        *,
        plan_cid: str,
        required: tuple[str, ...],
        records: Sequence[UnitExecutionRecord],
        message: str,
    ) -> IncrementalProofResult:
        by_id = {item.unit_id: item for item in records}
        reused = tuple(
            item.unit_id
            for item in records
            if item.disposition is UnitDisposition.REUSED
        )
        newly = tuple(
            item.unit_id
            for item in records
            if item.disposition is UnitDisposition.NEWLY_PROVED
        )
        removed = tuple(
            item.unit_id
            for item in records
            if item.disposition is UnitDisposition.REMOVED
        )
        rejected = tuple(
            item.unit_id
            for item in records
            if item.disposition is UnitDisposition.REJECTED
        )
        cancelled = tuple(
            item.unit_id
            for item in records
            if item.disposition is UnitDisposition.CANCELLED
        )
        unavailable = tuple(
            item.unit_id
            for item in records
            if item.disposition is UnitDisposition.UNAVAILABLE
        )
        failed = tuple(
            item.unit_id
            for item in records
            if item.disposition is UnitDisposition.FAILED
        )

        covered = set(reused) | set(newly)
        required_set = set(required)
        blocking_required = (
            (set(rejected) | set(cancelled) | set(unavailable) | set(failed))
            & required_set
        )
        coverage_complete = covered == required_set and not blocking_required

        # Every required unit must have a ready, verified reuse/prove record.
        for unit_id in required:
            record = by_id.get(unit_id)
            if record is None or not record.ready_for_aggregation:
                coverage_complete = False
                break
            if record.disposition not in {
                UnitDisposition.REUSED,
                UnitDisposition.NEWLY_PROVED,
            }:
                coverage_complete = False
                break
            if not record.freshly_verified or record.cache_admission_record is None:
                coverage_complete = False
                break

        # Fail-closed aggregation gate: exact coverage, no blocking outcomes.
        ready = (
            coverage_complete
            and not cancelled
            and not unavailable
            and not rejected
            and not failed
            and not self._cancellation.cancelled
            and not (set(reused) & set(newly))
        )

        if cancelled:
            status = ExecutionStatus.CANCELLED
            ready = False
            message = "cancelled; cannot proceed to aggregation"
        elif unavailable:
            status = ExecutionStatus.UNAVAILABLE
            ready = False
            message = "unavailable; cannot proceed to aggregation"
        elif rejected:
            status = ExecutionStatus.REJECTED
            ready = False
            message = "evidence rejected; cannot proceed to aggregation"
        elif failed:
            status = ExecutionStatus.FAILED
            ready = False
            message = "prove/verify failed for one or more required units"
        elif ready:
            status = ExecutionStatus.COMPLETED
            message = (
                "reused and newly proved sets exactly cover plan requirements"
            )
        else:
            status = ExecutionStatus.INCOMPLETE
            message = "execution incomplete; aggregation blocked"

        admissions = tuple(
            item.cache_admission_record
            for item in records
            if item.cache_admission_record is not None
        )
        tombstone_ids = _sorted_unique(tuple(uid for uid, _reason in self._tombstones))

        return IncrementalProofResult(
            schema=RESULT_SCHEMA,
            evidence_subset=EVIDENCE_SUBSET,
            status=status,
            plan_cid=plan_cid,
            reused_unit_ids=reused,
            newly_proved_unit_ids=newly,
            removed_unit_ids=removed,
            rejected_unit_ids=rejected,
            cancelled_unit_ids=cancelled,
            unavailable_unit_ids=unavailable,
            failed_unit_ids=failed,
            required_unit_ids=required,
            unit_records=tuple(records),
            admissions=admissions,
            tombstones=tombstone_ids,
            ready_for_aggregation=ready,
            coverage_complete=coverage_complete,
            message=message,
        )


def execute_incremental_plan(
    plan: IncrementalProofPlan,
    resource_policy: ProofResourcePolicy | None = None,
    *,
    candidates: Mapping[str, CacheCandidatePayload] | None = None,
    prove_payloads: Mapping[str, FreshProvePayload] | None = None,
    admission_policy: AdmissionPolicy | None = None,
    verifier: VerifierFn | None = None,
    cancellation: CancellationToken | None = None,
    candidate_lookup: Callable[[str], CacheCandidatePayload | None] | None = None,
    prove_lookup: Callable[[str], FreshProvePayload | None] | None = None,
    on_tombstone: Callable[[str, str], None] | None = None,
    on_admission: Callable[[CacheAdmissionRecord], None] | None = None,
) -> IncrementalProofResult:
    """Execute ``plan`` with fresh cache verification and verified admission.

    Returns an :class:`IncrementalProofResult`.  ``ready_for_aggregation`` is
    True only when reused and newly proved sets exactly cover required units
    and no cancelled/unavailable/rejected/failed required unit exists.
    """

    if not isinstance(plan, IncrementalProofPlan):
        raise ExecutorError("plan must be IncrementalProofPlan")
    executor = IncrementalPlanExecutor(
        candidates=candidates,
        prove_payloads=prove_payloads,
        admission_policy=admission_policy,
        verifier=verifier,
        resource_policy=resource_policy,
        cancellation=cancellation,
        candidate_lookup=candidate_lookup,
        prove_lookup=prove_lookup,
        on_tombstone=on_tombstone,
        on_admission=on_admission,
    )
    return executor.execute_incremental_plan(plan, resource_policy)


__all__ = (
    "CACHE_REVERIFY_EVIDENCE_SUBSET",
    "EVIDENCE_SUBSET",
    "RESULT_SCHEMA",
    "REVERIFY_SCHEMA",
    "UNIT_RECORD_SCHEMA",
    "CacheCandidatePayload",
    "ExecutionReasonCode",
    "ExecutionStatus",
    "ExecutorError",
    "FreshProvePayload",
    "IncrementalPlanExecutor",
    "IncrementalProofResult",
    "UnitDisposition",
    "UnitExecutionRecord",
    "closed_execution_reason_codes",
    "closed_execution_statuses",
    "closed_unit_dispositions",
    "execute_incremental_plan",
    "plan_prove_unit_ids",
    "plan_required_unit_ids",
)
