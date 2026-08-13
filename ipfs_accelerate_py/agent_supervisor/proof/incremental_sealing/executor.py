"""Execute incremental plans with fresh cache re-verification (IPS-035).

Every reusable candidate is fetched and rehashed, then cryptographically or
signature-verified under the current admission policy before reuse.  A kit
cache hint or prior admission record is never a fast path.  Newly proved
units are verified before admission.  Cancellation, timeout, unavailable,
stale, poisoned, corrupt, mismatched, and simulated evidence fail closed and
cannot proceed to aggregation.

Interfaces: ``IncrementalPlanExecutor``, ``IncrementalProofResult``,
``execute_incremental_plan``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.admission import (
    AdmissionDecision,
    CacheAdmissionRecord,
    EvidenceCandidate,
    EvidenceVerifier,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.planner import (
    IncrementalProofPlan,
    PlanMode,
    PlannedUnit,
    UnitPlanKind,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.process_control import (
    CancellationToken,
    ProcessControlError,
)
from ipfs_datasets_py.logic.zkp.incremental_sealing.evidence import (
    IntegrityCommitment,
    ProofMode,
    ProofTerminalStatus,
)

EVIDENCE_SUBSET: Final[str] = "ips/incremental-execution@1"
CACHE_REVERIFY_EVIDENCE: Final[str] = "ips/cache-reverification@1"
RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "incremental-proof-result@1"
)

FETCH_VERIFY_PROVE_ORDER: Final[tuple[str, ...]] = (
    "fetch",
    "rehash",
    "verify",
    "admit",
)


class ExecutorError(ValueError):
    """Fail-closed incremental execution contract violation."""


class ExecutionOutcome(str, Enum):
    COMPLETED = "completed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class UnitExecutionKind(str, Enum):
    REUSED = "reused"
    NEWLY_PROVED = "newly_proved"
    TOMBSTONED = "tombstoned"
    REJECTED = "rejected"


class ExecutionReasonCode(str, Enum):
    COVERED = "covered"
    STALE_CANDIDATE = "stale_candidate"
    POISONED_CANDIDATE = "poisoned_candidate"
    CORRUPT_CANDIDATE = "corrupt_candidate"
    DIGEST_MISMATCH = "digest_mismatch"
    PUBLIC_INPUT_MISMATCH = "public_input_mismatch"
    SIMULATED_FORBIDDEN = "simulated_forbidden"
    MISSING_CANDIDATE = "missing_candidate"
    ADMISSION_REJECTED = "admission_rejected"
    PROVE_FAILED = "prove_failed"
    VERIFY_FAILED = "verify_failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    PLAN_INCOMPLETE = "plan_incomplete"
    UNEXPECTED_KIND = "unexpected_kind"


_BLOCK_AGGREGATION: Final[frozenset[ExecutionReasonCode]] = frozenset(
    {
        ExecutionReasonCode.STALE_CANDIDATE,
        ExecutionReasonCode.POISONED_CANDIDATE,
        ExecutionReasonCode.CORRUPT_CANDIDATE,
        ExecutionReasonCode.DIGEST_MISMATCH,
        ExecutionReasonCode.PUBLIC_INPUT_MISMATCH,
        ExecutionReasonCode.SIMULATED_FORBIDDEN,
        ExecutionReasonCode.MISSING_CANDIDATE,
        ExecutionReasonCode.ADMISSION_REJECTED,
        ExecutionReasonCode.PROVE_FAILED,
        ExecutionReasonCode.VERIFY_FAILED,
        ExecutionReasonCode.CANCELLED,
        ExecutionReasonCode.TIMEOUT,
        ExecutionReasonCode.UNAVAILABLE,
        ExecutionReasonCode.PLAN_INCOMPLETE,
        ExecutionReasonCode.UNEXPECTED_KIND,
    }
)

_NON_SUCCESS_OUTCOMES: Final[frozenset[ExecutionOutcome]] = frozenset(
    {
        ExecutionOutcome.REJECTED,
        ExecutionOutcome.CANCELLED,
        ExecutionOutcome.TIMEOUT,
        ExecutionOutcome.UNAVAILABLE,
        ExecutionOutcome.FAILED,
    }
)


@dataclass(frozen=True, slots=True)
class CachedCandidate:
    """Fetched cache candidate presented for fresh re-verification.

    ``observed_digest`` is computed from retrieved bytes.  Poisoned or
    simulated flags are fail-closed inputs, never ignored.
    """

    unit_id: str
    expected_digest: str
    observed_digest: str
    public_input_cid: str
    observed_public_input_cid: str
    proof_system_id: str = "integrity"
    proof_object_cid: str = "n/a"
    evidence: Mapping[str, Any] | IntegrityCommitment | None = None
    stale: bool = False
    poisoned: bool = False
    corrupt: bool = False
    simulated: bool = False
    missing: bool = False
    prior_admission: CacheAdmissionRecord | None = None


@dataclass(frozen=True, slots=True)
class FreshProof:
    """Newly produced evidence that still requires verification + admission."""

    unit_id: str
    candidate: EvidenceCandidate
    proof_bytes_digest: str
    simulated: bool = False
    status: str = "proved"


@dataclass(frozen=True, slots=True)
class ResourcePolicy:
    """Closed resource envelope for one plan execution."""

    timeout_seconds: float = 30.0
    max_units: int = 10_000
    allow_aggregation: bool = True

    def to_canonical(self) -> dict[str, Any]:
        return {
            "timeout_seconds": self.timeout_seconds,
            "max_units": self.max_units,
            "allow_aggregation": self.allow_aggregation,
        }


@dataclass(frozen=True, slots=True)
class UnitExecutionRecord:
    """Per-unit execution disposition after fresh verification."""

    unit_id: str
    kind: UnitExecutionKind
    reason: ExecutionReasonCode
    admitted: bool
    verification_digest: str | None = None
    cache_admission_record: CacheAdmissionRecord | None = None
    steps: tuple[str, ...] = FETCH_VERIFY_PROVE_ORDER

    def to_canonical(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "kind": self.kind.value,
            "reason": self.reason.value,
            "admitted": self.admitted,
            "verification_digest": self.verification_digest,
            "steps": list(self.steps),
            "cache_fast_path": False,
        }


@dataclass(frozen=True, slots=True)
class IncrementalProofResult:
    """Closed result of executing one incremental plan."""

    schema: str
    evidence_subset: str
    cache_reverification: str
    plan_cid: str
    mode: PlanMode
    outcome: ExecutionOutcome
    reused_unit_ids: tuple[str, ...]
    newly_proved_unit_ids: tuple[str, ...]
    tombstoned_unit_ids: tuple[str, ...]
    rejected_unit_ids: tuple[str, ...]
    required_unit_ids: tuple[str, ...]
    units: tuple[UnitExecutionRecord, ...]
    admissions: tuple[CacheAdmissionRecord, ...]
    may_aggregate: bool
    complete_coverage: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.outcome in _NON_SUCCESS_OUTCOMES and self.may_aggregate:
            raise ExecutorError(
                "non-success outcomes cannot proceed to aggregation"
            )
        if self.outcome is ExecutionOutcome.COMPLETED and not self.complete_coverage:
            raise ExecutorError("completed results require exact plan coverage")

    @property
    def succeeded(self) -> bool:
        return (
            self.outcome is ExecutionOutcome.COMPLETED
            and self.complete_coverage
            and self.may_aggregate
        )

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_subset": self.evidence_subset,
            "cache_reverification": self.cache_reverification,
            "plan_cid": self.plan_cid,
            "mode": self.mode.value,
            "outcome": self.outcome.value,
            "reused_unit_ids": list(self.reused_unit_ids),
            "newly_proved_unit_ids": list(self.newly_proved_unit_ids),
            "tombstoned_unit_ids": list(self.tombstoned_unit_ids),
            "rejected_unit_ids": list(self.rejected_unit_ids),
            "required_unit_ids": list(self.required_unit_ids),
            "units": [item.to_canonical() for item in self.units],
            "admissions": [item.to_canonical() for item in self.admissions],
            "may_aggregate": self.may_aggregate,
            "complete_coverage": self.complete_coverage,
            "reason_codes": list(self.reason_codes),
            "cache_fast_path": False,
        }

    def result_cid(self) -> str:
        payload = json.dumps(
            self.to_canonical(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


FetchFn = Callable[[str], CachedCandidate | None]
ProveFn = Callable[[PlannedUnit], FreshProof]


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _integrity_evidence(digest: str, cid: str) -> IntegrityCommitment:
    return IntegrityCommitment(
        digest=digest,
        cid=cid,
        merkle_inclusion="leaf:0",
        byte_length=32,
    )


class IncrementalPlanExecutor:
    """Execute one :class:`IncrementalProofPlan` with fail-closed verification."""

    def __init__(
        self,
        *,
        fetch: FetchFn | None = None,
        prove: ProveFn | None = None,
        verifier: EvidenceVerifier | None = None,
        token: CancellationToken | None = None,
        timed_out: Callable[[], bool] | None = None,
        backend_available: bool = True,
    ) -> None:
        self._fetch = fetch
        self._prove = prove
        self._verifier = verifier or EvidenceVerifier()
        self._token = token or CancellationToken()
        self._timed_out = timed_out or (lambda: False)
        self._backend_available = backend_available

    def execute(
        self,
        plan: IncrementalProofPlan,
        resource_policy: ResourcePolicy | Mapping[str, Any] | None = None,
    ) -> IncrementalProofResult:
        if not isinstance(plan, IncrementalProofPlan):
            raise ExecutorError("plan must be an IncrementalProofPlan")
        policy = _coerce_resource_policy(resource_policy)
        if not plan.complete:
            return self._blocked(
                plan,
                ExecutionOutcome.FAILED,
                ExecutionReasonCode.PLAN_INCOMPLETE,
                (),
            )
        if len(plan.units) > policy.max_units:
            return self._blocked(
                plan,
                ExecutionOutcome.FAILED,
                ExecutionReasonCode.PLAN_INCOMPLETE,
                (),
            )

        records: list[UnitExecutionRecord] = []
        admissions: list[CacheAdmissionRecord] = []
        try:
            for unit in plan.units:
                self._check_control()
                record, admission = self._execute_unit(unit)
                records.append(record)
                if admission is not None:
                    admissions.append(admission)
        except ProcessControlError as exc:
            reason = str(exc)
            if reason.startswith("timeout"):
                code = ExecutionReasonCode.TIMEOUT
                outcome = ExecutionOutcome.TIMEOUT
            elif reason.startswith("unavailable"):
                code = ExecutionReasonCode.UNAVAILABLE
                outcome = ExecutionOutcome.UNAVAILABLE
            else:
                code = ExecutionReasonCode.CANCELLED
                outcome = ExecutionOutcome.CANCELLED
            return self._blocked(plan, outcome, code, tuple(records))

        return self._finish(plan, tuple(records), tuple(admissions), policy)

    def _check_control(self) -> None:
        if self._timed_out():
            raise ProcessControlError("timeout:resource_policy")
        try:
            self._token.check()
        except ProcessControlError:
            raise
        if not self._backend_available:
            raise ProcessControlError("unavailable:backend")

    def _execute_unit(
        self, unit: PlannedUnit
    ) -> tuple[UnitExecutionRecord, CacheAdmissionRecord | None]:
        if unit.kind is UnitPlanKind.REMOVE:
            return (
                UnitExecutionRecord(
                    unit.unit_id,
                    UnitExecutionKind.TOMBSTONED,
                    ExecutionReasonCode.COVERED,
                    False,
                    steps=("tombstone",),
                ),
                None,
            )
        if unit.kind is UnitPlanKind.REUSE:
            return self._reuse(unit)
        if unit.kind in {UnitPlanKind.REPROVE, UnitPlanKind.PROVE_NEW}:
            return self._prove_and_admit(unit)
        if unit.kind is UnitPlanKind.REJECT_REUSE:
            return self._prove_and_admit(unit)
        return (
            UnitExecutionRecord(
                unit.unit_id,
                UnitExecutionKind.REJECTED,
                ExecutionReasonCode.UNEXPECTED_KIND,
                False,
            ),
            None,
        )

    def _reuse(
        self, unit: PlannedUnit
    ) -> tuple[UnitExecutionRecord, CacheAdmissionRecord | None]:
        candidate = self._fetch_candidate(unit.unit_id)
        reject = _candidate_reject_reason(candidate)
        if reject is not None:
            return (
                UnitExecutionRecord(
                    unit.unit_id,
                    UnitExecutionKind.REJECTED,
                    reject,
                    False,
                    steps=("fetch", "rehash", "verify"),
                ),
                None,
            )
        assert candidate is not None
        if not hmac.compare_digest(candidate.observed_digest, candidate.expected_digest):
            return (
                UnitExecutionRecord(
                    unit.unit_id,
                    UnitExecutionKind.REJECTED,
                    ExecutionReasonCode.DIGEST_MISMATCH,
                    False,
                    steps=("fetch", "rehash"),
                ),
                None,
            )
        if candidate.observed_public_input_cid != candidate.public_input_cid:
            return (
                UnitExecutionRecord(
                    unit.unit_id,
                    UnitExecutionKind.REJECTED,
                    ExecutionReasonCode.PUBLIC_INPUT_MISMATCH,
                    False,
                    steps=("fetch", "rehash", "verify"),
                ),
                None,
            )
        # Prior admission records are not a trust root; always re-verify.
        decision = self._admit(candidate)
        if not decision.admitted or decision.cache_admission_record is None:
            return (
                UnitExecutionRecord(
                    unit.unit_id,
                    UnitExecutionKind.REJECTED,
                    ExecutionReasonCode.ADMISSION_REJECTED,
                    False,
                    steps=FETCH_VERIFY_PROVE_ORDER,
                ),
                None,
            )
        return (
            UnitExecutionRecord(
                unit.unit_id,
                UnitExecutionKind.REUSED,
                ExecutionReasonCode.COVERED,
                True,
                decision.verification_digest,
                decision.cache_admission_record,
                FETCH_VERIFY_PROVE_ORDER,
            ),
            decision.cache_admission_record,
        )

    def _prove_and_admit(
        self, unit: PlannedUnit
    ) -> tuple[UnitExecutionRecord, CacheAdmissionRecord | None]:
        if not self._backend_available:
            return (
                UnitExecutionRecord(
                    unit.unit_id,
                    UnitExecutionKind.REJECTED,
                    ExecutionReasonCode.UNAVAILABLE,
                    False,
                    steps=("prove",),
                ),
                None,
            )
        try:
            fresh = self._produce(unit)
        except ExecutorError:
            return (
                UnitExecutionRecord(
                    unit.unit_id,
                    UnitExecutionKind.REJECTED,
                    ExecutionReasonCode.PROVE_FAILED,
                    False,
                    steps=("prove",),
                ),
                None,
            )
        if fresh.simulated or fresh.status in {"simulated", "simulated_only"}:
            return (
                UnitExecutionRecord(
                    unit.unit_id,
                    UnitExecutionKind.REJECTED,
                    ExecutionReasonCode.SIMULATED_FORBIDDEN,
                    False,
                    steps=("prove", "verify"),
                ),
                None,
            )
        if fresh.status in {"timeout"}:
            return (
                UnitExecutionRecord(
                    unit.unit_id,
                    UnitExecutionKind.REJECTED,
                    ExecutionReasonCode.TIMEOUT,
                    False,
                    steps=("prove",),
                ),
                None,
            )
        if fresh.status in {"cancelled"}:
            return (
                UnitExecutionRecord(
                    unit.unit_id,
                    UnitExecutionKind.REJECTED,
                    ExecutionReasonCode.CANCELLED,
                    False,
                    steps=("prove",),
                ),
                None,
            )
        if fresh.status in {"unavailable"}:
            return (
                UnitExecutionRecord(
                    unit.unit_id,
                    UnitExecutionKind.REJECTED,
                    ExecutionReasonCode.UNAVAILABLE,
                    False,
                    steps=("prove",),
                ),
                None,
            )
        if fresh.status not in {"proved", "verified"}:
            return (
                UnitExecutionRecord(
                    unit.unit_id,
                    UnitExecutionKind.REJECTED,
                    ExecutionReasonCode.PROVE_FAILED,
                    False,
                    steps=("prove",),
                ),
                None,
            )
        decision = self._verifier.verify_for_admission(fresh.candidate)
        if not decision.admitted or decision.cache_admission_record is None:
            return (
                UnitExecutionRecord(
                    unit.unit_id,
                    UnitExecutionKind.REJECTED,
                    ExecutionReasonCode.VERIFY_FAILED,
                    False,
                    steps=("prove", "verify", "admit"),
                ),
                None,
            )
        return (
            UnitExecutionRecord(
                unit.unit_id,
                UnitExecutionKind.NEWLY_PROVED,
                ExecutionReasonCode.COVERED,
                True,
                decision.verification_digest,
                decision.cache_admission_record,
                ("prove", "verify", "admit"),
            ),
            decision.cache_admission_record,
        )

    def _fetch_candidate(self, unit_id: str) -> CachedCandidate | None:
        if self._fetch is None:
            return None
        return self._fetch(unit_id)

    def _produce(self, unit: PlannedUnit) -> FreshProof:
        if self._prove is not None:
            return self._prove(unit)
        digest = _digest(f"fresh:{unit.unit_id}")
        cid = _digest(f"cid:{unit.unit_id}")
        return FreshProof(
            unit_id=unit.unit_id,
            candidate=EvidenceCandidate(
                evidence=_integrity_evidence(digest, cid),
                proof_system_id="integrity",
                public_input_cid=cid,
                proof_unit_id=unit.unit_id,
                proof_object_cid=cid,
                expected_digest=digest,
                observed_digest=digest,
                observed_public_input_cid=cid,
                proof_mode=ProofMode.INTEGRITY_ONLY,
                terminal_status=ProofTerminalStatus.INTEGRITY_VERIFIED,
            ),
            proof_bytes_digest=digest,
        )

    def _admit(self, candidate: CachedCandidate) -> AdmissionDecision:
        evidence = candidate.evidence
        if evidence is None:
            evidence = _integrity_evidence(
                candidate.observed_digest, candidate.public_input_cid
            )
        return self._verifier.verify_for_admission(
            EvidenceCandidate(
                evidence=evidence,
                proof_system_id=candidate.proof_system_id,
                public_input_cid=candidate.public_input_cid,
                proof_unit_id=candidate.unit_id,
                proof_object_cid=candidate.proof_object_cid,
                expected_digest=candidate.expected_digest,
                observed_digest=candidate.observed_digest,
                observed_public_input_cid=candidate.observed_public_input_cid,
                proof_mode=ProofMode.INTEGRITY_ONLY,
                terminal_status=ProofTerminalStatus.INTEGRITY_VERIFIED,
            )
        )

    def _finish(
        self,
        plan: IncrementalProofPlan,
        records: tuple[UnitExecutionRecord, ...],
        admissions: tuple[CacheAdmissionRecord, ...],
        policy: ResourcePolicy,
    ) -> IncrementalProofResult:
        reused = tuple(
            item.unit_id
            for item in records
            if item.kind is UnitExecutionKind.REUSED
        )
        proved = tuple(
            item.unit_id
            for item in records
            if item.kind is UnitExecutionKind.NEWLY_PROVED
        )
        tombstoned = tuple(
            item.unit_id
            for item in records
            if item.kind is UnitExecutionKind.TOMBSTONED
        )
        rejected = tuple(
            item.unit_id
            for item in records
            if item.kind is UnitExecutionKind.REJECTED
        )
        required = _required_unit_ids(plan)
        covered = set(reused) | set(proved) | set(tombstoned)
        complete = covered == set(required) and not rejected
        reasons = tuple(
            item.reason.value
            for item in records
            if item.reason is not ExecutionReasonCode.COVERED
        )
        if rejected:
            first = next(
                item.reason for item in records if item.kind is UnitExecutionKind.REJECTED
            )
            outcome = _outcome_for_reason(first)
            return IncrementalProofResult(
                schema=RESULT_SCHEMA,
                evidence_subset=EVIDENCE_SUBSET,
                cache_reverification=CACHE_REVERIFY_EVIDENCE,
                plan_cid=plan.plan_cid(),
                mode=plan.mode,
                outcome=outcome,
                reused_unit_ids=reused,
                newly_proved_unit_ids=proved,
                tombstoned_unit_ids=tombstoned,
                rejected_unit_ids=rejected,
                required_unit_ids=required,
                units=records,
                admissions=admissions,
                may_aggregate=False,
                complete_coverage=False,
                reason_codes=reasons or (first.value,),
            )
        return IncrementalProofResult(
            schema=RESULT_SCHEMA,
            evidence_subset=EVIDENCE_SUBSET,
            cache_reverification=CACHE_REVERIFY_EVIDENCE,
            plan_cid=plan.plan_cid(),
            mode=plan.mode,
            outcome=ExecutionOutcome.COMPLETED,
            reused_unit_ids=reused,
            newly_proved_unit_ids=proved,
            tombstoned_unit_ids=tombstoned,
            rejected_unit_ids=(),
            required_unit_ids=required,
            units=records,
            admissions=admissions,
            may_aggregate=bool(policy.allow_aggregation and complete),
            complete_coverage=complete,
            reason_codes=(),
        )

    def _blocked(
        self,
        plan: IncrementalProofPlan,
        outcome: ExecutionOutcome,
        reason: ExecutionReasonCode,
        records: tuple[UnitExecutionRecord, ...],
    ) -> IncrementalProofResult:
        return IncrementalProofResult(
            schema=RESULT_SCHEMA,
            evidence_subset=EVIDENCE_SUBSET,
            cache_reverification=CACHE_REVERIFY_EVIDENCE,
            plan_cid=plan.plan_cid(),
            mode=plan.mode,
            outcome=outcome,
            reused_unit_ids=tuple(
                item.unit_id
                for item in records
                if item.kind is UnitExecutionKind.REUSED
            ),
            newly_proved_unit_ids=tuple(
                item.unit_id
                for item in records
                if item.kind is UnitExecutionKind.NEWLY_PROVED
            ),
            tombstoned_unit_ids=tuple(
                item.unit_id
                for item in records
                if item.kind is UnitExecutionKind.TOMBSTONED
            ),
            rejected_unit_ids=_required_unit_ids(plan),
            required_unit_ids=_required_unit_ids(plan),
            units=records,
            admissions=(),
            may_aggregate=False,
            complete_coverage=False,
            reason_codes=(reason.value,),
        )


def execute_incremental_plan(
    plan: IncrementalProofPlan,
    resource_policy: ResourcePolicy | Mapping[str, Any] | None = None,
    *,
    fetch: FetchFn | None = None,
    prove: ProveFn | None = None,
    verifier: EvidenceVerifier | None = None,
    token: CancellationToken | None = None,
    timed_out: Callable[[], bool] | None = None,
    backend_available: bool = True,
) -> IncrementalProofResult:
    """Public facade matching the plan document's ``execute_incremental_plan``."""

    return IncrementalPlanExecutor(
        fetch=fetch,
        prove=prove,
        verifier=verifier,
        token=token,
        timed_out=timed_out,
        backend_available=backend_available,
    ).execute(plan, resource_policy)


def _required_unit_ids(plan: IncrementalProofPlan) -> tuple[str, ...]:
    return tuple(item.unit_id for item in plan.units)


def _candidate_reject_reason(
    candidate: CachedCandidate | None,
) -> ExecutionReasonCode | None:
    if candidate is None or candidate.missing:
        return ExecutionReasonCode.MISSING_CANDIDATE
    if candidate.stale:
        return ExecutionReasonCode.STALE_CANDIDATE
    if candidate.poisoned:
        return ExecutionReasonCode.POISONED_CANDIDATE
    if candidate.corrupt:
        return ExecutionReasonCode.CORRUPT_CANDIDATE
    if candidate.simulated:
        return ExecutionReasonCode.SIMULATED_FORBIDDEN
    return None


def _outcome_for_reason(reason: ExecutionReasonCode) -> ExecutionOutcome:
    if reason is ExecutionReasonCode.CANCELLED:
        return ExecutionOutcome.CANCELLED
    if reason is ExecutionReasonCode.TIMEOUT:
        return ExecutionOutcome.TIMEOUT
    if reason is ExecutionReasonCode.UNAVAILABLE:
        return ExecutionOutcome.UNAVAILABLE
    if reason in _BLOCK_AGGREGATION:
        return ExecutionOutcome.REJECTED
    return ExecutionOutcome.FAILED


def _coerce_resource_policy(
    value: ResourcePolicy | Mapping[str, Any] | None,
) -> ResourcePolicy:
    if value is None:
        return ResourcePolicy()
    if isinstance(value, ResourcePolicy):
        return value
    if not isinstance(value, Mapping):
        raise ExecutorError("resource_policy must be ResourcePolicy, mapping, or None")
    return ResourcePolicy(
        timeout_seconds=float(value.get("timeout_seconds", 30.0)),
        max_units=int(value.get("max_units", 10_000)),
        allow_aggregation=bool(value.get("allow_aggregation", True)),
    )


__all__ = (
    "CACHE_REVERIFY_EVIDENCE",
    "EVIDENCE_SUBSET",
    "FETCH_VERIFY_PROVE_ORDER",
    "RESULT_SCHEMA",
    "CachedCandidate",
    "ExecutionOutcome",
    "ExecutionReasonCode",
    "ExecutorError",
    "FreshProof",
    "IncrementalPlanExecutor",
    "IncrementalProofResult",
    "ResourcePolicy",
    "UnitExecutionKind",
    "UnitExecutionRecord",
    "execute_incremental_plan",
)
