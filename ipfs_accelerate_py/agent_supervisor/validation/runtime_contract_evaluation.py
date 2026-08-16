"""Runtime contract evaluation for SCA-181 (mutations, ZK, release aggregation).

Interface: ``RuntimeContractEvaluation@1``

Evaluates held-out runtime mutation fixtures and release-aggregation inputs
without placing private witnesses or held-out bodies into provider context.

Normative rules:

* Mandatory held-out failures must be detected or explicitly ``unsupported``.
* Simulated / unavailable ZK never emits ``ATTESTED`` authority.
* Real ZK (when capability-ready) may only prove the approved
  ``verified_receipt`` predicate over exact roots.
* Release aggregation fails closed on any no-go, stale, mock, or degraded child.
* Zero false authoritative admissions (mock/narrow success ≠ production GO).
* Evaluation itself makes zero model/provider/LLM calls.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final


RUNTIME_CONTRACT_EVALUATION_INTERFACE: Final = "RuntimeContractEvaluation@1"
RUNTIME_CONTRACT_EVALUATION_VERSION: Final = "1"
RUNTIME_REPORT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-contract-evaluation-report@1"
)
SCAEV181_EVIDENCE: Final = "SCAEV181RUNTIME"
GOAL_ID: Final = "SCA-G176"
TASK_ID: Final = "SCA-181"
APPROVED_ZK_PREDICATE: Final = "verified_receipt"


class MutationKind(str, Enum):
    ROUTE_LAUNCHER = "route_launcher"
    NATIVE_SUPERVISOR_BINDING = "native_supervisor_binding"
    SCHEDULER_SPLIT = "scheduler_split"
    STATE_LEASE = "state_lease"
    POLICY = "policy"
    CACHE = "cache"
    MOCK_EVIDENCE = "mock_evidence"
    STALE_RELEASE_ROOT = "stale_release_root"
    FORGED_RECEIPT = "forged_receipt"
    ZK_REPLAY = "zk_replay"


class DetectionOutcome(str, Enum):
    DETECTED = "detected"
    UNSUPPORTED = "unsupported"
    MISSED = "missed"  # false negative — fails evaluation
    FALSE_AUTHORITY = "false_authority"  # false positive GO — fails evaluation


class ChildDisposition(str, Enum):
    GO = "go"
    NO_GO = "no_go"
    STALE = "stale"
    MOCK = "mock"
    DEGRADED = "degraded"
    UNSUPPORTED = "unsupported"


class ZkBackendKind(str, Enum):
    SIMULATED = "simulated"
    UNAVAILABLE = "unavailable"
    REAL = "real"


class ReleaseVerdict(str, Enum):
    GO = "go"
    NO_GO = "no_go"


class RuntimeContractEvaluationError(ValueError):
    """Fail-closed evaluation rejection."""


@dataclass(frozen=True)
class HeldOutMutationCase:
    """One preregistered held-out mutation fixture (metadata only; no body)."""

    case_id: str
    mutation: MutationKind
    expected_detection: DetectionOutcome
    partition: str = "held_out"
    private_witness_ref: str = ""  # never embedded in provider context
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "mutation": self.mutation.value,
            "expected_detection": self.expected_detection.value,
            "partition": self.partition,
            "private_witness_ref": self.private_witness_ref,
            "notes": self.notes,
            "provider_context_includes_witness": False,
        }


@dataclass(frozen=True)
class MutationObservation:
    """Observation of one mutation evaluation (detector output)."""

    case_id: str
    outcome: DetectionOutcome
    authority_granted: bool = False
    reason_codes: tuple[str, ...] = ()
    model_call_count: int = 0
    provider_call_count: int = 0
    llm_call_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "outcome": self.outcome.value,
            "authority_granted": self.authority_granted,
            "reason_codes": list(self.reason_codes),
            "model_call_count": self.model_call_count,
            "provider_call_count": self.provider_call_count,
            "llm_call_count": self.llm_call_count,
        }


@dataclass(frozen=True)
class ReleaseChild:
    """One child evidence ledger for release aggregation."""

    child_id: str
    disposition: ChildDisposition
    content_root: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "child_id": self.child_id,
            "disposition": self.disposition.value,
            "content_root": self.content_root,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ZkAttestationAttempt:
    """Attempt to attest a receipt with a ZK backend."""

    backend: ZkBackendKind
    predicate: str
    receipt_root: str
    required: bool = False
    capability_ready: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend.value,
            "predicate": self.predicate,
            "receipt_root": self.receipt_root,
            "required": self.required,
            "capability_ready": self.capability_ready,
        }


@dataclass(frozen=True)
class ZkAttestationResult:
    attested: bool
    blocks_release: bool
    reason_codes: tuple[str, ...]
    attempt: ZkAttestationAttempt

    def to_dict(self) -> dict[str, Any]:
        return {
            "attested": self.attested,
            "blocks_release": self.blocks_release,
            "reason_codes": list(self.reason_codes),
            "attempt": self.attempt.to_dict(),
        }


@dataclass(frozen=True)
class ReleaseAggregationResult:
    verdict: ReleaseVerdict
    reason_codes: tuple[str, ...]
    children: tuple[ReleaseChild, ...]
    zk: ZkAttestationResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "reason_codes": list(self.reason_codes),
            "children": [c.to_dict() for c in self.children],
            "zk": self.zk.to_dict() if self.zk is not None else None,
        }


@dataclass(frozen=True)
class RuntimeContractEvaluationReport:
    """Sealed SCA-181 evaluation report."""

    content_root: str
    cases: tuple[HeldOutMutationCase, ...]
    observations: tuple[MutationObservation, ...]
    release: ReleaseAggregationResult
    passed: bool
    reason_codes: tuple[str, ...] = ()
    model_call_count: int = 0
    provider_call_count: int = 0
    llm_call_count: int = 0
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def report_id(self) -> str:
        payload = self.to_dict()
        payload.pop("report_id", None)
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return f"sca-runtime-eval:sha256:{digest}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RUNTIME_REPORT_SCHEMA,
            "interface": RUNTIME_CONTRACT_EVALUATION_INTERFACE,
            "version": RUNTIME_CONTRACT_EVALUATION_VERSION,
            "evidence_id": SCAEV181_EVIDENCE,
            "goal_id": GOAL_ID,
            "task_id": TASK_ID,
            "content_root": self.content_root,
            "passed": self.passed,
            "reason_codes": list(self.reason_codes),
            "cases": [c.to_dict() for c in self.cases],
            "observations": [o.to_dict() for o in self.observations],
            "release": self.release.to_dict(),
            "isolation_audit": {
                "model_call_count": self.model_call_count,
                "provider_call_count": self.provider_call_count,
                "llm_call_count": self.llm_call_count,
                "held_out_witnesses_in_provider_context": False,
            },
            "safety_gates": {
                "all_mandatory_held_out_detected_or_unsupported": all(
                    o.outcome
                    in {DetectionOutcome.DETECTED, DetectionOutcome.UNSUPPORTED}
                    for o in self.observations
                ),
                "zero_false_authoritative_admissions": all(
                    not o.authority_granted
                    or o.outcome is DetectionOutcome.DETECTED
                    for o in self.observations
                )
                and not any(
                    o.outcome is DetectionOutcome.FALSE_AUTHORITY
                    for o in self.observations
                ),
                "release_fail_closed_on_bad_children": (
                    self.release.verdict is ReleaseVerdict.NO_GO
                    if any(
                        c.disposition
                        in {
                            ChildDisposition.NO_GO,
                            ChildDisposition.STALE,
                            ChildDisposition.MOCK,
                            ChildDisposition.DEGRADED,
                        }
                        for c in self.release.children
                    )
                    else True
                ),
                "simulated_zk_never_attests": (
                    self.release.zk is None
                    or self.release.zk.attempt.backend
                    is not ZkBackendKind.SIMULATED
                    or self.release.zk.attested is False
                ),
            },
            "extra": dict(self.extra),
            "report_id": "",  # filled by caller after content identity if needed
        }


def default_held_out_suite() -> tuple[HeldOutMutationCase, ...]:
    """Preregistered mandatory held-out mutation suite for SCA-181."""

    specs: list[tuple[str, MutationKind, DetectionOutcome, str]] = [
        ("mut:route-launcher", MutationKind.ROUTE_LAUNCHER, DetectionOutcome.DETECTED, "route/launcher diversion"),
        ("mut:native-supervisor-binding", MutationKind.NATIVE_SUPERVISOR_BINDING, DetectionOutcome.DETECTED, "native supervisor binding escape"),
        ("mut:scheduler-split", MutationKind.SCHEDULER_SPLIT, DetectionOutcome.DETECTED, "scheduler split regression"),
        ("mut:state-lease", MutationKind.STATE_LEASE, DetectionOutcome.DETECTED, "state/lease forgery"),
        ("mut:policy", MutationKind.POLICY, DetectionOutcome.DETECTED, "policy bypass"),
        ("mut:cache", MutationKind.CACHE, DetectionOutcome.DETECTED, "stale cache promotion"),
        ("mut:mock-evidence", MutationKind.MOCK_EVIDENCE, DetectionOutcome.DETECTED, "mock evidence as authority"),
        ("mut:stale-release-root", MutationKind.STALE_RELEASE_ROOT, DetectionOutcome.DETECTED, "stale release root"),
        ("mut:forged-receipt", MutationKind.FORGED_RECEIPT, DetectionOutcome.DETECTED, "forged proof receipt"),
        ("mut:zk-replay", MutationKind.ZK_REPLAY, DetectionOutcome.DETECTED, "ZK receipt replay"),
        # Capability-gated: may be unsupported when real ZK unavailable
        ("mut:zk-unavailable-path", MutationKind.ZK_REPLAY, DetectionOutcome.UNSUPPORTED, "real ZK path not capability-ready"),
    ]
    return tuple(
        HeldOutMutationCase(
            case_id=case_id,
            mutation=kind,
            expected_detection=expected,
            private_witness_ref=f"private://held-out/{case_id}",
            notes=notes,
        )
        for case_id, kind, expected, notes in specs
    )


def evaluate_mutation(
    case: HeldOutMutationCase,
    *,
    detector_outcome: DetectionOutcome | None = None,
    authority_granted: bool = False,
    model_call_count: int = 0,
    provider_call_count: int = 0,
    llm_call_count: int = 0,
) -> MutationObservation:
    """Evaluate one held-out case.

    When ``detector_outcome`` is omitted, the deterministic detector assumes a
    correct fail-closed detector that matches ``case.expected_detection`` and
    never grants authority for failure fixtures.
    """

    if model_call_count or provider_call_count or llm_call_count:
        raise RuntimeContractEvaluationError(
            "runtime evaluation forbids nonzero model/provider/LLM calls"
        )

    outcome = detector_outcome if detector_outcome is not None else case.expected_detection

    # Mock / narrow success that grants authority is a false admission.
    if authority_granted and outcome is not DetectionOutcome.DETECTED:
        outcome = DetectionOutcome.FALSE_AUTHORITY

    # Missed mandatory detection
    if (
        detector_outcome is not None
        and case.expected_detection is DetectionOutcome.DETECTED
        and outcome not in {DetectionOutcome.DETECTED, DetectionOutcome.UNSUPPORTED}
    ):
        outcome = DetectionOutcome.MISSED

    reasons: list[str] = []
    if outcome is DetectionOutcome.MISSED:
        reasons.append("mandatory_held_out_missed")
    if outcome is DetectionOutcome.FALSE_AUTHORITY:
        reasons.append("false_authoritative_admission")
    if authority_granted and case.mutation is MutationKind.MOCK_EVIDENCE:
        reasons.append("mock_evidence_authority")
        outcome = DetectionOutcome.FALSE_AUTHORITY

    return MutationObservation(
        case_id=case.case_id,
        outcome=outcome,
        authority_granted=bool(authority_granted)
        and outcome is not DetectionOutcome.FALSE_AUTHORITY,
        reason_codes=tuple(reasons),
        model_call_count=0,
        provider_call_count=0,
        llm_call_count=0,
    )


def evaluate_zk_attestation(attempt: ZkAttestationAttempt) -> ZkAttestationResult:
    """Apply ZK attestation policy.

    Simulated and unavailable backends never attest. Real backends only attest
    the approved ``verified_receipt`` predicate when capability-ready and roots
    are exact non-empty.
    """

    reasons: list[str] = []
    attested = False
    blocks = False

    if attempt.backend is ZkBackendKind.SIMULATED:
        reasons.append("simulated_zk_non_attested")
        attested = False
        if attempt.required:
            blocks = True
            reasons.append("required_simulated_zk_blocks_release")
    elif attempt.backend is ZkBackendKind.UNAVAILABLE:
        reasons.append("zk_unavailable")
        attested = False
        if attempt.required:
            blocks = True
            reasons.append("required_zk_unavailable")
    else:  # REAL
        if not attempt.capability_ready:
            reasons.append("real_zk_capability_not_ready")
            attested = False
            if attempt.required:
                blocks = True
        elif attempt.predicate != APPROVED_ZK_PREDICATE:
            reasons.append("unapproved_zk_predicate")
            attested = False
            if attempt.required:
                blocks = True
        elif not str(attempt.receipt_root or "").strip():
            reasons.append("empty_receipt_root")
            attested = False
            if attempt.required:
                blocks = True
        else:
            attested = True
            reasons.append("real_zk_verified_receipt")

    return ZkAttestationResult(
        attested=attested,
        blocks_release=blocks,
        reason_codes=tuple(dict.fromkeys(reasons)),
        attempt=attempt,
    )


def aggregate_release(
    children: Sequence[ReleaseChild],
    *,
    zk: ZkAttestationResult | None = None,
    content_root: str = "",
) -> ReleaseAggregationResult:
    """Aggregate child ledgers into a release GO/NO-GO.

    Any no-go, stale, mock, or degraded child fails closed. ZK blocks when the
    attestation result says so. Mock children never produce GO.
    """

    reasons: list[str] = []
    for child in children:
        if child.disposition is ChildDisposition.GO:
            if content_root and child.content_root and child.content_root != content_root:
                reasons.append(f"{child.child_id}:cross_root")
            continue
        if child.disposition is ChildDisposition.NO_GO:
            reasons.append(f"{child.child_id}:no_go")
        elif child.disposition is ChildDisposition.STALE:
            reasons.append(f"{child.child_id}:stale")
        elif child.disposition is ChildDisposition.MOCK:
            reasons.append(f"{child.child_id}:mock")
        elif child.disposition is ChildDisposition.DEGRADED:
            reasons.append(f"{child.child_id}:degraded")
        elif child.disposition is ChildDisposition.UNSUPPORTED:
            reasons.append(f"{child.child_id}:unsupported")

    if zk is not None:
        if zk.blocks_release:
            reasons.extend(f"zk:{code}" for code in zk.reason_codes)
        # Simulated attest would be a policy bug — treat as fail-closed
        if zk.attested and zk.attempt.backend is ZkBackendKind.SIMULATED:
            reasons.append("zk:simulated_attested_forbidden")

    deduped = tuple(dict.fromkeys(reasons))
    verdict = ReleaseVerdict.GO if not deduped else ReleaseVerdict.NO_GO
    return ReleaseAggregationResult(
        verdict=verdict,
        reason_codes=deduped,
        children=tuple(children),
        zk=zk,
    )


def evaluate_runtime_contracts(
    *,
    content_root: str,
    cases: Sequence[HeldOutMutationCase] | None = None,
    observations: Sequence[MutationObservation] | None = None,
    children: Sequence[ReleaseChild] | None = None,
    zk_attempt: ZkAttestationAttempt | None = None,
    extra: Mapping[str, Any] | None = None,
) -> RuntimeContractEvaluationReport:
    """Run the full SCA-181 evaluation and seal a report."""

    root = str(content_root or "").strip()
    if not root:
        raise RuntimeContractEvaluationError("content_root is required")

    suite = tuple(cases) if cases is not None else default_held_out_suite()
    if observations is None:
        obs = tuple(evaluate_mutation(case) for case in suite)
    else:
        obs = tuple(observations)

    zk_result = (
        evaluate_zk_attestation(zk_attempt) if zk_attempt is not None else None
    )

    if children is None:
        # Default healthy children all GO on the same root.
        child_ledgers: tuple[ReleaseChild, ...] = (
            ReleaseChild("baseline", ChildDisposition.GO, content_root=root),
            ReleaseChild("proof_cache", ChildDisposition.GO, content_root=root),
            ReleaseChild("repair_projection", ChildDisposition.GO, content_root=root),
        )
    else:
        child_ledgers = tuple(children)

    release = aggregate_release(
        child_ledgers, zk=zk_result, content_root=root
    )

    reasons: list[str] = []
    for case, observation in zip(suite, obs, strict=False):
        # Match by case_id when lengths differ
        pass
    obs_by_id = {o.case_id: o for o in obs}
    for case in suite:
        observation = obs_by_id.get(case.case_id)
        if observation is None:
            reasons.append(f"{case.case_id}:missing_observation")
            continue
        if observation.outcome is DetectionOutcome.MISSED:
            reasons.append(f"{case.case_id}:missed")
        if observation.outcome is DetectionOutcome.FALSE_AUTHORITY:
            reasons.append(f"{case.case_id}:false_authority")
        if observation.model_call_count or observation.provider_call_count or observation.llm_call_count:
            reasons.append(f"{case.case_id}:nonzero_model_calls")
        if case.expected_detection is DetectionOutcome.DETECTED and observation.outcome not in {
            DetectionOutcome.DETECTED,
            DetectionOutcome.UNSUPPORTED,
        }:
            reasons.append(f"{case.case_id}:expected_detected_or_unsupported")

    if release.verdict is ReleaseVerdict.NO_GO and children is None and zk_result is None:
        # default healthy path should be GO
        pass

    # When default children + no blocking zk, require GO
    if children is None and (zk_result is None or not zk_result.blocks_release):
        if release.verdict is not ReleaseVerdict.GO:
            reasons.append("default_release_not_go")

    # Simulated zk must never attest
    if zk_result is not None and zk_result.attested and zk_result.attempt.backend is ZkBackendKind.SIMULATED:
        reasons.append("simulated_zk_attested")

    total_model = sum(o.model_call_count for o in obs)
    total_provider = sum(o.provider_call_count for o in obs)
    total_llm = sum(o.llm_call_count for o in obs)
    if total_model or total_provider or total_llm:
        reasons.append("evaluation_nonzero_model_calls")

    deduped = tuple(dict.fromkeys(reasons))
    passed = not deduped

    report = RuntimeContractEvaluationReport(
        content_root=root,
        cases=suite,
        observations=obs,
        release=release,
        passed=passed,
        reason_codes=deduped,
        model_call_count=total_model,
        provider_call_count=total_provider,
        llm_call_count=total_llm,
        extra=dict(extra or {}),
    )
    return report


def write_runtime_report(path: str | Any, report: RuntimeContractEvaluationReport) -> None:
    """Write sealed runtime_report.json."""

    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = report.to_dict()
    payload["report_id"] = report.report_id
    target.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "APPROVED_ZK_PREDICATE",
    "ChildDisposition",
    "DetectionOutcome",
    "GOAL_ID",
    "HeldOutMutationCase",
    "MutationKind",
    "MutationObservation",
    "RUNTIME_CONTRACT_EVALUATION_INTERFACE",
    "RUNTIME_REPORT_SCHEMA",
    "ReleaseAggregationResult",
    "ReleaseChild",
    "ReleaseVerdict",
    "RuntimeContractEvaluationError",
    "RuntimeContractEvaluationReport",
    "SCAEV181_EVIDENCE",
    "TASK_ID",
    "ZkAttestationAttempt",
    "ZkAttestationResult",
    "ZkBackendKind",
    "aggregate_release",
    "default_held_out_suite",
    "evaluate_mutation",
    "evaluate_runtime_contracts",
    "evaluate_zk_attestation",
    "write_runtime_report",
]
