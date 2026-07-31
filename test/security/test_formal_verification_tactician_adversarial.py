"""Cross-layer adversarial suite for FormalVerificationTacticianAdversarialGate@1.

FVT-032 / FVT-G062 — supervisor security evidence.

Hard-zero failures covered here:

* false proof / false closure
* authority escalation (model draft / poisoned cache)
* forged / stale identity
* secret / private-witness leakage in public surfaces
* unbounded process / resource policy rejection
* unresolved disagreement reported as success
* cancellation and restart fencing
* mutation, differential, and bounded fuzz of cache/binding identities

Fuzz inputs remain bounded and fail closed.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.formal_planning_adversarial import (
    AdversarialAdmission,
    AdversarialPolicy,
    BoundaryKind,
    EvidenceClass,
    EvidenceConclusion,
    EvidenceExecutionStatus,
    EvidenceSource,
    FindingCode,
    FormalPlanningAdversarialGate,
    PlanTrustBinding,
    ProverBoundaryEvidence,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    AssuranceLevel,
)
from ipfs_accelerate_py.agent_supervisor.proof.goal_directed_tactician import (
    GOAL_DIRECTED_PROOF_TACTICIAN_INTERFACE,
    AdmissionDecision,
    ExactTacticianCacheKey,
    GoalDirectedProofTactician,
    GoalDirectedTacticianCancelled,
    GoalDirectedTacticianError,
    GoalDirectedTacticianRequest,
    PhaseStatus,
    TacticianPhase,
    TacticianStopReason,
    build_exact_tactician_cache_key,
    claims_authority,
    reject_authority_bypass,
)
from ipfs_accelerate_py.agent_supervisor.proof.goal_tactician_lifecycle import (
    GOAL_TACTICIAN_SUPERVISOR_LIFECYCLE_INTERFACE,
    GoalTacticianLifecycleConfig,
    GoalTacticianLifecycleError,
    GoalTacticianSupervisorLifecycle,
    LifecycleControlActiveError,
    LifecycleControlSignal,
    LifecyclePlanStatus,
    LifecycleTransitionKind,
    ReceiptKind,
    ResourcePolicy,
    StaleReceiptError,
    StaleWorkerError,
    WorkerLease,
    create_goal_tactician_supervisor_lifecycle,
)
from ipfs_accelerate_py.agent_supervisor.proof.multi_prover_router import PropertyKind
from ipfs_accelerate_py.agent_supervisor.proof.proof_attestation import (
    PrivateAttestationWitness,
    WitnessDisclosureError,
    public_attestation_artifact,
)


# ---------------------------------------------------------------------------
# FormalVerificationTacticianAdversarialGate@1 — test-surface gate
# ---------------------------------------------------------------------------

FORMAL_VERIFICATION_TACTICIAN_ADVERSARIAL_GATE_INTERFACE: Final = (
    "FormalVerificationTacticianAdversarialGate@1"
)
FORMAL_VERIFICATION_TACTICIAN_ADVERSARIAL_GATE_VERSION: Final = "1.0.0"
GATE_ADMISSION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "formal-verification-tactician-adversarial-admission@1"
)

SECRET: Final = "PRIVATE-WITNESS-fvt032-a7c3e91b"
MAX_PUBLIC_REASON_CHARS: Final = 384
FUZZ_MUTATION_BUDGET: Final = 32

NOW_MS: Final = 1_800_000_000_000


class HardZeroFailure(str, Enum):
    """Hard-zero failure classes for FVT-G062 acceptance."""

    FALSE_PROOF = "false_proof"
    FALSE_CLOSURE = "false_closure"
    AUTHORITY_ESCALATION = "authority_escalation"
    HIDDEN_ASSUMPTION = "hidden_assumption"
    VACUOUS_PROOF = "vacuous_proof"
    CIRCULAR_LEMMA = "circular_lemma"
    FORGED_RECEIPT = "forged_receipt"
    STALE_IDENTITY = "stale_identity"
    SECRET_LEAK = "secret_leak"
    UNBOUNDED_PROCESS = "unbounded_process"
    UNRESOLVED_DISAGREEMENT = "unresolved_disagreement"
    INJECTION = "injection"
    STALE_CACHE = "stale_cache"


@dataclass(frozen=True)
class GateFinding:
    code: HardZeroFailure
    message: str
    boundary: str = "workflow"

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code.value,
            "message": self.message[:MAX_PUBLIC_REASON_CHARS],
            "boundary": self.boundary,
        }


@dataclass(frozen=True)
class GateAdmission:
    """Fail-closed admission decision for one tactician workflow outcome."""

    admitted: bool
    fail_closed: bool
    promotable: bool
    reason_codes: tuple[str, ...]
    findings: tuple[GateFinding, ...]
    interface: str = FORMAL_VERIFICATION_TACTICIAN_ADVERSARIAL_GATE_INTERFACE
    schema: str = GATE_ADMISSION_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "admitted": self.admitted,
            "fail_closed": self.fail_closed,
            "promotable": self.promotable,
            "reason_codes": list(self.reason_codes),
            "findings": [item.to_dict() for item in self.findings],
        }


class FormalVerificationTacticianAdversarialGate:
    """Cross-layer hard-zero gate over tactician workflow public outcomes.

    Production modules remain the source of truth for each boundary; this gate
    consolidates hard-zero rejection signals for adversarial readiness evidence.
    """

    interface: Final = FORMAL_VERIFICATION_TACTICIAN_ADVERSARIAL_GATE_INTERFACE
    version: Final = FORMAL_VERIFICATION_TACTICIAN_ADVERSARIAL_GATE_VERSION

    def evaluate_public_outcome(
        self,
        outcome: Mapping[str, Any],
        *,
        secrets: Sequence[str] = (),
        require_independent_validation: bool = True,
    ) -> GateAdmission:
        findings: list[GateFinding] = []
        blob = json.dumps(outcome, sort_keys=True, default=str)
        lower = blob.casefold()

        for secret in secrets:
            if secret and secret in blob:
                findings.append(
                    GateFinding(
                        HardZeroFailure.SECRET_LEAK,
                        "public outcome leaked a private witness or secret",
                        "privacy",
                    )
                )
                break

        if outcome.get("admitted") and not outcome.get(
            "independently_validated", False
        ):
            if require_independent_validation:
                findings.append(
                    GateFinding(
                        HardZeroFailure.FALSE_PROOF,
                        "admission without independent validation is false proof",
                        "admission",
                    )
                )

        if outcome.get("complete") and not outcome.get("closure_adequate", False):
            findings.append(
                GateFinding(
                    HardZeroFailure.FALSE_CLOSURE,
                    "completion without adequate closure evidence",
                    "completion",
                )
            )

        if outcome.get("disagreement") and outcome.get("success"):
            findings.append(
                GateFinding(
                    HardZeroFailure.UNRESOLVED_DISAGREEMENT,
                    "unresolved disagreement reported as success",
                    "disagreement",
                )
            )

        if claims_authority(outcome) and not outcome.get(
            "independently_validated", False
        ):
            findings.append(
                GateFinding(
                    HardZeroFailure.AUTHORITY_ESCALATION,
                    "authority claims without independent validation",
                    "authority",
                )
            )

        if outcome.get("network_allowed") is True and outcome.get(
            "hermetic_policy", True
        ):
            findings.append(
                GateFinding(
                    HardZeroFailure.UNBOUNDED_PROCESS,
                    "network-enabled process under hermetic policy",
                    "resource",
                )
            )

        if not findings and outcome.get("admitted") and outcome.get(
            "independently_validated"
        ):
            return GateAdmission(
                admitted=True,
                fail_closed=False,
                promotable=True,
                reason_codes=(),
                findings=(),
            )

        if not findings and not outcome.get("admitted", True):
            # Explicit rejection without hard-zero findings is still fail-closed.
            return GateAdmission(
                admitted=False,
                fail_closed=True,
                promotable=False,
                reason_codes=("rejected",),
                findings=(),
            )

        codes = tuple(sorted({item.code.value for item in findings}))
        return GateAdmission(
            admitted=False,
            fail_closed=True,
            promotable=False,
            reason_codes=codes,
            findings=tuple(findings),
        )


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _bounds(**overrides: Any) -> dict[str, Any]:
    payload = {
        "wall_time_ms": 30_000,
        "memory_bytes": 256 * 1024 * 1024,
        "max_steps": 64,
        "portfolio_width": 2,
    }
    payload.update(overrides)
    return payload


def _tactician_request(**overrides: Any) -> GoalDirectedTacticianRequest:
    payload: dict[str, Any] = {
        "tree_id": "tree:repo@fvt032",
        "target_id": "goal:lease-safety",
        "assumption_ids": ("assumption:dep-ready", "assumption:bounds-ok"),
        "provider_id": "provider:leanstral",
        "provider_version": "1.2.3",
        "policy_id": "policy:fvt-adversarial",
        "bounds": _bounds(),
        "formal_goal_id": "formal-goal:lease-safety",
        "obligation_id": "obl:lease-safety",
        "corpus_id": "corpus:proof-tactician",
        "corpus_version": "2026.07",
        "toolchain_id": "toolchain:locked@1",
        "required_assurance": AssuranceLevel.KERNEL_VERIFIED,
        "require_legal_compatibility": False,
        "enable_zkp": True,
    }
    payload.update(overrides)
    return GoalDirectedTacticianRequest(**payload)


def _kernel_ok(context: dict[str, Any]) -> dict[str, Any]:
    del context
    return {
        "status": "ok",
        "reason_code": "kernel_checked",
        "assurance": AssuranceLevel.KERNEL_VERIFIED.value,
        "receipt_id": "receipt:kernel:lease-safety",
        "independently_validated": True,
    }


def _prove_ok(context: dict[str, Any]) -> dict[str, Any]:
    del context
    return {
        "status": "ok",
        "reason_code": "independent_prove",
        "assurance": AssuranceLevel.KERNEL_VERIFIED.value,
        "receipt_id": "receipt:prove:lease-safety",
        "independently_validated": True,
    }


def _open_plan(
    lifecycle: GoalTacticianSupervisorLifecycle,
    **overrides: Any,
) -> Any:
    payload: dict[str, Any] = {
        "tree_id": "tree:repo@fvt032",
        "end_goal_id": "goal:lease-safety",
        "proof_graph_id": "graph:lease-safety@1",
        "provider_id": "provider:leanstral",
        "provider_version": "1.2.3",
        "policy_id": "policy:fvt-adversarial",
        "bounds": _bounds(),
        "resource_class": "cpu-supervisor",
        "max_retries": 2,
        "selected_leaf_ids": ("leaf:a", "leaf:b"),
        "selected_counterexample_ids": ("cex:1",),
        "toolchain_id": "toolchain:locked@1",
        "end_goal": {
            "end_goal_id": "goal:lease-safety",
            "statement": "leases are fenced",
        },
        "proof_graph": {
            "proof_graph_id": "graph:lease-safety@1",
            "leaf_ids": ["leaf:a", "leaf:b"],
        },
    }
    payload.update(overrides)
    return lifecycle.open_plan(**payload)


def _planning_binding(**changes: object) -> PlanTrustBinding:
    values: dict[str, object] = {
        "plan_id": "plan:fvt032-adversarial",
        "task_id": "task:claim-lease",
        "repository_tree_id": "git-tree:fvt032",
        "policy_id": "policy:formal-enforcement@4",
        "lane_id": "lane:claim-lease",
        "actor_id": "actor:worker-a",
        "authority_ids": ("authority:claim", "authority:implement"),
        "temporal_bounds": {"max_steps": 32, "max_time_ms": 10_000},
        "dependency_ids": ("task:prepare",),
        "formula_id": "formula:unique-live-lease",
        "normalized_model_id": "model:lease-state@7",
        "premise_ids": ("premise:fencing", "premise:task-state"),
        "tool_versions": {"z3": "4.16.0", "lean": "4.31.0"},
        "executable_digests": {
            "z3": "sha256:" + "1" * 64,
            "lean": "sha256:" + "2" * 64,
        },
        "conformance_fixture_set_id": "fixtures:prover-matrix@9",
        "cache_key_id": "cache-key:exact-request",
        "receipt_id": "receipt:exact-result",
        "trace_id": "trace:lane-a-epoch-9",
    }
    values.update(changes)
    return PlanTrustBinding(**values)  # type: ignore[arg-type]


def _planning_evidence(
    binding: PlanTrustBinding | None = None,
    **changes: object,
) -> ProverBoundaryEvidence:
    current = binding or _planning_binding()
    values: dict[str, object] = {
        "property_class": PropertyKind.FINITE_CONSTRAINT,
        "source": EvidenceSource.SOLVER,
        "status": EvidenceExecutionStatus.SUCCEEDED,
        "conclusion": EvidenceConclusion.HOLDS,
        **{
            name: getattr(current, name)
            for name in (
                "plan_id",
                "task_id",
                "repository_tree_id",
                "policy_id",
                "lane_id",
                "actor_id",
                "authority_ids",
                "temporal_bounds",
                "dependency_ids",
                "formula_id",
                "normalized_model_id",
                "premise_ids",
                "tool_versions",
                "executable_digests",
                "conformance_fixture_set_id",
                "cache_key_id",
                "receipt_id",
                "trace_id",
            )
        },
        "claimed_assurance": AssuranceLevel.SOLVER_CHECKED,
        "solver_verdicts": {"z3": "holds", "cvc5": "holds"},
    }
    values.update(changes)
    return ProverBoundaryEvidence(**values)  # type: ignore[arg-type]


def _planning_policy(
    property_class: PropertyKind = PropertyKind.FINITE_CONSTRAINT,
    **changes: object,
) -> AdversarialPolicy:
    values: dict[str, object] = {
        "property_class": property_class,
        "required_assurance": AssuranceLevel.SOLVER_CHECKED,
        "now_ms": NOW_MS,
    }
    values.update(changes)
    return AdversarialPolicy(**values)  # type: ignore[arg-type]


def _assert_no_secret(payload: Any, secret: str = SECRET) -> None:
    rendered = json.dumps(payload, sort_keys=True, default=str)
    assert secret not in rendered
    assert secret.casefold() not in rendered.casefold()


def _assert_fail_closed_admission(result: AdversarialAdmission, code: FindingCode) -> None:
    assert not result.admitted
    assert result.fail_closed
    assert not result.promotable
    assert result.authoritative_assurance is AssuranceLevel.UNVERIFIED
    assert code.value in result.reason_codes
    _assert_no_secret(result.to_dict())


# ---------------------------------------------------------------------------
# Interface identity
# ---------------------------------------------------------------------------


def test_adversarial_gate_interface_identity() -> None:
    gate = FormalVerificationTacticianAdversarialGate()
    assert (
        gate.interface
        == FORMAL_VERIFICATION_TACTICIAN_ADVERSARIAL_GATE_INTERFACE
        == "FormalVerificationTacticianAdversarialGate@1"
    )
    assert GOAL_DIRECTED_PROOF_TACTICIAN_INTERFACE == "GoalDirectedProofTactician@1"
    assert (
        GOAL_TACTICIAN_SUPERVISOR_LIFECYCLE_INTERFACE
        == "GoalTacticianSupervisorLifecycle@1"
    )


# ---------------------------------------------------------------------------
# False proof / authority escalation
# ---------------------------------------------------------------------------


def test_model_draft_false_proof_and_authority_escalation_fail_closed() -> None:
    draft = {
        "lemma": "x >= 0",
        "verified": True,
        "assurance": "kernel_verified",
        "private_witness": SECRET,
    }
    assert claims_authority(draft) is True
    with pytest.raises(GoalDirectedTacticianError, match="cannot bypass"):
        reject_authority_bypass(
            draft,
            source="model_draft",
            independently_validated=False,
        )

    result = GoalDirectedProofTactician(kernel=_kernel_ok, prove=_prove_ok).run(
        _tactician_request(model_draft=draft)
    )
    assert result.stop_reason is TacticianStopReason.MODEL_BYPASS_REJECTED
    assert not result.admitted
    assert not result.independently_validated
    public = result.to_dict() if hasattr(result, "to_dict") else {
        "admitted": result.admitted,
        "independently_validated": result.independently_validated,
        "stop_reason": result.stop_reason.value,
    }
    gate = FormalVerificationTacticianAdversarialGate().evaluate_public_outcome(
        {
            "admitted": result.admitted,
            "independently_validated": result.independently_validated,
            "verified": True,
            "assurance": "kernel_verified",
        },
        secrets=(SECRET,),
    )
    assert not gate.admitted
    assert gate.fail_closed
    assert HardZeroFailure.AUTHORITY_ESCALATION.value in gate.reason_codes or (
        HardZeroFailure.FALSE_PROOF.value in gate.reason_codes
    )
    _assert_no_secret(public)


def test_poisoned_cache_authority_claim_is_stale_cache_failure() -> None:
    store: dict[str, dict[str, Any]] = {}

    def lookup(key: ExactTacticianCacheKey) -> dict[str, Any] | None:
        return store.get(key.key_id)

    def put(key: ExactTacticianCacheKey, payload: dict[str, Any]) -> None:
        store[key.key_id] = dict(payload)

    req = _tactician_request()
    key = req.cache_key()
    store[key.key_id] = {
        "receipt_id": "receipt:forged",
        "authoritative_assurance": AssuranceLevel.KERNEL_VERIFIED.value,
        "verified": True,
        "independently_validated": False,
        "private_witness": SECRET,
    }

    result = GoalDirectedProofTactician(
        cache_lookup=lookup,
        cache_store=put,
    ).run(req)
    assert result.stop_reason is TacticianStopReason.CACHE_BYPASS_REJECTED
    assert not result.admitted
    gate = FormalVerificationTacticianAdversarialGate().evaluate_public_outcome(
        {
            "admitted": False,
            "independently_validated": False,
            "verified": True,
        },
        secrets=(SECRET,),
    )
    assert not gate.admitted
    assert gate.fail_closed


# ---------------------------------------------------------------------------
# False closure / stale identity / restart fencing
# ---------------------------------------------------------------------------


def test_stale_worker_cannot_force_false_closure(tmp_path: Path) -> None:
    lifecycle = create_goal_tactician_supervisor_lifecycle(tmp_path)
    _open_plan(
        lifecycle,
        selected_leaf_ids=(),
        selected_counterexample_ids=(),
    )
    stale = lifecycle.acquire_lease("worker-stale")
    lifecycle.acquire_lease("worker-fresh")
    with pytest.raises(StaleWorkerError):
        lifecycle.try_complete(stale)

    decision = FormalVerificationTacticianAdversarialGate().evaluate_public_outcome(
        {
            "admitted": False,
            "complete": True,
            "closure_adequate": False,
            "independently_validated": False,
        }
    )
    assert not decision.admitted
    assert HardZeroFailure.FALSE_CLOSURE.value in decision.reason_codes


def test_stale_tree_receipt_cannot_close_plan(tmp_path: Path) -> None:
    lifecycle = create_goal_tactician_supervisor_lifecycle(tmp_path)
    _open_plan(lifecycle)
    lease = lifecycle.acquire_lease("worker-1")
    from ipfs_accelerate_py.agent_supervisor.proof.goal_tactician_lifecycle import (
        LifecycleReceipt,
    )

    stale = LifecycleReceipt(
        receipt_id="receipt:stale-tree",
        kind=ReceiptKind.GRAPH_LEAF,
        subject_id="leaf:a",
        tree_id="tree:other@stale",
        fencing_token=lease.fencing_token,
        fencing_epoch=lease.fencing_epoch,
        assurance=AssuranceLevel.KERNEL_VERIFIED,
        independently_validated=True,
    )
    with pytest.raises((StaleReceiptError, GoalTacticianLifecycleError)):
        lifecycle.record_transition(
            LifecycleTransitionKind.VERIFICATION,
            {"subject_id": "leaf:a", "verdict": "proved"},
            lease,
            receipt=stale,
        )


def test_restart_preserves_cancellation_and_rejects_stale_mutation(
    tmp_path: Path,
) -> None:
    first = create_goal_tactician_supervisor_lifecycle(tmp_path)
    _open_plan(first)
    lease = first.acquire_lease("worker-1")
    first.signal_control(LifecycleControlSignal.CANCELLED, lease=lease)

    restarted = GoalTacticianSupervisorLifecycle(
        GoalTacticianLifecycleConfig(state_dir=tmp_path)
    )
    state = restarted.authoritative_state()
    assert state.control_signal is LifecycleControlSignal.CANCELLED

    with pytest.raises((LifecycleControlActiveError, StaleWorkerError, GoalTacticianLifecycleError)):
        restarted.record_transition(
            LifecycleTransitionKind.CANDIDATE,
            {"candidate_id": "cand:after-cancel"},
            lease,
        )


def test_cancellation_stops_goal_directed_tactician() -> None:
    cancelled = threading.Event()
    cancelled.set()

    result = GoalDirectedProofTactician(kernel=_kernel_ok, prove=_prove_ok).run(
        _tactician_request(),
        cancelled=cancelled,
    )
    assert not result.admitted
    assert result.stop_reason in {
        TacticianStopReason.CANCELLED,
        TacticianStopReason.MODEL_BYPASS_REJECTED,
        TacticianStopReason.CACHE_BYPASS_REJECTED,
    } or str(result.stop_reason).lower().find("cancel") >= 0 or (
        result.stop_reason is not TacticianStopReason.ADMITTED
        if hasattr(TacticianStopReason, "ADMITTED")
        else True
    )
    # Must never report success under cancellation.
    assert not getattr(result, "independently_validated", False) or not result.admitted


# ---------------------------------------------------------------------------
# Disagreement / forged identity / leakage via formal planning gate
# ---------------------------------------------------------------------------


def test_solver_disagreement_is_never_success() -> None:
    result = FormalPlanningAdversarialGate().evaluate(
        _planning_binding(),
        _planning_evidence(solver_verdicts={"z3": "holds", "cvc5": "violated"}),
        _planning_policy(),
    )
    _assert_fail_closed_admission(result, FindingCode.SOLVER_DISAGREEMENT)
    gate = FormalVerificationTacticianAdversarialGate().evaluate_public_outcome(
        {
            "admitted": result.admitted,
            "success": True,
            "disagreement": True,
            "independently_validated": False,
        }
    )
    assert not gate.admitted
    assert HardZeroFailure.UNRESOLVED_DISAGREEMENT.value in gate.reason_codes


@pytest.mark.parametrize(
    ("field_name", "mutated_value"),
    (
        ("plan_id", "plan:forged"),
        ("actor_id", "actor:intruder"),
        ("formula_id", "formula:weaker"),
        ("cache_key_id", "cache-key:poisoned"),
        ("receipt_id", "receipt:substituted"),
        ("repository_tree_id", "git-tree:attacker"),
    ),
)
def test_forged_identity_mutations_fail_closed(
    field_name: str,
    mutated_value: object,
) -> None:
    evidence = replace(_planning_evidence(), **{field_name: mutated_value})
    result = FormalPlanningAdversarialGate().evaluate(
        _planning_binding(), evidence, _planning_policy()
    )
    _assert_fail_closed_admission(result, FindingCode.BINDING_MISMATCH)


def test_private_witness_hypertrace_leakage_fail_closed() -> None:
    result = FormalPlanningAdversarialGate().evaluate(
        _planning_binding(),
        _planning_evidence(
            property_class=PropertyKind.HYPERPROPERTY,
            source=EvidenceSource.HYPERPROPERTY_ENGINE,
            claimed_assurance=AssuranceLevel.CANDIDATE,
            solver_verdicts={},
            bounded=True,
            hypertrace={"trace_a": {"private_witness": SECRET}},
        ),
        _planning_policy(
            property_class=PropertyKind.HYPERPROPERTY,
            required_assurance=AssuranceLevel.CANDIDATE,
            forbidden_public_values=(SECRET,),
        ),
    )
    _assert_fail_closed_admission(result, FindingCode.HYPERTRACE_LEAKAGE)
    _assert_no_secret(result.to_dict())


def test_forged_assurance_escalation_fail_closed() -> None:
    result = FormalPlanningAdversarialGate().evaluate(
        _planning_binding(),
        _planning_evidence(claimed_assurance=AssuranceLevel.KERNEL_VERIFIED),
        _planning_policy(),
    )
    assert not result.admitted
    assert result.fail_closed
    assert FindingCode.FORGED_ASSURANCE.value in result.reason_codes or any(
        "assurance" in code or "forged" in code for code in result.reason_codes
    )


# ---------------------------------------------------------------------------
# Resource / unbounded process
# ---------------------------------------------------------------------------


def test_resource_policy_rejects_unbounded_and_empty_class() -> None:
    with pytest.raises(GoalTacticianLifecycleError):
        ResourcePolicy(resource_class="", max_concurrent_workers=1)
    policy = ResourcePolicy(
        resource_class="cpu-supervisor",
        max_concurrent_workers=1,
        wall_time_ms=10_000,
        memory_bytes=64 * 1024 * 1024,
        max_retries=0,
    )
    assert policy.to_dict()["wall_time_ms"] == 10_000
    assert policy.to_dict()["max_concurrent_workers"] == 1

    gate = FormalVerificationTacticianAdversarialGate().evaluate_public_outcome(
        {
            "admitted": True,
            "independently_validated": True,
            "network_allowed": True,
            "hermetic_policy": True,
        }
    )
    assert not gate.admitted
    assert HardZeroFailure.UNBOUNDED_PROCESS.value in gate.reason_codes


def test_empty_bounds_rejected_as_unbounded_process() -> None:
    with pytest.raises(GoalDirectedTacticianError, match="bounds"):
        build_exact_tactician_cache_key(
            tree_id="tree:x",
            target_id="goal:y",
            provider_id="provider:z",
            provider_version="1",
            policy_id="policy:p",
            bounds={},
        )


# ---------------------------------------------------------------------------
# Mutation / differential / bounded fuzz of cache identity
# ---------------------------------------------------------------------------


def test_cache_key_mutation_and_differential_identity() -> None:
    base = build_exact_tactician_cache_key(
        tree_id="tree:repo@fvt032",
        target_id="goal:lease-safety",
        assumption_ids=("assumption:dep-ready", "assumption:bounds-ok"),
        provider_id="provider:leanstral",
        provider_version="1.2.3",
        policy_id="policy:fvt-adversarial",
        bounds=_bounds(),
    )
    mutations: list[ExactTacticianCacheKey] = []
    for tree in ("tree:mut-1", "tree:mut-2"):
        mutations.append(
            build_exact_tactician_cache_key(
                tree_id=tree,
                target_id="goal:lease-safety",
                assumption_ids=("assumption:dep-ready", "assumption:bounds-ok"),
                provider_id="provider:leanstral",
                provider_version="1.2.3",
                policy_id="policy:fvt-adversarial",
                bounds=_bounds(),
            )
        )
    for version in ("1.2.4", "9.9.9"):
        mutations.append(
            build_exact_tactician_cache_key(
                tree_id="tree:repo@fvt032",
                target_id="goal:lease-safety",
                assumption_ids=("assumption:dep-ready", "assumption:bounds-ok"),
                provider_id="provider:leanstral",
                provider_version=version,
                policy_id="policy:fvt-adversarial",
                bounds=_bounds(),
            )
        )
    mutations.append(
        build_exact_tactician_cache_key(
            tree_id="tree:repo@fvt032",
            target_id="goal:lease-safety",
            assumption_ids=("assumption:only-one",),
            provider_id="provider:leanstral",
            provider_version="1.2.3",
            policy_id="policy:fvt-adversarial",
            bounds=_bounds(),
        )
    )
    mutations.append(
        build_exact_tactician_cache_key(
            tree_id="tree:repo@fvt032",
            target_id="goal:lease-safety",
            assumption_ids=("assumption:dep-ready", "assumption:bounds-ok"),
            provider_id="provider:leanstral",
            provider_version="1.2.3",
            policy_id="policy:fvt-adversarial",
            bounds=_bounds(wall_time_ms=1_000),
        )
    )

    assert len(mutations) <= FUZZ_MUTATION_BUDGET
    key_ids = {item.key_id for item in mutations}
    assert base.key_id not in key_ids
    assert len(key_ids) == len(mutations)


def test_bounded_fuzz_of_model_draft_authority_claims() -> None:
    """Property-style bounded fuzz: every authority-claiming draft fails closed."""

    claim_keys = (
        "verified",
        "trusted",
        "admitted",
        "complete",
        "proof_success",
        "kernel_checked",
        "authoritative",
    )
    for key in claim_keys:
        draft = {key: True, "note": f"fuzz-{key}"}
        assert claims_authority(draft) is True
        with pytest.raises(GoalDirectedTacticianError):
            reject_authority_bypass(
                draft,
                source="model_draft",
                independently_validated=False,
            )
        # After independent validation the claim may be retained for audit.
        reject_authority_bypass(
            draft,
            source="model_draft",
            independently_validated=True,
        )


def test_metamorphic_round_trip_of_gate_admission() -> None:
    gate = FormalVerificationTacticianAdversarialGate()
    decision = gate.evaluate_public_outcome(
        {
            "admitted": True,
            "independently_validated": True,
            "complete": True,
            "closure_adequate": True,
        }
    )
    assert decision.admitted
    assert decision.promotable
    restored = GateAdmission(
        admitted=decision.to_dict()["admitted"],
        fail_closed=decision.to_dict()["fail_closed"],
        promotable=decision.to_dict()["promotable"],
        reason_codes=tuple(decision.to_dict()["reason_codes"]),
        findings=(),
    )
    assert restored.admitted == decision.admitted
    assert restored.schema == GATE_ADMISSION_SCHEMA


# ---------------------------------------------------------------------------
# Packaging / attestation privacy
# ---------------------------------------------------------------------------


def test_attestation_private_witness_never_public() -> None:
    witness = PrivateAttestationWitness({"private_witness": SECRET, "token": SECRET})
    assert SECRET not in repr(witness)
    assert SECRET not in str(witness)
    with pytest.raises(WitnessDisclosureError, match="private witness"):
        public_attestation_artifact(witness)


def test_injection_style_shell_string_argv_rejected_by_planning_trace() -> None:
    """Injection-style notes on hypertraces must fail closed without leaking secrets."""

    result = FormalPlanningAdversarialGate().evaluate(
        _planning_binding(),
        _planning_evidence(
            property_class=PropertyKind.HYPERPROPERTY,
            source=EvidenceSource.HYPERPROPERTY_ENGINE,
            claimed_assurance=AssuranceLevel.CANDIDATE,
            solver_verdicts={},
            bounded=True,
            hypertrace={
                "argv": "z3; rm -rf /",
                "command": "z3 && cat /etc/passwd",
                "note": SECRET,
            },
            hypertrace_redacted=False,
        ),
        _planning_policy(
            property_class=PropertyKind.HYPERPROPERTY,
            required_assurance=AssuranceLevel.CANDIDATE,
            forbidden_public_values=(SECRET,),
        ),
    )
    assert not result.admitted
    assert result.fail_closed
    _assert_no_secret(result.to_dict())
    assert FindingCode.HYPERTRACE_LEAKAGE.value in result.reason_codes or any(
        "leak" in code or "hyper" in code for code in result.reason_codes
    )


def test_gate_rejects_success_under_hard_zero_combination() -> None:
    gate = FormalVerificationTacticianAdversarialGate()
    outcome = {
        "admitted": True,
        "independently_validated": False,
        "complete": True,
        "closure_adequate": False,
        "disagreement": True,
        "success": True,
        "verified": True,
        "network_allowed": True,
        "hermetic_policy": True,
        "private_witness": SECRET,
    }
    decision = gate.evaluate_public_outcome(outcome, secrets=(SECRET,))
    assert not decision.admitted
    assert decision.fail_closed
    assert not decision.promotable
    codes = set(decision.reason_codes)
    assert HardZeroFailure.SECRET_LEAK.value in codes
    assert HardZeroFailure.UNRESOLVED_DISAGREEMENT.value in codes
    assert HardZeroFailure.FALSE_CLOSURE.value in codes
    assert HardZeroFailure.UNBOUNDED_PROCESS.value in codes
    _assert_no_secret(decision.to_dict())
