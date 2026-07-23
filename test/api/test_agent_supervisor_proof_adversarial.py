from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.formal_verification_cache import (
    CacheLookupStatus,
    CacheRejectionReason,
    FormalVerificationCache,
    SingleFlightExecutionError,
    SingleFlightTimeout,
    build_proof_cache_key,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    MAX_REJECTION_REASON_CHARS,
    AssuranceLevel,
    AttemptStatus,
    ContractValidationError,
    EvidenceAuthority,
    EvidenceFreshness,
    EvidenceKind,
    EvidenceVerdict,
    ProofAttempt,
    ProofEvidence,
    ProofReceipt,
    ProofStage,
    ProofVerdict,
    ResourceBudget,
    bounded_rejection_reason,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_policy import (
    ChangedScope,
    FormalVerificationPolicy,
    MergeProofGateReceipt,
    PolicyValidationError,
    ProofOutcome,
    ProofPolicyRule,
    ProofResultStatus,
    RiskLevel,
    RolloutMode,
)
from ipfs_accelerate_py.agent_supervisor.kernel_verification import (
    KernelFailureCode,
    admit_lean_proof_text,
)
from ipfs_accelerate_py.agent_supervisor.proof_attestation import (
    AttestationBackendMode,
    AttestationBackendPolicy,
    PrivateAttestationWitness,
    WitnessDisclosureError,
    public_attestation_artifact,
)


SECRET = "PRIVATE-WITNESS-7f3d2d5e"
TREE = "git-tree:adversarial-current"
OBLIGATION = "obligation:lease-fencing"
PREMISES = ("premise:lease-state", "premise:token-order")
KERNEL = "kernel:lean-4.19"
TOOLCHAIN = "toolchain:lean-lock-2026-07"


def _budget() -> ResourceBudget:
    return ResourceBudget(
        wall_time_ms=10_000,
        cpu_time_ms=8_000,
        memory_bytes=128 * 1024 * 1024,
        disk_bytes=16 * 1024 * 1024,
        max_processes=2,
        max_premises=8,
        max_output_bytes=256_000,
        network_allowed=False,
    )


def _key(**changes: object):
    values: dict[str, object] = {
        "obligation": OBLIGATION,
        "premises": PREMISES,
        "translator": "translator:python-lean@1",
        "solver": "solver:z3@4.13",
        "kernel": KERNEL,
        "toolchain": TOOLCHAIN,
        "theorem_registry": "registry:reviewed@3",
        "policy": "policy:enforcement@7",
        "resource_budget": _budget().to_dict(),
        "candidate_tree": TREE,
    }
    values.update(changes)
    return build_proof_cache_key(**values)  # type: ignore[arg-type]


def _kernel_evidence(
    *,
    obligation_id: str = OBLIGATION,
    kernel_id: str = KERNEL,
    freshness: EvidenceFreshness = EvidenceFreshness.CURRENT,
) -> ProofEvidence:
    return ProofEvidence(
        kind=EvidenceKind.KERNEL_VERIFICATION,
        authority=EvidenceAuthority.KERNEL,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id="artifact:kernel-receipt",
        subject_id=obligation_id,
        verifier_id=kernel_id,
        freshness=freshness,
        independent=True,
        metadata={
            "bindings_verified": True,
            "statement_verified": True,
            "toolchain_id": TOOLCHAIN,
        },
    )


def _solver_evidence() -> ProofEvidence:
    return ProofEvidence(
        kind=EvidenceKind.SOLVER_RESULT,
        authority=EvidenceAuthority.SOLVER,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id="artifact:solver-model",
        subject_id=OBLIGATION,
        verifier_id="solver:z3@4.13",
        independent=True,
    )


def _receipt(
    *,
    evidence: tuple[ProofEvidence, ...] | None = None,
    verdict: ProofVerdict = ProofVerdict.PROVED,
    freshness: EvidenceFreshness = EvidenceFreshness.CURRENT,
    repository_tree_id: str = TREE,
    premise_ids: tuple[str, ...] = PREMISES,
    toolchain_id: str = TOOLCHAIN,
    metadata: dict[str, object] | None = None,
) -> ProofReceipt:
    return ProofReceipt(
        obligation_id=OBLIGATION,
        plan_id="plan:adversarial",
        attempt_id="attempt:adversarial",
        repository_id="repository:complaint-generator",
        repository_tree_id=repository_tree_id,
        ast_scope_ids=("scope:LeaseStore.acquire",),
        premise_ids=premise_ids,
        translator_id="translator:python-lean@1",
        solver_id="solver:z3@4.13",
        kernel_id=KERNEL,
        toolchain_id=toolchain_id,
        theorem_registry_id="registry:reviewed@3",
        policy_id="policy:enforcement@7",
        resource_budget=_budget(),
        verdict=verdict,
        evidence=evidence if evidence is not None else (_kernel_evidence(),),
        provider_id="provider:untrusted",
        provider_claimed_assurance=AssuranceLevel.ATTESTED,
        freshness=freshness,
        kernel_receipt_id="kernel-receipt:adversarial",
        metadata=metadata or {},
    )


def _assert_public_reason(reason: str) -> None:
    assert reason
    assert len(reason) <= MAX_REJECTION_REASON_CHARS
    assert SECRET not in reason
    assert any(
        instruction in reason
        for instruction in (
            "discard",
            "rebuild",
            "recreate",
            "recompute",
            "remove",
            "rerun",
            "retry",
            "run ",
            "use ",
            "inspect",
            "quarantine",
        )
    )


def test_forged_verified_status_and_provider_claims_fail_closed() -> None:
    forged = ProofEvidence(
        kind=EvidenceKind.KERNEL_VERIFICATION,
        authority=EvidenceAuthority.PROVIDER,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id="provider:looks-formal",
        subject_id=OBLIGATION,
        verifier_id=KERNEL,
        independent=True,
    )
    attempt = ProofAttempt(
        plan_id="plan:adversarial",
        step_id="kernel",
        obligation_id=OBLIGATION,
        repository_tree_id=TREE,
        provider_id="provider:untrusted",
        stage=ProofStage.KERNEL_VERIFY,
        status=AttemptStatus.SUCCEEDED,
        evidence=(forged,),
        provider_claimed_assurance=AssuranceLevel.ATTESTED,
    )
    receipt = _receipt(evidence=(forged,))

    assert attempt.status is AttemptStatus.SUCCEEDED
    assert attempt.authoritative_assurance is AssuranceLevel.CANDIDATE
    assert receipt.authoritative_verdict is ProofVerdict.INCONCLUSIVE
    assert not receipt.satisfies_completion()

    payload = receipt.to_dict()
    payload["verified"] = True
    payload["status"] = "verified"
    with pytest.raises(ContractValidationError) as rejected:
        ProofReceipt.from_dict(payload)
    _assert_public_reason(
        bounded_rejection_reason("malformed_proof_receipt", rejected.value)
    )

    payload = receipt.to_dict()
    payload["authoritative_assurance"] = "attested"
    with pytest.raises(ContractValidationError, match="does not match derived"):
        ProofReceipt.from_dict(payload)


def test_solver_only_success_never_becomes_a_proof_or_cache_hit(
    tmp_path: Path,
) -> None:
    receipt = _receipt(evidence=(_solver_evidence(),))
    cache = FormalVerificationCache(tmp_path)

    assert receipt.authoritative_assurance is AssuranceLevel.SOLVER_CHECKED
    assert receipt.authoritative_verdict is ProofVerdict.INCONCLUSIVE
    assert not receipt.satisfies_completion(AssuranceLevel.KERNEL_VERIFIED)

    rejected = cache.put(_key(), receipt)
    assert not rejected.stored
    assert CacheRejectionReason.SOLVER_ONLY_ENTRY.value in rejected.reason_codes
    assert (
        CacheRejectionReason.INSUFFICIENT_ASSURANCE.value
        in rejected.reason_codes
    )
    _assert_public_reason(rejected.actionable_reason)


@pytest.mark.parametrize(
    ("mutation", "receipt_changes"),
    (
        ({"candidate_tree": "git-tree:stale"}, {}),
        ({"premises": ("premise:lease-state", "premise:changed")}, {}),
        ({"toolchain": "toolchain:lean-lock-drifted"}, {}),
        ({}, {"repository_tree_id": "git-tree:stale"}),
        ({}, {"premise_ids": ("premise:changed",)}),
        ({}, {"toolchain_id": "toolchain:lean-lock-drifted"}),
    ),
)
def test_stale_tree_changed_premises_and_toolchain_drift_cannot_reuse_receipts(
    tmp_path: Path,
    mutation: dict[str, object],
    receipt_changes: dict[str, object],
) -> None:
    cache = FormalVerificationCache(tmp_path)
    assert cache.put(_key(), _receipt()).stored

    changed_key = _key(**mutation)
    if mutation:
        lookup = cache.lookup(changed_key)
        assert lookup.status is CacheLookupStatus.MISS
        assert lookup.receipt is None
        _assert_public_reason(lookup.actionable_reason)

    if receipt_changes:
        changed_receipt = _receipt(**receipt_changes)  # type: ignore[arg-type]
        rejected = cache.put(_key(), changed_receipt)
        assert not rejected.stored
        assert (
            CacheRejectionReason.BINDING_MISMATCH.value
            in rejected.reason_codes
        )
        _assert_public_reason(rejected.actionable_reason)


def test_empty_and_unidentifiable_premise_bindings_fail_closed(
    tmp_path: Path,
) -> None:
    cache = FormalVerificationCache(tmp_path)
    nonempty_receipt = _receipt()
    empty_key = _key(premises=())
    opaque_key = _key(premises=({"formula": "changed without an identity"},))

    for key in (empty_key, opaque_key):
        rejected = cache.put(key, nonempty_receipt)
        assert not rejected.stored
        assert (
            CacheRejectionReason.BINDING_MISMATCH.value
            in rejected.reason_codes
        )
        _assert_public_reason(rejected.actionable_reason)


def test_cache_poisoning_and_malformed_receipts_are_quarantined(
    tmp_path: Path,
) -> None:
    cache = FormalVerificationCache(tmp_path)
    assert cache.put(_key(), _receipt()).stored

    connection = sqlite3.connect(cache.db_path)
    try:
        encoded = connection.execute(
            "SELECT entry_json FROM proof_cache_entries WHERE key_id=?",
            (_key().key_id,),
        ).fetchone()[0]
        payload = json.loads(encoded)
        payload["entry_digest"] = "sha256:" + "0" * 64
        connection.execute(
            "UPDATE proof_cache_entries SET entry_json=? WHERE key_id=?",
            (json.dumps(payload), _key().key_id),
        )
        connection.commit()
    finally:
        connection.close()

    poisoned = cache.lookup(_key())
    assert poisoned.status is CacheLookupStatus.REJECTED
    assert poisoned.receipt is None
    assert CacheRejectionReason.POISONED_ENTRY.value in poisoned.reason_codes
    _assert_public_reason(poisoned.actionable_reason)

    malformed = _receipt().to_dict()
    malformed["verified"] = True
    rejected = cache.put(_key(), malformed)
    assert not rejected.stored
    assert rejected.reason_codes == (
        CacheRejectionReason.MALFORMED_ENTRY.value,
    )
    _assert_public_reason(rejected.actionable_reason)


@pytest.mark.parametrize(
    ("proof_text", "failure_code"),
    (
        ("by sorry", KernelFailureCode.INCOMPLETE_PROOF),
        ("by admit", KernelFailureCode.INCOMPLETE_PROOF),
        ("import Mathlib\nby trivial", KernelFailureCode.FORBIDDEN_IMPORT),
        (
            "unsafe def injected : True := by trivial",
            KernelFailureCode.FORBIDDEN_DECLARATION,
        ),
        ("by\n  exact exact_statement", KernelFailureCode.THEOREM_SUBSTITUTION),
        ("by\x00trivial", KernelFailureCode.MALFORMED_RECONSTRUCTION),
    ),
)
def test_incomplete_substituted_and_malicious_prover_output_is_rejected(
    proof_text: str,
    failure_code: KernelFailureCode,
) -> None:
    admission = admit_lean_proof_text(
        proof_text,
        "theorem exact_statement : True := by\n  sorry\n",
        theorem_id=OBLIGATION,
        declaration_name="exact_statement",
        model_artifact_id="model:malicious",
        expected_statement="True",
    )

    assert not admission.accepted
    assert admission.assurance is AssuranceLevel.UNVERIFIED
    assert admission.failure_code is failure_code
    assert len(admission.reason) <= MAX_REJECTION_REASON_CHARS
    assert SECRET not in admission.reason


def test_simulated_zkp_and_stale_verification_keys_cannot_promote_assurance(
    tmp_path: Path,
) -> None:
    simulated = ProofEvidence(
        kind=EvidenceKind.CRYPTOGRAPHIC_ATTESTATION,
        authority=EvidenceAuthority.ATTESTATION_VERIFIER,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id="artifact:simulated-zkp",
        subject_id="kernel-receipt:adversarial",
        verifier_id="zkp:simulator",
        independent=True,
        simulated=True,
    )
    receipt = _receipt(evidence=(_kernel_evidence(), simulated))

    assert receipt.authoritative_assurance is AssuranceLevel.KERNEL_VERIFIED
    assert not receipt.satisfies(AssuranceLevel.ATTESTED)
    rejected = FormalVerificationCache(tmp_path).put(_key(), receipt)
    assert not rejected.stored
    assert (
        CacheRejectionReason.SIMULATED_ATTESTATION.value
        in rejected.reason_codes
    )
    _assert_public_reason(rejected.actionable_reason)

    policy = AttestationBackendPolicy(
        backend_id="zkp:production",
        backend_version="1",
        circuit_id="circuit:receipt",
        circuit_version="2",
        public_input_schema_id="schema:receipt-public",
        public_input_schema_version="1",
        verification_key_id="vk:2026-06",
        verification_key_version="4",
        backend_mode=AttestationBackendMode.CRYPTOGRAPHIC,
        verification_key_expires_at="2026-07-01T00:00:00Z",
    )
    assert policy.key_is_current_at("2026-06-30T23:59:59Z")
    assert not policy.key_is_current_at("2026-07-23T00:00:00Z")
    _assert_public_reason(bounded_rejection_reason("expired_attestation_entry"))


def test_hidden_witnesses_never_enter_contracts_flights_or_storage(
    tmp_path: Path,
) -> None:
    witness = PrivateAttestationWitness({"assignment": SECRET})
    with pytest.raises(WitnessDisclosureError) as disclosure:
        public_attestation_artifact(witness)
    assert SECRET not in str(disclosure.value)

    with pytest.raises(ContractValidationError) as contract_rejection:
        _receipt(metadata={"nested": {"hidden_witness": SECRET}})
    assert SECRET not in str(contract_rejection.value)
    _assert_public_reason(str(contract_rejection.value))

    cache = FormalVerificationCache(tmp_path)
    with pytest.raises(ValueError) as flight_rejection:
        cache.single_flight(
            _key(),
            lambda: {"result": {"private_witness": SECRET}},
        )
    assert SECRET not in str(flight_rejection.value)
    _assert_public_reason(str(flight_rejection.value))
    assert SECRET.encode("utf-8") not in cache.db_path.read_bytes()


def test_forged_policy_outcome_cannot_bypass_enforcement_merge_gate() -> None:
    rule = ProofPolicyRule(
        rule_id="critical-supervisor",
        path_patterns=("src/agent_supervisor/**",),
        minimum_risk=RiskLevel.HIGH,
        invariant_classes=("lease_safety",),
        required_assurance=AssuranceLevel.KERNEL_VERIFIED,
        allow_fallback=False,
    )
    policy = FormalVerificationPolicy(
        name="proof-enforcement",
        version="7",
        rollout_mode=RolloutMode.ENFORCEMENT,
        rules=(rule,),
    )
    selection = policy.select(
        (
            ChangedScope(
                path="src/agent_supervisor/lease.py",
                ast_scope_ids=("class:LeaseStore",),
                risk=RiskLevel.CRITICAL,
                invariant_classes=("lease_safety",),
            ),
        ),
        repository_tree_id=TREE,
    )
    forged = ProofOutcome(
        requirement_id=selection.requirements[0].requirement_id,
        status=ProofResultStatus.PROVED,
        authoritative_assurance=AssuranceLevel.ATTESTED,
        receipt_id="receipt:forged",
    )

    with pytest.raises(PolicyValidationError) as rejected:
        MergeProofGateReceipt.build(
            policy=policy,
            selection=selection,
            repository_tree_id=TREE,
            outcomes=(forged,),
            proof_receipt_ids=("receipt:forged",),
        )
    reason = bounded_rejection_reason("artifact_rejected", rejected.value)
    _assert_public_reason(reason)


def test_single_flight_timeout_remains_fail_closed_and_secret_free(
    tmp_path: Path,
) -> None:
    cache = FormalVerificationCache(tmp_path)
    lease = cache.acquire_lease(_key(), owner_id="stuck-worker", lease_seconds=5)
    assert lease.acquired
    try:
        with pytest.raises(SingleFlightTimeout) as rejected:
            cache.single_flight(
                _key(),
                lambda: {"verified": True},
                wait_timeout_seconds=0.05,  # type: ignore[arg-type]
                poll_interval_seconds=0.005,
            )
        _assert_public_reason(str(rejected.value))
        assert cache.lookup(_key()).status is CacheLookupStatus.MISS
    finally:
        lease.release()


@pytest.mark.parametrize("failure_type", (RuntimeError, KeyboardInterrupt))
def test_single_flight_crash_and_cancellation_are_shared_but_never_promoted(
    tmp_path: Path,
    failure_type: type[BaseException],
) -> None:
    cache = FormalVerificationCache(tmp_path)
    started = threading.Event()
    fail = threading.Event()
    follower_waiting = threading.Event()
    leader_errors: list[BaseException] = []
    follower_errors: list[BaseException] = []
    follower_executed = [False]
    acquire_lease = cache.acquire_lease

    def observed_acquire(*args: object, **kwargs: object):
        lease = acquire_lease(*args, **kwargs)  # type: ignore[arg-type]
        if threading.current_thread().name == "adversarial-follower":
            if not lease.acquired:
                follower_waiting.set()
        return lease

    cache.acquire_lease = observed_acquire  # type: ignore[method-assign]

    def leader_work() -> dict[str, object]:
        started.set()
        assert fail.wait(timeout=2)
        raise failure_type(SECRET)

    def leader() -> None:
        try:
            cache.single_flight(_key(), leader_work, poll_interval_seconds=0.005)
        except BaseException as exc:
            leader_errors.append(exc)

    def follower() -> None:
        def should_not_run() -> dict[str, object]:
            follower_executed[0] = True
            return {"verified": True}

        try:
            cache.single_flight(
                _key(),
                should_not_run,
                poll_interval_seconds=0.005,
            )
        except BaseException as exc:
            follower_errors.append(exc)

    leader_thread = threading.Thread(target=leader)
    leader_thread.start()
    assert started.wait(timeout=2)
    follower_thread = threading.Thread(
        target=follower, name="adversarial-follower"
    )
    follower_thread.start()
    assert follower_waiting.wait(timeout=2)
    fail.set()
    leader_thread.join(timeout=3)
    follower_thread.join(timeout=3)

    assert not leader_thread.is_alive()
    assert not follower_thread.is_alive()
    assert len(leader_errors) == 1
    assert len(follower_errors) == 1
    assert isinstance(follower_errors[0], SingleFlightExecutionError)
    assert not follower_executed[0]
    assert SECRET not in str(follower_errors[0])
    _assert_public_reason(str(follower_errors[0]))
    assert cache.lookup(_key()).receipt is None
    assert SECRET.encode("utf-8") not in cache.db_path.read_bytes()


def test_restart_fences_abandoned_work_and_coordination_never_promotes_verdict(
    tmp_path: Path,
) -> None:
    now = [1_000.0]
    before_crash = FormalVerificationCache(tmp_path, clock=lambda: now[0])
    abandoned = before_crash.acquire_lease(
        _key(), owner_id="crashed-process", lease_seconds=1
    )
    assert abandoned.acquired

    now[0] += 2
    after_restart = FormalVerificationCache(tmp_path, clock=lambda: now[0])
    executions = [0]

    def recompute() -> dict[str, object]:
        executions[0] += 1
        return {"candidate": "not-authoritative"}

    assert after_restart.single_flight(_key(), recompute) == {
        "candidate": "not-authoritative"
    }
    assert executions == [1]
    assert after_restart.lookup(_key()).status is CacheLookupStatus.MISS


def test_duplicate_single_flight_executes_once_without_creating_assurance(
    tmp_path: Path,
) -> None:
    cache = FormalVerificationCache(tmp_path)
    barrier = threading.Barrier(6)
    lock = threading.Lock()
    executions = [0]
    outputs: list[dict[str, object]] = []

    def execute() -> dict[str, object]:
        with lock:
            executions[0] += 1
        time.sleep(0.08)
        return {"candidate": "shared", "authoritative": False}

    def worker() -> None:
        barrier.wait()
        output = cache.single_flight(
            _key(), execute, poll_interval_seconds=0.005
        )
        with lock:
            outputs.append(output)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert all(not thread.is_alive() for thread in threads)
    assert executions == [1]
    assert outputs == [
        {"candidate": "shared", "authoritative": False}
    ] * 6
    assert cache.lookup(_key()).receipt is None


def test_unknown_rejection_detail_is_never_reflected() -> None:
    reason = bounded_rejection_reason(SECRET, detail={"raw_output": SECRET * 100})
    _assert_public_reason(reason)
    assert SECRET not in reason
