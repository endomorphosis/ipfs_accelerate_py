from __future__ import annotations

import json
import multiprocessing
import duckdb
import threading
import time
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.formal_verification_cache import (
    CacheLookupStatus,
    CacheRejectionReason,
    DraftCacheKey,
    FormalVerificationCache,
    build_draft_cache_key,
    build_proof_cache_key,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    AssuranceLevel,
    EvidenceAuthority,
    EvidenceFreshness,
    EvidenceKind,
    EvidenceVerdict,
    ProofEvidence,
    ProofReceipt,
    ProofVerdict,
    ResourceBudget,
)


def _budget() -> ResourceBudget:
    return ResourceBudget(
        wall_time_ms=10_000,
        cpu_time_ms=8_000,
        memory_bytes=64 * 1024 * 1024,
        max_processes=2,
        max_premises=4,
        network_allowed=False,
    )


def _key(**changes: object):
    values = {
        "obligation": "obligation-1",
        "premises": ("premise-a", "premise-b"),
        "translator": "translator-1",
        "solver": "solver-1",
        "kernel": "kernel-1",
        "toolchain": "toolchain-1",
        "theorem_registry": "registry-1",
        "policy": "policy-1",
        "resource_budget": _budget().to_dict(),
        "candidate_tree": "tree-1",
    }
    values.update(changes)
    return build_proof_cache_key(**values)


def _draft_key(**changes: object) -> DraftCacheKey:
    values = {
        "goal_digest": "sha256:goal-1",
        "repository_tree_digest": "sha256:tree-1",
        "vocabulary_digest": "sha256:vocabulary-1",
        "compiler_digest": "sha256:compiler-1",
        "model_route_digest": "sha256:route-1",
        "model_version": "leanstral:7b@sha256:model-1",
        "assumptions_digest": "sha256:assumptions-1",
        "bounds_digest": "sha256:bounds-1",
        "policy_digest": "sha256:policy-1",
    }
    values.update(changes)
    return build_draft_cache_key(**values)


def _kernel_evidence(
    *,
    obligation_id: str = "obligation-1",
    kernel_id: str = "kernel-1",
) -> ProofEvidence:
    return ProofEvidence(
        kind=EvidenceKind.KERNEL_VERIFICATION,
        authority=EvidenceAuthority.KERNEL,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id="kernel-artifact-1",
        subject_id=obligation_id,
        verifier_id=kernel_id,
        independent=True,
    )


def _receipt(
    *,
    evidence: tuple[ProofEvidence, ...] | None = None,
    verdict: ProofVerdict = ProofVerdict.PROVED,
    freshness: EvidenceFreshness = EvidenceFreshness.CURRENT,
    metadata: dict[str, object] | None = None,
    kernel_receipt_id: str = "kernel-receipt-1",
) -> ProofReceipt:
    return ProofReceipt(
        obligation_id="obligation-1",
        plan_id="plan-1",
        attempt_id="attempt-1",
        repository_id="repository-1",
        repository_tree_id="tree-1",
        ast_scope_ids=("scope-1",),
        premise_ids=("premise-a", "premise-b"),
        translator_id="translator-1",
        solver_id="solver-1",
        kernel_id="kernel-1",
        toolchain_id="toolchain-1",
        theorem_registry_id="registry-1",
        policy_id="policy-1",
        resource_budget=_budget(),
        verdict=verdict,
        evidence=evidence if evidence is not None else (_kernel_evidence(),),
        freshness=freshness,
        kernel_receipt_id=kernel_receipt_id,
        metadata=metadata or {},
    )


def test_key_binds_every_semantic_execution_and_candidate_input() -> None:
    baseline = _key()
    mutations = {
        "obligation": "obligation-2",
        "premises": ("premise-a", "premise-c"),
        "translator": "translator-2",
        "solver": "solver-2",
        "kernel": "kernel-2",
        "toolchain": "toolchain-2",
        "theorem_registry": "registry-2",
        "policy": "policy-2",
        "resource_budget": ResourceBudget(
            wall_time_ms=9_999,
            cpu_time_ms=8_000,
            memory_bytes=64 * 1024 * 1024,
            max_processes=2,
            max_premises=4,
        ).to_dict(),
        "candidate_tree": "tree-2",
    }

    assert baseline.key_id == _key(premises=("premise-b", "premise-a")).key_id
    for name, value in mutations.items():
        assert _key(**{name: value}).key_id != baseline.key_id, name


def test_draft_key_binds_every_route_aware_reuse_dimension() -> None:
    baseline = _draft_key()
    mutations = {
        "goal_digest": "sha256:goal-2",
        "repository_tree_digest": "sha256:tree-2",
        "vocabulary_digest": "sha256:vocabulary-2",
        "compiler_digest": "sha256:compiler-2",
        "model_route_digest": "sha256:route-2",
        "model_version": "leanstral:7b@sha256:model-2",
        "assumptions_digest": "sha256:assumptions-2",
        "bounds_digest": "sha256:bounds-2",
        "policy_digest": "sha256:policy-2",
    }

    assert DraftCacheKey.from_dict(baseline.to_dict()) == baseline
    for name, value in mutations.items():
        assert _draft_key(**{name: value}).key_id != baseline.key_id, name


def test_untrusted_drafts_are_physically_and_logically_separate_from_receipts(
    tmp_path: Path,
) -> None:
    cache = FormalVerificationCache(tmp_path)
    draft = {
        "draft_id": "draft:1",
        "proposal": {"goals": ["child-a"]},
        "verified": False,
        "authoritative": False,
        "complete": False,
        "trust": "untrusted",
    }

    stored = cache.put_draft(_draft_key(), draft)
    assert stored.stored
    assert stored.entry is not None
    serialized = stored.entry.to_dict()
    assert serialized["trust"] == "untrusted"
    assert serialized["authoritative"] is False
    assert serialized["completion_evidence"] is False

    hit = cache.lookup_draft(_draft_key())
    assert hit.status is CacheLookupStatus.HIT
    assert hit.draft == draft
    assert hit.authoritative is False
    assert hit.completion_evidence is False
    assert cache.lookup(_key()).status is CacheLookupStatus.MISS

    connection = duckdb.connect(str(cache.db_path))
    try:
        assert connection.execute(
            "SELECT COUNT(*) FROM proof_draft_cache_entries"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM proof_cache_entries"
        ).fetchone()[0] == 0
    finally:
        connection.close()

    # Restarting must retain the same integrity-checked, explicitly untrusted
    # projection.
    restarted = FormalVerificationCache(tmp_path)
    assert restarted.get_draft(_draft_key()) == draft


def test_explicit_draft_context_bindings_must_match_the_route_aware_key(
    tmp_path: Path,
) -> None:
    cache = FormalVerificationCache(tmp_path)
    mismatch = cache.put_draft(
        _draft_key(),
        {
            "draft_id": "draft:wrong-route",
            "goal_digest": "sha256:goal-1",
            "repository_tree_digest": "sha256:tree-1",
            "vocabulary_digest": "sha256:vocabulary-1",
            "compiler_digest": "sha256:compiler-1",
            "model_route_digest": "sha256:route-wrong",
            "model_version": "leanstral:7b@sha256:model-1",
            "assumptions_digest": "sha256:assumptions-1",
            "bounds_digest": "sha256:bounds-1",
            "policy_digest": "sha256:policy-1",
        },
    )
    assert not mismatch.stored
    assert (
        mismatch.reason_code
        == CacheRejectionReason.DRAFT_BINDING_MISMATCH.value
    )


@pytest.mark.parametrize(
    "claim",
    [
        {"verified": True},
        {"authoritative": True},
        {"complete": True},
        {"assurance": "kernel_verified"},
        {"nested": {"proof_success": True}},
        {"nested": {"verdict": "proved"}},
    ],
)
def test_draft_cache_rejects_nested_authority_and_completion_claims(
    tmp_path: Path,
    claim: dict[str, object],
) -> None:
    cache = FormalVerificationCache(tmp_path)
    result = cache.put_draft(
        _draft_key(),
        {"draft_id": "draft:unsafe", **claim},
    )
    assert not result.stored
    assert result.reason_codes == (
        CacheRejectionReason.DRAFT_TRUST_CLAIM.value,
    )
    assert cache.lookup_draft(_draft_key()).status is CacheLookupStatus.MISS


def test_draft_ttl_size_bound_and_corruption_fail_closed(tmp_path: Path) -> None:
    now = [1000.0]
    cache = FormalVerificationCache(
        tmp_path,
        max_draft_bytes=100,
        clock=lambda: now[0],
    )
    oversized = cache.put_draft(
        _draft_key(),
        {"draft_id": "draft:large", "text": "x" * 100},
    )
    assert not oversized.stored
    assert oversized.reason_code == CacheRejectionReason.DRAFT_TOO_LARGE.value

    assert cache.put_draft(
        _draft_key(), {"draft_id": "draft:1"}, ttl_seconds=2
    )
    now[0] += 3
    stale = cache.lookup_draft(_draft_key())
    assert stale.status is CacheLookupStatus.REJECTED
    assert stale.reason_code == CacheRejectionReason.DRAFT_STALE.value
    assert cache.purge_expired() == 1
    assert cache.lookup_draft(_draft_key()).status is CacheLookupStatus.MISS

    # Re-store a current row and alter its content without recomputing the
    # entry digest. It must never be returned as a usable model draft.
    assert cache.put_draft(_draft_key(), {"draft_id": "draft:2"})
    connection = duckdb.connect(str(cache.db_path))
    try:
        raw = connection.execute(
            "SELECT entry_json FROM proof_draft_cache_entries"
        ).fetchone()[0]
        payload = json.loads(raw)
        payload["draft"]["draft_id"] = "draft:tampered"
        connection.execute(
            "UPDATE proof_draft_cache_entries SET entry_json=?",
            (json.dumps(payload, sort_keys=True, separators=(",", ":")),),
        )
        connection.commit()
    finally:
        connection.close()
    poisoned = cache.lookup_draft(_draft_key())
    assert poisoned.status is CacheLookupStatus.REJECTED
    assert poisoned.reason_code == CacheRejectionReason.DRAFT_POISONED.value


def test_only_current_receipts_meeting_requested_assurance_are_hits(
    tmp_path: Path,
) -> None:
    now = [1000.0]
    cache = FormalVerificationCache(tmp_path, clock=lambda: now[0])
    stored = cache.put(_key(), _receipt(), ttl_seconds=60)
    assert stored.stored

    hit = cache.lookup(
        _key(), required_assurance=AssuranceLevel.KERNEL_VERIFIED
    )
    assert hit.status is CacheLookupStatus.HIT
    assert hit.receipt is not None
    assert hit.receipt.authoritative_assurance is AssuranceLevel.KERNEL_VERIFIED

    too_weak = cache.lookup(
        _key(), required_assurance=AssuranceLevel.ATTESTED
    )
    assert too_weak.status is CacheLookupStatus.REJECTED
    assert (
        CacheRejectionReason.INSUFFICIENT_ASSURANCE.value
        in too_weak.reason_codes
    )

    now[0] += 61
    stale = cache.lookup(_key())
    assert CacheRejectionReason.STALE_ENTRY.value in stale.reason_codes
    assert (
        CacheRejectionReason.FRESHNESS_NOT_SATISFIED.value
        in stale.reason_codes
    )


def test_maximum_age_is_stric_even_before_entry_ttl(tmp_path: Path) -> None:
    now = [1000.0]
    cache = FormalVerificationCache(tmp_path, clock=lambda: now[0])
    assert cache.put(_key(), _receipt(), ttl_seconds=600)
    now[0] += 11

    lookup = cache.lookup(_key(), max_age_seconds=10)
    assert lookup.status is CacheLookupStatus.REJECTED
    assert CacheRejectionReason.STALE_ENTRY.value in lookup.reason_codes


def test_partial_solver_only_and_simulated_attestation_are_not_stored(
    tmp_path: Path,
) -> None:
    cache = FormalVerificationCache(tmp_path)

    partial = cache.put(
        _key(), _receipt(verdict=ProofVerdict.INCONCLUSIVE)
    )
    assert not partial.stored
    assert CacheRejectionReason.PARTIAL_ENTRY.value in partial.reason_codes

    solver_evidence = ProofEvidence(
        kind=EvidenceKind.SOLVER_RESULT,
        authority=EvidenceAuthority.SOLVER,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id="solver-artifact",
        subject_id="obligation-1",
        verifier_id="solver-1",
        independent=True,
    )
    solver_only = cache.put(_key(), _receipt(evidence=(solver_evidence,)))
    assert not solver_only.stored
    assert (
        CacheRejectionReason.SOLVER_ONLY_ENTRY.value
        in solver_only.reason_codes
    )

    simulated = ProofEvidence(
        kind=EvidenceKind.CRYPTOGRAPHIC_ATTESTATION,
        authority=EvidenceAuthority.ATTESTATION_VERIFIER,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id="attestation-artifact",
        subject_id="kernel-receipt-1",
        verifier_id="attestation-verifier-1",
        independent=True,
        simulated=True,
    )
    simulated_result = cache.put(
        _key(), _receipt(evidence=(_kernel_evidence(), simulated))
    )
    assert not simulated_result.stored
    assert (
        CacheRejectionReason.SIMULATED_ATTESTATION.value
        in simulated_result.reason_codes
    )


def test_malformed_and_binding_poisoned_entries_have_reason_codes(
    tmp_path: Path,
) -> None:
    cache = FormalVerificationCache(tmp_path)
    malformed = cache.put(_key(), {"not": "a receipt"})
    assert not malformed.stored
    assert (
        CacheRejectionReason.MALFORMED_ENTRY.value
        in malformed.reason_codes
    )

    assert cache.put(_key(), _receipt())
    connection = duckdb.connect(str(cache.db_path))
    try:
        raw = connection.execute(
            "SELECT entry_json FROM proof_cache_entries"
        ).fetchone()[0]
        payload = json.loads(raw)
        payload["receipt"]["attempt_id"] = "tampered-attempt"
        connection.execute(
            "UPDATE proof_cache_entries SET entry_json=?",
            (json.dumps(payload, sort_keys=True, separators=(",", ":")),),
        )
        connection.commit()
    finally:
        connection.close()

    poisoned = cache.lookup(_key())
    assert poisoned.status is CacheLookupStatus.REJECTED
    assert (
        CacheRejectionReason.POISONED_ENTRY.value
        in poisoned.reason_codes
    )


def test_mismatched_receipt_binding_is_rejected_before_persistence(
    tmp_path: Path,
) -> None:
    cache = FormalVerificationCache(tmp_path)
    result = cache.put(_key(candidate_tree="other-tree"), _receipt())
    assert not result.stored
    assert (
        CacheRejectionReason.BINDING_MISMATCH.value in result.reason_codes
    )


def test_single_flight_deduplicates_threads(tmp_path: Path) -> None:
    cache = FormalVerificationCache(tmp_path)
    barrier = threading.Barrier(8)
    counter = [0]
    counter_lock = threading.Lock()
    results: list[dict[str, object]] = []

    def execute() -> dict[str, object]:
        with counter_lock:
            counter[0] += 1
        time.sleep(0.15)
        return {"proof": "shared", "ordinal": 1}

    def worker() -> None:
        barrier.wait()
        result = cache.single_flight(_key(), execute)
        with counter_lock:
            results.append(result)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert counter == [1]
    assert results == [{"proof": "shared", "ordinal": 1}] * 8


def test_draft_single_flight_uses_the_route_aware_namespace(
    tmp_path: Path,
) -> None:
    cache = FormalVerificationCache(tmp_path)
    barrier = threading.Barrier(6)
    counter = [0]
    lock = threading.Lock()
    results: list[dict[str, object]] = []

    def execute() -> dict[str, object]:
        with lock:
            counter[0] += 1
        time.sleep(0.1)
        return {"draft_id": "draft:shared", "trust": "untrusted"}

    def worker() -> None:
        barrier.wait()
        result = cache.single_flight_draft(_draft_key(), execute)
        with lock:
            results.append(result)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert counter == [1]
    assert results == [
        {"draft_id": "draft:shared", "trust": "untrusted"}
    ] * 6


def test_draft_single_flight_rejects_authority_before_publishing(
    tmp_path: Path,
) -> None:
    cache = FormalVerificationCache(tmp_path)
    with pytest.raises(PermissionError, match="claims proof authority"):
        cache.single_flight_draft(
            _draft_key(),
            lambda: {"draft_id": "draft:unsafe", "verdict": "proved"},
        )

    connection = duckdb.connect(str(cache.db_path))
    try:
        row = connection.execute(
            "SELECT status, outcome_json FROM proof_flight_outcomes"
        ).fetchone()
    finally:
        connection.close()
    assert row is not None
    assert row[0] == "error"
    assert "draft:unsafe" not in row[1]
    assert "proved" not in row[1]


def test_single_flight_heartbeats_long_running_leader(tmp_path: Path) -> None:
    cache = FormalVerificationCache(tmp_path)
    started = threading.Event()
    counter = [0]
    results: list[dict[str, object]] = []

    def execute() -> dict[str, object]:
        counter[0] += 1
        started.set()
        time.sleep(1.4)
        return {"proof": "heartbeat-shared"}

    leader = threading.Thread(
        target=lambda: results.append(
            cache.single_flight(_key(), execute, lease_seconds=1)
        )
    )
    leader.start()
    assert started.wait(timeout=1)
    # This arrives after the original lease would have expired. The heartbeat
    # must keep it a follower of the same active proof execution.
    time.sleep(1.05)
    follower = threading.Thread(
        target=lambda: results.append(
            cache.single_flight(_key(), execute, lease_seconds=1)
        )
    )
    follower.start()
    leader.join(timeout=5)
    follower.join(timeout=5)

    assert counter == [1]
    assert results == [{"proof": "heartbeat-shared"}] * 2


def _process_flight_worker(
    cache_path: str,
    barrier: multiprocessing.synchronize.Barrier,
    counter: multiprocessing.sharedctypes.Synchronized,
    output: multiprocessing.queues.Queue,
) -> None:
    cache = FormalVerificationCache(cache_path)
    barrier.wait()

    def execute() -> dict[str, object]:
        with counter.get_lock():
            counter.value += 1
        time.sleep(0.2)
        return {"proof": "cross-process"}

    try:
        output.put(cache.single_flight(_key(), execute))
    except BaseException as exc:  # pragma: no cover - assertion reports child error
        output.put({"error": f"{type(exc).__name__}: {exc}"})


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="cross-process lease test requires fork",
)
def test_single_flight_deduplicates_processes(tmp_path: Path) -> None:
    context = multiprocessing.get_context("fork")
    process_count = 4
    barrier = context.Barrier(process_count)
    counter = context.Value("i", 0)
    output = context.Queue()
    processes = [
        context.Process(
            target=_process_flight_worker,
            args=(str(tmp_path), barrier, counter, output),
        )
        for _ in range(process_count)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)

    assert all(process.exitcode == 0 for process in processes)
    assert counter.value == 1
    assert [output.get(timeout=1) for _ in processes] == [
        {"proof": "cross-process"}
    ] * process_count


def test_expired_lease_is_fenced_and_can_be_taken_over(tmp_path: Path) -> None:
    now = [1000.0]
    cache = FormalVerificationCache(tmp_path, clock=lambda: now[0])
    first = cache.acquire_lease(_key(), owner_id="first", lease_seconds=2)
    follower = cache.acquire_lease(_key(), owner_id="second", lease_seconds=2)
    assert first.acquired
    assert not follower.acquired
    assert follower.fencing_token == first.fencing_token

    now[0] += 3
    replacement = cache.acquire_lease(
        _key(), owner_id="second", lease_seconds=2
    )
    assert replacement.acquired
    assert replacement.fencing_token == first.fencing_token + 1
    assert not first.release()
    assert replacement.release()
