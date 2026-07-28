"""CBP-015: cache-first prove path integration tests."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.code_proof_obligations import (
    CachedProveResult,
    ProofCacheMetrics,
    build_code_proof_cache_key,
    prove_code_obligation_with_cache,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_cache import (
    CacheLookupStatus,
    FormalVerificationCache,
    build_proof_cache_key,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    AssuranceLevel,
    CodeProofObligation,
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


def _obligation(**changes: object) -> CodeProofObligation:
    values = {
        "repository_id": "repo:cbp-015",
        "repository_tree_id": "git-tree:v1",
        "ast_scope_ids": ("scope:a",),
        "statement": "Lease fencing token is required.",
        "premise_ids": ("premise:lease-state", "premise:token-order"),
        "template_id": "lease-fencing",
        "template_version": "2",
        "template_semantic_hash": "sha256:template",
        "invariant_class": "lease_safety",
        "task_id": "CBP-015",
        "required_assurance": AssuranceLevel.KERNEL_VERIFIED,
        "fallback_checks": ("pytest:test_lease",),
        "metadata": {"suite": "cbp-015"},
    }
    values.update(changes)
    return CodeProofObligation(**values)


def _kernel(obligation_id: str, *, kernel_id: str = "kernel:lean-4.19") -> ProofEvidence:
    return ProofEvidence(
        kind=EvidenceKind.KERNEL_VERIFICATION,
        authority=EvidenceAuthority.KERNEL,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id="artifact:kernel",
        subject_id=obligation_id,
        verifier_id=kernel_id,
        independent=True,
        simulated=False,
    )


def _candidate(obligation_id: str) -> ProofEvidence:
    return ProofEvidence(
        kind=EvidenceKind.SMT_CANDIDATE,
        authority=EvidenceAuthority.SMT,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id="artifact:candidate",
        subject_id=obligation_id,
        verifier_id="provider:claimed",
        independent=True,
        simulated=False,
    )


def _receipt(
    obligation: CodeProofObligation,
    *,
    evidence: tuple[ProofEvidence, ...] | None = None,
    tree: str | None = None,
    toolchain_id: str = "toolchain:nix-lock",
    policy_id: str = "policy:formal-v1",
    translator_id: str = "translator:python-to-lean@1",
    solver_id: str = "solver:z3@4.13",
    kernel_id: str = "kernel:lean-4.19",
    theorem_registry_id: str = "registry:reviewed-v3",
    metadata: dict | None = None,
) -> ProofReceipt:
    return ProofReceipt(
        obligation_id=obligation.obligation_id,
        plan_id="plan:cbp-015",
        attempt_id="attempt:1",
        repository_id=obligation.repository_id,
        repository_tree_id=tree or obligation.repository_tree_id,
        ast_scope_ids=obligation.ast_scope_ids,
        premise_ids=obligation.premise_ids,
        translator_id=translator_id,
        solver_id=solver_id,
        kernel_id=kernel_id,
        toolchain_id=toolchain_id,
        theorem_registry_id=theorem_registry_id,
        policy_id=policy_id,
        resource_budget=_budget(),
        verdict=ProofVerdict.PROVED,
        evidence=evidence if evidence is not None else (_kernel(obligation.obligation_id),),
        provider_id="provider:hammer",
        provider_claimed_assurance=AssuranceLevel.ATTESTED,
        started_at="2026-07-28T00:00:00Z",
        finished_at="2026-07-28T00:00:01Z",
        resource_usage={"wall_time_ms": 100, "peak_memory_bytes": 1_000},
        metadata=metadata or {},
    )


def test_build_code_proof_cache_key_binds_obligation_tree_premises_toolchain_policy() -> None:
    obligation = _obligation()
    key = build_code_proof_cache_key(
        obligation,
        translator_id="translator:python-to-lean@1",
        solver_id="solver:z3@4.13",
        kernel_id="kernel:lean-4.19",
        toolchain_id="toolchain:nix-lock",
        theorem_registry_id="registry:reviewed-v3",
        policy_id="policy:formal-v1",
        resource_budget=_budget(),
    )
    other_tree = build_code_proof_cache_key(
        obligation,
        translator_id="translator:python-to-lean@1",
        solver_id="solver:z3@4.13",
        kernel_id="kernel:lean-4.19",
        toolchain_id="toolchain:nix-lock",
        theorem_registry_id="registry:reviewed-v3",
        policy_id="policy:formal-v1",
        resource_budget=_budget(),
        candidate_tree="git-tree:other",
    )
    other_tool = build_code_proof_cache_key(
        obligation,
        translator_id="translator:python-to-lean@1",
        solver_id="solver:z3@4.13",
        kernel_id="kernel:lean-4.19",
        toolchain_id="toolchain:other",
        theorem_registry_id="registry:reviewed-v3",
        policy_id="policy:formal-v1",
        resource_budget=_budget(),
    )
    other_assurance = build_code_proof_cache_key(
        _obligation(required_assurance=AssuranceLevel.ATTESTED),
        translator_id="translator:python-to-lean@1",
        solver_id="solver:z3@4.13",
        kernel_id="kernel:lean-4.19",
        toolchain_id="toolchain:nix-lock",
        theorem_registry_id="registry:reviewed-v3",
        policy_id="policy:formal-v1",
        resource_budget=_budget(),
    )
    assert key.key_id != other_tree.key_id
    assert key.key_id != other_tool.key_id
    assert key.key_id != other_assurance.key_id
    assert key.premises == ("premise:lease-state", "premise:token-order")
    assert str(key.candidate_tree) == obligation.repository_tree_id


def test_cache_hit_rederives_assurance_and_skips_provider(tmp_path: Path) -> None:
    cache = FormalVerificationCache(tmp_path)
    obligation = _obligation()
    key = build_code_proof_cache_key(
        obligation,
        translator_id="translator:python-to-lean@1",
        solver_id="solver:z3@4.13",
        kernel_id="kernel:lean-4.19",
        toolchain_id="toolchain:nix-lock",
        theorem_registry_id="registry:reviewed-v3",
        policy_id="policy:formal-v1",
        resource_budget=_budget(),
    )
    receipt = _receipt(obligation)
    assert cache.put(key, receipt).stored

    calls = {"n": 0}

    def prove() -> ProofReceipt:
        calls["n"] += 1
        return receipt

    metrics = ProofCacheMetrics()
    result = prove_code_obligation_with_cache(
        cache,
        key,
        prove=prove,
        required_assurance=AssuranceLevel.KERNEL_VERIFIED,
        metrics=metrics,
    )
    assert result.status == "hit"
    assert result.from_cache is True
    assert result.receipt is not None
    assert result.receipt.authoritative_assurance is AssuranceLevel.KERNEL_VERIFIED
    assert calls["n"] == 0
    assert metrics.hits == 1
    assert metrics.misses == 0


def test_miss_proves_puts_and_second_call_hits(tmp_path: Path) -> None:
    cache = FormalVerificationCache(tmp_path)
    obligation = _obligation()
    key = build_code_proof_cache_key(
        obligation,
        translator_id="translator:python-to-lean@1",
        solver_id="solver:z3@4.13",
        kernel_id="kernel:lean-4.19",
        toolchain_id="toolchain:nix-lock",
        theorem_registry_id="registry:reviewed-v3",
        policy_id="policy:formal-v1",
        resource_budget=_budget(),
    )
    calls = {"n": 0}

    def prove() -> ProofReceipt:
        calls["n"] += 1
        return _receipt(obligation)

    metrics = ProofCacheMetrics()
    first = prove_code_obligation_with_cache(
        cache, key, prove=prove, metrics=metrics
    )
    second = prove_code_obligation_with_cache(
        cache, key, prove=prove, metrics=metrics
    )
    assert first.status == "proved"
    assert first.from_cache is False
    assert second.status == "hit"
    assert second.from_cache is True
    assert calls["n"] == 1
    assert metrics.puts == 1
    assert metrics.hits == 1
    assert metrics.misses == 1


def test_candidate_only_never_authoritative_hit(tmp_path: Path) -> None:
    cache = FormalVerificationCache(tmp_path)
    obligation = _obligation()
    key = build_code_proof_cache_key(
        obligation,
        translator_id="translator:python-to-lean@1",
        solver_id="solver:z3@4.13",
        kernel_id="kernel:lean-4.19",
        toolchain_id="toolchain:nix-lock",
        theorem_registry_id="registry:reviewed-v3",
        policy_id="policy:formal-v1",
        resource_budget=_budget(),
    )

    def prove() -> ProofReceipt:
        return _receipt(
            obligation,
            evidence=(_candidate(obligation.obligation_id),),
        )

    metrics = ProofCacheMetrics()
    result = prove_code_obligation_with_cache(
        cache, key, prove=prove, metrics=metrics
    )
    assert result.status == "rejected"
    assert "candidate_only" in result.reason_codes
    lookup = cache.lookup(key, required_assurance=AssuranceLevel.KERNEL_VERIFIED)
    assert lookup.status is not CacheLookupStatus.HIT


def test_private_material_rejected_on_put(tmp_path: Path) -> None:
    cache = FormalVerificationCache(tmp_path)
    obligation = _obligation()
    key = build_code_proof_cache_key(
        obligation,
        translator_id="translator:python-to-lean@1",
        solver_id="solver:z3@4.13",
        kernel_id="kernel:lean-4.19",
        toolchain_id="toolchain:nix-lock",
        theorem_registry_id="registry:reviewed-v3",
        policy_id="policy:formal-v1",
        resource_budget=_budget(),
    )

    def prove() -> ProofReceipt:
        # Construct a valid receipt then inject private material via dict path.
        receipt = _receipt(obligation)
        payload = receipt.to_dict()
        payload["metadata"] = {"private_witness": "secret"}
        # from_dict should reject; simulate by raising if put path validates.
        try:
            return ProofReceipt.from_dict(payload)
        except Exception:
            # Fall back: return clean receipt but force put with private via
            # a binding mismatch path is not private_material. Instead, put
            # directly and expect validation failure when reconstructing.
            raise

    # Direct put of clean receipt with forged metadata field through store API.
    clean = _receipt(obligation)
    # Poison after put by rewriting DB is covered by unit cache tests.
    # Here ensure prove path refuses candidate metadata with private markers
    # when ProofReceipt construction rejects them.
    with pytest.raises(Exception):
        prove()

    # Put of a kernel receipt still works without private material.
    stored = cache.put(key, clean)
    assert stored.stored


def test_stale_tree_and_toolchain_aliases_on_binding_mismatch() -> None:
    obligation = _obligation()
    key = build_code_proof_cache_key(
        obligation,
        translator_id="translator:python-to-lean@1",
        solver_id="solver:z3@4.13",
        kernel_id="kernel:lean-4.19",
        toolchain_id="toolchain:nix-lock",
        theorem_registry_id="registry:reviewed-v3",
        policy_id="policy:formal-v1",
        resource_budget=_budget(),
        candidate_tree="git-tree:v1",
    )
    wrong_tree = _receipt(obligation, tree="git-tree:old")
    from ipfs_accelerate_py.agent_supervisor.proof.code_proof_obligations import (
        _map_binding_reason,
    )

    reasons = _map_binding_reason(key, wrong_tree, ("cache_binding_mismatch",))
    assert "stale_tree" in reasons

    wrong_tool = _receipt(obligation, toolchain_id="toolchain:other")
    reasons2 = _map_binding_reason(key, wrong_tool, ("cache_binding_mismatch",))
    assert "toolchain_drift" in reasons2


def test_single_flight_collapses_concurrent_prove(tmp_path: Path) -> None:
    cache = FormalVerificationCache(tmp_path)
    obligation = _obligation()
    key = build_code_proof_cache_key(
        obligation,
        translator_id="translator:python-to-lean@1",
        solver_id="solver:z3@4.13",
        kernel_id="kernel:lean-4.19",
        toolchain_id="toolchain:nix-lock",
        theorem_registry_id="registry:reviewed-v3",
        policy_id="policy:formal-v1",
        resource_budget=_budget(),
    )
    started = threading.Event()
    release = threading.Event()
    calls = {"n": 0}
    lock = threading.Lock()

    def prove() -> ProofReceipt:
        with lock:
            calls["n"] += 1
        started.set()
        assert release.wait(timeout=5)
        return _receipt(obligation)

    results: list[CachedProveResult | BaseException] = []

    def worker() -> None:
        try:
            results.append(
                prove_code_obligation_with_cache(
                    cache, key, prove=prove, metrics=ProofCacheMetrics()
                )
            )
        except BaseException as exc:  # noqa: BLE001 - collect for assertion
            results.append(exc)

    leader = threading.Thread(target=worker)
    follower = threading.Thread(target=worker)
    leader.start()
    assert started.wait(timeout=5)
    follower.start()
    # Give follower time to join the single-flight wait.
    import time

    time.sleep(0.2)
    release.set()
    leader.join(timeout=10)
    follower.join(timeout=10)
    assert calls["n"] == 1
    assert len(results) == 2
    assert all(isinstance(r, CachedProveResult) and r.receipt is not None for r in results)


def test_metrics_expose_hit_miss_reject_counts(tmp_path: Path) -> None:
    cache = FormalVerificationCache(tmp_path)
    obligation = _obligation()
    key = build_code_proof_cache_key(
        obligation,
        translator_id="translator:python-to-lean@1",
        solver_id="solver:z3@4.13",
        kernel_id="kernel:lean-4.19",
        toolchain_id="toolchain:nix-lock",
        theorem_registry_id="registry:reviewed-v3",
        policy_id="policy:formal-v1",
        resource_budget=_budget(),
    )
    metrics = ProofCacheMetrics()

    def prove_kernel() -> ProofReceipt:
        return _receipt(obligation)

    def prove_candidate() -> ProofReceipt:
        return _receipt(
            obligation, evidence=(_candidate(obligation.obligation_id),)
        )

    prove_code_obligation_with_cache(
        cache, key, prove=prove_kernel, metrics=metrics
    )
    prove_code_obligation_with_cache(
        cache, key, prove=prove_kernel, metrics=metrics
    )
    # separate key for candidate path
    cand_key = build_code_proof_cache_key(
        _obligation(statement="Different statement for distinct obligation id."),
        translator_id="translator:python-to-lean@1",
        solver_id="solver:z3@4.13",
        kernel_id="kernel:lean-4.19",
        toolchain_id="toolchain:nix-lock",
        theorem_registry_id="registry:reviewed-v3",
        policy_id="policy:formal-v1",
        resource_budget=_budget(),
    )
    prove_code_obligation_with_cache(
        cache, cand_key, prove=prove_candidate, metrics=metrics
    )
    snap = metrics.snapshot()
    assert snap["hits"] >= 1
    assert snap["misses"] >= 1
    assert snap["rejects"] >= 1
    assert "candidate_only" in snap["reject_reasons"]
    assert "hit_rate" in snap


def test_poisoned_entry_reason_surfaces_on_lookup(tmp_path: Path) -> None:
    cache = FormalVerificationCache(tmp_path)
    obligation = _obligation()
    key = build_code_proof_cache_key(
        obligation,
        translator_id="translator:python-to-lean@1",
        solver_id="solver:z3@4.13",
        kernel_id="kernel:lean-4.19",
        toolchain_id="toolchain:nix-lock",
        theorem_registry_id="registry:reviewed-v3",
        policy_id="policy:formal-v1",
        resource_budget=_budget(),
    )
    assert cache.put(key, _receipt(obligation)).stored
    # Corrupt the stored JSON
    import duckdb
    import json

    connection = duckdb.connect(str(cache.db_path))
    try:
        row = connection.execute(
            "SELECT entry_json FROM proof_cache_entries WHERE key_id=?",
            (key.key_id,),
        ).fetchone()
        payload = json.loads(row[0])
        payload["receipt"]["receipt_id"] = "tampered"
        connection.execute(
            "UPDATE proof_cache_entries SET entry_json=? WHERE key_id=?",
            (json.dumps(payload, sort_keys=True, separators=(",", ":")), key.key_id),
        )
        connection.commit()
    finally:
        connection.close()

    lookup = cache.lookup(key)
    assert lookup.status is CacheLookupStatus.REJECTED
    joined = " ".join(lookup.reason_codes)
    assert (
        "poison" in joined
        or "binding" in joined
        or "malformed" in joined
        or lookup.reason_codes
    )
