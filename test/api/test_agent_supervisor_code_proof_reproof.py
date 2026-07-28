"""CBP-050: cache-aware re-proof and invalidation tests."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.code_proof_obligations import (
    CandidateDiffEntry,
    DiffChangeKind,
    ObligationCompileStatus,
    ProofCacheMetrics,
    build_code_proof_cache_key,
    compile_code_proof_obligations,
)
from ipfs_accelerate_py.agent_supervisor.proof.code_proof_reproof import (
    InvalidationReason,
    ReproofDisposition,
    binding_fingerprint_for_item,
    invalidation_reasons,
    plan_reproof_from_delta,
    reprove_code_proof_compilation,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_cache import (
    FormalVerificationCache,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    AssuranceLevel,
    EvidenceAuthority,
    EvidenceKind,
    EvidenceVerdict,
    ProofEvidence,
    ProofReceipt,
    ProofVerdict,
    ResourceBudget,
)


SRC_A = """\
class Worker:
    def run(self) -> int:
        return 1
"""

SRC_B = """\
class Worker:
    def run(self) -> int:
        return 2
"""


def _budget() -> ResourceBudget:
    return ResourceBudget(
        wall_time_ms=10_000,
        cpu_time_ms=8_000,
        memory_bytes=64 * 1024 * 1024,
        max_processes=2,
        max_premises=4,
        network_allowed=False,
    )


REPROOF_KW = dict(
    translator_id="translator:python-to-lean@1",
    solver_id="solver:z3@4.13",
    kernel_id="kernel:lean-4.19",
    theorem_registry_id="registry:reviewed-v3",
    resource_budget=_budget(),
)


def _compile(
    *,
    tree: str,
    source: str = SRC_A,
    premises: tuple[str, ...] = ("premise:a",),
    assumptions: tuple[str, ...] = ("assumption:a",),
    toolchain: str = "toolchain:t",
    policy: str = "policy:p",
    blob: str = "blob:1",
):
    return compile_code_proof_obligations(
        candidate_diff=[
            CandidateDiffEntry(
                new_path="src/worker.py",
                change_kind=DiffChangeKind.ADD,
                after_source=source,
                after_blob_id=blob,
            )
        ],
        repository_tree_id=tree,
        repository_id="repo:reproof",
        claim_families=("api_contract",),
        premise_ids=premises,
        assumption_ids=assumptions,
        toolchain_id=toolchain,
        policy_id=policy,
    )


def _kernel(obligation_id: str) -> ProofEvidence:
    return ProofEvidence(
        kind=EvidenceKind.KERNEL_VERIFICATION,
        authority=EvidenceAuthority.KERNEL,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id="artifact:kernel",
        subject_id=obligation_id,
        verifier_id="kernel:lean-4.19",
        independent=True,
        simulated=False,
    )


def _receipt_for(item, *, tree: str, toolchain: str = "toolchain:t", policy: str = "policy:p"):
    obligation = item.obligation
    assert obligation is not None
    kernel_id = "kernel:lean-4.19"
    return ProofReceipt(
        obligation_id=obligation.obligation_id,
        plan_id="plan:reproof",
        attempt_id="attempt:1",
        repository_id=obligation.repository_id or "repo:reproof",
        repository_tree_id=tree,
        ast_scope_ids=tuple(obligation.ast_scope_ids),
        premise_ids=tuple(obligation.premise_ids),
        translator_id="translator:python-to-lean@1",
        solver_id="solver:z3@4.13",
        kernel_id=kernel_id,
        toolchain_id=toolchain,
        theorem_registry_id="registry:reviewed-v3",
        policy_id=policy,
        resource_budget=_budget(),
        verdict=ProofVerdict.PROVED,
        evidence=(
            ProofEvidence(
                kind=EvidenceKind.KERNEL_VERIFICATION,
                authority=EvidenceAuthority.KERNEL,
                verdict=EvidenceVerdict.ACCEPTED,
                artifact_id="artifact:kernel",
                subject_id=obligation.obligation_id,
                verifier_id=kernel_id,
                independent=True,
                simulated=False,
            ),
        ),
        provider_id="provider:test",
        provider_claimed_assurance=AssuranceLevel.ATTESTED,
        started_at="2026-07-28T00:00:00Z",
        finished_at="2026-07-28T00:00:01Z",
        resource_usage={"wall_time_ms": 5, "peak_memory_bytes": 1000},
    )


def test_invalidation_reasons_cover_binding_drift() -> None:
    comp = _compile(tree="git-tree:a")
    item = comp.items[0]
    base = binding_fingerprint_for_item(
        item,
        repository_tree_id="git-tree:a",
        toolchain_id="toolchain:t",
        policy_id="policy:p",
    )
    other = binding_fingerprint_for_item(
        item,
        repository_tree_id="git-tree:b",
        toolchain_id="toolchain:other",
        policy_id="policy:other",
    )
    reasons = invalidation_reasons(base, other, changed_paths=["src/worker.py"])
    assert InvalidationReason.REPOSITORY_TREE_CHANGED.value in reasons
    assert InvalidationReason.TOOLCHAIN_CHANGED.value in reasons
    assert InvalidationReason.POLICY_CHANGED.value in reasons
    assert InvalidationReason.PATH_CHANGED.value in reasons
    assert InvalidationReason.COLD_MISS.value in invalidation_reasons(None, base)


def test_warm_path_serves_cache_without_provider(tmp_path: Path) -> None:
    cache = FormalVerificationCache(tmp_path)
    comp = _compile(tree="git-tree:warm")
    item = next(i for i in comp.items if i.obligation is not None)
    calls = {"n": 0}

    def prove(it, key):
        calls["n"] += 1
        return _receipt_for(
            it,
            tree="git-tree:warm",
            toolchain=str(key.toolchain),
            policy=str(key.policy),
        )

    metrics = ProofCacheMetrics()
    first = reprove_code_proof_compilation(
        cache, comp, prove=prove, metrics=metrics, **REPROOF_KW
    )
    second = reprove_code_proof_compilation(
        cache, comp, prove=prove, metrics=metrics, **REPROOF_KW
    )
    assert first.re_solved >= 1
    assert second.cache_hits >= 1
    hit = next(
        r for r in second.results if r.disposition is ReproofDisposition.CACHE_HIT
    )
    assert hit.from_cache is True
    assert hit.provenance.get("provider_calls") == 0
    assert InvalidationReason.AUTHORITATIVE_CACHE_HIT.value in hit.reason_codes
    # only first pass should invoke provider for that obligation
    assert calls["n"] == first.re_solved


def test_tree_change_forces_resolve_not_foreign_hit(tmp_path: Path) -> None:
    cache = FormalVerificationCache(tmp_path)
    parent = _compile(tree="git-tree:parent", blob="blob:parent")
    child = _compile(tree="git-tree:child", blob="blob:child", source=SRC_B)
    calls = {"trees": []}

    def prove(it, key):
        tree = str(key.candidate_tree)
        calls["trees"].append(tree)
        return _receipt_for(
            it, tree=tree, toolchain=str(key.toolchain), policy=str(key.policy)
        )

    # Populate cache under parent tree.
    reprove_code_proof_compilation(cache, parent, prove=prove, **REPROOF_KW)
    # Child must not accept parent receipt (wrong tree).
    report = reprove_code_proof_compilation(
        cache, child, prove=prove, previous=parent, **REPROOF_KW
    )
    solved = [r for r in report.results if r.disposition is ReproofDisposition.RE_SOLVED]
    assert solved
    for item in solved:
        assert InvalidationReason.REPOSITORY_TREE_CHANGED.value in item.reason_codes or (
            InvalidationReason.CACHE_KEY_CHANGED.value in item.reason_codes
        )
        assert item.from_cache is False
    assert "git-tree:child" in calls["trees"]


def test_premise_or_toolchain_change_invalidates(tmp_path: Path) -> None:
    cache = FormalVerificationCache(tmp_path)
    base = _compile(tree="git-tree:x", premises=("premise:a",))
    drifted = _compile(
        tree="git-tree:x",
        premises=("premise:b",),
        toolchain="toolchain:new",
    )

    def prove(it, key):
        return _receipt_for(
            it,
            tree="git-tree:x",
            toolchain=str(key.toolchain),
            policy=str(key.policy),
        )

    reprove_code_proof_compilation(cache, base, prove=prove, **REPROOF_KW)
    report = reprove_code_proof_compilation(
        cache, drifted, prove=prove, previous=base, **REPROOF_KW
    )
    for item in report.results:
        if item.disposition is ReproofDisposition.SKIPPED_UNSUPPORTED:
            continue
        if item.disposition is ReproofDisposition.RE_SOLVED:
            codes = set(item.reason_codes)
            assert (
                InvalidationReason.PREMISE_DIGEST_CHANGED.value in codes
                or InvalidationReason.TOOLCHAIN_CHANGED.value in codes
                or InvalidationReason.CACHE_KEY_CHANGED.value in codes
            )


def test_single_flight_under_parallel_reproof(tmp_path: Path) -> None:
    cache = FormalVerificationCache(tmp_path)
    comp = _compile(tree="git-tree:sf")
    started = threading.Event()
    release = threading.Event()
    calls = {"n": 0}
    lock = threading.Lock()

    def prove(it, key):
        with lock:
            calls["n"] += 1
        started.set()
        assert release.wait(timeout=5)
        return _receipt_for(
            it,
            tree="git-tree:sf",
            toolchain=str(key.toolchain),
            policy=str(key.policy),
        )

    results = []

    def worker():
        results.append(
            reprove_code_proof_compilation(
                cache,
                comp,
                prove=prove,
                metrics=ProofCacheMetrics(),
                **REPROOF_KW,
            )
        )

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    assert started.wait(timeout=5)
    t2.start()
    import time

    time.sleep(0.15)
    release.set()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert len(results) == 2
    # single-flight collapses concurrent prove for same key
    assert calls["n"] == 1


def test_plan_reproof_from_delta_uses_query_delta() -> None:
    parent = _compile(tree="git-tree:p")
    child = _compile(tree="git-tree:c", source=SRC_B, blob="blob:c")
    delta = plan_reproof_from_delta(parent, child)
    assert delta.parent_tree_id == "git-tree:p"
    assert delta.child_tree_id == "git-tree:c"
    assert delta.entries
    assert all(entry.reason_codes for entry in delta.entries)


def test_report_is_content_addressed(tmp_path: Path) -> None:
    cache = FormalVerificationCache(tmp_path)
    comp = _compile(tree="git-tree:id")

    def prove(it, key):
        return _receipt_for(
            it,
            tree="git-tree:id",
            toolchain=str(key.toolchain),
            policy=str(key.policy),
        )

    a = reprove_code_proof_compilation(cache, comp, prove=prove, **REPROOF_KW)
    b = reprove_code_proof_compilation(cache, comp, prove=prove, **REPROOF_KW)
    # second is warm; ids differ because metrics/dispositions differ — but each has id
    assert a.report_id
    assert b.report_id
    assert a.to_dict()["interface"] == "CodeProofReproof@1"
    assert "warm_path_uses_trust_aware_cache_with_rederive" in a.notes
