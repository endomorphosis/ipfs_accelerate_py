"""DCR-034: cache proof evidence with exact invalidation dependencies.

Acceptance:
* Stale or cross-epoch evidence cannot be selected.
* Cache-hit reconstruction equals a cold run.
* Only reconstructed evidence is admitted.
* Any input/policy/solver/schema/source/runtime/capability/epoch/kernel/graph/
  tree/toolchain root change invalidates descendants.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.dcr_proof_cache import (
    DCR_PROOF_CACHE_INDEX_SCHEMA,
    DcrCacheDisposition,
    DcrCacheReason,
    DcrEvidenceKind,
    DcrProofCache,
    DcrProofCacheError,
    DcrProofCacheKey,
    PROOF_CACHE_INTERFACE,
    PROOF_INVALIDATION_INTERFACE,
    ProofDependencyRoot,
    ProofDependencyRootKind,
    ProofInvalidationReceipt,
    build_dcr_proof_cache_key,
    build_dependency_roots,
)
from ipfs_accelerate_py.agent_supervisor.proof.kernel_reconstruction import (
    DEFAULT_KERNEL_VERSION,
    Counterexample,
    ProofClaim,
    ReconstructionStatus,
    proof_term_digest,
    reconstruct_proof,
)


def _roots(
    *,
    epoch: str = "epoch-2026-08-01",
    policy: str = "policy:v1",
    solver: str = "solver:z3@1",
    kernel: str = "kernel:dcr@1",
    tree: str = "tree:abc",
    graph: str = "graph:def",
    **overrides: str,
) -> tuple[ProofDependencyRoot, ...]:
    values = {
        "input": "input:obligation-body@1",
        "policy": policy,
        "solver": solver,
        "schema": "schema:proof-kernel@1",
        "source": "source:mcp-contract-graph@1",
        "runtime": "runtime:witness@1",
        "capability": "capability:portfolio@1",
        "epoch": epoch,
        "kernel": kernel,
        "graph": graph,
        "tree": tree,
        "toolchain": "toolchain:locked@1",
    }
    values.update(overrides)
    return build_dependency_roots(**values)


def _key(
    *,
    obligation_id: str = "obligation:dcr034:sound",
    epoch: str = "epoch-2026-08-01",
    evidence_kind: DcrEvidenceKind = DcrEvidenceKind.PROOF_KERNEL_RECEIPT,
    **root_overrides: str,
) -> DcrProofCacheKey:
    return DcrProofCacheKey(
        obligation_id=obligation_id,
        dependency_roots=_roots(epoch=epoch, **root_overrides),
        evidence_kind=evidence_kind,
        kernel_version=DEFAULT_KERNEL_VERSION,
    )


def _claim(
    *,
    obligation_id: str = "obligation:dcr034:sound",
    tree_id: str = "tree:abc",
    graph_root: str = "graph:def",
) -> ProofClaim:
    term = {
        "theorem": "contract_edge_sound",
        "steps": ["intro", "exact h"],
        "conclusion": "True",
    }
    return ProofClaim(
        obligation_id=obligation_id,
        proof_term=term,
        certificate_digest=proof_term_digest(term),
        kernel_version=DEFAULT_KERNEL_VERSION,
        root_ids=("ipfs-accelerate", "swissknife"),
        tree_id=tree_id,
        graph_root=graph_root,
        independent=True,
        proof_children=tuple(proof_term_digest(step) for step in term["steps"]),
    )


def _reconstructed_receipt(
    *,
    obligation_id: str = "obligation:dcr034:sound",
) -> tuple[ProofClaim, object]:
    claim = _claim(obligation_id=obligation_id)
    receipt = reconstruct_proof(
        claim,
        expected_root_ids=claim.root_ids,
        expected_tree_id=claim.tree_id,
        expected_graph_root=claim.graph_root,
        kernel_version=DEFAULT_KERNEL_VERSION,
    )
    assert receipt.valid
    assert receipt.status is ReconstructionStatus.RECONSTRUCTED
    return claim, receipt


def _counterexample(
    *,
    obligation_id: str = "obligation:dcr034:refute",
) -> Counterexample:
    return Counterexample(
        obligation_id=obligation_id,
        violated_property="route_must_resolve",
        summary="unknown tool call rejected",
        witness={
            "edge_id": "edge:route-to-dispatcher",
            "method": "tools/call",
            "terminal_state": "refuted",
            "receipt_cid": "receipt:live-unknown-call",
        },
        graph_edge_ids=("edge:route-to-dispatcher",),
        transcript_receipt_ids=("receipt:live-unknown-call",),
        root_ids=("ipfs-accelerate",),
        tree_id="tree:abc",
        graph_root="graph:def",
        minimized=True,
        inferred_observations=False,
    )


def test_interfaces_are_declared() -> None:
    assert PROOF_CACHE_INTERFACE == "ProofCache@1"
    assert PROOF_INVALIDATION_INTERFACE == "ProofInvalidation@1"
    assert DcrProofCache.INTERFACE == PROOF_CACHE_INTERFACE
    assert DcrProofCache.INVALIDATION_INTERFACE == PROOF_INVALIDATION_INTERFACE
    assert ProofDependencyRootKind.EPOCH.value == "epoch"


def test_dependency_roots_require_complete_set() -> None:
    incomplete = (
        ProofDependencyRoot(kind="epoch", digest="epoch-1"),
        ProofDependencyRoot(kind="policy", digest="policy-1"),
    )
    with pytest.raises(DcrProofCacheError, match="missing required kinds"):
        DcrProofCacheKey(
            obligation_id="obligation:x",
            dependency_roots=incomplete,
        )


def test_build_helpers_produce_stable_key_identity() -> None:
    first = build_dcr_proof_cache_key(
        obligation_id="obligation:a",
        input="in-1",
        policy="pol-1",
        solver="sol-1",
        schema="sch-1",
        source="src-1",
        runtime="rt-1",
        capability="cap-1",
        epoch="epoch-1",
        kernel="ker-1",
        graph="gr-1",
        tree="tr-1",
        toolchain="tc-1",
    )
    second = build_dcr_proof_cache_key(
        obligation_id="obligation:a",
        dependency_roots=build_dependency_roots(
            input="in-1",
            policy="pol-1",
            solver="sol-1",
            schema="sch-1",
            source="src-1",
            runtime="rt-1",
            capability="cap-1",
            epoch="epoch-1",
            kernel="ker-1",
            graph="gr-1",
            tree="tr-1",
            toolchain="tc-1",
        ),
    )
    assert first.key_id == second.key_id
    assert first.epoch == "epoch-1"
    assert first.to_dict() == second.to_dict()


def test_put_rejects_unreconstructed_evidence(tmp_path: Path) -> None:
    cache = DcrProofCache(tmp_path)
    key = _key()
    claim = _claim()
    # Force invalid by stripping independence.
    bad_claim = ProofClaim(
        obligation_id=claim.obligation_id,
        proof_term=claim.proof_term,
        certificate_digest=claim.certificate_digest,
        kernel_version=claim.kernel_version,
        root_ids=claim.root_ids,
        tree_id=claim.tree_id,
        graph_root=claim.graph_root,
        independent=False,
        provider_status="verified",
        proof_children=claim.proof_children,
    )
    invalid = reconstruct_proof(bad_claim)
    assert not invalid.valid

    result = cache.put(key, invalid, claim=bad_claim)
    assert not result.stored
    assert DcrCacheReason.NOT_RECONSTRUCTED.value in result.reason_codes
    assert cache.lookup(key).disposition is DcrCacheDisposition.MISS


def test_store_and_hit_reconstructs_equal_to_cold_run(tmp_path: Path) -> None:
    cache = DcrProofCache(tmp_path)
    key = _key()
    claim, receipt = _reconstructed_receipt()

    cold = reconstruct_proof(
        claim,
        expected_root_ids=claim.root_ids,
        expected_tree_id=claim.tree_id,
        expected_graph_root=claim.graph_root,
        kernel_version=DEFAULT_KERNEL_VERSION,
    )
    assert cold.content_id == receipt.content_id

    stored = cache.put(key, receipt, claim=claim)
    assert stored.stored
    assert stored.receipt_cid == receipt.content_id
    assert DcrCacheReason.STORED.value in stored.reason_codes

    hit = cache.lookup(key)
    assert hit.disposition is DcrCacheDisposition.HIT
    assert hit.hit
    assert hit.reconstructed
    assert hit.authoritative
    assert hit.receipt is not None
    assert hit.receipt.content_id == cold.content_id
    assert hit.receipt.to_dict() == cold.to_dict()
    assert hit.receipt_cid == cold.content_id
    assert DcrCacheReason.CACHE_HIT.value in hit.reason_codes

    # get() returns the reconstructed receipt.
    got = cache.get(key)
    assert got is not None
    assert got.content_id == cold.content_id


def test_cross_epoch_evidence_cannot_be_selected(tmp_path: Path) -> None:
    cache = DcrProofCache(tmp_path)
    key = _key(epoch="epoch-2026-08-01")
    claim, receipt = _reconstructed_receipt()
    assert cache.put(key, receipt, claim=claim).stored

    # Same semantic inputs but a different epoch must miss/reject.
    other_epoch_key = _key(epoch="epoch-2026-09-01")
    cross = cache.lookup(other_epoch_key)
    assert not cross.hit
    # Different key_id => miss (content-addressed), never a hit.
    assert cross.disposition is DcrCacheDisposition.MISS

    # Explicit current_epoch check against the original key rejects cross-epoch.
    rejected = cache.lookup(key, current_epoch="epoch-2026-09-01")
    assert not rejected.hit
    assert rejected.disposition is DcrCacheDisposition.REJECTED
    assert DcrCacheReason.CROSS_EPOCH.value in rejected.reason_codes
    assert DcrCacheReason.STALE.value in rejected.reason_codes


def test_stale_dependency_root_cannot_be_selected(tmp_path: Path) -> None:
    cache = DcrProofCache(tmp_path)
    key = _key(policy="policy:v1")
    claim, receipt = _reconstructed_receipt()
    assert cache.put(key, receipt, claim=claim).stored

    live_roots = _roots(policy="policy:v2")
    stale = cache.lookup(key, current_roots=live_roots)
    assert not stale.hit
    assert stale.disposition is DcrCacheDisposition.REJECTED
    assert DcrCacheReason.ROOT_MISMATCH.value in stale.reason_codes
    assert DcrCacheReason.STALE.value in stale.reason_codes


def test_invalidate_dependency_root_tombstones_descendants(tmp_path: Path) -> None:
    cache = DcrProofCache(tmp_path)
    key = _key(solver="solver:z3@1")
    claim, receipt = _reconstructed_receipt()
    assert cache.put(key, receipt, claim=claim).stored
    assert cache.lookup(key).hit

    solver_root = ProofDependencyRoot(
        kind=ProofDependencyRootKind.SOLVER,
        digest="solver:z3@1",
    )
    invalidation = cache.invalidate(solver_root)
    assert isinstance(invalidation, ProofInvalidationReceipt)
    assert invalidation.INTERFACE == PROOF_INVALIDATION_INTERFACE
    assert key.key_id in invalidation.invalidated_key_ids
    assert cache.is_tombstoned(key)

    after = cache.lookup(key)
    assert after.disposition is DcrCacheDisposition.INVALIDATED
    assert not after.hit
    assert DcrCacheReason.TOMBSTONED.value in after.reason_codes

    # Re-store after invalidation is refused while tombstoned.
    refused = cache.put(key, receipt, claim=claim)
    assert not refused.stored
    assert DcrCacheReason.TOMBSTONED.value in refused.reason_codes


def test_each_required_root_kind_invalidates(tmp_path: Path) -> None:
    claim, receipt = _reconstructed_receipt()
    for kind in ProofDependencyRootKind:
        cache = DcrProofCache(tmp_path / kind.value)
        key = _key()
        assert cache.put(key, receipt, claim=claim).stored
        root = ProofDependencyRoot(kind=kind, digest=key.root_digest(kind))
        result = cache.invalidate(root)
        assert key.key_id in result.invalidated_key_ids
        assert cache.lookup(key).disposition is DcrCacheDisposition.INVALIDATED


def test_equivocation_poisons_key(tmp_path: Path) -> None:
    cache = DcrProofCache(tmp_path)
    key = _key()
    claim, receipt = _reconstructed_receipt()
    assert cache.put(key, receipt, claim=claim).stored

    alt_term = {
        "theorem": "contract_edge_sound",
        "steps": ["intro", "simp", "exact h"],
        "conclusion": "True",
    }
    alt_claim = ProofClaim(
        obligation_id=claim.obligation_id,
        proof_term=alt_term,
        certificate_digest=proof_term_digest(alt_term),
        kernel_version=DEFAULT_KERNEL_VERSION,
        root_ids=claim.root_ids,
        tree_id=claim.tree_id,
        graph_root=claim.graph_root,
        independent=True,
        proof_children=tuple(proof_term_digest(step) for step in alt_term["steps"]),
    )
    alt_receipt = reconstruct_proof(
        alt_claim,
        expected_root_ids=alt_claim.root_ids,
        expected_tree_id=alt_claim.tree_id,
        expected_graph_root=alt_claim.graph_root,
    )
    assert alt_receipt.valid
    assert alt_receipt.content_id != receipt.content_id

    poisoned = cache.put(key, alt_receipt, claim=alt_claim)
    assert not poisoned.stored
    assert DcrCacheReason.POISONED.value in poisoned.reason_codes
    assert cache.is_tombstoned(key)
    assert not cache.lookup(key).hit


def test_expired_entry_is_stale(tmp_path: Path) -> None:
    clock = {"now": 1_000.0}

    def _clock() -> float:
        return clock["now"]

    cache = DcrProofCache(tmp_path, clock=_clock, default_ttl_seconds=10)
    key = _key()
    claim, receipt = _reconstructed_receipt()
    assert cache.put(key, receipt, claim=claim, ttl_seconds=10).stored
    assert cache.lookup(key).hit

    clock["now"] = 1_000.0 + 11
    expired = cache.lookup(key)
    assert not expired.hit
    assert expired.disposition is DcrCacheDisposition.REJECTED
    assert DcrCacheReason.EXPIRED.value in expired.reason_codes
    assert DcrCacheReason.STALE.value in expired.reason_codes


def test_counterexample_cache_roundtrip(tmp_path: Path) -> None:
    cache = DcrProofCache(tmp_path)
    key = _key(
        obligation_id="obligation:dcr034:refute",
        evidence_kind=DcrEvidenceKind.COUNTEREXAMPLE,
    )
    cx = _counterexample()
    stored = cache.put(key, cx)
    assert stored.stored
    assert stored.receipt_cid == cx.content_id

    hit = cache.lookup(key)
    assert hit.hit
    assert hit.counterexample is not None
    assert hit.counterexample.content_id == cx.content_id
    assert hit.counterexample.to_dict() == cx.to_dict()


def test_durable_reload_preserves_hits_and_index(tmp_path: Path) -> None:
    path = tmp_path / "cache"
    first = DcrProofCache(path)
    key = _key()
    claim, receipt = _reconstructed_receipt()
    assert first.put(key, receipt, claim=claim).stored
    index_path = first.write_cache_index(tmp_path / "cache-index.json")
    assert index_path.is_file()
    payload = index_path.read_text(encoding="utf-8")
    assert DCR_PROOF_CACHE_INDEX_SCHEMA in payload
    assert key.key_id in payload

    reloaded = DcrProofCache(path)
    hit = reloaded.lookup(key)
    assert hit.hit
    assert hit.receipt is not None
    assert hit.receipt.content_id == receipt.content_id


def test_binding_mismatch_on_obligation_rejected(tmp_path: Path) -> None:
    cache = DcrProofCache(tmp_path)
    key = _key(obligation_id="obligation:a")
    claim, receipt = _reconstructed_receipt(obligation_id="obligation:b")
    result = cache.put(key, receipt, claim=claim)
    assert not result.stored
    assert DcrCacheReason.BINDING_MISMATCH.value in result.reason_codes


def test_reconstruction_mismatch_on_put_rejected(tmp_path: Path) -> None:
    cache = DcrProofCache(tmp_path)
    key = _key()
    claim, receipt = _reconstructed_receipt()
    # Supply a claim that does not reconstruct to the stored receipt.
    other_term = {
        "theorem": "other",
        "steps": ["skip"],
        "conclusion": "False",
    }
    other_claim = ProofClaim(
        obligation_id=claim.obligation_id,
        proof_term=other_term,
        certificate_digest=proof_term_digest(other_term),
        kernel_version=DEFAULT_KERNEL_VERSION,
        root_ids=claim.root_ids,
        tree_id=claim.tree_id,
        graph_root=claim.graph_root,
        independent=True,
        proof_children=(proof_term_digest("skip"),),
    )
    result = cache.put(key, receipt, claim=other_claim)
    assert not result.stored
    assert DcrCacheReason.RECONSTRUCTION_MISMATCH.value in result.reason_codes


def test_stats_and_contains(tmp_path: Path) -> None:
    cache = DcrProofCache(tmp_path)
    key = _key()
    claim, receipt = _reconstructed_receipt()
    assert key not in cache
    assert cache.put(key, receipt, claim=claim).stored
    assert key in cache
    stats = cache.stats()
    assert stats["entry_count"] == 1
    assert stats["interface"] == PROOF_CACHE_INTERFACE
    assert stats["invalidation_interface"] == PROOF_INVALIDATION_INTERFACE
