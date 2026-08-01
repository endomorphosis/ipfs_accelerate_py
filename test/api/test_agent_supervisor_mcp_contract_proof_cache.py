"""SCA-070 contract tests for MCP trust-aware proof caching."""

from __future__ import annotations

import importlib
import json
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis import (
    content_identity_bridge,
)
from ipfs_accelerate_py.agent_supervisor.analysis.content_identity_bridge import (
    LOGIC_IR_PROFILE,
    MULTICODEC_RAW,
    STRICT_ARTIFACT_PROFILE,
    identify_strict_artifact,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_cache import (
    CacheLookupStatus,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
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
from ipfs_accelerate_py.agent_supervisor.proof.mcp_contract_proof_cache import (
    DEFAULT_NEGATIVE_TTL_SECONDS,
    IdentityBinding,
    MAX_NEGATIVE_TTL_SECONDS,
    ProofCacheKey,
    ProofCacheReason,
    ProofCacheValidationError,
    TrustAwareProofCache,
)
from ipfs_accelerate_py.agent_supervisor.proof.mcp_contract_prover import (
    ContractProofOutcome,
    ContractProofRoute,
    McpContractProofResult,
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


def _identity(name: str, logical_id: str | None = None, version: object = 1):
    identity = identify_strict_artifact(
        {"component": name, "version": version}
    )
    return IdentityBinding.from_identity(
        identity, logical_id=logical_id or f"{name}-1"
    )


def _key(**changes: object) -> ProofCacheKey:
    values: dict[str, object] = {
        "snapshot": _identity("snapshot", "tree-1"),
        "scope": (_identity("scope", "scope-1"),),
        "property_catalog": _identity("catalog", "catalog-1"),
        "obligation": _identity("obligation", "obligation-1"),
        "premises": (
            _identity("premise-a", "premise-a"),
            _identity("premise-b", "premise-b"),
        ),
        "assumptions": (_identity("assumption", "assumption-1"),),
        "provider": _identity("provider", "provider-1"),
        "translator": _identity("translator", "translator-1"),
        "solver": _identity("solver", "solver-1"),
        "kernel": _identity("kernel", "kernel-1"),
        "toolchain": _identity("toolchain", "toolchain-1"),
        "theorem_registry": _identity("registry", "registry-1"),
        "policy": _identity("policy", "policy-1"),
        "capability_report": _identity("capability", "capability-1"),
        "resource_budget": _budget(),
        "required_assurance": AssuranceLevel.KERNEL_VERIFIED,
        "route": ContractProofRoute.LOCAL_SCHEMA,
    }
    values.update(changes)
    return ProofCacheKey(**values)


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
    obligation_id: str = "obligation-1",
    tree_id: str = "tree-1",
    evidence: tuple[ProofEvidence, ...] | None = None,
    verdict: ProofVerdict = ProofVerdict.PROVED,
    freshness: EvidenceFreshness = EvidenceFreshness.CURRENT,
) -> ProofReceipt:
    return ProofReceipt(
        obligation_id=obligation_id,
        plan_id=f"plan:{obligation_id}",
        attempt_id=f"attempt:{obligation_id}",
        repository_id="repository-1",
        repository_tree_id=tree_id,
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
        evidence=evidence
        if evidence is not None
        else (_kernel_evidence(obligation_id=obligation_id),),
        freshness=freshness,
        kernel_receipt_id=f"kernel-receipt:{obligation_id}",
    )


def _proved_result(
    *,
    receipt: ProofReceipt | None = None,
) -> McpContractProofResult:
    accepted = receipt or _receipt()
    return McpContractProofResult(
        obligation_id=accepted.obligation_id,
        outcome=ContractProofOutcome.PROVED,
        route=ContractProofRoute.LOCAL_SCHEMA,
        reason_codes=("local_schema_proved",),
        receipt=accepted,
    )


def _inconclusive_result() -> McpContractProofResult:
    receipt = _receipt(
        evidence=(),
        verdict=ProofVerdict.INCONCLUSIVE,
    )
    return McpContractProofResult(
        obligation_id=receipt.obligation_id,
        outcome=ContractProofOutcome.INCONCLUSIVE,
        route=ContractProofRoute.LOCAL_SCHEMA,
        reason_codes=("local_schema_inconclusive",),
        receipt=receipt,
    )


def test_identity_retains_and_revalidates_canonical_bytes_against_cid() -> None:
    binding = _identity("artifact")
    assert binding.profile == STRICT_ARTIFACT_PROFILE
    assert binding.canonical_bytes
    assert IdentityBinding.from_dict(binding.to_dict()) == binding

    with pytest.raises(ProofCacheValidationError) as error:
        replace(binding, canonical_bytes=binding.canonical_bytes + b" ")
    assert error.value.reason_code == ProofCacheReason.POISONED.value


def test_identity_binding_uses_live_bridge_after_module_reload() -> None:
    reloaded = importlib.reload(content_identity_bridge)
    identity = reloaded.identify_strict_artifact(
        {"component": "reload-boundary", "version": 1}
    )

    binding = IdentityBinding.from_identity(
        identity,
        logical_id="reload-boundary-1",
    )
    assert IdentityBinding.from_dict(binding.to_dict()) == binding

    with pytest.raises(ProofCacheValidationError) as poisoned:
        IdentityBinding.from_identity(
            replace(identity, cid="not-a-cid"),
            logical_id="reload-boundary-poisoned",
        )
    assert poisoned.value.reason_code == ProofCacheReason.POISONED.value


def test_key_binds_every_semantic_dimension_and_is_order_invariant() -> None:
    baseline = _key()
    mutations = {
        "snapshot": _identity("snapshot", "tree-1", 2),
        "scope": (_identity("scope", "scope-1", 2),),
        "property_catalog": _identity("catalog", "catalog-1", 2),
        "obligation": _identity("obligation", "obligation-1", 2),
        "premises": (
            _identity("premise-a", "premise-a", 2),
            _identity("premise-b", "premise-b"),
        ),
        "assumptions": (_identity("assumption", "assumption-1", 2),),
        "provider": _identity("provider", "provider-1", 2),
        "translator": _identity("translator", "translator-1", 2),
        "solver": _identity("solver", "solver-1", 2),
        "kernel": _identity("kernel", "kernel-1", 2),
        "toolchain": _identity("toolchain", "toolchain-1", 2),
        "theorem_registry": _identity("registry", "registry-1", 2),
        "policy": _identity("policy", "policy-1", 2),
        "capability_report": _identity("capability", "capability-1", 2),
        "resource_budget": replace(_budget(), wall_time_ms=9_999),
        "required_assurance": AssuranceLevel.ATTESTED,
        "route": ContractProofRoute.KERNEL,
    }
    for name, value in mutations.items():
        assert _key(**{name: value}).key_id != baseline.key_id, name

    assert _key(
        premises=tuple(reversed(baseline.premises)),
        assumptions=tuple(reversed(baseline.assumptions)),
    ).key_id == baseline.key_id


def test_warm_exact_hit_avoids_provider_and_rederives_assurance(
    tmp_path: Path,
) -> None:
    cache = TrustAwareProofCache(tmp_path)
    calls = 0

    def provider() -> McpContractProofResult:
        nonlocal calls
        calls += 1
        return _proved_result()

    cold = cache.get_or_prove(_key(), provider)
    warm = cache.get_or_prove(_key(), provider)

    assert calls == 1
    assert not cold.cache_hit
    assert warm.cache_hit
    assert warm.result.reason_codes == (ProofCacheReason.CACHE_HIT.value,)
    assert warm.authoritative_assurance is AssuranceLevel.KERNEL_VERIFIED
    assert warm.receipt.provider_claimed_assurance is AssuranceLevel.UNVERIFIED


def test_wrong_tree_store_and_close_sibling_lookup_reject_with_reason(
    tmp_path: Path,
) -> None:
    cache = TrustAwareProofCache(tmp_path)
    wrong_receipt = _receipt(tree_id="tree-other")
    rejected = cache.put(_key(), wrong_receipt)
    assert not rejected.stored
    assert ProofCacheReason.WRONG_TREE.value in rejected.reason_codes

    assert cache.put(_key(), _receipt()).stored
    other_tree = _key(snapshot=_identity("snapshot", "tree-2"))
    miss = cache.lookup(other_tree)
    assert miss.status is CacheLookupStatus.REJECTED
    assert miss.reason_code == ProofCacheReason.WRONG_TREE.value


def test_cross_profile_sibling_is_not_aliased(tmp_path: Path) -> None:
    from multiformats import CID, multihash

    cache = TrustAwareProofCache(tmp_path)
    base_key = _key()
    assert cache.put(base_key, _receipt()).stored
    original = base_key.capability_report
    raw_cid = str(
        CID(
            "base32",
            1,
            "raw",
            multihash.digest(original.canonical_bytes, "sha2-256"),
        )
    )
    cross_profile = IdentityBinding(
        logical_id=original.logical_id,
        profile=LOGIC_IR_PROFILE,
        cid=raw_cid,
        canonical_bytes=original.canonical_bytes,
        digest=original.digest,
        multicodec=MULTICODEC_RAW,
        domain="cache-test",
        artifact_schema="1",
    )
    lookup = cache.lookup(_key(capability_report=cross_profile))
    assert lookup.status is CacheLookupStatus.REJECTED
    assert lookup.reason_code == ProofCacheReason.CROSS_PROFILE.value


def test_private_candidate_and_stale_material_fail_closed(tmp_path: Path) -> None:
    cache = TrustAwareProofCache(tmp_path)
    private_key = _key(
        assumptions=(
            IdentityBinding.from_identity(
                identify_strict_artifact(
                    {"assumption": "opaque", "private_witness": "never-reflect"}
                ),
                logical_id="assumption-1",
            ),
        )
    )
    private = cache.lookup(private_key)
    assert private.status is CacheLookupStatus.REJECTED
    assert private.reason_code == ProofCacheReason.PRIVATE_MATERIAL.value

    solver_evidence = ProofEvidence(
        kind=EvidenceKind.SOLVER_RESULT,
        authority=EvidenceAuthority.SOLVER,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id="solver-artifact",
        subject_id="obligation-1",
        verifier_id="solver-1",
        independent=True,
    )
    candidate = cache.put(_key(), _receipt(evidence=(solver_evidence,)))
    assert not candidate.stored
    assert ProofCacheReason.CANDIDATE_ONLY.value in candidate.reason_codes

    stale = cache.put(
        _key(), _receipt(freshness=EvidenceFreshness.STALE)
    )
    assert not stale.stored
    assert ProofCacheReason.STALE.value in stale.reason_codes


def test_detached_provider_result_is_rejected_instead_of_returned(
    tmp_path: Path,
) -> None:
    cache = TrustAwareProofCache(tmp_path)
    detached = _proved_result(receipt=_receipt(tree_id="tree-other"))
    with pytest.raises(ProofCacheValidationError) as error:
        cache.get_or_prove(_key(), lambda: detached)
    assert error.value.reason_code == ProofCacheReason.WRONG_TREE.value


def test_poisoned_authoritative_row_rejects_with_reason(tmp_path: Path) -> None:
    cache = TrustAwareProofCache(tmp_path)
    key = _key()
    assert cache.put(key, _receipt()).stored
    connection = cache.authoritative_cache._connect()
    try:
        row = connection.execute(
            "SELECT entry_json FROM proof_cache_entries"
        ).fetchone()
        payload = json.loads(row["entry_json"])
        payload["entry_digest"] = "sha256:" + "0" * 64
        connection.execute(
            "UPDATE proof_cache_entries SET entry_json=?",
            (json.dumps(payload, sort_keys=True, separators=(",", ":")),),
        )
    finally:
        connection.close()

    rejected = cache.lookup(key)
    assert rejected.status is CacheLookupStatus.REJECTED
    assert rejected.reason_code == ProofCacheReason.POISONED.value


def test_concurrent_identical_requests_use_one_provider_flight(
    tmp_path: Path,
) -> None:
    cache = TrustAwareProofCache(tmp_path)
    key = _key()
    barrier = threading.Barrier(8)
    lock = threading.Lock()
    calls = 0
    results = []

    def provider() -> McpContractProofResult:
        nonlocal calls
        with lock:
            calls += 1
        time.sleep(0.05)
        return _proved_result()

    def worker() -> None:
        barrier.wait()
        result = cache.get_or_prove(
            key,
            provider,
            lease_seconds=2,
            wait_timeout_seconds=5,
        )
        with lock:
            results.append(result)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert calls == 1
    assert len(results) == 8
    assert all(
        item.authoritative_assurance is AssuranceLevel.KERNEL_VERIFIED
        for item in results
    )
    assert sum(not item.shared_flight for item in results) == 1


def test_positive_ttl_and_negative_flight_ttl_are_bounded(tmp_path: Path) -> None:
    now = [1_000.0]
    cache = TrustAwareProofCache(
        tmp_path,
        positive_ttl_seconds=10,
        negative_ttl_seconds=3,
        clock=lambda: now[0],
    )
    assert cache.put(_key(), _receipt(), ttl_seconds=1_000).stored
    now[0] += 10
    stale = cache.lookup(_key())
    assert stale.status is CacheLookupStatus.REJECTED
    assert ProofCacheReason.STALE.value in stale.reason_codes

    another = TrustAwareProofCache(
        tmp_path / "negative",
        negative_ttl_seconds=3,
    )
    outcome = another.get_or_prove(_key(), _inconclusive_result)
    assert outcome.result.outcome is ContractProofOutcome.INCONCLUSIVE
    connection = another.authoritative_cache._connect()
    try:
        row = connection.execute(
            "SELECT created_at_ms, expires_at_ms FROM proof_flight_outcomes"
        ).fetchone()
        assert row["expires_at_ms"] - row["created_at_ms"] == 3_000
        assert connection.execute(
            "SELECT COUNT(*) FROM proof_cache_entries"
        ).fetchone()[0] == 0
    finally:
        connection.close()
    assert DEFAULT_NEGATIVE_TTL_SECONDS <= 60


def test_entry_and_byte_retention_bounds_evict_oldest(tmp_path: Path) -> None:
    cache = TrustAwareProofCache(
        tmp_path,
        max_entries=2,
        max_bytes=10 * 1024 * 1024,
    )
    keys = []
    for index in range(3):
        obligation_id = f"obligation-{index}"
        key = _key(
            obligation=_identity(
                f"obligation-{index}", obligation_id
            )
        )
        receipt = _receipt(obligation_id=obligation_id)
        assert cache.put(key, receipt).stored
        keys.append(key)

    stats = cache.retention_stats()
    assert stats.entries == 2
    assert stats.encoded_bytes <= stats.max_bytes
    assert cache.lookup(keys[0]).status is CacheLookupStatus.MISS
    assert cache.lookup(keys[-1]).hit


def test_retention_configuration_rejects_unbounded_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        TrustAwareProofCache(tmp_path, max_entries=0)
    with pytest.raises(ValueError):
        TrustAwareProofCache(tmp_path, max_bytes=0)
    with pytest.raises(ValueError):
        TrustAwareProofCache(tmp_path, negative_ttl_seconds=0)
    with pytest.raises(ValueError):
        TrustAwareProofCache(
            tmp_path,
            negative_ttl_seconds=MAX_NEGATIVE_TTL_SECONDS + 1,
        )
