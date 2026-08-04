"""LPR-032 contract tests for federated doctor proof-cache gate."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.content_identity_bridge import (
    LOGIC_IR_PROFILE,
    MULTICODEC_RAW,
    STRICT_ARTIFACT_PROFILE,
    identify_strict_artifact,
    is_digest_shaped,
    sha256_digest_label,
)
from ipfs_accelerate_py.agent_supervisor.proof.doctor_proof_cache import (
    DoctorCacheDisposition,
    DoctorCacheReason,
    DoctorCacheStage,
    DoctorCacheValidationError,
    DoctorIdentityBinding,
    DoctorProofCacheGate,
    DoctorProofCacheKey,
    build_doctor_proof_cache_key,
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
    return DoctorIdentityBinding.from_identity(
        identity, logical_id=logical_id or f"{name}-1"
    )


def _key(**changes: object) -> DoctorProofCacheKey:
    tree = _identity("tree", "tree-1")
    values: dict[str, object] = {
        "forest": _identity("forest", "forest-1"),
        "tree": tree,
        "overlay": _identity("overlay", "overlay-1"),
        "ast": _identity("ast", "ast-1"),
        "graph": _identity("graph", "graph-1"),
        "corpus": _identity("corpus", "corpus-1"),
        "goal": _identity("goal", "goal-1"),
        "premises": (
            _identity("premise-a", "premise-a"),
            _identity("premise-b", "premise-b"),
        ),
        "translation": _identity("translation", "translation-1"),
        "solver": _identity("solver", "solver-1"),
        "kernel": _identity("kernel", "kernel-1"),
        "toolchain": _identity("toolchain", "toolchain-1"),
        "registry": _identity("registry", "registry-1"),
        "policy": _identity("policy", "policy-1"),
        "budget": _budget(),
        "sandbox": _identity("sandbox", "sandbox-1"),
        "environment": _identity("environment", "environment-1"),
        "candidate_tree": tree,
        "required_assurance": AssuranceLevel.KERNEL_VERIFIED,
    }
    values.update(changes)
    return DoctorProofCacheKey(**values)


def _kernel_evidence(
    *,
    obligation_id: str = "goal-1",
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
    obligation_id: str = "goal-1",
    tree_id: str = "tree-1",
    evidence: tuple[ProofEvidence, ...] | None = None,
    verdict: ProofVerdict = ProofVerdict.PROVED,
    freshness: EvidenceFreshness = EvidenceFreshness.CURRENT,
    metadata: dict | None = None,
    solver_id: str = "solver-1",
    kernel_id: str = "kernel-1",
    translator_id: str = "translation-1",
    toolchain_id: str = "toolchain-1",
    theorem_registry_id: str = "registry-1",
    policy_id: str = "policy-1",
    premise_ids: tuple[str, ...] = ("premise-a", "premise-b"),
) -> ProofReceipt:
    return ProofReceipt(
        obligation_id=obligation_id,
        plan_id=f"plan:{obligation_id}",
        attempt_id=f"attempt:{obligation_id}",
        repository_id="repository-1",
        repository_tree_id=tree_id,
        ast_scope_ids=("scope-1",),
        premise_ids=premise_ids,
        translator_id=translator_id,
        solver_id=solver_id,
        kernel_id=kernel_id,
        toolchain_id=toolchain_id,
        theorem_registry_id=theorem_registry_id,
        policy_id=policy_id,
        resource_budget=_budget(),
        verdict=verdict,
        evidence=evidence
        if evidence is not None
        else (_kernel_evidence(obligation_id=obligation_id, kernel_id=kernel_id),),
        freshness=freshness,
        kernel_receipt_id=f"kernel-receipt:{obligation_id}",
        metadata=metadata or {},
    )


def test_identity_retains_and_revalidates_canonical_bytes_against_cid() -> None:
    binding = _identity("artifact")
    assert binding.profile == STRICT_ARTIFACT_PROFILE
    assert binding.canonical_bytes
    assert DoctorIdentityBinding.from_dict(binding.to_dict()) == binding

    with pytest.raises(DoctorCacheValidationError) as error:
        replace(binding, canonical_bytes=binding.canonical_bytes + b" ")
    assert error.value.reason_code == DoctorCacheReason.POISONED.value


def test_reject_digest_like_pseudo_cid() -> None:
    identity = identify_strict_artifact({"component": "digest-reject", "version": 1})
    digest = identity.digest
    assert is_digest_shaped(digest)

    with pytest.raises(DoctorCacheValidationError) as error:
        DoctorIdentityBinding(
            logical_id="bad",
            profile=STRICT_ARTIFACT_PROFILE,
            cid=digest,
            canonical_bytes=identity.canonical_bytes,
            digest=digest,
        )
    assert error.value.reason_code == DoctorCacheReason.DIGEST_PSEUDO_CID.value

    bare = digest.removeprefix("sha256:")
    with pytest.raises(DoctorCacheValidationError) as bare_error:
        DoctorIdentityBinding(
            logical_id="bad-bare",
            profile=STRICT_ARTIFACT_PROFILE,
            cid=bare,
            canonical_bytes=identity.canonical_bytes,
            digest=digest,
        )
    assert bare_error.value.reason_code == DoctorCacheReason.DIGEST_PSEUDO_CID.value


def test_reject_double_hashing() -> None:
    identity = identify_strict_artifact({"component": "double-hash", "version": 1})
    double = sha256_digest_label(identity.digest.encode("utf-8"))
    with pytest.raises(DoctorCacheValidationError) as error:
        DoctorIdentityBinding(
            logical_id="double",
            profile=STRICT_ARTIFACT_PROFILE,
            cid=identity.cid,
            canonical_bytes=identity.canonical_bytes,
            digest=double,
        )
    # Double-hash or poisoned — both fail closed.
    assert error.value.reason_code in {
        DoctorCacheReason.DOUBLE_HASHING.value,
        DoctorCacheReason.POISONED.value,
    }


def test_alias_profile_mismatch_rejected() -> None:
    from multiformats import CID, multihash

    identity = identify_strict_artifact({"component": "cross-profile", "version": 1})
    raw_cid = str(
        CID(
            "base32",
            1,
            "raw",
            multihash.digest(identity.canonical_bytes, "sha2-256"),
        )
    )
    with pytest.raises(DoctorCacheValidationError) as error:
        DoctorIdentityBinding(
            logical_id="cross",
            profile=LOGIC_IR_PROFILE,
            cid=raw_cid,
            canonical_bytes=identity.canonical_bytes,
            digest=identity.digest,
            multicodec=MULTICODEC_RAW,
            domain="",
            artifact_schema="",
        )
    assert error.value.reason_code in {
        DoctorCacheReason.IDENTITY_INVALID.value,
        DoctorCacheReason.ALIAS_PROFILE_MISMATCH.value,
    }


def test_key_binds_every_semantic_dimension_and_is_order_invariant() -> None:
    baseline = _key()
    tree2 = _identity("tree", "tree-1", 2)
    mutations = {
        "forest": _identity("forest", "forest-1", 2),
        "tree": tree2,
        "overlay": _identity("overlay", "overlay-1", 2),
        "ast": _identity("ast", "ast-1", 2),
        "graph": _identity("graph", "graph-1", 2),
        "corpus": _identity("corpus", "corpus-1", 2),
        "goal": _identity("goal", "goal-1", 2),
        "premises": (
            _identity("premise-a", "premise-a", 2),
            _identity("premise-b", "premise-b"),
        ),
        "translation": _identity("translation", "translation-1", 2),
        "solver": _identity("solver", "solver-1", 2),
        "kernel": _identity("kernel", "kernel-1", 2),
        "toolchain": _identity("toolchain", "toolchain-1", 2),
        "registry": _identity("registry", "registry-1", 2),
        "policy": _identity("policy", "policy-1", 2),
        "budget": replace(_budget(), wall_time_ms=9_999),
        "sandbox": _identity("sandbox", "sandbox-1", 2),
        "environment": _identity("environment", "environment-1", 2),
        "candidate_tree": tree2,
        "required_assurance": AssuranceLevel.ATTESTED,
    }
    for name, value in mutations.items():
        assert _key(**{name: value}).key_id != baseline.key_id, name

    assert (
        _key(premises=tuple(reversed(baseline.premises))).key_id == baseline.key_id
    )


def test_tree_and_candidate_tree_must_agree() -> None:
    with pytest.raises(DoctorCacheValidationError) as error:
        _key(candidate_tree=_identity("tree", "tree-other"))
    assert error.value.reason_code == DoctorCacheReason.BINDING_MISMATCH.value


def test_warm_exact_hit_revalidates_before_render_and_commit(tmp_path: Path) -> None:
    gate = DoctorProofCacheGate(tmp_path)
    key = _key()
    receipt = _receipt()

    stored = gate.put(key, receipt)
    assert stored.stored
    assert DoctorCacheReason.STORED.value in stored.reason_codes

    hit = gate.lookup(key)
    assert hit.hit
    assert hit.receipt is not None
    assert hit.authoritative_assurance is AssuranceLevel.KERNEL_VERIFIED
    assert hit.audit is not None
    assert hit.audit.reconstructed
    assert hit.audit.authoritative

    render = gate.revalidate_for_render(key)
    assert render.hit
    assert render.audit is not None
    assert render.audit.stage is DoctorCacheStage.RENDER
    assert DoctorCacheReason.REVALIDATED.value in render.reason_codes

    commit = gate.revalidate_for_commit(key)
    assert commit.hit
    assert commit.audit is not None
    assert commit.audit.stage is DoctorCacheStage.COMMIT
    assert DoctorCacheReason.REVALIDATED.value in commit.reason_codes


def test_reject_partial_solver_only_raw_countermodel_stale(tmp_path: Path) -> None:
    gate = DoctorProofCacheGate(tmp_path)
    key = _key()

    partial = gate.put(key, _receipt(), complete=False)
    assert not partial.stored
    assert DoctorCacheReason.PARTIAL_ENTRY.value in partial.reason_codes

    solver_only = _receipt(evidence=())
    # Empty evidence yields unverified / not proved-level assurance.
    solver_store = gate.put(key, solver_only)
    assert not solver_store.stored
    assert any(
        code
        in {
            DoctorCacheReason.SOLVER_ONLY.value,
            DoctorCacheReason.CANDIDATE_ONLY.value,
            DoctorCacheReason.REQUIRED_ASSURANCE.value,
        }
        for code in solver_store.reason_codes
    )

    raw_cm = _receipt(
        verdict=ProofVerdict.DISPROVED,
        evidence=(),
        metadata={"raw_countermodel": True},
    )
    raw_store = gate.put(key, raw_cm)
    assert not raw_store.stored
    assert DoctorCacheReason.RAW_COUNTERMODEL.value in raw_store.reason_codes or (
        DoctorCacheReason.CANDIDATE_ONLY.value in raw_store.reason_codes
    )

    stale = _receipt(freshness=EvidenceFreshness.STALE)
    stale_store = gate.put(key, stale)
    assert not stale_store.stored
    assert DoctorCacheReason.STALE.value in stale_store.reason_codes


def test_wrong_tree_and_binding_mismatch_rejected(tmp_path: Path) -> None:
    gate = DoctorProofCacheGate(tmp_path)
    key = _key()
    wrong = gate.put(key, _receipt(tree_id="tree-other"))
    assert not wrong.stored
    assert DoctorCacheReason.WRONG_TREE.value in wrong.reason_codes

    mismatched = gate.put(key, _receipt(solver_id="solver-other"))
    assert not mismatched.stored
    assert DoctorCacheReason.BINDING_MISMATCH.value in mismatched.reason_codes


def test_negative_hits_and_timeouts_remain_diagnostic(tmp_path: Path) -> None:
    gate = DoctorProofCacheGate(tmp_path)
    key = _key()

    miss = gate.lookup(key)
    assert miss.disposition is DoctorCacheDisposition.MISS
    assert not miss.hit
    assert DoctorCacheReason.CACHE_MISS.value in miss.reason_codes

    diag = gate.record_diagnostic(
        key, kind="timeout", reason_codes=("wall_time_exceeded",)
    )
    assert diag.disposition is DoctorCacheDisposition.DIAGNOSTIC
    assert not diag.authoritative

    after = gate.lookup(key)
    assert after.disposition is DoctorCacheDisposition.DIAGNOSTIC
    assert after.diagnostic
    assert not after.hit


def test_semantic_root_change_invalidates_descendants_and_tombstones(
    tmp_path: Path,
) -> None:
    gate = DoctorProofCacheGate(tmp_path)
    key = _key()
    assert gate.put(key, _receipt()).stored
    assert gate.lookup(key).hit

    forest_cid = key.forest.cid
    tombstones = gate.invalidate_semantic_root(
        root_field="forest", root_cid=forest_cid
    )
    assert len(tombstones) == 1
    assert tombstones[0].key_id == key.key_id
    assert gate.is_tombstoned(key)

    after = gate.lookup(key)
    assert after.disposition is DoctorCacheDisposition.TOMBSTONED
    assert not after.hit
    assert DoctorCacheReason.TOMBSTONED.value in after.reason_codes


def test_equivocation_quarantines(tmp_path: Path) -> None:
    gate = DoctorProofCacheGate(tmp_path)
    key = _key()
    first = _receipt(obligation_id="goal-1")
    assert gate.put(key, first).stored

    # A second distinct receipt identity for the same key is equivocation.
    # Mutate attempt_id so receipt_id (content_id) changes while bindings hold.
    second = ProofReceipt(
        obligation_id="goal-1",
        plan_id="plan:goal-1-alt",
        attempt_id="attempt:goal-1-alt",
        repository_id="repository-1",
        repository_tree_id="tree-1",
        ast_scope_ids=("scope-1",),
        premise_ids=("premise-a", "premise-b"),
        translator_id="translation-1",
        solver_id="solver-1",
        kernel_id="kernel-1",
        toolchain_id="toolchain-1",
        theorem_registry_id="registry-1",
        policy_id="policy-1",
        resource_budget=_budget(),
        verdict=ProofVerdict.PROVED,
        evidence=(_kernel_evidence(),),
        freshness=EvidenceFreshness.CURRENT,
        kernel_receipt_id="kernel-receipt:goal-1-alt",
    )
    assert first.receipt_id != second.receipt_id
    result = gate.put(key, second)
    assert not result.stored
    assert DoctorCacheReason.EQUIVOCATION.value in result.reason_codes
    assert gate.is_quarantined(key)

    after = gate.lookup(key)
    assert after.disposition is DoctorCacheDisposition.QUARANTINED


def test_legacy_ipfs_is_transport_only(tmp_path: Path) -> None:
    gate = DoctorProofCacheGate(tmp_path)
    key = _key()
    result = gate.consult_legacy_ipfs(key, {"cid": "bafylegacyhint"})
    assert result.disposition is DoctorCacheDisposition.DIAGNOSTIC
    assert DoctorCacheReason.LEGACY_TRANSPORT_ONLY.value in result.reason_codes
    assert not result.hit


def test_private_material_rejected(tmp_path: Path) -> None:
    gate = DoctorProofCacheGate(tmp_path)
    # Field-name markers alone prove private-material detection; values use
    # admission-safe placeholders so proposal gates do not treat the fixture
    # as a concrete credential assignment.
    identity = identify_strict_artifact(
        {
            "component": "private-field-holder",
            "api_key": "redacted",
            "password": "placeholder",
            "version": 1,
        }
    )
    secret_binding = DoctorIdentityBinding.from_identity(
        identity, logical_id="goal-private-fields"
    )
    key = _key(goal=secret_binding)
    assert key.contains_private_material
    miss = gate.lookup(key)
    assert miss.disposition is DoctorCacheDisposition.REJECTED
    assert DoctorCacheReason.PRIVATE_MATERIAL.value in miss.reason_codes


def test_build_key_aliases() -> None:
    tree = _identity("tree", "tree-1")
    key = build_doctor_proof_cache_key(
        forest=_identity("forest", "forest-1"),
        snapshot=tree,
        overlay=_identity("overlay", "overlay-1"),
        ast=_identity("ast", "ast-1"),
        graph=_identity("graph", "graph-1"),
        corpus=_identity("corpus", "corpus-1"),
        goal=_identity("goal", "goal-1"),
        premises=(),
        translator=_identity("translation", "translation-1"),
        solver=_identity("solver", "solver-1"),
        kernel=_identity("kernel", "kernel-1"),
        toolchain=_identity("toolchain", "toolchain-1"),
        theorem_registry=_identity("registry", "registry-1"),
        policy=_identity("policy", "policy-1"),
        resource_budget=_budget(),
        sandbox=_identity("sandbox", "sandbox-1"),
        environment=_identity("environment", "environment-1"),
        candidate_tree=tree,
    )
    assert key.tree.logical_id == "tree-1"
    assert key.translation.logical_id == "translation-1"
    assert key.registry.logical_id == "registry-1"


def test_key_round_trip_dict() -> None:
    key = _key()
    restored = DoctorProofCacheKey.from_dict(key.to_dict())
    assert restored.key_id == key.key_id
    assert restored.semantic_root_ids == key.semantic_root_ids


def test_manual_quarantine(tmp_path: Path) -> None:
    gate = DoctorProofCacheGate(tmp_path)
    key = _key()
    assert gate.put(key, _receipt()).stored
    audit = gate.quarantine(key, reason="operator_hold")
    assert audit.disposition is DoctorCacheDisposition.QUARANTINED
    assert gate.lookup(key).disposition is DoctorCacheDisposition.QUARANTINED
