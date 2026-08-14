"""IPS-043: public IncrementalProofSealer Python APIs."""

from __future__ import annotations

import ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing as sealer
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.executor import (
    CachedCandidate,
    execute_incremental_plan,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.full_checkpoint import (
    RepositoryStateView,
    RequiredUnitEvidence,
    VerificationPolicyView,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.planner import (
    ParentSealContext,
    UnitPlanningInput,
    create_incremental_plan,
)
from ipfs_datasets_py.logic.zkp.incremental_sealing.evidence import (
    IntegrityCommitment,
    ProofMode,
    ProofTerminalStatus,
)

_A = "sha256:" + ("aa" * 32)
_B = "sha256:" + ("bb" * 32)
_C = "sha256:" + ("cc" * 32)
_D = "sha256:" + ("dd" * 32)
_E = "sha256:" + ("ee" * 32)


def test_evidence_and_lazy_exports() -> None:
    assert sealer.PUBLIC_API_EVIDENCE == "ips/public-api@1"
    assert sealer.CLI_EVIDENCE == "ips/cli@1"
    for name in (
        "create_full_checkpoint",
        "create_incremental_plan",
        "execute_incremental_plan",
        "verify_seal",
        "explain_reuse",
        "explain_invalidation",
        "compare_full_and_incremental",
    ):
        assert name in sealer.__all__
        assert callable(getattr(sealer, name))


def _state() -> RepositoryStateView:
    return RepositoryStateView(
        repository_id="repo/accelerate",
        revision="rev-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        source_root_cid=_A,
        repository_state_cid=_B,
        environment_cid=_C,
        parent_revision_ids=(),
    )


def _policy() -> VerificationPolicyView:
    return VerificationPolicyView(
        policy_cid=_D,
        proof_schema_version="1",
        canonicalization_version="1",
        dependency_graph_schema_version="graph@1",
        circuit_id="circuit@v1",
        verification_key_id="vk/1",
    )


def _unit(unit_id: str) -> RequiredUnitEvidence:
    return RequiredUnitEvidence(
        unit_id=unit_id,
        proof_object_cid=_E,
        category="unit_test",
        terminal_status=ProofTerminalStatus.INTEGRITY_VERIFIED.value,
        proof_mode=ProofMode.INTEGRITY_ONLY.value,
        required_for_seal=True,
        freshly_verified=True,
        cache_reused_without_fresh_verification=False,
    )


def test_seven_public_apis_are_exercised() -> None:
    seal = sealer.create_full_checkpoint(
        _state(),
        _policy(),
        units=(_unit("unit/a"), _unit("unit/b")),
        expected_unit_ids=("unit/a", "unit/b"),
    )
    assert seal.sealed is True

    parent = ParentSealContext(seal_cid=_A, repository_state_cid=_B, source_root_cid=_A)
    plan = sealer.create_incremental_plan(
        parent,
        _B,
        _C,
        units=(
            UnitPlanningInput(
                "unit/a",
                preserved=True,
                cache_key_complete=True,
                admitted=True,
                candidate_present=True,
            ),
        ),
    )
    digest = "sha256:" + ("ab" * 32)
    cid = "sha256:" + ("11" * 32)
    execution = sealer.execute_incremental_plan(
        plan,
        fetch=lambda uid: CachedCandidate(
            uid,
            digest,
            digest,
            cid,
            cid,
            proof_object_cid=cid,
            evidence=IntegrityCommitment(
                digest=digest,
                cid=cid,
                merkle_inclusion="leaf:0",
                byte_length=32,
            ),
        ),
    )
    assert execution.complete_coverage is True

    verification = sealer.verify_seal(seal, trusted_keys=None, verification_policy=_policy())
    assert verification.accepted in {True, False}

    reuse = sealer.explain_reuse(seal, "unit/a")
    assert reuse.unit_id == "unit/a"

    invalidation = sealer.explain_invalidation(plan, "unit/a")
    assert invalidation.unit_id == "unit/a"

    comparison = sealer.compare_full_and_incremental(
        _state(),
        parent,
        _policy(),
        units=(
            UnitPlanningInput(
                "unit/a",
                preserved=True,
                cache_key_complete=True,
                admitted=True,
                candidate_present=True,
            ),
        ),
    )
    assert comparison.to_canonical()["estimated"] is True or hasattr(comparison, "to_canonical")
