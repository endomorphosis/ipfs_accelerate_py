"""IPS-038: full checkpoint seal construction."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.full_checkpoint import (
    EVIDENCE_SUBSET,
    FOREST_CATEGORIES,
    GENESIS_PARENT_SEAL,
    SEAL_SCHEMA,
    CheckpointContext,
    FullCheckpointBuilder,
    FullCheckpointError,
    FullCheckpointReason,
    FullCheckpointSeal,
    RepositoryStateView,
    RequiredUnitEvidence,
    VerificationPolicyView,
    create_full_checkpoint,
)
from ipfs_datasets_py.logic.zkp.incremental_sealing.evidence import (
    ProofMode,
    ProofTerminalStatus,
    SealStatus,
)

_DIGEST_A = "sha256:" + ("aa" * 32)
_DIGEST_B = "sha256:" + ("bb" * 32)
_DIGEST_C = "sha256:" + ("cc" * 32)
_DIGEST_D = "sha256:" + ("dd" * 32)
_DIGEST_E = "sha256:" + ("ee" * 32)
_DIGEST_F = "sha256:" + ("ff" * 32)
_PARENT = "sha256:" + ("11" * 32)


def _state(**overrides: object) -> RepositoryStateView:
    payload = {
        "repository_id": "repo/accelerate",
        "revision": "rev-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "source_root_cid": _DIGEST_A,
        "repository_state_cid": _DIGEST_B,
        "environment_cid": _DIGEST_C,
        "parent_revision_ids": (),
    }
    payload.update(overrides)
    return RepositoryStateView(**payload)  # type: ignore[arg-type]


def _policy(**overrides: object) -> VerificationPolicyView:
    payload = {
        "policy_cid": _DIGEST_D,
        "proof_schema_version": "1",
        "canonicalization_version": "1",
        "dependency_graph_schema_version": "graph@1",
        "circuit_id": "circuit@v1",
        "verification_key_id": "vk/1",
    }
    payload.update(overrides)
    return VerificationPolicyView(**payload)  # type: ignore[arg-type]


def _unit(unit_id: str, **overrides: object) -> RequiredUnitEvidence:
    payload = {
        "unit_id": unit_id,
        "proof_object_cid": _DIGEST_E,
        "category": "unit_test",
        "terminal_status": ProofTerminalStatus.INTEGRITY_VERIFIED.value,
        "proof_mode": ProofMode.INTEGRITY_ONLY.value,
        "required_for_seal": True,
        "freshly_verified": True,
        "cache_reused_without_fresh_verification": False,
    }
    payload.update(overrides)
    return RequiredUnitEvidence(**payload)  # type: ignore[arg-type]


def _good_units() -> tuple[RequiredUnitEvidence, ...]:
    return (
        _unit("unit/a"),
        _unit(
            "unit/b",
            category="static_analysis",
            proof_object_cid=_DIGEST_F,
            terminal_status=ProofTerminalStatus.PROVED.value,
            proof_mode=ProofMode.DIRECT_EXECUTION_PROOF.value,
        ),
    )


def test_evidence_subset_and_schema() -> None:
    assert EVIDENCE_SUBSET == "ips/full-seal@1"
    assert SEAL_SCHEMA.endswith("full-checkpoint-seal@1")
    assert GENESIS_PARENT_SEAL == "ips.forest.genesis@1"
    assert "unit_test" in FOREST_CATEGORIES
    assert "release_invariant" in FOREST_CATEGORIES


def test_first_state_seals_only_after_all_required_units_and_roots_verify() -> None:
    seal = create_full_checkpoint(
        _state(),
        _policy(),
        units=_good_units(),
        expected_unit_ids=("unit/a", "unit/b"),
        parent_seal_cid=None,
        fallback_reasons=("first_state", "missing_parent"),
    )
    assert isinstance(seal, FullCheckpointSeal)
    assert seal.seal_status is SealStatus.SEALED_FULL
    assert seal.sealed is True
    assert seal.reason is FullCheckpointReason.SEALED
    assert seal.context is CheckpointContext.FIRST_STATE
    assert seal.is_genesis is True
    assert seal.parent_seal_cid == GENESIS_PARENT_SEAL
    assert seal.required_unit_ids == ("unit/a", "unit/b")
    assert seal.verified_unit_ids == ("unit/a", "unit/b")
    assert seal.rejected_unit_ids == ()
    assert seal.every_unit_freshly_verified is True
    assert seal.cache_reuse_hidden is False
    assert seal.manifest_root_cid.startswith("sha256:")
    assert seal.repository_proof_root.startswith("sha256:")
    assert seal.aggregation_root.startswith("sha256:")
    assert set(seal.category_roots) == set(FOREST_CATEGORIES)
    assert seal.repository_id == "repo/accelerate"
    assert seal.environment_cid == _DIGEST_C
    assert seal.policy_cid == _DIGEST_D
    assert seal.circuit_id == "circuit@v1"
    assert seal.verification_key_id == "vk/1"
    assert seal.proof_schema_version == "1"
    assert seal.canonicalization_version == "1"
    assert "first_state" in seal.fallback_reasons


def test_mandated_fallback_context_seals_only_when_complete() -> None:
    seal = create_full_checkpoint(
        _state(),
        _policy(),
        units=_good_units(),
        parent_seal_cid=_PARENT,
        fallback_reasons=("schema_change", "trust_policy_change"),
    )
    assert seal.seal_status is SealStatus.SEALED_FULL
    assert seal.context is CheckpointContext.MANDATED_FALLBACK
    assert seal.parent_seal_cid == _PARENT
    assert seal.is_genesis is False
    assert "schema_change" in seal.fallback_reasons


def test_historical_parent_relation_is_bound() -> None:
    seal = create_full_checkpoint(
        _state(parent_revision_ids=("parent-rev-a",)),
        _policy(),
        units=_good_units(),
        parent_seal_cid=_PARENT,
    )
    assert seal.seal_status is SealStatus.SEALED_FULL
    assert seal.context is CheckpointContext.HISTORICAL_PARENT
    assert seal.parent_seal_cid == _PARENT
    assert seal.parent_revision_ids == ("parent-rev-a",)


def test_simulated_required_unit_prevents_sealed_full() -> None:
    units = (
        _unit("unit/a"),
        _unit(
            "unit/sim",
            proof_mode=ProofMode.SIMULATED.value,
            terminal_status=ProofTerminalStatus.SIMULATED.value,
        ),
    )
    seal = create_full_checkpoint(_state(), _policy(), units=units)
    assert seal.sealed is False
    assert seal.seal_status is SealStatus.SIMULATED_ONLY
    assert seal.reason is FullCheckpointReason.SIMULATED_REQUIRED_UNIT
    assert "unit/sim" in seal.rejected_unit_ids
    assert seal.seal_status is not SealStatus.SEALED_FULL


@pytest.mark.parametrize(
    ("status", "expected_seal", "expected_reason"),
    [
        (
            ProofTerminalStatus.UNKNOWN.value,
            SealStatus.UNKNOWN,
            FullCheckpointReason.UNKNOWN_REQUIRED_UNIT,
        ),
        (
            ProofTerminalStatus.UNAVAILABLE.value,
            SealStatus.UNAVAILABLE,
            FullCheckpointReason.UNAVAILABLE_REQUIRED_UNIT,
        ),
        (
            ProofTerminalStatus.FAILED.value,
            SealStatus.PROOF_FAILED,
            FullCheckpointReason.FAILED_REQUIRED_UNIT,
        ),
        (
            ProofTerminalStatus.PROOF_FAILED.value,
            SealStatus.PROOF_FAILED,
            FullCheckpointReason.FAILED_REQUIRED_UNIT,
        ),
        (
            ProofTerminalStatus.TIMEOUT.value,
            SealStatus.TIMEOUT,
            FullCheckpointReason.TIMEOUT_REQUIRED_UNIT,
        ),
    ],
)
def test_blocking_required_statuses_prevent_sealed_full(
    status: str,
    expected_seal: SealStatus,
    expected_reason: FullCheckpointReason,
) -> None:
    units = (
        _unit("unit/a"),
        _unit("unit/bad", terminal_status=status),
    )
    seal = create_full_checkpoint(_state(), _policy(), units=units)
    assert seal.sealed is False
    assert seal.seal_status is expected_seal
    assert seal.reason is expected_reason
    assert seal.seal_status is not SealStatus.SEALED_FULL


def test_unverified_and_cache_reuse_without_fresh_verification_block_seal() -> None:
    unverified = create_full_checkpoint(
        _state(),
        _policy(),
        units=(_unit("unit/a", freshly_verified=False),),
    )
    assert unverified.sealed is False
    assert unverified.seal_status is SealStatus.VERIFICATION_FAILED
    assert unverified.reason is FullCheckpointReason.UNVERIFIED_REQUIRED_UNIT
    assert unverified.every_unit_freshly_verified is False

    cached = create_full_checkpoint(
        _state(),
        _policy(),
        units=(
            _unit(
                "unit/a",
                cache_reused_without_fresh_verification=True,
                freshly_verified=False,
            ),
        ),
    )
    assert cached.sealed is False
    assert cached.seal_status is SealStatus.INVALID_CACHE
    assert cached.reason is FullCheckpointReason.CACHE_REUSE_WITHOUT_VERIFICATION
    assert cached.cache_reuse_hidden is True


def test_incomplete_manifest_prevents_sealed_full() -> None:
    seal = create_full_checkpoint(
        _state(),
        _policy(),
        units=(_unit("unit/a"),),
        expected_unit_ids=("unit/a", "unit/b"),
    )
    assert seal.sealed is False
    assert seal.seal_status is SealStatus.INCOMPLETE_MANIFEST
    assert seal.reason is FullCheckpointReason.INCOMPLETE_MANIFEST
    assert "unit/b" in seal.rejected_unit_ids


def test_root_mismatch_prevents_sealed_full() -> None:
    seal = create_full_checkpoint(
        _state(),
        _policy(),
        units=_good_units(),
        expected_repository_proof_root="sha256:" + ("00" * 32),
    )
    assert seal.sealed is False
    assert seal.seal_status is SealStatus.VERIFICATION_FAILED
    assert seal.reason is FullCheckpointReason.ROOT_VERIFICATION_FAILED


def test_seal_is_deterministic_and_binds_bindings() -> None:
    first = create_full_checkpoint(
        _state(),
        _policy(),
        units=_good_units(),
        parent_seal_cid=None,
        fallback_reasons=("first_state",),
    )
    second = FullCheckpointBuilder().create(
        {
            "repository_id": "repo/accelerate",
            "revision": "rev-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "source_root_cid": _DIGEST_A,
            "repository_state_cid": _DIGEST_B,
            "environment_cid": _DIGEST_C,
        },
        {
            "policy_cid": _DIGEST_D,
            "circuit_id": "circuit@v1",
            "verification_key_id": "vk/1",
        },
        units=[unit.to_canonical() for unit in _good_units()],
        parent_seal_cid=None,
        fallback_reasons=("first_state",),
    )
    assert first.to_canonical() == second.to_canonical()
    assert first.seal_cid() == second.seal_cid()
    assert first.seal_cid().startswith("sha256:")

    # One-bit binding change changes the repository proof root.
    flipped = create_full_checkpoint(
        _state(environment_cid=_DIGEST_F),
        _policy(),
        units=_good_units(),
        parent_seal_cid=None,
        fallback_reasons=("first_state",),
    )
    assert flipped.repository_proof_root != first.repository_proof_root
    assert flipped.seal_cid() != first.seal_cid()


def test_non_required_units_do_not_block_seal() -> None:
    seal = create_full_checkpoint(
        _state(),
        _policy(),
        units=(
            _unit("unit/a"),
            _unit(
                "unit/optional",
                required_for_seal=False,
                terminal_status=ProofTerminalStatus.FAILED.value,
            ),
        ),
        expected_unit_ids=("unit/a",),
    )
    assert seal.sealed is True
    assert seal.seal_status is SealStatus.SEALED_FULL
    assert seal.required_unit_ids == ("unit/a",)


def test_empty_required_set_cannot_seal() -> None:
    seal = create_full_checkpoint(_state(), _policy(), units=())
    assert seal.sealed is False
    assert seal.seal_status is SealStatus.INCOMPLETE_MANIFEST
    assert seal.reason is FullCheckpointReason.EMPTY_REQUIRED_SET


def test_missing_policy_or_state_fields_fail_closed() -> None:
    with pytest.raises(FullCheckpointError, match="verification_policy"):
        create_full_checkpoint(_state(), None, units=_good_units())
    with pytest.raises(FullCheckpointError, match="repository_state"):
        create_full_checkpoint(
            {"repository_id": "repo/x"},
            _policy(),
            units=_good_units(),
        )


def test_sealed_full_invariant_rejects_hidden_cache_reuse() -> None:
    with pytest.raises(FullCheckpointError, match="cache reuse"):
        FullCheckpointSeal(
            schema=SEAL_SCHEMA,
            evidence_subset=EVIDENCE_SUBSET,
            seal_status=SealStatus.SEALED_FULL,
            reason=FullCheckpointReason.SEALED,
            context=CheckpointContext.FIRST_STATE,
            repository_id="repo/x",
            revision="rev",
            source_root_cid=_DIGEST_A,
            repository_state_cid=_DIGEST_B,
            environment_cid=_DIGEST_C,
            policy_cid=_DIGEST_D,
            proof_schema_version="1",
            canonicalization_version="1",
            dependency_graph_schema_version="graph@1",
            circuit_id="c",
            verification_key_id="vk",
            parent_seal_cid=GENESIS_PARENT_SEAL,
            parent_revision_ids=(),
            required_unit_ids=("unit/a",),
            verified_unit_ids=("unit/a",),
            rejected_unit_ids=(),
            manifest_root_cid=_DIGEST_E,
            category_roots={cat: _DIGEST_F for cat in FOREST_CATEGORIES},
            repository_proof_root=_DIGEST_A,
            aggregation_root=_DIGEST_B,
            fallback_reasons=("first_state",),
            every_unit_freshly_verified=True,
            cache_reuse_hidden=True,
            sealed=True,
        )
