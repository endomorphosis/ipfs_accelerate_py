"""IPS-047: full/delta lifecycle, branches, merge, rollback, and compaction.

Acceptance:

* each seal is accepted only on its declared lineage;
* complete current units survive merge/rollback/compaction;
* repeated histories yield deterministic roots and retained historical
  references.

Evidence subset: ``ips/seal-lifecycle-positive@1``.
"""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.checkpoint_policy import (
    CheckpointMode,
    CheckpointTrigger,
    decide_checkpoint,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.compaction import (
    CompactionReason,
    RetentionPolicy,
    SealChainEntry,
    compact_seal_chain,
    entry_from_seal,
    verify_seal_chain,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.delta_seal import (
    TRANSITION_SCHEMA,
    DeltaSeal,
    DeltaSealReason,
    DeltaTransitionStatement,
    DeltaUnitEvidence,
    DiffCommitmentView,
    ParentSealView,
    build_delta_seal,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.full_checkpoint import (
    FOREST_CATEGORIES,
    GENESIS_PARENT_SEAL,
    CheckpointContext,
    FullCheckpointReason,
    FullCheckpointSeal,
    RepositoryStateView,
    RequiredUnitEvidence,
    VerificationPolicyView,
    create_full_checkpoint,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.sealer import (
    IncrementalProofSealer,
)
from ipfs_datasets_py.logic.zkp.incremental_sealing.evidence import (
    ProofMode,
    ProofTerminalStatus,
    SealStatus,
)

EVIDENCE_SUBSET = "ips/seal-lifecycle-positive@1"

_DIGEST_A = "sha256:" + ("aa" * 32)
_DIGEST_B = "sha256:" + ("bb" * 32)
_DIGEST_C = "sha256:" + ("cc" * 32)
_DIGEST_D = "sha256:" + ("dd" * 32)
_DIGEST_E = "sha256:" + ("ee" * 32)
_DIGEST_F = "sha256:" + ("ff" * 32)
_DIGEST_1 = "sha256:" + ("11" * 32)
_DIGEST_2 = "sha256:" + ("22" * 32)
_DIGEST_3 = "sha256:" + ("33" * 32)
_DIGEST_4 = "sha256:" + ("44" * 32)
_DIGEST_5 = "sha256:" + ("55" * 32)
_DIGEST_6 = "sha256:" + ("66" * 32)
_DIGEST_7 = "sha256:" + ("77" * 32)
_DIGEST_8 = "sha256:" + ("88" * 32)
_DIGEST_9 = "sha256:" + ("99" * 32)
_VK = "vk/lifecycle-1"
_POLICY = _DIGEST_D
_TRUSTED = (_VK, "n/a")
_REPO = "repo/accelerate"
_REV_GENESIS = "rev-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
_REV_DELTA = "rev-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
_REV_BRANCH_A = "rev-cccccccccccccccccccccccccccccccccccccccc"
_REV_BRANCH_B = "rev-dddddddddddddddddddddddddddddddddddddddd"
_REV_MERGE = "rev-eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
_REV_ROLLBACK = "rev-ffffffffffffffffffffffffffffffffffffffff"
_REV_CHECKPOINT = "rev-0123456789abcdef0123456789abcdef01234567"
_PROOF_A = _DIGEST_E
_PROOF_B = _DIGEST_F
_PROOF_B_V2 = _DIGEST_7
_PROOF_A_V2 = _DIGEST_4
_PROOF_B_V3 = _DIGEST_5
_PROOF_MERGE = _DIGEST_8
_PROOF_ROLLBACK = _DIGEST_9


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _state(**overrides: object) -> RepositoryStateView:
    payload: dict[str, object] = {
        "repository_id": _REPO,
        "revision": _REV_GENESIS,
        "source_root_cid": _DIGEST_A,
        "repository_state_cid": _DIGEST_B,
        "environment_cid": _DIGEST_C,
        "parent_revision_ids": (),
    }
    payload.update(overrides)
    return RepositoryStateView(**payload)  # type: ignore[arg-type]


def _policy(**overrides: object) -> VerificationPolicyView:
    payload: dict[str, object] = {
        "policy_cid": _POLICY,
        "proof_schema_version": "1",
        "canonicalization_version": "1",
        "dependency_graph_schema_version": "graph@1",
        "circuit_id": "circuit@v1",
        "verification_key_id": _VK,
    }
    payload.update(overrides)
    return VerificationPolicyView(**payload)  # type: ignore[arg-type]


def _full_unit(unit_id: str, **overrides: object) -> RequiredUnitEvidence:
    payload: dict[str, object] = {
        "unit_id": unit_id,
        "proof_object_cid": _PROOF_A if unit_id == "unit/a" else _PROOF_B,
        "category": "unit_test" if unit_id == "unit/a" else "static_analysis",
        "terminal_status": (
            ProofTerminalStatus.INTEGRITY_VERIFIED.value
            if unit_id == "unit/a"
            else ProofTerminalStatus.PROVED.value
        ),
        "proof_mode": (
            ProofMode.INTEGRITY_ONLY.value
            if unit_id == "unit/a"
            else ProofMode.DIRECT_EXECUTION_PROOF.value
        ),
        "required_for_seal": True,
        "freshly_verified": True,
        "cache_reused_without_fresh_verification": False,
        "circuit_id": "circuit@v1",
        "verification_key_id": _VK,
    }
    payload.update(overrides)
    return RequiredUnitEvidence(**payload)  # type: ignore[arg-type]


def _full_units(
    *,
    proof_a: str = _PROOF_A,
    proof_b: str = _PROOF_B,
) -> tuple[RequiredUnitEvidence, ...]:
    return (
        _full_unit("unit/a", proof_object_cid=proof_a),
        _full_unit("unit/b", proof_object_cid=proof_b),
    )


def _genesis_full(**overrides: object) -> FullCheckpointSeal:
    return create_full_checkpoint(
        _state(),
        _policy(),
        units=_full_units(),
        expected_unit_ids=("unit/a", "unit/b"),
        parent_seal_cid=None,
        fallback_reasons=("first_state", "missing_parent"),
        **overrides,
    )


def _diff(
    *,
    commitment: str = _DIGEST_6,
    paths: tuple[str, ...] = ("src/module.py",),
) -> DiffCommitmentView:
    return DiffCommitmentView(
        diff_algorithm="exact_artifact_set@1",
        changed_artifact_commitment=commitment,
        complete=True,
        changed_paths=paths,
    )


def _parent_view(
    seal: FullCheckpointSeal | DeltaSeal,
    *,
    branch_id: str = "main",
    unit_proof_cids: dict[str, str] | None = None,
    logical_epoch: int = 1,
) -> ParentSealView:
    if isinstance(seal, FullCheckpointSeal):
        status = SealStatus.SEALED_FULL.value
        manifest = seal.manifest_root_cid
        forest = seal.repository_proof_root
        aggregation = seal.aggregation_root
        proofs = unit_proof_cids or {"unit/a": _PROOF_A, "unit/b": _PROOF_B}
        parent_revs = seal.parent_revision_ids
    else:
        status = SealStatus.SEALED_INCREMENTAL.value
        manifest = seal.new_manifest_root_cid
        forest = seal.new_forest_root_cid
        aggregation = seal.new_aggregation_root
        proofs = unit_proof_cids or {
            uid: (
                _PROOF_A
                if uid == "unit/a"
                else _PROOF_B_V2
            )
            for uid in seal.required_unit_ids
        }
        parent_revs = seal.parent_revision_ids
    return ParentSealView(
        seal_cid=seal.seal_cid(),
        accepted=True,
        seal_status=status,
        repository_id=seal.repository_id,
        branch_id=branch_id,
        revision=seal.revision,
        source_root_cid=seal.source_root_cid,
        repository_state_cid=seal.repository_state_cid,
        environment_cid=seal.environment_cid,
        policy_cid=seal.policy_cid,
        manifest_root_cid=manifest,
        forest_root_cid=forest,
        aggregation_root=aggregation,
        required_unit_ids=seal.required_unit_ids,
        unit_proof_cids=proofs,
        parent_revision_ids=parent_revs,
        proof_schema_version=seal.proof_schema_version,
        canonicalization_version=seal.canonicalization_version,
        dependency_graph_schema_version=seal.dependency_graph_schema_version,
        circuit_id=seal.circuit_id,
        verification_key_id=seal.verification_key_id,
        logical_epoch=logical_epoch,
    )


def _delta_unit(unit_id: str, disposition: str, **overrides: object) -> DeltaUnitEvidence:
    default_proof = _PROOF_A if unit_id == "unit/a" else _PROOF_B_V2
    parent_proof = _PROOF_A if unit_id == "unit/a" else _PROOF_B
    payload: dict[str, object] = {
        "unit_id": unit_id,
        "disposition": disposition,
        "proof_object_cid": "" if disposition == "remove" else default_proof,
        "category": "unit_test" if unit_id == "unit/a" else "static_analysis",
        "terminal_status": (
            ProofTerminalStatus.INTEGRITY_VERIFIED.value
            if unit_id == "unit/a"
            else ProofTerminalStatus.PROVED.value
        ),
        "proof_mode": (
            ProofMode.INTEGRITY_ONLY.value
            if unit_id == "unit/a"
            else ProofMode.DIRECT_EXECUTION_PROOF.value
        ),
        "required_for_seal": True,
        "cache_key_complete": True,
        "cache_key_unchanged": disposition == "reuse",
        "freshly_verified": True,
        "newly_admitted": disposition in {"replace", "add"},
        "removal_authorized": disposition == "remove",
        "parent_proof_object_cid": parent_proof if disposition != "add" else "",
        "stale": False,
    }
    payload.update(overrides)
    return DeltaUnitEvidence(**payload)  # type: ignore[arg-type]


def _build_delta(
    parent: ParentSealView,
    *,
    revision: str,
    source_root_cid: str,
    repository_state_cid: str,
    parent_revision_ids: tuple[str, ...],
    units: tuple[DeltaUnitEvidence, ...],
    branch_id: str | None = None,
    logical_epoch: int | None = None,
    diff_commitment: str = _DIGEST_6,
    changed_paths: tuple[str, ...] = ("src/module.py",),
    expected_manifest: tuple[str, ...] | None = None,
    expected_survivors: tuple[str, ...] | None = None,
    transition_id: str = "",
) -> DeltaSeal:
    branch = branch_id if branch_id is not None else parent.branch_id
    epoch = logical_epoch if logical_epoch is not None else parent.logical_epoch + 1
    present = tuple(
        unit.unit_id for unit in units if unit.disposition != "remove"
    )
    manifest = expected_manifest if expected_manifest is not None else present
    survivors = (
        expected_survivors
        if expected_survivors is not None
        else tuple(uid for uid in parent.required_unit_ids if uid in set(present))
    )
    state = _state(
        revision=revision,
        source_root_cid=source_root_cid,
        repository_state_cid=repository_state_cid,
        parent_revision_ids=parent_revision_ids,
    )
    transition = DeltaTransitionStatement(
        schema=TRANSITION_SCHEMA,
        parent_seal_cid=parent.seal_cid,
        branch_id=branch,
        old_source_root_cid=parent.source_root_cid,
        old_repository_state_cid=parent.repository_state_cid,
        old_manifest_root_cid=parent.manifest_root_cid,
        old_forest_root_cid=parent.forest_root_cid,
        old_aggregation_root=parent.aggregation_root,
        new_source_root_cid=source_root_cid,
        new_repository_state_cid=repository_state_cid,
        new_revision=revision,
        parent_revision_ids=parent_revision_ids,
        diff=_diff(commitment=diff_commitment, paths=changed_paths),
        expected_manifest_unit_ids=manifest,
        expected_surviving_leaf_ids=survivors,
        forest_rebuilt=True,
        aggregation_rebuilt=True,
        logical_epoch=epoch,
        transition_id=transition_id,
    )
    return build_delta_seal(parent, state, _policy(), transition, units=units)


def _localized_delta_from_full(
    full: FullCheckpointSeal,
    *,
    branch_id: str = "main",
    revision: str = _REV_DELTA,
    source_root_cid: str = _DIGEST_1,
    repository_state_cid: str = _DIGEST_2,
    proof_b: str = _PROOF_B_V2,
    logical_epoch: int = 2,
    transition_id: str = "transition/localized",
) -> DeltaSeal:
    parent = _parent_view(full, branch_id=branch_id, logical_epoch=1)
    return _build_delta(
        parent,
        revision=revision,
        source_root_cid=source_root_cid,
        repository_state_cid=repository_state_cid,
        parent_revision_ids=(full.revision,),
        branch_id=branch_id,
        logical_epoch=logical_epoch,
        transition_id=transition_id,
        units=(
            _delta_unit(
                "unit/a",
                "reuse",
                proof_object_cid=_PROOF_A,
                parent_proof_object_cid=_PROOF_A,
                cache_key_unchanged=True,
            ),
            _delta_unit(
                "unit/b",
                "replace",
                proof_object_cid=proof_b,
                parent_proof_object_cid=_PROOF_B,
                newly_admitted=True,
                cache_key_unchanged=False,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Evidence / schema surface
# ---------------------------------------------------------------------------


def test_lifecycle_evidence_subset() -> None:
    assert EVIDENCE_SUBSET == "ips/seal-lifecycle-positive@1"
    assert GENESIS_PARENT_SEAL == "ips.forest.genesis@1"
    assert "unit_test" in FOREST_CATEGORIES


# ---------------------------------------------------------------------------
# First full seal and localized deltas
# ---------------------------------------------------------------------------


def test_first_full_seal_accepts_complete_required_set() -> None:
    seal = _genesis_full()
    assert seal.sealed is True
    assert seal.seal_status is SealStatus.SEALED_FULL
    assert seal.reason is FullCheckpointReason.SEALED
    assert seal.context is CheckpointContext.FIRST_STATE
    assert seal.is_genesis is True
    assert seal.parent_seal_cid == GENESIS_PARENT_SEAL
    assert set(seal.required_unit_ids) == {"unit/a", "unit/b"}
    assert set(seal.verified_unit_ids) == {"unit/a", "unit/b"}
    assert seal.rejected_unit_ids == ()
    assert seal.manifest_root_cid.startswith("sha256:")
    assert seal.repository_proof_root.startswith("sha256:")
    assert seal.aggregation_root.startswith("sha256:")
    assert set(seal.category_roots) == set(FOREST_CATEGORIES)


def test_localized_delta_accepts_on_declared_parent_lineage() -> None:
    full = _genesis_full()
    delta = _localized_delta_from_full(full)
    assert isinstance(delta, DeltaSeal)
    assert delta.sealed is True
    assert delta.seal_status is SealStatus.SEALED_INCREMENTAL
    assert delta.reason is DeltaSealReason.SEALED
    assert delta.parent_seal_cid == full.seal_cid()
    assert delta.branch_id == "main"
    assert delta.reused_unit_ids == ("unit/a",)
    assert delta.replaced_unit_ids == ("unit/b",)
    assert set(delta.required_unit_ids) == {"unit/a", "unit/b"}
    assert delta.logical_epoch == 2
    assert delta.all_invariants_passed() is True
    assert delta.new_forest_root_cid != full.repository_proof_root
    assert delta.new_aggregation_root != full.aggregation_root


def test_seal_accepted_only_on_declared_lineage() -> None:
    full = _genesis_full()
    good = _localized_delta_from_full(full)
    assert good.sealed is True
    assert good.parent_seal_cid == full.seal_cid()

    mismatched = build_delta_seal(
        _parent_view(full),
        _state(
            revision=_REV_DELTA,
            source_root_cid=_DIGEST_1,
            repository_state_cid=_DIGEST_2,
            parent_revision_ids=(full.revision,),
        ),
        _policy(),
        DeltaTransitionStatement(
            schema=TRANSITION_SCHEMA,
            parent_seal_cid="sha256:" + ("00" * 32),
            branch_id="main",
            old_source_root_cid=full.source_root_cid,
            old_repository_state_cid=full.repository_state_cid,
            old_manifest_root_cid=full.manifest_root_cid,
            old_forest_root_cid=full.repository_proof_root,
            old_aggregation_root=full.aggregation_root,
            new_source_root_cid=_DIGEST_1,
            new_repository_state_cid=_DIGEST_2,
            new_revision=_REV_DELTA,
            parent_revision_ids=(full.revision,),
            diff=_diff(),
            expected_manifest_unit_ids=("unit/a", "unit/b"),
            expected_surviving_leaf_ids=("unit/a", "unit/b"),
            forest_rebuilt=True,
            aggregation_rebuilt=True,
            logical_epoch=2,
            transition_id="transition/lineage-reject",
        ),
        units=(
            _delta_unit(
                "unit/a",
                "reuse",
                proof_object_cid=_PROOF_A,
                parent_proof_object_cid=_PROOF_A,
            ),
            _delta_unit(
                "unit/b",
                "replace",
                proof_object_cid=_PROOF_B_V2,
                parent_proof_object_cid=_PROOF_B,
                newly_admitted=True,
                cache_key_unchanged=False,
            ),
        ),
    )
    assert mismatched.sealed is False
    assert mismatched.reason is DeltaSealReason.WRONG_PARENT
    assert mismatched.seal_status is SealStatus.STALE_PARENT


def test_wrong_branch_lineage_rejects() -> None:
    full = _genesis_full()
    parent = _parent_view(full, branch_id="main")
    seal = _build_delta(
        parent,
        revision=_REV_DELTA,
        source_root_cid=_DIGEST_1,
        repository_state_cid=_DIGEST_2,
        parent_revision_ids=(full.revision,),
        branch_id="feature/other",
        units=(
            _delta_unit(
                "unit/a",
                "reuse",
                proof_object_cid=_PROOF_A,
                parent_proof_object_cid=_PROOF_A,
            ),
            _delta_unit(
                "unit/b",
                "replace",
                proof_object_cid=_PROOF_B_V2,
                parent_proof_object_cid=_PROOF_B,
                newly_admitted=True,
                cache_key_unchanged=False,
            ),
        ),
    )
    assert seal.sealed is False
    assert seal.reason is DeltaSealReason.WRONG_BRANCH


# ---------------------------------------------------------------------------
# Branches, concurrent nonstale tips, merge
# ---------------------------------------------------------------------------


def test_correct_parent_branches_accept_independently() -> None:
    genesis = _genesis_full()
    parent_a = _parent_view(genesis, branch_id="feature/a")
    parent_b = _parent_view(genesis, branch_id="feature/b")
    branch_a = _build_delta(
        parent_a,
        revision=_REV_BRANCH_A,
        source_root_cid=_DIGEST_3,
        repository_state_cid=_DIGEST_4,
        parent_revision_ids=(genesis.revision,),
        branch_id="feature/a",
        transition_id="transition/a",
        units=(
            _delta_unit(
                "unit/a",
                "replace",
                proof_object_cid=_PROOF_A_V2,
                parent_proof_object_cid=_PROOF_A,
                newly_admitted=True,
                cache_key_unchanged=False,
            ),
            _delta_unit(
                "unit/b",
                "reuse",
                proof_object_cid=_PROOF_B,
                parent_proof_object_cid=_PROOF_B,
                cache_key_unchanged=True,
                category="static_analysis",
                terminal_status=ProofTerminalStatus.PROVED.value,
                proof_mode=ProofMode.DIRECT_EXECUTION_PROOF.value,
            ),
        ),
    )
    branch_b = _build_delta(
        parent_b,
        revision=_REV_BRANCH_B,
        source_root_cid=_DIGEST_5,
        repository_state_cid=_DIGEST_6,
        parent_revision_ids=(genesis.revision,),
        branch_id="feature/b",
        transition_id="transition/b",
        units=(
            _delta_unit(
                "unit/a",
                "reuse",
                proof_object_cid=_PROOF_A,
                parent_proof_object_cid=_PROOF_A,
                cache_key_unchanged=True,
            ),
            _delta_unit(
                "unit/b",
                "replace",
                proof_object_cid=_PROOF_B_V3,
                parent_proof_object_cid=_PROOF_B,
                newly_admitted=True,
                cache_key_unchanged=False,
            ),
        ),
    )
    assert branch_a.sealed is True
    assert branch_b.sealed is True
    assert branch_a.parent_seal_cid == genesis.seal_cid()
    assert branch_b.parent_seal_cid == genesis.seal_cid()
    assert branch_a.branch_id == "feature/a"
    assert branch_b.branch_id == "feature/b"
    assert branch_a.seal_cid() != branch_b.seal_cid()
    # Cross-lineage: branch B tip cannot be sealed against branch A's parent view.
    cross = _build_delta(
        _parent_view(
            branch_a,
            branch_id="feature/a",
            unit_proof_cids={"unit/a": _PROOF_A_V2, "unit/b": _PROOF_B},
            logical_epoch=2,
        ),
        revision=_REV_MERGE,
        source_root_cid=_DIGEST_7,
        repository_state_cid=_DIGEST_8,
        parent_revision_ids=(branch_a.revision,),
        branch_id="feature/b",
        logical_epoch=3,
        units=(
            _delta_unit(
                "unit/a",
                "reuse",
                proof_object_cid=_PROOF_A_V2,
                parent_proof_object_cid=_PROOF_A_V2,
            ),
            _delta_unit(
                "unit/b",
                "replace",
                proof_object_cid=_PROOF_MERGE,
                parent_proof_object_cid=_PROOF_B,
                newly_admitted=True,
                cache_key_unchanged=False,
            ),
        ),
    )
    assert cross.sealed is False
    assert cross.reason is DeltaSealReason.WRONG_BRANCH


def test_merge_binds_all_repository_parents_and_keeps_complete_units() -> None:
    genesis = _genesis_full()
    branch_a = _build_delta(
        _parent_view(genesis, branch_id="feature/a"),
        revision=_REV_BRANCH_A,
        source_root_cid=_DIGEST_3,
        repository_state_cid=_DIGEST_4,
        parent_revision_ids=(genesis.revision,),
        branch_id="feature/a",
        transition_id="transition/merge-a",
        units=(
            _delta_unit(
                "unit/a",
                "replace",
                proof_object_cid=_PROOF_A_V2,
                parent_proof_object_cid=_PROOF_A,
                newly_admitted=True,
                cache_key_unchanged=False,
            ),
            _delta_unit(
                "unit/b",
                "reuse",
                proof_object_cid=_PROOF_B,
                parent_proof_object_cid=_PROOF_B,
                cache_key_unchanged=True,
                category="static_analysis",
                terminal_status=ProofTerminalStatus.PROVED.value,
                proof_mode=ProofMode.DIRECT_EXECUTION_PROOF.value,
            ),
        ),
    )
    branch_b = _build_delta(
        _parent_view(genesis, branch_id="feature/b"),
        revision=_REV_BRANCH_B,
        source_root_cid=_DIGEST_5,
        repository_state_cid=_DIGEST_6,
        parent_revision_ids=(genesis.revision,),
        branch_id="feature/b",
        transition_id="transition/merge-b",
        units=(
            _delta_unit(
                "unit/a",
                "reuse",
                proof_object_cid=_PROOF_A,
                parent_proof_object_cid=_PROOF_A,
                cache_key_unchanged=True,
            ),
            _delta_unit(
                "unit/b",
                "replace",
                proof_object_cid=_PROOF_B_V3,
                parent_proof_object_cid=_PROOF_B,
                newly_admitted=True,
                cache_key_unchanged=False,
            ),
        ),
    )
    assert branch_a.sealed and branch_b.sealed

    merge_parents = tuple(sorted((_REV_BRANCH_A, _REV_BRANCH_B)))
    merge = _build_delta(
        _parent_view(
            branch_a,
            branch_id="feature/a",
            unit_proof_cids={"unit/a": _PROOF_A_V2, "unit/b": _PROOF_B},
            logical_epoch=2,
        ),
        revision=_REV_MERGE,
        source_root_cid=_DIGEST_7,
        repository_state_cid=_DIGEST_8,
        parent_revision_ids=merge_parents,
        branch_id="feature/a",
        logical_epoch=3,
        transition_id="transition/merge",
        units=(
            _delta_unit(
                "unit/a",
                "reuse",
                proof_object_cid=_PROOF_A_V2,
                parent_proof_object_cid=_PROOF_A_V2,
            ),
            _delta_unit(
                "unit/b",
                "replace",
                proof_object_cid=_PROOF_MERGE,
                parent_proof_object_cid=_PROOF_B,
                newly_admitted=True,
                cache_key_unchanged=False,
            ),
        ),
    )
    assert merge.sealed is True
    assert merge.parent_seal_cid == branch_a.seal_cid()
    assert tuple(merge.parent_revision_ids) == merge_parents
    assert set(merge.required_unit_ids) == {"unit/a", "unit/b"}
    assert set(merge.verified_unit_ids) == {"unit/a", "unit/b"}
    assert merge.rejected_unit_ids == ()
    # Both branch tips remain distinct historical references.
    assert branch_a.seal_cid() != merge.seal_cid()
    assert branch_b.seal_cid() != merge.seal_cid()


def test_concurrent_nonstale_branches_publish_independently(tmp_path) -> None:
    sealer = IncrementalProofSealer(tmp_path)
    genesis = sealer.publish_full_checkpoint(
        _state(),
        _policy(),
        units=_full_units(),
        expected_unit_ids=("unit/a", "unit/b"),
        parent_seal_cid=GENESIS_PARENT_SEAL,
        fallback_reasons=("first_state",),
        branch_id="main",
    )
    assert genesis.published is True
    assert genesis.full_seal is not None

    # Each branch receives its own first seal without stomping the other.
    branch_a = sealer.publish_full_checkpoint(
        _state(
            revision=_REV_BRANCH_A,
            source_root_cid=_DIGEST_3,
            repository_state_cid=_DIGEST_4,
            parent_revision_ids=(genesis.full_seal.revision,),
        ),
        _policy(),
        units=_full_units(proof_a=_PROOF_A_V2, proof_b=_PROOF_B),
        expected_unit_ids=("unit/a", "unit/b"),
        parent_seal_cid=genesis.seal_cid,
        fallback_reasons=("historical_parent",),
        branch_id="feature/a",
    )
    branch_b = sealer.publish_full_checkpoint(
        _state(
            revision=_REV_BRANCH_B,
            source_root_cid=_DIGEST_5,
            repository_state_cid=_DIGEST_6,
            parent_revision_ids=(genesis.full_seal.revision,),
        ),
        _policy(),
        units=_full_units(proof_a=_PROOF_A, proof_b=_PROOF_B_V3),
        expected_unit_ids=("unit/a", "unit/b"),
        parent_seal_cid=genesis.seal_cid,
        fallback_reasons=("historical_parent",),
        branch_id="feature/b",
    )
    assert branch_a.published is True
    assert branch_b.published is True
    assert branch_a.seal_cid != branch_b.seal_cid

    tip_main = sealer.get_current_seal(_REPO, branch_id="main")
    tip_a = sealer.get_current_seal(_REPO, branch_id="feature/a")
    tip_b = sealer.get_current_seal(_REPO, branch_id="feature/b")
    assert tip_main is not None and tip_main.seal_cid == genesis.seal_cid
    assert tip_a is not None and tip_a.seal_cid == branch_a.seal_cid
    assert tip_b is not None and tip_b.seal_cid == branch_b.seal_cid
    assert set(branch_a.full_seal.required_unit_ids) == {"unit/a", "unit/b"}  # type: ignore[union-attr]
    assert set(branch_b.full_seal.required_unit_ids) == {"unit/a", "unit/b"}  # type: ignore[union-attr]
    sealer.close()


# ---------------------------------------------------------------------------
# Rollback: new parent-bound transition, no history rewrite / replay
# ---------------------------------------------------------------------------


def test_rollback_creates_new_parent_bound_transition_not_replay() -> None:
    full = _genesis_full()
    advanced = _localized_delta_from_full(full)
    assert advanced.sealed is True
    assert advanced.source_root_cid == _DIGEST_1

    # Rollback restores the genesis source root but advances epoch and parent.
    parent = _parent_view(
        advanced,
        unit_proof_cids={"unit/a": _PROOF_A, "unit/b": _PROOF_B_V2},
        logical_epoch=2,
    )
    rollback = _build_delta(
        parent,
        revision=_REV_ROLLBACK,
        source_root_cid=full.source_root_cid,  # source bytes may repeat
        repository_state_cid=_DIGEST_9,
        parent_revision_ids=(advanced.revision,),
        logical_epoch=3,
        diff_commitment=_DIGEST_2,
        changed_paths=("src/module.py",),
        transition_id="transition/rollback",
        units=(
            _delta_unit(
                "unit/a",
                "reuse",
                proof_object_cid=_PROOF_A,
                parent_proof_object_cid=_PROOF_A,
            ),
            _delta_unit(
                "unit/b",
                "replace",
                proof_object_cid=_PROOF_ROLLBACK,
                parent_proof_object_cid=_PROOF_B_V2,
                newly_admitted=True,
                cache_key_unchanged=False,
            ),
        ),
    )
    assert rollback.sealed is True
    assert rollback.parent_seal_cid == advanced.seal_cid()
    assert rollback.source_root_cid == full.source_root_cid
    assert rollback.logical_epoch == 3
    assert rollback.seal_cid() != full.seal_cid()
    assert rollback.seal_cid() != advanced.seal_cid()
    assert set(rollback.required_unit_ids) == {"unit/a", "unit/b"}

    # Replaying the advanced seal against a stale epoch is rejected.
    replay = _build_delta(
        parent,
        revision=_REV_ROLLBACK,
        source_root_cid=full.source_root_cid,
        repository_state_cid=_DIGEST_9,
        parent_revision_ids=(advanced.revision,),
        logical_epoch=2,  # does not advance past parent
        transition_id="transition/replay",
        units=(
            _delta_unit(
                "unit/a",
                "reuse",
                proof_object_cid=_PROOF_A,
                parent_proof_object_cid=_PROOF_A,
            ),
            _delta_unit(
                "unit/b",
                "replace",
                proof_object_cid=_PROOF_ROLLBACK,
                parent_proof_object_cid=_PROOF_B_V2,
                newly_admitted=True,
                cache_key_unchanged=False,
            ),
        ),
    )
    assert replay.sealed is False
    assert replay.reason is DeltaSealReason.REPLAY_REJECTED


# ---------------------------------------------------------------------------
# Periodic / forced full checkpoints
# ---------------------------------------------------------------------------


def test_periodic_and_forced_checkpoint_policy_require_full() -> None:
    incremental = decide_checkpoint(
        seals_since_last_full_checkpoint=3,
        delta_chain_depth=3,
        estimated_reuse_ratio_basis_points=8000,
        has_accepted_parent=True,
    )
    assert incremental.mode is CheckpointMode.INCREMENTAL
    assert incremental.require_full_checkpoint is False

    periodic = decide_checkpoint(
        seals_since_last_full_checkpoint=50,
        delta_chain_depth=1,
        estimated_reuse_ratio_basis_points=9000,
    )
    assert periodic.require_full_checkpoint is True
    assert CheckpointTrigger.PERIODIC_CADENCE.value in periodic.reasons

    forced = decide_checkpoint(
        seals_since_last_full_checkpoint=1,
        force_full_checkpoint=True,
        estimated_reuse_ratio_basis_points=9000,
    )
    assert forced.require_full_checkpoint is True
    assert CheckpointTrigger.EXPLICIT_FORCE.value in forced.reasons

    release = decide_checkpoint(
        seals_since_last_full_checkpoint=1,
        is_release_tag=True,
        estimated_reuse_ratio_basis_points=9000,
    )
    assert release.require_full_checkpoint is True
    assert CheckpointTrigger.RELEASE_TAG.value in release.reasons


def test_forced_full_checkpoint_after_delta_chain_is_parent_bound() -> None:
    full = _genesis_full()
    delta = _localized_delta_from_full(full)
    checkpoint = create_full_checkpoint(
        _state(
            revision=_REV_CHECKPOINT,
            source_root_cid=delta.source_root_cid,
            repository_state_cid=delta.repository_state_cid,
            parent_revision_ids=(delta.revision,),
        ),
        _policy(),
        units=_full_units(proof_a=_PROOF_A, proof_b=_PROOF_B_V2),
        expected_unit_ids=("unit/a", "unit/b"),
        parent_seal_cid=delta.seal_cid(),
        fallback_reasons=("periodic_cadence", "explicit_force"),
    )
    assert checkpoint.sealed is True
    assert checkpoint.seal_status is SealStatus.SEALED_FULL
    assert checkpoint.parent_seal_cid == delta.seal_cid()
    assert checkpoint.is_genesis is False
    assert set(checkpoint.required_unit_ids) == {"unit/a", "unit/b"}
    assert "periodic_cadence" in checkpoint.fallback_reasons


# ---------------------------------------------------------------------------
# Chain verification, retention, compaction
# ---------------------------------------------------------------------------


def test_chain_verification_accepts_linked_history() -> None:
    full = _genesis_full()
    delta = _localized_delta_from_full(full)
    ok, cids, message, _details = verify_seal_chain(
        (full, delta),
        current_seal_cid=delta.seal_cid(),
    )
    assert ok is True
    assert cids == (full.seal_cid(), delta.seal_cid())
    assert "verified" in message

    broken_payload = entry_from_seal(delta).to_canonical()
    broken_payload["parent_seal_cid"] = "sha256:" + ("00" * 32)
    broken_entry = SealChainEntry(
        seal_cid=str(broken_payload["seal_cid"]),
        parent_seal_cid=str(broken_payload["parent_seal_cid"]),
        seal_status=str(broken_payload["seal_status"]),
        seal_kind=str(broken_payload["seal_kind"]),
        accepted=True,
    )
    broken_ok, _, _, _ = verify_seal_chain(
        (entry_from_seal(full), broken_entry),
        current_seal_cid=delta.seal_cid(),
    )
    assert broken_ok is False


def test_compaction_retains_historical_references_and_complete_units() -> None:
    full = _genesis_full()
    delta = _localized_delta_from_full(full)
    units = _full_units(proof_a=_PROOF_A, proof_b=_PROOF_B_V2)
    retention = RetentionPolicy(
        required_historical_seal_cids=(full.seal_cid(), delta.seal_cid()),
        required_evidence_cids=(_PROOF_A, _PROOF_B_V2),
        retain_entire_verified_chain=True,
    )
    outcome = compact_seal_chain(
        delta,
        retention,
        _policy(),
        seal_chain=(full, delta),
        units=units,
        expected_unit_ids=("unit/a", "unit/b"),
        repository_state=_state(
            revision=delta.revision,
            source_root_cid=delta.source_root_cid,
            repository_state_cid=delta.repository_state_cid,
            parent_revision_ids=delta.parent_revision_ids,
        ),
        trusted_keys=_TRUSTED,
        verify_current_cryptographically=True,
    )
    assert outcome.sealed is True
    assert outcome.reason is CompactionReason.COMPACTED
    assert outcome.chain_verified is True
    assert outcome.manifest_verified is True
    assert outcome.units_verified is True
    assert outcome.forest_verified is True
    assert outcome.retention_satisfied is True
    assert outcome.seal is not None
    assert outcome.seal.sealed is True
    assert outcome.seal.seal_status is SealStatus.SEALED_FULL
    assert outcome.seal.parent_seal_cid == delta.seal_cid()
    assert set(outcome.seal.required_unit_ids) == {"unit/a", "unit/b"}
    assert full.seal_cid() in outcome.retained_historical_seal_cids
    assert delta.seal_cid() in outcome.retained_historical_seal_cids
    assert outcome.compacted_seal_cid in outcome.retained_historical_seal_cids
    assert _PROOF_A in outcome.retained_evidence_cids
    assert _PROOF_B_V2 in outcome.retained_evidence_cids
    assert outcome.to_canonical()["history_rewritten"] is False
    assert outcome.to_canonical()["evidence_silently_deleted"] is False


def test_complete_units_survive_merge_rollback_and_compaction() -> None:
    full = _genesis_full()
    advanced = _localized_delta_from_full(full)

    # Merge-shaped multi-parent transition on main lineage.
    merge_parents = tuple(sorted((full.revision, advanced.revision)))
    merge = _build_delta(
        _parent_view(
            advanced,
            unit_proof_cids={"unit/a": _PROOF_A, "unit/b": _PROOF_B_V2},
            logical_epoch=2,
        ),
        revision=_REV_MERGE,
        source_root_cid=_DIGEST_7,
        repository_state_cid=_DIGEST_8,
        parent_revision_ids=merge_parents,
        logical_epoch=3,
        transition_id="transition/survive-merge",
        units=(
            _delta_unit(
                "unit/a",
                "reuse",
                proof_object_cid=_PROOF_A,
                parent_proof_object_cid=_PROOF_A,
            ),
            _delta_unit(
                "unit/b",
                "replace",
                proof_object_cid=_PROOF_MERGE,
                parent_proof_object_cid=_PROOF_B_V2,
                newly_admitted=True,
                cache_key_unchanged=False,
            ),
        ),
    )
    assert merge.sealed is True
    assert set(merge.required_unit_ids) == {"unit/a", "unit/b"}

    rollback = _build_delta(
        _parent_view(
            merge,
            unit_proof_cids={"unit/a": _PROOF_A, "unit/b": _PROOF_MERGE},
            logical_epoch=3,
        ),
        revision=_REV_ROLLBACK,
        source_root_cid=full.source_root_cid,
        repository_state_cid=_DIGEST_9,
        parent_revision_ids=(merge.revision,),
        logical_epoch=4,
        transition_id="transition/survive-rollback",
        units=(
            _delta_unit(
                "unit/a",
                "reuse",
                proof_object_cid=_PROOF_A,
                parent_proof_object_cid=_PROOF_A,
            ),
            _delta_unit(
                "unit/b",
                "replace",
                proof_object_cid=_PROOF_ROLLBACK,
                parent_proof_object_cid=_PROOF_MERGE,
                newly_admitted=True,
                cache_key_unchanged=False,
            ),
        ),
    )
    assert rollback.sealed is True
    assert set(rollback.required_unit_ids) == {"unit/a", "unit/b"}

    outcome = compact_seal_chain(
        rollback,
        RetentionPolicy(
            required_historical_seal_cids=(
                full.seal_cid(),
                advanced.seal_cid(),
                merge.seal_cid(),
                rollback.seal_cid(),
            ),
            required_evidence_cids=(_PROOF_A, _PROOF_ROLLBACK),
        ),
        _policy(),
        seal_chain=(full, advanced, merge, rollback),
        units=_full_units(proof_a=_PROOF_A, proof_b=_PROOF_ROLLBACK),
        expected_unit_ids=("unit/a", "unit/b"),
        repository_state=_state(
            revision=rollback.revision,
            source_root_cid=rollback.source_root_cid,
            repository_state_cid=rollback.repository_state_cid,
            parent_revision_ids=rollback.parent_revision_ids,
        ),
        trusted_keys=_TRUSTED,
        verify_current_cryptographically=True,
    )
    assert outcome.sealed is True
    assert outcome.seal is not None
    assert set(outcome.seal.required_unit_ids) == {"unit/a", "unit/b"}
    for cid in (
        full.seal_cid(),
        advanced.seal_cid(),
        merge.seal_cid(),
        rollback.seal_cid(),
    ):
        assert cid in outcome.retained_historical_seal_cids


# ---------------------------------------------------------------------------
# Deterministic repeated histories
# ---------------------------------------------------------------------------


def test_repeated_histories_yield_deterministic_roots() -> None:
    first = _run_positive_history()
    second = _run_positive_history()
    assert first["genesis_cid"] == second["genesis_cid"]
    assert first["delta_cid"] == second["delta_cid"]
    assert first["branch_a_cid"] == second["branch_a_cid"]
    assert first["branch_b_cid"] == second["branch_b_cid"]
    assert first["merge_cid"] == second["merge_cid"]
    assert first["checkpoint_cid"] == second["checkpoint_cid"]
    assert first["compaction_cid"] == second["compaction_cid"]
    assert first["retained"] == second["retained"]
    assert first["manifest_roots"] == second["manifest_roots"]
    assert first["forest_roots"] == second["forest_roots"]
    assert first["aggregation_roots"] == second["aggregation_roots"]


def _run_positive_history() -> dict[str, object]:
    genesis = _genesis_full()
    delta = _localized_delta_from_full(genesis)
    branch_a = _build_delta(
        _parent_view(genesis, branch_id="feature/a"),
        revision=_REV_BRANCH_A,
        source_root_cid=_DIGEST_3,
        repository_state_cid=_DIGEST_4,
        parent_revision_ids=(genesis.revision,),
        branch_id="feature/a",
        transition_id="transition/hist-a",
        units=(
            _delta_unit(
                "unit/a",
                "replace",
                proof_object_cid=_PROOF_A_V2,
                parent_proof_object_cid=_PROOF_A,
                newly_admitted=True,
                cache_key_unchanged=False,
            ),
            _delta_unit(
                "unit/b",
                "reuse",
                proof_object_cid=_PROOF_B,
                parent_proof_object_cid=_PROOF_B,
                cache_key_unchanged=True,
                category="static_analysis",
                terminal_status=ProofTerminalStatus.PROVED.value,
                proof_mode=ProofMode.DIRECT_EXECUTION_PROOF.value,
            ),
        ),
    )
    branch_b = _build_delta(
        _parent_view(genesis, branch_id="feature/b"),
        revision=_REV_BRANCH_B,
        source_root_cid=_DIGEST_5,
        repository_state_cid=_DIGEST_6,
        parent_revision_ids=(genesis.revision,),
        branch_id="feature/b",
        transition_id="transition/hist-b",
        units=(
            _delta_unit(
                "unit/a",
                "reuse",
                proof_object_cid=_PROOF_A,
                parent_proof_object_cid=_PROOF_A,
                cache_key_unchanged=True,
            ),
            _delta_unit(
                "unit/b",
                "replace",
                proof_object_cid=_PROOF_B_V3,
                parent_proof_object_cid=_PROOF_B,
                newly_admitted=True,
                cache_key_unchanged=False,
            ),
        ),
    )
    merge_parents = tuple(sorted((_REV_BRANCH_A, _REV_BRANCH_B)))
    merge = _build_delta(
        _parent_view(
            branch_a,
            branch_id="feature/a",
            unit_proof_cids={"unit/a": _PROOF_A_V2, "unit/b": _PROOF_B},
            logical_epoch=2,
        ),
        revision=_REV_MERGE,
        source_root_cid=_DIGEST_7,
        repository_state_cid=_DIGEST_8,
        parent_revision_ids=merge_parents,
        branch_id="feature/a",
        logical_epoch=3,
        transition_id="transition/hist-merge",
        units=(
            _delta_unit(
                "unit/a",
                "reuse",
                proof_object_cid=_PROOF_A_V2,
                parent_proof_object_cid=_PROOF_A_V2,
            ),
            _delta_unit(
                "unit/b",
                "replace",
                proof_object_cid=_PROOF_MERGE,
                parent_proof_object_cid=_PROOF_B,
                newly_admitted=True,
                cache_key_unchanged=False,
            ),
        ),
    )
    checkpoint = create_full_checkpoint(
        _state(
            revision=_REV_CHECKPOINT,
            source_root_cid=merge.source_root_cid,
            repository_state_cid=merge.repository_state_cid,
            parent_revision_ids=(merge.revision,),
        ),
        _policy(),
        units=_full_units(proof_a=_PROOF_A_V2, proof_b=_PROOF_MERGE),
        expected_unit_ids=("unit/a", "unit/b"),
        parent_seal_cid=merge.seal_cid(),
        fallback_reasons=("periodic_cadence", "release_tag"),
    )
    # Compaction over main-line full → localized delta (stable chain).
    outcome = compact_seal_chain(
        delta,
        RetentionPolicy(
            required_historical_seal_cids=(genesis.seal_cid(), delta.seal_cid()),
            required_evidence_cids=(_PROOF_A, _PROOF_B_V2),
        ),
        _policy(),
        seal_chain=(genesis, delta),
        units=_full_units(proof_a=_PROOF_A, proof_b=_PROOF_B_V2),
        expected_unit_ids=("unit/a", "unit/b"),
        repository_state=_state(
            revision=delta.revision,
            source_root_cid=delta.source_root_cid,
            repository_state_cid=delta.repository_state_cid,
            parent_revision_ids=delta.parent_revision_ids,
        ),
        trusted_keys=_TRUSTED,
        verify_current_cryptographically=True,
    )
    assert all(
        seal.sealed
        for seal in (genesis, delta, branch_a, branch_b, merge, checkpoint)
    )
    assert outcome.sealed is True and outcome.seal is not None
    return {
        "genesis_cid": genesis.seal_cid(),
        "delta_cid": delta.seal_cid(),
        "branch_a_cid": branch_a.seal_cid(),
        "branch_b_cid": branch_b.seal_cid(),
        "merge_cid": merge.seal_cid(),
        "checkpoint_cid": checkpoint.seal_cid(),
        "compaction_cid": outcome.compacted_seal_cid,
        "retained": tuple(outcome.retained_historical_seal_cids),
        "manifest_roots": (
            genesis.manifest_root_cid,
            delta.new_manifest_root_cid,
            merge.new_manifest_root_cid,
            checkpoint.manifest_root_cid,
        ),
        "forest_roots": (
            genesis.repository_proof_root,
            delta.new_forest_root_cid,
            merge.new_forest_root_cid,
            checkpoint.repository_proof_root,
        ),
        "aggregation_roots": (
            genesis.aggregation_root,
            delta.new_aggregation_root,
            merge.new_aggregation_root,
            checkpoint.aggregation_root,
        ),
    }


def test_end_to_end_positive_lifecycle_matrix() -> None:
    """Single cohesive walk of the positive lifecycle matrix."""

    genesis = _genesis_full()
    assert genesis.sealed is True and genesis.is_genesis is True

    localized = _localized_delta_from_full(genesis)
    assert localized.sealed is True
    assert localized.parent_seal_cid == genesis.seal_cid()

    branch_a = _build_delta(
        _parent_view(genesis, branch_id="feature/a"),
        revision=_REV_BRANCH_A,
        source_root_cid=_DIGEST_3,
        repository_state_cid=_DIGEST_4,
        parent_revision_ids=(genesis.revision,),
        branch_id="feature/a",
        transition_id="transition/e2e-a",
        units=(
            _delta_unit(
                "unit/a",
                "replace",
                proof_object_cid=_PROOF_A_V2,
                parent_proof_object_cid=_PROOF_A,
                newly_admitted=True,
                cache_key_unchanged=False,
            ),
            _delta_unit(
                "unit/b",
                "reuse",
                proof_object_cid=_PROOF_B,
                parent_proof_object_cid=_PROOF_B,
                cache_key_unchanged=True,
                category="static_analysis",
                terminal_status=ProofTerminalStatus.PROVED.value,
                proof_mode=ProofMode.DIRECT_EXECUTION_PROOF.value,
            ),
        ),
    )
    branch_b = _build_delta(
        _parent_view(genesis, branch_id="feature/b"),
        revision=_REV_BRANCH_B,
        source_root_cid=_DIGEST_5,
        repository_state_cid=_DIGEST_6,
        parent_revision_ids=(genesis.revision,),
        branch_id="feature/b",
        transition_id="transition/e2e-b",
        units=(
            _delta_unit(
                "unit/a",
                "reuse",
                proof_object_cid=_PROOF_A,
                parent_proof_object_cid=_PROOF_A,
                cache_key_unchanged=True,
            ),
            _delta_unit(
                "unit/b",
                "replace",
                proof_object_cid=_PROOF_B_V3,
                parent_proof_object_cid=_PROOF_B,
                newly_admitted=True,
                cache_key_unchanged=False,
            ),
        ),
    )
    assert branch_a.sealed and branch_b.sealed

    merge = _build_delta(
        _parent_view(
            branch_a,
            branch_id="feature/a",
            unit_proof_cids={"unit/a": _PROOF_A_V2, "unit/b": _PROOF_B},
            logical_epoch=2,
        ),
        revision=_REV_MERGE,
        source_root_cid=_DIGEST_7,
        repository_state_cid=_DIGEST_8,
        parent_revision_ids=tuple(sorted((_REV_BRANCH_A, _REV_BRANCH_B))),
        branch_id="feature/a",
        logical_epoch=3,
        transition_id="transition/e2e-merge",
        units=(
            _delta_unit(
                "unit/a",
                "reuse",
                proof_object_cid=_PROOF_A_V2,
                parent_proof_object_cid=_PROOF_A_V2,
            ),
            _delta_unit(
                "unit/b",
                "replace",
                proof_object_cid=_PROOF_MERGE,
                parent_proof_object_cid=_PROOF_B,
                newly_admitted=True,
                cache_key_unchanged=False,
            ),
        ),
    )
    assert merge.sealed is True
    assert set(merge.required_unit_ids) == {"unit/a", "unit/b"}

    rollback = _build_delta(
        _parent_view(
            merge,
            branch_id="feature/a",
            unit_proof_cids={"unit/a": _PROOF_A_V2, "unit/b": _PROOF_MERGE},
            logical_epoch=3,
        ),
        revision=_REV_ROLLBACK,
        source_root_cid=genesis.source_root_cid,
        repository_state_cid=_DIGEST_9,
        parent_revision_ids=(merge.revision,),
        branch_id="feature/a",
        logical_epoch=4,
        transition_id="transition/e2e-rollback",
        units=(
            _delta_unit(
                "unit/a",
                "reuse",
                proof_object_cid=_PROOF_A_V2,
                parent_proof_object_cid=_PROOF_A_V2,
            ),
            _delta_unit(
                "unit/b",
                "replace",
                proof_object_cid=_PROOF_ROLLBACK,
                parent_proof_object_cid=_PROOF_MERGE,
                newly_admitted=True,
                cache_key_unchanged=False,
            ),
        ),
    )
    assert rollback.sealed is True
    assert rollback.parent_seal_cid == merge.seal_cid()
    assert rollback.source_root_cid == genesis.source_root_cid

    decision = decide_checkpoint(
        seals_since_last_full_checkpoint=50,
        is_release_tag=True,
        estimated_reuse_ratio_basis_points=9000,
    )
    assert decision.require_full_checkpoint is True

    checkpoint = create_full_checkpoint(
        _state(
            revision=_REV_CHECKPOINT,
            source_root_cid=rollback.source_root_cid,
            repository_state_cid=rollback.repository_state_cid,
            parent_revision_ids=(rollback.revision,),
        ),
        _policy(),
        units=_full_units(proof_a=_PROOF_A_V2, proof_b=_PROOF_ROLLBACK),
        expected_unit_ids=("unit/a", "unit/b"),
        parent_seal_cid=rollback.seal_cid(),
        fallback_reasons=decision.reasons,
    )
    assert checkpoint.sealed is True
    assert set(checkpoint.required_unit_ids) == {"unit/a", "unit/b"}

    ok, cids, _, _ = verify_seal_chain(
        (genesis, localized),
        current_seal_cid=localized.seal_cid(),
    )
    assert ok is True
    assert genesis.seal_cid() in cids and localized.seal_cid() in cids

    outcome = compact_seal_chain(
        localized,
        RetentionPolicy(
            required_historical_seal_cids=(
                genesis.seal_cid(),
                localized.seal_cid(),
            ),
            required_evidence_cids=(_PROOF_A, _PROOF_B_V2),
        ),
        _policy(),
        seal_chain=(genesis, localized),
        units=_full_units(proof_a=_PROOF_A, proof_b=_PROOF_B_V2),
        expected_unit_ids=("unit/a", "unit/b"),
        repository_state=_state(
            revision=localized.revision,
            source_root_cid=localized.source_root_cid,
            repository_state_cid=localized.repository_state_cid,
            parent_revision_ids=localized.parent_revision_ids,
        ),
        trusted_keys=_TRUSTED,
        verify_current_cryptographically=True,
    )
    assert outcome.sealed is True
    assert outcome.retention_satisfied is True
    assert genesis.seal_cid() in outcome.retained_historical_seal_cids
    assert localized.seal_cid() in outcome.retained_historical_seal_cids
