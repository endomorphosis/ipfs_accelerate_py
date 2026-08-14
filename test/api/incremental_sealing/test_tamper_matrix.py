"""IPS-048: reject cache-context, manifest, and proof-forest tampering.

Every single-field cache-context mutation, unauthorized deletion, changed
manifest with old aggregate, wrong parent, missing invalidated unit, missing
unaffected leaf, duplicate/reordered leaf, corruption, and poisoning must
fail closed with a typed reason.  Omitting any complete-key component never
regains reuse.

Evidence subset: ``ips/cache-tamper@1``, ``ips/forest-tamper@1``.
"""

from __future__ import annotations

from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.aggregation import (
    AggregationReason,
    VerifiedUnit,
    aggregate_verified_units,
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
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.executor import (
    CachedCandidate,
    ExecutionOutcome,
    ExecutionReasonCode,
    execute_incremental_plan,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.explanations import (
    BOUND_CACHE_KEY_FIELDS,
    explain_reuse,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.full_checkpoint import (
    RepositoryStateView,
    VerificationPolicyView,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.planner import (
    ParentSealContext,
    UnitPlanningInput,
    create_incremental_plan,
)
from ipfs_datasets_py.logic.zkp.incremental_sealing.cache_key import (
    REQUIRED_FIELDS,
    CacheKeyError,
    ProofCacheKey,
    known_vectors,
    sample_proof_cache_key,
)
from ipfs_datasets_py.logic.zkp.incremental_sealing.evidence import (
    IntegrityCommitment,
    ProofMode,
    ProofTerminalStatus,
    SealStatus,
)
from ipfs_datasets_py.logic.zkp.incremental_sealing.forest_codec import (
    ForestCodecError,
    compute_category_root,
    known_vectors as forest_known_vectors,
    sample_leaf,
)
from ipfs_datasets_py.logic.zkp.incremental_sealing.identity import canonical_cid

# ---------------------------------------------------------------------------
# Evidence / closed contracts
# ---------------------------------------------------------------------------

CACHE_TAMPER_EVIDENCE = "ips/cache-tamper@1"
FOREST_TAMPER_EVIDENCE = "ips/forest-tamper@1"
EVIDENCE_SUBSETS = (CACHE_TAMPER_EVIDENCE, FOREST_TAMPER_EVIDENCE)

# Acceptance-named context roots mapped onto ProofCacheKey@1 fields.
# checked-spec binds through statement_cid (the checked property statement).
CONTEXT_ROOT_FIELDS: tuple[tuple[str, str], ...] = (
    ("source", "source_root_cid"),
    ("environment", "environment_cid"),
    ("selector", "test_selector_cid"),
    ("fixture", "fixture_cids"),
    ("config", "configuration_cid"),
    ("network-policy", "network_policy_cid"),
    ("policy", "policy_cid"),
    ("lock", "dependency_lock_cid"),
    ("tool", "tool_or_prover_version"),
    ("schema", "proof_schema_version"),
    ("canonicalization", "canonicalization_version"),
    ("checked-spec", "statement_cid"),
    ("dependency", "dependency_unit_roots"),
)

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
_PARENT_SEAL = "sha256:" + ("99" * 32)
_OLD_SOURCE = _DIGEST_A
_OLD_STATE = _DIGEST_B
_NEW_SOURCE = _DIGEST_1
_NEW_STATE = _DIGEST_2
_ENV = _DIGEST_C
_POLICY = _DIGEST_D
_OLD_MANIFEST = _DIGEST_E
_OLD_FOREST = _DIGEST_F
_OLD_AGG = _DIGEST_3
_PROOF_A = _DIGEST_4
_PROOF_B = _DIGEST_5
_PROOF_B_NEW = _DIGEST_6


# ---------------------------------------------------------------------------
# Shared builders (delta seal / executor)
# ---------------------------------------------------------------------------


def _parent(**overrides: object) -> ParentSealView:
    payload: dict[str, object] = {
        "seal_cid": _PARENT_SEAL,
        "accepted": True,
        "seal_status": SealStatus.SEALED_FULL.value,
        "repository_id": "repo/accelerate",
        "branch_id": "main",
        "revision": "rev-old",
        "source_root_cid": _OLD_SOURCE,
        "repository_state_cid": _OLD_STATE,
        "environment_cid": _ENV,
        "policy_cid": _POLICY,
        "manifest_root_cid": _OLD_MANIFEST,
        "forest_root_cid": _OLD_FOREST,
        "aggregation_root": _OLD_AGG,
        "required_unit_ids": ("unit/a", "unit/b"),
        "unit_proof_cids": {"unit/a": _PROOF_A, "unit/b": _PROOF_B},
        "parent_revision_ids": (),
        "logical_epoch": 1,
    }
    payload.update(overrides)
    return ParentSealView(**payload)  # type: ignore[arg-type]


def _state(**overrides: object) -> RepositoryStateView:
    payload: dict[str, object] = {
        "repository_id": "repo/accelerate",
        "revision": "rev-new",
        "source_root_cid": _NEW_SOURCE,
        "repository_state_cid": _NEW_STATE,
        "environment_cid": _ENV,
        "parent_revision_ids": ("rev-old",),
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
        "verification_key_id": "vk/1",
    }
    payload.update(overrides)
    return VerificationPolicyView(**payload)  # type: ignore[arg-type]


def _diff(**overrides: object) -> DiffCommitmentView:
    payload: dict[str, object] = {
        "diff_algorithm": (
            "ipfs_datasets_py/logic/zkp/incremental_sealing/"
            "repository_diff/algorithm@1"
        ),
        "changed_artifact_commitment": "sha256:" + ("88" * 32),
        "complete": True,
        "changed_paths": ("src/module.py",),
    }
    payload.update(overrides)
    return DiffCommitmentView(**payload)  # type: ignore[arg-type]


def _transition(**overrides: object) -> DeltaTransitionStatement:
    payload: dict[str, object] = {
        "schema": TRANSITION_SCHEMA,
        "parent_seal_cid": _PARENT_SEAL,
        "branch_id": "main",
        "old_source_root_cid": _OLD_SOURCE,
        "old_repository_state_cid": _OLD_STATE,
        "old_manifest_root_cid": _OLD_MANIFEST,
        "old_forest_root_cid": _OLD_FOREST,
        "old_aggregation_root": _OLD_AGG,
        "new_source_root_cid": _NEW_SOURCE,
        "new_repository_state_cid": _NEW_STATE,
        "new_revision": "rev-new",
        "parent_revision_ids": ("rev-old",),
        "diff": _diff(),
        "expected_manifest_unit_ids": ("unit/a", "unit/b"),
        "expected_surviving_leaf_ids": ("unit/a", "unit/b"),
        "forest_rebuilt": True,
        "aggregation_rebuilt": True,
        "logical_epoch": 2,
    }
    payload.update(overrides)
    return DeltaTransitionStatement(**payload)  # type: ignore[arg-type]


def _unit(unit_id: str, disposition: str, **overrides: object) -> DeltaUnitEvidence:
    payload: dict[str, object] = {
        "unit_id": unit_id,
        "disposition": disposition,
        "proof_object_cid": _PROOF_A if unit_id == "unit/a" else _PROOF_B_NEW,
        "category": "unit_test",
        "terminal_status": ProofTerminalStatus.INTEGRITY_VERIFIED.value,
        "proof_mode": ProofMode.INTEGRITY_ONLY.value,
        "required_for_seal": True,
        "cache_key_complete": True,
        "cache_key_unchanged": True,
        "freshly_verified": True,
        "newly_admitted": disposition in {"replace", "add"},
        "removal_authorized": disposition == "remove",
        "parent_proof_object_cid": (
            _PROOF_A if unit_id == "unit/a" else _PROOF_B
        ),
        "stale": False,
    }
    payload.update(overrides)
    return DeltaUnitEvidence(**payload)  # type: ignore[arg-type]


def _good_units() -> tuple[DeltaUnitEvidence, ...]:
    return (
        _unit(
            "unit/a",
            "reuse",
            proof_object_cid=_PROOF_A,
            parent_proof_object_cid=_PROOF_A,
        ),
        _unit(
            "unit/b",
            "replace",
            proof_object_cid=_PROOF_B_NEW,
            parent_proof_object_cid=_PROOF_B,
            newly_admitted=True,
            terminal_status=ProofTerminalStatus.PROVED.value,
            proof_mode=ProofMode.DIRECT_EXECUTION_PROOF.value,
        ),
    )


def _seal(**kwargs: object) -> DeltaSeal:
    return build_delta_seal(
        kwargs.pop("parent", _parent()),  # type: ignore[arg-type]
        kwargs.pop("state", _state()),  # type: ignore[arg-type]
        kwargs.pop("policy", _policy()),  # type: ignore[arg-type]
        kwargs.pop("transition", _transition()),  # type: ignore[arg-type]
        units=kwargs.pop("units", _good_units()),  # type: ignore[arg-type]
    )


def _agg_unit(unit_id: str, **overrides: object) -> VerifiedUnit:
    payload: dict[str, object] = {
        "unit_id": unit_id,
        "proof_object_cid": "sha256:" + ("ab" * 32),
        "category": "unit_test",
        "terminal_status": "integrity_verified",
        "repository_state_cid": "sha256:" + ("11" * 32),
        "environment_cid": "sha256:" + ("22" * 32),
    }
    payload.update(overrides)
    return VerifiedUnit(**payload)  # type: ignore[arg-type]


def _plan_unit(unit_id: str, **overrides: object) -> UnitPlanningInput:
    payload: dict[str, object] = {
        "unit_id": unit_id,
        "preserved": True,
        "cache_key_complete": True,
        "admitted": True,
        "candidate_present": True,
    }
    payload.update(overrides)
    return UnitPlanningInput(**payload)  # type: ignore[arg-type]


def _reuse_plan(**unit_overrides: object):
    parent = ParentSealContext(
        seal_cid=_PARENT_SEAL,
        repository_state_cid=_OLD_STATE,
        source_root_cid=_OLD_SOURCE,
    )
    return create_incremental_plan(
        parent,
        _OLD_STATE,
        _NEW_STATE,
        units=(
            _plan_unit("unit/reuse", **unit_overrides),
            _plan_unit(
                "unit/reprove",
                preserved=False,
                invalidated=True,
                admitted=False,
            ),
        ),
    )


def _good_candidate(unit_id: str, digest: str, **flags: bool) -> CachedCandidate:
    cid = "sha256:" + ("11" * 32)
    return CachedCandidate(
        unit_id=unit_id,
        expected_digest=digest,
        observed_digest=digest,
        public_input_cid=cid,
        observed_public_input_cid=cid,
        proof_object_cid=cid,
        evidence=IntegrityCommitment(
            digest=digest,
            cid=cid,
            merkle_inclusion="leaf:0",
            byte_length=32,
        ),
        stale=bool(flags.get("stale", False)),
        poisoned=bool(flags.get("poisoned", False)),
        corrupt=bool(flags.get("corrupt", False)),
        simulated=bool(flags.get("simulated", False)),
    )


def _mutate_cache_field(field: str, base: ProofCacheKey) -> Any:
    """Produce a single-field mutation that changes the key CID.

    Mirrors the hermetic single-field mutation path used by
    ``cache_key.known_vectors`` so every acceptance-named context root
    stays constructible under the fail-closed ProofCacheKey@1 validators.
    """

    if field in {"source_artifact_cids", "dependency_unit_roots", "fixture_cids"}:
        current = list(getattr(base, field))
        extra = canonical_cid(
            {"ips_tamper_matrix": field, "extra": True, "v": 1}
        )
        return sorted(set(current) | {extra})
    if field in {
        "statement_cid",
        "public_input_cid",
        "private_input_commitment",
        "source_root_cid",
        "environment_cid",
        "dependency_lock_cid",
        "configuration_cid",
        "network_policy_cid",
        "test_selector_cid",
        "policy_cid",
    }:
        return canonical_cid(
            {"ips_tamper_matrix": field, "mutated": True, "v": 1}
        )
    if field == "evidence_class":
        return "IntegrityCommitment"
    if field == "proof_unit_kind":
        return "formal_obligation"
    if field == "proof_mode":
        return "theorem_certificate"
    if field == "dependency_roots_complete":
        # Completeness is True-only; incompleteness fails closed separately.
        return True
    current = getattr(base, field)
    if isinstance(current, str) and current and current != "n/a":
        return f"{current}/mutated"
    return f"mutated-{field}"


# ---------------------------------------------------------------------------
# Evidence and contract surface
# ---------------------------------------------------------------------------


def test_evidence_subsets_and_bound_complete_key() -> None:
    assert CACHE_TAMPER_EVIDENCE == "ips/cache-tamper@1"
    assert FOREST_TAMPER_EVIDENCE == "ips/forest-tamper@1"
    assert EVIDENCE_SUBSETS == (
        "ips/cache-tamper@1",
        "ips/forest-tamper@1",
    )
    assert tuple(BOUND_CACHE_KEY_FIELDS) == tuple(REQUIRED_FIELDS)
    assert len(REQUIRED_FIELDS) >= 20
    names = {name for name, _ in CONTEXT_ROOT_FIELDS}
    for required in (
        "source",
        "environment",
        "selector",
        "fixture",
        "config",
        "network-policy",
        "policy",
        "lock",
        "tool",
        "schema",
        "canonicalization",
        "checked-spec",
        "dependency",
    ):
        assert required in names
    for _, field in CONTEXT_ROOT_FIELDS:
        assert field in REQUIRED_FIELDS


# ---------------------------------------------------------------------------
# Cache-context single-field mutations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "field"),
    CONTEXT_ROOT_FIELDS,
    ids=[label for label, _ in CONTEXT_ROOT_FIELDS],
)
def test_different_context_root_changes_cache_key_and_blocks_reuse(
    label: str, field: str
) -> None:
    """Different source/env/selector/... roots must change the key and reject reuse."""

    base = sample_proof_cache_key()
    base_cid = base.key_cid()
    mutated_value = _mutate_cache_field(field, base)
    payload = base.to_canonical()
    payload[field] = mutated_value
    mutated = ProofCacheKey.from_canonical(payload)
    assert mutated.key_cid() != base_cid, label

    # Changed complete key cannot be presented as reuse on a delta seal.
    units = (
        _unit(
            "unit/a",
            "reuse",
            proof_object_cid=_PROOF_A,
            cache_key_unchanged=False,
        ),
        _unit(
            "unit/b",
            "replace",
            proof_object_cid=_PROOF_B_NEW,
            newly_admitted=True,
        ),
    )
    seal = _seal(units=units)
    assert seal.sealed is False
    assert seal.reason in {
        DeltaSealReason.STALE_REUSE,
        DeltaSealReason.INCOMPLETE_CACHE_KEY,
    }
    assert "reuse_complete_cache_key" in seal.invariants_failed
    assert seal.reason.value  # typed non-empty reason code


def test_known_vectors_cover_every_required_field_mutation() -> None:
    vectors = known_vectors()
    mutations = vectors["single_field_mutations"]
    for field in REQUIRED_FIELDS:
        if field == "dependency_roots_complete":
            # Completeness is True-only; incompleteness fails closed separately.
            continue
        assert field in mutations
        assert mutations[field]["base_key_cid"] != mutations[field]["mutated_key_cid"]


# ---------------------------------------------------------------------------
# No complete-key component can be omitted to regain reuse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", list(REQUIRED_FIELDS), ids=list(REQUIRED_FIELDS))
def test_omitting_complete_key_component_fails_closed(field: str) -> None:
    base = sample_proof_cache_key().to_canonical()
    del base[field]
    with pytest.raises(CacheKeyError) as excinfo:
        ProofCacheKey.from_canonical(base)
    message = str(excinfo.value).casefold()
    assert "missing" in message or field.casefold() in message
    # A missing complete-key component cannot be reconstructed into a reusable key.
    with pytest.raises(CacheKeyError):
        ProofCacheKey.from_canonical({**base, field: None})  # type: ignore[dict-item]


def test_incomplete_dependency_roots_fail_closed() -> None:
    payload = sample_proof_cache_key().to_canonical()
    payload["dependency_roots_complete"] = False
    with pytest.raises(CacheKeyError) as excinfo:
        ProofCacheKey.from_canonical(payload)
    assert "incomplete" in str(excinfo.value).casefold()


def test_incomplete_cache_key_cannot_regain_reuse_via_planner() -> None:
    parent = ParentSealContext(
        seal_cid=_PARENT_SEAL,
        repository_state_cid=_OLD_STATE,
        source_root_cid=_OLD_SOURCE,
    )
    plan = create_incremental_plan(
        parent,
        _OLD_STATE,
        _NEW_STATE,
        units=(_plan_unit("unit/partial", cache_key_complete=False, admitted=True),),
    )
    assert plan.reusable_unit_ids == ()
    assert plan.units[0].unit_id == "unit/partial"
    assert plan.units[0].reason == "incomplete_cache_key"
    assert plan.units[0].kind.value == "reprove"
    # Candidate presence and prior admission never restore reuse authority.
    rejected = create_incremental_plan(
        parent,
        _OLD_STATE,
        _NEW_STATE,
        units=(
            _plan_unit(
                "unit/hinted",
                cache_key_complete=False,
                admitted=True,
                candidate_present=True,
                preserved=True,
            ),
        ),
    )
    assert rejected.reusable_unit_ids == ()
    assert rejected.units[0].kind.value == "reprove"
    assert rejected.units[0].reason == "incomplete_cache_key"


def test_incomplete_cache_key_on_delta_reuse_rejects_with_typed_reason() -> None:
    units = (
        _unit(
            "unit/a",
            "reuse",
            proof_object_cid=_PROOF_A,
            cache_key_complete=False,
        ),
        _unit(
            "unit/b",
            "replace",
            proof_object_cid=_PROOF_B_NEW,
            newly_admitted=True,
        ),
    )
    seal = _seal(units=units)
    assert seal.sealed is False
    assert seal.reason is DeltaSealReason.INCOMPLETE_CACHE_KEY
    assert seal.reason.value == "incomplete_cache_key"
    assert "reuse_complete_cache_key" in seal.invariants_failed


def test_omitted_key_component_cannot_be_explained_as_reuse() -> None:
    """Even with a sealed unit set, incomplete/changed fields reject reuse explanation."""

    seal = _seal()
    assert seal.sealed is True
    explanation = explain_reuse(
        seal,
        "unit/a",
        cache_key_complete=False,
        changed_fields=("source_root_cid",),
        admitted=False,
    )
    # Unit is in the reused set of the seal payload, but incomplete key forces
    # the explanation to surface unequal/changed bindings for the full field set.
    assert explanation.bound_cache_key_fields == BOUND_CACHE_KEY_FIELDS
    assert "source_root_cid" in {
        item.field_name for item in explanation.unequal_cache_key_fields
    }
    # Completeness flag is recorded; reuse authority still requires the seal
    # path above (incomplete keys never seal as reuse).
    assert explanation.cache_key_complete is False


# ---------------------------------------------------------------------------
# Unauthorized deleted test
# ---------------------------------------------------------------------------


def test_unauthorized_deleted_test_rejects_with_typed_reason() -> None:
    units = (
        _unit("unit/a", "reuse", proof_object_cid=_PROOF_A),
        _unit(
            "unit/b",
            "remove",
            removal_authorized=False,
            proof_object_cid=_PROOF_B,
        ),
    )
    seal = _seal(
        units=units,
        transition=_transition(
            expected_manifest_unit_ids=("unit/a",),
            expected_surviving_leaf_ids=("unit/a",),
        ),
    )
    assert seal.sealed is False
    assert seal.reason is DeltaSealReason.UNAUTHORIZED_DELETION
    assert seal.reason.value == "unauthorized_deletion"
    assert "deletions_authorized" in seal.invariants_failed
    assert "unit/b" in seal.rejected_unit_ids


# ---------------------------------------------------------------------------
# Changed manifest with old aggregate
# ---------------------------------------------------------------------------


def test_changed_manifest_rejects_with_typed_reason() -> None:
    units = (
        _agg_unit("unit/a"),
        _agg_unit("unit/b"),
        _agg_unit("unit/c"),
    )
    good = aggregate_verified_units(units)
    assert good.accepted is True
    changed = aggregate_verified_units(
        units, expected_root="sha256:" + ("ff" * 32)
    )
    assert changed.accepted is False
    assert changed.reason is AggregationReason.CHANGED_MANIFEST
    assert changed.reason.value == "changed_manifest"


def test_old_aggregate_with_changed_unit_set_rejects() -> None:
    seal = _seal(transition=_transition(aggregation_rebuilt=False))
    assert seal.sealed is False
    assert seal.reason is DeltaSealReason.OLD_AGGREGATE
    assert seal.reason.value == "old_aggregate"
    assert "forest_commits_exact_units" in seal.invariants_failed


def test_changed_manifest_with_old_aggregate_both_reject() -> None:
    """Composite: membership/root drift plus stale aggregate both fail closed."""

    units = (_agg_unit("unit/a"), _agg_unit("unit/b"))
    first = aggregate_verified_units(units)
    stale = aggregate_verified_units(
        units,
        previous_root="sha256:" + ("ee" * 32),
        expected_root="sha256:" + ("ff" * 32),
    )
    assert first.accepted is True
    assert stale.accepted is False
    assert stale.reason in {
        AggregationReason.CHANGED_MANIFEST,
        AggregationReason.STALE_AGGREGATE,
    }
    old_agg = _seal(transition=_transition(aggregation_rebuilt=False))
    assert old_agg.sealed is False
    assert old_agg.reason is DeltaSealReason.OLD_AGGREGATE


# ---------------------------------------------------------------------------
# Wrong parent
# ---------------------------------------------------------------------------


def test_wrong_parent_rejects_with_typed_reason() -> None:
    seal = _seal(
        transition=_transition(parent_seal_cid="sha256:" + ("00" * 32)),
    )
    assert seal.sealed is False
    assert seal.reason is DeltaSealReason.WRONG_PARENT
    assert seal.reason.value == "wrong_parent"
    assert "exact_parent_bound" in seal.invariants_failed


# ---------------------------------------------------------------------------
# Missing invalidated unit / missing unaffected leaf
# ---------------------------------------------------------------------------


def test_missing_invalidated_unit_rejects_with_typed_reason() -> None:
    """Invalidated unit presented without a newly admitted replacement."""

    units = (
        _unit("unit/a", "reuse", proof_object_cid=_PROOF_A),
        _unit(
            "unit/b",
            "replace",
            proof_object_cid=_PROOF_B,  # same as parent → not a real replacement
            newly_admitted=True,
            parent_proof_object_cid=_PROOF_B,
        ),
    )
    seal = _seal(units=units)
    assert seal.sealed is False
    assert seal.reason is DeltaSealReason.MISSING_REPLACEMENT
    assert seal.reason.value == "missing_replacement"
    assert "invalidated_have_new_proofs" in seal.invariants_failed
    assert "unit/b" in seal.rejected_unit_ids


def test_missing_unaffected_leaf_rejects_with_typed_reason() -> None:
    units = (
        _unit("unit/a", "reuse", proof_object_cid=_PROOF_A),
        _unit(
            "unit/b",
            "replace",
            proof_object_cid=_PROOF_B_NEW,
            newly_admitted=True,
        ),
    )
    seal = _seal(
        units=units,
        transition=_transition(
            expected_manifest_unit_ids=("unit/a", "unit/b"),
            # Claims unit/ghost survived but it is absent from the unit set.
            expected_surviving_leaf_ids=("unit/a", "unit/b", "unit/ghost"),
        ),
    )
    assert seal.sealed is False
    assert seal.reason is DeltaSealReason.LOST_LEAF
    assert seal.reason.value == "lost_leaf"
    assert "forest_commits_exact_units" in seal.invariants_failed
    assert "unit/ghost" in seal.rejected_unit_ids


def test_silent_parent_unit_loss_rejects() -> None:
    # unit/b from parent is neither replaced, reused, nor removed.
    units = (_unit("unit/a", "reuse", proof_object_cid=_PROOF_A),)
    seal = _seal(
        units=units,
        transition=_transition(
            expected_manifest_unit_ids=("unit/a",),
            expected_surviving_leaf_ids=("unit/a", "unit/b"),
        ),
    )
    assert seal.sealed is False
    assert seal.reason in {
        DeltaSealReason.INCOMPLETE_MANIFEST,
        DeltaSealReason.LOST_LEAF,
    }
    assert seal.reason.value in {"incomplete_manifest", "lost_leaf"}


# ---------------------------------------------------------------------------
# Duplicate / reordered leaf (forest + manifest)
# ---------------------------------------------------------------------------


def test_duplicate_forest_leaf_rejects_with_typed_error() -> None:
    vectors = forest_known_vectors()
    duplicate_payload = vectors["fail_closed"]["duplicate_unit_ids"]
    with pytest.raises(ForestCodecError, match="duplicate") as excinfo:
        compute_category_root("unit_test", duplicate_payload)
    assert "duplicate" in str(excinfo.value).casefold()


def test_reordered_forest_leaf_rejects_with_typed_error() -> None:
    vectors = forest_known_vectors()
    reordered_payload = vectors["fail_closed"]["reordered_leaves"]
    with pytest.raises(ForestCodecError, match="canonical") as excinfo:
        compute_category_root("unit_test", reordered_payload)
    message = str(excinfo.value).casefold()
    assert "canonical" in message or "order" in message


def test_duplicate_and_reordered_manifest_children_reject() -> None:
    missing = aggregate_verified_units(
        (_agg_unit("unit/a"), _agg_unit("unit/b")),
        expected_unit_ids=("unit/a", "unit/b", "unit/c"),
    )
    assert missing.accepted is False
    assert missing.reason is AggregationReason.MISSING_CHILD

    duplicate = aggregate_verified_units(
        (_agg_unit("unit/a"), _agg_unit("unit/a")),
        expected_unit_ids=("unit/a", "unit/a"),
    )
    assert duplicate.accepted is False
    assert duplicate.reason is AggregationReason.DUPLICATE_CHILD
    assert duplicate.reason.value == "duplicate_child"

    reordered = aggregate_verified_units(
        (_agg_unit("unit/b"), _agg_unit("unit/a")),
        expected_unit_ids=("unit/a", "unit/b"),
    )
    assert reordered.accepted is False
    assert reordered.reason is AggregationReason.REORDERED_CHILDREN
    assert reordered.reason.value == "reordered_children"


def test_duplicate_leaf_positions_reject() -> None:
    leaves = (
        sample_leaf(proof_unit_id="unit/a", category="unit_test", position=0),
        sample_leaf(proof_unit_id="unit/b", category="unit_test", position=0),
    )
    with pytest.raises(ForestCodecError) as excinfo:
        compute_category_root("unit_test", leaves)
    assert "duplicate" in str(excinfo.value).casefold()


# ---------------------------------------------------------------------------
# Corruption and poisoning
# ---------------------------------------------------------------------------


def test_corrupt_candidate_rejects_with_typed_reason() -> None:
    digest = "sha256:" + ("ab" * 32)
    candidate = _good_candidate("unit/reuse", digest, corrupt=True)
    result = execute_incremental_plan(
        _reuse_plan(),
        fetch=lambda uid, c=candidate: c if uid == "unit/reuse" else None,
    )
    assert result.outcome is ExecutionOutcome.REJECTED
    assert result.succeeded is False
    assert result.may_aggregate is False
    assert "unit/reuse" in result.rejected_unit_ids
    assert ExecutionReasonCode.CORRUPT_CANDIDATE.value in result.reason_codes
    assert ExecutionReasonCode.CORRUPT_CANDIDATE.value == "corrupt_candidate"


def test_poisoned_candidate_rejects_with_typed_reason() -> None:
    digest = "sha256:" + ("ab" * 32)
    candidate = _good_candidate("unit/reuse", digest, poisoned=True)
    result = execute_incremental_plan(
        _reuse_plan(),
        fetch=lambda uid, c=candidate: c if uid == "unit/reuse" else None,
    )
    assert result.outcome is ExecutionOutcome.REJECTED
    assert result.succeeded is False
    assert result.may_aggregate is False
    assert "unit/reuse" in result.rejected_unit_ids
    assert ExecutionReasonCode.POISONED_CANDIDATE.value in result.reason_codes
    assert ExecutionReasonCode.POISONED_CANDIDATE.value == "poisoned_candidate"


def test_stale_and_digest_mismatch_also_block_aggregation() -> None:
    digest = "sha256:" + ("ab" * 32)
    cases = {
        "stale": (
            _good_candidate("unit/reuse", digest, stale=True),
            ExecutionReasonCode.STALE_CANDIDATE,
        ),
        "mismatch": (
            CachedCandidate(
                "unit/reuse",
                digest,
                "sha256:" + ("ff" * 32),
                "sha256:" + ("11" * 32),
                "sha256:" + ("11" * 32),
            ),
            ExecutionReasonCode.DIGEST_MISMATCH,
        ),
    }
    for name, (candidate, expected) in cases.items():
        result = execute_incremental_plan(
            _reuse_plan(),
            fetch=lambda uid, c=candidate: c if uid == "unit/reuse" else None,
        )
        assert result.outcome is ExecutionOutcome.REJECTED, name
        assert result.may_aggregate is False, name
        assert expected.value in result.reason_codes, name


# ---------------------------------------------------------------------------
# Composite acceptance matrix
# ---------------------------------------------------------------------------


def test_acceptance_matrix_all_cases_reject_with_typed_reasons() -> None:
    """Compact end-to-end matrix covering every acceptance bullet."""

    failures: list[tuple[str, str]] = []

    def record(name: str, reason: str, *, sealed: bool = False) -> None:
        assert reason, name
        assert sealed is False, name
        failures.append((name, reason))

    # Context roots → key CID change.
    base = sample_proof_cache_key()
    base_cid = base.key_cid()
    for label, field in CONTEXT_ROOT_FIELDS:
        payload = base.to_canonical()
        payload[field] = _mutate_cache_field(field, base)
        mutated = ProofCacheKey.from_canonical(payload)
        assert mutated.key_cid() != base_cid, label
        failures.append((f"context/{label}", "cache_key_changed"))

    # Unauthorized deletion.
    unauth = _seal(
        units=(
            _unit("unit/a", "reuse", proof_object_cid=_PROOF_A),
            _unit("unit/b", "remove", removal_authorized=False),
        ),
        transition=_transition(
            expected_manifest_unit_ids=("unit/a",),
            expected_surviving_leaf_ids=("unit/a",),
        ),
    )
    record("unauthorized_deletion", unauth.reason.value, sealed=unauth.sealed)

    # Changed manifest + old aggregate.
    changed = aggregate_verified_units(
        (_agg_unit("unit/a"), _agg_unit("unit/b")),
        expected_root="sha256:" + ("ff" * 32),
    )
    record("changed_manifest", changed.reason.value, sealed=changed.accepted)
    old_agg = _seal(transition=_transition(aggregation_rebuilt=False))
    record("old_aggregate", old_agg.reason.value, sealed=old_agg.sealed)

    # Wrong parent.
    wrong_parent = _seal(
        transition=_transition(parent_seal_cid="sha256:" + ("00" * 32)),
    )
    record("wrong_parent", wrong_parent.reason.value, sealed=wrong_parent.sealed)

    # Missing invalidated unit.
    missing_inv = _seal(
        units=(
            _unit("unit/a", "reuse", proof_object_cid=_PROOF_A),
            _unit(
                "unit/b",
                "replace",
                proof_object_cid=_PROOF_B,
                newly_admitted=True,
            ),
        )
    )
    record(
        "missing_invalidated_unit",
        missing_inv.reason.value,
        sealed=missing_inv.sealed,
    )

    # Missing unaffected leaf.
    lost = _seal(
        transition=_transition(
            expected_surviving_leaf_ids=("unit/a", "unit/b", "unit/ghost"),
        )
    )
    record("missing_unaffected_leaf", lost.reason.value, sealed=lost.sealed)

    # Duplicate / reordered leaves.
    with pytest.raises(ForestCodecError):
        compute_category_root(
            "unit_test",
            forest_known_vectors()["fail_closed"]["duplicate_unit_ids"],
        )
    failures.append(("duplicate_leaf", "forest_codec_error"))
    with pytest.raises(ForestCodecError):
        compute_category_root(
            "unit_test",
            forest_known_vectors()["fail_closed"]["reordered_leaves"],
        )
    failures.append(("reordered_leaf", "forest_codec_error"))
    dup_manifest = aggregate_verified_units(
        (_agg_unit("unit/a"), _agg_unit("unit/a")),
        expected_unit_ids=("unit/a", "unit/a"),
    )
    record(
        "duplicate_manifest_child",
        dup_manifest.reason.value,
        sealed=dup_manifest.accepted,
    )
    reorder_manifest = aggregate_verified_units(
        (_agg_unit("unit/b"), _agg_unit("unit/a")),
        expected_unit_ids=("unit/a", "unit/b"),
    )
    record(
        "reordered_manifest_child",
        reorder_manifest.reason.value,
        sealed=reorder_manifest.accepted,
    )

    # Corruption / poisoning.
    digest = "sha256:" + ("ab" * 32)
    for flag, code in (
        ("corrupt", ExecutionReasonCode.CORRUPT_CANDIDATE),
        ("poisoned", ExecutionReasonCode.POISONED_CANDIDATE),
    ):
        result = execute_incremental_plan(
            _reuse_plan(),
            fetch=lambda uid, f=flag: (
                _good_candidate("unit/reuse", digest, **{f: True})
                if uid == "unit/reuse"
                else None
            ),
        )
        assert result.outcome is ExecutionOutcome.REJECTED
        assert code.value in result.reason_codes
        failures.append((flag, code.value))

    # Omitting any complete-key component fails closed — never regains reuse.
    for field in REQUIRED_FIELDS:
        payload = sample_proof_cache_key().to_canonical()
        del payload[field]
        with pytest.raises(CacheKeyError) as excinfo:
            ProofCacheKey.from_canonical(payload)
        message = str(excinfo.value).casefold()
        assert "missing" in message or field.casefold() in message
        failures.append((f"omit/{field}", "cache_key_error"))

    # Incomplete key never reuses via delta seal or planner.
    incomplete = _seal(
        units=(
            _unit(
                "unit/a",
                "reuse",
                proof_object_cid=_PROOF_A,
                cache_key_complete=False,
            ),
            _unit(
                "unit/b",
                "replace",
                proof_object_cid=_PROOF_B_NEW,
                newly_admitted=True,
            ),
        )
    )
    record(
        "incomplete_cache_key",
        incomplete.reason.value,
        sealed=incomplete.sealed,
    )
    assert incomplete.reason is DeltaSealReason.INCOMPLETE_CACHE_KEY
    incomplete_plan = create_incremental_plan(
        ParentSealContext(
            seal_cid=_PARENT_SEAL,
            repository_state_cid=_OLD_STATE,
            source_root_cid=_OLD_SOURCE,
        ),
        _OLD_STATE,
        _NEW_STATE,
        units=(_plan_unit("unit/partial", cache_key_complete=False, admitted=True),),
    )
    assert incomplete_plan.reusable_unit_ids == ()
    failures.append(("incomplete_cache_key_planner", "incomplete_cache_key"))

    expected_names = {
        "unauthorized_deletion",
        "changed_manifest",
        "old_aggregate",
        "wrong_parent",
        "missing_invalidated_unit",
        "missing_unaffected_leaf",
        "duplicate_leaf",
        "reordered_leaf",
        "duplicate_manifest_child",
        "reordered_manifest_child",
        "corrupt",
        "poisoned",
        "incomplete_cache_key",
        "incomplete_cache_key_planner",
    }
    observed = {name for name, _ in failures}
    assert expected_names <= observed
    # Every context root label must appear.
    for label, _ in CONTEXT_ROOT_FIELDS:
        assert f"context/{label}" in observed
    # Every complete-key component omission must appear.
    for field in REQUIRED_FIELDS:
        assert f"omit/{field}" in observed
    # All recorded reasons are non-empty typed codes.
    assert all(reason for _, reason in failures)
    # Context mutations and omit cases only record typed drift/error codes;
    # every other acceptance bullet was forced through record() (sealed=False).
    non_context = {
        name
        for name, _ in failures
        if not name.startswith("context/") and not name.startswith("omit/")
    }
    assert expected_names <= non_context


def test_happy_path_still_seals_when_untampered() -> None:
    """Sanity: the matrix builders do not over-reject legitimate seals."""

    seal = _seal()
    assert seal.sealed is True
    assert seal.reason is DeltaSealReason.SEALED
    assert seal.reason.value == "sealed"
