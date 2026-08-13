"""IPS-036: bounded manifest aggregation and capability-gated recursion."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.aggregation import (
    DEFAULT_FAN_IN,
    MANIFEST_AGGREGATION_EVIDENCE,
    MANIFEST_DOES_NOT_ESTABLISH,
    MANIFEST_ESTABLISHES,
    RECEIPT_AGGREGATION_DOES_NOT_ESTABLISH,
    RECEIPT_AGGREGATION_ESTABLISHES,
    RECURSIVE_AGGREGATION_EVIDENCE,
    AggregationContext,
    AggregationError,
    AggregationMode,
    AggregationOutcome,
    AggregationReasonCode,
    HermeticRecursiveAggregationBackend,
    ManifestAggregationResult,
    ProofAggregator,
    RecursiveAggregationResult,
    VerifiedChild,
    aggregate_verified_units,
    closed_aggregation_levels,
    closed_aggregation_modes,
    closed_aggregation_reason_codes,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.backends import (
    AggregationDisposition,
    HermeticTestOnlyRecursiveBackend,
    probe_backend_capability,
)

_REPO = "repository:test-ips-036"
_ENV = "sha256:" + ("11" * 32)
_POLICY = "sha256:" + ("22" * 32)


def _ctx(**overrides: object) -> AggregationContext:
    payload = {
        "repository_id": _REPO,
        "environment_cid": _ENV,
        "policy_cid": _POLICY,
    }
    payload.update(overrides)
    return AggregationContext(**payload)  # type: ignore[arg-type]


def _child(unit_id: str, **overrides: object) -> VerifiedChild:
    digest = "sha256:" + (unit_id.encode().hex()[:64].ljust(64, "0"))
    payload = {
        "unit_id": unit_id,
        "proof_object_cid": digest,
        "category": "unit_test",
        "terminal_status": "proved",
        "repository_id": _REPO,
        "environment_cid": _ENV,
        "verification_digest": digest,
        "evidence_class": "IntegrityCommitment",
        "verified": True,
        "policy_cid": _POLICY,
    }
    payload.update(overrides)
    return VerifiedChild(**payload)  # type: ignore[arg-type]


def _children(*unit_ids: str, **shared: object) -> tuple[VerifiedChild, ...]:
    return tuple(_child(unit_id, **shared) for unit_id in unit_ids)


def _recursive_capability():
    return probe_backend_capability(
        "groth16",
        recursive_backend=HermeticTestOnlyRecursiveBackend(),
        availability_overrides={"groth16": True},
    )


def test_evidence_subsets_and_closed_vocabularies() -> None:
    assert MANIFEST_AGGREGATION_EVIDENCE == "ips/manifest-aggregation@1"
    assert RECURSIVE_AGGREGATION_EVIDENCE == "ips/recursive-aggregation@1"
    modes = closed_aggregation_modes()
    assert "manifest_aggregation" in modes
    assert "recursive_verification" in modes
    reasons = closed_aggregation_reason_codes()
    for required in (
        "missing_child",
        "duplicate_child",
        "reordered_child",
        "failed_child",
        "changed_manifest",
        "old_aggregate",
        "recursion_not_admitted",
        "child_not_verified",
    ):
        assert required in reasons
    levels = closed_aggregation_levels()
    assert levels == frozenset({"leaf", "batch", "category", "repository"})
    assert DEFAULT_FAN_IN >= 2


def test_manifest_aggregation_happy_path_binds_identities_and_nonclaims() -> None:
    kids = _children("unit/a", "unit/b", "unit/c")
    result = aggregate_verified_units(
        kids,
        _ctx(),
        fan_in=2,
        prefer_recursive=False,
        expected_unit_ids=("unit/a", "unit/b", "unit/c"),
    )
    assert isinstance(result, ManifestAggregationResult)
    assert result.outcome is AggregationOutcome.AGGREGATED
    assert result.succeeded is True
    assert result.mode is AggregationMode.MANIFEST_AGGREGATION
    assert result.evidence_subset == MANIFEST_AGGREGATION_EVIDENCE
    assert result.child_count == 3
    assert result.child_unit_ids == ("unit/a", "unit/b", "unit/c")
    assert result.recursive_verification is False
    assert result.children_individually_verified is True
    assert result.test_execution_directly_proven is False
    assert result.aggregate_root is not None
    assert result.manifest_root is not None
    assert result.repository_node is not None
    assert len(result.batch_nodes) >= 1
    assert len(result.category_nodes) == 1
    assert "unit_test" in result.category_roots
    assert result.rebuilt_categories == ("unit_test",)
    assert "recursive verification" in result.does_not_establish.casefold()
    assert "test execution" in result.does_not_establish.casefold()
    assert result.establishes == MANIFEST_ESTABLISHES
    assert result.does_not_establish == MANIFEST_DOES_NOT_ESTABLISH
    # Deterministic roots.
    again = aggregate_verified_units(
        kids, _ctx(), fan_in=2, prefer_recursive=False
    )
    assert again.aggregate_root == result.aggregate_root
    assert again.manifest_root == result.manifest_root


def test_missing_child_rejects() -> None:
    kids = _children("unit/a", "unit/c")
    result = aggregate_verified_units(
        kids,
        _ctx(),
        prefer_recursive=False,
        expected_unit_ids=("unit/a", "unit/b", "unit/c"),
    )
    assert result.outcome is AggregationOutcome.REJECTED
    assert AggregationReasonCode.MISSING_CHILD.value in result.reason_codes
    assert result.aggregate_root is None


def test_duplicate_child_rejects() -> None:
    kids = (_child("unit/a"), _child("unit/a"), _child("unit/b"))
    result = aggregate_verified_units(kids, _ctx(), prefer_recursive=False)
    assert result.outcome is AggregationOutcome.REJECTED
    assert AggregationReasonCode.DUPLICATE_CHILD.value in result.reason_codes


def test_reordered_child_rejects() -> None:
    # Expected order a,b,c; presented c,a,b.
    kids = _children("unit/c", "unit/a", "unit/b")
    result = aggregate_verified_units(
        kids,
        _ctx(),
        prefer_recursive=False,
        expected_unit_ids=("unit/a", "unit/b", "unit/c"),
    )
    assert result.outcome is AggregationOutcome.REJECTED
    assert AggregationReasonCode.REORDERED_CHILD.value in result.reason_codes

    # Without expected list, non-canonical order also rejects.
    unordered = aggregate_verified_units(
        _children("unit/b", "unit/a"),
        _ctx(),
        prefer_recursive=False,
    )
    assert unordered.outcome is AggregationOutcome.REJECTED
    assert AggregationReasonCode.REORDERED_CHILD.value in unordered.reason_codes


def test_failed_child_rejects() -> None:
    kids = (
        _child("unit/a"),
        _child("unit/b", terminal_status="failed"),
    )
    result = aggregate_verified_units(kids, _ctx(), prefer_recursive=False)
    assert result.outcome is AggregationOutcome.REJECTED
    assert AggregationReasonCode.FAILED_CHILD.value in result.reason_codes

    unverified = (
        _child("unit/a"),
        _child("unit/b", verified=False),
    )
    result2 = aggregate_verified_units(unverified, _ctx(), prefer_recursive=False)
    assert result2.outcome is AggregationOutcome.REJECTED
    assert AggregationReasonCode.FAILED_CHILD.value in result2.reason_codes


def test_changed_manifest_and_old_aggregate_reject_on_verify() -> None:
    kids = _children("unit/a", "unit/b")
    aggregator = ProofAggregator(fan_in=2, prefer_recursive=False)
    original = aggregator.aggregate(
        kids, _ctx(), expected_unit_ids=("unit/a", "unit/b")
    )
    assert original.succeeded is True

    # Unchanged children accept.
    ok = aggregator.verify_aggregate(original, kids, _ctx())
    assert ok.accepted is True
    assert ok.reason_code is AggregationReasonCode.COVERED

    # Changed child set (replacement unit) changes the manifest.
    changed = _children("unit/a", "unit/z")
    bad_manifest = aggregator.verify_aggregate(
        original,
        changed,
        _ctx(),
        expected_unit_ids=("unit/a", "unit/b"),
    )
    assert bad_manifest.accepted is False
    assert bad_manifest.reason_code in {
        AggregationReasonCode.MISSING_CHILD,
        AggregationReasonCode.CHANGED_MANIFEST,
    }

    # Same unit IDs but mutated verification digest yields a different root
    # while still matching the expected ID set → old aggregate.
    mutated = (
        _child("unit/a"),
        _child(
            "unit/b",
            verification_digest="sha256:" + ("ff" * 32),
            proof_object_cid="sha256:" + ("ee" * 32),
        ),
    )
    old = aggregator.verify_aggregate(
        original,
        mutated,
        _ctx(),
        expected_unit_ids=("unit/a", "unit/b"),
    )
    assert old.accepted is False
    assert old.reason_code in {
        AggregationReasonCode.OLD_AGGREGATE,
        AggregationReasonCode.CHANGED_MANIFEST,
    }
    assert old.expected_aggregate_root == original.aggregate_root
    assert old.observed_aggregate_root is not None
    assert old.observed_aggregate_root != original.aggregate_root


def test_reordered_and_duplicate_reject_on_verify() -> None:
    kids = _children("unit/a", "unit/b", "unit/c")
    aggregator = ProofAggregator(prefer_recursive=False)
    original = aggregator.aggregate(
        kids, _ctx(), expected_unit_ids=("unit/a", "unit/b", "unit/c")
    )

    reordered = aggregator.verify_aggregate(
        original,
        _children("unit/c", "unit/a", "unit/b"),
        _ctx(),
    )
    assert reordered.accepted is False
    assert reordered.reason_code is AggregationReasonCode.REORDERED_CHILD

    duplicates = aggregator.verify_aggregate(
        original,
        (_child("unit/a"), _child("unit/a"), _child("unit/b")),
        _ctx(),
    )
    assert duplicates.accepted is False
    assert duplicates.reason_code is AggregationReasonCode.DUPLICATE_CHILD


def test_failed_child_rejects_on_verify() -> None:
    kids = _children("unit/a", "unit/b")
    aggregator = ProofAggregator(prefer_recursive=False)
    original = aggregator.aggregate(kids, _ctx())
    failed = aggregator.verify_aggregate(
        original,
        (_child("unit/a"), _child("unit/b", terminal_status="proof_failed")),
        _ctx(),
    )
    assert failed.accepted is False
    assert failed.reason_code is AggregationReasonCode.FAILED_CHILD


def test_without_capability_defaults_to_manifest_not_recursive() -> None:
    kids = _children("unit/a", "unit/b")
    result = aggregate_verified_units(kids, _ctx())
    assert isinstance(result, ManifestAggregationResult)
    assert result.mode is AggregationMode.MANIFEST_AGGREGATION
    assert result.recursive_verification is False
    assert "not recursive" in result.message.casefold() or (
        "recursive verification" in result.does_not_establish.casefold()
    )


def test_recursive_claims_only_when_backend_verifies_children() -> None:
    kids = _children("unit/a", "unit/b")
    capability = _recursive_capability()
    assert capability.recursive_verification is True
    assert capability.aggregation_disposition is AggregationDisposition.RECURSIVE_VERIFICATION

    backend = HermeticRecursiveAggregationBackend()
    result = aggregate_verified_units(
        kids,
        _ctx(),
        capability=capability,
        recursive_backend=backend,
        prefer_recursive=True,
    )
    assert isinstance(result, RecursiveAggregationResult)
    assert result.outcome is AggregationOutcome.AGGREGATED
    assert result.mode is AggregationMode.RECURSIVE_VERIFICATION
    assert result.recursive_verification is True
    assert result.children_backend_verified is True
    assert result.test_execution_directly_proven is False
    assert result.evidence_subset == RECURSIVE_AGGREGATION_EVIDENCE
    assert result.aggregate_root is not None
    assert result.child_root is not None
    assert result.recursive_proof_digest is not None
    assert "backend verified every child" in result.message.casefold()

    # Backend refuses child verification → no recursive claim.
    refusing = HermeticRecursiveAggregationBackend(verify_children=False)
    denied = aggregate_verified_units(
        kids,
        _ctx(),
        capability=capability,
        recursive_backend=refusing,
        prefer_recursive=True,
    )
    assert isinstance(denied, RecursiveAggregationResult)
    assert denied.outcome is AggregationOutcome.REJECTED
    assert denied.recursive_verification is False
    assert denied.children_backend_verified is False
    assert AggregationReasonCode.CHILD_NOT_VERIFIED.value in denied.reason_codes

    # Backend fails a specific child.
    partial = HermeticRecursiveAggregationBackend(fail_child_ids=frozenset({"unit/b"}))
    partial_result = aggregate_verified_units(
        kids,
        _ctx(),
        capability=capability,
        recursive_backend=partial,
    )
    assert partial_result.outcome is AggregationOutcome.REJECTED
    assert partial_result.recursive_verification is False
    assert AggregationReasonCode.CHILD_NOT_VERIFIED.value in partial_result.reason_codes


def test_forced_recursive_without_admission_raises() -> None:
    kids = _children("unit/a")
    with pytest.raises(AggregationError, match="recursive mode forced"):
        aggregate_verified_units(
            kids,
            _ctx(),
            force_mode=AggregationMode.RECURSIVE_VERIFICATION,
        )


def test_receipt_aggregation_states_signer_trust_not_execution() -> None:
    kids = (
        _child(
            "unit/a",
            evidence_class="SignedExecutionReceipt",
            terminal_status="signed_assertion_verified",
            signer_id="signer:allowlisted",
        ),
        _child(
            "unit/b",
            evidence_class="ReceiptAggregationZkProof",
            terminal_status="proved",
            signer_id="signer:allowlisted",
        ),
    )
    result = aggregate_verified_units(kids, _ctx(), prefer_recursive=False)
    assert isinstance(result, ManifestAggregationResult)
    assert result.succeeded is True
    assert result.receipt_aggregation is True
    assert result.signer_trust_stated is True
    assert result.establishes == RECEIPT_AGGREGATION_ESTABLISHES
    assert result.does_not_establish == RECEIPT_AGGREGATION_DOES_NOT_ESTABLISH
    assert "signer trust" in result.establishes.casefold()
    assert "test execution" in result.does_not_establish.casefold()
    assert result.test_execution_directly_proven is False
    # Must not claim underlying execution in establishes.
    for token in ("tests executed", "tests ran", "pytest"):
        assert token not in result.establishes.casefold()


def test_receipt_overclaim_cannot_construct_result() -> None:
    with pytest.raises(AggregationError, match="test execution"):
        ManifestAggregationResult(
            schema="x",
            evidence_subset=MANIFEST_AGGREGATION_EVIDENCE,
            outcome=AggregationOutcome.AGGREGATED,
            mode=AggregationMode.MANIFEST_AGGREGATION,
            aggregate_root="sha256:" + ("aa" * 32),
            manifest_root="sha256:" + ("bb" * 32),
            child_count=1,
            child_unit_ids=("unit/a",),
            child_digests=("sha256:" + ("cc" * 32),),
            category_roots={"unit_test": "sha256:" + ("dd" * 32)},
            batch_nodes=(),
            category_nodes=(),
            repository_node=None,
            rebuilt_categories=("unit_test",),
            repository_id=_REPO,
            environment_cid=_ENV,
            policy_cid=_POLICY,
            establishes="underlying tests ran and completed successfully",
            does_not_establish="recursive verification of child proofs",
            recursive_verification=False,
            children_individually_verified=True,
            test_execution_directly_proven=False,
            receipt_aggregation=True,
            signer_trust_stated=True,
            reason_codes=("covered",),
            message="bad",
        )


def test_bounded_fan_in_builds_batch_category_repository_levels() -> None:
    kids = _children(
        "unit/a",
        "unit/b",
        "unit/c",
        "unit/d",
        "unit/e",
        category="integration_test",
    )
    result = aggregate_verified_units(
        kids, _ctx(), fan_in=2, prefer_recursive=False
    )
    assert isinstance(result, ManifestAggregationResult)
    assert result.succeeded is True
    # 5 leaves / fan_in 2 → 3 batches
    assert len(result.batch_nodes) == 3
    assert all(node.level.value == "batch" for node in result.batch_nodes)
    assert all(len(node.child_ids) <= 2 for node in result.batch_nodes)
    assert len(result.category_nodes) == 1
    assert result.category_nodes[0].level.value == "category"
    assert result.repository_node is not None
    assert result.repository_node.level.value == "repository"
    assert result.repository_node.unit_ids == result.child_unit_ids


def test_affected_categories_recorded_for_partial_rebuild() -> None:
    kids = (
        _child("unit/a", category="unit_test"),
        _child("unit/b", category="unit_test"),
        _child("unit/c", category="static_analysis"),
    )
    result = aggregate_verified_units(
        kids,
        _ctx(),
        prefer_recursive=False,
        affected_unit_ids=("unit/c",),
    )
    assert isinstance(result, ManifestAggregationResult)
    assert result.succeeded is True
    assert result.rebuilt_categories == ("static_analysis",)
    # Full repository still binds both categories.
    assert set(result.category_roots) == {"static_analysis", "unit_test"}


def test_context_mismatch_rejects() -> None:
    kids = (
        _child("unit/a"),
        _child("unit/b", environment_cid="sha256:" + ("99" * 32)),
    )
    result = aggregate_verified_units(kids, _ctx(), prefer_recursive=False)
    assert result.outcome is AggregationOutcome.REJECTED
    assert AggregationReasonCode.CONTEXT_MISMATCH.value in result.reason_codes


def test_empty_children_reject() -> None:
    result = aggregate_verified_units((), _ctx(), prefer_recursive=False)
    assert result.outcome is AggregationOutcome.REJECTED
    assert AggregationReasonCode.EMPTY_CHILDREN.value in result.reason_codes


def test_recursive_verify_round_trip() -> None:
    kids = _children("unit/a", "unit/b")
    capability = _recursive_capability()
    backend = HermeticRecursiveAggregationBackend()
    aggregator = ProofAggregator(
        capability=capability,
        recursive_backend=backend,
        prefer_recursive=True,
    )
    original = aggregator.aggregate(kids, _ctx())
    assert isinstance(original, RecursiveAggregationResult)
    assert original.succeeded is True

    ok = aggregator.verify_aggregate(original, kids, _ctx())
    assert ok.accepted is True

    mutated = (
        _child("unit/a"),
        _child(
            "unit/b",
            verification_digest="sha256:" + ("ab" * 32),
            proof_object_cid="sha256:" + ("cd" * 32),
        ),
    )
    stale = aggregator.verify_aggregate(original, mutated, _ctx())
    assert stale.accepted is False
    assert stale.reason_code in {
        AggregationReasonCode.OLD_AGGREGATE,
        AggregationReasonCode.CHANGED_MANIFEST,
    }


def test_facade_accepts_mapping_context() -> None:
    kids = _children("unit/a")
    result = aggregate_verified_units(
        kids,
        {
            "repository_id": _REPO,
            "environment_cid": _ENV,
            "policy_cid": _POLICY,
        },
        prefer_recursive=False,
    )
    assert result.succeeded is True
    assert result.repository_id == _REPO
