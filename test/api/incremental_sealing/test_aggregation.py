"""IPS-036: bounded manifest aggregation and capability-gated recursion."""

from __future__ import annotations

from types import SimpleNamespace

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.aggregation import (
    AGGREGATION_LABEL_MANIFEST,
    AGGREGATION_LABEL_RECURSIVE,
    MANIFEST_EVIDENCE,
    RECURSIVE_EVIDENCE,
    AggregationMode,
    AggregationReason,
    ManifestAggregationResult,
    ProofAggregator,
    RecursiveAggregationResult,
    VerifiedUnit,
    aggregate_verified_units,
)


def _unit(unit_id: str, **overrides: object) -> VerifiedUnit:
    payload = {
        "unit_id": unit_id,
        "proof_object_cid": "sha256:" + ("ab" * 32),
        "category": "unit_test",
        "terminal_status": "integrity_verified",
        "repository_state_cid": "sha256:" + ("11" * 32),
        "environment_cid": "sha256:" + ("22" * 32),
    }
    payload.update(overrides)
    return VerifiedUnit(**payload)  # type: ignore[arg-type]


def _units() -> tuple[VerifiedUnit, ...]:
    return (
        _unit("unit/a"),
        _unit("unit/b"),
        _unit("unit/c"),
    )


def test_evidence_subsets_and_labels() -> None:
    assert MANIFEST_EVIDENCE == "ips/manifest-aggregation@1"
    assert RECURSIVE_EVIDENCE == "ips/recursive-aggregation@1"
    assert AGGREGATION_LABEL_MANIFEST == "manifest_aggregation"
    assert AGGREGATION_LABEL_RECURSIVE == "recursive_verification"


def test_manifest_aggregation_binds_identities_and_does_not_recurse() -> None:
    result = aggregate_verified_units(
        _units(),
        expected_unit_ids=("unit/a", "unit/b", "unit/c"),
    )
    assert isinstance(result, ManifestAggregationResult)
    assert result.accepted is True
    assert result.mode is AggregationMode.MANIFEST
    assert result.recursively_verifies_children is False
    assert result.claims_test_execution is False
    assert result.label == AGGREGATION_LABEL_MANIFEST
    assert result.child_unit_ids == ("unit/a", "unit/b", "unit/c")
    assert result.child_count == 3
    assert result.child_root.startswith("sha256:")
    assert result.signer_trust == "signer_allowlist_only"
    assert result.affected_levels == ("leaf", "batch", "category", "repository")


def test_missing_duplicate_reordered_and_failed_children_reject() -> None:
    missing = aggregate_verified_units(
        (_unit("unit/a"), _unit("unit/b")),
        expected_unit_ids=("unit/a", "unit/b", "unit/c"),
    )
    assert missing.accepted is False
    assert missing.reason is AggregationReason.MISSING_CHILD

    duplicate = aggregate_verified_units(
        (_unit("unit/a"), _unit("unit/a")),
        expected_unit_ids=("unit/a", "unit/a"),
    )
    assert duplicate.accepted is False
    assert duplicate.reason is AggregationReason.DUPLICATE_CHILD

    reordered = aggregate_verified_units(
        (_unit("unit/b"), _unit("unit/a")),
        expected_unit_ids=("unit/a", "unit/b"),
    )
    assert reordered.accepted is False
    assert reordered.reason is AggregationReason.REORDERED_CHILDREN

    failed = aggregate_verified_units((_unit("unit/a", failed=True),))
    assert failed.accepted is False
    assert failed.reason is AggregationReason.FAILED_CHILD


def test_changed_manifest_and_stale_aggregate_reject() -> None:
    first = aggregate_verified_units(_units())
    assert isinstance(first, ManifestAggregationResult)
    changed = aggregate_verified_units(_units(), expected_root="sha256:" + ("ff" * 32))
    assert changed.accepted is False
    assert changed.reason is AggregationReason.CHANGED_MANIFEST

    stale = aggregate_verified_units(
        _units(),
        previous_root="sha256:" + ("ee" * 32),
        expected_root="sha256:" + ("ff" * 32),
    )
    assert stale.accepted is False
    assert stale.reason is AggregationReason.STALE_AGGREGATE


def test_receipt_aggregation_does_not_claim_test_execution() -> None:
    result = aggregate_verified_units(
        _units(),
        receipt_claim="receipt aggregation proves tests executed",
    )
    assert result.accepted is False
    assert result.reason is AggregationReason.EXECUTION_OVERCLAIM
    if isinstance(result, ManifestAggregationResult):
        assert result.claims_test_execution is False


def test_recursion_requires_successful_capability_probe() -> None:
    denied = aggregate_verified_units(
        _units(),
        prefer_recursion=True,
        capability=SimpleNamespace(
            recursive_verification=False,
            backend_id="provekit",
        ),
    )
    assert isinstance(denied, RecursiveAggregationResult)
    assert denied.accepted is False
    assert denied.recursively_verifies_children is False
    assert denied.reason is AggregationReason.RECURSION_NOT_ADMITTED

    admitted = aggregate_verified_units(
        _units(),
        prefer_recursion=True,
        capability=SimpleNamespace(
            recursive_verification=True,
            backend_id="provekit",
        ),
    )
    assert isinstance(admitted, RecursiveAggregationResult)
    assert admitted.accepted is True
    assert admitted.recursively_verifies_children is True
    assert admitted.label == AGGREGATION_LABEL_RECURSIVE
    assert admitted.backend_id == "provekit"


def test_aggregation_is_deterministic_and_order_sensitive() -> None:
    first = ProofAggregator().aggregate_verified_units(_units())
    second = ProofAggregator().aggregate_verified_units(_units())
    assert first.to_canonical() == second.to_canonical()
    swapped = ProofAggregator().aggregate_verified_units(
        (_unit("unit/c"), _unit("unit/b"), _unit("unit/a"))
    )
    assert swapped.child_root != first.child_root
