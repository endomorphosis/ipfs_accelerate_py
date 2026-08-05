"""Tests for closeout analyzer/population/quorum premise producers."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.validation.proof_test_reuse_closeout_materializer import (
    CloseoutMaterializerIdentity,
)
from ipfs_accelerate_py.agent_supervisor.validation.proof_test_reuse_closeout_premise_producers import (
    produce_adversarial_population_inputs,
    produce_analyzer_inputs,
    produce_closeout_premises,
    produce_quorum_inputs,
)
from ipfs_accelerate_py.agent_supervisor.validation.proof_test_reuse_goal_evidence import (
    REQUIRED_ADVERSARIAL_POPULATIONS,
    REQUIRED_ANALYZER_CHANNELS,
    REQUIRED_QUORUM_MEMBERS,
)

NOW_MS = 1_800_000_000_000


def _identity() -> CloseoutMaterializerIdentity:
    return CloseoutMaterializerIdentity(
        repository_id="lift_coding/proof-backed-test-reuse",
        repository_state_cid="git-commit:" + ("a" * 40),
        git_commit_id="a" * 40,
        git_tree_id="b" * 40,
        gitlink_state_cid="baguqeera-gitlinks-current",
        repository_forest_cid="baguqeera-forest-current",
        dirty=False,
        dirty_overlay_cid="cid:dirty-overlay:none",
        objective_revision="objective-sha256:test",
        policy_cid="policy:test",
        capability_cid="capability:test",
        verifying_key_cid="key:test",
        circuit_cid="circuit:test",
    )


def test_analyzer_probes_emit_all_required_channels() -> None:
    analyzers, probes = produce_analyzer_inputs(_identity(), now_ms=NOW_MS)
    assert {p.analyzer_id for p in probes} == set(REQUIRED_ANALYZER_CHANNELS)
    assert all(p.healthy for p in probes)
    assert {row["analyzer_id"] for row in analyzers} == set(REQUIRED_ANALYZER_CHANNELS)
    for row in analyzers:
        assert row["healthy"] is True
        assert row["exhaustive"] is True
        assert row["conclusive"] is True
        assert row["git_tree_id"] == "b" * 40


def test_population_inputs_from_supporting_validations() -> None:
    identity = _identity()
    receipts = [
        {
            "task_id": "PTR-090",
            "goal_id": "PTR-G100",
            "passed": True,
            "proof_reuse_mode": "off",
            "skipped_count": 0,
            "git_commit_id": identity.git_commit_id,
            "git_tree_id": identity.git_tree_id,
            "validation_receipt_cid": "baguqeera-val-090",
        },
        {
            "task_id": "PTR-050",
            "goal_id": "PTR-G050",
            "passed": True,
            "proof_reuse_mode": "off",
            "skipped_count": 0,
            "git_commit_id": identity.git_commit_id,
            "git_tree_id": identity.git_tree_id,
            "validation_receipt_cid": "baguqeera-val-050",
        },
        {
            "task_id": "PTR-070",
            "goal_id": "PTR-G070",
            "passed": True,
            "proof_reuse_mode": "off",
            "skipped_count": 0,
            "git_commit_id": identity.git_commit_id,
            "git_tree_id": identity.git_tree_id,
            "validation_receipt_cid": "baguqeera-val-070",
        },
    ]
    populations = produce_adversarial_population_inputs(
        identity, validation_receipts=receipts, now_ms=NOW_MS
    )
    ids = {row["population_id"] for row in populations}
    assert ids == set(REQUIRED_ADVERSARIAL_POPULATIONS)
    for row in populations:
        assert row["passed"] is True
        assert row["false_skips"] == 0
        assert row["supporting_validation_count"] >= 1


def test_quorum_requires_independent_healthy_analyzers() -> None:
    identity = _identity()
    analyzers, _probes = produce_analyzer_inputs(identity, now_ms=NOW_MS)
    quorum = produce_quorum_inputs(
        identity, analyzer_inputs=analyzers, now_ms=NOW_MS
    )
    assert len(quorum) >= REQUIRED_QUORUM_MEMBERS
    member_ids = {row["member_id"] for row in quorum}
    channels = {row["evidence_channel"] for row in quorum}
    receipts = {row["receipt_cid"] for row in quorum}
    assert len(member_ids) == len(quorum)
    assert len(channels) == len(quorum)
    assert len(receipts) == len(quorum)
    assert all(row["healthy"] and row["exhaustive"] for row in quorum)


def test_produce_closeout_premises_bundle() -> None:
    identity = _identity()
    receipts = [
        {
            "task_id": f"PTR-{i:03d}",
            "goal_id": goal,
            "passed": True,
            "proof_reuse_mode": "off",
            "skipped_count": 0,
            "git_commit_id": identity.git_commit_id,
            "git_tree_id": identity.git_tree_id,
            "validation_receipt_cid": f"baguqeera-val-{i}",
        }
        for i, goal in enumerate(
            ("PTR-G100", "PTR-G070", "PTR-G050", "PTR-G040", "PTR-G090"), start=1
        )
    ]
    bundle = produce_closeout_premises(
        identity, validation_receipts=receipts, now_ms=NOW_MS
    )
    assert len(bundle.analyzer_inputs) == len(REQUIRED_ANALYZER_CHANNELS)
    assert len(bundle.population_inputs) == len(REQUIRED_ADVERSARIAL_POPULATIONS)
    assert len(bundle.quorum_inputs) >= REQUIRED_QUORUM_MEMBERS
    assert bundle.authority is False
