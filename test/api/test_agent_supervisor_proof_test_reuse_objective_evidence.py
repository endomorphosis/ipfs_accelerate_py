"""Bound goal evidence and completion-gate assembly tests (PTR-120)."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.objectives.goal_completion import (
    validate_completion_evidence,
)
from ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon import (
    OBJECTIVE_COMPLETION_EVIDENCE_ARTIFACT_SCHEMA,
    load_goal_completion_evidence_records,
    load_goal_completion_gate_records,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    canonical_json_bytes,
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.validation.proof_test_reuse_goal_evidence import (
    CoverageStatus,
    GoalQuorumMember,
    ProofReuseAnalyzerReceipt,
    AcceptanceCoverageReceipt,
    goal_requirements_by_id,
    load_objective_goals,
)
from ipfs_accelerate_py.agent_supervisor.validation.proof_test_reuse_objective_contracts import (
    CanonicalPremiseBlock,
    ObjectiveArtifactStore,
    ProofTestReuseCompletionArtifact,
    require_verified_cid,
    verify_retained_bytes,
)
from ipfs_accelerate_py.agent_supervisor.validation.proof_test_reuse_objective_evidence import (
    ANALYZER_HEALTH_ARTIFACT_RELATIVE,
    COVERAGE_ARTIFACT_RELATIVE,
    DEFAULT_CHANNEL_PROOF_REVISION,
    DEFAULT_PRODUCER_CHANNEL,
    EVIDENCE_ARTIFACT_RELATIVE,
    EXHAUSTION_QUORUM_ARTIFACT_RELATIVE,
    GATE_ARTIFACT_RELATIVE,
    GOAL_COMPLETION_ARTIFACT_GAP_INTERFACE,
    PRODUCING_TASK_ID,
    PROOF_TEST_REUSE_OBJECTIVE_EVIDENCE_ASSEMBLER_INTERFACE,
    PROOF_TEST_REUSE_OBJECTIVE_EVIDENCE_BUNDLE_INTERFACE,
    GoalAssemblyIdentity,
    GoalCompletionArtifactGap,
    ObjectiveEvidenceGapKind,
    ProofTestReuseObjectiveEvidenceAssembler,
    ProofTestReuseObjectiveEvidenceBundle,
    ProofTestReuseObjectiveEvidenceError,
    assemble_objective_evidence,
    load_and_validate_written_artifacts,
    replay_premise_blocks,
)

NOW = 1_786_000_000.0
NOW_MS = int(NOW * 1000)
FRESH_UNTIL_MS = NOW_MS + 60_000

GIT_TREE = "a" * 40
FOREST = "baguqeera" + "f" * 50
COMPLETION_TREE = "sha256:" + "b" * 64
REPO_ID = "repository:sha256:" + "c" * 64
OBJECTIVE_REV = "baguqeera" + "1" * 50
ANALYZER_REV = "baguqeera" + "2" * 50
CONFIG_REV = "baguqeera" + "3" * 50
POLICY_REV = "baguqeera" + "4" * 50
CAPABILITY_REV = "baguqeera" + "5" * 50
CIRCUIT_REV = "baguqeera" + "6" * 50
KEY_REV = "baguqeera" + "7" * 50
STATE = "baguqeera" + "s" * 50
COMMIT = "d" * 40
GITLINKS = "baguqeera" + "g" * 50
OVERLAY = "baguqeera" + "o" * 50

MINI_HEAP = """
## PTR-G010 Contracts

- Status: active
- Parent: PTR-G000
- Evidence: ptr/test-execution-contracts@1, ptr/reuse-authority-policy@1
- Acceptance criteria: ptr/test-execution-contracts@1; ptr/reuse-authority-policy@1
- Acceptance: contracts reject nonfinite inputs
- Validation: pytest -q
- Gap task: PTR-001
- Refinement: separate schemas
- Embedding query: contracts
- AST query: dataclasses
- Goal: Define contracts
- Outputs: contracts.py
- Fib priority: 2
- Priority: P0
- Track: foundation
- Bundle: foundation

## PTR-G020 Identity

- Status: active
- Parent: PTR-G000
- Evidence: ptr/test-locator-key@1
- Acceptance criteria: ptr/test-locator-key@1
- Acceptance: CIDs reproduce
- Validation: pytest -q
- Gap task: PTR-010
- Refinement: keys
- Embedding query: identity
- AST query: nodeid
- Goal: Canonical identity
- Outputs: identity.py
- Fib priority: 2
- Priority: P0
- Track: identity
- Bundle: identity
"""

OBJECTIVES_PATH = (
    Path(__file__).resolve().parents[4]
    / "implementation_plan"
    / "docs"
    / "46-proof-backed-test-reuse.objectives.md"
)
if not OBJECTIVES_PATH.is_file():
    for parent in Path(__file__).resolve().parents:
        candidate = (
            parent
            / "implementation_plan"
            / "docs"
            / "46-proof-backed-test-reuse.objectives.md"
        )
        if candidate.is_file():
            OBJECTIVES_PATH = candidate
            break


def _identity(**overrides: Any) -> GoalAssemblyIdentity:
    values: dict[str, Any] = {
        "repository_id": REPO_ID,
        "git_tree_id": GIT_TREE,
        "repository_forest_cid": FOREST,
        "objective_completion_tree_id": COMPLETION_TREE,
        "objective_revision": OBJECTIVE_REV,
        "analyzer_revision": ANALYZER_REV,
        "configuration_revision": CONFIG_REV,
        "policy_revision": POLICY_REV,
        "capability_revision": CAPABILITY_REV,
        "circuit_revision": CIRCUIT_REV,
        "verifying_key_revision": KEY_REV,
        "git_commit_id": COMMIT,
        "gitlink_state_cid": GITLINKS,
        "repository_state_cid": STATE,
    }
    values.update(overrides)
    return GoalAssemblyIdentity(**values)


def _retained(payload: dict[str, Any]) -> tuple[str, str]:
    raw = canonical_json_bytes(dict(payload))
    encoded = base64.b64encode(raw).decode("ascii")
    digest_payload = {
        "kind": "retained_validation_bytes",
        "sha256_b64": base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii"),
        "size": len(raw),
    }
    return encoded, content_identity(digest_payload)


def _coverage(
    requirement_id: str,
    goal_id: str,
    **overrides: Any,
) -> AcceptanceCoverageReceipt:
    retained_payload = {
        "schema": "coverage-validation@1",
        "requirement_id": requirement_id,
        "goal_id": goal_id,
        "status": "passed",
        "n": 1,
    }
    b64, cid = _retained(retained_payload)
    values: dict[str, Any] = {
        "requirement_id": requirement_id,
        "goal_id": goal_id,
        "status": CoverageStatus.VERIFIED,
        "producer_channel": "goal-assurance",
        "channel_proof_revision": "channel:ptr-111@1",
        "repository_id": REPO_ID,
        "repository_state_cid": STATE,
        "git_commit_id": COMMIT,
        "git_tree_id": GIT_TREE,
        "gitlink_state_cid": GITLINKS,
        "repository_forest_cid": FOREST,
        "dirty": False,
        "dirty_overlay_cid": OVERLAY,
        "policy_cid": POLICY_REV,
        "capability_cid": CAPABILITY_REV,
        "verifying_key_cid": KEY_REV,
        "circuit_cid": CIRCUIT_REV,
        "objective_revision": OBJECTIVE_REV,
        "observed_at_ms": NOW_MS - 1_000,
        "fresh_until_ms": FRESH_UNTIL_MS,
        "retained_validation_bytes_b64": b64,
        "retained_validation_cid": cid,
        "locally_verified": True,
        "validation_passed": True,
    }
    values.update(overrides)
    return AcceptanceCoverageReceipt(**values)


def _analyzer(analyzer_id: str = "static-dependency", **overrides: Any) -> ProofReuseAnalyzerReceipt:
    retained_payload = {
        "schema": "analyzer-health@1",
        "analyzer_id": analyzer_id,
        "healthy": True,
        "exhaustive": True,
    }
    b64, cid = _retained(retained_payload)
    values: dict[str, Any] = {
        "analyzer_id": analyzer_id,
        "producer_channel": f"analyzer:{analyzer_id}",
        "channel_proof_revision": "channel:ptr-111@1",
        "repository_id": REPO_ID,
        "git_tree_id": GIT_TREE,
        "repository_forest_cid": FOREST,
        "objective_revision": OBJECTIVE_REV,
        "observed_at_ms": NOW_MS - 500,
        "fresh_until_ms": FRESH_UNTIL_MS,
        "retained_validation_bytes_b64": b64,
        "retained_validation_cid": cid,
        "healthy": True,
        "exhaustive": True,
        "conclusive": True,
    }
    values.update(overrides)
    return ProofReuseAnalyzerReceipt(**values)


def _quorum_members() -> tuple[GoalQuorumMember, GoalQuorumMember]:
    return (
        GoalQuorumMember(
            member_id="exhaustive-scan",
            evidence_channel="static-dependency-exhaustive",
            receipt_cid="baguqeera-quorum-exhaustive-001",
            healthy=True,
            exhaustive=True,
            conclusive=True,
            fresh=True,
            uncontradicted=True,
            observed_at_ms=NOW_MS - 500,
            fresh_until_ms=FRESH_UNTIL_MS,
        ),
        GoalQuorumMember(
            member_id="audit-scan",
            evidence_channel="independent-audit",
            receipt_cid="baguqeera-quorum-audit-001",
            healthy=True,
            exhaustive=True,
            conclusive=True,
            fresh=True,
            uncontradicted=True,
            observed_at_ms=NOW_MS - 400,
            fresh_until_ms=FRESH_UNTIL_MS,
        ),
    )


def _coverage_for_heap(heap: str) -> list[AcceptanceCoverageReceipt]:
    population = goal_requirements_by_id(heap)
    receipts: list[AcceptanceCoverageReceipt] = []
    for goal_id, reqs in population.items():
        for req in reqs:
            receipts.append(_coverage(req, goal_id))
    return receipts


def _assemble(
    heap: str = MINI_HEAP,
    *,
    write_root: Path | None = None,
    store: ObjectiveArtifactStore | None = None,
    coverage_receipts: list[AcceptanceCoverageReceipt] | None = None,
    analyzer_receipts: list[ProofReuseAnalyzerReceipt] | None = None,
    quorum_members: tuple[GoalQuorumMember, ...] | None = None,
    **identity_overrides: Any,
) -> ProofTestReuseObjectiveEvidenceBundle:
    return assemble_objective_evidence(
        heap,
        identity=_identity(**identity_overrides),
        coverage_receipts=coverage_receipts
        if coverage_receipts is not None
        else _coverage_for_heap(heap),
        analyzer_receipts=analyzer_receipts
        if analyzer_receipts is not None
        else [_analyzer()],
        quorum_members=quorum_members
        if quorum_members is not None
        else _quorum_members(),
        write_root=write_root,
        store=store,
        clock=lambda: NOW,
        now_ms=NOW_MS,
    )


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------


def test_interfaces_and_producing_task_are_stable() -> None:
    assert (
        PROOF_TEST_REUSE_OBJECTIVE_EVIDENCE_ASSEMBLER_INTERFACE
        == "ProofTestReuseObjectiveEvidenceAssembler@1"
    )
    assert (
        PROOF_TEST_REUSE_OBJECTIVE_EVIDENCE_BUNDLE_INTERFACE
        == "ProofTestReuseObjectiveEvidenceBundle@1"
    )
    assert GOAL_COMPLETION_ARTIFACT_GAP_INTERFACE == "GoalCompletionArtifactGap@1"
    assert PRODUCING_TASK_ID == "PTR-120"
    assert DEFAULT_PRODUCER_CHANNEL == "objective-evidence-assembler"
    assert DEFAULT_CHANNEL_PROOF_REVISION == "channel:ptr-120@1"


# ---------------------------------------------------------------------------
# Happy path: one binding + exact acceptance population per goal
# ---------------------------------------------------------------------------


def test_assembler_emits_one_binding_and_exact_acceptance_population() -> None:
    bundle = _assemble()
    goals = load_objective_goals(MINI_HEAP)
    expected_population = goal_requirements_by_id(MINI_HEAP)

    assert bundle.goal_ids == tuple(goal.goal_id for goal in goals)
    assert set(bundle.bindings) == set(bundle.goal_ids)
    assert set(bundle.acceptance_population) == set(bundle.goal_ids)
    assert len(bundle.bindings) == len(bundle.goal_ids)

    for goal_id in bundle.goal_ids:
        binding = bundle.binding_for(goal_id)
        assert binding.goal_id == goal_id
        assert binding.repository_id == REPO_ID
        assert binding.git_tree_id == GIT_TREE
        assert binding.repository_forest_cid == FOREST
        assert binding.objective_completion_tree_id == COMPLETION_TREE
        assert binding.objective_revision == OBJECTIVE_REV
        assert binding.analyzer_revision == ANALYZER_REV
        assert binding.policy_revision == POLICY_REV
        assert binding.circuit_revision == CIRCUIT_REV
        assert binding.verifying_key_revision == KEY_REV
        assert bundle.acceptance_for(goal_id) == expected_population[goal_id]
        # Exactly one artifact per acceptance criterion when fully assured.
        arts = bundle.completion_artifacts[goal_id]
        assert len(arts) == len(expected_population[goal_id])
        assert {item.acceptance_criterion for item in arts} == set(
            expected_population[goal_id]
        )

    assert bundle.authoritative
    assert not bundle.gaps


def test_full_heap_has_twelve_goals_and_exact_acceptance_population() -> None:
    assert OBJECTIVES_PATH.is_file(), f"missing heap at {OBJECTIVES_PATH}"
    heap_text = OBJECTIVES_PATH.read_text(encoding="utf-8")
    expected = goal_requirements_by_id(heap_text)
    assert len(expected) == 12

    bundle = _assemble(heap_text)
    assert len(bundle.goal_ids) == 12
    assert set(bundle.goal_ids) == set(expected)
    for goal_id, reqs in expected.items():
        assert bundle.acceptance_for(goal_id) == reqs
        assert len(bundle.bindings[goal_id].binding_cid) > 10
        assert len(bundle.completion_artifacts[goal_id]) == len(reqs)
    assert bundle.authoritative
    assert bundle.producing_task_id == "PTR-120"


# ---------------------------------------------------------------------------
# Premise replay before write; atomic write + readback rehash
# ---------------------------------------------------------------------------


def test_premises_are_replayed_by_canonical_cid_before_write(tmp_path: Path) -> None:
    store = ObjectiveArtifactStore(local_root=tmp_path / "store")
    bundle = _assemble(write_root=tmp_path / "state", store=store)

    for goal_id in bundle.goal_ids:
        for artifact in bundle.completion_artifacts[goal_id]:
            replayed = replay_premise_blocks(artifact.premise_blocks)
            assert len(replayed) == len(artifact.premise_blocks)
            for block in replayed:
                require_verified_cid(block.cid, block.data)
                assert verify_retained_bytes(block.cid, block.data)
            # Nested contract replay also succeeds.
            artifact.replay_premises()
            assert verify_retained_bytes(
                artifact.artifact_cid, artifact.canonical_bytes()
            )

    assert bundle.store_cids
    for cid in bundle.store_cids:
        loaded = store.get_bytes(cid)
        require_verified_cid(cid, loaded)


def test_atomic_state_root_write_with_readback_and_daemon_round_trip(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state"
    bundle = _assemble(write_root=state)

    assert EVIDENCE_ARTIFACT_RELATIVE in bundle.written_paths
    assert GATE_ARTIFACT_RELATIVE in bundle.written_paths
    assert COVERAGE_ARTIFACT_RELATIVE in bundle.written_paths
    assert ANALYZER_HEALTH_ARTIFACT_RELATIVE in bundle.written_paths
    assert EXHAUSTION_QUORUM_ARTIFACT_RELATIVE in bundle.written_paths

    evidence_path = state / EVIDENCE_ARTIFACT_RELATIVE
    gate_path = state / GATE_ARTIFACT_RELATIVE
    evidence_payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence_payload["schema"] == OBJECTIVE_COMPLETION_EVIDENCE_ARTIFACT_SCHEMA
    assert "binding" in evidence_payload
    assert "goals" in evidence_payload

    evidence_records = load_goal_completion_evidence_records(evidence_path)
    gate_records = load_goal_completion_gate_records(gate_path, repo_root=state)
    assert set(evidence_records) == set(bundle.goal_ids)
    assert set(gate_records) == set(bundle.goal_ids)

    for goal_id in bundle.goal_ids:
        assert len(evidence_records[goal_id]) == len(bundle.acceptance_for(goal_id))
        for record in evidence_records[goal_id]:
            assert record.repository_id == REPO_ID
            assert record.tree_id == GIT_TREE
            assert record.objective_revision == OBJECTIVE_REV
            assert record.analyzer_version == ANALYZER_REV
            assert record.configuration_revision == CONFIG_REV
            assert record.validation_passed is True
            assert record.producer_channel
            assert record.channel_proof_revision
            result = validate_completion_evidence(
                record,
                repository_tree=GIT_TREE,
                repository_id=REPO_ID,
                objective_revision=OBJECTIVE_REV,
                analyzer_version=ANALYZER_REV,
                configuration_revision=CONFIG_REV,
                require_artifact_binding=True,
                now=__import__("datetime").datetime.fromtimestamp(
                    NOW_MS / 1000.0, tz=__import__("datetime").timezone.utc
                ),
            )
            assert result.valid, (goal_id, result.reason_codes)

        gate = gate_records[goal_id]
        assert isinstance(gate.get("coverage"), dict)
        assert isinstance(gate.get("analyzer_health"), dict)
        assert isinstance(gate.get("exhaustion_quorum"), dict)
        assert gate["coverage"]["verified"] is True
        assert gate["analyzer_health"]["healthy"] is True
        assert gate["exhaustion_quorum"]["satisfied"] is True
        assert gate["exhaustion_quorum"]["member_count"] >= 2
        assert "binding" in gate

    reloaded = load_and_validate_written_artifacts(state, now_ms=NOW_MS)
    for goal_id, results in reloaded["validations"].items():
        assert results
        assert all(item.valid for item in results), (
            goal_id,
            [item.reason_codes for item in results],
        )


# ---------------------------------------------------------------------------
# Gate requirements: fresh coverage, healthy analyzer, two quorum members
# ---------------------------------------------------------------------------


def test_missing_coverage_becomes_bounded_gap_not_success() -> None:
    population = goal_requirements_by_id(MINI_HEAP)
    # Drop one requirement's coverage.
    first_goal = next(iter(population))
    first_req = population[first_goal][0]
    receipts = [
        item
        for item in _coverage_for_heap(MINI_HEAP)
        if item.requirement_id != first_req
    ]
    bundle = _assemble(coverage_receipts=receipts)
    assert not bundle.authoritative
    kinds = {gap.kind for gap in bundle.gaps}
    assert ObjectiveEvidenceGapKind.COVERAGE_MISSING in kinds
    assert any(gap.acceptance_criterion == first_req for gap in bundle.gaps)
    # No success artifact for the missing criterion.
    arts = bundle.completion_artifacts[first_goal]
    assert first_req not in {item.acceptance_criterion for item in arts}


def test_stale_coverage_is_rejected() -> None:
    receipts = _coverage_for_heap(MINI_HEAP)
    stale = [
        _coverage(
            item.requirement_id,
            item.goal_id,
            observed_at_ms=NOW_MS - 10_000,
            fresh_until_ms=NOW_MS - 1,
        )
        for item in receipts
    ]
    bundle = _assemble(coverage_receipts=stale)
    assert not bundle.authoritative
    assert ObjectiveEvidenceGapKind.COVERAGE_STALE in {gap.kind for gap in bundle.gaps}
    assert all(not arts for arts in bundle.completion_artifacts.values())


def test_unhealthy_analyzer_blocks_completion_artifacts() -> None:
    unhealthy = _analyzer(healthy=False, exhaustive=False, conclusive=False)
    # Need valid retained bytes for the unhealthy analyzer still.
    retained_payload = {
        "schema": "analyzer-health@1",
        "analyzer_id": "static-dependency",
        "healthy": False,
    }
    b64, cid = _retained(retained_payload)
    unhealthy = ProofReuseAnalyzerReceipt(
        analyzer_id="static-dependency",
        producer_channel="analyzer:static-dependency",
        channel_proof_revision="channel:ptr-111@1",
        repository_id=REPO_ID,
        git_tree_id=GIT_TREE,
        repository_forest_cid=FOREST,
        objective_revision=OBJECTIVE_REV,
        observed_at_ms=NOW_MS - 500,
        fresh_until_ms=FRESH_UNTIL_MS,
        retained_validation_bytes_b64=b64,
        retained_validation_cid=cid,
        healthy=False,
        exhaustive=False,
        conclusive=False,
    )
    bundle = _assemble(analyzer_receipts=[unhealthy])
    assert not bundle.authoritative
    assert ObjectiveEvidenceGapKind.ANALYZER_UNHEALTHY in {
        gap.kind for gap in bundle.gaps
    }
    assert all(not arts for arts in bundle.completion_artifacts.values())


def test_requires_two_independent_quorum_members() -> None:
    only_one = (
        GoalQuorumMember(
            member_id="only",
            evidence_channel="same",
            receipt_cid="baguqeera-only",
            healthy=True,
            exhaustive=True,
            conclusive=True,
            fresh=True,
            uncontradicted=True,
            observed_at_ms=NOW_MS - 100,
            fresh_until_ms=FRESH_UNTIL_MS,
        ),
    )
    bundle = _assemble(quorum_members=only_one)
    assert not bundle.authoritative
    kinds = {gap.kind for gap in bundle.gaps}
    assert ObjectiveEvidenceGapKind.QUORUM_INSUFFICIENT in kinds
    assert all(not arts for arts in bundle.completion_artifacts.values())

    # Two members with identical independence keys also fail.
    twins = (
        GoalQuorumMember(
            member_id="same",
            evidence_channel="same-channel",
            receipt_cid="baguqeera-same",
            healthy=True,
            exhaustive=True,
            conclusive=True,
            fresh=True,
            uncontradicted=True,
            observed_at_ms=NOW_MS - 100,
            fresh_until_ms=FRESH_UNTIL_MS,
        ),
        GoalQuorumMember(
            member_id="same",
            evidence_channel="same-channel",
            receipt_cid="baguqeera-same",
            healthy=True,
            exhaustive=True,
            conclusive=True,
            fresh=True,
            uncontradicted=True,
            observed_at_ms=NOW_MS - 50,
            fresh_until_ms=FRESH_UNTIL_MS,
        ),
    )
    bundle2 = _assemble(quorum_members=twins)
    assert not bundle2.authoritative
    assert all(not arts for arts in bundle2.completion_artifacts.values())


# ---------------------------------------------------------------------------
# Gaps, self-verification, edit authorization
# ---------------------------------------------------------------------------


def test_gap_records_are_bounded_and_typed() -> None:
    gap = GoalCompletionArtifactGap(
        goal_id="PTR-G010",
        kind=ObjectiveEvidenceGapKind.UNAVAILABLE_INPUT,
        detail="optional capability missing",
        acceptance_criterion="ptr/test-execution-contracts@1",
        observed_at_ms=NOW_MS,
    )
    payload = gap.to_record()
    restored = GoalCompletionArtifactGap.from_dict(payload)
    assert restored.kind is ObjectiveEvidenceGapKind.UNAVAILABLE_INPUT
    assert restored.gap_cid == gap.gap_cid
    assert restored.interface == GOAL_COMPLETION_ARTIFACT_GAP_INTERFACE


def test_artifact_cannot_verify_own_bytes_or_authorize_edits() -> None:
    bundle = _assemble()
    with pytest.raises(ProofTestReuseObjectiveEvidenceError) as exc:
        bundle.verify_own_bytes(b"anything")
    assert exc.value.reason_code is ObjectiveEvidenceGapKind.SELF_VERIFICATION_FORBIDDEN

    with pytest.raises(ProofTestReuseObjectiveEvidenceError) as exc:
        bundle.authorize_edit(path="foo.py", patch="x")
    assert exc.value.reason_code is ObjectiveEvidenceGapKind.EDIT_AUTHORIZATION_FORBIDDEN


def test_bundle_validate_all_completion_evidence_passes() -> None:
    bundle = _assemble()
    results = bundle.validate_all_completion_evidence(now_ms=NOW_MS)
    assert results
    assert all(item.valid for item in results)


def test_completion_artifacts_project_through_contract_as_completion_evidence() -> None:
    bundle = _assemble()
    for goal_id in bundle.goal_ids:
        for artifact in bundle.completion_artifacts[goal_id]:
            assert isinstance(artifact, ProofTestReuseCompletionArtifact)
            projected = artifact.as_completion_evidence()
            assert projected.acceptance_criterion == artifact.acceptance_criterion
            assert projected.repository_id == REPO_ID
            assert projected.tree_id == GIT_TREE
            assert projected.provenance_cid == artifact.artifact_cid
            # Channel-bound daemon projection also validates.
            daemon = bundle._channel_bound_completion_evidence(artifact)
            result = validate_completion_evidence(
                daemon,
                repository_tree=GIT_TREE,
                repository_id=REPO_ID,
                objective_revision=OBJECTIVE_REV,
                analyzer_version=ANALYZER_REV,
                configuration_revision=CONFIG_REV,
                require_artifact_binding=True,
                now=__import__("datetime").datetime.fromtimestamp(
                    NOW_MS / 1000.0, tz=__import__("datetime").timezone.utc
                ),
            )
            assert result.valid, result.reason_codes


def test_assembler_class_interface_surface() -> None:
    assembler = ProofTestReuseObjectiveEvidenceAssembler(identity=_identity())
    assert assembler.interface == PROOF_TEST_REUSE_OBJECTIVE_EVIDENCE_ASSEMBLER_INTERFACE
    assert assembler.producing_task_id == "PTR-120"


def test_identity_rejects_incomplete_binding() -> None:
    with pytest.raises(ProofTestReuseObjectiveEvidenceError):
        GoalAssemblyIdentity(
            repository_id="",
            git_tree_id=GIT_TREE,
            repository_forest_cid=FOREST,
            objective_completion_tree_id=COMPLETION_TREE,
            objective_revision=OBJECTIVE_REV,
            analyzer_revision=ANALYZER_REV,
            configuration_revision=CONFIG_REV,
            policy_revision=POLICY_REV,
            capability_revision=CAPABILITY_REV,
            circuit_revision=CIRCUIT_REV,
            verifying_key_revision=KEY_REV,
        )


def test_coverage_binding_mismatch_is_a_gap() -> None:
    receipts = _coverage_for_heap(MINI_HEAP)
    # Corrupt one receipt's tree binding.
    first = receipts[0]
    mismatched = _coverage(
        first.requirement_id,
        first.goal_id,
        git_tree_id="e" * 40,
    )
    receipts[0] = mismatched
    bundle = _assemble(coverage_receipts=receipts)
    assert ObjectiveEvidenceGapKind.COVERAGE_BINDING_MISMATCH in {
        gap.kind for gap in bundle.gaps
    }


def test_gate_records_embed_completion_evidence_for_combined_loader(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state"
    bundle = _assemble(write_root=state)
    from ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon import (
        completion_evidence_records_from_gate_records,
    )

    embedded = completion_evidence_records_from_gate_records(bundle.gate_records)
    for goal_id in bundle.goal_ids:
        assert goal_id in embedded
        assert len(embedded[goal_id]) == len(bundle.acceptance_for(goal_id))
        assert embedded[goal_id][0].objective_revision == OBJECTIVE_REV
