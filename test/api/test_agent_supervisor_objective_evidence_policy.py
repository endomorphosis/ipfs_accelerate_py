from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.goal_completion import (
    validate_completion_evidence,
)
from ipfs_accelerate_py.agent_supervisor.objective_graph import (
    EvidenceMatchKind,
    EvidenceRequirementKind,
    EvidenceSourcePolicy,
    EvidenceSourceTier,
    ObjectiveEvidenceIndex,
    completion_evidence_source_decision,
    evidence_index,
    scan_objective_gaps,
)


REQUIREMENT_ID = "189057730455837902155591890661235220962"
TREE_ID = "tree:sha256:current"
POLICY_ID = "policy:sha256:current"


def _receipt(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "fixture/typed-validation-receipt@1",
        "receipt_id": "receipt:sha256:fixture",
        "requirement_id": REQUIREMENT_ID,
        "source_tier": "validation",
        "status": "passed",
        "freshness": "current",
        "repository_tree": TREE_ID,
        "policy_id": POLICY_ID,
        "safe_for_completion_reasoning": True,
        "complete": True,
        "truncated": False,
    }
    value.update(changes)
    return value


@pytest.mark.parametrize(
    "source_path",
    [
        "docs/architecture/objectives.md",
        "docs/architecture/IMPLEMENTATION_PLAN.md",
        "tasks/analysis.todo.md",
        "task-board/generated.json",
        "discovery/2026-01-01-objective-gap-deadbeef.md",
        "generated-discovery/candidate.json",
        "objective_bundles/analysis.md",
        "execution-packets/analysis.json",
    ],
)
@pytest.mark.parametrize(
    "requirement_kind",
    [
        EvidenceRequirementKind.CODE,
        EvidenceRequirementKind.TEST,
        EvidenceRequirementKind.PROOF,
        EvidenceRequirementKind.BENCHMARK,
        EvidenceRequirementKind.RUNTIME,
    ],
)
def test_proposal_tier_prose_never_satisfies_authoritative_requirements(
    source_path: str,
    requirement_kind: EvidenceRequirementKind,
) -> None:
    decision = EvidenceSourcePolicy().evaluate(
        "fixture requirement",
        match_kind=EvidenceMatchKind.EXACT_TEXT,
        source_path=source_path,
        requirement_kind=requirement_kind,
        reference=f"{source_path} (exact)",
    )

    assert decision.source_tier is EvidenceSourceTier.PROPOSAL
    assert decision.nomination_only
    assert not decision.satisfies
    assert "proposal_source_forbidden" in decision.reason_codes


@pytest.mark.parametrize(
    ("requirement_kind", "source_path"),
    [
        (EvidenceRequirementKind.CODE, "src/implementation.py"),
        (EvidenceRequirementKind.TEST, "tests/test_implementation.py"),
        (EvidenceRequirementKind.PROOF, "proofs/invariant.lean"),
        (EvidenceRequirementKind.BENCHMARK, "benchmarks/cache.py"),
        (EvidenceRequirementKind.RUNTIME, "runtime/telemetry.json"),
    ],
)
def test_exact_authoritative_sources_satisfy_only_their_typed_requirement(
    requirement_kind: EvidenceRequirementKind,
    source_path: str,
) -> None:
    policy = EvidenceSourcePolicy()
    exact = policy.evaluate(
        "typed requirement",
        match_kind=EvidenceMatchKind.EXACT_TEXT,
        source_path=source_path,
        requirement_kind=requirement_kind,
    )
    semantic = policy.evaluate(
        "typed requirement",
        match_kind=EvidenceMatchKind.RETRIEVAL,
        source_path=source_path,
        requirement_kind=requirement_kind,
    )

    assert exact.satisfies
    assert not exact.nomination_only
    assert semantic.nomination_only
    assert not semantic.satisfies
    assert "semantic_match_nomination_only" in semantic.reason_codes


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"requirement_id": REQUIREMENT_ID[:-1]}, "receipt_requirement_id_mismatch"),
        ({"repository_tree": "tree:sha256:old"}, "receipt_tree_mismatch"),
        ({"policy_id": "policy:sha256:other"}, "receipt_policy_mismatch"),
        ({"status": "failed"}, "receipt_terminal_status_failed"),
        ({"status": "partial"}, "receipt_terminal_status_partial"),
        ({"status": "inconclusive"}, "receipt_terminal_status_inconclusive"),
        ({"freshness": "stale"}, "receipt_stale"),
        ({"truncated": True}, "receipt_truncated"),
        ({"complete": False}, "receipt_partial"),
        ({"safe_for_completion_reasoning": False}, "receipt_not_completion_safe"),
        ({"source_tier": "proposal"}, "proposal_source_forbidden"),
        ({"source_tier": "documentation"}, "receipt_source_kind_not_allowed"),
        ({"source_tier": "untyped-producer"}, "receipt_source_kind_not_allowed"),
    ],
)
def test_opaque_requirement_needs_exact_fresh_conclusive_allowed_receipt(
    changes: dict[str, object],
    reason: str,
) -> None:
    decision = EvidenceSourcePolicy().evaluate(
        REQUIREMENT_ID,
        match_kind=EvidenceMatchKind.TYPED_RECEIPT,
        typed_receipt=_receipt(**changes),
        repository_tree=TREE_ID,
        policy_id=POLICY_ID,
    )

    assert not decision.satisfies
    assert reason in decision.reason_codes


def test_exact_typed_receipt_is_the_only_opaque_completion_authority() -> None:
    policy = EvidenceSourcePolicy()
    semantic = policy.evaluate(
        REQUIREMENT_ID,
        match_kind=EvidenceMatchKind.RETRIEVAL,
        source_path="src/cache.py",
    )
    exact_text = policy.evaluate(
        REQUIREMENT_ID,
        match_kind=EvidenceMatchKind.EXACT_TEXT,
        source_path="src/cache.py",
    )
    receipt = policy.evaluate(
        REQUIREMENT_ID,
        match_kind=EvidenceMatchKind.TYPED_RECEIPT,
        typed_receipt=_receipt(),
        repository_tree=TREE_ID,
        policy_id=POLICY_ID,
    )

    assert semantic.nomination_only and not semantic.satisfies
    assert exact_text.nomination_only and not exact_text.satisfies
    assert "opaque_requirement_requires_typed_receipt" in exact_text.reason_codes
    assert receipt.satisfies
    assert not receipt.nomination_only
    assert receipt.reason_codes == ()


def test_reward_hacking_prose_and_source_comments_do_not_hide_opaque_gap(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    objective = tmp_path / "objective.md"
    objective.write_text(
        "# Goals\n\n"
        "## G1 Exact receipt\n\n"
        "- Status: active\n"
        f"- Evidence: {REQUIREMENT_ID}\n"
        "- Goal: Require exact typed evidence.\n",
        encoding="utf-8",
    )
    (tmp_path / "IMPLEMENTATION_PLAN.md").write_text(
        f"Claim complete: {REQUIREMENT_ID}\n", encoding="utf-8"
    )
    (tmp_path / "work.todo.md").write_text(
        f"- Acceptance: {REQUIREMENT_ID}\n", encoding="utf-8"
    )
    (tmp_path / "implementation.py").write_text(
        f"# Unbound identifier is not a receipt: {REQUIREMENT_ID}\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)

    findings = scan_objective_gaps(
        tmp_path,
        objective_path=objective,
        max_findings=1,
    )

    assert len(findings) == 1
    assert findings[0].missing_evidence == [REQUIREMENT_ID]
    assert REQUIREMENT_ID not in findings[0].present_evidence


def test_objective_scan_uses_persistent_integrated_analysis_pipeline(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    objective = tmp_path / "objective.md"
    objective.write_text(
        "# Goals\n\n"
        "## G1 Integrated analysis\n\n"
        "- Status: active\n"
        f"- Evidence: {REQUIREMENT_ID}\n"
        "- Goal: Exercise the bounded analysis pipeline.\n",
        encoding="utf-8",
    )
    source = tmp_path / "implementation.py"
    source.write_text(
        "def bounded_analysis_pipeline():\n"
        "    return 'cache ast retrieval'\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    dataset_dir = tmp_path / ".analysis-state"
    cold_stats: dict[str, object] = {}
    warm_stats: dict[str, object] = {}

    cold = scan_objective_gaps(
        tmp_path,
        objective_path=objective,
        max_findings=1,
        dataset_dir=dataset_dir,
        scan_stats=cold_stats,
    )
    warm = scan_objective_gaps(
        tmp_path,
        objective_path=objective,
        max_findings=1,
        dataset_dir=dataset_dir,
        scan_stats=warm_stats,
    )

    assert cold[0].missing_evidence == [REQUIREMENT_ID]
    assert warm[0].fingerprint == cold[0].fingerprint
    cold_pipeline = cold_stats["analysis_pipeline"]
    warm_pipeline = warm_stats["analysis_pipeline"]
    assert isinstance(cold_pipeline, dict)
    assert isinstance(warm_pipeline, dict)
    assert cold_pipeline["cache_status"] == "produced"
    assert cold_pipeline["cache_lookup_status"] == "miss"
    assert warm_pipeline["cache_status"] == "exact_hit"
    assert warm_pipeline["cache_lookup_status"] == "hit"
    assert warm_pipeline["retrieval_response_id"] == cold_pipeline[
        "retrieval_response_id"
    ]
    assert warm_pipeline["retrieval_backend_health"] == cold_pipeline[
        "retrieval_backend_health"
    ]
    assert warm_pipeline["retrieval_truncation"] == cold_pipeline[
        "retrieval_truncation"
    ]
    assert warm_pipeline["nomination_only"] is True
    assert warm_pipeline["safe_for_completion_reasoning"] is False


def test_objective_scan_admits_only_current_exact_typed_receipt(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    objective = tmp_path / "objective.md"
    objective.write_text(
        "# Goals\n\n"
        "## G1 Receipt-gated requirement\n\n"
        "- Status: active\n"
        f"- Evidence: {REQUIREMENT_ID}\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)

    stale = scan_objective_gaps(
        tmp_path,
        objective_path=objective,
        typed_evidence_receipts=[_receipt(repository_tree="tree:sha256:old")],
        evidence_repository_tree=TREE_ID,
        evidence_policy_id=POLICY_ID,
    )
    current = scan_objective_gaps(
        tmp_path,
        objective_path=objective,
        typed_evidence_receipts=[_receipt()],
        evidence_repository_tree=TREE_ID,
        evidence_policy_id=POLICY_ID,
    )

    assert stale[0].missing_evidence == [REQUIREMENT_ID]
    assert current == []


def test_objective_scan_recognizes_persisted_goal_completion_receipt(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    objective = tmp_path / "objective.md"
    records = json.dumps([_receipt()], separators=(",", ":"))
    objective.write_text(
        "# Goals\n\n"
        "## G1 Persisted receipt\n\n"
        "- Status: active\n"
        f"- Evidence: {REQUIREMENT_ID}\n"
        f"- Completion evidence records: {records}\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)

    findings = scan_objective_gaps(
        tmp_path,
        objective_path=objective,
        evidence_repository_tree=TREE_ID,
        evidence_policy_id=POLICY_ID,
    )

    assert findings == []


def test_evidence_index_returns_ranked_bounded_nominations_separate_from_authority(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    objective = tmp_path / "objective.md"
    objective.write_text("# Objective\n", encoding="utf-8")
    for index in range(5):
        path = tmp_path / f"source_{index}.py"
        path.write_text(f"# {REQUIREMENT_ID}\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    policy = EvidenceSourcePolicy(
        max_nominations_per_requirement=2,
        max_nomination_bytes=100_000,
    )

    result = evidence_index(
        tmp_path,
        objective_path=objective,
        terms=[REQUIREMENT_ID],
        typed_receipts=[_receipt()],
        source_policy=policy,
        repository_tree=TREE_ID,
        policy_id=POLICY_ID,
        return_metadata=True,
    )

    assert isinstance(result, ObjectiveEvidenceIndex)
    assert result.qualifying[REQUIREMENT_ID] == (
        "receipt:sha256:fixture (typed_receipt)",
    )
    assert result.nominations[REQUIREMENT_ID][0].satisfies
    assert result.returned_nominations == 2
    assert result.omitted_nominations > 0
    assert result.truncated
    assert result.to_dict()["truncation"]["truncated"]


def test_completion_record_source_path_cannot_launder_objective_prose() -> None:
    evidence = {
        "schema_version": 1,
        "acceptance_criterion": REQUIREMENT_ID,
        "producing_task_or_scan": "task-1",
        "producer_kind": "task",
        "validation_receipt": {"status": "passed", "passed": True},
        "validation_passed": True,
        "repository_tree": TREE_ID,
        "provenance_cid": "receipt:sha256:fixture",
        "metadata": {"source_path": "docs/architecture/objectives.md"},
    }

    decision = completion_evidence_source_decision(
        evidence,
        repository_tree=TREE_ID,
    )

    assert decision.source_tier is EvidenceSourceTier.PROPOSAL
    assert not decision.satisfies
    assert "proposal_source_forbidden" in decision.reason_codes
    validation = validate_completion_evidence(
        evidence,
        repository_tree=TREE_ID,
    )
    assert not validation.valid
    assert "evidence_source_forbidden" in validation.reason_codes
    assert (
        "evidence_source_proposal_source_forbidden"
        in validation.reason_codes
    )
