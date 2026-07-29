from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.code_evidence_graph import (
    POST_MERGE_EVIDENCE_ACCEPTANCE_CRITERIA,
    POST_MERGE_EVIDENCE_GATE_KINDS,
    POST_MERGE_EVIDENCE_REQUIREMENT_ID,
    CodeImpactIndex,
    EvidenceGraphValidationError,
    PostMergeEvidenceReceipt,
    assemble_post_merge_evidence,
    verify_post_merge_evidence,
)
from ipfs_accelerate_py.agent_supervisor.validation.validation_scheduler import (
    ImpactValidationCheck,
    ImpactValidationDAGReceipt,
    ImpactValidationKind,
    ImpactValidationNodeReceipt,
    RepositoryValidationPolicy,
    ValidationNodeDisposition,
    build_impact_selected_validation_dag,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    EvidenceAuthority,
    EvidenceKind,
    EvidenceVerdict,
    ProofEvidence,
    ProofReceipt,
    ProofVerdict,
    ResourceBudget,
)
from ipfs_accelerate_py.agent_supervisor.planning.formal_plan_conformance import (
    PostMergeCompletionAdmissionGate,
    evaluate_post_merge_completion_admission,
)
from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import MergeQueue
from ipfs_accelerate_py.agent_supervisor.merge.merge_train import MergeTrain


CANDIDATE_TREE = "tree-candidate"
MERGED_TREE = "tree-merged"
NOW_DT = datetime.now(timezone.utc)
NOW = NOW_DT.isoformat()
DEADLINE = (NOW_DT + timedelta(hours=1)).isoformat()
EXPIRED = (NOW_DT + timedelta(hours=2)).isoformat()


def _proof(obligation_id: str, *, kernel_id: str) -> dict[str, object]:
    return ProofReceipt(
        obligation_id=obligation_id,
        plan_id="plan:asi-109",
        attempt_id=f"attempt:{obligation_id}",
        repository_id="repository:fixture",
        repository_tree_id=MERGED_TREE,
        ast_scope_ids=("post-merge-receipt",),
        premise_ids=(),
        translator_id="translator:fixture",
        solver_id="solver:fixture",
        kernel_id=kernel_id,
        toolchain_id="toolchain:fixture",
        policy_id="policy:fixture",
        resource_budget=ResourceBudget(
            wall_time_ms=10_000,
            cpu_time_ms=8_000,
            memory_bytes=64 * 1024 * 1024,
            max_processes=2,
        ),
        verdict=ProofVerdict.PROVED,
        evidence=(
            ProofEvidence(
                kind=EvidenceKind.KERNEL_VERIFICATION,
                authority=EvidenceAuthority.KERNEL,
                verdict=EvidenceVerdict.ACCEPTED,
                artifact_id=f"artifact:{obligation_id}",
                subject_id=obligation_id,
                verifier_id=kernel_id,
                independent=True,
            ),
        ),
        started_at=NOW,
        finished_at=NOW,
    ).to_dict()


def _validation() -> tuple[dict[str, object], dict[str, object]]:
    plan = build_impact_selected_validation_dag(
        impact_index=CodeImpactIndex(
            repository_tree_id=MERGED_TREE,
            symbol_paths={},
            symbol_dependencies={},
            path_dependencies={"pkg/changed.py": ()},
            validation_targets={},
        ),
        checks=(
            ImpactValidationCheck(
                "unit",
                ImpactValidationKind.UNIT,
                "pytest -q test/unit",
                cacheable=False,
            ),
        ),
        changed_paths=("pkg/changed.py",),
        repository_policy=RepositoryValidationPolicy(
            required_kinds=(ImpactValidationKind.UNIT,),
            kind_dependencies={},
            require_acceptance_coverage=False,
            require_transitive_validation=False,
        ),
    )
    planned = plan.nodes[0]
    receipt_object = ImpactValidationDAGReceipt(
        dag=plan,
        nodes=(
            ImpactValidationNodeReceipt(
                check_id=planned.check_id,
                kind=planned.check.kind,
                technique=planned.check.technique,
                command=planned.check.command,
                disposition=ValidationNodeDisposition.SUCCEEDED,
                reason="validation_passed",
                mandatory=planned.mandatory,
                selection_reasons=planned.selection_reasons,
                depends_on=planned.depends_on,
                returncode=0,
                result_digest="digest-unit",
            ),
        ),
        passed=True,
        started_at=NOW,
        finished_at=NOW,
    )
    runtime = {
        "runtime_id": "runtime-unit",
        "repository_tree_id": MERGED_TREE,
        "network_mode": "none",
        "filesystem_mode": "read_only_root_workspace",
    }
    receipt = receipt_object.to_dict()
    report = {
        "passed": True,
        "target_tree_id": MERGED_TREE,
        "hermetic": True,
        "hermetic_policy": {
            "policy_id": "hermetic@1",
            "complete_selected_dag": True,
        },
        "impact_validation_receipt": receipt,
        "results": [
            {
                "validation_id": "unit",
                "returncode": 0,
                "outcome": "passed",
                "authoritative": True,
                "stable": True,
                "attempts": [
                    {
                        "attempt_number": 1,
                        "returncode": 0,
                    },
                    {
                        "attempt_number": 2,
                        "returncode": 0,
                    },
                ],
                "runtime_id": "runtime-unit",
                "hermetic_runtime": runtime,
                "validation_result_digest": "digest-unit",
            }
        ],
        "seeded_defect_summary": {
            "seeded_count": 1,
            "detected_count": 1,
            "escaped_count": 0,
            "zero_escaped": True,
        },
        "escaped_seeded_defect_ids": [],
    }
    return report, receipt


def _kwargs() -> dict[str, object]:
    validation_report, validation_receipt = _validation()
    criterion = POST_MERGE_EVIDENCE_ACCEPTANCE_CRITERIA[0]
    return {
        "repository_id": "repository:fixture",
        "task_id": "ASI-109",
        "policy_id": "policy:fixture",
        "candidate_tree_id": CANDIDATE_TREE,
        "merged_tree_id": MERGED_TREE,
        "merge_commit_id": "commit-merged",
        "assembled_at": NOW,
        "freshness_deadline": DEADLINE,
        "proposal_admission": {
            "proposal_id": "proposal-1",
            "receipt_id": "proposal-receipt",
            "task_id": "ASI-109",
            "policy_id": "policy:fixture",
            "repository_tree_id": CANDIDATE_TREE,
            "accepted": True,
        },
        "validation_report": validation_report,
        "validation_receipt": validation_receipt,
        "semantic_checks": [
            {
                "validation_receipt_id": "semantic-1",
                "repository_tree_id": MERGED_TREE,
                "status": "passed",
                "freshness": "current",
                "observed_at": NOW,
            }
        ],
        "protocol_checks": [
            {
                "validation_receipt_id": "protocol-1",
                "repository_tree_id": MERGED_TREE,
                "status": "passed",
                "freshness": "current",
                "observed_at": NOW,
            }
        ],
        "legal_logic_obligations": [
            {
                "obligation_id": "legal-1",
                "receipt_id": "legal-receipt",
                "repository_tree_id": MERGED_TREE,
                "status": "proved",
                "freshness": "current",
                "observed_at": NOW,
            }
        ],
        "theorem_obligations": [
            {
                "obligation_id": "theorem-1",
                "receipt_id": "theorem-receipt",
                "repository_tree_id": MERGED_TREE,
                "status": "proved",
                "freshness": "current",
                "observed_at": NOW,
            }
        ],
        "proof_receipts": [
            _proof("legal-1", kernel_id="kernel:legal"),
            _proof("theorem-1", kernel_id="kernel:theorem"),
        ],
        "merge_record": {
            "merge_receipt_id": "merge-receipt",
            "task_id": "ASI-109",
            "candidate_tree_id": CANDIDATE_TREE,
            "repository_tree_id": MERGED_TREE,
            "merged_tree_id": MERGED_TREE,
            "merge_commit_id": "commit-merged",
            "status": "merged",
            "completion_status": "completed",
            "freshness": "current",
            "observed_at": NOW,
        },
        "criterion_coverage": [
            {
                "criterion": criterion,
                "repository_tree_id": MERGED_TREE,
                "implementation": [
                    "ipfs_accelerate_py/agent_supervisor/code_evidence_graph.py"
                ],
                "receipt_ids": [
                    "proposal-receipt",
                    validation_receipt["receipt_id"],
                    "semantic-1",
                    "protocol-1",
                    "proof receipts are content addressed",
                    "merge-receipt",
                ],
                "freshness": "current",
                "observed_at": NOW,
            }
        ],
        "merged_tree_records": {
            "ast_records": [
                {
                    "scope_id": "post-merge-receipt",
                    "kind": "qualified_symbol",
                    "qualified_symbol": (
                        "agent_supervisor.code_evidence_graph."
                        "PostMergeEvidenceReceipt"
                    ),
                    "repository_tree_id": MERGED_TREE,
                    "path": (
                        "ipfs_accelerate_py/agent_supervisor/"
                        "code_evidence_graph.py"
                    ),
                    "source_hash": "sha256:merged-source",
                }
            ]
        },
    }


def test_authoritative_receipt_rebuilds_graph_and_round_trips() -> None:
    receipt = assemble_post_merge_evidence(**_kwargs())

    assert receipt.accepted is True
    assert receipt.authoritative is True
    assert receipt.merge_eligible is True
    assert receipt.merge_authoritative is True
    assert receipt.completion_authoritative is True
    assert receipt.freshness_authoritative is True
    assert receipt.proved_requirement_ids == (
        POST_MERGE_EVIDENCE_REQUIREMENT_ID,
    )
    assert receipt.gate_kinds == tuple(sorted(POST_MERGE_EVIDENCE_GATE_KINDS))
    assert receipt.acceptance_criteria == POST_MERGE_EVIDENCE_ACCEPTANCE_CRITERIA
    assert {
        node.tree_id for node in receipt.graph.nodes if node.tree_id
    } == {MERGED_TREE}

    restored = PostMergeEvidenceReceipt.from_json(receipt.to_json())
    assert restored == receipt
    assert restored.receipt_id == receipt.receipt_id
    assert restored.graph_id == receipt.graph_id


def test_verifier_closes_authority_for_changed_tree_or_expired_receipt() -> None:
    receipt = assemble_post_merge_evidence(**_kwargs())

    changed = verify_post_merge_evidence(
        receipt,
        "tree-changed-after-merge",
        now=NOW,
    )
    expired = verify_post_merge_evidence(
        receipt,
        MERGED_TREE,
        now=EXPIRED,
    )

    assert changed.accepted is False
    assert changed.completion_authoritative is False
    assert "repository_tree_changed" in changed.reason_codes
    assert expired.freshness_authoritative is False
    assert "stale_evidence" in expired.reason_codes
    assert PostMergeEvidenceReceipt.from_dict(changed.to_dict()) == changed


def test_formal_admission_replays_exact_tree_commit_graph_and_criteria() -> None:
    receipt = assemble_post_merge_evidence(**_kwargs())

    admitted = evaluate_post_merge_completion_admission(
        receipt,
        current_repository_tree_id=MERGED_TREE,
        expected_repository_id="repository:fixture",
        expected_merge_commit_id="commit-merged",
        expected_evidence_graph_id=receipt.graph_id,
        expected_acceptance_criteria=POST_MERGE_EVIDENCE_ACCEPTANCE_CRITERIA,
        now=NOW,
    )
    wrong_commit = evaluate_post_merge_completion_admission(
        receipt,
        current_repository_tree_id=MERGED_TREE,
        expected_merge_commit_id="commit:foreign",
        now=NOW,
    )

    assert admitted.admitted is True
    assert admitted.post_merge_evidence_receipt_id == receipt.receipt_id
    assert admitted.revalidated_receipt_id
    assert PostMergeCompletionAdmissionGate.from_dict(
        admitted.to_dict()
    ) == admitted
    assert wrong_commit.admitted is False
    assert "post_merge_commit_mismatch" in wrong_commit.reason_codes


def test_merge_authority_adapter_rejects_foreign_candidate_task_or_policy() -> None:
    receipt = assemble_post_merge_evidence(**_kwargs())

    payload, reasons = MergeTrain._verified_post_merge_evidence(
        receipt,
        candidate_tree_id=CANDIDATE_TREE,
        repository_tree_id=MERGED_TREE,
        merge_commit="commit-merged",
        expected_repository_id="repository:fixture",
        expected_task_id="ASI-109",
        expected_policy_id="policy:fixture",
    )
    assert payload["receipt_id"] == receipt.receipt_id
    assert reasons == ()

    _, foreign_reasons = MergeTrain._verified_post_merge_evidence(
        receipt,
        candidate_tree_id="tree:foreign",
        repository_tree_id=MERGED_TREE,
        merge_commit="commit-merged",
        expected_repository_id="repository:fixture",
        expected_task_id="ASI:foreign",
        expected_policy_id="policy:foreign",
    )
    assert {
        "post_merge_evidence_candidate_tree_mismatch",
        "post_merge_evidence_task_mismatch",
        "post_merge_evidence_policy_mismatch",
    }.issubset(foreign_reasons)


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        (
            lambda values: values.update(
                gate_kinds=(*POST_MERGE_EVIDENCE_GATE_KINDS, "provider_claim")
            ),
            "extra_gate",
        ),
        (
            lambda values: values.update(
                gate_kinds=POST_MERGE_EVIDENCE_GATE_KINDS[:-1]
            ),
            "missing_gate",
        ),
        (
            lambda values: values["semantic_checks"].clear(),
            "semantic_evidence_missing",
        ),
        (
            lambda values: values["proof_receipts"][0].update(
                freshness="stale"
            ),
            "stale_evidence",
        ),
        (
            lambda values: values["proof_receipts"][0].update(
                verdict="counterexample"
            ),
            "contradictory_proof",
        ),
        (
            lambda values: values["protocol_checks"][0].update(
                repository_tree_id=CANDIDATE_TREE
            ),
            "protocol_tree_mismatch",
        ),
        (
            lambda values: values["validation_report"].update(results=[]),
            "validation_population_incomplete",
        ),
        (
            lambda values: values.update(
                acceptance_criteria=(
                    *POST_MERGE_EVIDENCE_ACCEPTANCE_CRITERIA,
                    "caller-added gate",
                )
            ),
            "acceptance_criteria_mismatch",
        ),
        (
            lambda values: values["semantic_checks"][0].update(
                repository_id="repository:foreign"
            ),
            "foreign_repository_evidence",
        ),
    ),
)
def test_any_missing_extra_stale_foreign_or_contradictory_gate_fails_closed(
    mutation,
    reason: str,
) -> None:
    values = copy.deepcopy(_kwargs())
    mutation(values)

    receipt = assemble_post_merge_evidence(**values)

    assert receipt.accepted is False
    assert receipt.merge_authoritative is False
    assert receipt.completion_authoritative is False
    assert receipt.proved_requirement_ids == ()
    assert reason in receipt.reason_codes


def test_unknown_merged_tree_record_channel_is_rejected() -> None:
    values = _kwargs()
    values["merged_tree_records"]["provider_verdicts"] = [
        {"status": "passed"}
    ]

    with pytest.raises(
        EvidenceGraphValidationError,
        match="unsupported merged-tree record channels",
    ):
        assemble_post_merge_evidence(**values)


def test_deserialization_rejects_forged_authority_identity_and_graph() -> None:
    receipt = assemble_post_merge_evidence(**_kwargs())
    payload = receipt.to_dict()
    payload["completion_authoritative"] = False
    with pytest.raises(
        EvidenceGraphValidationError, match="completion_authoritative"
    ):
        PostMergeEvidenceReceipt.from_dict(payload)

    payload = receipt.to_dict()
    payload["receipt_id"] = "foreign-receipt"
    with pytest.raises(EvidenceGraphValidationError, match="identity mismatch"):
        PostMergeEvidenceReceipt.from_dict(payload)

    payload = receipt.to_dict()
    payload["graph"]["nodes"][0]["record"]["tampered"] = True
    payload["graph"]["nodes"][0].pop("node_id")
    payload["graph"].pop("graph_id")
    with pytest.raises(EvidenceGraphValidationError, match="rebuilt merged-tree"):
        PostMergeEvidenceReceipt.from_dict(payload)


def test_post_merge_evidence_rejects_target_mutating_merge_callback(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    queue = MergeQueue(tmp_path / "queue")

    with pytest.raises(
        ValueError,
        match="built-in synthesized-commit CAS path",
    ):
        MergeTrain(
            repo,
            queue,
            merge_callback=lambda _request: {},
            post_merge_validation=lambda *_args, **_kwargs: {},
            post_merge_evidence=lambda *_args, **_kwargs: {},
        )
