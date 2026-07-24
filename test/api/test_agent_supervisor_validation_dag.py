from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.code_proof_obligations import (
    CandidateDiffEntry,
    DiffChangeKind,
)
from ipfs_accelerate_py.agent_supervisor.proposal_validation import (
    ImplementationProposal,
    NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_REQUIREMENT_ID,
    ProposalFindingCode,
    ProposalValidationPolicy,
    validate_implementation_proposal,
)
from ipfs_accelerate_py.agent_supervisor.validation_commands import (
    ValidationCommand,
    ValidationStage,
)
from ipfs_accelerate_py.agent_supervisor.validation_scheduler import (
    ImpactDependencyGraph,
    TRANSITIVE_IMPACT_REQUIREMENT_ID,
    ValidationDAGError,
    ValidationDAGReceipt,
    ValidationNodeDisposition,
    ValidationScheduler,
)


G100_FAIL_FAST_REQUIREMENT = "314133036252270790078901745919131980427"
G101_TRANSITIVE_IMPACT_REQUIREMENT = "266404049326363900535699811645710804440"
TREE_ID = "tree:validation-dag"


def _policy() -> ProposalValidationPolicy:
    return ProposalValidationPolicy(
        allowed_paths=("pkg/",),
        expected_task_id="ASI-031",
        expected_plan_id="plan:validation-dag",
        expected_repository_id="repo:fixture",
        expected_repository_tree_id=TREE_ID,
        expected_objective_id="ASI-G100",
    )


def _proposal(
    candidate_diff: tuple[CandidateDiffEntry, ...],
    *,
    declared_paths: tuple[str, ...] | None = None,
) -> ImplementationProposal:
    return ImplementationProposal(
        task_id="ASI-031",
        accepted_plan_id="plan:validation-dag",
        repository_id="repo:fixture",
        repository_tree_id=TREE_ID,
        objective_id="ASI-G100",
        baseline_id="tree:baseline",
        candidate_diff=candidate_diff,
        declared_paths=(
            tuple(
                sorted(
                    {
                        path
                        for entry in candidate_diff
                        for path in (entry.old_path, entry.new_path)
                        if path
                    }
                )
            )
            if declared_paths is None
            else declared_paths
        ),
    )


def _source_change(path: str = "pkg/source.py") -> CandidateDiffEntry:
    return CandidateDiffEntry(
        old_path=path,
        new_path=path,
        change_kind=DiffChangeKind.MODIFY,
        before_source="def value() -> int:\n    return 1\n",
        after_source="def value() -> int:\n    return 2\n",
    )


def _commands() -> tuple[ValidationCommand, ...]:
    return (
        ValidationCommand(
            command="python -m compileall -q pkg",
            stage=ValidationStage.CHEAP,
            cacheable=False,
            ordinal=0,
        ),
        ValidationCommand(
            command="pytest -q test/api/test_transitive_consumer.py",
            stage=ValidationStage.TARGETED,
            impact_paths=("test/api/test_transitive_consumer.py",),
            cacheable=False,
            ordinal=1,
        ),
    )


@pytest.mark.parametrize(
    ("proposal", "expected_code"),
    [
        (_proposal((), declared_paths=()), ProposalFindingCode.EMPTY_PATCH),
        (
            _proposal((_source_change("outside/source.py"),)),
            ProposalFindingCode.PATH_OUTSIDE_SCOPE,
        ),
    ],
)
def test_rejected_proposal_closes_dispatch_and_proves_exact_g100_requirement(
    tmp_path: Path,
    proposal: ImplementationProposal,
    expected_code: ProposalFindingCode,
) -> None:
    calls: list[str] = []

    def runner(*, spec: ValidationCommand, **_kwargs: object) -> dict[str, object]:
        calls.append(spec.command)
        return {"returncode": 0, "output": "must not execute"}

    validation = validate_implementation_proposal(proposal, policy=_policy())
    report = ValidationScheduler().run_validated(
        validation,
        _commands(),
        workspace_path=tmp_path,
        dependency_state="fixture",
        runner=runner,
    )

    assert not validation.accepted
    assert any(finding.code is expected_code for finding in validation.findings)
    assert calls == []
    assert report["attempted"] is False
    assert report["error"] == "proposal_validation_failed"
    assert report["validation_dag_receipt"] is None
    assert {node["disposition"] for node in report["nodes"]} == {"blocked"}

    receipt = report["proposal_receipt"]
    evidence = receipt["rejection_evidence"]
    assert receipt["expensive_checks_started"] == 0
    assert evidence["expensive_checks_started"] == 0
    assert tuple(report["proved_requirement_ids"]) == (G100_FAIL_FAST_REQUIREMENT,)
    assert tuple(receipt["proved_requirement_ids"]) == (G100_FAIL_FAST_REQUIREMENT,)
    assert evidence["requirement_id"] == G100_FAIL_FAST_REQUIREMENT
    assert (
        NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_REQUIREMENT_ID
        == G100_FAIL_FAST_REQUIREMENT
    )


def _failing_transitive_report(
    tmp_path: Path,
) -> tuple[dict[str, object], list[str]]:
    proposal = _proposal((_source_change(),))
    validation = validate_implementation_proposal(proposal, policy=_policy())
    graph = ImpactDependencyGraph(
        repository_tree_id=TREE_ID,
        dependencies={
            "pkg/consumer.py": ("pkg/source.py",),
            "test/api/test_transitive_consumer.py": ("pkg/consumer.py",),
        },
    )
    calls: list[str] = []

    def runner(*, spec: ValidationCommand, **_kwargs: object) -> dict[str, object]:
        calls.append(spec.command)
        return {
            "returncode": (
                7
                if spec.command
                == "pytest -q test/api/test_transitive_consumer.py"
                else 0
            ),
            "output": "seeded transitive failure",
        }

    report = ValidationScheduler(max_workers=1).run_validated(
        validation,
        _commands(),
        workspace_path=tmp_path,
        impact_graph=graph,
        seeded_defect_id="seed:g101",
        seeded_defect_path="pkg/source.py",
        dependency_state="fixture",
        runner=runner,
    )
    return report, calls


def test_transitive_impact_selects_failing_test_and_proves_exact_g101_requirement(
    tmp_path: Path,
) -> None:
    report, calls = _failing_transitive_report(tmp_path)
    receipt = ValidationDAGReceipt.from_dict(report["validation_dag_receipt"])

    assert calls == [
        "python -m compileall -q pkg",
        "pytest -q test/api/test_transitive_consumer.py",
    ]
    assert report["attempted"] is True
    assert report["passed"] is False
    assert tuple(report["affected_paths"]) == (
        "pkg/consumer.py",
        "pkg/source.py",
        "test/api/test_transitive_consumer.py",
    )
    assert receipt.passed is False
    assert receipt.transitive_evidence is not None
    assert receipt.transitive_evidence.impact_path == (
        "pkg/source.py",
        "pkg/consumer.py",
        "test/api/test_transitive_consumer.py",
    )
    assert receipt.transitive_evidence.seeded_defect_id == "seed:g101"
    assert tuple(report["proved_requirement_ids"]) == (
        G101_TRANSITIVE_IMPACT_REQUIREMENT,
    )
    assert receipt.proved_requirement_ids == (G101_TRANSITIVE_IMPACT_REQUIREMENT,)
    assert (
        receipt.transitive_evidence.requirement_id
        == G101_TRANSITIVE_IMPACT_REQUIREMENT
    )
    assert TRANSITIVE_IMPACT_REQUIREMENT_ID == G101_TRANSITIVE_IMPACT_REQUIREMENT
    failed = [
        node
        for node in receipt.nodes
        if node.disposition is ValidationNodeDisposition.FAILED
    ]
    assert len(failed) == 1
    assert failed[0].command == "pytest -q test/api/test_transitive_consumer.py"


def test_stale_impact_graph_is_rejected_before_runner_dispatch(
    tmp_path: Path,
) -> None:
    validation = validate_implementation_proposal(
        _proposal((_source_change(),)),
        policy=_policy(),
    )
    stale_graph = ImpactDependencyGraph(
        repository_tree_id="tree:stale",
        dependencies={
            "pkg/consumer.py": ("pkg/source.py",),
            "test/api/test_transitive_consumer.py": ("pkg/consumer.py",),
        },
    )
    calls: list[str] = []

    def runner(*, spec: ValidationCommand, **_kwargs: object) -> dict[str, object]:
        calls.append(spec.command)
        return {"returncode": 0}

    with pytest.raises(ValidationDAGError, match="impact graph is stale"):
        ValidationScheduler().run_validated(
            validation,
            _commands(),
            workspace_path=tmp_path,
            impact_graph=stale_graph,
            dependency_state="fixture",
            runner=runner,
        )

    assert calls == []


def test_missing_or_uncovered_impact_fails_closed_without_false_completion(
    tmp_path: Path,
) -> None:
    validation = validate_implementation_proposal(
        _proposal((_source_change(),)),
        policy=_policy(),
    )
    calls: list[str] = []

    def runner(*, spec: ValidationCommand, **_kwargs: object) -> dict[str, object]:
        calls.append(spec.command)
        return {"returncode": 0}

    missing = ValidationScheduler().run_validated(
        validation,
        _commands(),
        workspace_path=tmp_path,
        dependency_state="fixture",
        runner=runner,
    )
    assert missing["error"] == "impact_graph_missing"
    assert missing["passed"] is False
    assert calls == []
    assert ValidationDAGReceipt.from_dict(
        missing["validation_dag_receipt"]
    ).uncovered_impact

    unrelated = ImpactDependencyGraph(
        repository_tree_id=TREE_ID,
        dependencies={"pkg/consumer.py": ("pkg/source.py",)},
    )
    uncovered = ValidationScheduler().run_validated(
        validation,
        (
            ValidationCommand(
                command="pytest -q test/api/test_unrelated.py",
                stage=ValidationStage.TARGETED,
                impact_paths=("test/api/test_unrelated.py",),
                cacheable=False,
            ),
        ),
        workspace_path=tmp_path,
        impact_graph=unrelated,
        dependency_state="fixture",
        runner=runner,
    )
    assert uncovered["error"] == "uncovered_validation_impact"
    assert uncovered["passed"] is False
    assert calls == []
    receipt = ValidationDAGReceipt.from_dict(
        uncovered["validation_dag_receipt"]
    )
    assert receipt.uncovered_impact
    assert receipt.completion_authoritative is False


@pytest.mark.parametrize(
    "tamper",
    [
        lambda payload: payload.__setitem__("graph_id", "graph:tampered"),
        lambda payload: payload["nodes"][0].__setitem__(
            "result_digest", "digest:tampered"
        ),
        lambda payload: payload["transitive_evidence"].__setitem__(
            "receipt_id", "receipt:tampered"
        ),
    ],
    ids=("graph-binding", "result-binding", "evidence-binding"),
)
def test_validation_dag_receipt_rejects_tampering(
    tmp_path: Path,
    tamper,
) -> None:
    report, _calls = _failing_transitive_report(tmp_path)
    payload = deepcopy(report["validation_dag_receipt"])
    tamper(payload)

    with pytest.raises(ValidationDAGError):
        ValidationDAGReceipt.from_dict(payload)
