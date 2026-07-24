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
    ValidationDAGNodeRecord,
    ValidationDAGReceipt,
    ValidationNodeDisposition,
    ValidationScheduler,
)


G100_FAIL_FAST_REQUIREMENT = "314133036252270790078901745919131980427"
G101_TRANSITIVE_IMPACT_REQUIREMENT = "266404049326363900535699811645710804440"
G102_PROOF_CANDIDATE_REQUIREMENT = "006818797857632260116084792540150258746"
TREE_ID = "tree:validation-dag"
TASK_ID = "ASI-032"
OBJECTIVE_ID = "ASI-G101"
VALIDATION_ID = "transitive-consumer"


def _policy() -> ProposalValidationPolicy:
    return ProposalValidationPolicy(
        allowed_paths=("pkg/",),
        expected_task_id=TASK_ID,
        expected_plan_id="plan:validation-dag",
        expected_repository_id="repo:fixture",
        expected_repository_tree_id=TREE_ID,
        expected_objective_id=OBJECTIVE_ID,
    )


def _proposal(
    candidate_diff: tuple[CandidateDiffEntry, ...],
    *,
    declared_paths: tuple[str, ...] | None = None,
) -> ImplementationProposal:
    return ImplementationProposal(
        task_id=TASK_ID,
        accepted_plan_id="plan:validation-dag",
        repository_id="repo:fixture",
        repository_tree_id=TREE_ID,
        objective_id=OBJECTIVE_ID,
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
            validation_id=VALIDATION_ID,
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
        validation_targets={
            VALIDATION_ID: ("test/api/test_transitive_consumer.py",),
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
            "seeded_defect_id": (
                "seed:g101"
                if spec.command
                == "pytest -q test/api/test_transitive_consumer.py"
                else ""
            ),
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
    assert receipt.coverage_complete is True
    assert receipt.repository_tree_id == TREE_ID
    assert receipt.objective_id == OBJECTIVE_ID
    assert receipt.proposal_receipt_id == report["proposal_receipt"]["receipt_id"]
    assert receipt.graph_id == receipt.impact_graph.graph_id
    assert receipt.required_validation_ids == (VALIDATION_ID,)
    assert set(receipt.selected_node_ids) == {
        node.node_id for node in receipt.nodes if node.selected
    }
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
    assert failed[0].validation_id == VALIDATION_ID
    assert failed[0].mandatory is True
    assert failed[0].observed_seeded_defect_id == "seed:g101"
    assert failed[0].depends_on
    assert {gate.disposition.value for gate in receipt.authority_gates} == {
        "blocked"
    }
    assert {gate.gate for gate in receipt.authority_gates} == {
        "semantic",
        "proof",
        "merge",
        "freshness",
        "completion",
    }
    assert report["proof_authoritative"] is False
    assert receipt.proof_authoritative is False
    assert report["completion_authoritative"] is False
    assert report["freshness_authoritative"] is False
    assert report["merge_eligible"] is False
    assert G102_PROOF_CANDIDATE_REQUIREMENT not in receipt.proved_requirement_ids


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


def test_declared_transitive_validation_cannot_be_omitted_from_population(
    tmp_path: Path,
) -> None:
    validation = validate_implementation_proposal(
        _proposal((_source_change(),)),
        policy=_policy(),
    )
    graph = ImpactDependencyGraph(
        repository_tree_id=TREE_ID,
        dependencies={
            "pkg/consumer.py": ("pkg/source.py",),
            "test/api/test_transitive_consumer.py": ("pkg/consumer.py",),
            "test/api/test_unrelated.py": (),
        },
        validation_targets={
            VALIDATION_ID: ("test/api/test_transitive_consumer.py",),
        },
    )
    calls: list[str] = []

    def runner(*, spec: ValidationCommand, **_kwargs: object) -> dict[str, object]:
        calls.append(spec.command)
        return {"returncode": 0}

    report = ValidationScheduler().run_validated(
        validation,
        (
            ValidationCommand(
                command="pytest -q test/api/test_unrelated.py",
                stage=ValidationStage.TARGETED,
                impact_paths=("test/api/test_unrelated.py",),
                validation_id="unrelated",
                cacheable=False,
            ),
        ),
        workspace_path=tmp_path,
        impact_graph=graph,
        dependency_state="fixture",
        runner=runner,
    )
    receipt = ValidationDAGReceipt.from_dict(report["validation_dag_receipt"])

    assert calls == []
    assert report["error"] == "uncovered_validation_impact"
    assert (
        f"validation_population:{VALIDATION_ID}:0"
        in report["coverage_errors"]
    )
    assert receipt.required_validation_ids == (VALIDATION_ID,)
    assert receipt.coverage_complete is False
    assert receipt.uncovered_impact is True
    assert receipt.proved_requirement_ids == ()


def test_transitive_failure_blocks_dependent_semantic_and_proof_nodes(
    tmp_path: Path,
) -> None:
    validation = validate_implementation_proposal(
        _proposal((_source_change(),)),
        policy=_policy(),
    )
    graph = ImpactDependencyGraph(
        repository_tree_id=TREE_ID,
        dependencies={
            "pkg/consumer.py": ("pkg/source.py",),
            "test/api/test_transitive_consumer.py": ("pkg/consumer.py",),
        },
        validation_targets={
            VALIDATION_ID: ("test/api/test_transitive_consumer.py",),
        },
    )
    commands = (
        *_commands(),
        ValidationCommand(
            command="semantic-check",
            stage=ValidationStage.TRANSLATION,
            cacheable=False,
            ordinal=2,
        ),
        ValidationCommand(
            command="proof-solver",
            stage=ValidationStage.SOLVER,
            cacheable=False,
            ordinal=3,
        ),
        ValidationCommand(
            command="proof-kernel",
            stage=ValidationStage.KERNEL,
            cacheable=False,
            ordinal=4,
        ),
    )
    calls: list[str] = []

    def runner(*, spec: ValidationCommand, **_kwargs: object) -> dict[str, object]:
        calls.append(spec.command)
        failed = spec.validation_id == VALIDATION_ID
        return {
            "returncode": 7 if failed else 0,
            "seeded_defect_id": "seed:g101" if failed else "",
        }

    report = ValidationScheduler(max_workers=2).run_validated(
        validation,
        commands,
        workspace_path=tmp_path,
        impact_graph=graph,
        seeded_defect_id="seed:g101",
        seeded_defect_path="pkg/source.py",
        dependency_state="fixture",
        runner=runner,
    )
    receipt = ValidationDAGReceipt.from_dict(report["validation_dag_receipt"])

    assert calls == [
        "python -m compileall -q pkg",
        "pytest -q test/api/test_transitive_consumer.py",
    ]
    nodes = {node.command: node for node in receipt.nodes}
    assert nodes["semantic-check"].disposition is ValidationNodeDisposition.BLOCKED
    assert nodes["proof-solver"].disposition is ValidationNodeDisposition.BLOCKED
    assert nodes["proof-kernel"].disposition is ValidationNodeDisposition.BLOCKED
    assert nodes["semantic-check"].depends_on == (
        nodes["pytest -q test/api/test_transitive_consumer.py"].node_id,
    )
    assert nodes["proof-solver"].depends_on == (
        nodes["semantic-check"].node_id,
    )
    assert nodes["proof-kernel"].depends_on == (
        nodes["proof-solver"].node_id,
    )


@pytest.mark.parametrize(
    ("graph_dependencies", "returncode", "seed_path", "observed_seed"),
    [
        (
            {
                "pkg/consumer.py": ("pkg/source.py",),
                "test/api/test_transitive_consumer.py": ("pkg/consumer.py",),
            },
            0,
            "pkg/source.py",
            "",
        ),
        (
            {"test/api/test_transitive_consumer.py": ("pkg/source.py",)},
            7,
            "pkg/source.py",
            "seed:g101",
        ),
        (
            {
                "pkg/consumer.py": ("pkg/source.py",),
                "test/api/test_transitive_consumer.py": ("pkg/consumer.py",),
            },
            7,
            "pkg/not-changed.py",
            "seed:g101",
        ),
        (
            {
                "pkg/consumer.py": ("pkg/source.py",),
                "test/api/test_transitive_consumer.py": ("pkg/consumer.py",),
            },
            7,
            "pkg/source.py",
            "",
        ),
    ],
    ids=(
        "passing-consumer",
        "direct-only-failure",
        "wrong-seed-path",
        "unobserved-seed",
    ),
)
def test_nonqualifying_results_never_emit_transitive_requirement(
    tmp_path: Path,
    graph_dependencies: dict[str, tuple[str, ...]],
    returncode: int,
    seed_path: str,
    observed_seed: str,
) -> None:
    validation = validate_implementation_proposal(
        _proposal((_source_change(),)),
        policy=_policy(),
    )
    graph = ImpactDependencyGraph(
        repository_tree_id=TREE_ID,
        dependencies=graph_dependencies,
        validation_targets={
            VALIDATION_ID: ("test/api/test_transitive_consumer.py",),
        },
    )

    def runner(*, spec: ValidationCommand, **_kwargs: object) -> dict[str, object]:
        if spec.validation_id != VALIDATION_ID:
            return {"returncode": 0}
        return {
            "returncode": returncode,
            "seeded_defect_id": observed_seed,
        }

    report = ValidationScheduler(max_workers=1).run_validated(
        validation,
        _commands(),
        workspace_path=tmp_path,
        impact_graph=graph,
        seeded_defect_id="seed:g101",
        seeded_defect_path=seed_path,
        dependency_state="fixture",
        runner=runner,
    )
    receipt = ValidationDAGReceipt.from_dict(report["validation_dag_receipt"])

    assert receipt.transitive_evidence is None
    assert receipt.proved_requirement_ids == ()
    assert tuple(report["proved_requirement_ids"]) == ()


def test_empty_or_omitted_only_receipts_cannot_claim_a_passing_dag() -> None:
    with pytest.raises(ValidationDAGError):
        ValidationDAGReceipt(
            repository_tree_id=TREE_ID,
            objective_id=OBJECTIVE_ID,
            policy_id="policy:forged",
            proposal_receipt_id="proposal:forged",
            graph_id="",
            changed_paths=(),
            affected_paths=(),
            nodes=(),
            passed=True,
        )

    omitted = ValidationDAGNodeRecord(
        node_id="node:omitted",
        command="pytest -q test/api/test_transitive_consumer.py",
        stage=ValidationStage.TARGETED.label,
        disposition=ValidationNodeDisposition.OMITTED,
        reason="not_selected",
    )
    with pytest.raises(ValidationDAGError):
        ValidationDAGReceipt(
            repository_tree_id=TREE_ID,
            objective_id=OBJECTIVE_ID,
            policy_id="policy:forged",
            proposal_receipt_id="proposal:forged",
            graph_id="",
            changed_paths=("pkg/source.py",),
            affected_paths=("pkg/source.py",),
            nodes=(omitted,),
            passed=True,
        )


def _tamper_evidence_path(payload: dict[str, object]) -> None:
    evidence = payload["transitive_evidence"]
    evidence["impact_path"] = [
        "pkg/source.py",
        "pkg/forged.py",
        "test/api/test_transitive_consumer.py",
    ]
    evidence.pop("evidence_id", None)


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
        _tamper_evidence_path,
        lambda payload: payload["nodes"][0].__setitem__(
            "depends_on", [payload["nodes"][0]["node_id"]]
        ),
        lambda payload: payload["nodes"][0].__setitem__("selected", False),
        lambda payload: payload["authority_gates"][0].__setitem__(
            "disposition", "pending"
        ),
        lambda payload: payload["impact_graph"]["validation_targets"].__setitem__(
            VALIDATION_ID, ["test/api/test_forged.py"]
        ),
        lambda payload: payload.__setitem__("proof_authoritative", True),
    ],
    ids=(
        "graph-binding",
        "result-binding",
        "evidence-binding",
        "impact-path-binding",
        "dependency-cycle",
        "selected-population",
        "authority-closure",
        "required-validation-binding",
        "proof-authority",
    ),
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
