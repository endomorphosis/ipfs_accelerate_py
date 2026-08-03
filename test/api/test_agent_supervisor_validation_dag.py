from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shlex
import sys
import threading

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.code_evidence_graph import (
    ChangedASTSymbol,
    CodeImpactIndex,
)
from ipfs_accelerate_py.agent_supervisor.proof.code_proof_obligations import (
    CandidateDiffEntry,
    DiffChangeKind,
)
from ipfs_accelerate_py.agent_supervisor.validation.proposal_validation import (
    ImplementationProposal,
    NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_REQUIREMENT_ID,
    ProposalFindingCode,
    ProposalValidationPolicy,
    validate_implementation_proposal,
)
from ipfs_accelerate_py.agent_supervisor.validation.validation_commands import (
    ValidationCommand,
    ValidationStage,
)
from ipfs_accelerate_py.agent_supervisor.validation.validation_runtime import (
    VALIDATION_PYTHON_ENV,
)
from ipfs_accelerate_py.agent_supervisor.validation.validation_scheduler import (
    HermeticValidationPolicy,
    ImpactSelectedValidationDAG,
    ImpactDependencyGraph,
    ImpactValidationCheck,
    ImpactValidationDAGReceipt,
    ImpactValidationKind,
    RepositoryValidationPolicy,
    STRICT_VALIDATION_GATE_KINDS,
    STRICT_VALIDATION_PARENT_OBJECTIVE_ID,
    STRICT_VALIDATION_SCHEDULER_GATE_KINDS,
    TRANSITIVE_IMPACT_REQUIREMENT_ID,
    StrictValidationDAGCompletionEvidence,
    ValidationDAGError,
    ValidationDAGNodeRecord,
    ValidationDAGReceipt,
    ValidationNodeDisposition,
    ValidationScheduler,
    build_declared_validation_plan_graph,
    build_impact_selected_validation_dag,
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
    assert evidence["task_id"] == proposal.task_id
    assert evidence["repository_id"] == proposal.repository_id
    assert evidence["repository_tree_id"] == proposal.repository_tree_id
    assert evidence["baseline_id"] == proposal.baseline_id
    assert evidence["diff_digest"] == proposal.diff_digest
    assert tuple(evidence["allowed_paths"]) == validation.policy.allowed_paths
    assert (
        tuple(evidence["task_owned_paths"])
        == validation.policy.task_owned_paths
    )
    assert tuple(evidence["changed_paths"]) == proposal.changed_paths
    assert tuple(evidence["gate_trace"]) == tuple(
        gate.value for gate in validation.receipt.gate_trace
    )
    assert tuple(report["proved_requirement_ids"]) == (G100_FAIL_FAST_REQUIREMENT,)
    assert tuple(receipt["proved_requirement_ids"]) == (G100_FAIL_FAST_REQUIREMENT,)
    assert evidence["requirement_id"] == G100_FAIL_FAST_REQUIREMENT
    for non_authority_field in (
        "proof_authoritative",
        "code_proof_authoritative",
        "completion_authoritative",
        "freshness_authoritative",
        "authoritative",
        "merge_eligible",
    ):
        assert report[non_authority_field] is False
        assert report["proposal_validation"][non_authority_field] is False
    assert (
        NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_REQUIREMENT_ID
        == G100_FAIL_FAST_REQUIREMENT
    )


def test_accepted_proposal_default_runner_uses_validation_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        "IPFS_ACCELERATE_VALIDATION_PYTHON_EXECUTABLE",
        raising=False,
    )
    validation = validate_implementation_proposal(
        _proposal((_source_change(),)),
        policy=_policy(),
    )
    validation_id = "strict-python-runtime"
    command = ValidationCommand(
        command="python -m pytest --version",
        stage=ValidationStage.TARGETED,
        impact_paths=("pkg/source.py",),
        validation_id=validation_id,
        cacheable=False,
    )
    graph = ImpactDependencyGraph(
        repository_tree_id=TREE_ID,
        dependencies={"pkg/source.py": ()},
        validation_targets={validation_id: ("pkg/source.py",)},
    )

    report = ValidationScheduler(max_workers=1).run_validated(
        validation,
        (command,),
        workspace_path=tmp_path,
        impact_graph=graph,
        dependency_state="fixture",
    )

    assert validation.accepted is True
    assert report["passed"] is True, report
    assert report["results"][0]["returncode"] == 0
    assert "pytest " in report["results"][0]["output"]


def test_validated_scheduler_rejects_unmarked_runner_under_hermetic_policy(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def runner(*, spec: ValidationCommand, **_kwargs: object) -> dict[str, object]:
        calls.append(spec.command)
        return {"returncode": 0, "output": "false hermetic pass"}

    validation = validate_implementation_proposal(
        _proposal((_source_change(),)),
        policy=_policy(),
    )
    command = ValidationCommand(
        command='test -z "${IPFS_ACCELERATE_VALIDATION_RUNTIME_ID-}"',
        stage=ValidationStage.TARGETED,
        impact_paths=("pkg/source.py",),
        validation_id="hermetic-capability",
        cacheable=False,
    )
    graph = ImpactDependencyGraph(
        repository_tree_id=TREE_ID,
        dependencies={"pkg/source.py": ()},
        validation_targets={"hermetic-capability": ("pkg/source.py",)},
    )

    report = ValidationScheduler(
        max_workers=1,
        runner=runner,
        hermetic_policy=HermeticValidationPolicy(),
    ).run_validated(
        validation,
        (command,),
        workspace_path=tmp_path,
        impact_graph=graph,
        dependency_state="fixture",
    )

    assert calls == []
    assert report["passed"] is False
    result = report["results"][0]
    assert result["returncode"] == 75
    assert (
        result["error"]
        == "hermetic_validation_runner_capability_missing"
    )
    assert result["outcome"] == "infrastructure_failure"
    assert result["authoritative"] is False


def test_strict_validation_builds_hardened_python_environment(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    profile_marker = tmp_path / "bash-environment-ran"
    bash_environment = tmp_path / "bash-environment"
    bash_environment.write_text(
        f"touch {shlex.quote(str(profile_marker))}\n",
        encoding="utf-8",
    )
    validation = validate_implementation_proposal(
        _proposal((_source_change(),)),
        policy=_policy(),
    )
    graph = ImpactDependencyGraph(
        repository_tree_id=TREE_ID,
        dependencies={"pkg/source.py": ()},
        validation_targets={VALIDATION_ID: ("pkg/source.py",)},
    )
    command = ValidationCommand(
        command=(
            "python -c 'import os, sys; "
            "assert os.environ.get("
            '"IPFS_ACCELERATE_VALIDATION_PYTHON_EXECUTABLE"); '
            'assert "VALIDATION_SECRET" not in os.environ; '
            "print(sys.executable)'"
        ),
        stage=ValidationStage.TARGETED,
        impact_paths=("pkg/source.py",),
        validation_id=VALIDATION_ID,
        cacheable=False,
    )

    report = ValidationScheduler(max_workers=1).run_validated(
        validation,
        (command,),
        workspace_path=workspace,
        impact_graph=graph,
        dependency_state="fixture",
        environment={
            "BASH_ENV": str(bash_environment),
            "VALIDATION_SECRET": "must-not-leak",
        },
    )

    assert report["passed"] is True, [
        (
            result.get("returncode"),
            result.get("output"),
            result.get("error"),
        )
        for result in report["results"]
    ]
    assert str(Path(sys.executable).resolve()) in str(
        report["results"][0]["output"]
    )
    assert not profile_marker.exists()


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


def test_strict_validation_runner_receives_sanitized_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    captured_environments: list[dict[str, str]] = []

    def runner(
        *,
        spec: ValidationCommand,
        environment: dict[str, str],
        **_kwargs: object,
    ) -> dict[str, object]:
        captured_environments.append(dict(environment))
        return _result(spec)

    monkeypatch.setenv(VALIDATION_PYTHON_ENV, "/usr/bin/python3")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-cross-validation-boundary")

    report = ValidationScheduler(max_workers=1).run_validated(
        validation,
        _commands(),
        workspace_path=tmp_path,
        impact_graph=graph,
        dependency_state="fixture",
        runner=runner,
    )

    assert report["attempted"] is True
    assert captured_environments
    assert all(
        environment["IPFS_ACCELERATE_VALIDATION_PYTHON_EXECUTABLE"]
        == str(Path("/usr/bin/python3").resolve())
        for environment in captured_environments
    )
    assert all(
        "AWS_SECRET_ACCESS_KEY" not in environment
        for environment in captured_environments
    )
    assert all(
        environment["HOME"] == "/nonexistent/ipfs-accelerate-validation"
        for environment in captured_environments
    )


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
    assert report["code_proof_authoritative"] is False
    assert receipt.code_proof_authoritative is False
    assert report["completion_authoritative"] is False
    assert report["freshness_authoritative"] is False
    assert report["merge_eligible"] is False
    assert G102_PROOF_CANDIDATE_REQUIREMENT not in receipt.proved_requirement_ids


def test_strict_validation_parent_projection_binds_complete_scheduler_gate_surface(
    tmp_path: Path,
) -> None:
    report, _calls = _failing_transitive_report(tmp_path)
    receipt = ValidationDAGReceipt.from_dict(report["validation_dag_receipt"])

    evidence = receipt.strict_validation_completion_evidence()
    restored = StrictValidationDAGCompletionEvidence.from_dict(
        evidence.to_dict()
    )

    assert restored.evidence_id == evidence.evidence_id
    assert restored.validation_dag.receipt_id == receipt.receipt_id
    assert restored.objective_id == STRICT_VALIDATION_PARENT_OBJECTIVE_ID
    assert restored.child_objective_id == OBJECTIVE_ID
    assert restored.requirement_id == G101_TRANSITIVE_IMPACT_REQUIREMENT
    assert restored.proved_requirement_ids == (
        G101_TRANSITIVE_IMPACT_REQUIREMENT,
    )
    assert restored.repository_tree_id == TREE_ID
    assert restored.validation_policy_id == receipt.policy_id
    assert restored.operational_receipt_id == receipt.receipt_id
    assert restored.gate_kinds == STRICT_VALIDATION_GATE_KINDS == (
        "schema",
        "authority",
        "patch",
        "path",
        "ast_interface",
        "impact_test",
        "semantic_proof",
        "merge",
        "freshness",
    )
    assert (
        restored.scheduler_gate_kinds
        == STRICT_VALIDATION_SCHEDULER_GATE_KINDS
        == ("impact_test", "semantic_proof", "merge", "freshness")
    )
    assert restored.impact_test_node_ids == tuple(
        node.node_id
        for node in receipt.nodes
        if node.validation_id == VALIDATION_ID
    )
    assert callable(restored.evaluate_parent_completion)
    with pytest.raises(
        TypeError,
        match="supplied by the scheduler evidence",
    ):
        restored.evaluate_parent_completion(validation_projection={})
    assert restored.qualifies is True
    assert restored.completion_authoritative is False

    payload = evidence.to_dict()
    assert payload["qualifies"] is True
    assert payload["completion_authoritative"] is False
    assert payload["gate_kinds"] == STRICT_VALIDATION_GATE_KINDS
    assert payload["receipt_id"] == payload["operational_receipt_id"]
    assert payload["proved_requirement_ids"] == (
        G101_TRANSITIVE_IMPACT_REQUIREMENT,
    )


def test_strict_validation_parent_projection_rejects_tamper_and_non_witness_dag(
    tmp_path: Path,
) -> None:
    report, _calls = _failing_transitive_report(tmp_path)
    receipt = ValidationDAGReceipt.from_dict(report["validation_dag_receipt"])
    payload = receipt.strict_validation_completion_evidence().to_dict()

    payload["gate_kinds"] = tuple(payload["gate_kinds"])[:-1]
    with pytest.raises(
        ValidationDAGError,
        match="completion projection is inconsistent",
    ):
        StrictValidationDAGCompletionEvidence.from_dict(payload)

    truncated = receipt.strict_validation_completion_evidence().to_dict()
    truncated.pop("scheduler_gate_kinds")
    with pytest.raises(
        ValidationDAGError,
        match="completion evidence is incomplete",
    ):
        StrictValidationDAGCompletionEvidence.from_dict(truncated)

    unknown = receipt.strict_validation_completion_evidence().to_dict()
    unknown["caller_authority"] = True
    with pytest.raises(
        ValidationDAGError,
        match="unknown fields: caller_authority",
    ):
        StrictValidationDAGCompletionEvidence.from_dict(unknown)

    completion_gate_tamper = deepcopy(
        receipt.strict_validation_completion_evidence().to_dict()
    )
    completion_gate = next(
        gate
        for gate in completion_gate_tamper["validation_dag"][
            "authority_gates"
        ]
        if gate["gate"] == "completion"
    )
    completion_gate["disposition"] = "pending"
    with pytest.raises(
        ValidationDAGError,
        match="identity mismatch|disposition does not match",
    ):
        StrictValidationDAGCompletionEvidence.from_dict(
            completion_gate_tamper
        )

    detached = receipt.to_dict()
    detached["transitive_evidence"] = None
    detached["proved_requirement_ids"] = []
    detached.pop("receipt_id", None)
    without_witness = ValidationDAGReceipt.from_dict(detached)
    with pytest.raises(
        ValidationDAGError,
        match="does not qualify.*parent completion projection",
    ):
        without_witness.strict_validation_completion_evidence()


def test_passing_all_stage_dag_cannot_claim_code_proof_or_completion_authority(
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
            command="semantic-translation",
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
        ValidationCommand(
            command="proof-attestation",
            stage=ValidationStage.ATTESTATION,
            cacheable=False,
            ordinal=5,
        ),
    )
    calls: list[str] = []

    def runner(*, spec: ValidationCommand, **_kwargs: object) -> dict[str, object]:
        calls.append(spec.command)
        return {"returncode": 0, "output": f"{spec.stage.label}: passed"}

    report = ValidationScheduler(max_workers=2).run_validated(
        validation,
        commands,
        workspace_path=tmp_path,
        impact_graph=graph,
        dependency_state="fixture",
        runner=runner,
    )
    receipt = ValidationDAGReceipt.from_dict(report["validation_dag_receipt"])

    assert calls == [spec.command for spec in commands]
    assert report["passed"] is True
    assert receipt.passed is True
    assert receipt.coverage_complete is True
    assert {node.stage for node in receipt.nodes} == {
        "cheap",
        "targeted",
        "translation",
        "solver",
        "kernel",
        "attestation",
    }
    assert {
        gate.disposition.value for gate in receipt.authority_gates
    } == {"pending"}
    assert receipt.proof_authoritative is False
    assert receipt.code_proof_authoritative is False
    assert receipt.completion_authoritative is False
    assert report["proof_authoritative"] is False
    assert report["code_proof_authoritative"] is False
    assert report["completion_authoritative"] is False
    assert report["merge_eligible"] is False
    assert G102_PROOF_CANDIDATE_REQUIREMENT not in receipt.proved_requirement_ids
    with pytest.raises(
        ValidationDAGError,
        match="does not qualify.*parent completion projection",
    ):
        receipt.strict_validation_completion_evidence()


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


def test_impact_graph_rejects_multi_node_dependency_cycles() -> None:
    with pytest.raises(ValidationDAGError, match="contains a cycle"):
        ImpactDependencyGraph(
            repository_tree_id=TREE_ID,
            dependencies={
                "pkg/a.py": ("pkg/b.py",),
                "pkg/b.py": ("pkg/c.py",),
                "pkg/c.py": ("pkg/a.py",),
            },
        )


def test_receipt_rejects_recanonicalized_incomplete_affected_closure(
    tmp_path: Path,
) -> None:
    report, _calls = _failing_transitive_report(tmp_path)
    payload = deepcopy(report["validation_dag_receipt"])
    payload["affected_paths"] = [
        "pkg/source.py",
        "test/api/test_transitive_consumer.py",
    ]
    # Simulate an attacker recomputing all outer identities.  Closure is a
    # semantic invariant and must fail before an identity can be accepted.
    payload.pop("receipt_id", None)
    payload["transitive_evidence"] = None
    payload["proved_requirement_ids"] = []

    with pytest.raises(ValidationDAGError, match="graph closure"):
        ValidationDAGReceipt.from_dict(payload)


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


def test_declared_validation_plan_builds_proposal_local_coverage(
    tmp_path: Path,
) -> None:
    validation = validate_implementation_proposal(
        _proposal((_source_change(),)),
        policy=_policy(),
    )
    commands, graph = build_declared_validation_plan_graph(
        ("pytest -q test/api/test_transitive_consumer.py",),
        repository_tree_id=TREE_ID,
        changed_paths=("pkg/source.py",),
    )
    calls: list[str] = []

    def runner(*, spec: ValidationCommand, **_kwargs: object) -> dict[str, object]:
        calls.append(spec.command)
        return {"returncode": 0, "output": "passed"}

    report = ValidationScheduler().run_validated(
        validation,
        commands,
        workspace_path=tmp_path,
        impact_graph=graph,
        dependency_state="fixture",
        runner=runner,
    )
    receipt = ValidationDAGReceipt.from_dict(
        report["validation_dag_receipt"]
    )

    assert calls == ["pytest -q test/api/test_transitive_consumer.py"]
    assert report["passed"] is True
    assert graph.graph_version == "declared-validation-plan-v1"
    assert receipt.coverage_complete is True
    assert receipt.affected_paths == (
        "pkg/source.py",
        "test/api/test_transitive_consumer.py",
    )
    assert len(receipt.required_validation_ids) == 1
    assert receipt.required_validation_ids[0].startswith("declared:")
    assert all(node.mandatory for node in receipt.nodes)
    assert tuple(report["proved_requirement_ids"]) == ()


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
    failed_id = nodes[
        "pytest -q test/api/test_transitive_consumer.py"
    ].node_id
    assert nodes["semantic-check"].blocked_by_failed_node_ids == (failed_id,)
    assert nodes["proof-solver"].blocked_by_failed_node_ids == (failed_id,)
    assert nodes["proof-kernel"].blocked_by_failed_node_ids == (failed_id,)


def test_fail_fast_same_stage_peer_is_recorded_without_false_dependency(
    tmp_path: Path,
) -> None:
    validation = validate_implementation_proposal(
        _proposal((_source_change(),)),
        policy=_policy(),
    )
    second_validation_id = "transitive-consumer-secondary"
    target = "test/api/test_transitive_consumer.py"
    graph = ImpactDependencyGraph(
        repository_tree_id=TREE_ID,
        dependencies={
            "pkg/consumer.py": ("pkg/source.py",),
            target: ("pkg/consumer.py",),
        },
        validation_targets={
            VALIDATION_ID: (target,),
            second_validation_id: (target,),
        },
    )
    commands = (
        _commands()[0],
        _commands()[1],
        ValidationCommand(
            command="pytest -q test/api/test_transitive_consumer_secondary.py",
            stage=ValidationStage.TARGETED,
            impact_paths=(target,),
            validation_id=second_validation_id,
            cacheable=False,
            ordinal=2,
        ),
    )
    calls: list[str] = []

    def runner(*, spec: ValidationCommand, **_kwargs: object) -> dict[str, object]:
        calls.append(spec.command)
        if spec.validation_id == VALIDATION_ID:
            return {
                "returncode": 7,
                "seeded_defect_id": "seed:g101",
            }
        return {"returncode": 0}

    report = ValidationScheduler(max_workers=1).run_validated(
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
    nodes = {node.command: node for node in receipt.nodes}
    peer = nodes[
        "pytest -q test/api/test_transitive_consumer_secondary.py"
    ]

    assert calls == [
        "python -m compileall -q pkg",
        "pytest -q test/api/test_transitive_consumer.py",
    ]
    assert peer.selected and peer.mandatory
    assert peer.disposition is ValidationNodeDisposition.BLOCKED
    assert peer.reason == "fail_fast_after_stage_failure"
    assert peer.blocked_by_failed_node_ids == ()
    assert receipt.coverage_complete
    assert receipt.proved_requirement_ids == (
        G101_TRANSITIVE_IMPACT_REQUIREMENT,
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
        lambda payload: payload.__setitem__("code_proof_authoritative", True),
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
        "code-proof-authority",
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


IMPACT_TREE_ID = "tree:ast-impact-candidate"
ACCEPTANCE_CRITERION = "The changed public value is preserved by consumers."


def _ast_impact_index(
    tree_id: str = IMPACT_TREE_ID,
) -> CodeImpactIndex:
    return CodeImpactIndex(
        repository_tree_id=tree_id,
        symbol_paths={
            "pkg.source.value": "pkg/source.py",
            "pkg.consumer.read": "pkg/consumer.py",
            "tests.integration.test_read": (
                "test/integration/test_consumer.py"
            ),
            "pkg.unrelated.noop": "pkg/unrelated.py",
        },
        symbol_dependencies={
            "pkg.consumer.read": ("pkg.source.value",),
            "tests.integration.test_read": ("pkg.consumer.read",),
        },
        path_dependencies={
            "pkg/consumer.py": ("pkg/source.py",),
            "test/integration/test_consumer.py": ("pkg/consumer.py",),
            "pkg/unrelated.py": (),
        },
        validation_targets={
            "integration": ("tests.integration.test_read",),
        },
    )


def _impact_checks(
    *,
    cacheable: bool = False,
) -> tuple[ImpactValidationCheck, ...]:
    return (
        ImpactValidationCheck(
            check_id="syntax",
            kind=ImpactValidationKind.SYNTAX,
            command="python -m compileall -q pkg",
            cacheable=cacheable,
        ),
        ImpactValidationCheck(
            check_id="type",
            kind=ImpactValidationKind.TYPE,
            command="python -m mypy pkg",
            cacheable=cacheable,
        ),
        ImpactValidationCheck(
            check_id="interface",
            kind=ImpactValidationKind.INTERFACE,
            command="python scripts/check_interface.py",
            cacheable=cacheable,
        ),
        ImpactValidationCheck(
            check_id="unit",
            kind=ImpactValidationKind.UNIT,
            command="pytest -q test/unit/test_source.py",
            cacheable=cacheable,
        ),
        ImpactValidationCheck(
            check_id="integration",
            kind=ImpactValidationKind.INTEGRATION,
            command="pytest -q test/integration/test_consumer.py",
            targets=("tests.integration.test_read",),
            cacheable=cacheable,
        ),
        ImpactValidationCheck(
            check_id="contract",
            kind=ImpactValidationKind.CONTRACT,
            command="pytest -q test/contract/test_public_api.py",
            acceptance_criteria=(ACCEPTANCE_CRITERION,),
            cacheable=cacheable,
        ),
        ImpactValidationCheck(
            check_id="runtime",
            kind=ImpactValidationKind.RUNTIME,
            command="python scripts/smoke.py",
            cacheable=cacheable,
        ),
        ImpactValidationCheck(
            check_id="unrelated-unit",
            kind=ImpactValidationKind.UNIT,
            command="pytest -q test/unit/test_unrelated.py",
            targets=("pkg.unrelated.noop",),
            cacheable=cacheable,
        ),
    )


def _changed_public_symbol() -> ChangedASTSymbol:
    return ChangedASTSymbol(
        symbol="pkg.source.value",
        path="pkg/source.py",
        interface_changed=True,
    )


def test_rejected_proposal_closes_every_proposal_bound_scheduler_entrypoint(
    tmp_path: Path,
) -> None:
    validation = validate_implementation_proposal(
        _proposal((), declared_paths=()),
        policy=_policy(),
    )
    calls: list[str] = []

    report = ValidationScheduler().run_impact_selected(
        _impact_checks(),
        workspace_path=tmp_path,
        impact_index=_ast_impact_index(),
        proposal_validation=validation,
        dependency_state="fixture",
        runner=lambda *, spec, **_kwargs: calls.append(spec.command),
    )

    assert calls == []
    assert report["error"] == "proposal_validation_failed"
    assert tuple(report["proved_requirement_ids"]) == (
        NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_REQUIREMENT_ID,
    )
    assert report["impact_validation_receipt"] is None
    assert {node["disposition"] for node in report["nodes"]} == {"blocked"}
    for non_authority_field in (
        "proof_authoritative",
        "code_proof_authoritative",
        "completion_authoritative",
        "freshness_authoritative",
        "authoritative",
        "merge_eligible",
    ):
        assert report[non_authority_field] is False


def test_impact_selected_runner_receives_sanitized_validation_environment(
    tmp_path: Path,
) -> None:
    environments: list[dict[str, str]] = []

    def runner(
        *,
        environment: dict[str, str],
        **_kwargs: object,
    ) -> dict[str, object]:
        environments.append(dict(environment))
        return {"returncode": 0, "output": "passed"}

    report = ValidationScheduler(max_workers=1).run_impact_selected(
        _impact_checks(),
        workspace_path=tmp_path,
        impact_index=_ast_impact_index(),
        changed_symbols=(_changed_public_symbol(),),
        acceptance_criteria=(ACCEPTANCE_CRITERION,),
        environment={"UNSAFE_PARENT_SECRET": "must-not-propagate"},
        dependency_state="fixture",
        runner=runner,
    )

    assert report["passed"] is True, report
    assert environments
    assert all(
        environment.get("IPFS_ACCELERATE_VALIDATION_PYTHON_EXECUTABLE")
        and "UNSAFE_PARENT_SECRET" not in environment
        for environment in environments
    )


def test_impact_selected_explicit_hermetic_policy_rejects_unmarked_runner(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def runner(*, spec: ValidationCommand, **_kwargs: object) -> dict[str, object]:
        calls.append(spec.command)
        return {"returncode": 0, "output": "false hermetic pass"}

    report = ValidationScheduler(max_workers=1).run_impact_selected(
        _impact_checks(),
        workspace_path=tmp_path,
        impact_index=_ast_impact_index(),
        changed_symbols=(_changed_public_symbol(),),
        acceptance_criteria=(ACCEPTANCE_CRITERION,),
        dependency_state="fixture",
        runner=runner,
        hermetic_policy=HermeticValidationPolicy(required_techniques=()),
    )

    assert calls == []
    assert report["passed"] is False
    result = report["results"][0]
    assert result["returncode"] == 75
    assert (
        result["error"]
        == "hermetic_validation_runner_capability_missing"
    )
    assert result["outcome"] == "infrastructure_failure"
    assert result["authoritative"] is False


def test_ast_impact_index_selects_transitive_consumer_outside_changed_path() -> None:
    index = _ast_impact_index()
    impact = index.impact(changed_symbols=(_changed_public_symbol(),))

    assert impact.changed_symbols == ("pkg.source.value",)
    assert impact.affected_symbols == (
        "pkg.consumer.read",
        "pkg.source.value",
        "tests.integration.test_read",
    )
    assert impact.affected_paths == (
        "pkg/consumer.py",
        "pkg/source.py",
        "test/integration/test_consumer.py",
    )
    assert impact.dependency_chains["tests.integration.test_read"] == (
        "pkg.source.value",
        "pkg.consumer.read",
        "tests.integration.test_read",
    )
    assert impact.required_validation_ids == ("integration",)
    assert CodeImpactIndex.from_dict(index.to_dict()) == index


def test_dag_derives_all_mandatory_kinds_acceptance_and_skip_reasons() -> None:
    plan = build_impact_selected_validation_dag(
        impact_index=_ast_impact_index(),
        checks=_impact_checks(),
        changed_symbols=(_changed_public_symbol(),),
        acceptance_criteria=(ACCEPTANCE_CRITERION,),
        repository_policy=RepositoryValidationPolicy(),
    )
    restored = ImpactSelectedValidationDAG.from_dict(plan.to_dict())
    selected = {node.check_id: node for node in plan.selected_nodes}

    assert restored.dag_id == plan.dag_id
    assert plan.coverage_complete
    assert {node.check.kind for node in selected.values()} == set(
        ImpactValidationKind
    )
    assert "changed_ast_interface" in selected["interface"].selection_reasons
    assert (
        "dependency_graph_validation_target"
        in selected["integration"].selection_reasons
    )
    assert (
        f"task_acceptance:{ACCEPTANCE_CRITERION}"
        in selected["contract"].selection_reasons
    )
    assert selected["interface"].depends_on == ("type",)
    assert selected["unit"].depends_on == ("type",)
    assert selected["integration"].depends_on == ("interface", "unit")
    assert selected["contract"].depends_on == ("interface", "unit")
    assert selected["runtime"].depends_on == ("contract", "integration")
    unrelated = next(
        node for node in plan.nodes if node.check_id == "unrelated-unit"
    )
    assert unrelated.selected is False
    assert (
        unrelated.skipped_reason
        == "no_changed_symbol_dependency_acceptance_or_policy_match"
    )
    forged = deepcopy(plan.to_dict())
    forged["impact"]["affected_symbols"] = ["pkg.source.value"]
    forged.pop("dag_id")
    with pytest.raises(ValidationDAGError, match="complete graph closure"):
        ImpactSelectedValidationDAG.from_dict(forged)


def test_uncovered_acceptance_or_mandatory_kind_fails_closed(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def runner(*, spec: ValidationCommand, **_kwargs: object) -> dict[str, object]:
        calls.append(spec.command)
        return {"returncode": 0}

    checks = tuple(
        check
        for check in _impact_checks()
        if check.kind is not ImpactValidationKind.RUNTIME
    )
    report = ValidationScheduler().run_impact_selected(
        checks,
        workspace_path=tmp_path,
        impact_index=_ast_impact_index(),
        changed_symbols=(_changed_public_symbol(),),
        acceptance_criteria=("An unmapped criterion.",),
        dependency_state="fixture",
        runner=runner,
    )

    assert calls == []
    assert report["attempted"] is False
    assert report["passed"] is False
    assert report["error"] == "uncovered_validation_impact"
    assert "missing_mandatory_runtime_check" in report["uncovered_impact"]
    assert (
        "uncovered_acceptance:An unmapped criterion."
        in report["uncovered_impact"]
    )
    assert report["time_to_first_useful_failure_ms"] == 0.0
    assert len(report["nodes"]) == len(checks)


def test_dependency_aware_dag_parallelism_fail_fast_and_complete_receipt(
    tmp_path: Path,
) -> None:
    interface_and_unit = threading.Barrier(2)
    calls: list[str] = []
    calls_lock = threading.Lock()

    def runner(*, spec: ValidationCommand, **_kwargs: object) -> dict[str, object]:
        with calls_lock:
            calls.append(spec.validation_id)
        if spec.validation_id in {"interface", "unit"}:
            interface_and_unit.wait(timeout=5)
        failed = spec.validation_id == "integration"
        return {
            "returncode": 7 if failed else 0,
            "seeded_defect_id": "seed:provider-defect" if failed else "",
        }

    report = ValidationScheduler(
        max_workers=2,
        resource_budget=2,
    ).run_impact_selected(
        _impact_checks(),
        workspace_path=tmp_path,
        impact_index=_ast_impact_index(),
        changed_symbols=(_changed_public_symbol(),),
        acceptance_criteria=(ACCEPTANCE_CRITERION,),
        dependency_state="fixture",
        runner=runner,
    )
    receipt = ImpactValidationDAGReceipt.from_dict(
        report["impact_validation_receipt"]
    )
    nodes = {node.check_id: node for node in receipt.nodes}

    assert report["passed"] is False
    assert report["first_useful_failure_check_id"] == "integration"
    assert report["time_to_first_useful_failure_ms"] >= 0
    assert set(calls).issuperset(
        {"syntax", "type", "interface", "unit", "integration"}
    )
    assert nodes["integration"].disposition is ValidationNodeDisposition.FAILED
    assert (
        nodes["integration"].observed_seeded_defect_id
        == "seed:provider-defect"
    )
    assert nodes["runtime"].disposition is ValidationNodeDisposition.BLOCKED
    assert nodes["runtime"].blocked_by == ("integration",)
    assert (
        nodes["unrelated-unit"].disposition
        is ValidationNodeDisposition.OMITTED
    )
    assert len(nodes) == len(_impact_checks())
    assert report["selection_reasons"]["integration"]
    assert report["skipped_reasons"]["unrelated-unit"]
    assert receipt.receipt_id == report["impact_validation_receipt"]["receipt_id"]
