"""Actual captured source, Python/Lean observations and signed candidate custody."""
from pathlib import Path
import json
import pytest

from benchmarks.agent_supervisor.container_coding.local_planning_qualification import prepare_local_task
from ipfs_accelerate_py.agent_supervisor.runtime.repository_finite_handoff import prepare_finite_repository_handoff
from ipfs_accelerate_py.agent_supervisor.runtime.repository_finite_runner import materialize_finite_candidate
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from test.api.test_finite_integer_codebase import finite_prepared, finite_tools, finite_git
from test.api.test_finite_integer_plan_preview import preview_arguments


@pytest.mark.parametrize("instruction_path", [None, "instruction.txt"])
def test_fresh_matched_facts_and_residual_select_a_real_checked_candidate(finite_prepared, finite_tools, tmp_path, instruction_path):
    if instruction_path is not None:
        from test.api.test_finite_integer_codebase import finite_text, VIEW
        (finite_prepared["repository"] / instruction_path).write_text(finite_text())
        finite_git(finite_prepared["repository"], "add", instruction_path)
        finite_git(finite_prepared["repository"], "commit", "-qm", "Complete immutable instruction")
        finite_prepared["expected_head"] = finite_prepared["index"].prepare_current(
            finite_prepared["repository"], repository_id=VIEW, operation_id="instruction",
            expected_head=finite_prepared["expected_head"], scheduler=finite_prepared["scheduler"]).head
    options = preview_arguments(finite_prepared, finite_tools)
    options.pop("output")
    repository = finite_prepared["repository"]
    (repository / ".git/info/exclude").write_text(".runtime/\n")
    with IntentRepository(tmp_path / "intent.duckdb") as intent:
        declared = prepare_local_task(repository=repository, state=tmp_path / "policy", intent=intent,
            scope_paths=["calc.py", *([instruction_path] if instruction_path else [])], output_path="calc.py",
            objective=options["source_text"].replace("\n", " ") if instruction_path else options["source_text"],
            validation_argv=["python3", "-B", "-c", "from calc import increment; assert increment(0)==2"])
        result = prepare_finite_repository_handoff(**options, admission=declared["admission"], intent=intent,
            task_cid=declared["task_cid"], state=tmp_path / "handoff", instruction_path=instruction_path)
        assert result["status"] == "candidate_ready"
        assert result["preview"]["current_facts_count"] == 1
        assert result["preview"]["selected_task_ids"] == ["task:finite:offset"]
        assert result["candidate_observation"]["type_clause_satisfied"]
        assert result["candidate_observation"]["offset_clause_satisfied"]
        assert result["native_task_population_unchanged"]
        assert intent.get_task(declared["task_cid"])["status"] == "ready"
        assert (repository / "calc.py").read_text().endswith("return n + 1\n")
        workspace = tmp_path / "allocated"
        finite_git(repository, "worktree", "add", "--detach", str(workspace), "HEAD")
        binding = result["signed_evidence"]["binding"]
        materialized = materialize_finite_candidate(artifact=Path(result["handoff_path"]),
            expected_sha256=result["handoff_sha256"], task_cid=declared["task_cid"],
            owner_did=binding["identity"], profile_id=binding["profile_id"],
            prompt=json.dumps({"objective_id": declared["task_id"]}), workspace=workspace)
        assert materialized["status"] == "candidate_materialized"
        assert (workspace / "calc.py").read_text().endswith("return n + 2\n")
        assert (repository / "calc.py").read_text().endswith("return n + 1\n")


def test_already_satisfied_cannot_complete_or_omit_a_signed_task(finite_prepared, finite_tools, tmp_path):
    from test.api.test_finite_integer_codebase import finite_source, VIEW
    repository = finite_prepared["repository"]
    (repository / "calc.py").write_bytes(finite_source(2))
    finite_git(repository, "add", "calc.py")
    finite_git(repository, "commit", "-qm", "Already satisfies the finite instruction")
    finite_prepared["expected_head"] = finite_prepared["index"].prepare_current(repository,
        repository_id=VIEW, operation_id="already-satisfied", expected_head=finite_prepared["expected_head"],
        scheduler=finite_prepared["scheduler"]).head
    options = preview_arguments(finite_prepared, finite_tools); options.pop("output")
    with IntentRepository(tmp_path / "intent.duckdb") as intent:
        declared = prepare_local_task(repository=repository, state=tmp_path / "policy", intent=intent,
            scope_paths=["calc.py"], output_path="calc.py", objective=options["source_text"],
            validation_argv=["python3", "-B", "-c", "from calc import increment; assert increment(0)==2"])
        result = prepare_finite_repository_handoff(**options, admission=declared["admission"], intent=intent,
            task_cid=declared["task_cid"], state=tmp_path / "handoff")
        assert result["status"] == "residual"
        assert result["preview"]["current_facts_count"] == 2
        assert result["handoff_path"] is None
        assert intent.get_task(declared["task_cid"])["status"] == "ready"
