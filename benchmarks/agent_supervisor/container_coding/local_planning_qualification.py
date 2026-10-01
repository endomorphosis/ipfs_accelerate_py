"""Actual signed local planning/acceptance lifecycle, with no model calls.

This is a contract qualification, not a coding benchmark score: its proposal
and deterministic repair are explicitly authored by the qualification driver.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import Supervisor
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
    PromptAcceptanceRecord,
    PromptGoalGraph,
    PromptGoalRecord,
    PromptOutputRecord,
    PromptTaskRecord,
    PromptValidationRecord,
)
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    IntentCompletionError,
    IntentRepository,
)


def prepare_local_task(
    *,
    repository: Path,
    state: Path,
    intent: IntentRepository,
    scope_paths: list[str],
    output_path: str,
    validation_argv: list[str],
    objective: str,
    profile_dir: Path | None = None,
    lifecycle_dir: Path | None = None,
) -> dict:
    """Admit one independently declared task in the caller's real repository.

    The source must already have a real Git baseline. ``state`` and owner keys
    are outside the worker repository. No validation passes are asserted here.
    ``intent`` may point at an initialized native file or a live Quack owner.
    """
    repository, state = Path(repository).resolve(strict=True), Path(state).absolute()
    if state.is_relative_to(repository):
        raise ValueError("qualification state must be outside the worker repository")
    state.mkdir(parents=True, exist_ok=True)
    profile_dir = profile_dir or state / "profile"
    lifecycle_dir = lifecycle_dir or state / "lifecycle"
    Supervisor.init_local(
        repository=repository, consent=True, profile_dir=profile_dir, lifecycle_dir=lifecycle_dir
    )
    policy = content_identity(local.LOCAL_POLICY)
    check = PromptValidationRecord(
        validation_key="local-public-check", argv=tuple(validation_argv), policy_cid=policy
    )
    acceptance = PromptAcceptanceRecord(
        criterion_key="local-public-acceptance",
        criterion=objective,
        validation_keys=(check.validation_key,),
    )
    output = PromptOutputRecord(path=output_path, effect="modify", media_type="text/x-python")
    spec = {
        "task_key": "LOCAL-TASK",
        "scope_paths": sorted(set(scope_paths)),
        "dependencies": [],
        "outputs": [
            {"path": output.path, "effect": output.effect, "media_type": output.media_type}
        ],
        "validations": [
            {
                name: local._plain(getattr(check, name))
                for name in ("validation_key", "argv", "cwd", "expected_exit_codes", "policy_cid")
            }
        ],
        "acceptance": [
            {
                name: local._plain(getattr(acceptance, name))
                for name in ("criterion_key", "criterion", "evidence_cids", "validation_keys")
            }
        ],
    }
    observed_sources = local._sources(
        repository, [name for name in local._git(repository, "ls-files", "-z").split("\0") if name]
    )
    roots = {
        "request_cid": content_identity(
            {
                "schema": "local-independent-planning-request@1",
                "objective": objective,
                "task_spec": spec,
            }
        ),
        "scan_cid": content_identity(
            {"schema": "local-observed-source-inventory@1", "sources": observed_sources}
        ),
        "program_root": content_identity(
            {
                "schema": "local-observed-program@1",
                "baseline_commit": local._git(repository, "rev-parse", "HEAD"),
                "source_root": local._tree(observed_sources),
            }
        ),
    }
    # Independent benchmark scope is signed before constructing its proposal.
    manifest = local.author_local_benchmark_manifest(
        repository=repository,
        profile_dir=profile_dir,
        lifecycle_dir=lifecycle_dir,
        task_specs=[spec],
        planning_roots=roots,
    )
    goal = PromptGoalRecord(
        goal_key="LOCAL-GOAL",
        parent_goal_cid="",
        dependency_goal_cids=(),
        title=objective,
        objective=objective,
        rationale="Independently declared isolated benchmark acceptance",
        scope_paths=tuple(scope_paths),
        acceptance=(acceptance,),
    )
    task = PromptTaskRecord(
        task_key="LOCAL-TASK",
        goal_cid=goal.goal_cid,
        dependency_task_cids=(),
        objective=objective,
        rationale="Perform the explicitly bounded local repair",
        scope_paths=tuple(scope_paths),
        outputs=(output,),
        validations=(check,),
        acceptance=(acceptance,),
        evidence_cids=(),
        policy_roots=(policy,),
        predicted_files=(output_path,),
    )
    graph = PromptGoalGraph(
        **roots, policy_roots=(policy,), goals=(goal,), tasks=(task,), evidence=()
    )
    admission = local.admit_local_benchmark_plan(graph=graph, manifest=manifest)
    materialized = local.materialize_local_benchmark_plan(admission=admission, intent=intent)
    result = {
        "schema": "local-planning-qualification-preparation@1",
        "repository": str(repository),
        "profile_dir": str(profile_dir),
        "lifecycle_dir": str(lifecycle_dir),
        "intent_database": str(intent.database_path),
        "task_cid": task.task_cid,
        "task_id": task.task_key,
        "task_title": task.objective,
        "paths": list(task.scope_paths),
        "required_raw_paths": [output_path],
        "graph": graph.to_dict(),
        "manifest": manifest,
        "admission": admission,
        "materialized": materialized,
        "proposal_origin": "independent-qualification-declaration",
        "model_calls": 0,
        "completion_authority": False,
    }
    (state / "prepared.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


def prepare_local_planning_qualification(output: Path, *, intent_database=None) -> dict:
    """Create a disposable actual failing repository and leave its task ready."""
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    repository = output / "repository"
    repository.mkdir()
    (repository / "answer.py").write_text("def answer():\n    return 1\n")
    (repository / "test_answer.py").write_text(
        "from answer import answer\nassert answer() == 2, 'answer must be two'\n"
    )
    for argv in [
        ["init", "-q"],
        ["add", "."],
        [
            "-c",
            "user.name=Local qualification",
            "-c",
            "user.email=local@example.invalid",
            "commit",
            "-qm",
            "independent public acceptance",
        ],
    ]:
        subprocess.run(["git", "-C", str(repository), *argv], check=True, capture_output=True)
    target = intent_database or output / "intent.duckdb"
    with IntentRepository(target) as intent:
        result = prepare_local_task(
            repository=repository,
            state=output / "state",
            intent=intent,
            scope_paths=["answer.py", "test_answer.py"],
            output_path="answer.py",
            validation_argv=[sys.executable, "-B", "test_answer.py"],
            objective="Return two and pass the immutable public answer check",
        )
    return result


def qualify_local_planning(output: Path) -> dict:
    started = time.monotonic()
    prepared = prepare_local_planning_qualification(output)
    with IntentRepository(prepared["intent_database"], install_schema=False) as intent:
        cid = prepared["task_cid"]
        task = intent.get_task(cid)
        intent.cas_task_status(
            task_cid=cid, expected_revision=task["revision"], new_status="in_progress"
        )
        failed = local.run_local_task_validations(
            intent=intent, task_cid=cid, attempt_id="local-qualification:before"
        )
        if failed["passed"]:
            raise RuntimeError("qualification fixture unexpectedly passed before repair")
        try:
            intent.cas_task_status(
                task_cid=cid,
                expected_revision=intent.get_task(cid)["revision"],
                new_status="completed",
                receipt=prepared["admission"]["receipt"]["payload"],
                allow_completion_without_evidence=True,
            )
        except IntentCompletionError:
            rejected = True
        else:
            raise RuntimeError("planning/failed validation bypassed native completion gate")
        (Path(prepared["repository"]) / "answer.py").write_text("def answer():\n    return 2\n")
        passed = local.run_local_task_validations(
            intent=intent, task_cid=cid, attempt_id="local-qualification:after"
        )
        if not passed["passed"]:
            raise RuntimeError("exact public validation did not pass after deterministic repair")
        completion = intent.cas_task_status(
            task_cid=cid,
            expected_revision=intent.get_task(cid)["revision"],
            new_status="completed",
            evidence_digests=[row["evidence_digest"] for row in passed["results"]],
        )
        final = intent.get_task(cid)
        result = {
            "schema": "local-planning-lifecycle-qualification@1",
            "qualified": True,
            "proposal_origin": prepared["proposal_origin"],
            "repair_origin": "deterministic-qualification-driver",
            "task_cid": cid,
            "manifest_cid": prepared["materialized"]["manifest_cid"],
            "pending_cid": prepared["materialized"]["pending_cid"],
            "plan_id": prepared["materialized"]["plan_id"],
            "before": failed,
            "after": passed,
            "planning_completion_rejected": rejected,
            "final_status": final["status"],
            "completion_event_id": completion.event_id,
            "elapsed_seconds": time.monotonic() - started,
            "model_calls": 0,
            "provider_tokens": 0,
            "code_proofs_created": 0,
            "production_activation": False,
            "daemon_coding_exercised": False,
            "benchmark_performance_claimed": False,
        }
        (Path(output) / "result.json").write_text(
            json.dumps(result, sort_keys=True, indent=2) + "\n"
        )
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    result = (
        prepare_local_planning_qualification(args.output)
        if args.prepare_only
        else qualify_local_planning(args.output)
    )
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
