"""Real local owner + native pending tasks + actually failing/passing Python."""

from copy import deepcopy
from dataclasses import replace
import subprocess
import sys

import pytest

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
    IntentRepository,
    IntentCompletionError,
)


@pytest.fixture
def scenario(tmp_path):
    repository = tmp_path / "isolated"
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
            "fixture",
        ],
    ]:
        subprocess.run(["git", "-C", str(repository), *argv], check=True)
    profile = tmp_path / "profile"
    lifecycle = tmp_path / "lifecycle"
    Supervisor.init_local(
        repository=repository, consent=True, profile_dir=profile, lifecycle_dir=lifecycle
    )
    policy = content_identity(local.LOCAL_POLICY)
    validation = PromptValidationRecord(
        validation_key="public-answer", argv=(sys.executable, "test_answer.py"), policy_cid=policy
    )
    acceptance = PromptAcceptanceRecord(
        criterion_key="answer-is-two",
        criterion="The public answer check passes",
        validation_keys=("public-answer",),
    )
    output = PromptOutputRecord(path="answer.py", effect="modify", media_type="text/x-python")
    goal = PromptGoalRecord(
        goal_key="LOCAL-GOAL",
        parent_goal_cid="",
        dependency_goal_cids=(),
        title="Repair the answer",
        objective="Return the required answer",
        rationale="Public benchmark acceptance",
        scope_paths=("answer.py", "test_answer.py"),
        acceptance=(acceptance,),
    )
    task = PromptTaskRecord(
        task_key="LOCAL-TASK",
        goal_cid=goal.goal_cid,
        dependency_task_cids=(),
        objective="Make the public answer check pass",
        rationale="Implement the declared repair",
        scope_paths=("answer.py", "test_answer.py"),
        outputs=(output,),
        validations=(validation,),
        acceptance=(acceptance,),
        evidence_cids=(),
        policy_roots=(policy,),
        predicted_files=("answer.py",),
    )
    roots = {
        "request_cid": content_identity({"prompt": "Repair the answer"}),
        "scan_cid": content_identity(
            {"sources": local._sources(repository, ["answer.py", "test_answer.py"])}
        ),
        "program_root": content_identity(
            {
                "git_tree": subprocess.check_output(
                    ["git", "-C", str(repository), "rev-parse", "HEAD^{tree}"], text=True
                ).strip()
            }
        ),
    }
    graph = PromptGoalGraph(
        **roots, policy_roots=(policy,), goals=(goal,), tasks=(task,), evidence=()
    )
    specs = [
        {
            "task_key": task.task_key,
            "scope_paths": list(task.scope_paths),
            "dependencies": [],
            "outputs": [
                {"path": output.path, "effect": output.effect, "media_type": output.media_type}
            ],
            "validations": [
                {
                    name: local._plain(getattr(validation, name))
                    for name in (
                        "validation_key",
                        "argv",
                        "cwd",
                        "expected_exit_codes",
                        "policy_cid",
                    )
                }
            ],
            "acceptance": [
                {
                    name: local._plain(getattr(acceptance, name))
                    for name in ("criterion_key", "criterion", "evidence_cids", "validation_keys")
                }
            ],
        }
    ]
    manifest = local.author_local_benchmark_manifest(
        repository=repository,
        profile_dir=profile,
        lifecycle_dir=lifecycle,
        task_specs=specs,
        planning_roots=roots,
    )
    with IntentRepository(tmp_path / "intent.duckdb") as intent:
        yield {
            "repository": repository,
            "manifest": manifest,
            "graph": graph,
            "intent": intent,
            "task_cid": task.task_cid,
            "profile": profile,
            "lifecycle": lifecycle,
        }


def _start(scenario):
    admission = local.admit_local_benchmark_plan(
        graph=scenario["graph"], manifest=scenario["manifest"]
    )
    materialized = local.materialize_local_benchmark_plan(
        admission=admission, intent=scenario["intent"]
    )
    intent, cid = scenario["intent"], scenario["task_cid"]
    task = intent.get_task(cid)
    intent.cas_task_status(
        task_cid=cid, expected_revision=task["revision"], new_status="in_progress"
    )
    return admission, materialized


def _complete(scenario, **kwargs):
    intent, cid = scenario["intent"], scenario["task_cid"]
    task = intent.get_task(cid)
    return intent.cas_task_status(
        task_cid=cid, expected_revision=task["revision"], new_status="completed", **kwargs
    )


def test_real_failing_acceptance_can_plan_but_only_observed_pass_can_complete(scenario):
    admission, materialized = _start(scenario)
    planning = admission["receipt"]["payload"]
    assert planning["planning_permitted"] is True
    assert planning["completion_authority"] is False
    assert planning["code_proof_authority"] is False
    assert planning["plan_evidence"]["status"] == "consistent"
    assert all(
        row["phase"] == "post_execution" and row["required"]
        for row in planning["pending_requirements"]
    )
    assert materialized["pending_cid"] == planning["pending_cid"]
    assert scenario["intent"].get_task(scenario["task_cid"])["status"] == "in_progress"
    with pytest.raises(IntentCompletionError, match="local pending"):
        _complete(scenario, receipt=planning, allow_completion_without_evidence=True)
    failed = local.run_local_task_validations(
        intent=scenario["intent"], task_cid=scenario["task_cid"], attempt_id="actual-repair:1"
    )
    assert failed["passed"] is False
    assert failed["results"][0]["outcome"] == "failed"
    with pytest.raises(IntentCompletionError, match="local pending"):
        _complete(scenario)
    (scenario["repository"] / "answer.py").write_text("def answer():\n    return 2\n")
    passed = local.run_local_task_validations(
        intent=scenario["intent"], task_cid=scenario["task_cid"], attempt_id="actual-repair:2"
    )
    assert passed["passed"] is True
    assert passed["source_tree_id"] != failed["source_tree_id"]
    _complete(scenario, evidence_digests=[passed["results"][0]["evidence_digest"]])
    assert scenario["intent"].get_task(scenario["task_cid"])["status"] == "completed"


def test_stale_result_and_foreign_test_mutation_cannot_complete(scenario):
    _start(scenario)
    (scenario["repository"] / "answer.py").write_text("def answer():\n    return 2\n")
    result = local.run_local_task_validations(
        intent=scenario["intent"], task_cid=scenario["task_cid"], attempt_id="actual:1"
    )
    assert result["passed"]
    (scenario["repository"] / "answer.py").write_text("def answer():\n    return 3\n")
    with pytest.raises(IntentCompletionError, match="local pending"):
        _complete(scenario, allow_completion_without_evidence=True)
    (scenario["repository"] / "test_answer.py").write_text("pass\n")
    with pytest.raises(local.LocalPlanningError, match="outside permitted"):
        local.run_local_task_validations(
            intent=scenario["intent"], task_cid=scenario["task_cid"], attempt_id="actual:2"
        )


@pytest.mark.parametrize("change", ["remove", "replace", "validation", "complete", "dependencies"])
def test_native_owner_preserves_local_contract(scenario, change):
    _start(scenario)
    intent, cid = scenario["intent"], scenario["task_cid"]
    task = intent.get_task(cid)
    body = deepcopy(task["body"])
    kwargs = {}
    if change == "remove":
        body.pop(local.CONTRACT_KEY)
    elif change == "replace":
        body[local.CONTRACT_KEY]["payload"]["pending_requirements"] = []
    elif change == "validation":
        kwargs["validations"] = [[sys.executable, "-c", "pass"]]
    elif change == "complete":
        kwargs["status"] = "completed"
    else:
        kwargs["dependencies"] = ["another-task"]
    with pytest.raises(local.LocalPlanningError):
        intent.upsert_task(
            task_cid=cid,
            task_alias=task["task_alias"],
            goal_cid=task["goal_cid"],
            body=body,
            identity=task["identity"],
            expected_revision=task["revision"],
            **kwargs,
        )


def test_forged_phase_or_external_policy_cannot_defer_proof(scenario):
    forged = deepcopy(scenario["manifest"])
    forged["payload"]["policy"]["explicit_proof_obligations"] = ["safety-kernel-proof"]
    # Even an owner-signed extra obligation is unsupported, not silently lost.
    forged = local._signed(forged["payload"], forged["payload"])
    with pytest.raises(local.LocalPlanningError, match="explicit proof"):
        local.admit_local_benchmark_plan(graph=scenario["graph"], manifest=forged)
    forged = deepcopy(scenario["manifest"])
    forged["payload"]["tasks"][0]["phase"] = "post_execution"
    forged = local._signed(forged["payload"], forged["payload"])
    with pytest.raises(local.LocalPlanningError, match="phase/proofs"):
        local.admit_local_benchmark_plan(graph=scenario["graph"], manifest=forged)
    graph = replace(
        scenario["graph"],
        policy_roots=(*scenario["graph"].policy_roots, content_identity({"external": "safety"})),
    )
    with pytest.raises(local.LocalPlanningError, match="external policy"):
        local.admit_local_benchmark_plan(graph=graph, manifest=scenario["manifest"])


def test_changed_command_and_tampered_owner_signature_rejected(scenario):
    original = scenario["graph"]
    task = original.tasks[0]
    validation = replace(task.validations[0], argv=(sys.executable, "-c", "pass"))
    graph = replace(original, tasks=(replace(task, validations=(validation,)),))
    with pytest.raises(local.LocalPlanningError, match="changed signed"):
        local.admit_local_benchmark_plan(graph=graph, manifest=scenario["manifest"])
    forged = deepcopy(scenario["manifest"])
    forged["payload"]["sources"]["answer.py"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="signature"):
        local.admit_local_benchmark_plan(graph=original, manifest=forged)


def test_goal_check_bindings_are_retained_without_manufacturing_evidence(scenario):
    declared, _, sources = local._manifest(scenario["manifest"], initial=True)
    compiled, validation, pending = local._graph_contract(
        scenario["graph"], declared, local._tree(sources)
    )
    goal = next(row for row in compiled.plan.evidence_requirements if row.kind.value == "review")
    assert goal.fallback_check_ids == ("public-answer",)
    assert set(goal.source_scope_ids) == {"answer.py", "test_answer.py"}
    assert compiled.proof_results == ()
    assert validation["plan_check_only"] is True
    assert any(row["requirement_id"] == goal.requirement_id for row in pending)
    unknown = replace(
        scenario["graph"].root_goal.acceptance[0], validation_keys=("unknown-safety-check",)
    )
    goal_record = replace(scenario["graph"].root_goal, acceptance=(unknown,))
    # The goal CID changed; update only the task's parent link, not its checks.
    graph = replace(
        scenario["graph"],
        goals=(goal_record,),
        tasks=(replace(scenario["graph"].tasks[0], goal_cid=goal_record.goal_cid),),
    )
    with pytest.raises(local.LocalPlanningError, match="aggregation"):
        local.admit_local_benchmark_plan(graph=graph, manifest=scenario["manifest"])


def test_native_materialization_rolls_back_on_source_drift(scenario, monkeypatch):
    admission = local.admit_local_benchmark_plan(
        graph=scenario["graph"], manifest=scenario["manifest"]
    )
    before = scenario["intent"].event_watermark()
    original = local._materialize_local_benchmark_plan

    def mutate_after_insert(**kwargs):
        result = original(**kwargs)
        (scenario["repository"] / "answer.py").write_text("def answer(): return 5\n")
        return result

    monkeypatch.setattr(local, "_materialize_local_benchmark_plan", mutate_after_insert)
    with pytest.raises(local.LocalPlanningError, match="source changed"):
        local.materialize_local_benchmark_plan(admission=admission, intent=scenario["intent"])
    assert scenario["intent"].get_task(scenario["task_cid"]) is None
    assert scenario["intent"].event_watermark() == before


def test_untracked_validation_input_is_rejected(scenario):
    (scenario["repository"] / "shadow.py").write_text("pass\n")
    with pytest.raises(local.LocalPlanningError, match="untracked"):
        local.admit_local_benchmark_plan(graph=scenario["graph"], manifest=scenario["manifest"])
