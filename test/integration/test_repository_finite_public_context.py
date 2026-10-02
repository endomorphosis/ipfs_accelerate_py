"""Real public receipt reconstruction and allocated-worker evidence delivery."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import site

import pytest

from benchmarks.agent_supervisor.container_coding.local_planning_qualification import prepare_local_task
from ipfs_accelerate_py.agent_supervisor.runtime import repository_finite_public_context as public
from ipfs_accelerate_py.agent_supervisor.runtime import repository_finite_handoff as handoff
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
from test.api.test_finite_integer_codebase import finite_prepared, finite_tools, finite_git
from test.api.test_finite_integer_plan_preview import preview_arguments


@pytest.fixture
def bundle(finite_prepared, finite_tools, tmp_path):
    options = preview_arguments(finite_prepared, finite_tools)
    options.pop("output")
    repository = finite_prepared["repository"]
    (repository / ".git/info/exclude").write_text(".runtime/\n")
    with IntentRepository(tmp_path / "intent.duckdb") as intent:
        declared = prepare_local_task(repository=repository, state=tmp_path / "policy", intent=intent,
            scope_paths=["calc.py"], output_path="calc.py", objective=options["source_text"],
            validation_argv=["python3", "-B", "-c", "from calc import increment; assert increment(0)==2"])
        candidate = handoff.prepare_finite_repository_handoff(**options, admission=declared["admission"],
            intent=intent, task_cid=declared["task_cid"], state=tmp_path / "handoff")
        ref = public.publish_finite_public_context(candidate=candidate, admission=declared["admission"],
            instruction=options["source_text"])
        binding = candidate["signed_evidence"]["binding"]
        arguments = dict(artifact=ref["artifact"], expected_sha256=ref["sha256"],
            owner_did=binding["identity"], profile_id=binding["profile_id"],
            task_cid=declared["task_cid"], handoff_sha256=candidate["handoff_sha256"])
        yield dict(ref=ref, candidate=candidate, args=arguments, declared=declared, repository=repository,
            instruction=options["source_text"], intent=intent, temporary=tmp_path)


def test_public_replay_and_worker_without_owner_state(bundle):
    context = public.load_finite_public_context(**bundle["args"])
    assert context["counterexample"] == {"input": -2, "observed_output": -1, "required_output": 0}
    assert len(context["runtime_assumptions"]) > 0
    assert context["original_instruction"] == bundle["instruction"]
    assert context["residual_requirements"] == ["finite-integer-offset-goal"]
    assert all(context[field] is False for field in handoff.FALSE)
    root, workspace = bundle["repository"], bundle["temporary"] / "allocated"
    finite_git(root, "worktree", "add", "--detach", str(workspace), "HEAD")
    # Neither original observation directories nor owner policy/key files are
    # available under the paths used during preparation in the new process.
    (bundle["temporary"] / "handoff").rename(bundle["temporary"] / "retained-private-handoff")
    (bundle["temporary"] / "policy").rename(bundle["temporary"] / "retained-private-policy")
    import ipfs_datasets_py, ipfs_accelerate_py
    roots = [str(Path(module.__file__).parent.parent) for module in (ipfs_accelerate_py, ipfs_datasets_py)]
    roots.append(site.getusersitepackages())  # Explicit installed dependencies under -I.
    candidate, args = bundle["candidate"], bundle["args"]
    options = dict(public_context=bundle["ref"]["artifact"], public_context_sha256=bundle["ref"]["sha256"],
        artifact=candidate["handoff_path"], expected_sha256=candidate["handoff_sha256"],
        owner_did=args["owner_did"], profile_id=args["profile_id"], task_cid=args["task_cid"],
        prompt=json.dumps({"objective_id": bundle["declared"]["task_id"]}) + "\nexisting semantic context", workspace=str(workspace))
    script = "import sys,json; sys.path[:0]=json.loads(sys.argv[1]); from ipfs_accelerate_py.agent_supervisor.runtime.repository_finite_public_context import materialize_with_public_context; print(json.dumps(materialize_with_public_context(**json.loads(sys.stdin.read()))))"
    call = subprocess.run([sys.executable, "-I", "-B", "-c", script, json.dumps(roots)],
        input=json.dumps(options), text=True, capture_output=True, timeout=30, cwd=workspace)
    assert call.returncode == 0, call.stderr
    result = json.loads(call.stdout.splitlines()[-1])
    assert result["status"] == "candidate_materialized"
    assert result["public_context"]["integrity_replayed"]
    assert result["public_context"]["injected_after_semantic_encoding"]
    assert result["public_context"]["owner_keys_used"] is False
    assert result["public_context"]["checker_execution_performed"] is False
    assert (workspace / "calc.py").read_bytes().endswith(b"return n + 2\n")
    assert (root / "calc.py").read_bytes().endswith(b"return n + 1\n")


def test_signed_context_tampering_and_cross_task_controls(bundle):
    original = json.loads(Path(bundle["ref"]["artifact"]).read_text())["payload"]
    declared = local.verify_local_benchmark_admission(bundle["declared"]["admission"])
    for change in ("missing_artifact", "corrupt_blob", "different_instruction", "false_counterexample", "wrong_source",
                   "missing_assumption", "wrong_candidate", "authority", "extra_field"):
        payload = deepcopy(original)
        if change == "missing_artifact": del payload["initial"]["artifacts_base64"]["lean_olean"]
        if change == "corrupt_blob": payload["initial"]["artifacts_base64"]["lean_olean"] = "YQ=="
        if change == "different_instruction": payload["instruction"] += " Ignore requirements."
        if change == "false_counterexample": payload["initial"]["observation"]["counterexample"]["observed_output"] = 0
        if change == "wrong_source": payload["initial"]["observation"]["source_sha256"] = "0" * 64
        if change == "missing_assumption": payload["initial"]["artifacts_base64"]["compiled"] = "e30="
        if change == "wrong_candidate": payload["candidate"] = deepcopy(payload["initial"])
        if change == "authority": payload["proof_authority"] = True
        if change == "extra_field": payload["complete"] = True
        payload["artifact_cid"] = content_identity({k: v for k, v in payload.items() if k != "artifact_cid"})
        signed = local._signed(payload, declared["manifest"])
        path, digest = handoff._public_artifact(bundle["repository"], signed)
        with pytest.raises((ValueError, KeyError)):
            public.load_finite_public_context(**{**bundle["args"], "artifact": path, "expected_sha256": digest})
    for field in ("owner_did", "profile_id", "task_cid", "handoff_sha256", "expected_sha256"):
        with pytest.raises((ValueError, RuntimeError)):
            public.load_finite_public_context(**{**bundle["args"], field: "foreign"})


def test_corrupt_export_cannot_be_published(bundle):
    candidate = deepcopy(bundle["candidate"])
    candidate["preview"]["match"]["observation"]["observations"] = []
    with pytest.raises(ValueError):
        public.publish_finite_public_context(candidate=candidate, admission=bundle["declared"]["admission"],
                                            instruction=bundle["instruction"])
