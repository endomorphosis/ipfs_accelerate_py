"""Closed edits and real owner-signed Git materialization; no proof claims from fixtures."""
from copy import deepcopy
import base64
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from benchmarks.agent_supervisor.container_coding.local_planning_qualification import prepare_local_task
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime import repository_finite_handoff as author
from ipfs_accelerate_py.agent_supervisor.runtime import repository_finite_runner as worker
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import IntegerOffsetContract

SOURCE = b"def increment(n: int) -> int:\n    return n + 1\n"
AFTER = b"def increment(n: int) -> int:\n    return n + 2\n"


@pytest.mark.parametrize("source,offset,fragment", [
    (SOURCE, 2, b"n + 2"),
    (b"# retained\ndef increment(n: int) -> int:\n\treturn (n - 9) # comment\n", -3, b"(n - 3) # comment"),
    (b"def increment(n: int) -> int:\n    return n\n", 4, b"n + 4"),
    (SOURCE, 0, b"return n\n"),
])
def test_only_closed_return_expression_changes(source, offset, fragment):
    result = author.propose_offset_source(source, IntegerOffsetContract("calc.py", "increment", "n", offset))
    assert fragment in result
    assert result.split(b"return")[0] == source.split(b"return")[0]


@pytest.mark.parametrize("source", [SOURCE + b"print('effect')\n",
    b"def increment(n: float) -> int:\n    return n + 1\n",
    b"def increment(n: int) -> int:\n    return another(n)\n"])
def test_unknown_source_semantics_cannot_use_offset_operator(source):
    with pytest.raises(ValueError):
        author.propose_offset_source(source, IntegerOffsetContract("calc.py", "increment", "n", 2))


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL).decode().strip()


@pytest.fixture
def signed(tmp_path):
    root = tmp_path / "repository"; root.mkdir()
    (root / "calc.py").write_bytes(SOURCE)
    for args in [("init", "-q"), ("config", "user.name", "Finite worker fixture"),
                 ("config", "user.email", "finite@example.invalid"), ("add", "."), ("commit", "-qm", "fixture")]:
        git(root, *args)
    (root / ".git/info/exclude").write_text(".runtime/\n")
    with IntentRepository(tmp_path / "intent.duckdb") as intent:
        declared = prepare_local_task(repository=root, state=tmp_path / "policy", intent=intent,
            scope_paths=["calc.py"], output_path="calc.py", objective="independent materialization fixture",
            validation_argv=["python3", "-B", "-c", "from calc import increment; assert increment(0)==2"])
        verified = local.verify_local_benchmark_admission(declared["admission"])
        payload = dict(schema=author.SCHEMA, repository=str(root), baseline_commit=git(root, "rev-parse", "HEAD"),
            task_cid=declared["task_cid"], task_id=declared["task_id"], task_revision=1,
            manifest_cid=verified["receipt"]["manifest_cid"], instruction_sha256="1" * 64,
            instruction_path=None,
            permitted_output=dict(path="calc.py", effect="modify", media_type="text/x-python"),
            edit=dict(path="calc.py", before_sha256=author._sha(SOURCE), after_sha256=author._sha(AFTER),
                      after_bytes_base64=base64.b64encode(AFTER).decode()),
            evidence=dict(query_cid="test:query", source_head={"scope": "authored worker-only control"},
                preview_cid="test:preview", selected_operation=dict(requirement_id="finite-integer-offset-goal",
                    task_id="finite:offset", producer_id="producer:offset", path="calc.py", function_name="increment",
                    parameter="n", review_ref="test:explicit-control", operation="update"),
                complete_requirements=["finite-integer-offset-goal", "finite-integer-type-goal"],
                initial_observation_cid="test:initial", candidate_observation_cid="test:candidate",
                input_sha256="2" * 64, scope=author.SCOPE, inputs=[-1, 0, 1]), scope=author.SCOPE, **author.FALSE)
        payload["artifact_cid"] = content_identity(payload)
        envelope = local._signed(payload, verified["manifest"])
        artifact, digest = author._public_artifact(root, envelope)
        workspace = tmp_path / "allocated"
        git(root, "worktree", "add", "--detach", str(workspace), "HEAD")
        options = dict(artifact=artifact, expected_sha256=digest, task_cid=declared["task_cid"],
            owner_did=envelope["binding"]["identity"], profile_id=envelope["binding"]["profile_id"],
            prompt=json.dumps({"objective_id": declared["task_id"]}), workspace=workspace)
        yield dict(root=root, intent=intent, declared=declared, verified=verified,
                   payload=payload, envelope=envelope, options=options)


def test_signed_worker_changes_only_allocated_source(signed):
    result = worker.materialize_finite_candidate(**signed["options"])
    assert result["status"] == "candidate_materialized"
    assert (signed["options"]["workspace"] / "calc.py").read_bytes() == AFTER
    assert (signed["root"] / "calc.py").read_bytes() == SOURCE
    assert signed["intent"].get_task(signed["declared"]["task_cid"])["status"] == "ready"
    assert all(result[field] is False for field in author.FALSE)


@pytest.mark.parametrize("change", ["wrong_owner", "wrong_profile", "wrong_task", "wrong_digest", "bad_prompt",
    "canonical_workspace", "writable_artifact", "source_drift", "canonical_drift", "symlink_source", "writable_parent"])
def test_worker_rejects_mutated_bindings(signed, change):
    options = dict(signed["options"])
    if change == "wrong_owner": options["owner_did"] += "foreign"
    if change == "wrong_profile": options["profile_id"] += "foreign"
    if change == "wrong_task": options["task_cid"] = "foreign"
    if change == "wrong_digest": options["expected_sha256"] = "f" * 64
    if change == "bad_prompt": options["prompt"] = '{"objective_id":"foreign"}'
    if change == "canonical_workspace": options["workspace"] = signed["root"]
    if change == "writable_artifact": options["artifact"].chmod(0o644)
    if change == "source_drift": (options["workspace"] / "calc.py").write_bytes(SOURCE + b"# drift\n")
    if change == "canonical_drift": (signed["root"] / "calc.py").write_bytes(SOURCE + b"# drift\n")
    if change == "symlink_source":
        path = options["workspace"] / "calc.py"; path.unlink(); path.symlink_to(signed["root"] / "calc.py")
    if change == "writable_parent": options["artifact"].parent.chmod(0o777)
    with pytest.raises((ValueError, OSError, subprocess.CalledProcessError)):
        worker.materialize_finite_candidate(**options)
    assert signed["intent"].get_task(signed["declared"]["task_cid"])["status"] == "ready"


@pytest.mark.parametrize("change", ["authority", "empty_domain", "wrong_clause", "wrong_path", "extra_field", "bad_signature"])
def test_rehashed_signed_payload_still_requires_closed_finite_scope(signed, change):
    payload = deepcopy(signed["payload"])
    if change == "authority": payload["proof_authority"] = True
    if change == "empty_domain": payload["evidence"]["inputs"] = []
    if change == "wrong_clause": payload["evidence"]["complete_requirements"] = []
    if change == "wrong_path": payload["evidence"]["selected_operation"]["path"] = "foreign.py"
    if change == "extra_field": payload["proof_receipt_id"] = "invented"
    payload["artifact_cid"] = content_identity({key: value for key, value in payload.items() if key != "artifact_cid"})
    envelope = local._signed(payload, signed["verified"]["manifest"])
    if change == "bad_signature": envelope["payload"]["instruction_sha256"] = "e" * 64
    artifact, digest = author._public_artifact(signed["root"], envelope)
    with pytest.raises((ValueError, RuntimeError)):
        worker.materialize_finite_candidate(**{**signed["options"], "artifact": artifact, "expected_sha256": digest})
    assert (signed["options"]["workspace"] / "calc.py").read_bytes() == SOURCE
