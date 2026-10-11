"""Numerical successor joins actual owner publication; no provider or model calls.

The mixed case deliberately authors only the Source384 inference/validation
seam. Numerical grammar, lakes, bundle, owner and publication checks are real.
"""
import hashlib
import json
from pathlib import Path

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from test.api.test_local_completion_bridge import complete, native_published_transition, typed_claim
from test.api.semantic_state.test_published_task_context import initial_bundle
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime import numerical_formalization_context as numerical
from ipfs_accelerate_py.agent_supervisor.runtime import published_numerical, published_task_context as published
from ipfs_accelerate_py.agent_supervisor.runtime import source384_repository_context as source384
from ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge import run_owner_local_task_validations
from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import (
    load_task_context_selection, write_task_context_bundle,
)


def raw(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def authored_source384(monkeypatch, root, output, hashes):
    def validate(*, repository, expected_receipt, **kwargs):
        assert expected_receipt["source_hashes"] == {
            name: hashlib.sha256((repository / name).read_bytes()).hexdigest()
            for name in expected_receipt["source_hashes"]}
        return expected_receipt
    def historical(*, repository, expected_receipt):
        assert (Path(expected_receipt["output"]) / "receipt.json").read_bytes() == raw(expected_receipt)
        return expected_receipt
    def prepare(*, repository, source_hashes, output, predecessor_receipt,
                timeout_seconds, excluded_new_output_paths):
        assert 0 < timeout_seconds <= 180
        value = {**predecessor_receipt, "schema": source384.SUCCESSOR_SCHEMA,
            "output": str(output), "source_hashes": source_hashes,
            "source_head": local._git(repository, "rev-parse", "HEAD").strip(),
            "successor_lineage": {
                "predecessor_receipt_sha256": hashlib.sha256(raw(predecessor_receipt)).hexdigest(),
                "excluded_new_output_paths": excluded_new_output_paths},
            "inference_execution": {"native_worker_executed": False, "inference_executed": False, "model_loads": 0}}
        output.mkdir()
        (output / "receipt.json").write_bytes(raw(value))
        return value
    monkeypatch.setattr(source384, "validate_source384_context", validate)
    monkeypatch.setattr(source384, "validate_historical_source384_selection", historical)
    monkeypatch.setattr(source384, "prepare_source384_successor_context", prepare)
    value = {"schema": source384.SCHEMA, "repository": str(root), "output": str(output),
        "source_hashes": hashes, "completion_authority": False,
        "summary": {"schema": "authored-source384-summary@1", "candidate_samples": [], "proof_authority": False},
        "config_sha256": "a" * 64, "checkpoint_sha256": "b" * 64, "inference_sha256": "c" * 64,
        "version_id": "authored-no-inference"}
    output.mkdir()
    (output / "receipt.json").write_bytes(raw(value))
    return value


@pytest.mark.parametrize("mode", ["combined", "dirty_during", "dirty_reload"])
def test_numerical_successor_preserves_mixed_scope_and_owner_currentness(scenario, tmp_path, monkeypatch, mode):
    root = scenario["repository"]
    baseline = {name: row["sha256"] for name, row in scenario["manifest"]["payload"]["sources"].items()}
    previous = numerical.prepare_numerical_context(repository=root,
        source_hashes={"answer.py": baseline["answer.py"]}, output=tmp_path / "previous-numerical", profile=numerical.PROFILE)
    output_root = tmp_path / "numerical-successors"
    output_root.mkdir(mode=0o700)
    admission = local.admit_local_benchmark_plan(graph=scenario["graph"], manifest=scenario["manifest"])
    policy = published_numerical.bind_policy(repository=root, receipt=previous,
        manifest=admission["manifest"]["payload"], output_root=output_root)
    selected_receipts = {"numerical_formalization": previous}
    if mode == "combined":
        selected_receipts["source384_context"] = authored_source384(
            monkeypatch, root, tmp_path / "previous-source384", baseline)
    with typed_claim(scenario, tmp_path) as (owner, _daemon, attempt):
        prior = initial_bundle(scenario, owner)
        task_context = json.loads((root / prior["artifact"]).read_bytes())["tasks"][0]
        bundle = write_task_context_bundle(repository=root, prepared=[{
            "schema": "supervisor-task-context-preparation@1", **task_context, **selected_receipts}],
            output=root / ".runtime/formalization.json")
        original = (root / bundle["artifact"]).read_bytes()
        task = owner.source.get_task(attempt.task_cid)
        transition = native_published_transition(scenario, tmp_path, owner, attempt)
        passed = run_owner_local_task_validations(server=owner.server, task_cid=task.task_cid,
            attempt_id=attempt.attempt_id, expected_revision=task.revision, source_transition=transition)
        assert passed["passed"] is True
        complete(owner, attempt, passed["results"][0]["evidence_digest"])
        completed = owner.source.get_task(task.task_cid)
        arguments = dict(server=owner.server, admission=admission, predecessor_bundle=bundle,
            task_cid=task.task_cid, output=root / ".runtime/successor", numerical_policy=policy,
            numerical_output=output_root / "attempt-1")
        if mode == "combined":
            arguments["source384_output"] = tmp_path / "next-source384"
        # Even a copied correct manifest CID cannot bless another baseline,
        # path population, original source digest, or declared output scope.
        altered_policies = [
            {**policy, "manifest_cid": "foreign"},
            {**policy, "baseline_commit": "a" * 40},
            {**policy, "source_hashes": {"outside.py": baseline["answer.py"]}},
            {**policy, "source_hashes": {"answer.py": "a" * 64}},
            {**policy, "excluded_new_output_paths": ["invented.py"]},
        ]
        for changed_policy in altered_policies:
            with pytest.raises(ValueError, match="admitted"):
                published.refresh_published_task_context(**{**arguments, "numerical_policy": changed_policy})
        assert not arguments["output"].exists() and not arguments["numerical_output"].exists()
        accepted_head = local._git(root, "rev-parse", "HEAD")
        if mode == "dirty_during":
            native_prepare = published_numerical.prepare_successor
            def mutate_after_prepare(**kwargs):
                result = native_prepare(**kwargs)
                (root / "answer.py").write_text("def answer(): return 9\n")
                return result
            monkeypatch.setattr(published_numerical, "prepare_successor", mutate_after_prepare)
            with pytest.raises(ValueError):
                published.refresh_published_task_context(**arguments)
            assert local._git(root, "rev-parse", "HEAD") == accepted_head
            assert not arguments["output"].exists()
            assert (arguments["numerical_output"] / "receipt.json").exists()
        else:
            result = published.refresh_published_task_context(**arguments)
            assert result["numerical_refresh"]["source_obligations_discharged"] == 0
            assert result["numerical_refresh"]["proof_authority"] is False
            assert result["numerical_formalization"]["source_hashes"].keys() == {"answer.py"}
            selected = load_task_context_selection(repository=root,
                artifact=result["context_bundle"]["artifact"], expected_sha256=result["context_bundle"]["sha256"],
                task_cid=task.task_cid, task_id=task.task_alias)
            assert selected["numerical_formalization"] == result["numerical_formalization"]
            if mode == "combined":
                assert result["source384_context"]["source_hashes"].keys() == baseline.keys()
                assert selected["source384_context"] == result["source384_context"]
                assert {row["kind"] for row in selected["worker_evidence"]["contexts"]} == {
                    "source384-candidate-context", "numerical-formalization-context"}
            monkeypatch.setattr(numerical, "prepare_numerical_context", lambda **kwargs: pytest.fail("reload regenerated"))
            reload_args = dict(server=owner.server, admission=admission,
                artifact=result["refresh_artifact"], expected_sha256=result["refresh_sha256"], numerical_policy=policy)
            assert published.load_published_task_context(**reload_args)["context_bundle"] == result["context_bundle"]
            # Rewrite all public hashes consistently: exact owner admission
            # still rejects a drifted historical baseline or exclusion list.
            for i, changed_policy in enumerate((altered_policies[1], altered_policies[-1])):
                changed = {key: value for key, value in result.items()
                    if key not in {"refresh_artifact", "refresh_sha256"}}
                changed["numerical_refresh_policy"] = changed_policy
                changed["numerical_refresh"] = {**result["numerical_refresh"],
                    "policy_cid": local.content_identity(changed_policy)}
                changed_path = root / f".runtime/changed-policy-{i}.json"
                changed_path.write_bytes(raw(changed))
                with pytest.raises(ValueError, match="admitted"):
                    published.load_published_task_context(**{**reload_args,
                        "artifact": changed_path.relative_to(root).as_posix(),
                        "expected_sha256": hashlib.sha256(changed_path.read_bytes()).hexdigest(),
                        "numerical_policy": changed_policy})
            if mode == "dirty_reload":
                (root / "answer.py").write_text("def answer(): return 9\n")
                assert local._git(root, "rev-parse", "HEAD") == accepted_head
                with pytest.raises(ValueError):
                    published.load_published_task_context(**reload_args)
        assert owner.source.get_task(task.task_cid) == completed
        assert (root / bundle["artifact"]).read_bytes() == original
