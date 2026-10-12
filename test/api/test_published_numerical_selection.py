"""Successor policy controls with explicitly authored native-owner seams.

Real checkpoint, source capture, publication and STOP qualification lives in
the integration suite. These controls isolate refusal before any model work.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import published_numerical_selection as owner


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def case(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    sources = {"answer.py": "def f(x):\n    return x\n", "instruction.md": "Change the code."}
    for name, text in sources.items():
        (root / name).write_text(text)
    previous = tmp_path / "previous"
    previous.mkdir()
    hashes = {name: sha(text) for name, text in sources.items()}
    receipt = dict(schema=owner.native.SCHEMA, profile=owner.native.PROFILE, repository=str(root),
        output=str(previous), source_hashes=hashes, selected_paths=["answer.py"],
        config_path=str(tmp_path / "config.json"), config_sha256="c" * 64,
        checkpoint_file_sha256="d" * 64, receipt_sha256="e" * 64,
        summary={key: "f" * 64 for key in owner.MODEL_FIELDS}, producers={"authored": "pin"})
    manifest = dict(sources={name: {"sha256": digest} for name, digest in hashes.items()},
        baseline_commit="a" * 40, tasks=[{"outputs": [{"path": "answer.py"}, {"path": "new.py"}]}])
    current_head = [manifest["baseline_commit"]]
    calls, exclusions = [], {str(previous): [".runtime"]}

    def validate(*, repository, expected_receipt, **kwargs):
        assert expected_receipt["source_hashes"] == {
            name: sha((repository / name).read_text()) for name in expected_receipt["source_hashes"]}
        return expected_receipt

    monkeypatch.setattr(owner.native, "validate_numerical_selection_context", validate)
    monkeypatch.setattr(owner.native, "validate_historical_numerical_selection_context", lambda **kw: kw["expected_receipt"])
    monkeypatch.setattr(owner, "_capture_exclusions", lambda r: exclusions[r["output"]])
    monkeypatch.setattr(owner, "_git", lambda *a: current_head[0].encode())

    def prepare(**kw):
        calls.append(kw)
        output = Path(kw["output"])
        output.mkdir()
        exclusions[str(output)] = sorted([".runtime", *kw["excluded_new_output_paths"]])
        value = dict(receipt, output=str(output), source_hashes=kw["source_hashes"],
            receipt_sha256="9" * 64, inference_execution=dict(native_worker_executed=True,
                inference_executed=True, model_loads=1))
        (output / "inference.json").write_text(json.dumps({"report": {"output": {
            "inference": {"rows": [{"authored": True}]},
            "embedding_receipt": {"status_counts": {"embedded": 1}},
        }}}))
        return value

    monkeypatch.setattr(owner.native, "prepare_numerical_selection_context", prepare)
    output_root = tmp_path / "successors"
    output_root.mkdir()
    policy = owner.bind_policy(repository=root, receipt=receipt, manifest=manifest, output_root=output_root)
    binding = dict(task_status="completed", published_commit="b" * 40,
        task_cid="task:1", task_id="one", accepted_source_tree_id="tree:new", completion_receipt_cid="completion:1")
    return SimpleNamespace(root=root, receipt=receipt, manifest=manifest, policy=policy,
        binding=binding, current_head=current_head, calls=calls, exclusions=exclusions, output_root=output_root)


def publish(case):
    (case.root / "answer.py").write_text("def f(x):\n    return x + 1\n")
    (case.root / "new.py").write_text("# declared new output\n")
    case.current_head[0] = case.binding["published_commit"]


def options(case):
    return dict(repository=case.root, predecessor_receipt=case.receipt, policy=case.policy,
        output=case.output_root / "attempt-1", publication_binding=case.binding)


def test_refresh_preserves_scope_and_zeroes_only_current_cached_execution(case, monkeypatch):
    publish(case)
    monkeypatch.setattr(owner.time, "monotonic", lambda: 100.)
    result = owner.prepare_successor(**options(case), deadline_monotonic=100.5)
    assert len(case.calls) == 1
    assert case.calls[0]["timeout_seconds"] == .5
    assert case.calls[0]["program_paths"] == ["answer.py"]
    assert case.calls[0]["excluded_new_output_paths"] == ["new.py"]
    assert set(case.calls[0]["source_hashes"]) == {"answer.py", "instruction.md"}
    assert case.calls[0]["config_path"] == case.receipt["config_path"]
    observation = result["learned_numerical_refresh"]
    fresh = owner.execution_observation(observation, reused=False)
    cached = owner.execution_observation(observation, reused=True)
    assert fresh["inference_executed"] is True and fresh["inferred_units"] == 1
    assert cached["inference_executed"] is False and cached["head_inference_calls"] == 0
    assert cached["embedding_production_calls"] == cached["embedded_texts"] == cached["model_loads"] == 0
    assert observation["inference_execution"]["inference_executed"] is True
    assert observation["requires_independent_manifest"] is True
    assert observation["proof_authority"] is observation["scope_expanded"] is False
    assert observation["source_obligations_discharged"] == 0
    # Validate retained lineage; never regenerate it on a cache hit.
    monkeypatch.setattr(owner.native, "prepare_numerical_selection_context", lambda **kw: pytest.fail("replayed inference"))
    owner.validate_successor(repository=case.root, predecessor_receipt=case.receipt, policy=case.policy,
        receipt=result["numerical_selection_384"], observation=observation,
        publication_binding=case.binding, deadline_monotonic=100.5)


@pytest.mark.parametrize("field,value", [
    ("selected_paths", ["instruction.md"]), ("source_hashes", {"answer.py": "1" * 64}),
    ("checkpoint_file_sha256", "1" * 64), ("config_sha256", "1" * 64),
    ("head_sha256", "1" * 64), ("projection_sha256", "1" * 64),
    ("producers", {"authored": "other"}), ("proof_authority", True),
    ("source_obligations_discharged", False), ("preparation_timeout_seconds", 3600),
    ("source_capture_exclusions", [".runtime", "answer.py"]),
    ("successor_helper_sha256", "1" * 64),
])
def test_mutated_policy_refuses_before_inference(case, field, value):
    publish(case)
    case.policy[field] = value
    with pytest.raises(ValueError):
        owner.prepare_successor(**options(case))
    assert not case.calls


@pytest.mark.parametrize("change", ["missing", "extra", "digest", "output"])
def test_signed_manifest_join_cannot_change_source_or_output_population(case, change):
    manifest = deepcopy(case.manifest)
    if change == "missing":
        del manifest["sources"]["instruction.md"]
    elif change == "extra":
        manifest["sources"]["foreign.py"] = {"sha256": "1" * 64}
    elif change == "digest":
        manifest["sources"]["answer.py"]["sha256"] = "1" * 64
    else:
        manifest["tasks"][0]["outputs"].append({"path": "foreign.py"})
    # Merely copying the new manifest identity must not change inherited scope.
    case.policy["manifest_cid"] = owner.content_identity(manifest)
    with pytest.raises(ValueError):
        owner.validate_manifest(policy=case.policy, manifest=manifest)
    assert not case.calls


@pytest.mark.parametrize("change", ["not_completed", "baseline", "wrong_head", "no_completion", "outside_output", "expired"])
def test_unpublished_or_unadmitted_attempt_never_starts_model(case, change, monkeypatch):
    publish(case)
    args = options(case)
    if change == "not_completed":
        case.binding["task_status"] = "in_progress"
    elif change == "baseline":
        case.binding["published_commit"] = case.manifest["baseline_commit"]
    elif change == "wrong_head":
        case.current_head[0] = "c" * 40
    elif change == "no_completion":
        del case.binding["completion_receipt_cid"]
    elif change == "outside_output":
        args["output"] = case.output_root.parent / "foreign"
    else:
        monkeypatch.setattr(owner.time, "monotonic", lambda: 100.)
        args["deadline_monotonic"] = 99.
    with pytest.raises((ValueError, TimeoutError)):
        owner.prepare_successor(**args)
    assert not case.calls


@pytest.mark.parametrize("mode", ["unsupported", "token_deferred"])
def test_abstention_does_not_claim_head_inference(case, mode):
    receipt = dict(case.receipt, inference_execution=dict(native_worker_executed=True,
        inference_executed=False, model_loads=int(mode == "token_deferred")))
    (Path(receipt["output"]) / "inference.json").write_text(json.dumps({"report": {"output": {
        "inference": None, "embedding_receipt": {"status_counts": {"embedded": 0}} if mode == "token_deferred" else None,
    }}}))
    execution = owner._execution(receipt)
    assert execution["head_inference_calls"] == execution["inferred_units"] == execution["embedded_texts"] == 0
    assert execution["inference_executed"] is False
    assert execution["embedding_production_calls"] == int(mode == "token_deferred")
