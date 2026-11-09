"""Signed owner triplets and complete launch custody, without providers."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding.container_worker_deployment import semantic_decision_arguments
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_decision_owner as owner
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_decision_worker as worker
from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import write_task_context_bundle
from test.api.test_semantic_retrieval_query_reference import build_retrieval_native_fixture


def _arguments():
    return ["--model", "pinned", "--semantic-repository", "/source",
            "--semantic-decision-nomination-artifact", "/owner/nomination.json",
            "--semantic-decision-nomination-sha256", "a" * 64,
            "--semantic-decision-task-id", "task"]


def _retrieval():
    return ["--semantic-decision-context-bundle-artifact", ".runtime/task-context.json",
            "--semantic-decision-context-bundle-sha256", "b" * 64,
            "--semantic-decision-task-cid", "task:admitted"]


def test_signed_retrieval_flags_are_consumed_by_owner_and_not_forwarded_to_model():
    values, remaining = owner._literal_arguments(_arguments() + _retrieval())
    assert set(values) == owner._OWNER_FLAGS | owner._RETRIEVAL_FLAGS
    assert remaining == ["--model", "pinned", "--semantic-repository", "/source"]


@pytest.mark.parametrize("flag", sorted(owner._RETRIEVAL_FLAGS))
@pytest.mark.parametrize("mutation", ["partial", "duplicate", "equals", "no_value"])
def test_signed_retrieval_triplet_refuses_partial_or_ambiguous_bindings(flag, mutation):
    parts = _retrieval()
    index = parts.index(flag)
    if mutation == "partial":
        del parts[index:index + 2]
    elif mutation == "duplicate":
        parts += parts[index:index + 2]
    elif mutation == "equals":
        parts[index:index + 2] = [flag + "=" + parts[index + 1]]
    else:
        parts = parts[:index + 1]
    with pytest.raises(ValueError):
        owner._literal_arguments(_arguments() + parts)


def test_signed_retrieval_triplet_is_consumed_with_explicit_typed_nomination_before_dispatch():
    values, remaining = owner._literal_arguments(_arguments() + _retrieval()
        + ["--semantic-decision-typed-nomination", "enabled"])
    assert values["--semantic-decision-typed-nomination"] == "enabled"
    assert set(owner._RETRIEVAL_FLAGS) <= set(values)
    assert not set(owner._RETRIEVAL_FLAGS).intersection(remaining)


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_query_reference_configuration_requires_literal_boolean(value):
    with pytest.raises(ValueError):
        semantic_decision_arguments(semantic_retrieval_query_reference=value)


@pytest.mark.parametrize("settings", [
    {},
    {"semantic_decision_policy_artifact": "/owner/policy.json", "semantic_decision_policy_sha256": "a" * 64},
 ])
def test_query_reference_configuration_requires_explicit_owner_nomination(settings):
    with pytest.raises(ValueError):
        semantic_decision_arguments(semantic_retrieval_query_reference=True, **settings)


def test_query_reference_switch_does_not_invent_parent_flags_or_change_legacy_encoding():
    settings = {"semantic_decision_nomination_artifact": "/owner/nomination.json",
                "semantic_decision_nomination_sha256": "a" * 64,
                "disable_semantic_typed_nomination": True, "owner_binding": True}
    legacy = semantic_decision_arguments(**settings)
    opted_in = semantic_decision_arguments(**settings, semantic_retrieval_query_reference=True)
    assert opted_in == legacy
    assert not any(part in owner._RETRIEVAL_FLAGS for part in opted_in)
    assert semantic_decision_arguments() == []


@pytest.fixture
def native_launch(tmp_path):
    """Real producers; the bundle's local task CID is authored fixture authority."""
    native = build_retrieval_native_fixture(tmp_path)
    metadata = {**native.metadata,
        "Semantic context artifact": native.semantic_bindings.artifact,
        "Semantic context sha256": native.semantic_bindings.expected_sha256}
    bundle = write_task_context_bundle(repository=native.repository,
        prepared=[{"schema": "supervisor-task-context-preparation@1",
                   "task_cid": "fixture:admitted-task", "task_id": native.binding.task_id,
                   "metadata": metadata}], output=native.repository / ".runtime/context-bundle.json")
    nomination = tmp_path / "nomination.json"
    nomination.write_bytes(json.dumps(owner.build_decision_nomination(native.policy),
        sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())
    arguments = ["--model", "pinned", "--semantic-repository", str(native.repository),
        "--semantic-decision-nomination-artifact", str(nomination),
        "--semantic-decision-nomination-sha256", hashlib.sha256(nomination.read_bytes()).hexdigest(),
        "--semantic-decision-task-id", native.binding.task_id,
        "--semantic-decision-typed-nomination", "disabled",
        "--semantic-decision-context-bundle-artifact", bundle["artifact"],
        "--semantic-decision-context-bundle-sha256", bundle["sha256"],
        "--semantic-decision-task-cid", "fixture:admitted-task"]
    return native, bundle, arguments


def _dispatch(native_launch):
    native, _, arguments = native_launch
    return owner.prepare_decision_owner_dispatch(arguments=arguments,
        prompt_bytes=native.native_prompt.encode(), workspace=native.workspace)


def test_owner_authenticates_bundle_and_publishes_complete_retrieval_launch(native_launch):
    native, _, arguments = native_launch
    dispatch = _dispatch(native_launch)
    assert dispatch.prompt_bytes == native.native_prompt.encode()
    assert not set(owner._RETRIEVAL_FLAGS).intersection(dispatch.worker_arguments)
    path = Path(dispatch.receipt["launch_policy_artifact"])
    descriptor = json.loads(path.read_bytes())
    assert descriptor["schema"] == worker.RETRIEVAL_SCHEMA
    assert descriptor["retrieval_query_binding"] == native.binding.to_dict()
    assert descriptor["policy"] == native.policy.to_dict()
    assert path.stat().st_mode & 0o777 == 0o444
    settings = dict(artifact=path, expected_sha256=dispatch.receipt["launch_policy_sha256"],
        native_prompt=native.native_prompt, repository=native.repository, workspace=native.workspace)
    assert worker.load_decision_launch_configuration(**settings) == (native.policy, native.binding)
    with pytest.raises(ValueError, match="complete launch configuration"):
        worker.load_decision_launch_policy(**settings)
    assert arguments[-6:] == ["--semantic-decision-context-bundle-artifact", native_launch[1]["artifact"],
        "--semantic-decision-context-bundle-sha256", native_launch[1]["sha256"],
        "--semantic-decision-task-cid", "fixture:admitted-task"]


@pytest.mark.parametrize("mutation", ["bundle_digest", "foreign_task", "raw_parent", "legacy_policy"])
def test_owner_refuses_wrong_independent_bundle_task_or_parent_before_publication(native_launch, mutation):
    native, bundle, arguments = native_launch
    if mutation == "bundle_digest":
        (native.repository / bundle["artifact"]).write_bytes(b"{}")
    elif mutation == "foreign_task":
        arguments[-1] = "fixture:foreign-task"
    elif mutation == "raw_parent":
        (native.repository / native.binding.artifact).write_bytes(b"{}")
    else:
        nomination = Path(arguments[arguments.index("--semantic-decision-nomination-artifact") + 1])
        nomination.write_bytes(json.dumps(owner.build_decision_nomination(
            replace(native.policy, task_text_source_path=None)), sort_keys=True).encode())
        arguments[arguments.index("--semantic-decision-nomination-sha256") + 1] = hashlib.sha256(nomination.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        _dispatch(native_launch)
    assert not (native.repository / ".runtime/semantic-decision-policies").exists()


def test_v3_worker_forwards_independent_retrieval_binding_with_literal_native_prompt(native_launch, monkeypatch):
    native, _, _ = native_launch
    dispatch = _dispatch(native_launch)
    monkeypatch.chdir(native.workspace)
    arguments, receipt = worker._prepare_launch(prompt=native.native_prompt, model="pinned",
        semantic_repository=native.repository,
        semantic_decision_policy_artifact=Path(dispatch.receipt["launch_policy_artifact"]),
        semantic_decision_policy_sha256=dispatch.receipt["launch_policy_sha256"])
    assert arguments["retrieval_query_binding"] == native.binding
    assert arguments["native_prompt"] == native.native_prompt
    assert receipt["retrieval_query_binding"] == native.binding.to_dict()


@pytest.mark.parametrize("mutation", ["missing", "extra", "task", "digest", "policy"])
def test_v3_launch_refuses_incomplete_or_repaired_unbound_parent(native_launch, mutation):
    native, _, _ = native_launch
    dispatch = _dispatch(native_launch)
    descriptor = json.loads(Path(dispatch.receipt["launch_policy_artifact"]).read_bytes())
    if mutation == "missing":
        del descriptor["retrieval_query_binding"]
    elif mutation == "extra":
        descriptor["retrieval_query_binding"]["discover_current"] = True
    elif mutation == "task":
        descriptor["retrieval_query_binding"]["task_id"] = "foreign"
    elif mutation == "digest":
        descriptor["retrieval_query_binding"]["expected_sha256"] = "0" * 64
    else:
        descriptor["policy"].pop("task_text_source_path")
    path = native.repository.parent / "repaired-policy.json"
    path.write_bytes(json.dumps(descriptor, sort_keys=True).encode())
    with pytest.raises(ValueError):
        worker.load_decision_launch_configuration(artifact=path,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), native_prompt=native.native_prompt,
            repository=native.repository, workspace=native.workspace)


@pytest.mark.parametrize("disable_typed", [False, True])
def test_reference_configuration_preserves_explicit_typed_choice(disable_typed):
    settings = {"semantic_decision_nomination_artifact": "/owner/nomination.json",
        "semantic_decision_nomination_sha256": "a" * 64,
        "disable_semantic_typed_nomination": disable_typed, "owner_binding": True}
    argv = semantic_decision_arguments(**settings, semantic_retrieval_query_reference=True)
    assert argv[argv.index("--semantic-decision-typed-nomination") + 1] == (
        "disabled" if disable_typed else "enabled")
    assert not set(owner._RETRIEVAL_FLAGS).intersection(argv)


@pytest.fixture(params=("inline", "referenced"))
def typed_retrieval_launch(native_launch, monkeypatch, request):
    """Actual public native producers and owner; no native verifier is replaced."""
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_packet import SemanticDecisionPolicy
    monkeypatch.setenv("IPFS_ACCELERATE_TYPED_VERIFICATION_CACHE", "0")
    native, bundle, arguments = native_launch
    arguments[arguments.index("--semantic-decision-typed-nomination") + 1] = "enabled"
    selected = native.semantic_parent["worker_projection"]["selected_symbols"]
    seed = next(row["stable_symbol_id"] for row in selected if row["qualified_name"].split(".")[-1] == "another")
    capsules = {row["stable_symbol_id"]: row["capsule_cid"] for row in native.semantic_parent["capsules"]}
    dependency = next(row["stable_symbol_id"] for row in selected
        if row["qualified_name"].split(".")[-1] == "calculate")
    native.expected_dependency_capsules = (capsules[dependency],)
    required = (capsules[seed],)
    native.origin_policy = SemanticDecisionPolicy(required_capsule_ids=required,
        optional_capsule_ids=tuple(row["capsule_cid"] for row in native.semantic_parent["capsules"]
                                   if row["capsule_cid"] not in required),
        task_text_source_path=native.instruction_source_path)
    nomination = Path(arguments[arguments.index("--semantic-decision-nomination-artifact") + 1])
    raw = (json.dumps(owner.build_decision_nomination(native.origin_policy),
                      sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()
    nomination.write_bytes(raw)
    nomination.chmod(0o444)
    arguments[arguments.index("--semantic-decision-nomination-sha256") + 1] = hashlib.sha256(raw).hexdigest()
    native.origin_path, native.origin_raw = nomination, raw
    if request.param == "referenced":
        from ipfs_accelerate_py.agent_supervisor.runtime import code_retrieval_context as retrieval
        payload = json.loads(native.retrieval_artifact.read_bytes())
        limit = len(retrieval._json(payload).encode()) - len(retrieval._json(payload["snapshot"]).encode()) // 2
        monkeypatch.setattr(retrieval, "MAX_ARTIFACT_BYTES", limit)
        native.retrieval_artifact.unlink()
        result = retrieval.prepare_code_retrieval_context(repository=native.repository,
            task_id=native.binding.task_id, query_text=native.query, snapshot=native.snapshot,
            result=native.result, output=native.retrieval_artifact)
        native.binding = replace(native.binding, expected_sha256=result["sha256"])
        payload = json.loads(native.retrieval_artifact.read_bytes())
        assert payload["schema"] == retrieval.REFERENCED_SCHEMA
        native.snapshot_artifact = native.repository / payload["snapshot_ref"]["path"]
        Path(native.repository / bundle["artifact"]).unlink()
        bundle = write_task_context_bundle(repository=native.repository,
            prepared=[{"schema": "supervisor-task-context-preparation@1", "task_cid": "fixture:admitted-task",
                "task_id": native.binding.task_id, "metadata": {**native.metadata,
                    "Code retrieval sha256": native.binding.expected_sha256}}],
            output=native.repository / ".runtime/context-bundle.json")
        arguments[arguments.index("--semantic-decision-context-bundle-sha256") + 1] = bundle["sha256"]
    else:
        native.snapshot_artifact = None
    return native, bundle, arguments


def _typed_history(native, dispatch, *, current=False):
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_typed_nomination_lineage import (
        load_and_verify_typed_nomination_lineage,
    )
    binding = dispatch.receipt["typed_nomination"]
    return load_and_verify_typed_nomination_lineage(
        artifact=Path(binding["lineage_artifact"]), expected_sha256=binding["lineage_sha256"],
        native_prompt=native.native_prompt, repository=native.repository, workspace=native.workspace,
        expected_origin_nomination_sha256=hashlib.sha256(native.origin_raw).hexdigest(),
        expected_task_id=native.binding.task_id, owner_binding=dispatch.receipt, require_current_source=current)


def test_actual_typed_retrieval_owner_publishes_v2_lineage_and_exact_v3_for_derived_policy(typed_retrieval_launch, monkeypatch):
    native, _, _ = typed_retrieval_launch
    dispatch = _dispatch(typed_retrieval_launch)
    receipt = dispatch.receipt
    lineage = json.loads(Path(receipt["typed_nomination"]["lineage_artifact"]).read_bytes())
    assert lineage["schema"] == "supervisor-semantic-typed-nomination-lineage@2"
    retained = lineage["typed_nomination"]
    assert retained["schema"] == "supervisor-semantic-typed-owner-nomination@2"
    assert retained["receipt"]["schema"] == "supervisor-semantic-typed-owner-policy-receipt@2"
    assert retained["receipt"]["retrieval_query_binding"] == native.binding.to_dict()
    policy = worker.SemanticDecisionPolicy.from_dict(retained["policy"])
    assert policy.required_capsule_ids == native.origin_policy.required_capsule_ids
    assert policy.optional_capsule_ids == native.origin_policy.optional_capsule_ids
    assert policy.selected_optional_capsule_ids == native.expected_dependency_capsules
    assert set(native.origin_policy.optional_capsule_ids) - set(policy.selected_optional_capsule_ids)
    assert retained["launch_policy"]["schema"] == worker.RETRIEVAL_SCHEMA
    assert retained["launch_policy"]["retrieval_query_binding"] == native.binding.to_dict()
    assert worker.load_decision_launch_configuration(
        artifact=Path(receipt["launch_policy_artifact"]), expected_sha256=receipt["launch_policy_sha256"],
        native_prompt=native.native_prompt, repository=native.repository, workspace=native.workspace) == (policy, native.binding)
    monkeypatch.chdir(native.workspace)
    arguments, launch = worker._prepare_launch(prompt=native.native_prompt, model="pinned",
        semantic_repository=native.repository, semantic_decision_policy_artifact=Path(receipt["launch_policy_artifact"]),
        semantic_decision_policy_sha256=receipt["launch_policy_sha256"])
    assert arguments["retrieval_query_binding"] == native.binding
    assert arguments["policy"] == policy and arguments["native_prompt"] == native.native_prompt
    assert launch["retrieval_query_binding"] == native.binding.to_dict()
    verified = _typed_history(native, dispatch)
    assert verified["source_freshness_revalidated"] is False
    assert verified["provider_calls"] == 0
    assert all(verified[key] is False for key in ("proof_authority", "execution_authority", "completion_authority"))
    assert dispatch.prompt_bytes == native.native_prompt.encode()


def test_actual_typed_retrieval_history_replays_after_origin_source_and_workspace_removed(typed_retrieval_launch):
    import shutil
    native, _, _ = typed_retrieval_launch
    dispatch = _dispatch(typed_retrieval_launch)
    native.origin_path.unlink()
    (native.repository / "mod.py").unlink()
    shutil.rmtree(native.workspace)
    verified = _typed_history(native, dispatch)
    assert verified["source_freshness_revalidated"] is False and verified["provider_calls"] == 0
    with pytest.raises((ValueError, OSError)):
        _typed_history(native, dispatch, current=True)
    assert not (native.repository / ".runtime/semantic-typed-verification-cache").exists()


@pytest.mark.parametrize("mutation", ["bundle", "task", "raw-retrieval", "task-text-policy"])
def test_actual_typed_retrieval_refuses_independent_input_changes_before_publication(typed_retrieval_launch, mutation):
    native, bundle, arguments = typed_retrieval_launch
    if mutation == "bundle":
        (native.repository / bundle["artifact"]).write_bytes(b"{}")
    elif mutation == "task":
        arguments[arguments.index("--semantic-decision-task-cid") + 1] = "fixture:foreign-task"
    elif mutation == "raw-retrieval":
        native.retrieval_artifact.write_bytes(b"{}")
    else:
        native.origin_path.chmod(0o600)
        raw = json.dumps(owner.build_decision_nomination(replace(native.origin_policy,
            task_text_source_path=None)), sort_keys=True).encode()
        native.origin_path.write_bytes(raw)
        arguments[arguments.index("--semantic-decision-nomination-sha256") + 1] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError):
        _dispatch(typed_retrieval_launch)
    assert not (native.repository / ".runtime/semantic-decision-policies").exists()
    assert not (native.repository / ".runtime/semantic-typed-nominations").exists()


@pytest.mark.parametrize("mutation", ["raw-retrieval", "snapshot", "retained-source"])
def test_actual_typed_retrieval_history_refuses_mutated_retained_dependencies(typed_retrieval_launch, mutation):
    native, _, _ = typed_retrieval_launch
    dispatch = _dispatch(typed_retrieval_launch)
    if mutation == "raw-retrieval":
        path = native.retrieval_artifact
    elif mutation == "snapshot":
        if native.snapshot_artifact is None:
            # Inline snapshot bytes are part of the independently bound raw parent.
            path = native.retrieval_artifact
        else:
            path = native.snapshot_artifact
    else:
        source_cid = native.semantic_parent["manifest"]["mod.py"]["source_cid"]
        path = native.semantic_artifact.parent / "blocks" / source_cid
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        _typed_history(native, dispatch)


@pytest.mark.parametrize("mutation", ["outer-revision", "result-revision", "receipt-revision", "binding-task", "binding-sha"])
def test_actual_typed_retrieval_lineage_cannot_rehash_version_or_parent_substitutions(typed_retrieval_launch, mutation):
    from copy import deepcopy
    native, _, _ = typed_retrieval_launch
    dispatch = _dispatch(typed_retrieval_launch)
    owner_receipt = dispatch.receipt
    commit = owner_receipt["typed_nomination"]
    lineage = json.loads(Path(commit["lineage_artifact"]).read_bytes())
    if mutation == "outer-revision":
        lineage["schema"] = "supervisor-semantic-typed-nomination-lineage@1"
    elif mutation == "result-revision":
        lineage["typed_nomination"]["schema"] = "supervisor-semantic-typed-owner-nomination@1"
    elif mutation == "receipt-revision":
        lineage["typed_nomination"]["receipt"]["schema"] = "supervisor-semantic-typed-owner-policy-receipt@1"
    else:
        field = "task_id" if mutation == "binding-task" else "expected_sha256"
        lineage["typed_nomination"]["receipt"]["retrieval_query_binding"][field] = (
            "foreign-task" if mutation == "binding-task" else "0" * 64)
    path, digest = owner._publish(native.repository, native.workspace, lineage,
        directory="semantic-typed-nominations", maximum=2_000_000)
    changed = deepcopy(owner_receipt)
    changed["typed_nomination"].update(lineage_artifact=str(path), lineage_sha256=digest)
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_typed_nomination_lineage import load_and_verify_typed_nomination_lineage
    with pytest.raises(ValueError):
        load_and_verify_typed_nomination_lineage(artifact=path, expected_sha256=digest,
            native_prompt=native.native_prompt, repository=native.repository, workspace=native.workspace,
            expected_origin_nomination_sha256=hashlib.sha256(native.origin_raw).hexdigest(),
            expected_task_id=native.binding.task_id, owner_binding=changed)
