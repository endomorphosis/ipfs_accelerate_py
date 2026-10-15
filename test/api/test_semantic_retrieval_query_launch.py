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


def test_signed_retrieval_triplet_refuses_typed_nomination_before_dispatch():
    with pytest.raises(ValueError, match="typed nomination disabled"):
        owner._literal_arguments(_arguments() + _retrieval()
            + ["--semantic-decision-typed-nomination", "enabled"])


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_query_reference_configuration_requires_literal_boolean(value):
    with pytest.raises(ValueError):
        semantic_decision_arguments(semantic_retrieval_query_reference=value)


@pytest.mark.parametrize("settings", [
    {},
    {"semantic_decision_policy_artifact": "/owner/policy.json", "semantic_decision_policy_sha256": "a" * 64},
    {"semantic_decision_nomination_artifact": "/owner/nomination.json", "semantic_decision_nomination_sha256": "a" * 64},
])
def test_query_reference_configuration_requires_explicit_plain_owner_nomination(settings):
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
