"""Versioned nomination custody and standalone validation without providers.

Native producer verification is mocked only in the light custody and audit
cases. Packet source admission and text reconstruction are tested separately.
"""

import hashlib
import json
from pathlib import Path
import stat

import pytest

from benchmarks.agent_supervisor.container_coding import semantic_decision_nomination_delivery as delivery
from benchmarks.agent_supervisor.container_coding import semantic_decision_input_audit as input_audit
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_decision_owner as owner
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_decision_worker as worker
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_packet import SemanticDecisionPolicy


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _policy(opted_in):
    return SemanticDecisionPolicy(
        required_capsule_ids=("required-λ",), optional_capsule_ids=("optional-雪",),
        **({"task_text_source_path": "instructions/task λ.md"} if opted_in else {}),
    )


@pytest.fixture
def remote_validate():
    # Load the exact stdlib-only remote validator, without running main or
    # touching upload files. No ownership or root-authority adaptation occurs.
    code = delivery.NOMINATION_DELIVERY_CODE
    footer = "\ntry:\n    main()\nfinally:"
    assert code.count(footer) == 1
    namespace = {"__name__": "authored_nomination_validator"}
    exec(compile(code.split(footer, 1)[0], "<authored-nomination-validator>", "exec"), namespace)
    return namespace["validate"]


@pytest.mark.parametrize("opted_in", [False, True], ids=["legacy", "task-text-reference"])
def test_policy_revision_survives_snapshot_owner_binding_and_worker_loader(
    tmp_path, monkeypatch, remote_validate, opted_in,
):
    repository, workspace = tmp_path / "repository", tmp_path / "candidate"
    repository.mkdir(mode=0o755)
    workspace.mkdir(mode=0o755)
    snapshot_directory = tmp_path / "snapshot"
    snapshot_directory.mkdir(mode=0o755)
    policy = _policy(opted_in)
    revision = "2" if opted_in else "1"
    nomination = owner.build_decision_nomination(policy)
    assert nomination["schema"] == "supervisor-semantic-decision-nomination@" + revision
    assert set(nomination) == {"schema", "policy"}
    assert set(nomination["policy"]) == {
        "required_capsule_ids", "optional_capsule_ids", "selected_optional_capsule_ids",
    } | ({"task_text_source_path"} if opted_in else set())
    raw = ("  " + json.dumps(nomination, ensure_ascii=False, indent=2) + "\r\n").encode()
    source = tmp_path / "nomination.json"
    source.write_bytes(raw)
    digest = _sha(raw)
    assert delivery._validate_nomination(raw) is None
    assert remote_validate(raw) is None
    snapshot = delivery.capture_nomination_snapshot(
        source=source, expected_sha256=digest, directory=snapshot_directory,
    )
    assert delivery.verify_nomination_snapshot(snapshot, directory=snapshot_directory) == raw
    assert owner.load_decision_nomination(
        artifact=Path(snapshot["artifact"]), expected_sha256=digest, workspace=workspace,
    ) == policy

    prompt = b'{"objective_id":"literal-task"}\nLITERAL PUBLIC AND DOCTOR SUFFIX\n'
    producer_calls = []

    def parts(text):
        wire, end = json.JSONDecoder().raw_decode(text)
        return wire, text[end:], (), (), "retained native parent", ".runtime/semantic/worker-context.json"

    def verify_projection(**kwargs):
        producer_calls.append(kwargs)

    monkeypatch.setattr(worker.codec, "_prompt_parts", parts)
    monkeypatch.setattr(worker, "project_semantic_router_prompt", verify_projection)
    arguments = ["--model", "pinned", "--semantic-repository", str(repository),
                 "--timeout", "120", "--semantic-decision-nomination-artifact", snapshot["artifact"],
                 "--semantic-decision-nomination-sha256", digest,
                 "--semantic-decision-task-id", "literal-task"]
    dispatch = owner.prepare_decision_owner_dispatch(
        arguments=arguments, prompt_bytes=prompt, workspace=workspace,
    )
    assert dispatch.prompt_bytes == prompt
    assert dispatch.worker_arguments[:-4] == tuple(arguments[:6])
    assert dispatch.worker_arguments[-4::2] == (
        "--semantic-decision-policy-artifact", "--semantic-decision-policy-sha256",
    )
    launch_path = Path(dispatch.receipt["launch_policy_artifact"])
    launch_raw = launch_path.read_bytes()
    launch = json.loads(launch_raw)
    assert launch == {
        "schema": "supervisor-semantic-decision-launch-policy@" + revision,
        "task_id": "literal-task", "parent_artifact": ".runtime/semantic/worker-context.json",
        "parent_sha256": _sha(b"retained native parent"), "policy": policy.to_dict(),
    }
    assert stat.S_IMODE(launch_path.stat().st_mode) == 0o444 and launch_path.stat().st_nlink == 1
    assert dispatch.receipt["policy_sha256"] == _sha(_encoded(policy.to_dict()))
    assert dispatch.receipt["provider_called"] is dispatch.receipt["completion_authority"] is False
    assert worker.load_decision_launch_policy(
        artifact=launch_path, expected_sha256=_sha(launch_raw), native_prompt=prompt.decode(),
        repository=repository, workspace=workspace,
    ) == policy
    assert producer_calls == [{"prompt": prompt.decode(), "repository": repository, "decision_policy": policy}] * 2


@pytest.mark.parametrize("alteration", [
    "unknown-schema", "legacy-with-path", "new-without-path", "outer-extra", "policy-extra",
    "path-none", "path-bool", "path-list", "path-empty", "path-whitespace", "path-absolute",
    "path-parent", "path-dot", "path-empty-component", "path-trailing-slash", "path-newline",
    "path-nul", "path-over-bound", "nonlist", "duplicate-id", "overlap", "selected-outside",
    "duplicate-outer-key", "duplicate-policy-key",
])
def test_owner_host_and_literal_remote_reject_version_or_closed_policy_mismatch(
    tmp_path, remote_validate, alteration,
):
    value = owner.build_decision_nomination(_policy(True))
    if alteration == "unknown-schema":
        value["schema"] = "supervisor-semantic-decision-nomination@3"
    elif alteration == "legacy-with-path":
        value["schema"] = owner.NOMINATION_SCHEMA
    elif alteration == "new-without-path":
        del value["policy"]["task_text_source_path"]
    elif alteration == "outer-extra":
        value["authority"] = True
    elif alteration == "policy-extra":
        value["policy"]["infer_duplicate_text"] = True
    elif alteration.startswith("path-"):
        value["policy"]["task_text_source_path"] = {
            "path-none": None, "path-bool": True, "path-list": ["task.md"], "path-empty": "",
            "path-whitespace": " ", "path-absolute": "/task.md", "path-parent": "../task.md",
            "path-dot": "./task.md", "path-empty-component": "instructions//task.md",
            "path-trailing-slash": "instructions/", "path-newline": "task\n.md",
            "path-nul": "task\x00.md", "path-over-bound": "x" * 4097,
        }[alteration]
    elif alteration == "nonlist":
        value["policy"]["required_capsule_ids"] = "required-λ"
    elif alteration == "duplicate-id":
        value["policy"]["required_capsule_ids"] *= 2
    elif alteration == "overlap":
        value["policy"]["optional_capsule_ids"] = value["policy"]["required_capsule_ids"]
    elif alteration == "selected-outside":
        value["policy"]["selected_optional_capsule_ids"] = ["foreign"]
    raw = _encoded(value)
    if alteration == "duplicate-outer-key":
        raw = b'{"schema":"shadow",' + raw[1:]
    elif alteration == "duplicate-policy-key":
        raw = raw.replace(b'"policy":{', b'"policy":{"task_text_source_path":"shadow.md",', 1)
    artifact = tmp_path / "invalid-nomination.json"
    artifact.write_bytes(raw)
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    with pytest.raises(ValueError):
        delivery.read_nomination_source(source=artifact, expected_sha256=_sha(raw))
    with pytest.raises(ValueError):
        remote_validate(raw)
    with pytest.raises(ValueError):
        owner.load_decision_nomination(artifact=artifact, expected_sha256=_sha(raw), workspace=workspace)


@pytest.mark.parametrize("opted_in", [False, True], ids=["legacy-policy-in-new-schema", "new-policy-in-legacy-schema"])
def test_launch_policy_revision_mismatch_refuses_before_parent_replay(tmp_path, monkeypatch, opted_in):
    repository, workspace = tmp_path / "repository", tmp_path / "candidate"
    repository.mkdir()
    workspace.mkdir()
    descriptor = {
        "schema": "supervisor-semantic-decision-launch-policy@" + ("1" if opted_in else "2"),
        "task_id": "task", "parent_artifact": ".runtime/semantic/worker-context.json",
        "parent_sha256": "a" * 64, "policy": _policy(opted_in).to_dict(),
    }
    raw = _encoded(descriptor)
    artifact = tmp_path / "launch.json"
    artifact.write_bytes(raw)
    calls = []
    monkeypatch.setattr(worker, "build_decision_launch_policy", lambda **kwargs: calls.append(kwargs))
    with pytest.raises(ValueError, match="revision differ"):
        worker.load_decision_launch_policy(
            artifact=artifact, expected_sha256=_sha(raw), native_prompt="must not be decoded",
            repository=repository, workspace=workspace,
        )
    assert calls == []


def _audit_launch(tmp_path, opted_in):
    repository, workspace = tmp_path / "repository", tmp_path / "candidate"
    repository.mkdir()
    workspace.mkdir()
    parent_artifact = ".runtime/semantic/worker-context.json"
    parent_path = repository / parent_artifact
    parent_path.parent.mkdir(parents=True)
    parent_raw = _encoded({"schema": "authored-native-parent@1", "task_id": "literal-task"})
    parent_path.write_bytes(parent_raw)
    wire = {"objective_id": "literal-task", "repository_id": "repository-current", "tree_id": "tree-current"}
    suffix = "\nLITERAL PUBLIC AND DOCTOR SUFFIX\n"
    native_prompt = _encoded(wire).decode() + suffix
    policy = _policy(opted_in)
    descriptor = {
        "schema": "supervisor-semantic-decision-launch-policy@" + policy.schema_revision,
        "task_id": wire["objective_id"], "parent_artifact": parent_artifact,
        "parent_sha256": _sha(parent_raw), "policy": policy.to_dict(),
    }
    artifact = tmp_path / "immutable-launch.json"
    artifact.write_bytes(_encoded(descriptor))
    artifact.chmod(0o444)
    launch = {"launch_policy_artifact": str(artifact), "launch_policy_sha256": _sha(artifact.read_bytes())}
    return repository, workspace, parent_artifact, parent_raw, wire, suffix, native_prompt, policy, descriptor, artifact, launch


@pytest.mark.parametrize("opted_in", [False, True], ids=["legacy", "task-text-reference"])
def test_historical_audit_policy_parser_retains_both_closed_revisions(tmp_path, monkeypatch, opted_in):
    (repository, workspace, parent_artifact, parent_raw, wire, suffix, native_prompt,
     policy, _, artifact, launch) = _audit_launch(tmp_path, opted_in)
    calls = []

    def parts(prompt):
        assert prompt == native_prompt
        return wire, suffix, (0,), (), parent_raw.decode(), parent_artifact

    def verified(root, *, artifact, text, task_id, current):
        calls.append((root, artifact, text, task_id, current))
        assert (root / artifact).read_bytes() == parent_raw == text.encode()
        return {"semantic_root_cid": "root-current", "scope_cid": "scope-current"}, ()

    monkeypatch.setattr(worker.codec, "_prompt_parts", parts)
    monkeypatch.setattr(worker.codec, "_verified_semantic", verified)
    loaded, bindings, retained_wire, retained_suffix, positions, references, text, policy_digest = input_audit._policy(
        launch=launch, workspace=workspace, native_prompt=native_prompt, repository=repository,
    )
    assert loaded == policy
    assert loaded.to_dict() == policy.to_dict()
    assert bindings.to_dict() == {
        "artifact": parent_artifact, "expected_sha256": _sha(parent_raw), "task_id": wire["objective_id"],
        "semantic_root_cid": "root-current", "scope_cid": "scope-current",
        "repository_id": wire["repository_id"], "tree_id": wire["tree_id"],
    }
    assert (retained_wire, retained_suffix, positions, references, text) == (
        wire, suffix, (0,), (), parent_raw.decode(),
    )
    assert policy_digest == _sha(_encoded(policy.to_dict()))
    assert calls == [(repository, parent_artifact, parent_raw.decode(), wire["objective_id"], False)]
    assert stat.S_IMODE(artifact.stat().st_mode) == 0o444 and artifact.stat().st_nlink == 1
    assert launch["launch_policy_sha256"] == _sha(artifact.read_bytes())


@pytest.mark.parametrize("opted_in", [False, True], ids=["legacy-policy-in-new-schema", "new-policy-in-legacy-schema"])
def test_historical_audit_rejects_policy_revision_mismatch_before_native_replay(tmp_path, monkeypatch, opted_in):
    (repository, workspace, _, _, _, _, native_prompt, _, descriptor,
     artifact, launch) = _audit_launch(tmp_path, opted_in)
    descriptor["schema"] = "supervisor-semantic-decision-launch-policy@" + ("1" if opted_in else "2")
    artifact.chmod(0o644)
    artifact.write_bytes(_encoded(descriptor))
    artifact.chmod(0o444)
    launch["launch_policy_sha256"] = _sha(artifact.read_bytes())
    calls = []
    monkeypatch.setattr(worker.codec, "_prompt_parts", lambda *args, **kwargs: calls.append("prompt-parts"))
    monkeypatch.setattr(worker.codec, "_verified_semantic", lambda *args, **kwargs: calls.append("native-parent"))
    with pytest.raises(ValueError, match="invalid_selection_policy_revision"):
        input_audit._policy(launch=launch, workspace=workspace, native_prompt=native_prompt, repository=repository)
    assert calls == []
