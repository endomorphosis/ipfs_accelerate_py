"""Owner launch custody, literal requirements and dialogue receipts, offline."""

import hashlib
import io
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import semantic_decision_worker as worker
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import PromptTokenCounter
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_dialogue import (
    FINAL_SCHEMA, REQUEST_SCHEMA, ROUND_SCHEMA, SCHEMA as DIALOGUE_SCHEMA,
    SemanticDecisionDialogueError, SemanticDecisionReply, SemanticDecisionResult,
    SemanticDecisionProviderFailure,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_packet import SemanticDecisionPolicy

pytest_plugins = ("test.api.test_semantic_router_integration",)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def launch(allocated, monkeypatch):
    repository, workspace, prompt, source = allocated
    parent = json.loads((repository / ".runtime/semantic/worker-context.json").read_text())
    ids = tuple(row["capsule_cid"] for row in parent["capsules"])
    policy = SemanticDecisionPolicy(required_capsule_ids=ids[1:], optional_capsule_ids=ids[:1])
    descriptor = worker.build_decision_launch_policy(native_prompt=prompt, policy=policy, repository=repository)
    artifact = repository.parent / "owner-selection.json"
    artifact.write_text(json.dumps(descriptor, sort_keys=True, separators=(",", ":")))
    counter = PromptTokenCounter(
        provider="codex_cli", model="pinned", tokenizer_id="authored-fixture",
        tokenizer_revision="1", special_token_policy="fixture only", count_kind="estimate",
        compatibility_basis="authored fixture only",
        count_tokens=lambda text: 100 if ROUND_SCHEMA in text else 1000,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime import automatic_prompt_tokenizer
    monkeypatch.setattr(automatic_prompt_tokenizer, "resolve_prompt_tokenizer", lambda **kwargs:
                        (counter, {"schema": "fixture-tokenizer", "remote_compatibility": False}))
    return repository, workspace, prompt, source, policy, descriptor, artifact, counter


def _kwargs(launch):
    repository, _, prompt, _, _, _, artifact, _ = launch
    return dict(
        prompt=prompt, model="pinned", semantic_repository=repository,
        semantic_decision_policy_artifact=artifact,
        semantic_decision_policy_sha256=_sha(artifact.read_bytes()),
    )


def _load(launch, artifact=None, expected_sha256=None):
    repository, workspace, prompt, _, _, _, original, _ = launch
    selected = artifact or original
    return worker.load_decision_launch_policy(
        artifact=selected, expected_sha256=expected_sha256 or _sha(selected.read_bytes()),
        native_prompt=prompt, repository=repository, workspace=workspace,
    )


def test_launch_descriptor_authenticates_complete_native_parent(launch):
    repository, _, prompt, _, policy, descriptor, _, _ = launch
    wire, _, _, _, text, artifact = worker.codec._prompt_parts(prompt)
    assert descriptor == {
        "schema": worker.SCHEMA, "task_id": wire["objective_id"],
        "parent_artifact": artifact, "parent_sha256": _sha(text.encode()), "policy": policy.to_dict(),
    }
    assert _load(launch) == policy
    assert (repository / artifact).read_bytes() == text.encode()


@pytest.mark.parametrize("alteration", ["digest", "schema", "task", "parent", "extra", "partition", "policy_extra", "policy_type"])
def test_launch_descriptor_rejects_changed_bindings_and_policy(launch, alteration):
    *_, descriptor, artifact, _ = launch
    expected = _sha(artifact.read_bytes())
    if alteration == "digest":
        artifact.write_bytes(artifact.read_bytes() + b" ")
    elif alteration == "schema":
        descriptor["schema"] = "unknown@1"
    elif alteration == "task":
        descriptor["task_id"] = "foreign-task"
    elif alteration == "parent":
        descriptor["parent_sha256"] = "a" * 64
    elif alteration == "extra":
        descriptor["authority"] = True
    elif alteration == "partition":
        descriptor["policy"]["optional_capsule_ids"] = []
    elif alteration == "policy_extra":
        descriptor["policy"]["infer_optional"] = True
    else:
        descriptor["policy"]["required_capsule_ids"] = "capsules"
    if alteration != "digest":
        artifact.write_text(json.dumps(descriptor))
        expected = _sha(artifact.read_bytes())
    with pytest.raises(ValueError):
        _load(launch, expected_sha256=expected)


@pytest.mark.parametrize("nested", [False, True])
def test_launch_descriptor_rejects_duplicate_json_fields(launch, nested):
    artifact = launch[6]
    text = artifact.read_text()
    if nested:
        text = text.replace('"policy":{', '"policy":{"required_capsule_ids":[],', 1)
    else:
        text = '{"schema":"shadow",' + text[1:]
    artifact.write_text(text)
    with pytest.raises(ValueError, match="duplicate"):
        _load(launch)


@pytest.mark.parametrize("location", ["candidate", "symlink", "directory", "over_bound"])
def test_launch_policy_is_bounded_regular_canonical_external_input(launch, location):
    _, workspace, _, _, _, _, original, _ = launch
    if location == "candidate":
        path = workspace / "policy.json"
        path.write_bytes(original.read_bytes())
    elif location == "symlink":
        path = original.parent / "policy-link.json"
        path.symlink_to(original)
    elif location == "directory":
        path = original.parent / "policy-directory"
        path.mkdir()
    else:
        path = original
        path.write_bytes(b" " * (worker.MAX_POLICY_BYTES + 1))
    with pytest.raises(ValueError):
        _load(launch, artifact=path, expected_sha256="a" * 64)


def test_launch_replays_current_source_before_provider(launch, monkeypatch):
    repository, _, _, source, _, _, _, _ = launch
    calls = []
    monkeypatch.setattr(worker, "run_shared_router_decision_dialogue", lambda **kwargs: calls.append(kwargs))
    (repository / "mod.py").write_text(source + "\n# stale canonical source\n")
    with pytest.raises(ValueError):
        worker.run(**_kwargs(launch))
    assert calls == []


def test_worker_loads_requirements_against_original_and_counts_literal_suffix(launch, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import doctor_residual_context, router_public_instruction

    repository, workspace, prompt, _, policy, _, _, counter = launch
    loaders, calls = [], []

    def residual(**kwargs):
        loaders.append(("residual", kwargs))
        return "\nLITERAL DOCTOR REQUIREMENT", {"schema": "fixture-residual"}

    def instruction(**kwargs):
        loaders.append(("instruction", kwargs))
        return "\nLITERAL PUBLIC REQUIREMENT", {"schema": "fixture-public"}

    def dialogue(**kwargs):
        calls.append(kwargs)
        return SemanticDecisionResult("candidate response", json.dumps({"schema": DIALOGUE_SCHEMA, "attempts": []}))

    monkeypatch.setattr(doctor_residual_context, "load_doctor_residual_advisory", residual)
    monkeypatch.setattr(router_public_instruction, "load_public_instruction", instruction)
    monkeypatch.setattr(worker, "run_shared_router_decision_dialogue", dialogue)
    result = worker.run(
        **_kwargs(launch), doctor_residual_artifact=Path("/fixture/residual.json"),
        doctor_residual_sha256="b" * 64, doctor_residual_task_cid="residual-task",
        public_instruction_artifact=Path("/fixture/public.json"), public_instruction_sha256="c" * 64,
        public_instruction_task_cid="public-task", semantic_decision_max_calls=2,
        max_input_tokens=1500, semantic_decision_max_total_input_tokens=2500,
        reasoning_effort="medium", timeout=40, max_output_tokens=512,
    )
    assert [name for name, _ in loaders] == ["residual", "instruction"]
    assert all(row["prompt"] == prompt and row["workspace"] == workspace for _, row in loaders)
    invocation = calls[0]
    assert invocation["native_prompt"] == prompt + "\nLITERAL DOCTOR REQUIREMENT\nLITERAL PUBLIC REQUIREMENT"
    assert invocation["policy"] == policy and invocation["repository"] == repository
    assert invocation["prompt_token_counter"] is counter
    assert invocation["max_calls"] == 2 and invocation["max_total_input_tokens"] == 2500
    assert invocation["max_input_tokens"] == 1500 and invocation["timeout"] == 40
    assert invocation["max_output_tokens"] == 512 and invocation["reasoning_effort"] == "medium"
    assert result.receipt["worker_launch"]["original_native_prompt_sha256"] == _sha(prompt.encode())
    assert result.receipt["worker_launch"]["public_instruction"] == {"schema": "fixture-public"}


@pytest.mark.parametrize("override", [
    {"provider": "grok_cli"}, {"purpose": "planning"}, {"timeout": 601},
    {"semantic_decision_max_calls": 0}, {"semantic_decision_max_calls": True},
    {"semantic_decision_max_total_input_tokens": 0}, {"max_input_tokens": False},
    {"max_output_tokens": 0}, {"public_instruction_sha256": "a" * 64},
    {"doctor_residual_task_cid": "partial"}, {"prompt_tokenizer_sha256": "a" * 64},
    {"container_boundary_sha256": "a" * 64},
])
def test_worker_rejects_invalid_options_before_provider(launch, monkeypatch, override):
    calls = []
    monkeypatch.setattr(worker, "run_shared_router_decision_dialogue", lambda **kwargs: calls.append(kwargs))
    with pytest.raises(ValueError):
        worker.run(**(_kwargs(launch) | override))
    assert calls == []


def test_worker_explicit_tokenizer_overrides_automatic_binding(launch, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import automatic_prompt_tokenizer, local_prompt_tokenizer

    counter, calls = launch[-1], []

    class Local:
        def binding(self):
            return counter.binding()

        def count(self, text):
            return 123

        def provenance(self):
            return {"schema": "explicit-fixture-tokenizer"}

    def local(**kwargs):
        calls.append(kwargs)
        return Local()

    def automatic(**kwargs):
        raise AssertionError("explicit descriptor must not select automatic tokenizer")

    def dialogue(**kwargs):
        assert kwargs["prompt_token_counter"].count("complete text") == 123
        assert kwargs["prompt_token_counter"].compatibility_basis == "owner_declaration"
        return SemanticDecisionResult("done", json.dumps({"schema": DIALOGUE_SCHEMA}))

    monkeypatch.setattr(local_prompt_tokenizer, "load_owner_prompt_tokenizer", local)
    monkeypatch.setattr(automatic_prompt_tokenizer, "resolve_prompt_tokenizer", automatic)
    monkeypatch.setattr(worker, "run_shared_router_decision_dialogue", dialogue)
    result = worker.run(**_kwargs(launch), prompt_tokenizer_artifact=Path("/fixture/tokenizer.json"),
                        prompt_tokenizer_sha256="d" * 64)
    assert calls == [{"artifact": Path("/fixture/tokenizer.json"), "expected_sha256": "d" * 64,
                      "provider": "codex_cli", "model": "pinned"}]
    assert result.receipt["worker_launch"]["prompt_tokenizer_source"] == {"schema": "explicit-fixture-tokenizer"}


@pytest.mark.parametrize("budget", [50, 150, 250])
def test_worker_real_dialogue_preserves_budget_attempts_and_container_binding(launch, monkeypatch, budget):
    from ipfs_accelerate_py.agent_supervisor.runtime import container_worker_boundary, semantic_decision_provider

    calls, adapters, boundary_checks = [], [], []
    optional = launch[4].optional_capsule_ids[0]
    boundary = Path("/fixture/container-boundary.json")
    monkeypatch.setattr(container_worker_boundary, "verify_container_worker_boundary", lambda **kwargs:
                        boundary_checks.append(kwargs) or {"schema": "fixture-boundary"})

    def factory(**bindings):
        adapters.append(bindings)

        def invoke(call):
            calls.append(call)
            row = json.loads(call.prompt.splitlines()[0])
            response = {name: row[name] for name in ("session_id", "generation", "binding_sha256")}
            if len(calls) == 1:
                response.update(schema=REQUEST_SCHEMA, capsule_ids=[optional])
            else:
                response.update(schema=FINAL_SCHEMA, text="native candidate ready")
            return SemanticDecisionReply(json.dumps(response), {
                "provider": call.provider, "model": call.model,
                "model_prompt_sha256": call.model_prompt_sha256, "provider_called": True,
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            })

        return invoke

    monkeypatch.setattr(semantic_decision_provider, "SharedRouterDecisionProvider", factory)
    options = _kwargs(launch) | dict(
        semantic_decision_max_total_input_tokens=budget,
        container_boundary=boundary, container_boundary_sha256="e" * 64,
    )
    if budget < 200:
        with pytest.raises(SemanticDecisionDialogueError) as captured:
            worker.run(**options)
        receipt = captured.value.receipt
        assert receipt["status"] == "failed"
        assert receipt["failure_reason"] == "total_input_budget_exceeded"
    else:
        result = worker.run(**options)
        receipt = result.receipt
        assert result.text == "native candidate ready"
    expected_calls = 0 if budget == 50 else 1 if budget == 150 else 2
    assert len(calls) == len(receipt["attempts"]) == expected_calls
    assert receipt["local_input_content_token_sum"] == expected_calls * 100
    assert receipt["worker_launch"]["launch_policy_sha256"] == options["semantic_decision_policy_sha256"]
    assert len(boundary_checks) == 1
    assert adapters[0]["container_boundary"] == boundary
    assert adapters[0]["container_boundary_sha256"] == "e" * 64
    assert all(row["provider_receipt"]["usage"]["prompt_tokens"] == 10 for row in receipt["attempts"])


def _argv(launch):
    options = _kwargs(launch)
    return ["--model", options["model"], "--semantic-repository", str(options["semantic_repository"]),
            "--semantic-decision-policy-artifact", str(options["semantic_decision_policy_artifact"]),
            "--semantic-decision-policy-sha256", options["semantic_decision_policy_sha256"]]


@pytest.mark.parametrize("failure", [False, True])
def test_worker_cli_prints_full_receipt_before_text_or_failed_exit(launch, monkeypatch, capsys, failure):
    receipt = {"schema": DIALOGUE_SCHEMA, "attempts": [{"provider_receipt": {
        "usage": {"prompt_tokens": 13, "completion_tokens": 2}, "provider_called": True,
    }}], "status": "failed" if failure else "returned"}

    def dialogue(**kwargs):
        if failure:
            raise SemanticDecisionDialogueError("provider_failed", receipt)
        return SemanticDecisionResult("literal final text", json.dumps(receipt))

    monkeypatch.setattr(worker, "run_shared_router_decision_dialogue", dialogue)
    monkeypatch.setattr(worker.sys, "stdin", io.TextIOWrapper(io.BytesIO(launch[2].encode())))
    assert worker.main(_argv(launch)) == (1 if failure else 0)
    lines = capsys.readouterr().out.splitlines()
    returned = json.loads(lines[0])
    assert returned["attempts"] == receipt["attempts"]
    assert "worker_launch" in returned
    assert lines[1:] == ([] if failure else ["literal final text"])


@pytest.mark.parametrize("input_bytes", [b"\xff", b"x" * (worker.MAX_MODEL_PROMPT_BYTES + 1)],
                         ids=["invalid_utf8", "over_byte_limit"])
def test_worker_cli_pre_provider_refusal_does_not_invent_attempt(launch, monkeypatch, capsys, input_bytes):
    calls = []
    monkeypatch.setattr(worker, "run_shared_router_decision_dialogue", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(worker.sys, "stdin", io.TextIOWrapper(io.BytesIO(input_bytes)))
    assert worker.main(_argv(launch)) == 1
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["schema"] == worker.REFUSAL_SCHEMA
    assert receipt["provider_called"] is False and receipt["usage"] is None
    assert "attempts" not in receipt and calls == []


def test_worker_cli_unexpected_dialogue_error_retains_receipt_and_unknown_dispatch(launch, monkeypatch, capsys):
    native = {"provider_called": True, "usage": {"prompt_tokens": 17}, "status": "failed"}

    def fail(**kwargs):
        raise SemanticDecisionProviderFailure("authored unexpected provider error", receipt=native)

    monkeypatch.setattr(worker, "run_shared_router_decision_dialogue", fail)
    monkeypatch.setattr(worker.sys, "stdin", io.TextIOWrapper(io.BytesIO(launch[2].encode())))
    assert worker.main(_argv(launch)) == 1
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "failed" and receipt["provider_called"] is None
    assert receipt["retained_receipt"] == native
    assert "attempts" not in receipt
