"""Retrieval references preserve complete round inputs and historical custody.

Parent/source authentication callbacks below are explicitly authored fixtures.
The real reference/packet/chunk builders and dialogue/audit are exercised, but
these checks do not claim native producer admission, model usage or savings.
"""
from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ContextCompiler, build_text_context_references, render_context_capsule,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_decision_dialogue as dialogue
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_retrieval_query_reference as retrieval
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_projection_dispatch as dispatch
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import PromptTokenCounter
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_packet import SemanticDecisionPolicy
from test.api.semantic_state.test_semantic_decision_task_text_projection import (
    INSTRUCTION, SOURCE, original_parent, parent as retrieval_parent, policy as text_policy,
)
from test.api.semantic_state.test_semantic_decision_packet import OPTIONAL
from test.api.test_semantic_decision_input_audit import _provider_receipt, _assert_observation_only
from test.api.test_semantic_retrieval_query_reference import build_retrieval_native_fixture
import benchmarks.agent_supervisor.container_coding.semantic_decision_input_audit as audit


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def counter(*, fallback=False, after_count=None):
    def measure(text):
        if after_count is not None:
            after_count(text)
        return 100 if fallback or dialogue.ROUND_SCHEMA in text else 1000
    return PromptTokenCounter(provider="codex_cli", model="pinned", tokenizer_id="authored-retrieval-counter",
        tokenizer_revision="fixture@1", special_token_policy="literal", count_tokens=measure,
        count_kind="estimate", compatibility_basis="authored_fixture_only")


@pytest.fixture
def retrieval_case(retrieval_parent, tmp_path, monkeypatch):
    parent = copy.deepcopy(retrieval_parent)
    repository, workspace = tmp_path / "repository", tmp_path / "worktree"
    repository.mkdir()
    workspace.mkdir()
    artifact = repository / ".runtime/semantic/worker-context.json"
    artifact.parent.mkdir(parents=True)
    text = encoded(parent)
    artifact.write_text(text)
    raw_parent = repository / ".runtime/retrieval.json"
    raw_parent.write_text(encoded({"authored_raw_parent": "independently pinned fixture"}))
    binding = retrieval.SemanticRetrievalQueryBinding(
        artifact=raw_parent.relative_to(repository).as_posix(), expected_sha256=sha(raw_parent.read_bytes()),
        task_id=parent["task_id"])
    native = {
        "schema": "supervisor-code-retrieval-context@1", "task_id": parent["task_id"],
        "status": "current", "stale_paths": [], "index_id": "authored-index",
        "query_id": "authored-query", "result_id": "authored-result", "query_text": INSTRUCTION,
        "source_sha256": {"dependency.py": "d" * 64},
        "hits": [{"path": "dependency.py", "symbol": "dependency.work", "rank": 1, "score": 0.75,
                  "call_refs": ["authored-call"], "effect_refs": ["authored-effect"]}],
        "nomination_only": True, "semantic_authority": False,
        "execution_authority": False, "completion_authority": False,
    }
    native_text = encoded(native)
    state = {"stale": False, "live_checks": 0, "historical_checks": 0, "native_text": native_text}

    def verified(root, *, artifact: str, text: str, task_id: str, current: bool):
        assert root == repository and task_id == parent["task_id"]
        assert artifact == ".runtime/semantic/worker-context.json"
        assert text == (repository / artifact).read_text()
        if current and state["stale"]:
            raise ValueError("authored source freshness refusal")
        return copy.deepcopy(parent), tuple(row["stable_symbol_id"] for row in parent["capsules"])

    def authenticated(*, repository, binding, semantic_original_text, semantic_bindings, current):
        assert repository == artifact.parents[2] and semantic_original_text == text
        assert semantic_bindings.expected_sha256 == sha(text)
        assert binding.task_id == semantic_bindings.task_id == parent["task_id"]
        if sha(raw_parent.read_bytes()) != binding.expected_sha256:
            raise ValueError("authored raw parent pin refusal")
        state["live_checks" if current else "historical_checks"] += 1
        if current and state["stale"]:
            raise ValueError("authored live retrieval source refusal")
        return state["native_text"] if current else native_text

    monkeypatch.setattr(codec, "_verified_semantic", verified)
    monkeypatch.setattr(retrieval, "load_live_retrieval_query_context",
                        lambda **kwargs: authenticated(**kwargs, current=True))
    if hasattr(dispatch, "load_live_retrieval_query_context"):
        monkeypatch.setattr(dispatch, "load_live_retrieval_query_context",
                            lambda **kwargs: authenticated(**kwargs, current=True))
    monkeypatch.setattr(retrieval, "load_historical_retrieval_query_context",
                        lambda **kwargs: authenticated(**kwargs, current=False))
    options = {"repository_id": "repo:retrieval-dialogue", "tree_id": "tree:retrieval-dialogue", "required": True}
    semantic_refs = build_text_context_references(text, reference_prefix="semantic-context", kind="semantic-context",
        path=artifact.relative_to(repository).as_posix(), chunk_bytes=1201, **options)
    retrieval_refs = build_text_context_references(native_text, reference_prefix="code-retrieval-context",
        kind="code-retrieval-context", path=binding.artifact, chunk_bytes=331,
        coverage_ids=("original-retrieval-coverage",), **options)
    unrelated = build_text_context_references("Keep this unrelated literal constraint.  \r\n",
        reference_prefix="unrelated", kind="literal-constraint", **options)
    compiled = ContextCompiler(ContextBudget(max_input_tokens=32768, max_items=128,
        max_item_bytes=16384, max_serialized_bytes=262144)).compile(
        repository_id=options["repository_id"], tree_id=options["tree_id"], objective_id=parent["task_id"],
        objective_revision="sha256:task", policy_id="policy:authored", policy_revision="sha256:policy",
        caller="supervisor:authored", stage="implementation", goal={"id": parent["task_id"]},
        authority={"completion_authority": False}, scope={"allowed_paths": ["target.py"]},
        acceptance={"criteria": ["Retain every literal constraint"]},
        evidence=(*unrelated, *semantic_refs, *retrieval_refs))
    suffix = "\r\nLiteral café suffix with spaces.  \r\n"
    prompt = render_context_capsule(compiled.capsule) + suffix
    policy = text_policy()
    descriptor = {"schema": "supervisor-semantic-decision-launch-policy@3", "task_id": parent["task_id"],
        "parent_artifact": artifact.relative_to(repository).as_posix(), "parent_sha256": sha(text),
        "policy": policy.to_dict(), "retrieval_query_binding": binding.to_dict()}
    policy_artifact = tmp_path / "protected-policy.json"
    policy_artifact.write_text(encoded(descriptor))
    policy_artifact.chmod(0o444)
    launch = {"launch_policy_artifact": str(policy_artifact), "launch_policy_sha256": sha(policy_artifact.read_bytes()),
        "original_native_prompt_sha256": sha(prompt), "original_native_prompt_bytes": len(prompt.encode()),
        "required_suffix_sha256": sha(b""), "required_suffix_bytes": 0,
        "retrieval_query_binding": binding.to_dict()}
    return SimpleNamespace(repository=repository, workspace=workspace, workspace_root=tmp_path,
        artifact=artifact, raw_parent=raw_parent, parent=parent, text=text, native=native,
        native_text=native_text, prompt=prompt, suffix=suffix, policy=policy, binding=binding, state=state,
        descriptor=descriptor, policy_artifact=policy_artifact, launch=launch)


def run(case, invoke, **changes):
    options = dict(native_prompt=case.prompt, repository=case.repository, workspace=case.workspace,
        policy=case.policy, retrieval_query_binding=case.binding, provider="codex_cli", model="pinned",
        count_tokens=counter(), invoke=invoke, max_calls=3)
    options.update(changes)
    return dialogue.run_semantic_decision_dialogue(**options)


def response(case, call, *, expand=False, fallback=False):
    value = {"schema": dialogue.REQUEST_SCHEMA if expand else dialogue.FINAL_SCHEMA,
             "session_id": call.session_id, "generation": call.generation, "binding_sha256": call.binding_sha256}
    value.update(capsule_ids=[OPTIONAL]) if expand else value.update(text="Literal final café.  \r\n")
    return dialogue.SemanticDecisionReply("Native final literal.  \r\n" if fallback else encoded(value),
                                         _provider_receipt(case, call))


def retained_result(case, *, fallback=False, expand=False):
    calls = []
    def invoke(call):
        calls.append(call)
        return response(case, call, fallback=fallback, expand=expand and len(calls) == 1)
    result = run(case, invoke, count_tokens=counter(fallback=fallback))
    receipt = result.receipt
    assert all("rejection_observation" not in row for row in receipt["attempts"])
    receipt["worker_launch"] = copy.deepcopy(case.launch)
    return result, receipt, calls


def test_two_rounds_bind_both_raw_parents_and_replay_full_required_retrieval(retrieval_case):
    case = retrieval_case
    result, receipt, calls = retained_result(case, expand=True)
    assert len(calls) == 2 and result.text == "Literal final café.  \r\n"
    held = receipt["retrieval_query_reference"]
    assert held["binding"] == case.binding.to_dict()
    assert held["native_context_sha256"] == sha(case.native_text)
    for call in calls:
        header = json.loads(call.prompt.split("\n", 1)[0])
        assert header["retrieval_query_reference"] == held
        assert header["parent_sha256"] == sha(case.text)
        assert "original-retrieval-coverage" in call.prompt
        assert "authored-call" in call.prompt and "authored-effect" in call.prompt
        assert case.suffix in call.prompt and "Keep this unrelated literal constraint." in call.prompt
    result = audit.audit_semantic_decision_dialogue(native_prompt=case.prompt, dialogue=receipt,
        repository=case.repository, workspace_root=case.workspace_root)
    assert result["all_recorded_round_inputs_verified"] is True and result["verified_round_count"] == 2
    assert case.state["historical_checks"] > 0
    _assert_observation_only(result)


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("mutation", ["raw_parent", "semantic_source", "native_envelope"])
def test_changed_bound_input_after_reply_refuses_delivery_and_retains_attempt(retrieval_case, fallback, mutation):
    case, calls = retrieval_case, []
    def invoke(call):
        calls.append(call)
        reply = response(case, call, fallback=fallback)
        if mutation == "raw_parent":
            case.raw_parent.write_text("changed raw parent")
        elif mutation == "semantic_source":
            case.state["stale"] = True
        else:
            case.state["native_text"] = encoded({**case.native, "hits": []})
        return reply
    with pytest.raises(dialogue.SemanticDecisionDialogueError) as failed:
        run(case, invoke, count_tokens=counter(fallback=fallback))
    receipt = failed.value.receipt
    assert len(calls) == len(receipt["attempts"]) == 1
    assert receipt["attempts"][0]["provider_receipt_bound"] is True
    assert receipt["attempts"][0]["status"] == "rejected"
    assert receipt["selected_representation"] == ("native" if fallback else "decision_packet")
    assert receipt["attempts"][0]["rejection_observation"] == {
        "schema": "supervisor-semantic-decision-rejection-observation@1",
        "phase": "post_return_current_verification", "reason_code": "current_context_refused",
        "exception_type": "ValueError",
    }


@pytest.mark.parametrize("mutation", ["raw_parent", "native_envelope"])
def test_count_interleaving_revalidates_retrieval_before_any_invocation(retrieval_case, mutation):
    case, calls = retrieval_case, []
    def change(_):
        if mutation == "raw_parent":
            case.raw_parent.write_text("changed during count")
        else:
            case.state["native_text"] = encoded({**case.native, "hits": []})
    with pytest.raises(dialogue.SemanticDecisionDialogueError) as failed:
        run(case, lambda call: calls.append(call), count_tokens=counter(after_count=change))
    assert calls == [] and failed.value.receipt["attempts"] == []


def test_complete_count_includes_required_retrieval_and_round_overhead_before_budget_refusal(retrieval_case):
    case, seen, calls = retrieval_case, [], []
    with pytest.raises(dialogue.SemanticDecisionDialogueError) as failed:
        run(case, lambda call: calls.append(call),
            count_tokens=counter(after_count=seen.append), max_input_tokens=99)
    assert failed.value.receipt["failure_reason"] == "complete_prompt_budget_exceeded"
    assert calls == [] and len(seen) == 3
    assert all("authored-call" in text and "authored-effect" in text and case.suffix in text for text in seen)
    assert any(dialogue.ROUND_SCHEMA in text and "retrieval_query_reference" in text for text in seen)


def test_retrieval_revalidation_after_expansion_prevents_continuation(retrieval_case, monkeypatch):
    case, calls = retrieval_case, []
    original = dialogue.expand_semantic_router_decision_packet
    def expand(**kwargs):
        expanded = original(**kwargs)
        case.raw_parent.write_text("changed while constructing expansion")
        return expanded
    monkeypatch.setattr(dialogue, "expand_semantic_router_decision_packet", expand)
    def invoke(call):
        calls.append(call)
        return response(case, call, expand=True)
    with pytest.raises(dialogue.SemanticDecisionDialogueError) as failed:
        run(case, invoke)
    assert len(calls) == 1 and failed.value.receipt["expansions"] == []


@pytest.mark.parametrize("fallback", [False, True])
def test_historical_replay_retains_original_retrieval_after_current_source_edit(retrieval_case, fallback):
    case = retrieval_case
    _, receipt, _ = retained_result(case, fallback=fallback)
    case.state["stale"] = True
    result = audit.audit_semantic_decision_dialogue(native_prompt=case.prompt, dialogue=receipt,
        repository=case.repository, workspace_root=case.workspace_root)
    assert result["all_recorded_round_inputs_verified"] is True
    assert result["freshness_revalidated"] is False
    _assert_observation_only(result)


@pytest.mark.parametrize("fallback", [False, True])
def test_native_producer_retrieval_and_retained_sources_replay_after_real_source_edit(tmp_path, fallback):
    """Actual native loaders; only provider responses and counts are authored."""
    case = build_retrieval_native_fixture(tmp_path)
    case.workspace_root = tmp_path
    descriptor = {
        "schema": "supervisor-semantic-decision-launch-policy@3", "task_id": case.binding.task_id,
        "parent_artifact": case.semantic_bindings.artifact,
        "parent_sha256": case.semantic_bindings.expected_sha256,
        "policy": case.policy.to_dict(), "retrieval_query_binding": case.binding.to_dict(),
    }
    artifact = tmp_path / "protected-native-policy.json"
    artifact.write_text(encoded(descriptor))
    artifact.chmod(0o444)
    case.launch = {
        "launch_policy_artifact": str(artifact), "launch_policy_sha256": sha(artifact.read_bytes()),
        "original_native_prompt_sha256": sha(case.prompt), "original_native_prompt_bytes": len(case.prompt.encode()),
        "required_suffix_sha256": sha(b""), "required_suffix_bytes": 0,
        "retrieval_query_binding": case.binding.to_dict(),
    }
    _, receipt, calls = retained_result(case, fallback=fallback)
    assert len(calls) == 1
    (case.repository / "mod.py").write_text(case.source + "\n# Authored completed candidate edit.\n")
    result = audit.audit_semantic_decision_dialogue(native_prompt=case.prompt, dialogue=receipt,
        repository=case.repository, workspace_root=tmp_path)
    assert result["all_recorded_round_inputs_verified"] is True
    assert result["freshness_revalidated"] is False
    _assert_observation_only(result)


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("mutation", ["raw_parent", "semantic_source"])
def test_native_retrieval_post_reply_rejection_observes_custody_without_private_reply(tmp_path, fallback, mutation):
    """Real native parents/loaders; only reply and complete-input counter are authored."""
    case = build_retrieval_native_fixture(tmp_path)
    private_marker = "PRIVATE-RETRIEVAL-POST-RETURN-REPLY"
    calls, children = [], []

    def invoke(call):
        calls.append(call)
        child = response(case, call, fallback=fallback)
        children.append(child.receipt)
        if fallback:
            text = private_marker
        else:
            value = json.loads(child.text)
            value["text"] = private_marker
            text = encoded(value)
        if mutation == "raw_parent":
            case.retrieval_artifact.write_bytes(case.retrieval_artifact.read_bytes() + b"\n")
        else:
            (case.repository / "mod.py").write_text(case.source + "\n# " + private_marker + "\n")
        return dialogue.SemanticDecisionReply(text, child.receipt)

    with pytest.raises(dialogue.SemanticDecisionDialogueError) as error:
        run(case, invoke, count_tokens=counter(fallback=fallback))
    receipt = error.value.receipt
    assert len(calls) == len(receipt["attempts"]) == 1
    assert receipt["attempts"][0]["status"] == "rejected"
    assert receipt["selected_representation"] == ("native" if fallback else "decision_packet")
    assert receipt["attempts"][0]["provider_receipt_bound"] is True
    assert receipt["attempts"][0]["provider_receipt"] == children[0]
    assert receipt["provider_dispatches_reported"] == 1 and receipt["missing_provider_receipt_count"] == 0
    assert receipt["local_input_content_token_sum"] == 100
    assert receipt["attempts"][0]["rejection_observation"] == {
        "schema": "supervisor-semantic-decision-rejection-observation@1",
        "phase": "post_return_current_verification", "reason_code": "current_context_refused",
        "exception_type": type(error.value.__cause__).__name__,
    }
    assert private_marker not in encoded(receipt)


def test_native_historical_projection_refuses_new_reference_id_collision_without_changing_legacy(tmp_path):
    case = build_retrieval_native_fixture(tmp_path)
    arguments = dict(native_text=case.native_text, binding=case.binding, semantic_view=case.semantic_view,
        semantic_original_text=case.semantic_original_text, semantic_bindings=case.semantic_bindings,
        policy=case.policy)
    query_view = retrieval.build_retrieval_query_reference(**arguments)
    projected = retrieval.retrieval_query_reference_references(view=query_view, **arguments)
    wire, suffix, positions, references, text, _ = codec._prompt_parts(case.prompt)
    changed = copy.deepcopy(wire)
    unrelated = next(row for row in changed["evidence"] if row["kind"] == "failure-context")
    unrelated["reference_id"] = projected[0].reference_id
    packet_arguments = dict(wire=changed, suffix=suffix, positions=positions, references=references,
        text=text, policy=case.policy, bindings=case.semantic_bindings)
    legacy_prompt, _ = audit._packet(**packet_arguments)
    assert unrelated in codec._prompt_parts(legacy_prompt)[0]["evidence"]
    with pytest.raises(ValueError, match="retrieval_projected_reference_ids_duplicate"):
        audit._packet(**packet_arguments, retrieval_query_binding=case.binding, repository=case.repository)


@pytest.mark.parametrize("mutation", ["raw_parent", "raw_pin", "native_pin", "view_pin", "reference_pin", "launch_pin"])
def test_historical_audit_refuses_changed_independent_parent_or_derived_binding(retrieval_case, mutation):
    case = retrieval_case
    _, receipt, _ = retained_result(case)
    if mutation == "raw_parent":
        case.raw_parent.write_text("untrusted changed historical parent")
    elif mutation == "launch_pin":
        receipt["worker_launch"]["retrieval_query_binding"]["expected_sha256"] = "f" * 64
    elif mutation == "raw_pin":
        receipt["retrieval_query_reference"]["binding"]["expected_sha256"] = "f" * 64
    else:
        key = {"native_pin": "native_context_sha256", "view_pin": "view_sha256",
               "reference_pin": "native_reference_records_sha256"}[mutation]
        receipt["retrieval_query_reference"][key] = "f" * 64
    result = audit.audit_semantic_decision_dialogue(native_prompt=case.prompt, dialogue=receipt,
        repository=case.repository, workspace_root=case.workspace_root)
    assert result["original_native_input_verified"] is False
    assert result["verified_round_count"] == 0
    _assert_observation_only(result)


@pytest.mark.parametrize("mutation", ["missing_binding", "extra_field", "legacy_policy", "binding_task", "launch_binding"])
def test_launch_v3_requires_closed_binding_exact_task_and_task_text_policy(retrieval_case, mutation):
    case = retrieval_case
    descriptor = copy.deepcopy(case.descriptor)
    if mutation == "missing_binding":
        del descriptor["retrieval_query_binding"]
    elif mutation == "extra_field":
        descriptor["caller_authority"] = True
    elif mutation == "legacy_policy":
        descriptor["policy"] = SemanticDecisionPolicy(
            required_capsule_ids=case.policy.required_capsule_ids,
            optional_capsule_ids=case.policy.optional_capsule_ids,
            selected_optional_capsule_ids=case.policy.selected_optional_capsule_ids,
        ).to_dict()
    elif mutation == "binding_task":
        descriptor["retrieval_query_binding"]["task_id"] = "foreign-task"
    else:
        case.launch["retrieval_query_binding"]["expected_sha256"] = "f" * 64
    case.policy_artifact.chmod(0o644)
    case.policy_artifact.write_text(encoded(descriptor))
    case.policy_artifact.chmod(0o444)
    case.launch["launch_policy_sha256"] = sha(case.policy_artifact.read_bytes())
    with pytest.raises(ValueError):
        audit._policy(launch=case.launch, workspace=case.workspace,
                      native_prompt=case.prompt, repository=case.repository)


def test_round_default_bytes_omit_new_binding_and_each_new_pin_changes_round_identity(retrieval_case):
    case = retrieval_case
    _, receipt, _ = retained_result(case)
    policy, bindings, *_ = audit._policy(launch=case.launch, workspace=case.workspace,
        native_prompt=case.prompt, repository=case.repository)
    options = dict(projected="literal projected prompt", session="session", generation=0,
                   bindings=bindings, policy=policy, provider="codex_cli", model="pinned", delivered=[])
    original = dialogue._round_prompt(**options)
    assert original == dialogue._round_prompt(**options, retrieval_query_reference=None)
    assert "retrieval_query_reference" not in json.loads(original[0].split("\n", 1)[0])
    held = receipt["retrieval_query_reference"]
    bound = dialogue._round_prompt(**options, retrieval_query_reference=held)
    assert bound[1] != original[1]
    for name in ("native_context_sha256", "view_sha256", "native_reference_records_sha256"):
        changed = {**held, name: "f" * 64}
        assert dialogue._round_prompt(**options, retrieval_query_reference=changed)[1] != bound[1]


@pytest.mark.parametrize("enabled", [False, True])
def test_provider_helper_forwards_binding_only_when_supplied(retrieval_case, monkeypatch, enabled):
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_provider import run_shared_router_decision_dialogue
    case, captured = retrieval_case, []
    def execute(**kwargs):
        captured.append(kwargs)
        return dialogue.SemanticDecisionResult("fixture result", "{}")
    monkeypatch.setattr(dialogue, "run_semantic_decision_dialogue", execute)
    options = {"retrieval_query_binding": case.binding} if enabled else {}
    run_shared_router_decision_dialogue(native_prompt=case.prompt, repository=case.repository,
        workspace=case.workspace, policy=case.policy, provider="codex_cli", model="pinned",
        prompt_token_counter=counter(), **options)
    assert len(captured) == 1
    if enabled:
        assert captured[0]["retrieval_query_binding"] is case.binding
    else:
        assert "retrieval_query_binding" not in captured[0]


@pytest.mark.parametrize("invalid", ["untyped", "missing_text_policy"])
def test_binding_or_policy_refusal_precedes_automatic_tokenizer_preparation(retrieval_case, monkeypatch, invalid):
    from ipfs_accelerate_py.agent_supervisor.runtime import automatic_prompt_tokenizer
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_provider import run_shared_router_decision_dialogue
    case, prepared = retrieval_case, []
    monkeypatch.setattr(automatic_prompt_tokenizer, "resolve_prompt_tokenizer", lambda **kwargs: prepared.append(kwargs))
    policy = case.policy
    if invalid == "missing_text_policy":
        policy = SemanticDecisionPolicy(
            required_capsule_ids=case.policy.required_capsule_ids,
            optional_capsule_ids=case.policy.optional_capsule_ids,
            selected_optional_capsule_ids=case.policy.selected_optional_capsule_ids,
        )
    with pytest.raises(ValueError, match="trusted retrieval binding"):
        run_shared_router_decision_dialogue(native_prompt=case.prompt, repository=case.repository,
            workspace=case.workspace, policy=policy,
            provider="codex_cli", model="pinned",
            retrieval_query_binding=case.binding.to_dict() if invalid == "untyped" else case.binding)
    assert prepared == []
