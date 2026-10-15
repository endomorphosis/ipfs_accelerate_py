"""Current native optional selection stays separate from required invariants."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.residual_llm_packet import ResidualLlmPacket
from ipfs_accelerate_py.agent_supervisor.runtime import repository_world_residual as focus
from ipfs_accelerate_py.agent_supervisor.runtime.repository_context_compiler import PREFIX
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet import (
    PreparedRepositoryWorldPacket, prepare_repository_world_packet, repository_world_task_binding,
)
from test.api.test_repository_context_compiler import (
    CORE, compile_context, compiler, native_session, obligation, required, scheduler,  # noqa: F401
)
from test.api.test_repository_world_packet import TASK, EFFECTS, PATHS, compile_bound, prepare
from test.api.test_repository_intent_world import intent_case, capture  # noqa: F401
from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from test.api.test_intent_requirement_observation import _start


@pytest.fixture
def baseline_case(native_session, required):
    source, session = native_session
    adapter = session.context_adapter(compiler())
    compilation = compile_bound(adapter, required)
    baseline = prepare(adapter, compilation)
    return dict(source=source, session=session, adapter=adapter, compilation=compilation, baseline=baseline)


def residual(baseline, **changes):
    recipe = json.loads(baseline._replay_material_json)["recipe"]
    capsule = baseline._capsule
    options = dict(task_id=recipe["task_id"], repository_id=capsule.repository_id, tree_id=capsule.tree_id,
        policy_id=capsule.policy_id, policy_revision=capsule.policy_revision,
        write_paths=tuple(recipe["write_paths"]), validation_commands=tuple(recipe["validation_commands"]),
        acceptance_ids=tuple(recipe["acceptance_ids"]), obligation_ids=("arithmetic:remaining-check",),
        counterexample_capsule={"target_ids": [recipe["task_id"]], "counterexamples": [{
            "counterexample_id": "authored-counterexample", "summary": "The declared arithmetic result remains unresolved.",
            "expected": 3, "actual": 2}]})
    options.update(changes)
    return ResidualLlmPacket(**options)


def request(case, **changes):
    options = dict(baseline=case["baseline"], query="add",
        unresolved_question="Which definition implements the remaining arithmetic obligation?", limit=1)
    options.update(changes)
    if "residual_packet" not in options:
        options["residual_packet"] = residual(case["baseline"])
    return focus.prepare_repository_world_residual_request(**options)


def evidence(packet):
    admitted = json.loads(packet.prompt.split("\n", 1)[1])["packet"]
    return {item["payload"]["evidence_id"]: item["payload"]
        for item in admitted["members"] if item["kind"] == "evidence"}


def test_actual_query_focus_preserves_full_task_required_references_and_invariants(baseline_case):
    case = baseline_case
    baseline = case["baseline"]
    result = request(case)
    candidate = result.packet
    assert type(candidate) is PreparedRepositoryWorldPacket
    before, after = baseline._capsule, candidate._capsule
    expected = {item.reference_id: item.canonical_bytes() for item in before.evidence if item.required}
    actual = {item.reference_id: item.canonical_bytes() for item in after.evidence if item.required}
    assert expected == actual
    for name in ("repository_id", "tree_id", "objective_id", "objective_revision", "policy_id", "policy_revision",
                 "goal", "scope", "acceptance"):
        assert getattr(before, name) == getattr(after, name)
    assert before.authority["task"] == after.authority["task"]
    assert before.authority["repository_context"]["query"] != after.authority["repository_context"]["query"]
    before_recipe = json.loads(baseline._replay_material_json)["recipe"]
    after_recipe = json.loads(candidate._replay_material_json)["recipe"]
    for key, value in before_recipe.items():
        assert after_recipe[key] == value
    assert before_recipe["current_world"] is after_recipe["current_world"] is None
    assert evidence(candidate)["task-text"]["task_text"] == TASK
    assert residual(baseline).packet_id in candidate.prompt
    observed = result.receipt()
    assert observed["required_references_preserved"] is observed["full_task_invariants_preserved"] is True
    assert observed["baseline_optional_selected_count"] == 2
    assert observed["candidate_optional_selected_count"] == 1
    assert observed["omitted_baseline_native_optional_count"] == 1
    assert observed["world_projection_status"] == "unavailable"
    assert observed["world_authority_created"] is False
    assert observed["optional_omission_equivalence_verified"] is observed["resolved_work_claimed"] is False
    assert observed["provider_calls"] == 0 and observed["provider_input_tokens"] is None
    assert observed["token_savings_established"] is observed["default_adoption_authorized"] is False
    assert observed["baseline_prompt_bytes"] == len(baseline.prompt.encode())
    assert observed["candidate_prompt_bytes"] == len(candidate.prompt.encode())
    PreparedRepositoryWorldPacket.validate_current(candidate, workspace=case["source"], task_text=TASK)


def test_receipt_is_detached_content_free_and_wrapper_cannot_mutate_packet(baseline_case):
    result = request(baseline_case)
    observed = result.receipt()
    observed["source_binding"]["generation"] = 99
    assert result.receipt()["source_binding"]["generation"] == 1
    text = json.dumps(result.receipt())
    assert str(baseline_case["source"]) not in text and TASK.strip() not in text
    assert "Preserve intent_ir" not in text and "declared arithmetic result" not in text
    with pytest.raises(FrozenInstanceError):
        result.packet = baseline_case["baseline"]


def test_optional_omission_population_matches_actual_baseline_reference_ids(baseline_case):
    baseline = baseline_case["baseline"]
    result = request(baseline_case)
    before = {item.reference_id: item for item in baseline._capsule.evidence if not item.required}
    after = {item.reference_id for item in result.packet._capsule.evidence if not item.required}
    omitted = {identity: reference for identity, reference in before.items() if identity not in after}
    assert len(omitted) == 1 and all(identity.startswith(PREFIX) for identity in omitted)
    observed = result.receipt()["omitted_baseline_optional"]
    assert observed == [dict(reference_id_sha256=hashlib.sha256(identity.encode()).hexdigest(),
        reference_content_id=reference.content_id) for identity, reference in sorted(omitted.items())]
    assert result.receipt()["unqueried_native_reference_ids_exported"] is False


@pytest.mark.parametrize("field,value", [("task_id", "another-task"), ("repository_id", "another-repository"),
    ("tree_id", "another-tree"), ("policy_id", "another-policy"), ("policy_revision", "another-revision"),
    ("write_paths", ("notes.txt",)), ("validation_commands", ("python -c 'pass'",)),
    ("acceptance_ids", ("unbound-acceptance",))])
def test_foreign_residual_refuses_before_new_optional_compilation(baseline_case, monkeypatch, field, value):
    packet = residual(baseline_case["baseline"], **{field: value})
    monkeypatch.setattr(baseline_case["adapter"], "compile", lambda **kwargs: pytest.fail("new optional query attempted"))
    with pytest.raises(ValueError):
        request(baseline_case, residual_packet=packet)


@pytest.mark.parametrize("change", ["replacement", "prompt", "recipe", "method"])
def test_replaced_or_shadowed_baseline_cannot_authorize_focus(baseline_case, change):
    baseline = baseline_case["baseline"]
    captured_residual = residual(baseline)
    if change == "replacement":
        baseline = replace(baseline)
    elif change == "prompt":
        object.__setattr__(baseline, "prompt", baseline.prompt + " changed")
    elif change == "recipe":
        object.__setattr__(baseline, "_replay_material_json", "changed")
    else:
        object.__setattr__(baseline, "validate_current", lambda **kwargs: {})
    with pytest.raises(ValueError, match="custody|shadowed"):
        request(baseline_case, baseline=baseline, residual_packet=captured_residual)


@pytest.mark.parametrize("mutation", ["before", "during_compile"])
def test_source_drift_refuses_before_returning_candidate(baseline_case, monkeypatch, mutation):
    case = baseline_case
    original = case["adapter"].compile

    def edit():
        (case["source"] / "calc.py").write_text("def add(value):\n    return value + 2\n")

    if mutation == "before":
        edit()
    else:
        def compile_and_drift(**kwargs):
            value = original(**kwargs)
            edit()
            return value
        monkeypatch.setattr(case["adapter"], "compile", compile_and_drift)
    with pytest.raises(ValueError):
        request(case)


@pytest.mark.parametrize("dimension", ["goal", "required"])
def test_compiler_cannot_change_required_business_content(baseline_case, monkeypatch, dimension):
    case = baseline_case
    original = case["adapter"].compile

    def changed(**kwargs):
        if dimension == "goal":
            kwargs["goal"] = {"intent": "Unreviewed replacement goal"}
        else:
            required = list(kwargs["required_evidence"])
            first = required[0]
            required[0] = replace(first, summary="Changed mandatory evidence", byte_count=0)
            kwargs["required_evidence"] = tuple(required)
        return original(**kwargs)

    monkeypatch.setattr(case["adapter"], "compile", changed)
    with pytest.raises(ValueError, match="required_reference_changed|task_invariants_changed"):
        request(case)


def test_repeated_residual_lineage_requires_fresh_review(baseline_case):
    first = request(baseline_case)
    with pytest.raises(ValueError, match="already_residual"):
        request(baseline_case, baseline=first.packet, residual_packet=residual(first.packet))


def test_readonly_baseline_cannot_be_upgraded_to_residual_dispatch(baseline_case, required, monkeypatch):
    case = baseline_case
    effects = ("repository_read",)
    compiled = compile_context(case["adapter"], required,
        authority=dict(CORE["authority"], world_task_packet=repository_world_task_binding(
            "task:arithmetic", TASK, effects, PATHS)))
    readonly = prepare_repository_world_packet(adapter=case["adapter"], compilation=compiled, generation=1,
        task_id="task:arithmetic", task_text=TASK, unresolved_obligations=("Observe the current arithmetic.",),
        validation_commands=(), allowed_effects=effects, write_paths=PATHS, validation_profile="read_only_observation@1")
    monkeypatch.setattr(case["adapter"], "compile", lambda **kwargs: pytest.fail("optional compilation attempted"))
    with pytest.raises(ValueError, match="readonly_provider_dispatch_refused"):
        request(case, baseline=readonly)


def test_source_generation_refresh_does_not_revive_old_baseline(baseline_case):
    case = baseline_case
    case["session"].refresh(generation=1)
    with pytest.raises(ValueError):
        request(case)


@pytest.mark.parametrize("options", [{"query": ""}, {"unresolved_question": ""}, {"limit": True},
    {"limit": 33}, {"max_bytes": 1023}, {"max_bytes": True}, {"residual_packet": {}}, {"baseline": {}}])
def test_invalid_focus_arguments_refuse_before_owner_query(baseline_case, monkeypatch, options):
    monkeypatch.setattr(baseline_case["adapter"], "compile", lambda **kwargs: pytest.fail("compile attempted"))
    with pytest.raises(ValueError):
        request(baseline_case, **options)


def test_original_nested_packet_mutation_cannot_change_captured_request(baseline_case, monkeypatch):
    case = baseline_case
    packet = residual(case["baseline"])
    expected = packet.packet_id
    original = case["adapter"].compile

    def mutate_after_capture(**kwargs):
        value = original(**kwargs)
        packet.counterexample_capsule["counterexamples"][0]["expected"] = 999
        return value

    monkeypatch.setattr(case["adapter"], "compile", mutate_after_capture)
    result = request(case, residual_packet=packet)
    assert result.receipt()["residual_packet_id"] == expected
    assert '"expected":3' in result.packet.prompt
    assert '"expected":999' not in result.packet.prompt


def native_intent_baseline(case):
    observed = capture(case)
    material = json.loads(observed.reference.summary)
    spec = material["task_spec"]
    task_text = spec["objective"]
    effects, paths = ("write_source", "run_validation"), ("answer.py",)
    adapter = case["session"].context_adapter(compiler(tokens=100000))
    compiled = compile_context(adapter, (observed.reference,), query="answer", goal={"intent": task_text},
        scope={"paths": spec["scope_paths"]}, acceptance={"criteria": spec["acceptance"]},
        authority=dict(CORE["authority"], world_task_packet=repository_world_task_binding(
            case["task_id"], task_text, effects, paths)))
    baseline = prepare_repository_world_packet(adapter=adapter, compilation=compiled, generation=1,
        task_id=case["task_id"], task_text=task_text,
        unresolved_obligations=("Retain the exact admitted task and its native public checks.",),
        validation_commands=tuple(json.dumps(row, sort_keys=True, separators=(",", ":"))
            for row in spec["validations"]), allowed_effects=effects, write_paths=paths,
        intent_world_observation=observed, max_bytes=262144)
    return dict(baseline=baseline, adapter=adapter, observed=observed, task_text=task_text)


def test_current_native_intent_observation_remains_required_and_unavailable_world_is_explicit(intent_case):
    case = native_intent_baseline(intent_case)
    result = request(case, query="answer")
    candidate = result.packet
    assert candidate._intent_world_observation is case["observed"]
    actual = [reference for reference in candidate._capsule.evidence if reference.kind == "repository-intent-world"]
    assert len(actual) == 1 and actual[0].required
    assert actual[0].canonical_bytes() == case["observed"].reference.canonical_bytes()
    value = json.loads(actual[0].summary)
    assert value["intent_world"]["repository_world_projection"] == "unavailable"
    assert result.receipt()["world_projection_status"] == "unavailable"
    assert result.receipt()["world_authority_created"] is False


@pytest.mark.parametrize("when", ["before", "during_compile"])
def test_native_intent_event_drift_refuses_focused_candidate(intent_case, monkeypatch, when):
    case = native_intent_baseline(intent_case)
    original = case["adapter"].compile

    def changed():
        _start(intent_case["case"], intent_case["task_id"])

    if when == "before":
        changed()
    else:
        def compile_and_change(**kwargs):
            value = original(**kwargs)
            changed()
            return value
        monkeypatch.setattr(case["adapter"], "compile", compile_and_change)
    with pytest.raises(ValueError):
        request(case, query="answer")
