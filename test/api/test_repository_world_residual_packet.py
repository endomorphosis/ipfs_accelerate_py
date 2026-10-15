"""Residual nominations augment the complete source-bound worker request.

Native source fixtures and the existing authored provider boundary exercise
custody and preservation. No live model result or token saving is claimed.
"""
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.residual_llm_packet import seal_residual_llm_packet
from ipfs_accelerate_py.agent_supervisor.runtime import repository_world_packet as owner
from ipfs_accelerate_py.agent_supervisor.runtime import repository_world_packet_replay as replay
from ipfs_accelerate_py.agent_supervisor.runtime.repository_residual_context import (
    prepare_repository_residual_context,
)
from test.api.test_repository_context_compiler import CORE, compile_context
from test.api.test_repository_world_packet import (
    TASK, PATHS, compiled, evidence, native_session, payload, prepare, required, scheduler,
)
from test.api.test_repository_world_dispatch import (
    TASK as DISPATCH_TASK, dispatch_case, invoke, world_session,
)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha(value):
    return hashlib.sha256(value).hexdigest()


def seal(compilation, **changes):
    capsule = compilation.capsule
    options = dict(task_id="task:arithmetic", repository_id=capsule.repository_id,
        tree_id=capsule.tree_id, policy_id=capsule.policy_id,
        policy_revision=capsule.policy_revision, write_paths=PATHS,
        obligation_ids=("residual:increment-counterexample",),
        counterexample_capsule=dict(summary="A bounded arithmetic counterexample remains.",
            target_ids=["symbol:calc.add"], case_id="case:increment", actual=2, expected=3),
        validation_commands=("python -m pytest tests/test_calc.py",),
        acceptance_ids=("acceptance:compatible-callers",),
        forest_id="forest:authored-residual", authority_roots={"policy_root": "policy:authored"},
        codex_packet_ref="sha256:authored-codex", transition_ref="sha256:authored-transition")
    options.update(changes)
    return seal_residual_llm_packet(**options)


def binding(compilation, residual, **changes):
    capsule = compilation.capsule
    options = dict(residual_packet=residual, expected_task_id="task:arithmetic",
        repository_id=capsule.repository_id, tree_id=capsule.tree_id,
        policy_id=capsule.policy_id, policy_revision=capsule.policy_revision,
        write_paths=PATHS, validation_commands=("python -m pytest tests/test_calc.py",),
        acceptance_ids=("acceptance:compatible-callers",))
    options.update(changes)
    return prepare_repository_residual_context(**options)


def restore(data):
    return replay.replay_repository_world_packet(data, expected_sha256=sha(data))


@pytest.fixture
def focused(compiled):
    source, session, adapter, compilation = compiled
    residual = seal(compilation)
    packet = prepare(adapter, compilation, residual_packet=residual)
    return source, session, adapter, compilation, residual, packet


def test_explicit_none_keeps_legacy_prompt_receipt_and_artifact_bytes(compiled):
    _source, _session, adapter, compilation = compiled
    ordinary = prepare(adapter, compilation)
    explicit = prepare(adapter, compilation, residual_packet=None)
    assert ordinary.prompt == explicit.prompt
    assert ordinary.receipt() == explicit.receipt()
    assert replay.freeze_repository_world_packet(ordinary) == replay.freeze_repository_world_packet(explicit)
    assert "residual_context" not in ordinary.receipt()
    assert "residual_context" not in json.loads(replay.freeze_repository_world_packet(ordinary))["recipe"]


def test_residual_is_one_existing_context_and_preserves_complete_native_request(focused):
    source, _session, adapter, compilation, residual, packet = focused
    ordinary = prepare(adapter, compilation)
    expected = binding(compilation, residual)
    assert evidence(packet) == evidence(ordinary)
    assert evidence(packet)["task-text"]["task_text"] == TASK
    for reference in compilation.capsule.evidence:
        assert evidence(packet)["context-reference:" + reference.reference_id]["reference"] == reference.to_record()
    for key in ("effect_scope", "requirements"):
        assert payload(packet)[key] == payload(ordinary)[key]
    original_obligations = [row["payload"] for row in payload(ordinary)["members"] if row["kind"] == "obligation"]
    assert [row["payload"] for row in payload(packet)["members"] if row["kind"] == "obligation"] == original_obligations
    failures = [row["payload"] for row in payload(packet)["members"] if row["kind"] == "failure"]
    assert failures == [dict(failure_id=residual.packet_id,
        summary="Residual nomination context for the current task.", residual_context=expected.context)]
    assert not [row for row in payload(packet)["members"] if row["kind"] == "counterexample"]
    assert "residual_packet" not in failures[0]
    assert "packet_version" not in failures[0]["residual_context"]
    assert failures[0]["residual_context"]["counterexample_capsule"] == dict(residual.counterexample_capsule)
    assert packet.receipt()["residual_context"] == expected.receipt
    assert packet.receipt()["packet_id"] != residual.packet_id
    assert packet.receipt()["delivery_mode"] == "self_contained"
    assert packet.receipt()["provider_retained_parent"] is False
    assert packet.receipt()["provider_input_tokens"] is None
    for key in ("omission_authority", "proof_authority", "write_authority", "completion_authority",
                "source_semantics_verified", "world_proof_verified"):
        assert packet.receipt()["residual_context"][key] is False
    diagnostic = json.dumps(packet.receipt())
    assert TASK.strip() not in diagnostic
    assert "bounded arithmetic counterexample" not in diagnostic
    assert str(source) not in diagnostic
    packet.require_provider_dispatch()
    packet.validate_current(workspace=source, task_text=TASK)


@pytest.mark.parametrize("field,value", [
    ("task_id", "task:other"), ("repository_id", "repository:other"),
    ("tree_id", "tree:other"), ("policy_id", "policy:other"),
    ("policy_revision", "revision:other"), ("write_paths", ("other.py",)),
    ("validation_commands", ("python -m pytest tests/other.py",)),
    ("acceptance_ids", ("acceptance:other",)),
])
def test_residual_cannot_change_independently_bound_native_scope(compiled, field, value):
    _source, _session, adapter, compilation = compiled
    with pytest.raises(ValueError, match="residual_.*mismatch"):
        prepare(adapter, compilation, residual_packet=seal(compilation, **{field: value}))


@pytest.mark.parametrize("kind", ["mapping", "string", "false"])
def test_public_preparation_requires_exact_residual_packet_type(compiled, kind):
    _source, _session, adapter, compilation = compiled
    residual = seal(compilation)
    value = {"mapping": residual.to_dict(), "string": residual.packet_id, "false": False}[kind]
    with pytest.raises(ValueError, match="exact_residual_required"):
        prepare(adapter, compilation, residual_packet=value)


def test_readonly_residual_refuses_before_compiler_work(compiled, monkeypatch):
    _source, _session, adapter, compilation = compiled
    residual = seal(compilation)
    monkeypatch.setattr(adapter.compiler, "verify_compile_result", lambda *args:
        pytest.fail("read-only residual reached compiler work"))
    with pytest.raises(ValueError, match="readonly_residual_refused"):
        prepare(adapter, compilation, residual_packet=residual,
                validation_profile="read_only_observation@1")


def test_noise_normalization_cannot_silently_drop_residual_content(compiled):
    _source, _session, adapter, compilation = compiled
    residual = seal(compilation, counterexample_capsule={
        "summary": "The timestamp is required counterexample data.",
        "target_ids": ["symbol:calc.add"], "timestamp": "authored-value"})
    with pytest.raises(ValueError, match="residual_context_transformed"):
        prepare(adapter, compilation, residual_packet=residual)


def test_residual_cannot_force_budget_omission_of_original_evidence(focused):
    _source, _session, adapter, compilation, residual, _packet = focused
    ordinary = prepare(adapter, compilation)
    with pytest.raises(ValueError, match="overflow|omitted|budget"):
        prepare(adapter, compilation, residual_packet=residual,
                max_bytes=ordinary.receipt()["prompt_bytes"])


def test_captured_residual_does_not_reopen_mutable_original(focused):
    source, _session, _adapter, _compilation, residual, packet = focused
    raw = replay.freeze_repository_world_packet(packet)
    residual.counterexample_capsule["summary"] = "Changed original after capture."
    packet.validate_current(workspace=source, task_text=TASK)
    assert replay.freeze_repository_world_packet(packet) == raw
    assert restore(raw).prompt == packet.prompt


def test_historical_record_retains_full_packet_and_independent_pins(focused):
    _source, _session, _adapter, compilation, residual, packet = focused
    raw = replay.freeze_repository_world_packet(packet)
    lineage = json.loads(raw)["recipe"]["residual_context"]
    expected = binding(compilation, residual)
    assert lineage["record"] == expected.record
    assert lineage["record"]["packet"] == residual.to_dict()
    assert lineage["record_sha256"] == sha(encoded(expected.record))
    assert lineage["packet_id"] == residual.packet_id
    assert lineage["context_id"] == expected.context_id
    assert lineage["packet_sha256"] == sha(encoded(expected.packet))
    assert lineage["context_sha256"] == sha(encoded(expected.context))
    restored = restore(raw)
    assert restored.prompt == packet.prompt
    assert restored.recorded_packet_receipt() == packet.receipt()
    assert restored.receipt()["residual_context"] == expected.receipt
    assert restored.receipt()["provider_dispatch_verified"] is False
    assert restored.receipt()["live_source_freshness_checked"] is False
    assert restored.receipt()["completion_authority"] is False
    with pytest.raises(ValueError, match="live_packet_required"):
        replay.freeze_repository_world_packet(restored)


@pytest.mark.parametrize("key", ["record_sha256", "packet_id", "context_id", "packet_sha256", "context_sha256"])
def test_replay_refuses_independently_wrong_residual_pin_after_outer_repin(focused, key):
    value = json.loads(replay.freeze_repository_world_packet(focused[-1]))
    value["recipe"]["residual_context"][key] = "0" * 64
    with pytest.raises(ValueError, match="residual"):
        restore(encoded(value))


@pytest.mark.parametrize("key", ["record", "record_sha256", "packet_id", "context_id", "packet_sha256", "context_sha256"])
def test_replay_requires_complete_residual_lineage_after_outer_repin(focused, key):
    value = json.loads(replay.freeze_repository_world_packet(focused[-1]))
    del value["recipe"]["residual_context"][key]
    with pytest.raises(ValueError, match="residual"):
        restore(encoded(value))


@pytest.mark.parametrize("kind", ["missing_recipe", "null_recipe", "extra_lineage", "missing_receipt",
                                  "changed_context", "changed_packet", "changed_member"])
def test_replay_cannot_hide_or_replace_residual_material_after_outer_repin(focused, kind):
    value = json.loads(replay.freeze_repository_world_packet(focused[-1]))
    lineage = value["recipe"]["residual_context"]
    if kind == "missing_recipe":
        del value["recipe"]["residual_context"]
    elif kind == "null_recipe":
        value["recipe"]["residual_context"] = None
    elif kind == "extra_lineage":
        lineage["unreviewed"] = True
    elif kind == "missing_receipt":
        del value["packet_receipt"]["residual_context"]
    elif kind == "changed_context":
        lineage["record"]["context"]["counterexample_capsule"]["summary"] = "Different nomination."
        lineage["record_sha256"] = sha(encoded(lineage["record"]))
    elif kind == "changed_packet":
        lineage["record"]["packet"]["forest_id"] = "forest:other"
        lineage["record_sha256"] = sha(encoded(lineage["record"]))
    else:
        failure = next(row for row in value["native_packet"]["unresolved_delta"] if row["kind"] == "failure")
        failure["payload"]["residual_context"]["task_id"] = "task:other"
    with pytest.raises(ValueError):
        restore(encoded(value))


def test_residual_does_not_weaken_source_freshness_or_historical_separation(focused, monkeypatch):
    source, _session, adapter, _compilation, _residual, packet = focused
    raw = replay.freeze_repository_world_packet(packet)
    (source / "calc.py").write_text("def add(value):\n    return value + 99\n")
    with pytest.raises(ValueError):
        packet.validate_current(workspace=source, task_text=TASK)
    with pytest.raises(ValueError):
        replay.freeze_repository_world_packet(packet)
    monkeypatch.setattr(adapter, "validate_current", lambda *args, **kwargs:
        pytest.fail("historical residual replay reopened the live owner"))
    assert restore(raw).prompt == packet.prompt


def test_focused_packet_uses_current_runner_and_its_fresh_invocation(dispatch_case, required):
    _source, _session, original, state = dispatch_case
    recipe = json.loads(original._replay_material_json)["recipe"]
    adapter = original._adapter
    compilation = compile_context(adapter, required, authority=dict(CORE["authority"],
        world_task_packet=owner.repository_world_task_binding(recipe["task_id"], DISPATCH_TASK,
            tuple(recipe["allowed_effects"]), tuple(recipe["write_paths"]))))
    residual = seal(compilation, validation_commands=tuple(recipe["validation_commands"]), acceptance_ids=())
    options = dict(recipe)
    for key in ("unresolved_obligations", "validation_commands", "proof_obligations",
                "acceptance_ids", "allowed_effects", "write_paths"):
        options[key] = tuple(options[key])
    packet = owner.prepare_repository_world_packet(adapter=adapter, compilation=compilation,
        residual_packet=residual, **options)
    output, receipt = invoke(dispatch_case, world_task_packet=packet)
    assert output
    assert len(state["seen"]) == 1
    model_prompt, provider_options = state["seen"][0]
    assert packet.prompt in model_prompt
    assert model_prompt.count('"residual_context":') == 1
    assert provider_options["allocation_session_id"] == receipt["invocation_id"]
    assert "resume_session_id" not in provider_options
    assert receipt["world_task_packet"]["dispatch_rechecked"] is True
    assert receipt["world_task_packet"]["provider_retained_parent"] is False
    assert receipt["world_task_packet"]["residual_context"] == packet.receipt()["residual_context"]
