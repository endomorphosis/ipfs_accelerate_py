"""Current repository context reaches a bounded standalone packet unchanged."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import ArtifactBindings
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.world_model import (
    RepositoryWorldState, WorldProjectionStatus,
)
from ipfs_accelerate_py.agent_supervisor.runtime.repository_context_compiler import RepositoryContextAdapter
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet import (
    RepositoryWorldPacketError, prepare_repository_world_packet, repository_world_task_binding,
)
from test.api.test_repository_context_compiler import (
    CORE, compiler, compile_context, native_session, required, scheduler,
)


TASK = "  Add bounded arithmetic without changing callers.\n"
EFFECTS = ("inspect_repository", "edit_source", "run_tests")
PATHS = ("calc.py",)


def compile_bound(adapter, required, *, task_text=TASK, **changes):
    arguments = dict(authority=dict(CORE["authority"], world_task_packet=repository_world_task_binding(
        "task:arithmetic", task_text, EFFECTS, PATHS)))
    arguments.update(changes)
    return compile_context(adapter, required, **arguments)


def prepare(adapter, compilation, **changes):
    arguments = dict(adapter=adapter, compilation=compilation, generation=1,
        task_id="task:arithmetic", task_text=TASK,
        unresolved_obligations=("Update arithmetic while retaining validation.",),
        validation_commands=("python -m pytest tests/test_calc.py",),
        proof_obligations=("property:bounded-result",),
        acceptance_ids=("acceptance:compatible-callers",),
        allowed_effects=EFFECTS, write_paths=PATHS)
    arguments.update(changes)
    return prepare_repository_world_packet(**arguments)


def payload(prepared):
    return json.loads(prepared.prompt.split("\n", 1)[1])["packet"]


def evidence(prepared):
    return {item["payload"]["evidence_id"]: item["payload"]
            for item in payload(prepared)["members"] if item["kind"] == "evidence"}


@pytest.fixture
def compiled(native_session, required):
    source, session = native_session
    adapter = session.context_adapter(compiler())
    return source, session, adapter, compile_bound(adapter, required)


def world_for(compilation, **changes):
    capsule = compilation.capsule
    bindings = ArtifactBindings(repository_id=capsule.repository_id,
        repository_commit="sha256:authored-commit", tree_id=capsule.tree_id,
        objective_id=capsule.objective_id, task_id="task:arithmetic",
        contract_revision="sha256:authored-contract", policy_revision=capsule.policy_revision,
        environment_id="sha256:authored-environment")
    value = RepositoryWorldState(bindings=bindings,
        world_snapshot_cid="sha256:authored-world-snapshot",
        repository_reference="sha256:authored-repository",
        repository_snapshot_id="sha256:authored-repository-snapshot",
        analysis_head_tree_id="sha256:authored-head", analysis_index_tree_id="sha256:authored-index",
        unavailable_dimensions=("test_status", "proof_status"),
        projection_status=WorldProjectionStatus.INCOMPLETE)
    return replace(value, **changes)


def test_current_native_context_and_exact_task_are_delivered_once(compiled, required):
    source, _session, adapter, compilation = compiled
    prepared = prepare(adapter, compilation)
    delivered = evidence(prepared)
    assert delivered["task-text"]["task_text"] == TASK
    core = delivered["compiled-core"]["context"]
    for name in ("goal", "scope", "acceptance", "authority"):
        assert core[name] == json.loads(json.dumps(compilation.capsule.to_dict()[name]))
    for reference in compilation.capsule.evidence:
        assert delivered["context-reference:" + reference.reference_id]["reference"] == reference.to_record()
    assert {item["reference"]["reference_id"] for item in delivered.values()
            if "reference" in item} == {item.reference_id for item in compilation.capsule.evidence}
    assert delivered["world-availability"]["world"]["status"] == "unavailable"
    assert payload(prepared)["frontier"]["has_more"] is False
    assert payload(prepared)["requirements"]["proof_obligations"] == ["property:bounded-result"]
    assert len([item for item in payload(prepared)["members"] if item["kind"] == "obligation"]) == 1
    receipt = prepared.validate_current(workspace=source, task_text=TASK)
    assert receipt["provider_retained_parent"] is False
    assert receipt["delivery_mode"] == "self_contained"
    assert receipt["provider_input_tokens"] is None
    assert receipt["packet_item_token_estimate"] > 0
    assert receipt["prompt_bytes"] == len(prepared.prompt.encode())
    assert receipt["prompt_sha256"] == hashlib.sha256(prepared.prompt.encode()).hexdigest()
    assert receipt["required_reference_count"] == len(required)
    assert str(source) not in json.dumps(receipt)
    assert TASK.strip() not in json.dumps(receipt)
    assert required[0].summary not in json.dumps(receipt)


def test_large_exact_task_retains_whitespace_and_unicode(compiled, required):
    source, _session, adapter, compilation = compiled
    task = " \n" + "Inspect arithmetic π with caller compatibility.\n" * 120 + "  "
    compilation = compile_bound(adapter, required, task_text=task)
    prepared = prepare(adapter, compilation, task_text=task)
    assert evidence(prepared)["task-text"]["task_text"] == task
    prepared.validate_current(workspace=source, task_text=task)


def test_prepared_prompt_and_receipt_are_immutable(compiled):
    source, _session, adapter, compilation = compiled
    prepared = prepare(adapter, compilation)
    with pytest.raises(FrozenInstanceError):
        prepared.prompt = "changed"
    copy = prepared.receipt()
    copy["source_binding"]["generation"] = 99
    assert prepared.receipt()["source_binding"]["generation"] == 1
    forged = replace(prepared, prompt=prepared.prompt + "\nextra")
    with pytest.raises(RepositoryWorldPacketError, match="not_issued"):
        forged.validate_current(workspace=source, task_text=TASK)
    forged_receipt = prepared.receipt()
    forged_receipt["task_text_sha256"] = hashlib.sha256(b"another task").hexdigest()
    forged = replace(prepared, _receipt_json=json.dumps(forged_receipt))
    with pytest.raises(RepositoryWorldPacketError, match="not_issued"):
        forged.validate_current(workspace=source, task_text="another task")


@pytest.mark.parametrize("change", ["task", "workspace", "relative_workspace"])
def test_consumer_boundary_refuses_different_task_or_workspace(compiled, change):
    source, _session, adapter, compilation = compiled
    prepared = prepare(adapter, compilation)
    workspace = source.parent if change == "workspace" else source
    if change == "relative_workspace":
        workspace = source.relative_to(source.anchor)
    task = TASK.strip() if change == "task" else TASK
    with pytest.raises(RepositoryWorldPacketError, match="mismatch"):
        prepared.validate_current(workspace=workspace, task_text=task)


def test_direct_source_edit_invalidates_prepared_dispatch(compiled):
    source, _session, adapter, compilation = compiled
    prepared = prepare(adapter, compilation)
    (source / "calc.py").write_text("def add(value):\n    return value + 2\n")
    with pytest.raises(ValueError, match="source|changed|refresh"):
        prepared.validate_current(workspace=source, task_text=TASK)
    with pytest.raises(ValueError, match="source|changed|refresh"):
        prepare(adapter, compilation)


def test_generation_refresh_invalidates_prepared_dispatch(compiled):
    source, session, adapter, compilation = compiled
    prepared = prepare(adapter, compilation)
    session.refresh(generation=1)
    with pytest.raises(ValueError, match="generation|source|changed"):
        prepared.validate_current(workspace=source, task_text=TASK)


def test_context_must_be_issued_by_exact_adapter(compiled):
    _source, session, _adapter, compilation = compiled
    other = RepositoryContextAdapter(session, compiler())
    with pytest.raises(ValueError, match="unknown_context_parent"):
        prepare(other, compilation)


@pytest.mark.parametrize("field,value", [
    ("unresolved_obligations", ()), ("validation_commands", ()),
    ("allowed_effects", ()), ("unresolved_obligations", ("duplicate", "duplicate")),
    ("write_paths", ()),
    ("unresolved_obligations", ["not a tuple"]),
    ("task_text", "   "), ("max_bytes", True),
])
def test_missing_or_malformed_task_requirements_refuse(compiled, field, value):
    _source, _session, adapter, compilation = compiled
    with pytest.raises(RepositoryWorldPacketError):
        prepare(adapter, compilation, **{field: value})


def test_packet_budget_cannot_drop_selected_evidence(compiled):
    _source, _session, adapter, compilation = compiled
    with pytest.raises(ValueError, match="budget|overflow|omitted"):
        prepare(adapter, compilation, max_bytes=1024)


def test_optional_delta_item_omission_refuses_instead_of_delivering_a_frontier(compiled):
    _source, _session, adapter, compilation = compiled
    with pytest.raises(RepositoryWorldPacketError, match="world_packet_omitted_context"):
        prepare(adapter, compilation, max_bytes=5000)


def test_final_wire_envelope_is_independently_bounded(compiled):
    _source, _session, adapter, compilation = compiled
    prepared = prepare(adapter, compilation)
    with pytest.raises(ValueError, match="budget|overflow|omitted"):
        prepare(adapter, compilation, max_bytes=prepared.receipt()["prompt_bytes"] - 1)


def test_scope_change_cannot_silently_broaden_or_change_input(compiled):
    _source, _session, adapter, compilation = compiled
    with pytest.raises(RepositoryWorldPacketError, match="task_binding_mismatch"):
        prepare(adapter, compilation, write_paths=("./calc.py",))


def test_existing_compiler_omissions_stay_supervisor_side(native_session, required):
    _source, session = native_session
    adapter = session.context_adapter(compiler(optional_items=1))
    compilation = compile_bound(adapter, required)
    assert compilation.capsule.expansion_references
    prepared = prepare(adapter, compilation)
    selected = evidence(prepared)
    for omitted in compilation.capsule.expansion_references:
        assert "context-reference:" + omitted.reference_id not in selected
    assert prepared.receipt()["selected_reference_count"] == len(required) + 1


def test_supplied_world_preserves_unknowns_without_authority_claims(compiled):
    _source, _session, adapter, compilation = compiled
    world = world_for(compilation)
    prepared = prepare(adapter, compilation, current_world=world)
    projected = evidence(prepared)["world-availability"]["world"]
    assert projected["projection"] == world.to_dict()
    assert projected["independent_owner_freshness"] == "not_revalidated"
    assert prepared.receipt()["world_projection_id"] == world.content_id
    assert prepared.receipt()["world_projection_status"] == "incomplete"
    assert prepared.receipt()["proof_authority"] is False


@pytest.mark.parametrize("field", ["repository_id", "tree_id", "objective_id", "task_id", "policy_revision"])
def test_foreign_world_binding_is_refused(compiled, field):
    _source, _session, adapter, compilation = compiled
    world = world_for(compilation)
    world = replace(world, bindings=replace(world.bindings, **{field: "sha256:foreign"}))
    with pytest.raises(RepositoryWorldPacketError, match="world_projection_binding_mismatch"):
        prepare(adapter, compilation, current_world=world)


@pytest.mark.parametrize("status", [WorldProjectionStatus.STALE, WorldProjectionStatus.INCOMPATIBLE])
def test_stale_or_incompatible_world_is_refused(compiled, status):
    _source, _session, adapter, compilation = compiled
    with pytest.raises(RepositoryWorldPacketError, match="world_projection_not_current"):
        prepare(adapter, compilation, current_world=world_for(compilation, projection_status=status))


def test_packet_noise_normalization_cannot_drop_compiled_required_values(native_session, required):
    _source, session = native_session
    adapter = session.context_adapter(compiler())
    compilation = compile_bound(adapter, required,
        goal={"intent": "Retain the exact timestamp field.", "timestamp": "authored-required-value"})
    with pytest.raises(RepositoryWorldPacketError, match="context_transformed"):
        prepare(adapter, compilation)


@pytest.mark.parametrize("changes", [
    {"task_text": "An unrelated task."}, {"task_id": "task:other"},
    {"allowed_effects": ("inspect_repository",)}, {"write_paths": ("unreviewed.py",)},
])
def test_preparation_requires_exact_task_and_scope_bound_before_compilation(compiled, changes):
    _source, _session, adapter, compilation = compiled
    with pytest.raises(RepositoryWorldPacketError, match="task_binding_mismatch"):
        prepare(adapter, compilation, **changes)


def test_older_unbound_context_cannot_be_implicitly_upgraded(native_session, required):
    _source, session = native_session
    adapter = session.context_adapter(compiler())
    compilation = compile_context(adapter, required)
    with pytest.raises(RepositoryWorldPacketError, match="task_binding_mismatch"):
        prepare(adapter, compilation)
