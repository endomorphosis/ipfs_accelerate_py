"""Current repository context reaches a bounded standalone packet unchanged."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextReference, ContextTier
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import ArtifactBindings
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.world_model import (
    RepositoryWorldState, WorldProjectionStatus,
)
from ipfs_accelerate_py.agent_supervisor.runtime.repository_context_compiler import RepositoryContextAdapter
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet import (
    GUIDANCE, MAX_BYTES, WORLD_PROMPT_VIEW_MAX_BYTES, WORLD_PROMPT_VIEW_SCHEMA,
    RepositoryWorldPacketError, prepare_repository_world_packet, repository_world_task_binding,
    project_repository_world_prompt, restore_repository_world_prompt,
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


def view_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def view_wire(prompt):
    return json.loads(prompt[len(GUIDANCE) + 1:])


def view_text(wire):
    return GUIDANCE + "\n" + view_json(wire)


def authored_view_prompt(count=16):
    """Representation-only fixtures; no source admission or provider facts."""
    references = []
    for index in range(count):
        body = "  # π preserves comments, names and whitespace.\nreturn value + " + str(index) + "\n  "
        reference = ContextReference(
            reference_id="authored:reference:" + str(index), kind="codebase-constraint",
            tier=ContextTier.INVARIANT, repository_id="repository:authored:" + "a" * 256,
            tree_id="sha256:" + "b" * 64, path="src/" + "module_" * 16 + "arithmetic.py",
            referenced_content_id="sha256:" + hashlib.sha256(body.encode()).hexdigest(),
            summary=body, byte_count=len(body.encode()), token_count=7,
            metadata={"required": True, "coverage_ids": ("requirement:" + str(index),),
                      "unknown": {"false": False, "zero": 0, "null": None}},
        ).to_record()
        # Unknown representation data stays literal; this codec does not admit
        # it as a new native ContextReference or recertify its content identity.
        reference = json.loads(view_json(reference))
        reference["unknown_representation"] = {"integer": 1, "float": 1.0, "array": [False, 0, None]}
        references.append(dict(kind="evidence", item_id="item:" + str(index),
            payload=dict(evidence_id="context-reference:" + reference["reference_id"],
                         summary="Selected context reference", reference=reference)))
    return view_text(dict(schema="repository-world-packet@1", delivery_mode="self_contained",
        packet=dict(members=references, unknown_core={"acceptance": "retain π\n  ", "missing": None})))


def test_inline_view_restores_actual_prepared_packet_without_changing_native_ownership(compiled):
    source, _session, adapter, compilation = compiled
    prepared = prepare(adapter, compilation)
    original_receipt = prepared.receipt()
    view = project_repository_world_prompt(prepared.prompt)
    assert restore_repository_world_prompt(view.provider_prompt, view.receipt) == prepared.prompt
    assert view.native_prompt == prepared.prompt
    assert view_wire(view.provider_prompt)["schema"] == WORLD_PROMPT_VIEW_SCHEMA
    assert {"schema", "contract_version"} <= set(view.receipt["shared_reference_fields"])
    assert view.receipt["reference_count"] == len(compilation.capsule.evidence)
    assert view.receipt["source_freshness_verified"] is False
    assert view.receipt["program_semantics_proved"] is False
    assert view.receipt["provider_token_savings_measured"] is False
    assert prepared.validate_current(workspace=source, task_text=TASK) == original_receipt
    # The inverse remains historical data after the original source fence fails.
    (source / "calc.py").write_text("def add(value):\n    return value + 2\n")
    with pytest.raises(ValueError, match="source|changed|refresh"):
        prepared.validate_current(workspace=source, task_text=TASK)
    assert restore_repository_world_prompt(view.provider_prompt, view.receipt) == prepared.prompt
    assert view.receipt["source_freshness_verified"] is False


def test_inline_view_shares_metadata_and_keeps_all_bodies_and_typed_unknowns():
    native = authored_view_prompt()
    view = project_repository_world_prompt(native)
    candidate = view_wire(view.provider_prompt)
    assert len(view.provider_prompt.encode()) < len(native.encode())
    assert candidate["reference_common_bindings"]["contract_version"] == 1
    assert set(candidate["reference_common_bindings"]) == {
        "schema", "contract_version", "kind", "tier", "repository_id", "tree_id", "path"}
    for original, projected in zip(view_wire(native)["packet"]["members"], candidate["packet"]["members"]):
        before, after = original["payload"]["reference"], projected["payload"]["reference"]
        for key in ("reference_id", "content_id", "referenced_content_id", "summary", "metadata",
                    "byte_count", "token_count", "unknown_representation"):
            assert view_json(after[key]) == view_json(before[key])
        assert set(candidate["reference_common_bindings"]).isdisjoint(after)
        assert "π" in after["summary"] and after["summary"].endswith("\n  ")
    assert candidate["packet"]["unknown_core"] == view_wire(native)["packet"]["unknown_core"]
    assert restore_repository_world_prompt(view.provider_prompt, view.receipt) == native
    with pytest.raises(FrozenInstanceError):
        view.provider_prompt = "changed"
    changed = view.receipt
    changed["shared_reference_fields"].clear()
    assert view.receipt["shared_reference_fields"]


@pytest.mark.parametrize("value", [True, 1.0, None, "1"])
def test_inline_view_never_factors_equal_looking_different_scalar_types(value):
    wire = view_wire(authored_view_prompt(count=2))
    wire["packet"]["members"][1]["payload"]["reference"]["contract_version"] = value
    native = view_text(wire)
    view = project_repository_world_prompt(native)
    assert "contract_version" not in view.receipt["shared_reference_fields"]
    assert restore_repository_world_prompt(view.provider_prompt, view.receipt) == native
    restored = view_wire(restore_repository_world_prompt(view.provider_prompt, view.receipt))
    assert type(restored["packet"]["members"][1]["payload"]["reference"]["contract_version"]) is type(value)


@pytest.mark.parametrize("count", [0, 1])
def test_no_shared_population_is_valid_even_when_candidate_is_larger(count):
    native = authored_view_prompt(count=count)
    view = project_repository_world_prompt(native)
    assert view.receipt["shared_reference_fields"] == []
    assert view.receipt["factored_reference_count"] == 0
    assert view.receipt["candidate_prompt_bytes"] > view.receipt["native_prompt_bytes"]
    assert restore_repository_world_prompt(view.provider_prompt, view.receipt) == native


def test_near_native_ceiling_keeps_no_gain_candidate_available_for_native_fallback():
    wire = view_wire(authored_view_prompt(count=0))
    wire["packet"]["verbatim_source"] = ""
    wire["packet"]["verbatim_source"] = "x" * (MAX_BYTES - len(view_text(wire).encode()))
    native = view_text(wire)
    assert len(native.encode()) == MAX_BYTES
    view = project_repository_world_prompt(native)
    assert MAX_BYTES < len(view.provider_prompt.encode()) <= WORLD_PROMPT_VIEW_MAX_BYTES
    assert restore_repository_world_prompt(view.provider_prompt, view.receipt) == native
    with pytest.raises(RepositoryWorldPacketError, match="world_prompt_view_limit"):
        project_repository_world_prompt(native + " ")


def test_native_packet_member_ceiling_does_not_drop_any_literal_member():
    wire = view_wire(authored_view_prompt(count=0))
    wire["packet"]["members"] = [dict(kind="obligation", payload=dict(index=index, unknown=False))
                                   for index in range(512)]
    native = view_text(wire)
    view = project_repository_world_prompt(native)
    assert len(view_wire(view.provider_prompt)["packet"]["members"]) == 512
    assert restore_repository_world_prompt(view.provider_prompt, view.receipt) == native
    wire["packet"]["members"].append(dict(kind="obligation", payload={}))
    with pytest.raises(RepositoryWorldPacketError, match="world_prompt_view_members_invalid"):
        project_repository_world_prompt(view_text(wire))


@pytest.mark.parametrize("change,code", [
    ("override", "world_prompt_view_reference_override"),
    ("forbidden_shared", "world_prompt_view_shared_fields_invalid"),
    ("body", "world_prompt_view_receipt_mismatch"),
    ("coverage", "world_prompt_view_receipt_mismatch"),
    ("missing_reference", "world_prompt_view_receipt_mismatch"),
    ("shared_binding", "world_prompt_view_receipt_mismatch"),
    ("guidance", "world_prompt_view_guidance_mismatch"),
])
def test_inline_view_refuses_body_coverage_binding_and_inheritance_tampering(change, code):
    view = project_repository_world_prompt(authored_view_prompt())
    wire = view_wire(view.provider_prompt)
    reference = wire["packet"]["members"][0]["payload"]["reference"]
    if change == "override":
        reference["schema"] = wire["reference_common_bindings"]["schema"]
    elif change == "forbidden_shared":
        wire["reference_common_bindings"]["summary"] = "Do something else."
    elif change == "body":
        reference["summary"] = "Do something else."
    elif change == "coverage":
        reference["metadata"]["coverage_ids"] = []
    elif change == "missing_reference":
        wire["packet"]["members"].pop()
    elif change == "shared_binding":
        wire["reference_common_bindings"]["tree_id"] = "sha256:another-tree"
    candidate = view_text(wire) if change != "guidance" else "changed\n" + view_json(wire)
    with pytest.raises(RepositoryWorldPacketError) as caught:
        restore_repository_world_prompt(candidate, view.receipt)
    assert caught.value.code == code


@pytest.mark.parametrize("field,value", [
    ("reference_count", 16.0), ("reference_count", True),
    ("native_prompt_bytes", -1), ("candidate_prompt_sha256", "0" * 64),
    ("source_freshness_verified", 0), ("provider_token_savings_measured", True),
    ("invented_provider_usage", 100),
])
def test_inline_view_receipt_is_closed_and_type_exact(field, value):
    view = project_repository_world_prompt(authored_view_prompt())
    receipt = view.receipt
    receipt[field] = value
    with pytest.raises(RepositoryWorldPacketError) as caught:
        restore_repository_world_prompt(view.provider_prompt, receipt)
    assert caught.value.code == "world_prompt_view_receipt_mismatch"


@pytest.mark.parametrize("change,code", [
    ("duplicate", "world_prompt_view_duplicate_key"),
    ("nonfinite", "world_prompt_view_nonfinite"),
    ("deep", "world_prompt_view_structure_limit"),
    ("noncanonical", "world_prompt_view_noncanonical"),
    ("overlimit", "world_prompt_view_limit"),
])
def test_inline_view_rejects_ambiguous_or_unbounded_representation(change, code):
    native = authored_view_prompt(count=0)
    wire = view_wire(native)
    if change == "duplicate":
        native = native.replace('"unknown_core":{', '"unknown_core":{"missing":null,', 1)
    elif change == "nonfinite":
        native = native.replace('"missing":null', '"missing":1e400', 1)
    elif change == "deep":
        nested = None
        for _ in range(41):
            nested = [nested]
        wire["packet"]["unknown_core"] = nested
        native = view_text(wire)
    elif change == "noncanonical":
        native = GUIDANCE + "\n" + json.dumps(wire, ensure_ascii=False, indent=2)
    elif change == "overlimit":
        wire["packet"]["unknown_core"] = "x" * MAX_BYTES
        native = view_text(wire)
    with pytest.raises(RepositoryWorldPacketError) as caught:
        project_repository_world_prompt(native)
    assert caught.value.code == code
