"""Original native business projections and read-only compiled packet joins.

Native claims, SQL tables, source capture and the compiler are real fixture
owners. FakeQuackTransport is an explicit authored control-plane seam. These
units launch no worker/provider, import no target code and assert no semantics.
"""
from contextlib import contextmanager
from copy import copy
from dataclasses import replace
import hashlib
import json
import os
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_world_worker_packet as owner
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_prelaunch import capture_repository_world_prelaunch_assignment
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_worker_handoff import capture_repository_world_worker_handoff
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet import (
    prepare_repository_world_packet, repository_world_task_binding, PreparedRepositoryWorldPacket,
)
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet_replay import (
    freeze_repository_world_packet, replay_repository_world_packet,
)
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_index import NativeRepositoryIndex
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_session import RepositorySourceSession
from test.api.test_current_source_worktree_allocation import _allocation_case, _record_change
from test.api.test_current_source_task_admission import _update_body
from test.api.test_repository_world_prelaunch import provision_authored_prelaunch_sources
from test.api.test_repository_context_compiler import compiler, obligation, scheduler  # noqa: F401
from test.api.test_repository_world_packet import compiled, native_session, required, prepare  # noqa: F401


@pytest.fixture
def native_projection(tmp_path, monkeypatch):
    with _allocation_case(tmp_path, monkeypatch, defer_source_binding=True) as case:
        case.native.binding = case.guard.bind_current_source_lookup(**case.source_binding)
        provision_authored_prelaunch_sources(case)
        projection = owner.capture_repository_world_task_projection(task_admission=case.native.handle)
        yield case, projection


@contextmanager
def relation_change(case, name, *, column, value):
    table, columns, ordering = owner.RELATIONS[name]
    connection = case.native.server._connection
    with case.native.server._lock:
        original = connection.execute("SELECT " + ",".join(columns) + " FROM " + table
            + " WHERE task_cid = ? ORDER BY " + ordering, [case.native.task_cid]).fetchall()
        original = [tuple(row[index] for index in range(len(columns))) for row in original]
        if name == "dependencies":
            connection.execute("INSERT INTO task_dependencies VALUES (?, ?, ?)",
                [case.native.task_cid, "authored:dependency", "requires"])
        elif name == "validations":
            connection.execute("INSERT INTO task_validations VALUES (?, ?, ?, ?)",
                [case.native.task_cid, 0, '["authored-validation-not-executed"]', '{}'])
        else:
            connection.execute("UPDATE " + table + " SET " + column + " = ? WHERE task_cid = ?",
                [value, case.native.task_cid])
    try:
        yield
    finally:
        with case.native.server._lock:
            connection.execute("DELETE FROM " + table + " WHERE task_cid = ?", [case.native.task_cid])
            for row in original:
                connection.execute("INSERT INTO " + table + " VALUES (" + ",".join("?" for _ in columns) + ")", list(row))


def compile_projection(session, projection, *, changes=None):
    adapter = session.context_adapter(compiler())
    core = projection.compile_core()
    metadata = projection.require_current()
    refs = (projection.context_reference(), *(obligation(name) for name in ("intent_ir", "security_ir", "ui_ux_ir")))
    args = dict(generation=1, query="VALUE", unresolved_question="Observe complete native source and task evidence.",
        required_evidence=refs, required_reference_ids=tuple(ref.reference_id for ref in refs),
        required_coverage_ids=tuple(value for ref in refs for value in ref.coverage_ids),
        objective_id=metadata["task_cid"], objective_revision=metadata["expected_task_body_cid"],
        policy_id="policy:authored-readonly", policy_revision="sha256:authored-policy",
        **{key: core[key] for key in ("goal", "scope", "acceptance", "authority")})
    args.update(changes or {})
    compilation = adapter.compile(**args)
    packet = prepare_repository_world_packet(adapter=adapter, compilation=compilation, generation=1,
        task_id=metadata["task_cid"], task_text=projection.task_text,
        unresolved_obligations=("Native validation, semantic approval and execution remain unavailable.",),
        validation_commands=(), allowed_effects=core["allowed_effects"], write_paths=core["write_paths"],
        validation_profile="read_only_observation@1", max_bytes=262144)
    return packet, compilation


@pytest.fixture
def bound(native_projection, tmp_path, scheduler):
    case, projection = native_projection
    assignment = capture_repository_world_prelaunch_assignment(task_admission=case.native.handle,
        worktree_admission=case.guard, prompt=projection.task_text)
    handoff = capture_repository_world_worker_handoff(assignment=assignment, worker_uid=os.geteuid() + 1,
        worker_gid=case.native.workspace.stat().st_gid, timeout_seconds=180)
    try:
        handoff.apply_readonly_transition()
        with RepositorySourceSession(case.native.workspace, tmp_path / "packet-session",
            index_factory=lambda path: NativeRepositoryIndex(path, scheduler=scheduler)) as session:
            packet, compilation = compile_projection(session, projection)
            binding = owner.capture_repository_world_compiled_packet_binding(projection=projection,
                handoff=handoff, packet=packet)
            yield SimpleNamespace(case=case, projection=projection, handoff=handoff, session=session,
                packet=packet, compilation=compilation, binding=binding)
    finally:
        handoff.close()


def test_complete_native_business_population_and_closed_inert_observation(native_projection):
    case, projection = native_projection
    public, receipt = projection.public_task(), projection.require_current()
    assert set(public) == {"schema", "profile", "task", "relations", "custody", "administrative_omissions"}
    assert set(public["relations"]) == set(owner.RELATIONS)
    assert public["task"]["task_cid"] == receipt["task_cid"] == case.native.task_cid
    assert public["task"]["revision"] == receipt["task_revision"]
    assert "completion_receipt" not in public["task"]["body"]
    assert "created_at" not in public["task"] and "updated_at" not in public["task"]
    assert "extension_schema" in public["task"] and "extension_json" in public["task"]
    assert receipt["relation_counts"] == {"dependencies": 0, "outputs": 1, "acceptance": 1, "validations": 0}
    assert public["relations"]["outputs"][0]["path"] == "module.py"
    assert public["relations"]["acceptance"][0]["criterion"] == "Source retrieval remains read only"
    core = projection.compile_core()
    assert json.loads(core["goal"]["native_task_contract_json"]) == public["task"]
    assert json.loads(core["scope"]["native_outputs_json"]) == public["relations"]["outputs"]
    assert json.loads(core["acceptance"]["native_acceptance_json"]) == public["relations"]["acceptance"]
    assert all(receipt[key] is False for key in owner.FALSE_FIELDS)
    assert projection.task_text.encode() == owner.canonical_json_bytes(public)
    assert public["custody"]["native_task_relations_cid"] == receipt["native_task_relations_cid"]
    assert str(case.native.workspace) not in json.dumps(receipt)
    assert projection.task_text not in json.dumps(receipt)
    receipt["native_task_relations_bound"] = False
    assert projection.require_current()["native_task_relations_bound"] is True


@pytest.mark.parametrize("name,column,value", [
    ("outputs", "effect_json", '{"path":"module.py","unknown":"λ","sequence":[2,1,2]}'),
    ("acceptance", "evidence_policy_json", '{"unknown":{"nullable":null,"empty":[]}}'),
    ("dependencies", "kind", None), ("validations", "argv_json", None),
])
def test_each_native_side_table_change_refuses_without_body_or_revision_change(native_projection, name, column, value):
    case, projection = native_projection
    before = projection.require_current()
    with relation_change(case, name, column=column, value=value):
        assert case.native.handle.require_current()["expected_task_body_cid"] == before["expected_task_body_cid"]
        with pytest.raises(ValueError, match="population_changed"):
            projection.require_current()
    assert projection.require_current() == before


def test_new_projection_preserves_unknown_native_relation_json_and_raw_spelling(native_projection):
    case, _projection = native_projection
    raw = ' {"path":"module.py","custom":{"ordered":[2,1,2],"unicode":"π","null":null}} '
    with relation_change(case, "outputs", column="effect_json", value=raw):
        projection = owner.capture_repository_world_task_projection(task_admission=case.native.handle)
        row = projection.public_task()["relations"]["outputs"][0]
        assert row["effect_json"] == raw and row["effect"]["custom"]["ordered"] == [2, 1, 2]


@pytest.mark.parametrize("column,value", [("priority", "urgent"), ("objective_id", "other:objective"),
    ("extension_schema", "authored-extension@2"), ("extension_json", '{"new":"business field"}')])
def test_native_scalar_extension_drift_is_not_hidden_by_body_cid(native_projection, column, value):
    case, projection = native_projection
    connection = case.native.server._connection
    with case.native.server._lock:
        original = connection.execute("SELECT " + column + " FROM tasks WHERE task_cid = ?", [case.native.task_cid]).fetchone()[0]
        connection.execute("UPDATE tasks SET " + column + " = ? WHERE task_cid = ?", [value, case.native.task_cid])
    try:
        with pytest.raises(ValueError, match="population_changed"):
            projection.require_current()
    finally:
        with case.native.server._lock:
            connection.execute("UPDATE tasks SET " + column + " = ? WHERE task_cid = ?", [original, case.native.task_cid])


@pytest.mark.parametrize("raw", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":1.5}'])
def test_strict_native_relation_json_has_no_duplicate_nonfinite_or_untyped_coercion(native_projection, raw):
    case, _projection = native_projection
    with relation_change(case, "outputs", column="effect_json", value=raw):
        with pytest.raises(ValueError):
            owner.capture_repository_world_task_projection(task_admission=case.native.handle)


@pytest.mark.parametrize("key", ["access_token", "clientSecret", "credential", "password", "private-key"])
def test_nested_credential_keys_are_refused_not_silently_filtered(native_projection, key):
    case, _projection = native_projection
    with relation_change(case, "outputs", column="effect_json", value=json.dumps({"custom": {key: "authored-sensitive-value"}})):
        with pytest.raises(ValueError, match="credential_key_refused"):
            owner.capture_repository_world_task_projection(task_admission=case.native.handle)


@pytest.mark.parametrize("limit,value", [("MAX_ROWS_PER_RELATION", 0), ("MAX_TOTAL_RELATION_ROWS", 1),
    ("MAX_CELL_CHARACTERS", 10), ("MAX_SOURCE_CHARACTERS", 100), ("MAX_PUBLIC_TASK_BYTES", 100)])
def test_native_projection_population_and_wire_bounds_refuse_instead_of_truncating(native_projection, monkeypatch, limit, value):
    case, _projection = native_projection
    monkeypatch.setattr(owner, limit, value)
    with pytest.raises(ValueError, match="bound"):
        owner.capture_repository_world_task_projection(task_admission=case.native.handle)


@pytest.mark.parametrize("kind", ["receipt", "copy", "empty", "subclass"])
def test_native_projection_requires_original_issued_exact_object(native_projection, kind):
    _case, projection = native_projection
    if kind == "receipt":
        bad = projection.require_current()
    elif kind == "copy": bad = copy(projection)
    elif kind == "empty": bad = object.__new__(owner.NativeRepositoryWorldTaskProjection)
    else:
        class Other(owner.NativeRepositoryWorldTaskProjection): pass
        bad = object.__new__(Other)
    with pytest.raises(ValueError):
        owner.NativeRepositoryWorldTaskProjection.require_current(bad)


@pytest.mark.parametrize("method", ["require_current", "_retained", "compile_core"])
def test_projection_instance_methods_cannot_shadow_native_owner_fences(native_projection, method):
    projection = native_projection[1]
    with pytest.raises(AttributeError):
        setattr(projection, method, lambda *_a, **_kw: {})


def test_posttransition_packet_binds_native_task_and_original_compiler_separately(bound):
    row, text = bound.binding.delivery_material(handoff=bound.handoff, original_prompt=bound.projection.task_text)
    assert text == bound.packet.prompt and text != bound.projection.task_text
    assert row["task_cid"] == bound.case.native.task_cid
    assert row["compiled_task_packet_cid"] == bound.packet.receipt()["task_cid"]
    assert row["compiled_task_packet_cid"] != row["task_cid"]
    assert row["context_cid"] == bound.compilation.capsule.capsule_id
    assert row["prompt_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert row["native_task_relations_cid"] == bound.projection.require_current()["native_task_relations_cid"]
    assert all(row[field] is False for field in owner.FALSE_FIELDS)
    assert row["structural_compiler_context_current"] is row["native_task_projection_current"] is True
    assert bound.handoff.require_current()["compiled_context_verified"] is False
    assert bound.binding.require_for_delivery(handoff=bound.handoff, original_prompt=bound.projection.task_text.encode()) == row
    with pytest.raises(ValueError):
        bound.case.guard.require_current()
    row["execution_authority"] = True
    assert bound.binding.require_current()["execution_authority"] is False


@pytest.mark.parametrize("change", ["prompt", "handoff"])
def test_delivery_membership_cannot_retarget_original_handoff_or_raw_task(bound, change):
    values = dict(handoff=bound.handoff, original_prompt=bound.projection.task_text)
    values[change if change == "handoff" else "original_prompt"] = object() if change == "handoff" else "another task"
    with pytest.raises(ValueError, match="assignment_mismatch"):
        bound.binding.delivery_material(**values)


@pytest.mark.parametrize("name,column,value", [("outputs", "path", "other.py"),
    ("acceptance", "criterion", "altered criterion"), ("dependencies", "kind", None), ("validations", "argv_json", None)])
def test_live_compiled_binding_rechecks_all_original_native_business_relations(bound, name, column, value):
    with relation_change(bound.case, name, column=column, value=value):
        with pytest.raises(ValueError):
            bound.binding.require_current()


def test_live_compiled_binding_refuses_native_claim_release(bound):
    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import DatabaseCoordinationExpiredError
    coordinator = bound.case.native.daemon.coordinator
    claim = coordinator.get_task_claim(bound.case.native.attempt.claim_id)
    lease = coordinator.protect_task_claim(claim)
    coordinator.release(lease, reason="authored test release")
    with pytest.raises(DatabaseCoordinationExpiredError):
        bound.binding.require_current()


def test_live_binding_refuses_source_change_and_keeps_original_source_session(bound):
    path = bound.case.native.workspace / "module.py"
    original = path.read_bytes()
    path.write_bytes(original + b"# changed\n")
    try:
        with pytest.raises(ValueError):
            bound.binding.require_current()
    finally:
        path.write_bytes(original)


def test_lifecycle_loss_cannot_be_accepted_as_structural_compilation(bound):
    with _record_change(bound.case, lambda value: value.update(expires_at=0)):
        with pytest.raises(ValueError):
            bound.binding.require_current()


@pytest.mark.parametrize("field", ["_projection", "_handoff", "_packet", "_compiler", "_session", "_metadata"])
def test_original_owner_and_byte_identical_private_witness_replacement_refuses(bound, field):
    original = getattr(bound.binding, field)
    value = bytes(bytearray(original)) if field == "_metadata" else copy(original)
    setattr(bound.binding, field, value)
    try:
        with pytest.raises(ValueError):
            bound.binding.require_current()
    finally:
        setattr(bound.binding, field, original)


@pytest.mark.parametrize("field", ["_capsule", "_workspace", "_receipt_json"])
def test_equal_packet_field_replacement_before_new_capture_is_not_original_issuance(bound, field):
    from pathlib import Path
    original = getattr(bound.packet, field)
    value = (replace(original) if field == "_capsule" else Path(str(original)) if field == "_workspace"
             else original.encode().decode())
    assert value == original and value is not original
    object.__setattr__(bound.packet, field, value)
    try:
        with pytest.raises(ValueError, match="original_issued_packet_replaced"):
            owner.capture_repository_world_compiled_packet_binding(projection=bound.projection,
                handoff=bound.handoff, packet=bound.packet)
    finally:
        object.__setattr__(bound.packet, field, original)


def test_packet_receipt_method_shadow_is_not_native_profile_authority(bound):
    expected = bound.packet.receipt()
    object.__setattr__(bound.packet, "receipt", lambda: expected)
    try:
        with pytest.raises(ValueError, match="owner_method_replaced"):
            owner.capture_repository_world_compiled_packet_binding(projection=bound.projection,
                handoff=bound.handoff, packet=bound.packet)
    finally:
        object.__delattr__(bound.packet, "receipt")


def test_original_source_material_cannot_be_resealed_in_place(bound):
    receipt = bound.session._capture_receipt
    original = receipt["source_tree_sha256"]
    receipt["source_tree_sha256"] = "0" * 64
    try:
        with pytest.raises(ValueError, match="original_source_material_changed"):
            bound.binding.require_current()
    finally:
        receipt["source_tree_sha256"] = original


@pytest.mark.parametrize("method", ["require_current", "_checked", "require_for_delivery", "delivery_material"])
def test_binding_instance_methods_cannot_replace_checked_packet_authority(bound, method):
    with pytest.raises(AttributeError):
        setattr(bound.binding, method, lambda *_a, **_kw: {})


@pytest.mark.parametrize("partial,method", [(False, "require_current"), (True, "require_current"),
    (False, "delivery_material"), (True, "delivery_material")])
def test_empty_and_partially_populated_binding_has_closed_valueerror(partial, method):
    value = object.__new__(owner.NativeRepositoryWorldCompiledPacketBinding)
    if partial: value._seal = owner._SEAL
    with pytest.raises(ValueError):
        if method == "require_current": owner.NativeRepositoryWorldCompiledPacketBinding.require_current(value)
        else: owner.NativeRepositoryWorldCompiledPacketBinding.delivery_material(value, handoff=None, original_prompt=b"x")


def readonly_compilation(compiled, required):
    _source, _session, adapter, _old = compiled
    from test.api.test_repository_world_packet import TASK, CORE, compile_bound
    authority = {**CORE["authority"], "world_task_packet": repository_world_task_binding(
        "task:arithmetic", TASK, ("repository_read",), ("calc.py",))}
    return adapter, compile_bound(adapter, required, authority=authority)


def test_readonly_empty_validation_profile_preserves_full_context_and_historical_inverse(compiled, required):
    adapter, compilation = readonly_compilation(compiled, required)
    packet = prepare(adapter, compilation, allowed_effects=("repository_read",), validation_commands=(),
        proof_obligations=(), validation_profile="read_only_observation@1")
    receipt = packet.receipt()
    assert receipt["native_validation_count"] == receipt["validation_count"] == 0
    assert receipt["native_validation_coverage"] == "no_native_validation_commands_declared"
    assert receipt["world_dispatch_ready"] is receipt["provider_launch"] is False
    payload = json.loads(packet.prompt.split("\n", 1)[1])["packet"]
    assert payload["requirements"]["validation_commands"] == []
    assert payload["requirements"]["require_validation"] is False
    assert payload["effect_scope"]["allowed_effects"] == ["repository_read"]
    raw = freeze_repository_world_packet(packet)
    replay = replay_repository_world_packet(raw, expected_sha256=hashlib.sha256(raw).hexdigest())
    assert replay.prompt == packet.prompt
    assert replay.recorded_packet_receipt() == receipt
    assert replay.receipt()["provider_dispatch_verified"] is False
    assert json.loads(raw)["recipe"]["validation_profile"] == "read_only_observation@1"


@pytest.mark.parametrize("changes", [{"allowed_effects": ("write_source",)},
    {"allowed_effects": ("repository_read", "write_source")}, {"proof_obligations": ("proof:x",)},
    {"validation_commands": ("invented validation",)}, {"validation_profile": "other"},
    {"validation_profile": False}])
def test_readonly_optin_cannot_broaden_effects_or_invent_native_validation(compiled, required, changes):
    adapter, compilation = readonly_compilation(compiled, required)
    values = dict(allowed_effects=("repository_read",), validation_commands=(), proof_obligations=(),
        validation_profile="read_only_observation@1") | changes
    with pytest.raises(ValueError):
        prepare(adapter, compilation, **values)


def test_default_strict_profile_keeps_original_field_population_and_refuses_empty_commands(compiled):
    _source, _session, adapter, compilation = compiled
    first = prepare(adapter, compilation)
    explicit_none = prepare(adapter, compilation, validation_profile=None)
    assert first.prompt == explicit_none.prompt and first.receipt() == explicit_none.receipt()
    assert "validation_profile" not in first.receipt()
    assert "validation_profile" not in json.loads(freeze_repository_world_packet(first))["recipe"]
    with pytest.raises(ValueError, match="missing_validation"):
        prepare(adapter, compilation, validation_commands=())


def test_readonly_packet_cannot_enter_provider_discovery_or_environment_mutation(compiled, required, monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
    from test.api.test_repository_world_packet import TASK
    adapter, compilation = readonly_compilation(compiled, required)
    packet = prepare(adapter, compilation, allowed_effects=("repository_read",), validation_commands=(),
        proof_obligations=(), validation_profile="read_only_observation@1")
    def forbidden(*_a, **_kw):
        pytest.fail("read-only packet reached a provider or workspace side effect")
    monkeypatch.setattr(intelligence_index, "discover_available_providers", forbidden)
    monkeypatch.setattr(intelligence_index, "select_efficient_route", forbidden)
    monkeypatch.setattr(llm_router, "generate_text", forbidden)
    monkeypatch.setattr(llm_router, "get_llm_provider", forbidden)
    monkeypatch.setattr(runner.subprocess, "check_output", forbidden)
    before = dict(os.environ)
    with pytest.raises(ValueError, match="readonly_provider_dispatch_refused"):
        runner.run(prompt="native task", provider="codex_cli", model="authored", timeout=30,
            max_output_tokens=128, world_task_packet=packet)
    assert dict(os.environ) == before
    # Currentness is still valid for the separate authenticated evidence service.
    assert packet.validate_current(workspace=compiled[0], task_text=TASK)["validation_count"] == 0


@pytest.mark.parametrize("change", ["profile", "missing_profile", "effects", "proofs", "validation"])
def test_resealed_readonly_history_cannot_enable_dispatch_or_skip_profile_checks(compiled, required, change):
    adapter, compilation = readonly_compilation(compiled, required)
    packet = prepare(adapter, compilation, allowed_effects=("repository_read",), validation_commands=(),
        proof_obligations=(), validation_profile="read_only_observation@1")
    artifact = json.loads(freeze_repository_world_packet(packet))
    if change == "profile": artifact["recipe"]["validation_profile"] = "writable@1"
    elif change == "missing_profile": artifact["recipe"].pop("validation_profile")
    elif change == "effects": artifact["recipe"]["allowed_effects"] = ["write_source"]
    elif change == "proofs": artifact["recipe"]["proof_obligations"] = ["proof:x"]
    else: artifact["recipe"]["validation_commands"] = ["invented"]
    raw = json.dumps(artifact, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    with pytest.raises(ValueError):
        replay_repository_world_packet(raw, expected_sha256=hashlib.sha256(raw).hexdigest())
