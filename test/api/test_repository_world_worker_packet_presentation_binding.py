"""Issued presentation custody on genuine native/compiled packet fixtures.

No worker/provider is launched and no tokenizer or target source is executed.
The existing fixture's authored transport does not prove TCP or model delivery.
"""
from copy import copy
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_world_worker_packet as owner
from ipfs_accelerate_py.agent_supervisor.runtime import repository_world_packet_presentation as view
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_worker_handoff import NativeRepositoryWorldWorkerHandoff
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet import PreparedRepositoryWorldPacket, prepare_repository_world_packet
from test.api.test_repository_world_worker_packet import bound, native_projection, scheduler  # noqa: F401
from test.api.test_repository_world_worker_packet import relation_change, _update_body


def presented_binding(case, presentation=None):
    presentation = view.prepare_repository_world_packet_presentation(case.packet.prompt) if presentation is None else presentation
    binding = owner.capture_repository_world_compiled_packet_binding(projection=case.projection,
        handoff=case.handoff, packet=case.packet, presentation=presentation)
    return binding, presentation


def unissued_presentation_copy(value):
    """Clone fields deliberately; never create or modify an issuer entry."""
    assert type(value) is view.PreparedRepositoryWorldPacketPresentation
    cloned = object.__new__(view.PreparedRepositoryWorldPacketPresentation)
    for field in ("prompt", "sidecar_bytes", "_receipt_bytes", "_seal"):
        object.__setattr__(cloned, field, getattr(value, field))
    assert cloned is not value
    return cloned


def detached_equal(value):
    if type(value) is view.PreparedRepositoryWorldPacketPresentation:
        return unissued_presentation_copy(value)
    if type(value) is bytes:
        return bytes(bytearray(value))
    if type(value) is str:
        return (value + "!")[:-1]
    if type(value) is tuple:
        return tuple(list(value))
    return copy(value)


def test_explicit_none_preserves_original_default_metadata_prompt_and_empty_new_refs(bound):
    expected = bound.binding.require_current()
    same = owner.capture_repository_world_compiled_packet_binding(projection=bound.projection,
        handoff=bound.handoff, packet=bound.packet, presentation=None)
    actual, text = same.delivery_material(handoff=bound.handoff, original_prompt=bound.projection.task_text)
    assert actual == expected and text is bound.packet.prompt
    assert not any(key.startswith("presentation_") or key.startswith("native_prompt_") for key in actual)
    assert all(getattr(same, field) is None for field in ("_presentation", "_presentation_refs",
        "_presentation_receipt_json", "_presentation_sidecar_bytes", "_presentation_prompt"))


def test_presented_packet_retains_native_prompt_identity_and_exact_inverse(bound):
    binding, presentation = presented_binding(bound)
    record, text = binding.delivery_material(handoff=bound.handoff, original_prompt=bound.projection.task_text)
    receipt = view.PreparedRepositoryWorldPacketPresentation.validate_original(presentation, bound.packet.prompt)
    original = bound.packet.receipt()
    assert text is presentation.prompt
    assert receipt["codec_supported"] is True and receipt["model_view_mode"] == view.MODE
    assert text != bound.packet.prompt
    assert record["native_prompt_sha256"] == original["prompt_sha256"] == receipt["original_prompt_sha256"]
    assert record["native_prompt_bytes"] == original["prompt_bytes"] == len(bound.packet.prompt.encode())
    assert record["prompt_sha256"] == hashlib.sha256(text.encode()).hexdigest() == receipt["presented_prompt_sha256"]
    assert record["prompt_bytes"] == len(text.encode())
    assert record["presentation_profile"] == view.MODE
    assert record["presentation_model_view_mode"] == receipt["model_view_mode"]
    assert record["presentation_codec_supported"] is receipt["codec_supported"]
    assert record["presentation_receipt_sha256"] == hashlib.sha256(owner._wire(receipt)).hexdigest()
    assert record["presentation_sidecar_sha256"] == hashlib.sha256(presentation.sidecar_bytes).hexdigest()
    assert record["presentation_sidecar_bytes"] == len(presentation.sidecar_bytes)
    restored = view.replay_repository_world_packet_presentation(view_prompt=text, sidecar_bytes=presentation.sidecar_bytes,
        expected_sidecar_sha256=record["presentation_sidecar_sha256"], expected_original_sha256=record["native_prompt_sha256"])
    assert restored == bound.packet.prompt
    for key in ("task_cid", "expected_task_body_cid", "native_task_relations_cid", "packet_id", "context_cid", "packet_receipt_sha256"):
        assert record[key] == bound.binding.require_current()[key]
    assert all(record[key] is False for key in owner.FALSE_FIELDS)
    assert "sidecar_bytes" not in record and bound.packet.prompt not in json.dumps(record)


@pytest.mark.parametrize("field", ["prompt", "sidecar_bytes", "_receipt_bytes"])
def test_equal_byte_presentation_field_replacement_before_capture_is_not_original_issuance(bound, field):
    presentation = view.prepare_repository_world_packet_presentation(bound.packet.prompt)
    original = getattr(presentation, field)
    replacement = detached_equal(original)
    assert replacement == original and replacement is not original
    object.__setattr__(presentation, field, replacement)
    try:
        with pytest.raises(ValueError):
            owner.capture_repository_world_compiled_packet_binding(projection=bound.projection,
                handoff=bound.handoff, packet=bound.packet, presentation=presentation)
    finally:
        object.__setattr__(presentation, field, original)


@pytest.mark.parametrize("field", ["_presentation", "_presentation_refs", "_presentation_receipt_json",
    "_presentation_sidecar_bytes", "_presentation_prompt"])
def test_equal_or_copied_binding_presentation_witness_cannot_replace_original(bound, field):
    binding, _presentation = presented_binding(bound)
    original = getattr(binding, field)
    replacement = detached_equal(original)
    assert replacement is not original
    setattr(binding, field, replacement)
    try:
        with pytest.raises(ValueError): binding.require_current()
    finally:
        setattr(binding, field, original)


@pytest.mark.parametrize("method", ["receipt", "validate_original", "_retained", "_seal_material"])
def test_presentation_instance_method_shadowing_is_refused(bound, method):
    _binding, presentation = presented_binding(bound)
    with pytest.raises((AttributeError, TypeError)):
        object.__setattr__(presentation, method, lambda *_arguments, **_keywords: {})


@pytest.mark.parametrize("value", [None, {}, object(), "historical receipt"])
def test_empty_or_unissued_presentation_has_closed_valueerror_without_native_activation(value):
    if value is None:
        value = object.__new__(view.PreparedRepositoryWorldPacketPresentation)
    with pytest.raises(ValueError):
        owner.capture_repository_world_compiled_packet_binding(
            projection=object.__new__(owner.NativeRepositoryWorldTaskProjection),
            handoff=object.__new__(NativeRepositoryWorldWorkerHandoff), packet=object.__new__(PreparedRepositoryWorldPacket),
            presentation=value)


def test_presentation_for_different_original_prompt_cannot_bind_live_packet(bound):
    core = bound.projection.compile_core()
    other_packet = prepare_repository_world_packet(adapter=bound.packet._adapter, compilation=bound.compilation,
        generation=1, task_id=bound.case.native.task_cid, task_text=bound.projection.task_text,
        unresolved_obligations=("Different original packet obligation, with no semantic or completion claim.",),
        validation_commands=(), allowed_effects=core["allowed_effects"], write_paths=core["write_paths"],
        validation_profile="read_only_observation@1", max_bytes=262144)
    assert other_packet.prompt != bound.packet.prompt
    different = view.prepare_repository_world_packet_presentation(other_packet.prompt)
    with pytest.raises(ValueError):
        owner.capture_repository_world_compiled_packet_binding(projection=bound.projection,
            handoff=bound.handoff, packet=bound.packet, presentation=different)


def test_copy_of_issued_presentation_is_not_a_new_native_owner(bound):
    presentation = view.prepare_repository_world_packet_presentation(bound.packet.prompt)
    unissued = unissued_presentation_copy(presentation)
    with pytest.raises(ValueError):
        owner.capture_repository_world_compiled_packet_binding(projection=bound.projection,
            handoff=bound.handoff, packet=bound.packet, presentation=unissued)


def test_tampered_sidecar_bytes_cannot_be_promoted_by_native_binding(bound):
    binding, presentation = presented_binding(bound)
    original = presentation.sidecar_bytes
    object.__setattr__(presentation, "sidecar_bytes", original + b" ")
    try:
        with pytest.raises(ValueError): binding.require_current()
    finally:
        object.__setattr__(presentation, "sidecar_bytes", original)


def test_equal_native_packet_capsule_replacement_still_refuses_presented_capture(bound):
    presentation = view.prepare_repository_world_packet_presentation(bound.packet.prompt)
    original = bound.packet._capsule
    object.__setattr__(bound.packet, "_capsule", copy(original))
    try:
        with pytest.raises(ValueError):
            owner.capture_repository_world_compiled_packet_binding(projection=bound.projection,
                handoff=bound.handoff, packet=bound.packet, presentation=presentation)
    finally:
        object.__setattr__(bound.packet, "_capsule", original)


def test_presented_binding_cannot_hide_native_business_relation_change(bound):
    binding, _presentation = presented_binding(bound)
    with relation_change(bound.case, "acceptance", column="criterion", value="changed native criterion"):
        with pytest.raises(ValueError): binding.require_current()


def test_presented_binding_cannot_hide_native_body_change(bound):
    binding, _presentation = presented_binding(bound)
    original = dict(bound.case.native.native_task.body)
    _update_body(bound.case.native, {**original, "title": "Different full native task"})
    try:
        with pytest.raises((ValueError, RuntimeError)): binding.require_current()
    finally:
        _update_body(bound.case.native, original)


def test_presented_binding_cannot_hide_current_source_change(bound):
    binding, _presentation = presented_binding(bound)
    path = bound.case.native.workspace / "module.py"
    original = path.read_bytes()
    path.write_bytes(original + b"# changed source\n")
    try:
        with pytest.raises(ValueError): binding.require_current()
    finally:
        path.write_bytes(original)


def test_presented_binding_cannot_hide_released_original_coordinator_claim(bound):
    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import DatabaseCoordinationExpiredError
    binding, _presentation = presented_binding(bound)
    coordinator = bound.case.native.daemon.coordinator
    claim = coordinator.get_task_claim(bound.case.native.attempt.claim_id)
    lease = coordinator.protect_task_claim(claim)
    coordinator.release(lease, reason="authored presentation test release")
    with pytest.raises(DatabaseCoordinationExpiredError): binding.require_current()


def test_pure_presentation_validation_precedes_final_native_closing_fences(bound, monkeypatch):
    binding, _presentation = presented_binding(bound)
    original_body = dict(bound.case.native.native_task.body)
    original = view.PreparedRepositoryWorldPacketPresentation.validate_original
    calls = []
    def changed(presentation, original_prompt):
        receipt = original(presentation, original_prompt)
        calls.append("presentation")
        _update_body(bound.case.native, {**original_body, "title": "Changed during presentation validation"})
        return receipt
    monkeypatch.setattr(view.PreparedRepositoryWorldPacketPresentation, "validate_original", changed)
    try:
        with pytest.raises((ValueError, RuntimeError)):
            binding.delivery_material(handoff=bound.handoff, original_prompt=bound.projection._text)
        assert calls == ["presentation"]
    finally:
        _update_body(bound.case.native, original_body)
