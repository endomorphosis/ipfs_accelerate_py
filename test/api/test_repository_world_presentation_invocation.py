"""Historical input custody using the actual router and authored provider seam.

The source/session/compiler are real local fixtures. Model calls and their usage
observations are authored; none of these cases establish model quality or cost.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_world_presentation_invocation as archive
from ipfs_accelerate_py.agent_supervisor.runtime import repository_world_packet_presentation as codec
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet_replay import freeze_repository_world_packet
from test.api.test_repository_world_dispatch import (  # noqa: F401
    dispatch_case, world_session, required, scheduler, counter, invocation_receipt,
)
from test.api.test_repository_world_presentation_dispatch import presented, invoke  # noqa: F401


def wire(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def captured(presented):
    case, presentation = presented
    original = freeze_repository_world_packet(case[2])  # Before any authored provider edit.
    _, receipt = invoke(presented)
    args = dict(packet_artifact=original, expected_packet_sha256=sha(original),
        invocation_receipt=receipt, expected_invocation_sha256=archive.invocation_receipt_sha256(receipt),
        presentation=presentation)
    data = archive.freeze_repository_world_presentation_invocation(**args)
    pins = dict(expected_sha256=sha(data), expected_packet_sha256=sha(original),
        expected_invocation_sha256=args["expected_invocation_sha256"],
        expected_sidecar_sha256=sha(presentation.sidecar_bytes))
    return case, presentation, receipt, args, data, pins


def test_actual_rendered_input_and_complete_receipt_are_reconstructed(captured):
    case, presentation, receipt, _, data, pins = captured
    replay = archive.replay_repository_world_presentation_invocation(data, **pins)
    assert replay.model_prompt == case[3]["seen"][0][0]
    assert replay.model_prompt.endswith(presentation.prompt)
    assert replay.recorded_invocation_receipt() == receipt
    diagnostic = replay.receipt()
    assert diagnostic["model_prompt_sha256"] == sha(replay.model_prompt.encode())
    assert diagnostic["model_prompt_bytes"] == len(replay.model_prompt.encode())
    assert diagnostic["recorded_usage_sha256"] == sha(wire(receipt["usage"]))
    assert diagnostic["recorded_provider_attempted"] is True
    assert diagnostic["historical_model_input_reconstructed"] is True
    assert all(diagnostic[key] is False for key in archive.FALSE_FIELDS)
    assert not hasattr(replay, "validate_current") and not hasattr(replay, "require_provider_dispatch")
    detached = replay.recorded_invocation_receipt()
    detached["usage"]["prompt_tokens"] = 999
    assert replay.recorded_invocation_receipt() == receipt
    assert presentation.prompt not in repr(replay) and presentation.prompt not in wire(diagnostic).decode()


def test_explicit_independent_view_and_sidecar_produce_identical_bundle(captured):
    _, presentation, _, args, data, _ = captured
    explicit = args | dict(presentation=None, view_prompt=presentation.prompt,
        sidecar_bytes=presentation.sidecar_bytes, expected_sidecar_sha256=sha(presentation.sidecar_bytes))
    assert archive.freeze_repository_world_presentation_invocation(**explicit) == data


def test_valid_source_edit_after_preprovider_capture_does_not_require_live_packet(presented, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet import PreparedRepositoryWorldPacket
    case, presentation = presented
    original = freeze_repository_world_packet(case[2])
    def edit(_prompt, _options):
        path = case[0] / "calc.py"
        path.write_text(path.read_text().replace("return value + 1", "return value + 2"))
    case[3]["callback"] = edit
    _, receipt = invoke(presented)
    assert "return value + 2" in (case[0] / "calc.py").read_text()
    def forbidden(*_args, **_kwargs):
        pytest.fail("historical capture reopened the original live packet")
    monkeypatch.setattr(PreparedRepositoryWorldPacket, "validate_current", forbidden)
    monkeypatch.setattr(PreparedRepositoryWorldPacket, "require_provider_dispatch", forbidden)
    data = archive.freeze_repository_world_presentation_invocation(packet_artifact=original,
        expected_packet_sha256=sha(original), invocation_receipt=receipt,
        expected_invocation_sha256=archive.invocation_receipt_sha256(receipt), presentation=presentation)
    replay = archive.replay_repository_world_presentation_invocation(data, expected_sha256=sha(data),
        expected_packet_sha256=sha(original), expected_invocation_sha256=archive.invocation_receipt_sha256(receipt),
        expected_sidecar_sha256=sha(presentation.sidecar_bytes))
    assert replay.model_prompt == case[3]["seen"][0][0]
    assert replay.receipt()["source_currentness_verified"] is False


def test_replay_never_reads_workspace_issues_owner_or_calls_provider(captured, monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.agent_supervisor.runtime import repository_world_packet as packet_owner
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_session import RepositorySourceSession
    case, _, _, _, data, pins = captured
    def forbidden(*_args, **_kwargs):
        pytest.fail("historical replay attempted live IO or issuance")
    # The original owner remains available for normal fixture teardown. Refuse
    # each live entry point and all filesystem path resolution during replay.
    with monkeypatch.context() as patch:
        patch.setattr(packet_owner, "prepare_repository_world_packet", forbidden)
        patch.setattr(codec, "prepare_repository_world_packet_presentation", forbidden)
        patch.setattr(RepositorySourceSession, "__init__", forbidden)
        patch.setattr(llm_router, "generate_text", forbidden)
        patch.setattr(Path, "resolve", forbidden)
        patch.setattr(Path, "stat", forbidden)
        patch.setattr(Path, "read_bytes", forbidden)
        patch.setattr(Path, "read_text", forbidden)
        replay = archive.replay_repository_world_presentation_invocation(data, **pins)
    assert replay.model_prompt == case[3]["seen"][0][0]


def test_actual_provider_failure_and_incomplete_usage_are_preserved(presented, monkeypatch, capsys):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
    case, presentation = presented
    original = freeze_repository_world_packet(case[2])
    def fail(prompt, **_options):
        case[3]["seen"].append((prompt, {}))
        set_last_cli_observation("codex_cli", {"exit_code": 7, "prompt_tokens": 21})
        raise RuntimeError("authored provider failure")
    monkeypatch.setattr(llm_router, "generate_text", fail)
    with pytest.raises(RuntimeError, match="authored provider failure"):
        invoke(presented)
    receipt = invocation_receipt(capsys)
    data = archive.freeze_repository_world_presentation_invocation(packet_artifact=original,
        expected_packet_sha256=sha(original), invocation_receipt=receipt,
        expected_invocation_sha256=archive.invocation_receipt_sha256(receipt), presentation=presentation)
    replay = archive.replay_repository_world_presentation_invocation(data, expected_sha256=sha(data),
        expected_packet_sha256=sha(original), expected_invocation_sha256=archive.invocation_receipt_sha256(receipt),
        expected_sidecar_sha256=sha(presentation.sidecar_bytes))
    assert replay.recorded_invocation_receipt() == receipt
    assert replay.model_prompt == case[3]["seen"][0][0]
    assert replay.receipt()["recorded_status"] == "failed"
    assert replay.receipt()["recorded_provider_attempted"] is True
    assert replay.receipt()["billing_verified"] is replay.receipt()["provider_response_replayed"] is False


def test_predispatch_source_failure_keeps_false_attempt_and_rendered_input(presented, monkeypatch, capsys):
    from ipfs_accelerate_py import llm_router
    case, presentation = presented
    original = freeze_repository_world_packet(case[2])
    path = case[0] / "calc.py"
    def initialize(*_args, **_kwargs):
        path.write_bytes(path.read_bytes() + b"# changed before dispatch\n")
        return object()
    monkeypatch.setattr(llm_router, "get_llm_provider", initialize)
    with pytest.raises(ValueError):
        invoke(presented)
    receipt = invocation_receipt(capsys)
    data = archive.freeze_repository_world_presentation_invocation(packet_artifact=original,
        expected_packet_sha256=sha(original), invocation_receipt=receipt,
        expected_invocation_sha256=archive.invocation_receipt_sha256(receipt), presentation=presentation)
    replay = archive.replay_repository_world_presentation_invocation(data, expected_sha256=sha(data),
        expected_packet_sha256=sha(original), expected_invocation_sha256=archive.invocation_receipt_sha256(receipt),
        expected_sidecar_sha256=sha(presentation.sidecar_bytes))
    assert case[3]["seen"] == []
    assert replay.recorded_invocation_receipt() == receipt
    assert replay.receipt()["recorded_provider_attempted"] is False
    assert replay.receipt()["recorded_dispatch_rechecked"] is False


def test_full_receipt_keeps_unknown_observations_and_order_without_upgrading_usage(captured):
    _, _, receipt, args, _, _ = captured
    # Future recorded fields are archival data, never promoted into trust flags.
    recorded = deepcopy(receipt)
    recorded["future_observation"] = {"rows": ["duplicate", "duplicate", None], "counter": 0.5}
    recorded["usage"] = {"prompt_tokens": None, "completion_tokens": None, "attempts": [None, {"failed": True}]}
    data = archive.freeze_repository_world_presentation_invocation(**(args | dict(invocation_receipt=recorded,
        expected_invocation_sha256=archive.invocation_receipt_sha256(recorded))))
    full = json.loads(data)["invocation"]
    assert full == recorded and full["future_observation"]["rows"] == ["duplicate", "duplicate", None]


@pytest.mark.parametrize("key", ["expected_sha256", "expected_packet_sha256", "expected_invocation_sha256", "expected_sidecar_sha256"])
def test_every_replay_component_requires_independent_pin(captured, key):
    *_, data, pins = captured
    with pytest.raises(archive.RepositoryWorldPresentationInvocationError):
        archive.replay_repository_world_presentation_invocation(data, **(pins | {key: "0" * 64}))


@pytest.mark.parametrize("where", ["drop_usage", "change_usage", "drop_world_receipt", "changed_original", "changed_view", "changed_sidecar"])
def test_resealed_bundle_cannot_override_independently_retained_components(captured, where):
    *_, data, pins = captured
    body = json.loads(data)
    if where == "drop_usage": del body["invocation"]["usage"]
    elif where == "change_usage": body["invocation"]["usage"]["prompt_tokens"] = 100000
    elif where == "drop_world_receipt": del body["invocation"]["world_task_packet"]
    elif where == "changed_original": body["packet_artifact"]["prompt"] += "foreign"
    elif where == "changed_view": body["presentation"]["view_prompt"] += "foreign"
    else: body["presentation"]["sidecar"]["foreign"] = True
    changed = wire(body)
    with pytest.raises(archive.RepositoryWorldPresentationInvocationError):
        archive.replay_repository_world_presentation_invocation(changed, **(pins | {"expected_sha256": sha(changed)}))


@pytest.mark.parametrize("key,value", [
    ("provider", "grok_cli"), ("purpose", "planning"), ("completion_authority", True),
    ("model_prompt_sha256", "0" * 64), ("router_prompt_bytes", True),
    ("workspace_advisory_sha256", "0" * 64), ("native_prompt_sha256", "0" * 64),
    ("workspace", "/different/historical/workspace"), ("status", "private status body"),
])
def test_repinning_receipt_does_not_bypass_original_presentation_or_renderer(captured, key, value):
    _, _, receipt, args, _, _ = captured
    changed = deepcopy(receipt)
    changed[key] = value
    with pytest.raises(archive.RepositoryWorldPresentationInvocationError):
        archive.freeze_repository_world_presentation_invocation(**(args | dict(invocation_receipt=changed,
            expected_invocation_sha256=archive.invocation_receipt_sha256(changed))))


def test_original_packet_ids_and_presentation_receipt_are_not_caller_substitutable(captured):
    _, _, receipt, args, _, _ = captured
    for section, field in (("world_task_packet", "packet_id"), ("world_task_presentation", "sidecar_sha256"),
                           ("world_task_presentation", "dispatch_rechecked")):
        changed = deepcopy(receipt)
        changed[section][field] = False if field == "dispatch_rechecked" else "0" * 64
        with pytest.raises(archive.RepositoryWorldPresentationInvocationError):
            archive.freeze_repository_world_presentation_invocation(**(args | dict(invocation_receipt=changed,
                expected_invocation_sha256=archive.invocation_receipt_sha256(changed))))


def test_explicit_sidecar_pin_missing_wrong_foreign_or_mixed_inputs_refuse(captured):
    _, presentation, _, args, _, _ = captured
    base = args | dict(presentation=None, view_prompt=presentation.prompt, sidecar_bytes=presentation.sidecar_bytes)
    for changes in ({}, {"expected_sidecar_sha256": "0" * 64},
                    {"expected_sidecar_sha256": sha(presentation.sidecar_bytes), "sidecar_bytes": b"{}"},
                    {"presentation": presentation, "expected_sidecar_sha256": sha(presentation.sidecar_bytes)}):
        with pytest.raises(archive.RepositoryWorldPresentationInvocationError):
            archive.freeze_repository_world_presentation_invocation(**(base | changes))


def test_unissued_equal_byte_presentation_cannot_be_issued_by_history(captured):
    _, presentation, _, args, _, _ = captured
    duplicate = object.__new__(codec.PreparedRepositoryWorldPacketPresentation)
    for key in ("prompt", "sidecar_bytes", "_receipt_bytes", "_seal"):
        object.__setattr__(duplicate, key, getattr(presentation, key))
    with pytest.raises(archive.RepositoryWorldPresentationInvocationError):
        archive.freeze_repository_world_presentation_invocation(**(args | {"presentation": duplicate}))


def test_readonly_original_packet_is_not_provider_eligible_history(dispatch_case, required):
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet import prepare_repository_world_packet, repository_world_task_binding
    from test.api.test_repository_context_compiler import CORE, compile_context
    from test.api.test_repository_world_dispatch import TASK
    adapter = dispatch_case[2]._adapter
    effects, paths = ("repository_read",), ("calc.py",)
    compiled = compile_context(adapter, required, authority={**CORE["authority"], "world_task_packet":
        repository_world_task_binding("task:arithmetic", TASK, effects, paths)})
    readonly = prepare_repository_world_packet(adapter=adapter, compilation=compiled, generation=1,
        task_id="task:arithmetic", task_text=TASK, unresolved_obligations=("Observe source only.",),
        validation_commands=(), validation_profile="read_only_observation@1", allowed_effects=effects, write_paths=paths)
    original = freeze_repository_world_packet(readonly)
    with pytest.raises(archive.RepositoryWorldPresentationInvocationError, match="strict_provider_packet_required"):
        archive.freeze_repository_world_presentation_invocation(packet_artifact=original,
            expected_packet_sha256=sha(original), invocation_receipt={}, expected_invocation_sha256=sha(b"{}"),
            view_prompt="not inspected", sidecar_bytes=b"{}", expected_sidecar_sha256=sha(b"{}"))


def test_mixed_legacy_retention_and_unsupported_composition_do_not_replay(captured):
    _, _, receipt, args, _, _ = captured
    for key, value in (("world_packet_artifacts", {"schema": "foreign"}),
                       ("source_input_view", {"observed": True}), ("prompt_selection", {})):
        changed = deepcopy(receipt)
        changed[key] = value
        with pytest.raises(archive.RepositoryWorldPresentationInvocationError):
            archive.freeze_repository_world_presentation_invocation(**(args | dict(invocation_receipt=changed,
                expected_invocation_sha256=archive.invocation_receipt_sha256(changed))))


@pytest.mark.parametrize("raw", [b"", b"{}\n", b'{"a":1,"a":2}', b'{"x":NaN}', b'\xff'])
def test_noncanonical_or_malformed_archive_refuses_before_component_use(raw):
    with pytest.raises(archive.RepositoryWorldPresentationInvocationError):
        archive.replay_repository_world_presentation_invocation(raw, expected_sha256=sha(raw),
            expected_packet_sha256="0" * 64, expected_invocation_sha256="0" * 64,
            expected_sidecar_sha256="0" * 64)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), b"bytes", (1,), {1: "integer key"}, 1 << 4097])
def test_full_invocation_digest_rejects_nonfinite_or_non_json_populations(value):
    with pytest.raises(ValueError):
        archive.invocation_receipt_sha256({"observation": value})


def test_population_depth_cycles_and_byte_bounds_fail_without_partial_archives(monkeypatch):
    cyclic = []; cyclic.append(cyclic)
    with pytest.raises(ValueError): archive.invocation_receipt_sha256({"cycle": cyclic})
    with pytest.raises(ValueError): archive.invocation_receipt_sha256({"text": "é" * archive.MAX_INVOCATION_BYTES})
    with monkeypatch.context() as patch:
        patch.setattr(archive, "MAX_NODES", 10)
        with pytest.raises(ValueError): archive.invocation_receipt_sha256({"rows": [None] * 11})
    with monkeypatch.context() as patch:
        patch.setattr(archive, "MAX_DEPTH", 3)
        with pytest.raises(ValueError): archive.invocation_receipt_sha256({"rows": [[[[0]]]]})


def test_pin_is_full_canonical_receipt_not_pretty_file_bytes():
    receipt = {"b": None, "a": ["é", "é"], "seconds": 0.25}
    assert archive.invocation_receipt_sha256(receipt) == sha(wire(receipt))
    assert archive.invocation_receipt_sha256(receipt) != sha(json.dumps(receipt, indent=2).encode())


def test_aliased_nested_population_refuses_before_large_pending_allocation(monkeypatch):
    import tracemalloc
    # A depth-first traversal used to queue every sibling before visiting its
    # node cap. Construct the shared input before measuring traversal memory.
    value = None
    for _ in range(40):
        value = {str(index): None for index in range(499)} | {"last": value}
    monkeypatch.setattr(archive, "MAX_NODES", 1000)
    archive.invocation_receipt_sha256({})  # Exclude lazy dependency imports.
    tracemalloc.start()
    try:
        with pytest.raises(ValueError): archive.invocation_receipt_sha256(value)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 512 * 1024
