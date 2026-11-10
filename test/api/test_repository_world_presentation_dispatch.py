"""Lossless presentation reaches the existing strict authored provider seam.

Native source/session/compiler and router are real; provider discovery, calls
and usage are explicitly authored. This suite proves no live model quality,
provider ingestion, task validation success, billing or completion.
"""
from copy import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime import repository_world_packet_presentation as codec
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet import (
    PreparedRepositoryWorldPacket, prepare_repository_world_packet, repository_world_task_binding,
)
from test.api.test_repository_world_dispatch import (
    TASK, dispatch_case, world_session, required, scheduler, counter, invocation_receipt,  # noqa: F401
)


@pytest.fixture
def presented(dispatch_case):
    presentation = codec.prepare_repository_world_packet_presentation(dispatch_case[2].prompt)
    assert presentation.receipt()["codec_supported"] is True
    return dispatch_case, presentation


def invoke(presented, **changes):
    case, presentation = presented
    options = dict(prompt=TASK, provider="codex_cli", model="pinned", timeout=30,
        max_output_tokens=128, world_task_packet=case[2], world_task_presentation=presentation)
    return runner.run(**(options | changes))


def native_view_business(presentation):
    """Independent documented marker decoder for full required-content census."""
    view = json.loads(presentation.prompt.split("\n", 2)[2])
    pool = []
    def expand(value):
        if type(value) is dict:
            if set(value) == {"$fact"}: return pool[value["$fact"]]
            if set(value) == {"$json"}: return json.dumps(expand(value["$json"]), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            if set(value) == {"$literal"}: return {key: expand(child) for key, child in value["$literal"]}
            return {key: expand(child) for key, child in value.items()}
        if type(value) is list: return [expand(child) for child in value]
        return value
    for fact in view["shared_facts"]: pool.append(expand(fact))
    return expand(view["packet"])


def test_supported_view_is_complete_actual_authored_provider_input_with_separate_original_pins(presented):
    case, presentation = presented
    output, receipt = invoke(presented)
    assert output == "Authored provider returned; owner validation remains required."
    expected, advice = runner.render_model_prompt(prompt=presentation.prompt, purpose="coding", workspace=case[0], world_packet=True)
    actual, options = case[3]["seen"][0]
    assert actual == expected and advice
    assert actual != runner.render_model_prompt(prompt=case[2].prompt, purpose="coding", workspace=case[0], world_packet=True)[0]
    assert receipt["native_prompt_sha256"] == hashlib.sha256(TASK.encode()).hexdigest()
    assert receipt["world_task_packet"]["prompt_sha256"] == hashlib.sha256(case[2].prompt.encode()).hexdigest()
    assert receipt["router_prompt_sha256"] == hashlib.sha256(presentation.prompt.encode()).hexdigest()
    assert receipt["model_prompt_sha256"] == hashlib.sha256(actual.encode()).hexdigest()
    row, original = receipt["world_task_presentation"], presentation.receipt()
    assert row["original_prompt_sha256"] == original["original_prompt_sha256"]
    assert row["presented_prompt_sha256"] == original["presented_prompt_sha256"]
    assert row["sidecar_sha256"] == hashlib.sha256(presentation.sidecar_bytes).hexdigest()
    assert row["inverse_verified"] is row["dispatch_rechecked"] is True
    assert row["schema"] == "router-world-task-presentation@1" and row["codec_supported"] is True
    assert row["profile"] == row["model_view_mode"] == codec.MODE
    for key in ("proof_authority", "execution_authority", "completion_authority", "omission_authority",
                "provider_ingestion_verified", "total_token_savings_measured", "source_freshness_verified"):
        assert row[key] is False
    assert options["allow_local_fallback"] is options["allow_cross_provider_fallback"] is False
    assert options["side_effecting"] is True
    assert receipt["usage"]["prompt_tokens"] == 15 and receipt["completion_authority"] is False
    assert TASK not in json.dumps(row) and str(case[0]) not in json.dumps(row)


def test_original_effects_validations_proofs_and_all_required_roles_remain_model_visible(presented):
    case, presentation = presented
    original = json.loads(case[2].prompt.split("\n", 1)[1])["packet"]
    view = native_view_business(presentation)
    assert view["requirements"] == original["requirements"]
    assert view["effect_scope"] == original["effect_scope"]
    assert view["requirements"]["require_validation"] is True
    assert len(view["members"]) == len(original["members"])
    for actual, expected in zip(view["members"], original["members"]):
        assert actual == {key: value for key, value in expected.items() if key not in {"item_id", "digest"}}
    invoke(presented)
    assert case[3]["seen"][0][0].endswith(presentation.prompt)


def test_none_preserves_original_rendering_and_has_no_new_receipt_fields(dispatch_case):
    options = dict(prompt=TASK, provider="codex_cli", model="pinned", timeout=30, max_output_tokens=128, world_task_packet=dispatch_case[2])
    _, first = runner.run(**options)
    _, explicit = runner.run(**options, world_task_presentation=None)
    assert dispatch_case[3]["seen"][0][0] == dispatch_case[3]["seen"][1][0]
    assert "world_task_presentation" not in first and "world_task_presentation" not in explicit
    assert first["router_prompt_sha256"] == explicit["router_prompt_sha256"] == hashlib.sha256(dispatch_case[2].prompt.encode()).hexdigest()


def test_complete_rendered_view_budget_uses_existing_counter_and_suffix(presented):
    case, presentation = presented
    expected, _ = runner.render_model_prompt(prompt=presentation.prompt, purpose="coding", workspace=case[0], world_packet=True)
    counted = []
    def count(text): counted.append(text); return len(text.encode())
    _, receipt = invoke(presented, prompt_token_counter=counter(count_tokens=count), max_input_tokens=len(expected.encode()))
    assert counted == [expected] and case[3]["seen"][0][0] == expected
    assert receipt["world_task_packet"]["prompt_measurement"]["selected_input_tokens"] == len(expected.encode())
    assert receipt["world_task_presentation"]["original_prompt_bytes"] != receipt["model_prompt_bytes"]


def test_provider_failure_still_publishes_presented_input_accounting_without_completion(presented, monkeypatch, capsys):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
    def fail(prompt, **_options):
        presented[0][3]["seen"].append(prompt)
        set_last_cli_observation("codex_cli", {"exit_code": 7, "prompt_tokens": 21})
        raise RuntimeError("authored provider failure")
    monkeypatch.setattr(llm_router, "generate_text", fail)
    with pytest.raises(RuntimeError, match="authored provider failure"): invoke(presented)
    receipt = invocation_receipt(capsys)
    assert receipt["status"] == "failed" and receipt["world_task_packet"]["provider_attempted"] is True
    assert receipt["world_task_presentation"]["dispatch_rechecked"] is True
    assert receipt["model_prompt_sha256"] == hashlib.sha256(presented[0][3]["seen"][0].encode()).hexdigest()
    assert receipt["completion_authority"] is False and "context_delivery_observation" in receipt


@pytest.mark.parametrize("kind", ["dict", "empty", "partial", "copy", "subclass", "foreign_original", "fallback"])
def test_malformed_detached_or_foreign_presentation_never_discovers_provider(presented, monkeypatch, kind):
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    _case, valid = presented
    if kind == "dict": bad = valid.receipt()
    elif kind == "empty": bad = object.__new__(codec.PreparedRepositoryWorldPacketPresentation)
    elif kind == "partial":
        bad = object.__new__(codec.PreparedRepositoryWorldPacketPresentation); object.__setattr__(bad, "_seal", codec._SEAL)
    elif kind == "copy":
        bad = object.__new__(codec.PreparedRepositoryWorldPacketPresentation)
        for key in ("prompt", "sidecar_bytes", "_receipt_bytes", "_seal"): object.__setattr__(bad, key, getattr(valid, key))
    elif kind == "subclass":
        class Other(codec.PreparedRepositoryWorldPacketPresentation): pass
        bad = object.__new__(Other)
    elif kind == "fallback": bad = codec.prepare_repository_world_packet_presentation("Original exact unrecognized transport.")
    else:
        altered = presented[0][2].prompt.replace("Current task invariants", "Foreign task invariants")
        assert altered != presented[0][2].prompt
        bad = codec.prepare_repository_world_packet_presentation(altered)
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda: pytest.fail("invalid presentation reached discovery"))
    with pytest.raises(ValueError): invoke(presented, world_task_presentation=bad)
    assert presented[0][3]["seen"] == []


def test_presentation_without_original_packet_is_not_a_provider_path(presented, monkeypatch):
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda: pytest.fail("missing original packet reached discovery"))
    with pytest.raises(ValueError, match="original provider-eligible packet"):
        invoke(presented, world_task_packet=None)
    assert presented[0][3]["seen"] == []


def test_presentation_refuses_old_original_only_artifact_format_before_discovery(presented, monkeypatch, tmp_path):
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda: pytest.fail("unsupported artifact combination reached discovery"))
    with pytest.raises(ValueError, match="original-only packet artifact retention"):
        invoke(presented, world_packet_artifact_dir=tmp_path / "must-not-be-created")
    assert not (tmp_path / "must-not-be-created").exists() and presented[0][3]["seen"] == []


@pytest.mark.parametrize("changes", [{"provider": "grok_cli", "model": "grok-4.7"}, {"purpose": "planning"},
    {"semantic_metadata_view": "capsule-columns@1"}, {"semantic_transport_schema": "supervisor-semantic-router-input@2"},
    {"source_input_policy": "query-reference@1"}, {"census_only": True}])
def test_presentation_cannot_expand_existing_world_route_profiles(presented, monkeypatch, changes):
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda: pytest.fail("cross-profile route reached discovery"))
    with pytest.raises(ValueError, match="independently selected Codex coding route"):
        invoke(presented, **changes)
    assert presented[0][3]["seen"] == []


@pytest.mark.parametrize("field", ["prompt", "sidecar_bytes", "_receipt_bytes"])
def test_equal_byte_presentation_substitution_before_capture_refuses(presented, field):
    presentation = presented[1]
    old = getattr(presentation, field)
    new = old.encode().decode() if type(old) is str else bytes(bytearray(old))
    assert new == old and new is not old
    object.__setattr__(presentation, field, new)
    try:
        with pytest.raises(ValueError, match="original_presentation_replaced"): invoke(presented)
        assert presented[0][3]["seen"] == []
    finally: object.__setattr__(presentation, field, old)


@pytest.mark.parametrize("field", ["prompt", "sidecar_bytes", "_receipt_bytes"])
def test_provider_initialization_cannot_replace_captured_equal_byte_presentation(presented, monkeypatch, capsys, field):
    from ipfs_accelerate_py import llm_router
    presentation = presented[1]
    old = getattr(presentation, field)
    new = old.encode().decode() if type(old) is str else bytes(bytearray(old))
    def initialize(*_a, **_kw): object.__setattr__(presentation, field, new); return object()
    monkeypatch.setattr(llm_router, "get_llm_provider", initialize)
    try:
        with pytest.raises(ValueError, match="original_presentation_replaced"): invoke(presented)
        assert presented[0][3]["seen"] == []
        receipt = invocation_receipt(capsys)
        assert receipt["world_task_packet"]["provider_attempted"] is False
        assert receipt["world_task_presentation"]["dispatch_rechecked"] is False
    finally: object.__setattr__(presentation, field, old)


@pytest.mark.parametrize("field", ["prompt", "_capsule", "_workspace", "receipt", "validate_current", "require_provider_dispatch"])
def test_provider_initialization_cannot_rewrite_or_shadow_original_packet(presented, monkeypatch, field):
    from ipfs_accelerate_py import llm_router
    packet = presented[0][2]
    original = getattr(packet, field)
    if field == "prompt": replacement = original.encode().decode()
    elif field == "_capsule": replacement = replace(original)
    elif field == "_workspace": replacement = Path(str(original))
    else: replacement = lambda *_a, **_kw: packet.receipt()
    def initialize(*_a, **_kw): object.__setattr__(packet, field, replacement); return object()
    monkeypatch.setattr(llm_router, "get_llm_provider", initialize)
    try:
        with pytest.raises(ValueError, match="original packet custody changed"): invoke(presented)
        assert presented[0][3]["seen"] == []
    finally:
        if field in {"receipt", "validate_current", "require_provider_dispatch"}: object.__delattr__(packet, field)
        else: object.__setattr__(packet, field, original)


def test_provider_initialization_source_edit_is_refused_by_original_closing_source_fence(presented, monkeypatch, capsys):
    from ipfs_accelerate_py import llm_router
    path = presented[0][0] / "calc.py"
    original = path.read_bytes()
    def initialize(*_a, **_kw): path.write_bytes(original + b"# changed\n"); return object()
    monkeypatch.setattr(llm_router, "get_llm_provider", initialize)
    try:
        with pytest.raises(ValueError, match="source|changed|refresh"): invoke(presented)
        assert presented[0][3]["seen"] == []
        receipt = invocation_receipt(capsys)
        assert receipt["world_task_packet"]["dispatch_rechecked"] is False
        assert receipt["world_task_presentation"]["dispatch_rechecked"] is False
    finally: path.write_bytes(original)


def test_codec_operations_finish_before_original_final_source_fence(presented, monkeypatch):
    events = []
    original_codec = codec.PreparedRepositoryWorldPacketPresentation.validate_original
    original_packet = PreparedRepositoryWorldPacket.validate_current
    def codec_check(self, original_prompt): events.append("codec"); return original_codec(self, original_prompt)
    def packet_check(self, workspace, task_text): events.append("packet"); return original_packet(self, workspace, task_text)
    monkeypatch.setattr(codec.PreparedRepositoryWorldPacketPresentation, "validate_original", codec_check)
    monkeypatch.setattr(PreparedRepositoryWorldPacket, "validate_current", packet_check)
    presented[0][3]["callback"] = lambda *_a: events.append("provider")
    invoke(presented)
    assert events[-2:] == ["packet", "provider"]
    assert "codec" in events and events.index("codec") < events.index("packet")


def test_readonly_original_is_refused_before_any_presentation_handling_or_discovery(dispatch_case, required, monkeypatch):
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    from test.api.test_repository_context_compiler import CORE, compile_context
    adapter = dispatch_case[2]._adapter
    effects, paths = ("repository_read",), ("calc.py",)
    compiled = compile_context(adapter, required, authority={**CORE["authority"], "world_task_packet": repository_world_task_binding(
        "task:arithmetic", TASK, effects, paths)})
    readonly = prepare_repository_world_packet(adapter=adapter, compilation=compiled, generation=1,
        task_id="task:arithmetic", task_text=TASK, unresolved_obligations=("Observe source only.",),
        validation_commands=(), validation_profile="read_only_observation@1", allowed_effects=effects, write_paths=paths)
    def forbidden(*_a, **_kw): pytest.fail("read-only packet reached presentation or provider discovery")
    monkeypatch.setattr(runner, "_world_task_presentation_material", forbidden)
    monkeypatch.setattr(intelligence_index, "discover_available_providers", forbidden)
    before = dict(os.environ)
    with pytest.raises(ValueError, match="readonly_provider_dispatch_refused"):
        runner.run(prompt=TASK, provider="codex_cli", model="pinned", timeout=30, max_output_tokens=128,
            world_task_packet=readonly, world_task_presentation=object())
    assert dict(os.environ) == before and dispatch_case[3]["seen"] == []
