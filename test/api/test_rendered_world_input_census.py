"""World byte census with real pure packet rendering; no owner/provider claims."""
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.prompt.delta_task_packet import (
    DeltaTaskPacketRequest, PacketBudget, admit_provider_packet, build_delta_task_packet,
)
from ipfs_accelerate_py.agent_supervisor.runtime.coding_reply_contract import apply_coding_reply_contract
from ipfs_accelerate_py.agent_supervisor.runtime.rendered_input_census import census_rendered_input
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet import GUIDANCE, SCHEMA
from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import public_replay_without_metadata


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@pytest.fixture
def world_input():
    # Authored historical identities. Construction uses genuine packet code,
    # with metadata mirrors suppressed; no native index or scheduler exists.
    hidden = canonical({"schema": "semantic-router-input@1", "raw_sources": {
        "private-name.py": "private source body\n"}, "manifest": {}})
    with public_replay_without_metadata():
        request = DeltaTaskPacketRequest(task_cid="task:authored", repository_id="repo:authored",
            tree_id="sha256:" + "1" * 64, context_cid="context:authored",
            policy_id="policy:authored", policy_digest="sha256:" + "2" * 64,
            task_summary="Review the recorded task", open_obligations=(
                {"obligation_id": "action:0", "summary": "Preserve current behavior"},),
            evidence=({"evidence_id": "context:0", "summary": "Recorded reference",
                "reference": {"summary": hidden}},),
            validations=("python -m pytest -q",), allowed_effects=("read", "modify"),
            write_paths=("private-name.py",), budget=PacketBudget(
                max_bytes=65536, max_tokens=16384, max_rows=64, max_items=64))
        packet = admit_provider_packet(build_delta_task_packet(request))
    envelope = {"schema": SCHEMA, "delivery_mode": "self_contained", "packet": packet}
    # Multibyte advisory text deliberately distinguishes byte and character offsets.
    advisory = "Recorded workspace advisory: λ雪\n"
    prompt = advisory + GUIDANCE + "\n" + canonical(envelope)
    return {"advisory": advisory, "prompt": prompt, "envelope": envelope, "hidden": hidden}


def observe(fixture, **kwargs):
    return census_rendered_input(model_prompt=fixture["prompt"],
        workspace_advisory=fixture["advisory"], invocation_id="a" * 32,
        input_representation="world_task_packet", **kwargs)


def test_world_guidance_and_exact_final_message_have_disjoint_byte_partitions(world_input):
    result = observe(world_input)
    part = result["disjoint_byte_partition"]
    assert part == {"workspace_advisory": len(world_input["advisory"].encode()),
        "world_packet_guidance": len((GUIDANCE + "\n").encode()),
        "json_prefix": len(canonical(world_input["envelope"]).encode()), "router_suffix": 0}
    assert sum(part.values()) == result["model_prompt_bytes"] == len(world_input["prompt"].encode())
    assert result["model_prompt_sha256"] == sha(world_input["prompt"])
    assert result["world_packet_guidance"] == {
        "bytes": len((GUIDANCE + "\n").encode()), "sha256": sha(GUIDANCE + "\n")}
    assert result["byte_partition_reconciles"] is result["canonical_partition_reconciles"] is True
    assert result["json_prefix"]["is_canonical"] is True


def test_reference_summary_is_never_decoded_as_whole_file_source(world_input):
    result = observe(world_input)
    inline = result["inline_sources"]
    assert inline["status"] == "unavailable"
    assert inline["reason"] == "unsupported_world_reference_source_carrier"
    assert inline["entries"] is None
    assert result["nested_json_strings_inspected"] is None
    assert result["nested_json_inspection_scope"] == "declared_complete_semantic_context_only"
    assert result["repetition_census_nested_json_strings_inspected"] is False
    assert result["content_tokens"] is result["provider_usage_tokens"] is None
    assert result["token_savings_established"] is result["omission_authority"] is False
    exported = json.dumps(result)
    assert "private-name.py" not in exported and "private source body" not in exported


def test_reply_suffix_is_opaque_and_separate_from_world_json(world_input):
    original = world_input["prompt"]
    world_input["prompt"], receipt = apply_coding_reply_contract(model_prompt=original,
        mode="ordinary-completion@1", purpose="coding", provider="codex_cli")
    result = observe(world_input)
    assert result["disjoint_byte_partition"]["router_suffix"] == receipt["instruction_bytes"]
    assert result["model_prompt_bytes"] == len(original.encode()) + receipt["instruction_bytes"]
    assert result["json_prefix"]["sha256"] == sha(canonical(world_input["envelope"]))
    # Native output-schema bytes are delivered separately and are never added.
    assert "native_output_schema_bytes" not in result


def test_world_mode_is_explicit_and_default_census_shape_remains_unchanged(world_input):
    with pytest.raises(ValueError):
        census_rendered_input(model_prompt=world_input["prompt"],
            workspace_advisory=world_input["advisory"], invocation_id="a" * 32)
    result = census_rendered_input(model_prompt='{"goal":"literal"}',
        workspace_advisory="", invocation_id="ordinary")
    assert "input_representation" not in result
    assert "world_packet_guidance" not in result
    assert set(result["disjoint_byte_partition"]) == {"workspace_advisory", "json_prefix", "router_suffix"}


@pytest.mark.parametrize("change", ["guidance", "newline", "advisory", "outer_schema",
    "delivery", "extra_outer", "packet_schema", "authority", "members", "noncanonical"])
def test_world_census_refuses_unbound_or_malformed_carriers(world_input, change):
    if change == "guidance":
        world_input["prompt"] = world_input["prompt"].replace(GUIDANCE, GUIDANCE + " ", 1)
    elif change == "newline":
        world_input["prompt"] = world_input["prompt"].replace(GUIDANCE + "\n", GUIDANCE + "\r\n", 1)
    elif change == "advisory":
        world_input["advisory"] += " "
    elif change == "noncanonical":
        world_input["prompt"] = world_input["advisory"] + GUIDANCE + "\n" + json.dumps(world_input["envelope"])
    else:
        envelope = world_input["envelope"]
        if change == "outer_schema":
            envelope["schema"] += "-future"
        elif change == "delivery":
            envelope["delivery_mode"] = "parent_reference"
        elif change == "extra_outer":
            envelope["raw_sources"] = {}
        elif change == "packet_schema":
            envelope["packet"]["schema"] += "-future"
        elif change == "authority":
            envelope["packet"]["completion_authority"] = 0
        else:
            envelope["packet"]["members"] = {}
        world_input["prompt"] = world_input["advisory"] + GUIDANCE + "\n" + canonical(envelope)
    with pytest.raises(ValueError):
        observe(world_input)


@pytest.mark.parametrize("representation", [True, {}, "world_task_packet@2", "native"])
def test_unknown_representation_does_not_enable_world_parsing(world_input, representation):
    with pytest.raises(ValueError):
        census_rendered_input(model_prompt=world_input["prompt"],
            workspace_advisory=world_input["advisory"], invocation_id="a" * 32,
            input_representation=representation)
