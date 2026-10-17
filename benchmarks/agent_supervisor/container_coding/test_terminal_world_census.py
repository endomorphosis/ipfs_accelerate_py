"""Authored historical bindings, real compiler/world renderer/replay, no live owner.

These tests do not establish provider delivery, native freshness, or savings.
No source index, scheduler, provider, or metadata controller is instantiated.
"""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
from benchmarks.agent_supervisor.container_coding.test_terminal_context_audit import compiled as legacy_compiled
from ipfs_accelerate_py.agent_supervisor.context.context_compiler import ContextCompiler
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget, ContextCapsule
from ipfs_accelerate_py.agent_supervisor.runtime.coding_reply_contract import apply_coding_reply_contract
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_invocation_replay import (
    freeze_repository_world_invocation, replay_repository_world_invocation,
)
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet import (
    ARTIFACT_SCHEMA, GUIDANCE, _plain, _render_repository_world_packet, repository_world_task_binding,
)
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet_replay import _canonical_bytes
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import render_model_prompt
from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import public_replay_without_metadata


def sha(raw):
    return hashlib.sha256(raw if type(raw) is bytes else raw.encode("utf-8")).hexdigest()


def authored_historical_world(tmp_path, *, reply_mode="legacy"):
    # The retained original fixture supplies genuine chunk/manifest joins, but
    # its repository ownership labels are intentionally authored historical data.
    with public_replay_without_metadata():
        fixture = legacy_compiled.__wrapped__(tmp_path)
        original = ContextCapsule.from_dict(json.loads(fixture["capsule"].read_text()))
        effects, paths = ("read", "modify"), ("bottle.py",)
        binding = {"generation": 1, "repository_id": original.repository_id,
            "source_tree_sha256": "1" * 64, "semantic_state_cid": "cid:authored-state",
            "semantic_root_cid": "cid:semantic", "capsule_index_cid": "cid:authored-index",
            "required_reference_ids": sorted(row.reference_id for row in original.evidence if row.required),
            "required_coverage_ids": sorted({value for row in original.evidence if row.required
                                             for value in row.coverage_ids})}
        authority = {"repository_context": binding, "task": {"world_task_packet":
            repository_world_task_binding("TB-CODE-TASK", fixture["query"], effects, paths)}}
        compiled = ContextCompiler(ContextBudget(max_input_tokens=80000, max_items=128,
            max_item_bytes=32768, max_serialized_bytes=262144)).compile(
            repository_id=original.repository_id, tree_id="sha256:" + binding["source_tree_sha256"],
            objective_id=original.objective_id, objective_revision=original.objective_revision,
            policy_id=original.policy_id, policy_revision=original.policy_revision,
            caller=original.caller, stage=original.stage, goal=original.goal, authority=authority,
            scope=original.scope, acceptance=original.acceptance, evidence=original.evidence)
        recipe = dict(generation=1, task_id="TB-CODE-TASK", task_text=fixture["query"],
            unresolved_obligations=("Preserve the public task behavior",),
            validation_commands=("python -m pytest -q",), proof_obligations=(), acceptance_ids=(),
            allowed_effects=effects, write_paths=paths, current_world=None, max_bytes=262144)
        world, packet_receipt, native_packet = _render_repository_world_packet(compilation=compiled, **recipe)
        packet_artifact = _canonical_bytes(_plain(dict(schema=ARTIFACT_SCHEMA, mode="historical_only",
            capsule=compiled.capsule.to_record(), compilation_receipt=compiled.receipt.to_record(),
            recipe=recipe, native_packet=native_packet, prompt=world, packet_receipt=packet_receipt)))
        workspace = fixture["receipts"][0]["workspace"]
        model, advisory = render_model_prompt(prompt=world, purpose="coding",
            workspace=Path(workspace), world_packet=True)
        model, reply = apply_coding_reply_contract(model_prompt=model, mode=reply_mode,
            purpose="coding", provider="codex_cli")
        receipt = dict(schema="router-implementation-invocation@1", phase="coding",
            invocation_id="a" * 32, provider="codex_cli", model="authored-historical-model",
            purpose="coding", workspace=workspace, completion_authority=False,
            prompt_sha256=sha(fixture["query"]), prompt_bytes=len(fixture["query"].encode()),
            native_prompt_sha256=sha(fixture["query"]), native_prompt_bytes=len(fixture["query"].encode()),
            router_prompt_sha256=sha(world), router_prompt_bytes=len(world.encode()),
            model_prompt_sha256=sha(model), model_prompt_bytes=len(model.encode()),
            workspace_advisory_sha256=sha(advisory), workspace_advisory_bytes=len(advisory.encode()),
            world_task_packet={**packet_receipt, "dispatch_rechecked": False,
                "provider_attempted": False, "prompt_measurement": None})
        if reply is not None:
            receipt["coding_reply_contract"] = reply
        raw = freeze_repository_world_invocation(packet_artifact=packet_artifact,
            expected_packet_sha256=sha(packet_artifact), invocation_receipt=receipt)
        replay = replay_repository_world_invocation(raw, expected_sha256=sha(raw))
        assert replay.model_prompt == model
    artifacts = fixture["state"] / "world-invocations"
    artifacts.mkdir()
    (artifacts / (receipt["invocation_id"] + ".json")).write_bytes(raw)
    receipt["world_packet_replay_artifact"] = {
        "schema": "repository-world-invocation-artifact-reference@1",
        "invocation_id": receipt["invocation_id"], "artifact_sha256": sha(raw),
        "artifact_bytes": len(raw), "packet_artifact_sha256": sha(packet_artifact)}
    fixture.update(legacy_receipt=fixture["receipts"][0], receipts=[receipt],
        model=model, advisory=advisory, world_prompt=world, reply=reply)
    return fixture


@pytest.fixture
def historical_world(tmp_path):
    return authored_historical_world(tmp_path)


def observe(fixture):
    return audit.audit_terminal_context(state=fixture["state"], receipts=fixture["receipts"],
        workspace_root=fixture["workspace_root"])


def test_actual_historical_world_replay_censuses_final_message_not_compiled_capsule(historical_world):
    result = observe(historical_world)
    assert result["all_observed_coding_inputs_verified"] is True, result
    census = result["rendered_input_census"]
    assert census["status"] == "complete_for_verified_inputs"
    assert census["verified_invocation_count"] == census["observed_invocation_count"] == 1
    row, = census["observations"]
    assert row["model_prompt_sha256"] == sha(historical_world["model"])
    assert row["input_representation"] == "world_task_packet"
    assert row["disjoint_byte_partition"]["world_packet_guidance"] == len((GUIDANCE + "\n").encode())
    match, = result["matches"]
    assert row["model_prompt_sha256"] != match["compiled_capsule_prompt_sha256"]
    assert match["native_prompt_bytes"] == len(historical_world["query"].encode())
    assert row["inline_sources"]["reason"] == "unsupported_world_reference_source_carrier"
    assert row["inline_sources"]["entries"] is None
    assert result["provider_calls"] == 0
    assert match["world_packet_replay"]["provider_delivery_verified"] is False
    assert not historical_world["workspace_root"].exists()
    exported = json.dumps(result)
    assert "private source query" not in exported
    assert historical_world["receipts"][0]["workspace"] not in exported


def test_actual_world_reply_contract_suffix_remains_separate(tmp_path):
    fixture = authored_historical_world(tmp_path, reply_mode="ordinary-completion@1")
    result = observe(fixture)
    assert result["all_observed_coding_inputs_verified"] is True, result
    row, = result["rendered_input_census"]["observations"]
    assert row["disjoint_byte_partition"]["router_suffix"] == fixture["reply"]["instruction_bytes"]
    assert sum(row["disjoint_byte_partition"].values()) == len(fixture["model"].encode())
    assert row["provider_usage_tokens"] is None


def test_world_and_legacy_censuses_merge_without_losing_population(historical_world):
    historical_world["receipts"].append(historical_world["legacy_receipt"])
    result = observe(historical_world)
    assert result["all_observed_coding_inputs_verified"] is True, result
    census = result["rendered_input_census"]
    assert census["status"] == "complete_for_verified_inputs"
    assert census["verified_invocation_count"] == census["observed_invocation_count"] == 2
    assert {row["invocation_id_sha256"] for row in census["observations"]} == {
        sha(receipt["invocation_id"]) for receipt in historical_world["receipts"]}
    assert sum(row.get("input_representation") == "world_task_packet" for row in census["observations"]) == 1


@pytest.mark.parametrize("malformed", ["raise", "object", "array", "identity", "nan"])
def test_optional_world_census_failure_preserves_verified_base_audit(historical_world, monkeypatch, malformed):
    from ipfs_accelerate_py.agent_supervisor.runtime import rendered_input_census as census
    original = census.census_rendered_input

    def broken(**kwargs):
        if malformed == "raise":
            raise ValueError("private content must not escape")
        if malformed == "object":
            return {}
        if malformed == "array":
            return []
        row = original(**kwargs)
        if malformed == "identity":
            row["model_prompt_sha256"] = "0" * 64
        else:
            row["json_nodes"] = float("nan")
        return row

    monkeypatch.setattr(census, "census_rendered_input", broken)
    result = observe(historical_world)
    assert result["all_observed_coding_inputs_verified"] is True, result
    assert result["read_errors"] == []
    report = result["rendered_input_census"]
    assert report["verified_invocation_count"] == 1
    assert report["observations"] == [] and report["observed_invocation_count"] == 0
    assert len(report["unavailable"]) == 1
    assert "private content" not in json.dumps(result)


def test_aggregate_census_budget_yields_without_demoting_mixed_audit(historical_world, monkeypatch):
    historical_world["receipts"].append(historical_world["legacy_receipt"])
    monkeypatch.setattr(audit, "MAX_CENSUS_BYTES", 0)
    result = observe(historical_world)
    assert result["all_observed_coding_inputs_verified"] is True, result
    census = result["rendered_input_census"]
    assert census["verified_invocation_count"] == 2
    assert census["observed_invocation_count"] == 0
    assert len(census["unavailable"]) == 2


def test_foreign_final_model_pin_never_produces_world_census(historical_world):
    historical_world["receipts"][0]["model_prompt_sha256"] = "0" * 64
    result = observe(historical_world)
    assert result["any_model_input_verified"] is False
    assert result["rendered_input_census"]["observations"] == []
    assert result["rendered_input_census"]["verified_invocation_count"] == 0


def test_world_artifact_tamper_never_produces_world_census(historical_world):
    path = historical_world["state"] / "world-invocations" / ("a" * 32 + ".json")
    path.write_bytes(path.read_bytes() + b" ")
    result = observe(historical_world)
    assert result["any_model_input_verified"] is False
    assert result["rendered_input_census"]["observations"] == []


def test_bounded_historical_child_exports_world_census(historical_world):
    result = audit.collect_terminal_context_audit(state=historical_world["state"],
        receipts=historical_world["receipts"], workspace_root=historical_world["workspace_root"])
    assert result["all_observed_coding_inputs_verified"] is True, result
    assert result["rendered_input_census"]["observed_invocation_count"] == 1
    assert result["rendered_input_census"]["observations"][0]["input_representation"] == "world_task_packet"
    assert result["provider_calls"] == 0


def test_optional_census_yields_overall_result_budget(historical_world, monkeypatch):
    baseline = observe(historical_world)
    without = copy.deepcopy(baseline)
    without.pop("rendered_input_census")
    monkeypatch.setattr(audit, "MAX_RESULT_BYTES", len(json.dumps(without, sort_keys=True).encode()) + 1)
    result = observe(historical_world)
    assert result["all_observed_coding_inputs_verified"] is True
    assert "rendered_input_census" not in result
