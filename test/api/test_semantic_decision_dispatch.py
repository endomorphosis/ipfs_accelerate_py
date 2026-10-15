"""Authenticated native artifacts reach explicit packet selection and expansion."""

import copy
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import build_text_context_references
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_projection import SemanticProjectionBindings
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_packet import SemanticDecisionPolicy
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_projection_dispatch import (
    expand_semantic_router_decision_packet, project_semantic_router_prompt,
)

pytest_plugins = ("test.api.test_semantic_router_translation",)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _packet(prompt):
    wire, _ = json.JSONDecoder().raw_decode(prompt)
    return json.loads("".join(row["summary"] for row in wire["evidence"]
                             if row["kind"] == "semantic-context"))


def _policy(parent):
    identities = tuple(row["capsule_cid"] for row in parent["capsules"])
    assert len(identities) > 1
    return SemanticDecisionPolicy(required_capsule_ids=identities[:1], optional_capsule_ids=identities[1:])


def _case(native):
    root, artifact, prompt, _ = native
    parent = json.loads(artifact.read_text())
    policy = _policy(parent)
    projected = project_semantic_router_prompt(prompt=prompt, repository=root, decision_policy=policy)
    view = _packet(projected.projected_prompt)
    return root, artifact, prompt, parent, policy, projected, view


def test_real_producer_packet_preserves_task_source_admissions_and_assurance(native):
    root, artifact, prompt, parent, policy, projected, view = _case(native)
    expected = copy.deepcopy(parent)
    expected["capsules"] = parent["capsules"][:1]
    del expected["pack"]["token_estimate"]
    del expected["pack"]["pack"]["token_totals"]
    assert view["required_context"] == expected
    assert view["selection_policy"] == policy.to_dict()
    assert view["selection_summary"]["parent_capsule_count"] == len(parent["capsules"])
    old, old_end = json.JSONDecoder().raw_decode(prompt)
    new, new_end = json.JSONDecoder().raw_decode(projected.projected_prompt)
    assert {key: value for key, value in old.items() if key != "evidence"} == {
        key: value for key, value in new.items() if key != "evidence"
    }
    assert prompt[old_end:] == projected.projected_prompt[new_end:]
    assert artifact.read_text() == _json(parent)
    assert projected.receipt["provider_called"] is False
    assert projected.receipt["provider_expansion_registered"] is False
    assert projected.receipt["total_session_token_savings"] is None
    expanded = expand_semantic_router_decision_packet(
        repository=root, view=view, reference_id=view["deferred_references"][0]["reference_id"],
        bindings=SemanticProjectionBindings(**view["origin"]), policy=policy,
        count_tokens=lambda text: 1, max_bytes=65536,
    )
    fields = json.loads(expanded.summary)["fields"]
    assert [row["value"] for row in fields if row["field_selector"][0] == "capsules"] == parent["capsules"][1:]
    assert expanded.metadata["required"] is False


@pytest.mark.parametrize("interleaved", [False, True])
def test_expansion_authenticates_current_source_before_and_after_counter(native, interleaved):
    root, _, _, _, policy, _, view = _case(native)
    path = root / "mod.py"
    source = path.read_text()

    def counter(_text):
        if interleaved:
            path.write_text(source + "# concurrent source change\n")
        return 1

    if not interleaved:
        path.write_text(source + "# source changed before expansion\n")
    with pytest.raises(ValueError, match="stale"):
        expand_semantic_router_decision_packet(
            repository=root, view=view, reference_id=view["deferred_references"][0]["reference_id"],
            bindings=SemanticProjectionBindings(**view["origin"]), policy=policy,
            count_tokens=counter, max_bytes=65536,
        )


def test_packet_is_not_admitted_to_cli_codec_without_trusted_policy_bridge(native):
    root, _, _, _, _, projected, _ = _case(native)
    with pytest.raises(ValueError):
        codec.encode_semantic_router_prompt(
            prompt=projected.projected_prompt, repository=root,
            slot_policy_revision="closed-typed-identifiers@2",
        )


def test_native_verified_expansion_can_fetch_one_deferred_capsule(native):
    root, _, _, parent, policy, _, view = _case(native)
    wanted = parent["capsules"][1]
    expanded = expand_semantic_router_decision_packet(
        repository=root, view=view, reference_id=view["deferred_references"][0]["reference_id"],
        bindings=SemanticProjectionBindings(**view["origin"]), policy=policy,
        count_tokens=lambda text: 1, capsule_ids=(wanted["capsule_cid"],),
    )
    assert json.loads(expanded.summary)["fields"] == [{"field_selector": ["capsules", 1], "value": wanted}]
    assert expanded.metadata["complete_archive"] is False
    assert expanded.to_dict()["metadata"]["expanded_capsule_ids"] == [wanted["capsule_cid"]]
    assert expanded.referenced_content_id == "sha256:" + hashlib.sha256(expanded.summary.encode()).hexdigest()


def test_parent_digest_and_repaired_chunks_do_not_authenticate_forged_producer_capsules(native):
    root, artifact, prompt, _ = native
    parent = json.loads(artifact.read_text())
    policy = _policy(parent)
    signature = parent["capsules"][0]["signature"]
    signature["is_generator"] = not signature["is_generator"]
    text = _json(parent)
    artifact.write_text(text)
    wire, end = json.JSONDecoder().raw_decode(prompt)
    refs = build_text_context_references(
        text, reference_prefix="semantic-context", kind="semantic-context",
        path=artifact.relative_to(root).as_posix(), repository_id=wire["repository_id"],
        tree_id=wire["tree_id"], required=True,
    )
    wire["evidence"] = [row.to_record() for row in refs]
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == hashlib.sha256(text.encode()).hexdigest()
    with pytest.raises(ValueError, match="capsule"):
        project_semantic_router_prompt(prompt=_json(wire) + prompt[end:], repository=root, decision_policy=policy)


def test_matched_chunk_comparison_records_packet_and_expansion_costs(native, record_property):
    root, artifact, prompt, _ = native
    parent = json.loads(artifact.read_text())
    wire, end = json.JSONDecoder().raw_decode(prompt)
    refs = build_text_context_references(
        artifact.read_text(), reference_prefix="semantic-context", kind="semantic-context",
        path=artifact.relative_to(root).as_posix(), repository_id=wire["repository_id"],
        tree_id=wire["tree_id"], required=True, chunk_bytes=6144,
    )
    wire["evidence"] = [row.to_record() for row in refs]
    matched_native = _json(wire) + prompt[end:]
    policy = _policy(parent)
    projected = project_semantic_router_prompt(
        prompt=matched_native, repository=root, decision_policy=policy, chunk_bytes=6144,
    )
    view = _packet(projected.projected_prompt)
    expanded = expand_semantic_router_decision_packet(
        repository=root, view=view, reference_id=view["deferred_references"][0]["reference_id"],
        bindings=SemanticProjectionBindings(**view["origin"]), policy=policy,
        count_tokens=lambda text: 1, max_bytes=65536,
    )
    # These are complete native packet/expanded descriptor bytes, not tokens or
    # session totals. Any actual expansion costs must be included in an ablation.
    record_property("native_prompt_bytes", len(matched_native.encode()))
    record_property("decision_prompt_bytes", len(projected.projected_prompt.encode()))
    record_property("expanded_reference_bytes", len(_json(expanded.to_dict()).encode()))
    record_property("whole_session_savings_measured", False)
    assert projected.receipt["native_prompt_bytes"] == len(matched_native.encode())
    assert projected.receipt["projected_prompt_bytes"] == len(projected.projected_prompt.encode())
    assert parent["capsules"][1:] == [row["value"] for row in json.loads(expanded.summary)["fields"]
                                      if row["field_selector"][0] == "capsules"]
