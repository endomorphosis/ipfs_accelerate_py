"""Compare full and projected context with the same reference chunk policy.

The pinned producer and current codec execute normally. Bytes are observations;
no model tokenizer, provider usage or semantic admission is inferred.
"""

from __future__ import annotations

import hashlib
import json

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    build_text_context_references,
    context_provider_input_payload,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import (
    canonical_context_json_bytes,
)
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import (
    render_model_prompt,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_projection_dispatch import (
    project_semantic_router_prompt,
)

pytest_plugins = ("test.api.test_semantic_router_translation",)


def test_projection_comparison_controls_reference_chunk_bytes(native, record_property):
    repository, artifact, prompt, source = native
    original_artifact = artifact.read_bytes()
    original_source = (repository / "mod.py").read_bytes()
    wire, offset = json.JSONDecoder().raw_decode(prompt)
    suffix = prompt[offset:]
    core = {key: value for key, value in wire.items() if key not in {"evidence", "contract_version"}}
    chunk_bytes = 6144  # The actual projection adapter's existing chunk policy.
    full_references = build_text_context_references(
        original_artifact.decode(), reference_prefix="semantic-context", kind="semantic-context",
        path=artifact.relative_to(repository).as_posix(),
        repository_id=wire["repository_id"], tree_id=wire["tree_id"],
        required=True, chunk_bytes=chunk_bytes,
    )
    full_wire = context_provider_input_payload(**core, evidence=full_references)
    full_prompt = canonical_context_json_bytes(full_wire).decode() + suffix
    assert {key: value for key, value in full_wire.items() if key != "evidence"} == {
        key: value for key, value in wire.items() if key != "evidence"
    }
    assert "".join(row.summary for row in full_references) == original_artifact.decode()

    projected = project_semantic_router_prompt(prompt=full_prompt, repository=repository)
    projected_wire, projected_offset = json.JSONDecoder().raw_decode(projected.projected_prompt)
    projected_references = [row for row in projected_wire["evidence"] if row["kind"] == "semantic-context"]
    assert all(row["metadata"]["required"] is True for row in projected_references)
    assert all(row["byte_count"] <= chunk_bytes for row in projected_references)
    assert all(row.byte_count <= chunk_bytes for row in full_references)
    view = json.loads("".join(row["summary"] for row in projected_references))
    assert view["required_context"]["raw_sources"]["mod.py"] == source
    assert {key: value for key, value in projected_wire.items() if key != "evidence"} == {
        key: value for key, value in full_wire.items() if key != "evidence"
    }
    assert projected.projected_prompt[projected_offset:] == suffix

    full_v2 = codec.encode_semantic_router_prompt(
        prompt=full_prompt, repository=repository, slot_policy_revision=codec.SLOT_POLICY_V2,
    )
    projected_v2 = codec.encode_semantic_router_prompt(
        prompt=projected.projected_prompt, repository=repository, slot_policy_revision=codec.SLOT_POLICY_V2,
    )
    assert codec.restore_semantic_router_prompt(
        provider_prompt=full_v2.provider_prompt, table=full_v2.table, repository=repository,
    ) == full_prompt
    assert codec.restore_semantic_router_prompt(
        provider_prompt=projected_v2.provider_prompt, table=projected_v2.table, repository=repository,
    ) == projected.projected_prompt
    assert artifact.read_bytes() == original_artifact
    assert (repository / "mod.py").read_bytes() == original_source

    representations = {}
    for name, candidate, transport in (
        ("full_native", full_prompt, False),
        ("plain_role_view", projected.projected_prompt, False),
        ("full_v2", full_v2.provider_prompt, True),
        ("role_view_v2", projected_v2.provider_prompt, True),
    ):
        rendered, _ = render_model_prompt(
            prompt=candidate, purpose="coding", workspace=repository, semantic_transport=transport,
        )
        raw = rendered.encode()
        representations[name] = {"model_prompt_bytes": len(raw),
                                 "model_prompt_sha256": hashlib.sha256(raw).hexdigest(),
                                 "input_tokens": None}
    record_property("matched_chunk_projection_probe", json.dumps({
        "schema": "supervisor-matched-chunk-projection-probe@1",
        "fixture_basis": "actual pinned producer and compiled context fixture",
        "assigned_task_count": 1, "task_id": wire["objective_id"],
        "reference_chunk_bytes": {"full_native": chunk_bytes, "plain_role_view": chunk_bytes},
        "reference_counts": {"full_native": len(full_references), "plain_role_view": len(projected_references)},
        "representations": representations,
        "plain_role_view_byte_delta": representations["full_native"]["model_prompt_bytes"]
            - representations["plain_role_view"]["model_prompt_bytes"],
        "role_view_v2_byte_delta": representations["full_v2"]["model_prompt_bytes"]
            - representations["role_view_v2"]["model_prompt_bytes"],
        "source_sha256": hashlib.sha256(original_source).hexdigest(),
        "worker_artifact_sha256": hashlib.sha256(original_artifact).hexdigest(),
        "original_source_and_worker_artifact_unchanged": True,
        "task_core_and_literal_suffix_unchanged": True,
        "required_parent_verified": True,
        "tokenizer_binding": None, "provider_input_tokens": None, "provider_output_tokens": None,
        "total_session_token_savings": None, "provider_calls": 0, "downloads": 0,
        "task_acceptance_verified": False, "semantic_equivalence_claimed": False,
        "execution_authority": False, "completion_authority": False,
    }, sort_keys=True))
