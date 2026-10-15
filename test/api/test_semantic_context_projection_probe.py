"""One matched offline native-producer probe; bytes are not provider tokens."""
from __future__ import annotations

import copy
import json
from hashlib import sha256

from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
from ipfs_accelerate_py.agent_supervisor.runtime.context_projection_comparison import (
    measure_context_candidates,
)
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import (
    render_model_prompt,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_projection_dispatch import (
    project_semantic_router_prompt,
)

pytest_plugins = ("test.api.test_semantic_router_translation",)


def test_native_fixture_matched_five_representation_probe(native, record_property):
    repository, artifact, prompt, source = native
    original_artifact, original_source = artifact.read_bytes(), (repository / "mod.py").read_bytes()
    original_wire, original_end = json.JSONDecoder().raw_decode(prompt)
    projected = project_semantic_router_prompt(prompt=prompt, repository=repository)
    projected_wire, projected_end = json.JSONDecoder().raw_decode(projected.projected_prompt)
    references = [row for row in projected_wire["evidence"] if row["kind"] == "semantic-context"]
    view = json.loads("".join(row["summary"] for row in references))
    required, original = view["required_context"], json.loads(original_artifact)
    assert required["raw_sources"]["mod.py"] == source
    for name in ("objective", "manifest", "required_raw_paths", "capsules", "admissions",
                 "reconstruction", "refresh_lineage", "worker_projection"):
        if name in original:
            assert required[name] == original[name]
    for name in ("assumptions", "exclusions", "risk", "route", "obligation_cids", "counterexample_cids"):
        assert required["pack"]["pack"][name] == original["pack"]["pack"][name]
    assert required["pack"]["coverage_satisfied"] == original["pack"]["coverage_satisfied"]
    assert {key: value for key, value in projected_wire.items() if key != "evidence"} == {
        key: value for key, value in original_wire.items() if key != "evidence"}
    assert projected.projected_prompt[projected_end:] == prompt[original_end:]
    assert all(row["metadata"]["required"] is True for row in references)

    legacy = codec.encode_semantic_router_prompt(prompt=prompt, repository=repository)
    v2 = codec.encode_semantic_router_prompt(prompt=prompt, repository=repository,
                                           slot_policy_revision="closed-typed-identifiers@2")
    role_v2 = codec.encode_semantic_router_prompt(prompt=projected.projected_prompt, repository=repository,
                                                slot_policy_revision="closed-typed-identifiers@2")
    for encoded, expected in ((legacy, prompt), (v2, prompt), (role_v2, projected.projected_prompt)):
        assert codec.restore_semantic_router_prompt(provider_prompt=encoded.provider_prompt,
                                                    table=encoded.table, repository=repository) == expected
        assert encoded.receipt["semantic_equivalence_claimed"] is False
        assert encoded.receipt["execution_authority"] is encoded.receipt["completion_authority"] is False
    # Original source spelling, comments, names and literals are equal after
    # each exact restoration, including '$semantic_ref' and the literal 's0'.
    for encoded in (legacy, v2):
        restored = codec.restore_semantic_router_prompt(provider_prompt=encoded.provider_prompt,
                                                        table=encoded.table, repository=repository)
        wire, _ = json.JSONDecoder().raw_decode(restored)
        semantic = json.loads("".join(row["summary"] for row in wire["evidence"]
                                      if row["kind"] == "semantic-context"))
        assert semantic["raw_sources"]["mod.py"] == source
    assert artifact.read_bytes() == original_artifact
    assert (repository / "mod.py").read_bytes() == original_source

    wires = {
        "full_native": (prompt, False), "legacy_aliases": (legacy.provider_prompt, True),
        "v2_aliases": (v2.provider_prompt, True), "plain_role_view": (projected.projected_prompt, False),
        "role_view_v2_aliases": (role_v2.provider_prompt, True),
    }
    measurements = {}
    for name, (candidate, semantic_transport) in wires.items():
        rendered, _ = render_model_prompt(prompt=candidate, purpose="coding", workspace=repository,
                                         semantic_transport=semantic_transport)
        raw = rendered.encode()
        measurements[name] = {"model_prompt_bytes": len(raw), "model_prompt_sha256": sha256(raw).hexdigest(),
                              "input_tokens": None}
    arguments = dict(native_prompt=prompt, suffix="", purpose="coding", workspace=repository,
                     provider="codex_cli", model="fixture-model", token_counter=None,
                     measurement_basis="authored_fixture")
    comparisons = {
        "legacy": measure_context_candidates(projected_prompt=prompt, alias_prompt=legacy.provider_prompt, **arguments),
        "v2": measure_context_candidates(projected_prompt=prompt, alias_prompt=v2.provider_prompt, **arguments),
        "role_view": measure_context_candidates(projected_prompt=projected.projected_prompt,
                                                alias_prompt=role_v2.provider_prompt, **arguments),
    }
    for name, comparison, candidate in (
        ("full_native", comparisons["role_view"], "native"),
        ("legacy_aliases", comparisons["legacy"], "alias"),
        ("v2_aliases", comparisons["v2"], "alias"),
        ("plain_role_view", comparisons["role_view"], "projected"),
        ("role_view_v2_aliases", comparisons["role_view"], "alias"),
    ):
        assert comparison["candidates"][candidate] == measurements[name]
    assert all(row["projected_alias_selection"] == "native" for row in comparisons.values())
    assert all(row["measurement_status"] == {"native": "unavailable", "projected_and_alias": "unavailable"}
               for row in comparisons.values())
    baseline_bytes = measurements["full_native"]["model_prompt_bytes"]
    report = {
        "schema": "supervisor-semantic-context-projection-offline-probe@1",
        "fixture_basis": "existing test_semantic_router_translation.native; actual native source producer and compiler",
        "assigned_task_count": 1, "task_id": original_wire["objective_id"],
        "source": {"sha256": sha256(original_source).hexdigest(), "size_bytes": len(original_source)},
        "worker_artifact": {"sha256": sha256(original_artifact).hexdigest(), "size_bytes": len(original_artifact)},
        "render_scope": "complete rendered coding user prompt; same workspace and literal suffix",
        "representations": measurements,
        "rendered_user_byte_deltas_from_full_native": {
            name: baseline_bytes - row["model_prompt_bytes"] for name, row in measurements.items()},
        "exact_restoration_targets": {"legacy_aliases": "full_native", "v2_aliases": "full_native",
                                      "role_view_v2_aliases": "plain_role_view"},
        "projection_policy": view["projection_revision"],
        "deferred_reference_count": len(view["deferred_references"]),
        "complete_required_decision_core_and_literals_preserved": True,
        "original_source_and_worker_artifact_unchanged": True,
        "tokenizer_binding": None, "all_local_token_counts_unknown": True,
        "counter_unavailable_preserves_native_selection": True,
        "provider_input_tokens": None, "provider_output_tokens": None, "total_session_token_savings": None,
        "task_acceptance_verified": False, "campaign_usage_complete": None,
        "provider_calls": 0, "downloads": 0, "execution_authority": False, "completion_authority": False,
        "semantic_equivalence_claimed": False,
    }
    assert copy.deepcopy(report) == report
    record_property("semantic_context_projection_probe", json.dumps(report, sort_keys=True, separators=(",", ":")))
