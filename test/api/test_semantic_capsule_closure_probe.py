"""Native producer reproduction of a bounded declared-graph nomination."""

from __future__ import annotations

import json
from hashlib import sha256

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import (
    render_model_prompt,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_capsule_closure_nomination import (
    nominate_declared_capsule_closure,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import (
    prepare_semantic_context,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_selection_comparison import (
    compare_semantic_selection,
)
from test.api.test_semantic_selection_comparison_probe import (
    OBJECTIVE,
    SOURCES,
    SUFFIX,
    TASK_ID,
    _block_hashes,
    _prompt,
)


def test_known_declared_closure_native_reproduction_probe(tmp_path, record_property):
    repository = tmp_path / "repository"
    repository.mkdir()
    for name, text in SOURCES.items():
        (repository / name).write_text(text)
    source_before = {name: (repository / name).read_bytes() for name in SOURCES}
    bounds = {
        "repository": repository,
        "paths": sorted(SOURCES),
        "required_raw_paths": sorted(SOURCES),
        "objective": OBJECTIVE,
        "task_id": TASK_ID,
        "max_source_bytes": 65536,
        "max_files": 8,
        "max_symbols": 64,
        "context_input_tokens": 32768,
        "worker_max_bytes": 65536,
    }
    queries = {"full": "", "seed": "target_operation",
               "known_closure": "target_operation helper_operation"}
    outputs = {arm: repository / ".runtime" / arm for arm in queries}
    preparations = {
        arm: prepare_semantic_context(
            output=outputs[arm], worker_query=query,
            worker_capsule_limit=2 if arm == "known_closure" else 1, **bounds,
        )
        for arm, query in queries.items()
    }
    artifacts = {arm: output / "worker-context.json" for arm, output in outputs.items()}
    artifact_before = {arm: path.read_bytes() for arm, path in artifacts.items()}
    block_before = {arm: _block_hashes(output) for arm, output in outputs.items()}
    payloads = {arm: json.loads(raw) for arm, raw in artifact_before.items()}
    full, seed, lexical = (payloads[arm] for arm in ("full", "seed", "known_closure"))
    assert "worker_projection" not in full
    assert len(full["capsules"]) == 11
    full_capsules = {capsule["stable_symbol_id"]: capsule for capsule in full["capsules"]}
    seed_ids = [capsule["stable_symbol_id"] for capsule in seed["capsules"]]
    assert len(seed_ids) == 1
    assert seed["worker_projection"]["selected_symbols"][0]["qualified_name"].endswith(
        ".target_operation")
    for arm, payload in payloads.items():
        assert payload["raw_sources"] == SOURCES
        assert payload["required_raw_paths"] == sorted(SOURCES)
        assert preparations[arm]["ducklake"]["status"] == "projected"
        for capsule in payload["capsules"]:
            assert capsule == full_capsules[capsule["stable_symbol_id"]]
        for name in (
            "schema", "task_id", "objective", "scope_cid", "semantic_root_cid", "manifest",
            "preparation_bounds", "reconstruction", "doctor_snapshot_id", "doctor_manifest_cid",
            "refresh_lineage",
        ):
            assert payload[name] == full[name]

    # This nomination traverses declared stable-ID edges only. In particular,
    # it does not rewrite global:helper_operation into a known helper ID.
    nomination = nominate_declared_capsule_closure(
        full_capsules=full["capsules"], seed_symbol_ids=seed_ids,
        max_capsules=32, max_packet_bytes=65536,
    )
    assert nomination["nomination_available"] is True
    assert nomination["nomination_status"] == "nomination_available_with_unresolved"
    assert nomination["initial_seed_symbol_ids"] == sorted(seed_ids)
    assert nomination["known_closure_capsule_count"] == 2
    assert len(nomination["added_known_symbol_ids"]) == 1
    helper_id = nomination["added_known_symbol_ids"][0]
    assert helper_id in seed["capsules"][0]["dependency_stable_ids"]
    assert nomination["unresolved_dependency_symbol_ids"] == ["global:helper_operation"]
    assert nomination["declared_graph_closed"] is False
    nominee_capsules = nomination["nominee_packet"]["capsules"]
    assert nominee_capsules == [
        full_capsules[symbol_id] for symbol_id in nomination["nominee_symbol_ids"]
    ]
    assert nomination["nominee_symbol_ids"] == nomination["known_closure_symbol_ids"]

    refused = nominate_declared_capsule_closure(
        full_capsules=full["capsules"], seed_symbol_ids=seed_ids,
        max_capsules=1, max_packet_bytes=65536,
    )
    assert refused["nomination_available"] is False
    assert refused["nominee_symbol_ids"] == []
    assert refused["nominee_packet"] is refused["nominee_packet_bytes"] is refused["nominee_packet_sha256"] is None
    assert refused["known_closure_symbol_ids"] == nomination["known_closure_symbol_ids"]
    assert refused["unresolved_dependency_symbol_ids"] == ["global:helper_operation"]

    # Reproduce the nominated pair using the unchanged native lexical selector;
    # no producer schema, capsule, admission, or pack is repaired after creation.
    lexical_ids = sorted(capsule["stable_symbol_id"] for capsule in lexical["capsules"])
    reproduced = lexical_ids == nomination["nominee_symbol_ids"]
    if reproduced:
        assert sorted(row["qualified_name"] for row in lexical["worker_projection"]["selected_symbols"]) == [
            "operations.helper_operation", "operations.target_operation",
        ]
        assert [full_capsules[symbol_id] for symbol_id in lexical_ids] == nominee_capsules
    prompts = {arm: _prompt(repository, artifact) for arm, artifact in artifacts.items()}
    wires_and_ends = {arm: json.JSONDecoder().raw_decode(prompt) for arm, prompt in prompts.items()}
    full_wire = wires_and_ends["full"][0]
    for arm, (wire, end) in wires_and_ends.items():
        assert {key: value for key, value in wire.items() if key != "evidence"} == {
            key: value for key, value in full_wire.items() if key != "evidence"
        }
        assert prompts[arm][end:] == SUFFIX
        assert all(row["metadata"]["required"] is True for row in wire["evidence"])

    comparison_args = {
        "repository": repository, "full_prompt": prompts["full"],
        "workspace": repository, "purpose": "coding", "provider": "codex_cli",
        "model": "fixture-model", "token_counter": None,
        "measurement_basis": "authored_fixture",
    }
    seed_comparison = compare_semantic_selection(selected_prompt=prompts["seed"], **comparison_args)
    assert seed_comparison["dependency_audit"]["missing_selected_dependency_symbol_ids"] == [helper_id]
    assert seed_comparison["dependency_audit"]["unresolved_dependency_symbol_ids"] == ["global:helper_operation"]
    closure_comparison = (
        compare_semantic_selection(selected_prompt=prompts["known_closure"], **comparison_args)
        if reproduced else None
    )
    if closure_comparison is not None:
        closure_audit = closure_comparison["dependency_audit"]
        assert closure_audit["missing_selected_dependency_symbol_ids"] == []
        assert closure_audit["unresolved_dependency_symbol_ids"] == ["global:helper_operation"]
        assert closure_audit["declared_graph_closure_selected"] is False
        assert closure_audit["dependency_semantics_complete"] is False
        assert closure_comparison["representations"]["full_native"] == seed_comparison["representations"]["full_native"]
        assert closure_comparison["representations"]["full_v2_aliases"] == seed_comparison["representations"]["full_v2_aliases"]
    for comparison in (seed_comparison, closure_comparison):
        if comparison is None:
            continue
        assert comparison["measurement_status"] == "unavailable"
        assert comparison["tokenizer_binding"] is comparison["whole_session_token_savings"] is None
        assert comparison["campaign_usage_complete"] is None
        assert comparison["matching_reference_chunk_policy"]["chunk_bytes"] == 6144
        assert comparison["matching_reference_chunk_policy"]["verified"] is True
        for name in (
            "task_acceptance_verified", "deployment_ready", "runtime_activation",
            "program_equivalence_verified", "semantic_equivalence_claimed",
            "execution_authority", "completion_authority", "dependency_closure_verified",
        ):
            assert comparison[name] is False

    representations = {
        "full_native": seed_comparison["representations"]["full_native"],
        "full_v2_aliases": seed_comparison["representations"]["full_v2_aliases"],
        "seed_native": seed_comparison["representations"]["query_selected_native"],
        "seed_v2_aliases": seed_comparison["representations"]["query_selected_v2_aliases"],
    }
    if closure_comparison is not None:
        representations.update(
            known_closure_native=closure_comparison["representations"]["query_selected_native"],
            known_closure_v2_aliases=closure_comparison["representations"]["query_selected_v2_aliases"],
        )
    for arm, prompt in prompts.items():
        encoded = codec.encode_semantic_router_prompt(
            prompt=prompt, repository=repository, slot_policy_revision=codec.SLOT_POLICY_V2,
        )
        assert codec.restore_semantic_router_prompt(
            provider_prompt=encoded.provider_prompt, table=encoded.table, repository=repository,
        ) == prompt
        for translated, text in ((False, prompt), (True, encoded.provider_prompt)):
            label = arm + ("_v2_aliases" if translated else "_native")
            rendered, _ = render_model_prompt(
                prompt=text, purpose="coding", workspace=repository, semantic_transport=translated,
            )
            if label in representations:
                assert representations[label] == {
                    "model_prompt_sha256": sha256(rendered.encode()).hexdigest(),
                    "model_prompt_bytes": len(rendered.encode()), "input_tokens": None,
                }

    def assert_originals_unchanged():
        assert payloads == {arm: json.loads(raw) for arm, raw in artifact_before.items()}
        assert {name: (repository / name).read_bytes() for name in SOURCES} == source_before
        assert {arm: path.read_bytes() for arm, path in artifacts.items()} == artifact_before
        assert {arm: _block_hashes(output) for arm, output in outputs.items()} == block_before

    assert_originals_unchanged()
    changed = repository / "operations.py"
    changed.write_bytes(source_before["operations.py"] + b"\n# changed source\n")
    try:
        with pytest.raises(ValueError, match="stale"):
            compare_semantic_selection(
                selected_prompt=prompts["known_closure"] if reproduced else prompts["seed"],
                **comparison_args,
            )
    finally:
        changed.write_bytes(source_before["operations.py"])
    assert_originals_unchanged()

    report = {
        "schema": "supervisor-semantic-capsule-closure-probe@1",
        "measurement_basis": "authored_fixture", "count_scope": "rendered_user_prompt_only",
        "measurement_status": "unavailable", "tokenizer_binding": None,
        "native_nomination_reproduction_status": "reproduced" if reproduced else "cannot_reproduce",
        "native_selected_ids_equal_nominee_ids": reproduced,
        "lexical_selected_symbol_ids": lexical_ids,
        "nomination": {key: value for key, value in nomination.items() if key != "nominee_packet"},
        "capsule_budget_refusal": {key: value for key, value in refused.items() if key != "nominee_packet"},
        "nominee_capsule_bodies_in_report": False,
        "representations": representations,
        "comparisons": {"seed": seed_comparison, "known_closure": closure_comparison},
        "source_inventory": {
            name: {"sha256": sha256(raw).hexdigest(), "bytes": len(raw)}
            for name, raw in source_before.items()
        },
        "worker_artifacts": {
            arm: {"sha256": sha256(raw).hexdigest(), "bytes": len(raw),
                  "capsule_count": len(payloads[arm]["capsules"])}
            for arm, raw in artifact_before.items()
        },
        "same_required_complete_raw_inventory": True,
        "same_literal_task_acceptance_and_suffix": True,
        "exact_v2_restoration_of_each_own_native_prompt": True,
        "original_sources_artifacts_and_producer_blocks_unchanged": True,
        "current_source_change_refused": True,
        "declared_edges_inferred_from_source_text": False,
        "known_dependency_edge_added": reproduced,
        "unresolved_dependency_alias_rewritten": False,
        "remaining_unresolved_dependency_count": len(nomination["unresolved_dependency_symbol_ids"]),
        "public_acceptance_executed": False, "task_acceptance_verified": False,
        "provider_calls": 0, "downloads": 0, "real_tokenizer": None,
        "whole_session_token_savings": None, "campaign_usage_complete": None,
        "dependency_semantics_complete": False, "program_equivalence_verified": False,
        "semantic_equivalence_claimed": False, "deployment_ready": False,
        "runtime_activation": False, "execution_authority": False,
        "completion_authority": False, "proof_authority": False,
    }
    record_property("semantic_capsule_closure_probe", json.dumps(report, sort_keys=True))
