"""Recorded native relations and finite Python binding observations."""

from __future__ import annotations

import json
from hashlib import sha256

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_capsule_closure_nomination import (
    nominate_declared_capsule_closure,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import (
    prepare_semantic_context,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_selection_relation_audit import (
    audit_selected_capsule_relations,
)
from ipfs_datasets_py.logic.software_contracts.semantic_state.api import (
    open_semantic_state,
)
from ipfs_datasets_py.logic.software_contracts.semantic_state.models import (
    SemanticStateBundle,
    SemanticStateRoot,
)
from test.api.test_semantic_selection_comparison_probe import (
    OBJECTIVE,
    SOURCES,
    TASK_ID,
    _block_hashes,
    _prompt,
)


def test_native_selection_typed_relation_and_binding_probe(tmp_path, record_property):
    repository = tmp_path / "repository"
    repository.mkdir()
    for name, source in SOURCES.items():
        (repository / name).write_text(source)
    source_before = {name: (repository / name).read_bytes() for name in SOURCES}
    output = repository / ".runtime" / "full"
    preparation = prepare_semantic_context(
        repository=repository, paths=sorted(SOURCES), required_raw_paths=sorted(SOURCES),
        objective=OBJECTIVE, task_id=TASK_ID, output=output,
        max_source_bytes=65536, max_files=8, max_symbols=64,
        context_input_tokens=32768, worker_query="", worker_capsule_limit=1,
        worker_max_bytes=65536,
    )
    assert preparation["ducklake"]["status"] == "projected"
    artifact = output / "worker-context.json"
    artifact_before = artifact.read_bytes()
    payload = json.loads(artifact_before)
    blocks_before = _block_hashes(output)
    root_cid = payload["semantic_root_cid"]
    blocks = {path.name: path.read_bytes() for path in (output / "blocks").iterdir()}
    root = SemanticStateRoot.from_dict({**json.loads(blocks[root_cid]), "root_cid": root_cid})
    bundle = SemanticStateBundle(root=root, blocks=blocks)
    view = open_semantic_state(root_cid, bundle.get_block)
    assert len(payload["capsules"]) == 11
    assert payload["raw_sources"] == SOURCES
    assert payload["required_raw_paths"] == sorted(SOURCES)
    assert all(view.capsule(row["stable_symbol_id"]).to_dict() == row for row in payload["capsules"])
    symbols = {
        view.symbol_fact(row["stable_symbol_id"]).symbol.qualified_name: row["stable_symbol_id"]
        for row in payload["capsules"]
    }
    target_id = symbols["operations.target_operation"]
    helper_id = symbols["operations.helper_operation"]
    test_id = symbols["test_operations.test_public_acceptance"]
    nomination = nominate_declared_capsule_closure(
        full_capsules=payload["capsules"], seed_symbol_ids=[target_id],
    )
    assert nomination["nominee_symbol_ids"] == sorted([target_id, helper_id])
    selections = {
        "seed": [target_id],
        "known_dependency_closure": nomination["nominee_symbol_ids"],
        "known_dependency_and_test": sorted([target_id, helper_id, test_id]),
    }
    audits = {
        arm: audit_selected_capsule_relations(
            bundle=bundle, root_cid=root_cid, selected_symbol_ids=selected,
        )
        for arm, selected in selections.items()
    }
    seed, closure, with_test = (audits[arm] for arm in selections)
    assert seed["missing_known_call_dependency_symbol_ids"] == [helper_id]
    assert closure["missing_known_call_dependency_symbol_ids"] == []
    assert closure["declared_known_call_closure_selected"] is True
    assert closure["omitted_outgoing_test_symbol_ids"] == [test_id]
    assert with_test["omitted_outgoing_test_symbol_ids"] == []
    for audit in audits.values():
        assert audit["root_cid"] == root_cid
        assert audit["full_capsule_count"] == 11
        assert audit["full_link_count"] == 6
        assert audit["root_index_records_verified"] is True
        assert audit["node_adjacency_verified"] is True
        assert audit["capsule_relation_projections_verified"] is True
        assert audit["selection_eligibility"] == "experimental_only"
        for key in (
            "source_freshness_verified", "dependency_semantics_complete",
            "task_acceptance_verified", "program_equivalence_verified",
            "deployment_ready", "runtime_activation", "completion_authority",
        ):
            assert audit[key] is False
        binding = [row for row in audit["unresolved_state_binding_rows"]
                   if row["source_stable_id"] == target_id]
        assert len(binding) == 1
        assert binding[0]["unresolved_target"] == "global:helper_operation"
        assert binding[0]["relation"] == "reads_state"
        assert binding[0]["extraction_method"] == "global_binding"
        assert binding[0]["target_kind"] == "unresolved"
        assert binding[0]["target_stable_id"] is None
    call = [row for row in closure["outgoing_relation_rows"]
            if row["source_stable_id"] == target_id and row["relation"] == "calls"]
    assert len(call) == 1
    assert call[0]["target_stable_id"] == helper_id
    assert call[0]["target_kind"] == "symbol"
    assert call[0]["resolution"] == "definite"
    assert call[0]["edge_id"] != call[0]["link_cid"]
    assert call[0]["edge_id"] in view.capsule(target_id).dependency_link_ids
    assert call[0]["link_cid"] in view.symbol_node(target_id).outgoing_link_cids
    assert any(row["source_stable_id"] == test_id and row["target_stable_id"] == target_id
               and row["relation"] == "calls" for row in closure["incoming_omitted_relation_rows"])

    # Two finite observations on this authored fixture explain why the state
    # binding cannot be discharged by retaining the helper's definition alone.
    namespace = {"__name__": "authored_binding_fixture"}
    exec(compile(source_before["operations.py"], "operations.py", "exec"), namespace)
    target = namespace["target_operation"]
    initial_result = target(4)
    namespace["helper_operation"] = lambda value: value * 5
    rebound_result = target(4)
    assert initial_result == 10
    assert rebound_result == 22

    prompt = _prompt(repository, artifact)
    encoded = codec.encode_semantic_router_prompt(
        prompt=prompt, repository=repository, slot_policy_revision=codec.SLOT_POLICY_V2,
    )
    assert codec.restore_semantic_router_prompt(
        provider_prompt=encoded.provider_prompt, table=encoded.table, repository=repository,
    ) == prompt
    changed = repository / "operations.py"
    changed.write_bytes(source_before["operations.py"] + b"\n# changed current source\n")
    try:
        # The pure ledger still describes the same old root, without claiming
        # freshness. The existing transport rejects reuse at its source boundary.
        assert audit_selected_capsule_relations(
            bundle=bundle, root_cid=root_cid, selected_symbol_ids=selections["known_dependency_closure"],
        ) == closure
        with pytest.raises(ValueError):
            codec.restore_semantic_router_prompt(
                provider_prompt=encoded.provider_prompt, table=encoded.table, repository=repository,
            )
    finally:
        changed.write_bytes(source_before["operations.py"])
    assert codec.restore_semantic_router_prompt(
        provider_prompt=encoded.provider_prompt, table=encoded.table, repository=repository,
    ) == prompt
    assert {name: (repository / name).read_bytes() for name in SOURCES} == source_before
    assert artifact.read_bytes() == artifact_before
    assert _block_hashes(output) == blocks_before
    report = {
        "schema": "supervisor-native-selection-relation-probe@1",
        "root_cid": root_cid, "scope_cid": payload["scope_cid"],
        "source_manifest_sha256": sha256(json.dumps(payload["manifest"], sort_keys=True).encode()).hexdigest(),
        "audits": audits,
        "finite_binding_observations": {
            "fixture_code_executed": True, "input": 4, "original_result": initial_result,
            "rebound_helper_result": rebound_result,
            "scope": "two CPython observations on the authored fixture only",
            "universal_source_equivalence_verified": False,
        },
        "known_call_omission_removed": True,
        "state_binding_not_rewritten": True,
        "known_test_omission_removed_only_with_explicit_test_selection": True,
        "same_complete_raw_inventory_retained": True,
        "native_sources_artifact_and_blocks_unchanged": True,
        "existing_transport_current_source_change_refused": True,
        "root_ledger_claims_current_source_freshness": False,
        "model_tokens": None, "whole_session_token_savings": None,
        "task_acceptance_verified": False, "terminal_bench_outcomes_verified": False,
        "deployment_ready": False, "runtime_activation": False,
        "provider_calls": 0, "downloads": 0, "private_native_preparations": 1,
    }
    record_property("semantic_selection_relation_probe", json.dumps(report, sort_keys=True))
