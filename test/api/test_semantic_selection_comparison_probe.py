"""Matched native producer selection and encoding; no provider-token claim."""

from __future__ import annotations

import json
from hashlib import sha256

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ContextCompiler,
    build_text_context_references,
    render_context_capsule,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import (
    render_model_prompt,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import (
    prepare_semantic_context,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_selection_comparison import (
    compare_semantic_selection,
)
from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import SupervisorMetaIndex

TASK_ID = "TASK-SELECTION-PROBE"
OBJECTIVE = "Inspect target_operation and preserve its helper call and public acceptance checks."
SUFFIX = "\nLiteral instruction: retain source spelling s0 and $semantic_ref.\n"
SOURCES = {
    "operations.py": (
        "# Literal aliases s0 and $semantic_ref are source text.\n"
        "def helper_operation(value):\n"
        "    return value * 2\n\n"
        "def target_operation(value):\n"
        "    return helper_operation(value) + 2\n\n"
        "def alpha_unrelated(value):\n"
        "    return value + 11\n\n"
        "def bravo_unrelated(value):\n"
        "    return value + 22\n\n"
        "def charlie_unrelated(value):\n"
        "    return value + 33\n\n"
        "def delta_unrelated(value):\n"
        "    return value + 44\n\n"
        "def echo_unrelated(value):\n"
        "    return value + 55\n\n"
        "def foxtrot_unrelated(value):\n"
        "    return value + 66\n"
    ),
    "test_operations.py": (
        "from operations import target_operation\n\n"
        "def test_public_acceptance():\n"
        "    assert target_operation(4) == 10\n"
    ),
    "instruction.md": (
        "Inspect target_operation. Preserve its helper dependency and all unrelated operations.\n"
        "Public acceptance: python -m pytest -q test_operations.py.\n"
        "This fixture nominates context only; task acceptance remains unverified.\n"
    ),
}


def _prompt(repository, artifact):
    references = build_text_context_references(
        artifact.read_text(),
        reference_prefix="semantic-context",
        kind="semantic-context",
        path=artifact.relative_to(repository).as_posix(),
        repository_id="repo:selection-probe",
        tree_id="tree:selection-probe",
        required=True,
        chunk_bytes=6144,
        coverage_ids=("requirement:target", "requirement:public-acceptance"),
    )
    compiled = ContextCompiler(
        ContextBudget(
            max_input_tokens=32768,
            max_items=128,
            max_item_bytes=16384,
            max_serialized_bytes=262144,
        )
    ).compile(
        repository_id="repo:selection-probe",
        tree_id="tree:selection-probe",
        objective_id=TASK_ID,
        objective_revision="sha256:selection-probe-task",
        policy_id="policy:selection-probe",
        policy_revision="sha256:selection-probe-policy",
        caller="supervisor:selection-probe",
        stage="implementation",
        goal={"id": TASK_ID, "objective": OBJECTIVE},
        authority={"mode": "candidate_only", "completion_authority": False},
        scope={"allowed_paths": ["operations.py"]},
        acceptance={
            "criteria": ["helper call and unrelated operations preserved"],
            "validation": ["python -m pytest -q test_operations.py"],
        },
        evidence=references,
    )
    assert len(compiled.capsule.evidence) == len(references)
    return render_context_capsule(compiled.capsule) + SUFFIX


def _block_hashes(output):
    return {path.name: sha256(path.read_bytes()).hexdigest() for path in (output / "blocks").iterdir()}


def test_native_selection_and_v2_four_arm_probe(tmp_path, record_property):
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
        "worker_capsule_limit": 1,
        "worker_max_bytes": 65536,
    }
    outputs = {arm: repository / ".runtime" / arm for arm in ("full", "selected")}
    preparations = {
        arm: prepare_semantic_context(
            output=outputs[arm], worker_query="" if arm == "full" else "target_operation", **bounds
        )
        for arm in outputs
    }
    artifacts = {arm: output / "worker-context.json" for arm, output in outputs.items()}
    artifact_before = {arm: path.read_bytes() for arm, path in artifacts.items()}
    block_before = {arm: _block_hashes(output) for arm, output in outputs.items()}
    payloads = {arm: json.loads(raw) for arm, raw in artifact_before.items()}
    full, selected = payloads["full"], payloads["selected"]
    assert "worker_projection" not in full
    assert len(full["capsules"]) > 6
    projection = selected["worker_projection"]
    assert projection["selected_capsule_count"] == len(selected["capsules"]) == 1
    assert projection["query"] == "target_operation"
    assert projection["selected_symbols"][0]["qualified_name"].endswith("target_operation")
    assert projection["omitted_capsule_count"] == len(full["capsules"]) - 1
    assert selected["raw_sources"] == full["raw_sources"] == SOURCES
    assert selected["required_raw_paths"] == full["required_raw_paths"] == sorted(SOURCES)
    for name in (
        "task_id", "objective", "scope_cid", "semantic_root_cid", "manifest",
        "preparation_bounds", "reconstruction", "doctor_snapshot_id", "doctor_manifest_cid",
        "refresh_lineage",
    ):
        assert selected[name] == full[name]
    for arm in outputs:
        assert preparations[arm]["ducklake"]["status"] == "projected"
        assert (outputs[arm] / "blocks" / projection["capsule_index_cid"]).is_file()
    full_capsules = {capsule["stable_symbol_id"]: capsule for capsule in full["capsules"]}
    for capsule in selected["capsules"]:
        assert capsule == full_capsules[capsule["stable_symbol_id"]]

    prompts = {arm: _prompt(repository, artifact) for arm, artifact in artifacts.items()}
    full_wire, full_end = json.JSONDecoder().raw_decode(prompts["full"])
    selected_wire, selected_end = json.JSONDecoder().raw_decode(prompts["selected"])
    assert {key: value for key, value in full_wire.items() if key != "evidence"} == {
        key: value for key, value in selected_wire.items() if key != "evidence"
    }
    assert prompts["full"][full_end:] == prompts["selected"][selected_end:] == SUFFIX
    assert all(row["metadata"]["required"] is True for wire in (full_wire, selected_wire)
               for row in wire["evidence"])

    report = compare_semantic_selection(
        repository=repository,
        full_prompt=prompts["full"],
        selected_prompt=prompts["selected"],
        workspace=repository,
        purpose="coding",
        provider="codex_cli",
        model="fixture-model",
        token_counter=None,
        measurement_basis="authored_fixture",
    )
    independently_rendered = {}
    for arm, native in prompts.items():
        encoded = codec.encode_semantic_router_prompt(
            prompt=native, repository=repository, slot_policy_revision=codec.SLOT_POLICY_V2
        )
        assert codec.restore_semantic_router_prompt(
            provider_prompt=encoded.provider_prompt, table=encoded.table, repository=repository
        ) == native
        for translated, text in ((False, native), (True, encoded.provider_prompt)):
            label = ("full" if arm == "full" else "query_selected") + (
                "_v2_aliases" if translated else "_native"
            )
            rendered, _ = render_model_prompt(
                prompt=text, purpose="coding", workspace=repository, semantic_transport=translated
            )
            independently_rendered[label] = {
                "model_prompt_sha256": sha256(rendered.encode()).hexdigest(),
                "model_prompt_bytes": len(rendered.encode()),
                "input_tokens": None,
            }
    assert report["representations"] == independently_rendered
    assert report["measurement_status"] == "unavailable"
    assert report["tokenizer_binding"] is report["whole_session_token_savings"] is None
    assert report["campaign_usage_complete"] is None
    for name in (
        "task_acceptance_verified", "deployment_ready", "runtime_activation",
        "program_equivalence_verified", "semantic_equivalence_claimed",
        "execution_authority", "completion_authority",
    ):
        assert report[name] is False
    assert report["pack_decision_differences"]
    assert report["dependency_audit"]["dependency_semantics_complete"] is False
    selected_ids = {capsule["stable_symbol_id"] for capsule in selected["capsules"]}
    target_dependencies = set(selected["capsules"][0]["dependency_stable_ids"])
    directly_omitted = target_dependencies.intersection(full_capsules) - selected_ids
    assert directly_omitted <= set(report["dependency_audit"]["missing_selected_dependency_symbol_ids"])
    assert target_dependencies - set(full_capsules) <= set(
        report["dependency_audit"]["unresolved_dependency_symbol_ids"]
    )
    if directly_omitted or target_dependencies - set(full_capsules):
        assert report["dependency_audit"]["declared_graph_closure_selected"] is False

    # Only the preparer's private metadata outputs are opened. Their rows locate
    # immutable artifacts; they do not establish currentness or query quality.
    nominations = {}
    for arm, output in outputs.items():
        nomination = SupervisorMetaIndex(output / "metadata.duckdb").compose_for_subject(
            subject_kind="record_cid", subject_ref=full["scope_cid"], limit=8
        )
        assert nomination["completion_authority"] is nomination["decision_authority"] is False
        assert nomination["ducklake_authoritative"] is False
        capsule_rows = [row for row in nomination["linked"] if row["record_kind"] == "capsule"]
        assert len(capsule_rows) == 1
        assert capsule_rows[0]["record_ref"] == projection["capsule_index_cid"]
        assert capsule_rows[0]["locator_ref"] == str(output / "blocks")
        nominations[arm] = {
            "subject_kind": "record_cid", "subject_ref": full["scope_cid"],
            "limit": 8, "returned_records": nomination["n"],
            "capsule_index_cid": capsule_rows[0]["record_ref"],
            "completion_authority": False, "freshness_verified": False,
            "ranking_quality_verified": False, "dependency_closure_verified": False,
        }
    assert {arm: path.read_bytes() for arm, path in artifacts.items()} == artifact_before
    assert {arm: _block_hashes(output) for arm, output in outputs.items()} == block_before
    assert {name: (repository / name).read_bytes() for name in SOURCES} == source_before
    report["offline_probe"] = {
        "source_inventory": {
            name: {"sha256": sha256(raw).hexdigest(), "bytes": len(raw)}
            for name, raw in source_before.items()
        },
        "worker_artifacts": {
            arm: {"sha256": sha256(raw).hexdigest(), "bytes": len(raw)}
            for arm, raw in artifact_before.items()
        },
        "metadata_nominations": nominations,
        "target_declared_dependency_count": len(target_dependencies),
        "directly_omitted_known_dependency_count": len(directly_omitted),
        "declared_edges_inferred_from_source_text": False,
        "same_required_complete_raw_inventory": True,
        "same_literal_task_acceptance_and_suffix": True,
        "exact_v2_restoration_of_each_own_native_prompt": True,
        "original_sources_artifacts_and_producer_blocks_unchanged": True,
        "public_acceptance_executed": False,
        "provider_calls": 0, "downloads": 0, "real_tokenizer": None,
    }
    # Reuse this producer fixture for a current-source refusal, then restore its
    # exact original bytes. A retained artifact digest cannot bless dirty source.
    changed = repository / "operations.py"
    changed.write_bytes(source_before["operations.py"] + b"\n# changed source\n")
    try:
        with pytest.raises(ValueError, match="stale"):
            compare_semantic_selection(
                repository=repository,
                full_prompt=prompts["full"], selected_prompt=prompts["selected"],
                workspace=repository, purpose="coding", provider="codex_cli", model="fixture-model",
                token_counter=None, measurement_basis="authored_fixture",
            )
    finally:
        changed.write_bytes(source_before["operations.py"])
    assert {name: (repository / name).read_bytes() for name in SOURCES} == source_before
    assert {arm: path.read_bytes() for arm, path in artifacts.items()} == artifact_before
    assert {arm: _block_hashes(output) for arm, output in outputs.items()} == block_before
    report["offline_probe"]["current_source_change_refused"] = True
    record_property("semantic_selection_comparison_probe", json.dumps(report, sort_keys=True))
